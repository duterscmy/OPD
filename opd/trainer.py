from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
from transformers import StoppingCriteria, StoppingCriteriaList, Trainer
from trl.experimental.gkd import GKDTrainer
from trl.models.utils import unwrap_model_for_generation

from .alignment import build_text_span_alignment
from .behavior_markers import (
    RolloutBehaviorAnalyzer,
    aggregate_occurrence_logs,
    compact_behavior_summary,
)
from .behavior_probabilities import (
    flatten_probability_logs,
    summarize_next_token_probabilities,
)
from .collator import apply_chat_template_ids
from .rollout_safety import (
    RolloutEOSInfo,
    TruncatedCompletion,
    finalize_math_completion,
    first_complete_boxed_line_end,
    repeated_ngram_ratio,
    resolve_rollout_eos,
    truncate_completion,
)
from .schedules import HorizonSchedule


def _tokenizers_identical(student: Any, teacher: Any) -> bool:
    try:
        return len(student) == len(teacher) and student.get_vocab() == teacher.get_vocab()
    except Exception:
        return False


def _tokenizers_prefix_compatible(student: Any, teacher: Any) -> bool:
    """True when every student token has the same ID in the teacher vocabulary."""
    try:
        if len(teacher) < len(student):
            return False
        teacher_vocab = teacher.get_vocab()
        return all(teacher_vocab.get(token) == token_id for token, token_id in student.get_vocab().items())
    except Exception:
        return False


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    middle = len(values) // 2
    if len(values) % 2:
        return float(values[middle])
    return float(values[middle - 1] + values[middle]) / 2.0


class _BoxedAnswerStoppingCriteria(StoppingCriteria):
    """Stop each rollout as soon as its completion contains a full boxed answer.

    Only completion tokens are decoded, so the empty ``\\boxed{}`` instruction in
    the prompt cannot trigger the criterion.  ``stop_lengths`` records the exact
    generated length before padding is added for other rows in the batch.
    """

    def __init__(self, tokenizer: Any, prompt_width: int) -> None:
        self.tokenizer = tokenizer
        self.prompt_width = int(prompt_width)
        self.stop_lengths: dict[int, int] = {}

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor | None,
        **kwargs: Any,
    ) -> torch.BoolTensor:
        del scores, kwargs
        done = torch.zeros(
            input_ids.shape[0],
            dtype=torch.bool,
            device=input_ids.device,
        )
        generated_width = max(int(input_ids.shape[1]) - self.prompt_width, 0)
        if generated_width <= 0:
            return done

        for batch_index in range(int(input_ids.shape[0])):
            if batch_index in self.stop_lengths:
                done[batch_index] = True
                continue
            completion_ids = (
                input_ids[batch_index, self.prompt_width :]
                .detach()
                .cpu()
                .tolist()
            )
            text = self.tokenizer.decode(
                completion_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if first_complete_boxed_line_end(text) is not None:
                self.stop_lengths[batch_index] = len(completion_ids)
                done[batch_index] = True
        return done


class AdaptiveOPDTrainer(GKDTrainer):
    """On-policy rollout trainer supporting Top-K OPD and sampled RKL.

    Full-vocabulary TRL/GJSD, curriculum, and reflection routes are deliberately
    absent. Loss routing is strict so a YAML cannot silently run another loss.
    """

    def __init__(
        self,
        *args: Any,
        teacher_tokenizer: Any,
        experiment_config: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.teacher_tokenizer = teacher_tokenizer
        self.experiment_config = experiment_config
        self.strategy = str(experiment_config["strategy"])
        self.loss_backend = str(experiment_config["loss_backend"])
        self.opd_loss_mode = str(experiment_config["opd_loss_mode"])
        self.teacher_use_chat_template = bool(experiment_config.get("teacher_use_chat_template", True))
        self.teacher_enable_thinking = bool(experiment_config.get("teacher_enable_thinking", False))
        self.minimum_aligned_chars = int(experiment_config.get("minimum_aligned_chars", 1))
        self.rkl_advantage_clip = experiment_config.get("rkl_advantage_clip")
        self.same_tokenizer = _tokenizers_identical(self.processing_class, teacher_tokenizer)
        self.prefix_compatible_tokenizer = _tokenizers_prefix_compatible(
            self.processing_class, teacher_tokenizer
        )
        self.common_vocab_size = min(len(self.processing_class), len(teacher_tokenizer))
        extra_eos_tokens = experiment_config.get("rollout_extra_eos_tokens", [])
        if isinstance(extra_eos_tokens, str):
            extra_eos_tokens = [extra_eos_tokens]
        include_teacher_eos = bool(
            experiment_config.get("rollout_include_teacher_eos", True)
        )
        teacher_generation_eos = None
        if include_teacher_eos:
            teacher_generation_eos = getattr(
                getattr(self.teacher_model, "generation_config", None),
                "eos_token_id",
                None,
            )
        self.rollout_eos_info: RolloutEOSInfo = resolve_rollout_eos(
            self.processing_class,
            teacher_tokenizer,
            student_generation_eos=getattr(self.generation_config, "eos_token_id", None),
            teacher_generation_eos=teacher_generation_eos,
            extra_eos_tokens=extra_eos_tokens,
            include_teacher_eos=include_teacher_eos,
        )
        self.rollout_eos_token_ids = list(self.rollout_eos_info.token_ids)
        self.truncated_rollout_weight = float(
            experiment_config.get("truncated_rollout_weight", 0.0)
        )
        self.repetition_ngram_size = int(
            experiment_config.get("rollout_repetition_ngram_size", 4)
        )
        self.truncate_after_boxed_answer = bool(
            experiment_config.get("rollout_truncate_after_boxed_answer", False)
        )
        self.append_eos_after_boxed_answer = bool(
            experiment_config.get("rollout_append_eos_after_boxed_answer", False)
        )
        self.boxed_terminal_eos_token_id = (
            self.rollout_eos_info.preferred_teacher_eos_id()
        )
        self.schedule = HorizonSchedule(
            strategy=self.strategy,
            prefix_length=int(experiment_config["prefix_length"]),
            full_max_new_tokens=int(experiment_config["full_max_new_tokens"]),
        )
        self._loss_call_index = 0

        if self.loss_backend not in {"adaptive_opd", "sampled_rkl"}:
            raise ValueError(
                f"Unsupported loss_backend={self.loss_backend!r}; "
                "expected adaptive_opd or sampled_rkl."
            )
        if self.loss_backend == "sampled_rkl" and self.opd_loss_mode != "sampled_rkl":
            raise ValueError(
                "sampled_rkl requires opd_loss_mode=sampled_rkl; refusing silent rerouting."
            )
        if self.loss_backend == "adaptive_opd" and not (
            self.same_tokenizer or self.prefix_compatible_tokenizer
        ):
            raise ValueError(
                "adaptive_opd requires identical or prefix-compatible token IDs. "
                "Use sampled_rkl for genuinely different tokenizers."
            )

        debug_path = str(experiment_config.get("debug_jsonl_path", "token_debug_rank{rank}.jsonl"))
        debug_path = debug_path.format(rank=self.accelerator.process_index)
        path = Path(debug_path)
        if not path.is_absolute():
            path = Path(self.args.output_dir) / path
        self.debug_jsonl_path = path

        self.behavior_monitor_enabled = bool(
            experiment_config.get("behavior_monitor_enabled", False)
        )
        self.behavior_console_enabled = bool(
            experiment_config.get("behavior_console_enabled", False)
        )
        behavior_every = experiment_config.get(
            "behavior_monitor_every_n_loss_calls"
        )
        if behavior_every is None:
            behavior_every = experiment_config.get(
                "gradient_accumulation_steps", 1
            )
        self.behavior_monitor_every_n_loss_calls = max(int(behavior_every), 1)
        extra_behavior_markers = experiment_config.get(
            "behavior_marker_dictionary"
        )
        focus_behavior_markers = experiment_config.get("behavior_focus_markers")
        self.behavior_analyzer = RolloutBehaviorAnalyzer(
            self.processing_class,
            extra_markers=extra_behavior_markers,
            focus_markers=focus_behavior_markers,
        )
        self.teacher_behavior_analyzer = RolloutBehaviorAnalyzer(
            self.teacher_tokenizer,
            extra_markers=extra_behavior_markers,
            focus_markers=focus_behavior_markers,
        )
        self.student_behavior_token_sets = (
            self.behavior_analyzer.probability_token_sets(
                eos_token_ids=self.rollout_eos_token_ids
            )
        )
        # adaptive_opd requires prefix-compatible token IDs, so the resolved
        # rollout EOS IDs are also valid teacher-logit indices in this route.
        self.teacher_behavior_token_sets = (
            self.teacher_behavior_analyzer.probability_token_sets(
                eos_token_ids=self.rollout_eos_token_ids
            )
        )

        self.behavior_probe_enabled = self.behavior_monitor_enabled and bool(
            experiment_config.get("behavior_probe_enabled", False)
        )
        self.behavior_probe_every_n_steps = max(
            int(experiment_config.get("behavior_probe_every_n_steps", 25)), 1
        )
        self.behavior_probe_samples = max(
            int(experiment_config.get("behavior_probe_samples", 1)), 1
        )
        probe_max_length = experiment_config.get("behavior_probe_max_length")
        if probe_max_length is None:
            probe_max_length = experiment_config.get(
                "effective_max_length",
                experiment_config.get("max_length", 4096),
            )
        self.behavior_probe_max_length = int(probe_max_length)
        self.behavior_manifest_path = self._rank_output_path(
            experiment_config.get(
                "behavior_marker_manifest_path",
                "behavior_marker_manifest_rank{rank}.json",
            )
        )
        self.behavior_probe_set_path = self._rank_output_path(
            experiment_config.get(
                "behavior_probe_set_path", "behavior_probe_set_rank{rank}.json"
            )
        )
        self.behavior_probe_jsonl_path = self._rank_output_path(
            experiment_config.get(
                "behavior_probe_jsonl_path", "behavior_probe_rank{rank}.jsonl"
            )
        )
        probe_input = experiment_config.get("behavior_probe_input_path")
        self.behavior_probe_input_path = (
            Path(str(probe_input).format(rank=self.accelerator.process_index))
            if probe_input
            else self.behavior_probe_set_path
        )
        self._behavior_probe_payload: dict[str, Any] | None = None
        self._behavior_probe_teacher_summary: dict[str, Any] | None = None
        self._last_behavior_probe_step: int | None = None
        if self.behavior_probe_input_path.exists():
            self._behavior_probe_payload = json.loads(
                self.behavior_probe_input_path.read_text(encoding="utf-8")
            )
            self._validate_behavior_probe_payload(self._behavior_probe_payload)
            cached_teacher = self._behavior_probe_payload.get(
                "teacher_probability_summary"
            )
            if isinstance(cached_teacher, dict):
                self._behavior_probe_teacher_summary = cached_teacher
        elif probe_input:
            raise FileNotFoundError(
                f"behavior_probe_input_path not found: {self.behavior_probe_input_path}"
            )

        if self.behavior_monitor_enabled:
            self._write_behavior_manifest()

        self.accelerator.print(
            "[OPD route] "
            f"backend={self.loss_backend} mode={self.opd_loss_mode} "
            f"strategy={self.strategy} common_vocab={self.common_vocab_size}"
        )
        self.accelerator.print(
            "[rollout EOS] "
            + json.dumps(
                {
                    "token_ids": self.rollout_eos_token_ids,
                    "tokens": {
                        str(token_id): self.rollout_eos_info.token_strings[token_id]
                        for token_id in self.rollout_eos_token_ids
                    },
                    "sources": {
                        str(token_id): self.rollout_eos_info.sources[token_id]
                        for token_id in self.rollout_eos_token_ids
                    },
                    "pad_token_id": self.processing_class.pad_token_id,
                    "truncated_rollout_weight": self.truncated_rollout_weight,
                    "truncate_after_boxed_answer": self.truncate_after_boxed_answer,
                    "append_eos_after_boxed_answer": self.append_eos_after_boxed_answer,
                    "boxed_terminal_eos_token_id": self.boxed_terminal_eos_token_id,
                    "boxed_terminal_eos_token": self.rollout_eos_info.token_strings[
                        self.boxed_terminal_eos_token_id
                    ],
                    "loss_normalization": experiment_config.get(
                        "loss_normalization", "per_sequence"
                    ),
                },
                ensure_ascii=False,
            )
        )
        if self.behavior_monitor_enabled:
            self.accelerator.print(
                "[behavior monitor] "
                + json.dumps(
                    {
                        "occurrence_logging": True,
                        "probability_every_n_loss_calls": (
                            self.behavior_monitor_every_n_loss_calls
                        ),
                        "fixed_probe": self.behavior_probe_enabled,
                        "fixed_probe_every_n_steps": self.behavior_probe_every_n_steps,
                        "fixed_probe_samples": self.behavior_probe_samples,
                        "manifest": str(self.behavior_manifest_path),
                        "probe_set": str(self.behavior_probe_set_path),
                        "probe_jsonl": str(self.behavior_probe_jsonl_path),
                    },
                    ensure_ascii=False,
                )
            )

    def _rank_output_path(self, value: Any) -> Path:
        path = Path(str(value).format(rank=self.accelerator.process_index))
        if not path.is_absolute():
            path = Path(self.args.output_dir) / path
        return path

    def _write_behavior_manifest(self) -> None:
        payload = {
            "version": 1,
            "rank": int(self.accelerator.process_index),
            "student": self.behavior_analyzer.manifest(
                eos_token_ids=self.rollout_eos_token_ids
            ),
            "teacher": self.teacher_behavior_analyzer.manifest(
                eos_token_ids=self.rollout_eos_token_ids
            ),
            "notes": [
                "Legacy rollout-length console lines and JSONL fields are unchanged.",
                "Online probabilities use live rollout prefixes and are not fixed-context comparisons.",
                "Fixed-probe probabilities use the same saved prefixes at every probe step.",
            ],
        }
        self.behavior_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.behavior_manifest_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _validate_behavior_probe_payload(self, payload: dict[str, Any]) -> None:
        if int(payload.get("version", -1)) != 1:
            raise ValueError("Unsupported behavior probe set version")
        samples = payload.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError("Behavior probe set contains no samples")
        vocab_size = payload.get("student_tokenizer_vocab_size")
        if vocab_size is not None and int(vocab_size) != len(self.processing_class):
            raise ValueError(
                "Behavior probe tokenizer vocabulary does not match the current student"
            )

    def _save_behavior_probe_payload(self) -> None:
        if self._behavior_probe_payload is None:
            return
        self.behavior_probe_set_path.parent.mkdir(parents=True, exist_ok=True)
        self.behavior_probe_set_path.write_text(
            json.dumps(
                self._behavior_probe_payload,
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )

    def _capture_behavior_probe(self, batch: dict[str, Any]) -> None:
        samples: list[dict[str, Any]] = []
        sample_count = min(
            self.behavior_probe_samples, int(batch["input_ids"].shape[0])
        )
        for batch_index in range(sample_count):
            mask = batch["attention_mask"][batch_index].bool()
            input_ids = [
                int(value)
                for value in batch["input_ids"][batch_index][mask]
                .detach()
                .cpu()
                .tolist()
            ]
            labels = [
                int(value)
                for value in batch["labels"][batch_index][mask]
                .detach()
                .cpu()
                .tolist()
            ]
            truncation = None
            if len(input_ids) > self.behavior_probe_max_length:
                removed = len(input_ids) - self.behavior_probe_max_length
                input_ids = input_ids[-self.behavior_probe_max_length :]
                labels = labels[-self.behavior_probe_max_length :]
                truncation = {
                    "side": "left",
                    "removed_tokens": removed,
                }
            completion_ids = [
                token_id
                for token_id, label in zip(input_ids, labels, strict=True)
                if label != -100
            ]
            samples.append(
                {
                    "sample_index": batch_index,
                    "input_ids": input_ids,
                    "labels": labels,
                    "completion_ids": completion_ids,
                    "sequence_length": len(input_ids),
                    "completion_length": len(completion_ids),
                    "truncation": truncation,
                    "problem": batch.get("debug_problem", [""] * sample_count)[
                        batch_index
                    ],
                    "prompt_text": batch.get(
                        "debug_prompt_text", [""] * sample_count
                    )[batch_index],
                    "rollout_text": batch.get(
                        "debug_rollout_text", [""] * sample_count
                    )[batch_index],
                }
            )
        self._behavior_probe_payload = {
            "version": 1,
            "created_global_step": int(self.state.global_step),
            "created_rank": int(self.accelerator.process_index),
            "student_model": str(
                self.experiment_config.get("model_name_or_path", "")
            ),
            "teacher_model": str(
                self.experiment_config.get("teacher_model_name_or_path", "")
            ),
            "student_tokenizer_vocab_size": len(self.processing_class),
            "measurement": (
                "Fixed teacher-forced prefixes captured once and reused across steps."
            ),
            "samples": samples,
        }
        self._save_behavior_probe_payload()

    def _behavior_probe_tensors(
        self,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]]]:
        if self._behavior_probe_payload is None:
            raise RuntimeError("Behavior probe set has not been initialized")
        samples = self._behavior_probe_payload["samples"]
        width = max(len(sample["input_ids"]) for sample in samples)
        pad_id = int(self.processing_class.pad_token_id)
        input_rows: list[list[int]] = []
        mask_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        completion_ids: list[list[int]] = []
        for sample in samples:
            ids = [int(value) for value in sample["input_ids"]]
            labels = [int(value) for value in sample["labels"]]
            padding = width - len(ids)
            input_rows.append(ids + [pad_id] * padding)
            mask_rows.append([1] * len(ids) + [0] * padding)
            label_rows.append(labels + [-100] * padding)
            completion_ids.append(
                [int(value) for value in sample["completion_ids"]]
            )
        return (
            torch.tensor(input_rows, dtype=torch.long, device=device),
            torch.tensor(mask_rows, dtype=torch.long, device=device),
            torch.tensor(label_rows, dtype=torch.long, device=device),
            completion_ids,
        )

    @staticmethod
    def _final_position_logits(
        logits: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        rows: list[torch.Tensor] = []
        for batch_index in range(int(logits.shape[0])):
            positions = torch.where(attention_mask[batch_index].bool())[0]
            if positions.numel() == 0:
                rows.append(logits[batch_index, 0])
            else:
                rows.append(logits[batch_index, int(positions[-1].item())])
        return torch.stack(rows)

    def _summarize_probe_model(
        self,
        module: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        completion_ids: list[list[int]],
        token_sets: dict[str, tuple[int, ...]],
    ) -> dict[str, Any]:
        was_training = bool(module.training)
        module.eval()
        try:
            with torch.no_grad():
                outputs = module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                summary = summarize_next_token_probabilities(
                    outputs.logits[:, :-1, :],
                    labels[:, 1:].ne(-100),
                    token_sets,
                    targets=labels[:, 1:],
                    terminal_logits=self._final_position_logits(
                        outputs.logits, attention_mask
                    ),
                    completion_ids=completion_ids,
                    repetition_ngram_size=self.repetition_ngram_size,
                )
                del outputs
        finally:
            module.train(was_training)
        return summary

    def _append_behavior_probe_record(self, record: dict[str, Any]) -> None:
        self.behavior_probe_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.behavior_probe_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(record, ensure_ascii=False, default=str) + "\n"
            )

    def _maybe_run_fixed_behavior_probe(
        self,
        model: torch.nn.Module,
        batch: dict[str, Any],
    ) -> None:
        if not self.behavior_probe_enabled:
            return
        step = int(self.state.global_step)
        if step % self.behavior_probe_every_n_steps != 0:
            return
        if self._last_behavior_probe_step == step:
            return
        self._last_behavior_probe_step = step
        if self._behavior_probe_payload is None:
            self._capture_behavior_probe(batch)

        input_ids, attention_mask, labels, completion_ids = (
            self._behavior_probe_tensors(batch["input_ids"].device)
        )
        student_summary = self._summarize_probe_model(
            model,
            input_ids,
            attention_mask,
            labels,
            completion_ids,
            self.student_behavior_token_sets,
        )

        teacher_supported = self.same_tokenizer or self.prefix_compatible_tokenizer
        if self._behavior_probe_teacher_summary is None and teacher_supported:
            self._behavior_probe_teacher_summary = self._summarize_probe_model(
                self.teacher_model,
                input_ids,
                attention_mask,
                labels,
                completion_ids,
                self.teacher_behavior_token_sets,
            )
            assert self._behavior_probe_payload is not None
            self._behavior_probe_payload["teacher_probability_summary"] = (
                self._behavior_probe_teacher_summary
            )
            self._save_behavior_probe_payload()

        record = {
            "event": "behavior_fixed_probe",
            "global_step": step,
            "loss_call_index": int(self._loss_call_index),
            "rank": int(self.accelerator.process_index),
            "probe_created_global_step": int(
                self._behavior_probe_payload.get("created_global_step", -1)
            ),
            "sample_count": len(self._behavior_probe_payload["samples"]),
            "sequence_lengths": [
                int(sample["sequence_length"])
                for sample in self._behavior_probe_payload["samples"]
            ],
            "student": student_summary,
            "teacher": self._behavior_probe_teacher_summary,
            "teacher_summary_is_static": bool(
                self._behavior_probe_teacher_summary is not None
            ),
        }
        self._append_behavior_probe_record(record)
        probe_logs = flatten_probability_logs(
            student_summary, prefix="behavior_probe/student"
        )
        if self._behavior_probe_teacher_summary is not None:
            probe_logs.update(
                flatten_probability_logs(
                    self._behavior_probe_teacher_summary,
                    prefix="behavior_probe/teacher",
                )
            )
        self.log(probe_logs)

        if self.accelerator.is_local_main_process:
            sets = student_summary.get("sets", {})
            eos = sets.get("marker/eos", {})
            planning = sets.get("category/planning", {})
            correction = sets.get("category/self_correction", {})
            repetition = student_summary.get("repetition_continuation") or {}
            print(
                "[behavior fixed probe] "
                f"step={step} "
                f"student_eos_mean={float(eos.get('mean', 0.0)):.8f} "
                f"student_eos_terminal={float(eos.get('terminal_mean', 0.0)):.8f} "
                f"student_planning={float(planning.get('mean', 0.0)):.8f} "
                f"student_self_correction={float(correction.get('mean', 0.0)):.8f} "
                f"student_repeat_continuation={float(repetition.get('mean_probability_at_eligible_positions', 0.0)):.8f}",
                flush=True,
            )

    def _current_horizon(self) -> int:
        return self.schedule.horizon(int(self.state.global_step))

    def _decode(self, ids: list[int] | torch.Tensor, max_chars: int | None = None) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        text = self.processing_class.decode(
            [int(x) for x in ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return text if max_chars is None else text[:max_chars]

    def _effective_rollout_horizon(
        self,
        model: Any,
        inputs: dict[str, Any],
        requested_horizon: int,
    ) -> tuple[int, int]:
        prompt_width = int(inputs["prompts"].shape[1])
        configured_limit = int(
            self.experiment_config.get(
                "effective_max_length",
                self.experiment_config.get("max_length", 4096),
            )
        )
        model_config = getattr(model, "config", None)
        model_limit = getattr(model_config, "max_position_embeddings", None)
        total_limit = configured_limit
        if model_limit is not None and int(model_limit) > 0:
            total_limit = min(total_limit, int(model_limit))
        available = total_limit - prompt_width
        if available <= 0:
            raise ValueError(
                f"Prompt tensor width {prompt_width} leaves no rollout space under "
                f"the effective context limit {total_limit}."
            )
        return min(int(requested_horizon), available), total_limit

    def _strip_completion(
        self,
        row: torch.Tensor,
        horizon: int,
    ) -> TruncatedCompletion:
        return truncate_completion(
            row.tolist(),
            eos_info=self.rollout_eos_info,
            pad_token_id=self.processing_class.pad_token_id,
            horizon=horizon,
        )

    @torch.no_grad()
    def _generate_student_rollouts(
        self,
        model: Any,
        inputs: dict[str, Any],
        horizon: int,
    ) -> tuple[list[list[int]], list[TruncatedCompletion], int, int]:
        effective_horizon, total_limit = self._effective_rollout_horizon(
            model, inputs, horizon
        )
        generation_config = copy.deepcopy(self.generation_config)
        generation_config.max_new_tokens = int(effective_horizon)
        generation_config.temperature = float(self.experiment_config["temperature"])
        generation_config.do_sample = bool(self.experiment_config.get("rollout_do_sample", True))
        generation_config.top_k = int(self.experiment_config.get("top_k", 0))
        generation_config.top_p = float(self.experiment_config.get("top_p", 1.0))
        generation_config.eos_token_id = list(self.rollout_eos_token_ids)
        generation_config.pad_token_id = int(self.processing_class.pad_token_id)

        generation_kwargs = dict(self.generation_kwargs)
        generation_kwargs.update(
            max_new_tokens=int(effective_horizon),
            temperature=float(self.experiment_config["temperature"]),
            do_sample=bool(self.experiment_config.get("rollout_do_sample", True)),
            top_k=int(self.experiment_config.get("top_k", 0)),
            top_p=float(self.experiment_config.get("top_p", 1.0)),
            eos_token_id=list(self.rollout_eos_token_ids),
            pad_token_id=int(self.processing_class.pad_token_id),
        )
        prompt_width = int(inputs["prompts"].shape[1])
        boxed_stopping: _BoxedAnswerStoppingCriteria | None = None
        stopping_criteria: StoppingCriteriaList | None = None
        if self.truncate_after_boxed_answer:
            boxed_stopping = _BoxedAnswerStoppingCriteria(
                self.processing_class,
                prompt_width=prompt_width,
            )
            stopping_criteria = StoppingCriteriaList([boxed_stopping])

        generate_kwargs: dict[str, Any] = {
            "input_ids": inputs["prompts"],
            "attention_mask": inputs.get("prompt_attention_mask"),
            "generation_config": generation_config,
            "return_dict_in_generate": True,
        }
        if stopping_criteria is not None:
            generate_kwargs["stopping_criteria"] = stopping_criteria
        with unwrap_model_for_generation(
            model,
            self.accelerator,
            generation_kwargs=generation_kwargs,
        ) as unwrapped_model:
            outputs = unwrapped_model.generate(**generate_kwargs)

        completions: list[list[int]] = []
        metadata: list[TruncatedCompletion] = []
        for batch_index, row in enumerate(outputs.sequences[:, prompt_width:]):
            generation_stopped_after_boxed_answer = bool(
                boxed_stopping is not None
                and batch_index in boxed_stopping.stop_lengths
            )
            if generation_stopped_after_boxed_answer:
                # A row that finishes before other rows may be padded by
                # ``generate``. Slice at the exact boxed boundary so padding
                # (which equals EOS for Qwen2.5) is not mistaken for real EOS.
                row = row[: boxed_stopping.stop_lengths[batch_index]]
            item = self._strip_completion(row, effective_horizon)
            item = finalize_math_completion(
                item,
                self.processing_class,
                repetition_ngram_size=self.repetition_ngram_size,
                truncate_after_boxed_answer=self.truncate_after_boxed_answer,
                append_eos_after_boxed_answer=self.append_eos_after_boxed_answer,
                terminal_eos_token_id=self.boxed_terminal_eos_token_id,
                generation_stopped_after_boxed_answer=(
                    generation_stopped_after_boxed_answer
                ),
                rollout_horizon=effective_horizon,
            )
            completions.append(item.token_ids)
            metadata.append(item)
        return completions, metadata, effective_horizon, total_limit

    def _build_student_batch(
        self,
        source: dict[str, Any],
        completions: list[list[int]],
        metadata: list[TruncatedCompletion],
        horizon: int,
        requested_horizon: int,
        total_limit: int,
    ) -> dict[str, Any]:
        pad_id = int(self.processing_class.pad_token_id)
        rows: list[list[int]] = []
        labels: list[list[int]] = []
        prompt_rows: list[list[int]] = []
        for index, completion in enumerate(completions):
            prompt_mask = source["prompt_attention_mask"][index].bool()
            prompt = [int(x) for x in source["prompts"][index][prompt_mask].tolist()]
            row = prompt + [int(x) for x in completion]
            rows.append(row)
            labels.append([-100] * len(prompt) + [int(x) for x in completion])
            prompt_rows.append(prompt)

        width = max(len(row) for row in rows)
        padded_ids: list[list[int]] = []
        padded_masks: list[list[int]] = []
        padded_labels: list[list[int]] = []
        for row, label in zip(rows, labels, strict=True):
            padding = width - len(row)
            padded_ids.append(row + [pad_id] * padding)
            padded_masks.append([1] * len(row) + [0] * padding)
            padded_labels.append(label + [-100] * padding)

        device = source["prompts"].device
        rollout_texts = [self._decode(ids) for ids in completions]
        behavior_records = (
            [
                self.behavior_analyzer.analyze(
                    ids,
                    text,
                    eos_token_ids=self.rollout_eos_token_ids,
                    repetition_ngram_size=self.repetition_ngram_size,
                )
                for ids, text in zip(completions, rollout_texts, strict=True)
            ]
            if self.behavior_monitor_enabled
            else [{} for _ in completions]
        )
        sequence_loss_weights = [
            self.truncated_rollout_weight if item.hit_horizon else 1.0
            for item in metadata
        ]
        batch: dict[str, Any] = {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(padded_masks, dtype=torch.long, device=device),
            "labels": torch.tensor(padded_labels, dtype=torch.long, device=device),
            "student_prompt_lengths": [len(row) for row in prompt_rows],
            "student_completion_ids": completions,
            "prompt_messages": source["prompt_messages"],
            "debug_problem": source.get("problem", [""] * len(completions)),
            "debug_prompt_text": source.get("prompt_texts", [""] * len(completions)),
            "sequence_loss_weights": torch.tensor(
                sequence_loss_weights, dtype=torch.float32, device=device
            ),
            "debug_rollout_text": rollout_texts,
            "debug_prompt_length": [len(row) for row in prompt_rows],
            "debug_rollout_length": [len(ids) for ids in completions],
            "debug_raw_rollout_length": [item.raw_tensor_length for item in metadata],
            "debug_eos": [item.emitted_eos for item in metadata],
            "debug_stop_reason": [item.stop_reason for item in metadata],
            "debug_stop_token_id": [item.stop_token_id for item in metadata],
            "debug_stop_token": [
                self.rollout_eos_info.token_strings.get(item.stop_token_id, "")
                if item.stop_token_id is not None
                else ""
                for item in metadata
            ],
            "debug_hit_horizon": [item.hit_horizon for item in metadata],
            "debug_raw_hit_horizon": [item.raw_hit_horizon for item in metadata],
            "debug_boxed_truncated": [item.boxed_truncated for item in metadata],
            "debug_appended_eos": [item.appended_eos for item in metadata],
            "debug_repeated_ngram_ratio": [
                item.raw_repeated_ngram_ratio for item in metadata
            ],
            "debug_effective_repeated_ngram_ratio": [
                repeated_ngram_ratio(ids, self.repetition_ngram_size)
                for ids in completions
            ],
            # Keep this count on the raw (pre-semantic-truncation) completion so
            # repeated-answer collapse remains visible even after safe masking.
            "debug_boxed_count": [item.raw_boxed_count for item in metadata],
            "debug_effective_boxed_count": [
                text.count("\\boxed{") for text in rollout_texts
            ],
            "debug_behavior": behavior_records,
            "debug_sequence_loss_weight": sequence_loss_weights,
            "debug_horizon": int(horizon),
            "debug_requested_horizon": int(requested_horizon),
            "debug_total_length_limit": int(total_limit),
        }
        return batch

    def _build_cross_tokenizer_batch(self, batch: dict[str, Any]) -> None:
        teacher_rows: list[list[int]] = []
        alignment_groups: list[list[dict[str, Any]]] = []
        for messages, student_ids, student_prompt_len in zip(
            batch["prompt_messages"],
            batch["student_completion_ids"],
            batch["student_prompt_lengths"],
            strict=True,
        ):
            teacher_prompt = apply_chat_template_ids(
                self.teacher_tokenizer,
                messages,
                add_generation_prompt=True,
                use_chat_template=self.teacher_use_chat_template,
                enable_thinking=self.teacher_enable_thinking,
            )
            _, teacher_completion, groups = build_text_span_alignment(
                self.processing_class,
                self.teacher_tokenizer,
                student_ids,
                minimum_aligned_chars=self.minimum_aligned_chars,
            )
            teacher_rows.append(teacher_prompt + teacher_completion)
            alignment_groups.append(
                [
                    {
                        "student": [student_prompt_len + i for i in group.student_indices],
                        "teacher": [len(teacher_prompt) + i for i in group.teacher_indices],
                        "chars": group.end_char - group.start_char,
                    }
                    for group in groups
                ]
            )

        teacher_pad = int(self.teacher_tokenizer.pad_token_id)
        width = max(len(row) for row in teacher_rows)
        padded: list[list[int]] = []
        masks: list[list[int]] = []
        for row in teacher_rows:
            padding = width - len(row)
            padded.append(row + [teacher_pad] * padding)
            masks.append([1] * len(row) + [0] * padding)
        device = batch["input_ids"].device
        batch["teacher_input_ids"] = torch.tensor(padded, dtype=torch.long, device=device)
        batch["teacher_attention_mask"] = torch.tensor(masks, dtype=torch.long, device=device)
        batch["alignment_groups"] = alignment_groups

    def _write_jsonl(self, record: dict[str, Any]) -> None:
        if not bool(self.experiment_config.get("debug_jsonl_enabled", True)):
            return
        self.debug_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.debug_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _print_rollouts(self, batch: dict[str, Any]) -> None:
        if not bool(self.experiment_config.get("debug_print_rollouts", True)):
            return
        if not self.accelerator.is_local_main_process:
            return
        count = min(
            int(self.experiment_config.get("debug_print_samples", 1)),
            len(batch["debug_rollout_text"]),
        )
        max_chars = int(self.experiment_config.get("debug_print_max_chars", 2500))
        for index in range(count):
            # Keep the legacy parseable line unchanged: it now means the
            # effective completion length after EOS/horizon truncation.
            print(f"【student rollout length】{batch['debug_rollout_length'][index]}")
            print(f"【student raw rollout length】{batch['debug_raw_rollout_length'][index]}")
            print(f"【student rollout stop reason】{batch['debug_stop_reason'][index]}")
            print(f"【student rollout stop token】{batch['debug_stop_token'][index]}")
            print(f"【student rollout hit horizon】{batch['debug_hit_horizon'][index]}")
            print(
                f"【student raw rollout hit horizon】"
                f"{batch['debug_raw_hit_horizon'][index]}"
            )
            print(
                f"【student rollout truncated after boxed answer】"
                f"{batch['debug_boxed_truncated'][index]}"
            )
            print(
                f"【student rollout appended EOS】"
                f"{batch['debug_appended_eos'][index]}"
            )
            print(
                f"【student raw rollout repeated {self.repetition_ngram_size}-gram ratio】"
                f"{batch['debug_repeated_ngram_ratio'][index]:.6f}"
            )
            print(
                f"【student effective rollout repeated {self.repetition_ngram_size}-gram ratio】"
                f"{batch['debug_effective_repeated_ngram_ratio'][index]:.6f}"
            )
            print(f"【student rollout boxed count】{batch['debug_boxed_count'][index]}")
            print(
                f"【student effective rollout boxed count】"
                f"{batch['debug_effective_boxed_count'][index]}"
            )
            print("\n" + "=" * 100)
            print(
                f"[student rollout] step={self.state.global_step} sample={index} "
                f"strategy={self.strategy} requested_horizon={batch['debug_requested_horizon']} "
                f"horizon={batch['debug_horizon']} "
                f"prompt_length={batch['debug_prompt_length'][index]} "
                f"length={batch['debug_rollout_length'][index]} "
                f"eos={batch['debug_eos'][index]} "
                f"stop_reason={batch['debug_stop_reason'][index]}"
            )
            print(batch["debug_rollout_text"][index][:max_chars])
            print("=" * 100, flush=True)
            if self.behavior_console_enabled and batch.get("debug_behavior"):
                print(
                    "[student behavior] "
                    f"step={self.state.global_step} sample={index} "
                    + json.dumps(
                        compact_behavior_summary(
                            batch["debug_behavior"][index]
                        ),
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        del num_items_in_batch
        requested_horizon = self._current_horizon()
        completions, metadata, horizon, total_limit = self._generate_student_rollouts(
            model, inputs, requested_horizon
        )
        batch = self._build_student_batch(
            inputs,
            completions,
            metadata,
            horizon,
            requested_horizon,
            total_limit,
        )
        if self.loss_backend == "sampled_rkl":
            self._build_cross_tokenizer_batch(batch)

        lengths = [len(ids) for ids in completions]
        raw_lengths = [item.raw_tensor_length for item in metadata]
        prompt_lengths = batch["debug_prompt_length"]
        eos_flags = [item.emitted_eos for item in metadata]
        hit_horizon = [item.hit_horizon for item in metadata]
        raw_hit_horizon = [item.raw_hit_horizon for item in metadata]
        boxed_truncated = [item.boxed_truncated for item in metadata]
        appended_eos = [item.appended_eos for item in metadata]
        stop_reasons = [item.stop_reason for item in metadata]
        repetition = batch["debug_repeated_ngram_ratio"]
        effective_repetition = batch["debug_effective_repeated_ngram_ratio"]
        boxed_counts = batch["debug_boxed_count"]
        rollout_logs: dict[str, float] = {
                "rollout/horizon": float(horizon),
                "rollout/requested_horizon": float(requested_horizon),
                "rollout/total_length_limit": float(total_limit),
                "rollout/mean_generated_tokens": float(sum(lengths) / max(len(lengths), 1)),
                "rollout/raw_mean_generated_tokens": float(
                    sum(raw_lengths) / max(len(raw_lengths), 1)
                ),
                "rollout/mean_removed_tokens": float(
                    sum(
                        max(raw - effective, 0)
                        for raw, effective in zip(raw_lengths, lengths, strict=True)
                    )
                    / max(len(lengths), 1)
                ),
                "rollout/mean_prompt_tokens": float(
                    sum(prompt_lengths) / max(len(prompt_lengths), 1)
                ),
                "rollout/mean_total_effective_tokens": float(
                    sum(prompt + completion for prompt, completion in zip(
                        prompt_lengths, lengths, strict=True
                    ))
                    / max(len(lengths), 1)
                ),
                "rollout/context_limited_fraction": float(horizon < requested_horizon),
                "rollout/median_generated_tokens": _median(lengths),
                "rollout/max_generated_tokens": float(max(lengths) if lengths else 0),
                "rollout/min_generated_tokens": float(min(lengths) if lengths else 0),
                "rollout/eos_fraction": float(sum(eos_flags) / max(len(eos_flags), 1)),
                "rollout/truncated_fraction": float(
                    sum(hit_horizon) / max(len(hit_horizon), 1)
                ),
                "rollout/raw_truncated_fraction": float(
                    sum(raw_hit_horizon) / max(len(raw_hit_horizon), 1)
                ),
                "rollout/boxed_answer_stop_fraction": float(
                    sum(reason == "boxed_answer" for reason in stop_reasons)
                    / max(len(stop_reasons), 1)
                ),
                "rollout/boxed_truncation_fraction": float(
                    sum(boxed_truncated) / max(len(boxed_truncated), 1)
                ),
                "rollout/appended_eos_fraction": float(
                    sum(appended_eos) / max(len(appended_eos), 1)
                ),
                "rollout/student_eos_fraction": float(
                    sum(reason == "student_eos" for reason in stop_reasons)
                    / max(len(stop_reasons), 1)
                ),
                "rollout/teacher_eos_fraction": float(
                    sum(reason == "teacher_eos" for reason in stop_reasons)
                    / max(len(stop_reasons), 1)
                ),
                "rollout/configured_eos_fraction": float(
                    sum(reason == "configured_eos" for reason in stop_reasons)
                    / max(len(stop_reasons), 1)
                ),
                "rollout/no_eos_fraction": float(
                    sum(not flag for flag in eos_flags) / max(len(eos_flags), 1)
                ),
                f"rollout/repeated_{self.repetition_ngram_size}gram_ratio": float(
                    sum(repetition) / max(len(repetition), 1)
                ),
                f"rollout/effective_repeated_{self.repetition_ngram_size}gram_ratio": float(
                    sum(effective_repetition) / max(len(effective_repetition), 1)
                ),
                "rollout/multi_boxed_fraction": float(
                    sum(count > 1 for count in boxed_counts) / max(len(boxed_counts), 1)
                ),
                "rollout/mean_sequence_loss_weight": float(
                    sum(batch["debug_sequence_loss_weight"])
                    / max(len(batch["debug_sequence_loss_weight"]), 1)
                ),
            }
        if self.behavior_monitor_enabled:
            rollout_logs.update(
                aggregate_occurrence_logs(batch["debug_behavior"])
            )
        self.log(rollout_logs)
        self._print_rollouts(batch)
        self._maybe_run_fixed_behavior_probe(model, batch)
        # Bypass GKDTrainer.training_step: rollout generation is handled above.
        return Trainer.training_step(self, model, batch, None)

    @staticmethod
    def _target_log_probs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shifted_logits = logits[:, :-1, :]
        targets = input_ids[:, 1:]
        selected = shifted_logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1).float()
        return selected - torch.logsumexp(shifted_logits.float(), dim=-1)

    def _sampled_rkl_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
        )
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=inputs["teacher_input_ids"],
                attention_mask=inputs["teacher_attention_mask"],
                use_cache=False,
            )
        student_lp = self._target_log_probs(student_outputs.logits, inputs["input_ids"])
        teacher_lp = self._target_log_probs(teacher_outputs.logits, inputs["teacher_input_ids"])
        sequence_terms: list[torch.Tensor] = []
        sequence_weights: list[torch.Tensor] = []
        advantages: list[float] = []
        for batch_index, groups in enumerate(inputs["alignment_groups"]):
            terms: list[torch.Tensor] = []
            token_count = 0
            for group in groups:
                student_pos = [i - 1 for i in group["student"] if i > 0]
                teacher_pos = [i - 1 for i in group["teacher"] if i > 0]
                if not student_pos or not teacher_pos:
                    continue
                student_logp = student_lp[batch_index, student_pos].sum()
                teacher_logp = teacher_lp[batch_index, teacher_pos].sum()
                advantage = student_logp.detach() - teacher_logp.detach()
                if self.rkl_advantage_clip is not None:
                    advantage = advantage.clamp(
                        -float(self.rkl_advantage_clip), float(self.rkl_advantage_clip)
                    )
                terms.append(advantage * student_logp)
                token_count += len(student_pos)
                advantages.append(float(advantage.cpu().item()))
            if terms and token_count > 0:
                sequence_terms.append(torch.stack(terms).sum() / float(token_count))
                sequence_weights.append(inputs["sequence_loss_weights"][batch_index])
        if sequence_terms:
            stacked_terms = torch.stack(sequence_terms)
            stacked_weights = torch.stack(sequence_weights).to(stacked_terms)
            loss = (stacked_terms * stacked_weights).sum() / stacked_weights.sum().clamp_min(1.0)
        else:
            loss = student_outputs.logits.sum() * 0.0
        self.log(
            {
                "sampled_rkl/loss": float(loss.detach().cpu().item()),
                "sampled_rkl/aligned_groups": float(len(advantages)),
                "sampled_rkl/mean_advantage": float(sum(advantages) / max(len(advantages), 1)),
                "sampled_rkl/effective_sequence_fraction": float(
                    sum(float(weight.detach().cpu().item()) > 0.0 for weight in sequence_weights)
                    / max(len(sequence_weights), 1)
                ),
            }
        )
        return (loss, student_outputs) if return_outputs else loss

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> Any:
        del num_items_in_batch
        self._loss_call_index += 1
        if self.loss_backend == "sampled_rkl":
            return self._sampled_rkl_loss(model, inputs, return_outputs)
        raise RuntimeError(
            "adaptive_opd must be handled by AdaptiveKLTrainer; check the imported trainer class."
        )
