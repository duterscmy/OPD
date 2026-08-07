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
        self.log(
            {
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
        )
        self._print_rollouts(batch)
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
