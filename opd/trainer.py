from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
from transformers import Trainer
from trl.experimental.gkd import GKDTrainer
from trl.models.utils import unwrap_model_for_generation

from .alignment import build_text_span_alignment
from .collator import apply_chat_template_ids
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

    def _strip_completion(self, row: torch.Tensor) -> tuple[list[int], bool]:
        result: list[int] = []
        eos = self.processing_class.eos_token_id
        pad = self.processing_class.pad_token_id
        saw_eos = False
        for raw in row.tolist():
            token = int(raw)
            if pad is not None and token == int(pad):
                break
            result.append(token)
            if eos is not None and token == int(eos):
                saw_eos = True
                break
        return result, saw_eos

    @torch.no_grad()
    def _generate_student_rollouts(
        self,
        model: Any,
        inputs: dict[str, Any],
        horizon: int,
    ) -> tuple[list[list[int]], list[bool]]:
        generation_config = copy.deepcopy(self.generation_config)
        generation_config.max_new_tokens = int(horizon)
        generation_config.temperature = float(self.experiment_config["temperature"])
        generation_config.do_sample = bool(self.experiment_config.get("rollout_do_sample", True))
        generation_config.top_k = int(self.experiment_config.get("top_k", 0))
        generation_config.top_p = float(self.experiment_config.get("top_p", 1.0))

        generation_kwargs = dict(self.generation_kwargs)
        generation_kwargs.update(
            max_new_tokens=int(horizon),
            temperature=float(self.experiment_config["temperature"]),
            do_sample=bool(self.experiment_config.get("rollout_do_sample", True)),
            top_k=int(self.experiment_config.get("top_k", 0)),
            top_p=float(self.experiment_config.get("top_p", 1.0)),
        )
        with unwrap_model_for_generation(
            model,
            self.accelerator,
            generation_kwargs=generation_kwargs,
        ) as unwrapped_model:
            outputs = unwrapped_model.generate(
                input_ids=inputs["prompts"],
                attention_mask=inputs.get("prompt_attention_mask"),
                generation_config=generation_config,
                return_dict_in_generate=True,
            )

        prompt_width = int(inputs["prompts"].shape[1])
        completions: list[list[int]] = []
        eos_flags: list[bool] = []
        for row in outputs.sequences[:, prompt_width:]:
            ids, saw_eos = self._strip_completion(row)
            completions.append(ids)
            eos_flags.append(saw_eos)
        return completions, eos_flags

    def _build_student_batch(
        self,
        source: dict[str, Any],
        completions: list[list[int]],
        eos_flags: list[bool],
        horizon: int,
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
        batch: dict[str, Any] = {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(padded_masks, dtype=torch.long, device=device),
            "labels": torch.tensor(padded_labels, dtype=torch.long, device=device),
            "student_prompt_lengths": [len(row) for row in prompt_rows],
            "student_completion_ids": completions,
            "prompt_messages": source["prompt_messages"],
            "debug_problem": source.get("problem", [""] * len(completions)),
            "debug_prompt_text": source.get("prompt_texts", [""] * len(completions)),
            "debug_rollout_text": [self._decode(ids) for ids in completions],
            "debug_rollout_length": [len(ids) for ids in completions],
            "debug_eos": eos_flags,
            "debug_horizon": int(horizon),
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
            print("\n" + "=" * 100)
            print(
                f"[student rollout] step={self.state.global_step} sample={index} "
                f"strategy={self.strategy} horizon={batch['debug_horizon']} "
                f"length={batch['debug_rollout_length'][index]} eos={batch['debug_eos'][index]}"
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
        horizon = self._current_horizon()
        completions, eos_flags = self._generate_student_rollouts(model, inputs, horizon)
        batch = self._build_student_batch(inputs, completions, eos_flags, horizon)
        if self.loss_backend == "sampled_rkl":
            self._build_cross_tokenizer_batch(batch)

        lengths = [len(ids) for ids in completions]
        self.log(
            {
                "rollout/horizon": float(horizon),
                "rollout/mean_generated_tokens": float(sum(lengths) / max(len(lengths), 1)),
                "rollout/median_generated_tokens": _median(lengths),
                "rollout/max_generated_tokens": float(max(lengths) if lengths else 0),
                "rollout/min_generated_tokens": float(min(lengths) if lengths else 0),
                "rollout/eos_fraction": float(sum(eos_flags) / max(len(eos_flags), 1)),
                "rollout/truncated_fraction": float(
                    sum((not eos) and length >= horizon for length, eos in zip(lengths, eos_flags, strict=True))
                    / max(len(lengths), 1)
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
        terms: list[torch.Tensor] = []
        weights: list[int] = []
        advantages: list[float] = []
        for batch_index, groups in enumerate(inputs["alignment_groups"]):
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
                weights.append(len(student_pos))
                advantages.append(float(advantage.cpu().item()))
        loss = (
            torch.stack(terms).sum() / max(float(sum(weights)), 1.0)
            if terms
            else student_outputs.logits.sum() * 0.0
        )
        self.log(
            {
                "sampled_rkl/loss": float(loss.detach().cpu().item()),
                "sampled_rkl/aligned_groups": float(len(terms)),
                "sampled_rkl/mean_advantage": float(sum(advantages) / max(len(advantages), 1)),
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
