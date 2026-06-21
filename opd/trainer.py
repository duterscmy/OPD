from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import Trainer
from trl.experimental.gkd import GKDTrainer
from trl.models.utils import unwrap_model_for_generation

from .alignment import build_text_span_alignment
from .collator import apply_chat_template_ids
from .reflection import build_reflection_prompt, judge_batch, split_token_chunks
from .schedules import HorizonSchedule


def _tokenizers_identical(a: Any, b: Any) -> bool:
    if len(a) != len(b):
        return False
    try:
        return a.get_vocab() == b.get_vocab()
    except Exception:
        return False


class AdaptiveOPDTrainer(GKDTrainer):
    """TRL GKDTrainer with rollout-horizon control and teacher-reflection truncation.

    Backends:
    - Same tokenizer: TRL's full-vocabulary generalized JSD (`beta=1` gives reverse-KL limit).
    - Different tokenizers: sampled reverse-KL policy-gradient estimator over greedily aligned text spans.
    """

    def __init__(
        self,
        *args,
        teacher_tokenizer: Any,
        experiment_config: dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.teacher_tokenizer = teacher_tokenizer
        self.experiment_config = experiment_config
        self.strategy = experiment_config["strategy"]
        self.schedule = HorizonSchedule(
            strategy=self.strategy,
            prefix_length=int(experiment_config["prefix_length"]),
            full_max_new_tokens=int(experiment_config["full_max_new_tokens"]),
            curriculum_lengths=[int(x) for x in experiment_config["curriculum_lengths"]],
            curriculum_boundaries=[int(x) for x in experiment_config["curriculum_boundaries"]],
            reflection_rollout_max_tokens=int(experiment_config["reflection_rollout_max_tokens"]),
        )
        self.same_tokenizer = _tokenizers_identical(self.processing_class, teacher_tokenizer)
        requested = experiment_config.get("loss_backend", "auto")
        if requested == "auto":
            self.loss_backend = "trl_gjsd" if self.same_tokenizer else "sampled_rkl"
        else:
            self.loss_backend = requested
        if self.loss_backend == "trl_gjsd" and not self.same_tokenizer:
            raise ValueError(
                "TRL full-vocabulary GKD requires identical student/teacher tokenizers. "
                "Use loss_backend=sampled_rkl for cross-tokenizer pairs."
            )
        if self.loss_backend == "sampled_rkl" and float(experiment_config["beta"]) != 1.0:
            raise ValueError("The cross-tokenizer backend implements sampled reverse KL and requires beta=1.0")
        self.minimum_aligned_chars = int(experiment_config.get("minimum_aligned_chars", 1))
        self.rkl_advantage_clip = experiment_config.get("rkl_advantage_clip")
        self.reflection_log_path = experiment_config.get("reflection_log_path")
        if not self.reflection_log_path:
            self.reflection_log_path = str(
                Path(self.args.output_dir) / f"reflection_rank{self.accelerator.process_index}.jsonl"
            )

    def _current_horizon(self) -> int:
        return self.schedule.horizon(int(self.state.global_step))

    def _strip_completion(self, row: torch.Tensor) -> list[int]:
        ids = []
        eos = self.processing_class.eos_token_id
        pad = self.processing_class.pad_token_id
        for token in row.tolist():
            if pad is not None and token == pad:
                break
            ids.append(int(token))
            if eos is not None and token == eos:
                break
        return ids

    @torch.no_grad()
    def _generate_student_rollouts(self, model: Any, inputs: dict[str, Any], horizon: int) -> list[list[int]]:
        generation_config = copy.deepcopy(self.generation_config)
        generation_config.max_new_tokens = int(horizon)
        generation_config.temperature = float(self.experiment_config["temperature"])
        generation_config.do_sample = True
        generation_config.top_k = 0
        generation_kwargs = dict(self.generation_kwargs)
        generation_kwargs["max_new_tokens"] = int(horizon)
        generation_kwargs["temperature"] = float(self.experiment_config["temperature"])
        generation_kwargs["do_sample"] = True
        generation_kwargs["top_k"] = 0

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
        prompt_width = inputs["prompts"].shape[1]
        return [self._strip_completion(row) for row in outputs.sequences[:, prompt_width:]]

    def _reflection_cut_lengths(
        self,
        completion_ids: list[list[int]],
        inputs: dict[str, Any],
    ) -> list[int]:
        chunk_size = int(self.experiment_config["reflection_chunk_size"])
        prompts, all_chunks = [], []
        for ids, problem, reference in zip(
            completion_ids,
            inputs["problem"],
            inputs["reference_solution"],
            strict=True,
        ):
            chunks = split_token_chunks(self.processing_class, ids, chunk_size)
            all_chunks.append(chunks)
            prompts.append(
                build_reflection_prompt(
                    problem=problem,
                    chunks=chunks,
                    reference_solution=reference,
                    use_reference=bool(self.experiment_config["reflection_use_reference"]),
                )
            )

        teacher = self.accelerator.unwrap_model(self.teacher_model)
        teacher.eval()
        decisions = judge_batch(
            teacher,
            self.teacher_tokenizer,
            prompts,
            max_new_tokens=int(self.experiment_config["reflection_max_new_tokens"]),
        )
        cut_lengths: list[int] = []
        logs = []
        fallback = self.experiment_config["reflection_parse_failure"]
        fallback_length = int(self.experiment_config["reflection_fallback_length"])

        for ids, chunks, decision, problem in zip(
            completion_ids, all_chunks, decisions, inputs["problem"], strict=True
        ):
            if decision.has_error is False:
                cut = len(ids)
            elif decision.has_error is True and decision.earliest_error_chunk_id is not None:
                chunk_id = decision.earliest_error_chunk_id
                if 0 <= chunk_id < len(chunks):
                    cut = chunks[chunk_id]["start_token"]
                else:
                    cut = min(fallback_length, len(ids))
            elif fallback == "full":
                cut = len(ids)
            elif fallback == "skip":
                cut = 0
            else:
                cut = min(fallback_length, len(ids))
            cut_lengths.append(cut)
            logs.append(
                {
                    "global_step": int(self.state.global_step),
                    "problem": problem,
                    "completion_tokens": len(ids),
                    "cut_tokens": cut,
                    "decision": {
                        "has_error": decision.has_error,
                        "earliest_error_chunk_id": decision.earliest_error_chunk_id,
                        "earliest_error_span": decision.earliest_error_span,
                        "explanation": decision.explanation,
                        "raw_output": decision.raw_output,
                    },
                }
            )

        if self.accelerator.is_local_main_process or self.accelerator.num_processes > 1:
            path = Path(self.reflection_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                for record in logs:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return cut_lengths

    def _build_student_batch(
        self,
        inputs: dict[str, Any],
        completions: list[list[int]],
        cut_lengths: list[int],
    ) -> dict[str, Any]:
        pad_id = int(self.processing_class.pad_token_id)
        rows, labels = [], []
        prompt_rows: list[list[int]] = []
        for i, (completion, cut) in enumerate(zip(completions, cut_lengths, strict=True)):
            mask = inputs["prompt_attention_mask"][i].bool()
            prompt = inputs["prompts"][i][mask].tolist()
            truncated = completion[:cut]
            row = prompt + truncated
            rows.append(row)
            labels.append([-100] * len(prompt) + truncated)
            prompt_rows.append(prompt)

        width = max(len(row) for row in rows)
        padded_ids, padded_masks, padded_labels = [], [], []
        for row, label in zip(rows, labels, strict=True):
            pad = width - len(row)
            padded_ids.append(row + [pad_id] * pad)
            padded_masks.append([1] * len(row) + [0] * pad)
            padded_labels.append(label + [-100] * pad)

        device = inputs["prompts"].device
        batch = {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(padded_masks, dtype=torch.long, device=device),
            "labels": torch.tensor(padded_labels, dtype=torch.long, device=device),
            "student_prompt_lengths": [len(row) for row in prompt_rows],
            "student_completion_ids": [c[:cut] for c, cut in zip(completions, cut_lengths, strict=True)],
            "prompt_messages": inputs["prompt_messages"],
        }
        return batch

    def _build_cross_tokenizer_batch(self, batch: dict[str, Any]) -> None:
        teacher_rows: list[list[int]] = []
        alignment_groups: list[list[dict[str, Any]]] = []

        for prompt_messages, student_ids, student_prompt_len in zip(
            batch["prompt_messages"],
            batch["student_completion_ids"],
            batch["student_prompt_lengths"],
            strict=True,
        ):
            teacher_prompt_ids = apply_chat_template_ids(
                self.teacher_tokenizer, prompt_messages, add_generation_prompt=True
            )
            text, teacher_completion_ids, groups = build_text_span_alignment(
                self.processing_class,
                self.teacher_tokenizer,
                student_ids,
                minimum_aligned_chars=self.minimum_aligned_chars,
            )
            teacher_rows.append(teacher_prompt_ids + teacher_completion_ids)
            converted = []
            for group in groups:
                converted.append(
                    {
                        # Absolute target token indices in the unpadded sequences.
                        "student": [student_prompt_len + idx for idx in group.student_indices],
                        "teacher": [len(teacher_prompt_ids) + idx for idx in group.teacher_indices],
                        "chars": group.end_char - group.start_char,
                    }
                )
            alignment_groups.append(converted)

        if self.teacher_tokenizer.pad_token_id is None:
            self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token
        teacher_pad = int(self.teacher_tokenizer.pad_token_id)
        width = max(len(row) for row in teacher_rows)
        padded, masks = [], []
        for row in teacher_rows:
            pad = width - len(row)
            padded.append(row + [teacher_pad] * pad)
            masks.append([1] * len(row) + [0] * pad)

        device = batch["input_ids"].device
        batch["teacher_input_ids"] = torch.tensor(padded, dtype=torch.long, device=device)
        batch["teacher_attention_mask"] = torch.tensor(masks, dtype=torch.long, device=device)
        batch["alignment_groups"] = alignment_groups

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        horizon = self._current_horizon()
        completions = self._generate_student_rollouts(model, inputs, horizon)
        if self.strategy == "reflection":
            cut_lengths = self._reflection_cut_lengths(completions, inputs)
        else:
            cut_lengths = [len(ids) for ids in completions]

        batch = self._build_student_batch(inputs, completions, cut_lengths)
        if self.loss_backend == "sampled_rkl":
            self._build_cross_tokenizer_batch(batch)

        if self.state.global_step % max(int(self.args.logging_steps), 1) == 0:
            valid = [x for x in cut_lengths if x > 0]
            self.log(
                {
                    "rollout/horizon": float(horizon),
                    "rollout/mean_used_tokens": float(sum(valid) / max(len(valid), 1)),
                    "rollout/skipped_fraction": float(sum(x == 0 for x in cut_lengths) / len(cut_lengths)),
                }
            )

        return Trainer.training_step(self, model, batch, None)

    @staticmethod
    def _target_log_probs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shifted_logits = logits[:, :-1, :]
        targets = input_ids[:, 1:]
        selected = shifted_logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1).float()
        log_z = torch.logsumexp(shifted_logits.float(), dim=-1)
        return selected - log_z

    def _sampled_rkl_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool):
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

        terms = []
        weights = []
        for batch_idx, groups in enumerate(inputs["alignment_groups"]):
            for group in groups:
                # Target token j is predicted by logits position j-1, hence -1 here.
                s_positions = [idx - 1 for idx in group["student"] if idx > 0]
                t_positions = [idx - 1 for idx in group["teacher"] if idx > 0]
                if not s_positions or not t_positions:
                    continue
                s_logp = student_lp[batch_idx, s_positions].sum()
                t_logp = teacher_lp[batch_idx, t_positions].sum()
                advantage = (s_logp.detach() - t_logp.detach())
                if self.rkl_advantage_clip is not None:
                    clip = float(self.rkl_advantage_clip)
                    advantage = advantage.clamp(-clip, clip)
                terms.append(advantage * s_logp)
                weights.append(max(len(s_positions), 1))

        if not terms:
            loss = student_outputs.logits.sum() * 0.0
        else:
            # Approximate per-student-token normalization across aligned text segments.
            numerator = torch.stack(terms).sum()
            denominator = float(sum(weights))
            loss = numerator / max(denominator, 1.0)

        return (loss, student_outputs) if return_outputs else loss

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        if self.loss_backend == "sampled_rkl":
            return self._sampled_rkl_loss(model, inputs, return_outputs)

        clean = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["labels"],
        }
        return super().compute_loss(
            model,
            clean,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
