from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import Trainer
from trl.experimental.gkd import GKDTrainer
from trl.models.utils import unwrap_model_for_generation

from .alignment import build_text_span_alignment
from .answers import judge_correctness
from .collator import apply_chat_template_ids
from .reflection import build_reflection_prompt, judge_batch, split_token_chunks
from .schedules import HorizonSchedule


def _tokenizers_identical(a: Any, b: Any) -> bool:
    """Strict tokenizer equality for full-vocabulary losses.

    This is intentionally strict: full-vocab TRL GKD/JSD-style losses require
    exactly the same token-id space.
    """
    if len(a) != len(b):
        return False
    try:
        return a.get_vocab() == b.get_vocab()
    except Exception:
        return False


def _tokenizers_prefix_compatible(student_tokenizer: Any, teacher_tokenizer: Any) -> bool:
    """Allow teacher vocab to append extra tokens after the student vocab.

    This matches the Qwen2.5/Qwen3-style case you described: the common token-id
    prefix is shared, while the teacher may have a few additional special tokens
    at the end.  Adaptive top-k overlap/KL can still be computed safely if we
    restrict all distributions to the common student vocabulary.
    """
    try:
        if len(teacher_tokenizer) < len(student_tokenizer):
            return False
        student_vocab = student_tokenizer.get_vocab()
        teacher_vocab = teacher_tokenizer.get_vocab()
        for token, student_id in student_vocab.items():
            if teacher_vocab.get(token) != student_id:
                return False
        return True
    except Exception:
        # Conservative fallback: if vocab dictionaries are unavailable, only rely
        # on lengths.  The adaptive loss will still restrict logits to the common
        # prefix, but exact ID equality cannot be verified here.
        try:
            return len(teacher_tokenizer) >= len(student_tokenizer)
        except Exception:
            return False


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return float(values[mid - 1] + values[mid]) / 2.0


class AdaptiveOPDTrainer(GKDTrainer):
    """TRL GKDTrainer with OPD / ESR / curriculum / reflection / correctness-gated OPD.

    Strategies:
      - full: full student rollout, distill all generated tokens.
      - esr: generate and distill only prefix_length tokens.
      - curriculum: horizon follows curriculum_lengths / curriculum_boundaries.
      - reflection: generate long rollout, teacher localizes first error, distill only preceding prefix.
      - correctness_esr: generate long rollout, if final answer correct use full rollout, else use ESR prefix.
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
        self.strategy = str(experiment_config["strategy"])
        self.student_use_chat_template = bool(experiment_config.get("student_use_chat_template", True))
        self.teacher_use_chat_template = bool(experiment_config.get("teacher_use_chat_template", True))
        self.teacher_enable_thinking = bool(experiment_config.get("teacher_enable_thinking", False))

        self.schedule = HorizonSchedule(
            strategy=self.strategy,
            prefix_length=int(experiment_config["prefix_length"]),
            full_max_new_tokens=int(experiment_config["full_max_new_tokens"]),
            curriculum_lengths=[int(x) for x in experiment_config["curriculum_lengths"]],
            curriculum_boundaries=[int(x) for x in experiment_config["curriculum_boundaries"]],
            reflection_rollout_max_tokens=int(experiment_config["reflection_rollout_max_tokens"]),
            correctness_rollout_max_tokens=int(experiment_config.get("correctness_rollout_max_tokens", experiment_config["full_max_new_tokens"])),
        )

        self.same_tokenizer = _tokenizers_identical(self.processing_class, teacher_tokenizer)
        self.prefix_compatible_tokenizer = _tokenizers_prefix_compatible(self.processing_class, teacher_tokenizer)
        self.common_vocab_size = min(int(len(self.processing_class)), int(len(teacher_tokenizer)))

        requested = experiment_config.get("loss_backend", "auto")
        if requested == "auto":
            self.loss_backend = "trl_gjsd" if self.same_tokenizer else "sampled_rkl"
        else:
            self.loss_backend = requested
        if self.loss_backend == "trl_gjsd" and not self.same_tokenizer:
            raise ValueError("TRL full-vocabulary GKD requires identical tokenizers. Use sampled_rkl.")
        if self.loss_backend == "sampled_rkl" and float(experiment_config["beta"]) != 1.0:
            raise ValueError("sampled_rkl implements sampled reverse KL and requires beta=1.0")
        if self.loss_backend == "adaptive_opd":
            allow_prefix_compatible = bool(experiment_config.get("adaptive_allow_prefix_vocab_compatible", True))
            if not self.same_tokenizer:
                if not allow_prefix_compatible or not self.prefix_compatible_tokenizer:
                    raise ValueError(
                        "adaptive_opd requires identical tokenizers or prefix-compatible tokenizers "
                        "where the teacher only appends extra tokens after the student vocab. "
                        "For cross-tokenizer teachers, switch loss_backend to sampled_rkl."
                    )
                self.accelerator.print(
                    "[adaptive_opd] Student/teacher tokenizers are prefix-compatible but not identical. "
                    f"Using common_vocab_size={self.common_vocab_size}; teacher-only extra tokens are masked out "
                    "for overlap, reverse top-k KL, and forward top-k KL."
                )

        self.minimum_aligned_chars = int(experiment_config.get("minimum_aligned_chars", 1))
        self.rkl_advantage_clip = experiment_config.get("rkl_advantage_clip")

        self.reflection_log_path = experiment_config.get("reflection_log_path")
        if not self.reflection_log_path:
            self.reflection_log_path = str(Path(self.args.output_dir) / f"reflection_rank{self.accelerator.process_index}.jsonl")

        self.debug_log_jsonl = experiment_config.get("debug_log_jsonl")
        if self.debug_log_jsonl and not Path(self.debug_log_jsonl).is_absolute():
            self.debug_log_jsonl = str(Path(self.args.output_dir) / self.debug_log_jsonl)

    def _current_horizon(self) -> int:
        return self.schedule.horizon(int(self.state.global_step))

    def _safe_decode(self, ids: list[int] | torch.Tensor, max_chars: int | None = None) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        try:
            text = self.processing_class.decode([int(x) for x in ids], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        except Exception:
            text = str(ids[:128])
        if max_chars is not None:
            return text[:max_chars]
        return text

    def _strip_completion(self, row: torch.Tensor) -> tuple[list[int], bool]:
        ids: list[int] = []
        saw_eos = False
        eos = self.processing_class.eos_token_id
        pad = self.processing_class.pad_token_id
        for token in row.tolist():
            token = int(token)
            if pad is not None and token == pad:
                break
            ids.append(token)
            if eos is not None and token == eos:
                saw_eos = True
                break
        return ids, saw_eos

    @torch.no_grad()
    def _generate_student_rollouts(self, model: Any, inputs: dict[str, Any], horizon: int) -> tuple[list[list[int]], list[bool]]:
        generation_config = copy.deepcopy(self.generation_config)
        generation_config.max_new_tokens = int(horizon)
        generation_config.temperature = float(self.experiment_config["temperature"])
        generation_config.do_sample = bool(self.experiment_config.get("rollout_do_sample", True))
        generation_config.top_k = int(self.experiment_config.get("top_k", 0))
        generation_config.top_p = float(self.experiment_config.get("top_p", 1.0))

        generation_kwargs = dict(self.generation_kwargs)
        generation_kwargs["max_new_tokens"] = int(horizon)
        generation_kwargs["temperature"] = float(self.experiment_config["temperature"])
        generation_kwargs["do_sample"] = bool(self.experiment_config.get("rollout_do_sample", True))
        generation_kwargs["top_k"] = int(self.experiment_config.get("top_k", 0))
        generation_kwargs["top_p"] = float(self.experiment_config.get("top_p", 1.0))

        with unwrap_model_for_generation(model, self.accelerator, generation_kwargs=generation_kwargs) as unwrapped_model:
            outputs = unwrapped_model.generate(
                input_ids=inputs["prompts"],
                attention_mask=inputs.get("prompt_attention_mask"),
                generation_config=generation_config,
                return_dict_in_generate=True,
            )
        prompt_width = inputs["prompts"].shape[1]
        completions, eos_flags = [], []
        for row in outputs.sequences[:, prompt_width:]:
            ids, saw_eos = self._strip_completion(row)
            completions.append(ids)
            eos_flags.append(saw_eos)

        lengths = [len(ids) for ids in completions]
        self._last_rollout_diagnostics = {
            "generated_lengths": lengths,
            "eos_flags": eos_flags,
            "empty_generated_fraction": float(sum(x == 0 for x in lengths) / max(len(lengths), 1)),
            "eos_fraction": float(sum(eos_flags) / max(len(eos_flags), 1)),
            "truncated_fraction": float(sum((not e) and (l >= int(horizon)) for l, e in zip(lengths, eos_flags, strict=True)) / max(len(lengths), 1)),
            "mean_generated_tokens": float(sum(lengths) / max(len(lengths), 1)),
            "median_generated_tokens": _median(lengths),
            "max_generated_tokens": float(max(lengths) if lengths else 0),
            "min_generated_tokens": float(min(lengths) if lengths else 0),
        }
        return completions, eos_flags

    def _reflection_cut_lengths(self, completion_ids: list[list[int]], inputs: dict[str, Any]) -> tuple[list[int], list[dict[str, Any]]]:
        chunk_size = int(self.experiment_config["reflection_chunk_size"])
        prompts, all_chunks = [], []
        for ids, problem, reference in zip(completion_ids, inputs["problem"], inputs["reference_solution"], strict=True):
            chunks = split_token_chunks(self.processing_class, ids, chunk_size)
            all_chunks.append(chunks)
            prompts.append(build_reflection_prompt(problem, chunks, reference, use_reference=bool(self.experiment_config["reflection_use_reference"])))

        teacher = self.accelerator.unwrap_model(self.teacher_model)
        teacher.eval()
        decisions = judge_batch(
            teacher,
            self.teacher_tokenizer,
            prompts,
            max_new_tokens=int(self.experiment_config["reflection_max_new_tokens"]),
            use_chat_template=self.teacher_use_chat_template,
            enable_thinking=self.teacher_enable_thinking,
        )

        cut_lengths: list[int] = []
        logs: list[dict[str, Any]] = []
        fallback = self.experiment_config["reflection_parse_failure"]
        fallback_length = int(self.experiment_config["reflection_fallback_length"])
        min_keep = int(self.experiment_config.get("reflection_min_keep_tokens", 0))

        for ids, chunks, decision, problem in zip(completion_ids, all_chunks, decisions, inputs["problem"], strict=True):
            if decision.has_error is False:
                cut = len(ids)
            elif decision.has_error is True and decision.earliest_error_chunk_id is not None:
                cid = decision.earliest_error_chunk_id
                if 0 <= cid < len(chunks):
                    cut = chunks[cid]["start_token"]
                    if min_keep > 0:
                        cut = max(cut, min(min_keep, len(ids)))
                else:
                    cut = min(fallback_length, len(ids))
            elif fallback == "full":
                cut = len(ids)
            elif fallback == "skip":
                cut = 0
            else:
                cut = min(fallback_length, len(ids))
            cut_lengths.append(cut)
            logs.append({
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
                    "parsed": decision.parsed,
                },
            })

        if self.accelerator.is_local_main_process or self.accelerator.num_processes > 1:
            path = Path(self.reflection_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                for rec in logs:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return cut_lengths, logs

    def _correctness_cut_lengths(self, completion_ids: list[list[int]], inputs: dict[str, Any]) -> tuple[list[int], list[dict[str, Any]]]:
        fallback = str(self.experiment_config.get("correctness_wrong_fallback", "esr"))
        prefix = int(self.experiment_config["prefix_length"])
        mode = str(self.experiment_config.get("answer_extraction", "auto"))
        cut_lengths, logs = [], []
        for ids, problem, gt, ref in zip(completion_ids, inputs["problem"], inputs["ground_truth"], inputs["reference_solution"], strict=True):
            text = self.processing_class.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            cj = judge_correctness(text, ground_truth=gt, reference_solution=ref, mode=mode)
            if cj["is_correct"]:
                cut = len(ids)
            elif fallback == "skip":
                cut = 0
            else:
                cut = min(prefix, len(ids))
            cut_lengths.append(cut)
            logs.append({
                "global_step": int(self.state.global_step),
                "problem": problem,
                "student_rollout_ans": cj["student_answer"],
                "ground_truth_ans": cj["ground_truth_answer"],
                "student_correct": int(cj["is_correct"]),
                "completion_tokens": len(ids),
                "cut_tokens": cut,
            })
        return cut_lengths, logs

    def _build_student_batch(self, inputs: dict[str, Any], completions: list[list[int]], cut_lengths: list[int]) -> dict[str, Any]:
        pad_id = int(self.processing_class.pad_token_id)
        rows, labels, prompt_rows = [], [], []
        for i, (completion, cut) in enumerate(zip(completions, cut_lengths, strict=True)):
            mask = inputs["prompt_attention_mask"][i].bool()
            prompt = [int(x) for x in inputs["prompts"][i][mask].tolist()]
            truncated = [int(x) for x in completion[:cut]]
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
        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(padded_masks, dtype=torch.long, device=device),
            "labels": torch.tensor(padded_labels, dtype=torch.long, device=device),
            "student_prompt_lengths": [len(row) for row in prompt_rows],
            "student_completion_ids": [c[:cut] for c, cut in zip(completions, cut_lengths, strict=True)],
            "prompt_messages": inputs["prompt_messages"],
        }

    def _build_cross_tokenizer_batch(self, batch: dict[str, Any]) -> None:
        teacher_rows, alignment_groups = [], []
        for prompt_messages, student_ids, student_prompt_len in zip(batch["prompt_messages"], batch["student_completion_ids"], batch["student_prompt_lengths"], strict=True):
            teacher_prompt_ids = apply_chat_template_ids(
                self.teacher_tokenizer,
                prompt_messages,
                add_generation_prompt=True,
                use_chat_template=self.teacher_use_chat_template,
                enable_thinking=self.teacher_enable_thinking,
            )
            _, teacher_completion_ids, groups = build_text_span_alignment(
                self.processing_class,
                self.teacher_tokenizer,
                student_ids,
                minimum_aligned_chars=self.minimum_aligned_chars,
            )
            teacher_rows.append(teacher_prompt_ids + teacher_completion_ids)
            converted = []
            for group in groups:
                converted.append({
                    "student": [student_prompt_len + idx for idx in group.student_indices],
                    "teacher": [len(teacher_prompt_ids) + idx for idx in group.teacher_indices],
                    "chars": group.end_char - group.start_char,
                })
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

    def _write_debug_jsonl(self, rec: dict[str, Any]) -> None:
        if not self.debug_log_jsonl:
            return
        path = Path(self.debug_log_jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _print_step_debug(self, inputs: dict[str, Any], completions: list[list[int]], cut_lengths: list[int], strategy_logs: list[dict[str, Any]], horizon: int) -> None:
        if not bool(self.experiment_config.get("debug_print_every_step", True)):
            return
        if not self.accelerator.is_local_main_process:
            return
        n = min(int(self.experiment_config.get("debug_print_samples", 1)), len(completions))
        max_chars = int(self.experiment_config.get("debug_print_max_chars", 2500))
        for i in range(n):
            prompt_ids = inputs["prompts"][i][inputs["prompt_attention_mask"][i].bool()].tolist()
            rollout_text = self._safe_decode(completions[i], max_chars=max_chars)
            used_text = self._safe_decode(completions[i][: cut_lengths[i]], max_chars=max_chars)
            prompt_text = self._safe_decode(prompt_ids, max_chars=max_chars)
            correctness = judge_correctness(
                rollout_text,
                ground_truth=inputs["ground_truth"][i],
                reference_solution=inputs["reference_solution"][i],
                mode=str(self.experiment_config.get("answer_extraction", "auto")),
            )
            log_rec = strategy_logs[i] if i < len(strategy_logs) else {}
            print("\n" + "=" * 120, flush=True)
            print(f"【global step】{int(self.state.global_step)}", flush=True)
            print(f"【strategy】{self.strategy}", flush=True)
            print(f"【horizon】{horizon}", flush=True)
            print(f"【student prompt】\n{prompt_text}", flush=True)
            print(f"【student rollout】\n{rollout_text}", flush=True)
            print(f"【student rollout length】{len(completions[i])}", flush=True)
            print(f"【student rollout ans】{correctness['student_answer']}", flush=True)
            print(f"【ground truth ans】{correctness['ground_truth_answer']}", flush=True)
            print(f"【student correct】{int(correctness['is_correct'])}", flush=True)
            print(f"【cut tokens】{cut_lengths[i]}", flush=True)
            print(f"【used prefix】\n{used_text}", flush=True)
            if log_rec:
                print(f"【strategy decision】\n{json.dumps(log_rec, indent=2, ensure_ascii=False)[:max_chars]}", flush=True)
            print("=" * 120 + "\n", flush=True)
            self._write_debug_jsonl({
                "global_step": int(self.state.global_step),
                "strategy": self.strategy,
                "horizon": int(horizon),
                "sample_index": i,
                "problem": inputs["problem"][i],
                "student_prompt": prompt_text,
                "student_rollout": rollout_text,
                "student_rollout_length": len(completions[i]),
                "student_rollout_ans": correctness["student_answer"],
                "ground_truth_ans": correctness["ground_truth_answer"],
                "student_correct": int(correctness["is_correct"]),
                "cut_tokens": int(cut_lengths[i]),
                "used_prefix": used_text,
                "strategy_decision": log_rec,
            })

    def training_step(self, model: torch.nn.Module, inputs: dict[str, Any], num_items_in_batch: int | None = None) -> torch.Tensor:
        horizon = self._current_horizon()
        completions, _ = self._generate_student_rollouts(model, inputs, horizon)

        strategy_logs: list[dict[str, Any]] = []
        if self.strategy == "reflection":
            cut_lengths, strategy_logs = self._reflection_cut_lengths(completions, inputs)
        elif self.strategy == "correctness_esr":
            cut_lengths, strategy_logs = self._correctness_cut_lengths(completions, inputs)
        else:
            cut_lengths = [len(ids) for ids in completions]

        batch = self._build_student_batch(inputs, completions, cut_lengths)
        if self.loss_backend == "sampled_rkl":
            self._build_cross_tokenizer_batch(batch)

        generated = getattr(self, "_last_rollout_diagnostics", {})
        all_used = [int(x) for x in cut_lengths]
        valid_used = [x for x in all_used if x > 0]
        log_payload = {
            "rollout/horizon": float(horizon),
            "rollout/mean_used_tokens": float(sum(valid_used) / max(len(valid_used), 1)),
            "rollout/median_used_tokens": _median(valid_used),
            "rollout/max_used_tokens": float(max(all_used) if all_used else 0),
            "rollout/min_used_tokens": float(min(all_used) if all_used else 0),
            "rollout/skipped_fraction": float(sum(x == 0 for x in all_used) / max(len(all_used), 1)),
            "rollout/mean_generated_tokens": float(generated.get("mean_generated_tokens", 0.0)),
            "rollout/median_generated_tokens": float(generated.get("median_generated_tokens", 0.0)),
            "rollout/truncated_fraction": float(generated.get("truncated_fraction", 0.0)),
            "rollout/eos_fraction": float(generated.get("eos_fraction", 0.0)),
        }
        if self.strategy == "correctness_esr" and strategy_logs:
            log_payload["rollout/student_correct_fraction"] = float(sum(x.get("student_correct", 0) for x in strategy_logs) / len(strategy_logs))
        self.log(log_payload)
        self._print_step_debug(inputs, completions, cut_lengths, strategy_logs, horizon)
        return Trainer.training_step(self, model, batch, None)

    @staticmethod
    def _target_log_probs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shifted_logits = logits[:, :-1, :]
        targets = input_ids[:, 1:]
        selected = shifted_logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1).float()
        log_z = torch.logsumexp(shifted_logits.float(), dim=-1)
        return selected - log_z


    @staticmethod
    def _safe_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean over a boolean mask, preserving gradient when values require grad."""
        denom = mask.float().sum().clamp_min(1.0)
        return (values * mask.float()).sum() / denom

    @staticmethod
    def _topk_directional_kl(
        source_logits: torch.Tensor,
        target_logits: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """Approximate KL(source || target) on source top-k tokens.

        Returns one KL value per position with shape [batch, seq_minus_1].
        The distribution is renormalized inside the selected source top-k set.
        This is intentionally a compact top-k approximation, not full-vocab KL.
        """
        vocab = source_logits.shape[-1]
        k = max(1, min(int(top_k), int(vocab)))
        src_top = torch.topk(source_logits.float(), k=k, dim=-1)
        idx = src_top.indices
        src_selected_logits = src_top.values
        tgt_selected_logits = target_logits.float().gather(-1, idx)

        src_logp = F.log_softmax(src_selected_logits, dim=-1)
        tgt_logp = F.log_softmax(tgt_selected_logits, dim=-1)
        src_p = src_logp.exp()
        return (src_p * (src_logp - tgt_logp)).sum(dim=-1)

    @staticmethod
    def _topk_overlap(student_logits: torch.Tensor, teacher_logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Top-k set overlap |TopK_s ∩ TopK_t| / k for every token position."""
        vocab = student_logits.shape[-1]
        k = max(1, min(int(top_k), int(vocab)))
        s_idx = torch.topk(student_logits.float(), k=k, dim=-1).indices
        t_idx = torch.topk(teacher_logits.float(), k=k, dim=-1).indices
        # [B, T, k, k] equality matrix; k is small, normally 16.
        matches = (s_idx.unsqueeze(-1) == t_idx.unsqueeze(-2)).any(dim=-1)
        return matches.float().sum(dim=-1) / float(k)

    @staticmethod
    def _first_response_position(mask: torch.Tensor, active_mask: torch.Tensor) -> int:
        """Return first 1-indexed generated-token position satisfying mask, or -1."""
        if not bool(mask.any().item()):
            return -1
        response_positions = torch.cumsum(active_mask.long(), dim=1) - 1
        selected = response_positions[mask]
        if selected.numel() == 0:
            return -1
        return int(selected.min().detach().cpu().item()) + 1

    def _adaptive_opd_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool):
        """Overlap-gated OPD objective.

        Current intended behavior:
          - high overlap: reverse KL only
          - mid/low overlap: reverse KL + lambda * forward KL

        This keeps reverse KL as the on-policy backbone and uses forward KL only
        as a weak auxiliary correction. Low band can be disabled; when disabled,
        all below-threshold tokens are treated as mid.
        """
        student_outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False)
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
            )

        # Token t is predicted by logits at position t-1.
        student_logits = student_outputs.logits[:, :-1, :]
        teacher_logits = teacher_outputs.logits[:, :-1, :]

        # If the teacher has extra special tokens appended at the end of the vocab,
        # mask them out by slicing both logits to the shared token-id prefix.
        # This preserves compatibility with Qwen-style tokenizer pairs where
        # token ids 0..len(student)-1 are shared and only teacher-only tokens are
        # appended after that range.
        common_vocab = min(
            int(self.common_vocab_size),
            int(student_logits.shape[-1]),
            int(teacher_logits.shape[-1]),
        )
        student_logits_common = student_logits[..., :common_vocab]
        teacher_logits_common = teacher_logits[..., :common_vocab]

        shifted_labels = inputs["labels"][:, 1:]
        shifted_attn = inputs["attention_mask"][:, 1:].bool()
        active = (shifted_labels != -100) & shifted_attn

        reverse_top_k = int(self.experiment_config.get("adaptive_reverse_top_k", 16))
        forward_top_k = int(self.experiment_config.get("adaptive_forward_top_k", 16))
        overlap_top_k = int(self.experiment_config.get("adaptive_overlap_top_k", 16))
        eps = float(self.experiment_config.get("adaptive_loss_eps", 1.0e-8))

        # reverse_loss approximates KL(student || teacher) on student top-k.
        reverse_loss = self._topk_directional_kl(student_logits_common, teacher_logits_common, reverse_top_k)
        # forward_loss approximates KL(teacher || student) on teacher top-k.
        # Teacher-only extra special tokens are excluded by the common-vocab slice above.
        forward_loss = self._topk_directional_kl(teacher_logits_common, student_logits_common, forward_top_k)
        overlap = self._topk_overlap(student_logits_common, teacher_logits_common, overlap_top_k)

        high_thr = float(self.experiment_config.get("adaptive_overlap_threshold", 0.55))
        low_thr = float(self.experiment_config.get("adaptive_overlap_low_threshold", 0.0))
        use_low = bool(self.experiment_config.get("adaptive_use_low_band", False))
        mid_lambda = float(self.experiment_config.get("adaptive_forward_lambda", 0.10))
        low_lambda = float(self.experiment_config.get("adaptive_low_forward_lambda", mid_lambda))
        reverse_weight = float(self.experiment_config.get("adaptive_reverse_weight", 1.0))
        forward_weight = float(self.experiment_config.get("adaptive_forward_weight", 1.0))

        high_mask = active & (overlap >= high_thr)
        if use_low:
            low_mask = active & (overlap < low_thr)
        else:
            low_mask = torch.zeros_like(active, dtype=torch.bool)
        mid_mask = active & (~high_mask) & (~low_mask)
        forward_mask = mid_mask | low_mask

        lambda_map = torch.zeros_like(reverse_loss, dtype=reverse_loss.dtype)
        lambda_map = torch.where(mid_mask, torch.full_like(lambda_map, mid_lambda), lambda_map)
        lambda_map = torch.where(low_mask, torch.full_like(lambda_map, low_lambda), lambda_map)

        token_loss = reverse_weight * reverse_loss + forward_weight * lambda_map * forward_loss
        loss = self._safe_mean(token_loss, active)

        # Lightweight diagnostics. These mirror your existing logs and help check
        # whether forward KL dominates despite a small trigger fraction.
        with torch.no_grad():
            prefix = str(self.experiment_config.get("adaptive_log_prefix", "adaptive_loss/opd"))
            active_count = active.float().sum().clamp_min(1.0)
            high_fraction = high_mask.float().sum() / active_count
            mid_fraction = mid_mask.float().sum() / active_count
            low_fraction = low_mask.float().sum() / active_count
            mean_overlap = self._safe_mean(overlap.detach(), active)
            mean_reverse = self._safe_mean(reverse_loss.detach(), active)
            if bool(forward_mask.any().item()):
                mean_forward = self._safe_mean(forward_loss.detach(), forward_mask)
            else:
                mean_forward = torch.zeros((), device=loss.device)
            forward_reverse_ratio = mean_forward / (mean_reverse + eps)

            # Position buckets over generated tokens, not prompt tokens.
            response_pos = torch.cumsum(active.long(), dim=1) - 1
            response_len = active.long().sum(dim=1).clamp_min(1).unsqueeze(1)
            rel_pos = (response_pos.float() + 1.0) / response_len.float()
            early_mask = active & (rel_pos <= (1.0 / 3.0))
            middle_mask = active & (rel_pos > (1.0 / 3.0)) & (rel_pos <= (2.0 / 3.0))
            late_mask = active & (rel_pos > (2.0 / 3.0))

            ov = overlap.detach()[active]
            if ov.numel() > 0:
                q = torch.quantile(ov.float(), torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], device=ov.device))
                p10, p25, p50, p75, p90 = [float(x.detach().cpu().item()) for x in q]
            else:
                p10 = p25 = p50 = p75 = p90 = 0.0

            self.log({
                f"{prefix}/loss": float(loss.detach().cpu().item()),
                f"{prefix}/mean_reverse_loss": float(mean_reverse.detach().cpu().item()),
                f"{prefix}/mean_forward_loss": float(mean_forward.detach().cpu().item()),
                f"{prefix}/forward_reverse_ratio": float(forward_reverse_ratio.detach().cpu().item()),
                f"{prefix}/mean_overlap": float(mean_overlap.detach().cpu().item()),
                f"{prefix}/adaptive_high_fraction": float(high_fraction.detach().cpu().item()),
                f"{prefix}/adaptive_mid_fraction": float(mid_fraction.detach().cpu().item()),
                f"{prefix}/adaptive_low_fraction": float(low_fraction.detach().cpu().item()),
                f"{prefix}/first_forward_position": float(self._first_response_position(forward_mask, active)),
                f"{prefix}/first_low_position": float(self._first_response_position(low_mask, active)),
                f"{prefix}/overlap_early": float(self._safe_mean(overlap.detach(), early_mask).detach().cpu().item()),
                f"{prefix}/overlap_middle": float(self._safe_mean(overlap.detach(), middle_mask).detach().cpu().item()),
                f"{prefix}/overlap_late": float(self._safe_mean(overlap.detach(), late_mask).detach().cpu().item()),
                f"{prefix}/overlap_p10": p10,
                f"{prefix}/overlap_p25": p25,
                f"{prefix}/overlap_p50": p50,
                f"{prefix}/overlap_p75": p75,
                f"{prefix}/overlap_p90": p90,
                f"{prefix}/common_vocab_size": float(common_vocab),
                f"{prefix}/student_vocab_size": float(student_logits.shape[-1]),
                f"{prefix}/teacher_vocab_size": float(teacher_logits.shape[-1]),
            })

        return (loss, student_outputs) if return_outputs else loss

    def _sampled_rkl_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool):
        student_outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False)
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(input_ids=inputs["teacher_input_ids"], attention_mask=inputs["teacher_attention_mask"], use_cache=False)

        student_lp = self._target_log_probs(student_outputs.logits, inputs["input_ids"])
        teacher_lp = self._target_log_probs(teacher_outputs.logits, inputs["teacher_input_ids"])
        terms, weights = [], []
        for batch_idx, groups in enumerate(inputs["alignment_groups"]):
            for group in groups:
                s_positions = [idx - 1 for idx in group["student"] if idx > 0]
                t_positions = [idx - 1 for idx in group["teacher"] if idx > 0]
                if not s_positions or not t_positions:
                    continue
                s_logp = student_lp[batch_idx, s_positions].sum()
                t_logp = teacher_lp[batch_idx, t_positions].sum()
                advantage = s_logp.detach() - t_logp.detach()
                if self.rkl_advantage_clip is not None:
                    clip = float(self.rkl_advantage_clip)
                    advantage = advantage.clamp(-clip, clip)
                terms.append(advantage * s_logp)
                weights.append(max(len(s_positions), 1))
        if not terms:
            loss = student_outputs.logits.sum() * 0.0
        else:
            loss = torch.stack(terms).sum() / max(float(sum(weights)), 1.0)
        return (loss, student_outputs) if return_outputs else loss

    def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, num_items_in_batch: int | None = None):
        if self.loss_backend == "sampled_rkl":
            return self._sampled_rkl_loss(model, inputs, return_outputs)
        if self.loss_backend == "adaptive_opd":
            return self._adaptive_opd_loss(model, inputs, return_outputs)
        clean = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "labels": inputs["labels"]}
        return super().compute_loss(model, clean, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
