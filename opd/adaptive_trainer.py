from __future__ import annotations

import time
from typing import Any

import torch

from .adaptive_kl_losses import ROUTE_NAMES, TopKOPDLossOutput, compute_topk_opd_loss
from .trainer import AdaptiveOPDTrainer


class AdaptiveKLTrainer(AdaptiveOPDTrainer):
    """Top-K OPD trainer with scalar and per-token JSONL diagnostics."""

    def _should_collect_token_debug(self) -> bool:
        if not bool(self.experiment_config.get("debug_jsonl_enabled", True)):
            return False
        every = max(int(self.experiment_config.get("debug_every_n_loss_calls", 1)), 1)
        return self._loss_call_index % every == 0

    @staticmethod
    def _tensor_description(tensor: torch.Tensor) -> dict[str, Any]:
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "numel": int(tensor.numel()),
        }

    @staticmethod
    def _to_float(tensor: torch.Tensor, index: tuple[int, ...]) -> float:
        return float(tensor[index].detach().float().cpu().item())

    def _token_string(self, token_id: int, teacher: bool = False) -> str:
        tokenizer = self.teacher_tokenizer if teacher else self.processing_class
        try:
            return str(tokenizer.convert_ids_to_tokens(int(token_id)))
        except Exception:
            return tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

    def _topk_entries(
        self,
        ids: torch.Tensor,
        local_prob: torch.Tensor,
        full_prob: torch.Tensor | None,
        batch_index: int,
        sequence_index: int,
        teacher: bool,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for rank in range(int(ids.shape[-1])):
            token_id = int(ids[batch_index, sequence_index, rank].detach().cpu().item())
            entry = {
                "rank": rank + 1,
                "token_id": token_id,
                "token": self._token_string(token_id, teacher=teacher),
                "local_topk_probability": self._to_float(
                    local_prob, (batch_index, sequence_index, rank)
                ),
            }
            if full_prob is not None:
                entry["full_vocabulary_probability"] = self._to_float(
                    full_prob, (batch_index, sequence_index, rank)
                )
            result.append(entry)
        return result

    def _write_detailed_records(
        self,
        inputs: dict[str, Any],
        student_outputs: Any,
        teacher_outputs: Any,
        output: TopKOPDLossOutput,
        timing: dict[str, float],
    ) -> None:
        diagnostics = output.diagnostics
        active = diagnostics["active"]
        batch_size = int(active.shape[0])
        configured_samples = int(self.experiment_config.get("debug_samples_per_batch", 0))
        sample_count = batch_size if configured_samples <= 0 else min(configured_samples, batch_size)
        configured_tokens = int(self.experiment_config.get("debug_max_tokens_per_sample", 0))

        student_full_prob = diagnostics.get("student_topk_full_prob")
        teacher_full_prob = diagnostics.get("teacher_topk_full_prob")
        student_target_logp = diagnostics.get("student_target_logp")
        teacher_target_logp = diagnostics.get("teacher_target_logp")

        for batch_index in range(sample_count):
            positions = torch.where(active[batch_index])[0].detach().cpu().tolist()
            if configured_tokens > 0:
                positions = positions[:configured_tokens]
            token_records: list[dict[str, Any]] = []
            for response_index, sequence_index in enumerate(positions, start=1):
                target_id = int(
                    diagnostics["targets"][batch_index, sequence_index].detach().cpu().item()
                )
                route_code = int(
                    diagnostics["route_code"][batch_index, sequence_index].detach().cpu().item()
                )
                student_ids = diagnostics["student_topk_ids"]
                teacher_ids = diagnostics["teacher_topk_ids"]
                target_in_student_topk = bool(
                    student_ids[batch_index, sequence_index].eq(target_id).any().item()
                )
                target_in_teacher_topk = bool(
                    teacher_ids[batch_index, sequence_index].eq(target_id).any().item()
                )
                student_target_matches = torch.where(
                    student_ids[batch_index, sequence_index].eq(target_id)
                )[0]
                teacher_target_matches = torch.where(
                    teacher_ids[batch_index, sequence_index].eq(target_id)
                )[0]
                student_id_list = [
                    int(x) for x in student_ids[batch_index, sequence_index].detach().cpu().tolist()
                ]
                teacher_id_set = {
                    int(x) for x in teacher_ids[batch_index, sequence_index].detach().cpu().tolist()
                }
                intersection_ids = [token_id for token_id in student_id_list if token_id in teacher_id_set]
                record: dict[str, Any] = {
                    "response_position": response_index,
                    "sequence_position": int(sequence_index) + 1,
                    "target_token_id": target_id,
                    "target_token": self._token_string(target_id),
                    "loss_route": ROUTE_NAMES.get(route_code, f"unknown_{route_code}"),
                    "overlap": self._to_float(diagnostics["overlap"], (batch_index, sequence_index)),
                    "reverse_kl": self._to_float(diagnostics["reverse_loss"], (batch_index, sequence_index)),
                    "forward_kl": self._to_float(diagnostics["forward_loss"], (batch_index, sequence_index)),
                    "reverse_coefficient": self._to_float(
                        diagnostics["reverse_coefficient"], (batch_index, sequence_index)
                    ),
                    "forward_coefficient": self._to_float(
                        diagnostics["forward_coefficient"], (batch_index, sequence_index)
                    ),
                    "prune_weight": self._to_float(
                        diagnostics["prune_weight"], (batch_index, sequence_index)
                    ),
                    "cumulative_low_overlap_count": self._to_float(
                        diagnostics["cumulative_low_overlap_count"],
                        (batch_index, sequence_index),
                    ),
                    "final_token_loss": self._to_float(
                        diagnostics["token_loss"], (batch_index, sequence_index)
                    ),
                    "target_in_student_topk": target_in_student_topk,
                    "target_in_teacher_topk": target_in_teacher_topk,
                    "target_rank_in_student_topk": (
                        int(student_target_matches[0].detach().cpu().item()) + 1
                        if student_target_matches.numel() else None
                    ),
                    "target_rank_in_teacher_topk": (
                        int(teacher_target_matches[0].detach().cpu().item()) + 1
                        if teacher_target_matches.numel() else None
                    ),
                    "topk_intersection_count": len(intersection_ids),
                    "topk_intersection": [
                        {
                            "token_id": token_id,
                            "token": self._token_string(token_id),
                        }
                        for token_id in intersection_ids
                    ],
                    "student_topk_local_entropy": self._to_float(
                        diagnostics["student_topk_local_entropy"],
                        (batch_index, sequence_index),
                    ),
                    "teacher_topk_local_entropy": self._to_float(
                        diagnostics["teacher_topk_local_entropy"],
                        (batch_index, sequence_index),
                    ),
                    "student_topk": self._topk_entries(
                        student_ids,
                        diagnostics["student_topk_local_prob"],
                        student_full_prob,
                        batch_index,
                        sequence_index,
                        teacher=False,
                    ),
                    "teacher_topk": self._topk_entries(
                        teacher_ids,
                        diagnostics["teacher_topk_local_prob"],
                        teacher_full_prob,
                        batch_index,
                        sequence_index,
                        teacher=True,
                    ),
                }
                if student_target_logp is not None and teacher_target_logp is not None:
                    student_logp = self._to_float(
                        student_target_logp, (batch_index, sequence_index)
                    )
                    teacher_logp = self._to_float(
                        teacher_target_logp, (batch_index, sequence_index)
                    )
                    record.update(
                        student_target_log_probability=student_logp,
                        teacher_target_log_probability=teacher_logp,
                        sampled_token_log_ratio=student_logp - teacher_logp,
                        student_topk_probability_mass=self._to_float(
                            diagnostics["student_topk_mass"],
                            (batch_index, sequence_index),
                        ),
                        teacher_topk_probability_mass=self._to_float(
                            diagnostics["teacher_topk_mass"],
                            (batch_index, sequence_index),
                        ),
                    )
                token_records.append(record)

            rollout_texts = inputs.get("debug_rollout_text", [""] * batch_size)
            rollout_lengths = inputs.get("debug_rollout_length", [len(positions)] * batch_size)
            problems = inputs.get("debug_problem", [""] * batch_size)
            prompt_texts = inputs.get("debug_prompt_text", [""] * batch_size)
            eos_flags = inputs.get("debug_eos", [False] * batch_size)
            record = {
                "event": "topk_opd_sample_debug",
                "global_step": int(self.state.global_step),
                "loss_call_index": int(self._loss_call_index),
                "rank": int(self.accelerator.process_index),
                "backend": self.loss_backend,
                "loss_mode": self.opd_loss_mode,
                "strategy": self.strategy,
                "horizon": int(inputs.get("debug_horizon", -1)),
                "sample_index_in_batch": batch_index,
                "problem": problems[batch_index],
                "prompt_text": prompt_texts[batch_index],
                "student_rollout": rollout_texts[batch_index],
                "student_rollout_length": int(rollout_lengths[batch_index]),
                "student_emitted_eos": bool(eos_flags[batch_index]),
                "logged_token_count": len(token_records),
                "student_forward": {
                    "input_ids": self._tensor_description(inputs["input_ids"]),
                    "attention_mask": self._tensor_description(inputs["attention_mask"]),
                    "logits": self._tensor_description(student_outputs.logits),
                    "elapsed_seconds": timing["student_forward_sec"],
                },
                "teacher_forward": {
                    "input_ids": self._tensor_description(inputs["input_ids"]),
                    "attention_mask": self._tensor_description(inputs["attention_mask"]),
                    "logits": self._tensor_description(teacher_outputs.logits),
                    "elapsed_seconds": timing["teacher_forward_sec"],
                },
                "loss_compute_seconds": timing["loss_compute_sec"],
                "total_compute_seconds": timing["total_compute_sec"],
                "batch_scalar_metrics": output.logs,
                "tokens": token_records,
            }
            if torch.cuda.is_available():
                record["cuda_memory"] = {
                    "allocated_bytes": int(torch.cuda.memory_allocated()),
                    "reserved_bytes": int(torch.cuda.memory_reserved()),
                    "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                }
            self._write_jsonl(record)

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> Any:
        if self.loss_backend == "sampled_rkl":
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        del num_items_in_batch
        self._loss_call_index += 1
        collect_debug = self._should_collect_token_debug()

        total_start = time.perf_counter()
        student_start = time.perf_counter()
        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
        )
        student_elapsed = time.perf_counter() - student_start

        teacher_start = time.perf_counter()
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
            )
        teacher_elapsed = time.perf_counter() - teacher_start

        loss_start = time.perf_counter()
        output = compute_topk_opd_loss(
            student_logits_raw=student_outputs.logits,
            teacher_logits_raw=teacher_outputs.logits,
            labels=inputs["labels"],
            cfg=self.experiment_config,
            collect_diagnostics=collect_debug,
        )
        loss_elapsed = time.perf_counter() - loss_start
        total_elapsed = time.perf_counter() - total_start
        output.logs.update(
            {
                "timing/student_forward_sec": float(student_elapsed),
                "timing/teacher_forward_sec": float(teacher_elapsed),
                "timing/loss_compute_sec": float(loss_elapsed),
                "timing/total_compute_sec": float(total_elapsed),
                "tensor/student_batch_size": float(student_outputs.logits.shape[0]),
                "tensor/student_sequence_length": float(student_outputs.logits.shape[1]),
                "tensor/student_vocab_size": float(student_outputs.logits.shape[2]),
                "tensor/teacher_sequence_length": float(teacher_outputs.logits.shape[1]),
                "tensor/teacher_vocab_size": float(teacher_outputs.logits.shape[2]),
            }
        )
        self.log(output.logs)

        if collect_debug:
            self._write_detailed_records(
                inputs,
                student_outputs,
                teacher_outputs,
                output,
                {
                    "student_forward_sec": student_elapsed,
                    "teacher_forward_sec": teacher_elapsed,
                    "loss_compute_sec": loss_elapsed,
                    "total_compute_sec": total_elapsed,
                },
            )

        summary_every = max(
            int(self.experiment_config.get("debug_summary_every_n_loss_calls", 1)), 1
        )
        if self.accelerator.is_local_main_process and self._loss_call_index % summary_every == 0:
            print(
                "[Top-K OPD] "
                f"step={self.state.global_step} call={self._loss_call_index} "
                f"mode={self.opd_loss_mode} loss={output.logs['opd/loss']:.6f} "
                f"RKL={output.logs['opd/mean_reverse_loss']:.6f} "
                f"FKL={output.logs['opd/mean_forward_loss']:.6f} "
                f"overlap={output.logs['opd/mean_overlap']:.4f} "
                f"low={output.logs['opd/low_overlap_fraction']:.4f} "
                f"student_shape={list(student_outputs.logits.shape)} "
                f"teacher_shape={list(teacher_outputs.logits.shape)}",
                flush=True,
            )

        return (output.loss, student_outputs) if return_outputs else output.loss
