from __future__ import annotations

import time
from typing import Any

import torch

from .trainer import AdaptiveOPDTrainer as BaseAdaptiveOPDTrainer
from .adaptive_kl_losses import compute_adaptive_kl_loss


class AdaptiveKLTrainer(BaseAdaptiveOPDTrainer):
    """Drop-in trainer adding forward/reverse/mixture/prune/adaptive KL losses.

    This version adds detailed logging for all training methods:
      - original / trl_gjsd / sampled_rkl paths delegated to BaseAdaptiveOPDTrainer
      - adaptive KL / top-k overlap / prune / forward KL / reverse KL paths

    Use this trainer only when teacher and student tokenizers are identical for
    adaptive KL / top-k overlap losses. The original BaseAdaptiveOPDTrainer path
    remains available for cross-tokenizer sampled reverse KL.
    """

    def _get_global_step_safe(self) -> int:
        try:
            return int(self.state.global_step)
        except Exception:
            return -1

    def _should_verbose_log(self) -> bool:
        """Control detailed training logs.

        Config options:
          debug_train_log: bool, default True
          debug_train_log_steps: int, default logging_steps or 1

        Example YAML:
          debug_train_log: true
          debug_train_log_steps: 1
        """
        if not bool(self.experiment_config.get("debug_train_log", True)):
            return False

        if not self.accelerator.is_main_process:
            return False

        step = self._get_global_step_safe()

        log_every = int(
            self.experiment_config.get(
                "debug_train_log_steps",
                self.experiment_config.get("logging_steps", 1),
            )
        )
        log_every = max(log_every, 1)

        # step can stay the same during gradient accumulation, but this is still
        # useful for debugging early training behavior.
        return step < 5 or step % log_every == 0

    def _sync_cuda_if_needed(self) -> None:
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

    def _tensor_stats(self, x: torch.Tensor, name: str) -> dict[str, Any]:
        if x is None:
            return {f"{name}/is_none": True}

        stats: dict[str, Any] = {
            f"{name}/shape": list(x.shape),
            f"{name}/dtype": str(x.dtype),
            f"{name}/device": str(x.device),
        }

        if x.numel() > 0 and torch.is_floating_point(x):
            with torch.no_grad():
                finite = torch.isfinite(x)
                stats.update(
                    {
                        f"{name}/finite_ratio": float(finite.float().mean().item()),
                        f"{name}/mean": float(x[finite].mean().item()) if finite.any() else float("nan"),
                        f"{name}/std": float(x[finite].std().item()) if finite.any() and finite.sum() > 1 else 0.0,
                        f"{name}/min": float(x[finite].min().item()) if finite.any() else float("nan"),
                        f"{name}/max": float(x[finite].max().item()) if finite.any() else float("nan"),
                    }
                )
        return stats

    def _summarize_batch(self, inputs: dict[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {}

        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")

        if isinstance(input_ids, torch.Tensor):
            summary["batch/input_ids_shape"] = list(input_ids.shape)
            summary["batch/input_ids_dtype"] = str(input_ids.dtype)
            summary["batch/input_ids_device"] = str(input_ids.device)
            summary["batch/batch_size"] = int(input_ids.shape[0]) if input_ids.ndim >= 1 else 0
            summary["batch/seq_len"] = int(input_ids.shape[1]) if input_ids.ndim >= 2 else 0

        if isinstance(attention_mask, torch.Tensor):
            with torch.no_grad():
                lengths = attention_mask.sum(dim=1).float() if attention_mask.ndim == 2 else attention_mask.float()
                summary["batch/attention_tokens_total"] = int(attention_mask.sum().item())
                summary["batch/attention_len_mean"] = float(lengths.mean().item())
                summary["batch/attention_len_min"] = float(lengths.min().item())
                summary["batch/attention_len_max"] = float(lengths.max().item())

        if isinstance(labels, torch.Tensor):
            with torch.no_grad():
                valid = labels.ne(-100)
                valid_per_sample = valid.sum(dim=1).float() if labels.ndim == 2 else valid.float()
                summary["batch/label_shape"] = list(labels.shape)
                summary["batch/label_valid_tokens_total"] = int(valid.sum().item())
                summary["batch/label_valid_len_mean"] = float(valid_per_sample.mean().item())
                summary["batch/label_valid_len_min"] = float(valid_per_sample.min().item())
                summary["batch/label_valid_len_max"] = float(valid_per_sample.max().item())

                if isinstance(attention_mask, torch.Tensor) and attention_mask.shape == labels.shape:
                    prompt_like = attention_mask.bool() & labels.eq(-100)
                    prompt_like_per_sample = prompt_like.sum(dim=1).float()
                    summary["batch/prompt_like_tokens_total"] = int(prompt_like.sum().item())
                    summary["batch/prompt_like_len_mean"] = float(prompt_like_per_sample.mean().item())
                    summary["batch/prompt_like_len_min"] = float(prompt_like_per_sample.min().item())
                    summary["batch/prompt_like_len_max"] = float(prompt_like_per_sample.max().item())

        return summary

    def _print_debug_block(self, title: str, payload: dict[str, Any]) -> None:
        if not self._should_verbose_log():
            return

        step = self._get_global_step_safe()
        rank = getattr(self.accelerator, "process_index", 0)

        print("")
        print("=" * 90)
        print(f"[AdaptiveKLTrainer DEBUG] {title}")
        print(f"step={step} rank={rank}")
        print("-" * 90)

        for key in sorted(payload.keys()):
            value = payload[key]
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")

        print("=" * 90)
        print("", flush=True)

    def _log_to_trainer(self, payload: dict[str, Any]) -> None:
        """Log numeric values into Trainer/Accelerate logger."""
        numeric_payload: dict[str, float] = {}
        for key, value in payload.items():
            if isinstance(value, bool):
                numeric_payload[key] = float(value)
            elif isinstance(value, int):
                numeric_payload[key] = float(value)
            elif isinstance(value, float):
                numeric_payload[key] = value
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                numeric_payload[key] = float(value.detach().float().cpu().item())

        if numeric_payload and self.accelerator.sync_gradients:
            self.log(numeric_payload)

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        mode = str(self.experiment_config.get("opd_loss_mode", "original"))
        strategy = str(self.experiment_config.get("strategy", "unknown"))
        step = self._get_global_step_safe()

        base_payload: dict[str, Any] = {
            "step": step,
            "mode/opd_loss_mode": mode,
            "mode/strategy": strategy,
            "mode/loss_backend": getattr(self, "loss_backend", None),
            "mode/same_tokenizer": bool(getattr(self, "same_tokenizer", False)),
            "mode/return_outputs": bool(return_outputs),
            "mode/num_items_in_batch": num_items_in_batch,
            "config/max_length": self.experiment_config.get("max_length"),
            "config/max_prompt_length": self.experiment_config.get("max_prompt_length"),
            "config/full_max_new_tokens": self.experiment_config.get("full_max_new_tokens"),
            "config/lite_max_new_tokens": self.experiment_config.get("lite_max_new_tokens"),
            "config/esr_cut_length": self.experiment_config.get("esr_cut_length"),
            "args/max_new_tokens": getattr(self.args, "max_new_tokens", None),
            "args/per_device_train_batch_size": getattr(self.args, "per_device_train_batch_size", None),
            "args/gradient_accumulation_steps": getattr(self.args, "gradient_accumulation_steps", None),
        }
        base_payload.update(self._summarize_batch(inputs))

        # ---------------------------------------------------------------------
        # Original TRL/GKD/sample reverse KL path.
        # ---------------------------------------------------------------------
        if mode in {"original", "trl_gjsd", "sampled_rkl"}:
            self._print_debug_block(
                title="Entering base OPD/GKD loss path",
                payload={
                    **base_payload,
                    "path": "super.compute_loss",
                    "note": (
                        "This path is delegated to BaseAdaptiveOPDTrainer. "
                        "If it is slow, inspect rollout generation and teacher forward inside the base trainer."
                    ),
                },
            )

            self._sync_cuda_if_needed()
            t0 = time.perf_counter()

            loss_or_outputs = super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

            self._sync_cuda_if_needed()
            elapsed = time.perf_counter() - t0

            if return_outputs:
                loss_value = loss_or_outputs[0]
            else:
                loss_value = loss_or_outputs

            final_payload = {
                **base_payload,
                "path": "super.compute_loss",
                "timing/total_compute_loss_sec": elapsed,
                "loss/value": float(loss_value.detach().float().cpu().item())
                if isinstance(loss_value, torch.Tensor)
                else None,
            }

            self._print_debug_block(
                title="Finished base OPD/GKD loss path",
                payload=final_payload,
            )
            self._log_to_trainer(
                {
                    "debug/base_total_compute_loss_sec": elapsed,
                    "debug/base_loss": final_payload["loss/value"]
                    if final_payload["loss/value"] is not None
                    else 0.0,
                }
            )

            return loss_or_outputs

        # ---------------------------------------------------------------------
        # Adaptive KL / prune / top-k overlap path.
        # ---------------------------------------------------------------------
        if not self.same_tokenizer:
            raise ValueError(
                "Adaptive KL / top-k overlap losses require identical teacher/student tokenizers. "
                "For cross-tokenizer pairs, either use loss_backend=sampled_rkl or choose models with shared vocab."
            )

        self._print_debug_block(
            title="Entering adaptive KL loss path",
            payload={
                **base_payload,
                "path": "adaptive_kl",
                "note": (
                    "This path performs student forward and teacher forward on inputs['input_ids']. "
                    "If lite_prune is as slow as full OPD, check whether input_ids length and teacher forward length are actually reduced."
                ),
            },
        )

        total_t0 = time.perf_counter()

        # Student forward.
        self._sync_cuda_if_needed()
        student_t0 = time.perf_counter()

        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
        )

        self._sync_cuda_if_needed()
        student_elapsed = time.perf_counter() - student_t0

        # Teacher forward.
        self.teacher_model.eval()
        self._sync_cuda_if_needed()
        teacher_t0 = time.perf_counter()

        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
            )

        self._sync_cuda_if_needed()
        teacher_elapsed = time.perf_counter() - teacher_t0

        # Loss computation.
        loss_t0 = time.perf_counter()

        out = compute_adaptive_kl_loss(
            student_logits_raw=student_outputs.logits,
            teacher_logits_raw=teacher_outputs.logits,
            labels=inputs["labels"],
            cfg=self.experiment_config,
        )

        self._sync_cuda_if_needed()
        loss_elapsed = time.perf_counter() - loss_t0
        total_elapsed = time.perf_counter() - total_t0

        debug_payload: dict[str, Any] = {
            **base_payload,
            "path": "adaptive_kl",
            "timing/student_forward_sec": student_elapsed,
            "timing/teacher_forward_sec": teacher_elapsed,
            "timing/loss_compute_sec": loss_elapsed,
            "timing/total_compute_loss_sec": total_elapsed,
            "loss/value": float(out.loss.detach().float().cpu().item()),
        }

        debug_payload.update(self._tensor_stats(student_outputs.logits, "student_logits"))
        debug_payload.update(self._tensor_stats(teacher_outputs.logits, "teacher_logits"))

        # Add internal loss logs from compute_adaptive_kl_loss.
        if hasattr(out, "logs") and isinstance(out.logs, dict):
            for key, value in out.logs.items():
                safe_key = f"adaptive_loss/{key}"
                if isinstance(value, torch.Tensor):
                    if value.numel() == 1:
                        debug_payload[safe_key] = float(value.detach().float().cpu().item())
                    else:
                        debug_payload[f"{safe_key}/shape"] = list(value.shape)
                else:
                    debug_payload[safe_key] = value

        # Helpful derived speed ratios.
        if total_elapsed > 0:
            debug_payload["timing/student_forward_ratio"] = student_elapsed / total_elapsed
            debug_payload["timing/teacher_forward_ratio"] = teacher_elapsed / total_elapsed
            debug_payload["timing/loss_compute_ratio"] = loss_elapsed / total_elapsed

        if teacher_elapsed > 0:
            debug_payload["timing/student_to_teacher_forward_ratio"] = student_elapsed / teacher_elapsed

        self._print_debug_block(
            title="Finished adaptive KL loss path",
            payload=debug_payload,
        )

        # Log to Trainer as scalar metrics.
        scalar_log_payload = {
            "debug/student_forward_sec": student_elapsed,
            "debug/teacher_forward_sec": teacher_elapsed,
            "debug/loss_compute_sec": loss_elapsed,
            "debug/total_compute_loss_sec": total_elapsed,
            "debug/adaptive_loss": float(out.loss.detach().float().cpu().item()),
        }

        if hasattr(out, "logs") and isinstance(out.logs, dict):
            for key, value in out.logs.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    scalar_log_payload[f"loss_detail/{key}"] = float(value.detach().float().cpu().item())
                elif isinstance(value, (int, float, bool)):
                    scalar_log_payload[f"loss_detail/{key}"] = float(value)

        self._log_to_trainer(scalar_log_payload)

        # Keep the original behavior.
        if self.accelerator.sync_gradients and hasattr(out, "logs"):
            self.log(out.logs)

        return (out.loss, student_outputs) if return_outputs else out.loss