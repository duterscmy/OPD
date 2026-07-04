from __future__ import annotations

from typing import Any

import torch

from .trainer import AdaptiveOPDTrainer as BaseAdaptiveOPDTrainer
from .adaptive_kl_losses import compute_adaptive_kl_loss


class AdaptiveKLTrainer(BaseAdaptiveOPDTrainer):
    """Drop-in trainer adding forward/reverse/mixture/prune/adaptive KL losses.

    Use this trainer only when teacher and student tokenizers are identical.
    The original BaseAdaptiveOPDTrainer remains available for cross-tokenizer
    sampled reverse KL.
    """

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        mode = str(self.experiment_config.get("opd_loss_mode", "original"))
        if mode in {"original", "trl_gjsd", "sampled_rkl"}:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        if not self.same_tokenizer:
            raise ValueError(
                "Adaptive KL / top-k overlap losses require identical teacher/student tokenizers. "
                "For cross-tokenizer pairs, either use loss_backend=sampled_rkl or choose models with shared vocab."
            )

        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
        )
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
            )

        out = compute_adaptive_kl_loss(
            student_logits_raw=student_outputs.logits,
            teacher_logits_raw=teacher_outputs.logits,
            labels=inputs["labels"],
            cfg=self.experiment_config,
        )
        if self.accelerator.sync_gradients:
            self.log(out.logs)
        return (out.loss, student_outputs) if return_outputs else out.loss
