from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F


@dataclass
class AdaptiveKLLossOutput:
    loss: torch.Tensor
    logs: Dict[str, float]


def _get(cfg: dict[str, Any], key: str, default: Any) -> Any:
    return cfg.get(key, default)


def _shift_for_next_token(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return logits for positions predicting token t+1, target ids, and OPD mask.

    logits: [B, L, V]
    labels: [B, L], with -100 on prompt/pad and token ids on supervised completion tokens.
    We supervise next-token prediction at positions 0..L-2 where labels[:, 1:] is valid.
    """
    pred_logits = logits[:, :-1, :]
    target_ids = labels[:, 1:]
    mask = target_ids.ne(-100)
    target_ids = target_ids.clamp_min(0)
    return pred_logits, target_ids, mask


def _topk_log_probs_on_ids(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Full-vocab normalized log-probs for selected ids.

    logits: [B, T, V]
    ids: [B, T, K]
    returns: [B, T, K]
    """
    picked = logits.gather(-1, ids)
    log_z = torch.logsumexp(logits.float(), dim=-1, keepdim=True)
    return picked.float() - log_z


def _teacher_forward_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    renorm_topk: bool = True,
) -> torch.Tensor:
    """Top-k approximation to KL(T||S), returned per token [B,T].

    Gradient flows only through student logits. Teacher top-k defines support.
    If renorm_topk=True, teacher probabilities are normalized inside teacher top-k.
    """
    k = min(int(k), teacher_logits.shape[-1])
    with torch.no_grad():
        t_top = torch.topk(teacher_logits.float(), k=k, dim=-1).indices
        t_logp = _topk_log_probs_on_ids(teacher_logits, t_top)
        if renorm_topk:
            t_w = F.softmax(t_logp, dim=-1)
        else:
            t_w = t_logp.exp()
    s_logp = _topk_log_probs_on_ids(student_logits, t_top)
    # Cross entropy part of KL; teacher entropy is constant w.r.t. student.
    per_tok = -(t_w * s_logp).sum(dim=-1)
    return per_tok.masked_fill(~mask, 0.0)


def _student_reverse_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    renorm_topk: bool = True,
) -> torch.Tensor:
    """Top-k approximation to KL(S||T), returned per token [B,T].

    Student top-k defines support. This is a differentiable top-k approximation:
    the selected ids are detached by topk, while log-probs/probs on those ids carry
    gradients through student logits.
    """
    k = min(int(k), student_logits.shape[-1])
    s_top = torch.topk(student_logits.float().detach(), k=k, dim=-1).indices
    s_logp_full = _topk_log_probs_on_ids(student_logits, s_top)
    with torch.no_grad():
        t_logp_full = _topk_log_probs_on_ids(teacher_logits, s_top)
    if renorm_topk:
        s_logp = F.log_softmax(s_logp_full, dim=-1)
        s_w = s_logp.exp()
        # Teacher is renormalized over the same student support only for the KL term.
        # This avoids penalizing missing teacher mass twice; support coverage is handled by forward KL.
        t_logp = F.log_softmax(t_logp_full, dim=-1)
    else:
        s_logp = s_logp_full
        s_w = s_logp_full.exp()
        t_logp = t_logp_full
    per_tok = (s_w * (s_logp - t_logp)).sum(dim=-1)
    return per_tok.masked_fill(~mask, 0.0)


def _topk_overlap_ratio(student_logits: torch.Tensor, teacher_logits: torch.Tensor, k: int) -> torch.Tensor:
    """Student/teacher top-k overlap ratio per next-token position, [B,T]."""
    k = min(int(k), student_logits.shape[-1])
    with torch.no_grad():
        s_ids = torch.topk(student_logits.float(), k=k, dim=-1).indices
        t_ids = torch.topk(teacher_logits.float(), k=k, dim=-1).indices
        # For every student top-k id, check whether it appears in teacher top-k.
        hit = s_ids.unsqueeze(-1).eq(t_ids.unsqueeze(-2)).any(dim=-1).float()
        return hit.mean(dim=-1)


def _stage_average(values: torch.Tensor, mask: torch.Tensor, stage_size: int) -> torch.Tensor:
    """Replace each token value by the masked average value inside its stage/window."""
    if stage_size <= 1:
        return values
    out = values.clone()
    B, T = values.shape
    for start in range(0, T, stage_size):
        end = min(start + stage_size, T)
        m = mask[:, start:end]
        denom = m.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        avg = (values[:, start:end] * m.float()).sum(dim=1, keepdim=True) / denom
        out[:, start:end] = avg
    return out


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.float().sum().clamp_min(1.0)
    return (x * mask.float()).sum() / denom


def compute_adaptive_kl_loss(
    student_logits_raw: torch.Tensor,
    teacher_logits_raw: torch.Tensor,
    labels: torch.Tensor,
    cfg: dict[str, Any],
) -> AdaptiveKLLossOutput:
    """Compute OPD experimental losses for same-tokenizer teacher/student.

    Supported cfg['opd_loss_mode']:
      - reverse_kl
      - forward_kl
      - fixed_mixture
      - prune_opd_lite
      - adaptive_kl

    This assumes teacher and student share token ids. If they do not, use your
    original sampled_rkl backend, or map tokens before calling this function.
    """
    s_logits, _, mask = _shift_for_next_token(student_logits_raw, labels)
    t_logits, _, _ = _shift_for_next_token(teacher_logits_raw, labels)

    mode = str(_get(cfg, "opd_loss_mode", "reverse_kl"))
    renorm = bool(_get(cfg, "kl_renorm_topk", True))
    rev_k = int(_get(cfg, "reverse_top_k", _get(cfg, "kd_top_k", 16)))
    fwd_k = int(_get(cfg, "forward_top_k", _get(cfg, "kd_top_k", 16)))
    overlap_k = int(_get(cfg, "overlap_top_k", 16))

    rev = _student_reverse_kl_loss(s_logits, t_logits, mask, k=rev_k, renorm_topk=renorm)
    fwd = _teacher_forward_kl_loss(s_logits, t_logits, mask, k=fwd_k, renorm_topk=renorm)
    overlap = _topk_overlap_ratio(s_logits, t_logits, k=overlap_k).masked_fill(~mask, 0.0)

    logs: Dict[str, float] = {}
    logs["opd/mean_overlap"] = float(_masked_mean(overlap.detach(), mask).cpu())
    logs["opd/mask_tokens"] = float(mask.float().sum().detach().cpu())

    if mode == "reverse_kl":
        per_tok = rev
        logs["opd/reverse_fraction"] = 1.0
        logs["opd/forward_fraction"] = 0.0
        logs["opd/downweight_fraction"] = 0.0

    elif mode == "forward_kl":
        per_tok = fwd
        logs["opd/reverse_fraction"] = 0.0
        logs["opd/forward_fraction"] = 1.0
        logs["opd/downweight_fraction"] = 0.0

    elif mode == "fixed_mixture":
        alpha = float(_get(cfg, "mixture_forward_alpha", 0.5))
        per_tok = alpha * fwd + (1.0 - alpha) * rev
        logs["opd/mixture_forward_alpha"] = alpha
        logs["opd/reverse_fraction"] = 1.0 - alpha
        logs["opd/forward_fraction"] = alpha
        logs["opd/downweight_fraction"] = 0.0

    elif mode == "prune_opd_lite":
        threshold = float(_get(cfg, "prune_overlap_threshold", 0.7))
        w_drop = float(_get(cfg, "prune_w_drop", 0.01))
        w_base = float(_get(cfg, "prune_w_base", 0.5))
        cumulative = bool(_get(cfg, "prune_cumulative", True))
        base_mode = str(_get(cfg, "prune_base_loss", "reverse_kl"))
        bad = (overlap < threshold) & mask
        if cumulative:
            bad_count = torch.cumsum(bad.float(), dim=1)
            # Starts at 1 and decays with cumulative incompatibility; clipped by floor.
            weights = torch.clamp(1.0 - w_drop * bad_count, min=w_base, max=1.0)
        else:
            weights = torch.where(bad, torch.full_like(overlap, w_base), torch.ones_like(overlap))
        base_loss = fwd if base_mode == "forward_kl" else rev
        per_tok = weights * base_loss
        logs["opd/prune_bad_fraction"] = float(_masked_mean(bad.float(), mask).detach().cpu())
        logs["opd/prune_mean_weight"] = float(_masked_mean(weights.detach(), mask).cpu())
        logs["opd/reverse_fraction"] = 1.0 if base_mode != "forward_kl" else 0.0
        logs["opd/forward_fraction"] = 1.0 if base_mode == "forward_kl" else 0.0
        logs["opd/downweight_fraction"] = float(_masked_mean((weights < 0.999).float(), mask).detach().cpu())

    elif mode == "adaptive_kl":
        low = float(_get(cfg, "adaptive_low_threshold", 0.3))
        high = float(_get(cfg, "adaptive_high_threshold", 0.7))
        granularity = str(_get(cfg, "adaptive_granularity", "token"))  # token | stage
        stage_size = int(_get(cfg, "adaptive_stage_size", 50))
        low_action = str(_get(cfg, "adaptive_low_action", "downweight"))  # downweight | forward | skip | reverse
        low_weight = float(_get(cfg, "adaptive_low_weight", 0.2))
        medium_weight = float(_get(cfg, "adaptive_medium_weight", 1.0))
        high_weight = float(_get(cfg, "adaptive_high_weight", 1.0))

        gate_overlap = overlap
        if granularity == "stage":
            gate_overlap = _stage_average(overlap, mask, stage_size=stage_size)

        high_mask = (gate_overlap >= high) & mask
        mid_mask = (gate_overlap >= low) & (gate_overlap < high) & mask
        low_mask = (gate_overlap < low) & mask

        per_tok = torch.zeros_like(rev)
        # High compatibility: mode-seeking on-policy refinement.
        per_tok = per_tok + high_mask.float() * high_weight * rev
        # Medium compatibility: support repair with stronger/top-k forward KL.
        per_tok = per_tok + mid_mask.float() * medium_weight * fwd
        # Extremely low compatibility: configurable safety behavior.
        if low_action == "forward":
            per_tok = per_tok + low_mask.float() * low_weight * fwd
        elif low_action == "reverse":
            per_tok = per_tok + low_mask.float() * low_weight * rev
        elif low_action == "skip":
            per_tok = per_tok + low_mask.float() * 0.0
        else:  # downweight: conservative default, use forward repair but heavily downweighted.
            per_tok = per_tok + low_mask.float() * low_weight * fwd

        logs["opd/adaptive_low_threshold"] = low
        logs["opd/adaptive_high_threshold"] = high
        logs["opd/adaptive_high_fraction"] = float(_masked_mean(high_mask.float(), mask).detach().cpu())
        logs["opd/adaptive_mid_fraction"] = float(_masked_mean(mid_mask.float(), mask).detach().cpu())
        logs["opd/adaptive_low_fraction"] = float(_masked_mean(low_mask.float(), mask).detach().cpu())
        logs["opd/reverse_fraction"] = logs["opd/adaptive_high_fraction"]
        logs["opd/forward_fraction"] = logs["opd/adaptive_mid_fraction"] + (logs["opd/adaptive_low_fraction"] if low_action in {"forward", "downweight"} else 0.0)
        logs["opd/downweight_fraction"] = logs["opd/adaptive_low_fraction"] if low_action != "skip" else 0.0

    else:
        raise ValueError(f"Unknown opd_loss_mode={mode!r}")

    loss = _masked_mean(per_tok, mask)
    logs["opd/loss"] = float(loss.detach().cpu())
    logs["opd/mean_forward_loss"] = float(_masked_mean(fwd.detach(), mask).cpu())
    logs["opd/mean_reverse_loss"] = float(_masked_mean(rev.detach(), mask).cpu())
    return AdaptiveKLLossOutput(loss=loss, logs=logs)
