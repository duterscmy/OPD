from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


ROUTE_NAMES = {
    0: "inactive",
    1: "reverse",
    2: "forward",
    3: "reverse+forward",
    4: "reverse_pruned",
    5: "pruned_reverse+forward",
}


@dataclass
class TopKOPDLossOutput:
    loss: torch.Tensor
    logs: dict[str, float]
    diagnostics: dict[str, torch.Tensor]


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def _masked_quantile(
    values: torch.Tensor,
    mask: torch.Tensor,
    quantile: float,
) -> float:
    selected = values.detach().float()[mask]
    if selected.numel() == 0:
        return 0.0
    return float(torch.quantile(selected, quantile).cpu().item())


def _masked_correlation(
    left: torch.Tensor,
    right: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    x = left.detach().float()[mask]
    y = right.detach().float()[mask]
    if x.numel() < 2:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = x.square().sum().sqrt() * y.square().sum().sqrt()
    if float(denom.cpu().item()) <= 1.0e-12:
        return 0.0
    return float(((x * y).sum() / denom).cpu().item())


def _crop_common_vocab(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    student_vocab = int(student_logits.shape[-1])
    teacher_vocab = int(teacher_logits.shape[-1])
    common_vocab = min(student_vocab, teacher_vocab)
    return (
        student_logits[..., :common_vocab],
        teacher_logits[..., :common_vocab],
        common_vocab,
        student_vocab,
        teacher_vocab,
    )


def _topk(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    k = max(1, min(int(k), int(logits.shape[-1])))
    result = torch.topk(logits.float(), k=k, dim=-1)
    return result.indices, result.values


def _truncated_reverse_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """KL(student || teacher) on Student Top-K, renormalizing both sides."""
    ids, student_values = _topk(student_logits.detach(), k)
    # Re-gather from the non-detached tensor so student gradients are preserved.
    student_values = student_logits.float().gather(-1, ids)
    teacher_values = teacher_logits.float().gather(-1, ids)
    student_logp = F.log_softmax(student_values, dim=-1)
    teacher_logp = F.log_softmax(teacher_values, dim=-1)
    student_prob = student_logp.exp()
    loss = (student_prob * (student_logp - teacher_logp)).sum(dim=-1)
    return loss, ids, student_prob


def _truncated_forward_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """KL(teacher || student) on Teacher Top-K, renormalizing both sides."""
    ids, teacher_values = _topk(teacher_logits.detach(), k)
    teacher_values = teacher_logits.float().gather(-1, ids)
    student_values = student_logits.float().gather(-1, ids)
    teacher_logp = F.log_softmax(teacher_values, dim=-1)
    student_logp = F.log_softmax(student_values, dim=-1)
    teacher_prob = teacher_logp.exp()
    loss = (teacher_prob * (teacher_logp - student_logp)).sum(dim=-1)
    return loss, ids, teacher_prob


def _topk_overlap(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    student_ids, _ = _topk(student_logits.detach(), k)
    teacher_ids, _ = _topk(teacher_logits.detach(), k)
    matches = student_ids.unsqueeze(-1).eq(teacher_ids.unsqueeze(-2)).any(dim=-1)
    overlap = matches.float().sum(dim=-1) / float(student_ids.shape[-1])
    return overlap, student_ids, teacher_ids


def _prune_weights(
    overlap: torch.Tensor,
    active: torch.Tensor,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    threshold = float(cfg.get("prune_overlap_threshold", 0.65))
    bad = active & (overlap < threshold)
    base = float(cfg.get("prune_w_base", 0.5))
    cumulative = bool(cfg.get("prune_cumulative", True))
    if cumulative:
        drop = float(cfg.get("prune_w_drop", 0.01))
        bad_count = torch.cumsum(bad.float(), dim=1)
        weights = torch.clamp(1.0 - drop * bad_count, min=base, max=1.0)
    else:
        bad_count = bad.float()
        weights = torch.where(bad, torch.full_like(overlap, base), torch.ones_like(overlap))
    weights = torch.where(active, weights, torch.zeros_like(weights))
    return weights, bad, bad_count


def _first_response_position(mask: torch.Tensor, active: torch.Tensor) -> float:
    if not bool(mask.any().item()):
        return -1.0
    response_pos = torch.cumsum(active.long(), dim=1) - 1
    return float(response_pos[mask].min().detach().cpu().item() + 1)


def _add_region_logs(
    logs: dict[str, float],
    name: str,
    values: torch.Tensor,
    active: torch.Tensor,
) -> None:
    response_pos = torch.cumsum(active.long(), dim=1) - 1
    response_len = active.long().sum(dim=1).clamp_min(1).unsqueeze(1)
    relative = (response_pos.float() + 1.0) / response_len.float()
    regions = {
        "early": active & (relative <= 1.0 / 3.0),
        "middle": active & (relative > 1.0 / 3.0) & (relative <= 2.0 / 3.0),
        "late": active & (relative > 2.0 / 3.0),
    }
    for region, region_mask in regions.items():
        logs[f"opd/{name}_{region}"] = float(
            _masked_mean(values.detach(), region_mask).cpu().item()
        )


def _full_distribution_debug(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    student_topk_ids: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Extra probability diagnostics, computed only when detailed logging is due."""
    student_log_z = torch.logsumexp(student_logits.float(), dim=-1, keepdim=True)
    teacher_log_z = torch.logsumexp(teacher_logits.float(), dim=-1, keepdim=True)
    student_topk_full_prob = (
        student_logits.float().gather(-1, student_topk_ids) - student_log_z
    ).exp()
    teacher_topk_full_prob = (
        teacher_logits.float().gather(-1, teacher_topk_ids) - teacher_log_z
    ).exp()
    targets = labels[:, 1:].clamp_min(0).unsqueeze(-1)
    student_target_logp = (
        student_logits.float().gather(-1, targets) - student_log_z
    ).squeeze(-1)
    teacher_target_logp = (
        teacher_logits.float().gather(-1, targets) - teacher_log_z
    ).squeeze(-1)
    return {
        "student_topk_full_prob": student_topk_full_prob,
        "teacher_topk_full_prob": teacher_topk_full_prob,
        "student_topk_mass": student_topk_full_prob.sum(dim=-1),
        "teacher_topk_mass": teacher_topk_full_prob.sum(dim=-1),
        "student_target_logp": student_target_logp,
        "teacher_target_logp": teacher_target_logp,
    }


def compute_topk_opd_loss(
    student_logits_raw: torch.Tensor,
    teacher_logits_raw: torch.Tensor,
    labels: torch.Tensor,
    cfg: dict[str, Any],
    collect_diagnostics: bool = False,
) -> TopKOPDLossOutput:
    """Compute all Top-K OPD variants through one explicit, testable route."""
    student_logits = student_logits_raw[:, :-1, :]
    teacher_logits = teacher_logits_raw[:, :-1, :]
    active = labels[:, 1:].ne(-100)
    (
        student_logits,
        teacher_logits,
        common_vocab,
        student_vocab,
        teacher_vocab,
    ) = _crop_common_vocab(student_logits, teacher_logits)

    reverse_k = int(cfg.get("reverse_top_k", 16))
    forward_k = int(cfg.get("forward_top_k", 16))
    overlap_k = int(cfg.get("overlap_top_k", 16))
    reverse_loss, reverse_ids, reverse_local_prob = _truncated_reverse_kl(
        student_logits, teacher_logits, reverse_k
    )
    forward_loss, forward_ids, forward_local_prob = _truncated_forward_kl(
        student_logits, teacher_logits, forward_k
    )
    overlap, student_overlap_ids, teacher_overlap_ids = _topk_overlap(
        student_logits, teacher_logits, overlap_k
    )
    reverse_loss = reverse_loss.masked_fill(~active, 0.0)
    forward_loss = forward_loss.masked_fill(~active, 0.0)
    overlap = overlap.masked_fill(~active, 0.0)

    mode = str(cfg.get("opd_loss_mode", "reverse_kl"))
    threshold = float(cfg.get("adaptive_overlap_threshold", 0.65))
    low_overlap = active & (overlap < threshold)
    reverse_coef = torch.zeros_like(reverse_loss)
    forward_coef = torch.zeros_like(forward_loss)
    prune_weight = torch.ones_like(reverse_loss).masked_fill(~active, 0.0)
    bad_count = torch.zeros_like(reverse_loss)
    route = torch.zeros_like(labels[:, 1:], dtype=torch.long)

    if mode == "reverse_kl":
        reverse_coef = active.float()
        route = torch.where(active, torch.ones_like(route), route)
    elif mode == "forward_kl":
        forward_coef = active.float()
        route = torch.where(active, torch.full_like(route, 2), route)
    elif mode == "fixed_mixture":
        alpha = float(cfg.get("mixture_forward_alpha", 0.5))
        reverse_coef = active.float() * (1.0 - alpha)
        forward_coef = active.float() * alpha
        route = torch.where(active, torch.full_like(route, 3), route)
    elif mode == "prune_opd":
        prune_weight, low_overlap, bad_count = _prune_weights(overlap, active, cfg)
        reverse_coef = prune_weight
        route = torch.where(active, torch.ones_like(route), route)
        route = torch.where(active & prune_weight.lt(1.0), torch.full_like(route, 4), route)
    elif mode == "adaptive_v1":
        reverse_coef = (active & ~low_overlap).float()
        forward_coef = low_overlap.float() * float(cfg.get("adaptive_forward_lambda", 1.0))
        route = torch.where(active & ~low_overlap, torch.ones_like(route), route)
        route = torch.where(low_overlap, torch.full_like(route, 2), route)
    elif mode == "adaptive_v2":
        reverse_coef = active.float()
        forward_coef = low_overlap.float() * float(cfg.get("adaptive_forward_lambda", 0.1))
        route = torch.where(active & ~low_overlap, torch.ones_like(route), route)
        route = torch.where(low_overlap, torch.full_like(route, 3), route)
    elif mode == "prune_plus_forward":
        prune_weight, low_overlap, bad_count = _prune_weights(overlap, active, cfg)
        reverse_coef = prune_weight
        forward_coef = low_overlap.float() * float(cfg.get("adaptive_forward_lambda", 0.1))
        route = torch.where(active, torch.ones_like(route), route)
        route = torch.where(active & prune_weight.lt(1.0), torch.full_like(route, 4), route)
        route = torch.where(low_overlap, torch.full_like(route, 5), route)
    else:
        raise ValueError(f"Unsupported opd_loss_mode={mode!r}")

    token_loss = reverse_coef * reverse_loss + forward_coef * forward_loss
    final_loss = _masked_mean(token_loss, active)

    logs: dict[str, float] = {
        "opd/loss": float(final_loss.detach().cpu().item()),
        "opd/mean_reverse_loss": float(_masked_mean(reverse_loss.detach(), active).cpu().item()),
        "opd/mean_forward_loss": float(_masked_mean(forward_loss.detach(), active).cpu().item()),
        "opd/mean_token_loss": float(_masked_mean(token_loss.detach(), active).cpu().item()),
        "opd/mean_overlap": float(_masked_mean(overlap.detach(), active).cpu().item()),
        "opd/low_overlap_fraction": float(_masked_mean(low_overlap.float(), active).cpu().item()),
        "opd/reverse_active_fraction": float(_masked_mean(reverse_coef.gt(0).float(), active).cpu().item()),
        "opd/forward_active_fraction": float(_masked_mean(forward_coef.gt(0).float(), active).cpu().item()),
        "opd/reverse_plus_forward_fraction": float(
            _masked_mean((reverse_coef.gt(0) & forward_coef.gt(0)).float(), active).cpu().item()
        ),
        "opd/mean_reverse_coefficient": float(_masked_mean(reverse_coef.detach(), active).cpu().item()),
        "opd/mean_forward_coefficient": float(_masked_mean(forward_coef.detach(), active).cpu().item()),
        "opd/mean_prune_weight": float(_masked_mean(prune_weight.detach(), active).cpu().item()),
        "opd/first_low_overlap_position": _first_response_position(low_overlap, active),
        "opd/common_vocab_size": float(common_vocab),
        "opd/student_vocab_size": float(student_vocab),
        "opd/teacher_vocab_size": float(teacher_vocab),
        "opd/reverse_top_k": float(reverse_k),
        "opd/forward_top_k": float(forward_k),
        "opd/overlap_top_k": float(overlap_k),
        "opd/overlap_reverse_loss_corr": _masked_correlation(overlap, reverse_loss, active),
        "opd/overlap_forward_loss_corr": _masked_correlation(overlap, forward_loss, active),
        "opd/overlap_total_loss_corr": _masked_correlation(overlap, token_loss, active),
    }
    for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.99):
        suffix = int(q * 100)
        logs[f"opd/overlap_p{suffix}"] = _masked_quantile(overlap, active, q)
        logs[f"opd/reverse_loss_p{suffix}"] = _masked_quantile(reverse_loss, active, q)
        logs[f"opd/forward_loss_p{suffix}"] = _masked_quantile(forward_loss, active, q)
        logs[f"opd/token_loss_p{suffix}"] = _masked_quantile(token_loss, active, q)
    _add_region_logs(logs, "overlap", overlap, active)
    _add_region_logs(logs, "reverse_loss", reverse_loss, active)
    _add_region_logs(logs, "forward_loss", forward_loss, active)
    _add_region_logs(logs, "token_loss", token_loss, active)

    diagnostics: dict[str, torch.Tensor] = {
        "active": active,
        "targets": labels[:, 1:],
        "overlap": overlap,
        "reverse_loss": reverse_loss,
        "forward_loss": forward_loss,
        "token_loss": token_loss,
        "reverse_coefficient": reverse_coef,
        "forward_coefficient": forward_coef,
        "prune_weight": prune_weight,
        "cumulative_low_overlap_count": bad_count,
        "route_code": route,
        "student_topk_ids": student_overlap_ids,
        "teacher_topk_ids": teacher_overlap_ids,
        "student_topk_local_prob": F.softmax(
            student_logits.float().gather(-1, student_overlap_ids), dim=-1
        ),
        "teacher_topk_local_prob": F.softmax(
            teacher_logits.float().gather(-1, teacher_overlap_ids), dim=-1
        ),
        "reverse_support_ids": reverse_ids,
        "reverse_support_student_local_prob": reverse_local_prob,
        "forward_support_ids": forward_ids,
        "forward_support_teacher_local_prob": forward_local_prob,
    }
    diagnostics["student_topk_local_entropy"] = -(
        diagnostics["student_topk_local_prob"]
        * diagnostics["student_topk_local_prob"].clamp_min(1.0e-12).log()
    ).sum(dim=-1)
    diagnostics["teacher_topk_local_entropy"] = -(
        diagnostics["teacher_topk_local_prob"]
        * diagnostics["teacher_topk_local_prob"].clamp_min(1.0e-12).log()
    ).sum(dim=-1)
    logs["opd/mean_student_topk_local_entropy"] = float(
        _masked_mean(diagnostics["student_topk_local_entropy"], active).detach().cpu().item()
    )
    logs["opd/mean_teacher_topk_local_entropy"] = float(
        _masked_mean(diagnostics["teacher_topk_local_entropy"], active).detach().cpu().item()
    )
    if collect_diagnostics:
        diagnostics.update(
            _full_distribution_debug(
                student_logits,
                teacher_logits,
                labels,
                student_overlap_ids,
                teacher_overlap_ids,
            )
        )
        logs["opd/mean_student_topk_mass"] = float(
            _masked_mean(diagnostics["student_topk_mass"], active).detach().cpu().item()
        )
        logs["opd/mean_teacher_topk_mass"] = float(
            _masked_mean(diagnostics["teacher_topk_mass"], active).detach().cpu().item()
        )
        top1_agree = student_overlap_ids[..., 0].eq(teacher_overlap_ids[..., 0])
        logs["opd/top1_agreement_fraction"] = float(
            _masked_mean(top1_agree.float(), active).detach().cpu().item()
        )

    return TopKOPDLossOutput(final_loss, logs, diagnostics)
