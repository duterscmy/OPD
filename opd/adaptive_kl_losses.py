
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from .tokenizer_alignment import map_teacher_topk_to_student




def _use_cross_tokenizer(tokenizer_alignment):
    return (
        tokenizer_alignment is not None
        and hasattr(tokenizer_alignment, "teacher_to_student")
        and hasattr(tokenizer_alignment, "student_to_teacher")
    )


def _map_ids(ids, mapping):
    out = torch.full_like(ids, -1)
    for src, dst in mapping.items():
        out[ids == src] = dst
    return out


def _cross_tokenizer_forward_kl_topk(
    student_logits,
    teacher_logits,
    mask,
    k,
    tokenizer_alignment,
):
    """Forward KL with teacher top-k projected to student vocabulary."""
    k = min(k, teacher_logits.shape[-1])

    with torch.no_grad():
        teacher_ids = torch.topk(
            teacher_logits.float(), k=k, dim=-1
        ).indices

        student_ids = _map_ids(
            teacher_ids,
            tokenizer_alignment.teacher_to_student,
        )

        valid = student_ids.ge(0)
        student_ids = student_ids.clamp_min(0)

        teacher_logp = _selected_log_probs(
            teacher_logits,
            teacher_ids,
        )
        teacher_prob = F.softmax(
            teacher_logp,
            dim=-1,
        )

    student_logp = _selected_log_probs(
        student_logits,
        student_ids,
    )

    loss = -(teacher_prob * student_logp).sum(-1)
    loss = loss * valid.float().mean(-1)

    return loss.masked_fill(~mask, 0)


def _cross_tokenizer_reverse_kl_topk(
    student_logits,
    teacher_logits,
    mask,
    k,
    tokenizer_alignment,
):
    """Reverse KL with student top-k projected to teacher vocabulary."""
    k = min(k, student_logits.shape[-1])

    student_ids = torch.topk(
        student_logits.detach().float(),
        k=k,
        dim=-1,
    ).indices

    with torch.no_grad():
        teacher_ids = _map_ids(
            student_ids,
            tokenizer_alignment.student_to_teacher,
        )

        valid = teacher_ids.ge(0)
        teacher_ids = teacher_ids.clamp_min(0)

        teacher_logp = _selected_log_probs(
            teacher_logits,
            teacher_ids,
        )

    student_logp = _selected_log_probs(
        student_logits,
        student_ids,
    )

    student_logp = F.log_softmax(student_logp, dim=-1)
    teacher_logp = F.log_softmax(teacher_logp, dim=-1)

    student_prob = student_logp.exp()

    loss = (
        student_prob *
        (student_logp - teacher_logp)
    ).sum(-1)

    loss = loss * valid.float().mean(-1)

    return loss.masked_fill(~mask, 0)


@dataclass
class AdaptiveKLLossOutput:
    loss: torch.Tensor
    logs: Dict[str, float]


def _masked_mean(x, mask):
    denom = mask.float().sum().clamp_min(1.0)
    return (x * mask.float()).sum() / denom


def _shift_logits(logits, labels):
    logits = logits[:, :-1, :]
    target = labels[:, 1:].clamp_min(0)
    mask = labels[:, 1:].ne(-100)
    return logits, target, mask


def _selected_log_probs(logits, ids):
    selected = logits.gather(-1, ids)
    log_z = torch.logsumexp(logits.float(), dim=-1, keepdim=True)
    return selected.float() - log_z


def _forward_kl_topk(student_logits, teacher_logits, mask, k):
    k = min(k, teacher_logits.shape[-1])

    with torch.no_grad():
        teacher_ids = torch.topk(
            teacher_logits.float(),
            k=k,
            dim=-1
        ).indices

        teacher_logp = _selected_log_probs(
            teacher_logits,
            teacher_ids
        )

        teacher_prob = F.softmax(
            teacher_logp,
            dim=-1
        )

    student_logp = _selected_log_probs(
        student_logits,
        teacher_ids
    )

    loss = -(teacher_prob * student_logp).sum(-1)

    return loss.masked_fill(~mask, 0)


def _reverse_kl_topk(student_logits, teacher_logits, mask, k):

    k = min(k, student_logits.shape[-1])

    student_ids = torch.topk(
        student_logits.detach().float(),
        k=k,
        dim=-1
    ).indices

    student_logp = _selected_log_probs(
        student_logits,
        student_ids
    )

    with torch.no_grad():
        teacher_logp = _selected_log_probs(
            teacher_logits,
            student_ids
        )

    student_logp = F.log_softmax(
        student_logp,
        dim=-1
    )

    student_prob = student_logp.exp()

    teacher_logp = F.log_softmax(
        teacher_logp,
        dim=-1
    )

    loss = (
        student_prob *
        (student_logp - teacher_logp)
    ).sum(-1)

    return loss.masked_fill(~mask, 0)




def _topk_overlap_aligned(
    student_logits,
    teacher_logits,
    mask,
    k,
    tokenizer_alignment=None,
):
    """
    Cross-tokenizer top-k overlap.

    Teacher top-k token ids are mapped into student vocabulary
    before computing overlap.
    """

    k = min(k, student_logits.shape[-1])

    s_ids = torch.topk(
        student_logits.detach().float(),
        k=k,
        dim=-1
    ).indices

    t_ids = torch.topk(
        teacher_logits.detach().float(),
        k=k,
        dim=-1
    ).indices

    if tokenizer_alignment is not None:
        t_ids = map_teacher_topk_to_student(
            t_ids,
            tokenizer_alignment.teacher_to_student,
        )

    valid = t_ids.ge(0)

    overlap = (
        s_ids.unsqueeze(-1)
        .eq(t_ids.unsqueeze(-2))
        .any(-1)
        .float()
    )

    overlap = overlap * valid.any(-1).float()

    return overlap.mean(-1).masked_fill(~mask, 0)


def _topk_overlap(student_logits, teacher_logits, mask, k):

    k = min(k, student_logits.shape[-1])

    s_ids = torch.topk(
        student_logits.detach().float(),
        k=k,
        dim=-1
    ).indices

    t_ids = torch.topk(
        teacher_logits.detach().float(),
        k=k,
        dim=-1
    ).indices

    overlap = (
        s_ids.unsqueeze(-1)
        .eq(t_ids.unsqueeze(-2))
        .any(-1)
        .float()
        .mean(-1)
    )

    return overlap.masked_fill(~mask, 0)


def _add_overlap_logs(logs, overlap, mask):

    valid = overlap[mask]

    if valid.numel() == 0:
        return

    logs["opd/mean_overlap"] = float(
        valid.mean().detach().cpu()
    )

    for q in [0.1,0.25,0.5,0.75,0.9]:
        logs[
            f"opd/overlap_p{int(q*100)}"
        ] = float(
            torch.quantile(valid,q)
            .detach()
            .cpu()
        )

    for p in [
        0,64,128,256,
        512,768,1023
    ]:
        if p < overlap.shape[1]:
            if mask[:,p].any():
                logs[
                    f"opd/overlap_pos_{p}"
                ] = float(
                    overlap[:,p][mask[:,p]]
                    .mean()
                    .detach()
                    .cpu()
                )

    regions = {
        "early":(0,256),
        "middle":(256,768),
        "late":(768,overlap.shape[1])
    }

    for name,(s,e) in regions.items():

        if s >= overlap.shape[1]:
            continue

        e=min(e,overlap.shape[1])

        m=mask[:,s:e]

        if m.any():
            logs[
                f"opd/overlap_{name}"
            ] = float(
                overlap[:,s:e][m]
                .mean()
                .detach()
                .cpu()
            )


def _add_kl_logs(logs, fwd, rev, mask):

    f=_masked_mean(fwd,mask)
    r=_masked_mean(rev,mask)

    logs["opd/mean_forward_loss"]=float(
        f.detach().cpu()
    )

    logs["opd/mean_reverse_loss"]=float(
        r.detach().cpu()
    )

    logs["opd/forward_reverse_ratio"]=float(
        (f/(r+1e-8)).detach().cpu()
    )


def _first_position(mask):

    idx=torch.where(mask)

    if len(idx[0])==0:
        return -1

    return int(idx[1].min().cpu())


def compute_adaptive_kl_loss(
    student_logits_raw,
    teacher_logits_raw,
    labels,
    cfg:dict[str,Any],
    tokenizer_alignment=None,
):

    student_logits,_,mask = _shift_logits(
        student_logits_raw,
        labels
    )

    teacher_logits,_,_ = _shift_logits(
        teacher_logits_raw,
        labels
    )

    reverse_k=int(
        cfg.get("reverse_top_k",16)
    )

    forward_k=int(
        cfg.get("forward_top_k",64)
    )

    overlap_k=int(
        cfg.get("overlap_top_k",16)
    )


    if tokenizer_alignment is not None:

        reverse_loss = _cross_tokenizer_reverse_kl_topk(
            student_logits,
            teacher_logits,
            mask,
            k=reverse_k,
            tokenizer_alignment=tokenizer_alignment,
        )

        forward_loss = _cross_tokenizer_forward_kl_topk(
            student_logits,
            teacher_logits,
            mask,
            k=forward_k,
            tokenizer_alignment=tokenizer_alignment,
        )

    else:

        reverse_loss = _reverse_kl_topk(
            student_logits,
            teacher_logits,
            mask,
            k=reverse_k,
        )

        forward_loss = _forward_kl_topk(
            student_logits,
            teacher_logits,
            mask,
            k=forward_k,
        )


    # 这里暂时还没做对齐，但是因为tokenizer基本相同，所以先不管
    overlap=_topk_overlap(
        student_logits,
        teacher_logits,
        mask,
        overlap_k
    )


    logs={}

    _add_overlap_logs(
        logs,
        overlap.detach(),
        mask
    )

    _add_kl_logs(
        logs,
        forward_loss.detach(),
        reverse_loss.detach(),
        mask
    )


    mode=cfg.get(
        "opd_loss_mode",
        "reverse_kl"
    )


    if mode=="reverse_kl":

        loss=reverse_loss

        logs["opd/reverse_fraction"]=1.0
        logs["opd/forward_fraction"]=0.0


    elif mode=="forward_kl":

        loss=forward_loss

        logs["opd/reverse_fraction"]=0.0
        logs["opd/forward_fraction"]=1.0


    elif mode=="fixed_mixture":

        alpha=float(
            cfg.get(
                "mixture_forward_alpha",
                0.5
            )
        )

        loss=(
            alpha*forward_loss+
            (1-alpha)*reverse_loss
        )

        logs[
            "opd/mixture_forward_alpha"
        ]=alpha

        logs["opd/forward_fraction"]=alpha
        logs["opd/reverse_fraction"]=1-alpha


    elif mode=="prune_opd_lite":

        threshold=float(
            cfg.get(
                "prune_overlap_threshold",
                0.7
            )
        )

        w_drop=float(
            cfg.get(
                "prune_w_drop",
                0.01
            )
        )

        w_base=float(
            cfg.get(
                "prune_w_base",
                0.5
            )
        )

        bad=(overlap<threshold)&mask

        weights=torch.clamp(
            1-w_drop*torch.cumsum(
                bad.float(),
                dim=1
            ),
            min=w_base,
            max=1
        )

        loss=weights*reverse_loss

        logs[
            "opd/prune_bad_fraction"
        ]=float(
            _masked_mean(
                bad.float(),
                mask
            ).cpu()
        )

        logs[
            "opd/prune_mean_weight"
        ]=float(
            _masked_mean(
                weights,
                mask
            ).cpu()
        )

        logs[
            "opd/first_prune_position"
        ]=_first_position(bad)


    elif mode=="adaptive_kl":

        low=float(
            cfg.get(
                "adaptive_low_threshold",
                0.3
            )
        )

        high=float(
            cfg.get(
                "adaptive_high_threshold",
                0.7
            )
        )

        high_mask=(overlap>=high)&mask
        mid_mask=(
            (overlap>=low)&
            (overlap<high)&
            mask
        )

        low_mask=(overlap<low)&mask

        loss=(
            high_mask.float()*reverse_loss+
            mid_mask.float()*forward_loss
        )

        low_action=cfg.get(
            "adaptive_low_action",
            "downweight"
        )

        if low_action=="forward":
            loss += low_mask.float()*forward_loss

        elif low_action=="reverse":
            loss += low_mask.float()*reverse_loss

        elif low_action=="downweight":
            loss += (
                low_mask.float()*
                float(cfg.get(
                    "adaptive_low_weight",
                    0.2
                ))*
                forward_loss
            )

        logs["opd/adaptive_high_fraction"]=float(
            _masked_mean(
                high_mask.float(),
                mask
            ).cpu()
        )

        logs["opd/adaptive_mid_fraction"]=float(
            _masked_mean(
                mid_mask.float(),
                mask
            ).cpu()
        )

        logs["opd/adaptive_low_fraction"]=float(
            _masked_mean(
                low_mask.float(),
                mask
            ).cpu()
        )

        logs["opd/first_forward_position"]=_first_position(mid_mask)
        logs["opd/first_low_position"]=_first_position(low_mask)


    else:
        raise ValueError(
            f"Unknown mode {mode}"
        )


    final_loss=_masked_mean(
        loss,
        mask
    )

    logs["opd/loss"]=float(
        final_loss.detach().cpu()
    )

    return AdaptiveKLLossOutput(
        loss=final_loss,
        logs=logs
    )



def compute_cross_or_same_forward_kl(
    student_logits,
    teacher_logits,
    mask,
    k,
    tokenizer_alignment=None,
):
    if _use_cross_tokenizer(tokenizer_alignment):
        return _cross_tokenizer_forward_kl_topk(
            student_logits,
            teacher_logits,
            mask,
            k,
            tokenizer_alignment,
        )

    return _forward_kl_topk(
        student_logits,
        teacher_logits,
        mask,
        k,
    )


def compute_cross_or_same_reverse_kl(
    student_logits,
    teacher_logits,
    mask,
    k,
    tokenizer_alignment=None,
):
    if _use_cross_tokenizer(tokenizer_alignment):
        return _cross_tokenizer_reverse_kl_topk(
            student_logits,
            teacher_logits,
            mask,
            k,
            tokenizer_alignment,
        )

    return _reverse_kl_topk(
        student_logits,
        teacher_logits,
        mask,
        k,
    )
