
import torch

def add_overlap_statistics(logs, overlap, mask, prefix="opd"):
    valid = overlap[mask]
    if valid.numel() == 0:
        return

    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        logs[f"{prefix}/overlap_p{int(q*100)}"] = float(
            torch.quantile(valid, q).detach().cpu()
        )

    for p in [0, 64, 128, 256, 512, 768, 1023]:
        if p < overlap.shape[1] and mask[:, p].any():
            logs[f"{prefix}/overlap_pos_{p}"] = float(
                overlap[:, p][mask[:, p]].mean().detach().cpu()
            )

    for name, s, e in [
        ("early", 0, 256),
        ("middle", 256, 768),
        ("late", 768, overlap.shape[1]),
    ]:
        if s < overlap.shape[1]:
            e = min(e, overlap.shape[1])
            m = mask[:, s:e]
            if m.any():
                logs[f"{prefix}/overlap_{name}"] = float(
                    overlap[:, s:e][m].mean().detach().cpu()
                )


def add_kl_statistics(logs, forward_loss, reverse_loss, mask, prefix="opd"):
    fwd = (forward_loss * mask.float()).sum() / mask.float().sum().clamp_min(1)
    rev = (reverse_loss * mask.float()).sum() / mask.float().sum().clamp_min(1)

    logs[f"{prefix}/mean_forward_loss"] = float(fwd.detach().cpu())
    logs[f"{prefix}/mean_reverse_loss"] = float(rev.detach().cpu())
    logs[f"{prefix}/forward_reverse_ratio"] = float(
        (fwd / (rev + 1e-8)).detach().cpu()
    )


def first_position(mask):
    idx = torch.where(mask)
    if len(idx[0]) == 0:
        return -1
    return int(idx[1].min().detach().cpu())


def add_prune_statistics(logs, weights, bad_mask, mask, prefix="opd"):
    valid = weights[mask]

    if valid.numel():
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            logs[f"{prefix}/prune_weight_p{int(q*100)}"] = float(
                torch.quantile(valid, q).detach().cpu()
            )

    logs[f"{prefix}/first_prune_position"] = first_position(bad_mask)
    logs[f"{prefix}/effective_response_length"] = float(
        (weights > 0).float().sum(dim=1).mean().detach().cpu()
    )
