from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

import torch

from .behavior_markers import repetition_continuation_candidates


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    if selected.numel() == 0:
        return 0.0
    return float(selected.float().mean().detach().cpu().item())


def _masked_max(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    if selected.numel() == 0:
        return 0.0
    return float(selected.float().max().detach().cpu().item())


def _region_masks(active: torch.Tensor) -> dict[str, torch.Tensor]:
    response_position = torch.cumsum(active.long(), dim=1) - 1
    response_length = active.long().sum(dim=1).clamp_min(1).unsqueeze(1)
    relative = (response_position.float() + 1.0) / response_length.float()
    return {
        "early": active & (relative <= 1.0 / 3.0),
        "middle": active
        & (relative > 1.0 / 3.0)
        & (relative <= 2.0 / 3.0),
        "late": active & (relative > 2.0 / 3.0),
    }


def _target_in_set(
    targets: torch.Tensor | None,
    token_ids: Iterable[int],
) -> torch.Tensor | None:
    if targets is None:
        return None
    result = torch.zeros_like(targets, dtype=torch.bool)
    for token_id in token_ids:
        result |= targets.eq(int(token_id))
    return result


def _normalise_log_z(
    logits: torch.Tensor,
    log_z: torch.Tensor | None,
) -> torch.Tensor:
    if log_z is None:
        # Keeping logits in their native dtype avoids a second full-vocabulary
        # float32 tensor.  The result is promoted before subtraction below.
        return torch.logsumexp(logits.detach(), dim=-1).float()
    result = log_z.detach().float()
    if result.ndim == logits.ndim:
        result = result.squeeze(-1)
    return result


def _token_set_probability(
    logits: torch.Tensor,
    log_z: torch.Tensor,
    token_ids: Iterable[int],
) -> torch.Tensor:
    valid_ids = sorted(
        {
            int(token_id)
            for token_id in token_ids
            if 0 <= int(token_id) < int(logits.shape[-1])
        }
    )
    if not valid_ids:
        return torch.zeros(logits.shape[:-1], dtype=torch.float32, device=logits.device)
    index = torch.tensor(valid_ids, dtype=torch.long, device=logits.device)
    selected = logits.detach().index_select(-1, index).float()
    return torch.exp(selected - log_z.unsqueeze(-1)).sum(dim=-1)


def _probability_metrics(
    probability: torch.Tensor,
    active: torch.Tensor,
    regions: Mapping[str, torch.Tensor],
    *,
    target_mask: torch.Tensor | None,
    terminal_probability: torch.Tensor | None,
) -> dict[str, float]:
    metrics = {
        "mean": _masked_mean(probability, active),
        "max": _masked_max(probability, active),
        "early_mean": _masked_mean(probability, regions["early"]),
        "middle_mean": _masked_mean(probability, regions["middle"]),
        "late_mean": _masked_mean(probability, regions["late"]),
        "target_fraction": (
            _masked_mean(target_mask.float(), active)
            if target_mask is not None
            else 0.0
        ),
        "mean_at_emitted_start": (
            _masked_mean(probability, active & target_mask)
            if target_mask is not None
            else 0.0
        ),
        "terminal_mean": (
            float(terminal_probability.float().mean().detach().cpu().item())
            if terminal_probability is not None and terminal_probability.numel()
            else 0.0
        ),
    }
    return metrics


def _sample_probability_metrics(
    probability: torch.Tensor,
    active: torch.Tensor,
    regions: Mapping[str, torch.Tensor],
    target_mask: torch.Tensor | None,
    terminal_probability: torch.Tensor | None,
    sample_index: int,
) -> dict[str, float]:
    sample_target = target_mask[sample_index] if target_mask is not None else None
    sample_terminal = (
        terminal_probability[sample_index : sample_index + 1]
        if terminal_probability is not None
        else None
    )
    return _probability_metrics(
        probability[sample_index],
        active[sample_index],
        {
            name: mask[sample_index] for name, mask in regions.items()
        },
        target_mask=sample_target,
        terminal_probability=sample_terminal,
    )


def _repetition_probability_summary(
    logits: torch.Tensor,
    log_z: torch.Tensor,
    active: torch.Tensor,
    completion_ids: list[list[int]],
    ngram_size: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    batch_masses = torch.zeros_like(active, dtype=torch.float32)
    eligible_mask = torch.zeros_like(active, dtype=torch.bool)
    actual_mask = torch.zeros_like(active, dtype=torch.bool)

    for batch_index in range(int(active.shape[0])):
        active_positions = (
            torch.where(active[batch_index])[0].detach().cpu().tolist()
        )
        completion = [int(token) for token in completion_ids[batch_index]]
        candidates = repetition_continuation_candidates(completion, ngram_size)
        usable = min(len(completion), len(active_positions))
        grouped: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for response_index in range(usable):
            valid_candidates = tuple(
                token_id
                for token_id in candidates[response_index]
                if 0 <= int(token_id) < int(logits.shape[-1])
            )
            if not valid_candidates:
                continue
            sequence_position = int(active_positions[response_index])
            grouped[valid_candidates].append(sequence_position)
            eligible_mask[batch_index, sequence_position] = True
            if completion[response_index] in valid_candidates:
                actual_mask[batch_index, sequence_position] = True

        for candidate_ids, sequence_positions in grouped.items():
            position_index = torch.tensor(
                sequence_positions, dtype=torch.long, device=logits.device
            )
            candidate_index = torch.tensor(
                candidate_ids, dtype=torch.long, device=logits.device
            )
            selected = (
                logits[batch_index]
                .index_select(0, position_index)
                .index_select(-1, candidate_index)
                .detach()
                .float()
            )
            probability = torch.exp(
                selected
                - log_z[batch_index].index_select(0, position_index).unsqueeze(-1)
            ).sum(dim=-1)
            batch_masses[batch_index, position_index] = probability

    regions = _region_masks(active)
    aggregate = {
        "eligible_position_fraction": _masked_mean(
            eligible_mask.float(), active
        ),
        "actual_continuation_fraction": _masked_mean(actual_mask.float(), active),
        "actual_given_eligible_fraction": _masked_mean(
            actual_mask.float(), eligible_mask
        ),
        "mean_probability_at_eligible_positions": _masked_mean(
            batch_masses, eligible_mask
        ),
        "early_mean_probability": _masked_mean(
            batch_masses, eligible_mask & regions["early"]
        ),
        "middle_mean_probability": _masked_mean(
            batch_masses, eligible_mask & regions["middle"]
        ),
        "late_mean_probability": _masked_mean(
            batch_masses, eligible_mask & regions["late"]
        ),
    }
    samples: list[dict[str, float]] = []
    for batch_index in range(int(active.shape[0])):
        sample_active = active[batch_index]
        sample_eligible = eligible_mask[batch_index]
        sample_actual = actual_mask[batch_index]
        samples.append(
            {
                "eligible_position_fraction": _masked_mean(
                    sample_eligible.float(), sample_active
                ),
                "actual_continuation_fraction": _masked_mean(
                    sample_actual.float(), sample_active
                ),
                "actual_given_eligible_fraction": _masked_mean(
                    sample_actual.float(), sample_eligible
                ),
                "mean_probability_at_eligible_positions": _masked_mean(
                    batch_masses[batch_index], sample_eligible
                ),
                "early_mean_probability": _masked_mean(
                    batch_masses[batch_index],
                    sample_eligible & regions["early"][batch_index],
                ),
                "middle_mean_probability": _masked_mean(
                    batch_masses[batch_index],
                    sample_eligible & regions["middle"][batch_index],
                ),
                "late_mean_probability": _masked_mean(
                    batch_masses[batch_index],
                    sample_eligible & regions["late"][batch_index],
                ),
            }
        )
    return aggregate, samples


@torch.no_grad()
def summarize_next_token_probabilities(
    logits: torch.Tensor,
    active: torch.Tensor,
    token_sets: Mapping[str, Iterable[int]],
    *,
    log_z: torch.Tensor | None = None,
    targets: torch.Tensor | None = None,
    terminal_logits: torch.Tensor | None = None,
    completion_ids: list[list[int]] | None = None,
    repetition_ngram_size: int = 4,
) -> dict[str, Any]:
    """Summarize sparse marker probabilities without materializing softmax(V)."""

    if logits.ndim != 3:
        raise ValueError(f"Expected logits [B, S, V], got {list(logits.shape)}")
    if active.shape != logits.shape[:2]:
        raise ValueError(
            f"active shape {list(active.shape)} does not match logits {list(logits.shape)}"
        )
    active = active.bool()
    normalizer = _normalise_log_z(logits, log_z)
    terminal_normalizer = (
        _normalise_log_z(terminal_logits, None)
        if terminal_logits is not None
        else None
    )
    regions = _region_masks(active)
    batch_size = int(logits.shape[0])
    sample_records: list[dict[str, Any]] = [
        {"sets": {}} for _ in range(batch_size)
    ]
    set_records: dict[str, dict[str, float]] = {}

    for name, raw_ids in token_sets.items():
        token_ids = tuple(
            sorted(
                {
                    int(token_id)
                    for token_id in raw_ids
                    if 0 <= int(token_id) < int(logits.shape[-1])
                }
            )
        )
        if not token_ids:
            continue
        probability = _token_set_probability(logits, normalizer, token_ids)
        target_mask = _target_in_set(targets, token_ids)
        terminal_probability = (
            _token_set_probability(
                terminal_logits, terminal_normalizer, token_ids
            )
            if terminal_logits is not None and terminal_normalizer is not None
            else None
        )
        set_records[name] = _probability_metrics(
            probability,
            active,
            regions,
            target_mask=target_mask,
            terminal_probability=terminal_probability,
        )
        for batch_index in range(batch_size):
            sample_records[batch_index]["sets"][name] = (
                dict(set_records[name])
                if batch_size == 1
                else _sample_probability_metrics(
                    probability,
                    active,
                    regions,
                    target_mask,
                    terminal_probability,
                    batch_index,
                )
            )

    repetition_record: dict[str, float] | None = None
    if completion_ids is not None and len(completion_ids) == batch_size:
        repetition_record, repetition_samples = _repetition_probability_summary(
            logits,
            normalizer,
            active,
            completion_ids,
            repetition_ngram_size,
        )
        for sample, repetition in zip(
            sample_records, repetition_samples, strict=True
        ):
            sample["repetition_continuation"] = (
                dict(repetition_record)
                if batch_size == 1 and repetition_record is not None
                else repetition
            )

    return {
        "normalization": "full probability over the logits vocabulary",
        "sets": set_records,
        "repetition_continuation": repetition_record,
        "samples": sample_records,
    }


def flatten_probability_logs(
    summary: Mapping[str, Any],
    *,
    prefix: str,
) -> dict[str, float]:
    logs: dict[str, float] = {}
    for name, metrics in summary.get("sets", {}).items():
        slug = str(name).replace("/", "_")
        for metric in (
            "mean",
            "late_mean",
            "terminal_mean",
            "mean_at_emitted_start",
        ):
            logs[f"{prefix}/{slug}_{metric}"] = float(metrics.get(metric, 0.0))
    repetition = summary.get("repetition_continuation")
    if repetition:
        logs[f"{prefix}/repetition_eligible_position_fraction"] = float(
            repetition.get("eligible_position_fraction", 0.0)
        )
        logs[f"{prefix}/repetition_actual_continuation_fraction"] = float(
            repetition.get("actual_continuation_fraction", 0.0)
        )
        logs[f"{prefix}/repetition_mean_probability"] = float(
            repetition.get("mean_probability_at_eligible_positions", 0.0)
        )
        logs[f"{prefix}/repetition_late_mean_probability"] = float(
            repetition.get("late_mean_probability", 0.0)
        )
    return logs
