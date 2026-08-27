from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

try:
    import torch
except ImportError:  # Aggregation helpers remain usable in CPU-light environments.
    torch = None  # type: ignore[assignment]


DEFAULT_CATEGORY_PRIORITY = (
    "termination",
    "self_correction",
    "verification",
    "alternative_approach",
    "planning",
    "conclusion",
    "expansion",
    "structure",
    "code_tool",
)


def full_distribution_metrics(
    logits: torch.Tensor,
    *,
    log_z: torch.Tensor | None = None,
    targets: torch.Tensor | None = None,
    token_set: Iterable[int] = (),
    compute_entropy: bool = True,
    compute_ranks: bool = True,
    vocab_chunk_size: int = 8192,
) -> dict[str, torch.Tensor]:
    """Compute entropy/ranks without materialising a full softmax tensor.

    All quantities use exactly the vocabulary represented by ``logits``.  In
    cross-family OPD this should be the common, prefix-compatible vocabulary.
    """

    if torch is None:
        raise RuntimeError("PyTorch is required for logit-level diagnostics")
    if logits.ndim != 3:
        raise ValueError(f"Expected logits [B, S, V], got {list(logits.shape)}")
    vocab_size = int(logits.shape[-1])
    chunk_size = max(int(vocab_chunk_size), 1)
    normalizer = (
        torch.logsumexp(logits.float(), dim=-1)
        if log_z is None
        else log_z.detach().float().squeeze(-1)
    )
    result: dict[str, torch.Tensor] = {"log_z": normalizer}

    target_logits: torch.Tensor | None = None
    target_rank: torch.Tensor | None = None
    if targets is not None:
        if targets.shape != logits.shape[:2]:
            raise ValueError("targets must match logits [B, S]")
        safe_targets = targets.clamp(min=0, max=max(vocab_size - 1, 0))
        target_logits = logits.float().gather(
            -1, safe_targets.unsqueeze(-1)
        ).squeeze(-1)
        result["target_logp"] = target_logits - normalizer
        if compute_ranks:
            target_rank = torch.ones_like(target_logits, dtype=torch.long)

    valid_set = sorted(
        {int(token_id) for token_id in token_set if 0 <= int(token_id) < vocab_size}
    )
    best_set_logit: torch.Tensor | None = None
    set_rank: torch.Tensor | None = None
    if valid_set:
        index = torch.tensor(valid_set, device=logits.device, dtype=torch.long)
        set_logits = logits.float().index_select(-1, index)
        result["token_set_probability"] = torch.exp(
            set_logits - normalizer.unsqueeze(-1)
        ).sum(dim=-1)
        best_set_logit = set_logits.max(dim=-1).values
        if compute_ranks:
            set_rank = torch.ones_like(best_set_logit, dtype=torch.long)
    else:
        result["token_set_probability"] = torch.zeros_like(normalizer)

    expected_logit = torch.zeros_like(normalizer) if compute_entropy else None
    if compute_entropy or target_rank is not None or set_rank is not None:
        for start in range(0, vocab_size, chunk_size):
            block = logits[..., start : start + chunk_size].float()
            if expected_logit is not None:
                probability = torch.exp(block - normalizer.unsqueeze(-1))
                expected_logit += (probability * block).sum(dim=-1)
            if target_rank is not None and target_logits is not None:
                target_rank += block.gt(target_logits.unsqueeze(-1)).sum(dim=-1)
            if set_rank is not None and best_set_logit is not None:
                set_rank += block.gt(best_set_logit.unsqueeze(-1)).sum(dim=-1)

    if expected_logit is not None:
        result["entropy"] = normalizer - expected_logit
    if target_rank is not None:
        result["target_rank"] = target_rank
    if set_rank is not None:
        result["token_set_best_rank"] = set_rank
    else:
        result["token_set_best_rank"] = torch.full_like(
            normalizer, -1, dtype=torch.long
        )
    return result


def sparse_topk_logit_gradient_norm(
    diagnostics: Mapping[str, torch.Tensor],
    *,
    loss_normalization: str,
    sequence_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact per-token logit-gradient norm for the configured truncated KL.

    This is a cheap post-hoc proxy for gradient mass.  It is exact with respect
    to the selected Top-K logit supports, but it is *not* parameter-gradient
    attribution: parameter gradients mix tokens through shared weights and do
    not admit a unique additive category decomposition.
    """

    if torch is None:
        raise RuntimeError("PyTorch is required for logit-gradient diagnostics")
    reverse_student = diagnostics["reverse_support_student_local_prob"].float()
    reverse_teacher = diagnostics["reverse_support_teacher_local_prob"].float()
    reverse_kl = diagnostics["reverse_loss"].float().unsqueeze(-1)
    reverse_coef = diagnostics["reverse_coefficient"].float().unsqueeze(-1)
    reverse_gradient = reverse_coef * reverse_student * (
        reverse_student.clamp_min(1.0e-12).log()
        - reverse_teacher.clamp_min(1.0e-12).log()
        - reverse_kl
    )

    forward_student = diagnostics["forward_support_student_local_prob"].float()
    forward_teacher = diagnostics["forward_support_teacher_local_prob"].float()
    forward_coef = diagnostics["forward_coefficient"].float().unsqueeze(-1)
    forward_gradient = forward_coef * (forward_student - forward_teacher)

    support_ids = torch.cat(
        [diagnostics["reverse_support_ids"], diagnostics["forward_support_ids"]],
        dim=-1,
    )
    support_gradient = torch.cat(
        [reverse_gradient, forward_gradient], dim=-1
    )
    sorted_ids, order = torch.sort(support_ids, dim=-1)
    sorted_gradient = support_gradient.gather(-1, order)
    squared_norm = sorted_gradient.square().sum(dim=-1)
    # Each Top-K list has unique IDs, so a token occurs at most twice in the
    # concatenation.  Adjacent cross terms therefore combine the two routes.
    duplicate = sorted_ids[..., 1:].eq(sorted_ids[..., :-1])
    squared_norm += 2.0 * (
        sorted_gradient[..., 1:]
        * sorted_gradient[..., :-1]
        * duplicate.float()
    ).sum(dim=-1)
    raw_norm = squared_norm.clamp_min(0.0).sqrt()
    active = diagnostics["active"].bool()
    raw_norm = raw_norm.masked_fill(~active, 0.0)

    if str(loss_normalization) == "per_sequence":
        counts = active.sum(dim=1).clamp_min(1).float().unsqueeze(1)
        training_weighted = raw_norm / counts
    elif str(loss_normalization) == "per_token":
        # The global token-average constant cancels in a mass share.
        training_weighted = raw_norm
    else:
        raise ValueError(
            "loss_normalization must be 'per_sequence' or 'per_token'"
        )
    if sequence_weights is not None:
        training_weighted = training_weighted * sequence_weights.to(
            device=training_weighted.device,
            dtype=training_weighted.dtype,
        ).unsqueeze(1)
    return raw_norm, training_weighted


def _float_list(values: Sequence[Any] | torch.Tensor) -> list[float]:
    if torch is not None and isinstance(values, torch.Tensor):
        values = values.detach().float().cpu().tolist()
    return [float(value) for value in values]


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _sum(values: Sequence[float]) -> float:
    return float(math.fsum(values)) if values else 0.0


SUMMED_TOKEN_METRICS = (
    "configured_loss",
    "reverse_loss",
    "forward_loss",
    "signed_advantage",
    "absolute_advantage",
    "student_target_logp",
    "teacher_target_logp",
    "student_target_probability",
    "teacher_target_probability",
    "student_entropy",
    "teacher_entropy",
    "overlap",
    "student_topk_mass",
    "teacher_topk_mass",
    "student_topk_local_entropy",
    "teacher_topk_local_entropy",
    "student_eos_probability",
    "teacher_eos_probability",
    "student_target_rank",
    "teacher_target_rank",
    "student_eos_rank",
    "teacher_eos_rank",
    "logit_gradient_proxy",
    "training_weighted_logit_gradient_proxy",
    "target_in_student_topk",
    "target_in_teacher_topk",
)


def make_sample_diagnostic(
    *,
    metadata: Mapping[str, Any],
    category_labels: Sequence[str],
    marker_rows: Sequence[Mapping[str, Any]],
    token_metrics: Mapping[str, Sequence[Any] | torch.Tensor],
    probability_summary: Mapping[str, Any] | None,
    loss_normalization: str,
    sequence_loss_weight: float = 1.0,
    terminal_metrics: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build one compact JSONL record plus inspectable marker-start rows."""

    length = len(category_labels)
    values = {name: _float_list(token_metrics[name]) for name in SUMMED_TOKEN_METRICS}
    for name, metric_values in values.items():
        if len(metric_values) != length:
            raise ValueError(
                f"{name} has {len(metric_values)} values for a {length}-token completion"
            )
    normalization_weight = (
        1.0 / float(max(length, 1))
        if loss_normalization == "per_sequence"
        else 1.0
    )
    sequence_weight = float(sequence_loss_weight) * normalization_weight

    events_by_category: dict[str, int] = defaultdict(int)
    for marker in marker_rows:
        events_by_category[str(marker["category"])] += 1
    categories = sorted(set(category_labels) | set(events_by_category))
    category_records: dict[str, dict[str, Any]] = {}
    total_configured = _sum(values["configured_loss"])
    total_reverse = _sum(values["reverse_loss"])
    total_forward = _sum(values["forward_loss"])
    total_gradient = _sum(values["logit_gradient_proxy"])
    total_training_gradient = _sum(
        values["training_weighted_logit_gradient_proxy"]
    )

    for category in categories:
        positions = [
            index for index, label in enumerate(category_labels) if label == category
        ]
        count = len(positions)
        item: dict[str, Any] = {
            "span_token_count": count,
            "span_token_fraction": float(count) / float(max(length, 1)),
            "span_density_per_1k": 1000.0 * float(count) / float(max(length, 1)),
            "marker_event_count": int(events_by_category.get(category, 0)),
            "marker_density_per_1k": (
                1000.0 * float(events_by_category.get(category, 0)) / float(max(length, 1))
            ),
            "response_position_sum": float(sum(index + 1 for index in positions)),
            "relative_position_sum": float(
                sum((index + 1) / float(max(length, 1)) for index in positions)
            ),
        }
        for name in SUMMED_TOKEN_METRICS:
            selected = [values[name][index] for index in positions]
            item[f"{name}_sum"] = _sum(selected)
            item[f"mean_{name}"] = _mean(selected)
        signed = [values["signed_advantage"][index] for index in positions]
        positive = [value for value in signed if value > 0.0]
        negative = [value for value in signed if value < 0.0]
        item.update(
            {
                "positive_advantage_count": len(positive),
                "negative_advantage_count": len(negative),
                "positive_advantage_sum": _sum(positive),
                "negative_advantage_sum": _sum(negative),
                "positive_advantage_fraction": float(len(positive)) / float(max(count, 1)),
                "negative_advantage_fraction": float(len(negative)) / float(max(count, 1)),
                "configured_loss_training_mass_numerator": item[
                    "configured_loss_sum"
                ]
                * sequence_weight,
                "reverse_loss_training_mass_numerator": item["reverse_loss_sum"]
                * sequence_weight,
                "forward_loss_training_mass_numerator": item["forward_loss_sum"]
                * sequence_weight,
                "configured_loss_raw_mass_share": (
                    item["configured_loss_sum"] / total_configured
                    if total_configured > 0.0
                    else 0.0
                ),
                "reverse_loss_raw_mass_share": (
                    item["reverse_loss_sum"] / total_reverse
                    if total_reverse > 0.0
                    else 0.0
                ),
                "forward_loss_raw_mass_share": (
                    item["forward_loss_sum"] / total_forward
                    if total_forward > 0.0
                    else 0.0
                ),
                "logit_gradient_proxy_raw_mass_share": (
                    item["logit_gradient_proxy_sum"] / total_gradient
                    if total_gradient > 0.0
                    else 0.0
                ),
                "logit_gradient_proxy_training_mass_share": (
                    item["training_weighted_logit_gradient_proxy_sum"]
                    / total_training_gradient
                    if total_training_gradient > 0.0
                    else 0.0
                ),
            }
        )
        category_records[category] = item

    sequence: dict[str, Any] = {
        "active_token_count": length,
        "loss_normalization": str(loss_normalization),
        "sequence_loss_weight": float(sequence_loss_weight),
        "training_objective_denominator": (
            float(sequence_loss_weight)
            if loss_normalization == "per_sequence"
            else float(sequence_loss_weight) * float(length)
        ),
        "configured_loss_sum": total_configured,
        "configured_loss_train_normalized": total_configured * sequence_weight,
        "reverse_loss_sum": total_reverse,
        "reverse_loss_train_normalized": total_reverse * sequence_weight,
        "forward_loss_sum": total_forward,
        "forward_loss_train_normalized": total_forward * sequence_weight,
        "logit_gradient_proxy_sum": total_gradient,
        "training_weighted_logit_gradient_proxy_sum": total_training_gradient,
    }
    for name in SUMMED_TOKEN_METRICS:
        sequence[f"{name}_sum"] = _sum(values[name])
        sequence[f"mean_{name}"] = _mean(values[name])
    signed_all = values["signed_advantage"]
    sequence["positive_advantage_fraction"] = (
        float(sum(value > 0.0 for value in signed_all)) / float(max(length, 1))
    )
    region_indices = {
        "early": [index for index in range(length) if (index + 1) / max(length, 1) <= 1.0 / 3.0],
        "middle": [
            index
            for index in range(length)
            if 1.0 / 3.0 < (index + 1) / max(length, 1) <= 2.0 / 3.0
        ],
        "late": [index for index in range(length) if (index + 1) / max(length, 1) > 2.0 / 3.0],
    }
    for region, positions in region_indices.items():
        sequence[f"{region}_token_count"] = len(positions)
        for name in (
            "configured_loss",
            "signed_advantage",
            "student_entropy",
            "teacher_entropy",
            "student_eos_probability",
            "teacher_eos_probability",
            "overlap",
        ):
            selected = [values[name][index] for index in positions]
            sequence[f"{region}_{name}_sum"] = _sum(selected)
            sequence[f"{region}_mean_{name}"] = _mean(selected)
    sequence.update(dict(terminal_metrics))

    record = {
        "schema_version": 1,
        **dict(metadata),
        "sequence": sequence,
        "categories": category_records,
        "probability_sets": dict(probability_summary or {}),
    }

    marker_token_rows: list[dict[str, Any]] = []
    for marker in marker_rows:
        position = int(marker["response_position"])
        index = position - 1
        if not 0 <= index < length:
            continue
        marker_names = [str(value) for value in marker.get("markers", [])] or [""]
        for marker_name in marker_names:
            row = {
                key: metadata.get(key)
                for key in (
                    "view",
                    "scoring_checkpoint",
                    "scoring_checkpoint_step",
                    "source_checkpoint",
                    "sample_id",
                    "dataset_index",
                    "is_correct",
                    "rollout_length",
                    "fixed_prefix_cohort",
                )
            }
            row.update(
                {
                    "category": marker["category"],
                    "marker": marker_name,
                    "marker_group_size": len(marker_names),
                    "response_position": position,
                    "response_end_position": marker.get("response_end_position", position),
                    "relative_position": float(position) / float(max(length, 1)),
                }
            )
            for name in SUMMED_TOKEN_METRICS:
                row[name] = values[name][index]
            marker_token_rows.append(row)
    return record, marker_token_rows


def attach_correctness_transitions(records: Sequence[dict[str, Any]]) -> None:
    """Annotate on-policy rows relative to each sample's earliest checkpoint."""

    first: dict[str, tuple[int, bool]] = {}
    for record in records:
        if record.get("view") != "on_policy" or record.get("is_correct") is None:
            continue
        sample_id = str(record["sample_id"])
        step = int(record["scoring_checkpoint_step"])
        candidate = (step, bool(record["is_correct"]))
        if sample_id not in first or candidate[0] < first[sample_id][0]:
            first[sample_id] = candidate
    for record in records:
        if record.get("view") != "on_policy" or record.get("is_correct") is None:
            continue
        initial = first[str(record["sample_id"])][1]
        current = bool(record["is_correct"])
        record["correctness_transition"] = (
            ("correct" if initial else "wrong")
            + "_to_"
            + ("correct" if current else "wrong")
        )


def _subsets(record: Mapping[str, Any]) -> list[tuple[str, str]]:
    result = [("all", "all")]
    if bool(record.get("fixed_prefix_cohort", False)):
        result.append(("cohort", "fixed_prefix"))
    result.append(
        ("stop_status", "hit_horizon" if bool(record.get("hit_horizon", False)) else "completed")
    )
    if record.get("is_correct") is not None:
        result.append(
            ("correctness", "correct" if bool(record["is_correct"]) else "incorrect")
        )
    if record.get("correctness_transition"):
        result.append(("correctness_transition", str(record["correctness_transition"])))
    return result


def _group_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, int, str, str, str], list[Mapping[str, Any]]]:
    groups: dict[
        tuple[str, str, int, str, str, str], list[Mapping[str, Any]]
    ] = defaultdict(list)
    for record in records:
        for subset_type, subset in _subsets(record):
            key = (
                str(record["view"]),
                str(record["scoring_checkpoint"]),
                int(record["scoring_checkpoint_step"]),
                str(record.get("source_checkpoint", "")),
                subset_type,
                subset,
            )
            groups[key].append(record)
    return groups


def aggregate_checkpoint_rows(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, group in _group_records(records).items():
        view, checkpoint, step, source, subset_type, subset = key
        token_count = sum(int(item["sequence"]["active_token_count"]) for item in group)
        row: dict[str, Any] = {
            "view": view,
            "scoring_checkpoint": checkpoint,
            "scoring_checkpoint_step": step,
            "source_checkpoint": source,
            "subset_type": subset_type,
            "subset": subset,
            "sample_count": len(group),
            "token_count": token_count,
            "mean_rollout_length": _mean(
                [float(item.get("rollout_length", 0)) for item in group]
            ),
            "correct_fraction": _mean(
                [
                    float(bool(item["is_correct"]))
                    for item in group
                    if item.get("is_correct") is not None
                ]
            ),
            "eos_fraction": _mean(
                [float(bool(item.get("emitted_eos", False))) for item in group]
            ),
            "horizon_fraction": _mean(
                [float(bool(item.get("hit_horizon", False))) for item in group]
            ),
        }
        for name in SUMMED_TOKEN_METRICS:
            total = _sum([float(item["sequence"][f"{name}_sum"]) for item in group])
            row[f"mean_{name}"] = total / float(max(token_count, 1))
        for name in (
            "configured_loss_train_normalized",
            "reverse_loss_train_normalized",
            "forward_loss_train_normalized",
            "training_weighted_logit_gradient_proxy_sum",
        ):
            row[f"mean_{name}"] = _mean(
                [float(item["sequence"].get(name, 0.0)) for item in group]
            )
        weight_sum = _sum(
            [float(item["sequence"].get("sequence_loss_weight", 1.0)) for item in group]
        )
        row["sequence_loss_weight_sum"] = weight_sum
        training_denominator = _sum(
            [
                float(item["sequence"].get("training_objective_denominator", 0.0))
                for item in group
            ]
        )
        row["training_objective_denominator"] = training_denominator
        row["effective_sequence_fraction"] = _mean(
            [
                float(item["sequence"].get("sequence_loss_weight", 1.0) > 0.0)
                for item in group
            ]
        )
        for name in ("configured_loss", "reverse_loss", "forward_loss"):
            numerator = _sum(
                [
                    float(item["sequence"].get(f"{name}_train_normalized", 0.0))
                    for item in group
                ]
            )
            row[f"training_objective_{name}"] = (
                numerator / training_denominator
                if training_denominator > 0.0
                else 0.0
            )
        for region in ("early", "middle", "late"):
            region_count = sum(
                int(item["sequence"].get(f"{region}_token_count", 0))
                for item in group
            )
            row[f"{region}_token_count"] = region_count
            for name in (
                "configured_loss",
                "signed_advantage",
                "student_entropy",
                "teacher_entropy",
                "student_eos_probability",
                "teacher_eos_probability",
                "overlap",
            ):
                total = _sum(
                    [
                        float(item["sequence"].get(f"{region}_{name}_sum", 0.0))
                        for item in group
                    ]
                )
                row[f"{region}_mean_{name}"] = total / float(max(region_count, 1))
        terminal_keys = sorted(
            {
                metric
                for item in group
                for metric in item["sequence"]
                if str(metric).startswith("terminal_")
            }
        )
        for name in terminal_keys:
            values = [
                float(item["sequence"][name])
                for item in group
                if item["sequence"].get(name) is not None
            ]
            row[f"mean_{name}"] = _mean(values)
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            str(row["view"]),
            int(row["scoring_checkpoint_step"]),
            str(row["subset_type"]),
            str(row["subset"]),
        ),
    )


def aggregate_category_rows(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, group in _group_records(records).items():
        view, checkpoint, step, source, subset_type, subset = key
        categories = sorted(
            {category for item in group for category in item["categories"]}
        )
        total_tokens = sum(int(item["sequence"]["active_token_count"]) for item in group)
        denominators = {
            "configured_loss_raw": _sum(
                [float(item["sequence"]["configured_loss_sum"]) for item in group]
            ),
            "configured_loss_training": _sum(
                [
                    float(item["sequence"]["configured_loss_train_normalized"])
                    for item in group
                ]
            ),
            "reverse_loss_raw": _sum(
                [float(item["sequence"]["reverse_loss_sum"]) for item in group]
            ),
            "reverse_loss_training": _sum(
                [
                    float(item["sequence"]["reverse_loss_train_normalized"])
                    for item in group
                ]
            ),
            "forward_loss_raw": _sum(
                [float(item["sequence"]["forward_loss_sum"]) for item in group]
            ),
            "forward_loss_training": _sum(
                [
                    float(item["sequence"]["forward_loss_train_normalized"])
                    for item in group
                ]
            ),
            "gradient_raw": _sum(
                [float(item["sequence"]["logit_gradient_proxy_sum"]) for item in group]
            ),
            "gradient_training": _sum(
                [
                    float(item["sequence"]["training_weighted_logit_gradient_proxy_sum"])
                    for item in group
                ]
            ),
        }
        for category in categories:
            items = [
                item["categories"].get(category, {}) for item in group
            ]
            span_count = sum(int(item.get("span_token_count", 0)) for item in items)
            marker_count = sum(int(item.get("marker_event_count", 0)) for item in items)
            row: dict[str, Any] = {
                "view": view,
                "scoring_checkpoint": checkpoint,
                "scoring_checkpoint_step": step,
                "source_checkpoint": source,
                "subset_type": subset_type,
                "subset": subset,
                "category": category,
                "category_scope": "exclusive_marker_span",
                "sample_count": len(group),
                "total_token_count": total_tokens,
                "span_token_count": span_count,
                "span_token_fraction": float(span_count) / float(max(total_tokens, 1)),
                "span_density_per_1k": 1000.0 * float(span_count) / float(max(total_tokens, 1)),
                "marker_event_count": marker_count,
                "marker_density_per_1k": 1000.0 * float(marker_count) / float(max(total_tokens, 1)),
            }
            for name in SUMMED_TOKEN_METRICS:
                total = _sum([float(item.get(f"{name}_sum", 0.0)) for item in items])
                row[f"mean_{name}"] = total / float(max(span_count, 1))
                row[f"{name}_sum"] = total
            positive_count = sum(int(item.get("positive_advantage_count", 0)) for item in items)
            negative_count = sum(int(item.get("negative_advantage_count", 0)) for item in items)
            row["positive_advantage_fraction"] = float(positive_count) / float(max(span_count, 1))
            row["negative_advantage_fraction"] = float(negative_count) / float(max(span_count, 1))

            raw_configured = row["configured_loss_sum"]
            raw_reverse = row["reverse_loss_sum"]
            raw_forward = row["forward_loss_sum"]
            raw_gradient = row["logit_gradient_proxy_sum"]
            train_configured = _sum(
                [float(item.get("configured_loss_training_mass_numerator", 0.0)) for item in items]
            )
            train_reverse = _sum(
                [float(item.get("reverse_loss_training_mass_numerator", 0.0)) for item in items]
            )
            train_forward = _sum(
                [float(item.get("forward_loss_training_mass_numerator", 0.0)) for item in items]
            )
            train_gradient = row["training_weighted_logit_gradient_proxy_sum"]
            for name, numerator in (
                ("configured_loss_raw", raw_configured),
                ("configured_loss_training", train_configured),
                ("reverse_loss_raw", raw_reverse),
                ("reverse_loss_training", train_reverse),
                ("forward_loss_raw", raw_forward),
                ("forward_loss_training", train_forward),
                ("gradient_raw", raw_gradient),
                ("gradient_training", train_gradient),
            ):
                denominator = denominators[name]
                row[f"{name}_mass_share"] = (
                    float(numerator) / float(denominator)
                    if denominator > 0.0
                    else 0.0
                )
                row[f"{name}_enrichment_over_token_share"] = (
                    row[f"{name}_mass_share"] / row["span_token_fraction"]
                    if row["span_token_fraction"] > 0.0
                    else 0.0
                )
            rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            str(row["view"]),
            int(row["scoring_checkpoint_step"]),
            str(row["subset_type"]),
            str(row["subset"]),
            str(row["category"]),
        ),
    )


def aggregate_probability_rows(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, group in _group_records(records).items():
        view, checkpoint, step, source, subset_type, subset = key
        set_names = sorted(
            {
                set_name
                for item in group
                for set_name in item.get("probability_sets", {}).get("sets", {})
            }
        )
        for set_name in set_names:
            set_items = [
                item.get("probability_sets", {}).get("sets", {}).get(set_name, {})
                for item in group
            ]
            metric_names = sorted({name for item in set_items for name in item})
            row: dict[str, Any] = {
                "view": view,
                "scoring_checkpoint": checkpoint,
                "scoring_checkpoint_step": step,
                "source_checkpoint": source,
                "subset_type": subset_type,
                "subset": subset,
                "token_set": set_name,
                "sample_count": len(group),
            }
            for name in metric_names:
                row[name] = _mean(
                    [float(item.get(name, 0.0)) for item in set_items]
                )
            rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            str(row["view"]),
            int(row["scoring_checkpoint_step"]),
            str(row["subset_type"]),
            str(row["subset"]),
            str(row["token_set"]),
        ),
    )


def aggregate_marker_signal_rows(
    records: Sequence[Mapping[str, Any]],
    marker_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate individual marker starts without claiming additive mass.

    A position may match multiple nested phrases (for example ``check`` and
    ``check again``), so these rows are appropriate for per-marker means and
    densities, not for summing marker-level mass shares.
    """

    marker_index: dict[tuple[str, str, int, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in marker_rows:
        key = (
            str(row.get("view", "")),
            str(row.get("scoring_checkpoint", "")),
            int(row.get("scoring_checkpoint_step", 0)),
            str(row.get("source_checkpoint", "")),
            str(row.get("sample_id", "")),
        )
        marker_index[key].append(row)

    output: list[dict[str, Any]] = []
    for key, group in _group_records(records).items():
        view, checkpoint, step, source, subset_type, subset = key
        selected: list[Mapping[str, Any]] = []
        for record in group:
            selected.extend(
                marker_index.get(
                    (
                        view,
                        checkpoint,
                        step,
                        source,
                        str(record.get("sample_id", "")),
                    ),
                    [],
                )
            )
        grouped_markers: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in selected:
            grouped_markers[(str(row.get("category", "")), str(row.get("marker", "")))].append(row)
        total_tokens = sum(int(record["sequence"]["active_token_count"]) for record in group)
        sample_count = len(group)
        for (category, marker), items in grouped_markers.items():
            result: dict[str, Any] = {
                "view": view,
                "scoring_checkpoint": checkpoint,
                "scoring_checkpoint_step": step,
                "source_checkpoint": source,
                "subset_type": subset_type,
                "subset": subset,
                "category": category,
                "marker": marker,
                "sample_count": sample_count,
                "event_count": len(items),
                "document_fraction": float(len({str(item.get('sample_id', '')) for item in items}))
                / float(max(sample_count, 1)),
                "event_density_per_1k": 1000.0 * float(len(items)) / float(max(total_tokens, 1)),
                "mean_relative_position": _mean(
                    [float(item.get("relative_position", 0.0)) for item in items]
                ),
            }
            for name in SUMMED_TOKEN_METRICS:
                result[f"mean_{name}"] = _mean(
                    [float(item.get(name, 0.0)) for item in items]
                )
            signed = [float(item.get("signed_advantage", 0.0)) for item in items]
            result["positive_advantage_fraction"] = float(
                sum(value > 0.0 for value in signed)
            ) / float(max(len(signed), 1))
            output.append(result)
    return sorted(
        output,
        key=lambda row: (
            str(row["view"]),
            int(row["scoring_checkpoint_step"]),
            str(row["subset_type"]),
            str(row["subset"]),
            str(row["category"]),
            str(row["marker"]),
        ),
    )
