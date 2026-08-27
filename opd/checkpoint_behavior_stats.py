from __future__ import annotations

import math
import re
import statistics
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from .behavior_markers import aggregate_occurrence_logs


_CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)$")


def checkpoint_step(name: str) -> int:
    match = _CHECKPOINT_PATTERN.search(str(name).rstrip("/"))
    if match is None:
        raise ValueError(f"Checkpoint name does not end in checkpoint-N: {name!r}")
    return int(match.group(1))


def percentile(values: Sequence[float | int], q: float) -> float:
    """Return a linearly interpolated percentile without requiring NumPy."""

    if not values:
        return 0.0
    if not 0.0 <= float(q) <= 1.0:
        raise ValueError("q must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = float(q) * float(len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - float(low)
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def linear_slope(xs: Sequence[float | int], ys: Sequence[float | int]) -> float:
    """Least-squares slope. A single point or constant x-axis has slope zero."""

    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    if len(xs) < 2:
        return 0.0
    x_values = [float(value) for value in xs]
    y_values = [float(value) for value in ys]
    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator == 0.0:
        return 0.0
    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values, strict=True)
    )
    return numerator / denominator


def _fraction(records: Sequence[Mapping[str, Any]], predicate: Any) -> float:
    if not records:
        return 0.0
    return float(sum(bool(predicate(record)) for record in records)) / float(
        len(records)
    )


def summarize_checkpoint_records(
    checkpoint: str,
    step: int,
    records: Sequence[Mapping[str, Any]],
    *,
    repetition_ngram_size: int = 4,
) -> dict[str, Any]:
    """Aggregate one checkpoint using names compatible with trainer logs."""

    lengths = [int(record.get("rollout_length", 0)) for record in records]
    raw_lengths = [
        int(record.get("raw_rollout_length", record.get("rollout_length", 0)))
        for record in records
    ]
    prompt_lengths = [int(record.get("prompt_length", 0)) for record in records]
    repeated = [
        float(record.get("raw_repeated_ngram_ratio", 0.0)) for record in records
    ]
    effective_repeated = [
        float(record.get("effective_repeated_ngram_ratio", 0.0))
        for record in records
    ]
    behavior_records = [
        record.get("student_behavior", {})
        for record in records
        if isinstance(record.get("student_behavior"), dict)
    ]

    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(step),
        "sample_count": len(records),
        "rollout/mean_generated_tokens": (
            statistics.fmean(lengths) if lengths else 0.0
        ),
        "rollout/raw_mean_generated_tokens": (
            statistics.fmean(raw_lengths) if raw_lengths else 0.0
        ),
        "rollout/median_generated_tokens": (
            statistics.median(lengths) if lengths else 0.0
        ),
        "rollout/min_generated_tokens": min(lengths) if lengths else 0,
        "rollout/max_generated_tokens": max(lengths) if lengths else 0,
        "rollout/p10_generated_tokens": percentile(lengths, 0.10),
        "rollout/p25_generated_tokens": percentile(lengths, 0.25),
        "rollout/p75_generated_tokens": percentile(lengths, 0.75),
        "rollout/p90_generated_tokens": percentile(lengths, 0.90),
        "rollout/p95_generated_tokens": percentile(lengths, 0.95),
        "rollout/std_generated_tokens": (
            statistics.pstdev(lengths) if len(lengths) > 1 else 0.0
        ),
        "rollout/mean_prompt_tokens": (
            statistics.fmean(prompt_lengths) if prompt_lengths else 0.0
        ),
        "rollout/eos_fraction": _fraction(
            records, lambda record: bool(record.get("emitted_eos", False))
        ),
        "rollout/truncated_fraction": _fraction(
            records, lambda record: bool(record.get("hit_horizon", False))
        ),
        "rollout/raw_truncated_fraction": _fraction(
            records, lambda record: bool(record.get("raw_hit_horizon", False))
        ),
        "rollout/boxed_answer_stop_fraction": _fraction(
            records,
            lambda record: record.get("stop_reason") == "boxed_answer",
        ),
        "rollout/boxed_truncation_fraction": _fraction(
            records, lambda record: bool(record.get("boxed_truncated", False))
        ),
        "rollout/appended_eos_fraction": _fraction(
            records, lambda record: bool(record.get("appended_eos", False))
        ),
        "rollout/student_eos_fraction": _fraction(
            records, lambda record: record.get("stop_reason") == "student_eos"
        ),
        "rollout/teacher_eos_fraction": _fraction(
            records, lambda record: record.get("stop_reason") == "teacher_eos"
        ),
        "rollout/configured_eos_fraction": _fraction(
            records,
            lambda record: record.get("stop_reason") == "configured_eos",
        ),
        "rollout/no_eos_fraction": _fraction(
            records, lambda record: not bool(record.get("emitted_eos", False))
        ),
        f"rollout/repeated_{int(repetition_ngram_size)}gram_ratio": (
            statistics.fmean(repeated) if repeated else 0.0
        ),
        f"rollout/effective_repeated_{int(repetition_ngram_size)}gram_ratio": (
            statistics.fmean(effective_repeated) if effective_repeated else 0.0
        ),
        "rollout/multi_boxed_fraction": _fraction(
            records, lambda record: int(record.get("raw_boxed_count", 0)) > 1
        ),
    }
    if behavior_records:
        summary.update(aggregate_occurrence_logs(behavior_records))
    return summary


def flatten_sample_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Make one rollout compact enough for a long-form CSV."""

    row: dict[str, Any] = {
        "checkpoint": record.get("checkpoint"),
        "checkpoint_step": record.get("checkpoint_step"),
        "sample_id": record.get("sample_id"),
        "dataset_index": record.get("dataset_index"),
        "prompt_length": record.get("prompt_length"),
        "rollout_length": record.get("rollout_length"),
        "raw_rollout_length": record.get("raw_rollout_length"),
        "stop_reason": record.get("stop_reason"),
        "emitted_eos": record.get("emitted_eos"),
        "hit_horizon": record.get("hit_horizon"),
        "raw_hit_horizon": record.get("raw_hit_horizon"),
        "boxed_truncated": record.get("boxed_truncated"),
        "appended_eos": record.get("appended_eos"),
        "raw_boxed_count": record.get("raw_boxed_count"),
        "effective_boxed_count": record.get("effective_boxed_count"),
        "raw_repeated_ngram_ratio": record.get("raw_repeated_ngram_ratio"),
        "effective_repeated_ngram_ratio": record.get(
            "effective_repeated_ngram_ratio"
        ),
    }
    behavior = record.get("student_behavior", {})
    for category, item in behavior.get("categories", {}).items():
        prefix = f"behavior/{category}"
        row[f"{prefix}/count"] = item.get("count", 0)
        row[f"{prefix}/document_hit"] = item.get("document_hit", False)
        row[f"{prefix}/first_token_position"] = item.get(
            "first_token_position"
        )
        row[f"{prefix}/first_relative_position"] = item.get(
            "first_relative_position"
        )
        token_count = max(int(behavior.get("token_count", 0)), 1)
        row[f"{prefix}/density_per_1k"] = (
            1000.0 * float(item.get("count", 0)) / float(token_count)
        )
    repetition = behavior.get("repetition_continuation", {})
    for key in (
        "eligible_position_fraction",
        "actual_continuation_fraction",
        "actual_given_eligible_fraction",
        "first_actual_continuation_position",
    ):
        row[f"behavior/repetition/{key}"] = repetition.get(key)
    return row


def marker_summary_rows(
    checkpoint_records: Mapping[str, Sequence[Mapping[str, Any]]],
    checkpoint_steps: Mapping[str, int],
    *,
    marker_universe: Iterable[tuple[str, str]] = (),
) -> list[dict[str, Any]]:
    """Aggregate individual dictionary phrases, including zero-hit rows."""

    markers = {(str(category), str(marker)) for category, marker in marker_universe}
    for records in checkpoint_records.values():
        for record in records:
            behavior = record.get("student_behavior", {})
            for category, category_item in behavior.get("categories", {}).items():
                for marker in category_item.get("matched_markers", {}):
                    markers.add((str(category), str(marker)))

    rows: list[dict[str, Any]] = []
    for checkpoint in sorted(
        checkpoint_records, key=lambda name: int(checkpoint_steps[name])
    ):
        records = checkpoint_records[checkpoint]
        denominator = max(len(records), 1)
        for category, marker in sorted(markers):
            counts: list[int] = []
            token_counts: list[int] = []
            relative_positions: list[float] = []
            for record in records:
                token_count = int(
                    record.get("student_behavior", {}).get(
                        "token_count", record.get("rollout_length", 0)
                    )
                )
                token_counts.append(max(token_count, 0))
                item = (
                    record.get("student_behavior", {})
                    .get("categories", {})
                    .get(category, {})
                    .get("matched_markers", {})
                    .get(marker)
                )
                if not item:
                    counts.append(0)
                    continue
                counts.append(int(item.get("count", 0)))
                first = item.get("first_token_position")
                if first is not None and token_count > 0:
                    relative_positions.append(float(first) / float(token_count))
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "checkpoint_step": int(checkpoint_steps[checkpoint]),
                    "category": category,
                    "marker": marker,
                    "document_fraction": float(sum(count > 0 for count in counts))
                    / float(denominator),
                    "mean_count": statistics.fmean(counts) if counts else 0.0,
                    "density_per_1k": (
                        1000.0 * float(sum(counts)) / float(sum(token_counts))
                        if sum(token_counts) > 0
                        else 0.0
                    ),
                    "mean_document_density_per_1k": (
                        statistics.fmean(
                            1000.0 * float(count) / float(max(token_count, 1))
                            for count, token_count in zip(
                                counts, token_counts, strict=True
                            )
                        )
                        if counts
                        else 0.0
                    ),
                    "mean_first_relative_position": (
                        statistics.fmean(relative_positions)
                        if relative_positions
                        else -1.0
                    ),
                }
            )
    return rows


def sample_length_outputs(
    checkpoint_records: Mapping[str, Sequence[Mapping[str, Any]]],
    checkpoint_steps: Mapping[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return wide trajectories and per-sample first/last/slope summaries."""

    checkpoints = sorted(
        checkpoint_records, key=lambda name: int(checkpoint_steps[name])
    )
    by_checkpoint = {
        checkpoint: {
            str(record["sample_id"]): record
            for record in checkpoint_records[checkpoint]
        }
        for checkpoint in checkpoints
    }
    sample_ids = sorted(
        {
            sample_id
            for records in by_checkpoint.values()
            for sample_id in records
        }
    )
    wide_rows: list[dict[str, Any]] = []
    trend_rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        available = [
            (checkpoint, by_checkpoint[checkpoint][sample_id])
            for checkpoint in checkpoints
            if sample_id in by_checkpoint[checkpoint]
        ]
        first_record = available[0][1]
        wide: dict[str, Any] = {
            "sample_id": sample_id,
            "dataset_index": first_record.get("dataset_index"),
        }
        for checkpoint in checkpoints:
            record = by_checkpoint[checkpoint].get(sample_id)
            wide[checkpoint] = (
                int(record.get("rollout_length", 0)) if record else None
            )
        wide_rows.append(wide)

        xs = [float(checkpoint_steps[name]) for name, _ in available]
        ys = [float(record.get("rollout_length", 0)) for _, record in available]
        first = ys[0]
        last = ys[-1]
        delta = last - first
        trend_rows.append(
            {
                "sample_id": sample_id,
                "dataset_index": first_record.get("dataset_index"),
                "checkpoint_count": len(available),
                "first_checkpoint": available[0][0],
                "last_checkpoint": available[-1][0],
                "first_length": first,
                "last_length": last,
                "delta_first_to_last": delta,
                "relative_change_first_to_last": (
                    delta / first if first != 0.0 else None
                ),
                "slope_tokens_per_step": linear_slope(xs, ys),
                "slope_tokens_per_25_steps": 25.0 * linear_slope(xs, ys),
                "min_length": min(ys),
                "max_length": max(ys),
            }
        )
    return wide_rows, trend_rows


def overall_trend_summary(
    checkpoint_summaries: Sequence[Mapping[str, Any]],
    sample_trends: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(
        checkpoint_summaries, key=lambda item: int(item["checkpoint_step"])
    )
    if not ordered:
        return {}
    xs = [float(item["checkpoint_step"]) for item in ordered]
    means = [float(item["rollout/mean_generated_tokens"]) for item in ordered]
    delta = means[-1] - means[0]
    sample_deltas = [
        float(item["delta_first_to_last"])
        for item in sample_trends
        if item.get("delta_first_to_last") is not None
    ]
    denominator = max(len(sample_deltas), 1)
    return {
        "first_checkpoint": ordered[0]["checkpoint"],
        "last_checkpoint": ordered[-1]["checkpoint"],
        "checkpoint_count": len(ordered),
        "first_mean_length": means[0],
        "last_mean_length": means[-1],
        "mean_length_delta_first_to_last": delta,
        "mean_length_relative_change_first_to_last": (
            delta / means[0] if means[0] != 0.0 else None
        ),
        "mean_length_slope_tokens_per_step": linear_slope(xs, means),
        "mean_length_slope_tokens_per_25_steps": 25.0
        * linear_slope(xs, means),
        "sample_median_delta_first_to_last": (
            statistics.median(sample_deltas) if sample_deltas else 0.0
        ),
        "sample_fraction_increased": float(
            sum(delta_value > 0.0 for delta_value in sample_deltas)
        )
        / float(denominator),
        "sample_fraction_decreased": float(
            sum(delta_value < 0.0 for delta_value in sample_deltas)
        )
        / float(denominator),
        "sample_fraction_unchanged": float(
            sum(delta_value == 0.0 for delta_value in sample_deltas)
        )
        / float(denominator),
    }
