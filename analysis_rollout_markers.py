#!/usr/bin/env python3
"""Analyze termination, conclusion, expansion, and repetition markers in OPD rollouts.

This script intentionally analyzes decoded ``student_rollout`` text for phrases.
A tokenizer is not required for occurrence statistics.  Token IDs already stored
in ``tokens[].target_token_id`` are used for repeated n-gram continuation stats.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_EVENT = "topk_opd_sample_debug"


@dataclass(frozen=True)
class TextMarker:
    category: str
    name: str
    pattern: re.Pattern[str] | None = None
    metadata_eos: bool = False
    complete_boxed: bool = False


TEXT_MARKERS: tuple[TextMarker, ...] = (
    TextMarker("termination", "eos", metadata_eos=True),
    TextMarker("termination", "boxed", complete_boxed=True),
    TextMarker(
        "termination",
        "final_answer",
        re.compile(r"\bfinal\s+answer\b", flags=re.IGNORECASE),
    ),
    TextMarker(
        "conclusion_transition",
        "finally",
        re.compile(r"\bfinally\b", flags=re.IGNORECASE),
    ),
    TextMarker(
        "conclusion_transition",
        "therefore",
        re.compile(r"\btherefore\b", flags=re.IGNORECASE),
    ),
    TextMarker(
        "conclusion_transition",
        "thus",
        re.compile(r"\bthus\b", flags=re.IGNORECASE),
    ),
    TextMarker(
        "conclusion_transition",
        "hence",
        re.compile(r"\bhence\b", flags=re.IGNORECASE),
    ),
    TextMarker(
        "expansion",
        "additionally",
        re.compile(r"\badditionally\b", flags=re.IGNORECASE),
    ),
    TextMarker(
        "expansion",
        "moreover",
        re.compile(r"\bmoreover\b", flags=re.IGNORECASE),
    ),
    TextMarker(
        "expansion",
        "next",
        re.compile(r"\bnext\b", flags=re.IGNORECASE),
    ),
    TextMarker(
        "expansion",
        "lets",
        re.compile(r"\blet(?:['\u2018\u2019])s\b", flags=re.IGNORECASE),
    ),
    TextMarker(
        "expansion",
        "step",
        re.compile(r"\bsteps?\b", flags=re.IGNORECASE),
    ),
    TextMarker("expansion", "heading_###", re.compile(r"###")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze phrase categories and repeated-token continuations in OPD "
            "token_debug JSONL rollouts."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="One or more JSONL files, directories, or glob patterns.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--event",
        default=DEFAULT_EVENT,
        help=f"JSONL event to analyze (default: {DEFAULT_EVENT}).",
    )
    parser.add_argument(
        "--step-bin-size",
        type=int,
        default=10,
        help="Number of optimizer steps per trend bin (default: 10; use 1 for exact steps).",
    )
    parser.add_argument(
        "--repetition-ngram-size",
        type=int,
        default=4,
        help="Token n-gram size used for repetition continuation (default: 4).",
    )
    parser.add_argument(
        "--strict-json",
        action="store_true",
        help="Fail on malformed JSON lines instead of skipping them with a warning.",
    )
    return parser.parse_args()


def resolve_inputs(specs: Iterable[str]) -> list[Path]:
    paths: set[Path] = set()
    for spec in specs:
        candidate = Path(spec).expanduser()
        if candidate.is_file():
            paths.add(candidate.resolve())
            continue
        if candidate.is_dir():
            paths.update(path.resolve() for path in candidate.rglob("*.jsonl"))
            continue
        for match in glob.glob(spec, recursive=True):
            path = Path(match).expanduser()
            if path.is_file():
                paths.add(path.resolve())
            elif path.is_dir():
                paths.update(item.resolve() for item in path.rglob("*.jsonl"))
    return sorted(paths)


def complete_nonempty_boxed_starts(text: str) -> list[int]:
    """Return starts of complete, non-empty ``\\boxed{...}`` expressions."""
    marker = "\\boxed{"
    starts: list[int] = []
    cursor = 0
    while cursor < len(text):
        start = text.find(marker, cursor)
        if start < 0:
            break
        opening_brace = start + len(marker) - 1
        depth = 0
        closing_brace: int | None = None
        for index in range(opening_brace, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    closing_brace = index + 1
                    break
        if closing_brace is None:
            cursor = start + len(marker)
            continue
        payload = text[opening_brace + 1 : closing_brace - 1].strip()
        if payload:
            starts.append(start)
        cursor = closing_brace
    return starts


def real_eos_emitted(record: dict[str, Any]) -> bool:
    if bool(record.get("student_emitted_eos", False)):
        return True
    return str(record.get("student_stop_reason", "")) in {
        "student_eos",
        "teacher_eos",
        "configured_eos",
    }


def extract_rollout_token_ids(record: dict[str, Any]) -> list[int]:
    token_rows = record.get("tokens", [])
    if not isinstance(token_rows, list):
        return []
    ordered: list[tuple[int, int]] = []
    for fallback_position, row in enumerate(token_rows, start=1):
        if not isinstance(row, dict) or row.get("target_token_id") is None:
            continue
        try:
            position = int(row.get("response_position", fallback_position))
            token_id = int(row["target_token_id"])
        except (TypeError, ValueError):
            continue
        ordered.append((position, token_id))
    ordered.sort(key=lambda item: item[0])
    return [token_id for _, token_id in ordered]


def repeated_ngram_continuations(
    token_ids: Iterable[int],
    n: int,
) -> dict[str, Any]:
    """Count tokens that complete an n-gram already observed earlier."""
    ids = [int(token_id) for token_id in token_ids]
    n = max(int(n), 1)
    total = max(len(ids) - n + 1, 0)
    seen: set[tuple[int, ...]] = set()
    count = 0
    first_token_position: int | None = None
    for start in range(total):
        ngram = tuple(ids[start : start + n])
        if ngram in seen:
            count += 1
            if first_token_position is None:
                # One-indexed response position of the token completing the n-gram.
                first_token_position = start + n
        else:
            seen.add(ngram)
    return {
        "count": count,
        "total_ngram_opportunities": total,
        "rate": count / total if total else 0.0,
        "first_token_position": first_token_position,
    }


def empty_marker_result() -> dict[str, Any]:
    return {
        "count": 0,
        "present": False,
        "first_char_position": None,
        "first_char_fraction": None,
        "first_token_position": None,
        "first_token_fraction": None,
    }


def analyze_text_marker(
    marker: TextMarker,
    text: str,
    record: dict[str, Any],
    rollout_tokens: int,
) -> dict[str, Any]:
    result = empty_marker_result()
    starts: list[int]
    if marker.metadata_eos:
        if not real_eos_emitted(record):
            return result
        result.update(
            count=1,
            present=True,
            first_char_position=len(text),
            first_char_fraction=1.0,
            first_token_position=rollout_tokens if rollout_tokens > 0 else None,
            first_token_fraction=1.0 if rollout_tokens > 0 else None,
        )
        return result
    if marker.complete_boxed:
        starts = complete_nonempty_boxed_starts(text)
    else:
        assert marker.pattern is not None
        starts = [match.start() for match in marker.pattern.finditer(text)]
    if not starts:
        return result
    first = starts[0]
    result.update(
        count=len(starts),
        present=True,
        first_char_position=first,
        first_char_fraction=first / max(len(text), 1),
    )
    return result


def analyze_record(
    record: dict[str, Any],
    *,
    source_file: str,
    line_number: int,
    repetition_ngram_size: int,
) -> dict[str, Any]:
    text = str(record.get("student_rollout", ""))
    token_ids = extract_rollout_token_ids(record)
    rollout_tokens = int(
        record.get("student_rollout_length", len(token_ids)) or len(token_ids)
    )
    markers: dict[tuple[str, str], dict[str, Any]] = {}
    for marker in TEXT_MARKERS:
        markers[(marker.category, marker.name)] = analyze_text_marker(
            marker,
            text,
            record,
            rollout_tokens,
        )

    repetition = repeated_ngram_continuations(token_ids, repetition_ngram_size)
    repeat_result = empty_marker_result()
    repeat_result.update(
        count=int(repetition["count"]),
        present=bool(repetition["count"]),
        first_token_position=repetition["first_token_position"],
        first_token_fraction=(
            float(repetition["first_token_position"]) / rollout_tokens
            if repetition["first_token_position"] is not None and rollout_tokens > 0
            else None
        ),
    )
    repetition_name = f"repeated_{max(int(repetition_ngram_size), 1)}gram"
    markers[("repetition_continuation", repetition_name)] = repeat_result

    categories: dict[str, dict[str, Any]] = {}
    category_names = sorted({category for category, _ in markers})
    for category in category_names:
        selected = [
            value for (marker_category, _), value in markers.items()
            if marker_category == category
        ]
        first_chars = [
            int(value["first_char_position"])
            for value in selected
            if value["first_char_position"] is not None
        ]
        first_tokens = [
            int(value["first_token_position"])
            for value in selected
            if value["first_token_position"] is not None
        ]
        first_char = min(first_chars) if first_chars else None
        first_token = min(first_tokens) if first_tokens else None
        categories[category] = {
            "count": sum(int(value["count"]) for value in selected),
            "present": any(bool(value["present"]) for value in selected),
            "first_char_position": first_char,
            "first_char_fraction": (
                first_char / max(len(text), 1) if first_char is not None else None
            ),
            "first_token_position": first_token,
            "first_token_fraction": (
                first_token / rollout_tokens
                if first_token is not None and rollout_tokens > 0
                else None
            ),
        }

    return {
        "source_file": source_file,
        "line_number": line_number,
        "global_step": int(record.get("global_step", -1)),
        "loss_call_index": int(record.get("loss_call_index", -1)),
        "rank": int(record.get("rank", -1)),
        "sample_index_in_batch": int(record.get("sample_index_in_batch", -1)),
        "loss_mode": str(record.get("loss_mode", "")),
        "strategy": str(record.get("strategy", "")),
        "rollout_tokens": rollout_tokens,
        "rollout_characters": len(text),
        "logged_token_count": len(token_ids),
        "token_sequence_complete": len(token_ids) >= rollout_tokens,
        "stop_reason": str(record.get("student_stop_reason", "")),
        "emitted_eos": real_eos_emitted(record),
        "hit_horizon": bool(record.get("student_hit_horizon", False)),
        "reported_repeated_ngram_ratio": float(
            record.get("student_effective_repeated_ngram_ratio", 0.0)
        ),
        "recomputed_repeated_ngram_ratio": float(repetition["rate"]),
        "repetition_ngram_opportunities": int(
            repetition["total_ngram_opportunities"]
        ),
        "markers": markers,
        "categories": categories,
    }


def step_bin(step: int, size: int) -> tuple[int, int, str]:
    size = max(int(size), 1)
    if step < 0:
        return -1, -1, "unknown"
    start = (step // size) * size
    end = start + size - 1
    return start, end, str(start) if size == 1 else f"{start}-{end}"


def mean_or_none(values: Iterable[float | int | None]) -> float | None:
    selected = [float(value) for value in values if value is not None]
    return statistics.fmean(selected) if selected else None


def percentile(values: Iterable[int | float], q: float) -> float:
    selected = sorted(float(value) for value in values)
    if not selected:
        return 0.0
    if len(selected) == 1:
        return selected[0]
    location = (len(selected) - 1) * min(max(float(q), 0.0), 1.0)
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return selected[lower]
    fraction = location - lower
    return selected[lower] * (1.0 - fraction) + selected[upper] * fraction


def pearson(left: Iterable[float], right: Iterable[float]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(left, right)]
    if len(pairs) < 2:
        return None
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    x_centered = [value - x_mean for value in xs]
    y_centered = [value - y_mean for value in ys]
    denominator = math.sqrt(
        sum(value * value for value in x_centered)
        * sum(value * value for value in y_centered)
    )
    if denominator <= 1.0e-15:
        return None
    return sum(x * y for x, y in zip(x_centered, y_centered)) / denominator


def linear_slope(xs: Iterable[float], ys: Iterable[float]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys)]
    if len(pairs) < 2:
        return None
    x_mean = statistics.fmean(x for x, _ in pairs)
    y_mean = statistics.fmean(y for _, y in pairs)
    denominator = sum((x - x_mean) ** 2 for x, _ in pairs)
    if denominator <= 1.0e-15:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in pairs) / denominator


def entity_keys(samples: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    if not samples:
        return []
    marker_keys = sorted(samples[0]["markers"])
    category_keys = sorted(samples[0]["categories"])
    return [
        ("category", category, "") for category in category_keys
    ] + [
        ("marker", category, marker) for category, marker in marker_keys
    ]


def entity_result(
    sample: dict[str, Any], level: str, category: str, marker: str
) -> dict[str, Any]:
    if level == "category":
        return sample["categories"][category]
    return sample["markers"][(category, marker)]


def summarize_entities(
    samples: list[dict[str, Any]],
    *,
    step_bin_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[step_bin(sample["global_step"], step_bin_size)].append(sample)

    rows: list[dict[str, Any]] = []
    for (start, end, label), group in sorted(groups.items()):
        token_total = sum(max(int(sample["rollout_tokens"]), 0) for sample in group)
        for level, category, marker in entity_keys(samples):
            results = [entity_result(sample, level, category, marker) for sample in group]
            occurrences = sum(int(result["count"]) for result in results)
            present = sum(bool(result["present"]) for result in results)
            rows.append(
                {
                    "step_bin": label,
                    "step_start": start,
                    "step_end": end,
                    "step_center": (start + end) / 2.0 if start >= 0 else -1.0,
                    "level": level,
                    "category": category,
                    "marker": marker,
                    "samples": len(group),
                    "rollout_tokens": token_total,
                    "occurrences": occurrences,
                    "samples_with_marker": present,
                    "occurrences_per_sample": occurrences / max(len(group), 1),
                    "occurrences_per_1k_tokens": (
                        1000.0 * occurrences / token_total if token_total else 0.0
                    ),
                    "presence_fraction": present / max(len(group), 1),
                    "mean_first_char_fraction": mean_or_none(
                        result["first_char_fraction"] for result in results
                    ),
                    "mean_first_token_fraction": mean_or_none(
                        result["first_token_fraction"] for result in results
                    ),
                }
            )

    by_entity: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_entity[(row["level"], row["category"], row["marker"])].append(row)
    trends: list[dict[str, Any]] = []
    for (level, category, marker), entity_rows in sorted(by_entity.items()):
        valid = [row for row in entity_rows if int(row["step_start"]) >= 0]
        valid.sort(key=lambda row: int(row["step_start"]))
        xs = [float(row["step_center"]) for row in valid]
        frequency = [float(row["occurrences_per_1k_tokens"]) for row in valid]
        presence = [float(row["presence_fraction"]) for row in valid]
        frequency_slope = linear_slope(xs, frequency)
        presence_slope = linear_slope(xs, presence)
        first_frequency = frequency[0] if frequency else None
        last_frequency = frequency[-1] if frequency else None
        trends.append(
            {
                "level": level,
                "category": category,
                "marker": marker,
                "step_bins": len(valid),
                "first_step_bin": valid[0]["step_bin"] if valid else None,
                "last_step_bin": valid[-1]["step_bin"] if valid else None,
                "first_occurrences_per_1k_tokens": first_frequency,
                "last_occurrences_per_1k_tokens": last_frequency,
                "change_occurrences_per_1k_tokens": (
                    last_frequency - first_frequency
                    if first_frequency is not None and last_frequency is not None
                    else None
                ),
                "frequency_slope_per_100_steps": (
                    frequency_slope * 100.0 if frequency_slope is not None else None
                ),
                "frequency_step_correlation": pearson(xs, frequency),
                "first_presence_fraction": presence[0] if presence else None,
                "last_presence_fraction": presence[-1] if presence else None,
                "presence_slope_per_100_steps": (
                    presence_slope * 100.0 if presence_slope is not None else None
                ),
                "presence_step_correlation": pearson(xs, presence),
            }
        )
    return rows, trends


def summarize_overall_entities(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    token_total = sum(max(int(sample["rollout_tokens"]), 0) for sample in samples)
    rows: list[dict[str, Any]] = []
    lengths = [float(sample["rollout_tokens"]) for sample in samples]
    for level, category, marker in entity_keys(samples):
        results = [entity_result(sample, level, category, marker) for sample in samples]
        counts = [float(result["count"]) for result in results]
        densities = [
            1000.0 * count / max(float(sample["rollout_tokens"]), 1.0)
            for count, sample in zip(counts, samples)
        ]
        present_lengths = [
            length
            for length, result in zip(lengths, results)
            if bool(result["present"])
        ]
        absent_lengths = [
            length
            for length, result in zip(lengths, results)
            if not bool(result["present"])
        ]
        occurrences = int(sum(counts))
        samples_with = sum(bool(result["present"]) for result in results)
        rows.append(
            {
                "level": level,
                "category": category,
                "marker": marker,
                "samples": len(samples),
                "rollout_tokens": token_total,
                "occurrences": occurrences,
                "samples_with_marker": samples_with,
                "occurrences_per_sample": occurrences / max(len(samples), 1),
                "occurrences_per_1k_tokens": (
                    1000.0 * occurrences / token_total if token_total else 0.0
                ),
                "presence_fraction": samples_with / max(len(samples), 1),
                "mean_first_char_fraction": mean_or_none(
                    result["first_char_fraction"] for result in results
                ),
                "mean_first_token_fraction": mean_or_none(
                    result["first_token_fraction"] for result in results
                ),
                "count_length_correlation": pearson(counts, lengths),
                "density_length_correlation": pearson(densities, lengths),
                "mean_length_when_present": mean_or_none(present_lengths),
                "mean_length_when_absent": mean_or_none(absent_lengths),
            }
        )
    return rows


def summarize_rollout_steps(
    samples: list[dict[str, Any]], step_bin_size: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[step_bin(sample["global_step"], step_bin_size)].append(sample)
    rows: list[dict[str, Any]] = []
    for (start, end, label), group in sorted(groups.items()):
        lengths = [int(sample["rollout_tokens"]) for sample in group]
        rows.append(
            {
                "step_bin": label,
                "step_start": start,
                "step_end": end,
                "samples": len(group),
                "mean_rollout_tokens": statistics.fmean(lengths) if lengths else 0.0,
                "median_rollout_tokens": statistics.median(lengths) if lengths else 0.0,
                "p90_rollout_tokens": percentile(lengths, 0.90),
                "max_rollout_tokens": max(lengths, default=0),
                "mean_rollout_characters": statistics.fmean(
                    int(sample["rollout_characters"]) for sample in group
                ),
                "eos_fraction": sum(bool(sample["emitted_eos"]) for sample in group)
                / max(len(group), 1),
                "boxed_stop_fraction": sum(
                    sample["stop_reason"] == "boxed_answer" for sample in group
                )
                / max(len(group), 1),
                "horizon_fraction": sum(bool(sample["hit_horizon"]) for sample in group)
                / max(len(group), 1),
                "mean_reported_repeated_ngram_ratio": statistics.fmean(
                    float(sample["reported_repeated_ngram_ratio"])
                    for sample in group
                ),
                "mean_recomputed_repeated_ngram_ratio": statistics.fmean(
                    float(sample["recomputed_repeated_ngram_ratio"])
                    for sample in group
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sample_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "source_file",
        "line_number",
        "global_step",
        "loss_call_index",
        "rank",
        "sample_index_in_batch",
        "loss_mode",
        "strategy",
        "rollout_tokens",
        "rollout_characters",
        "logged_token_count",
        "token_sequence_complete",
        "stop_reason",
        "emitted_eos",
        "hit_horizon",
        "reported_repeated_ngram_ratio",
        "recomputed_repeated_ngram_ratio",
        "repetition_ngram_opportunities",
    ]
    rows: list[dict[str, Any]] = []
    for sample in samples:
        row = {field: sample[field] for field in fields}
        for category, result in sorted(sample["categories"].items()):
            row[f"{category}__count"] = result["count"]
            row[f"{category}__present"] = result["present"]
            row[f"{category}__first_char_fraction"] = result[
                "first_char_fraction"
            ]
            row[f"{category}__first_token_fraction"] = result[
                "first_token_fraction"
            ]
        rows.append(row)
    return rows


def marker_occurrence_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        for (category, marker), result in sorted(sample["markers"].items()):
            rows.append(
                {
                    "source_file": sample["source_file"],
                    "line_number": sample["line_number"],
                    "global_step": sample["global_step"],
                    "loss_call_index": sample["loss_call_index"],
                    "rank": sample["rank"],
                    "sample_index_in_batch": sample["sample_index_in_batch"],
                    "rollout_tokens": sample["rollout_tokens"],
                    "category": category,
                    "marker": marker,
                    **result,
                }
            )
    return rows


def print_trend_preview(trends: list[dict[str, Any]]) -> None:
    marker_rows = [row for row in trends if row["level"] == "marker"]
    print("\nMarker frequency trends (occurrences per 1k rollout tokens):")
    print(
        f"{'category':24} {'marker':18} {'first':>10} {'last':>10} "
        f"{'change':>10} {'slope/100':>11}"
    )
    for row in marker_rows:
        def display(value: Any) -> str:
            return "NA" if value is None else f"{float(value):.4f}"

        print(
            f"{str(row['category']):24.24} {str(row['marker']):18.18} "
            f"{display(row['first_occurrences_per_1k_tokens']):>10} "
            f"{display(row['last_occurrences_per_1k_tokens']):>10} "
            f"{display(row['change_occurrences_per_1k_tokens']):>10} "
            f"{display(row['frequency_slope_per_100_steps']):>11}"
        )


def main() -> None:
    args = parse_args()
    if args.step_bin_size <= 0:
        raise ValueError("--step-bin-size must be positive")
    if args.repetition_ngram_size <= 0:
        raise ValueError("--repetition-ngram-size must be positive")

    paths = resolve_inputs(args.input)
    if not paths:
        raise FileNotFoundError(f"No JSONL files matched: {args.input!r}")

    samples: list[dict[str, Any]] = []
    malformed_lines = 0
    skipped_events = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    if args.strict_json:
                        raise ValueError(
                            f"Malformed JSON in {path}:{line_number}: {error}"
                        ) from error
                    malformed_lines += 1
                    continue
                if record.get("event") != args.event:
                    skipped_events += 1
                    continue
                samples.append(
                    analyze_record(
                        record,
                        source_file=str(path),
                        line_number=line_number,
                        repetition_ngram_size=args.repetition_ngram_size,
                    )
                )

    if not samples:
        raise ValueError(
            f"No records with event={args.event!r} were found in {len(paths)} files"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step_rows, trend_rows = summarize_entities(
        samples,
        step_bin_size=args.step_bin_size,
    )
    overall_rows = summarize_overall_entities(samples)
    rollout_step_rows = summarize_rollout_steps(samples, args.step_bin_size)

    write_csv(output_dir / "rollout_samples.csv", sample_rows(samples))
    write_csv(
        output_dir / "marker_occurrences.csv",
        marker_occurrence_rows(samples),
    )
    write_csv(output_dir / "marker_overall_summary.csv", overall_rows)
    write_csv(output_dir / "marker_step_summary.csv", step_rows)
    write_csv(output_dir / "marker_trend_summary.csv", trend_rows)
    write_csv(output_dir / "rollout_step_summary.csv", rollout_step_rows)

    incomplete_sequences = sum(
        not bool(sample["token_sequence_complete"]) for sample in samples
    )
    summary = {
        "input_files": [str(path) for path in paths],
        "event": args.event,
        "sample_records": len(samples),
        "malformed_lines_skipped": malformed_lines,
        "nonmatching_events_skipped": skipped_events,
        "step_bin_size": args.step_bin_size,
        "repetition_ngram_size": args.repetition_ngram_size,
        "samples_with_incomplete_token_records": incomplete_sequences,
        "notes": {
            "phrase_unit": "decoded student_rollout text",
            "frequency_unit": "occurrences per 1000 rollout tokens",
            "eos_source": "student_emitted_eos / student_stop_reason metadata",
            "repetition_definition": (
                "a token is a repetition continuation when it completes a token "
                "n-gram observed earlier in the same rollout"
            ),
        },
        "outputs": [
            "rollout_samples.csv",
            "marker_occurrences.csv",
            "marker_overall_summary.csv",
            "marker_step_summary.csv",
            "marker_trend_summary.csv",
            "rollout_step_summary.csv",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"Analyzed {len(samples)} rollout records from {len(paths)} JSONL files; "
        f"wrote results to {output_dir}"
    )
    if malformed_lines:
        print(f"Warning: skipped {malformed_lines} malformed JSON lines")
    if incomplete_sequences:
        print(
            "Warning: token-level repetition is incomplete for "
            f"{incomplete_sequences} samples because not all target tokens were logged"
        )
    print_trend_preview(trend_rows)


if __name__ == "__main__":
    main()