#!/usr/bin/env python3
"""Discover human-readable rollout markers from a small random JSONL sample.

The intended input is a very large OPD ``token_debug_rank*.jsonl`` collection.
The script does *not* scan every JSONL record.  It memory-maps each file, uses
random byte seeks inside requested training-step ranges, and extracts only a
few scalar fields plus ``student_rollout``.  The large ``tokens`` array at the
end of each selected record is never deserialized.

Outputs include a compact sampled JSONL that can be shared for manual review,
ranked phrase candidates, early-vs-late and long-vs-short comparisons, and a
candidate monitoring dictionary.  Phrase discovery operates on decoded text;
a model tokenizer is only needed later when measuring phrase probabilities.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import mmap
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


DEFAULT_EVENT = "topk_opd_sample_debug"
GLOBAL_STEP_RE = re.compile(rb'"global_step"\s*:\s*(-?\d+)')
EVENT_RE = re.compile(rb'"event"\s*:\s*"([^"\\]*)"')
ROLLOUT_LENGTH_RE = re.compile(rb'"student_rollout_length"\s*:\s*(\d+)')
ROLLOUT_FIELD = b'"student_rollout"'

# Words preceded by a LaTeX backslash are excluded from phrase mining.  This
# removes command names such as ``frac`` while retaining surrounding prose.
WORD_RE = re.compile(r"(?<!\\)[A-Za-z]+(?:['\u2018\u2019][A-Za-z]+)?")
SEGMENT_SPLIT_RE = re.compile(r"(?:\r?\n)+|(?<=[.!?;:])\s+")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "for", "from", "had", "has", "have", "he", "her", "here", "hers",
    "him", "his", "i", "if", "in", "into", "is", "it", "its", "of",
    "on", "or", "our", "ours", "she", "so", "than", "that", "the",
    "their", "theirs", "them", "then", "there", "these", "they", "this",
    "those", "to", "us", "was", "we", "were", "what", "when", "where",
    "which", "who", "will", "with", "you", "your",
}

INTERPRETABLE_UNIGRAMS = {
    "additionally", "actually", "alternatively", "answer", "but",
    "consequently", "continue", "finally", "first", "furthermore", "hence",
    "however", "instead", "let's", "moreover", "next", "now", "otherwise",
    "reconsider", "second", "similarly", "step", "therefore", "thus",
    "verify", "wait",
}

CATEGORY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "termination",
        (
            "<boxed>", "final answer", "the answer is", "answer is boxed",
            "in conclusion", "we conclude", "this completes", "final result",
        ),
    ),
    (
        "conclusion_transition",
        (
            "finally", "therefore", "thus", "hence", "consequently", "accordingly",
            "we obtain", "we get", "this gives", "this yields", "it follows",
        ),
    ),
    (
        "expansion",
        (
            "additionally", "moreover", "furthermore", "next", "another",
            "alternatively", "we can also", "also consider", "now consider",
            "proceed to", "continue", "step",
        ),
    ),
    (
        "planning",
        (
            "let's", "we need", "to solve", "first", "begin by", "consider",
            "define", "our goal", "we want", "the problem asks",
        ),
    ),
    (
        "verification",
        (
            "verify", "check", "substitute back", "substituting back",
            "satisfies", "confirm", "validation", "sanity check",
        ),
    ),
    (
        "self_correction",
        (
            "wait", "however", "actually", "instead", "reconsider", "mistake",
            "incorrect", "correction", "but this", "this is wrong",
        ),
    ),
    (
        "code_or_tool",
        ("<code_block>", "python", "sympy", "using code", "implement this"),
    ),
    (
        "structure",
        (
            "<markdown_heading>", "<numbered_item>", "<bullet_item>",
            "<display_math>",
        ),
    ),
)


@dataclass(frozen=True)
class StageRange:
    name: str
    start: int
    end: int

    def contains(self, step: int) -> bool:
        return self.start <= step <= self.end


@dataclass
class SampledRollout:
    source_file: str
    line_start_byte: int
    record_bytes: int
    global_step: int
    stage: str
    student_rollout_length: int
    student_rollout: str
    inverse_byte_weight: float = 0.0
    analysis_weight: float = 1.0


@dataclass
class SampleFeatures:
    sample: SampledRollout
    phrase_counts: Counter[tuple[str, ...]]
    boundary_counts: Counter[tuple[str, ...]]
    word_count: int


@dataclass
class PhraseAggregate:
    raw_count: int = 0
    document_count: int = 0
    boundary_count: int = 0
    weighted_count_by_stage: dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )
    raw_count_by_stage: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    examples: list[str] = field(default_factory=list)


class MappedJSONL:
    """Random-access helper that never deserializes the full JSON record."""

    def __init__(
        self,
        path: Path,
        *,
        event: str,
        max_extract_bytes: int,
        header_search_bytes: int = 262_144,
    ) -> None:
        self.path = path
        self.event = event
        self.max_extract_bytes = max(int(max_extract_bytes), 65_536)
        self.header_search_bytes = max(int(header_search_bytes), 4_096)
        self._handle = path.open("rb")
        self.size = path.stat().st_size
        if self.size <= 0:
            self._handle.close()
            raise ValueError(f"JSONL file is empty: {path}")
        self._mmap = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)

    def close(self) -> None:
        self._mmap.close()
        self._handle.close()

    def __enter__(self) -> "MappedJSONL":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def line_bounds(self, offset: int) -> tuple[int, int]:
        offset = min(max(int(offset), 0), self.size - 1)
        if offset == 0:
            start = 0
        else:
            previous_newline = self._mmap.rfind(b"\n", 0, offset)
            start = previous_newline + 1
        next_newline = self._mmap.find(b"\n", offset)
        end = self.size if next_newline < 0 else next_newline
        # An offset can point exactly at a newline. Move to the next non-empty
        # line so boundary searches remain monotonic.
        if start >= end and end < self.size:
            start = end + 1
            next_newline = self._mmap.find(b"\n", start)
            end = self.size if next_newline < 0 else next_newline
        return start, end

    def _header(self, start: int, end: int) -> bytes:
        return self._mmap[start : min(end, start + self.header_search_bytes)]

    def step_at_offset(self, offset: int) -> int | None:
        start, end = self.line_bounds(offset)
        match = GLOBAL_STEP_RE.search(self._header(start, end))
        return int(match.group(1)) if match else None

    def boundary_for_step(self, target_step: int) -> int:
        """Approximate the first byte whose containing record reaches a step."""
        low, high = 0, self.size
        # global_step is non-decreasing in each rank JSONL produced by one run.
        # Binary search touches O(log(file_size)) records rather than scanning.
        for _ in range(64):
            if low >= high:
                break
            middle = (low + high) // 2
            step = self.step_at_offset(middle)
            if step is None:
                # Malformed/foreign records are uncommon; moving right avoids
                # accidentally declaring a large early interval empty.
                low = middle + 1
            elif step < target_step:
                low = middle + 1
            else:
                high = middle
        return min(max(low, 0), self.size)

    def stage_interval(self, stage: StageRange) -> tuple[int, int] | None:
        start = self.boundary_for_step(stage.start)
        end = self.boundary_for_step(stage.end + 1)
        if end <= start:
            # If the file ends inside this stage, the upper boundary is EOF.
            last_step = self.step_at_offset(self.size - 1)
            if last_step is not None and stage.contains(last_step):
                end = self.size
        return (start, end) if end > start else None

    def _json_string_at(self, value_start: int, hard_end: int) -> tuple[str, int] | None:
        if value_start >= hard_end or self._mmap[value_start] != ord('"'):
            return None
        cursor = value_start + 1
        while cursor < hard_end:
            byte = self._mmap[cursor]
            if byte == ord("\\"):
                cursor += 2
                continue
            if byte == ord('"'):
                encoded = self._mmap[value_start : cursor + 1]
                try:
                    return str(json.loads(encoded.decode("utf-8"))), cursor + 1
                except (UnicodeDecodeError, json.JSONDecodeError):
                    return None
            cursor += 1
        return None

    def extract_at_offset(self, offset: int, stage_name: str) -> SampledRollout | None:
        start, end = self.line_bounds(offset)
        header = self._header(start, end)
        event_match = EVENT_RE.search(header)
        if event_match is not None:
            try:
                event = event_match.group(1).decode("utf-8")
            except UnicodeDecodeError:
                return None
            if event != self.event:
                return None
        step_match = GLOBAL_STEP_RE.search(header)
        if step_match is None:
            return None
        global_step = int(step_match.group(1))

        hard_end = min(end, start + self.max_extract_bytes)
        field_position = self._mmap.find(ROLLOUT_FIELD, start, hard_end)
        if field_position < 0:
            return None
        colon = self._mmap.find(b":", field_position + len(ROLLOUT_FIELD), hard_end)
        if colon < 0:
            return None
        value_start = colon + 1
        while value_start < hard_end and self._mmap[value_start] in b" \t\r":
            value_start += 1
        parsed = self._json_string_at(value_start, hard_end)
        if parsed is None:
            return None
        rollout, value_end = parsed

        scalar_end = min(end, value_end + 131_072)
        length_match = ROLLOUT_LENGTH_RE.search(self._mmap[value_end:scalar_end])
        if length_match is None:
            # This should not happen for current OPD logs. Word count is a safe
            # fallback for dictionary discovery, but the summary flags it.
            rollout_length = len(WORD_RE.findall(rollout))
        else:
            rollout_length = int(length_match.group(1))
        record_bytes = max(end - start + (1 if end < self.size else 0), 1)
        return SampledRollout(
            source_file=str(self.path),
            line_start_byte=start,
            record_bytes=record_bytes,
            global_step=global_step,
            stage=stage_name,
            student_rollout_length=rollout_length,
            student_rollout=rollout,
            inverse_byte_weight=1.0 / record_bytes,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Random-seek a small sample from huge OPD JSONL logs and discover "
            "candidate behavioral marker phrases."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="JSONL files, directories, or glob patterns.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--event", default=DEFAULT_EVENT)
    parser.add_argument(
        "--samples-per-stage",
        type=int,
        default=30,
        help="Random rollouts retained from each training stage (default: 30).",
    )
    parser.add_argument(
        "--stage-ranges",
        default=None,
        help=(
            "Inclusive ranges such as 'early:0-49,middle:50-99,late:100-199'. "
            "By default --max-step is split into three equal ranges."
        ),
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=200,
        help="Number of optimizer steps used for default stage ranges (default: 200).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-document-frequency", type=int, default=3)
    parser.add_argument("--max-ngram", type=int, default=5)
    parser.add_argument("--top-per-category", type=int, default=40)
    parser.add_argument(
        "--max-attempts-per-stage",
        type=int,
        default=1000,
        help="Safety limit for random seeks in each stage (default: 1000).",
    )
    parser.add_argument(
        "--max-extract-mb",
        type=float,
        default=4.0,
        help=(
            "Maximum bytes copied from the beginning of a selected record while "
            "locating its rollout; the token array is not copied (default: 4 MB)."
        ),
    )
    return parser.parse_args()


def resolve_inputs(specs: Iterable[str]) -> list[Path]:
    import glob

    paths: set[Path] = set()
    for spec in specs:
        candidate = Path(spec).expanduser()
        if candidate.is_file():
            paths.add(candidate.resolve())
        elif candidate.is_dir():
            paths.update(path.resolve() for path in candidate.rglob("*.jsonl"))
        else:
            for match in glob.glob(spec, recursive=True):
                path = Path(match).expanduser()
                if path.is_file():
                    paths.add(path.resolve())
                elif path.is_dir():
                    paths.update(item.resolve() for item in path.rglob("*.jsonl"))
    return sorted(path for path in paths if path.stat().st_size > 0)


def parse_stage_ranges(spec: str | None, max_step: int) -> list[StageRange]:
    if spec:
        ranges: list[StageRange] = []
        for item in spec.split(","):
            name, separator, bounds = item.strip().partition(":")
            if not separator:
                raise ValueError(f"Invalid stage range {item!r}; expected name:start-end")
            lower, dash, upper = bounds.partition("-")
            if not dash:
                raise ValueError(f"Invalid stage bounds {bounds!r}; expected start-end")
            ranges.append(StageRange(name.strip(), int(lower), int(upper)))
    else:
        if max_step < 3:
            raise ValueError("--max-step must be at least 3")
        first_end = max_step // 3 - 1
        second_end = 2 * max_step // 3 - 1
        ranges = [
            StageRange("early", 0, first_end),
            StageRange("middle", first_end + 1, second_end),
            StageRange("late", second_end + 1, max_step - 1),
        ]
    for stage in ranges:
        if not stage.name or stage.start < 0 or stage.end < stage.start:
            raise ValueError(f"Invalid stage range: {stage}")
    ordered = sorted(ranges, key=lambda stage: stage.start)
    for previous, current in zip(ordered, ordered[1:]):
        if current.start <= previous.end:
            raise ValueError(f"Stage ranges overlap: {previous} and {current}")
    return ordered


def weighted_choice(
    rng: random.Random,
    choices: list[tuple[MappedJSONL, int, int]],
) -> tuple[MappedJSONL, int, int]:
    weights = [max(end - start, 0) for _, start, end in choices]
    return rng.choices(choices, weights=weights, k=1)[0]


def sample_rollouts(
    sources: list[MappedJSONL],
    stages: list[StageRange],
    *,
    samples_per_stage: int,
    max_attempts_per_stage: int,
    seed: int,
) -> tuple[list[SampledRollout], dict[str, Any]]:
    rng = random.Random(seed)
    sampled: list[SampledRollout] = []
    used: set[tuple[str, int]] = set()
    sampling_details: dict[str, Any] = {}

    for stage in stages:
        intervals: list[tuple[MappedJSONL, int, int]] = []
        for source in sources:
            interval = source.stage_interval(stage)
            if interval is not None:
                intervals.append((source, interval[0], interval[1]))
        if not intervals:
            sampling_details[stage.name] = {
                "requested": samples_per_stage,
                "sampled": 0,
                "attempts": 0,
                "warning": "No byte interval found for this stage",
            }
            continue

        stage_samples: list[SampledRollout] = []
        attempts = 0
        while (
            len(stage_samples) < samples_per_stage
            and attempts < max_attempts_per_stage
        ):
            attempts += 1
            source, start, end = weighted_choice(rng, intervals)
            if end <= start:
                continue
            offset = rng.randrange(start, end)
            record = source.extract_at_offset(offset, stage.name)
            if record is None or not stage.contains(record.global_step):
                continue
            key = (record.source_file, record.line_start_byte)
            if key in used:
                continue
            used.add(key)
            stage_samples.append(record)
        sampled.extend(stage_samples)
        detail: dict[str, Any] = {
            "requested": samples_per_stage,
            "sampled": len(stage_samples),
            "attempts": attempts,
            "byte_intervals": [
                {
                    "file": str(source.path),
                    "start": start,
                    "end": end,
                    "bytes": end - start,
                }
                for source, start, end in intervals
            ],
        }
        if len(stage_samples) < samples_per_stage:
            detail["warning"] = (
                "Stage quota was not reached. Increase --max-attempts-per-stage "
                "or lower --samples-per-stage."
            )
        sampling_details[stage.name] = detail

    # Random-byte sampling favors physically large JSONL records.  An inverse
    # record-byte weight is normalized to mean one within each stage, providing
    # an approximately line-uniform correction for phrase-rate estimates.
    by_stage: dict[str, list[SampledRollout]] = defaultdict(list)
    for sample in sampled:
        by_stage[sample.stage].append(sample)
    for stage_samples in by_stage.values():
        mean_inverse = statistics.fmean(
            sample.inverse_byte_weight for sample in stage_samples
        )
        for sample in stage_samples:
            sample.analysis_weight = (
                sample.inverse_byte_weight / mean_inverse if mean_inverse else 1.0
            )
    sampled.sort(key=lambda sample: (sample.global_step, sample.source_file, sample.line_start_byte))
    return sampled, sampling_details


def remove_code_bodies(text: str) -> str:
    parts = text.split("```")
    if len(parts) == 1:
        prose = text
    else:
        kept: list[str] = []
        for index, part in enumerate(parts):
            if index % 2 == 0:
                kept.append(part)
            else:
                kept.append("\n code block \n")
        prose = "".join(kept)
    # Remove full LaTeX command names before word tokenization. A negative
    # lookbehind alone would suppress the initial ``b`` in ``\\boxed`` but
    # incorrectly leave ``oxed`` as an ordinary word candidate.
    return re.sub(r"\\[A-Za-z]+", " ", prose)


def normalized_words(segment: str) -> list[str]:
    return [
        match.group(0).lower().replace("\u2018", "'").replace("\u2019", "'")
        for match in WORD_RE.finditer(segment)
    ]


def keep_phrase(tokens: tuple[str, ...]) -> bool:
    if not tokens:
        return False
    if len(tokens) == 1:
        return tokens[0] in INTERPRETABLE_UNIGRAMS
    if all(token in STOPWORDS for token in tokens):
        return False
    # Very long repeated words are usually code identifiers or corrupted text.
    return all(len(token) <= 30 for token in tokens)


def structural_counts(text: str) -> Counter[tuple[str, ...]]:
    patterns = {
        ("<boxed>",): re.compile(r"\\boxed\s*\{", flags=re.IGNORECASE),
        ("<code_block>",): re.compile(r"```"),
        ("<markdown_heading>",): re.compile(r"(?m)^\s*#{2,6}(?:\s|$)"),
        ("<numbered_item>",): re.compile(
            r"(?im)^\s*(?:step\s*)?\d+\s*[.) :]"
        ),
        ("<bullet_item>",): re.compile(r"(?m)^\s*[-*+]\s+"),
        ("<display_math>",): re.compile(r"\$\$|\\\[|\\\]"),
    }
    counts: Counter[tuple[str, ...]] = Counter()
    for key, pattern in patterns.items():
        count = len(pattern.findall(text))
        if count > 0:
            counts[key] = count
    return counts


def extract_phrase_features(
    sample: SampledRollout,
    max_ngram: int,
) -> SampleFeatures:
    text = sample.student_rollout
    phrase_counts: Counter[tuple[str, ...]] = structural_counts(text)
    boundary_counts: Counter[tuple[str, ...]] = Counter()
    word_count = 0
    prose = remove_code_bodies(text)
    for segment in SEGMENT_SPLIT_RE.split(prose):
        words = normalized_words(segment)
        if not words:
            continue
        word_count += len(words)
        for start in range(len(words)):
            for width in range(1, min(max_ngram, len(words) - start) + 1):
                phrase = tuple(words[start : start + width])
                if keep_phrase(phrase):
                    phrase_counts[phrase] += 1
                    if start == 0:
                        boundary_counts[phrase] += 1
    for phrase, count in structural_counts(text).items():
        if count:
            boundary_counts[phrase] += count
    return SampleFeatures(
        sample=sample,
        phrase_counts=phrase_counts,
        boundary_counts=boundary_counts,
        word_count=max(word_count, 1),
    )


def phrase_text(phrase: tuple[str, ...]) -> str:
    return " ".join(phrase)


def classify_phrase(phrase: str) -> str:
    lowered = phrase.lower()
    for category, patterns in CATEGORY_PATTERNS:
        for pattern in patterns:
            if lowered == pattern or lowered.startswith(pattern + " ") or f" {pattern} " in f" {lowered} ":
                return category
    return "discovered_candidate"


def excerpt_for_phrase(text: str, phrase: tuple[str, ...], width: int = 180) -> str:
    if len(phrase) == 1 and phrase[0].startswith("<"):
        literal_map = {
            "<boxed>": "\\boxed",
            "<code_block>": "```",
            "<markdown_heading>": "###",
        }
        needle = literal_map.get(phrase[0], "")
        position = text.find(needle) if needle else -1
    else:
        pattern = re.compile(
            r"\b" + r"\W+".join(re.escape(token) for token in phrase) + r"\b",
            flags=re.IGNORECASE,
        )
        match = pattern.search(text)
        position = match.start() if match else -1
    if position < 0:
        return text[:width].replace("\n", " ")
    half = width // 2
    start = max(position - half, 0)
    end = min(position + half, len(text))
    return text[start:end].replace("\n", " ")


def pearson(xs: Iterable[float], ys: Iterable[float]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys)]
    if len(pairs) < 3:
        return None
    x_mean = statistics.fmean(x for x, _ in pairs)
    y_mean = statistics.fmean(y for _, y in pairs)
    x_centered = [x - x_mean for x, _ in pairs]
    y_centered = [y - y_mean for _, y in pairs]
    denominator = math.sqrt(
        sum(value * value for value in x_centered)
        * sum(value * value for value in y_centered)
    )
    if denominator <= 1.0e-15:
        return None
    return sum(x * y for x, y in zip(x_centered, y_centered)) / denominator


def smoothed_log2_ratio(
    numerator_count: float,
    numerator_words: float,
    denominator_count: float,
    denominator_words: float,
) -> float:
    # 0.5 pseudo-occurrences per 500 words corresponds to a weak 1/1000 prior.
    numerator_rate = (numerator_count + 0.5) / (numerator_words + 500.0)
    denominator_rate = (denominator_count + 0.5) / (denominator_words + 500.0)
    return math.log2(numerator_rate / denominator_rate)


def build_candidate_rows(
    features: list[SampleFeatures],
    stages: list[StageRange],
    *,
    min_document_frequency: int,
) -> list[dict[str, Any]]:
    aggregates: dict[tuple[str, ...], PhraseAggregate] = defaultdict(PhraseAggregate)
    weighted_words_by_stage: dict[str, float] = defaultdict(float)
    raw_words_by_stage: dict[str, int] = defaultdict(int)

    for feature in features:
        stage = feature.sample.stage
        weight = feature.sample.analysis_weight
        weighted_words_by_stage[stage] += weight * feature.word_count
        raw_words_by_stage[stage] += feature.word_count
        for phrase, count in feature.phrase_counts.items():
            aggregate = aggregates[phrase]
            aggregate.raw_count += count
            aggregate.document_count += 1
            aggregate.boundary_count += feature.boundary_counts.get(phrase, 0)
            aggregate.raw_count_by_stage[stage] += count
            aggregate.weighted_count_by_stage[stage] += weight * count
            # Avoid an O(unique n-grams x rollout length) search for one-off
            # phrases. Examples are only needed after a phrase reaches the
            # configured support threshold (structural markers are exempt).
            structural = len(phrase) == 1 and phrase[0].startswith("<")
            if (
                len(aggregate.examples) < 2
                and (aggregate.document_count >= min_document_frequency or structural)
            ):
                example = excerpt_for_phrase(feature.sample.student_rollout, phrase)
                if example not in aggregate.examples:
                    aggregate.examples.append(example)

    sorted_features = sorted(
        features,
        key=lambda item: item.sample.student_rollout_length,
    )
    group_size = max(len(sorted_features) // 3, 1)
    short_features = sorted_features[:group_size]
    long_features = sorted_features[-group_size:]

    def group_phrase_rate(
        group: list[SampleFeatures], phrase: tuple[str, ...]
    ) -> tuple[float, float]:
        weighted_count = 0.0
        weighted_words = 0.0
        inverse_values = [item.sample.inverse_byte_weight for item in group]
        mean_inverse = statistics.fmean(inverse_values) if inverse_values else 1.0
        for item in group:
            weight = item.sample.inverse_byte_weight / mean_inverse if mean_inverse else 1.0
            weighted_count += weight * item.phrase_counts.get(phrase, 0)
            weighted_words += weight * item.word_count
        return weighted_count, weighted_words

    early_name = stages[0].name
    late_name = stages[-1].name
    rows: list[dict[str, Any]] = []
    for phrase, aggregate in aggregates.items():
        structural = len(phrase) == 1 and phrase[0].startswith("<")
        if aggregate.document_count < min_document_frequency and not structural:
            continue
        text = phrase_text(phrase)
        category = classify_phrase(text)
        early_count = aggregate.weighted_count_by_stage.get(early_name, 0.0)
        late_count = aggregate.weighted_count_by_stage.get(late_name, 0.0)
        early_words = weighted_words_by_stage.get(early_name, 0.0)
        late_words = weighted_words_by_stage.get(late_name, 0.0)
        early_rate = 1000.0 * early_count / early_words if early_words else 0.0
        late_rate = 1000.0 * late_count / late_words if late_words else 0.0
        late_early_log2 = smoothed_log2_ratio(
            late_count, late_words, early_count, early_words
        )

        short_count, short_words = group_phrase_rate(short_features, phrase)
        long_count, long_words = group_phrase_rate(long_features, phrase)
        short_rate = 1000.0 * short_count / short_words if short_words else 0.0
        long_rate = 1000.0 * long_count / long_words if long_words else 0.0
        long_short_log2 = smoothed_log2_ratio(
            long_count, long_words, short_count, short_words
        )

        steps = [float(item.sample.global_step) for item in features]
        sample_densities = [
            1000.0 * item.phrase_counts.get(phrase, 0) / item.word_count
            for item in features
        ]
        step_correlation = pearson(steps, sample_densities)
        boundary_fraction = aggregate.boundary_count / max(aggregate.raw_count, 1)
        support = math.sqrt(max(aggregate.document_count, 1))
        trend_score = late_early_log2 * support
        length_score = long_short_log2 * support
        dictionary_score = (
            math.log1p(aggregate.document_count)
            + abs(trend_score)
            + 0.5 * abs(length_score)
            + boundary_fraction
        )
        row: dict[str, Any] = {
            "phrase": text,
            "ngram_size": 0 if structural else len(phrase),
            "suggested_category": category,
            "sample_occurrences": aggregate.raw_count,
            "sample_document_frequency": aggregate.document_count,
            "sample_document_fraction": aggregate.document_count / max(len(features), 1),
            "boundary_occurrences": aggregate.boundary_count,
            "boundary_fraction": boundary_fraction,
            "early_rate_per_1k_words": early_rate,
            "late_rate_per_1k_words": late_rate,
            "late_vs_early_log2_ratio": late_early_log2,
            "trend_score": trend_score,
            "short_rate_per_1k_words": short_rate,
            "long_rate_per_1k_words": long_rate,
            "long_vs_short_log2_ratio": long_short_log2,
            "length_score": length_score,
            "step_density_correlation": step_correlation,
            "dictionary_score": dictionary_score,
            "examples": json.dumps(aggregate.examples, ensure_ascii=False),
        }
        for stage in stages:
            name = stage.name
            weighted_count = aggregate.weighted_count_by_stage.get(name, 0.0)
            weighted_words = weighted_words_by_stage.get(name, 0.0)
            row[f"{name}_raw_count"] = aggregate.raw_count_by_stage.get(name, 0)
            row[f"{name}_weighted_rate_per_1k_words"] = (
                1000.0 * weighted_count / weighted_words if weighted_words else 0.0
            )
        rows.append(row)
    rows.sort(key=lambda row: (-float(row["dictionary_score"]), str(row["phrase"])))
    return rows


def dictionary_from_candidates(
    rows: list[dict[str, Any]],
    *,
    stages: list[StageRange],
    top_per_category: int,
    sample_count: int,
) -> dict[str, Any]:
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        category = str(row["suggested_category"])
        if category == "discovered_candidate":
            is_interesting = (
                abs(float(row["late_vs_early_log2_ratio"])) >= 0.5
                or abs(float(row["long_vs_short_log2_ratio"])) >= 0.5
                or float(row["boundary_fraction"]) >= 0.25
            )
            if not is_interesting:
                continue
        grouped_rows[category].append(row)

    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for category, candidates in grouped_rows.items():
        # Prefer concise phrases, then remove mechanical extensions such as
        # ``additionally`` -> ``additionally we`` -> ``additionally we can``.
        candidates.sort(
            key=lambda row: (
                int(row["ngram_size"]) if int(row["ngram_size"]) > 0 else 0,
                -float(row["dictionary_score"]),
                str(row["phrase"]),
            )
        )
        selected_phrases: list[tuple[str, ...]] = []
        for row in candidates:
            phrase = str(row["phrase"])
            tokens = tuple(phrase.split())
            redundant = False
            for selected in selected_phrases:
                if not selected or not tokens:
                    continue
                if len(selected) == 1 and tokens[:1] == selected:
                    redundant = True
                    break
                if len(selected) >= 2:
                    for start in range(len(tokens) - len(selected) + 1):
                        if tokens[start : start + len(selected)] == selected:
                            redundant = True
                            break
                if redundant:
                    break
            if redundant:
                continue
            selected_phrases.append(tokens)
            categories[category].append(
                {
                    "phrase": phrase,
                    "match_type": (
                        "structural_pattern"
                        if int(row["ngram_size"]) == 0
                        else "case_insensitive_phrase"
                    ),
                    "sample_document_frequency": row["sample_document_frequency"],
                    "late_vs_early_log2_ratio": row["late_vs_early_log2_ratio"],
                    "long_vs_short_log2_ratio": row["long_vs_short_log2_ratio"],
                    "boundary_fraction": row["boundary_fraction"],
                    "examples": json.loads(str(row["examples"])),
                }
            )
            if len(categories[category]) >= top_per_category:
                break

    # These fields are not ordinary text phrases but belong in the final
    # monitoring schema regardless of whether the sampled rollouts contain them.
    categories["termination"].insert(
        0,
        {
            "phrase": "<EOS>",
            "match_type": "jsonl_stop_metadata_or_model_probability",
            "note": "Use emitted-EOS metadata for rollouts and EOS token probability for probes.",
        },
    )
    categories["repetition_continuation"] = [
        {
            "phrase": "<repeated_ngram_continuation>",
            "match_type": "token_sequence_rule",
            "note": "True when the current token completes an n-gram already seen in the rollout.",
        }
    ]
    return {
        "status": "candidate_dictionary_requires_review",
        "sample_count": sample_count,
        "stage_ranges": [asdict(stage) for stage in stages],
        "matching_note": (
            "Phrases were discovered from decoded text. After manual review, tokenize "
            "each retained phrase (including whitespace/case variants) for probability probes."
        ),
        "categories": dict(sorted(categories.items())),
    }


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


def write_outputs(
    output_dir: Path,
    samples: list[SampledRollout],
    candidate_rows: list[dict[str, Any]],
    dictionary: dict[str, Any],
    *,
    input_files: list[Path],
    stages: list[StageRange],
    sampling_details: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "sampled_rollouts.jsonl").open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")

    write_csv(output_dir / "candidate_phrases.csv", candidate_rows)
    rising = sorted(
        candidate_rows,
        key=lambda row: float(row["trend_score"]),
        reverse=True,
    )[:200]
    falling = sorted(
        candidate_rows,
        key=lambda row: float(row["trend_score"]),
    )[:200]
    long_associated = sorted(
        candidate_rows,
        key=lambda row: float(row["length_score"]),
        reverse=True,
    )[:200]
    write_csv(output_dir / "top_rising_phrases.csv", rising)
    write_csv(output_dir / "top_falling_phrases.csv", falling)
    write_csv(output_dir / "top_long_associated_phrases.csv", long_associated)
    (output_dir / "rollout_marker_dictionary.json").write_text(
        json.dumps(dictionary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "input_files": [str(path) for path in input_files],
        "input_total_bytes": sum(path.stat().st_size for path in input_files),
        "input_files_were_fully_scanned": False,
        "sampling_method": (
            "memory-mapped random byte seeks stratified by monotonic global_step; "
            "inverse-record-byte weighting corrects phrase-rate estimates"
        ),
        "sample_count": len(samples),
        "stage_ranges": [asdict(stage) for stage in stages],
        "sampling_details": sampling_details,
        "parameters": {
            "seed": args.seed,
            "samples_per_stage": args.samples_per_stage,
            "min_document_frequency": args.min_document_frequency,
            "max_ngram": args.max_ngram,
            "top_per_category": args.top_per_category,
            "max_extract_mb": args.max_extract_mb,
        },
        "outputs": [
            "sampled_rollouts.jsonl",
            "candidate_phrases.csv",
            "top_rising_phrases.csv",
            "top_falling_phrases.csv",
            "top_long_associated_phrases.csv",
            "rollout_marker_dictionary.json",
        ],
        "important_limitations": [
            "The dictionary is a candidate list and should be manually reviewed.",
            "Byte-seek sampling is approximate; inverse-byte weighting reduces long-record bias.",
            "Files must have non-decreasing global_step within each rank file for stage stratification.",
            "Occurrence trends do not equal fixed-prefix model probability trends.",
        ],
    }
    (output_dir / "sampling_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_preview(rows: list[dict[str, Any]], title: str, score: str, reverse: bool) -> None:
    eligible = [
        row
        for row in rows
        if (float(row[score]) > 0.0 if reverse else float(row[score]) < 0.0)
    ]
    selected = sorted(
        eligible,
        key=lambda row: float(row[score]),
        reverse=reverse,
    )[:15]
    print(f"\n{title}")
    print(f"{'phrase':38} {'category':23} {score:>14} {'df':>6}")
    for row in selected:
        print(
            f"{str(row['phrase']):38.38} "
            f"{str(row['suggested_category']):23.23} "
            f"{float(row[score]):14.4f} "
            f"{int(row['sample_document_frequency']):6d}"
        )


def main() -> None:
    args = parse_args()
    if args.samples_per_stage <= 0:
        raise ValueError("--samples-per-stage must be positive")
    if not 1 <= args.max_ngram <= 8:
        raise ValueError("--max-ngram must be between 1 and 8")
    if args.min_document_frequency <= 0:
        raise ValueError("--min-document-frequency must be positive")
    if args.max_extract_mb <= 0:
        raise ValueError("--max-extract-mb must be positive")

    paths = resolve_inputs(args.input)
    if not paths:
        raise FileNotFoundError(f"No non-empty JSONL files matched {args.input!r}")
    stages = parse_stage_ranges(args.stage_ranges, args.max_step)
    sources: list[MappedJSONL] = []
    try:
        for path in paths:
            sources.append(
                MappedJSONL(
                    path,
                    event=args.event,
                    max_extract_bytes=int(args.max_extract_mb * 1024 * 1024),
                )
            )
        samples, sampling_details = sample_rollouts(
            sources,
            stages,
            samples_per_stage=args.samples_per_stage,
            max_attempts_per_stage=args.max_attempts_per_stage,
            seed=args.seed,
        )
    finally:
        for source in sources:
            source.close()

    if not samples:
        raise RuntimeError(
            "No rollout records were sampled. Check --event, --stage-ranges, and "
            "whether each rank file has monotonically increasing global_step."
        )

    features = [
        extract_phrase_features(sample, args.max_ngram)
        for sample in samples
    ]
    candidate_rows = build_candidate_rows(
        features,
        stages,
        min_document_frequency=args.min_document_frequency,
    )
    dictionary = dictionary_from_candidates(
        candidate_rows,
        stages=stages,
        top_per_category=args.top_per_category,
        sample_count=len(samples),
    )
    output_dir = Path(args.output_dir)
    write_outputs(
        output_dir,
        samples,
        candidate_rows,
        dictionary,
        input_files=paths,
        stages=stages,
        sampling_details=sampling_details,
        args=args,
    )

    total_gb = sum(path.stat().st_size for path in paths) / (1024 ** 3)
    print(
        f"Sampled {len(samples)} rollouts from {len(paths)} files ({total_gb:.2f} GiB) "
        f"without a full scan; wrote {len(candidate_rows)} phrase candidates to {output_dir}"
    )
    print_preview(candidate_rows, "Most increasing phrases", "trend_score", True)
    print_preview(candidate_rows, "Most decreasing phrases", "trend_score", False)
    print_preview(candidate_rows, "Most associated with long rollouts", "length_score", True)


if __name__ == "__main__":
    main()
