from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable, Mapping


# These phrases were curated from the sampled rollout dictionary.  They describe
# reasoning behaviour rather than mathematical subject matter, so expressions
# such as "number of", "the function", and variable-name n-grams are excluded.
DEFAULT_BEHAVIOR_MARKERS: dict[str, tuple[str, ...]] = {
    "termination": (
        "final answer",
        "the answer is",
        "final result",
        "correct answer",
    ),
    "conclusion": (
        "therefore",
        "thus",
        "hence",
        "finally",
        "we get",
        "this gives",
        "it follows",
        "we obtain",
        "we conclude",
    ),
    "planning": (
        "let's",
        "now let's",
        "consider",
        "we need",
        "we want",
        "our goal",
    ),
    "expansion": (
        "next",
        "continue",
        "step",
        "we can also",
        "proceed",
    ),
    "alternative_approach": (
        "another approach",
        "try another",
        "another way",
        "different approach",
        "start over",
    ),
    "self_correction": (
        "however",
        "instead",
        "actually",
        "reconsider",
        "but this",
        "mistake",
        "incorrect",
        "not helpful",
        "not correct",
    ),
    "verification": (
        "check",
        "verify",
        "confirm",
        "satisfies",
        "substitute back",
        "substituting back",
        "double check",
        "check again",
    ),
    "structure": (),
    "code_tool": (
        "python",
        "sympy",
    ),
}


# Only the first token of a phrase can be read from one next-token distribution.
# The manifest emitted by the trainer labels these explicitly as phrase starts.
DEFAULT_FOCUS_MARKERS: dict[str, str] = {
    "boxed": r"\boxed{",
    "final_answer": "final answer",
    "answer_is": "the answer is",
    "therefore": "therefore",
    "thus": "thus",
    "hence": "hence",
    "finally": "finally",
    "lets": "let's",
    "consider": "consider",
    "next": "next",
    "another_approach": "another approach",
    "however": "however",
    "instead": "instead",
    "mistake": "mistake",
    "incorrect": "incorrect",
    "check": "check",
    "verify": "verify",
    "step": "step",
    "markdown_heading": "###",
}


STRUCTURAL_PATTERNS: dict[str, tuple[str, re.Pattern[str]]] = {
    "boxed": ("termination", re.compile(r"\\boxed\s*\{")),
    "code_block": ("code_tool", re.compile(r"```")),
    "markdown_heading": (
        "structure",
        re.compile(r"(?m)^\s*#{2,}\s+"),
    ),
    "numbered_item": (
        "structure",
        re.compile(r"(?m)^\s*\d+[.)]\s+"),
    ),
    "bullet_item": (
        "structure",
        re.compile(r"(?m)^\s*[-*+]\s+"),
    ),
    "display_math": (
        "structure",
        re.compile(r"\$\$|\\\["),
    ),
}


def _normalise_dictionary(
    extra: Mapping[str, Iterable[str]] | None,
) -> dict[str, tuple[str, ...]]:
    merged: dict[str, list[str]] = {
        category: list(phrases)
        for category, phrases in DEFAULT_BEHAVIOR_MARKERS.items()
    }
    if extra:
        for raw_category, raw_phrases in extra.items():
            category = str(raw_category).strip().lower().replace(" ", "_")
            if not category:
                continue
            if isinstance(raw_phrases, str):
                phrases = [raw_phrases]
            else:
                phrases = [str(phrase) for phrase in raw_phrases]
            merged.setdefault(category, []).extend(phrases)

    result: dict[str, tuple[str, ...]] = {}
    for category, phrases in merged.items():
        seen: set[str] = set()
        cleaned: list[str] = []
        for phrase in phrases:
            phrase = str(phrase).strip()
            key = phrase.casefold()
            if phrase and key not in seen:
                cleaned.append(phrase)
                seen.add(key)
        result[category] = tuple(cleaned)
    return result


def _surface_forms(phrase: str) -> tuple[str, ...]:
    forms = {phrase}
    if any(char.isalpha() for char in phrase):
        forms.add(phrase.lower())
        forms.add(phrase[:1].upper() + phrase[1:])
    with_boundaries = set(forms)
    for form in forms:
        with_boundaries.add(" " + form)
        with_boundaries.add("\n" + form)
    return tuple(sorted(with_boundaries, key=lambda item: (len(item), item)))


def _encode(tokenizer: Any, text: str) -> tuple[int, ...]:
    try:
        values = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        values = tokenizer.encode(text)
    return tuple(int(value) for value in values)


def _encoded_variants(tokenizer: Any, phrase: str) -> tuple[tuple[int, ...], ...]:
    seen: set[tuple[int, ...]] = set()
    variants: list[tuple[int, ...]] = []
    for surface in _surface_forms(phrase):
        encoded = _encode(tokenizer, surface)
        if encoded and encoded not in seen:
            variants.append(encoded)
            seen.add(encoded)
    return tuple(variants)


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    escaped = re.escape(phrase)
    left = r"(?<!\w)" if phrase[:1].isalnum() else ""
    right = r"(?!\w)" if phrase[-1:].isalnum() else ""
    return re.compile(left + escaped + right, re.IGNORECASE)


def _decoded_piece(tokenizer: Any, token_id: int) -> str:
    try:
        return str(
            tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
    except Exception:
        try:
            return str(tokenizer.convert_ids_to_tokens(int(token_id)))
        except Exception:
            return str(token_id)


def _start_token_ids(
    tokenizer: Any,
    variants: Iterable[tuple[int, ...]],
) -> tuple[int, ...]:
    token_ids: set[int] = set()
    for variant in variants:
        if not variant:
            continue
        token_id = int(variant[0])
        # A variant beginning with a standalone newline/space measures generic
        # whitespace rather than the marker.  Leading-space BPE tokens decode to
        # a non-empty word and are retained.
        if _decoded_piece(tokenizer, token_id).strip():
            token_ids.add(token_id)
    return tuple(sorted(token_ids))


def repetition_continuation_candidates(
    token_ids: Iterable[int],
    n: int = 4,
) -> list[tuple[int, ...]]:
    """Return previously observed continuations available at every position.

    At response position ``i``, the tuple contains tokens that have previously
    followed the same ``n-1`` token suffix.  If the emitted token is in this
    tuple, generation has continued an already-seen n-gram.
    """

    tokens = [int(token) for token in token_ids]
    n = max(int(n), 1)
    histories: dict[tuple[int, ...], set[int]] = defaultdict(set)
    candidates: list[tuple[int, ...]] = []
    for index, token in enumerate(tokens):
        if n == 1:
            prefix: tuple[int, ...] = ()
        elif index >= n - 1:
            prefix = tuple(tokens[index - n + 1 : index])
        else:
            candidates.append(())
            continue
        candidates.append(tuple(sorted(histories.get(prefix, set()))))
        histories[prefix].add(token)
    return candidates


class RolloutBehaviorAnalyzer:
    """Token-aware occurrence analysis and tokenizer-specific marker sets."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        extra_markers: Mapping[str, Iterable[str]] | None = None,
        focus_markers: Mapping[str, str] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.categories = _normalise_dictionary(extra_markers)
        self.focus_markers = dict(DEFAULT_FOCUS_MARKERS)
        if focus_markers:
            self.focus_markers.update(
                {
                    str(name).strip().lower().replace(" ", "_"): str(phrase)
                    for name, phrase in focus_markers.items()
                    if str(name).strip() and str(phrase).strip()
                }
            )
        self._category_variants: dict[
            str, dict[str, tuple[tuple[int, ...], ...]]
        ] = {
            category: {
                phrase: _encoded_variants(tokenizer, phrase)
                for phrase in phrases
            }
            for category, phrases in self.categories.items()
        }
        self._focus_variants = {
            name: _encoded_variants(tokenizer, phrase)
            for name, phrase in self.focus_markers.items()
        }
        self._category_patterns = {
            category: {
                phrase: _phrase_pattern(phrase) for phrase in phrases
            }
            for category, phrases in self.categories.items()
        }

    def probability_token_sets(
        self,
        *,
        eos_token_ids: Iterable[int] = (),
    ) -> dict[str, tuple[int, ...]]:
        sets: dict[str, set[int]] = {}
        for category, phrase_map in self._category_variants.items():
            ids: set[int] = set()
            for variants in phrase_map.values():
                ids.update(_start_token_ids(self.tokenizer, variants))
            sets[f"category/{category}"] = ids
        for name, variants in self._focus_variants.items():
            sets[f"marker/{name}"] = set(
                _start_token_ids(self.tokenizer, variants)
            )

        eos_ids = {int(token_id) for token_id in eos_token_ids}
        sets["marker/eos"] = eos_ids
        sets.setdefault("category/termination", set()).update(eos_ids)
        sets["category/termination"].update(sets.get("marker/boxed", set()))
        return {
            name: tuple(sorted(token_ids))
            for name, token_ids in sets.items()
            if token_ids
        }

    def manifest(
        self,
        *,
        eos_token_ids: Iterable[int] = (),
    ) -> dict[str, Any]:
        token_sets = self.probability_token_sets(eos_token_ids=eos_token_ids)
        return {
            "version": 1,
            "measurement": (
                "Each probability set contains possible first tokens of a phrase; "
                "multi-token phrase probabilities are not claimed."
            ),
            "categories": {
                category: list(phrases)
                for category, phrases in self.categories.items()
            },
            "focus_markers": dict(self.focus_markers),
            "probability_token_sets": {
                name: [
                    {
                        "token_id": token_id,
                        "token": _decoded_piece(self.tokenizer, token_id),
                    }
                    for token_id in token_ids
                ]
                for name, token_ids in token_sets.items()
            },
        }

    def _token_position_from_char(self, text: str, char_position: int) -> int:
        if char_position <= 0:
            return 1
        try:
            prefix = _encode(self.tokenizer, text[:char_position])
            return len(prefix) + 1
        except Exception:
            return 1

    def marker_start_records(
        self,
        token_ids: Iterable[int],
        text: str,
        *,
        eos_token_ids: Iterable[int] = (),
    ) -> list[dict[str, Any]]:
        """Return every marker start as a token-aligned diagnostic record.

        The occurrence analysis above counts surface phrases.  Post-hoc OPD
        diagnostics need a deterministic token position at which to attach the
        next-token loss/advantage.  We attach a multi-token phrase to its first
        emitted token and merge overlapping phrases from the same category at
        the same position.  Positions are one-indexed within the completion.

        This is deliberately a *marker-start* annotation, not a claim that the
        whole surrounding reasoning span belongs to the category.
        """

        tokens = [int(token_id) for token_id in token_ids]
        grouped: dict[tuple[str, int], set[str]] = defaultdict(set)
        end_positions: dict[tuple[str, int], int] = {}

        def record_match(
            category: str,
            position: int,
            end_position: int,
            marker: str,
        ) -> None:
            key = (str(category), int(position))
            grouped[key].add(str(marker))
            end_positions[key] = max(
                int(end_positions.get(key, position)),
                min(int(end_position), len(tokens)),
            )

        for category, phrase_map in self._category_patterns.items():
            for phrase, pattern in phrase_map.items():
                for match in pattern.finditer(text):
                    position = self._token_position_from_char(text, match.start())
                    end_position = max(
                        position,
                        len(_encode(self.tokenizer, text[: match.end()])),
                    )
                    if 1 <= position <= len(tokens):
                        record_match(
                            str(category), position, end_position, str(phrase)
                        )

        for name, (category, pattern) in STRUCTURAL_PATTERNS.items():
            for match in pattern.finditer(text):
                position = self._token_position_from_char(text, match.start())
                end_position = max(
                    position,
                    len(_encode(self.tokenizer, text[: match.end()])),
                )
                if 1 <= position <= len(tokens):
                    record_match(
                        str(category), position, end_position, f"<{name}>"
                    )

        eos_set = {int(token_id) for token_id in eos_token_ids}
        for index, token_id in enumerate(tokens, start=1):
            if token_id in eos_set:
                record_match("termination", index, index, "<EOS>")

        return [
            {
                "category": category,
                "response_position": position,
                "response_end_position": end_positions[(category, position)],
                "markers": sorted(markers),
            }
            for (category, position), markers in sorted(
                grouped.items(), key=lambda item: (item[0][1], item[0][0])
            )
        ]

    def exclusive_marker_span_labels(
        self,
        token_ids: Iterable[int],
        text: str,
        *,
        eos_token_ids: Iterable[int] = (),
        category_priority: Iterable[str] = (),
        other_label: str = "other",
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Assign each emitted token to at most one matched marker span.

        This annotation is used for category-level loss and logit-gradient
        mass. Overlapping phrases are resolved by ``category_priority`` and
        all non-marker tokens are labelled ``other``. Consequently every
        reported mass share has an explicit denominator and category shares
        (including ``other``) add to one.
        """

        tokens = [int(token_id) for token_id in token_ids]
        marker_rows = self.marker_start_records(
            tokens,
            text,
            eos_token_ids=eos_token_ids,
        )
        rank = {
            str(category): index
            for index, category in enumerate(category_priority)
        }
        candidates: dict[int, list[str]] = defaultdict(list)
        for row in marker_rows:
            start = int(row["response_position"])
            end = int(row.get("response_end_position", start))
            for position in range(max(start, 1), min(end, len(tokens)) + 1):
                candidates[position].append(str(row["category"]))

        labels = [str(other_label)] * len(tokens)
        for position, categories in candidates.items():
            labels[position - 1] = min(
                set(categories),
                key=lambda category: (rank.get(category, len(rank)), category),
            )
        return labels, marker_rows

    def exclusive_marker_start_labels(
        self,
        token_ids: Iterable[int],
        text: str,
        *,
        eos_token_ids: Iterable[int] = (),
        category_priority: Iterable[str] = (),
        other_label: str = "other",
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Assign one exclusive category to every emitted token.

        Only marker-start positions receive a named behaviour category.  Every
        other response token receives ``other_label``.  Exclusive labels make
        category loss-mass shares add to exactly one; the returned marker rows
        preserve all overlapping surface matches for inspection.
        """

        tokens = [int(token_id) for token_id in token_ids]
        marker_rows = self.marker_start_records(
            tokens,
            text,
            eos_token_ids=eos_token_ids,
        )
        rank = {
            str(category): index
            for index, category in enumerate(category_priority)
        }
        by_position: dict[int, list[str]] = defaultdict(list)
        for row in marker_rows:
            by_position[int(row["response_position"])].append(
                str(row["category"])
            )

        labels = [str(other_label)] * len(tokens)
        for position, categories in by_position.items():
            selected = min(
                set(categories),
                key=lambda category: (rank.get(category, len(rank)), category),
            )
            if 1 <= position <= len(labels):
                labels[position - 1] = selected
        return labels, marker_rows

    def analyze(
        self,
        token_ids: Iterable[int],
        text: str,
        *,
        eos_token_ids: Iterable[int] = (),
        repetition_ngram_size: int = 4,
    ) -> dict[str, Any]:
        tokens = [int(token) for token in token_ids]

        category_details: dict[str, dict[str, dict[str, Any]]] = {
            category: {} for category in self.categories
        }
        for category, phrase_map in self._category_patterns.items():
            for phrase, pattern in phrase_map.items():
                matches = list(pattern.finditer(text))
                if matches:
                    category_details[category][phrase] = {
                        "count": len(matches),
                        "first_token_position": self._token_position_from_char(
                            text, matches[0].start()
                        ),
                    }

        eos_set = {int(token_id) for token_id in eos_token_ids}
        eos_positions = [
            index for index, token_id in enumerate(tokens) if token_id in eos_set
        ]
        if eos_positions:
            category_details.setdefault("termination", {})["<EOS>"] = {
                "count": len(eos_positions),
                "first_token_position": eos_positions[0] + 1,
            }

        structural: dict[str, dict[str, Any]] = {}
        for name, (category, pattern) in STRUCTURAL_PATTERNS.items():
            matches = list(pattern.finditer(text))
            if not matches:
                continue
            first_position = self._token_position_from_char(
                text, matches[0].start()
            )
            structural[name] = {
                "count": len(matches),
                "first_token_position": first_position,
            }
            category_details.setdefault(category, {})[f"<{name}>"] = {
                "count": len(matches),
                "first_token_position": first_position,
            }

        category_records: dict[str, dict[str, Any]] = {}
        denominator = max(len(tokens), 1)
        for category in self.categories:
            matches = category_details.get(category, {})
            count = sum(int(item["count"]) for item in matches.values())
            first_positions = [
                int(item["first_token_position"]) for item in matches.values()
            ]
            first = min(first_positions) if first_positions else None
            category_records[category] = {
                "count": count,
                "document_hit": bool(count),
                "first_token_position": first,
                "first_relative_position": (
                    float(first) / float(denominator) if first is not None else None
                ),
                "matched_markers": matches,
            }

        continuation_candidates = repetition_continuation_candidates(
            tokens, repetition_ngram_size
        )
        eligible = [
            index
            for index, candidates in enumerate(continuation_candidates)
            if candidates
        ]
        repeated = [
            index
            for index in eligible
            if tokens[index] in continuation_candidates[index]
        ]
        return {
            "version": 1,
            "position_method": "retokenized decoded prefix",
            "token_count": len(tokens),
            "categories": category_records,
            "structure": structural,
            "repetition_continuation": {
                "ngram_size": int(repetition_ngram_size),
                "eligible_position_count": len(eligible),
                "eligible_position_fraction": float(len(eligible))
                / float(denominator),
                "actual_continuation_count": len(repeated),
                "actual_continuation_fraction": float(len(repeated))
                / float(denominator),
                "actual_given_eligible_fraction": float(len(repeated))
                / float(max(len(eligible), 1)),
                "first_actual_continuation_position": (
                    repeated[0] + 1 if repeated else None
                ),
            },
        }


def aggregate_occurrence_logs(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {}
    logs: dict[str, float] = {}
    categories = sorted(
        {
            category
            for record in records
            for category in record.get("categories", {})
        }
    )
    for category in categories:
        items = [
            record.get("categories", {}).get(category, {})
            for record in records
        ]
        hits = [item for item in items if bool(item.get("document_hit", False))]
        logs[f"behavior_occurrence/{category}_document_fraction"] = float(
            len(hits)
        ) / float(len(items))
        logs[f"behavior_occurrence/{category}_mean_count"] = sum(
            float(item.get("count", 0)) for item in items
        ) / float(len(items))
        total_tokens = sum(
            max(int(record.get("token_count", 0)), 0) for record in records
        )
        total_count = sum(float(item.get("count", 0)) for item in items)
        logs[f"behavior_occurrence/{category}_density_per_1k"] = (
            1000.0 * total_count / float(total_tokens)
            if total_tokens > 0
            else 0.0
        )
        document_densities = [
            1000.0
            * float(item.get("count", 0))
            / float(max(int(record.get("token_count", 0)), 1))
            for record, item in zip(records, items, strict=True)
        ]
        logs[f"behavior_occurrence/{category}_mean_document_density_per_1k"] = (
            sum(document_densities) / float(len(document_densities))
        )
        relative = [
            float(item["first_relative_position"])
            for item in hits
            if item.get("first_relative_position") is not None
        ]
        logs[f"behavior_occurrence/{category}_mean_first_relative_position"] = (
            sum(relative) / float(len(relative)) if relative else -1.0
        )

    repetition = [
        record.get("repetition_continuation", {}) for record in records
    ]
    for name in (
        "eligible_position_fraction",
        "actual_continuation_fraction",
        "actual_given_eligible_fraction",
    ):
        logs[f"behavior_occurrence/repetition_{name}"] = sum(
            float(item.get(name, 0.0)) for item in repetition
        ) / float(len(repetition))
    return logs


def compact_behavior_summary(record: dict[str, Any]) -> dict[str, Any]:
    categories: dict[str, Any] = {}
    for category, item in record.get("categories", {}).items():
        if int(item.get("count", 0)) <= 0:
            continue
        categories[category] = {
            "count": int(item["count"]),
            "first_token_position": item.get("first_token_position"),
        }
    repetition = record.get("repetition_continuation", {})
    return {
        "categories": categories,
        "repetition_continuation": {
            "eligible_position_fraction": repetition.get(
                "eligible_position_fraction", 0.0
            ),
            "actual_continuation_fraction": repetition.get(
                "actual_continuation_fraction", 0.0
            ),
            "first_actual_continuation_position": repetition.get(
                "first_actual_continuation_position"
            ),
        },
    }
