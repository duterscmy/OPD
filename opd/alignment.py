from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AlignmentGroup:
    student_indices: list[int]
    teacher_indices: list[int]
    start_char: int
    end_char: int


def _incremental_offsets(tokenizer: Any, token_ids: list[int]) -> tuple[str, list[tuple[int, int]]]:
    """Build character offsets for the actual generated token IDs.

    Incremental decoding is slower than a fast-tokenizer offset mapping, but it follows the actual sampled IDs and
    is robust to context-sensitive retokenization differences. Special tokens that add no text receive zero-width spans.
    """
    offsets: list[tuple[int, int]] = []
    previous = ""
    for i in range(1, len(token_ids) + 1):
        current = tokenizer.decode(
            token_ids[:i], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if not current.startswith(previous):
            # Rare cleanup mismatch. Use the longest common prefix as a conservative boundary.
            common = 0
            for a, b in zip(previous, current):
                if a != b:
                    break
                common += 1
            start = common
        else:
            start = len(previous)
        offsets.append((start, len(current)))
        previous = current
    return previous, offsets


def tokenization_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    if getattr(tokenizer, "is_fast", False):
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]]
        return list(encoded["input_ids"]), offsets
    ids = tokenizer.encode(text, add_special_tokens=False)
    _, offsets = _incremental_offsets(tokenizer, ids)
    return ids, offsets


def build_text_span_alignment(
    student_tokenizer: Any,
    teacher_tokenizer: Any,
    student_token_ids: list[int],
    minimum_aligned_chars: int = 1,
) -> tuple[str, list[int], list[AlignmentGroup]]:
    completion_text, student_offsets = _incremental_offsets(student_tokenizer, student_token_ids)
    teacher_ids, teacher_offsets = tokenization_with_offsets(teacher_tokenizer, completion_text)

    # Drop zero-width spans (typically EOS/special tokens) from text matching.
    s_items = [(idx, span) for idx, span in enumerate(student_offsets) if span[1] > span[0]]
    t_items = [(idx, span) for idx, span in enumerate(teacher_offsets) if span[1] > span[0]]

    groups: list[AlignmentGroup] = []
    s_begin = t_begin = 0
    s_cursor = t_cursor = 0
    last_boundary = 0

    while s_cursor < len(s_items) and t_cursor < len(t_items):
        s_end = s_items[s_cursor][1][1]
        t_end = t_items[t_cursor][1][1]
        if s_end == t_end:
            if s_end - last_boundary >= minimum_aligned_chars:
                groups.append(
                    AlignmentGroup(
                        student_indices=[x[0] for x in s_items[s_begin : s_cursor + 1]],
                        teacher_indices=[x[0] for x in t_items[t_begin : t_cursor + 1]],
                        start_char=last_boundary,
                        end_char=s_end,
                    )
                )
            last_boundary = s_end
            s_cursor += 1
            t_cursor += 1
            s_begin = s_cursor
            t_begin = t_cursor
        elif s_end < t_end:
            s_cursor += 1
        else:
            t_cursor += 1

    return completion_text, teacher_ids, groups
