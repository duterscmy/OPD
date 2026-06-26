from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AlignmentGroup:
    start_char: int
    end_char: int
    student_indices: list[int]
    teacher_indices: list[int]


def _token_char_spans(tokenizer: Any, ids: list[int]) -> tuple[str, list[tuple[int, int]]]:
    pieces = []
    spans = []
    cursor = 0
    for tid in ids:
        piece = tokenizer.decode([int(tid)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        pieces.append(piece)
        start = cursor
        cursor += len(piece)
        spans.append((start, cursor))
    return "".join(pieces), spans


def build_text_span_alignment(
    student_tokenizer: Any,
    teacher_tokenizer: Any,
    student_ids: list[int],
    minimum_aligned_chars: int = 1,
) -> tuple[str, list[int], list[AlignmentGroup]]:
    """Greedy character-span alignment for cross-tokenizer sampled RKL.

    It decodes the student completion text, re-tokenizes it with the teacher tokenizer,
    and groups tokens whose decoded character spans overlap.
    """
    student_ids = [int(x) for x in student_ids]
    text, s_spans = _token_char_spans(student_tokenizer, student_ids)
    teacher_ids = [int(x) for x in teacher_tokenizer.encode(text, add_special_tokens=False)]
    _, t_spans = _token_char_spans(teacher_tokenizer, teacher_ids)

    groups: list[AlignmentGroup] = []
    for si, (ss, se) in enumerate(s_spans):
        if se <= ss:
            continue
        t_idxs = []
        for ti, (ts, te) in enumerate(t_spans):
            overlap = max(0, min(se, te) - max(ss, ts))
            if overlap >= int(minimum_aligned_chars):
                t_idxs.append(ti)
        if t_idxs:
            groups.append(AlignmentGroup(ss, se, [si], t_idxs))

    return text, teacher_ids, groups
