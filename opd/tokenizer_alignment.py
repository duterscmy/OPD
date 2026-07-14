
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import torch


@dataclass
class TokenizerAlignment:
    teacher_to_student: Dict[int, int]
    student_to_teacher: Dict[int, int]
    shared_ratio: float
    teacher_only: int
    student_only: int


def build_tokenizer_alignment(student_tokenizer, teacher_tokenizer):
    """
    Build token alignment based on decoded token strings.

    This is designed for Qwen2.5 -> Qwen3 style cases:
    vocabularies are almost identical but not byte-for-byte equal.
    """

    s_vocab = student_tokenizer.get_vocab()
    t_vocab = teacher_tokenizer.get_vocab()

    s_inv = {v: k for k, v in s_vocab.items()}
    t_inv = {v: k for k, v in t_vocab.items()}

    teacher_to_student = {}
    student_to_teacher = {}

    for tid, token in t_inv.items():
        if token in s_vocab:
            sid = s_vocab[token]
            teacher_to_student[tid] = sid
            student_to_teacher[sid] = tid

    shared = len(teacher_to_student)
    total = max(len(t_vocab), len(s_vocab))

    return TokenizerAlignment(
        teacher_to_student=teacher_to_student,
        student_to_teacher=student_to_teacher,
        shared_ratio=shared / total,
        teacher_only=len(t_vocab) - shared,
        student_only=len(s_vocab) - shared,
    )


def map_teacher_topk_to_student(
    teacher_ids: torch.Tensor,
    mapping: Dict[int, int],
):
    """
    teacher_ids:
        [B,T,K]

    return:
        mapped ids in student vocabulary
        invalid ids are -1
    """

    device = teacher_ids.device

    out = torch.full_like(
        teacher_ids,
        -1,
        device=device
    )

    for k, v in mapping.items():
        out[teacher_ids == k] = v

    return out
