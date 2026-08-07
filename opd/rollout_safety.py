from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable


def _normalise_token_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, (list, tuple, set)):
        return [int(item) for item in value if item is not None]
    try:
        return [int(value)]
    except (TypeError, ValueError):
        return []


def _token_for_id(tokenizer: Any, token_id: int) -> str | None:
    try:
        token = tokenizer.convert_ids_to_tokens(int(token_id))
    except Exception:
        return None
    return str(token) if token is not None else None


def _id_for_token(tokenizer: Any, token: str) -> int | None:
    try:
        token_id = tokenizer.convert_tokens_to_ids(token)
    except Exception:
        return None
    if token_id is None:
        return None
    token_id = int(token_id)
    # Some tokenizers return unk_token_id for every unknown string.
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None and token_id == int(unk_id):
        roundtrip = _token_for_id(tokenizer, token_id)
        if roundtrip != token:
            return None
    return token_id


@dataclass(frozen=True)
class RolloutEOSInfo:
    token_ids: tuple[int, ...]
    token_strings: dict[int, str]
    sources: dict[int, tuple[str, ...]]
    student_native_ids: frozenset[int]

    def stop_reason(self, token_id: int | None) -> str:
        if token_id is None:
            return "sequence_end"
        if int(token_id) in self.student_native_ids:
            return "student_eos"
        source_names = self.sources.get(int(token_id), ())
        if any(name.startswith("teacher") for name in source_names):
            return "teacher_eos"
        return "configured_eos"

    def preferred_teacher_eos_id(self) -> int:
        """Choose a teacher EOS represented in the student's vocabulary."""
        teacher_ids = [
            token_id
            for token_id in self.token_ids
            if any(
                source.startswith("teacher")
                for source in self.sources.get(token_id, ())
            )
        ]
        teacher_only_ids = [
            token_id
            for token_id in teacher_ids
            if token_id not in self.student_native_ids
        ]
        if teacher_only_ids:
            return int(teacher_only_ids[0])
        if teacher_ids:
            return int(teacher_ids[0])
        return int(self.token_ids[0])


def resolve_rollout_eos(
    student_tokenizer: Any,
    teacher_tokenizer: Any,
    *,
    student_generation_eos: Any = None,
    teacher_generation_eos: Any = None,
    extra_eos_tokens: Iterable[str] = (),
    include_teacher_eos: bool = True,
) -> RolloutEOSInfo:
    """Resolve student-side IDs for every valid student/teacher EOS token.

    Teacher IDs are mapped by token string rather than copied numerically. This
    remains correct when tokenizers share token strings but use different IDs.
    """

    token_strings: dict[int, str] = {}
    source_sets: dict[int, set[str]] = {}
    student_native_ids: set[int] = set()

    def add_student_id(token_id: int, source: str, *, native: bool = False) -> None:
        token = _token_for_id(student_tokenizer, token_id)
        if token is None:
            return
        token_id = int(token_id)
        token_strings[token_id] = token
        source_sets.setdefault(token_id, set()).add(source)
        if native:
            student_native_ids.add(token_id)

    def add_teacher_id(token_id: int, source: str) -> None:
        token = _token_for_id(teacher_tokenizer, token_id)
        if token is None:
            return
        student_id = _id_for_token(student_tokenizer, token)
        if student_id is None:
            return
        token_strings[student_id] = token
        source_sets.setdefault(student_id, set()).add(source)

    for token_id in _normalise_token_ids(getattr(student_tokenizer, "eos_token_id", None)):
        add_student_id(token_id, "student_tokenizer", native=True)
    for token_id in _normalise_token_ids(student_generation_eos):
        add_student_id(token_id, "student_generation_config", native=True)
    if include_teacher_eos:
        for token_id in _normalise_token_ids(getattr(teacher_tokenizer, "eos_token_id", None)):
            add_teacher_id(token_id, "teacher_tokenizer")
        for token_id in _normalise_token_ids(teacher_generation_eos):
            add_teacher_id(token_id, "teacher_generation_config")
    for token in extra_eos_tokens:
        token = str(token)
        student_id = _id_for_token(student_tokenizer, token)
        if student_id is not None:
            token_strings[student_id] = token
            source_sets.setdefault(student_id, set()).add("configured_extra")

    if not token_strings:
        raise ValueError("Could not resolve any rollout EOS token in the student vocabulary")

    token_ids = tuple(sorted(token_strings))
    sources = {token_id: tuple(sorted(source_sets[token_id])) for token_id in token_ids}
    return RolloutEOSInfo(
        token_ids=token_ids,
        token_strings=token_strings,
        sources=sources,
        student_native_ids=frozenset(student_native_ids),
    )


@dataclass(frozen=True)
class TruncatedCompletion:
    token_ids: list[int]
    raw_tensor_length: int
    stop_token_id: int | None
    stop_reason: str
    hit_horizon: bool
    raw_hit_horizon: bool
    boxed_truncated: bool = False
    raw_boxed_count: int = 0
    raw_repeated_ngram_ratio: float = 0.0
    appended_eos: bool = False

    @property
    def emitted_eos(self) -> bool:
        return self.stop_token_id is not None


def truncate_completion(
    row: Iterable[int],
    *,
    eos_info: RolloutEOSInfo,
    pad_token_id: int | None,
    horizon: int,
) -> TruncatedCompletion:
    """Cut a generated row at the first compatible EOS or the hard horizon.

    EOS is intentionally checked before padding because Qwen2.5 uses the same
    ID for ``pad_token_id`` and its native ``eos_token_id``.
    """

    raw = [int(token) for token in row]
    limited = raw[: max(int(horizon), 0)]
    stop_ids = set(eos_info.token_ids)
    result: list[int] = []
    stop_token_id: int | None = None

    for token in limited:
        if token in stop_ids:
            result.append(token)
            stop_token_id = token
            break
        if pad_token_id is not None and token == int(pad_token_id):
            break
        result.append(token)

    hit_horizon = stop_token_id is None and len(result) >= int(horizon)
    if stop_token_id is not None:
        reason = eos_info.stop_reason(stop_token_id)
    elif hit_horizon:
        reason = "max_length"
    elif pad_token_id is not None and len(result) < len(limited):
        reason = "padding"
    else:
        reason = "sequence_end"

    return TruncatedCompletion(
        token_ids=result,
        raw_tensor_length=len(limited),
        stop_token_id=stop_token_id,
        stop_reason=reason,
        hit_horizon=hit_horizon,
        raw_hit_horizon=hit_horizon,
    )


def repeated_ngram_ratio(token_ids: Iterable[int], n: int = 4) -> float:
    tokens = [int(token) for token in token_ids]
    n = max(int(n), 1)
    total = len(tokens) - n + 1
    if total <= 0:
        return 0.0
    ngrams = [tuple(tokens[index : index + n]) for index in range(total)]
    return float(total - len(set(ngrams))) / float(total)


def first_complete_boxed_line_end(text: str) -> int | None:
    """Return the character offset after the first complete ``\\boxed{...}`` line."""
    marker = "\\boxed{"
    cursor = 0
    while cursor < len(text):
        start = text.find(marker, cursor)
        if start < 0:
            return None
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
            # An incomplete placeholder must not hide a later valid answer.
            cursor = start + len(marker)
            continue
        payload = text[opening_brace + 1 : closing_brace - 1].strip()
        if payload:
            newline = text.find("\n", closing_brace)
            return newline + 1 if newline >= 0 else closing_brace
        # Ignore an echoed empty ``\\boxed{}`` instruction/placeholder.
        cursor = closing_brace
    return None


def truncate_after_first_boxed(
    tokenizer: Any,
    token_ids: Iterable[int],
) -> tuple[list[int], bool, int]:
    """Trim a math completion after the line containing its first boxed answer."""
    tokens = [int(token) for token in token_ids]
    text = tokenizer.decode(
        tokens,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    boxed_count = text.count("\\boxed{")
    char_end = first_complete_boxed_line_end(text)
    if char_end is None:
        return tokens, False, boxed_count

    # Find the smallest token prefix whose decoded text reaches the semantic
    # boundary. Prefix decoding is used instead of concatenating token pieces,
    # which can be incorrect for byte-level tokenizers.
    low, high = 1, len(tokens)
    while low < high:
        middle = (low + high) // 2
        prefix_text = tokenizer.decode(
            tokens[:middle],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if len(prefix_text) >= char_end:
            high = middle
        else:
            low = middle + 1
    trimmed = tokens[:low]
    return trimmed, len(trimmed) < len(tokens), boxed_count


def finalize_math_completion(
    completion: TruncatedCompletion,
    tokenizer: Any,
    *,
    repetition_ngram_size: int,
    truncate_after_boxed_answer: bool,
    append_eos_after_boxed_answer: bool,
    terminal_eos_token_id: int,
    generation_stopped_after_boxed_answer: bool = False,
    rollout_horizon: int | None = None,
) -> TruncatedCompletion:
    """Attach raw diagnostics and apply the optional boxed-answer boundary.

    ``generation_stopped_after_boxed_answer`` distinguishes a real online
    boxed stop from the legacy fallback that trims an already generated tail.
    """
    boxed_prefix, boxed_truncated, raw_boxed_count = truncate_after_first_boxed(
        tokenizer,
        completion.token_ids,
    )
    result = replace(
        completion,
        raw_boxed_count=raw_boxed_count,
        raw_repeated_ngram_ratio=repeated_ngram_ratio(
            completion.token_ids, repetition_ngram_size
        ),
    )
    apply_boxed_boundary = (
        truncate_after_boxed_answer
        and not result.emitted_eos
        and (boxed_truncated or generation_stopped_after_boxed_answer)
    )
    if not apply_boxed_boundary:
        return result

    effective_ids = boxed_prefix
    appended_eos = False
    can_append_eos = (
        rollout_horizon is None or len(boxed_prefix) < int(rollout_horizon)
    )
    if append_eos_after_boxed_answer and can_append_eos:
        effective_ids = boxed_prefix + [int(terminal_eos_token_id)]
        appended_eos = True
    return replace(
        result,
        token_ids=effective_ids,
        stop_reason="boxed_answer",
        hit_horizon=False,
        raw_hit_horizon=(
            False
            if generation_stopped_after_boxed_answer
            else result.raw_hit_horizon
        ),
        boxed_truncated=True,
        appended_eos=appended_eos,
    )
