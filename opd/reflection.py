from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import torch

from .collator import apply_chat_template_ids


@dataclass
class ReflectionDecision:
    has_error: bool | None
    earliest_error_chunk_id: int | None
    earliest_error_span: str
    explanation: str
    raw_output: str


def split_token_chunks(tokenizer: Any, token_ids: list[int], chunk_size: int) -> list[dict[str, Any]]:
    chunks = []
    for start in range(0, len(token_ids), chunk_size):
        ids = token_ids[start : start + chunk_size]
        chunks.append(
            {
                "id": len(chunks),
                "start_token": start,
                "end_token": start + len(ids),
                "text": tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
            }
        )
    return chunks


def build_reflection_prompt(
    problem: str,
    chunks: list[dict[str, Any]],
    reference_solution: str = "",
    use_reference: bool = True,
) -> str:
    reference = ""
    if use_reference and reference_solution:
        reference = f"\nReference solution:\n{reference_solution}\n"
    chunk_text = "\n".join(f"[{c['id']}] {c['text']}" for c in chunks)
    return f"""You are a strict mathematical reasoning verifier.

Identify the earliest numbered chunk in which the student's reasoning first becomes mathematically wrong,
unsupported, contradictory, or irrecoverably incomplete. The erroneous chunk itself must not be used for
knowledge-distillation training; only chunks before it are considered correct.

Rules:
- Ignore style and formatting issues.
- If the whole solution is mathematically correct, return has_error=false.
- If there is an error, return the earliest erroneous chunk id, not a later consequence.
- earliest_error_span must be a short verbatim quote from that numbered chunk.
- Return JSON only, with no markdown and no text outside JSON.

Problem:
{problem}
{reference}
Student solution chunks:
{chunk_text}

Schema:
{{
  "has_error": true,
  "earliest_error_chunk_id": 0,
  "earliest_error_span": "verbatim quote",
  "explanation": "brief mathematical explanation"
}}

For a fully correct solution:
{{
  "has_error": false,
  "earliest_error_chunk_id": null,
  "earliest_error_span": "",
  "explanation": "The solution is correct."
}}
"""


def extract_json(text: str) -> dict[str, Any] | None:
    original = text.strip()
    if "</think>" in original:
        original = original.split("</think>", 1)[1].strip()
    original = re.sub(r"^```(?:json)?", "", original).strip()
    original = re.sub(r"```$", "", original).strip()
    try:
        return json.loads(original)
    except Exception:
        pass

    for start, char in enumerate(original):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escape = False
        for end in range(start, len(original)):
            c = original[end]
            if in_string:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_string = False
                continue
            if c == '"':
                in_string = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(original[start : end + 1])
                    except Exception:
                        break
    return None


@torch.no_grad()
def judge_batch(
    teacher_model: Any,
    teacher_tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
) -> list[ReflectionDecision]:
    encoded_rows = []
    for prompt in prompts:
        messages = [
            {
                "role": "system",
                "content": "Return only valid JSON. Do not reveal chain-of-thought or use markdown.",
            },
            {"role": "user", "content": prompt},
        ]
        encoded_rows.append(apply_chat_template_ids(teacher_tokenizer, messages, add_generation_prompt=True))

    if teacher_tokenizer.pad_token_id is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    pad_id = int(teacher_tokenizer.pad_token_id)
    width = max(len(row) for row in encoded_rows)
    padded = [[pad_id] * (width - len(row)) + row for row in encoded_rows]
    masks = [[0] * (width - len(row)) + [1] * len(row) for row in encoded_rows]

    device = next(teacher_model.parameters()).device
    input_ids = torch.tensor(padded, dtype=torch.long, device=device)
    attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
    outputs = teacher_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_id,
        eos_token_id=teacher_tokenizer.eos_token_id,
        use_cache=True,
    )

    decisions = []
    for row in outputs[:, width:]:
        raw = teacher_tokenizer.decode(row, skip_special_tokens=True)
        parsed = extract_json(raw)
        if parsed is None:
            decisions.append(ReflectionDecision(None, None, "", "JSON parse failure", raw))
            continue
        has_error = parsed.get("has_error")
        if not isinstance(has_error, bool):
            has_error = None
        chunk_id = parsed.get("earliest_error_chunk_id")
        try:
            chunk_id = int(chunk_id) if chunk_id is not None else None
        except Exception:
            chunk_id = None
        decisions.append(
            ReflectionDecision(
                has_error=has_error,
                earliest_error_chunk_id=chunk_id,
                earliest_error_span=str(parsed.get("earliest_error_span", "")),
                explanation=str(parsed.get("explanation", "")),
                raw_output=raw,
            )
        )
    return decisions
