from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import torch

from .collator import apply_chat_template_ids, apply_chat_template_to_text


@dataclass
class ReflectionDecision:
    has_error: bool | None
    earliest_error_chunk_id: int | None
    earliest_error_span: str
    explanation: str
    raw_output: str
    parsed: dict[str, Any] | None = None


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

Your task is to identify the earliest numbered chunk where the student's mathematical reasoning first becomes wrong.

Important:
The chunks are arbitrary token chunks. A chunk may start or end in the middle of a sentence.
Do NOT mark a chunk as erroneous merely because it is incomplete, truncated, awkward, or starts/ends mid-sentence.
Do NOT mark formatting, wording, or style issues as mathematical errors.
Judge the student's reasoning in context across all chunks.

A chunk should be marked as erroneous only if it introduces one of the following:
- a wrong mathematical setup
- a false formula or theorem application
- an invalid inference
- an arithmetic or algebraic error
- a contradiction with the problem
- an unsupported mathematical claim that is necessary for the solution
- an irrelevant generation that switches to another problem
- an irrecoverable omission that makes the final answer unsupported

If the student has not yet made a clear mathematical claim, do not mark it as an error.
If the student later self-corrects a tentative false start, do not cut before the self-correction.
If the solution is mathematically correct, return has_error=false.
If there is an error, return the earliest chunk id that introduces the mathematical error, not a later consequence.

Return ONLY valid JSON.
Do not use markdown.
Do not output chain-of-thought.
earliest_error_span should be a short plain-text quote, and may be empty if quoting is difficult.

Problem:
{problem}
{reference}
Student solution chunks:
{chunk_text}

Schema for an erroneous solution:
{{
  "has_error": true,
  "earliest_error_chunk_id": 0,
  "earliest_error_span": "",
  "error_type": "wrong_setup | wrong_formula | algebra_error | arithmetic_error | invalid_inference | contradiction | unsupported_claim | irrelevant_generation | irrecoverable_omission",
  "explanation": "brief explanation of the mathematical error"
}}

Schema for a correct solution:
{{
  "has_error": false,
  "earliest_error_chunk_id": null,
  "earliest_error_span": "",
  "error_type": "none",
  "explanation": "The solution is mathematically correct."
}}
"""


def _repair_json_string(s: str) -> str:
    # Common invalid escapes emitted by LLMs inside JSON strings.
    s = s.replace("\\{", "{").replace("\\}", "}")
    s = s.replace("\\_", "_")
    # Remove invalid escape before dollar if present.
    s = s.replace("\\$", "$")
    return s


def extract_json(text: str) -> dict[str, Any] | None:
    original = text.strip()
    if "</think>" in original:
        original = original.split("</think>", 1)[1].strip()
    if original.startswith("<think>"):
        return None
    original = re.sub(r"^```(?:json)?", "", original).strip()
    original = re.sub(r"```$", "", original).strip()
    original = _repair_json_string(original)

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
                    candidate = _repair_json_string(original[start : end + 1])
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break
    return None


def strip_thinking(text: str) -> str:
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if text.startswith("<think>"):
        return ""
    return text


@torch.no_grad()
def judge_batch(
    teacher_model: Any,
    teacher_tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
    use_chat_template: bool = True,
    enable_thinking: bool = False,
    return_prompt_texts: bool = False,
) -> list[ReflectionDecision] | tuple[list[ReflectionDecision], list[str]]:
    encoded_rows = []
    prompt_texts = []
    for prompt in prompts:
        messages = [
            {
                "role": "system",
                "content": "Return only valid JSON. Do not reveal chain-of-thought. Do not output <think>. Do not use markdown.",
            },
            {"role": "user", "content": prompt},
        ]
        prompt_text = apply_chat_template_to_text(
            teacher_tokenizer,
            messages,
            add_generation_prompt=True,
            use_chat_template=use_chat_template,
            enable_thinking=enable_thinking,
        )
        ids = [int(x) for x in teacher_tokenizer.encode(prompt_text, add_special_tokens=False)]
        encoded_rows.append(ids)
        prompt_texts.append(prompt_text)

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

    prompt_width = input_ids.shape[1]
    decisions: list[ReflectionDecision] = []
    for row in outputs[:, prompt_width:]:
        raw = teacher_tokenizer.decode(row, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        raw = strip_thinking(raw)
        parsed = extract_json(raw)
        if parsed is None:
            decisions.append(ReflectionDecision(None, None, "", "JSON parse failure", raw, None))
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
                parsed=parsed,
            )
        )
    if return_prompt_texts:
        return decisions, prompt_texts
    return decisions
