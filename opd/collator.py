from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any

import torch


def _normalise_messages(messages: Any) -> list[dict[str, str]]:
    """
    Make prompt_messages robust.

    Expected:
      [{"role": "user", "content": "..."}]

    But datasets may store it as:
      - JSON string
      - Python repr string
      - plain string
      - list of malformed dicts
    """
    if messages is None:
        return [{"role": "user", "content": "Please solve the problem."}]

    # If serialized as JSON / repr string
    if isinstance(messages, str):
        s = messages.strip()

        # Try JSON
        try:
            obj = json.loads(s)
            return _normalise_messages(obj)
        except Exception:
            pass

        # Try Python literal
        try:
            obj = ast.literal_eval(s)
            return _normalise_messages(obj)
        except Exception:
            pass

        # Plain text prompt
        return [{"role": "user", "content": s}]

    # If single dict
    if isinstance(messages, dict):
        role = str(messages.get("role", "user"))
        content = str(
            messages.get("content")
            or messages.get("text")
            or messages.get("value")
            or ""
        )
        return [{"role": role, "content": content}]

    # If list
    if isinstance(messages, list):
        out: list[dict[str, str]] = []
        for item in messages:
            if isinstance(item, dict):
                role = str(item.get("role", "user"))
                content = str(
                    item.get("content")
                    or item.get("text")
                    or item.get("value")
                    or ""
                )
                if content:
                    out.append({"role": role, "content": content})
            elif isinstance(item, str):
                # If list of strings, treat them as user text chunks.
                if item.strip():
                    out.append({"role": "user", "content": item.strip()})
        if out:
            return out

    return [{"role": "user", "content": str(messages)}]


def _ensure_int_ids(tokenizer, ids_or_text: Any) -> list[int]:
    """
    Convert tokenizer output to a flat list[int].
    Handles:
      - list[int]
      - torch.Tensor
      - string
      - nested list
    """
    if isinstance(ids_or_text, torch.Tensor):
        ids_or_text = ids_or_text.detach().cpu().tolist()

    if isinstance(ids_or_text, str):
        return tokenizer.encode(ids_or_text, add_special_tokens=True)

    if isinstance(ids_or_text, int):
        return [int(ids_or_text)]

    if isinstance(ids_or_text, list):
        # Flatten one or more nested list levels.
        flat = []
        stack = list(ids_or_text)
        while stack:
            x = stack.pop(0)
            if isinstance(x, list):
                stack = list(x) + stack
            elif isinstance(x, torch.Tensor):
                stack = x.detach().cpu().tolist() + stack
            elif isinstance(x, int):
                flat.append(int(x))
            elif isinstance(x, str):
                # This should not happen for token ids, but if it does,
                # encode the string instead of letting torch.tensor crash.
                flat.extend(tokenizer.encode(x, add_special_tokens=False))
            else:
                try:
                    flat.append(int(x))
                except Exception:
                    flat.extend(tokenizer.encode(str(x), add_special_tokens=False))
        return flat

    return tokenizer.encode(str(ids_or_text), add_special_tokens=True)


def apply_chat_template_ids(
    tokenizer,
    messages: Any,
    add_generation_prompt: bool,
) -> list[int]:
    """
    Return token ids for a chat prompt.
    Robust against tokenizers returning text despite tokenize=True.
    """
    messages = _normalise_messages(messages)

    if getattr(tokenizer, "chat_template", None):
        kwargs = {
            "tokenize": True,
            "add_generation_prompt": add_generation_prompt,
        }
        try:
            ids = tokenizer.apply_chat_template(
                messages,
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            ids = tokenizer.apply_chat_template(messages, **kwargs)

        return _ensure_int_ids(tokenizer, ids)

    # Fallback for base models without chat template.
    lines = []
    for msg in messages:
        role = str(msg.get("role", "user")).capitalize()
        content = str(msg.get("content", ""))
        lines.append(f"{role}: {content}")

    if add_generation_prompt:
        lines.append("Assistant:")

    text = "\n".join(lines)
    return tokenizer.encode(text, add_special_tokens=True)


@dataclass
class OPDDataCollator:
    tokenizer: Any
    max_length: int = 4096
    max_prompt_length: int = 2048

    def __post_init__(self) -> None:
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer needs either pad_token_id or eos_token_id")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_id = int(self.tokenizer.pad_token_id)

    def _pad_left(self, rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        rows = [[int(x) for x in row] for row in rows]
        width = max(len(row) for row in rows)
        ids = []
        masks = []
        for row in rows:
            pad = width - len(row)
            ids.append([self.pad_id] * pad + row)
            masks.append([0] * pad + [1] * len(row))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(masks, dtype=torch.long)

    def _pad_right(
        self,
        rows: list[list[int]],
        labels: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = [[int(x) for x in row] for row in rows]
        labels = [[int(x) for x in row] for row in labels]

        width = max(len(row) for row in rows)
        out_ids, out_masks, out_labels = [], [], []
        for row, label in zip(rows, labels, strict=True):
            pad = width - len(row)
            out_ids.append(row + [self.pad_id] * pad)
            out_masks.append([1] * len(row) + [0] * pad)
            out_labels.append(label + [-100] * pad)

        return (
            torch.tensor(out_ids, dtype=torch.long),
            torch.tensor(out_masks, dtype=torch.long),
            torch.tensor(out_labels, dtype=torch.long),
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_rows: list[list[int]] = []
        full_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        normalised_prompt_messages = []

        for feature in features:
            prompt_messages = _normalise_messages(feature.get("prompt_messages"))
            normalised_prompt_messages.append(prompt_messages)

            prompt_ids = apply_chat_template_ids(
                self.tokenizer,
                prompt_messages,
                add_generation_prompt=True,
            )
            prompt_ids = prompt_ids[-self.max_prompt_length :]

            target_text = feature.get("target_text", "")
            if target_text is None:
                target_text = ""
            target_text = str(target_text)

            target_ids = self.tokenizer.encode(
                target_text,
                add_special_tokens=False,
            )

            if self.tokenizer.eos_token_id is not None:
                target_ids = target_ids + [int(self.tokenizer.eos_token_id)]

            remain = max(self.max_length - len(prompt_ids), 0)
            target_ids = target_ids[:remain]

            full = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids

            # Final safety check
            prompt_ids = [int(x) for x in prompt_ids]
            full = [int(x) for x in full]
            labels = [int(x) for x in labels]

            prompt_rows.append(prompt_ids)
            full_rows.append(full)
            label_rows.append(labels)

        prompts, prompt_attention_mask = self._pad_left(prompt_rows)
        input_ids, attention_mask, labels = self._pad_right(full_rows, label_rows)

        return {
            "prompts": prompts,
            "prompt_attention_mask": prompt_attention_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_messages": normalised_prompt_messages,
            "problem": [str(feature.get("problem", "")) for feature in features],
            "reference_solution": [
                str(feature.get("reference_solution", "")) for feature in features
            ],
            "task_type": [str(feature.get("task_type", "unknown")) for feature in features],
        }