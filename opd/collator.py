from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any

import torch


def _normalise_messages(messages: Any) -> list[dict[str, str]]:
    if messages is None:
        return [{"role": "user", "content": "Please solve the problem."}]

    if isinstance(messages, str):
        s = messages.strip()
        try:
            return _normalise_messages(json.loads(s))
        except Exception:
            pass
        try:
            return _normalise_messages(ast.literal_eval(s))
        except Exception:
            pass
        return [{"role": "user", "content": s}]

    if isinstance(messages, dict):
        role = str(messages.get("role", "user"))
        content = str(messages.get("content") or messages.get("text") or messages.get("value") or "")
        return [{"role": role, "content": content}]

    if isinstance(messages, list):
        out: list[dict[str, str]] = []
        for item in messages:
            if isinstance(item, dict):
                role = str(item.get("role", "user"))
                content = str(item.get("content") or item.get("text") or item.get("value") or "")
                if content:
                    out.append({"role": role, "content": content})
            elif isinstance(item, str) and item.strip():
                out.append({"role": "user", "content": item.strip()})
        if out:
            return out

    return [{"role": "user", "content": str(messages)}]


def messages_to_plain_text(messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    """Plain prompt for base models. Does not expose role tokens."""
    contents = []
    for msg in messages:
        content = str(msg.get("content", "")).strip()
        if content:
            contents.append(content)
    text = "\n\n".join(contents).rstrip()
    if add_generation_prompt:
        text = text + "\n\n"
    return text


def apply_chat_template_to_text(
    tokenizer: Any,
    messages: Any,
    add_generation_prompt: bool,
    use_chat_template: bool = True,
    enable_thinking: bool = False,
) -> str:
    """Return the actual prompt text.

    Critical implementation detail:
    always call apply_chat_template(..., tokenize=False) and then encode text.
    Some tokenizer versions return dicts for tokenize=True; encoding that dict
    as text caused prompts like {'input_ids': ..., 'attention_mask': ...}.
    """
    messages = _normalise_messages(messages)

    if use_chat_template and getattr(tokenizer, "chat_template", None):
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        if not isinstance(text, str):
            raise TypeError(f"apply_chat_template(tokenize=False) returned {type(text)}")
        return text

    return messages_to_plain_text(messages, add_generation_prompt=add_generation_prompt)


def apply_chat_template_ids(
    tokenizer: Any,
    messages: Any,
    add_generation_prompt: bool,
    use_chat_template: bool = True,
    enable_thinking: bool = False,
) -> list[int]:
    text = apply_chat_template_to_text(
        tokenizer=tokenizer,
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        use_chat_template=use_chat_template,
        enable_thinking=enable_thinking,
    )
    return [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]


@dataclass
class OPDDataCollator:
    tokenizer: Any
    max_length: int = 4096
    max_prompt_length: int = 2048
    use_chat_template: bool = True
    enable_thinking: bool = False

    def __post_init__(self) -> None:
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer needs either pad_token_id or eos_token_id")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_id = int(self.tokenizer.pad_token_id)

    def _pad_left(self, rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        rows = [[int(x) for x in row] for row in rows]
        width = max(len(row) for row in rows)
        ids, masks = [], []
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
        normalised_prompt_messages: list[list[dict[str, str]]] = []
        prompt_texts: list[str] = []

        for feature in features:
            prompt_messages = _normalise_messages(feature.get("prompt_messages"))
            normalised_prompt_messages.append(prompt_messages)

            prompt_text = apply_chat_template_to_text(
                tokenizer=self.tokenizer,
                messages=prompt_messages,
                add_generation_prompt=True,
                use_chat_template=self.use_chat_template,
                enable_thinking=self.enable_thinking,
            )
            prompt_texts.append(prompt_text)
            prompt_ids = [int(x) for x in self.tokenizer.encode(prompt_text, add_special_tokens=False)]
            prompt_ids = prompt_ids[-self.max_prompt_length :]

            target_text = str(feature.get("target_text") or "")
            target_ids = [int(x) for x in self.tokenizer.encode(target_text, add_special_tokens=False)]
            if self.tokenizer.eos_token_id is not None:
                target_ids = target_ids + [int(self.tokenizer.eos_token_id)]

            remain = max(self.max_length - len(prompt_ids), 0)
            target_ids = target_ids[:remain]

            full = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids

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
            "prompt_texts": prompt_texts,
            "problem": [str(feature.get("problem", "")) for feature in features],
            "reference_solution": [str(feature.get("reference_solution", "")) for feature in features],
            "ground_truth": [str(feature.get("ground_truth", "")) for feature in features],
            "task_type": [str(feature.get("task_type", "unknown")) for feature in features],
        }
