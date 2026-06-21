from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def apply_chat_template_ids(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool) -> list[int]:
    if getattr(tokenizer, "chat_template", None):
        kwargs = {
            "tokenize": True,
            "add_generation_prompt": add_generation_prompt,
        }
        try:
            ids = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            ids = tokenizer.apply_chat_template(messages, **kwargs)
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return list(ids)

    lines = []
    for msg in messages:
        role = msg["role"].capitalize()
        lines.append(f"{role}: {msg['content']}")
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
        width = max(len(row) for row in rows)
        ids = []
        masks = []
        for row in rows:
            pad = width - len(row)
            ids.append([self.pad_id] * pad + row)
            masks.append([0] * pad + [1] * len(row))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(masks, dtype=torch.long)

    def _pad_right(
        self, rows: list[list[int]], labels: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        for feature in features:
            prompt_ids = apply_chat_template_ids(
                self.tokenizer, feature["prompt_messages"], add_generation_prompt=True
            )
            prompt_ids = prompt_ids[-self.max_prompt_length :]

            target_ids = self.tokenizer.encode(feature.get("target_text", ""), add_special_tokens=False)
            if self.tokenizer.eos_token_id is not None:
                target_ids = target_ids + [int(self.tokenizer.eos_token_id)]
            target_ids = target_ids[: max(self.max_length - len(prompt_ids), 0)]

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
            "prompt_messages": [feature["prompt_messages"] for feature in features],
            "problem": [feature.get("problem", "") for feature in features],
            "reference_solution": [feature.get("reference_solution", "") for feature in features],
            "task_type": [feature.get("task_type", "unknown") for feature in features],
        }
