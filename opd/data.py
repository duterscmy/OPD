from __future__ import annotations

import json
import re
from typing import Any

from datasets import Dataset, load_dataset


def _text_from_message(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("content", "value", "text", "response"):
            if isinstance(value.get(key), str):
                return value[key]
    if isinstance(value, list):
        for item in reversed(value):
            text = _text_from_message(item)
            if text:
                return text
    return ""


def _best_code_response(example: dict[str, Any]) -> str:
    chosen = example.get("chosen")
    if chosen:
        text = _text_from_message(chosen)
        if text:
            return text

    responses = example.get("responses") or []
    annotations = example.get("annotations") or []
    ratings: dict[str, float] = {}
    for row in annotations:
        if isinstance(row, dict):
            model = str(row.get("model", ""))
            try:
                ratings[model] = float(row.get("rating", -1))
            except Exception:
                ratings[model] = -1
    candidates: list[tuple[float, str]] = []
    for row in responses:
        if isinstance(row, dict):
            model = str(row.get("model", ""))
            response = str(row.get("response", ""))
            if response:
                candidates.append((ratings.get(model, -1), response))
    if candidates:
        return max(candidates, key=lambda x: x[0])[1]
    return _text_from_message(example.get("response") or example.get("output"))


def _normalise_role(role: str) -> str:
    role = role.lower().strip()
    if role in {"human", "user"}:
        return "user"
    if role in {"assistant", "gpt", "model", "function_call"}:
        return "assistant"
    if role in {"system"}:
        return "system"
    if role in {"tool", "function", "function_response", "observation"}:
        return "user"
    return "user"


def _parse_glaive_chat(chat: Any, system: str = "") -> list[dict[str, str]]:
    if isinstance(chat, list):
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        for item in chat:
            if not isinstance(item, dict):
                continue
            role = _normalise_role(str(item.get("role") or item.get("from") or "user"))
            content = _text_from_message(item)
            if content:
                messages.append({"role": role, "content": content})
        return messages

    text = str(chat or "")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    pattern = re.compile(
        r"(?im)^(USER|ASSISTANT|FUNCTION RESPONSE|FUNCTION|TOOL|SYSTEM)\s*:\s*"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        if text:
            messages.append({"role": "user", "content": text})
        return messages
    for i, match in enumerate(matches):
        role_raw = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if not content:
            continue
        role = _normalise_role(role_raw.replace(" ", "_"))
        if role_raw.upper() in {"FUNCTION RESPONSE", "FUNCTION", "TOOL"}:
            content = f"[{role_raw.title()}]\n{content}"
        messages.append({"role": role, "content": content})
    return messages


def _prompt_target_from_messages(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], str]:
    # Use the first assistant turn as the reference target. This keeps a valid prompt-ending-in-user structure.
    for i, message in enumerate(messages):
        if message["role"] == "assistant":
            prompt = messages[:i]
            if prompt and prompt[-1]["role"] == "user":
                return prompt, message["content"]
    # Fallback for malformed samples.
    users = [m for m in messages if m["role"] == "user"]
    prompt = messages if messages else [{"role": "user", "content": "Please respond."}]
    if prompt[-1]["role"] == "assistant":
        target = prompt[-1]["content"]
        prompt = prompt[:-1]
    else:
        target = ""
    if not prompt and users:
        prompt = [users[0]]
    return prompt, target


def adapt_example(example: dict[str, Any], adapter: str, math_prompt_template: str) -> dict[str, Any]:
    if adapter == "math":
        problem = str(example.get("problem") or example.get("question") or example.get("prompt") or "")
        solution = str(example.get("solution") or example.get("answer") or example.get("response") or "")
        prompt = math_prompt_template.format(problem=problem)
        prompt_messages = [{"role": "user", "content": prompt}]
        return {
            "prompt_messages": prompt_messages,
            "target_text": solution,
            "problem": problem,
            "reference_solution": solution,
            "task_type": "math",
        }

    if adapter == "code":
        instruction = str(example.get("instruction") or example.get("prompt") or example.get("question") or "")
        target = _best_code_response(example)
        return {
            "prompt_messages": [{"role": "user", "content": instruction}],
            "target_text": target,
            "problem": instruction,
            "reference_solution": target,
            "task_type": "code",
        }

    if adapter == "function_calling":
        system = str(example.get("system") or "")
        raw = example.get("conversations") or example.get("messages") or example.get("chat")
        messages = _parse_glaive_chat(raw, system=system)
        prompt_messages, target = _prompt_target_from_messages(messages)
        return {
            "prompt_messages": prompt_messages,
            "target_text": target,
            "problem": json.dumps(prompt_messages, ensure_ascii=False),
            "reference_solution": target,
            "task_type": "function_calling",
        }

    if adapter == "auto":
        if "problem" in example or "solution" in example:
            return adapt_example(example, "math", math_prompt_template)
        if "instruction" in example:
            return adapt_example(example, "code", math_prompt_template)
        return adapt_example(example, "function_calling", math_prompt_template)

    raise ValueError(f"Unknown dataset adapter: {adapter}")


def load_training_dataset(cfg: dict[str, Any]) -> Dataset:
    ds = load_dataset(
        cfg["dataset_name"],
        cfg.get("dataset_config"),
        split=cfg["dataset_split"],
    )
    ds = ds.shuffle(seed=cfg["shuffle_seed"])
    max_samples = cfg.get("max_train_samples")
    if max_samples is not None:
        ds = ds.select(range(min(int(max_samples), len(ds))))

    adapter = cfg["dataset_adapter"]
    template = cfg["math_prompt_template"]

    def mapper(example: dict[str, Any]) -> dict[str, Any]:
        return adapt_example(example, adapter, template)

    ds = ds.map(mapper, desc=f"Adapting {cfg['dataset_name']} as {adapter}")
    keep = {"prompt_messages", "target_text", "problem", "reference_solution", "task_type"}
    remove = [column for column in ds.column_names if column not in keep]
    if remove:
        ds = ds.remove_columns(remove)
    ds = ds.filter(
        lambda x: bool(x["prompt_messages"]) and bool(x["problem"]),
        desc="Dropping empty prompts",
    )
    return ds
