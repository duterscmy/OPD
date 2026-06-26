from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset

from .answers import extract_answer


def _first_nonempty(row: dict[str, Any], keys: list[str]) -> str:
    for k in keys:
        v = row.get(k)
        if v is not None and str(v).strip():
            return str(v)
    return ""


def _make_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_prompt})
    return msgs


def map_math(row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    problem_key = cfg.get("dataset_problem_field")
    target_key = cfg.get("dataset_target_field")
    answer_key = cfg.get("dataset_answer_field")

    problem = str(row.get(problem_key) if problem_key else "" or "").strip()
    if not problem:
        problem = _first_nonempty(row, ["problem", "question", "prompt", "query", "input"])

    solution = str(row.get(target_key) if target_key else "" or "").strip()
    if not solution:
        solution = _first_nonempty(row, ["solution", "response", "answer", "target", "output"])

    ground_truth = str(row.get(answer_key) if answer_key else "" or "").strip()
    if not ground_truth:
        ground_truth = _first_nonempty(row, ["ground_truth", "gt", "final_answer", "answer"])
    if not ground_truth and solution:
        ground_truth = extract_answer(solution, mode="auto")

    user_prompt = str(cfg["user_prompt_template"]).format(problem=problem)
    messages = _make_messages(str(cfg.get("system_prompt") or ""), user_prompt)

    return {
        "prompt_messages": messages,
        "target_text": solution,
        "problem": problem,
        "reference_solution": solution,
        "ground_truth": ground_truth,
        "task_type": "math",
    }


def map_code(row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    problem = _first_nonempty(row, ["instruction", "prompt", "question", "query", "input"])
    target = _first_nonempty(row, ["response", "chosen", "answer", "output", "completion"])
    if not target and isinstance(row.get("messages"), list):
        for msg in row["messages"]:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                target = str(msg.get("content", ""))
                break
    user_prompt = problem
    messages = _make_messages(str(cfg.get("system_prompt") or ""), user_prompt)
    return {
        "prompt_messages": messages,
        "target_text": target,
        "problem": problem,
        "reference_solution": target,
        "ground_truth": "",
        "task_type": "code",
    }


def map_generic(row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    text_key = cfg.get("dataset_text_field")
    target_key = cfg.get("dataset_target_field")
    prompt = str(row.get(text_key) if text_key else _first_nonempty(row, ["prompt", "question", "input"]))
    target = str(row.get(target_key) if target_key else _first_nonempty(row, ["target", "answer", "response", "output"]))
    messages = _make_messages(str(cfg.get("system_prompt") or ""), prompt)
    return {
        "prompt_messages": messages,
        "target_text": target,
        "problem": prompt,
        "reference_solution": target,
        "ground_truth": extract_answer(target, mode="auto"),
        "task_type": "generic",
    }


def load_training_dataset(cfg: dict[str, Any]) -> Dataset:
    ds = load_dataset(cfg["dataset_name"], split=cfg.get("dataset_split", "train"))
    if bool(cfg.get("shuffle_dataset", True)):
        ds = ds.shuffle(seed=int(cfg.get("seed", 42)))
    max_n = cfg.get("max_train_examples")
    if max_n is not None:
        ds = ds.select(range(min(int(max_n), len(ds))))

    adapter = cfg.get("dataset_adapter", "math")
    if adapter in {"math", "dapo_math"}:
        fn = lambda row: map_math(row, cfg)
    elif adapter in {"code", "function_calling"}:
        fn = lambda row: map_code(row, cfg)
    else:
        fn = lambda row: map_generic(row, cfg)

    mapped = ds.map(fn, remove_columns=ds.column_names)
    return mapped
