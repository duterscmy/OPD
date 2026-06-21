from __future__ import annotations

import ast
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULTS: dict[str, Any] = {
    # Models
    "model_name_or_path": "Qwen/Qwen2.5-Math-1.5B",
    "teacher_model_name_or_path": "Qwen/Qwen3-4B",
    "trust_remote_code": True,
    "dtype": "bfloat16",
    "attn_implementation": "sdpa",
    # Data
    "dataset_name": "AI-MO/NuminaMath-1.5",
    "dataset_config": None,
    "dataset_split": "train",
    "dataset_adapter": "math",
    "max_train_samples": 3200,
    "shuffle_seed": 42,
    "math_prompt_template": (
        "Solve the following mathematics problem. Show your reasoning clearly and put the final answer "
        "in \\boxed{{}}.\n\n{problem}"
    ),
    # Strategy
    "strategy": "esr",  # full | esr | curriculum | reflection
    "prefix_length": 100,
    "full_max_new_tokens": 1024,
    "curriculum_lengths": [50, 100, 200],
    "curriculum_boundaries": [0, 67, 134],
    "reflection_rollout_max_tokens": 1024,
    "reflection_chunk_size": 16,
    "reflection_max_new_tokens": 256,
    "reflection_use_reference": True,
    "reflection_parse_failure": "esr",  # esr | full | skip
    "reflection_fallback_length": 100,
    "reflection_log_path": None,
    # Distillation
    "loss_backend": "auto",  # auto | trl_gjsd | sampled_rkl
    "beta": 1.0,
    "lmbda": 1.0,
    "seq_kd": False,
    "temperature": 0.7,
    "rkl_advantage_clip": None,
    "minimum_aligned_chars": 1,
    # Paper-style training defaults
    "output_dir": "outputs/qwen25_math_15b_to_qwen3_4b_esr100",
    "max_steps": 200,
    "learning_rate": 5e-5,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "save_steps": 50,
    "logging_steps": 1,
    "warmup_ratio": 0.0,
    "weight_decay": 0.0,
    "lr_scheduler_type": "constant",
    "seed": 42,
    "max_length": 4096,
    "max_prompt_length": 2048,
    "gradient_checkpointing": True,
    "use_lora": True,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.0,
    "lora_target_modules": "all-linear",
    "report_to": "none",
    "resume_from_checkpoint": None,
    "save_only_model": False,
    "num_workers": 0,
}


def _parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(value)
    except Exception:
        pass
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def _set_nested(cfg: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cur = cfg
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    cfg = deepcopy(DEFAULTS)
    with Path(path).open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg.update(user_cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got: {item}")
        key, raw = item.split("=", 1)
        _set_nested(cfg, key.strip(), _parse_value(raw.strip()))
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    strategy = cfg["strategy"]
    if strategy not in {"full", "esr", "curriculum", "reflection"}:
        raise ValueError(f"Unsupported strategy: {strategy}")
    if cfg["lmbda"] != 1.0:
        raise ValueError("This experiment package implements pure OPD and expects lmbda=1.0.")
    if strategy == "curriculum":
        lengths = cfg["curriculum_lengths"]
        boundaries = cfg["curriculum_boundaries"]
        if len(lengths) != len(boundaries):
            raise ValueError("curriculum_lengths and curriculum_boundaries must have the same length")
        if not boundaries or boundaries[0] != 0:
            raise ValueError("curriculum_boundaries must start at 0")
        if boundaries != sorted(boundaries):
            raise ValueError("curriculum_boundaries must be sorted")
    if strategy == "reflection" and cfg["reflection_chunk_size"] <= 0:
        raise ValueError("reflection_chunk_size must be positive")
    if cfg["beta"] < 0 or cfg["beta"] > 1:
        raise ValueError("beta must be in [0, 1]")
