from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    # Reproducibility
    "seed": 42,

    # Models
    "model_name_or_path": "Qwen/Qwen2.5-Math-1.5B",
    "teacher_model_name_or_path": "Qwen/Qwen3-4B",
    "trust_remote_code": True,
    "dtype": "bfloat16",  # bfloat16 | float16 | float32
    "attn_implementation": None,

    # Chat template controls
    # For Qwen2.5-Math-1.5B base, keep false. For instruct/chat students, set true.
    "student_use_chat_template": False,
    # For Qwen3 teacher/judge, usually true.
    "teacher_use_chat_template": True,
    "teacher_enable_thinking": False,
    "student_enable_thinking": False,

    # Dataset
    "dataset_name": "AI-MO/NuminaMath-1.5",
    "dataset_split": "train",
    "dataset_adapter": "math",  # math | dapo_math | code | function_calling | generic
    "dataset_text_field": None,
    "dataset_target_field": None,
    "dataset_problem_field": None,
    "dataset_answer_field": None,
    "max_train_examples": 3200,
    "shuffle_dataset": True,
    "system_prompt": "Please reason step by step, and put your final answer within \\boxed{}.",
    "user_prompt_template": "Solve the following mathematics problem. Show your reasoning clearly and put the final answer in \\boxed{{}}.\n\n{problem}",

    # Sequence limits
    "max_length": 4096,
    "max_prompt_length": 2048,

    # Strategies: full | esr | curriculum | reflection | correctness_esr
    "strategy": "esr",
    "prefix_length": 100,
    "full_max_new_tokens": 1024,
    "curriculum_lengths": [50, 100, 200],
    "curriculum_boundaries": [0, 67, 134],

    # Reflection strategy
    "reflection_rollout_max_tokens": 1024,
    "reflection_chunk_size": 32,
    "reflection_max_new_tokens": 512,
    "reflection_use_reference": True,
    "reflection_parse_failure": "esr",  # esr | full | skip
    "reflection_fallback_length": 100,
    "reflection_min_keep_tokens": 0,
    "reflection_log_path": None,

    # Correctness-gated strategy: correct -> full, wrong -> ESR(prefix_length)
    "correctness_rollout_max_tokens": 1024,
    "correctness_wrong_fallback": "esr",  # esr | skip
    "answer_extraction": "auto",  # auto | boxed | answer_marker | last_number

    # Generation
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": 0,
    "rollout_do_sample": True,

    # GKD / loss
    # loss_backend:
    #   auto         -> trl_gjsd if tokenizers match, otherwise sampled_rkl
    #   trl_gjsd     -> original TRL GKD loss, requires same tokenizer
    #   sampled_rkl  -> sampled reverse-KL-style loss for different tokenizers
    #   adaptive_opd -> custom overlap-gated reverse + auxiliary forward KL, requires same tokenizer
    "loss_backend": "auto",
    "lmbda": 1.0,
    "beta": 1.0,
    "seq_kd": False,
    "minimum_aligned_chars": 1,
    "rkl_advantage_clip": None,

    # Adaptive OPD loss. Used only when loss_backend=adaptive_opd.
    # Recommended first trial for your current results:
    #   overlap >= adaptive_overlap_threshold: reverse KL
    #   overlap <  adaptive_overlap_threshold: reverse KL + adaptive_forward_lambda * forward KL
    "adaptive_overlap_threshold": 0.55,
    "adaptive_overlap_low_threshold": 0.0,
    "adaptive_use_low_band": False,
    "adaptive_forward_lambda": 0.10,
    "adaptive_low_forward_lambda": 0.05,
    "adaptive_reverse_top_k": 16,
    "adaptive_forward_top_k": 16,
    "adaptive_overlap_top_k": 16,
    "adaptive_reverse_weight": 1.0,
    "adaptive_forward_weight": 1.0,
    "adaptive_loss_eps": 1.0e-8,
    "adaptive_log_prefix": "adaptive_loss/opd",

    # Training hyperparameters
    "output_dir": "outputs/qwen25_math_15b_to_qwen3_4b_esr100",
    "max_steps": 200,
    "learning_rate": 5e-5,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "save_steps": 50,
    "logging_steps": 1,
    "warmup_ratio": 0.03,
    "warmup_steps": None,
    "weight_decay": 0.0,
    "lr_scheduler_type": "constant",
    "gradient_checkpointing": True,
    "num_workers": 0,
    "report_to": "none",
    "save_only_model": False,
    "resume_from_checkpoint": None,

    # LoRA
    "use_lora": True,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.0,
    "lora_target_modules": "all-linear",

    # Debug logging
    "debug_print_every_step": True,
    "debug_print_samples": 1,
    "debug_print_max_chars": 2500,
    "debug_log_jsonl": None,
}


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def _set_by_dotted_key(cfg: dict[str, Any], key: str, value: Any) -> None:
    cur = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    path = Path(path)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config must be a mapping: {path}")
        cfg = _deep_update(cfg, loaded)
    else:
        raise FileNotFoundError(path)

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        _set_by_dotted_key(cfg, key, _parse_scalar(value))

    # Normalise list-ish lora target modules.
    if isinstance(cfg.get("lora_target_modules"), str):
        if cfg["lora_target_modules"] != "all-linear":
            cfg["lora_target_modules"] = [x.strip() for x in cfg["lora_target_modules"].split(",") if x.strip()]

    return cfg
