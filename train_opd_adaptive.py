#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import yaml

from peft import LoraConfig
from transformers import AutoTokenizer, set_seed
from trl.experimental.gkd import GKDConfig

from opd.collator import OPDDataCollator
from opd.data import load_training_dataset
from opd.adaptive_trainer import AdaptiveKLTrainer


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base and return base."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config_with_base(config_path: str, overrides: list[str]) -> dict[str, Any]:
    """Load YAML config with optional `base_config: path/to/base.yaml`.

    Relative base paths are resolved from:
      1. current working directory
      2. child config directory

    CLI overrides use:
      --set key=value
      --set nested.key=value
    """
    path = Path(config_path)
    raw = yaml.safe_load(path.read_text()) or {}

    base_ref = raw.pop("base_config", None)
    if base_ref is not None:
        base_path = Path(base_ref)
        if not base_path.exists():
            alt = path.parent / base_ref
            if alt.exists():
                base_path = alt

        if not base_path.exists():
            raise FileNotFoundError(
                f"base_config not found: {base_ref}. Tried: {Path(base_ref)} and {path.parent / base_ref}"
            )

        base = yaml.safe_load(base_path.read_text()) or {}
        cfg = _deep_update(base, raw)
        cfg["_loaded_base_config"] = str(base_path)
    else:
        cfg = raw
        cfg["_loaded_base_config"] = None

    cfg["_loaded_config"] = str(path)

    parsed_overrides: dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --set override {item!r}; expected KEY=VALUE")

        key, value = item.split("=", 1)
        try:
            parsed = yaml.safe_load(value)
        except Exception:
            parsed = value

        cur = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = parsed
        parsed_overrides[key] = parsed

    cfg["_cli_overrides"] = parsed_overrides
    return cfg



def find_latest_checkpoint(output_dir: str) -> str | None:
    """Automatically find the latest checkpoint-xxxx directory.

    The checkpoint with the largest numerical suffix is selected.
    Example:
        checkpoint-100
        checkpoint-200
        checkpoint-500  -> selected
    """
    output_path = Path(output_dir)

    if not output_path.exists():
        return None

    candidates = []
    for p in output_path.glob("checkpoint-*"):
        if not p.is_dir():
            continue

        match = re.match(r"checkpoint-(\d+)$", p.name)
        if match:
            candidates.append((int(match.group(1)), p))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return str(candidates[-1][1])


def resolve_resume_checkpoint(cfg: dict[str, Any], output_dir: Path) -> None:
    """Resolve resume checkpoint automatically.

    Priority:
      1. Explicit cfg['resume_from_checkpoint']
      2. Auto-detect largest checkpoint-* under output_dir
      3. None (train from scratch)
    """
    explicit = cfg.get("resume_from_checkpoint")

    if explicit:
        cfg["resume_from_checkpoint_source"] = "config"
        return

    if bool(cfg.get("auto_resume", True)):
        latest = find_latest_checkpoint(str(output_dir))
        if latest is not None:
            cfg["resume_from_checkpoint"] = latest
            cfg["resume_from_checkpoint_source"] = "auto_detect"
            return

    cfg["resume_from_checkpoint"] = None
    cfg["resume_from_checkpoint_source"] = "none"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OPD / ESR / adaptive-KL experiments")
    parser.add_argument("--config", required=True, help="YAML configuration file")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override YAML value. Can be repeated.",
    )
    return parser.parse_args()


def main_process_print(enabled: bool, title: str, payload: dict[str, Any] | None = None) -> None:
    if not enabled:
        return

    print("")
    print("=" * 100)
    print(title)
    print("=" * 100)
    if payload is not None:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    print("=" * 100)
    print("", flush=True)


def infer_effective_lengths(cfg: dict[str, Any]) -> None:
    """Infer actual generation length and actual max sequence length.

    Important:
    - full_max_new_tokens controls full OPD rollout.
    - lite_max_new_tokens controls short rollout for lite_prune / ESR-style experiments.
    - effective_max_new_tokens can hard override both.
    - auto_shrink_max_length can reduce max_length to max_prompt_length + effective_max_new_tokens.

    This function writes:
      cfg["effective_max_new_tokens"]
      cfg["effective_max_length"]
    """
    strategy = str(cfg.get("strategy", "full"))

    short_rollout_strategies = {
        "esr",
        "lite_prune",
        "prune",
        "adaptive_prune",
        "prefix",
        "prefix_opd",
        "early_cut",
        "short_rollout",
    }

    if cfg.get("effective_max_new_tokens") is not None:
        effective_max_new_tokens = int(cfg["effective_max_new_tokens"])
        effective_source = "effective_max_new_tokens"
    elif strategy in short_rollout_strategies:
        effective_max_new_tokens = int(
            cfg.get(
                "lite_max_new_tokens",
                cfg.get("esr_cut_length", cfg["full_max_new_tokens"]),
            )
        )
        if cfg.get("lite_max_new_tokens") is not None:
            effective_source = "lite_max_new_tokens"
        elif cfg.get("esr_cut_length") is not None:
            effective_source = "esr_cut_length"
        else:
            effective_source = "full_max_new_tokens_fallback"
    else:
        effective_max_new_tokens = int(cfg["full_max_new_tokens"])
        effective_source = "full_max_new_tokens"

    effective_max_length = int(cfg["max_length"])
    length_source = "max_length"

    if bool(cfg.get("auto_shrink_max_length", False)):
        effective_max_length = min(
            effective_max_length,
            int(cfg["max_prompt_length"]) + effective_max_new_tokens,
        )
        length_source = "min(max_length, max_prompt_length + effective_max_new_tokens)"

    cfg["effective_max_new_tokens"] = int(effective_max_new_tokens)
    cfg["effective_max_new_tokens_source"] = effective_source
    cfg["effective_max_length"] = int(effective_max_length)
    cfg["effective_max_length_source"] = length_source


def main() -> None:
    start_time = time.time()
    cli = parse_args()
    cfg = load_config_with_base(cli.config, cli.set)

    # -------------------------------------------------------------------------
    # Resolve effective training lengths before creating collator / GKDConfig.
    # -------------------------------------------------------------------------
    infer_effective_lengths(cfg)

    set_seed(int(cfg["seed"]))

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Automatically resume from the latest checkpoint-* if available.
    resolve_resume_checkpoint(cfg, output_dir)

    # Save resolved config after CLI overrides and effective length inference.
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False, default=str)

    # At this point we do not yet have trainer.accelerator, so just print from
    # process rank 0-ish environment. In single-GPU jobs this is always fine.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_print_process = local_rank in {-1, 0}

    main_process_print(
        is_print_process,
        "Resolved OPD training config",
        {
            "loaded_config": cfg.get("_loaded_config"),
            "loaded_base_config": cfg.get("_loaded_base_config"),
            "cli_overrides": cfg.get("_cli_overrides"),
            "output_dir": str(output_dir),
            "latest_checkpoint_detected": cfg.get("resume_from_checkpoint"),
            "strategy": cfg.get("strategy"),
            "opd_loss_mode": cfg.get("opd_loss_mode", "original"),
            "seed": cfg.get("seed"),
            "resume_from_checkpoint": cfg.get("resume_from_checkpoint"),
            "resume_from_checkpoint_source": cfg.get("resume_from_checkpoint_source"),
            "auto_resume": cfg.get("auto_resume", True),
            "debug_train_log": cfg.get("debug_train_log"),
            "debug_train_log_steps": cfg.get("debug_train_log_steps"),
        },
    )

    main_process_print(
        is_print_process,
        "Length configuration",
        {
            "max_length": cfg.get("max_length"),
            "max_prompt_length": cfg.get("max_prompt_length"),
            "full_max_new_tokens": cfg.get("full_max_new_tokens"),
            "lite_max_new_tokens": cfg.get("lite_max_new_tokens"),
            "esr_cut_length": cfg.get("esr_cut_length"),
            "effective_max_new_tokens": cfg.get("effective_max_new_tokens"),
            "effective_max_new_tokens_source": cfg.get("effective_max_new_tokens_source"),
            "effective_max_length": cfg.get("effective_max_length"),
            "effective_max_length_source": cfg.get("effective_max_length_source"),
            "auto_shrink_max_length": cfg.get("auto_shrink_max_length", False),
        },
    )

    # -------------------------------------------------------------------------
    # Tokenizers
    # -------------------------------------------------------------------------
    tokenizer_t0 = time.time()

    student_tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name_or_path"],
        trust_remote_code=bool(cfg["trust_remote_code"]),
        padding_side="left",
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        cfg["teacher_model_name_or_path"],
        trust_remote_code=bool(cfg["trust_remote_code"]),
        padding_side="left",
    )

    if student_tokenizer.pad_token_id is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token
    if teacher_tokenizer.pad_token_id is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    tokenizer_elapsed = time.time() - tokenizer_t0

    main_process_print(
        is_print_process,
        "Tokenizer summary",
        {
            "student_model": cfg["model_name_or_path"],
            "teacher_model": cfg["teacher_model_name_or_path"],
            "student_vocab_size": len(student_tokenizer),
            "teacher_vocab_size": len(teacher_tokenizer),
            "student_pad_token": student_tokenizer.pad_token,
            "student_pad_token_id": student_tokenizer.pad_token_id,
            "student_eos_token": student_tokenizer.eos_token,
            "student_eos_token_id": student_tokenizer.eos_token_id,
            "teacher_pad_token": teacher_tokenizer.pad_token,
            "teacher_pad_token_id": teacher_tokenizer.pad_token_id,
            "teacher_eos_token": teacher_tokenizer.eos_token,
            "teacher_eos_token_id": teacher_tokenizer.eos_token_id,
            "same_vocab_size": len(student_tokenizer) == len(teacher_tokenizer),
            "tokenizer_load_sec": round(tokenizer_elapsed, 3),
        },
    )

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    data_t0 = time.time()
    train_dataset = load_training_dataset(cfg)
    data_elapsed = time.time() - data_t0

    try:
        dataset_len = len(train_dataset)
    except Exception:
        dataset_len = None

    main_process_print(
        is_print_process,
        "Dataset summary",
        {
            "dataset_name": cfg.get("dataset_name"),
            "dataset_split": cfg.get("dataset_split"),
            "dataset_adapter": cfg.get("dataset_adapter"),
            "max_train_examples": cfg.get("max_train_examples"),
            "shuffle_dataset": cfg.get("shuffle_dataset"),
            "dataset_len": dataset_len,
            "dataset_load_sec": round(data_elapsed, 3),
        },
    )

    # -------------------------------------------------------------------------
    # Data collator
    # -------------------------------------------------------------------------
    collator = OPDDataCollator(
        tokenizer=student_tokenizer,
        max_length=int(cfg["effective_max_length"]),
        max_prompt_length=int(cfg["max_prompt_length"]),
        use_chat_template=bool(cfg.get("student_use_chat_template", True)),
        enable_thinking=bool(cfg.get("student_enable_thinking", False)),
    )

    main_process_print(
        is_print_process,
        "Collator summary",
        {
            "collator": "OPDDataCollator",
            "collator_max_length": int(cfg["effective_max_length"]),
            "collator_max_prompt_length": int(cfg["max_prompt_length"]),
            "student_use_chat_template": bool(cfg.get("student_use_chat_template", True)),
            "student_enable_thinking": bool(cfg.get("student_enable_thinking", False)),
        },
    )

    # -------------------------------------------------------------------------
    # Model init kwargs
    # -------------------------------------------------------------------------
    attn_impl = cfg.get("attn_implementation")

    model_init_kwargs = {
        "trust_remote_code": bool(cfg["trust_remote_code"]),
        "dtype": cfg["dtype"],
        "use_cache": not bool(cfg["gradient_checkpointing"]),
        "low_cpu_mem_usage": True,
    }

    teacher_init_kwargs = {
        "trust_remote_code": bool(cfg["trust_remote_code"]),
        "dtype": cfg["dtype"],
        "use_cache": True,
        "low_cpu_mem_usage": True,
    }

    if attn_impl:
        model_init_kwargs["attn_implementation"] = attn_impl
        teacher_init_kwargs["attn_implementation"] = attn_impl

    main_process_print(
        is_print_process,
        "Model init kwargs",
        {
            "student_model_init_kwargs": model_init_kwargs,
            "teacher_model_init_kwargs": teacher_init_kwargs,
            "gradient_checkpointing": cfg.get("gradient_checkpointing"),
            "attn_implementation": attn_impl,
        },
    )

    # -------------------------------------------------------------------------
    # GKD / Training args
    # -------------------------------------------------------------------------
    gkd_kwargs = dict(
        output_dir=str(output_dir),
        max_steps=int(cfg["max_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        save_steps=int(cfg["save_steps"]),
        logging_steps=int(cfg["logging_steps"]),
        weight_decay=float(cfg["weight_decay"]),
        lr_scheduler_type=cfg["lr_scheduler_type"],
        seed=int(cfg["seed"]),
        data_seed=int(cfg["seed"]),
        bf16=cfg["dtype"] == "bfloat16",
        fp16=cfg["dtype"] == "float16",
        gradient_checkpointing=bool(cfg["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=int(cfg["effective_max_length"]),
        max_new_tokens=int(cfg["effective_max_new_tokens"]),
        temperature=float(cfg["temperature"]),
        lmbda=float(cfg["lmbda"]),
        beta=float(cfg["beta"]),
        seq_kd=bool(cfg["seq_kd"]),
        teacher_model_name_or_path=cfg["teacher_model_name_or_path"],
        model_init_kwargs=model_init_kwargs,
        teacher_model_init_kwargs=teacher_init_kwargs,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to=[] if cfg["report_to"] == "none" else [cfg["report_to"]],
        save_only_model=bool(cfg["save_only_model"]),
        dataloader_num_workers=int(cfg["num_workers"]),
        optim=cfg.get("optim", "adamw_torch"),
        eval_strategy="no",
        do_eval=False,
    )

    if cfg.get("warmup_steps") is not None:
        gkd_kwargs["warmup_steps"] = int(cfg["warmup_steps"])
    else:
        gkd_kwargs["warmup_ratio"] = float(cfg["warmup_ratio"])

    training_args = GKDConfig(**gkd_kwargs)

    global_batch_size = (
        int(cfg["per_device_train_batch_size"])
        * int(cfg["gradient_accumulation_steps"])
        * int(os.environ.get("WORLD_SIZE", "1"))
    )

    main_process_print(
        is_print_process,
        "Training args summary",
        {
            "max_steps": cfg.get("max_steps"),
            "learning_rate": cfg.get("learning_rate"),
            "per_device_train_batch_size": cfg.get("per_device_train_batch_size"),
            "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps"),
            "world_size_env": os.environ.get("WORLD_SIZE", "1"),
            "global_batch_size_estimate": global_batch_size,
            "save_steps": cfg.get("save_steps"),
            "logging_steps": cfg.get("logging_steps"),
            "warmup_steps": cfg.get("warmup_steps"),
            "warmup_ratio": cfg.get("warmup_ratio"),
            "weight_decay": cfg.get("weight_decay"),
            "lr_scheduler_type": cfg.get("lr_scheduler_type"),
            "dtype": cfg.get("dtype"),
            "bf16": gkd_kwargs["bf16"],
            "fp16": gkd_kwargs["fp16"],
            "gkd_max_length": gkd_kwargs["max_length"],
            "gkd_max_new_tokens": gkd_kwargs["max_new_tokens"],
            "temperature": cfg.get("temperature"),
            "lmbda": cfg.get("lmbda"),
            "beta": cfg.get("beta"),
            "seq_kd": cfg.get("seq_kd"),
            "optim": gkd_kwargs["optim"],
            "report_to": gkd_kwargs["report_to"],
            "save_only_model": cfg.get("save_only_model"),
            "num_workers": cfg.get("num_workers"),
        },
    )

    # -------------------------------------------------------------------------
    # LoRA config
    # -------------------------------------------------------------------------
    peft_config = None
    if bool(cfg["use_lora"]):
        peft_config = LoraConfig(
            r=int(cfg["lora_r"]),
            lora_alpha=int(cfg["lora_alpha"]),
            lora_dropout=float(cfg["lora_dropout"]),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=cfg["lora_target_modules"],
        )

    main_process_print(
        is_print_process,
        "PEFT / LoRA summary",
        {
            "use_lora": bool(cfg["use_lora"]),
            "lora_r": cfg.get("lora_r") if bool(cfg["use_lora"]) else None,
            "lora_alpha": cfg.get("lora_alpha") if bool(cfg["use_lora"]) else None,
            "lora_dropout": cfg.get("lora_dropout") if bool(cfg["use_lora"]) else None,
            "lora_target_modules": cfg.get("lora_target_modules") if bool(cfg["use_lora"]) else None,
        },
    )

    # -------------------------------------------------------------------------
    # Trainer
    # -------------------------------------------------------------------------
    trainer_t0 = time.time()

    trainer = AdaptiveKLTrainer(
        model=cfg["model_name_or_path"],
        teacher_model=cfg["teacher_model_name_or_path"],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=student_tokenizer,
        data_collator=collator,
        peft_config=peft_config,
        teacher_tokenizer=teacher_tokenizer,
        experiment_config=cfg,
    )

    trainer_elapsed = time.time() - trainer_t0

    if trainer.accelerator.is_main_process:
        print("")
        print("=" * 100)
        print("Trainer initialized")
        print("=" * 100)
        print(json.dumps({
            "strategy": cfg["strategy"],
            "opd_loss_mode": cfg.get("opd_loss_mode", "original"),
            "student": cfg["model_name_or_path"],
            "teacher": cfg["teacher_model_name_or_path"],
            "dataset": cfg["dataset_name"],
            "student_use_chat_template": cfg.get("student_use_chat_template"),
            "teacher_use_chat_template": cfg.get("teacher_use_chat_template"),
            "loss_backend": trainer.loss_backend,
            "same_tokenizer": trainer.same_tokenizer,
            "accelerator_num_processes": trainer.accelerator.num_processes,
            "accelerator_process_index": trainer.accelerator.process_index,
            "global_batch_size": int(cfg["per_device_train_batch_size"])
            * int(cfg["gradient_accumulation_steps"])
            * trainer.accelerator.num_processes,
            "resume_from_checkpoint": cfg.get("resume_from_checkpoint"),
            "effective_max_length": cfg.get("effective_max_length"),
            "effective_max_new_tokens": cfg.get("effective_max_new_tokens"),
            "trainer_init_sec": round(trainer_elapsed, 3),
        }, indent=2, ensure_ascii=False, default=str))
        print("=" * 100)
        print("", flush=True)

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    if trainer.accelerator.is_main_process:
        print("")
        print("=" * 100)
        print("Starting training")
        print("=" * 100)
        print(json.dumps({
            "resume_from_checkpoint": cfg.get("resume_from_checkpoint"),
            "output_dir": str(output_dir),
            "max_steps": cfg.get("max_steps"),
            "effective_max_length": cfg.get("effective_max_length"),
            "effective_max_new_tokens": cfg.get("effective_max_new_tokens"),
        }, indent=2, ensure_ascii=False, default=str))
        print("=" * 100)
        print("", flush=True)

    train_t0 = time.time()
    train_result = trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint"))
    train_elapsed = time.time() - train_t0

    if trainer.accelerator.is_main_process:
        print("")
        print("=" * 100)
        print("Training finished")
        print("=" * 100)
        print(json.dumps({
            "train_elapsed_sec": round(train_elapsed, 3),
            "train_elapsed_min": round(train_elapsed / 60.0, 3),
            "train_result": str(train_result),
        }, indent=2, ensure_ascii=False, default=str))
        print("=" * 100)
        print("", flush=True)

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    save_t0 = time.time()

    trainer.save_model(str(output_dir))
    student_tokenizer.save_pretrained(str(output_dir))

    save_elapsed = time.time() - save_t0
    total_elapsed = time.time() - start_time

    if trainer.accelerator.is_main_process:
        print("")
        print("=" * 100)
        print("Model saved")
        print("=" * 100)
        print(json.dumps({
            "output_dir": str(output_dir),
            "save_elapsed_sec": round(save_elapsed, 3),
            "total_elapsed_sec": round(total_elapsed, 3),
            "total_elapsed_min": round(total_elapsed / 60.0, 3),
            "resolved_config_path": str(output_dir / "resolved_config.json"),
            "resume_from_checkpoint": cfg.get("resume_from_checkpoint"),
            "resume_from_checkpoint_source": cfg.get("resume_from_checkpoint_source"),
        }, indent=2, ensure_ascii=False, default=str))
        print("=" * 100)
        print("", flush=True)


if __name__ == "__main__":
    main()