#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from peft import LoraConfig
from transformers import AutoTokenizer, set_seed
from trl.experimental.gkd import GKDConfig

from opd.collator import OPDDataCollator
# We still import the original loader for normal configs, but add a small
# base_config merge layer for experiment overlay YAMLs.
from opd.config import load_config
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

    This is intentionally local to this script so it does not require modifying the
    original OPD repo's `opd/config.py`. Relative base paths are resolved from the
    current working directory first, then from the child config's directory.
    CLI --set overrides are delegated to the original `load_config` by writing the
    merged YAML is avoided here; instead we parse simple KEY=VALUE overrides after
    merge to keep the patch self-contained.
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
        base = yaml.safe_load(base_path.read_text()) or {}
        cfg = _deep_update(base, raw)
    else:
        cfg = raw

    # Support the same common --set KEY=VALUE pattern used by the original script.
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
    return cfg

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OPD / ESR / adaptive-KL experiments")
    parser.add_argument("--config", required=True, help="YAML configuration file")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override YAML value. Can be repeated.")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    cfg = load_config_with_base(cli.config, cli.set)
    set_seed(int(cfg["seed"]))

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

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

    train_dataset = load_training_dataset(cfg)
    collator = OPDDataCollator(
        tokenizer=student_tokenizer,
        max_length=int(cfg["max_length"]),
        max_prompt_length=int(cfg["max_prompt_length"]),
        use_chat_template=bool(cfg.get("student_use_chat_template", True)),
        enable_thinking=bool(cfg.get("student_enable_thinking", False)),
    )

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
        max_length=int(cfg["max_length"]),
        max_new_tokens=int(cfg["full_max_new_tokens"]),
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
        optim="adamw_torch",
        eval_strategy="no",
        do_eval=False,
    )
    if cfg.get("warmup_steps") is not None:
        gkd_kwargs["warmup_steps"] = int(cfg["warmup_steps"])
    else:
        gkd_kwargs["warmup_ratio"] = float(cfg["warmup_ratio"])
    training_args = GKDConfig(**gkd_kwargs)

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

    if trainer.accelerator.is_main_process:
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
            "global_batch_size": int(cfg["per_device_train_batch_size"]) * int(cfg["gradient_accumulation_steps"]) * trainer.accelerator.num_processes,
            "resume_from_checkpoint": cfg.get("resume_from_checkpoint"),
        }, indent=2))

    trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint"))
    trainer.save_model(str(output_dir))
    student_tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
