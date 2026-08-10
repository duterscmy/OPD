#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from peft import LoraConfig
from transformers import AutoTokenizer, set_seed
from trl.experimental.gkd import GKDConfig

from opd.adaptive_trainer import AdaptiveKLTrainer
from opd.collator import OPDDataCollator
from opd.config import (
    find_latest_checkpoint,
    infer_effective_lengths,
    load_config_with_base,
    validate_experiment_config,
)
from opd.data import load_training_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Top-K OPD / ESR experiments")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def rank_zero_print(title: str, payload: dict[str, Any]) -> None:
    if int(os.environ.get("LOCAL_RANK", "0")) not in {-1, 0}:
        return
    print("\n" + "=" * 100)
    print(title)
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    print("=" * 100 + "\n", flush=True)


def main() -> None:
    started = time.time()
    args = parse_args()
    cfg = load_config_with_base(args.config, args.set)
    validate_experiment_config(cfg)
    infer_effective_lengths(cfg)
    set_seed(int(cfg["seed"]))

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.get("resume_from_checkpoint") is None and bool(cfg.get("auto_resume", True)):
        cfg["resume_from_checkpoint"] = find_latest_checkpoint(output_dir)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    rank_zero_print(
        "Resolved Top-K OPD experiment",
        {
            "config": cfg.get("_loaded_config"),
            "backend": cfg["loss_backend"],
            "loss_mode": cfg["opd_loss_mode"],
            "strategy": cfg["strategy"],
            "rollout_horizon": cfg["effective_max_new_tokens"],
            "rollout_include_teacher_eos": cfg.get("rollout_include_teacher_eos", True),
            "rollout_extra_eos_tokens": cfg.get("rollout_extra_eos_tokens", []),
            "rollout_truncate_after_boxed_answer": cfg.get(
                "rollout_truncate_after_boxed_answer", False
            ),
            "rollout_append_eos_after_boxed_answer": cfg.get(
                "rollout_append_eos_after_boxed_answer", False
            ),
            "truncated_rollout_weight": cfg.get("truncated_rollout_weight", 0.0),
            "loss_normalization": cfg.get("loss_normalization", "per_sequence"),
            "rollout_repetition_ngram_size": cfg.get("rollout_repetition_ngram_size", 4),
            "reverse_top_k": cfg.get("reverse_top_k"),
            "forward_top_k": cfg.get("forward_top_k"),
            "overlap_top_k": cfg.get("overlap_top_k"),
            "overlap_threshold": cfg.get("adaptive_overlap_threshold"),
            "prune_threshold": cfg.get("prune_overlap_threshold"),
            "output_dir": str(output_dir),
            "resume_from_checkpoint": cfg.get("resume_from_checkpoint"),
            "debug_jsonl_enabled": cfg.get("debug_jsonl_enabled"),
            "debug_every_n_loss_calls": cfg.get("debug_every_n_loss_calls"),
            "behavior_monitor_enabled": cfg.get("behavior_monitor_enabled", False),
            "behavior_monitor_every_n_loss_calls": cfg.get(
                "behavior_monitor_every_n_loss_calls", 1
            ),
            "behavior_probe_enabled": cfg.get("behavior_probe_enabled", False),
            "behavior_probe_every_n_steps": cfg.get(
                "behavior_probe_every_n_steps", 25
            ),
        },
    )

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
        max_length=int(cfg["effective_max_length"]),
        max_prompt_length=int(cfg["max_prompt_length"]),
        use_chat_template=bool(cfg.get("student_use_chat_template", False)),
        enable_thinking=bool(cfg.get("student_enable_thinking", False)),
    )

    model_init_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(cfg["trust_remote_code"]),
        "dtype": cfg["dtype"],
        "use_cache": not bool(cfg["gradient_checkpointing"]),
        "low_cpu_mem_usage": True,
    }
    teacher_init_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(cfg["trust_remote_code"]),
        "dtype": cfg["dtype"],
        "use_cache": True,
        "low_cpu_mem_usage": True,
    }
    if cfg.get("attn_implementation"):
        model_init_kwargs["attn_implementation"] = cfg["attn_implementation"]
        teacher_init_kwargs["attn_implementation"] = cfg["attn_implementation"]

    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "max_steps": int(cfg["max_steps"]),
        "learning_rate": float(cfg["learning_rate"]),
        "per_device_train_batch_size": int(cfg["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(cfg["gradient_accumulation_steps"]),
        "save_steps": int(cfg["save_steps"]),
        "logging_steps": int(cfg["logging_steps"]),
        "weight_decay": float(cfg["weight_decay"]),
        "lr_scheduler_type": cfg["lr_scheduler_type"],
        "seed": int(cfg["seed"]),
        "data_seed": int(cfg["seed"]),
        "bf16": cfg["dtype"] == "bfloat16",
        "fp16": cfg["dtype"] == "float16",
        "gradient_checkpointing": bool(cfg["gradient_checkpointing"]),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "max_length": int(cfg["effective_max_length"]),
        "max_new_tokens": int(cfg["effective_max_new_tokens"]),
        "temperature": float(cfg["temperature"]),
        "teacher_model_name_or_path": cfg["teacher_model_name_or_path"],
        "model_init_kwargs": model_init_kwargs,
        "teacher_model_init_kwargs": teacher_init_kwargs,
        "remove_unused_columns": False,
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "report_to": [] if cfg["report_to"] == "none" else [cfg["report_to"]],
        "save_only_model": bool(cfg["save_only_model"]),
        "dataloader_num_workers": int(cfg["num_workers"]),
        "optim": cfg.get("optim", "adamw_torch"),
        "eval_strategy": "no",
        "do_eval": False,
    }
    if cfg.get("warmup_steps") is None:
        training_kwargs["warmup_ratio"] = float(cfg["warmup_ratio"])
    else:
        training_kwargs["warmup_steps"] = int(cfg["warmup_steps"])
    training_args = GKDConfig(**training_kwargs)

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
    trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint"))
    trainer.save_model(str(output_dir))
    student_tokenizer.save_pretrained(str(output_dir))

    if trainer.accelerator.is_main_process:
        print(
            json.dumps(
                {
                    "status": "finished",
                    "output_dir": str(output_dir),
                    "elapsed_minutes": round((time.time() - started) / 60.0, 3),
                    "debug_jsonl": str(trainer.debug_jsonl_path),
                    "behavior_manifest": str(trainer.behavior_manifest_path),
                    "behavior_probe_set": str(trainer.behavior_probe_set_path),
                    "behavior_probe_jsonl": str(trainer.behavior_probe_jsonl_path),
                },
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
