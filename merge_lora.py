#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    adapter = Path(args.adapter)
    peft_cfg = PeftConfig.from_pretrained(str(adapter))
    dtype = getattr(torch, args.dtype)
    base = AutoModelForCausalLM.from_pretrained(
        peft_cfg.base_model_name_or_path,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter))
    merged = model.merge_and_unload()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output, safe_serialization=True, max_shard_size="8GB")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter), trust_remote_code=True)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
