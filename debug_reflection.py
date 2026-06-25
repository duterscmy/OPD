#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from opd.config import load_config
from opd.data import load_training_dataset
from opd.collator import OPDDataCollator
from opd.reflection import (
    build_reflection_prompt,
    split_token_chunks,
    extract_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug student rollout + teacher reflection without training."
    )
    parser.add_argument("--config", required=True, help="YAML config file.")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--student-max-new-tokens", type=int, default=None)
    parser.add_argument("--teacher-max-new-tokens", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--output-jsonl", type=str, default=None)
    parser.add_argument("--print-max-chars", type=int, default=6000)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def strip_thinking(text: str) -> str:
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if text.startswith("<think>"):
        return ""
    return text


def safe_decode(tokenizer, ids: list[int] | torch.Tensor, max_chars: int | None = None) -> str:
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()

    text = tokenizer.decode(
        [int(x) for x in ids],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    if max_chars is not None:
        return text[:max_chars]
    return text


def apply_chat_template_to_text(
    tokenizer,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
    enable_thinking: bool = False,
) -> str:
    """
    Robust chat-template helper.

    Important:
    We intentionally use tokenize=False and then encode manually.
    This avoids the bug where apply_chat_template(tokenize=True) returns:
        {"input_ids": [...], "attention_mask": [...]}
    and the dict is accidentally converted into a text prompt.
    """
    if getattr(tokenizer, "chat_template", None):
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

    lines = []
    for msg in messages:
        role = str(msg.get("role", "user")).capitalize()
        content = str(msg.get("content", ""))
        lines.append(f"{role}: {content}")

    if add_generation_prompt:
        lines.append("Assistant:")

    return "\n".join(lines)


def encode_chat_messages(
    tokenizer,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
    enable_thinking: bool = False,
) -> tuple[list[int], str]:
    text = apply_chat_template_to_text(
        tokenizer=tokenizer,
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids = [int(x) for x in ids]
    return ids, text


def build_teacher_encoded_rows(
    teacher_tokenizer,
    reflection_prompts: list[str],
) -> tuple[list[list[int]], list[str]]:
    """Build teacher judge inputs with enable_thinking=False.

    Returns:
        encoded_rows: token ids
        teacher_prompt_texts: actual teacher prompt texts for debugging
    """
    encoded_rows: list[list[int]] = []
    teacher_prompt_texts: list[str] = []

    for prompt in reflection_prompts:
        messages = [
            {
                "role": "system",
                "content": (
                    "Return only valid JSON. "
                    "Do not reveal chain-of-thought. "
                    "Do not output <think>. "
                    "Do not use markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        ids, prompt_text = encode_chat_messages(
            tokenizer=teacher_tokenizer,
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        encoded_rows.append(ids)
        teacher_prompt_texts.append(prompt_text)

    return encoded_rows, teacher_prompt_texts


@torch.no_grad()
def generate_student_rollouts(
    student_model,
    student_tokenizer,
    prompts: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
) -> tuple[list[list[int]], list[bool], torch.Tensor]:
    student_model.eval()

    outputs = student_model.generate(
        input_ids=prompts,
        attention_mask=prompt_attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=0,
        pad_token_id=student_tokenizer.pad_token_id,
        eos_token_id=student_tokenizer.eos_token_id,
        use_cache=True,
        return_dict_in_generate=True,
    )

    prompt_width = prompts.shape[1]
    completion_rows = outputs.sequences[:, prompt_width:]

    completions: list[list[int]] = []
    eos_flags: list[bool] = []

    eos = student_tokenizer.eos_token_id
    pad = student_tokenizer.pad_token_id

    for row in completion_rows:
        ids: list[int] = []
        saw_eos = False

        for tok in row.tolist():
            tok = int(tok)

            if pad is not None and tok == pad:
                break

            ids.append(tok)

            if eos is not None and tok == eos:
                saw_eos = True
                break

        completions.append(ids)
        eos_flags.append(saw_eos)

    return completions, eos_flags, outputs.sequences


@torch.no_grad()
def generate_teacher_reflections(
    teacher_model,
    teacher_tokenizer,
    reflection_prompts: list[str],
    max_new_tokens: int,
) -> tuple[list[str], list[str], list[dict[str, Any] | None]]:
    teacher_model.eval()

    encoded_rows, teacher_prompt_texts = build_teacher_encoded_rows(
        teacher_tokenizer=teacher_tokenizer,
        reflection_prompts=reflection_prompts,
    )

    if teacher_tokenizer.pad_token_id is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    pad_id = int(teacher_tokenizer.pad_token_id)
    width = max(len(row) for row in encoded_rows)

    # Left padding, matching your training code.
    padded = [[pad_id] * (width - len(row)) + row for row in encoded_rows]
    masks = [[0] * (width - len(row)) + [1] * len(row) for row in encoded_rows]

    device = next(teacher_model.parameters()).device
    input_ids = torch.tensor(padded, dtype=torch.long, device=device)
    attention_mask = torch.tensor(masks, dtype=torch.long, device=device)

    outputs = teacher_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_id,
        eos_token_id=teacher_tokenizer.eos_token_id,
        use_cache=True,
    )

    # IMPORTANT:
    # Because input_ids are left-padded to uniform width,
    # generated tokens start at input_ids.shape[1],
    # not at attention_mask.sum().
    prompt_width = input_ids.shape[1]

    raw_outputs: list[str] = []
    parsed_outputs: list[dict[str, Any] | None] = []

    for i in range(outputs.shape[0]):
        gen_ids = outputs[i, prompt_width:]
        raw = teacher_tokenizer.decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        raw = strip_thinking(raw)
        parsed = extract_json(raw)

        raw_outputs.append(raw)
        parsed_outputs.append(parsed)

    return teacher_prompt_texts, raw_outputs, parsed_outputs


def compute_cut_length(
    parsed: dict[str, Any] | None,
    chunks: list[dict[str, Any]],
    completion_len: int,
    fallback: str,
    fallback_length: int,
) -> int:
    if parsed is None:
        if fallback == "full":
            return completion_len
        if fallback == "skip":
            return 0
        return min(fallback_length, completion_len)

    has_error = parsed.get("has_error")

    if has_error is False:
        return completion_len

    if has_error is True:
        cid = parsed.get("earliest_error_chunk_id")
        try:
            cid = int(cid) if cid is not None else None
        except Exception:
            cid = None

        if cid is not None and 0 <= cid < len(chunks):
            return int(chunks[cid]["start_token"])

    if fallback == "full":
        return completion_len
    if fallback == "skip":
        return 0
    return min(fallback_length, completion_len)


def truncate_text(text: str, max_chars: int) -> str:
    if text is None:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[truncated {len(text) - max_chars} chars]"


def print_section(title: str, content: str | None = None, max_chars: int = 6000) -> None:
    print(f"\n【{title}】\n")
    if content is not None:
        print(truncate_text(content, max_chars))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(int(cfg["seed"]))

    student_name = cfg["model_name_or_path"]
    teacher_name = cfg["teacher_model_name_or_path"]

    student_max_new_tokens = (
        args.student_max_new_tokens
        if args.student_max_new_tokens is not None
        else int(cfg.get("reflection_rollout_max_tokens", cfg.get("full_max_new_tokens", 1024)))
    )
    teacher_max_new_tokens = (
        args.teacher_max_new_tokens
        if args.teacher_max_new_tokens is not None
        else int(cfg.get("reflection_max_new_tokens", 512))
    )
    chunk_size = (
        args.chunk_size
        if args.chunk_size is not None
        else int(cfg.get("reflection_chunk_size", 16))
    )

    print("=" * 100)
    print("【DEBUG REFLECTION CONFIG】")
    print("student:", student_name)
    print("teacher:", teacher_name)
    print("dataset:", cfg["dataset_name"])
    print("student_max_new_tokens:", student_max_new_tokens)
    print("teacher_max_new_tokens:", teacher_max_new_tokens)
    print("chunk_size:", chunk_size)
    print("reflection_use_reference:", cfg.get("reflection_use_reference"))
    print("reflection_parse_failure:", cfg.get("reflection_parse_failure"))
    print("reflection_fallback_length:", cfg.get("reflection_fallback_length"))
    print("=" * 100)

    student_tokenizer = AutoTokenizer.from_pretrained(
        student_name,
        trust_remote_code=bool(cfg["trust_remote_code"]),
        padding_side="left",
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        teacher_name,
        trust_remote_code=bool(cfg["trust_remote_code"]),
        padding_side="left",
    )

    if student_tokenizer.pad_token_id is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token
    if teacher_tokenizer.pad_token_id is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    dtype = torch.bfloat16 if str(cfg["dtype"]) == "bfloat16" else torch.float16
    if str(cfg["dtype"]) == "float32":
        dtype = torch.float32

    student_model = AutoModelForCausalLM.from_pretrained(
        student_name,
        trust_remote_code=bool(cfg["trust_remote_code"]),
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation=cfg.get("attn_implementation", None),
        low_cpu_mem_usage=True,
    )

    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_name,
        trust_remote_code=bool(cfg["trust_remote_code"]),
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation=cfg.get("attn_implementation", None),
        low_cpu_mem_usage=True,
    )

    dataset = load_training_dataset(cfg)
    end_index = min(args.start_index + args.num_samples, len(dataset))
    subset = dataset.select(range(args.start_index, end_index))

    collator = OPDDataCollator(
        tokenizer=student_tokenizer,
        max_length=int(cfg["max_length"]),
        max_prompt_length=int(cfg["max_prompt_length"]),
    )

    features = [subset[i] for i in range(len(subset))]
    batch = collator(features)

    device = next(student_model.parameters()).device
    prompts = batch["prompts"].to(device)
    prompt_attention_mask = batch["prompt_attention_mask"].to(device)

    completions, eos_flags, _ = generate_student_rollouts(
        student_model=student_model,
        student_tokenizer=student_tokenizer,
        prompts=prompts,
        prompt_attention_mask=prompt_attention_mask,
        max_new_tokens=student_max_new_tokens,
        temperature=float(cfg["temperature"]),
    )

    reflection_prompts: list[str] = []
    all_chunks: list[list[dict[str, Any]]] = []

    for ids, problem, reference in zip(
        completions,
        batch["problem"],
        batch["reference_solution"],
        strict=True,
    ):
        chunks = split_token_chunks(student_tokenizer, ids, chunk_size)
        all_chunks.append(chunks)
        reflection_prompts.append(
            build_reflection_prompt(
                problem=problem,
                chunks=chunks,
                reference_solution=reference,
                use_reference=bool(cfg["reflection_use_reference"]),
            )
        )

    teacher_prompt_texts, teacher_raw_outputs, teacher_parsed = generate_teacher_reflections(
        teacher_model=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        reflection_prompts=reflection_prompts,
        max_new_tokens=teacher_max_new_tokens,
    )

    output_f = None
    if args.output_jsonl:
        out_path = Path(args.output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output_f = out_path.open("w", encoding="utf-8")

    fallback = cfg.get("reflection_parse_failure", "esr")
    fallback_len = int(cfg.get("reflection_fallback_length", 100))

    for i, feature in enumerate(features):
        prompt_ids = prompts[i][prompt_attention_mask[i].bool()].tolist()
        completion_ids = completions[i]
        chunks = all_chunks[i]
        parsed = teacher_parsed[i]

        cut = compute_cut_length(
            parsed=parsed,
            chunks=chunks,
            completion_len=len(completion_ids),
            fallback=fallback,
            fallback_length=fallback_len,
        )

        record = {
            "sample_index": i,
            "dataset_index": args.start_index + i,
            "problem": batch["problem"][i],
            "reference_solution": batch["reference_solution"][i],
            "student_prompt_text": safe_decode(student_tokenizer, prompt_ids),
            "student_completion_text": safe_decode(student_tokenizer, completion_ids),
            "student_completion_tokens": len(completion_ids),
            "student_saw_eos": bool(eos_flags[i]),
            "student_hit_horizon": bool((not eos_flags[i]) and len(completion_ids) >= student_max_new_tokens),
            "chunks": chunks,
            "teacher_reflection_prompt": reflection_prompts[i],
            "teacher_encoded_prompt_text": teacher_prompt_texts[i],
            "teacher_raw_output": teacher_raw_outputs[i],
            "teacher_parsed": parsed,
            "cut_tokens": cut,
            "used_prefix_text": safe_decode(student_tokenizer, completion_ids[:cut]),
        }

        if output_f:
            output_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_f.flush()

        print("\n" + "=" * 100)
        print(f"【SAMPLE {i} | dataset_idx={args.start_index + i}】")
        print("=" * 100)

        print_section("PROBLEM", record["problem"], args.print_max_chars)
        print_section("REFERENCE SOLUTION", record["reference_solution"], args.print_max_chars)
        print_section("STUDENT PROMPT TEXT", record["student_prompt_text"], args.print_max_chars)

        print_section(
            "STUDENT ROLLOUT META",
            (
                f"tokens={record['student_completion_tokens']}\n"
                f"eos={record['student_saw_eos']}\n"
                f"hit_horizon={record['student_hit_horizon']}"
            ),
            args.print_max_chars,
        )
        print_section("STUDENT ROLLOUT", record["student_completion_text"], args.print_max_chars)

        print_section("STUDENT CHUNKS")
        for c in chunks[:20]:
            print(f"【CHUNK {c['id']}】 tokens {c['start_token']}..{c['end_token']}")
            print(c["text"])
        if len(chunks) > 20:
            print(f"... {len(chunks) - 20} more chunks")

        print_section("TEACHER REFLECTION PROMPT", record["teacher_reflection_prompt"], args.print_max_chars)
        print_section("TEACHER ENCODED PROMPT TEXT", record["teacher_encoded_prompt_text"], args.print_max_chars)
        print_section("TEACHER RAW OUTPUT", record["teacher_raw_output"], args.print_max_chars)

        print_section(
            "TEACHER PARSED",
            json.dumps(record["teacher_parsed"], indent=2, ensure_ascii=False),
            args.print_max_chars,
        )

        print_section(
            "CUT RESULT",
            f"cut_tokens={cut} / completion_tokens={len(completion_ids)}",
            args.print_max_chars,
        )

        print_section("USED PREFIX TEXT", record["used_prefix_text"], args.print_max_chars)

        print_section(
            "MANUAL CHECK QUESTIONS",
            (
                "1. Does 【STUDENT PROMPT TEXT】 contain the real math problem, not input_ids/attention_mask?\n"
                "2. Does 【STUDENT ROLLOUT】 answer the math problem rather than explain token IDs?\n"
                "3. Does 【TEACHER ENCODED PROMPT TEXT】 contain the intended math verifier prompt?\n"
                "4. Does 【TEACHER RAW OUTPUT】 contain valid JSON?\n"
                "5. Is earliest_error_chunk_id reasonable compared with 【STUDENT CHUNKS】?\n"
                "6. If parse failed, is it because of prompt, decoding offset, or max_new_tokens?"
            ),
            args.print_max_chars,
        )

    if output_f:
        output_f.close()
        print(f"\n【OUTPUT】Wrote JSONL debug records to: {args.output_jsonl}")


if __name__ == "__main__":
    main()