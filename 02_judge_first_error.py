#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Judge earliest reasoning error position in student CoT on MATH-500.

This version uses chunk-based localization:
  1. Read lm-eval --log_samples outputs.
  2. Extract problem, reference solution, student generation.
  3. Split student generation into token chunks using the student tokenizer.
  4. Ask teacher model to return earliest_error_chunk_id.
  5. Map chunk_id to token position.
  6. Report whether earliest error is within the first N generated tokens.

Recommended models:
  student tokenizer: Qwen/Qwen2.5-Math-1.5B
  teacher judge:     Qwen/Qwen3-4B

Example:
python 02_judge_first_error_chunk.py \
  --samples outputs/math500_student_qwen25_15b \
  --teacher Qwen/Qwen3-4B \
  --student-tokenizer Qwen/Qwen2.5-Math-1.5B \
  --threshold 100 \
  --chunk-size 32 \
  --max-cases 20 \
  --out outputs/first_error_chunk_judged.jsonl \
  --use-reference-cot
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# =========================
# IO helpers
# =========================

def iter_json_records(path: Path) -> Iterable[Dict[str, Any]]:
    """
    Robustly iterate over json/jsonl records from lm-eval output.
    Accepts:
      - a directory containing json/jsonl files
      - a single jsonl file
      - a single json file
    """
    if path.is_dir():
        files = []
        files.extend(path.rglob("*.jsonl"))
        files.extend(path.rglob("*.json"))
    else:
        files = [path]

    # Prefer files likely to contain samples.
    files = sorted(files, key=lambda p: ("samples" not in p.name.lower(), str(p)))

    for fp in files:
        if fp.suffix == ".jsonl":
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        yield obj

        elif fp.suffix == ".json":
            with open(fp, "r", encoding="utf-8") as f:
                try:
                    obj = json.load(f)
                except Exception:
                    continue

            if isinstance(obj, list):
                for x in obj:
                    if isinstance(x, dict):
                        yield x

            elif isinstance(obj, dict):
                # Common lm-eval output structures.
                if "samples" in obj and isinstance(obj["samples"], list):
                    for x in obj["samples"]:
                        if isinstance(x, dict):
                            yield x

                # Some lm-eval versions store samples by task name.
                elif "samples" in obj and isinstance(obj["samples"], dict):
                    for _, sample_list in obj["samples"].items():
                        if isinstance(sample_list, list):
                            for x in sample_list:
                                if isinstance(x, dict):
                                    yield x

                # Nested structure fallback.
                else:
                    yielded = False
                    for v in obj.values():
                        if isinstance(v, list):
                            for x in v:
                                if isinstance(x, dict) and looks_like_sample(x):
                                    yielded = True
                                    yield x
                    if not yielded and looks_like_sample(obj):
                        yield obj


def looks_like_sample(obj: Dict[str, Any]) -> bool:
    keys = set(obj.keys())
    sample_like_keys = {
        "doc", "arguments", "resps", "filtered_resps", "target", "problem",
        "question", "prompt", "exact_match", "acc", "correct"
    }
    return len(keys & sample_like_keys) > 0


def first_existing(sample: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in sample and sample[k] is not None:
            return sample[k]
    return None


# =========================
# lm-eval sample parsing
# =========================

def get_generation_from_sample(sample: Dict[str, Any]) -> str:
    """
    Extract generated text from lm-eval log_samples.

    Common lm-eval formats:
      resps: [["..."]]
      filtered_resps: ["..."] or [["..."]]
      doc: {...}
    """
    candidates = [
        "filtered_resps",
        "resps",
        "response",
        "responses",
        "prediction",
        "pred",
        "generation",
        "model_output",
        "output",
    ]

    val = first_existing(sample, candidates)

    if isinstance(val, str):
        return val

    if isinstance(val, list):
        cur = val
        while isinstance(cur, list) and len(cur) > 0:
            cur = cur[0]
        if isinstance(cur, str):
            return cur
        if isinstance(cur, dict):
            for k in ["text", "output", "generation", "response"]:
                if k in cur and isinstance(cur[k], str):
                    return cur[k]

    args = sample.get("arguments")
    if isinstance(args, dict):
        for k in candidates:
            if k in args and isinstance(args[k], str):
                return args[k]

    return ""


def get_problem_from_sample(sample: Dict[str, Any]) -> str:
    candidates = [
        "problem",
        "question",
        "query",
        "prompt",
        "input",
    ]

    val = first_existing(sample, candidates)
    if isinstance(val, str):
        return val

    if isinstance(val, dict):
        for k in candidates:
            if k in val and isinstance(val[k], str):
                return val[k]

    doc = sample.get("doc")
    if isinstance(doc, dict):
        for k in candidates:
            if k in doc and isinstance(doc[k], str):
                return doc[k]

    args = sample.get("arguments")
    if isinstance(args, dict):
        for k in candidates:
            if k in args and isinstance(args[k], str):
                return args[k]

    # Sometimes the prompt is in arguments as first element.
    if isinstance(args, list):
        for x in args:
            if isinstance(x, str) and len(x) > 20:
                return x
            if isinstance(x, dict):
                for k in candidates:
                    if k in x and isinstance(x[k], str):
                        return x[k]

    return ""


def get_reference_solution_from_sample(sample: Dict[str, Any]) -> str:
    candidates = [
        "solution",
        "answer",
        "target",
        "reference",
        "gold",
        "gold_answer",
    ]

    val = first_existing(sample, candidates)
    if isinstance(val, str):
        return val

    if isinstance(val, dict):
        for k in candidates:
            if k in val and isinstance(val[k], str):
                return val[k]

    doc = sample.get("doc")
    if isinstance(doc, dict):
        for k in candidates:
            if k in doc and isinstance(doc[k], str):
                return doc[k]

    return ""


def get_correctness_from_sample(sample: Dict[str, Any]) -> Optional[bool]:
    """
    Infer lm-eval correctness if available.

    Warning:
      lm-eval may mark a mathematically correct response as wrong due to answer parsing.
      We keep this field only for filtering/diagnosis.
    """
    keys = [
        "exact_match",
        "acc",
        "accuracy",
        "correct",
        "is_correct",
        "score",
    ]

    for k in keys:
        if k in sample:
            v = sample[k]
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v > 0)

    metrics = sample.get("metrics")
    if isinstance(metrics, dict):
        for k in keys:
            if k in metrics:
                v = metrics[k]
                if isinstance(v, bool):
                    return v
                if isinstance(v, (int, float)):
                    return bool(v > 0)

    # Some lm-eval versions store metrics in "filtered_resps" separate from sample score;
    # do not guess if not explicit.
    return None


# =========================
# Chunking
# =========================

def make_token_chunks(text: str, tokenizer, chunk_size: int = 32) -> List[Dict[str, Any]]:
    """
    Split generated student CoT into token chunks using student tokenizer.
    Token positions are 1-indexed.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    for i in range(0, len(token_ids), chunk_size):
        ids = token_ids[i:i + chunk_size]
        chunk_text = tokenizer.decode(ids, skip_special_tokens=True)
        chunks.append({
            "id": len(chunks),
            "text": chunk_text,
            "start_token_pos": i + 1,
            "end_token_pos": i + len(ids),
        })

    return chunks


# =========================
# Prompt and JSON parsing
# =========================

def build_chunk_judge_prompt(
    problem: str,
    chunks: List[Dict[str, Any]],
    reference_solution: str = "",
    use_reference_cot: bool = False,
) -> str:
    ref_block = ""
    if use_reference_cot and reference_solution.strip():
        ref_block = f"""
Reference solution / gold answer:
{reference_solution}
"""

    chunk_text = "\n".join(
        f"[{c['id']}] {c['text']}" for c in chunks
    )

    return f"""You are a strict mathematical reasoning judge.

Given a MATH problem and a student's solution split into numbered chunks, identify the earliest chunk where the student's reasoning first becomes mathematically wrong, invalid, unsupported, inconsistent with the problem, or insufficient in a way that causes the final answer to be wrong.

Rules:
1. Return the earliest chunk id where the first actual reasoning error occurs.
2. If the solution is mathematically correct, set has_error=false and earliest_error_chunk_id=null.
3. Do not mark formatting-only issues as reasoning errors.
4. If the final answer is wrong, find the earliest reasoning step that caused it.
5. If the student solution is incomplete and cannot support the final answer, mark the earliest chunk where it becomes incomplete or jumps unjustifiably.
6. Return ONLY valid JSON. No markdown. No <think>. No extra text.

Problem:
{problem}

{ref_block}

Student solution chunks:
{chunk_text}

If there is an error, return exactly:
{{
  "has_error": true,
  "earliest_error_chunk_id": 0,
  "earliest_error_span": "short quote from that chunk",
  "error_type": "wrong_setup | wrong_formula | algebra_error | arithmetic_error | invalid_inference | contradiction | incomplete | final_answer_only | other",
  "explanation": "brief explanation"
}}

If there is no error, return exactly:
{{
  "has_error": false,
  "earliest_error_chunk_id": null,
  "earliest_error_span": "",
  "error_type": "none",
  "explanation": "The student solution is mathematically correct."
}}
"""


def extract_json(text: str) -> Dict[str, Any]:
    """
    Extract first valid JSON object from model output.
    If parsing fails, return has_error=None, not True.
    """
    original = text
    text = text.strip()

    # Remove thinking if present.
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()

    # Remove markdown fences.
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    # Direct parse.
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try balanced JSON objects.
    starts = [i for i, ch in enumerate(text) if ch == "{"]
    for s in starts:
        depth = 0
        in_str = False
        escape = False
        for e in range(s, len(text)):
            ch = text[e]

            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[s:e + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break

    return {
        "has_error": None,
        "earliest_error_chunk_id": None,
        "earliest_error_span": "",
        "error_type": "parse_failed",
        "explanation": f"Could not parse judge output: {original[:800]}",
    }


# =========================
# Model loading and generation
# =========================

def load_model_and_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


@torch.no_grad()
def generate_judgment(model, tokenizer, prompt: str, max_new_tokens: int = 1024) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict math judge. "
                "Return ONLY valid JSON. "
                "Do not output thoughts, reasoning, markdown, or explanation outside JSON."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    else:
        input_text = (
            "Return ONLY valid JSON. Do not output thoughts.\n\n" + prompt
        )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    gen = out[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True)

    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()

    return text.strip()


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples",
        type=str,
        required=True,
        help="Path to lm-eval output directory, json, or jsonl.",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default="Qwen/Qwen3-4B",
        help="Teacher judge model.",
    )
    parser.add_argument(
        "--student-tokenizer",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
        help="Tokenizer used to count student generated tokens.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=100,
        help="Token threshold, e.g. 100 for ESR analysis.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="Student-token chunk size for localization.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/first_error_chunk_judged.jsonl",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Maximum number of samples to judge after filtering.",
    )
    parser.add_argument(
        "--use-reference-cot",
        action="store_true",
        help="Provide reference solution/gold answer to teacher judge.",
    )
    parser.add_argument(
        "--judge-all",
        action="store_true",
        help=(
            "Judge all samples. By default, if lm-eval correctness is available, "
            "only samples marked incorrect are judged."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Max new tokens for teacher JSON judgment.",
    )

    args = parser.parse_args()

    samples_path = Path(args.samples)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading teacher judge: {args.teacher}")
    teacher_model, teacher_tok = load_model_and_tokenizer(args.teacher)

    print(f"Loading student tokenizer: {args.student_tokenizer}")
    student_tok = AutoTokenizer.from_pretrained(
        args.student_tokenizer,
        trust_remote_code=True,
    )

    records = list(iter_json_records(samples_path))
    print(f"Loaded {len(records)} records from {samples_path}")

    # Counters
    total_seen = 0
    skipped_correct_by_lmeval = 0
    missing_problem_or_generation = 0

    judged = 0
    parse_failed = 0

    teacher_says_no_error = 0
    teacher_says_error = 0
    teacher_error_located = 0
    teacher_error_unlocated = 0

    le_threshold = 0
    gt_threshold = 0

    lm_eval_wrong_or_unknown_judged = 0
    lm_eval_wrong_teacher_no_error = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for idx, sample in enumerate(records):
            total_seen += 1

            if args.max_cases is not None and judged >= args.max_cases:
                break

            correctness = get_correctness_from_sample(sample)

            # By default, only judge samples that lm-eval did not mark correct.
            if not args.judge_all and correctness is True:
                skipped_correct_by_lmeval += 1
                continue

            problem = get_problem_from_sample(sample)
            student_cot = get_generation_from_sample(sample)
            reference_solution = get_reference_solution_from_sample(sample)

            if not problem or not student_cot:
                missing_problem_or_generation += 1
                continue

            chunks = make_token_chunks(
                student_cot,
                student_tok,
                chunk_size=args.chunk_size,
            )

            prompt = build_chunk_judge_prompt(
                problem=problem,
                chunks=chunks,
                reference_solution=reference_solution,
                use_reference_cot=args.use_reference_cot,
            )

            raw_judge = generate_judgment(
                teacher_model,
                teacher_tok,
                prompt,
                max_new_tokens=args.max_new_tokens,
            )
            judge = extract_json(raw_judge)

            has_error = judge.get("has_error", None)
            chunk_id_raw = judge.get("earliest_error_chunk_id", None)

            token_pos = None
            within_threshold = None
            chunk_id = None

            if has_error is None:
                parse_failed += 1

            elif has_error is False:
                teacher_says_no_error += 1
                if correctness is False:
                    lm_eval_wrong_teacher_no_error += 1

            elif has_error is True:
                teacher_says_error += 1

                if chunk_id_raw is not None:
                    try:
                        chunk_id = int(chunk_id_raw)
                        if 0 <= chunk_id < len(chunks):
                            token_pos = chunks[chunk_id]["start_token_pos"]
                            within_threshold = token_pos <= args.threshold
                    except Exception:
                        chunk_id = None

                if token_pos is not None:
                    teacher_error_located += 1
                    if within_threshold:
                        le_threshold += 1
                    else:
                        gt_threshold += 1
                else:
                    teacher_error_unlocated += 1

            if correctness is False or correctness is None:
                lm_eval_wrong_or_unknown_judged += 1

            judged += 1

            out_obj = {
                "idx": idx,
                "correctness_from_lm_eval": correctness,
                "problem": problem,
                "reference_solution": reference_solution,
                "student_cot": student_cot,
                "num_student_tokens": len(student_tok.encode(student_cot, add_special_tokens=False)),
                "chunks": chunks,
                "judge": judge,
                "raw_judge": raw_judge,
                "has_error": has_error,
                "earliest_error_chunk_id": chunk_id,
                "earliest_error_token_pos": token_pos,
                "threshold": args.threshold,
                "within_threshold": within_threshold,
            }

            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            fout.flush()

            print(
                f"[{judged}] idx={idx} "
                f"lm_eval_correct={correctness} "
                f"has_error={has_error} "
                f"chunk={chunk_id} "
                f"pos={token_pos} "
                f"within={within_threshold}"
            )

    summary = {
        "samples_path": str(samples_path),
        "teacher": args.teacher,
        "student_tokenizer": args.student_tokenizer,
        "threshold": args.threshold,
        "chunk_size": args.chunk_size,

        "total_seen": total_seen,
        "judged": judged,
        "skipped_correct_by_lm_eval": skipped_correct_by_lmeval,
        "missing_problem_or_generation": missing_problem_or_generation,

        "parse_failed": parse_failed,

        "lm_eval_wrong_or_unknown_judged": lm_eval_wrong_or_unknown_judged,
        "lm_eval_wrong_teacher_no_error": lm_eval_wrong_teacher_no_error,

        "teacher_says_no_error": teacher_says_no_error,
        "teacher_says_error": teacher_says_error,
        "teacher_error_located": teacher_error_located,
        "teacher_error_unlocated": teacher_error_unlocated,

        "earliest_error_le_threshold": le_threshold,
        "earliest_error_gt_threshold": gt_threshold,

        "ratio_error_le_threshold_among_located_errors": (
            le_threshold / teacher_error_located
            if teacher_error_located > 0
            else None
        ),
        "ratio_teacher_no_error_among_judged": (
            teacher_says_no_error / judged
            if judged > 0
            else None
        ),
        "ratio_parse_failed_among_judged": (
            parse_failed / judged
            if judged > 0
            else None
        ),
    }

    summary_path = out_path.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nSummary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote judgments to: {out_path}")
    print(f"Wrote summary to: {summary_path}")


if __name__ == "__main__":
    main()