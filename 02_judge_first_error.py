#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Judge earliest error span in student CoT on MATH-500.

Input:
  - lm-eval --log_samples output directory or json/jsonl file
  - teacher model, e.g. Qwen/Qwen3-4B
  - student tokenizer, e.g. Qwen/Qwen2.5-Math-1.5B

Output:
  - jsonl with teacher judgments
  - summary json

Main metric:
  Among wrong student samples where earliest error is found,
  what percentage have earliest error token position <= threshold?
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# -------------------------
# IO helpers
# -------------------------

def iter_json_records(path: Path) -> Iterable[Dict[str, Any]]:
    if path.is_dir():
        files = []
        files.extend(path.rglob("*.jsonl"))
        files.extend(path.rglob("*.json"))
    else:
        files = [path]

    for fp in files:
        if fp.suffix == ".jsonl":
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, dict):
                                yield obj
                        except Exception:
                            continue
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
                # lm-eval sometimes stores samples under nested keys.
                if "samples" in obj and isinstance(obj["samples"], list):
                    for x in obj["samples"]:
                        if isinstance(x, dict):
                            yield x
                else:
                    yield obj


# -------------------------
# Flexible lm-eval sample parser
# -------------------------

def first_existing(sample: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in sample and sample[k] is not None:
            return sample[k]
    return None


def get_generation_from_sample(sample: Dict[str, Any]) -> str:
    """
    Try to robustly extract model generation from lm-eval log_samples.
    Different lm-eval versions use slightly different fields.
    """
    candidates = [
        "resps",
        "filtered_resps",
        "response",
        "responses",
        "prediction",
        "pred",
        "generation",
        "model_output",
        "output",
    ]

    val = first_existing(sample, candidates)

    # Common lm-eval format:
    # "resps": [[ "... generated text ..." ]]
    if isinstance(val, list):
        cur = val
        while isinstance(cur, list) and len(cur) > 0:
            cur = cur[0]
        if isinstance(cur, str):
            return cur

        # sometimes list of dicts
        if isinstance(cur, dict):
            for k in ["text", "output", "generation", "response"]:
                if k in cur and isinstance(cur[k], str):
                    return cur[k]

    if isinstance(val, str):
        return val

    # Some samples store full doc and target only; generation might be in "arguments"
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
        "doc",
        "document",
    ]

    val = first_existing(sample, candidates)

    if isinstance(val, str):
        return val

    if isinstance(val, dict):
        for k in ["problem", "question", "query", "prompt", "input"]:
            if k in val and isinstance(val[k], str):
                return val[k]

    # lm-eval sometimes has "doc"
    doc = sample.get("doc")
    if isinstance(doc, dict):
        for k in ["problem", "question", "query", "prompt", "input"]:
            if k in doc and isinstance(doc[k], str):
                return doc[k]

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
    Try to infer whether lm-eval marked the sample correct.
    If unavailable, return None.
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

    # lm-eval sometimes stores metrics in a dict
    metrics = sample.get("metrics")
    if isinstance(metrics, dict):
        for k in keys:
            if k in metrics:
                v = metrics[k]
                if isinstance(v, bool):
                    return v
                if isinstance(v, (int, float)):
                    return bool(v > 0)

    return None


# -------------------------
# Judgment prompt and parsing
# -------------------------

def build_judge_prompt(
    problem: str,
    student_cot: str,
    reference_solution: str = "",
    use_reference_cot: bool = False,
) -> str:
    ref_block = ""
    if use_reference_cot and reference_solution.strip():
        ref_block = f"""
Reference solution / gold answer:
{reference_solution}
"""

    return f"""You are a strict mathematical reasoning judge.

Your task:
Given a MATH problem and a student's chain-of-thought solution, identify the earliest span in the student's solution where the reasoning first becomes mathematically wrong, invalid, unsupported, or inconsistent with the problem.

Important rules:
1. Find the earliest actual reasoning error, not just a stylistic issue.
2. If the student solution is correct, set has_error=false.
3. If the final answer is wrong, there must usually be an earlier error. Find the earliest one.
4. Return a short exact substring from the student solution as earliest_error_span.
5. The substring must appear verbatim in the student solution.
6. If the solution is too incomplete to judge, set has_error=true and use the earliest span where it becomes insufficient or goes off track.
7. Output JSON only. No markdown.

Problem:
{problem}

{ref_block}

Student solution:
{student_cot}

Return JSON with this schema:
{{
  "has_error": true or false,
  "earliest_error_span": "exact substring from student solution, or empty string if no error",
  "error_type": "wrong_setup | wrong_formula | algebra_error | arithmetic_error | invalid_inference | contradiction | incomplete | final_answer_only | other | none",
  "explanation": "brief explanation"
}}
"""


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()

    # Remove markdown fences if any.
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    # Try direct.
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try first JSON object.
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    return {
        "has_error": True,
        "earliest_error_span": "",
        "error_type": "parse_failed",
        "explanation": f"Could not parse judge output: {text[:500]}",
    }


# -------------------------
# Token position helper
# -------------------------

def find_span_token_position(
    text: str,
    span: str,
    tokenizer,
) -> Optional[int]:
    """
    Return 1-indexed generated-token position where span begins.
    If not found, return None.
    """
    if not span:
        return None

    char_pos = text.find(span)
    if char_pos < 0:
        # Try normalized whitespace search.
        norm_text = re.sub(r"\s+", " ", text)
        norm_span = re.sub(r"\s+", " ", span)
        norm_pos = norm_text.find(norm_span)
        if norm_pos < 0:
            return None

        # Cannot reliably map normalized char pos back.
        return None

    prefix = text[:char_pos]
    token_ids = tokenizer.encode(prefix, add_special_tokens=False)
    return len(token_ids) + 1


# -------------------------
# Teacher generation
# -------------------------

def load_model_and_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


@torch.no_grad()
def generate_judgment(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    messages = [
        {"role": "user", "content": prompt}
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        input_text = prompt

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    gen = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=str, required=True,
                        help="Path to lm-eval output dir/json/jsonl.")
    parser.add_argument("--teacher", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--student-tokenizer", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--threshold", type=int, default=100)
    parser.add_argument("--out", type=str, default="outputs/first_error_judged.jsonl")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--use-reference-cot", action="store_true")
    parser.add_argument("--judge-all", action="store_true",
                        help="Judge all samples. By default, only judge samples marked wrong by lm-eval if correctness is available.")
    args = parser.parse_args()

    samples_path = Path(args.samples)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading teacher: {args.teacher}")
    teacher_model, teacher_tok = load_model_and_tokenizer(args.teacher)

    print(f"Loading student tokenizer: {args.student_tokenizer}")
    student_tok = AutoTokenizer.from_pretrained(args.student_tokenizer, trust_remote_code=True)

    records = list(iter_json_records(samples_path))
    print(f"Loaded {len(records)} raw records from {samples_path}")

    total_seen = 0
    judged = 0
    wrong_samples_judged = 0
    error_found = 0
    le_threshold = 0
    gt_threshold = 0
    no_position = 0
    skipped_correct = 0
    unknown_correctness = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for idx, sample in enumerate(records):
            if args.max_cases is not None and judged >= args.max_cases:
                break

            total_seen += 1

            correctness = get_correctness_from_sample(sample)
            if correctness is None:
                unknown_correctness += 1

            # By default, only judge wrong samples if lm-eval correctness is available.
            if not args.judge_all and correctness is True:
                skipped_correct += 1
                continue

            problem = get_problem_from_sample(sample)
            student_cot = get_generation_from_sample(sample)
            reference_solution = get_reference_solution_from_sample(sample)

            if not problem or not student_cot:
                continue

            prompt = build_judge_prompt(
                problem=problem,
                student_cot=student_cot,
                reference_solution=reference_solution,
                use_reference_cot=args.use_reference_cot,
            )

            raw_judge = generate_judgment(teacher_model, teacher_tok, prompt)
            judge = extract_json(raw_judge)

            has_error = bool(judge.get("has_error", True))
            span = str(judge.get("earliest_error_span", "") or "")

            token_pos = None
            within_threshold = None

            if has_error and span:
                token_pos = find_span_token_position(student_cot, span, student_tok)
                if token_pos is not None:
                    error_found += 1
                    within_threshold = token_pos <= args.threshold
                    if within_threshold:
                        le_threshold += 1
                    else:
                        gt_threshold += 1
                else:
                    no_position += 1

            # We are mainly interested in wrong samples.
            if correctness is False or correctness is None or args.judge_all:
                wrong_samples_judged += 1

            judged += 1

            out_obj = {
                "idx": idx,
                "correctness_from_lm_eval": correctness,
                "problem": problem,
                "reference_solution": reference_solution,
                "student_cot": student_cot,
                "judge": judge,
                "raw_judge": raw_judge,
                "earliest_error_span": span,
                "earliest_error_token_pos": token_pos,
                "threshold": args.threshold,
                "within_threshold": within_threshold,
            }

            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            fout.flush()

            print(
                f"[{judged}] idx={idx} correct={correctness} "
                f"has_error={has_error} pos={token_pos} within={within_threshold}"
            )

    summary = {
        "samples_path": str(samples_path),
        "teacher": args.teacher,
        "student_tokenizer": args.student_tokenizer,
        "threshold": args.threshold,
        "total_seen": total_seen,
        "judged": judged,
        "skipped_correct": skipped_correct,
        "unknown_correctness": unknown_correctness,
        "wrong_samples_judged_or_unknown": wrong_samples_judged,
        "wrong_with_error_position_found": error_found,
        "earliest_error_le_threshold": le_threshold,
        "earliest_error_gt_threshold": gt_threshold,
        "error_span_found_but_position_not_found": no_position,
        "ratio_error_le_threshold_among_found": (
            le_threshold / error_found if error_found > 0 else None
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