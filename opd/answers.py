from __future__ import annotations

import re
from typing import Any


def _last_boxed_content(text: str) -> str | None:
    starts = [m.start() for m in re.finditer(r"\\boxed\s*\{", text)]
    if not starts:
        return None
    start = starts[-1]
    brace = text.find("{", start)
    if brace < 0:
        return None
    depth = 0
    for i in range(brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1 : i].strip()
    return None


def extract_answer(text: str, mode: str = "auto") -> str:
    text = str(text or "").strip()
    if not text:
        return ""

    if mode in {"auto", "boxed"}:
        boxed = _last_boxed_content(text)
        if boxed is not None:
            return boxed
        if mode == "boxed":
            return ""

    if mode in {"auto", "answer_marker"}:
        # Match common markers, use the last occurrence.
        patterns = [
            r"####\s*([^\n]+)",
            r"Answer\s*:\s*([^\n]+)",
            r"Final answer\s*:?\s*([^\n]+)",
            r"The answer is\s*([^\n\.]+)",
        ]
        found = []
        for p in patterns:
            found.extend(re.findall(p, text, flags=re.IGNORECASE))
        if found:
            return str(found[-1]).strip().strip(".$")
        if mode == "answer_marker":
            return ""

    # Last number / simple expression fallback.
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", text)
    if nums:
        return nums[-1]
    return text.splitlines()[-1].strip() if text.splitlines() else text


def normalize_answer(ans: str) -> str:
    ans = str(ans or "").strip()
    ans = ans.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    ans = ans.replace("\\left", "").replace("\\right", "")
    ans = ans.replace("\\,", "").replace("\\!", "")
    ans = ans.replace("$", "")
    ans = ans.strip()
    # Remove common wrappers.
    ans = re.sub(r"^\\boxed\s*\{(.+)\}$", r"\1", ans)
    ans = ans.strip().strip(" .")
    # Normalize whitespace and commas.
    ans = re.sub(r"\s+", "", ans)
    ans = ans.replace(",", "")
    # Normalize simple LaTeX pi.
    ans = ans.replace("\\pi", "pi").replace("π", "pi")
    return ans.lower()


def answers_match(pred: str, gold: str) -> bool:
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    # Numeric fallback.
    try:
        return abs(float(p) - float(g)) < 1e-9
    except Exception:
        return False


def judge_correctness(student_text: str, ground_truth: str, reference_solution: str = "", mode: str = "auto") -> dict[str, Any]:
    pred = extract_answer(student_text, mode=mode)
    gold = str(ground_truth or "").strip()
    if not gold and reference_solution:
        gold = extract_answer(reference_solution, mode=mode)
    return {
        "student_answer": pred,
        "ground_truth_answer": gold,
        "is_correct": bool(answers_match(pred, gold)),
    }
