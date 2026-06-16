python 02_judge_first_error.py \
  --samples outputs/math500_student_qwen25_15b \
  --teacher Qwen/Qwen3-4B \
  --student-tokenizer Qwen/Qwen2.5-Math-1.5B \
  --threshold 100 \
  --max-cases 200 \
  --out outputs/first_error_judged.jsonl \
  --use-reference-cot