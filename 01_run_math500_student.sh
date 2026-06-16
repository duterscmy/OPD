#!/bin/bash
set -euo pipefail

# =========================
# Run student CoT on MATH-500 with lm-eval
# =========================

STUDENT_MODEL="Qwen/Qwen2.5-Math-1.5B"
TASK="minerva_math500"
LIMIT=200
OUT_DIR="outputs/math500_student_qwen25_15b"

# You can override this from command line.
# Example:
export GEN_KWARGS="max_gen_toks=512,temperature=0.7,do_sample=True,top_p=0.95"
# GEN_KWARGS=${GEN_KWARGS:-"max_gen_toks=2048,temperature=0.0,do_sample=False"}

mkdir -p "$OUT_DIR"

echo "Student model: $STUDENT_MODEL"
echo "Task: $TASK"
echo "Limit: $LIMIT"
echo "Output dir: $OUT_DIR"
echo "Generation kwargs: $GEN_KWARGS"

lm_eval \
  --model hf \
  --model_args "pretrained=${STUDENT_MODEL},trust_remote_code=True,dtype=bfloat16,device_map=auto" \
  --tasks "$TASK" \
  --limit "$LIMIT" \
  --batch_size 4 \
  --gen_kwargs "$GEN_KWARGS" \
  --log_samples \
  --output_path "$OUT_DIR"

echo "Done. Check samples under: $OUT_DIR"