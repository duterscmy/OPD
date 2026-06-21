#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:?Set MODEL to a merged model directory or Hub ID}
TASKS=${TASKS:-hendrycks_math500}
OUTPUT_PATH=${OUTPUT_PATH:-eval_outputs/$(basename "$MODEL")}
BATCH_SIZE=${BATCH_SIZE:-auto}
DEVICE=${DEVICE:-cuda:0}
LIMIT=${LIMIT:-}
APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-1}
GEN_KWARGS=${GEN_KWARGS:-max_gen_toks=2048,temperature=0.7,do_sample=True,top_p=1.0}
NUM_FEWSHOT=${NUM_FEWSHOT:-0}

mkdir -p "$OUTPUT_PATH"
EXTRA=()
if [[ -n "$LIMIT" ]]; then
  EXTRA+=(--limit "$LIMIT")
fi
if [[ "$APPLY_CHAT_TEMPLATE" == "1" ]]; then
  EXTRA+=(--apply_chat_template)
fi

lm_eval \
  --model hf \
  --model_args "pretrained=${MODEL},trust_remote_code=True,dtype=bfloat16" \
  --tasks "$TASKS" \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --num_fewshot "$NUM_FEWSHOT" \
  --gen_kwargs "$GEN_KWARGS" \
  --log_samples \
  --output_path "$OUTPUT_PATH" \
  "${EXTRA[@]}"
