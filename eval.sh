#!/bin/bash
#SBATCH --job-name="eval_opd"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

# set -euo pipefail
source ~/.bashrc
conda activate opd

MODEL=${MODEL:-${1:-}}
if [ -z "$MODEL" ]; then
  echo "Usage: MODEL=/path/to/model bash eval.sh"
  exit 1
fi
TASKS=${TASKS:-minerva_math500}
DEVICE=${DEVICE:-cuda:0}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_FEWSHOT=${NUM_FEWSHOT:-0}
GEN_KWARGS=${GEN_KWARGS:-"max_gen_toks=1024,temperature=0.0,do_sample=False"}
APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-0}
OUTPUT_PATH=${OUTPUT_PATH:-eval_results/$(basename "$MODEL")}
mkdir -p "$OUTPUT_PATH"

EXTRA=()
if [ "${APPLY_CHAT_TEMPLATE}" = "1" ]; then
  EXTRA+=(--apply_chat_template)
fi
if [ -n "${LIMIT:-}" ]; then
  EXTRA+=(--limit "$LIMIT")
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
  "${EXTRA[@]}" 2>&1 | tee "$OUTPUT_PATH/eval.$TASKS.log"
