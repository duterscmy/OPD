#!/bin/bash
#SBATCH --job-name="eval_opd"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err


### 激活 conda 环境
source ~/.bashrc
conda activate opd

MODEL=$1
OUTPUT_PATH=$MODEL/eval
BASE_MODEL="/lus/lfs1aip2/projects/public/u6nc/mingyu/models/Qwen2.5-Math-1.5B"
BATCH_SIZE=8
# BATCH_SIZE=${BATCH_SIZE:-auto}
# DEVICE=${DEVICE:-cuda:0}
# LIMIT=${LIMIT:-}
APPLY_CHAT_TEMPLATE="1"
GEN_KWARGS=${GEN_KWARGS:-max_gen_toks=256,temperature=0.7,do_sample=True,top_p=1.0}
NUM_FEWSHOT=${NUM_FEWSHOT:-0}

mkdir -p "$OUTPUT_PATH"
EXTRA=()
if [[ -n "$LIMIT" ]]; then
  EXTRA+=(--limit "$LIMIT")
fi
if [[ "$APPLY_CHAT_TEMPLATE" == "1" ]]; then
  EXTRA+=(--apply_chat_template)
fi

echo $EXTRA

lm_eval \
  --model hf \
  --model_args "pretrained=${BASE_MODEL},peft=${MODEL},trust_remote_code=True,dtype=bfloat16",max_gen_toks=256,temperature=0.7,do_sample=True,top_p=1.0 \
  --tasks "minerva_math500" \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --num_fewshot "$NUM_FEWSHOT" \
  --gen_kwargs "$GEN_KWARGS" \
  --log_samples \
  --output_path "$OUTPUT_PATH" \
  "${EXTRA[@]}" #&> "$OUTPUT_PATH/eval.log"
