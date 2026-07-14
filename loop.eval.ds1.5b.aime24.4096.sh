#!/bin/bash
#SBATCH --job-name="eval_aime24"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err


source ~/.bashrc
conda activate opd


if [ $# -lt 1 ]; then
  echo "Usage: sbatch eval_all_checkpoints_aime.sh /path/to/experiment_dir [min_checkpoint_step]"
  echo "Example: sbatch eval_all_checkpoints_aime.sh outputs/ds_justrl_adaptive"
  exit 1
fi


ROOT_DIR="$1"
MIN_CKPT_STEP="${2:-0}"


if ! [[ "$MIN_CKPT_STEP" =~ ^[0-9]+$ ]]; then
  echo "Error: min_checkpoint_step must be integer"
  exit 1
fi


if [ ! -d "$ROOT_DIR" ]; then
  echo "Error: ROOT_DIR does not exist: $ROOT_DIR"
  exit 1
fi


BASE_MODEL=${BASE_MODEL:-"/lus/lfs1aip2/projects/public/u6nc/mingyu/models/DeepSeek-R1-Distill-Qwen-1.5B"}

# AIME24
TASKS=${TASKS:-"aime24"}

BATCH_SIZE=${BATCH_SIZE:-4}

DEVICE=${DEVICE:-"cuda:0"}

LIMIT=${LIMIT:-}

APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-"1"}

# Reasoning benchmark setting
GEN_KWARGS=${GEN_KWARGS:-"max_gen_toks=4096,temperature=0.6,do_sample=True,top_p=0.95"}

NUM_FEWSHOT=${NUM_FEWSHOT:-0}



LOG_ROOT="${ROOT_DIR}/eval_aime24"
mkdir -p "$LOG_ROOT"


echo "======================================"
echo "Eval root dir: $ROOT_DIR"
echo "Base model: $BASE_MODEL"
echo "Task: $TASKS"
echo "Batch size: $BATCH_SIZE"
echo "Generation: $GEN_KWARGS"
echo "Fewshot: $NUM_FEWSHOT"
echo "======================================"


mapfile -t CHECKPOINTS < <(
  find "$ROOT_DIR" -maxdepth 1 -type d -name "checkpoint-*" \
    | awk -v min_step="$MIN_CKPT_STEP" '
        {
          path=$0
          name=$0
          sub(/^.*checkpoint-/, "", name)

          if(name ~ /^[0-9]+$/ && name+0 >= min_step)
              print path
        }
      ' \
    | sort -V
)


if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
  echo "No checkpoints found"
  exit 1
fi


echo "Found checkpoints:"
printf " %s\n" "${CHECKPOINTS[@]}"



for MODEL in "${CHECKPOINTS[@]}"; do

  CKPT_NAME=$(basename "$MODEL")

  OUTPUT_PATH="${LOG_ROOT}/${CKPT_NAME}"

  mkdir -p "$OUTPUT_PATH"


  EXTRA=()


  if [[ -n "${LIMIT}" ]]; then
      EXTRA+=(--limit "$LIMIT")
  fi


  if [[ "$APPLY_CHAT_TEMPLATE" == "1" ]]; then
      EXTRA+=(--apply_chat_template)
  fi



  echo ""
  echo "======================================"
  echo "Evaluating $CKPT_NAME"
  echo "Started: $(date)"
  echo "======================================"


  lm_eval \
    --model hf \
    --model_args "pretrained=${BASE_MODEL},peft=${MODEL},trust_remote_code=True,dtype=bfloat16" \
    --tasks "$TASKS" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --num_fewshot "$NUM_FEWSHOT" \
    --gen_kwargs "$GEN_KWARGS" \
    --log_samples \
    --output_path "$OUTPUT_PATH" \
    "${EXTRA[@]}" \
    &> "$OUTPUT_PATH/eval.aime24.length4096.log"



  echo "Finished $CKPT_NAME at $(date)"

done


echo ""
echo "All AIME24 checkpoints evaluated."
echo "Results saved under $LOG_ROOT"