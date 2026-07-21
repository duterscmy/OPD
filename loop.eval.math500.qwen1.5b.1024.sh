#!/bin/bash
#SBATCH --job-name="eval_opd"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err


### 激活 conda 环境
source ~/.bashrc
conda activate opd

if [ $# -lt 1 ]; then
  echo "Usage: sbatch eval_all_checkpoints.sh /path/to/experiment_dir [min_checkpoint_step]"
  echo "Example: sbatch eval_all_checkpoints.sh outputs/qwen25_math_esr100"
  echo "Example: sbatch eval_all_checkpoints.sh outputs/qwen25_math_esr100 200"
  exit 1
fi

ROOT_DIR="$1"
MIN_CKPT_STEP="${2:-0}"

if ! [[ "$MIN_CKPT_STEP" =~ ^[0-9]+$ ]]; then
  echo "Error: min_checkpoint_step must be a non-negative integer, got: $MIN_CKPT_STEP"
  exit 1
fi

if [ ! -d "$ROOT_DIR" ]; then
  echo "Error: ROOT_DIR does not exist: $ROOT_DIR"
  exit 1
fi

BASE_MODEL=${BASE_MODEL:-"/lus/lfs1aip2/projects/public/u6nc/mingyu/models/Qwen2.5-Math-1.5B-1024"}
TASKS=${TASKS:-"minerva_math500"}
BATCH_SIZE=${BATCH_SIZE:-8}
DEVICE=${DEVICE:-"cuda:0"}
LIMIT=${LIMIT:-}
APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-"1"}
GEN_KWARGS=${GEN_KWARGS:-"max_gen_toks=1024,temperature=0.0,do_sample=False,top_p=1.0"}
NUM_FEWSHOT=${NUM_FEWSHOT:-0}

LOG_ROOT="${ROOT_DIR}/eval"
mkdir -p "$LOG_ROOT"

echo "======================================"
echo "Eval root dir: $ROOT_DIR"
echo "Min checkpoint step: $MIN_CKPT_STEP"
echo "Base model: $BASE_MODEL"
echo "Tasks: $TASKS"
echo "Batch size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo "Gen kwargs: $GEN_KWARGS"
echo "Num fewshot: $NUM_FEWSHOT"
echo "Apply chat template: $APPLY_CHAT_TEMPLATE"
echo "Limit: ${LIMIT:-none}"
echo "======================================"

mapfile -t CHECKPOINTS < <(
  find "$ROOT_DIR" -maxdepth 1 -type d -name "checkpoint-*" \
    | awk -v min_step="$MIN_CKPT_STEP" '
        {
          path = $0
          name = $0
          sub(/^.*checkpoint-/, "", name)
          if (name ~ /^[0-9]+$/ && name + 0 >= min_step) {
            print path
          }
        }
      ' \
    | sort -V
)

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
  echo "Error: no checkpoint-* directories >= checkpoint-${MIN_CKPT_STEP} found under $ROOT_DIR"
  exit 1
fi

echo "Found ${#CHECKPOINTS[@]} checkpoints after filtering:"
printf '  %s\n' "${CHECKPOINTS[@]}"

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
  echo "Evaluating adapter: $MODEL"
  echo "Output path: $OUTPUT_PATH"
  echo "Started at: $(date)"
  echo "Extra args: ${EXTRA[*]:-none}"
  echo "======================================"

  lm_eval \
    --model hf \
    --model_args "pretrained=${BASE_MODEL},peft=${MODEL},trust_remote_code=True,dtype=bfloat16" \
    --tasks "$TASKS" \
    --device "$DEVICE" \
    --batch_size "auto" \
    --num_fewshot "$NUM_FEWSHOT" \
    --gen_kwargs "$GEN_KWARGS" \
    --log_samples \
    --output_path "$OUTPUT_PATH" \
    "${EXTRA[@]}" &> "$OUTPUT_PATH/eval.1024.log"

  echo "Finished $CKPT_NAME at: $(date)"
done

echo ""
echo "All checkpoints evaluated."
echo "Results saved under: $LOG_ROOT"