#!/bin/bash
#SBATCH --job-name="ckpt_behavior"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

set -eo pipefail

source ~/.bashrc
conda activate opd

if [ $# -lt 1 ]; then
  echo "Usage: sbatch eval_checkpoint_rollout_behavior.sh EXPERIMENT_DIR [extra Python args]"
  echo "Example: sbatch eval_checkpoint_rollout_behavior.sh outputs/my_run"
  echo "Example: NUM_SAMPLES=200 BATCH_SIZE=4 sbatch eval_checkpoint_rollout_behavior.sh outputs/my_run"
  exit 1
fi

EXPERIMENT_DIR="$1"
shift

if [ ! -d "$EXPERIMENT_DIR" ]; then
  echo "Error: experiment directory not found: $EXPERIMENT_DIR"
  exit 1
fi

NUM_SAMPLES=${NUM_SAMPLES:-150}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-3072}
DEVICE=${DEVICE:-cuda:0}
OUTPUT_DIR=${OUTPUT_DIR:-"${EXPERIMENT_DIR}/checkpoint_behavior_eval"}

EXTRA_ARGS=()
if [ -n "${BASE_MODEL:-}" ]; then
  EXTRA_ARGS+=(--base-model "$BASE_MODEL")
fi
if [ -n "${TEACHER_TOKENIZER:-}" ]; then
  EXTRA_ARGS+=(--teacher-tokenizer "$TEACHER_TOKENIZER")
fi

mkdir -p "$OUTPUT_DIR"
LOG_FILE="${OUTPUT_DIR}/eval_checkpoint_behavior_$(date +%Y%m%d_%H%M%S).log"

echo "======================================"
echo "Experiment directory: $EXPERIMENT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Samples: $NUM_SAMPLES"
echo "Batch size: $BATCH_SIZE"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Device: $DEVICE"
echo "Extra args: $*"
echo "Log file: $LOG_FILE"
echo "======================================"

python -u eval_checkpoint_rollout_behavior.py  \
  "$EXPERIMENT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --num-samples "$NUM_SAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --device "$DEVICE" \
  "${EXTRA_ARGS[@]}" \
  "$@" 2>&1 | tee "$LOG_FILE"
