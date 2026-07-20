#!/bin/bash
#SBATCH --job-name="train_adaptive"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

# set -euo pipefail

source ~/.bashrc
conda activate opd

if [ $# -lt 1 ]; then
  echo "Error: missing config file."
  echo "Usage: sbatch train_adaptive.sh configs/math_adaptive_kl_token.yaml [--set key=value ...]"
  echo ""
  echo "Examples:"
  echo "  sbatch train_adaptive.sh configs/math_full_reverse.yaml"
  echo "  sbatch train_adaptive.sh configs/math_adaptive_kl_token.yaml --set max_steps=20 --set max_train_examples=512"
  exit 1
fi

CONFIG="$1"
shift

if [ ! -f "$CONFIG" ]; then
  echo "Error: config file not found: $CONFIG"
  exit 1
fi

TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_opd_adaptive.py}

if [ ! -f "$TRAIN_SCRIPT" ]; then
  echo "Error: training script not found: $TRAIN_SCRIPT"
  echo "Please make sure train_opd_adaptive.py is in the current directory."
  exit 1
fi

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

NUM_PROCESSES=${NUM_PROCESSES:-4}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}

mkdir -p logs
LOG_FILE="logs/adaptive_$(basename "$CONFIG" .yaml)_$(date +%Y%m%d_%H%M%S).log"

echo "======================================"
echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Config: $CONFIG"
echo "Train script: $TRAIN_SCRIPT"
echo "Extra args: $*"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not_set}"
echo "NUM_PROCESSES: $NUM_PROCESSES"
echo "MIXED_PRECISION: $MIXED_PRECISION"
echo "Logging to: $LOG_FILE"
echo "======================================"

accelerate launch \
  --num_processes "$NUM_PROCESSES" \
  --mixed_precision "$MIXED_PRECISION" \
  "$TRAIN_SCRIPT" \
  --config "$CONFIG" \
  "$@" &> "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo "======================================"
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Log file: $LOG_FILE"
echo "======================================"

exit "$EXIT_CODE"