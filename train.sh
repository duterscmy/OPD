#!/bin/bash
#SBATCH --job-name="train_opd"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

# set -euo pipefail

source ~/.bashrc
conda activate opd

if [ $# -lt 1 ]; then
  echo "Error: missing config file."
  echo "Usage: sbatch train.sh configs/math_esr100.yaml [--set key=value ...]"
  exit 1
fi

CONFIG="$1"
shift

if [ ! -f "$CONFIG" ]; then
  echo "Error: config file not found: $CONFIG"
  exit 1
fi

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
NUM_PROCESSES=${NUM_PROCESSES:-1}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}

mkdir -p logs
LOG_FILE="logs/train_$(basename "$CONFIG" .yaml)_$(date +%Y%m%d_%H%M%S).log"

echo "======================================"
echo "Job started at: $(date)"
echo "Config: $CONFIG"
echo "Extra args: $*"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not_set}"
echo "NUM_PROCESSES: $NUM_PROCESSES"
echo "MIXED_PRECISION: $MIXED_PRECISION"
echo "Logging to: $LOG_FILE"
echo "======================================"

accelerate launch \
  --num_processes "$NUM_PROCESSES" \
  --mixed_precision "$MIXED_PRECISION" \
  train_opd.py \
  --config "$CONFIG" \
  "$@" &> "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo "======================================"
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================"
exit "$EXIT_CODE"
