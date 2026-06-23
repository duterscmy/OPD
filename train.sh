#!/bin/bash
#SBATCH --job-name="train_opd"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err


### 激活 conda 环境
source ~/.bashrc
conda activate opd

### 必须传 config
if [ $# -lt 1 ]; then
  echo "Error: missing config file."
  echo "Usage: sbatch train.sh configs/math_esr100.yaml"
  exit 1
fi

CONFIG="$1"
shift

if [ ! -f "$CONFIG" ]; then
  echo "Error: config file not found: $CONFIG"
  exit 1
fi

### 单卡默认配置
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

NUM_PROCESSES=1
MIXED_PRECISION=${MIXED_PRECISION:-bf16}

echo "======================================"
echo "Job started at: $(date)"
echo "Config: $CONFIG"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "NUM_PROCESSES: $NUM_PROCESSES"
echo "MIXED_PRECISION: $MIXED_PRECISION"
echo "======================================"

mkdir -p logs

LOG_FILE="logs/train_$(basename "$CONFIG" .yaml)_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG_FILE"

accelerate launch \
  --num_processes "$NUM_PROCESSES" \
  --mixed_precision "$MIXED_PRECISION" \
  train_opd.py \
  --config "$CONFIG" \
  "$@" > "$LOG_FILE" 2>&1

echo "======================================"
echo "Job finished at: $(date)"
echo "======================================"