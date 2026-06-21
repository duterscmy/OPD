#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/math_esr100.yaml}
shift || true
NUM_PROCESSES=${NUM_PROCESSES:-1}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

accelerate launch \
  --num_processes "$NUM_PROCESSES" \
  --mixed_precision "$MIXED_PRECISION" \
  train_opd.py --config "$CONFIG" "$@"
