#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/math_esr100.yaml}
for N in 50 100 200; do
  echo "=== ESR N=${N} ==="
  ./train.sh "$CONFIG" \
    --set "prefix_length=${N}" \
    --set "output_dir=outputs/esr_${N}"
done
