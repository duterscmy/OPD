#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_all_adaptive_experiments.sh
# Optional overrides:
#   EXTRA="--set max_steps=50 --set max_train_examples=512" bash scripts/run_all_adaptive_experiments.sh

EXTRA=${EXTRA:-""}
CONFIGS=(
  configs/math_full_reverse.yaml
  configs/math_full_forward.yaml
  configs/math_esr50_reverse.yaml
  configs/math_esr100_reverse.yaml
  configs/math_esr200_reverse.yaml
  configs/math_prune_opd_lite.yaml
  configs/math_fixed_mixture.yaml
  configs/math_adaptive_kl_token.yaml
  configs/math_adaptive_kl_stage.yaml
)

for cfg in "${CONFIGS[@]}"; do
  echo "================================================================"
  echo "Running ${cfg}"
  echo "================================================================"
  bash train.sh "${cfg}" ${EXTRA}
done
