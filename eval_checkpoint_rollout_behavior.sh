#!/bin/bash
#SBATCH --job-name="ckpt_behavior"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

set -eo pipefail

source ~/.bashrc
conda activate opd

if [ $# -lt 1 ]; then
  echo "Usage: sbatch eval_checkpoint_rollout_behavior.sh EXPERIMENT_DIR [extra Python args]"
  echo "Example: sbatch eval_checkpoint_rollout_behavior.sh outputs/my_run"
  echo "Example: NUM_SAMPLES=200 BATCH_SIZE=4 sbatch eval_checkpoint_rollout_behavior.sh outputs/my_run"
  echo "Post-hoc only: DIAGNOSTICS_ONLY=1 sbatch eval_checkpoint_rollout_behavior.sh outputs/my_run"
  exit 1
fi

EXPERIMENT_DIR="$1"
shift

if [ ! -d "$EXPERIMENT_DIR" ]; then
  echo "Error: experiment directory not found: $EXPERIMENT_DIR"
  exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
NUM_SAMPLES=${NUM_SAMPLES:-150}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-3072}
DEVICE=${DEVICE:-cuda:0}
OUTPUT_DIR=${OUTPUT_DIR:-"${EXPERIMENT_DIR}/checkpoint_behavior_eval"}
RUN_LOGIT_DIAGNOSTICS=${RUN_LOGIT_DIAGNOSTICS:-1}
DIAGNOSTICS_ONLY=${DIAGNOSTICS_ONLY:-0}
SAVE_TOKEN_IDS=${SAVE_TOKEN_IDS:-1}
DIAGNOSTICS_OUTPUT_DIR=${DIAGNOSTICS_OUTPUT_DIR:-"${OUTPUT_DIR}/logit_diagnostics"}
ANALYSIS_BATCH_SIZE=${ANALYSIS_BATCH_SIZE:-1}
VOCAB_CHUNK_SIZE=${VOCAB_CHUNK_SIZE:-8192}
FIXED_PREFIX_ANALYSIS=${FIXED_PREFIX_ANALYSIS:-1}
FIXED_PREFIX_SOURCE=${FIXED_PREFIX_SOURCE:-first}
FIXED_PREFIX_MAX_SAMPLES=${FIXED_PREFIX_MAX_SAMPLES:-32}
DIAGNOSTICS_OVERWRITE=${DIAGNOSTICS_OVERWRITE:-0}

EXTRA_ARGS=()
if [ -n "${BASE_MODEL:-}" ]; then
  EXTRA_ARGS+=(--base-model "$BASE_MODEL")
fi
if [ -n "${TEACHER_TOKENIZER:-}" ]; then
  EXTRA_ARGS+=(--teacher-tokenizer "$TEACHER_TOKENIZER")
fi
if [ "$SAVE_TOKEN_IDS" = "1" ]; then
  EXTRA_ARGS+=(--save-token-ids)
fi

mkdir -p "$OUTPUT_DIR"
RUN_STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${OUTPUT_DIR}/eval_checkpoint_behavior_${RUN_STAMP}.log"

echo "======================================"
echo "Experiment directory: $EXPERIMENT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Samples: $NUM_SAMPLES"
echo "Batch size: $BATCH_SIZE"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Device: $DEVICE"
echo "Save exact completion token IDs: $SAVE_TOKEN_IDS"
echo "Run logit diagnostics: $RUN_LOGIT_DIAGNOSTICS"
echo "Diagnostics only: $DIAGNOSTICS_ONLY"
echo "Fixed-prefix analysis: $FIXED_PREFIX_ANALYSIS (source=$FIXED_PREFIX_SOURCE, samples=$FIXED_PREFIX_MAX_SAMPLES)"
echo "Extra args: $*"
echo "Log file: $LOG_FILE"
echo "======================================"

if [ "$DIAGNOSTICS_ONLY" != "1" ]; then
  python -u eval_checkpoint_rollout_behavior.py \
    "$EXPERIMENT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --num-samples "$NUM_SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --device "$DEVICE" \
    "${EXTRA_ARGS[@]}" \
    "$@" 2>&1 | tee "$LOG_FILE"
fi

if [ "$RUN_LOGIT_DIAGNOSTICS" = "1" ]; then
  DIAGNOSTIC_ARGS=(
    --rollout-output-dir "$OUTPUT_DIR"
    --output-dir "$DIAGNOSTICS_OUTPUT_DIR"
    --device "$DEVICE"
    --batch-size "$ANALYSIS_BATCH_SIZE"
    --vocab-chunk-size "$VOCAB_CHUNK_SIZE"
    --fixed-prefix-source "$FIXED_PREFIX_SOURCE"
    --fixed-prefix-max-samples "$FIXED_PREFIX_MAX_SAMPLES"
  )
  if [ "$FIXED_PREFIX_ANALYSIS" != "1" ]; then
    DIAGNOSTIC_ARGS+=(--fixed-prefix-source none)
  fi
  if [ -n "${BASE_MODEL:-}" ]; then
    DIAGNOSTIC_ARGS+=(--base-model "$BASE_MODEL")
  fi
  if [ -n "${TEACHER_MODEL:-}" ]; then
    DIAGNOSTIC_ARGS+=(--teacher-model "$TEACHER_MODEL")
  fi
  if [ -n "${ANALYSIS_TOP_K:-}" ]; then
    DIAGNOSTIC_ARGS+=(--analysis-top-k "$ANALYSIS_TOP_K")
  fi
  if [ -n "${ANALYSIS_CHECKPOINT_STEPS:-}" ]; then
    DIAGNOSTIC_ARGS+=(--checkpoint-steps "$ANALYSIS_CHECKPOINT_STEPS")
  fi
  if [ -n "${ANALYSIS_MAX_SAMPLES:-}" ]; then
    DIAGNOSTIC_ARGS+=(--max-samples "$ANALYSIS_MAX_SAMPLES")
  fi
  if [ "$DIAGNOSTICS_OVERWRITE" = "1" ]; then
    DIAGNOSTIC_ARGS+=(--overwrite)
  fi
  if [ "${REQUIRE_SAVED_TOKEN_IDS:-0}" = "1" ]; then
    DIAGNOSTIC_ARGS+=(--require-saved-token-ids)
  fi
  if [ "${LOCAL_FILES_ONLY:-0}" = "1" ]; then
    DIAGNOSTIC_ARGS+=(--local-files-only)
  fi

  DIAGNOSTICS_LOG_FILE="${DIAGNOSTICS_OUTPUT_DIR}/logit_diagnostics_${RUN_STAMP}.log"
  mkdir -p "$DIAGNOSTICS_OUTPUT_DIR"
  python -u analyze_checkpoint_rollout_logits.py \
    "$EXPERIMENT_DIR" \
    "${DIAGNOSTIC_ARGS[@]}" 2>&1 | tee "$DIAGNOSTICS_LOG_FILE"
fi
