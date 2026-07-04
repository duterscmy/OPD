#!/bin/bash
set -euo pipefail

COMMON_ARGS="--set max_steps=200 --set save_steps=50 --set save_only_model=true --set debug_print_every_step=false"

sbatch train_adaptive.sh configs/ds_justrl_full_reverse_topk.yaml $COMMON_ARGS
sbatch train_adaptive.sh configs/ds_justrl_full_forward_topk.yaml $COMMON_ARGS
sbatch train_adaptive.sh configs/ds_justrl_esr50_reverse_topk.yaml $COMMON_ARGS
sbatch train_adaptive.sh configs/ds_justrl_esr100_reverse_topk.yaml $COMMON_ARGS
sbatch train_adaptive.sh configs/ds_justrl_esr200_reverse_topk.yaml $COMMON_ARGS
sbatch train_adaptive.sh configs/ds_justrl_prune_opd_lite_topk.yaml $COMMON_ARGS
sbatch train_adaptive.sh configs/ds_justrl_fixed_mixture_fwd0p5_topk.yaml $COMMON_ARGS
sbatch train_adaptive.sh configs/ds_justrl_adaptive_kl_token_topk.yaml $COMMON_ARGS
sbatch train_adaptive.sh configs/ds_justrl_adaptive_kl_stage50_topk.yaml $COMMON_ARGS
