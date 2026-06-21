# OPD / ESR experiments with TRL GKDTrainer

This package implements four rollout-horizon strategies on top of Hugging Face TRL's experimental
`GKDTrainer`:

- `full`: full-rollout OPD up to `full_max_new_tokens`.
- `esr`: fixed prefix horizon, e.g. 50/100/200 tokens.
- `curriculum`: a schedule such as 50 → 100 → 200 tokens.
- `reflection`: generate a longer student rollout, split it into 16-token chunks, ask the teacher for the
  earliest erroneous chunk, and distill only the preceding correct prefix.

## Important tokenizer detail

TRL's native full-vocabulary GKD loss assumes that teacher and student share a tokenizer and vocabulary.
The default pair in this repository, Qwen2.5-Math-1.5B → Qwen3-4B, is a cross-generation pair. In `auto`
mode the code therefore uses:

- identical tokenizer: native TRL full-vocabulary generalized JSD (`beta=1` is the reverse-KL limit);
- different tokenizers: a sampled reverse-KL policy-gradient estimator on greedy character-span-aligned
  student/teacher token groups.

The second path follows the cross-tokenizer idea described in the ESR paper, but is an approximation rather
than an exact full-vocabulary KL because the vocabularies differ.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
accelerate config
```

TRL 1.6.0 is pinned because `GKDTrainer` is under `trl.experimental.gkd` in that release.

Preview the mapped dataset before a long run:

```bash
python inspect_data.py --config configs/math_esr100.yaml --num-examples 3
```

## Train

Paper-style defaults are: reverse KL, learning rate `5e-5`, temperature `0.7`, LoRA `r=32, alpha=64`,
global batch size 16, one rollout per problem, 200 optimizer steps, checkpoints every 50 steps.

```bash
# Fixed ESR, N=100
./train.sh configs/math_esr100.yaml

# ESR sweeps
./train.sh configs/math_esr100.yaml --set prefix_length=50 \
  --set output_dir=outputs/math_esr50
./train.sh configs/math_esr100.yaml --set prefix_length=200 \
  --set output_dir=outputs/math_esr200
# Or run 50/100/200 sequentially
./sweep_esr.sh configs/math_esr100.yaml

# Full rollout OPD
./train.sh configs/math_full.yaml

# 50 -> 100 -> 200 curriculum
./train.sh configs/math_curriculum.yaml

# Teacher-reflection truncation, 16-token chunks
./train.sh configs/math_reflection.yaml
```

The curriculum boundaries are optimizer-step boundaries. The default `[0, 67, 134]` divides 200 steps into
roughly three equal phases.

### Multi-GPU

The paper uses a global batch of 16. On one GPU the config uses micro-batch 1 and gradient accumulation 16.
For two GPUs, for example:

```bash
NUM_PROCESSES=2 ./train.sh configs/math_esr100.yaml \
  --set gradient_accumulation_steps=8
```

## Datasets

Adapters are included for:

- `AI-MO/NuminaMath-1.5` (`dataset_adapter: math`)
- `coseal/CodeUltraFeedback` (`dataset_adapter: code`)
- `glaiveai/glaive-function-calling-v2` (`dataset_adapter: function_calling`)

The Glaive raw format is parsed conservatively and uses the first assistant turn as the target. For a
production function-calling experiment, inspect a few mapped examples first because raw formatting can
change across dataset revisions.

## Evaluate with lm-eval-harness

The training output is normally a LoRA adapter. Merge it first:

```bash
python merge_lora.py \
  --adapter outputs/qwen25_math_15b_to_qwen3_4b_esr100 \
  --output outputs/qwen25_math_15b_to_qwen3_4b_esr100_merged
```

Then evaluate MATH-500:

```bash
MODEL=outputs/qwen25_math_15b_to_qwen3_4b_esr100_merged \
TASKS=hendrycks_math500 \
BATCH_SIZE=auto \
./eval.sh
```

Other examples:

```bash
# HumanEval; deterministic pass@1-like generation
MODEL=... TASKS=humaneval \
GEN_KWARGS='max_gen_toks=1024,temperature=0.0,do_sample=False' ./eval.sh

# Check task names in the installed harness
lm-eval ls tasks | grep -Ei 'math500|humaneval|gsm8k'
```

BFCL is not a standard lm-eval task in many installations; use the official BFCL evaluator for leaderboard-
comparable function-calling results.

## Reflection strategy semantics

Reflection mode is quality-oriented, not compute-efficient:

1. Generate up to `reflection_rollout_max_tokens`.
2. Split the sampled student completion into `reflection_chunk_size`-token regions.
3. Ask the teacher for the earliest erroneous chunk.
4. Exclude that chunk and everything after it from the distillation loss.
5. If the teacher says the solution is correct, use the full sampled rollout.

If JSON parsing fails, the default fallback uses the first 100 tokens. Set `reflection_parse_failure` to
`full` or `skip` to change this behavior. Per-sample decisions are written to
`OUTPUT_DIR/reflection_rank*.jsonl`.

## Limitations

- The cross-tokenizer sampled reverse-KL loss is not the same as exact full-vocabulary reverse KL.
- Teacher reflection is an LLM-judge signal and should be manually audited on a sample.
- The ESR paper does not disclose the exact full-rollout generation cap; this repository defaults to 1024
  and makes it configurable.
- LoRA target modules default to `all-linear`; the paper reports LoRA rank/alpha but not a complete target-
  module list.
- The paper does not fully specify optimizer/scheduler details. This package uses AdamW with a constant
  learning-rate schedule by default; both are editable in the YAML/config code.
