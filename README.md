# Rewritten OPD / ESR / Reflection Code

This rewrite adds three requested capabilities:

1. Separate config switches for student and teacher chat templates:
   - `student_use_chat_template`
   - `teacher_use_chat_template`
   - `student_enable_thinking`
   - `teacher_enable_thinking`

2. Per-step debug logs with clear markers:
   - `【student prompt】`
   - `【student rollout】`
   - `【student rollout length】`
   - `【student rollout ans】`
   - `【ground truth ans】`
   - `【student correct】`
   - `【cut tokens】`

3. New strategy `correctness_esr`:
   - Generate a long student rollout.
   - Extract the student answer and compare with the ground-truth answer.
   - If correct: use the full rollout for OPD.
   - If wrong: use ESR prefix length, e.g. first 100 tokens.

## Important default

For `Qwen/Qwen2.5-Math-1.5B` base student, the config sets:

```yaml
student_use_chat_template: false
teacher_use_chat_template: true
```

This avoids the previously observed issue where the base model continued the chat/template data distribution rather than solving the math problem.

## Training

```bash
bash train.sh configs/math_esr100.yaml
bash train.sh configs/math_correctness_esr.yaml
bash train.sh configs/math_reflection.yaml
```

Resume:

```bash
bash train.sh configs/math_reflection.yaml \
  --set resume_from_checkpoint=outputs/qwen25_math_15b_to_qwen3_4b_reflection/checkpoint-100
```

## Strategies

```yaml
strategy: full
strategy: esr
strategy: curriculum
strategy: reflection
strategy: correctness_esr
```

## Evaluation

Merge LoRA first:

```bash
python merge_lora.py \
  --base Qwen/Qwen2.5-Math-1.5B \
  --adapter outputs/qwen25_math_15b_to_qwen3_4b_esr100/checkpoint-200 \
  --output outputs/esr100_merged
```

Then evaluate:

```bash
MODEL=outputs/esr100_merged TASKS=minerva_math500 bash eval.sh
```
