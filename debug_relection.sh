### 激活 conda 环境
source ~/.bashrc
conda activate opd

python debug_reflection.py \
  --config configs/math_reflection.yaml \
  --num-samples 5 \
  --student-max-new-tokens 1024 \
  --teacher-max-new-tokens 512 \
  --chunk-size 16 \
  --output-jsonl outputs/debug_reflection_5.jsonl