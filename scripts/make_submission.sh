#!/bin/bash
# Usage: bash scripts/make_submission.sh <config.yaml> [threshold]
# 전체 fold 체크포인트를 앙상블해 submission.csv 생성.
CFG=${1:?Usage: $0 <config.yaml> [threshold]}
THR=${2:-0.5}

EXP_NAME=$(python -c "
from src.utils.config import load_config
cfg = load_config('$CFG')
print(cfg.exp_name)
")

echo "실험: $EXP_NAME  threshold: $THR"

python inference.py \
    --config "$CFG" \
    --ckpt "outputs/$EXP_NAME" \
    --threshold "$THR"

echo "완료 → outputs/$EXP_NAME/submission.csv"
