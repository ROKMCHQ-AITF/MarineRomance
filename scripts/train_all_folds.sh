#!/bin/bash
# Usage: bash scripts/train_all_folds.sh configs/exp002_advanced.yaml
CFG=${1:?Usage: $0 <config.yaml>}
for f in 0 1 2 3 4; do
    python main.py --config "$CFG" "train.folds=[$f]"
done
