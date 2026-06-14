"""train_metadata.csv → folds.csv + label_map.json 생성.

StratifiedKFold (single-label) 또는 GroupKFold (group_col 지정 시) 를 config에 따라 선택.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

from src.utils.config import load_config


def build_label_map(df: pd.DataFrame, label_col: str) -> dict[str, int]:
    """모든 unique label을 수집해 정렬 후 {name: idx} 매핑 반환."""
    labels: set[str] = set()
    for raw in df[label_col].dropna().astype(str):
        for tok in raw.split():
            labels.add(tok)
    return {name: i for i, name in enumerate(sorted(labels))}


def main() -> None:
    parser = argparse.ArgumentParser(description="K-Fold 분할 및 label_map 생성")
    parser.add_argument("--config", required=True, help="실험 yaml 경로")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    data_cfg = cfg.data

    train_csv = Path(data_cfg.train_csv)
    folds_csv = Path(data_cfg.folds_csv)
    folds_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_csv)
    print(f"로드 완료: {len(df)} rows from {train_csv}")

    # label_map 생성 및 저장
    label_map = build_label_map(df, data_cfg.label_col)
    label_map_path = folds_csv.parent / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"label_map 저장: {len(label_map)} classes → {label_map_path}")

    # fold 분할
    n_folds = cfg.train.n_folds
    df["fold"] = -1

    if data_cfg.group_col:
        kf = GroupKFold(n_splits=n_folds)
        groups = df[data_cfg.group_col].values
        for fold_idx, (_, val_idx) in enumerate(kf.split(df, groups=groups)):
            df.loc[val_idx, "fold"] = fold_idx
        print(f"GroupKFold (group={data_cfg.group_col}, n_folds={n_folds})")
    else:
        # StratifiedKFold: primary_label의 첫 번째 토큰으로 stratify
        stratify_labels = df[data_cfg.label_col].astype(str).str.split().str[0]
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg.seed)
        for fold_idx, (_, val_idx) in enumerate(kf.split(df, y=stratify_labels)):
            df.loc[val_idx, "fold"] = fold_idx
        print(f"StratifiedKFold (n_folds={n_folds})")

    df.to_csv(folds_csv, index=False)
    print(f"folds.csv 저장 → {folds_csv}")

    # fold별 샘플 수 요약
    print("\n[fold 분포]")
    print(df["fold"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
