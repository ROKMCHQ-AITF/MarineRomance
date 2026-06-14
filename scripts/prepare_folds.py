"""train_metadata.csv → folds.csv + label_map.json 생성.

fold 전략:
  use_existing_fold=true  → fold 컬럼 그대로 사용
  group_col + multilabel  → StratifiedGroupKFold
  group_col only          → GroupKFold
  multilabel only         → iterative-stratification
  기본                    → StratifiedKFold
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)

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

    train_csv  = Path(data_cfg.train_csv)
    folds_csv  = Path(data_cfg.folds_csv)
    folds_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_csv)
    print(f"로드 완료: {len(df)} rows from {train_csv}")

    # label_map 생성 및 저장
    label_map = build_label_map(df, data_cfg.label_col)
    label_map_path = folds_csv.parent / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"label_map 저장: {len(label_map)} classes → {label_map_path}")
    print(f"num_classes = {len(label_map)}  ← config model.num_classes에 반영")

    # ── fold 분할 ──────────────────────────────────────────
    n_folds    = cfg.train.n_folds
    group_col  = getattr(data_cfg, "group_col", None)
    multilabel = getattr(data_cfg, "multilabel", False)
    use_existing = getattr(data_cfg, "use_existing_fold", False)
    fold_col   = getattr(data_cfg, "fold_col", "fold")  # 기존 fold 컬럼명

    df["fold"] = -1

    # 전략 1: fold 컬럼이 이미 있는 경우 (ESC-50 등)
    if use_existing:
        assert fold_col in df.columns, \
            f"use_existing_fold=true인데 '{fold_col}' 컬럼이 없습니다"
        df["fold"] = df[fold_col] - 1  # 1-based → 0-based로 통일
        print(f"기존 fold 사용 (컬럼: '{fold_col}', 0-based로 변환)")

    # 전략 2: group + multilabel → StratifiedGroupKFold
    elif group_col and not multilabel:
        groups = df[group_col].values
        stratify = df[data_cfg.label_col].astype(str).str.split().str[0]
        kf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                                   random_state=cfg.seed)
        for fold_idx, (_, val_idx) in enumerate(
                kf.split(df, y=stratify, groups=groups)):
            df.loc[val_idx, "fold"] = fold_idx
        print(f"StratifiedGroupKFold (group={group_col}, n_folds={n_folds})")

    # 전략 3: group만 있는 경우 → GroupKFold
    elif group_col and multilabel:
        groups = df[group_col].values
        kf = GroupKFold(n_splits=n_folds)
        for fold_idx, (_, val_idx) in enumerate(
                kf.split(df, groups=groups)):
            df.loc[val_idx, "fold"] = fold_idx
        print(f"GroupKFold (group={group_col}, n_folds={n_folds})")

    # 전략 4: multilabel → iterative-stratification
    elif multilabel:
        try:
            from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
            # multi-hot 행렬 생성
            mlb = np.zeros((len(df), len(label_map)), dtype=int)
            for i, raw in enumerate(df[data_cfg.label_col].astype(str)):
                for tok in raw.split():
                    if tok in label_map:
                        mlb[i, label_map[tok]] = 1
            kf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True,
                                            random_state=cfg.seed)
            for fold_idx, (_, val_idx) in enumerate(kf.split(df, mlb)):
                df.loc[val_idx, "fold"] = fold_idx
            print(f"MultilabelStratifiedKFold (n_folds={n_folds})")
        except ImportError:
            print("⚠ iterstrat 없음 → StratifiedKFold로 대체")
            print("  pip install iterative-stratification 으로 설치 권장")
            stratify = df[data_cfg.label_col].astype(str).str.split().str[0]
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                  random_state=cfg.seed)
            for fold_idx, (_, val_idx) in enumerate(kf.split(df, y=stratify)):
                df.loc[val_idx, "fold"] = fold_idx

    # 전략 5: 기본 → StratifiedKFold
    else:
        stratify = df[data_cfg.label_col].astype(str).str.split().str[0]
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=cfg.seed)
        for fold_idx, (_, val_idx) in enumerate(kf.split(df, y=stratify)):
            df.loc[val_idx, "fold"] = fold_idx
        print(f"StratifiedKFold (n_folds={n_folds})")

    # 검증
    assert (df["fold"] == -1).sum() == 0, "fold 배정 안 된 샘플이 있습니다"

    df.to_csv(folds_csv, index=False)
    print(f"\nfolds.csv 저장 → {folds_csv}")
    print("\n[fold 분포]")
    print(df["fold"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()