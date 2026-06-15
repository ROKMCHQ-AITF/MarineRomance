"""공식 라벨 검증셋(val.csv) 평가: 학습된 체크포인트(fold 앙상블)의 LB 프록시 점수 산출.

inference.py가 라벨 없는 test.csv로 제출파일을 만든다면, evaluate.py는 라벨 있는
val.csv로 공식 지표(macro_F1 등) + 클래스별 리포트를 출력한다. 학습엔 절대 쓰지 않는다.

사용:
  python evaluate.py --config configs/default.yaml --ckpt outputs/default
  python evaluate.py --config configs/default.yaml --ckpt outputs/default/best_fold0.pth
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from inference import _find_checkpoints, predict
from src.data.dataset import AudioDataset
from src.models.factory import build_model
from src.utils.config import load_config
from src.utils.metrics import get_metric_fn
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="공식 val.csv 평가 (LB 프록시)")
    parser.add_argument("--config", required=True, help="실험 yaml 경로")
    parser.add_argument("--ckpt", required=True, help="체크포인트 디렉토리 또는 단일 .pth")
    parser.add_argument("--val_csv", default=None, help="검증 CSV (기본: data.val_csv)")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def _encode_labels(df: pd.DataFrame, cfg, label_map: dict[str, int], num_classes: int) -> np.ndarray:
    """val_df 라벨을 dataset._encode_label과 동일 규칙으로 인코딩.

    single-label → (N,) class indices, multi-label → (N, C) multi-hot.
    """
    col = cfg.data.label_col
    if cfg.data.multilabel:
        y = np.zeros((len(df), num_classes), dtype=np.float32)
        for i, raw in enumerate(df[col].astype(str)):
            for name in raw.split():
                idx = label_map.get(name)
                if idx is not None:
                    y[i, idx] = 1.0
        return y
    return df[col].astype(str).map(lambda r: label_map.get(r, 0)).to_numpy(dtype=np.int64)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.seed, benchmark=cfg.train.get("cudnn_benchmark", True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.train.amp and device.type == "cuda"

    val_csv = Path(args.val_csv) if args.val_csv else Path(cfg.data.val_csv)
    if not val_csv.exists():
        raise FileNotFoundError(f"검증 CSV 없음: {val_csv}")
    val_df = pd.read_csv(val_csv)
    if cfg.data.label_col not in val_df.columns:
        raise ValueError(f"val.csv에 라벨 컬럼 '{cfg.data.label_col}' 없음 → 평가 불가")
    print(f"검증 샘플: {len(val_df)}개  ({val_csv})", flush=True)

    # 라벨맵 로드 (학습 때와 동일한 클래스 인덱스)
    label_map_path = Path(cfg.data.folds_csv).parent / "label_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError(f"label_map.json 없음: {label_map_path} (prepare_folds.py 먼저 실행)")
    with open(label_map_path) as f:
        label_map: dict[str, int] = json.load(f)
    num_classes = len(label_map)

    # DataLoader: shuffle=False → 예측 순서가 val_df 순서와 일치
    loader = DataLoader(
        AudioDataset(val_df, cfg, mode="valid"),
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    ckpt_paths = _find_checkpoints(args.ckpt)
    print(f"체크포인트 {len(ckpt_paths)}개 앙상블: {[p.name for p in ckpt_paths]}", flush=True)

    multilabel = cfg.data.multilabel
    ensemble: list[np.ndarray] = []
    for ckpt_path in ckpt_paths:
        model = build_model(cfg).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        ensemble.append(predict(model, loader, device, use_amp, multilabel=multilabel, cfg=cfg))
    preds = np.mean(ensemble, axis=0)  # (N, C)

    y_true = _encode_labels(val_df, cfg, label_map, num_classes)

    # ── 점수 출력 ──────────────────────────────────────────────────────────────
    metric_fn = get_metric_fn(cfg)
    score = metric_fn(y_true, preds)
    print("\n" + "=" * 56)
    print(f"공식 지표 [{cfg.metric.name}]: {score:.4f}")

    if not multilabel:
        from sklearn.metrics import accuracy_score, classification_report
        pred_cls = preds.argmax(axis=1)
        idx_to_label = {v: k for k, v in label_map.items()}
        target_names = [idx_to_label[i] for i in range(num_classes)]
        print(f"accuracy: {accuracy_score(y_true, pred_cls):.4f}")
        print("\n클래스별 리포트:")
        print(classification_report(y_true, pred_cls, target_names=target_names,
                                    digits=4, zero_division=0))
    print("=" * 56)
    print("[참고] 이 val 점수와 학습 CV 점수의 상관을 보고 CV 신뢰도를 판단하라.", flush=True)


if __name__ == "__main__":
    main()
