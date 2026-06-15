"""Frozen 인코더 임베딩을 1회 추출해 .npz로 캐싱한다.

frozen encoder + augment off + 고정 길이 클립 → 클립별 임베딩이 epoch마다 동일하다.
따라서 한 번만 계산해 두면 head 학습(train_head.py)에서 무한 재사용할 수 있어
인코더 forward 반복(수십 분/epoch)을 통째로 없앤다.

저장 형식: outputs/<exp>/embeddings.npz
  emb      (N, D) float32  — 인코더 임베딩
  y        (N,) or (N, C)  — 인코딩된 라벨 (single=인덱스, multi=multi-hot)
  fold     (N,) int        — folds.csv의 fold 컬럼
  ids      (N,) str        — id_col (filename)
  feat_dim int             — D
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src.data.dataloader import build_dataloader
from src.models.factory import AudioModel, build_model
from src.utils.config import load_config
from src.utils.seed import seed_everything


def embed(model: AudioModel, x: torch.Tensor) -> torch.Tensor:
    """AudioModel에서 head 직전 임베딩 (B, D)을 뽑는다 (frontend가 있으면 통과)."""
    if model.frontend is not None:
        x = model.frontend(x)
    return model.backbone(x)  # (B, D)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="frozen 인코더 임베딩 캐싱")
    parser.add_argument("--config", required=True, help="실험 yaml 경로")
    parser.add_argument("--out", default=None, help="저장 경로 (기본: outputs/<exp>/embeddings.npz)")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg: DictConfig = load_config(args.config, args.overrides)
    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.train.amp and device.type == "cuda"

    df = pd.read_csv(cfg.data.folds_csv)
    print(f"대상: {len(df)} clips from {cfg.data.folds_csv}")

    # mode='valid' → center crop, augment 없음, shuffle 없음 → 결정적·순서 보존
    loader = build_dataloader(df, cfg, "valid")

    model = build_model(cfg).to(device).eval()

    embs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for x, y in tqdm(loader, desc="extract"):
        x = x.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            e = embed(model, x)  # (B, D)
        embs.append(e.cpu().float().numpy())
        ys.append(y.cpu().numpy())

    emb = np.concatenate(embs, axis=0)  # (N, D)
    y = np.concatenate(ys, axis=0)      # (N,) or (N, C)
    feat_dim = emb.shape[1]

    # loader는 shuffle=False라 df 순서와 일치 → fold·id를 df에서 그대로 정렬해 가져온다
    assert len(emb) == len(df), f"임베딩 수({len(emb)}) != df 행 수({len(df)})"
    fold = df["fold"].to_numpy()
    ids = df[cfg.data.id_col].astype(str).to_numpy()

    out_path = Path(args.out) if args.out else Path("outputs") / cfg.exp_name / "embeddings.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, emb=emb, y=y, fold=fold, ids=ids, feat_dim=feat_dim)

    print(f"\n저장 완료 → {out_path}")
    print(f"  emb {emb.shape} {emb.dtype}  |  y {y.shape} {y.dtype}  |  feat_dim={feat_dim}")
    print(f"  용량 ≈ {emb.nbytes / 1e6:.1f} MB")
    print("[fold 분포]")
    print(pd.Series(fold).value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
