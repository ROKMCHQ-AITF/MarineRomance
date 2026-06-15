"""캐시된 임베딩으로 head만 K-Fold 학습한다.

extract_embeddings.py가 만든 embeddings.npz를 읽어, 인코더는 건너뛰고
head(linear 등)만 학습한다. 학습 루프는 기존 Trainer를 그대로 재사용하므로
AMP·scheduler·metric·checkpoint·wandb 로깅이 자동으로 따라온다.

입력 x는 waveform이 아니라 (B, D) 임베딩이고, model은 head 단독이다.
Trainer 입장에선 model(x) → logits 라는 계약만 지켜지면 되므로 그대로 동작한다.
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
from torch.utils.data import DataLoader, TensorDataset

from src.models.heads import build_head
from src.training.trainer import Trainer
from src.utils.config import load_config, save_config
from src.utils.logger import Logger
from src.utils.seed import seed_everything


def sog_log_stats(df: pd.DataFrame) -> tuple[float, float]:
    """log1p(clip(sog))의 (mean, std). 추론 시 train 통계를 고정해 재사용하기 위함."""
    sog = pd.to_numeric(df["sog"], errors="coerce").fillna(0.0).clip(0.0, 50.0).to_numpy()
    s = np.log1p(sog)
    return float(s.mean()), float(s.std())


def encode_ais(df: pd.DataFrame, sog_mean: float, sog_std: float) -> np.ndarray:
    """AIS 데이터프레임(정렬된 행) → (N, 10) float32. sog 표준화는 주어진 통계 사용.

    [0]    sog        → log1p + clip + (train 통계로) 표준화
    [1:4]  speed bucket → 정지(<0.5) / 저속(0.5~5) / 순항(≥5) 원핫
    [4:6]  cog        → sin/cos (순환값); 무효(0~360 밖)는 0 벡터
    [6:8]  heading    → sin/cos
    [8]    drift      → cos(heading − cog): 선수/진행 정렬도(정박·표류 식별). 둘 다 유효할 때만
    [9]    heading 유효 플래그
    """
    sog = pd.to_numeric(df["sog"], errors="coerce").fillna(0.0).clip(0.0, 50.0).to_numpy()
    feats = [((np.log1p(sog) - sog_mean) / (sog_std + 1e-6))[:, None]]

    feats.append((sog < 0.5).astype(np.float32)[:, None])
    feats.append(((sog >= 0.5) & (sog < 5.0)).astype(np.float32)[:, None])
    feats.append((sog >= 5.0).astype(np.float32)[:, None])

    cog = pd.to_numeric(df["cog"], errors="coerce").to_numpy()
    head = pd.to_numeric(df["true_heading"], errors="coerce").to_numpy()
    cog_valid = (cog >= 0.0) & (cog <= 360.0)
    head_valid = (head >= 0.0) & (head <= 360.0)

    for ang, valid in ((cog, cog_valid), (head, head_valid)):
        rad = np.deg2rad(np.where(valid, ang, 0.0))
        feats.append(np.where(valid, np.sin(rad), 0.0)[:, None])
        feats.append(np.where(valid, np.cos(rad), 0.0)[:, None])

    drift = np.cos(np.deg2rad(head - cog))
    feats.append(np.where(cog_valid & head_valid, drift, 0.0)[:, None])
    feats.append(head_valid.astype(np.float32)[:, None])

    return np.concatenate(feats, axis=1).astype(np.float32)  # (N, 10)


def load_ais_features(cfg: DictConfig, ids: np.ndarray) -> np.ndarray:
    """train 경로: folds_csv를 ids 순서로 정렬해 AIS 인코딩 (sog 통계는 train 자체에서)."""
    df = pd.read_csv(cfg.data.folds_csv).set_index(cfg.data.id_col).loc[list(ids)]
    mean, std = sog_log_stats(df)
    return encode_ais(df, mean, std)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="캐시 임베딩으로 head만 학습")
    parser.add_argument("--config", required=True, help="실험 yaml 경로")
    parser.add_argument("--emb", default=None, help="embeddings.npz 경로 (기본: outputs/<exp>/embeddings.npz)")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def _build_loader(emb: torch.Tensor, y: torch.Tensor, cfg: DictConfig, train: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(emb, y),
        batch_size=cfg.train.batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=0,  # 임베딩은 이미 RAM에 있음 → worker 불필요
        pin_memory=True,
    )


def main() -> None:
    args = parse_args()
    cfg: DictConfig = load_config(args.config, args.overrides)
    seed_everything(cfg.seed)

    out_dir = Path("outputs") / cfg.exp_name
    save_config(cfg, out_dir)

    emb_path = Path(args.emb) if args.emb else out_dir / "embeddings.npz"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"임베딩 캐시 없음: {emb_path}\n먼저 extract_embeddings.py 를 실행하세요."
        )
    data = np.load(emb_path, allow_pickle=True)
    emb = torch.from_numpy(data["emb"]).float()  # (N, D)
    fold = data["fold"]
    feat_dim = int(data["feat_dim"])
    if cfg.data.multilabel:
        y = torch.from_numpy(data["y"]).float()  # (N, C) multi-hot
    else:
        y = torch.from_numpy(data["y"]).long()   # (N,) class index
    print(f"임베딩 로드: {tuple(emb.shape)}  feat_dim={feat_dim}")

    # AIS 메타데이터(sog/cog/heading)를 임베딩에 concat → head 입력 차원 확장
    if cfg.data.get("use_ais", False):
        ais = load_ais_features(cfg, data["ids"])
        emb = torch.cat([emb, torch.from_numpy(ais)], dim=1)
        feat_dim = emb.shape[1]
        print(f"AIS concat: +{ais.shape[1]} feats → feat_dim={feat_dim}")

    logger = Logger(cfg)

    scores: list[float] = []
    for f in cfg.train.folds:
        tr = fold != f
        va = fold == f
        loaders = {
            "train": _build_loader(emb[tr], y[tr], cfg, train=True),
            "valid": _build_loader(emb[va], y[va], cfg, train=False),
        }
        head = build_head(cfg, feat_dim)  # model = head 단독, forward: (B,D) → (B,C)
        trainer = Trainer(head, loaders, cfg, logger, f)
        score = trainer.fit()
        scores.append(score)
        print(f"Fold {f} best {cfg.metric.monitor}: {score:.4f}")

    cv_mean = float(np.mean(scores))
    print(f"CV mean: {cv_mean:.4f}")
    logger.log({"cv_mean": cv_mean})
    logger.finish()


if __name__ == "__main__":
    main()
