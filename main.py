"""K-Fold training orchestration. Fold loop here; training logic lives in Trainer."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.data.dataloader import build_dataloader
from src.models.factory import build_model
from src.training.trainer import Trainer
from src.utils.config import load_config, save_config
from src.utils.logger import Logger
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaggle audio classification — K-Fold training")
    parser.add_argument("--config", required=True, help="path to experiment yaml")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dot-list overrides, e.g. train.epochs=2")
    return parser.parse_args()


def _make_dummy_df(cfg: DictConfig) -> pd.DataFrame:
    """Synthetic fold DataFrame for debug/smoke-test — no real audio needed."""
    n = cfg.train.n_folds * 4  # 4 samples per fold → 20 total with n_folds=5
    return pd.DataFrame(
        {
            cfg.data.id_col: [f"dummy_{i:04d}.ogg" for i in range(n)],
            cfg.data.label_col: [f"cls_{i % cfg.model.num_classes}" for i in range(n)],
            "fold": [i % cfg.train.n_folds for i in range(n)],
        }
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.seed)

    out_dir = Path("outputs") / cfg.exp_name
    save_config(cfg, out_dir)

    logger = Logger(cfg)

    if cfg.get("debug", False):
        print("[debug] Using synthetic dummy data (no real audio required).")
        df = _make_dummy_df(cfg)
    else:
        df = pd.read_csv(cfg.data.folds_csv)

    scores: list[float] = []
    for fold in cfg.train.folds:
        train_df = df[df.fold != fold]
        valid_df = df[df.fold == fold]
        loaders = {
            "train": build_dataloader(train_df, cfg, "train"),
            "valid": build_dataloader(valid_df, cfg, "valid"),
        }
        model = build_model(cfg)
        trainer = Trainer(model, loaders, cfg, logger, fold)
        score = trainer.fit()
        scores.append(score)
        print(f"Fold {fold} best {cfg.metric.monitor}: {score:.4f}")

    cv_mean = float(np.mean(scores))
    print(f"CV mean: {cv_mean:.4f}")
    logger.log({"cv_mean": cv_mean})
    logger.finish()


if __name__ == "__main__":
    main()
