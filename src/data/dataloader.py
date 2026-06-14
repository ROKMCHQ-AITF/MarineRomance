# DataLoader builder: wraps AudioDataset with correct shuffle/drop_last per mode.
from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data.dataset import AudioDataset


def build_dataloader(df: pd.DataFrame, cfg: DictConfig, mode: str) -> DataLoader:
    """Build DataLoader for train / valid / test.

    shuffle and drop_last are True only for train split.
    """
    dataset = AudioDataset(df, cfg, mode=mode)
    is_train = mode == "train"
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=is_train,
        drop_last=is_train,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        persistent_workers=cfg.train.num_workers > 0,
    )
