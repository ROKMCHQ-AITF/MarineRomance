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
    # preload_to_ram이면 데이터가 이미 메인 프로세스 RAM에 있으므로 worker 불필요
    num_workers = 0 if cfg.data.get("preload_to_ram", False) else cfg.train.num_workers
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=is_train,
        drop_last=is_train,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )
