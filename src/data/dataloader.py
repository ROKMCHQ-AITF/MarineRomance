# DataLoader builder: wraps AudioDataset with correct shuffle/drop_last per mode.
from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import torch
from src.data.dataset import AudioDataset


def build_dataloader(df: pd.DataFrame, cfg: DictConfig, mode: str) -> DataLoader:
    """Build DataLoader for train / valid / test.

    shuffle and drop_last are True only for train split.
    """
    dataset = AudioDataset(df, cfg, mode=mode)
    is_train = mode == "train"
    num_workers = cfg.train.num_workers
    kwargs = {}
    if num_workers > 0:
        # 워커가 GPU 학습 중 다음 배치를 미리 적재 → I/O 레이턴시 은닉
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = cfg.train.get("prefetch_factor", 4)
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=is_train,
        drop_last=is_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **kwargs,
    )
