# Optimizer and LR scheduler builders.
from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def build_optimizer(model: nn.Module, cfg: DictConfig) -> Optimizer:
    """Create optimizer from config."""
    opt_type = cfg.optimizer.type.lower()
    if opt_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
        )
    elif opt_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)
    elif opt_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.optimizer.lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer type: '{opt_type}'")


def build_scheduler(
    optimizer: Optimizer,
    cfg: DictConfig,
    steps_per_epoch: int,
) -> LRScheduler | None:
    """Create LR scheduler from config. Returns None when scheduler='none'."""
    sched = cfg.optimizer.scheduler
    total_steps = cfg.train.epochs * steps_per_epoch

    if sched == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR

        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=cfg.optimizer.min_lr)
    elif sched == "onecycle":
        from torch.optim.lr_scheduler import OneCycleLR

        return OneCycleLR(optimizer, max_lr=cfg.optimizer.lr, total_steps=total_steps)
    elif sched == "plateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        return ReduceLROnPlateau(optimizer, mode=cfg.metric.mode, patience=2)
    elif sched == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: '{sched}'")
