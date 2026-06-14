# Loss builders: BCE / Focal / CE / LSEP.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class FocalLoss(nn.Module):
    """Sigmoid focal loss for multi-label classification."""

    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        return (((1 - pt) ** self.gamma) * bce).mean()


def build_loss(cfg: DictConfig) -> nn.Module:
    """Return loss module from config."""
    loss_type = cfg.loss.type
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss_type == "focal":
        return FocalLoss(gamma=cfg.loss.focal_gamma)
    elif loss_type == "ce":
        return nn.CrossEntropyLoss(label_smoothing=cfg.loss.label_smoothing)
    elif loss_type == "lsep":
        raise NotImplementedError("LSEP loss not yet implemented")
    else:
        raise ValueError(f"Unknown loss type: '{loss_type}'")
