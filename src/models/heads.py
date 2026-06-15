# Classification heads: linear, attention pooling, SED.
from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


class LinearHead(nn.Module):
    """Dropout + linear classifier."""

    def __init__(self, feat_dim: int, num_classes: int, drop_rate: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=drop_rate)
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, feat_dim)
        return self.fc(self.dropout(x))  # (B, num_classes)


class MLPHead(nn.Module):
    """2-layer MLP head: BN + ReLU + dropout. Non-linear fusion of features."""

    def __init__(
        self, feat_dim: int, num_classes: int, hidden_dim: int = 512, drop_rate: float = 0.0
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, feat_dim)
        return self.net(x)  # (B, num_classes)


class AttentionHead(nn.Module):
    """Attention pooling head (PANNs-style)."""

    def __init__(self, feat_dim: int, num_classes: int) -> None:
        super().__init__()
        raise NotImplementedError("AttentionHead not yet implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SEDHead(nn.Module):
    """Sound Event Detection: clip-level + frame-level dual output."""

    def __init__(self, feat_dim: int, num_classes: int) -> None:
        super().__init__()
        raise NotImplementedError("SEDHead not yet implemented")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


def build_head(cfg: DictConfig, feat_dim: int) -> nn.Module:
    """Instantiate head from config."""
    head_type = cfg.model.head
    num_classes = cfg.model.num_classes
    if head_type == "linear":
        return LinearHead(feat_dim, num_classes, drop_rate=cfg.model.drop_rate)
    elif head_type == "mlp":
        return MLPHead(
            feat_dim, num_classes,
            hidden_dim=int(cfg.model.get("mlp_hidden", 512)),
            drop_rate=cfg.model.drop_rate,
        )
    elif head_type == "attention":
        return AttentionHead(feat_dim, num_classes)
    elif head_type == "sed":
        return SEDHead(feat_dim, num_classes)
    else:
        raise ValueError(f"Unknown head type: '{head_type}'")
