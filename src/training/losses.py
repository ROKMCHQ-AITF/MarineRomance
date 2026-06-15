# Loss builders: BCE / Focal / CE / LSEP.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


def _smooth_targets(targets: torch.Tensor, smoothing: float) -> torch.Tensor:
    """Apply label smoothing for multi-label: 1 → 1-s/2, 0 → s/2."""
    return targets * (1.0 - smoothing) + 0.5 * smoothing


class BCEWithLabelSmoothing(nn.Module):
    """BCE with optional label smoothing for multi-label classification."""

    def __init__(self, smoothing: float = 0.0) -> None:
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.smoothing > 0.0:
            targets = _smooth_targets(targets, self.smoothing)
        return F.binary_cross_entropy_with_logits(logits, targets)


class FocalLoss(nn.Module):
    """Sigmoid focal loss for multi-label classification, with optional label smoothing."""

    def __init__(self, gamma: float = 2.0, smoothing: float = 0.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.smoothing > 0.0:
            targets = _smooth_targets(targets, self.smoothing)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        return (((1 - pt) ** self.gamma) * bce).mean()


class ComboLoss(nn.Module):
    """Weighted sum of multiple losses defined under loss.components."""

    def __init__(self, losses: list[nn.Module], weights: list[float]) -> None:
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return sum(w * l(logits, targets) for w, l in zip(self.weights, self.losses))


def _build_single(loss_cfg: DictConfig, global_smoothing: float) -> nn.Module:
    """Build one loss component from a component-level config dict."""
    loss_type = loss_cfg.type
    smoothing = float(loss_cfg.get("label_smoothing", global_smoothing))
    if loss_type == "bce":
        if smoothing > 0.0:
            return BCEWithLabelSmoothing(smoothing=smoothing)
        return nn.BCEWithLogitsLoss()
    elif loss_type == "focal":
        return FocalLoss(gamma=float(loss_cfg.get("focal_gamma", 2.0)), smoothing=smoothing)
    elif loss_type == "ce":
        return nn.CrossEntropyLoss(label_smoothing=smoothing)
    elif loss_type == "lsep":
        raise NotImplementedError("LSEP loss not yet implemented")
    else:
        raise ValueError(f"Unknown loss type: '{loss_type}'")


def build_loss(cfg: DictConfig) -> nn.Module:
    """Return loss module from config.

    loss.type=combo 일 때 loss.components 리스트에서 각 loss를 조립해 가중합한다.
    """
    loss_type = cfg.loss.type
    global_smoothing = float(cfg.loss.get("label_smoothing", 0.0))

    if loss_type == "combo":
        components = cfg.loss.components  # list of {type, weight, ...}
        losses, weights = [], []
        for comp in components:
            losses.append(_build_single(comp, global_smoothing))
            weights.append(float(comp.get("weight", 1.0)))
        total = sum(weights)
        weights = [w / total for w in weights]  # 합이 1이 되도록 정규화
        return ComboLoss(losses, weights)

    return _build_single(cfg.loss, global_smoothing)
