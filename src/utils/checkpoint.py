# Checkpoint save/load and best-score tracking.
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    score: float,
    path: Path,
) -> None:
    """Save model and optimizer state dicts to path."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "score": score,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """Load checkpoint into model (and optionally optimizer). Returns state dict."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state


class BestTracker:
    """Track best validation score and persist checkpoint on improvement."""

    def __init__(self, cfg: DictConfig) -> None:
        self.mode = cfg.metric.mode
        self.best_score: float = float("-inf") if self.mode == "max" else float("inf")

    def _is_better(self, score: float) -> bool:
        return score > self.best_score if self.mode == "max" else score < self.best_score

    def update(
        self,
        score: float,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        path: Path,
    ) -> bool:
        """Save checkpoint if score improved. Returns True if a new best was set."""
        if self._is_better(score):
            self.best_score = score
            save_checkpoint(model, optimizer, epoch, score, path)
            return True
        return False
