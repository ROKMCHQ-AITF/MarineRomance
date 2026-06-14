# Factory: assembles frontend + backbone + head into AudioModel from config.
from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.backbones import build_backbone
from src.models.frontend import Frontend
from src.models.heads import build_head
from src.models.pretrained_audio import build_audio_pretrained


class AudioModel(nn.Module):
    """End-to-end model: (B,1,T) waveform → (B, num_classes) logits.

    When cfg.feature.compute_on=='cpu', x is expected to be a pre-computed
    (B, C, H, W) feature tensor and the frontend is skipped.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        use_frontend = cfg.feature.compute_on == "gpu"
        self.frontend: nn.Module | None = Frontend(cfg) if use_frontend else None

        if cfg.model.type == "timm":
            self.backbone, feat_dim = build_backbone(cfg)
        else:
            self.backbone, feat_dim = build_audio_pretrained(cfg)

        self.head = build_head(cfg, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T) waveform  or  (B, C, H, W) cached feature
        if self.frontend is not None:
            x = self.frontend(x)       # (B, C, H, W)
        features = self.backbone(x)    # (B, feat_dim)
        return self.head(features)     # (B, num_classes)


def build_model(cfg: DictConfig) -> AudioModel:
    """Single entry point to construct a model from config."""
    return AudioModel(cfg)
