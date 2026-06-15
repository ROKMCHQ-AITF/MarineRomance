# Factory: assembles frontend + backbone + head into AudioModel from config.
from __future__ import annotations

import torch.nn as nn
from omegaconf import DictConfig

from src.models.backbones import build_backbone
from src.models.frontend import Frontend
from src.models.heads import build_head


class AudioModel(nn.Module):
    """End-to-end model: (B,1,T) waveform → (B, num_classes) logits.

    model.type routing:
      timm   → timm backbone via backbones.py (uses Frontend for spectrogram)
      ast    → ASTModel via pretrained_audio.py (uses Frontend for spectrogram)
      beats  → Microsoft BEATs via pretrained_audio.py (skips Frontend, raw waveform in)
      htsat  → HTS-AT (CLAP audio tower) via pretrained_audio.py (skips Frontend, raw waveform in)

    When cfg.feature.compute_on=='cpu', x is expected to be a pre-computed
    (B, C, H, W) feature tensor and the frontend is skipped.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        model_type = cfg.model.get("type", "timm")

        if model_type == "timm":
            self.backbone, feat_dim = build_backbone(cfg)
        elif model_type in ("ast", "beats", "htsat"):
            from src.models.pretrained_audio import build_audio_pretrained
            self.backbone, feat_dim = build_audio_pretrained(cfg)
        else:
            raise ValueError(
                f"Unknown model.type='{model_type}'. Choose timm / ast / beats / htsat."
            )

        # beats/htsat consume raw waveform directly — no spectrogram frontend needed.
        skip_frontend = model_type in ("beats", "htsat")
        use_frontend = (not skip_frontend) and (cfg.feature.compute_on == "gpu")
        self.frontend: nn.Module | None = Frontend(cfg) if use_frontend else None

        self.head = build_head(cfg, feat_dim)

    def forward(self, x):
        # x: (B, 1, T) waveform  or  (B, C, H, W) pre-computed feature
        if self.frontend is not None:
            x = self.frontend(x)     # (B, C, H, W)
        features = self.backbone(x)  # (B, feat_dim)
        return self.head(features)   # (B, num_classes)


def build_model(cfg: DictConfig) -> AudioModel:
    """Single entry point to construct a model from config."""
    return AudioModel(cfg)
