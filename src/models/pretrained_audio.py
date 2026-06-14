# Audio-pretrained model loader: PANNs / wav2vec2 / BirdNET. Stub for phase-2.
from __future__ import annotations

import torch.nn as nn
from omegaconf import DictConfig


def build_audio_pretrained(cfg: DictConfig) -> tuple[nn.Module, int]:
    """Load audio-pretrained backbone. Returns (module, feat_dim).

    Implement PANNs/wav2vec2 here when needed. Interface must match build_backbone.
    """
    raise NotImplementedError(
        f"Audio pretrained model type='{cfg.model.type}' is not yet implemented. "
        "Add PANNs / wav2vec2 support in src/models/pretrained_audio.py."
    )
