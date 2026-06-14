# Waveform augmentations (CPU, inside Dataset) and spectrogram augmentations (GPU, inside Trainer).
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig


def apply_waveform_aug(wav: np.ndarray, cfg: DictConfig) -> np.ndarray:
    """Apply gain / additive noise based on augment config flags. Returns float32 array."""
    if cfg.augment.gain:
        gain = np.random.uniform(0.6, 1.4)
        wav = wav * gain
    if cfg.augment.noise:
        amp = np.random.uniform(0.0, 0.005)
        wav = wav + amp * np.random.randn(*wav.shape).astype(np.float32)
    # pitch_shift / time_shift: heavy ops, not yet implemented
    return wav.astype(np.float32)


class SpecAugment(nn.Module):
    """Frequency masking + time masking applied on GPU after frontend."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.freq_mask_param = cfg.augment.freq_mask
        self.time_mask_param = cfg.augment.time_mask

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        # spec: (B, C, H, W)
        raise NotImplementedError("SpecAugment.forward not yet implemented")


def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    mode: str = "max",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch-level mixup. mode='max' → element-wise max labels (multi-label safe)."""
    raise NotImplementedError("mixup_batch not yet implemented")
