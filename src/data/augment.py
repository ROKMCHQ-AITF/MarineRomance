# Waveform augmentations (CPU, inside Dataset) and spectrogram augmentations (GPU, inside Trainer).
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T
from omegaconf import DictConfig


def apply_waveform_aug(wav: np.ndarray, cfg: DictConfig) -> np.ndarray:
    """Apply gain / additive noise based on augment config flags. Returns float32 array."""
    if cfg.augment.gain:
        lo, hi = cfg.augment.gain_range
        wav = wav * np.random.uniform(lo, hi)
    if cfg.augment.noise:
        amp = np.random.uniform(0.0, cfg.augment.noise_amp)
        wav = wav + amp * np.random.randn(*wav.shape).astype(np.float32)
    return wav.astype(np.float32)


class SpecAugment(nn.Module):
    """Frequency masking + time masking applied on GPU after frontend."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.freq_mask = T.FrequencyMasking(freq_mask_param=cfg.augment.freq_mask)
        self.time_mask = T.TimeMasking(time_mask_param=cfg.augment.time_mask)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        # spec: (B, C, H, W)
        spec = self.freq_mask(spec)
        spec = self.time_mask(spec)
        return spec


def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    mode: str = "max",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch-level mixup. mode='max' → element-wise max labels (multi-label safe)."""
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    if mode == "max":
        y_mix = torch.max(y, y[idx])
    else:
        y_mix = lam * y + (1 - lam) * y[idx]
    return x_mix, y_mix
