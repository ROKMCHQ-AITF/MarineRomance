# Frontend: waveform (B,1,T) → spectrogram image (B,C,H,W) computed on GPU.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class Frontend(nn.Module):
    """Convert waveform batch to spectrogram image on GPU.

    Supports melspec now; mfcc/cqt/raw raise NotImplementedError until added.
    New feature types: add a branch here + a matching entry in feature_extractor.py.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        feat = cfg.feature
        self.feature_type = feat.type
        self.channel_mode = feat.channel_mode
        self.n_channels = feat.n_channels
        self.image_size = list(feat.image_size) if feat.image_size else None

        if feat.type == "melspec":
            import torchaudio

            self.transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=cfg.data.sample_rate,
                n_fft=feat.n_fft,
                hop_length=feat.hop_length,
                win_length=feat.win_length,
                n_mels=feat.n_mels,
                f_min=feat.fmin,
                f_max=feat.fmax,
                power=feat.power,
            )
            self.amp_to_db = torchaudio.transforms.AmplitudeToDB() if feat.to_db else None
        else:
            raise NotImplementedError(f"frontend type '{feat.type}' not yet implemented")

        self.do_normalize = feat.normalize

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (B, 1, T)
        x = wav.squeeze(1)              # (B, T)
        spec = self.transform(x)        # (B, n_mels, T')
        if self.amp_to_db is not None:
            spec = self.amp_to_db(spec)
        if self.do_normalize:
            mean = spec.mean(dim=(-2, -1), keepdim=True)
            std = spec.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
            spec = (spec - mean) / std
        spec = self._make_channels(spec)  # (B, C, H, W)
        if self.image_size is not None:
            spec = F.interpolate(spec, size=self.image_size, mode="bilinear", align_corners=False)
        return spec  # (B, C, H, W)

    def _make_channels(self, spec: torch.Tensor) -> torch.Tensor:
        # spec: (B, H, W)
        spec = spec.unsqueeze(1)  # (B, 1, H, W)
        if self.channel_mode == "repeat":
            return spec.expand(-1, self.n_channels, -1, -1).contiguous()
        elif self.channel_mode == "delta":
            raise NotImplementedError("channel_mode='delta' not yet implemented")
        elif self.channel_mode == "multi_res":
            raise NotImplementedError("channel_mode='multi_res' not yet implemented")
        else:
            raise ValueError(f"Unknown channel_mode: {self.channel_mode}")
