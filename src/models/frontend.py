# Frontend: waveform (B,1,T) → spectrogram image (B,C,H,W) computed on GPU.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF
import torchaudio.transforms as T
from omegaconf import DictConfig


class Frontend(nn.Module):
    """Convert waveform batch to spectrogram image on GPU.

    Supports feature types: melspec | mfcc | cqt | raw
    Supports channel modes: repeat | delta | multi_res
    New feature type: add a branch in __init__ + a matching entry in feature_extractor.py.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        feat = cfg.feature
        self.feature_type = feat.type
        self.channel_mode = feat.channel_mode
        self.n_channels = feat.n_channels
        self.image_size = list(feat.image_size) if feat.image_size else None
        self.do_normalize = feat.normalize
        sr = cfg.data.sample_rate

        if feat.type == "melspec":
            self.transform = T.MelSpectrogram(
                sample_rate=sr,
                n_fft=feat.n_fft,
                hop_length=feat.hop_length,
                win_length=feat.win_length,
                n_mels=feat.n_mels,
                f_min=feat.fmin,
                f_max=feat.fmax,
                power=feat.power,
            )
            self.amp_to_db = T.AmplitudeToDB() if feat.to_db else None

        elif feat.type == "mfcc":
            self.transform = T.MFCC(
                sample_rate=sr,
                n_mfcc=feat.get("n_mfcc", 40),
                melkwargs={
                    "n_fft": feat.n_fft,
                    "hop_length": feat.hop_length,
                    "win_length": feat.win_length,
                    "n_mels": feat.n_mels,
                    "f_min": feat.fmin,
                    "f_max": feat.fmax,
                },
            )
            self.amp_to_db = None  # MFCC is already log-scaled internally

        elif feat.type == "cqt":
            try:
                from nnAudio import Spectrogram as nnSpec
                self.transform = nnSpec.CQT(
                    sr=sr,
                    hop_length=feat.hop_length,
                    fmin=feat.fmin,
                    n_bins=feat.get("n_bins", feat.n_mels),
                    bins_per_octave=feat.get("bins_per_octave", 12),
                    output_format="Magnitude",
                )
                self.amp_to_db = T.AmplitudeToDB() if feat.to_db else None
            except ImportError:
                raise ImportError(
                    "nnAudio is required for CQT. Install with: pip install nnAudio"
                )

        elif feat.type == "raw":
            # No spectral transform; waveform is reshaped into 2D grid.
            # Best used with model.type=wav2vec2 (which handles raw waveform natively).
            self.transform = None
            self.amp_to_db = None

        else:
            raise NotImplementedError(
                f"feature type '{feat.type}' not implemented. "
                "Add a branch here + a matching entry in data/feature_extractor.py."
            )

        # multi_res: 3 mel transforms with different hop lengths (melspec only)
        if feat.channel_mode == "multi_res":
            if feat.type != "melspec":
                raise ValueError(
                    "channel_mode='multi_res' is only supported with feature.type='melspec'"
                )
            hops = [feat.hop_length // 2, feat.hop_length, feat.hop_length * 2]
            self.multi_transforms = nn.ModuleList([
                T.MelSpectrogram(
                    sample_rate=sr,
                    n_fft=feat.n_fft,
                    hop_length=h,
                    win_length=feat.win_length,
                    n_mels=feat.n_mels,
                    f_min=feat.fmin,
                    f_max=feat.fmax,
                    power=feat.power,
                )
                for h in hops
            ])
            self.multi_amp_to_db = T.AmplitudeToDB() if feat.to_db else None

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (B, 1, T)
        x = wav.squeeze(1)  # (B, T)

        if self.feature_type == "raw":
            return self._forward_raw(x)

        if self.channel_mode == "multi_res":
            return self._forward_multi_res(x)

        spec = self.transform(x)  # (B, F, T')
        if self.amp_to_db is not None:
            spec = self.amp_to_db(spec)
        if self.do_normalize:
            mean = spec.mean(dim=(-2, -1), keepdim=True)
            std = spec.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
            spec = (spec - mean) / std

        img = self._make_channels(spec)  # (B, C, H, W)
        if self.image_size is not None:
            img = F.interpolate(img, size=self.image_size, mode="bilinear", align_corners=False)
        return img  # (B, C, H, W)

    def _forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape raw waveform (B, T) into (B, C, H, W) for CNN input."""
        B, T = x.shape
        H, W = self.image_size if self.image_size else (int(T ** 0.5), int(T ** 0.5))
        target = H * W
        if T < target:
            x = F.pad(x, (0, target - T))
        img = x[:, :target].view(B, 1, H, W)  # (B, 1, H, W)
        return img.expand(-1, self.n_channels, -1, -1).contiguous()  # (B, C, H, W)

    def _forward_multi_res(self, x: torch.Tensor) -> torch.Tensor:
        """Stack 3 mel spectrograms at different hop lengths as separate channels."""
        specs = []
        for tr in self.multi_transforms:
            s = tr(x)  # (B, n_mels, T'_i)
            if self.multi_amp_to_db is not None:
                s = self.multi_amp_to_db(s)
            if self.do_normalize:
                mean = s.mean(dim=(-2, -1), keepdim=True)
                std = s.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
                s = (s - mean) / std
            specs.append(s)

        # Resize all to same spatial size then stack as (B, 3, H, W)
        target = self.image_size if self.image_size else [specs[0].shape[-2], specs[0].shape[-1]]
        resized = [
            F.interpolate(s.unsqueeze(1), size=target, mode="bilinear", align_corners=False).squeeze(1)
            for s in specs
        ]
        return torch.stack(resized, dim=1)  # (B, 3, H, W)

    def _make_channels(self, spec: torch.Tensor) -> torch.Tensor:
        # spec: (B, F, T')
        if self.channel_mode == "repeat":
            return spec.unsqueeze(1).expand(-1, self.n_channels, -1, -1).contiguous()
        elif self.channel_mode == "delta":
            # [mel, delta, delta-delta] — standard acoustic feature stack
            delta = AF.compute_deltas(spec)    # (B, F, T')
            delta2 = AF.compute_deltas(delta)
            return torch.stack([spec, delta, delta2], dim=1)  # (B, 3, F, T')
        else:
            raise ValueError(f"Unknown channel_mode: '{self.channel_mode}'")
