# Frontend: waveform (B,1,T) → spectrogram image (B,C,H,W) computed on GPU.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF
import torchaudio.transforms as T
from omegaconf import DictConfig

# Feature types:  melspec | log_mel | mfcc | cqt | lofar | demon | raw
# Channel modes:  repeat | delta | log_linear | harmonic_percussive | multi_res | multi_feat


class Frontend(nn.Module):
    """Convert waveform batch to spectrogram image on GPU.

    Routing logic:
      channel_mode=multi_res  → _init_multi_res  / _forward_multi_res
      channel_mode=multi_feat → _init_multi_feat / _forward_multi_feat
      everything else         → _init_single     / forward → _compute_feature → _make_channels
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
        self._logged_shape = False
        print(
            f"[Frontend] type={feat.type}  channel_mode={feat.channel_mode}"
            f"  image_size={list(feat.image_size)}  sr={sr}",
            flush=True,
        )

        if self.channel_mode == "multi_feat":
            self._init_multi_feat(feat, sr)
        elif self.channel_mode == "multi_res":
            self._init_multi_res(feat, sr)
        else:
            self._init_single(feat, sr)

    # ── init helpers ──────────────────────────────────────────────────────────

    def _build_mel(self, feat: DictConfig, sr: int, hop_length: int | None = None) -> T.MelSpectrogram:
        return T.MelSpectrogram(
            sample_rate=sr,
            n_fft=feat.n_fft,
            hop_length=hop_length or feat.hop_length,
            win_length=feat.win_length,
            n_mels=feat.n_mels,
            f_min=feat.fmin,
            f_max=feat.fmax,
            power=feat.power,
        )

    def _build_lofar(self, feat: DictConfig) -> T.Spectrogram:
        n_fft = int(feat.get("lofar_n_fft", 4096))
        return T.Spectrogram(n_fft=n_fft, hop_length=feat.hop_length, win_length=n_fft, power=2.0)

    def _build_demon_mel(self, feat: DictConfig) -> T.MelSpectrogram:
        env_sr = int(feat.get("demon_env_sr", 400))
        n_fft = int(feat.get("demon_n_fft", 256))
        return T.MelSpectrogram(
            sample_rate=env_sr,
            n_fft=n_fft,
            hop_length=int(feat.get("demon_hop_length", 8)),
            win_length=n_fft,
            n_mels=feat.n_mels,
            f_min=0.0,
            f_max=float(env_sr) / 2,
            power=float(feat.power),
        )

    def _store_demon_params(self, feat: DictConfig, sr: int) -> None:
        self.demon_sr = sr
        self.demon_fmin = float(feat.get("demon_fmin", 1000.0))
        self.demon_fmax = float(feat.get("demon_fmax", 10000.0))
        self.demon_env_cutoff = float(feat.get("demon_env_cutoff", 50.0))
        self.demon_env_sr = int(feat.get("demon_env_sr", 400))

    def _init_single(self, feat: DictConfig, sr: int) -> None:
        ftype = feat.type
        self.amp_to_db = T.AmplitudeToDB() if feat.to_db else None
        self.log_eps = float(feat.get("log_eps", 1e-6))

        if ftype in ("melspec", "log_mel"):
            self.transform = self._build_mel(feat, sr)

        elif ftype == "mfcc":
            self.transform = T.MFCC(
                sample_rate=sr,
                n_mfcc=feat.get("n_mfcc", 40),
                melkwargs={
                    "n_fft": feat.n_fft, "hop_length": feat.hop_length,
                    "win_length": feat.win_length, "n_mels": feat.n_mels,
                    "f_min": feat.fmin, "f_max": feat.fmax,
                },
            )
            self.amp_to_db = None  # MFCC is already log-scaled

        elif ftype == "cqt":
            try:
                from nnAudio import Spectrogram as nnSpec
                self.transform = nnSpec.CQT(
                    sr=sr, hop_length=feat.hop_length, fmin=feat.fmin,
                    n_bins=feat.get("n_bins", feat.n_mels),
                    bins_per_octave=feat.get("bins_per_octave", 12),
                    output_format="Magnitude",
                )
            except ImportError:
                raise ImportError("nnAudio is required for CQT: pip install nnAudio")

        elif ftype == "lofar":
            self.transform = self._build_lofar(feat)
            self.lofar_n_bins = int(feat.get("lofar_n_bins", feat.n_mels))

        elif ftype == "demon":
            self._store_demon_params(feat, sr)
            self.transform = self._build_demon_mel(feat)

        elif ftype == "raw":
            self.transform = None
            self.amp_to_db = None

        else:
            raise NotImplementedError(
                f"feature type '{ftype}' not implemented. "
                "Choices: melspec | log_mel | mfcc | cqt | lofar | demon | raw"
            )

        if self.channel_mode == "harmonic_percussive":
            self.hpss_kernel_harm = int(feat.get("hpss_kernel_harm", 31))
            self.hpss_kernel_perc = int(feat.get("hpss_kernel_perc", 31))

    def _init_multi_res(self, feat: DictConfig, sr: int) -> None:
        if feat.type != "melspec":
            raise ValueError("channel_mode='multi_res' requires feature.type='melspec'")
        hops = [feat.hop_length // 2, feat.hop_length, feat.hop_length * 2]
        self.res_transforms = nn.ModuleList([self._build_mel(feat, sr, h) for h in hops])
        self.res_amp_to_db = T.AmplitudeToDB() if feat.to_db else None
        self.transform = None

    def _init_multi_feat(self, feat: DictConfig, sr: int) -> None:
        """One transform per channel; feature types given by cfg.feature.channel_features."""
        channel_features = list(feat.get("channel_features", ["melspec", "melspec", "melspec"]))
        self.channel_feat_types = channel_features
        if len(channel_features) != self.n_channels:
            raise ValueError(
                f"len(channel_features)={len(channel_features)} must equal n_channels={self.n_channels}"
            )

        self.feat_transforms = nn.ModuleDict()
        for ftype in set(channel_features):
            if ftype in ("melspec", "log_mel"):
                self.feat_transforms[ftype] = self._build_mel(feat, sr)
            elif ftype == "lofar":
                self.feat_transforms["lofar"] = self._build_lofar(feat)
            elif ftype == "demon":
                self.feat_transforms["demon"] = self._build_demon_mel(feat)
            else:
                raise NotImplementedError(f"channel_features: '{ftype}' not supported in multi_feat")

        if "lofar" in set(channel_features):
            self.lofar_n_bins = int(feat.get("lofar_n_bins", feat.n_mels))
        if "demon" in set(channel_features):
            self._store_demon_params(feat, sr)

        self.amp_to_db = T.AmplitudeToDB() if feat.to_db else None
        self.log_eps = float(feat.get("log_eps", 1e-6))
        self.transform = None

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (B, 1, T)
        x = wav.squeeze(1)  # (B, T)
        if not self._logged_shape:
            print(f"[Frontend.forward] input={tuple(wav.shape)}  device={wav.device}", flush=True)
            self._logged_shape = True

        if self.feature_type == "raw":
            return self._forward_raw(x)
        if self.channel_mode == "multi_res":
            return self._forward_multi_res(x)
        if self.channel_mode == "multi_feat":
            return self._forward_multi_feat(x)

        spec = self._compute_feature(x, self.feature_type, self.transform)  # (B, F, T')
        spec = self._normalize(spec)
        img = self._make_channels(spec)                                       # (B, C, H, W)
        if self.image_size is not None:
            img = F.interpolate(img, size=self.image_size, mode="bilinear", align_corners=False)
        if not hasattr(self, "_logged_out") or not self._logged_out:
            print(f"[Frontend.forward] output={tuple(img.shape)}", flush=True)
            self._logged_out = True
        return img

    def _forward_multi_res(self, x: torch.Tensor) -> torch.Tensor:
        specs = []
        for tr in self.res_transforms:
            s = tr(x)
            if self.res_amp_to_db is not None:
                s = self.res_amp_to_db(s)
            specs.append(self._normalize(s))
        target = self.image_size or list(specs[0].shape[-2:])
        resized = [
            F.interpolate(s.unsqueeze(1), size=target, mode="bilinear", align_corners=False).squeeze(1)
            for s in specs
        ]
        return torch.stack(resized, dim=1)  # (B, 3, H, W)

    def _forward_multi_feat(self, x: torch.Tensor) -> torch.Tensor:
        channels = []
        for ftype in self.channel_feat_types:
            tr = self.feat_transforms[ftype]
            spec = self._compute_feature(x, ftype, tr)
            spec = self._normalize(spec)
            target = self.image_size or list(spec.shape[-2:])
            s = F.interpolate(spec.unsqueeze(1), size=target, mode="bilinear", align_corners=False).squeeze(1)
            channels.append(s)
        return torch.stack(channels, dim=1)  # (B, C, H, W)

    def _forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        H, W = self.image_size if self.image_size else (int(T**0.5), int(T**0.5))
        target = H * W
        if T < target:
            x = F.pad(x, (0, target - T))
        img = x[:, :target].view(B, 1, H, W)
        return img.expand(-1, self.n_channels, -1, -1).contiguous()

    # ── feature computation ───────────────────────────────────────────────────

    def _compute_feature(
        self, x: torch.Tensor, ftype: str, transform: nn.Module
    ) -> torch.Tensor:
        """Compute a 2-D feature map (B, F, T') from waveform (B, T)."""
        if ftype == "melspec":
            spec = transform(x)
            if self.amp_to_db is not None:
                spec = self.amp_to_db(spec)

        elif ftype == "log_mel":
            # log(mel + ε) — natural log, distinct from dB (10·log10) scaling
            spec = transform(x)
            spec = torch.log(spec.clamp(min=self.log_eps))

        elif ftype in ("mfcc", "cqt"):
            spec = transform(x)
            if self.amp_to_db is not None:
                spec = self.amp_to_db(spec)

        elif ftype == "lofar":
            # High freq-resolution STFT; keep only low-freq bins (lofargram)
            spec = transform(x)[:, :self.lofar_n_bins, :]
            if self.amp_to_db is not None:
                spec = self.amp_to_db(spec)

        elif ftype == "demon":
            spec = self._compute_demon(x, transform)
            if self.amp_to_db is not None:
                spec = self.amp_to_db(spec)

        else:
            raise NotImplementedError(f"_compute_feature: unknown ftype '{ftype}'")

        return spec  # (B, F, T')

    def _compute_demon(self, x: torch.Tensor, mel_transform: nn.Module) -> torch.Tensor:
        """DEMON: bandpass → envelope → mel spectrogram of modulation signal.

        Detects amplitude modulations (e.g., propeller blade/shaft rates).
        """
        sr = self.demon_sr
        # 1. Bandpass: isolate noise band of interest
        x = AF.highpass_biquad(x, sr, cutoff_freq=self.demon_fmin)
        x = AF.lowpass_biquad(x, sr, cutoff_freq=self.demon_fmax)
        # 2. Envelope via full-wave rectification + lowpass smoothing
        x = x.abs()
        x = AF.lowpass_biquad(x, sr, cutoff_freq=self.demon_env_cutoff)
        # 3. Decimate to envelope SR so modulation frequencies are resolvable
        x = AF.resample(x, sr, self.demon_env_sr)
        # 4. Spectrogram of the envelope signal
        return mel_transform(x)  # (B, n_mels, T')

    # ── channel construction ──────────────────────────────────────────────────

    def _normalize(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.do_normalize:
            return spec
        mean = spec.mean(dim=(-2, -1), keepdim=True)
        std = spec.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        return (spec - mean) / std

    def _make_channels(self, spec: torch.Tensor) -> torch.Tensor:
        # spec: (B, F, T')
        if self.channel_mode == "repeat":
            # All 3 channels are identical — baseline for any single feature
            return spec.unsqueeze(1).expand(-1, self.n_channels, -1, -1).contiguous()

        elif self.channel_mode == "delta":
            # [spec, Δ, ΔΔ] — standard acoustic feature stack
            d1 = AF.compute_deltas(spec)
            d2 = AF.compute_deltas(d1)
            return torch.stack([spec, d1, d2], dim=1)  # (B, 3, F, T')

        elif self.channel_mode == "log_linear":
            # [linear, log, Δlog] — gives model both scale contexts
            log_s = torch.log(spec.clamp(min=1e-6))
            delta = AF.compute_deltas(log_s)
            # Each channel normalized independently so scales are comparable
            channels = [spec, log_s, delta]
            normed = []
            for c in channels:
                m = c.mean(dim=(-2, -1), keepdim=True)
                s = c.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
                normed.append((c - m) / s)
            return torch.stack(normed, dim=1)  # (B, 3, F, T')

        elif self.channel_mode == "harmonic_percussive":
            # [harmonic, percussive, residual] via median-filter HPSS
            H, P, R = self._hpss(spec)
            return torch.stack([H, P, R], dim=1)  # (B, 3, F, T')

        else:
            raise ValueError(
                f"Unknown channel_mode: '{self.channel_mode}'. "
                "Choices: repeat | delta | log_linear | harmonic_percussive | multi_res | multi_feat"
            )

    def _hpss(
        self, spec: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """GPU-compatible HPSS via median filtering + Wiener soft masks.

        Harmonic filter: median along time axis (highlights sustained tones).
        Percussive filter: median along freq axis (highlights transients).
        """
        sp = spec.clamp(min=0)

        # Median filter along time (last dim)
        kh = self.hpss_kernel_harm
        sp_t = F.pad(sp, (kh // 2, kh // 2), mode="reflect")
        harm = sp_t.unfold(-1, kh, 1).median(dim=-1).values  # (B, F, T)

        # Median filter along freq (second-to-last dim)
        kp = self.hpss_kernel_perc
        sp_f = F.pad(sp, (0, 0, kp // 2, kp // 2), mode="reflect")
        perc = sp_f.unfold(-2, kp, 1).median(dim=-1).values  # (B, F, T)

        # Wiener soft masks
        H2 = harm ** 2
        P2 = perc ** 2
        denom = H2 + P2 + 1e-8
        H = sp * H2 / denom
        P = sp * P2 / denom
        R = (sp - H - P).clamp(min=0)
        return H, P, R
