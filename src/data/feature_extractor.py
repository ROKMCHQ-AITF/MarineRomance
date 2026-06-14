# CPU 특징추출: librosa 기반 mel/mfcc/cqt → np.ndarray. 캐싱 경로 전용.
from __future__ import annotations

import numpy as np
import librosa
from omegaconf import DictConfig


def extract_feature(wav: np.ndarray, cfg: DictConfig) -> np.ndarray:
    """waveform (T,) → spectrogram (H, W) on CPU.

    frontend.py 의 GPU 경로와 동일한 파라미터를 공유한다.
    반환 shape는 (n_mels, T') 또는 (n_mfcc, T') 등 2D.
    """
    feat = cfg.feature
    sr = cfg.data.sample_rate

    if feat.type == "melspec":
        S = librosa.feature.melspectrogram(
            y=wav,
            sr=sr,
            n_fft=feat.n_fft,
            hop_length=feat.hop_length,
            win_length=feat.win_length,
            n_mels=feat.n_mels,
            fmin=feat.fmin,
            fmax=feat.fmax,
            power=feat.power,
        )
        if feat.to_db:
            S = librosa.power_to_db(S, ref=np.max)

    elif feat.type == "mfcc":
        S = librosa.feature.mfcc(
            y=wav,
            sr=sr,
            n_mfcc=feat.get("n_mfcc", 40),
            n_fft=feat.n_fft,
            hop_length=feat.hop_length,
            n_mels=feat.n_mels,
            fmin=feat.fmin,
            fmax=feat.fmax,
        )

    elif feat.type == "cqt":
        C = librosa.cqt(
            y=wav,
            sr=sr,
            hop_length=feat.hop_length,
            fmin=feat.fmin,
            n_bins=feat.get("n_bins", 84),
            bins_per_octave=feat.get("bins_per_octave", 12),
        )
        S = np.abs(C)
        if feat.to_db:
            S = librosa.amplitude_to_db(S, ref=np.max)

    else:
        raise NotImplementedError(f"feature type '{feat.type}' not implemented in CPU extractor")

    if feat.normalize:
        mean = S.mean()
        std = S.std() + 1e-6
        S = (S - mean) / std

    return S.astype(np.float32)  # (H, W)


def make_channels(S: np.ndarray, cfg: DictConfig) -> np.ndarray:
    """(H, W) → (C, H, W) multi-channel 변환 (frontend._make_channels CPU 등가)."""
    feat = cfg.feature
    if feat.channel_mode == "repeat":
        return np.stack([S] * feat.n_channels, axis=0)  # (C, H, W)
    elif feat.channel_mode == "delta":
        d1 = librosa.feature.delta(S)
        d2 = librosa.feature.delta(S, order=2)
        return np.stack([S, d1, d2], axis=0)
    else:
        raise NotImplementedError(f"channel_mode '{feat.channel_mode}' not implemented in CPU extractor")
