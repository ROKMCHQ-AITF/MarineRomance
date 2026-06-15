# CPU audio loading and waveform preprocessing utilities.
from __future__ import annotations

import numpy as np
import torchaudio
import torchaudio.functional as AF
from pathlib import Path


def load_audio(path: Path, target_sr: int) -> np.ndarray:
    """Load audio file, resample to target_sr, convert to mono. Returns (T,) float32."""
    wav, sr = torchaudio.load(str(path))  # (C, T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  # mono
    if sr != target_sr:
        wav = AF.resample(wav, sr, target_sr)
    return wav.squeeze(0).numpy().astype(np.float32)


def normalize_wave(wav: np.ndarray) -> np.ndarray:
    """Peak-normalize waveform to [-1, 1]."""
    peak = np.abs(wav).max()
    if peak > 0:
        wav = wav / peak
    return wav


def fix_length(wav: np.ndarray, length: int, mode: str = "random") -> np.ndarray:
    """Crop or repeat-pad waveform to exactly `length` samples."""
    if len(wav) < length:
        repeats = int(np.ceil(length / len(wav)))
        wav = np.tile(wav, repeats)
    if len(wav) > length:
        if mode == "random":
            start = np.random.randint(0, len(wav) - length + 1)
        elif mode == "first":
            start = 0
        else:  # center
            start = (len(wav) - length) // 2
        wav = wav[start : start + length]
    return wav[:length]


def trim_silence(wav: np.ndarray, top_db: float = 30.0) -> np.ndarray:
    """Remove leading/trailing silence using amplitude threshold."""
    trimmed, _ = librosa.effects.trim(wav, top_db=top_db)
    # Return original if trim removes everything
    return trimmed if len(trimmed) > 0 else wav
