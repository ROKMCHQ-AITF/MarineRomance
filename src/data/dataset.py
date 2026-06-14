# AudioDataset: CSV row → (waveform tensor, label tensor). No feature extraction.
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.data.augment import apply_waveform_aug
from src.data.preprocessing import fix_length, load_audio, normalize_wave


class AudioDataset(Dataset):
    """Map CSV rows to (waveform, label) tensors.

    Feature computation is the frontend's job (GPU). This class only handles
    disk I/O, waveform preprocessing, and label encoding.
    """

    def __init__(self, df: pd.DataFrame, cfg: DictConfig, mode: str = "train") -> None:
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.mode = mode
        self.audio_dir = Path(cfg.data.audio_dir)
        self.target_len = int(cfg.data.sample_rate * cfg.data.duration)

        label_map_path = Path(cfg.data.folds_csv).parent / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path) as f:
                self.label_map: dict[str, int] = json.load(f)
        else:
            self.label_map = {}
        self.num_classes = len(self.label_map) if self.label_map else cfg.model.num_classes

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Debug mode: skip disk I/O entirely, return deterministic random tensors.
        if self.cfg.get("debug", False):
            rng = torch.Generator().manual_seed(i)
            wav_t = torch.randn(1, self.target_len, generator=rng)  # (1, T)
            label = torch.zeros(self.num_classes, dtype=torch.float32)
            label[i % self.num_classes] = 1.0
            return wav_t, label

        row = self.df.iloc[i]
        path = self.audio_dir / row[self.cfg.data.id_col]
        wav = load_audio(path, self.cfg.data.sample_rate)
        wav = normalize_wave(wav)
        crop_mode = self.cfg.data.crop if self.mode == "train" else "center"
        wav = fix_length(wav, self.target_len, mode=crop_mode)
        if self.mode == "train":
            wav = apply_waveform_aug(wav, self.cfg)
        wav_t = torch.from_numpy(wav).unsqueeze(0)  # (1, T)
        label = self._encode_label(row)
        return wav_t, label

    def _encode_label(self, row: pd.Series) -> torch.Tensor:
        """Encode raw label string to multi-hot or class-index tensor."""
        label_col = self.cfg.data.label_col
        if self.mode == "test" or label_col not in row.index:
            return torch.zeros(self.num_classes, dtype=torch.float32)
        raw = row[label_col]
        if self.cfg.data.multilabel:
            label = torch.zeros(self.num_classes, dtype=torch.float32)
            for name in str(raw).split():
                idx = self.label_map.get(name)
                if idx is not None:
                    label[idx] = 1.0
        else:
            idx = self.label_map.get(str(raw), 0)
            label = torch.tensor(idx, dtype=torch.long)
        return label
