# AudioDataset: CSV row → (waveform tensor, label tensor). No feature extraction.
# compute_on=gpu: raw audio → (1, T) waveform, Frontend handles spectrogram on GPU.
# compute_on=cpu: cached .npy → (C, H, W) image tensor, Frontend is skipped.
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

# RAM 상주 packed array를 train/valid 데이터셋이 공유 (경로별 1회만 로드 → 중복 적재 방지).
_RAM_CACHE: dict[str, np.ndarray] = {}


def _load_packed_ram(path: Path) -> np.ndarray:
    """packed .npy를 RAM에 1회 로드 후 경로별 캐시. 같은 파일은 재사용."""
    key = str(path)
    if key not in _RAM_CACHE:
        _RAM_CACHE[key] = np.load(path)
    return _RAM_CACHE[key]


class AudioDataset(Dataset):
    """Map CSV rows to (waveform, label) tensors.

    compute_on=gpu : raw audio 로드 → (1, T), Frontend가 GPU에서 스펙트로그램 변환.
    compute_on=cpu : 캐시된 .npy 로드 → (C, H, W), Frontend 완전 skip.
    """

    def __init__(self, df: pd.DataFrame, cfg: DictConfig, mode: str = "train") -> None:
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.mode = mode
        self.use_cache = cfg.feature.compute_on == "cpu"

        if self.use_cache:
            self.cache_dir = Path(cfg.feature.cache_dir)
            self.image_size = list(cfg.feature.image_size)
            self.channel_mode = cfg.feature.channel_mode
            self.n_channels = cfg.feature.n_channels
            if self.channel_mode == "multi_feat":
                self.feat_types: list[str] = list(cfg.feature.channel_features)
            else:
                self.feat_types = [cfg.feature.type]

            # cache_in_ram=true면 packed array를 통째로 RAM에 올린다 (랜덤접근 mmap 디스크
            # 페이지폴트 제거 → 배치당 수초 → 수ms). false면 mmap (RAM 부족 시).
            self.cache_in_ram = bool(cfg.feature.get("cache_in_ram", False))
            # packed 파일 감지: {cache_dir}/{feat_type}_packed.npy 있으면 사용
            self.packed: dict[str, np.ndarray] = {}
            self.packed_index: dict[str, dict[str, int]] = {}
            for ft in self.feat_types:
                packed_npy = self.cache_dir / f"{ft}_packed.npy"
                packed_idx = self.cache_dir / f"{ft}_index.json"
                if packed_npy.exists() and packed_idx.exists():
                    if self.cache_in_ram:
                        self.packed[ft] = _load_packed_ram(packed_npy)  # 전체 RAM 상주(공유)
                        mode_str = "packed RAM"
                    else:
                        self.packed[ft] = np.load(packed_npy, mmap_mode="r")
                        mode_str = "packed mmap"
                    with open(packed_idx) as f:
                        self.packed_index[ft] = json.load(f)
                    print(f"[Dataset:{mode}] {ft} → {mode_str} {tuple(self.packed[ft].shape)}", flush=True)
                else:
                    print(f"[Dataset:{mode}] {ft} → 개별 .npy 로드 (패킹 권장: scripts/pack_features.py)", flush=True)
        else:
            self.audio_dir = Path(cfg.data.audio_dir)
            self.target_len = int(cfg.data.sample_rate * cfg.data.duration)

        label_map_path = Path(cfg.data.folds_csv).parent / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path) as f:
                self.label_map: dict[str, int] = json.load(f)
        else:
            print(
                f"[WARN] label_map.json not found at {label_map_path}. "
                "Run prepare_folds.py first. All labels will fall back to class 0.",
                flush=True,
            )
            self.label_map = {}
        self.num_classes = len(self.label_map) if self.label_map else cfg.model.num_classes
        print(
            f"[Dataset:{mode}] samples={len(df)}  num_classes={self.num_classes}"
            f"  multilabel={cfg.data.multilabel}  compute_on={cfg.feature.compute_on}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.get("debug", False):
            if self.use_cache:
                H, W = self.image_size
                x = torch.randn(self.n_channels, H, W)
            else:
                rng = torch.Generator().manual_seed(i)
                x = torch.randn(1, self.target_len, generator=rng)  # (1, T)
            if self.cfg.data.multilabel:
                label = torch.zeros(self.num_classes, dtype=torch.float32)
                label[i % self.num_classes] = 1.0
            else:
                label = torch.tensor(i % self.num_classes, dtype=torch.long)
            return x, label

        row = self.df.iloc[i]
        label = self._encode_label(row)

        if self.use_cache:
            x = self._load_cached(row)
        else:
            x = self._load_audio(row)

        return x, label

    def _load_cached(self, row: pd.Series) -> torch.Tensor:
        """캐시된 feature 로드 → (C, F, T). resize·repeat은 GPU에서 배치 단위로 처리."""
        stem = Path(str(row[self.cfg.data.id_col])).stem
        channels = []
        for feat_type in self.feat_types:
            # packed에 있으면 packed에서, 없으면(val/test 등) 개별 .npy로 fallback
            if feat_type in self.packed and stem in self.packed_index[feat_type]:
                idx = self.packed_index[feat_type][stem]
                feat = np.asarray(self.packed[feat_type][idx], dtype=np.float32)  # (F, T)
            else:
                npy_path = self.cache_dir / feat_type / f"{stem}.npy"
                feat = np.load(npy_path).astype(np.float32)  # (F, T)
            channels.append(feat)

        if self.channel_mode == "repeat":
            # 1채널만 넘기고 n_channels 복제는 GPU에서 (CPU memcpy·PCIe 전송 3× 절감)
            arr = channels[0][None, ...]  # (1, F, T)
        elif self.channel_mode == "delta":
            base = channels[0]
            d1 = self._delta(base)
            d2 = self._delta(d1)
            arr = np.stack([base, d1, d2], axis=0)  # (3, F, T)
        else:
            # multi_feat: 각 채널 그대로 stack
            arr = np.stack(channels, axis=0)  # (C, F, T)

        return torch.from_numpy(arr)  # (C, F, T) — resize는 GPU에서 배치 단위로

    def _load_audio(self, row: pd.Series) -> torch.Tensor:
        """Raw audio 로드 → (1, T)."""
        path = self.audio_dir / row[self.cfg.data.id_col]
        wav = load_audio(path, self.cfg.data.sample_rate)
        wav = normalize_wave(wav)
        crop_mode = self.cfg.data.crop if self.mode == "train" else "center"
        wav = fix_length(wav, self.target_len, mode=crop_mode)
        if self.mode == "train":
            wav = apply_waveform_aug(wav, self.cfg)
        return torch.from_numpy(wav).unsqueeze(0)  # (1, T)

    @staticmethod
    def _delta(feat: np.ndarray, width: int = 9) -> np.ndarray:
        """1차 차분 (librosa delta 근사)."""
        pad = width // 2
        padded = np.pad(feat, ((0, 0), (pad, pad)), mode="edge")
        slope = np.arange(-pad, pad + 1, dtype=np.float32)
        norm = (slope ** 2).sum()
        return np.stack([
            (padded[:, t:t + width] * slope).sum(axis=1) / (norm + 1e-8)
            for t in range(feat.shape[1])
        ], axis=1)

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
