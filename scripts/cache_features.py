"""오디오 파일을 미리 spectrogram .npy로 변환해 I/O 병목 제거.

저장 경로: outputs/<exp_name>/cache/<filename_without_ext>.npy
학습 시 dataset.py가 캐시를 우선 로드하도록 연동된다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.feature_extractor import extract_feature, make_channels
from src.data.preprocessing import fix_length, load_audio, normalize_wave
from src.utils.config import load_config


def cache_one(
    path: Path,
    out_path: Path,
    cfg,
    target_len: int,
) -> bool:
    """단일 파일을 .npy로 저장. 성공 여부 반환."""
    if out_path.exists():
        return True
    try:
        wav = load_audio(path, cfg.data.sample_rate)
        wav = normalize_wave(wav)
        wav = fix_length(wav, target_len, mode="center")
        S = extract_feature(wav, cfg)      # (H, W)
        arr = make_channels(S, cfg)        # (C, H, W)
        # image_size 리사이즈 (선택)
        if cfg.feature.image_size:
            from PIL import Image
            h, w = cfg.feature.image_size
            arr = np.stack([
                np.array(Image.fromarray(arr[c]).resize((w, h), Image.BILINEAR))
                for c in range(arr.shape[0])
            ])
        np.save(out_path, arr)
        return True
    except Exception as e:
        print(f"[skip] {path.name}: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="특징 사전 캐싱")
    parser.add_argument("--config", required=True)
    parser.add_argument("--csv", default=None, help="기본: data.train_csv")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    data_cfg = cfg.data

    csv_path = Path(args.csv) if args.csv else Path(data_cfg.train_csv)
    audio_dir = Path(data_cfg.audio_dir)
    cache_dir = Path("outputs") / cfg.exp_name / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    target_len = int(data_cfg.sample_rate * data_cfg.duration)

    print(f"캐싱 시작: {len(df)} files → {cache_dir}")
    print(f"feature: {cfg.feature.type}, workers: {args.workers}")

    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        futures = {}
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for _, row in df.iterrows():
                fname = str(row[data_cfg.id_col])
                src = audio_dir / fname
                dst = cache_dir / (Path(fname).stem + ".npy")
                fut = ex.submit(cache_one, src, dst, cfg, target_len)
                futures[fut] = fname
            ok = sum(1 for f in tqdm(as_completed(futures), total=len(futures), desc="cache") if f.result())
    else:
        ok = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc="cache"):
            fname = str(row[data_cfg.id_col])
            src = audio_dir / fname
            dst = cache_dir / (Path(fname).stem + ".npy")
            ok += cache_one(src, dst, cfg, target_len)

    print(f"완료: {ok}/{len(df)} 파일 캐싱 성공")
    print(f"학습 시 cfg.feature.compute_on=cpu 로 설정하면 캐시를 사용합니다.")


if __name__ == "__main__":
    main()
