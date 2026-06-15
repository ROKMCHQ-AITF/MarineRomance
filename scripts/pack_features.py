"""개별 .npy 파일을 하나의 packed array로 묶어 I/O 병목을 제거한다.

출력:
  cache_dir/{feat_type}_packed.npy   shape: (N, F, T) float32
  cache_dir/{feat_type}_index.json   {"stem": int_index, ...}

사용:
  python scripts/pack_features.py --config configs/default.yaml
  python scripts/pack_features.py --config configs/default.yaml --feat_types melspec lofar demon
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# `python scripts/pack_features.py`로 직접 실행해도 src 패키지를 찾도록 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config


def pack_one(feat_dir: Path, out_dir: Path, feat_type: str, dtype: str = "auto") -> None:
    files = sorted(feat_dir.glob("*.npy"))
    if not files:
        print(f"[WARN] {feat_dir} 에 .npy 파일 없음. 스킵.", flush=True)
        return

    # 첫 파일로 shape·원본 dtype 확인
    sample = np.load(files[0])
    shape = (len(files), *sample.shape)  # (N, F, T)

    # auto: 개별 .npy의 원본 dtype 보존 (fp16 소스면 fp16 유지 → 불필요한 upcast 방지)
    if dtype == "auto":
        np_dtype = sample.dtype
    elif dtype == "float16":
        np_dtype = np.float16
    else:
        np_dtype = np.float32
    print(f"[{feat_type}] {len(files)}개 파일 패킹 중 (source={sample.dtype} → packed={np.dtype(np_dtype)})...", flush=True)

    packed = np.zeros(shape, dtype=np_dtype)
    index: dict[str, int] = {}

    for i, f in enumerate(tqdm(files, desc=feat_type, dynamic_ncols=True)):
        packed[i] = np.load(f).astype(np_dtype)
        index[f.stem] = i

    out_npy = out_dir / f"{feat_type}_packed.npy"
    out_idx = out_dir / f"{feat_type}_index.json"

    np.save(out_npy, packed)
    with open(out_idx, "w") as f:
        json.dump(index, f)

    size_mb = out_npy.stat().st_size / 1024 ** 2
    print(f"[{feat_type}] 저장 완료 → {out_npy} ({size_mb:.1f} MB)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="feature .npy 파일 패킹")
    parser.add_argument("--config", required=True)
    parser.add_argument("--feat_types", nargs="*", default=None,
                        help="패킹할 feature 타입 지정. 미지정시 config에서 자동 감지")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16"], default="auto",
                        help="auto: 개별 .npy 원본 dtype 보존(권장) | float16: 강제 반감 | float32: 강제 승격")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    cache_dir = Path(cfg.feature.cache_dir)

    if args.feat_types:
        feat_types = args.feat_types
    elif cfg.feature.channel_mode == "multi_feat":
        feat_types = list(cfg.feature.channel_features)
    else:
        feat_types = [cfg.feature.type]

    print(f"cache_dir: {cache_dir}", flush=True)
    print(f"패킹 대상: {feat_types}", flush=True)

    for feat_type in feat_types:
        feat_dir = cache_dir / feat_type
        if not feat_dir.exists():
            print(f"[WARN] {feat_dir} 없음. 스킵.", flush=True)
            continue
        pack_one(feat_dir, cache_dir, feat_type, dtype=args.dtype)

    print("전체 패킹 완료.", flush=True)


if __name__ == "__main__":
    main()
