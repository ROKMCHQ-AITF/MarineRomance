"""오디오 파일 일괄 검증: 손상 파일 / SR 불일치 / 무음 / 이상 duration 검출.

try/except를 허용하는 유일한 스크립트 (CLAUDE.md §5 참고).
결과는 stdout 요약 + bad_files.csv 저장.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.config import load_config


def _check_file(path: Path, target_sr: int, min_dur: float = 0.5, silence_db: float = -60.0) -> dict[str, Any]:
    """단일 파일 검증. 에러는 삼키고 결과 dict에 기록한다 (verify 전용 예외)."""
    result: dict[str, Any] = {"path": str(path), "ok": True, "issues": []}

    if not path.exists():
        result["ok"] = False
        result["issues"].append("not_found")
        return result

    try:
        import librosa
        wav, sr = librosa.load(str(path), sr=None, mono=True)
    except Exception as e:
        result["ok"] = False
        result["issues"].append(f"load_error:{e}")
        return result

    # SR 불일치
    if sr != target_sr:
        result["issues"].append(f"sr_mismatch(got={sr},expect={target_sr})")

    duration = len(wav) / sr
    result["duration"] = round(duration, 3)
    result["sr"] = sr

    # 너무 짧음
    if duration < min_dur:
        result["issues"].append(f"too_short({duration:.2f}s)")

    # 무음 (RMS dB)
    rms = np.sqrt(np.mean(wav**2))
    db = 20 * np.log10(rms + 1e-9)
    result["rms_db"] = round(float(db), 1)
    if db < silence_db:
        result["issues"].append(f"silent(rms={db:.1f}dB)")

    if result["issues"]:
        result["ok"] = False

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="오디오 파일 일괄 검증")
    parser.add_argument("--config", required=True)
    parser.add_argument("--csv", default=None, help="검증할 CSV (기본: data.train_csv)")
    parser.add_argument("--out", default="outputs/bad_files.csv", help="불량 파일 목록 저장 경로")
    parser.add_argument("--silence_db", type=float, default=-60.0)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    data_cfg = cfg.data

    csv_path = Path(args.csv) if args.csv else Path(data_cfg.train_csv)
    audio_dir = Path(data_cfg.audio_dir)
    df = pd.read_csv(csv_path)
    print(f"검증 대상: {len(df)} files from {csv_path}")

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="verify"):
        path = audio_dir / str(row[data_cfg.id_col])
        res = _check_file(path, target_sr=data_cfg.sample_rate, silence_db=args.silence_db)
        results.append(res)

    bad = [r for r in results if not r["ok"]]
    print(f"\n총 {len(df)}개 중 불량 {len(bad)}개 ({len(bad)/len(df)*100:.1f}%)")

    if bad:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "ok", "sr", "duration", "rms_db", "issues"])
            writer.writeheader()
            for r in bad:
                r["issues"] = " | ".join(r["issues"])
                writer.writerow({k: r.get(k, "") for k in writer.fieldnames})
        print(f"불량 파일 목록 저장 → {out_path}")

        # 이슈 유형별 집계
        from collections import Counter
        counter: Counter = Counter()
        for r in bad:
            for issue in r["issues"].split(" | "):
                counter[issue.split("(")[0]] += 1
        print("\n[이슈 유형별 집계]")
        for issue, cnt in counter.most_common():
            print(f"  {issue}: {cnt}")
    else:
        print("불량 파일 없음. 데이터 검증 통과.")


if __name__ == "__main__":
    main()
