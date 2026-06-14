"""테스트 추론: TTA + 다중 체크포인트 앙상블 → submission.csv 생성."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import AudioDataset
from src.models.factory import build_model
from src.utils.config import load_config
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaggle audio classification — 추론 및 제출 파일 생성")
    parser.add_argument("--config", required=True, help="실험 yaml 경로")
    parser.add_argument("--ckpt", required=True, help="체크포인트 디렉토리 또는 단일 .pth 파일")
    parser.add_argument("--test_csv", default=None, help="테스트 메타데이터 CSV (기본: data.test_csv)")
    parser.add_argument("--out", default=None, help="submission.csv 저장 경로")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


@torch.no_grad()
def predict_one_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta_modes: list[str],
    use_amp: bool,
) -> np.ndarray:
    """단일 모델 + TTA 예측. 반환: (N, C) sigmoid 확률."""
    model.eval()
    all_preds: list[np.ndarray] = []

    for tta in tta_modes:
        fold_preds: list[np.ndarray] = []
        for x, _ in tqdm(loader, desc=f"TTA={tta}", leave=False):
            x = _apply_tta(x, tta).to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
            probs = torch.sigmoid(logits).cpu().float().numpy()
            fold_preds.append(probs)
        all_preds.append(np.concatenate(fold_preds, axis=0))  # (N, C)

    return np.mean(all_preds, axis=0)  # TTA 평균


def _apply_tta(x: torch.Tensor, mode: str) -> torch.Tensor:
    """TTA 변환 적용. x: (B, 1, T) waveform."""
    if mode == "none":
        return x
    elif mode == "flip":
        return x.flip(-1)
    elif mode == "gain_up":
        return (x * 1.2).clamp(-1, 1)
    elif mode == "gain_down":
        return x * 0.8
    else:
        raise ValueError(f"Unknown TTA mode: '{mode}'")


def _find_checkpoints(ckpt_arg: str) -> list[Path]:
    """단일 .pth 파일 또는 디렉토리 내 best_foldN.pth 목록 반환."""
    p = Path(ckpt_arg)
    if p.is_file():
        return [p]
    ckpts = sorted(p.glob("best_fold*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"체크포인트를 찾을 수 없음: {p}")
    return ckpts


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.train.amp and device.type == "cuda"

    # 테스트 데이터 로드
    test_csv = Path(args.test_csv) if args.test_csv else Path(cfg.data.get("test_csv", "input/test_metadata.csv"))
    if not test_csv.exists():
        raise FileNotFoundError(f"테스트 CSV 없음: {test_csv}")
    test_df = pd.read_csv(test_csv)
    print(f"테스트 샘플: {len(test_df)}개")

    dataset = AudioDataset(test_df, cfg, mode="test")
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    tta_modes: list[str] = list(cfg.inference.tta) if cfg.inference.tta else ["none"]
    ckpt_paths = _find_checkpoints(args.ckpt)
    print(f"체크포인트 {len(ckpt_paths)}개, TTA {tta_modes}")

    # 모델별 예측 후 앙상블 평균
    ensemble_preds: list[np.ndarray] = []
    for ckpt_path in ckpt_paths:
        print(f"로드: {ckpt_path}")
        model = build_model(cfg).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        preds = predict_one_model(model, loader, device, tta_modes, use_amp)
        ensemble_preds.append(preds)

    final_preds = np.mean(ensemble_preds, axis=0)  # (N, C)

    # submission.csv 생성
    out_path = Path(args.out) if args.out else Path("outputs") / cfg.exp_name / "submission.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # binary threshold 적용 (multi-label의 경우 공백 구분 레이블)
    # label_map 역매핑: idx → name
    import json
    label_map_path = Path(cfg.data.folds_csv).parent / "label_map.json"
    if label_map_path.exists():
        with open(label_map_path) as f:
            label_map: dict[str, int] = json.load(f)
        idx_to_label = {v: k for k, v in label_map.items()}
    else:
        idx_to_label = {i: str(i) for i in range(final_preds.shape[1])}

    rows = []
    thr = args.threshold
    for i, row in test_df.iterrows():
        pred_classes = [idx_to_label[c] for c in np.where(final_preds[i] > thr)[0]]
        rows.append({
            cfg.data.id_col: row[cfg.data.id_col],
            "prediction": " ".join(pred_classes) if pred_classes else "unknown",
        })

    sub_df = pd.DataFrame(rows)
    sub_df.to_csv(out_path, index=False)
    print(f"submission.csv 저장 → {out_path}")
    print(sub_df.head())


if __name__ == "__main__":
    main()
