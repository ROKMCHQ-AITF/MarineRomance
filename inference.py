"""테스트 추론: 다중 체크포인트 앙상블 → submission.csv 생성."""
from __future__ import annotations

import argparse
import json
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


def _find_checkpoints(ckpt_arg: str) -> list[Path]:
    """단일 .pth 파일 또는 디렉토리 내 best_foldN.pth 목록 반환."""
    p = Path(ckpt_arg)
    if p.is_file():
        return [p]
    ckpts = sorted(p.glob("best_fold*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"체크포인트를 찾을 수 없음: {p}")
    return ckpts


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    multilabel: bool = True,
    cfg: DictConfig = None,
) -> np.ndarray:
    """단일 모델 예측. 반환: (N, C) 확률 (multilabel=sigmoid, single=softmax)."""
    model.eval()
    gpu_resize = cfg is not None and cfg.feature.compute_on == "cpu"
    image_size: list[int] = list(cfg.feature.image_size) if cfg is not None else []
    n_channels = int(cfg.feature.n_channels) if cfg is not None else 1
    channels_last = (cfg is not None and bool(cfg.train.get("channels_last", False))
                     and device.type == "cuda")
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    preds: list[np.ndarray] = []
    for x, _ in tqdm(loader, desc="infer", leave=False):
        x = x.to(device, non_blocking=True)
        if gpu_resize:
            x = torch.nn.functional.interpolate(x, size=image_size, mode="bilinear", align_corners=False)
            if x.shape[1] == 1 and n_channels > 1:
                x = x.repeat(1, n_channels, 1, 1)
        if channels_last:
            x = x.to(memory_format=torch.channels_last)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
        if multilabel:
            p = torch.sigmoid(logits)
        else:
            p = torch.softmax(logits, dim=-1)
        preds.append(p.cpu().float().numpy())
    return np.concatenate(preds, axis=0)  # (N, C)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.train.amp and device.type == "cuda"

    test_csv = Path(args.test_csv) if args.test_csv else Path(cfg.data.get("test_csv", "input/test_metadata.csv"))
    if not test_csv.exists():
        raise FileNotFoundError(f"테스트 CSV 없음: {test_csv}")
    test_df = pd.read_csv(test_csv)
    print(f"테스트 샘플: {len(test_df)}개")

    loader = DataLoader(
        AudioDataset(test_df, cfg, mode="test"),
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    ckpt_paths = _find_checkpoints(args.ckpt)
    print(f"체크포인트 {len(ckpt_paths)}개 앙상블")

    multilabel = cfg.data.multilabel
    ensemble_preds: list[np.ndarray] = []
    for ckpt_path in ckpt_paths:
        print(f"로드: {ckpt_path}")
        model = build_model(cfg).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        ensemble_preds.append(predict(model, loader, device, use_amp, multilabel=multilabel, cfg=cfg))

    final_preds = np.mean(ensemble_preds, axis=0)  # (N, C)

    out_path = Path(args.out) if args.out else Path("outputs") / cfg.exp_name / "submission.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    label_map_path = Path(cfg.data.folds_csv).parent / "label_map.json"
    if label_map_path.exists():
        with open(label_map_path) as f:
            label_map: dict[str, int] = json.load(f)
        idx_to_label = {v: k for k, v in label_map.items()}
    else:
        idx_to_label = {i: str(i) for i in range(final_preds.shape[1])}

    rows = []
    for i, row in test_df.iterrows():
        if multilabel:
            pred_classes = [idx_to_label[c] for c in np.where(final_preds[i] > args.threshold)[0]]
            prediction = " ".join(pred_classes) if pred_classes else "unknown"
        else:
            pred_cls = int(np.argmax(final_preds[i]))
            prediction = idx_to_label[pred_cls]
        rows.append({
            cfg.data.id_col: row[cfg.data.id_col],
            "prediction": prediction,
        })

    sub_df = pd.DataFrame(rows)
    sub_df.to_csv(out_path, index=False)
    print(f"submission.csv 저장 → {out_path}")
    print(sub_df.head())


if __name__ == "__main__":
    main()
