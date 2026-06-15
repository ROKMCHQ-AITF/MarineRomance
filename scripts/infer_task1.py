"""Task1 추론/평가: frozen 인코더 임베딩 + AIS → MLP head → 선종 예측.

val(라벨 有)  → macro-F1 + 클래스별 리포트
test(라벨 無) → submission_task1.csv (filename, predicted_class)

학습된 head(best_fold*.pth)는 인코더 없이 head 가중치만 들었으므로,
frozen 인코더는 config로 새로 만들고(embedding 추출용) head만 체크포인트에서 로드한다.
AIS의 sog 표준화는 train 통계로 고정한다(분포 누수 방지).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.train_head import encode_ais, sog_log_stats
from src.data.dataset import AudioDataset
from src.models.factory import build_model
from src.models.heads import build_head
from src.utils.config import load_config
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task1 추론/평가")
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True, help="학습된 head 체크포인트 (best_fold0.pth)")
    p.add_argument("--csv", required=True, help="val.csv 또는 test.csv")
    p.add_argument("--audio_dir", required=True, help="해당 split의 audio 디렉토리")
    p.add_argument("--out", default=None, help="submission 저장 경로 (test일 때)")
    p.add_argument("overrides", nargs="*")
    return p.parse_args()


@torch.no_grad()
def extract_embeddings(model, loader, device, use_amp) -> np.ndarray:
    """frontend(있으면)+backbone으로 (N, D) 임베딩 추출."""
    model.eval()
    out = []
    for x, _ in tqdm(loader, desc="embed", leave=False):
        x = x.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            z = model.frontend(x) if model.frontend is not None else x
            z = model.backbone(z)
        out.append(z.cpu().float().numpy())
    return np.concatenate(out, axis=0)


def main() -> None:
    args = parse_args()
    cfg: DictConfig = load_config(args.config, args.overrides)
    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.train.amp and device.type == "cuda"

    df = pd.read_csv(args.csv)
    label_col = cfg.data.label_col
    has_labels = label_col in df.columns
    print(f"추론 대상: {len(df)} clips ({'val·라벨有' if has_labels else 'test·라벨無'})")

    # frozen 인코더로 임베딩 추출 (audio_dir만 split에 맞게 교체)
    cfg.data.audio_dir = args.audio_dir
    loader = DataLoader(
        AudioDataset(df, cfg, mode="test"),
        batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.train.num_workers, pin_memory=True,
    )
    model = build_model(cfg).to(device)
    emb = extract_embeddings(model, loader, device, use_amp)  # (N, 768)

    # AIS concat — sog 표준화는 train(folds_csv) 통계로 고정
    if cfg.data.get("use_ais", False):
        sog_mean, sog_std = sog_log_stats(pd.read_csv(cfg.data.folds_csv))
        ais = encode_ais(df, sog_mean, sog_std)          # (N, 10)
        feat = np.concatenate([emb, ais], axis=1)
        print(f"AIS concat: +{ais.shape[1]} → feat_dim={feat.shape[1]}")
    else:
        feat = emb

    # head 로드 후 예측
    head = build_head(cfg, feat.shape[1])
    state = torch.load(args.ckpt, map_location="cpu")
    head.load_state_dict(state["model"])
    head.to(device).eval()
    with torch.no_grad():
        logits = head(torch.from_numpy(feat).float().to(device))
        pred_idx = logits.argmax(dim=1).cpu().numpy()

    # 인덱스 → 라벨명
    label_map_path = Path(cfg.data.folds_csv).parent / "label_map.json"
    with open(label_map_path) as f:
        label_map: dict[str, int] = json.load(f)
    idx_to_label = {v: k for k, v in label_map.items()}
    pred_names = [idx_to_label[i] for i in pred_idx]

    if has_labels:
        from sklearn.metrics import classification_report, f1_score
        y_true = df[label_col].astype(str).to_numpy()
        macro_f1 = f1_score(y_true, pred_names, average="macro", zero_division=0)
        print(f"\n>>> 공식 val macro-F1 = {macro_f1:.4f}")
        print(classification_report(y_true, pred_names, zero_division=0, digits=4))
    else:
        out_path = Path(args.out) if args.out else Path("outputs") / cfg.exp_name / "submission_task1.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sub = pd.DataFrame({cfg.data.id_col: df[cfg.data.id_col], "predicted_class": pred_names})
        sub.to_csv(out_path, index=False)
        print(f"\nsubmission 저장 → {out_path}")
        print(sub["predicted_class"].value_counts().to_string())


if __name__ == "__main__":
    main()
