"""
ESC-50 Confusion Matrix 기반 오류 분석
- 학습된 체크포인트(best_<model>.pt)를 로드해 test fold 추론
- Confusion Matrix + 클래스별 precision/recall/f1
- 가장 많이 혼동되는 클래스 쌍 Top-N
- 오분류 샘플 목록(filename, true, pred) CSV 저장
- Confusion Matrix 히트맵 PNG 저장

사용 예:
    python error_analysis.py --model efficientnet --test_fold 5
    python error_analysis.py --model efficientnet --topk 20 --outdir analysis
"""

import os
import sys
import argparse

# Windows 콘솔에서 한글 출력 깨짐 방지
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)
import matplotlib

matplotlib.use("Agg")  # 화면 없이 파일 저장
import matplotlib.pyplot as plt

from train import ESC50Dataset, evaluate
from models import ResNet18Audio
# from models import EfficientNetAudio, CRNNAudio

MODEL_MAP = {
    "resnet18": ResNet18Audio,
    # "efficientnet": EfficientNetAudio,
    # "crnn": CRNNAudio,
}


def plot_confusion_matrix(cm, categories, out_path, normalize=True):
    """50x50 Confusion Matrix 히트맵 저장."""
    mat = cm.astype(np.float64)
    if normalize:
        row_sum = mat.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        mat = mat / row_sum

    n = len(categories)
    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Recall (row-normalized)" if normalize else "Count")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(categories, rotation=90, fontsize=6)
    ax.set_yticklabels(categories, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [저장] 히트맵 -> {out_path}")


def print_confusion_matrix(cm, categories):
    """50x50 혼동행렬 원시 카운트를 콘솔에 전부 출력 (행=True, 열=Pred)."""
    n = len(categories)

    # 열 인덱스 = 클래스 범례 (열은 숫자로만 표기하므로)
    print("\n  ── 열 인덱스 범례 (열 번호 = 예측 클래스) ──")
    for i, c in enumerate(categories):
        print(f"    {i:>2}={c:<20s}", end="" if (i + 1) % 4 else "\n")
    if n % 4:
        print()

    # 각 행: 실제 클래스 이름 + 50개 카운트
    width = max(len(str(int(cm.max()))), 1)
    print(f"\n  ── Confusion Matrix (raw counts, 행=True / 열=Pred) ──")
    for i in range(n):
        counts = " ".join(f"{cm[i, j]:>{width}d}" for j in range(n))
        print(f"  {i:>2} {categories[i]:>20s} [ {counts} ]")


def top_confused_pairs(cm, categories, topk=15):
    """대각선 제외, 가장 많이 혼동되는 (true -> pred) 쌍 Top-N."""
    n = cm.shape[0]
    pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                row_total = cm[i].sum()
                rate = cm[i, j] / row_total if row_total else 0.0
                pairs.append((cm[i, j], rate, categories[i], categories[j]))
    pairs.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return pairs[:topk]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.outdir, exist_ok=True)

    # ── 데이터 로드 ──
    csv_path = os.path.join(args.data_dir, "esc50.csv")
    audio_dir = os.path.join(args.data_dir, "audio", "audio")
    df = pd.read_csv(csv_path)

    test_df = df[df["fold"] == args.test_fold].reset_index(drop=True)
    print(f"Test fold {args.test_fold}: {len(test_df)} samples")

    test_ds = ESC50Dataset(test_df, audio_dir, sr=args.sr, augment=False)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # ── 모델 로드 ──
    if args.model not in MODEL_MAP:
        raise ValueError(f"Unknown model: {args.model}. Choose from {list(MODEL_MAP.keys())}")
    ckpt = args.ckpt or f"best_{args.model}.pt"
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"체크포인트 없음: {ckpt}")

    model = MODEL_MAP[args.model](num_classes=50, dropout=0.0).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    print(f"Loaded checkpoint: {ckpt}")

    # ── 추론 ──
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)

    # category 이름 (target 인덱스 0..49 순서로 정렬)
    cat_map = (
        df[["target", "category"]]
        .drop_duplicates()
        .sort_values("target")["category"]
        .tolist()
    )
    label_idx = list(range(len(cat_map)))

    print(f"\n[TEST RESULT]  loss={test_loss:.4f}  acc={test_acc:.4f} ({test_acc*100:.1f}%)")

    # ── Confusion Matrix ──
    cm = confusion_matrix(labels, preds, labels=label_idx)
    print_confusion_matrix(cm, cat_map)

    # ── 클래스별 지표 ──
    report_dict = classification_report(
        labels, preds, labels=label_idx, target_names=cat_map,
        output_dict=True, zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_path = os.path.join(args.outdir, f"{args.model}_classification_report.csv")
    report_df.to_csv(report_path)
    print(f"  [저장] 클래스별 지표 -> {report_path}")

    # 클래스별 정확도(recall) 정렬
    class_acc = cm.diagonal() / np.clip(cm.sum(axis=1), 1, None)
    order = np.argsort(class_acc)

    print("\n  ── 최약체 클래스 (recall 하위 10) ──")
    for idx in order[:10]:
        print(f"    {cat_map[idx]:>22s}: {class_acc[idx]*100:5.1f}%  (n={cm[idx].sum()})")

    print("\n  ── 최강 클래스 (recall 상위 5) ──")
    for idx in order[::-1][:5]:
        print(f"    {cat_map[idx]:>22s}: {class_acc[idx]*100:5.1f}%")

    # ── 가장 많이 혼동되는 쌍 ──
    pairs = top_confused_pairs(cm, cat_map, topk=args.topk)
    print(f"\n  ── 가장 혼동 많은 쌍 Top {args.topk}  (true -> pred) ──")
    print(f"    {'count':>5}  {'rate':>6}  true -> pred")
    pair_rows = []
    for cnt, rate, t, p in pairs:
        print(f"    {cnt:>5}  {rate*100:5.1f}%  {t} -> {p}")
        pair_rows.append({"count": cnt, "rate": rate, "true": t, "pred": p})
    pd.DataFrame(pair_rows).to_csv(
        os.path.join(args.outdir, f"{args.model}_confused_pairs.csv"), index=False
    )

    # ── 오분류 샘플 목록 ──
    test_df = test_df.copy()
    test_df["pred_target"] = preds
    test_df["pred_category"] = [cat_map[p] for p in preds]
    test_df["correct"] = test_df["target"] == test_df["pred_target"]
    mis = test_df[~test_df["correct"]][
        ["filename", "category", "pred_category", "fold"]
    ].rename(columns={"category": "true_category"})
    mis_path = os.path.join(args.outdir, f"{args.model}_misclassified.csv")
    mis.to_csv(mis_path, index=False)
    print(f"\n  오분류 {len(mis)}/{len(test_df)}건  -> {mis_path}")

    # ── Confusion Matrix 저장 (CSV + 히트맵) ──
    cm_df = pd.DataFrame(cm, index=cat_map, columns=cat_map)
    cm_df.to_csv(os.path.join(args.outdir, f"{args.model}_confusion_matrix.csv"))
    plot_confusion_matrix(
        cm, cat_map,
        os.path.join(args.outdir, f"{args.model}_confusion_matrix.png"),
        normalize=args.normalize,
    )

    print(f"\n완료. 결과물은 '{args.outdir}/' 폴더에 저장되었습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESC-50 Confusion Matrix 오류 분석")
    parser.add_argument("--data_dir", type=str, default="archive")
    parser.add_argument("--model", type=str, default="resnet18", choices=list(MODEL_MAP.keys()))
    parser.add_argument("--ckpt", type=str, default=None, help="체크포인트 경로 (기본 best_<model>.pt)")
    parser.add_argument("--test_fold", type=int, default=5, help="테스트 fold (학습 때와 동일하게)")
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--topk", type=int, default=15, help="혼동 쌍 Top-N")
    parser.add_argument("--outdir", type=str, default="analysis")
    parser.add_argument("--normalize", action="store_true", default=True, help="히트맵 행 정규화")
    args = parser.parse_args()
    main(args)
