"""
ESC-50 Baseline Training Pipeline
- Mel Spectrogram 입력
- Fold 기반 분할 (GroupKFold 효과 내장)
- AdamW + Cosine Annealing + Warmup
- Label Smoothing + Dropout + Gradient Clipping
- Early Stopping + Checkpoint
- 모델만 교체하면 되는 구조
"""

import os
import torch
import torch.nn as nn
import torchaudio
torchaudio.set_audio_backend("soundfile")
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, confusion_matrix
import argparse
import time

from models import ResNet18Audio
#,CRNNAudio, EfficientNetAudio


class ESC50Dataset(Dataset):
    def __init__(self, df, audio_dir, sr=22050, duration=5, n_mels=128, augment=False):
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.sr = sr
        self.duration = duration
        self.target_length = sr * duration
        self.augment = augment

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=1024,
            hop_length=512,
            n_mels=n_mels,
        )

        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=20)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=40)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.audio_dir, row["filename"])

        # 오디오 로드
        waveform, sr = torchaudio.load(path)

        # 리샘플링 (필요 시)
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            waveform = resampler(waveform)

        # 모노 변환
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 길이 맞추기 (패딩 or 자르기)
        if waveform.shape[1] < self.target_length:
            pad = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :self.target_length]

        # Mel Spectrogram
        mel = self.mel_transform(waveform)  # (1, n_mels, time)
        mel = torch.log(mel + 1e-9)  # Log Mel

        # SpecAugment (학습 시만)
        if self.augment:
            mel = self.freq_mask(mel)
            mel = self.time_mask(mel)

        # 3채널로 복제 (사전학습 모델 호환)
        mel = mel.repeat(3, 1, 1)  # (3, n_mels, time)

        label = row["target"]
        return mel, label


# ============================================================
# 2. Training / Evaluation
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device, max_norm=1.0):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for mel, label in loader:
        mel, label = mel.to(device), label.to(device)

        output = model(mel)
        loss = criterion(output, label)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item() * mel.size(0)
        preds = output.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(label.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for mel, label in loader:
        mel, label = mel.to(device), label.to(device)

        output = model(mel)
        loss = criterion(output, label)

        total_loss += loss.item() * mel.size(0)
        preds = output.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(label.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, np.array(all_preds), np.array(all_labels)


# ============================================================
# 3. Main
# ============================================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 데이터 로드 ──
    csv_path = os.path.join(args.data_dir, "esc50.csv")
    audio_dir = os.path.join(args.data_dir, "audio", "audio")
    df = pd.read_csv(csv_path)

    print(f"Total samples: {len(df)}")
    print(f"Classes: {df['category'].nunique()}")
    print(f"Folds: {sorted(df['fold'].unique())}")

    # ── Fold 기반 분할 (데이터 누수 없음) ──
    test_fold = args.test_fold
    val_fold = (test_fold % 5) + 1  # 다음 fold를 val로

    train_df = df[~df["fold"].isin([test_fold, val_fold])]
    val_df = df[df["fold"] == val_fold]
    test_df = df[df["fold"] == test_fold]

    print(f"\nSplit (test_fold={test_fold}, val_fold={val_fold}):")
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # ── Dataset & DataLoader ──
    train_ds = ESC50Dataset(train_df, audio_dir, sr=args.sr, augment=True)
    val_ds = ESC50Dataset(val_df, audio_dir, sr=args.sr, augment=False)
    test_ds = ESC50Dataset(test_df, audio_dir, sr=args.sr, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # ── 모델 선택 ──
    model_map = {
        "resnet18": ResNet18Audio,
        #"crnn": CRNNAudio,
        #"efficientnet": EfficientNetAudio,
    }

    if args.model not in model_map:
        raise ValueError(f"Unknown model: {args.model}. Choose from {list(model_map.keys())}")

    model = model_map[args.model](num_classes=50, dropout=args.dropout).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {args.model}")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # ── 손실 함수 (Label Smoothing) ──
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ── 옵티마이저 (AdamW) ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ── 스케줄러 (Cosine Annealing) ──
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── 학습 루프 + Early Stopping ──
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    checkpoint_path = f"best_{args.model}.pt"

    print(f"\n{'='*60}")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | {'LR':>10}")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, max_norm=args.max_norm
        )

        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start

        print(
            f"{epoch:>5} | {train_loss:>10.4f} | {train_acc:>8.4f} | "
            f"{val_loss:>8.4f} | {val_acc:>7.4f} | {current_lr:>10.6f}"
        )

        # Early Stopping + Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
                break

    # ── 최종 테스트 ──
    print(f"\n{'='*60}")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print(f"Loading best checkpoint: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\n[TEST RESULT]")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")

    # 클래스별 정확도 상위/하위 5개
    cm = confusion_matrix(test_labels, test_preds)
    class_acc = cm.diagonal() / cm.sum(axis=1)
    categories = sorted(df["category"].unique())

    print(f"\n  Top 5 classes:")
    for idx in np.argsort(class_acc)[-5:][::-1]:
        print(f"    {categories[idx]:>25s}: {class_acc[idx]*100:.1f}%")

    print(f"\n  Bottom 5 classes:")
    for idx in np.argsort(class_acc)[:5]:
        print(f"    {categories[idx]:>25s}: {class_acc[idx]*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESC-50 Baseline")
    parser.add_argument("--data_dir", type=str, default="archive", help="Path to ESC-50 archive folder")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "crnn", "efficientnet"])
    parser.add_argument("--test_fold", type=int, default=5, help="Fold number for test (1-5)")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    args = parser.parse_args()
    main(args)
