"""
ESC-50 모델 3종 — train.py에서 교체해서 사용

1. ResNet18Audio     - CNN 베이스라인 (빠르고 가벼움)
2. CRNNAudio         - CNN + LSTM (시간 패턴 처리)
3. EfficientNetAudio - 경량 고성능 CNN (대회 상위권)

공통 인터페이스:
    model = ModelClass(num_classes=50, dropout=0.3)
    output = model(mel)  # mel: (batch, 3, 128, time)
    output shape: (batch, num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNet18Audio(nn.Module):
    def __init__(self, num_classes=50, dropout=0.3):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


# ============================================================
# 2. CRNNAudio — CNN + Bidirectional LSTM
# ============================================================
#
# CNN으로 주파수 축 로컬 패턴 추출
# BiLSTM으로 시간 축 순차 패턴 처리
# 가볍지만 시간 정보 활용 가능
#
class CRNNAudio(nn.Module):
    def __init__(self, num_classes=50, dropout=0.3):
        super().__init__()

        # CNN: 주파수 패턴 추출
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # 주파수 축만 압축
            nn.Dropout2d(0.1),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.1),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.1),
        )

        # LSTM: 시간 패턴 처리
        # CNN 출력: (batch, 128, freq//8, time)
        # freq=128 → 128//8=16 → LSTM input = 128*16 = 2048
        self.lstm = nn.LSTM(
            input_size=128 * 16,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # 분류기
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),  # BiLSTM: 128*2=256
        )

    def forward(self, x):
        # CNN
        x = self.cnn(x)  # (B, 128, 16, T)

        # reshape for LSTM: (B, T, features)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2)  # (B, T, C, F)
        x = x.reshape(B, T, C * F)  # (B, T, 2048)

        # LSTM
        x, _ = self.lstm(x)  # (B, T, 256)

        # 마지막 시간 스텝 + 평균 풀링 결합
        last = x[:, -1, :]          # (B, 256)
        avg = x.mean(dim=1)         # (B, 256)
        x = last + avg              # (B, 256)

        return self.classifier(x)


# ============================================================
# 3. EfficientNetAudio — 경량 고성능 CNN
# ============================================================
#
# EfficientNet-B0 ImageNet 사전학습
# BirdCLEF 상위권에서 자주 사용
# ResNet보다 파라미터 효율 좋음
#
class EfficientNetAudio(nn.Module):
    def __init__(self, num_classes=50, dropout=0.3):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)  # ← DEFAULT → None
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

if __name__ == "__main__":
    dummy = torch.randn(2, 3, 128, 216)  # (batch=2, channels=3, n_mels=128, time)

    test_models = [
        ("CRNNAudio", CRNNAudio),
    ]

    try:
        test_models.insert(0, ("ResNet18Audio", ResNet18Audio))
        test_models.append(("EfficientNetAudio", EfficientNetAudio))
    except Exception:
        pass

    for name, ModelClass in test_models:
        try:
            model = ModelClass(num_classes=50, dropout=0.3)
            output = model(dummy)
            params = sum(p.numel() for p in model.parameters())
            print(f"{name:>20s} | output: {output.shape} | params: {params:>12,}")
        except Exception as e:
            print(f"{name:>20s} | skipped (pretrained download needed): {e}")
