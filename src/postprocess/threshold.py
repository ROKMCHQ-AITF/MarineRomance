# OOF 예측을 기반으로 최적 임계값(전역 or per-class)을 탐색하고 적용.
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score


def optimize_threshold_global(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "f1",
    n_steps: int = 100,
) -> float:
    """전역 단일 threshold를 grid search로 최적화. y_pred는 sigmoid 확률."""
    best_thr, best_score = 0.5, 0.0
    for thr in np.linspace(0.01, 0.99, n_steps):
        pred_bin = (y_pred > thr).astype(int)
        score = f1_score(y_true, pred_bin, average="samples", zero_division=0)
        if score > best_score:
            best_score = score
            best_thr = thr
    return float(best_thr)


def optimize_threshold_per_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_steps: int = 100,
) -> np.ndarray:
    """클래스별 threshold를 독립적으로 최적화. 반환: (num_classes,) array."""
    num_classes = y_pred.shape[1]
    thresholds = np.full(num_classes, 0.5)

    for c in range(num_classes):
        best_thr, best_score = 0.5, 0.0
        for thr in np.linspace(0.01, 0.99, n_steps):
            pred_bin = (y_pred[:, c] > thr).astype(int)
            score = f1_score(y_true[:, c], pred_bin, zero_division=0)
            if score > best_score:
                best_score = score
                best_thr = thr
        thresholds[c] = best_thr

    return thresholds


def apply_threshold(y_pred: np.ndarray, thresholds: float | np.ndarray) -> np.ndarray:
    """확률 예측 (N, C)에 threshold 적용 → binary (N, C)."""
    return (y_pred > thresholds).astype(np.int32)


def smooth_predictions(y_pred: np.ndarray, window: int = 3) -> np.ndarray:
    """시계열 예측에 이동 평균 스무딩 적용 (SED 등). (N, C) → (N, C)."""
    if window <= 1:
        return y_pred
    kernel = np.ones(window) / window
    smoothed = np.stack([
        np.convolve(y_pred[:, c], kernel, mode="same") for c in range(y_pred.shape[1])
    ], axis=1)
    return smoothed.astype(np.float32)


def load_oof_and_optimize(
    oof_paths: list[str],
    label_path: str,
    mode: str = "global",
    n_steps: int = 100,
) -> float | np.ndarray:
    """여러 fold OOF .npy를 합쳐 최적 threshold 계산.

    Args:
        oof_paths: 각 fold의 oof_foldN.npy 경로 목록
        label_path: 대응하는 정답 레이블 .npy 경로
        mode: 'global' | 'per_class'
    """
    preds = np.concatenate([np.load(p) for p in oof_paths], axis=0)
    labels = np.load(label_path)

    if mode == "global":
        return optimize_threshold_global(labels, preds, n_steps=n_steps)
    elif mode == "per_class":
        return optimize_threshold_per_class(labels, preds, n_steps=n_steps)
    else:
        raise ValueError(f"Unknown mode: '{mode}'. Use 'global' or 'per_class'.")
