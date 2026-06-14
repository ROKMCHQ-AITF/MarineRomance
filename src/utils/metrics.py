# Competition metric functions: f1 / macro_f1 / auc / cmap.
from __future__ import annotations

from typing import Callable

import numpy as np
from omegaconf import DictConfig


def _f1_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    from sklearn.metrics import f1_score

    preds_bin = (y_pred > threshold).astype(int)
    return float(f1_score(y_true, preds_bin, average="samples", zero_division=0))


def _macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    from sklearn.metrics import f1_score

    preds_bin = (y_pred > threshold).astype(int)
    return float(f1_score(y_true, preds_bin, average="macro", zero_division=0))


def _auc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    try:
        return float(roc_auc_score(y_true, y_pred, average="macro"))
    except ValueError:
        return 0.0


def _cmap_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Class-mean average precision (common in audio classification competitions)."""
    from sklearn.metrics import average_precision_score

    try:
        return float(average_precision_score(y_true, y_pred, average="macro"))
    except ValueError:
        return 0.0


def get_metric_fn(cfg: DictConfig) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return (y_true, y_pred) -> float metric function from config."""
    name = cfg.metric.name
    if name == "f1":
        return _f1_score
    elif name == "macro_f1":
        return _macro_f1_score
    elif name == "auc":
        return _auc_score
    elif name == "cmap":
        return _cmap_score
    else:
        raise ValueError(f"Unknown metric: '{name}'")
