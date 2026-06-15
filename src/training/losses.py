# Loss builders: BCE / Focal / CE / LSEP / Combo. Mode-aware (single vs multi-label).
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


def _smooth_targets(targets: torch.Tensor, smoothing: float) -> torch.Tensor:
    """Apply label smoothing for multi-label: 1 → 1-s/2, 0 → s/2."""
    return targets * (1.0 - smoothing) + 0.5 * smoothing


def _resolve_class_weight(cw) -> torch.Tensor | None:
    """class_weight config → (C,) tensor or None. Accepts list or 'none'."""
    if cw is None:
        return None
    if isinstance(cw, str):
        if cw.lower() in ("none", ""):
            return None
        raise ValueError(
            f"class_weight: only an explicit list or 'none' is supported (got '{cw}'). "
            "'balanced'는 클래스 빈도가 필요해 미지원 — 리스트로 직접 지정하세요."
        )
    return torch.tensor([float(x) for x in cw], dtype=torch.float32)


# ── multi-label (sigmoid) ─────────────────────────────────────────────────────

class BCEWithLabelSmoothing(nn.Module):
    """BCE with optional label smoothing for multi-label classification."""

    def __init__(self, smoothing: float = 0.0) -> None:
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.smoothing > 0.0:
            targets = _smooth_targets(targets, self.smoothing)
        return F.binary_cross_entropy_with_logits(logits, targets)


class FocalLoss(nn.Module):
    """Sigmoid focal loss for multi-label classification, with optional label smoothing."""

    def __init__(self, gamma: float = 2.0, smoothing: float = 0.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.smoothing > 0.0:
            targets = _smooth_targets(targets, self.smoothing)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        return (((1 - pt) ** self.gamma) * bce).mean()


# ── single-label (softmax) ────────────────────────────────────────────────────

class WeightedCE(nn.Module):
    """Cross-entropy for single-label with optional per-class weight + smoothing."""

    def __init__(self, weight: torch.Tensor | None = None, label_smoothing: float = 0.0) -> None:
        super().__init__()
        # buffer로 등록 → model.to(device) 따라 이동 + state_dict에는 안 들어가도 무방
        self.register_buffer("weight", weight if weight is not None else None, persistent=False)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(logits.device) if self.weight is not None else None
        return F.cross_entropy(logits, targets, weight=w, label_smoothing=self.label_smoothing)


class MultiClassFocalLoss(nn.Module):
    """Softmax focal loss for single-label, with optional per-class weight (α) + smoothing."""

    def __init__(
        self, gamma: float = 2.0, weight: torch.Tensor | None = None, label_smoothing: float = 0.0
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None, persistent=False)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(logits.device) if self.weight is not None else None
        # 클래스 가중은 ce(reduction=none)에 반영, focal 변조는 정답확률 pt로 적용
        ce = F.cross_entropy(
            logits, targets, weight=w, label_smoothing=self.label_smoothing, reduction="none"
        )
        pt = torch.softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
        return (((1 - pt) ** self.gamma) * ce).mean()


class ComboLoss(nn.Module):
    """Weighted sum of multiple losses defined under loss.components."""

    def __init__(self, losses: list[nn.Module], weights: list[float]) -> None:
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return sum(w * l(logits, targets) for w, l in zip(self.weights, self.losses))


def _build_single(
    loss_cfg: DictConfig, global_smoothing: float, multilabel: bool, default_weight: torch.Tensor | None
) -> nn.Module:
    """Build one loss component. Routes to sigmoid (multi) or softmax (single) variants."""
    loss_type = loss_cfg.type
    smoothing = float(loss_cfg.get("label_smoothing", global_smoothing))
    # 컴포넌트 단위 class_weight 우선, 없으면 상위(global) class_weight 사용
    cw = _resolve_class_weight(loss_cfg.get("class_weight", None))
    if cw is None:
        cw = default_weight

    if multilabel:
        if loss_type == "bce":
            if smoothing > 0.0:
                return BCEWithLabelSmoothing(smoothing=smoothing)
            return nn.BCEWithLogitsLoss(pos_weight=cw)
        elif loss_type == "focal":
            return FocalLoss(gamma=float(loss_cfg.get("focal_gamma", 2.0)), smoothing=smoothing)
        elif loss_type == "ce":
            return nn.CrossEntropyLoss(weight=cw, label_smoothing=smoothing)
        elif loss_type == "lsep":
            raise NotImplementedError("LSEP loss not yet implemented")
        raise ValueError(f"Unknown loss type: '{loss_type}'")

    # single-label (softmax 기반)
    if loss_type == "ce":
        return WeightedCE(weight=cw, label_smoothing=smoothing)
    elif loss_type == "focal":
        return MultiClassFocalLoss(
            gamma=float(loss_cfg.get("focal_gamma", 2.0)), weight=cw, label_smoothing=smoothing
        )
    elif loss_type == "bce":
        raise ValueError("loss.type='bce'는 multi-label 전용입니다. single-label엔 'ce'/'focal'을 쓰세요.")
    elif loss_type == "lsep":
        raise NotImplementedError("LSEP loss not yet implemented")
    raise ValueError(f"Unknown loss type: '{loss_type}'")


def build_loss(cfg: DictConfig) -> nn.Module:
    """Return loss module from config.

    loss.type=combo 일 때 loss.components 리스트에서 각 loss를 조립해 가중합한다.
    loss.class_weight(리스트)는 CE의 weight=·focal의 α로 반영된다(single/multi 공통).
    """
    loss_type = cfg.loss.type
    global_smoothing = float(cfg.loss.get("label_smoothing", 0.0))
    multilabel = bool(cfg.data.multilabel)
    default_weight = _resolve_class_weight(cfg.loss.get("class_weight", None))

    if loss_type == "combo":
        components = cfg.loss.components  # list of {type, weight, ...}
        losses, weights = [], []
        for comp in components:
            losses.append(_build_single(comp, global_smoothing, multilabel, default_weight))
            weights.append(float(comp.get("weight", 1.0)))
        total = sum(weights)
        weights = [w / total for w in weights]  # 합이 1이 되도록 정규화
        return ComboLoss(losses, weights)

    return _build_single(cfg.loss, global_smoothing, multilabel, default_weight)
