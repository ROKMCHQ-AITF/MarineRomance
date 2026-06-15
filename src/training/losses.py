# Loss builders: BCE / Focal / CE / LSEP.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


def _smooth_targets(targets: torch.Tensor, smoothing: float) -> torch.Tensor:
    """Apply label smoothing for multi-label: 1 → 1-s/2, 0 → s/2."""
    return targets * (1.0 - smoothing) + 0.5 * smoothing


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


class MulticlassFocalLoss(nn.Module):
    """Softmax(CE) focal loss for single-label classification.

    loss = α_c · (1 - p_t)^γ · CE,  p_t = softmax 확률 중 정답 클래스 값.
    weight(α): 클래스별 가중치 (불균형 보정). gamma(γ): 쉬운 샘플 down-weight.
    """

    def __init__(
        self, gamma: float = 2.0, weight: torch.Tensor | None = None, smoothing: float = 0.0
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.register_buffer("weight", weight)  # None이면 등록만, .to(device) 따라감

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C), targets: (B,) long
        logp = F.log_softmax(logits, dim=-1)                       # (B, C)
        logp_t = logp.gather(1, targets.unsqueeze(1)).squeeze(1)   # (B,) 정답 log prob
        pt = logp_t.exp()                                          # (B,)
        focal = (1.0 - pt).clamp(min=0.0) ** self.gamma            # (B,) 변조항

        if self.smoothing > 0.0:
            # label smoothing: (1-ε)·정답 NLL + ε·전체 평균 NLL
            ce = (1.0 - self.smoothing) * (-logp_t) + self.smoothing * (-logp.mean(dim=1))
        else:
            ce = -logp_t

        loss = focal * ce                                          # (B,)
        if self.weight is not None:
            at = self.weight.gather(0, targets)                    # (B,) 샘플별 α
            return (at * loss).sum() / at.sum().clamp(min=1e-8)    # 가중 평균
        return loss.mean()


class ComboLoss(nn.Module):
    """Weighted sum of multiple losses defined under loss.components."""

    def __init__(self, losses: list[nn.Module], weights: list[float]) -> None:
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return sum(w * l(logits, targets) for w, l in zip(self.weights, self.losses))


def _build_single(
    loss_cfg: DictConfig,
    global_smoothing: float,
    class_weights: torch.Tensor | None = None,
    multilabel: bool = True,
) -> nn.Module:
    """Build one loss component from a component-level config dict.

    class_weights: (C,) 텐서. ce는 weight=, bce는 pos_weight=, softmax focal은 α로 적용.
    multilabel=False면 focal은 softmax(CE) 기반 MulticlassFocalLoss를 사용.
    """
    loss_type = loss_cfg.type
    smoothing = float(loss_cfg.get("label_smoothing", global_smoothing))
    if loss_type == "bce":
        if smoothing > 0.0:
            return BCEWithLabelSmoothing(smoothing=smoothing)
        return nn.BCEWithLogitsLoss(pos_weight=class_weights)
    elif loss_type == "focal":
        gamma = float(loss_cfg.get("focal_gamma", 2.0))
        if multilabel:
            return FocalLoss(gamma=gamma, smoothing=smoothing)  # sigmoid focal
        return MulticlassFocalLoss(gamma=gamma, weight=class_weights, smoothing=smoothing)
    elif loss_type == "ce":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=smoothing)
    elif loss_type == "lsep":
        raise NotImplementedError("LSEP loss not yet implemented")
    else:
        raise ValueError(f"Unknown loss type: '{loss_type}'")


def build_loss(cfg: DictConfig, class_weights: torch.Tensor | None = None) -> nn.Module:
    """Return loss module from config.

    loss.type=combo 일 때 loss.components 리스트에서 각 loss를 조립해 가중합한다.
    class_weights: 클래스 불균형 보정용 (C,) 텐서. trainer가 train 분포에서 계산해 전달.
    """
    loss_type = cfg.loss.type
    global_smoothing = float(cfg.loss.get("label_smoothing", 0.0))
    multilabel = bool(cfg.data.multilabel)

    if loss_type == "combo":
        components = cfg.loss.components  # list of {type, weight, ...}
        losses, weights = [], []
        for comp in components:
            losses.append(_build_single(comp, global_smoothing, class_weights, multilabel))
            weights.append(float(comp.get("weight", 1.0)))
        total = sum(weights)
        weights = [w / total for w in weights]  # 합이 1이 되도록 정규화
        return ComboLoss(losses, weights)

    return _build_single(cfg.loss, global_smoothing, class_weights, multilabel)
