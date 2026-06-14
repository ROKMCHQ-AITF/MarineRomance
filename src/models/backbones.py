# timm backbone builder: removes classifier head and returns (backbone, feat_dim).
from __future__ import annotations

import timm
import torch.nn as nn
from omegaconf import DictConfig


def build_backbone(cfg: DictConfig) -> tuple[nn.Module, int]:
    """Create timm backbone with pooled output, no classifier.

    Returns (backbone, feat_dim).
    """
    backbone = timm.create_model(
        cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        in_chans=cfg.model.in_chans,
        num_classes=0,       # strip classifier
        global_pool="avg",
    )
    feat_dim: int = backbone.num_features
    return backbone, feat_dim
