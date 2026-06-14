# Audio pretrained model loaders: PANNs CNN14 and wav2vec2.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


# ── PANNs CNN14 ────────────────────────────────────────────────────────────────

class _ConvBlock(nn.Module):
    """Two Conv2d layers with BN + ReLU, followed by avg pooling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, pool_size: tuple[int, int] = (2, 2)) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, kernel_size=pool_size)  # pool_size=(1,1) = no spatial reduction


class CNN14(nn.Module):
    """PANNs-style CNN14 backbone.

    Input:  (B, C, H, W) spectrogram from Frontend
    Output: (B, 2048) embedding — fed into heads.py classifier.

    Architecture matches Gong et al. PANNs CNN14. Pretrained weights can be
    loaded via cfg.model.pretrained_path (strict=False so head is ignored).
    """

    FEAT_DIM = 2048

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.conv_block1 = _ConvBlock(in_channels, 64)
        self.conv_block2 = _ConvBlock(64, 128)
        self.conv_block3 = _ConvBlock(128, 256)
        self.conv_block4 = _ConvBlock(256, 512)
        self.conv_block5 = _ConvBlock(512, 1024)
        self.conv_block6 = _ConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = F.dropout(self.conv_block1(x, (2, 2)), p=0.2, training=self.training)
        x = F.dropout(self.conv_block2(x, (2, 2)), p=0.2, training=self.training)
        x = F.dropout(self.conv_block3(x, (2, 2)), p=0.2, training=self.training)
        x = F.dropout(self.conv_block4(x, (2, 2)), p=0.2, training=self.training)
        x = F.dropout(self.conv_block5(x, (2, 2)), p=0.2, training=self.training)
        x = F.dropout(self.conv_block6(x, (1, 1)), p=0.2, training=self.training)
        # (B, 2048, H', W') — global max+avg pooling over spatial dims
        x = x.amax(dim=(-2, -1)) + x.mean(dim=(-2, -1))  # (B, 2048)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return x  # (B, 2048)


# ── wav2vec2 wrapper ───────────────────────────────────────────────────────────

class _Wav2Vec2Wrapper(nn.Module):
    """Thin wrapper around HuggingFace Wav2Vec2Model.

    Input:  (B, 1, T) raw waveform (frontend is skipped for this model type)
    Output: (B, D) mean-pooled hidden states
    """

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T)
        x = x.squeeze(1)                          # (B, T)
        hidden = self.encoder(x).last_hidden_state  # (B, T', D)
        return hidden.mean(dim=1)                  # (B, D)


# ── factory ───────────────────────────────────────────────────────────────────

def build_audio_pretrained(cfg: DictConfig) -> tuple[nn.Module, int]:
    """Load audio pretrained backbone from config.

    Returns (module, feat_dim) — same interface as backbones.build_backbone.
    """
    model_type = cfg.model.get("type", "timm")

    if model_type == "panns":
        in_channels = cfg.model.get("in_chans", 1)
        model = CNN14(in_channels=in_channels)
        pretrained_path = cfg.model.get("pretrained_path", None)
        if pretrained_path and cfg.model.pretrained:
            ckpt = torch.load(pretrained_path, map_location="cpu")
            # Official PANNs checkpoints nest weights under "model"
            state_dict = ckpt.get("model", ckpt)
            # Drop classifier head keys if present
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc_audioset")}
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[pretrained_audio] {len(missing)} keys missing (expected for fc1 init)")
        return model, CNN14.FEAT_DIM

    elif model_type == "wav2vec2":
        try:
            from transformers import Wav2Vec2Model
        except ImportError:
            raise ImportError(
                "transformers package is required for wav2vec2. "
                "Install with: pip install transformers"
            )
        hf_name = cfg.model.get("hf_model", "facebook/wav2vec2-base")
        encoder = Wav2Vec2Model.from_pretrained(hf_name)
        feat_dim: int = encoder.config.hidden_size
        return _Wav2Vec2Wrapper(encoder), feat_dim

    else:
        raise ValueError(
            f"build_audio_pretrained: unknown model.type='{model_type}'. "
            "Expected 'panns' or 'wav2vec2'."
        )
