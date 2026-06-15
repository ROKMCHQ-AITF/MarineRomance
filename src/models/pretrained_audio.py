# Audio pretrained model wrappers: AST (HuggingFace) and BEATs (Microsoft local).
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


# ── AST (Audio Spectrogram Transformer) ───────────────────────────────────────

class _ASTWrapper(nn.Module):
    """AST wrapper: accepts (B, C, H, W) spectrogram from Frontend.

    Collapses channel dim by mean, transposes to (B, T, n_mels) for ASTModel.
    Output: (B, hidden_size)
    """

    def __init__(self, encoder: nn.Module, feat_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) — H=n_mels, W=T_frames
        if x.dim() == 4:
            x = x.mean(dim=1)    # (B, n_mels, T)
        # Resize to AST's expected (num_mel_bins, max_length) if needed.
        # Pretrained positional embeddings are fixed, so shape must match.
        t_exp = self.encoder.config.max_length
        f_exp = self.encoder.config.num_mel_bins
        if x.shape[1] != f_exp or x.shape[2] != t_exp:
            x = F.interpolate(
                x.unsqueeze(1), size=(f_exp, t_exp), mode="bilinear", align_corners=False
            ).squeeze(1)         # (B, f_exp, t_exp)
        x = x.transpose(1, 2)   # (B, T, n_mels)
        out = self.encoder(input_values=x)
        return out.pooler_output  # (B, hidden_size)


# ── BEATs (Microsoft unilm) ───────────────────────────────────────────────────

class _BEATsWrapper(nn.Module):
    """BEATs wrapper: accepts raw waveform (B, 1, T), skips Frontend.

    Requires BEATs.py from Microsoft's unilm repo placed anywhere on PYTHONPATH
    (e.g., project root or src/). Download checkpoint from:
      https://github.com/microsoft/unilm/tree/master/beats
    and set cfg.model.pretrained_path to the .pt file path.

    Output: (B, hidden_size)
    """

    def __init__(self, encoder: nn.Module, feat_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T) raw waveform at 16 kHz
        x = x.squeeze(1)  # (B, T)
        # extract_features returns (features, padding_mask); features: (B, T', D)
        features, _ = self.encoder.extract_features(x)
        return features.mean(dim=1)  # (B, hidden_size)


# ── factory ───────────────────────────────────────────────────────────────────

def build_audio_pretrained(cfg: DictConfig) -> tuple[nn.Module, int]:
    """Load audio pretrained backbone from config.

    Returns (module, feat_dim) — same interface as backbones.build_backbone.
    """
    model_type = cfg.model.get("type", "timm")
    hf_model = cfg.model.get("hf_model", "") or ""
    use_pretrained: bool = cfg.model.get("pretrained", True)

    if model_type == "ast":
        try:
            from transformers import ASTConfig, ASTModel
        except ImportError:
            raise ImportError(
                "transformers package is required for AST. "
                "Install with: pip install transformers"
            )
        if not hf_model:
            hf_model = "MIT/ast-finetuned-audioset-10-10-0.4593"
        if use_pretrained:
            encoder = ASTModel.from_pretrained(hf_model)
        else:
            encoder = ASTModel(ASTConfig())
        feat_dim: int = encoder.config.hidden_size
        return _ASTWrapper(encoder, feat_dim), feat_dim

    elif model_type == "beats":
        try:
            from beats.BEATs import BEATs, BEATsConfig as MsBEATsConfig
        except ImportError:
            raise ImportError(
                "beats/ package not found. Expected at project root: "
                "beats/BEATs.py, beats/Tokenizers.py, beats/backbone.py, beats/modules.py"
            )
        pretrained_path = cfg.model.get("pretrained_path", None)
        if use_pretrained:
            if not pretrained_path:
                raise ValueError(
                    "BEATs requires cfg.model.pretrained_path. "
                    "Default: input/pretrained/BEATs_iter3_plus_AS2M.pt"
                )
            ckpt = torch.load(pretrained_path, map_location="cpu")
            beats_cfg = MsBEATsConfig(ckpt["cfg"])
            encoder = BEATs(beats_cfg)
            encoder.load_state_dict(ckpt["model"])
        else:
            beats_cfg = MsBEATsConfig()
            encoder = BEATs(beats_cfg)

        feat_dim = getattr(beats_cfg, "encoder_embed_dim", 768)
        return _BEATsWrapper(encoder, feat_dim), feat_dim

    else:
        raise ValueError(
            f"build_audio_pretrained: unknown model.type='{model_type}'. "
            "Expected 'ast' or 'beats'."
        )
