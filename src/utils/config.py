# Config loading: default.yaml deep-merge with experiment yaml + CLI overrides.
from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

_DEFAULT_YAML = Path(__file__).parents[2] / "configs" / "default.yaml"


def load_config(path: str, overrides: list[str] | None = None) -> DictConfig:
    """Load default.yaml, deep-merge experiment yaml, then apply CLI overrides."""
    base = OmegaConf.load(_DEFAULT_YAML)
    exp = OmegaConf.load(path)
    cfg = OmegaConf.merge(base, exp)
    if overrides:
        cli = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli)
    if cfg.get("debug", False):
        cfg.train.epochs = 1
        cfg.train.batch_size = 4
        cfg.train.num_workers = 0  # Windows multiprocessing 안전
        cfg.wandb.mode = "disabled"
    return cfg


def save_config(cfg: DictConfig, out_dir: Path) -> None:
    """Dump runtime config to outputs/<exp>/config.yaml for reproducibility."""
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")
