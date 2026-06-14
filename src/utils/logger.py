# wandb wrapper that becomes no-op when mode='disabled'.
from __future__ import annotations

import numpy as np
from omegaconf import DictConfig, OmegaConf


class Logger:
    """wandb wrapper. All methods are no-op when cfg.wandb.mode == 'disabled'."""

    def __init__(self, cfg: DictConfig) -> None:
        self._enabled = cfg.wandb.mode != "disabled"
        if self._enabled:
            import wandb

            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.get("entity"),
                name=cfg.exp_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                mode=cfg.wandb.mode,
                tags=list(cfg.wandb.tags),
            )
            self._wandb = wandb
        else:
            self._wandb = None

    def log(self, metrics: dict, step: int | None = None) -> None:
        """Log scalar metrics."""
        if self._enabled:
            self._wandb.log(metrics, step=step)

    def log_audio(self, name: str, waveform: np.ndarray, sr: int) -> None:
        """Log an audio sample for debugging."""
        if self._enabled:
            self._wandb.log({name: self._wandb.Audio(waveform, sample_rate=sr)})

    def watch(self, model) -> None:
        """Watch model gradients/parameters."""
        if self._enabled:
            self._wandb.watch(model)

    def finish(self) -> None:
        """Finalize wandb run."""
        if self._enabled:
            self._wandb.finish()
