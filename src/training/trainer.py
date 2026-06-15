# Trainer: single-fold training loop with AMP, grad accumulation, and OOF saving.
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.losses import build_loss
from src.training.optimizers import build_optimizer, build_scheduler
from src.utils.checkpoint import BestTracker
from src.utils.logger import Logger
from src.utils.metrics import get_metric_fn


class Trainer:
    """Handles one fold: train_one_epoch + validate + fit loop.

    K-Fold orchestration lives in main.py; this class owns epoch-level logic.
    """

    def __init__(
        self,
        model: nn.Module,
        loaders: dict[str, DataLoader],
        cfg: DictConfig,
        logger: Logger,
        fold: int,
    ) -> None:
        self.model = model
        self.loaders = loaders
        self.cfg = cfg
        self.logger = logger
        self.fold = fold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.loss_fn = build_loss(cfg)
        self.optimizer = build_optimizer(model, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg, len(loaders["train"]))
        self.use_amp = cfg.train.amp and self.device.type == "cuda"
        # torch.amp.GradScaler available in 2.3+; fall back to cuda.amp for 2.1/2.2
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except TypeError:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.metric_fn = get_metric_fn(cfg)
        self.tracker = BestTracker(cfg)

    def train_one_epoch(self, epoch: int) -> dict:
        """One pass over the training set. Returns {'train_loss': float}."""
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        pbar = tqdm(self.loaders["train"], desc=f"[fold={self.fold} epoch={epoch:02d}] train", leave=False)
        for step, (x, y) in enumerate(pbar):
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(x)
                loss = self.loss_fn(logits, y) / self.cfg.train.grad_accum

            self.scaler.scale(loss).backward()

            if (step + 1) % self.cfg.train.grad_accum == 0:
                if self.cfg.train.clip_grad:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                if self.scheduler is not None and self.cfg.optimizer.scheduler != "plateau":
                    self.scheduler.step()

            total_loss += loss.item() * self.cfg.train.grad_accum
            pbar.set_postfix(loss=f"{loss.item() * self.cfg.train.grad_accum:.4f}")

        return {"train_loss": total_loss / len(self.loaders["train"])}

    @torch.no_grad()
    def validate(self) -> dict:
        """One pass over the validation set. Returns metrics dict (+ '_preds' key)."""
        self.model.eval()
        all_logits, all_labels = [], []
        total_loss = 0.0

        for x, y in tqdm(self.loaders["valid"], desc=f"[fold={self.fold}] valid", leave=False):
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(x)
                loss = self.loss_fn(logits, y)
            total_loss += loss.item()
            all_logits.append(logits.cpu().float().numpy())
            all_labels.append(y.cpu().float().numpy())

        logits = np.concatenate(all_logits, axis=0)   # (N, C) raw logits
        labels = np.concatenate(all_labels, axis=0)  # (N, C) or (N,) for single-label
        if self.cfg.data.multilabel:
            preds = 1.0 / (1.0 + np.exp(-logits))   # sigmoid
        else:
            e = np.exp(logits - logits.max(axis=-1, keepdims=True))
            preds = e / e.sum(axis=-1, keepdims=True)  # softmax
        score = self.metric_fn(labels, preds)
        monitor = self.cfg.metric.monitor
        return {
            "val_loss": total_loss / len(self.loaders["valid"]),
            monitor: score,
            "_preds": preds,
        }

    def fit(self) -> float:
        """Full training loop. Returns best validation score."""
        out_dir = Path("outputs") / self.cfg.exp_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.cfg.train.epochs):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate()
            preds = val_metrics.pop("_preds")
            metrics = {**train_metrics, **val_metrics, "epoch": epoch, "fold": self.fold}
            self.logger.log(metrics, step=epoch)

            monitor = self.cfg.metric.monitor
            score = val_metrics[monitor]
            improved = self.tracker.update(
                score, self.model, self.optimizer, epoch,
                out_dir / f"best_fold{self.fold}.pth",
            )
            if improved:
                np.save(out_dir / f"oof_fold{self.fold}.npy", preds)

            if self.cfg.optimizer.scheduler == "plateau" and self.scheduler is not None:
                self.scheduler.step(score)

            print(f"[fold={self.fold} epoch={epoch:02d}] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k not in ("epoch", "fold")))

        return self.tracker.best_score
