# Trainer: single-fold training loop with AMP, grad accumulation, and OOF saving.
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import time

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
        print(f"[Trainer fold={fold}] device={self.device}  loss={cfg.loss.type}  metric={cfg.metric.name}", flush=True)
        self.model.to(self.device)
        # channels_last: Ampere 텐서코어 conv 가속 (~24% speedup, 정확도·메모리 동일)
        self.channels_last = bool(cfg.train.get("channels_last", False)) and self.device.type == "cuda"
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        self.loss_fn = build_loss(cfg)
        self.optimizer = build_optimizer(model, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg, len(loaders["train"]))
        self.use_amp = cfg.train.amp and self.device.type == "cuda"
        print(f"[Trainer fold={fold}] amp={self.use_amp}  scheduler={cfg.optimizer.scheduler}  epochs={cfg.train.epochs}", flush=True)
        # torch.amp.GradScaler available in 2.3+; fall back to cuda.amp for 2.1/2.2
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except TypeError:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.metric_fn = get_metric_fn(cfg)
        self.tracker = BestTracker(cfg)
        # compute_on=cpu이면 DataLoader가 (B,C,F,T) raw feature를 넘김 → GPU에서 resize/채널확장
        self.gpu_resize = cfg.feature.compute_on == "cpu"
        self.image_size: list[int] = list(cfg.feature.image_size)
        self.n_channels = int(cfg.feature.n_channels)

    def _prep_gpu(self, x: torch.Tensor) -> torch.Tensor:
        """compute_on=cpu일 때 GPU에서 (B,C,F,T)→(B,n_channels,H,W) resize + 채널 확장."""
        if not self.gpu_resize:
            return x
        x = torch.nn.functional.interpolate(x, size=self.image_size, mode="bilinear", align_corners=False)
        if x.shape[1] == 1 and self.n_channels > 1:
            x = x.repeat(1, self.n_channels, 1, 1)  # (B,1,H,W)→(B,C,H,W), GPU상 복제
        if self.channels_last:
            x = x.to(memory_format=torch.channels_last)
        return x

    def train_one_epoch(self, epoch: int) -> dict:
        """One pass over the training set. Returns {'train_loss': float}."""
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()
        total_steps = len(self.loaders["train"])

        pbar = tqdm(self.loaders["train"], desc=f"epoch {epoch:02d} train", leave=False, dynamic_ncols=True)
        t_data_total = 0.0
        t_forward_total = 0.0
        t_batch_start = time.perf_counter()
        for step, (x, y) in enumerate(pbar):
            t_data = time.perf_counter() - t_batch_start
            t_data_total += t_data

            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            x = self._prep_gpu(x)

            t_fwd_start = time.perf_counter()
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(x)
                loss = self.loss_fn(logits, y) / self.cfg.train.grad_accum
            t_forward_total += time.perf_counter() - t_fwd_start

            if step == 0:
                print(f"  [timing step0] data_load={t_data:.3f}s  forward={time.perf_counter()-t_fwd_start:.3f}s", flush=True)

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
            avg_loss = total_loss / (step + 1)
            pbar.set_postfix(loss=f"{avg_loss:.4f}")
            t_batch_start = time.perf_counter()

        n = total_steps or 1
        print(f"  epoch {epoch:02d} train_loss={total_loss / total_steps:.4f}"
              f"  avg_data={t_data_total/n:.3f}s  avg_fwd={t_forward_total/n:.3f}s", flush=True)
        return {"train_loss": total_loss / total_steps}

    @torch.no_grad()
    def validate(self) -> dict:
        """One pass over the validation set. Returns metrics dict (+ '_preds' key)."""
        self.model.eval()
        all_logits, all_labels = [], []
        total_loss = 0.0

        for x, y in tqdm(self.loaders["valid"], desc="  validate", leave=False, dynamic_ncols=True):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            x = self._prep_gpu(x)
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
        print(f"  [validate] logits={logits.shape}  labels={labels.shape}  preds={preds.shape}", flush=True)
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

        use_early_stop = self.cfg.train.get("early_stopping", False)
        patience = int(self.cfg.train.get("patience", 5))
        no_improve = 0

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
                no_improve = 0
            else:
                no_improve += 1

            if self.cfg.optimizer.scheduler == "plateau" and self.scheduler is not None:
                self.scheduler.step(score)

            print(f"[fold={self.fold} epoch={epoch:02d}] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k not in ("epoch", "fold")), flush=True)

            if use_early_stop and no_improve >= patience:
                print(f"[fold={self.fold}] early stopping at epoch {epoch} (no improve for {patience} epochs)", flush=True)
                break

        return self.tracker.best_score
