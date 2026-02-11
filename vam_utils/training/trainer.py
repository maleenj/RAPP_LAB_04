"""VAMTrainer: training loop with logging, checkpointing, and early stopping."""

import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from vam_utils.config.model_config import ModelConfig, TrainingConfig
from vam_utils.data.normalization import NormalizationStats, normalize_joint_limits
from vam_utils.training.losses import vam_loss


class VAMTrainer:
    """Handles the full training loop for the Action Chunking Transformer.

    Features:
        - AdamW optimizer with linear warmup + cosine annealing
        - Composite loss (prediction + smoothness + acceleration)
        - Gradient clipping
        - TensorBoard logging
        - Best-val-loss checkpointing + periodic saves
        - Early stopping
        - Resume from checkpoint

    Usage:
        trainer = VAMTrainer(model, dataloaders, model_config, train_config, device)
        history = trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        dataloaders: Dict[str, DataLoader],
        model_config: ModelConfig,
        train_config: TrainingConfig,
        device: torch.device,
        experiment_name: str = "vam",
        norm_stats: Optional[NormalizationStats] = None,
    ):
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.model_config = model_config
        self.train_config = train_config
        self.device = device
        self.experiment_name = experiment_name

        # Precompute normalized joint limits for the loss penalty
        if norm_stats is not None and train_config.joint_limit_weight > 0:
            self.joint_limits_normed = normalize_joint_limits(norm_stats).to(device)
        else:
            self.joint_limits_normed = None

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )

        # Scheduler: linear warmup then cosine decay
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=train_config.warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=train_config.epochs - train_config.warmup_epochs,
            eta_min=1e-6,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[train_config.warmup_epochs],
        )

        # TensorBoard
        log_path = train_config.log_dir / experiment_name
        log_path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_path))

        # Checkpoint directory
        self.ckpt_dir = train_config.checkpoint_dir / experiment_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history: Dict[str, List[float]] = {
            "train_total": [], "train_prediction": [],
            "train_smoothness": [], "train_acceleration": [],
            "train_joint_limit": [],
            "val_total": [], "val_prediction": [],
            "val_smoothness": [], "val_acceleration": [],
            "val_joint_limit": [],
            "lr": [],
        }

    def train(self) -> Dict[str, List[float]]:
        """Run the full training loop. Returns history dict."""
        print(f"Training on {self.device} for {self.train_config.epochs} epochs")
        print(f"  Train batches: {len(self.dataloaders['train'])}")
        print(f"  Val batches:   {len(self.dataloaders['val'])}")
        print(f"  Checkpoints:   {self.ckpt_dir}")
        print(f"  TensorBoard:   {self.writer.log_dir}")
        print()

        for epoch in range(self.start_epoch, self.train_config.epochs):
            t0 = time.time()

            # Train
            train_losses = self._train_epoch()

            # Validate
            val_losses = self._validate_epoch()

            # Get current LR
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Step scheduler
            self.scheduler.step()

            # Record history
            for key in ["total", "prediction", "smoothness", "acceleration", "joint_limit"]:
                self.history[f"train_{key}"].append(train_losses[key])
                self.history[f"val_{key}"].append(val_losses[key])
            self.history["lr"].append(current_lr)

            # TensorBoard logging
            self._log_epoch(epoch, train_losses, val_losses, current_lr)

            # Checkpointing
            is_best = val_losses["total"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses["total"]
                self.patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1

            if (epoch + 1) % self.train_config.save_every_n_epochs == 0:
                self._save_checkpoint(epoch, is_best=False)

            # Print epoch summary
            elapsed = time.time() - t0
            best_marker = " *" if is_best else ""
            print(
                f"Epoch {epoch+1:3d}/{self.train_config.epochs} | "
                f"train {train_losses['total']:.5f} | "
                f"val {val_losses['total']:.5f}{best_marker} | "
                f"lr {current_lr:.2e} | "
                f"{elapsed:.1f}s"
            )

            # Early stopping
            if self.patience_counter >= self.train_config.early_stopping_patience:
                print(
                    f"\nEarly stopping at epoch {epoch+1} "
                    f"(no improvement for {self.train_config.early_stopping_patience} epochs)"
                )
                break

        self.writer.close()
        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.6f}")
        print(f"Best model saved to: {self.ckpt_dir / 'best.pt'}")
        return self.history

    def _train_epoch(self) -> Dict[str, float]:
        """Run one training epoch. Returns average losses."""
        self.model.train()
        running = {"total": 0.0, "prediction": 0.0,
                   "smoothness": 0.0, "acceleration": 0.0, "joint_limit": 0.0}
        n_batches = 0

        for batch in self.dataloaders["train"]:
            inputs = batch["input"].to(self.device)    # [B, T_in, 54]
            targets = batch["target"].to(self.device)  # [B, T_out, 6]

            # Forward
            predictions = self.model(inputs)           # [B, T_out, 6]

            # Loss
            losses = vam_loss(
                predictions, targets, self.train_config, self.joint_limits_normed
            )

            # Backward
            self.optimizer.zero_grad()
            losses["total"].backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.train_config.grad_clip_norm
            )

            self.optimizer.step()

            # Accumulate
            for key in running:
                if key in losses:
                    running[key] += losses[key].item()
            n_batches += 1

        return {k: v / n_batches for k, v in running.items()}

    @torch.no_grad()
    def _validate_epoch(self) -> Dict[str, float]:
        """Run one validation epoch. Returns average losses."""
        self.model.eval()
        running = {"total": 0.0, "prediction": 0.0,
                   "smoothness": 0.0, "acceleration": 0.0, "joint_limit": 0.0}
        n_batches = 0

        for batch in self.dataloaders["val"]:
            inputs = batch["input"].to(self.device)
            targets = batch["target"].to(self.device)

            predictions = self.model(inputs)
            losses = vam_loss(
                predictions, targets, self.train_config, self.joint_limits_normed
            )

            for key in running:
                if key in losses:
                    running[key] += losses[key].item()
            n_batches += 1

        return {k: v / n_batches for k, v in running.items()}

    def _log_epoch(
        self,
        epoch: int,
        train_losses: Dict[str, float],
        val_losses: Dict[str, float],
        lr: float,
    ):
        """Write epoch metrics to TensorBoard."""
        for key in ["total", "prediction", "smoothness", "acceleration", "joint_limit"]:
            self.writer.add_scalars(
                f"loss/{key}",
                {"train": train_losses[key], "val": val_losses[key]},
                epoch,
            )
        self.writer.add_scalar("lr", lr, epoch)

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "model_config": self.model_config.__dict__,
            "train_config": {
                k: str(v) if isinstance(v, Path) else v
                for k, v in self.train_config.__dict__.items()
            },
        }

        if is_best:
            torch.save(state, self.ckpt_dir / "best.pt")
        else:
            torch.save(state, self.ckpt_dir / f"epoch_{epoch+1:04d}.pt")

    def load_checkpoint(self, path: str | Path):
        """Resume training from a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint.get("history", self.history)
        print(f"Resumed from epoch {self.start_epoch}, best val loss: {self.best_val_loss:.6f}")
