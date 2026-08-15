import logging
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from crop_damage.models.utils import calc_batch_metrics, calc_epoch_metrics, move_to_device, save_checkpoint


class Trainer:
    def __init__(
        self,
        cfg: DictConfig,
        exp_dir: str | Path,
        ckpt_dir: str | Path,
        device: str | torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        encoder: nn.Module,
        change_fusion: nn.Module,
        decoder: nn.Module,
        criterion: nn.Module,
        optimizer,
        logger: logging.Logger | None = None,
        use_wandb: bool = False,
    ) -> None:
        self.cfg = cfg
        self.exp_dir = Path(exp_dir)
        self.ckpt_dir = Path(ckpt_dir)
        self.device = torch.device(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.encoder = encoder
        self.change_fusion = change_fusion
        self.decoder = decoder
        self.criterion = criterion
        self.optimizer = optimizer
        self.logger = logger or logging.getLogger(__name__)
        self.use_wandb = use_wandb
        self.writer = SummaryWriter(log_dir=str(self.exp_dir))

        self.model_cfg   = cfg.model
        self.encoder_cfg = cfg.encoder
        self.train_cfg   = cfg.train_loader
        self.val_cfg     = cfg.validation_loader
        self.trainer_cfg = cfg.trainer

        self.n_epochs = int(getattr(self.trainer_cfg, "n_epochs", self.model_cfg.num_epochs))

        self._best: dict = {"val_loss": float("inf"), "metrics": None, "epoch": None}

        # _last_val_metrics holds the most recent epoch's metrics dict so
        # _save_best_checkpoint can write it into _best without re-computing.
        self._last_val_metrics: dict[str, float] | None = None

        # Early stopping: consecutive-no-improvement counter. Patience is read from trainer.early_stopping_patience (default 3).
        self._early_stop_patience = int(getattr(self.trainer_cfg, "early_stopping_patience", 3))
        self._no_improve: int = 0

        encoder_name = str(getattr(self.encoder_cfg, "name", "Terramind")).strip().lower()
        self.encoder_mode = (
            self.encoder.train
            if (encoder_name == "unet" or bool(getattr(self.encoder_cfg, "finetune", False)))
            else self.encoder.eval
        )

        if self.use_wandb:
            import wandb
            self.wandb = wandb


    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> float:
        self.logger.info("Trainer started")
        self.logger.info("Output directory: %s", self.exp_dir)
        self.logger.info("Device: %s", self.device)
        self.logger.info("Model config: %s", OmegaConf.to_container(self.model_cfg, resolve=True))
        self.logger.info("Train loader config: %s", OmegaConf.to_container(self.train_cfg, resolve=True))
        self.logger.info("Validation loader config: %s", OmegaConf.to_container(self.val_cfg, resolve=True))
        self.logger.info("Criterion config: %s", OmegaConf.to_container(self.cfg.criterion, resolve=True))
        self.logger.info("Train patches: %d", len(self.train_loader.dataset))
        self.logger.info("Validation patches: %d", len(self.val_loader.dataset))
        self.logger.info("Train batches: %d", len(self.train_loader))
        self.logger.info("Validation batches: %d", len(self.val_loader))

        sched_cfg = getattr(self.trainer_cfg, "scheduler", None)
        sched_name = str(getattr(sched_cfg, "name", "none")) if sched_cfg is not None else "none"
        self.logger.info("Early stopping patience: %d epoch(s) | LR scheduler: %s", self._early_stop_patience, sched_name,)

        try:
            self.logger.info("Starting training for max %d epoch(s)", self.n_epochs)
            self._run_training(max_epochs=self.n_epochs)
            self.logger.info("Training completed successfully")
        except Exception:
            self.logger.exception("Trainer failed")
            raise
        finally:
            self.writer.close()
            self.logger.info("Closed TensorBoard writer")

        if self._best["metrics"] is None:
            raise RuntimeError("Trainer completed without recording best validation metrics.")
        return float(self._best["metrics"]["IoU"])


    def _run_training(self, max_epochs: int) -> int:
        """
        Run training for up to max_epochs epochs.

        Applies per-epoch early stopping (patience from trainer.early_stopping_patience)
        and steps a freshly-built LR scheduler after each validation pass.

        Returns the number of epochs actually completed (≤ max_epochs).
        """
        self._no_improve = 0
        scheduler = self._build_scheduler(max_epochs)

        for epoch in range(max_epochs):
            train_loss          = self._train_one_epoch(epoch, max_epochs)
            val_loss, val_metrics = self.validate()
            self._last_val_metrics = val_metrics

            self._log_epoch(epoch, max_epochs, train_loss, val_loss, val_metrics)

            improved = self._save_best_checkpoint(epoch, val_loss)
            if improved:
                self._no_improve = 0
            else:
                self._no_improve += 1

            # Step scheduler and log new LR
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("LR/learning_rate", current_lr, epoch)
                if self.use_wandb:
                    self.wandb.log({"train/learning_rate": current_lr, "val/epoch": epoch + 1})

            self._write_tensorboard(epoch, train_loss, val_loss, val_metrics)
            self._write_wandb(epoch, train_loss, val_loss, val_metrics)

            # Early stopping check
            if self._no_improve >= self._early_stop_patience:
                self.logger.info(
                    "Early stopping: val_loss did not improve for %d consecutive epoch(s). "
                    "Stopped at epoch %d/%d.",
                    self._early_stop_patience, epoch + 1, max_epochs,
                )
                return epoch + 1

        return max_epochs


    def _build_scheduler(self, max_epochs: int):
        """
        Build a fresh LR scheduler instance from trainer.scheduler config.
        Returns None if scheduler is not configured.
        """
        sched_cfg = getattr(self.trainer_cfg, "scheduler", None)
        if sched_cfg is None:
            return None

        name = str(getattr(sched_cfg, "name", "ReduceLROnPlateau")).strip()

        if name == "ReduceLROnPlateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode    = "min",
                factor  = float(getattr(sched_cfg, "factor",  0.5)),
                patience= int(getattr(sched_cfg,   "patience", 3)),
                min_lr  = float(getattr(sched_cfg, "min_lr",  1e-6)),
            )

        if name == "CosineAnnealingLR":
            # T_max defaults to the run's total epoch budget so the cosine
            # cycle completes within training. Override via trainer.scheduler.T_max.
            t_max = int(getattr(sched_cfg, "T_max", max_epochs))
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max   = t_max,
                eta_min = float(getattr(sched_cfg, "min_lr", 1e-6)),
            )

        self.logger.warning(
            "Unknown scheduler '%s' in trainer.scheduler.name — no scheduler will be used.", name
        )
        return None


    # ------------------------------------------------------------------
    # Core training / validation
    # ------------------------------------------------------------------

    def validate(self) -> tuple[float, dict[str, float]]:
        self.encoder.eval()
        self.change_fusion.eval()
        self.decoder.eval()

        running_val_loss = 0.0
        true_positive = false_positive = false_negative = true_negative = 0.0

        with torch.no_grad():
            for inputs, target in self.val_loader:
                inputs = move_to_device(inputs, self.device)
                target = target.to(self.device)

                logits     = self._forward(inputs)
                batch_loss = self.criterion(logits, target)
                batch_size = next(iter(inputs["before"].values())).size(0)
                running_val_loss += batch_loss.item() * batch_size

                batch_metrics = calc_batch_metrics(
                    logits,
                    target,
                    ignore_index   = self.model_cfg.ignore_index,
                    positive_class = self.model_cfg.positive_class,
                    negative_class = self.model_cfg.negative_class,
                )
                true_positive  += batch_metrics[0]
                false_positive += batch_metrics[1]
                false_negative += batch_metrics[2]
                true_negative  += batch_metrics[3]

        n_val = len(self.val_loader.dataset)
        if n_val == 0:
            raise RuntimeError(
                "Validation dataset is empty. Check val directory and patch_size/stride settings."
            )
        val_loss = running_val_loss / n_val
        metrics  = calc_epoch_metrics(true_positive, false_positive, false_negative, true_negative)
        return val_loss, metrics


    def _train_one_epoch(self, epoch: int, total_epochs: int) -> float:
        self.encoder_mode()
        self.change_fusion.train()
        self.decoder.train()
        running_train_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, (inputs, target) in enumerate(self.train_loader, start=1):
            inputs = move_to_device(inputs, self.device)
            target = target.to(self.device)

            logits     = self._forward(inputs)
            train_loss = self.criterion(logits, target)
            batch_size = next(iter(inputs["before"].values())).size(0)
            running_train_loss += train_loss.item() * batch_size

            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()

            if batch_idx % getattr(self.trainer_cfg, "log_interval", 1) == 0:
                self.logger.info(
                    "Epoch %d/%d | batch %d/%d | train_loss=%.4f",
                    epoch + 1, total_epochs,
                    batch_idx, num_batches, train_loss.item(),
                )

            if self.use_wandb:
                global_step = epoch * num_batches + batch_idx
                self.wandb.log({
                    "train/global_step": global_step,
                    "train/batch_loss":  train_loss.item(),
                    "train/epoch":       epoch + 1,
                })

        n_train = len(self.train_loader.dataset)
        if n_train == 0:
            raise RuntimeError(
                "Training dataset is empty. Check train directory and patch_size/stride settings."
            )
        return running_train_loss / n_train


    def _forward(self, inputs: dict) -> torch.Tensor:
        z_before       = self.encoder(inputs["before"])
        z_after        = self.encoder(inputs["after"])
        fused_features = self.change_fusion(z_before, z_after)
        return self.decoder(fused_features)


    def _log_epoch(
        self, epoch: int, total_epochs: int,
        train_loss: float, val_loss: float, metrics: dict[str, float],
    ) -> None:
        patience_left = self._early_stop_patience - self._no_improve
        self.logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | "
            "IoU=%.4f | Acc=%.4f | Prec=%.4f | Recall=%.4f | F1=%.4f | "
            "lr=%.2e | patience=%d/%d",
            epoch + 1, total_epochs,
            train_loss, val_loss,
            metrics["IoU"], metrics["Accuracy"],
            metrics["Precision"], metrics["Recall"], metrics["F1"],
            self.optimizer.param_groups[0]["lr"],
            self._no_improve, self._early_stop_patience,
        )


    def _save_best_checkpoint(self, epoch: int, val_loss: float) -> bool:
        """Save checkpoint if val_loss improves. Returns True if improved."""
        if val_loss >= self._best["val_loss"]:
            return False

        self._best["val_loss"] = val_loss
        self._best["epoch"]    = epoch
        self._best["metrics"]  = self._last_val_metrics

        save_checkpoint(
            self.encoder, self.change_fusion, self.decoder, self.optimizer,
            epoch, val_loss, self.cfg,
            save_dir=str(self.ckpt_dir),
            prefix="best",
        )
        self.logger.info(
            "New best checkpoint at epoch %d | val_loss=%.4f | IoU=%.4f",
            epoch + 1, val_loss,
            (self._last_val_metrics or {}).get("IoU", float("nan")),
        )
        return True


    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _write_tensorboard(
        self, epoch: int, train_loss: float,
        val_loss: float, metrics: dict[str, float],
    ) -> None:
        self.writer.add_scalar("Loss/train",        train_loss,           epoch)
        self.writer.add_scalar("Loss/validation",   val_loss,             epoch)
        self.writer.add_scalar("Metrics/IoU",       metrics["IoU"],       epoch)
        self.writer.add_scalar("Metrics/Accuracy",  metrics["Accuracy"],  epoch)
        self.writer.add_scalar("Metrics/Precision", metrics["Precision"], epoch)
        self.writer.add_scalar("Metrics/Recall",    metrics["Recall"],    epoch)
        self.writer.add_scalar("Metrics/F1",        metrics["F1"],        epoch)
        self.writer.add_scalar("LR/learning_rate",  self.optimizer.param_groups[0]["lr"], epoch)


    def _write_wandb(
        self, epoch: int, train_loss: float,
        val_loss: float, metrics: dict[str, float],
    ) -> None:
        if not self.use_wandb:
            return

        payload = {
            "val/epoch":            epoch + 1,
            "train/epoch_loss":     train_loss,
            "val/loss":             val_loss,
            "val/IoU":              metrics["IoU"],
            "val/Accuracy":         metrics["Accuracy"],
            "val/Precision":        metrics["Precision"],
            "val/Recall":           metrics["Recall"],
            "val/F1":               metrics["F1"],
            "best/val_loss":        self._best["val_loss"],
            "train/learning_rate":  self.optimizer.param_groups[0]["lr"],
        }

        best_metrics = self._best["metrics"]
        if best_metrics is not None:
            payload["best/IoU"]       = best_metrics["IoU"]
            payload["best/Accuracy"]  = best_metrics["Accuracy"]
            payload["best/Precision"] = best_metrics["Precision"]
            payload["best/Recall"]    = best_metrics["Recall"]
            payload["best/F1"]        = best_metrics["F1"]

        self.wandb.log(payload)
