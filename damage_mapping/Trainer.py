import logging
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from damage_mapping.models.utils import calc_batch_metrics, calc_epoch_metrics, move_to_device, save_checkpoint


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
        curriculum_manager=None,
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
        self.curriculum_manager = curriculum_manager
        self.writer = SummaryWriter(log_dir=str(self.exp_dir))

        self.model_cfg   = cfg.model
        self.encoder_cfg = cfg.encoder
        self.train_cfg   = cfg.train_loader
        self.val_cfg     = cfg.validation_loader
        self.trainer_cfg = cfg.trainer

        self.n_epochs = int(getattr(self.trainer_cfg, "n_epochs", self.model_cfg.num_epochs))

        # Stage tracking: "flood" during Stage 1, "conflict" in Stage 2 / no-curriculum
        self.current_stage: str = "flood" if curriculum_manager is not None else "conflict"

        # Independent best-checkpoint tracking per stage.
        # Stage 1 ("flood") and Stage 2 ("conflict") losses live on different distributions and must NOT be compared against each other.
        self._stage_best: dict[str, dict] = {
            "flood":    {"val_loss": float("inf"), "metrics": None, "epoch": None},
            "conflict": {"val_loss": float("inf"), "metrics": None, "epoch": None},
        }

        # _last_val_metrics holds the most recent epoch's metrics dict so
        # _save_best_checkpoint can write it into _stage_best without re-computing.
        self._last_val_metrics: dict[str, float] | None = None

        # Early stopping: per-stage consecutive-no-improvement counters. Patience is read from trainer.early_stopping_patience (default 3).
        self._early_stop_patience = int(getattr(self.trainer_cfg, "early_stopping_patience", 3))
        self._stage_no_improve: dict[str, int] = {"flood": 0, "conflict": 0}

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
        self.logger.info("Train loader config (conflict): %s", OmegaConf.to_container(self.train_cfg, resolve=True))
        self.logger.info("Validation loader config (conflict): %s", OmegaConf.to_container(self.val_cfg, resolve=True))
        self.logger.info("Criterion config: %s", OmegaConf.to_container(self.cfg.criterion, resolve=True))
        self.logger.info("Train patches: %d", len(self.train_loader.dataset))
        self.logger.info("Validation patches: %d", len(self.val_loader.dataset))
        self.logger.info("Train batches: %d", len(self.train_loader))
        self.logger.info("Validation batches: %d", len(self.val_loader))

        sched_cfg = getattr(self.trainer_cfg, "scheduler", None)
        sched_name = str(getattr(sched_cfg, "name", "none")) if sched_cfg is not None else "none"
        self.logger.info("Early stopping patience: %d epoch(s) per stage | LR scheduler: %s", self._early_stop_patience, sched_name,)

        try:
            if self.curriculum_manager is not None:
                self._train_curriculum()
            else:
                self.logger.info("Starting training (no curriculum) for max %d epoch(s)", self.n_epochs)
                self.current_stage = "conflict"
                self._run_stage("conflict", global_offset=0, max_epochs=self.n_epochs)
            self.logger.info("Training completed successfully")
        except Exception:
            self.logger.exception("Trainer failed")
            raise
        finally:
            self.writer.close()
            self.logger.info("Closed TensorBoard writer")

        # Return best conflict-stage IoU (the stage used by the Evaluator).
        # Fall back to flood if conflict never ran (e.g. conflict_epochs=0).
        final_metrics = (
            self._stage_best["conflict"]["metrics"]
            or self._stage_best["flood"]["metrics"]
        )
        if final_metrics is None:
            raise RuntimeError("Trainer completed without recording best validation metrics.")
        return float(final_metrics["IoU"])


    def _train_curriculum(self) -> None:
        """Orchestrate two-stage curriculum: flood → conflict."""
        cm = self.curriculum_manager

        self.logger.info(
            "Curriculum: flood stage max=%d ep (lr=%.2e) → conflict stage max=%d ep (lr=%.2e)",
            cm.flood_epochs, self.optimizer.param_groups[0]["lr"],
            cm.conflict_epochs, cm.stage2_lr,
        )
        self.logger.info(
            "Flood   train=%d patches | val=%s patches",
            len(cm.flood_loader.dataset),
            len(cm.flood_val_loader.dataset) if cm.flood_val_loader is not None else "N/A",
        )
        self.logger.info(
            "Conflict train=%d patches | val=%s patches",
            len(cm.conflict_loader.dataset),
            len(cm.conflict_val_loader.dataset) if cm.conflict_val_loader is not None else "N/A",
        )

        # --- Stage 1: Flood ---
        self.current_stage = "flood"
        self.train_loader  = cm.flood_loader
        if cm.flood_val_loader is not None:
            self.val_loader = cm.flood_val_loader
        else:
            self.logger.warning(
                "flood_validation_loader not configured: Stage 1 validation metrics "
                "will be computed on the conflict validation set."
            )

        flood_run = self._run_stage("flood", global_offset=0, max_epochs=cm.flood_epochs)

        # --- Stage switch ---
        self._do_stage_switch(flood_run)

        # --- Stage 2: Conflict ---
        self._run_stage("conflict", global_offset=flood_run, max_epochs=cm.conflict_epochs)


    def _run_stage(self, stage: str, global_offset: int, max_epochs: int) -> int:
        """
        Run one training stage for up to max_epochs epochs.

        Applies per-epoch early stopping (patience from trainer.early_stopping_patience)
        and steps a freshly-built LR scheduler after each validation pass.

        Returns the number of epochs actually completed (≤ max_epochs).
        """
        self._stage_no_improve[stage] = 0
        scheduler = self._build_scheduler(max_epochs)

        for local_ep in range(max_epochs):
            global_ep   = global_offset + local_ep
            total_shown = global_offset + max_epochs  # denominator for log line

            train_loss          = self._train_one_epoch(global_ep, total_shown)
            val_loss, val_metrics = self.validate()
            self._last_val_metrics = val_metrics

            self._log_epoch(global_ep, total_shown, train_loss, val_loss, val_metrics)

            improved = self._save_best_checkpoint(global_ep, val_loss)
            if improved:
                self._stage_no_improve[stage] = 0
            else:
                self._stage_no_improve[stage] += 1

            # Step scheduler and log new LR
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("LR/learning_rate", current_lr, global_ep)
                if self.use_wandb:
                    self.wandb.log({"train/learning_rate": current_lr, "val/epoch": global_ep + 1})

            self._write_tensorboard(global_ep, train_loss, val_loss, val_metrics)
            self._write_wandb(global_ep, train_loss, val_loss, val_metrics)

            # Early stopping check
            if self._stage_no_improve[stage] >= self._early_stop_patience:
                self.logger.info(
                    "[%s] Early stopping: val_loss did not improve for %d consecutive epoch(s). "
                    "Stopped at epoch %d/%d (global %d).",
                    stage.upper(), self._early_stop_patience,
                    local_ep + 1, max_epochs, global_ep + 1,
                )
                return local_ep + 1

        return max_epochs


    def _do_stage_switch(self, flood_epochs_run: int) -> None:
        """Apply the flood → conflict transition: swap loaders, reset LR to stage2_lr, log."""
        cm = self.curriculum_manager

        self.current_stage = "conflict"
        self.train_loader  = cm.conflict_loader

        if cm.conflict_val_loader is not None:
            self.val_loader = cm.conflict_val_loader
            self.logger.info(
                "Val loader swapped to conflict validation set (%d patches)",
                len(self.val_loader.dataset),
            )
        else:
            self.logger.info(
                "Val loader unchanged (no conflict_val_loader configured; %d patches)",
                len(self.val_loader.dataset),
            )

        for pg in self.optimizer.param_groups:
            pg["lr"] = cm.stage2_lr

        if getattr(getattr(self.cfg, "criterion", None), "apply_weight_loss", False):
            self.logger.warning(
                "Curriculum stage switch: criterion class weights were computed from "
                "the flood dataset and will remain in effect for Stage 2 (conflict). "
                "If class distributions differ between stages, consider setting "
                "criterion.apply_weight_loss: false or recomputing weights manually."
            )

        flood_best = self._stage_best["flood"]
        self.logger.info(
            "Curriculum: flood → conflict after %d epoch(s) | lr → %.2e | "
            "flood best: epoch=%s val_loss=%.4f IoU=%.4f",
            flood_epochs_run,
            cm.stage2_lr,
            flood_best["epoch"],
            flood_best["val_loss"],
            (flood_best["metrics"] or {}).get("IoU", float("nan")),
        )


    def _build_scheduler(self, max_epochs: int):
        """
        Build a fresh LR scheduler instance from trainer.scheduler config.
        Returns None if scheduler is not configured.
        A fresh instance is built per stage so each stage starts with a clean state.
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
            # T_max defaults to the stage's epoch budget so the cosine cycle
            # completes within the stage. Override via trainer.scheduler.T_max.
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
                f"[{self.current_stage}] Validation dataset is empty. "
                "Check val directory and patch_size/stride settings."
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
        stage_tag   = f"[{self.current_stage.upper()}] " if self.curriculum_manager is not None else ""

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
                    "%sEpoch %d/%d | batch %d/%d | train_loss=%.4f",
                    stage_tag, epoch + 1, total_epochs,
                    batch_idx, num_batches, train_loss.item(),
                )

            if self.use_wandb:
                global_step = epoch * num_batches + batch_idx
                stage = self.current_stage
                self.wandb.log({
                    "train/global_step":         global_step,
                    f"train/{stage}/batch_loss": train_loss.item(),
                    "train/epoch":               epoch + 1,
                    "train/stage":               stage,
                })

        n_train = len(self.train_loader.dataset)
        if n_train == 0:
            raise RuntimeError(
                f"[{self.current_stage}] Training dataset is empty. "
                "Check train directory and patch_size/stride settings."
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
        stage_tag = f"[{self.current_stage.upper()}] " if self.curriculum_manager is not None else ""
        patience_left = self._early_stop_patience - self._stage_no_improve[self.current_stage]
        self.logger.info(
            "%sEpoch %d/%d | train_loss=%.4f | val_loss=%.4f | "
            "IoU=%.4f | Acc=%.4f | Prec=%.4f | Recall=%.4f | F1=%.4f | "
            "lr=%.2e | patience=%d/%d",
            stage_tag, epoch + 1, total_epochs,
            train_loss, val_loss,
            metrics["IoU"], metrics["Accuracy"],
            metrics["Precision"], metrics["Recall"], metrics["F1"],
            self.optimizer.param_groups[0]["lr"],
            self._stage_no_improve[self.current_stage], self._early_stop_patience,
        )


    def _save_best_checkpoint(self, epoch: int, val_loss: float) -> bool:
        """Save checkpoint if val_loss improves for the current stage. Returns True if improved."""
        stage     = self.current_stage
        stage_rec = self._stage_best[stage]

        if val_loss >= stage_rec["val_loss"]:
            return False

        stage_rec["val_loss"] = val_loss
        stage_rec["epoch"]    = epoch
        stage_rec["metrics"]  = self._last_val_metrics

        # Distinct filename prefix per stage keeps Stage-1 and Stage-2
        # checkpoints from overwriting each other.
        ckpt_prefix = "best_flood" if stage == "flood" else "best"
        save_checkpoint(
            self.encoder, self.change_fusion, self.decoder, self.optimizer,
            epoch, val_loss, self.cfg,
            save_dir=str(self.ckpt_dir),
            prefix=ckpt_prefix,
        )
        self.logger.info(
            "[%s] New best checkpoint at epoch %d | val_loss=%.4f | IoU=%.4f",
            stage.upper(), epoch + 1, val_loss,
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
        # Stage-prefixed tags when curriculum is active so flood and conflict
        # curves appear as separate series in TensorBoard.
        if self.curriculum_manager is not None:
            s   = self.current_stage
            tag = lambda base: f"{s}/{base}"
        else:
            tag = lambda base: base

        self.writer.add_scalar(tag("Loss/train"),        train_loss,           epoch)
        self.writer.add_scalar(tag("Loss/validation"),   val_loss,             epoch)
        self.writer.add_scalar(tag("Metrics/IoU"),       metrics["IoU"],       epoch)
        self.writer.add_scalar(tag("Metrics/Accuracy"),  metrics["Accuracy"],  epoch)
        self.writer.add_scalar(tag("Metrics/Precision"), metrics["Precision"], epoch)
        self.writer.add_scalar(tag("Metrics/Recall"),    metrics["Recall"],    epoch)
        self.writer.add_scalar(tag("Metrics/F1"),        metrics["F1"],        epoch)
        self.writer.add_scalar("LR/learning_rate",       self.optimizer.param_groups[0]["lr"], epoch)


    def _write_wandb(
        self, epoch: int, train_loss: float,
        val_loss: float, metrics: dict[str, float],
    ) -> None:
        if not self.use_wandb:
            return

        stage = self.current_stage
        s     = f"{stage}/" if self.curriculum_manager is not None else ""

        payload = {
            "val/epoch":              epoch + 1,
            f"train/{s}epoch_loss":   train_loss,
            f"val/{s}loss":           val_loss,
            f"val/{s}IoU":            metrics["IoU"],
            f"val/{s}Accuracy":       metrics["Accuracy"],
            f"val/{s}Precision":      metrics["Precision"],
            f"val/{s}Recall":         metrics["Recall"],
            f"val/{s}F1":             metrics["F1"],
            f"best/{s}val_loss":      self._stage_best[stage]["val_loss"],
            "train/learning_rate":    self.optimizer.param_groups[0]["lr"],
        }

        stage_metrics = self._stage_best[stage]["metrics"]
        if stage_metrics is not None:
            payload[f"best/{s}IoU"]       = stage_metrics["IoU"]
            payload[f"best/{s}Accuracy"]  = stage_metrics["Accuracy"]
            payload[f"best/{s}Precision"] = stage_metrics["Precision"]
            payload[f"best/{s}Recall"]    = stage_metrics["Recall"]
            payload[f"best/{s}F1"]        = stage_metrics["F1"]

        self.wandb.log(payload)
