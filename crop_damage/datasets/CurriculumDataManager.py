# damage_mapping/datasets/CurriculumDataManager.py

from torch.utils.data import DataLoader
from damage_mapping.datasets.DataLoader import Train_Val_Loader


class CurriculumDataManager:
    """
    Manages automatic dataset switching between flood and conflict stages
    within a single training run.  Both training AND validation loaders are
    swapped at the stage boundary so metrics are always computed on the
    distribution currently being trained on.
    """

    def __init__(self, cfg, flood_train_cfg, conflict_train_cfg,
                 flood_val_cfg=None, conflict_val_cfg=None):
        self.flood_epochs    = int(cfg.curriculum.flood_epochs)
        self.conflict_epochs = int(cfg.curriculum.conflict_epochs)
        self.stage2_lr       = float(cfg.curriculum.stage2_lr)
        self.current_stage   = "flood"

        if self.flood_epochs < 0 or self.conflict_epochs < 0:
            raise ValueError(
                f"curriculum.flood_epochs and conflict_epochs must be >= 0, "
                f"got flood_epochs={self.flood_epochs}, conflict_epochs={self.conflict_epochs}."
            )
        if self.flood_epochs == 0:
            import warnings
            warnings.warn(
                "curriculum.flood_epochs=0: Stage 1 (flood) will be skipped entirely. "
                "The stage switch fires at epoch 1 with uninitialized flood metrics.",
                UserWarning, stacklevel=2,
            )
        if self.conflict_epochs == 0:
            import warnings
            warnings.warn(
                "curriculum.conflict_epochs=0: Stage 2 (conflict) will never run. "
                "The Evaluator will load the flood-stage checkpoint, not a conflict checkpoint.",
                UserWarning, stacklevel=2,
            )

        self.flood_loader    = self._build(flood_train_cfg,    "train")
        self.conflict_loader = self._build(conflict_train_cfg, "train")

        # Validation loaders are optional; None means the caller keeps its
        # existing val_loader unchanged for that stage.
        self.flood_val_loader    = self._build(flood_val_cfg,    "validation") if flood_val_cfg    is not None else None
        self.conflict_val_loader = self._build(conflict_val_cfg, "validation") if conflict_val_cfg is not None else None

    # ------------------------------------------------------------------
    # Internal builder
    # ------------------------------------------------------------------

    def _build(self, loader_cfg, split: str) -> DataLoader:
        modalities = {
            name: (paths.before, paths.after)
            for name, paths in loader_cfg.modalities.items()
        }
        dataset = Train_Val_Loader(
            modalities        = modalities,
            label_dir         = loader_cfg.label_dir,
            split             = split,
            num_augmentations = getattr(loader_cfg, "num_augmentations", 0),
            patch_size        = loader_cfg.patch_size,
            stride            = loader_cfg.stride,
            preload           = loader_cfg.preload,
        )
        return DataLoader(
            dataset,
            batch_size  = loader_cfg.batch_size,
            shuffle     = loader_cfg.shuffle,
            num_workers = getattr(loader_cfg, "num_workers", 0),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_stage(self, epoch: int) -> str:
        """Return the stage name for a 1-indexed epoch."""
        return "flood" if epoch <= self.flood_epochs else "conflict"

    def get_loader(self, epoch: int) -> DataLoader:
        """Return the training DataLoader for a 1-indexed epoch."""
        stage = self.get_stage(epoch)
        return self.flood_loader if stage == "flood" else self.conflict_loader

    def get_val_loader(self, epoch: int) -> DataLoader | None:
        """
        Return the validation DataLoader matching the current stage.
        Returns None if no stage-specific val loader was configured
        (caller keeps its existing val_loader unchanged).
        """
        stage = self.get_stage(epoch)
        return self.flood_val_loader if stage == "flood" else self.conflict_val_loader

    def should_switch(self, epoch: int) -> bool:
        """True at the exact 1-indexed epoch where the stage transition happens."""
        return epoch == self.flood_epochs + 1

    @property
    def total_epochs(self) -> int:
        return self.flood_epochs + self.conflict_epochs
