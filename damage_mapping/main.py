import os
import pathlib
from pathlib import Path

import hydra
import torch
import torch.optim as optim
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from damage_mapping.Evaluator import Evaluator
from damage_mapping.Trainer import Trainer
from damage_mapping.datasets.CurriculumDataManager import CurriculumDataManager
from damage_mapping.datasets.DataLoader import TestLoader, Train_Val_Loader
from damage_mapping.logger import init_logger
from damage_mapping.models.change_fusion import build_change_fusion
from damage_mapping.models.decoders import build_decoder
from damage_mapping.models.encoders import build_encoder
from damage_mapping.utils.losses import build_criterion
from damage_mapping.models.utils import set_seeds

REPO_DIR = pathlib.Path(__file__).parent.parent
WORK_DIR = REPO_DIR / "data/input"
EXPERIMENT_DIR = REPO_DIR / "data/experiments"
CONFIG_DIR = REPO_DIR / "configs/"


def init_wandb_run(cfg: DictConfig, exp_dir: Path, exp_name: str):
    import wandb

    # Hydra multirun can execute multiple jobs in a single local process.
    # Close any previous active run so each sweep member gets its own W&B run.
    if wandb.run is not None:
        wandb.finish()

    wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
    wandb_settings = {
        "project": cfg.wandb.project,
        "dir": str(exp_dir),
        "config": wandb_cfg,
        "name": cfg.wandb.name or exp_name,
        "mode": getattr(cfg.wandb, "mode", "online"),
        "reinit": True,
    }

    run = wandb.init(**wandb_settings)
    wandb.define_metric("train/global_step")
    wandb.define_metric("train/*",          step_metric="train/global_step")
    wandb.define_metric("train/flood/*",    step_metric="train/global_step")
    wandb.define_metric("train/conflict/*", step_metric="train/global_step")
    wandb.define_metric("val/epoch")
    wandb.define_metric("val/*",            step_metric="val/epoch")
    wandb.define_metric("val/flood/*",      step_metric="val/epoch")
    wandb.define_metric("val/conflict/*",   step_metric="val/epoch")
    wandb.define_metric("best/*",           step_metric="val/epoch")
    wandb.define_metric("best/flood/*",     step_metric="val/epoch")
    wandb.define_metric("best/conflict/*",  step_metric="val/epoch")
    return run


def build_loader(loader_cfg: DictConfig, split: str) -> DataLoader:
    modalities = {name: (paths.before, paths.after) for name, paths in loader_cfg.modalities.items()}
    dataset = Train_Val_Loader(
        modalities=modalities,
        label_dir=loader_cfg.label_dir,
        split=split,
        num_augmentations=getattr(loader_cfg, "num_augmentations", 0),
        patch_size=loader_cfg.patch_size,
        stride=loader_cfg.stride,
        preload=loader_cfg.preload,
    )
    return DataLoader(
        dataset,
        batch_size=loader_cfg.batch_size,
        shuffle=loader_cfg.shuffle,
        num_workers=getattr(loader_cfg, "num_workers", 0),
    )


def build_holdout_loader(loader_cfg: DictConfig | None) -> DataLoader | None:
    if loader_cfg is None:
        return None

    modalities = {name: (paths.before, paths.after) for name, paths in loader_cfg.modalities.items()}
    dataset = TestLoader(
        modalities=modalities,
        label_dir=loader_cfg.label_dir,
        patch_size=loader_cfg.patch_size,
        stride=loader_cfg.stride,
    )
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=getattr(loader_cfg, "num_workers", 0),
    )


def build_holdout_loaders(cfg: DictConfig) -> dict[str, DataLoader]:
    holdout_loaders = {}
    for eval_name, loader_key in (
        ("conflict", "holdout_loader"),
        ("flood", "flood_holdout_loader"),
    ):
        loader = build_holdout_loader(cfg.get(loader_key))
        if loader is not None:
            holdout_loaders[eval_name] = loader
    return holdout_loaders


def build_model_components(cfg: DictConfig, device: torch.device, train_loader: DataLoader):
    criterion = build_criterion(
        criterion_cfg=cfg.criterion,
        num_classes=cfg.model.num_classes,
        ignore_index=cfg.model.ignore_index,
        train_loader=train_loader,
        device=device,
    )

    if str(getattr(cfg.encoder, "name", "Terramind")).strip().lower() == "unet":
        single_input = next(iter(train_loader))[0]
        modality_channels = {name: int(value.shape[1]) for name, value in single_input['before'].items()}
        print('modality_channels: ', modality_channels)
        with open_dict(cfg.encoder):
            cfg.encoder.build_kwargs = dict(getattr(cfg.encoder, "build_kwargs", {}) or {})
            cfg.encoder.build_kwargs["modality_channels"] = modality_channels

    encoder = build_encoder(cfg.encoder)
    change_fusion = build_change_fusion(cfg.change, encoder)
    decoder = build_decoder(cfg.decoder, change_fusion, num_classes=cfg.model.num_classes)
    encoder.to(device)
    change_fusion.to(device)
    decoder.to(device)

    encoder_name = str(getattr(cfg.encoder, "name", "Terramind")).strip().lower()
    optimized_params = list(change_fusion.parameters()) + list(decoder.parameters())
    if encoder_name == "unet" or bool(getattr(cfg.encoder, "finetune", False)):
        optimizer = optim.Adam(list(encoder.parameters()) + optimized_params, lr=cfg.model.learning_rate)
    else:
        optimizer = optim.Adam(optimized_params, lr=cfg.model.learning_rate)

    return encoder, change_fusion, decoder, criterion, optimizer


@hydra.main(version_base="1.2", config_path=str(CONFIG_DIR), config_name="terramind")
def main(cfg: DictConfig):
    set_seeds(cfg.model.seed)

    distributed = bool(getattr(cfg, "distributed", False))

    if distributed:
        torch.distributed.init_process_group(backend="nccl")
    else:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    hc = HydraConfig.get()
    # exp_dir = Path(os.path.join(hc["sweep"]["dir"], hc["sweep"]["subdir"])) if "sweep" in hc else Path(hc.runtime.output_dir)
    exp_dir = Path(hc.runtime.output_dir)
    exp_name = exp_dir.name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger_path = exp_dir / "out.log"

    wandb_run = None
    if cfg.use_wandb and rank == 0:
        wandb_run = init_wandb_run(cfg, exp_dir, exp_name)
    logger = init_logger(logger_path, rank=rank)
    logger.info("Experiment name: %s", exp_name)
    logger.info("Device name: %s", device)
    logger.info("The experiment is stored in %s", exp_dir)

    curriculum_cfg = getattr(cfg, "curriculum", None)
    curriculum_enabled = curriculum_cfg is not None and bool(getattr(curriculum_cfg, "enabled", False))

    if curriculum_enabled:
        logger.info("Curriculum learning enabled: flood→conflict")
        curriculum_manager = CurriculumDataManager(
            cfg,
            flood_train_cfg    = cfg.flood_loader,
            conflict_train_cfg = cfg.train_loader,
            flood_val_cfg      = getattr(cfg, "flood_validation_loader", None),
            conflict_val_cfg   = cfg.validation_loader,
        )
        train_loader = curriculum_manager.flood_loader
        # Start with flood val loader; Trainer will swap at stage boundary
        val_loader = curriculum_manager.flood_val_loader or build_loader(cfg.validation_loader, "validation")
    else:
        curriculum_manager = None
        train_loader = build_loader(cfg.train_loader, "train")
        val_loader   = build_loader(cfg.validation_loader, "validation")
    holdout_loaders = build_holdout_loaders(cfg)
    encoder, change_fusion, decoder, criterion, optimizer = build_model_components(cfg, device, train_loader)

    encoder_total = sum(p.numel() for p in encoder.parameters())
    encoder_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    change_total = sum(p.numel() for p in change_fusion.parameters())
    change_trainable = sum(p.numel() for p in change_fusion.parameters() if p.requires_grad)
    decoder_total = sum(p.numel() for p in decoder.parameters())
    decoder_trainable= sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    
    logger.info("Built {}.".format(encoder.__class__.__module__))
    logger.info("Encoder params: total=%d | trainable=%d", encoder_total, encoder_trainable)
    logger.info("Built {}.".format(change_fusion.__class__.__module__))
    logger.info("Change fusion params: total=%d | trainable=%d", change_total, change_trainable)
    logger.info("Built {}.".format(decoder.__class__.__module__))
    logger.info("Decoder params: total=%d | trainable=%d", decoder_total, decoder_trainable)

    trainer = Trainer(
        cfg=cfg,
        exp_dir=exp_dir,
        ckpt_dir=ckpt_dir,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        encoder=encoder,
        change_fusion=change_fusion,
        decoder=decoder,
        criterion=criterion,
        optimizer=optimizer,
        logger=logger,
        use_wandb=cfg.use_wandb,
        curriculum_manager=curriculum_manager,
    )
    best_val_iou = trainer.train()
    if holdout_loaders:
        for eval_name, holdout_loader in holdout_loaders.items():
            evaluator = Evaluator(
                cfg=cfg,
                exp_dir=exp_dir,
                ckpt_dir=ckpt_dir,
                device=device,
                dataloader=holdout_loader,
                logger=logger,
                use_wandb=cfg.use_wandb,
                eval_name=eval_name,
            )
            checkpoint_prefix = "best_flood" if eval_name == "flood" else "best"
            evaluator.evaluate(checkpoint_prefix=checkpoint_prefix)
    else:
        logger.info("Holdout evaluator skipped: no holdout dataloaders were configured.")

    if distributed:
        torch.distributed.destroy_process_group()

    if wandb_run is not None:
        wandb_run.finish()

    return best_val_iou


if __name__ == "__main__":
    main()
