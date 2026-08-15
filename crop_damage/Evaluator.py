import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio as rio
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from crop_damage.models.change_fusion import build_change_fusion
from crop_damage.models.decoders import build_decoder
from crop_damage.models.encoders import build_encoder
from crop_damage.models.utils import (
    bootstrap_ci,
    calc_confusion_counts,
    calc_epoch_metrics,
    calc_test_metrics,
    move_to_device,
    tensor_to_color_image,
)

METRIC_NAMES = ("Accuracy", "Precision", "Recall", "F1", "IoU")


COLOR_TABLE = {
    0: (0, 0, 0, 255),
    1: (0, 255, 0, 255),
    2: (255, 0, 0, 255),
    3: (255, 255, 0, 255),
}


class Evaluator:
    def __init__(
        self,
        cfg: DictConfig,
        exp_dir: str | Path,
        ckpt_dir: str | Path,
        device: str | torch.device,
        dataloader: DataLoader | None,
        logger: logging.Logger | None = None,
        use_wandb: bool = False,
        eval_name: str = "holdout",
    ) -> None:
        self.cfg = cfg
        self.exp_dir = Path(exp_dir)
        self.ckpt_dir = Path(ckpt_dir)
        self.device = torch.device(device)
        self.dataloader = dataloader
        self.logger = logger or logging.getLogger(__name__)
        self.eval_name = eval_name
        self.output_dir = self.exp_dir / eval_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.output_dir))
        self.use_wandb = use_wandb

        if self.use_wandb:
            import wandb
            self.wandb = wandb

    def is_configured(self) -> bool:
        return self.dataloader is not None

    def evaluate(
        self,
        checkpoint_path: str | Path | None = None,
        checkpoint_prefix: str = "best",
    ) -> dict | None:
        if not self.is_configured():
            self.logger.info("%s evaluator skipped: no dataloader was provided.", self.eval_name)
            self.writer.close()
            return None

        checkpoint_path = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else self._find_best_checkpoint(checkpoint_prefix)
        )
        encoder, change_fusion, decoder = self._load_models(checkpoint_path)

        self.logger.info("%s evaluator started", self.eval_name)
        self.logger.info("%s patches: %d", self.eval_name, len(self.dataloader.dataset))
        self.logger.info("Using checkpoint: %s", checkpoint_path)

        tile_reconstruction, padding, metas = self._collect_patch_outputs(self.dataloader, encoder, change_fusion, decoder)
        image_tiles_true, image_tiles_pred = self._reconstruct_tiles(
            tile_reconstruction,
            patch_size=self.dataloader.dataset.patch_size,
        )
        self._remove_padding(image_tiles_true, padding)
        self._remove_padding(image_tiles_pred, padding)
        self._mask_background_predictions(image_tiles_pred, image_tiles_true)

        geotiff_dir = self._save_geotiffs(image_tiles_pred, metas)
        metrics = self._save_metrics_and_visualizations(image_tiles_pred, image_tiles_true, checkpoint_path)
        self.writer.close()

        self.logger.info("Saved %s GeoTIFFs to %s", self.eval_name, geotiff_dir)
        self.logger.info("%s evaluation completed", self.eval_name)
        return metrics

    def _find_best_checkpoint(self, checkpoint_prefix: str = "best") -> Path:
        matches = sorted(self.ckpt_dir.glob(f"{checkpoint_prefix}_model_*.pt"))
        if not matches:
            raise FileNotFoundError(
                f"No {checkpoint_prefix} checkpoint found in {self.ckpt_dir}"
            )
        return matches[-1]

    def _load_models(self, checkpoint_path: Path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        encoder = build_encoder(self.cfg.encoder)
        change_fusion = build_change_fusion(self.cfg.change, encoder)
        decoder = build_decoder(self.cfg.decoder, change_fusion, num_classes=self.cfg.model.num_classes)
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        if "change_fusion_state_dict" in checkpoint:
            change_fusion.load_state_dict(checkpoint["change_fusion_state_dict"])
        decoder.load_state_dict(checkpoint["decoder_state_dict"])
        encoder.to(self.device).eval()
        change_fusion.to(self.device).eval()
        decoder.to(self.device).eval()
        return encoder, change_fusion, decoder

    def _collect_patch_outputs(
        self,
        dataloader: DataLoader,
        encoder,
        change_fusion,
        decoder,
    ) -> tuple[dict[int, list], dict[int, tuple[int, int, int, int]], dict[int, dict]]:
        padding = {}
        metas = {}
        tile_reconstruction = defaultdict(list)

        with torch.no_grad():
            for inputs, target, (idx, coord_y, coord_x), pad, meta in dataloader:
                inputs = move_to_device(inputs, self.device)
                z_before = encoder(inputs["before"])
                z_after = encoder(inputs["after"])
                fused_features = change_fusion(z_before, z_after)
                logits = decoder(fused_features)
                prediction = torch.argmax(logits, dim=1).cpu()

                idx = self._to_int(idx)
                tile_reconstruction[idx].append(
                    (prediction, target.cpu(), self._to_int(coord_y), self._to_int(coord_x))
                )
                if idx not in padding:
                    padding[idx] = tuple(self._to_int(value) for value in pad)
                if idx not in metas:
                    metas[idx] = meta

        if not tile_reconstruction:
            raise RuntimeError(
                f"No {self.eval_name} patches were generated. Check the configured loader paths."
            )
        return tile_reconstruction, padding, metas

    def _reconstruct_tiles(
        self,
        tile_reconstruction: dict[int, list],
        patch_size: int,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        image_tiles_true = {}
        image_tiles_pred = {}

        for idx, patches in tile_reconstruction.items():
            max_coord_x = max(coord_x for _, _, _, coord_x in patches)
            max_coord_y = max(coord_y for _, _, coord_y, _ in patches)
            height = max_coord_y + patch_size
            width = max_coord_x + patch_size
            image_tiles_true[idx] = torch.zeros((height, width), dtype=torch.float32)
            image_tiles_pred[idx] = torch.zeros((height, width), dtype=torch.float32)

        for idx, patches in tile_reconstruction.items():
            for prediction, target, coord_y, coord_x in patches:
                y_slice = slice(coord_y, coord_y + patch_size)
                x_slice = slice(coord_x, coord_x + patch_size)
                image_tiles_true[idx][y_slice, x_slice] = target.squeeze()
                image_tiles_pred[idx][y_slice, x_slice] = prediction.squeeze()

        return image_tiles_true, image_tiles_pred

    def _remove_padding(self, image_tiles: dict[int, torch.Tensor], padding: dict[int, tuple[int, int, int, int]]) -> None:
        for idx, image in image_tiles.items():
            height, width = image.shape[-2], image.shape[-1]
            pad_left, pad_right, pad_top, pad_bottom = padding[idx]
            image_tiles[idx] = image[pad_top:height - pad_bottom, pad_left:width - pad_right]

    def _mask_background_predictions(
        self,
        image_tiles_pred: dict[int, torch.Tensor],
        image_tiles_true: dict[int, torch.Tensor],
    ) -> None:
        for idx, prediction in image_tiles_pred.items():
            truth = image_tiles_true[idx]
            image_tiles_pred[idx] = torch.where(truth == 0, torch.zeros_like(prediction), prediction)

    def _save_geotiffs(self, image_tiles_pred: dict[int, torch.Tensor], metas: dict[int, dict]) -> Path:
        geotiff_dir = self.output_dir / "geotiffs"
        geotiff_dir.mkdir(parents=True, exist_ok=True)

        for idx, prediction in image_tiles_pred.items():
            meta_out = metas[idx].copy()
            meta_out.pop("photometric", None)
            meta_out.update(
                {
                    "driver": "GTiff",
                    "height": prediction.shape[0],
                    "width": prediction.shape[1],
                    "count": 1,
                    "dtype": "uint8",
                    "nodata": 0,
                }
            )
            output_path = geotiff_dir / f"predicted_map_{idx}_colored.tif"
            with rio.open(output_path, "w", **meta_out) as dst:
                dst.write(prediction.numpy().astype(np.uint8), 1)
                dst.write_colormap(1, COLOR_TABLE)

        return geotiff_dir

    def _event_id_for_chip(self, idx: int) -> str:
        # Chips whose dataset doesn't expose event_id_for_chip (or that lack
        # event grouping) each become their own singleton "event", so macro
        # metrics degenerate gracefully to per-chip instead of crashing.
        dataset = getattr(self.dataloader, "dataset", None)
        get_event_id = getattr(dataset, "event_id_for_chip", None)
        return get_event_id(idx) if get_event_id is not None else str(idx)

    def _macro_and_micro_metrics(
        self,
        image_tiles_pred: dict[int, torch.Tensor],
        image_tiles_true: dict[int, torch.Tensor],
    ) -> dict:
        """
        Event-grouped macro metrics (+bootstrap CI) and globally-pooled micro
        metrics, per CLAUDE.md's evaluation protocol: macro-by-event is the
        headline generalization number, micro is the finer-grained pixel-pooled
        complement, and the CI flags noise when the event count is small.
        """
        ignore_index = self.cfg.model.ignore_index
        positive_class = self.cfg.model.positive_class
        negative_class = self.cfg.model.negative_class

        chip_counts = {
            idx: calc_confusion_counts(
                image_tiles_pred[idx], image_tiles_true[idx],
                ignore_index=ignore_index, positive_class=positive_class, negative_class=negative_class,
            )
            for idx in image_tiles_pred
        }

        event_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
        for idx, counts in chip_counts.items():
            acc = event_counts[self._event_id_for_chip(idx)]
            for i in range(4):
                acc[i] += counts[i]

        per_event_metrics = {
            event_id: calc_epoch_metrics(*counts) for event_id, counts in event_counts.items()
        }

        macro, macro_ci = {}, {}
        for name in METRIC_NAMES:
            values = [m[name] for m in per_event_metrics.values()]
            macro[name] = float(np.mean(values)) if values else float("nan")
            macro_ci[name] = bootstrap_ci(values)

        micro_counts = [sum(c[i] for c in chip_counts.values()) for i in range(4)]
        micro = calc_epoch_metrics(*micro_counts)

        return {
            "per_event": per_event_metrics,
            "macro": macro,
            "macro_ci": macro_ci,
            "micro": micro,
            "n_events": len(event_counts),
            "n_chips": len(chip_counts),
        }

    def _save_metrics_and_visualizations(
        self,
        image_tiles_pred: dict[int, torch.Tensor],
        image_tiles_true: dict[int, torch.Tensor],
        checkpoint_path: Path,
    ) -> dict:
        per_chip = calc_test_metrics(
            image_tiles_pred,
            image_tiles_true,
            ignore_index=self.cfg.model.ignore_index,
            positive_class=self.cfg.model.positive_class,
            negative_class=self.cfg.model.negative_class,
        )
        agg = self._macro_and_micro_metrics(image_tiles_pred, image_tiles_true)

        metrics_path = self.output_dir / "metrics.txt"
        with metrics_path.open("w") as handle:
            handle.write(
                f"Summary: {agg['n_chips']} chip(s) across {agg['n_events']} event(s)\n"
            )
            handle.write("\nMacro (mean over events, [95% bootstrap CI]):\n")
            for name in METRIC_NAMES:
                low, high = agg["macro_ci"][name]
                handle.write(f"  {name}: {agg['macro'][name]:.4f}  [{low:.4f}, {high:.4f}]\n")
                self.writer.add_scalar(f"{self.eval_name}/macro/{name}", agg["macro"][name], global_step=0)

            handle.write("\nMicro (pooled over all chips/pixels):\n")
            for name in METRIC_NAMES:
                handle.write(f"  {name}: {agg['micro'][name]:.4f}\n")
                self.writer.add_scalar(f"{self.eval_name}/micro/{name}", agg["micro"][name], global_step=0)

            handle.write("\nPer-event metrics:\n")
            for event_id, event_metrics in agg["per_event"].items():
                handle.write(f"\n  Event {event_id}:\n")
                for key, value in event_metrics.items():
                    handle.write(f"    {key}: {value:.4f}\n")

            handle.write("\nPer-chip metrics:\n")
            for idx, image_metrics in per_chip.items():
                handle.write(f"\n  Chip {idx}:\n")
                for key, value in image_metrics.items():
                    handle.write(f"    {key}: {value:.4f}\n")
                    self.writer.add_scalar(f"{self.eval_name}/per_chip/{key}", value, global_step=idx)

        self._save_metrics_json(agg, checkpoint_path)

        for idx in list(image_tiles_pred.keys())[:3]:
            pred_rgb = tensor_to_color_image(image_tiles_pred[idx], num_classes=self.cfg.model.num_classes)
            true_rgb = tensor_to_color_image(image_tiles_true[idx], num_classes=self.cfg.model.num_classes)
            self.writer.add_image(f"{self.eval_name}/comparison_{idx}", torch.cat((true_rgb, pred_rgb), dim=2))

        self.logger.info("Saved %s metrics to %s", self.eval_name, metrics_path)
        self.logger.info(
            "%s macro IoU=%.4f [%.4f, %.4f] | macro F1=%.4f [%.4f, %.4f] | micro IoU=%.4f | %d event(s), %d chip(s)",
            self.eval_name,
            agg["macro"]["IoU"], *agg["macro_ci"]["IoU"],
            agg["macro"]["F1"], *agg["macro_ci"]["F1"],
            agg["micro"]["IoU"],
            agg["n_events"], agg["n_chips"],
        )

        result = {"per_chip": per_chip, **agg}
        self._write_wandb(result)
        return result

    def _save_metrics_json(self, agg: dict, checkpoint_path: Path) -> None:
        """
        Machine-readable companion to metrics.txt: {experiment}/{eval_name}/metrics.json.
        Lets a results-collection script glob across many experiment dirs and
        build comparison tables (e.g. in-distribution vs LOHO IoU) without
        re-parsing the human-readable text report or the training config.
        """
        payload = {
            "experiment_name": str(getattr(self.cfg, "experiment_name", None)),
            "eval_name": self.eval_name,
            "task": str(getattr(self.cfg, "task", None)),
            "encoder": str(getattr(self.cfg.encoder, "name", None)),
            "encoder_finetune": bool(getattr(self.cfg.encoder, "finetune", False)),
            "train_hazards": list(getattr(self.cfg.train_loader, "hazards", []) or []),
            "checkpoint": str(checkpoint_path),
            "n_events": agg["n_events"],
            "n_chips": agg["n_chips"],
            "macro": agg["macro"],
            "macro_ci": {name: list(ci) for name, ci in agg["macro_ci"].items()},
            "micro": agg["micro"],
            "per_event": agg["per_event"],
        }
        json_path = self.output_dir / "metrics.json"
        with json_path.open("w") as handle:
            json.dump(payload, handle, indent=2)

    def _write_wandb(self, result: dict) -> None:
        if not self.use_wandb or not result:
            return

        summary = {}
        for name in METRIC_NAMES:
            summary[f"{self.eval_name}/macro/{name}"] = result["macro"][name]
            low, high = result["macro_ci"][name]
            summary[f"{self.eval_name}/macro/{name}_ci_low"] = low
            summary[f"{self.eval_name}/macro/{name}_ci_high"] = high
            summary[f"{self.eval_name}/micro/{name}"] = result["micro"][name]

        per_chip = result["per_chip"]
        if per_chip:
            metric_names = next(iter(per_chip.values())).keys()
            for name in metric_names:
                values = [image_metrics[name] for image_metrics in per_chip.values()]
                summary[f"{self.eval_name}/per_chip/{name}_mean"] = float(np.mean(values))
                summary[f"{self.eval_name}/per_chip/{name}_min"] = float(np.min(values))
                summary[f"{self.eval_name}/per_chip/{name}_max"] = float(np.max(values))

        self.wandb.summary.update(summary)

    @staticmethod
    def _to_int(value) -> int:
        if torch.is_tensor(value):
            return int(value.item())
        return int(value)
