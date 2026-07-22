import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio as rio
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from damage_mapping.models.change_fusion import build_change_fusion
from damage_mapping.models.decoders import build_decoder
from damage_mapping.models.encoders import build_encoder
from damage_mapping.models.utils import calc_test_metrics, move_to_device, tensor_to_color_image


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
    ) -> dict[int, dict[str, float]] | None:
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
        metrics = self._save_metrics_and_visualizations(image_tiles_pred, image_tiles_true)
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

    def _save_metrics_and_visualizations(
        self,
        image_tiles_pred: dict[int, torch.Tensor],
        image_tiles_true: dict[int, torch.Tensor],
    ) -> dict[int, dict[str, float]]:
        metrics = calc_test_metrics(
            image_tiles_pred,
            image_tiles_true,
            ignore_index=self.cfg.model.ignore_index,
            positive_class=self.cfg.model.positive_class,
            negative_class=self.cfg.model.negative_class,
        )
        metrics_path = self.output_dir / "metrics.txt"
        with metrics_path.open("w") as handle:
            for idx, image_metrics in metrics.items():
                handle.write(f"\nImage {idx} metrics:\n")
                for key, value in image_metrics.items():
                    handle.write(f"  {key}: {value:.4f}\n")
                    self.writer.add_scalar(f"{self.eval_name}/{key}", value, global_step=idx)

        for idx in list(image_tiles_pred.keys())[:3]:
            pred_rgb = tensor_to_color_image(image_tiles_pred[idx], num_classes=self.cfg.model.num_classes)
            true_rgb = tensor_to_color_image(image_tiles_true[idx], num_classes=self.cfg.model.num_classes)
            self.writer.add_image(f"{self.eval_name}/comparison_{idx}", torch.cat((true_rgb, pred_rgb), dim=2))

        self.logger.info("Saved %s metrics to %s", self.eval_name, metrics_path)
        self._write_wandb(metrics)
        return metrics

    def _write_wandb(self, metrics: dict[int, dict[str, float]]) -> None:
        if not self.use_wandb or not metrics:
            return

        metric_names = next(iter(metrics.values())).keys()
        summary = {}
        for name in metric_names:
            values = [image_metrics[name] for image_metrics in metrics.values()]
            summary[f"{self.eval_name}/{name}_mean"] = float(np.mean(values))
            summary[f"{self.eval_name}/{name}_min"] = float(np.min(values))
            summary[f"{self.eval_name}/{name}_max"] = float(np.max(values))

        self.wandb.summary.update(summary)

    @staticmethod
    def _to_int(value) -> int:
        if torch.is_tensor(value):
            return int(value.item())
        return int(value)
