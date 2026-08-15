"""Map-style Dataset reading AgDamage_v2 WebDataset .tar shards directly.

Each hazard directory under `data_root` (e.g. Flood/, Burnt/) holds
`manifest.parquet` (chip_id, event_id, hazard, split, shard, ...) and
`shards/<split>/<shard>.tar` files. Every chip is one WebDataset "sample":
`<chip_id>.{s1_pre,s1_post,s2_pre,s2_post,label}.tif` + `<chip_id>.json`.
Chips are natively 512x512; this loader tiles each chip into a grid of
overlapping/non-overlapping patches using the same pad-to-grid + sliding-window
logic as the legacy Images_large loader in DataLoader.py (Train_Val_Loader /
TestLoader), so a chip is no longer assumed to be one atomic training sample.
`webdataset`'s tar-decoding utilities are used internally, but the class stays
map-style (indexable, has __len__) so it drops into the existing
DataLoader(...)/Trainer.py/Evaluator.py machinery unchanged.
"""
import logging
import math
import tarfile
import warnings
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from rasterio.io import MemoryFile
from torch.utils.data import Dataset
from torchvision import transforms

from crop_damage.models.utils import RandomFlipPair, RandomRotationPair, standardize

LOGGER = logging.getLogger(__name__)

# Shard rasters carry no GDAL band descriptions, so bands are selected by
# position/count rather than by name (unlike DataLoader.py's
# _select_expected_bands). Verified against sample chips: S2 rasters already
# carry 12 bands, S1 rasters 2 bands, matching DataLoader.EXPECTED_MODALITY_BANDS
# counts -- band *order* within each modality is assumed canonical, not verified.
EXPECTED_MODALITY_BAND_COUNTS = {"S2L2A": 12, "S1GRD": 2}
_MODALITY_PRE_POST_KEYS = {
    "S2L2A": ("s2_pre", "s2_post"),
    "S1GRD": ("s1_pre", "s1_post"),
}

# Label raster is 5 bands, uint8, nodata=255. Confirmed empirically (see
# scratchpad inspect_labels.py run) against the JSON sidecar's
# flooded_crop_frac/burned_crop_frac, cropland_frac and excluded_crop_frac:
#   band 1: per-pixel class -- 1=damaged (flooded/burnt) cropland,
#           2=unaffected cropland, 3=excluded cropland (QC), 4=non-cropland.
#   band 4: binary cropland-intersection mask (1 where band1 in {1,2,3}).
# Remapped here to match the pre-existing model.{ignore_index,negative_class,
# positive_class} = (0,1,2) convention already used in configs/terramind.yaml.
DEFAULT_LABEL_BAND = 1
DEFAULT_LABEL_NODATA = 255
DEFAULT_IGNORE_INDEX = 0
DEFAULT_LABEL_CLASS_REMAP = {1: 2, 2: 1, 3: 0, 4: 0}


class AgDamageShardDataset(Dataset):
    """
    Args:
        data_root: path to AgDamage_v2/ (contains one directory per hazard).
        hazards: list of hazard directory names, e.g. ["Flood"], ["Burnt"],
            or ["Flood", "Burnt"] for pooled training / leave-one-hazard-out.
        split: "train", "val", or "test" -- must match a value in manifest's
            `split` column (splits are event-grouped and leakage-checked
            upstream; never re-derive splits here).
        modalities: which modalities to load, subset of ("S2L2A", "S1GRD").
        label_band: 1-indexed band of label.tif to use as the segmentation
            target (see DEFAULT_LABEL_CLASS_REMAP docstring above).
        class_remap: raw label value -> training class id. Values not present
            as keys (including label_nodata) fall back to ignore_index.
        patch_size: size of the sliding-window patch extracted from each chip.
            Chips are natively 512x512; each chip is padded (zero-pad, split
            evenly top/bottom & left/right) so (H - patch_size) is a multiple
            of `stride`, then tiled into a grid of patches -- identical logic
            to Train_Val_Loader/TestLoader in DataLoader.py. patch_size >= 512
            degenerates to one patch per chip.
        stride: sliding-window stride between patches. stride < patch_size
            gives overlapping patches; stride > patch_size means some pixels
            are never covered by any patch (a UserWarning is raised).
        num_augmentations: 0 disables augmentation; >0 replays each patch that
            many times per epoch with random flip/rotation (train split only).
        preload: cache decoded chip tensors in memory across epochs.
        mode: "train_val" returns (sample, y); "test" additionally returns
            (chip_idx, coord_y, coord_x), (0, 0, 0, 0), meta to match
            TestLoader's contract for Evaluator's tile-reconstruction code.
    """

    def __init__(
        self,
        data_root,
        hazards,
        split: str,
        modalities=("S2L2A", "S1GRD"),
        label_band: int = DEFAULT_LABEL_BAND,
        label_nodata: int = DEFAULT_LABEL_NODATA,
        ignore_index: int = DEFAULT_IGNORE_INDEX,
        class_remap: dict | None = None,
        patch_size: int = 224,
        stride: int = 224,
        num_augmentations: int = 0,
        preload: bool = False,
        mode: str = "train_val",
    ):
        if mode not in ("train_val", "test"):
            raise ValueError(f"Invalid mode '{mode}'. Must be 'train_val' or 'test'.")

        if stride > patch_size:
            warnings.warn(
                "Caution: with stride > patch_size, some pixels may not be seen by the model.",
                UserWarning,
            )

        self.data_root = Path(data_root)
        self.hazards = list(hazards)
        self.split = split
        self.modalities = list(modalities)
        self.label_band = label_band
        self.label_nodata = label_nodata
        self.ignore_index = ignore_index
        self.class_remap = dict(class_remap) if class_remap else dict(DEFAULT_LABEL_CLASS_REMAP)
        self.patch_size = patch_size
        self.stride = stride
        self.num_augmentations = num_augmentations
        self.preload = preload
        self.mode = mode
        self.size_helper = 1 if self.num_augmentations == 0 else self.num_augmentations

        self.index = self._build_index()
        if not self.index:
            raise ValueError(
                f"No chips found for hazards={self.hazards} split={split!r} under {self.data_root}"
            )
        LOGGER.info(
            "AgDamageShardDataset: %d chips (hazards=%s, split=%s)",
            len(self.index), self.hazards, split,
        )

        self._open_tars: dict[Path, tarfile.TarFile] = {}
        self._cache: dict[int, dict] = {}
        if self.preload:
            for i in range(len(self.index)):
                self._cache[i] = self._read_chip(i)

        # Precompute patch coordinates for all chips (same grid-of-patches
        # logic as Train_Val_Loader/TestLoader in DataLoader.py).
        self.index_map = self._build_patch_index_map()

        self.augment = None
        if self.mode == "train_val" and self.split == "train" and self.num_augmentations > 0:
            self.augment = transforms.Compose([RandomFlipPair(), RandomRotationPair()])

    def _build_index(self) -> list[tuple[str, Path, str]]:
        rows = []
        for hazard in self.hazards:
            hazard_dir = self.data_root / hazard
            try:
                manifest = pd.read_parquet(hazard_dir / "manifest.parquet")
            except Exception:
                manifest = pd.read_csv(hazard_dir / "manifest.csv")
            manifest = manifest[manifest["split"] == self.split]
            for _, row in manifest.iterrows():
                shard_path = hazard_dir / "shards" / self.split / row["shard"]
                rows.append((hazard, shard_path, row["chip_id"]))
        return rows

    # ------------------- patch grid (mirrors DataLoader.py) -------------------

    # Pad chip to fit multiples of patch_size/stride for H & W.
    def _pad_image(self, img: torch.Tensor):
        if not img.is_floating_point():
            img = img.float()

        _, H, W = img.shape
        pad_h = (math.ceil((H - self.patch_size) / self.stride) * self.stride + self.patch_size) - H
        pad_w = (math.ceil((W - self.patch_size) / self.stride) * self.stride + self.patch_size) - W
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        padded = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)
        return padded, pad_top, pad_left

    # Get coordinates of top-left of patches in grid.
    def _extract_patch_coords(self, img: torch.Tensor):
        _, H, W = img.shape
        coords = []
        for y in range(0, H - self.patch_size + 1, self.stride):
            for x in range(0, W - self.patch_size + 1, self.stride):
                coords.append((y, x))
        return coords

    def _chip_shape(self, i: int):
        hazard, shard_path, chip_id = self.index[i]
        tar = self._get_tar(shard_path)
        first_mod = self.modalities[0]
        pre_key, _ = _MODALITY_PRE_POST_KEYS[first_mod]
        raw = tar.extractfile(f"{chip_id}.{pre_key}.tif").read()
        with MemoryFile(raw) as mf, mf.open() as src:
            return src.height, src.width

    def _build_patch_index_map(self):
        index_map = []
        for i in range(len(self.index)):
            H, W = self._chip_shape(i)
            dummy = torch.zeros((1, H, W), dtype=torch.float32)
            dummy, _, _ = self._pad_image(dummy)
            coords = self._extract_patch_coords(dummy)
            for c in coords:
                index_map.append((i, c))
        return index_map

    def __len__(self):
        return len(self.index_map) * self.size_helper

    # ------------------- tar / raster reading -------------------

    def _get_tar(self, shard_path: Path) -> tarfile.TarFile:
        # Each DataLoader worker process gets its own copy of self (post-fork),
        # so caching one open TarFile per shard per worker is safe; TarFile
        # objects must not be shared across processes.
        tar = self._open_tars.get(shard_path)
        if tar is None:
            tar = tarfile.open(shard_path, "r")
            self._open_tars[shard_path] = tar
        return tar

    def _read_raster(self, tar: tarfile.TarFile, chip_id: str, member: str, modality: str | None = None):
        raw = tar.extractfile(f"{chip_id}.{member}.tif").read()
        with MemoryFile(raw) as mf, mf.open() as src:
            arr = src.read().astype("float32")
            nodata = src.nodata
            meta = src.meta.copy()

        if modality is not None:
            expected = EXPECTED_MODALITY_BAND_COUNTS.get(modality)
            if expected is not None and arr.shape[0] != expected:
                raise ValueError(
                    f"{modality} chip {chip_id} ({member}) has {arr.shape[0]} bands, "
                    f"expected {expected}. Shard rasters carry no band descriptions, "
                    "so bands are selected by position -- verify band order."
                )

        tensor = torch.from_numpy(arr)
        if nodata is not None:
            tensor[tensor == nodata] = 0.0
        tensor = torch.nan_to_num(tensor, nan=0.0)
        return tensor, meta

    def _read_label(self, tar: tarfile.TarFile, chip_id: str):
        raw = tar.extractfile(f"{chip_id}.label.tif").read()
        with MemoryFile(raw) as mf, mf.open() as src:
            arr = src.read()
            meta = src.meta.copy()

        band = arr[self.label_band - 1]
        y = torch.full(band.shape, self.ignore_index, dtype=torch.long)
        for raw_value, mapped_value in self.class_remap.items():
            y[torch.from_numpy(band) == raw_value] = mapped_value
        # Anything not in class_remap (including label_nodata=255) stays ignore_index.
        return y, meta

    def _read_chip(self, i: int) -> dict:
        hazard, shard_path, chip_id = self.index[i]
        tar = self._get_tar(shard_path)

        before, after = {}, {}
        for name in self.modalities:
            pre_key, post_key = _MODALITY_PRE_POST_KEYS[name]
            pre_arr, _ = self._read_raster(tar, chip_id, pre_key, modality=name)
            post_arr, _ = self._read_raster(tar, chip_id, post_key, modality=name)
            # Pad once per chip (not once per patch): every patch drawn from
            # this chip slices the same padded tensor in _extract_patch, so
            # padding here -- rather than inside _extract_patch -- avoids
            # redoing the pad on every __getitem__ call when preload=True
            # caches this dict across the whole training run.
            before[name], _, _ = self._pad_image(pre_arr)
            after[name], _, _ = self._pad_image(post_arr)

        y, label_meta = self._read_label(tar, chip_id)
        y_padded, _, _ = self._pad_image(y.unsqueeze(0))
        y = y_padded[0]

        return {
            "before": before,
            "after": after,
            "y": y,
            "label_meta": label_meta,
            "chip_id": chip_id,
            "hazard": hazard,
        }

    # ------------------- patch extraction -------------------

    def _extract_patch(self, before: dict, after: dict, y: torch.Tensor, coord_y: int, coord_x: int):
        # before/after/y are already padded (see _read_chip); this only
        # slices out the patch and standardizes it, matching the per-patch
        # standardization order used in Train_Val_Loader/TestLoader.
        before_patch, after_patch = {}, {}
        for name in self.modalities:
            patch_before = before[name][:, coord_y:coord_y + self.patch_size, coord_x:coord_x + self.patch_size]
            patch_after = after[name][:, coord_y:coord_y + self.patch_size, coord_x:coord_x + self.patch_size]

            before_patch[name] = standardize(patch_before, dim=1).float()
            after_patch[name] = standardize(patch_after, dim=1).float()

        y_patch = y[coord_y:coord_y + self.patch_size, coord_x:coord_x + self.patch_size]
        return before_patch, after_patch, y_patch

    # ------------------- Dataset interface -------------------

    def __getitem__(self, index: int):
        patch_index = index // self.size_helper
        img_index, (coord_y, coord_x) = self.index_map[patch_index]
        data = self._cache[img_index] if self.preload else self._read_chip(img_index)

        # Shallow-copy the before/after dicts (not their tensors) so
        # augmentation -- which reassigns dict entries to new flipped/rotated
        # tensors -- never mutates a cached chip shared across epochs.
        before = dict(data["before"])
        after = dict(data["after"])
        y = data["y"]

        before, after, y = self._extract_patch(before, after, y, coord_y, coord_x)

        sample = {"before": before, "after": after, "y": y}
        if self.augment is not None:
            sample = self.augment(sample)

        y = sample.pop("y").long()

        if self.mode == "test":
            meta = data["label_meta"].copy()
            return sample, y, (img_index, coord_y, coord_x), (0, 0, 0, 0), meta

        return sample, y
