from collections.abc import Mapping
from typing import Sequence

import torch
import torch.nn as nn
from terratorch.datasets.utils import HLSBands, SARBands
from terratorch.registry import BACKBONE_REGISTRY


class PrithviEncoder(nn.Module):
    """
    Thin wrapper around a Prithvi backbone registered in Terratorch.
    """

    S2_TO_HLS_INDICES = (0, 1, 2, 7, 9, 10)
    DEFAULT_MODEL_BANDS = (
        HLSBands.BLUE,
        HLSBands.GREEN,
        HLSBands.RED,
        HLSBands.NIR_NARROW,
        HLSBands.SWIR_1,
        HLSBands.SWIR_2,
        SARBands.VV,
        SARBands.VH,
    )

    def __init__(
        self,
        version: str,
        pretrained: bool = True,
        finetune: bool = False,
        modalities: Sequence[str] = ("S2L2A",),
        **build_kwargs,
    ) -> None:
        super().__init__()
        self.version = version
        self.pretrained = pretrained
        self.finetune = finetune
        self.modalities = tuple(modalities)
        self.build_kwargs = dict(build_kwargs)
        self.model_bands = self._resolve_model_bands()

        self.model = BACKBONE_REGISTRY.build(
            version,
            pretrained=pretrained,
            bands=list(self.model_bands),
            **self.build_kwargs,
        )

        if not self.finetune:
            for param in self.model.parameters():
                param.requires_grad = False


    def forward(self, x):
        if isinstance(x, Mapping):
            x = self._prepare_input_tensor(x)
        return self.model(x)

    @property
    def decoder_spec(self) -> dict[str, object]:
        return {
            "input_adapter": "tokens",
            "token_dim": int(getattr(self, "token_dim", 768)),
            "feature_indices": list(getattr(self, "feature_indices", (3, 5, 7, 9, 11))),
            "remove_cls_token": True,
        }

    def _resolve_model_bands(self) -> tuple[HLSBands | SARBands | str, ...]:
        raw_bands = self.build_kwargs.pop("model_bands", None)
        if raw_bands is None:
            raw_bands = self.build_kwargs.pop("bands", None)
        if raw_bands is None:
            return self.DEFAULT_MODEL_BANDS
        return tuple(self._coerce_band_enum(band) for band in raw_bands)

    def _prepare_input_tensor(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        reference = next(iter(x.values()), None)
        if reference is None:
            raise ValueError("PrithviEncoder received an empty modality mapping.")
        tensors = []
        for modality in self.modalities:
            if modality == "S2L2A":
                tensors.append(self._select_s2_hls_channels(x, reference))
                continue
            if modality == "S1GRD":
                tensors.append(self._select_s1_channels(x, reference))
                continue
            raise ValueError(f"PrithviEncoder does not support modality '{modality}'.")

        if not tensors:
            raise ValueError("PrithviEncoder requires at least one configured modality.")
        return torch.cat(tensors, dim=1)

    @staticmethod
    def _coerce_band_enum(band):
        band = HLSBands.try_convert_to_hls_bands_enum(band)
        band = SARBands.try_convert_to_optical_bands_enum(band)
        return band

    def _select_s2_hls_channels(
        self,
        x: Mapping[str, torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if "S2L2A" not in x:
            return reference.new_zeros(reference.shape[0], len(self.S2_TO_HLS_INDICES), reference.shape[2], reference.shape[3])
        s2 = x["S2L2A"]
        if s2.ndim != 4:
            raise ValueError(f"PrithviEncoder expects BCHW tensors, got shape {tuple(s2.shape)}.")
        if s2.shape[1] == len(self.S2_TO_HLS_INDICES):
            return s2
        if s2.shape[1] >= max(self.S2_TO_HLS_INDICES) + 1:
            return s2[:, self.S2_TO_HLS_INDICES, :, :]
        raise ValueError(
            "PrithviEncoder expected S2L2A input with either 6 HLS bands or the repo's 12-band layout."
        )

    def _select_s1_channels(
        self,
        x: Mapping[str, torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if "S1GRD" not in x:
            return reference.new_zeros(reference.shape[0], 2, reference.shape[2], reference.shape[3])
        s1 = x["S1GRD"]
        if s1.ndim != 4:
            raise ValueError(f"PrithviEncoder expects BCHW tensors, got shape {tuple(s1.shape)}.")
        if s1.shape[1] != 2:
            raise ValueError(f"PrithviEncoder expected S1GRD with 2 channels, got {s1.shape[1]}.")
        return s1
