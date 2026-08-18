from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Sequence

import torch
import torch.nn as nn

from crop_damage.models.third_party.croma_model import PretrainedCROMA

_VERSION_TO_SIZE = {
    "croma_base": "base",
    "croma_large": "large",
}


class CromaEncoder(nn.Module):
    """
    Thin wrapper around the vendored CROMA implementation
    (crop_damage/models/third_party/croma_model.py, from antofuller/CROMA).

    Unlike TerraMind/Prithvi, CROMA has no terratorch registry entry: the
    backbone is built directly from a local checkpoint file, and its native
    forward(SAR_images=, optical_images=) returns a single final-layer tensor
    per stream rather than a per-block list. To satisfy this pipeline's
    "tokens" decoder_spec contract (a list of (B, N, token_dim) tensors
    indexable by feature_indices), forward hooks on the joint cross-encoder's
    per-layer FFN modules capture one hidden state per layer.
    """

    SAR_CHANNELS = 2
    OPTICAL_CHANNELS = 12
    SAR_MODALITY_KEYS = ("S1GRD", "S1RTC")

    def __init__(
        self,
        version: str,
        pretrained: bool = True,
        finetune: bool = False,
        modalities: Sequence[str] = ("S2L2A", "S1GRD"),
        image_size: int = 224,
        **build_kwargs,
    ) -> None:
        super().__init__()
        self.version = version
        self.pretrained = pretrained  # NOTE: no-op -- CROMA always loads pretrained_path in __init__
        self.finetune = finetune
        self.modalities = tuple(modalities)
        self.image_size = int(image_size)
        self.build_kwargs = dict(build_kwargs)

        size = _VERSION_TO_SIZE.get(version)
        if size is None:
            raise ValueError(
                f"Unsupported CROMA version '{version}'. Expected one of: {sorted(_VERSION_TO_SIZE)}."
            )

        pretrained_path = self.build_kwargs.pop("pretrained_path", None)
        if not pretrained_path or not os.path.isfile(pretrained_path):
            raise FileNotFoundError(
                f"CromaEncoder requires encoder.build_kwargs.pretrained_path to point at a downloaded "
                f"CROMA checkpoint (got {pretrained_path!r}). Download one first, e.g.:\n"
                f"  wget https://huggingface.co/antofuller/CROMA/resolve/main/CROMA_{size}.pt "
                f"-P data/checkpoints/croma/"
            )

        modality = self.build_kwargs.pop("modality", "both")

        self.model = PretrainedCROMA(
            pretrained_path=pretrained_path,
            size=size,
            modality=modality,
            image_resolution=self.image_size,
            **self.build_kwargs,
        )

        self._layer_features: list[torch.Tensor] = []
        for _self_attn, _cross_attn, ffn in self.model.cross_encoder.layers:
            ffn.register_forward_pre_hook(self._capture_layer_input)

        if not self.finetune:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        if isinstance(x, Mapping):
            x = self._prepare_croma_inputs(x)
            self._layer_features = []
            self.model(SAR_images=x["x_sar"], optical_images=x["x_optical"])
            return list(self._layer_features)
        raise TypeError("CromaEncoder.forward expects a modality mapping (e.g. {'S2L2A': ..., 'S1GRD': ...}).")

    @property
    def decoder_spec(self) -> dict[str, object]:
        return {
            "input_adapter": "tokens",
            "token_dim": int(getattr(self, "token_dim", 768)),
            "feature_indices": list(getattr(self, "feature_indices", (3, 5, 7, 9, 11))),
            "remove_cls_token": False,
        }

    def _capture_layer_input(self, module: nn.Module, args: tuple) -> None:
        self._layer_features.append(args[0])

    def _prepare_croma_inputs(self, x: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        reference = next(iter(x.values()), None)
        if reference is None:
            raise ValueError("CromaEncoder received an empty modality mapping.")

        x_sar = self._select_sar_channels(x, reference)
        x_optical = self._select_optical_channels(x, reference)

        for name, tensor in (("x_sar", x_sar), ("x_optical", x_optical)):
            height, width = tensor.shape[2], tensor.shape[3]
            if height != self.image_size or width != self.image_size:
                raise ValueError(
                    f"CromaEncoder configured image_size={self.image_size} but received {name} of "
                    f"shape {tuple(tensor.shape)}. CROMA's positional bias is sized at construction "
                    "time and requires an exact match -- set encoder.image_size (or the dataset "
                    "group's patch_size) to match the actual input resolution."
                )

        return {"x_sar": x_sar, "x_optical": x_optical}

    def _select_sar_channels(
        self,
        x: Mapping[str, torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        sar_key = next((key for key in self.SAR_MODALITY_KEYS if key in x), None)
        if sar_key is None:
            return reference.new_zeros(
                reference.shape[0], self.SAR_CHANNELS, reference.shape[2], reference.shape[3]
            ).float()

        s1 = x[sar_key]
        if s1.ndim != 4:
            raise ValueError(f"CromaEncoder expects BCHW tensors, got shape {tuple(s1.shape)}.")
        if s1.shape[1] != self.SAR_CHANNELS:
            raise ValueError(
                f"CromaEncoder expected '{sar_key}' with {self.SAR_CHANNELS} channels, got {s1.shape[1]}."
            )
        return s1.float()

    def _select_optical_channels(
        self,
        x: Mapping[str, torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if "S2L2A" not in x:
            return reference.new_zeros(
                reference.shape[0], self.OPTICAL_CHANNELS, reference.shape[2], reference.shape[3]
            ).float()

        s2 = x["S2L2A"]
        if s2.ndim != 4:
            raise ValueError(f"CromaEncoder expects BCHW tensors, got shape {tuple(s2.shape)}.")
        if s2.shape[1] != self.OPTICAL_CHANNELS:
            raise ValueError(
                f"CromaEncoder expected 'S2L2A' with {self.OPTICAL_CHANNELS} channels, got {s2.shape[1]}."
            )
        return s2.float()
