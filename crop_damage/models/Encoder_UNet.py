from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetEncoder(nn.Module):
    """
    Vanilla U-Net encoder that returns multi-scale feature maps.
    """

    def __init__(
        self,
        modalities: Sequence[str] = ("S2L2A",),
        modality_channels: Mapping[str, int] | None = None,
        channels: Sequence[int] = (64, 128, 256, 512, 1024),
        **_,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("UNetEncoder requires at least one encoder channel.")

        self.modalities = tuple(modalities)
        self.modality_channels = {str(name): int(value) for name, value in dict(modality_channels or {}).items()}
        self.channels = list(channels)
        in_channels = sum(self.modality_channels.get(modality, 0) for modality in self.modalities)
        if in_channels <= 0:
            raise ValueError("UNetEncoder requires positive input channels. Check modality_channels.")

        blocks = []
        prev_channels = in_channels
        for out_channels in self.channels:
            blocks.append(DoubleConv(prev_channels, out_channels))
            prev_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    @property
    def output_channels(self) -> list[int]:
        return list(self.channels)

    @property
    def decoder_spec(self) -> dict[str, object]:
        return {
            "input_adapter": "feature_maps",
            "feature_channels": list(self.channels),
        }

    def forward(self, x: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        features = []
        current = self._stack_modalities(x)
        for idx, block in enumerate(self.blocks):
            current = block(current)
            features.append(current)
            if idx != len(self.blocks) - 1:
                current = self.pool(current)
        return features

    def _stack_modalities(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        tensors = []
        reference = next(iter(x.values()))
        for modality in self.modalities:
            if modality in x:
                tensors.append(x[modality])
                continue

            channels = self.modality_channels.get(modality)
            if channels is None:
                raise KeyError(f"Missing modality '{modality}' and no channel count is configured for zero-fill.")
            tensors.append(reference.new_zeros(reference.size(0), channels, reference.size(2), reference.size(3)))
        return torch.cat(tensors, dim=1)
