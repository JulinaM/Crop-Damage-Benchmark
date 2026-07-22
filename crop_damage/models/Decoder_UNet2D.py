from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from terratorch.models import necks


class ConvBlock(nn.Module):
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


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        #Why Transposed Convolution Can Cause Checkerboard Artifacts TODO #Upsample (bilinear)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2) #Why Transposed Convolution Can Cause Checkerboard Artifacts
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        #TODO
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    """
    U-Net style segmentation decoder for TerraMind token features.

    Expects a list of token tensors from the encoder, selects multi-scale indices,
    reshapes tokens to images, then decodes to pixel logits.
    """

    def __init__(
        self,
        num_classes: int = 3,
        input_adapter: str = "tokens",
        token_dim: int = 768,
        indices: Sequence[int] = (3, 5, 7, 9, 11),
        feature_channels: Sequence[int] | None = None,
        decoder_channels: Sequence[int] = (64, 128, 256, 512, 1024),
        remove_cls_token: bool = False,
    ) -> None:
        super().__init__()

        self.input_adapter = input_adapter
        if self.input_adapter == "tokens":
            feature_count = len(indices)
        elif self.input_adapter == "feature_maps":
            if feature_channels is None:
                raise ValueError("`feature_channels` is required when input_adapter='feature_maps'.")
            feature_count = len(feature_channels)
        else:
            raise ValueError(f"Unsupported input_adapter '{input_adapter}'.")

        if feature_count != len(decoder_channels):
            raise ValueError("Encoder features and decoder_channels must have the same length.")
        if feature_count < 2:
            raise ValueError("Decoder requires at least two feature scales.")

        self.decoder_channels = list(decoder_channels)
        n_levels = feature_count

        if self.input_adapter == "tokens":
            self.indices = list(indices)
            channel_list = [token_dim] * n_levels
            self.select_indices = necks.SelectIndices(channel_list=channel_list, indices=self.indices)
            self.reshape_tokens = necks.ReshapeTokensToImage(
                channel_list=channel_list,
                remove_cls_token=remove_cls_token,
            )
            projection_inputs = [token_dim] * n_levels
        else:
            self.indices = list(range(n_levels))
            self.select_indices = None
            self.reshape_tokens = None
            projection_inputs = list(feature_channels)

        self.projections = nn.ModuleList(
            # [nn.Conv2d(token_dim, out_ch, kernel_size=1) for out_ch in self.decoder_channels]
            [nn.Conv2d(in_ch, out_ch, kernel_size=1) for in_ch, out_ch in zip(projection_inputs, self.decoder_channels)]
        )

        #To build Terramind as Feature Pyramid Network
        if self.input_adapter == "tokens":
            self.upsamplers = nn.ModuleList(
                [
                    nn.Identity()
                    if i == n_levels - 1
                    else nn.Upsample(scale_factor=2 ** (n_levels - 1 - i), mode="bilinear", align_corners=False)
                    for i in range(n_levels)
                ]
            )
        else:
            self.upsamplers = nn.ModuleList([nn.Identity() for _ in range(n_levels)])

        self.bottom = ConvBlock(self.decoder_channels[-1], self.decoder_channels[-1])
        self.up_blocks = nn.ModuleList()
        curr_channels = self.decoder_channels[-1]
        for skip_channels in reversed(self.decoder_channels[:-1]):
            self.up_blocks.append(
                UpBlock(
                    in_channels=curr_channels,
                    skip_channels=skip_channels,
                    out_channels=skip_channels,
                )
            )
            curr_channels = skip_channels

        self.head = nn.Conv2d(curr_channels, num_classes, kernel_size=1)

    def forward(self, embeddings):
        if self.input_adapter == "tokens":
            features = self.select_indices(embeddings)
            features = self.reshape_tokens(features)
        else:
            features = embeddings

        projected = []
        for feature, projection, upsample in zip(features, self.projections, self.upsamplers):
            projected.append(upsample(projection(feature)))

        x = self.bottom(projected[-1])
        skips = projected[:-1]
        for up_block, skip in zip(self.up_blocks, reversed(skips)):
            x = up_block(x, skip)

        return self.head(x)
