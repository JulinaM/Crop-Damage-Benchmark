from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def build_change_fusion(change_cfg, encoder: nn.Module) -> nn.Module:
    encoder_spec = getattr(encoder, "decoder_spec", None)
    if encoder_spec is None:
        raise ValueError(f"Encoder '{type(encoder).__name__}' does not expose decoder_spec.")
    return ChangeFusion(
        method=str(getattr(change_cfg, "method", "difference")),
        encoder_spec=encoder_spec,
        attention_heads=int(getattr(change_cfg, "attention_heads", 4)),
    )


class PointwiseProjection(nn.Module):
    def __init__(self, input_adapter: str, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.input_adapter = input_adapter
        if input_adapter == "tokens":
            self.proj = nn.Linear(in_channels, out_channels)
        elif input_adapter == "feature_maps":
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            raise ValueError(f"Unsupported input_adapter '{input_adapter}'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_adapter == "tokens":
            return self.proj(x)
        return self.proj(x)


class CrossAttentionFusionLevel(nn.Module):
    def __init__(self, input_adapter: str, channels: int, attention_heads: int) -> None:
        super().__init__()
        self.input_adapter = input_adapter
        num_heads = _resolve_attention_heads(channels, attention_heads)
        self.attention = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.projection = PointwiseProjection(input_adapter, channels * 2, channels)

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        if self.input_adapter == "tokens":
            after_ctx, _ = self.attention(after, before, before, need_weights=False)
            before_ctx, _ = self.attention(before, after, after, need_weights=False)
            fused = torch.cat((after - before, after_ctx - before_ctx), dim=-1)
            return self.projection(fused)

        pooled_tokens = torch.stack(
            (before.mean(dim=(2, 3)), after.mean(dim=(2, 3))),
            dim=1,
        )
        attended_tokens, _ = self.attention(pooled_tokens, pooled_tokens, pooled_tokens, need_weights=False)
        context = (attended_tokens[:, 1, :] - attended_tokens[:, 0, :]).unsqueeze(-1).unsqueeze(-1)
        context = context.expand(-1, -1, before.shape[2], before.shape[3])
        fused = torch.cat((after - before, context), dim=1)
        return self.projection(fused)


class ChangeFusion(nn.Module):
    def __init__(
        self,
        method: str,
        encoder_spec: dict[str, object],
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        self.method = _normalize_method_name(method)
        self.input_adapter = str(encoder_spec["input_adapter"])
        self.selected_indices = None

        if self.input_adapter == "tokens":
            feature_indices = [int(index) for index in list(encoder_spec["feature_indices"])]
            self.selected_indices = feature_indices
            self.base_channels = [int(encoder_spec["token_dim"])] * len(feature_indices)
            self._decoder_spec = {
                "input_adapter": "tokens",
                "token_dim": int(encoder_spec["token_dim"]) * self._channel_multiplier(),
                "feature_indices": list(range(len(feature_indices))),
                "remove_cls_token": bool(encoder_spec.get("remove_cls_token", False)),
            }
        elif self.input_adapter == "feature_maps":
            feature_channels = [int(channel) for channel in list(encoder_spec["feature_channels"])]
            self.base_channels = feature_channels
            self._decoder_spec = {
                "input_adapter": "feature_maps",
                "feature_channels": [channel * self._channel_multiplier() for channel in feature_channels],
            }
        else:
            raise ValueError(f"Unsupported input_adapter '{self.input_adapter}'.")

        if self.method == "siamese_fusion":
            self.fusion_layers = nn.ModuleList(
                [PointwiseProjection(self.input_adapter, channels * 4, channels) for channels in self.base_channels]
            )
            self._reset_decoder_spec()
        elif self.method == "attention_based_cross_time_fusion":
            self.fusion_layers = nn.ModuleList(
                [CrossAttentionFusionLevel(self.input_adapter, channels, attention_heads) for channels in self.base_channels]
            )
            self._reset_decoder_spec()
        elif self.method not in {
            "difference",
            "concatenate_all_three",
            "signed_difference_plus_absolute_difference",
            "attention_based_cross_time_fusion",
        }:
            raise ValueError(f"Unsupported change fusion method '{method}'.")

    @property
    def decoder_spec(self) -> dict[str, object]:
        return dict(self._decoder_spec)

    def forward(self, before_features: Sequence[torch.Tensor], after_features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if self.selected_indices is not None:
            before_features = [before_features[idx] for idx in self.selected_indices]
            after_features = [after_features[idx] for idx in self.selected_indices]

        fused = []
        for idx, (before, after) in enumerate(zip(before_features, after_features)):
            difference = after - before
            if self.method == "difference":
                fused.append(difference)
            elif self.method == "concatenate_all_three":
                fused.append(torch.cat((before, after, difference), dim=self._cat_dim()))
            elif self.method == "signed_difference_plus_absolute_difference":
                fused.append(torch.cat((difference, difference.abs()), dim=self._cat_dim()))
            elif self.method == "siamese_fusion":
                stacked = torch.cat((before, after, difference, difference.abs()), dim=self._cat_dim())
                fused.append(self.fusion_layers[idx](stacked))
            else:
                fused.append(self.fusion_layers[idx](before, after))
        return fused

    def _channel_multiplier(self) -> int:
        if self.method == "concatenate_all_three":
            return 3
        if self.method == "signed_difference_plus_absolute_difference":
            return 2
        return 1

    def _cat_dim(self) -> int:
        return -1 if self.input_adapter == "tokens" else 1

    def _reset_decoder_spec(self) -> None:
        if self.input_adapter == "tokens":
            self._decoder_spec["token_dim"] = self.base_channels[0]
        else:
            self._decoder_spec["feature_channels"] = list(self.base_channels)


def _resolve_attention_heads(channels: int, requested_heads: int) -> int:
    for heads in range(min(requested_heads, channels), 0, -1):
        if channels % heads == 0:
            return heads
    return 1


def _normalize_method_name(method: str) -> str:
    normalized = method.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "default": "difference",
        "subtract": "difference",
        "difference": "difference",
        "concatenate_all_three": "concatenate_all_three",
        "concat_all_three": "concatenate_all_three",
        "siamese_fusion_with_learnable_layer": "siamese_fusion",
        "siamese_fusion": "siamese_fusion",
        "attention_based_cross_time_fusion": "attention_based_cross_time_fusion",
        "cross_time_attention": "attention_based_cross_time_fusion",
        "signed_difference_plus_absolute_difference": "signed_difference_plus_absolute_difference",
        "signed_difference_and_absolute_difference": "signed_difference_plus_absolute_difference",
    }
    return aliases.get(normalized, normalized)
