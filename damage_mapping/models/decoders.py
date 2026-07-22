from __future__ import annotations

import torch.nn as nn

from damage_mapping.models.Decoder_UNet2D import UNet2D


def build_decoder(decoder_cfg, feature_source: nn.Module, num_classes: int) -> nn.Module:
    decoder_name = str(getattr(decoder_cfg, "name", "UNet")).strip().lower()
    if decoder_name != "unet":
        raise ValueError(f"Unsupported decoder '{decoder_cfg.name}'. Expected 'UNet'.")

    decoder_channels = list(getattr(decoder_cfg, "channels", (64, 128, 256, 512, 1024)))
    encoder_spec = getattr(feature_source, "decoder_spec", None)
    if encoder_spec is None:
        raise ValueError(f"Feature source '{type(feature_source).__name__}' does not expose decoder_spec.")

    input_adapter = str(encoder_spec["input_adapter"])
    if input_adapter == "tokens": # for TM and Prithvi
        return UNet2D(
            num_classes=num_classes,
            input_adapter="tokens",
            token_dim=int(encoder_spec["token_dim"]),
            indices=list(encoder_spec["feature_indices"]),
            decoder_channels=decoder_channels,
            remove_cls_token=bool(encoder_spec.get("remove_cls_token", False)),
        )

    if input_adapter == "feature_maps": # For Unet Encoder
        return UNet2D(
            num_classes=num_classes,
            input_adapter="feature_maps",
            feature_channels=list(encoder_spec["feature_channels"]),
            decoder_channels=decoder_channels,
        )

    raise ValueError(f"Unsupported decoder input_adapter '{input_adapter}'.")
