import torch.nn as nn

from damage_mapping.models.Encoder_UNet import UNetEncoder
from damage_mapping.models.Encoder_Prithvi import PrithviEncoder
from damage_mapping.models.Encoder_TerraMind import TerraMindEncoder


def build_encoder(encoder_cfg) -> nn.Module:
    encoder_name = str(getattr(encoder_cfg, "name", "Terramind")).strip().lower()
    build_kwargs = dict(getattr(encoder_cfg, "build_kwargs", {}) or {})

    if encoder_name in {"terramind", "terra_mind"}:
        encoder = TerraMindEncoder(
            version=encoder_cfg.version,
            pretrained=bool(getattr(encoder_cfg, "pretrained", True)),
            finetune=bool(getattr(encoder_cfg, "finetune", False)),
            modalities=list(encoder_cfg.modalities),
            **build_kwargs,
        )
        encoder.token_dim = int(getattr(encoder_cfg, "token_dim", 768))
        encoder.feature_indices = list(getattr(encoder_cfg, "feature_indices", (3, 5, 7, 9, 11)))
        return encoder

    if encoder_name == "prithvi": 
        encoder = PrithviEncoder(
            version=encoder_cfg.version,
            pretrained=bool(getattr(encoder_cfg, "pretrained", True)),
            finetune=bool(getattr(encoder_cfg, "finetune", False)),
            modalities=list(encoder_cfg.modalities),
            **build_kwargs,
        )
        encoder.token_dim = int(getattr(encoder.model, "embed_dim", getattr(encoder_cfg, "token_dim", 768),))
        encoder.feature_indices = list(getattr(encoder_cfg, "feature_indices", (3, 5, 7, 9, 11)))
        return encoder

    if encoder_name == "unet":
        return UNetEncoder(
            modalities=list(encoder_cfg.modalities),
            **build_kwargs,
        )

    raise ValueError(
        f"Unsupported encoder '{encoder_cfg.name}'. Expected one of: 'Terramind', 'Prithvi', or 'UNet'."
    )
