from typing import Iterable, Sequence

import torch.nn as nn
from terratorch.registry import BACKBONE_REGISTRY


class TerraMindEncoder(nn.Module):
    """
    Thin wrapper around TerraMind backbone with trainability controls.

    Expected usage in segmentation:
    - `forward(x)` returns multi-scale token features from TerraMind.
    - Decoder consumes selected scales after temporal differencing.
    """

    def __init__(
        self,
        version: str = "terramind_v1_base",
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

        self.model = BACKBONE_REGISTRY.build(
            version,
            pretrained=pretrained,
            modalities=list(self.modalities),
            **self.build_kwargs,
        )
        # self.has_lora_adapters = False

        if not self.finetune:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)

    @property
    def decoder_spec(self) -> dict[str, object]:
        return {
            "input_adapter": "tokens",
            "token_dim": int(getattr(self, "token_dim", 768)),
            "feature_indices": list(getattr(self, "feature_indices", (3, 5, 7, 9, 11))),
        }

    # def inject_lora_adapters(
    #     self,
    #     *,
    #     r: int = 8,
    #     alpha: int = 16,
    #     dropout: float = 0.05,
    #     target_modules: Sequence[str] = ("qkv", "proj", "fc1", "fc2"),
    #     bias: str = "none",
    # ) -> None:
    #     """
    #     Inject PEFT LoRA adapters into the TerraMind backbone.
    #     """
    #     try:
    #         from peft import LoraConfig, TaskType, get_peft_model
    #     except ImportError as exc:
    #         raise ImportError(
    #             "LoRA mode requires `peft`. Install it in your training environment, "
    #             "for example: `pip install peft`."
    #         ) from exc

    #     lora_config = LoraConfig(
    #         r=r,
    #         lora_alpha=alpha,
    #         lora_dropout=dropout,
    #         target_modules=list(target_modules),
    #         bias=bias,
    #         task_type=TaskType.FEATURE_EXTRACTION,
    #     )
    #     self.model = get_peft_model(self.model, lora_config)
    #     self.has_lora_adapters = True

    # def set_train_mode(
    #     self,
    #     mode: str,
    #     *,
    #     few_shot_last_n_blocks: int = 2,
    #     train_norm_layers_in_few_shot: bool = True,
    # ) -> None:
    #     """
    #     Configure parameter trainability for common transfer setups.

    #     Modes:
    #     - `zero_shot`: freeze all encoder params (decoder-only adaptation/inference).
    #     - `few_shot`: freeze backbone, unfreeze last `few_shot_last_n_blocks`.
    #     - `full_finetune`: unfreeze all backbone params.
    #     - `lora`: freeze base params, train only injected LoRA params (`lora_` names).
    #     """
    #     normalized = mode.strip().lower()
    #     if normalized not in {"zero_shot", "few_shot", "full_finetune", "lora"}:
    #         raise ValueError(f"Unsupported train mode: {mode}")

    #     if normalized == "full_finetune":
    #         self._set_requires_grad(self.model.parameters(), True)
    #         return

    #     self._set_requires_grad(self.model.parameters(), False)

    #     if normalized == "few_shot":
    #         self._unfreeze_last_encoder_blocks(few_shot_last_n_blocks)
    #         if train_norm_layers_in_few_shot:
    #             for module in self.model.modules():
    #                 if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
    #                     for param in module.parameters():
    #                         param.requires_grad = True
    #         return

    #     if normalized == "lora":
    #         if not self.has_lora_adapters:
    #             raise RuntimeError(
    #                 "LoRA train mode requested but adapters are not injected. "
    #                 "Call `inject_lora_adapters(...)` first."
    #             )
    #         self.enable_lora_only_training()

    # def enable_lora_only_training(self) -> None:
    #     """
    #     Enable gradients only for LoRA adapter params.
    #     Call this after LoRA adapters are injected into the backbone.
    #     """
    #     for name, param in self.model.named_parameters():
    #         param.requires_grad = "lora_" in name.lower()

    # def _unfreeze_last_encoder_blocks(self, n_blocks: int) -> None:
    #     if n_blocks <= 0 or not hasattr(self.model, "encoder"):
    #         return

    #     encoder = self.model.encoder
    #     if isinstance(encoder, Iterable):
    #         blocks = list(encoder)
    #         for block in blocks[-n_blocks:]:
    #             for param in block.parameters():
    #                 param.requires_grad = True

    # @staticmethod
    # def _set_requires_grad(parameters, flag: bool) -> None:
    #     for param in parameters:
    #         param.requires_grad = flag
