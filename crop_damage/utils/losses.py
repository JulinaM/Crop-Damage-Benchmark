import torch
import torch.nn as nn
import torch.nn.functional as F

from damage_mapping.models.utils import weights


class MulticlassCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        ignore_index: int | None = None,
        weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        ce_ignore_index = -100 if ignore_index is None else ignore_index
        self.loss = nn.CrossEntropyLoss(ignore_index=ce_ignore_index, weight=weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, target)


class MulticlassDiceLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        ignore_index: int | None = None,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError(f"Expected logits with shape [B, C, H, W], got {tuple(logits.shape)}")
        if target.ndim != 3:
            raise ValueError(f"Expected target with shape [B, H, W], got {tuple(target.shape)}")
        if logits.shape[1] != self.num_classes:
            raise ValueError(
                f"Configured for {self.num_classes} classes, but logits have {logits.shape[1]} channels"
            )

        probabilities = torch.softmax(logits, dim=1)
        valid_mask = torch.ones_like(target, dtype=torch.bool)
        safe_target = target.clone()

        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            safe_target = target.masked_fill(~valid_mask, 0)

        target_one_hot = F.one_hot(safe_target.long(), num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).to(probabilities.dtype)

        valid_mask = valid_mask.unsqueeze(1)
        probabilities = probabilities * valid_mask
        target_one_hot = target_one_hot * valid_mask

        reduce_dims = (0, 2, 3)
        intersection = (probabilities * target_one_hot).sum(dim=reduce_dims)
        denominator = probabilities.sum(dim=reduce_dims) + target_one_hot.sum(dim=reduce_dims)
        dice_per_class = (2 * intersection + self.smooth) / (denominator + self.smooth)

        class_mask = torch.ones(self.num_classes, dtype=torch.bool, device=logits.device)
        if self.ignore_index is not None and 0 <= self.ignore_index < self.num_classes:
            class_mask[self.ignore_index] = False

        if not torch.any(class_mask):
            raise ValueError("Dice loss has no valid classes to average after applying ignore_index")

        return 1.0 - dice_per_class[class_mask].mean()


class MulticlassFocalLoss(nn.Module):
    def __init__(
        self,
        ignore_index: int | None = None,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        alpha: float | None = None,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_ignore_index = -100 if self.ignore_index is None else self.ignore_index
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            ignore_index=ce_ignore_index,
            reduction="none",
        )

        valid_mask = torch.ones_like(target, dtype=torch.bool)
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index

        safe_target = target.masked_fill(~valid_mask, 0)
        probs = torch.softmax(logits, dim=1)
        pt = probs.gather(1, safe_target.unsqueeze(1)).squeeze(1)
        pt = pt.clamp_min(1e-8)

        focal_factor = (1.0 - pt) ** self.gamma
        if self.alpha is not None:
            focal_factor = focal_factor * self.alpha

        loss = focal_factor * ce
        loss = loss[valid_mask]
        if loss.numel() == 0:
            return logits.new_tensor(0.0)
        return loss.mean()


class HybridDiceFocalLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        ignore_index: int | None = None,
        weight: torch.Tensor | None = None,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_gamma: float = 2.0,
        focal_alpha: float | None = None,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice = MulticlassDiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            smooth=smooth,
        )
        self.focal = MulticlassFocalLoss(
            ignore_index=ignore_index,
            weight=weight,
            gamma=focal_gamma,
            alpha=focal_alpha,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(logits, target)
        focal_loss = self.focal(logits, target)
        return self.dice_weight * dice_loss + self.focal_weight * focal_loss


def build_criterion(
    criterion_cfg,
    num_classes,
    ignore_index,
    train_loader,
    device: torch.device | str,
) -> nn.Module:
    # from omegaconf import OmegaConf
    # print("criterion config: %s", OmegaConf.to_container(criterion_cfg,  resolve=True))
    criterion_name = criterion_cfg.name.lower()
    weight = None

    if criterion_name in {"ce", "cross_entropy", "crossentropy"}:
        if criterion_cfg.apply_weight_loss:
            inverse, pixels = weights(
                train_loader,
                num_classes=num_classes,
                ignore_index=ignore_index,
                device=device,
            )
            if criterion_cfg.weight_type == "pixels":
                weight = pixels
            elif criterion_cfg.weight_type == "inverse":
                weight = inverse
            else:
                raise ValueError(f"Invalid weight_type: {criterion_cfg.weight_type}")
        return MulticlassCrossEntropyLoss(ignore_index=ignore_index, weight=weight)

    if criterion_name == "dice":
        return MulticlassDiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            smooth=float(getattr(criterion_cfg, "dice_smooth", 1e-6)),
        )

    if criterion_name in {"dice_focal", "hybrid_dice_focal", "dicefocal"}:
        return HybridDiceFocalLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            weight=weight,
            dice_weight=float(getattr(criterion_cfg, "dice_weight", 0.5)),
            focal_weight=float(getattr(criterion_cfg, "focal_weight", 0.5)),
            focal_gamma=float(getattr(criterion_cfg, "focal_gamma", 2.0)),
            focal_alpha=getattr(criterion_cfg, "focal_alpha", None),
            smooth=float(getattr(criterion_cfg, "dice_smooth", 1e-6)),
        )

    raise ValueError(
        f"Unsupported criterion '{criterion_name}'. Expected one of: "
        "'cross_entropy', 'ce', 'dice', or 'dice_focal'."
    )
