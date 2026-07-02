"""Segmentation losses used by FloodNet experiments."""

from __future__ import annotations

from typing import Mapping, Any

import torch
import torch.nn.functional as F

from .boundary_context import boundary_context_loss
from .constants import IGNORE_INDEX, NUM_CLASSES
from .models import SegmentationModelOutput, extract_logits


def multiclass_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
    num_classes: int = NUM_CLASSES,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Soft Dice loss for mutually exclusive semantic classes."""

    if logits.ndim != 4 or target.ndim != 3:
        raise ValueError(f"Expected logits [B,C,H,W] and target [B,H,W], got {logits.shape}, {target.shape}")
    valid = target != ignore_index
    safe_target = target.clamp(min=0, max=num_classes - 1)
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(safe_target, num_classes=num_classes).permute(0, 3, 1, 2).to(probabilities.dtype)
    valid_float = valid.unsqueeze(1).to(probabilities.dtype)
    probabilities = probabilities * valid_float
    one_hot = one_hot * valid_float
    dims = (0, 2, 3)
    intersection = torch.sum(probabilities * one_hot, dim=dims)
    denominator = torch.sum(probabilities + one_hot, dim=dims)
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    present = torch.sum(one_hot, dim=dims) > 0
    if not torch.any(present):
        return logits.sum() * 0.0
    return 1.0 - dice[present].mean()


def segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    config: Mapping[str, Any] | None = None,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Compute configured supervised semantic loss.

    Default is CE+Dice for new supervised protocols. Older callers can pass
    ``{"name": "cross_entropy"}`` to reproduce CE-only behavior.
    """

    config = dict(config or {"name": "ce_dice"})
    name = str(config.get("name", "ce_dice")).casefold()
    ce = F.cross_entropy(logits, target, ignore_index=ignore_index)
    if name in {"cross_entropy", "ce"}:
        return ce
    if name in {"ce_dice", "cross_entropy_dice"}:
        ce_weight = float(config.get("ce_weight", 1.0))
        dice_weight = float(config.get("dice_weight", 1.0))
        dice = multiclass_dice_loss(logits, target, ignore_index=ignore_index)
        return ce_weight * ce + dice_weight * dice
    raise ValueError(f"Unsupported loss.name: {name}")

def supervised_objective(
    output: object,
    target: torch.Tensor,
    config: Mapping[str, Any] | None = None,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Compute semantic loss plus any enabled config-gated auxiliary losses."""

    config = dict(config or {})
    logits = extract_logits(output)
    total = segmentation_loss(
        logits,
        target,
        config.get("loss"),
        ignore_index=ignore_index,
    )
    modules = config.get("modules", {})
    boundary_config = modules.get("boundary_context", {}) if isinstance(modules, Mapping) else {}
    auxiliary = output.auxiliary if isinstance(output, SegmentationModelOutput) else {}
    total = total + boundary_context_loss(
        semantic_logits=logits,
        auxiliary=auxiliary,
        target=target,
        config=boundary_config,
        ignore_index=ignore_index,
    )
    return total
