"""Boundary supervision and boundary-guided context refinement."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F

from .constants import IGNORE_INDEX


def semantic_boundary_target(
    target: torch.Tensor,
    *,
    width: int = 3,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Create binary semantic-boundary targets from class-index masks."""

    if target.ndim != 3:
        raise ValueError(f"Expected target [B,H,W], got {target.shape}")
    if width <= 0:
        raise ValueError("Boundary width must be positive")
    valid = target != ignore_index
    boundary = torch.zeros_like(target, dtype=torch.bool)

    horizontal_valid = valid[:, :, 1:] & valid[:, :, :-1]
    horizontal = (target[:, :, 1:] != target[:, :, :-1]) & horizontal_valid
    boundary[:, :, 1:] |= horizontal
    boundary[:, :, :-1] |= horizontal

    vertical_valid = valid[:, 1:, :] & valid[:, :-1, :]
    vertical = (target[:, 1:, :] != target[:, :-1, :]) & vertical_valid
    boundary[:, 1:, :] |= vertical
    boundary[:, :-1, :] |= vertical

    if width > 1:
        radius = width // 2
        boundary = (
            F.max_pool2d(
                boundary.float().unsqueeze(1),
                kernel_size=2 * radius + 1,
                stride=1,
                padding=radius,
            ).squeeze(1)
            > 0
        )
    return boundary.float()


def soft_semantic_boundary(semantic_logits: torch.Tensor) -> torch.Tensor:
    """Approximate a differentiable semantic edge map from class probabilities."""

    probabilities = torch.softmax(semantic_logits, dim=1)
    dx = torch.abs(probabilities[:, :, :, 1:] - probabilities[:, :, :, :-1]).mean(dim=1, keepdim=True)
    dy = torch.abs(probabilities[:, :, 1:, :] - probabilities[:, :, :-1, :]).mean(dim=1, keepdim=True)
    edge = probabilities.new_zeros((probabilities.shape[0], 1, probabilities.shape[2], probabilities.shape[3]))
    edge[:, :, :, 1:] = torch.maximum(edge[:, :, :, 1:], dx)
    edge[:, :, :, :-1] = torch.maximum(edge[:, :, :, :-1], dx)
    edge[:, :, 1:, :] = torch.maximum(edge[:, :, 1:, :], dy)
    edge[:, :, :-1, :] = torch.maximum(edge[:, :, :-1, :], dy)
    return edge.clamp(0.0, 1.0)


def binary_dice_loss(logits: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    target = target.to(dtype=probabilities.dtype)
    dims = (0, 2, 3)
    intersection = torch.sum(probabilities * target, dim=dims)
    denominator = torch.sum(probabilities + target, dim=dims)
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - dice.mean()


def boundary_context_loss(
    *,
    semantic_logits: torch.Tensor,
    auxiliary: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    config: Mapping[str, Any] | None,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Compute boundary target and semantic-boundary consistency losses."""

    module_config = dict(config or {})
    if not bool(module_config.get("enabled", False)):
        return semantic_logits.sum() * 0.0
    if "boundary" not in auxiliary:
        raise ValueError("boundary_context requires 'boundary' auxiliary logits")
    boundary_logits = auxiliary["boundary"]
    if boundary_logits.shape[-2:] != target.shape[-2:]:
        boundary_logits = F.interpolate(boundary_logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
    boundary_target = semantic_boundary_target(
        target,
        width=int(module_config.get("target_width", 3)),
        ignore_index=ignore_index,
    ).unsqueeze(1)
    valid = (target != ignore_index).unsqueeze(1)
    positive = boundary_target[valid].sum()
    negative = valid.sum() - positive
    pos_weight = (negative / positive.clamp_min(1.0)).clamp(max=float(module_config.get("max_pos_weight", 20.0)))
    bce = F.binary_cross_entropy_with_logits(
        boundary_logits[valid],
        boundary_target[valid],
        pos_weight=pos_weight,
    )
    dice = binary_dice_loss(boundary_logits * valid, boundary_target * valid)
    semantic_edge = soft_semantic_boundary(semantic_logits).detach()
    consistency = torch.abs(torch.sigmoid(boundary_logits) - semantic_edge)[valid].mean()
    return (
        float(module_config.get("bce_weight", 1.0)) * bce
        + float(module_config.get("dice_weight", 1.0)) * dice
        + float(module_config.get("consistency_weight", 0.1)) * consistency
    )


def refine_logits_with_boundary_context(
    semantic_logits: torch.Tensor,
    boundary_logits: torch.Tensor,
    *,
    strength: float = 0.25,
    kernel_size: int = 5,
) -> torch.Tensor:
    """Use boundary confidence to aggregate context inside non-boundary regions."""

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("Boundary context kernel_size must be a positive odd integer")
    gate = torch.sigmoid(boundary_logits)
    context = F.avg_pool2d(
        semantic_logits,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    interior_weight = 1.0 - gate
    return semantic_logits + float(strength) * interior_weight * (context - semantic_logits)
