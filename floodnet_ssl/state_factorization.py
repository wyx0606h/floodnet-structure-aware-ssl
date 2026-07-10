"""Conditional object/state factorization for FloodNet semantic labels."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F

from .constants import IGNORE_INDEX, NUM_CLASSES

OBJECT_CLASS_NAMES = (
    "background",
    "building",
    "road",
    "water",
    "tree",
    "vehicle",
    "pool",
    "grass",
)
NUM_OBJECT_CLASSES = len(OBJECT_CLASS_NAMES)
NUM_STATE_CLASSES = 2
NON_FLOODED_STATE = 0
FLOODED_STATE = 1
CONDITIONAL_STATE_CHANNELS = 4

SEMANTIC_TO_OBJECT = torch.tensor([0, 1, 1, 2, 2, 3, 4, 5, 6, 7], dtype=torch.long)


def semantic_to_object_target(
    target: torch.Tensor, *, ignore_index: int = IGNORE_INDEX
) -> torch.Tensor:
    """Map ten FloodNet semantic labels to eight object-identity labels."""

    mapping = SEMANTIC_TO_OBJECT.to(device=target.device)
    result = torch.full_like(target, ignore_index)
    valid = (target >= 0) & (target < NUM_CLASSES) & (target != ignore_index)
    result[valid] = mapping[target[valid]]
    return result


def semantic_to_state_target(
    target: torch.Tensor, *, ignore_index: int = IGNORE_INDEX
) -> torch.Tensor:
    """Map building/road labels to flooded/non-flooded state labels."""

    result = torch.full_like(target, ignore_index)
    result[(target == 1) | (target == 3)] = FLOODED_STATE
    result[(target == 2) | (target == 4)] = NON_FLOODED_STATE
    return result


def compose_hierarchical_probabilities(
    object_logits: torch.Tensor, state_logits: torch.Tensor
) -> torch.Tensor:
    """Compose ``P(object|x) P(state|object,x)`` into ten-class probabilities.

    Two state channels reproduce the original shared-state ablation. Four channels
    represent building-specific and road-specific two-way state experts.
    """

    if object_logits.ndim != 4 or object_logits.shape[1] != NUM_OBJECT_CLASSES:
        raise ValueError(
            f"Expected object logits [B,{NUM_OBJECT_CLASSES},H,W], got {object_logits.shape}"
        )
    if state_logits.ndim != 4 or state_logits.shape[1] not in {
        NUM_STATE_CLASSES,
        CONDITIONAL_STATE_CHANNELS,
    }:
        raise ValueError(
            "Expected shared state logits [B,2,H,W] or conditional state logits "
            f"[B,4,H,W], got {state_logits.shape}"
        )
    if (
        object_logits.shape[0] != state_logits.shape[0]
        or object_logits.shape[-2:] != state_logits.shape[-2:]
    ):
        raise ValueError("Object and state logits must share batch and spatial dimensions")

    object_prob = torch.softmax(object_logits, dim=1)
    if state_logits.shape[1] == NUM_STATE_CLASSES:
        building_state = road_state = torch.softmax(state_logits, dim=1)
    else:
        building_state = torch.softmax(state_logits[:, 0:2], dim=1)
        road_state = torch.softmax(state_logits[:, 2:4], dim=1)

    composed = object_prob.new_zeros(
        (object_prob.shape[0], NUM_CLASSES, object_prob.shape[2], object_prob.shape[3])
    )
    composed[:, 0] = object_prob[:, 0]
    composed[:, 1] = object_prob[:, 1] * building_state[:, FLOODED_STATE]
    composed[:, 2] = object_prob[:, 1] * building_state[:, NON_FLOODED_STATE]
    composed[:, 3] = object_prob[:, 2] * road_state[:, FLOODED_STATE]
    composed[:, 4] = object_prob[:, 2] * road_state[:, NON_FLOODED_STATE]
    composed[:, 5] = object_prob[:, 3]
    composed[:, 6] = object_prob[:, 4]
    composed[:, 7] = object_prob[:, 5]
    composed[:, 8] = object_prob[:, 6]
    composed[:, 9] = object_prob[:, 7]
    return composed


def fuse_semantic_and_hierarchical_logits(
    semantic_logits: torch.Tensor,
    hierarchical_probabilities: torch.Tensor,
    *,
    fusion_weight: float,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Fuse flat and factorized predictions as a log-space product of experts."""

    if not 0.0 <= fusion_weight <= 1.0:
        raise ValueError("fusion_weight must be in [0, 1]")
    if fusion_weight == 0.0:
        return semantic_logits
    if semantic_logits.shape != hierarchical_probabilities.shape:
        raise ValueError("Semantic logits and hierarchical probabilities must match")
    semantic_log_prob = torch.log_softmax(semantic_logits, dim=1)
    hierarchical_log_prob = hierarchical_probabilities.clamp_min(epsilon).log()
    return (1.0 - fusion_weight) * semantic_log_prob + fusion_weight * hierarchical_log_prob


def _safe_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    ignore_index: int,
) -> torch.Tensor:
    if not torch.any(target != ignore_index):
        return logits.sum() * 0.0
    return F.cross_entropy(logits, target, ignore_index=ignore_index)


def _masked_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    valid = target != ignore_index
    if not torch.any(valid):
        return logits.sum() * 0.0
    safe_target = target.clamp(min=0, max=num_classes - 1)
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(safe_target, num_classes=num_classes).permute(0, 3, 1, 2)
    one_hot = one_hot.to(probabilities.dtype)
    valid_float = valid.unsqueeze(1).to(probabilities.dtype)
    probabilities = probabilities * valid_float
    one_hot = one_hot * valid_float
    dims = (0, 2, 3)
    intersection = torch.sum(probabilities * one_hot, dim=dims)
    denominator = torch.sum(probabilities + one_hot, dim=dims)
    present = torch.sum(one_hot, dim=dims) > 0
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - dice[present].mean()


def _conditional_state_loss(
    state_logits: torch.Tensor,
    semantic_target: torch.Tensor,
    *,
    dice_weight: float,
    ignore_index: int,
) -> torch.Tensor:
    if state_logits.ndim != 4 or state_logits.shape[1] not in {
        NUM_STATE_CLASSES,
        CONDITIONAL_STATE_CHANNELS,
    }:
        raise ValueError("state logits must have 2 shared or 4 conditional channels")
    state_target = semantic_to_state_target(semantic_target, ignore_index=ignore_index)
    if state_logits.shape[1] == NUM_STATE_CLASSES:
        return _safe_cross_entropy(
            state_logits, state_target, ignore_index=ignore_index
        ) + dice_weight * _masked_dice_loss(
            state_logits,
            state_target,
            num_classes=NUM_STATE_CLASSES,
            ignore_index=ignore_index,
        )

    losses: list[torch.Tensor] = []
    expert_specs = (
        (state_logits[:, 0:2], (semantic_target == 1) | (semantic_target == 2)),
        (state_logits[:, 2:4], (semantic_target == 3) | (semantic_target == 4)),
    )
    for expert_logits, expert_mask in expert_specs:
        expert_target = torch.full_like(state_target, ignore_index)
        expert_target[expert_mask] = state_target[expert_mask]
        if torch.any(expert_mask):
            losses.append(
                _safe_cross_entropy(
                    expert_logits, expert_target, ignore_index=ignore_index
                )
                + dice_weight
                * _masked_dice_loss(
                    expert_logits,
                    expert_target,
                    num_classes=NUM_STATE_CLASSES,
                    ignore_index=ignore_index,
                )
            )
    if not losses:
        return state_logits.sum() * 0.0
    return torch.stack(losses).mean()


def js_divergence_loss(
    semantic_logits: torch.Tensor,
    hierarchical_probabilities: torch.Tensor,
    target: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
    reduction: str = "class_mean",
) -> torch.Tensor:
    """Jensen-Shannon consistency with pixel- or target-class-balanced reduction."""

    semantic_prob = torch.softmax(semantic_logits, dim=1).clamp_min(1e-8)
    hierarchical_probabilities = hierarchical_probabilities.clamp_min(1e-8)
    if semantic_prob.shape != hierarchical_probabilities.shape:
        raise ValueError("Semantic and hierarchical probabilities must have identical shape")
    valid = target != ignore_index
    if not torch.any(valid):
        return semantic_logits.sum() * 0.0
    mixture = 0.5 * (semantic_prob + hierarchical_probabilities)
    kl_semantic = torch.sum(
        semantic_prob * (semantic_prob.log() - mixture.log()), dim=1
    )
    kl_hierarchical = torch.sum(
        hierarchical_probabilities
        * (hierarchical_probabilities.log() - mixture.log()),
        dim=1,
    )
    pixel_js = 0.5 * (kl_semantic + kl_hierarchical)
    if reduction == "pixel_mean":
        return pixel_js[valid].mean()
    if reduction == "class_mean":
        class_losses = [
            pixel_js[(target == class_id) & valid].mean()
            for class_id in range(NUM_CLASSES)
            if torch.any((target == class_id) & valid)
        ]
        return torch.stack(class_losses).mean()
    raise ValueError("consistency_reduction must be 'pixel_mean' or 'class_mean'")


def state_factorization_terms(
    *,
    semantic_logits: torch.Tensor,
    auxiliary: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    config: Mapping[str, Any] | None,
    ignore_index: int = IGNORE_INDEX,
) -> dict[str, torch.Tensor]:
    """Return weighted total and inspectable object/state/consistency terms."""

    module_config = dict(config or {})
    zero = semantic_logits.sum() * 0.0
    if not bool(module_config.get("enabled", False)):
        return {"total": zero, "object": zero, "state": zero, "consistency": zero}
    if "object" not in auxiliary or "state" not in auxiliary:
        raise ValueError("state_factorization requires 'object' and 'state' auxiliary logits")

    object_logits = auxiliary["object"]
    state_logits = auxiliary["state"]
    direct_semantic = auxiliary.get("semantic_direct", semantic_logits)
    target_size = target.shape[-2:]
    if object_logits.shape[-2:] != target_size:
        object_logits = F.interpolate(
            object_logits, size=target_size, mode="bilinear", align_corners=False
        )
    if state_logits.shape[-2:] != target_size:
        state_logits = F.interpolate(
            state_logits, size=target_size, mode="bilinear", align_corners=False
        )
    if direct_semantic.shape[-2:] != target_size:
        direct_semantic = F.interpolate(
            direct_semantic, size=target_size, mode="bilinear", align_corners=False
        )

    object_target = semantic_to_object_target(target, ignore_index=ignore_index)
    object_loss = _safe_cross_entropy(
        object_logits, object_target, ignore_index=ignore_index
    ) + float(module_config.get("object_dice_weight", 1.0)) * _masked_dice_loss(
        object_logits,
        object_target,
        num_classes=NUM_OBJECT_CLASSES,
        ignore_index=ignore_index,
    )
    state_loss = _conditional_state_loss(
        state_logits,
        target,
        dice_weight=float(module_config.get("state_dice_weight", 1.0)),
        ignore_index=ignore_index,
    )
    hierarchical = compose_hierarchical_probabilities(object_logits, state_logits)
    consistency = js_divergence_loss(
        direct_semantic,
        hierarchical,
        target,
        ignore_index=ignore_index,
        reduction=str(module_config.get("consistency_reduction", "class_mean")),
    )
    total = (
        float(module_config.get("object_weight", 0.25)) * object_loss
        + float(module_config.get("state_weight", 0.25)) * state_loss
        + float(module_config.get("consistency_weight", 0.1)) * consistency
    )
    return {
        "total": total,
        "object": object_loss,
        "state": state_loss,
        "consistency": consistency,
    }


def state_factorization_loss(
    *,
    semantic_logits: torch.Tensor,
    auxiliary: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    config: Mapping[str, Any] | None,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Compute the config-gated factorization loss."""

    return state_factorization_terms(
        semantic_logits=semantic_logits,
        auxiliary=auxiliary,
        target=target,
        config=config,
        ignore_index=ignore_index,
    )["total"]
