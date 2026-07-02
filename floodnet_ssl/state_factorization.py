"""Object/state factorization utilities for FloodNet semantic labels."""

from __future__ import annotations

from typing import Mapping, Any

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
    """Compose object and state probabilities into FloodNet's ten classes."""

    if object_logits.ndim != 4 or object_logits.shape[1] != NUM_OBJECT_CLASSES:
        raise ValueError(
            f"Expected object logits [B,{NUM_OBJECT_CLASSES},H,W], got {object_logits.shape}"
        )
    if state_logits.ndim != 4 or state_logits.shape[1] != NUM_STATE_CLASSES:
        raise ValueError(
            f"Expected state logits [B,{NUM_STATE_CLASSES},H,W], got {state_logits.shape}"
        )
    if object_logits.shape[0] != state_logits.shape[0] or object_logits.shape[-2:] != state_logits.shape[-2:]:
        raise ValueError("Object and state logits must share batch and spatial dimensions")

    object_prob = torch.softmax(object_logits, dim=1)
    state_prob = torch.softmax(state_logits, dim=1)
    composed = object_prob.new_zeros(
        (object_prob.shape[0], NUM_CLASSES, object_prob.shape[2], object_prob.shape[3])
    )
    composed[:, 0] = object_prob[:, 0]
    composed[:, 1] = object_prob[:, 1] * state_prob[:, FLOODED_STATE]
    composed[:, 2] = object_prob[:, 1] * state_prob[:, NON_FLOODED_STATE]
    composed[:, 3] = object_prob[:, 2] * state_prob[:, FLOODED_STATE]
    composed[:, 4] = object_prob[:, 2] * state_prob[:, NON_FLOODED_STATE]
    composed[:, 5] = object_prob[:, 3]
    composed[:, 6] = object_prob[:, 4]
    composed[:, 7] = object_prob[:, 5]
    composed[:, 8] = object_prob[:, 6]
    composed[:, 9] = object_prob[:, 7]
    return composed.clamp_min(1e-8)


def js_divergence_loss(
    semantic_logits: torch.Tensor,
    hierarchical_probabilities: torch.Tensor,
    target: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Jensen-Shannon consistency on valid pixels."""

    semantic_prob = torch.softmax(semantic_logits, dim=1).clamp_min(1e-8)
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
    return (0.5 * (kl_semantic + kl_hierarchical))[valid].mean()


def state_factorization_loss(
    *,
    semantic_logits: torch.Tensor,
    auxiliary: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    config: Mapping[str, Any] | None,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Compute object/state auxiliary losses when the module is enabled."""

    module_config = dict(config or {})
    if not bool(module_config.get("enabled", False)):
        return semantic_logits.sum() * 0.0
    if "object" not in auxiliary or "state" not in auxiliary:
        raise ValueError("state_factorization requires 'object' and 'state' auxiliary logits")

    object_logits = auxiliary["object"]
    state_logits = auxiliary["state"]
    if object_logits.shape[-2:] != target.shape[-2:]:
        object_logits = F.interpolate(object_logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
    if state_logits.shape[-2:] != target.shape[-2:]:
        state_logits = F.interpolate(state_logits, size=target.shape[-2:], mode="bilinear", align_corners=False)

    object_target = semantic_to_object_target(target, ignore_index=ignore_index)
    state_target = semantic_to_state_target(target, ignore_index=ignore_index)
    object_loss = F.cross_entropy(object_logits, object_target, ignore_index=ignore_index)
    state_loss = F.cross_entropy(state_logits, state_target, ignore_index=ignore_index)
    hierarchical = compose_hierarchical_probabilities(object_logits, state_logits)
    consistency = js_divergence_loss(
        semantic_logits,
        hierarchical,
        target,
        ignore_index=ignore_index,
    )
    return (
        float(module_config.get("object_weight", 0.25)) * object_loss
        + float(module_config.get("state_weight", 0.25)) * state_loss
        + float(module_config.get("consistency_weight", 0.1)) * consistency
    )
