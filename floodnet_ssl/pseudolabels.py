"""Pseudo-label scoring and EMA utilities for structure-aware SSL."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PseudoLabelBatch:
    labels: torch.Tensor
    confidence: torch.Tensor
    score: torch.Tensor
    mask: torch.Tensor
    threshold: float
    coverage: float


@torch.no_grad()
def update_ema_teacher(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    *,
    decay: float,
) -> None:
    """Update teacher parameters with exponential moving average weights."""

    if not 0.0 <= decay < 1.0:
        raise ValueError("EMA decay must be in [0, 1)")
    teacher_state = dict(teacher.named_parameters())
    for name, student_parameter in student.named_parameters():
        if name in teacher_state:
            teacher_state[name].mul_(decay).add_(student_parameter.detach(), alpha=1.0 - decay)
    teacher_buffers = dict(teacher.named_buffers())
    for name, student_buffer in student.named_buffers():
        if name in teacher_buffers and torch.is_floating_point(teacher_buffers[name]):
            teacher_buffers[name].mul_(decay).add_(student_buffer.detach(), alpha=1.0 - decay)


def soft_boundary_from_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    """Differentiable semantic boundary strength from probability maps."""

    if probabilities.ndim != 4:
        raise ValueError(f"Expected probabilities [B,C,H,W], got {probabilities.shape}")
    dx = torch.abs(probabilities[:, :, :, 1:] - probabilities[:, :, :, :-1]).mean(dim=1)
    dy = torch.abs(probabilities[:, :, 1:, :] - probabilities[:, :, :-1, :]).mean(dim=1)
    edge = probabilities.new_zeros((probabilities.shape[0], probabilities.shape[2], probabilities.shape[3]))
    edge[:, :, 1:] = torch.maximum(edge[:, :, 1:], dx)
    edge[:, :, :-1] = torch.maximum(edge[:, :, :-1], dx)
    edge[:, 1:, :] = torch.maximum(edge[:, 1:, :], dy)
    edge[:, :-1, :] = torch.maximum(edge[:, :-1, :], dy)
    return edge.clamp(0.0, 1.0)


def multiview_consistency_score(primary_logits: torch.Tensor, secondary_logits: torch.Tensor) -> torch.Tensor:
    """Score agreement between two teacher views by probability dot product."""

    primary = torch.softmax(primary_logits, dim=1)
    secondary = torch.softmax(secondary_logits, dim=1)
    if primary.shape != secondary.shape:
        raise ValueError("Teacher view logits must have the same shape")
    return torch.sum(primary * secondary, dim=1).clamp(0.0, 1.0)


def boundary_stability_score(primary_logits: torch.Tensor, secondary_logits: torch.Tensor) -> torch.Tensor:
    """High score where semantic boundaries are stable across two views."""

    primary_edge = soft_boundary_from_probabilities(torch.softmax(primary_logits, dim=1))
    secondary_edge = soft_boundary_from_probabilities(torch.softmax(secondary_logits, dim=1))
    return torch.exp(-torch.abs(primary_edge - secondary_edge)).clamp(0.0, 1.0)


def region_consistency_score(labels: torch.Tensor, *, kernel_size: int = 5) -> torch.Tensor:
    """Score local label agreement around each pseudo-labeled pixel."""

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    num_classes = int(labels.max().item()) + 1 if labels.numel() else 1
    one_hot = F.one_hot(labels.clamp_min(0), num_classes=num_classes).permute(0, 3, 1, 2).float()
    pooled = F.avg_pool2d(one_hot, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return pooled.gather(1, labels.unsqueeze(1).clamp(max=num_classes - 1)).squeeze(1).clamp(0.0, 1.0)


def structure_score(
    primary_logits: torch.Tensor,
    secondary_logits: torch.Tensor | None = None,
    *,
    confidence_weight: float = 1.0,
    multiview_weight: float = 0.25,
    boundary_weight: float = 0.25,
    region_weight: float = 0.25,
    region_kernel_size: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return labels, confidence, and combined structure-aware score."""

    probabilities = torch.softmax(primary_logits, dim=1)
    confidence, labels = torch.max(probabilities, dim=1)
    if secondary_logits is None:
        secondary_logits = primary_logits
    multiview = multiview_consistency_score(primary_logits, secondary_logits)
    boundary = boundary_stability_score(primary_logits, secondary_logits)
    region = region_consistency_score(labels, kernel_size=region_kernel_size)
    total_weight = confidence_weight + multiview_weight + boundary_weight + region_weight
    if total_weight <= 0:
        raise ValueError("At least one pseudo-label score weight must be positive")
    score = (
        confidence_weight * confidence
        + multiview_weight * multiview
        + boundary_weight * boundary
        + region_weight * region
    ) / total_weight
    return labels, confidence, score.clamp(0.0, 1.0)


def threshold_for_coverage(score: torch.Tensor, coverage: float) -> float:
    """Choose a score threshold that keeps approximately the requested coverage."""

    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    flat = score.reshape(-1)
    keep = max(1, int(round(float(flat.numel()) * coverage)))
    values, _ = torch.sort(flat, descending=True)
    return float(values[min(keep - 1, values.numel() - 1)])


def make_pseudo_labels(
    primary_logits: torch.Tensor,
    secondary_logits: torch.Tensor | None = None,
    *,
    threshold: float = 0.8,
    matched_coverage: float | None = None,
    confidence_weight: float = 1.0,
    multiview_weight: float = 0.25,
    boundary_weight: float = 0.25,
    region_weight: float = 0.25,
    region_kernel_size: int = 5,
) -> PseudoLabelBatch:
    labels, confidence, score = structure_score(
        primary_logits,
        secondary_logits,
        confidence_weight=confidence_weight,
        multiview_weight=multiview_weight,
        boundary_weight=boundary_weight,
        region_weight=region_weight,
        region_kernel_size=region_kernel_size,
    )
    effective_threshold = (
        threshold_for_coverage(score, matched_coverage)
        if matched_coverage is not None
        else float(threshold)
    )
    mask = score >= effective_threshold
    coverage = float(mask.float().mean().item())
    return PseudoLabelBatch(
        labels=labels,
        confidence=confidence,
        score=score,
        mask=mask,
        threshold=effective_threshold,
        coverage=coverage,
    )
