"""Model factory and lightweight network skeletons for FloodNet experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
from typing import Any, Mapping, Sequence

import torch

from .constants import CLASS_NAMES


AUXILIARY_HEAD_NUM_LABELS = {
    "object": 8,
    "state": 2,
    "boundary": 1,
    "relation": 2,
}


@dataclass(frozen=True)
class SegmentationModelOutput:
    """Unified model output contract used by training and inference utilities."""

    logits: torch.Tensor
    auxiliary: Mapping[str, torch.Tensor] = field(default_factory=dict)


class MissingSegFormerDependency(RuntimeError):
    """Raised when the configured SegFormer backend is unavailable."""


class UnsupportedModelError(ValueError):
    """Raised when a config requests an unknown model family."""


class ConvPredictionHead(torch.nn.Module):
    """Minimal prediction head used by dependency-free skeleton tests."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)


class MultiHeadSegmentationModel(torch.nn.Module):
    """Lightweight multi-head container with disabled-by-default auxiliary heads.

    This is an interface skeleton rather than the final hierarchy/boundary/relation
    method. It lets future modules share one output contract while Week 1 keeps the
    supervised semantic path unchanged.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        *,
        feature_channels: int,
        num_labels: int = len(CLASS_NAMES),
        enabled_auxiliary_heads: Sequence[str] = (),
    ) -> None:
        super().__init__()
        unknown = sorted(set(enabled_auxiliary_heads) - set(AUXILIARY_HEAD_NUM_LABELS))
        if unknown:
            raise ValueError(f"Unknown auxiliary head(s): {', '.join(unknown)}")
        self.backbone = backbone
        self.semantic_head = ConvPredictionHead(feature_channels, num_labels)
        self.auxiliary_heads = torch.nn.ModuleDict(
            {
                name: ConvPredictionHead(feature_channels, AUXILIARY_HEAD_NUM_LABELS[name])
                for name in enabled_auxiliary_heads
            }
        )

    def forward(self, image: torch.Tensor) -> SegmentationModelOutput:
        features = self.backbone(image)
        if not torch.is_tensor(features):
            raise TypeError("MultiHeadSegmentationModel backbone must return a tensor")
        logits = self.semantic_head(features)
        auxiliary = {
            name: head(features) for name, head in self.auxiliary_heads.items()
        }
        return SegmentationModelOutput(logits=logits, auxiliary=auxiliary)


class LogitAuxiliaryWrapper(torch.nn.Module):
    """Attach lightweight auxiliary heads to a segmentation model's logits."""

    def __init__(
        self,
        base_model: torch.nn.Module,
        *,
        num_labels: int,
        enabled_auxiliary_heads: Sequence[str],
    ) -> None:
        super().__init__()
        unknown = sorted(set(enabled_auxiliary_heads) - set(AUXILIARY_HEAD_NUM_LABELS))
        if unknown:
            raise ValueError(f"Unknown auxiliary head(s): {', '.join(unknown)}")
        self.base_model = base_model
        self.auxiliary_heads = torch.nn.ModuleDict(
            {
                name: ConvPredictionHead(num_labels, AUXILIARY_HEAD_NUM_LABELS[name])
                for name in enabled_auxiliary_heads
            }
        )

    def forward(self, image: torch.Tensor) -> SegmentationModelOutput:
        base_output = self.base_model(image)
        logits = extract_logits(base_output)
        auxiliary = {
            name: head(logits) for name, head in self.auxiliary_heads.items()
        }
        if isinstance(base_output, SegmentationModelOutput):
            auxiliary = {**dict(base_output.auxiliary), **auxiliary}
        return SegmentationModelOutput(logits=logits, auxiliary=auxiliary)


def segformer_dependency_status() -> dict[str, bool]:
    return {
        package: importlib.util.find_spec(package) is not None
        for package in ("transformers", "safetensors")
    }


def require_segformer_dependencies(*, pretrained: bool = True) -> None:
    status = segformer_dependency_status()
    required = ("transformers", "safetensors") if pretrained else ("transformers",)
    missing = [name for name in required if not status[name]]
    if missing:
        raise MissingSegFormerDependency(
            "Missing SegFormer dependencies: "
            + ", ".join(missing)
            + ". Install only after user approval."
        )


def build_segformer_b0(model_config: Mapping[str, Any]) -> torch.nn.Module:
    """Build SegFormer-B0 without importing transformers at module import time."""

    pretrained = bool(model_config.get("pretrained", True))
    require_segformer_dependencies(pretrained=pretrained)
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    num_labels = int(model_config.get("num_labels", len(CLASS_NAMES)))
    id2label = {index: name for index, name in enumerate(CLASS_NAMES)}
    label2id = {name: index for index, name in id2label.items()}
    if pretrained:
        model_name = str(model_config["pretrained_model_name_or_path"])
        return SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            local_files_only=bool(model_config.get("local_files_only", True)),
        )

    configuration = SegformerConfig(
        num_channels=3,
        num_encoder_blocks=4,
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        hidden_sizes=[32, 64, 160, 256],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        num_attention_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        decoder_hidden_size=256,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    return SegformerForSemanticSegmentation(configuration)


def build_model(model_config: Mapping[str, Any]) -> torch.nn.Module:
    """Build the configured model without importing optional backends eagerly."""

    name = str(model_config.get("name", "segformer_b0"))
    if name == "segformer_b0":
        model = build_segformer_b0(model_config)
        enabled_auxiliary_heads = tuple(model_config.get("auxiliary_heads", ()))
        if enabled_auxiliary_heads:
            model = LogitAuxiliaryWrapper(
                model,
                num_labels=int(model_config.get("num_labels", len(CLASS_NAMES))),
                enabled_auxiliary_heads=enabled_auxiliary_heads,
            )
        return model
    raise UnsupportedModelError(f"Unsupported model.name: {name}")


def extract_logits(output: object) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, SegmentationModelOutput):
        return output.logits
    if hasattr(output, "logits") and torch.is_tensor(output.logits):
        return output.logits
    if isinstance(output, dict) and torch.is_tensor(output.get("logits")):
        return output["logits"]
    raise TypeError("Model output must be a tensor or expose tensor logits")

