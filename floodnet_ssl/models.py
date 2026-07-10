"""Model factory and lightweight network skeletons for FloodNet experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .constants import CLASS_NAMES
from .state_factorization import (
    CONDITIONAL_STATE_CHANNELS,
    compose_hierarchical_probabilities,
    fuse_semantic_and_hierarchical_logits,
)


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


class MultiScaleFactorDecoder(torch.nn.Module):
    """Fuse all SegFormer encoder stages without changing its semantic decoder."""

    def __init__(
        self,
        hidden_sizes: Sequence[int],
        *,
        decoder_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one encoder stage")
        if decoder_channels <= 0:
            raise ValueError("decoder_channels must be positive")
        self.projections = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Conv2d(int(channels), decoder_channels, kernel_size=1),
                    torch.nn.BatchNorm2d(decoder_channels),
                    torch.nn.GELU(),
                )
                for channels in hidden_sizes
            ]
        )
        self.fusion = torch.nn.Sequential(
            torch.nn.Conv2d(
                decoder_channels * len(hidden_sizes),
                decoder_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(decoder_channels),
            torch.nn.GELU(),
            torch.nn.Dropout2d(dropout),
        )

    def forward(self, hidden_states: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(hidden_states) != len(self.projections):
            raise ValueError(
                f"Expected {len(self.projections)} encoder stages, got {len(hidden_states)}"
            )
        if any(state.ndim != 4 for state in hidden_states):
            raise ValueError("SegFormer factorization requires NCHW encoder hidden states")
        target_size = hidden_states[0].shape[-2:]
        projected = []
        for state, projection in zip(hidden_states, self.projections):
            feature = projection(state)
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(
                    feature, size=target_size, mode="bilinear", align_corners=False
                )
            projected.append(feature)
        return self.fusion(torch.cat(projected, dim=1))


class ObjectConditionedStateHead(torch.nn.Module):
    """Predict separate building and road states conditioned on object posterior."""

    def __init__(
        self,
        channels: int,
        *,
        conditional: bool,
        detach_object_posterior: bool,
    ) -> None:
        super().__init__()
        self.conditional = conditional
        self.detach_object_posterior = detach_object_posterior
        state_channels = CONDITIONAL_STATE_CHANNELS if conditional else 2
        self.object_conditioner = (
            torch.nn.Conv2d(2, channels, kernel_size=1) if conditional else None
        )
        self.refinement = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(channels),
            torch.nn.GELU(),
        )
        self.classifier = torch.nn.Conv2d(channels, state_channels, kernel_size=1)

    def forward(
        self, features: torch.Tensor, object_logits: torch.Tensor
    ) -> torch.Tensor:
        if self.conditional:
            object_posterior = torch.softmax(object_logits, dim=1)[:, 1:3]
            if self.detach_object_posterior:
                object_posterior = object_posterior.detach()
            assert self.object_conditioner is not None
            features = features + self.object_conditioner(object_posterior)
        return self.classifier(self.refinement(features))


class ConditionalStateFactorizationWrapper(torch.nn.Module):
    """SegFormer with a feature-level conditional object/state factorization path."""

    def __init__(
        self,
        base_model: torch.nn.Module,
        *,
        hidden_sizes: Sequence[int],
        decoder_channels: int = 64,
        dropout: float = 0.1,
        state_mode: str = "conditional",
        detach_object_posterior: bool = False,
        fusion_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if not hasattr(base_model, "segformer") or not hasattr(base_model, "decode_head"):
            raise TypeError("Feature factorization requires a SegFormer segmentation model")
        if state_mode not in {"shared", "conditional"}:
            raise ValueError("state_mode must be 'shared' or 'conditional'")
        if not 0.0 <= fusion_weight <= 1.0:
            raise ValueError("fusion_weight must be in [0, 1]")
        self.base_model = base_model
        self.factor_decoder = MultiScaleFactorDecoder(
            hidden_sizes,
            decoder_channels=decoder_channels,
            dropout=dropout,
        )
        self.object_head = torch.nn.Conv2d(
            decoder_channels, AUXILIARY_HEAD_NUM_LABELS["object"], kernel_size=1
        )
        self.state_head = ObjectConditionedStateHead(
            decoder_channels,
            conditional=state_mode == "conditional",
            detach_object_posterior=detach_object_posterior,
        )
        self.fusion_weight = fusion_weight

    def forward(self, image: torch.Tensor) -> SegmentationModelOutput:
        encoder_output = self.base_model.segformer(
            pixel_values=image,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = encoder_output.hidden_states
        if hidden_states is None:
            raise RuntimeError("SegFormer encoder did not return hidden states")
        semantic_direct = self.base_model.decode_head(hidden_states)
        factor_features = self.factor_decoder(hidden_states)
        object_logits = self.object_head(factor_features)
        state_logits = self.state_head(factor_features, object_logits)
        if object_logits.shape[-2:] != semantic_direct.shape[-2:]:
            object_for_composition = F.interpolate(
                object_logits,
                size=semantic_direct.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            state_for_composition = F.interpolate(
                state_logits,
                size=semantic_direct.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        else:
            object_for_composition = object_logits
            state_for_composition = state_logits
        hierarchical = compose_hierarchical_probabilities(
            object_for_composition, state_for_composition
        )
        logits = fuse_semantic_and_hierarchical_logits(
            semantic_direct,
            hierarchical,
            fusion_weight=self.fusion_weight,
        )
        return SegmentationModelOutput(
            logits=logits,
            auxiliary={
                "semantic_direct": semantic_direct,
                "object": object_logits,
                "state": state_logits,
                "hierarchical": hierarchical,
            },
        )


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
        factorization = model_config.get("state_factorization", {})
        if factorization and bool(factorization.get("enabled", False)):
            feature_source = str(
                factorization.get("feature_source", "encoder_multiscale")
            )
            if feature_source == "logits":
                return LogitAuxiliaryWrapper(
                    model,
                    num_labels=int(model_config.get("num_labels", len(CLASS_NAMES))),
                    enabled_auxiliary_heads=tuple(
                        model_config.get("auxiliary_heads", ("object", "state"))
                    ),
                )
            hidden_sizes = tuple(int(value) for value in model.config.hidden_sizes)
            return ConditionalStateFactorizationWrapper(
                model,
                hidden_sizes=hidden_sizes,
                decoder_channels=int(factorization.get("decoder_channels", 64)),
                dropout=float(factorization.get("dropout", 0.1)),
                state_mode=str(factorization.get("state_mode", "conditional")),
                detach_object_posterior=bool(
                    factorization.get("detach_object_posterior", False)
                ),
                fusion_weight=float(factorization.get("fusion_weight", 0.0)),
            )
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

