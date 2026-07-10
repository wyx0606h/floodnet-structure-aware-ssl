"""Configuration loading and validation for supervised FloodNet runs."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is incomplete or inconsistent."""


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ConfigError(f"Top-level YAML value must be a mapping: {config_path}")
    config = _expand_environment(copy.deepcopy(loaded))
    config["_config_path"] = str(config_path)
    validate_supervised_config(config)
    return config


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if "$" in expanded:
            raise ConfigError(f"Unresolved environment variable in value: {value}")
        return expanded
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"Config section '{key}' must be a mapping")
    return value


def _require_positive_int(section: Mapping[str, Any], key: str) -> int:
    value = section.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"'{key}' must be a positive integer, got {value!r}")
    return value


def validate_supervised_config(config: Mapping[str, Any]) -> None:
    experiment = _require_mapping(config, "experiment")
    data = _require_mapping(config, "data")
    model = _require_mapping(config, "model")
    training = _require_mapping(config, "training")
    evaluation = _require_mapping(config, "evaluation")
    overfit = config.get("overfit_gate", {})
    if overfit is not None and not isinstance(overfit, Mapping):
        raise ConfigError("Config section 'overfit_gate' must be a mapping")
    dataset = config.get("dataset", {})
    if dataset is not None and not isinstance(dataset, Mapping):
        raise ConfigError("Config section 'dataset' must be a mapping")
    loss = config.get("loss", {})
    if loss is not None and not isinstance(loss, Mapping):
        raise ConfigError("Config section 'loss' must be a mapping")
    modules = config.get("modules", {})
    if modules is not None and not isinstance(modules, Mapping):
        raise ConfigError("Config section 'modules' must be a mapping")

    if not str(experiment.get("run_id", "")).strip():
        raise ConfigError("experiment.run_id is required")
    if not str(experiment.get("output_dir", "")).strip():
        raise ConfigError("experiment.output_dir is required")
    if experiment.get("kind") not in {"supervised_baseline", "overfit4"}:
        raise ConfigError(
            "experiment.kind must be 'supervised_baseline' or 'overfit4'"
        )
    protocol = str((dataset or {}).get("protocol", "")).strip()
    if protocol and protocol not in {"sup398", "full1445", "ssl398_1047", "overfit4"}:
        raise ConfigError("dataset.protocol must be sup398, full1445, ssl398_1047, or overfit4")

    for key in ("data_root", "manifest"):
        if not str(data.get(key, "")).strip():
            raise ConfigError(f"data.{key} is required")
    crop_size = _require_positive_int(data, "crop_size")
    if crop_size != 512:
        raise ConfigError(
            "Week 1 canonical crop_size is 512; record a decision before changing it"
        )

    if model.get("name") != "segformer_b0":
        raise ConfigError("Week 1 model.name must be 'segformer_b0'")
    if int(model.get("num_labels", -1)) != 10:
        raise ConfigError("model.num_labels must be 10")
    auxiliary_heads = model.get("auxiliary_heads", ())
    if auxiliary_heads in (None, ""):
        auxiliary_heads = ()
    if not isinstance(auxiliary_heads, (list, tuple)):
        raise ConfigError("model.auxiliary_heads must be a list when set")
    unknown_heads = sorted(set(auxiliary_heads) - {"object", "state", "boundary", "relation"})
    if unknown_heads:
        raise ConfigError(f"Unknown model.auxiliary_heads: {unknown_heads}")
    model_factorization = model.get("state_factorization", {})
    if model_factorization and not isinstance(model_factorization, Mapping):
        raise ConfigError("model.state_factorization must be a mapping")
    if bool((model_factorization or {}).get("enabled", False)):
        feature_source = str(
            model_factorization.get("feature_source", "encoder_multiscale")
        )
        if feature_source not in {"logits", "encoder_multiscale"}:
            raise ConfigError(
                "model.state_factorization.feature_source must be logits or encoder_multiscale"
            )
        decoder_channels = model_factorization.get("decoder_channels", 64)
        if (
            not isinstance(decoder_channels, int)
            or isinstance(decoder_channels, bool)
            or decoder_channels <= 0
        ):
            raise ConfigError("model.state_factorization.decoder_channels must be positive")
        state_mode = str(model_factorization.get("state_mode", "conditional"))
        if state_mode not in {"shared", "conditional"}:
            raise ConfigError("model.state_factorization.state_mode must be shared or conditional")
        fusion_weight = model_factorization.get("fusion_weight", 0.0)
        if not isinstance(fusion_weight, (int, float)) or not 0 <= fusion_weight <= 1:
            raise ConfigError("model.state_factorization.fusion_weight must be in [0, 1]")
        dropout = model_factorization.get("dropout", 0.1)
        if not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
            raise ConfigError("model.state_factorization.dropout must be in [0, 1)")
        if feature_source == "logits" and (
            state_mode != "shared" or float(fusion_weight) != 0.0
        ):
            raise ConfigError(
                "logits feature_source supports only state_mode=shared and fusion_weight=0"
            )
    if model.get("pretrained") and not str(
        model.get("pretrained_model_name_or_path", "")
    ).strip():
        raise ConfigError(
            "model.pretrained_model_name_or_path is required when pretrained=true"
        )

    if "epochs" in training:
        _require_positive_int(training, "epochs")
    if "max_iterations" in training:
        _require_positive_int(training, "max_iterations")
    if "epochs" not in training and "max_iterations" not in training:
        raise ConfigError("training.epochs or training.max_iterations is required")
    _require_positive_int(training, "batch_size")
    _require_positive_int(training, "gradient_accumulation_steps")
    learning_rate = training.get("learning_rate")
    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        raise ConfigError("training.learning_rate must be positive")
    if training.get("optimizer") not in {"adamw", "adam"}:
        raise ConfigError("training.optimizer must be 'adamw' or 'adam'")
    scheduler = str(training.get("scheduler", "constant")).casefold()
    if scheduler not in {"constant", "none", "poly"}:
        raise ConfigError("training.scheduler must be 'constant' or 'poly'")
    if "warmup_iterations" in training:
        warmup = training.get("warmup_iterations")
        if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
            raise ConfigError("training.warmup_iterations must be a non-negative integer")
    if "poly_power" in training:
        power = training.get("poly_power")
        if not isinstance(power, (int, float)) or power <= 0:
            raise ConfigError("training.poly_power must be positive")
    if "gradient_clip_norm" in training:
        clip = training.get("gradient_clip_norm")
        if clip not in (None, "") and (not isinstance(clip, (int, float)) or clip <= 0):
            raise ConfigError("training.gradient_clip_norm must be positive when set")

    _require_positive_int(evaluation, "tile_size")
    _require_positive_int(evaluation, "stride")
    if evaluation["stride"] > evaluation["tile_size"]:
        raise ConfigError("evaluation.stride must not exceed tile_size")

    if experiment.get("kind") == "overfit4":
        if not overfit:
            raise ConfigError("overfit_gate is required for overfit4 runs")
        loss_ratio = overfit.get("maximum_final_to_initial_loss_ratio")
        minimum_miou = overfit.get("minimum_train_miou10")
        if not isinstance(loss_ratio, (int, float)) or not 0 < loss_ratio < 1:
            raise ConfigError(
                "overfit_gate.maximum_final_to_initial_loss_ratio must be in (0, 1)"
            )
        if not isinstance(minimum_miou, (int, float)) or not 0 < minimum_miou <= 1:
            raise ConfigError("overfit_gate.minimum_train_miou10 must be in (0, 1]")
    if loss:
        loss_name = str(loss.get("name", "ce_dice")).casefold()
        if loss_name not in {"ce", "cross_entropy", "ce_dice", "cross_entropy_dice"}:
            raise ConfigError("loss.name must be cross_entropy or ce_dice")
    if modules:
        state_factorization = modules.get("state_factorization", {})
        if state_factorization and not isinstance(state_factorization, Mapping):
            raise ConfigError("modules.state_factorization must be a mapping")
        if bool((state_factorization or {}).get("enabled", False)):
            required = {"object", "state"}
            if not required.issubset(set(auxiliary_heads)):
                raise ConfigError("state_factorization requires object and state auxiliary heads")
            if not bool((model_factorization or {}).get("enabled", False)):
                raise ConfigError(
                    "modules.state_factorization requires model.state_factorization.enabled"
                )
            for key in (
                "object_weight",
                "state_weight",
                "consistency_weight",
                "object_dice_weight",
                "state_dice_weight",
            ):
                value = state_factorization.get(key, 0.0)
                if not isinstance(value, (int, float)) or value < 0:
                    raise ConfigError(f"modules.state_factorization.{key} must be non-negative")
            reduction = str(
                state_factorization.get("consistency_reduction", "class_mean")
            )
            if reduction not in {"pixel_mean", "class_mean"}:
                raise ConfigError(
                    "modules.state_factorization.consistency_reduction must be pixel_mean or class_mean"
                )

