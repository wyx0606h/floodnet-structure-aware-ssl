"""Shared training/evaluation helpers for FloodNet protocol experiments."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .constants import CLASS_NAMES, IGNORE_INDEX
from .dataset import FloodNetDataset
from .inference import sliding_window_predict
from .metrics import SegmentationMeter, boundary_f1, grouped_object_iou, state_metrics_from_semantic
from .models import extract_logits
from .training import collect_runtime_metadata
from .transforms import CenterCrop, build_supervised_train_transform


def apply_path_overrides(
    config: dict[str, Any],
    *,
    supervised_root: str | Path | None = None,
    challenge_root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    if supervised_root is not None:
        config.setdefault("data", {})["data_root"] = str(Path(supervised_root).expanduser())
        config.setdefault("dataset", {})["supervised_root"] = str(Path(supervised_root).expanduser())
    if challenge_root is not None:
        config.setdefault("dataset", {})["challenge_root"] = str(Path(challenge_root).expanduser())
    if output_dir is not None:
        config.setdefault("experiment", {})["output_dir"] = str(Path(output_dir).expanduser())
    return config


def ensure_run_layout(output_dir: Path) -> dict[str, Path]:
    paths = {
        "root": output_dir,
        "checkpoints": output_dir / "checkpoints",
        "metrics": output_dir / "metrics",
        "curves": output_dir / "curves",
        "predictions": output_dir / "predictions",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_resolved_yaml(path: Path, config: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True), encoding="utf-8")


def configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("floodnet_experiment")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def make_transform(config: Mapping[str, Any], *, split: str, training: bool):
    data = config["data"]
    seed = int(config["experiment"].get("seed", 0))
    crop_size = int(data.get("crop_size", 512))
    if training:
        return build_supervised_train_transform(
            crop_size=crop_size,
            seed=seed,
            scale_range=tuple(float(v) for v in data.get("scale_range", (0.75, 1.25))),
            class_ids=tuple(int(v) for v in data.get("class_aware_ids", (1, 3))),
            class_aware_probability=float(data.get("class_aware_probability", 0.5)),
        )
    if bool(data.get("center_crop_eval", False)):
        return CenterCrop(crop_size)
    return None


def make_dataset(config: Mapping[str, Any], split: str, *, training: bool = False) -> FloodNetDataset:
    data = config["data"]
    return FloodNetDataset(
        data["data_root"],
        data["manifest"],
        split=split,
        transform=make_transform(config, split=split, training=training),
        image_mean=tuple(float(value) for value in data["image_mean"]),
        image_std=tuple(float(value) for value in data["image_std"]),
    )


def make_loader(config: Mapping[str, Any], split: str, *, training: bool) -> DataLoader:
    dataset = make_dataset(config, split, training=training)
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=training,
        num_workers=int(config["data"].get("num_workers", 0)),
        drop_last=bool(training and config["data"].get("drop_last", False)),
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if not np.isnan(float(value))]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def write_metrics_files(output_dir: Path, payload: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = payload["metrics"]
    (output_dir / "metrics.json").write_text(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    matrix = np.asarray(metrics["confusion_matrix"])
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    with (output_dir / "class_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_id", "class_name", "iou", "precision", "recall", "f1"])
        writer.writeheader()
        for class_id, class_name in enumerate(CLASS_NAMES):
            writer.writerow(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "iou": metrics["iou_per_class"][class_id],
                    "precision": metrics["precision_per_class"][class_id],
                    "recall": metrics["recall_per_class"][class_id],
                    "f1": metrics["f1_per_class"][class_id],
                }
            )
    rows = payload.get("per_sample", [])
    if rows:
        with (output_dir / "per_sample_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataset: FloodNetDataset,
    *,
    config: Mapping[str, Any],
    split: str,
    device: torch.device,
    checkpoint: str | None = None,
    max_samples: int | None = None,
    save_predictions_dir: Path | None = None,
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    tile_size = int(evaluation.get("tile_size", 512))
    stride = int(evaluation.get("stride", 384))
    tile_batch_size = int(evaluation.get("tile_batch_size", 4))
    use_amp = bool(evaluation.get("use_amp", config.get("training", {}).get("use_amp", False)))
    meter = SegmentationMeter(ignore_index=IGNORE_INDEX)
    per_sample: list[dict[str, Any]] = []
    boundary_values: list[float] = []
    building_iou_values: list[float] = []
    road_iou_values: list[float] = []
    state_macro_f1_values: list[float] = []

    sample_count = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    if save_predictions_dir is not None:
        save_predictions_dir.mkdir(parents=True, exist_ok=True)
    for index in range(sample_count):
        sample = dataset[index]
        target = sample["mask"]
        if target is None:
            raise ValueError(f"Cannot evaluate sample without mask: {sample['id']}")
        probabilities = sliding_window_predict(
            model,
            sample["image"],
            tile_size=tile_size,
            stride=stride,
            tile_batch_size=tile_batch_size,
            device=device,
            use_amp=use_amp,
        )
        prediction = probabilities.argmax(dim=1).squeeze(0).cpu()
        target_cpu = target.cpu()
        meter.update(prediction, target_cpu)
        grouped = grouped_object_iou(prediction, target_cpu)
        state = state_metrics_from_semantic(prediction, target_cpu)
        bf1 = boundary_f1(prediction, target_cpu, tolerance=int(evaluation.get("boundary_tolerance", 3)))
        boundary_values.append(bf1)
        building_iou_values.append(grouped["building_iou"])
        road_iou_values.append(grouped["road_iou"])
        state_macro_f1_values.append(state["state_macro_f1"])
        per_sample.append(
            {
                "sample_id": sample["id"],
                "split": split,
                "boundary_f1": bf1,
                "building_iou": grouped["building_iou"],
                "road_iou": grouped["road_iou"],
                "state_macro_f1": state["state_macro_f1"],
            }
        )
        if save_predictions_dir is not None:
            from PIL import Image
            Image.fromarray(prediction.numpy().astype(np.uint8), mode="L").save(save_predictions_dir / f"{sample['id']}.png")

    metrics = meter.compute()
    metrics.update(
        {
            "mean_boundary_f1": _mean(boundary_values),
            "mean_building_iou": _mean(building_iou_values),
            "mean_road_iou": _mean(road_iou_values),
            "mean_state_macro_f1": _mean(state_macro_f1_values),
        }
    )
    return {
        "experiment_name": config["experiment"].get("name", config["experiment"].get("run_id")),
        "protocol": config.get("dataset", {}).get("protocol"),
        "split": split,
        "checkpoint": checkpoint,
        "num_samples": sample_count,
        "tile_size": tile_size,
        "stride": stride,
        "tile_batch_size": tile_batch_size,
        "metrics": _json_ready(metrics),
        "per_sample": _json_ready(per_sample),
        "runtime": collect_runtime_metadata(device),
    }


def append_history_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    best_miou: float,
    history: list[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_miou": best_miou,
            "history": list(history),
            "config": dict(config),
        },
        path,
    )
