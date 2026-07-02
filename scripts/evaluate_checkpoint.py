"""Evaluate a trained checkpoint on a labeled local FloodNet split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from floodnet_ssl.config import load_yaml_config  # noqa: E402
from floodnet_ssl.constants import CLASS_NAMES  # noqa: E402
from floodnet_ssl.dataset import FloodNetDataset  # noqa: E402
from floodnet_ssl.inference import sliding_window_predict  # noqa: E402
from floodnet_ssl.metrics import (  # noqa: E402
    SegmentationMeter,
    boundary_f1,
    grouped_object_iou,
    state_metrics_from_semantic,
)
from floodnet_ssl.models import build_model  # noqa: E402
from floodnet_ssl.training import collect_runtime_metadata, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a checkpoint on a labeled local split with sliding-window "
            "probability fusion. This script does not train."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", help="Override config training.device")
    parser.add_argument("--tile-size", type=int, help="Override config evaluation.tile_size")
    parser.add_argument("--stride", type=int, help="Override config evaluation.stride")
    parser.add_argument("--tile-batch-size", type=int, help="Override config evaluation.tile_batch_size")
    parser.add_argument("--use-amp", action="store_true", help="Use autocast during inference when supported")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    finite = [value for value in values if not math.isnan(value)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def _prepare_output_dir(path: Path, *, overwrite: bool) -> Path:
    output_dir = path.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Refusing to write into non-empty evaluation directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _make_dataset(config: dict[str, Any], split: str) -> FloodNetDataset:
    data = config["data"]
    return FloodNetDataset(
        data["data_root"],
        data["manifest"],
        split=split,
        transform=None,
        image_mean=tuple(float(value) for value in data["image_mean"]),
        image_std=tuple(float(value) for value in data["image_std"]),
    )


def _load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path.expanduser().resolve(), map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain a model_state_dict")
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def _write_class_csv(path: Path, metrics: dict[str, Any]) -> None:
    ious = metrics["iou_per_class"]
    f1s = metrics["f1_per_class"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_id", "class_name", "iou", "f1"])
        writer.writeheader()
        for class_id, class_name in enumerate(CLASS_NAMES):
            writer.writerow(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "iou": ious[class_id],
                    "f1": f1s[class_id],
                }
            )


def _write_sample_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No per-sample rows to write")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml_config(args.config)
    device = torch.device(args.device or str(config["training"]["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device is unavailable")

    output_dir = _prepare_output_dir(args.output_dir, overwrite=args.overwrite)
    dataset = _make_dataset(config, args.split)
    model = build_model(config["model"]).to(device)
    checkpoint = _load_checkpoint(model, args.checkpoint, device)

    tile_size = int(args.tile_size or config["evaluation"]["tile_size"])
    stride = int(args.stride or config["evaluation"]["stride"])
    tile_batch_size = int(args.tile_batch_size or config["evaluation"].get("tile_batch_size", 1))

    meter = SegmentationMeter()
    sample_rows: list[dict[str, Any]] = []
    boundary_values: list[float] = []
    building_boundary_values: list[float] = []
    road_boundary_values: list[float] = []
    building_iou_values: list[float] = []
    road_iou_values: list[float] = []
    state_accuracy_values: list[float] = []
    state_macro_f1_values: list[float] = []
    flooded_precision_values: list[float] = []
    flooded_recall_values: list[float] = []

    for index in range(len(dataset)):
        sample = dataset[index]
        image = sample["image"]
        target = sample["mask"]
        if target is None:
            raise ValueError(f"Selected split contains sample without mask: {sample['id']}")
        probabilities = sliding_window_predict(
            model,
            image,
            tile_size=tile_size,
            stride=stride,
            tile_batch_size=tile_batch_size,
            device=device,
            use_amp=args.use_amp,
        )
        prediction = probabilities.argmax(dim=1).squeeze(0).cpu()
        target_cpu = target.cpu()
        meter.update(prediction, target_cpu)

        grouped = grouped_object_iou(prediction, target_cpu)
        state = state_metrics_from_semantic(prediction, target_cpu)
        bf1 = boundary_f1(prediction, target_cpu, tolerance=3)
        building_bf1 = boundary_f1(prediction, target_cpu, tolerance=3, class_ids=(1, 2))
        road_bf1 = boundary_f1(prediction, target_cpu, tolerance=3, class_ids=(3, 4))

        boundary_values.append(bf1)
        building_boundary_values.append(building_bf1)
        road_boundary_values.append(road_bf1)
        building_iou_values.append(grouped["building_iou"])
        road_iou_values.append(grouped["road_iou"])
        state_accuracy_values.append(state["state_accuracy"])
        state_macro_f1_values.append(state["state_macro_f1"])
        flooded_precision_values.append(state["flooded_precision"])
        flooded_recall_values.append(state["flooded_recall"])
        sample_rows.append(
            {
                "sample_id": sample["id"],
                "split": sample["split"],
                "boundary_f1": bf1,
                "building_boundary_f1": building_bf1,
                "road_boundary_f1": road_bf1,
                "building_iou": grouped["building_iou"],
                "road_iou": grouped["road_iou"],
                "state_accuracy": state["state_accuracy"],
                "state_macro_f1": state["state_macro_f1"],
                "flooded_precision": state["flooded_precision"],
                "flooded_recall": state["flooded_recall"],
            }
        )

    metrics = meter.compute()
    metrics.update(
        {
            "mean_boundary_f1": _mean(boundary_values),
            "mean_building_boundary_f1": _mean(building_boundary_values),
            "mean_road_boundary_f1": _mean(road_boundary_values),
            "mean_building_iou": _mean(building_iou_values),
            "mean_road_iou": _mean(road_iou_values),
            "mean_state_accuracy": _mean(state_accuracy_values),
            "mean_state_macro_f1": _mean(state_macro_f1_values),
            "mean_flooded_precision": _mean(flooded_precision_values),
            "mean_flooded_recall": _mean(flooded_recall_values),
        }
    )
    payload = {
        "run_id": config["experiment"]["run_id"],
        "split": args.split,
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "num_samples": len(dataset),
        "tile_size": tile_size,
        "stride": stride,
        "tile_batch_size": tile_batch_size,
        "use_amp": bool(args.use_amp),
        "runtime": collect_runtime_metadata(device),
        "metrics": _json_ready(metrics),
    }
    write_json(output_dir / "metrics.json", payload)
    _write_class_csv(output_dir / "class_metrics.csv", _json_ready(metrics))
    _write_sample_csv(output_dir / "per_sample_metrics.csv", sample_rows)
    return payload


def main() -> int:
    payload = evaluate_checkpoint(parse_args())
    print(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
