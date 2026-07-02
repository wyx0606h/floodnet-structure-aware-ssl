"""Minimal supervised training and evaluation engine."""

from __future__ import annotations

import csv
import json
import platform
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as torch_functional

from .losses import segmentation_loss
from .constants import IGNORE_INDEX
from .metrics import SegmentationMeter
from .models import extract_logits


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(
    model: torch.nn.Module, training_config: Mapping[str, Any]
) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_name = str(training_config["optimizer"]).casefold()
    options = {
        "lr": float(training_config["learning_rate"]),
        "weight_decay": float(training_config.get("weight_decay", 0.0)),
    }
    if optimizer_name == "adamw":
        return torch.optim.AdamW(parameters, **options)
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, **options)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _resize_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 4:
        raise ValueError(f"Expected logits [B,C,H,W], got {logits.shape}")
    if logits.shape[-2:] != labels.shape[-2:]:
        logits = torch_functional.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return logits


@dataclass(frozen=True)
class EpochResult:
    loss: float
    miou10: float
    miou9: float
    pixel_accuracy: float
    optimizer_steps: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "loss": self.loss,
            "miou10": self.miou10,
            "miou9": self.miou9,
            "pixel_accuracy": self.pixel_accuracy,
            "optimizer_steps": self.optimizer_steps,
        }


def run_supervised_epoch(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device | str,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_accumulation_steps: int = 1,
    ignore_index: int = IGNORE_INDEX,
    use_amp: bool = False,
    loss_config: Mapping[str, Any] | None = None,
) -> EpochResult:
    training = optimizer is not None
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    target_device = torch.device(device)
    model.train(training)
    meter = SegmentationMeter(ignore_index=ignore_index)
    total_loss = 0.0
    batch_count = 0
    optimizer_steps = 0
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp and target_device.type == "cuda"
    )
    for batch_index, batch in enumerate(loader):
        images = batch["image"].to(target_device)
        labels = batch["mask"].to(target_device)
        amp_enabled = use_amp and target_device.type == "cuda"
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=target_device.type,
                enabled=amp_enabled,
            ):
                logits = _resize_logits(extract_logits(model(images)), labels)
                loss = segmentation_loss(
                    logits, labels, loss_config, ignore_index=ignore_index
                )
            if training:
                scaled_loss = loss / gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()
                should_step = (batch_index + 1) % gradient_accumulation_steps == 0
                if should_step:
                    assert optimizer is not None
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1

        total_loss += float(loss.detach())
        batch_count += 1
        meter.update(logits.detach().argmax(dim=1), labels.detach())

    if not batch_count:
        raise ValueError("DataLoader produced no batches")
    if training and batch_count % gradient_accumulation_steps:
        assert optimizer is not None
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1

    metrics = meter.compute()
    return EpochResult(
        loss=total_loss / batch_count,
        miou10=float(metrics["miou10"]),
        miou9=float(metrics["miou9"]),
        pixel_accuracy=float(metrics["pixel_accuracy"]),
        optimizer_steps=optimizer_steps,
    )


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": dict(config),
        },
        checkpoint_path,
    )


def write_history_csv(path: str | Path, history: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        raise ValueError("Cannot write empty training history")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def write_resolved_config(path: str | Path, config: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(config), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def collect_runtime_metadata(device: torch.device | str) -> dict[str, Any]:
    target_device = torch.device(device)
    cuda_available = torch.cuda.is_available()
    metadata: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "configured_device": str(target_device),
    }
    if target_device.type == "cuda" and cuda_available:
        device_index = target_device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        metadata.update(
            {
                "cuda_device_index": device_index,
                "cuda_device_name": properties.name,
                "cuda_total_memory_gb": properties.total_memory / 1024**3,
            }
        )
    return metadata


def peak_memory_gb(device: torch.device | str) -> float | None:
    target_device = torch.device(device)
    if target_device.type != "cuda" or not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated(target_device) / 1024**3)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def evaluate_overfit_gate(
    history: list[Mapping[str, Any]],
    *,
    maximum_final_to_initial_loss_ratio: float,
    minimum_train_miou10: float,
) -> dict[str, Any]:
    if not history:
        raise ValueError("Overfit gate requires non-empty history")
    initial_loss = float(history[0]["train_loss"])
    final_loss = float(history[-1]["train_loss"])
    final_miou = float(history[-1]["train_miou10"])
    loss_ratio = final_loss / initial_loss if initial_loss > 0 else float("inf")
    passed = (
        loss_ratio <= maximum_final_to_initial_loss_ratio
        and final_miou >= minimum_train_miou10
    )
    return {
        "passed": passed,
        "initial_train_loss": initial_loss,
        "final_train_loss": final_loss,
        "final_to_initial_loss_ratio": loss_ratio,
        "final_train_miou10": final_miou,
        "maximum_final_to_initial_loss_ratio": maximum_final_to_initial_loss_ratio,
        "minimum_train_miou10": minimum_train_miou10,
    }
