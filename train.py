"""Unified FloodNet supervised training entry point."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from floodnet_ssl.config import load_yaml_config  # noqa: E402
from floodnet_ssl.experiment import (  # noqa: E402
    append_history_csv,
    apply_path_overrides,
    configure_logger,
    ensure_run_layout,
    evaluate_model,
    make_dataset,
    make_loader,
    save_training_checkpoint,
    write_metrics_files,
    write_resolved_yaml,
)
from floodnet_ssl.losses import segmentation_loss  # noqa: E402
from floodnet_ssl.models import build_model, extract_logits  # noqa: E402
from floodnet_ssl.training import build_optimizer, collect_runtime_metadata, set_reproducible_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FloodNet SegFormer-B0 with a protocol config.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--supervised-root", type=Path, help="Override dataset/data supervised root")
    parser.add_argument("--challenge-root", type=Path, help="Stored in resolved config for sup398 provenance")
    parser.add_argument("--output-dir", type=Path, help="Override experiment.output_dir")
    parser.add_argument("--resume", type=Path, help="Resume from checkpoint, usually outputs/<exp>/checkpoints/last.pth")
    parser.add_argument("--dry-run", action="store_true", help="Resolve config and data counts but do not build model or train")
    parser.add_argument("--max-iterations", type=int, help="Temporary CLI override for smoke tests")
    parser.add_argument("--val-interval", type=int, help="Temporary CLI override for smoke tests")
    parser.add_argument("--max-eval-samples", type=int, help="Limit validation samples, mainly for smoke tests")
    return parser.parse_args()


def _resize_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2:] != labels.shape[-2:]:
        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    return logits


def _scheduled_lr(training: dict[str, Any], step: int, max_iterations: int) -> float:
    base_lr = float(training["learning_rate"])
    scheduler = str(training.get("scheduler", "constant")).casefold()
    warmup = int(training.get("warmup_iterations", 0))
    if warmup > 0 and step <= warmup:
        return base_lr * float(step) / float(warmup)
    if scheduler in {"", "none", "constant"}:
        return base_lr
    if scheduler == "poly":
        power = float(training.get("poly_power", 1.0))
        denominator = max(max_iterations - warmup, 1)
        progress = min(max(step - warmup, 0) / denominator, 1.0)
        return base_lr * ((1.0 - progress) ** power)
    raise ValueError(f"Unsupported training.scheduler: {scheduler}")


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def main() -> int:
    args = parse_args()
    config = load_yaml_config(args.config)
    apply_path_overrides(
        config,
        supervised_root=args.supervised_root,
        challenge_root=args.challenge_root,
        output_dir=args.output_dir,
    )
    if args.max_iterations is not None:
        config["training"]["max_iterations"] = args.max_iterations
    if args.val_interval is not None:
        config["training"]["val_interval"] = args.val_interval
    if args.max_eval_samples is not None:
        config.setdefault("evaluation", {})["max_eval_samples"] = args.max_eval_samples

    experiment = config["experiment"]
    training = config["training"]
    output_dir = Path(experiment["output_dir"]).expanduser().resolve()
    if output_dir.exists() and args.resume is None and not args.dry_run:
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")

    train_dataset = make_dataset(config, str(config["data"].get("train_split", "train")), training=True)
    val_dataset = make_dataset(config, str(config["data"].get("validation_split", "validation")), training=False)
    plan = {
        "experiment": experiment.get("name", experiment.get("run_id")),
        "protocol": config.get("dataset", {}).get("protocol"),
        "output_dir": str(output_dir),
        "data_root": config["data"]["data_root"],
        "manifest": config["data"]["manifest"],
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "model": config["model"],
        "loss": config.get("loss", {"name": "ce_dice"}),
        "training": training,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("Dry run only: no model was built and no training was started.")
        return 0

    paths = ensure_run_layout(output_dir)
    logger = configure_logger(output_dir / "train.log")
    write_resolved_yaml(output_dir / "config_resolved.yaml", config)
    (output_dir / "runtime_metadata.json").write_text(
        json.dumps(collect_runtime_metadata(training["device"]), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Resolved plan: %s", json.dumps(plan, ensure_ascii=False))

    device = torch.device(str(training["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device is unavailable")
    set_reproducible_seed(int(experiment.get("seed", 0)))
    model = build_model(config["model"]).to(device)
    optimizer = build_optimizer(model, training)
    use_amp = bool(training.get("use_amp", False))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    start_iteration = 0
    best_miou = float("-inf")
    history: list[dict[str, Any]] = []
    if args.resume is not None:
        checkpoint = torch.load(args.resume.expanduser().resolve(), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iteration = int(checkpoint.get("iteration", 0))
        best_miou = float(checkpoint.get("best_miou", float("-inf")))
        history = list(checkpoint.get("history", []))
        logger.info("Resumed from %s at iteration %d", args.resume, start_iteration)

    train_loader = make_loader(config, str(config["data"].get("train_split", "train")), training=True)
    train_iter = itertools.cycle(train_loader)
    max_iterations = int(training["max_iterations"])
    val_interval = int(training.get("val_interval", max_iterations))
    grad_accum = int(training.get("gradient_accumulation_steps", 1))
    max_eval_samples = config.get("evaluation", {}).get("max_eval_samples")
    max_eval_samples = int(max_eval_samples) if max_eval_samples not in (None, "") else None

    for iteration in range(start_iteration + 1, max_iterations + 1):
        model.train()
        current_lr = _scheduled_lr(training, iteration, max_iterations)
        _set_optimizer_lr(optimizer, current_lr)
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for _ in range(grad_accum):
            batch = next(train_iter)
            images = batch["image"].to(device)
            labels = batch["mask"].to(device)
            amp_enabled = use_amp and device.type == "cuda"
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = _resize_logits(extract_logits(model(images)), labels)
                loss = segmentation_loss(logits, labels, config.get("loss")) / grad_accum
            scaler.scale(loss).backward()
            loss_value += float(loss.detach())
        clip_norm = training.get("gradient_clip_norm")
        if clip_norm not in (None, ""):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_norm))
        scaler.step(optimizer)
        scaler.update()

        should_eval = iteration == 1 or iteration % val_interval == 0 or iteration == max_iterations
        row: dict[str, Any] = {"iteration": iteration, "train_loss": loss_value, "learning_rate": current_lr}
        if should_eval:
            val_payload = evaluate_model(
                model,
                val_dataset,
                config=config,
                split=str(config["data"].get("validation_split", "validation")),
                device=device,
                max_samples=max_eval_samples,
            )
            val_metrics = val_payload["metrics"]
            row.update(
                {
                    "validation_miou": val_metrics["miou10"],
                    "validation_macro_f1": val_metrics["macro_f1"],
                    "validation_pixel_accuracy": val_metrics["pixel_accuracy"],
                    "validation_flooded_miou": val_metrics["flooded_miou"],
                }
            )
            metrics_dir = paths["metrics"] / f"validation_iter_{iteration:07d}"
            write_metrics_files(metrics_dir, val_payload)
            if float(val_metrics["miou10"]) >= best_miou:
                best_miou = float(val_metrics["miou10"])
                save_training_checkpoint(
                    paths["checkpoints"] / "best_miou.pth",
                    model=model,
                    optimizer=optimizer,
                    iteration=iteration,
                    best_miou=best_miou,
                    history=history + [row],
                    config=config,
                )
                logger.info("New best mIoU %.6f at iteration %d", best_miou, iteration)
        history.append(row)
        append_history_csv(paths["curves"] / "history.csv", history)
        if should_eval or iteration % 500 == 0:
            save_training_checkpoint(
                paths["checkpoints"] / "last.pth",
                model=model,
                optimizer=optimizer,
                iteration=iteration,
                best_miou=best_miou,
                history=history,
                config=config,
            )
        logger.info("iteration=%d train_loss=%.6f best_miou=%.6f", iteration, loss_value, best_miou)

    summary = {
        "experiment": experiment.get("name", experiment.get("run_id")),
        "protocol": config.get("dataset", {}).get("protocol"),
        "max_iterations": max_iterations,
        "best_miou": best_miou,
        "best_checkpoint": str(paths["checkpoints"] / "best_miou.pth"),
        "last_checkpoint": str(paths["checkpoints"] / "last.pth"),
        "history_rows": len(history),
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Training complete: %s", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
