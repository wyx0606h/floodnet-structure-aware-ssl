"""Configuration-driven supervised SegFormer training entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from floodnet_ssl.config import load_yaml_config  # noqa: E402
from floodnet_ssl.dataset import FloodNetDataset  # noqa: E402
from floodnet_ssl.models import build_model  # noqa: E402
from floodnet_ssl.training import (  # noqa: E402
    build_optimizer,
    collect_runtime_metadata,
    evaluate_overfit_gate,
    peak_memory_gb,
    run_supervised_epoch,
    save_checkpoint,
    set_reproducible_seed,
    write_history_csv,
    write_json,
    write_resolved_config,
)
from floodnet_ssl.transforms import (  # noqa: E402
    CenterCrop,
    DeterministicClassCrop,
    build_supervised_train_transform,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train SegFormer from a validated config. Without --execute this "
            "command only prints a non-mutating plan."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-run-id")
    return parser.parse_args()


def _make_dataset(config: dict, split: str, *, training: bool) -> FloodNetDataset:
    data = config["data"]
    seed = int(config["experiment"]["seed"])
    crop_size = int(data["crop_size"])
    if data.get("deterministic_overfit_crop"):
        transform = DeterministicClassCrop(
            crop_size, class_ids=tuple(data.get("class_aware_ids", (1, 3)))
        )
    elif training:
        transform = build_supervised_train_transform(
            crop_size=crop_size,
            seed=seed,
            class_ids=tuple(data.get("class_aware_ids", (1, 3))),
            class_aware_probability=float(data.get("class_aware_probability", 0.5)),
        )
    else:
        transform = CenterCrop(crop_size)
    return FloodNetDataset(
        data["data_root"],
        data["manifest"],
        split=split,
        transform=transform,
        image_mean=tuple(float(value) for value in data["image_mean"]),
        image_std=tuple(float(value) for value in data["image_std"]),
    )


def main() -> int:
    args = parse_args()
    config = load_yaml_config(args.config)
    run_id = str(config["experiment"]["run_id"])
    output_dir = Path(config["experiment"]["output_dir"]).expanduser().resolve()
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "run_id": run_id,
        "output_dir": str(output_dir),
        "model": config["model"],
        "training": config["training"],
        "data": config["data"],
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.execute:
        print("\nDry run only: no model was built and no training was started.")
        return 0
    if args.confirm_run_id != run_id:
        raise ValueError("--confirm-run-id must exactly match experiment.run_id")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run directory: {output_dir}")

    device = torch.device(str(config["training"]["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device is unavailable")
    set_reproducible_seed(int(config["experiment"]["seed"]))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_dataset = _make_dataset(
        config, str(config["data"]["train_split"]), training=True
    )
    validation_dataset = _make_dataset(
        config, str(config["data"]["validation_split"]), training=False
    )
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["data"].get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    model = build_model(config["model"]).to(device)
    optimizer = build_optimizer(model, config["training"])
    output_dir.mkdir(parents=True, exist_ok=False)
    write_resolved_config(output_dir / "resolved_config.json", config)
    write_json(output_dir / "runtime_metadata.json", collect_runtime_metadata(device))
    history: list[dict] = []
    best_validation_miou = float("-inf")
    best_epoch = 0
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_result = run_supervised_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            gradient_accumulation_steps=int(
                config["training"]["gradient_accumulation_steps"]
            ),
            use_amp=bool(config["training"].get("use_amp", False)),
        )
        validation_result = run_supervised_epoch(
            model,
            validation_loader,
            device=device,
            optimizer=None,
            use_amp=bool(config["training"].get("use_amp", False)),
        )
        row = {
            "epoch": epoch,
            "train_loss": train_result.loss,
            "train_miou10": train_result.miou10,
            "train_miou9": train_result.miou9,
            "validation_loss": validation_result.loss,
            "validation_miou10": validation_result.miou10,
            "validation_miou9": validation_result.miou9,
        }
        history.append(row)
        write_history_csv(output_dir / "history.csv", history)
        save_checkpoint(
            output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            history=history,
            config=config,
        )
        if validation_result.miou10 >= best_validation_miou:
            best_validation_miou = validation_result.miou10
            best_epoch = epoch
            save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                history=history,
                config=config,
            )
        print(json.dumps(row, ensure_ascii=False))

    train_summary = {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_miou10": best_validation_miou,
        "last_epoch": history[-1] if history else None,
        "peak_vram_gb": peak_memory_gb(device),
        "best_checkpoint": str(output_dir / "best.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
    }
    write_json(output_dir / "train_summary.json", train_summary)

    if config["experiment"]["kind"] == "overfit4":
        gate = evaluate_overfit_gate(
            history,
            maximum_final_to_initial_loss_ratio=float(
                config["overfit_gate"]["maximum_final_to_initial_loss_ratio"]
            ),
            minimum_train_miou10=float(
                config["overfit_gate"]["minimum_train_miou10"]
            ),
        )
        (output_dir / "overfit_gate.json").write_text(
            json.dumps(gate, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(gate, ensure_ascii=False, indent=2))
        return 0 if gate["passed"] else 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
