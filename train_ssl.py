"""EMA teacher-student training entry point for FloodNet structure-aware SSL."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from floodnet_ssl.config import load_yaml_config  # noqa: E402
from floodnet_ssl.dataset import FloodNetDataset  # noqa: E402
from floodnet_ssl.experiment import apply_path_overrides, make_dataset, make_loader  # noqa: E402
from floodnet_ssl.losses import segmentation_loss  # noqa: E402
from floodnet_ssl.models import build_model, extract_logits  # noqa: E402
from floodnet_ssl.pseudolabels import make_pseudo_labels, update_ema_teacher  # noqa: E402
from floodnet_ssl.training import build_optimizer, set_reproducible_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FloodNet EMA teacher-student SSL.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--supervised-root", type=Path, help="Override data root")
    parser.add_argument("--output-dir", type=Path, help="Override experiment.output_dir")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-iterations", type=int, help="Temporary smoke-test override")
    return parser.parse_args()


def read_id_list(path: str | Path) -> set[str]:
    return {
        line.strip().casefold()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def filter_manifest_by_ids(
    source: str | Path,
    destination: str | Path,
    *,
    sample_ids: set[str],
    split: str = "train",
) -> int:
    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [
            row
            for row in reader
            if row.get("split") == split and row.get("sample_id", "").casefold() in sample_ids
        ]
        fieldnames = reader.fieldnames or []
    with destination_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def build_ssl_plan(config: dict[str, Any]) -> dict[str, Any]:
    labeled_dataset = make_dataset(config, str(config["data"].get("train_split", "train")), training=True)
    validation_dataset = make_dataset(config, str(config["data"].get("validation_split", "validation")), training=False)
    unlabeled_ids = read_id_list(config["data"]["unlabeled_id_list"])
    return {
        "experiment": config["experiment"].get("name", config["experiment"].get("run_id")),
        "protocol": config.get("dataset", {}).get("protocol"),
        "data_root": config["data"]["data_root"],
        "labeled_manifest": config["data"]["manifest"],
        "unlabeled_manifest": config["data"]["unlabeled_manifest"],
        "unlabeled_id_list": config["data"]["unlabeled_id_list"],
        "labeled_train_samples": len(labeled_dataset),
        "unlabeled_train_samples": len(unlabeled_ids),
        "validation_samples": len(validation_dataset),
        "ssl": config["ssl"],
    }


def _resize_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2:] != labels.shape[-2:]:
        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    return logits


def main() -> int:
    args = parse_args()
    config = load_yaml_config(args.config)
    apply_path_overrides(config, supervised_root=args.supervised_root, output_dir=args.output_dir)
    if args.max_iterations is not None:
        config["training"]["max_iterations"] = args.max_iterations

    plan = build_ssl_plan(config)
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("Dry run only: no model was built and no SSL training was started.")
        return 0

    output_dir = Path(config["experiment"]["output_dir"]).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    unlabeled_manifest = output_dir / "unlabeled_manifest.csv"
    count = filter_manifest_by_ids(
        config["data"]["unlabeled_manifest"],
        unlabeled_manifest,
        sample_ids=read_id_list(config["data"]["unlabeled_id_list"]),
        split=str(config["data"].get("train_split", "train")),
    )
    if count != plan["unlabeled_train_samples"]:
        raise ValueError("Filtered unlabeled manifest count does not match unlabeled ID list")

    set_reproducible_seed(int(config["experiment"].get("seed", 0)))
    device = torch.device(str(config["training"]["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device is unavailable")
    student = build_model(config["model"]).to(device)
    teacher = build_model(config["model"]).to(device)
    teacher.load_state_dict(student.state_dict())
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    optimizer = build_optimizer(student, config["training"])
    labeled_loader = make_loader(config, str(config["data"].get("train_split", "train")), training=True)
    unlabeled_dataset = FloodNetDataset(
        config["data"]["data_root"],
        unlabeled_manifest,
        split=str(config["data"].get("train_split", "train")),
        transform=None,
        image_mean=tuple(float(value) for value in config["data"]["image_mean"]),
        image_std=tuple(float(value) for value in config["data"]["image_std"]),
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 0)),
        drop_last=True,
    )
    labeled_iter = itertools.cycle(labeled_loader)
    unlabeled_iter = itertools.cycle(unlabeled_loader)
    ssl_config = config["ssl"]
    max_iterations = int(config["training"]["max_iterations"])
    for iteration in range(1, max_iterations + 1):
        student.train()
        labeled = next(labeled_iter)
        unlabeled = next(unlabeled_iter)
        labeled_images = labeled["image"].to(device)
        labels = labeled["mask"].to(device)
        unlabeled_images = unlabeled["image"].to(device)
        optimizer.zero_grad(set_to_none=True)
        supervised_logits = _resize_logits(extract_logits(student(labeled_images)), labels)
        supervised_loss = segmentation_loss(supervised_logits, labels, config.get("loss"))
        with torch.no_grad():
            teacher_logits = extract_logits(teacher(unlabeled_images))
            pseudo = make_pseudo_labels(
                teacher_logits,
                threshold=float(ssl_config.get("threshold", 0.8)),
                matched_coverage=ssl_config.get("matched_coverage"),
                confidence_weight=float(ssl_config.get("confidence_weight", 1.0)),
                multiview_weight=float(ssl_config.get("multiview_weight", 0.25)),
                boundary_weight=float(ssl_config.get("boundary_weight", 0.25)),
                region_weight=float(ssl_config.get("region_weight", 0.25)),
                region_kernel_size=int(ssl_config.get("region_kernel_size", 5)),
            )
        student_logits = extract_logits(student(unlabeled_images))
        if student_logits.shape[-2:] != pseudo.labels.shape[-2:]:
            student_logits = F.interpolate(student_logits, size=pseudo.labels.shape[-2:], mode="bilinear", align_corners=False)
        unsupervised_map = F.cross_entropy(student_logits, pseudo.labels, reduction="none")
        if pseudo.mask.any():
            unsupervised_loss = unsupervised_map[pseudo.mask].mean()
        else:
            unsupervised_loss = supervised_loss.detach() * 0.0
        weight = float(ssl_config.get("unsupervised_weight", 1.0))
        loss = supervised_loss + weight * unsupervised_loss
        loss.backward()
        optimizer.step()
        update_ema_teacher(teacher, student, decay=float(ssl_config.get("ema_decay", 0.999)))

    torch.save(
        {
            "iteration": max_iterations,
            "student_state_dict": student.state_dict(),
            "teacher_state_dict": teacher.state_dict(),
            "config": config,
        },
        output_dir / "last_ssl.pth",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
