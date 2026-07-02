"""Unified FloodNet checkpoint evaluation entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from floodnet_ssl.config import load_yaml_config  # noqa: E402
from floodnet_ssl.experiment import apply_path_overrides, evaluate_model, make_dataset, write_metrics_files  # noqa: E402
from floodnet_ssl.models import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FloodNet checkpoint on validation or test split.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("validation", "test", "train"))
    parser.add_argument("--output-dir", type=Path, help="Defaults to <experiment.output_dir>/metrics/<split>_<checkpoint>")
    parser.add_argument("--supervised-root", type=Path, help="Override data root")
    parser.add_argument("--device", help="Override training.device")
    parser.add_argument("--max-samples", type=int, help="Limit samples for smoke tests")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Resolve dataset/evaluation plan but do not build model or load checkpoint")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml_config(args.config)
    apply_path_overrides(config, supervised_root=args.supervised_root)
    device = torch.device(args.device or str(config["training"]["device"]))
    checkpoint_path = args.checkpoint.expanduser().resolve()
    dataset = make_dataset(config, args.split, training=False)
    output_dir = args.output_dir
    if output_dir is None:
        name = checkpoint_path.stem
        output_dir = Path(config["experiment"]["output_dir"]) / "metrics" / f"{args.split}_{name}"
    output_dir = output_dir.expanduser().resolve()
    plan = {
        "mode": "dry-run" if args.dry_run else "execute",
        "experiment": config["experiment"].get("name", config["experiment"].get("run_id")),
        "protocol": config.get("dataset", {}).get("protocol"),
        "split": args.split,
        "samples": len(dataset),
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "device": str(device),
        "evaluation": config["evaluation"],
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print("Dry run only: no model was built and no checkpoint was loaded.")
        return 0
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device is unavailable")

    model = build_model(config["model"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
    if state is None:
        raise ValueError("Checkpoint must contain model_state_dict")
    model.load_state_dict(state)

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to write into non-empty evaluation directory: {output_dir}")
    predictions_dir = output_dir / "predictions" if args.save_predictions else None
    payload = evaluate_model(
        model,
        dataset,
        config=config,
        split=args.split,
        device=device,
        checkpoint=str(checkpoint_path),
        max_samples=args.max_samples,
        save_predictions_dir=predictions_dir,
    )
    write_metrics_files(output_dir, payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "per_sample"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
