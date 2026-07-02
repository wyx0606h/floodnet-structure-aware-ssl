"""Read-only server readiness check for FloodNet training."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from floodnet_ssl.config import ConfigError, load_yaml_config  # noqa: E402
from floodnet_ssl.dataset import FloodNetDataset  # noqa: E402
from floodnet_ssl.training import collect_runtime_metadata, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check server dependencies and paths without installing or training."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-pretrained-cache", action="store_true")
    return parser.parse_args()


def _package_status(packages: tuple[str, ...]) -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in packages}


def _count_manifest_rows(config: dict[str, Any], split: str) -> int:
    dataset = FloodNetDataset(
        config["data"]["data_root"],
        config["data"]["manifest"],
        split=split,
        transform=None,
        to_tensor=False,
        validate_masks=False,
    )
    return len(dataset)


def check_server_environment(args: argparse.Namespace) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    try:
        config = load_yaml_config(args.config)
    except ConfigError as error:
        return {
            "ready": False,
            "config": str(args.config.expanduser().resolve()),
            "blockers": [f"config error: {error}"],
            "warnings": warnings,
        }

    device = torch.device(str(config["training"]["device"]))
    packages = _package_status(("transformers", "safetensors", "accelerate", "yaml"))
    if not packages["transformers"]:
        blockers.append("missing dependency: transformers")
    if bool(config["model"].get("pretrained")) and not packages["safetensors"]:
        blockers.append("missing dependency for pretrained baseline: safetensors")
    if args.require_cuda and not torch.cuda.is_available():
        blockers.append("CUDA was required but torch.cuda.is_available() is false")
    if device.type == "cuda" and not torch.cuda.is_available():
        blockers.append("config requests CUDA but CUDA is unavailable")
    if not packages["accelerate"]:
        warnings.append("accelerate is not installed; not required for current scripts")

    pretrained_source = str(config["model"].get("pretrained_model_name_or_path", ""))
    local_model_path_found: bool | None
    if not bool(config["model"].get("pretrained")):
        local_model_path_found = None
    elif Path(pretrained_source).expanduser().exists():
        local_model_path_found = True
    else:
        local_model_path_found = False
        if args.require_pretrained_cache:
            blockers.append(
                "pretrained_model_name_or_path is not a local path; ensure Hugging Face cache is ready or set a local path"
            )
        else:
            warnings.append(
                "pretrained source is a model ID or missing local path; download/cache must be handled before training"
            )

    split_counts: dict[str, int] = {}
    try:
        for split in ("train", "validation", "test"):
            split_counts[split] = _count_manifest_rows(config, split)
    except Exception as error:  # noqa: BLE001 - reported as readiness blocker
        blockers.append(f"data/manifest check failed: {error}")

    payload = {
        "ready": not blockers,
        "config": str(args.config.expanduser().resolve()),
        "runtime": collect_runtime_metadata(device),
        "packages": packages,
        "configured_device": str(device),
        "pretrained_source": pretrained_source,
        "local_model_path_found": local_model_path_found,
        "split_counts": split_counts,
        "blockers": blockers,
        "warnings": warnings,
    }
    if args.output:
        write_json(args.output, payload)
    return payload


def main() -> int:
    payload = check_server_environment(parse_args())
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
