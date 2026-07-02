"""Validate a supervised configuration without training or downloading."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from floodnet_ssl.config import load_yaml_config  # noqa: E402
from floodnet_ssl.layout import read_manifest, resolve_track1_root  # noqa: E402
from floodnet_ssl.models import segformer_dependency_status  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check config, data, device, dependencies and local model path."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path; existing files are never overwritten.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml_config(args.config)
    data = config["data"]
    model = config["model"]
    training = config["training"]
    blockers: list[str] = []
    warnings: list[str] = []

    try:
        track1_root = resolve_track1_root(data["data_root"])
    except Exception as error:
        track1_root = None
        blockers.append(f"data_root: {error}")
    manifest_path = Path(data["manifest"]).expanduser().resolve()
    if not manifest_path.is_file():
        blockers.append(f"manifest does not exist: {manifest_path}")
        manifest_rows = 0
    else:
        manifest_rows = len(read_manifest(manifest_path))

    dependency_status = segformer_dependency_status()
    required_dependencies = (
        ("transformers", "safetensors")
        if bool(model.get("pretrained"))
        else ("transformers",)
    )
    for dependency in required_dependencies:
        installed = dependency_status[dependency]
        if not installed:
            blockers.append(f"missing dependency: {dependency}")

    configured_device = str(training["device"])
    if configured_device.startswith("cuda") and not torch.cuda.is_available():
        blockers.append("configured CUDA device is unavailable in this environment")

    pretrained_source = str(model.get("pretrained_model_name_or_path", ""))
    if not bool(model.get("pretrained")):
        local_model_found = None
    elif bool(model.get("local_files_only", True)):
        if Path(pretrained_source).expanduser().exists():
            local_model_found = True
        else:
            local_model_found = False
            warnings.append(
                "pretrained source is a model ID and local_files_only=true; "
                "the model must already exist in the Hugging Face cache"
            )
    else:
        local_model_found = Path(pretrained_source).expanduser().exists()

    report = {
        "ready": not blockers,
        "config": str(Path(args.config).resolve()),
        "data_root": str(track1_root) if track1_root else None,
        "manifest": str(manifest_path),
        "manifest_rows": manifest_rows,
        "configured_device": configured_device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "dependencies": dependency_status,
        "pretrained_source": pretrained_source,
        "local_model_path_found": local_model_found,
        "blockers": blockers,
        "warnings": warnings,
    }
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
