"""Create manifests for FloodNet-Supervised_v1.0 official train/val/test split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from floodnet_ssl.constants import SUPERVISED_EXPECTED_COUNTS  # noqa: E402
from floodnet_ssl.layout import resolve_track1_root  # noqa: E402

SPLIT_LAYOUT = {
    "train": ("train/train-org-img", "train/train-label-img"),
    "validation": ("val/val-org-img", "val/val-label-img"),
    "test": ("test/test-org-img", "test/test-label-img"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate official supervised FloodNet train/val/test manifests."
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _files_by_stem(directory: Path, suffixes: set[str], *, strip_lab: bool = False) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.casefold() not in suffixes:
            continue
        stem = path.stem
        if strip_lab and stem.endswith("_lab"):
            stem = stem[:-4]
        key = stem.casefold()
        if key in files:
            raise ValueError(f"Duplicate sample id {stem!r} in {directory}")
        files[key] = path
    return files


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def build_rows(data_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = resolve_track1_root(data_root)
    all_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"data_root": str(root), "splits": {}}
    seen_ids: dict[str, str] = {}
    for split, (image_rel, mask_rel) in SPLIT_LAYOUT.items():
        image_dir = root / image_rel
        mask_dir = root / mask_rel
        images = _files_by_stem(image_dir, {".jpg", ".jpeg", ".png"})
        masks = _files_by_stem(mask_dir, {".png"}, strip_lab=True)
        missing_masks = sorted(set(images) - set(masks))
        missing_images = sorted(set(masks) - set(images))
        if missing_masks or missing_images:
            raise ValueError(
                f"Image/mask mismatch for {split}: "
                f"missing_masks={missing_masks[:10]}, missing_images={missing_images[:10]}"
            )
        duplicate_ids = sorted(set(images) & set(seen_ids))
        if duplicate_ids:
            raise ValueError(f"Duplicate sample ids across splits: {duplicate_ids[:10]}")
        rows: list[dict[str, Any]] = []
        for key in sorted(images):
            seen_ids[key] = split
            row = {
                "sample_id": images[key].stem,
                "split": split,
                "scene_label": "",
                "image_path": _relative(images[key], root),
                "mask_path": _relative(masks[key], root),
                "official_split": split,
            }
            rows.append(row)
            all_rows.append(row)
        expected = SUPERVISED_EXPECTED_COUNTS[split]
        if len(rows) != expected:
            raise ValueError(f"Expected {expected} {split} samples, found {len(rows)}")
        summary["splits"][split] = {"samples": len(rows)}
    summary["total_samples"] = len(all_rows)
    return all_rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    fieldnames = ["sample_id", "split", "scene_label", "image_path", "mask_path", "official_split"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows, summary = build_rows(args.data_root)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "manifest.csv", rows)
    for split in SPLIT_LAYOUT:
        write_csv(output_dir / f"{split}.csv", [row for row in rows if row["split"] == split])
    summary_path = output_dir / "split_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {summary_path}")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())