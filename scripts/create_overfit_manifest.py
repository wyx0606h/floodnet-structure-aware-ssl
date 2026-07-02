"""Create a deterministic four-image Local Train overfit manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from floodnet_ssl.constants import CLASS_NAMES, NUM_CLASSES  # noqa: E402
from floodnet_ssl.layout import (  # noqa: E402
    read_manifest,
    resolve_track1_root,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select four Local Train images that jointly cover all ten classes."
    )
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--local-train", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _classes(row: dict[str, str]) -> set[int]:
    return {
        class_id
        for class_id in range(NUM_CLASSES)
        if row.get(f"class_{class_id}_present") == "1"
    }


def main() -> int:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite overfit manifest: {output}")

    inventory = {
        row["sample_id"]: row
        for row in read_manifest(args.inventory.expanduser().resolve())
        if row.get("official_split") == "Train/Labeled"
    }
    train_rows = read_manifest(args.local_train.expanduser().resolve())
    candidates = [inventory[row["sample_id"]] for row in train_rows]
    data_root = resolve_track1_root(args.data_root)

    crop_candidates: list[dict[str, object]] = []
    for row in candidates:
        with Image.open(data_root / row["mask_path"]) as handle:
            mask = np.asarray(handle)
        preview = np.asarray(
            Image.fromarray(mask).resize((500, 375), Image.Resampling.NEAREST)
        )
        for class_id in np.unique(preview):
            coordinates = np.argwhere(preview == class_id)
            for quantile in (0.2, 0.5, 0.8):
                coordinate_index = min(
                    len(coordinates) - 1, int(len(coordinates) * quantile)
                )
                preview_row, preview_column = coordinates[coordinate_index]
                center_row = round(preview_row * mask.shape[0] / preview.shape[0])
                center_column = round(
                    preview_column * mask.shape[1] / preview.shape[1]
                )
                top = min(max(center_row - 256, 0), mask.shape[0] - 512)
                left = min(max(center_column - 256, 0), mask.shape[1] - 512)
                crop_classes = frozenset(
                    int(value)
                    for value in np.unique(mask[top : top + 512, left : left + 512])
                )
                crop_candidates.append(
                    {
                        "row": row,
                        "top": top,
                        "left": left,
                        "classes": crop_classes,
                    }
                )

    selected: list[dict[str, object]] = []
    covered: set[int] = set()
    affected = {1, 3}
    while len(selected) < 4:
        selected_ids = {
            str(candidate["row"]["sample_id"]) for candidate in selected
        }
        remaining = [
            candidate
            for candidate in crop_candidates
            if str(candidate["row"]["sample_id"]) not in selected_ids
        ]
        best = max(
            remaining,
            key=lambda candidate: (
                len(set(candidate["classes"]) - covered)
                + 10
                * len(
                    (set(candidate["classes"]) & affected) - covered
                ),
                len(candidate["classes"]),
                str(candidate["row"]["sample_id"]),
            ),
        )
        selected.append(best)
        covered.update(best["classes"])

    if covered != set(range(NUM_CLASSES)):
        raise RuntimeError(
            f"Four-image selection failed to cover all classes: {sorted(covered)}"
        )
    if not affected.issubset(covered):
        raise RuntimeError("Overfit selection must cover affected class IDs 1 and 3")

    output.mkdir(parents=True, exist_ok=False)
    fieldnames = [
        "sample_id",
        "split",
        "scene_label",
        "image_path",
        "mask_path",
        "target_class_ids",
        "crop_top",
        "crop_left",
        "crop_height",
        "crop_width",
        "crop_class_ids",
        "selection_reason",
    ]
    rows = [
        {
            "sample_id": candidate["row"]["sample_id"],
            "split": "train",
            "scene_label": candidate["row"]["scene_label"],
            "image_path": candidate["row"]["image_path"],
            "mask_path": candidate["row"]["mask_path"],
            "target_class_ids": ";".join(
                str(class_id)
                for class_id in sorted(set(candidate["classes"]) & affected)
            ),
            "crop_top": candidate["top"],
            "crop_left": candidate["left"],
            "crop_height": 512,
            "crop_width": 512,
            "crop_class_ids": ";".join(
                str(class_id) for class_id in sorted(candidate["classes"])
            ),
            "selection_reason": (
                "greedy_fixed_crop_all_class_coverage_with_affected_priority"
            ),
        }
        for candidate in selected
    ]
    write_csv(output / "manifest.csv", rows, fieldnames)
    summary = {
        "sample_ids": [candidate["row"]["sample_id"] for candidate in selected],
        "crops": [
            {
                "sample_id": candidate["row"]["sample_id"],
                "top": candidate["top"],
                "left": candidate["left"],
                "height": 512,
                "width": 512,
                "class_ids": sorted(candidate["classes"]),
            }
            for candidate in selected
        ],
        "covered_class_ids": sorted(covered),
        "covered_class_names": [CLASS_NAMES[index] for index in sorted(covered)],
        "selection_algorithm": (
            "greedy_fixed_crop_all_class_coverage_with_affected_priority_v1"
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
