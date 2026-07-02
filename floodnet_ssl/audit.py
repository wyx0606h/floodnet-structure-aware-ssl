"""Post-extraction audit for FloodNet Track 1."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from .constants import CLASS_NAMES, EXPECTED_COUNTS, NUM_CLASSES
from .layout import (
    FloodNetSample,
    iter_labeled_samples,
    iter_unlabeled_samples,
    relative_posix,
    resolve_track1_root,
    write_csv,
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def combined_dhash(image: Image.Image) -> str:
    """Return a deterministic 128-bit horizontal+vertical difference hash."""

    gray = image.convert("L")
    horizontal = np.asarray(gray.resize((9, 8), Image.Resampling.BILINEAR))
    horizontal_bits = horizontal[:, 1:] > horizontal[:, :-1]
    vertical = np.asarray(gray.resize((8, 9), Image.Resampling.BILINEAR))
    vertical_bits = vertical[1:, :] > vertical[:-1, :]
    bits = np.concatenate([horizontal_bits.ravel(), vertical_bits.ravel()])
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:032x}"


def hamming_distance_hex(left: str, right: str) -> int:
    return bin(int(left, 16) ^ int(right, 16)).count("1")


def _read_class_mapping(track1_root: Path) -> dict[int, str]:
    mapping_path = track1_root / "class_mapping.csv"
    segmentation_mapping: dict[int, str] = {}
    with mapping_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) >= 5 and row[3].strip().isdigit():
                segmentation_mapping[int(row[3])] = row[4].strip()
    return segmentation_mapping


def _inspect_image(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
        perceptual_hash = combined_dhash(image)
    return {
        "width": width,
        "height": height,
        "image_mode": mode,
        "image_sha256": sha256_file(path),
        "image_dhash": perceptual_hash,
    }


def _inspect_mask(path: Path) -> dict[str, object]:
    with Image.open(path) as mask_image:
        mode = mask_image.mode
        mask = np.asarray(mask_image)
    if mask.ndim != 2:
        raise ValueError(f"Mask must be single-channel, got shape {mask.shape}: {path}")
    unique_values = np.unique(mask)
    invalid = unique_values[(unique_values < 0) | (unique_values >= NUM_CLASSES)]
    if invalid.size:
        raise ValueError(f"Invalid mask IDs {invalid.tolist()} in {path}")
    counts = np.bincount(mask.reshape(-1), minlength=NUM_CLASSES)[:NUM_CLASSES]
    total = int(mask.size)
    result: dict[str, object] = {
        "mask_mode": mode,
        "mask_sha256": sha256_file(path),
        "mask_width": int(mask.shape[1]),
        "mask_height": int(mask.shape[0]),
    }
    for class_id in range(NUM_CLASSES):
        pixels = int(counts[class_id])
        result[f"class_{class_id}_pixels"] = pixels
        result[f"class_{class_id}_present"] = int(pixels > 0)
        result[f"class_{class_id}_fraction"] = pixels / total
    return result


def _sample_row(sample: FloodNetSample, track1_root: Path) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_id": sample.sample_id,
        "official_split": sample.official_split,
        "scene_label": sample.scene_label,
        "image_path": relative_posix(sample.image_path, track1_root),
        "mask_path": relative_posix(sample.mask_path, track1_root),
        "exact_duplicate_group": "",
        "requires_image_resize": 0,
    }
    image_info = _inspect_image(sample.image_path)
    row.update(image_info)
    if sample.mask_path is not None:
        mask_info = _inspect_mask(sample.mask_path)
        row["requires_image_resize"] = int(
            image_info["width"] != mask_info["mask_width"]
            or image_info["height"] != mask_info["mask_height"]
        )
        row.update(mask_info)
    else:
        row.update({"mask_mode": "", "mask_sha256": "", "mask_width": "", "mask_height": ""})
        for class_id in range(NUM_CLASSES):
            row[f"class_{class_id}_pixels"] = ""
            row[f"class_{class_id}_present"] = ""
            row[f"class_{class_id}_fraction"] = ""
    return row


def _assign_exact_duplicate_groups(rows: list[dict[str, object]]) -> list[list[str]]:
    by_hash: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_hash[str(row["image_sha256"])].append(row)
    duplicate_groups: list[list[str]] = []
    for index, group in enumerate(
        sorted(
            (items for items in by_hash.values() if len(items) > 1),
            key=lambda items: tuple(str(item["sample_id"]) for item in items),
        ),
        start=1,
    ):
        group_id = f"exact_{index:04d}"
        ids = sorted(str(item["sample_id"]) for item in group)
        duplicate_groups.append(ids)
        for item in group:
            item["exact_duplicate_group"] = group_id
    return duplicate_groups


def _near_duplicate_candidates(
    labeled_rows: list[dict[str, object]],
    training_unlabeled_rows: list[dict[str, object]],
    max_hamming_distance: int,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    labeled = sorted(labeled_rows, key=lambda row: str(row["sample_id"]))
    comparison_rows = sorted(
        labeled_rows + training_unlabeled_rows,
        key=lambda row: (str(row["official_split"]), str(row["sample_id"])),
    )
    for left in labeled:
        for right in comparison_rows:
            if str(right["official_split"]) == "Train/Labeled":
                if str(right["sample_id"]) <= str(left["sample_id"]):
                    continue
            elif str(right["official_split"]) != "Train/Unlabeled":
                continue
            if left["image_sha256"] == right["image_sha256"]:
                continue
            distance = hamming_distance_hex(
                str(left["image_dhash"]), str(right["image_dhash"])
            )
            if distance <= max_hamming_distance:
                candidates.append(
                    {
                        "sample_id_a": left["sample_id"],
                        "sample_id_b": right["sample_id"],
                        "hamming_distance_128": distance,
                        "scene_label_a": left["scene_label"],
                        "scene_label_b": right["scene_label"],
                        "official_split_a": left["official_split"],
                        "official_split_b": right["official_split"],
                        "image_path_a": left["image_path"],
                        "image_path_b": right["image_path"],
                        "decision": "",
                        "review_notes": "",
                    }
                )
    return candidates


def _inventory_fingerprint(rows: Iterable[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: (str(item["official_split"]), str(item["sample_id"]))):
        digest.update(
            (
                f"{row['official_split']}\0{row['sample_id']}\0{row['image_path']}\0"
                f"{row['image_sha256']}\0{row['mask_sha256']}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def audit_dataset(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    expected_counts: dict[str, int] | None = EXPECTED_COUNTS,
    near_duplicate_hamming: int = 6,
) -> dict[str, object]:
    """Audit extracted data and create machine-readable reports.

    The output directory must not already exist, preventing accidental report
    replacement. Near-duplicate rows are candidates only; the ``decision``
    column must be reviewed before a canonical split is frozen.
    """

    track1_root = resolve_track1_root(data_root)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    errors: list[str] = []
    warnings: list[str] = []
    expected_mapping = dict(enumerate(CLASS_NAMES))
    actual_mapping = _read_class_mapping(track1_root)
    if actual_mapping != expected_mapping:
        errors.append(
            f"class_mapping.csv mismatch: expected={expected_mapping}, actual={actual_mapping}"
        )

    labeled_samples = list(iter_labeled_samples(track1_root))
    unlabeled_samples = list(iter_unlabeled_samples(track1_root))
    all_samples = labeled_samples + unlabeled_samples

    split_ids: dict[str, set[str]] = defaultdict(set)
    for sample in all_samples:
        normalized_id = sample.sample_id.casefold()
        if normalized_id in split_ids[sample.official_split]:
            errors.append(
                f"Duplicate ID in {sample.official_split}: {sample.sample_id}"
            )
        split_ids[sample.official_split].add(normalized_id)
    split_names = sorted(split_ids)
    for index, left_name in enumerate(split_names):
        for right_name in split_names[index + 1 :]:
            overlap = sorted(split_ids[left_name] & split_ids[right_name])
            if overlap:
                errors.append(
                    f"ID overlap between {left_name} and {right_name}: {overlap[:20]}"
                )

    rows = [_sample_row(sample, track1_root) for sample in all_samples]
    duplicate_groups = _assign_exact_duplicate_groups(rows)
    duplicate_rows_by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["exact_duplicate_group"]:
            duplicate_rows_by_group[str(row["exact_duplicate_group"])].append(row)
    for group_id, group_rows in duplicate_rows_by_group.items():
        official_splits = {str(row["official_split"]) for row in group_rows}
        if len(official_splits) > 1:
            errors.append(
                f"Exact duplicate image crosses official splits "
                f"{sorted(official_splits)} in {group_id}: "
                f"{[row['sample_id'] for row in group_rows]}"
            )
    labeled_rows = [
        row for row in rows if str(row["official_split"]) == "Train/Labeled"
    ]
    training_unlabeled_rows = [
        row for row in rows if str(row["official_split"]) == "Train/Unlabeled"
    ]
    near_candidates = _near_duplicate_candidates(
        labeled_rows,
        training_unlabeled_rows,
        max_hamming_distance=near_duplicate_hamming,
    )
    if near_candidates:
        warnings.append(
            f"{len(near_candidates)} near-duplicate candidates involving labeled "
            "images require review"
        )

    actual_counts = {
        "labeled_flooded": sum(
            sample.scene_label == "Flooded" for sample in labeled_samples
        ),
        "labeled_non_flooded": sum(
            sample.scene_label == "Non-Flooded" for sample in labeled_samples
        ),
        "unlabeled": len(split_ids["Train/Unlabeled"]),
        "validation": len(split_ids["Validation"]),
        "test": len(split_ids["Test"]),
    }
    if expected_counts is not None and actual_counts != expected_counts:
        errors.append(
            f"Dataset counts mismatch: expected={expected_counts}, actual={actual_counts}"
        )

    mode_counts = Counter(str(row["image_mode"]) for row in rows)
    dimension_counts = Counter(
        (int(row["width"]), int(row["height"])) for row in rows
    )
    labeled_dimension_pairs = Counter(
        (
            int(row["width"]),
            int(row["height"]),
            int(row["mask_width"]),
            int(row["mask_height"]),
        )
        for row in labeled_rows
    )
    resize_required_count = sum(
        int(row["requires_image_resize"]) for row in labeled_rows
    )
    if resize_required_count:
        warnings.append(
            f"{resize_required_count} labeled RGB images require bilinear alignment "
            "to the mask grid during loading"
        )
    mask_mode_counts = Counter(
        str(row["mask_mode"]) for row in labeled_rows
    )

    class_statistics: list[dict[str, object]] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        pixels = sum(int(row[f"class_{class_id}_pixels"]) for row in labeled_rows)
        images = sum(int(row[f"class_{class_id}_present"]) for row in labeled_rows)
        total_labeled_pixels = sum(
            int(row["width"]) * int(row["height"]) for row in labeled_rows
        )
        class_statistics.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "pixel_count": pixels,
                "pixel_fraction": pixels / total_labeled_pixels,
                "image_count": images,
                "image_fraction": images / len(labeled_rows),
            }
        )

    inventory_fields = [
        "sample_id",
        "official_split",
        "scene_label",
        "image_path",
        "mask_path",
        "width",
        "height",
        "image_mode",
        "mask_mode",
        "image_sha256",
        "mask_sha256",
        "image_dhash",
        "exact_duplicate_group",
        "requires_image_resize",
    ]
    for class_id in range(NUM_CLASSES):
        inventory_fields.extend(
            [
                f"class_{class_id}_pixels",
                f"class_{class_id}_present",
                f"class_{class_id}_fraction",
            ]
        )
    write_csv(output / "inventory.csv", rows, inventory_fields)
    write_csv(
        output / "class_statistics.csv",
        class_statistics,
        [
            "class_id",
            "class_name",
            "pixel_count",
            "pixel_fraction",
            "image_count",
            "image_fraction",
        ],
    )

    duplicate_rows = [
        {"group_id": f"exact_{index:04d}", "sample_id": sample_id}
        for index, group in enumerate(duplicate_groups, start=1)
        for sample_id in group
    ]
    write_csv(
        output / "exact_duplicate_groups.csv",
        duplicate_rows,
        ["group_id", "sample_id"],
    )
    write_csv(
        output / "near_duplicate_candidates.csv",
        near_candidates,
        [
            "sample_id_a",
            "sample_id_b",
            "hamming_distance_128",
            "scene_label_a",
            "scene_label_b",
            "official_split_a",
            "official_split_b",
            "image_path_a",
            "image_path_b",
            "decision",
            "review_notes",
        ],
    )

    summary: dict[str, object] = {
        "passed": not errors,
        "track1_root": str(track1_root),
        "actual_counts": actual_counts,
        "expected_counts": expected_counts,
        "class_mapping": actual_mapping,
        "image_modes": dict(mode_counts),
        "mask_modes": dict(mask_mode_counts),
        "image_dimensions": {
            f"{width}x{height}": count
            for (width, height), count in sorted(dimension_counts.items())
        },
        "labeled_image_mask_dimension_pairs": {
            f"{image_width}x{image_height}->{mask_width}x{mask_height}": count
            for (
                image_width,
                image_height,
                mask_width,
                mask_height,
            ), count in sorted(labeled_dimension_pairs.items())
        },
        "labeled_images_requiring_resize": resize_required_count,
        "exact_duplicate_group_count": len(duplicate_groups),
        "near_duplicate_candidate_count": len(near_candidates),
        "near_duplicate_hamming_threshold_128": near_duplicate_hamming,
        "inventory_sha256": _inventory_fingerprint(rows),
        "errors": errors,
        "warnings": warnings,
    }
    (output / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
