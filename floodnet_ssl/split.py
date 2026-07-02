"""Deterministic group-aware multi-label split generation."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .constants import CLASS_NAMES, NUM_CLASSES
from .layout import read_manifest, write_csv

ALGORITHM_VERSION = "grouped_multilabel_greedy_swap_v1"


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


@dataclass(frozen=True)
class SplitResult:
    assignments: dict[str, str]
    feature_names: tuple[str, ...]
    objective: float
    groups: dict[str, tuple[str, ...]]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_labeled_inventory(inventory_path: Path) -> list[dict[str, str]]:
    rows = [
        row
        for row in read_manifest(inventory_path)
        if row.get("official_split") == "Train/Labeled"
    ]
    if not rows:
        raise ValueError(f"No Train/Labeled rows found in {inventory_path}")
    ids = [row["sample_id"].casefold() for row in rows]
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate labeled sample IDs: {duplicates}")
    required = {"scene_label"}
    for class_id in range(NUM_CLASSES):
        required.update(
            {
                f"class_{class_id}_present",
                f"class_{class_id}_fraction",
            }
        )
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Inventory is missing stratification columns: {sorted(missing)}")
    return sorted(rows, key=lambda row: row["sample_id"])


def _read_near_duplicate_review(
    review_path: Path | None,
) -> list[dict[str, str]]:
    if review_path is None or not review_path.exists():
        return []
    with review_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _build_groups(
    rows: list[dict[str, str]],
    all_inventory_rows: list[dict[str, str]],
    near_duplicate_review: Path | None,
    *,
    allow_unreviewed: bool,
) -> tuple[dict[str, tuple[str, ...]], set[str]]:
    canonical_ids = {row["sample_id"].casefold(): row["sample_id"] for row in rows}
    all_id_counts = Counter(row["sample_id"].casefold() for row in all_inventory_rows)
    repeated_all_ids = sorted(
        sample_id for sample_id, count in all_id_counts.items() if count > 1
    )
    if repeated_all_ids:
        raise ValueError(
            f"Inventory contains IDs repeated across official splits: {repeated_all_ids}"
        )
    all_rows_by_id = {row["sample_id"].casefold(): row for row in all_inventory_rows}
    union_find = UnionFind(canonical_ids.values())
    forced_train_ids: set[str] = set()

    exact_groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        group = row.get("exact_duplicate_group", "").strip()
        if group:
            exact_groups[group].append(row["sample_id"])
    for members in exact_groups.values():
        for member in members[1:]:
            union_find.union(members[0], member)

    review_rows = _read_near_duplicate_review(near_duplicate_review)
    unresolved: list[tuple[str, str]] = []
    for review in review_rows:
        left_raw = review.get("sample_id_a", "").strip()
        right_raw = review.get("sample_id_b", "").strip()
        if not left_raw or not right_raw:
            raise ValueError("Near-duplicate review row is missing sample IDs")
        try:
            left_row = all_rows_by_id[left_raw.casefold()]
            right_row = all_rows_by_id[right_raw.casefold()]
        except KeyError as error:
            raise ValueError(f"Near-duplicate review references unknown ID: {error}") from error
        decision = review.get("decision", "").strip().casefold().replace("-", "_")
        if decision in {"same_scene", "same_group", "duplicate"}:
            labeled = [
                row
                for row in (left_row, right_row)
                if row.get("official_split") == "Train/Labeled"
            ]
            if len(labeled) == 2:
                union_find.union(labeled[0]["sample_id"], labeled[1]["sample_id"])
            elif len(labeled) == 1:
                other = right_row if labeled[0] is left_row else left_row
                if other.get("official_split") == "Train/Unlabeled":
                    forced_train_ids.add(labeled[0]["sample_id"])
            else:
                raise ValueError(
                    f"Near-duplicate review contains no labeled sample: "
                    f"{left_raw}/{right_raw}"
                )
        elif decision in {"different_scene", "different_group", "not_duplicate"}:
            continue
        elif not decision:
            unresolved.append((left_raw, right_raw))
        else:
            raise ValueError(
                f"Unsupported near-duplicate decision '{decision}' for {left}/{right}"
            )
    if unresolved and not allow_unreviewed:
        raise ValueError(
            f"{len(unresolved)} near-duplicate candidates are unreviewed; "
            "fill the decision column before freezing the canonical split"
        )

    grouped: dict[str, list[str]] = defaultdict(list)
    for sample_id in canonical_ids.values():
        grouped[union_find.find(sample_id)].append(sample_id)
    groups = {
        root: tuple(sorted(members))
        for root, members in sorted(grouped.items())
    }
    return groups, forced_train_ids


def _rank_quantile_bins(values: list[tuple[str, float]], bins: int = 3) -> dict[str, int]:
    ordered = sorted(values, key=lambda item: (item[1], item[0]))
    count = len(ordered)
    result: dict[str, int] = {}
    for rank, (sample_id, _) in enumerate(ordered):
        result[sample_id] = min(bins - 1, rank * bins // max(count, 1))
    return result


def build_stratification_features(
    rows: list[dict[str, str]],
) -> tuple[np.ndarray, tuple[str, ...]]:
    feature_columns: list[np.ndarray] = []
    feature_names: list[str] = []
    sample_ids = [row["sample_id"] for row in rows]

    for scene_label in ("Flooded", "Non-Flooded"):
        feature_columns.append(
            np.asarray([row["scene_label"] == scene_label for row in rows], dtype=np.float64)
        )
        feature_names.append(f"scene:{scene_label}")

    for class_id, class_name in enumerate(CLASS_NAMES):
        presence = np.asarray(
            [int(row[f"class_{class_id}_present"]) for row in rows],
            dtype=np.float64,
        )
        feature_columns.append(presence)
        feature_names.append(f"present:{class_id}:{class_name}")

        positive_values = [
            (row["sample_id"], float(row[f"class_{class_id}_fraction"]))
            for row in rows
            if int(row[f"class_{class_id}_present"])
        ]
        quantile_bins = _rank_quantile_bins(positive_values, bins=3)
        for bin_index in range(3):
            feature_columns.append(
                np.asarray(
                    [
                        quantile_bins.get(sample_id, -1) == bin_index
                        for sample_id in sample_ids
                    ],
                    dtype=np.float64,
                )
            )
            feature_names.append(
                f"fraction_tertile:{class_id}:{class_name}:{bin_index + 1}"
            )

    return np.column_stack(feature_columns), tuple(feature_names)


def _objective(
    assignments: np.ndarray,
    sample_features: np.ndarray,
    target_sizes: np.ndarray,
) -> float:
    total = sample_features.sum(axis=0)
    target = target_sizes[:, None] / target_sizes.sum() * total[None, :]
    actual = np.vstack(
        [
            sample_features[assignments == split_index].sum(axis=0)
            for split_index in range(len(target_sizes))
        ]
    )
    normalized_error = np.abs(actual - target) / np.maximum(target, 1.0)
    return float(normalized_error.mean())


def stratified_group_split(
    rows: list[dict[str, str]],
    groups: dict[str, tuple[str, ...]],
    *,
    split_sizes: tuple[int, int, int] = (278, 60, 60),
    seed: int = 20260624,
    optimization_steps: int = 10000,
    forced_train_ids: set[str] | None = None,
) -> SplitResult:
    if sum(split_sizes) != len(rows):
        raise ValueError(
            f"Split sizes {split_sizes} do not sum to {len(rows)} labeled samples"
        )
    if any(size <= 0 for size in split_sizes):
        raise ValueError(f"Split sizes must be positive: {split_sizes}")

    sample_features, feature_names = build_stratification_features(rows)
    sample_index = {row["sample_id"]: index for index, row in enumerate(rows)}
    group_items = list(groups.items())
    group_features = np.vstack(
        [
            sample_features[[sample_index[sample_id] for sample_id in members]].sum(axis=0)
            for _, members in group_items
        ]
    )
    group_sizes = np.asarray([len(members) for _, members in group_items], dtype=np.int64)
    label_totals = np.maximum(sample_features.sum(axis=0), 1.0)
    rarity = (group_features / label_totals[None, :]).sum(axis=1)

    rng = random.Random(seed)
    tie_breakers = [rng.random() for _ in group_items]
    order = sorted(
        range(len(group_items)),
        key=lambda index: (
            -int(group_sizes[index]),
            -float(rarity[index]),
            tie_breakers[index],
            group_items[index][0],
        ),
    )

    target_sizes = np.asarray(split_sizes, dtype=np.int64)
    remaining = target_sizes.copy()
    target_features = (
        target_sizes[:, None] / target_sizes.sum() * sample_features.sum(axis=0)[None, :]
    )
    current_features = np.zeros_like(target_features)
    group_assignments = np.full(len(group_items), -1, dtype=np.int64)
    forced_train_ids = forced_train_ids or set()
    forced_group_indices = {
        group_index
        for group_index, (_, members) in enumerate(group_items)
        if any(sample_id in forced_train_ids for sample_id in members)
    }
    for group_index in sorted(forced_group_indices):
        size = group_sizes[group_index]
        if remaining[0] < size:
            raise ValueError(
                "Reviewed labeled/unlabeled near-duplicate groups exceed Local Train capacity"
            )
        group_assignments[group_index] = 0
        remaining[0] -= size
        current_features[0] += group_features[group_index]

    for group_index in order:
        if group_index in forced_group_indices:
            continue
        size = group_sizes[group_index]
        feasible = [
            split_index
            for split_index in range(3)
            if remaining[split_index] >= size
        ]
        if not feasible:
            raise ValueError(
                "Could not satisfy exact split sizes while keeping duplicate groups intact; "
                "review duplicate grouping or choose compatible sizes"
            )
        active = group_features[group_index] > 0
        scored: list[tuple[float, float, int]] = []
        for split_index in feasible:
            deficits = np.maximum(
                target_features[split_index] - current_features[split_index], 0.0
            )
            label_score = float(
                (
                    deficits[active]
                    / np.maximum(target_features[split_index, active], 1.0)
                ).sum()
            )
            capacity_score = float(remaining[split_index] / target_sizes[split_index])
            scored.append((label_score + 0.25 * capacity_score, rng.random(), split_index))
        _, _, chosen = max(scored)
        group_assignments[group_index] = chosen
        remaining[chosen] -= size
        current_features[chosen] += group_features[group_index]

    if np.any(remaining != 0):
        raise AssertionError(f"Internal split capacity error: remaining={remaining.tolist()}")

    sample_assignments = np.full(len(rows), -1, dtype=np.int64)
    for group_index, (_, members) in enumerate(group_items):
        for sample_id in members:
            sample_assignments[sample_index[sample_id]] = group_assignments[group_index]

    best_objective = _objective(sample_assignments, sample_features, target_sizes)
    by_size: dict[int, list[int]] = defaultdict(list)
    for group_index, size in enumerate(group_sizes.tolist()):
        if group_index not in forced_group_indices:
            by_size[size].append(group_index)
    swappable_sizes = [size for size, indices in by_size.items() if len(indices) >= 2]

    for _ in range(optimization_steps):
        if not swappable_sizes:
            break
        size = rng.choice(swappable_sizes)
        candidates = by_size[size]
        left, right = rng.sample(candidates, 2)
        if group_assignments[left] == group_assignments[right]:
            continue
        left_split = int(group_assignments[left])
        right_split = int(group_assignments[right])
        group_assignments[left], group_assignments[right] = right_split, left_split
        for sample_id in group_items[left][1]:
            sample_assignments[sample_index[sample_id]] = right_split
        for sample_id in group_items[right][1]:
            sample_assignments[sample_index[sample_id]] = left_split
        candidate_objective = _objective(
            sample_assignments, sample_features, target_sizes
        )
        if candidate_objective <= best_objective:
            best_objective = candidate_objective
        else:
            group_assignments[left], group_assignments[right] = left_split, right_split
            for sample_id in group_items[left][1]:
                sample_assignments[sample_index[sample_id]] = left_split
            for sample_id in group_items[right][1]:
                sample_assignments[sample_index[sample_id]] = right_split

    split_names = ("train", "validation", "test")
    assignments = {
        row["sample_id"]: split_names[int(sample_assignments[index])]
        for index, row in enumerate(rows)
    }
    return SplitResult(
        assignments=assignments,
        feature_names=feature_names,
        objective=best_objective,
        groups=groups,
    )


def _distribution_summary(
    rows: list[dict[str, str]], assignments: dict[str, str]
) -> dict[str, object]:
    summary: dict[str, object] = {}
    for split_name in ("train", "validation", "test"):
        split_rows = [
            row for row in rows if assignments[row["sample_id"]] == split_name
        ]
        class_summary = {}
        for class_id, class_name in enumerate(CLASS_NAMES):
            class_summary[str(class_id)] = {
                "name": class_name,
                "image_count": sum(
                    int(row[f"class_{class_id}_present"]) for row in split_rows
                ),
                "pixel_fraction_mean": float(
                    np.mean(
                        [float(row[f"class_{class_id}_fraction"]) for row in split_rows]
                    )
                ),
            }
        summary[split_name] = {
            "count": len(split_rows),
            "scene_counts": dict(Counter(row["scene_label"] for row in split_rows)),
            "classes": class_summary,
        }
    return summary


def create_versioned_split(
    inventory_path: str | Path,
    output_dir: str | Path,
    *,
    near_duplicate_review: str | Path | None = None,
    split_sizes: tuple[int, int, int] = (278, 60, 60),
    seed: int = 20260624,
    optimization_steps: int = 10000,
    allow_unreviewed: bool = False,
) -> dict[str, object]:
    inventory = Path(inventory_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing split directory: {output}")

    all_inventory_rows = read_manifest(inventory)
    rows = _load_labeled_inventory(inventory)
    review = (
        Path(near_duplicate_review).expanduser().resolve()
        if near_duplicate_review is not None
        else inventory.parent / "near_duplicate_candidates.csv"
    )
    groups, forced_train_ids = _build_groups(
        rows,
        all_inventory_rows,
        review if review.exists() else None,
        allow_unreviewed=allow_unreviewed,
    )
    result = stratified_group_split(
        rows,
        groups,
        split_sizes=split_sizes,
        seed=seed,
        optimization_steps=optimization_steps,
        forced_train_ids=forced_train_ids,
    )

    output.mkdir(parents=True, exist_ok=False)
    output_fields = [
        "sample_id",
        "split",
        "scene_label",
        "image_path",
        "mask_path",
        "group_id",
    ]
    member_to_group = {
        sample_id: group_id
        for group_id, members in groups.items()
        for sample_id in members
    }
    output_rows = [
        {
            "sample_id": row["sample_id"],
            "split": result.assignments[row["sample_id"]],
            "scene_label": row["scene_label"],
            "image_path": row["image_path"],
            "mask_path": row["mask_path"],
            "group_id": member_to_group[row["sample_id"]],
        }
        for row in rows
    ]
    write_csv(output / "manifest.csv", output_rows, output_fields)
    for split_name in ("train", "validation", "test"):
        write_csv(
            output / f"{split_name}.csv",
            [row for row in output_rows if row["split"] == split_name],
            output_fields,
        )

    summary = {
        "algorithm": ALGORITHM_VERSION,
        "seed": seed,
        "optimization_steps": optimization_steps,
        "split_sizes": {
            "train": split_sizes[0],
            "validation": split_sizes[1],
            "test": split_sizes[2],
        },
        "inventory_path": str(inventory),
        "inventory_file_sha256": _sha256_file(inventory),
        "near_duplicate_review_path": str(review) if review.exists() else None,
        "near_duplicate_review_sha256": _sha256_file(review) if review.exists() else None,
        "group_count": len(groups),
        "largest_group_size": max(len(members) for members in groups.values()),
        "forced_train_sample_ids": sorted(forced_train_ids),
        "stratification_feature_names": list(result.feature_names),
        "objective": result.objective,
        "distribution": _distribution_summary(rows, result.assignments),
    }
    (output / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# FloodNet local 278/60/60 split\n\n"
        "This directory is generated from the versioned audit inventory. "
        "Local Test must not be used for training, pseudo-label generation, "
        "threshold selection, or checkpoint selection.\n\n"
        f"- Algorithm: `{ALGORITHM_VERSION}`\n"
        f"- Seed: `{seed}`\n"
        f"- Inventory SHA-256: `{summary['inventory_file_sha256']}`\n",
        encoding="utf-8",
    )
    return summary
