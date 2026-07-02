"""FloodNet supervised protocol split construction."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .constants import EXPECTED_COUNTS, SUPERVISED_EXPECTED_COUNTS
from .layout import (
    _files_by_stem,
    iter_labeled_samples,
    iter_unlabeled_samples,
    relative_posix,
    resolve_challenge_root,
    resolve_supervised_root,
)

DEFAULT_PROTOCOL_COUNTS = {
    "challenge_labeled": 398,
    "challenge_unlabeled": 1047,
    "full_train": 1445,
    "validation": 450,
    "test": 448,
}

LIST_FILENAMES = {
    "challenge_labeled": "challenge_labeled_398.txt",
    "challenge_unlabeled": "challenge_unlabeled_1047.txt",
    "full_train": "full_train_1445.txt",
    "validation": "val_450.txt",
    "test": "test_448.txt",
}

MANIFEST_FILENAMES = {
    "sup398": "sup398_manifest.csv",
    "full1445": "full1445_manifest.csv",
}


@dataclass(frozen=True)
class SupervisedFilePair:
    sample_id: str
    image_path: Path
    mask_path: Path


def _read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_ids(path: Path, ids: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def _check_count(name: str, ids: list[str], expected_counts: Mapping[str, int]) -> None:
    expected = expected_counts[name]
    if len(ids) != expected:
        raise ValueError(f"Expected {expected} IDs for {name}, found {len(ids)}")
    if len(set(id.casefold() for id in ids)) != len(ids):
        raise ValueError(f"Duplicate sample IDs in {name}")


def _supervised_pairs(root: Path, split: str) -> dict[str, SupervisedFilePair]:
    layout = {
        "train": ("train/train-org-img", "train/train-label-img"),
        "validation": ("val/val-org-img", "val/val-label-img"),
        "test": ("test/test-org-img", "test/test-label-img"),
    }
    if split not in layout:
        raise ValueError(f"Unsupported supervised split: {split}")
    image_rel, mask_rel = layout[split]
    images = _files_by_stem(root / image_rel, {".jpg", ".jpeg", ".png"})
    masks = _files_by_stem(root / mask_rel, {".png"}, required_stem_suffix="_lab")
    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        raise ValueError(
            f"Image/mask mismatch for supervised {split}: "
            f"missing_masks={missing_masks[:10]}, missing_images={missing_images[:10]}"
        )
    return {
        key: SupervisedFilePair(images[key].stem, images[key], masks[key])
        for key in sorted(images)
    }


def _challenge_labeled_ids(challenge_root: Path) -> list[str]:
    return sorted(sample.sample_id for sample in iter_labeled_samples(challenge_root))


def _challenge_unlabeled_ids(challenge_root: Path) -> list[str]:
    return sorted(
        sample.sample_id
        for sample in iter_unlabeled_samples(challenge_root)
        if sample.official_split == "Train/Unlabeled"
    )


def _rows_from_ids(ids: Iterable[str], pairs: Mapping[str, SupervisedFilePair], root: Path, split: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for sample_id in ids:
        key = sample_id.casefold()
        if key not in pairs:
            raise ValueError(f"Sample ID {sample_id} is missing from supervised {split} pairs")
        pair = pairs[key]
        rows.append(
            {
                "sample_id": pair.sample_id,
                "split": split,
                "scene_label": "",
                "image_path": relative_posix(pair.image_path, root),
                "mask_path": relative_posix(pair.mask_path, root),
                "official_split": split,
            }
        )
    return rows


def _write_manifest(path: Path, rows: list[dict[str, str]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "split", "scene_label", "image_path", "mask_path", "official_split"],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_floodnet_splits(
    *,
    supervised_root: str | Path,
    challenge_root: str | Path,
    output_dir: str | Path = "splits",
    overwrite: bool = False,
    expected_counts: Mapping[str, int] = DEFAULT_PROTOCOL_COUNTS,
) -> dict[str, object]:
    supervised = resolve_supervised_root(supervised_root)
    challenge = resolve_challenge_root(challenge_root)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    challenge_labeled = _challenge_labeled_ids(challenge)
    challenge_unlabeled = _challenge_unlabeled_ids(challenge)
    train_pairs = _supervised_pairs(supervised, "train")
    val_pairs = _supervised_pairs(supervised, "validation")
    test_pairs = _supervised_pairs(supervised, "test")

    full_train = sorted(pair.sample_id for pair in train_pairs.values())
    validation = sorted(pair.sample_id for pair in val_pairs.values())
    test = sorted(pair.sample_id for pair in test_pairs.values())

    lists = {
        "challenge_labeled": challenge_labeled,
        "challenge_unlabeled": challenge_unlabeled,
        "full_train": full_train,
        "validation": validation,
        "test": test,
    }
    for name, ids in lists.items():
        _check_count(name, ids, expected_counts)

    labeled_set = {sid.casefold() for sid in challenge_labeled}
    unlabeled_set = {sid.casefold() for sid in challenge_unlabeled}
    full_set = {sid.casefold() for sid in full_train}
    if labeled_set & unlabeled_set:
        raise ValueError("Challenge labeled and unlabeled ID lists overlap")
    if labeled_set | unlabeled_set != full_set:
        missing = sorted((labeled_set | unlabeled_set) - full_set)[:10]
        extra = sorted(full_set - (labeled_set | unlabeled_set))[:10]
        raise ValueError(
            "Challenge labeled ∪ unlabeled does not equal full supervised train: "
            f"missing_from_full={missing}, extra_in_full={extra}"
        )

    for name, filename in LIST_FILENAMES.items():
        path = output / filename
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing split list: {path}")
        _write_ids(path, lists[name])

    sup398_rows = []
    sup398_rows.extend(_rows_from_ids(challenge_labeled, train_pairs, supervised, "train"))
    sup398_rows.extend(_rows_from_ids(validation, val_pairs, supervised, "validation"))
    sup398_rows.extend(_rows_from_ids(test, test_pairs, supervised, "test"))
    full_rows = []
    full_rows.extend(_rows_from_ids(full_train, train_pairs, supervised, "train"))
    full_rows.extend(_rows_from_ids(validation, val_pairs, supervised, "validation"))
    full_rows.extend(_rows_from_ids(test, test_pairs, supervised, "test"))
    _write_manifest(output / MANIFEST_FILENAMES["sup398"], sup398_rows, overwrite=overwrite)
    _write_manifest(output / MANIFEST_FILENAMES["full1445"], full_rows, overwrite=overwrite)

    summary = {
        "supervised_root": str(supervised),
        "challenge_root": str(challenge),
        "output_dir": str(output),
        "counts": {name: len(ids) for name, ids in lists.items()},
        "manifests": {
            "sup398": str(output / MANIFEST_FILENAMES["sup398"]),
            "full1445": str(output / MANIFEST_FILENAMES["full1445"]),
        },
        "txt_lists": {name: str(output / filename) for name, filename in LIST_FILENAMES.items()},
        "checks": {
            "labeled_unlabeled_disjoint": True,
            "challenge_union_equals_full_train": True,
            "all_supervised_masks_present": True,
        },
    }
    summary_path = output / "floodnet_split_summary.json"
    if summary_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing split summary: {summary_path}")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
