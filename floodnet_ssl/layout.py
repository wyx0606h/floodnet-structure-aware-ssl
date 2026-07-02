"""FloodNet directory discovery and manifest helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from .constants import SCENE_DIRECTORIES, SUPERVISED_ROOT_NAME, TRACK1_ROOT_NAME


@dataclass(frozen=True)
class FloodNetSample:
    sample_id: str
    official_split: str
    scene_label: str
    image_path: Path
    mask_path: Path | None


def is_supervised_root(path: Path) -> bool:
    return (
        (path / "train" / "train-org-img").is_dir()
        and (path / "train" / "train-label-img").is_dir()
        and (path / "val" / "val-org-img").is_dir()
        and (path / "val" / "val-label-img").is_dir()
        and (path / "test" / "test-org-img").is_dir()
        and (path / "test" / "test-label-img").is_dir()
    )


def is_challenge_root(path: Path) -> bool:
    return (path / "class_mapping.csv").is_file() and (path / "Train").is_dir()


def _resolve_root(data_root: str | Path, *, prefer: str | None = None) -> Path:
    root = Path(data_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"FloodNet root does not exist: {root}")

    if prefer == "supervised":
        if is_supervised_root(root):
            return root
        supervised = root / SUPERVISED_ROOT_NAME
        if is_supervised_root(supervised):
            return supervised.resolve()
    elif prefer == "challenge":
        if is_challenge_root(root):
            return root
        exact = root / TRACK1_ROOT_NAME
        if is_challenge_root(exact):
            return exact.resolve()
    else:
        if is_supervised_root(root) or is_challenge_root(root):
            return root
        supervised = root / SUPERVISED_ROOT_NAME
        if is_supervised_root(supervised):
            return supervised.resolve()
        exact = root / TRACK1_ROOT_NAME
        if is_challenge_root(exact):
            return exact.resolve()

    candidates: list[Path] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if prefer in (None, "supervised") and child.name.casefold() == SUPERVISED_ROOT_NAME.casefold() and is_supervised_root(child):
            candidates.append(child)
        if prefer in (None, "challenge") and child.name.casefold() == TRACK1_ROOT_NAME.casefold() and is_challenge_root(child):
            candidates.append(child)
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        kind = f" {prefer}" if prefer else " supported"
        raise FileNotFoundError(f"Could not find a{kind} FloodNet dataset root below: {root}")
    raise RuntimeError(f"Ambiguous FloodNet roots below {root}: {candidates}")


def resolve_supervised_root(data_root: str | Path) -> Path:
    """Resolve the full `FloodNet-Supervised_v1.0` root or its parent."""

    return _resolve_root(data_root, prefer="supervised")


def resolve_challenge_root(data_root: str | Path) -> Path:
    """Resolve the EARTHVISION 2021 Track 1 challenge root or its parent."""

    return _resolve_root(data_root, prefer="challenge")


def resolve_track1_root(data_root: str | Path) -> Path:
    """Resolve known FloodNet roots used by this project.

    Supports both the older EARTHVISION Track 1 challenge layout and the full
    ``FloodNet-Supervised_v1.0`` layout with official train/val/test masks.
    """

    return _resolve_root(data_root, prefer=None)


def _files_by_stem(
    directory: Path,
    suffixes: Iterable[str],
    *,
    required_stem_suffix: str = "",
) -> dict[str, Path]:
    allowed = {suffix.casefold() for suffix in suffixes}
    files: dict[str, Path] = {}
    if not directory.is_dir():
        raise FileNotFoundError(f"Required directory does not exist: {directory}")
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.casefold() not in allowed:
            continue
        stem = path.stem
        if required_stem_suffix:
            if not stem.casefold().endswith(required_stem_suffix.casefold()):
                raise ValueError(f"Expected filename stem suffix {required_stem_suffix!r}: {path}")
            stem = stem[: -len(required_stem_suffix)]
        key = stem.casefold()
        if key in files:
            raise ValueError(f"Duplicate case-insensitive sample ID '{path.stem}' in {directory}")
        files[key] = path
    return files


def iter_labeled_samples(track1_root: str | Path) -> Iterator[FloodNetSample]:
    root = resolve_challenge_root(track1_root)
    seen_ids: set[str] = set()
    for scene_label, relative_dir in SCENE_DIRECTORIES.items():
        scene_root = root / Path(relative_dir)
        images = _files_by_stem(scene_root / "image", {".jpg", ".jpeg", ".JPG"})
        masks = _files_by_stem(scene_root / "mask", {".png", ".PNG"}, required_stem_suffix="_lab")
        missing_masks = sorted(set(images) - set(masks))
        missing_images = sorted(set(masks) - set(images))
        if missing_masks or missing_images:
            raise ValueError(
                f"Image/mask mismatch in {scene_root}: missing_masks={missing_masks[:10]}, missing_images={missing_images[:10]}"
            )
        for key in sorted(images):
            if key in seen_ids:
                raise ValueError(f"Duplicate labeled sample ID across scenes: {key}")
            seen_ids.add(key)
            yield FloodNetSample(
                sample_id=images[key].stem,
                official_split="Train/Labeled",
                scene_label=scene_label,
                image_path=images[key],
                mask_path=masks[key],
            )


def iter_unlabeled_samples(track1_root: str | Path) -> Iterator[FloodNetSample]:
    root = resolve_challenge_root(track1_root)
    split_directories = {
        "Train/Unlabeled": root / "Train" / "Unlabeled" / "image",
        "Validation": root / "Validation" / "image",
        "Test": root / "Test" / "image",
    }
    seen_ids: set[str] = set()
    for official_split, directory in split_directories.items():
        images = _files_by_stem(directory, {".jpg", ".jpeg", ".JPG"})
        for key in sorted(images):
            if key in seen_ids:
                raise ValueError(f"Duplicate unlabeled sample ID across splits: {key}")
            seen_ids.add(key)
            yield FloodNetSample(
                sample_id=images[key].stem,
                official_split=official_split,
                scene_label="",
                image_path=images[key],
                mask_path=None,
            )


def relative_posix(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    return path.resolve().relative_to(root.resolve()).as_posix()


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    required = {"sample_id", "image_path"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Manifest {manifest_path} is missing columns: {sorted(missing)}")
    return rows


def write_csv(path: str | Path, rows: Iterable[Mapping[str, object]], fieldnames: Iterable[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
