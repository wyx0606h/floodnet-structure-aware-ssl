"""Safe inspection and merge extraction for the seven Track 1 ZIP packages."""

from __future__ import annotations

import binascii
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable
from zipfile import ZipFile, ZipInfo

from .constants import TRACK1_ROOT_NAME

TRACK1_ZIP_RE = re.compile(
    r"^(?P<prefix>FloodNet Challenge @ EARTHVISION 2021 - Track 1-.*-)"
    r"(?P<part>00[1-7])\.zip$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ArchiveMember:
    archive: str
    member: str
    size: int
    compressed_size: int
    crc32: str


@dataclass(frozen=True)
class MergePlan:
    source_dir: str
    destination_dir: str
    archives: tuple[str, ...]
    file_count: int
    expanded_bytes: int
    members: tuple[ArchiveMember, ...]


def discover_track1_archives(source_dir: str | Path) -> list[Path]:
    source = Path(source_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"ZIP source directory does not exist: {source}")

    matched: dict[int, tuple[str, Path]] = {}
    prefixes: set[str] = set()
    for path in source.iterdir():
        if not path.is_file():
            continue
        match = TRACK1_ZIP_RE.match(path.name)
        if not match:
            continue
        part = int(match.group("part"))
        if part in matched:
            raise ValueError(f"Multiple Track 1 archives found for part {part:03d}")
        prefix = match.group("prefix").casefold()
        prefixes.add(prefix)
        matched[part] = (prefix, path.resolve())

    expected_parts = set(range(1, 8))
    if set(matched) != expected_parts:
        missing = sorted(expected_parts - set(matched))
        extra = sorted(set(matched) - expected_parts)
        raise ValueError(
            f"Expected exactly Track 1 parts 001-007; missing={missing}, extra={extra}"
        )
    if len(prefixes) != 1:
        raise ValueError("Track 1 ZIP parts do not belong to the same download batch")
    return [matched[index][1] for index in range(1, 8)]


def _safe_member_path(filename: str) -> PurePosixPath:
    normalized = filename.replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute():
        raise ValueError(f"Unsafe absolute or empty ZIP member path: {filename!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe ZIP member path: {filename!r}")
    if ":" in path.parts[0]:
        raise ValueError(f"Unsafe drive-qualified ZIP member path: {filename!r}")
    return path


def _is_symlink(info: ZipInfo) -> bool:
    mode = info.external_attr >> 16
    return stat.S_ISLNK(mode)


def build_merge_plan(
    source_dir: str | Path,
    destination_dir: str | Path,
    *,
    max_expanded_gib: float = 64.0,
    max_compression_ratio: float = 1000.0,
) -> MergePlan:
    source = Path(source_dir).expanduser().resolve()
    destination = Path(destination_dir).expanduser().resolve()
    if destination == source:
        raise ValueError("Destination must not be the ZIP source directory itself")
    try:
        source.relative_to(destination)
    except ValueError:
        pass
    else:
        raise ValueError("Destination must not be an ancestor of the ZIP source directory")
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"Destination exists but is not a directory: {destination}")
    archives = discover_track1_archives(source)

    members: list[ArchiveMember] = []
    seen_paths: dict[str, str] = {}
    expanded_bytes = 0
    top_levels: set[str] = set()

    for archive in archives:
        with ZipFile(archive) as handle:
            for info in handle.infolist():
                if info.is_dir():
                    continue
                if info.flag_bits & 0x1:
                    raise ValueError(f"Encrypted ZIP member is not allowed: {info.filename}")
                if _is_symlink(info):
                    raise ValueError(f"Symbolic link ZIP member is not allowed: {info.filename}")
                safe_path = _safe_member_path(info.filename)
                if info.compress_size == 0:
                    compression_ratio = float("inf") if info.file_size else 1.0
                else:
                    compression_ratio = info.file_size / info.compress_size
                if compression_ratio > max_compression_ratio:
                    raise ValueError(
                        f"Suspicious compression ratio {compression_ratio:.1f} for "
                        f"{info.filename}"
                    )
                top_levels.add(safe_path.parts[0].casefold())
                key = safe_path.as_posix().casefold()
                if key in seen_paths:
                    raise ValueError(
                        f"Duplicate member path across ZIPs: {info.filename} "
                        f"({seen_paths[key]} and {archive.name})"
                    )
                seen_paths[key] = archive.name
                expanded_bytes += info.file_size
                members.append(
                    ArchiveMember(
                        archive=archive.name,
                        member=safe_path.as_posix(),
                        size=info.file_size,
                        compressed_size=info.compress_size,
                        crc32=f"{info.CRC:08x}",
                    )
                )

    if top_levels != {TRACK1_ROOT_NAME.casefold()}:
        raise ValueError(
            f"Unexpected top-level directories in Track 1 archives: {sorted(top_levels)}"
        )
    max_bytes = int(max_expanded_gib * 1024**3)
    if expanded_bytes > max_bytes:
        raise ValueError(
            f"Expanded data size {expanded_bytes / 1024**3:.2f} GiB exceeds "
            f"safety limit {max_expanded_gib:.2f} GiB"
        )
    return MergePlan(
        source_dir=str(source),
        destination_dir=str(destination),
        archives=tuple(str(path) for path in archives),
        file_count=len(members),
        expanded_bytes=expanded_bytes,
        members=tuple(members),
    )


def _crc32_file(path: Path, chunk_size: int = 1024 * 1024) -> int:
    checksum = 0
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            checksum = binascii.crc32(chunk, checksum)
    return checksum & 0xFFFFFFFF


def _publish_without_overwrite(temp_path: Path, target_path: Path) -> None:
    try:
        os.link(temp_path, target_path)
    except FileExistsError:
        raise
    except OSError:
        try:
            with temp_path.open("rb") as source, target_path.open("xb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
        except Exception:
            target_path.unlink(missing_ok=True)
            raise
    finally:
        temp_path.unlink(missing_ok=True)


def _extract_member(handle: ZipFile, info: ZipInfo, destination: Path) -> str:
    relative = Path(*_safe_member_path(info.filename).parts)
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        if not target.is_file():
            raise ValueError(f"Existing extraction target is not a file: {target}")
        if target.stat().st_size == info.file_size and _crc32_file(target) == info.CRC:
            return "skipped"
        raise FileExistsError(
            f"Refusing to overwrite mismatched existing file: {target}"
        )

    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".partial", dir=target.parent
    )
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        with handle.open(info, "r") as source, temp_path.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        if temp_path.stat().st_size != info.file_size:
            raise IOError(f"Size mismatch while extracting {info.filename}")
        if _crc32_file(temp_path) != info.CRC:
            raise IOError(f"CRC mismatch while extracting {info.filename}")
        _publish_without_overwrite(temp_path, target)
        return "extracted"
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _set_read_only(path: Path) -> None:
    current_mode = path.stat().st_mode
    path.chmod(current_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)


def _validate_existing_destination(plan: MergePlan, destination: Path) -> None:
    if not destination.exists():
        return
    allowed = {member.member.casefold() for member in plan.members}
    allowed.add("floodnet_merge_manifest.json")
    for path in destination.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Destination contains a symbolic link: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(destination).as_posix().casefold()
        if relative not in allowed:
            raise FileExistsError(
                f"Destination contains an unrelated file; refusing to merge: {path}"
            )


def execute_merge_plan(
    plan: MergePlan,
    *,
    confirmed_destination: str | Path,
    set_read_only: bool = True,
) -> dict[str, object]:
    destination = Path(plan.destination_dir).resolve()
    confirmed = Path(confirmed_destination).expanduser().resolve()
    if destination != confirmed:
        raise ValueError(
            "Destination confirmation mismatch: "
            f"planned={destination}, confirmed={confirmed}"
        )

    _validate_existing_destination(plan, destination)
    destination.mkdir(parents=True, exist_ok=True)
    extracted = 0
    skipped = 0
    for archive_name in plan.archives:
        archive = Path(archive_name)
        with ZipFile(archive) as handle:
            for info in handle.infolist():
                if info.is_dir():
                    continue
                status = _extract_member(handle, info, destination)
                if status == "extracted":
                    extracted += 1
                else:
                    skipped += 1
                if set_read_only:
                    relative = Path(*_safe_member_path(info.filename).parts)
                    _set_read_only(destination / relative)

    persistent_manifest = {
        "source_dir": plan.source_dir,
        "destination_dir": plan.destination_dir,
        "archives": list(plan.archives),
        "file_count": plan.file_count,
        "expanded_bytes": plan.expanded_bytes,
        "plan_sha256": merge_plan_sha256(plan),
        "files_set_read_only": set_read_only,
    }
    manifest_path = destination / "floodnet_merge_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != persistent_manifest:
            raise FileExistsError(
                f"Refusing to overwrite a different merge manifest: {manifest_path}"
            )
    else:
        manifest_path.write_text(
            json.dumps(persistent_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if set_read_only:
            _set_read_only(manifest_path)
    return {
        **persistent_manifest,
        "extracted_this_run": extracted,
        "skipped_existing_this_run": skipped,
    }


def merge_plan_summary(plan: MergePlan) -> dict[str, object]:
    archive_names = [Path(path).name for path in plan.archives]
    return {
        "source_dir": plan.source_dir,
        "destination_dir": plan.destination_dir,
        "archives": archive_names,
        "file_count": plan.file_count,
        "expanded_gib": round(plan.expanded_bytes / 1024**3, 3),
    }


def merge_plan_sha256(plan: MergePlan) -> str:
    digest = hashlib.sha256()
    for member in plan.members:
        digest.update(
            (
                f"{member.archive}\0{member.member}\0{member.size}\0"
                f"{member.compressed_size}\0{member.crc32}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()
