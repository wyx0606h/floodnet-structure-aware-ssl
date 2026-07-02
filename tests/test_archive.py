from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from floodnet_ssl.archive import (
    build_merge_plan,
    discover_track1_archives,
    execute_merge_plan,
)
from floodnet_ssl.constants import TRACK1_ROOT_NAME


class ArchiveSafetyTest(unittest.TestCase):
    def _write_archives(self, directory: Path, unsafe_member: str | None = None) -> None:
        prefix = "FloodNet Challenge @ EARTHVISION 2021 - Track 1-batch-"
        for part in range(1, 8):
            path = directory / f"{prefix}{part:03d}.zip"
            with ZipFile(path, "w") as archive:
                member = (
                    unsafe_member
                    if part == 1 and unsafe_member is not None
                    else f"{TRACK1_ROOT_NAME}/part_{part:03d}.txt"
                )
                archive.writestr(member, f"part {part}")
        with ZipFile(
            directory / "FloodNet Challenge @ EARTHVISION 2021 - Track 2-batch-001.zip",
            "w",
        ) as archive:
            archive.writestr("track2/file.txt", "not track 1")

    def test_discovers_only_complete_track1_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_archives(root)
            archives = discover_track1_archives(root)
            self.assertEqual(7, len(archives))
            self.assertTrue(all("Track 1" in path.name for path in archives))

    def test_plan_is_non_mutating_and_execute_requires_matching_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_archives(root)
            destination = root / "merged"
            plan = build_merge_plan(root, destination, max_expanded_gib=1)
            self.assertFalse(destination.exists())
            with self.assertRaises(ValueError):
                execute_merge_plan(
                    plan,
                    confirmed_destination=root / "wrong",
                    set_read_only=False,
                )
            result = execute_merge_plan(
                plan,
                confirmed_destination=destination,
                set_read_only=False,
            )
            self.assertEqual(7, result["extracted_this_run"])
            self.assertTrue((destination / TRACK1_ROOT_NAME / "part_001.txt").is_file())

    def test_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_archives(root, unsafe_member="../escape.txt")
            with self.assertRaises(ValueError):
                build_merge_plan(root, root / "merged", max_expanded_gib=1)


if __name__ == "__main__":
    unittest.main()
