from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from floodnet_ssl.audit import audit_dataset
from tests.helpers import create_synthetic_track1


class AuditTest(unittest.TestCase):
    def test_audit_writes_inventory_and_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_synthetic_track1(root)
            output = root / "audit"
            summary = audit_dataset(
                root,
                output,
                expected_counts={
                    "labeled_flooded": 2,
                    "labeled_non_flooded": 3,
                    "unlabeled": 1,
                    "validation": 1,
                    "test": 1,
                },
            )
            self.assertTrue(summary["passed"])
            self.assertTrue((output / "inventory.csv").is_file())
            self.assertTrue((output / "class_statistics.csv").is_file())
            saved = json.loads(
                (output / "audit_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(5, sum(saved["actual_counts"][key] for key in ("labeled_flooded", "labeled_non_flooded")))

    def test_refuses_to_overwrite_audit_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_synthetic_track1(root)
            output = root / "audit"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                audit_dataset(root, output, expected_counts=None)


if __name__ == "__main__":
    unittest.main()
