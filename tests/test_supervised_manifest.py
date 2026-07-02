from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from floodnet_ssl.layout import resolve_track1_root
from scripts.create_supervised_manifest import build_rows
from tests.helpers import create_synthetic_supervised


class SupervisedManifestTest(unittest.TestCase):
    def test_resolves_supervised_root_from_parent_or_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            supervised = create_synthetic_supervised(parent)
            self.assertEqual(supervised.resolve(), resolve_track1_root(parent))
            self.assertEqual(supervised.resolve(), resolve_track1_root(supervised))

    def test_build_rows_pairs_official_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            supervised = create_synthetic_supervised(parent, train=2, validation=1, test=1)
            expected = {"train": 2, "validation": 1, "test": 1}
            with patch("scripts.create_supervised_manifest.SUPERVISED_EXPECTED_COUNTS", expected):
                rows, summary = build_rows(supervised)
            self.assertEqual(4, len(rows))
            self.assertEqual(4, summary["total_samples"])
            self.assertEqual(expected, {split: summary["splits"][split]["samples"] for split in expected})
            self.assertEqual({"train", "validation", "test"}, {row["split"] for row in rows})
            self.assertTrue(all(row["mask_path"].endswith("_lab.png") for row in rows))

    def test_build_rows_rejects_missing_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            supervised = create_synthetic_supervised(parent, train=1, validation=1, test=1)
            next((supervised / "train" / "train-label-img").glob("*_lab.png")).unlink()
            expected = {"train": 1, "validation": 1, "test": 1}
            with patch("scripts.create_supervised_manifest.SUPERVISED_EXPECTED_COUNTS", expected):
                with self.assertRaises(ValueError):
                    build_rows(supervised)


if __name__ == "__main__":
    unittest.main()
