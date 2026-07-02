from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from floodnet_ssl.protocols import build_floodnet_splits
from tests.helpers import create_synthetic_supervised, create_synthetic_track1


class ProtocolSplitTest(unittest.TestCase):
    def test_builds_sup398_and_full1445_manifests_from_challenge_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            challenge = create_synthetic_track1(root, flooded=2, non_flooded=3, unlabeled=4, validation=1, test=1)
            supervised = create_synthetic_supervised(root, train=9, validation=2, test=3, start_id=100)
            output = root / "splits"
            summary = build_floodnet_splits(
                supervised_root=supervised,
                challenge_root=challenge,
                output_dir=output,
                expected_counts={
                    "challenge_labeled": 5,
                    "challenge_unlabeled": 4,
                    "full_train": 9,
                    "validation": 2,
                    "test": 3,
                },
            )
            self.assertEqual(5, summary["counts"]["challenge_labeled"])
            self.assertTrue((output / "challenge_labeled_398.txt").is_file())
            with (output / "sup398_manifest.csv").open(encoding="utf-8", newline="") as handle:
                sup_rows = list(csv.DictReader(handle))
            with (output / "full1445_manifest.csv").open(encoding="utf-8", newline="") as handle:
                full_rows = list(csv.DictReader(handle))
            self.assertEqual(5, sum(row["split"] == "train" for row in sup_rows))
            self.assertEqual(9, sum(row["split"] == "train" for row in full_rows))
            self.assertEqual(2, sum(row["split"] == "validation" for row in sup_rows))
            self.assertEqual(3, sum(row["split"] == "test" for row in full_rows))

    def test_rejects_if_challenge_union_does_not_equal_full_train(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            challenge = create_synthetic_track1(root, flooded=1, non_flooded=0, unlabeled=1, validation=0, test=0)
            supervised = create_synthetic_supervised(root, train=3, validation=1, test=1, start_id=100)
            with self.assertRaises(ValueError):
                build_floodnet_splits(
                    supervised_root=supervised,
                    challenge_root=challenge,
                    output_dir=root / "splits",
                    expected_counts={
                        "challenge_labeled": 1,
                        "challenge_unlabeled": 1,
                        "full_train": 3,
                        "validation": 1,
                        "test": 1,
                    },
                )


if __name__ == "__main__":
    unittest.main()
