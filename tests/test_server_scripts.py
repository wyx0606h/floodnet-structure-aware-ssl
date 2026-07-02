from __future__ import annotations

import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from scripts.evaluate_checkpoint import _mean, _prepare_output_dir
from scripts.summarize_run import summarize_run


class ServerScriptUtilitiesTest(unittest.TestCase):
    def test_mean_ignores_nan_values(self) -> None:
        self.assertAlmostEqual(2.0, _mean([1.0, float("nan"), 3.0]))

    def test_evaluation_output_dir_refuses_non_empty_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "eval"
            output_dir.mkdir()
            (output_dir / "existing.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _prepare_output_dir(output_dir, overwrite=False)
            self.assertEqual(output_dir, _prepare_output_dir(output_dir, overwrite=True))

    def test_summarize_run_collects_history_checkpoints_and_evaluations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            eval_dir = root / "eval_validation"
            run_dir.mkdir()
            eval_dir.mkdir()
            (run_dir / "best.pt").write_bytes(b"checkpoint")
            (run_dir / "last.pt").write_bytes(b"checkpoint")
            (run_dir / "resolved_config.json").write_text(
                json.dumps(
                    {
                        "experiment": {"run_id": "run_a", "kind": "supervised_baseline"},
                        "model": {"name": "segformer_b0"},
                        "data": {"crop_size": 512},
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "train_summary.json").write_text(
                json.dumps({"best_epoch": 2, "best_validation_miou10": 0.4}),
                encoding="utf-8",
            )
            with (run_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["epoch", "validation_miou10", "train_loss"]
                )
                writer.writeheader()
                writer.writerow({"epoch": 1, "validation_miou10": 0.3, "train_loss": 1.0})
                writer.writerow({"epoch": 2, "validation_miou10": 0.4, "train_loss": 0.8})
            (eval_dir / "metrics.json").write_text(
                json.dumps({"split": "validation", "metrics": {"miou10": 0.5}}),
                encoding="utf-8",
            )

            summary = summarize_run(run_dir, [eval_dir])
            self.assertEqual("run_a", summary["run_id"])
            self.assertEqual("2", summary["best_history_row"]["epoch"])
            self.assertTrue(summary["checkpoints"]["best"].endswith("best.pt"))
            self.assertEqual("validation", summary["evaluations"][0]["split"])


if __name__ == "__main__":
    unittest.main()
