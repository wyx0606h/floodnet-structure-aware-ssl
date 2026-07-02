from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from floodnet_ssl.models import (
    MultiHeadSegmentationModel,
    MissingSegFormerDependency,
    SegmentationModelOutput,
    build_model,
    extract_logits,
    require_segformer_dependencies,
    segformer_dependency_status,
)
from floodnet_ssl.training import (
    build_optimizer,
    evaluate_overfit_gate,
    run_supervised_epoch,
    save_checkpoint,
)


class TinySegmenter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = torch.nn.Conv2d(3, 10, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network(image)


class TinyOutputSegmenter(TinySegmenter):
    def forward(self, image: torch.Tensor) -> SegmentationModelOutput:
        return SegmentationModelOutput(logits=self.network(image))


class TinyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = torch.nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.features(image)


def tiny_batches() -> list[dict[str, torch.Tensor]]:
    image = torch.zeros((2, 3, 8, 8), dtype=torch.float32)
    image[0, 0] = 1.0
    image[1, 1] = 1.0
    mask = torch.zeros((2, 8, 8), dtype=torch.long)
    mask[0] = 1
    mask[1] = 3
    return [{"image": image, "mask": mask}]


class TrainingEngineTest(unittest.TestCase):
    def test_train_and_evaluate_epoch(self) -> None:
        torch.manual_seed(0)
        model = TinySegmenter()
        optimizer = build_optimizer(
            model,
            {
                "optimizer": "adam",
                "learning_rate": 0.05,
                "weight_decay": 0.0,
            },
        )
        first = run_supervised_epoch(
            model, tiny_batches(), device="cpu", optimizer=optimizer
        )
        for _ in range(20):
            last = run_supervised_epoch(
                model, tiny_batches(), device="cpu", optimizer=optimizer
            )
        evaluation = run_supervised_epoch(model, tiny_batches(), device="cpu")
        self.assertLess(last.loss, first.loss)
        self.assertGreater(evaluation.miou10, 0.9)
        self.assertEqual(1, last.optimizer_steps)

    def test_train_epoch_accepts_unified_model_output(self) -> None:
        torch.manual_seed(0)
        model = TinyOutputSegmenter()
        optimizer = build_optimizer(
            model,
            {
                "optimizer": "adam",
                "learning_rate": 0.05,
                "weight_decay": 0.0,
            },
        )
        result = run_supervised_epoch(
            model, tiny_batches(), device="cpu", optimizer=optimizer
        )
        self.assertTrue(torch.isfinite(torch.tensor(result.loss)))

    def test_checkpoint_contains_reproducible_state(self) -> None:
        model = TinySegmenter()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                epoch=2,
                history=[{"epoch": 1}],
                config={"experiment": {"run_id": "test"}},
            )
            checkpoint = torch.load(path, map_location="cpu")
            self.assertEqual(2, checkpoint["epoch"])
            self.assertIn("model_state_dict", checkpoint)
            self.assertIn("optimizer_state_dict", checkpoint)

    def test_overfit_gate_is_objective(self) -> None:
        gate = evaluate_overfit_gate(
            [
                {"train_loss": 2.0, "train_miou10": 0.1},
                {"train_loss": 0.3, "train_miou10": 0.95},
            ],
            maximum_final_to_initial_loss_ratio=0.2,
            minimum_train_miou10=0.9,
        )
        self.assertTrue(gate["passed"])

    def test_missing_segformer_dependency_has_actionable_error(self) -> None:
        if all(segformer_dependency_status().values()):
            self.skipTest("SegFormer dependencies are installed")
        with self.assertRaises(MissingSegFormerDependency):
            require_segformer_dependencies(pretrained=True)

    def test_extract_logits_accepts_unified_output(self) -> None:
        logits = torch.randn(2, 10, 8, 8)
        output = SegmentationModelOutput(
            logits=logits, auxiliary={"boundary": torch.randn(2, 1, 8, 8)}
        )
        self.assertIs(extract_logits(output), logits)

    def test_build_model_dispatches_current_segformer_config(self) -> None:
        sentinel = torch.nn.Identity()
        with patch("floodnet_ssl.models.build_segformer_b0", return_value=sentinel):
            model = build_model({"name": "segformer_b0", "num_labels": 10})
        self.assertIs(model, sentinel)

    def test_multi_head_skeleton_forward_without_optional_dependencies(self) -> None:
        model = MultiHeadSegmentationModel(
            TinyBackbone(),
            feature_channels=4,
            num_labels=10,
            enabled_auxiliary_heads=("object", "state", "boundary", "relation"),
        )
        output = model(torch.zeros((2, 3, 16, 16), dtype=torch.float32))
        self.assertEqual((2, 10, 16, 16), tuple(output.logits.shape))
        self.assertEqual((2, 8, 16, 16), tuple(output.auxiliary["object"].shape))
        self.assertEqual((2, 2, 16, 16), tuple(output.auxiliary["state"].shape))
        self.assertEqual((2, 1, 16, 16), tuple(output.auxiliary["boundary"].shape))
        self.assertEqual((2, 2, 16, 16), tuple(output.auxiliary["relation"].shape))

    def test_multi_head_skeleton_defaults_to_no_auxiliary_heads(self) -> None:
        model = MultiHeadSegmentationModel(TinyBackbone(), feature_channels=4)
        output = model(torch.zeros((1, 3, 8, 8), dtype=torch.float32))
        self.assertEqual((1, 10, 8, 8), tuple(output.logits.shape))
        self.assertEqual({}, dict(output.auxiliary))


    def test_poly_warmup_scheduler_values(self) -> None:
        from train import _scheduled_lr

        cfg = {
            "learning_rate": 0.00006,
            "scheduler": "poly",
            "warmup_iterations": 10,
            "poly_power": 1.0,
        }
        self.assertAlmostEqual(0.000006, _scheduled_lr(cfg, 1, 100))
        self.assertAlmostEqual(0.00006, _scheduled_lr(cfg, 10, 100))
        self.assertLess(_scheduled_lr(cfg, 50, 100), 0.00006)
        self.assertAlmostEqual(0.0, _scheduled_lr(cfg, 100, 100))


if __name__ == "__main__":
    unittest.main()
