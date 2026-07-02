from __future__ import annotations

import unittest

import torch

from floodnet_ssl.boundary_context import (
    boundary_context_loss,
    refine_logits_with_boundary_context,
    semantic_boundary_target,
    soft_semantic_boundary,
)
from floodnet_ssl.config import load_yaml_config
from floodnet_ssl.losses import supervised_objective
from floodnet_ssl.models import SegmentationModelOutput


class BoundaryContextTest(unittest.TestCase):
    def test_semantic_boundary_target_detects_class_edges(self) -> None:
        target = torch.zeros((1, 6, 6), dtype=torch.long)
        target[:, :, 3:] = 1
        boundary = semantic_boundary_target(target, width=1)
        self.assertGreater(float(boundary.sum()), 0.0)
        self.assertEqual(0.0, float(boundary[:, :, 0].sum()))

    def test_soft_semantic_boundary_has_expected_shape(self) -> None:
        logits = torch.randn(2, 10, 5, 7)
        edge = soft_semantic_boundary(logits)
        self.assertEqual((2, 1, 5, 7), tuple(edge.shape))
        self.assertGreaterEqual(float(edge.min()), 0.0)
        self.assertLessEqual(float(edge.max()), 1.0)

    def test_boundary_context_refinement_preserves_shape(self) -> None:
        logits = torch.randn(2, 10, 8, 8)
        boundary = torch.randn(2, 1, 8, 8)
        refined = refine_logits_with_boundary_context(logits, boundary, strength=0.5, kernel_size=3)
        self.assertEqual(tuple(logits.shape), tuple(refined.shape))
        self.assertFalse(torch.allclose(logits, refined))

    def test_boundary_context_loss_is_differentiable(self) -> None:
        semantic = torch.randn(2, 10, 6, 6, requires_grad=True)
        boundary = torch.randn(2, 1, 6, 6, requires_grad=True)
        target = torch.zeros((2, 6, 6), dtype=torch.long)
        target[:, :, 3:] = 1
        loss = boundary_context_loss(
            semantic_logits=semantic,
            auxiliary={"boundary": boundary},
            target=target,
            config={"enabled": True, "target_width": 3},
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(boundary.grad)

    def test_supervised_objective_accepts_boundary_auxiliary(self) -> None:
        semantic = torch.randn(1, 10, 4, 4, requires_grad=True)
        output = SegmentationModelOutput(
            logits=semantic,
            auxiliary={"boundary": torch.randn(1, 1, 4, 4, requires_grad=True)},
        )
        target = torch.randint(0, 10, (1, 4, 4), dtype=torch.long)
        loss = supervised_objective(
            output,
            target,
            {
                "loss": {"name": "ce_dice"},
                "modules": {"boundary_context": {"enabled": True}},
            },
        )
        self.assertTrue(torch.isfinite(loss))

    def test_boundary_config_loads(self) -> None:
        config = load_yaml_config("configs/segformer_b0_sup398_boundary_context.yaml")
        self.assertTrue(config["model"]["boundary_context"]["enabled"])
        self.assertTrue(config["modules"]["boundary_context"]["enabled"])


if __name__ == "__main__":
    unittest.main()
