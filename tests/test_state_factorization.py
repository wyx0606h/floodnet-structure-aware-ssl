from __future__ import annotations

import unittest

import torch

from floodnet_ssl.config import load_yaml_config
from floodnet_ssl.losses import supervised_objective
from floodnet_ssl.models import SegmentationModelOutput
from floodnet_ssl.state_factorization import (
    compose_hierarchical_probabilities,
    semantic_to_object_target,
    semantic_to_state_target,
    state_factorization_loss,
)


class StateFactorizationTest(unittest.TestCase):
    def test_semantic_targets_factorize_buildings_and_roads(self) -> None:
        target = torch.tensor([[[0, 1, 2, 3, 4, 5, 9, 255]]], dtype=torch.long)
        self.assertEqual(
            [[[0, 1, 1, 2, 2, 3, 7, 255]]],
            semantic_to_object_target(target).tolist(),
        )
        self.assertEqual(
            [[[255, 1, 0, 1, 0, 255, 255, 255]]],
            semantic_to_state_target(target).tolist(),
        )

    def test_hierarchical_probabilities_sum_to_one(self) -> None:
        object_logits = torch.randn(2, 8, 5, 7)
        state_logits = torch.randn(2, 2, 5, 7)
        composed = compose_hierarchical_probabilities(object_logits, state_logits)
        self.assertEqual((2, 10, 5, 7), tuple(composed.shape))
        self.assertTrue(torch.allclose(composed.sum(dim=1), torch.ones(2, 5, 7), atol=1e-6))

    def test_state_factorization_loss_is_differentiable(self) -> None:
        semantic = torch.randn(2, 10, 4, 4, requires_grad=True)
        output = SegmentationModelOutput(
            logits=semantic,
            auxiliary={
                "object": torch.randn(2, 8, 4, 4, requires_grad=True),
                "state": torch.randn(2, 2, 4, 4, requires_grad=True),
            },
        )
        target = torch.randint(0, 10, (2, 4, 4), dtype=torch.long)
        loss = supervised_objective(
            output,
            target,
            {
                "loss": {"name": "ce_dice"},
                "modules": {"state_factorization": {"enabled": True}},
            },
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(semantic.grad)

    def test_disabled_state_factorization_returns_zero(self) -> None:
        semantic = torch.randn(1, 10, 2, 2)
        target = torch.zeros((1, 2, 2), dtype=torch.long)
        loss = state_factorization_loss(
            semantic_logits=semantic,
            auxiliary={},
            target=target,
            config={"enabled": False},
        )
        self.assertAlmostEqual(0.0, float(loss))

    def test_state_config_loads(self) -> None:
        config = load_yaml_config("configs/segformer_b0_sup398_state_factorization.yaml")
        self.assertTrue(config["modules"]["state_factorization"]["enabled"])
        self.assertEqual(["object", "state"], config["model"]["auxiliary_heads"])


if __name__ == "__main__":
    unittest.main()
