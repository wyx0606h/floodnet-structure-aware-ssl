from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from floodnet_ssl.config import ConfigError, load_yaml_config, validate_supervised_config
from floodnet_ssl.losses import segmentation_loss, supervised_objective
from floodnet_ssl.models import (
    ConditionalStateFactorizationWrapper,
    SegmentationModelOutput,
)
from floodnet_ssl.state_factorization import (
    compose_hierarchical_probabilities,
    fuse_semantic_and_hierarchical_logits,
    semantic_to_object_target,
    semantic_to_state_target,
    state_factorization_loss,
    state_factorization_terms,
)


class _FakeEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stages = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(3, 4, 3, stride=2, padding=1),
                torch.nn.Conv2d(4, 8, 3, stride=2, padding=1),
                torch.nn.Conv2d(8, 16, 3, stride=2, padding=1),
                torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
            ]
        )

    def forward(self, pixel_values, output_hidden_states=True, return_dict=True):
        hidden_states = []
        features = pixel_values
        for stage in self.stages:
            features = stage(features)
            hidden_states.append(features)
        return SimpleNamespace(hidden_states=tuple(hidden_states))


class _FakeDecodeHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = torch.nn.Conv2d(4, 10, 1)

    def forward(self, hidden_states):
        return self.classifier(hidden_states[0])


class _FakeSegformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_sizes=[4, 8, 16, 32])
        self.segformer = _FakeEncoder()
        self.decode_head = _FakeDecodeHead()


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

    def test_shared_and_conditional_probabilities_sum_to_one(self) -> None:
        object_logits = torch.randn(2, 8, 5, 7)
        for channels in (2, 4):
            state_logits = torch.randn(2, channels, 5, 7)
            composed = compose_hierarchical_probabilities(object_logits, state_logits)
            self.assertEqual((2, 10, 5, 7), tuple(composed.shape))
            self.assertTrue(
                torch.allclose(
                    composed.sum(dim=1), torch.ones(2, 5, 7), atol=1e-6
                )
            )

    def test_conditional_experts_can_disagree_by_object(self) -> None:
        object_logits = torch.zeros(1, 8, 1, 1)
        state_logits = torch.tensor([[[[0.0]], [[8.0]], [[8.0]], [[0.0]]]])
        composed = compose_hierarchical_probabilities(object_logits, state_logits)
        self.assertGreater(float(composed[:, 1]), float(composed[:, 2]))
        self.assertGreater(float(composed[:, 4]), float(composed[:, 3]))

    def test_zero_fusion_preserves_direct_semantic_logits(self) -> None:
        semantic = torch.randn(1, 10, 3, 3)
        hierarchical = torch.softmax(torch.randn(1, 10, 3, 3), dim=1)
        fused = fuse_semantic_and_hierarchical_logits(
            semantic, hierarchical, fusion_weight=0.0
        )
        self.assertTrue(torch.equal(semantic, fused))

    def test_feature_wrapper_returns_conditional_factorization(self) -> None:
        model = ConditionalStateFactorizationWrapper(
            _FakeSegformer(),
            hidden_sizes=[4, 8, 16, 32],
            decoder_channels=8,
            dropout=0.0,
            state_mode="conditional",
            fusion_weight=0.25,
        )
        output = model(torch.randn(2, 3, 32, 32))
        self.assertEqual((2, 10, 16, 16), tuple(output.logits.shape))
        self.assertEqual((2, 8, 16, 16), tuple(output.auxiliary["object"].shape))
        self.assertEqual((2, 4, 16, 16), tuple(output.auxiliary["state"].shape))
        self.assertEqual(
            (2, 10, 16, 16), tuple(output.auxiliary["hierarchical"].shape)
        )

    def test_state_factorization_loss_is_differentiable(self) -> None:
        semantic = torch.randn(2, 10, 4, 4, requires_grad=True)
        output = SegmentationModelOutput(
            logits=semantic,
            auxiliary={
                "semantic_direct": semantic,
                "object": torch.randn(2, 8, 4, 4, requires_grad=True),
                "state": torch.randn(2, 4, 4, 4, requires_grad=True),
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

    def test_crop_without_building_or_road_has_finite_zero_state_loss(self) -> None:
        semantic = torch.randn(1, 10, 4, 4)
        target = torch.full((1, 4, 4), 9, dtype=torch.long)
        terms = state_factorization_terms(
            semantic_logits=semantic,
            auxiliary={
                "object": torch.randn(1, 8, 4, 4),
                "state": torch.randn(1, 4, 4, 4),
            },
            target=target,
            config={"enabled": True},
        )
        self.assertTrue(torch.isfinite(terms["total"]))
        self.assertEqual(0.0, float(terms["state"]))

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

    def test_disabled_module_preserves_baseline_objective(self) -> None:
        semantic = torch.randn(1, 10, 4, 4)
        target = torch.randint(0, 10, (1, 4, 4), dtype=torch.long)
        baseline = segmentation_loss(semantic, target, {"name": "ce_dice"})
        objective = supervised_objective(
            semantic,
            target,
            {"loss": {"name": "ce_dice"}},
        )
        self.assertTrue(torch.equal(baseline, objective))

    def test_state_config_loads_conditional_feature_model(self) -> None:
        config = load_yaml_config(
            "configs/segformer_b0_sup398_state_factorization.yaml"
        )
        self.assertTrue(config["modules"]["state_factorization"]["enabled"])
        self.assertEqual(["object", "state"], config["model"]["auxiliary_heads"])
        factorization = config["model"]["state_factorization"]
        self.assertEqual("encoder_multiscale", factorization["feature_source"])
        self.assertEqual("conditional", factorization["state_mode"])
        self.assertEqual(0.25, factorization["fusion_weight"])

    def test_logit_head_ablation_is_configurable_but_requires_shared_state(self) -> None:
        config = load_yaml_config(
            "configs/segformer_b0_sup398_state_factorization.yaml"
        )
        factorization = config["model"]["state_factorization"]
        factorization["feature_source"] = "logits"
        factorization["state_mode"] = "shared"
        factorization["fusion_weight"] = 0.0
        validate_supervised_config(config)
        factorization["state_mode"] = "conditional"
        with self.assertRaises(ConfigError):
            validate_supervised_config(config)


if __name__ == "__main__":
    unittest.main()
