from __future__ import annotations

import unittest

import torch

from floodnet_ssl.losses import multiclass_dice_loss, segmentation_loss


class LossesTest(unittest.TestCase):
    def test_ce_dice_loss_is_finite_and_differentiable(self) -> None:
        logits = torch.randn(2, 10, 8, 8, requires_grad=True)
        target = torch.randint(0, 10, (2, 8, 8), dtype=torch.long)
        loss = segmentation_loss(logits, target, {"name": "ce_dice"})
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_dice_loss_ignores_ignore_index(self) -> None:
        logits = torch.randn(1, 10, 4, 4, requires_grad=True)
        target = torch.full((1, 4, 4), 255, dtype=torch.long)
        loss = multiclass_dice_loss(logits, target)
        self.assertAlmostEqual(0.0, float(loss.detach()))


if __name__ == "__main__":
    unittest.main()
