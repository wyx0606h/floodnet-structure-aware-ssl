from __future__ import annotations

import unittest

import torch

from floodnet_ssl.inference import sliding_positions, sliding_window_predict


class PointwiseModel(torch.nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        value = image[:, :1]
        return torch.cat((value, -value), dim=1)


class HalfResolutionPointwiseModel(torch.nn.Module):
    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        value = torch.nn.functional.avg_pool2d(image[:, :1], kernel_size=2)
        return {"logits": torch.cat((value, -value), dim=1)}


class SlidingWindowInferenceTest(unittest.TestCase):
    def test_positions_cover_last_pixel_without_gaps(self) -> None:
        positions = sliding_positions(length=13, tile=6, stride=4)
        self.assertEqual([0, 4, 7], positions)
        covered = [False] * 13
        for start in positions:
            for index in range(start, start + 6):
                covered[index] = True
        self.assertTrue(all(covered))

    def test_probability_fusion_matches_pointwise_full_image_result(self) -> None:
        image = torch.linspace(-2, 2, 3 * 11 * 13).reshape(3, 11, 13)
        model = PointwiseModel()
        expected = torch.softmax(model(image.unsqueeze(0)), dim=1)
        actual = sliding_window_predict(
            model,
            image,
            tile_size=(6, 7),
            stride=(4, 5),
            tile_batch_size=3,
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            actual.sum(dim=1), torch.ones((1, 11, 13)), atol=1e-6, rtol=0
        )

    def test_small_image_is_padded_and_cropped_back(self) -> None:
        image = torch.rand(3, 4, 5)
        probabilities = sliding_window_predict(
            PointwiseModel(), image, tile_size=8, stride=6
        )
        self.assertEqual((1, 2, 4, 5), tuple(probabilities.shape))

    def test_low_resolution_logits_are_resized(self) -> None:
        image = torch.rand(3, 10, 12)
        probabilities = sliding_window_predict(
            HalfResolutionPointwiseModel(),
            image,
            tile_size=(6, 8),
            stride=(4, 6),
        )
        self.assertEqual((1, 2, 10, 12), tuple(probabilities.shape))
        self.assertTrue(torch.isfinite(probabilities).all())


if __name__ == "__main__":
    unittest.main()
