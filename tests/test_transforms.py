from __future__ import annotations

import random
import unittest

import numpy as np
from PIL import Image

from floodnet_ssl.transforms import (
    ClassAwareRandomCrop,
    DeterministicClassCrop,
    PairedCompose,
    RandomHorizontalFlip,
    RandomScale,
    RandomVerticalFlip,
)


class PairedTransformsTest(unittest.TestCase):
    def test_flips_keep_rgb_and_mask_geometry_aligned(self) -> None:
        mask = np.zeros((6, 8), dtype=np.uint8)
        mask[1:4, 2:5] = 3
        image = np.repeat(mask[:, :, None], 3, axis=2)
        transform = PairedCompose(
            (
                RandomHorizontalFlip(probability=1.0, rng=random.Random(1)),
                RandomVerticalFlip(probability=1.0, rng=random.Random(2)),
            )
        )
        transformed_image, transformed_mask = transform(
            Image.fromarray(image), Image.fromarray(mask)
        )
        self.assertIsNotNone(transformed_mask)
        np.testing.assert_array_equal(
            np.asarray(transformed_image)[:, :, 0], np.asarray(transformed_mask)
        )

    def test_scale_uses_nearest_neighbor_for_mask(self) -> None:
        mask = np.asarray([[1, 3], [7, 9]], dtype=np.uint8)
        image = np.repeat(mask[:, :, None], 3, axis=2)
        transformed_image, transformed_mask = RandomScale(
            minimum=2.0, maximum=2.0, rng=random.Random(0)
        )(Image.fromarray(image), Image.fromarray(mask))
        self.assertEqual((4, 4), transformed_image.size)
        self.assertIsNotNone(transformed_mask)
        self.assertEqual({1, 3, 7, 9}, set(np.unique(transformed_mask).tolist()))

    def test_class_aware_crop_contains_target_class(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[18, 18] = 1
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        _, cropped_mask = ClassAwareRandomCrop(
            size=8,
            class_ids=(1, 3),
            class_aware_probability=1.0,
            rng=random.Random(4),
        )(Image.fromarray(image), Image.fromarray(mask))
        self.assertIsNotNone(cropped_mask)
        self.assertIn(1, np.unique(cropped_mask))
        self.assertEqual((8, 8), cropped_mask.size)

    def test_crop_padding_uses_ignore_index(self) -> None:
        mask = Image.fromarray(np.ones((3, 4), dtype=np.uint8))
        image = Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8))
        _, cropped_mask = ClassAwareRandomCrop(
            size=8,
            class_ids=(1,),
            class_aware_probability=0.0,
            mask_fill=255,
            rng=random.Random(0),
        )(image, mask)
        self.assertIsNotNone(cropped_mask)
        self.assertIn(255, np.unique(cropped_mask))

    def test_deterministic_class_crop_repeats_exactly(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[15:18, 16:19] = 3
        image = np.repeat(mask[:, :, None], 3, axis=2)
        transform = DeterministicClassCrop(size=8, class_ids=(1, 3))
        first_image, first_mask = transform(
            Image.fromarray(image), Image.fromarray(mask)
        )
        second_image, second_mask = transform(
            Image.fromarray(image), Image.fromarray(mask)
        )
        np.testing.assert_array_equal(first_image, second_image)
        np.testing.assert_array_equal(first_mask, second_mask)
        self.assertIn(3, np.unique(first_mask))


if __name__ == "__main__":
    unittest.main()
