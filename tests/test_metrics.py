from __future__ import annotations

import unittest

import numpy as np

from floodnet_ssl.metrics import (
    SegmentationMeter,
    boundary_f1,
    confusion_matrix,
    segmentation_metrics,
)


class MetricsTest(unittest.TestCase):
    def test_perfect_prediction(self) -> None:
        target = np.asarray(
            [
                [0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9],
            ],
            dtype=np.uint8,
        )
        metrics = segmentation_metrics(target, target)
        self.assertAlmostEqual(1.0, metrics["miou10"])
        self.assertAlmostEqual(1.0, metrics["miou9"])
        self.assertAlmostEqual(1.0, metrics["macro_f1"])
        self.assertAlmostEqual(1.0, metrics["precision_per_class"][1])
        self.assertAlmostEqual(1.0, metrics["recall_per_class"][1])
        self.assertAlmostEqual(1.0, metrics["flooded_miou"])
        self.assertAlmostEqual(1.0, metrics["affected_miou"])
        self.assertAlmostEqual(1.0, metrics["building_iou"])
        self.assertAlmostEqual(1.0, metrics["road_iou"])
        self.assertAlmostEqual(1.0, metrics["state_macro_f1"])

    def test_ignore_index_is_excluded(self) -> None:
        target = np.asarray([[0, 255], [1, 1]], dtype=np.int64)
        prediction = np.asarray([[0, 9], [1, 2]], dtype=np.int64)
        matrix = confusion_matrix(prediction, target)
        self.assertEqual(3, int(matrix.sum()))
        self.assertEqual(0, int(matrix[:, 9].sum()))

    def test_boundary_f1_with_tolerance(self) -> None:
        target = np.zeros((12, 12), dtype=np.uint8)
        prediction = np.zeros_like(target)
        target[:, 5:] = 1
        prediction[:, 6:] = 1
        self.assertAlmostEqual(1.0, boundary_f1(prediction, target, tolerance=1))
        self.assertLess(boundary_f1(prediction, target, tolerance=0), 1.0)

    def test_meter_accumulates_confusion(self) -> None:
        meter = SegmentationMeter()
        target = np.asarray([[0, 1], [2, 3]])
        meter.update(target, target)
        meter.update(target, target)
        result = meter.compute()
        self.assertEqual(8, int(result["confusion_matrix"].sum()))
        self.assertAlmostEqual(1.0, result["miou10"])


if __name__ == "__main__":
    unittest.main()
