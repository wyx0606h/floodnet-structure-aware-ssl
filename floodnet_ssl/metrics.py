"""NumPy reference metrics for FloodNet semantic segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .constants import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES


def _as_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def confusion_matrix(
    prediction: object,
    target: object,
    *,
    num_classes: int = NUM_CLASSES,
    ignore_index: int | None = IGNORE_INDEX,
) -> np.ndarray:
    prediction_array = _as_numpy(prediction).astype(np.int64, copy=False)
    target_array = _as_numpy(target).astype(np.int64, copy=False)
    if prediction_array.shape != target_array.shape:
        raise ValueError(
            f"Prediction and target shapes differ: "
            f"{prediction_array.shape} versus {target_array.shape}"
        )
    prediction_flat = prediction_array.reshape(-1)
    target_flat = target_array.reshape(-1)
    valid = np.ones_like(target_flat, dtype=bool)
    if ignore_index is not None:
        valid &= target_flat != ignore_index
    valid &= (target_flat >= 0) & (target_flat < num_classes)
    prediction_flat = prediction_flat[valid]
    target_flat = target_flat[valid]
    if np.any((prediction_flat < 0) | (prediction_flat >= num_classes)):
        invalid = np.unique(
            prediction_flat[
                (prediction_flat < 0) | (prediction_flat >= num_classes)
            ]
        )
        raise ValueError(f"Prediction contains invalid class IDs: {invalid.tolist()}")
    encoded = target_flat * num_classes + prediction_flat
    return np.bincount(encoded, minlength=num_classes**2).reshape(
        num_classes, num_classes
    )


def metrics_from_confusion(matrix: object) -> dict[str, object]:
    confusion = _as_numpy(matrix).astype(np.float64, copy=False)
    if confusion.shape != (NUM_CLASSES, NUM_CLASSES):
        raise ValueError(
            f"Expected {NUM_CLASSES}x{NUM_CLASSES} confusion matrix, got {confusion.shape}"
        )
    true_positive = np.diag(confusion)
    false_positive = confusion.sum(axis=0) - true_positive
    false_negative = confusion.sum(axis=1) - true_positive
    iou_denominator = true_positive + false_positive + false_negative
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    f1_denominator = 2 * true_positive + false_positive + false_negative
    iou = np.divide(
        true_positive,
        iou_denominator,
        out=np.full(NUM_CLASSES, np.nan),
        where=iou_denominator > 0,
    )
    precision = np.divide(
        true_positive,
        precision_denominator,
        out=np.full(NUM_CLASSES, np.nan),
        where=precision_denominator > 0,
    )
    recall = np.divide(
        true_positive,
        recall_denominator,
        out=np.full(NUM_CLASSES, np.nan),
        where=recall_denominator > 0,
    )
    f1 = np.divide(
        2 * true_positive,
        f1_denominator,
        out=np.full(NUM_CLASSES, np.nan),
        where=f1_denominator > 0,
    )
    total = confusion.sum()
    pixel_accuracy = float(true_positive.sum() / total) if total else float("nan")
    affected = np.asarray([iou[1], iou[3]], dtype=np.float64)
    return {
        "class_names": CLASS_NAMES,
        "iou_per_class": iou,
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "miou10": float(np.nanmean(iou)),
        "miou9": float(np.nanmean(iou[1:])),
        "macro_f1": float(np.nanmean(f1)),
        "pixel_accuracy": pixel_accuracy,
        "affected_miou": float(np.nanmean(affected)),
        "flooded_miou": float(np.nanmean(affected)),
    }


def segmentation_metrics(
    prediction: object,
    target: object,
    *,
    ignore_index: int | None = IGNORE_INDEX,
) -> dict[str, object]:
    matrix = confusion_matrix(
        prediction, target, num_classes=NUM_CLASSES, ignore_index=ignore_index
    )
    result = metrics_from_confusion(matrix)
    result["confusion_matrix"] = matrix
    result.update(grouped_object_iou(prediction, target, ignore_index=ignore_index))
    result.update(state_metrics_from_semantic(prediction, target, ignore_index=ignore_index))
    return result


def _binary_iou(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    prediction = prediction & valid
    target = target & valid
    intersection = np.count_nonzero(prediction & target)
    union = np.count_nonzero(prediction | target)
    return float(intersection / union) if union else float("nan")


def grouped_object_iou(
    prediction: object,
    target: object,
    *,
    ignore_index: int | None = IGNORE_INDEX,
) -> dict[str, float]:
    pred = _as_numpy(prediction)
    truth = _as_numpy(target)
    if pred.shape != truth.shape:
        raise ValueError("Prediction and target shapes differ")
    valid = np.ones(truth.shape, dtype=bool)
    if ignore_index is not None:
        valid &= truth != ignore_index
    return {
        "building_iou": _binary_iou(
            np.isin(pred, (1, 2)), np.isin(truth, (1, 2)), valid
        ),
        "road_iou": _binary_iou(
            np.isin(pred, (3, 4)), np.isin(truth, (3, 4)), valid
        ),
    }


def state_metrics_from_semantic(
    prediction: object,
    target: object,
    *,
    ignore_index: int | None = IGNORE_INDEX,
) -> dict[str, float]:
    pred = _as_numpy(prediction)
    truth = _as_numpy(target)
    if pred.shape != truth.shape:
        raise ValueError("Prediction and target shapes differ")
    valid = np.isin(truth, (1, 2, 3, 4))
    if ignore_index is not None:
        valid &= truth != ignore_index
    if not np.any(valid):
        return {
            "state_accuracy": float("nan"),
            "state_macro_f1": float("nan"),
            "flooded_precision": float("nan"),
            "flooded_recall": float("nan"),
        }
    true_state = np.where(np.isin(truth, (1, 3)), 1, 0)
    predicted_state = np.full(pred.shape, -1, dtype=np.int8)
    predicted_state[np.isin(pred, (1, 3))] = 1
    predicted_state[np.isin(pred, (2, 4))] = 0

    state_f1: list[float] = []
    precision_by_state: dict[int, float] = {}
    recall_by_state: dict[int, float] = {}
    for state in (0, 1):
        true_positive = np.count_nonzero(
            valid & (true_state == state) & (predicted_state == state)
        )
        false_positive = np.count_nonzero(
            valid & (true_state != state) & (predicted_state == state)
        )
        false_negative = np.count_nonzero(
            valid & (true_state == state) & (predicted_state != state)
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else float("nan")
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else float("nan")
        )
        f1 = (
            2 * true_positive / (2 * true_positive + false_positive + false_negative)
            if 2 * true_positive + false_positive + false_negative
            else float("nan")
        )
        precision_by_state[state] = float(precision)
        recall_by_state[state] = float(recall)
        state_f1.append(float(f1))
    accuracy = np.count_nonzero(
        valid & (predicted_state == true_state)
    ) / np.count_nonzero(valid)
    return {
        "state_accuracy": float(accuracy),
        "state_macro_f1": float(np.nanmean(state_f1)),
        "flooded_precision": precision_by_state[1],
        "flooded_recall": recall_by_state[1],
    }


def semantic_boundary(
    labels: object,
    *,
    class_ids: Iterable[int] | None = None,
    ignore_index: int | None = IGNORE_INDEX,
) -> np.ndarray:
    array = _as_numpy(labels)
    if array.ndim != 2:
        raise ValueError(f"Boundary extraction expects a 2D mask, got {array.shape}")
    valid = np.ones(array.shape, dtype=bool)
    if ignore_index is not None:
        valid &= array != ignore_index
    values = np.isin(array, tuple(class_ids)) if class_ids is not None else array
    boundary = np.zeros(array.shape, dtype=bool)

    horizontal_valid = valid[:, 1:] & valid[:, :-1]
    horizontal_difference = (values[:, 1:] != values[:, :-1]) & horizontal_valid
    boundary[:, 1:] |= horizontal_difference
    boundary[:, :-1] |= horizontal_difference

    vertical_valid = valid[1:, :] & valid[:-1, :]
    vertical_difference = (values[1:, :] != values[:-1, :]) & vertical_valid
    boundary[1:, :] |= vertical_difference
    boundary[:-1, :] |= vertical_difference
    return boundary


def _dilate(binary: np.ndarray, radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError("Dilation radius must be non-negative")
    if radius == 0:
        return binary.copy()
    height, width = binary.shape
    padded = np.pad(binary, radius, mode="constant", constant_values=False)
    dilated = np.zeros_like(binary, dtype=bool)
    for row_offset in range(2 * radius + 1):
        for column_offset in range(2 * radius + 1):
            dilated |= padded[
                row_offset : row_offset + height,
                column_offset : column_offset + width,
            ]
    return dilated


def boundary_f1(
    prediction: object,
    target: object,
    *,
    tolerance: int = 3,
    class_ids: Iterable[int] | None = None,
    ignore_index: int | None = IGNORE_INDEX,
) -> float:
    predicted_boundary = semantic_boundary(
        prediction, class_ids=class_ids, ignore_index=ignore_index
    )
    target_boundary = semantic_boundary(
        target, class_ids=class_ids, ignore_index=ignore_index
    )
    predicted_count = np.count_nonzero(predicted_boundary)
    target_count = np.count_nonzero(target_boundary)
    if predicted_count == 0 and target_count == 0:
        return 1.0
    if predicted_count == 0 or target_count == 0:
        return 0.0
    matched_prediction = np.count_nonzero(
        predicted_boundary & _dilate(target_boundary, tolerance)
    )
    matched_target = np.count_nonzero(
        target_boundary & _dilate(predicted_boundary, tolerance)
    )
    precision = matched_prediction / predicted_count
    recall = matched_target / target_count
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


@dataclass
class SegmentationMeter:
    ignore_index: int | None = IGNORE_INDEX

    def __post_init__(self) -> None:
        self.matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    def update(self, prediction: object, target: object) -> None:
        self.matrix += confusion_matrix(
            prediction,
            target,
            num_classes=NUM_CLASSES,
            ignore_index=self.ignore_index,
        )

    def compute(self) -> dict[str, object]:
        result = metrics_from_confusion(self.matrix)
        result["confusion_matrix"] = self.matrix.copy()
        return result

    def reset(self) -> None:
        self.matrix.fill(0)
