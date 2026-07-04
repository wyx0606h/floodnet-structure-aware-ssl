# Supervised Comparison Results

Date: 2026-07-04

Source artifacts: `F:\floodnet_output\segformer_b0_sup398` and `F:\floodnet_output\segformer_b0_full1445`.

This note summarizes the first completed supervised comparison under the current FloodNet supervised protocol. The values below are from the checkpoints selected by Validation mIoU and then evaluated once on the official supervised Test split. Raw checkpoints, prediction PNGs, logs, and training artifacts remain outside the repository.

## 1. Protocol

Both runs use the same experimental protocol except for the number of labeled training images.

| Item | `sup398` | `full1445` |
|---|---:|---:|
| Labeled train images | 398 | 1445 |
| Validation images | 450 | 450 |
| Test images | 448 | 448 |
| Backbone | SegFormer-B0 | SegFormer-B0 |
| Pretraining | ImageNet / `nvidia/mit-b0` | ImageNet / `nvidia/mit-b0` snapshot |
| Loss | CE + Dice | CE + Dice |
| Optimizer | AdamW | AdamW |
| Crop | 512 x 512 | 512 x 512 |
| Effective batch size | 8 | 8 |
| Max optimizer steps | 40000 | 40000 |
| Validation interval | 2000 | 2000 |
| Seed | 20260702 | 20260702 |
| Evaluation | sliding-window softmax fusion | sliding-window softmax fusion |

The `sup398` config resolved the pretrained model as `nvidia/mit-b0`, while `full1445` resolved it as a local HuggingFace snapshot path. The snapshot corresponds to the same model family, but future reproduction notes should keep the snapshot/hash explicit.

## 2. Main Results

Validation is used only for checkpoint selection. The primary reportable numbers are the Test metrics from `metrics/test_best/metrics.json`.

| Method | Best Val Iter | Val mIoU-10 | Val mIoU-9 | Test mIoU-10 | Test mIoU-9 | Test Macro-F1 | Test Affected mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| SegFormer-B0 `sup398` | 18000 | 49.88 | 55.31 | 47.67 | 52.77 | 60.89 | 34.34 |
| SegFormer-B0 `full1445` | 32000 | 56.73 | 61.61 | 52.74 | 57.75 | 66.00 | 38.10 |

`full1445` improves over `sup398` on the Test split by:

- +5.07 mIoU-10.
- +4.98 mIoU-9.
- +5.12 Macro-F1.
- +3.77 Affected mIoU.
- +0.87 pixel accuracy.

This supports the basic sanity expectation that the full 1445-label supervision setting should outperform the 398-label supervised baseline under the same model and training protocol.

## 3. Structure-Oriented Metrics

These metrics are especially relevant for later state-factorization and boundary-context experiments.

| Method | Test Boundary F1 | Test Building IoU | Test Road IoU | Test State Macro-F1 |
|---|---:|---:|---:|---:|
| SegFormer-B0 `sup398` | 16.36 | 42.93 | 43.53 | 68.99 |
| SegFormer-B0 `full1445` | 16.85 | 52.75 | 50.52 | 80.49 |

The full supervision run gives a large gain in grouped Building IoU, grouped Road IoU, and State Macro-F1. Boundary F1 changes only slightly, which leaves room for the dedicated boundary-context branch to demonstrate whether boundary supervision and boundary-guided refinement add value beyond more labels.

## 4. Per-Class Test IoU

| Class | `sup398` IoU | `full1445` IoU | Delta |
|---|---:|---:|---:|
| Background | 1.74 | 7.68 | +5.94 |
| Building-flooded | 42.12 | 44.52 | +2.40 |
| Building-non-flooded | 59.10 | 69.47 | +10.36 |
| Road-flooded | 26.55 | 31.69 | +5.14 |
| Road-non-flooded | 66.77 | 73.28 | +6.51 |
| Water | 57.86 | 58.67 | +0.81 |
| Tree | 70.35 | 71.95 | +1.61 |
| Vehicle | 36.47 | 47.14 | +10.67 |
| Pool | 33.62 | 39.75 | +6.13 |
| Grass | 82.09 | 83.25 | +1.16 |

The largest gains occur on non-flooded building, vehicle, non-flooded road, pool, and flooded road. The key affected classes both improve, but flooded building improves modestly and flooded road remains difficult.

## 5. Error Pattern Notes

The results are plausible, but they reveal several important weaknesses:

- Background IoU is very low in both runs. In the Test confusion matrix, much of the ground-truth Background area is predicted as Water or Grass. This likely reflects the heterogeneous meaning and low support of the Background class rather than a total training failure. Main tables should therefore report both mIoU-10 and mIoU-9.
- `full1445` improves affected-class recall but reduces affected-class precision. Building-flooded recall rises from 60.33 to 72.30, while precision drops from 58.25 to 53.67. Road-flooded recall rises from 34.37 to 55.64, while precision drops from 53.86 to 42.41. The full-supervision model is less conservative about predicting flooded structures.
- Water versus flooded road remains a major confusion mode. This is aligned with the research motivation that flood damage state cannot be solved by confidence alone and may benefit from state, boundary, and context signals.
- `sup398` reaches its best Validation mIoU at 18000 steps and then plateaus. Its final training loss is very low, suggesting expected overfitting under limited labels. `full1445` peaks later around 32000 steps and keeps a stronger validation/test profile.

## 6. Interpretation for Next Experiments

The supervised comparison gate is passed for starting supervised structure ablations. The next experiments should remain fixed-seed pilot runs unless compute becomes available for multi-seed repetition.

Recommended order:

1. Run `exp/state-factorization` against the `sup398` baseline and inspect Affected mIoU, Building/Road IoU, and State Macro-F1.
2. Run `exp/boundary-context` against the same `sup398` baseline and inspect Boundary F1, Affected mIoU, and per-class IoU.
3. Start SSL only after the supervised structure ablations are understood, beginning with an EMA confidence-only baseline before structure-aware pseudo-label filtering.

For writing, these values should be described as fixed-seed results, not as mean +/- standard deviation. A suitable sentence is:

> Unless otherwise specified, all experiments are conducted under the same data split and training protocol with a fixed random seed for reproducibility.

