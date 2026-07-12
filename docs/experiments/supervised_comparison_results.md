# Supervised Comparison Results

Date: 2026-07-04

Last evidence audit: 2026-07-11

Source artifacts: `F:\floodnet_output\segformer_b0_sup398` and `F:\floodnet_output\segformer_b0_full1445`.

This note summarizes the first completed supervised comparison under the current FloodNet supervised protocol. The values below are from the checkpoints selected by Validation mIoU and then evaluated once on the official supervised Test split. Raw checkpoints, prediction PNGs, logs, and training artifacts remain outside the repository.

## 0. Evidence and reporting scope

The following files were read directly for both runs:

- `config_resolved.yaml` for the realized protocol;
- `train_summary.json`, `train.log`, and `curves/history.csv` for completion and Validation checkpoint selection;
- `metrics/test_best/metrics.json`, `class_metrics.csv`, `confusion_matrix.csv`, and `per_sample_metrics.csv` for Test analysis;
- `runtime_metadata.json` for the software and GPU environment.

Both histories contain exactly 40000 optimizer-step rows, both summaries report completion at 40000 steps, and no exception or non-finite-loss event was found in the recorded training logs. The Test folders contain metrics for 448 samples. These checks support artifact completeness; they do not establish statistical repeatability because each setting has only one run with seed `20260702`.

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

This path difference is a reproducibility caveat. The artifacts identify the full-supervision snapshot as `80983a413c30d36a39c20203974ae7807835e2b4`, while the `sup398` artifact records only the model ID. The evidence supports the same model family and pretraining source, but it does not by itself prove byte-identical cached weights. Future runs should pin the same snapshot or record a weight checksum.

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

Paired per-image diagnostics qualify this aggregate view:

| Per-image metric | Valid paired images | `full1445` wins | Ties | Mean delta | Median delta |
|---|---:|---:|---:|---:|---:|
| Boundary F1 | 448 | 200 | 18 | +0.49 | -0.19 |
| Building IoU | 260 | 115 | 64 | +3.69 | 0.00 |
| Road IoU | 342 | 197 | 68 | +6.44 | +1.13 |
| State Macro-F1 | 310 | 237 | 2 | +11.50 | +1.90 |

`NaN` entries are excluded where a sample has no valid building, road, or state target. The state gain is broad across valid images, whereas the global Boundary F1 increase is not a uniform per-image improvement: fewer than half of Test images improve and the median delta is slightly negative. This is evidence for running a dedicated boundary experiment, not evidence that a boundary module will necessarily work.

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

The last statement should be read as a diagnostic interpretation, not a proven mechanism. Numerically, `sup398` falls from 49.88 best Validation mIoU-10 to 49.16 at step 40000, while `full1445` falls from 56.73 to 56.41. Neither curve is still clearly rising at the end, so the current 40000-step budget is adequate for the next matched pilot and should not be extended for only one method.

The best-Validation to Test mIoU-10 gaps are -2.21 points for `sup398` and -3.99 points for `full1445`. A lower Test score than Validation is plausible under a fixed split, but the larger full-supervision gap should be retained as a split-specific observation rather than interpreted as proof of overfitting.

## 6. Interpretation for Next Experiments

The supervised comparison gate is passed for starting supervised structure ablations. The next experiments should remain fixed-seed pilot runs unless compute becomes available for multi-seed repetition.

Recommended order:

1. Pull and check out `origin/exp/state-factorization`, run the real-model/small-step smoke, and then start frozen S1 using `configs/segformer_b0_sup398_state_s1_logit_shared.yaml`.
2. Continue S2 -> S3 -> S4 only with the frozen configs. S1 is a logit/shared-state control; it is not the complete proposed method.
3. Run `exp/boundary-context` against the same `sup398` baseline and inspect Boundary F1, Affected mIoU, and per-class IoU.
4. Start SSL only after the supervised structure ablations are understood, beginning with an EMA confidence-only baseline before structure-aware pseudo-label filtering.

For writing, these values should be described as fixed-seed results, not as mean +/- standard deviation. A suitable sentence is:

> Unless otherwise specified, all experiments are conducted under the same data split and training protocol with a fixed random seed for reproducibility.

Do not label these values as an average, mean, standard deviation, significant improvement, or stable improvement. The current evidence supports: (a) both runs completed normally, (b) the full-label reference is stronger on the recorded Test metrics, and (c) the fixed-seed supervised gate is sufficient to start controlled state-factorization pilots. It does not yet support a variance or causal-mechanism claim.

