# Structure-Aware Pseudo-Label Filtering

Branch: `exp/structure-aware-pl`

## Motivation

FloodNet's pseudo labels are unreliable when selected only by maximum softmax probability. Flooded buildings and roads are rare, and boundary pixels can be confidently wrong. This branch implements a basic EMA teacher-student loop and makes pseudo-label selection depend on semantic confidence plus structural stability.

## Method

The teacher is an exponential moving average of the student:

\[
\theta_T \leftarrow \alpha \theta_T + (1-\alpha)\theta_S.
\]

For an unlabeled image, the teacher predicts logits \(Z_T\). The pseudo label is:

\[
\hat y = \arg\max_c \operatorname{softmax}(Z_T)_c.
\]

The reliability score combines four terms:

\[
s = \frac{
w_c s_c + w_m s_m + w_b s_b + w_r s_r
}{
w_c+w_m+w_b+w_r
}.
\]

Terms:

- \(s_c\): maximum class probability.
- \(s_m\): multi-view consistency by teacher probability dot product.
- \(s_b\): boundary stability from soft semantic edge maps.
- \(s_r\): region consistency, the local fraction of pixels sharing the pseudo label.

Pixels are kept by either an absolute threshold or a matched-coverage threshold. Matched coverage is required for fair comparison against confidence-only filtering.

Student loss:

\[
\mathcal{L} = \mathcal{L}_{CE+Dice}^{labeled}
+ \lambda_u \mathcal{L}_{CE}^{unlabeled}(\hat y, M),
\]

where \(M\) is the structure-aware keep mask.

## Code Structure

- `floodnet_ssl/pseudolabels.py`: EMA update, structure scores, matched coverage, pseudo-label masks.
- `train_ssl.py`: EMA teacher-student entry point for `ssl398_1047`.
- `configs/segformer_b0_ssl398_1047_structure_pl.yaml`: SSL run config.
- `tests/test_pseudolabels.py`: unit coverage.

## Configuration

```yaml
ssl:
  ema_decay: 0.999
  threshold: 0.8
  matched_coverage:
  unsupervised_weight: 1.0
  confidence_weight: 1.0
  multiview_weight: 0.25
  boundary_weight: 0.25
  region_weight: 0.25
  region_kernel_size: 5
```

The branch intentionally leaves the original `train.py` supervised baseline untouched. SSL is run through `train_ssl.py`.

## Run Commands

Dry-run:

```powershell
& 'D:\Anaconda3\python.exe' train_ssl.py `
  --config configs\segformer_b0_ssl398_1047_structure_pl.yaml `
  --supervised-root F:\FloodNet `
  --dry-run
```

Training, only after supervised comparison and four-image gates:

```powershell
& 'D:\Anaconda3\python.exe' train_ssl.py `
  --config configs\segformer_b0_ssl398_1047_structure_pl.yaml `
  --supervised-root F:\FloodNet
```

## Ablation

- Confidence only.
- Confidence + multi-view consistency.
- Confidence + boundary stability.
- Confidence + region consistency.
- Full score.
- Matched coverage: 20%, 40%, 60%, 80% pseudo-label coverage.

## Risks

This branch is intentionally not a final method claim until the supervised baselines and a basic EMA/Mean Teacher baseline are verified. The current code provides the first runnable EMA loop and filtering mechanics; final experiments must add full weak/strong augmentation parity and report pseudo-label quality at matched coverage.
