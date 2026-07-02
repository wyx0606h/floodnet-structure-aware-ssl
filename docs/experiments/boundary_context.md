# Boundary Context Modeling

Branch: `exp/boundary-context`

## Motivation

FloodNet's affected roads and buildings are small, thin, and boundary-sensitive. A plain boundary loss can make the network draw sharper edges, but it does not force semantic features to use boundary information. This branch therefore adds a boundary prediction head and uses the predicted boundary map as a gate for local context aggregation.

## Method

Boundary targets are generated from semantic masks:

\[
E_{gt} = \operatorname{Dilate}_w(\{y_p \neq y_q \mid p,q \text{ are 4-neighbors}\}).
\]

The model predicts semantic logits \(Z\) and boundary logits \(B\). The boundary gate controls context refinement:

\[
G=\sigma(B), \quad C=\operatorname{AvgPool}_k(Z),
\]

\[
Z' = Z + \alpha(1-G)(C-Z).
\]

Interior pixels receive more local context smoothing, while high-boundary pixels preserve sharper semantic logits. This makes boundary a modeling signal, not only an auxiliary supervision target.

The boundary objective is:

\[
\mathcal{L}_{bd}=
\lambda_b\mathcal{L}_{BCE}(B,E_{gt})
\lambda_d\mathcal{L}_{Dice}(B,E_{gt})
\lambda_c\lVert\sigma(B)-\widetilde{E}(Z')\rVert_1,
\]

where \(\widetilde{E}(Z')\) is a differentiable soft semantic edge map from class probabilities.

## Code Structure

- `floodnet_ssl/boundary_context.py`: target generation, soft semantic edge, BCE+Dice loss, boundary-guided refinement.
- `floodnet_ssl/models.py`: `BoundaryContextWrapper` attaches the boundary head and refines logits.
- `floodnet_ssl/losses.py`: `supervised_objective` adds boundary-context loss when enabled.
- `configs/segformer_b0_sup398_boundary_context.yaml`: sup398 experiment config.
- `tests/test_boundary_context.py`: unit coverage.

## Configuration

```yaml
model:
  boundary_context:
    enabled: true
    strength: 0.25
    kernel_size: 5

modules:
  boundary_context:
    enabled: true
    target_width: 3
    bce_weight: 1.0
    dice_weight: 1.0
    consistency_weight: 0.1
```

Existing baseline configs do not contain `model.boundary_context` or `modules.boundary_context`, so they retain the original supervised behavior.

## Run Commands

Dry-run:

```powershell
& 'D:\Anaconda3\python.exe' train.py `
  --config configs\segformer_b0_sup398_boundary_context.yaml `
  --supervised-root F:\FloodNet `
  --dry-run
```

Training after the supervised comparison gate:

```powershell
& 'D:\Anaconda3\python.exe' train.py `
  --config configs\segformer_b0_sup398_boundary_context.yaml `
  --supervised-root F:\FloodNet
```

## Ablation

- Baseline `sup398`.
- Boundary head with BCE+Dice only.
- Boundary head plus semantic-boundary consistency.
- Full boundary-context refinement.
- Context kernel size 3/5/7.
- Boundary target width 1/3/5.

## Risks

Boundary pixels are sparse and can dominate noisy thin structures if overweighted. A successful run must improve Boundary F1 without reducing flooded building/road IoU. If Boundary F1 improves but affected mIoU drops, the branch should be treated as a visualization/auxiliary signal rather than a final method component.
