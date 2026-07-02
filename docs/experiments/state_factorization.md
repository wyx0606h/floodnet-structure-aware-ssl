# Object-State Factorization

Branch: `exp/state-factorization`

## Motivation

FloodNet encodes damage state inside semantic class names: flooded/non-flooded building and flooded/non-flooded road. A flat ten-way classifier must learn four rare classes independently, even though the object identity evidence is shared. This branch tests whether explicit object/state supervision improves object localization and state recognition without changing class IDs or the official splits.

## Method

The semantic target \(y \in \{0,\ldots,9\}\) is mapped to:

\[
y^{obj} \in \{\text{background}, \text{building}, \text{road}, \text{water}, \text{tree}, \text{vehicle}, \text{pool}, \text{grass}\}
\]

and, only for building/road pixels:

\[
y^{state} \in \{\text{non-flooded}, \text{flooded}\}.
\]

The model keeps the original semantic head and adds object/state auxiliary heads:

\[
P^{sem}=H_{sem}(x),\quad P^{obj}=H_{obj}(x),\quad P^{state}=H_{state}(x).
\]

The hierarchical distribution is composed as:

\[
P^{hier}(\text{building-flooded}) =
P^{obj}(\text{building})P^{state}(\text{flooded}),
\]

with analogous formulas for non-flooded building and both road states. Non-factorized classes inherit their object probability directly.

Training loss:

\[
\mathcal{L}=\mathcal{L}_{CE+Dice}^{sem}
\lambda_o\mathcal{L}_{CE}^{obj}
\lambda_s\mathcal{L}_{CE}^{state}
\lambda_h D_{JS}(P^{sem},P^{hier}).
\]

## Code Structure

- `floodnet_ssl/state_factorization.py`: label mappings, probability composition, JS consistency, auxiliary loss.
- `floodnet_ssl/models.py`: `LogitAuxiliaryWrapper` attaches object/state heads when configured.
- `floodnet_ssl/losses.py`: `supervised_objective` combines semantic and config-gated auxiliary losses.
- `configs/segformer_b0_sup398_state_factorization.yaml`: sup398 experiment config.
- `tests/test_state_factorization.py`: unit coverage.

## Configuration

```yaml
model:
  auxiliary_heads: [object, state]

modules:
  state_factorization:
    enabled: true
    object_weight: 0.25
    state_weight: 0.25
    consistency_weight: 0.1
```

All existing supervised baseline configs omit `modules.state_factorization`, so the original path remains unchanged.

## Run Commands

Dry-run:

```powershell
& 'D:\Anaconda3\python.exe' train.py `
  --config configs\segformer_b0_sup398_state_factorization.yaml `
  --supervised-root F:\FloodNet `
  --dry-run
```

Training, after the supervised baseline gate is satisfied:

```powershell
& 'D:\Anaconda3\python.exe' train.py `
  --config configs\segformer_b0_sup398_state_factorization.yaml `
  --supervised-root F:\FloodNet
```

Final test evaluation uses `evaluate.py` with the best validation checkpoint only after the config is frozen.

## Ablation

- Baseline `sup398`.
- `+ object` head only.
- `+ object + state`.
- `+ object + state + JS consistency`.
- Weight sweep for `state_weight` and `consistency_weight` on one seed before any final three-seed runs.

## Risks

State labels are meaningful only on building/road pixels. If state loss is overweighted, it can overfit rare affected pixels or reduce general semantic quality. The branch therefore uses small default weights and must be judged by affected mIoU, state macro-F1, and standard mIoU together.
