# Codex Research Instructions

## Research scope

- The primary dataset and source of all core claims is FloodNet.
- Do not add AIFloodSense experiments until the FloodNet core method passes the predefined stage gates.
- Do not include UrbanSARFloods in the current main experiments. It is a future cross-modal direction.
- Use PyTorch and SegFormer-B0 unless a decision is first recorded in the decision log.
- Assume one GPU with approximately 24 GB VRAM and an initial crop size of 512×512.
- The current main dataset is `FloodNet-Supervised_v1.0`: 1445 official training images, 450 validation images, and 448 test images, all with segmentation masks.
- Main local metrics now use the official supervised Validation/Test masks from `splits/floodnet_supervised_v1/manifest.csv`.
- The older EARTHVISION 2021 Track 1 challenge-release audit with 398 public masks, 1047 unlabeled training images, and the 278/60/60 split is retained only as a historical artifact. Do not compare old-protocol numbers directly with new supervised-protocol numbers.

## Required reading order

At the beginning of every new Codex thread or remote-machine session, read:

1. `outputs/floodnet_handoff_state.md`
2. `outputs/floodnet_decision_log.md`
3. `outputs/floodnet_codex_8week_execution_plan.md`
4. `outputs/floodnet_idea_experiment_spec.md`
5. `outputs/floodnet_experiment_registry.csv`
6. `outputs/floodnet_dataset_audit.md`

Treat these files as the persistent research memory.

## Execution rules

- Follow the current stage gate before implementing later modules.
- Never fabricate, estimate, or silently overwrite experiment results.
- Append every completed, failed, or interrupted run to `outputs/floodnet_experiment_registry.csv`.
- Record any module addition, removal, replacement, threshold change, protocol change, or plan contraction in `outputs/floodnet_decision_log.md` before executing it.
- Update `outputs/floodnet_handoff_state.md` before ending each working session.
- Preserve fixed dataset splits, seeds, configs, commands, checkpoints, logs, and code commit identifiers.
- Report mean and standard deviation over three seeds for final primary results.
- Compare structural pseudo-label selection at matched coverage, not only at different thresholds.
- If the full relation module fails its gate, deliver the validated hierarchy-plus-boundary contraction instead of selectively reporting results.

## Data and artifact safety

- Never commit raw datasets, credentials, checkpoints, TensorBoard logs, generated crops, caches, or large downloaded papers.
- Keep local dataset paths in ignored local configuration or environment variables.
- Before any training run, verify that train, validation, and test identifiers do not overlap.
- Do not modify original FloodNet files in place.
- Develop and test the code locally before renting a GPU. A four-image overfit test must pass before any paid full training.

## Immediate next objective

Complete the current supervised comparison stage:

1. keep both FloodNet data roots read-only;
2. use `tools/build_floodnet_splits.py` to version the Challenge-derived 398 labeled IDs, 1047 unlabeled IDs, full 1445 train IDs, val 450 IDs, and test 448 IDs;
3. train both `sup398` and `full1445` through the unified `train.py` entry point with matched SegFormer-B0, ImageNet pretrained weights, CE+Dice, AdamW, augmentations, crop/batch, max_iterations, seed, val/test, metrics, and sliding-window inference;
4. use Validation only for checkpoint selection and Test only for final evaluation through `evaluate.py`;
5. do not implement full `ssl398_1047` until the supervised comparison is verified;
6. write exact commands and update registry, decision log, and handoff state after every run.
