# Codex Research Instructions

## Research scope

- The primary dataset and source of all core claims is FloodNet.
- Do not add AIFloodSense experiments until the FloodNet core method passes the predefined stage gates.
- Do not include UrbanSARFloods in the current main experiments. It is a future cross-modal direction.
- Use PyTorch and SegFormer-B0 unless a decision is first recorded in the decision log.
- Assume one GPU with approximately 24 GB VRAM and an initial crop size of 512×512.
- The available dataset is the EARTHVISION 2021 Track 1 challenge release: 398 labeled training images, 1047 unlabeled training images, 450 validation images without public masks, and 448 test images without public masks.
- Local metrics must use the versioned 278/60/60 split derived from the 398 public masks. Never report the challenge Validation/Test sets as locally evaluated ground truth.

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

Complete Week 1:

1. merge-extract the seven Track 1 ZIP packages into one read-only data root;
2. generate and version the canonical 278/60/60 multi-label stratified split;
3. implement and unit-test the dataset, metrics, augmentations, and sliding-window inference;
4. pass CPU smoke tests and a four-image overfit test locally;
5. rent a GPU only after the local gate passes, then train the SegFormer-B0 supervised baseline;
6. write the exact commands and update all three state files.
