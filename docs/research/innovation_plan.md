# FloodNet Structure-Aware SSL Innovation Plan

Date: 2026-07-02

Base repository state reviewed: `be475f2901a06ca40a4d3cc166a3fbce3eddd440` on `main`.

## 1. Code Audit

The current repository already has a clean supervised SegFormer-B0 baseline path:

- Data: `FloodNetDataset` reads versioned manifests and keeps original FloodNet roots read-only. RGB images are resized to the mask grid when required, masks remain nearest-neighbor class-index tensors, and `sup398`/`full1445` share the official Validation/Test masks.
- Training: `train.py` is iteration-based, uses CE+Dice, AdamW, poly decay, warmup, gradient clipping, AMP, validation-only checkpoint selection, and refuses to overwrite an existing output directory.
- Evaluation: `evaluate.py` loads a checkpoint and runs sliding-window softmax fusion. Test evaluation is separated from training.
- Metrics: `metrics.py` already reports mIoU-10, mIoU-9, macro-F1, affected mIoU, grouped building/road IoU, state macro-F1, and boundary F1.
- Model interface: `SegmentationModelOutput` and `MultiHeadSegmentationModel` already define a disabled-by-default auxiliary-head contract, but the active SegFormer path still exposes only the semantic logits.
- Config: YAML validation currently permits `sup398`, `full1445`, `ssl398_1047`, and `overfit4`, while preserving the canonical 512 crop and SegFormer-B0 policy.

The safest extension strategy is therefore not to rewrite the training stack. Each module should add a config-gated auxiliary path and a narrowly scoped loss/metric/test layer. When the module is disabled, `extract_logits(model(images))` and the original CE+Dice path must behave exactly as the supervised baselines do now.

## 2. Literature Grounding

FloodNet-specific evidence:

- Rahnemoonfar et al., "FloodNet: A High Resolution Aerial Imagery Dataset for Post Flood Scene Understanding" presents UAV imagery after Hurricane Harvey with pixel-wise semantic labels and explicitly highlights flooded roads/buildings and confusion with natural water as hard cases. Source: https://arxiv.org/abs/2012.02951
- Khose et al., "Semi-Supervised Classification and Segmentation on High Resolution Aerial Images" uses FloodNet's 398 labeled and 1047 unlabeled challenge split with pseudo labels, but the segmentation gain is small and does not analyze structural pseudo-label quality at matched coverage. Source: https://arxiv.org/abs/2105.08655

Semi-supervised segmentation and pseudo-label reliability:

- Mean Teacher uses EMA model weights to stabilize consistency targets. Source: https://arxiv.org/abs/1703.01780
- FixMatch combines confidence-thresholded pseudo labels from weak views with strong-view consistency. Source: https://arxiv.org/abs/2001.07685
- CPS trains two perturbed networks with cross pseudo supervision for semantic segmentation. Source: https://arxiv.org/abs/2106.01226
- ST++ shows that holistic prediction stability can be a better reliability signal than raw pixel confidence alone. Source: https://arxiv.org/abs/2106.05095
- UniMatch extends weak-to-strong segmentation consistency with dual strong perturbations and feature perturbation. Source: https://arxiv.org/abs/2208.09910

Boundary and structure modeling:

- Gated-SCNN separates a shape stream and uses semantic gates to sharpen object boundaries, especially thin and small structures. Source: https://arxiv.org/abs/1907.05740
- SegFix treats boundary pixels as less reliable than interior pixels and learns boundary-to-interior correction directions. Source: https://arxiv.org/abs/2007.04269
- Boundary IoU argues that conventional IoU can hide boundary errors and motivates boundary-sensitive evaluation. Source: https://arxiv.org/abs/2103.16562
- BCANet uses boundary guidance for context aggregation, not merely a boundary loss, which is closest to the desired boundary-context direction. Source: https://arxiv.org/abs/2110.14587

Hierarchy, multitask, and structured outputs:

- Kendall et al. show that semantic segmentation can benefit from carefully weighted multi-task supervision. Source: https://arxiv.org/abs/1705.07115
- PAD-Net uses auxiliary tasks as intermediate predictions that guide final scene parsing. Source: https://arxiv.org/abs/1805.04409
- HSSN formulates semantic segmentation with label hierarchy constraints instead of only flat mutually exclusive classes. Source: https://arxiv.org/abs/2203.14335
- Output-space structure has been used as a meaningful signal for segmentation adaptation, supporting the idea that segmentation maps contain exploitable spatial/class structure. Source: https://arxiv.org/abs/1802.10349

## 3. Design Rationale

FloodNet's hardest labels are not arbitrary ten-way classes. Four classes encode a product of object identity and damage state:

- Building-flooded = building x flooded
- Building-non-flooded = building x non-flooded
- Road-flooded = road x flooded
- Road-non-flooded = road x non-flooded

This structure suggests a staged research path:

1. Factor the flat labels into object and state predictions. This tests whether sharing object evidence across flooded/non-flooded variants improves minority disaster classes.
2. Add boundary-context modeling. Flooded road/building errors often occur at object boundaries and thin structures, so boundary should guide context aggregation rather than act only as an auxiliary target.
3. Use the resulting structure signals to filter pseudo labels. Confidence alone is brittle under class imbalance; pseudo labels should also be stable across views, boundary-consistent, and region-consistent.

The modules are logically related but independently ablatable:

- State factorization can run in pure supervised mode.
- Boundary context can run without state factorization.
- Structure-aware pseudo-label filtering can start from semantic confidence and optionally consume state/boundary signals if their branches are enabled.

## 4. Module Plans

### Module A: Object-State Factorization

Branch: `exp/state-factorization`

Implementation target:

- Add deterministic mappings from semantic IDs to 8 object IDs and 2 state IDs.
- Add object and state auxiliary heads through the existing unified model-output contract.
- Add CE losses for object and state targets, with `IGNORE_INDEX` outside valid state pixels.
- Compose a hierarchical ten-class distribution from object and state probabilities.
- Add a Jensen-Shannon consistency term between direct semantic probabilities and the composed hierarchical distribution.
- Add `configs/segformer_b0_sup398_state_factorization.yaml`.
- Add tests for target mapping, composed probabilities, JS loss, and disabled-by-default behavior.

Primary metrics:

- mIoU-9/mIoU-10, affected mIoU.
- Building/Road IoU.
- State macro-F1 and flooded precision/recall.
- Consistency loss magnitude.

Risk:

- State supervision is valid only on building/road pixels. Overweighting it may hurt other classes, so the first branch keeps small default weights and explicit YAML control.

### Module B: Boundary Context Modeling

Branch: `exp/boundary-context`

Implementation target:

- Generate boundary targets from class-index masks with configurable width.
- Add a boundary prediction head.
- Add BCE+Dice boundary supervision and semantic-boundary consistency.
- Add a boundary-guided context refinement block so the branch changes feature/logit aggregation, not only the loss.
- Add `configs/segformer_b0_sup398_boundary_context.yaml`.
- Add tests for boundary target generation, differentiable loss, context refinement shape, and disabled baseline behavior.

Primary metrics:

- Mean Boundary F1.
- Building/Road/Flooded boundary F1 where available.
- Affected mIoU to verify boundary gains do not come at the cost of disaster classes.

Risk:

- Boundary targets are sparse. The implementation should use positive weighting and Dice to avoid the all-background boundary solution.

### Module C: Structure-Aware Pseudo-Label Filtering

Branch: `exp/structure-aware-pl`

Implementation target:

- Add a basic EMA teacher-student training entry point for `ssl398_1047`.
- Use labeled CE+Dice and unlabeled pseudo-label CE with ramp-up.
- Compute pseudo labels from a weak teacher view and supervise a strong student view.
- Filter pseudo labels with confidence, multi-view consistency, boundary stability, and region consistency.
- Support matched-coverage evaluation utilities so structure-aware filtering is compared fairly against confidence-only selection.
- Add `configs/segformer_b0_ssl398_1047_structure_pl.yaml`.
- Add tests for EMA updates, pseudo-label masks, score composition, region consistency, and dry-run counts.

Primary metrics:

- Validation/Test segmentation metrics after supervised protocol gates pass.
- Pseudo-label mIoU, per-class precision/recall, and coverage using hidden train labels only for offline analysis.
- Matched-coverage pseudo-label quality curves.

Risk:

- This module should not be used to claim final gains until the supervised comparison and a basic Mean Teacher/FixMatch-style baseline are verified.

## 5. Recommended Priority

1. State factorization first: it is the smallest supervised structural hypothesis and directly targets FloodNet's class taxonomy.
2. Boundary context second: it is still supervised and improves the quality signals needed by pseudo-label filtering.
3. Structure-aware pseudo labels third: it has the largest training-schedule risk and should only be run after the supervised baselines and module smoke tests are stable.

## 6. Branching and Test Policy

All three branches will be created from the same main commit containing this plan. Each branch will have:

- One module implementation commit.
- One module-specific experiment document under `docs/experiments/`.
- YAML config flags that default to off in existing baseline configs.
- Unit tests and dry-run commands.

No raw data, checkpoints, caches, `outputs/`, TensorBoard logs, or generated training artifacts will be committed.

## 7. 2026-07-03 中文设计展开与控制变量复核

当前状态：`sup398` 与 `full1445` 两个监督实验正在运行，尚未形成可汇报最终结果。三个结构模块已经分别在独立分支实现并通过单元测试与 dry-run；当前 main 只记录研究设计、审查结论和后续执行约束，不记录任何未完成训练的结果。

详细中文设计文档：

- `docs/experiments/state_factorization.md`
- `docs/experiments/boundary_context.md`
- `docs/experiments/structure_aware_pseudolabel.md`
- `docs/experiments/control_variable_review.md`

分支与提交：

- `exp/state-factorization`：`8332c34`
- `exp/boundary-context`：`dd88bff`
- `exp/structure-aware-pl`：`a39ce76`

复核结论：

1. 三个分支均从共同 main 提交 `d4fedcf` 创建，满足同源分支要求。
2. 三个分支均未修改 FloodNet 数据划分和类别编号。
3. 两个监督结构分支保持与 `sup398` 基线一致的 SegFormer-B0、ImageNet 预训练、CE+Dice、AdamW、512 crop、40000 iterations、Validation checkpoint selection 和 sliding-window evaluation。
4. 新功能均通过 YAML 控制，原有 `sup398` 和 `full1445` 基线配置不启用新增模块。
5. `structure-aware-pl` 是后置半监督阶段，不能在当前监督比较完成前作为主实验启动；正式运行前必须先补齐 confidence-only EMA baseline 与 matched coverage 比较。

推荐优先级保持不变：先跑 state factorization，再跑 boundary context，最后跑 structure-aware pseudo-label filtering。原因是前两个模块仍是监督结构消融，能在当前 protocol 下直接和 `sup398` 公平比较；第三个模块引入无标签数据和 teacher-student 训练，变量更多，必须等待监督门禁通过后再进入。

## 8. 2026-07-03 后续实验展开补充

本项目中的半监督特指 `ssl398_1047`：398 张 labeled 图像使用官方 mask 计算监督损失，1047 张 unlabeled 图像只使用 RGB 图像，由 EMA teacher 产生伪标签后参与无标签损失。1047 张图像即使在本地 supervised 数据中存在 mask，也必须作为隐藏标签处理；这些 mask 只能用于离线伪标签质量审计，不能用于训练、调参、checkpoint 选择或 Test 前决策。

后续暂定路线已经写入 `docs/experiments/control_variable_review.md` 和 `docs/experiments/structure_aware_pseudolabel.md`。核心顺序为：

1. 完成 `sup398` 与 `full1445` 两个监督基线。
2. 做监督错误分析，明确受灾状态、边界和水体混淆问题。
3. 依次跑 `state-factorization` 与 `boundary-context` 的监督消融。
4. 做 `sup398` teacher 的 1047 张 unlabeled 伪标签质量审计。
5. 跑 EMA confidence-only 半监督基线。
6. 在 matched coverage 下逐项加入 region、boundary、multiview 结构筛选。
7. 仅对单 seed 通过门禁的配置做三 seed 和最终 Test。
