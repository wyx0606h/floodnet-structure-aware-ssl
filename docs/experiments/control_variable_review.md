# 三个结构模块的控制变量审查

日期：2026-07-03
审查对象：`exp/state-factorization`、`exp/boundary-context`、`exp/structure-aware-pl`
当前进度：`sup398` 与 `full1445` 两个监督实验的 first-seed run 已完成，`test_best` 结果已整理到 `docs/experiments/supervised_comparison_results.md`。

## 1. 总体结论

三个分支的设计总体遵循了控制变量原则：它们均从同一个 main 提交 `d4fedcf` 创建，均未修改数据划分和类别编号，监督模块均保持与 `sup398` 基线相同的 SegFormer-B0、ImageNet 预训练、CE+Dice 主损失、AdamW、512 crop、40000 iterations、Validation checkpoint selection 和滑窗评估设置。

其中：

- `state-factorization` 是最干净的监督结构消融，适合第一个跑。
- `boundary-context` 也属于监督消融，但因为改变了推理图，需要额外报告推理成本，并拆开“boundary loss only”和“context refinement”。
- `structure-aware-pl` 属于半监督阶段，已具备基础实现，但必须等监督门禁通过后再正式训练，并且必须先建立 confidence-only EMA baseline。

## 2. 共同控制项

| 控制项 | 审查结果 |
|---|---|
| 共同起点 | 三个分支均从 `d4fedcf` 创建 |
| 数据划分 | 未修改 `splits/`，仍使用固定 sup398/full1445/ssl398_1047 入口 |
| 类别编号 | 未改变 FloodNet 10 类 class id |
| 主模型 | SegFormer-B0，`nvidia/mit-b0` ImageNet 预训练 |
| 主损失 | CE+Dice 保持不变 |
| 训练长度 | 40000 optimizer steps |
| crop | 512×512 |
| batch | batch size 2，gradient accumulation 4，有效 batch 8 |
| 优化器 | AdamW，lr 0.00006，weight decay 0.01 |
| 验证 | 每 2000 steps 在 Validation 450 上评估 |
| 测试 | Test 448 仅用于最终冻结配置后的 `evaluate.py` |
| 模块开关 | 新功能均通过 YAML 配置控制 |
| 原基线影响 | 原 `sup398` 和 `full1445` 配置不启用新增模块 |

## 3. State Factorization 审查

分支：`exp/state-factorization`
提交：`8332c34`

### 符合严谨性的点

- 只在 `sup398` 监督路径上增加 object/state 辅助目标，主语义 10 类输出仍保留。
- 标签分解完全由原 mask 派生，不引入外部标注。
- 非 building/road 像素的 state target 使用 ignore，不会错误监督状态。
- 模块权重较小，避免把主任务改造成状态二分类。
- 测试覆盖了标签映射、概率组合、损失可反传和关闭行为。

### 需要注意的点

- 当前辅助头从语义 logits 派生，是低侵入轻量实现。它适合第一轮控制变量实验，但不能宣称为完整 decoder-level 多任务结构。
- 消融要按 `object only`、`object+state`、`object+state+JS` 展开，否则无法判断收益来自哪个部分。
- 如果 mIoU 下降但 state macro-F1 上升，应判为 trade-off，而不是直接判为有效提升。

### 审查结论

该分支符合当前监督阶段控制变量要求，优先级最高。

## 4. Boundary Context 审查

分支：`exp/boundary-context`
提交：`dd88bff`

### 符合严谨性的点

- 边界标签由语义 mask 派生，没有改变原监督目标。
- 不仅增加 boundary loss，还实现了边界门控上下文 refinement，符合“不能只加普通 boundary loss”的要求。
- 训练配置与 `sup398` 监督基线一致。
- 模块可通过 YAML 关闭，原基线不受影响。
- 测试覆盖边界 target、loss、refinement shape 和关闭路径。

### 需要注意的点

- refinement 改变推理图，因此应报告额外推理成本。
- 必须设置 `boundary loss only` 对照，否则无法证明改进来自 boundary-context，而不是边界辅助监督。
- 边界 target width 与评估 tolerance 需要固定或成组消融，不能在不同实验中随意改变。

### 审查结论

该分支符合监督消融原则，但正式结果必须拆分 boundary loss 与 context refinement。优先级第二。

## 5. Structure-Aware PL 审查

分支：`exp/structure-aware-pl`
提交：`a39ce76`

### 符合严谨性的点

- 独立 `train_ssl.py`，不污染监督训练入口。
- 保持固定 labeled 398 与 unlabeled 1047 ID。
- teacher 采用 EMA，筛选策略包含 confidence、multiview、boundary、region 四类结构信号。
- 支持 matched coverage 思路，便于公平比较不同伪标签筛选策略。
- 测试覆盖 EMA、keep mask、结构分数和 dry-run 数据计数。

### 需要注意的点

- 该分支不应在当前监督比较完成前进入正式主实验。
- 必须先跑 EMA confidence-only baseline，再逐项加入结构分数。
- 多视图一致性需要确保 weak/strong 或双视图增强真正生效，不能只在同一 logits 上自比较。
- 使用 full manifest 定位 unlabeled 图像是可以的，但训练 loss 不得读取 1047 张 unlabeled 的 mask。
- hidden train labels 只能用于离线伪标签质量分析，不能用于阈值选择或 checkpoint 选择。

### 审查结论

该分支作为后续半监督 scaffold 合格，但实验严谨性依赖后续 baseline 和 matched coverage 执行。优先级第三。

## 6. 后续暂定实验顺序

当前已经完成的 `sup398` 与 `full1445` 是后续全部实验的坐标系。`sup398` 是低标注监督下界，`full1445` 是同模型、同训练协议下的全监督上界。当前 first-seed Test 结果为：

| Method | Test mIoU-10 | Test mIoU-9 | Test affected-mIoU | Test State Macro-F1 | Test Boundary F1 |
|---|---:|---:|---:|---:|---:|
| `sup398` | 47.67 | 52.77 | 34.34 | 68.99 | 16.36 |
| `full1445` | 52.74 | 57.75 | 38.10 | 80.49 | 16.85 |

后续方法不仅要报告相对 `sup398` 的提升，也应报告缩小了多少监督差距：

\[
\text{gap closed} = \frac{\text{method} - \text{sup398}}{\text{full1445} - \text{sup398}}
\]

暂定顺序：

1. 完成 `sup398` 与 `full1445` 监督训练：已完成 first-seed run。Validation 曲线正常，best checkpoint 只由 Validation 选择，Test 只最终评估。当前结果支持进入结构模块 pilot。
2. 做监督错误分析：比较 `sup398` 与 `full1445` 在 flooded building、flooded road、building/road 边界、水体混淆上的差异，明确后续结构模块要解决的主要错误类型。
3. 跑 `state-factorization` 监督消融：按 `object only`、`object + state`、`object + state + JS consistency` 展开。若只提升 state macro-F1 但不提升 affected mIoU 或主 mIoU，应视为诊断性收益而非主贡献。
4. 跑 `boundary-context` 监督消融：先跑 `boundary loss only`，再跑 `boundary loss + semantic-boundary consistency`，最后跑完整 `boundary context refinement`。必须区分收益来自边界辅助监督还是上下文建模。
5. 生成监督阶段可视化：包含 per-class IoU 差值柱状图、flooded/non-flooded 混淆矩阵、边界预测图、典型成功/失败案例。这一步用于判断论文机制是否成立，而不是只看总 mIoU。
6. 启动半监督前置审计：使用 `sup398` teacher 在 1047 张 unlabeled 图像上生成伪标签，先不训练 student，比较 confidence-only 与结构分数在 matched coverage 下的伪标签质量。隐藏 mask 只允许用于这个离线分析。
7. 跑 EMA confidence-only SSL baseline：这是半监督最小对照，必须先完成，不能直接用完整 structure-aware PL 对比纯监督 `sup398`。
8. 跑 structure-aware PL 消融：固定所有训练变量，逐项加入 region consistency、boundary stability、multiview consistency，最后跑完整 score，并在 20%、40%、60%、80% matched coverage 下比较。
9. 尝试结构模块组合：只有当 state 或 boundary 在监督阶段通过门禁时，才把对应结构信号接入 SSL；否则不强行组合，避免变量过多导致归因不清。
10. 三 seed 与最终 Test：只对单 seed 通过门禁的配置做三 seed 均值/标准差。所有超参和 checkpoint 选择固定后，再对 Test 448 做最终评估。

## 7. 当前阶段禁止事项

- 不在监督基线完成前报告 SSL 最终收益。
- 不用 Test 选择阈值、权重、coverage 或 checkpoint。
- 不跨协议比较旧 398/1047 challenge 数字与当前 supervised Validation/Test 数字。
- 不提交数据、权重、cache、TensorBoard log 或 `outputs/`。
- 不把 dry-run、unit test 或设计预期写成实验结果。
