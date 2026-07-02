# Boundary Context Modeling 实验设计

分支：`exp/boundary-context`
实现提交：`dd88bff`
共同基线提交：`d4fedcf`

## 1. 研究问题

FloodNet 中的 flooded road、road-non-flooded、building-flooded 等类别对边界非常敏感。道路细长、建筑轮廓复杂，洪水区域又经常贴着道路或建筑边缘出现。普通 CE+Dice 主要优化区域重叠，容易出现两类问题：

- 区域内部预测较稳定，但边界错位导致 building/road 与 water 混淆。
- 边界附近像素置信度很高但类别错误，后续伪标签会把这些错误放大。

简单增加一个 boundary loss 只能让模型“知道哪里有边界”，但不一定让语义预测真正利用边界信息。本实验因此把边界预测作为上下文建模的门控信号，让它参与语义 logit 的局部聚合。

研究问题是：

> 在保持 SegFormer-B0、训练协议和官方 10 类评价不变的前提下，边界引导的上下文建模是否能改善 FloodNet 的薄结构、物体边缘和受灾类别分割？

## 2. 改进了什么

该模块包含三部分：

1. 从语义 mask 生成边界监督。
2. 增加边界预测头，并用 BCE+Dice 学习边界。
3. 用预测边界作为门控，对语义 logit 做边界感知的局部上下文增强。

关键点是第三部分：边界不仅是辅助损失，而是实际改变语义预测的上下文聚合方式。

## 3. 边界监督生成

给定语义 mask \(y\)，先找相邻像素类别变化的位置：

\[
E_0(p)=\mathbb{1}[\exists q \in \mathcal{N}_4(p), y_p \neq y_q]
\]

再按配置宽度 \(w\) 做膨胀：

\[
E_{gt}=\operatorname{Dilate}_w(E_0)
\]

这样得到的边界标签不改变原 mask，只作为派生监督。当前默认 `target_width: 3`，与评估中的 `boundary_tolerance: 3` 保持一致。

## 4. 边界引导上下文建模

模型输出语义 logits \(Z\) 和边界 logits \(B\)。边界概率为：

\[
G=\sigma(B)
\]

局部上下文为：

\[
C=\operatorname{AvgPool}_k(Z)
\]

用边界门控进行 refinement：

\[
Z' = Z + \alpha(1-G)(C-Z)
\]

含义是：

- 在非边界区域，\(G\) 小，模型更多融合邻域上下文，提高区域一致性。
- 在边界区域，\(G\) 大，模型减少平滑，避免把道路、建筑、水体边缘抹掉。

这与只加 boundary loss 的区别在于：边界预测直接调节语义 logit 的空间传播方式。

## 5. 损失函数

边界损失为：

\[
\mathcal{L}_{bd} =
\lambda_{bce}\mathcal{L}_{BCE}(B,E_{gt})
+ \lambda_{dice}\mathcal{L}_{Dice}(B,E_{gt})
+ \lambda_c \lVert \sigma(B)-\widetilde{E}(Z') \rVert_1
\]

其中 \(\widetilde{E}(Z')\) 是从语义概率中计算的 soft edge map。最后一项要求边界预测与语义概率变化位置一致，减少“边界头学到了边界，但语义头不响应”的情况。

总损失为：

\[
\mathcal{L} =
\mathcal{L}^{sem}_{CE+Dice} + \mathcal{L}_{bd}
\]

## 6. 当前代码实现

主要文件：

- `floodnet_ssl/boundary_context.py`：边界 target 生成、soft semantic edge、BCE+Dice 边界损失、边界引导 refinement。
- `floodnet_ssl/models.py`：`BoundaryContextWrapper` 按配置挂载边界头，并输出 refined logits。
- `floodnet_ssl/losses.py`：在 `supervised_objective` 中按配置加入边界损失。
- `configs/segformer_b0_sup398_boundary_context.yaml`：模块实验入口。
- `tests/test_boundary_context.py`：覆盖边界 target、损失可反传、refinement shape 和关闭模块行为。

当前实现同样是轻量版，边界头从语义输出侧派生。它足以验证“边界作为上下文门控是否有价值”，但还不是完整的多尺度边界网络。若该模块有效，下一阶段可比较接入 shallow feature 的边界分支。

## 7. 配置

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
    max_pos_weight: 20.0
```

关闭方式：

- 原监督基线配置没有 `model.boundary_context` 和 `modules.boundary_context` 字段。
- 将 `enabled` 设为 `false` 后，模型和损失回到原语义路径。

## 8. 实验方案

必须以完成的 `sup398` 监督结果为直接对照。推荐消融顺序：

1. `sup398` 基线。
2. `boundary loss only`：边界头 + BCE+Dice，不启用 context refinement。
3. `boundary loss + semantic-boundary consistency`。
4. `boundary context full`：启用边界门控上下文增强。
5. 若 full 有收益，再比较 `kernel_size` 3/5/7 和 `target_width` 1/3/5。

评价指标：

- 主指标：Validation mIoU-10、mIoU-9、affected mIoU。
- 边界指标：mean boundary F1、building/road 相关边界表现。
- 风险指标：flooded building IoU、flooded road IoU 是否下降。
- 成本指标：训练/推理耗时变化，因为该模块改变了推理图。

## 9. 控制变量审查

已符合的控制项：

- 从共同 main 提交 `d4fedcf` 创建分支。
- 不修改类别编号、不修改 split。
- 使用 `sup398` 同一训练列表、Validation/Test 同一官方 mask。
- SegFormer-B0、预训练权重、CE+Dice 主损失、AdamW、crop 512、40000 iterations、val interval 2000、滑窗推理参数与 `sup398` 基线一致。
- 模块由 YAML 控制，关闭后不影响监督基线。

需要补充约束：

- 正式报告必须区分“只加边界损失”和“边界上下文建模”，否则无法证明收益来自结构建模。
- 因为 refinement 会改变推理图，需报告推理成本，不能只报告精度。
- 如果 Boundary F1 提升但 affected mIoU 下降，应判定为不通过主贡献门禁。

## 10. 风险

边界像素非常稀疏，BCE 容易被负样本主导，因此当前配置加入 Dice 和 `max_pos_weight`。但如果边界监督过强，模型可能过度锐化，导致内部区域一致性下降。第一轮应优先观察“边界变好是否同时带来 affected mIoU 或 building/road IoU 改善”。
