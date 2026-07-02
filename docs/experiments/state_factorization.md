# Object-State Factorization 实验设计

分支：`exp/state-factorization`
实现提交：`8332c34`
共同基线提交：`d4fedcf`

## 1. 研究问题

FloodNet 的 10 类语义标签不是完全扁平的类别体系。其中建筑和道路各自又被拆成 flooded 与 non-flooded 两种状态：

- `building-flooded`
- `building-non-flooded`
- `road-flooded`
- `road-non-flooded`

普通 SegFormer-B0 监督基线把这四个类当作互斥的独立类别学习。这样做的缺点是：模型需要分别学习“建筑外观”“道路外观”“是否被水淹”三类证据，但标签结构没有显式告诉模型这些类别之间的共享关系。对于 FloodNet 这种受灾区域类别不均衡、flooded building/road 像素较少的数据，扁平分类容易把“物体身份”和“受灾状态”纠缠在一起。

本实验要验证的问题是：

> 在不改变原始 10 类标签编号、不改变数据划分、不改变主语义输出的前提下，显式建模“物体身份”和“淹没状态”是否能改善建筑/道路及受灾类别的识别？

## 2. 改进了什么

该模块不是替换原有 10 类语义分割头，而是在原监督路径旁边增加两个可关闭的辅助预测：

1. 物体身份预测：把 flooded/non-flooded 的建筑合并为 `building`，把 flooded/non-flooded 的道路合并为 `road`，其他类别保持为各自的物体/地物类别。
2. 状态预测：只在建筑和道路像素上预测 `flooded` 或 `non-flooded`。
3. 一致性约束：用物体概率和状态概率重新组合出 10 类层次概率，并要求它与原始 10 类语义概率一致。

主语义分割损失仍然是原来的 CE+Dice。也就是说，最终评价仍然看官方 10 类 mask，不把任务改成新标签体系。

## 3. 标签分解

设原语义标签为 \(y \in \{0,\ldots,9\}\)。模块构造两个派生监督：

\[
y^{obj} = f_{obj}(y)
\]

\[
y^{state} = f_{state}(y)
\]

物体标签包括 background、building、road、water、tree、vehicle、pool、grass 等合并后的身份类。状态标签只对 building/road 像素有效：

\[
y^{state} \in \{\text{non-flooded}, \text{flooded}\}
\]

非 building/road 像素的状态标签设为 `IGNORE_INDEX`，不参与状态损失。

## 4. 模型与损失

模型输出保持一个主语义头：

\[
P^{sem} = \operatorname{softmax}(Z^{sem})
\]

并增加两个辅助头：

\[
P^{obj} = \operatorname{softmax}(Z^{obj}), \quad
P^{state} = \operatorname{softmax}(Z^{state})
\]

层次组合概率用乘法因子化。例如：

\[
P^{hier}(\text{building-flooded}) =
P^{obj}(\text{building})P^{state}(\text{flooded})
\]

\[
P^{hier}(\text{road-non-flooded}) =
P^{obj}(\text{road})P^{state}(\text{non-flooded})
\]

对于 water、tree、vehicle 等没有状态的类别，直接继承其物体身份概率。

总损失为：

\[
\mathcal{L} =
\mathcal{L}^{sem}_{CE+Dice}
+ \lambda_o \mathcal{L}^{obj}_{CE}
+ \lambda_s \mathcal{L}^{state}_{CE}
+ \lambda_h D_{JS}(P^{sem}, P^{hier})
\]

其中 \(D_{JS}\) 是主语义分布与层次组合分布之间的 Jensen-Shannon 一致性约束。

## 5. 当前代码实现

主要文件：

- `floodnet_ssl/state_factorization.py`：语义标签到物体/状态标签的映射、层次概率组合、JS 一致性损失。
- `floodnet_ssl/models.py`：`LogitAuxiliaryWrapper` 根据 YAML 配置挂载 object/state 辅助头。
- `floodnet_ssl/losses.py`：在 `supervised_objective` 中按配置合并主损失和辅助损失。
- `configs/segformer_b0_sup398_state_factorization.yaml`：模块实验入口。
- `tests/test_state_factorization.py`：覆盖标签映射、概率归一化、损失可反传、关闭模块后的零损失行为。

当前实现是轻量级版本：辅助头从语义 logit 特征派生，而不是直接接入 SegFormer 解码器中间特征。这能最大限度控制变量，优点是对主干训练流程侵入小；缺点是物体/状态分支的独立表征能力有限。若本版本出现正向信号，下一版可以再比较“decoder feature auxiliary heads”。

## 6. 配置

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

关闭方式：

- 原始 `configs/segformer_b0_sup398.yaml` 不包含该模块字段，因此监督基线不受影响。
- 在模块配置中把 `modules.state_factorization.enabled` 设为 `false`，辅助损失为 0。

## 7. 实验方案

必须先等待当前 `sup398` 和 `full1445` 两个监督实验完成，并确认 Validation/Test 流程正常，再启动该模块正式训练。

推荐实验：

1. `sup398` 基线：当前正在运行，用作直接对照。
2. `sup398 + object`：只开物体身份 CE。
3. `sup398 + object + state`：增加状态 CE。
4. `sup398 + object + state + JS`：完整模块。
5. 若单 seed 有收益，再做三 seed 均值和标准差。

评价指标：

- 主指标：Validation mIoU-10、mIoU-9、affected mIoU。
- 结构指标：building/road IoU、state macro-F1、flooded precision/recall。
- 诊断指标：JS consistency loss、object/state auxiliary loss。

## 8. 控制变量审查

已符合的控制项：

- 从共同 main 提交 `d4fedcf` 创建分支。
- 不修改数据划分、不修改类别编号。
- 使用同一 `sup398` manifest、同一 Validation 450、同一 Test 448。
- SegFormer-B0、ImageNet 预训练、CE+Dice、AdamW、crop 512、40000 iterations、val interval 2000、滑窗推理参数均与 `sup398` 基线一致。
- 模块通过 YAML 控制，原有基线配置不启用该模块。

需要在报告中明确的限制：

- 当前辅助头是 logit-level 轻量实现，不能夸大为完整多层解码器分支。
- 若只提升 state macro-F1 但降低 mIoU 或 affected mIoU，不能视为有效改进。
- 权重搜索只能用 Validation，Test 只能在配置冻结后最终评估一次。

## 9. 风险

状态监督只在建筑/道路像素上有效。若状态损失权重过大，模型可能过度关注 flooded/non-flooded 的二分类，损害 water、tree、grass 等普通类别。第一轮实验应保持小权重，并将 affected mIoU 与整体 mIoU 同时作为判断依据。
