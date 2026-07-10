# 条件物体-状态因子化实验

分支：`exp/state-factorization`

状态：方法与代码协议已冻结，尚未产生训练结果。本文中的“预期”“假设”均不是实验结论。

## 1. 研究问题

FloodNet 将物体身份与受灾状态压缩在一个平坦的十分类标签中：

- `Building-flooded` / `Building-non-flooded` 共享“建筑”身份；
- `Road-flooded` / `Road-non-flooded` 共享“道路”身份；
- flooded 状态在建筑和道路上的外观并不相同。

平坦分类器需要分别学习四个类别，无法显式共享建筑/道路定位证据，也没有约束“受淹建筑仍然必须是建筑”。本实验检验的核心假设是：

> 在 398 张标注数据下，先学习稳定的物体身份，再以物体身份为条件估计受淹状态，可以降低建筑/道路定位与状态识别之间的样本复杂度；这种收益应当超过普通辅助头或额外参数带来的正则化收益。

## 2. 文献依据与设计吸收

1. [FloodNet](https://arxiv.org/abs/2012.02951) 明确指出受淹建筑、受淹道路以及自然水体之间的区分是数据集的核心困难，因此状态建模应围绕建筑/道路，而不能把 `Water` 当作完整洪水真值。
2. [Deep Hierarchical Semantic Segmentation, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Deep_Hierarchical_Semantic_Segmentation_CVPR_2022_paper.html) 表明层级标签可用于约束像素表示和预测一致性。本实验吸收“父类/子类一致性”，但不照搬一般树形层级。
3. [Learning Conditional Attributes for Compositional Zero-Shot Learning, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Learning_Conditional_Attributes_for_Compositional_Zero-Shot_Learning_CVPR_2023_paper.html) 指出同一属性在不同物体上的视觉表现不同，属性应以物体和图像为条件。对应到 FloodNet，建筑受淹与道路受淹不能只共享一个无条件二分类器。
4. [Flattening the Parent Bias, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Weber_Flattening_the_Parent_Bias_Hierarchical_Semantic_Segmentation_in_the_Poincare_CVPR_2024_paper.html) 证明部分层级分割增益可能并非来自层级本身，并指出平坦分类器在跨域条件下可能更稳健。因此本方法保留原平坦语义分支，并强制加入“普通多任务/额外参数”对照。
5. [A Conditional Probability Framework for Compositional Zero-shot Learning, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wu_A_Conditional_Probability_Framework_for_Compositional_Zero-shot_Learning_ICCV_2025_paper.html) 使用概率链式分解 (P(o,a\mid x)=P(o\mid x)P(a\mid o,x))。本实验把该思想改写为像素级受灾语义分割，不引入文本编码器或零样本设定。
6. [PAD-Net, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Xu_PAD-Net_Multi-Tasks_Guided_CVPR_2018_paper.html) 与 [MTI-Net, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/2449_ECCV_2020_paper.php) 说明中间任务预测应来自共享特征并与多尺度信息交互，而不是只从最终 logits 做线性变换。本方法据此使用四阶段 SegFormer 编码特征构建独立的轻量因子解码器。

这些工作提供的是设计依据，不意味着本模块已经达到相同贡献强度，也不能替代 FloodNet 上的消融证据。

## 3. 原实现审查

初版实现具备标签映射、object/state CE 和 JS 一致性，适合作为低侵入原型，但存在四项论文级风险：

1. object/state 头接在十类语义 logits 上，仅为 `1x1 Conv`，可能退化成固定标签重编码，无法证明学习了独立的物体与状态表示。
2. 全局两类 state 头隐含 (P(s\mid x))，没有建模 (P(s\mid o,x))，忽略建筑和道路的状态外观差异。
3. 推理只使用平坦语义输出，因子化分支只在训练中正则化，方法贡献很难在架构图和推理机制上形成闭环。
4. 若某个 crop 不含建筑或道路，全部为 `ignore_index` 的 state cross-entropy 可能产生 NaN。初版单元测试没有覆盖该情况。

本次升级保留初版作为 `logit-head` 消融概念，但完整方法改为条件概率因子化。

## 4. 方法架构

```text
RGB image
   |
SegFormer-B0 encoder: E1, E2, E3, E4
   |                              \
original semantic decoder         multi-scale factor decoder
   |                               |
Z_sem (10 classes)                 F_factor
                                   |       \
                              object head   object-conditioned state head
                                   |             |
                              Z_obj (8)     Z_state (2 x 2 experts)
                                   \             /
                    P_hier(y|x) = P(o|x) P(s|o,x)
                                   |
             log-space fusion with P_sem (config-gated)
                                   |
                          final 10-class logits
```

原 SegFormer 语义解码器不做结构修改。因子解码器读取四个编码阶段，经 `1x1 Conv` 对齐通道、双线性上采样到最高分辨率，再通过 `3x3 Conv + BN + GELU` 融合。这样既获得多尺度证据，又不把主语义头替换成另一套解码器。

### 4.1 物体身份分支

十类标签映射为八类物体身份：

\[
\{background, building, road, water, tree, vehicle, pool, grass\}.
\]

映射固定为：

```text
[0, 1, 1, 2, 2, 3, 4, 5, 6, 7]
```

物体分支输出 (P_o(o\mid x))，监督使用 CE+Dice。Dice 仅对当前 batch 中出现的类别求平均，避免不存在类别造成无意义梯度。

### 4.2 物体条件状态专家

完整模型输出四个状态通道：

```text
[building-non-flooded, building-flooded,
 road-non-flooded, road-flooded]
```

先从 object posterior 中取 building/road 两个通道，经投影后加到状态特征：

\[
F_s = F_{factor} + \phi([P_o(building),P_o(road)]).
\]

建筑状态专家估计 (P(s\mid o=building,x))，道路状态专家估计 (P(s\mid o=road,x))。两种专家分别计算 CE+Dice 后平均，避免道路像素数量直接淹没建筑状态损失。非建筑/道路像素继续使用 `ignore_index=255`。

配置仍支持 `state_mode: shared`，用于复现无条件两通道 state 头这一关键消融。

### 4.3 十类组合概率

建筑与道路细类由概率链式法则重建：

\[
P_h(building,flooded\mid x)=P_o(building\mid x)
P_s(flooded\mid building,x),
\]

\[
P_h(road,nonflooded\mid x)=P_o(road\mid x)
P_s(nonflooded\mid road,x).
\]

其余六类直接继承 object posterior。十个通道严格归一化为 1，不进行不透明的后处理。

### 4.4 平坦-因子化融合

完整模型以对数空间 product-of-experts 融合两条路径：

\[
\log \tilde P(y\mid x)=(1-\alpha)\log P_{sem}(y\mid x)
+\alpha\log P_h(y\mid x).
\]

默认 (alpha=0.25)，目的是让成熟的平坦分支保持主导，同时让结构分支参与最终推理。`fusion_weight: 0.0` 会精确返回原语义 logits，用于“只做辅助正则化”的控制实验。

### 4.5 训练目标

\[
\mathcal L = \mathcal L^{final}_{CE+Dice}
+\lambda_o\mathcal L^{obj}_{CE+Dice}
+\lambda_s\mathcal L^{state}_{expert\ CE+Dice}
+\lambda_c D_{JS}^{class}(P_{sem},P_h).
\]

JS 默认按真值类别先分别求均值、再跨出现类别平均，防止 Grass 等大类支配一致性损失。训练历史额外记录 semantic、factorization、object、state 和 consistency 五项损失，便于判断模块是否发生塌缩或负迁移。

## 5. 配置

S1-S4 均已冻结为独立 YAML。完整 S4 配置位于 `configs/segformer_b0_sup398_state_factorization.yaml`：

```yaml
model:
  auxiliary_heads: [object, state]
  state_factorization:
    enabled: true
    feature_source: encoder_multiscale
    decoder_channels: 64
    dropout: 0.1
    state_mode: conditional
    detach_object_posterior: false
    fusion_weight: 0.25

modules:
  state_factorization:
    enabled: true
    object_weight: 0.25
    state_weight: 0.25
    consistency_weight: 0.1
    object_dice_weight: 1.0
    state_dice_weight: 1.0
    consistency_reduction: class_mean
```

原 `segformer_b0_sup398.yaml` 与 `segformer_b0_full1445.yaml` 不包含开关，关闭模块时仍走原监督基线路径。

| ID | 配置文件 | 唯一改动轴 |
|---|---|---|
| S0 | `configs/segformer_b0_sup398.yaml` | 无模块基线 |
| S1 | `configs/segformer_b0_sup398_state_s1_logit_shared.yaml` | logits 特征、共享状态、无融合 |
| S2 | `configs/segformer_b0_sup398_state_s2_feature_shared.yaml` | 多尺度特征、共享状态、无融合 |
| S3 | `configs/segformer_b0_sup398_state_s3_feature_conditional.yaml` | 多尺度特征、条件状态、无融合 |
| S4 | `configs/segformer_b0_sup398_state_factorization.yaml` | 多尺度特征、条件状态、融合 0.25 |

配置校验会拒绝 `logits + conditional` 或 `logits + fusion`，防止把一个没有条件特征和组合推理能力的对照误写成完整方法。

单元测试逐字段确认 S1-S4 与 S0 的 `dataset`、`data`、主损失、训练策略、评估策略、seed 以及 SegFormer-B0 主配置完全一致；四个实验的辅助损失权重也完全一致。允许变化的只有实验标识/输出目录以及 `feature_source`、`state_mode`、`fusion_weight` 三个预注册方法轴。

## 6. 严格消融与控制变量

第一轮固定 seed `20260702`、398 标签、SegFormer-B0、ImageNet 权重、CE+Dice、AdamW、crop 512、有效 batch 8、40000 optimizer steps、Validation 选 checkpoint 和同一滑窗 Test。只改变下列方法字段：

| ID | 目的 | 特征来源 | state | fusion |
|---|---|---|---|---:|
| S0 | 监督基线 | 无 | 无 | 0 |
| S1 | 排除标签重编码收益 | semantic logits | shared | 0 |
| S2 | 检验独立多尺度特征 | encoder multiscale | shared | 0 |
| S3 | 检验条件状态建模 | encoder multiscale | conditional | 0 |
| S4 | 检验结构参与推理 | encoder multiscale | conditional | 0.25 |
| S5 | 排除普通参数增益（最终主张门禁） | encoder multiscale | 参数量匹配的普通辅助 10 类头 | 0 |

在 S3/S4 之间再分别关闭 object loss、state loss 和 JS，确认最终收益不是某一个普通辅助损失单独造成。S0-S4 已有独立 YAML；S5 只在 S3/S4 通过 pilot 后实现并冻结，避免在核心假设尚未成立时扩张实验。只有 S3 显著优于 S2，才能支持“状态需要以物体为条件”的机制主张；只有 S2/S3 优于 S5，才能弱化“只是参数更多”的替代解释。

超参只允许在 Validation 上选择。第一轮不同时扫描 loss weight、decoder width、fusion weight 和 dropout；建议先固定 loss 与 width，仅比较 `fusion_weight in {0, 0.25}`。若 pilot 有效，再在 Validation 上做最小范围敏感性分析，并对冻结配置运行三个 seed。

## 7. 指标与诊断

主指标：

- mIoU-10 与 mIoU-9；
- affected mIoU；
- Building grouped IoU、Road grouped IoU；
- State Macro-F1；
- 四个 building/road 细类的 IoU、precision、recall。

机制诊断：

- flat semantic 与 hierarchical prediction 的分歧率；
- building/road object IoU；
- building 和 road 各自的 flooded/non-flooded confusion matrix；
- (alpha=0) 与 (alpha=0.25) 的 calibration/ECE；
- 参数量、FLOPs 和滑窗推理时间；
- 成功案例与失败案例，重点看 Water 与 Road-flooded 混淆。

如果只提升 State Macro-F1 而 mIoU/affected mIoU 下降，应报告为任务权衡，不能写成整体提升。Test 不参与权重、融合系数或 checkpoint 的选择。

## 8. 运行入口

本地/服务器预检以 S1 为第一项：

```powershell
python train.py `
  --config configs/segformer_b0_sup398_state_s1_logit_shared.yaml `
  --supervised-root F:\FloodNet `
  --dry-run
```

Linux 服务器按 S1 -> S2 -> S3 -> S4 依次训练，每次先运行 `--dry-run`：

```bash
python train.py \
  --config configs/segformer_b0_sup398_state_s1_logit_shared.yaml \
  --supervised-root /path/to/FloodNet-Supervised_v1.0

python train.py \
  --config configs/segformer_b0_sup398_state_s2_feature_shared.yaml \
  --supervised-root /path/to/FloodNet-Supervised_v1.0

python train.py \
  --config configs/segformer_b0_sup398_state_s3_feature_conditional.yaml \
  --supervised-root /path/to/FloodNet-Supervised_v1.0

python train.py \
  --config configs/segformer_b0_sup398_state_factorization.yaml \
  --supervised-root /path/to/FloodNet-Supervised_v1.0
```

正式长训练前仍需通过四图像过拟合/小步 smoke，并使用新的 output directory，不能覆盖监督基线产物。

## 9. 论文价值与证据边界

相较初版，新设计已经具备可画架构图、可写概率公式、可形成机制消融的完整方法闭环。潜在贡献可概括为“面向灾后遥感分割的像素级条件物体-状态组合建模”，而不是泛化地声称发明了层级分割或条件概率。

但代码完整不等于论文贡献成立。至少满足以下条件后，才适合把它作为核心模块：

1. S3/S4 在三个 seed 上稳定提升 affected mIoU 与 State Macro-F1，且主 mIoU 不出现不可接受退化；
2. 条件状态优于共享状态，feature head 优于 logit head；
3. 参数量匹配控制不能解释全部收益；
4. 定性结果显示改进集中在建筑/道路身份与状态混淆，而非随机类别波动；
5. 不使用 Test 调参，最终结果报告 mean±std 和计算开销。

主要风险包括条件分支错误放大、稀有 flooded 状态过拟合、辅助任务负迁移，以及单一 FloodNet 数据集的外部有效性有限。若条件专家没有优于共享 state，论文叙事应收缩为多尺度层级监督，不应保留未经证实的条件概率主张。
