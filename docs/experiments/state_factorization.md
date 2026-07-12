# 条件物体-状态因子化实验

分支：`exp/state-factorization`

方法提交：`f67f4e3`

冻结配置提交：`76fbeb7`

状态：代码、测试和 S1-S4 配置已冻结并推送；尚无真实训练结果。

本文中的“假设”“预期”不是实验结论。main 只保存稳定协议与进度记录，实际方法代码和配置位于远程分支 `origin/exp/state-factorization`。

## 1. 研究问题

FloodNet 的四个关键类别同时编码物体身份和受灾状态：Building-flooded、Building-non-flooded、Road-flooded 和 Road-non-flooded。平坦十分类器没有显式共享同一物体的身份证据，也没有约束“受淹建筑仍属于建筑”。本实验检验：

> 在固定 398 张标注数据和统一 SegFormer-B0 协议下，多尺度物体身份建模与物体条件状态估计，能否改善建筑/道路及其受淹状态，并且其收益能否超过简单 logit 辅助头或额外参数带来的正则化。

## 2. 当前架构

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
                              Z_obj (8)     building/road state experts
                                   \             /
                    P_hier(y|x) = P(o|x) P(s|o,x)
                                   |
                 optional log-space fusion with P_sem
                                   |
                          final 10-class logits
```

原 SegFormer 语义分支保留。独立因子解码器读取四阶段 encoder 特征，经通道对齐、上采样和卷积融合得到 `F_factor`。物体分支预测八类身份：background、building、road、water、tree、vehicle、pool、grass。状态分支分别估计建筑和道路的 flooded/non-flooded 条件概率，非建筑/道路像素不参与状态监督。

十类层次概率按链式法则组合，例如：

\[
P(\text{building-flooded}|x)=P(\text{building}|x)
P(\text{flooded}|\text{building},x).
\]

完整 S4 可在对数概率空间融合平坦语义分支与层次分支；`fusion_weight: 0` 会保留原语义 logits，用于区分“训练期辅助正则化”与“结构参与最终推理”。

## 3. 训练目标

\[
\mathcal L = \mathcal L^{final}_{CE+Dice}
+\lambda_o\mathcal L^{obj}_{CE+Dice}
+\lambda_s\mathcal L^{state}_{CE+Dice}
+\lambda_c D_{JS}^{class}(P_{sem},P_{hier}).
\]

一致性损失使用类别均衡归约，避免 Grass 等大类支配梯度。训练历史分别记录 semantic、factorization、object、state 和 consistency 损失。代码已处理 crop 中不存在有效状态像素的情况，防止 all-ignore cross-entropy 产生 NaN。

## 4. 冻结消融

S0-S4 的数据、seed、SegFormer-B0 主配置、CE+Dice、优化器、调度、40000 optimizer steps、有效 batch 8、Validation/Test 和滑窗评估保持一致。允许变化的只有 `feature_source`、`state_mode` 和 `fusion_weight`。

| ID | 配置 | 目的 | 特征来源 | State | Fusion |
|---|---|---|---|---|---:|
| S0 | `configs/segformer_b0_sup398.yaml` | 已完成监督基线 | 无 | 无 | 0 |
| S1 | `configs/segformer_b0_sup398_state_s1_logit_shared.yaml` | 简单标签重编码/多任务控制 | semantic logits | shared | 0 |
| S2 | `configs/segformer_b0_sup398_state_s2_feature_shared.yaml` | 检验独立多尺度特征 | encoder multiscale | shared | 0 |
| S3 | `configs/segformer_b0_sup398_state_s3_feature_conditional.yaml` | 检验物体条件状态 | encoder multiscale | conditional | 0 |
| S4 | `configs/segformer_b0_sup398_state_factorization.yaml` | 检验结构参与推理 | encoder multiscale | conditional | 0.25 |

S5 参数量匹配的普通辅助头只在 S3/S4 通过 pilot 后实现。只有 S3 优于 S2，才能支持“条件状态优于共享状态”；只有结构配置优于参数量匹配控制，才能弱化“只是参数更多”的替代解释。

## 5. 评价与门禁

主指标：mIoU-10、mIoU-9、Affected mIoU、Building/Road grouped IoU、State Macro-F1，以及四个建筑/道路细类的 IoU、precision、recall。

判读规则：

- 只提高 State Macro-F1、但主 mIoU 或 Affected mIoU下降，只能报告为任务权衡或诊断性收益。
- S1 是控制实验，不是完整方法；不能用 S1 的结果直接宣称条件状态因子化有效。
- Test 不参与损失权重、融合系数、配置或 checkpoint 选择。
- 当前第一轮是固定 seed pilot；最终核心结论按项目规范补三 seed mean +/- std。

## 6. 服务器执行顺序

服务器先拉取远程分支：

```bash
git fetch origin
git switch exp/state-factorization
git pull --ff-only origin exp/state-factorization
```

先确认提交包含 `76fbeb7`，运行测试和 S1 dry-run：

```bash
git log -1 --oneline
python -m unittest discover -s tests -v
python train.py \
  --config configs/segformer_b0_sup398_state_s1_logit_shared.yaml \
  --supervised-root /path/to/FloodNet-Supervised_v1.0 \
  --dry-run
```

随后运行真实 SegFormer forward 和 50-100 step 小步 smoke，确认无维度、显存、NaN 和输出目录问题后，才启动 S1 的 40000-step 正式训练：

```bash
python train.py \
  --config configs/segformer_b0_sup398_state_s1_logit_shared.yaml \
  --supervised-root /path/to/FloodNet-Supervised_v1.0
```

S1 完成后先检查训练完整性和 Validation 曲线，再按冻结配置依次执行 S2 -> S3 -> S4。每个 run 使用独立 output directory，不覆盖 S0 或其他消融。Test 最好等 S1-S4 的 Validation 比较完成、配置冻结后统一评估，避免测试集反馈进入方法选择。

## 7. 当前证据边界

截至 2026-07-11，本分支通过语法检查、64 项单元测试和 S1-S4 dry-run；dry-run 确认 train=398、Validation=450。尚未在本地执行真实 HuggingFace SegFormer forward，也未产生任何 S1-S4 训练指标。因此现在可以说“实现和协议已准备好进入服务器 smoke/S1”，不能说“状态分解已经有效”。
