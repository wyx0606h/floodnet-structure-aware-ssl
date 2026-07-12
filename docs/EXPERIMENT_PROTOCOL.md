# 实验协议

## 协议定义

### `sup398`

- 训练：Challenge 官方 398 labeled train ID；
- 图像/mask 实际从 `FloodNet-Supervised_v1.0` 读取；
- Validation：官方 supervised val 450；
- Test：官方 supervised test 448。

### `full1445`

- 训练：完整监督版 train 1445；
- Validation：官方 supervised val 450；
- Test：官方 supervised test 448；
- 作为当前监督上界。

### 后续 `ssl398_1047`

- labeled：Challenge 官方 398；
- unlabeled：Challenge 官方 1047；
- 将在同一代码结构中加入 EMA Teacher 和伪标签；
- 本轮不实现完整半监督训练。

## 公平比较原则

`sup398` 和 `full1445` 除训练标签数量外必须保持一致：

- SegFormer-B0；
- ImageNet pretrained 权重；
- CE + Dice loss；
- AdamW；
- 数据增强；
- crop size、batch size；
- `max_iterations`；
- seed；
- val/test；
- 滑窗推理策略；
- 评估指标。

## Validation/Test 使用规则

- Validation 450 用于定期评估、选择 `best_miou.pth` 和调参；
- Test 448 只在配置冻结后用于最终评价；
- 禁止用 Test 指标调参或选择 checkpoint。

## 统一指标

所有实验调用同一套评估代码，输出：

- confusion matrix；
- Pixel Accuracy；
- per-class IoU；
- mIoU；
- per-class Precision、Recall、F1；
- macro-F1；
- Flooded-mIoU = (Building Flooded IoU + Road Flooded IoU) / 2。

## 结果汇报规则

当前算力预算下，监督基线和结构模块 pilot 优先采用固定 seed 的单次结果进行横向比较。表格中直接报告单值，不写作 mean +/- std，也不把单次小幅提升描述为统计显著。论文或报告中应说明：

> Unless otherwise specified, all experiments are conducted under the same data split and training protocol with a fixed random seed for reproducibility.

若后续算力允许，只对关键 baseline、最强对比方法和最终完整方法补充多 seed mean +/- std。

## 当前 SegFormer-B0 统一训练配置

2026-07-02 统一 `configs/segformer_b0_sup398.yaml` 与 `configs/segformer_b0_full1445.yaml`，除实验名、protocol、训练 list/manifest 外，其余关键参数一致：

```yaml
training:
  max_iterations: 40000
  val_interval: 2000
  batch_size: 2
  gradient_accumulation_steps: 4
  optimizer: adamw
  learning_rate: 0.00006
  weight_decay: 0.01
  scheduler: poly
  warmup_iterations: 1000
  poly_power: 1.0
  gradient_clip_norm: 1.0
  use_amp: true
  device: cuda
```

有效 batch size 为 8。`max_iterations` 按 optimizer step 计数，不按 micro-batch 计数。训练流程不再单独设计 2000–5000 iteration pilot；只保留 50–100 iteration GPU smoke test，smoke 通过后直接执行 40000 optimizer steps 正式训练。每 2000 optimizer steps 在 Validation 上评估并按 Validation mIoU 保存 `best_miou.pth`。

### 配置理由

- SegFormer 官方常用 AdamW + poly schedule；FloodNet 数据量较小但图像分辨率高，因此使用 B0、512 crop、有效 batch size 8 和 40000 optimizer steps，避免机械照搬大数据集的超长训练。
- `local_files_only: false`：允许新服务器自动下载 `nvidia/mit-b0`；若服务器离线，应改为本地权重目录并记录。
- `class_aware_probability: 0.0`：第一版基础基线关闭类别感知裁剪，避免在监督基线中额外引入采样变量。
- `drop_last: true`：保证每个 optimizer step 都由 4 个 batch size=2 的 micro-batch 累积而成，维持有效 batch size 8。
- FloodNet Background 类别 0 是有效类别；当前代码不使用 SegFormer image processor 的 label reduction，Dataset 直接保留 0–9 mask。
- mask resize 使用最近邻，图像 resize 使用双线性。

### 延长训练规则

如果 `sup398` 与 `full1445` 在 40000 steps 末期都仍明显上升，可统一延长 `max_iterations`。不得只延长其中一组。

## 固定 seed pilot 的报告规则

- 当前 `sup398`、`full1445` 和第一轮结构消融使用 seed `20260702` 的单次运行值，表格直接报告单值。
- 不得把单次结果写成 average、mean、mean +/- std、稳定提升或显著提升。
- 所有可比 pilot 固定 split、seed、训练长度、优化器、Validation checkpoint selection 和 Test policy。
- 最终核心方法与关键 baseline 按项目规范补三个 seed，并报告 mean +/- std；若因算力未完成，论文需明确写固定 seed 单次运行的限制。

## 状态分解 S1-S4 评估规则

- 实际代码位于 `exp/state-factorization`，冻结配置提交为 `76fbeb7`。
- 执行顺序固定为 S1 logit-shared -> S2 feature-shared -> S3 feature-conditional -> S4 feature-conditional-fusion。
- 长训练前必须通过真实模型 forward 和 50-100 step smoke；smoke 结果不能作为方法性能结果。
- Validation 用于 checkpoint 选择与配置判断；为减少测试集反馈，S1-S4 完成 Validation 比较并冻结方案后再统一运行 Test。
- `full1445` 仅是监督参考上界；状态分解第一轮全部使用 `sup398`，无需运行 `full1445 + state-factorization`。

