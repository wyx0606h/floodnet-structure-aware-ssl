# Structure-Aware Pseudo-Label Filtering 实验设计

分支：`exp/structure-aware-pl`
实现提交：`a39ce76`
共同基线提交：`d4fedcf`

## 1. 研究问题

FloodNet 的半监督场景使用 398 张有标签图像和 1047 张未标注训练图像。常见伪标签方法会用 teacher 的最大 softmax 概率筛选像素，但在 FloodNet 中，仅用置信度有明显风险：

- flooded building/road 是少数类，高置信度不一定代表结构正确。
- 道路和水体边界附近容易出现高置信度错分。
- 大块区域可能局部置信度高，但整体形状不稳定。

因此，本实验要验证：

> 在基础 EMA teacher-student 框架上，结合置信度、多视图一致性、边界稳定性和区域一致性的结构感知筛选，是否比 confidence-only 伪标签更可靠？

注意：根据当前项目阶段门禁，正式半监督训练必须在 `sup398` 与 `full1445` 监督比较完成后再启动。


## 2. 半监督在本项目中的定义

本项目中的半监督不是 `full1445` 全监督训练，也不是把 1047 张训练图像的 mask 加入监督 loss。它特指 `ssl398_1047` 协议：

- 有标签集合：398 张图像及其官方 mask，参与正常的 CE+Dice 监督损失。
- 无标签集合：1047 张图像，只使用 RGB 图像，不在训练 loss 中读取其 mask。
- Validation/Test：仍然使用官方 supervised Validation 450 和 Test 448；Validation 选 checkpoint，Test 只做最终评估。

因此，训练目标由两部分组成：

\[
\mathcal{L} = \mathcal{L}_{sup}(x_l, y_l) + \lambda_u \mathcal{L}_{unsup}(x_u, \hat{y}_u, M)
\]

其中 \((x_l,y_l)\) 来自 398 张 labeled 数据，\(x_u\) 来自 1047 张 unlabeled 数据，\(\hat{y}_u\) 是 teacher 生成的伪标签，\(M\) 是保留可靠伪标签像素的 mask。

1047 张图像在当前本地 supervised 数据中可能存在 mask 文件，但在半监督训练中这些 mask 必须视为隐藏标签：只能用于离线伪标签质量分析，例如计算 pseudo-label mIoU、coverage-precision 曲线；不能用于训练、阈值选择、checkpoint 选择或调参。

## 3. 改进了什么

该模块实现两个层次：

1. 基础 EMA teacher-student 半监督入口。
2. 结构感知伪标签筛选策略。

基础 teacher-student 只负责产生伪标签并训练 student。结构感知策略负责决定哪些像素可以进入无标签损失。

## 4. EMA Teacher-Student

student 参数为 \(\theta_S\)，teacher 参数为 \(\theta_T\)。teacher 由 student 的指数滑动平均更新：

\[
\theta_T \leftarrow \alpha\theta_T + (1-\alpha)\theta_S
\]

teacher 对未标注图像产生概率：

\[
P_T = \operatorname{softmax}(Z_T)
\]

伪标签为：

\[
\hat{y} = \arg\max_c P_T(c)
\]

student 的总损失为：

\[
\mathcal{L} =
\mathcal{L}^{labeled}_{CE+Dice}
+ \lambda_u \mathcal{L}^{unlabeled}_{CE}(\hat{y}, M)
\]

其中 \(M\) 是结构感知保留 mask。

## 5. 结构感知筛选

每个像素的可靠性分数由四类信号组成：

\[
s =
\frac{
w_c s_c + w_m s_m + w_b s_b + w_r s_r
}{
w_c+w_m+w_b+w_r
}
\]

### 5.1 置信度 \(s_c\)

最大类别概率：

\[
s_c = \max_c P_T(c)
\]

这是 confidence-only baseline 的核心。结构感知筛选必须与它做公平比较。

### 5.2 多视图一致性 \(s_m\)

对同一未标注样本的两个 teacher 视图，比较类别概率分布一致性。当前代码以概率点积作为一致性得分：

\[
s_m = \sum_c P_T^{(1)}(c)P_T^{(2)}(c)
\]

如果两个视图在同一像素附近给出相近分布，说明预测更稳定。

### 5.3 边界稳定性 \(s_b\)

从 teacher 概率图中计算 soft semantic edge。若两个视图的边界响应差异小，则边界稳定性高。直觉是：边界位置不稳定的像素更可能是伪标签噪声源。

### 5.4 区域一致性 \(s_r\)

在局部窗口内统计邻域像素与中心伪标签一致的比例：

\[
s_r(p) =
\frac{1}{|\Omega_p|}
\sum_{q \in \Omega_p}
\mathbb{1}[\hat{y}_q=\hat{y}_p]
\]

孤立小块、边界毛刺和散点伪标签会被降低分数，大块稳定区域会被保留。

## 6. 阈值与 matched coverage

结构感知筛选支持两种方式：

1. 绝对阈值：保留 \(s \ge \tau\) 的像素。
2. matched coverage：固定保留比例，例如 20%、40%、60%、80%。

正式比较必须报告 matched coverage，因为不同筛选策略如果保留像素数量不同，直接比较伪标签 mIoU 会不公平。推荐把 confidence-only 和 structure-aware 放在同一 coverage 下比较伪标签质量。

## 7. 当前代码实现

主要文件：

- `floodnet_ssl/pseudolabels.py`：EMA 更新、结构分数、matched coverage、伪标签 mask。
- `train_ssl.py`：`ssl398_1047` 的 EMA teacher-student 训练入口。
- `configs/segformer_b0_ssl398_1047_structure_pl.yaml`：半监督配置。
- `tests/test_pseudolabels.py`：覆盖 EMA、keep mask、score composition、region consistency、dry-run 数据计数。

当前实现是可运行 scaffold。它已经具备 teacher-student、伪标签筛选和 dry-run 入口，但正式实验前仍需补齐或确认：

- weak/strong augmentation 是否与监督训练增强公平对齐。
- Mean Teacher 或 FixMatch-style confidence-only baseline 是否先跑通。
- Validation checkpoint 选择与 `evaluate.py` 最终 Test 评估是否完全接入。

## 8. 配置

```yaml
ssl:
  ema_decay: 0.999
  threshold: 0.8
  matched_coverage:
  unsupervised_weight: 1.0
  confidence_weight: 1.0
  multiview_weight: 0.25
  boundary_weight: 0.25
  region_weight: 0.25
  region_kernel_size: 5
```

数据仍使用固定 split：

- labeled：`splits/challenge_labeled_398.txt`
- unlabeled：`splits/challenge_unlabeled_1047.txt`
- validation：`splits/val_450.txt`
- test：`splits/test_448.txt`

虽然 `unlabeled_manifest` 可以从 full manifest 找到图像路径，训练时不得使用这 1047 张图像的 mask 参与 loss。隐藏标签只允许用于离线伪标签质量分析，并且必须明确标注为 analysis，不得用于调参选择。

## 9. 实验方案

必须在监督比较完成后执行。暂定展开顺序如下：

1. 完成 `sup398` 与 `full1445`：确认两条监督曲线正常收敛，固定 Validation checkpoint selection 和最终 Test 流程。`full1445` 作为全监督上界，用于衡量后续方法缩小了多少 `sup398` 到全监督之间的差距。
2. 做 `sup398` teacher 的伪标签审计：先不训练 student，只在 1047 张 unlabeled 图像上生成伪标签，并用隐藏 mask 离线分析 confidence、boundary stability、region consistency 与 matched coverage 下的 pseudo-label 质量。
3. 跑 EMA confidence-only baseline：只用 \(s_c\) 过滤伪标签，建立 SSL 的最小公平基线。
4. 逐项加入结构信号：`confidence + region`、`confidence + boundary`、`confidence + multiview`，再跑完整 structure-aware score。每一步固定 split、训练步数、增强、batch、EMA decay、无标签损失权重。
5. 在 matched coverage 20%、40%、60%、80% 下比较 confidence-only 与 structure-aware pseudo-label filtering，避免“保留像素数量不同”导致不公平。
6. 若 state factorization 或 boundary context 在监督阶段通过门禁，再尝试把对应结构信号接入 SSL teacher/student；若监督阶段无效，则不强行融合。
7. 单 seed 确认有效后，再做三 seed 均值和标准差，并只在最终冻结配置后评估 Test。

评价指标：

- 最终分割：Validation mIoU-10、mIoU-9、affected mIoU、state macro-F1。
- 伪标签质量：pseudo-label mIoU、per-class precision/recall、coverage。
- 公平性诊断：matched coverage 曲线。
- 训练稳定性：teacher/student loss、保留像素比例、各类别伪标签覆盖率。

## 10. 控制变量审查

已符合的控制项：

- 从共同 main 提交 `d4fedcf` 创建分支。
- 不修改 split、不修改类别编号。
- 保持 SegFormer-B0、预训练、crop 512、AdamW、40000 iterations、val interval 2000、滑窗评估参数。
- 不改原监督 `train.py` 主路径，SSL 通过独立 `train_ssl.py` 进入。

需要严格补充的控制项：

- 必须先有 confidence-only EMA baseline，不能直接拿结构感知结果和纯监督 `sup398` 比。
- 结构筛选和 confidence-only 要在 matched coverage 下比较。
- 弱/强增强策略、batch 组成、无标签 loss 权重 schedule 必须在所有 SSL 消融中固定。
- Test 只能在最终冻结配置后评估，不能用于选择阈值、coverage 或权重。
- 当前两个监督实验正在运行，SSL 正式训练应等待监督门禁完成。

## 11. 风险

该分支风险最大，因为它同时引入 teacher-student 训练、无标签数据和筛选策略。若没有先跑通 confidence-only EMA，对完整结构感知方法的收益无法归因。当前阶段应把它视为“已实现基础入口和筛选机制”，而不是已经可以直接宣称半监督最终提升。
