# FloodNet Structure-Aware Semi-Supervised Segmentation

面向少标注洪灾无人机影像的结构感知半监督语义分割研究。

## 当前研究主线

第一阶段实验**仅以 FloodNet 为核心数据集**，验证以下问题：

1. 将物体身份与洪水状态进行层次分解，能否改善受淹建筑和受淹道路；
2. 边界监督能否改善建筑、道路轮廓；
3. 物体内部与边界外部上下文能否改善受淹状态判断；
4. 层次、边界与关系一致性能否比单纯置信度更可靠地筛选伪标签。

默认骨干为 SegFormer-B0，使用 PyTorch，在单卡约 24 GB 显存条件下开展实验。

## 数据集范围

- **FloodNet**：主训练、验证与测试数据集，承载全部核心结论；
- **AIFloodSense**：仅在 FloodNet 核心方法通过阶段门后，用于外部预训练或跨地区泛化验证；
- **UrbanSARFloods**：本轮不进入主实验，保留为未来跨模态研究方向。

数据集、模型权重和训练输出不提交到 Git 仓库。

## 研究文档

- [`outputs/floodnet_idea_experiment_spec.md`](outputs/floodnet_idea_experiment_spec.md)：研究问题、方法、公式、实验协议、指标、消融和论文故事；
- [`outputs/floodnet_codex_8week_execution_plan.md`](outputs/floodnet_codex_8week_execution_plan.md)：2026年6月25日至8月19日的八周执行计划；
- [`outputs/floodnet_handoff_state.md`](outputs/floodnet_handoff_state.md)：当前阶段、最新实验、阻塞项和下一步；
- [`outputs/floodnet_experiment_registry.csv`](outputs/floodnet_experiment_registry.csv)：不可覆盖的实验注册表；
- [`outputs/floodnet_decision_log.md`](outputs/floodnet_decision_log.md)：阶段门和研究路线决策记录。

## Codex 跨机器继续方式

在新电脑、VS Code Remote SSH 或租用 GPU 服务器中打开本仓库后，向 Codex 输入：

> 请先读取 AGENTS.md，然后依次读取 outputs/floodnet_handoff_state.md、outputs/floodnet_decision_log.md、outputs/floodnet_codex_8week_execution_plan.md 和 outputs/floodnet_idea_experiment_spec.md，从当前交接状态继续执行。每次会话结束前更新交接状态、实验注册表和决策日志。

仓库文件是跨机器研究连续性的事实来源，不依赖聊天记忆。

## 当前状态

当前处于 `Pre-Week 1`。下一阶段是审计 FloodNet 本地目录、标签映射、官方划分和类别统计，并建立 SegFormer-B0 100% 监督基线。
