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

- **FloodNet Challenge Track 1**：当前主数据版本。七个 ZIP 合计包含 398 张有掩码训练图像、1047 张无标签训练图像、450 张无公开掩码验证图像和 448 张无公开掩码测试图像；
- **AIFloodSense**：仅在 FloodNet 核心方法通过阶段门后，用于外部预训练或跨地区泛化验证；
- **UrbanSARFloods**：本轮不进入主实验，保留为未来跨模态研究方向。

数据集、模型权重和训练输出不提交到 Git 仓库。

当前本地评价从 398 张公开真值中建立固定的多标签分层划分：278 张训练、60 张验证、60 张测试。1047 张官方无标签图像仅进入训练池；官方 Validation/Test 因没有公开掩码，不用于本地 mIoU。

## 研究文档

- [`outputs/floodnet_idea_experiment_spec.md`](outputs/floodnet_idea_experiment_spec.md)：研究问题、方法、公式、实验协议、指标、消融和论文故事；
- [`outputs/floodnet_codex_8week_execution_plan.md`](outputs/floodnet_codex_8week_execution_plan.md)：2026年6月25日至8月19日的八周执行计划；
- [`outputs/floodnet_handoff_state.md`](outputs/floodnet_handoff_state.md)：当前阶段、最新实验、阻塞项和下一步；
- [`outputs/floodnet_experiment_registry.csv`](outputs/floodnet_experiment_registry.csv)：不可覆盖的实验注册表；
- [`outputs/floodnet_decision_log.md`](outputs/floodnet_decision_log.md)：阶段门和研究路线决策记录。
- [`outputs/floodnet_dataset_audit.md`](outputs/floodnet_dataset_audit.md)：当前挑战版压缩包、公开标签和本地评价协议审计。

## Codex 跨机器继续方式

在新电脑、VS Code Remote SSH 或租用 GPU 服务器中打开本仓库后，向 Codex 输入：

> 请先读取 AGENTS.md，然后依次读取 outputs/floodnet_handoff_state.md、outputs/floodnet_decision_log.md、outputs/floodnet_codex_8week_execution_plan.md 和 outputs/floodnet_idea_experiment_spec.md，从当前交接状态继续执行。每次会话结束前更新交接状态、实验注册表和决策日志。

仓库文件是跨机器研究连续性的事实来源，不依赖聊天记忆。

## 当前状态

当前处于 `Week 1 / M1`。下一阶段是在本地合并解压七个 ZIP，生成可复现的 278/60/60 划分，完成数据管线、指标单元测试和四图像过拟合测试。GPU 租用只用于通过本地检查后的正式训练。
