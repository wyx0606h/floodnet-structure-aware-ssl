# FloodNet 实验交接状态

> 最后更新：2026-06-24  
> 状态：挑战版数据审计完成，准备本地工程实现
> 当前阶段：Week 1 / M1

## 1. 当前目标

先在本地完成合并解压、固定 278/60/60 划分、评测代码和四图像过拟合测试，再租用 GPU 运行 SegFormer-B0 Local Full Supervision 基线。

## 2. 已知事实

- 数据源位于 `F:\数据集\Flood`，由 Track 1 的 `001` 至 `007` 七个 ZIP 下载分包构成；
- 七个 ZIP 合计 398 张有掩码训练图像、1047 张无标签训练图像、450 张无公开掩码验证图像和 448 张无公开掩码测试图像；
- 398 张图像与掩码全部配对，无跨划分 ID 重叠；掩码为 4000×3000 单通道 PNG，类别值 0–9；
- 官方 Validation/Test 没有公开掩码，不能用于本地 mIoU；
- 本地核心协议改为从 398 张真值生成固定 278/60/60 划分；
- 目标 GPU 为单卡约 24GB；
- 使用 PyTorch 和 SegFormer-B0；
- 初始 crop 为 512×512；
- 主目标为 IGARSS 级会议论文；
- 采用有边界的自适应研究策略；
- AIFloodSense 仅在核心方法通过阶段门后加入；
- UrbanSARFloods 不进入本轮主实验。

## 3. 已完成

- [x] 数据集与相关论文调研；
- [x] 研究背景、方法和实验协议设计；
- [x] 八周执行计划；
- [x] 自适应决策树；
- [x] 实验 registry 和 decision log 初始化；
- [x] 初始化本地 Git 仓库并创建首个研究提交；
- [x] 添加 README、AGENTS.md 与数据/权重忽略规则；
- [x] 明确第一阶段核心实验仅基于 FloodNet；
- [x] 创建并上传私有远程仓库 `wyx0606h/floodnet-structure-aware-ssl`；
- [x] 只读审计七个 Track 1 ZIP 的目录、数量、配对、重叠和类别映射；
- [x] 确认 2021 半监督论文使用 DeepLabV3+/EfficientNet-B3 与 398/1047 协议；
- [x] 在 `E:\CodexProjects\floodnet-structure-aware-ssl` 建立并登记 Codex 本地项目；
- [x] 创建并置顶项目线程“FloodNet 半监督分割实验”；
- [x] 将挑战版协议更新推送至 GitHub 提交 `538ed01`。

## 4. 尚未完成

- [ ] 将七个 ZIP 合并解压至单一数据根目录；
- [ ] 统计 398 张掩码的分类别像素和图像覆盖；
- [ ] 生成固定 278/60/60 多标签分层划分并检查近重复泄漏；
- [ ] 建立训练与评测工程；
- [ ] 通过四图像过拟合测试；
- [ ] 租用 GPU 后跑通 Local Full Supervision 基线。

## 5. 下一步

1. 在项目线程中确认合并解压七个 ZIP 的目标路径；
2. 合并解压七个 ZIP 到仓库外的数据目录；
3. 生成数据统计和 278/60/60 split；
4. 实现 Dataset、指标、增强和滑窗推理；
5. 运行最小数据加载、单元测试与四图像过拟合；
6. 本地门通过后再租用 GPU；
7. 启动 SegFormer-B0 Local Full Supervision 基线。

## 6. 下一条命令

新项目线程应先读取 `AGENTS.md` 与本文件，然后建立本地工程骨架和数据划分脚本；在未通过四图像过拟合测试前不要启动付费长训练。

## 7. 当前阻塞项

- GPU、CUDA、PyTorch 版本尚未审计；
- 七个 ZIP 尚未合并解压；
- 278/60/60 split 尚未生成。

这些属于 Week 1 可发现事实，不需要在开始前向用户重复询问。

## 8. 当前阶段门

M1：数据与监督基线。通过条件详见 `floodnet_codex_8week_execution_plan.md`。

## 9. 最近实验

暂无。

## 10. 最近决定

- D001：SegFormer-B0 作为默认骨干；
- D002：FloodNet 为唯一核心数据集；
- D003：AIFloodSense 为条件性扩展；
- D004：UrbanSARFloods 不进入主实验；
- D005：关系模块失败时收缩为层次+边界方案；
- D013：第一阶段全部核心方法与基线先在 FloodNet 上完成；
- D014：Git 仓库中的状态文件是跨线程、跨机器研究连续性的事实来源；
- D016：挑战版官方 Validation/Test 无公开掩码，本地结论采用 278/60/60 固定划分；
- D017：本地代码、单元测试和四图像过拟合通过后才租用 GPU；
- D022：本地工程的规范项目根目录为 `E:\CodexProjects\floodnet-structure-aware-ssl`。
