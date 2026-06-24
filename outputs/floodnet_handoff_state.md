# FloodNet 实验交接状态

> 最后更新：2026-06-24  
> 状态：规划完成，尚未开始 Week 1 实验  
> 当前阶段：Pre-Week 1

## 1. 当前目标

在 2026-06-25 至 2026-07-01 完成 FloodNet 数据审计、评测代码和 SegFormer-B0 100% 监督基线。

## 2. 已知事实

- FloodNet 数据集已准备，但具体本地路径尚未在当前线程中确认；
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
- [x] 创建并上传私有远程仓库 `wyx0606h/floodnet-structure-aware-ssl`。

## 4. 尚未完成

- [ ] 定位 FloodNet 数据路径；
- [ ] 审计数据目录和 mask 编码；
- [ ] 确认是否已有实验代码仓库；
- [ ] 建立训练与评测工程；
- [ ] 跑通 100% 监督基线。

## 5. 下一步

1. 搜索当前 workspace 及用户明确提供的位置，定位 FloodNet 图像和掩码；
2. 只读检查数据目录、文件数量、划分文件和 mask 编码；
3. 检查是否已有可复用的训练工程；
4. 若无工程，在 `work/floodnet_project/` 建立独立实验仓库；
5. 创建 Week 1 数据审计报告；
6. 运行最小数据加载和指标测试；
7. 启动 SegFormer-B0 监督基线。

## 6. 下一条命令

尚未确定。新线程应先读取本文件，并以只读方式搜索 FloodNet 数据路径和已有代码。

## 7. 当前阻塞项

- FloodNet 的本地绝对路径未知；
- GPU、CUDA、PyTorch 版本尚未审计；
- 是否存在既有分割工程未知。

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
- D014：Git 仓库中的状态文件是跨线程、跨机器研究连续性的事实来源。
