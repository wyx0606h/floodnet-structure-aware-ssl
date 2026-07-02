# FloodNet 实验交接状态

> 最后更新：2026-07-02  
> 状态：已新增 sup398/full1445 统一监督实验入口；尚未训练，下一步为服务器环境真实 SegFormer-B0 smoke/训练  
> 当前阶段：Week 1 / M1

## 1. 当前目标

保留当前本地代码状态，后续在服务器或用户指定训练环境中安装 `transformers`，先运行随机初始化 SegFormer-B0 四图过拟合。四图门通过前，不下载正式预训练权重，不运行完整监督 baseline。

## 2. 已知事实

- 当前主数据源：`F:\FloodNet\FloodNet-Supervised_v1.0`；也可把环境变量 `FLOODNET_DATA_ROOT` 指向其父目录 `F:\FloodNet`；
- 官方 supervised split：Train 1445、Validation 450、Test 448，三者均有单通道分割 mask；
- 只读复审确认图像与 mask 一一配对，train/validation/test 无跨 split ID 重叠；
- 抽样检查确认 mask 为 `L` 模式，像素类别值位于 0–9；
- 部分图像尺寸为 4000×3000，部分为 4592×3072；数据集加载时以 mask 网格为真值坐标系对齐 RGB；
- 旧 `F:\数据集\Flood\FloodNet_Track1_Merged` challenge release、398-mask 与 278/60/60 split 仅保留为历史审计产物；
- 目标 GPU 为单卡约 24GB；默认模型 PyTorch + SegFormer-B0；初始 crop 为 512×512；
- AIFloodSense 仅在 FloodNet 核心方法通过阶段门后加入；UrbanSARFloods 不进入本轮主实验。

## 3. 已完成

- [x] 研究规格、八周计划、registry、decision log 和交接机制已建立；
- [x] D031 已记录：主数据协议升级为 `FloodNet-Supervised_v1.0` 官方 1445/450/448 split；
- [x] 新 supervised 数据只读审计通过，报告位于 `reports/floodnet_supervised_v1_audit/`；
- [x] 官方 supervised manifest 已冻结于 `splits/floodnet_supervised_v1/`；
- [x] manifest SHA-256：`fb3c0295bb7113923f8c0d9564bb52fe0155d612eeb26094221d92ed179799ba`；
- [x] 四图过拟合 manifest 已冻结于 `splits/overfit4_supervised_v1/`；
- [x] 四图 manifest SHA-256：`79e982b173f6d5cba3baced5478b8c90bdc63adf8029b789faa0d62b2af4c25e`；
- [x] Dataset、mIoU/F1/层次/边界指标、同步空间增强和滑窗概率融合已实现；
- [x] SegFormer-B0 adapter、`build_model`、统一模型输出和默认禁用的多头骨架已实现；
- [x] 监督训练脚本、checkpoint 评估脚本、run 汇总脚本和服务器只读环境检查脚本已实现；
- [x] 新 official train/validation/test CPU DataLoader smoke 已通过：`reports/week1_data_smoke_supervised_v1.json`；
- [x] 新四图 manifest 预检可读，当前仅阻塞于本地未安装 `transformers`：`reports/training_preflight_overfit4_supervised_v2.json`。
- [x] `sup398`/`full1445` 统一入口完成：`train.py`、`evaluate.py`、`tools/build_floodnet_splits.py`、两份 SegFormer-B0 配置与文档已更新。
- [x] 配置统一审查完成：scheduler/warmup/gradient clipping 已接入代码，两份 YAML 除允许字段外一致。

## 4. 尚未完成

- [ ] 在服务器或用户指定训练环境安装/准备 `transformers` 与 SegFormer 依赖；
- [ ] 执行四图过拟合训练门；
- [ ] 四图门通过后，训练 SegFormer-B0 full-supervision baseline；
- [ ] 对 Validation/Test 执行 checkpoint 滑窗评估；
- [ ] 将真实训练结果追加到 registry，并更新 handoff/decision log。

## 5. 下一步命令

本地或服务器先跑单元测试：

```powershell
& 'D:\Anaconda3\python.exe' -m unittest discover -s tests -v
```

只读预检，不训练：

```powershell
$env:FLOODNET_DATA_ROOT='F:\FloodNet'
& 'D:\Anaconda3\python.exe' scripts\check_training_prereqs.py `
  --config 'configs\overfit4_segformer_b0.yaml' `
  --output 'reports\training_preflight_overfit4_supervised_v2.json'
```

四图门仅在依赖齐全且用户确认允许写 `runs/` 后执行：

```powershell
$env:FLOODNET_DATA_ROOT='F:\FloodNet'
& 'D:\Anaconda3\python.exe' scripts\train_supervised.py `
  --config 'configs\overfit4_segformer_b0.yaml' `
  --execute `
  --confirm-run-id '20260702_overfit4_100_s20260702_segformer_b0'
```

## 6. 当前阻塞项

- 本地当前环境未安装 `transformers`，因此真实 SegFormer 构建/训练会被预检阻塞；
- 用户计划在服务器上配置环境并训练；本仓库当前不安装依赖、不训练、不写 F 盘原始数据；
- 未经确认不得执行会创建 `runs/` 的训练命令。

## 7. 最近实验/审计记录

- `20260702_data_audit_na_s0_supervised_v1`：完整 supervised 数据复审通过，确认 1445/450/448 全部配对且有 mask；
- `20260702_split_100_s0_supervised_official_v1`：官方 supervised manifest 生成；
- `20260702_data_smoke_100_s20260702_supervised_v1`：官方 train/validation/test CPU DataLoader smoke 通过；
- `20260702_week1_local_tests_na_s0_supervised_manifest_v1`：42 项单元测试通过，包含 supervised-v1 manifest 单测；
- `20260702_training_preflight_100_s20260702_supervised_overfit4_v2`：新版 run_id 预检可读，仍仅阻塞于 `transformers`。

## 8. 最近决定

- D029：Week 1 只实现网络骨架和扩展接口，不提前实现真实结构模块；
- D030：补齐服务器训练前的只读检查、评估和汇总闭环；
- D031：主数据协议升级为 `FloodNet-Supervised_v1.0` 官方 1445/450/448 split，旧 398-mask/278 split 只保留为历史产物。
- D032：新增公平对比协议 `sup398`/`full1445` 和统一入口；半监督仅预留，不在本轮实现。
- D033：统一两组监督配置为 40000 optimizer steps、poly warmup、有效 batch size 8、CE+Dice、关闭类别感知裁剪。

## 9. 2026-07-02 权重处理说明

- 用户取消本地下载预训练权重；本轮未创建 `weights/`，未下载 `nvidia/mit-b0`，未运行训练；
- `README.md` 已补充服务器端 HuggingFace 缓存路径、手动预拉取命令、离线权重目录示例和禁止提交权重/checkpoint 的说明；
- `.gitignore` 已补充 `*.bin`、`*.safetensors`，并继续忽略 `weights/`、checkpoint 和输出产物。

