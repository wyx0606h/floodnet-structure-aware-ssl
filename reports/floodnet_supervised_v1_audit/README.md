# FloodNet-Supervised_v1.0 审计摘要

> 审计日期：2026-07-02  
> 数据位置：`F:\FloodNet\FloodNet-Supervised_v1.0`  
> 结论：该目录提供官方 supervised train/validation/test 分割 mask，可作为新的主实验协议。

## 1. 数量

| Split | 图像数 | Mask 数 | 配对数 |
|---|---:|---:|---:|
| Train | 1445 | 1445 | 1445 |
| Validation | 450 | 450 | 450 |
| Test | 448 | 448 | 448 |
| 合计 | 2343 | 2343 | 2343 |

三个 split 的 sample ID 无重叠，抽样检查图像与 mask 尺寸在样本内一致。

## 2. 目录结构

```text
F:\FloodNet\FloodNet-Supervised_v1.0\
├── train\
│   ├── train-org-img\
│   └── train-label-img\
├── val\
│   ├── val-org-img\
│   └── val-label-img\
└── test\
    ├── test-org-img\
    └── test-label-img\
```

另有彩色 mask 目录：

```text
F:\FloodNet\ColorMasks-FloodNetv1.0\
├── ColorMasks-TrainSet  # 1445
├── ColorMasks-ValSet    # 450
├── ColorMasks-TestSet   # 448
└── ColorPalette-Values.xlsx
```

训练代码使用 `FloodNet-Supervised_v1.0` 中的单通道 `*_lab.png`，彩色 mask 仅用于可视化/人工核对。

## 3. 新协议影响

旧 challenge release 只有 398 张公开 mask，因此曾建立 `278/60/60` local split。新数据提供官方 Validation/Test mask 后，主协议改为：

- Train：1445，用于训练和少标注模拟母集；
- Validation：450，用于 checkpoint 选择和调参；
- Test：448，用于配置冻结后的最终本地评估。

旧 `splits/local_278_60_60_v1/` 保留为历史审计产物，不再作为主实验协议。

## 4. 新产物

- 官方 supervised manifest：`splits/floodnet_supervised_v1/manifest.csv`
- 四图 overfit manifest：`splits/overfit4_supervised_v1/manifest.csv`
- 四图 crop 仍覆盖 0–9 十类，manifest SHA-256：`79e982b173f6d5cba3baced5478b8c90bdc63adf8029b789faa0d62b2af4c25e`
