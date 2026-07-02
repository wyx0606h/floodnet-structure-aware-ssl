# FloodNet Challenge Track 1 数据审计

> 审计日期：2026-06-24 至 2026-06-25
> 审计方式：ZIP dry-run、受控合并解压、全量图像/掩码哈希与像素统计
> 数据位置：`F:\数据集\Flood\FloodNet_Track1_Merged`

## 1. 数据包性质

文件名后缀为 `001` 至 `007` 的七个 ZIP 是同一份 Google Drive 大目录的下载分包，不是七个训练划分。不同 ZIP 中可能分别存放同一目录下的图像与掩码，因此必须合并解压到同一个目标目录后使用。

七个 ZIP 内部没有重复文件路径。

源目录中还存在一个不属于本研究当前主线的 Track 2 ZIP。合并工具不得使用“读取目录中全部 ZIP”的宽泛规则，必须严格匹配同一批次 Track 1 的 `001` 至 `007`。

2026-06-24 使用本地安全工具再次进行只读 dry-run，确认：

- 严格匹配到 7 个 Track 1 ZIP；
- 合计 2742 个文件条目，即 2343 JPG、398 PNG 和 1 个 CSV；
- 预计展开大小为 12.063 GiB；
- 合并计划 SHA-256 为 `dbccc39ca636e79668006415abeab70b69a3b89314070a4b3fc643604cc44546`；
- 未创建目标目录，未解压或修改任何原始 ZIP。

2026-06-25 经用户明确批准后完成合并解压：

- 2742 个文件全部新写入，0 个跳过；
- 展开字节数为 12,952,903,612；
- 解压文件与合并 manifest 均设为只读；
- 原始七个 ZIP 的大小、修改时间与属性保持不变。

## 2. 合并后的公开内容

| 目录 | 图像数 | 掩码数 |
|---|---:|---:|
| Train/Labeled/Flooded | 51 | 51 |
| Train/Labeled/Non-Flooded | 347 | 347 |
| Train/Unlabeled | 1047 | 0 |
| Validation | 450 | 0 |
| Test | 448 | 0 |
| 合计 | 2343 | 398 |

398 张有标签图像均存在对应掩码，没有图像缺掩码或掩码缺图像。Train/Labeled、Train/Unlabeled、Validation 和 Test 的文件 ID 没有交集。

## 3. 文件与标签格式

- 图像：JPG；
- 掩码：单通道 PNG；
- 掩码文件名为 `ID_lab.png`，与 `ID.jpg` 配对时需移除末尾 `_lab`；
- 339 张有标签 JPG 为 4000×3000，59 张为 4592×3072；398 张 mask 均为 4000×3000；
- 对 59 张异尺寸样本，以 mask 网格为真值坐标系，在 Dataset 加载时将 RGB 双线性缩放到 4000×3000，不预先重采样 mask；
- 掩码类别值：0–9；
- 类别数：10。

| ID | 类别 |
|---:|---|
| 0 | Background |
| 1 | Building-flooded |
| 2 | Building-non-flooded |
| 3 | Road-flooded |
| 4 | Road-non-flooded |
| 5 | Water |
| 6 | Tree |
| 7 | Vehicle |
| 8 | Pool |
| 9 | Grass |

目录名 `Flooded/Non-Flooded` 是整图场景分类标签；每张 PNG 掩码仍可包含多个语义类别。

### 3.1 全量类别统计

| ID | 类别 | 像素占比 | 出现图像数 |
|---:|---|---:|---:|
| 0 | Background | 1.2983% | 29 |
| 1 | Building-flooded | 1.5616% | 36 |
| 2 | Building-non-flooded | 2.9035% | 142 |
| 3 | Road-flooded | 1.8543% | 37 |
| 4 | Road-non-flooded | 5.1204% | 194 |
| 5 | Water | 12.0441% | 187 |
| 6 | Tree | 16.1297% | 311 |
| 7 | Vehicle | 0.1776% | 126 |
| 8 | Pool | 0.1877% | 88 |
| 9 | Grass | 56.1864% | 368 |

完整整数像素数和精确比例见 `reports/floodnet_track1_audit_v1/class_statistics.csv`。

### 3.2 重复与连续帧

- 全部 2343 张 JPG 的 SHA-256 未发现精确重复；
- 感知哈希阈值 6/128 产生 3 个涉及有标签图像的候选；
- `7049/7050` 为连续近相同帧，固定在同一 local subset；
- `7081/7082/7083` 为同一牧场道路的连续偏移帧；`7082` 属于官方无标签池，因此 `7081`、`7083` 强制进入 Local Train；
- 审查结论保存在 `reports/floodnet_track1_audit_v1/near_duplicate_candidates.csv`。

## 4. 对评价协议的影响

当前挑战版只公开 398 张分割真值。官方 Validation 和 Test 没有公开掩码，因此不能在本地计算 mIoU、Boundary F1 或分类别 IoU，也不能直接复现论文的官方测试分数。

本研究采用固定本地协议：

- Local Train：278；
- Local Validation：60；
- Local Test：60；
- 官方无标签训练池：1047。

划分脚本必须综合整图 Flooded/Non-Flooded 标签、十类出现情况和像素占比，并检查连续航拍近重复。Local Test 固定后不得参与训练、伪标签生成、阈值选择或模型选择。

canonical v1 已生成于 `splits/local_278_60_60_v1/`：

| 子集 | 总数 | Flooded 场景 | Non-Flooded 场景 | 含 Building-flooded | 含 Road-flooded |
|---|---:|---:|---:|---:|---:|
| Local Train | 278 | 35 | 243 | 24 | 25 |
| Local Validation | 60 | 8 | 52 | 6 | 6 |
| Local Test | 60 | 8 | 52 | 6 | 6 |

- 三个集合 ID 无交集；
- `7049/7050` 均位于 Local Validation；
- `7081/7083` 均位于 Local Train；
- `manifest.csv` SHA-256：`4bed3f9acccbf40572e3f5d10e9292f4598816e4dd97668abf0fe13580c96f4b`；
- `train.csv` SHA-256：`b589ec6a3551d50a9b7688b2e9d31e8ab875ab6c5d896ddcd4432e5f648e04db`；
- `validation.csv` SHA-256：`8c965ab3e5d6a30bb4b884180e5f76c0a1f7720afdc2828092d2913645cfe458`；
- `test.csv` SHA-256：`3490f30b6e91f14d1921eac7c014f1dbc682bab2d10e314f0062d3fd15ff97df`。

## 5. 推荐数据目录

七个 ZIP 已合并解压至仓库外的独立目录：

```text
F:\数据集\Flood\FloodNet_Track1_Merged\
└── FloodNet Challenge @ EARTHVISION 2021 - Track 1\
    ├── class_mapping.csv
    ├── Train\
    │   ├── Labeled\
    │   │   ├── Flooded\
    │   │   │   ├── image\
    │   │   │   └── mask\
    │   │   └── Non-Flooded\
    │   │       ├── image\
    │   │       └── mask\
    │   └── Unlabeled\
    │       └── image\
    ├── Validation\
    │   └── image\
    └── Test\
        └── image\
```

原始 ZIP 保留不动，实验代码通过 `DATA_ROOT` 读取合并目录。

## 6. 下一步检查

- 生成样例可视化；
- 保留已通过的 39 个数据、增强、指标、滑窗、训练闭环、网络骨架和服务器脚本单元测试；
- 保留 `reports/week1_data_smoke_v1.json` 作为三个 local subset 的 CPU 加载证据；
- 在四张图像上完成过拟合测试。

---

## 7. 2026-07-02 完整 supervised 数据复审

用户提供的新数据根目录为 `F:\FloodNet`，其中 `FloodNet-Supervised_v1.0` 提供官方 supervised split，并且 Train、Validation、Test 均有单通道分割 mask：

| Split | 图像数 | Mask 数 | 当前用途 |
|---|---:|---:|---|
| Train | 1445 | 1445 | 主训练集与少标注模拟母集 |
| Validation | 450 | 450 | checkpoint 选择与调参 |
| Test | 448 | 448 | 配置冻结后的最终本地评估 |

只读审计确认：图像与 `*_lab.png` mask 一一配对，三个 split 的 sample ID 无重叠；抽样检查 mask 为 `L` 模式，类别值为 0–9 整数。另有 `ColorMasks-FloodNetv1.0` 彩色 mask 目录和 `ColorPalette-Values.xlsx`，仅作为可视化/人工核对资料，训练代码使用 `FloodNet-Supervised_v1.0` 的单通道 label mask。

新主协议产物：

- `reports/floodnet_supervised_v1_audit/`
- `splits/floodnet_supervised_v1/manifest.csv`
- `splits/overfit4_supervised_v1/manifest.csv`
- `reports/week1_data_smoke_supervised_v1.json`
- `reports/training_preflight_overfit4_supervised_v2.json`

因此，旧 `398` 张公开 mask challenge release 与 `278/60/60` local split 仅保留为历史审计产物；后续主实验不得再把旧协议结果与新 supervised 协议结果直接比较。