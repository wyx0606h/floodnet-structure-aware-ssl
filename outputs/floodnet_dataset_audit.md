# FloodNet Challenge Track 1 数据审计

> 审计日期：2026-06-24  
> 审计方式：只读读取 ZIP 目录、类别映射和样例掩码  
> 数据位置：`F:\数据集\Flood`

## 1. 数据包性质

文件名后缀为 `001` 至 `007` 的七个 ZIP 是同一份 Google Drive 大目录的下载分包，不是七个训练划分。不同 ZIP 中可能分别存放同一目录下的图像与掩码，因此必须合并解压到同一个目标目录后使用。

七个 ZIP 内部没有重复文件路径。

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
- 原始尺寸：4000×3000；
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

## 4. 对评价协议的影响

当前挑战版只公开 398 张分割真值。官方 Validation 和 Test 没有公开掩码，因此不能在本地计算 mIoU、Boundary F1 或分类别 IoU，也不能直接复现论文的官方测试分数。

本研究采用固定本地协议：

- Local Train：278；
- Local Validation：60；
- Local Test：60；
- 官方无标签训练池：1047。

划分脚本必须综合整图 Flooded/Non-Flooded 标签、十类出现情况和像素占比，并检查连续航拍近重复。Local Test 固定后不得参与训练、伪标签生成、阈值选择或模型选择。

## 5. 推荐数据目录

七个 ZIP 应合并解压至仓库外的独立目录，例如：

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

- 合并解压后再次核对文件数量和哈希；
- 统计每个类别的像素数与图像覆盖率；
- 生成样例可视化；
- 检查近重复和连续帧；
- 生成并提交 `splits/local_278_60_60_v1/`；
- 在四张图像上完成过拟合测试。
