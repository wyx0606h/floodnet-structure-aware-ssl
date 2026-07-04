# 实验日志

## 2026-07-02 代码协议更新

### 本次修改

- 新增统一入口：`train.py`、`evaluate.py`；
- 新增 split 构建入口：`tools/build_floodnet_splits.py`；
- 新增协议构建模块：`floodnet_ssl/protocols.py`；
- 新增 CE+Dice loss：`floodnet_ssl/losses.py`；
- 新增共享实验工具：`floodnet_ssl/experiment.py`；
- 增强 mask 读取：支持二维/RGB/RGBA 类别索引图，拒绝未映射彩色 mask；
- 指标补齐 precision、recall、Flooded-mIoU；
- 新增配置：`configs/segformer_b0_sup398.yaml`、`configs/segformer_b0_full1445.yaml`。

### 当前完成项

- `sup398`/`full1445` 配置结构已统一；
- split 构建逻辑会检查 Challenge 398/1047 与 full train 1445 的集合关系；
- 训练使用固定 `max_iterations`，按 Validation mIoU 保存 `best_miou.pth` 和 `last.pth`；
- 评估统一使用滑窗推理和同一套 metrics。

### 当前完成项更新

- `sup398` 与 `full1445` 的 first-seed 正式监督训练已经完成；
- 两个 run 均训练到 40000 optimizer steps，并按 Validation mIoU 选择 `best_miou.pth`；
- Test 448 仅在 best checkpoint 固定后评估；
- 详细结果与分析记录在 `docs/experiments/supervised_comparison_results.md`。

### 待完成项

- 后续结构模块实验先以固定 seed 进行 pilot 对照；
- 优先运行 `state-factorization`，再运行 `boundary-context`；
- 启动半监督前必须先补齐 EMA confidence-only baseline；
- 若算力允许，仅对核心最终方法和关键 baseline 补充多 seed 统计。

### 配置文件

| 协议 | 配置 |
|---|---|
| `sup398` | `configs/segformer_b0_sup398.yaml` |
| `full1445` | `configs/segformer_b0_full1445.yaml` |

### 运行命令

```bash
python tools/build_floodnet_splits.py --supervised-root "$SUPERVISED_ROOT" --challenge-root "$CHALLENGE_ROOT" --output-dir splits
python train.py --config configs/segformer_b0_sup398.yaml --supervised-root "$SUPERVISED_ROOT"
python train.py --config configs/segformer_b0_full1445.yaml --supervised-root "$SUPERVISED_ROOT"
python evaluate.py --config configs/segformer_b0_sup398.yaml --supervised-root "$SUPERVISED_ROOT" --checkpoint outputs/segformer_b0_sup398/checkpoints/best_miou.pth --split validation
python evaluate.py --config configs/segformer_b0_sup398.yaml --supervised-root "$SUPERVISED_ROOT" --checkpoint outputs/segformer_b0_sup398/checkpoints/best_miou.pth --split test
```

### 实验结果摘要

Validation 只用于 checkpoint 选择；论文或报告中优先使用 `test_best` 指标。

| 实验 | Seed | Best Val Iter | Val mIoU-10 | Val mIoU-9 | Test mIoU-10 | Test mIoU-9 | Test macro-F1 | Test affected-mIoU | 备注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `sup398` | 20260702 | 18000 | 49.88 | 55.31 | 47.67 | 52.77 | 60.89 | 34.34 | completed |
| `full1445` | 20260702 | 32000 | 56.73 | 61.61 | 52.74 | 57.75 | 66.00 | 38.10 | completed |

## 2026-07-02 配置统一审查

### 修改内容

- `segformer_b0_sup398.yaml` 与 `segformer_b0_full1445.yaml` 保持除实验名、protocol、训练 list/manifest 外一致；
- `val_interval` 从 1000 改为 2000；
- 新增并接入 `scheduler: poly`、`warmup_iterations: 1000`、`poly_power: 1.0`；
- 新增并接入 `gradient_clip_norm: 1.0`；
- `class_aware_probability` 从 0.5 改为 0.0；
- `drop_last` 从 false 改为 true；
- `local_files_only` 从 true 改为 false，便于新服务器下载 `nvidia/mit-b0`；
- `train.py` 已按 optimizer step 统计 `max_iterations`，并在每个 optimizer step 上执行 warmup/poly lr、gradient clipping 和 checkpoint 保存。

### 当前阶段

只开展两组监督实验：

1. `sup398`：Challenge 官方 398 labeled；
2. `full1445`：完整监督版 1445 labeled，全监督上界。

不再单独设置 2000–5000 iteration pilot。服务器上只进行 50–100 optimizer steps GPU smoke test；通过后直接运行 40000 optimizer steps 正式训练。

### 结果记录

| 实验 | 类型 | Seed | Steps | Best Val mIoU-10 | Test mIoU-10 | Test mIoU-9 | Test affected-mIoU | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `sup398` | 正式训练 | 20260702 | 40000 | 49.88 | 47.67 | 52.77 | 34.34 | best iter 18000 |
| `full1445` | 正式训练 | 20260702 | 40000 | 56.73 | 52.74 | 57.75 | 38.10 | best iter 32000 |

## 2026-07-04 监督对照结果分析

### 结论

`full1445` 在 Test mIoU-9、Macro-F1 和 Affected mIoU 上均优于 `sup398`，说明统一监督协议下的全监督上界方向合理。`sup398` 可作为后续少标注监督模块的固定基线，`full1445` 可作为同骨干、同训练协议下的监督上界参考。

### 关键观察

- Test mIoU-9 从 52.77 提升到 57.75，提升 4.98 点；
- Test affected-mIoU 从 34.34 提升到 38.10，提升 3.77 点；
- grouped Building IoU 从 42.93 提升到 52.75；
- grouped Road IoU 从 43.53 提升到 50.52；
- State Macro-F1 从 68.99 提升到 80.49；
- Boundary F1 仅从 16.36 提升到 16.85，说明后续 boundary 模块仍有独立验证空间；
- Background IoU 在两组中都偏低，报告时应同时给出 mIoU-10 与 mIoU-9，并在分析中说明 Background 类的异质性和混淆问题。

### 后续执行

监督比较门禁已通过，可以进入结构模块 pilot：

1. `exp/state-factorization`；
2. `exp/boundary-context`；
3. EMA confidence-only SSL baseline；
4. structure-aware pseudo-label filtering。

