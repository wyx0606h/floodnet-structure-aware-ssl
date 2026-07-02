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

### 待完成项

- 在服务器安装依赖后运行真实 SegFormer-B0 forward 和训练；
- 生成真实 split 文件并提交可审计名单；
- 运行 `sup398` 和 `full1445` 三种 seed 的正式实验；
- 后续实现 `ssl398_1047` 的 unlabeled loader、EMA Teacher 和伪标签训练。

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

### 实验结果占位表

| 实验 | Seed | Val mIoU | Val macro-F1 | Test mIoU | Test macro-F1 | Flooded-mIoU | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| `sup398` | 20260702 | TBD | TBD | TBD | TBD | TBD | 未训练 |
| `full1445` | 20260702 | TBD | TBD | TBD | TBD | TBD | 未训练 |

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

| 实验 | 类型 | Seed | Steps | Val mIoU | Test mIoU | 备注 |
|---|---|---:|---:|---:|---:|---|
| `sup398` | GPU smoke | 20260702 | 50–100 | TBD | - | 未运行 |
| `full1445` | GPU smoke | 20260702 | 50–100 | TBD | - | 未运行 |
| `sup398` | 正式训练 | 20260702 | 40000 | TBD | TBD | 未运行 |
| `full1445` | 正式训练 | 20260702 | 40000 | TBD | TBD | 未运行 |

