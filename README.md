# FloodNet SegFormer-B0 Supervised Baselines

本仓库用于 FloodNet 洪灾无人机影像十类语义分割实验。当前阶段先建立统一、可复现的监督基线，比较两种训练标签规模：

- `sup398`：使用 Challenge 官方 398 张 labeled train 图像；
- `full1445`：使用完整监督版 1445 张 train 图像；

两组实验除训练标签数量和实验名外，保持模型、预训练权重、损失、优化器、增强、crop size、batch size、`max_iterations`、Validation/Test、指标、滑窗推理方式和 seed 一致。后续 `ssl398_1047` 会在同一协议结构上加入 1047 张 unlabeled、EMA teacher 和伪标签，本轮不实现完整半监督训练。

## 环境安装

建议在服务器上创建独立环境并安装 PyTorch、Transformers 等依赖。示例：

```bash
pip install torch torchvision transformers safetensors pyyaml pillow numpy
```

本地开发可先运行不建模型的 dry-run 或单元测试；真实 SegFormer-B0 训练需要 `transformers`。

## 数据路径配置

仓库不硬编码本地盘符。训练只需要完整监督版数据根；只有重新生成 split 时才需要 Challenge 数据根：

```bash
export SUPERVISED_ROOT=/path/to/FloodNet/FloodNet-Supervised_v1.0
export CHALLENGE_ROOT=/path/to/FloodNet-Challenge-Track1
```

Windows PowerShell 示例：

```powershell
$env:SUPERVISED_ROOT='D:\data\FloodNet\FloodNet-Supervised_v1.0'
$env:CHALLENGE_ROOT='D:\data\FloodNetChallenge\FloodNet Challenge @ EARTHVISION 2021 - Track 1'
```

`SUPERVISED_ROOT` 可以指向 `FloodNet-Supervised_v1.0` 或其父目录。若仓库已包含 `splits/`，服务器训练不需要上传 Challenge 版；`CHALLENGE_ROOT` 只在重新运行 `tools/build_floodnet_splits.py` 时需要。


## 预训练权重

当前配置使用 HuggingFace 模型名 `nvidia/mit-b0`，并设置 `local_files_only: false`。因此服务器联网时，第一次运行 smoke test 或正式训练会自动下载 SegFormer-B0 ImageNet 预训练权重到 HuggingFace 缓存。建议把缓存放在仓库外，避免误提交：

```bash
export HF_HOME=/data/cache/huggingface
export TRANSFORMERS_CACHE=/data/cache/huggingface/transformers
```

也可以在训练前手动预拉取一次：

```bash
python - <<'PY'
from transformers import SegformerForSemanticSegmentation

SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b0",
    num_labels=10,
    ignore_mismatched_sizes=True,
)
PY
```

如果服务器离线，请先在有网络的机器下载到本地目录，例如 `weights/nvidia_mit-b0/`，再手动上传到服务器，并把本地运行用配置中的：

```yaml
model:
  pretrained_model_name_or_path: weights/nvidia_mit-b0
  local_files_only: true
```

权重、checkpoint、缓存和训练输出不要提交到 Git。本仓库已在 `.gitignore` 中忽略 `weights/`、`*.bin`、`*.safetensors`、`*.pth`、`checkpoints/` 和 `outputs/*/` 等模型产物。

## 当前关键训练配置

两组监督实验统一使用：SegFormer-B0、`nvidia/mit-b0` ImageNet 预训练、CE+Dice、AdamW、poly schedule、1000 steps warmup、gradient clip 1.0、512 crop、AMP、滑窗推理 tile 512 / stride 384。第一版基础基线关闭类别感知裁剪，`drop_last: true` 保证有效 batch size 稳定为 8。

## 生成 split

先由两套数据生成固定名单和 manifest：

```bash
python tools/build_floodnet_splits.py \
  --supervised-root "$SUPERVISED_ROOT" \
  --challenge-root "$CHALLENGE_ROOT" \
  --output-dir splits
```

输出包括：

```text
splits/challenge_labeled_398.txt
splits/challenge_unlabeled_1047.txt
splits/full_train_1445.txt
splits/val_450.txt
splits/test_448.txt
splits/sup398_manifest.csv
splits/full1445_manifest.csv
```

脚本会检查 398/1047/1445/450/448 数量、labeled/unlabeled 无交集、二者并集等于完整 train、图像和 mask 存在。
> 若你没有上传 `splits/`，才需要先上传 Challenge 版并运行 split 生成命令；若 `splits/` 已随仓库上传，可直接跳过 split 生成。


## 训练命令

正式配置使用 40000 optimizer steps、每 2000 steps 做一次 validation；`max_iterations` 按 optimizer step 计数，不按 micro-batch 计数。有效 batch size 为 `batch_size 2 × gradient_accumulation_steps 4 = 8`。

服务器上先做 50–100 steps GPU smoke test。建议把输出写到临时目录，确认 forward/backward、loss、checkpoint、validation 都正常：

```bash
python train.py \
  --config configs/segformer_b0_sup398.yaml \
  --supervised-root "$SUPERVISED_ROOT" \
  --output-dir outputs/smoke_sup398 \
  --max-iterations 50 \
  --val-interval 50 \
  --max-eval-samples 8

python train.py \
  --config configs/segformer_b0_full1445.yaml \
  --supervised-root "$SUPERVISED_ROOT" \
  --output-dir outputs/smoke_full1445 \
  --max-iterations 50 \
  --val-interval 50 \
  --max-eval-samples 8
```

Smoke 通过后运行正式训练。

`sup398`：

```bash
python train.py \
  --config configs/segformer_b0_sup398.yaml \
  --supervised-root "$SUPERVISED_ROOT"
```

`full1445`：

```bash
python train.py \
  --config configs/segformer_b0_full1445.yaml \
  --supervised-root "$SUPERVISED_ROOT"
```

断点续训：

```bash
python train.py \
  --config configs/segformer_b0_sup398.yaml \
  --supervised-root "$SUPERVISED_ROOT" \
  --resume outputs/segformer_b0_sup398/checkpoints/last.pth
```

## Validation/Test 评估

Validation 用于选择 `best_miou.pth`，Test 只用于配置冻结后的最终评价。

```bash
python evaluate.py \
  --config configs/segformer_b0_sup398.yaml \
  --supervised-root "$SUPERVISED_ROOT" \
  --checkpoint outputs/segformer_b0_sup398/checkpoints/best_miou.pth \
  --split validation

python evaluate.py \
  --config configs/segformer_b0_sup398.yaml \
  --supervised-root "$SUPERVISED_ROOT" \
  --checkpoint outputs/segformer_b0_sup398/checkpoints/best_miou.pth \
  --split test
```

`full1445` 只需替换配置和 checkpoint 路径。

## 输出目录

每个实验保存到：

```text
outputs/<experiment_name>/
├── config_resolved.yaml
├── train.log
├── checkpoints/
│   ├── best_miou.pth
│   └── last.pth
├── metrics/
├── curves/
└── predictions/
```

指标包括 confusion matrix、Pixel Accuracy、每类 IoU、mIoU、每类 Precision/Recall/F1、macro-F1 和 Flooded-mIoU。

## 主要入口

```text
train.py
evaluate.py
tools/build_floodnet_splits.py
configs/
floodnet_ssl/
docs/
tests/
```

旧 `scripts/` 目录保留为审计、服务器检查和历史兼容工具；新训练/评估优先使用顶层统一入口。
