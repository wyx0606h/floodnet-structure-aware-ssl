# FloodNet 服务器训练快速说明

目的：到服务器后尽快完成 Week 1 四图过拟合门、SegFormer-B0 全监督 baseline、Validation/Test 评估和 run 汇总。本文只给命令顺序，不改变数据协议。

## 1. 上传/准备内容

- Git 仓库代码；
- `FloodNet-Supervised_v1.0` 数据根目录，或包含它的父目录；
- `splits/floodnet_supervised_v1/`；
- `splits/overfit4_supervised_v1/`；
- `configs/overfit4_segformer_b0.yaml`；
- `configs/supervised_segformer_b0.yaml`。

服务器上的数据路径用环境变量提供，不写死到代码里：

```bash
export FLOODNET_DATA_ROOT=/path/to/FloodNet
```

`FLOODNET_DATA_ROOT` 可以指向 `/path/to/FloodNet`，也可以直接指向 `/path/to/FloodNet/FloodNet-Supervised_v1.0`。

## 2. 只读环境检查

先检查依赖、CUDA、manifest 和数据路径，不训练：

```bash
python scripts/check_server_environment.py \
  --config configs/overfit4_segformer_b0.yaml \
  --output reports/server_check_overfit4_supervised_v1.json
```

正式监督 baseline 前检查 CUDA 和预训练权重缓存：

```bash
python scripts/check_server_environment.py \
  --config configs/supervised_segformer_b0.yaml \
  --require-cuda \
  --require-pretrained-cache \
  --output reports/server_check_supervised_v1.json
```

如果 `supervised_segformer_b0.yaml` 使用 `local_files_only: true` 和 `pretrained_model_name_or_path: nvidia/mit-b0`，需要提前在服务器 Hugging Face 缓存中准备权重，或把配置改为本地权重目录。

## 3. 四图过拟合门

先 dry-run，确认 run ID 和输出目录：

```bash
python scripts/train_supervised.py \
  --config configs/overfit4_segformer_b0.yaml
```

确认无误后执行训练：

```bash
python scripts/train_supervised.py \
  --config configs/overfit4_segformer_b0.yaml \
  --execute \
  --confirm-run-id 20260702_overfit4_100_s20260702_segformer_b0
```

通过条件写在 `runs/.../overfit_gate.json`：

- final train loss / initial train loss ≤ 0.20；
- final train mIoU-10 ≥ 0.90。

四图门未通过前，不运行完整监督 baseline。

## 4. Full Supervision baseline

四图门通过后先 dry-run：

```bash
python scripts/train_supervised.py \
  --config configs/supervised_segformer_b0.yaml
```

执行正式训练：

```bash
python scripts/train_supervised.py \
  --config configs/supervised_segformer_b0.yaml \
  --execute \
  --confirm-run-id 20260702_supervised_100_s20260702_segformer_b0
```

训练输出目录会包含：

- `resolved_config.json`；
- `runtime_metadata.json`；
- `history.csv`；
- `best.pt`；
- `last.pt`；
- `train_summary.json`。

## 5. Checkpoint 评估

新版 supervised 协议下，Validation 和 Test 都有 mask，可以本地定量评估。训练时用 Validation 选 checkpoint；最终报告可在配置冻结后给出 Test 指标。

```bash
python scripts/evaluate_checkpoint.py \
  --config configs/supervised_segformer_b0.yaml \
  --checkpoint runs/20260702_supervised_100_s20260702_segformer_b0/best.pt \
  --split validation \
  --output-dir runs/20260702_supervised_100_s20260702_segformer_b0/eval_validation_best

python scripts/evaluate_checkpoint.py \
  --config configs/supervised_segformer_b0.yaml \
  --checkpoint runs/20260702_supervised_100_s20260702_segformer_b0/best.pt \
  --split test \
  --output-dir runs/20260702_supervised_100_s20260702_segformer_b0/eval_test_best
```

评估输出：

- `metrics.json`；
- `class_metrics.csv`；
- `per_sample_metrics.csv`。

## 6. Run 汇总

```bash
python scripts/summarize_run.py \
  --run-dir runs/20260702_supervised_100_s20260702_segformer_b0 \
  --evaluation-dir runs/20260702_supervised_100_s20260702_segformer_b0/eval_validation_best \
  --evaluation-dir runs/20260702_supervised_100_s20260702_segformer_b0/eval_test_best \
  --output runs/20260702_supervised_100_s20260702_segformer_b0/run_summary.json
```

把 `run_summary.json`、`metrics.json`、`history.csv` 和关键日志带回本仓库后，再更新 `outputs/floodnet_experiment_registry.csv` 与 `outputs/floodnet_handoff_state.md`。不要提交 checkpoint 或大型日志目录。
