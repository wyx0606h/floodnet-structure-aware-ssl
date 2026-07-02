# Week 1 本地命令

当前 Windows 环境建议使用 `D:\Anaconda3\python.exe`。默认 `d:\py\Anaconda3\python.exe` 曾出现 NumPy 2 与当前 PyTorch ABI 冲突。

以下命令均不训练模型，除非显式出现 `--execute`。

## 1. 当前数据协议

主数据源为 `F:\FloodNet\FloodNet-Supervised_v1.0`，也可以把 `FLOODNET_DATA_ROOT` 指向父目录 `F:\FloodNet`。

旧 `F:\数据集\Flood\FloodNet_Track1_Merged` challenge release 和 278/60/60 split 只保留为历史审计，不再作为主训练/评估协议。

## 2. supervised manifest

官方 1445/450/448 manifest 已生成在 `splits/floodnet_supervised_v1/`。如需在新机器上复现生成过程，命令为：

```powershell
& 'D:\Anaconda3\python.exe' scripts\create_supervised_manifest.py `
  --data-root 'F:\FloodNet' `
  --output-dir 'splits\floodnet_supervised_v1'
```

脚本会拒绝覆盖已有输出文件。

## 3. 单元测试与真实数据 smoke

```powershell
& 'D:\Anaconda3\python.exe' -m unittest discover -s tests -v

& 'D:\Anaconda3\python.exe' scripts\smoke_data_pipeline.py `
  --data-root 'F:\FloodNet' `
  --manifest 'splits\floodnet_supervised_v1\manifest.csv' `
  --output 'reports\week1_data_smoke_supervised_v1.json' `
  --crop-size 512 `
  --batch-size 2 `
  --seed 20260702
```

当前结果：新 official train/validation/test DataLoader smoke 已通过；未训练、未写 `runs/`。

## 4. 训练预检

训练入口默认 dry-run，不建模型、不写 run、不训练：

```powershell
$env:FLOODNET_DATA_ROOT='F:\FloodNet'
& 'D:\Anaconda3\python.exe' scripts\train_supervised.py `
  --config 'configs\overfit4_segformer_b0.yaml'
```

只读依赖预检并保存报告：

```powershell
$env:FLOODNET_DATA_ROOT='F:\FloodNet'
& 'D:\Anaconda3\python.exe' scripts\check_training_prereqs.py `
  --config 'configs\overfit4_segformer_b0.yaml' `
  --output 'reports\training_preflight_overfit4_supervised_v2.json'
```

当前预检能读到四图 manifest 和数据根，但本地缺少 `transformers`，因此按预期阻塞真实 SegFormer 构建/训练。

## 5. 四图门训练命令（需用户确认）

只有依赖预检通过且用户明确允许写 `runs/` 后，才执行：

```powershell
$env:FLOODNET_DATA_ROOT='F:\FloodNet'
& 'D:\Anaconda3\python.exe' scripts\train_supervised.py `
  --config 'configs\overfit4_segformer_b0.yaml' `
  --execute `
  --confirm-run-id '20260702_overfit4_100_s20260702_segformer_b0'
```

四图门通过条件：

- final train loss / initial train loss ≤ 0.20；
- final train mIoU-10 ≥ 0.90。

## 6. 服务器训练、评估和汇总

完整服务器命令见 [`docs/server_training_quickstart.md`](../docs/server_training_quickstart.md)。
