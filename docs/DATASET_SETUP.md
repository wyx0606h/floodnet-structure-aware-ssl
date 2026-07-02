# FloodNet 数据准备

## 两套数据的用途

本项目同时使用两套 FloodNet 数据：

| 数据 | 用途 | 规模 |
|---|---|---|
| Challenge Track 1 release | 确定官方 398 labeled 与 1047 unlabeled 名单 | 398 labeled train、1047 unlabeled train |
| FloodNet-Supervised_v1.0 | 提供所有训练/验证/测试图像的 mask | train 1445、val 450、test 448 |

`sup398` 的训练图像 ID 必须来自 Challenge 版 labeled 名单，但训练时读取完整监督版中的对应图像和 mask。`full1445` 使用完整监督版全部 train mask。

## 路径配置

Linux/macOS：

```bash
export SUPERVISED_ROOT=/data/FloodNet/FloodNet-Supervised_v1.0
export CHALLENGE_ROOT=/data/FloodNetChallenge/FloodNet\ Challenge\ @\ EARTHVISION\ 2021\ -\ Track\ 1
```

Windows PowerShell：

```powershell
$env:SUPERVISED_ROOT='D:\data\FloodNet\FloodNet-Supervised_v1.0'
$env:CHALLENGE_ROOT='D:\data\FloodNetChallenge\FloodNet Challenge @ EARTHVISION 2021 - Track 1'
```

源码中不硬编码本地路径；路径通过 YAML、命令行或环境变量传入。

## 生成 398/1047/1445/450/448 名单

```bash
python tools/build_floodnet_splits.py \
  --supervised-root "$SUPERVISED_ROOT" \
  --challenge-root "$CHALLENGE_ROOT" \
  --output-dir splits
```

生成文件：

- `splits/challenge_labeled_398.txt`
- `splits/challenge_unlabeled_1047.txt`
- `splits/full_train_1445.txt`
- `splits/val_450.txt`
- `splits/test_448.txt`
- `splits/sup398_manifest.csv`
- `splits/full1445_manifest.csv`

检查项：labeled=398、unlabeled=1047、full train=1445、val=450、test=448、labeled/unlabeled 无交集、二者并集等于完整 train、所有监督图像与 mask 存在。

## Mask 编码与读取

FloodNet mask 按类别索引读取：

- 二维 mask 直接使用；
- RGB/RGBA mask 若前三通道完全一致，则取第一个通道作为类别索引；
- RGB/RGBA 前三通道不一致时视为彩色 mask，当前训练代码会报错，避免静默错读；
- 标签值必须为 0–9，保留 `255` 作为 ignore index；
- mask 转为 `torch.long`；
- mask resize 只能使用最近邻；
- mask 不归一化。


## 服务器训练是否需要 Challenge 版

如果仓库中的 `splits/` 已经上传到服务器，则正式训练和评估只需要 `FloodNet-Supervised_v1.0`。Challenge 版数据只用于重新生成或复审以下名单：

- `challenge_labeled_398.txt`
- `challenge_unlabeled_1047.txt`
- `sup398_manifest.csv`
- `full1445_manifest.csv`

因此，为节省服务器上传空间，常规训练可以只上传完整监督版数据和仓库内的 `splits/`。
