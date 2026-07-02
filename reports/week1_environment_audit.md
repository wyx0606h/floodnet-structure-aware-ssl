# Week 1 本地运行环境审计

> 审计日期：2026-06-25  
> 审计范围：本地 CPU 开发环境，不安装依赖、不下载权重、不启动训练

## 可用解释器

```text
D:\Anaconda3\python.exe
Python 3.9.19
NumPy 1.26.4
PyTorch 2.3.0
```

PyTorch 为 CPU-only 构建：

- `torch.cuda.is_available() == False`
- `torch.version.cuda is None`
- CUDA device count 为 0

默认解释器 `d:\py\Anaconda3\python.exe` 使用 NumPy 2.0.2，与其中现有
PyTorch 二进制存在 ABI 冲突，不用于本项目当前本地测试。

## 依赖状态

| 依赖 | 状态 |
|---|---|
| `torchvision` | 已安装 |
| `PyYAML` | 已安装 |
| `transformers` | 未安装 |
| `safetensors` | 未安装 |
| `accelerate` | 未安装 |

## 预训练权重缓存

以下本地缓存目录存在，但未发现 SegFormer、MiT-B0 或 NVIDIA SegFormer-B0
模型目录：

```text
C:\Users\吴\.cache\huggingface\hub
C:\Users\吴\.cache\torch\hub
```

## 影响

- Dataset、指标、增强、滑窗推理和纯 PyTorch 单元测试可继续在 CPU 完成；
- 可以先实现配置、训练循环和依赖检查；
- 随机初始化的本地四图门只需要安装 `transformers`，不需要下载权重；
- 正式预训练基线还需要 `safetensors` 和本地/可下载的 `nvidia/mit-b0` 权重；
- 任何联网安装或权重下载都应在用户确认后执行。
