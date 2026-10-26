# Transformer 性能对比分析

在当前目录下使用项目自带的 Python 环境运行以下命令。每条命令都会启动一个独立的进程。

**默认实验配置：**
- 模型规模：`small`
- 上下文长度：`context=128`
- 数据精度：`FP32`
- 批次大小：`batch=4`
- 词表大小：`vocab=10000`
- 关闭 TF32 加速
- 预热次数：5 次
- 正式测量次数：10 次
- 不生成 CSV 文件

## 1. PyTorch Profiler：按算子统计 GPU 耗时

```bash
../.venv/bin/python compare_training_attention.py --tool torch --mode inference
../.venv/bin/python compare_training_attention.py --tool torch --mode train
```

--tool torch	选择 PyTorch Profiler 进行性能分析。 .venv/bin/python 使用上一级虚拟环境中的python
需要同时比较 **GEMM 的执行时间和 GEMM 的耗时占比**。

计算 GEMM 耗时占比时，分母为所有算子归属的 GPU 自身耗时（self-attributed GPU operator time），而不是 CPU 墙钟时间（CPU wall time）。

程序会打印叶子级 GEMM 算子的分类结果。

对于作业 **1.1.4**，需要进一步使用 Nsight 的 CUDA Kernel 统计结果，验证算子级性能分析结论是否与 Kernel 级分析一致。

**Attention 统计说明：**

- Attention 的耗时统计为单个训练或推理 step 中，所有 Transformer 层的 Attention 前向计算总耗时。
- `attention_total` 已经包含各子模块耗时，不能再将其与子模块耗时相加，否则会重复统计。
- `QK+PV` 只包含注意力矩阵乘法，不包含 Scaling 和 Masking 操作。
- Softmax 的运算量统计中，每次比较操作和 `exp` 操作均按一次运算计数。这只是一种统计约定，不代表它们具有相同的实际浮点指令执行成本。
- 使用原始实现的自定义数值稳定 Softmax（custom stable softmax），没有替换为其他 Softmax 实现。

## 2. Nsight Systems：不启用 PyTorch Profiler

### 2.1 运行 Nsight 性能采集

首先创建结果保存目录：

```bash
mkdir -p nsys_results
```

采集 Inference 模式下的 CUDA 性能数据：

```bash
nsys profile --trace=cuda,nvtx --sample=none --capture-range=cudaProfilerApi --capture-range-end=stop -o nsys_results/compare_inference ../.venv/bin/python compare_training_attention.py --tool nsys --mode inference
```

采集 Train 模式下的 CUDA 性能数据：

```bash
nsys profile --trace=cuda,nvtx --sample=none --capture-range=cudaProfilerApi --capture-range-end=stop -o nsys_results/compare_train ../.venv/bin/python compare_training_attention.py --tool nsys --mode train
```

### 2.2 查看 CUDA Kernel 耗时统计

分别生成 Inference 和 Train 模式下的 CUDA Kernel 汇总统计：

```bash
nsys stats --report cuda_gpu_kern_sum nsys_results/compare_inference.nsys-rep
nsys stats --report cuda_gpu_kern_sum nsys_results/compare_train.nsys-rep
```

这些命令用于查看不同 CUDA Kernel 的累计执行时间、调用次数等信息，从而分析模型的主要 GPU 计算开销。

### 2.3 使用 Nsight Systems 图形界面查看结果

在 Ubuntu 图形桌面终端中执行：

```bash
nsys-ui nsys_results/compare_inference.nsys-rep
nsys-ui nsys_results/compare_train.nsys-rep
```

通过 Nsight Systems GUI，可以进一步查看 CUDA Kernel 时间线以及不同计算阶段对应的 GPU 活动。

### 2.4 Nsight 统计与分析注意事项

**Warmup 与正式采集：**

使用 `cudaProfilerStart/Stop` 控制采集范围，确保 Warmup 阶段不会进入正式性能分析结果。

在 Nsight Systems GUI 中，应选择相互对应的训练或推理 step 区间进行比较。

**Kernel 耗时计算：**

整个报告中的 Kernel 汇总统计覆盖 10 个正式测量 step。

因此，计算单个 step 的平均 Kernel 耗时时，需要将总耗时除以 10：

\[
T_{\mathrm{kernel,step}}
=
\frac{T_{\mathrm{kernel,total}}}{10}
\]

注意保持时间单位一致。如果总耗时以毫秒表示，则结果单位为 `ms/step`。

**NVTX 标记范围：**

程序中使用以下 NVTX 范围标记不同计算阶段：

```text
step
forward
loss
backward
optimizer
attention_total
attention_qk
attention_scale
attention_mask
attention_softmax
attention_pv
```

分析这些阶段时，应统计与对应 NVTX 范围关联的 **GPU Kernel 实际执行时间**，而不是直接使用 NVTX 范围在 CPU 时间线上的持续时间。

**GEMM 分析：**

根据每份 Nsight 报告中实际出现的 CUDA Kernel 名称，对 GEMM Kernel 进行分类和统计。

不要将 CPU 侧 CUDA API 调用时间相加后当作 GPU GEMM 执行时间。

**Attention 分析：**

统计一次 Forward 中所有 Transformer 层对应的 Attention GPU Kernel 执行时间，并将这些 Kernel 的耗时求和。

## 3. Python 计时：不使用性能分析器

如果只需要测量模型运行延迟，而不采集 PyTorch 算子或 CUDA Kernel 的详细性能数据，可以使用 Python 计时模式。

运行 Inference 模式：

```bash
../.venv/bin/python compare_training_attention.py --tool python --mode inference
```

运行 Train 模式：

```bash
../.venv/bin/python compare_training_attention.py --tool python --mode train
```

该模式不启用 PyTorch Profiler 或 Nsight，适合进行基础运行延迟对比。

## 4. 可选实验参数

可以通过命令行修改模型规模、上下文长度、预热次数和正式测量次数。

例如：

```bash
--size medium --context 256 --warmups 5 --steps 10
```

**仅执行开启梯度追踪的前向传播，不执行优化器更新：**

```bash
--mode forward
```

**可选：使用 PyTorch Profiler 导出 Chrome Trace 文件：**

```bash
--trace comparison_trace.json
```

导出的 Trace 文件可用于可视化分析。

## 5. 其他注意事项

- 如果某个 Nsight 报告文件已经存在，请使用新的输出文件名，避免覆盖原有分析结果。
- 如果多个程序同时争用 GPU 显存或计算资源，即使两组实验的其他配置完全一致，性能对比结果也可能失效。
- 本实验不会修改现有的 Notebook 或模型文件。

---

**实验分析目标：** 分别使用 PyTorch Profiler、Nsight Systems 和 Python 计时，对 Transformer 的 Inference 与 Train 模式进行性能对比，重点分析 GEMM 计算耗时及占比、Attention 各子模块的 GPU 执行开销，以及不同计算阶段的性能瓶颈。