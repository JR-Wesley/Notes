
[PyTorch Distributed Overview — PyTorch Tutorials 2.13.0+cu130 documentation](https://docs.pytorch.org/tutorials/beginner/dist_overview.html)

# PyTorch Distributed Overview

## torch.distributed 整体四大核心组件
### 1. 高层并行API（开箱即用并行方案）
提供封装好的并行训练接口，可与现有模型无缝组合：
- DDP（Distributed Data-Parallel）：数据并行
- FSDP2（Fully Sharded Data-Parallel）：全分片数据并行
- TP（Tensor Parallel）：张量并行
- PP（Pipeline Parallel）：流水线并行

### 2. 分片底层原语（多维度并行基础）
构建分片/复制张量的底层基础组件，支撑多维并行：
- `DTensor`：分片/复制张量，算子执行时自动重分片、完成通信
- `DeviceMesh`：多维设备通信抽象，统一管理多进程组，用于多维混合并行

### 3. 通信底层C10D（所有并行方案底层依赖）
提供两类通信原语，是上层并行API的底层实现：
- 集合通信：`all_reduce`、`all_gather` 等多进程同步算子
- 点对点通信：`send`/`isend` 一对一数据收发
配套教程提供原生通信API手写分布式程序示例

### 4. 分布式启动器
`torchrun`：官方标准多进程启动工具，支持单机多卡、多机分布式任务拉起

## 分布式并行选型核心指南（核心业务指导）
按模型大小、硬件承载能力分3档推荐方案，支持混合多维并行：
1. **模型单卡可放下，仅需多卡提速** → 选用 DDP
   - 多节点场景用 `torchrun` 启动
   - 可搭配自动混合精度AMP使用
2. **单卡放不下大模型（LLM等）** → 选用 FSDP2
3. **FSDP2 仍存在显存/算力瓶颈** → 叠加 TP张量并行 + PP流水线并行，组成3D并行
补充：数据并行可与模型并行组合，实现N维混合并行训练


# Pytorch 分布式实践

在 **PyTorch** 中，选择使用 **多线程** 还是 **多进程** 主要取决于你的任务类型（如数据加载、模型训练、推理等）、硬件环境（CPU/GPU）、以及是否涉及 Python 的 GIL 限制。下面我们结合 PyTorch 的实际应用场景来详细说明。

---

## 一、PyTorch 中的多线程适用场景

### ✅ 1. **数据加载（DataLoader 的 `num_workers > 0`）**

- PyTorch 的 `DataLoader` 支持通过 `num_workers` 参数开启多个子进程（⚠️注意：**默认是多进程，不是多线程！**）来并行加载和预处理数据。
- 但如果你设置 `num_workers=0`，则在主线程中加载数据，此时可以配合 Python 的 `threading` 手动实现轻量级并发（例如并发读取小文件）。
- **为什么有时用线程？**
    - 数据预处理涉及大量 **I/O 操作**（如读图、解码），线程在等待 I/O 时不会占用 CPU，适合用多线程提高吞吐。
    - 若预处理逻辑简单且不涉及大量 Python 计算，多线程可避免进程间通信开销。

> ⚠️ 注意：PyTorch 的 `DataLoader` 默认使用 **多进程**（`fork`）来避免 GIL 影响，尤其是在 Linux 上。但在 Windows 上，由于 `fork` 不可用，会使用 `spawn` 启动新进程。

### ✅ 2. **GPU 上的并行计算（自动多线程）**

- PyTorch 在 **GPU 上执行操作时是天然并行的**，底层使用 CUDA 的流（stream）和多线程。
- 你不需要手动创建线程，PyTorch 和 CUDA 驱动会自动利用 GPU 的多核并行能力。
- 即使在单个 Python 线程中，GPU 计算也是高度并行的。

### ✅ 3. **轻量级并发任务（如日志记录、监控、异步保存）**

- 在训练过程中，你可以使用 Python 多线程来执行非计算密集型任务：
    - 异步保存模型
    - 记录日志到文件或远程服务器
    - 实时可视化（如 TensorBoard 写入）
- 这些任务通常是 I/O 密集型，适合用 `threading` 或 `concurrent.futures.ThreadPoolExecutor`

---

## 二、PyTorch 中的多进程适用场景

### ✅ 1. **分布式训练（Distributed Training）**

这是 PyTorch 中使用多进程最典型、最重要的场景。

#### 使用 `torch.distributed` + `multiprocessing`

- **`torch.distributed.launch` 或 `torchrun`** 启动多个进程，每个进程绑定一个 GPU。
- 每个进程运行相同的代码，但处理不同的数据批次（Data Parallelism）或模型部分（Model Parallelism）。
- 常见策略：
    - **DataParallel (DP)**：单机多卡，主进程分发数据（已逐渐被 DDP 取代）
    - **DistributedDataParallel (DDP)**：**推荐方式**，每个 GPU 一个进程，独立前向/反向传播，通过 `all-reduce` 同步梯度，性能更高。

python

深色版本

```
# 示例：使用 DDP 的典型结构
def main(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    model = MyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    # 训练循环…
```

> ✅ **为什么用多进程？**
> 
> - 避免 GIL，充分发挥多核 CPU 和多 GPU 能力
> - 进程间隔离，稳定性高
> - 更好地控制 GPU 绑定（每个进程独占一个 GPU）

### ✅ 2. **CPU 上的并行数据预处理**

- 如果你在 CPU 上进行非常耗时的数据增强（如大规模图像变换、NLP 分词等），可以使用 `multiprocessing` 来并行处理。
- 例如：使用 `concurrent.futures.ProcessPoolExecutor` 预先处理数据集。

### ✅ 3. **多模型并行推理（Ensemble 推理）**

- 同时运行多个不同模型进行推理（如模型集成），每个模型放在独立进程中，避免相互阻塞。
- 特别适用于 CPU 推理或多个 GPU 上部署不同模型。

---

## 三、关键对比：PyTorch 中多线程 vs 多进程

|场景|推荐方式|原因|
|---|---|---|
|GPU 训练（单机多卡）|**多进程（DDP）**|绕过 GIL，避免串行化，性能最优|
|数据加载（I/O 密集）|**多进程（DataLoader num_workers）**|避免主进程阻塞，GIL 不影响子进程|
|CPU 计算密集型预处理|**多进程**|利用多核并行，绕过 GIL|
|GPU 计算本身|**无需手动线程**|CUDA 自动并行|
|日志、保存、监控等 I/O 任务|**多线程**|轻量、共享内存、避免 IPC 开销|
|异步任务（非计算）|`asyncio` 或 多线程|更高效|

---

## 四、特别注意：Python GIL 与 PyTorch 的关系

- **PyTorch 的 C++ 后端（如 ATen）不受 GIL 影响**：
    - 一旦进入 `.forward()`、`torch.matmul`、`conv2d` 等底层操作，GIL 会被释放。
    - 所以即使在单线程中，PyTorch 的张量计算也是并行的（尤其是 GPU）。
- **GIL 只影响纯 Python 代码和 CPU 上的 Python 回调函数**。

> ✅ 结论：**PyTorch 的计算瓶颈通常不在 Python 层，所以多线程在计算上帮助有限；真正需要并行时，应使用多进程。**

---

## 五、总结：PyTorch 中如何选择？

|任务类型|推荐方案|工具|
|---|---|---|
|单机单卡训练|无需多进程/线程|直接使用|
|单机多卡训练|**多进程 + DDP**|`torch.distributed`, `torchrun`|
|数据加载加速|**多进程 DataLoader**|`DataLoader(num_workers>0)`|
|CPU 数据预处理|多进程|`multiprocessing.Pool`|
|异步 I/O（保存、日志）|多线程或 `asyncio`|`threading`, `concurrent.futures`|
|多模型并发推理|多进程|`multiprocessing` 或 `torch.multiprocessing`|

---

### ✅ 一句话总结：

> **在 PyTorch 中，涉及 GPU 训练和 CPU 计算密集型任务时优先使用多进程（尤其是 DDP）；I/O 和轻量级并发任务可用多线程。**

合理利用 `torch.distributed` 和 `DataLoader` 的并行机制，是构建高效深度学习系统的最佳实践。