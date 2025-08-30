---
dateCreated: 2025-08-08
dateModified: 2025-08-08
---
结合 [[2 组件概述和加速方法]] 中讲解了总体调用 Python API 的流程，下面分析各个 API 具体做了什么，是如何调用 C++ 接口的。

# Python API

## 构造函数

初始化中，最重要部分是创建 `Buffer`，接口定义与 `deep_ep/buffer.py`，构造函数创建 `runtime` 调用 `csrc/deep_ep.cpp` 里的 `Buffer` 的构造函数。

```python

# deep_ep/buffer.py

@@ -32, 79

```

### **输入参数解析**

1. 通信组与基础配置

- `group`: PyTorch 分布式通信组（`ProcessGroup`），定义参与通信的 rank 集合。
- `explicitly_destroy`: 是否需要显式调用 `destroy()` 释放资源（默认由析构函数释放，避免 Python 异常时挂起）。

1. 缓冲区大小配置

- `num_nvl_bytes`: 节点内 NVLink 通信缓冲区大小（字节），用于高吞吐量节点内通信。
- `num_rdma_bytes`: 节点间 RDMA 通信缓冲区大小（字节），用于低延迟模式或跨节点通信。

1. 通信模式与硬件控

- `low_latency_mode`: 是否启用低延迟模式（优化通信延迟，依赖 RDMA 和 IBGDA 技术）。
- `allow_nvlink_for_low_latency_mode`: 低延迟模式下是否允许 NVLink 通信（需注意与 hook 机制的兼容性）。
- `allow_mnnvl`: 是否允许多节点 NVLink 检测（禁用可减少节点间通信复杂度）。

1. RDMA 与 QP 配置

- `num_qps_per_rank`: RDMA 通信的队列对（QP）数量，低延迟模式下需等于本地专家数（影响通信并行度）。

### 主要流程

1. 硬件连接检查

```python

check_nvlink_connections(group)

```

首先调用 `check_nvlink_connections` 验证通信组内所有 rank 之间的 NVLink 连接是否可用（节点内通信依赖 NVLink，节点间依赖 RDMA），避免因硬件连接问题导致通信失败。

1. 基础属性初始化

包括以下参数：

```python

self.rank = group.rank() # 当前 rank 编号

self.group_size = group.size() # 通信组内总 rank 数

[self.group](http://self.group/) = group # 通信组对象（如 PyTorch 分布式 ProcessGroup）

self.num_nvl_bytes = num_nvl_bytes # NVLink 缓冲区大小（字节）

self.num_rdma_bytes = num_rdma_bytes # RDMA 缓冲区大小（字节）

self.low_latency_mode = low_latency_mode # 是否启用低延迟模式

self.explicitly_destroy = explicitly_destroy # 是否需要显式释放资源

```

记录通信组信息、缓冲区大小、模式配置等基础参数，为后续资源分配和模式切换提供依据。

1. C++ 运行时初始化

```python

self.runtime = deep_ep_cpp.Buffer(…)

```

创建 C++ 底层运行时实例（`deep_ep_cpp.Buffer`），封装了实际的通信逻辑（如 NVLink/RDMA 操作）。Python 层通过调用 `self.runtime` 的方法间接操作底层通信。

1. 分布式信息同步

为确保所有 rank 协同工作，需要同步三类关键信息：

- **设备 ID 同步**：

每个 rank 获取本地设备 ID（GPU 编号），并通过 `dist.all_gather_object` 收集所有 rank 的设备 ID，确保跨设备通信时的设备一致性。

- **IPC 句柄同步**：

收集所有 rank 的进程间通信（IPC）句柄，用于节点内多进程共享内存访问（如同一节点内不同进程的 GPU 间通信）。

- **NVSHMEM 配置与 ID 同步**：

若启用低延迟模式或存在多 RDMA rank（节点间通信），则配置 NVSHMEM（NVIDIA 分布式共享内存库）环境变量（如 QP 数量、缓冲区粒度、禁用多节点 NVLink 检测等），并同步 root 节点的 NVSHMEM 唯一 ID，确保所有 rank 能通过 RDMA 建立连接。

1. 运行时就绪确认

```python

self.runtime.sync(device_ids, ipc_handles, root_unique_id)

assert self.runtime.is_available()

```

调用 `sync` 方法将同步后的设备 ID、IPC 句柄、NVSHMEM ID 传入 C++ 运行时，完成最终初始化，并断言运行时就绪（确保后续通信操作可用）。

### Distributed

同步 `device IDs, IPC handles, NVSHMEM unique IDs` 使用的是 `dist.ProcessGroup`，它是 PyTorch 分布式计算库 `torch.distributed` 里的一个核心类，它代表了一组参与分布式计算的进程。在分布式训练或者计算场景中，通常会有多个进程同时运行，这些进程需要相互通信和协作，`ProcessGroup` 就为这些进程提供了一个逻辑上的分组，方便管理和控制进程间的通信。

### 作用

在当前 `Buffer` 类的 `__init__` 方法里，`dist.ProcessGroup` 对象 `group` 主要有以下几个用途：

#### 1. 获取进程信息

```python

self.rank = group.rank()

self.group_size = group.size()

```

- `group.rank()`：返回当前进程在该进程组里的唯一标识符，即进程编号。`rank` 是一个非负整数，范围从 0 到 `group.size() - 1`。不同进程的 `rank` 不同，可用于区分不同进程，在分布式通信里，不同 `rank` 的进程可能承担不同任务。
- `group.size()`：返回进程组里的进程总数，也就是参与分布式计算的进程数量。

#### 2. 进程间通信同步

```python

dist.all_gather_object(device_ids, local_device_id, group)

dist.all_gather_object(ipc_handles, local_ipc_handle, group)

dist.all_gather_object(nvshmem_unique_ids, root_unique_id, group)

```

- `dist.all_gather_object` 是 PyTorch 分布式库提供的一个集体通信函数，作用是把每个进程的某个对象收集到所有进程中。这里通过传入 `group` 参数，指定在哪个进程组内进行通信同步操作。借助这个函数，所有进程能获取到其他进程的设备 ID、IPC 句柄以及 NVSHMEM 唯一 ID 等信息。

## Dispatch API

Buffer 有一个 `runtime` 成员实例化 C++ 接口的实现，在 [[2 组件概述和加速方法]] 中可以看到 Buffer 提供了的几个 API 里调用 `runtime` 来实现 dispatch 和 combine 功能。

`dispatch` 是 Buffer 类的核心方法，负责将输入的 tokens **分发到不同的分布式 rank**（支持节点内 NVLink 和节点间 RDMA 通信），是 MoE 模型中专家并行（EP）的关键步骤。其核心逻辑是根据 token 选择的专家索引（`topk_idx`），将 tokens 路由到对应专家所在的 rank，并返回接收端的 tokens 及通信元信息。

### 输入参数解析

1. 输入数据（待分发的 tokens）

- `x`: 待分发的 tokens 张量或 FP8 元组
- 单张量模式：`torch.bfloat16` 类型，形状 `[num_tokens, hidden]`
- FP8 元组模式：`(x_e4m3, x_scales)`，其中 `x_e4m3` 为 `torch.float8_e4m3fn` 类型（`[num_tokens, hidden]`），`x_scales` 为 `torch.float` 类型（`[num_tokens, hidden//128]`，需满足 `hidden` 可被 128 整除）

1. 缓存模式（复用布局信息，`handle≠None` 时）

- `handle`: 元组，缓存的通信布局信息（如前缀矩阵、接收索引等），由非缓存模式首次调用返回，用于后续分发时跳过布局计算，提升效率。缓存模式下需设置 `topk_idx=None`

1. 通信布局（非缓存模式必需，`handle=None` 时）

- `num_tokens_per_rank`: 张量 `[num_ranks]`（`torch.int`），每个 rank 需接收的 tokens 数量
- `num_tokens_per_rdma_rank`: 张量 `[num_rdma_ranks]`（`torch.int`），每个 rdma rank 需接收的 tokens 数量，intranode 为 `None`
- `is_token_in_rank`: 张量 `[num_tokens, num_ranks]`（`torch.bool`），标记每个 token 是否发送到对应 rank
- `num_tokens_per_expert`: 张量 `[num_experts]`（`torch.int`），每个专家需接收的 tokens 数量

1. 专家选择信息

- `topk_idx`: 张量 `[num_tokens, num_topk]`（`torch.int64`），每个 token 选择的专家索引（`-1` 表示未选择），非缓存模式必需
- `topk_weights`: 张量 `[num_tokens, num_topk]`（`torch.float`），每个 token 对所选专家的权重，用于后续聚合加权

1. 性能与内存配置

- `expert_alignment`: 整数，接收的 tokens 数量需对齐到该值（如 16/32），优化内存访问效率，默认 1
- `num_worst_tokens`: 整数，指定接收 tokens 的最大可能数量（仅节点内模式支持），避免 CPU-GPU 同步，提升 CUDA Graph 兼容性，默认 0
- `config`: `Config` 对象，分发内核的性能调优参数（如 SM 数量、块大小等），默认通过 `get_dispatch_config` 根据 rank 数量自动选择

1. 通信同步与流控制

- `previous_event`: `EventOverlap` 对象，分发开始前需等待的前置事件（如上游计算完成）
- `async_finish`: 布尔值，若为 `True`，当前 CUDA 流不等待分发内核完成，直接返回（需通过返回的 `event` 手动同步），默认 `False`
- `allocate_on_comm_stream`: 布尔值，若为 `True`，分配的输出张量所有权归通信流（而非默认计算流），优化异步通信效率

### 输出参数解析

返回一个元组，包含接收的 tokens、专家信息、通信句柄及同步事件，具体如下：

- `recv_x`: 接收的 tokens 数据，类型与输入 `x` 一致
- 单张量模式：`torch.bfloat16` 类型，形状 `[num_recv_tokens, hidden]`
- FP8 元组模式：`(recv_x_e4m3, recv_x_scales)`，其中 `recv_x_e4m3` 为 `torch.float8_e4m3fn` 类型，`recv_x_scales` 为 `torch.float` 类型
- `recv_topk_idx`: 接收的专家索引张量（仅非缓存模式返回，缓存模式为 `None`）
- 形状 `[num_recv_tokens, num_topk]`，`torch.int64` 类型，与输入 `topk_idx` 对应
- `recv_topk_weights`: 接收的专家权重张量（仅非缓存模式返回，缓存模式为 `None`）
- 形状 `[num_recv_tokens, num_topk]`，`torch.float` 类型，与输入 `topk_weights` 对应
- `num_recv_tokens_per_expert_list`: 本地专家接收 tokens 数量列表（仅非缓存模式返回，缓存模式为 `None`）
- Python 列表 `[num_local_experts]`，每个元素为对应专家接收的 tokens 数（已按 `expert_alignment` 对齐）；若 `num_worst_tokens > 0`，返回空列表
- `handle`: 通信布局句柄（仅非缓存模式返回，缓存模式为 `None`）
- 元组，包含前缀矩阵、接收索引等布局信息，用于后续缓存模式分发（`handle≠None` 时复用）
- `event`: 分发内核完成事件（仅 `async_finish=True` 时有效）
- `EventOverlap` 对象，用于同步异步通信操作的完成状态

### **关键逻辑分支**

1. **节点间/节点内通信切换**：

- 若 `runtime.get_num_rdma_ranks() > 1`（存在多节点 RDMA 通信），调用 `internode_dispatch`；
- 否则为节点内通信，调用 `intranode_dispatch`（依赖 NVLink）。

1. **缓存/非缓存模式**：

- **缓存模式**（`handle is not None`）：复用 `handle` 中的布局信息，直接执行分发，跳过布局计算，减少重复计算，提升效率；
- **非缓存模式**（`handle is None`）：需提供 `num_tokens_per_rank`/`is_token_in_rank`/`num_tokens_per_expert`，计算布局并返回新 `handle`。

1. `async_finish` 与 `event`：异步通信与同步控制

两者配合实现通信过程的异步执行与显式同步，避免阻塞当前 CUDA 流，提升计算与通信的重叠效率。

**`async_finish`：控制是否异步执行**

- **类型**：布尔值（默认 `False`）。
- **作用**：
- 当 `async_finish=False`（默认）：当前 CUDA 流会等待通信内核执行完成后再继续，确保数据就绪。
- 当 `async_finish=True`：当前 CUDA 流不等待通信完成，直接返回，允许后续计算与通信并行执行（需通过 `event` 手动同步）。

**`event`：通信完成的同步标记**

- **类型**：`EventOverlap` 对象（封装 CUDA 事件）。
- **作用**：

当 `async_finish=True` 时，`event` 记录通信内核的完成状态。用户可通过 `event.wait()` 等操作显式等待通信完成，避免后续计算访问未就绪的数据。

1. 调用 C++ 接口：

节点内就直接调用 C++ runtime `dispatch intranode`。

节点间再次处理 python 接口的 `dispatch internode`。

### **总结**

`dispatch` 方法通过灵活的参数配置，支持 MoE 模型中 tokens 的高效分布式分发，兼顾高吞吐量（节点内 NVLink）和低延迟（节点间 RDMA）场景。其核心价值在于：

- 支持 BF16/FP8 数据类型，平衡精度与带宽；
- 提供缓存机制复用布局信息，减少重复计算；
- 通过事件同步和流控制，支持通信与计算重叠，提升整体性能。

---
