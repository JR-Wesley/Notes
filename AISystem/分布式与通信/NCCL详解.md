---
dateCreated: 2025-08-15
dateModified: 2025-08-15
---

> [!note] Reference

> <a href="[https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)">NCCL UG</a>

> [JSC Advanced Course: Using NCCL and NVSHMEM]([https://juser.fz-juelich.de/record/1019178/files/02-NCCL_NVSHMEM.pdf](https://juser.fz-juelich.de/record/1019178/files/02-NCCL_NVSHMEM.pdf))

> [xCCL: 对工业界深度学习集合通信库的综述]([https://jcst.ict.ac.cn/cn/article/doi/10.1007/s11390-023-2894-6](https://jcst.ict.ac.cn/cn/article/doi/10.1007/s11390-023-2894-6))

> [Massively Scale Your Deep Learning Training with NCCL 2.4 | NVIDIA Technical Blog]([https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/))

[nccl_KIDGINBROOK 的博客-CSDN 博客]([https://blog.csdn.net/kidgin7439/category_11998768.html](https://blog.csdn.net/kidgin7439/category_11998768.html))

# Overview

## Introduction

The NVIDIA Collective Communications Library (NCCL, pronounced “Nickel”) is a library providing **inter-GPU communication primitives** that are topology-aware and can be easily integrated into applications.

NCCL implements both **collective communication and point-to-point send/receive primitives**. It is not a full-blown parallel programming framework; rather, it is a library focused on accelerating inter-GPU communication.

NCCL provides the following collective communication primitives:

- AllReduce
- Broadcast
- Reduce
- AllGather
- ReduceScatter

Additionally, it allows for point-to-point send/receive communication which allows for scatter, gather, or all-to-all operations.

## Feature

1. NCCL implements each collective in a single kernel handling both communication and computation operations. This allows for fast synchronization and minimizes the resources needed to reach peak bandwidth.
2. It supports a variety of interconnect technologies including PCIe, NVLINK, InfiniBand Verbs, and IP sockets.
3. NCCL uses a simple C API, which can be easily accessed from a variety of programming languages. NCCL closely follows the popular collectives API defined by MPI (Message Passing Interface). Anyone familiar with MPI will thus find NCCL’s API very natural to use. In a minor departure from MPI, NCCL collectives take a “stream” argument which provides direct integration with the CUDA programming model.
4. NCCL is compatible with virtually any multi-GPU parallelization model, for example:

- single-threaded control of all GPUs
- multi-threaded, for example, using one thread per GPU
- multi-process, for example, MPI

1. NCCL relies therefore on the application’s process management system and CPU-side communication system for its own bootstrap.
2. Similarly to MPI and other libraries which are optimized for performance, NCCL does not provide secure network communication between GPUs

# Usage

Using NCCL is similar to using any other library in your code:

1. Install the NCCL library on your system
2. Modify your application to link to that library
3. Include the header file nccl. h in your application
4. Create a communicator (see [Creating a Communicator]([https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#communicator-label](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#communicator-label)))
5. Use NCCL collective communication primitives to perform data communication.

## 1. Primitives

The communication primitives include **Collective Communication** and **Point-to-point Communication**, see [Collective Operations — NCCL 2.27.5 documentation]([https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)) for details.

## 2. Communicator

### Data Pointers

## 3. Concurrency

### CUDA Stream Semantics

### Using NCCL with CUDA Graphs

Starting with NCCL 2.9, NCCL operations can be captured by CUDA Graphs. CUDA Graphs provide a way to define workflows as graphs rather than single operations. They may reduce overhead by launching multiple GPU operations through a single CPU operation. NCCL’s collective, P 2 P and group operations all support CUDA Graph captures. This support requires a minimum CUDA version of 11.3.

> [!note] More About CUDA Graph

> [Accelerating PyTorch with CUDA Graphs – PyTorch]([https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/))

> [CUDA Programming Guide]([https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs))

> [CUDA stream Slide 1]([https://developer.download.nvidia.cn/CUDA/training/StreamsAndConcurrencyWebinar.pdf](https://developer.download.nvidia.cn/CUDA/training/StreamsAndConcurrencyWebinar.pdf))

Whether an operation launch is graph-captured is considered a collective property of that operation and therefore **must be uniform over all ranks participating in the launch** (for collectives this is all ranks in the communicator, for peer-to-peer this is both the sender and receiver). The launch of a graph (via cudaGraphLaunch, etc.) containing a captured NCCL operation is considered collective for the same set of ranks that were present in the capture, and each of those ranks must be using the graph derived from that collective capture.

### Group Calls

## 4. Miscellaneous

### User Buffer Registration

User Buffer Registration is a feature that **allows NCCL to directly send/receive/operate data through the user buffer without extra internal copy (zero-copy)**. It can accelerate collectives and greatly reduce the resource usage (e.g. # channel usage). NCCL provides two ways to register user buffers; **one is _CUDA Graph_ registration, and the other is _Local_ registration**. NCCL requires that for all NCCL communication function calls (e.g., allreduce, sendrecv, and so on), if any rank in a communicator passes registered buffers to a NCCL communication function, all other ranks in the same communicator must pass their registered buffers; otherwise, mixing registered and non-registered buffers can result in undefined behavior; in addition, source and destination buffers must be registered in order to enable user buffer

### In-place Operations

Contrary to MPI, NCCL does not define a special “in-place” value to replace pointers. Instead, NCCL optimizes the case where the provided pointers are effectively “in place”.

For ncclBroadcast, ncclReduce and ncclAllreduce functions, this means that passing will perform in place operations, storing final results at the same place as initial data was read from. `sendBuff == recvBuff`

For ncclReduceScatter and ncclAllGather, in place operations are done when the per-rank pointer is located at the rank offset of the global buffer. More precisely, these calls are considered in place:

```cpp

ncclReduceScatter(data, data+rank*recvcount, recvcount, datatype, op, comm, stream);

ncclAllGather(data+rank*sendcount, data, sendcount, datatype, op, comm, stream);registration for NCCL operations.

```

### Thread Safety

# API

The following sections describe the NCCL methods and operations.

- [NCCL API]([https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html))
- [Communicator Creation and Management Functions]([https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html)): These functions handle the setup and teardown of communication contexts between GPUs. These form the foundation for all other NCCL operations by establishing the communication topology.
- [Collective Communication Functions]([https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#)): These functions contains the core data aggregation operations that synchronize data across multiple GPUs.
- [Group Calls]([https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html)): These functions allow batching multiple collective operations together for improved efficiency:
- [Point To Point Communication Functions]([https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html)): These complement the collective operations with more fine-grained direct communication between GPU pairs.
- [Types]([https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html)): Defines core data types used throughout the API.
- [User Defined Reduction Operators]([https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/ops.html](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/ops.html)): Extends NCCL's capabilities with custom operations.
- [NCCL API Supported Flags]([https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/flags.html](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/flags.html)): Special behavior modifiers for operations.

## Types

- `ncclComm_t`: Communication object handle
- `ncclResult_t`: Error code return type
- `ncclDataType_t`: Enumeration of supported data types (float, double, int, etc.)
- `ncclRedOp_t`: Reduction operations (sum, product, min, max, etc.)

# Examples

## 不同的通信执行方式

PyTorch Distributed（`torch.distributed`）通过灵活的进程组（Process Group）机制、设备抽象和通信后端，支持多种通信模式。其核心是通过**进程组**管理通信范围，通过**通信原语**（如 `all_reduce`、`broadcast` 等）实现数据交换，并适配不同的设备（CPU/GPU）和线程 / 进程架构。以下针对四种通信模式，分别说明其实现方式和原理：

### 基础概念

- **进程组（Process Group）**：分布式通信的基本单位，包含一组参与通信的进程。默认进程组为 `WORLD`（包含所有进程），也可通过 `new_group()` 创建自定义进程组（即 “通信器”）。
- **通信后端**：底层通信实现（如 `NCCL`（GPU 间高效通信）、`Gloo`（CPU/GPU 通用）、`MPI`（多进程通信标准）），不同后端对设备和线程的支持不同（例如 `NCCL` 主要优化 GPU，且不支持多线程）。
- **Rank**：进程组内的唯一标识（整数），用于定位进程；`local_rank` 表示进程在本地节点的序号（常用于绑定设备）。

### 1. Single Process, Single Thread, Multiple Devices（单进程单线程，多设备）

**场景**：一个进程（单线程）内使用多个设备（如多个 GPU），设备间需要通信（如同一进程内的多 GPU 数据同步）。

**实现方式**：

- 无需多进程，只需在单进程内初始化分布式环境（通常用 `backend="nccl"`，因 GPU 通信效率高）。
- 通过 `device` 参数显式指定数据所在设备，通信操作会自动在指定设备间完成。

**代码示例**：

```python

import torch

import torch.distributed as dist

  

# 单进程初始化分布式（需指定初始化方式，如file://或env://）

dist.init_process_group(

backend="nccl", # GPU通信优先用NCCL

init_method="file:///tmp/dist_init", # 共享文件初始化（单节点内简单）

world_size=1, # 只有1个进程

rank=0 # 当前进程rank为0

)

  

# 单进程内使用2个GPU

device0 = torch.device("cuda:0")

device1 = torch.device("cuda:1")

  

# 在两个设备上创建张量

tensor0 = torch.tensor([1.0, 2.0], device=device0)

tensor1 = torch.tensor([3.0, 4.0], device=device1)

  

# 设备间通信（如all_reduce，结果会广播到所有参与的张量）

dist.all_reduce(tensor0, op=dist.ReduceOp.SUM) # tensor0 = [1+3, 2+4] = [4,6]

dist.all_reduce(tensor1, op=dist.ReduceOp.SUM) # tensor1 = [4,6]（与tensor0同步）

  

print(tensor0, tensor1)

```

**注意**：

- 单进程多设备通信依赖后端支持（`NCCL` 天然支持同一进程内多 GPU 通信）。
- 无需多进程管理，适合小规模多设备任务（如模型并行的部分通信）。

### 2. One Device per Process or Thread（每个进程 / 线程一个设备）

**场景**：最常见的分布式训练模式（如数据并行），每个进程（或线程）绑定一个设备（GPU/CPU），进程 / 线程间通过通信同步数据。

#### 2.1 多进程（每个进程一个设备）

这是 PyTorch 分布式的主流用法（如 `DistributedDataParallel`），通过多进程隔离设备，避免资源竞争。

**实现方式**：

- 用 `torch.multiprocessing.spawn` 启动多进程（每个进程对应一个 `local_rank`）。
- 每个进程绑定一个设备（`cuda:local_rank`），通过 `init_process_group` 初始化跨进程通信。

**代码示例**：

```python

import torch

import torch.distributed as dist

import torch.multiprocessing as mp

  

def run(rank, world_size):

# 初始化进程组（多进程通信）

dist.init_process_group(

backend="nccl",

init_method="env://", # 依赖环境变量（如MASTER_ADDR, MASTER_PORT）

world_size=world_size,

rank=rank

)

# 每个进程绑定一个设备（rank=0→cuda:0，rank=1→cuda:1）

device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

tensor = torch.tensor([rank + 1.0], device=device)

# 跨进程通信（如all_reduce求和）

dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

print(f"Rank {rank}, Device {device}: {tensor}") # 所有进程输出 [1+2+...+world_size]

  

if __name__ == "__main__":

world_size = 2 # 2个进程（对应2个设备）

mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)

```

#### 2.2 多线程（每个线程一个设备）

**场景**：用多线程代替多进程（较少见，因 Python GIL 限制，但可用于特定场景）。

**实现方式**：

- 线程间共享进程组，但需确保通信操作在正确的线程上下文（绑定设备）中执行。
- 注意：`NCCL` 不支持多线程（会报错），需用 `Gloo` 后端（支持多线程但 GPU 效率较低）。

**代码示例**：

```python

import torch

import torch.distributed as dist

import threading

  

def thread_func(rank, device):

# 线程绑定设备

torch.cuda.set_device(device)

tensor = torch.tensor([rank + 1.0], device=device)

# 线程内通信（需用Gloo后端）

dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

print(f"Thread {rank}, Device {device}: {tensor}")

  

if __name__ == "__main__":

world_size = 2

# 初始化多线程通信（必须用Gloo后端，NCCL不支持多线程）

dist.init_process_group(

backend="gloo",

init_method="file:///tmp/dist_init",

world_size=world_size,

rank=0 # 单进程内多线程，共享同一个rank（或自定义子进程组）

)

# 启动2个线程，每个绑定一个设备

threads = []

for i in range(world_size):

t = threading.Thread(target=thread_func, args=(i, torch.device(f"cuda:{i}")))

threads.append(t)

t.start()

for t in threads:

t.join()

```

### 3. Multiple Devices per Thread（每个线程多个设备）

**场景**：一个线程管理多个设备（如模型并行中，线程内的不同设备负责模型的不同层，需频繁通信）。

**实现方式**：

- 线程内显式管理多个设备（通过 `device` 参数指定）。
- 通信操作需明确指定参与的设备，或通过进程组覆盖所有设备。

**代码示例**：

```python

import torch

import torch.distributed as dist

  

def run():

dist.init_process_group(

backend="nccl",

init_method="file:///tmp/dist_init",

world_size=1,

rank=0

)

# 一个线程管理2个设备

device0 = torch.device("cuda:0")

device1 = torch.device("cuda:1")

# 设备0上的张量（模拟模型层1输出）

tensor0 = torch.tensor([1.0, 2.0], device=device0)

# 设备1上的张量（模拟模型层2输出）

tensor1 = torch.tensor([3.0, 4.0], device=device1)

# 线程内多设备通信（如all_gather合并结果）

gathered = [torch.empty_like(tensor0, device=device0) for _ in range(2)]

dist.all_gather(gathered, tensor0) # 收集所有设备的tensor0（这里仅示例，实际需匹配设备）

print("Gathered on device0:", gathered)

  

if __name__ == "__main__":

run()

```

**注意**：

- 需确保通信操作的输入张量分布在正确的设备上，避免跨设备内存拷贝开销。
- 适合模型并行场景（如 Transformer 的层拆分到不同设备）。

### 4. Multiple Communicators per Device（每个设备多个通信器）

**场景**：一个设备（如 GPU）上存在多个独立的通信需求（如同时运行两个不同的分布式任务），需用多个通信器（进程组）隔离。

**实现方式**：

- 通过 `dist.new_group()` 创建多个进程组（即 “通信器”），每个通信器对应一组进程和通信逻辑。
- 设备上的张量可根据需求选择不同的通信器进行通信。

**代码示例**：

```python

import torch

import torch.distributed as dist

import torch.multiprocessing as mp

  

def run(rank, world_size):

dist.init_process_group(

backend="nccl",

init_method="env://",

world_size=world_size,

rank=rank

)

device = torch.device(f"cuda:{rank}")

# 创建2个通信器（进程组）：

# 通信器1：包含所有进程（类似WORLD）

comm1 = dist.new_group(ranks=list(range(world_size)))

# 通信器2：仅包含偶数rank的进程（如rank 0）

comm2 = dist.new_group(ranks=[0] if world_size > 1 else [0])

# 用通信器1通信（所有进程参与）

tensor1 = torch.tensor([rank + 1.0], device=device)

dist.all_reduce(tensor1, op=dist.ReduceOp.SUM, group=comm1)

# 用通信器2通信（仅部分进程参与）

tensor2 = torch.tensor([rank * 2.0], device=device)

if rank in [0]: # 仅comm2包含的进程执行

dist.all_reduce(tensor2, op=dist.ReduceOp.SUM, group=comm2)

print(f"Rank {rank}: comm1 result={tensor1}, comm2 result={tensor2}")

  

if __name__ == "__main__":

world_size = 2

mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)

```

**注意**：

- 每个通信器独立管理通信状态，避免不同任务的通信冲突。
- 过多通信器可能增加资源开销，需合理规划。

### 总结

PyTorch Distributed 通过**进程组（通信器）** 实现通信范围隔离，通过**多后端支持**适配不同设备（CPU/GPU），通过**显式设备指定**和**线程 / 进程管理**支持多样化的通信模式：

- 单进程多设备：依赖 `NCCL` 后端，直接在进程内的设备间通信。
- 每进程 / 线程一设备：主流模式（如 DDP），多进程用 `spawn` 启动，多线程需 `Gloo` 后端。
- 每线程多设备：适合模型并行，显式管理设备间通信。
- 每设备多通信器：通过 `new_group()` 创建多个进程组，隔离不同通信任务。

实际使用中需根据硬件（CPU/GPU 数量）、任务类型（数据并行 / 模型并行）选择合适的模式，并优先使用 `NCCL` 后端（GPU 场景）以获得高效通信。

## Example 2: One Device per Process or Thread

> [Example 2: One Device per Process or Thread]([https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html#example-2-one-device-per-process-or-thread](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html#example-2-one-device-per-process-or-thread) "Permalink to this headline")

## Example 3: Multiple Devices per Thread

> [Example 3: Multiple Devices per Thread]([https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html#example-3-multiple-devices-per-thread](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html#example-3-multiple-devices-per-thread) "Permalink to this headline")

You can combine both multiple process or threads and multiple device per process or thread. In this case, we need to use group semantics.

### Flow

The following code is an example of a communicator creation in the context of MPI, using one device per MPI rank.

First, we retrieve MPI information about processes:

```

int myRank, nRanks;

MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

```

Next, a single rank will create a unique ID and send it to all other ranks to make sure everyone has it:

```

ncclUniqueId id;

if (myRank == 0) ncclGetUniqueId(&id);

MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

```

Finally, we create the communicator:

```

ncclComm_t comm;

ncclCommInitRank(&comm, nRanks, id, myRank);

```

We can now call the NCCL collective operations using the communicator.

```

ncclAllReduce( ... , comm);

```

Finally, we destroy the communicator object:

```

ncclCommDestroy(comm);

```

### Code Review

#### **1. 初始化 MPI 环境**

```

MPICHECK(MPI_Init(&argc, &argv));

MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));

MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

```

- **功能**：
- `MPI_Init`：初始化 MPI 环境，必须在任何 MPI 调用之前调用。
- `MPI_Comm_rank`：获取当前进程的全局编号（`myRank`），用于标识进程在通信组中的身份。
- `MPI_Comm_size`：获取通信组中总进程数（`nRanks`）。
- **目的**：为后续分布式计算提供进程管理和通信基础。

---

#### **2. 计算本地 GPU 编号（`localRank`）**

```

uint64_t hostHashs[nRanks];

char hostname[1024];

getHostName(hostname, 1024);

hostHashs[myRank] = getHostHash(hostname);

MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));

for (int p=0; p<nRanks; p++) {

if (p == myRank) break;

if (hostHashs[p] == hostHashs[myRank]) localRank++;

}

```

- **功能**：
- 通过 `getHostName` 获取当前节点的主机名，并计算哈希值（`hostHash`）。
- 使用 `MPI_Allgather` 收集所有进程的 `hostHash`，判断当前进程所在的主机。
- 根据 `hostHash` 的重复情况计算 `localRank`（同一主机内的 GPU 编号）。
- **目的**：确定当前进程在主机内的 GPU 分配，确保每个进程正确选择本地 GPU 设备。

---

#### **3. 分配 GPU 设备和内存**

```

int nDev = 2;

float** sendbuff = (float**)malloc(nDev * sizeof(float*));

float** recvbuff = (float**)malloc(nDev * sizeof(float*));

cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);

  

for (int i = 0; i < nDev; ++i) {

CUDACHECK(cudaSetDevice(localRank*nDev + i));

CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));

CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));

CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));

CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));

CUDACHECK(cudaStreamCreate(s+i));

}

```

- **功能**：
- 每个进程使用 2 个 GPU（`nDev = 2`），通过 `cudaSetDevice` 设置当前 GPU。
- 使用 `cudaMalloc` 在 GPU 上分配发送缓冲区（`sendbuff`）和接收缓冲区（`recvbuff`）。
- 使用 `cudaMemset` 初始化缓冲区数据（`sendbuff` 全为 1，`recvbuff` 全为 0）。
- 创建 CUDA 流（`cudaStream_t`）用于异步通信。
- **目的**：为每个 GPU 设备准备数据和资源，确保后续 NCCL 通信可以正确执行。

---

#### **4. 生成并广播 NCCL 唯一 ID**

```

ncclUniqueId id;

if (myRank == 0) ncclGetUniqueId(&id);

MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

```

- **功能**：
- `ncclGetUniqueId`：由主进程（`myRank == 0`）生成一个 NCCL 唯一 ID（`id`），用于初始化通信器。
- `MPI_Bcast`：将 `id` 广播到所有进程，确保所有进程使用相同的 ID 初始化 NCCL 通信器。
- **目的**：为 NCCL 通信器提供统一的初始化标识符，确保所有进程加入同一个通信组。

---

#### **5. 初始化 NCCL 通信器**

```

NCCLCHECK(ncclGroupStart());

for (int i=0; i<nDev; i++) {

CUDACHECK(cudaSetDevice(localRank*nDev + i));

NCCLCHECK(ncclCommInitRank(comms+i, nRanks*nDev, id, myRank*nDev + i));

}

NCCLCHECK(ncclGroupEnd());

```

- **功能**：
- `ncclGroupStart` 和 `ncclGroupEnd`：将多个 GPU 的通信器初始化操作分组，确保线程安全。
- `ncclCommInitRank`：为每个 GPU 初始化一个 NCCL 通信器（`comms[i]`），参数说明：
- `nRanks*nDev`：总通信节点数（每个 GPU 是一个独立节点）。
- `id`：广播的 NCCL 唯一 ID。
- `myRank*nDev + i`：当前 GPU 的全局 rank（每个 GPU 独立编号）。
- **目的**：为每个 GPU 设备创建独立的 NCCL 通信器，支持多设备并行通信。

---

#### **6. 调用 NCCL 集合通信 API**

```

NCCLCHECK(ncclGroupStart());

for (int i=0; i<nDev; i++)

NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,

comms[i], s[i]));

NCCLCHECK(ncclGroupEnd());

```

- **功能**：
- `ncclGroupStart` 和 `ncclGroupEnd`：将多个 GPU 的集合通信操作分组，确保异步执行。
- `ncclAllReduce`：对每个 GPU 的数据执行全归约操作（`ncclSum`），将 `sendbuff` 中的数据相加后存入 `recvbuff`。
- 参数说明：
- `sendbuff[i]`：发送缓冲区地址。
- `recvbuff[i]`：接收缓冲区地址。
- `size`：数据元素数量。
- `ncclFloat`：数据类型（`float`）。
- `ncclSum`：归约操作类型（求和）。
- `comms[i]`：当前 GPU 的通信器。
- `s[i]`：对应的 CUDA 流。
- **目的**：通过 NCCL 实现跨所有 GPU 的分布式归约计算，验证数据一致性。

---

#### **7. 同步和清理资源**

```

for (int i=0; i<nDev; i++)

CUDACHECK(cudaStreamSynchronize(s[i]));

  

for (int i=0; i<nDev; i++) {

CUDACHECK(cudaFree(sendbuff[i]));

CUDACHECK(cudaFree(recvbuff[i]));

}

  

for (int i=0; i<nDev; i++) {

ncclCommDestroy(comms[i]);

}

  

MPICHECK(MPI_Finalize());

```

- **功能**：
- `cudaStreamSynchronize`：等待所有 CUDA 流中的操作完成，确保通信结果已写入 `recvbuff`。
- `cudaFree`：释放 GPU 上的内存。
- `ncclCommDestroy`：销毁 NCCL 通信器，释放资源。
- `MPI_Finalize`：结束 MPI 环境。
- **目的**：确保程序正确释放资源，避免内存泄漏或未完成的异步操作导致错误。

#### Case Study

在 **8 卡、4 进程、每个进程 2 卡** 为例：

##### **GPU Rank 的计算**

- **`nRanks = 4`**：总进程数（4 个进程）。
- **`nDev = 2`**：每个进程使用 2 个 GPU。
- **`localRank`**：当前进程在主机内的 GPU 编号（通过主机名哈希计算）。
- **示例**：
- 假设 `localRank = 0`，则 GPU 编号为 `0` 和 `1`。
- 假设 `localRank = 1`，则 GPU 编号为 `2` 和 `3`。
- 以此类推，总共有 8 个 GPU（0~7）。
- 每个进程的 GPU 编号由 `localRank` 和 `i` 决定，确保每个 GPU 在全局范围内有唯一的编号（0~7）。

##### **NCCL 通信器的分配**

- 每个进程为每个 GPU 创建一个 NCCL 通信器：

```

ncclCommInitRank(comms+i, nRanks*nDev, id, myRank*nDev + i);

```

- `nRanks*nDev = 4*2 = 8`：总通信节点数（每个 GPU 是一个独立节点）。
- `myRank*nDev + i`：当前 GPU 的全局 rank（`0~7`）。
- **示例**：
- 进程 0 的两个 GPU 对应全局 rank 0 和 1。
- 进程 1 的两个 GPU 对应全局 rank 2 和 3。
- 以此类推，进程 3 的两个 GPU 对应全局 rank 6 和 7。
- **组通信操作**：
- 使用 `ncclGroupStart()` 和 `ncclGroupEnd()` 包裹多 GPU 通信操作，确保线程安全。
- 调用 `ncclAllReduce` 时，每个 GPU 的通信器（`comms[i]`）和 CUDA 流（`s[i]`）独立工作。
- 总结
- 每个 GPU 有一个独立的 NCCL 通信器，全局 rank 范围为 `0~7`。
- 所有 GPU 通过统一的 NCCL ID 和全局 rank 加入同一个通信网络，支持跨节点和跨进程的高效通信。
- 每个 GPU 的全局 rank 为 `myRank * nDev + i`（`i = 0,1`）。
- 所有 8 个 GPU 的通信器通过统一的 NCCL ID 和全局 rank 组成通信网络。

#### **通信效果**

- **AllReduce 操作**：所有 8 个 GPU 的数据会被归约，最终每个 GPU 的 `recvbuff[i]` 存储所有 GPU 数据的总和（`1 * 8 * size`）。

# High-Level Architecture

1. Topology Sensing Implementation
2. Algorithm Selection After API Launch
3. Channel Building Based on Physical Connections
4. Kernel Setup and Launch

These four fundamental operations form a pipeline: topology sensing provides the hardware map, algorithm selection chooses optimal communication patterns, channel building establishes logical-to-physical mappings, and kernel setup/launch executes the actual communication work on the GPU. Each phase builds upon the previous one's output, creating a comprehensive system that automatically adapts to different hardware configurations.

# Initialization

## ncclGetUniqueId

### API

```c

ncclResult_t ncclGetUniqueId`(ncclUniqueId* _uniqueId_)

```

Generates an Id to be used in `ncclCommInitRank`. `ncclGetUniqueId` should be called once when creating a communicator and the Id should be distributed to all ranks in the communicator before calling `ncclCommInitRank`. _uniqueId_ should point to a `ncclUniqueId` object allocated by the user.

### 初始化

> [https://zhuanlan.zhihu.com/p/614746112](https://zhuanlan.zhihu.com/p/614746112)

### 流程

在 `ncclInit` 中：

1. `initEnv` 初始化环境变量；
2. `bootstrapNetInit` 初始化 bootstrap 网络，获取底层通信层（如 PCIe/NVLink 网络）生成的唯一标识符，用于初始化交换一些简单信息如机器 IP 端口，由于数据量小，只在初始化执行一次，使用 tcp；
3. `initGdrCopy` 初始化通信网络。

`[init.cc](http://init.cc/) 77-85` 的初始化代码主要完成 NCCL 环境准备、硬件加速配置和网络引导初始化。

1. `[param.cc](http://param.cc/) 52-68 ​`: `initEnv()`：环境变量初始化，解析并应用 NCCL 相关环境变量配置；
2. `gdrwrap.h 160-188 ​`: `initGdrCopy()`：GPU Direct RDMA 初始化；
3. `[bootstrap.cc](http://bootstrap.cc/) 92-129` ​: `bootstrapNetInit()`：引导网络初始化，遍历所有网卡信息，通过 `ncclFindInterfaceMatchSubnet`（定向匹配）和 `ncclFindInterfaces`（自动发现）。

### Bootstrap

> [https://zhuanlan.zhihu.com/p/620499558](https://zhuanlan.zhihu.com/p/620499558)

`bootstrapNetInit()` 初始化流程：

1. **环境变量检查**：优先读取 `NCCL_COMM_ID` 确定通信目标
2. **接口发现**：根据场景选择定向匹配或自动发现网络接口
3. **合法性验证**：确保至少有一个可用网络接口
4. **日志输出**：记录选中的接口名称和 IP 地址
5. **状态标记**：设置 `bootstrapNetInitDone=1` 避免重复初始化

网络接口选择机制：

- **筛选依据**：
- 子网匹配（优先选择与目标节点同子网的接口）
- 接口类型（优先 InfiniBand > 以太网）
- 用户配置（通过 `NCCL_SOCKET_IFNAME` 指定网卡）
- **容错处理**：至少需要 1 个可用接口，否则返回 `ncclSystemError`
- **传输层协议**：TCP（通过 `socket` API 实现）
- **地址格式**：支持 IPv 4（`x.x.x.x:port`）和 IPv 6（`[ipv6]:port`）
- **安全机制**：线程安全（通过 `pthread_mutex_lock` 实现单例初始化）
- **性能优化**：自动选择低延迟接口，避免跨子网通信

## Proxy

## ncclCommInitRank

### API

```cpp

ncclResult_t ncclCommInitRank(ncclComm_t* _comm_, int _nranks_, ncclUniqueId _commId_, int _rank_)

```

Creates a new communicator (multi thread/process version). _rank_ must be between 0 and _nranks_-1 and unique within a communicator clique. Each rank is associated to a CUDA device, which has to be set before calling `ncclCommInitRank`. `ncclCommInitRank` implicitly synchronizes with other ranks, hence it must be called by different threads/processes or used within ` ncclGroupStart/ncclGroupEnd`.

### Core Functionality

The `ncclCommInitRank` function in `[init.cc](http://init.cc/): 1848-1864` is responsible for **initializing a NCCL communicator** - the fundamental object that enables collective communication between GPUs.

This API call creates a new NCCL communication context (`ncclComm_t`) that connects multiple GPUs (ranks) into a single communication group. It's part of the **Communicator Creation and Management** section of the NCCL API.

### Critical Components

1. **Communicator Identity**

- `ncclUniqueId commId`: A global identifier that ensures all ranks join the same communication group
- `int nranks`: Total number of GPUs in the communicator
- `int myrank`: Current process's rank (0-based index within the communicator)

1. **Device Management**

- `cudaGetDevice(&cudaDev)`: Captures the current CUDA device context
- The communicator is bound to this device for all subsequent operations

1. **Configuration**

- `ncclConfig_t`: Contains communication parameters and optimizations
- Uses default initialization (`NCCL_CONFIG_INITIALIZER`) in this implementation

1. **Profiling Integration**

- `NVTX3_RANGE`: Marks the start/end of initialization for NVIDIA profiling tools
- `NVTX3_PAYLOAD`: Adds metadata (communicator hash, ranks, device) for performance analysis

### What Happens Internally

The actual initialization work is delegated to `ncclCommInitRankDev` in `[init.cc](http://init.cc/): 1848-1864`, which:

1. Validates input parameters and CUDA context
2. Creates communication endpoints between GPUs
3. Negotiates connection details using the provided `ncclUniqueId`
4. Initializes transport layers (NVLink, PCIe, or Ethernet for multi-node)
5. Allocates and initializes internal data structures
6. Returns a handle to the new communicator via `*newcomm`

The critical process follows a structured sequence:

1. **Configuration Parsing** - `parseCommConfig()` processes user configuration options
2. **Communicator Allocation** - `commAlloc()` creates the basic communicator structure
3. **Bootstrap Initialization** - `bootstrapInit()` or `bootstrapSplit()` establishes inter-rank coordination
4. **Transport Setup** - `initTransportsRank()` discovers topology and initializes transport connections
5. **Device Setup** - `devCommSetup()` creates device-side communicator structures

## Topology Sensing Implementation

> topogetsystem: [https://zhuanlan.zhihu.com/p/625606436](https://zhuanlan.zhihu.com/p/625606436)

> topocompute: [NCCL 源码解析⑤：路径计算]([https://mp.weixin.qq.com/s?__biz=MzU5ODY2MTk3Nw==&mid=2247491866&idx=1&sn=4535a2bd3eff0c45c3933845a31d5862&chksm=fe426f2cc935e63a6fec1b075cfe2ef66ab1a24e714b964e6f931eed37868f58f53b4bb591f0&scene=21#wechat_redirect](https://mp.weixin.qq.com/s?__biz=MzU5ODY2MTk3Nw==&mid=2247491866&idx=1&sn=4535a2bd3eff0c45c3933845a31d5862&chksm=fe426f2cc935e63a6fec1b075cfe2ef66ab1a24e714b964e6f931eed37868f58f53b4bb591f0&scene=21#wechat_redirect))

```cpp

// init.cc:814-826

// Topo detection / System graph creation

NCCLCHECKGOTO(ncclTopoGetSystem(comm, &comm->topo), ret, fail);

// Compute paths between GPUs and NICs

NCCLCHECKGOTO(ncclTopoComputePaths(comm->topo, comm), ret, fail);

// Remove inaccessible GPUs and unused NICs

NCCLCHECKGOTO(ncclTopoTrimSystem(comm->topo, comm), ret, fail);

// Recompute paths after trimming

NCCLCHECKGOTO(ncclTopoComputePaths(comm->topo, comm), ret, fail);

// Init search

NCCLCHECKGOTO(ncclTopoSearchInit(comm->topo), ret, fail);

// Decide on comm's CPU architecture.

NCCLCHECKGOTO(ncclTopoComputeCommCPU(comm), ret, fail);

// Print final topology

NCCLCHECKGOTO(ncclTopoPrint(comm->topo), ret, fail);

```

Topology detection happens during communicator initialization in `init.cc:814-826` of the function `initTransportsRank`. The core topology system is implemented in:

- **System Discovery**: `[topo.cc](http://topo.cc/) ncclTopoGetSystem` handles the main topology detection via `ncclTopoGetSystem()`, which reads XML files, detects GPUs/NICs, and builds the system graph
- **Path Computation**: `[paths.cc](http://paths.cc/) ncclTopoComputePaths` computes optimal paths between all GPU and network device pairs
- **Search Initialization**: `[search.cc](http://search.cc/) ncclTopoSearchInit` prepares the topology for algorithm selection by computing bandwidth matrices and connection patterns

This topology sensing is topology-agnostic and happens regardless of which communication APIs you'll later call.

## Channel Building Based on Physical Connections

> [https://zhuanlan.zhihu.com/p/653440728](https://zhuanlan.zhihu.com/p/653440728)

> [https://zhuanlan.zhihu.com/p/658868934](https://zhuanlan.zhihu.com/p/658868934)

Channel setup happens in two phases:

**Graph Computation**: init. cc: 847-866 computes communication graphs for each algorithm (ring, tree, CollNet, NVLS) using `ncclTopoCompute()`

**Channel Connection**: `ncclTopoPreset()` in `connect.cc` maps the computed graphs to actual channel structures, setting up ring/tree topologies and peer connections

**Transport Setup**: `initTransportsRank` in `init.cc` establishes the actual physical connections:

- Ring connections via `ncclTransportRingConnect()`
- Tree connections via `ncclTransportTreeConnect()`
- NVLS setup via `ncclNvlsSetup()`
- CollNet setup via `ncclCollNetSetup()`

# The Implementation of the Communication API

## `ncclAllReduce` API

All the API are implemented in `src\collectives.cc`. Take `ncclAllReduce` as an example.

```cpp

ncclResult_t`ncclAllReduce`(const void* _sendbuff_, void* _recvbuff_, size_t _count_, ncclDataType_t _datatype_, ncclRedOp_t _op_, ncclComm_t _comm_, cudaStream_t _stream_)

```

Reduces data arrays of length `count` in `sendbuff` using the `op` operation and leaves identical copies of the result in each `recvbuff`.

In-place operation will happen if `sendbuff == recvbuff`.

## NVTX-related Function

The NVTX-related function calls in `collectives.cc` lines 91-96 serve as **profiling and instrumentation markers** for NVIDIA's performance analysis tools. Here's a detailed breakdown:

### Function: `NVTX3_FUNC_WITH_PARAMS`

This macro creates **NVIDIA Tools Extension (NVTX) markers** that integrate with profiling tools like NVIDIA Nsight Systems. The specific call in your code:

```

NVTX3_FUNC_WITH_PARAMS(AllReduce, NcclNvtxParamsAllReduce,

NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), op));

```

### Key Purposes

1. **Performance Tracing**

- Marks the start/end of the `ncclAllReduce` operation in profiling timelines
- Enables precise measurement of execution duration for collective operations
- Helps identify bottlenecks in multi-GPU communication workflows

1. **Payload Metadata** The `NVTX3_PAYLOAD` includes critical context for analysis:

- `comm->commHash`: Unique identifier for the communication group
- `count * ncclTypeSize(datatype)`: Total data size transferred
- `op`: The reduction operation type (e.g., sum, max)

1. **Tool Integration** These markers are visible in:

- NVIDIA Nsight Systems (for timeline analysis)
- NVIDIA Visual Profiler
- Other CUDA-aware profiling tools They enable correlation between application logic and low-level hardware behavior.

### Implementation Context

This instrumentation is lightweight and conditional-compiled (typically only enabled in debug/profiling builds). It has minimal runtime overhead while providing invaluable insights for optimizing collective communication performance in distributed GPU applications.

## Enqueue

The `ncclEnqueueCheck` function in `enqueue.cc` (lines 2404-2437) is a critical component of NCCL's collective operation execution pipeline. Its primary role is to **validate, log, and queue collective communication operations** for execution on the GPU stream. Here's a detailed breakdown of its functionality:

### 1. **Group Operation Management**

```

NCCLCHECK(ncclGroupStartInternal());

// ... operation processing ...

ncclGroupErrCheck(ret);

NCCLCHECK(ncclGroupEndInternal());

```

- Wraps the operation in `ncclGroupStartInternal()` / `ncclGroupEndInternal()` to support **batched collective operations**
- Ensures proper error propagation across grouped operations via `ncclGroupErrCheck()`

### 2. **Pre-Execution Validation**

The function performs multiple layers of validation before queuing the operation:

```

NCCLCHECKGOTO(CommCheck(info->comm, info->opName, "comm"), ret, fail);

NCCLCHECKGOTO(ncclCommEnsureReady(info->comm), ret, fail);

if (info->comm->checkPointers) { /* Device pointer validation */ }

NCCLCHECKGOTO(ArgsCheck(info), ret, fail);

```

- Validates communicator state with `CommCheck()` and `ncclCommEnsureReady()`
- Checks device pointers if enabled (`checkPointers` flag)
- Validates operation parameters (buffer addresses, count, datatype, etc.) via `ArgsCheck()`

### 3. **Operation Logging & Tracing**

```

INFO(NCCL_COLL,"%s: opCount %lx sendbuff %p recvbuff %p count %zu ...", info->opName, ...);

TRACE_CALL("nccl%s(%" PRIx64 ",%" PRIx64 ",%zu,%d,%d,%d,%p,%p)", ...);

```

- Logs operation details to NCCL's debugging infrastructure (`INFO` macro)
- Creates trace events for performance analysis tools via `TRACE_CALL()`
- Captures critical metadata: operation type, buffer addresses, data size, communicator properties

### 4. **Task Queuing**

```

NCCLCHECKGOTO(taskAppend(info->comm, info), ret, fail);

```

- Adds the validated operation to the communicator's execution queue via `taskAppend()`
- This queues the operation for execution on the specified CUDA stream

### 5. **Device Context Management**

```

if (info->comm->checkPointers) {

CUDACHECKGOTO(cudaGetDevice(&devOld), ret, fail);

CUDACHECKGOTO(cudaSetDevice(info->comm->cudaDev), ret, fail);

}

// ...

exit:

if (devOld != -1) CUDACHECK(cudaSetDevice(devOld));

```

- Temporarily switches to the communicator's target GPU device when validating pointers
- Restores the original device context before returning

### 6. **Error Handling**

- Uses `NCCLCHECKGOTO` / `CUDACHECKGOTO` macros for centralized error handling
- Sets asynchronous error state on communicator for non-blocking operations:

```

if (info->comm && !info->comm->config.blocking) {

(void) ncclCommSetAsyncError(info->comm, ret);

}

```

- Retrieves async error status before returning with `ncclCommGetAsyncError()`

### Overall Purpose

`ncclEnqueueCheck` acts as a **gatekeeper** for collective operations, ensuring only valid operations are queued for execution. It bridges the high-level API with the low-level execution engine by:

1. Validating operation prerequisites
2. Managing execution context
3. Enabling performance tracing
4. Ensuring proper error propagation
5. Integrating with NCCL's batching/grouping mechanism

This function is critical for maintaining reliability and performance in multi-GPU communication scenarios.

# Task Plan

## Algorithm Selection After API Launch

Algorithm selection occurs during task preparation in enqueue. cc: 374-440 . The process flows:

1. **Task Binning**: Operations are grouped by `(function, operation, datatype)` tuples
2. **Algorithm Selection**: enqueue. cc: 1883-1940 calls `getAlgoInfo()` which:

- Generates a cost table via `updateCollCostTable()`
- Uses enqueue. cc: 1791-1877 `topoGetAlgoInfo()` to find the minimum-cost algorithm-protocol combination
- Applies channel and thread count optimizations based on message size

The cost model is built during initialization in tuning. cc: 213-320 via `ncclTopoTuneModel()`, which computes bandwidth and latency parameters for all algorithm-protocol combinations.

## Kernel Setup and Launch Details

Kernel preparation and launch happens in enqueue. cc: 1393-1490 :

**Plan Creation**: `ncclLaunchPrepare()` creates kernel plans by:

- Scheduling collective tasks via `scheduleCollTasksToPlan()`
- Scheduling P 2 P tasks via `scheduleP2pTasksToPlan()`
- Finalizing plans with `finishPlan()`

**Kernel Execution**: The actual GPU kernel is launched through `doLaunches()` which calls `ncclLaunchKernel()` to execute the prepared plans.

**Device-Side Execution**: common. h: 331-402 shows the kernel main function `ncclKernelMain()` which:

- Loads kernel arguments and channel information to shared memory
- Maps block indices to channel IDs using the channel mask
- Executes the appropriate device function based on the function ID
- Handles work batch processing and synchronization

The kernel receives work through comm. h: 420-507 the communicator structure which contains all channel configurations, peer information, and topology data needed for execution.

# Kernel Launch

After `ncclGroupEnd()` completes, the kernel launch happens through a specific execution pipeline:

### 1. Group End Triggers Execution

When `ncclGroupEnd()` is called, it internally calls `ncclGroupEndInternal()` group. cc: 101-108 . This function decrements the group depth and when it reaches 0, triggers the actual execution.

### 2. Group Launch Coordination

The execution flows through `groupLaunch()` group. cc: 447-598 , which coordinates the entire launch process. This function handles:

- Preconnection jobs for transport setup
- Symmetric memory registration
- Collective operation preparation
- The actual kernel launch via `doLaunches()`

### 3. Kernel Launch Function

The core kernel launching happens in `doLaunches()` group. cc: 252-318 . This function:

- Iterates through communicator cliques
- Calls `ncclLaunchPrepare()` to prepare each communicator
- Executes `ncclLaunchKernel()` for each kernel plan
- Handles round-based execution for multiple kernel launches

### 4. Individual Kernel Launch

The actual GPU kernel launch occurs in `ncclLaunchKernel()` enqueue. cc: 1539-1635 . This function:

- Sets up CUDA kernel launch parameters (grid, block dimensions)
- Configures advanced features like cluster scheduling and memory sync domains
- Calls `cuLaunchKernelEx()` or `cuLaunchKernel()` to actually launch the GPU kernel

### 5. Launch Preparation and Finalization

The launch process also involves:

- `ncclLaunchPrepare()` enqueue. cc: 1393-1522 - prepares kernel plans and manages stream dependencies
- `ncclLaunchFinish()` enqueue. cc: 1662-1719 - handles post-launch cleanup and stream synchronization

## Execution Flow Summary

The complete flow is:

1. `ncclGroupEnd()` → `ncclGroupEndInternal()`
2. `groupLaunch()` coordinates the entire process
3. `doLaunches()` manages communicator-level execution
4. `ncclLaunchKernel()` performs the actual GPU kernel launch

## Notes

The kernel launch is the final step in a complex orchestration process that includes task preparation, transport setup, and resource management. The `doLaunches()` function is the key orchestrator that bridges between the group management system and the actual GPU kernel execution, handling multiple communicators and ensuring proper synchronization across all participating ranks.

# Group Calls

## Internal Vs External Group Calls

Here what should be clarified is the relationship between the internal `groupStart` / `groupEnd` calls that happen automatically when launching communication operations like `ncclAllReduce`, versus the explicit `ncclGroupStart` / `ncclGroupEnd` calls that users make when managing multiple devices from a single thread.

### Internal Group Calls (Automatic)

When you call a single communication operation like `ncclAllReduce`, NCCL automatically wraps it with internal group calls [1](#4-0). The `ncclEnqueueCheck` function calls:

1. `ncclGroupStartInternal()` at the beginning
2. `ncclGroupEndInternal()` at the end

This creates a minimal group context for the single operation, allowing it to use the same execution infrastructure as batched operations.

### External Group Calls (User-Initiated)

When you explicitly call `ncclGroupStart()` and `ncclGroupEnd()`, you're creating a batching context for multiple operations `enqueue.cc:2407-2432`. The public APIs delegate to the same internal functions:

- `ncclGroupStart()` calls `ncclGroupStartInternal()`
- `ncclGroupEnd()` calls `ncclGroupEndInternal()`

## Group Depth and Nesting

The key difference is managed through `ncclGroupDepth` `nccl.h.in:493-504`:

### Single Operation (Internal Groups)

- `ncclGroupStartInternal()` increments `ncclGroupDepth` to 1
- The operation is enqueued via `taskAppend()`
- `ncclGroupEndInternal()` decrements depth back to 0, triggering immediate execution

### Multi-Device Batching (External Groups)

- User calls `ncclGroupStart()`, setting `ncclGroupDepth` to 1
- Multiple operations are enqueued but not executed (depth > 0)
- User calls `ncclGroupEnd()`, decrementing depth to 0 and triggering batch execution

## Task Collection and Execution

Both scenarios use the same underlying infrastructure:

### Task Collection

Operations are collected in thread-local storage through `ncclGroupCommJoin()` `group.h:92-119`. Each communicator joins the group and gets organized by task type and clique membership.

### Execution Coordination

When `ncclGroupDepth` reaches 0, `ncclGroupEndInternal()` triggers the execution pipeline `group.cc:604-714`:

1. **Task Preparation**: `ncclPrepareTasks()` processes collected operations
2. **Transport Setup**: Preconnection jobs establish channels if needed
3. **Kernel Launch**: `doLaunches()` coordinates execution across communicators

## Multi-Device Management Benefits

The explicit group calls provide several advantages for single-thread multi-device scenarios:

1. **Reduced Synchronization**: Operations are batched and launched together, reducing CPU-GPU synchronization overhead
2. **Kernel Fusion**: Multiple operations can be fused into fewer kernel launches
3. **Deadlock Prevention**: For operations like `ncclSend` / `ncclRecv` that require coordination, grouping ensures they can progress concurrently

## Notes

The internal and external group calls use identical infrastructure - the only difference is who initiates the grouping and when execution is triggered. Single operations get automatic minimal grouping for consistency, while explicit grouping enables batching optimizations for multi-device scenarios. The `ncclGroupDepth` mechanism ensures proper nesting and prevents premature execution of incomplete operation batches.

Wiki pages you might want to explore:

- [Core Architecture (NVIDIA/nccl)](/wiki/NVIDIA/nccl#2)
- [Bootstrap and Group Management (NVIDIA/nccl)](/wiki/NVIDIA/nccl#2.3)

# Top-down

```

topo() + connection(transport: nvlink/rdma/shmem) + operation

|

|

algorithm

|

|

graph

|

|

(kernel)

```
