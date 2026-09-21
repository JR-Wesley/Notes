NCCL 基础解读：https://aijishu.com/a/1060000000483892

NCCL 解读 https://zhuanlan.zhihu.com/p/1932137763840458794


GPU d2d  https://zhuanlan.zhihu.com/p/2847929235
# NCCL UG

https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html

## Overview of NCCL

来源：[NVIDIA NCCL User Guide — Overview of NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html#overview-of-nccl)。

### 中英文对照

> The NVIDIA Collective Communications Library (NCCL, pronounced “Nickel”) is a library providing inter-GPU communication primitives that are topology-aware and can be easily integrated into applications.

NVIDIA 集合通信库（NCCL，读作“Nickel”）提供具有拓扑感知能力的 GPU 间通信原语，并且易于集成到应用程序中。

> NCCL implements both collective communication and point-to-point send/receive primitives. It is not a full-blown parallel programming framework; rather, it is a library focused on accelerating inter-GPU communication.

NCCL 同时实现集合通信和点对点发送／接收原语。它不是完整的并行编程框架，而是专注于加速 GPU 间通信的库。

> NCCL provides the following collective communication primitives:

NCCL 提供以下集合通信原语：

> - AllReduce
> - Broadcast
> - Reduce
> - AllGather
> - ReduceScatter
> - AlltoAll
> - Gather
> - Scatter

- AllReduce（全归约）
- Broadcast（广播）
- Reduce（归约）
- AllGather（全收集）
- ReduceScatter（归约散发）
- AlltoAll（全交换）
- Gather（收集）
- Scatter（散发）

> Additionally, it allows for point-to-point send/receive communication which allows for scatter, gather, or all-to-all operations.

此外，NCCL 也支持点对点发送／接收通信，可利用它们实现 scatter、gather 或 all-to-all 操作。

> Tight synchronization between communicating processors is a key aspect of collective communication. CUDA based collectives would traditionally be realized through a combination of CUDA memory copy operations and CUDA kernels for local reductions. NCCL, on the other hand, implements each collective in a single kernel handling both communication and computation operations. This allows for fast synchronization and minimizes the resources needed to reach peak bandwidth.

参与通信的处理器之间需要紧密同步，这是集合通信的关键。传统的 CUDA 集合通信通常结合 CUDA 内存拷贝操作与执行本地归约的 CUDA kernel。NCCL 则在单个 kernel 中实现每项集合操作，同时处理通信和计算，因此能够快速同步，并减少达到峰值带宽所需的资源。

> NCCL conveniently removes the need for developers to optimize their applications for specific machines. NCCL provides fast collectives over multiple GPUs both within and across nodes. It supports a variety of interconnect technologies including PCIe, NVLINK, InfiniBand Verbs, and IP sockets.

NCCL 免去开发者为特定机器逐一优化应用程序的负担。它为节点内和跨节点的多个 GPU 提供快速集合通信，并支持 PCIe、NVLINK、InfiniBand Verbs 和 IP sockets 等互连技术。

> Next to performance, ease of programming was the primary consideration in the design of NCCL. NCCL uses a simple C API, which can be easily accessed from a variety of programming languages. NCCL closely follows the popular collectives API defined by MPI (Message Passing Interface). Anyone familiar with MPI will thus find NCCL’s API very natural to use. In a minor departure from MPI, NCCL collectives take a “stream” argument which provides direct integration with the CUDA programming model. Finally, NCCL is compatible with virtually any multi-GPU parallelization model, for example:

除了性能，易于编程也是 NCCL 设计时的主要考量。NCCL 使用简洁的 C API，便于多种编程语言调用。它紧密遵循 MPI（Message Passing Interface）定义的常用集合通信 API，因此熟悉 MPI 的人会觉得 NCCL API 很自然。与 MPI 略有不同，NCCL 集合操作接收一个 `stream` 参数，可直接融入 CUDA 编程模型。NCCL 几乎兼容任何多 GPU 并行化模型，例如：

> - single-threaded control of all GPUs
> - multi-threaded, for example, using one thread per GPU
> - multi-process, for example, MPI

- 单线程控制所有 GPU
- 多线程，例如每个 GPU 使用一个线程
- 多进程，例如 MPI

> NCCL has found great application in Deep Learning Frameworks, where the AllReduce collective is heavily used for neural network training. Efficient scaling of neural network training is possible with the multi-GPU and multi-node communication provided by NCCL.

NCCL 已广泛用于深度学习框架，其中 AllReduce 大量用于神经网络训练。NCCL 提供的多 GPU 和多节点通信使神经网络训练能够高效扩展。

### 重点解读

**定位与边界。** 这段概述把 NCCL 定位为 GPU 间通信库：应用或框架负责训练任务、进程组织及计算逻辑，NCCL 提供通信原语。这里的“拓扑感知”意味着通信实现会考虑 GPU 与互连的连接关系；它不能单独保证任意硬件、消息规模和程序配置下都达到最优性能。

**两类通信原语。** 集合通信描述一组参与者共同完成的数据交换或归约，点对点通信则明确发送方与接收方。原文同时列出 Scatter、Gather、AlltoAll 集合原语，又说可以用 Send/Recv 实现这些模式：前者是直接提供的集合操作，后者是按需组合点对点操作来表达相同的数据流，并不表示两种写法的调用方式或性能相同。

| 原语 | 数据如何流动 | 每个参与者最终得到什么 |
| --- | --- | --- |
| AllReduce | 所有参与者的数据参与同一归约 | 完整的归约结果 |
| Broadcast | 一个根参与者向所有参与者发送数据 | 根参与者的数据 |
| Reduce | 所有参与者的数据归约到根参与者 | 只有根参与者得到完整归约结果 |
| AllGather | 汇集每个参与者的数据并发给所有参与者 | 拼接后的完整数据 |
| ReduceScatter | 先归约，再把结果分片分发 | 归约结果的一个分片 |
| AlltoAll | 每个参与者向所有参与者发送各自对应的数据片 | 来自所有参与者、发给自己的数据片 |
| Gather | 汇集所有参与者的数据到根参与者 | 只有根参与者得到汇集结果 |
| Scatter | 根参与者把数据分片发给各参与者 | 各自对应的数据片 |

**通信与计算放在一起。** 原文强调同步开销：若通信拷贝和本地归约拆成多个 CUDA 操作，就需要协调这些操作的执行。NCCL 所述的单 kernel 实现把数据移动、归约及参与者间的同步放在同一操作中，因而有利于减少协调开销和提高带宽利用率。这是文档对设计思路的概述，不应理解为所有场景都只发生一次设备端执行，或所有消息规模都能达到峰值带宽。

**硬件适配与 CUDA stream。** NCCL 面向单机多 GPU 和跨节点场景，列出的 PCIe、NVLINK、InfiniBand Verbs 与 IP sockets 说明它需要适配不同通信路径。API 中的 `stream` 参数指定操作进入哪个 CUDA stream，使调用方能够把 NCCL 操作放进已有的 CUDA 执行顺序；实际计算与通信能否重叠，仍取决于依赖关系与资源条件。

**为什么训练常用 AllReduce。** 数据并行训练中，每个参与者可先计算本地梯度，再用 AllReduce 汇总梯度，使所有参与者获得一致的归约结果。这解释了它为何是训练框架中的常见操作；原文所说的“高效扩展”仍需要结合计算量、通信量和实际拓扑判断。

## Setup

来源：[NVIDIA NCCL User Guide — Setup](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/setup.html#setup)。

### 中英文对照

> NCCL is a communication library providing optimized GPU-to-GPU communication for high-performance applications. It is not, like MPI, providing a parallel environment including a process launcher and manager. NCCL relies therefore on the application’s process management system and CPU-side communication system for its own bootstrap.

NCCL 是为高性能应用提供优化 GPU 间通信的通信库。它不像 MPI 那样提供包含进程启动器和管理器的并行环境。因此，NCCL 依赖应用的进程管理系统和 CPU 侧通信系统完成自身的 bootstrap（初始化引导）。

> Similarly to MPI and other libraries which are optimized for performance, NCCL does not provide secure network communication between GPUs. It is therefore the responsibility of the user to ensure NCCL operates over a secure network, both for bootstrap (controlled by [`NCCL_SOCKET_IFNAME`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-socket-ifname)) and for high-speed communication.

与 MPI 等注重性能的库类似，NCCL 不提供 GPU 间安全网络通信。因此，用户需要确保 NCCL 运行在安全网络上：既要保护 bootstrap 所用的网络（由 [`NCCL_SOCKET_IFNAME`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-socket-ifname) 控制），也要保护高速通信所用的网络。

### 重点解读

**进程管理和通信库的分工。** 启动并管理各个进程、让它们在初始化阶段交换必要信息，是应用及其 CPU 侧运行环境需要解决的问题。NCCL 提供 GPU 间通信能力，但不会替应用启动整组工作进程。这里的 bootstrap 指通信正式开始前的引导过程，不能与后续传输业务数据的 GPU 间通信混为一谈。

**安全边界覆盖两条路径。** 原文明确说 NCCL 本身不提供 GPU 间安全网络通信，因此网络隔离和保护需要由部署环境负责。`NCCL_SOCKET_IFNAME` 涉及 bootstrap 使用的网络接口；配置它并不等于保护了高速数据通信路径，部署时应分别识别和保护这两部分网络。这是对原文安全责任划分的解读，不暗示 NCCL 自动提供加密或身份认证。

## Using NCCL

来源：[NVIDIA NCCL User Guide — Using NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage.html#using-nccl)。

### 中英文对照

> Using NCCL is similar to using any other library in your code:

在代码中使用 NCCL 与使用其他库类似：

> 1. Install the NCCL library on your system
> 2. Modify your application to link to that library
> 3. Include the header file `nccl.h` in your application
> 4. Create a communicator (see [Creating a Communicator](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#communicator-label))
> 5. Use NCCL collective communication primitives to perform data communication. You can familiarize yourself with the [NCCL API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html#api-label) documentation to maximize your usage performance.

1. 在系统上安装 NCCL 库。
2. 修改应用程序，使其链接该库。
3. 在应用程序中包含头文件 `nccl.h`。
4. 创建 communicator（参见 [Creating a Communicator](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#communicator-label)）。
5. 用 NCCL 集合通信原语传输数据。可以阅读 [NCCL API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html#api-label) 文档，以更充分地发挥性能。

> Collective communication primitives are common patterns of data transfer among a group of CUDA devices. A communication algorithm involves many processors that are communicating together. Each CUDA device is identified within the communication group by a zero-based index or rank. Each rank uses a communicator object to refer to the collection of GPUs that are intended to work together. The creation of a communicator is the first step needed before launching any communication operation.

集合通信原语是一组 CUDA 设备之间常见的数据传输模式。一个通信算法涉及多个共同通信的处理器。在通信组内，每个 CUDA 设备由从零开始的索引（即 rank）标识。每个 rank 使用一个 communicator 对象指代需要协同工作的 GPU 集合。在发起任何通信操作之前，首先要创建 communicator。

### 重点解读

**使用顺序。** 前三步解决库是否可用以及代码能否调用其接口；创建 communicator 后，参与者才有共同的通信组上下文，随后才能发起通信操作。此处只给出接入流程，没有提供具体的初始化参数或可直接运行的程序。

**rank 与 communicator 的关系。** rank 是某个通信组内部从 0 开始的参与者编号，用来区分该组内的 CUDA 设备；它不应被直接当作操作系统进程号或全局 GPU 编号。communicator 则表示这组共同通信的 GPU，以及本 rank 在该组中的通信上下文。原文强调先创建 communicator，是因为后续操作需要明确哪些参与者属于同一组。

**集合操作的参与范围。** 集合通信是组内多个参与者共同执行的模式，不能只从单个 GPU 的本地调用理解其数据流。上面的五步说明以集合通信为例；前文 Overview 还指出 NCCL 另有点对点 Send/Recv 原语。

## Creating a Communicator

来源：[NVIDIA NCCL User Guide — Creating a Communicator](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#communicator-label)。

### 中英文对照

> When creating a communicator, a unique rank between 0 and n-1 has to be assigned to each of the n CUDA devices which are part of the communicator. Using the same CUDA device multiple times as different ranks of the same NCCL communicator is not supported and may lead to hangs.

创建 communicator 时，属于该 communicator 的 $n$ 个 CUDA 设备必须分别分配 $0$ 到 $n-1$ 的唯一 rank。不支持在同一个 NCCL communicator 中把同一个 CUDA 设备用作多个不同 rank，否则可能导致挂起。

> Given a static mapping of ranks to CUDA devices, the `ncclCommInitRank()`, `ncclCommInitRankConfig()` and `ncclCommInitAll()` functions will create communicator objects, each communicator object being associated to a fixed rank and CUDA device. Those objects will then be used to launch communication operations.

当 rank 到 CUDA 设备的映射固定时，`ncclCommInitRank()`、`ncclCommInitRankConfig()` 和 `ncclCommInitAll()` 会创建 communicator 对象，每个对象固定关联一个 rank 和一个 CUDA 设备。这些对象随后用于发起通信操作。

> Before calling `ncclCommInitRank()`, you need to first create a unique object which will be used by all processes and threads to synchronize and understand they are part of the same communicator. This is done by calling the `ncclGetUniqueId()` function.

调用 `ncclCommInitRank()` 之前，必须先创建一个唯一对象，供所有参与的进程和线程同步，并确认它们属于同一个 communicator。通过调用 `ncclGetUniqueId()` 来创建该对象。

> The `ncclGetUniqueId()` function returns an ID which has to be broadcast to all participating threads and processes using any CPU communication system, for example, passing the ID pointer to multiple threads, or broadcasting it to other processes using MPI or another parallel environment using, for example, sockets.

`ncclGetUniqueId()` 返回的 ID 必须借助任意 CPU 侧通信系统分发给所有参与的线程和进程。例如，把 ID 指针传给多个线程，或在 MPI 等并行环境中通过 socket 等方式广播给其他进程。

> You can also call the `ncclCommInitAll` operation to create n communicator objects at once within a single process. As it is limited to a single process, this function does not permit inter-node communication. `ncclCommInitAll` is equivalent to calling a combination of `ncclGetUniqueId` and `ncclCommInitRank`.

也可以在单个进程中调用 `ncclCommInitAll()`，一次创建 $n$ 个 communicator 对象。由于它仅限单进程，因此这种调用不能用于跨节点通信。`ncclCommInitAll()` 等价于组合调用 `ncclGetUniqueId()` 和 `ncclCommInitRank()`。

> The following sample code is a simplified implementation of `ncclCommInitAll`.

下列示例代码是 `ncclCommInitAll()` 的简化实现。

```c
ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist) {
  ncclUniqueId Id;
  ncclGetUniqueId(&Id);
  ncclGroupStart();
  for (int i=0; i<ndev; i++) {
    cudaSetDevice(devlist[i]);
    ncclCommInitRank(comm+i, ndev, Id, i);
  }
  ncclGroupEnd();
}
```

> Related links:
> `ncclCommInitAll()`
> `ncclGetUniqueId()`
> `ncclCommInitRank()`

相关链接：`ncclCommInitAll()`、`ncclGetUniqueId()`、`ncclCommInitRank()`。

#### Using MIG instances｜使用 MIG 实例

> (since 2.31) (experimental)

（自 2.31 起，实验性功能。）

> Starting with NCCL 2.31, a communicator can use multiple Multi-Instance GPU (MIG) instances. Each rank must select a distinct MIG instance as its CUDA device; multiple ranks using the same MIG instance are not supported. Because each MIG instance is a distinct CUDA device, this configuration does not require `NCCL_MULTI_RANK_GPU_ENABLE`.

从 NCCL 2.31 开始，一个 communicator 可以使用多个 Multi-Instance GPU（MIG）实例。每个 rank 必须选择不同的 MIG 实例作为其 CUDA 设备；不支持多个 rank 使用同一个 MIG 实例。由于每个 MIG 实例都是独立的 CUDA 设备，这种配置不需要 `NCCL_MULTI_RANK_GPU_ENABLE`。

> Since MIG does not support NVLink P2P from CUDA driver, Multi-Node NVLink (MNNVL) needs to be disabled when using NCCL with MIG: `NCCL_MNNVL_ENABLE=0`.

由于 CUDA 驱动不支持 MIG 的 NVLink P2P，NCCL 配合 MIG 使用时必须关闭 Multi-Node NVLink（MNNVL）：`NCCL_MNNVL_ENABLE=0`。

> NCCL will not use NVLink SHARP because NVLS is not supported with MIG.

NCCL 不会使用 NVLink SHARP，因为 MIG 不支持 NVLS。

> Requires CUDA driver version 13.0 or greater.

要求 CUDA 驱动版本为 13.0 或更高。

> Warning

**警告**

> Note that MIG support in NCCL is experimental and not thoroughly tested with large scale deployment. In the 2.31 release, we have validated up to 8 GPUs with 7 MIG instances on each GPU and running the typical nccl-tests collectives. Given the massive validation combinations possible with this feature, we will keep expanding test coverage in future releases.

请注意，NCCL 的 MIG 支持仍是实验性功能，尚未经过大规模部署的充分测试。2.31 版本已验证的范围是最多 8 块 GPU、每块 GPU 7 个 MIG 实例，并运行常见的 `nccl-tests` 集合操作。由于该功能可能有大量配置组合，后续还会扩大测试覆盖范围。

> For information about configuring MIG instances and selecting them as CUDA devices, see the NVIDIA Multi-Instance GPU User Guide.

有关配置 MIG 实例并将其选为 CUDA 设备的方法，请参阅 NVIDIA Multi-Instance GPU User Guide。

#### Creating a communicator with options｜带选项创建 communicator

> The `ncclCommInitRankConfig()` function allows creating a NCCL communicator with specific options.

`ncclCommInitRankConfig()` 允许使用指定选项创建 NCCL communicator。

> The config parameters NCCL supports are listed here `ncclConfig_t`.

NCCL 支持的配置参数列于 `ncclConfig_t`。

> For example, “blocking” can be set to 0 to ask NCCL to never block in any NCCL call, and at the same time other config parameters can be set as well to more precisely define communicator behavior. A simple example code is shown below:

例如，可以把 `blocking` 设为 0，要求 NCCL 调用不阻塞；同时还可设置其他配置参数，更精确地定义 communicator 行为。下方是一个简单的代码示例。

```c
ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
config.blocking = 0;
config.minCTAs = 4;
config.maxCTAs = 16;
config.cgaClusterSize = 2;
config.netName = "Socket";
CHECK(ncclCommInitRankConfig(&comm, nranks, id, rank, &config));
do {
  CHECK(ncclCommGetAsyncError(comm, &state));
  // Handle outside events, timeouts, progress, ...
} while(state == ncclInProgress);
```

> Related link: `ncclCommGetAsyncError()`

相关链接：`ncclCommGetAsyncError()`。

#### Creating a communicator using multiple ncclUniqueIds｜使用多个 ncclUniqueId 创建 communicator

> The `ncclCommInitRankScalable()` function enables the creation of a NCCL communicator using many `ncclUniqueIds`. All NCCL ranks have to provide the same array of `ncclUniqueIds` (same `ncclUniqueIds`, and in the same order). For the best performance, we recommend distributing the `ncclUniqueIds` as evenly as possible amongst the NCCL ranks.

`ncclCommInitRankScalable()` 能用多个 `ncclUniqueId` 创建 NCCL communicator。所有 NCCL rank 都必须提供相同的 `ncclUniqueId` 数组，且顺序也必须一致。为了获得较好的性能，建议尽可能均匀地把这些 ID 分布到各 NCCL rank。

> Internally, NCCL ranks will mostly communicate with a single `ncclUniqueId`. Therefore, to obtain the best results, we recommend to evenly distribute `ncclUniqueIds` across the ranks.

在内部，每个 NCCL rank 大部分时候只与一个 `ncclUniqueId` 通信。因此，为取得较好的结果，建议把这些 ID 均匀分配给各 rank。

> The following function can be used to decide if a NCCL rank should create a `ncclUniqueIds`:

可以用下列函数决定某个 NCCL rank 是否应创建一个 `ncclUniqueId`：

```c
bool rankHasRoot(const int rank, const int nRanks, const int nIds) {
  const int rmr = nRanks % nIds;
  const int rpr = nRanks / nIds;
  const int rlim = rmr * (rpr+1);
  if (rank < rlim) {
    return !(rank % (rpr + 1));
  } else {
    return !((rank - rlim) % rpr);
  }
}
```

> For example, if 3 `ncclUniqueIds` are to be distributed across 7 NCCL ranks, the first `ncclUniqueId` will be associated to ranks 0-2, while the others will be associated to ranks 3-4, and 5-6. This function will therefore return true on rank 0, 3, and 5, and false otherwise.

例如，要把 3 个 `ncclUniqueId` 分配给 7 个 NCCL rank，第一个 ID 对应 rank 0–2，其余两个分别对应 rank 3–4 和 rank 5–6。因此，该函数在 rank 0、3、5 上返回 `true`，在其他 rank 上返回 `false`。

> Note: only the first `ncclUniqueId` will be used to create the communicator hash id, which is used to identify the communicator in the log file and in the replay tool.

注意：只有第一个 `ncclUniqueId` 用于生成 communicator 的 hash ID；日志文件和 replay 工具用这个 hash ID 识别 communicator。

#### Shrinking a communicator｜缩小 communicator

> The `ncclCommShrink()` function allows you to create a new communicator by removing specific ranks from an existing one. This is useful when you need to exclude certain GPUs or nodes from a collective operation, for example in fault tolerance scenarios or when dynamically adjusting resource utilization.

`ncclCommShrink()` 可以从现有 communicator 中移除指定 rank，从而创建新 communicator。例如在故障容忍或动态调整资源利用率时，可用它把某些 GPU 或节点排除在集合操作之外。

> The following example demonstrates how to create a new communicator by excluding rank 1:

下例展示如何排除 rank 1 并创建新 communicator：

```c
int excludeRanks[] = {1};  // Rank to exclude
int excludeCount = 1;      // Number of ranks to exclude
ncclComm_t newcomm;

// Only ranks that will be in the new communicator should call ncclCommShrink
if (myRank != 1) {
  ncclResult_t res = ncclCommShrink(comm, excludeRanks, excludeCount, &newcomm, NULL, NCCL_SHRINK_DEFAULT);
  if (res != ncclSuccess) {
    // Handle error
  }
  // Use the new communicator for collective operations
  // ...
  // When done, destroy the new communicator
  ncclCommDestroy(newcomm);
}
```

> When recovering from communication errors, you may want to use the error mode:

从通信错误中恢复时，可以使用错误处理模式：

```c
if (myRank != 1) {
  // When shrinking after an error, use NCCL_SHRINK_ABORT to abort operations on the parent communicator
  // This mode is also useful when there might be ongoing operations on the parent communicator
  ncclResult_t res = ncclCommShrink(comm, excludeRanks, excludeCount, &newcomm, NULL, NCCL_SHRINK_ABORT);
  // ...
}
```

> Note that:

请注意：

> Only ranks that will be part of the new communicator should call `ncclCommShrink()`.

只有将成为新 communicator 成员的 rank 才应调用 `ncclCommShrink()`。

> Ranks listed in the exclusion list should not call this function.

排除列表中的 rank 不应调用该函数。

> The new communicator will have ranks re-ordered to maintain contiguous numbering.

新 communicator 中的 rank 会重新排列，以保持编号连续。

> You can use the `ncclGroupStart`/`ncclGroupEnd` mechanism to synchronize the creation of new communicators.

可以使用 `ncclGroupStart`／`ncclGroupEnd` 机制同步新 communicator 的创建。

> Related link: `ncclCommShrink()`

相关链接：`ncclCommShrink()`。

#### Growing a communicator｜扩大 communicator

> The `ncclCommGrow()` function allows you to create a new communicator by adding new ranks to an existing one. This is useful when you need to dynamically scale up your computation by adding more GPUs or nodes to a running collective operation.

`ncclCommGrow()` 通过向现有 communicator 增加新 rank 来创建新 communicator。当计算任务需要动态加入更多 GPU 或节点，以扩大运行中的集合操作规模时，可以使用它。

> Growing a communicator involves coordination between existing ranks (from the parent communicator) and new ranks (joining the communicator). The process requires a coordinator rank from the existing communicator to generate a unique identifier using `ncclCommGetUniqueId()`, which is then distributed to all new ranks through an out-of-band mechanism (e.g., MPI, sockets, or shared memory).

扩大 communicator 需要原有 rank（父 communicator 成员）与新加入的 rank 协调。原组中的一个协调 rank 用 `ncclCommGetUniqueId()` 生成唯一 ID，再通过 MPI、socket 或共享内存等带外机制把 ID 传给所有新 rank。

> The following example demonstrates how to grow a 4-rank communicator to 8 ranks:

下例演示如何把一个 4-rank communicator 扩大到 8 个 rank：

```c
// Step 1: Coordinator (e.g., rank 0) generates the grow identifier
ncclUniqueId growId;
if (myRank == 0) {
  ncclResult_t res = ncclCommGetUniqueId(comm, &growId);
  if (res != ncclSuccess) {
    // Handle error
  }
  // Distribute growId to all new ranks using out-of-band communication
  // (e.g., MPI_Send, sockets, shared memory, etc.)
}

// Step 2: All existing ranks call ncclCommGrow
ncclComm_t newcomm;
ncclResult_t res = ncclCommGrow(comm, 8, NULL, -1, &newcomm, NULL);
if (res != ncclSuccess) {
  // Handle error
}

// Step 3: New ranks (4-7) call ncclCommGrow with the received growId
cudaSetDevice(myDevice);
ncclComm_t newcomm;
ncclResult_t res = ncclCommGrow(NULL, 8, &growId, myNewRank, &newcomm, NULL);

// Step 4: Wait for grow operation to complete (if non-blocking)
ncclResult_t asyncErr;
do {
  res = ncclCommGetAsyncError(newcomm, &asyncErr);
} while (asyncErr == ncclInProgress);

// Step 5: Use the new communicator for collective operations
// ...

// Step 6: Existing ranks should destroy the parent communicator
ncclCommDestroy(comm);

// Step 7: When done, destroy the new communicator
ncclCommDestroy(newcomm);
```

> For non-blocking grow operations with error handling:

下面是带错误处理的非阻塞 grow 操作示例：

```c
ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
config.blocking = 0;  // Non-blocking mode

// Existing ranks
ncclComm_t newcomm;
ncclResult_t res = ncclCommGrow(comm, 8, NULL, -1, &newcomm, &config);

// Poll for completion
ncclResult_t asyncErr;
do {
  res = ncclCommGetAsyncError(newcomm, &asyncErr);
  if (res != ncclSuccess) {
    // Handle error
    ncclCommAbort(newcomm);
    break;
  }
  // Handle timeouts or other events
} while (asyncErr == ncclInProgress);

if (asyncErr == ncclSuccess) {
  // Grow completed successfully
  // Destroy parent communicator
  ncclCommDestroy(comm);
}
```

> Important considerations:

重要注意事项：

> Coordinator selection: Any rank from the existing communicator can be the coordinator. The coordinator calls `ncclCommGetUniqueId()` to generate the grow identifier.

**协调 rank 的选择：**原 communicator 中的任意 rank 都可担任协调者。协调者调用 `ncclCommGetUniqueId()` 生成 grow ID。

> Rank assignment: Existing ranks retain their original rank numbers in the new communicator. New ranks must be assigned ranks starting from the size of the parent communicator.

**rank 分配：**原有 rank 在新 communicator 中保留原编号。新 rank 的编号从父 communicator 的规模开始。

> Out-of-band communication: The grow identifier must be distributed from the coordinator to all new ranks using a communication mechanism outside of NCCL (e.g., MPI, sockets, shared files).

**带外通信：**协调者必须用 NCCL 之外的通信机制（例如 MPI、socket 或共享文件）把 grow ID 分发给所有新 rank。

> Parent communicator cleanup: After the grow operation completes successfully, existing ranks should destroy the parent communicator using `ncclCommDestroy()` to free resources.

**父 communicator 清理：**grow 成功后，原有 rank 应调用 `ncclCommDestroy()` 销毁父 communicator，释放资源。

> No outstanding operations: There should not be any outstanding NCCL operations on the parent communicator when calling `ncclCommGrow()` to avoid potential deadlocks.

**未完成操作：**调用 `ncclCommGrow()` 时，父 communicator 上不应存在未完成的 NCCL 操作，以避免潜在死锁。

> Configuration inheritance: The new communicator inherits the configuration from the parent communicator for existing ranks. New ranks use the provided configuration or default settings.

**配置继承：**对原有 rank，新 communicator 继承父 communicator 的配置；新 rank 使用传入的配置或默认设置。

> Related links:
> `ncclCommGrow()`
> `ncclCommGetUniqueId()`

相关链接：`ncclCommGrow()`、`ncclCommGetUniqueId()`。

#### Creating more communicators｜从现有 communicator 创建更多组

> The `ncclCommSplit` function can be used to create communicators based on an existing one. This allows splitting an existing communicator into multiple sub-partitions, duplicate an existing communicator, or even create a single communicator with fewer ranks.

`ncclCommSplit` 可基于现有 communicator 创建新 communicator：把原组拆成多个子组、复制原组，或创建一个 rank 更少的组。

> The `ncclCommSplit` function needs to be called by all ranks in the original communicator. If some ranks will not be part of any sub-group, they still need to call `ncclCommSplit` with color being `NCCL_SPLIT_NOCOLOR`.

原 communicator 中的所有 rank 都必须调用 `ncclCommSplit`。即使某些 rank 不加入任何子组，也仍要以 `NCCL_SPLIT_NOCOLOR` 作为 `color` 调用。

> Newly created communicators will inherit the parent communicator configuration (e.g. non-blocking). If the parent communicator operates in non-blocking mode, a `ncclCommSplit` operation may be stopped by calling `ncclCommAbort` on the parent communicator, then on any new communicator returned. This is because a hang could happen during operations on any of the two communicators.

新创建的 communicator 会继承父 communicator 的配置（例如非阻塞设置）。如果父 communicator 是非阻塞的，可以先对父对象、再对返回的任何新对象调用 `ncclCommAbort` 来停止 split。因为在两个 communicator 的任一对象上执行操作，都可能发生挂起。

> The following code duplicates an existing communicator:

下列代码复制现有 communicator：

```c
int rank;
ncclCommUserRank(comm, &rank);
ncclCommSplit(comm, 0, rank, &newcomm, NULL);
```

> This splits a communicator in two halves:

下列代码把 communicator 拆成两半：

```c
int rank, nranks;
ncclCommUserRank(comm, &rank);
ncclCommCount(comm, &nranks);
ncclCommSplit(comm, rank/(nranks/2), rank%(nranks/2), &newcomm, NULL);
```

> This creates a communicator with only the first 2 ranks:

下列代码创建一个仅含前两个 rank 的 communicator：

```c
int rank;
ncclCommUserRank(comm, &rank);
ncclCommSplit(comm, rank<2 ? 0 : NCCL_SPLIT_NOCOLOR, rank, &newcomm, NULL);
```

> Related links:
> `ncclCommSplit()`

相关链接：`ncclCommSplit()`。

#### Using multiple NCCL communicators concurrently｜并发使用多个 NCCL communicator

> Prior to NCCL 2.26, using multiple NCCL communicators per-device required serializing the order of all communication operations (via CUDA stream dependencies or synchronization) into a consistent total global order otherwise deadlocks could ensue. As of 2.26, NCCL introduces `NCCL_LAUNCH_ORDER_IMPLICIT` which when enabled implicitly creates this order dynamically by following the order operations are issued from the host. Thus to remain deadlock free, users must ensure the order of host-side launches matches for all devices. This is most easily accomplished by using a determinstic order issued from a single host thread per-device. For example:

在 NCCL 2.26 之前，同一设备使用多个 NCCL communicator 时，所有通信操作的顺序需要通过 CUDA stream 依赖或同步串成一致的全局总顺序，否则可能死锁。从 2.26 起，NCCL 引入 `NCCL_LAUNCH_ORDER_IMPLICIT`；启用后，它根据 host 发起操作的顺序动态建立隐式顺序。为避免死锁，用户仍须保证各设备的 host 侧发起顺序一致。最容易做到这一点的方式是每个设备由一个 host 线程按确定顺序发起操作。例如：

```c
ncclAllReduce(..., comm1, stream1); // all ranks do this first
ncclAllReduce(..., comm2, stream2); // and this second
```

> When NCCL is captured in a CUDA graph the same rules apply to both capture time and launch time. At capture time this means NCCL calls in the same graph must be captured in the same order:

在 CUDA graph 中捕获 NCCL 时，捕获阶段和启动阶段都适用同样的规则。在捕获阶段，同一 graph 中的 NCCL 调用必须按相同顺序捕获：

```c
// both stream1 and stream2 are capturing in the same graph
ncclAllReduce(..., comm1, stream1); // all ranks do this first
ncclAllReduce(..., comm2, stream2); // and this second
```

> And at graph launch time different graphs must be launched in a globally consistent order:

在 graph 启动阶段，不同 graph 必须按全局一致的顺序启动：

```c
cudaGraphLaunch(graph1, stream1); // all ranks do this first
cudaGraphLaunch(graph2, stream2); // and this second
```

> When running on CUDA 12.3 or later, the implicit ordering of the operations is created using CUDA launch completion events which permits parallel execution of the two communicator’s kernels.

使用 CUDA 12.3 或更新版本时，操作的隐式顺序由 CUDA launch completion event 建立，因此两个 communicator 的 kernel 仍可并行执行。

#### Finalizing a communicator｜Finalize communicator

> `ncclCommFinalize` will transition a communicator from the `ncclSuccess` state to the `ncclInProgress` state, start completing all operations in the background and synchronize with other ranks which may be using resources for their communications with other ranks. All uncompleted operations and network-related resources associated to a communicator will be flushed and freed with `ncclCommFinalize`. Once all NCCL operations are complete, the communicator will transition to the `ncclSuccess` state. Users can query that state with `ncclCommGetAsyncError`. If a communicator is marked as nonblocking, this operation is nonblocking; otherwise, it is blocking.

`ncclCommFinalize` 会使 communicator 从 `ncclSuccess` 状态转为 `ncclInProgress`，开始在后台完成所有操作，并与可能仍在使用通信资源的其他 rank 同步。与 communicator 关联的未完成操作和网络资源会通过 `ncclCommFinalize` 完成清理。全部 NCCL 操作完成后，communicator 会回到 `ncclSuccess`；用户可用 `ncclCommGetAsyncError` 查询该状态。若 communicator 设置为非阻塞，这次调用也非阻塞；否则会阻塞。

> `ncclCommFinalize` is an intra-node collective call. When a single thread finalizes multiple ranks (multiple GPUs per thread), the calls must be grouped with `ncclGroupStart`/`ncclGroupEnd` to avoid a hang.

`ncclCommFinalize` 是节点内的集合调用。如果一个线程负责 finalize 多个 rank（每线程多个 GPU），这些调用必须放在 `ncclGroupStart`／`ncclGroupEnd` 中，以避免挂起。

> Related link: `ncclCommFinalize()`

相关链接：`ncclCommFinalize()`。

#### Destroying a communicator｜Destroy communicator

> Once a communicator has been finalized, the next step is to free all resources, including the communicator itself. Local resources associated to a communicator can be destroyed with `ncclCommDestroy`. If the state of a communicator is `ncclSuccess` when calling `ncclCommDestroy`, the call is guaranteed to be nonblocking; otherwise `ncclCommDestroy` might block. In all cases, `ncclCommDestroy` call will free the resources of the communicator and return, and the communicator should no longer be accessed after `ncclCommDestroy` returns.

communicator 完成 finalize 后，还需释放包括对象本身在内的全部资源。可用 `ncclCommDestroy` 释放关联的本地资源。若调用时 communicator 状态为 `ncclSuccess`，调用保证非阻塞；否则可能阻塞。无论哪种情况，`ncclCommDestroy` 返回时对象资源已释放，此后不应再访问该 communicator。

> Related link: `ncclCommDestroy()`

相关链接：`ncclCommDestroy()`。

#### Error handling and communicator abort｜错误处理与 communicator abort

> All NCCL calls return a NCCL error code which is summarized in the table below. If a NCCL call returns an error code different from `ncclSuccess` and `ncclInternalError`, and if `NCCL_DEBUG` is set to WARN, NCCL will print a human-readable message explaining what happened. If `NCCL_DEBUG` is set to INFO, NCCL will also print the call stack which led to the error. This message is intended to help the user fix the problem.

所有 NCCL 调用都会返回一个 NCCL 错误码，汇总见下表。如果返回值既不是 `ncclSuccess` 也不是 `ncclInternalError`，且设置 `NCCL_DEBUG=WARN`，NCCL 会打印可读的错误说明；若设为 `INFO`，还会打印导致错误的调用栈。这些信息用于帮助用户修复问题。

> The table below summarizes how different errors should be understood and handled. Each case is explained in details in the following sections.

下表概括各错误的含义与处理方式；后续段落分别说明具体情况。

##### NCCL Errors｜NCCL 错误码

> | Error | Description | Resolution | Error handling | Group behavior |
> | --- | --- | --- | --- | --- |
> | ncclSuccess | No error | None | None | None |
> | ncclUnhandledCudaError | Error during a CUDA call (1) | CUDA configuration / usage (1) | Communicator abort (5) | Global (6) |
> | ncclSystemError | Error during a system call (1) | System configuration / usage (1) | Communicator abort (5) | Global (6) |
> | ncclInternalError | Error inside NCCL (2) | Fix in NCCL (2) | Communicator abort (5) | Global (6) |
> | ncclInvalidArgument | An argument to a NCCL call is invalid (3) | Fix in the application (3) | None (3) | Individual (3) |
> | ncclInvalidUsage | The usage of NCCL calls is invalid (4) | Fix in the application (4) | Communicator abort (5) | Global (6) |
> | ncclInProgress | The NCCL call is still in progress | Poll for completion using ncclCommGetAsyncError | None | None |

| 错误码 | 描述 | 处理方向 | 错误处理 | 组行为 |
| --- | --- | --- | --- | --- |
| `ncclSuccess` | 无错误 | 无 | 无 | 无 |
| `ncclUnhandledCudaError` | CUDA 调用出错（1） | 检查 CUDA 配置／使用方式（1） | 中止 communicator（5） | 全局（6） |
| `ncclSystemError` | 系统调用出错（1） | 检查系统配置／使用方式（1） | 中止 communicator（5） | 全局（6） |
| `ncclInternalError` | NCCL 内部错误（2） | 修复 NCCL（2） | 中止 communicator（5） | 全局（6） |
| `ncclInvalidArgument` | NCCL 调用参数无效（3） | 修正应用（3） | 无需（3） | 仅影响该调用（3） |
| `ncclInvalidUsage` | NCCL API 使用方式无效（4） | 修正应用（4） | 中止 communicator（5） | 全局（6） |
| `ncclInProgress` | 调用仍在进行 | 使用 `ncclCommGetAsyncError` 轮询完成 | 无 | 无 |

> (1) `ncclUnhandledCudaError` and `ncclSystemError` indicate that a call NCCL made to an external component failed, which caused the NCCL operation to fail. The error message should explain which component the user should look at and try to fix, potentially with the help of the administrators of the system.

（1）`ncclUnhandledCudaError` 和 `ncclSystemError` 表示 NCCL 调用外部组件时失败，导致 NCCL 操作失败。错误消息应指出需要检查并尝试修复的组件；必要时可请系统管理员协助。

> (2) `ncclInternalError` denotes a NCCL bug. It might not report a message with `NCCL_DEBUG`=WARN since it requires a fix in the NCCL source code. `NCCL_DEBUG`=INFO will print the back trace which led to the error.

（2）`ncclInternalError` 表示 NCCL 自身存在缺陷。由于需要修复 NCCL 源码，设置 `NCCL_DEBUG=WARN` 时可能没有提示；设置为 `INFO` 会打印导致错误的回溯。

> (3) `ncclInvalidArgument` indicates an argument value is incorrect, like a NULL pointer or an out-of-bounds value. When this error is returned, the NCCL call had no effect. The group state remains unchanged, the communicator is still functioning normally. The application can call `ncclCommAbort` or continue as if the call did not happen. This error will be returned immediately for a call happening within a group and applies to that specific NCCL call. It will not be returned by `ncclGroupEnd` since `ncclGroupEnd` takes no argument.

（3）`ncclInvalidArgument` 表示参数值有误，例如空指针或越界值。返回此错误时，该 NCCL 调用没有产生效果；group 状态不变，communicator 仍正常工作。应用可以调用 `ncclCommAbort`，也可以视作该调用未发生而继续。在 group 内，此错误立即由具体调用返回，仅影响该调用；它不会由 `ncclGroupEnd` 返回，因为 `ncclGroupEnd` 不接收参数。

> (4) `ncclInvalidUsage` is returned when a dynamic condition causes a failure, which denotes an incorrect usage of the NCCL API.

（4）`ncclInvalidUsage` 表示动态条件导致失败，即 NCCL API 使用方式有误。

> (5) These errors are fatal for the communicator. To recover, the application needs to call `ncclCommAbort` on the communicator and re-create it.

（5）这些错误对 communicator 是致命的。要恢复，应用必须对该 communicator 调用 `ncclCommAbort`，然后重新创建它。

> (6) Dynamic errors for operations within a group are always reported by `ncclGroupEnd` and apply to all operations within the group, which may or may not have completed. The application must call `ncclCommAbort` on all communicators within the group.

（6）group 内操作的动态错误总由 `ncclGroupEnd` 报告，并适用于组内所有操作；这些操作可能完成，也可能未完成。应用必须对该 group 中的所有 communicator 调用 `ncclCommAbort`。

#### Asynchronous errors and error handling｜异步错误及其处理

> Some communication errors, and in particular network errors, are reported through the `ncclCommGetAsyncError` function. Operations experiencing an asynchronous error will usually not progress and never complete. When an asynchronous error happens, the operation should be aborted and the communicator destroyed using `ncclCommAbort`. When waiting for NCCL operations to complete, applications should call `ncclCommGetAsyncError` and destroy the communicator when an error happens.

某些通信错误，尤其是网络错误，会通过 `ncclCommGetAsyncError` 报告。发生异步错误的操作通常不会继续推进，也永远无法完成。出现异步错误时，应使用 `ncclCommAbort` 中止操作并销毁 communicator。等待 NCCL 操作完成时，应用应查询 `ncclCommGetAsyncError`，并在出错时销毁 communicator。

> The following code shows how to wait on NCCL operations and poll for asynchronous errors, instead of using `cudaStreamSynchronize`.

下面的代码展示如何等待 NCCL 操作并轮询异步错误，而不是直接使用 `cudaStreamSynchronize`。

```c
int ncclStreamSynchronize(cudaStream_t stream, ncclComm_t comm) {
  cudaError_t cudaErr;
  ncclResult_t ncclErr, ncclAsyncErr;
  while (1) {
   cudaErr = cudaStreamQuery(stream);
   if (cudaErr == cudaSuccess)
     return 0;

   if (cudaErr != cudaErrorNotReady) {
     printf("CUDA Error : cudaStreamQuery returned %d\n", cudaErr);
     return 1;
   }

   ncclErr = ncclCommGetAsyncError(comm, &ncclAsyncErr);
   if (ncclErr != ncclSuccess) {
     printf("NCCL Error : ncclCommGetAsyncError returned %d\n", ncclErr);
     return 1;
   }

   if (ncclAsyncErr != ncclSuccess) {
     // An asynchronous error happened. Stop the operation and destroy
     // the communicator
     ncclErr = ncclCommAbort(comm);
     if (ncclErr != ncclSuccess)
       printf("NCCL Error : ncclCommDestroy returned %d\n", ncclErr);
     // Caller may abort or try to create a new communicator.
     return 2;
   }

   // We might want to let other threads (including NCCL threads) use the CPU.
   sched_yield();
  }
}
```

> Related links:
> `ncclCommGetAsyncError()`
> `ncclCommAbort()`

相关链接：`ncclCommGetAsyncError()`、`ncclCommAbort()`。

#### Fault Tolerance｜故障容忍

> NCCL provides a set of features to allow applications to recover from fatal errors such as a network failure, a node failure, or a process failure. When such an error happens, the application should be able to call `ncclCommAbort` on the communicator to free all resources, then create a new communicator to continue.

NCCL 提供一组功能，供应用在网络、节点或进程故障等致命错误后恢复。发生此类错误时，应用应能调用 `ncclCommAbort` 释放 communicator 的全部资源，然后创建新 communicator 继续运行。

> For more advanced recovery, the `ncclCommShrink` function with `NCCL_SHRINK_ABORT` can be used to create a new communicator by removing failed ranks from the existing communicator while safely handling in-progress operations. This approach is particularly useful in distributed environments where only some ranks have failed.

需要更复杂的恢复时，可使用带 `NCCL_SHRINK_ABORT` 的 `ncclCommShrink`，从现有 communicator 中移除故障 rank，创建新 communicator，同时安全处理正在进行的操作。这尤其适合只有部分 rank 失效的分布式环境。

> In order to abort NCCL communicators safely, NCCL requires applications to set communicators as nonblocking and make sure no thread is calling any NCCL operations while calling `ncclCommAbort`. After nonblocking is set, all NCCL calls (except `ncclCommDestroy`/Abort) become nonblocking so that `ncclCommAbort` can be called at any point, during initialization, communication or finalizing the communicator. If NCCL communicators are set blocking, the thread can possibly get stuck inside NCCL calls due to network errors; in this case, NCCL communicators might hang forever.

为了安全地中止 NCCL communicator，应用必须将其设为非阻塞，并确保调用 `ncclCommAbort` 时没有线程仍在调用任何 NCCL 操作。设置非阻塞后，除 `ncclCommDestroy`／`ncclCommAbort` 外的 NCCL 调用均为非阻塞，因此可在初始化、通信或 finalize 期间执行 abort。若 communicator 设置为阻塞模式，网络错误可能使线程卡在 NCCL 调用中，communicator 甚至可能一直挂起。

> To correctly abort, when any rank in a communicator fails (e.g., due to a segmentation fault), all other ranks need to call `ncclCommAbort` to abort their own NCCL communicator. Users can implement methods to decide when and whether to abort the communicators and restart the NCCL operation. Here is an example showing how to initialize and split a communicator in a non-blocking manner, allowing for an abort at any point:

要正确执行 abort，当 communicator 中任一 rank 失效（如进程出现段错误），其他所有 rank 都需对各自的 NCCL communicator 调用 `ncclCommAbort`。用户可自行设计何时、是否中止并重启 NCCL 操作。下面的示例以非阻塞方式初始化并拆分 communicator，使应用能够随时中止：

```c
bool globalFlag;
bool abortFlag = false;
ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
/* set communicator as nonblocking */
config.blocking = 0;
CHECK(ncclCommInitRankConfig(&comm, nRanks, id, myRank, &config));
do {
  CHECK(ncclCommGetAsyncError(comm, &state));
} while(state == ncclInProgress && checkTimeout() != true);

if (checkTimeout() == true || state != ncclSuccess) abortFlag = true;

/* sync abortFlag among all healthy ranks. */
reportErrorGlobally(abortFlag, &globalFlag);

if (globalFlag) {
  /* time is out or initialization failed: every rank needs to abort and restart. */
  ncclCommAbort(comm);
  /* restart NCCL; this is a user implemented function, it might include
   * resource cleanup and ncclCommInitRankConfig() to create new communicators. */
  restartNCCL(&comm);
}

/* nonblocking communicator split. */
CHECK(ncclCommSplit(comm, color, key, &childComm, &config));
do {
  CHECK(ncclCommGetAsyncError(comm, &state));
} while(state == ncclInProgress && checkTimeout() != true);

if (checkTimeout() == true || state != ncclSuccess) abortFlag = true;

/* sync abortFlag among all healthy ranks. */
reportErrorGlobally(abortFlag, &globalFlag);

if (globalFlag) {
  ncclCommAbort(comm);
  /* if chilComm is not NCCL_COMM_NULL, user should abort child communicator
   * here as well for resource reclamation. */
  if (childComm != NCCL_COMM_NULL) ncclCommAbort(childComm);
  restartNCCL(&comm);
}
/* application workload */
```

> The checkTimeout function needs to be provided by users to determine what is the longest time the application should wait for NCCL initialization; likewise, users can apply other methods to detect errors besides a timeout function. Similar methods can be applied to NCCL finalization as well.

用户需实现 `checkTimeout`，用于规定应用等待 NCCL 初始化的最长时间；也可用超时以外的方法检测错误。类似方法还可以用于 NCCL finalize。

#### Quality of Service｜服务质量

> Applications which overlap communication may benefit from network Quality of Service (QoS) features. NCCL allows an application to assign a traffic class (TC) to each communicator to identify the communication requirements of the communicator. All network operations on a communicator will use the assigned TC.

存在重叠通信的应用可能受益于网络 Quality of Service（QoS）。NCCL 允许应用为每个 communicator 指定 traffic class（TC），以标识其通信需求；该 communicator 上的全部网络操作都使用指定的 TC。

> The meaning of TC is specific to the network plugin in use by the communicator (e.g. IB networks use service level, RoCE networks use type of service). TCs are defined by the system configuration. Applications must understand the TCs available on a system and their relative behavior in order to use them effectively.

TC 的含义取决于 communicator 使用的网络插件，例如 IB 网络使用 service level，RoCE 网络使用 type of service。TC 由系统配置定义。应用必须了解系统可用的 TC 及其相对行为，才能有效使用它们。

> TC is specified during communicator creation using `ncclConfig_t`.

创建 communicator 时，通过 `ncclConfig_t` 指定 TC。

```c
ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
config.trafficClass = 1;
CHECK(ncclCommInitRankConfig(&comm, nranks, id, rank, &config));
```

> Infiniband networks support QoS through the use of Service Levels (SL). Each IB SL is mapped to Virtual Lane (VL), which defines the relative priority of traffic. SL behavior is defined within the subnet manager, such as OpenSM. Refer to subnet manager documentation for more detail. An example configuration is shown below.

InfiniBand 网络通过 Service Level（SL）支持 QoS。每个 IB SL 都映射到 Virtual Lane（VL），后者定义流量的相对优先级。SL 的行为由 OpenSM 等子网管理器决定；详情请参阅子网管理器文档。下面给出一个配置示例。

```text
...
qos_max_vls 2
qos_high_limit 255
qos_vlarb_high 1:4
qos_vlarb_low 0:1,1:4
qos_sl2vl 0,1

max_op_vls 2
....
```

> The example defines one low priority and one high priority VL which are mapped to SL 0 and 1, respectively. The high priority SL will be given a larger share of network bandwidth at each port. In NCCL, the communicator’s traffic class corresponds to the SL on IB networks. Using this configuration, applications can assign TC 0 to low-priority communicators and TC 1 to high-priority ones.

该示例定义了一个低优先级 VL 和一个高优先级 VL，分别映射到 SL 0 和 SL 1。高优先级 SL 在每个端口获得更大的网络带宽份额。在 IB 网络上，NCCL communicator 的 traffic class 对应 SL；采用此配置时，应用可将 TC 0 分配给低优先级 communicator，将 TC 1 分配给高优先级 communicator。

> On RoCE networks, the NCCL communicator `trafficClass` is interpreted as an IP Type of Service (ToS). Refer to network management tools to understand how to configure QoS for a given workload.

在 RoCE 网络上，NCCL communicator 的 `trafficClass` 被解释为 IP Type of Service（ToS）。具体如何为工作负载配置 QoS，应参考网络管理工具。

### 重点解读

**建立通信组的共同前提。** `rank` 是 communicator 内的编号，每个 rank 要对应不同的 CUDA 设备。跨进程初始化时，`ncclGetUniqueId()` 生成的 ID 必须由应用经 CPU 侧通道传给所有参与者，然后各 rank 使用同一 ID 建组。这与前面 Setup 中“bootstrap 依赖应用的进程管理和 CPU 侧通信”相呼应。`ncclCommInitAll()` 只方便单进程内一次创建多个对象，不能把它等同于跨节点的进程启动与初始化。

**不同的组变更有不同参与规则。** `ncclCommSplit()` 要求父组所有 rank 调用，退出子组的 rank 也要传 `NCCL_SPLIT_NOCOLOR`；`ncclCommShrink()` 只由留在新组的 rank 调用，移除者不参与；`ncclCommGrow()` 则要求旧 rank 与新 rank 通过带外机制协调。它们都创建新的 communicator，而不是原地修改 rank 列表。尤其在 shrink 后，留下的 rank 会重新编号；grow 后，旧 rank 保留编号，新 rank 从父组规模之后开始编号。

**正确性先于并发性能。** 同一设备上多个 communicator 的通信顺序必须在各参与设备之间保持一致；CUDA graph 也要在捕获和启动两阶段保持对应顺序。隐式发起顺序能够建立所需关系，但不能替应用决定跨设备一致的 host 调用顺序。原文还要求 grow 前父组没有未完成操作，说明动态扩容必须与现有工作协调。

**正常结束与故障恢复是两条路径。** 正常结束时先 `ncclCommFinalize()`，等待操作完成并查询状态，再 `ncclCommDestroy()`；发生致命或异步错误时，应协调相关 rank 调用 `ncclCommAbort()`，必要时重新创建或缩小 communicator。`ncclInProgress` 只是尚未完成的状态，不等于失败；`ncclInvalidArgument` 在原文中属于可局部修正的调用错误。故障恢复涉及应用级超时、跨 rank 决策和重建逻辑，NCCL 的示例并未替应用实现这些部分。

**示例代码的使用边界。** 原文把 `ncclCommInitAll()` 实现称为简化示例；其他片段也依赖未给出的 `CHECK`、`checkTimeout`、`reportErrorGlobally` 等辅助定义，不能直接当成完整可运行程序。异步错误示例调用的是 `ncclCommAbort()`，但随后的打印文字写成了 `ncclCommDestroy returned`；故障容忍示例注释中的 `chilComm` 也与代码变量 `childComm` 不一致。这些文字差异保留在原文中，阅读时以实际函数调用和变量名为准。

**MIG 与 QoS 的配置边界。** MIG 一节同时给出“每个 rank 选不同实例”“关闭 MNNVL”“不使用 NVLS”及驱动要求，不能只看是否能创建 communicator。QoS 中的 TC 也不是跨网络通用的固定优先级数值；它如何映射到 SL、VL 或 ToS，取决于网络插件和系统配置。



## Collective Operations

来源：[NVIDIA NCCL User Guide — Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#collective-operations)。

### 中英文对照

> Collective operations have to be called for each rank (hence CUDA device), using the same count and the same datatype, to form a complete collective operation. Failure to do so will result in undefined behavior, including hangs, crashes, or data corruption.

要形成一次完整的集合操作，每个 rank（因而每个 CUDA 设备）都必须调用该操作，并使用相同的 `count` 和数据类型。否则会产生未定义行为，包括挂起、崩溃或数据损坏。

#### AllReduce

> The AllReduce operation performs reductions on data (for example, sum, min, max) across devices and stores the result in the receive buffer of every rank.

AllReduce 在设备之间对数据执行归约（例如求和、求最小值、求最大值），并将结果存入每个 rank 的接收缓冲区。

> In a *sum* allreduce operation between *k* ranks, each rank will provide an array `in` of N values, and receive identical results in array `out` of N values, where `out[i] = in0[i]+in1[i]+…+in(k-1)[i]`.

在 $k$ 个 rank 参与的求和 AllReduce 中，每个 rank 提供一个包含 $N$ 个值的 `in` 数组，并在包含 $N$ 个值的 `out` 数组中得到相同结果，其中 `out[i] = in0[i]+in1[i]+…+in(k-1)[i]`。

> *All-Reduce operation: each rank receives the reduction of input values across ranks.*

*AllReduce 操作：每个 rank 都收到所有 rank 输入值的归约结果。*

> Related links: [`ncclAllReduce()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclAllReduce).

相关 API：[`ncclAllReduce()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclAllReduce)。

#### Broadcast

> The Broadcast operation copies an N-element buffer from the root rank to all the ranks.

Broadcast 将 root rank 上包含 $N$ 个元素的缓冲区复制到所有 rank。

> *Broadcast operation: all ranks receive data from a “root” rank.*

*Broadcast 操作：所有 rank 都收到来自 root rank 的数据。*

> Important note: The root argument is one of the ranks, not a device number, and is therefore impacted by a different rank to device mapping.

**重要提示：**`root` 参数是一个 rank 编号，不是设备编号；因此 rank 到设备的映射改变时，`root` 指向的设备也可能改变。

> Related links: [`ncclBroadcast()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclBroadcast).

相关 API：[`ncclBroadcast()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclBroadcast)。

#### Reduce

> The Reduce operation performs the same operation as AllReduce, but stores the result only in the receive buffer of a specified root rank.

Reduce 执行与 AllReduce 相同的归约，但只把结果存入指定 root rank 的接收缓冲区。

> *Reduce operation: one rank receives the reduction of input values across ranks.*

*Reduce 操作：只有一个 rank 收到所有 rank 输入值的归约结果。*

> Important note: The root argument is one of the ranks (not a device number), and is therefore impacted by a different rank to device mapping.

**重要提示：**`root` 参数是一个 rank 编号（不是设备编号）；因此 rank 到设备的映射改变时，`root` 指向的设备也可能改变。

> Note: A Reduce, followed by a Broadcast, is equivalent to the AllReduce operation.

**注意：**先执行 Reduce，再执行 Broadcast，在结果语义上等价于 AllReduce。

> Related links: [`ncclReduce()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclReduce).

相关 API：[`ncclReduce()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclReduce)。

#### AllGather

> The AllGather operation gathers N values from k ranks into an output buffer of size k*N, and distributes that result to all ranks.

AllGather 从 $k$ 个 rank 分别收集 $N$ 个值，组成大小为 $kN$ 的输出缓冲区，并把该结果分发给所有 rank。

> The output is ordered by the rank index. The AllGather operation is therefore impacted by a different rank to device mapping.

输出按 rank 编号排序。因此，rank 到设备的映射改变会影响 AllGather 的输出布局。

> *AllGather operation: each rank receives the aggregation of data from all ranks in the order of the ranks.*

*AllGather 操作：每个 rank 都按 rank 顺序收到所有 rank 汇集后的数据。*

> Note: Executing ReduceScatter, followed by AllGather, is equivalent to the AllReduce operation.

**注意：**先执行 ReduceScatter，再执行 AllGather，在结果语义上等价于 AllReduce。

> Related links: [`ncclAllGather()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclAllGather).

相关 API：[`ncclAllGather()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclAllGather)。

#### ReduceScatter

> The ReduceScatter operation performs the same operation as Reduce, except that the result is scattered in equal-sized blocks between ranks, each rank getting a chunk of data based on its rank index.

ReduceScatter 执行与 Reduce 相同的归约，但会将结果按等大的数据块散发给各 rank；每个 rank 根据自己的 rank 编号得到对应的数据块。

> The ReduceScatter operation is impacted by a different rank to device mapping since the ranks determine the data layout.

由于 rank 决定数据布局，rank 到设备的映射改变会影响 ReduceScatter 的结果分配。

> *Reduce-Scatter operation: input values are reduced across ranks, with each rank receiving a subpart of the result.*

*ReduceScatter 操作：各 rank 的输入值先归约，每个 rank 再收到结果的一部分。*

> Related links: [`ncclReduceScatter()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclReduceScatter)

相关 API：[`ncclReduceScatter()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclReduceScatter)。

#### AlltoAll

> In an AlltoAll operation between k ranks, each rank provides an input buffer of size k*N values, where the j-th chunk of N values is sent to destination rank j. Each rank receives an output buffer of size k*N values, where the i-th chunk of N values comes from source rank i.

在 $k$ 个 rank 参与的 AlltoAll 中，每个 rank 提供一个包含 $kN$ 个值的输入缓冲区，其中第 $j$ 块包含 $N$ 个值，发送到目标 rank $j$。每个 rank 收到一个包含 $kN$ 个值的输出缓冲区，其中第 $i$ 块包含 $N$ 个值，来自源 rank $i$。

> *AlltoAll operation: exchanges data between all ranks, where each rank sends different data to every other rank and receives different data from every other rank.*

*AlltoAll 操作：所有 rank 之间交换数据，每个 rank 向其他各 rank 发送不同的数据，并从其他各 rank 接收不同的数据。*

> Related links: [`ncclAlltoAll()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclAlltoAll).

相关 API：[`ncclAlltoAll()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclAlltoAll)。

#### Gather

> The Gather operation gathers N values from k ranks into an output buffer on the root rank of size k*N.

Gather 从 $k$ 个 rank 分别收集 $N$ 个值，在 root rank 上组成大小为 $kN$ 的输出缓冲区。

> *Gather operation: root rank receives data from all ranks.*

*Gather 操作：root rank 收到所有 rank 的数据。*

> Important note: The root argument is one of the ranks, not a device number, and is therefore impacted by a different rank to device mapping.

**重要提示：**`root` 参数是一个 rank 编号，不是设备编号；因此 rank 到设备的映射改变时，`root` 指向的设备也可能改变。

> Related links: [`ncclGather()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclGather).

相关 API：[`ncclGather()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclGather)。

#### Scatter

> The Scatter operation distributes a total of N*k values from the root rank to k ranks, each rank receiving N values.

Scatter 从 root rank 将总计 $Nk$ 个值分发给 $k$ 个 rank，每个 rank 收到 $N$ 个值。

> *Scatter operation: root rank distributes data to all ranks.*

*Scatter 操作：root rank 向所有 rank 分发数据。*

> Important note: The root argument is one of the ranks, not a device number, and is therefore impacted by a different rank to device mapping.

**重要提示：**`root` 参数是一个 rank 编号，不是设备编号；因此 rank 到设备的映射改变时，`root` 指向的设备也可能改变。

> Related links: [`ncclScatter()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclScatter).

相关 API：[`ncclScatter()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclScatter)。

### 重点解读

**调用必须由整组共同完成。** 集合操作以 communicator 中的 rank 为参与者。原文要求每个 rank 都调用，并保持相同的 `count` 与数据类型；漏掉某个 rank 或参数不一致，不只是结果错误，还可能造成挂起、崩溃或数据损坏。调用前应先确定参与组和每个 rank 的数据形状。

**结果给谁，与如何排列，是两件不同的事。** AllReduce、AllGather 的完整结果出现在所有 rank；Reduce、Gather 只把完整结果交给 root；Broadcast、Scatter 从 root 向其他 rank 分发；ReduceScatter 让各 rank 各得一块归约结果；AlltoAll 则让每个 rank 与所有 rank 交换各自对应的数据块。`root` 总是组内 rank 编号。AllGather、ReduceScatter 和 AlltoAll 的数据块位置按 rank 编号确定，所以改变 rank 与设备的对应关系可能改变结果布局。

**两种 AllReduce 分解只说明结果语义。** Reduce 后接 Broadcast，或 ReduceScatter 后接 AllGather，都可以得到与 AllReduce 相同的结果。实际使用时仍需让前后操作的归约类型、数据量和块顺序相匹配；这种等价不表示调用次数、通信路径或性能相同。

## Data Pointers

来源：[NVIDIA NCCL User Guide — Data Pointers](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/data.html#data-pointers)。

### 中英文对照

> In general, NCCL will accept any CUDA pointers that are accessible from the CUDA device associated to the communicator object. This includes:

一般来说，只要某个 CUDA 指针能被 communicator 对象所关联的 CUDA 设备访问，NCCL 就会接受它。这包括：

> - device memory local to the CUDA device
> - host memory registered using CUDA SDK APIs `cudaHostRegister` or `cudaGetDevicePointer`
> - managed and unified memory

- 该 CUDA 设备上的本地 device memory；
- 使用 CUDA SDK API `cudaHostRegister` 或 `cudaGetDevicePointer` 注册的 host memory；
- managed memory 和 unified memory。

> The only exception is device memory located on another device but accessible from the current device using peer access. NCCL will return an error in that case to avoid programming errors (only when `NCCL_CHECK_POINTERS=1` since 2.2.12).

唯一的例外是位于另一设备上、但当前设备可通过 peer access 访问的 device memory。为避免编程错误，NCCL 在这种情况下会返回错误（自 2.2.12 起，仅在设置 `NCCL_CHECK_POINTERS=1` 时进行此项检查）。

### 重点解读

**指针的访问主体是 communicator 对应的 CUDA 设备。** 判断能否传入指针时，要看该设备能否访问所指内存，而不只是看 host 代码是否持有这个地址。原文列出三类通常可接受的内存：设备本地内存、按所述 CUDA API 处理的 host memory，以及 managed／unified memory。

**peer access 不等于可以把远端 device pointer 直接传给 NCCL。** 即使当前设备能通过 peer access 访问另一设备的 device memory，原文仍把这种指针列为例外。传参时应区分“本设备的本地 device memory”和“另一设备的 memory”；原文还特别限定，所述报错检查需要启用 `NCCL_CHECK_POINTERS=1`。关闭检查只会影响是否报告该错误，不会改变这项指针使用限制。

## CUDA Stream Semantics

来源：[NVIDIA NCCL User Guide — CUDA Stream Semantics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/streams.html#cuda-stream-semantics)。

### 中英文对照

> NCCL calls are associated to a stream which is passed as the last argument of the collective communication function. The NCCL call returns when the operation has been effectively enqueued to the given stream, or returns an error. The collective operation is then executed asynchronously on the CUDA device. The operation status can be queried using standard CUDA semantics, for example, calling `cudaStreamSynchronize` or using CUDA events.

NCCL 调用与一条 stream 关联，该 stream 作为集合通信函数的最后一个参数传入。操作已实际加入指定 stream 的队列时，NCCL 调用便会返回；若出错，则返回错误。随后，集合操作在 CUDA 设备上异步执行。可以按标准 CUDA 语义查询操作状态，例如调用 `cudaStreamSynchronize` 或使用 CUDA event。

#### Mixing Multiple Streams within the same `ncclGroupStart/End()` group｜同一 group 内混用多条 stream

> NCCL allows for using multiple streams within a group call. This will enforce a stream dependency of all streams before the NCCL kernel starts and block all streams until the NCCL kernel completes.

NCCL 允许在一次 group 调用中使用多条 stream。这会在 NCCL kernel 开始前，对所有这些 stream 建立依赖关系；在该 kernel 完成前，这些 stream 都会等待。

> It will behave as if the NCCL group operation was posted on every stream, but given it is a single operation, it will cause a global synchronization point between the streams.

其行为相当于把 NCCL group 操作提交到每条 stream；但由于它实际上是单个操作，因此会在这些 stream 之间形成一个全局同步点。

### 重点解读

**调用返回不代表设备端完成。** 普通 NCCL 调用成功返回，表示通信操作已进入指定 CUDA stream，后续仍由设备异步执行。若程序需要确认结果已经可用，应按 CUDA stream 或 event 的完成状态判断；不能仅凭 host 侧函数返回就读取结果。

**同一 group 的多条 stream 会在通信处汇合。** 这些 stream 在 NCCL kernel 启动前需要满足各自此前工作的依赖，kernel 完成后才能继续执行各自后续工作。原文的“block all streams”描述的是 stream 执行顺序中的等待，并不表示这里一定会让 host 线程同步等待。即使传入多条 stream，原文描述的仍是一次 NCCL 操作，而不是每条 stream 各执行一次独立的通信。

## Group Calls

来源：[NVIDIA NCCL User Guide — Group Calls](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html)。

### 中英文对照

> Group functions (`ncclGroupStart`/`ncclGroupEnd`) can be used to merge multiple calls into one. This is needed for three purposes: managing multiple GPUs from one thread (to avoid deadlocks), aggregating communication operations to improve performance, or merging multiple send/receive point-to-point operations (see [Point-to-point communication](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html#point-to-point) section). All three usages can be combined together, with one exception: calls to [`ncclCommInitRank()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#c.ncclCommInitRank) cannot be merged with others.

Group 函数（`ncclGroupStart`／`ncclGroupEnd`）可以把多个调用合为一组，主要用于三种情况：单个线程管理多个 GPU（避免死锁）、聚合通信操作以改善性能，以及合并多个点对点发送／接收操作（参见 [Point-to-point communication](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html#point-to-point)）。三种用法可以结合，但 [`ncclCommInitRank()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#c.ncclCommInitRank) 调用不能与其他调用合并。

#### Management Of Multiple GPUs From One Thread｜单线程管理多个 GPU

> When a single thread is managing multiple devices, group semantics must be used. This is because every NCCL call may have to block, waiting for other threads/ranks to arrive, before effectively posting the NCCL operation on the given stream. Hence, a simple loop on multiple devices like shown below could block on the first call waiting for the other ones:

单个线程管理多个设备时，必须使用 group 语义。原因是每次 NCCL 调用在真正把操作提交到指定 stream 前，都可能阻塞并等待其他线程或 rank 到达。因此，下面这种逐设备调用的简单循环可能在第一次调用处等待后续设备，从而无法继续执行：

> **Warning**
>
> We do not recommed using CUDA graph capture when managing multiple GPUs from one thread. In some cases `cudaGraphLaunch` may block, preventing the launch across all GPUs. See [Using NCCL with CUDA Graphs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html#using-nccl-with-cuda-graphs) for details.

**警告：**原文不建议在单线程管理多个 GPU 时使用 CUDA graph capture。某些情况下，`cudaGraphLaunch` 可能阻塞，使程序无法完成所有 GPU 上的启动。详情参见 [Using NCCL with CUDA Graphs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html#using-nccl-with-cuda-graphs)。

```c
for (int i=0; i<nLocalDevs; i++) {
  ncclAllReduce(..., comm[i], stream[i]);
}
```

> To define that these calls are part of the same collective operation, `ncclGroupStart` and `ncclGroupEnd` should be used:

要表明这些调用属于同一次集合操作，应使用 `ncclGroupStart` 和 `ncclGroupEnd`：

```c
ncclGroupStart();
for (int i=0; i<nLocalDevs; i++) {
  ncclAllReduce(..., comm[i], stream[i]);
}
ncclGroupEnd();
```

> This will tell NCCL to treat all calls between `ncclGroupStart` and `ncclGroupEnd` as a single call to many devices.

这会让 NCCL 将 `ncclGroupStart` 与 `ncclGroupEnd` 之间的所有调用视为面向多个设备的一次调用。

> Caution: When called inside a group, stream operations (like `ncclAllReduce`) can return without having enqueued the operation on the stream. Stream operations like `cudaStreamSynchronize` can therefore be called only after `ncclGroupEnd` returns.

**注意：**在 group 内调用时，`ncclAllReduce` 等 stream 操作可能在尚未将工作加入 stream 队列时就返回。因此，`cudaStreamSynchronize` 等 stream 操作只能在 `ncclGroupEnd` 返回之后调用。

> Group calls must also be used to create a communicator when one thread manages more than one device:

一个线程管理多个设备时，创建 communicator 也必须使用 group 调用：

```c
ncclGroupStart();
for (int i=0; i<nLocalDevs; i++) {
  cudaSetDevice(device[i]);
  ncclCommInitRank(comms+i, nranks, commId, rank[i]);
}
ncclGroupEnd();
```

> Note: Contrary to NCCL 1.x, there is no need to set the CUDA device before every NCCL communication call within a group, but it is still needed when calling `ncclCommInitRank` within a group.

**注意：**与 NCCL 1.x 不同，在 group 内不需要每次通信调用前都设置 CUDA 设备；但在 group 内调用 `ncclCommInitRank` 时仍须设置。

> Related links: [`ncclGroupStart()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html#c.ncclGroupStart), [`ncclGroupEnd()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html#c.ncclGroupEnd).

相关 API：[`ncclGroupStart()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html#c.ncclGroupStart)、[`ncclGroupEnd()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html#c.ncclGroupEnd)。

#### Aggregated Operations｜聚合操作

> The group semantics can also be used to have multiple collective operations performed within a single NCCL launch. This is useful for reducing the launch overhead, in other words, latency, as it only occurs once for multiple operations. Init functions cannot be aggregated with other init functions, nor with communication functions.

Group 语义还能让多个集合操作在一次 NCCL launch 中执行。多个操作只承担一次启动开销，因此可以降低这部分延迟。初始化函数不能与其他初始化函数或通信函数进行这种聚合。

> Aggregation of collective operations can be done simply by having multiple calls to NCCL within a `ncclGroupStart` / `ncclGroupEnd` section.

要聚合集合操作，只需在 `ncclGroupStart`／`ncclGroupEnd` 区间内放入多个 NCCL 调用。

> In the following example, we launch one broadcast and two allReduce operations together as a single NCCL launch.

下例将一次 Broadcast 和两次 AllReduce 合在一次 NCCL launch 中启动。

```c
ncclGroupStart();
ncclBroadcast(sendbuff1, recvbuff1, count1, datatype, root, comm, stream);
ncclAllReduce(sendbuff2, recvbuff2, count2, datatype, comm, stream);
ncclAllReduce(sendbuff3, recvbuff3, count3, datatype, comm, stream);
ncclGroupEnd();
```

> It is permitted to combine aggregation with multi-GPU launch and use different communicators in a group launch as shown in the Management Of Multiple GPUs From One Thread topic. When combining multi-GPU launch and aggregation, `ncclGroupStart` and `ncclGroupEnd` can be either used once or at each level. The following example groups the allReduce operations from different layers and on multiple CUDA devices:

可以将聚合与多 GPU 启动结合，并在同一次 group launch 中使用不同的 communicator，如前面的单线程管理多 GPU 示例所示。结合这两种用法时，`ncclGroupStart`／`ncclGroupEnd` 可以只在外层调用一次，也可以在每层调用。下例把不同层、多个 CUDA 设备上的 AllReduce 放入同一 group：

```c
ncclGroupStart();
for (int i=0; i<nlayers; i++) {
  ncclGroupStart();
  for (int g=0; g<ngpus; g++) {
    ncclAllReduce(sendbuffs[g]+offsets[i], recvbuffs[g]+offsets[i], counts[i], datatype[i], comms[g], streams[g]);
  }
  ncclGroupEnd();
}
ncclGroupEnd();
```

> Note: The NCCL operation will only be started as a whole during the last call to `ncclGroupEnd`. The `ncclGroupStart` and `ncclGroupEnd` calls within the for loop are not necessary and do nothing.

**注意：**只有最后一次调用 `ncclGroupEnd` 时，整个 NCCL 操作才会启动。示例中 `for` 循环内部的 `ncclGroupStart`／`ncclGroupEnd` 并非必需，也不起额外作用。

> Related links: [`ncclGroupStart()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html#c.ncclGroupStart), [`ncclGroupEnd()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html#c.ncclGroupEnd).

相关 API：[`ncclGroupStart()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html#c.ncclGroupStart)、[`ncclGroupEnd()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html#c.ncclGroupEnd)。

#### Group Operation Ordering Semantics｜group 操作的发起顺序

> Although NCCL group allows different operations to be issued in one shot, users still need to guarantee the same issuing order of the operations among different GPUs no matter whether the operations are issued to the same or different communicators.

虽然 NCCL group 允许一次发起不同操作，用户仍须保证各 GPU 上的操作发起顺序一致；这些操作使用同一个还是不同的 communicator，都不改变这一要求。

> For example, the following code provides the correct order of the operations. In this example, `comm0` and `comm1` are duplicated independent communicators that include rank 0 and 1.

下面的示例采用正确顺序。这里 `comm0` 与 `comm1` 是两个相互独立、由相同成员复制得到的 communicator，均包含 rank 0 和 rank 1。

```text
RANK0/GPU0/Process0:
ncclGroupStart();
ncclBroadcast(sendbuff1, recvbuff1, count1, datatype, root, comm0, stream);
ncclAllReduce(sendbuff2, recvbuff2, count2, datatype, comm0, stream);
ncclAllReduce(sendbuff3, recvbuff3, count3, datatype, comm0, stream);
ncclAllReduce(sendbuff4, recvbuff4, count4, datatype, comm1, stream);
ncclGroupEnd();

RANK1/GPU1/Process1:
ncclGroupStart();
ncclBroadcast(sendbuff1, recvbuff1, count1, datatype, root, comm0, stream);
ncclAllReduce(sendbuff2, recvbuff2, count2, datatype, comm0, stream);
ncclAllReduce(sendbuff3, recvbuff3, count3, datatype, comm0, stream);
ncclAllReduce(sendbuff4, recvbuff4, count4, datatype, comm1, stream);
ncclGroupEnd();
```

> However, changing the order of any operations will lead to incorrect results or hang as shown in the following 2 examples:

如果改变其中任何操作的顺序，就可能得到错误结果或发生挂起。下面有两个反例：

```text
RANK0/GPU0/Process0:
ncclGroupStart();
ncclBroadcast(sendbuff1, recvbuff1, count1, datatype, root, comm0, stream);
ncclAllReduce(sendbuff3, recvbuff3, count3, datatype, comm0, stream); // WRONG: reversed order
ncclAllReduce(sendbuff2, recvbuff2, count2, datatype, comm0, stream); // WRONG: reversed order
ncclAllReduce(sendbuff4, recvbuff4, count4, datatype, comm1, stream);
ncclGroupEnd();

RANK1/GPU1/Process1:
ncclGroupStart();
ncclBroadcast(sendbuff1, recvbuff1, count1, datatype, root, comm0, stream);
ncclAllReduce(sendbuff2, recvbuff2, count2, datatype, comm0, stream); // WRONG: reversed order
ncclAllReduce(sendbuff3, recvbuff3, count3, datatype, comm0, stream); // WRONG: reversed order
ncclAllReduce(sendbuff4, recvbuff4, count4, datatype, comm1, stream);
ncclGroupEnd();
```

```text
RANK0/GPU0/Process0:
ncclGroupStart();
ncclAllReduce(sendbuff4, recvbuff4, count4, datatype, comm1, stream); // WRONG: reversed order
ncclBroadcast(sendbuff1, recvbuff1, count1, datatype, root, comm0, stream);
ncclAllReduce(sendbuff2, recvbuff2, count2, datatype, comm0, stream);
ncclAllReduce(sendbuff3, recvbuff3, count3, datatype, comm0, stream);
ncclGroupEnd();

RANK1/GPU1/Process1:
ncclGroupStart();
ncclBroadcast(sendbuff1, recvbuff1, count1, datatype, root, comm0, stream);
ncclAllReduce(sendbuff2, recvbuff2, count2, datatype, comm0, stream);
ncclAllReduce(sendbuff3, recvbuff3, count3, datatype, comm0, stream);
ncclAllReduce(sendbuff4, recvbuff4, count4, datatype, comm1, stream); // WRONG: reversed order
ncclGroupEnd();
```

#### Nonblocking Group Operation｜非阻塞 group 操作

> If a communicator is marked as nonblocking through `ncclCommInitRankConfig`, the group functions become asynchronous correspondingly. In this case, if users issue multiple NCCL operations in one group, returning from `ncclGroupEnd()` might not mean the NCCL communication kernels have been issued to CUDA streams. If `ncclGroupEnd()` returns `ncclSuccess`, it means NCCL kernels have been issued to streams; if it returns `ncclInProgress`, it means NCCL kernels are being issued to streams in the background. It is users’ responsibility to make sure the state of the communicator changes into `ncclSuccess` before calling related CUDA calls (e.g. `cudaStreamSynchronize`):

如果通过 `ncclCommInitRankConfig` 将 communicator 设为非阻塞，group 函数也会相应变为异步。在一个 group 中发起多个 NCCL 操作时，`ncclGroupEnd()` 返回不一定表示通信 kernel 已加入 CUDA stream。若返回 `ncclSuccess`，表示 kernel 已加入 stream；若返回 `ncclInProgress`，表示 NCCL 仍在后台完成提交。调用相关 CUDA 函数（如 `cudaStreamSynchronize`）之前，用户必须确保 communicator 状态已变为 `ncclSuccess`：

```c
ncclGroupStart();
for (int g=0; g<ngpus; g++) {
  ncclAllReduce(sendbuffs[g]+offsets[i], recvbuffs[g]+offsets[i], counts[i], datatype[i], comms[g], streams[g]);
}
ret = ncclGroupEnd();
if (ret == ncclInProgress) {
  for (int g=0; g<ngpus; g++) {
    do {
      ncclCommGetAsyncError(comms[g], &state);
    } while (state == ncclInProgress);
  }
} else if (ret == ncclSuccess) {
  /* Successfully issued */
  printf("NCCL kernel issue succeeded\n");
} else {
  /* Errors happen */
  reportErrorAndRestart();
}

for (int g=0; g<ngpus; g++) {
  cudaStreamSynchronize(streams[g]);
}
```

> Related links: [`ncclCommInitRankConfig()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#c.ncclCommInitRankConfig), [`ncclCommGetAsyncError()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#c.ncclCommGetAsyncError).

相关 API：[`ncclCommInitRankConfig()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#c.ncclCommInitRankConfig)、[`ncclCommGetAsyncError()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#c.ncclCommGetAsyncError)。

### 重点解读

**Group 同时用于正确性和性能。** 单线程逐个管理多个 GPU 时，若第一条 NCCL 调用在等待其他 rank，就无法走到后续设备；group 让这些调用一起提交。聚合多次集合操作则主要减少启动开销。示例中的多次 `ncclCommInitRank` 可以放入 group，以便单线程完成多设备初始化，但不能把初始化函数与其他初始化或通信函数当作普通操作聚合。

**Group 内函数返回与 CUDA stream 提交要分开判断。** 普通调用的 stream 语义是“提交成功后返回”；原文在这里明确给出 group 内的例外：内部操作可能先返回，到 `ncclGroupEnd` 时才整体提交。若 communicator 是非阻塞的，`ncclGroupEnd` 返回 `ncclInProgress` 时还要继续查询状态；只有确认达到 `ncclSuccess` 后，才能执行依赖已提交工作的 CUDA stream 调用。`ncclSuccess` 在这里表示已提交到 stream，不表示设备端通信已经完成。

**嵌套 group 只在最外层结束时启动。** 多层循环示例中的内层 `ncclGroupStart`／`ncclGroupEnd` 不会各启动一次 NCCL 操作。需要核对的是各 rank 的最终操作序列：同一 communicator 内交换两个 AllReduce 的顺序，或跨 communicator 改变相对顺序，都可能使不同 GPU 的调用无法正确配对。

**示例代码仍需应用补全错误处理。** 非阻塞示例展示了 `ncclInProgress` 的轮询方向，但没有完整处理轮询调用本身的返回值、超时和失败恢复。实际代码在开始 `cudaStreamSynchronize` 前，应分别确认相关 communicator 已进入 `ncclSuccess`，而不是仅仅退出轮询循环。

## Point-to-point communication

来源：[NVIDIA NCCL User Guide — Point-to-point communication](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html)。

### 中英文对照

#### Two-sided communication｜双边通信

> (Since NCCL 2.7) Point-to-point communication can be used to express any communication pattern between ranks. Any point-to-point communication needs two NCCL calls: a call to `ncclSend()` on one rank and a corresponding `ncclRecv()` on the other rank, with the same count and data type.

（自 NCCL 2.7 起）点对点通信可用于表达 rank 之间的任意通信模式。每次双边点对点通信都需要两个匹配的 NCCL 调用：一个 rank 调用 `ncclSend()`，另一个 rank 调用对应的 `ncclRecv()`，两端的 `count` 和数据类型必须相同。

> Multiple calls to `ncclSend()` and `ncclRecv()` targeting different peers can be fused together with `ncclGroupStart()` and `ncclGroupEnd()` to form more complex communication patterns such as one-to-all (scatter), all-to-one (gather), all-to-all or communication with neighbors in an N-dimensional space.

面向不同 peer 的多次 `ncclSend()` 和 `ncclRecv()` 可以用 `ncclGroupStart()` 与 `ncclGroupEnd()` 合成一组，表达一对多（scatter）、多对一（gather）、全对全，以及 N 维空间中相邻节点之间的通信等复杂模式。

> Point-to-point calls within a group will be blocking until that group of calls completes, but calls within a group can be seen as progressing independently, hence should never block each other. It is therefore important to merge calls that need to progress concurrently to avoid deadlocks. The only exception is point-to-point calls within a group targeting the same peer, which are executed in order.

Group 中的点对点调用会阻塞，直到这一组调用完成；但组内各调用可视为独立推进，因此不应互相阻塞。需要同时推进的调用应放入同一个 group，以避免死锁。唯一的例外是组内面向同一 peer 的点对点调用，它们会按顺序执行。

> Below are a few examples of classic point-to-point communication patterns used by parallel applications. NCCL semantics allow for all variants with different sizes, datatypes, and buffers, per rank.

下面给出并行应用中常见的点对点通信模式。NCCL 语义允许不同 rank 使用不同的数据大小、数据类型和缓冲区来构造这些模式。

##### Sendrecv

> In MPI terms, a sendrecv operation is when two ranks exchange data, both sending and receiving at the same time. This can be done by merging both `ncclSend` and `ncclRecv` calls into one:

按 MPI 术语，sendrecv 是两个 rank 同时发送和接收、相互交换数据的操作。可把 `ncclSend` 与 `ncclRecv` 放入同一个 group 来实现：

```c
ncclGroupStart();
ncclSend(sendbuff, sendcount, sendtype, peer, comm, stream);
ncclRecv(recvbuff, recvcount, recvtype, peer, comm, stream);
ncclGroupEnd();
```

##### One-to-all (scatter)｜一对多散发

> A one-to-all operation from a root rank can be expressed by merging all send and receive operations in a group:

从 root rank 向所有 rank 的一对多操作，可通过把所有发送和接收调用放入同一个 group 表达：

```c
ncclGroupStart();
if (rank == root) {
  for (int r=0; r<nranks; r++)
    ncclSend(sendbuff[r], size, type, r, comm, stream);
}
ncclRecv(recvbuff, size, type, root, comm, stream);
ncclGroupEnd();
```

##### All-to-one (gather)｜多对一收集

> Similarly, an all-to-one operation to a root rank would be implemented this way:

类似地，向 root rank 收集数据的多对一操作可按下述方式实现：

```c
ncclGroupStart();
if (rank == root) {
  for (int r=0; r<nranks; r++)
    ncclRecv(recvbuff[r], size, type, r, comm, stream);
}
ncclSend(sendbuff, size, type, root, comm, stream);
ncclGroupEnd();
```

##### All-to-all｜全对全

> An all-to-all operation would be a merged loop of send/recv operations to/from all peers:

全对全操作可由一个面向所有 peer 的发送／接收循环组成，并放在同一 group 中：

```c
ncclGroupStart();
for (int r=0; r<nranks; r++) {
  ncclSend(sendbuff[r], sendcount, sendtype, r, comm, stream);
  ncclRecv(recvbuff[r], recvcount, recvtype, r, comm, stream);
}
ncclGroupEnd();
```

##### Neighbor exchange｜邻居交换

> Finally, exchanging data with neighbors in an N-dimensional space could be done with:

在 N 维空间中与相邻 rank 交换数据，可按下述方式实现：

```c
ncclGroupStart();
for (int d=0; d<ndims; d++) {
  ncclSend(sendbuff[d], sendcount, sendtype, next[d], comm, stream);
  ncclRecv(recvbuff[d], recvcount, recvtype, prev[d], comm, stream);
}
ncclGroupEnd();
```

#### One-sided communication｜单边通信

> (Since NCCL 2.29) One-sided communication enables a rank to write data to remote memory using `ncclPutSignal()` without requiring the target rank to issue a matching operation. The target memory must be pre-registered using `ncclCommWindowRegister()`. Point-to-point synchronization can be achieved by having the target rank call `ncclWaitSignal()` to wait for signals.

（自 NCCL 2.29 起）单边通信允许一个 rank 用 `ncclPutSignal()` 向远端内存写入数据，不要求目标 rank 发起匹配的操作。目标内存必须先通过 `ncclCommWindowRegister()` 注册。目标 rank 可以调用 `ncclWaitSignal()` 等待 signal，从而完成点对点同步。

> Multiple `ncclPutSignal()` calls can be grouped using `ncclGroupStart()` and `ncclGroupEnd()`. Operations to different peers or contexts within a group may execute concurrently and complete in any order. The completion of `ncclGroupEnd()` guarantees that all operations in the group have achieved completion. Operations to the same peer and context are executed in order: both data delivery and signal updates on the remote peer follow the program order.

多次 `ncclPutSignal()` 可以用 `ncclGroupStart()` 和 `ncclGroupEnd()` 组成一个 group。组内面向不同 peer 或不同 context 的操作可以并发执行，完成顺序也可以不同。`ncclGroupEnd()` 完成时，保证组内所有操作都已完成。面向同一个 peer、同一个 context 的操作则按顺序执行：远端的数据送达和 signal 更新都遵循程序顺序。

> The communication context is selected by the `ctx` argument and must be in [0, numRmaCtx), where `numRmaCtx` is fixed at communicator creation (default 1). Multiple communication contexts are supported starting with NCCL 2.31; earlier versions support only `ctx` 0.

通信 context 由 `ctx` 参数选择，取值必须在 `[0, numRmaCtx)` 内；`numRmaCtx` 在创建 communicator 时确定，默认值为 1。从 NCCL 2.31 开始支持多个通信 context；更早的版本仅支持 `ctx` 为 0。

> For best performance, provision multiple contexts and distribute traffic across them – for example, split a large message into chunks and put each chunk on a different context. The number of contexts that best saturates the network is platform-dependent; provisioning at least one per local NIC is a reasonable starting point. Distributing traffic pays off only for large transfers: each additional context adds launch and progress overhead, which dominates for small messages. Keep small transfers on a single context and distribute only large ones; the crossover size is platform-dependent.

为了获得较好性能，可以配置多个 context 并将流量分摊给它们，例如把大消息拆成多个分块，让每块使用不同的 context。使网络充分利用所需的 context 数取决于平台；每个本地 NIC 至少配置一个 context 是合理的起点。分摊流量只对大传输有利：新增 context 会带来启动和推进开销，小消息时这些开销占主导。因此，小传输宜保留在单个 context 上，只将大传输分摊出去；两者的分界大小也取决于平台。

> **Note**
>
> `ncclSignal()` and `ncclWaitSignal()` used without any prior `ncclCommWindowRegister()` require `rmaEagerInit` (or `NCCL_RMA_EAGER_INIT`) set to 1; otherwise they return `ncclInvalidUsage`.

**注意：**若此前没有调用过 `ncclCommWindowRegister()`，使用 `ncclSignal()` 和 `ncclWaitSignal()` 时必须将 `rmaEagerInit`（或 `NCCL_RMA_EAGER_INIT`）设为 1；否则会返回 `ncclInvalidUsage`。

> Below are a few examples of classic one-sided communication patterns used by parallel applications.

下面给出并行应用中常见的几种单边通信模式。

##### PutSignal and WaitSignal

> A ping-pong pattern using `ncclPutSignal()` and `ncclWaitSignal()`. This example shows the full setup including memory allocation and window registration:

下面用 `ncclPutSignal()` 与 `ncclWaitSignal()` 实现 ping-pong 模式，并展示包含内存分配、window 注册在内的完整设置流程：

```c
// Allocate symmetric memory for RMA operations
void *sendbuff, *recvbuff;
NCCLCHECK(ncclMemAlloc((void**)&sendbuff, size));
NCCLCHECK(ncclMemAlloc((void**)&recvbuff, size));

// Register buffers as symmetric windows
ncclWindow_t sendWindow, recvWindow;
NCCLCHECK(ncclCommWindowRegister(comm, sendbuff, size, &sendWindow, NCCL_WIN_COLL_SYMMETRIC));
NCCLCHECK(ncclCommWindowRegister(comm, recvbuff, size, &recvWindow, NCCL_WIN_COLL_SYMMETRIC));

int peer = (rank == 0) ? 1 : 0;
ncclWaitSignalDesc_t waitDesc = {.opCnt = 1, .peer = peer, .sigIdx = 0, .ctx = ctx};

if (rank == 0) {
  // Rank 0: wait then put
  NCCLCHECK(ncclWaitSignal(1, &waitDesc, comm, stream));
  NCCLCHECK(ncclPutSignal(sendbuff, count, datatype, peer, recvWindow, 0,
                    0, 0, 0, comm, stream));
} else {
  // Rank 1: put then wait
  NCCLCHECK(ncclPutSignal(sendbuff, count, datatype, peer, recvWindow, 0,
                    0, 0, 0, comm, stream));
  NCCLCHECK(ncclWaitSignal(1, &waitDesc, comm, stream));
}

CUDACHECK(cudaStreamSynchronize(stream));

// Cleanup
NCCLCHECK(ncclCommWindowDeregister(comm, sendWindow));
NCCLCHECK(ncclCommWindowDeregister(comm, recvWindow));
NCCLCHECK(ncclMemFree(sendbuff));
NCCLCHECK(ncclMemFree(recvbuff));
```

##### Barrier

> A barrier pattern using `ncclSignal()` and `ncclWaitSignal()`. Each rank signals to all other ranks and waits for signals from all ranks:

下例用 `ncclSignal()` 和 `ncclWaitSignal()` 实现 barrier：每个 rank 向其他所有 rank 发 signal，再等待来自所有 rank 的 signal。

```c
ncclWaitSignalDesc_t *waitDescs = malloc(nranks * sizeof(ncclWaitSignalDesc_t));
for (int r = 0; r < nranks; r++) {
  waitDescs[r].opCnt = 1;
  waitDescs[r].peer = r;
  waitDescs[r].sigIdx = 0;
  waitDescs[r].ctx = 0;
}

ncclGroupStart();
for (int r = 0; r < nranks; r++) {
  ncclSignal(r, 0, 0, 0, comm, stream);
}
ncclGroupEnd();

ncclWaitSignal(nranks, waitDescs, comm, stream);
```

##### All-to-all｜全对全

> An all-to-all operation using `ncclPutSignal()`. Each rank sends data to all other ranks and waits for signals from all ranks. User needs to register the memory window for each peer using `ncclCommWindowRegister()` in advance. User needs to guarantee the buffers are ready before calling `ncclPutSignal()`. This could be done with the barrier shown above.

下例用 `ncclPutSignal()` 实现全对全：每个 rank 向其他所有 rank 发送数据，并等待来自所有 rank 的 signal。用户须预先用 `ncclCommWindowRegister()` 为每个 peer 注册内存 window，并在调用 `ncclPutSignal()` 前确保缓冲区已准备好；上面的 barrier 可用于完成这一步同步。

```c
size_t offset[nranks];
ncclWaitSignalDesc_t *waitDescs = malloc(nranks * sizeof(ncclWaitSignalDesc_t));
for (int r = 0; r < nranks; r++) {
  offset[r] = r * count * wordSize(datatype);
  waitDescs[r].opCnt = 1;
  waitDescs[r].peer = r;
  waitDescs[r].sigIdx = 0;
  waitDescs[r].ctx = 0;
}

ncclGroupStart();
for (int r = 0; r < nranks; r++) {
  ncclPutSignal(sendbuff[r], count, datatype, r, window, offset[r],
          0, 0, 0, comm, stream);
}
ncclGroupEnd();

ncclWaitSignal(nranks, waitDescs, comm, stream);
```

### 重点解读

**双边通信先要配对，再谈模式。** 每个 `ncclSend()` 都需要另一 rank 上对应的 `ncclRecv()`，两端的元素数量和数据类型一致。不同配对之间可以有不同大小、类型和缓冲区；scatter、gather、all-to-all 等模式只是将多组配对放入 group。需要共同推进的发送和接收应在同一个 group 中，避免某个调用等待另一个尚未发出的调用。面向同一 peer 的组内操作仍按顺序执行。

**单边写入依赖远端 window 与 signal。** `ncclPutSignal()` 无需目标 rank 发起匹配的接收调用，但远端内存要事先注册。目标 rank 可用 `ncclWaitSignal()` 等待完成通知；同一 peer、同一 context 的数据和 signal 依程序顺序到达，不同 peer 或 context 的完成顺序则不保证。原文关于 `ncclGroupEnd()` 完成即保证组内操作完成的说法，属于此处单边操作的语义，不应直接套用于前面普通集合通信的 stream 入队规则。

**context 数量需要按消息规模选。** 多 context 有利于把大传输分散到不同通道，但也带来启动和推进开销。小消息先用单个 context；大消息再按 NIC 与实际平台测试增加 context。`ctx` 必须落在 communicator 创建时确定的 `[0, numRmaCtx)` 范围内。

**示例展示流程，仍需补齐应用代码。** Ping-pong 示例给出了分配、注册、等待、写入和清理的顺序；barrier 示例展示 signal／wait 序列；全对全片段还使用了未在片段中定义的缓冲区、window 和辅助函数。在全对全写入前，需要确保远端 window 已注册且缓冲区可用。实际使用前还应核对各 rank 的 signal 描述符、错误处理和资源释放。

## Thread Safety

来源：[NVIDIA NCCL User Guide — Thread Safety](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/threadsafety.html#thread-safety)。

### 中英文对照

> NCCL primitives are generally not thread-safe, however, they are reentrant. Under multi-thread environment, it is not allowed to issue NCCL operations to a single communicator in parallel with multiple threads; it is not safe to issue NCCL operations in parallel to independent communicators located on the same device with multiple threads (see [Using multiple NCCL communicators concurrently](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#multi-thread-concurrent-usage)). If the child communicator shares the resources with the parent communicator (i.e., [`ncclConfig_t`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#ncclconfig) by `splitShare`), it is not allowed to issue NCCL operations to the child and parent communicators in parallel.

NCCL 原语通常不是 thread-safe，但具有 reentrant（可重入）特性。在多线程环境下，不允许多个线程同时向同一个 communicator 发起 NCCL 操作；多个线程也不能安全地同时操作位于同一设备上的不同 communicator（参见 [Using multiple NCCL communicators concurrently](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#multi-thread-concurrent-usage)）。如果子 communicator 与父 communicator 共享资源（即通过 [`ncclConfig_t`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#ncclconfig) 的 `splitShare` 设置），也不允许并发操作父子 communicator。

> It is safe to operate a communicator from multiple threads as long as users can guarantee only one thread operates the communicator at a time. However, for any grouped NCCL operations, users need to ensure only one thread issues all the operations in the group.

只要能保证任一时刻只有一个线程操作某个 communicator，就可以由多个线程轮流使用它。但对于任何 NCCL group 操作，必须由同一个线程发起该 group 内的全部操作。

> For example, the following code provides a simple thread-safe example where threads are executed in sequence and only one thread is accessing the communicator at a time.

下面的简单示例中，线程依次执行，任一时刻只有一个线程访问 communicator，因此符合上述线程安全条件。

```text
Thread 0:
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = 0;
  cudaSetDevice(0);
  ncclCommInitRankConfig(&comm, nranks, id, rank, &config);
  ncclGroupStart();
  ncclAllReduce(sendbuff0, recvbuff0, count0, datatype, redOp, comm, stream);
  ncclAllReduce(sendbuff1, recvbuff1, count1, datatype, redOp, comm, stream);
  ncclGroupEnd();
  thread_exit();

Thread 1:
  ncclResult_t state = ncclSuccess;
  // wait for previous issued allreduce ops by Thread 0
  do {
    ncclCommGetAsyncError(comm, &state);
  } while (state == ncclInProgress);
  assert(state == ncclSuccess);
  ncclAllReduce(sendbuff2, recvbuff2, count2, datatype, redOp, comm, stream);
  do {
    ncclCommGetAsyncError(comm, &state);
  } while (state == ncclInProgress);
  assert(state == ncclSuccess);
```

> It is also valid to issue grouped NCCL operations from one thread and poll the status of each NCCL communicator with one thread as shown in the following code.

也可以由一个线程发起 group 内的 NCCL 操作，再由各线程分别轮询所负责的 NCCL communicator 状态，如下例所示。

```text
Thread 0:
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = 0;
  ncclGroupStart();
  for (int i = 0; i < nGpus; i++) {
    cudaSetDevice(i);
    ncclCommInitRankConfig(&comms[i], nranks, id, ranks[i], &config);
  }
  ncclGroupEnd();

Thread 0/1/2/3:
  ncclResult_t state = ncclSuccess;
  // wait for previous issued init ops by Thread 0
  do {
    ncclCommGetAsyncError(comms[thread_id], &state);
  } while (state == ncclInProgress);
  assert(state == ncclSuccess);
  ncclAllReduce(sendbuff, recvbuff, count, datatype, redOp, comms[thread_id], stream);
  do {
    ncclCommGetAsyncError(comms[thread_id], &state);
  } while (state == ncclInProgress);
  assert(state == ncclSuccess);
```

### 重点解读

**可重入并不等于可以任意并发。** 原文给出的安全边界取决于 communicator、设备以及是否共享资源。多个线程可以先后使用同一个 communicator，但不能同时操作它；不同 communicator 若位于同一设备，或父子 communicator 共享资源，也不能据此认定并发操作安全。前面“并发使用多个 NCCL communicator”一节讨论操作发起顺序，本节进一步限制同设备上由多个 host 线程并行发起调用的做法。

| 多线程访问方式 | 原文给出的结论 |
| --- | --- |
| 多个线程同时操作同一 communicator | 不允许 |
| 多个线程依次操作同一 communicator，任一时刻只有一个线程访问 | 安全 |
| 多个线程同时操作同一设备上的不同 communicator | 不安全 |
| 同时操作共享资源的父子 communicator | 不允许 |
| 把同一 group 内的调用分散给多个线程发起 | 不允许；同一线程须发起全部调用 |

**两个示例都以明确的线程分工成立。** 第一个示例先由 Thread 0 发起 grouped AllReduce，线程退出后再由 Thread 1 访问同一个 communicator。第二个示例由 Thread 0 完成多设备的 grouped 初始化，随后各线程只操作各自负责的 communicator。示例中的 `ncclCommGetAsyncError` 轮询用于检查非阻塞 NCCL 状态；它不能代替 CUDA stream 或 event 对设备端操作完成情况的确认。

**代码是访问顺序示意。** 两段代码展示线程间职责与调用顺序，没有给出完整的线程创建、交接同步、错误处理和资源释放。实际程序必须确保“只有一个线程访问”这一条件由明确的同步机制维持，而不只是依赖示例中的文本顺序。

## In-place Operations

来源：[NVIDIA NCCL User Guide — In-place Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/inplace.html#in-place-operations)。

### 中英文对照

> Contrary to MPI, NCCL does not define a special “in-place” value to replace pointers. Instead, NCCL optimizes the case where the provided pointers are effectively “in place”.

与 MPI 不同，NCCL 没有定义一个用于代替指针的特殊“in-place”值。NCCL 会识别并优化所传指针实际上构成原地操作的情况。

> For `ncclBroadcast`, `ncclReduce` and `ncclAllreduce` functions, this means that passing `sendBuff == recvBuff` will perform in place operations, storing final results at the same place as initial data was read from.

对于 `ncclBroadcast`、`ncclReduce` 和原文所写的 `ncclAllreduce`，传入 `sendBuff == recvBuff` 就会执行原地操作：最终结果保存在读取初始数据的同一位置。

> For `ncclReduceScatter` and `ncclAllGather`, in place operations are done when the per-rank pointer is located at the rank offset of the global buffer. More precisely, these calls are considered in place:

对于 `ncclReduceScatter` 和 `ncclAllGather`，当每个 rank 使用的指针位于全局缓冲区中对应 rank 的偏移位置时，操作才被视为原地执行。更准确地说，原文给出以下两种调用：

```c
ncclReduceScatter(data, data+rank*recvcount, recvcount, datatype, op, comm, stream);
ncclAllGather(data+rank*sendcount, data, sendcount, datatype, op, comm, stream);
```

### 重点解读

**“原地”取决于指针关系。** Broadcast、Reduce 和 AllReduce 的条件是发送与接收指针相同；ReduceScatter 与 AllGather 的条件则是其中一个指针恰好指向总缓冲区中本 rank 的分块，两个参数不必是同一个地址。`rank*recvcount` 与 `rank*sendcount` 的指针运算单位取决于 `data` 的元素类型，实际代码须让指针类型、`datatype` 和缓冲区布局一致，并为所有 rank 的分块留足空间。

**原文有两处 API 差异。** 实际函数名是 `ncclAllReduce`，而原文写成了 `ncclAllreduce`。上面保留的 `ncclAllGather` 示例则多了一个 `op` 参数；[NCCL Collective Communication Functions](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclAllGather) 中的函数签名没有这个参数。按 API 签名，第二行应写为：

```c
ncclAllGather(data+rank*sendcount, data, sendcount, datatype, comm, stream);
```

此处讨论的是原文列出的几种操作，不能由它推断其他 NCCL 集合操作也支持原地执行。

## Using NCCL with CUDA Graphs

来源：[NVIDIA NCCL User Guide — Using NCCL with CUDA Graphs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html#using-nccl-with-cuda-graphs)。

### 中英文对照

> Starting with NCCL 2.9, NCCL operations can be captured by CUDA Graphs.

从 NCCL 2.9 开始，NCCL 操作可以被 CUDA Graphs 捕获。

> CUDA Graphs provide a way to define workflows as graphs rather than single operations. They may reduce overhead by launching multiple GPU operations through a single CPU operation. More details about CUDA Graphs can be found in the [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs).

CUDA Graphs 允许将工作流程定义为图，而非逐个定义独立操作。通过一次 CPU 操作启动多个 GPU 操作，可以减少启动开销。有关 CUDA Graphs 的更多内容，可参阅 [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)。

> NCCL’s collective, P2P and group operations all support CUDA Graph captures. This support requires a minimum CUDA version of 11.3.

NCCL 的集合通信、点对点通信以及 group 操作都支持 CUDA Graph 捕获。该功能要求 CUDA 版本至少为 11.3。

#### Requirements and Limitations｜要求与限制

> Using multiple GPUs per process with CUDA graph capture may result in deadlocks. A deadlock is likely to occur with single-threaded applications: in some cases `cudaGraphLaunch` may block, preventing the launch across all GPUs. We recommend running with a single GPU per process for the most reliable experience.

在单个进程中使用多个 GPU 并进行 CUDA Graph 捕获可能导致死锁。单线程应用尤其容易遇到这种情况：某些时候 `cudaGraphLaunch` 可能阻塞，使程序无法在所有 GPU 上完成 graph 启动。原文建议每个进程只使用一个 GPU，以获得更可靠的运行行为。

> Whether an operation launch is graph-captured is considered a collective property of that operation and therefore must be uniform over all ranks participating in the launch (for collectives this is all ranks in the communicator, for peer-to-peer this is both the sender and receiver). The launch of a graph (via `cudaGraphLaunch`, etc.) containing a captured NCCL operation is considered collective for the same set of ranks that were present in the capture, and each of those ranks must be using the graph derived from that collective capture.

一次操作是否通过 CUDA Graph 捕获并启动，被视为该操作的集体属性，因此所有参与启动的 rank 必须保持一致：集合通信涉及 communicator 内的所有 rank，点对点通信涉及发送端和接收端。通过 `cudaGraphLaunch` 等方式启动包含已捕获 NCCL 操作的 graph，也被视为同一组 rank 共同参与的操作；这些 rank 都必须使用由那次共同捕获得到的 graph。

> The following sample code shows how to capture computational kernels and NCCL operations in a CUDA Graph:

下面的示例展示如何将计算 kernel 和 NCCL 操作一并捕获到 CUDA Graph 中：

```c
cudaGraph_t graph;
cudaStreamBeginCapture(stream);
kernel_A<<< ..., stream >>>(...);
kernel_B<<< ..., stream >>>(...);
ncclAllreduce(..., stream);
kernel_C<<< ..., stream >>>(...);
cudaStreamEndCapture(stream, &graph);

cudaGraphExec_t instance;
cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
cudaGraphLaunch(instance, stream);
cudaStreamSynchronize(stream);
```

> Starting with NCCL 2.11, when NCCL communication is captured and the CollNet algorithm is used, NCCL allows for further performance improvement via user buffer registration. For details, please see the environment variable [NCCL_GRAPH_REGISTER](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-graph-register).

从 NCCL 2.11 开始，如果 NCCL 通信已被 graph 捕获，且使用 CollNet 算法，还可以通过注册用户缓冲区进一步提高性能。详情参阅环境变量 [NCCL_GRAPH_REGISTER](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-graph-register)。

> Mixing graph-captured and non-graph-captured NCCL operations is supported by NCCL. However, when graphs involving multiple communicators are `cudaGraphLaunch`’d from the same thread, the internal mechanism NCCL uses to support this mixing can contribute to the deadlocks described above. To disable this mechanism, see the environment variable [NCCL_GRAPH_MIXING_SUPPORT](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-graph-mixing-support).

NCCL 支持混合使用被 graph 捕获和未被 graph 捕获的操作。不过，如果同一个线程通过 `cudaGraphLaunch` 启动涉及多个 communicator 的 graph，NCCL 用于支持这种混合使用的内部机制可能加剧前述死锁风险。有关禁用该机制的方法，参阅环境变量 [NCCL_GRAPH_MIXING_SUPPORT](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-graph-mixing-support)。

> Disabling NCCL’s capture-time serialization of communication kernels (see [NCCL_GRAPH_STREAM_ORDERING](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-graph-stream-ordering) and `graphStreamOrdering` in [ncclConfig_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#ncclconfig)) together with graph mixing (communicator `graphUsageMode=2`) is **not supported**. If ordering is disabled for a communicator, **graph mixing must be off** (`graphUsageMode` `0` or `1`); workloads that need mixing must keep the default ordering (`1`).

**不支持**同时关闭 NCCL 在捕获阶段对通信 kernel 的串行排序（参阅 [NCCL_GRAPH_STREAM_ORDERING](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-graph-stream-ordering) 和 [ncclConfig_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#ncclconfig) 中的 `graphStreamOrdering`），又启用 graph mixing（communicator 的 `graphUsageMode=2`）。如果某个 communicator 关闭了排序，**必须关闭 graph mixing**（`graphUsageMode` 为 `0` 或 `1`）；需要混合使用的工作负载则必须保留默认排序设置 `1`。

### 重点解读

**Graph 捕获和启动都需要参与 rank 协调。** 同一次 NCCL 操作的各参与 rank 必须一致选择 graph 捕获方式；之后启动所捕获 graph 时，也要由原先参与捕获的同一组 rank 配合，不能仅在部分 rank 上调用 `cudaGraphLaunch`。点对点操作同样要求发送端与接收端保持一致。

**单进程多 GPU 的风险发生在 graph 启动阶段。** 如果一个线程要依次为多个 GPU 调用 `cudaGraphLaunch`，前一次调用可能先阻塞，导致后续 GPU 永远无法启动。原文因此推荐一个进程对应一个 GPU；这是一项可靠性建议，不能理解为所有单进程多 GPU 用法都会死锁。

**Graph mixing 与 stream ordering 有组合限制。** 一般情况下可以混用捕获与未捕获的 NCCL 操作，但同线程启动涉及多个 communicator 的 graph 时须留意前述死锁风险；若通过配置关闭捕获阶段的通信 kernel 排序，就不能再将 `graphUsageMode` 设为 `2`。需要 graph mixing 的工作负载应维持默认排序值 `1`。

**示例代码保留原文拼写。** 官方页面示例写作 `ncclAllreduce`，而 [NCCL 集合通信 API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclAllReduce) 中的实际函数名为 `ncclAllReduce`。示例中的省略号只是占位符，使用时还需填写完整调用参数。

## User Buffer Registration

来源：[NVIDIA NCCL User Guide — User Buffer Registration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#user-buffer-registration)。

### 中英文对照

> User Buffer Registration is a feature that allows NCCL to directly send/receive/operate data through the user buffer without extra internal copy (zero-copy). It can accelerate collectives and greatly reduce the resource usage (e.g. #channel usage). NCCL provides two ways to register user buffers; one is CUDA Graph registration, and the other is Local registration.

User Buffer Registration 允许 NCCL 直接在用户缓冲区上发送、接收或处理数据，避免额外的内部拷贝（zero-copy）。它可以加速集合通信，并显著减少资源占用，例如 channel 的使用量。NCCL 提供两种用户缓冲区注册方式：CUDA Graph 注册和 Local 注册。

> NCCL requires that for all NCCL communication function calls (e.g., allreduce, sendrecv, and so on), if any rank in a communicator passes registered buffers to a NCCL communication function, all other ranks in the same communicator must pass their registered buffers; otherwise, mixing registered and non-registered buffers can result in undefined behavior; in addition, source and destination buffers must be registered in order to enable user buffer registration for NCCL operations.

对于 NCCL 通信函数调用（例如 AllReduce、Send/Recv），如果 communicator 中有任意 rank 传入已注册缓冲区，同一 communicator 中的其他 rank 也必须传入各自已注册的缓冲区；混用已注册与未注册缓冲区可能导致未定义行为。此外，要为 NCCL 操作启用用户缓冲区注册，源缓冲区和目标缓冲区都必须注册。

#### NVLink Sharp Buffer Registration｜NVLink Sharp 缓冲区注册

> Since 2.19.x, NCCL supports user buffer registration for NVLink Sharp (NVLS); any NCCL collectives (e.g., allreduce) that support NVLS algorithm can utilize this feature.

从 NCCL 2.19.x 起，NVLink Sharp（NVLS）支持用户缓冲区注册；任何支持 NVLS 算法的 NCCL 集合操作（例如 AllReduce）都可以利用此功能。

> To enable the CUDA Graph based buffer registration for NVLS, users have to comply with several requirements:

要通过 CUDA Graph 为 NVLS 启用缓冲区注册，需要满足以下要求：

> - The buffer is allocated through `ncclMemAlloc()` or a qualified allocator (see [Memory Allocator](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#memory-allocator)).
> - The NCCL operation is launched on a stream captured by a CUDA graph for each rank.
> - Offset to the head address of the buffer is the same in collectives for each rank.

- 使用 `ncclMemAlloc()` 或符合要求的 allocator 分配缓冲区（参见下文 [Memory Allocator](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#memory-allocator)）。
- 每个 rank 的 NCCL 操作都在被 CUDA Graph 捕获的 stream 上启动。
- 在集合操作中，各 rank 所用数据相对于其缓冲区起始地址的偏移必须相同。

> Registered buffers will be deregistered when the CUDA graph is destroyed. Here is a CUDA graph based buffer registration example:

销毁 CUDA Graph 时，已注册的缓冲区会解除注册。下面是基于 CUDA Graph 的缓冲区注册示例：

```c
void* sendbuff;
void* recvbuff;
size_t count = 1 << 25;
CHECK(ncclMemAlloc(&sendbuff, count * sizeof(float)));
CHECK(ncclMemAlloc(&recvbuff, count * sizeof(float)));

cudaGraph_t graph;
CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
CHECK(ncclAllReduce(sendbuff, recvbuff, 1024, ncclFloat, ncclSum, comm, stream));
// Same offset to the sendbuff and recvbuff head address for each rank
CHECK(ncclAllReduce((void*)((float*)sendbuff + 1024), (void*)((float*)recvbuff + 2048), 1024, ncclFloat, ncclSum, comm, stream));
CHECK(cudaStreamEndCapture(stream, &graph));

cudaGraphExec_t instance;
CHECK(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
CHECK(cudaGraphLaunch(instance, stream));
CHECK(cudaStreamSynchronize(stream));
CHECK(cudaGraphExecDestroy(instance));
CHECK(cudaGraphDestroy(graph));

CHECK(ncclMemFree(sendbuff));
CHECK(ncclMemFree(recvbuff));
```

> On the other hand, to enable the Local based buffer registration for NVLS, users have to comply with the following requirements:

若通过 Local 注册为 NVLS 启用缓冲区注册，则需要满足以下要求：

> - The buffer is allocated through `ncclMemAlloc()` or a qualified allocator (see [Memory Allocator](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#memory-allocator)).
> - Register buffer with `ncclCommRegister()` before calling collectives for each rank.
> - Call NCCL collectives as usual but similarly keep the offset to the head address of the buffer the same for each rank.

- 使用 `ncclMemAlloc()` 或符合要求的 allocator 分配缓冲区（参见下文 [Memory Allocator](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#memory-allocator)）。
- 每个 rank 在调用集合操作前，先用 `ncclCommRegister()` 注册缓冲区。
- 按常规方式调用 NCCL 集合操作，但仍须保持各 rank 相对于缓冲区起始地址的偏移一致。

> Registered buffers will be deregistered when users explicitly call `ncclCommDeregister()`. Here is a local based buffer registration example:

用户显式调用 `ncclCommDeregister()` 时，已注册缓冲区才会解除注册。下面是基于 Local 注册的示例：

```c
void* sendbuff;
void* recvbuff;
size_t count = 1 << 25;
void* sendRegHandle;
void* recvRegHandle;
CHECK(ncclMemAlloc(&sendbuff, count * sizeof(float)));
CHECK(ncclMemAlloc(&recvbuff, count * sizeof(float)));

CHECK(ncclCommRegister(comm, sendbuff, count * sizeof(float), &sendRegHandle));
CHECK(ncclCommRegister(comm, recvbuff, count * sizeof(float), &recvRegHandle));

CHECK(ncclAllReduce(sendbuff, recvbuff, 1024, ncclFloat, ncclSum, comm, stream));
CHECK(ncclAllReduce((void*)((float*)sendbuff + 1024), (void*)((float*)recvbuff + 2048), 1024, ncclFloat, ncclSum, comm, stream));
CHECK(cudaStreamSynchronize(stream));

CHECK(ncclCommDeregister(comm, sendRegHandle));
CHECK(ncclCommDeregister(comm, recvRegHandle));

CHECK(ncclMemFree(sendbuff));
CHECK(ncclMemFree(recvbuff));
```

> For local based registration, users can register the buffer once at the beginning of the program and reuse the buffer multiple times to utilize registration benefits.

对于 Local 注册，可以在程序开始时注册一次缓冲区，之后多次复用该缓冲区，以持续利用注册带来的收益。

> To save the memory, it is also valid to allocate a large chunk of buffer and register it once. `sendbuff` and `recvbuff` can be further allocated through the big chunk for zero-copy NCCL operations as long as `sendbuff` and `recvbuff` satisfy the offset requirements. The following example shows a use case:

为了节省内存，也可以一次分配并注册一大块缓冲区。只要 `sendbuff` 和 `recvbuff` 满足偏移要求，就可以在这块大缓冲区内划出它们，供 zero-copy NCCL 操作使用。示例如下：

```c
void* buffer;
void* handle;
void* sendbuff;
void* recvbuff;
size_t size = 1 << 29;

CHECK(ncclMemAlloc(&buffer, size));
CHECK(ncclCommRegister(comm, buffer, size, &handle));

// assign buffer chunk to sendbuff and recvbuff
sendbuff = buffer;
recvbuff = (void*)((uint8_t*)buffer + (1 << 20));

CHECK(ncclAllReduce(sendbuff, recvbuff, 1024, ncclFloat, ncclSum, comm, stream));
CHECK(ncclAllGather(sendbuff, recvbuff, 1024, ncclInt8, comm, stream));
CHECK(cudaStreamSynchronize(stream));

CHECK(ncclCommDeregister(comm, handle));

CHECK(ncclMemFree(sendbuff));
```

#### IB Sharp Buffer Registration｜IB Sharp 缓冲区注册

> NCCL 2.21.x supports IB Sharp buffer registration, any NCCL collectives that support IB Sharp algorithm can benefit from the feature such as allreduce, reducescatter, and allgather. Currently, NCCL only supports IB Sharp buffer registration for the communicators which contain 1 rank per node, and the registration can reduce the number of NCCL SM usage down to 1.

NCCL 2.21.x 支持 IB Sharp 缓冲区注册。支持 IB Sharp 算法的集合操作，例如 AllReduce、ReduceScatter 和 AllGather，都可以利用该功能。目前，IB Sharp 缓冲区注册只支持每个节点恰好有 1 个 rank 的 communicator；注册后，NCCL 占用的 SM 数量可以降至 1。

> To enable IB Sharp buffer registration by CUDA graph:

通过 CUDA Graph 启用 IB Sharp 缓冲区注册，需要：

> - Allocate send and recv buffer with any CUDA allocator (e.g., `cudaMalloc`/`ncclMemAlloc`).
> - Launch NCCL collectives with CUDA graph.

- 使用任意 CUDA allocator（例如 `cudaMalloc` 或 `ncclMemAlloc`）分配发送和接收缓冲区。
- 通过 CUDA Graph 启动 NCCL 集合操作。

> To enable IB Sharp buffer registration by local registration:

通过 Local 注册启用 IB Sharp 缓冲区注册，需要：

> - Allocate send and recv buffer with any CUDA allocator (e.g., `cudaMalloc`/`ncclMemAlloc`).
> - Register send and recv buffer for each rank in the communicator with `ncclCommRegister`.
> - Launch NCCL collectives.

- 使用任意 CUDA allocator（例如 `cudaMalloc` 或 `ncclMemAlloc`）分配发送和接收缓冲区。
- communicator 中的每个 rank 使用 `ncclCommRegister` 注册发送和接收缓冲区。
- 启动 NCCL 集合操作。

#### General Buffer Registration｜通用缓冲区注册

> Since 2.23.x, NCCL supports intra-node buffer registration, which targets all peer-to-peer intra-node communications (e.g., Allgather Ring) and brings less memory pressure, better communication and computation overlap performance. Either registering buffers by `ncclCommRegister` in the beginning or applying CUDA graph can enable intra-node buffer registration for NCCL collectives and sendrecv.

从 NCCL 2.23.x 起，NCCL 支持节点内缓冲区注册，面向节点内点对点通信（例如 AllGather Ring），可降低内存压力，并改善通信与计算的重叠效果。对于 NCCL 集合操作和 Send/Recv，可以在开始时通过 `ncclCommRegister` 注册缓冲区，或使用 CUDA Graph 来启用节点内缓冲区注册。

> The user buffers can be allocated through VMM API (i.e., `cuMem*`), any VMM-based allocators ([Memory Allocator](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#memory-allocator)) or `ncclMemAlloc` will work. The buffers allocated through legacy cuda API (e.g., `cudaMalloc`) can also be used for registration. However, it is not safe due to the potential hang during execution and segmentation fault during failure and abort, so using legacy buffers for registration is not recommended; currently, legacy buffer registration is disabled by default, users can set `NCCL_LEGACY_CUDA_REGISTER=1` to enable it.

用户缓冲区可以通过 VMM API（即 `cuMem*`）、基于 VMM 的 allocator（参见下文 [Memory Allocator](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#memory-allocator)）或 `ncclMemAlloc` 分配。传统 CUDA API（例如 `cudaMalloc`）分配的缓冲区也可以用于注册，但存在执行期间挂起，以及失败或中止期间发生段错误的风险，因此不建议注册此类缓冲区。传统缓冲区注册目前默认关闭；用户可设置 `NCCL_LEGACY_CUDA_REGISTER=1` 启用。

#### Buffer Registration, GPU Direct RDMA, and MPS with MLOPart｜缓冲区注册、GPU Direct RDMA 与 MPS with MLOPart

> To ensure optimal performance for scale-out communications, we recommend the usage of `ncclMemAlloc`, or alternatively `cuMemCreate` with the attribute `gpuDirectRDMACapable`. Failing to do so might force NCCL to use an internal staging buffer and therefore offset the gain provided by the user-buffer registration.

为获得跨节点扩展通信的理想性能，原文建议使用 `ncclMemAlloc`，或使用带有 `gpuDirectRDMACapable` 属性的 `cuMemCreate`。否则 NCCL 可能不得不使用内部暂存缓冲区，从而抵消用户缓冲区注册的收益。

> Further, mixing buffers allocated with different allocators maybe result in undefined behavior.

此外，混用不同 allocator 分配的缓冲区可能导致未定义行为。

#### Buffer Registration and PXN｜缓冲区注册与 PXN

> Buffer registration for network communication (e.g., InfiniBand) and PXN are inherently incompatible. PXN is enabled by default in NCCL as long as the platform supports it, and it can be used for sendrecv-based operations and collectives. When PXN is enabled, the network buffer registration will not be enabled even if users have called `ncclCommRegister` to register the buffers. To enable network buffer registration, users can set `NCCL_PXN_DISABLE=1` to disable PXN.

用于网络通信（例如 InfiniBand）的缓冲区注册与 PXN 在机制上不兼容。只要平台支持，NCCL 默认启用 PXN；它可用于基于 Send/Recv 的操作及集合操作。启用 PXN 时，即使调用 `ncclCommRegister` 注册缓冲区，网络缓冲区注册也不会生效。若要启用网络缓冲区注册，可以设置 `NCCL_PXN_DISABLE=1` 以关闭 PXN。

#### Memory Allocator｜内存分配器

> For convenience, NCCL provides `ncclMemAlloc` function to help users to allocate buffers through VMM API, which can be used for NCCL registration later. It is only designed for NCCL so that it is not recommended to use `ncclMemAlloc` allocated buffers everywhere in the applications.

为方便使用，NCCL 提供 `ncclMemAlloc`，帮助用户通过 VMM API 分配之后可用于 NCCL 注册的缓冲区。该函数专为 NCCL 设计，因此原文不建议在应用的所有位置都使用它分配缓冲区。

> For advanced users, if you want to create your own memory allocator for NVLS UB, the allocated buffer of the allocator needs to satisfy the following requirements:

对于要自行实现 NVLS UB 内存分配器的高级用户，所分配的缓冲区须满足以下要求：

> - Allocate buffer with shared flag `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR` and also `CU_MEM_HANDLE_TYPE_FABRIC` on GPUs where it’s supported.
> - Buffer physical memory size is multiple of CUMEM recommended granularity (i.e. `cuMemGetAllocationGranularity(…, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)`)
> - Buffer virtual head address is at least aligned to CUMEM recommended granularity and size is multiple of CUMEM recommended granularity.

- 分配缓冲区时指定共享句柄标志 `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`；在支持的 GPU 上还要指定 `CU_MEM_HANDLE_TYPE_FABRIC`。
- 物理内存大小为 CUMEM 建议粒度的整数倍（参见 `cuMemGetAllocationGranularity(…, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)`）。
- 虚拟地址的起始地址至少按 CUMEM 建议粒度对齐，大小也为该粒度的整数倍。

> For general buffer registration with VMM API, the allocator needs to satisfy the same requirements as NVLS UB allocators.

通过 VMM API 进行通用缓冲区注册时，allocator 也需要满足与 NVLS UB allocator 相同的要求。

#### Window Registration｜窗口注册

> Since 2.27, NCCL supports window registration, which allows users to register local buffers into NCCL window and enables extremely low latency and high bandwidth communication in NCCL. Currently, window registration supports input buffers only from VMM-based allocators ([Memory Allocator](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#memory-allocator)) and `ncclMemAlloc`; any other type of cuda buffers will fail to be registered.

从 NCCL 2.27 起，NCCL 支持 window registration：用户可以将本地缓冲区注册到 NCCL window 中，以实现极低延迟和高带宽通信。目前，window registration 只接受由基于 VMM 的 allocator（参见下文 [Memory Allocator](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#memory-allocator)）或 `ncclMemAlloc` 分配的输入缓冲区；其他类型的 CUDA 缓冲区无法完成注册。

> NCCL window registration is enabled by default. However, if users do not use window registration and need to turn it off, set `NCCL_WIN_ENABLE=0` to disable it. In addition, users can also control the behavior of window registration through flags in [Window Registration Flags](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/flags.html#window-registration-flags).

NCCL 默认启用 window registration。如需关闭，可设置 `NCCL_WIN_ENABLE=0`。用户也可以通过 [Window Registration Flags](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/flags.html#window-registration-flags) 中的标志控制注册行为。

> For the device API, symmetrically registered windows (e.g. with `NCCL_WIN_COLL_SYMMETRIC`) provide LSA (load/store accessible) memory: device code can access peer buffers via load/store operations. See [Device API – Memory and LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_memory.html#lsa) for device-side pointer accessors and reduce/copy operations.

对于 device API，对称注册的 window（例如使用 `NCCL_WIN_COLL_SYMMETRIC`）提供 LSA（可通过 load/store 访问）内存：设备端代码可以用 load/store 操作访问对端缓冲区。有关设备端指针访问器及 reduce/copy 操作，参见 [Device API – Memory and LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_memory.html#lsa)。

> The following example shows how to register buffers into NCCL window and use it for communication:

下面的示例展示如何将缓冲区注册到 NCCL window 并用于通信：

```c
void* src;
void* dst;
ncclWindow_t src_win;
ncclWindow_t dst_win;

CHECK(ncclMemAlloc(&src, src_size));
CHECK(ncclMemAlloc(&dst, dst_size));
// Passing NCCL_WIN_COLL_SYMMETRIC requires users to provide the symmetric buffers among all ranks in collectives.
// Every rank needs to call ncclCommWindowRegister to register its buffers.
CHECK(ncclCommWindowRegister(comm, src, src_size, &src_win, NCCL_WIN_COLL_SYMMETRIC));
CHECK(ncclCommWindowRegister(comm, dst, dst_size, &dst_win, NCCL_WIN_COLL_SYMMETRIC));
// Use the registered buffers for communication to enable symmetric communication benefits.
// In this example, every rank has 0x1000 offset and 0x2000 offset from the head address of
// src and dst respectively, which satisfies the symmetric buffer requirement.
CHECK(ncclAllGather((uint8_t*)src + 0x1000, (uint8_t*)dst + 0x2000, 1, ncclInt8, comm, stream));
CHECK(cudaStreamSynchronize(stream));

CHECK(ncclCommWindowDeregister(comm, src_win));
CHECK(ncclCommWindowDeregister(comm, dst_win));

CHECK(ncclMemFree(src));
CHECK(ncclMemFree(dst));
```

> See the description of [`ncclCommWindowRegister()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#c.ncclCommWindowRegister) and [`ncclCommWindowDeregister()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#c.ncclCommWindowDeregister) for additional details.

更多细节参见 [`ncclCommWindowRegister()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#c.ncclCommWindowRegister) 和 [`ncclCommWindowDeregister()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#c.ncclCommWindowDeregister) 的说明。

#### Zero-CTA Optimization｜Zero-CTA 优化

> NCCL supports zero-CTA optimization to avoid the use of CTA for communication and to overlap communication and computation.

NCCL 支持 Zero-CTA 优化，使通信可以不占用 CTA，从而有利于通信与计算重叠执行。

> Zero-CTA over NVLink with Copy Engine (CE) is supported since NCCL 2.28; zero-CTA across the network (CPU proxy inter-node and CE intra-node) is supported since NCCL 2.30.6. The following are the requirements to enable zero-CTA optimization:

从 NCCL 2.28 起，支持通过 Copy Engine（CE）在 NVLink 上执行 Zero-CTA 通信；从 NCCL 2.30.6 起，支持跨网络的 Zero-CTA 通信（节点间使用 CPU proxy，节点内使用 CE）。启用 Zero-CTA 优化需要满足以下要求：

> - CUDA driver version >= 12.5
> - The buffer is symmetrically registered with the NCCL window
> - The communicator is configured with the `NCCL_CTA_POLICY_ZERO` flag (please see [NCCL Communicator CTA Policy Flags](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/flags.html#nccl-communicator-cta-policy-flags))
> - Supported collectives:
>   - Within a single NVL or MNNVL domain: AlltoAll, AllGather, Scatter, and Gather
>   - Across the network: AlltoAll and AllGather

- CUDA driver 版本不低于 12.5。
- 缓冲区以对称方式注册到 NCCL window。
- communicator 配置了 `NCCL_CTA_POLICY_ZERO` 标志（参见 [NCCL Communicator CTA Policy Flags](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/flags.html#nccl-communicator-cta-policy-flags)）。
- 支持的集合操作依通信范围而异：
  - 单个 NVL 或 MNNVL 域内：AlltoAll、AllGather、Scatter、Gather。
  - 跨网络：AlltoAll、AllGather。

> The following example shows how to enable zero-CTA optimization:

下面的示例展示如何启用 Zero-CTA 优化：

```c
ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
// NCCL_CTA_POLICY_ZERO to enable zero-CTA optimization whenever possible
config.CTAPolicy = NCCL_CTA_POLICY_ZERO;
CHECK(ncclCommInitRankConfig(&comm, nranks, id, rank, &config));

void* src;
void* dst;
ncclWindow_t src_win;
ncclWindow_t dst_win;

CHECK(ncclMemAlloc(&src, src_size));
CHECK(ncclMemAlloc(&dst, dst_size));

// Register the buffers into NCCL symmetric window
CHECK(ncclCommWindowRegister(comm, src, src_size, &src_win, NCCL_WIN_COLL_SYMMETRIC));
CHECK(ncclCommWindowRegister(comm, dst, dst_size, &dst_win, NCCL_WIN_COLL_SYMMETRIC));

CHECK(ncclAllGather(src, dst, 1, ncclInt8, comm, stream));
CHECK(cudaStreamSynchronize(stream));

CHECK(ncclCommWindowDeregister(comm, src_win));
CHECK(ncclCommWindowDeregister(comm, dst_win));

CHECK(ncclMemFree(src));
CHECK(ncclMemFree(dst));
```

### 重点解读

**注册是各 rank 共同遵守的条件。** 同一个 NCCL 通信调用中，不能让部分 rank 传入已注册缓冲区、其余 rank 传入未注册缓冲区；发送端与接收端缓冲区也都要注册。注册方式可以选 CUDA Graph 或 Local，但是否满足条件需按具体算法和通信路径判断。

**注册生命周期取决于方式。** CUDA Graph 注册随 graph 销毁而解除；Local 注册由 `ncclCommRegister` 建立，复用缓冲区后由 `ncclCommDeregister` 显式解除。NVLS 示例还要求各 rank 对同一项集合操作采用相同的缓冲区内偏移；这不要求发送与接收缓冲区的偏移彼此相等，例如原文第二次 AllReduce 的发送偏移为 1024 个 `float`，接收偏移为 2048 个 `float`。

**注册成功不保证通信走 zero-copy 路径。** 跨节点通信可能因为分配方式而使用内部暂存缓冲区；PXN 启用时，网络缓冲区注册不会生效，即使已调用 `ncclCommRegister`。选择 allocator、PXN 与注册方式时，需要一起核对实际通信路径。

**Window registration 的约束更具体。** 它要求 VMM 类分配器或 `ncclMemAlloc` 提供输入缓冲区；使用 `NCCL_WIN_COLL_SYMMETRIC` 时，各 rank 还须满足对称缓冲区和偏移要求。Zero-CTA 在此基础上还受 driver 版本、communicator 配置及支持的集合操作范围限制。

**大块缓冲区示例中的释放对象。** 原文第三个示例的 `sendbuff = buffer`，因此末尾的 `ncclMemFree(sendbuff)` 实际释放整块 `buffer`；`recvbuff` 只是这块内存中的内部地址，不能单独释放。解除注册和等待 stream 完成后，才应释放底层分配。

## Device-Initiated Communication

来源：[NVIDIA NCCL User Guide — Device-Initiated Communication](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#device-initiated-communication)。

### 中英文对照

> Starting with version 2.28, NCCL provides a device-side communication API, making it possible to use communication primitives directly from user CUDA kernels.

从 NCCL 2.28 起，NCCL 提供设备端通信 API，用户可以直接在自己的 CUDA kernel 中调用通信原语。

#### Device API｜设备端 API

> Device API consists of the following modules:

Device API 由以下模块组成：

> - LSA (Load/Store Accessible) – for communication between devices accessible via memory load/store operations, using CUDA P2P. This includes devices connected over NVLink and some devices connected over PCIe, so long as they have P2P connectivity with each other (as indicated by `nvidia-smi topo -p2p p`). Up to NCCL 2.28.3, the availability of LSA was also subject to the `NCCL_P2P_LEVEL` distance check, but that is no longer the case with newer versions. See LSA.
> - Multimem – for communication between devices using the hardware multicast feature provided by NVLink SHARP (available on some datacenter GPUs since the Hopper generation).
> - GIN (GPU-Initiated Networking) – for communication over the network (since NCCL 2.28.7).
> - CFT (Compute Fabric Transport) – for communication through CUDA fabric logical endpoints (since NCCL 2.31). See Compute Fabric Transport.
> - Reduce, Broadcast, and Fused Building Blocks — Building Blocks for Computation-Fused Kernels: reduce, copy (broadcast), and reduce-then-copy (see Device API – Remote Reduce and Copy: Building Blocks for Custom Communication Kernels in the API reference).

- **LSA（Load/Store Accessible）**：使用 CUDA P2P，通过内存 load/store 操作在可互访的设备之间通信。只要具备 P2P 连通性（可用 `nvidia-smi topo -p2p p` 查看），就包括通过 NVLink 相连的设备及部分通过 PCIe 相连的设备。截至 NCCL 2.28.3，LSA 可用性还受 `NCCL_P2P_LEVEL` 距离检查约束；较新版本不再如此。参见 LSA。
- **Multimem**：利用 NVLink SHARP 提供的硬件 multicast 功能，在设备之间通信；该功能自 Hopper 一代起在部分数据中心 GPU 上可用。
- **GIN（GPU-Initiated Networking）**：用于跨网络通信，从 NCCL 2.28.7 起提供。
- **CFT（Compute Fabric Transport）**：通过 CUDA fabric 逻辑端点通信，从 NCCL 2.31 起提供。参见 Compute Fabric Transport。
- **Reduce、Broadcast 与融合构件**：用于融合计算与通信的 kernel，提供 reduce、copy（broadcast）及先 reduce 后 copy 的构件；参见 API 参考文档中的 Device API – Remote Reduce and Copy: Building Blocks for Custom Communication Kernels。

#### Requirements｜环境要求

> The device API relies on symmetric memory (see Window Registration), which in turn depends on GPU virtual memory management (see `NCCL_CUMEM_ENABLE`) and optionally – for multimem support – on NVLink SHARP (see `NCCL_NVLS_ENABLE`). GIN supports muiltiple networking backends, each with their own set of requirements.

Device API 依赖对称内存（参见 Window Registration）；对称内存又依赖 GPU 虚拟内存管理（参见 `NCCL_CUMEM_ENABLE`），而 Multimem 支持还可能依赖 NVLink SHARP（参见 `NCCL_NVLS_ENABLE`）。GIN 支持多种网络后端，各有自己的环境要求。

> GIN has the following requirements:

GIN 的通用要求如下：

> - CUDA 12.2 or later when compiling the GPU code
> - NVIDIA GPUs: Volta or newer. NVIDIA GPU drivers >= 510.40.3
> - Using GIN for buffers that are backed by multiple cuMem segments requires DMA-BUF

- 编译 GPU 代码时使用 CUDA 12.2 或更新版本。
- 使用 Volta 或更新架构的 NVIDIA GPU，且 NVIDIA GPU driver 版本不低于 510.40.3。
- 如果 GIN 使用由多个 cuMem segment 支撑的缓冲区，则需要 DMA-BUF。

> The GIN CPU Proxy backend has the following requirements:

GIN CPU Proxy 后端的要求如下：

> - NVIDIA NICs: CX4 or newer
> - GPUDirect RDMA with the internal plugin:
>   - If using DMA-BUF, rdma-core >= 34.0.
>   - If using nvidia-peermem, Mellanox OFED >= 5.0 or DOCA.

- 使用 CX4 或更新的 NVIDIA NIC。
- 使用内置插件实现 GPUDirect RDMA 时：
  - 如使用 DMA-BUF，要求 rdma-core 不低于 34.0。
  - 如使用 nvidia-peermem，要求 Mellanox OFED 不低于 5.0，或使用 DOCA。

> The GIN GDAKI backend has the following requirements:

GIN GDAKI 后端的要求如下：

> - NVIDIA NICs: CX4 or newer. rdma-core >= 44.0
> - GPU Direct RDMA:
>   - If using DMA-BUF, linux kernel >= 6.1.
>   - If using nvidia-peermem on baremetal, linux kernel >= 5.12 and either Mellanox OFED or DOCA driver.
>   - If using nvidia-peermem on virtualized environments, linux kernel >= 6.1 and either Mellanox OFED or DOCA driver.
>   - Mellanox OFED and DOCA driver packages may replace some parts of the upstream drivers. If you plan to install any of those packages, you need Mellanox OFED >= 5.8 or DOCA >= 3.1.0.

- 使用 CX4 或更新的 NVIDIA NIC，且 rdma-core 不低于 44.0。
- GPU Direct RDMA 的要求按所用机制区分：
  - 如使用 DMA-BUF，Linux kernel 不低于 6.1。
  - 如在裸机上使用 nvidia-peermem，Linux kernel 不低于 5.12，且使用 Mellanox OFED 或 DOCA driver。
  - 如在虚拟化环境中使用 nvidia-peermem，Linux kernel 不低于 6.1，且使用 Mellanox OFED 或 DOCA driver。
  - Mellanox OFED 和 DOCA driver 包可能替换部分上游驱动；若要安装这些包，要求 Mellanox OFED 不低于 5.8 或 DOCA 不低于 3.1.0。

> When using host-backed buffers, the following additional limitations apply:

使用 host-backed 缓冲区时，还有以下限制：

> - Host segments must be allocated with `CU_MEM_LOCATION_TYPE_HOST_NUMA`.
> - DirectNIC is not supported.
> - LSA Multimem is not supported.
> - Host RMA APIs are not supported.

- Host segment 必须以 `CU_MEM_LOCATION_TYPE_HOST_NUMA` 分配。
- 不支持 DirectNIC。
- 不支持 LSA Multimem。
- 不支持 Host RMA API。

> Using the host RMA API requires CUDA 12.5 or greater.

使用 Host RMA API 要求 CUDA 12.5 或更新版本。

> Building with `EMIT_LLVM_IR=1` (to generate readable LLVM intermediate representation code) requires CUDA 12.

使用 `EMIT_LLVM_IR=1` 构建以生成可读的 LLVM 中间表示代码时，需要 CUDA 12。

> CFT has the following requirements:

CFT 的要求如下：

> - CUDA 13.3 or later when compiling the GPU code.
> - NVIDIA GPUs: Blackwell or newer. NVIDIA GPU drivers >= 610.43.02

- 编译 GPU 代码时使用 CUDA 13.3 或更新版本。
- 使用 Blackwell 或更新架构的 NVIDIA GPU，且 NVIDIA GPU driver 版本不低于 610.43.02。

> See Compute Fabric Transport and Device API - CFT for CFT setup and API details.

CFT 的配置和 API 细节参见 Compute Fabric Transport 与 Device API - CFT。

#### Cross-Version Compatibility｜跨版本兼容性

> NCCL assumes the compile-time version of the device code is the same as the compile-time version of the corresponding host code (i.e., the call to `ncclDevCommCreate()`). Starting with NCCL 2.29, the host-side structures are versioned, to enable cross-version compatibility checks. In general, the compile-time version cannot be newer than the runtime version (e.g., the version of `libnccl.so`). As of NCCL 2.29, backwards compatibility is supported for kernels utilizing LSA and multimem, i.e., a kernel compiled with NCCL 2.29.2/2.29.3 should continue to work when running with NCCL 2.29.7. Kernels utilizing GIN are currently not backwards compatible and need to be recompiled when NCCL is upgraded.

NCCL 假定设备端代码与对应的 host 端代码（即调用 `ncclDevCommCreate()` 的代码）使用同一 NCCL 版本编译。从 NCCL 2.29 起，host 端结构体带有版本信息，可用于跨版本兼容性检查。一般而言，编译时版本不能新于运行时版本（例如 `libnccl.so` 的版本）。截至 NCCL 2.29，使用 LSA 和 Multimem 的 kernel 支持向后兼容，例如以 NCCL 2.29.2/2.29.3 编译的 kernel 应可在 NCCL 2.29.7 上继续运行。使用 GIN 的 kernel 目前不具备这种向后兼容性，升级 NCCL 时需要重新编译。

#### Host-Side Setup｜Host 端准备

> To perform communication from the device kernel, a device communicator needs to be created first, using `ncclDevCommCreate()`. Data transfer operations on buffers require symmetric memory windows (see Window Registration). A custom communication kernel can then be launched using the standard CUDA syntax. The code excerpt below demonstrates these steps:

要在设备端 kernel 中通信，首先需要使用 `ncclDevCommCreate()` 创建 device communicator。缓冲区数据传输还需要对称内存 window（参见 Window Registration）。完成这些准备后，就可以用标准 CUDA 语法启动自定义通信 kernel。示例展示了这些步骤：

```cpp
int main() {
  [...]
  NCCLCHECK(ncclCommInitRank(&comm, nranks, id, rank));

  /* Buffer initialization and window creation */
  char* buffer;
  size_t size = 256*1048576;
  NCCLCHECK(ncclMemAlloc((void**)&buffer, size));
  ncclWindow_t win;
  NCCLCHECK(ncclCommWindowRegister(comm, buffer, size, &win, NCCL_WIN_COLL_SYMMETRIC));

  /* Get device communicator */
  ncclDevComm devComm;
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  int nCTAs = 16;
  reqs.lsaBarrierCount = nCTAs;
  NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));

  /* Launch user kernel */
  customKernel<<<nCTAs, 512>>>(devComm, win);
  [...]
}
```

> Depending on the kernel and application requirements, the same window can be used for input and output, or multiple windows may be needed. When creating a device communicator, the resources that the kernel will need should be specified via the requirements list (see `ncclDevCommRequirements`). In the above example we specify just the number of barriers that our LSA kernel will need, in this case one for each CTA the kernel is to be launched on (16, each CTA running 512 threads).

根据 kernel 和应用需求，可以让输入与输出共用一个 window，也可能需要多个 window。创建设备端 communicator 时，应通过 `ncclDevCommRequirements` 指定 kernel 所需资源。上例只指定了 LSA kernel 所需的 barrier 数量：计划启动 16 个 CTA，每个 CTA 一个 barrier，每个 CTA 包含 512 个线程。

#### Simple LSA Kernel｜简单的 LSA Kernel

```cpp
template <typename T>
__global__ void inPlaceAllReduceKernel(ncclDevComm devComm, ncclWindow_t win, size_t offset, size_t count) {
  ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamTagLsa(), blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire);

  const int rank = devComm.lsaRank, nRanks = devComm.lsaSize;
  const int globalTid = threadIdx.x + blockDim.x * (rank + blockIdx.x * nRanks);
  const int globalNthreads = blockDim.x * gridDim.x * nRanks;

  for (size_t o = globalTid; o < count; o += globalNthreads) {
    T v = 0;
    for (int peer = 0; peer < nRanks; peer++) {
      T* inputPtr = (T*)ncclGetLsaPointer(win, offset, peer);
      v += inputPtr[o];
    }
    for (int peer = 0; peer < nRanks; peer++) {
      T* outputPtr = (T*)ncclGetLsaPointer(win, offset, peer);
      outputPtr[o] = v;
    }
  }

  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}
```

> The above code excerpt shows a simple device kernel – an in-place variant (the input buffer is reused for the output) of AllReduce, utilizing LSA support (data is transferred via memory load/store instructions).

上述代码展示了一个简单的设备端 kernel：它利用 LSA 以内存 load/store 指令传输数据，实现原地 AllReduce，即把输入缓冲区也用作输出缓冲区。

> The start of the buffer is specified as a (byte-based) offset within the previously registered window `win` (see Window Registration); the buffer consists of `count` elements of type `T`.

缓冲区起点由先前注册的 window `win` 内的字节偏移指定（参见 Window Registration）；缓冲区包含 `count` 个 `T` 类型元素。

> Before the kernel can start processing data, it needs to ensure that all participants are ready. It creates a memory barrier session `bar` (see `ncclLsaBarrierSession`) and uses it to synchronize across all the threads of the CTA (`ncclCoopCta()`; see Thread Groups) and the ranks of the communicator (`devComm`). `ncclTeamTagLsa` indicates the subset of ranks the barrier will apply to (see Teams) – this kernel assumes that all ranks are LSA-connected. `blockIdx.x` is the CTA’s local index, used to select the barrier.

kernel 在处理数据前须确保所有参与者已就绪。它创建 memory barrier session `bar`（参见 `ncclLsaBarrierSession`），用来同步 CTA 中的所有线程（`ncclCoopCta()`，参见 Thread Groups）和 communicator 中的 rank（`devComm`）。`ncclTeamTagLsa` 指定 barrier 作用的 rank 子集（参见 Teams）；这个 kernel 假设所有 rank 均可通过 LSA 互连。`blockIdx.x` 是 CTA 的本地索引，用于选择对应的 barrier。

> The kernel then calculates a globally unique index for each thread as well as the overall thread count, and can finally start processing data, using an all-to-all communication pattern. In each iteration of the outer loop, every participating thread loads a single input element from each communicator rank (the first inner loop). `ncclGetLsaPointer()` is used to calculate the locally-accessible address of the start of the buffer within each rank (remote device memory was previously mapped into the local address space – see Window Registration). Extracted input data is accumulated and the result is stored back at each rank (the second inner loop). Before the kernel terminates, another memory synchronization needs to take place to ensure that all participants have finished processing their data.

随后，kernel 计算每个线程在所有 rank 中唯一的索引以及总线程数，并以 all-to-all 通信模式处理数据。外层循环每次迭代中，每个参与线程先从 communicator 的每个 rank 读取一个输入元素（第一个内层循环）。`ncclGetLsaPointer()` 根据此前映射到本地地址空间的远端设备内存，计算各 rank 缓冲区起点在本地可访问的地址（参见 Window Registration）。线程将读取值累加，再把结果写回各 rank（第二个内层循环）。kernel 结束前还需再次同步内存，确保所有参与者都已处理完数据。

> Note that this simple implementation would likely fall short of achieving the peak bandwidth, as it utilizes neither vectorization nor loop unrolling. For optimized LSA reduce, copy, and fused reduce-then-copy building blocks (e.g. for AllReduce, AllGather, ReduceScatter), see Device API – Remote Reduce and Copy: Building Blocks for Custom Communication Kernels in the Device API reference.

这个简单实现既没有向量化，也没有循环展开，因此可能达不到峰值带宽。对于优化过的 LSA reduce、copy 及先 reduce 后 copy 的融合构件（例如用于 AllReduce、AllGather、ReduceScatter），参见 Device API 参考文档中的 Device API – Remote Reduce and Copy: Building Blocks for Custom Communication Kernels。

#### Multimem Device Kernel｜Multimem 设备端 Kernel

```cpp
int main() {
  [...]
  reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  int nCTAs = 16;
  reqs.lsaBarrierCount = nCTAs;
  reqs.lsaMultimem = true;
  NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));
  [...]
}

template <typename T>
__global__ void inPlaceAllReduceKernel(ncclDevComm devComm, ncclWindow_t win, size_t offset, size_t count) {
  ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamTagLsa(), blockIdx.x, /*multimem*/true };
  [...]
  T* mmPtr = (T*)ncclGetLsaMultimemPointer(win, offset, devComm);
  for (size_t o = globalTid; o < count; o += globalNthreads) {
    T v = multimem_sum(mmPtr+o);
    multimem_st(mmPtr+o, v);
  }
  [...]
}
```

> The above code excerpt demonstrates modifications needed to the earlier code segments to enable multimem support (the lines with critical changes are highlighted). On the host side, `lsaMultimem` needs to be set in the requirements prior to creating the device communicator (`ncclDevCommCreate()` will fail if the necessary hardware support is unavailable).

这段代码展示了在前面示例基础上启用 Multimem 所需的改动（原页面以高亮标出关键行）。Host 端必须先在 requirements 中设置 `lsaMultimem`，再创建设备端 communicator；如果硬件不支持，`ncclDevCommCreate()` 会失败。

> Within the device kernel, we can switch the memory barrier to a multimem-optimized variant by adding an extra argument to the constructor. The processing loop is actually simpler with multimem: `ncclGetLsaMultimemPointer()` needs to be invoked just once per kernel. The returned multicast memory pointer enables access to the device memory of all the ranks of the communicator without having to iterate over them, and the data can be reduced in hardware. To keep this example simple, the implementations of `multimem_sum` and `multimem_st` are not included; they need to be implemented using PTX, e.g., `multimem.ld_reduce.global.add` and `multimem.st.global`.

在设备端 kernel 中，给 barrier 的构造函数增加一个参数，就能切换到针对 Multimem 优化的变体。Multimem 使处理循环更简单：每个 kernel 只需调用一次 `ncclGetLsaMultimemPointer()`。返回的 multicast 内存指针可以访问 communicator 内所有 rank 的设备内存，无须逐个遍历 rank，并可在硬件中完成归约。为了保持示例简洁，原文没有给出 `multimem_sum` 和 `multimem_st` 的实现；它们需要用 PTX 实现，例如 `multimem.ld_reduce.global.add` 和 `multimem.st.global`。

#### Thread Groups｜线程组

> Many functions in the device API take a thread cooperative group as input to indicate which threads within the CTA will take part in the operation. NCCL provides three predefined ones: `ncclCoopThread()`, `ncclCoopWarp()`, and (the most commonly used) `ncclCoopCta()`.

Device API 中许多函数将协作线程组作为输入，用于指定 CTA 中哪些线程参与操作。NCCL 预定义了三种：`ncclCoopThread()`、`ncclCoopWarp()` 和最常用的 `ncclCoopCta()`。

> Users may also pass CUDA cooperative groups, or any class which provides `thread_rank()`, `size()`, and `sync()` methods.

用户也可以传入 CUDA cooperative group，或任何提供 `thread_rank()`、`size()` 与 `sync()` 方法的类。

#### Teams｜Rank 子集

> To address remote ranks or perform barriers, NCCL refers to subsets of ranks within a communicator as “teams”. NCCL provides five predefined ones:

为寻址远端 rank 或执行 barrier，NCCL 将 communicator 中的 rank 子集称为“team”。NCCL 预定义了五种 team：

> - `ncclTeamWorld()` – the “world” team, encompassing all the ranks of a given communicator.
> - `ncclTeamLsa()` – all the peers accessible from the local rank using load/store operations.
> - `ncclTeamRail()` – the set of peers that have the same rank number within their LSA team (a rail team is orthogonal to an LSA team).
> - `ncclTeamCft()` – the CFT team for CUDA fabric logical endpoint unicast operations.
> - `ncclTeamCftMultimem()` – the CFT team for CUDA fabric logical endpoint multicast operations.

- `ncclTeamWorld()`：world team，包含指定 communicator 的所有 rank。
- `ncclTeamLsa()`：包含本地 rank 可以通过 load/store 访问的所有 peer。
- `ncclTeamRail()`：包含在各自 LSA team 内具有相同 rank 编号的 peer；rail team 与 LSA team 正交。
- `ncclTeamCft()`：用于 CUDA fabric 逻辑端点 unicast 操作的 CFT team。
- `ncclTeamCftMultimem()`：用于 CUDA fabric 逻辑端点 multicast 操作的 CFT team。

> The `ncclTeam` structure contains fairly self-explanatory elements `nRanks`, `rank`, and `stride`. The device API contains functions to verify team membership, convert rank numbers between teams, etc. The world and LSA teams are always contiguous (stride `1`), whereas the rail team is typically not – its stride equals the size of the LSA team (the assumption is thus that each rank n within the local LSA team has direct network connectivity with corresponding ranks n of all remote LSA teams).

`ncclTeam` 结构包含 `nRanks`、`rank` 和 `stride`。Device API 还提供检查 team 成员身份、在不同 team 间转换 rank 编号等函数。World team 与 LSA team 始终连续（`stride` 为 `1`）；rail team 通常不连续，其 `stride` 等于 LSA team 的大小。这隐含的假设是：本地 LSA team 中编号为 n 的 rank 与所有远端 LSA team 中相应编号为 n 的 rank 之间具有直接网络连通性。

#### Segment Types｜内存 Segment 类型

> The `SegmentType` template parameter of `ncclGin::put()` and `ncclGin::get()` describes the physical memory composition of the source and destination virtual addresses. Three tag types are defined:

`ncclGin::put()` 和 `ncclGin::get()` 的模板参数 `SegmentType` 描述源、目标虚拟地址对应的物理内存组成。定义了三种 tag 类型：

> - `ncclGin_SegmentDevice` (default) — the virtual addresses only contain cuMem segments of type `CU_MEM_LOCATION_TYPE_DEVICE`.
> - `ncclGin_SegmentHostNuma` — the virtual addresses only contain cuMem segments of type `CU_MEM_LOCATION_TYPE_HOST_NUMA`.
> - `ncclGin_SegmentMixed` — the virtual addresses contain a mix of `CU_MEM_LOCATION_TYPE_DEVICE` and `CU_MEM_LOCATION_TYPE_HOST_NUMA` segments.

- `ncclGin_SegmentDevice`（默认）：虚拟地址只包含 `CU_MEM_LOCATION_TYPE_DEVICE` 类型的 cuMem segment。
- `ncclGin_SegmentHostNuma`：虚拟地址只包含 `CU_MEM_LOCATION_TYPE_HOST_NUMA` 类型的 cuMem segment。
- `ncclGin_SegmentMixed`：虚拟地址同时包含 `CU_MEM_LOCATION_TYPE_DEVICE` 和 `CU_MEM_LOCATION_TYPE_HOST_NUMA` segment。

#### Host-Accessible Device Pointer Functions｜Host 可访问的设备指针函数

> Starting with version 2.29, NCCL provides host-accessible functions that enable host code to obtain pointers to LSA memory regions.

从 NCCL 2.29 起，NCCL 提供可从 host 端调用的函数，使 host 代码能够取得指向 LSA 内存区域的指针。

> The four functions are `ncclGetLsaMultimemDevicePointer()` (multimem base pointer), `ncclGetMultimemDevicePointer()` (multimem base pointer with custom handle), `ncclGetLsaDevicePointer()` (LSA peer pointer), and `ncclGetPeerDevicePointer()` (world rank peer pointer). Functions automatically discover the associated communicator from the window object and return `ncclResult_t` error codes.

这四个函数分别是：`ncclGetLsaMultimemDevicePointer()`（Multimem 基址指针）、`ncclGetMultimemDevicePointer()`（带自定义 handle 的 Multimem 基址指针）、`ncclGetLsaDevicePointer()`（LSA peer 指针）和 `ncclGetPeerDevicePointer()`（world rank peer 指针）。它们会从 window 对象自动找到关联的 communicator，并返回 `ncclResult_t` 错误码。

> Usage Example:

使用示例：

```cpp
int main() {
  [...]
  // Allocate symmetric memory buffer
  char* buffer;
  size_t size = 256 * 1024 * 1024;  // 256 MB buffer
  NCCLCHECK(ncclMemAlloc((void**)&buffer, size));

  // Create window with the allocated buffer
  ncclWindow_t win;
  NCCLCHECK(ncclCommWindowRegister(comm, buffer, size, &win, NCCL_WIN_COLL_SYMMETRIC));

  // Get host-accessible pointers
  void* multimemPtr;
  void* lsaPtr;
  void* peerPtr;

  // Get multimem pointer (returns nullptr if multimem not supported)
  NCCLCHECK(ncclGetLsaMultimemDevicePointer(win, 0, &multimemPtr));
  if (multimemPtr == nullptr) {
      // Multimem not available, use fallback
  }

  // Get LSA pointer for peer 1
  NCCLCHECK(ncclGetLsaDevicePointer(win, 0, 1, &lsaPtr));

  // Get peer pointer for world rank 2
  NCCLCHECK(ncclGetPeerDevicePointer(win, 0, 2, &peerPtr));

  // Use pointers in custom kernels or legacy code
  customKernel<<<nCTAs, 256>>>(multimemPtr, lsaPtr, peerPtr);

  // Cleanup
  NCCLCHECK(ncclCommWindowDeregister(comm, &win));
  // Device pointers are invalidated after window deregistration
  NCCLCHECK(ncclMemFree(buffer));
  [...]
}
```

> Important notes: Pointer lifetime is limited to the shorter of Window and Communicator lifetime. Functions should be called once and pointers cached for reuse. For detailed function documentation, see Host-Accessible Device Pointer Functions.

重要说明：这些指针的有效期以 window 和 communicator 中较短的生命周期为准。建议只调用一次取指针函数，然后缓存指针以便复用。函数细节参见 Host-Accessible Device Pointer Functions。

#### GIN Device Kernel｜GIN 设备端 Kernel

> The following illustrates pure GIN AlltoAll: all peer data moves over the network. The host creates a `ncclDevComm` with GIN-specific resources, registers symmetric memory windows (see Window Registration), and launches a kernel that performs the collective using GIN.

下例展示纯 GIN AlltoAll：所有 peer 的数据均经网络传输。Host 端先创建具有 GIN 专用资源的 `ncclDevComm`，注册对称内存 window（参见 Window Registration），再启动使用 GIN 完成集合操作的 kernel。

```cpp
// Grid width (CTAs). Must match reqs.worldGinBarrierCount and reqs.ginSignalCount.
#define NCCL_DEVICE_CTA_COUNT 16
#define NCCL_DEVICE_THREADS_PER_CTA 512

int main() {
  [...]
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.worldGinBarrierCount = NCCL_DEVICE_CTA_COUNT;
  reqs.ginSignalCount = NCCL_DEVICE_CTA_COUNT;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));
  [...]
}

template <typename T>
__global__ void PureGinAlltoAllKernel(ncclWindow_t sendwin, size_t sendoffset,
                                      ncclWindow_t recvwin, size_t recvoffset,
                                      size_t count, struct ncclDevComm devComm) {
  int ginContext = 0; // single context for simplicity
  unsigned int signalIndex = blockIdx.x;
  ncclGin gin { devComm, ginContext };
  uint64_t signalValue = gin.readSignal(signalIndex);

  ncclGinBarrierSession<ncclCoopCta> bar { ncclCoopCta(), gin, ncclTeamTagWorld(), blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire, ncclGinFenceLevel::None);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  const size_t size = count * sizeof(T);
  for (int r = tid; r < devComm.nRanks; r += nthreads) {
    gin.put(ncclTeamWorld(devComm), r,
        recvwin, recvoffset + devComm.rank * size,
        sendwin, sendoffset + r * size,
        size, ncclGin_WeakSignalInc{signalIndex});
  }

  // Wait only on the CTA whose blockIdx.x (signalIndex) accumulates all puts to this rank.
  int receivingCta = (devComm.rank % nthreads) / blockDim.x;
  if (blockIdx.x == receivingCta)
    gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + devComm.nRanks);

  gin.flush(ncclCoopCta());
  bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::None);
}
```

> The above code excerpt shows the GIN-related host setup for NCCL 2.30 and later (highlighted lines) together with the `PureGinAlltoAllKernel` kernel definition. GPU-initiated networking is available since NCCL 2.28.7. Version-specific host and kernel changes for older NCCL builds are summarized under Compatibility adjustments at the end of this section.

上述代码展示适用于 NCCL 2.30 及以后版本的 GIN host 端准备步骤（原页面高亮关键行），以及 `PureGinAlltoAllKernel` 的定义。GPU 发起的网络通信从 NCCL 2.28.7 起可用。旧版 NCCL 的 host 端和 kernel 端改动见本节末尾的 Compatibility adjustments。

> In `ncclDevCommRequirements`, `worldGinBarrierCount` reserves slots for `ncclGinBarrierSession` (network-side barriers) and `ginSignalCount` reserves per-CTA signals for completion. Both are set to the number of CTAs in the launch grid (here `NCCL_DEVICE_CTA_COUNT`), matching `gridDim.x`, so each thread block uses `blockIdx.x` as its barrier index and signal index. GIN relies on these barriers and signals for cross-rank synchronization and for tracking asynchronous work. Set `ginConnectionType` to `NCCL_GIN_CONNECTION_FULL` to connect each rank to all peers (see `ncclGinConnectionType_t`). `ncclDevCommCreate()` fails if GIN cannot be provided.

在 `ncclDevCommRequirements` 中，`worldGinBarrierCount` 为 `ncclGinBarrierSession` 预留网络端 barrier 槽位，`ginSignalCount` 为每个 CTA 预留跟踪完成状态的 signal。两者都设为启动 grid 的 CTA 数（此处为 `NCCL_DEVICE_CTA_COUNT`），并与 `gridDim.x` 一致，因此每个线程块都可用 `blockIdx.x` 作为 barrier 和 signal 的索引。GIN 依赖这些 barrier 和 signal 实现跨 rank 同步及异步工作的状态跟踪。将 `ginConnectionType` 设为 `NCCL_GIN_CONNECTION_FULL`，使每个 rank 与所有 peer 建立连接（参见 `ncclGinConnectionType_t`）。如果无法提供 GIN，`ncclDevCommCreate()` 会失败。

> On the device, GIN barriers synchronize across ranks over the network. Each thread block uses `blockIdx.x` to select its barrier so blocks can coordinate with corresponding blocks on other nodes. A single GIN context is used here. Construct `ncclGin` with context index `0`. Each thread block reads its own per-CTA signal slot (`signalIndex == blockIdx.x`) before `bar.sync()` at kernel entry. The `ncclGinBarrierSession` uses `ncclTeamTagWorld()` and `blockIdx.x`. The barrier ensures all ranks are ready before the AlltoAll exchange (`bar.sync()` at kernel entry).

在设备端，GIN barrier 通过网络同步不同 rank。每个线程块用 `blockIdx.x` 选择自己的 barrier，以便与其他节点上对应的线程块协调。示例只使用一个 GIN context，因此构造 `ncclGin` 时 context 索引取 `0`。kernel 入口处，每个线程块先读取对应 CTA 的 signal 槽位（`signalIndex == blockIdx.x`），再调用 `bar.sync()`。`ncclGinBarrierSession` 使用 `ncclTeamTagWorld()` 和 `blockIdx.x`；入口 barrier 确保所有 rank 在开始 AlltoAll 交换前都已就绪。

> Unlike AllReduce-style kernels, for AlltoAll the per-thread index only needs to be unique within this rank. That index then selects the destination peer. The main data transfer is performed using the one-sided `put()`, launched in parallel on all participating threads with one `put()` per destination peer. The loop is needed whenever the communicator size exceeds the number of threads that take part in the loop (here, `threadIdx.x + blockIdx.x * blockDim.x` stepping by `blockDim.x * gridDim.x`). `put()` takes the usual arguments: destination rank, destination and source windows and offsets, transfer size, and optional actions. This example passes `ncclGin_WeakSignalInc{signalIndex}` as remoteAction so the destination rank receives one completion increment once the payload is settled. The receiver waits for the count of completed incoming puts for that signal slot.

与 AllReduce 类 kernel 不同，AlltoAll 的线程索引只需在当前 rank 内唯一，再由该索引选出目标 peer。主要数据传输由单边 `put()` 完成：参与线程并行发起调用，每个目标 peer 对应一次 `put()`。当 communicator 的 rank 数超过参与循环的线程数时，需要循环分配任务；此处索引从 `threadIdx.x + blockIdx.x * blockDim.x` 开始，以 `blockDim.x * gridDim.x` 为步长。`put()` 的参数包括目标 rank、目标与源 window 及偏移、传输大小和可选动作。示例将 `ncclGin_WeakSignalInc{signalIndex}` 作为 remoteAction，使数据写入目标端后，对应的完成 signal 增加一次；接收端等待该 signal 上已完成的入站 put 数量。

> Each CTA uses `signalIndex = blockIdx.x` on its outgoing `ncclGin::put()` operations. On the destination rank, each peer’s `ncclGin::put()` contributes one increment to the signal slot indexed by that sender CTA’s `blockIdx.x`. All CTAs participate in issuing `ncclGin::put()`, but only the receiving CTA, a single thread block on this rank, must observe completion for that rank’s signal slot. The kernel sets `receivingCta = (devComm.rank % nthreads) / blockDim.x` so that exactly that thread block runs `waitSignal()` for `signalIndex == receivingCta`. Every other CTA skips `waitSignal()` and only issues `ncclGin::put()` and later `flush()`.

每个 CTA 对外发起 `ncclGin::put()` 时都使用 `signalIndex = blockIdx.x`。在目标 rank 上，每个 peer 的一次 `put()` 会使发送方 CTA 的 `blockIdx.x` 所对应 signal 槽位增加一次。所有 CTA 都参与发起 `put()`，但当前 rank 只有一个接收 CTA 需要观察该 rank 对应 signal 槽位的完成情况。kernel 用 `receivingCta = (devComm.rank % nthreads) / blockDim.x` 选出这个线程块，让它对 `signalIndex == receivingCta` 调用 `waitSignal()`；其他 CTA 跳过等待，只发起 `put()`，之后执行 `flush()`。

> Once the signal watched by `receivingCta` has been incremented `nRanks` times, every peer has deposited its contribution into this rank’s receive buffer and the buffer is ready for consumption. That CTA’s `waitSignal()` blocks until that threshold using `signalValue + devComm.nRanks`, because each peer issues one inbound `ncclGin::put()` that advances this rank’s counter for that `signalIndex`. Before terminating, the kernel still calls `flush()` on all CTAs to commit outstanding outgoing `put()` operations. While `flush()` does not guarantee full remote completion of every side effect, it does ensure the local send buffer is safe to reuse from this kernel’s perspective. After `waitSignal()` and `flush()`, `bar.sync()` runs again. The barrier is added so that all ranks complete the collective before any rank exits the kernel.

当 `receivingCta` 监视的 signal 累加了 `nRanks` 次后，说明每个 peer 都已向当前 rank 的接收缓冲区写入数据，缓冲区可以消费。该 CTA 的 `waitSignal()` 以 `signalValue + devComm.nRanks` 为目标值等待，因为每个 peer 都会发起一次入站 `ncclGin::put()`，推动此 rank 对应 `signalIndex` 的计数增加。kernel 结束前，各 CTA 还会调用 `flush()`，提交未完成的出站 `put()`。`flush()` 不保证所有远端副作用都完全完成，但从当前 kernel 的角度，它可保证本地发送缓冲区可以安全复用。完成 `waitSignal()` 和 `flush()` 后，再执行一次 `bar.sync()`，使所有 rank 完成集合操作之后才有 rank 退出 kernel。

##### Compatibility adjustments｜兼容性调整

> The host setup, kernel, and explanation above reflect the NCCL 2.30 version and later. When targeting an older build, use the following as needed.

上面的 host 准备步骤、kernel 和解释适用于 NCCL 2.30 及以后版本。针对更旧的构建版本，需要按情况调整：

> - GPU-initiated networking (GIN) baseline — GIN is available since NCCL 2.28.7. `ncclDevCommCreate()` and `ncclGin` require a communicator that supports the device API and GIN.
> - Before NCCL 2.30 — no `worldGinBarrierCount` — `ncclGinBarrierSession` was only usable for rail connectivity, with the corresponding `railGinBarrierCount`. For world-team GIN barriers, set `barrierCount` to the number of CTAs (same as `gridDim.x`). In the kernel, use the hybrid `ncclBarrierSession` with `ncclTeamTagWorld()` together with `ncclGin` instead of `ncclGinBarrierSession` with `ncclTeamTagWorld()`.
> - Before NCCL 2.29.7 — no `ginConnectionType` — Set `ginForceEnable` to `true` to enable full GIN connectivity (equivalent to `NCCL_GIN_CONNECTION_FULL` once `ginConnectionType` exists). The `ginConnectionType` field is available starting with NCCL 2.29.7 (see `ncclDevCommRequirements` in Device API – Host-Side Setup).
> - Deprecated `ginForceEnable` — Prefer `ginConnectionType` on NCCL 2.29.7 and later. `ginForceEnable` is deprecated since NCCL 2.29.7.

- **GIN 基线**：GIN 从 NCCL 2.28.7 起可用；`ncclDevCommCreate()` 与 `ncclGin` 需要 communicator 支持 Device API 和 GIN。
- **NCCL 2.30 之前没有 `worldGinBarrierCount`**：`ncclGinBarrierSession` 当时只能通过相应的 `railGinBarrierCount` 用于 rail 连通性。若使用 world team GIN barrier，应将 `barrierCount` 设为 CTA 数（即 `gridDim.x`）；kernel 中使用结合 `ncclGin` 的混合 `ncclBarrierSession` 与 `ncclTeamTagWorld()`，而不是 `ncclGinBarrierSession` 与 `ncclTeamTagWorld()`。
- **NCCL 2.29.7 之前没有 `ginConnectionType`**：将 `ginForceEnable` 设为 `true` 以启用完整 GIN 连通性，相当于后续版本的 `NCCL_GIN_CONNECTION_FULL`。`ginConnectionType` 字段从 NCCL 2.29.7 起提供（参见 Device API – Host-Side Setup 中的 `ncclDevCommRequirements`）。
- **`ginForceEnable` 已弃用**：NCCL 2.29.7 及以后应优先使用 `ginConnectionType`；`ginForceEnable` 从 NCCL 2.29.7 起弃用。

### 重点解读

**Device API 仍需 host 端准备资源。** 自定义 kernel 可以从 GPU 端直接执行通信，但示例先在 host 端创建 communicator、注册对称 window，再通过 `ncclDevCommRequirements` 申请 barrier、signal 和 GIN 等资源。所需资源数量须与 kernel 的 CTA 布局相匹配。

**LSA、Multimem、GIN 和 CFT 的通信路径不同。** LSA 依靠设备间 P2P load/store；Multimem 利用 NVLink SHARP multicast；GIN 面向网络；CFT 使用 CUDA fabric 逻辑端点。能使用哪种路径取决于设备互连、注册内存、驱动、CUDA 版本与后端能力，不能只凭 API 名称判断可用性。

**GIN 示例区分三种完成条件。** 入口 barrier 确保参与 rank 已就绪；接收 CTA 的 `waitSignal()` 等待本 rank 的所有入站数据；各 CTA 的 `flush()` 使本地发送缓冲区可安全复用；末尾 barrier 再确保各 rank 完成集合操作。`flush()` 本身不代表所有远端副作用都已完成。

**指针示例保留原文，但解除注册参数有误。** 原文写 `ncclCommWindowDeregister(comm, &win)`，而 [NCCL API 签名](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#c.ncclCommWindowDeregister) 的第二个参数是 `ncclWindow_t win`，因此调用时应传 `win`，而非 `&win`。由 window 获取的指针在 window 或 communicator 生命周期结束时失效。

## Compute Fabric Transport

来源：[NVIDIA NCCL User Guide — Compute Fabric Transport](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cft.html#compute-fabric-transport)。

### 中英文对照

> Starting with NCCL 2.31, the Device API includes Compute Fabric Transport (CFT) helpers for kernels that communicate through CUDA fabric logical endpoints. CFT is useful when an application launches custom device-side communication kernels on CFT-capable systems and needs direct fabric put, get, reduction, or barrier operations.

从 NCCL 2.31 起，Device API 提供 Compute Fabric Transport（CFT）辅助功能，供 kernel 通过 CUDA fabric 逻辑端点通信。在支持 CFT 的系统上，如果应用要启动自定义设备端通信 kernel，直接执行 fabric put、get、归约或 barrier 操作，就可以使用 CFT。

#### Requirements｜要求

> CFT kernels require hardware, driver, and CUDA Toolkit support for fabric logical endpoints and fabric PTX instructions. In this release, the example CFT barrier program documents the practical requirements as CUDA Toolkit 13.3 and SM_100 architectures.

CFT kernel 要求硬件、驱动和 CUDA Toolkit 均支持 fabric 逻辑端点及 fabric PTX 指令。原文所述版本中的 CFT barrier 示例程序给出的实际要求是 CUDA Toolkit 13.3 和 SM_100 架构。

> Before using CFT in an application:

应用使用 CFT 前需要：

> - Allocate and register symmetric memory windows with NCCL.
> - Create a device communicator with CFT capabilities in [`ncclDevCommRequirements`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclDevCommRequirements).
> - Use CFT team helpers to address peers by CFT team rank.
> - Query logical endpoint IDs and offsets before issuing CFT operations.
> - Match the number of requested CFT barriers to the number of barrier slots used by the kernel.

- 使用 NCCL 分配并注册对称内存 window。
- 通过 [`ncclDevCommRequirements`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclDevCommRequirements) 创建具备 CFT 能力的 device communicator。
- 使用 CFT team 辅助函数，按 CFT team rank 寻址 peer。
- 在发起 CFT 操作前查询逻辑端点 ID 和偏移。
- 申请的 CFT barrier 数量须与 kernel 使用的 barrier 槽位数匹配。

#### Device Communicator Setup｜设备端 Communicator 配置

> The host code requests CFT capability when creating the device communicator.

Host 端代码在创建设备端 communicator 时申请 CFT 能力：

```cpp
ncclDevComm devComm;
ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;

reqs.cftCaps = NCCL_CFT;
reqs.cftBarrierCount = nCTAs;

NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));
```

> Request [`NCCL_CFT_MULTIMEM`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_cft.html#c.NCCL_CFT_MULTIMEM) in addition to [`NCCL_CFT`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_cft.html#c.NCCL_CFT) when a kernel uses multicast CFT operations or multimem CFT barriers:

如果 kernel 使用 multicast CFT 操作或 Multimem CFT barrier，除 [`NCCL_CFT`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_cft.html#c.NCCL_CFT) 外，还需申请 [`NCCL_CFT_MULTIMEM`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_cft.html#c.NCCL_CFT_MULTIMEM)：

```cpp
reqs.cftCaps = NCCL_CFT | NCCL_CFT_MULTIMEM;
```

> If a communicator cannot provide the requested CFT capability on all ranks, [`ncclDevCommCreate()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclDevCommCreate) fails.

如果 communicator 无法在所有 rank 上提供所申请的 CFT 能力，[`ncclDevCommCreate()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclDevCommCreate) 会失败。

#### Teams and Endpoints｜Team 与端点

> CFT operations may address peers through CFT teams rather than directly through world ranks. Use [`ncclTeamCft()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_cft.html#c.ncclTeamCft) or [`ncclTeamCftMultimem()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_cft.html#c.ncclTeamCftMultimem) on the host side, and the corresponding device overloads in kernels, to obtain the team layout.

CFT 操作可以通过 CFT team 寻址 peer，而不直接使用 world rank。Host 端可调用 [`ncclTeamCft()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_cft.html#c.ncclTeamCft) 或 [`ncclTeamCftMultimem()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_cft.html#c.ncclTeamCftMultimem)，kernel 中可调用相应的 device overload，取得 team 布局。

> Use logical endpoint query helpers to translate a registered window, byte offset, and peer into the values consumed by [`ncclCft`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_cft.html#_CPPv4I0E7ncclCft) operations. For device-side kernels, [`ncclGetCftLeInfo()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_cft.html#_CPPv416ncclGetCftLeInfo12ncclWindow_t6size_ti8ncclTeamRK11ncclDevCommP11ncclCftLeIdP6size_t) accepts a CFT-team peer rank and [`ncclGetPeerLeInfo()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_cft.html#_CPPv417ncclGetPeerLeInfo12ncclWindow_t6size_tiRK11ncclDevCommP11ncclCftLeIdP6size_t) accepts a world-rank peer.

逻辑端点查询函数会把已注册 window、字节偏移和 peer 转换成 [`ncclCft`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_cft.html#_CPPv4I0E7ncclCft) 操作所需的值。在设备端 kernel 中，[`ncclGetCftLeInfo()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_cft.html#_CPPv416ncclGetCftLeInfo12ncclWindow_t6size_ti8ncclTeamRK11ncclDevCommP11ncclCftLeIdP6size_t) 接收 CFT team 内的 peer rank，而 [`ncclGetPeerLeInfo()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_cft.html#_CPPv417ncclGetPeerLeInfo12ncclWindow_t6size_tiRK11ncclDevCommP11ncclCftLeIdP6size_t) 接收 world rank。

```cpp
ncclTeam cftTeam = ncclTeamCft(devComm);
ncclCftLeId leId;
size_t leOffset;

ncclGetCftLeInfo(win, byteOffset, peerCft, cftTeam, devComm,
                 &leId, &leOffset);
```

> Host code can use [`ncclGetCftDeviceLeInfo()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_cft.html#c.ncclGetCftDeviceLeInfo), [`ncclGetPeerDeviceLeInfo()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_cft.html#c.ncclGetPeerDeviceLeInfo), and [`ncclGetMultimemDeviceLeInfo()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_cft.html#c.ncclGetMultimemDeviceLeInfo). If host-side endpoint queries are needed before creating a CFT-enabled device communicator, configure `hostCftMode` in [`ncclConfig_t`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclConfig_t) during communicator initialization.

Host 端代码可以使用 [`ncclGetCftDeviceLeInfo()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_cft.html#c.ncclGetCftDeviceLeInfo)、[`ncclGetPeerDeviceLeInfo()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_cft.html#c.ncclGetPeerDeviceLeInfo) 和 [`ncclGetMultimemDeviceLeInfo()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_cft.html#c.ncclGetMultimemDeviceLeInfo)。如果在创建支持 CFT 的 device communicator 之前就需要从 host 端查询端点，应在初始化 communicator 时通过 [`ncclConfig_t`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclConfig_t) 配置 `hostCftMode`。

#### CFT Operations｜CFT 操作

> A CFT kernel stages data through shared memory, creates a [`ncclCft`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_cft.html#_CPPv4I0E7ncclCft) object, issues operations, and then submits and flushes the work.

CFT kernel 通过 shared memory 暂存数据，创建 [`ncclCft`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_cft.html#_CPPv4I0E7ncclCft) 对象并发起操作，然后提交并 flush 这些工作。

```cpp
__global__ void cftPutKernel(ncclDevComm devComm, ncclWindow_t win) {
  __shared__ ncclCftSmem cftSmem;
  __shared__ alignas(16) char smem[128];

  ncclCoopCta coop;
  ncclCft<ncclCoopCta> cft{coop, cftSmem};

  ncclTeam cftTeam = ncclTeamCft(devComm);
  int peer = (cftTeam.rank + 1) % cftTeam.nRanks;

  ncclCftLeId leId;
  size_t leOffset;
  ncclGetCftLeInfo(win, 0, peer, cftTeam, devComm, &leId, &leOffset);

  cft.put(coop, leId, leOffset, smem, sizeof(smem));
  cft.submit(coop);
  cft.flush(coop);
}
```

> CFT source and destination shared memory pointers must be 16-byte aligned, and the byte count must be a multiple of 16. See [Device API - CFT](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_cft.html#device-api-cft) for the full list of put, get, multicast, reduction, and pull-reduce helpers.

CFT 操作使用的源与目标 shared memory 指针必须按 16 字节对齐，传输字节数也必须是 16 的整数倍。put、get、multicast、归约及 pull-reduce 辅助函数的完整列表见 [Device API - CFT](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_cft.html#device-api-cft)。

#### CFT Barriers｜CFT Barrier

> CFT barriers synchronize device threads across the CFT team. The common setup is to request one barrier per CTA and use `blockIdx.x` as the barrier index.

CFT barrier 用于同步 CFT team 中的设备端线程。常见做法是每个 CTA 申请一个 barrier，以 `blockIdx.x` 作为 barrier 索引。

```cpp
__global__ void cftBarrierKernel(ncclDevComm devComm) {
  ncclCoopCta coop;
  ncclCftBarrierSession<ncclCoopCta> bar{coop, devComm, blockIdx.x};

  bar.sync(coop, cuda::memory_order_acq_rel,
           ncclMemProxyType::Generic, ncclMemProxyType::Fabric);
}
```

> For multicast CFT barriers, create the device communicator with [`NCCL_CFT_MULTIMEM`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_cft.html#c.NCCL_CFT_MULTIMEM) and pass `multimem=true` to the barrier session constructor.

对于 multicast CFT barrier，创建设备端 communicator 时需申请 [`NCCL_CFT_MULTIMEM`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_cft.html#c.NCCL_CFT_MULTIMEM)，并向 barrier session 构造函数传入 `multimem=true`。

#### Examples｜示例

> See the CFT barrier example under `docs/examples/06_device_api/04_cft_barrier` for a complete runnable CFT setup and kernel.

完整且可运行的 CFT 配置及 kernel 示例见 `docs/examples/06_device_api/04_cft_barrier` 中的 CFT barrier 示例。

#### Cross-proxy Fences｜跨 Proxy Fence

> Cross-proxy fences enforce memory ordering between operations issued by different producer and consumer proxies. The common use case for fence is ordering of data and flag updates to establish a `happens-before` relationship between producer stores and consumer loads to the same memory region. In the following example, the producer is the Fabric proxy, while the consumer (ommitted) is the Generic proxy. The consumer polls on the flag using relaxed loads and, after observing the flag update, issues a fence with `memory_order_acquire`, Fabric producer, Generic consumer and memory scope Sys.

Cross-proxy fence 用于约束不同生产者与消费者 proxy 发起的操作之间的内存顺序。常见用法是规定数据更新与 flag 更新的先后关系，从而让同一内存区域上的生产者 store 与消费者 load 建立 `happens-before` 关系。下例的生产者是 Fabric proxy，未展示的消费者是 Generic proxy。消费者以 relaxed load 轮询 flag；观察到 flag 更新后，再发出一个采用 `memory_order_acquire`、Fabric 为生产者、Generic 为消费者且作用域为 Sys 的 fence。

```cpp
__global__ void cftPutKernel(ncclDevComm devComm, ncclWindow_t win) {
  __shared__ ncclCftSmem cftSmem;
  __shared__ alignas(16) char smem[128];
  __shared__ alignas(16) int flag[4];

  ncclCoopCta coop;
  ncclCft<ncclCoopCta> cft{coop, cftSmem};

  ncclTeam cftTeam = ncclTeamCft(devComm);
  int peer = (cftTeam.rank + 1) % cftTeam.nRanks;

  ncclCftLeId leId;
  size_t leOffset;
  ncclGetCftLeInfo(win, 0, peer, cftTeam, devComm, &leId, &leOffset);

  cft.put(coop, leId, leOffset, smem, sizeof(smem));
  cft.submit(coop);
  cft.flush(coop);

  ncclMemFence(coop, cuda::memory_order_release, ncclMemProxyType::Fabric, ncclMemProxyType::Generic, ncclMemFenceScope::Sys);

  cft.put(coop, leId, leOffset + flagOffset, flag, 16);
  cft.submit(coop);
  cft.flush(coop);
}
```

> Another use case for fence is ordering accesses to shared memory between the Generic proxy and the Fabric proxy. In the following example, the Generic proxy updates shared memory and makes the updates visible to the Fabric proxy using a fence.

Fence 的另一种用途是规定 Generic proxy 与 Fabric proxy 对 shared memory 的访问顺序。下例由 Generic proxy 更新 shared memory，再用 fence 使更新对 Fabric proxy 可见。

```cpp
__shared__ ncclCftSmem cftSmem;
ncclCoopCta coop;
ncclCft<ncclCoopCta> cft{coop, cftSmem};

__shared__ alignas(16) payload[4];

payload[0] = payload[1] = payload[2] = payload[3] = 1;

ncclMemFence(coop, cuda::memory_order_release, ncclMemProxyType::Generic, ncclMemProxyType::Fabric, ncclMemFenceScope::Cta);

cft.put(coop, leId, leOffset, payload, 16);
cft.submit(coop);
cft.flush(coop);
```

### 重点解读

**CFT 使用逻辑端点寻址。** 与直接按 world rank 访问不同，CFT 可以先取得 CFT team，再把 window、字节偏移和目标 peer 转成逻辑端点 ID 与偏移。`ncclGetCftLeInfo()` 使用 CFT team rank，`ncclGetPeerLeInfo()` 使用 world rank；调用前需确定所持 rank 属于哪种编号空间。

**能力和 barrier 数在创建时确定。** 普通 CFT 操作申请 `NCCL_CFT`；multicast 操作或 Multimem barrier 还要申请 `NCCL_CFT_MULTIMEM`。`cftBarrierCount` 要覆盖 kernel 实际使用的 barrier 槽位；示例按每个 CTA 一个槽位，以 `blockIdx.x` 索引。只要有 rank 无法提供所请求能力，`ncclDevCommCreate()` 就会失败。

**`submit`、`flush` 与跨 proxy fence 解决不同问题。** 示例先发起 put，再提交和 flush；在“先数据、后 flag”的场景中，还用 Fabric→Generic 的 release fence 建立内存顺序。消费者需在观察到 flag 后执行相应的 acquire fence，才能按原文的协议读取数据。另一示例则用 Generic→Fabric 的 CTA 范围 fence，使 shared memory 中的写入对 Fabric 操作可见。

**示例是片段，原文有未定义标识符。** Fence 示例中的 `flagOffset` 未在片段内定义；最后一段的 `payload` 声明未写元素类型，`leId` 与 `leOffset` 也依赖前文上下文。代码已按原文保留，不能直接作为完整可编译程序使用。

## 既有学习笔记

#### **2. NCCL 的接口与使用方式**

NCCL 提供 **C/C++ API**，用户可以直接调用底层接口，或通过深度学习框架（如 PyTorch）间接使用。以下是关键接口和使用流程：

##### **2.1 核心 API 接口**
- **通信域初始化**

  ```c
  ncclCommInitAll(ncclComm_t* comms, int ndev, int* devlist);
  // 创建通信域（Communicator），指定参与通信的 GPU 设备列表。
  ```

- **集合通信操作**

  ```c
  ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
  // 执行 AllReduce 操作，支持数据类型和归约操作（如 SUM、MAX）。
  ```

- **点对点通信**

  ```c
  ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int pe,
           ncclComm_t comm, cudaStream_t stream);
  ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int pe,
           ncclComm_t comm, cudaStream_t stream);
  // 自定义点对点发送/接收操作。
  ```

- **资源释放**

  ```c
  ncclCommDestroy(ncclComm_t comm);
  // 销毁通信域，释放资源。
  ```

##### **2.2 使用流程示例**

以下是一个简单的 AllReduce 操作示例（基于 C/C++）：

```c
#include <nccl.h>
#include <cuda_runtime.h>

int main() {
  int rank, nDevices;
  cudaGetDeviceCount(&nDevices);
  ncclComm_t comm;
  ncclCommInitAll(&comm, nDevices, NULL);  // 初始化通信域

  float sendbuff = 1.0f, recvbuff = 0.0f;
  ncclAllReduce(&sendbuff, &recvbuff, 1, ncclFloat, ncclSum, comm, 0);  // 执行 AllReduce

  ncclCommDestroy(comm);  // 释放通信域
  return 0;
}
```

##### **2.3 与深度学习框架的集成**
- **PyTorch**：
  PyTorch 默认使用 NCCL 作为分布式训练后端。通过 `torch.distributed` 模块调用：

  ```python
  import torch.distributed as dist
  dist.init_process_group(backend='nccl')  # 初始化 NCCL 后端
  dist.all_reduce(tensor)  # 调用 AllReduce 操作
  ```

---

#### **3. NCCL 的工作原理**

##### **3.1 拓扑感知优化**

NCCL 会自动探测 GPU 间的连接拓扑（如 NVLink、PCIe、节点间网络），并构建最优通信结构（如 Ring 或 Tree）：

- **Ring 拓扑**：适用于 NVLink 连接的 GPU，通过环形结构高效传递数据。
- **Tree 拓扑**：适用于跨节点通信，通过树形结构减少跨网络设备的负载。

##### **3.2 并行与异步通信**
- **多线程调度**：NCCL 使用多线程管理通信任务，充分利用硬件资源。
- **CUDA 流绑定**：通信操作与 CUDA 流绑定，实现计算与通信的重叠（Overlap）。

##### **3.3 硬件加速技术**
- **GPUDirect P 2 P**：允许 GPU 直接通信，绕过 CPU 内存。
- **GPUDirect RDMA**：通过 RDMA 技术实现跨节点的 GPU 直接内存访问。
- 
---

## **核心功能**

1. **集合通信操作**
    
    - **AllReduce**：跨多个设备/节点聚合数据（如梯度同步）。
    - **Broadcast**：从一个设备向所有设备广播数据。
    - **Reduce**：汇总多个设备的数据到目标设备。
    - **AllGather**：收集所有设备的数据到每个设备。
    - **ReduceScatter**：分片汇总数据后分发到各设备。
2. **点对点通信**
    
    - **Send/Recv**：直接在设备间传输数据。
    - **Scatter/Gather**：分发/收集数据到多个设备。
    - **All-to-all**：全互连通信模式。
3. **多 GPU 管理**
    
    - 支持单线程管理多个 GPU。
    - 可创建多个通信器（communicators）并行运行。
    - 支持 CUDA 流（CUDA Stream）和 CUDA Graphs 集成。
4. **容错与错误处理**
    
    - 异步错误检测（如 `ncclCommGetAsyncError`）。
    - 通信器销毁和异常终止（`ncclCommAbort`）。


---

## **关键 API**

1. **通信器管理**
    
    - `ncclGetUniqueId`：生成唯一通信器 ID。
    - `ncclCommInitRank`：初始化通信器（指定设备、ID 和进程排名）。
    - `ncclCommFinalize` / `ncclCommDestroy`：销毁通信器。
2. **集体通信函数**
    
    - `ncclAllReduce` / `ncclBroadcast` / `ncclReduce` / `ncclAllGather` / `ncclReduceScatter`。
3. **组操作（Group Calls）**
    
    - `ncclGroupStart` / `ncclGroupEnd`：组合多个操作为原子操作。
4. **内存管理**
    
    - `ncclMemAlloc` / `ncclMemFree`：分配/释放内存（支持 NVLink、IB 等优化）。

---

## **环境变量**

1. **网络配置**
    
    - `NCCL_SOCKET_IFNAME`：指定网络接口（如 `eth0`）。
    - `NCCL_IB_HCA`：InfiniBand HCA 设备选择（如 `mlx5_0`）。
    - `NCCL_IB_TIMEOUT` / `NCCL_IB_RETRY_CNT`：InfiniBand 超时与重试策略。
2. **性能优化**
    
    - `NCCL_ALGO` / `NCCL_PROTO`：指定通信算法（环形/树形）和协议（LL/LL128）。
    - `NCCL_NET_GDR_LEVEL`：控制 GPU Direct RDMA 级别。
3. **调试与日志**
    
    - `NCCL_DEBUG=INFO`：启用详细日志输出。
    - `NCCL_DEBUG_FILE`：指定日志文件路径。
4. **其他配置**
    
    - `NCCL_IGNORE_CPU_AFFINITY`：忽略 CPU 亲和性设置。
    - `NCCL_P2P_DISABLE`：禁用 P2P 通信（用于调试）。

---

## **与 MPI 集成**

- **多设备支持**：在 MPI 程序中结合 NCCL 实现多 GPU 通信。
- **混合模式**：NCCL 处理设备间通信，MPI 处理跨节点通信（通过 CUDA-aware MPI）。
- **示例**：使用 `ncclCommInitRank` 在每个进程中初始化 NCCL 通信器。

---

## **常见问题与调试**

1. **GPU Direct 问题**
    
    - 检查驱动版本、PCIe 拓扑（`NCCL_TOPO_DUMP_FILE`）。
    - 禁用 P2P（`NCCL_P2P_DISABLE=1`）排查问题。
2. **网络问题**
    
    - InfiniBand 配置（`NCCL_IB_HCA`、`NCCL_IB_SL`）。
    - RoCE/以太网适配（`NCCL_SOCKET_FAMILY`）。
3. **性能瓶颈**
    
    - 使用 `NCCL_DEBUG=INFO` 分析通信路径。
    - 调整 `NCCL_NET_GDR_READ` 优化内存传输。
4. **容器环境**
    
    - Docker 需启用 `--gpus` 和共享 IPC（`--ipc=host`）。

---

## **版本迁移**

- **从 NCCL 1 到 2 的差异**：
    - 通信器初始化方式变化（如 `ncclCommInitRank`）。
    - 集合操作参数顺序调整（如 `AllGather`）。
    - 新增非阻塞组操作（2.2+）。

---

## **文档资源**

- **官方链接**：NCCL User Guide
- **版本**：2.23.4（当前总结基于此版本）。

---

此总结覆盖了 NCCL 的核心功能、API、配置选项及常见调试方法，适用于分布式训练、多 GPU 通信场景的开发与优化。

# 分层

You're asking about the differences between three distinct layers in NCCL's communication architecture: communication protocols, hardware types, and transport layers.

## Answer

These three concepts operate at different abstraction levels in NCCL's communication stack and serve distinct purposes <cite/>.

## Communication Protocols (Device-Level)

Communication protocols (LL, LL128, Simple) are device-side data transfer mechanisms that define how data is packaged and synchronized during GPU kernel execution [1](#4-0) .

Each protocol has specific characteristics:

- **LL**: Uses flag-based synchronization with 8-byte granularity and half buffer for data [2](#4-1)

- **LL128**: Cache-line optimized with 120/128 data/total ratio [3](#4-2)

- **Simple**: Direct data transfer without per-element flags [4](#4-3)

## Hardware Types (Topology-Level)

Hardware types represent the physical interconnect topology between GPUs and determine bandwidth and latency characteristics [5](#4-4) . The main types include:

- **NVLINK**: High-bandwidth GPU-to-GPU connections (20-40 GB/s depending on generation)

- **PCI**: PCIe-based connections (~12 GB/s)

- **Network**: Inter-node network connections

These are used in the tuning system to model performance characteristics [6](#4-5) . The hardware type affects protocol selection - for example, LL protocol has different latency profiles for NVLink (0.6μs) vs PCI (1.0μs) vs Network (2.7μs) connections.

## Transport Layer (Connection-Level)

Transport layers handle the actual establishment and management of communication channels between ranks [7](#4-6) . The main transports include:

- **P2P Transport**: Direct GPU-to-GPU memory access using CUDA IPC or cuMem APIs

- **SHM Transport**: Shared memory communication

- **NET Transport**: Network-based communication

- **NVLS Transport**: NVLink SHARP multicast for collective operations [8](#4-7)

Transport selection happens during communicator initialization and determines how memory regions are mapped and connected between ranks [9](#4-8) .

## Interaction Between Layers

These layers interact hierarchically:

1. **Transport layer** establishes the physical connections and memory mappings

2. **Hardware type** influences performance modeling and protocol selection during tuning [10](#4-9)

3. **Communication protocol** determines the actual data transfer mechanism used by device kernels

For example, a P2P transport over NVLink hardware might select LL protocol for small messages to minimize latency, while the same transport over PCI might prefer Simple protocol for larger messages to maximize bandwidth.

**Notes**

The key distinction is abstraction level: protocols are device-side data handling mechanisms, hardware types describe physical topology characteristics, and transports manage connection establishment and memory mapping. Each serves a specific role in NCCL's layered communication architecture.

Wiki pages you might want to explore:

- [Communication System Foundation (NVIDIA/nccl)](/wiki/NVIDIA/nccl#2.2)

- [Bootstrap and Group Management (NVIDIA/nccl)](/wiki/NVIDIA/nccl#2.3)

- [Topology Management (NVIDIA/nccl)](/wiki/NVIDIA/nccl#4)
