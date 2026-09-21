> https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html#

# CUDA Cooperative Groups（协同线程组）官方文档总结

Cooperative Groups are an extension to the CUDA programming model for organizing groups of collaborating threads. Cooperative Groups allow developers to control the granularity at which threads are collaborating, helping them to express richer, more efficient parallel decompositions. Cooperative Groups also provide implementations of common parallel primitives like scan and parallel reduce.

协作组是对 CUDA 编程模型的扩展，用于组织协作线程组。协作组允许开发人员控制线程协作的粒度，帮助他们实现更丰富、更高效的并行分解。协作组还提供了常见并行原语的实现，如扫描和并行归约。

## 一、核心概述

Historically, the CUDA programming model has provided a single, simple construct for synchronizing cooperating threads: a barrier across all threads of a thread block, as implemented with the `__syncthreads()` intrinsic function.

1. **传统局限**：从历史来看，CUDA 编程模型一直为同步协作线程提供了单一且简单的构造：即线程块所有线程的屏障，这一功能通过 `__syncthreads()` 内置函数实现；如需 warp 内、跨 block 同步，开发者只能手写非标准、脆弱、难以跨代维护的同步代码。
2. **CG 价值**
    - 细粒度可控线程分组，表达更丰富高效的并行分解；
    - 内置标准化并行原语：规约 reduce、扫描 scan；
    - 统一安全、向前兼容的线程协作机制。

> 完整 API 见 [Cooperative Groups API](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#cg-api-partition-header)

## 二、协同组句柄与成员接口

Cooperative Groups are managed via a Cooperative Group Handle. The Cooperative Group handle allows participating threads to learn their position in the group, the group size, and other group information. Select group member functions are shown in the following table.

协作组通过协作组句柄进行管理。协作组句柄允许参与的线程了解其在组中的位置、组大小以及其他组信息。部分组成员函数如下表所示。

所有操作依托**组句柄（Cooperative Group Handle）**，线程可通过句柄查询组信息：

| 接口 | 返回内容 |
|------|--------|
| `thread_rank()` | 当前线程在组内序号 |
| `num_threads()` | 组总线程数 |
| `thread_index()` | 线程在启动 block 内三维索引 |
| `dim_threads()` | block 三维线程维度 |

## 三、隐式内置分组（无手动创建即可使用）

Groups representing the grid and thread blocks are implicitly created based on the kernel launch configuration. These “implicit” groups provide a starting point that developers can explicitly decompose into finer grained groups. Implicit groups can be accessed using the following methods:

表示网格和线程块的组是基于内核启动配置隐式创建的。这些“隐式”组为开发者提供了一个起点，开发者可将其显式分解为更细粒度的组。可通过以下方法访问隐式组：

| 获取接口 | 分组范围 | 限制说明 |
|---------|---------|---------|
| `this_thread_block()` | 当前线程块所有线程 | 通用 |
| `this_grid()` | 整个网格全部线程 | 跨 block 同步需特殊内核启动接口 |
| `coalesced_threads()` | 当前 warp 活跃线程 | 活跃线程集合不固定、运行中不保证连续 |
| `this_cluster()` | 当前 cluster 线程组 | 算力 9.0+；非 cluster 启动默认 1x1x1 cluster |

> [!info] 性能编码规范
> 1. 尽早创建隐式组句柄（分支前），全程复用；
> 2. 组句柄传参**必须传引用**，禁止拷贝构造；句柄声明时必须初始化，无默认构造函数。

## 四、手动创建子分组 TODO

基于父组划分生成细粒度子组，**划分是集合操作，父组全部线程必须参与**，否则死锁/数据损坏。三种划分方式：

1. `tiled_partition<N>`：一维行优先、固定大小分片（例：block 拆分为 8 线程 tile）；
2. `labeled_partition`：基于整型标签一维分组；
3. `binary_partition`：标签仅 0/1 的特殊二元划分。

示例：

```cpp
namespace cg = cooperative_groups;
cg::thread_block my_block = cg::this_thread_block();
cg::thread_block_tile<8> tile8 = cg::tiled_partition<8>(my_block);
```

## 五、多粒度同步机制

Prior to the introduction of Cooperative Groups, the CUDA programming model only allowed synchronization between thread blocks at a kernel completion boundary. Cooperative groups allows developers to synchronize groups of cooperating threads at different granularities.在引入协作组之前，CUDA 编程模型仅允许在内核完成边界处对线程块进行同步。协作组则让开发者能够以不同的粒度对协作线程组进行同步操作。

### 1. 基础同步 sync()

You can synchronize a group by calling the collective `sync()` function. Like `__syncthreads()`, the `sync()` function makes the following guarantees: 你可以通过调用 `sync()` 函数来同步一个组。与 `__syncthreads()` 一样，`sync()` 函数做出以下保证：

- All memory accesses (e.g., reads and writes) made by threads in the group before the synchronization point are visible to all threads in the group after the synchronization point.组内线程在同步点之前进行的所有内存访问（例如读取和写入），在同步点之后对组内所有线程均可见。
    
- All threads in the group reach the synchronization point before any thread is allowed to proceed beyond it.组内所有线程在允许任何线程越过同步点之前都需到达该同步点。

`cg::sync(group)` 等价 `__syncthreads()`，保障两点语义：

- 同步点前所有内存读写对组内所有线程可见；
- 组内全部线程抵达同步点后，才允许继续执行。
- 注意：CUDA 13 起 CG 不再支持多设备同步。

.下面的示例展示了一个与 `__syncthreads()` 等效的 `cooperative_groups::sync()`。

```c
namespace cg = cooperative_groups;

cg::thread_block my_group = cg::this_thread_block();

// Synchronize threads in the block
cg::sync(my_group);
```

Cooperative groups can be used to synchronize the entire grid.

### 2. 高级屏障 Barrier

Cooperative Groups provides a barrier API similar to `cuda::barrier` that can be used for more advanced synchronization. Cooperative Groups barrier API differs from `cuda::barrier` in a few key ways: 协作组提供了一个与 `cuda::barrier` 类似的屏障 API，可用于更高级别的同步。协作组屏障 API 与 `cuda::barrier` 在几个关键方面存在差异：

- Cooperative Groups barriers are automatically initialized 协作组屏障是自动初始化的
- All threads in the group must arrive and wait at the barrier once per phase.组内所有线程每个阶段都必须到达屏障并在屏障处等待。
- `barrier_arrive` returns an `arrival_token` object that must be passed into the corresponding `barrier_wait`, where it is consumed and cannot be used again.`barrier_arrive` 返回一个 `arrival_token` 对象，该对象必须传入对应的 `barrier_wait` 中，在其中会被消耗且无法再次使用。

Programmers must take care to avoid hazards when using Cooperative Groups barriers: 程序员在使用协作组屏障时必须小心，以避免潜在风险：

- No collective operations can be used by a group after calling `barrier_arrive` and before calling `barrier_wait`.组在调用 `barrier_arrive` 之后、调用 `barrier_wait` 之前，不得执行任何集体操作。
- `barrier_wait` only guarantees that all threads in the group have called `barrier_arrive`. `barrier_wait` does NOT guarantee that all threads have called `barrier_wait`.`barrier_wait` 仅保证组内的所有线程都已调用 `barrier_arrive`。`barrier_wait` 不保证所有线程都已调用 `barrier_wait`。

典型场景：cluster 内多 block 同步访问分布式共享内存 dsmem。

## 六、内置集合并行原语 TODO

所有集合操作要求组内全部线程参与，参数值规则需遵循 API 说明，否则行为未定义。

### 1. Reduce 并行规约

支持求和、最大/最小、位运算；算力 8.0+ 硬件加速（仅 4 字节类型），低算力自动软件回退。

示例：block 内求和，rank0 写出结果。

### 2. Scan 前缀扫描

支持 `inclusive_scan`（包含自身）、`exclusive_scan`（不含自身），支持自定义规约算子。

### 3. invoke_one / invoke_one_broadcast

- `invoke_one`：组内任选一个线程执行串行逻辑；
- `invoke_one_broadcast`：执行结果广播给组所有线程；
限制：传入的回调函数内**禁止本组内通信/同步**，允许和组外线程交互。常用于仅单次打印、初始化操作。

## 七、异步内存搬运 memcpy_asyncTODO

用于全局 ↔ 共享内存异步拷贝，计算与数据传输重叠掩盖访存延迟，类似预取：

1. `cg::memcpy_async` 发起异步加载；
2. 中间可执行无关计算（不可访问未就绪共享内存）；
3. `cg::wait(group)` 阻塞组全部线程，等待拷贝完成后才可读取共享内存。

### 对齐要求

- 仅源为全局、目标为共享内存且 ≥4 字节对齐时，拷贝才是异步；
- 最优性能推荐全局、共享内存均 16 字节对齐。

## 八、大规模网格组 this_grid()TODO

Cooperative Groups allows for large groups that span the entire grid. All Cooperative Group functionality described previously is available to these large groups, with one notable exception: synchronizing the entire grid requires using the `cudaLaunchCooperativeKernel` runtime launch API.协作组支持跨越整个网格的大型组。之前描述的所有协作组功能都可用于这些大型组，但有一个明显的例外：同步整个网格需要使用 `cudaLaunchCooperativeKernel` 运行时启动 API。

Multi-device launch APIs and related references for Cooperative Groups have been removed as of CUDA 13.截至 CUDA 13，多设备启动 API 以及协作组的相关引用已被移除。

### cudaLaunchCooperativeKernel 使用条件

`cudaLaunchCooperativeKernel` is a CUDA runtime API function used to launch a single-device kernel that employs cooperative groups, specifically designed for executing kernels that require inter-block synchronization. This function ensures that all threads in the kernel can synchronize and cooperate across the entire grid, which is not possible with traditional CUDA kernels that only allow synchronization within individual thread blocks. `cudaLaunchCooperativeKernel` ensures that the kernel launch is atomic, i.e. if the API call succeeds, then the provided number of thread blocks will launch on the specified device.`cudaLaunchCooperativeKernel` 是一种 CUDA 运行时 API 函数，用于启动采用协作组的单设备内核，专为需要块间同步的内核执行而设计。该函数可确保内核中的所有线程在整个网格范围内实现同步与协作，而仅支持单个线程块内同步的传统 CUDA 内核则无法做到这一点。`cudaLaunchCooperativeKernel` 保证内核启动具有原子性，即如果 API 调用成功，指定数量的线程块将在指定设备上启动。

It is good practice to first ensure the device supports cooperative launches by querying the device attribute `cudaDevAttrCooperativeLaunch`: 首先通过查询设备属性 `cudaDevAttrCooperativeLaunch` 来确保设备支持协作式启动是一种良好的做法：

```c
int dev = 0;
int supportsCoopLaunch = 0;
cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
```

which will set `supportsCoopLaunch` to 1 if the property is supported on device 0. Only devices with compute capability of 6.0 and higher are supported. In addition, you need to be running on either of these: 如果设备 0 上支持该属性，这会将 `supportsCoopLaunch` 设置为 1。仅支持计算能力为 6.0 及更高版本的设备。此外，你需要运行在以下任一环境中：

- The Linux platform without MPS 无 MPS 的 Linux 平台
- The Linux platform with MPS and on a device with compute capability 7.0 or higher 运行多进程服务（MPS）的 Linux 平台，且设备的计算能力为 7.0 及更高版本
- The latest Windows platform 最新的 Windows 平台

## 九、关键风险与最佳实践汇总

1. 分组划分、sync、barrier、reduce/scan 均为集合操作，组内所有线程必须执行，分支不均会引发死锁；
2. 组句柄尽早创建、传引用、不拷贝；
3. 跨 block 同步必须用 cooperative kernel 启动；
4. memcpy_async 重视内存对齐，拷贝完成前禁止访问共享内存；
5. barrier arrive 与 wait 间不能执行本组集合操作，token 一次性使用；
6. invoke_one 回调内不能做本组同步/通信；
7. CUDA 13 不再支持 CG 多设备同步。
