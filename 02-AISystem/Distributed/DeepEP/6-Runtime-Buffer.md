下面就开始讲解 C++ 提供的接口里对 Buffer 的实现。

# Cmake

首先根据编译确定接口用到了哪些依赖，实现的功能是什么。

## **一、核心功能：构建 PyTorch 扩展模块 `deep_ep_cpp`**

该文件是用于编译 `deep_ep_cpp` 扩展模块的 CMake 配置脚本，核心目标是将 C++/CUDA 代码（含分布式通信逻辑）封装为 Python 可调用的 `deep_ep_cpp` 模块，支持高性能 GPU 通信（如 NVLink/RDMA）和 MoE（Mixture of Experts）模型的 token 分发/聚合。

## **二、关键编译配置**

1. **优化与兼容性设置**
	- 启用最高级优化（`-O3`）和位置无关代码（`-fPIC`），提升运行效率并支持动态链接。
	- 开启 CUDA 分离编译（`CUDA_SEPARABLE_COMPILATION ON`），允许 CUDA 代码独立编译为设备端模块，减少整体编译时间。
	- NVCC 编译参数：
	- `-DENABLE_FAST_DEBUG`：启用快速调试模式（可能关闭部分安全检查以加速）。
	- `--ptxas-options`：控制 PTX 汇编器行为，包括输出详细信息（`--verbose`）、寄存器使用级别（`--register-usage-level=10`）和本地内存使用警告（`--warn-on-local-memory-usage`），优化 GPU 内核资源占用。
2. **GPU 架构支持**
	- `CUDA_ARCH_LIST "9.0"`：指定编译支持 NVIDIA Hopper 架构（如 H100 GPU），确保生成适配 SM90 的优化代码。
	- `TORCH_CUDA_ARCH_LIST`：同步 PyTorch 的 CUDA 架构列表，避免兼容性问题。
3. **代码标准**
	- 设置 C++ 和 CUDA 标准为 C++17（`CMAKE_CXX_STANDARD 17`、`CMAKE_CUDA_STANDARD 17`），支持现代 C++ 特性（如结构化绑定、折叠表达式）。

## **三、依赖项说明**

通过 `find_package` 和路径配置引入以下核心依赖，确保编译和运行时能正确链接库文件和头文件：

| 依赖项             | 作用                                                                                 |     |
| --------------- | ---------------------------------------------------------------------------------- | --- |
| **CUDAToolkit** | 提供 CUDA 运行时库、NVCC 编译器及 GPU 通信 API（如 CUDA IPC、NVLink），是 GPU 加速的基础。                  |     |
| **pybind11**    | 实现 C++ 到 Python 的绑定，将 C++ 类/函数封装为 Python 可调用的模块（如 `Buffer` 类的方法）。                  |     |
| **Torch**       | PyTorch 库，提供张量操作、CUDA 流管理、自动求导等核心功能，扩展模块需与 PyTorch 类型系统兼容。                         |     |
| **NVSHMEM**     | NVIDIA 分布式共享内存库，支持跨节点（RDMA）的 GPU 内存直接访问，是 `internode_dispatch`/`combine` 跨节点通信的基础。 |     |

## **四、构建流程关键步骤**

1. **依赖定位与路径配置**

- `include_directories`：添加 CUDA、PyTorch、Python、NVSHMEM 的头文件路径，确保编译器能找到 `torch/tensor.h`、`nvshmem.h` 等关键头文件。
- `link_directories`：指定 Torch、CUDA、NVSHMEM 的库文件路径，确保链接器能找到 `libtorch.so`、`libnvshmem.so` 等动态库。

1. **子模块编译**

- `add_subdirectory(kernels)`：编译 `kernels` 子目录下的 CUDA 内核代码（如 `intranode_dispatch`/`combine` 的底层 GPU 核函数），生成静态库供主模块链接。

1. **扩展模块生成**

- `pybind11_add_module(deep_ep_cpp deep_ep.cpp)`：将 `deep_ep.cpp`（含 `Buffer` 类的 C++ 实现和 pybind11 绑定代码）编译为 Python 扩展模块 `deep_ep_cpp`。
- `target_link_libraries`：链接依赖库，包括 CUDA 运行时库、PyTorch 库（`${TORCH_LIBRARIES}`）和 Python 绑定库（`torch_python`），确保模块能被 Python 导入并调用 GPU 功能。

我把这部分按当前实现重写成更贴近真实代码的版本，并直接按“成员/函数/职责”分成三层，去掉旧版里过时的概念和不准确的字段描述。

# Buffer 概述：按当前实现重写版

下面这份内容是按当前代码真实状态整理的，依据是：

- Python 侧：legacy.py
- C++ 侧：buffer.hpp

它更接近现在这个版本的真实设计，而不是过去的旧版描述。

---

# Buffer 成员

## 1. 静态断言

```cpp
EP_STATIC_ASSERT(LEGACY_NUM_MAX_NVL_PEERS == 8, "The number of maximum NVLink peers must be 8");
```

这是一个编译期断言，确保当前实现只支持最多 8 个 NVLink peer。

它说明了这个设计的一个关键事实：

- “节点内 EP 通信的 peer 规模被假设为 8 个 GPU”
- 这是底层 kernel / buffer 分配 / rank 计算所依赖的硬约束
- 如果不是 8，就不符合当前实现的设计假设

这个断言对应你旧版里提到的 `NUM_MAX_NVL_PEERS`，在当前版本里它已改名为 `LEGACY_NUM_MAX_NVL_PEERS`，但是含义基本相同。

---

## 2. 缓冲区管理（Buffer Management）

这一部分负责“通信所需的共享内存 / buffer 申请与维护”。

### 核心字段

- `int low_latency_buffer_idx = 0;`
  - 低延迟模式下的 buffer 索引
  - 低延迟 dispatch/combine 通常复用几块固定 buffer，不是每轮都 new 一个新数组
  - 这里记录“当前使用的是第几块低延迟 buffer”

- `bool low_latency_mode = false;`
  - 当前 buffer 是否处于低延迟模式
  - 这个 flag 会影响:
    - RDMA/NVSHMEM 初始化方式
    - 通信路径选择
    - buffer 分配和销毁策略

- `int64_t num_nvl_bytes;`
  - 该 rank 在 NVLink 组内所需的共享 buffer 大小
  - 这是节点内通信的主缓冲区

- `void* buffer_ptrs[LEGACY_NUM_MAX_NVL_PEERS] = {nullptr};`
  - 每个 NVLink peer 的本地共享内存地址
  - 设计上是按 rank 组织的一组指针
  - 这些 pointer 能让 GPU / kernel 访问其他 peer 的 buffer

- `void** buffer_ptrs_gpu = nullptr;`
  - GPU 可见的 `buffer_ptrs` 数组
  - 这是 kernel 真正访问“其他 peer buffer”所需的地址表

- `int64_t num_rdma_bytes;`
  - RDMA/NVSHMEM 需要的 buffer 大小
  - 这是跨节点或 low-latency 模式的核心共享区域

- `void* rdma_buffer_ptr = nullptr;`
  - NVSHMEM 分配出来的 RDMA buffer 基址
  - 对低延迟路径非常关键

### 这部分的意义

当前版本的 `Buffer` 是“多种通信模式共用的 runtime 容器”，而不是单纯的 NVLink or RDMA 其中一个：

- NVLink 用于节点内高吞吐
- RDMA / NVSHMEM 用于跨节点或低延迟方案
- low_latency_mode 则会强制使用更轻量的路径

所以这里的 buffer 不是只有一份，而是“多套共享内存同步状态”在同一个对象中。

---

## 3. 设备与通信标识（Device & Communication Identity）

这一层负责“谁是谁、怎么分组、怎么定位各自本地硬件”。

### 核心字段

- `int device_id;`
  - 当前 CUDA device id

- `int num_device_sms;`
  - 当前 GPU 的 SM 数
  - 与 kernel 任务切分、并行度、channel 数相关

- `int rank, rdma_rank, nvl_rank;`
  - `rank`: 全局 rank
  - `rdma_rank`: 在 RDMA 组中的 rank
  - `nvl_rank`: 在 NVLink 组中的 rank
  - 这是当前实现最关键的分层标识

- `int num_ranks, num_rdma_ranks, num_nvl_ranks;`
  - 不同通信域中的 rank 总数
  - 例如 `num_ranks` 是全局总数，`num_nvl_ranks` 往往等于节点内 GPU 数

- `shared_memory::MemHandle ipc_handles[LEGACY_NUM_MAX_NVL_PEERS];`
  - 用于共享本地 GPU 显存的 IPC handle
  - 这些 handle 会在 `sync()` 中广播给全局所有 rank

### 关键设计

当前代码里 rank 被明确拆成：

```cpp
rdma_rank = rank / LEGACY_NUM_MAX_NVL_PEERS;
nvl_rank = rank % LEGACY_NUM_MAX_NVL_PEERS;
```

也就是说：

- 全局 rank 会被分成两层：RDMA 维度和 NVLink 维度
- 这是因为 EP 通信不是单一一维结构，而是“节点内 + 节点间”混合拓扑

所以这里的 rank 可以看成是双层索引，不再只是一个简单的全局编号。

---

## 4. 跨进程 / GPU 通信状态（Inter-process/Device Communication）

这一部分负责“共享内存和同步信号”的连接。

### 核心字段

- `bool available = false;`
  - 表示运行时是否已经初始化完成
  - 只有 `sync()` 成功调用并完成 IPC/NVSHMEM 同步后，才设为 true

- `int* barrier_signal_ptrs[LEGACY_NUM_MAX_NVL_PEERS] = {nullptr};`
  - 用于 NVLink 通信的 barrier 信号
  - 这类信号用于跨 rank 同步“某轮 communications 已完成/已 ready”

- `int** barrier_signal_ptrs_gpu = nullptr;`
  - GPU 可见的 barrier 指针数组
  - kernel 会直接读/写这些地址来同步状态

### 这部分的意义

当前版本没有把“同步”压缩成一个简单的 bool，而是使用：

- shared memory pointer
- IPC handle
- barrier signal pointer

来组成通信运行时的全局状态。

这说明底层 kernel 不是简单地用 API call 启动，而是需要：

- 进程间共享内存 ready
- barrier ready
- 所有 ranks 都拥有同一份状态

否则就不能安全执行 dispatch / combine。

---

## 5. Stream 与同步控制（Stream & Synchronization）

这一部分控制“计算流和通信流如何分离”。

### 核心字段

- `at::cuda::CUDAStream comm_stream;`
  - 专门用于通信的 CUDA stream
  - 把通信和计算隔离开，避免互相阻塞

- `volatile int* moe_recv_counter = nullptr;`
  - 总接收 token 数计数器
  - 这是 CPU 侧等待 dispatch 是否完成的关键状态

- `int* moe_recv_counter_mapped = nullptr;`
  - 对应 device 可见的映射指针

- `volatile int* moe_recv_expert_counter = nullptr;`
  - 每个 local expert 的接收计数器
  - 用于观察“每个本地专家收到多少 token”

- `int* moe_recv_expert_counter_mapped = nullptr;`
  - 对应 device 可见的 expert 计数器

- `volatile int* moe_recv_rdma_counter = nullptr;`
  - RDMA 级接收计数器
  - 适用于跨节点或低延迟 RDMA 参与的通信

- `int* moe_recv_rdma_counter_mapped = nullptr;`
  - 对应 RDMA 计数器设备指针

### 这部分的关键点

当前版本的代码明显使用“计数器 + busy wait”来协调通信状态：

- 发送方发数据
- 接收方要等 `moe_recv_counter` / `moe_recv_expert_counter` 变成有效值
- 之后才继续做接收和 reduce

这说明当前实现依赖“显式同步计数器”，而不是只调用一个同步 API 然后结束。

---

## 6. 资源生命周期（Resource Lifecycle）

这一部分负责“内存与对象的正确释放”。

### 核心字段

- `bool explicitly_destroy;`
  - 是否要求用户手动调用 `destroy()`
  - 如果为 true，则析构时不会自动释放，需要调用 `destroy()`

- `bool destroyed = false;`
  - 标识是否已经释放
  - 避免重复 free

- `void* workspace = nullptr;`
  - 临时工作区
  - 用于一类通信中的中间数据（前缀和、队列头尾偏移等）

### 这部分的意义

当前代码中非常强调：

- `destroy()` 必须被正确调用
- `available`、`destroyed` 的状态需要管理得准确
- 若不做，容易出现：
  - IPC 句柄泄漏
  - NVSHMEM free 错误
  - GPU 显存/host memory leak

---

# Buffer 成员函数概述

## 1. 生命周期管理（Lifecycle Management）

### `Buffer(...)`

构造函数，完成：

- rank / group / device 信息初始化
- NVLink buffer 分配
- IPC handle 获取
- work space 分配
- host-side receive counter 初始化
- 由 `sync(...)` 把全部状态同步到其他 rank

它的核心不是“创建一个对象”，而是“创建一套可通信的 runtime 状态”。

### `~Buffer()`

析构函数。
如果 `explicitly_destroy == false`，会自动调用 `destroy()`。
如果为 true，则要求手动调用 `destroy()`，否则会打 warning。

---

### `sync(...)`

这是当前版本特别重要的初始化函数，位于 buffer.hpp。

它做的事情非常多：

- 同步 NVLink IPC handles
- 打开其他 rank 的共享内存
- 把 buffer pointer 和 barrier signal pointer 复制到 GPU
- 如果 `num_rdma_bytes > 0`，进一步初始化 NVSHMEM
- 分配 `rdma_buffer_ptr`
- 如果启用 shrink，则初始化 `mask_buffer_ptr` / `sync_buffer_ptr`
- 最后设置 `available = true`

这一层可以说是“真正把 runtime 从‘一个对象’变成‘一个可用通信系统’”。

### `destroy()`

释放资源，步骤包括：

- 同步 GPU
- 对 NVLink buffer 做 barrier
- close 远端 IPC handle
- free buffer_ptrs[nvl_rank]
- free NVSHMEM 分配的 rdma buffer
- free workspace
- free MoE host counters

它是当前实现里非常关键的“清理入口”。

---

## 2. 状态检查（Status Check）

### `is_available() const`

判断当前 buffer 是否已经 fully initialized。

- 若 `available == false`，默认不能使用 dispatch/combine

### `is_internode_available() const`

判断跨节点通信是否可用：

- 需要 `is_available()`
- 也要求 `num_ranks > LEGACY_NUM_MAX_NVL_PEERS`

也就是说：

- 节点内通信有自己的状态
- 跨节点通信需要更高要求

---

## 3. 通信与设备信息（Communication & Device Info）

### `get_num_rdma_ranks() const`

返回 RDMA 组中的 rank 数量

### `get_rdma_rank() const`

返回当前进程在 RDMA 组中的 rank

### `get_root_rdma_rank(bool global) const`

返回 root RDMA rank。
当前实现里它会根据 `global` 参数选择使用哪个 rank 作为 root。

### `get_local_device_id() const`

返回当前 GPU 的 device id

---

## 4. 内存共享句柄（Memory Sharing Handles）

### `get_local_ipc_handle()`

返回当前 rank 的 IPC handle，供其他 rank 建立共享访问

### `get_local_nvshmem_unique_id()`

返回 NVSHMEM 的 unique id。
当前实现要求只有 RDMA root 才能获取这个 id，因为它是全局初始化所必需的。

---

## 5. 缓冲区 & 流访问（Buffer & Stream Access）

### `get_local_buffer_tensor(...)`

从本地 buffer 解析成 PyTorch tensor，支持：

- 指定 dtype
- offset
- 是否使用 RDMA buffer

这个 API 是 Python 层最直接访问底层显存的入口。

### `get_comm_stream()`

返回专用 communication stream，方便后续 kernel 和通信任务做封装。

---

## 6. Core Communication：dispatch / Combine

### `get_dispatch_layout(...)`

根据 `topk_idx` 计算：

- `num_tokens_per_rank`
- `num_tokens_per_rdma_rank`
- `num_tokens_per_expert`
- `is_token_in_rank`

这个函数的本质是：

- 先分析“哪些 token 要送给哪个 rank/expert”
- 再生成后续通信需要的布局信息

---

### `intranode_dispatch(...)`

节点内分发核心函数，基于 NVLink 进行同节点 GPU 间传输。

它做的事情包括：

- 计算/检查分发布局
- 发送 token 数量 metadata
- 等待其他 rank 的接收计数器
- 分配接收 buffer
- 调用底层 `intranode::dispatch` kernel 进行真正传输

---

### `intranode_combine(...)`

节点内聚合核心函数，反向处理 dispatch 的 token。

它用于：

- 结合 `src_idx`、`rank_prefix_matrix`、`channel_prefix_matrix`
- 把从其他 rank 发回来的 token 合并到本地输出
- 执行 reduce / bias / weighting 等逻辑

---

### `internode_dispatch(...)`

跨节点 dispatch。
通常不会直接手工调用，而是由 high-level `dispatch()` 自动选择。

### `internode_combine(...)`

跨节点 combine。
本质和 intranode 版本类似，只是路径改为 RDMA / NVSHMEM。

---

## 7. 低延迟模式（Low-Latency Mode）

这是当前版本最关键的优化方向之一。

### `clean_low_latency_buffer(...)`

用于清理低延迟缓存区，让低延迟 kernel 的 buffer 恢复为“干净状态”，避免 stale data 影响下一轮通信。

### `low_latency_dispatch(...)`

低延迟版 dispatch，特点：

- 走 RDMA / IBGDA 路径
- 适合 latency 优先场景
- 支持 `use_fp8`, `round_scale`, `use_ue8m0`
- 支持 `async_finish`, `return_recv_hook`

### `low_latency_combine(...)`

低延迟版 combine，特点：

- 支持 `zero_copy`
- 支持 `use_logfmt`
- 支持 `async_finish`
- 通过 handle 携带 `src_info` 和 `layout_range`，让 combine 知道“来自哪里、怎么还原”

### `get_next_low_latency_combine_buffer(...)`

返回下一轮 combine 时可直接写入的 raw buffer，支持零拷贝模式。

---

## 8. 一句话概括

当前版本的 `Buffer` 可以概括为：

“一个以 GPU 共享内存 + NVLink + NVSHMEM + IPC 为底层的 EP 通信 runtime，提供 divide / merge / low-latency 语义，并通过 rank/domain metadata + barrier signals + receive counters 实现跨 rank 协同。”

它的核心价值不是“有一个 buffer 对象”，而是：

- 组织全局通信状态
- 统一 NVLink / RDMA / low-latency 路径
- 为 dispatch / combine 提供稳定的底层 runtime

# Buffer 构造函数

## 1. 构造函数的定位

`Buffer` 的构造函数不是简单的“对象初始化”，而是：

- 完成 EP 通信运行时的 bootstrap
- 申请 NVLink / RDMA 共享 buffer
- 生成/同步 IPC 和 NVSHMEM 需要的状态
- 准备 dispatch / combine 所需的元数据和计数器
- 让 `self.runtime` 进入“可用状态”

也就是说，构造函数本质上是在搭建整个低延迟 / EP 通信的底层资源环境。

---

## 2. 当前构造函数签名

当前版本的签名是：

```cpp
Buffer(int rank,
       int num_ranks,
       int64_t num_nvl_bytes,
       int64_t num_rdma_bytes,
       bool low_latency_mode,
       bool explicitly_destroy,
       bool enable_shrink,
       bool use_fabric)
```

和旧版相比，当前版本多了两项：

- `enable_shrink`
- `use_fabric`

同时，`comm_stream` 是通过成员初始化列表直接初始化的：

```cpp
comm_stream(at::cuda::getStreamFromPool(true)),
shared_memory_allocator(use_fabric)
```

这说明当前版本把：

- 通信流
- shared memory allocator
- fabric 开关

都放进了构造的初始化阶段。

---

## 3. 成员初始化列表

```cpp
Buffer(...) : rank(rank),
  num_ranks(num_ranks),
  num_nvl_bytes(num_nvl_bytes),
  num_rdma_bytes(num_rdma_bytes),
  enable_shrink(enable_shrink),
  low_latency_mode(low_latency_mode),
  explicitly_destroy(explicitly_destroy),
  comm_stream(at::cuda::getStreamFromPool(true)),
  shared_memory_allocator(use_fabric) {
```

这里的含义是：

- 直接把构造参数写入成员变量
- 初始化通信 stream
- 初始化 shared memory allocator

注意：这个 allocator 很重要，因为它负责 IPC shared memory 的分配和打开。

所以当前构造函数的初始化顺序非常关键：
先设参数，再准备两大资源基础设施：

- communication stream
- shared memory allocator

---

## 4. 元数据内存计算

```cpp
int64_t barrier_signal_bytes = LEGACY_NUM_MAX_NVL_PEERS * sizeof(int);
int64_t buffer_ptr_bytes = LEGACY_NUM_MAX_NVL_PEERS * sizeof(void*);
int64_t barrier_signal_ptr_bytes = LEGACY_NUM_MAX_NVL_PEERS * sizeof(int*);
```

这些值对应的是：

- `barrier_signal_bytes`
  - barrier 信号所需内存大小
- `buffer_ptr_bytes`
  - 共享 buffer pointer 数组的大小
- `barrier_signal_ptr_bytes`
  - barrier signal pointer 数组的大小

它们被合并到 NVLink buffer 的连续区域中，形如：

```cpp
[num_nvl_bytes | barrier_signal | buffer_ptr_array | barrier_signal_ptr_array]
```

这样做的目的是：

- 在同一块共享内存里放多种元数据
- 让其他 GPU / rank 能通过 pointer array 找到对应位置
- 让 GPU kernel 直接访问 barrier / pointer table，而不必再用额外独立分配

这和旧版说明“计算 barrier signal / ptr 数组大小”是对的，但当前代码里它们被更明确地布局进同一块 buffer 里。

---

## 5. 通用检查：参数合法性

```cpp
EP_STATIC_ASSERT(LEGACY_NUM_BUFFER_ALIGNMENT_BYTES % sizeof(int4) == 0, "Invalid alignment");
EP_HOST_ASSERT(num_nvl_bytes % LEGACY_NUM_BUFFER_ALIGNMENT_BYTES == 0 and
               (num_nvl_bytes <= std::numeric_limits<int>::max() or num_rdma_bytes == 0));
EP_HOST_ASSERT(num_rdma_bytes % LEGACY_NUM_BUFFER_ALIGNMENT_BYTES == 0 and
               (low_latency_mode or num_rdma_bytes <= std::numeric_limits<int>::max()));
EP_HOST_ASSERT(num_nvl_bytes / sizeof(int4) < std::numeric_limits<int>::max());
EP_HOST_ASSERT(num_rdma_bytes / sizeof(int4) < std::numeric_limits<int>::max());
EP_HOST_ASSERT(0 <= rank and rank < num_ranks and (num_ranks <= LEGACY_NUM_MAX_NVL_PEERS * LEGACY_NUM_MAX_RDMA_PEERS or low_latency_mode));
EP_HOST_ASSERT(num_ranks < LEGACY_NUM_MAX_NVL_PEERS or num_ranks % LEGACY_NUM_MAX_NVL_PEERS == 0);
if (num_rdma_bytes > 0)
    EP_HOST_ASSERT(num_ranks > LEGACY_NUM_MAX_NVL_PEERS or low_latency_mode);
```

这是构造函数最重要的“硬约束检查”。

### 逐项解释

#### 1）对齐要求

```cpp
num_nvl_bytes % LEGACY_NUM_BUFFER_ALIGNMENT_BYTES == 0
num_rdma_bytes % LEGACY_NUM_BUFFER_ALIGNMENT_BYTES == 0
```

用途：

- buffer 必须按固定字节对齐
- 否则 kernel 访问 / TMA / vectorized memory copy 会出问题

#### 2）size 上限

```cpp
num_nvl_bytes <= std::numeric_limits<int>::max()
```

说明 buffer 大小最终都要落到某个 int 范围值内，避免溢出。

#### 3）rank 范围

```cpp
0 <= rank and rank < num_ranks
```

确保 rank 在合法区间。

#### 4）总 Rank 与 Topological Assumptions

```cpp
num_ranks <= LEGACY_NUM_MAX_NVL_PEERS * LEGACY_NUM_MAX_RDMA_PEERS or low_latency_mode
```

这说明：

- 只有在低延迟模式下才允许特殊放宽
- 普通情况下，rank 总数必须满足当前硬件拓扑假设

#### 5）rank 能否被 8 整除

```cpp
num_ranks < LEGACY_NUM_MAX_NVL_PEERS or num_ranks % LEGACY_NUM_MAX_NVL_PEERS == 0
```

这和 `rdma_rank = rank / 8` / `nvl_rank = rank % 8` 对应，说明实际 rank 组织依赖 8 个 NVLink peer 一组。

---

## 6. 计算 Rank 维度

```cpp
CUDA_RUNTIME_CHECK(cudaGetDevice(&device_id));
rdma_rank = rank / LEGACY_NUM_MAX_NVL_PEERS, nvl_rank = rank % LEGACY_NUM_MAX_NVL_PEERS;
num_rdma_ranks = std::max(1, num_ranks / LEGACY_NUM_MAX_NVL_PEERS), num_nvl_ranks = std::min(num_ranks, LEGACY_NUM_MAX_NVL_PEERS);
```

这几行是当前版本真正的“rank 分层”。

### 含义

- `rdma_rank`
  - 当前 rank 在 RDMA 组里的位置
- `nvl_rank`
  - 当前 rank 在 NVLink 组里的位置

例如：如果总共 8 rank，则每个 node 内 rank 0~7 都是同一个 RDMA rank group 下的不同 NVLink rank。

### 公式解释

```cpp
rdma_rank = rank / 8
nvl_rank = rank % 8
```

说明：

- 一组 8 个 rank 视作一个 NVLink 组
- 组间再用 RDMA 连接

换句话说，当前设计的拓扑是：

- 先按 8 个 GPU 为一组
- 若更大规模，则再多组 RDMA 互联

---

## 7. 获取设备属性

```cpp
cudaDeviceProp device_prop = {};
CUDA_RUNTIME_CHECK(cudaGetDeviceProperties(&device_prop, device_id));
num_device_sms = device_prop.multiProcessorCount;
```

这里获取当前 GPU 的属性，重点记录：

- `device_id`
- `multiProcessorCount`

也就是说，当前构造函数会把“当前 GPU 上的 SM 数量”记到成员变量里，后面 kernel 运行时会根据这个值决定：

- channel 数
- block 数
- 通信并行度
- 分块大小

所以这一步不是“可有可无”，而是为了后续 kernel 的数值配置做准备。

---

## 8. 进一步约束：channel Bytes 数量不能太大

```cpp
EP_HOST_ASSERT(ceil_div<int64_t>(num_nvl_bytes, num_device_sms / 2) < std::numeric_limits<int>::max());
EP_HOST_ASSERT(ceil_div<int64_t>(num_rdma_bytes, num_device_sms / 2) < std::numeric_limits<int>::max());
```

这里是为了保证：

- 每个 channel 对应的 payload 不会超过 int max
- 否则 kernel 里的偏移和索引值会溢出

这说明当前实现还会把 SM 数与 buffer 规模绑在一起，确保后续 channel 计算合法。

---

## 9. NVLink Buffer 的分配

```cpp
if (num_nvl_bytes > 0) {
    shared_memory_allocator.malloc(&buffer_ptrs[nvl_rank],
                                   num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes + barrier_signal_ptr_bytes);
    shared_memory_allocator.get_mem_handle(&ipc_handles[nvl_rank], buffer_ptrs[nvl_rank]);
    buffer_ptrs_gpu = reinterpret_cast<void**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes);

    barrier_signal_ptrs[nvl_rank] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes);
    barrier_signal_ptrs_gpu =
        reinterpret_cast<int**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes);

    CUDA_RUNTIME_CHECK(cudaMemsetAsync(barrier_signal_ptrs[nvl_rank], 0, barrier_signal_bytes, comm_stream));
}
```

这是构造函数里面最关键的内存布局。

### 这段做了什么

#### 1）分配共享内存

```cpp
shared_memory_allocator.malloc(...)
```

分配的是一个连续的共享内存块，大小是：

- `num_nvl_bytes`
- + barrier signals
- + buffer pointer array
- + barrier signal pointer array

#### 2）生成 IPC Handle

```cpp
shared_memory_allocator.get_mem_handle(...)
```

这个 handle 会让其他 rank 能“打开”这片显存并访问它。

#### 3）设置 buffer_ptrs_gpu

```cpp
buffer_ptrs_gpu = ... + num_nvl_bytes + barrier_signal_bytes
```

这个位置开始存放真正的 void* 数组，用于让 GPU kernel 访问 peer buffer pointer。

#### 4）设置 Barrier Signal Pointer

```cpp
barrier_signal_ptrs[nvl_rank] = ... + num_nvl_bytes
```

也就是说，barrier signal 被直接塞在 buffer 头部之后的区域。

#### 5）初始化 Barrier Memory

```cpp
cudaMemsetAsync(..., 0, barrier_signal_bytes, comm_stream)
```

把 barrier signal 区域清零，这样之后 kernel 能安全用它做同步。

---

## 10. 工作区分配

```cpp
CUDA_RUNTIME_CHECK(cudaMalloc(&workspace, LEGACY_NUM_WORKSPACE_BYTES));
CUDA_RUNTIME_CHECK(cudaMemsetAsync(workspace, 0, LEGACY_NUM_WORKSPACE_BYTES, comm_stream));
```

这里分配的是临时工作区 (workspace)。

### 作用

它不直接存“token 数据”，而是给通信 kernel 用作：

- prefix 计算中间值
- 临时索引/缓冲
- channel 级元数据
- 其他临时 scratch 区域

这个工作区的大小是常量 `LEGACY_NUM_WORKSPACE_BYTES`，当前实现中注释写的是：

```cpp
// Create 32 MiB workspace
```

也就是大约 32 MiB。

---

## 11. 初始化 MoE 接收计数器

```cpp
CUDA_RUNTIME_CHECK(cudaMallocHost(&moe_recv_counter, sizeof(int64_t), cudaHostAllocMapped));
CUDA_RUNTIME_CHECK(cudaHostGetDevicePointer(&moe_recv_counter_mapped, const_cast<int*>(moe_recv_counter), 0));
*moe_recv_counter = -1;
```

这部分很关键，说明当前实现的接收状态并不是只在 GPU 里持有，而是：

- host 端分配一个映射内存
- 让 device 能直接访问它
- 初始化为 -1，代表“还没收到任何数据”

### 为什么是 `volatile int*`

因为它要被 CPU 和 GPU 同时读写，且 CPU 等待时会轮询这个计数器。

---

```cpp
CUDA_RUNTIME_CHECK(cudaMallocHost(&moe_recv_expert_counter, sizeof(int) * LEGACY_NUM_MAX_LOCAL_EXPERTS, cudaHostAllocMapped));
CUDA_RUNTIME_CHECK(cudaHostGetDevicePointer(&moe_recv_expert_counter_mapped, const_cast<int*>(moe_recv_expert_counter), 0));
for (int i = 0; i < LEGACY_NUM_MAX_LOCAL_EXPERTS; ++i)
    moe_recv_expert_counter[i] = -1;
```

这里是“每个本地 expert 的接收计数器”。

它的作用是：

- 每个 expert 在 dispatch 完成之前，都先置为 -1
- 之后其他 rank 会写入它
- CPU 可以轮询它，判断专家接收是否完成

---

```cpp
if (num_rdma_ranks > 0) {
    CUDA_RUNTIME_CHECK(cudaMallocHost(&moe_recv_rdma_counter, sizeof(int), cudaHostAllocMapped));
    CUDA_RUNTIME_CHECK(cudaHostGetDevicePointer(&moe_recv_rdma_counter_mapped, const_cast<int*>(moe_recv_rdma_counter), 0));
    *moe_recv_rdma_counter = -1;
}
```

RDMA 级计数器同理，用于跨节点通信同步状态。

---

## 12. 总结：当前版构造函数的真实职责

可以把当前版本的构造函数总结成下面这段更准确的话：

> `Buffer` 构造函数不是在做“纯内存分配”，而是在初始化一套可用于 EP 通信的 runtime。它依次完成：参数校验、rank 拓扑计算、GPU 设备属性获取、NVLink 共享区分配、IPC 句柄生成、Workspace 初始化、MoE 接收计数器准备，以及为后续 `dispatch` / `combine` / low_latency 通信预留同步状态。真正使其可用的关键动作是 `sync()`，而构造函数只是把这些基础资源和系统状态搭起来。

# Sync

这个函数位于 buffer.hpp。它的职责是：

- 连接同一节点内各 rank 的 IPC 共享内存
- 建立跨节点 / 跨 RDMA rank 的 NVSHMEM 共享缓冲区
- 初始化 barrier 相关指针
- 最后把 `available` 标记为 true，表示 Buffer 已经可以投入后续 dispatch/combine 逻辑

---

## 1. 函数入口与前置校验

```cpp
void sync(const std::vector<int>& device_ids,
          const std::vector<std::optional<pybind11::bytearray>>& all_gathered_handles,
          const std::optional<pybind11::bytearray>& root_unique_id_opt) {
    EP_HOST_ASSERT(not is_available());
```

这几行说明：

- 这是一个“初始化/绑定”阶段，不允许重复调用。
- `not is_available()` 表示当前 Buffer 还没准备好，否则会有重复初始化风险。
- `available` 是成员变量，最终在函数最后设为 true。

这就是对“只初始化一次”的保护。

---

## 2. NVLink / IPC 同步分支

```cpp
if (num_nvl_bytes > 0) {
```

只有当这个 Buffer 需要本地 NVLink 通信时才进入此分支。也就是 `num_nvl_bytes > 0` 表示存在同节点内的共享缓冲区。

### 2.1 参数一致性校验

```cpp
EP_HOST_ASSERT(num_ranks == device_ids.size());
EP_HOST_ASSERT(device_ids.size() == all_gathered_handles.size());
```

这里检查三件事：

- `num_ranks`：总 rank 数
- `device_ids`：各 rank 对应的 GPU device id
- `all_gathered_handles`：每个 rank 的 IPC handle 集合

这保证了“每个 rank 的 peer 信息都齐了”，否则后面拿 handle 时会出问题。

---

### 2.2 遍历同一 RDMA 域下的 peer

```cpp
for (int i = 0, offset = rdma_rank * num_nvl_ranks; i < num_nvl_ranks; ++i) {
```

这里 `rdma_rank` 和 `num_nvl_ranks` 决定了当前 rank 所属的“本地组”范围。

- `offset = rdma_rank * num_nvl_ranks`
- `offset + i` 控制当前 RDMA 组内的 rank 索引
- `i` 从 0 到 `num_nvl_ranks - 1`，就是说当前组内所有 peer 都被遍历

也就是：同一个 RDMA 域内，多个 NVLink rank 之间互相建立 IPC 连接。

---

### 2.3 断言 handle 一定存在

```cpp
EP_HOST_ASSERT(all_gathered_handles[offset + i].has_value());
```

此处说明所有同组 rank 的 IPC handle 都应该已经全部收集回来。如果缺失，就说明在前置 all_gather 阶段出了问题。

---

### 2.4 把 bytearray 转成 handle

```cpp
auto handle_str = std::string(all_gathered_handles[offset + i].value());
EP_HOST_ASSERT(handle_str.size() == sizeof(shared_memory::MemHandle));
```

`shared_memory::MemHandle` 是一个结构体，表示共享内存的句柄。这里做了两件事：

- 从 Python 侧传回来的 `bytearray` 转成 `std::string`
- 校验长度确实等于 `sizeof(shared_memory::MemHandle)`

这是一个典型的“跨 Python/C++ 边界的内存句柄恢复”动作。

---

### 2.5 自己 rank 和远端 rank 分别处理

```cpp
if (offset + i != rank) {
    std::memcpy(&ipc_handles[i], handle_str.c_str(), sizeof(shared_memory::MemHandle));
    shared_memory_allocator.open_mem_handle(&buffer_ptrs[i], &ipc_handles[i]);
    barrier_signal_ptrs[i] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[i]) + num_nvl_bytes);
} else {
    EP_HOST_ASSERT(std::memcmp(&ipc_handles[i], handle_str.c_str(), sizeof(shared_memory::MemHandle)) == 0);
}
```

这里分两种情况：

1. 远端 rank
   - `std::memcpy`：把远端的 IPC handle 拷贝进 `ipc_handles[i]`
   - `open_mem_handle`：把远端共享内存映射到当前进程
   - `buffer_ptrs[i]` 成为该 peer 的本地虚拟地址
   - `barrier_signal_ptrs[i]` 计算为：
     `buffer_ptrs[i] + num_nvl_bytes`
     这说明共享缓冲区布局是：
     - 前 `num_nvl_bytes`：真正的数据缓冲区
     - 后面部分：barrier signal 区域

2. 自己 rank
   - 直接检查本地 handle 和收到的 handle 是不是一致
   - 这是为了确保同一个 shared memory 对象在各个 rank 上做了同一份标识

也就是说，这段代码就完成了：
- “本地 + 远端共享内存的句柄对齐”
- “远端地址映射”
- “barrier 信号区指针准备”

---

### 2.6 把所有 peer 指针拷到 GPU

```cpp
CUDA_RUNTIME_CHECK(cudaMemcpy(buffer_ptrs_gpu, buffer_ptrs, sizeof(void*) * LEGACY_NUM_MAX_NVL_PEERS, cudaMemcpyHostToDevice));
CUDA_RUNTIME_CHECK(cudaMemcpy(barrier_signal_ptrs_gpu, barrier_signal_ptrs, sizeof(int*) * LEGACY_NUM_MAX_NVL_PEERS, cudaMemcpyHostToDevice));
CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
```

这三行非常关键：

- `buffer_ptrs_gpu`：GPU 侧的全局指针数组，用于后续 kernel 访问其他 rank 的 buffer
- `barrier_signal_ptrs_gpu`：GPU 侧的 barrier signal 指针数组
- `cudaMemcpy`：把 host 侧地址数组复制到 device 侧

这里 `LEGACY_NUM_MAX_NVL_PEERS` 是最大 NVLink peer 数，通常是 8。  
所以这里相当于把“各 rank 通信所需的地址表”交给 GPU kernels。

最后的 `cudaDeviceSynchronize()` 是为了确保这两个复制已经完全完成，后续代码不会读到未完成的 GPU 指针数组。

---

## 3. NVSHMEM 初始化和 RDMA 缓冲区分支

```cpp
if (num_rdma_bytes > 0) {
```

这里说明还有跨节点 / 跨 RDMA rank 的共享内存需要准备。

---

### 3.1 读取 root unique ID

```cpp
EP_HOST_ASSERT(root_unique_id_opt.has_value());
std::vector<uint8_t> root_unique_id(root_unique_id_opt->size());
auto root_unique_id_str = root_unique_id_opt->cast<std::string>();
std::memcpy(root_unique_id.data(), root_unique_id_str.c_str(), root_unique_id_opt->size());
```

`root_unique_id_opt` 是 NVSHMEM 所需的 root unique id。它通常由 rank 0 生成并广播。  
这里：

- 先检查它存在
- 把 Python 的 `bytearray` 转成 `std::vector<uint8_t>`
- 再复制到 `root_unique_id`

这是标准的“跨语言对象转字节串”操作。

---

### 3.2 计算 NVSHMEM rank / size

```cpp
auto nvshmem_rank = low_latency_mode ? rank : rdma_rank;
auto num_nvshmem_ranks = low_latency_mode ? num_ranks : num_rdma_ranks;
EP_HOST_ASSERT(nvshmem_rank == nvshmem::init(root_unique_id, nvshmem_rank, num_nvshmem_ranks,
                                             low_latency_mode ? LEGACY_NUM_MAX_NVL_PEERS : 0));
```

这里非常关键：

- `low_latency_mode` 下，NVSHMEM rank 可能直接用全局 `rank`
- 非低延迟模式下，使用 `rdma_rank`
- `num_nvshmem_ranks` 也相应变化

`nvshmem::init(...)` 返回当前 rank 在 NVSHMEM 组中的索引，必须等于我们计算出来的 `nvshmem_rank`。  
这保证了每个进程都在正确的逻辑分组内。

这一步的意义是：  
“让所有进程在同一个 NVSHMEM 会话中注册自己，并知道自己的 rank / total size。”

---

### 3.3 分配 RDMA buffer

```cpp
rdma_buffer_ptr = nvshmem::alloc(num_rdma_bytes, LEGACY_NUM_BUFFER_ALIGNMENT_BYTES);
```

这里分配的是跨节点可访问的共享内存。  
它是后续 internode dispatch/combine 所使用的主 buffer。

- `num_rdma_bytes`：每个 rank 需要的总 RDMA buffer 大小
- alignment：保证对齐，满足 GPU/NVSHMEM 的要求

---

### 3.4 清零缓冲区

```cpp
CUDA_RUNTIME_CHECK(cudaMemset(rdma_buffer_ptr, 0, num_rdma_bytes));
```

清零 RDMA buffer，减少脏数据干扰。  
特别是 low latency 模式下，这很重要，因为它可能复用相同的 buffer 区域。

---

### 3.5 若启用 shrink，则额外分配 mask/sync buffer

```cpp
if (enable_shrink) {
    int num_mask_buffer_bytes = num_ranks * sizeof(int);
    int num_sync_buffer_bytes = num_ranks * sizeof(int);
    mask_buffer_ptr = static_cast<int*>(nvshmem::alloc(num_mask_buffer_bytes, LEGACY_NUM_BUFFER_ALIGNMENT_BYTES));
    sync_buffer_ptr = static_cast<int*>(nvshmem::alloc(num_sync_buffer_bytes, LEGACY_NUM_BUFFER_ALIGNMENT_BYTES));
    CUDA_RUNTIME_CHECK(cudaMemset(mask_buffer_ptr, 0, num_mask_buffer_bytes));
    CUDA_RUNTIME_CHECK(cudaMemset(sync_buffer_ptr, 0, num_sync_buffer_bytes));
}
```

这里额外分配两个共享数组：

- `mask_buffer_ptr`
- `sync_buffer_ptr`

它们通常用于 shrink / tracking / synchronization 的状态控制。  
这说明 DeepEP 在 internode path 里不只需要数据 buffer，还需要控制状态 buffer。

---

### 3.6 公共 barrier

```cpp
nvshmem::barrier(true);
CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
```

这一步非常重要：  
它确保所有 rank 都已经分配完 NVSHMEM buffer，并且已完成清零。

- `nvshmem::barrier(true)`：跨 rank 同步
- `cudaDeviceSynchronize()`：CPU 等待 GPU 侧 memset 完成，保证后面的使用不会出现读到未初始化数据的情况

---

## 4. 最终设置为 ready

```cpp
available = true;
```

这里是整个 `sync` 的最终状态。  
一旦设置为 true，后续所有 `dispatch` / `combine` / `destroy` 等函数都可以假定：

- 本地 NVLink IPC buffer 可用
- RDMA/NVSHMEM buffer 可用
- barrier 信号区已准备好
- 后续 kernels 可以安全访问这些共享资源

---

## 5. 整体语义总结

这个 `sync` 函数本质上做的是“Buffer 初始化的收尾工作”，它把下面这些东西连接起来：

- IPC shared memory: 同节点不同 rank 之间的数据共享
- NVSHMEM: 跨节点或跨 RDMA group 的全局共享内存
- GPU pointer arrays: 给 kernel 直接访问 remote buffer
- barrier signal pointers: 给内部分组 barrier 使用
- `available` 标志: 表示 buffer 已经完全 ready

可以把它理解为：

> “所有通信基础设施都已经在各个 rank 间建立完毕，并准备好后续高性能 dispatch/combine 工作。”

---

## 6. 关键设计点

1. `EP_HOST_ASSERT(not is_available())`
   - 防止重复初始化

2. `all_gathered_handles[offset + i]`
   - 说明这里依赖先前的全局 all-gather，拿到所有 peer 的 IPC handle

3. `buffer_ptrs_gpu` / `barrier_signal_ptrs_gpu`
   - 把 host 端地址表复制给 GPU，供 kernel 直接访问

4. `nvshmem::init(...)`
   - 这是跨节点共享内存的逻辑入口

5. `available = true`
   - 最终判定该 buffer 已经可用于实际通信

如果你愿意，我下一步可以继续按“从函数的调用链角度”分析：  
谁调用了 `sync()`，它在整个 DeepEP 初始化流程中担任什么角色。

# get_dispatch_layout

# intranode_dispatch

```cpp

// csrc\deep_ep.cpp

@@ -305, 235

```

`intranode_dispatch` 是 **节点内 token 分发核心函数**，负责通过 NVLink 在同一节点内的多个 GPU 之间高效分发输入 token 数据，支持缓存模式（复用布局信息）和非缓存模式（动态计算布局），并处理 Top-k 选择、FP8 缩放等特性，最终返回接收方的 token 数据及元信息。

## **关键参数与前置检查**

```cpp

// csrc\deep_ep.cpp

@@ -312, 39

```

- **核心输入**：
- `x`：输入 token 数据张量（形状 `[num_tokens, hidden]`）；
- `topk_idx`/`topk_weights`：Top-k 专家索引及权重（可选，MoE 场景使用）；
- `num_tokens_per_rank`/`num_tokens_per_expert`：每个 rank/专家的 token 数量（非缓存模式必需）；
- `cached_rank_prefix_matrix`/`cached_channel_prefix_matrix`：缓存的布局前缀矩阵（缓存模式必需）；
- `config`：通信配置（含 SM 数量、分块大小等）。
- **前置检查**：
- **硬件兼容性**：`config.num_sms`（流多处理器数量）必须为偶数（因每个通信通道占用 2 个 SM 块）；
- **数据合法性**：输入张量需连续（`is_contiguous()`），`x` 的隐藏维度大小需为 `int4` 倍数（内存对齐要求）；
- **模式一致性**：缓存模式下必须提供缓存的前缀矩阵，非缓存模式下必须提供 token 数量统计。

## 数据、通信准备

```cpp

// csrc\deep_ep.cpp

@@ -353, 53

```

### **1. 模式区分与通道初始化**

- **缓存模式（`cached_mode = true`）**：复用已计算的 `cached_rank_prefix_matrix`（rank 级前缀和）和 `cached_channel_prefix_matrix`（通道级前缀和），跳过布局计算，直接基于缓存信息分发。
- **非缓存模式（`cached_mode = false`）**：需动态计算分发布局，依赖 `num_tokens_per_rank`（每个 rank 的发送 token 数）和 `num_tokens_per_expert`（每个专家的接收 token 数）。
- **通道数量**：`num_channels = config.num_sms / 2`（每个通道对应 2 个 SM 块，分别用于发送和接收）。

### **2. 元数据准备与同步**

- **Top-k 与 FP8 处理**：若提供 `topk_idx`/`topk_weights`，则获取其设备指针，用于分发时同步专家索引和权重；若输入为 FP8 格式（`x_scales` 存在），则获取缩放因子指针及步长信息。
- **流管理**：若 `allocate_on_comm_stream` 为真，切换到专用通信流（`comm_stream`）分配张量，避免阻塞计算流；通过 `previous_event` 等待前置任务完成（或直接等待计算流）。

# 分发通知与接收计数（非缓存模式关键）

```cpp

// csrc\deep_ep.cpp

@@ -408, 61

```

- **发送元数据**：调用 `intranode::notify_dispatch` 向节点内其他 rank 发送 token 数量等元信息，通过共享内存（`buffer_ptrs_gpu`）和屏障信号（`barrier_signal_ptrs_gpu`）同步。
- **接收计数等待**：CPU 忙等待 GPU 接收其他 rank 的元数据，通过 `moe_recv_counter`（总接收 token 数）和 `moe_recv_expert_counter`（专家级接收 token 数）判断是否就绪，超时则抛出异常。

> 执行的 kernel 详细见 `kernel.intranode::notify_dispatch` `kernel.intra_node::dispatch`

## 内存分配与内核调用

- **接收张量分配**：根据接收 token 数（`num_recv_tokens`）分配 `recv_x`（接收数据）、`recv_src_idx`（源 token 索引）、`recv_topk_idx`（接收的 Top-k 索引，若使用）等张量。
- **分发内核启动**：调用 `intranode::dispatch` 内核，通过 NVLink 传输数据：
- 输入数据（`x`）、缩放因子（`x_scales`）、Top-k 信息（`topk_idx`/`topk_weights`）从发送方拷贝到接收方；
- 使用分块传输（`config.num_max_nvl_chunked_send_tokens`/`recv_tokens` 控制块大小），避免大内存连续访问瓶颈。

# 流同步与结果返回

```cpp

// csrc\deep_ep.cpp

@@ -512, 23

```

- **异步处理**：若 `async = true`，记录通信流事件（`EventHandle`），并将张量关联到通信流和计算流，实现异步执行；否则等待计算流完成。
- **返回结果**：返回接收数据（`recv_x`）、元信息（前缀矩阵、源索引等）及同步事件，供后续聚合阶段（`intranode_combine`）使用。

# **关键特性与优化**

- **缓存复用**：缓存模式下跳过布局计算和元数据同步，直接使用历史前缀矩阵，降低通信开销。
- **分块传输**：通过 `num_max_nvl_chunked_send/recv_tokens` 控制分块大小，平衡带宽利用率和延迟。
- **流隔离**：使用专用通信流（`comm_stream`）分离通信与计算任务，避免相互阻塞，提升并行效率。
- **严格检查**：通过 `EP_HOST_ASSERT` 确保输入合法性（如张量连续性、内存对齐、参数一致性），提前暴露错误。

# Kernel

---

# `intranode::notify_dispatch`

`notify_dispatch` 是 **节点内分发通知与同步的核心主机函数**，负责启动 CUDA 内核以完成以下任务：

1. 跨 rank 同步 token 分发元信息（如每个 rank/expert 的 token 数量）；
2. 计算分发布局（rank 级和通道级前缀和矩阵）；
3. 初始化通信缓冲区（如清零信号量和队列）。

## **参数详解**

（按功能分组，`const` 标识输入参数，非 `const` 指针多为输出/双向参数）

### **1. 分发元信息（输入）**

| 参数名 | 类型 | 作用 |

| ------------------------- | --------------- | -------------------------------------------------------------------------------------------------------------------- |

| `num_tokens_per_rank` | `const int*` | 长度为 `num_ranks` 的数组，存储**当前 rank 发送给每个目标 rank 的 token 数量**（如 `num_tokens_per_rank[i]` 表示发送给 rank `i` 的 token 数）。|

| `num_tokens_per_expert` | `const int*` | 长度为 `num_experts` 的数组，存储**每个 expert 接收的 token 数量**（全局视角，需跨 rank 同步）。|

| `num_ranks` | `int` | 节点内总 rank 数（如 8 表示 8 个 GPU）。|

| `num_experts` | `int` | 全局 expert 总数（需满足 `num_experts % num_ranks == 0`，确保每个 rank 分配到整数个 expert）。|

| `num_tokens` | `int` | 当前 rank 待分发的总 token 数。|

| `is_token_in_rank` | `const bool*` | 2D 数组（形状 `[num_tokens, num_ranks]`），`is_token_in_rank[token_idx][rank] = true` 表示第 `token_idx` 个 token 需要发送给 `rank`。|

### **2. 输出布局矩阵（输出）**

| 参数名 | 类型 | 作用 |

| --------------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |

| `channel_prefix_matrix` | `int*` | 2D 数组（形状 `[num_ranks, num_channels]`），存储**通道级前缀和**：`channel_prefix_matrix[rank][channel]` 表示 rank `rank` 的第 `channel` 个通道累计处理的 token 数，用于划分通道任务范围。|

| `rank_prefix_matrix_copy` | `int*` | 2D 数组（形状 `[num_ranks, num_ranks]`），复制**rank 级前缀和矩阵**：`rank_prefix_matrix_copy[i][j]` 表示 rank `i` 发送给 rank `j` 的 token 累计数量，用于后续数据分发偏移计算。|

### **3. 同步与缓冲区（双向）**

| 参数名 | 类型 | 作用 |

| ---------------------------------- | ---------- | ------------------------------------------------------------------------------------------------ |

| `moe_recv_counter_mapped` | `int*` | 指向主机映射的设备内存，存储**当前 rank 接收的总 token 数**（跨 rank 同步后更新）。|

| `moe_recv_expert_counter_mapped` | `int*` | 指向主机映射的设备内存，长度为 `num_experts/num_ranks`，存储**当前 rank 每个 local expert 接收的 token 数**（跨 rank 同步后更新）。|

| `buffer_ptrs` | `void**` | 数组（长度 `num_ranks`），存储**跨 rank 共享缓冲区指针**（用于 NVLink 通信的设备内存）。|

| `barrier_signal_ptrs` | `int**` | 数组（长度 `num_ranks`），存储**屏障信号指针**（用于跨 rank 同步，如等待所有 rank 完成元信息发送）。|

### **4. 配置参数（输入）**

| 参数名 | 类型 | 作用 |

| -------------------- | ---------------- | ------------------------------------------ |

| `num_memset_int` | `int` | 需要清零的整数数量，用于初始化通信队列（如通道级缓冲区的信号量）。|

| `expert_alignment` | `int` | expert 接收 token 数的对齐值（如 16），确保内存访问对齐，提升效率。|

| `rank` | `int` | 当前 rank 的 ID（0 基）。|

| `stream` | `cudaStream_t` | 内核启动的 CUDA 流，用于异步执行，避免阻塞计算流。|

| `num_channels` | `int` | 通信通道数（每个通道由 1 个 warp 处理，通常与 SM 数量相关）。|

## 条件配置

函数通过宏定义和模板特化实现对不同 `num_ranks`（rank 数量）的适配，核心语法元素如下：

- `#define NOTIFY_DISPATCH_LAUNCH_CASE(ranks)`：定义内核启动模板，根据 `ranks`（模板参数，实际为 `num_ranks`）实例化 `notify_dispatch<ranks>` 模板内核，并传递参数。
- `#undef NOTIFY_DISPATCH_LAUNCH_CASE` 仅用于**取消宏定义**，避免宏污染后续代码。
- `SWITCH_RANKS(NOTIFY_DISPATCH_LAUNCH_CASE)`：根据 `num_ranks` 切换到对应模板实例（如 `num_ranks=4` 时调用 `notify_dispatch<4>`），通过宏展开为 `switch-case` 语句实现。
- `SETUP_LAUNCH_CONFIG(1 + num_ranks, kNumThreads, stream)`

设置内核启动配置：

- **网格大小**：`1 + num_ranks`（1 个块用于全局同步，`num_ranks` 个块用于通道级计算）；
- **块大小**：`kNumThreads=128`（每个块 128 线程）；
- **流**：`stream`（指定 CUDA 流，避免阻塞默认流）。

# Launch Kernel

- `#define NOTIFY_DISPATCH_LAUNCH_CASE(ranks)`：封装 CUDA 内核启动逻辑，传入内核函数指针、参数和配置（`cfg`）。

# `intranode::dispatch`

```cpp

// csrc\kernels\[intranode.cu](http://intranode.cu/)

@@ -475, 34

```

DeepEP 中的 `intranode dispatch` 函数是其核心通信内核之一，专为节点内（同一服务器内 GPU 间）的混合专家（MoE）模型设计。以下是对该函数及其代码的详细解释：

---

## **1. 函数目的**

`intranode dispatch` 负责在**同一节点内的多个 GPU 之间**高效分发数据（如 MoE 模型中的专家输入）。其目标是：

- **高吞吐量**：利用 NVLink 实现节点内 GPU 的高速互联（最高 158 GB/s）。
- **动态负载均衡**：根据每个专家的令牌分布（`num_tokens_per_expert`）动态调整数据分区。
- **低精度优化**：支持 FP8/BF16 数据格式，减少通信开销（显存占用降低 50%）。
- **计算 - 通信重叠**：通过 CUDA 事件和 Hook 机制，最大化 GPU 利用率。

---

## **2. 代码结构解析**

### **(1) 参数说明**

```cpp

void dispatch(

void* recv_x, float* recv_x_scales, int* recv_src_idx, int64_t* recv_topk_idx, float* recv_topk_weights, int* recv_channel_offset,

int* send_head, const void* x, const float* x_scales, const int64_t* topk_idx, const float* topk_weights,

const bool* is_token_in_rank, const int* channel_prefix_matrix,

int num_tokens, int num_worst_tokens, int hidden_int4, int num_topk, int num_experts, int num_scales,

int scale_token_stride, int scale_hidden_stride,

void** buffer_ptrs, int rank, int num_ranks,

cudaStream_t stream, int num_sms, int num_max_send_tokens, int num_recv_buffer_tokens

)

```

- **输入输出数据**：
- `x`: 当前 GPU 的输入张量（可能为 FP8/BF16）。
- `recv_x`: 接收其他 GPU 发送的数据。
- `topk_idx/weights`: 每个 token 选择的专家索引及权重（MoE 门控结果）。
- **元数据**：
- `num_tokens`: 当前批次的总 token 数。
- `num_experts`: 专家数量（如 Top-8 Experts）。
- `num_ranks`: 当前节点内的 GPU 数量。
- **优化参数**：
- `hidden_int4`: 隐藏层维度（可能为 FP8 量化后的 4 位整数）。
- `num_sms`: 使用的流多处理器（SM）数量，控制并行度。
- `buffer_ptrs`: 缓冲区指针，用于存储中间数据。

### **(2) 硬件优化**

- **线程配置**：

```cpp

constexpr int kNumThreads = 768; // 每个线程块使用768线程

constexpr int kNumTMABytesPerWarp = 8192; // 每Warp使用8KB TMA（Tensor Memory Acceleration）

```

- **TMA（Tensor Memory Acceleration）**：NVIDIA Hopper 架构特性，通过预取和批量传输优化内存带宽。
- **线程分配**：每个线程块 768 线程，覆盖 32 个 Warp（768/24=32），充分利用 Hopper 的 SM 资源。
- **共享内存**：

```cpp

#ifndef DISABLE_SM90_FEATURES

constexpr int smem_size = kNumTMABytesPerWarp * (kNumThreads / 32);

#endif

```

- 共享内存大小根据 TMA 和线程数动态计算，确保每个 Warp 有足够的空间存储 TMA 元数据。

### **(3) 内核启动逻辑**

- **宏定义**：

```cpp

#define DISPATCH_LAUNCH_CASE(ranks) { \

auto kernel = dispatch<ranks, kNumThreads, kNumTMABytesPerWarp>; \

SET_SHARED_MEMORY_FOR_TMA(kernel); \

LAUNCH_KERNEL(&cfg, kernel, …); \

} break

```

- **模板实例化**：`dispatch<ranks, …>` 为不同 `ranks`（GPU 数量）生成专用内核。
- **动态配置**：`SWITCH_RANKS(DISPATCH_LAUNCH_CASE)` 根据 `num_ranks` 选择对应内核版本。
- **负载均衡**：

```cpp

EP_HOST_ASSERT(num_sms % 2 == 0); // SM数必须为偶数

```

- **发送/接收分离**：偶数 SM 用于发送数据，奇数 SM 用于接收数据，避免资源冲突。

---

## **3. 技术亮点**

### **(1) NVLink 优化**

- **节点内通信**：通过 NVLink 实现 GPU 间直接内存访问（DMA），带宽高达 160 GB/s（接近硬件极限）。
- **共享内存**：使用 `ipc_handles` 和 `dist.all_gather_object` 减少跨 GPU 同步开销。

### **(2) 低精度计算**

- **FP8 量化**：
- 输入数据 `x` 和输出 `recv_x` 支持 FP8（`float8_e4m3fn`）。
- 通过 `x_scales` 和 `recv_x_scales` 存储动态缩放因子，实现混合精度通信。
- **优势**：显存占用减少 50%，通信带宽需求降低。

### **(3) 计算 - 通信重叠**

- **CUDA 事件**：通过 `torch.cuda.Event` 和 `EventOverlap` 类管理通信与计算的异步执行。
- **Hook 机制**：在前向传播时预加载数据，在反向传播时异步传输梯度，不占用 SM 资源。

### **(4) 动态负载均衡**

- **专家令牌统计**：
- `get_dispatch_layout` 内核统计 `num_tokens_per_expert` 和 `num_tokens_per_rank`。
- 动态调整 `send_head` 和 `recv_topk_idx`，避免令牌分布不均导致的带宽浪费。

---

## **4. 性能表现**

- **节点内吞吐量**：153-158 GB/s（NVLink 带宽）。
- **延迟**：通过 `low_latency_dispatch` 实现<163 微秒延迟（纯 RDMA 模式）。
- **扩展性**：在 2048 卡 H800 集群中，训练吞吐量提升 3.8 倍，推理延迟降低 80%。

---

## **5. 应用场景**

- **MoE 模型训练**：适用于 DeepSeek-V3 等大规模模型的分布式训练。
- **低延迟推理**：支持实时推理解码（如聊天机器人），延迟敏感场景优先使用纯 RDMA 内核。
- **异构网络优化**：自动适配 NVLink→RDMA 的非对称带宽转发，避免跨节点通信瓶颈。

---

## **总结**

`intranode dispatch` 是 DeepEP 的核心组件，通过**硬件感知优化**（NVLink/TMA）、**低精度计算**（FP8）和**动态负载均衡**，实现了节点内 GPU 的高效通信。其设计充分结合了 Hopper 架构的特性，为 MoE 模型提供了接近硬件极限的性能表现。

# Kernel

# Combine

# Low Lantency
