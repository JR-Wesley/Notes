---
dateCreated: 2025-08-08
dateModified: 2025-08-08
---


下面就开始讲解 C++ 提供的接口里对 Buffer 的实现。

# Cmake

首先根据编译确定接口用到了哪些依赖，实现的功能是什么。

#### **一、核心功能：构建 PyTorch 扩展模块 `deep_ep_cpp`**

该文件是用于编译 `deep_ep_cpp` 扩展模块的 CMake 配置脚本，核心目标是将 C++/CUDA 代码（含分布式通信逻辑）封装为 Python 可调用的 `deep_ep_cpp` 模块，支持高性能 GPU 通信（如 NVLink/RDMA）和 MoE（Mixture of Experts）模型的 token 分发/聚合。

#### **二、关键编译配置**

1. **优化与兼容性设置**

- 启用最高级优化（`-O3`）和位置无关代码（`-fPIC`），提升运行效率并支持动态链接。
- 开启 CUDA 分离编译（`CUDA_SEPARABLE_COMPILATION ON`），允许 CUDA 代码独立编译为设备端模块，减少整体编译时间。
- NVCC 编译参数：
- `-DENABLE_FAST_DEBUG`：启用快速调试模式（可能关闭部分安全检查以加速）。
- `--ptxas-options`：控制 PTX 汇编器行为，包括输出详细信息（`--verbose`）、寄存器使用级别（`--register-usage-level=10`）和本地内存使用警告（`--warn-on-local-memory-usage`），优化 GPU 内核资源占用。

1. **GPU 架构支持**

- `CUDA_ARCH_LIST "9.0"`：指定编译支持 NVIDIA Hopper 架构（如 H100 GPU），确保生成适配 SM90 的优化代码。
- `TORCH_CUDA_ARCH_LIST`：同步 PyTorch 的 CUDA 架构列表，避免兼容性问题。

1. **代码标准**

- 设置 C++ 和 CUDA 标准为 C++17（`CMAKE_CXX_STANDARD 17`、`CMAKE_CUDA_STANDARD 17`），支持现代 C++ 特性（如结构化绑定、折叠表达式）。

#### **三、依赖项说明**

通过 `find_package` 和路径配置引入以下核心依赖，确保编译和运行时能正确链接库文件和头文件：

| 依赖项 | 作用 | |

| ----------------- | ---------------------------------------------------------------------------------- | --- |

| **CUDAToolkit** | 提供 CUDA 运行时库、NVCC 编译器及 GPU 通信 API（如 CUDA IPC、NVLink），是 GPU 加速的基础。| |

| **pybind11** | 实现 C++ 到 Python 的绑定，将 C++ 类/函数封装为 Python 可调用的模块（如 `Buffer` 类的方法）。| |

| **Torch** | PyTorch 库，提供张量操作、CUDA 流管理、自动求导等核心功能，扩展模块需与 PyTorch 类型系统兼容。| |

| **NVSHMEM** | NVIDIA 分布式共享内存库，支持跨节点（RDMA）的 GPU 内存直接访问，是 `internode_dispatch`/`combine` 跨节点通信的基础。| |

#### **四、构建流程关键步骤**

1. **依赖定位与路径配置**

- `include_directories`：添加 CUDA、PyTorch、Python、NVSHMEM 的头文件路径，确保编译器能找到 `torch/tensor.h`、`nvshmem.h` 等关键头文件。
- `link_directories`：指定 Torch、CUDA、NVSHMEM 的库文件路径，确保链接器能找到 `libtorch.so`、`libnvshmem.so` 等动态库。

1. **子模块编译**

- `add_subdirectory(kernels)`：编译 `kernels` 子目录下的 CUDA 内核代码（如 `intranode_dispatch`/`combine` 的底层 GPU 核函数），生成静态库供主模块链接。

1. **扩展模块生成**

- `pybind11_add_module(deep_ep_cpp deep_ep.cpp)`：将 `deep_ep.cpp`（含 `Buffer` 类的 C++ 实现和 pybind11 绑定代码）编译为 Python 扩展模块 `deep_ep_cpp`。
- `target_link_libraries`：链接依赖库，包括 CUDA 运行时库、PyTorch 库（`${TORCH_LIBRARIES}`）和 Python 绑定库（`torch_python`），确保模块能被 Python 导入并调用 GPU 功能。

# Buffer 成员

```cpp

// csrc\deep_ep.hpp

@@ -25, 137

```

### 成员变量

```cpp

EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8, "The number of maximum NVLink peers must be 8");

```

这是一个静态断言，确保最大 NVLink 对等体的数量为 8。

#### **1. 缓冲区管理（Buffer Management）**

负责不同通信模式（NVLink/RDMA/低延迟）的内存缓冲区分配与维护：

- `int low_latency_buffer_idx = 0;`：低延迟模式下的缓冲区索引，用于多轮通信时的缓冲区切换与复用。
- `bool low_latency_mode = false;`：标记是否启用低延迟模式，决定缓冲区分配策略。
- `int64_t num_nvl_bytes;`：NVLink 通信缓冲区大小（字节），用于节点内高吞吐量通信。
- `void* buffer_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};`：节点内 NVLink 通信的本地缓冲区指针数组（最多支持 `NUM_MAX_NVL_PEERS` 个对等体）。
- `void** buffer_ptrs_gpu = nullptr;`：设备端可见的 NVLink 缓冲区指针数组，供 GPU 直接访问远程缓冲区。
- `int64_t num_rdma_bytes;`：RDMA 通信缓冲区大小（字节），用于节点间或低延迟模式通信。
- `void* rdma_buffer_ptr = nullptr;`：RDMA 通信的共享内存指针（基于 NVSHMEM 实现跨节点内存共享）。

#### **2. 设备与通信标识（Device & Communication Identification）**

记录设备硬件信息及分布式通信中的身份标识，确保跨设备/节点通信的正确性：

- `int device_id;`：当前设备的 GPU 编号，用于设备亲和性控制。
- `int num_device_sms;`：当前 GPU 的 SM（流多处理器）数量，用于内核性能调优（如线程块分配）。
- `int rank, rdma_rank, nvl_rank;`：全局 _rank_、RDMA 分组内 _rank_、NVLink 分组内 _rank_，标识当前进程在不同通信域中的位置。
- `int num_ranks, num_rdma_ranks, num_nvl_ranks;`：全局 _rank_ 总数、RDMA 分组内 _rank_ 总数、NVLink 分组内 _rank_ 总数，用于通信范围控制。
- `cudaIpcMemHandle_t ipc_handles[NUM_MAX_NVL_PEERS];`：CUDA IPC 内存句柄数组，用于节点内不同进程间共享 GPU 内存。

#### **3. 跨进程/设备通信（Inter-process/Device Communication）**

通过 IPC、NVSHMEM 等机制实现跨进程/GPU 的数据传输与同步：

- `bool available = false;`：标记缓冲区是否初始化完成（IPC/NVSHMEM 同步后设为 `true`），防止未就绪时使用。
- `int* barrier_signal_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};`：节点内同步信号指针数组，用于 NVLink 通信时的进程间 barrier 同步。
- `int** barrier_signal_ptrs_gpu = nullptr;`：设备端可见的 barrier 信号指针数组，供 GPU 内核直接操作同步信号。

#### **4. 流与同步控制（Stream & Synchronization）**

管理通信流、事件同步及数据接收状态，确保通信操作有序执行与异步效率：

- `at::cuda::CUDAStream comm_stream;`：专用通信 CUDA 流，隔离通信与计算任务以提升并行效率。
- `volatile int* moe_recv_counter = nullptr;`：MoE 接收计数器（volatile 指针），用于跟踪远端发送的 token 数量，实现异步接收状态同步。
- `int* moe_recv_counter_mapped = nullptr;`：设备端映射的接收计数器指针，供 GPU 内核直接更新接收状态。
- `volatile int* moe_recv_expert_counter = nullptr;`：专家级 MoE 接收计数器（volatile 指针），跟踪每个专家的 token 接收进度。
- `int* moe_recv_expert_counter_mapped = nullptr;`：设备端映射的专家级接收计数器指针，供 GPU 内核直接更新。
- `volatile int* moe_recv_rdma_counter = nullptr;`：RDMA 级 MoE 接收计数器（volatile 指针），跟踪跨节点 token 接收进度。
- `int* moe_recv_rdma_counter_mapped = nullptr;`：设备端映射的 RDMA 级接收计数器指针，供 GPU 内核直接更新。

#### **5. 资源生命周期（Resource Lifecycle）**

控制内存资源的分配与释放，避免内存泄漏或重复释放：

- `bool explicitly_destroy;`：标记是否需要显式调用 `destroy()` 释放资源（`true` 时需手动调用，否则析构函数自动释放）。
- `bool destroyed = false;`：标记资源是否已释放（`destroy()` 调用后设为 `true`），防止重复释放。
- `void* workspace = nullptr;`：临时工作区指针，用于存储通信过程中的中间数据（如前缀和、索引表等）。

### 成员函数

#### **1. 生命周期管理（Lifecycle Management）**

- `Buffer`（构造函数）：初始化缓冲区，设置 _rank_、缓冲区大小、低延迟模式等参数。
- `~Buffer`（析构函数）：释放未显式销毁的资源（若 `explicitly_destroy=false`）。
- `sync`：同步跨 _rank_ 设备 ID、IPC 句柄及 NVSHMEM 配置，完成缓冲区初始化。
- `destroy`：显式释放缓冲区内存、IPC/NVSHMEM 句柄及同步信号。

#### **2. 状态检查（Status Check）**

- `is_available`：检查缓冲区是否完成初始化（IPC/NVSHMEM 同步后返回 `true`）。
- `is_internode_available`：检查跨节点（RDMA）通信是否可用（如 NVSHMEM 初始化成功）。

#### **3. 通信与设备信息（Communication & Device Info）**

- `get_num_rdma_ranks`：返回 RDMA 通信组内的 _rank_ 总数。
- `get_rdma_rank`：返回当前进程在 RDMA 通信组内的 _rank_。
- `get_root_rdma_rank`：返回 RDMA 通信组的根 _rank_（支持全局/本地根节点切换）。
- `get_local_device_id`：返回当前进程绑定的 GPU 设备 ID。

#### **4. 内存共享句柄（Memory Sharing Handles）**

- `get_local_ipc_handle`：返回本地 GPU 内存的 IPC 句柄，用于节点内跨进程内存共享。
- `get_local_nvshmem_unique_id`：返回 NVSHMEM 初始化所需的唯一 ID，用于跨节点通信组同步。

#### **5. 缓冲区与流访问（Buffer & Stream Access）**

- `get_local_buffer_tensor`：将本地缓冲区（NVLink/RDMA）转换为 PyTorch 张量，支持指定数据类型和偏移量。
- `get_comm_stream`：返回专用通信 CUDA 流，用于隔离通信与计算任务。

#### **6. 核心通信：分发与聚合（Core Communication: Dispatch & Combine）**

- `get_dispatch_layout`：计算分发布局，生成 token 分配到各 _rank_/expert 的元信息（如 `num_tokens_per_rank`）。
- `intranode_dispatch`：通过 NVLink 在节点内分发 token，支持缓存/非缓存模式。
- `intranode_combine`：通过 NVLink 在节点内聚合 token，支持加权求和与偏置。
- `internode_dispatch`：通过 RDMA 在跨节点分发 token（通常不直接调用，由 `dispatch` 自动路由）。
- `internode_combine`：通过 RDMA 在跨节点聚合 token（通常不直接调用，由 `combine` 自动路由）。

#### **7. 低延迟模式（Low-Latency Mode）**

- `clean_low_latency_buffer`：清理低延迟模式下的缓冲区，释放冗余内存。
- `low_latency_dispatch`：低延迟版 token 分发，支持 FP8 压缩、动态接收统计等优化。
- `low_latency_combine`：低延迟版 token 聚合，支持零拷贝、对数格式权重等优化。
- `get_next_low_latency_combine_buffer`：获取下一个低延迟聚合缓冲区（多轮通信时切换缓冲区避免冲突）。

# Buffer 构造

```cpp

// csrc/deep_ep.cpp

@@ -15, 67

```

`Buffer` 类的构造函数，用于初始化 `Buffer` 对象，管理混合专家（MoE）模型中专家并行（EP）通信所需的各种缓冲区。该构造函数主要完成了 `Buffer` 对象的初始化工作，包括参数检查、排名计算、设备信息获取、内存分配和计数器初始化等操作，为后续的通信操作做好准备。

主要功能包括：

- **设备信息获取**：在 C++ 构造函数里获取设备属性，像多处理器数量等。
- **内存分配**：分配本地内存和工作空间，并且设置本地 IPC 句柄。
- **计数器初始化**：初始化主机端 MoE 计数器。

### 构造函数签名

```cpp

Buffer::Buffer(int rank, int num_ranks, int64_t num_nvl_bytes, int64_t num_rdma_bytes, bool low_latency_mode, bool explicitly_destroy):

```

`Buffer` 构造函数接收以下参数：

1. **`int rank`**：当前进程的全局排名，用于标识不同的进程，在多进程通信中区分各个进程。
2. **`int num_ranks`**：总的进程数量，代表整个通信组中进程的总数。
3. **`int64_t num_nvl_bytes`**：非 RDMA（NVLink）缓冲区所需的内存大小，单位为字节。
4. **`int64_t num_rdma_bytes`**：RDMA（远程直接内存访问）缓冲区所需的内存大小，单位为字节。
5. **`bool low_latency_mode`**：是否启用低延迟模式的标志。若为 `true`，则采用低延迟的通信策略。
6. **`bool explicitly_destroy`**：是否需要显式调用 `destroy` 方法来释放资源的标志。若为 `true`，则需要手动调用 `destroy` 方法；否则，析构函数会自动处理。

- 成员初始化列表：对类的成员变量进行初始化，同时将 `comm_stream` 初始化为从 CUDA 流池中获取的异步流。

### 元数据内存计算

```cpp

int64_t barrier_signal_bytes = NUM_MAX_NVL_PEERS * sizeof(int);

int64_t buffer_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(void*);

int64_t barrier_signal_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(int*);

```

计算用于屏障信号、缓冲区指针和屏障信号指针的内存大小。

### 通用检查

```cpp

EP_HOST_ASSERT(num_nvl_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and (num_nvl_bytes <= std::numeric_limits<int>::max() or num_rdma_bytes == 0));

EP_HOST_ASSERT(num_rdma_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and (low_latency_mode or num_rdma_bytes <= std::numeric_limits<int>::max()));

EP_HOST_ASSERT(0 <= rank and rank < num_ranks and (num_ranks <= NUM_MAX_NVL_PEERS * NUM_MAX_RDMA_PEERS or low_latency_mode));

EP_HOST_ASSERT(num_ranks < NUM_MAX_NVL_PEERS or num_ranks % NUM_MAX_NVL_PEERS == 0);

if (num_rdma_bytes > 0)

EP_HOST_ASSERT(num_ranks > NUM_MAX_NVL_PEERS or low_latency_mode);

```

使用 `EP_HOST_ASSERT` 进行一系列检查，确保缓冲区大小对齐、排名范围正确等。

### 获取排名信息

```cpp

CUDA_CHECK(cudaGetDevice(&device_id));

rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;

num_rdma_ranks = std::max(1, num_ranks / NUM_MAX_NVL_PEERS), num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS);

#ifdef DISABLE_NVSHMEM

EP_HOST_ASSERT(num_rdma_ranks == 1 and not low_latency_mode and "NVSHMEM is disabled during compilation");

#endif

```

- 获取当前 CUDA 设备 ID。
- 计算 RDMA 排名和 NVLink 排名。
- 计算总的 RDMA 进程数量和 NVLink 进程数量。
- 如果禁用了 NVSHMEM，进行相应检查。

### 获取设备信息

```cpp

cudaDeviceProp device_prop = {};

CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));

num_device_sms = device_prop.multiProcessorCount;

```

获取当前 CUDA 设备的属性，记录设备上流式多处理器（SMs）的数量。

### NVLink 缓冲区处理

```cpp

if (num_nvl_bytes > 0) {

CUDA_CHECK(cudaMalloc(&buffer_ptrs[nvl_rank], num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes + barrier_signal_ptr_bytes));

CUDA_CHECK(cudaIpcGetMemHandle(&ipc_handles[nvl_rank], buffer_ptrs[nvl_rank]));

buffer_ptrs_gpu = reinterpret_cast<void**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes);

  

barrier_signal_ptrs[nvl_rank] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes);

barrier_signal_ptrs_gpu = reinterpret_cast<int**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes);

  

CUDA_CHECK(cudaMemsetAsync(barrier_signal_ptrs[nvl_rank], 0, barrier_signal_bytes, comm_stream));

}

```

- 分配 NVLink 缓冲区内存，包括屏障信号、缓冲区指针和屏障信号指针的空间。
- 获取本地 IPC 内存句柄。
- 设置缓冲区指针和屏障信号指针。
- 异步将屏障信号内存初始化为 0。

### 创建工作空间

```cpp

CUDA_CHECK(cudaMalloc(&workspace, NUM_WORKSPACE_BYTES));

CUDA_CHECK(cudaMemsetAsync(workspace, 0, NUM_WORKSPACE_BYTES, comm_stream));

```

分配 32 MiB 的工作空间并异步初始化为 0。

### MoE 计数器初始化

```cpp

CUDA_CHECK(cudaMallocHost(&moe_recv_counter, sizeof(int64_t), cudaHostAllocMapped));

CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_counter_mapped, const_cast<int*>(moe_recv_counter), 0));

*moe_recv_counter = -1;

  

CUDA_CHECK(cudaMallocHost(&moe_recv_expert_counter, sizeof(int) * NUM_MAX_LOCAL_EXPERTS, cudaHostAllocMapped));

CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_expert_counter_mapped, const_cast<int*>(moe_recv_expert_counter), 0));

for (int i = 0; i < NUM_MAX_LOCAL_EXPERTS; ++ i)

moe_recv_expert_counter[i] = -1;

  

if (num_rdma_ranks > 0) {

CUDA_CHECK(cudaMallocHost(&moe_recv_rdma_counter, sizeof(int), cudaHostAllocMapped));

CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_rdma_counter_mapped, const_cast<int*>(moe_recv_rdma_counter), 0));

*moe_recv_rdma_counter = -1;

}

```

- 分配主机内存用于 MoE 接收计数器，并获取设备指针，初始化为 -1。
- 分配主机内存用于 MoE 专家级接收计数器，并获取设备指针，初始化为 -1。
- 如果存在 RDMA 进程，分配主机内存用于 MoE RDMA 级接收计数器，并获取设备指针，初始化为 -1。

# Sync

# get_dispatch_layout

# intranode_dispatch

```cpp

// csrc\deep_ep.cpp

@@ -305, 235

```

`intranode_dispatch` 是 **节点内 token 分发核心函数**，负责通过 NVLink 在同一节点内的多个 GPU 之间高效分发输入 token 数据，支持缓存模式（复用布局信息）和非缓存模式（动态计算布局），并处理 Top-k 选择、FP8 缩放等特性，最终返回接收方的 token 数据及元信息。

#### **关键参数与前置检查**

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

#### 数据、通信准备

```cpp

// csrc\deep_ep.cpp

@@ -353, 53

```

##### **1. 模式区分与通道初始化**

- **缓存模式（`cached_mode = true`）**：复用已计算的 `cached_rank_prefix_matrix`（rank 级前缀和）和 `cached_channel_prefix_matrix`（通道级前缀和），跳过布局计算，直接基于缓存信息分发。
- **非缓存模式（`cached_mode = false`）**：需动态计算分发布局，依赖 `num_tokens_per_rank`（每个 rank 的发送 token 数）和 `num_tokens_per_expert`（每个专家的接收 token 数）。
- **通道数量**：`num_channels = config.num_sms / 2`（每个通道对应 2 个 SM 块，分别用于发送和接收）。

##### **2. 元数据准备与同步**

- **Top-k 与 FP8 处理**：若提供 `topk_idx`/`topk_weights`，则获取其设备指针，用于分发时同步专家索引和权重；若输入为 FP8 格式（`x_scales` 存在），则获取缩放因子指针及步长信息。
- **流管理**：若 `allocate_on_comm_stream` 为真，切换到专用通信流（`comm_stream`）分配张量，避免阻塞计算流；通过 `previous_event` 等待前置任务完成（或直接等待计算流）。

## 分发通知与接收计数（非缓存模式关键）

```cpp

// csrc\deep_ep.cpp

@@ -408, 61

```

- **发送元数据**：调用 `intranode::notify_dispatch` 向节点内其他 rank 发送 token 数量等元信息，通过共享内存（`buffer_ptrs_gpu`）和屏障信号（`barrier_signal_ptrs_gpu`）同步。
- **接收计数等待**：CPU 忙等待 GPU 接收其他 rank 的元数据，通过 `moe_recv_counter`（总接收 token 数）和 `moe_recv_expert_counter`（专家级接收 token 数）判断是否就绪，超时则抛出异常。

> 执行的 kernel 详细见 `kernel.intranode::notify_dispatch` `kernel.intra_node::dispatch`

##### 内存分配与内核调用

- **接收张量分配**：根据接收 token 数（`num_recv_tokens`）分配 `recv_x`（接收数据）、`recv_src_idx`（源 token 索引）、`recv_topk_idx`（接收的 Top-k 索引，若使用）等张量。
- **分发内核启动**：调用 `intranode::dispatch` 内核，通过 NVLink 传输数据：
- 输入数据（`x`）、缩放因子（`x_scales`）、Top-k 信息（`topk_idx`/`topk_weights`）从发送方拷贝到接收方；
- 使用分块传输（`config.num_max_nvl_chunked_send_tokens`/`recv_tokens` 控制块大小），避免大内存连续访问瓶颈。

## 流同步与结果返回

```cpp

// csrc\deep_ep.cpp

@@ -512, 23

```

- **异步处理**：若 `async = true`，记录通信流事件（`EventHandle`），并将张量关联到通信流和计算流，实现异步执行；否则等待计算流完成。
- **返回结果**：返回接收数据（`recv_x`）、元信息（前缀矩阵、源索引等）及同步事件，供后续聚合阶段（`intranode_combine`）使用。

#### **关键特性与优化**

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

### **参数详解**

（按功能分组，`const` 标识输入参数，非 `const` 指针多为输出/双向参数）

#### **1. 分发元信息（输入）**

| 参数名 | 类型 | 作用 |

| ------------------------- | --------------- | -------------------------------------------------------------------------------------------------------------------- |

| `num_tokens_per_rank` | `const int*` | 长度为 `num_ranks` 的数组，存储**当前 rank 发送给每个目标 rank 的 token 数量**（如 `num_tokens_per_rank[i]` 表示发送给 rank `i` 的 token 数）。|

| `num_tokens_per_expert` | `const int*` | 长度为 `num_experts` 的数组，存储**每个 expert 接收的 token 数量**（全局视角，需跨 rank 同步）。|

| `num_ranks` | `int` | 节点内总 rank 数（如 8 表示 8 个 GPU）。|

| `num_experts` | `int` | 全局 expert 总数（需满足 `num_experts % num_ranks == 0`，确保每个 rank 分配到整数个 expert）。|

| `num_tokens` | `int` | 当前 rank 待分发的总 token 数。|

| `is_token_in_rank` | `const bool*` | 2D 数组（形状 `[num_tokens, num_ranks]`），`is_token_in_rank[token_idx][rank] = true` 表示第 `token_idx` 个 token 需要发送给 `rank`。|

#### **2. 输出布局矩阵（输出）**

| 参数名 | 类型 | 作用 |

| --------------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |

| `channel_prefix_matrix` | `int*` | 2D 数组（形状 `[num_ranks, num_channels]`），存储**通道级前缀和**：`channel_prefix_matrix[rank][channel]` 表示 rank `rank` 的第 `channel` 个通道累计处理的 token 数，用于划分通道任务范围。|

| `rank_prefix_matrix_copy` | `int*` | 2D 数组（形状 `[num_ranks, num_ranks]`），复制**rank 级前缀和矩阵**：`rank_prefix_matrix_copy[i][j]` 表示 rank `i` 发送给 rank `j` 的 token 累计数量，用于后续数据分发偏移计算。|

#### **3. 同步与缓冲区（双向）**

| 参数名 | 类型 | 作用 |

| ---------------------------------- | ---------- | ------------------------------------------------------------------------------------------------ |

| `moe_recv_counter_mapped` | `int*` | 指向主机映射的设备内存，存储**当前 rank 接收的总 token 数**（跨 rank 同步后更新）。|

| `moe_recv_expert_counter_mapped` | `int*` | 指向主机映射的设备内存，长度为 `num_experts/num_ranks`，存储**当前 rank 每个 local expert 接收的 token 数**（跨 rank 同步后更新）。|

| `buffer_ptrs` | `void**` | 数组（长度 `num_ranks`），存储**跨 rank 共享缓冲区指针**（用于 NVLink 通信的设备内存）。|

| `barrier_signal_ptrs` | `int**` | 数组（长度 `num_ranks`），存储**屏障信号指针**（用于跨 rank 同步，如等待所有 rank 完成元信息发送）。|

#### **4. 配置参数（输入）**

| 参数名 | 类型 | 作用 |

| -------------------- | ---------------- | ------------------------------------------ |

| `num_memset_int` | `int` | 需要清零的整数数量，用于初始化通信队列（如通道级缓冲区的信号量）。|

| `expert_alignment` | `int` | expert 接收 token 数的对齐值（如 16），确保内存访问对齐，提升效率。|

| `rank` | `int` | 当前 rank 的 ID（0 基）。|

| `stream` | `cudaStream_t` | 内核启动的 CUDA 流，用于异步执行，避免阻塞计算流。|

| `num_channels` | `int` | 通信通道数（每个通道由 1 个 warp 处理，通常与 SM 数量相关）。|

### 条件配置

函数通过宏定义和模板特化实现对不同 `num_ranks`（rank 数量）的适配，核心语法元素如下：

- `#define NOTIFY_DISPATCH_LAUNCH_CASE(ranks)`：定义内核启动模板，根据 `ranks`（模板参数，实际为 `num_ranks`）实例化 `notify_dispatch<ranks>` 模板内核，并传递参数。
- `#undef NOTIFY_DISPATCH_LAUNCH_CASE` 仅用于**取消宏定义**，避免宏污染后续代码。
- `SWITCH_RANKS(NOTIFY_DISPATCH_LAUNCH_CASE)`：根据 `num_ranks` 切换到对应模板实例（如 `num_ranks=4` 时调用 `notify_dispatch<4>`），通过宏展开为 `switch-case` 语句实现。
- `SETUP_LAUNCH_CONFIG(1 + num_ranks, kNumThreads, stream)`

设置内核启动配置：

- **网格大小**：`1 + num_ranks`（1 个块用于全局同步，`num_ranks` 个块用于通道级计算）；
- **块大小**：`kNumThreads=128`（每个块 128 线程）；
- **流**：`stream`（指定 CUDA 流，避免阻塞默认流）。

## Launch Kernel

- `#define NOTIFY_DISPATCH_LAUNCH_CASE(ranks)`：封装 CUDA 内核启动逻辑，传入内核函数指针、参数和配置（`cfg`）。

# `intranode::dispatch`

```cpp

// csrc\kernels\[intranode.cu](http://intranode.cu/)

@@ -475, 34

```

DeepEP 中的 `intranode dispatch` 函数是其核心通信内核之一，专为节点内（同一服务器内 GPU 间）的混合专家（MoE）模型设计。以下是对该函数及其代码的详细解释：

---

### **1. 函数目的**

`intranode dispatch` 负责在**同一节点内的多个 GPU 之间**高效分发数据（如 MoE 模型中的专家输入）。其目标是：

- **高吞吐量**：利用 NVLink 实现节点内 GPU 的高速互联（最高 158 GB/s）。
- **动态负载均衡**：根据每个专家的令牌分布（`num_tokens_per_expert`）动态调整数据分区。
- **低精度优化**：支持 FP8/BF16 数据格式，减少通信开销（显存占用降低 50%）。
- **计算 - 通信重叠**：通过 CUDA 事件和 Hook 机制，最大化 GPU 利用率。

---

### **2. 代码结构解析**

#### **(1) 参数说明**

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

#### **(2) 硬件优化**

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

#### **(3) 内核启动逻辑**

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

### **3. 技术亮点**

#### **(1) NVLink 优化**

- **节点内通信**：通过 NVLink 实现 GPU 间直接内存访问（DMA），带宽高达 160 GB/s（接近硬件极限）。
- **共享内存**：使用 `ipc_handles` 和 `dist.all_gather_object` 减少跨 GPU 同步开销。

#### **(2) 低精度计算**

- **FP8 量化**：
- 输入数据 `x` 和输出 `recv_x` 支持 FP8（`float8_e4m3fn`）。
- 通过 `x_scales` 和 `recv_x_scales` 存储动态缩放因子，实现混合精度通信。
- **优势**：显存占用减少 50%，通信带宽需求降低。

#### **(3) 计算 - 通信重叠**

- **CUDA 事件**：通过 `torch.cuda.Event` 和 `EventOverlap` 类管理通信与计算的异步执行。
- **Hook 机制**：在前向传播时预加载数据，在反向传播时异步传输梯度，不占用 SM 资源。

#### **(4) 动态负载均衡**

- **专家令牌统计**：
- `get_dispatch_layout` 内核统计 `num_tokens_per_expert` 和 `num_tokens_per_rank`。
- 动态调整 `send_head` 和 `recv_topk_idx`，避免令牌分布不均导致的带宽浪费。

---

### **4. 性能表现**

- **节点内吞吐量**：153-158 GB/s（NVLink 带宽）。
- **延迟**：通过 `low_latency_dispatch` 实现<163 微秒延迟（纯 RDMA 模式）。
- **扩展性**：在 2048 卡 H800 集群中，训练吞吐量提升 3.8 倍，推理延迟降低 80%。

---

### **5. 应用场景**

- **MoE 模型训练**：适用于 DeepSeek-V3 等大规模模型的分布式训练。
- **低延迟推理**：支持实时推理解码（如聊天机器人），延迟敏感场景优先使用纯 RDMA 内核。
- **异构网络优化**：自动适配 NVLink→RDMA 的非对称带宽转发，避免跨节点通信瓶颈。

---

### **总结**

`intranode dispatch` 是 DeepEP 的核心组件，通过**硬件感知优化**（NVLink/TMA）、**低精度计算**（FP8）和**动态负载均衡**，实现了节点内 GPU 的高效通信。其设计充分结合了 Hopper 架构的特性，为 MoE 模型提供了接近硬件极限的性能表现。

# Kernel

```cpp

// csrc\kernels\[intranode.cu](http://intranode.cu/)

@@ -166, 307

```

# Combine

# ow lantency