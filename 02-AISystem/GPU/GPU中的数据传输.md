# GPU的DMA

在GPU CUDA编程中，**直接内存访问（DMA, Direct Memory Access）** 的概念与传统CPU系统有所不同，但其核心思想——**在不占用CPU核心的情况下实现高效的数据传输**——是贯穿始终的。CUDA通过其**异步内存拷贝机制**和**统一内存（Unified Memory）** 等特性，实现了类似DMA的功能。

以下是CUDA中实现“DMA式”数据传输的关键技术和使用方法：

---

### 1. 异步内存拷贝（Asynchronous Memory Copy）

这是最接近传统DMA概念的机制。CUDA允许在主机（CPU）和设备（GPU）之间、或设备之间进行**异步**数据传输，这些传输由**硬件DMA引擎**（通常集成在GPU或PCIe控制器中）执行，无需CPU持续干预。

**核心API：**

*   `cudaMemcpyAsync(...)`: 用于异步内存拷贝。
*   `cudaMemPrefetchAsync(...)`: 用于异步预取统一内存。

**关键要素：**

*   **流（Stream）**: 异步操作必须在CUDA流（`cudaStream_t`）中执行。流是操作的队列，确保操作按序执行，但允许不同流的操作并发。
*   **事件（Event）**: 用于同步和查询操作完成状态。

**使用步骤：**

1.  **创建流**:
2.  **发起异步拷贝**:
    *   此调用立即返回，CPU可以继续执行其他任务。
    *   实际的数据传输由GPU的DMA引擎在后台执行。
3.  **（可选）使用事件进行精细同步**:
4.  **销毁资源**:

**优势**:
*   **CPU卸载**: CPU不参与数据传输过程，可以并行执行计算或其他任务。
*   **重叠（Overlap）**: 可以将数据传输与GPU计算重叠，提高整体效率。例如，在GPU执行核函数计算的同时，DMA引擎传输下一批数据。

---

### 2. 统一内存（Unified Memory - UM）

统一内存提供了一个单一的内存地址空间，供CPU和GPU访问。虽然底层实现复杂，但它极大地简化了内存管理，并利用了类似DMA的机制进行**按需迁移（demand-paging）**。

**核心API：**

*   `cudaMallocManaged(...)`: 分配统一内存。

**工作原理：**

1.  **分配**: 使用 `cudaMallocManaged` 分配内存。
2.  **访问**: CPU和GPU都可以直接通过指针 `data` 访问该内存。
3.  **按需迁移**: 当CPU或GPU首次访问某个内存页时，如果该页不在其本地内存中，CUDA运行时会自动触发一个**后台迁移操作**。这个迁移通常由硬件（如GPU的MMU和DMA引擎）或驱动高效完成，对程序员透明。
4.  **预取（Prefetching）**: 可以使用 `cudaMemPrefetchAsync` 主动将数据迁移到目标处理器的内存中，避免运行时迁移的延迟。

**优势**:
*   **简化编程**: 无需显式管理 `cudaMemcpy`。
*   **自动迁移**: 数据根据访问模式自动移动。

**注意**:
*   统一内存的延迟可能高于显式管理的内存，尤其是在频繁跨处理器访问时。
*   高效使用通常需要结合 `cudaMemPrefetchAsync` 进行预取。

---

### 3. 零拷贝内存（Zero-Copy Memory / Host-Registered Memory）

*   **`cudaMallocHost`** 或 **`cudaHostAlloc`** 分配的内存是**页锁定内存（Pinned Memory）**。
*   页锁定内存不会被操作系统换出，允许GPU的DMA引擎通过PCIe总线**直接访问**主机内存。
*   这可以加速 `cudaMemcpy` 操作（因为DMA引擎可以直接读写主机内存），但直接访问（通过GPU核函数）通常比访问设备内存慢。
*   这也是一种DMA应用，因为它允许设备直接访问主机内存。

---
# 存储之间的数据传输

在现代GPU计算体系中，数据传输是性能的关键瓶颈之一。从编程与硬件协同的角度来看，GPU内部及外部的数据流动涉及多个层级的存储结构和传输机制。以下从**硬件架构**与**CUDA编程模型**两个维度，详细解析GPU中各类数据传输方式，涵盖：

- HBM（高带宽内存）到计算单元（SM）
- 片上存储（Shared Memory、L1 Cache、Registers）之间的数据流动
- 多GPU间（Intra-node）
- 多节点间（Inter-node）

---

## 一、GPU 内部数据传输：从 HBM 到计算单元（SM）

### 1. 硬件结构概览

现代GPU（如NVIDIA A100/H100、AMD MI系列）采用**多级存储架构**：

| 层级 | 类型 | 延迟 | 带宽 | 编程可见性 |
|------|------|------|------|------------|
| L2 Cache | 全局缓存（6MB~40MB） | ~200 cycles | ~2TB/s | 不可编程 |
| L1 Cache / Shared Memory | 每个SM共享（可配置） | ~30 cycles | ~1TB/s | 部分可编程 |
| Register File | 每个线程私有 | 1 cycle | 极高 | 编译器管理 |
| HBM（显存） | GDDR6/HBM2e/HBM3 | ~500+ cycles | 1.5~3.5 TB/s | `cudaMalloc` |

---

### 2. 数据路径：HBM → SM（Streaming Multiprocessor）

#### ✅ 编程方式：`cudaMemcpy` / `cudaMemcpyAsync`

- **硬件机制**：DMA引擎（NVLink/NVSwitch或PCIe控制器）负责主机内存与HBM之间的数据搬移。
- **流程**：
  1. CPU调用 `cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice)`。
  2. CUDA驱动将请求提交给**GPU内存控制器**。
  3. **PCIe DMA引擎**将数据从主机RAM搬至GPU显存（HBM）。
  4. 数据存入HBM，等待核函数访问。

> ⚠️ 注意：HBM本身不直接连接SM，数据需通过L2缓存或显存控制器读入。

---

### 3. SM内部数据流动（HBM → L2 → L1/Shared Mem → Register）

#### （1）全局内存访问 → 寄存器（Global Load）

```cuda
__global__ void kernel(float* data) {
    int idx = threadIdx.x;
    float val = data[idx];  // 从HBM加载到寄存器
    // ...
}
```

- **硬件路径**：
  - 请求 → L2 Cache → L1 Cache → Register
  - 若命中L1/L2，则避免访问HBM
  - 访问模式（合并访问 coalescing）极大影响带宽利用率

#### （2）使用 Shared Memory（片上高速缓存）

```cuda
__global__ void kernel(float* input) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    sdata[tid] = input[tid];
    __syncthreads();
    // 使用sdata进行计算
}
```

- **硬件路径**：
  - 数据从HBM → L2 → **Shared Memory（SRAM）**
  - SM内的所有线程可高速访问（延迟 ~10~30 cycles）
- **优势**：避免重复从HBM读取，实现**数据复用**

#### （3）使用 Constant / Texture Memory（只读缓存）

- 硬件专用缓存，适合只读数据（如权重、查找表）
- 自动缓存，减少HBM访问

---

## 二、GPU内部不同单元之间的数据传输

### 1. SM之间数据共享

- **无直接硬件通道**：SM之间不能直接通信。
- **必须通过全局内存或L2缓存**：
  - SM A 写数据到 global memory
  - SM B 从 global memory 读取
- **同步**：需使用 `__threadfence()` 确保写入可见性

```cuda
// SM A
output[0] = result;
__threadfence(); // 确保写入对其他SM可见
```

### 2. 多SM协作：Grid-Level 同步（需Compute Capability 7.0+）

- 使用 `__syncthreads()` 仅限于一个block内
- 跨block同步需使用**网格同步（Grid Sync）**：
  ```cuda
  __syncthreads(); // block内
  // 跨block需使用 cooperative groups
  #include <cooperative_groups.h>
  namespace cg = cooperative_groups;
  cg::grid_group grid = cg::this_grid();
  grid.sync(); // 需编译选项 -rdc=true 和 cudaLaunchCooperativeKernel
  ```

---

## 三、多GPU间数据传输（Intra-node）

### 1. 硬件互联技术

| 技术 | 带宽（单向） | 延迟 | 支持 |
|------|-------------|------|------|
| PCIe 4.0 x16 | ~32 GB/s | 高 | 所有GPU |
| NVLink 2.0 | ~25 GB/s per link | 低 | Tesla/V100/A100 |
| NVLink 3.0 | ~50 GB/s per link | 极低 | A100/H100 |
| NVSwitch | 全互联拓扑 | 极低 | DGX系统 |

> 示例：NVIDIA A100 支持 12条NVLink，总带宽可达 600 GB/s

---

### 2. 编程方式

#### （1）P2P（Peer-to-Peer）直接访问
P2P 允许一个 GPU **直接访问另一个 GPU 的显存（HBM）**，无需经过主机内存中转。


```c++
// 启用P2P访问
cudaDeviceEnablePeerAccess(device_id_dst, 0);

// 直接从GPU0访问GPU1的内存
float* ptr_on_gpu1; // 在GPU1上分配
cudaSetDevice(0);
kernel<<<..., stream>>>(ptr_on_gpu1); // GPU0直接读取GPU1内存
```

- **硬件机制**：通过NVLink/NVSwitch直接访问远程GPU的HBM
	- **物理链路：NVLink（或 NVSwitch）**。
	    - NVLink 是 GPU 之间的高速串行互连总线，带宽远高于 PCIe。
	    - 在 DGX 等系统中，多个 NVLink 通过 NVSwitch 实现全互联拓扑。
	- **传输机制：硬件 DMA 引擎 + 内存控制器**。
	    - 当 GPU A 访问 GPU B 的内存地址时，GPU 的内存管理单元（MMU）识别这是远程地址。
	    - 触发 **P2P DMA 引擎**，通过 NVLink 将数据从 GPU B 的 HBM 读取到 GPU A 的 L2 缓存或直接传送到 SM。
	    - 整个过程由硬件自动完成，CPU 不参与。
- **无需经过主机内存**
- 需检查是否支持：`cudaDeviceCanAccessPeer()`
- 数据路径：`GPU A (SM) → L2 Cache → NVLink/NVSwitch → GPU B 的 HBM（显存）`

> ✅ **总结**：P2P = **NVLink 物理链路 + 硬件 DMA 引擎 + 远程内存映射**
#### （2）显式P2P拷贝

```c++
cudaMemcpyPeer(
    dst_ptr, dst_device, 
    src_ptr, src_device, 
    size
);
```

- 使用DMA引擎在GPU之间直接传输
- 比通过主机中转快数倍

|特性|`cudaMemcpyPeer`（显式拷贝）|直接访问（P2P Access）|
|---|---|---|
|**操作类型**|显式数据搬移（copy）|隐式内存访问（load/store）|
|**编程方式**|`cudaMemcpyPeer(dst, src, size)`|在 kernel 中直接使用远程指针|
|**控制粒度**|按字节拷贝|按访问模式（如 coalesced）|
|**同步性**|同步或异步流中执行|异步，由 kernel 执行流控制|
|**适用场景**|大块数据预拷贝|算法需要跨 GPU 共享数据结构（如稀疏矩阵）|
|**性能特点**|可重叠计算与通信|可能产生随机访问，带宽利用率低|

#### （3）使用 GPUDirect RDMA（高级）

- 允许第三方设备（如网卡、存储）直接访问GPU内存
- 用于高性能网络（如InfiniBand）
- 需驱动和硬件支持
- **物理链路**：
    - **InfiniBand 网卡（NIC）或 RoCE 网卡** 通过 **PCIe 总线** 连接到主机。
    - GPU 也通过 PCIe 连接到主机（或通过 NVLink 走 Host Bridge）。
- **GPUDirect RDMA** 就是 **跨节点 RDMA 在 GPU 上的扩展**。
- **协议栈**：
    - 使用 **InfiniBand Verbs** 或 **RoCE v2** 协议。
    - 支持 **Remote Direct Memory Access (RDMA)** 操作：`RDMA_READ`, `RDMA_WRITE`, `RDMA_SEND`。
- **关键机制**：
    - 网卡驱动与 NVIDIA 驱动协作（通过 `nv_peer_mem` 模块）。
    - 网卡获取 GPU 内存的 **物理地址或 IOMMU 映射**。
    - 网卡通过 PCIe DMA 直接读写 GPU HBM。

> ✅ **结论**：高性能网络对 GPU 内存的访问就是 **基于 RDMA 协议 + GPUDirect 技术** 的跨节点直接访问，与节点内 P2P 类似，只是链路换成了 InfiniBand/RoCE。


#### （4）NVSHMEM 

- **全称**：NVIDIA SHMEM（Scalable Hierarchical Memory）
- **定位**：一种 **PGAS（Partitioned Global Address Space）** 编程模型库，专为多 GPU 多节点设计。
- **前身**：基于开源 OpenSHMEM，由 NVIDIA 优化并深度集成 CUDA。
- **目标**：简化大规模并行编程，替代复杂的 MPI + CUDA 混合编程。
- **访问的是“远程 GPU 的全局内存（HBM）”**。
- 提供 **单边通信（One-sided Communication）** 接口：
	- ```
	    nvshmem_float_p(&remote_addr, value, remote_pe);  // put
	    float val = nvshmem_float_g(&remote_addr, remote_pe);  // get
	    ```
- `remote_addr` 是另一个 GPU 上分配的设备内存指针（通过 `nvshmem_malloc` 分配）。
- 数据直接在 **GPU HBM 之间传输**，路径：
    - 节点内：通过 NVLink P2P
    - 节点间：通过 InfiniBand + GPUDirect RDMA

优势
- **统一编程模型**：无需区分 MPI_Send/Recv 与 CUDA memcpy。
- **自动路径选择**：NVSHMEM 库自动选择最优传输路径（NVLink → IB）。
- **与 NCCL 互补**：NCCL 用于集合通信（AllReduce），NVSHMEM 用于点对点或非规则通信。

> ✅ **适用场景**：图计算、不规则并行算法、动态负载均衡。


---

## 四、跨节点数据传输（Inter-node）


当 GPU 分布在不同服务器节点时，需通过高速网络互联。

### 1. 硬件架构

- **网卡（NIC）**：
    - **NVIDIA Quantum-2 InfiniBand**：支持 400 Gb/s HDR InfiniBand
    - **NVIDIA ConnectX-7**：支持 400 Gb/s Ethernet（RoCE v2）
- **交换机**：NVIDIA Quantum-2 QM9700 交换机（支持 64×400Gb/s）
- **拓扑**：Fat-Tree、Dragonfly 等高性能拓扑

### 2. 核心技术：GPUDirect RDMA

- **定义**：允许第三方设备（如网卡）**直接访问 GPU 显存**，绕过 CPU 和主机内存。
- **作用**：
    - 节点间 GPU 通信无需 `D2H + NIC send + NIC recv + H2D`
    - 实现“零拷贝”跨节点传输
- **数据路径**：

- ```
    GPU A (HBM)
        → NVLink → Host Bridge
        → PCIe → 网卡（NIC）
        → 网络 → 远程网卡
        → PCIe → GPU B (HBM)
    ```
    
    全程无 CPU 参与，无主机内存拷贝。

### 3. 支持的网络协议

|协议|全称|特点|
|---|---|---|
|**InfiniBand (IB)**|-|原生 RDMA，低延迟，高带宽|
|**RoCE v2**|RDMA over Converged Ethernet|基于以太网的 RDMA，需支持 DCQCN 流控|
|**TCP/IP**|-|不支持 RDMA，性能差，仅作 fallback|

> ✅ 推荐使用 **InfiniBand** 或 **RoCE v2** 以支持 GPUDirect RDMA。


### 2. 编程方式

#### （1）MPI + CUDA（主流方式）

- 支持直接传递 GPU 设备指针
- 支持的 MPI 实现：
	- **OpenMPI + UCX**（推荐）
	- **MVAPICH2-GPU**
	- **Intel MPI**
- 底层使用 **GPUDirect RDMA**：
  - MPI库调用CUDA驱动获取GPU内存物理地址
  - 网卡（支持RDMA）通过PCIe DMA直接读取GPU内存
  - 数据经网络发送到远程节点的网卡
  - 远程网卡通过DMA写入目标GPU的HBM

> ✅ 全程无需CPU拷贝到主机内存，实现“零拷贝”跨节点传输

#### （2）NCCL（NVIDIA Collective Communications Library）

- 专为多GPU多节点优化的集合通信库
- 自动选择最优路径（NVLink → PCIe → InfiniBand）
	- 节点内：NVLink → P2P → PCIe
	- 节点间：InfiniBand/RoCE + GPUDirect RDMA
- 支持 `AllReduce`, `Broadcast`, `AllGather` 等

#### 3. UCX（Unified Communication X）

- 高性能通信框架，支持多种传输（IB、RoCE、 shared memory、 TCP）
- 被 OpenMPI、Spark、AI 框架广泛采用
- 自动调度最优路径，支持 GPUDirect
---

## 五、总结：GPU数据传输层级与技术对照表

| 传输路径           | 硬件机制                      | 编程接口                   | 是否使用DMA           |
| -------------- | ------------------------- | ---------------------- | ----------------- |
| Host → GPU HBM | PCIe DMA                  | `cudaMemcpyAsync`      | ✅                 |
| GPU HBM → SM   | Memory Controller + Cache | Kernel Load            | ✅（隐式）             |
| SM内共享          | Shared Memory (SRAM)      | `__shared__`           | ❌（片上）             |
| GPU间（同节点）      | NVLink / NVSwitch         | `cudaMemcpyPeer`, P2P  | ✅                 |
| 节点间（跨服务器）      | InfiniBand + RDMA         | MPI (CUDA-aware), NCCL | ✅（GPUDirect RDMA） |
| 外设→GPU         | GPUDirect RDMA            | IB驱动 + CUDA            | ✅                 |

---

## 六、性能优化建议

1. **最大化合并内存访问**（coalesced access）以提升HBM带宽利用率
2. **使用异步流**重叠计算与通信
3. **优先使用NVLink**而非PCIe进行多GPU通信
4. **启用CUDA-aware MPI**和**GPUDirect RDMA**用于分布式训练
5. **合理使用Shared Memory**减少HBM访问
6. **使用NCCL**进行集合通信，自动优化路径

---
# 存储的编程可见性

### **PTX 编程可见的存储层次 vs CUDA 编程**

#### **(1) CUDA 高级编程可见的存储层次**

在 `.cu` 文件中，程序员主要操作以下内存空间：

|存储类型|CUDA API / 关键字|可见性|
|---|---|---|
|全局内存|`cudaMalloc`|所有线程可读写|
|共享内存|`__shared__`|Block 内线程共享|
|常量内存|`__constant__`|只读，缓存|
|纹理内存|`texture`|只读，缓存|
|寄存器|自动分配（`float a;`）|线程私有|
|局部内存|自动（大数组溢出）|线程私有，实际在 HBM|

> ⚠️ **注意**：CUDA 程序员**不直接操作 L1/L2 缓存**，它们是透明的。

#### **(2) PTX 编程可见的存储层次**

PTX（Parallel Thread Execution）是 CUDA 的**虚拟汇编语言**，在 `.ptx` 文件中编写，可见更底层的存储空间：

| 存储类（PTX）           | 说明              |
| ------------------ | --------------- |
| `%r` / `%f` / `%d` | 32/64/64位通用寄存器  |
| `%p`               | 64位指针寄存器        |
| `.reg`             | 软件管理的寄存器变量      |
| `.shared`          | 共享内存（与 CUDA 一致） |
| `.global`          | 全局内存            |
| `.const`           | 常量内存            |
| `.local`           | 局部内存（HBM）       |
| `.param`           | kernel 参数内存     |
| `.surf` / `.tex`   | 表面/纹理内存         |
|                    |                 |

> ✅ **关键区别**：
> 
> - PTX 可以**直接操作寄存器命名**（如 `%r1`, `%f2`），而 CUDA 中由编译器分配。
> - PTX 可以使用更细粒度的**内存一致性控制**（如 `volatile`, `cache_hint`）。
> - PTX 支持**特定硬件指令**（如 warp shuffle, atomics, predication）。

#### **(3) 是否一样？**

❌ **不完全一样**。

- **抽象层级不同**：
    - CUDA C++ 是高级抽象，隐藏寄存器、指令调度等细节。
    - PTX 是低级虚拟汇编，暴露寄存器、内存类、控制流等。
- **存储空间映射一致**：
    - `.global` → `cudaMalloc`
    - `.shared` → `__shared__`
    - `.const` → `__constant__`
- **但 PTX 提供了更细粒度的控制**，例如：

- ```
    ld.global.ca.f32 %f1, [%rd1];  // 显式缓存提示（cache around）
    st.shared.relaxed.f32 [%r2], %f3; // 松散一致性写入
    ```

> ✅ **总结**：存储层次**逻辑上一致**，但 PTX 提供了**更低层级、更精细的控制能力**，适合性能调优或实现编译器无法优化的特殊模式。



# 常量内存（Constant Memory）和纹理内存（Texture Memory）

### **作用与使用方法**

这两种内存都是 **只读缓存内存**，专为特定访问模式优化，位于芯片上，延迟远低于 HBM。

|特性|常量内存（Constant Memory）|纹理内存（Texture Memory）|
|---|---|---|
|**用途**|存放只读常量（如权重、参数）|存放图像、网格数据，支持插值|
|**缓存位置**|专用常量缓存（每个 SM）|纹理缓存（L1级）|
|**带宽优化**|同一 warp 所有线程访问同一地址时高效|支持 1D/2D/3D 局部性访问|
|**最大大小**|64 KB（全局）|取决于架构（通常大）|
|**是否可写**|否（只读）|否（只读）|

---

#### **(1) 常量内存（Constant Memory）**

##### **作用**

- 用于存放**所有线程共享的只读常量**，如：
    - 神经网络的权重（小规模）
    - 物理模拟中的常数（如重力、光速）
    - 查找表（LUT）
- **优化机制**：
    - 如果一个 warp 的 32 个线程**同时访问同一个地址**（广播模式），常量缓存只需一次 HBM 访问，广播给所有线程，带宽利用率极高。

##### **使用方法**

```
// 1. 在全局作用域声明常量内存
__constant__ float const_data[256];

// 2. 主机端初始化
float h_data[256] = { /* 初始化数据 */ };
cudaMemcpyToSymbol(const_data, h_data, 256 * sizeof(float));

// 3. 在 kernel 中使用（像普通数组）
__global__ void kernel() {
    int idx = threadIdx.x;
    float val = const_data[idx];  // 自动从常量缓存加载
    // ...
}
```

> ⚠️ **注意**：
> 
> - `__constant__` 变量必须在文件作用域声明。
> - 使用 `cudaMemcpyToSymbol` 或 `cudaMemcpyFromSymbol` 进行主机-设备传输。

---

#### **(2) 纹理内存（Texture Memory）**

##### **作用**

- 原为图形处理设计，现广泛用于**科学计算中的局部性访问**。
- 适用于：
    - 图像处理（卷积、滤波）
    - 有限差分法（FDM）、流体模拟
    - 支持 **硬件插值**（线性插值）
- **优化机制**：
    - 纹理缓存针对 **2D/3D 空间局部性** 优化（类似图像像素邻域访问）。
    - 即使线程访问不同地址，只要空间接近，缓存命中率高。
---

### **总结对比表**

| 内存类型     | 适用场景      | 访问模式优化           | 是否支持插值 | 编程方式                                  |
| -------- | --------- | ---------------- | ------ | ------------------------------------- |
| **全局内存** | 通用数据      | 合并访问（coalescing） | 否      | `cudaMalloc`                          |
| **共享内存** | Block 内共享 | 手动管理，低延迟         | 否      | `__shared__`                          |
| **常量内存** | 只读常量、广播   | 同地址访问            | 否      | `__constant__` + `cudaMemcpyToSymbol` |
| **纹理内存** | 图像、网格、插值  | 2D/3D 局部性        | ✅ 是    | `texture` / `cudaTextureObject_t`     |

掌握这些存储层次和访问模式，是优化 CUDA 程序性能的关键。