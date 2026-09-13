>[!info] 本文参考：John Cheng, Max Grossman, Ty McKercher — *Professional CUDA C Programming* (Wrox, 2014)，第 9 章 **Multi-GPU Programming**（p.~414）。笔记按「中英文原文讲解 + 关键概念 + 代码/实测」整理，与 7-指令级原语调优、10-实现考量 同框架。

# Overview

> **WHAT'S IN THIS CHAPTER?**
> ➤ Managing multiple GPUs
> ➤ Executing kernels across multiple GPUs
> ➤ Overlapping computation and communication between GPUs
> ➤ Synchronizing across GPUs
> ➤ Exchanging data using CUDA-aware MPI
> ➤ Exchanging data using CUDA-aware MPI with GPUDirect RDMA
> ➤ Scaling applications across a GPU-accelerated cluster
> ➤ Understanding CPU and GPU affinity

本章把视角从「单设备」拉到「多设备」：如何在**单个节点内的多 GPU**，进而到**跨节点的 GPU 加速集群**上扩展应用。CUDA 提供的核心能力有三：① 单/多线程/多进程管理多设备；② 通过 **UVA（统一虚拟寻址）** 与 **GPUDirect** 直接访问别卡显存；③ 用**流 + 异步函数**在多设备间重叠计算与通信。

> 📖 以下内容整理自《Professional CUDA C Programming》Ch.9 — Multi-GPU Programming（多 GPU 编程）。建议结合 Ch.4（内存/UVA）、Ch.6（流与并发）一起阅读。

# Ch.9 多 GPU 编程（Multi-GPU Programming）

## 一、为什么走向多 GPU / 拓扑与通信模式

书中开宗明义：

> "The most common reasons for adding multi-GPU support to an application are: Problem domain size … Throughput and efficiency."

两类动机：① **数据量太大**塞不进单卡显存；② **吞吐/效率**——单任务能放进单卡时，用多卡并发处理多个任务提高吞吐。多 GPU 还能把单节点功耗摊到多卡上，提升「每瓦性能」。

**两种连接拓扑**（Figure 9-1，非互斥）：
- 单节点内：多 GPU 经 **PCIe 总线**互联（可能挂在同一 PCIe root node 或经 PCIe switch 树状连接）；
- 集群内：多节点经 **InfiniBand Switch** 互联。
- PCIe 是**全双工**的——可用 CUDA API 在 PCIe link 间规划路径，避免共享数据时的总线争用。

**两类问题划分 → 两种通信模式**：
1. **无数据交换**：各分区完全独立，只需学会跨设备传数与启动 kernel（如向量加法）。
2. **部分数据交换**：需在各卡间冗余存储并在分区边界交换数据——回扣 Ch.5 的 **halo region（幽灵区/边界区）** 概念：边界区数据要被邻居分区读取但不产生输出，通常**远小于内部区**；当「交换 halo 的通信开销」与「计算内部区的耗时」可重叠时，就能隐藏多 GPU 通信成本。

> "In general, you want to avoid staging data through host memory … It is important to pay attention to both how much data is transferred and how many transfers occur."

⚠️ **关键原则**：尽量避免「先拷回 Host 再拷到另一张卡」的二次中转（staging through host），并同时关注**传输量**与**传输次数**。

## 二、执行于多 GPU：设备管理 API

CUDA 4.0 起多 GPU 编程变得直接。运行时 API 支持多种管理/执行方式。

**① 查询设备数**

```c
cudaError_t cudaGetDeviceCount(int* count);
```

返回 compute capability ≥ 1.0 的设备数。遍历查属性：

```c
int ngpus;
cudaGetDeviceCount(&ngpus);
for (int i = 0; i < ngpus; i++) {
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, i);
    printf("Device %d has compute capability %d.%d.\n", i, devProp.major, devProp.minor);
}
```

**② 设置当前设备**

```c
cudaError_t cudaSetDevice(int id);
```

> "This function will not cause synchronization with other devices, and therefore is a low-overhead call."

- `cudaSetDevice` **不引发与其他设备的同步**，开销极低；可随时在任意 host 线程切到任意设备。
- 若不显式调用，首条 CUDA API 前默认 device 0。
- 一旦选定当前设备，**所有后续 CUDA 操作都作用于该设备**：其上调配的 device 内存物理驻留该卡；host 内存生命周期绑定该卡；该卡上创建的 stream/event；该卡上启动的 kernel。

**③ 单 host 线程循环调度多卡**（异步、可安全切卡）

```c
for (int i = 0; i < ngpus; i++) {
   cudaSetDevice(i);                                  // 设当前设备
   kernel<<<grid, block>>>(...);                      // 该卡上启动
   cudaMemcpyAsync(...);                              // 该卡异步 H2D/D2H
}
```

> "you can safely switch devices even if kernels or transfers issued by the current thread are still executing on the current device, because cudaSetDevice does not cause host synchronization."

多 GPU 可来自：单节点单线程 / 单节点多线程 / 单节点多进程 / 跨节点多进程。

## 三、Peer-to-Peer（P2P）通信

> "Kernels executing in 64-bit applications on devices with compute capability 2.0 and higher can directly access the global memory of any GPU connected to the same PCIe root node."

- 前置要求：CUDA ≥ 4.0、对应驱动、**两张及以上 Fermi/Kepler 且连在同一 PCIe root node**。
- 两种 P2P 模式：
  - **P2P Access**：kernel 内直接 load/store 别卡地址。
  - **P2P Transfer**：GPU 间直接拷贝数据。
- ⚠️ 若两卡在**不同 PCIe root node**，P2P access 不支持；但仍可调用 P2P transfer API——驱动会**透明地经 Host 内存中转**，性能下降。

**① 检测与启用 P2P Access**

```c
cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice);
cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flag); // flag 保留，须 0
cudaError_t cudaDeviceDisablePeerAccess(int peerDevice);
```

- `canAccessPeer` 返回 1 表示可直访、0 表示不可。
- 启用是**单向**的：`enablePeerAccess` 只开「当前设备 → peerDevice」方向；反向需再调一次。
- P2P access **不支持 32-bit 应用**，且保持启用直到显式 disable。

**② P2P 显存拷贝**

```c
cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDev,
                                void* src, int srcDev,
                                size_t nBytes, cudaStream_t stream);
```

- 相对于 host 与所有其他设备**异步**；若 srcDev/dstDev 同 PCIe root node，走最短 PCIe 路径、**不经 Host 中转**。

## 四、跨多 GPU 同步

Ch.6 的 stream/event API 直接适用于多 GPU——**每个 stream/event 只关联单一设备**。典型工作流：① 选定所用 GPU 集；② 每设备建 stream/event；③ 每设备分配资源；④ 经 stream 在每卡启动任务；⑤ 用 stream/event 查询/等待完成；⑥ 清理。

> "You can launch a kernel in a stream only if the device associated with that stream is the current device."
> "A memory copy can be issued in any stream at any time, regardless of what device it is associated with or what the current device is."

⚠️ **两条铁律**：① 在某 stream 启动 kernel / 记录 event **前必须先 cudaSetDevice 到该 stream 所属设备**；② **显存拷贝不用设当前设备**（任意时刻任意 stream 都可发，且设不设设备都不影响行为）。

## 五、实战①：多 GPU 向量加法（simpleMultiGPU.cu）

把 Ch.2 向量加法扩展到多卡，**无跨分区数据交换**的典型场景。

**资源声明**（用数组管理每卡变量）：

```c
float *d_A[NGPUS], *d_B[NGPUS], *d_C[NGPUS];
float *h_A[NGPUS], *h_B[NGPUS], *hostRef[NGPUS], *gpuRef[NGPUS];
cudaStream_t stream[NGPUS];
int size  = 1 << 24;          // 总 16M 元素
int iSize = size / ngpus;     // 均分
size_t iBytes = iSize * sizeof(float);
```

**每设备分配**（注意循环开头必 `cudaSetDevice`，且用 **pinned memory** 支撑异步拷贝）：

```c
for (int i = 0; i < ngpus; i++) {
   cudaSetDevice(i);
   cudaMalloc((void**)&d_A[i], iBytes); cudaMalloc((void**)&d_B[i], iBytes); cudaMalloc((void**)&d_C[i], iBytes);
   cudaMallocHost((void**)&h_A[i], iBytes); cudaMallocHost((void**)&h_B[i], iBytes);
   cudaMallocHost((void**)&hostRef[i], iBytes); cudaMallocHost((void**)&gpuRef[i], iBytes);
   cudaStreamCreate(&stream[i]);
}
```

**分发计算**（同流内 H2D→kernel→D2H，异步、可安全切卡）：

```c
for (int i = 0; i < ngpus; i++) {
   cudaSetDevice(i);
   cudaMemcpyAsync(d_A[i], h_A[i], iBytes, cudaMemcpyHostToDevice, stream[i]);
   cudaMemcpyAsync(d_B[i], h_B[i], iBytes, cudaMemcpyHostToDevice, stream[i]);
   iKernel<<<grid, block, 0, stream[i]>>>(d_A[i], d_B[i], d_C[i], iSize);
   cudaMemcpyAsync(gpuRef[i], d_C[i], iBytes, cudaMemcpyDeviceToHost, stream[i]);
}
cudaDeviceSynchronize();
```

**实测（Tesla M2090 ×2，16M 元素）**：2 GPU 用 35.35ms，1 GPU 用 42.25ms。时间未减半（因拷贝与 kernel 本身有固定开销），但仍显著更快。`nvprof --print-gpu-trace` 显示两卡操作被**完美平分**（H2D ~5.1 GB/s、DtoH ~3.4 GB/s、kernel ~0.72ms 各一）。

## 六、实战②：P2P 通信（simpleP2P_PingPong.cu）

**① 启用双向 P2P**

```c
inline void enableP2P (int ngpus) {
   for( int i = 0; i < ngpus; i++ ) {
      cudaSetDevice(i);
      for(int j = 0; j < ngpus; j++) {
         if(i == j) continue;
         int peer_access_available = 0;
         cudaDeviceCanAccessPeer(&peer_access_available, i, j);
         if (peer_access_available) cudaDeviceEnablePeerAccess(j, 0);
      }
   }
}
```

**② 单向 ping-pong（`cudaMemcpy`，不需切设备）**

```c
cudaSetDevice(0); cudaEventRecord(start, 0);
for (int i = 0; i < 100; i++) {
   if (i % 2 == 0) cudaMemcpy(d_src[1], d_src[0], iBytes, cudaMemcpyDeviceToDevice);
   else            cudaMemcpy(d_src[0], d_src[1], iBytes, cudaMemcpyDeviceToDevice);
}
cudaSetDevice(0); cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
float elapsed_time_ms; cudaEventElapsedTime(&elapsed_time_ms, start, stop);
```

实测（64MB buffer，100 次）：**单向 13.41ms → 5.00 GB/s**。

**③ 双向异步 ping-pong（`cudaMemcpyAsync` + 双流）**

```c
cudaEventRecord(start, 0);
for (int i = 0; i < 100; i++) {
   cudaMemcpyAsync(d_src[1], d_src[0], iBytes, cudaMemcpyDeviceToDevice, stream[0]);
   cudaMemcpyAsync(d_rcv[0], d_rcv[1], iBytes, cudaMemcpyDeviceToDevice, stream[1]);
}
```

实测：**双向 13.39ms → 10.02 GB/s**（带宽翻倍，因 PCIe 全双工双向同时用）。若 `enableP2P` 被去掉，传输改走 host 中转，带宽明显掉——但**仍不报错**。

**④ P2P + UVA 内核直访**

> "Combining the peer-to-peer CUDA APIs with UVA enables transparent access to memory on any device. … kernels executing on one device can dereference a pointer to memory on another device."

- UVA（Ch.4）把 CPU 系统内存与 device 全局内存映射到**单一虚拟地址空间**；从地址本身就能判断它属于哪张卡。
- 要求：64-bit + CC ≥ 2.0 + CUDA ≥ 4.0。检测：`prop.unifiedAddressing`。
- kernel 内直接解引用别卡指针：

```c
__global__ void iKernel(float *src, float *dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dst[idx] = src[idx] * 2.0f;
}
cudaSetDevice(0); iKernel<<<grid, block>>>(d_rcv[0], d_src[1]); // GPU0 读 GPU1 显存
cudaSetDevice(1); iKernel<<<grid, block>>>(d_rcv[1], d_src[0]); // GPU1 读 GPU0 显存
```

⚠️ **性能警示**：过度依赖 UVA 做 P2P 访问有负面代价——**大量跨 PCIe 的小传输开销显著**。

## 七、实战③：多 GPU 二维有限差分（simple2DFD.cu）

用 **2D 波动方程**有限差分演示**计算与通信重叠**（既有重计算又有 halo 通信）。

**halo 模式**（Figure 9-4/9-5）：沿 y 维切分数据，每块加 padding 存与邻居交换的 halo。每时间步通用模式：
1. 在 `stream_halo` 上**算 halo 区 + 与邻居交换 halo**；
2. 在 `stream_internal` 上**算内部区**；
3. **同步**所有设备后进入下一迭代。

> "If the computation time required for the internal calculations is longer than the time required for the halo operations, you can realize linear speedup using multiple GPUs by hiding the performance impact of halo communication."

**重叠伪代码**（两流把 halo 通信与内部计算重叠）：

```c
for (int istep = 0; istep < nsteps; istep++) {
   for (int i = 0; i < 2; i++) {            // 1) halo 计算（halo 流）
      cudaSetDevice(i);
      2dfd_kernel<<<..., stream_halo[i]>>>(...);
   }
   // 2) halo 交换（halo 流，任意设备可发，不需切设备）
   cudaMemcpyAsync(..., cudaMemcpyDeviceToDevice, stream_halo[0]);
   cudaMemcpyAsync(..., cudaMemcpyDeviceToDevice, stream_halo[1]);
   for (int i = 0; i < 2; i++) {            // 3) 内部计算（internal 流）
      cudaSetDevice(i);
      2dfd_kernel<<<..., stream_internal[i]>>>(...);
   }
   for(int i = 0; i < 2; i++) {             // 4) 同步后交换双缓冲指针
      cudaSetDevice(i); cudaDeviceSynchronize();
      float *tmpu0 = d_u1[i]; d_u1[i] = d_u2[i]; d_u2[i] = tmpu0;
   }
}
```

**kernel 关键优化**（8 阶空间 + 2 阶时间）：x 维 stencil 用 **shared memory**（含 4+4 padding 存邻居），y 维 stencil 用**寄存器数组** `float yval[9]` 减少冗余全局访问；`#pragma unroll` 展开循环。

**实测（Tesla M2090 ×2）**：2 GPU 性能 **962.77 MCells/sec**（gputime 0.27ms），1 GPU **502.98 MCells/sec**（0.52ms）→ **近线性扩展（96% 效率）**。结论：**halo 通信开销被 CUDA 流有效隐藏**。`nvcc -arch=sm_20 -Xptxas -v` 报告 `kernel_2dfd` 用 26 寄存器 + 160B 共享内存/线程。nvvp 时间线（Figure 9-8）可见每卡两流：一流通信用、一流纯计算。

## 八、跨 GPU 集群扩展：MPI

> "MPI (Message Passing Interface) is a standardized and portable API for communicating data via messages between distributed processes."
> "With traditional MPI, only the contents of host memory can be transmitted directly … With CUDA-aware MPI, you can pass GPU memory directly to MPI functions without staging that data through host memory."

**两类 MPI**：
- **传统 MPI**：只能直接传 host 内存；GPU 内存须先 `cudaMemcpy` 回 host 再 MPI。
- **CUDA-aware MPI**：可直接把 device 指针传给 MPI 函数，省去中转。常见实现：MVAPICH2、MVAPICH2-GDR（含 GPUDirect RDMA）、OpenMPI 1.7、CRAY MPI、IBM Platform MPI。

**① CPU→CPU 基线（MVAPICH2）**：MPI 四步（Init → 收发 → Barrier 同步 → Finalize）。用 `MPI_Isend`/`MPI_Irecv`/`MPI_Waitall` 做双向非阻塞传输。实测 4MB 消息 ~6328 MB/s、延迟 662μs。

**② CPU Affinity（亲和性）**：OS 可能在多核间迁移进程/线程，破坏数据局部性（新核缓存无该进程数据，需从系统内存重取）。绑定进程到特定核叫 CPU affinity，MVAPICH2 用 `MV2_ENABLE_AFFINITY=1/0` 控制。

**③ GPU↔GPU 传统 MPI**：每节点给每 GPU 绑一个 MPI 进程。流程 = device→host（cudaMemcpy）+ MPI 收发 + host→device。实测 4MB 仅 2736 MB/s、延迟 1532μs——比 C2C **带宽骤降、延迟激增**，因为多了 GPU↔Host 中转开销。

**④ GPU↔GPU CUDA-aware MPI**：直接把 device 指针给 `MPI_Isend(d_src,...)`/`MPI_Irecv(d_rcv,...)`，删掉显式中转步骤。实测 4MB **3211 MB/s**（比传统 MPI 同规模 **提升 17%**），且代码大幅简化。同节点内两卡（同 PCIe）自动走 P2P，4MB 达 **6487 MB/s**。

**⑤ 调整分块大小**：MVAPICH2 自动把大消息按 `MV2_CUDA_BLOCK_SIZE` 切块（默认 256KB）以重叠节点间通信与 host-device 通信。实测 4MB 消息下把块调到 512KB 后两节点间性能再升（3384 vs 3211 MB/s）——最优块大小依赖互联带宽/延迟/平台特性，需实测。

| 场景（4MB 消息，Fermi M2090） | 带宽 MB/s | 备注 |
|---|---|---|
| CPU→CPU（传统 MPI） | 6328 | 基线 |
| GPU→GPU 传统 MPI | 2737 | 经 host 中转，最慢 |
| GPU→GPU CUDA-aware MPI（跨节点） | 3211 | +17% vs 传统 |
| GPU→GPU CUDA-aware MPI（同节点） | 6488 | 自动走 P2P |

## 九、GPUDirect RDMA

> "GPUDirect facilitates peer-to-peer device memory access … without staging data through CPU memory. The RDMA feature … enables third-party devices such as … Network Interface Cards, and InfiniBand adapters to directly access GPU global memory, significantly decreasing latency."

GPUDirect 三代演进：
- **v1（CUDA 3.1）**：InfiniBand 设备与 GPU 共享同一 pinned buffer（仍要 1 次 GPU→shared buffer 拷贝）。
- **v2（CUDA 4.0）**：加 P2P API + UVA（本章前述），提升单节点多 GPU 性能与生产率。
- **v3（CUDA 5.0）**：加 **RDMA**——经标准 PCIe 适配器在跨节点 GPU 间建 InfiniBand 直连，**无需 host CPU 介入**，降 CPU 开销与延迟。

**实测（Kepler K40 ×2，Mellanox Connect-IB，MVAPICH2-GDR）**：64MB 消息，加 GPUDirect RDMA 后带宽 11540 vs CUDA-aware MPI 10240 MB/s（**提升约 13%**）。

> "as you accelerate the computational portion of your application using CUDA, the I/O of your application will rapidly become a bottleneck … GPUDirect offers a straightforward solution … by reducing latency between GPUs."

⚠️ **关键洞见**：用 CUDA 把计算加速后，**I/O 会迅速成为整体瓶颈**；GPUDirect 通过降延迟直接解此问题。

**GPU Affinity（亲和性）**：就像绑 CPU 核叫 CPU affinity，绑 MPI 进程到特定 GPU 叫 GPU affinity，须在 `MPI_Init` 前做。常用 `MV2_COMM_WORLD_LOCAL_RANK` 取进程节点内本地 ID 来选卡：

```c
int local_rank = atoi(getenv("MV2_COMM_WORLD_LOCAL_RANK"));
cudaGetDeviceCount(&n_devices);
int device = local_rank % n_devices;     // 轮询分配到各 GPU
cudaSetDevice(device);
...
MPI_Init(argc, argv);
```

若先设 CPU affinity 再设 GPU affinity，无法保证两者**物理邻近**——可用 **hwloc（Portable Hardware Locality）** 分析拓扑，把进程绑到与所分配 GPU 最优邻近的 CPU 核上（代码见原文 p.411），确保 host↔device 延迟/带宽不被劣化。

## 十、本章小结（Summary）

> "With a balanced workload that uses computation to hide communication latency, near-linear performance gains can be realized."

多 GPU 两种配置：单节点多设备、跨节点 GPU 集群。本章覆盖两粒度下的管理与执行 API：
- **CUDA-aware MPI**（如 MVAPICH2）直接把 device 内存交给 MPI，大幅简化开发并提升集群性能。
- **GPUDirect** 支持同节点/跨节点设备间不经 CPU 内存的直交换；**RDMA** 让 NIC/SSD/InfiniBand 直访 GPU 全局内存，显著降低延迟。
- 只要**负载均衡 + 用计算隐藏通信延迟**，即可实现近线性加速（simple2DFD 已验证 96% 效率）。

> 📖 本章结束。练习要点（Exercises）：用 event 替代 CPU timer 测 simpleMultiGPU 时间；对比单/双 GPU 的 nvprof/nvvp 行为；把 simpleP2P 单向拷贝改异步；用 `cudaMemcpyPeerAsync` 做 ping-pong；重排 simple2DFD 的 halo/内部计算顺序并解释性能变化；解释 CPU/GPU affinity 对执行时间的影响；描述 GPUDirect RDMA 三代演进与软硬件要求。

> 📖 本章核心结论
> 1. **多 GPU 的两大动机**：数据放不下单卡 / 用并发多任务提吞吐；通信拓扑分 PCIe（节点内）与 InfiniBand（跨节点）。
> 2. **设备管理三件套**：`cudaGetDeviceCount` → `cudaSetDevice`（低开销、不引发同步）→ 之后所有操作作用于该设备；多卡靠单线程循环 + 异步 API 调度，可安全随时切卡。
> 3. **P2P 直连**：`cudaDeviceCanAccessPeer`/`EnablePeerAccess`（单向、需同 PCIe root node、64-bit+CC≥2.0）；`cudaMemcpyPeerAsync` 不经 host 中转；配 UVA 后 kernel 可直接解引用别卡指针（但小传输过多有性能代价）。
> 4. **重叠计算与通信**：halo 模式（有限差分）用 halo 流 + 内部流两流重叠，靠「内部计算耗时 > halo 通信耗时」实现近线性扩展（96% 效率）。
> 5. **集群扩展**：传统 MPI 需 host 中转（最慢）；CUDA-aware MPI 直传 device 指针（+17%）；GPUDirect RDMA 进一步跨节点直连（再 +13%），且三代演进持续降延迟。
> 6. **亲和性是隐藏陷阱**：CPU affinity 防 OS 迁移保局部性；GPU affinity（用 `MV2_COMM_WORLD_LOCAL_RANK` + hwloc）保证进程绑到与 GPU 最优邻近的 CPU 核，否则 host↔device 带宽/延迟劣化。
> 7. **黄金法则**：用计算隐藏通信延迟 + 负载均衡 → 近线性加速；避免经 host 中转、关注传输量与传输次数。
