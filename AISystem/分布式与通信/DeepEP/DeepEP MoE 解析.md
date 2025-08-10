---
dateCreated: 2025-08-08
dateModified: 2025-08-08
---
# 1. DeepEP：面向 MoE 与专家并行的高性能通信库技术解析

DeepEP 作为 DeepSeek 团队开源的通信库，其核心设计围绕混合专家（MoE）和专家并行（EP）架构的通信需求展开，通过硬件感知优化、场景化内核设计和资源调度创新，解决了大模型训练与推理中的通信瓶颈。其特性包括：

1. **高吞吐量通信**：提供一系列针对非对称带宽转发进行优化的内核，可将数据从 NVLink 域转发到 RDMA 域，适用于训练和推理预填充任务，并且支持流多处理器（Streaming Multiprocessors, SM）数量控制。
2. **低延迟推理**：包含一组纯 RDMA 低延迟内核，用于对延迟敏感的推理解码任务，还引入了基于钩子的通信 - 计算重叠方法，且不占用 SM 资源。
3. **多场景支持**：提供普通内核，适用于模型训练和推理预填充阶段（无反向传播部分）；低延迟内核则适用于推理解码阶段。

以下从核心特性、技术实现和场景适配三个维度展开解析：

## 1.1 核心定位：为 MoE/EP 量身定制的通信优化

MoE 和专家并行的核心通信挑战在于 **“动态、多对多” 的数据交互**：每个输入 Token 需被分发到多个专家（跨 GPU），专家处理后结果需回传合并（All-to-All 通信）。传统通信库（如 NCCL）针对对称通信（如 AllReduce）优化，难以适配 MoE 的动态性，而 DeepEP 通过三个核心设计实现针对性优化：

- **动态路由感知**：内核设计适配门控网络的动态专家选择（如 Top-K 路由），避免无效数据传输；
- **硬件链路适配**：根据底层硬件（NVLink/RDMA/PCIe）特性优化数据路径，最大化带宽利用率；
- **资源重叠调度**：将通信与计算（专家推理）在时间和硬件资源上重叠，隐藏通信延迟。

## 1.2 核心技术特性与实现细节

### 1.2.1 高效全连接 GPU 内核：MoE 分发与合并的性能基石

MoE 的 “分发（Dispatch）” 和 “合并（Combine）” 是通信密集型操作（对应 Token 分发到专家、专家结果回传），DeepEP 通过定制化 GPU 内核实现高吞吐量与低延迟：

- **分发内核（Dispatch Kernel）**：

基于 “地址预计算 + 多通道并行” 机制，通过 `rank_prefix_matrix` 和 `channel_prefix_matrix` 预先规划每个 Token 在目标 GPU 的存储地址，避免运行时动态寻址开销；同时利用 GPU Warp 级并行（32 线程组）分块传输数据，单通道吞吐量可达 NVLink 理论带宽的 90% 以上。

- **合并内核（Combine Kernel）**：

依托 `send_head` 回执单实现精准回收，通过 24 个 Warp 线程（协调员 + 处理员）并行读取专家结果并累加，支持原子操作（如 `atomicAdd`）确保多线程写入安全，合并阶段延迟可降低至微秒级。

**性能优势**：在 8×A100 GPU 节点中，MoE 分发 / 合并的吞吐量较 NCCL 提升 3 倍，延迟降低 60%。

### 1.2.2 低精度计算支持：FP8 适配与灵活性

为适配大模型训练 / 推理的内存与算力需求，DeepEP 原生支持 FP8 等低精度格式：

- **低精度通信优化**：内核支持 FP8 数据的直接传输与合并，无需先转换为 FP32（传统库常需格式转换），减少内存带宽占用（FP8 仅为 FP16 的 1/2、FP32 的 1/4）；
- **精度安全机制**：在合并阶段（结果累加）采用 “FP8 存储 + FP16 计算” 的混合策略，避免低精度累加导致的精度损失，平衡性能与模型精度。

**适用场景**：千亿参数 MoE 模型训练（如 DeepSeek-V3）中，FP8 支持可减少 50% 通信数据量，同时保持模型收敛精度。

### 1.2.3 分组受限门控适配：非对称域带宽的定向优化

为配合 DeepSeek-V3 提出的 “分组受限门控” 算法（将专家划分为组，Token 仅在组内选择专家，减少跨组通信），DeepEP 设计了**非对称域带宽转发内核**：

- **域感知路由**：根据硬件域划分（如 “节点内 NVLink 域”“跨节点 RDMA 域”）优化数据路径 —— 节点内通信优先使用 NVLink（高带宽），跨节点通信使用 RDMA（低延迟），避免 “用 NVLink 带宽跑跨节点数据” 或 “用 RDMA 处理节点内通信” 的资源错配；
- **分组通信聚合**：将同一组专家的通信请求聚合为批量传输（而非单 Token 单独发送），减少通信协议开销（如 RDMA 的连接建立 / 释放成本）。

**效果**：在分组受限门控场景中，跨域通信吞吐量提升 40%，预填充阶段（训练前数据加载）耗时减少 30%。

### 1.2.4 SM 资源管理：硬件级资源利用率控制

GPU 的流式多处理器（SM）是计算与通信的核心资源，DeepEP 通过内置控制功能避免资源竞争：

- **SM 分区调度**：将 SM 划分为 “计算区”（专家推理）和 “通信区”（数据传输），例如用 50% SM 运行专家内核，50% SM 处理通信（通过 `num_sms/2` 配置通道数）；
- **动态资源调整**：根据实时负载（如专家计算耗时、通信量）调整 SM 分配比例 —— 若通信拥堵，自动增加通信区 SM 数量；若计算耗时更长，则减少通信区资源。

**优势**：避免 “通信抢占计算 SM” 或 “计算占用通信资源” 导致的性能波动，资源利用率稳定在 90% 以上。

### 1.2.5 推理场景优化：RDMA 内核与通信计算重叠

针对推理（尤其是解码阶段）的超低延迟需求，DeepEP 做了两点关键优化：

- **RDMA 专用内核**：解码阶段（生成单 Token）通信量小但延迟敏感，DeepEP 集成仅支持 RDMA 的轻量内核（移除 NVLink 适配等冗余逻辑），将跨节点通信延迟从数十微秒降至 5μs 以内；
- **基于钩子的重叠机制**：通过 CUDA Stream 和钩子（Hook）函数，在专家计算（如 Transformer 层推理）的 “空闲间隙”（如内存访问等待）触发通信操作，且通信不占用 SM 资源（由 GPU DMA 引擎处理），实现 “计算与通信零重叠耗时”。

**效果**：MoE 模型推理的解码延迟降低 40%，满足实时对话等低延迟场景需求。

## 1.3 总结：DeepEP 的核心价值与适用场景

DeepEP 通过 “场景感知的内核设计 + 硬件链路的深度适配 + 资源调度的精细化控制”，成为 MoE 和专家并行架构的 “通信加速器”。其核心价值在于：

- **针对性**：从底层适配 MoE 的动态 All-to-All 通信，而非用通用通信接口 “勉强支撑”；
- **硬件亲和**：充分利用 NVLink、RDMA、SM 等硬件特性，而非停留在软件协议优化；
- **场景覆盖**：兼顾训练（高吞吐量、低精度）与推理（超低延迟、资源重叠）的差异化需求。

目前，DeepEP 已在 DeepSeek-V3 等大模型中验证了有效性，未来随着 MoE 成为超大模型标配，其针对非对称通信、低精度传输的优化将成为高性能分布式训练的关键支撑。

# 2. DeepEP 单节点 MoE 通信机制：高效 All-to-All 通信

本文将介绍 DeepEP 节点内的高效 All-to-All 通信机制。

## 2.1 背景与核心问题

MoE（混合专家模型）已成为大模型架构的主流选择，其通过动态选择 “专家网络” 扩展模型容量，但需解决 All-to-All 通信的效率瓶颈 —— 传统点对点通信易导致网络拥堵，成为训练 / 推理的性能短板。DeepEP 作为 DeepSeek 团队开源的高性能通信库，通过重新设计通信范式，在单节点多 GPU 场景下实现了高效的 All-to-All 通信。

单节点内的多张 GPU（如 8 张）通过 NVLink 高速互联，核心通过**规划 - 派送 - 回收**三个阶段实现无锁、高并行的数据流管理。

## 2.2 核心通信阶段与技术实现

### 1. `notify_dispatch`：全局规划

该阶段的核心目标是通过全局协同生成精确的通信计划，避免后续数据传输的地址冲突与混乱。流程分为四步，依赖轻量级同步机制确保一致性。

- **Token 数量申报（寄件清单生成）**

每个 GPU（Rank）本地生成 `num_tokens_per_rank` 数组（长度为总 Rank 数），其中第 j 个元素表示当前 Rank 需***发送***给 Rank j 的 Token 数量。这一步为后续全局规划提供 “单节点寄件需求”。

- **数据交换与全局矩阵构建**

所有 Rank 通过 NVLink 交换 `num_tokens_per_rank`，形成二维矩阵 `per_rank_buffer`：行表示某 Rank 的发送分布，列表示某 Rank 的接收来源分布。该矩阵是全局通信的 “总览表”。

- **收件清单提取**

每个 Rank 从 `per_rank_buffer` 中提取自身对应的列，生成 `num_recv_tokens_per_rank` 数组，记录 “从各 Rank ***接收***的 Token 数量”。这是接收端分配内存、定位数据的关键依据。

- **全局同步（自定义 Barrier）**

为确保所有 Rank 完成规划，DeepEP 实现了基于原子操作的轻量级 Barrier（而非传统 `__syncthreads()`）：

- 每个 Rank 通过 `atomicAdd` 在本地标记 “已就绪”，并通过 `atomicSub` 向其他 Rank 发送 “就绪信号”；
- 当某 Rank 收到所有其他 Rank 的信号（本地计数值归 0），则确认全局规划完成。

该设计避免了重量级同步的性能开销，支持跨 Block 高效协同。

![[全局 token 表.png]]

### 2. `dispatch`：异步派送

基于 `notify_dispatch` 生成的计划，Token 数据通过多通道并行传输，最终被精准写入目标 GPU 的存储区。核心依赖 “地址预计算” 与 “异步流控” 提升效率。

每个 Rank 自己都具有的信息：

1. `rank_prefix_matrix`：全局投递总账，它告诉你每个 Rank 应该把多少 token 发给谁
2. `channel_prefix_matrix`：本地分渠道账本，细化了当前 Rank 在每个 Channel 上要发给每个目标 Rank 的 token 分布

#### **核心组件与分工**

在混合专家模型（MoE）的分布式训练中，DeepEP 的通信机制通过**目的地清单**、**多通道传输系统**、**临时收件篮**、**派送小组**和**最终存储货架**五个核心组件，实现了高效的全对全（All-to-All）数据传输。以下从技术原理、实现细节和协同机制三个层面展开深度解析：

##### 一、目的地清单（`is_token_in_rank`）：精准路由的核心依据

1. **数据结构与生成逻辑**

`is_token_in_rank` 是一个二维布尔数组，形状为 `(num_tokens, num_ranks)`，其中 `is_token_in_rank[i][j]` 表示第 `i` 个 Token 是否需要发送到第 `j` 个 Rank。该数组由门控网络的输出（`topk_idx`）动态生成：

- **生成步骤**：

1. **门控决策**：门控网络为每个 Token 选出 `topk` 个专家（Expert），生成 `topk_idx` 张量（形状为 `(num_tokens, num_topk)`）；
2. **专家到 Rank 映射**：将专家 ID 映射到对应的 GPU Rank（如专家 0-3 映射到 Rank 0，专家 4-7 映射到 Rank 1）；
3. **布尔标记**：遍历每个 Token 的 `topk_idx`，若专家对应的 Rank 为 `j`，则 `is_token_in_rank[i][j] = True`。
4. **核心作用**

- **过滤无效传输**：发送端仅处理 `is_token_in_rank` 为 `True` 的 Token，减少冗余数据搬运；
- **负载均衡依据**：通过统计 `is_token_in_rank` 中 `True` 的数量，生成 `num_tokens_per_rank` 数组，用于后续全局通信规划。

##### 二、多通道传输系统（Channel）：并行通信的硬件适配层

1. **通道数量的硬件适配逻辑**

Channel 数量通常设为 `num_sms / 2`（`num_sms` 为 GPU 的流式多处理器数量），其核心原因是：

- **SM 资源利用率优化**：每个 SM 包含多个 CUDA 核心，将通道数设为 `num_sms/2` 可确保每个通道独占部分 SM 资源，避免线程竞争；
- **计算与通信的平衡**：保留一半 SM 用于计算（如专家网络推理），另一半用于通信，实现计算与通信的部分重叠。

1. **通道间的负载分配**

- **动态分片策略**：根据 `num_tokens_per_rank` 和 `is_token_in_rank`，将 Token 按通道数均分，每个通道独立处理一部分数据；
- **跨通道协作**：通过 `channel_prefix_matrix` 记录各通道的 Token 分布，避免地址冲突。

##### 三、临时收件篮（`channel_x_buffers`）：无锁流控的缓冲区

1. **数据结构与指针机制**

- **环形缓冲区设计**：每个通道对应一个定长缓冲区（如 4KB），通过 `head`（已处理数据边界）和 `tail`（新数据边界）指针管理数据流动；
- **无锁同步机制**：
- **发送端**：通过原子操作 `atomicCAS` 更新 `tail` 指针，确保多线程写入安全；
- **接收端**：通过 `atomicCAS` 更新 `head` 指针，释放已处理空间。

1. **流控策略**

- **空间预判**：发送端在写入前通过 `tail - head` 判断剩余空间，不足时等待；
- **数据分块**：当数据量超过缓冲区大小时，自动拆分为多个块（Chunk）传输，避免阻塞。

##### 四、派送小组（`Sender Warp Group`）：线程级并行的执行单元

1. **线程分配与协作机制**

- **Warp 级任务划分**：每个通道由多个 Warp（32 线程 / 组）组成，例如 768 线程为 8 个 Rank 服务时，每个 Rank 分配 3 个 Warp（96 线程）；
- **流水线作业**：

1. **地址计算线程**：根据 `rank_prefix_matrix` 和 `channel_prefix_matrix` 预计算目标地址；
2. **数据打包线程**：将 Token 按通道分片，生成待发送的数据包；
3. **传输执行线程**：通过 NVLink/RDMA 将数据写入目标缓冲区。
4. **性能优化关键点**

- **指令级并行**：利用 GPU 的 SIMT（单指令多线程）架构，同一 Warp 内的线程执行相同指令，减少分支预测开销；
- **异步提交**：通过 CUDA 流（Stream）将通信任务异步提交，与计算任务重叠。

##### 五、最终存储货架（`recv_x`）：精准定位的内存布局

1. **地址计算的数学模型**

- **全局前缀和矩阵**：

1. **`rank_prefix_matrix`**：记录每个 Rank 的 Token 累计数量，例如 `rank_prefix_matrix[j]` 表示前 `j` 个 Rank 的总 Token 数；
2. **`channel_prefix_matrix`**：记录每个通道的 Token 累计数量，例如 `channel_prefix_matrix[c]` 表示前 `c` 个通道的总 Token 数；

- **绝对地址公式**：`\text{address} = \text{rank_prefix_matrix}[j] + \text{channel_prefix_matrix}[c] + \text{local_offset}$` 其中 `j` 为目标 Rank，`c` 为通道编号，`local_offset` 为通道内的相对偏移。

1. **数据写入的原子性保障**

- **原子写操作**：通过 `cudaMemcpyAsync` 或 NVSHMEM 的 `put` 操作，将数据直接写入 `recv_x` 的目标地址，确保单线程写入原子性；
- **校验机制**：接收端通过 CRC 校验或哈希值比对，确保数据完整性。

##### 六、五大组件的协同机制与性能优势

- **全流程协同示例**

1. **数据过滤**：`is_token_in_rank` 筛选需发送的 Token；
2. **并行传输**：多通道将数据分片，派送小组通过 NVLink/RDMA 写入临时收件篮；
3. **流控与回收**：接收端通过 `head/tail` 指针动态调整缓冲区，完成后将数据搬运至 `recv_x`；
4. **地址定位**：预计算的绝对地址确保数据零错误写入。

- **性能提升的核心原因**

1. **硬件利用率最大化**：通过 `num_sms/2` 的通道设计，平衡计算与通信资源；
2. **无锁流控**：`head/tail` 指针结合原子操作，避免重量级同步开销；
3. **零拷贝传输**：直接通过 NVLink/RDMA 写入目标显存，绕过 CPU 中转。

##### 七、技术细节与优化点总结

| **组件** | **核心技术** | **性能指标**（H800 GPU+InfiniBand）|

| ------- | ------------------ | ----------------------------- |

| 目的地清单 | 门控网络 + 布尔数组 | 过滤无效数据量达 60%+ |

| 多通道传输系统 | 动态 SM 分配 + 通道分片 | 节点内带宽 153 GB/s，跨节点 47 GB/s |

| 临时收件篮 | 无锁环形缓冲区 + 原子操作 | 流控延迟 < 100 ns |

| 派送小组 | Warp 级并行 + SIMT 指令 | 线程利用率 > 95% |

| 最终存储货架 | 前缀和矩阵 + 原子写操作 | 地址计算误差率 < 0.01% |

#### 跨 GPU 数据传输

在 MoE 模型的分布式训练中，DeepEP 通过**三阶段地址计算与流控机制**实现了高效的跨 GPU 数据传输。这一过程融合了数学规划、无锁同步和内存预取技术，是分布式系统工程的典范。以下从原理、实现和性能三个维度展开深度解析：

##### 一、绝对坐标计算：数学规划的确定性寻址

1. **双账本系统的数学模型**

DeepEP 通过两个前缀和矩阵实现地址预计算：

- **全局投递总账（`rank_prefix_matrix`）**

形状为 `[num_ranks][num_ranks]`，其中 `rank_prefix_matrix[i][j]` 表示**前 i 个 Rank 发送给 Rank j 的 Token 总数**。数学定义为：rank_prefix_matrix[i][j]=∑k=0i−1​num_tokens_per_rank[k][j]

例如，`rank_prefix_matrix[3][5]` 表示 Rank 0~2 发送给 Rank 5 的 Token 总数。

- **本地渠道账本（`channel_prefix_matrix`）**

形状为 `[num_ranks][num_channels]`，其中 `channel_prefix_matrix[j][k]` 表示**Rank j 中前 k 个通道接收的 Token 总数**。数学定义为：channel_prefix_matrix[j][k]=∑c=0k−1​channel_tokens_per_rank[j][c]

例如，`channel_prefix_matrix[5][3]` 表示 Rank 5 的通道 0~2 接收的 Token 总数。

1. **绝对地址计算公式**

对于 Rank i 的 Channel k 发送的第 t 个 Token，其在 Rank j 的 `recv_x` 中的地址为：address=rank_prefix_matrix[i][j]+channel_prefix_matrix[j][k]+t

**关键点**：

- 全局偏移（`rank_prefix_matrix`）确保不同 Rank 发送的数据不会重叠；
- 通道偏移（`channel_prefix_matrix`）确保同一 Rank 内不同通道的数据不会重叠；
- 本地序号（`t`）确保同一通道内的数据按序排列。

##### 二、临时收件篮机制：无锁流控的工程实现

1. **环形缓冲区设计**

每个通道的 `channel_x_buffers` 是固定大小的环形缓冲区，包含：

- **数据区**：存储 Token 内容（如 Embedding 向量）；
- **元数据区**：存储 Token 的绝对地址（来自步骤 1 的计算结果）。

1. **Tail 指针的原子更新**

发送端通过原子操作更新 Tail 指针：

```c

// 伪代码：发送端写入逻辑

size_t required_space = calculate_required_space(token);

size_t current_tail = atomic_load(&tail);

size_t new_tail = current_tail + required_space;

  

// 检查空间是否足够

if (new_tail - head <= buffer_size) {

// CAS操作确保原子性

if (atomic_compare_exchange_strong(&tail, &current_tail, new_tail)) {

// 写入数据到 [current_tail, new_tail) 区域

write_data_to_buffer(token, current_tail);

}

}

```

1. **数据分块策略**

当 Token 大小超过缓冲区剩余空间时，采用分块传输：

1. 将 Token 拆分为多个固定大小的 Chunk（如 256 字节）；
2. 每个 Chunk 附带元数据（如所属 Token ID、绝对地址偏移量）；
3. 接收端按 Chunk 重组完整 Token。

##### 三、数据归位与内存屏障

1. **Receiver 的触发机制**

接收端通过轮询 Tail 指针变化来检测新数据：

```c

// 伪代码：接收端读取逻辑

size_t last_seen_tail = 0;

while (1) {

size_t current_tail = atomic_load(&tail);

if (current_tail > last_seen_tail) {

// 有新数据到达，处理 [last_seen_tail, current_tail) 区域

process_new_data(last_seen_tail, current_tail);

last_seen_tail = current_tail;

}

}

```

1. **精准归位与 Head 更新**

接收端从缓冲区读取数据后：

1. 解析元数据中的绝对地址；
2. 通过 `cudaMemcpyAsync` 将数据写入 `recv_x` 的对应位置；
3. 原子更新 Head 指针释放缓冲区空间。
4. **内存屏障的应用**

为确保数据可见性，在关键操作间插入内存屏障：

```c

// 写入操作完成后

__threadfence_system(); // 确保所有内存写入完成

atomic_store(&tail, new_tail); // 发布新的Tail值

  

// 读取操作前

size_t current_tail = atomic_load(&tail);

__threadfence_system(); // 确保读取到最新的内存状态

```

##### 四、性能优化与工程权衡

1. **预取技术**

发送端在计算地址的同时，通过 `__ldg()` 指令预取数据到缓存，减少内存延迟：

```c

// 预取数据到L1缓存

__prefetch_cg(&token_data[token_idx]);

```

1. **通道间负载均衡**

通过动态调整 `channel_tokens_per_rank`，确保各通道处理的 Token 数量接近：channel_tokens_per_rank[j][k]=num_channelsnum_tokens_per_rank[j]​

1. **实测性能指标**

在 8×A100 GPU 系统中，该机制实现：

- 单通道吞吐量：257 GB/s（理论峰值的 92%）；
- 地址计算延迟：<15 ns/Token；
- 缓冲区利用率：>98%（通过自适应 Chunk 大小）。

##### 五、技术对比与演进

| **特性** | **DeepEP 机制** | **传统 NCCL All-to-All** |

| -------- | ---------------- | ---------------------- |

| 地址计算方式 | 预计算绝对地址 | 运行时动态分配 |

| 同步机制 | 原子操作 + 内存屏障 | 全局 Barrier |

| 数据传输模式 | 直接显存访问（P2P/IPC）| CPU 中转 + 集体通信 |

| 内存使用效率 | 环形缓冲区（利用率 > 98%）| 固定大小缓冲区（易碎片化）|

| 扩展性（节点数）| >1024 节点 | 通常 < 128 节点 |

这一机制使 DeepEP 在万亿参数模型训练中，将通信开销占比从传统方法的 42% 降至 13%，显著提升了分布式训练效率。

- **数据传输流程**

1. **地址计算**：发送端根据 `rank_prefix_matrix`（全局投递总账）和 `channel_prefix_matrix`（本地渠道账本），计算 Token 在目标 GPU 最终存储区（`recv_x`）的绝对地址，确保 “一次投递到位”；
2. **异步流控**：发送端通过比较 `head` 和 `tail` 判断收件篮剩余空间，足够时才分块（Chunked）写入数据，并推进 `tail` 指针通知接收端；
3. **精准写入**：接收端检测到 `tail` 更新后，从收件篮取出数据，按预计算地址写入 `recv_x`，并推进 `head` 释放缓冲区空间。

#### 底层通信支持

在 DeepEP 的通信机制中，NVIDIA 的**P2P（Peer-to-Peer）技术**和 CUDA 的**IPC（Inter-Process Communication）机制**是底层核心支撑。这两种技术突破了传统 “显式通信”（如 NCCL 的 Broadcast、AllReduce）的范式，实现了 “直接显存操作” 的高效通信模式。以下从技术原理、实现细节、与 DeepEP 的结合三个层面展开详解：

##### 一、NVIDIA P2P 技术：GPU 间显存直接访问的硬件基石

P2P 技术的核心是**允许一个 GPU 直接访问另一个 GPU 的物理显存**，无需经过 CPU 中转或显存拷贝。这种能力依赖硬件互联（NVLink 或 PCIe）和软件层的地址映射，是跨 GPU 通信的 “物理通道”。

1. 硬件基础：NVLink 与 PCIe 的角色

P2P 通信的性能由 GPU 间的物理链路决定，主流场景有两种：

- **NVLink 互联**：NVIDIA 专属的高速互联技术（如 H100 支持 6 条 NVLink，单链路带宽 400GB/s），延迟 <1μs，是多 GPU 服务器的首选。多个 GPU 通过 NVLink 形成 “全互联拓扑”（如 8 卡服务器内任意两卡直接相连），支持无阻塞的 P2P 访问。
- **PCIe 4.0/5.0 互联**：通用总线技术（PCIe 4.0 单通道带宽 32GB/s），延迟约 5-10μs。若 GPU 通过 PCIe 交换机互联（如 2 卡通过 PCIe Switch 连接），也可支持 P2P，但性能弱于 NVLink。

**关键前提**：需通过 `cudaDeviceCanAccessPeer` 接口验证 P2P 支持性 —— 只有硬件链路允许时，才能开启互访权限。

1. 软件层：地址映射与权限管理

P2P 并非 “无条件访问”，需通过软件层完成 “显存地址可见性” 配置：

- **步骤 1：开启互访权限**

通过 `cudaSetDevice` 切换到目标 GPU 后，调用 `cudaDeviceEnablePeerAccess` 开启对另一 GPU 的访问权限：

```cpp

// 允许GPU 0访问GPU 1的显存

cudaSetDevice(0);

cudaDeviceEnablePeerAccess(1, 0); // 第二个参数为flags，通常设0

```

此操作会在 GPU 0 的地址空间中注册 GPU 1 的显存映射表，使 GPU 0 能 “看到” GPU 1 的显存地址。

- **步骤 2：直接显存操作**

开启权限后，GPU 0 可通过指针直接读写 GPU 1 的显存，例如：

```cpp

// 在GPU 1上分配显存

cudaSetDevice(1);

float* gpu1_mem;

cudaMalloc(&gpu1_mem, 1024 * sizeof(float));

// GPU 0直接写入GPU 1的显存

cudaSetDevice(0);

kernel_write_to_peer<<<1, 32>>>(gpu1_mem); // 核函数直接操作GPU 1的指针

```

这里的 `gpu1_mem` 对 GPU 0 而言是 “远程指针”，但通过 P2P 映射，写入操作会被硬件直接路由到 GPU 1 的显存。

1. 核心优势：突破传统通信的 “中转瓶颈”

传统跨 GPU 通信（如 NCCL 的 Send）需经过 “本地显存→CPU 缓存→远程显存” 的中转，而 P2P 实现了 “远程显存直接读写”：

- **延迟降低**：省去 CPU 干预和数据拷贝，延迟从数十 μs 降至 1μs 以内（NVLink 场景）；
- **带宽提升**：直接利用 NVLink/PCIe 的物理带宽，无协议封装开销（如 NCCL 的集体通信协议会消耗 10-20% 带宽）；
- **灵活性**：支持任意粒度的访问（如单次写入 1 个 float 或 1 个 Tensor），无需按 “消息包” 传输。

##### 二、CUDA IPC 机制：跨进程显存共享的 “门禁系统”

IPC（Inter-Process Communication）机制的核心是**允许不同进程（甚至不同 GPU 上的进程）共享同一块显存**，本质是 “显存访问权限的跨进程传递”。它解决了 “多进程场景下 P2P 访问” 的痛点 —— 不同进程的 GPU 指针无法直接互通，需通过 IPC 生成 “跨进程可见的显存句柄”。

1. 核心问题：进程隔离与显存指针的 “私有性”

在分布式训练中，每个 GPU 通常对应一个独立进程（如 8 卡训练对应 8 个进程）。进程的内存空间是隔离的：GPU 0 进程中的指针 `ptr`，在 GPU 1 进程中是 “无效地址”（地址空间不互通）。

IPC 的作用就是打破这种隔离：通过生成 “显存句柄”（类似 “门禁卡”），让其他进程能基于句柄生成 “本地有效指针”，从而访问同一块显存。

1. 实现流程：从 “显存分配” 到 “跨进程访问”

IPC 的使用需严格遵循 “句柄生成→句柄传递→指针解析” 三步，以 “GPU 0 进程分配显存，GPU 1 进程访问” 为例：

- **Step 1：分配显存并生成句柄（GPU 0 进程）**

在 GPU 0 上分配显存后，调用 `cudaIpcGetMemHandle` 生成句柄（`cudaIpcMemHandle_t`）—— 这是一块显存的 “唯一标识”，可通过进程间通信（如共享内存、TCP）传递给其他进程：

```cpp

// GPU 0进程：分配显存并生成句柄

cudaSetDevice(0);

float* gpu0_mem;

cudaMalloc(&gpu0_mem, 1024 * sizeof(float)); // 分配显存

cudaIpcMemHandle_t handle;

cudaIpcGetMemHandle(&handle, gpu0_mem); // 生成句柄（门禁卡）

```

- **Step 2：传递句柄（进程间通信）**

通过传统进程通信方式（如 MPI、共享内存）将 `handle` 发送给 GPU 1 进程。句柄本身是轻量数据（约 64 字节），传递成本可忽略。

- **Step 3：解析句柄生成本地指针（GPU 1 进程）**

GPU 1 进程接收句柄后，调用 `cudaIpcOpenMemHandle` 生成 “本地有效指针”—— 该指针指向 GPU 0 的显存，且在 GPU 1 进程中可直接读写：

```cpp

// GPU 1进程：解析句柄并访问显存

cudaSetDevice(1);

float* gpu0_mem_in_gpu1; // 本地指针（指向GPU 0的显存）

cudaIpcOpenMemHandle(&gpu0_mem_in_gpu1, handle, cudaIpcMemLazyEnablePeerAccess);

// 直接写入GPU 0的显存（通过本地指针）

gpu0_mem_in_gpu1[0] = 1.0f; // 实际写入的是GPU 0的物理显存

```

- **Step 4：释放资源**

通信结束后，需调用 `cudaIpcCloseMemHandle` 关闭指针，释放访问权限。

1. 关键特性：“零拷贝” 与延迟优势

IPC 共享的显存是 “物理同一块”，而非拷贝：

- GPU 1 进程写入 `gpu0_mem_in_gpu1`，GPU 0 进程可直接从 `gpu0_mem` 读取到最新值（无需同步）；
- 避免了传统通信的 “发送→接收→拷贝” 三步，延迟仅取决于 P2P 链路（NVLink 下 < 1μs）。

##### 三、DeepEP 如何利用 P2P+IPC 实现 “无协议通信”

传统分布式通信依赖 NCCL 等库，需经过 “数据打包→协议封装→集体同步→数据解包” 等步骤（如 AllReduce 需多轮通信）。而 DeepEP 基于 P2P+IPC，实现了 “直接显存写入”，彻底跳过这些开销。

1. 核心设计：全局显存指针表

DeepEP 在初始化阶段完成以下操作，为后续通信铺路：

- 每个进程通过 IPC 生成自己 GPU 显存的句柄，并广播给所有其他进程；
- 每个进程基于收到的句柄，通过 `cudaIpcOpenMemHandle` 生成 “指向所有 GPU 显存的本地指针”，形成 `buffer_ptrs` 数组（`buffer_ptrs[j]` 表示指向 GPU j 显存的指针）；
- 通过 `cudaDeviceEnablePeerAccess` 开启所有 GPU 间的 P2P 权限，确保指针访问合法。

最终，每个进程都持有 “访问所有 GPU 显存的钥匙”——`buffer_ptrs` 数组，这是后续通信的 “全局地址簿”。

1. 通信过程：从 “显式通信” 到 “直接写入”

以 MoE 中的 Token 跨 GPU 传输为例（Rank i 的 Token 需发送到 Rank j）：

- **传统 NCCL 方式**：需调用 `ncclSend` 和 `ncclRecv`，内部经历 “数据从显存拷贝到通信缓冲区→协议封装→通过 NVLink 发送→接收方解封装→拷贝到目标显存”，多步开销累计延迟 > 10μs。
- **DeepEP 方式**：

1. 根据地址计算得到 Rank j 的 `recv_x` 绝对坐标（如 `addr = 0x123456`）；
2. 直接通过 `buffer_ptrs[j] + addr` 定位到 Rank j 的显存地址；
3. 调用 GPU 核函数，将 Token 数据直接写入该地址（`*(buffer_ptrs[j] + addr) = token_data`）。

整个过程无协议封装、无数据拷贝、无集体同步，延迟仅取决于 P2P 链路（NVLink 下 < 1μs），带宽接近硬件理论峰值（如 NVLink 的 400GB/s）。

1. 优势总结：为什么比传统通信更高效？

| 维度 | 传统 NCCL 通信 | DeepEP（P2P+IPC）|

| ----- | ---------------------- | ---------------------- |

| 数据路径 | 显存→通信缓冲区→目标显存 | 显存→直接写入目标显存 |

| 延迟 | 10-100μs（含协议开销）| <1μs（仅 P2P 链路延迟）|

| 带宽利用率 | 60-70%（协议封装损耗）| >90%（接近硬件峰值）|

| 同步开销 | 需集体同步（如 Barrier）| 无同步（依赖无锁指针流控）|

| 灵活性 | 仅支持预设通信模式（如 AllReduce）| 支持任意地址写入（灵活适配 MoE 等场景）|

##### 四、技术局限性与适用场景

P2P+IPC 虽高效，但并非万能，需注意适用边界：

- **硬件依赖**：仅支持 NVIDIA GPU，且需 NVLink/PCIe 链路支持（低端 GPU 或旧服务器可能不支持）；
- **进程管理**：IPC 句柄需跨进程传递，初始化阶段需额外逻辑（如 MPI 广播句柄）；
- **安全性**：直接显存写入无 “权限校验”，需确保地址计算正确（否则可能写入非法地址导致崩溃）。

因此，DeepEP 的这种设计特别适合**同节点多 GPU（或同集群低延迟互联）的 MoE 场景**——Token 传输是 “多对多” 的 All-to-All 模式，传统通信模式效率低，而 P2P+IPC 的直接写入能最大化带宽利用率。

##### 总结：DeepEP 底层依赖一览

| 类别 | 具体项 | 作用/说明 |

| ----------- | ---------------------------------- | ----------------------------------------------------- |

| **计算/编程框架** | CUDA ≥ 11.0 (SM80) / ≥ 12.3 (SM90) | GPU kernel 运行环境；提供 CUDA IPC 供单机多进程零拷贝共享显存 |

| | PyTorch ≥ 2.1 | Python 前端封装与 Tensor 生命周期管理 |

| **通信库** | **NVSHMEM** | 跨节点 PGAS 统一地址空间；GPU 内直接发起单边 `put/get` 和集合通信，支撑节点间专家并行 |

| | **CUDA IPC** (`cudaIpc*`) | 同一节点不同进程之间零拷贝共享显存 → 节点内专家并行 |

| **通信协议/网络** | **NVLink** (≥ 160 GB/s) | 节点内高速 GPU-GPU 数据转发 |

| | **InfiniBand RDMA** (CX7 400 Gb/s) | 节点间低延迟、高带宽传输；低时延 kernel 直接使用纯 RDMA |

| | **RoCE**（理论兼容，未官方测试）| 作为 IB 的替代 RDMA 传输 |

| **硬件要求** | Ampere/Hopper 架构 GPU | 支持 SM80/SM90 PTX ISA；具备 NVLink 与 RDMA NIC |

| | InfiniBand 交换机 | 支持 Virtual Lane (VL) 流量隔离与自适应路由 |

> [!note] **底层通信支撑：P2P 与 IPC**

> DeepEP 通过 **CUDA IPC + NVLink** 搞定**节点内**高效专家共享，通过 **NVSHMEM + InfiniBand RDMA** 搞定**节点间**专家并行，从而在全链路实现 MoE 的高吞吐、低延迟通信。

DeepEP 无需依赖 NCCL 等传统通信库，核心依赖 NVIDIA 的 P2P（点对点）与 CUDA IPC（进程间通信）机制：

- **P2P**：通过 NVLink/PCIe 实现 GPU 间显存直接访问，无需 CPU 中转；
- **IPC**：GPU 通过 `cudaIpcGetMemHandle` 生成显存 “访问凭证”，其他 GPU 通过 `cudaIpcOpenMemHandle` 获取指针，直接写入目标显存。

最终，每个 GPU 持有指向所有 Rank 通信缓冲区的指针数组（`buffer_ptrs`），实现 “零拷贝” 高效通信。

NVIDIA 的 P2P 技术提供了 “GPU 间显存直接访问” 的硬件能力，CUDA IPC 机制解决了 “跨进程显存共享” 的软件障碍，二者结合形成了 “无中介、低延迟” 的通信基础。DeepEP 的核心创新在于：**基于这两种底层技术，构建了一套 “地址预计算 + 直接显存写入” 的通信范式**，彻底摆脱了传统通信库的协议开销，实现了 MoE 训练中 All-to-All 通信的极致效率。这种 “回归底层硬件能力” 的设计，正是其性能优势的根源。

### 3. `send_head` 与 `combine` 阶段：MoE 通信的 “回执回收” 机制

在 DeepEP 的单节点 MoE 通信体系中，`send_head` 和 `combine` 是完成 “Token 发送→专家处理→结果返回” 闭环的核心环节。前者通过精准记录投递轨迹为回收提供依据，后者通过角色反转和并行协作实现高效的结果聚合，最终解决 MoE 中 “Token 分发后如何准确回收处理结果” 的关键问题。

#### 一、`send_head`：发货回执单的核心作用与数据结构

`send_head` 是 DeepEP 在 `dispatch`（派送）阶段生成的 “高精度投递记录”，本质是 MoE 通信的 “可追溯凭证”，为后续回收提供精确的定位依据。

##### 1. 数据结构与记录内容

`send_head` 是一个二维张量，形状为 `(num_tokens, num_ranks)`，其中 `send_head[token_idx, j]` 存储的核心信息是：

**“第 `token_idx` 个 Token 发送到 Rank j 后，在 Rank j 的 `channel_x_buffers`（临时收件篮）中的具体位置（slot index）”**。

例如：若 `send_head[5, 3] = 10`，表示 “第 5 个 Token 发送到 Rank 3 后，存放在 Rank 3 收件篮的第 10 号位置”。

##### 2. 核心价值：解决 “结果溯源” 难题

MoE 中，每个 Token 会被发送到多个专家（对应多个 Rank）处理，而处理后的结果需要返回给原发送 Rank 并合并。`send_head` 的作用是：

- 记录每个 Token 在目标 Rank 中的 “临时存放地址”，避免回收时因 “不知道结果存在哪里” 而遗漏；
- 为并行回收提供 “精准导航”，无需全局扫描目标 Rank 的内存（否则会产生巨大开销）。

#### 二、`combine` 阶段：角色反转与并行回收机制

`combine` 阶段是 MoE 通信的 “返程环节”：当目标 Rank 的专家（Experts）完成 Token 处理后，需将结果返回给原发送 Rank。此时系统角色发生反转 —— 原发送方（Sender）变为接收方（Receiver），原接收方变为发送方，通过 “回收小组” 的协作完成结果聚合。

##### 1. 回收小组的角色分工

DeepEP 为 `combine` 阶段分配了 24 个 Warp 线程（共 768 个线程，1 个 Warp=32 线程），分为 “协调员” 和 “处理员” 两类角色，各司其职：

|角色|数量|核心职责|

|---|---|---|

|协调员|1 个 Warp（32 线程）|监控所有处理员的工作进度，统一更新 “收件篮” 的 `head` 指针（释放空间），避免并发冲突。|

|处理员|23 个 Warp（736 线程）|根据 `send_head` 的记录，从目标 Rank 的收件篮中读取处理结果，累加（Reduce）到本地 `recv_x`。|

##### 2. 回收流程：从 “回执查询” 到 “结果聚合”

`combine` 阶段的核心逻辑是 “基于 `send_head` 的精准溯源 + 并行累加”，具体流程可分为 4 步：

- **Step 1：查询回执单，定位目标位置**

处理员线程根据本地负责的 `token_idx`，从 `send_head` 中读取目标信息：

```python

# 伪代码：获取第token_idx个Token在Rank j的存放位置

target_slot = send_head[token_idx, j] # j为目标专家所在Rank

```

该操作直接定位结果在目标 Rank 收件篮中的位置，避免盲目搜索。

- **Step 2：等待结果到达，确认数据就绪**

处理员通过监控目标 Rank 收件篮的 `tail` 指针（新数据边界），判断结果是否已写入：

- 若 `tail > target_slot`：表示目标位置的数据已就绪（专家已处理完成并写入）；
- 若 `tail <= target_slot`：循环等待（附带超时保护，避免死锁）。

这一设计通过 “指针比对” 实现轻量同步，无需重量级 Barrier。

- **Step 3：读取结果并累加至本地存储**

当数据就绪后，处理员从目标 Rank 收件篮的 `target_slot` 位置读取处理结果（如专家输出的特征向量），并通过 “原子累加” 写入本地 `recv_x`（最终存储货架）：

```python

# 伪代码：结果累加（以浮点型特征为例）

local_result = recv_x[token_idx] # 本地初始值

remote_result = rank_j_channel_buffer[target_slot] # 从目标Rank读取的处理结果

recv_x[token_idx] = local_result + remote_result # 累加合并

```

此处的 “原子累加” 确保多线程并行写入时的数据一致性（避免结果覆盖）。

- **Step 4：汇报进度，释放资源**

处理员完成当前 Token 的回收后，将自身进度（如 “已处理到第 N 个 Token”）记录到 `warp_channel_head_idx`（进度数组），并由协调员统一同步：

- 协调员周期性收集所有处理员的 `warp_channel_head_idx`，取最小值作为 “全局已完成进度”；
- 当全局进度更新时，协调员推进目标 Rank 收件篮的 `head` 指针，释放已处理的空间（供后续通信复用）。

##### 3. 设计优势：并行效率与无锁同步的结合

`combine` 阶段的设计通过 “角色分工 + 回执定位 + 轻量同步”，解决了传统回收机制的三大痛点：

- **并行效率最大化**：23 个处理员 Warp 并行处理不同 Token 的回收，充分利用 GPU 的多线程算力；协调员专注同步，避免处理员因 “各自更新指针” 导致的冲突。
- **无锁化同步**：通过 `head`/`tail` 指针比对和进度数组，替代传统的 `__syncthreads()` 或 NCCL Barrier，减少等待开销（同步延迟从数十 μs 降至 μs 级）。
- **精准回收零遗漏**：`send_head` 的 “一对一” 记录确保每个 Token 的处理结果都能被追溯，避免因 “地址未知” 导致的数据丢失。

#### 三、总结：从 “闭环” 看 MoE 通信的完整性

`send_head` 和 `combine` 阶段是 DeepEP 实现 MoE 通信闭环的关键：

- `send_head` 通过 “投递记录” 为回收提供 “可追溯性”，是 “精准回收” 的前提；
- `combine` 通过 “角色反转 + 并行协作” 实现结果的高效聚合，是 “闭环通信” 的收尾。

二者结合，既解决了 MoE 中 “Token 分发后结果难以定位” 的问题，又通过并行化和轻量同步最大化回收效率，最终支撑 MoE 模型在分布式训练中 “高效分发 - 准确回收” 的核心需求。这种设计也体现了 DeepEP 的核心思路 ——**“用确定性的规划替代不确定性的搜索，用并行化的协作替代串行化的等待”**。

当 “专家网络” 处理完 Token 后，结果需返回原发送端并合并。该阶段依赖 “派送回执” 实现精准回收与协同。

- **核心依据：发货回执（`send_head`）**

派送阶段，发送端会记录每个 Token 在目标 Rank 收件篮中的位置（`slot index`），存储于 `send_head` 张量。这是后续回收的 “精准地图”，确保 “按迹取回”。

- **角色分工与流程**

回收阶段角色反转（发送端→接收端），由 24 个 Warp 组成的 “回收小组” 协同工作：

- **核查员（Warp 0）**：监控所有处理员进度，统一更新收件篮 `head` 指针，避免冲突；
- **处理员（Warp 1~23）**：根据 `send_head` 定位返程数据，等待目标数据到达后，将结果累加到本地 `recv_x`，并向核查员汇报进度。

所有处理员完成后标记 “退休”，核查员确认后完成回收闭环。

## 2.3 设计优势总结

DeepEP 单节点通信机制通过 “规划先行、并行分解、异步流控” 三大核心策略，解决了 MoE All-to-All 通信的效率问题：

1. **全局规划**：通过 `notify_dispatch` 生成精确地址计划，避免数据冲突；
2. **并行传输**：多 Channel 与 Warp 小组提升任务并行度，充分利用 NVLink 带宽；
3. **轻量同步**：自定义 Barrier 与 head/tail 指针流控，减少等待开销；
4. **底层优化**：基于 P2P/IPC 的直接显存访问，替代传统通信库的冗余操作。

这一设计将混沌的 All-to-All 通信转化为有序的数据流，为 MoE 模型在单节点内的高效训练 / 推理提供了核心支撑。

## 参考

1. [deepseek-ai/DeepEP: DeepEP: an efficient expert-parallel communication library]([https://github.com/deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP))
2. [(82 封私信 / 80 条消息) DeepEP 邮局（一）：机内 MoE 通信 - 知乎]([DeepEP 邮局（一）：机内 MoE 通信](https://zhuanlan.zhihu.com/p/1927846896682133234))
3. [DeepSeek AI Infra(3) - DeepEP 的原理与代码剖析 - 知乎]([DeepSeek AI Infra(3) - DeepEP的原理与代码剖析](https://zhuanlan.zhihu.com/p/27777601573))
4. [DeepSeek最新开源汇总：5天完整版 - 知乎]([DeepSeek最新开源汇总：5天完整版](https://zhuanlan.zhihu.com/p/27057478016))

# DeepEP 跨节点通信机制

跨节点（Inter-Node）场景，其 “L 形路由” 策略将进一步解决跨节点延迟与带宽瓶颈，实现分布式环境下的高效通信（后续章节展开）。