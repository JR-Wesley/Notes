# C01：从 host RDMA 到 GPUDirect RDMA 与 IBGDA——分离调用、提交和数据路径

官方 Using NVSHMEM 页面关于 GDAKI、IBGDA、GPUNetIO、DOCA、TMA 与 transport 选择条件的英文原文和中文对照，见 [双语核对笔记：Using NVSHMEM](official-docs/r05-using-nvshmem.md)。这些滚动版本条件用于建立候选路径，实际 handler 与数据路径仍需按 C02 的证据链验证。

“GPU 通信”常被一句“数据直接从 GPU 经过网卡发送”概括，但这句话省略了决定架构和性能的三个问题：谁调用通信 API，谁把请求提交给 NIC，NIC 实际读取或写入哪种内存。传统 host verbs、GPUDirect RDMA、NVSHMEM device API、CPU proxy 和 IBGDA 可以在这三个维度上形成不同组合；只确认其中一项，不能推出另外两项。

本课从 [A01：RDMA 通信模型](a01-rdma-communication-model.md)的 CPU verbs 基线出发，逐步比较 host staging、同节点 GPU P2P、CPU 提交的 GPUDirect RDMA、NVSHMEM device API 的 proxy path，以及 IBGDA/GDAKI。重点是控制路径和 payload 数据路径，不进入某个固定 NVSHMEM commit 的 WQE、doorbell record、CQ 或 QP 映射实现；这些内容属于后续 D 模块。正文按 2026-09-12 可见的 NVSHMEM 3.7.2 发布周期文档、CUDA 13.4 GPUDirect RDMA Guide 和当前 CUDA Programming Guide 核对。本文没有检查实际机器、编译 NVSHMEM、运行通信程序或测量性能。

## 三个独立问题构成路径坐标

第一轴是 API caller。host API 由 CPU thread 调用，on-stream API 由 CPU enqueue 到 CUDA stream，device API 由 kernel 中的 GPU thread/warp/block 调用。这个轴只说明应用代码在哪里表达请求，不保证请求之后由谁驱动硬件。

第二轴是 work submitter。传统 verbs 由 CPU 构造/发布工作请求并通知 NIC；NVSHMEM proxy path 中，GPU 可以先产生通信请求，但 CPU proxy thread 再把请求交给网络 transport；GDAKI/IBGDA 的目标则是让 GPU 在稳态路径中直接准备和提交 NIC 工作。提交者决定每条请求是否必须经过 CPU 的进度路径，但不单独决定 payload 位于 host 还是 GPU memory。

第三轴是 data mover 与 operand memory。NIC 可以 DMA host memory，也可以在满足 GPUDirect RDMA 条件时直接 DMA GPU memory；同节点 GPU P2P 可能完全不经过 NIC，而由 GPU SM 的 load/store 或 CUDA copy engine 经 NVLink、NVSwitch 或 PCIe peer path 搬运。数据移动者与提交者可以不同：CPU post 的 RDMA WR 完全可以让 NIC 直接读 GPU memory；GPU 发起的 device API 也可能把请求交给 CPU proxy，但最终 NIC 仍直接访问 GPU memory。

| 路径 | API caller | 稳态 work submitter | payload 主要经过 | 每条跨节点请求是否需要 CPU proxy |
| --- | --- | --- | --- | --- |
| host staging + host RDMA | CPU/CUDA host runtime | CPU | GPU ↔ host staging ↔ NIC | 是，或由 host 应用直接 post |
| host-posted GPUDirect RDMA | CPU | CPU | GPU memory ↔ NIC | 是，提交需要 CPU；payload 不经 host staging |
| 同节点 GPU P2P | GPU thread 或 CUDA runtime | GPU/CUDA runtime 所选本地路径 | GPU memory ↔ NVLink/NVSwitch/PCIe ↔ GPU memory | 不适用，没有跨节点 NIC 请求 |
| NVSHMEM device API + proxy transport | GPU | CPU proxy | 通常是 GPU memory ↔ NIC；请求描述经过 proxy | 是 |
| NVSHMEM device API + IBGDA/GDAKI | GPU | GPU，或按 NIC handler 配置/能力回退 | GPU memory ↔ NIC | 目标是 GPU 直接提交；必须核验 handler |

这张表是教学分类，不是对任意安装环境的自动判定。NVSHMEM runtime 可以按目标 PE、拓扑、API、消息大小、构建能力和环境变量选择不同 transport；同一程序的节点内和节点间通信甚至可能走不同路径。

## 基线一：host staging 让网络只接触主机内存

最保守的跨节点 GPU 数据传输可以分成三段。源端先把 payload 从 GPU memory 复制到 pinned host staging buffer；CPU 通过 socket 或 host-memory RDMA 发送这段 host buffer；目标端 NIC 把数据放入目标 host buffer，再由 CUDA copy 把它搬到目标 GPU。若使用 RDMA，中间网络段仍可由 NIC DMA，但 NIC 的本地和远端 operand 都是 host memory。

```text
源 GPU ──D2H copy──> 源 pinned host buffer
                         │
                 CPU post / NIC DMA
                         │
                         ═════ network ═════
                                               │
目标 GPU <──H2D copy── 目标 pinned host buffer <── 目标 NIC
```

控制路径也由 CPU 串联：CPU 等待或用 event 确认 D2H 已生产 staging buffer，post network request，观察 completion，再在目标侧安排 H2D。即使三段都异步排队，正确性仍需 CUDA event、RDMA completion 和应用通知共同建立依赖。host staging 的优势是兼容性和路径可观察性较强；成本是额外的 GPU↔host copies、host memory 容量与流水管理。能否通过双缓冲隐藏这些成本必须实测，不能从“异步”二字推导已经重叠。

这里的 CPU 参与有两种含义：CPU 管理请求和 completion，payload 还确实经过 host memory。后续 proxy path 只保留前一种含义，因此“有 CPU proxy”不等于“payload 被复制到 CPU”。

## 基线二：普通 host verbs 已经能够使用 GPU operand

GPUDirect RDMA 允许兼容的第三方 PCIe 设备（典型是 NIC/HCA）直接读写 GPU memory。通信库和驱动把 GPU allocation 对应的页或 DMA-BUF 映射暴露给 RDMA 子系统，NIC 据此建立 DMA 地址转换；请求执行时，源端 NIC 可以直接读取 GPU memory，目标端 NIC 可以直接把数据写入 GPU memory。payload 不必先 bounce 到 host staging buffer。

```text
CPU application / verbs provider
        │ 创建资源、注册 GPU memory、post WR、观察 completion
        ▼
源 NIC  <──── PCIe peer DMA ────> 源 GPU memory
   ║
   ║ network
   ║
目标 NIC ──── PCIe peer DMA ────> 目标 GPU memory
```

图中 CPU 仍然创建 context、PD、MR、QP 和 CQ，交换远端地址/key，并调用 `ibv_post_send()` 或库封装。改变的是 WR 的 SGE 或 remote memory 指向 GPU allocation，以及 NIC 获得访问 GPU pages 的能力。也就是说，GPUDirect RDMA 是数据路径能力，不是 GPU 提交能力。一个完全由 CPU post 的程序仍然可以实现 GPU-memory-to-GPU-memory 的 NIC DMA；反过来，GPU 能调用某个 API 并不自动证明 NIC 已获得 GPU memory 映射。

[A02：RDMA 资源与内存注册](a02-rdma-resources-and-memory-registration.md)中“地址、范围、PD、lkey/rkey 和生命周期”仍然成立。GPU memory 不是因为来自 `cudaMalloc` 就天然可被任意 NIC 访问；需要 DMA-BUF 或 `nvidia-peermem` 等受支持的 peer-memory 路径、兼容的驱动/HCA，并满足平台拓扑和 IOMMU 等条件。注册建立 NIC 地址转换与生命周期，不负责应用层 ready/ack，也不保证 CUDA kernel 与 NIC 对同一 memory 的访问已经正确排序。

CUDA GPUDirect RDMA Guide 还特别强调 relaxed memory model：第三方设备形成一条独立数据流，CUDA synchronization/work submission API 才能建立相应内存顺序。应用不能在 NIC 正写 GPU buffer 时让 kernel 无协议地读取，也不能让 producer kernel 尚未完成就由 NIC 读取 source。NVSHMEM 会封装部分注册与 ordering，但 [B02](b02-completeness-ordering-and-visibility.md)和 [B04](b04-data-publishing-and-buffer-ownership.md)中的发布与消费边界并未消失。

## 同节点 CUDA P2P 不是 GPUDirect RDMA

若两个 GPU 位于同一节点并具有 peer access，GPU 可以经 NVLink、NVSwitch 或受支持的 PCIe P2P 路径访问对方 memory。CUDA 用 `cudaDeviceCanAccessPeer` 查询能力，并通过 peer access 或 VMM 映射建立可访问关系；随后 kernel 可能直接解引用 peer mapping，CUDA copy API 也可能选择专用 copy engine。这个路径的第三方设备不是网络 NIC，因此不属于 GPUDirect RDMA。

```text
GPU 0 SM / copy engine ── NVLink/NVSwitch/PCIe P2P ── GPU 1 memory
                    （不经过跨节点 NIC 与 fabric）
```

NVSHMEM 对本地 peer 可能利用 GPU SM 执行直接 load/store。官方 Best Practice Guide 指出，P2P transport 上的 device API 数据搬运由 GPU SMs 完成，warp/block API 可用更多 threads 搬运大块数据。这个实现特征不能推广到跨节点：当目标 PE 位于另一台主机时，请求必须进入网络 transport，SM 不能像访问本地 peer mapping 一样直接解引用远端主机的 C++ pointer。

P2P capability 也不是对称的逻辑假设。CUDA peer enable 是有方向的，拓扑和系统资源会限制可建立的 peer relationships；即使 UVA 让指针数值在映射后可使用，也不意味着任意 GPU 或任意节点都直接可达。[B01：对称对象与远端寻址](b01-symmetric-objects-and-remote-addressing.md)中的 symmetric address 仍由 runtime 解释，不能与 CUDA peer pointer 或 `nvshmem_ptr()` 的局部直接映射混为一谈。

## NVSHMEM device API 与 CPU proxy：GPU 发起，CPU 代为提交

在非 GDAKI 的跨节点 proxy transport 中，kernel 内的 GPU thread 调用 NVSHMEM device API。GPU 侧代码准备一个通信请求并把它发布给 runtime 的 proxy mechanism；CPU proxy thread 取得请求后，通过远端 transport 驱动 NIC；NIC 再直接读取或写入已注册的 GPU symmetric heap。这里是 GPU 发起 API、CPU 提交 NIC、NIC 搬运 GPU payload 的三段组合。

```text
GPU thread
   │ device nvshmem_put / get / AMO
   ▼
GPU-visible proxy request/state
   │                    控制路径：请求必须交给 CPU
   ▼
CPU proxy thread ── provider/transport ──> NIC queue / doorbell
                                              ║
payload 数据路径：GPU memory <── DMA ── NIC ══╬══ network
```

该图只描述公开架构关系，不承诺固定 proxy queue 格式、轮询算法或 completion counter。没有固定 NVSHMEM commit 时，不应写出具体 descriptor 字段、源码 symbol 或某次调用必然生成多少 WQE。稳定结论是：proxy transport 让 device-initiated network communication 经过 CPU proxy 才提交，因此大量细粒度请求可能受单 proxy thread 串行推进和请求交接开销限制。

官方 Best Practice Guide 建议 proxy path 上合并小元素，把多个 `p`/细粒度调用打包为较大的 `put`；它还指出 block-scoped PUT 在这种路径中可能只由一个内部 thread 发出一次 send，而所有应用 threads 各自调用标量 PUT 会形成多个串行交给 proxy 的操作。但这属于当前实现指导，不是所有版本和 transport 的永久 API 语义。应用应先保持调用粒度正确，再用实际 profiler 和 benchmark 判断打包收益。

最重要的纠错是：CPU proxy 不代表 CPU 复制 payload。proxy 可以只处理请求元数据和 NIC 提交，而 NIC 通过 GPUDirect RDMA 直接访问 GPU memory。因此，观察到 CPU thread 活跃不能据此断言发生 host staging；必须同时检查 payload buffer 类型和 DMA path。

## IBGDA/GDAKI：把稳态 NIC 提交推进到 GPU

NVSHMEM 将 GPUDirect Async Kernel-Initiated 能力称为 GDAKI；IBGDA 是面向支持条件下 Mellanox/ConnectX InfiniBand/RoCE 设备的一种 remote transport。它的目标是在 GPU 上实现稳态网络通信的控制路径和数据路径，避免 device API 请求先 reverse-proxy 到 CPU thread。GPU threads 能并行准备/提交网络请求，NIC 仍通过 peer-memory/DMA-BUF 等映射直接访问 GPU memory。

```text
GPU thread / warp / block
   │ device NVSHMEM API
   │ 准备请求、选择 runtime 管理的 QP/slot
   ▼
GPU-visible NIC work queue / doorbell path ─────> NIC
          ▲                                        ║
          │ 控制与提交位于 GPU 稳态路径             ║ network
GPU memory <──────────── payload DMA ──────────────╣
```

这里的“GPU 直接提交”不表示应用可绕过 runtime 任意写 NIC 寄存器。初始化阶段仍需 CPU、CUDA driver、NIC driver 和 NVSHMEM runtime 创建资源、注册 memory、建立 QP/keys、选择 NIC 并把必要映射提供给 GPU；错误处理和作业启动也不会消失。GDAKI 改变的是每条稳态请求是否必须由 CPU proxy 接力，而不是取消控制面生命周期。

IBGDA 还包含必须显式保留的 handler 边界。当前 `NVSHMEM_IBGDA_NIC_HANDLER` 可以取 `auto`、`gpu`、`cpu` 或 `cpu_cuda_memory`：`gpu` 使用 GPU SM 敲 NIC doorbell，`cpu` 使用 CPU proxy，`auto` 在 GPU handler 不受支持时回退到 CPU。因此，仅设置 `NVSHMEM_IB_ENABLE_IBGDA=1` 或看到 IBGDA transport 名称，还不足以证明实际 doorbell 由 GPU 执行；需要记录最终生效配置和 runtime 日志。

启用路径至少分成构建和运行两层。构建 NVSHMEM 时需要包含 IBGDA 支持，当前安装文档使用 `NVSHMEM_IBGDA_SUPPORT=1`；运行时再用 `NVSHMEM_IB_ENABLE_IBGDA=1` 启用 InfiniBand GDA transport。GDAKI 文档还列出 Mellanox HCA/NIC、MOFED 版本、NVIDIA driver 以及 DMA-BUF、`nvidia_peermem` 或 `nv_peer_mem` 之一等前提。环境变量存在、库成功启动与某次请求确实走 GPU handler 是三种不同证据。

## GPUDirect RDMA 与 GPUDirect Async 的包含关系

可以把两者理解为相邻但不同的能力层。GPUDirect RDMA 解决 NIC DMA “能否直接访问 GPU memory”；GPUDirect Async/GDAKI 解决 GPU “能否直接驱动 NIC 工作提交”。IBGDA 通常需要前者作为 payload path 的基础，再增加 GPU 可见的队列/doorbell 和 runtime 并发管理。

```text
数据路径能力：       NIC <──── DMA ────> GPU memory        GPUDirect RDMA

请求提交能力： GPU threads ── WQE/doorbell ──> NIC        GPUDirect Async / IBGDA
```

因此有一种常见且合法的中间组合：CPU post + GPUDirect RDMA。它没有 host payload bounce，却仍由 CPU 提交；另一个组合是 GPU device API + CPU proxy + GPUDirect RDMA，它从应用视角由 GPU 发起，但网络请求仍经 CPU。只有当 transport 和 handler 证据都支持时，才能标记为 GPU device API + GPU submit + NIC DMA GPU memory。

“GPU 能调用 NVSHMEM device API”“NIC 能访问 GPU memory”“GPU 能敲 NIC doorbell”分别对应三条能力测试，任何一条都不能从另外一条自动推出。这一分解也是 C02 做实际部署核验的表头。

## 五条路径的控制流与数据流对照

为了避免图中箭头过多，可以把每条路径写成两行。控制流描述请求描述和 completion 信息如何移动；数据流只描述 payload 字节实际经过的 memory/device。

```text
1. host staging
control: CPU/CUDA runtime -> D2H done -> CPU post -> NIC completion -> H2D enqueue
data:    source GPU -> source host -> NIC -> target host -> target GPU

2. host-posted GPUDirect RDMA
control: CPU verbs/runtime -> NIC queue/doorbell -> CPU observes completion
data:    source GPU -> NIC -> network -> NIC -> target GPU

3. same-node GPU P2P
control: GPU instruction or CUDA copy command -> local fabric operation
data:    GPU 0 -> NVLink/NVSwitch/PCIe P2P -> GPU 1

4. device API + CPU proxy
control: GPU request -> CPU proxy -> NIC queue/doorbell -> proxy/GPU completion state
data:    source GPU -> NIC -> network -> NIC -> target GPU

5. device API + IBGDA GPU handler
control: GPU request/WQE/doorbell -> NIC -> GPU-visible completion/progress state
data:    source GPU -> NIC -> network -> NIC -> target GPU
```

这五条路径可以共享上层 NVSHMEM PUT 语义，但完成实现不必相同。B02 的 `quiet` 是 API 契约，不能在 path 4 中简单解释成“CPU 等某条 CQE”，也不能在 path 5 中解释成“GPU 等当前 WQE 的 CQE”。runtime 可能批处理、使用 counters、代理状态或 transport-specific completion；只有固定源码和实验才能把抽象完成映射到内部机制。

## 地址、注册和 key 并未因 IBGDA 消失

NVSHMEM API 让应用用 `<local symmetric address, target PE>` 指定远端对象，不再手工传 `remote_addr+rkey`。但 runtime 仍要为各 transport 建立对称 heap 的远端地址映射和访问凭据，并保证 NIC/GPU 使用正确范围。IBGDA 只是让这些预先建立的信息能在 GPU 提交路径中使用，绝不是“GPU 有一个指针就能访问远端”。

对于本地 source，NVSHMEM symmetric heap 通常已由 runtime 管理；某些普通 host/device buffers 可通过 `nvshmemx_buffer_register` 成为本地 operand，但不会因此成为远端 symmetric object。对于 GPUDirect RDMA，NIC 的 DMA mapping 生命周期必须覆盖所有在途请求；对 IBGDA，GPU 可访问的 QP、doorbell、completion 和 key metadata 也必须在请求完成前保持有效。应用 finalize 或 free 前仍要完成通信和业务 drain。

安全边界也没有变。错误的 remote offset、越过 allocation、过期 key 或 transport capability 不匹配仍会导致 protection error 或未定义行为。GPU 提交提高了并行度，反而要求 runtime 更严格地管理多个 threads 对有限 queue slots、QPs 和 completion resources 的共享；这正是 D02 的并发队列主题。

## completion、通知与 CPU 是否在场无关

无论请求由 CPU proxy 还是 GPU handler 提交，PUT 的远端交付都不等于消费者已经处理。IBGDA 不会自动增加 ready/ack；CPU staging 也不会因为 H2D copy complete 就证明目标 kernel 已消费。路径选择只改变实现请求如何到达 S2/S3，不改变 [B04](b04-data-publishing-and-buffer-ownership.md)中 S4 消费和 S5 ack 的应用责任。

同样，GPU 直接提交不表示调用立即完成。GPU thread 仍可能受到队列容量、batch、QP 映射、NIC backpressure 和 completion 资源约束；NBI 仍需相应 flush/quiet，thread-group 调用仍需满足 B03 的参与约束。是否依赖 CPU 进度是架构问题，blocking/NBI 和 buffer reuse 是 API 语义问题，两者应分别记录。

错误恢复也通常需要 host/runtime 参与。GPU handler 能处理稳态 fast path，不代表 GPU 可以独立完成设备发现、作业 bootstrap、驱动错误复位或进程失败恢复。“CPU bypass”应限定为每条稳态数据请求不经 CPU proxy，而不是 CPU 从程序生命周期中消失。

## 并行度与粒度的性能取舍

host staging 增加 copy stages，却可能通过较大连续消息和成熟 host RDMA 路径获得稳定行为；GPUDirect RDMA 减少 host bounce，但性能仍受 GPU–NIC PCIe/NVLink 拓扑、BAR/DMA mapping、PCIe switch/root complex 和 network 限制。CPU proxy 对少量大消息可能不是瓶颈，对大量 GPU threads 发出的细粒度请求则可能形成串行提交点。

IBGDA 允许多个 GPU threads 映射到多个 QPs 并行提交，官方 Best Practice Guide 指出它能提高小消息 message rate，使较小 buffer 更早利用网络带宽。但同一指南也指出 GPU thread 填 WQE 的单请求延迟可能高于 CPU，且为每个 thread 配独立 QP 会消耗资源并增加 fence/quiet 成本。IBGDA 因而不是“所有消息大小都必然更快”，而是在请求粒度、并发、QP 数量、batch 和完成成本之间重新取舍。

当前环境变量提供 `NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH`、DCI/RC 数量与映射、fetch slots、多端口和 NIC handler 等调节项。这些参数只能在固定硬件、版本、拓扑和 workload 上逐项测量。修改多个变量后只看到总吞吐变化，无法归因；应遵循一次改变一个设置、同时记录延迟、带宽、message rate、SM 占用和 quiet/fence 成本的实验纪律。

## 不能从拓扑图直接推导实际 transport

`nvidia-smi topo -m` 可以显示 GPU、NIC 和 CPU/NUMA 的相对连接，`lspci -t` 可以查看 PCIe 层级，`ibv_devinfo` 可以确认 RDMA device/port；这些都是能力与性能风险的证据，却不是“本次 NVSHMEM PUT 已走 IBGDA”的运行证据。runtime 可能因构建选项、环境变量、handler 不支持、目标为本地 PE 或初始化失败而选择另一条路径。

实际核验至少需要四类证据：二进制/构建是否包含目标 transport；启动时生效的 `NVSHMEM_REMOTE_TRANSPORT`、`NVSHMEM_IB_ENABLE_IBGDA` 和 handler 配置；`NVSHMEM_INFO`、`NVSHMEM_DEBUG`/debug file 等 runtime 日志；实际 GPU–NIC topology、driver、MOFED、peer-memory/DMA-BUF module 与设备状态。必要时再用 Nsight Systems/NVTX、CPU utilization、transport counters 和受控 feature toggles 交叉验证。

即使日志显示 IBGDA，也要记录 `NVSHMEM_IBGDA_NIC_HANDLER` 最终选择。`auto` 的含义包含“不支持时回退 CPU”，所以“变量默认 auto”不能作为 GPU doorbell 证据。若强制 `gpu` 后初始化失败，这反而说明能力条件尚未满足；不能为了获得标签而忽略错误。

## 一套分层判定方法

面对“这个程序是否使用 GPU Direct”之类问题，可以按下列顺序回答，避免先给结论再找证据。

1. 目标是否同节点？若是，先判断 CUDA/NVSHMEM P2P，避免把 NVLink path 称为 GPUDirect RDMA。
2. 跨节点 payload 的 source/dest 位于 host 还是 GPU memory？若 NIC operand 是 host staging，就不是 GPU-memory direct DMA。
3. NIC 是否通过 DMA-BUF、`nvidia-peermem` 等受支持路径注册并访问 GPU memory？满足时才具备 GPUDirect RDMA 数据路径。
4. API 由 host、on-stream 还是 device code 调用？这只回答发起位置。
5. device 请求由 CPU proxy 还是 GDAKI transport 提交？进一步记录实际 NIC handler 是 GPU 还是 CPU。
6. 用构建、环境、日志、拓扑和 profiler/counters 形成闭环；缺少任何关键证据时标为“能力存在，实际路径未确认”。

最终记录不应只写“使用 IBGDA=true”，而应形成类似：`device API / remote target / IBGDA transport / NIC handler=gpu / GPU symmetric operand / NIC X / topology Y / evidence logs Z`。如果只能确认前三项，就把其余项明确标为未核验。

## 常见误解与定位

误解一是“CPU utilization 很低，所以一定是 IBGDA”。CPU 低占用也可能来自大消息、低请求率、异步 proxy 或采样粒度；应看 transport/handler 日志和请求路径。相反，CPU 有活动也不证明 payload staging，因为 proxy、bootstrap、completion 与错误处理都会使用 CPU。

误解二是“NIC 和 GPU 在同一 PCIe switch，所以程序一定用 GPUDirect RDMA”。拓扑只是支持条件之一；还需 driver、DMA-BUF/peer-memory、MR 注册和 runtime transport。IOMMU translation、ACS redirect、容器 device exposure 或错误 NIC mapping 都可能改变能力或性能。

误解三是“device API 等于 GPU 直接 post NIC”。device API 在 proxy transport 上由 GPU 发起、CPU proxy 提交；只有 GDAKI/IBGDA/GPUNetIO GDAKI 等受支持路径才目标明确地消除 reverse proxy，而且 handler 仍需核验。

误解四是“GPUDirect RDMA 等于 GPU P2P”。P2P 是同节点 GPU 间 peer memory access，可能使用 NVLink/NVSwitch/PCIe；GPUDirect RDMA 是 NIC 等第三方 PCIe 设备与 GPU memory 的 DMA。两者可能共享某些平台机制，但通信端点和验证方法不同。

误解五是“IBGDA 绕过 CPU，所以 fence/quiet/signal 不再需要”。这些 API 表达应用可依赖的 ordering、completion 和通知语义，与内部提交者无关。删除它们可能在一种 transport 上偶然运行，却不再具备可移植正确性。

## 诊断题与参考推理

### 问题一：CPU 调用 `ibv_post_send`，WR 的 SGE 指向已注册 GPU memory，这是不是 GPUDirect RDMA？是不是 IBGDA？

若 NIC 确实通过支持的 peer-memory/DMA-BUF mapping 直接访问 GPU memory，这是 GPUDirect RDMA 数据路径；请求仍由 CPU post，不是 IBGDA 的 GPU-initiated submit path。

### 问题二：kernel 调用 `nvshmem_put`，能否据此断言 GPU 敲了 NIC doorbell？

不能。device API 可以走 CPU proxy transport，也可以走 IBGDA/GDAKI；即使选择 IBGDA，NIC handler 还可能是 `auto` 回退或显式 CPU。需要 transport 和 handler 的运行证据。

### 问题三：CPU proxy 是否意味着 payload 经过 host buffer？

不意味着。proxy 可以只搬运请求元数据并提交 NIC，payload 仍由 NIC 通过 GPUDirect RDMA 在 GPU memories 之间移动。只有发现明确的 D2H/host/H2D staging buffers 和 copies，才能标记 host staging。

### 问题四：同节点两个 GPU 经 NVLink 传输，是否应称为 GPUDirect RDMA？

不应。它是 GPU peer-to-peer path，没有第三方网络 NIC 与 RDMA fabric。应记录为 CUDA/NVSHMEM P2P，并单独核验 peer access 和实际 topology。

### 问题五：`NVSHMEM_IB_ENABLE_IBGDA=1` 且程序运行成功，是否已证明 GPU handler 生效？

尚未。还要确认 library 构建包含 IBGDA、实际 remote transport 选择、`NVSHMEM_IBGDA_NIC_HANDLER` 的最终值和是否发生 auto fallback，并结合 startup/debug logs 核对。

### 问题六：IBGDA 是否会改变 PUT 的远端消费语义？

不会改变公开 API 语义。它改变请求如何提交和推进，PUT 仍需按 blocking/NBI、fence/quiet、signal/ack 判断本地完成、远端交付和消费结束。不能把 GPU direct submit 当成远端 consumer notification。

### 问题七：为什么 IBGDA 可能提高并发 message rate，却未必降低单条消息延迟？

多个 GPU threads/QPs 可以并行提交，减轻单 CPU proxy 串行瓶颈；但 GPU thread 准备 WQE、queue mapping、batch 和 completion 管理也有成本。官方指南明确指出单 GPU thread 填 WQE 可能不如 CPU thread 快，结果取决于消息大小和并发，必须实测。

## 掌握标准与后续路线

完成本课后，读者应能为一条通信路径分别填写 API caller、work submitter、data mover、source/destination memory 和 per-operation CPU involvement，并能画出 host staging、host-posted GPUDirect RDMA、P2P、device+proxy 和 device+IBGDA 的两条箭头：一条是控制/提交，另一条是 payload。读者还应能解释为什么 CPU proxy+GPUDirect RDMA 是合理组合，以及为什么 device API、GPUDirect RDMA 和 GPU doorbell 是三个独立能力。

读者还应能列出确认 IBGDA 的证据链，而不是从 GPU/NIC 型号或一个环境变量猜测 transport。下一课 `C02：Transport 与 GPU–NIC 拓扑核验` 将把这些字段整理成可执行检查流程；该正式文件尚未创建，所以本文不建立失效链接。D 模块随后再选择固定 NVSHMEM commit，把 API 层路径对应到真实 QP、WQE、doorbell、completion 和多生产者队列。

## 主要来源

- [CUDA GPUDirect RDMA Guide 13.4](https://docs.nvidia.com/cuda/gpudirect-rdma/)：第三方 PCIe 设备直接访问 GPU memory、GPU page pin/mapping、拓扑/IOMMU 条件以及 CUDA synchronization 与 GPUDirect RDMA memory ordering。
- [CUDA Programming Guide: Multi-GPU Systems](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html)：CUDA P2P capability、peer access、NVLink/PCIe topology 和 UVA/VMM 边界。
- [NVSHMEM Using: GPUDirect Async Kernel-Initiated Communication](https://docs.nvidia.com/nvshmem/api/latest/using.html)：GDAKI、IBGDA/GPUNetIO enable 方法及 Mellanox、MOFED、driver、DMA-BUF/peer-memory 前提。
- [NVSHMEM Environment Variables](https://docs.nvidia.com/nvshmem/api/latest/gen/env.html)：remote transport、`NVSHMEM_IB_ENABLE_IBGDA`、`NVSHMEM_IBGDA_NIC_HANDLER`、batch、QP 和 DirectNIC 相关配置。
- [NVSHMEM Best Practice Guide: Device APIs](https://docs.nvidia.com/nvshmem/release-notes-install-guide/best-practice-guide/apis.html)：P2P 由 GPU SM 搬运、proxy-based transport 的 CPU proxy 交接，以及 GDAKI 的 GPU 并行提交特征。
- [NVSHMEM Best Practice Guide: Performance](https://docs.nvidia.com/nvshmem/release-notes-install-guide/best-practice-guide/performance.html)：IBGDA/GDAKI 与 CPU proxy 的差异、QP/消息粒度权衡、transport/NIC mapping 和 profiling 建议。
- [NVSHMEM Installation](https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/nvshmem-install-proc.html)：IBGDA 构建选项和当前依赖说明。
- [NVSHMEM Troubleshooting and FAQs](https://docs.nvidia.com/nvshmem/api/latest/faq.html)：`NVSHMEM_INFO`、`NVSHMEM_DEBUG`、debug file 以及 InfiniBand/P2P 常见配置边界。
- [NVSHMEM 3.7.2 Release Notes](https://docs.nvidia.com/nvshmem/release-notes-install-guide/release-notes/release-3720.html)：本课核对时的当前发布周期；滚动 API/Best Practice 页面不代表固定源码 commit。
