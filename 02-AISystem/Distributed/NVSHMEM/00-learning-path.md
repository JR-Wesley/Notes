# NVSHMEM 与 IBGDA 学习路径

## 学习目标与边界

本目录面向具备 CUDA 基础、尚无 RDMA 使用经验的学习者。主线不是从 NVSHMEM API 名称开始背诵，而是反复追踪同一个数据传输任务：两个 Linux 进程分别拥有一段内存，发送侧要把 payload 交给接收侧；课程逐步解释谁发起、访问哪里、谁有权限、何时完成、谁被通知，以及缓冲区何时能够复用。

课程按以下因果链推进：

```text
A：RDMA 使用基础
  -> B：NVSHMEM 编程
  -> C：GPU 网络路径
  -> D：IBGDA 实现
  -> E：应用与性能
```

目录与知识框架已经完成重建，现按用户授权逐课撰写正式正文；未创建空白课程文件，未提供硬件环境时也不伪造编译、运行或性能结果。

## 必须贯穿全程的区分

后续每篇正文和实验都应能解释以下边界：

- 有效地址、网络可达性、访问权限和数据就绪是四个不同条件；
- 请求提交、本地源缓冲区可复用、远端数据交付和远端消费结束是四个不同事件；
- GPU 调用通信 API、NIC 访问 GPU 内存、GPU 直接提交 NIC 工作是三个不同能力；
- 对称对象要求各 PE 的分配对应，不要求各 PE 虚拟地址数值相等；
- `fence`、`quiet`、barrier、signal 和 CQE 必须按各自 API 契约解释，不能因作用相似就视为同一机制；
- 发布 ready 不能代替消费 ack，槽位重用必须有独立的所有权依据。

## 补充路线覆盖复核（2026-09-12）

依据新增的“NVSHMEM 基础原理与知识框架”逐项复核后，当前路线已经覆盖其主要因果链。PE、PGAS、symmetric heap/object、UVA 与 PGAS 的区别在 B00—B01；PUT/GET/NBI、local/remote completion、visibility、fence/quiet、signal/wait 在 B02；thread/warp/CTA、stream 和 collective launch 在 B03；producer-consumer、atomic reservation 的边界和 buffer ownership 在 B04；host runtime、P2P、GPUDirect RDMA、proxy、IBGDA 及 GPU/NIC 路径在 C01—D03；性能与 MoE/halo 应用在 E01—E02。

复核也发现五个主题目前只有零散说明，尚不足以独立验收，因此加入规划课程而不创建空文件：B05 系统讲解远程 AMO 与分布式协调；B06 讲 Team 管理和数据 collective；C00 讲初始化、bootstrap 和 runtime control plane；D04 在固定源码中追踪 AMO/signal/ordering 的 IBGDA 实现；E03 讨论 persistent kernel、图任务和分布式 work queue。这些补充不是重写已有路线，而是把输入材料中真正缺少的语义层、实现层和应用层闭合。

输入材料中的部分表达只适合作为第一层心智模型，后续正文必须继续保留边界：`remote_base + symmetric_offset` 是寻址教学模型，不是 NVSHMEM 对所有 transport 承诺的内部公式；NBI 提供较早返回，不自动保证计算通信重叠；高级 collective 不应未经源码证明就断言由 PUT/GET/signal 组合实现；NVSHMEM atomic 也不能在未固定 transport 与源码前一律映射成原生 mlx5 RDMA atomic WQE。

| 输入路线主题簇 | 当前承载位置 | 复核结论 |
| --- | --- | --- |
| PE、PGAS、symmetric heap/object、UVA、MPI/NCCL 对比 | B00、B01、`reference-nvshmem-overview.md` | 已详细覆盖；B00 已按当前标准重写 |
| PUT/GET、blocking/NBI、completion、visibility、ordering | B02 | 已详细覆盖 |
| signal/wait、producer-consumer、槽位复用 | B02、B04 | 已详细覆盖 |
| thread/warp/CTA/SM、GPU-initiated communication | B03 | 已详细覆盖；SM 与 QP 的实现映射留给 D02/D03 |
| remote AMO、counter、allocator、scheduler | B04 给出 fetch-add 边界；B05 系统讲解原子协调 | 已详细覆盖 API 与协议层；实现追踪留给 D04 |
| Team、broadcast/reduce/all-to-all/fcollect | B00/B03 覆盖参与和执行边界；B06 系统讲解 Team 与数据 collective | 已详细覆盖 API 语义与数据布局；具体实现算法留待固定源码或实验 |
| initialization、bootstrap、GPU/NIC discovery、heap/endpoint setup | B00、C01 提供概览；C00 系统讲解初始化控制面 | 已详细覆盖公开状态与职责；内部调用链留待固定源码 |
| P2P、IB、GPUDirect RDMA、proxy、IBGDA | A01—A04、C01 建模；C02 建立部署证据链；D00—D03 进入实现 | API、路径与核验方法已覆盖；D00 已建立固定源码总图，逐请求追踪仍待 D01—D03 |
| QP/WQE/CQ/doorbell、RC/DCI/DCT、并发队列 | D00—D03 | D00 已在 NVSHMEM v3.7.2-0 固定 commit 上建立对象与队列总图；细节分析待 D01—D03 |
| MoE、halo/stencil、性能与 overlap | E01—E02 | 已规划 |
| persistent kernel、graph、distributed work queue | B03/B04 只有基础机制 | 新增 E03，补足综合应用与活性分析 |

## 前置依赖

| 前置知识 | 起点要求 | 在课程中的用途 |
| --- | --- | --- |
| Linux 进程、线程和虚拟内存 | 能区分两个进程各自的地址空间 | 理解本地地址与远端地址不能直接互换 |
| C/C++ 指针与对象生命周期 | 能说明地址、长度和对象存活期 | 分析 MR、SGE、对称对象和缓冲区复用 |
| CUDA kernel、thread/warp/block | 已具备基础 | 理解 device API 与 thread-group cooperation |
| CUDA stream 与 event | 知道异步排队和依赖 | 区分 host、stream 和 device 调用域 |
| 基本并发概念 | 能理解生产者、消费者和环形队列 | 推导 ready/ack、背压和队列回绕 |

RDMA verbs、InfiniBand 队列资源、GPUDirect RDMA 和 IBGDA 均由本课程从零引入，不作为隐藏前置条件。A01 是 RDMA 零基础入口：第一次出现 context、PD、MR、QP、SQ/RQ、CQ、WR、SGE 和 WC 时先解释它们解决的问题及相互关系，再使用这些术语分析操作；A02 才要求读者能够辨认这些对象，并进一步学习参数、状态和完整生命周期。

## 目录规格

带 `[已有]` 的文件存在且包含内容；带 `[规划]` 的名称只登记在框架中，没有创建空文件。

```text
NVSHMEM/
├── 00-learning-path.md                                      [已有]
├── reference-nvshmem-overview.md                            [已有]
├── a01-rdma-communication-model.md                          [已有]
├── a02-rdma-resources-and-memory-registration.md            [已有]
├── a03-rdma-requests-and-buffer-protocol.md                 [已有]
├── a04-rping-full-rdma-flow.md                              [已有]
├── b00-nvshmem-programming-model.md                         [已有]
├── b01-symmetric-objects-and-remote-addressing.md            [已有]
├── b02-completeness-ordering-and-visibility.md              [已有]
├── b03-cuda-execution-and-cooperation.md                    [已有]
├── b04-data-publishing-and-buffer-ownership.md              [已有]
├── b05-atomic-operations-and-distributed-coordination.md     [已有]
├── b06-teams-and-collective-communication.md                [已有]
├── c00-runtime-initialization-and-control-plane.md           [已有]
├── c01-host-rdma-to-gpudirect-and-ibgda.md                  [已有]
├── c02-transport-verification-and-topology.md               [已有]
├── d00-ibgda-implementation-overview.md                     [已有]
├── d01-ibgda-request-lifecycle.md                           [规划]
├── d02-ibgda-concurrency-and-queue-management.md            [规划]
├── d03-ibgda-rc-dc-and-resource-management.md               [规划]
├── d04-ibgda-atomics-signals-and-ordering.md                [规划]
├── e01-performance-measurement-and-overlap.md               [规划]
├── e02-halo-and-moe-communication.md                        [规划]
├── e03-persistent-kernels-graph-and-work-queues.md           [规划]
├── experiments/
│   ├── experiment-a01-rdma-env-and-rping.md                 [规划]
│   ├── experiment-a02-rping-state-machine.md                [规划]
│   ├── experiment-b01-symmetric-addressing-and-put.md       [已有初稿]
│   ├── experiment-b02-put-get-completeness.md               [规划]
│   ├── experiment-b03-thread-block-cooperation.md           [规划]
│   ├── experiment-b04-single-dual-slot-pipeline.md          [规划]
│   ├── experiment-b04-protocol-model.py                     [规划]
│   ├── experiment-b05-atomic-coordination.md                [规划]
│   ├── experiment-b06-team-collectives.md                   [规划]
│   ├── experiment-c00-initialization-and-bootstrap.md        [规划]
│   ├── experiment-c01-transport-verification.md             [规划]
│   ├── experiment-d01-request-tracing.md                    [规划]
│   ├── experiment-d02-queue-concurrency.md                  [规划]
│   ├── experiment-d04-atomic-signal-path.md                 [规划]
│   ├── experiment-e01-performance-baseline.md               [规划]
│   └── experiment-e03-persistent-work-queue.md              [规划]
└── official-docs/
    ├── r01-symmetric-heap.md                                [规划]
    ├── r02-initialization.md                                [规划]
    ├── r03-usage-memory-model-and-consistency.md             [规划]
    └── r04-communication-and-scalability.md                 [规划]
```

`official-docs/` 尚无实际摘记，因此本轮不创建空目录。未来的参考摘记用于保存版本化核对结果，不复制官方原文，也不重复扩写课程正文。

## 课程主线

### A：从两机内存传输理解 RDMA

A01 是不要求 RDMA 经验的入口课。它从两个 Linux 进程的主机内存出发，先说明 RNIC 与用户态 verbs 的作用，建立 device context → PD/MR → CQ/QP → WR/WC 的最小资源模型，再比较 SEND/RECV、WRITE、READ 和 WRITE WITH IMM。重点不是 API 罗列，而是逐一追问发起方、目标地址的来源、接收方是否必须预贴接收、完成通知落在哪一侧，以及传输后谁可以复用哪个缓冲区。

A02 在 A01 已建立的直观模型上深入资源配置、建连和退出生命周期，系统讲解 device/port 查询、PD 保护关系、MR access flags、lkey/rkey、QP 能力与 RESET→INIT→RTR→RTS 状态、SQ/RQ/CQ 容量、RDMA CM 事件、失败回滚和反向销毁顺序。它不再承担术语的首次引入，而是解释如何正确创建、组合、排错和释放这些资源。

A03 沿 WR → SGE → WQE → CQE/WC 跟踪请求，区分软件描述、队列中的硬件工作项和完成记录。课程需要用 ready/ack 协议解释接收补充、源缓冲区复用、目标消费确认与背压。

[A04：从 rping 源码追踪一轮完整 RDMA 通信](a04-rping-full-rdma-flow.md)固定 rdma-core v60.0、commit `5321d809e095d6dd32a1355a5d4aa2ebbda4ba74` 核验 rping。课程沿客户端和服务端 call path，把控制消息、RDMA READ/WRITE、状态机、线程唤醒、消息长度和 MR 容量串成完整闭环，并明确区分固定源码事实、静态推导与未执行的硬件验证。

### B：从 verbs 语义映射到 NVSHMEM

[B00：NVSHMEM 编程模型](b00-nvshmem-programming-model.md)已经按零基础入口重写。课程先区分 job、PE、OS process、Team 与 CUDA thread/block，再从 PGAS 和 symmetric object 推导 `<symmetric address, target PE>`，说明 one-sided RMA 为什么只解耦数据移动与匹配接收而不取消同步，并统一整理 host/on-stream/device 发起域、thread-group API、collective launch、runtime/transport 边界和 MPI/NCCL 对比。正文按 NVSHMEM 3.7.2 发布周期的官方滚动 API 文档核对，示例未编译、未运行。

[B01：NVSHMEM 对称对象与远端寻址](b01-symmetric-objects-and-remote-addressing.md)聚焦 `<symmetric address, target PE>`。课程通过对象身份与对象内偏移推导远端位置，说明对称分配的集合生命周期，并区分 UVA、`nvshmem_ptr`、普通 local buffer registration 和真正映射进 symmetric heap 的 symmetric registration，避免“虚拟地址相同才能通信”或“注册后即可作为远端对象”的误解。

[B02：完成、排序与可见性](b02-completeness-ordering-and-visibility.md)把一次传输拆为源数据就绪、请求发起、本地完成、远端交付、目标观察和消费确认六个阶段，区分 blocking/NBI、`fence`、`quiet`、`flush`、signal/wait、barrier/sync 以及 host/device/on-stream 调用域。正文按 NVSHMEM 3.7.2 发布周期的官方滚动 API 文档核对契约，同时明确未绑定固定源码 commit。

[B03：CUDA 执行域、协作通信与前进性](b03-cuda-execution-and-cooperation.md)把 PE/Team、thread/warp/block/grid 与 host/device/stream 三套坐标分开，解释 thread-group API 的参与一致性、CTA 内两次线程交接、同一 Team 每 PE 恰好一个 collective 实例、collective launch 的驻留保证与边界，并用跨 block、跨 PE 和跨 stream 等待环分析 forward progress。正文按 NVSHMEM 3.7.2 发布周期滚动 API 文档与当前 CUDA 官方编程指南核对；示例未编译、未运行。

[B04：数据发布、缓冲区所有权与有界流水协议](b04-data-publishing-and-buffer-ownership.md)用 ticket 与每槽 `ready/ack` 序号统一推导单槽、双槽和 K 槽队列，分别证明本地 source 与远端 slot 的复用条件、容量背压、有限消息与 EOS 的最终排空，以及多生产者 atomic reservation 仍可能留下 publication hole。正文给出安全性不变量和活性假设，明确纸面协议没有经过编译与硬件验证。

[B05：远程原子操作与分布式协调](b05-atomic-operations-and-distributed-coordination.md)系统整理 non-fetching 与 fetching AMO、fetch-add、compare-swap、swap、set 和位运算的原子性及返回/完成边界，并区分“原子更新一个字”与“发布一段 payload”。课程从丢失更新和 ticket 唯一性证明出发，推导 distributed counter、CAS 状态转换、bump allocator、work queue reservation、publication hole、ABA/回绕和热点扩展问题；正文按 NVSHMEM 3.7.2 发布周期官方文档核对，代码与状态机未编译、未运行。

[B06：NVSHMEM Team 与集合通信](b06-teams-and-collective-communication.md)从 Team handle、world/team 编号翻译、split/destroy 生命周期和 collective matching 开始，系统区分 `sync`、barrier 与 broadcast、reduction、fcollect、all-to-all 的功能，并用公式说明 root、分块顺序、目标下标和缓冲区容量。课程同时整理 host、on-stream、device 与 warp/block 作用域、同一 Team 的并发限制、返回和完成边界，以及算法选择所需的拓扑与测量条件；正文按 NVSHMEM 3.7.2 发布周期官方滚动文档核对，不预设 collective 必然由某组 point-to-point primitives 实现，示例未编译、未运行。

### C：分离 API、数据路径与提交路径

[C00：NVSHMEM 运行时初始化与控制平面](c00-runtime-initialization-and-control-plane.md)沿 `nvshmem_init` 到 `nvshmem_finalize` 建立完整生命周期，区分 launcher、bootstrap、runtime 与 transport，解释 bootstrapped/device-initialized 两阶段状态、PE→GPU 绑定、三类初始化入口、动态与静态 symmetric heap、transport/NIC 准备、peer metadata 的逻辑职责、分层故障定位和反向释放。正文按 NVSHMEM 3.7.2 发布周期官方滚动文档核对；没有固定源码 commit，因此 endpoint、QP、key 和 device state 只写成有边界的职责或推论，示例未编译、未运行。

[C01：从 host RDMA 到 GPUDirect RDMA 与 IBGDA](c01-host-rdma-to-gpudirect-and-ibgda.md)把 API caller、NIC work submitter 与 payload data mover 分成三条独立轴，分别画出 host staging、host-posted GPUDirect RDMA、同节点 GPU P2P、device API + CPU proxy 和 device API + IBGDA/GDAKI 的控制路径与数据路径。课程特别说明 CPU proxy 不等于 host payload staging，device API 不等于 GPU doorbell，IBGDA transport 也必须继续核验实际 NIC handler。

[C02：NVSHMEM Transport 与 GPU–NIC 拓扑核验](c02-transport-verification-and-topology.md)把硬件能力、构建与配置意图、runtime selection、operation observation 和受控 A/B 分为不同证据层，系统整理 GPU/NIC BDF 与 NUMA 拓扑、RDMA fabric、GPUDirect RDMA、P2P、HCA mapping、IBGDA handler、日志、NVTX 与 counters 的核验方法。正文只提供未执行的采集和判定模板，不记录虚构环境结果；缺少证据时必须保持“未测试”或“未确认”。

### D：进入 IBGDA 实现

[D00：IBGDA 实现总览](d00-ibgda-implementation-overview.md)固定 NVIDIA 官方 NVSHMEM `v3.7.2-0`、commit `3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4`，从 plugin 装载、endpoint/QP/CQ 创建和 device state 交接进入稳态 RMA dispatch，建立地址/lkey/rkey、RC 与 DCI/DCT、WQ/WQE、DBREC/UAR、collapsed CQ，以及 `resv_head → ready_head → prod_idx → cons_idx` 多生产者协议的总图。课程同时区分 IBGDA transport selection、GPU NIC handler 与 CPU handler fallback；正文仅完成静态源码核对，没有编译或硬件运行。

D01 沿用 D00 的固定 NVSHMEM commit 逐步追踪一次 PUT，再与 GET、fetch-AMO 比较地址准备、WQE 填充、doorbell、排序和完成。不得在没有源码证据时把 `quiet` 描述成“等待某一条 CQE”。

D02 将 GPU 多生产者提交拆成预留、填充、ready 发布、提交、完成与回收，分析并发空洞、队列回绕、背压和批处理。纸面死锁与越界反例只做状态推演，不直接运行。

D03 对比 RC 与 DCI/DCT 的连接与资源模型，通过受控配置实验研究 QP 共享、连接规模和资源成本；实验观测与 API 保证分开记录。

D04 在与 D01 相同的固定 NVSHMEM commit 上，分别追踪 fetching/non-fetching AMO、put-with-signal、signal-op、fence 和 quiet 的地址/key、WQE opcode 或软件 fallback、完成记录与 ordering 边界。课程必须允许不同 transport 或 capability 走不同实现，不能先验断言所有 NVSHMEM atomic 都映射为原生 mlx5 atomic WQE。

### E：应用与性能

E01 建立延迟、带宽和 overlap 的测量口径，控制消息粒度、并发、批处理、预热、重复次数和基线。只有真实计算与通信时间线重叠并减少端到端时间，才能称为有效 overlap。

E02 把前述寻址、同步、ownership 和测量方法用于 halo 与 MoE 数据交换。应用课必须先画依赖与槽位协议，再选择 NVSHMEM RMA、signal、collective 或其他通信库。

E03 把 B03—B05 的前进性、槽位协议和原子预留用于 persistent kernel、图遍历与 distributed work queue，分析 termination detection、负载不均、抢占/驻留、publication hole 和失败传播。它关注综合正确性，不与 E01 的基准测试或 E02 的 halo/MoE 模式重复。

## 实验映射

| 实验 | 对应课程 | 核心问题 | 当前状态 |
| --- | --- | --- | --- |
| `experiment-a01-rdma-env-and-rping.md` | A01、A02 | 两端设备、IP、显式匹配的 `-S`、rping 成功与故障定位 | 规划 |
| `experiment-a02-rping-state-machine.md` | A03、A04 | 无硬件推演请求、事件依赖、地址范围和生命周期反例 | 规划 |
| [experiment-b01-symmetric-addressing-and-put.md](experiments/experiment-b01-symmetric-addressing-and-put.md) | B00、B01 | 两 PE 对称寻址和单元素 PUT | 初稿；未编译、未运行 |
| `experiment-b02-put-get-completeness.md` | B02 | blocking/NBI PUT/GET、源覆写与结果消费 | 规划 |
| `experiment-b03-thread-block-cooperation.md` | B03 | CTA 协作、线程交接、collective launch 与跨流依赖 | 规划 |
| `experiment-b04-single-dual-slot-pipeline.md` | B04 | 单槽/双槽、轮次、确认、背压与最终排空 | 规划 |
| `experiment-b04-protocol-model.py` | B04 | 有限状态模型发现覆盖反例，不替代硬件验证 | 规划 |
| `experiment-b05-atomic-coordination.md` | B05 | fetch-add/compare-swap、唯一 ticket、publication hole 与回收 | 规划 |
| `experiment-b06-team-collectives.md` | B06 | Team 编号、collective matching、buffer 约束和结果校验 | 规划 |
| `experiment-c00-initialization-and-bootstrap.md` | C00 | bootstrap、PE/GPU/NIC 映射、初始化失败与反向释放证据 | 规划 |
| `experiment-c01-transport-verification.md` | C01、C02 | 实际 transport/handler 与 GPU–NIC 路径证据 | 规划 |
| `experiment-d01-request-tracing.md` | D00、D01 | 固定 commit 的请求生命周期 | 规划 |
| `experiment-d02-queue-concurrency.md` | D02、D03 | 队列并发、回绕和资源配置 | 规划 |
| `experiment-d04-atomic-signal-path.md` | D04 | AMO/signal/ordering 的实际 opcode、fallback 与完成路径 | 规划 |
| `experiment-e01-performance-baseline.md` | E01、E02 | 可复现的延迟、带宽和 overlap 基线 | 规划 |
| `experiment-e03-persistent-work-queue.md` | E03 | persistent kernel 的任务发布、终止检测、前进性和负载不均 | 规划 |

## 每课写作与验收模板

后续生成正文时，每课至少包含前置知识、具体场景、机制与因果推导、代码或时序、错误反例、诊断题及参考推理、掌握标准和来源。结构可以按内容合并，不机械拆成大量短标题。

一课只有在学习者能够回答下列六问时，才具备进入实验的概念基础：

1. 谁发起这次操作？
2. 操作访问本地还是远端的哪一段内存？
3. 地址和访问权限从哪里获得，生命周期到何时？
4. 哪个事件仅代表提交，哪个事件代表本地或远端完成？
5. 谁以什么方式收到通知，通知覆盖哪一段内存序？
6. 生产者和消费者分别在什么条件下复用缓冲区？

## 分阶段掌握标准

| 阶段 | 掌握标准 |
| --- | --- |
| A 完成 | 能画出 SEND/RECV、WRITE、READ、WRITE WITH IMM 的两端时序；能解释 PD/MR/QP/CQ 与 WR/SGE/WQE/CQE；能从固定版本 rping 源码追踪完整生命周期 |
| B 完成 | 能从 symmetric address 与 PE 推导远端对象；能为 PUT/GET/NBI/signal 选择正确的完成与同步机制；能设计不覆盖未消费数据的单槽和双槽协议；能判断 AMO 只解决了唯一性还是已经闭合发布/回收；能正确匹配 Team collective |
| C 完成 | 能解释初始化中 bootstrap、GPU/transport 选择、heap/registration/endpoint setup 的依赖；能把 API 发起者、NIC 数据访问者和 NIC 工作提交者分别标出；能用实际拓扑与日志证明 transport，而不是凭机器型号猜测 |
| D 完成 | 能在固定 NVSHMEM commit 中追踪 PUT、GET、AMO、signal 和 ordering 的 IBGDA 路径；能解释 QP/WQE/doorbell/CQ 与并发队列的预留、发布、完成和回收 |
| E 完成 | 能设计可复现的性能基线，区分延迟、带宽和端到端 overlap；能把协议用于 halo、MoE、persistent kernel 或 graph/work queue，并解释正确性、活性和瓶颈 |

## 来源与证据策略

资料优先级为：API 契约与安装要求查 NVIDIA 官方 NVSHMEM 文档；verbs 与 rping 查 rdma-core 上游文档和固定版本源码；CUDA 行为查 CUDA 官方文档；硬件路径以实际拓扑、构建配置、运行日志和实验结果为准。

每条实现级结论记录版本或 commit、文件、symbol、核对日期和结论边界。参考摘记只做中文整理，不冒充官方原文。课程正文明确区分：

- API 保证：规范承诺、跨实现可依赖；
- 具体实现：固定版本源码中观察到的机制；
- 教学模型：为解释因果而简化的时序或状态机；
- 实验观测：在明确硬件、软件和拓扑上实际测得的结果。

硬件实验必须记录两端角色、完整构建与运行条件、校验方式、预期结果和实际结果。没有执行的命令标记为“待验证”，没有测量的数据不填写估计值。

## 进度表

状态维度独立记录；“—”表示本轮不适用，“未开始”不表示失败。

| 文件/模块 | 正文已写 | 源码已核对 | 模型已检查 | 已编译 | 硬件已运行 | 学习者已掌握 |
| --- | --- | --- | --- | --- | --- | --- |
| [reference-nvshmem-overview.md](reference-nvshmem-overview.md) | 初稿 | 未开始 | — | — | — | 未确认 |
| A01 | 已重写；作者自检 | 未开始 | 未开始 | 未开始 | 未开始 | 未确认 |
| [A02](a02-rdma-resources-and-memory-registration.md) | 已写；作者自检 | 当前上游手册已核对；非固定 commit | 静态检查 | 未开始 | 未开始 | 未确认 |
| [A03](a03-rdma-requests-and-buffer-protocol.md) | 已写；作者自检 | 当前上游手册已核对；非固定 commit | 静态检查 | 未开始 | 未开始 | 未确认 |
| [A04](a04-rping-full-rdma-flow.md) | 已写；作者自检 | 已核对 rdma-core v60.0 / `5321d809...` 的 `rping.c` | 静态检查 | 未开始 | 未开始 | 未确认 |
| [B00](b00-nvshmem-programming-model.md) | 已重写；作者自检 | 官方 NVSHMEM 3.7.2 发布周期 API 文档已核对；非固定源码 commit | 静态检查 | 未开始 | 未开始 | 未确认 |
| [B01](b01-symmetric-objects-and-remote-addressing.md) | 已写；作者自检 | 官方 NVSHMEM 3.7.2 发布周期 API 文档已核对；非固定源码 commit | 静态检查 | 未开始 | 未开始 | 未确认 |
| [B02](b02-completeness-ordering-and-visibility.md) | 已重写；作者自检 | 官方 NVSHMEM 3.7.2 发布周期 API 文档已核对；非固定源码 commit | 静态检查 | 未开始 | 未开始 | 未确认 |
| [B03](b03-cuda-execution-and-cooperation.md) | 已写；作者自检 | 官方 NVSHMEM 3.7.2 发布周期 API 文档与 CUDA 官方指南已核对；非固定源码 commit | 静态检查 | 未开始 | 未开始 | 未确认 |
| [B04](b04-data-publishing-and-buffer-ownership.md) | 已写；作者自检 | 官方 NVSHMEM 3.7.2 发布周期 API 文档已核对；非固定源码 commit | 状态模型静态检查 | 未开始 | 未开始 | 未确认 |
| [B05](b05-atomic-operations-and-distributed-coordination.md) | 已写；作者自检 | 官方 NVSHMEM 3.7.2 发布周期 API 文档已核对；非固定源码 commit | 原子性与队列不变量静态检查 | 未开始 | 未开始 | 未确认 |
| [B06](b06-teams-and-collective-communication.md) | 已写；作者自检 | 官方 NVSHMEM 3.7.2 发布周期 API 文档已核对；非固定源码 commit | Team 生命周期、collective matching 与数据布局静态检查 | 未开始 | 未开始 | 未确认 |
| [C00](c00-runtime-initialization-and-control-plane.md) | 已写；作者自检 | 官方 NVSHMEM 3.7.2 发布周期 API 文档已核对；初始化源码 commit 待选择 | 初始化状态机、生命周期与控制面职责静态检查 | 未开始 | 未开始 | 未确认 |
| [C01](c01-host-rdma-to-gpudirect-and-ibgda.md) | 已写；作者自检 | 官方 NVSHMEM 3.7.2 发布周期文档、CUDA 13.4 GPUDirect RDMA Guide 与当前 CUDA P2P 文档已核对；非固定源码 commit | 路径模型静态检查 | 未开始 | 未开始 | 未确认 |
| [C02](c02-transport-verification-and-topology.md) | 已写；作者自检 | 官方 NVSHMEM 3.7.2 发布周期文档、CUDA GPUDirect RDMA Guide 与 `nvidia-smi` 文档已核对；非固定源码 commit | transport 证据层、拓扑映射与 A/B 流程静态检查 | 未开始 | 未开始 | 未确认 |
| [D00](d00-ibgda-implementation-overview.md) | 已写；作者自检 | 已核对 NVIDIA NVSHMEM v3.7.2-0 / `3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4` | 初始化/dispatch、地址/key、QP/WQE/doorbell/CQ 与多生产者状态静态检查 | 未开始 | 未开始 | 未确认 |
| D01–D04 | 未开始 | 沿用 D00 固定 commit | 未开始 | 未开始 | 未开始 | 未确认 |
| E01–E03 | 未开始 | — | 未开始 | 未开始 | 未开始 | 未确认 |
| [实验 B01](experiments/experiment-b01-symmetric-addressing-and-put.md) | 初稿 | — | 静态检查 | 未验证 | 未验证 | 未确认 |
| 其余实验 | 未开始 | 按实验决定 | 未开始 | 未开始 | 未开始 | 未确认 |

## 当前阅读入口

完整主线应依次阅读 [A01：从零开始理解 RDMA 通信模型](a01-rdma-communication-model.md)、[A02：RDMA 资源、内存注册与建连生命周期](a02-rdma-resources-and-memory-registration.md)、[A03：RDMA 请求、完成与缓冲区协议](a03-rdma-requests-and-buffer-protocol.md)和 [A04：从 rping 源码追踪一轮完整 RDMA 通信](a04-rping-full-rdma-flow.md)。完成 A 模块后继续阅读 [B00：NVSHMEM 编程模型](b00-nvshmem-programming-model.md)、[B01：NVSHMEM 对称对象与远端寻址](b01-symmetric-objects-and-remote-addressing.md)、[B02：完成、排序与可见性](b02-completeness-ordering-and-visibility.md)、[B03：CUDA 执行域、协作通信与前进性](b03-cuda-execution-and-cooperation.md)、[B04：数据发布、缓冲区所有权与有界流水协议](b04-data-publishing-and-buffer-ownership.md)、[B05：远程原子操作与分布式协调](b05-atomic-operations-and-distributed-coordination.md)和 [B06：NVSHMEM Team 与集合通信](b06-teams-and-collective-communication.md)，依次解决基础 API 的参与、寻址、完成、前进性、ownership、原子协调和 collective 数据布局。随后阅读 [C00：NVSHMEM 运行时初始化与控制平面](c00-runtime-initialization-and-control-plane.md)、[C01：从 host RDMA 到 GPUDirect RDMA 与 IBGDA](c01-host-rdma-to-gpudirect-and-ibgda.md)和 [C02：NVSHMEM Transport 与 GPU–NIC 拓扑核验](c02-transport-verification-and-topology.md)，依次理解第一个 PUT 前的控制面、稳态请求的提交与 payload 路径，以及如何用运行证据确认实际 transport。然后进入 [D00：IBGDA 实现总览](d00-ibgda-implementation-overview.md)，在固定 NVSHMEM commit 上建立 host control plane、device dispatch、QP/WQE/doorbell/CQ 与多生产者队列的源码地图；下一篇应继续 D01 的具体请求追踪。已有 [实验 B01](experiments/experiment-b01-symmetric-addressing-and-put.md)可用于纸面推演或环境验证；其余配套实验仍是规划，不能据正文状态推断已运行。
