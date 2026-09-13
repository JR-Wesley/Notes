# A01 从零开始理解 RDMA 通信模型

## 本课要建立的模型

这篇笔记是整个 NVSHMEM 路线的 RDMA 零基础入口。读者只需要知道两个 Linux 进程拥有彼此独立的虚拟地址空间，并理解 C/C++ 指针和对象生命周期；不要求事先认识 verbs、context、PD、MR、QP 或 CQ。本文从一个稳定的问题出发：主机 A 的进程拥有一段 payload，怎样把它交给主机 B 的进程，并准确判断谁发起访问、数据落到哪里、访问权从哪里取得、哪一个事件表示完成，以及何时可以复用两端缓冲区。后续学习 NVSHMEM、GPUDirect RDMA 和 IBGDA 时，这组问题仍然成立，只是请求发起者、可访问的内存类型和队列提交路径会发生变化。

本文讨论用户态 RDMA 的通用通信模型，并用 rdma-core/libibverbs 的术语说明接口。讲授顺序是“问题与路径 → 最小资源模型 → 一次请求的生命周期 → 四类通信操作 → 完成和缓冲区所有权”，而不是先给出术语清单。A02 会在这些对象已经具备直观含义后，深入它们的创建参数、连接状态、权限组合和销毁依赖；A03 再详细分析请求描述与队列记录。文中的队列图是教学模型，不代表某一款 NIC 的内部微架构；没有硬件实验支持的性能、缓存一致性和具体 provider 行为不在本课中推断。

## 从普通消息传输到 RDMA

普通 socket 程序通常把 `send()` 理解为“将一段字节交给内核协议栈”，接收进程再通过 `recv()` 把字节取回用户缓冲区。实际路径可能包含系统调用、内核缓冲和数据复制，远端 CPU 也要运行接收路径。RDMA 的核心变化不是让两台机器真正共享同一个虚拟地址空间，而是让应用预先建立受保护的通信资源，再把数据移动描述提交给 RNIC（RDMA-capable NIC）。在资源与权限已经准备好的条件下，RNIC 可以直接通过 DMA（Direct Memory Access）读写已登记内存；DMA 表示设备无需 CPU 执行逐字节复制循环即可访问内存，不表示 CPU、驱动和内核从资源初始化及错误处理中消失。RNIC 通过队列执行请求，因此稳态数据面可以避免远端 CPU 为每次 READ/WRITE 调用接收函数，也可以减少内核参与和不必要的数据复制。

这里的“direct”需要谨慎理解。它表示数据移动可以由 NIC 在已授权内存之间完成，不表示网络没有分包、PCIe 传输、缓存与内存控制器，也不保证任何程序都天然实现严格意义上的零复制。类似地，“remote memory access”表示请求能够携带远端地址和访问凭据，由远端 RNIC 检查并访问指定范围；它不表示本地进程可以直接解引用远端 C 指针，更不表示仅知道一个数值地址就获得了访问权。

libibverbs 是 Linux 用户态程序访问 RDMA 设备的一组标准接口，接口中的各类操作通常称为 verbs。可以把一条典型的数据通路概括为：应用通过 verbs 在控制面创建资源并交换连接参数、远端地址和 key；发起侧把 Work Request（WR）提交到 Queue Pair（QP）的 Send Queue（SQ）；provider 将通用请求翻译为设备能够执行的工作项并通知 NIC；NIC 读取本地数据或接收远端返回值，通过网络与对端 NIC 交互；最后，完成结果进入 Completion Queue（CQ），应用轮询或等待完成事件。建连、地址交换和 key 交换属于控制面，真正的 payload 搬运属于数据面。控制面可以使用 TCP、RDMA CM 或其他 bootstrap 机制，但这种选择不改变 RMA 操作本身的语义。

```text
主机 A                                                        主机 B
应用/协议                                                     应用/协议
  │ WR: opcode + local SGE + remote_addr/rkey                    │
  ▼                                                              │
QP.SQ → provider/WQE → RNIC A ═══════ fabric ═══════ RNIC B → 已登记内存
  │                                      数据与传输确认           │
  └──────────── CQE/WC：本端完成 ────────────────────────────────┘

注意：上图中的本端完成不是“主机 B 的应用已经处理完数据”。
```

## 建立最小资源模型：通信为什么需要这些对象

第一次接触 verbs 时，context、PD、MR、QP 和 CQ 容易被误认为一组必须死记的缩写。实际上，它们分别回答五个不同问题：程序正在使用哪块 RDMA 设备；哪些资源属于同一个保护边界；NIC 可以访问哪些内存；请求从哪个通信端点发出或接收；程序到哪里取得完成结果。如果缺少其中任何一层，应用就不能同时满足“找到硬件、限制权限、描述内存、提交工作和确认结果”这些基本要求。

最小资源关系可以先记为以下有向关系。这里的箭头表示创建或关联依赖，不表示数据一定沿着箭头复制。一个 context 可以承载多个 PD、CQ 和其他设备资源；一个 PD 可以包含多个 MR 和 QP；QP 的发送、接收完成可以关联到同一个或不同的 CQ。

```text
RDMA device
    │ open
    ▼
device context
    ├── Protection Domain (PD)
    │      ├── Memory Region (MR) → 地址范围、访问权限、lkey/rkey
    │      └── Queue Pair (QP) ───→ Send Queue (SQ) + Receive Queue (RQ)
    └── Completion Queue (CQ) ←── QP 执行后产生的 Work Completion
```

### Device context：程序使用某块 RDMA 设备的入口

一台主机可能有零块、一块或多块 RDMA 设备，每块设备还可能有多个端口。程序先通过 `ibv_get_device_list()` 枚举设备，再用 `ibv_open_device()` 打开选定设备；返回的 `ibv_context` 是后续访问该设备的用户态上下文。这里的 context 不是 CUDA context，也不是线程上下文，而是“本进程与这块 RDMA 设备交互所使用的资源入口”。CQ、PD 和 QP 等对象最终都从某个 device context 派生，因此它们不能脱离所属设备随意组合。[rdma-core 的设备枚举与打开手册](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_get_device_list.3.md)还明确要求，在释放设备列表前先打开之后仍要使用的设备。

打开 context 并不等于网络连接已经建立。它只说明程序能够查询和创建这块设备上的资源；端口是否 active、链路使用 InfiniBand 还是 RoCE、对端地址是什么、QP 是否进入可发送状态，仍需后续配置。退出时也不能先关闭 context 再期待其子资源自动安全释放；[`ibv_open_device(3)`](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_open_device.3)要求应用先释放由该 context 创建的关联资源，以免泄漏或破坏生命周期。

### PD 与 MR：把一段普通内存变成受保护的 NIC 可访问区域

Protection Domain（PD）是设备内部的保护边界。应用用 `ibv_alloc_pd(context)` 从某个设备上下文分配 PD，之后在这个 PD 中注册 MR、创建 QP 或其他相关对象。PD 本身不存储 payload，也不提供网络连接；它的作用是约束“哪个通信资源可以使用哪份内存授权”。如果一个 QP 和一个 MR 不属于可匹配的保护关系，仅有正确地址也不足以让请求合法执行。

应用通过 `malloc()`、`mmap()` 或其他分配器得到的只是普通虚拟内存。Memory Region（MR）是调用 `ibv_reg_mr(pd, address, length, access_flags)` 后生成的注册对象，它记录一段地址范围、所属 PD 和允许的访问方式，并向程序提供 `lkey` 与 `rkey`。`lkey` 随本地 SGE 一起使用，使 NIC 能验证本地缓冲区；`rkey` 是可以交给远端请求者的访问凭据，用于验证远端 RDMA READ、WRITE 或 Atomic。注册授权的是有限区间而不是整个进程地址空间，所以 `remote_addr`、传输长度与 MR 边界必须同时匹配。

注册不能简单理解为“把地址发给网卡”或“必然锁页”。经典注册通常建立稳定的 DMA 映射，但 On-Demand Paging、DMA-BUF MR 等机制具有不同的页驻留和映射方式；跨实现稳定的结论是，MR 建立了 NIC 可使用的地址转换与访问控制。根据 [`ibv_reg_mr(3)`](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_reg_mr.3)，本地 SGE 使用 `lkey`，远端进程将 `rkey` 放入 RDMA/Atomic 请求。只交换 `rkey` 而没有对应地址和长度协议仍无法正确定位对象，提前释放底层缓冲区或注销 MR 则会使尚未完成的请求失去有效生命周期。

### CQ 与 QP：一个负责承载工作，一个负责报告结果

Completion Queue（CQ）是保存完成记录的队列。程序通过 `ibv_create_cq(context, cqe, ...)` 申请至少能够容纳一定数量 completion entry 的 CQ，再通过轮询或完成事件知道有工作结果可取。CQ 不存放业务 payload，也不自动说明“对端应用已处理数据”；它保存的是 Work Completion（WC），其中包括 `wr_id`、状态、opcode、传输长度和可选 immediate data 等字段。应用必须取出并检查 WC，尤其不能把“收到了事件通知”与“已经检查到成功 WC”混为一谈。[`ibv_create_cq(3)`](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_create_cq.3)还规定，仍有关联 QP 时不能销毁 CQ，这体现了资源释放的依赖顺序。

Queue Pair（QP）是提交和接收入站工作的通信端点，由 Send Queue（SQ）和 Receive Queue（RQ）组成。名称中的 pair 指这两条本地工作队列，不是指“两台主机各有一个队列”。本端应用把 SEND、RDMA WRITE、RDMA READ 等主动请求提交到 SQ；RQ 保存预贴的 Receive Request，供入站 SEND 或 WRITE WITH IMM 的通知消费。创建 QP 时要指定所属 PD、发送/接收 CQ、队列容量、SGE 能力和传输服务类型，rdma-core 的 [`ibv_create_qp(3)`](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_create_qp.3)正是通过这些关联把保护、工作队列和完成报告连接起来。

创建 QP 后仍不能立即假定它能与任意对端通信。以 RC 为例，QP 通常经历 RESET、INIT、RTR（Ready to Receive）和 RTS（Ready to Send）等状态，双方要交换 QP number、地址标识、路径参数以及应用所需的远端内存信息。程序可以手工交换并配置这些信息，也可以使用 RDMA CM 完成地址解析、路由解析、连接请求和状态转换。RDMA CM 是建连控制工具，不负责替应用决定 payload 格式、目标槽位或消费确认。

### 把资源准备过程连成一条可执行逻辑

对一个最小的两端 RC 程序，可以先把初始化理解为五个阶段。第一阶段选择并打开设备，得到 context；第二阶段创建 PD，并把本地发送、接收或远端目标缓冲区注册为 MR；第三阶段创建 CQ 和 QP，使 QP 归属该 PD 并把完成指向 CQ；第四阶段由控制面交换端点信息以及必要的 `remote_addr + rkey`，把双方 QP 推进到可通信状态；第五阶段才是预贴 Receive Request、提交发送或 RMA 请求并轮询 WC。实际 API 调用存在错误分支和更细的状态参数，但所有细节都服务于这条主线。

销毁过程原则上反向进行：先停止产生新请求并排空在途工作，再解除连接和销毁 QP，之后才能安全销毁仍被 QP 引用的 CQ、注销 MR、释放 PD，最后关闭 device context。具体程序可能还包含 completion channel、SRQ、Address Handle 或 RDMA CM ID，释放顺序必须依据真实依赖补充。A01 要掌握的是“父资源和被引用资源不能提前消失”；A02 将进一步给出完整创建/失败回滚/正常退出状态表。

## 地址、可达性、权限和就绪是四个条件

分析 RDMA 故障时，最有价值的习惯是把四个容易混淆的条件分开。第一，地址必须描述有效对象及合法的 `[base, base + length)` 范围；第二，通信端点和网络路径必须可达，QP 状态、路由、端口和传输类型要允许请求到达；第三，发起者必须持有与目标 Memory Region（MR）匹配的 `rkey`，目标 MR 和 QP 还必须启用相应远端访问权限；第四，即使数据已经被 RNIC 放入合法地址，接收应用也只有在协议规定的 ready 条件满足后才能消费。前三项解决“能不能访问”，最后一项解决“现在读到的是否是本轮完整数据”。

上述资源模型说明了为什么地址不能独立构成访问能力。MR 给出有限范围和 key，PD 与 QP 构成保护关系，QP 状态和网络路径决定请求能否到达，应用协议则决定到达的数据是否属于当前轮次。任何一项正确都不能替代其他条件；MR 和底层缓冲区还必须覆盖请求的整个执行期。

## 从 WR 到完成记录

应用提交的 WR 是软件层对一次操作的描述，其中 opcode 指定 SEND、WRITE 或 READ 等动作，SGE 数组描述本地内存片段，RMA 请求还携带 `remote_addr` 和 `rkey`。根据 [`ibv_post_send(3)`](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_post_send.3)，调用成功表示 WR 链表已经被提交给指定 QP 的发送队列；这不是数据已经越过网络的证明。硬件执行队列中的工作项通常称为 WQE（Work Queue Element），而完成队列中的硬件记录通常称为 CQE（Completion Queue Element）；[`ibv_poll_cq()`](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_poll_cq.3) 返回给程序的是解析后的 WC（Work Completion）。因此 WR/WQE 和 CQE/WC 分别位于软件描述、硬件队列与软件观察层，日常口语可以简写，但源码分析时不能把它们当成同一个对象。

发送 WR 是否产生成功完成，取决于 QP 配置和 `IBV_SEND_SIGNALED` 等设置。选择 unsignaled 并不表示请求无需完成，而是成功路径不为每条请求都生成可轮询的发送 WC；程序仍必须按照队列规则控制在途请求、错误和资源回收。对于普通非 inline 的发送缓冲区，rdma-core 的 `ibv_post_send(3)` 明确要求在请求完全执行并取回相应 work completion 后再安全复用；若使用 `IBV_SEND_INLINE`，数据在 post 返回前已经复制进 WQE，原应用缓冲区可以更早复用。这个例外恰好说明“API 返回”“NIC 不再读取源缓冲区”和“远端应用已消费”是三种不同事件。

## 四类操作的寻址与接收关系

下面四种操作都从发起侧 SQ 提交，但它们在“谁决定目标地址”和“远端是否消耗 Receive Request”上存在根本差异。表格先给出结论，后续段落解释为什么这些差异会改变应用协议。

| 操作 | payload 的本地描述 | 目标位置由谁提供 | 远端是否必须预贴 Receive Request | 远端通常观察到什么 |
| --- | --- | --- | --- | --- |
| SEND/RECV | 发送端 SGE 描述源；接收端 Receive WR 的 SGE 描述目标 | 接收端 | 是 | 接收 WC，payload 位于该 Receive WR 指定的缓冲区 |
| RDMA WRITE | 发送端 SGE 描述源 | 发送端在 WR 中给出 `remote_addr + rkey` | 否 | 默认没有由该 WRITE 直接产生的接收 WC |
| RDMA READ | 发起端 SGE 描述本地结果缓冲区 | 发起端在 WR 中给出 `remote_addr + rkey` | 否 | 响应端应用通常没有每请求通知；发起端等待 READ 完成 |
| RDMA WRITE WITH IMM | 发送端 SGE 描述源，payload 仍写到远端地址 | 发送端在 WR 中给出 `remote_addr + rkey` | 是，通知会消耗一个 Receive Request | 接收 WC 携带 32 位 immediate data；payload 不由 Receive WR 选址 |

### SEND/RECV：接收者拥有放置权

SEND 是双边消息语义：发送者负责提供源 SGE，接收者必须提前把 Receive WR 放入 RQ 或 Shared Receive Queue（SRQ），由 Receive WR 的 SGE 决定入站消息写到哪个本地缓冲区。发送 WR 不携带应用层远端虚拟地址和 `rkey`，所以发送者只表达“把这条消息交给对端接收队列”，不能选择对端任意对象。接收成功后，接收 CQ 中的 WC 把某个 `wr_id` 与收到的字节数关联起来，接收程序据此取得该缓冲区的所有权并解析消息。

这一模型的资源约束是 receive credit。若对端没有可消费的 Receive WR，可靠连接可能进入 Receiver Not Ready（RNR）处理并按配置重试，非可靠传输则可能直接丢失；因此“连接已经建立”不等于“任意时刻都可以 SEND”。应用通常要预贴一批接收缓冲区，在处理完一个接收完成后及时补充 RQ，并确保补充速度能够承受峰值流量。此处的双边并不要求两边同时调用函数，而是指发送请求的成功依赖接收侧预先参与缓冲区供给。

### RDMA WRITE：发起者拥有放置权

RDMA WRITE 的 WR 同时描述本地源 SGE、远端起始地址和 `rkey`，所以发起端 RNIC 可以把 payload 直接放进目标 MR 的指定偏移。普通 WRITE 不消耗远端 Receive WR，也通常不会仅因写入完成就在远端 CQ 生成一个接收 WC。它因此非常适合 push 型大数据传输：远端应用不必为每块 payload 执行一次 `recv()`，发送者还可以把数据精确放到预先分配的槽位。但代价是协议必须另行解决目标地址/key 的交换、槽位分配和通知，否则远端应用不知道哪一轮数据已经完整到达。

发送侧 WRITE 完成可用于判断本地源缓冲区何时不再被该 WR 使用，却不能证明远端业务逻辑已经读完目标槽位。即使传输层提供了可靠交付和有序执行，应用消费仍是更高一层事件：远端线程可能尚未被调度，可能仍在处理上一轮，也可能尚未观察协议的 ready 标志。因此，生产者在收到 WRITE 的本地完成后可以按 verbs 契约处理源缓冲区，却不能据此覆盖仍属于消费者的远端槽位。

### RDMA READ：请求者拉取并在本地完成

RDMA READ 同样由请求者的 SQ 发起，但数据方向与请求方向相反。WR 中的 `remote_addr + rkey` 指定远端源，发起端 SGE 指定返回数据写入的本地目标；对端 RNIC 读取获授权的 MR 并返回数据，对端应用既不需要预贴 Receive WR，也通常不会为每次 READ 获得一个接收完成。发起侧只有在成功的 READ work completion 到达后，才能把本地目标缓冲区当作此次 READ 的结果使用。

READ 适合消费者明确知道数据位置并希望主动拉取的场景，但它没有自动解决“远端数据是否已发布”。如果生产者正在修改对象，而消费者只凭地址和 key 发起 READ，消费者仍可能取得旧值或协议上不完整的一轮数据。正确设计仍需版本号、ready/signal、锁或其他发布机制建立先后关系；RMA 的读能力不是一致性协议本身。

### WRITE WITH IMM：数据放置与通知绑定，但仍不是消费确认

RDMA WRITE WITH IMM 的 payload 放置方式与 WRITE 相同：目标由发起者给出的远端地址和 `rkey` 决定。额外的 32 位 immediate data 随请求到达，并在远端以 `IBV_WC_RECV_RDMA_WITH_IMM` 类型的接收完成上报。这里最容易犯的错误是把它理解成“WRITE 不需要 RECV，WITH IMM 也不需要”。[rdma-core 扩展 WR 手册](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_wr_post.3.md)说明了远端接收完成及立即数，[NVIDIA 用户态 RDMA 编程手册 1.7](https://docs.nvidia.com/rdma-aware-networks-programming-user-manual-1-7.pdf)进一步明确 WRITE WITH IMM 会消耗一个 Receive Request；远端若想持续收到通知，就必须维持 RQ/SRQ credit。

被消耗的 Receive Request 承担的是接收事件资源，而不是决定 WRITE payload 的放置位置。应用可以把 immediate data 用作槽位号、消息类别或短序号，在观察到成功接收 WC 后再读取约定的远端目标区域。这个完成能够把数据到达与一个轻量通知关联起来，但它仍不代表消费者已经处理完 payload，也不自动授予发送者覆盖槽位的权利。若槽位会循环复用，消费者仍需通过反向 WRITE、SEND、Atomic 或其他协议返回 ack/credit。

## 传输服务与网络承载不是同一层

“RDMA 用什么网络”和“QP 提供什么传输服务”是两个层次。InfiniBand 是原生 RDMA 网络架构；RoCE 在以太网上承载 RDMA 语义，其中 RoCE v2 可经过 IP 路由；iWARP 在 TCP/IP 上实现 RDMA。libibverbs 给应用提供相近的 verbs 对象，但设备能力、可用 opcode、拥塞行为和部署要求仍由具体网络及 provider 决定。写程序时不能因为 API 名称相同，就假设不同设备支持完全相同的特性或具有相同的性能。

在 libibverbs QP 类型中，RC（Reliable Connection）、UC（Unreliable Connection）和 UD（Unreliable Datagram）体现不同服务。根据 rdma-core [`ibv_post_send(3)` 的 opcode 支持表](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_post_send.3)，SEND 可用于 UD、UC 和 RC，RDMA WRITE 可用于 UC 和 RC，而 RDMA READ 只用于 RC（以及本课不展开的 XRC send）。RC 提供面向连接的可靠传输与有序语义，并对丢包等情况执行协议级处理；UC 仍是面向连接且支持 WRITE，但不提供 RC 的可靠重传保证；UD 以数据报方式工作，不支持普通 RDMA READ/WRITE，单条消息还受设备 MTU 等条件约束。选择传输类型会改变失败处理、接收资源和可用操作，却不会消除应用层的 payload/ready/ack 责任。

| 层次 | 典型对象或技术 | 它回答的问题 |
| --- | --- | --- |
| 应用协议 | 槽位、序号、ready、ack、重试策略 | 哪一轮数据属于谁，何时可以消费或覆盖 |
| RDMA API | libibverbs、RDMA CM | 怎样创建资源、提交 WR、观察 WC |
| 传输服务 | RC、UC、UD | 是否连接、是否可靠、哪些 opcode 可用 |
| 网络承载 | InfiniBand、RoCE、iWARP | 报文在哪种网络上交换，怎样路由和拥塞控制 |
| provider/驱动/NIC | 用户态 provider、uverbs、设备驱动、RNIC | 如何把通用 verbs 映射为设备资源与硬件队列 |

libibverbs 的快路径常被概括为“kernel bypass”，但这也需要分层理解。资源创建、内存注册、权限建立和异常处理仍需要内核及驱动参与；资源准备好以后，用户态 provider 可以通过映射的队列和 doorbell 等机制让应用直接驱动硬件快路径，从而避免每条数据操作都进入内核。IBGDA 以后要改变的是“由谁填充并提交 NIC 工作”这一环节，而不是取消地址、key、队列容量和完成语义。

## 完成、通知与缓冲区所有权

对一次传输，至少要追踪五个不同边界。`post` 成功是提交边界，表示 WR 已被接受；NIC 开始读取非 inline 源缓冲区后，该缓冲区仍归请求使用；本端成功 WC 给出 verbs 定义的本地完成边界，使应用能够按操作类型复用源或读取 READ 结果；目标数据放置完成是远端内存事件；远端应用真正处理完 payload 则是消费边界。不同 opcode 能直接观察的边界不同，应用不能用一个完成事件替代全部状态。

以 A 向 B 循环推送数据为例，安全的单槽协议可以写成以下状态链。A 只有在持有 slot credit 时写 payload；随后发布 ready 或通过 WRITE WITH IMM 通知 B；B 观察到对应轮次的通知后读取 payload，处理结束再返回 ack；A 收到 ack 才重新获得目标槽位。A 的发送完成只允许它判断本地源何时可复用，B 的 ready 只允许它开始消费，真正允许 A 覆盖远端槽位的是消费 ack。若省略最后一步，网络越快、生产者越积极，反而越容易覆盖尚未消费的数据。

```text
A（生产者）                                      B（消费者）
持有 slot credit
写 payload ------------------------------------> slot[n]
发布 ready / WRITE_WITH_IMM -------------------> 检查轮次并开始消费
本地发送完成：可处理 A 的源缓冲区                 读取并处理 slot[n]
等待 ack <-------------------------------------- 返回 ack/credit
收到 ack：才可覆盖 B 的 slot[n]
```

如果改用 SEND/RECV，B 通过预贴 Receive WR 直接控制入站缓冲区，接收 WC 通常同时承担“消息已进入指定接收缓冲区”的通知，因此 payload 放置与接收通知天然配对；但 B 仍须在消费结束后决定何时重新把该缓冲区作为 Receive WR 发布。由此可见，SEND/RECV 与 WRITE 的核心差异不只是函数名称，而是缓冲区供给权分别掌握在接收者和发送者手中。

## 与 NVSHMEM 和 IBGDA 的连接

NVSHMEM 的 `put`/`get` 延续了 RMA 的主动方模型：调用 PE 根据对称对象和目标 PE 定位远端对象，runtime 负责隐藏部分连接、key 管理和 transport 选择。对称对象并不要求各 PE 的虚拟地址数值相同；runtime 可以根据本地对称地址的对象内偏移和目标 PE 的映射信息得到实际远端位置。也正因为 runtime 隐藏了显式 `remote_addr + rkey`，应用更容易误以为寻址、权限、完成和消费是一个动作，而实际上这些责任仍然分层存在。

后续 [B00：NVSHMEM 编程模型](b00-nvshmem-programming-model.md) 会解释 PE、PGAS 和对称堆，[B02：完成、排序与可见性](b02-completeness-ordering-and-visibility.md) 会依照 NVSHMEM API 契约区分 blocking/NBI、`fence`、`quiet`、barrier、signal 与等待。这里必须提前保留两条边界：NVSHMEM `fence` 不是 verbs WR 上 `IBV_SEND_FENCE` 标志的同义词，NVSHMEM `quiet` 也不能在没有固定版本源码证据时解释成“等待某一条 CQE”。它们属于不同抽象层，只能根据各自规范讨论作用域和保证。

GPUDirect RDMA 与 IBGDA 又分别改变不同问题。GPUDirect RDMA 关注 NIC 是否能够直接 DMA 访问 GPU memory；GPU 发起 NVSHMEM device API 关注哪一类执行单元调用通信接口；IBGDA 则进一步关注 GPU 是否能够直接构造或提交 NIC 工作。这三个能力可以组合，但逻辑上互不等价。即使请求由 GPU 直接提交，远端地址映射、访问 key、队列容量、顺序、完成和槽位所有权仍然必须由 runtime 或应用协议正确维护。

## 典型错误与诊断方法

遇到保护错误、超时或偶发数据错乱时，按层定位比反复增加 barrier 更有效。若 WR 立即提交失败，先检查 opcode、SGE 数量、队列容量和 QP 状态；若 WC 返回错误，检查 WC status，并进一步核对远端地址范围、`rkey`、MR access flags、QP 状态和连接参数；若没有错误但对端一直等不到事件，确认所用操作是否本来就产生远端接收完成，以及 RQ/SRQ 是否有 credit；若事件能够收到但数据轮次错误，则检查 payload 与 ready 的发布顺序、序号匹配和 ack 之前的槽位覆盖。同步原语只能建立其契约规定的顺序和完成关系，不能修复错误地址、失效 key 或没有 Receive WR 的资源问题。

另一个常见错误是把可靠传输等同于应用协议正确。RC 可以处理链路中的丢包和顺序问题，却不知道某个字节区间是否构成一条完整业务记录，也不知道消费者何时处理结束。相反，应用层即使设计了序号和 ack，也不能弥补越界 remote address、过期 `rkey` 或提前注销 MR。可靠性必须按网络传输、verbs 资源和业务所有权三层分别证明。

## 诊断题与参考推理

1. **A 的 `ibv_post_send()` 返回 0，是否可以立刻改写非 inline 源缓冲区？** 不可以。返回 0 只表示 WR 已成功提交；NIC 之后仍可能读取该缓冲区。需要等待与请求关联的成功完成，或者使用满足限制的 inline 发送并依据其特殊复用规则处理。
2. **B 没有预贴 Receive WR，A 的普通 RDMA WRITE 是否一定失败？** 不一定。普通 WRITE 不消费 B 的 RQ，但 B 必须事先提供有效 `remote_addr + rkey`，MR/QP 权限与生命周期也必须正确。若换成 WRITE WITH IMM，B 还需要可消费的 Receive Request 承载通知。
3. **A 收到 WRITE 的发送完成，能否立即覆盖 B 的目标槽位？** 不能由此推出。该完成解决 A 侧请求与源资源的本地完成问题，不证明 B 的应用已经消费目标槽位；覆盖需要独立的 ack 或 credit 协议。
4. **为什么 SEND 不要求发送者知道 B 的目标虚拟地址？** 因为接收位置由 B 预贴的 Receive WR/SGE 决定。发送者选择通信端点并提交消息，接收者保有缓冲区放置权。
5. **为什么拥有正确的远端地址仍可能收到 remote access error？** 地址值只是定位信息；`rkey`、MR 范围、access flags、PD/QP 关系和对象生命周期共同构成授权条件，任一不匹配都可能使访问失败。
6. **GPU 能调用 NVSHMEM device API，是否证明当前数据通过 IBGDA 由 GPU 直接敲 NIC doorbell？** 不能。API 发起者、NIC 可访问的内存类型和 NIC 工作提交者是三个独立问题，必须结合 runtime 配置、transport/handler 日志、拓扑和固定版本源码判断实际路径。

## 掌握标准与证据边界

完成本课后，应能在纸上分别画出 SEND/RECV、RDMA WRITE、RDMA READ 和 WRITE WITH IMM 的两端时序，并为每一种操作标出本地 SGE、远端地址/key、RQ 是否被消费、发送与接收完成落在哪一侧。面对一个“发送已经完成但数据仍然错乱”的问题，应能进一步追问完成属于哪一层、目标是否获得 ready、消费者是否返回 ack，而不是笼统地增加同步。还应能解释 RC/UC/UD 与 InfiniBand/RoCE/iWARP 不在同一分类层，并说明 kernel bypass 只描述稳态快路径，不能取消资源创建和保护检查。

本文没有编译或运行 RDMA 程序，也没有宣称在特定 NIC、驱动、网络或 GPU 上观察到这些路径。接口结论依据 2026-09-12 核对的 rdma-core 上游手册和 NVIDIA 用户态 RDMA 编程手册；涉及 rdma-core v60.0 固定 commit 的 rping 调用路径将在 A04 单独核验。不同 provider 的扩展能力、缓存一致性细节和性能必须在记录硬件、驱动、固件、网络配置与源码版本后通过实验确认。

## 参考资料

- [rdma-core：libibverbs 概览](https://github.com/linux-rdma/rdma-core/blob/master/Documentation/libibverbs.md)
- [rdma-core：ibv_post_send(3)](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_post_send.3)
- [rdma-core：ibv_poll_cq(3)](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_poll_cq.3)
- [rdma-core：扩展 Work Request 提交接口](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_wr_post.3.md)
- [rdma-core：ibv_reg_mr(3)](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_reg_mr.3)
- [rdma-core：ibv_create_qp(3)](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_create_qp.3)
- [NVIDIA RDMA Aware Networks Programming User Manual 1.7](https://docs.nvidia.com/rdma-aware-networks-programming-user-manual-1-7.pdf)
