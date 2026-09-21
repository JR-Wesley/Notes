# A02 RDMA 资源、内存注册与建连生命周期

## 本课解决的问题

[A01：从零开始理解 RDMA 通信模型](a01-rdma-communication-model.md)已经说明，RDMA 请求不是拿到一个远端指针后直接访问，而是依赖设备、保护、内存、队列和完成资源共同工作。本课进一步回答工程实现中的关键问题：程序怎样选择一块可用的 RDMA 设备和端口；为什么 QP 与 MR 必须建立在相容的保护关系中；`lkey`、`rkey`、地址和长度分别验证什么；RC QP 为什么要经历 INIT、RTR 和 RTS；双方在控制面必须交换哪些信息；初始化中途失败或进程退出时，资源应按什么顺序回收。

本文以两个 Linux 进程建立一条 RC 连接并执行 RDMA WRITE 为贯穿场景。目标不是提供可直接复制的完整程序，而是建立能够审查真实代码的资源生命周期模型。示意代码省略具体地址结构、错误日志和 provider 差异，未编译、未运行；接口事实依据 2026-09-12 核对的 rdma-core 上游手册和 NVIDIA RDMA Aware Networks Programming User Manual 1.7。A03 将在这些资源已经就绪的前提下继续追踪 WR、SGE、WQE、CQE/WC 和缓冲区协议。

## 先看全局：本地资源与远端资源不会自动对应

主机 A 和主机 B 都要独立打开本地 RDMA 设备、创建本地 PD、注册本地 MR、创建本地 CQ 和 QP。A 的 `ibv_context *`、`ibv_pd *`、`ibv_mr *` 等指针只在 A 的进程中有效，不能发送给 B 后直接使用；B 需要的是协议规定的网络标识和远端访问元数据，例如 QP number、路径地址、Packet Sequence Number（PSN），以及执行 RMA 时所需的远端 I/O 地址、`rkey` 和可访问长度。

资源可以按三条关系理解。设备关系决定对象由哪块 RNIC 承载；保护关系要求本地 QP 使用与其 PD 相容的 MR key；通信关系把本地 QP 配置为能够向指定远端 QP 和路径发送。CQ 不属于 PD，而属于 device context，并在创建 QP 时与其 SQ/RQ 关联。这个差异很重要：PD 管的是访问保护，CQ 管的是完成报告，二者不能相互替代。

```text
主机 A                                                        主机 B
context A                                                     context B
 ├─ PD A                                                       ├─ PD B
 │   ├─ MR A: addr/len/lkey/rkey                               │   ├─ MR B: addr/len/lkey/rkey
 │   └─ QP A: SQ/RQ ═════ 路径与对端 QPN ══════════════════════ │   └─ QP B: SQ/RQ
 └─ CQ A ← QP A 的完成                                         └─ CQ B ← QP B 的完成

控制面交换：端点地址、QPN、PSN、协议版本，以及需要暴露的 addr/rkey/length
数据面使用：本地 SGE(addr/length/lkey) + 远端目标(addr/rkey)
```

## 选择设备和端口：context 只是起点

程序首先通过 `ibv_get_device_list()` 获得可见的 RDMA 设备，再用 `ibv_open_device()` 创建 `ibv_context`。设备名称只表明系统发现了 provider 对应的设备，并不能证明某个端口已经可用。应用还应通过 `ibv_query_device()` 检查资源上限和能力，通过 `ibv_query_port()` 检查端口状态、active MTU、link layer 和地址表规模，并根据部署选择端口与 GID index。InfiniBand 与 RoCE 对路径地址的使用不同，尤其不能在多端口、多 GID 或容器环境中无条件使用索引 0。

[`ibv_query_device(3)`](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_query_device.3)给出的 `max_qp_wr`、`max_sge`、`max_cqe`、`max_mr_size` 等是设备上限，不是应用一定能够获得的配额。实际可创建数量还受主机内存、进程权限、驱动配置以及其他进程已占用资源影响。工程上应先查询能力，再以工作负载需要申请资源，并检查 API 返回的实际能力；不能把某台机器上的成功参数固化为所有节点都支持的常量。

选定 context 与端口后，程序才具备创建资源的设备入口。此时网络连接仍未建立，远端也不知道本进程存在。`ibv_context` 的生命周期必须覆盖所有由它派生的 PD、MR、CQ、QP 和事件资源；[`ibv_open_device(3)`](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_open_device.3)明确指出，关闭 context 不会替应用释放全部关联资源，因此正常退出和失败回滚都必须显式清理。

## Protection Domain：把资源组合限制在保护边界内

应用调用 `ibv_alloc_pd(context)` 创建 Protection Domain。PD 不对应网络连接，也不是一块内存；它是 RNIC 用来约束资源组合的保护域。MR 注册在某个 PD 中，QP 也在某个 PD 中创建。当 QP 的 WQE 引用本地 SGE 时，NIC 会依据 SGE 中的 `lkey` 验证这段内存是否属于可接受的保护关系；远端 RMA 请求到达目标 QP 后，目标 RNIC 也会结合目标地址、`rkey`、MR 权限和 QP 权限进行检查。

因此，PD 解决的是设备内部的资源隔离，而不是完整的网络安全。`rkey` 也不是密码学密钥：它与地址范围、QP 可达性和 MR 生命周期共同构成访问能力，但不提供加密、身份认证或防窃听。应用仍需依赖受控网络、连接认证和安全的控制面传递敏感元数据。权限设计应采用最小授权，只给确实需要被远端 READ、WRITE 或 Atomic 的 MR 设置相应标志，而不是为了省事对所有缓冲区开放全部远端访问。

[`ibv_alloc_pd(3)`](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_alloc_pd.3)说明，仍有其他资源与 PD 关联时，`ibv_dealloc_pd()` 可能失败。这意味着 PD 通常早于 MR/QP 创建、晚于 MR/QP 销毁。把 PD 当成普通无依赖句柄并提前释放，会破坏后续注销 MR 或销毁 QP 的前提。

## Memory Region：范围、映射和权限必须同时成立

### 注册对象与底层缓冲区是两个生命周期

假设进程 B 用 `malloc()` 得到一段长度为 $L$ 的缓冲区，基地址为 $B$。在普通 `ibv_reg_mr()` 模式下，应用把地址、长度、PD 和 access flags 交给 provider，得到 `ibv_mr`。MR 记录注册范围并提供 `lkey` 与 `rkey`，但它不拥有由 `malloc()` 创建的缓冲区。应用既要在适当时机调用 `ibv_dereg_mr()` 注销注册对象，也要单独释放底层缓冲区；正确顺序是先确保没有请求再引用该 MR，注销 MR，最后释放缓冲区。

一次访问合法的必要范围条件可以写为：

$$
[A, A + N) \subseteq [B, B + L)
$$

其中 $A$ 是请求使用的起始地址，$N$ 是传输字节数。仅有 $A \ge B$ 不够，还要防止 `A + N` 越过注册末端以及整数溢出。真实协议应交换或保存基址与长度，计算对象偏移时验证 `offset <= L` 且 `N <= L - offset`，而不是先计算可能溢出的 `offset + N` 再比较。

### lkey 和 rkey 分别服务于哪一侧

`lkey` 由本地 SGE 携带。以 A 向 B 执行 RDMA WRITE 为例，A 的源缓冲区 SGE 使用 A 本地 MR 的 `lkey`，使 A 的 RNIC 能读取源数据；WRITE WR 中的 `remote_addr` 与 `rkey` 则来自 B 暴露的 MR，使 B 的 RNIC 能验证目标写入。B 的 `lkey` 不需要交给 A，A 本地 MR 的 `rkey` 若不供远端访问也无需发给 B。把两种 key 都广播给所有节点既没有必要，也扩大了错误与越权范围。

[`ibv_reg_mr(3)`](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_reg_mr.3)给出的稳定接口规则是：本地 SGE 使用 `lkey`，远端 RDMA/Atomic 请求使用 `rkey`。在普通 `ibv_reg_mr(pd, addr, length, access)` 中，远端使用的 I/O 地址通常与注册时虚拟地址对应；`ibv_reg_mr_iova()`、zero-based MR 和 DMA-BUF MR 可以改变地址解释方式。因此，协议字段更准确的名称是 remote I/O virtual address，而不是假设它永远等于对端某个可直接解引用的 C 指针。

### access flags 是能力声明，不是性能选项

| 标志 | 允许的核心行为 | 约束与常见用途 |
| --- | --- | --- |
| `IBV_ACCESS_LOCAL_WRITE` | 本地 NIC 可以把数据写入该 MR | 接收缓冲区、RDMA READ 本地结果等需要 NIC 写入的区域使用 |
| `IBV_ACCESS_REMOTE_WRITE` | 远端请求可以 RDMA WRITE 到该 MR | 目标端显式暴露可写窗口；同时必须设置 local write |
| `IBV_ACCESS_REMOTE_READ` | 远端请求可以从该 MR 执行 RDMA READ | 只开放确实需要被拉取的只读数据区 |
| `IBV_ACCESS_REMOTE_ATOMIC` | 远端可以执行设备支持的原子操作 | 还受设备原子能力、大小和对齐等条件约束；同时必须设置 local write |
| `IBV_ACCESS_ON_DEMAND` | 使用 On-Demand Paging 注册方式 | 必须先查询设备 ODP 能力，缺页与性能特征不能按经典注册推断 |
| `IBV_ACCESS_RELAXED_ORDERING` | 允许特定网络写入以更宽松的顺序到达内存 | 可能改善性能，但会改变 write-after-write 数据顺序假设，必须依据手册设计 |

本地读权限在传统 MR 语义中隐含启用；若设置 remote write 或 remote atomic，rdma-core 手册要求同时设置 local write。MR 的 access flags 还不是唯一门槛：QP 在 INIT 等状态配置中也有 `qp_access_flags`，远端操作必须同时满足 MR 与 QP 两层授权。错误地只检查 MR 标志，可能得到“明明注册了 remote write 却仍然 remote access error”的表象。

### 注册不是无代价操作，也不等同于固定锁页实现

经典 MR 注册通常涉及页映射、固定和 RNIC 地址转换资源，频繁注册小缓冲区可能使控制路径成本成为瓶颈，因此通信库常复用注册区或维护 registration cache。不过“注册必然把所有页永久 pin 住”不是跨模式成立的定义：ODP 延迟建立页映射，DMA-BUF 注册又用于不同内存共享路径。本文只给出 API 与生命周期保证，不对某个 provider 的注册时延、页表结构或缓存策略作无证据推断。

内存注销是权限撤销与资源回收的一部分，但应用不能把 `ibv_dereg_mr()` 当作取消在途请求的同步原语。安全做法是先停止发布引用该 MR 的新 WR，等待相关工作达到可回收完成边界，再撤销或注销 key，最后释放底层内存。若需要频繁改变远端暴露范围，可研究 Memory Window 等机制；这属于后续扩展，不应通过不断注销仍在使用的 MR 模拟。

## CQ 与 QP 容量：资源上限必须来自并发模型

创建 CQ 时指定的 `cqe` 是希望容纳的完成项数量，provider 可能返回不小于请求值的实际容量。CQ 大小必须覆盖可能同时积累的 signaled send completion、receive completion 以及共享此 CQ 的所有 QP；应用还必须以足够速度轮询。[`ibv_poll_cq(3)`](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_poll_cq.3)明确指出，CQ overrun 会触发 `IBV_EVENT_CQ_ERR`，此后该 CQ 不能继续使用。使用 completion channel 只改变等待通知的方式，真正的 WC 仍需从 CQ 中轮询取出并检查。

创建 QP 时，`ibv_qp_init_attr.cap` 请求 `max_send_wr`、`max_recv_wr`、`max_send_sge`、`max_recv_sge` 和 `max_inline_data`。这些值分别约束 SQ/RQ 能容纳的 WR 数、每个 WR 的分散聚集段数和可内联数据量。[`ibv_create_qp(3)`](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_create_qp.3)规定 provider 会把 `cap` 更新为实际创建值，因此代码应读取返回值，不应假定实际值与请求值完全相同。

队列容量应从在途并发推导。例如，应用希望每个 QP 最多保留 $W$ 个发送请求，其中每隔 $K$ 个请求产生一个 signaled completion，就仍需保证 SQ 能承载所有尚未退休的请求，而不能只按 CQE 数量申请 SQ。RQ 则由可能到达但尚未补充的 SEND/WRITE WITH IMM 数量决定。A03 会进一步说明 unsignaled 请求怎样随之后的 signaled 完成一起回收；本课只保留结论：减少 CQE 不等于减少在途 WR，也不等于可以忽略 SQ/RQ credit。

## RC QP 建连：状态转换是在逐步授予能力

新创建的 QP 从 RESET 开始。对 RC QP，应用通过 `ibv_modify_qp()` 依次推进 INIT、RTR 和 RTS；每次转换要求一组属性完整且相容。如果属性或 mask 无效，[`ibv_modify_qp(3)`](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_modify_qp.3)规定这次修改不会只完成一部分。状态转换不是形式步骤，而是在逐层声明本地端口与权限、接收路径和发送策略。

| 转换 | 主要信息 | 建立的能力 |
| --- | --- | --- |
| RESET → INIT | port、P_Key index、QP access flags | QP 绑定本地端口和分区，并声明允许响应的远端操作 |
| INIT → RTR | path MTU、对端 QPN、接收 PSN、Address Vector；RC 还包括 responder resources 与 RNR 相关参数 | QP 知道从哪个路径接收哪个对端的报文，并准备响应 READ/Atomic 等请求 |
| RTR → RTS | 发送 PSN、timeout、retry count、RNR retry、initiator RDMA read/atomic 深度 | SQ 可以主动发送，可靠连接获得超时与重试策略 |

QP number（QPN）标识设备上的 QP，PSN 用于传输层的包序列跟踪；它们不是应用消息序号。应用即使正确配置了 PSN，仍需为业务轮次设计自己的 sequence number。`path_mtu` 也不是任意选择的消息长度，而是路径分包能力的一部分，必须与双方端口和路径相容；应用 payload 可以大于 MTU，由传输层分包完成。

`max_rd_atomic` 限制本端可发起的在途 RDMA READ/Atomic 数量，`max_dest_rd_atomic` 表示本端为对端此类操作提供的 responder resources。双方配置必须能够互相满足，不能只在发起侧把深度设大。SEND 和 WRITE 的队列深度、READ/Atomic responder 深度、RQ credit 是不同资源维度，故障定位时应分别检查。

## 控制面交换：连接参数与内存授权应版本化

手工 verbs 程序通常先借助 TCP socket 或其他 bootstrap 通道交换建连信息，再调用 `ibv_modify_qp()`。对于 RC，至少要交换对端 QPN、PSN 和路径地址所需字段；InfiniBand/RoCE 所需的 LID、GID、GID index 或其他 Address Vector 信息取决于实际 link layer 与路由。若随后执行 one-sided RMA，还要交换应用明确暴露的 remote address、`rkey` 和 length。SEND/RECV 不需要为 payload 放置交换 remote address/rkey，但仍需端点和路径建连信息。

控制消息应有显式协议，而不是直接发送本机 C struct 的原始字节。结构体可能包含 padding，整数存在端序差异，`uintptr_t` 宽度和 ABI 也可能不同。一个可审查的教学结构如下；它只表示需要序列化的逻辑字段，不是已经运行的 wire format。

```c
/* 示意：必须定义真实编码、端序和校验；本片段未编译、未运行。 */
struct endpoint_record {
    uint32_t protocol_version;
    uint32_t qp_num;
    uint32_t psn;
    uint8_t  gid[16];
    uint64_t remote_iova;
    uint64_t exposed_length;
    uint32_t rkey;
    uint32_t access_capabilities;
};
```

`protocol_version` 用于拒绝不兼容布局，`exposed_length` 用于请求前做范围校验，`access_capabilities` 说明这段 MR 是可读、可写还是支持原子操作。传输这个记录时应逐字段规定网络字节序或采用明确的序列化格式，并验证对端返回的长度、版本和能力。远端地址在本地不能被解引用，它只是提交 RDMA WR 时交给 RNIC 的 I/O 地址。

## RDMA CM：自动化地址解析与连接事件，不替代业务协议

RDMA CM（librdmacm）可以基于 socket address 选择本地设备、解析 RDMA 地址和路由，并驱动连接建立。主动侧的典型事件序列是创建 event channel 和 `rdma_cm_id`，调用 `rdma_resolve_addr()`，等待并确认 `ADDR_RESOLVED`，再调用 `rdma_resolve_route()`，等待 `ROUTE_RESOLVED`，创建 QP 后执行 `rdma_connect()`，最终等待 `ESTABLISHED`。被动侧绑定地址并 `rdma_listen()`，收到 `CONNECT_REQUEST` 时得到一个与该连接对应的新 `rdma_cm_id`，为它创建资源并调用 `rdma_accept()`。

[`rdma_cm(7)`](https://github.com/linux-rdma/rdma-core/blob/master/librdmacm/man/rdma_cm.7)给出了上述异步事件流程。每个通过 `rdma_get_cm_event()` 取得的事件都必须用 `rdma_ack_cm_event()` 确认，而且事件内容只能在 ack 前安全使用。应用还要处理 `ADDR_ERROR`、`ROUTE_ERROR`、`REJECTED`、`UNREACHABLE` 和 `DISCONNECTED` 等分支，不能只实现成功路径后无限等待。

RDMA CM 可以通过 connection private data 携带少量应用信息，但这不自动形成通用 MR 交换协议。应用仍需定义版本、长度、端序、授权范围和重连后的 key 更新。RDMA CM 帮助解决“怎样找到并连接远端 QP”，不会决定“远端暴露哪一段对象”或“何时允许覆盖业务缓冲区”。

## 把初始化、传输与退出连成一个状态机

下面的伪代码展示依赖顺序。它没有填充 RC 的全部 QP 属性，也没有实现 CM/TCP 控制通道，因此不是可编译示例；用途是审查成功路径与错误回滚是否覆盖每个已创建资源。

```c
/* 伪代码：未编译、未运行，不能直接作为示例程序。 */
device_list = ibv_get_device_list(&num_devices);
ctx = ibv_open_device(select_device(device_list));
query_and_select_active_port(ctx, &port, &gid_index);

pd = ibv_alloc_pd(ctx);
buffer = allocate_buffer(buffer_size);
mr = ibv_reg_mr(pd, buffer, buffer_size, required_access_flags);
cq = ibv_create_cq(ctx, planned_cqe, NULL, completion_channel, 0);
qp = ibv_create_qp(pd, &qp_init_attr_using_cq);

local_record = build_endpoint_record(qp, port, gid_index, mr);
remote_record = exchange_and_validate(local_record);
transition_rc_qp_to_init(qp, local_policy);
transition_rc_qp_to_rtr(qp, remote_record);
transition_rc_qp_to_rts(qp, local_policy);

post_required_receive_credits(qp);      /* SEND/WRITE_WITH_IMM 才需要 */
run_data_protocol(qp, mr, remote_record);
drain_all_relevant_work(qp, cq);

disconnect_or_stop_peer();
ibv_destroy_qp(qp);
ibv_destroy_cq(cq);
ibv_dereg_mr(mr);
free(buffer);
ibv_dealloc_pd(pd);
ibv_close_device(ctx);
ibv_free_device_list(device_list);
```

真实代码不能简单跳到统一 cleanup 并无条件调用所有销毁函数。每个句柄应初始化为 NULL，并只清理已成功创建的资源；若 QP 已有在途工作，先定义取消、转入 error 或排空策略；若 completion channel 仍有关联 CQ，应先销毁 CQ；若 PD 仍关联 MR/QP，dealloc 可能失败。清理函数的返回值也应记录，因为 teardown 错误可能揭示隐藏的资源引用或设备异常。

在双方正常退出时，还需要业务层关闭握手。若 A 在发送最后一次 WRITE 后立即销毁 QP，而 B 尚未读取数据或返回 ack，传输完成和应用完成仍可能错位。通常先停止接收新业务，等待已发布数据被消费，交换 close/ack，排空相关 WR/WC，再解除连接和释放资源。进程崩溃、链路故障和对端失联不能依赖优雅握手，应由超时、CM disconnect/error 事件和上层恢复策略处理。

## 贯穿例子：为一次 RDMA WRITE 准备什么

假设 A 要把 4 KiB payload 写到 B 的一个固定槽位。A 注册源缓冲区时至少需要满足 NIC 本地读取；B 注册目标缓冲区时需要 `IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE`，并在 QP INIT 权限中允许 remote write。双方建立 RC QP 后，B 通过控制面只向 A 暴露该槽位所在 MR 的 remote IOVA、`rkey`、长度和协议版本。A 验证 `4096 <= exposed_length`，用本地源地址、4096 和本地 `lkey` 构造 SGE，再用 B 的 remote IOVA 与 `rkey` 构造 WRITE WR。

A 成功 post WR 后仍不能修改非 inline 源缓冲区，直到获得满足复用条件的本地完成。普通 WRITE 不要求 B 预贴 Receive Request，也不会自动给 B 一个接收 WC，因此还需要 ready/signal 协议通知 B。B 完成消费后再返回 ack，A 才能认为远端槽位可以覆盖。这个例子说明资源正确只保证请求有可能合法执行；数据发布和槽位所有权仍由 A03/B04 所讨论的应用协议保证。

如果 B 把 MR 注册为 remote read 而没有 remote write，或者 QP 未允许 remote write，A 使用正确地址和 `rkey` 仍会得到远端访问错误。如果 A 把 4 KiB 写到只剩 2 KiB 的 MR 尾部，key 正确也不能使越界合法。如果 B 注销 MR 后重建注册，旧 `rkey` 和旧地址记录必须视为失效，重连或重新发布授权，不能继续使用缓存的 endpoint record。

## 分层诊断资源与建连故障

设备阶段无法枚举设备时，检查内核 RDMA 支持、provider、设备权限和容器映射；能够打开设备但端口不可用时，检查 port state、link layer、GID 表和实际网络配置。资源创建失败时，对照设备能力、申请数量、内存锁定限制、PD/context 归属和系统剩余资源，而不是只重复调用。

QP 无法进入 RTR 时，重点核对对端 QPN、路径地址、GID/LID、path MTU、接收 PSN 和 responder resources；无法进入 RTS 时，检查发送 PSN、timeout、retry、RNR retry 与 initiator 深度。连接已建立但 WR 返回保护错误时，再转向 MR 范围、access flags、`lkey/rkey`、QP access flags 和授权版本。连接成功却一直收不到 SEND/WRITE WITH IMM 时，还要检查 RQ credit；普通 WRITE 没有远端 CQE 则是语义预期，不应误判为 CQ 故障。

超时尤其需要区分网络不可达、RC 重试耗尽、RNR credit 不足和应用协议等待错误。只看到“timeout”不能推出远端机器宕机；应同时保存 CM event、WC status、vendor error、QP 状态、端口计数器和双方协议日志。本文没有实际环境，因此不列出某个设备专用错误码到故障原因的固定映射。

## 与 NVSHMEM 运行时的关系

NVSHMEM 应用通常不直接创建 verbs PD、MR 和 RC QP。runtime 在初始化和对称堆建立过程中管理 transport 资源、内存注册、key 交换和远端地址映射，并可能根据拓扑、配置和消息路径选择不同机制。但抽象被隐藏不等于资源不再存在：对称对象必须处于 runtime 能定位和访问的内存范围，底层队列仍有容量，远端访问仍需要正确授权，退出仍需在资源销毁前处理未完成通信。

理解 A02 后再阅读 [B00：NVSHMEM 编程模型](b00-nvshmem-programming-model.md)，应能区分“NVSHMEM API 不要求用户传 `rkey`”和“底层 transport 不需要 key 管理”。前者是 API 抽象，后者通常是不成立的实现推断。后续 IBGDA 课程会把队列提交者从 CPU proxy 路径进一步推进到 GPU，但 PD/MR、地址/key、QP 状态、队列深度和完成回收仍是实现必须维护的约束。

## 诊断题与掌握标准

1. **为什么两端不能交换 `ibv_mr *` 后直接执行 RDMA？** `ibv_mr *` 是本进程用户态对象指针；对端需要的是协议编码的 remote IOVA、`rkey`、长度和访问能力，而不是本地指针值。
2. **MR 设置了 `IBV_ACCESS_REMOTE_WRITE`，为什么 WRITE 仍可能失败？** 还要满足 local write 组合要求、QP access flags、地址范围、key 生命周期、PD/QP 保护关系、QP 状态和网络可达性。
3. **为什么 CQ 容量不能简单等于 SQ 容量？** 是否产生 CQE取决于 signaled 策略、接收完成和共享 CQ 的 QP；SQ 中的在途 WR 数与 CQ 中积累的完成数是相关但不同的量。
4. **INIT、RTR、RTS 分别解决什么问题？** INIT 绑定本地端口和权限，RTR 配置接收路径及对端身份，RTS 再配置主动发送的序列、超时、重试和发起深度。
5. **RDMA CM 是否消除了 endpoint record？** 它可以完成地址/路由解析和连接管理，也能携带有限 private data，但应用仍要定义 MR 授权、版本、长度、端序和业务协议。
6. **为什么销毁通常按创建依赖的反序进行？** QP、MR、CQ 等对象仍引用其父资源或彼此关联；先释放被依赖对象会导致销毁失败、资源泄漏或未定义的请求生命周期。

达到本课目标时，读者应能为一个两端 RC 程序画出资源图和初始化/销毁状态机，解释每个 key、地址和队列容量来自哪里，并针对 WRITE 场景列出 A 与 B 各自创建、交换和保留的资源。还应能从错误发生阶段区分设备发现、端口路径、资源配额、QP 状态、MR 权限和业务协议问题，而不是把所有失败统称为“RDMA 没连上”。

## 证据边界与后续入口

本课没有编译或运行示意代码，没有测量注册、建连或销毁开销，也没有验证特定 NIC、驱动、固件、RoCE 网络或容器环境。rdma-core 引用指向 2026-09-12 核对的上游 `master` 手册，适合说明当前接口，但不是固定 commit 的实现证据。Relaxed Ordering、ODP、Memory Window、SRQ 和 provider direct verbs 只说明边界，没有展开为配置指南。

下一篇 A03 将从“资源已经建立”开始，详细区分 WR、SGE、WQE、CQE 与 WC，解释 post、doorbell、完成、signaled/unsignaled、接收补充和 payload/ready/ack 协议。A04 再以固定的 rdma-core v60.0、commit `5321d809e095d6dd32a1355a5d4aa2ebbda4ba74` 核对 rping 真实源码路径。

## 参考资料

- [rdma-core：ibv_get_device_list(3)](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_get_device_list.3.md)
- [rdma-core：ibv_open_device(3)](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_open_device.3)
- [rdma-core：ibv_query_device(3)](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_query_device.3)
- [rdma-core：ibv_alloc_pd(3)](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_alloc_pd.3)
- [rdma-core：ibv_reg_mr(3)](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_reg_mr.3)
- [rdma-core：ibv_create_cq(3)](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_create_cq.3)
- [rdma-core：ibv_poll_cq(3)](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_poll_cq.3)
- [rdma-core：ibv_create_qp(3)](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_create_qp.3)
- [rdma-core：ibv_modify_qp(3)](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_modify_qp.3)
- [rdma-core：rdma_cm(7)](https://github.com/linux-rdma/rdma-core/blob/master/librdmacm/man/rdma_cm.7)
- [NVIDIA RDMA Aware Networks Programming User Manual 1.7](https://docs.nvidia.com/rdma-aware-networks-programming-user-manual-1-7.pdf)
