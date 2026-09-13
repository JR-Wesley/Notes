# A04 从 rping 源码追踪一轮完整 RDMA 通信

## 本课要解决的核心问题

[A01：从零开始理解 RDMA 通信模型](a01-rdma-communication-model.md)到[A03：RDMA 请求、完成与缓冲区协议](a03-rdma-requests-and-buffer-protocol.md)分别解释了资源、请求和完成，但这些概念只有放回真实程序，才能看清“建连、交换远端地址和 key、发起单边操作、等待本地完成、通知对端、复用缓冲区”如何首尾相接。本课以 rdma-core 的 `rping` 示例为唯一源码对象，沿客户端和服务端的实际 call path，跟踪一轮 ping/pong 中的 RDMA CM 事件、PD/MR/QP/CQ 创建、SEND/RECV 控制消息、RDMA READ/WRITE 数据传输、CQ 完成和线程唤醒。

本课不是逐行翻译源码，也不把示例程序当成生产库模板。重点是回答五个问题：远端虚拟地址和 `rkey` 从哪里来；为什么单边 READ/WRITE 之外仍需要 SEND/RECV；哪个完成允许哪一侧继续执行；控制状态怎样代替消息类型；两端缓冲区大小不一致时为什么可能越界或失败。阅读完后，读者应能把 A01—A03 的抽象对象逐一对应到真实字段和函数，并能指出 rping 为了教学简洁而省略的协议校验与并发设计。

## 固定源码基线与证据边界

本文核对的是 [linux-rdma/rdma-core 的 v60.0 标签](https://github.com/linux-rdma/rdma-core/releases/tag/v60.0)。GitHub tag object `89ddad4729501d5ad56815799aec012dab4f624d` 指向 commit `5321d809e095d6dd32a1355a5d4aa2ebbda4ba74`；实际读取文件为 [`librdmacm/examples/rping.c`](https://github.com/linux-rdma/rdma-core/blob/5321d809e095d6dd32a1355a5d4aa2ebbda4ba74/librdmacm/examples/rping.c)。读取副本共 1438 行，SHA-256 为 `4EE1C6F4B32A354C6111A6051CE1442861CE5A5EFDD7C30C3818B719C9D9528C`。下文出现的函数名、字段和行号范围均以该 commit 为准，不能无检查地套用到其他版本。

源码核验是静态阅读：本次没有编译 `rping`，没有运行客户端或服务端，也没有 RDMA 硬件结果。文中关于“源码做了什么”的结论来自固定版本代码；关于协议为何安全或可能失败的说明，则由 verbs 契约和代码中的长度、权限及状态依赖推导。凡涉及 provider 内部 WQE 格式、NIC 微架构、性能或特定硬件错误码，本文均不作实测结论。

## rping 的真正数据流：SEND 是控制面，READ/WRITE 是数据面

`rping` 的名字容易让初学者以为客户端 SEND 一个 ping、服务端再 SEND 一个 pong。固定版本源码顶部给出的流程并非如此：客户端先用 SEND 发送一段“源缓冲区公告”，其中包含地址、`rkey` 和长度；服务端收到公告后，用 RDMA READ 主动读取 ping payload。READ 在服务端本地完成后，服务端用 SEND 发一个继续令牌；客户端收到令牌，再 SEND 一段“目标缓冲区公告”；服务端随后用 RDMA WRITE 把刚才读到的数据写回客户端目标缓冲区，WRITE 本地完成后再 SEND 一个继续令牌。客户端收到第二个令牌后，才检查写回数据并进入下一轮。

```text
客户端                                                     服务端

start_buf 写入 ping
SEND {addr(start_buf), rkey(start_mr), size}  ────────────>  RECV 源公告
                                                           RDMA READ
start_buf <===============================================  rdma_buf
                                                           READ WC
RECV 继续令牌                               <──────────────  SEND 继续令牌
SEND {addr(rdma_buf), rkey(rdma_mr), size}   ────────────>  RECV 目标公告
                                                           RDMA WRITE
rdma_buf <===============================================  rdma_buf
                                                           WRITE WC
RECV 继续令牌                               <──────────────  SEND 继续令牌
校验 start_buf 与 rdma_buf，进入下一轮
```

图中的两条粗数据路径都由服务端发起：READ 把客户端 `start_buf` 拉到服务端 `rdma_buf`，WRITE 再把服务端 `rdma_buf` 推到客户端 `rdma_buf`。客户端没有直接 post RDMA READ/WRITE。两端的 SEND/RECV 只传递很小的控制信息或阶段令牌，这说明“one-sided”只描述具体 READ/WRITE 执行时远端 CPU 不必 post 匹配操作，并不意味着整个应用协议不需要连接管理、远端内存信息交换和阶段通知。

## 一个 control block 如何串起全部资源

`struct rping_cb` 是程序的连接上下文。它把 `rdma_cm_id`、PD、completion channel、CQ、QP、控制消息缓冲区、payload 缓冲区、预构造 WR、状态枚举、信号量和线程句柄放在同一对象中。理解这个结构时，不应把字段当作平铺清单，而应按依赖关系分组：CM ID 找到设备上下文并管理地址/连接事件；PD 约束 MR 与 QP 的保护域；CQ 和 completion channel 负责返回与唤醒；QP 的 SQ/RQ 接收预构造 WR；MR 为 SGE 提供 `lkey`，并在需要被对端访问时提供 `rkey`。

控制消息的线格式只有 16 字节：

```c
struct rping_rdma_info {
    __be64 buf;   /* 远端虚拟地址，网络字节序 */
    __be32 rkey;  /* 远端访问 key，网络字节序 */
    __be32 size;  /* 可访问/计划传输长度，网络字节序 */
};
```

`recv_buf` 与 `send_buf` 各有自己的 MR、SGE 和 RECV/SEND WR。`rdma_buf` 是两端都有的 payload 缓冲区；客户端另外有 `start_buf`，作为第一阶段 RDMA READ 的远端源。服务端没有 `start_buf`，因为服务端只需把 READ 结果保存在自己的 `rdma_buf`，再以同一块内存作为 WRITE 的本地源。`remote_addr`、`remote_rkey` 和 `remote_len` 则保存最近一次从客户端控制消息解析出的远端三元组。

这个设计有一个重要简化：所有 WR 的 `wr_id` 保持零值，CQ handler 只依靠 WC opcode 和共享 `state` 分派完成。它之所以勉强可行，是因为主循环严格串行，每个阶段只允许一种关键操作在途。若扩展为多槽、多请求或多连接共享 CQ，就必须用非零 `wr_id`、请求表、sequence/generation 等信息精确关联完成，不能继续依赖“当前状态应该对应当前 WC”。

## 从 main 到可通信 QP：控制线程与异步建连

`main()` 首先为 `rping_cb` 分配并清零内存，设置默认 payload 大小 64 字节、端口 7174 和 `IDLE` 状态，然后解析 `-s/-c`、地址、端口、次数、大小及验证选项。它创建一个 `eventfd` 作为关闭 CM 线程的本地信号，再调用 `create_event_channel()`、`rdma_create_id(..., RDMA_PS_TCP)` 建立 CM 事件通道和 CM ID，最后启动 `cm_thread()`。因此主线程不是直接同步取得“地址已解析”或“连接已建立”的返回值，而是提交 CM 操作，等待另一线程处理异步事件并通过信号量唤醒。

`cm_thread()` 同时 poll `eventfd` 与 `cm_channel->fd`。CM fd 可读时，它依次调用 `rdma_get_cm_event()`、`rping_cma_event_handler()` 和 `rdma_ack_cm_event()`。客户端收到 `RDMA_CM_EVENT_ADDR_RESOLVED` 后，handler 立即发起 `rdma_resolve_route()`；收到 `ROUTE_RESOLVED` 后才把 `state` 设为 `ROUTE_RESOLVED` 并 `sem_post()`。服务端收到 `CONNECT_REQUEST` 时记录新的 `child_cm_id`，再唤醒等待连接的主路径。连接建立或断开事件同样通过修改共享状态和 `sem_post()` 交给主线程。

客户端的实际启动路径可以压缩为：

```text
main
└─ rping_run_client
   ├─ rping_bind_client
   │  └─ rdma_resolve_addr
   │     └─ [CM thread] ADDR_RESOLVED → rdma_resolve_route
   │        └─ [CM thread] ROUTE_RESOLVED → sem_post
   ├─ rping_setup_qp
   ├─ rping_setup_buffers
   ├─ ibv_post_recv
   ├─ pthread_create(cq_thread)
   ├─ rping_connect_client → rdma_connect → 等待 CONNECTED
   └─ rping_test_client
```

服务端则先 `rdma_bind_addr()`、`rdma_listen()`，等待 CM thread 收到连接请求并设置 `child_cm_id`，然后为这个 child connection 创建 QP 和缓冲区，预贴 RECV、启动 CQ thread，再调用 `rdma_accept()`。普通服务端路径为 `main → rping_run_server → rping_bind_server → 等待 CONNECT_REQUEST → rping_setup_qp → rping_setup_buffers → ibv_post_recv → cq_thread → rping_accept → rping_test_server`。`-P` persistent 模式会克隆 control block，并为每个连接启动 detached worker；它改变连接并发管理，不改变单连接内部的 READ/WRITE 协议。

## PD、CQ、QP 和 MR 在源码中的创建顺序

`rping_setup_qp()` 先从 `cm_id->verbs` 分配 PD，再创建 completion channel 和容量为 `RPING_SQ_DEPTH * 2` 的 CQ，调用 `ibv_req_notify_cq()` 请求完成通知，最后创建 RC QP。QP 的 `max_send_wr` 为 16，`max_recv_wr` 为 2，发送和接收均只允许一个 SGE，并让 send CQ 与 recv CQ 指向同一个 CQ。这里的 CQ 容量 32 不等于 QP 队列深度；前者约束未消费完成的容量，后者约束相应工作队列可容纳的请求数量。

默认路径使用 `rdma_create_qp()`，让 RDMA CM 将 QP 与 CM ID 关联。`-q` 路径改用 `ibv_create_qp()` 并显式执行 INIT→RTR→RTS 的修改；`rping_init_conn_param()` 同时设置 `responder_resources = 1`、`initiator_depth = 1`、`retry_count = 7` 和 `rnr_retry_count = 7`。这些参数是示例的连接选择，不应脱离设备能力查询直接复制到任意程序。

`rping_setup_buffers()` 的注册结果揭示了每块内存承担的角色：

| 缓冲区 | 所在端 | 大小 | 注册权限 | 实际用途 |
| --- | --- | --- | --- | --- |
| `recv_buf` | 两端 | 16 字节 | `IBV_ACCESS_LOCAL_WRITE` | NIC 接收 SEND 控制消息，因而需要本地写入权限 |
| `send_buf` | 两端 | 16 字节 | `0` | 本地 SEND 源，NIC 读取它，不允许远端 RMA |
| `rdma_buf` | 两端 | `cb->size` | LOCAL_WRITE、REMOTE_READ、REMOTE_WRITE | 服务端为 READ 本地目标和 WRITE 本地源；客户端为 WRITE 远端目标 |
| `start_buf` | 仅客户端 | `cb->size` | LOCAL_WRITE、REMOTE_READ、REMOTE_WRITE | 第一阶段被服务端 READ 的远端源 |

权限集合比最小需要更宽。例如客户端 `start_buf` 的协议角色只要求远端 READ，服务端 `rdma_buf` 不需要被客户端做远端 RMA，但示例统一授予了 REMOTE_READ 和 REMOTE_WRITE。生产程序应按真实方向最小化权限，不能据此认为“payload MR 总要同时开启三种 access flag”。不过 `recv_buf` 的 LOCAL_WRITE、客户端 `rdma_buf` 的 REMOTE_WRITE，以及客户端 `start_buf` 的 REMOTE_READ，都能直接由数据方向推出。

`rping_setup_wr()` 将控制 RECV、控制 SEND 和数据 RMA 三个 WR 模板预先连到各自 SGE。SEND 与 RMA WR 都设置 `IBV_SEND_SIGNALED`。RMA opcode、远端地址、`rkey` 和长度在每个阶段临时填写；这再次说明 WR 是可复用的提交描述，而缓冲区和 MR 的生命周期必须覆盖所有引用它们的在途请求。

## 第一阶段：客户端公告源，服务端执行 RDMA READ

`rping_test_client()` 每轮先把 `state` 设为 `RDMA_READ_ADV`，在 `start_buf` 中生成以 `rdma-ping-N:` 开头的测试字符串，并把最后一个字节设为 `\0`。随后 `rping_format_send(start_buf, start_mr)` 把地址、`start_mr->rkey` 和客户端 `cb->size` 转成 big-endian，写入 16 字节 `send_buf`，再 post 一个 SEND。这个 SEND 不承载 ping payload，只公告“服务端可以从哪里读多少字节”。

服务端必须已经预贴 `rq_wr`。控制 SEND 到达后产生 `IBV_WC_RECV`；CQ handler 调用 `server_recv()`，先要求 `wc.byte_len == sizeof(recv_buf)`，再用 `be64toh()`/`be32toh()` 解析三元组。若当前状态不晚于 `CONNECTED`，或者上一轮处于 `RDMA_WRITE_COMPLETE`，它把状态改成 `RDMA_READ_ADV`。CQ handler 随即重新 post 同一个 RECV WR，再 `sem_post()` 唤醒服务端主循环。先补 RECV 再唤醒，有助于保证下一条控制 SEND 到达时 RQ 已有 credit。

服务端醒来后把 `rdma_sq_wr.opcode` 设为 `IBV_WR_RDMA_READ`，远端字段设为刚解析的地址和 `rkey`，本地 SGE 长度设为 `remote_len`。这个 WR 的本地 SGE 指向服务端 `rdma_buf`，因此数据方向是“客户端 `start_buf` → 服务端 `rdma_buf`”。`ibv_post_send()` 返回成功只说明 READ 请求已进入 SQ；服务端随后再次等待信号量，不能立即读取 `rdma_buf`。

READ 完成产生 `IBV_WC_RDMA_READ`。CQ handler 将状态置为 `RDMA_READ_COMPLETE` 并唤醒服务端；到这个边界，服务端才可将 `rdma_buf` 当作本轮读取结果。若启用 `-v`，服务端在此时打印字符串。随后服务端 post 一个普通 SEND 作为“可以继续”的令牌。客户端为这条 SEND 预贴的 RECV 完成后，`client_recv()` 看到客户端状态仍为 `RDMA_READ_ADV`，便将其推进到 `RDMA_WRITE_ADV` 并唤醒客户端主循环。

## 第二阶段：客户端公告目标，服务端执行 RDMA WRITE

客户端被第一枚继续令牌唤醒后，调用 `rping_format_send(rdma_buf, rdma_mr)`，公告客户端写回目标的地址、`rkey` 和大小，再 post 第二个控制 SEND。服务端 receive completion 进入 `server_recv()` 时，当前状态已是 `RDMA_READ_COMPLETE`，因此走另一分支，把状态置为 `RDMA_WRITE_ADV`。同一个 16 字节结构没有显式 `message_type` 字段；“这是源公告还是目标公告”完全由共享状态和消息到达顺序决定。

服务端随后把复用的 RMA WR opcode 改为 `IBV_WR_RDMA_WRITE`，远端地址和 `rkey` 取自第二次公告，本地 SGE 仍指向服务端 `rdma_buf`。值得注意的是，WRITE 长度不是公告中的 `remote_len`，而是 `strlen(rdma_buf) + 1`。在示例生成的以 NUL 结尾的字符串协议中，这等于实际字符串及终止符长度；但若把示例改成任意二进制 payload，`strlen()` 会在第一个零字节提前停止，甚至在缺少终止符时越界读取，因此不能把这段逻辑直接复用于二进制通信库。

WRITE 的本地完成产生 `IBV_WC_RDMA_WRITE`，CQ handler 将服务端状态置为 `RDMA_WRITE_COMPLETE` 并唤醒服务端主循环。此时服务端知道本地源缓冲区可以按该 WR 的完成语义复用，但客户端应用并不会仅凭单边 WRITE 自动醒来。因此服务端再 post 第二个普通 SEND；客户端收到它后，`client_recv()` 发现状态已不是 `RDMA_READ_ADV`，将状态改为 `RDMA_WRITE_COMPLETE`。客户端主线程由 receive completion 唤醒，才读取 `rdma_buf`，可选地用 `memcmp(start_buf, rdma_buf, cb->size)` 验证整个本地配置长度，然后开始下一轮。

这条链路展示了 A03 中“传输完成”和“业务可见通知”的区别。服务端 WRITE WC 是发起侧完成证据，第二个 SEND/RECV 才是客户端用于进入消费阶段的协议通知。若生产协议用 WRITE WITH IMM、单独的 doorbell 字段或 sequence flag 替代这枚 SEND，仍需明确等价的目标侧观察点和覆盖规则。

## 状态机不是网络协议字段，而是本地阶段解释器

源码定义了 `IDLE`、`CONNECT_REQUEST`、`ADDR_RESOLVED`、`ROUTE_RESOLVED`、`CONNECTED`、`RDMA_READ_ADV`、`RDMA_READ_COMPLETE`、`RDMA_WRITE_ADV`、`RDMA_WRITE_COMPLETE`、`DISCONNECTED` 和 `ERROR`。前半部分由 CM thread 推进，后半部分主要由 CQ thread 推进；主线程通过 `sem_wait()` 消费事件并检查期望状态。因而同一个 `sem` 同时承载 CM、RECV、READ completion、WRITE completion 和断连唤醒，状态值才是唤醒原因的判据。

每轮稳定的数据阶段如下：

| 顺序 | 事件处理者 | 状态结果 | 唤醒后执行者及动作 |
| --- | --- | --- | --- |
| 1 | 服务端 CQ thread 处理源公告 RECV | `RDMA_READ_ADV` | 服务端 post RDMA READ |
| 2 | 服务端 CQ thread 处理 READ WC | `RDMA_READ_COMPLETE` | 服务端读取结果并 SEND 令牌 |
| 3 | 客户端 CQ thread 处理第一枚令牌 RECV | `RDMA_WRITE_ADV` | 客户端 SEND 目标公告 |
| 4 | 服务端 CQ thread 处理目标公告 RECV | `RDMA_WRITE_ADV` | 服务端 post RDMA WRITE |
| 5 | 服务端 CQ thread 处理 WRITE WC | `RDMA_WRITE_COMPLETE` | 服务端 SEND 第二枚令牌 |
| 6 | 客户端 CQ thread 处理第二枚令牌 RECV | `RDMA_WRITE_COMPLETE` | 客户端校验并进入下一轮 |

控制 SEND 的成功 WC 只打印 debug 信息，不修改状态，也不 `sem_post()`。这是合理的，因为主循环等待的是 READ/WRITE 完成或对端控制消息，而不是控制 SEND 本地完成。但也意味着示例没有用 `wr_id` 追踪 send buffer 的精确复用水位；它依赖严格串行阶段和 RC QP 的队列顺序。若将两个控制 SEND 并发化或让 `send_buf` 在 SEND 完成前被改写，就需要显式处理 SEND WC 或使用满足大小限制的 inline SEND。

这种状态机很适合教学，却不是健壮的线上协议。控制包没有版本、消息类型、连接内 sequence、payload 校验和或声明容量；`server_recv()` 只校验 16 字节长度，客户端收到“继续令牌”时也只校验长度并根据当前状态解释。丢失同步、重复业务消息、错误状态中的合法长度包或未来协议扩展都难以诊断。生产设计至少应让 wire header 自描述，并验证 type、version、sequence、长度上限和当前允许的状态转换。

## CQ 通知线程：事件只是唤醒，WC 才是结果

`cq_thread()` 阻塞在 `ibv_get_cq_event()`。取得事件后，它确认返回的 CQ 正是 `cb->cq`，先调用 `ibv_req_notify_cq()` 重新 arm，再进入 `rping_cq_event_handler()` 循环 `ibv_poll_cq()`，最后用 `ibv_ack_cq_events()` 确认本次事件。这个顺序体现了 completion channel 只负责通知“CQ 可能有新内容”，真正的 opcode、status 和 `byte_len` 仍来自 WC。

handler 对每条 WC 先检查 `wc.status`。普通错误会把状态置为 `ERROR` 并唤醒主线程；`IBV_WC_WR_FLUSH_ERR` 被记为 flushed，用于让 CQ thread 退出。成功 WC 再按 opcode 分派：SEND 不唤醒，RDMA READ/WRITE 更新完成状态并唤醒，RECV 调用端侧 handler、补贴 receive credit，再唤醒。一个 CQ 同时承担 send 和 receive completion，所以分派依据是 WC opcode，而不是“这个 CQ 属于发送还是接收”。

源码的线程结构也给出了 forward progress 条件：CM thread 必须持续获取并 ack CM 事件，CQ thread 必须持续取得通知、poll WC 并 ack CQ 事件，主线程才能从信号量等待中前进。若把 CQ progress 和业务等待放到同一条执行流中，或者忘记重新 arm 通知，程序可能在请求已经完成时仍永久睡眠。

## 长度、容量与两端 `-S` 必须匹配

命令行 `-S` 设置本端 `cb->size`；默认值为 64，合法范围从 `RPING_MIN_BUFSIZE` 到 `RPING_BUFSIZE - 1`，其中 `RPING_BUFSIZE` 为 64 KiB。两端各自独立解析命令行，没有建连阶段的参数协商。客户端公告的 `size` 是客户端值，服务端却只按服务端值分配并注册自己的 `rdma_buf`。

第一阶段尤其危险：服务端把客户端公告的 `remote_len` 原样写入本地 RDMA READ SGE，而本地 `rdma_mr` 容量只有服务端 `cb->size`。如果客户端 `-S` 大于服务端 `-S`，请求描述的本地目标范围会超过服务端 MR，通常会导致本地长度或保护类完成错误，而不是自动截断。第二阶段中，服务端虽然收到客户端目标容量 `remote_len`，却没有在 WRITE 前验证 `strlen(rdma_buf) + 1 <= remote_len`。示例生成的数据通常让字符串长度与客户端大小一致，但协议本身并没有通用的容量证明。

因此运行示例时，应在两端显式给出相同的 `-S`，并把“配置相同”视为前置条件，不把它误写成源码已经协商并验证。若将 rping 改造成可靠协议，服务端至少应在 READ 前检查 `remote_len <= local_registered_capacity`，在 WRITE 前检查 `write_len <= advertised_remote_capacity`，并对整数转换、零长度、最大消息及二进制 payload 采用明确规则。

地址与 key 同样只有在公告对应 MR 仍有效时才有意义。客户端从发送公告到服务端相应 READ/WRITE 完成之前，不能注销 MR、释放缓冲区或改变其访问权限；连接退出时则要先停止新请求、让在途工作完成或 flush，再反向销毁 MR、QP/CQ/channel/PD 和 CM 对象。

## 运行方式与可观测点

源码 `usage()` 给出的基本模式是服务端 `rping -s`、客户端 `rping -c -a <server-address>`；`-C` 指定轮数，`-S` 指定 payload 大小，`-V` 验证数据，`-v` 打印数据，`-d` 输出 debug 信息。下面只是基于源码参数整理的示例命令，本次未编译、未运行；程序路径、设备可用性、路由和防火墙需要按实际环境确认。

```bash
# 服务端：显式指定与客户端相同的大小和轮数
rping -s -a 0.0.0.0 -p 7174 -S 4096 -C 10 -V -d

# 客户端：SERVER_IP 替换为服务端可达地址
rping -c -a SERVER_IP -p 7174 -S 4096 -C 10 -V -d
```

排错时应按阶段观察，而不是只看“连接失败”或“数据 mismatch”。地址解析之前检查 IP 与 RDMA device；`CONNECT_REQUEST/CONNECTED` 之前检查监听端口、route 和 CM 事件；第一条 SEND 卡住时检查双方是否已 post RECV；READ/WRITE WC 失败时核对本地 SGE 范围、远端地址、`rkey`、MR access flags 和 QP 状态；服务端 WRITE 已完成但客户端不前进时，检查第二枚 SEND 是否成功、客户端 RQ 是否有 credit、CQ thread 是否仍在 progress。若 `-S` 不同，应先纠正配置，而不是把保护错误归因于链路吞吐。

## 退出、错误路径与示例局限

正常有限轮数结束后，客户端调用 `rdma_disconnect()`，等待 CQ thread 退出，再释放 buffers 和 QP 资源。服务端的主循环若因断连被唤醒，会在 `state == DISCONNECTED` 时把退出视为正常。资源释放大体遵循先停止通信、再注销 MR 和释放内存、最后销毁 QP/CQ/completion channel/PD 与 CM ID 的反向顺序。错误路径使用 `goto` 分层回滚已创建对象，体现了 A02 所说的“只释放已经成功创建的层级”。

但 rping 的目标是展示 API 流程，而非覆盖所有健壮性要求。它没有为控制包设置自描述 header，不检查远端公告长度与本地 MR 容量的完整关系，不支持二进制 payload 的显式长度，不用 `wr_id` 关联多请求，也没有多槽 credit、超时取消、重连恢复或跨线程状态锁。共享 `state` 由多个线程读写，协调主要依赖信号量和严格事件次序；扩展代码时不能假定这种写法自然适用于高并发。

还需注意，rping 是 CPU 侧 librdmacm/libibverbs 示例。它没有 GPU 内存、CUDA stream、NVSHMEM symmetric heap、PE、transport 选择或 IBGDA doorbell。把它用于 NVSHMEM 学习的价值，是提供一条可见的基线：传统 host verbs 程序显式管理 CM、PD/MR/QP/CQ，显式交换地址和 `rkey`，由 CPU CQ thread 推进状态；NVSHMEM runtime 会隐藏或重组其中许多步骤，但内存授权、请求完成、目标通知、缓冲区 ownership 和 progress 依赖并不会消失。

## 从 rping 过渡到 NVSHMEM 时保留哪些问题

完成本课后再读 [B00：NVSHMEM 编程模型](b00-nvshmem-programming-model.md)，应把关注点从“NVSHMEM API 比 verbs 少”转为“哪些责任由 runtime 接管，哪些协议责任仍属于应用”。对称堆和 `<symmetric address, target PE>` 使应用不再手工 SEND `{addr, rkey, size}`；transport 也可能预建连接、注册内存并管理 QP/CQ。然而，一个 NBI put 的源何时可复用、远端何时可消费、signal 与 quiet 各证明什么、多个生产者如何避免覆盖同一槽位，仍对应 rping 中由 WC、继续令牌和状态机显式表达的问题。

进一步进入 GPUDirect RDMA 与 IBGDA 时，还要问“谁 post 请求、谁推进完成”。rping 的提交者是 CPU 主线程，progress 承担者是 CPU CQ thread；IBGDA 可能让 GPU 线程直接更新 NIC 工作队列或 doorbell，并使用不同的完成观察机制。实现路径发生变化，不代表可以省略依赖图。任何新路径都应重新标出远端地址/key 的建立方式、请求队列、完成边界、通知对象、缓冲区容量和退出时的在途请求处理。

## 本课结论与自测

rping 的核心不是某一个 verbs API，而是一条闭环协议：RDMA CM 负责建连，PD/MR/QP/CQ 建立可执行资源，SEND/RECV 交换远端内存能力并传递阶段令牌，服务端用 RDMA READ 拉取 ping、用 RDMA WRITE 写回 pong，CQ thread 把硬件完成转换为本地状态和信号量唤醒。READ/WRITE 的 one-sided 数据移动与 SEND/RECV 的双边控制面同时存在，两者共同解决“能访问哪里”和“何时可以继续”这两个不同问题。

读者应能独立回答以下问题：为什么服务端在收到第一条 16 字节消息后能读取客户端内存；为什么 READ WC 到达前不能打印 `rdma_buf`；为什么 WRITE WC 不足以唤醒客户端应用；同一种 `rping_rdma_info` 如何被解释成源公告和目标公告；为什么客户端 `-S` 大于服务端 `-S` 会威胁本地 MR 边界；为什么每次 receive completion 后必须重新 post RECV；以及把单请求示例改为多请求流水线时，为什么必须引入 `wr_id`、sequence、容量校验和多槽 ownership。若这些问题仍需依靠记忆答案，应回到 A02 的资源生命周期和 A03 的完成/缓冲区协议重新画一遍两端时序。

## 主要来源

- [rdma-core v60.0 固定版本 `rping.c`](https://github.com/linux-rdma/rdma-core/blob/5321d809e095d6dd32a1355a5d4aa2ebbda4ba74/librdmacm/examples/rping.c)：本文的 symbol、call path、状态转换、长度和 MR 权限依据。
- [rdma-core v60.0 release](https://github.com/linux-rdma/rdma-core/releases/tag/v60.0)：版本标签入口；本文另通过 GitHub tag API 核实 tag object 与目标 commit。
- [rdma-core `ibv_post_send(3)`](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_post_send.3)、[`ibv_post_recv(3)`](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_post_recv.3)、[`ibv_poll_cq(3)`](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_poll_cq.3)：用于与 A03 的请求、receive credit 和完成契约交叉核对；这些手册链接指向当前上游，不作为 v60.0 源码行级证据。
