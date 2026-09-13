# A03 RDMA 请求、完成与缓冲区协议

## 本课要解决的核心问题

[A02：RDMA 资源、内存注册与建连生命周期](a02-rdma-resources-and-memory-registration.md)已经把 QP、MR 和 CQ 建立起来，但“资源存在”只说明程序具备提交请求的条件，并不说明一次传输何时真正结束。本课沿一次 A 向 B 发送 payload 的过程，区分应用构造的 Work Request（WR）、描述本地内存的 Scatter/Gather Element（SGE）、provider 或硬件队列中的 Work Queue Element（WQE）、完成队列里的 Completion Queue Element（CQE）以及 `ibv_poll_cq()` 返回的 Work Completion（WC）。这些概念分别属于不同层，混在一起会直接导致缓冲区提前复用、CQ 误判、RQ 枯竭或远端数据被覆盖。

本课进一步把请求完成与应用协议连接起来。发送侧需要知道源缓冲区何时不再被 NIC 读取，接收侧需要知道哪块缓冲区装入了哪一轮消息，RDMA WRITE 的目标侧需要独立的 ready 通知，生产者还必须等消费者返回 ack 才能覆盖远端槽位。文中的 C 片段用于解释数据结构和状态，未编译、未运行；WQE、doorbell 和 CQE 的描述是通用教学模型，不代表某个 provider 的固定内存布局。

## 一次请求跨越哪些表示层

应用首先创建 WR。发送 WR 包含用户定义的 `wr_id`、opcode、send flags、SGE 列表以及某些 opcode 所需的远端地址、`rkey` 或 immediate data；接收 WR 则主要包含 `wr_id` 和接收 SGE 列表。SGE 的三元组是本地地址、长度和 `lkey`，多个 SGE 在逻辑上拼接为一次传输的数据区域。SGE 只描述本地内存：RDMA WRITE 的远端目标不放在 SGE 中，而放在发送 WR 的 RMA 字段里。

调用 `ibv_post_send()` 或 `ibv_post_recv()` 时，应用把一个 WR 或由 `next` 串接的 WR 链表交给 provider。provider 校验能够立即发现的错误，并把已接受请求翻译为 QP 工作队列能够执行的表示。硬件文档和源码中常把队列条目称为 WQE；应用随后通过 doorbell 或等价机制让 NIC 知道有新工作。具体 WQE 格式、队列是否完全映射到用户态、doorbell 是 MMIO 写还是其他机制取决于 provider，因此 API 层只能稳定地说“WR 已被 post 到工作队列”，不能仅凭通用术语推断设备字节布局。

NIC 执行 WQE 后，把需要报告的结果写入 CQ。硬件记录常称 CQE，libibverbs 轮询接口把它解析为 `struct ibv_wc`。WC 通过 `wr_id` 关联应用请求，并给出 status、opcode、byte length、QP number、immediate data 标志等有效字段。CQE 与 WC 常被口语化地互换，但源码分析中应保持区别：CQE 是 provider/设备完成格式，WC 是 libibverbs 对应用提供的完成结果。

```text
应用描述                 provider / QP 快路径                 完成观察

WR ──引用──> SGE[]  ──post──>  WQE 放入 SQ/RQ  ──执行──>  CQE ──poll──> WC
│             │                   │                           │           │
opcode        addr/len/lkey       设备相关格式与doorbell      设备相关格式  wr_id/status/opcode
远端addr/rkey

post 成功只到达中间的“已接受”边界，不等于最右侧完成。
```

## WR 和 SGE 的生命周期不能与 payload 混为一谈

下面的结构只保留传统 send WR 中与本课相关的字段，用于说明关系。真实 `struct ibv_send_wr` 还包含 UD、Atomic、Memory Window 等联合成员；不能用此片段替代系统头文件。

```c
/* 结构关系示意，未编译、未运行。 */
struct ibv_sge {
    uint64_t addr;      /* 本地 I/O 地址 */
    uint32_t length;
    uint32_t lkey;
};

struct ibv_send_wr {
    uint64_t wr_id;
    struct ibv_send_wr *next;
    struct ibv_sge *sg_list;
    int num_sge;
    enum ibv_wr_opcode opcode;
    unsigned int send_flags;
    /* opcode 相关字段，例如 remote_addr、rkey、imm_data */
};
```

WR 与 SGE 是提交描述，payload buffer 是它们引用的内存对象。传统 `ibv_post_send()` 在调用期间读取 WR 链表并构造队列工作，因此应用必须保证描述符在调用期间有效；真正的数据缓冲区可能在 post 返回后才被 NIC DMA 读取或写入，必须继续存活到该操作达到相应完成边界。对非 inline SEND/WRITE，源数据不能因为 `ibv_post_send()` 返回 0 就立即改写；对 RECV 或 RDMA READ 的本地目标，NIC 写入完成前也不能把其中内容当作新结果。

`wr_id` 完全由应用定义，常保存请求序号、数组索引或指向请求上下文的指针值。NIC 不理解它的业务含义，只在完成时原样带回。使用指针作为 `wr_id` 时，指向对象必须活到 WC 被处理；使用可回绕的数组索引时，要防止旧完成误匹配新一代请求。稳健实现通常把业务 sequence、buffer index 和请求 generation 分开记录，而不是仅凭一个可能复用的槽位编号判断完成归属。

SGE 列表适合描述分散的 header、payload 或尾部校验区，但硬件支持的 `max_send_sge`、`max_recv_sge` 有上限。多个 SGE 在传输语义上按列表顺序逻辑拼接，不代表它们在本地物理连续。每个 SGE 都必须落在有效 MR 范围内并携带正确 `lkey`；一个 WR 中任一 SGE 越界或 key 错误，都可能使整个请求以 local protection error 失败。

## post 返回值：成功提交与部分提交

[`ibv_post_send(3)`](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_post_send.3)和 [`ibv_post_recv(3)`](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_post_recv.3)都允许通过 `next` 一次提交 WR 链表。返回 0 表示链表中的请求已被接受进入对应工作队列；非零返回表示在某个 WR 处发现了可立即检测的错误，`bad_wr` 指向第一个未成功提交的 WR。关键结论是，失败点之前的 WR 可能已经进入队列，因此调用失败不能一律按“一个都没提交”回滚。

批量提交代码必须记录链表中每个请求的资源归属。若 `bad_wr` 指向第 $j$ 个请求，则前 $j-1$ 个请求需要按已提交状态等待或回收，从第 $j$ 个开始的请求才属于未提交集合。简单地释放整批 payload 或把所有槽位退回 freelist，可能让 NIC 随后访问已经复用的内存。A04 阅读 rping 时也要检查错误路径是否区分“构造失败”“post 立即失败”和“异步 WC 失败”。

post 返回错误只能覆盖当下能够检查的问题，例如队列容量、WR 形态或本地参数；远端 `rkey` 失效、链路故障和传输重试耗尽通常在异步执行后通过 WC status 暴露。应用需要同时处理同步返回值和异步完成错误，不能只检查其中一条路径。

## signaled、unsignaled 与 inline 改变的是哪条边界

发送 WR 设置 `IBV_SEND_SIGNALED` 后，成功执行会在关联 CQ 中产生发送完成；若 QP 创建时 `sq_sig_all` 已启用，则每个发送 WR 都要求完成。signaled 控制“是否为成功请求生成可轮询 WC”，不控制请求是否执行。unsignaled WR 仍会占用 SQ、读取或写入数据并参与队列顺序，只是不为正常成功路径提供一条与其一一对应的 WC。

全部 WR 都 signaled 便于教学和逐请求回收，但会增加 CQE 生成、PCIe 写入和轮询处理压力。生产实现常采用 selective signaling，每隔若干 WR 放置一个 signaled 请求，并把该 WC 作为同一 SQ 上一段请求的回收检查点。不过这不是“看到任意 CQE 就释放所有旧资源”：实现必须依据所用传输和 provider 的顺序保证维护单调提交序号，不能跨 QP 合并水位，错误完成也要单独处理，并确保最终批次有 signaled 请求或显式排空机制。本文没有在硬件上验证某个 signaling 间隔，因此不提供固定数值。

`IBV_SEND_INLINE` 改变源缓冲区复用边界。根据 rdma-core 手册，inline 数据在 post 期间被复制进 WQE，因此调用成功返回后，原应用源缓冲区即可改写；这不表示网络传输已经完成，也不表示远端已经收到或消费。inline 还受 QP 创建返回的 `max_inline_data` 限制，并非任意大小都可用。对于 RDMA READ 或接收缓冲区，inline 不适用，因为它们需要 NIC 把返回或入站数据写入本地内存。

| 情况 | post 返回后源/目标缓冲区 | 观察成功 WC 后 | 仍不能推出 |
| --- | --- | --- | --- |
| 非 inline SEND/WRITE | 源仍可能被 NIC 读取，不可复用 | 可按该操作的本地完成规则复用源 | 对端应用已消费 |
| inline SEND/WRITE | 原源缓冲区可复用 | 请求本身达到本地完成 | 对端应用已消费 |
| RDMA READ | 本地目标仍可能被写入，不可读取为结果 | 本地目标包含本次 READ 结果 | 远端应用参与或确认 |
| RECV | 接收缓冲区归 RQ/NIC 管理，不可复用 | 成功 receive WC 后应用取得该消息 | 业务处理已经结束 |

`IBV_SEND_FENCE` 与 signaling 也不是同一概念。fence 约束某个 WR 相对之前工作的启动条件，signaled 决定是否报告成功完成，inline 决定源数据何时复制进 WQE；三个 flag 解决不同问题。它们更不能与 NVSHMEM 的 `fence`、`quiet` 仅凭名称互换，后者必须按 NVSHMEM 调用域和 API 契约解释。

## WC 是结果记录，不是业务完成证明

[`ibv_poll_cq(3)`](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_poll_cq.3)从 CQ 取出至多指定数量的 WC；被 poll 出的完成会从 CQ 删除，不能再次读取。`ibv_poll_cq()` 返回负数表示轮询调用失败，返回 0 表示当前没有完成，返回正数才表示数组中有对应数量的 WC。对每条 WC，首先检查 `status`；若状态不是 `IBV_WC_SUCCESS`，手册只保证 `wr_id`、status、QP number 和 vendor error 等有限字段有效，不能继续无条件读取 byte length 或 immediate data。

成功 WC 的含义依 opcode 而变化。SEND 的发送 WC 说明发送请求在本端达到完成边界，接收方另有 receive WC；RDMA WRITE 的发送 WC 不构成远端应用通知；RDMA READ 的发送 WC 意味着本地目标可以作为读取结果使用；receive WC 则指出某个预贴缓冲区被入站消息消费。WRITE WITH IMM 在目标侧生成带 `IBV_WC_WITH_IMM` 标志的接收完成，应用还要按网络字节序解释 immediate data，并核对 WC opcode，而不能把任意 receive completion 当成同一种消息。

CQ 可以由多个 QP 共享，也可以同时承载 send 和 receive completion。应用必须用 QP number、`wr_id`、opcode 和自己的请求表完成分派。仅以“先 post 的请求应该先 poll 到”作为跨 QP 关联规则是不安全的；不同 QP 是不同顺序域。即便在单 QP 内，业务协议也应依据明确 ID 和状态转换，而不是依赖线程调度恰好与完成到达顺序一致。

## 轮询与完成事件：通知 CQ 变化之后仍要读取 WC

高吞吐路径可以持续 poll CQ，空闲负载则常用 completion channel 等待通知。事件模式中，`ibv_req_notify_cq()` 为下一次完成请求通知，`ibv_get_cq_event()` 从 channel 取得事件，`ibv_ack_cq_events()` 确认事件，随后仍需调用 `ibv_poll_cq()` 排空实际 WC。完成事件只是“某个 CQ 可能需要处理”的通知，不携带全部 WC 内容，也可能因重新 arm 与 drain 的竞态出现没有对应新 WC 的额外事件。

[`ibv_get_cq_event(3)`](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_get_cq_event.3)给出的典型顺序是：取得并 ack 事件，重新请求下一次通知，再排空 CQ。重新 arm 放在 drain 前，是为了避免在处理现有 CQE 期间错过新到达完成；相应地，应用必须接受一次额外事件。每个成功取得的事件都要 ack，否则销毁 CQ 时会等待未确认事件。性能实现可以批量 ack，但计数必须与实际取得事件一一对应。

无论 poll 还是事件模式，都要保证 forward progress。只 post 不 poll 会让 CQ 溢出或阻碍发送资源回收；只等事件但没有正确 arm 会永久睡眠；在承担 CQ progress 的线程上等待一个必须由同一线程处理的业务 ack，会形成自锁。进入 GPU/IBGDA 后，请求发起和 progress 责任可能分属不同执行单元，但这种依赖环问题仍然存在。

## Receive Request 与 credit：接收缓冲区是可消耗资源

SEND/RECV 中，接收端必须在消息到达前通过 `ibv_post_recv()` 或 SRQ 接口提供 Receive WR。每个 Receive WR 提供一组本地 SGE，入站 SEND 消耗其中一个请求，并在成功时产生 receive WC。接收缓冲区从 post 成功开始交给 RQ/NIC 管理，应用不能同时拿它存放其他数据；成功 receive WC 到达后，应用取得这轮消息并根据 `byte_len` 和协议 header 校验有效内容。

一个 Receive WR 只提供一次 credit。应用处理完消息后若要继续接收，必须重新 post 可用缓冲区。补充过晚会使 RQ 耗尽；在 RC 上，发送端可能经历 RNR 和重试，最终仍可能失败。补充过早同样危险：若应用尚未消费或复制走旧消息便把同一缓冲区再次 post，下一条入站消息可能覆盖旧数据。正确顺序是“receive WC → 校验与消费/转移 → 明确释放该 buffer → 重新 post”。

预贴数量至少要覆盖允许的入站并发和补充延迟。设协议允许对端最多有 $C$ 条尚未获得应用 credit 回收确认的消息，则 RQ 通常不能只准备一个 WR 再期望高并发无阻塞。具体水位还要考虑多个发送者、SRQ 共享、事件处理延迟和故障恢复，不能从链路带宽单独推出。WRITE WITH IMM 虽然 payload 不写入 Receive WR 的 SGE，却仍消耗 Receive Request 以产生通知，因此同样需要 credit 规划。

## 从传输完成到缓冲区所有权

缓冲区安全复用必须分别考虑发送侧源、接收侧消息缓冲和 RMA 目标槽位。发送侧源归 NIC 使用到本地完成边界；RECV 缓冲区从 post 到 receive WC 归 NIC/RQ，之后归应用消费；远端 WRITE 目标则由业务协议决定所有权，因为发送 WC 并不会通知目标应用已经读完。把三种缓冲区都用一个 `completed` 布尔量管理，会丢失必要状态。

对循环使用的单槽 push 协议，可以定义递增轮次 $s$，并给槽位明确四个状态：FREE 表示生产者可写；WRITING 表示 payload 正在传输；READY(s) 表示第 $s$ 轮已经按协议发布，消费者可以读取；CONSUMING(s) 表示消费者持有槽位。消费者处理完后返回 ACK(s)，只有匹配当前 generation 的 ack 才把槽位变回 FREE。网络或线程延迟可能改变各事件的时间，但不能跳过状态。

```text
生产者 A                                           消费者 B

等待 ACK(s-1)，确认 slot=FREE
准备本地 payload
post WRITE(payload, slot) -----------------------> slot=WRITING
确保 payload→ready 的发布顺序
post ready(s) / WRITE_WITH_IMM ------------------> slot=READY(s)
处理本地 send WC：只回收 A 的源请求                校验 s，slot=CONSUMING(s)
                                                    读取并处理 payload
等待 ACK(s) <------------------------------------- 发布 ACK(s)
收到匹配 ACK(s)：slot=FREE
```

其中“确保 payload→ready 的发布顺序”必须由经过核实的传输/API 保证实现。若 payload 和 ready 位于同一 RC QP，可以依据适用的有序规则设计；若跨 QP、跨线程、跨 memory ordering 模式或启用 Relaxed Ordering，则不能自动继承同一顺序域。最保守的工程方法是明确记录发布依赖所依据的规范条款，并用序号让消费者拒绝旧通知。本文不把某种通用 verbs fence 写成所有配置下的万能答案。

ready 只表示消费者可以开始读取，ack 才表示生产者可以覆盖目标槽位。如果 A 收到 WRITE 的本地 WC 后立即发下一轮并覆盖 B 的同一槽位，B 可能尚未调度或仍在处理上一轮；这与网络是否可靠无关。双槽或环形队列能提高并行度，但每个槽仍需要 generation 与 credit，容量扩大不能代替所有权协议。完整的单双槽、背压、最终排空和消费者退出证明留在 B04。

## signaled 水位和业务 ack 是两套独立记账

实现常同时维护本地发送回收水位与远端业务消费水位。前者由 send WC 推进，解决 SQ entry、请求对象和源缓冲区何时可回收；后者由 ACK(s) 推进，解决远端 slot 何时可覆盖。二者可能在时间上接近，但含义不同：本地 WR 已回收时，远端应用仍可能没有消费；远端 ack 到达前，发送侧甚至可能早已释放该 WR 的描述对象。

若采用 selective signaling，可以给每个提交 WR 分配单调 `post_seq`，给 signaled WR 记录一个本地回收边界；处理其成功 WC 后，只回收该 QP 上被该边界覆盖且满足顺序保证的请求。业务 `message_seq` 则放在 ready/ack 协议中，不应直接等同于 `post_seq`，因为一条业务消息可能由多个 WR 组成，一条 WR 也可能只是通知或 ack。分开编号能避免批处理、重试和多 QP 后语义混乱。

最后一批全是 unsignaled WR 是常见退出错误：程序等待一个永远不会生成的 send WC，或直接销毁仍有在途请求的 QP。协议必须在排空点强制放置 signaled 请求，或使用其他有明确保证的完成机制。具体 selective signaling 水位算法需要结合所用 transport 和 provider 验证；本文给出的是设计约束，不是已经测量的优化方案。

## 错误传播与恢复边界

post 立即失败时，依据 `bad_wr` 区分已提交前缀与未提交后缀；WC 失败时，先读取仍被保证有效的 status、vendor error、QP number 和 `wr_id`，再根据 QP 状态处理后续请求。一次异步错误可能使同 QP 的其他在途 WR 以 flush error 完成，因此不能只释放报错的单个 buffer 后继续假定 QP 正常。恢复通常需要停止新提交、排空错误完成、重建连接或 QP，并重新协商已失效的 MR 授权与业务 generation。

协议还要考虑对端在 ready 与 ack 之间崩溃。传输层完成不能证明业务提交，重连后也不能沿用旧内存内容和旧 key 推断上一轮是否消费。需要更强恢复语义的系统必须把 generation、持久化状态或幂等处理放在更高层；RDMA 本身只提供数据移动和规定范围的完成/顺序保证。

## 与 NVSHMEM 的映射

NVSHMEM runtime 隐藏了显式 WR、SGE 和多数 verbs completion，但应用仍面对同类边界。blocking 与 nonblocking RMA 的源复用规则、`quiet` 的完成范围、signal/wait 的发布与观察、不同 host/device/stream 调用域的顺序，都可以用“请求提交水位、数据发布水位、业务消费水位”重新审查。[B02：完成、排序与可见性](b02-completeness-ordering-and-visibility.md)会按 NVSHMEM API 契约给出这些定义。

不能因为某次 NVSHMEM `put` 最终映射为 RDMA WRITE，就把 verbs 的一条 send WC 直接等同于 NVSHMEM `quiet`；runtime 可能聚合、批处理或选择其他 transport。也不能把 NVSHMEM signal 自动当作槽位消费 ack：signal 可以发布 ready，消费者真正完成使用后仍需反向状态或 credit。A03 提供的是分析框架，实现映射必须在固定 NVSHMEM commit 中核查。

## 诊断题与掌握标准

1. **`ibv_post_send()` 返回非零时，为什么不能释放整条 WR 链表对应的所有 payload？** `bad_wr` 之前的前缀可能已经成功提交，NIC 仍可能访问其 buffer；只能把失败点及其后缀视为未提交。
2. **unsignaled WR 是否没有完成过程？** 不是。它仍被执行并占用 SQ，只是正常成功路径不生成一条与其一一对应的 WC；应用仍需设计回收水位和最终排空。
3. **inline WRITE 返回后可以复用源 buffer，是否说明 B 已看到数据？** 不能。inline 只把源数据复制进 WQE，改变 A 的源复用边界，不等于网络完成、远端可见或业务消费。
4. **收到 completion channel 事件后为什么仍要 poll CQ？** 事件只通知 CQ 可能有工作，WC 的 `wr_id`、status 和 opcode 仍在 CQ；事件与 WC 也不是严格一事件一完成。
5. **为什么 receive buffer 不能在收到 WC 后立即重新 post？** 应用必须先校验并完成本轮数据的消费或转移，否则重新 post 后的新消息可能覆盖旧数据。
6. **send WC 与 ACK(s) 分别推进什么状态？** send WC 推进本地请求和源资源回收；ACK(s) 推进远端槽位所有权，使生产者能够覆盖对应 generation。

达到本课目标时，读者应能画出 WR/SGE → WQE → CQE/WC 的层次图，针对 SEND、WRITE、READ 和 WRITE WITH IMM 分别指出本地与远端会产生什么完成，并说明每类 buffer 的复用条件。面对批量 post、selective signaling 和循环槽位，应能给出请求序号、回收水位、receive credit、ready 与 ack 的独立记账方法，并识别最后一批 unsignaled、RQ 枯竭和旧 ack 回绕等错误。

## 证据边界与下一步

本文没有编译或运行代码，没有测量 inline、批量 post、polling、event channel 或 selective signaling 的性能。上游手册引用核对日期为 2026-09-12，链接指向 rdma-core `master`，不替代固定版本源码。WQE/CQE 布局、doorbell 写法、硬件队列缓存以及错误恢复细节依赖 provider 和设备；进入 IBGDA 源码课程时必须重新绑定 NVSHMEM commit 与具体实现。

下一篇 A04 将固定 rdma-core v60.0、commit `5321d809e095d6dd32a1355a5d4aa2ebbda4ba74` 阅读 rping，用真实文件、symbol 和 call path 检查控制消息、RDMA READ/WRITE、receive 补充、状态变化、线程唤醒、消息长度和 MR 容量。理论课中的教学模型只有在源码核对后才能转化为该版本的实现结论。

## 参考资料

- [rdma-core：ibv_post_send(3)](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_post_send.3)
- [rdma-core：ibv_post_recv(3)](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_post_recv.3)
- [rdma-core：ibv_poll_cq(3)](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_poll_cq.3)
- [rdma-core：ibv_get_cq_event(3)](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_get_cq_event.3)
- [rdma-core：扩展 Work Request 提交接口](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_wr_post.3.md)
- [NVIDIA RDMA Aware Networks Programming User Manual 1.7](https://docs.nvidia.com/rdma-aware-networks-programming-user-manual-1-7.pdf)
