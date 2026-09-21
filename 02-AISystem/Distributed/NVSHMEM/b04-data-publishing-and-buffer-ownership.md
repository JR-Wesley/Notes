# B04：数据发布、缓冲区所有权与有界流水协议

把 payload 写到远端只是通信协议的中间一步。一个可重复运行的生产者—消费者程序还必须回答：生产者何时可以覆盖本地 source，何时可以再次写同一个远端 slot；消费者怎样确认某一代数据完整可读，处理结束后又怎样把槽位归还；当生产快于消费时谁应等待；最后一条消息如何排空，双方如何在不留下等待者的情况下退出。

本课以一个生产者 PE 0、一个消费者 PE 1 和容量为 `K` 的有界槽位队列为主模型。它把 [B02：完成、排序与可见性](b02-completeness-ordering-and-visibility.md)的六阶段状态与 [B03：CUDA 执行域、协作通信与前进性](b03-cuda-execution-and-cooperation.md)的 thread/block/stream 交接组合成完整协议。正文按 2026-09-12 可见的 NVSHMEM 3.7.2 发布周期滚动 API 文档核对；状态机和不变量是基于 API 契约的教学推导，不是 NVSHMEM 内部实现。代码均未编译、未运行，硬件行为和性能也未验证。

官方 Execution Model 关于 pointer argument overlap、buffer in-use 期间的读写限制和多 symmetric segment 参数的英文原文与中文对照，见 [双语核对笔记：NVSHMEM Execution Model](official-docs/r04-execution-model-and-progress.md)。该规范给出 calling PE 上的 buffer access discipline；本课进一步处理 remote consumer 尚未结束时的 slot ownership。

## 先区分两个缓冲区和三种所有权

设 PE 0 在本地 `source[p]` 中生成第 `p` 条消息，再 PUT 到 PE 1 的 `slot[j]`。这里至少有两个物理对象：生产者本地 source 和消费者所在 PE 的远端 slot。blocking PUT 返回或 NBI PUT 经 flush 后，本地 source 可能已经允许复用；但远端 slot 即使已经交付，也可能仍在被消费者读取。把二者都简称为“buffer”，容易错误地用一个完成条件同时释放两个不同对象。

协议需要跟踪三种所有权。生产者拥有本地 source 时可以写入下一条消息；通信层取得读取 source 的临时权利，直至满足本地完成；远端 slot 则在 producer-free、in-flight、consumer-readable、consumer-processing 和 acknowledged 等阶段之间转移。`quiet` 能把请求推进到远端交付，却不会替消费者归还 slot；只有由消费者在处理结束后发布的 ack，才能把远端 slot 的覆盖权交还生产者。

| 对象 | 谁会访问 | 可复用依据 | 常见错误 |
| --- | --- | --- | --- |
| 本地 source | 生产 kernel、NVSHMEM 发起路径 | blocking PUT 的本地返回契约，或 NBI PUT 对应的 flush/quiet | 把“请求已发起”当作 source 可覆盖 |
| 远端 slot | 生产者远端写、消费者本地读 | 消费者完成读取并返回该代 ack | 把 quiet 或 ready 当作消费结束 |
| ready signal | 生产者远端 signal、消费者本地 wait/test | 单调序号协议；通常不由消费者普通 store 清零 | 用 0/1 反复复位，制造丢通知或冲突访问 |
| ack signal | 消费者反向 signal、生产者本地 wait/test | ack 序号与具体 ticket/slot 对应 | 只确认收到 ready，没有确认 payload 已处理 |

因此，“发送完成”至少要进一步说明是 source 可复用、payload 已交付，还是 slot 已归还。没有对象和阶段，单独一句“PUT 已完成”不足以指导下一次写入。

## payload、ready 与 ack 分别证明什么

payload 是业务数据，可能还包含 `length`、消息类型、校验字段和 ticket。ready 是从生产者指向消费者的发布事件，表示某一 ticket 的 payload 已经交付并可开始消费。ack 是反方向的消费完成事件，表示消费者不再访问该 ticket 所在 slot，生产者可以在协议条件满足后覆盖它。

最直接的发布方式是 put-with-signal。生产者把 payload 和 `ready=p+1` 作为同一次 signaling operation；当消费者的本地 signal wait 观察到这个更新时，API 保证与该 signal 对应的 `dest` 数据字已经交付。若生产者改为普通 PUT 加独立 signal，就必须使用同目标 `fence` 或先 `quiet`，证明 payload 的交付先于 ready 被观察。signal 只覆盖它在契约中对应的数据，不能自动发布未排序的其他 header 或另一个 slot。

ack 不携带“远端交付 payload”的含义，它是应用协议事件。消费者必须先完成所有会读取 slot 的 CUDA threads、kernel 或 stream 工作，再向 PE 0 更新 ack。若一个 thread 负责回 ack，而其他 threads 共同处理数据，应先使用与作用域匹配的 CUDA 同步完成线程交接；否则 ack 可能在其他 threads 尚未读完时提前到达。生产者观察 ack 后取得 slot 所有权，但仍需单独满足下一代 source 准备和通信排序。

```text
PE 0：生产者                                      PE 1：消费者

写 source[p]                        producer owns source
发起 payload -> slot[j]             communication reads source
本地完成                             source can be reused
payload 远端交付 + ready=p+1  ─────────────────> slot becomes readable
                                          wait ready
                                          read/process slot[j]
                                 <──────── ack=p+K after consumption
观察 ack                                           slot returned to producer
下一代才可覆盖 slot[j]
```

ready 与 ack 必须指向同一个逻辑 ticket。若 ready 只是布尔值 1，消费者无法判断当前看到的是上一轮残留的 1，还是本轮新写入的 1；若消费者把它普通 store 为 0，而生产者同时 signal SET 为 1，还可能对同一位置产生不受允许的冲突访问。用单调序号并让 signal object 只由 signaling operations 更新，能够避免反复清零产生的 ABA 和并发写问题。

## 用 ticket 和每槽序号统一单槽、双槽与 K 槽

令生产者的消息 ticket 为 `p = 0, 1, 2, ...`，消费者按相同顺序使用 ticket `c`。队列容量为 `K`，ticket `p` 使用槽：

$$
j = p \bmod K
$$

每个 slot `j` 对应两个控制序号：位于消费者 PE 的 `ready[j]`，以及位于生产者 PE 的 `ack[j]`。协议初始化为：

$$
ready[j]=0,\qquad ack[j]=j,\qquad 0\le j<K
$$

当生产者准备 ticket `p` 时，它只能在 `ack[j] == p` 后使用槽 `j=p mod K`。发布 payload 后，把消费者的 `ready[j]` 设置为 `p+1`。消费者处理 ticket `c` 时等待 `ready[j] == c+1`；处理结束后，把生产者的 `ack[j]` 设置为 `c+K`。下一次映射到同一槽的 ticket 正好是 `p+K`，因此消费者返回的值就是该槽下一次允许生产的序号。

| 阶段 | 槽 `j=p mod K` 的条件 | 责任方 | 允许的动作 |
| --- | --- | --- | --- |
| Free for ticket `p` | 生产者本地 `ack[j] == p` | 生产者 | 写 source、准备该 ticket |
| Publishing | payload 正在发往消费者 | 生产者/通信层 | 不得让消费者读取，不得用下一代覆盖 slot |
| Ready | 消费者本地 `ready[j] == p+1` | 消费者 | 可以读取该 ticket 的 payload |
| Consuming | 消费者仍可能访问 slot | 消费者 | 生产者不得重用该 slot |
| Returned | 消费者向生产者发布 `ack[j]=p+K` | 消费者 | 下一次映射到该 slot 的 ticket 获得准入条件 |

这个表示法不需要让消费者把 ready 清零，也不需要生产者把 ack 清零。每次状态变化都携带 ticket 含义，可以从错误值直接定位丢失、重复或越序的轮次。`NVSHMEM_CMP_EQ` 适合严格的单生产者、单消费者、有序协议，因为同一槽在收到 ack 前不可能合法发布下一代；若使用 `CMP_GE` 支持累计确认，就必须额外证明不会发生无符号回绕和非法跳号。

## 单槽协议：正确但不能隐藏消费延迟

令 `K=1`。初始 `ack[0]=0`，ticket 0 可以立即使用 slot 0；消费者看到 `ready[0]=1` 后处理，结束时回 `ack[0]=1`。ticket 1 仍映射 slot 0，只有生产者看到 ack 1 后才可覆盖；ticket 2 等待 ack 2，依次类推。

```text
ticket 0: wait ack=0 -> publish ready=1 -> consume -> ack=1
ticket 1: wait ack=1 -> publish ready=2 -> consume -> ack=2
ticket 2: wait ack=2 -> publish ready=3 -> consume -> ack=3
```

单槽最多有一个未确认 ticket。它容易证明、占用内存少，也适合作为首个正确性基线；代价是生产者在消费者处理期间不能发布下一条消息。如果 payload 的计算、网络传输和消费都在同一串行关键路径上，单槽吞吐大致受三者总时间限制。这里的“受限”只是结构分析，不是测得的性能结论。

下面给出设备端伪代码。为突出协议，假设每个角色只有一个控制 thread，payload 的本地 CUDA 协作已在调用前后正确完成；使用 block API 时还必须应用 B03 的参与和交接规则。

```cpp
// 教学伪代码，未编译、未运行。
// PE 0: producer, K == 1
for (uint64_t p = 0; p < message_count; ++p) {
    nvshmem_signal_wait_until(&ack[0], NVSHMEM_CMP_EQ, p);

    produce(local_source, p);

    nvshmem_putmem_signal(remote_slot,
                          local_source, bytes,
                          remote_ready, p + 1,
                          NVSHMEM_SIGNAL_SET, /*pe=*/1);

    // blocking put-with-signal 返回后 source 可按其本地完成契约复用；
    // remote_slot 仍需等待 ack==p+1 才能再次覆盖。
}

// PE 1: consumer
for (uint64_t c = 0; c < message_count; ++c) {
    nvshmem_signal_wait_until(&ready[0], NVSHMEM_CMP_EQ, c + 1);
    consume(local_slot, c);

    nvshmemx_signal_op(remote_ack, c + 1,
                       NVSHMEM_SIGNAL_SET, /*pe=*/0);
}
```

最后一次循环结束时，生产者不能仅因 put-with-signal 已返回就释放 slot。若协议要求消费者已完成全部消息，应再等待 `ack[0] == message_count`。消费者的循环返回只在消费代码本身同步完成时成立；若 `consume` 只是把后续工作排入另一条 stream，ack 必须排在那个消费完成事件之后，而不能由当前 thread 立即发送。

## 双槽协议：允许一条在消费、一条在生产或传输

令 `K=2`。初始 `ack[0]=0`、`ack[1]=1`，所以 ticket 0 和 1 分别可以使用 slot 0 和 1，而不等待旧数据。消费者处理 ticket 0 后回 `ack[0]=2`，允许 ticket 2 重用 slot 0；处理 ticket 1 后回 `ack[1]=3`，允许 ticket 3 重用 slot 1。

```text
slot 0: ticket 0 --ack=2--> ticket 2 --ack=4--> ticket 4 ...
slot 1: ticket 1 --ack=3--> ticket 3 --ack=5--> ticket 5 ...
```

在理想时序中，消费者处理 slot 0 时，生产者可以填充并发布 slot 1；随后消费者转向 slot 1，生产者在收到 ack 2 后准备 ticket 2。双槽因此可能重叠相邻轮次的生产、传输和消费，但它并不自动产生 overlap：如果所有操作位于同一串行 stream、生产者在每次发布后立即 quiet 并等待 ack，或硬件路径本身无法并行，两个槽仍可能串行执行。是否获得端到端收益需要 E01 的时间线和实测证据。

双槽也没有消除每槽 ownership。生产者在 ticket 0 的 source 本地完成后，可以重用本地 source，但不能在收到 `ack[0]=2` 前用 ticket 2 覆盖远端 slot 0。若消费者仍在读取 ticket 0，提前发布 ticket 2 会让一个 payload 的不同 cache line 或元素来自两代数据，ready 2 即使稍后正确到达，也无法修复已经发生的混合读取。

统一算法可以写成容量 `K` 的形式：

```cpp
// 教学伪代码，省略 CUDA thread-group 和错误处理；未编译、未运行。
for (uint64_t p = 0; p < message_count; ++p) {
    size_t j = p % K;
    nvshmem_signal_wait_until(&ack[j], NVSHMEM_CMP_EQ, p);

    produce(source[j], p);
    nvshmem_putmem_signal(remote_slot[j], source[j], bytes,
                          remote_ready[j], p + 1,
                          NVSHMEM_SIGNAL_SET, consumer_pe);
}

// Drain: 证明每一个已发布 ticket 都已消费并归还。
uint64_t begin = (message_count > K) ? message_count - K : 0;
for (uint64_t p = begin; p < message_count; ++p) {
    size_t j = p % K;
    nvshmem_signal_wait_until(&ack[j], NVSHMEM_CMP_EQ, p + K);
}
```

上面的 drain 循环只等待每个可能仍在途的最后 ticket。更易审计的实现也可以记录每槽最后发布的 ticket，再逐槽等待 `ack[j] == last[j]+K`。不能机械地等待从未发布过的槽：当 `message_count < K` 时，未使用槽只保留初始 ack，没有消费者会为它产生新 ack。

## 背压不是性能故障，而是有界安全条件

当生产者的 ticket `p` 到达槽 `j`，却观察到 `ack[j] != p`，说明容量 `K` 内仍有未消费数据。此时等待是背压：它防止生产者越过消费者并覆盖槽位。删除 wait 会提高短期提交速度，却破坏队列安全；增加 K 可以容纳更多 outstanding messages，但只推迟拥塞，不能消除长期生产率高于消费率的问题。

可以定义已发布但未确认的数量 `O`。对于严格有序的单生产者—单消费者协议，应维持：

$$
0 \le O \le K
$$

若生产者累计发布 `P` 条，消费者累计确认 `A` 条，则 `O=P-A`。生产者只在目标槽 ack 匹配时发布下一条，正是用分散的每槽序号维护这个容量不变量。背压等待是否阻塞一个 thread、一个 block 或一条 stream，要结合 B03 的 forward progress 审查；协议上的“应该等待”不代表任何放置方式都不会形成调度环。

忙等并非唯一选择。device thread 可使用 wait/test，host 可让 CPU 等待 stream 或控制事件，应用也可批量测试多个槽的 ready/ack。`wait_until_any`/`test_any` 适合从多个本地对称控制对象中寻找可用项，但其返回顺序不自动等同于业务 ticket 顺序；若允许乱序消费，payload 必须携带 ticket 和长度，并为每槽独立确认。选择阻塞 wait 还是非阻塞 test 是调度策略，不能改变容量与所有权不变量。

## 初始化必须建立共同的序号基线

payload、ready 和 ack 均应位于满足相应 NVSHMEM API 要求的对称对象中。所有 PEs 以兼容顺序完成对称分配；随后初始化本地控制对象，使消费者的 `ready[j]=0`、生产者的 `ack[j]=j`。若初始化通过 CUDA kernel 或 asynchronous memset 完成，host 在进入 NVSHMEM barrier 前必须先同步对应 stream，因为 host barrier 不排空尚未执行的 GPU stream 工作。

初始 barrier 的作用是让所有参与 PE 在通信开始前看到“控制对象已经初始化、地址仍有效、角色已经确定”的共同阶段。它不能替代每轮 ready/ack，也不能允许不同 PEs 以不同顺序分配 `slot`、`ready` 和 `ack`。使用 `nvshmem_calloc` 可得到全零初值，但 `ack[j]=j` 中非零元素仍需显式写入并完成相应 CUDA/host 交接。

控制对象应有单一更新方式。signal object 由 put-with-signal 或 `nvshmemx_signal_op` 更新，并由 signal wait/fetch 观察；不要为了“清标志”混入普通 CUDA store。一般 wait/test 对象可由 AMO 更新，但并发访问仍须满足 OpenSHMEM 冲突例外。payload 在 producer owns 或 consumer owns 阶段分别只由合法一方访问，避免一方远端写、另一方同时普通读的未定义冲突。

## 长度、类型与校验也必须随 ticket 发布

真实消息经常不是固定长度。若 `length` 与 payload 分开写，而 ready 只与 payload 中一部分关联，消费者可能看到新 ready 却读取旧 length。稳妥做法是把 `ticket`、`length`、`type` 等 header 放入与 signal 对应的同一次 payload 区域，或明确使用 ordering 把所有 header 和 data 都排列在 ready 之前。消费者在 ready 后先验证 ticket 与预期一致，再检查 `length <= slot_capacity`，最后才访问 payload。

校验字段不能替代同步，但能暴露协议错误。调试版本可在 header 中加入 ticket、长度和内容 hash；若 ready ticket 正确但 header ticket 不同，说明发布范围或覆盖条件存在问题；若 header 正确但 hash 错误，需检查 source 交接、长度、越界或 transport。没有实际运行时不能填写“校验已通过”，只能记录预期不变量和失败含义。

空消息也要占 ticket，除非协议明确规定不占。否则生产者与消费者会对 ticket 计数产生分歧。`EOS`（end of stream）也应被视为一种带 ticket 的控制记录，而不是不经排序地另写一个 stop flag。

## 有限流与未知长度流的正确结束

若双方预先知道 `message_count=N`，消费者严格处理 ticket `0..N-1`，生产者在发布 N 条后等待所有已使用槽的最终 ack。双方确认队列为空，再结束 device kernel、同步相关 CUDA stream，并以所有 PEs 相同的顺序进入对称释放和 `nvshmem_finalize`。`nvshmem_free` 的入口 barrier 和 finalize 的隐式全局 barrier 能协调资源释放及待完成通信，但不能代替业务消费确认：pending PUT 完成不等于消费者处理了 payload。

若消息数事先未知，生产者应通过正常槽位发布一个 EOS record。消费者观察 EOS ready 后，先确认此前 ticket 已按协议处理，再消费 EOS、返回 ack，然后退出。生产者必须等待 EOS 的 ack，才能证明消费者已经离开读取阶段。把 EOS 放在同一 ticket 序列中，可让它继承既有 payload→ready 顺序和槽位 ownership。

```text
... data ticket N-1 -> ack
EOS ticket N        -> ready -> consumer sees EOS -> ack
producer waits EOS ack
双方同步 kernel/stream
按集合顺序 free / finalize
```

独立 stop flag 容易越过尚未完成的数据。如果确实需要 out-of-band cancellation，必须另行定义它是“停止接收新数据”“丢弃在途数据”还是“排空后退出”，并规定谁唤醒可能阻塞的 wait。NVSHMEM 本身不是容错消息队列：某个 PE 异常退出或永远不发送预期 ack 时，对端可能永久等待；timeout 只能帮助诊断，不能自行恢复一致状态。

## 序号回绕与比较操作不能被忽略

64 位 ticket 的实际回绕周期很长，但形式正确性仍需写明假设。若使用 `NVSHMEM_CMP_GE`，无符号序号从最大值回到 0 后，普通数值大小关系不再表示“更新”；旧大值可能错误地满足新一轮等待。最简单的约束是在一次作业中禁止 ticket 回绕，并在接近上限前排空、重建协议状态或终止作业。

若必须支持回绕，需要采用有界序号空间和明确的模比较，保证同时 outstanding 的距离小于序号空间的一半；但 NVSHMEM wait 的内建 `GE` 只是普通数值比较，不会自动执行这种模序关系。应用可能改用 `EQ` 加每槽准入，或用额外 epoch 字段和 AMO 状态机。不能因为类型是 `uint64_t` 就宣称回绕“不可能”。

signal `SET` 适合写入明确 ticket。signal `ADD` 更像累计到达计数：它能统计更新次数，却不能独自指出哪个 producer、哪个 slot 或哪段 payload 已准备好；重复发送会把计数推进两次，丢失发送会留下永久缺口。对于本课的一生产者一消费者槽位协议，SET 的状态含义更直接。ADD 可用于更复杂的 fan-in，但必须和独立的每消息发布记录组合。

## 多生产者：原子预留只解决唯一 ticket

当多个生产者写同一个消费者队列时，可以在消费者 PE 的对称 `tail` 上执行 atomic fetch-add，为每个生产者返回不同的 ticket。fetching AMO 在返回旧值并更新目标之间具有原子性，所以它适合解决“两个生产者是否拿到同一个编号”。但唯一 ticket 只解决预留，不解决发布、容量和消费顺序。

假设 producer A 预留 ticket 5 后暂停，producer B 预留 ticket 6 并完成 payload。若消费者必须按序处理，它仍要等待 ticket 5；若系统只维护一个“最大 ready=6”，消费者可能错误地认为 5 也已发布。这个尚未发布的 reservation 是 publication hole。正确设计需要每槽 sequence/ready、独立描述符或其他能区分 reserved 与 published 的状态，不能在 fetch-add 后立即递增全局 ready。

多个生产者还必须在容量上协调。ticket `p` 的生产者只有在槽 `p mod K` 已为该 ticket 空闲时才能写；否则即使 ticket 唯一，也会覆盖较早消息。生产者如何观察消费者的 free state 取决于状态放置：NVSHMEM wait 观察调用 PE 本地对象，远端状态若不主动回送，就需要 GET/AMO polling 或分布式 credit。每生产者独立队列通常更容易证明，代价是消费者要在多个 ready 数组中选择；共享 MPSC ring 更节省固定结构，却需要完整的预留—发布—消费—回收算法。

因此，本课不把“remote fetch-add tail + PUT”冒充为完整 MPSC 队列。它只给出三条必要条件：ticket 唯一、slot 不越过容量、消费者能区分已预留与已发布。D02 研究 IBGDA 多生产者提交队列时还会遇到相似的 reservation hole 和回收问题，但届时必须基于固定源码，而不是把本节教学模型当成实现事实。

## 一个简短的安全性证明

对 SPSC、容量 K 的协议，可以用三个不变量证明不覆盖未消费数据。

第一，ticket 到 slot 的映射固定为 `j=p mod K`，所以同一 slot 的相邻使用者相差 K。第二，生产者只有看到 `ack[j]=p` 才能发布 ticket p；该值初始为 j，之后只由消费者在完成 ticket `p-K` 后写成 p。第三，消费者只有看到 `ready[j]=p+1` 才读取 ticket p，而这个 signal 只在对应 payload 已交付后可观察。

由第二条可知，生产者写 ticket p 时，消费者已经结束对 ticket `p-K` 的所有访问，因此不会发生覆盖未消费数据。由第三条可知，消费者开始读 ticket p 时，对应 payload 已交付，因此不会在发布前读取。每个 ticket 消费结束才产生下一次 slot credit，所以 outstanding 数量不能超过 K。这个证明依赖 single producer、single consumer、无序号回绕、signal/wait 契约正确使用，以及参与 threads/streams 已按 B03 完成交接。

安全性不等于活性。若消费者永不运行、某个 signal 永不发送、collective launch/stream 形成等待环，协议可以保持“不读坏数据”却永远不结束。活性还依赖运行时最终交付、GPU 前进性、双方按照协议继续执行和没有进程故障。这些假设应与安全不变量分开记录。

## 常见错误及定位方式

目标偶尔读到混合代数据时，先检查生产者是否在 ack 前重用远端 slot，而不是继续增加 quiet。quiet 只完成写入；如果消费者仍在读，它反而可能让下一代更快覆盖当前数据。随后检查 ack 是否在所有消费 threads/streams 完成后才发送，以及 ticket、slot index 和字节长度是否一致。

消费者永久等待 ready 时，检查双方初始序号、ticket 是否从相同值开始、producer 是否等待了错误的 ack、put-with-signal 的 `sig_addr` 是否指向消费者本地对应 symmetric object，以及设备端 wait 是否满足 B03 的 collective launch 和前进性。若只在回绕后发生，检查 `GE` 比较是否错误处理无符号 wrap。

生产者永久等待 ack 时，检查消费者是否只“收到”而没有在处理后回 ack；EOS 是否走相同槽位协议；ack 的目标 PE、slot index 和更新值是否为 `c+K`；消费工作若在另一 stream，回 ack 的 stream 是否等待了消费完成 event。不要用 finalize 试图唤醒一个仍在 kernel wait 中的消费者。

双槽没有提升性能时，画出 producer、communication 和 consumer 的实际 CUDA 时间线。若每轮都等待 ack 后才发布下一个 ticket，程序实际上仍按单槽运行；若两条 stream 之间用 device-wide synchronization 串行化，第二个槽也无法形成 overlap。性能诊断要在正确性成立后进行，不能通过删除 ownership wait 获得虚假的吞吐。

## 诊断题与参考推理

### 问题一：blocking PUT 返回后，哪些对象可以复用？

按 blocking PUT 契约，本地 source 在数据复制出本地数组后可以复用；远端 slot 只达到或正在走向交付阶段，消费者可能尚未读取。若下一代仍写同一远端 slot，必须等待消费者对上一代返回 ack。

### 问题二：消费者看到 `ready=p+1` 后立即回 ack，再异步启动消费 kernel，是否安全？

不安全。ack 表示消费者已结束对 slot 的访问，生产者可能马上覆盖。应让消费 kernel/stream 完成读取，再在其后排入或执行 ack；若由另一个 thread 回 ack，还需本地同步完成线程交接。

### 问题三：双槽为何允许 ticket 0 和 1 连续发布，却不允许 ticket 2 无条件发布？

初始 `ack[0]=0`、`ack[1]=1` 分别授予前两个 ticket 对两个不同槽的所有权。ticket 2 再次映射 slot 0，只有消费者完成 ticket 0 并设置 `ack[0]=2` 后才获得该槽；这正是容量为 2 的背压。

### 问题四：消费者能否在看到 ready 后把它普通写回 0？

不应这样设计。signal wait 文档要求 signal object 只通过 signaling operations 更新；普通清零可能与远端 signal 并发冲突，也丢失代际信息。使用单调 ticket，让下一轮等待不同值，无需复位。

### 问题五：程序有 N 条消息，生产者发布完第 N 条后直接 barrier，能否证明消费者处理完成？

不能仅由发布完成推出消费完成。barrier 可以完成相应通信更新并让 PEs 会合，但消费者必须先完成业务处理并返回最后的 ack，再进入排空后的共同 barrier。否则 barrier 可能只证明 payload 已到达，而非应用消费结束。

### 问题六：atomic fetch-add 分配了唯一 ticket，为什么还会覆盖？

fetch-add 只保证编号不重复。若 ticket 与队列 head 的距离超过 K，或对应 slot 的上一代尚未 ack，唯一 ticket 仍会映射到繁忙 slot。还需要容量准入、per-slot publication state 和回收协议。

### 问题七：`ready >= expected` 为什么在 64 位回绕后可能错误？

内建 GE 是普通无符号数值比较。回绕后新序号变小，旧的大值可能继续满足条件，或合法新值无法满足原比较。应限制一次运行不回绕，或采用经过证明的模序/epoch 设计；不能默认 GE 理解环形序号。

## 掌握标准与后续路线

完成本课后，读者应能分别指出本地 source 和远端 slot 的复用条件，画出 payload→ready→consume→ack 的双向时序，并用 `ack[j]=p`、`ready[j]=p+1`、`ack[j]=p+K` 推导任意 K 槽协议。读者还应能证明 outstanding 不超过 K、解释背压为何是安全条件、为有限消息和 EOS 两种结束方式写出 drain 条件，并说明 finalize 的 barrier 为什么不能代替消费确认。

读者还应能够识别多生产者 reservation hole，理解 atomic fetch-add 只分配唯一 ticket，不自动发布消息或管理容量。若只能画“PUT 后 signal”，却不能说明下一代何时覆盖 slot，仍未完成 B 模块关于 ownership 的核心目标。

B00—B04 至此形成一条完整主线：编程模型定义 PE 与 PGAS，B01 确定远端对象，B02 确定完成与可见性，B03 确定执行者和前进性，本课最终闭合消费与复用。下一阶段 `C01：从 host RDMA 到 GPUDirect RDMA 与 IBGDA` 将分离 API 发起者、NIC 数据访问者和 NIC 工作提交者；该正式文件尚未创建，因此本课不建立失效链接。配套的单槽/双槽实验和有限状态模型也仍为规划，不能把本文纸面证明当成已运行结果。

## 主要来源

- [NVSHMEM Signaling Operations](https://docs.nvidia.com/nvshmem/api/latest/gen/api/signal.html)：put-with-signal 的 payload 交付保证、signal 原子更新、SET/ADD 以及与其他 transfer 之间的排序边界。
- [NVSHMEM Point-To-Point Synchronization](https://docs.nvidia.com/nvshmem/api/latest/gen/api/sync.html)：wait/test 在调用 PE 本地对称对象上的比较语义、signal wait 的更新要求以及 any/vector 变体。
- [NVSHMEM Remote Memory Access](https://docs.nvidia.com/nvshmem/api/latest/gen/api/rma.html)：blocking/NBI PUT 的本地 source 与远端 delivery 完成边界。
- [NVSHMEM Memory Ordering](https://docs.nvidia.com/nvshmem/api/latest/gen/api/ordering.html)：fence、quiet、flush 的顺序和完成范围，以及不同发起域的交接责任。
- [NVSHMEM Atomic Memory Operations](https://docs.nvidia.com/nvshmem/api/latest/gen/api/amo.html)：fetching/non-fetching AMO 的完成边界，以及 fetch-add 返回旧值并原子更新的保证。
- [Using NVSHMEM](https://docs.nvidia.com/nvshmem/api/latest/using.html)：OpenSHMEM 冲突访问规则、AMO 与 wait/test、signal 与 wait/test 的例外，以及更新的最终可见性。
- [NVSHMEM Collective Communication](https://docs.nvidia.com/nvshmem/api/latest/gen/api/collectives.html)：barrier 的远端更新完成语义和参与范围；本文据此区分通信排空与业务消费确认。
- [NVSHMEM Memory Management](https://docs.nvidia.com/nvshmem/api/latest/gen/api/memory.html)：对称分配/释放的 collective 顺序及 `nvshmem_free` 的入口 barrier。
- [Library Setup, Exit, and Query](https://docs.nvidia.com/nvshmem/api/latest/gen/api/setup.html)：`nvshmem_finalize` 的 collective/implicit barrier 语义与资源释放边界。
- [NVSHMEM Examples](https://docs.nvidia.com/nvshmem/api/latest/examples.html)：官方 ring 示例中 fence、signal wait 与 signal op 的点对点发布结构。
- [NVSHMEM 3.7.2 Release Notes](https://docs.nvidia.com/nvshmem/release-notes-install-guide/release-notes/release-3720.html)：本课核对时的当前发布周期；API Guide `latest` 仍为滚动文档，不是固定源码 commit。
