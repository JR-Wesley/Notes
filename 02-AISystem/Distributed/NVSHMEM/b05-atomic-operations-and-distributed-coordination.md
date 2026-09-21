# B05 NVSHMEM 远程原子操作与分布式协调：从单字更新到可证明的队列协议

## 本课要解决的问题

前面的 PUT/GET 适合搬运数据，却不能在多个 PE 同时修改同一控制变量时避免丢失更新。假设四个生产者都想为一批任务分配唯一编号，如果它们分别 GET 远端计数器、在本地加一、再 PUT 回去，那么多个生产者可能读到相同旧值并覆盖彼此的结果。Atomic Memory Operation（AMO）把读取、判断或更新组合为一个不可被其他受支持原子访问插入的操作，使多个 PE 可以围绕一个 symmetric object 协调。

但“使用 atomic”离“协议正确”仍有很大距离。一个 fetch-add 可以保证 ticket 不重复，却不能证明队列没有越界、对应 payload 已经写完、消费者按什么顺序处理，或槽位何时可以回收；一个 compare-swap 可以竞争某个状态字，却不会自动为状态字之外的数据提供发布—获取语义。本课的目标不是罗列 API，而是回答四类问题：AMO 原子性究竟保护什么；fetching 与 non-fetching 操作何时完成；怎样用返回旧值证明唯一性；以及为什么 counter、allocator、lock 和 work queue 都必须在 atomic 之外补充容量、publication、reclamation 与 progress 条件。

## 资料版本与论述边界

官方 Memory Model 对“同一位置、相同 datatype”的 AMO 排他性，以及不同 datatype、atomic/non-atomic NVSHMEM 和普通 load/store 混用的未定义行为，已按“英文原文—中文翻译—技术解读”整理在 [双语核对笔记：NVSHMEM Memory Model](official-docs/r03-usage-memory-model-and-consistency.md)。本课在此基础上继续推导 counter、allocator 和 queue protocol，不把单字原子性扩大为多对象事务。

本文于 2026-09-13 依据 NVIDIA 官方 NVSHMEM API Guide 的滚动 `latest` 页面核对，并以 [NVSHMEM 3.7.2 Release Notes](https://docs.nvidia.com/nvshmem/release-notes-install-guide/release-notes/release-3720.html)标识发布周期。主要契约来自 Atomic Memory Operations、Using NVSHMEM、Memory Ordering、Signaling Operations 和 Point-To-Point Synchronization。滚动 API 文档不是固定源码版本，因此本文不会断言一次 NVSHMEM AMO 必然变成某种 mlx5 WQE，也不会假定所有拓扑都使用 NIC 原生 atomic。

当前 Atomic Memory Operations 文档明确提示：对于 remote transports（文档列举 UCX、IBRC），AMO 只支持 device operation；host-side AMO 只支持 NVLink-connected PEs。具体平台还受 build、transport、GPU/NIC 和运行配置约束。3.7.2 Release Notes 进一步指出，只有 PCIe peer-to-peer 而没有 InfiniBand 的系统，需要使用能以 sockets 支持 atomics 的 UCX transport 才能覆盖 NVSHMEM atomic API。应用必须以目标环境的支持矩阵和实测为准。

文中的 CUDA 代码与状态机用于纸面推导，未编译、未运行，没有 GPU、NVLink、InfiniBand、UCX 或 IBGDA 实测结果。关于线性化点、序列化顺序和队列状态的表达是规范推理工具，不代表 runtime 暴露了可观测硬件时间戳。

## 一、为什么普通 GET—修改—PUT 会丢失更新

设 PE 0 上有一个 symmetric counter，初值为 10，PE 1 和 PE 2 都执行“读取、加一、写回”。如果读取和写入是两个独立通信操作，可能出现下面的交错：

```text
PE 1                       PE 0 counter                    PE 2
GET -> 10                  10                              GET -> 10
local 10 + 1                                               local 10 + 1
PUT 11                     11
                                                        PUT 11
                            11
```

两个 PE 都完成了逻辑上的加一，但最终结果只增加一次。问题不在 PUT 或 GET 本身是否可靠，而在“读旧值”和“写新值”之间存在可被其他参与者插入的窗口。仅在每个 PE 内增加 CUDA barrier 或 host mutex 也不能解决跨 PE 竞争，因为这些机制没有共同保护 PE 0 上的远端对象。

fetch-add 将“取旧值”和“加增量”合成一个 AMO。对同一位置、同一 datatype 的受支持 AMO 并发访问具有排他性，因此两个参与者不可能都把同一个旧值作为本次 fetch-add 的返回值。若初值为 10，一个调用返回 10，另一个返回 11；最终值为 12。究竟哪个 PE 先得到 10 取决于实际执行顺序，程序不能依赖 PE ID 或发起时间猜测赢家。

## 二、原子性边界是“一个位置的一次操作”，不是事务

NVSHMEM AMO 只能作用于 remotely accessible symmetric object。原子保证的核心范围是：对同一内存位置、使用相同 datatype 的 AMO 并发访问互斥；同一位置上匹配 datatype 的 AMO 与 wait/test 组合也属于规范允许的例外。可以把这些访问理解为存在一个与各次返回值一致的串行顺序，但这个顺序只用于推导目标原子对象，不会自动把附近的 payload、另一个 atomic word 或其他 PE 上的对象纳入同一事务。

下面的结构包含两个独立字段：

```cpp
struct Descriptor {
    uint64_t state;
    uint64_t owner;
};
```

即使分别对 `state` 和 `owner` 使用 AMO，也不能据此声称观察者永远看到一对原子快照。一次线程可能先更新 `state`，另一线程在 `owner` 更新前读取两个字段，从而观察到合法但业务上不一致的组合。若协议要求二者不可分割，就需要把状态编码进一个受支持的原子字，或设计带版本号的多阶段协议；不能把两个标量 AMO 拼成多字事务。

原子性还不覆盖混合访问方式。当前 NVSHMEM memory model 把“同一位置、同一 datatype 的 AMO 之间”以及“AMO 与 wait/test”列为并发冲突的例外；若同一位置同时存在 AMO 与普通 RMA PUT、不同 datatype 的 AMO，或 AMO 与未经契约覆盖的普通 CUDA load/store，则不能假设彼此互斥，可能构成未定义冲突。控制字应在设计时指定唯一访问纪律，而不是有时 atomic add、有时普通写零。

## 三、fetching 与 non-fetching AMO 的返回边界不同

NVSHMEM 把 AMO 分成 fetching 和 non-fetching 两类。Fetching AMO 返回目标对象的旧值，并可在同一次原子操作中更新目标；函数只有在旧值已从目标 PE 取回调用 PE 后才返回。Non-fetching AMO 只提交远端更新、不返回旧值，调用可能在该更新真正于目标执行前返回；需要 `nvshmem_quiet()`、相应 barrier 等完成机制来强制完成。

| 类别 | 典型接口 | 返回的信息 | 调用返回时能证明什么 |
| --- | --- | --- | --- |
| 只取值 | `atomic_fetch` | 目标旧值 | 本次原子读取结果已交付给调用者 |
| 条件交换 | `atomic_compare_swap` | 比较前旧值 | 已完成比较，并依据旧值决定是否写入 |
| 无条件交换 | `atomic_swap` | 交换前旧值 | 新值与旧值交换完成，旧值已返回 |
| fetching 算术 | `atomic_fetch_inc`、`atomic_fetch_add` | 增加前旧值 | 旧值返回与目标更新构成一次原子操作 |
| non-fetching 算术 | `atomic_inc`、`atomic_add` | 无 | 操作已发起；不能仅由返回推出远端已更新 |
| fetching 位运算 | `atomic_fetch_and/or/xor` | 运算前旧值 | 返回旧值并完成相应原子更新 |
| non-fetching 位运算 | `atomic_and/or/xor` | 无 | 操作已发起；强制远端完成需后续完成机制 |
| 原子写 | `atomic_set` | 无 | 原子写已发起；作为 non-fetching 更新处理 |

这里的“blocking”不能理解为远端应用线程参与或整个系统停下。Fetching AMO 必须等待自己的返回值，因此调用者可立即依据旧值分支；它不表示目标 PE 的 kernel 被同步，也不表示其他更早的 PUT 都已完成。Non-fetching AMO 返回得更早，却给应用留下 outstanding update。B02 已经说明，`quiet` 是调用 PE 的本地、非集合完成点，不会通知目标 PE，更不能证明目标业务逻辑已经消费更新。

Memory Ordering 表还说明，AMO 处于 fence/quiet 可影响的操作集合内，但 `fence` 只建立同一目标 PE 的交付顺序，不完成操作；`quiet` 才强制调用 PE 先前相关操作完成。若多个 CUDA threads 发起 AMO，再由一个线程调用 fence 或 quiet，必须先用相应 CUDA 同步保证其他线程的操作确实已经发起。

## 四、返回旧值为什么能分配唯一 ticket

设远端计数器初值为 $T_0$，有 $m$ 个参与者各执行一次 `atomic_fetch_add(counter, 1, owner_pe)`。对同一位置和 datatype，这些操作可以按某个顺序 $A_0,A_1,\ldots,A_{m-1}$ 推理。第 $i$ 个操作读取旧值 $T_0+i$，再把计数器更新为 $T_0+i+1$，所以返回值集合为：

$$
\{T_0,T_0+1,\ldots,T_0+m-1\}.
$$

集合中没有重复值，因此每个参与者取得唯一 ticket。这个证明不需要知道哪个 PE 对应哪个 $i$，也不要求网络请求按 host 发起时间排序；它只依赖同一对象上的原子排他性、所有参与者使用相同 datatype、计数器在区间内不回绕，以及没有其他未纳入模型的写入方式。

```cpp
// 教学片段：device 端执行，未编译、未运行。
size_t ticket =
    nvshmem_size_atomic_fetch_add(remote_tail, 1, queue_owner_pe);
```

变量名 `remote_tail` 容易误导。传入 AMO 的仍是调用 PE 上生成的 local symmetric address，`queue_owner_pe` 才指定实际更新哪一个 PE 的对应对象。它与 B01 的 `<symmetric address, target PE>` 寻址规则完全一致。

## 五、distributed counter：先明确要的是最终值还是每次旧值

若每个参与者只想贡献增量，且稍后由一个阶段统一读取总和，可以使用 non-fetching `atomic_add`，避免每次都把旧值返回发起 PE。但每次调用返回不证明远端计数器已经更新；在读取最终结果或进入依赖该结果的阶段前，必须用 quiet/barrier 等与实际执行域匹配的完成和同步机制。

若每个参与者需要知道“我是第几个”，就必须使用 fetching 形式。fetch-add 的返回值本身就是唯一编号，而在 fetch-add 之后再做一次 atomic fetch，不具有相同含义：两次操作之间可能已有其他参与者更新计数器，后一次 fetch 看到的是更晚状态。

计数器还必须定义数值语义。使用有符号类型时要考虑溢出的语言与接口边界；使用无符号类型虽有模运算，也不意味着 ticket 回绕后仍能唯一标识在途请求。如果旧 ticket 尚未退出系统，新 ticket 已回绕到相同数值，就产生代际歧义。最简单的工程约束是证明一次作业不会回绕；若必须长期运行，则需要 epoch、足够宽的序号和经过证明的模序比较。

计数器的最终值也不等于“成功完成的任务数”。如果 producer 在 fetch-add 后崩溃或永久停顿，计数器已经增加但对应任务未发布；如果失败重试再次增加，又可能重复计数。协议必须定义 reservation、publication 和 completion 分别由什么状态表示，不能让一个整数同时承担全部含义。

## 六、ticket allocator 只完成预留，不完成发布

多生产者共享队列常先用 fetch-add 在远端 `tail` 上预留 ticket，再把 ticket 映射到槽位：

$$
\text{slot}(t)=t\bmod K,
$$

其中 $K$ 是队列容量。原子操作只证明两个生产者不会得到同一个 $t$。它没有证明 $\text{slot}(t)$ 已空闲：当 $t-\text{head}\ge K$ 时，新 ticket 会映射到仍被消费者或旧生产者占用的槽，直接写入会覆盖在途数据。

预留和发布之间还存在 publication hole。producer A 取得 ticket 5 后被暂停，producer B 取得 ticket 6 并先写完。如果系统只把全局 `published_tail` 更新为 7，消费者可能错误地认为 ticket 5 也已准备好。严格有序消费者必须知道每个 ticket 是否单独发布，常见做法是每槽维护 sequence/ready；能够乱序消费的系统也需要独立 descriptor 证明 ticket 6 对应的 payload 已完成，不能从“更大的 ticket 已出现”推出所有更小 ticket 都已发布。

```text
tail fetch-add:
  producer A -> ticket 5 -> 被暂停，slot 5 尚未发布
  producer B -> ticket 6 -> payload 完成并发布

错误推论：published_tail = 7，所以 [0, 7) 全部 ready
事实：ticket 5 仍是 reservation hole
```

因此，一个有界 MPSC 队列至少要分别回答：

1. **Reservation：** 谁获得哪个唯一 ticket？
2. **Admission：** ticket 对应槽位是否已经从上一代回收？
3. **Publication：** 本代 payload 和 header 是否完整到达？
4. **Consumption：** 消费者何时取得读取权？
5. **Reclamation：** 消费结束后如何把 credit/sequence 交给下一代？

B04 已用每槽 `ready/ack` 序号证明这些阶段。B05 的 fetch-add 只补充多生产者 reservation；它不能删除 B04 的容量与所有权条件。

## 七、从唯一 ticket 到有界队列的必要不变量

设 ticket $t$ 对应槽 $j=t\bmod K$。一个可证明的有界协议通常为每槽保存 generation-aware sequence，而不是一个会反复清零的布尔 ready。生产者在写槽前必须观察到“槽 $j$ 已授权给 generation $t$”；写完 payload 后发布“generation $t$ ready”；消费者只在观察到相同 generation 后读取，并在消费完成后把槽授权给 $t+K$。

可用下面三条不变量检查设计：

- **唯一预留：** 对任意两个成功预留 $t_a,t_b$，有 $t_a\ne t_b$。
- **不覆盖：** producer 写 ticket $t$ 前，ticket $t-K$ 的 consumer 已完成所有读。
- **不提前消费：** consumer 读 ticket $t$ 前，producer 已完成该 ticket 的 payload 发布。

fetch-add 可以直接证明第一条，不能单独证明后两条。第二条需要 ack、head/credit 或每槽 free sequence；第三条需要 payload 与 ready 之间的 ordering，并由 consumer 的 wait/test 建立观察条件。若设计文档只写“atomic tail + PUT”，至少缺少两个安全性证明。

全局 `head` 也不能不加分析地由多个消费者 fetch-add。消费者先预留 ticket 再发现该 ticket 尚未发布，可能长时间占住处理顺序；一个慢消费者还可能阻塞对应槽的回收。多消费者队列需要定义 work claiming、空槽处理、乱序完成和 reclamation，复杂度通常高于多生产者预留。

## 八、compare-swap 表达条件状态转换

`atomic_compare_swap(dest, cond, value, pe)` 原子地读取旧值；若旧值等于 `cond`，则写入 `value`，无论成功与否都返回比较前的旧值。调用者通过 `old == cond` 判断是否赢得转换。它适合实现“只有状态仍是 EXPECTED 才更新”的状态机，而不只是自旋锁。

```cpp
// 教学片段：竞争一个 UNCLAIMED -> owner_id 状态转换。
size_t old = nvshmem_size_atomic_compare_swap(
    owner_word, UNCLAIMED, my_owner_id, target_pe);
bool won = (old == UNCLAIMED);
```

CAS loop 必须在每次失败后重新基于返回值决定下一步，且要说明哪些状态允许重试。若无条件持续重试，热点 PE 上会产生大量远端往返与争用；如果持有者依赖等待者占用的 GPU 资源才能释放，还可能形成前进性死锁。指数退避、分片或层次化仲裁可以降低流量，却不能替代安全性证明。

远端锁尤其容易被误用。CAS 获得一个 lock word 只证明当前调用者把该字从 unlocked 改为 owner，不自动使另一个 payload 成为一致快照。进入临界区前如何观察前一持有者的写入、释放前如何完成本次 payload、unlock 如何通知下一等待者，都需要明确的 ordering/completion 规则。若资源保护涉及大块数据，基于 owner/ticket 的消息协议往往比跨节点细粒度自旋更容易证明和优化。

## 九、swap、set 和位运算各自适合什么状态

`atomic_swap` 无条件写入新值并返回旧值，适合所有权令牌交换或取走一个单字状态；但“拿走指针”式协议仍须保证该数值在所有 PE 上具有有效含义，不能交换进程私有裸指针。若交换的是 descriptor index 或 symmetric heap offset，需要另外验证代际与对象生命周期。

`atomic_set` 是 non-fetching 原子写。它能避免与同 datatype AMO/wait-test 组合时的撕裂，却不返回被覆盖值，也可能在目标执行前返回。若覆盖旧状态本身需要条件，就应使用 compare-swap；若调用者必须确认本次 set 已完成，要增加 quiet 或适用的 barrier。把 set 当成“天然 release store”是没有依据的，payload 发布仍要遵循 NVSHMEM ordering 契约。

AND、OR、XOR 适合位图、feature flag 或到达集合。例如每个参与者用 atomic OR 设置一个互不重叠 bit，最终位图能表示哪些贡献已经到达。但单个机器字能表示的参与者数量有限；重复 OR 同一 bit 是幂等的，却无法统计重复到达；不同 bit 的设置也不能证明对应 payload 已经发布，除非在每个 producer 的 payload 和 OR 之间建立正确顺序。

## 十、Atomic 与 signal 不是可以任意混用的同义接口

AMO 针对一般受支持 symmetric scalar，提供 fetch、条件交换、算术和位运算。Signal operation 针对 `uint64_t` signal object，并为 put-with-signal 提供“先交付对应 payload，再原子更新 signal”的组合契约；远端 PE 可用 signal wait/test 或 signal fetch 观察。两者都包含原子更新，但对象类型、允许的并发访问组合和数据发布语义不同。

如果需求是“给远端计数器分配唯一 ticket”，fetch-add 合适；如果需求是“把一段 payload 写到远端，然后让消费者知道这段 payload 已交付”，put-with-signal 通常表达得更直接。先 PUT payload、再 atomic set 一个 ready word 也可能构成协议，但必须自己证明 PUT 与 AMO 的顺序、完成和目标观察，不能因为 ready 更新是 atomic 就省略 fence/quiet。

signal object 应只按 signaling 契约更新。B04 已说明，不要让同一 `sig_addr` 一会儿由 put-with-signal 更新、一会儿被普通 CUDA store 清零。类似地，一般 AMO control word 也应保持单一访问纪律。把不同 API 对同一地址“混着用”不仅难以推理，还可能超出 NVSHMEM 对并发冲突的例外范围。

## 十一、Work queue、allocator 与 scheduler 的边界

一个分布式 bump allocator 可以用 fetch-add 从远端 offset 计数器预留不重叠区间。若请求大小为 $n_i$，第 $i$ 个调用返回旧 offset $o_i$，则它获得半开区间 $[o_i,o_i+n_i)$。只要加法不溢出且所有成功区间之和不超过 pool capacity，这些区间互不重叠。

```cpp
// 只做区间预留；未处理越界、回收和失败回滚。
size_t offset = nvshmem_size_atomic_fetch_add(
    pool_offset, aligned_bytes, allocator_pe);
bool in_range = offset <= pool_bytes &&
                aligned_bytes <= pool_bytes - offset;
```

这个片段故意显示 allocator 的缺口：fetch-add 已经推进 offset，之后才发现越界时，预留不能自动撤销；简单 bump allocator 也没有 free、碎片合并和崩溃恢复。若必须拒绝越界请求，可以用 CAS loop 从旧 offset 计算候选新值，仅在不超过容量时提交，但仍要处理竞争失败重试、对齐、整数溢出和 ABA。

Work scheduler 也不能只有一个 atomic next index。对于只读、固定任务数组，fetch-add 分配索引可能已经足够，因为任务描述在调度开始前全部发布，数组长度固定，索引超过 $N$ 即停止。但对于动态产生任务的 work queue，还要区分已预留 tail、已发布 tail 和已完成 head；任务又可能生成新任务，终止检测必须证明没有在途 publication。把 `next >= N` 当成全局完成条件只适用于静态任务集。

## 十二、ABA、回绕与失败重试

CAS 只比较当前字值，不知道该值经历过多少次变化。若一个状态从 A 变成 B 又回到 A，迟到的参与者可能认为“它仍是我之前看到的 A”，这就是 ABA 问题。远端 freelist 若只在 control word 中保存节点 index，节点被弹出、复用、再回到同一 index 时尤其危险。

常见缓解方式是把 index 与 version/epoch 一起编码进一个可原子比较的字，每次状态变化递增 version。是否能在单个受支持 datatype 中容纳两者取决于值域；若拆成两个字，就失去了单次 CAS 的不可分割性。版本本身最终也会回绕，因此仍须证明在任何旧观察可能存活期间不会复用相同 `<version,index>`。

失败重试还涉及副作用幂等性。CAS 失败前如果已经写 payload，重试可能留下无人引用的数据；fetch-add 成功后若 producer 重启，重复执行会获得新 ticket。协议应把“可安全重试的预留动作”和“不可重复的外部副作用”分开，并为 abandoned reservation 设计清理或超时策略。NVSHMEM AMO 本身不提供进程故障容错。

## 十三、原子正确不等于可扩展

所有 PE 对一个远端 counter 执行 AMO，会形成热点。原子排他性要求这些更新在该位置上串行化，即使网络和 GPU 可以并行处理其他工作，热点字的吞吐仍受目标路径和争用限制。Fetching AMO 还需要把旧值返回调用者，通常比不需要返回值的贡献操作具有更长依赖链；实际差异必须测量，不能仅凭接口名称给出固定倍数。

常见扩展方法包括按 PE、warp、CTA 或节点分片计数，在本地或节点内先聚合，再较低频率地更新全局状态；批量一次 fetch-add $b$，为本地分配连续 ticket 区间；或用层次化队列减少跨节点争用。这些方法会引入新的权衡：批量 ticket 在持有者停顿时扩大 reservation hole，分片计数需要额外汇总，局部队列可能造成负载不均。

测量 AMO 时应至少记录 datatype、fetching/non-fetching、目标分布、同一地址或多地址、参与 PE/warp 数、节点内或跨节点路径、transport、消息依赖和完成点。只测 kernel launch 到返回，可能漏掉 non-fetching AMO 的 quiet 成本；每次都 quiet 又可能掩盖批处理潜力。E01 的性能课程应分别报告 issue rate、完成延迟和带完成点的端到端吞吐。

## 十四、不要提前把 NVSHMEM AMO 等同于 mlx5 RDMA atomic

在某些 InfiniBand/IBGDA 路径中，NVSHMEM atomic 可能利用 NIC 支持的原子操作；在其他拓扑或 transport 中，也可能使用 GPU P2P atomic、代理、UCX 或其他实现。API 契约规定程序可观察的原子性和完成边界，不规定所有实现必须生成同一种 WQE、使用同一 QP，或由同一处理器推进。

因此，从 `nvshmem_size_atomic_fetch_add` 直接画箭头到 “mlx5 fetch-and-add WQE” 只能作为待验证假设。D04 必须固定 NVSHMEM commit，追踪 device API、transport dispatch、地址/key 解析、请求编码、completion 和 fallback 条件，再判断具体路径。3.7.2 已提示不同平台对 atomic 的 transport 依赖，这正说明“API 可用”与“由哪种硬件执行”是两个问题。

## 十五、常见错误及其诊断方式

1. **用 fetch-add 代替完整队列。** 检查是否只有唯一 ticket，而没有容量准入、per-slot publication 和 reclamation。
2. **把 non-fetching AMO 返回当成远端完成。** 在读取最终状态或依赖该更新前，检查是否有匹配执行域的 quiet/barrier。
3. **认为 atomic ready 自动发布 payload。** 检查 payload PUT 与 ready AMO 之间是否有规范支持的 ordering，以及 consumer 是否通过 wait/test 观察。
4. **混用 atomic 与普通 store。** 同一控制字若由 AMO 和普通 CUDA/RMA 写并发访问，重新检查冲突规则；最安全的设计是规定唯一更新方式。
5. **CAS 成功后就认为获得了整段数据的一致快照。** CAS 只转换一个字，其他字段仍需要自己的发布与读取协议。
6. **忽略 ticket 回绕和 ABA。** 检查在途生命周期内是否可能复用相同数值，是否需要 epoch/version。
7. **用远端自旋锁保护可能阻塞的 GPU 工作。** 画出资源等待图，确认持有者能得到执行资源并最终释放。
8. **假设 host AMO 对所有远端 transport 可用。** 按 3.7.2 支持范围核对发起位置、连接类型、构建选项和实际 transport。
9. **只测 AMO 发起、不测完成。** 对 non-fetching 操作把 quiet 或 barrier 成本纳入端到端测量。

## 十六、掌握标准与后续路线

完成本课后，读者应能对同一 control word 列出所有访问方式和 datatype，判断它们是否处于 NVSHMEM 允许的原子并发范围；能区分 fetching AMO 的返回值完成与 non-fetching AMO 的 outstanding update；能证明 fetch-add 返回 ticket 的唯一性，并明确证明所依赖的无回绕和无混合写入条件。

对于一个队列设计，读者还应能分别指出 reservation、admission、publication、consumption 和 reclamation 状态，解释 publication hole 为什么使“最大已预留 ticket”不能充当“连续已发布前缀”，并用至少三个不变量证明不重复分配、不覆盖和不提前消费。如果只能说“atomic 保证线程安全”，却不能指出 atomic 保护的地址、返回值含义和 payload 顺序，仍未达到本课目标。

下一课 B06“Team 与集合通信”尚未创建，因此本文不建立失效链接。B06 将把关注点从一个远端 control word 扩展到一组 PE 的 matching、数据布局和 collective 并发。之后 C00 解释 AMO 第一次执行前 runtime 如何准备 heap、transport 和 device state；D04 再在固定源码版本中验证 atomic、signal 与 ordering 的 IBGDA 实现。

## 主要来源

- NVIDIA, [Atomic Memory Operations](https://docs.nvidia.com/nvshmem/api/latest/gen/api/amo.html)：fetching/non-fetching 分类、返回边界、各类 AMO 与 transport 支持提示。
- NVIDIA, [Using NVSHMEM](https://docs.nvidia.com/nvshmem/api/latest/using.html)：并发冲突、AMO 与 wait/test 例外、PGAS 寻址和数据一致性。
- NVIDIA, [Memory Ordering](https://docs.nvidia.com/nvshmem/api/latest/gen/api/ordering.html)：fence、quiet 对 blocking/nonblocking AMO 的排序与完成范围。
- NVIDIA, [Signaling Operations](https://docs.nvidia.com/nvshmem/api/latest/gen/api/signal.html)：signal object 原子性、SET/ADD 与 put-with-signal 的 payload 交付保证。
- NVIDIA, [Point-To-Point Synchronization](https://docs.nvidia.com/nvshmem/api/latest/gen/api/sync.html)：wait/test 对本地 symmetric object 的观察规则。
- NVIDIA, [NVSHMEM 3.7.2 Release Notes](https://docs.nvidia.com/nvshmem/release-notes-install-guide/release-notes/release-3720.html)：当前发布周期、PCIe P2P 系统的 atomic transport 限制及其他平台边界。

