### 9.7.15 Parallel Synchronization and Communication Instructions

#### English Original

These instructions are:

- `bar{.cta}`, `barrier{.cta}`
- `bar.warp.sync`
- `barrier.cluster`
- `membar`
- `atom`
- `red`
- `red.async`
- `multimem.red.async`
- `vote`
- `match.sync`
- `activemask`
- `redux.sync`
- `griddepcontrol`
- `elect.sync`
- `mbarrier.init`
- `mbarrier.inval`
- `mbarrier.arrive`
- `mbarrier.arrive_drop`
- `mbarrier.test_wait`
- `mbarrier.try_wait`
- `mbarrier.pending_count`
- `mbarrier.check_layout`
- `cp.async.mbarrier.arrive`
- `tensormap.cp_fenceproxy`
- `clusterlaunchcontrol.try_cancel`
- `clusterlaunchcontrol.query_cancel`

#### 中文翻译

本节列出 PTX 的并行同步与通信指令，包括 CTA、warp、cluster barrier，内存栅栏、原子与归约、warp 投票，以及 mbarrier 等。上方指令名称保持原文；本次附件仅包含以下 9.7.15.1–9.7.15.3 的详细内容。

#### 9.7.15.1 Parallel Synchronization and Communication Instructions: `bar`, `barrier`

##### English Original

bar, bar.cta, barrier, barrier.cta

Barrier synchronization.

**Syntax**

```ptx
barrier{.cta}.sync{.aligned}      a{, b};
barrier{.cta}.arrive{.aligned}    a, b;

barrier{.cta}.red.popc{.aligned}.u32  d, a{, b}, {!}c;
barrier{.cta}.red.op{.aligned}.pred   p, a{, b}, {!}c;

bar{.cta}.sync      a{, b};
bar{.cta}.arrive    a, b;

bar{.cta}.red.popc.u32  d, a{, b}, {!}c;
bar{.cta}.red.op.pred   p, a{, b}, {!}c;

.op = { .and, .or };
```

**Description**

Performs barrier synchronization and communication within a CTA. Each CTA instance has sixteen barriers numbered 0..15.

barrier{.cta} instructions can be used by the threads within the CTA for synchronization and communication.

Operands a, b, and d have type .u32; operands p and c are predicates. Source operand a specifies a logical barrier resource as an immediate constant or register with value 0 through 15. Operand b specifies the number of threads participating in the barrier. If no thread count is specified, all threads in the CTA participate in the barrier. When specifying a thread count, the value must be a multiple of the warp size. Note that a non-zero thread count is required for barrier{.cta}.arrive.

Depending on operand b, either specified number of threads (in multiple of warp size) or all threads in the CTA participate in barrier{.cta} instruction. The barrier{.cta} instructions signal the arrival of the executing threads at the named barrier.

barrier{.cta} instruction causes executing thread to wait for all non-exited threads from its warp and marks warps’ arrival at barrier. In addition to signaling its arrival at the barrier, the barrier{.cta}.red and barrier{.cta}.sync instructions causes executing thread to wait for non-exited threads of all other warps participating in the barrier to arrive. barrier{.cta}.arrive does not cause executing thread to wait for threads of other participating warps.

When a barrier completes, the waiting threads are restarted without delay, and the barrier is reinitialized so that it can be immediately reused.

The barrier{.cta}.sync or barrier{.cta}.red or barrier{.cta}.arrive instruction guarantees that when the barrier completes, prior memory accesses requested by this thread are performed relative to all threads participating in the barrier. The barrier{.cta}.sync and barrier{.cta}.red instruction further guarantees that no new memory access is requested by this thread before the barrier completes.

A memory read (e.g., by ld or atom) has been performed when the value read has been transmitted from memory and cannot be modified by another thread participating in the barrier. A memory write (e.g., by st, red or atom) has been performed when the value written has become visible to other threads participating in the barrier, that is, when the previous value can no longer be read.

barrier{.cta}.red performs a reduction operation across threads. The c predicate (or its complement) from all threads in the CTA are combined using the specified reduction operator. Once the barrier count is reached, the final value is written to the destination register in all threads waiting at the barrier.

The reduction operations for barrier{.cta}.red are population-count (.popc), all-threads-True (.and), and any-thread-True (.or). The result of .popc is the number of threads with a True predicate, while .and and .or indicate if all the threads had a True predicate or if any of the threads had a True predicate.

Instruction barrier{.cta} has optional .aligned modifier. When specified, it indicates that all threads in CTA will execute the same barrier{.cta} instruction. In conditionally executed code, an aligned barrier{.cta} instruction should only be used if it is known that all threads in CTA evaluate the condition identically, otherwise behavior is undefined.

Different warps may execute different forms of the barrier{.cta} instruction using the same barrier name and thread count. One example mixes barrier{.cta}.sync and barrier{.cta}.arrive to implement producer/consumer models. The producer threads execute barrier{.cta}.arrive to announce their arrival at the barrier and continue execution without delay to produce the next value, while the consumer threads execute the barrier{.cta}.sync to wait for a resource to be produced. The roles are then reversed, using a different barrier, where the producer threads execute a barrier{.cta}.sync to wait for a resource to consumed, while the consumer threads announce that the resource has been consumed with barrier{.cta}.arrive. Care must be taken to keep a warp from executing more barrier{.cta} instructions than intended (barrier{.cta}.arrive followed by any other barrier{.cta} instruction to the same barrier) prior to the reset of the barrier. barrier{.cta}.red should not be intermixed with barrier{.cta}.sync or barrier{.cta}.arrive using the same active barrier. Execution in this case is unpredictable.

The optional .cta qualifier simply indicates CTA-level applicability of the barrier and it doesn’t change the semantics of the instruction.

bar{.cta}.sync is equivalent to barrier{.cta}.sync.aligned. bar{.cta}.arrive is equivalent to barrier{.cta}.arrive.aligned. bar{.cta}.red is equivalent to barrier{.cta}.red.aligned.

**Note**

For .target sm_6x or below,

barrier{.cta} instruction without .aligned modifier is equivalent to .aligned variant and has the same restrictions as of .aligned variant.

All threads in warp (except for those have exited) must execute barrier{.cta} instruction in convergence.

**PTX ISA Notes**

bar.sync without a thread count introduced in PTX ISA version 1.0.

Register operands, thread count, and bar.{arrive,red} introduced in PTX ISA version 2.0.

barrier instruction introduced in PTX ISA version 6.0.

.cta qualifier introduced in PTX ISA version 7.8.

**Target ISA Notes**

Register operands, thread count, and bar{.cta}.{arrive,red} require sm_20 or higher.

Only bar{.cta}.sync with an immediate barrier number is supported for sm_1x targets.

barrier{.cta} instruction requires sm_30 or higher.

**Examples**

```ptx
// Use bar.sync to arrive at a pre-computed barrier number and
// wait for all threads in CTA to also arrive:
    st.shared [r0],r1;  // write my result to shared memory
    bar.cta.sync  1;    // arrive, wait for others to arrive
    ld.shared r2,[r3];  // use shared results from other threads

// Use bar.sync to arrive at a pre-computed barrier number and
// wait for fixed number of cooperating threads to arrive:
    #define CNT1 (8*12) // Number of cooperating threads

    st.shared [r0],r1;     // write my result to shared memory
    bar.cta.sync  1, CNT1; // arrive, wait for others to arrive
    ld.shared r2,[r3];     // use shared results from other threads

// Use bar.red.and to compare results across the entire CTA:
    setp.eq.u32 p,r1,r2;         // p is True if r1==r2
    bar.cta.red.and.pred r3,1,p; // r3=AND(p) forall threads in CTA

// Use bar.red.popc to compute the size of a group of threads
// that have a specific condition True:
    setp.eq.u32 p,r1,r2;         // p is True if r1==r2
    bar.cta.red.popc.u32 r3,1,p; // r3=SUM(p) forall threads in CTA

// Examples of barrier.cta.sync
    st.shared         [r0],r1;
    barrier.cta.sync  0;
    ld.shared         r1, [r0];

/* Producer/consumer model. The producer deposits a value in
 * shared memory, signals that it is complete but does not wait
 * using bar.arrive, and begins fetching more data from memory.
 * Once the data returns from memory, the producer must wait
 * until the consumer signals that it has read the value from
 * the shared memory location. In the meantime, a consumer
 * thread waits until the data is stored by the producer, reads
 * it, and then signals that it is done (without waiting).
 */
    // Producer code places produced value in shared memory.
    st.shared   [r0],r1;
    bar.arrive  0,64;
    ld.global   r1,[r2];
    bar.sync    1,64;
    ...

    // Consumer code, reads value from shared memory
    bar.sync   0,64;
    ld.shared  r1,[r0];
    bar.arrive 1,64;
    ...
```

##### 中文翻译

`bar` / `barrier` 在同一 CTA 内提供 barrier 同步和通信；每个 CTA 有编号 0–15 的 16 个逻辑 barrier。`barrier{.cta}` 可由 CTA 内 thread 使用。`a`、`b`、`d` 是 `.u32`，`p`、`c` 是 predicate；`a` 以立即数或 register 指定 barrier 编号，取值 0–15。`b` 指定参与 thread 数，省略时为整个 CTA；显式指定时必须是 warp size 的倍数，且 `barrier{.cta}.arrive` 要求非零计数。

执行 `barrier{.cta}` 时，当前 thread 会等待本 warp 内尚未退出的 thread，并标记该 warp 到达 barrier。`.sync` 和 `.red` 还要等待其他参与 warp 到达；`.arrive` 则不等待其他 warp。参与 thread 数由 `b` 决定。barrier 完成后，等待中的 thread 立即继续，barrier 自动重新初始化，可再次使用。

`.sync`、`.red` 或 `.arrive` 保证：barrier 完成时，当前 thread 此前请求的内存访问，相对于所有参与 thread 都已执行。`.sync` 与 `.red` 另保证在 barrier 完成前，当前 thread 不会请求新的内存访问。对 read 而言，“已执行”指读出的值已经从内存传出，不能再被其他参与 thread 改写；对 write 而言，写入值已经对其他参与 thread 可见，先前值不能再被读到。

`.red` 对参与 thread 的 `c` predicate（或其取反）进行归约，并在计数满足时把最终结果写入所有等待 thread 的目标 register。`.popc` 统计 `True` 的个数；`.and` 判断是否全部为 `True`；`.or` 判断是否至少一个为 `True`。

可选 `.aligned` 表示 CTA 内所有 thread 都会执行同一条 `barrier{.cta}` 指令。若在条件分支中使用，只有确定所有 CTA thread 对条件的求值相同时才安全，否则行为未定义。不同 warp 可以用同一 barrier 名称和计数执行不同形式，例如生产者用 `.arrive` 通知并继续工作，消费者用 `.sync` 等待；反向交接可使用另一个 barrier。但必须避免同一 warp 在 barrier reset 前对同一 barrier 多执行了指令；不能在同一个 active barrier 中混用 `.red` 与 `.sync` / `.arrive`，否则执行不可预测。

`.cta` 仅明示 barrier 的 CTA 作用层级，不改变语义。`bar{.cta}.sync`、`.arrive`、`.red` 分别等价于对应 `barrier{.cta}` 形式加 `.aligned`。对于 `sm_6x` 或更早目标，不带 `.aligned` 的 `barrier{.cta}` 也等价于 aligned 形式，并要求 warp 中尚未退出的 thread 收敛执行。原文保留了 PTX 引入版本、目标 ISA 的支持条件和全部示例。

##### 重点解读

`arrive` 与 `sync` 的核心区别是是否等待其他 warp；它们都不是可随意重复的“计数通知”。生产者/消费者示例使用两个不同 barrier 表示两个方向的交接，避免在同一轮 barrier 尚未 reset 时再次到达。`.aligned` 约束应按 CTA 一致执行理解，而非仅 warp 一致。

#### 9.7.15.2 Parallel Synchronization and Communication Instructions: `bar.warp.sync`

##### English Original

bar.warp.sync

Barrier synchronization for threads in a warp.

**Syntax**

```ptx
bar.warp.sync      membermask;
```

**Description**

bar.warp.sync will cause executing thread to wait until all threads corresponding to membermask have executed a bar.warp.sync with the same membermask value before resuming execution.

Operand membermask specifies a 32-bit integer which is a mask indicating threads participating in barrier where the bit position corresponds to thread’s laneid.

The behavior of bar.warp.sync is undefined if the executing thread is not in the membermask.

bar.warp.sync also guarantee memory ordering among threads participating in barrier. Thus, threads within warp that wish to communicate via memory can store to memory, execute bar.warp.sync, and then safely read values stored by other threads in warp.

**Note**

For .target sm_6x or below, all threads in membermask must execute the same bar.warp.sync instruction in convergence, and only threads belonging to some membermask can be active when the bar.warp.sync instruction is executed. Otherwise, the behavior is undefined.

**PTX ISA Notes**

Introduced in PTX ISA version 6.0.

**Target ISA Notes**

Requires sm_30 or higher.

**Examples**

```ptx
st.shared.u32 [r0],r1;         // write my result to shared memory
bar.warp.sync  0xffffffff;     // arrive, wait for others to arrive
ld.shared.u32 r2,[r3];         // read results written by other threads
```

##### 中文翻译

`bar.warp.sync` 是 warp 内参与 thread 的 barrier。执行 thread 会等待 `membermask` 指定的所有 lane 都执行带相同 `membermask` 值的 `bar.warp.sync`，之后才能继续。`membermask` 是 32-bit 整数掩码，bit 位置对应 lane ID；执行该指令的 thread 若不在 mask 内，行为未定义。

该指令还保证参与 thread 之间的内存排序：warp 内通信可以先写内存、执行 `bar.warp.sync`、再安全读取其他参与 thread 写入的值。对于 `sm_6x` 或更早目标，mask 中所有 thread 必须收敛执行同一条指令，而且执行时只能有属于某个 membermask 的 thread 活跃，否则行为未定义。该指令在 PTX ISA 6.0 引入，目标 ISA 要求 `sm_30` 或更高。

##### 重点解读

`membermask` 是参与集合也是同步契约：调用者自己必须在 mask 中，参与 lane 必须以相同 mask 到达。只把它当作“当前活跃 lane 的快照”而未保证实际参与集合一致，会有未定义行为风险。

#### 9.7.15.3 Parallel Synchronization and Communication Instructions: `barrier.cluster`

##### English Original

barrier.cluster

Barrier synchronization within a cluster.

**Syntax**

```ptx
barrier.cluster.arrive{.sem}{.aligned};
barrier.cluster.wait{.acquire}{.aligned};

.sem = {.release, .relaxed}
```

**Description**

Performs barrier synchronization and communication within a cluster.

barrier.cluster instructions can be used by the threads within the cluster for synchronization and communication.

barrier.cluster.arrive instruction marks warps’ arrival at barrier without causing executing thread to wait for threads of other participating warps.

barrier.cluster.wait instruction causes the executing thread to wait for all non-exited threads of the cluster to perform barrier.cluster.arrive.

In addition, barrier.cluster instructions cause the executing thread to wait for all non-exited threads from its warp.

When all non-exited threads in the cluster have executed barrier.cluster.arrive, the barrier completes and is automatically reinitialized. After using barrier.cluster.wait to detect completion of the barrier, a thread may immediately arrive at the barrier once again. Each thread must arrive at the barrier only once before the barrier completes.

The barrier.cluster.wait instruction guarantees that when it completes the execution, memory accesses (except asynchronous operations) requested, in program order, prior to the preceding barrier.cluster.arrive by all threads in the cluster are complete and visible to the executing thread.

There is no memory ordering and visibility guarantee for memory accesses requested by the executing thread, in program order, after barrier.cluster.arrive and prior to barrier.cluster.wait.

The optional .relaxed qualifier on barrier.cluster.arrive specifies that there are no memory ordering and visibility guarantees provided for the memory accesses performed prior to barrier.cluster.arrive.

The optional .sem and .acquire qualifiers on instructions barrier.cluster.arrive and barrier.cluster.wait specify the memory synchronization as described in the Memory Consistency Model. If the optional .sem qualifier is absent for barrier.cluster.arrive, .release is assumed by default. If the optional .acquire qualifier is absent for barrier.cluster.wait, .acquire is assumed by default.

The optional .aligned qualifier indicates that all threads in the warp must execute the same barrier.cluster instruction. In conditionally executed code, an aligned barrier.cluster instruction should only be used if it is known that all threads in the warp evaluate the condition identically, otherwise behavior is undefined.

**PTX ISA Notes**

Introduced in PTX ISA version 7.8.

Support for .acquire, .relaxed, .release qualifiers introduced in PTX ISA version 8.0.

**Target ISA Notes**

Requires sm_90 or higher.

**Examples**

```ptx
// use of arrive followed by wait
ld.shared::cluster.u32 r0, [addr];
barrier.cluster.arrive.aligned;
...
barrier.cluster.wait.aligned;
st.shared::cluster.u32 [addr], r1;

// use memory fence prior to arrive for relaxed barrier
@cta0 ld.shared::cluster.u32 r0, [addr];
fence.cluster.acq_rel;
barrier.cluster.arrive.relaxed.aligned;
...
barrier.cluster.wait.aligned;
@cta1 st.shared::cluster.u32 [addr], r1;
```

##### 中文翻译

`barrier.cluster` 在一个 cluster 内提供 barrier 同步和通信。`barrier.cluster.arrive` 标记 warp 到达，但不等待其他参与 warp；`barrier.cluster.wait` 则等待 cluster 中所有尚未退出的 thread 执行 arrive。两种指令都会等待本 warp 中尚未退出的 thread。

当 cluster 中所有尚未退出的 thread 都 arrive 后，barrier 完成并自动重新初始化。thread 用 wait 观察到本轮完成后可以再次 arrive；每个 thread 在同一轮完成前只能 arrive 一次。

默认的 `barrier.cluster.wait` 完成时，保证 cluster 所有 thread 在各自先前 `barrier.cluster.arrive` 之前按程序顺序请求的内存访问已完成、且对执行 wait 的 thread 可见；异步操作不在此保证内。对于当前 thread 在 arrive 之后、wait 之前请求的内存访问，不提供 ordering 或 visibility 保证。

在 `barrier.cluster.arrive` 上指定 `.relaxed`，表示不对 arrive 前的内存访问提供 ordering 和 visibility 保证。`.sem` 和 `.acquire` 分别控制 arrive 与 wait 的内存同步语义；省略时 arrive 默认为 `.release`，wait 默认为 `.acquire`。

可选 `.aligned` 表示同一 warp 中所有 thread 都执行相同的 `barrier.cluster` 指令。在条件分支中仅当 warp 内 thread 对条件求值相同才可使用，否则行为未定义。该指令于 PTX ISA 7.8 引入，`.acquire`、`.relaxed`、`.release` 于 8.0 引入；目标 ISA 要求 `sm_90` 或更高。

##### 重点解读

Cluster barrier 的默认 release/acquire 可同步 arrive 之前的普通内存访问，但不能替代异步 copy 自身的 completion wait；arrive 与 wait 之间发起的访问也不自动受到本轮 barrier 的保证。若选择 `.relaxed` arrive，需另行建立所需的内存顺序，原文示例用 `fence.cluster.acq_rel` 展示这一点。

#### 9.7.15.4 Parallel Synchronization and Communication Instructions: `membar` / `fence`

##### English Original

membar, fence

Enforce an ordering of memory operations.

**Syntax**

```ptx
// Thread fence:
fence{.sem}.scope;

// Thread fence:
fence.acquire.sync_restrict::shared::cluster.cluster;
fence.release.sync_restrict::shared::cta.cluster;

// Operation fence:
fence.op_restrict.release.cluster;

// Proxy fence (bi-directional):
fence.proxy.proxykind;
fence.proxy.proxykind{.sem}.scope;

// Proxy fence (uni-directional):
fence.proxy.to_proxykind::from_proxykind.release.scope;
fence.proxy.to_proxykind::from_proxykind.acquire.scope  [addr], size;
fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster;
fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;
fence.proxy.async::generic.release.sync_restrict::shared::cluster::read.cluster;
fence.proxy.to_proxykind::from_proxykind_fabric.alias.sem_fabric.sys;

// Old style membar:
membar.level;
membar.proxy.proxykind;

.sem       = { .sc, .acq_rel, .acquire, .release };
.scope     = { .cta, .cluster, .gpu, .sys };
.level     = { .cta, .gl, .sys };
.proxykind = { .alias, .async, .async.global, .async.shared::{cta, cluster} };
.sem_alias  = { .acquire, .release };
.scope_alias = { .sys };
.op_restrict = { .mbarrier_init };
.to_proxykind::from_proxykind = {.tensormap::generic};
.to_proxykind::from_proxykind_fabric = {.generic::fabric, .fabric::generic, .fabric::fabric };
.sem_fabric = { .acquire, .release };
```

**Description**

The membar instruction guarantees that prior memory accesses requested by this thread (ld, st, atom and red instructions) are performed at the specified level, before later memory operations requested by this thread following the membar instruction. The level qualifier specifies the set of threads that may observe the ordering effect of this operation.

A memory read (e.g., by ld or atom) has been performed when the value read has been transmitted from memory and cannot be modified by another thread at the indicated level. A memory write (e.g., by st, red or atom) has been performed when the value written has become visible to other threads at the specified level, that is, when the previous value can no longer be read.

The fence instruction establishes an ordering between memory accesses requested by this thread (ld, st, atom and red instructions) as described in the Memory Consistency Model. The scope qualifier specifies the set of threads that may observe the ordering effect of this operation.

fence.acq_rel is a light-weight fence that is sufficient for memory synchronization in most programs. Instances of fence.acq_rel synchronize when combined with additional memory operations as described in acquire and release patterns in the Memory Consistency Model. If the optional .sem qualifier is absent, .acq_rel is assumed by default.

fence.sc is a slower fence that can restore sequential consistency when used in sufficient places, at the cost of performance. Instances of fence.sc with sufficient scope always synchronize by forming a total order per scope, determined at runtime. This total order can be constrained further by other synchronization in the program.

Qualifiers .op_restrict and .sync_restrict restrict the class of memory operations for which the fence instruction provides the memory ordering guarantees. When .op_restrict is .mbarrier_init, the synchronizing effect of the fence only applies to the prior mbarrier.init operations executed by the same thread on mbarrier objects in .shared::cta state space. When .sync_restrict is .sync_restrict::shared::cta, .sem must be .release, and the effect of the fence only applies to operations performed on objects in .shared::cta state space. Likewise, when .sync_restrict is .sync_restrict::shared::cluster, .sem must be .acquire, and the effect of the fence only applies to operations performed on objects in .shared::cluster state space. When either .sync_restrict::shared::cta or .sync_restrict::shared::cluster is present, the .scope must be specified as .cluster.

When .sync_restrict is .sync_restrict::shared::cluster::read, .sem must be .release, and the effect of the fence only applies to read operations performed on objects in .shared::cluster state space. When this synchronization restriction is present, the .scope must be specified as .cluster.

**Caution**

fence.proxy.async::generic.release.sync_restrict::shared::cluster::read.cluster is a non-cumulative fence operation: it does not transitively establish ordering with respect to Causality Order.

The address operand addr and the operand size together specify the memory range [addr, addr+size-1] on which the ordering guarantees on the memory accesses across the proxies is to be provided. The only supported value for the size operand is 128, which must be a constant integer literal. Generic Addressing is used unconditionally, and the address specified by the operand addr must fall within the .global state space. Otherwise, the behavior is undefined.

On sm_70 and higher membar is a synonym for fence.sc1, and the membar levels cta, gl and sys are synonymous with the fence scopes cta, gpu and sys respectively.

membar.proxy and fence.proxy instructions establish an ordering between memory accesses that may happen through different proxies.

A uni-directional proxy ordering from the from-proxykind to the to-proxykind establishes ordering between a prior memory access performed via the from-proxykind and a subsequent memory access performed via the to-proxykind.

A bi-directional proxy ordering between two proxykinds establishes two uni-directional proxy orderings : one from the first proxykind to the second proxykind and the other from the second proxykind to the first proxykind.

The .proxykind qualifier indicates the bi-directional proxy ordering that is established between the memory accesses done between the generic proxy and the proxy specified by .proxykind.

Bi-directional proxy fences do not directly synchronize with other fences in the sense that fences with .release or .acquire semantics do. Instead, bi-directional proxy fences take effect within a single thread and therefore do not have a .scope qualifier. Their cross-proxy synchronizing effect composes with other forms of synchronization according to the rules of the Memory Consistency Model.

Value .alias of the .proxykind qualifier refers to memory accesses performed using virtually aliased addresses to the same memory location. Value .async of the .proxykind qualifier specifies that the memory ordering is established between the async proxy and the generic proxy. The memory ordering is limited only to operations performed on objects in the state space specified. If no state space is specified, then the memory ordering applies on all state spaces.

A .release proxy fence can form a release sequence that synchronizes with an acquire sequence that contains a .acquire proxy fence. The .to_proxykind and .from_proxykind qualifiers indicate the uni-directional proxy ordering that is established.

On sm_70 and higher, membar.proxy is a synonym for fence.proxy.

1 The semantics of fence.sc introduced with sm_70 is a superset of the semantics of membar and the two are compatible; when executing on sm_70 or later architectures, membar acquires the full semantics of fence.sc.

**PTX ISA Notes**

membar.{cta,gl} introduced in PTX ISA version 1.4.

membar.sys introduced in PTX ISA version 2.0.

fence introduced in PTX ISA version 6.0.

membar.proxy and fence.proxy introduced in PTX ISA version 7.5.

.cluster scope qualifier introduced in PTX ISA version 7.8.

.op_restrict qualifier introduced in PTX ISA version 8.0.

fence.proxy.async is introduced in PTX ISA version 8.0.

.to_proxykind::from_proxykind qualifier introduced in PTX ISA version 8.3.

.acquire and .release qualifiers for fence instruction introduced in PTX ISA version 8.6.

.sync_restrict qualifier introduced in PTX ISA version 8.6.

.to_proxykind::from_proxykind_fabric qualifier introduced in PTX ISA version 9.3.

.sync_restrict::shared::cluster::read qualifier introduced in PTX ISA version 9.4.

.acquire and .release qualifiers with .alias qualifier introduced in PTX ISA version 9.4.

**Target ISA Notes**

membar.{cta,gl} supported on all target architectures.

membar.sys requires sm_20 or higher.

fence requires sm_70 or higher.

membar.proxy requires sm_60 or higher.

fence.proxy requires sm_70 or higher.

.cluster scope qualifier requires sm_90 or higher.

.op_restrict qualifier requires sm_90 or higher.

fence.proxy.async requires sm_90 or higher.

.to_proxykind::from_proxykind qualifier requires sm_90 or higher.

.acquire and .release qualifiers for fence instruction require sm_90 or higher..

.sync_restrict qualifier requires sm_90 or higher.

.to_proxykind::from_proxykind_fabric qualifier requires sm_100 or higher.

.sync_restrict::shared::cluster::read qualifier requires sm_90 or higher.

.acquire and .release qualifiers with .alias qualifier require sm_90 or higher.

**Examples**

```ptx
membar.gl;
membar.cta;
membar.sys;
fence.sc.cta;
fence.sc.cluster;
fence.proxy.alias;
membar.proxy.alias;
fence.mbarrier_init.release.cluster;
fence.proxy.async;
fence.proxy.async.shared::cta;
fence.proxy.async.shared::cluster;
fence.proxy.async.global;

tensormap.replace.tile.global_address.global.b1024.b64   [gbl], new_addr;
fence.proxy.tensormap::generic.release.gpu;
cvta.global.u64  tmap, gbl;
fence.proxy.tensormap::generic.acquire.gpu [tmap], 128;
cp.async.bulk.tensor.1d.shared::cluster.global.tile  [addr0], [tmap, {tc0}], [mbar0];

// Acquire remote barrier state via async proxy.
barrier.cluster.wait.acquire;
fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster;

// Release local barrier state via generic proxy.
mbarrier.init [bar];
fence.mbarrier_init.release.cluster;
barrier.cluster.arrive.relaxed;

// Acquire local shared memory via generic proxy.
mbarrier.try_wait.relaxed.cluster.shared::cta.b64 complete, [addr], parity;
fence.acquire.sync_restrict::shared::cluster.cluster;

// Release local shared memory via generic proxy.
fence.release.sync_restrict::shared::cta.cluster;
mbarrier.arrive.relaxed.cluster.shared::cluster.b64 state, [bar];

fence.proxy.generic::fabric.alias.release.sys;
fence.proxy.fabric::generic.alias.acquire.sys;

// Release in producer via alias proxy.
st.weak.u32 [addr], value;
fence.proxy.alias.release.sys;
st.relaxed.sys [bar], 1;

// Acquire in consumer via alias proxy.
// when (ld.relaxed.sys [bar] == count):
fence.proxy.alias.acquire.sys;

// Release local shared memory via generic proxy (reads only).
fence.proxy.async::generic.release.sync_restrict::shared::cluster::read.cluster;
```

##### 中文翻译

`membar` 与 `fence` 用于约束内存操作顺序。`membar.level` 保证当前 thread 在它之前请求的 `ld`、`st`、`atom`、`red` 已在指定 level 执行，然后才执行它之后请求的内存操作；`level` 决定哪些 thread 可以观察到这种排序。一次 read 的“已执行”表示读值已从内存传出、不会再被该 level 的其他 thread 改写；一次 write 的“已执行”表示写值对该 level 的其他 thread 可见、旧值不能再被读到。

`fence` 根据 Memory Consistency Model 在当前 thread 请求的内存访问之间建立顺序；`.scope` 决定可直接观察该效果的 thread 集合。默认 `.sem` 是 `.acq_rel`，通常足以配合 acquire/release 模式中的其他内存操作实现同步。`.sc` 开销更高，在足够位置使用时能通过每个 scope 内运行时确定的全序恢复 sequential consistency；程序中的其他同步还可进一步约束这个全序。

`.op_restrict` 和 `.sync_restrict` 限缩 fence 所作用的操作。`.mbarrier_init` 只覆盖当前 thread 此前对 `.shared::cta` mbarrier object 的 `mbarrier.init`。`.sync_restrict::shared::cta` 要求 `.release`，仅作用于 `.shared::cta` object；`.sync_restrict::shared::cluster` 要求 `.acquire`，仅作用于 `.shared::cluster` object。两者都要求 `.cluster` scope。`.sync_restrict::shared::cluster::read` 则要求 `.release.cluster`，只覆盖 `.shared::cluster` object 上的 read。原文特别警告：`fence.proxy.async::generic.release.sync_restrict::shared::cluster::read.cluster` 是 non-cumulative fence，不通过 Causality Order 传递建立顺序。

带 `[addr], size` 的形式只对指定 `[addr, addr + size - 1]` 内存范围提供跨 proxy 顺序；`size` 唯一支持的值是立即数 `128`，地址始终采用 generic addressing，且必须落在 `.global`，否则行为未定义。对于 `sm_70` 及以上，`membar` 与 `fence.sc` 语义相容，`.cta`、`.gl`、`.sys` 分别对应 `.cta`、`.gpu`、`.sys` scope；原文脚注说明 `fence.sc` 语义是旧 `membar` 的超集。

`membar.proxy` 与 `fence.proxy` 处理不同 proxy 访问之间的顺序。单向形式从 `from-proxykind` 先前访问指向 `to-proxykind` 后续访问；双向形式包含两个相反方向的单向顺序。双向 proxy fence 在单个 thread 内生效，没有 `.scope`，也不直接像 release/acquire fence 那样彼此同步，而是按 Memory Consistency Model 与其他同步组合。`.alias` 指向同一位置的虚拟别名访问；`.async` 指 async 与 generic proxy，若附带 state space 则仅覆盖该空间，未指定时覆盖所有 state space。单向 `.release` proxy fence 可与包含 `.acquire` proxy fence 的 acquire sequence 组成同步。PTX ISA 引入版本与各目标架构要求完整保留在英文原文。

##### 重点解读

普通 `fence` 解决的是同一 proxy 中、指定 scope 内的内存顺序；`fence.proxy` 解决的是同一位置被不同 proxy 访问时的跨 proxy 顺序。两者都不是“等异步 copy 完成”的通用替代品：例如 async copy 仍要用相应的 mbarrier 或 bulk group 完成机制。`.sc`、`.acq_rel` 与 `.release`/`.acquire` 的选择影响同步强度和成本，而 `.sync_restrict` 明确限制所涵盖的访问，不能把受限 fence 当成全内存 fence。

#### 9.7.15.5 Parallel Synchronization and Communication Instructions: `atom`

##### English Original

atom

Atomic reduction operations for thread-to-thread communication.

**Syntax**

Atomic operation with scalar type:

```ptx
atom{.sem}{.scope}{.space}.op{.level::cache_hint}.type d, [a], b{, cache_policy};

atom{.sem}{.scope}{.space}.cas.type1 d, [a], b, c;

atom{.sem}{.scope}{.space}.exch{.level::cache_hint}.type2 d, [a], b {, cache_policy};

atom{.sem}{.scope}{.space}.add.noftz{.level::cache_hint}.type3   d, [a], b{, cache_policy};

.space =              { .global, .shared{::cta, ::cluster} };
.sem =                { .relaxed, .acquire, .release, .acq_rel };
.scope =              { .cta, .cluster, .gpu, .sys };

.op =                 { .and, .or, .xor,
                        .add, .inc, .dec,
                        .min, .max };
.level::cache_hint =  { .L2::cache_hint };
.type =               { .b32, .b64, .u32, .u64, .s32, .s64, .f32, .f64 };
.type1 =              { .b16, .b32, .b64, .b128 };
.type2 =              { .b32, .b64, .b128 };
.type3 =              { .f16, .f16x2, .bf16, .bf16x2, .f32 };
```

Atomic operation with vector type:

```ptx
atom{.sem}{.scope}{.global}.add{.noftz}{.level::cache_hint}.vec_32_bit.f32                  d, [a], b{, cache_policy};
atom{.sem}{.scope}{.global}.op.noftz{.level::cache_hint}.vec_16_bit.half_word_type  d, [a], b{, cache_policy};
atom{.sem}{.scope}{.global}.op.noftz{.level::cache_hint}.vec_32_bit.packed_type     d, [a], b{, cache_policy};

.sem =               { .relaxed, .acquire, .release, .acq_rel };
.scope =             { .cta, .cluster, .gpu, .sys };
.op =                { .add, .min, .max };
.half_word_type =    { .f16, .bf16 };
.packed_type =       { .f16x2, .bf16x2 };
.vec_16_bit =        { .v2, .v4, .v8 }
.vec_32_bit =        { .v2, .v4 };
.level::cache_hint = { .L2::cache_hint }
```

**Description**

Atomically loads the original value at location a into destination register d, performs a reduction operation with operand b and the value in location a, and stores the result of the specified operation at location a, overwriting the original value. For the .cas (compare-and-swap) operation, operand b is the compare value and operand c is the swap value. The operation compares the value at location a with operand b; if they are equal, it stores operand c at location a, otherwise it leaves the value at location a unchanged. Operand a specifies a location in the specified state space. If no state space is given, perform the memory accesses using Generic Addressing. atom with scalar type may be used only with .global and .shared spaces and with generic addressing, where the address points to .global or .shared space. atom with vector type may be used only with .global space and with generic addressing where the address points to .global space.

For atom with vector type, operands d and b are brace-enclosed vector expressions, size of which is equal to the size of vector qualifier.

If no sub-qualifier is specified with .shared state space, then ::cta is assumed by default.

The optional .sem qualifier specifies a memory synchronizing effect as described in the Memory Consistency Model. If the .sem qualifier is absent, .relaxed is assumed by default.

The optional .scope qualifier specifies the set of threads that can directly observe the memory synchronizing effect of this operation, as described in the Memory Consistency Model. If the .scope qualifier is absent, .gpu scope is assumed by default.

Table 41 shows the valid combinations of types and atomic operations for scalar atom instructions.

Table 41 Atom Scalar Type Operations

| Type | Integer Operations | Floating-Point Operations | Bit Operations |
|---|---|---|---|
| `.u32` | `.add`, `.inc`, `.dec`, `.min`, `.max` | Not supported | Not supported |
| `.s32` | `.add`, `.min`, `.max` | Not supported | Not supported |
| `.u64` | `.add`, `.min`, `.max` | Not supported | Not supported |
| `.s64` | `.min`, `.max` | Not supported | Not supported |
| `.f32` | Not supported | `.add{.noftz}` | Not supported |
| `.f64` | Not supported | `.add` (noftz by default) | Not supported |
| `.f16` | Not supported | `.add` (requires `.noftz`) | Not supported |
| `.f16x2` | Not supported | `.add` (requires `.noftz`) | Not supported |
| `.bf16` | Not supported | `.add` (requires `.noftz`) | Not supported |
| `.bf16x2` | Not supported | `.add` (requires `.noftz`) | Not supported |
| `.b16` | Not supported | Not supported | `.cas` |
| `.b32` | Not supported | Not supported | `.and`, `.or`, `.xor`, `.exch`, `.cas` |
| `.b64` | Not supported | Not supported | `.and`, `.or`, `.xor`, `.exch`, `.cas` |
| `.b128` | Not supported | Not supported | `.exch`, `.cas` |

For atom with vector type, the supported combinations of vector qualifier and types, and atomic operations supported on these combinations are depicted in the following table:

| Vector qualifier | `.f16` / `.bf16` | `.f16x2` / `.bf16x2` | `.f32` |
|---|---|---|---|
| `.v2` | `.add`, `.min`, `.max` (all require `.noftz`) | `.add`, `.min`, `.max` (all require `.noftz`) | `.add` |
| `.v4` | `.add`, `.min`, `.max` (all require `.noftz`) | `.add`, `.min`, `.max` (all require `.noftz`) | `.add` |
| `.v8` | `.add`, `.min`, `.max` (all require `.noftz`) | Not supported | Not Supported |

Two atomic operations (atom or red) are performed atomically with respect to each other only if each operation specifies a scope that includes the other. When this condition is not met, each operation observes the other operation being performed as if it were split into a read followed by a dependent write.

atom instruction on packed type or vector type, accesses adjacent scalar elements in memory. In such cases, the atomicity is guaranteed separately for each of the individual scalar elements; the entire atom is not guaranteed to be atomic as a single access.

For sm_6x and earlier architectures, atom operations on .shared state space do not guarantee atomicity with respect to normal store instructions to the same address. It is the programmer’s responsibility to guarantee correctness of programs that use shared memory atomic instructions, e.g., by inserting barriers between normal stores and atomic operations to a common address, or by using atom.exch to store to locations accessed by other atomic operations.

Supported addressing modes for operand a and alignment requirements are described in Addresses as Operands

Operation Descriptions:

Integer operations: .add, .inc, .dec, .min, .max

The .inc and .dec operations return a result in the range [0..b].

.inc and .dec are only supported for .u32 type.

Floating-point operations: .add, .min, .max

The .add operation rounds to nearest even.

The .min and .max operations implement the exact same floating-point instructions as the non-atomic min and max instructions, but perform them atomically.

The atom.add.f32 operation without .noftz, flushes subnormal inputs and results to sign-preserving zero on global memory but supports subnormal inputs and results and doesn’t flush them to zero on shared memory. atom.add.f16, atom.add.f16x2, atom.add.bf16 and atom.add.bf16x2 operations require the .noftz qualifier; it preserves subnormal inputs and results, and does not flush them to zero.

atom.add.f16, atom.add.f16x2, atom.add.bf16 and atom.add.bf16x2 operations require the .noftz qualifier; it preserves subnormal inputs and results, and does not flush them to zero.

Bit operations: .and, .or, .xor, .cas (compare-and-swap), and .exch (exchange)

The .cas operation takes three operands: d (destination), [a] (memory address), b (compare value), and c (swap value). It atomically compares the value at location a with operand b; if they are equal, it stores operand c at location a and returns the original value in d. If they are not equal, it leaves the value at location a unchanged and returns the original value in d.

When the optional argument cache_policy is specified, the qualifier .level::cache_hint is required. The 64-bit operand cache_policy specifies the cache eviction policy that may be used during the memory access.

The qualifier .level::cache_hint is only supported for .global state space and for generic addressing where the address points to the .global state space.

cache_policy is a hint to the cache subsystem and may not always be respected. It is treated as a performance hint only, and does not change the memory consistency behavior of the program.

**Semantics**

```text
atomic {
    d = *a;
    *a = (operation == cas) ? operation(*a, b, c)
                            : operation(*a, b);
}
where
    inc(r, s)  = (r >= s) ? 0 : r+1;
    dec(r, s)  = (r==0 || r > s)  ? s : r-1;
    exch(r, s) =  s;
    cas(r,s,t) = (r == s) ? t : r;
```

**Notes**

Simple reductions may be specified by using the bit bucket destination operand _.

**PTX ISA Notes**

32-bit atom.global introduced in PTX ISA version 1.1.

atom.shared and 64-bit atom.global.{add,cas,exch} introduced in PTX ISA 1.2.

atom.add.f32 and 64-bit atom.shared.{add,cas,exch} introduced in PTX ISA 2.0.

64-bit atom.{and,or,xor,min,max} introduced in PTX ISA 3.1.

atom.add.f64 introduced in PTX ISA 5.0.

.scope qualifier introduced in PTX ISA 5.0.

.sem qualifier introduced in PTX ISA version 6.0.

atom.add.noftz.f16x2 introduced in PTX ISA 6.2.

atom.add.noftz.f16 and atom.cas.b16 introduced in PTX ISA 6.3.

Per-element atomicity of atom.f16x2 clarified in PTX ISA version 6.3, with retrospective effect from PTX ISA version 6.2.

Support for .level::cache_hint qualifier introduced in PTX ISA version 7.4.

atom.add.noftz.bf16 and atom.add.noftz.bf16x2 introduced in PTX ISA 7.8.

Support for .cluster scope qualifier introduced in PTX ISA version 7.8.

Support for ::cta and ::cluster sub-qualifiers introduced in PTX ISA version 7.8.

Support for vector types introduced in PTX ISA version 8.1.

Support for .b128 type introduced in PTX ISA version 8.3.

Support for .sys scope with .b128 type introduced in PTX ISA version 8.4.

Qualifier .noftz with .f32 type introduced in PTX ISA version 9.4.

**Target ISA Notes**

atom.global requires sm_11 or higher.

atom.shared requires sm_12 or higher.

64-bit atom.global.{add,cas,exch} require sm_12 or higher.

64-bit atom.shared.{add,cas,exch} require sm_20 or higher.

64-bit atom.{and,or,xor,min,max} require sm_32 or higher.

atom.add.f32 requires sm_20 or higher.

atom.add.f64 requires sm_60 or higher.

.scope qualifier requires sm_60 or higher.

.sem qualifier requires sm_70 or higher.

Use of generic addressing requires sm_20 or higher.

atom.add.noftz.f16x2 requires sm_60 or higher.

atom.add.noftz.f16 and atom.cas.b16 requires sm_70 or higher.

Support for .level::cache_hint qualifier requires sm_80 or higher.

atom.add.noftz.bf16 and atom.add.noftz.bf16x2 require sm_90 or higher.

Support for .cluster scope qualifier requires sm_90 or higher.

Sub-qualifier ::cta requires sm_30 or higher.

Sub-qualifier ::cluster requires sm_90 or higher.

Support for vector types requires sm_90 or higher.

Support for .b128 type requires sm_90 or higher.

Support for .noftz with .f32 type requires sm_90 or higher.

**Examples**

```ptx
atom.global.add.s32  d,[a],1;
atom.shared::cta.max.u32  d,[x+4],0;
@p  atom.global.cas.b32  d,[p],my_val,my_new_val;
atom.global.sys.add.u32 d, [a], 1;
atom.global.acquire.sys.inc.u32 ans, [gbl], %r0;
atom.add.noftz.f16x2 d, [a], b;
atom.add.noftz.f16   hd, [ha], hb;
atom.global.cas.b16  hd, [ha], hb, hc;
atom.add.noftz.bf16   hd, [a], hb;
atom.add.noftz.bf16x2 bd, [b], bb;
atom.add.shared::cluster.noftz.f16   hd, [ha], hb;
atom.shared.b128.cas d, a, b, c; // 128-bit atom
atom.global.b128.exch d, a, b;   // 128-bit atom

atom.global.cluster.relaxed.add.u32 d, [a], 1;

createpolicy.fractional.L2::evict_last.b64 cache_policy, 0.25;
atom.global.add.L2::cache_hint.s32  d, [a], 1, cache_policy;

atom.global.v8.f16.max.noftz  {%hd0, %hd1, %hd2, %hd3, %hd4, %hd5, %hd6, %hd7}, [gbl],
                                              {%h0, %h1, %h2, %h3, %h4, %h5, %h6, %h7};
atom.global.v8.bf16.add.noftz  {%hd0, %hd1, %hd2, %hd3, %hd4, %hd5, %hd6, %hd7}, [gbl],
                                              {%h0, %h1, %h2, %h3, %h4, %h5, %h6, %h7};
atom.global.v2.f16.add.noftz  {%hd0, %hd1}, [gbl], {%h0, %h1};
atom.global.v2.bf16.add.noftz  {%hd0, %hd1}, [gbl], {%h0, %h1};
atom.global.v4.b16x2.min.noftz  {%hd0, %hd1, %hd2, %hd3}, [gbl], {%h0, %h1, %h2, %h3};
atom.global.v4.f32.add  {%f0, %f1, %f2, %f3}, [gbl], {%f0, %f1, %f2, %f3};
atom.global.v2.f16x2.min.noftz  {%bd0, %bd1}, [g], {%b0, %b1};
atom.global.v2.bf16x2.max.noftz  {%bd0, %bd1}, [g], {%b0, %b1};
atom.global.v2.f32.add  {%f0, %f1}, [g], {%f0, %f1};
atom.global.add.noftz.f32 %f0, [g], %f0;
```

##### 中文翻译

`atom` 是供 thread 间通信的原子归约/读-改-写指令：先把地址 `[a]` 的旧值原子地读入 `d`，用旧值和 `b` 执行指定运算，再把结果写回 `[a]`。`.cas` 用 `b` 作比较值、`c` 作替换值：相等才写入 `c`，否则原地址保持不变；无论哪种情况，`d` 都返回旧值。未指定 state space 时采用 Generic Addressing；scalar `atom` 只可访问 `.global` 或 `.shared`，vector `atom` 只可访问 `.global`。vector 形式的 `d` 与 `b` 是花括号向量，元素数等于 vector qualifier；`.shared` 未给子 qualifier 时默认 `::cta`。

可选 `.sem` 指定内存同步语义，省略时为 `.relaxed`；可选 `.scope` 指定能直接观察同步效果的 thread 集合，省略时为 `.gpu`。Table 41 与后续 vector 表分别列出 scalar、vector 指令的合法类型和操作组合；表格按原文完整保留，不另翻译。

两个 `atom` 或 `red` 操作，只有彼此指定的 scope 都包含对方，才相互保证原子性；否则，各自可能观察到对方被拆成一次 read 和依赖该 read 的 write。Packed 与 vector `atom` 虽访问相邻 scalar 元素，但只保证每个 scalar 元素分别原子，不保证整条向量指令作为一次整体访问原子。对 `sm_6x` 及更早架构，`.shared` 上的 `atom` 相对于同地址普通 store 也不保证原子性，程序需通过 barrier 分隔，或用 `atom.exch` 等方式保证正确性。地址模式与对齐要求见原文 Addresses as Operands。

整数运算包括 `.add`、`.inc`、`.dec`、`.min`、`.max`；后两种计数式 `.inc`、`.dec` 仅支持 `.u32`，结果位于 `[0,b]`，具体回绕规则见原文 Semantics。浮点 `.add` 使用 nearest-even 舍入，`.min` 和 `.max` 与非原子浮点 min/max 实现相同运算，但在这里以原子方式执行。未带 `.noftz` 的 `atom.add.f32` 对 global memory 上的 subnormal 输入与结果做保留符号的 flush-to-zero，而 shared memory 不做；`.f16`、`.f16x2`、`.bf16`、`.bf16x2` 的 add 则必须带 `.noftz`，保留 subnormal。原文重复说明了后者的 `.noftz` 要求，英文部分按原样保留。

位运算形式包括 `.and`、`.or`、`.xor`、`.cas` 和 `.exch`；`.cas` 的比较、条件写入及旧值返回语义见前述。指定 64-bit `cache_policy` 时必须同时带 `.level::cache_hint`，且仅能用于 `.global` 或指向 global 的 generic address。缓存驱逐策略只是可能不被采纳的性能提示，不改变 memory consistency。简单归约可把 destination 写为 bit-bucket `_`，不保留旧值。历史 PTX ISA 引入版本与各类型、qualifier 的目标架构门槛均保留在英文原文。

##### 重点解读

“原子”与“有序”是两层不同保证：默认 `.relaxed.gpu` 使受支持的读-改-写在 GPU scope 内原子，但不会自动提供 acquire/release 的通信顺序。跨 thread 用原子变量发布普通数据时，要按内存模型选择合适的 `.sem` 与足以覆盖通信双方的 `.scope`。Vector/packed 原子的保证是逐 scalar element，而非整块向量；不能拿它实现需要整体不可分割的多字段状态更新。


#### 9.7.15.6 Parallel Synchronization and Communication Instructions: `red`

##### English Original

red

Reduction operations on global and shared memory.

**Syntax**

Reduction operation with scalar type:

```ptx
red{.sem}{.scope}{.space}.op{.level::cache_hint}.type          [a], b{, cache_policy};

red{.sem}{.scope}{.space}.add.noftz{.level::cache_hint}.type1   [a], b{, cache_policy};

.space =              { .global, .shared{::cta, ::cluster} };
.sem =                { .relaxed, .release };
.scope =              { .cta, .cluster, .gpu, .sys };

.op =                 { .and, .or, .xor,
                        .add, .inc, .dec,
                        .min, .max };
.level::cache_hint =  { .L2::cache_hint };
.type =               { .b32, .b64, .u32, .u64, .s32, .s64, .f32, .f64 };
.type1 =              { .f16, .f16x2, .bf16, .bf16x2, .f32 };
```

Reduction operation with vector type:

```ptx
red{.sem}{.scope}{.global}.add{.noftz}{.level::cache_hint}.vec_32_bit.f32 [a], b{, cache_policy};
red{.sem}{.scope}{.global}.op.noftz{.level::cache_hint}. vec_16_bit.half_word_type [a], b{, cache_policy};
red{.sem}{.scope}{.global}.op.noftz{.level::cache_hint}.vec_32_bit.packed_type [a], b {, cache_policy};

.sem =                { .relaxed, .release };
.scope =              { .cta, .cluster, .gpu, .sys };
.op =                 { .add, .min, .max };
.half_word_type =     { .f16, .bf16 };
.packed_type =        { .f16x2,.bf16x2 };
.vec_16_bit =         { .v2, .v4, .v8 }
.vec_32_bit =         { .v2, .v4 };
.level::cache_hint =  { .L2::cache_hint }
```

**Description**

Performs a reduction operation with operand b and the value in location a, and stores the result of the specified operation at location a, overwriting the original value. Operand a specifies a location in the specified state space. If no state space is given, perform the memory accesses using Generic Addressing. red with scalar type may be used only with .global and .shared spaces and with generic addressing, where the address points to .global or .shared space. red with vector type may be used only with .global space and with generic addressing where the address points to .global space.

For red with vector type, operand b is brace-enclosed vector expressions, size of which is equal to the size of vector qualifier.

If no sub-qualifier is specified with .shared state space, then ::cta is assumed by default.

The optional .sem qualifier specifies a memory synchronizing effect as described in the Memory Consistency Model. If the .sem qualifier is absent, .relaxed is assumed by default.

The optional .scope qualifier specifies the set of threads that can directly observe the memory synchronizing effect of this operation, as described in the Memory Consistency Model. If the .scope qualifier is absent, .gpu scope is assumed by default.

Table 42 shows the valid combinations of types and reduction operations for scalar red instructions.

Table 42 Red Scalar Type Operations

| Type | Integer Operations | Floating-Point Operations | Bit Operations |
|---|---|---|---|
| `.u32` | `.add, .inc, .dec, .min, .max` | Not supported | Not supported |
| `.s32` | `.add, .min, .max` | Not supported | Not supported |
| `.u64` | `.add, .min, .max` | Not supported | Not supported |
| `.s64` | `.min, .max` | Not supported | Not supported |
| `.f32` | Not supported | `.add{.noftz}` | Not supported |
| `.f64` | Not supported | `.add` (noftz by default) | Not supported |
| `.f16` | Not supported | `.add` (requires `.noftz`) | Not supported |
| `.f16x2` | Not supported | `.add` (requires `.noftz`) | Not supported |
| `.bf16` | Not supported | `.add` (requires `.noftz`) | Not supported |
| `.bf16x2` | Not supported | `.add` (requires `.noftz`) | Not supported |
| `.b32` | Not supported | Not supported | `.and, .or, .xor` |
| `.b64` | Not supported | Not supported | `.and, .or, .xor` |

For red with vector type, the supported combinations of vector qualifier, types and reduction operations supported on these combinations are depicted in following table:

| Vector qualifier | `.f16` / `.bf16` | `.f16x2` / `.bf16x2` | `.f32` |
|---|---|---|---|
| `.v2` | `.add, .min, .max` (all require `.noftz`) | `.add, .min, .max` (all require `.noftz`) | `.add` |
| `.v4` | `.add, .min, .max` (all require `.noftz`) | `.add, .min, .max` (all require `.noftz`) | `.add` |
| `.v8` | `.add, .min, .max` (all require `.noftz`) | Not supported | Not Supported |

Two atomic operations (atom or red) are performed atomically with respect to each other only if each operation specifies a scope that includes the other. When this condition is not met, each operation observes the other operation being performed as if it were split into a read followed by a dependent write.

red instruction on packed type or vector type, accesses adjacent scalar elements in memory. In such case, the atomicity is guaranteed separately for each of the individual scalar elements; the entire red is not guaranteed to be atomic as a single access.

For sm_6x and earlier architectures, red operations on .shared state space do not guarantee atomicity with respect to normal store instructions to the same address. It is the programmer’s responsibility to guarantee correctness of programs that use shared memory reduction instructions, e.g., by inserting barriers between normal stores and reduction operations to a common address, or by using atom.exch to store to locations accessed by other reduction operations.

Supported addressing modes for operand a and alignment requirements are described in Addresses as Operands

**Operation Descriptions:**

Integer operations: .add, .inc, .dec, .min, .max

The .inc and .dec operations return a result in the range [0..b].

.inc and .dec are only supported for .u32 type.

Floating-point operations: .add, .min, .max

The .add operation rounds to nearest even.

The .min and .max operations implement the exact same floating-point instructions as the non-atomic min and max instructions, but perform them atomically.

The red.add.f32 operation without .noftz, flushes subnormal inputs and results to sign-preserving zero on global memory but supports subnormal inputs and results and doesn’t flush them to zero on shared memory. red.add.f16, red.add.f16x2, red.add.bf16 and red.add.bf16x2 operations require the .noftz qualifier; it preserves subnormal inputs and results, and does not flush them to zero.

red.add.f16, red.add.f16x2, red.add.bf16 and red.add.bf16x2 operations require the .noftz qualifier; it preserves subnormal inputs and results, and does not flush them to zero.

Bit operations: .and, .or, .xor

When the optional argument cache_policy is specified, the qualifier .level::cache_hint is required. The 64-bit operand cache_policy specifies the cache eviction policy that may be used during the memory access.

The qualifier .level::cache_hint is only supported for .global state space and for generic addressing where the address points to the .global state space.

cache_policy is a hint to the cache subsystem and may not always be respected. It is treated as a performance hint only, and does not change the memory consistency behavior of the program.

**Semantics**

```text
*a = operation(*a, b);

where
    inc(r, s) = (r >= s) ? 0 : r+1;
    dec(r, s) = (r==0 || r > s)  ? s : r-1;
```

**PTX ISA Notes**

Introduced in PTX ISA version 1.2.

red.add.f32 and red.shared.add.u64 introduced in PTX ISA 2.0.

64-bit red.{and,or,xor,min,max} introduced in PTX ISA 3.1.

red.add.f64 introduced in PTX ISA 5.0.

.scope qualifier introduced in PTX ISA 5.0.

.sem qualifier introduced in PTX ISA version 6.0.

red.add.noftz.f16x2 introduced in PTX ISA 6.2.

red.add.noftz.f16 introduced in PTX ISA 6.3.

Per-element atomicity of red.f16x2 clarified in PTX ISA version 6.3, with retrospective effect from PTX ISA version 6.2

Support for .level::cache_hint qualifier introduced in PTX ISA version 7.4.

red.add.noftz.bf16 and red.add.noftz.bf16x2 introduced in PTX ISA 7.8.

Support for .cluster scope qualifier introduced in PTX ISA version 7.8.

Support for ::cta and ::cluster sub-qualifiers introduced in PTX ISA version 7.8.

Support for vector types introduced in PTX ISA version 8.1.

Qualifier .noftz with .f32 type introduced in PTX ISA version 9.4.

**Target ISA Notes**

red.global requires sm_11 or higher

red.shared requires sm_12 or higher.

red.global.add.u64 requires sm_12 or higher.

red.shared.add.u64 requires sm_20 or higher.

64-bit red.{and,or,xor,min,max} require sm_32 or higher.

red.add.f32 requires sm_20 or higher.

red.add.f64 requires sm_60 or higher.

.scope qualifier requires sm_60 or higher.

.sem qualifier requires sm_70 or higher.

Use of generic addressing requires sm_20 or higher.

red.add.noftz.f16x2 requires sm_60 or higher.

red.add.noftz.f16 requires sm_70 or higher.

Support for .level::cache_hint qualifier requires sm_80 or higher.

red.add.noftz.bf16 and red.add.noftz.bf16x2 require sm_90 or higher.

Support for .cluster scope qualifier requires sm_90 or higher.

Sub-qualifier ::cta requires sm_30 or higher.

Sub-qualifier ::cluster requires sm_90 or higher.

Support for vector types requires sm_90 or higher.

Support for .noftz with .f32 type requires sm_90 or higher.

**Examples**

```ptx
red.global.add.s32  [a],1;
red.shared::cluster.max.u32  [x+4],0;
@p  red.global.and.b32  [p],my_val;
red.global.sys.add.u32 [a], 1;
red.global.acquire.sys.add.u32 [gbl], 1;
red.add.noftz.f16x2 [a], b;
red.add.noftz.bf16   [a], hb;
red.add.noftz.bf16x2 [b], bb;
red.global.cluster.relaxed.add.u32 [a], 1;
red.shared::cta.min.u32  [x+4],0;

createpolicy.fractional.L2::evict_last.b64 cache_policy, 0.25;
red.global.and.L2::cache_hint.b32 [a], 1, cache_policy;

red.global.v8.f16.add.noftz  [gbl], {%h0, %h1, %h2, %h3, %h4, %h5, %h6, %h7};
red.global.v8.bf16.min.noftz [gbl], {%h0, %h1, %h2, %h3, %h4, %h5, %h6, %h7};
red.global.v2.f16.add.noftz [gbl], {%h0, %h1};
red.global.v2.bf16.add.noftz [gbl], {%h0, %h1};
red.global.v4.f16x2.max.noftz [gbl], {%h0, %h1, %h2, %h3};
red.global.v4.f32.add  [gbl], {%f0, %f1, %f2, %f3};
red.global.v2.f16x2.max.noftz {%bd0, %bd1}, [g], {%b0, %b1};
red.global.v2.bf16x2.add.noftz {%bd0, %bd1}, [g], {%b0, %b1};
red.global.v2.f32.add  {%f0, %f1}, [g], {%f0, %f1};
red.add.noftz.f32 [addr], %f0;
```

##### 中文翻译

`red` 在内存位置 `a` 的现值与操作数 `b` 之间执行归约，并把结果写回 `a`，覆盖旧值；它不像 `atom` 那样将旧值返回寄存器。未指定 state space 时使用 Generic Addressing。标量形式仅支持指向 `.global` 或 `.shared` 的地址；向量形式仅支持 `.global`。向量 `b` 须写成花括号表达式，其元素数与 vector qualifier 一致。`.shared` 省略子限定符时默认 `::cta`。

可选的 `.sem` 决定内存同步效果，省略时为 `.relaxed`；`.scope` 决定可以直接观察这一同步效果的线程集合，省略时为 `.gpu`。标量类型和操作的合法组合见 Table 42；向量形式见随后的表格，表格内容按原文整理、不另翻译。两个 `atom` 或 `red` 操作只有在双方 scope 都覆盖对方时，才相互保证原子性；若不满足，彼此可能观察到对方仿佛被拆成 read 和依赖 read 的 write。Packed/vector `red` 访问相邻标量元素，但原子性只对各标量元素分别成立，不保证整条向量指令作为单一访问原子执行。

在 `sm_6x` 及更早架构，shared-memory `red` 相对于同地址普通 store 不保证原子性；程序需通过 barrier 分隔这些访问，或采用 `atom.exch` 等方式保证正确性。地址模式与对齐条件见原文的 Addresses as Operands。`.inc` 与 `.dec` 只支持 `.u32`，结果位于 `[0,b]`；具体回绕/重置公式保留于 Semantics。`.add` 的浮点舍入为 nearest-even；浮点 `.min`/`.max` 与对应非原子指令采用相同运算，但原子地执行。未带 `.noftz` 的 `red.add.f32` 在 global memory 对 subnormal 输入与结果做保留符号的 flush-to-zero，在 shared memory 则保留 subnormal。半精度及 `bf16` 的 add 必须带 `.noftz`，且保留 subnormal；原文对这点有重复，英文部分照录。

只有 global memory（或指向 global 的 generic address）支持 `.level::cache_hint`。提供 `cache_policy` 时须同时提供该限定符；缓存驱逐策略仅是可能不被采纳的性能提示，不改变 memory consistency。PTX 引入版本、目标架构门槛、全部示例均在英文原文中保留。

##### 重点解读

选择 `red` 的关键条件是“只需要更新内存，不需要取回旧值”；若后续计算需要旧值，应使用返回旧值的 `atom`。不过“不返回值”不等于“无同步约束”：`red` 仍有 scope 和 ordering，且只有 `.relaxed` 与 `.release`，没有 `.acquire` 形式。原文示例中的 `red.global.acquire.sys.add.u32` 与列出的合法 `.sem` 不一致；这里为忠实保留原文未改写，实际编写代码应以语法约束及编译器验证为准。

#### 9.7.15.7 Parallel Synchronization and Communication Instructions: `red.async`

##### English Original

red.async

Asynchronous reduction operation.

**Syntax**

```ptx
// Increment and Decrement reductions
red.async.sem.scope{.ss}.completion_mechanism.op.type [a], b, [mbar];

.sem  =                 { .relaxed };
.scope =                { .cluster };
.ss   =                 { .shared::cluster };
.op   =                 { .inc, .dec };
.type =                 { .u32 };
.completion_mechanism = { .mbarrier::complete_tx::bytes };


// MIN and MAX reductions
red.async.sem.scope{.ss}.completion_mechanism.op.type [a], b, [mbar];

.sem  = { .relaxed };
.scope = { .cluster };
.ss   = { .shared::cluster };
.op   = { .min, .max };
.type = { .u32, .s32 };
.completion_mechanism = { .mbarrier::complete_tx::bytes };

// Bitwise AND, OR and XOR reductions
red.async.sem.scope{.ss}.completion_mechanism.op.type [a], b, [mbar];

.sem  = { .relaxed };
.scope = { .cluster };
.ss   = { .shared::cluster };
.op   = { .and, .or, .xor };
.type = { .b32 };
.completion_mechanism = { .mbarrier::complete_tx::bytes };

// ADD reductions
red.async.sem.scope{.ss}.completion_mechanism.add.type [a], b, [mbar];

.sem  = { .relaxed };
.scope = { .cluster };
.ss   = { .shared::cluster };
.type = { .u32, .s32, .u64 };
.completion_mechanism = { .mbarrier::complete_tx::bytes };

red.async{.mmio}.sem.scope{.ss}.add.type [a], b;

.sem  = { .release };
.scope = { .gpu, .sys };
.ss   = { .global };
.type = { .u32, .s32, .u64, .s64 };
```

**Description**

red.async is a non-blocking instruction which initiates an asynchronous reduction operation specified by .op, with the operand b and the value at the destination memory location specified by operand a.

red.async is performed in the generic proxy.

**Operands**

a is a destination address, and must be either a register, or of the form register + immOff, as described in Addresses as Operands.

b is a source value, of the type indicated by qualifier .type.

mbar is an mbarrier object address.

**Qualifiers**

.mmio indicates whether this is an mmio Operation.

.sem specifies the memory ordering semantics as described in the Memory Consistency Model.

.scope specifies the set of threads with which this instruction can directly synchronize.

.ss specifies the state space of the destination operand a and the mbarrier operand mbar.

If .ss is not specified, Generic Addressing is used.

.completion_mechanism specifies the mechanism for observing the completion of the asynchronous operation.

When .completion_mechanism is .mbarrier::complete_tx::bytes: upon completion of the asynchronous operation, a complete-tx operation will be performed on the mbarrier object specified by the operand mbar, with completeCount argument equal to the amount of data stored in bytes.

When .completion_mechanism is not specified: the completion of the store synchronizes with the end of the CTA. This instruction accesses its mbarrier operand using generic-proxy.

.op specifies the reduction operation.

The .inc and .dec operations return a result in the range [0..b].

.type specifies the type of the source operand b.

**Conditions**

When .sem is .relaxed:

The reduce operation is a relaxed memory operation.

The complete-tx operation on the mbarrier has .release semantics at .cluster scope.

The shared-memory addresses of the destination operand a and the mbarrier operand mbar must meet all of the following conditions:

They belong to the same CTA.

The CTA to which they belong is different from the CTA of the executing thread, but must be within the same cluster.

Otherwise, the behavior is undefined.

.mmio must not be specified.

If .ss is specified, it must be .shared::cluster.

If .ss is not specified, generic addressing is used for operands a and mbar. If the generic addresses specified do not fall within the address window of .shared::cluster state space, the behavior is undefined.

If .completion_mechanism is specified, it must be .mbarrier::complete_tx::bytes.

If .completion_mechanism is not specified, it defaults to .mbarrier::complete_tx::bytes.

When .sem is .release:

The reduce operation is a strong memory operation with .release semantics at the scope specified by .scope.

If .mmio is specified, .scope must be .sys.

If .ss is specified, it must be .global.

If .ss is not specified, generic addressing is used for operand a. If the generic address specified does not fall within the address window of .global state space, the behavior is undefined.

.completion_mechanism must not be specified.

**PTX ISA Notes**

Introduced in PTX ISA version 8.1.

Support for .mmio qualifier, .release semantics, .global state space, and .gpu and .sys scopes introduced in PTX ISA version 8.7.

**Target ISA Notes**

Requires sm_90 or higher.

.mmio qualifier, .release semantics, .global state space, and .gpu and .sys scopes require sm_100 or higher.

**Examples**

```ptx
red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.min.u32 [addr], b, [mbar_addr];

red.async.release.sys.global.add.u32 [addr], b;
```

##### 中文翻译

`red.async` 发起非阻塞异步归约：将目标地址 `a` 中的值与 `b` 按 `.op` 运算并更新目标。该指令在 generic proxy 中执行。地址 `a` 必须是寄存器或“寄存器 + immOff”；`b` 的类型由 `.type` 指定；`mbar` 是 mbarrier 对象地址。限定符 `.mmio` 表示 MMIO 操作，`.sem` 表示内存顺序，`.scope` 表示可直接同步的线程范围，`.ss` 表示目标及 mbarrier 的 state space；省略 `.ss` 时采用 Generic Addressing。`.completion_mechanism` 指定如何观察异步操作完成。

`.mbarrier::complete_tx::bytes` 完成机制会在异步操作完成时对 `mbar` 执行 complete-tx，`completeCount` 等于写入的字节数。原文还说明，不指定完成机制时，store 的完成与 CTA 结束同步；该指令通过 generic proxy 访问其 mbarrier 操作数。`.inc`/`.dec` 的结果在 `[0,b]` 范围内。

`.relaxed.cluster.shared::cluster` 形式针对 cluster 内另一 CTA 的 shared memory，并使用 mbarrier 完成机制。归约本身是 relaxed memory operation，而对 mbarrier 的 complete-tx 具有 cluster scope 的 release 语义。目标 `a` 和 `mbar` 必须同属一个 CTA；它必须不同于执行指令的 CTA、但处于同一 cluster，否则行为未定义。该形式不能带 `.mmio`；显式 `.ss` 只能是 `.shared::cluster`，generic address 也必须落在该空间；完成机制显式给出或默认均为 `.mbarrier::complete_tx::bytes`。

`.release.{gpu,sys}.global` 形式是指定 scope 的强 release 内存操作，只支持列出的 `.add` 与整数类型；带 `.mmio` 时 scope 必须为 `.sys`。显式 `.ss` 只能是 `.global`，generic address 必须指向 global，否则行为未定义；该形式不能指定完成机制。原文保留了 PTX 引入版本、`sm_90`/新特性 `sm_100` 的架构要求和示例。

##### 重点解读

这不是一种形式加可选参数，而是两套不同的用法：cluster shared 形式用远端 CTA 的 `mbarrier` 跟踪完成；global release 形式不带 `mbar`，原文将 store 完成关联到 CTA 结束。不能把此处的 generic proxy 与 `cp.async.bulk` 的 asynchronous proxy 混同，也不能把发出 `red.async` 当作更新已经完成。

#### 9.7.15.8 Parallel Synchronization and Communication Instructions: `multimem.red.async`

##### English Original

multimem.red.async

Perform asynchronous reduction with release ordering on the multimem address.

**Syntax**

```ptx
multimem.red.async.sem.scope{.ss}.op.type [a], b;

.sem   = { .release };
.scope = { .gpu, .sys };
.ss    = { .global };
.op    = { .add };
.type  = { .u32, .s32, .u64 };
```

**Description**

multimem.red.async is a non-blocking instruction which initiates an asynchronous reduction operation specified by .op, with operand b and the value at memory locations residing on each GPU’s memory referred to by the destination multimem address operand a.

Address operand a must be a multimem address. Otherwise, the behavior is undefined.

**Operands**

a is a destination address, and must be either a register, or of the form register + immOff, as described in Addresses as Operands.

b is a source value, of the type indicated by qualifier .type.

**Qualifiers**

The mandatory .sem modifier specifies the memory ordering semantics as described in the Memory Consistency Model.

The mandatory .scope modifier specifies the set of threads with which this instruction can directly synchronize. The reduce operation is a strong memory operation with .release semantics at the scope specified by .scope.

.ss specifies the state space of the destination operand a. If .ss is specified, it must be .global. If .ss is not specified, generic addressing is used for operand a. If the generic address specified does not fall within the address window of .global state space, the behavior is undefined.

.type specifies the type of the source operand b.

**PTX ISA Notes**

Introduced in PTX ISA version 9.3.

**Target ISA Notes**

Requires architecture sm_100 or higher.

**Examples**

```ptx
// Asynchronous add reduction, GPU scope, explicit .global, 32-bit unsigned.
multimem.red.async.release.gpu.global.add.u32 [mm_addr], src_u32;

// System scope, generic addressing.
multimem.red.async.release.sys.add.s32 [mm_addr], src_s32;

// 64-bit unsigned add reduction.
multimem.red.async.release.gpu.global.add.u64 [mm_addr], src_u64;
```

##### 中文翻译

`multimem.red.async` 对 multimem 地址所指向的、每个 GPU 内存中的目标位置发起非阻塞异步归约，语义为指定 scope 的强 `.release`。这里仅支持 `.add`，数据类型为 `.u32`、`.s32`、`.u64`；`.sem` 和 `.scope` 都是必需的，scope 可以是 `.gpu` 或 `.sys`。地址 `a` 必须是 multimem 地址，并符合“寄存器”或“寄存器 + immOff”形式，否则行为未定义；`b` 的类型由 `.type` 指定。显式 `.ss` 只能是 `.global`；若省略则使用 Generic Addressing，但地址仍须落在 global state space，否则行为未定义。原文注明该指令始于 PTX ISA 9.3，要求 `sm_100` 或更高架构，并给出三个示例。

##### 重点解读

`red.async.release.*.global` 针对普通 global 地址；`multimem.red.async` 则要求 multimem 地址，将归约施加于该地址所引用的每个 GPU 的内存位置。两者不可仅凭名字相似而互换，尤其要检查地址类型、支持的数据类型和目标架构。


#### 9.7.15.10 Parallel Synchronization and Communication Instructions: `vote.sync`

##### English Original

vote.sync

Vote across thread group.

**Syntax**

```ptx
vote.sync.mode.pred  d, {!}a, membermask;
vote.sync.ballot.b32 d, {!}a, membermask;  // 'ballot' form, returns bitmask

.mode = { .all, .any, .uni };
```

**Description**

vote.sync will cause executing thread to wait until all non-exited threads corresponding to membermask have executed vote.sync with the same qualifiers and same membermask value before resuming execution.

Operand membermask specifies a 32-bit integer which is a mask indicating threads participating in this instruction where the bit position corresponds to thread’s laneid. Operand a is a predicate register.

In the mode form, vote.sync performs a reduction of the source predicate across all non-exited threads in membermask. The destination operand d is a predicate register and its value is the same across all threads in membermask.

The reduction modes are:

**.all**
True if source predicate is True for all non-exited threads in membermask. Negate the source predicate to compute .none.

**.any**
True if source predicate is True for some thread in membermask. Negate the source predicate to compute .not_all.

**.uni**
True if source predicate has the same value in all non-exited threads in membermask. Negating the source predicate also computes .uni.

In the ballot form, the destination operand d is a .b32 register. In this form, vote.sync.ballot.b32 simply copies the predicate from each thread in membermask into the corresponding bit position of destination register d, where the bit position corresponds to the thread’s lane id.

A thread not specified in membermask will contribute a 0 for its entry in vote.sync.ballot.b32.

The behavior of vote.sync is undefined if the executing thread is not in the membermask.

**Note**

For .target sm_6x or below, all threads in membermask must execute the same vote.sync instruction in convergence, and only threads belonging to some membermask can be active when the vote.sync instruction is executed. Otherwise, the behavior is undefined.

**PTX ISA Notes**

Introduced in PTX ISA version 6.0.

**Target ISA Notes**

Requires sm_30 or higher.

**Examples**

```ptx
vote.sync.all.pred    p,q,0xffffffff;
vote.sync.ballot.b32  r1,p,0xffffffff;  // get 'ballot' across warp
```

##### 中文翻译

`vote.sync` 在由 `membermask` 指定的线程组内投票。执行线程必须等待 mask 中所有尚未退出的线程执行具有相同限定符、相同 `membermask` 值的 `vote.sync`，然后才继续。32-bit `membermask` 中每一位对应一个线程的 `laneid`，源操作数 `a` 是 predicate register。执行线程不在 `membermask` 中时，行为未定义。

mode 形式对参与且尚未退出的线程的 predicate 做归约，结果 `d` 是 predicate register，在 mask 内各线程中相同。`.all` 表示所有参与线程的源 predicate 均为 true；对源 predicate 取反可计算 `.none`。`.any` 表示至少一个参与线程为 true；取反可计算 `.not_all`。`.uni` 表示所有参与线程的源 predicate 值相同；对所有源 predicate 取反后得到的仍是 `.uni`。

`vote.sync.ballot.b32` 的目标 `d` 为 `.b32` 寄存器：它将每个参与线程的 predicate 复制到结果中与其 `laneid` 对应的位。不在 `membermask` 中的线程，对应位为 0。原文的 Note 指出，在 `.target sm_6x` 或更早目标上，mask 中所有线程须以 convergence 方式执行同一条 `vote.sync`，且执行该指令时只有属于某个 membermask 的线程可以活跃，否则行为未定义。PTX ISA 引入版本、目标架构要求和示例保留在英文原文。

##### 重点解读

`membermask` 既定义参与集合，也构成同步契约：不能只让部分被标记的、尚未退出的 lane 执行该指令。mode 形式得到整个参与组共享的真假值，ballot 形式得到逐 lane 的位图；它们不能互相替代。

#### 9.7.15.11 Parallel Synchronization and Communication Instructions: `match.sync`

##### English Original

match.sync

Broadcast and compare a value across threads in warp.

**Syntax**

```ptx
match.any.sync.type  d, a, membermask;
match.all.sync.type  d[|p], a, membermask;

.type = { .b32, .b64 };
```

**Description**

match.sync will cause executing thread to wait until all non-exited threads from membermask have executed match.sync with the same qualifiers and same membermask value before resuming execution.

Operand membermask specifies a 32-bit integer which is a mask indicating threads participating in this instruction where the bit position corresponds to thread’s laneid.

match.sync performs broadcast and compare of operand a across all non-exited threads in membermask and sets destination d and optional predicate p based on mode.

Operand a has instruction type and d has .b32 type.

Destination d is a 32-bit mask where bit position in mask corresponds to thread’s laneid.

The matching operation modes are:

**.all**
d is set to mask corresponding to non-exited threads in membermask if all non-exited threads in membermask have same value of operand a; otherwise d is set to 0. Optionally predicate p is set to true if all non-exited threads in membermask have same value of operand a; otherwise p is set to false. The sink symbol ‘_’ may be used in place of any one of the destination operands.

**.any**
d is set to mask of non-exited threads in membermask that have same value of operand a.

The behavior of match.sync is undefined if the executing thread is not in the membermask.

**PTX ISA Notes**

Introduced in PTX ISA version 6.0.

**Target ISA Notes**

Requires sm_70 or higher.

**Release Notes**

Note that match.sync applies to threads in a single warp, not across an entire CTA.

**Examples**

```ptx
match.any.sync.b32    d, a, 0xffffffff;
match.all.sync.b64    d|p, a, mask;
```

##### 中文翻译

`match.sync` 在一个 warp 的参与线程之间广播并比较 `a`。执行线程会等待 `membermask` 中所有尚未退出的线程执行限定符和 mask 值相同的 `match.sync` 后继续。32-bit `membermask` 用 `laneid` 对应的位指示参与线程；执行线程若不在 mask 内，行为未定义。`a` 具有指令指定的 `.b32` 或 `.b64` 类型，目标 `d` 是 32-bit mask，其位位置对应线程的 `laneid`；`.all` 形式还可输出 predicate `p`。

`.all` 模式下，若 mask 内所有尚未退出的线程的 `a` 相同，`d` 为这些线程组成的 mask，否则为 0；可选 `p` 相应为 true 或 false。任一目标操作数可用 sink symbol `_` 代替。`.any` 模式下，`d` 为 mask 内与当前线程 `a` 值相同的尚未退出线程的位图。原文的 Release Notes 强调，这条指令只作用于一个 warp 内的线程，不能跨整个 CTA。版本、架构要求与示例见英文原文。

##### 重点解读

`match.any.sync` 会按当前线程的值给出“同值线程组”；不同值组的线程可得到不同的 `d`。`match.all.sync` 则判断整个参与集合是否同值，并可同时取得 predicate。两者都受 `membermask` 的完整参与要求约束。

#### 9.7.15.12 Parallel Synchronization and Communication Instructions: `activemask`

##### English Original

activemask

Queries the active threads within a warp.

**Syntax**

```ptx
activemask.b32 d;
```

**Description**

activemask queries predicated-on active threads from the executing warp and sets the destination d with 32-bit integer mask where bit position in the mask corresponds to the thread’s laneid.

Destination d is a 32-bit destination register.

An active thread will contribute 1 for its entry in the result and exited or inactive or predicated-off thread will contribute 0 for its entry in the result.

**PTX ISA Notes**

Introduced in PTX ISA version 6.2.

**Target ISA Notes**

Requires sm_30 or higher.

**Examples**

```ptx
activemask.b32  %r1;
```

##### 中文翻译

`activemask.b32` 查询执行线程所在 warp 当前处于 predicated-on active 状态的线程，并返回 32-bit 位图 `d`；每一位对应一个 `laneid`。活跃线程贡献 1，已退出、不活跃或被 predicate 关闭的线程贡献 0。目标 `d` 是 32-bit 寄存器。PTX ISA 引入版本、目标架构要求及示例均见英文原文。

##### 重点解读

`activemask` 是执行到该指令时的活跃线程快照，不是固定的“完整 warp”常量。若要把结果用作后续 `vote.sync`、`match.sync` 等指令的 `membermask`，仍须保证 mask 内线程实际执行匹配的同步指令；单独读取位图不会建立这种参与保证。

#### 9.7.15.13 Parallel Synchronization and Communication Instructions: `redux.sync`

##### English Original

redux.sync

Perform reduction operation on the data from each predicated active thread in the thread group.

**Syntax**

```ptx
redux.sync.op.type dst, src, membermask;
.op   = {.add, .min, .max}
.type = {.u32, .s32}

redux.sync.op.b32 dst, src, membermask;
.op   = {.and, .or, .xor}

redux.sync.op{.abs.}{.NaN}.f32 dst, src, membermask;
.op   = { .min, .max }
```

**Description**

redux.sync will cause the executing thread to wait until all non-exited threads corresponding to membermask have executed redux.sync with the same qualifiers and same membermask value before resuming execution.

Operand membermask specifies a 32-bit integer which is a mask indicating threads participating in this instruction where the bit position corresponds to thread’s laneid.

redux.sync performs a reduction operation .op of the 32 bit source register src across all non-exited threads in the membermask. The result of the reduction operation is written to the 32 bit destination register dst.

Reduction operation can be one of the bitwise operation in .and, .or, .xor or arithmetic operation in .add, .min , .max.

For the .add operation result is truncated to 32 bits.

For .f32 instruction type, if the input value is 0.0 then +0.0 > -0.0.

If .abs qualifier is specified, then the absolute value of the input is considered for the reduction operation.

If the .NaN qualifier is specified, then the result of the reduction operation is canonical NaN if the input to the reduction operation from any participating thread is NaN.

In the absence of .NaN qualifier, only non-NaN values are considered for the reduction operation and the result will be canonical NaN when all inputs are NaNs.

The behavior of redux.sync is undefined if the executing thread is not in the membermask.

**PTX ISA Notes**

Introduced in PTX ISA version 7.0.

Support for .f32 type is introduced in PTX ISA version 8.6.

Support for .abs and .NaN qualifiers is introduced in PTX ISA version 8.6.

**Target ISA Notes**

Requires sm_80 or higher.

.f32 type requires sm_100a and is supported on sm_100f from PTX ISA version 8.8.

Qualifiers .abs and .NaN require sm_100a and are supported on sm_100f or higher in the same family from PTX ISA version 8.8.

**Release Notes**

Note that redux.sync applies to threads in a single warp, not across an entire CTA.

**Examples**

```ptx
.reg .b32 dst, src, init, mask;
redux.sync.add.s32 dst, src, 0xff;
redux.sync.xor.b32 dst, src, mask;

redux.sync.min.abs.NaN.f32 dst, src, mask;
```

##### 中文翻译

`redux.sync` 在 `membermask` 指定的线程组中，对各线程的 32-bit 源寄存器 `src` 执行归约，将结果写到 32-bit `dst`。执行线程须等待 mask 内所有尚未退出的线程执行限定符和 mask 值相同的 `redux.sync` 后继续。32-bit `membermask` 的每一位对应一个 `laneid`；执行线程不在 mask 中时行为未定义。支持 `.u32`/`.s32` 的 `.add`、`.min`、`.max`，`.b32` 的 `.and`、`.or`、`.xor`，以及 `.f32` 的 `.min`、`.max`；`.add` 结果截断到 32 bit。

对 `.f32`，比较输入 0.0 时 `+0.0 > -0.0`。`.abs` 使归约按输入的绝对值进行。`.NaN` 表示只要任一参与输入为 NaN，结果就为 canonical NaN；省略该限定符时，仅非 NaN 值参与运算，只有所有输入均为 NaN 时结果才是 canonical NaN。原文的 Release Notes 指出，`redux.sync` 仅作用于单个 warp，不跨整个 CTA。整数/位运算形式要求 `sm_80` 或更高；`.f32`、`.abs`、`.NaN` 另有 `sm_100a` 与同系列 `sm_100f` 的要求，具体版本说明见英文原文。

##### 重点解读

这里的归约发生在 warp 线程的寄存器值之间，与上一组写回内存位置的 `red` 不同。尤其对浮点 min/max，NaN 传播与否由 `.NaN` 决定，不能只看 `.min` 或 `.max`。

#### 9.7.15.14 Parallel Synchronization and Communication Instructions: `griddepcontrol`

##### English Original

griddepcontrol

Control execution of dependent grids.

**Syntax**

```ptx
griddepcontrol.action;

.action   = { .launch_dependents, .wait }
```

**Description**

The griddepcontrol instruction allows the dependent grids and prerequisite grids as defined by the runtime, to control execution in the following way:

.launch_dependents modifier signals that specific dependents the runtime system designated to react to this instruction can be scheduled as soon as all other CTAs in the grid issue the same instruction or have completed. The dependent may launch before the completion of the current grid. There is no guarantee that the dependent will launch before the completion of the current grid. Repeated invocations of this instruction by threads in the current CTA will have no additional side effects past that of the first invocation. A release fence preceeding a griddepcontrol.launch_dependents in program-order synchronizes with the start of a dependent grid if either both grids have the same memory synchronization domain and the fence scope is gpu, or if the fence scope is sys.

.wait modifier causes the executing thread to wait until all prerequisite grids in flight have completed and all the memory operations from the prerequisite grids are performed and made visible to the current grid.

**Note**

If the prerequisite grid is using griddepcontrol.launch_dependents, then the dependent grid must use griddepcontrol.wait to ensure correct functional execution.

**PTX ISA Notes**

Introduced in PTX ISA version 7.8.

**Target ISA Notes**

Requires sm_90 or higher.

**Examples**

```ptx
griddepcontrol.launch_dependents;
griddepcontrol.wait;
```

##### 中文翻译

`griddepcontrol` 让运行时定义的 prerequisite grid（前置 grid）和 dependent grid（依赖 grid）控制执行。`.launch_dependents` 表示：一旦当前 grid 的所有其他 CTA 都发出同一指令或已经完成，运行时指定响应此信号的依赖 grid 就可以被调度。依赖 grid 可能在当前 grid 完成之前启动，但没有必须提前启动的保证；当前 CTA 内重复调用不会产生超出首次调用的额外副作用。

若 program order 中先于 `griddepcontrol.launch_dependents` 的 release fence 满足条件，它将与 dependent grid 的启动同步：两个 grid 使用相同 memory synchronization domain 且 fence scope 为 `.gpu`，或者 fence scope 为 `.sys`。`.wait` 使执行线程等待所有仍在运行的 prerequisite grid 完成，以及它们的所有内存操作执行完毕并对当前 grid 可见。原文的 Note 指出，如果前置 grid 使用 `.launch_dependents`，依赖 grid 必须使用 `.wait` 才能保证功能正确性。PTX ISA 与目标架构要求、示例见英文原文。

##### 重点解读

`.launch_dependents` 允许“前置 grid 未结束，依赖 grid 已启动”的重叠，但只发出 launch 信号并不意味着依赖数据已经可安全读取。数据交接需要满足 release fence 的作用域与 domain 条件，并由依赖 grid 的 `.wait` 等待前置 grid 及其内存操作完成。

#### 9.7.15.15 Parallel Synchronization and Communication Instructions: `elect.sync`

##### English Original

elect.sync

Elect a leader thread from a set of threads.

**Syntax**

```ptx
elect.sync d|p, membermask;
```

**Description**

elect.sync elects one predicated active leader thread from among a set of threads specified by membermask. laneid of the elected thread is returned in the 32-bit destination operand d. The sink symbol ‘_’ can be used for destination operand d. The predicate destination p is set to True for the leader thread, and False for all other threads.

Operand membermask specifies a 32-bit integer indicating the set of threads from which a leader is to be elected. The behavior is undefined if the executing thread is not in membermask.

Election of a leader thread happens deterministically, i.e. the same leader thread is elected for the same membermask every time.

The mandatory .sync qualifier indicates that elect causes the executing thread to wait until all threads in the membermask execute the elect instruction before resuming execution.

**PTX ISA Notes**

Introduced in PTX ISA version 8.0.

**Target ISA Notes**

Requires sm_90 or higher.

**Examples**

```ptx
elect.sync    %r0|%p0, 0xffffffff;
```

##### 中文翻译

`elect.sync` 从 `membermask` 指定、当前 predicated active 的线程中选出一个 leader，并在 32-bit 目标 `d` 中返回被选中线程的 `laneid`；若不需要该编号，`d` 可用 sink symbol `_`。目标 predicate `p` 在 leader 线程为 true，在其他线程为 false。`membermask` 是指示候选线程的 32-bit 整数；执行线程不在 mask 中时行为未定义。

同一个 `membermask` 每次会确定性地选出同一个 leader。必需的 `.sync` 限定符表示执行线程须等到 mask 内所有线程都执行该 `elect` 指令后才继续。PTX ISA 引入版本、`sm_90` 架构要求及示例保留于英文原文。

##### 重点解读

“确定性”只保证同一 mask 重复选举的一致性，不等于文档承诺选中最低 `laneid` 或其他特定 lane。实际使用时可让 `p` 控制单线程工作，让 `d` 提供被选中 leader 的编号，同时遵守整个 mask 的同步参与约束。

#### 9.7.15.16 Parallel Synchronization and Communication Instructions: `mbarrier`

##### English Original

mbarrier is a barrier created in shared memory that supports :

- Synchronizing any subset of threads within a CTA
- One-way synchronization of threads across CTAs of a cluster. As noted in [mbarrier support with shared memory](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-smem), threads can perform only arrive operations but not *_wait on an mbarrier located in shared::cluster space.
- Waiting for completion of asynchronous memory operations initiated by a thread and making them visible to other threads.

An mbarrier object is an opaque object in memory which can be initialized and invalidated using :

- mbarrier.init
- mbarrier.inval

Operations supported on mbarrier objects are :

- mbarrier.expect_tx
- mbarrier.complete_tx
- mbarrier.arrive
- mbarrier.arrive_drop
- mbarrier.test_wait
- mbarrier.try_wait
- mbarrier.pending_count
- mbarrier.check_layout
- cp.async.mbarrier.arrive

Performing any mbarrier operation except mbarrier.init on an uninitialized mbarrier object results in undefined behavior. Performing any non-mbarrier or mbarrier.init operations on an initialized mbarrier object results in undefined behavior.

Unlike bar{.cta}/barrier{.cta} instructions which can access a limited number of barriers per CTA, mbarrier objects are user defined and are only limited by the total shared memory size available.

mbarrier operations enable threads to perform useful work after the arrival at the mbarrier and before waiting for the mbarrier to complete.

###### 9.7.15.16.1 Size and alignment of mbarrier object

An mbarrier object is an opaque object with the following type and alignment requirements :

| Type | Alignment (bytes) | Memory space |
|---|---:|---|
| `.b64` | 8 | `.shared` |

###### 9.7.15.16.2 Layouts of the mbarrier object

An opaque mbarrier object supports two different layouts:

- .layout::v0
- .layout::v1

The exact layout of the mbarrier object can be specified at the time of creation. The asynchronous operations that can be tracked by an mbarrier object depend on the layout. Refer to [Contents of the mbarrier object](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-contents) for more details.

###### 9.7.15.16.3 Contents of the mbarrier object

An opaque mbarrier object irrespective of the layout keeps track of the following information :

- Current primary and conditional phases of the mbarrier object
- Count of pending arrivals for the current phase of the mbarrier object
- Count of expected arrivals for the next phase of the mbarrier object
- Count of pending asynchronous memory operations (or transactions) tracked by the current phase of the mbarrier object. This is also referred to as tx-count.

However, an mbarrier object with .layout::v1 additionally keeps track of the following information:

- Payload report corresponding to each primary phase

An mbarrier object progresses through a sequence of phases where each phase is defined by threads performing an expected number of [arrive-on](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on) operations.

The valid range of each of the counts differs based on the mbarrier layout. Refer [Table 43](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mbarrier-counts) for more details.

Table 43 Mbarrier counts for different layouts

| Layout | Count name | Minimum value | Maximum value |
|---|---|---:|---:|
| `.layout::v0` | Expected arrival count | 1 | 2^20 - 1 |
| `.layout::v0` | Pending arrival count | 0 | 2^20 - 1 |
| `.layout::v0` | tx-count | -(2^20 - 1) | 2^20 - 1 |
| `.layout::v1` | Expected arrival count | 1 | 2^9 - 1 |
| `.layout::v1` | Pending arrival count | 0 | 2^9 - 1 |
| `.layout::v1` | tx-count | -(2^20 - 1) | 2^20 - 1 |

###### 9.7.15.16.4 Lifecycle of the mbarrier object

The mbarrier object must be initialized prior to use.

An mbarrier object is used to synchronize threads, track asynchronous memory operations.

An mbarrier object may be used to perform a sequence of such synchronizations.

An mbarrier object must be invalidated to repurpose its memory for any purpose, including repurposing it for another mbarrier object.

###### 9.7.15.16.5 Phases of the mbarrier object

**9.7.15.16.5.1 Primary phase**

The primary phase of an mbarrier object is the number of times the mbarrier object has been used to synchronize threads and [asynchronous](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#program-order-async-operations) operations. In each primary phase {0, 1, 2, …}, threads perform in program order :

- [arrive-on](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on) operations to complete the current primary phase and
- test_wait / try_wait operations to check for the completion of the current primary phase.

An mbarrier object is automatically reinitialized upon completion of the current primary phase for immediate use in the next phase. The current primary phase is incomplete and all prior primary phases are complete.

For each primary phase of the mbarrier object, at least one test_wait or try_wait operation must be performed which returns True for waitComplete before an [arrive-on](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on) operation in the subsequent primary phase.

**9.7.15.16.5.2 Conditional phase**

The primary phase of an mbarrier object advances upon completion of arrivals, and transaction-count updates of asynchronous operations tracked by the mbarrier. The conditional phase advances if the primary phase advances, and no asynchronous operation tracked by the mbarrier prevents it via a [report-on](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-report-on) operation.

Mbarrier objects with .layout::v0 do not support tracking asynchronous operations that could prevent the conditional phase from advancing via [report-on](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-report-on), resulting in both phases advancing in unison. That is, either both phase advance, or no phase advances.

Mbarrier objects with .layout::v1 support tracking asynchronous operations that can prevent the conditional phase from advancing via a [report-on](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-report-on) operation. If the primary phase advances, but the conditional phase does not advance, the payload report is non-zero and may contain more information. If the conditional phase advances, the payload report is zero.

An mbarrier object is automatically reinitialized upon completion of the current primary phase for immediate use in the next phase. Refer [Phase Completion of the mbarrier object](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-phase-completion) for more details on conditional phase completion.

###### 9.7.15.16.6 Tracking asynchronous operations by the mbarrier object

Starting with the Hopper architecture (sm_9x), mbarrier object supports a new count, called tx-count, which is used for tracking the completion of asynchronous memory operations or transactions. tx-count tracks the number of asynchronous transactions, in units specified by the asynchronous memory operation, that are outstanding and yet to be complete.

The tx-count of an mbarrier object must be set to the total amount of asynchronous memory operations, in units as specified by the asynchronous operations, to be tracked by the current phase. Upon completion of each of the asynchronous operations, the [complete-tx](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx-operation) operation will be performed on the mbarrier object and thus progress the mbarrier towards the completion of the current phase.

**9.7.15.16.6.1 expect-tx operation**

The expect-tx operation, with an expectCount argument, increases the tx-count of an mbarrier object by the value specified by expectCount. This sets the current phase of the mbarrier object to expect and track the completion of additional asynchronous transactions.

**9.7.15.16.6.2 complete-tx operation**

The complete-tx operation, with an completeCount argument, on an mbarrier object consists of the following:

**mbarrier signaling**

Signals the completion of asynchronous transactions that were tracked by the current phase. As a result of this, tx-count is decremented by completeCount.

**mbarrier potentially completing the current phase**

If the current phase has been completed then the mbarrier transitions to the next phase. Refer to [Phase Completion of the mbarrier object](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-phase-completion) for details on phase completion requirements and phase transition process.

###### 9.7.15.16.7 Tracking successful completion of an operation using the mbarrier object

An mbarrier object with layout .layout::v1 can track asynchronous operations that prevent the conditional phase from advancing and provide additional information via the payload report on completion.

The mbarrier.test_wait and mbarrier.try_wait instructions with .phase_type::primary allow observing primary phase completion and the payload report. The payload report consists of a report predicate (reportPredicate) and a report value (reportValue). For the conditional phase to advance, the entire payload report must be zero. If the predicate i.e. reportPredicate is zero, the value reportValue is guaranteed to be zero and therefore the conditional phase has advanced.

The mbarrier.test_wait and mbarrier.try_wait instructions with .phase_type::conditional allow observing the conditional phase advance which means the payload report was zero.

##### 中文翻译

`mbarrier` 是放在 shared memory 中的用户定义 barrier，可同步 CTA 内任意线程子集、支持 cluster 内跨 CTA 的单向同步，还可等待线程发起的异步内存操作完成并使结果对其他线程可见。跨 CTA 使用时，对位于 `shared::cluster` 的 mbarrier 只能执行 arrive，不能执行 `*_wait`。对象先由 `mbarrier.init` 初始化，最终由 `mbarrier.inval` 失效；可用的操作包括 `expect_tx`、`complete_tx`、`arrive`、`arrive_drop`、`test_wait`、`try_wait`、`pending_count`、`check_layout` 和 `cp.async.mbarrier.arrive`。对未初始化对象执行 `init` 以外的 mbarrier 操作，或对已初始化对象执行非 mbarrier 操作、再次执行 `init`，均为未定义行为。

与每个 CTA 可用数量受限的 `bar{.cta}`/`barrier{.cta}` 不同，mbarrier 对象由用户定义，数量仅受 shared memory 容量约束。线程 arrive 后可以继续做有用工作，到需要结果时再等待，而不必在 arrive 处立即停下。

**9.7.15.16.1 大小与对齐。** mbarrier 是 opaque object，类型为 `.b64`，必须按 8 字节对齐，位于 `.shared`；原文规格表保留英文，不另翻译。

**9.7.15.16.2 布局。** 对象支持 `.layout::v0` 和 `.layout::v1`，创建时可指定。不同布局能跟踪的异步操作不同，具体内容见原文所链的 Contents of the mbarrier object。

**9.7.15.16.3 对象内容。** 两种布局都跟踪当前 primary phase 与 conditional phase、当前 phase 的 pending arrival count、下一 phase 的 expected arrival count，以及当前 phase 尚未完成的异步内存操作计数 `tx-count`。`.layout::v1` 还会为每个 primary phase 保存 payload report。phase 由预期数量的 `arrive-on` 操作推进。Table 43 保留了两种布局的各项计数范围：v0 的 expected/pending arrival 上限为 `2^20 - 1`，v1 为 `2^9 - 1`；两者的 `tx-count` 范围均为 `[-(2^20 - 1), 2^20 - 1]`。粘贴文本的上标排版丢失，这些指数已依照官方表格还原。

**9.7.15.16.4 生命周期。** 对象必须先初始化，之后可反复用于线程同步和异步内存操作跟踪。若要把它占用的内存改作任何其他用途，包括创建另一个 mbarrier 对象，必须先使原对象失效。

**9.7.15.16.5 Primary 与 conditional phase。** Primary phase 记录 mbarrier 已用于同步线程及异步操作的次数。在每个 phase 中，线程按程序顺序先通过 `arrive-on` 使当前 phase 走向完成，再用 `test_wait` 或 `try_wait` 检查完成。当前 primary phase 完成后，对象会自动为下一 phase 重新初始化；当前 phase 尚未完成，而之前的 primary phase 已完成。关键使用约束是：进入后续 primary phase 执行 `arrive-on` 前，对前一个 primary phase 至少要有一次返回 `waitComplete=True` 的 `test_wait` 或 `try_wait`。

Primary phase 在 arrival 完成、所跟踪异步操作的 transaction count 更新完成后推进。Conditional phase 只有在 primary phase 推进，且没有异步操作通过 `report-on` 阻止它时才推进。`.layout::v0` 不支持此类 `report-on` 阻止，因此两个 phase 总是一起推进或一起停留。`.layout::v1` 则允许 primary 已推进而 conditional 尚未推进；此时 payload report 非零，可能带有额外信息；conditional 推进时 payload report 为零。对象在 primary phase 完成后自动进入下一 phase，conditional phase 的细节见原文所链的 Phase Completion。

**9.7.15.16.6 异步操作跟踪。** 从 Hopper（`sm_9x`）开始，`tx-count` 用来计量尚未完成的异步内存操作或 transaction，单位由相应异步操作规定。当前 phase 的预期总量须设入 `tx-count`；每项异步操作完成后会对 mbarrier 执行 `complete-tx`，使 phase 向完成推进。`expect-tx(expectCount)` 把当前 phase 的 `tx-count` 增加 `expectCount`，表示又有相应数量的 transaction 需要等待。`complete-tx(completeCount)` 发出已完成信号并把 `tx-count` 减少 `completeCount`；若当前 phase 因此满足完成条件，mbarrier 转到下一 phase。具体完成条件与转换过程见官方的 Phase Completion 小节。

**9.7.15.16.7 成功完成的报告。** `.layout::v1` 能跟踪阻止 conditional phase 推进的异步操作，并在完成时提供 payload report。带 `.phase_type::primary` 的 `mbarrier.test_wait`/`try_wait` 可观察 primary 完成及 report，其内容包括 `reportPredicate` 和 `reportValue`。Conditional phase 要推进，整个 report 必须为零；若 `reportPredicate=0`，规范保证 `reportValue=0`，因此 conditional phase 已推进。带 `.phase_type::conditional` 的 wait 则直接观察 conditional phase 推进，也就意味着 payload report 为零。

##### 重点解读

可以把 mbarrier 理解为同时跟踪“参与者到齐”和“异步事务完成”的分阶段对象：pending arrivals 与 `tx-count` 不是同一个计数，`expect-tx`/`complete-tx` 管的是后者。采用 v1 时还要区分 primary 完成与 conditional 完成：primary 已完成不必然表示 report 为零。跨 CTA 的 `shared::cluster` mbarrier 只能从远端 arrive，等待应由对象所属 CTA 在允许的地址空间内完成；具体操作限制仍以所链的 shared-memory support 表为准。


##### 9.7.15.16.8 Phase Completion of the mbarrier object

**English Original**

The requirements for completion of the current phase are described below. Upon completion of the current phase, the phase transitions to the subsequent phase as described below.

**Current primary phase completion requirements**

An mbarrier object irrespective of the layout completes the current primary phase when all of the following conditions are met:

- The count of the pending arrivals has reached zero.
- The tx-count has reached zero.

**Current conditional phase completion requirements**

An mbarrier object completes the current conditional phase when all of the following conditions are met:

- The count of the pending arrivals has reached zero.
- The tx-count has reached zero.
- If the layout of the mbarrier is .layout::v1, then the payload report associated with the primary phase was zero.

**Phase transition**

When an mbarrier object with .layout::v0 completes the current phase, the following actions are performed atomically:

- The mbarrier object transitions to the next primary and conditional phases.
- The pending arrival count is reinitialized to the expected arrival count.

When an mbarrier object with .layout::v1 completes the current phase, the following actions are performed atomically:

- If the payload report associated with the current primary phase of the mbarrier is zero, then the conditional phase advances.
- The mbarrier object transitions to the next primary phase.
- The payload report corresponding to the next primary phase is reinitialized to zero.
- The pending arrival count is reinitialized to the expected arrival count.

**中文翻译**

当前 phase 完成后，mbarrier 按下述规则转换到下一 phase。无论布局为何，当前 primary phase 要完成，pending arrivals 计数和 `tx-count` 都必须归零。Conditional phase 同样要求这两个计数为零；如果采用 `.layout::v1`，还要求与该 primary phase 关联的 payload report 为零。

`.layout::v0` 当前 phase 完成时，两件事原子地发生：primary 和 conditional phase 一同进入下一 phase；pending arrival count 按 expected arrival count 重新初始化。`.layout::v1` 当前 phase 完成时，转换同样是原子的，但 conditional phase 只在当前 primary phase 的 payload report 为零时推进；primary phase 总是进入下一 phase，下一 primary phase 的 payload report 清零，pending arrival count 按 expected arrival count 重新初始化。

**重点解读**

`pending arrivals = 0` 和 `tx-count = 0` 是 primary 完成的共同门槛，不能只等线程 arrive。v1 的 conditional phase 多一个“报告为零”的门槛，所以 primary 已完成并不必然等于 conditional 已完成。

##### 9.7.15.16.9 Arrive-on operation on mbarrier object

**English Original**

An arrive-on operation, with an optional count argument, on an mbarrier object consists of the following 2 steps :

**mbarrier signalling:**

Signals the arrival of the executing thread OR completion of the asynchronous instruction which signals the arrive-on operation initiated by the executing thread on the mbarrier object. As a result of this, the pending arrival count is decremented by count. If the count argument is not specified, then it defaults to 1.

**mbarrier potentially completing the current phase:**

If the current phase has been completed then the mbarrier transitions to the next phase. Refer to Phase Completion of the mbarrier object for details on phase completion requirements and phase transition process.

**中文翻译**

一次 `arrive-on` 操作包含两个步骤。首先是 mbarrier signalling：执行线程到达，或者它发起的异步指令完成并向 mbarrier 发送 arrive-on 信号；pending arrival count 因此减少 `count`，省略该参数时默认减 1。其次，若当前 phase 已满足完成条件，mbarrier 可能在此时转换到下一 phase；具体条件及转换过程见上一小节。

**重点解读**

`arrive-on` 改变的是 pending arrival count；它可能促成 phase 完成，但若 `tx-count` 尚未归零，就不能仅凭 arrive 数量判断 phase 完成。

##### 9.7.15.16.10 Report-on operation on mbarrier object

**English Original**

The report-on operation on an mbarrier object signals miscellaneous information regarding asynchronous operation. For example, such a miscellaneous information can be about errors, warnings encountered; otherwise, it can as well be any extra information conveyed by asynchronous operation.

The miscellaneous information reported by asynchronous operation via report-on mbarrier operation would be updated in the payload report. The following asynchronous operations can potentially issue report-on operation on an associated mbarrier object:

- cp.async.bulk, cp.async.bulk.tensor instructions specifying .report_mechanism other than .mbarrier::report::disabled. Refer Report mechanisms for asynchronous copy operations for more details.
- Fabric operations such as fabric.try_get, fabric.try_put, fabric.try_red, fabric.try_pullred, fabric.try_get.async.tensor, fabric.try_put.async.tensor, fabric.try_red.async.tensor. Refer to Fabric Reporting Mechanism for more details.

**中文翻译**

`report-on` 操作向 mbarrier 传递异步操作的附加信息，例如错误、警告或其他信息，并将这些内容更新到 payload report。可能执行 `report-on` 的操作包括：指定了非 `.mbarrier::report::disabled` 的 `.report_mechanism` 的 `cp.async.bulk`、`cp.async.bulk.tensor`；以及 `fabric.try_get`、`fabric.try_put`、`fabric.try_red`、`fabric.try_pullred`、`fabric.try_get.async.tensor`、`fabric.try_put.async.tensor`、`fabric.try_red.async.tensor`。具体规则见原文所指的异步拷贝报告机制与 Fabric Reporting Mechanism。

**重点解读**

`report-on` 是附加状态报告，不是把 `tx-count` 加减一次的替代操作；在 v1 中，它会影响 conditional phase 能否推进。

##### 9.7.15.16.11 mbarrier support with shared memory

**English Original**

The following table summarizes the support of various mbarrier operations on mbarrier objects located at different shared memory locations:

| mbarrier operations | .shared::cta | .shared::cluster |
|---|---|---|
| mbarrier.arrive, mbarrier.arrive_drop | Supported | Supported, cannot return result |
| mbarrier.expect_tx | Supported | Supported |
| mbarrier.complete_tx | Supported | Supported |
| Other mbarrier operations | Supported | Not supported |

**中文翻译**

表格完整保留各项操作对 `.shared::cta` 和 `.shared::cluster` 中 mbarrier 对象的支持情况，不另翻译。关键边界是：位于 `.shared::cluster` 的对象允许 `mbarrier.arrive`、`mbarrier.arrive_drop`、`mbarrier.expect_tx`、`mbarrier.complete_tx`，但 arrive 类操作不能返回结果；其他 mbarrier 操作不受支持。

##### 9.7.15.16.12 Parallel Synchronization and Communication Instructions: mbarrier.init

**English Original**

mbarrier.init

Initialize the mbarrier object.

**Syntax**

```ptx
mbarrier.init{.layout}{.shared{::cta}}.b64 [addr], count;

.layout = { .layout::v0, .layout::v1 }
```

**Description**

mbarrier.init initializes the mbarrier object at the location specified by the address operand addr with the unsigned 32-bit integer count. The .layout qualifier specifies the layout that is used to initialize the mbarrier object. If not specified explicitly, a .layout::v0 mbarrier is initialized. Refer Layouts of the mbarrier object for more details.

The valid range of values for the operand count varies depending upon .layout as specified below:

- [1, …, 2^20 - 1] for mbarrier with .layout::v0
- [1, …, 2^9 - 1] for mbarrier with .layout::v1

The constituents of the mbarrier object are initialized as follows:

- The primary and conditional phases are initialized to zero.
- The expected arrival and pending arrival counts are initialized to count.
- The initial transaction count tx-count is initialized to zero.
- For an mbarrier object with .layout::v1, the payload report corresponding to the primary phase is initialized to zero.

If no state space is specified then Generic Addressing is used. If the address specified by addr does not fall within the address window of .shared::cta state space then the behavior is undefined.

Supported addressing modes for operand addr is as described in Addresses as Operands. Alignment for operand addr is as described in the Size and alignment of mbarrier object.

The behavior of performing an mbarrier.init operation on a memory location containing a valid mbarrier object is undefined; invalidate the mbarrier object using mbarrier.inval first, before repurposing the memory location for any other purpose, including another mbarrier object.

**PTX ISA Notes**

Introduced in PTX ISA version 7.0.

Support for sub-qualifier ::cta on .shared introduced in PTX ISA version 7.8.

Support for .layout qualifier introduced in PTX ISA version 9.3.

**Target ISA Notes**

Requires sm_80 or higher.

Qualifier .layout requires sm_90 or higher.

**Examples**

```ptx
.shared .b64 shMem, shMem2, shMem3;
.reg    .b64 addr;
.reg    .b32 %r1;

cvta.shared.u64          addr, shMem2;
mbarrier.init.b64        [addr],   %r1;
bar.cta.sync             0;
// ... other mbarrier operations on addr

mbarrier.init.shared::cta.b64 [shMem], 12;
mbarrier.init.layout::v1.shared::cta.b64 [shMem3], 4;
bar.sync                 0;
// ... other mbarrier operations on shMem
// ... other mbarrier operations on shMem3
```

**中文翻译**

`mbarrier.init` 在 `addr` 指定的位置用 32-bit 无符号 `count` 初始化 mbarrier 对象。`.layout` 指定布局；省略时默认 `.layout::v0`。合法 `count` 范围依布局而变：v0 为 `[1, 2^20 - 1]`，v1 为 `[1, 2^9 - 1]`。初始化将 primary/conditional phase 设为零、expected 和 pending arrival count 都设为 `count`、`tx-count` 设为零；v1 还将当前 primary phase 的 payload report 设为零。粘贴文本中的 `220`、`29` 为上标丢失，英文部分依据官方文档还原。

未明确指定 state space 时采用 Generic Addressing，但 `addr` 必须落在 `.shared::cta` 的地址窗口内，否则行为未定义。地址模式与对齐要求分别见 Addresses as Operands 和 Size and alignment of mbarrier object。对仍包含有效 mbarrier 对象的位置再次执行 `init` 也属于未定义行为；若要重用此位置，先执行 `mbarrier.inval`。PTX ISA 引入版本、目标架构门槛与示例保留在英文原文。

**重点解读**

`init` 设置的是这个 barrier 后续 phase 的起点和每轮预期到达数，不会自动替线程执行 arrive，也不会自动登记异步事务数。示例中的 CTA barrier 用于协调共享对象初始化后的使用。

##### 9.7.15.16.13 Parallel Synchronization and Communication Instructions: mbarrier.inval

**English Original**

mbarrier.inval

Invalidates the mbarrier object.

**Syntax**

```ptx
mbarrier.inval{.shared{::cta}}.b64 [addr];
```

**Description**

mbarrier.inval invalidates the mbarrier object at the location specified by the address operand addr. The invalidation is supported for all layouts described in Layouts of the mbarrier object.

An mbarrier object must be invalidated before using its memory location for any other purpose.

Performing any mbarrier operation except mbarrier.init on a memory location that does not contain a valid mbarrier object, results in undefined behaviour.

If no state space is specified then Generic Addressing is used. If the address specified by addr does not fall within the address window of .shared::cta state space then the behavior is undefined.

Supported addressing modes for operand addr is as described in Addresses as Operands. Alignment for operand addr is as described in the Size and alignment of mbarrier object.

**PTX ISA Notes**

Introduced in PTX ISA version 7.0.

Support for sub-qualifier ::cta on .shared introduced in PTX ISA version 7.8.

**Target ISA Notes**

Requires sm_80 or higher.

**Examples**

```ptx
.shared .b64 shmem;
.reg    .b64 addr;
.reg    .b32 %r1;
.reg    .pred t0;

// Example 1 :
bar.sync                      0;
@t0 mbarrier.init.b64     [addr], %r1;
// ... other mbarrier operations on addr
bar.sync                      0;
@t0 mbarrier.inval.b64    [addr];


// Example 2 :
bar.cta.sync                  0;
mbarrier.init.shared.b64           [shmem], 12;
// ... other mbarrier operations on shmem
bar.cta.sync                  0;
@t0 mbarrier.inval.shared.b64      [shmem];

// shmem can be reused here for unrelated use :
bar.cta.sync                  0;
st.shared.b64                      [shmem], ...;

// shmem can be re-initialized as mbarrier object :
bar.cta.sync                  0;
@t0 mbarrier.init.shared.b64       [shmem], 24;
// ... other mbarrier operations on shmem
bar.cta.sync                  0;
@t0 mbarrier.inval.shared::cta.b64 [shmem];
```

**中文翻译**

`mbarrier.inval` 使 `addr` 指定位置的 mbarrier 对象失效，两种布局均可用。对象占用的内存要改作任何其他用途之前，都必须先失效；反过来，若某位置不再包含有效对象，除 `init` 外对它执行任何 mbarrier 操作均为未定义行为。未写 state space 时使用 Generic Addressing，但 `addr` 必须指向 `.shared::cta`；地址模式和对齐要求见原文所指章节。版本、架构条件及完整示例保留在英文原文。

**重点解读**

`inval` 是对象生命周期的结束操作，不等于销毁整个 shared-memory 存储。示例展示了先协调失效，再把同一存储改作普通数据，或重新初始化为另一个 mbarrier。

##### 9.7.15.16.14 Parallel Synchronization and Communication Instructions: mbarrier.expect_tx

**English Original**

mbarrier.expect_tx

Performs expect-tx operation on the mbarrier object.

**Syntax**

```ptx
mbarrier.expect_tx{.sem.scope}{.space}.b64 [addr], txCount;

.sem   = { .relaxed }
.scope = { .cta, .cluster }
.space = { .shared{::cta} }

mbarrier.expect_tx{.sem.scope}{.space}{.multicast}.b64 [addr], txCount{, ctaMask};

.sem      = { .relaxed }
.scope    = { .cta, .cluster }
.space    = { .shared::cluster }
.multicast = { .multicast::cluster::32b }
```

**Description**

A thread executing mbarrier.expect_tx performs an expect-tx operation on the mbarrier object at the location specified by the address operand addr. The 32-bit unsigned integer operand txCount specifies the expectCount argument to the expect-tx operation.

If no state space is specified then Generic Addressing is used. If the address specified by addr does not fall within the address window of .shared::cta or .shared::cluster state space then the behavior is undefined.

Supported addressing modes for operand addr are as described in Addresses as Operands. Alignment for operand addr is as described in the Size and alignment of mbarrier object.

The optional .sem qualifier specifies a memory synchronizing effect as described in the Memory Consistency Model. The .relaxed qualifier does not provide any memory ordering semantics and visibility guarantees.

The optional .scope qualifier indicates the set of threads that directly observe the memory synchronizing effect of this operation, as described in the Memory Consistency Model.

Qualifiers .sem and .scope must be specified together.

The optional qualifier .multicast::cluster::32b allows performing the expect-tx operation on an mbarrier object from shared memory of multiple CTAs in the cluster. The operand txCount specifies the expectCount argument for the expect-tx operation. Operand ctaMask specifies the intended target CTAs in the cluster such that each bit position in the 32-bit ctaMask operand corresponds to the %cluster_ctarank of a target CTA. For example, a ctaMask with value 0x1 performs the operation on the CTA with rank 0, whereas a ctaMask with value 0x3 performs the operation on CTAs with ranks 0 and 1. The expect-tx operation is multicast to the same CTA-relative offset as addr in the shared memory of the target CTA.

**PTX ISA Notes**

Introduced in PTX ISA version 8.0.

Support for .multicast::cluster::32b qualifier introduced in PTX ISA version 9.4.

**Target ISA Notes**

Requires sm_90 or higher.

Qualifier .multicast::cluster::32b is supported on the following family-specific architectures:

sm_107f or higher in the same family

**Examples**

```ptx
mbarrier.expect_tx.b64                       [addr], 32;
mbarrier.expect_tx.relaxed.cta.shared.b64    [mbarObj1], 512;
mbarrier.expect_tx.relaxed.cta.shared.b64    [mbarObj2], 512;
mbarrier.expect_tx.relaxed.cta.shared::cluster.multicast::cluster::32b.b64    [mbarObj3], 512, 0x3;
```

**中文翻译**

`mbarrier.expect_tx` 在 `addr` 指向的 mbarrier 对象上执行 expect-tx：32-bit 无符号 `txCount` 作为 `expectCount`，增加当前 phase 预期跟踪的异步事务数。未指定 state space 时使用 Generic Addressing，但地址必须落在 `.shared::cta` 或 `.shared::cluster` 地址窗口内，否则行为未定义；地址模式与对齐要求仍按原文所引章节。可选的 `.sem` 表示内存同步效果，`.relaxed` 不提供内存顺序或可见性保证；可选 `.scope` 表示直接观察同步效果的线程集合。`.sem` 与 `.scope` 必须成对出现。

`.multicast::cluster::32b` 可把 expect-tx 施加于同一 cluster 中多个 CTA 的 shared-memory mbarrier。`ctaMask` 的 32 个 bit 对应目标 CTA 的 `%cluster_ctarank`；例如 `0x1` 选 rank 0，`0x3` 选 rank 0 和 1。对每个目标 CTA，操作作用于与 `addr` 相同的 CTA-relative offset。目标架构要求和完整示例保留于英文原文。

**重点解读**

`expect_tx` 是登记“还需完成多少异步事务”，并不发起数据搬运。Multicast 一次更新多个 CTA 的 mbarrier，必须保证每个目标 CTA 的对象布局、偏移及待完成事务数与之匹配。

##### 9.7.15.16.15 Parallel Synchronization and Communication Instructions: mbarrier.complete_tx

**English Original**

mbarrier.complete_tx

Performs complete-tx operation on the mbarrier object.

**Syntax**

```ptx
mbarrier.complete_tx{.sem.scope}{.space}.b64 [addr], txCount;

.sem   = { .relaxed }
.scope = { .cta, .cluster }
.space = { .shared{::cta} }

mbarrier.complete_tx{.sem.scope}{.space}{.multicast}.b64 [addr], txCount {, ctaMask};

.sem      = { .relaxed }
.scope    = { .cta, .cluster }
.space    = { .shared::cluster }
.multicast = { .multicast::cluster::32b }
```

**Description**

A thread executing mbarrier.complete_tx performs a complete-tx operation on the mbarrier object at the location specified by the address operand addr. The 32-bit unsigned integer operand txCount specifies the completeCount argument to the complete-tx operation.

mbarrier.complete_tx does not involve any asynchronous memory operations and only simulates the completion of an asynchronous memory operation and its side effect of signaling to the mbarrier object.

If no state space is specified then Generic Addressing is used. If the address specified by addr does not fall within the address window of .shared::cta or .shared::cluster state space then the behavior is undefined.

Supported addressing modes for operand addr are as described in Addresses as Operands. Alignment for operand addr is as described in the Size and alignment of mbarrier object.

The optional .sem qualifier specifies a memory synchronizing effect as described in the Memory Consistency Model. The .relaxed qualifier does not provide any memory ordering semantics and visibility guarantees.

The optional .scope qualifier indicates the set of threads that directly observe the memory synchronizing effect of this operation, as described in the Memory Consistency Model.

Qualifiers .sem and .scope must be specified together.

The optional qualifier .multicast::cluster::32b allows performing the complete-tx operation on the mbarrier object from shared memory of multiple CTAs in the cluster. The operand txCount specifies the completeCount argument for the complete-tx operation. Operand ctaMask specifies the intended target CTAs in the cluster; each bit position in the 32-bit ctaMask operand corresponds to the %cluster_ctarank of the target CTA. For example, ctaMask with value 0x1 performs the operation on CTA with rank 0, whereas ctaMask with value 0x3 performs the operation on CTAs with rank 0 and 1. The complete-tx operation is multicast to the same CTA-relative offset as addr in the shared memory of each target CTA.

**PTX ISA Notes**

Introduced in PTX ISA version 8.0.

Support for .multicast::cluster::32b qualifier introduced in PTX ISA version 9.4.

**Target ISA Notes**

Requires sm_90 or higher.

Qualifier .multicast::cluster::32b is supported on the following family-specific architectures:

sm_107f or higher in the same family

**Examples**

```ptx
mbarrier.complete_tx.b64             [addr],     32;
mbarrier.complete_tx.shared.b64      [mbarObj1], 512;
mbarrier.complete_tx.relaxed.cta.b64 [addr2],    32;
mbarrier.complete_tx.relaxed.cta.shared::cluster.multicast::cluster::32b.b64 [addr3], 512, mask;
```

**中文翻译**

`mbarrier.complete_tx` 在 `addr` 指定对象上执行 complete-tx：32-bit 无符号 `txCount` 作为 `completeCount`，减少当前 phase 的事务计数。它本身不涉及任何异步内存操作，只模拟异步操作完成后向 mbarrier 发信号的副作用。未指定 state space 时采用 Generic Addressing；地址必须落在 `.shared::cta` 或 `.shared::cluster` 中。地址模式和对齐要求见所引章节。`.sem` 与 `.scope` 必须同时指定；`.relaxed` 不提供内存顺序与可见性保证。

可选 `.multicast::cluster::32b` 对 cluster 中多个目标 CTA 的同一 CTA-relative offset 执行 complete-tx，目标由 `ctaMask` 的各 bit 对应 `%cluster_ctarank` 选出；`0x1` 为 rank 0，`0x3` 为 rank 0 与 1。版本、`sm_90` 基础要求、multicast 的 family-specific 架构要求及示例均保留在英文原文。

**重点解读**

这是显式“扣减事务计数”的信号指令，不应误认为它会等待或真正完成一次数据拷贝。计数若和实际异步操作不匹配，后续 phase 的完成判断就会失真。

##### 9.7.15.16.16 Parallel Synchronization and Communication Instructions: mbarrier.arrive

**English Original**

mbarrier.arrive

Performs arrive-on operation on the mbarrier object.

**Syntax**

```ptx
mbarrier.arrive{.sem.scope}{.shared{::cta}}.b64                         state, [addr]{, count};
mbarrier.arrive{.sem.scope}{.shared::cluster}{.multicast}.b64           _, [addr] {,count} {, ctaMask};
mbarrier.arrive.expect_tx{.sem.scope}{.shared{::cta}}.b64               state, [addr], txCount;
mbarrier.arrive.expect_tx{.sem.scope}{.shared::cluster}{.multicast}.b64 _, [addr], txCount {, ctaMask};
mbarrier.arrive.noComplete{.release.cta}{.shared{::cta}}.b64            state, [addr], count;

.sem   = { .release, .relaxed }
.scope = { .cta, .cluster }
.multicast = { .multicast::cluster::32b }
```

**Description**

A thread executing mbarrier.arrive performs an arrive-on operation on the mbarrier object at the location specified by the address operand addr. The 32-bit unsigned integer operand count specifies the count argument to the arrive-on operation.

If no state space is specified then Generic Addressing is used. If the address specified by addr does not fall within the address window of .shared::cta state space then the behavior is undefined.

Supported addressing modes for operand addr is as described in Addresses as Operands. Alignment for operand addr is as described in the Size and alignment of mbarrier object.

The optional qualifier .expect_tx specifies that an expect-tx operation is performed prior to the arrive-on operation. The 32-bit unsigned integer operand txCount specifies the expectCount argument to the expect-tx operation. When both qualifiers .arrive and .expect_tx are specified, then the count argument of the arrive-on operation is assumed to be 1.

A mbarrier.arrive operation with .noComplete qualifier must not cause the mbarrier to complete its current phase, otherwise the behavior is undefined.

The value of the operand count must be in the range as specified in Contents of the mbarrier object.

Note: for sm_8x, when the argument count is specified, the modifier .noComplete is required.

mbarrier.arrive operation on an mbarrier object located in .shared::cta returns an opaque 64-bit register capturing the phase of the mbarrier object prior to the arrive-on operation in the destination operand state. Contents of the state operand are implementation specific. Optionally, sink symbol '_' can be used for the state argument.

mbarrier.arrive operation on an mbarrier object located in .shared::cluster but not in .shared::cta cannot return a value. Sink symbol ‘_’ is mandatory for the destination operand for such cases.

The optional .sem qualifier specifies a memory synchronizing effect as described in the Memory Consistency Model. If the .sem qualifier is absent, .release is assumed by default.

The .relaxed qualifier does not provide any memory ordering semantics and visibility guarantees.

The optional .scope qualifier indicates the set of threads that directly observe the memory synchronizing effect of this operation, as described in the Memory Consistency Model. If the .scope qualifier is not specified then it defaults to .cta. In contrast, the .shared::<scope> indicates the state space where the mbarrier resides.

Qualifiers .sem and .scope must be specified together.

The optional qualifier .multicast::cluster::32b allows performing arrive-on and optionally expect-tx operation on an mbarrier object from shared memory of multiple CTAs in the cluster. Operand ctaMask specifies the intended target CTAs in the cluster; each bit position in the 32-bit ctaMask operand corresponds to the %cluster_ctarank of the target CTA. For example, ctaMask with the value 0x1 applies the operation on CTA with rank 0, while a value of 0x3 applies it to CTAs with ranks 0 and 1. The arrive-on and optionally expect-tx operation are multicast to the same CTA-relative offset as addr in the shared memory of each target CTA.

**PTX ISA Notes**

Introduced in PTX ISA version 7.0.

Support for sink symbol ‘_’ as the destination operand is introduced in PTX ISA version 7.1.

Support for sub-qualifier ::cta on .shared introduced in PTX ISA version 7.8.

Support for count argument without the modifier .noComplete introduced in PTX ISA version 7.8.

Support for sub-qualifier ::cluster introduced in PTX ISA version 8.0.

Support for qualifier .expect_tx is introduced in PTX ISA version 8.0.

Support for .scope and .sem qualifiers introduced in PTX ISA version 8.0

Support for .relaxed qualifier introduced in PTX ISA version 8.6.

Support for .multicast::cluster::32b introduced in PTX ISA version 9.4.

**Target ISA Notes**

Requires sm_80 or higher.

Support for count argument without the modifier .noComplete requires sm_90 or higher.

Qualifier .expect_tx requires sm_90 or higher.

Sub-qualifier ::cluster requires sm_90 or higher.

Support for .cluster scope requires sm_90 or higher.

Support for .relaxed qualifier requires sm_90 or higher.

Qualifier .multicast::cluster::32b is supported on the following family-specific architectures:

sm_107f or higher in the same family

**Examples**

```ptx
.reg .b32 cnt, remoteAddr32, remoteCTAId, addr32;
.reg .b64 %r<5>, addr, remoteAddr64;
.shared .b64 shMem, shMem2;

cvta.shared.u64            addr, shMem2;
mov.b32                    addr32, shMem2;
mapa.shared::cluster.u32   remoteAddr32, addr32, remoteCTAId;
mapa.u64                   remoteAddr64, addr,   remoteCTAId;

cvta.shared.u64          addr, shMem2;

mbarrier.arrive.shared.b64                       %r0, [shMem];
mbarrier.arrive.shared::cta.b64                  %r0, [shMem2];
mbarrier.arrive.release.cta.shared::cluster.b64  _, [remoteAddr32];
mbarrier.arrive.release.cluster.b64              _, [remoteAddr64], cnt;
mbarrier.arrive.expect_tx.release.cluster.b64    _, [remoteAddr64], tx_count;
mbarrier.arrive.noComplete.b64                   %r1, [addr], 2;
mbarrier.arrive.relaxed.cta.b64                  %r2, [addr], 4;
mbarrier.arrive.b64                              %r2, [addr], cnt;
mbarrier.arrive.release.cta.shared::cluster.multicast::cluster::32b.b64  _, [remoteAddr32], 0xa;
```

**中文翻译**

`mbarrier.arrive` 在 `addr` 指定对象上执行 arrive-on；可选的 32-bit 无符号 `count` 指定减少 pending arrivals 的数量。未写 state space 时使用 Generic Addressing；原文要求此时 `addr` 位于 `.shared::cta` 地址窗口，否则行为未定义；地址模式与对齐规则仍见所引章节。`.expect_tx` 形式先执行 expect-tx，再执行 arrive-on，`txCount` 给出 `expectCount`；这一组合下 arrive-on 的 `count` 默认为 1。带 `.noComplete` 的 arrive 不得导致当前 phase 完成，否则行为未定义。`count` 必须落在 Contents of the mbarrier object 给定范围内；在 `sm_8x` 上显式给出 `count` 时必须使用 `.noComplete`。

对于 `.shared::cta` 中的 mbarrier，`arrive` 可把 arrive 前的 phase 状态作为不透明 64-bit `state` 返回；其具体内容依实现而定，也可用 sink symbol `_` 丢弃。对于位于 `.shared::cluster`、但不属于 `.shared::cta` 的对象，不能返回值，目标必须写 `_`。可选 `.sem` 省略时默认 `.release`，而 `.relaxed` 不提供内存顺序及可见性保证。`.scope` 省略时默认 `.cta`；注意这里的 scope 指同步效果的观察范围，`.shared::<scope>` 则指定对象所在的 state space，两者含义不同。`.sem` 和 `.scope` 必须成对显式指定。

`.multicast::cluster::32b` 可将 arrive-on 及可选 expect-tx 同时作用于 cluster 内多个 CTA 的对象。`ctaMask` 按 `%cluster_ctarank` 选目标：`0x1` 选 rank 0，`0x3` 选 rank 0 与 1。每个目标都使用其 shared memory 中与 `addr` 相同的 CTA-relative offset。PTX ISA、目标架构支持与所有示例均保留于英文原文。

**重点解读**

`arrive` 同时涉及“计数”和“顺序”：`count` 更新 pending arrivals，`.expect_tx` 还会登记 `tx-count`；默认 `.release.cta` 具有内存顺序语义，但不能把它和对象所在的 `.shared::cta` 地址空间混为一谈。远端 cluster 对象的 arrive 必须丢弃返回状态，这与上一小节表格一致。

##### 9.7.15.16.17 Parallel Synchronization and Communication Instructions: mbarrier.arrive_drop

**English Original**

mbarrier.arrive_drop

Decrements the expected count of the mbarrier object and performs arrive-on operation.

**Syntax**

```ptx
mbarrier.arrive_drop{.sem.scope}{.shared{::cta}}.b64                          state, [addr] {, count};
mbarrier.arrive_drop{.sem.scope}{.shared::cluster}{.multicast}.b64            _,     [addr] {, count} {, ctaMask};
mbarrier.arrive_drop.expect_tx{.sem.scope}{.shared{::cta}}.b64                state, [addr], tx_count;
mbarrier.arrive_drop.expect_tx{.sem.scope}{.shared::cluster}{.multicast}.b64  _,     [addr], tx_count {, ctaMask};
mbarrier.arrive_drop.noComplete{.release.cta}{.shared{::cta}}.b64             state, [addr], count;

.sem   = { .release, .relaxed }
.scope = { .cta, .cluster }
.multicast = { .multicast::cluster::32b }
```

**Description**

A thread executing mbarrier.arrive_drop on the mbarrier object at the location specified by the address operand addr performs the following steps:

- Decrements the expected arrival count of the mbarrier object by the value specified by the 32-bit integer operand count. If count operand is not specified, it defaults to 1.
- Performs an arrive-on operation on the mbarrier object. The operand count specifies the count argument to the arrive-on operation.

The decrement done in the expected arrivals count of the mbarrier object will be for all the subsequent phases of the mbarrier object.

If the decrement causes the expected arrivals count to be zero, the behavior is undefined.

If no state space is specified then Generic Addressing is used. If the address specified by addr does not fall within the address window of .shared::cta or .shared::cluster state space then the behavior is undefined.

Supported addressing modes for operand addr is as described in Addresses as Operands. Alignment for operand addr is as described in the Size and alignment of mbarrier object.

The optional qualifier .expect_tx specifies that an expect-tx operation is performed prior to the arrive_drop operation, i.e. the decrement of arrival count and arrive-on operation. The 32-bit unsigned integer operand txCount specifies the expectCount argument to the expect-tx operation. When both qualifiers .arrive_drop and .expect_tx are specified, then the count argument of the arrive-on operation is assumed to be 1.

mbarrier.arrive_drop operation with .release qualifier forms the release pattern as described in the Memory Consistency Model and synchronizes with the acquire patterns.

The optional .sem qualifier specifies a memory synchronizing effect as described in the Memory Consistency Model. If the .sem qualifier is absent, .release is assumed by default. The .relaxed qualifier does not provide any memory ordering semantics and visibility guarantees.

The optional .scope qualifier indicates the set of threads that an mbarrier.arrive_drop instruction can directly synchronize. If the .scope qualifier is not specified then it defaults to .cta. In contrast, the .shared::<scope> indicates the state space where the mbarrier resides.

A mbarrier.arrive_drop with .noComplete qualifier must not complete the mbarrier, otherwise the behavior is undefined.

The value of the operand count must be in the range as specified in Contents of the mbarrier object.

Note: for sm_8x, when the argument count is specified, the modifier .noComplete is required.

A thread that wants to either exit or opt out of participating in the arrive-on operation can use mbarrier.arrive_drop to drop itself from the mbarrier.

mbarrier.arrive_drop operation on an mbarrier object located in .shared::cta returns an opaque 64-bit register capturing the phase of the mbarrier object prior to the arrive-on operation in the destination operand state. Contents of the returned state are implementation specific. Optionally, sink symbol '_' can be used for the state argument.

mbarrier.arrive_drop operation on an mbarrier object located in .shared::cluster but not in .shared::cta cannot return a value. Sink symbol ‘_’ is mandatory for the destination operand for such cases.

The optional qualifier .multicast::cluster::32b allows performing mbarrier.arrive_drop and optionally expect-tx operation on an mbarrier object from shared memory of multiple CTAs in the cluster. The operand ctaMask specifies the intended target CTAs in the cluster such that each bit in the 32-bit ctaMask corresponds to the %cluster_ctarank of the target CTA. For example, a ctaMask with value 0x1 performs the operation on CTA with rank 0, whereas ctaMask with value 0x3 performs the operation on CTAs with ranks 0 and 1. The mbarrier.arrive_drop and optionally expect-tx operation is multicast to the same CTA-relative offset as addr in the shared memory of the target CTA.

Qualifiers .sem and .scope must be specified together.

**PTX ISA Notes**

Introduced in PTX ISA version 7.0.

Support for sub-qualifier ::cta on .shared introduced in PTX ISA version 7.8.

Support for count argument without the modifier .noComplete introduced in PTX ISA version 7.8.

Support for qualifier .expect_tx is introduced in PTX ISA version 8.0.

Support for sub-qualifier ::cluster introduced in PTX ISA version 8.0.

Support for .scope and .sem qualifiers introduced in PTX ISA version 8.0

Support for .relaxed qualifier introduced in PTX ISA version 8.6.

Support for .multicast::cluster::32b qualifier introduced in PTX ISA version 9.4.

**Target ISA Notes**

Requires sm_80 or higher.

Support for count argument without the modifier .noComplete requires sm_90 or higher.

Qualifier .expect_tx requires sm_90 or higher.

Sub-qualifier ::cluster requires sm_90 or higher.

Support for .cluster scope requires sm_90 or higher.

Support for .relaxed qualifier requires sm_90 or higher.

Qualifier .multicast::cluster::32b is supported on the following family-specific architectures:

sm_107f or higher in the same family

**Examples**

```ptx
.reg .b32 cnt;
.reg .b64 %r1;
.shared .b64 shMem;

// Example 1
@p mbarrier.arrive_drop.shared.b64 _, [shMem];
@p exit;
@p2 mbarrier.arrive_drop.noComplete.shared.b64 _, [shMem], %a;
@p2 exit;
..
@!p mbarrier.arrive.shared.b64   %r1, [shMem];
@!p mbarrier.test_wait.shared.b64  q, [shMem], %r1;

// Example 2
mbarrier.arrive_drop.shared::cluster.multicast::cluster::32b.b64 _, [addr], mask;
mbarrier.arrive_drop.shared::cta.release.cluster.b64     _, [addr], cnt;

// Example 3
mbarrier.arrive_drop.expect_tx.shared::cta.relaxed.cluster.b64 state, [addr], tx_count;
```

**中文翻译**

`mbarrier.arrive_drop` 对 `addr` 指定的对象先执行两步：按 32-bit 整数 `count` 减少 expected arrival count（省略时默认 1），再用同一 `count` 执行 arrive-on。Expected arrival count 的减少适用于此后的所有 phase；若减少到零，行为未定义。这让准备退出或不再参与后续 arrive-on 的线程退出同步集合，而不仅是替当前 phase 报到。

未显式指定 state space 时使用 Generic Addressing，但 `addr` 必须位于 `.shared::cta` 或 `.shared::cluster` 地址窗口内，否则行为未定义。地址形式及对齐按原文所引章节。`.expect_tx` 形式会先以 32-bit 无符号 `txCount` 为 `expectCount` 执行 expect-tx，再执行 expected-count 减少和 arrive-on；此时 arrive-on 的 `count` 视为 1。带 `.noComplete` 的指令不得完成当前 phase，否则行为未定义；`count` 必须符合对象布局所允许的范围。对 `sm_8x`，显式提供 `count` 时必须加 `.noComplete`。

`.release` 形式构成内存模型中的 release pattern，可与 acquire pattern 同步。省略 `.sem` 默认 `.release`；`.relaxed` 不提供内存顺序或可见性保证。省略 `.scope` 默认 `.cta`；它描述直接同步的线程范围，而 `.shared::<scope>` 描述 mbarrier 所在的 state space，二者不同。显式写出 `.sem` 与 `.scope` 时必须成对指定。

对象位于 `.shared::cta` 时，`state` 可返回 arrive-on 前 phase 的不透明 64-bit 状态，也可用 `_` 丢弃。对象位于 `.shared::cluster`、但不属于 `.shared::cta` 时不能返回状态，目标必须是 `_`。`.multicast::cluster::32b` 可把 arrive_drop 及可选 expect-tx 施加到同一 cluster 的多个 CTA：`ctaMask` 每一位对应一个目标 `%cluster_ctarank`，`0x1` 选 rank 0，`0x3` 选 rank 0 和 1；各目标使用与 `addr` 相同的 CTA-relative offset。版本、架构要求和全部示例保留在英文原文。

**重点解读**

与 `mbarrier.arrive` 相比，`arrive_drop` 同时完成本轮 arrive，并永久减少后续 phase 的预期参与数。它不是简单地“跳过一次等待”；若把 expected count 减为 0，规范明确规定行为未定义。

##### 9.7.15.16.18 Parallel Synchronization and Communication Instructions: cp.async.mbarrier.arrive

**English Original**

cp.async.mbarrier.arrive

Makes the mbarrier object track all prior cp.async operations initiated by the executing thread.

**Syntax**

```ptx
cp.async.mbarrier.arrive{.noinc}{.shared{::cta}}.b64 [addr];
```

**Description**

Causes an arrive-on operation to be triggered by the system on the mbarrier object upon the completion of all prior cp.async operations initiated by the executing thread. The mbarrier object is at the location specified by the operand addr. The arrive-on operation is asynchronous to execution of cp.async.mbarrier.arrive.

When .noinc modifier is not specified, the pending count of the mbarrier object is incremented by 1 prior to the asynchronous arrive-on operation. This results in a zero-net change for the pending count from the asynchronous arrive-on operation during the current phase. The pending count of the mbarrier object after the increment should not exceed the limit as mentioned in Contents of the mbarrier object. Otherwise, the behavior is undefined.

When the .noinc modifier is specified, the increment to the pending count of the mbarrier object is not performed. Hence the decrement of the pending count done by the asynchronous arrive-on operation must be accounted for in the initialization of the mbarrier object.

If no state space is specified then Generic Addressing is used. If the address specified by addr does not fall within the address window of .shared::cta state space then the behavior is undefined.

Supported addressing modes for operand addr is as described in Addresses as Operands. Alignment for operand addr is as described in the Size and alignment of mbarrier object.

**PTX ISA Notes**

Introduced in PTX ISA version 7.0.

Support for sub-qualifier ::cta on .shared introduced in PTX ISA version 7.8.

**Target ISA Notes**

Requires sm_80 or higher.

**Examples**

```ptx
// Example 1: no .noinc
mbarrier.init.shared.b64 [shMem], threadCount;
....
cp.async.ca.shared.global [shard1], [gbl1], 4;
cp.async.cg.shared.global [shard2], [gbl2], 16;
....
// Absence of .noinc accounts for arrive-on from completion of prior cp.async operations.
// So mbarrier.init must only account for arrive-on from mbarrier.arrive.
cp.async.mbarrier.arrive.shared.b64 [shMem];
....
mbarrier.arrive.shared.b64 state, [shMem];

waitLoop:
mbarrier.test_wait.shared.b64 p, [shMem], state;
@!p bra waitLoop;



// Example 2: with .noinc

// Tracks arrive-on from mbarrier.arrive and cp.async.mbarrier.arrive.

// All threads participating in the mbarrier perform cp.async
mov.b32 copyOperationCnt, threadCount;

// 3 arrive-on operations will be triggered per-thread
mul.lo.u32 copyArrivalCnt, copyOperationCnt, 3;

add.u32 totalCount, threadCount, copyArrivalCnt;

mbarrier.init.shared.b64 [shMem], totalCount;
....
cp.async.ca.shared.global [shard1], [gbl1], 4;
cp.async.cg.shared.global [shard2], [gbl2], 16;
...
// Presence of .noinc requires mbarrier initialization to have accounted for arrive-on from cp.async
cp.async.mbarrier.arrive.noinc.shared.b64 [shMem]; // 1st instance
....
cp.async.ca.shared.global [shard3], [gbl3], 4;
cp.async.ca.shared.global [shard4], [gbl4], 16;
cp.async.mbarrier.arrive.noinc.shared::cta.b64 [shMem]; // 2nd instance
....
cp.async.ca.shared.global [shard5], [gbl5], 4;
cp.async.cg.shared.global [shard6], [gbl6], 16;
cp.async.mbarrier.arrive.noinc.shared.b64 [shMem]; // 3rd and last instance
....
mbarrier.arrive.shared.b64 state, [shMem];

waitLoop:
mbarrier.test_wait.shared.b64 p, [shMem], state;
@!p bra waitLoop;
```

**中文翻译**

`cp.async.mbarrier.arrive` 使对象跟踪执行线程此前发起的所有 `cp.async` 操作。只有这些操作全部完成，系统才在 `addr` 指定的 mbarrier 上触发 arrive-on；这个 arrive-on 与指令本身的执行异步。未指定 `.noinc` 时，会先把 pending arrival count 加 1，再由稍后的异步 arrive-on 减 1，因此当前 phase 的净变化为零。加 1 后的计数不能超过对象上限，否则行为未定义。

指定 `.noinc` 后，不做预先的加 1；异步 arrive-on 带来的减 1 必须在 mbarrier 初始化的 expected/pending count 中预留。未写 state space 时使用 Generic Addressing，但 `addr` 必须指向 `.shared::cta`，否则行为未定义；地址形式和对齐要求见原文所引章节。PTX ISA 与目标架构要求及两个完整示例均保留在英文原文。

**重点解读**

无 `.noinc` 的“先加后减”让异步完成受 barrier 跟踪，却不消耗初始化时为普通 arrive 预留的次数；有 `.noinc` 则直接消耗一次预期 arrive，初始化计数必须把它算进去。此处跟踪的是该线程此前的 `cp.async`，不是笼统地等待所有线程或所有异步拷贝。

##### 9.7.15.16.19 Parallel Synchronization and Communication Instructions: mbarrier.test_wait / mbarrier.try_wait

**English Original**

mbarrier.test_wait, mbarrier.try_wait

Checks whether the mbarrier object has completed the phase.

**Syntax**

```ptx
// without parity
mbarrier.test_wait{.phase_type::primary}{.sem.scope}{.ss}.b64      waitComplete, [addr], state;

mbarrier.test_wait.phase_type::primary{.sem}{.scope}{.ss}.b64      waitComplete|reportPredicate
                                                                   {, reportValue}, [addr], state;

// with parity
mbarrier.test_wait.parity{.phase_type}{.sem.scope}{.ss}.b64        waitComplete, [addr], phaseParity;

mbarrier.test_wait.parity.phase_type::primary{.sem}{.scope}{.ss}.b64  waitComplete|reportPredicate
                                                                      {, reportValue}, [addr], phaseParity;

// without parity
mbarrier.try_wait{.phase_type::primary}{.sem.scope}{.ss}.b64      waitComplete, [addr], state {, timeHint};

mbarrier.try_wait.phase_type::primary{.sem}{.scope}{.ss}.b64      waitComplete|reportPredicate {, reportValue},
                                                                  [addr], state {, timeHint};

// with parity
mbarrier.try_wait.parity{.phase_type}{.sem.scope}{.ss}.b64            waitComplete, [addr], phaseParity {, timeHint};

mbarrier.try_wait.parity.phase_type::primary{.sem}{.scope}{.ss}.b64   waitComplete|reportPredicate {, reportValue},
                                                                      [addr], phaseParity {, timeHint};


.ss   = { .shared{::cta} }
.sem   = { .acquire, .relaxed }
.scope = { .cta, .cluster }
.phase_type = { .phase_type::primary, .phase_type::conditional }
```

**Description**

The test_wait and try_wait operations test for the completion of the current or the immediately preceding phase of an mbarrier object at the location specified by the operand addr.

mbarrier.test_wait is a non-blocking instruction which tests for the completion of the phase.

mbarrier.try_wait is a potentially blocking instruction which tests for the completion of the phase. If the phase is not complete, the executing thread may be suspended. Suspended thread resumes execution when the specified phase completes OR before the phase completes following a system-dependent time limit. The optional 32-bit unsigned integer operand timeHint specifies the time limit, in nanoseconds, that may be used for the time limit instead of the system-dependent limit.

mbarrier.test_wait and mbarrier.try_wait test for completion of the phase :

- Specified by the 64-bit unsigned integer operand state, which was returned by an mbarrier.arrive instruction on the same mbarrier object during the current or the immediately preceding phase. Or
- Indicated by the 32-bit unsigned integer operand phaseParity, whose integer parity denotes either the current phase or the immediately preceding phase of the mbarrier object.

The .parity variant of the instructions test for the completion of the phase indicated by the integer parity of the operand phaseParity, which denotes either the current phase or the immediately preceding phase of the mbarrier object. An even phase has integer parity 0 and an odd phase has integer parity of 1.

Note: the use of the .parity variants of the instructions requires tracking the phase of an mbarrier object throughout its lifetime.

The test_wait and try_wait operations are valid only for :

- the current incomplete phase, for which waitComplete returns False.
- the immediately preceding phase, for which waitComplete returns True.

The qualifier .phase_type::* specifies the exact phase to test for completion of an operation. The semantics around possible combinations are summarized below:

- .phase_type::* is unspecified: checks completion of the primary phase.
- .phase_type::primary checks completion of the primary phase and sets the reportPredicate and reportValue. Note that the reportPredicate and reportValue operands are undefined when waitComplete is False. For .layout::v0, reportPredicate and reportValue are always zero.
- .phase_type::conditional requires .parity qualifier and checks for completion of the conditional phase, which may complete independently from the primary phase. For .layout::v0, the primary and conditional phase complete in unison.

Operand reportValue must be of type .b8.

If no state space is specified then Generic Addressing is used. If the address specified by addr does not fall within the address window of .shared::cta state space then the behavior is undefined.

Supported addressing modes for operand addr is as described in Addresses as Operands. Alignment for operand addr is as described in the Size and alignment of mbarrier object.

When mbarrier.test_wait and mbarrier.try_wait operations with .acquire qualifier returns True, they form the acquire pattern as described in the Memory Consistency Model.

The optional .sem qualifier specifies a memory synchronizing effect as described in the Memory Consistency Model. If the .sem qualifier is absent, .acquire is assumed by default. The .relaxed qualifier does not provide any memory ordering semantics and visibility guarantees.

The optional .scope qualifier indicates the set of threads that the mbarrier.test_wait and mbarrier.try_wait instructions can directly synchronize. If the .scope qualifier is not specified then it defaults to .cta. In contrast, the .shared::<scope> indicates the state space where the mbarrier resides.

Qualifiers .sem and .scope must be specified together.

The following ordering of memory operations hold for the executing thread when mbarrier.test_wait or mbarrier.try_wait having acquire semantics returns True :

- All memory accesses (except async operations) requested prior, in program order, to mbarrier.arrive having release semantics during the completed phase by the participating threads of the CTA are performed and are visible to the executing thread.
- All cp.async operations requested prior, in program order, to cp.async.mbarrier.arrive during the completed phase by the participating threads of the CTA are performed and made visible to the executing thread.
- All cp.async.bulk asynchronous operations using the same mbarrier object requested prior, in program order, to mbarrier.arrive having release semantics during the completed phase by the participating threads of the CTA are performed and made visible to the executing thread.
- All memory accesses requested after the mbarrier.test_wait or mbarrier.try_wait, in program order, are not performed and not visible to memory accesses performed prior to mbarrier.arrive having release semantics, in program order, by other threads participating in the mbarrier.
- There is no ordering and visibility guarantee for memory accesses requested by the thread after mbarrier.arrive having release semantics and prior to mbarrier.test_wait, in program order.

**PTX ISA Notes**

mbarrier.test_wait introduced in PTX ISA version 7.0.

Modifier .parity is introduced in PTX ISA version 7.1.

mbarrier.try_wait introduced in PTX ISA version 7.8.

Support for sub-qualifier ::cta on .shared introduced in PTX ISA version 7.8.

Support for .scope and .sem qualifiers introduced in PTX ISA version 8.0

Support for .relaxed qualifier introduced in PTX ISA version 8.6.

Support for .phase_type::* qualifier introduced in PTX ISA version 9.3.

Support for reportPredicate and reportValue operands introduced in PTX ISA version 9.3.

**Target ISA Notes**

mbarrier.test_wait requires sm_80 or higher.

mbarrier.try_wait requires sm_90 or higher.

Support for .cluster scope requires sm_90 or higher.

Support for .relaxed qualifier requires sm_90 or higher.

Operands reportPredicate and reportValue and qualifier .phase_type::* requires sm_90 or higher.

**Examples**

```ptx
// Example 1a, thread synchronization with test_wait:

.reg .b64 %r1;
.shared .b64 shMem;

mbarrier.init.shared.b64 [shMem], N;  // N threads participating in the mbarrier.
...
mbarrier.arrive.shared.b64  %r1, [shMem]; // N threads executing mbarrier.arrive

// computation not requiring mbarrier synchronization...

waitLoop:
mbarrier.test_wait.phase_type::primary.shared.b64    complete, [shMem], %r1;
@!complete nanosleep.u32 20;
@!complete bra waitLoop;

// Example 1b, thread synchronization with try_wait :

.reg .b64 %r1;
.shared .b64 shMem;

mbarrier.init.layout::v0.shared.b64 [shMem], N;  // N threads participating in the mbarrier.
...
mbarrier.arrive.shared.b64  %r1, [shMem]; // N threads executing mbarrier.arrive

// computation not requiring mbarrier synchronization...

waitLoop:
mbarrier.try_wait.phase_type::primary.relaxed.cluster.shared.b64    complete, [shMem], %r1;
@!complete bra waitLoop;


// Example 2, thread synchronization using phase parity :

.reg .b32 i, parArg;
.reg .b64 %r1;
.shared .b64 shMem;

mov.b32 i, 0;
mbarrier.init.shared.b64 [shMem], N;  // N threads participating in the mbarrier.
...
loopStart :                           // One phase per loop iteration
    ...
    mbarrier.arrive.shared.b64  %r1, [shMem]; // N threads
    ...
    and.b32 parArg, i, 1;
    waitLoop:
    mbarrier.test_wait.parity.shared.b64  complete, [shMem], parArg;
    @!complete nanosleep.u32 20;
    @!complete bra waitLoop;
    ...
    add.u32 i, i, 1;
    setp.lt.u32 p, i, IterMax;
@p bra loopStart;


// Example 3, Asynchronous copy completion waiting :

.reg .b64 state;
.shared .b64 shMem2;
.shared .b64 shard1, shard2;
.global .b64 gbl1, gbl2;

mbarrier.init.shared.b64 [shMem2], threadCount;
...
cp.async.ca.shared.global [shard1], [gbl1], 4;
cp.async.cg.shared.global [shard2], [gbl2], 16;

// Absence of .noinc accounts for arrive-on from prior cp.async operation
cp.async.mbarrier.arrive.shared.b64 [shMem2];
...
mbarrier.arrive.shared.b64 state, [shMem2];

waitLoop:
mbarrier.test_wait.shared::cta.b64 p, [shMem2], state;
@!p bra waitLoop;

// Example 4, Synchronizing the CTA0 threads with cluster threads
.reg .b64 %r1, addr, remAddr;
.shared .b64 shMem;

cvta.shared.u64          addr, shMem;
mapa.u64                 remAddr, addr, 0;     // CTA0's shMem instance

// One thread from CTA0 executing the below initialization operation
@p0 mbarrier.init.shared::cta.b64 [shMem], N;  // N = no of cluster threads

barrier.cluster.arrive;
barrier.cluster.wait;

// Entire cluster executing the below arrive operation
mbarrier.arrive.release.cluster.b64              _, [remAddr];

// computation not requiring mbarrier synchronization ...

// Only CTA0 threads executing the below wait operation
waitLoop:
mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64  complete, [shMem], 0;
@!complete bra waitLoop;

// Example 5 Tracking success-status of asynchronous operation using mbarrier.test_wait:

.reg .b64 %r1;
.shared .b64 shMem;

mbarrier.init.layout::v1.shared::cta.b64 [shMem], N;  // N threads participating in the mbarrier
...

// computation that issues asynchronous operation specifying report mechanism on mbarrier located at shMem
...

waitLoop:
mbarrier.test_wait.phase_type::primary.shared::cta.b64    complete|reportPred, reportValue, [shMem], %r1;
@!complete nanosleep.u32 20;
@!complete bra waitLoop;

@reportPred bra noSuccess;
// asynchronous operation completed successfully
...

exit;

noSuccess:
// Handle unsuccessful asynchronous operation
// Inspect reportVal for more details

// Example 6 Tracking successful completion of asynchronous operation by waiting on conditional phase:

.reg .b64 %r1;
.reg .b32 parity, timeHint;
.shared .b64 shMem;

mbarrier.init.layout::v1.shared::cta.b64 [shMem], N;  // N threads participating in the mbarrier
...

// computation that issues asynchronous operation specifying report mechanism on mbarrier located at shMem
...

waitLoop:
mbarrier.try_wait.parity.phase_type::conditional.relaxed.cluster.shared::cta.b64    complete, [shMem], parity, timeHint;
@!complete bra waitLoop;
// asynchronous operation completed successfully
```

**中文翻译**

`mbarrier.test_wait` 和 `mbarrier.try_wait` 检查对象当前或紧前一个 phase 是否完成。前者是非阻塞测试；后者可能阻塞：若 phase 未完成，执行线程可能挂起，并在 phase 完成时，或在系统决定的时间限制到达前恢复。可选的 32-bit 无符号 `timeHint` 给出纳秒单位的时间提示，可能取代系统默认限制；它并不保证返回时 phase 已完成。

可以用同一对象的 `mbarrier.arrive` 在当前或紧前一个 phase 返回的 64-bit `state` 指定待测 phase，也可以用 32-bit `phaseParity` 的整数奇偶位指出当前或紧前一个 phase：偶数 phase 为 0，奇数为 1。采用 `.parity` 形式，需要在对象整个生命周期中自行跟踪 phase。有效测试对象仅是当前尚未完成的 phase（`waitComplete=False`）或紧前一个已完成的 phase（`waitComplete=True`），不能拿更早的状态继续等待。

未指定 `.phase_type::*` 时检查 primary phase。`.phase_type::primary` 检查 primary，并输出 `reportPredicate`、`reportValue`；若 `waitComplete=False`，这两个报告输出未定义。对 `.layout::v0`，报告输出始终为零。`.phase_type::conditional` 必须搭配 `.parity`，检查可与 primary 独立完成的 conditional phase；v0 的两种 phase 则同时完成。`reportValue` 的类型必须为 `.b8`。未给 state space 时地址采用 Generic Addressing，但必须落在 `.shared::cta` 窗口内；地址和对齐限制见原文所引章节。

当带 `.acquire` 的 `test_wait` 或 `try_wait` 返回 `True` 时，才形成内存模型中的 acquire pattern。省略 `.sem` 默认 `.acquire`，而 `.relaxed` 不提供内存顺序和可见性保证；省略 `.scope` 默认 `.cta`。`.scope` 表示可直接同步的线程范围，`.shared::<scope>` 表示对象所在空间；显式 `.sem` 与 `.scope` 必须一起指定。

在 acquire wait 返回 `True` 后，文档保证以下顺序与可见性：参与线程在已完成 phase 的 release `mbarrier.arrive` 之前发出的普通内存访问（不含 async 操作）已完成并对执行 wait 的线程可见；它们在 `cp.async.mbarrier.arrive` 前发出的 `cp.async` 已完成并可见；使用同一 mbarrier、且在 release `mbarrier.arrive` 前发出的 `cp.async.bulk` 也已完成并可见。执行 wait 之后的内存访问不会抢到其他参与线程 release arrive 之前的内存访问之前执行或可见。相反，本线程在 release arrive 之后、wait 之前发出的内存访问，没有上述顺序与可见性保证。原文保留了版本、架构门槛及六组完整示例。

**重点解读**

把 `waitComplete=True`、`.acquire` 和足够的 `.scope` 放在一起看：单次 `test_wait` 返回 False 不建立完成或 acquire 保证；`try_wait` 也可能提前恢复，仍须检查结果。对 v1，等待 primary 可读报告状态，等待 conditional 则等于要求报告为零；选错 phase 类型会改变“成功完成”的判定。

##### 9.7.15.16.20 Parallel Synchronization and Communication Instructions: mbarrier.pending_count

**English Original**

mbarrier.pending_count

Query the pending arrival count from the opaque mbarrier state.

**Syntax**

```ptx
mbarrier.pending_count{.layout}.b64 count, state;

.layout = { .layout::v0 }
```

**Description**

The pending count can be queried from the opaque mbarrier state using mbarrier.pending_count.

The state operand is a 64-bit register that must be the result of a prior mbarrier.arrive.noComplete or mbarrier.arrive_drop.noComplete instruction. Otherwise, the behavior is undefined.

The destination register count is a 32-bit unsigned integer representing the pending count of the mbarrier object prior to the arrive-on operation from which the state register was obtained. The optional qualifier .layout::v0 denotes the layout of the corresponding mbarrier object as described in the section Layouts of the mbarrier object.

**PTX ISA Notes**

Introduced in PTX ISA version 7.0.

Support for .layout qualifier introduced in PTX ISA version 9.3.

**Target ISA Notes**

Requires sm_80 or higher.

Qualifier .layout requires sm_90 or higher.

**Examples**

```ptx
.reg .b32 %r1;
.reg .b64 state;
.shared .b64 shMem;

mbarrier.arrive.noComplete.b64 state, [shMem], 1;
mbarrier.pending_count.layout::v0.b64 %r1, state;
```

**中文翻译**

`mbarrier.pending_count` 从不透明 `state` 中查询 pending arrival count，而不是直接读取当前 mbarrier 对象。`state` 必须是之前的 `mbarrier.arrive.noComplete` 或 `mbarrier.arrive_drop.noComplete` 返回的 64-bit 寄存器，否则行为未定义。输出 `count` 是 32-bit 无符号整数，表示取得该 `state` 的 arrive-on 发生之前对象的 pending count。可选的 `.layout::v0` 指示对应对象布局；版本和架构门槛、示例见英文原文。

**重点解读**

这是对先前返回的状态快照进行解码，不是实时查询 barrier 剩余参与者。尤其不能把普通 `arrive` 返回的状态或任意 64-bit 值交给它。

##### 9.7.15.16.21 Parallel Synchronization and Communication Instructions: mbarrier.check_layout

**English Original**

mbarrier.check_layout

Check the layout of the mbarrier object.

**Syntax**

```ptx
mbarrier.check_layout.layout{.ss}.b64 p, [addr];

.layout = { .layout::v0, .layout::v1 }
.ss = { .shared::cta }
```

**Description**

The layout of the opaque mbarrier object can be queried using mbarrier.check_layout.

The address operand addr specifies the memory location of the mbarrier object whose layout is being inspected. The instruction sets the predicate operand p to True if the layout of the mbarrier object exactly matches the .layout qualifier. Refer Layouts of the mbarrier object for more details.

If no state space is specified then Generic Addressing is used. If the address specified by addr does not fall within the address window of .shared::cta state space then the behavior is undefined.

**PTX ISA Notes**

Introduced in PTX ISA version 9.3.

**Target ISA Notes**

Requires sm_90 or higher.

**Examples**

```ptx
.reg    .pred p;
.shared .b64  shMem;

mbarrier.check_layout.layout::v1.shared::cta.b64 p, [shMem];
@!p bra exit
// ... mbarrier operations on shMem

exit: ret;
```

**中文翻译**

`mbarrier.check_layout` 查询不透明 mbarrier 对象的布局。`addr` 指定待检查对象的位置；当其布局与指令中的 `.layout::v0` 或 `.layout::v1` 精确匹配时，目标 predicate `p=True`，否则为 False。未写 state space 时使用 Generic Addressing，但地址必须落在 `.shared::cta` 窗口内，否则行为未定义。指令始于 PTX ISA 9.3，要求 `sm_90` 或更高目标，示例保留在英文原文。

**重点解读**

这条指令只检查布局，不会初始化对象，也不表示 phase 已完成。若后续代码依赖 v1 的 payload report，先检查布局可避免按错误布局解释对象。

