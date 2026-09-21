# Persistent Kernel、图任务与分布式工作队列：正确性、活性和终止检测

Persistent kernel 把“启动 kernel—处理一批任务—退出”的阶段式执行改成长期驻留的 GPU workers：线程块不断从队列取任务、执行、产生新任务，并可通过 NVSHMEM 在 PE 之间推送任务或窃取负载。这种结构能够减少反复 launch 和 host coordination，也更适合不规则图遍历、动态搜索与事件驱动流水；但它同时移除了 kernel boundary 提供的天然阶段边界。队列暂时为空不再等于计算结束，远端 PUT 已发起不等于任务已发布，atomic fetch-add 得到唯一槽位也不等于槽位内容可消费。

本篇把 [B03：CUDA 执行域与前进性](b03-cuda-execution-and-cooperation.md)、[B04：缓冲区所有权与有界流水](b04-data-publishing-and-buffer-ownership.md)、[B05：远程原子与分布式协调](b05-atomic-operations-and-distributed-coordination.md)、[D04：IBGDA 原子、Signal 与排序](d04-ibgda-atomics-signals-and-ordering.md)应用于 persistent worker 系统。目标不是给出未经运行的完整框架，而是建立一套可审计的不变量：任务如何获得唯一位置、何时对消费者可见、何时回收、队列满时谁负责推进，以及怎样证明所有 PE 再也不可能产生新任务。

## 1. Persistent 的含义和三个“图”

Persistent kernel 通常指一个 kernel 在较长阶段内保持驻留，线程块通过循环处理多批工作，而不是让每批工作对应一次 kernel launch。它不是“永远不退出”的特殊 CUDA 类型，也不保证线程块被抢占、公平调度或始终获得网络进展。kernel 是否可以安全进行 grid-wide 或跨 PE 等待，取决于 launch 方式、实际 residency、同步参与集合以及等待环中是否存在可运行的推进者。

本课的“图”有三种不同含义。图算法（graph analytics）中的图由 vertex/edge 构成，BFS、SSSP 等会动态产生 frontier；task dependency graph 表示任务之间的先后关系，未必来自图数据结构；CUDA Graph 则是捕获并重放一组 CUDA operations 的执行机制。CUDA Graph 可以减少重复 launch overhead，但不会自动把动态远端队列变成正确协议，也不能替代任务发布、跨 PE termination 或 buffer ownership。本篇重点是前两种，只把 CUDA Graph 作为阶段式方案的对照。

## 2. 为什么长期驻留改变了正确性问题

阶段式实现可以让 kernel 退出、stream event 或 host barrier 成为阶段结束证据。persistent kernel 内，worker 看到本地 queue empty 时，可能存在四类尚未显现的工作：另一 worker 正在执行且稍后会生成任务；远端 PE 已预留本地槽位但 payload 尚未发布；网络中仍有 in-flight PUT/signal；某个 PE 正在持有 stolen batch 而尚未登记到本地 active count。因此“所有本地 head==tail 的一次快照”不足以证明全局终止。

长期驻留还影响资源。若 persistent grid 占满所有 SM residency，另一个负责推进通信、执行 producer 或满足同步的 kernel 可能无法被调度；即使同一 kernel 内包含全部角色，过高的寄存器/shared memory 使用也会降低 resident blocks，改变原先假定的 worker 数。CUDA 不承诺普通 oversized grid 的所有 blocks 同时运行，也不承诺特定调度顺序。需要跨 block/PE 同步的 kernel 应按 NVSHMEM 和 CUDA 的 cooperative launch 约束设计，并在实际 kernel resource usage 下查询允许的 grid size。

## 3. 单 PE 有界工作队列：先闭合四个指针语义

一个容量为 $K$ 的 ring queue 可以用单调 ticket 表示逻辑位置。多生产者先对 `reserve_tail` fetch-add 取得 ticket $t$，等待 `t < reclaim_head + K` 后写 `slot=t mod K`；写完 payload 才发布对应 ready sequence。消费者只能认领已经发布的 ticket，处理完成后推进 reclaim state。逻辑上至少存在 reservation、publication、consumption 和 reclamation 四个边界，不能只用一个 head 与一个 tail 承担全部含义。

安全性可以写成三条不变量：任何两个未回收 ticket 不映射到同一可写 generation；消费者只读其 ready sequence 等于期望 ticket 的槽；生产者只有观察到前一 generation 已回收才覆盖。原子 fetch-add 只证明 ticket 唯一；若 producer A 取得 $t$ 后停顿、producer B 已发布 $t+1$，就产生 publication hole，消费者不能仅因 `reserve_tail > t` 读取 A 的空槽。

```text
reserve(t) -> wait capacity -> write payload(t) -> publish ready(t)
     consumer waits ready(t) -> claim/read -> execute -> reclaim(t)
```

活性还需要额外前提：消费者最终会运行，持有 reservation 的生产者最终会发布或取消，满队列时等待者不会占用全部能够消费的执行资源。若每个 resident block 都在“等待队列有空槽”而消费代码位于尚未调度的 blocks 中，所有原子和 memory ordering 都正确，系统仍会死锁。

## 4. 批量认领、窃取与所有权转移

逐任务 atomic 会在热点 counter 上产生竞争。worker 可以一次 fetch-add 认领 $B$ 个连续 tickets，再在 block/warp 内分配；这降低原子频率，却会造成尾部碎片和负载颗粒变粗。最后一批必须用有效范围裁剪，不能因为预留了 $B$ 就读取尚未发布的槽。

Work stealing 让空闲 PE 从负载高的 victim 取得任务。安全协议必须确定哪一侧推进 dequeue boundary，以及任务被偷后由谁负责 completion 和回收。典型 pull-steal 可以先原子认领 victim 的已发布区间，再 GET payload；只有取得唯一 claim 的 thief 能执行该 ticket。若先 GET 再 claim，多个 thief 可能重复执行；若 claim 后 thief 永远不完成，则任务可能永久丢失，因此严格容错还需要 lease、重试或恢复日志，普通原子队列本身不提供这些能力。

Push-based load balancing 由 owner 把任务 PUT 到目标 queue。它需要先在目标预留容量，再写 payload 并发布 signal；目标不得把 remote reservation counter 当作 ready count。pull 与 push 的主要差别是决定目标和发起数据移动的角色，并没有消除 publication、背压或远端完成问题。

## 5. 图遍历：frontier 是动态任务集合

以分布式 BFS 为例，每个 vertex 有 owner PE，当前 frontier 中的 vertex 展开 outgoing edges。对邻接点 $v$，worker 需要判断是否首次发现，并将新任务发给 owner。若多个 PE 可同时发现 $v$，可在 owner 的 symmetric `distance[v]` 上使用 compare-swap，把未访问值原子转换为本层距离；只有 CAS 成功者生成后续任务。CAS 解决“谁首次发现”，但仍需在生成任务时完成 queue reservation、payload publication 和 outstanding accounting。

另一方案允许重复候选先进入队列，再由 owner 本地去重。它减少远端 atomic，却增加网络、队列容量和重复计算。哪种方案更好取决于 frontier density、图分区、重复率、atomic 路径和负载倾斜，不能脱离输入图与 transport 判断。高出度 vertex 还可能瞬间生成远超 queue capacity 的任务，producer 必须分块、溢出到备用结构或参与消费，不能无限 reserve 后再等待空间。

对于 level-synchronous BFS，每层 frontier 全局结束后才进入下一层，termination 相对简单但包含 team-wide phase synchronization。异步遍历允许不同 PE 处理不同逻辑深度，可能减少 barrier，却需要算法本身容忍乱序，并使用更强的 quiescence 检测。删除层间 barrier 不只是性能改动，它会改变算法状态和正确性证明。

## 6. 任务发布必须与 outstanding accounting 形成闭环

终止检测常引入 `outstanding`：系统中尚未完成的逻辑任务数量。危险的顺序是先把当前任务计为完成，再登记它产生的 children；此间 counter 可能瞬时为零，coordinator 错误宣布终止。安全顺序必须保证“父任务责任”在“子任务责任”可见之前不消失，例如先为将发布的 children 增加 outstanding，再发布 children，最后减少父任务计数。

但是单一全局 counter 仍有边界。若对 children 的计数增加成功，而 payload publication 永远失败，系统不会终止；若 payload 已远端可见而计数更新尚未完成，也可能提前终止。计数、payload 和 ready 分属多个位置或 QP 时，需要明确 ordering，并为异常路径定义撤销或失败状态。计数为零还应排除未提交 reservation、in-flight transport 和 active worker，因此更准确的全局 quiescence 条件是：

$$
\forall p:\;Q_p=0\;\land\;A_p=0\;\land\;R_p=0,
\qquad M_{inflight}=0,
$$

其中 $Q_p$ 是已发布未认领任务，$A_p$ 是正在执行且可能生成新任务的 worker，$R_p$ 是已预留未发布任务，$M_{inflight}$ 是尚未落入这些可见状态的跨 PE 消息。实际协议未必直接测量每一项，但必须用守恒量或 phase/ack 间接覆盖它们。

## 7. 两种终止检测思路及其假设

### 7.1 守恒计数或 credit

初始任务获得总 credit。处理任务时，父任务把 credit 分给 children，只有当 task 及其 descendants 全部完成，credit 才向上归还；root 收回全部 credit 且自身 idle 时宣布终止。集中 outstanding counter 是其简化形式，容易理解，但会成为远端 atomic 热点，并要求严格排列 child registration 与 parent completion。

分层 counter 可以按 PE 或子树聚合，减少中心竞争；coordinator 收集各 PE 的 `queued + active + reserved`，并确认通信 epoch 内没有新增工作。一次全局快照仍可能把“发送方已扣除、接收方尚未增加”的在途任务漏掉，因此需让发送责任在接收 acknowledgement 前继续计入，或使用可证明的一致快照协议。

### 7.2 Epoch 与双重观察

另一思路是每个 PE 在本地 idle 时上报当前 epoch 和发送/接收计数。coordinator 只有在一轮观察中所有 PE idle、全局发送数等于接收数，并且下一轮验证期间 epoch/state 没有变化，才发布 termination generation。双重观察减少瞬时空队列误判，但它的正确性依赖计数覆盖所有消息、状态读取的 ordering 以及 idle PE 在产生新工作时先撤销 idle。

无论采用哪种方案，termination signal 都应携带 generation，worker 退出前还要确认没有更新的工作 generation。只广播一个永久布尔 `stop=1`，下一轮复用时会产生 ABA；如果 stop 与新任务能并发到达，还要定义优先级和不可逆的关闭阶段。

## 8. Cooperative launch、驻留与等待环

NVSHMEM 当前文档要求：使用 NVSHMEM synchronization 或 collective device APIs 的 kernel 通过 `nvshmemx_collective_launch` 启动。该调用在各 PE 上使用 CUDA cooperative launch，检查 kernel 是否适配可协同驻留的 grid，并集体启动。它解决的是某类“等待者占满资源，使匹配调用永远无法调度”的风险，不代表应用可以忽略参与一致性、stream dependency 或跨 PE 环路。

CUDA cooperative grid sync 要求所有 grid threads 按相同 phase 参与；在分支中只有部分 blocks 到达会死锁。persistent worker 常有部分 blocks idle、部分繁忙，因此不能随意插入 `grid.sync()` 作为终止检测。更自然的做法通常是让所有 blocks 都执行明确的状态机，并只在能够证明全体到达的阶段使用 grid barrier，或由少量 coordinator threads 通过原子状态和 generation 协调。

grid size 为 0 时 NVSHMEM runtime 可以选择适合 cooperative launch 的最大 grid，但“最大 occupancy”不一定是最佳应用配置。过多 persistent blocks 会与通信提交争夺 SM/L2/HBM，过少则降低本地任务吞吐与隐藏延迟能力。应查询 grid 上限，再按 E01 分别测 compute throughput、steal latency、network injection 和端到端完成时间。

## 9. 背压与不均衡：不能让等待占满推进资源

有界队列满时，最简单的 producer blocking 可能形成环：PE 0 等 PE 1 空槽，PE 1 等 PE 0 空槽，双方所有 resident workers 都停在 enqueue。可行策略包括 worker 在 enqueue 失败时转为消费本地任务、限制每次任务生成的 reservation、保留专用 progress workers、分层 spill buffer，或让 owner 主动拉取。每种策略都改变内存上限、公平性和终止 accounting。

图分区或动态任务成本会造成负载不均。只按 queue length 选 victim 可能误判：短队列可能含高成本任务，长队列可能都是轻任务；远程 steal 的网络成本也可能超过任务计算。调度指标可以结合 published queue depth、active workers、历史 task time 和 locality，但任何启发式都需保持 claim 唯一性，不得为了估计负载直接消费未发布状态。

饥饿与死锁也要区分。系统仍有某些任务持续完成、但一个 victim 或 ticket 长期得不到服务是 starvation/fairness 问题；所有可能推进者形成等待环才是 deadlock；队列和 active count 最终为零但 termination protocol 不宣布结束则是 livelock 或检测协议问题。诊断时应记录状态转换，而不只记录 kernel “仍在运行”。

## 10. 失败传播与可恢复性的边界

普通 NVSHMEM RMA、atomic、signal 和 collective 提供通信与同步能力，不应被假定为分布式事务或进程失效恢复协议。若一个 PE 在持有 reservation、active task 或 credit 时失效，其他 PE 可能永久等待其 publication、ack 或 collective participation。应用若要求容错，必须额外定义 failure detector、任务幂等性、owner/lease、日志或 checkpoint，以及在 communicator/team 失效后的恢复边界。

即便没有 PE crash，也要处理任务执行错误。worker 不能悄悄丢弃失败任务后把 outstanding 减零；可将 task 状态从 `READY` 原子转换为 `RUNNING`，再转为 `DONE` 或 `FAILED`，由 coordinator 聚合 first-error 和取消 generation。取消同样不能立即复用所有 buffers：必须先停止新任务生成，排空或标记 in-flight publication，确认没有消费者再访问，然后才回收。

本文只建立 failure propagation 的需求，不声称 NVSHMEM 当前版本提供透明 fault tolerance，也不设计跨进程崩溃后可直接继续的实现。

## 11. 可观测性、验证与实验设计

调试应为每个 ticket 记录有限状态，而不是无限打印。可采样 `RESERVED→PUBLISHED→CLAIMED→DONE/FAILED→RECLAIMED` 的 ticket、generation、source/destination PE 和时间戳；为 termination 记录各 PE 的 queued、active、reserved、sent、received、idle epoch。日志 buffer 本身必须有界，且不能改变队列 ordering 或因打印同步掩盖 race。

安全性测试应使用很小容量、多个 producer/thief、随机延迟和 ticket 回绕，检查任务既不丢失也不重复、槽位不在消费前覆盖、payload 与 ready generation 一致。活性测试应构造满队列互推、publication hole、零长度生成、极端热点 vertex 和所有队列短暂为空但仍有 active worker 的情形。性能测试再逐步扫描 persistent blocks、queue capacity、batch claim、steal size 和 remote/local ratio，并同时保留正确性计数。

实验成功不能只看最终 output。至少验证：`generated = completed + failed + cancelled` 的定义是否覆盖重试；每个逻辑 task ID 的执行次数符合 exactly-once 或明确的 at-least-once 设计；termination 前所有 reservation 已发布/取消且 in-flight responsibility 已归零；kernel 退出后对称 buffer 才被释放。规划中的 `experiment-e03-persistent-work-queue.md` 尚未创建和运行。

## 12. 掌握标准与结论

完成本课后，应能为有界 MPMC queue 分别标出 reserve、publish、claim 和 reclaim；解释 publication hole 和满队列等待为何既是安全问题也是活性问题；为 BFS 的首次发现、frontier publication 和重复去重选择协议；证明 termination 不能由一次 empty snapshot 得出；并为 outstanding/credit 或 epoch snapshot 写出消息在途与 active worker 的覆盖关系。

Persistent kernel 的核心收益来自把控制循环留在 GPU，但正确性的代价是应用自己承担原本由 kernel boundary 隐含提供的阶段、完成和回收证明。NVSHMEM 的 one-sided RMA、AMO 与 signal 能够实现这些状态转换，却不会自动组合成工作队列、全局 quiescence 或容错系统。可靠设计必须同时满足安全性不变量、有限资源背压、实际 CUDA residency 和跨 PE 前进性。

本文没有编译或运行 persistent kernel、图遍历、work stealing 或 termination protocol，没有进行 GPU/NIC 实验，也没有生成性能结论。所有状态机和公式属于静态协议推演，需由规划实验在固定 CUDA、NVSHMEM、transport 与硬件环境中验证。

## 参考资料

- [NVIDIA NVSHMEM：NVSHMEM and the CUDA Model](https://docs.nvidia.com/nvshmem/api/latest/cuda-interactions.html)
- [NVIDIA NVSHMEM Kernel Launch Routines](https://docs.nvidia.com/nvshmem/api/latest/api/launch.html)
- [NVIDIA NVSHMEM Atomic Memory Operations](https://docs.nvidia.com/nvshmem/api/latest/gen/api/amo.html)
- [NVIDIA NVSHMEM Signaling Operations](https://docs.nvidia.com/nvshmem/api/latest/gen/api/signal.html)
- [CUDA Programming Guide：Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)
- [CUDA Programming Guide：Kernel Launch and Occupancy](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#kernel-launch-and-occupancy)
