# B03：CUDA 执行域、协作通信与前进性

NVSHMEM 允许通信直接出现在 CUDA kernel 中，这使计算、数据搬运和跨 GPU 同步能够组合在一个设备端流程里。但“在 kernel 中调用通信函数”并不足以说明程序正确：必须进一步确定究竟是一个 thread 发起一次操作，还是一个 warp 或整个 block 共同发起；参与线程怎样把本地数据交给通信操作；不同 CUDA stream 之间如何建立依赖；跨 PE 的阻塞等待能否获得调度前进性；同一个 Team 上是否出现了多个相互冲突的 collective 实例。

本课承接 [B02：完成、排序与可见性](b02-completeness-ordering-and-visibility.md)，但关注点不同。B02 解决的是“一次操作走到了源可复用、远端交付还是消费确认”；本课解决的是“哪些执行者共同发起这次操作，以及它们能否执行到那个完成点”。正文按 2026-09-12 可见的 NVSHMEM 3.7.2 发布周期滚动 API 文档和 CUDA 官方编程指南核对。示例均为教学片段，未编译、未运行，也未在具体 GPU、互连或 transport 上验证。

## 前置模型：先把三套坐标系分开

假设每个 PE 管理一块 GPU：PE 0 上的 kernel 计算一段数组，然后把数组发送到 PE 1；PE 1 的 kernel 等待数据到达并继续计算。这个场景同时存在三套坐标系。第一套是 NVSHMEM 的 PE/Team 坐标，回答哪些进程级通信端点参与；第二套是 CUDA 的 thread/warp/block/grid 坐标，回答同一 GPU 上哪些线程执行代码；第三套是 host、device 和 CUDA stream 的发起坐标，回答操作由 CPU 直接调用、由 GPU thread 调用，还是由 CPU 排入某条 stream。

三套坐标不能互相替代。一个 block 有 256 个 CUDA threads，不代表 NVSHMEM 作业有 256 个 PEs；一个 Team 包含 8 个 PEs，也不代表每个 PE 上必须恰有 8 个 threads；`nvshmemx_collective_launch` 是所有 PEs 共同参与的 host API，却在每个 PE 的 GPU 上启动包含许多 blocks 和 threads 的 kernel。程序审查时应先分别标出这三套坐标，再分析它们的组合。

| 维度 | 典型实体 | 它解决的问题 | 它不能自动保证的事情 |
| --- | --- | --- | --- |
| 分布式参与域 | PE、Team、`NVSHMEM_TEAM_WORLD` | 哪些通信端点拥有对称对象并参加 collective | 同一 GPU 内各 thread 的本地交接 |
| CUDA 执行域 | thread、warp、block、grid | 哪些 GPU threads 计算或协作调用 device API | 远端交付、跨 PE 会合与 Team 顺序 |
| 发起与排队域 | host thread、device thread、CUDA stream | 操作从哪里发起、与哪些 CUDA 工作有队列顺序 | 其他发起域的完成，也不自动建立跨 stream 依赖 |

[B00：NVSHMEM 编程模型](b00-nvshmem-programming-model.md)已经说明一个 PE 通常绑定一块 GPU，[B01：对称对象与远端寻址](b01-symmetric-objects-and-remote-addressing.md)说明 symmetric address 与目标 PE 共同确定远端对象。本课沿用这些前提，不重新讨论地址翻译和注册。

## thread、warp 与 block：作用域决定谁在共同调用

普通 device API 是 thread-scoped。若每个 thread 都执行一次 `nvshmem_int_p(remote, value, pe)`，语义上就是许多 thread 各自发起操作，而不是运行时自动把这些调用合并为一次 block 通信。多个 threads 若无所有权划分地写同一个远端元素，还会形成相互冲突的更新；即使最后恰好看到某个值，也不能据此推导写入顺序。

NVSHMEM 另提供带 `_warp` 和 `_block` 后缀的 thread-group API，例如 `nvshmemx_putmem_warp`、`nvshmemx_putmem_block`、相应的 NBI 版本和 thread-group collective。这些函数表达的是“一个 warp 或 block 协作完成一项逻辑操作”。参与组中的 threads 必须共同调用同一个实例，并传入一致的目标地址、源地址、元素数和目标 PE 等参数。程序不能让 thread 0 传 PE 1、thread 1 传 PE 2，也不能只让分支中的一部分 threads 进入 block-scoped 调用。

这里的“共同调用”与“每线程各发一条请求”不同。thread-group API 可以利用多个 threads 分担本地读写、协议准备或传输工作，从而减少把每个元素写成独立细粒度操作的开销；具体由多少 threads 搬运、产生多少底层请求以及是否走 NVLink、CPU proxy 或网络 transport，属于实现和环境问题。没有固定源码与运行证据时，不能把一次 `_block` 调用写成“必然只有一个 WQE”，也不能仅凭函数名断定性能一定更好。

选择作用域应服从数据所有权。若数据天然由一个 thread 产生且消息很小，thread-scoped 调用通常最直接；若一个 warp 共同计算连续 tile，warp-scoped RMA 可以减少不必要的 block 会合；若整个 block 共同生成较大连续缓冲区，block-scoped RMA 更贴合生产过程。作用域越大，潜在并行度越高，但参与一致性、分支收敛和同步成本也更严格。API 作用域不是越大越好，而是应与实际协作组一致。

## warp 不是无需同步的“天然锁步区”

CUDA 以 32 个 threads 组成 warp 执行，但现代 GPU 的 independent thread scheduling 使依赖隐式锁步的旧式 warp-synchronous 写法不再可靠。数据相关分支会让同一 warp 的 threads 走不同路径；提前退出的 lane 也可能不再是 active participant。因此，在进入 warp-scoped NVSHMEM API 前，必须确保所要求的参与 lanes 在控制流上收敛，并对由多个 lanes 生产、随后跨 lanes 消费的数据使用合适的 CUDA warp 同步，例如带正确 mask 的 `__syncwarp()`。

不能用 `if (lane == 0) nvshmemx_putmem_warp(...)` 把 warp API 当作单线程函数。反过来，如果设计本来只需要 lane 0 发起普通 thread-scoped API，则其他 lanes 不应假装参与；它们只需在本地数据交接点与 lane 0 建立清晰的同步和可见性。两种设计都可以成立，错误来自把函数的参与契约和程序的实际控制流混在一起。

对于不足 32 个有效元素的尾部 warp，也不要自然推断“只有处理有效元素的 lanes 调用 `_warp` 就可以”。通信函数的参数仍描述完整逻辑范围，参与要求由对应 API 契约决定；常见做法是让参与 warp 保持一致调用，把实际 `nelems` 设为剩余元素数，由协作函数内部划分工作。若 threads 已经提前退出或使用复杂 active mask，必须针对所用 NVSHMEM 版本的 API 要求重新核对，而不能依赖偶然的 active-lane 行为。

## block 协作需要两个方向的线程交接

block-scoped RMA 常见于一个 CTA 共同生成消息的情形。这里至少有两次本地交接：通信前，所有生产 threads 要把 source 准备完毕；通信后，如果由一个控制 thread 执行 `fence`、`quiet` 或发布独立 signal，它必须确认其他 threads 的通信调用已经发起。`__syncthreads()` 或等价 block group synchronization 负责 CUDA block 内的会合与本地内存可见性，NVSHMEM ordering API 负责通信操作之间的远端顺序或完成。二者解决不同层次的问题。

下面的片段展示“一整个 block 发起 NBI PUT，随后 thread 0 排序并通知”的结构。它只用于说明控制关系，未编译、未运行；实际工程应按安装版本头文件核对 API 可用性，并由 B04 的所有权协议补充 ack。

```cpp
// 教学片段：每个 block 只处理一个消息，所有 threads 都必须进入 block API。
__global__ void send_block(int *local_src,
                           int *remote_dst,
                           uint64_t *remote_ready,
                           size_t count,
                           int peer,
                           uint64_t generation) {
    for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
        local_src[i] = make_value(i, generation);
    }

    // 本地交接 1：source 的生产先于任何 thread 参加通信。
    __syncthreads();

    nvshmemx_putmem_nbi_block(remote_dst, local_src,
                              count * sizeof(int), peer);

    // 本地交接 2：thread 0 继续前，整个 block 已完成这次共同调用。
    __syncthreads();

    if (threadIdx.x == 0) {
        // 对同一 peer 建立 payload -> ready 的交付顺序；不是消费确认。
        nvshmem_fence();
        nvshmemx_signal_op(remote_ready, generation,
                           NVSHMEM_SIGNAL_SET, peer);
    }
}
```

这里第一处 `__syncthreads()` 不能由 `nvshmem_fence()` 代替：fence 不负责让 thread 0 看见其他 threads 刚写入的 source，也不负责使所有 threads 到达同一控制点。第二处同步也不能随意删除。NVSHMEM 官方 thread-group fence/signal 示例明确采用 CTA 完成共同 NBI PUT、同步后由一个 thread 执行 fence 和 signal 的模式；ordering 函数可以覆盖该 PE 在交接前已经发起的相关操作，但应用必须先用 CUDA 线程同步证明“其他线程的发起动作发生在控制线程的 ordering 调用之前”。

如果要求的是远端完成而不是同目标 ordering，控制 thread 可以在交接后调用 device `nvshmem_quiet()`；若仅要求 NBI PUT 的 source 可复用，则应核对 `nvshmemx_flush*` 的作用域。B02 已说明 fence、quiet 和 flush 的语义差别。无论采用哪一种，都不能从 ready 推导消费者已经处理完 payload；远端 slot 的再次覆盖仍需 ack。

反方向的 GET 也需要交接。若 block-scoped GET 把结果放到 block 后续共同读取的缓冲区，通信完成条件满足后，还需要 block 同步，才能让各 threads 安全进入消费阶段。NVSHMEM 完成负责数据到达目标缓冲区，CUDA block 同步负责本地消费者之间的控制会合与内存顺序；不能因为 API 名称含 `_block` 就假设所有后续普通 CUDA 读访问已自动按应用需要编排。

## collective 的“两层集合性”

GPU 上的 NVSHMEM collective 同时具有两层集合性。外层是 Team：Team 中每个 PE 都必须按相同程序顺序参加 broadcast、reduce、barrier 或 sync。内层是每个 PE 上的调用实例：普通 device collective 由恰好一个 thread 代表该 PE；warp-scoped 版本由恰好一个 warp 共同代表；block-scoped 版本由恰好一个 block 共同代表。

“恰好一个实例”是关键。假设一个 kernel 有两个 blocks，并且两个 blocks 都执行 `nvshmemx_barrier_all_block()`。即使 kernel 使用 collective launch，使两个 blocks 能同时驻留，这仍然不是一个合法的 world barrier：同一 PE 为同一 Team 并发提供了两个 barrier 实例，而其他 PEs 无法获得无歧义的 collective 顺序。collective launch 解决调度前进性，不会把重复实例合并，也不会修正 Team 调用序列。

同一 Team 上的 team-based collectives 不得由一个 PE 的多个 threads 同时调用。不同 Teams 上的 collectives 可以并发，但应用仍须保证每个 Team 内的参与集合和顺序正确，并保证所有实例都有执行机会。官方示例允许两个 blocks 在各自不同 Team 上调用 block barrier，前提是 collective launch 保证它们可共同前进；这不是“不同 Team 永远可以无条件并发”的许可。

```text
每个 PE 上的 kernel

错误：block 0 ── barrier(TEAM_WORLD) ──┐
      block 1 ── barrier(TEAM_WORLD) ──┴─ 同一 PE、同一 Team、两个并发实例

可能合法：block 0 ── barrier(team_A) ── team_A 的所有 PEs 匹配
          block 1 ── barrier(team_B) ── team_B 的所有 PEs 匹配
          前提：team_A/team_B 有效，调用顺序正确，且两个 block 都能获得前进性
```

因此，看到 collective hang 时要同时检查两个问题：远端 PEs 是否以相同顺序进入同一 Team 的 collective，以及本地 PE 是否恰好只有一个符合作用域的调用实例。只检查 PE 数量而忽略本地 blocks，或只检查 blocks 而忽略 Team 顺序，都会遗漏一半契约。

## 为什么普通 CUDA launch 可能让设备端等待死锁

普通 CUDA grid 可以远大于 GPU 同时驻留的容量。CUDA 允许 blocks 以任意顺序、并行或串行执行，不保证一个正在占用 SM 的 block 会被抢占以让尚未调度的 block 运行。如果已驻留 block 阻塞等待某个尚未驻留 block 的动作，后者又因为资源被前者占满而无法启动，就形成调度等待环。

```text
已驻留 blocks 占满 GPU
        │
        ├─ 等待 signal / barrier 的远端或本地匹配动作
        │                                      ▲
        ▼                                      │
不退出、不释放 block 资源          匹配动作位于尚未调度的 block
        ▲                                      │
        └──────── GPU 没有空闲驻留资源 ────────┘
```

跨 PE 时这个环更隐蔽。PE 0 的已驻留 block 等待 PE 1 的某个 block，PE 1 的已驻留 block 又等待 PE 0 尚未调度的另一个 block；每块 GPU 单独看都有正在运行的线程，但全局没有任何参与者能够产生所需事件。增加超时时间、把 wait 改成更紧密的轮询或提升 stream priority 都不能从契约上打破这个环，因为 priority 只是调度提示，不提供必然执行顺序。

NVSHMEM 当前文档要求：kernel 只要使用 `nvshmem_wait*`、barrier 或其他同步/collective device API，就必须通过 `nvshmemx_collective_launch` 启动，以保证无死锁的前进和完成。只使用一侧 RMA/AMO 等非同步通信 API 的 kernel 可以采用普通 CUDA launch，但它仍须满足数据依赖、完成与 buffer ownership；“允许普通 launch”不表示这些操作自动同步。

## collective launch 的保证与边界

`nvshmemx_collective_launch` 是 host API，也是整个 NVSHMEM 作业上的 collective call。各 PE 用兼容的 kernel、grid/block 配置、参数和 stream 进入调用；运行时使用 CUDA cooperative launch，在每个 PE 的 GPU 上启动 kernel，并限制 grid 使相关 threads 能够安全同步。目标 GPU 必须支持 cooperative launch。

在指定 block 尺寸、动态 shared memory 和实际 kernel 资源占用后，可以先用 `nvshmemx_collective_launch_query_gridsize` 查询允许的最大 grid size。寄存器、shared memory、block threads 数量都会影响同时可驻留 blocks，不能只按 SM 数量手算一个固定上限。若 collective launch 的 `gridDims` 设为 0，文档允许运行时为该 kernel 选择可用的最大 cooperative grid；显式 grid 则不能超过查询上限。

collective launch 提供的是启动与驻留层面的前进性基础，不是应用协议证明。它不能修正以下错误：某个 PE 根本未调用 launch；各 PE 进入不同 kernel 或不兼容的 collective 顺序；只有一部分 block threads 调用 `_block` API；两个 blocks 同时在同一 Team 上创建两个 collective 实例；程序等待一个逻辑上永远不会发送的 generation；消费者没有返回 slot ack。换言之，co-residency 消除了“匹配执行者永远排不上”的一类环，却不会制造缺失的匹配操作。

collective launch 也不等于 kernel 已经执行完毕。它带有 CUDA stream 参数，kernel 按 stream 语义执行；host 若要读取结果、释放对称内存或 finalize，仍需等待相应 stream/kernel 到达完成点，例如检查 launch 返回值后执行 `cudaStreamSynchronize(stream)`。同样，collective launch 不是 B02 意义上的通用 quiet，不能替代 kernel 内针对 RMA completion 的排序设计。

下面给出 host 侧结构示意。错误处理被保留，是因为 cooperative grid 超限或某个 PE launch 失败不能被当作通信 hang 继续等待。

```cpp
// 教学片段，未编译、未运行；参数布局应与实际 kernel 签名严格对应。
int max_grid = 0;
void *args[] = {&src, &dst, &ready, &count, &peer};

int rc = nvshmemx_collective_launch_query_gridsize(
    (const void *)kernel_with_wait, block_dim, args,
    dynamic_smem_bytes, &max_grid);
if (rc != 0 || requested_grid > max_grid) {
    // 所有 PE 应进入一致的错误处理路径，避免部分 PE 继续 collective。
    handle_launch_configuration_error(rc, max_grid);
}

rc = nvshmemx_collective_launch(
    (const void *)kernel_with_wait,
    dim3(requested_grid), block_dim, args,
    dynamic_smem_bytes, stream);
if (rc != 0) {
    handle_collective_launch_error(rc);
}

cudaError_t st = cudaStreamSynchronize(stream);
if (st != cudaSuccess) {
    handle_cuda_execution_error(st);
}
```

## stream 与 event：排队顺序不等于网络完成

CUDA stream 是有序工作队列。同一 stream 中，前面的 producer kernel 会先于后面的 on-stream NVSHMEM 操作执行；不同 streams 则没有默认先后关系，需要用 event 建立有向依赖。on-stream API 由 CPU 调用但立即返回，它表达的是“把通信节点排入这条 GPU 工作队列”，而不是“CPU 已完成通信”。

例如 stream `compute` 生成 source，stream `comm` 调用 `nvshmemx_putmem_nbi_on_stream`，正确的 CUDA 依赖骨架是：在 `compute` 中于 producer 后记录 event，让 `comm` 等待该 event，再排入 PUT。若 source 需要复用，还应在 `comm` 中排入满足所需完成层次的 flush/quiet，并在完成节点之后记录第二个 event，使下一轮 producer 等待。event 只传递 CUDA 队列进度；究竟排入 flush 还是 quiet，仍由 B02 的本地完成或远端完成需求决定。

```text
stream compute:  [produce source] --record produced--> [下一轮覆盖前 wait reusable]
                                      │                              ▲
                                      ▼                              │
stream comm:                 wait produced -> [PUT NBI] -> [flush/quiet] --record reusable
```

如果只记录在 PUT enqueue 之后而没有排入 flush/quiet，`reusable` 事件至多说明 stream 已执行到那个排队节点，不能凭空强化 NVSHMEM 完成语义。反过来，在 `comm` 中正确安排 `nvshmemx_quiet_on_stream` 后，CPU 仍不能只看函数的立即返回；它应等待对应 event 或 stream synchronization，才能知道 GPU 队列执行到了 quiet 之后。

跨 stream 的环同样会死锁。若 stream A 中的 kernel 等待由 stream B 产生的 signal，而 stream B 又通过 event 等待 stream A kernel 完成，依赖图形成环：A kernel 不完成，B 不能执行；B 不执行，A 等不到 signal。把两条 stream 放在同一 GPU 或提高 B 的 priority 不会使有向环合法。调试时应把每个 event wait、kernel wait 和远端 signal 画成边，确认依赖图存在至少一条可执行起点。

## host thread、device thread 与 stream 的并发支持

NVSHMEM 的 thread support level描述一个 PE 内多个 CPU threads 能否并发调用 host/stream APIs。当前文档说明 host 和 stream API 支持 `NVSHMEM_THREAD_SERIALIZED`，device API 支持 `NVSHMEM_THREAD_MULTIPLE`。serialized 的含义不是“库会替应用自动排好所有调用”，而是应用必须保证同一时刻至多一个 host thread 调用 NVSHMEM；若需要从多个 CPU threads 发起，应在外部串行化，并维持所有 PEs 的 collective 顺序。

device `THREAD_MULTIPLE` 允许不同 GPU threads 并发调用线程安全的 NVSHMEM 接口，但不取消某个具体 API 的参与约束。多个 threads 可以对独立对象发起普通 RMA，不代表它们可以同时在同一 Team 上发起多个 collectives；warp/block-scoped API 仍要求相应 group 共同调用；冲突写同一 symmetric object 仍受内存模型限制。线程安全只说明库内部不会因为并发入口本身失效，不说明应用层地址、顺序和 ownership 自动正确。

任一 thread 发起的 NVSHMEM 操作都属于该 PE 的动作，但 ordering 函数若要覆盖其他 threads 先前发起的操作，应用必须先建立线程交接。CPU 可使用 mutex、condition variable 或 thread barrier；GPU 可使用与参与域匹配的 warp/block synchronization。没有这条 happens-before，仅让另一个 thread “稍后大概调用 quiet”不能构成可移植证明。

host、device 与 on-stream 之间也不是一个全局顺序域。host `nvshmem_quiet()` 不完成 GPU-initiated 操作；device quiet 不完成 host 发起操作；on-stream quiet 按所在 CUDA stream 在 GPU 侧建立顺序。若一个协议混合三个域，应明确设置边界，例如 CPU 先等待 stream 执行到 GPU-side quiet，再进入 host collective，而不是直接在 on-stream PUT 返回后调用 host barrier。

## 多进程共享 GPU 会扩大前进性风险

一 PE 一 GPU 是理解 device-side synchronization 最简单的部署。多个 PEs 共享同一 GPU（MPG）时，各进程的 kernels 竞争同一组 SM、寄存器和 shared memory；如果 PE 0 的驻留 kernel 等待 PE 1，而 PE 1 的匹配 kernel 无法获得执行资源，就可能出现跨进程驻留环。MPS 可以改变多进程并发执行条件，但不能修复错误的 Team 顺序或缺失 signal。

NVSHMEM 的 Using 文档对包含 collective launch 和 kernel-side synchronization 的程序给出保守指导：多个 PEs 不应共享同一 GPU，且 NVSHMEM PEs 应独占 GPU，以保证正确性或性能可预测性。安装文档虽描述 MPG 与 MPS 配置能力，但“运行时支持某种部署”和“任意阻塞同步协议在该部署中都有前进性”不是同一结论。若项目必须使用 MPG，应把 GPU 共享比例、MPS 状态、active thread percentage、每 kernel 资源和实际并发驻留纳入单独的部署验证，不能沿用一 PE 一 GPU 的纸面保证。

显示服务、其他计算作业、动态资源占用和不同 GPU 架构也会影响驻留与性能。collective grid size 应在实际 kernel 与设备上查询；实验报告必须记录设备、CUDA/NVSHMEM 版本、launch 参数和共享环境。没有这些证据时，本课只给出规范层的风险边界，不声称某个 MPG 配置一定死锁或一定安全。

## 常见错误及其因果定位

第一类错误是把 `_block` 当作“只能由 thread 0 调用的优化版函数”。程序可能在某种路径上挂起，或只传输了部分数据。检查方法是定位调用点的控制流，证明整个要求的 thread group 都到达同一动态调用实例并使用相同参数；不要先通过增加 `__syncthreads()` 猜测修复，因为同步点若也只被部分 threads 到达，会产生第二个死锁。

第二类错误是让每个 block 都调用同一个 world collective。collective launch 可能保证所有 blocks 驻留，却无法满足“每 PE 恰好一个 collective 实例”。检查 kernel 中 collective 调用与 `blockIdx` 的关系：如果 `gridDim.x > 1`，通常需要指定唯一 block 代表该 PE，或者把 blocks 映射到互不冲突的 Teams；代表 block 内仍要满足 thread/warp/block 作用域。

第三类错误是普通 launch 中让 block 等待尚未调度 block 或远端 PE。小消息、小 grid 或空闲 GPU 上可能偶尔成功，放大 grid、增加 shared memory 或与其他作业共存后才挂起。症状随资源压力变化时，应优先审查前进性和 collective launch，而不是把问题全归因于网络。查询 cooperative gridsize、缩小并固定 grid、记录每 block 资源可以帮助验证这一层。

第四类错误是跨 stream 缺边或形成环。目标读取旧 source 时，检查 producer event 是否真的记录在 producer 后，通信 stream 是否在 PUT 前等待；程序全局挂起时，将 `cudaStreamWaitEvent`、kernel wait 和 signal 方向画成依赖图。若图中只有环没有起点，增加 device synchronization 可能只改变症状，不会恢复协议。

第五类错误是混用完成域。block NBI PUT 后由 CPU 立即调用 host quiet，或 on-stream PUT 返回后马上释放 source，都没有证明正确的 GPU 发起域到达完成点。应先确定操作由 device thread 还是 stream 节点发起，再在同一域安排 flush/quiet，最后用 kernel/stream completion 将结果交给 CPU。

## 一套可复用的审查流程

审查设备端 NVSHMEM kernel 时，可以沿以下顺序建立证明。这个列表是诊断次序，不是要求代码必须机械分成六层。

1. 画出 PE 与 Team：列出每个 collective 的参与 PEs、Team handle 和跨 PE 程序顺序。
2. 标出本地调用实例：每个 PE 上由哪一个 thread、warp 或 block 代表；是否恰好一个实例；thread-group 参数是否一致。
3. 标出生产和消费：source 由哪些 threads 写，dest 由哪些 threads 读；对应的 `__syncwarp()`、`__syncthreads()` 或 group sync 是否覆盖全部参与者。
4. 标出 CUDA 队列边：同 stream 顺序、event record/wait、host 对 stream/kernel 的完成观察，检查是否存在环。
5. 标出 NVSHMEM 完成边：blocking/NBI、fence/quiet/flush、signal/wait 与 ack，确认没有用 CUDA event 代替远端交付。
6. 验证前进性：凡出现设备端 wait、barrier 或 collective，确认使用 collective launch、grid 不超查询上限，并检查 GPU 独占与 MPG 条件。

只有六层都闭合，程序才同时具备数据正确性与可执行性。地址正确但 block 永远无法调度，结果仍是挂起；所有 blocks 都能驻留但两个生产者同时覆盖一个 slot，结果仍是数据错误；quiet 完成远端交付但消费者不回 ack，下一轮仍可能覆盖正在处理的数据。

## 诊断题与参考推理

### 问题一：256-thread block 中只有 thread 0 调用 `nvshmemx_putmem_block`，是否等价于由 block 发送？

不等价。`_block` 是 block collective API，要求整个参与 block 共同调用并使用相同参数。若只希望 thread 0 发起，应调用适合单 thread 的 device API，并在 source 由其他 threads 产生时先用 block 同步完成交接；若希望并行协作，则让整个 block 收敛进入 `_block` API。

### 问题二：两个 blocks 都调用 `nvshmemx_barrier_all_block`，而 kernel 使用 collective launch，为什么仍然错误？

collective launch 保证调度前进性，但同一 Team 的 collective 要求每个 PE 同时只有一个调用实例。两个 blocks 让一个 PE 并发提供两个 world-barrier 实例，破坏 collective matching；除非它们作用于经证明互不冲突的不同 Teams，否则不能这样组织。

### 问题三：普通 launch 的 kernel 只做 PUT，不做 wait 或 collective，是否一定要 collective launch？

按当前 NVSHMEM 文档，不使用同步或 collective device API 的 kernel 可以普通启动。它仍必须遵守对称寻址、thread-group 参与、源数据交接、RMA 完成和 ownership；若后来加入 wait/barrier，就必须重新审查并改用 collective launch。

### 问题四：stream A 生成 source，CPU 随后把 PUT 排到 stream B；调用顺序先后是否足够？

不足。不同 streams 没有默认执行顺序。应在 A 的 producer 后记录 event，让 B 在 PUT 前等待；若要复用 source，还要在 B 中排入所需的 NVSHMEM completion 操作，并把完成 event 反馈给下一轮 producer。CPU 调用两个 enqueue API 的先后不等于 GPU 跨 stream 的执行先后。

### 问题五：collective launch 查询出的最大 grid size 能否保存为所有 GPU 和所有 kernel 通用常量？

不能。上限依赖当前 GPU、block 尺寸、kernel 寄存器使用和动态 shared memory 等资源。kernel 改动、编译目标或 launch 配置变化都可能改变驻留能力，应针对实际函数和参数重新查询并检查返回值。

### 问题六：device API 支持 `THREAD_MULTIPLE`，是否意味着多个 threads 能同时在同一 Team 上调用 reduce？

不意味着。thread support 描述并发调用的库线程安全边界；team collective 另有“同一 Team 每 PE 仅一个实例、所有 PEs 顺序一致”的更强约束。普通、warp 和 block 版本应分别由恰好一个 thread、warp 或 block 代表该 PE。

### 问题七：`__syncthreads()`、collective launch 和 `nvshmem_quiet()` 能否互换？

不能。`__syncthreads()` 只协调一个 block 内 threads 并建立本地内存顺序；collective launch 保证跨 PE kernel 的 cooperative 启动与驻留前进基础；quiet 完成调用域中先前发起的 NVSHMEM 操作。一个正确流程可能同时需要三者，但每个原语只填补自己的依赖边。

## 掌握标准与后续阅读

完成本课后，读者应能对任意 device-side NVSHMEM 调用回答：它由一个 thread、warp 还是 block 发起；所有参与 threads 是否收敛并使用一致参数；source 生产与通信、通信与控制 thread 之间是否有本地交接；对应 kernel 是否包含同步 API 并需要 collective launch；每个 Team 在每个 PE 上是否恰好一个匹配实例；CUDA streams、events 与 NVSHMEM ordering 是否分别建立了正确的两层依赖。

读者还应能画出至少一个前进性等待环，解释为何普通 CUDA 调度不保证尚未驻留 block 获得执行机会，并指出 collective launch 只能解决驻留类风险，不能修复缺失 signal、错误 generation 或重复 collective。若只能说“用 cooperative launch 就不会死锁”，尚未达到掌握标准。

下一步 `B04：数据发布与缓冲区所有权` 将把 B02 的六阶段完成模型和本课的执行域组合起来，构造 payload/ready/ack、generation、单槽与双槽协议。B04 正式文件尚未创建，因此这里只记录课程名称而不建立失效链接。配套实验 `experiments/experiment-b03-thread-block-cooperation.md` 仍为规划，本文没有替代编译和硬件验证。

## 主要来源

- [NVSHMEM Overview of the APIs](https://docs.nvidia.com/nvshmem/api/api/overview.html)：host/device API 分类、thread/warp/block RMA、stream API、collective launch 和 thread-group flush 的概览。
- [NVSHMEM Remote Memory Access](https://docs.nvidia.com/nvshmem/api/latest/gen/api/rma.html)：thread、warp 与 block 版本的 PUT/GET/NBI 原型、参数和完成语义。
- [NVSHMEM Collective Communication](https://docs.nvidia.com/nvshmem/api/latest/gen/api/collectives.html)：Team collective 的参与顺序、同一 Team 并发限制、每 PE 一个 thread/warp/block 实例及不同 Team 示例。
- [NVSHMEM Kernel Launch Routines](https://docs.nvidia.com/nvshmem/api/latest/api/launch.html)：collective launch 的 collective/cooperative 属性、设备能力、grid size 查询与上限。
- [NVSHMEM and the CUDA Model](https://docs.nvidia.com/nvshmem/api/latest/cuda-interactions.html)：stream/event 依赖、非本地阻塞、CUDA 非抢占调度和 cooperative launch 的前进性原因。
- [Using NVSHMEM](https://docs.nvidia.com/nvshmem/api/latest/using.html)：含 synchronization/collective API 的 kernel 必须 collective launch，以及一 PE 一 GPU和 GPU 独占的指导。
- [Library Setup, Exit, and Query](https://docs.nvidia.com/nvshmem/api/latest/gen/api/setup.html)：host/stream 与 device thread support levels、blocking thread 行为和多线程 collective 顺序责任。
- [NVSHMEM Examples](https://docs.nvidia.com/nvshmem/api/latest/examples.html)：thread-group PUT 以及 CTA NBI PUT 后同步、单 thread fence/signal 的官方示例结构。
- [NVSHMEM 3.7.2 Release Notes](https://docs.nvidia.com/nvshmem/release-notes-install-guide/release-notes/release-3720.html)：本课核对时的当前发布周期；API Guide `latest` 仍是滚动文档，不代表固定源码 commit。
- [CUDA Programming Guide: Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)：blocks 可按任意顺序执行、不能默认依赖其他 block 获得调度。
- [CUDA Programming Guide: Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)：stream 的队列顺序、跨 stream event 依赖和 host 异步关系。
- [CUDA Programming Guide: Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)：group collective 参与、同步与本地内存可见性。
- [CUDA Programming Guide: SIMT Execution Model](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)：warp divergence、active threads 与 independent thread scheduling。
