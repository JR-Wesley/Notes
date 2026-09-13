# B02 NVSHMEM 完成、排序与可见性

## 本课要解决的核心问题

[B01：NVSHMEM 对称对象与远端寻址](b01-symmetric-objects-and-remote-addressing.md)解决了“远端对象在哪里”：发起 PE 使用本地 symmetric address 与目标 PE 共同标识远端对应位置。本课继续回答更容易出错的问题：调用返回时，操作究竟走到了哪一步；源缓冲区是否已经可以覆盖；目标内存是否已经收到数据；目标 PE 通过什么事件得知数据可读；目标消费结束后，生产者何时才能再次使用同一个远端槽位。

这些问题不能用一个模糊的“完成”概括。NVSHMEM 的 blocking、nonblocking immediate（NBI）、`fence`、`quiet`、`flush`、signal、wait/test、barrier 和 sync 分别建立不同边界，且 host、device 和 CUDA stream 发起的操作属于不同执行域。把 `fence` 当成完成、把 `quiet` 当成远端通知、把 barrier 当成能自动排空任意 stream，都会得到偶尔可运行但没有 API 保证的程序。

## 资料版本与证据边界

本文在 2026-09-12 核对 NVIDIA 官方 NVSHMEM API Guide 的滚动 `latest` 文档，并以 [NVSHMEM 3.7.2 Release Notes](https://docs.nvidia.com/nvshmem/release-notes-install-guide/release-notes/release-3720.html)标识当前发布周期。主要契约来自 Memory Ordering、Remote Memory Access、Signaling Operations、Point-To-Point Synchronization、Collective Communication、NVSHMEM and the CUDA Model 以及官方 FAQ。滚动页面不是固定源码版本，因此本文不把 API 语义映射为某个确定的 WQE、CQE、proxy 请求或 IBGDA 完成记录。

文中的代码是教学片段，未编译、未运行；没有 GPU、NVLink、InfiniBand 或 IBGDA 实测结果。关于“请求可能经过队列、proxy 或 NIC”的描述只用于解释为何调用返回与远端消费不同步，具体 transport 如何实现 `quiet`、是否使用 CQ、使用几条请求以及如何批处理，必须在后续固定 NVSHMEM commit 的源码课中核验。

## 先把一次传输拆成六个阶段

假设 PE 0 把本地 `source` 中的 payload 写入 PE 1 的对称槽位 `dest`，PE 1 随后消费数据，最后允许 PE 0 覆盖同一个槽位。为了判断每个 API 证明了什么，可以把协议拆成六个逻辑阶段：

| 阶段 | 含义 | 尚不能推出 |
| --- | --- | --- |
| S0：源数据就绪 | 产生 payload 的 CPU thread、CUDA thread 或 stream 已完成对 `source` 的写入，并与通信发起者建立顺序 | NVSHMEM 请求已经发起 |
| S1：请求已发起 | runtime 已接受或已将操作排入相应执行域 | 源缓冲区安全复用、远端已收到 |
| S2：本地完成 | transport 不再需要读取 PUT 的本地源，或 GET 的本地目标已经取得结果 | 对 blocking PUT 而言不一定表示远端交付；也不表示远端消费 |
| S3：远端交付 | payload 已送入目标 PE 的 `dest` | 目标控制流已经观察、开始或结束消费 |
| S4：目标观察并开始消费 | 目标通过 signal/wait、barrier 或其他协议确认数据可读 | 生产者已经知道消费结束 |
| S5：消费确认 | 目标发送 ack 或推进序号，生产者观察后取得槽位所有权 | 下一代数据天然有序；仍需按协议发布 |

这六个阶段是诊断模型，不是 NVSHMEM 公开的内部状态机。不同 API 可以一次跨越多个阶段，例如 blocking GET 返回时本地结果已经到达；也可能只建立顺序关系而不推动完成，例如 `fence`。最重要的是，S3 与 S5 相隔一个完整的目标侧消费过程：任何仅证明数据交付的 API，都不能单独证明远端槽位可被下一轮覆盖。

## blocking 与 NBI 描述的不是同一种返回边界

NVSHMEM 的 blocking RMA 名称容易产生误解。“blocking”通常只相对于调用 PE 的本地 operand 和返回值定义，并不普遍意味着目标 PE 已经完成业务处理。当前 RMA 契约规定，blocking PUT 在本地 `source` 的数据已经被复制出去后返回，所以调用返回后源缓冲区可以复用；但是数据字可能仍在向目标对象交付，目标 PE 更没有因此得到控制流通知。

```cpp
// 教学片段，未编译、未运行。
prepare(source);
nvshmem_putmem(remote_dest, source, bytes, peer);
// blocking PUT 返回后 source 可复用；不能据此断言 peer 已开始消费。
reuse(source);
```

blocking GET 的方向不同：它从远端 symmetric source 读取到本地 `dest`，并在数据已经交付到本地目标后返回。因此调用者在返回后可以读取 GET 结果。标量 `g` 和 fetching AMO 也必须在返回前取得返回值。non-fetching AMO 则更像远端更新：调用可能在目标执行前返回，需要 `quiet`、barrier 等机制强制完成。

带 `_nbi` 的 nonblocking immediate 操作在发起后即可返回。PUT NBI 返回不证明 source 已被复制完，GET NBI 返回不证明本地 dest 已装入结果；后续 `nvshmem_quiet()` 才完成调用 PE 先前发起的相应 NBI 操作。若唯一目标是让 NBI PUT 的 source 尽早复用，当前 API 还提供 `nvshmemx_flush*` 建立本地完成边界，但它不提供远端可见性。

| 操作 | 普通调用返回时 | 需要 `quiet` 的典型原因 | 远端消费者仍需要 |
| --- | --- | --- | --- |
| blocking PUT/`p` | 本地 source 可复用；远端交付未必完成 | 要求此前远端更新完成，或跨目标建立更强顺序 | signal/wait、barrier 或应用通知 |
| PUT NBI | 只保证发起；source 不能据此复用 | 完成本地 source 使用并完成远端交付 | 目标侧观察协议 |
| blocking GET/`g` | 本地结果已取得 | 该 GET 本身通常不需要 | 无远端消费，但本地后续使用要有线程/stream 顺序 |
| GET NBI | 只保证发起；本地 dest 尚不可读 | 让结果交付到本地 dest | 调用 PE 在 quiet 后使用结果 |
| non-fetching AMO | 远端更新可能尚未执行 | 强制远端原子更新完成 | 若目标控制流依赖结果，仍需 wait/test 等观察 |
| fetching AMO | 返回值已取回并完成该 fetch/update | 用于其他先前未完成操作，不是取得本次返回值的前提 | 按协议处理远端新值 |

这张表描述 API 层边界，不代表每种 transport 都在调用返回时执行相同硬件动作。尤其不能把 blocking PUT 简化成“内部一定等待一条 CQE”，也不能把 NBI 理解为“完全不做工作”：它已经建立了一个之后必须由完成机制覆盖的在途操作。

## `fence` 只排序，不等待完成

`nvshmem_fence()` 保证调用 PE 在 fence 之前发往某一目标 PE 的受支持访问，先于 fence 之后发往同一目标 PE 的访问交付。它解决的是“目标看到 A 和 B 时，不能先看到 B 后看到 A”，不解决“调用返回时 A 是否已经到达”。fence 是 per-target ordering；若前后操作发往不同目标 PE，不能用它推导跨目标完成或观察关系。

```text
PE 0                                  PE 1

PUT payload ───────────────┐
fence                      ├── 保证 PE 1 不先观察后续 ready、再观察 payload
write ready ───────────────┘

fence 返回时，payload 和 ready 都可能仍在途。
```

`fence` 适合“payload 后发布 ready”的同目标协议。例如先 PUT 数据，再 fence，最后对同一 PE 的 ready 对称对象执行远端更新。目标侧还必须 wait/test 自己本地的 ready，才能建立读取 payload 的控制边界。没有目标侧观察，源 PE 调用 fence 本身不会唤醒任何人。

fence 也不是所有操作的通用排序器。当前 Memory Ordering 表明确排除 nonblocking GET 和 nonblocking fetching AMO 的返回值排序；这些操作需要 completion，而不是只安排远端更新顺序。CPU 上的 fence 只排序 CPU 发起的通信，GPU device 上的 fence 只排序 GPU 发起的通信。多个 host threads 或 GPU threads 共同发起操作时，还必须先用相应线程模型保证这些操作已经对执行 fence 的线程“发起可见”，然后才能由一个线程代表此前工作执行 fence。

## `quiet` 完成调用 PE 的既有操作，但不是 rendezvous

`nvshmem_quiet()` 是本地、非集合操作。它保证调用 PE 在相应执行域中先前发起的受支持 symmetric-object 操作完成，并对所有目标 PE 建立比 fence 更强的前后顺序。当前文档将可见性保证限定在目标 PE：quiet 返回后，先前更新已经交付给相应目标，但其他 PE 不会因为这个调用自动停下、收到消息或与调用者会合。

```text
PE 0: PUT A to PE 1
      PUT B to PE 2
      quiet
      └─ A、B 的相应远端更新完成；PE 0 可进入下一本地阶段

PE 1/PE 2:
      不会自动执行 wait，也不会自动知道 PE 0 已调用 quiet
```

quiet 的作用域必须从“谁发起”判断。host 上的 `nvshmem_quiet()` 只完成 host 发起的操作；device code 中的 `nvshmem_quiet()` 完成 GPU 发起的操作，并非只限于承载该 kernel 的某一个 CUDA stream，但不会排空 host 发起的操作。`nvshmemx_quiet_on_stream(stream)` 把 GPU-side quiet 排入指定 stream，在该 stream 顺序中完成此前应由它覆盖的 GPU 侧操作。

多个线程共同提交时，quiet 只能覆盖在它之前确实已经发起的操作。若一个 CTA 使用 block-scoped PUT，然后由 thread 0 调用 quiet，CTA 必须先完成 thread-group API 并建立 `__syncthreads()` 等交接，使所有请求在 thread 0 执行 quiet 前已经发起。官方文档允许这种“多线程发起、同步、单线程 quiet”的模式；错误做法是让 thread 0 提前 quiet，而其他线程之后才 post。

quiet 不是消费确认。它达到 S3 后，远端 kernel 可能尚未调度、尚未 wait、尚未读取 payload，甚至可能永远不消费。如果源 PE 想覆盖同一个远端 slot，必须等待目标 PE 的 ack 或所有权序号，而不能仅在 quiet 返回后开始下一代覆盖。

## `flush` 只解决 PUT source 的本地复用

当前 NVSHMEM Memory Ordering API 提供 `nvshmemx_flush`、warp/block 变体和 `nvshmemx_flush_on_stream`。它们确保调用 PE 先前 outstanding nonblocking PUT 达到本地完成，使对应 source buffer 可以安全复用、覆盖或释放。flush 不保证数据在目标 PE 可见，也不替代 fence 或 quiet。

```text
PUT NBI source usage:

post PUT NBI
   ├─ flush  ──> source 可复用；remote dest 可能尚不可见
   └─ quiet  ──> source 可复用，且相应 remote update 已完成
```

因此 flush 适用于“源缓冲区昂贵，需要尽早回收，但远端消费发生在稍后协议阶段”的场景。若下一步是向目标发布 ready，就不能仅凭 flush 认为 payload 已经可读；应使用 quiet，或用满足契约的 ordering 与 signaling 组合。对于 IB/RoCE，文档也提示 flush 可能需要类似 transport drain 的代价，不能在没有测量时假定它总比 quiet 便宜。

warp/block flush 是 GPU thread-group collective，参与范围内每个线程必须按相同控制流调用。若一个线程要代表其他线程执行单线程 flush，仍需先建立请求发起的线程间顺序。B03 会进一步讨论这类 thread/warp/block 交接。

## signal 把“本次 payload 已交付”变成目标可观察事件

普通 PUT 与 quiet 都不主动改变远端控制流。signaling operations 提供一个远端 `uint64_t` signal object，目标 PE 可以通过 signal wait/test 观察。PUT-with-signal 将“复制本次 `dest` payload”与“随后原子更新 `sig_addr`”组合；当目标观察到对应 signal update 完成时，API 保证与该次 put-with-signal 配对的 payload 已交付到远端 dest。

```cpp
// 教学片段，未编译、未运行。dest 与 ready 都是目标 PE 的对称对象地址。
nvshmem_putmem_signal(dest, source, bytes,
                      ready, sequence,
                      NVSHMEM_SIGNAL_SET, peer);
```

`dest` 与 `sig_addr` 必须都是 remotely accessible symmetric objects，二者不能重叠。signal update 对与 signal API、signal-fetch 以及相应 wait/test 的并发访问具有文档规定的原子性；这不等于 payload 的每个字节写入具有原子事务语义。目标必须避免在 signal 到达前读取正在被写入的 payload。

PUT-with-signal 只为“它自己携带的 payload”提供 signal 后交付保证。假设程序先普通 PUT `header`，再 put-with-signal 写 `body`，目标观察到 body 的 signal 并不能自动推出更早的 header 已交付。若二者发往同一目标，可在两次操作间用 fence 建立顺序；若要求之前所有相关操作先完成，可用 quiet 后再发布 signal。必须按真实依赖选择，不能因为最后一次操作带 signal 就把它当成整个 PE 历史的隐式 barrier。

`NVSHMEM_SIGNAL_SET` 写入一个 sequence 值，`NVSHMEM_SIGNAL_ADD` 原子累加。单槽协议若反复使用布尔值 0/1，容易遇到旧轮次与新轮次无法区分的 ABA 问题；单调 generation 或带回绕证明的序号更容易诊断。signal 证明的是远端交付与通知，不证明消费者已经完成，所以仍需要反向 ack。

## wait/test 观察的是调用 PE 的本地对称对象

目标 PE 调用 `nvshmem_wait_until(local_ready, ...)` 或 signal 专用 `nvshmem_signal_wait_until(local_signal, ...)` 时，传入的是目标 PE 自己的 local symmetric address。wait 不是主动去源 PE GET 一个 flag，而是等待其他 PE 对本地对象的远端更新达到条件。test 执行一次非阻塞检查，适合在轮询间穿插其他工作；wait 阻塞调用线程直到条件成立。

```cpp
// 目标 PE 上的教学片段，未编译、未运行。
std::uint64_t seen =
    nvshmem_signal_wait_until(local_ready, NVSHMEM_CMP_GE, expected);
consume(local_payload);
```

官方契约要求 signal wait 不得在 signal memory update 完成前返回；put-with-signal 又将该 signal 与对应 payload 的交付关联起来，因此目标在 wait 返回后可以进入该代 payload 的消费阶段。这个推理只覆盖相应 signal 所代表的数据，不能扩展到未排序的其他 transfer。

generic wait/test 对象和 signal object 的并发写入规则不同。signal wait 文档期望 signal object 只通过 signaling operations 更新；不要同时用普通 store、普通 PUT 和 signal API 无约束地写同一个 flag。对相同 memory location 的冲突访问若不属于文档允许的 AMO 或 signal/wait 组合，可能构成未定义行为。

等待完成后，消费者还需向生产者返回 ack。反向 ack 可以是另一个 put-with-signal、signal update 或经证明的 AMO；生产者等待自己本地 ack 达到本轮 sequence 后，才从 S4 进入 S5 并重新取得 slot 所有权。payload-ready 与 consumed-ack 是两个方向、两个对象或两个明确状态，不能用同一个布尔值在无所有权规则下反复抢写。

## 一条完整的单槽协议

下面的时序把地址、交付、通知和复用连成闭环。它是协议结构示意，并非已运行程序：

```text
生产者 PE 0                                      消费者 PE 1

等待 local_ack >= generation
填充 local_source                         S0
put-with-signal(remote_slot,
                local_source,
                remote_ready=generation) ────────────────>
                                                payload 交付 S3
                                                ready 原子更新
                                      等待 local_ready >= generation
                                      读取并处理 local_slot       S4
                                      signal(remote_ack=generation) ──┐
收到 local_ack >= generation  <───────────────────────────────────────┘
允许覆盖 source/remote slot，generation++                    S5
```

若生产者使用 PUT NBI 加独立 ready，至少要证明 source 写入先于 PUT 发起、PUT payload 先于 ready 被目标观察；可以使用 PUT NBI→quiet→signal，或在相同目标且只要求 delivery ordering 时使用满足 API 契约的 fence 组合。消费者必须在处理完成后发 ack。双槽协议只是允许两个 generation 分别占用不同 slot，并没有消除每槽的 ready/ack 状态。

## barrier 与 sync：都有会合，完成语义不同

`nvshmem_barrier(team)` 是 team collective。所有成员必须以兼容顺序参与；控制返回前，它完成 team 内 PEs 先前发起的 stores 和相应 NVSHMEM RMA/AMO remote updates，并让所有成员跨过同一会合点。它适合真正的全组阶段切换，但会把不相关 PE 纳入关键路径，不能代替只涉及邻居的 signal/wait。

`nvshmem_sync(team)` 也让 team 成员会合，但当前契约只确保先前 memory stores 的完成与可见，不完成通过 NVSHMEM routines 发起的 remote updates。若使用 sync 协调一轮 RMA，发起 PE 应先 quiet 完成远端更新，再进入 sync。把 sync 当成廉价 barrier 而省略 quiet，会丢失 RMA completion 保证。

| 原语 | 是否集合 | 控制会合 | 完成此前 NVSHMEM RMA/AMO remote updates | 典型用途 |
| --- | --- | --- | --- | --- |
| `fence` | 否 | 否 | 否 | 同一目标的数据→通知顺序 |
| `quiet` | 否 | 否 | 是，限调用 PE/相应发起域 | 批量完成、NBI GET 结果、远端交付 |
| `flush` | 否 | 否 | 否，只做 NBI PUT 本地完成 | 尽早复用 source |
| signal + wait/test | 点对点协议 | 只阻塞/检查目标调用者 | signal 只证明对应契约范围 | 局部生产者—消费者通知 |
| `barrier(team)` | 是 | 是 | 是，针对参与 PEs 的此前相关更新 | 全 team 阶段切换 |
| `sync(team)` | 是 | 是 | 否 | 已自行完成通信后的轻量控制会合 |

所有 collective 都要求同一 team 的成员以匹配顺序参与。只有部分 PE 进入 world barrier、不同 PE 交换两个 team collective 的顺序，或者让同一 PE 上多个调用者无协议地并发发起同一组 collective，都可能死锁或违反契约。官方 FAQ 还指出，同一 PE 集合上不应同时存在多个 in-flight collective；需要并发邻域同步时，应缩小 team 或设计点对点协议，而不是重叠 world barriers。

## host、device 与 on-stream 是不同发起域

Host API 由 CPU thread 同步调用；device API 在 CUDA kernel 内由 GPU thread 调用；on-stream API 由 CPU 发起，但只是把相应 GPU 侧操作排入 CUDA stream，并立即返回。on-stream 函数的 host 返回只证明 enqueue 成功，不证明该操作已经到达 stream 头部，更不证明网络传输完成。

```text
CPU 调用 on-stream PUT 返回
        │
        └─ 只表示操作已排入 stream
             ↓ 等待前序 CUDA work
        NVSHMEM 操作实际执行
             ↓
        quiet_on_stream / barrier_on_stream 等后续节点
             ↓
        cudaStreamSynchronize 让 CPU 观察 stream 已完成
```

同一 stream 提供 CUDA 队列顺序，但不同 streams 可以并发执行。若 stream A 产生 source、stream B 发起 NVSHMEM PUT，必须用 CUDA event 让 B 等待 A；若 stream C 要消费 GET 结果，则让 C 等待完成该 GET/quiet 的 event。仅因为三条 stream 属于同一 PE，不会自动建立先后关系。

CPU 上直接调用 `nvshmem_quiet()` 不能排空尚在 CUDA stream 中等待执行的 on-stream 操作。官方 FAQ 明确指出，存在 in-flight on-stream PUT 时，host `nvshmem_barrier_all()` 也不负责完成这些 stream 操作；应先用 `cudaStreamSynchronize` 使相应 on-stream 工作完成，或在同一 stream 中安排 `nvshmemx_quiet_on_stream`、`nvshmemx_barrier*_on_stream` 等正确节点，再由 CPU 同步该 stream。

device `nvshmem_quiet()` 完成 GPU-initiated operations，而非 host 发起操作。若 CPU 必须观察 device 发起通信完成，GPU 侧需要执行 quiet，并且 CPU 还要通过 kernel completion、`cudaStreamSynchronize` 或 `cudaDeviceSynchronize` 确认执行到该点；或者用 stream-based quiet 在 CUDA 顺序中建立等价边界。NVSHMEM ordering API 与 CUDA synchronization 各管理一层依赖，不能只做其中一层。

## CUDA kernel 中的阻塞等待与 forward progress

device-side wait、barrier 和 collective 可能阻塞正在执行的 GPU threads。如果某个 kernel 等待另一个尚未被调度的 block，或多个 PE 共享 GPU 而相关 kernel 不能同时驻留，就可能形成 forward-progress 环。包含 NVSHMEM 同步或 collective API 的 kernel 必须遵循 collective launch 及网格驻留约束；普通 CUDA launch 不能因为测试规模较小时碰巧成功，就被视为有保证。

即使没有跨 PE 死锁，线程交接也必须正确。一个 warp/block cooperative PUT 返回后，如果仅 thread 0 负责 fence、quiet 或 signal，参与线程应先完成 thread-group API 并同步，使所有请求在控制线程继续之前已发起。反过来，不能让非参与线程调用 block-scoped ordering API，也不能让部分 active threads 因分支跳过 collective thread-group call。详细执行模型放在后续 B03，本课只保留与完成域直接相关的条件。

## 多目标、多生产者与覆盖问题

fence 是 per-target；quiet 对调用 PE 的先前访问提供跨目标完成和顺序。如果 PE 0 向 PE 1 写 payload，却向 PE 2 写 ready，单纯 fence 不能让 PE 2 的 ready 证明 PE 1 已收到数据。quiet 后再通知 PE 2 可以先完成之前的更新，但 PE 2 如何代表 PE 1 的消费状态仍是额外的分布式协议，不能由 memory ordering 自动推出。

多个生产者同时写同一目标 slot 时，即使每个生产者内部都执行 put-with-signal，payload 仍可能互相覆盖，signal 的先后也未必对应消费者希望的所有权次序。正确设计需要通过 AMO 预留独立 slot、每生产者队列、generation/owner 字段或集中调度消除写冲突。ordering 原语只排列已被允许的访问，不授予写所有权。

同一生产者连续复用单槽也必须等待 ack。sequence-ready 只能证明第 N 代已交付；若生产者在消费者处理第 N 代期间写入第 N+1 代，消费者可能读到混合数据。quiet 不能解决这个问题，因为第 N 次远端交付完成并不等于第 N 次远端读取完成。B04 将用单槽、双槽和最终排空状态机专门证明这一点。

## 与 verbs 完成模型的对应和边界

[A03：RDMA 请求、完成与缓冲区协议](a03-rdma-requests-and-buffer-protocol.md)中的 WC 是 verbs 应用显式观察的请求结果；[A04](a04-rping-full-rdma-flow.md)中服务端 RDMA WRITE WC 只证明写操作在发起侧完成，随后还要 SEND 一枚令牌通知客户端。NVSHMEM 的 quiet 与 put-with-signal 在语义层分别承担“完成此前操作”和“交付 payload 后发布通知”的角色，但不能据此断言 runtime 一定为每次调用等待一条 verbs CQE。

不同 transport 可能使用 NVLink direct load/store、CPU proxy、IBRC、UCX、IBGDA 或其他队列路径。API 层的正确推理只依赖公开边界：源何时可复用、远端更新何时完成、通知何时可观察、目标何时返回 ack。等到 D 模块选择固定 commit 后，才可以把这些边界对应到实际 WQE、doorbell、CQ、counter 或 proxy state。

## 常见错误的逐层诊断

当目标读到旧数据时，先检查 S0：产生 source 的 kernel/stream 是否真的先于通信；再检查 PUT 是 blocking 还是 NBI、是否只有 flush 而没有远端 completion；然后检查 payload 与 ready 是否同目标并有 fence/quiet 或 put-with-signal 契约；最后检查目标是否等待自己的 local symmetric flag。不要直接加入 `cudaDeviceSynchronize()` 和 world barrier 掩盖依赖，因为那可能让程序暂时工作，却无法说明缺失的是哪条边。

当 NBI GET 结果偶尔错误时，检查是否在 quiet 前读取本地 dest；fence 不完成 NBI GET，也不排列其 fetched values。当 producer 覆盖尚未消费的数据时，检查是否把 quiet 或 ready 错当成 ack；增加第二个 consumed sequence，而不是继续加强 producer 侧 fence。

当程序挂起在 barrier/wait 时，检查 team 参与集合、collective 顺序、kernel launch 与并发驻留，以及 signal 是否写到目标 PE 的正确 local symmetric object。若 host barrier 等待前有 on-stream 通信，确认相应 stream 已执行到完成点。若只有跨节点失败，再结合 B01 检查 symmetric address、transport 和注册，不要把所有 hang 都归因于 ordering。

## 诊断题与参考推理

### 问题一：blocking PUT 返回后，可以立即覆盖本地 source 吗？可以立即让目标读取吗？

本地 source 可以按 blocking PUT 契约复用，因为调用在数据被复制出 source 后返回；目标读取仍需要远端交付与通知协议。可以用 quiet 完成此前更新，再发 signal，也可以使用 put-with-signal 将对应 payload 与 signal 关联。

### 问题二：PUT NBI 后调用 flush，再写 ready 是否安全？

flush 只证明 source 可复用，不证明 remote dest 可见。若 ready 被目标当作 payload 可读证据，这个组合不充分。应使用 quiet 或满足同目标 ordering 契约的 fence/put-with-signal 方案，并由目标 wait/test ready。

### 问题三：PE 0 依次向 PE 1 写 payload、fence、向 PE 2 写 flag，PE 2 看到 flag 能否证明 PE 1 已收到 payload？

不能。fence 只建立对同一目标 PE 的 point-to-point ordering。即便改成 quiet 后通知 PE 2，也只证明交付先后；PE 2 是否可以代表 PE 1 的业务状态，需要额外协议。

### 问题四：device kernel 发起 PUT 后，CPU 调用 host `nvshmem_quiet()` 是否足够？

不够。host quiet 不完成 GPU-initiated operations。GPU 侧应执行 device quiet，CPU 再等待 kernel/stream；或者在正确 stream 中安排 `nvshmemx_quiet_on_stream` 并同步该 stream。

### 问题五：消费者观察到 put-with-signal 的 sequence 后，生产者能否立即覆盖远端 slot？

不能。sequence 只证明对应 payload 已交付并允许消费者开始读取。生产者必须等待消费者完成处理后返回的 ack/generation，才能重新取得 slot ownership。

### 问题六：为什么 `nvshmem_sync(team)` 不能无条件替代 barrier？

sync 提供 collective control rendezvous，并完成/显示此前 memory stores，但不完成通过 NVSHMEM routines 发起的 remote updates。若本轮有 RMA/AMO，发起 PE 需要先 quiet，或直接使用具有相应完成语义的 barrier。

## 掌握标准与后续阅读

完成本课后，应能针对 PUT、GET、NBI 和 AMO 分别指出调用返回、本地 operand 复用与远端交付边界；能在同目标协议中选择 fence，在批量完成中选择 quiet，在只需 PUT source 复用时识别 flush；能用 put-with-signal + wait 建立 ready，并用反向 ack 证明消费结束；还能判断 host、device 和 on-stream ordering API 分别覆盖哪一类发起操作。

读者还应能在纸上画出 S0—S5，并指出每条边由 CUDA event/thread synchronization、NVSHMEM ordering、signal/wait 或应用 ack 中哪一种机制建立。如果只能说“这里加一个同步”，还不足以证明程序正确。下一步 B03 将把这些边界放进 thread、warp、block、stream 和 collective launch；B04 再用单槽/双槽协议处理所有权、背压和最终排空。

## 主要来源

- [Memory Ordering](https://docs.nvidia.com/nvshmem/api/latest/gen/api/ordering.html)：`fence`、`quiet`、`flush` 的 ordering/completion 范围及 host/device/thread 交接规则。
- [Remote Memory Access](https://docs.nvidia.com/nvshmem/api/latest/gen/api/rma.html)：blocking PUT/GET、NBI PUT/GET 的返回与完成边界。
- [Signaling Operations](https://docs.nvidia.com/nvshmem/api/latest/gen/api/signal.html)：put-with-signal、signal 原子更新、对应 payload 交付保证及跨 transfer 限制。
- [Point-To-Point Synchronization](https://docs.nvidia.com/nvshmem/api/latest/gen/api/sync.html)：wait/test 对本地 symmetric object 的观察语义。
- [Collective Communication](https://docs.nvidia.com/nvshmem/api/latest/gen/api/collectives.html)：barrier、sync、team 参与范围及 CPU/GPU ordering 域。
- [NVSHMEM and the CUDA Model](https://docs.nvidia.com/nvshmem/api/latest/cuda-interactions.html)：on-stream enqueue、跨 stream CUDA event 与 GPU/CPU 交互。
- [Using NVSHMEM](https://docs.nvidia.com/nvshmem/api/latest/using.html)：弱排序、目标观察与 forward progress 的整体约束。
- [Troubleshooting and FAQs](https://docs.nvidia.com/nvshmem/api/latest/faq.html)：host barrier 不排空 in-flight stream 操作、device quiet 作用域和 collective 并发限制。
