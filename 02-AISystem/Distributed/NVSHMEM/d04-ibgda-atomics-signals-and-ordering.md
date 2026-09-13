# IBGDA 原子操作、Signal 与排序：从 API 语义到 WQE 和一致性闭环

IBGDA 不只是把 GPU 发起的 PUT/GET 改写成 NIC work request。原子操作必须同时处理远端 read-modify-write、fetching 返回值、结果缓冲区复用和 GPU 可见性；put-with-signal 还必须保证接收方观察到 signal 时，关联 payload 已经可以安全消费；而 `fence`、`quiet` 与 consistency 操作解决的又是不同层次的问题。本篇在 [D00](d00-ibgda-implementation-overview.md)、[D01](d01-ibgda-request-lifecycle.md)、[D02](d02-ibgda-concurrency-and-queue-management.md)和 [D03](d03-ibgda-rc-dc-and-resource-management.md)基础上，固定 NVIDIA NVSHMEM `v3.7.2-0`、commit `3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4`，追踪这些语义如何落到 mlx5 atomic、RDMA WRITE、DUMP、NOP、CQ 更新与多 QP 协调上。

本文是源码静态分析，不是硬件执行记录。公开 API 语义来自当前 NVIDIA NVSHMEM 文档，具体函数、字段和分支来自上述固定 commit；两者版本不完全相同之处不据此推断行为一致。源码中出现的等待有时比 API 最低保证更强，这属于该实现的策略，不能反向解释为应用可依赖的新 API 契约。

## 1. 先分清四种“完成或排序”

理解本篇最容易出错的地方，是把发送队列上的先后、NIC 完成、fetch 结果写回和目标 GPU 可见性混为一个事件。同一 QP 上先发布 WQE，只能先建立传输层的请求顺序；CQ 所报告的完成说明某个完成边界之前的工作已经达到该 opcode 和 transport 定义的完成条件；fetching AMO 还要求旧值写入发起端的 registered result buffer；目标 PE 上的 GPU load 是否已经能够一致地观察由 NIC 写入的数据，则可能需要额外 consistency 操作。

因此，本篇采用下面的因果链，而不使用含义模糊的“操作完成”：

1. **提交顺序**：WQE 已写入 SQ，并通过 doorbell 使 NIC 可见；同一 QP 可以利用队列顺序，跨 QP 通常不能天然建立先后。
2. **远端效果**：PUT 或 AMO 已在目标内存产生协议规定的效果。non-fetching AMO 的调用可以早于该时刻返回。
3. **本地返回值可用**：fetching AMO 的旧值已由 NIC DMA 到发起 GPU 的内部 buffer，并可被调用线程读取。
4. **GPU 一致性可见**：NIC 写入的数据已经穿过平台所需的一致性边界，后续 GPU load 不会仅因缓存或代理域而继续观察旧值。

`quiet` 主要闭合此前通信的 completion，`fence` 主要建立此前操作与后续操作的 ordering，consistency operation 则处理 NIC 写入对 GPU load 的可见性。三者相关但不能互换；signal 也不是通用 cache flush，它只是一个可被等待的远端状态更新。

## 2. AMO API 语义与 IBGDA 的两条返回路径

NVSHMEM atomic memory operation（AMO）对远端对称对象执行不可分割的 read-modify-write。fetching AMO 要把修改前的值返回给调用 PE，因此函数返回之前必须取得结果；non-fetching AMO 不返回旧值，可以在目标执行原子更新之前返回，调用者需要 `quiet` 或具有相应 completion 语义的同步操作来确认完成。这一区别直接决定 IBGDA 是否分配可独占的 result slot、是否等待 CQ completion，以及何时允许复用该 slot。

固定源码中的 `ibgda_write_atomic_wqe` 负责把 NVSHMEM 内部 AMO 枚举编码为 mlx5 atomic WQE。它接收目标 DCT 信息、remote address/rkey、操作数、local result address/lkey 和 4/8-byte 宽度。即使是 non-fetching 操作，mlx5 atomic WQE 仍包含一个接收旧值的 local data segment；“API 不取回旧值”并不意味着网卡指令格式没有返回数据，而是实现可以把该数据导向不参与用户返回值的内部位置。

主要映射关系如下。表格描述该 commit 的 active IBGDA 路径，而不是对所有 transport、设备代际或 NVSHMEM 版本的普遍承诺。

| NVSHMEM 内部操作 | mlx5 WQE 方向 | 关键编码思路 |
| --- | --- | --- |
| `INC`、`FETCH_INC` | masked fetch-and-add | 加数为 1；4/8-byte 使用相应 extended atomic opmod |
| `ADD`、`SIGNAL_ADD` | masked fetch-and-add | 加数为传入 operand |
| `SET`、`SWAP`、`SIGNAL_SET` | masked compare-and-swap | compare mask 为 0、swap mask 全 1，使新值无条件替换 |
| `AND`、`OR` 及 fetching 版本 | masked compare-and-swap | 通过 compare/swap 与 mask 字段表达位运算 |
| `XOR` 及 fetching 版本 | masked fetch-and-add | 使用 field-boundary/mask 编码 extended atomic XOR |
| `FETCH` | masked fetch-and-add | 加数为 0，只取得旧值 |
| 64-bit `FETCH_ADD`、`COMPARE_SWAP` | ordinary atomic FA/CS | 直接使用 mlx5 64-bit atomic opcode |

WQE 长度并不恒定。DCI atomic 在该实现中占两个 WQEBB；RC 上部分 64-bit masked atomic 也占两个，而其余情形可以占一个。两个 WQEBB 表示一个 atomic WQE 的 segment 布局跨越两个基本块，不代表向远端执行两次原子更新。源码在需要时于 atomic 后追加 NOP，把 CQ update 放到最后一个 WQEBB 对应的完成边界，避免把“WQE 跨块”误当成“多个独立请求”。

### 2.1 Non-fetching：接受旧值，但不为用户保留它

`nvshmemi_ibgda_amo_nonfetch_impl` 先选择 QP，计算对称目标的 remote address/rkey，再预留 WQE 槽位。它不为每个调用预留独立 ibuf slot，而是把 atomic 的 local result address 指向 `qp->ibuf.buf` 的保留位置；旧值由 NIC 写到这里，但调用者不读取它。若 atomic 占两个 WQEBB，源码追加 NOP 并在 NOP 上设置 completion update；否则直接在 atomic 上设置。最后提交请求但不调用 `ibgda_quiet`，这与 non-fetching API 允许较早返回一致。

固定内部 buffer 的第一位置专门承担这种被丢弃的 non-fetch result。它不等价于“多个 NIC 可以任意覆盖一个普通结果槽”：安全性依赖该位置不作为 fetching 返回值交给用户线程，并依赖 atomic 写回数据本身不被后续协议读取。若改写 buffer 布局或让软件消费该值，就必须重新建立所有权和复用证明。

### 2.2 Fetching：独占 result slot，等待并有序释放

`nvshmemi_ibgda_amo_fetch_impl` 为参与线程预留 ibuf slot 和 WQE 区间，每个 lane 把自己的 result address/lkey 写入 atomic data segment。提交前，如果平台不能跳过 consistency step，源码在 atomic 后追加带 `IBGDA_MLX5_FM_FENCE` 的 DUMP WQE；该 NIC fence 约束此前 READ/ATOMIC 与后续 DUMP。若不需要 DUMP、但 atomic 本身跨两个 WQEBB，则追加 NOP 作为完整完成边界。

提交后函数无条件执行 `ibgda_quiet(qp)`，随后以 `READ_ONCE` 读取内部结果，并对 4-byte 结果执行字节序转换，最后通过 `ibgda_release_ibuf` 按 reservation 顺序释放 slot。这里形成完整 ownership 闭环：预留者独占结果位置，NIC 是写入者，CQ/quiet 证明写入已经结束，GPU 读取结果，然后 release head 前进，后续请求才可复用该位置。若省略 quiet 或提前 release，就会出现读未完成或 NIC 与后继请求同时写同一槽位的风险。

该 commit 还显式断言 IBGDA 不支持浮点 `ADD/FETCH_ADD`，且这里没有可用的 proxy fallback。因此不能从整数 atomic 的原生 mlx5 映射推出“所有 NVSHMEM atomic 类型都能走 IBGDA”。能力判断必须同时固定数据类型、操作、设备能力、构建配置和实际 transport dispatch。

## 3. Signal-op 是受限 AMO，不是独立消息通道

NVSHMEM signal object 是位于 symmetric memory 中的 64-bit 状态，生产者可以执行 `SET` 或 `ADD`，消费者用 signal wait/test 观察条件。固定源码的通用 device dispatch 先考虑本地可达路径：peer mapping 可达且为 `SET` 时可直接 store；load/store 路径的 `ADD` 使用 system-scope atomic；只有走 transport 时，`nvshmemi_signal_op` 才把 signal operation 交给 `nvshmemi_transfer_amo_nonfetch<uint64_t>`，IBGDA 最终进入上一节的 non-fetching atomic 路径。

这解释了两个重要边界。第一，signal 的逻辑语义统一，但底层不一定是 mlx5 atomic：同节点 peer mapping、其他 device transport 或 capability 分支可以选择不同实现。第二，独立 signal-op 只是对 signal object 的原子更新，它不会自动替此前在其他 QP、其他代理域或其他内存路径提交的数据建立发布顺序。若协议要求“看到 signal 就能读 payload”，应使用 put-with-signal 或显式建立满足 API 语义的 ordering，而不是把两个无关调用在源码文本中的先后当作全系统 happens-before。

## 4. Put-with-signal 如何建立 payload → signal

put-with-signal 把“写 payload”和“更新 signal”组合为一个发布操作。公开语义的核心不是减少一次函数调用，而是 signal update 必须在关联 PUT 完成到目标之后发生；消费者据此可以把 signal 当成数据就绪标志。IBGDA 的常规 thread 路径先计算 payload 与 signal 各自的 remote address/rkey，在同一个已选择 QP 上预留一个连续提交组：前部写 RDMA WRITE WQE，尾部写 `SIGNAL_SET` 或 `SIGNAL_ADD` atomic WQE，必要时再补 NOP。随后一次提交该组；blocking 版本再 quiet，NBI 版本则保留较早返回语义。

cooperative 路径把 thread group 的多个 lane 用作 payload chunk 生产者，并由后续 lane 写 signal atomic。在 chunk 数不超过可用 group width 时，payload WQE 与 signal WQE 仍属于同一 QP、同一 reservation/group；同步后统一发布，因而可以利用该 QP 的 WQE 顺序把 signal 放在所有 payload chunk 之后。这里需要同时满足两个条件：软件不能在 payload WQE 尚未就绪时跨过 publication hole，NIC 也必须按该 transport 对同一 SQ 的顺序规则执行 WRITE 与 atomic。D02 已解释前一个条件由连续 ready 边界保证。

本地 peer-mapped 路径采取不同机制：thread group 先复制 payload 并同步，然后执行 `__threadfence_system()`，再由一个线程更新 signal；TMA 路径还明确使用 blocking TMA put，确保发 signal 前数据移动已经结束。由此可见，发布顺序是上层语义，具体实现可以是“同一 QP 的 WRITE→atomic”，也可以是“本地复制完成→system fence→store/atomic”。不能只搜索 mlx5 opcode 来判断 put-with-signal 是否正确。

### 4.1 超大分块 fallback 的证据边界

当 cooperative 请求所需 chunk 超过一个 group 可容纳的数量时，源码回退到 thread implementation；thread implementation 对跨越首个注册区间的 payload，先调用 NBI PUT helper 处理数据，再单独提交 signal atomic。静态调用链显示 helper 会自行执行 QP selection，而 signal 使用进入函数时已选出的 QP。结合 D03 的多 QP 轮转，仅凭这一局部代码不能证明两者总落在同一 QP，也不能仅凭函数调用顺序证明跨 QP ordering。

这是一个**待核查点**，不是本文认定的缺陷：完整正确性还可能依赖该尺寸/内存分块条件下的 selector 约束、外层 explicit QP、未在局部展开的 ordering 机制，或特定配置不进入该组合。后续实验应构造跨 registration chunk 且超过 cooperative group 容量的 put-with-signal，记录实际 QP/WQE 序列，并在目标端反复验证“signal 已满足但 payload 仍旧”的禁止结果。在动态证据闭合前，本文只确认常规同-QP group 路径，不把它泛化到所有慢路径。

## 5. GET consistency：为何 CQ completion 之后还会有 DUMP

PUT 的数据方向是本地 GPU 到远端内存；GET 和 fetching AMO 的数据方向包含 NIC 写本地 GPU memory。后一类操作即使已经由 CQ 表明传输完成，也可能需要平台相关的 consistency step，才能保证后续 GPU load 按预期观察 NIC 写入。IBGDA device state 用 `get_head` 记录包含 fetch 类操作的最新 WQE 边界，用 `get_tail` 记录已经执行 consistency operation 的边界；二者使用原子最大值推进，因为多个生产者可能并发更新。

`ibgda_cst` 在一个 DCI 上提交 DUMP WQE，让 NIC 读取 GPU 内部 buffer 的一个字节，然后对该 DCI quiet。源码注释明确指出这里选择 DUMP 是因为它让 NIC 发起 GPU memory read，成本低于额外 RDMA READ。它不是把用户 payload 再读一次，也不更新 `get_head`，因为它本身是内部 consistency marker，而不是新的用户 GET。

`ibgda_quiet_with_cst` 先读取需要覆盖的 `get_head`，执行普通 quiet，再比较 `get_tail`。若 tail 仍落后于 head，则执行 CST 并更新 tail。对于 DCI，可在该 DCI 上直接运行；对于 RC，源码指出不存在所需的 RC loopback，于是选择一个指向 self 的 DCI 执行 CST，同时把原 RC 的 consistency tail 推进到已 quiet 的边界。这再次说明 QP completion 与 GPU consistency 是两张账：RC 上的数据请求已完成，不代表 consistency marker 也必须由同一 RC 发出。

fetching AMO 因为必须立即把结果返回给当前线程，会在本次请求后直接追加带 NIC fence 的 DUMP，并在函数内部 quiet；nonblocking GET 则可以先记录 `get_head`，把 consistency 工作延迟到后续 quiet。两种实现服务于不同返回时机，不能看到一个有 DUMP、另一个只更新计数器就断言语义不一致。

## 6. Fence 与 quiet：API 保证和内部策略不可倒置

NVSHMEM `fence` 保证调用 PE 在 fence 之前发出的相关操作，按规定先于 fence 之后的操作到达目标，但不承诺前序操作在 fence 返回时已经完成。`quiet` 则等待此前由调用 PE 发出的相关通信完成。应用协议应按这两个公开定义编写，即使某个 transport 的 fence 为了建立跨 QP 顺序而内部执行了更强等待，也不能用该偶然行为替代 quiet。

固定 IBGDA 源码恰好展示了这种差别。`nvshmemi_ibgda_fence` 先检查相关 QP 数量：只有一个 QP 时，同一队列的自然顺序足够，函数可以直接返回；多个 QP 可能同时面向同一 PE 时，源码遍历 DCI 和默认 RC 并调用普通 `ibgda_quiet`。注释同时强调 fence 不保证 prior operations completion，GET 即使在数据尚未对 GPU 到达时也可以满足 fence 语义。换言之，内部 quiet 是用 completion 作为跨独立队列重新建立次序的保守手段，不是提升公开 fence 契约。

`nvshmemi_ibgda_qp_quiet` 则按 scope、PE hint、默认或 explicit QP 集合遍历相关 DCI/RC，并调用 `ibgda_quiet_with_cst`。当 `enforce_cst` 为真，它除等待 CQ completion 外还按上一节补齐 GET consistency；用于 ordering 的内部 fence 路径可选择不做 CST。explicit QP fence 若确实只涉及一个非默认 QP，同样可以依赖该 QP 自身顺序而不等待；若 scope 实际覆盖多个队列，仍需遍历协调。

因此，性能分析不能只数用户 API 调用。单 QP fence 可能接近无操作，多 QP fence 可能等待许多队列；quiet 的成本取决于相关 QP 数量、pending completion 边界以及是否需要额外 CST；fetching AMO 自带同步返回路径，而 non-fetching AMO 可以聚合到后续 quiet。哪一种更快必须在固定 RC/DCI 数量、消息模式、NIC 和 topology 上测量。

## 7. 一个发布—消费协议的分层推演

设 PE 0 向 PE 1 发布一个数组 `payload`，并把 `ready` signal 从 0 设为 1。正确推演不是笼统地说“原子操作保证一致性”，而应逐层检查：`payload` 与 `ready` 都必须是满足 API 要求的 symmetric objects；生产者使用 put-with-signal，使 payload PUT 在 signal update 之前到达目标；消费者等待本地 `ready == 1`，条件满足后再读取 payload；若槽位要复用，还需额外的 acknowledgement 或 generation 设计，避免生产者覆盖消费者尚未读完的数据。

在常规 IBGDA 远端路径中，这可落为同一 QP reservation 里的若干 RDMA WRITE WQE，随后一个 signal atomic WQE。消费者的 wait 是本地轮询，不会替生产者回收发送队列，也不会自动确认上一代 buffer 已经消费完。若把 signal-op 与 PUT 分开提交到不同 QP，或只在发送端调用 fence 却把它误当作 completion，协议就缺少可证明的发布边；若只用单比特 ready 重复多轮，则还会出现 ABA/代际混淆。这些问题属于协议设计，而不是 atomic opcode 本身能修复的事情。

## 8. 阅读源码时的核对清单

面对一次 AMO 或 signal trace，应先确定 dispatch 实际选择 peer load/store、IBGDA、其他 device transport 还是 fallback；再核对操作类型和宽度是否受当前 capability 支持。进入 IBGDA 后，需要记录 QP 类型与索引、remote address/rkey、atomic WQEBB 数量、result address/lkey、CQ update 落在 atomic、NOP 还是 DUMP，以及函数返回前是否调用 quiet。

对于 fetching 操作，还要追踪 ibuf reservation、NIC 写入、字节序转换和 release 的先后；对于 GET，要观察 `get_head/get_tail` 与 CST；对于 put-with-signal，要确认所有 payload chunk 与 signal 的实际 QP 和提交组，而不是只检查最后一个 atomic。最后再把 trace 对照 API 契约：源码中的额外等待可以解释性能，却不能被应用依赖为跨版本保证。

## 9. 结论与尚未验证的范围

IBGDA 用一套共同的原子 WQE writer承载整数 AMO和 transport signal，但 fetching 与 non-fetching 的资源协议不同：前者必须独占结果槽、等待完成、读取并释放，后者把旧值导向保留位置并较早返回。put-with-signal 的常规路径依靠同一 QP 中的 WRITE→atomic 顺序建立发布关系；本地路径则通过复制完成、thread-group 同步和 system fence 建立同一上层语义。GET/fetching AMO 还揭示了 completion 与 GPU consistency 的分离，DUMP/CST 不是额外 payload，而是跨越可见性边界的实现机制。

本文没有编译或运行 NVSHMEM，没有抓取 NIC WQE、CQE 或 GPU memory trace，也没有验证浮点 atomic capability、多 QP fence 成本和超大分块 put-with-signal fallback。特别是后者的 payload helper 与 signal QP 是否在所有可达配置下保持必要 ordering，仍需动态 trace、版本对比或维护者证据；在闭合前只保留为源码审计问题。

## 参考资料

- [NVIDIA NVSHMEM Atomic Memory Operations API](https://docs.nvidia.com/nvshmem/api/latest/gen/api/amo.html)
- [NVIDIA NVSHMEM Signaling Operations API](https://docs.nvidia.com/nvshmem/api/latest/gen/api/signal.html)
- [NVIDIA NVSHMEM Memory Ordering API](https://docs.nvidia.com/nvshmem/api/latest/gen/api/ordering.html)
- [固定源码：`ibgda_device.cuh`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh)
- [固定源码：common device dispatch](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/common/nvshmemi_common_device.cuh)
- [固定源码：device transport dispatch template](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in)
