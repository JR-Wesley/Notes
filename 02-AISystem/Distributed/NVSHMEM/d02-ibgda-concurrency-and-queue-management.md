# D02：IBGDA 多生产者并发与发送队列管理

D01 追踪了一条 PUT 如何变成 WQE，并把 `resv_head`、`ready_head`、`prod_idx` 与 `cons_idx` 标在同一条时间线上。本篇进一步回答并发条件下的核心问题：许多 GPU thread、warp 或 CTA 共享一个 QP 时，为什么“原子获得不同槽位”还不足以安全提交；某个较早生产者尚未写完时，后来的生产者为何必须形成 publication hole；批量 doorbell 如何既降低开销又避免请求永久滞留；有限环形队列又如何在逻辑索引持续增长、物理槽位不断回绕的情况下防止覆盖。

正文继续固定 NVIDIA 官方 NVSHMEM `v3.7.2-0`、commit [`3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4`](https://github.com/NVIDIA/nvshmem/tree/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4)。本文分析的是该实现中的 device-side IBGDA 发送队列协议，不把内部计数器或内存序机制提升为 NVSHMEM API 契约；没有编译源码，也没有运行 GPU/NIC 实验。D03 将继续解释 QP 如何按 CTA、SM、warp 或 PE 映射，本篇只把“多个执行者已经选中同一个 QP”作为起点。

## 1. 为什么发送队列需要多阶段发布

### 1.1 原子预留只解决槽位唯一性

假设线程 A 和 B 都要向同一个 QP 写入一条 WQE。如果二者直接读取同一个 producer index，再各自加一，它们可能写入同一物理槽位；因此，第一步必须通过原子加法分配互不重叠的逻辑区间。原子预留能证明“每个生产者有唯一位置”，却不能证明“较小索引的 WQE 已经写完”。A 可能先得到 100 后被延迟，B 得到 101 并率先完成填充。若 B 此时通知 NIC 消费到 102，NIC 会依次读取槽位 100 和 101，其中 100 仍是不完整的硬件命令。

这个问题与 B04 讨论的 distributed work queue publication hole 同构：reservation 是所有权分配，publication 才表示内容完整。区别在于这里的消费者是 NIC，错误发布的不再是普通 payload，而是包含 opcode、地址、key 和长度的 WQE；读取半初始化 WQE 可能导致错误访问或 QP fatal，而不只是读到旧数据。

### 1.2 四个单调边界分别回答四个问题

固定源码在 [`nvshmemi_ibgda_device_qp_management_t`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/device_host_transport/nvshmem_common_ibgda.h#L158-L181) 中保存四个 64-bit 单调索引，单位都是 WQEBB。它们都采用“最后一个元素之后的位置”作为边界：

| 边界 | 回答的问题 | 谁推进 | 推进前必须成立 |
| --- | --- | --- | --- |
| `resv_head` | 哪些逻辑位置已经分配给生产者 | 预留者原子加 | 只要求获得唯一、不重叠区间 |
| `ready_head` | 从队首开始，连续到哪里都已填充完毕 | 恰好衔接前缀的生产者 CAS | 自己的 WQE 已写完，且所有更早区间已经 ready |
| `prod_idx` | 连续到哪里已经提交给 NIC handler | 决定 post-send 的生产者 | 对应区间已经 ready，并完成提交所需可见性处理 |
| `cons_idx` | 连续到哪里已经由完成侧回收 | CQ polling 线程原子 max | CQE 表明相应硬件完成边界已经到达 |

在无错误的稳态下，它们满足

\[
cons\_idx \le prod\_idx \le ready\_head \le resv\_head.
\]

这个偏序只说明生命周期阶段，不能直接当作容量公式。特别是 `resv_head` 先原子前移、生产者随后才等待物理槽位可用，所以大量生产者同时竞争时，`resv_head-cons_idx` 可以暂时大于队列深度。真正阻止覆盖的是每个生产者在写 WQE 前执行的 availability wait，而不是限制 reservation counter 永远留在物理环容量之内。

## 2. 第一阶段：预留逻辑区间，但暂不写物理槽位

[`ibgda_reserve_wqe_slots`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L1926-L1951) 以 `atomicAdd` 或 `atomicAdd_block` 推进 `resv_head`。返回值是本组 WQE 的 `base_wqe_idx`，右边界为

\[
new\_wqe\_idx = base\_wqe\_idx + num\_wqes.
\]

QP 可能被多个 CTA 共享时使用 GPU scope 原子操作；确定只在 CTA 内共享时使用 block scope 版本。作用域选择不是单纯性能提示：如果另一个 CTA 也能修改同一管理状态，CTA scope 原子性便不足以建立全体生产者可见的唯一顺序。QP 映射如何决定 `is_qp_shared_among_ctas` 留到 D03，此处只需要记住“共享域必须覆盖全部潜在并发者”。

原子加法后，函数把右边界传给 [`ibgda_wait_for_slot_availability`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L1705-L1722)。若队列深度为 \(N\)，且 `new_wqe_idx >= N`，生产者等待

\[
cons\_idx \ge new\_wqe\_idx-N.
\]

右边界所对应的最后一个物理槽位若已结束上一轮使用，则更早槽位也因 QP 顺序完成而可用，所以检查这一完成边界即可保护本组所有槽位。availability wait 返回后，生产者才可把逻辑索引用 `index & (N-1)` 映射到物理 WQ，并开始覆盖其中的旧字节。

### 2.1 一个超过环容量的 reservation 例子

设队列深度为 8，`cons_idx=100`，多个线程连续预留后，某线程得到 `[108, 110)`。它的右边界 110 要求等待 `cons_idx >= 102`，因为逻辑 108、109 将覆盖上一轮物理槽位 4、5，对应的旧使用必须已经结束。与此同时，更后的线程仍可能把 `resv_head` 推到 120；这并不立即覆盖环，因为它们也各自停在 availability wait。由此可见，观察到 `resv_head-cons_idx=20` 不足以证明环已损坏，必须继续检查 `ready_head`、实际 WQE 写入和等待目标。

相反，如果某条路径绕过 availability wait，或把“预留成功”误当成“物理槽位可写”，才会出现真实覆盖。D02 的关键安全断言不是 `resv_head-cons_idx <= N`，而是：任何生产者写入逻辑区间 \([b,e)\) 的物理映射前，完成边界必须至少达到 \(e-N\)。

## 3. 第二阶段：独立填充 WQE，并建立写入可见性

取得安全槽位后，各生产者独立写自己的 WQE。WQE writer 使用 relaxed stores 写 control、remote-address、data 或 atomic segment，因为此时槽位尚未发布，NIC 不应消费这些字节。把每个字段都改成强序写入并不能代替发布协议，只会增加开销；正确做法是在所有字段完成后，以适当 scope 的 fence 和 ready CAS 建立“先写内容、后发布边界”的关系。

跨 CTA 共享 QP 时，`ibgda_submit_requests<true>` 在推进 `ready_head` 前调用 `IBGDA_MEMBAR_NO_OPTIMIZATION()`。根据 [`IBGDA_MEMBAR_NO_OPTIMIZATION`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L205-L239) 的实现，NIC buffer 在 GPU memory 时使用 `__threadfence()`，在相应 host memory 情形下使用 `__threadfence_system()`。源码注释明确指出，另一个 CTA 可能代替当前 CTA post-send，而另一个 CTA 执行的 membar 不能自动把本 CTA 先前的 WQE writes 推到所需可见点，所以发布者必须在交出 ready ownership 前自己完成强 flush。

若 QP 只在同一个 CTA 内共享，代码使用 block scope CAS 和 `IBGDA_MFENCE()`；真正决定 doorbell 的线程稍后在 `ibgda_post_send` 内完成提交前的 membar。这个分支依赖“所有可能代为提交的执行者都在同一 CTA”这一前提，不能把它复制到跨 CTA 共享的 QP 上。

## 4. 第三阶段：只推进连续的 `ready_head`

[`ibgda_submit_requests`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L1641-L1686) 使用下面的 CAS 逻辑发布一组 WQE：只有当当前 `ready_head` 恰好等于本组 `base_wqe_idx` 时，才能把它改为 `new_wqe_idx`。如果 CAS 观察到更小值，说明前面仍有区间未 ready，当前生产者持续等待；它不能跳过空洞。如果观察到更大值，在正确的互不重叠 reservation 协议中也不应把自己的旧边界重新写回。

### 4.1 两个生产者的 publication hole

设初始四个边界都为 100，A 与 B 各预留一个 WQEBB：

```text
A owns [100, 101)       B owns [101, 102)

time 0: resv=102, ready=100, prod=100, cons=100
time 1: B fills WQE 101, CAS expects ready=101, but observes 100 → wait
time 2: A fills WQE 100, CAS 100→101 succeeds
time 3: B retries, CAS 101→102 succeeds
```

B 在 time 1 已经完成自己的内容，却不能声明边界 102 ready，因为 ready 的定义是“从队首开始连续完整的右边界”，而不是“最大已写完索引”。这种等待形成 head-of-line blocking：最早空洞的延迟会传播给所有后继提交。它是维持 NIC 顺序消费的代价，而不是 CAS 实现偶然造成的性能问题。

从活性角度看，这个算法不容许生产者在 reservation 后永久消失。若 A 因控制流错误、断言、永久等待或执行资源问题永远不能发布 `[100,101)`，B 及其所有后继都会停在 ready CAS；协议没有跳过、撤销或由他人补写该槽位的恢复路径。这个结论是基于源码状态机的推论，不表示正常 NVSHMEM 调用会主动丢弃生产者。

### 4.2 warp coalescing 如何减少而非消除并发

在 [`ibgda_rma_thread`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L2077-L2216) 中，只有 active mask 为完整 warp、lane 的 proxy PE 相同，并在取得 QP 后确认 QP 相同时，才能进行 warp coalescing。lane 0 为整个 warp 一次预留

\[
num\_wqes=num\_wqes\_per\_cmd\times32+additional\_wqe,
\]

再把基址广播给所有 lane。每个 lane 填充自己的固定子区间，warp 同步后由最后一个 lane 写额外 DUMP/NOP（如有）并提交整组。这样把 32 次 reservation 和最多 32 次 publication 合并为一次，也让一组 lane 共享一个 CQ update 与 doorbell 机会。

合并并不意味着 NIC 看见一条“32-lane WQE”，也不消除不同 warp、不同 CTA 之间的 publication hole。只要请求目标或 QP 不一致、active mask 不完整，代码就退回 thread-granular 预留；即便成功合并，整个 warp 分配到的区间仍必须等待更早生产者推进 `ready_head`。

## 5. 第四阶段：批量提交与 doorbell 合并

`ready_head` 前移并不保证每组 WQE都立即敲 doorbell。固定实现用 `NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH` 控制提交粒度；初始化代码会把正值向上取整为 2 的幂，并要求不大于 QP depth。固定 commit 的环境定义默认值为 32，当前官方环境变量文档也将其描述为提交前的 batch 大小，并指出设为 1 可进行 aggressive submission。

变量名和文档使用“requests”，但 [`ibgda_submit_requests`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L1641-L1686) 的阈值计算直接使用 WQEBB 单位的 `base_wqe_idx`、`new_wqe_idx` 和 `num_wqes`。因此在该 commit 中，一条跨两个 WQEBB 的 DCI WQE、额外 NOP/DUMP 或一次 warp 合并组会比“一个 API 调用等于一个 request”的直觉更快触发阈值。这是实现级细节，调参和解释 trace 时应按 WQEBB 边界核对。

ready 发布后，满足下面任一条件便执行 post-send：

1. `new_wqe_idx == resv_head`：当前组已经追上 reservation tail，没有已知后继生产者，低流量请求不会无限等待凑满 batch；
2. `[base_wqe_idx,new_wqe_idx)` 跨过一个 batch 对齐边界：持续并发下即使生产者总看见更后的 reservation，也会周期性通知 NIC；
3. 当前组自身的 `num_wqes` 已达到 batch 大小：大型合并组不再额外等待。

若 batch 为 4，单 WQEBB 组从 `[100,101)`、`[101,102)`、`[102,103)` 到 `[103,104)` 连续 ready，前三组在存在已预留后继时可以不提交，第四组因跨越 4-WQEBB 边界而提交到 104。若第三组实际上已经是 reservation tail，它会因第一项条件提前提交到 103，而不是等待一个永远不会到来的第四请求。因此，“batch=4”表示允许的合并阈值，不表示 NIC 每次严格收到四条 API 请求。

### 5.1 多个线程同时决定 post-send 时如何收敛

GPU NIC handler 的 [`ibgda_post_send`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L1595-L1619) 用 `post_send_lock` 串行化 `prod_idx`、DBREC 和 UAR/BF 更新。临界区内先以 `atomicMax` 推进 `prod_idx`；只有 `new_prod_idx` 大于此前值时才执行 membar、更新 DBREC 并 ring doorbell。即使两个生产者都在锁外判断应提交，较新的边界先完成后，持有较旧边界的线程也不会把 `prod_idx` 或 doorbell 倒退。

这个 spinlock 只保护最终 doorbell 临界区，不保护 WQE 填充和 ready CAS；否则所有生产者会在最长路径上完全串行化。源码没有给该锁提供公平性保证，因而高竞争下某些线程等待更久是可能的推论，不过临界区只包含单调更新和两次 NIC 通知写，设计目标显然是保持它足够短。

CPU NIC handler 分支不让 GPU 直接写 UAR，而由 `ibgda_proxy_post_send` 原子提升 `prod_idx`，再用 system-scope atomic max 发布待处理边界给 host progress engine。多个 GPU 生产者仍可把多个通知合并成最大边界，但活性额外依赖 host progress 持续运行。payload 并未因此改为 CPU 搬运，变化的是 doorbell 的最终执行者。

## 6. 第五阶段：CQ polling、回收和索引回绕

### 6.1 完成侧推进的是可复用边界

[`ibgda_poll_cq`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L463-L580) 首先检查 `cons_idx` 是否已经达到调用者的目标；若没有，还要等待 `prod_idx >= target`，避免在目标 WQE 尚未提交时用旧的、已经回绕的 `wqe_counter` 得出假完成。随后代码读取 collapsed CQE 中的 16-bit `wqe_counter`，结合 64-bit 目标索引的高位和回绕修正重建新的逻辑完成边界，并用 `atomicMax` 推进 `cons_idx`。其他线程若已完成相同工作，当前线程可以直接观察到更大的 `cons_idx` 并返回。

CQ polling 因而也是共享 progress：等待新槽位的生产者和执行 quiet 的线程都可能推进同一个完成边界。它们不是分别消费“属于自己的 CQE”，也不会因两个线程同时 polling 就把完成记录重复释放；单调 `atomicMax` 把并发结果收敛到最大已证实边界。遇到 CQE request error 时，固定实现将其视为 fatal，而不是跳过错误 WQE继续复用队列。

### 6.2 为什么同时需要 64-bit 软件索引和 16-bit 硬件索引

物理 WQ 是有限环，DBREC 与 mlx5 `wqe_counter` 只携带低位索引；软件管理状态则使用 64-bit 单调计数，保留“已经绕过多少轮”的代际信息。初始化把 QP depth 取整为 2 的幂并限制在 128 到 32768；[`ibgda_get_wqe_ptr`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L423-L428) 用 `wqe_idx & (nwqes-1)` 取得物理槽位，而 polling 代码围绕目标边界解释 16-bit counter。

如果只保存取模后的物理槽号，就无法判断 CQE 中的 5 表示第一轮 WQE 5，还是第 8193 轮同一槽位。反过来，仅有 64-bit reservation index 也不足以读取硬件队列，因为 NIC ABI 使用有限宽度字段。固定实现通过受限队列深度、先确认目标已提交、再围绕目标重建 64-bit boundary，连接这两个编号空间。

### 6.3 一次跨物理环尾的完整例子

设深度 \(N=8\)，初始 `cons=prod=ready=resv=106`。生产者 A 预留 `[106,108)`，物理槽位为 2、3；生产者 B 预留 `[108,110)`，物理槽位为 4、5。两组的右边界分别要求旧完成至少达到 100 和 102，因此当前 `cons=106` 足够，二者都可填充。若 B 先写完，它仍在 ready CAS 等待 A；A 发布 106→108 后，B 才能发布 108→110。

假设 batch 条件令 B post-send 到 110，NIC 完成 WQE 109 后报告低位 counter 109，polling 将其转换为软件 next-boundary 110，`cons_idx` 随之变为 110。下一轮逻辑 114 会映射到物理槽位 2，但 reservation 后必须等待 `cons >= 106`；当前已经满足，所以覆盖逻辑 106 所用旧槽位是安全的。例子中的逻辑值刻意不从零开始，用于强调“物理位置相同”和“生命周期相同”不是一回事。

## 7. 安全性与活性必须分开证明

### 7.1 安全性依赖的局部不变量

在固定实现的正常路径上，可以从源码提炼出四条安全条件。第一，所有潜在共享者在足够大的原子作用域内推进 `resv_head`，所以 reservation 区间不重叠。第二，每组在 availability wait 通过后才写物理 WQ，所以不会覆盖尚未完成的上一轮槽位。第三，WQE 内容和必要的 flush 先于 `ready_head` 发布，且 ready 只能连续推进，所以 NIC 不会越过初始化空洞。第四，post-send 只单调推进 `prod_idx`，completion 只单调推进 `cons_idx`，因此旧 doorbell 或旧 CQ 观察不能让生命周期倒退。

这些条件保证“不会错误覆盖或过早提交”，却不自动保证“每个请求最终完成”。安全系统可以永久停在一个不错误但无进展的状态，例如 `resv=140, ready=100, prod=100, cons=100`：只要没有人通知 NIC，队列内容就不会被错误执行，但应用已经挂起。

### 7.2 活性需要哪些外部前提

要从安全性进一步得到最终完成，至少需要以下前提：最早 reservation 的所有者最终填充并发布，不在中途永久退出；CUDA 调度最终让持有最早 hole 的 resident execution context 继续运行；batch tail 或阈值条件最终触发 post-send；GPU direct handler 的 UAR 路径或 CPU handler 的 host progress 能继续工作；NIC、网络和对端不会永久停止，并产生可观察完成；polling 线程最终获得执行机会。

固定源码的普通 polling loop 没有用有限重试把这些条件变成恢复协议。启用 `NVSHMEM_TIMEOUT_DEVICE_POLLING` 时可以输出 `cons_idx`、`prod_idx`、`resv_head`、`ready_head`、QP/CQ 编号、opcode 和 `wqe_counter` 等诊断信息，但 timeout 本身不能修复 publication hole 或 NIC failure。把“检测到停滞”和“系统能够恢复”视为同一件事，会高估实现的容错能力。

### 7.3 三种典型停滞形态

| 观测关系 | 更可能停在哪一阶段 | 首要检查 |
| --- | --- | --- |
| `ready_head < resv_head`，长期不变 | 已预留但存在最早未发布区间 | 哪个 producer 拥有 `ready_head` 起始槽位；是否卡在填充、warp 同步或容量等待 |
| `prod_idx < ready_head`，长期不变 | ready 已连续，但 doorbell 未推进 | batch/tail 判断、`post_send_lock`、GPU UAR 或 CPU handler notification/progress |
| `cons_idx < prod_idx`，长期不变 | NIC 已被通知，但完成未被观察 | CQ owner/counter、QP error、NIC/network progress、目标 polling 与 counter 回绕解释 |

`resv_head-cons_idx` 很大只能说明积压的 reservation 多，不能单独定位故障。诊断必须寻找四个边界中第一个没有继续前移的位置，再把它对应到该阶段的所有者与可见性条件。

## 8. 批处理与竞争的性能取舍

减小 `NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH` 会更频繁地执行 `prod_idx` 更新、DBREC write 和 UAR doorbell，通常有利于低并发、小消息的提交延迟，却增加 doorbell 与锁竞争开销。增大 batch 可以让连续 WQEs 共用一次通知，在高并发下摊薄控制成本，但请求可能等待 tail 或阈值，延迟分布也更依赖生产者到达模式。由于固定源码以 WQEBB 而非纯 API 调用次数计算阈值，RC/DCI 布局、DUMP/NOP 和 warp coalescing 都会改变实际批量。

增大 QP depth 能容纳更多已提交但未回收的 WQEBB，也可能减少容量等待；它同时增加队列内存与潜在排队深度，不能修复 ready hole 或失效的 progress。增加 QP 数可能降低单 QP 上的原子、CAS 和 doorbell lock 竞争，但带来连接、CQ、NIC context 和映射成本；这个资源权衡属于 D03，不能只凭 D02 的队列局部模型断言“QP 越多越快”。

因此，调参实验至少应同时记录 API operation 数、实际 WQEBB 数、目标 QP、warp 合并比例、doorbell 次数、CQ 完成推进和端到端延迟。只比较环境变量数值与总带宽，无法判断收益来自 batching、QP 分散、消息分段还是 handler 变化。本文没有执行这些实验，以上关系是源码机制与性能因果假设，不是测量结论。

## 9. 诊断推演与常见误解

如果 B 已完成槽位 101，而 A 卡在槽位 100，能否让 B 先 doorbell 自己的 WQE？不能。SQ 按连续顺序消费，B 的 WQE 没有独立于槽位 100 的可提交前缀；若业务允许乱序，应在更高层选择不同 QP 或重新设计映射，不能破坏单 QP ready invariant。

如果 `resv_head-cons_idx` 大于 QP depth，是否已经发生内存覆盖？不一定。reservation 先发生，availability wait 后发生，尚未通过等待的生产者不能写物理槽位。应检查 `ready_head-cons_idx`、生产者等待目标以及实际填充位置。若有人已经在等待前写 WQE，才违反协议。

如果 batch=32，是否第 1 到第 31 条请求一定不会提交？也不是。当前 ready 组追上 `resv_head` 时会立即 post-send；一次 warp 合并或多 WQEBB 命令也可能单组达到或跨过阈值。32 在固定实现中参与的是 WQEBB 边界运算，不是严格的 API 调用计数器。

如果多个线程同时 poll 同一个 CQ，会不会把同一槽位释放两次？完成侧以单调 `atomicMax` 更新 `cons_idx`，线程等待的是边界而非领取独占 CQE；较旧结果不会使 `cons_idx` 倒退。但所有线程都无法替代 NIC completion，本地原子操作只能合并已经观察到的完成证据。

## 10. 结论与掌握标准

IBGDA 发送队列不是一个由原子计数器保护的普通数组，而是由 reservation、publication、submission 和 reclamation 四类所有权组成的并发状态机。`resv_head` 允许生产者快速取得唯一逻辑区间；availability wait 把无限增长的 reservation 空间安全映射到有限物理环；`ready_head` 强制内容按连续前缀发布；`prod_idx` 通过 batching 与单调 doorbell 合并控制 NIC 可见进度；`cons_idx` 再把 collapsed CQ 的有限宽度 counter 恢复为可复用的 64-bit 边界。

读者完成本篇后，应能解释为什么 atomic reservation 不能代替 ready publication，为什么 `resv_head-cons_idx` 可以暂时超过深度，为什么跨 CTA 共享 QP 要在交出 ready ownership 前做更强 flush，为什么 batch 阈值按 WQEBB 计算，以及为什么一个最早生产者的永久停滞会阻塞所有后继。还应能从四个边界的相对位置判断问题更可能发生在填充、doorbell 还是 completion 阶段，并把安全性证明与 progress 假设分开。

下一篇 D03 将沿用同一 commit，把本篇抽象的“共享 QP”落实到 RC 与 DCI/DCT：比较连接对象、目的端寻址、CTA/SM/warp 映射、shared/exclusive DCI、每 PE RC 数量，以及这些配置如何改变竞争域、连接规模和 NIC 资源成本。

## 参考资料

- [NVIDIA NVSHMEM v3.7.2-0 固定源码](https://github.com/NVIDIA/nvshmem/tree/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4)
- [NVIDIA NVSHMEM：Environment Variables](https://docs.nvidia.com/nvshmem/api/latest/gen/env.html)
- [NVIDIA NVSHMEM：Performance Best Practices](https://docs.nvidia.com/nvshmem/release-notes-install-guide/best-practice-guide/performance.html)
- [D00：IBGDA 实现总览](d00-ibgda-implementation-overview.md)
- [D01：从一次 PUT 追踪 IBGDA 请求生命周期](d01-ibgda-request-lifecycle.md)
- [B04：数据发布、缓冲区所有权与有界流水协议](b04-data-publishing-and-buffer-ownership.md)
