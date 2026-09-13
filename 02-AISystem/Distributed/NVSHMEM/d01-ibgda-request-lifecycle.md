# D01：从一次 PUT 追踪 IBGDA 请求生命周期

本篇在 D00 的对象总图之上回答一个更具体的问题：当 GPU 线程调用 `nvshmem_putmem`，并且请求最终选择 IBGDA transport 时，用户给出的本地地址、对称堆目标地址和 PE，如何逐步变成 NIC 能执行的 mlx5 WQE；多个 GPU 线程如何安全地把 WQE 放入共享发送队列；doorbell 何时真正通知 NIC；调用返回又对应哪一种完成边界。为了显出 PUT 的共性与差异，后半篇使用同一固定源码比较 GET 和 fetching AMO，但不展开所有 atomic opcode 与全局 ordering 规则，那些问题留给 D04。

正文固定到 NVIDIA 官方 NVSHMEM `v3.7.2-0`、commit [`3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4`](https://github.com/NVIDIA/nvshmem/tree/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4)。所有有关函数分派、WQE 布局和队列状态的判断都以该 commit 为准；NVSHMEM API 语义则以官方 API 文档为准。二者必须分开：实现可以在某个版本中做得比 API 最低保证更强，但应用程序不能据此获得跨版本、跨 transport 的额外可移植保证。

## 1. 追踪边界：先固定一条可重复推演的路径

一个 device-side PUT 并不天然等于 IBGDA 请求。同节点目标可能直接通过 peer mapping 或 TMA 写入；启用 logical endpoint 后也可能先进入另一条路径；只有运行时选择了 IBGDA device transport，才会执行本文分析的 mlx5 WQE 构造。因此，本文选择下面的代表性场景，把无关分支暂时固定下来：

1. 一个 GPU 线程调用 blocking `nvshmem_putmem`，目标 PE 不可通过本地 P2P 映射访问，logical endpoint 分支未被选中；
2. 运行时的 `selected_device_transport` 为 IBGDA，目标由 GPU NIC handler 负责；
3. 使用 thread scope，关闭或不满足整 warp 合并条件，仅一个 lane 产生请求；
4. 源区间和目标区间分别落在单个已注册内存区间内，请求不因 registration chunk 边界而拆分；
5. 先以 RC QP 为基线，再说明 DCI half-AV 与 full-AV 布局的增量。

这些条件不是 NVSHMEM 的普遍限制，而是一次源码追踪的坐标系。改变目标拓扑、handler、QP 类型、协作域或请求长度，会改变其中若干步骤，却不会改变“路由选择—地址与 key 准备—WQE 预留和填充—发布与 doorbell—完成和回收”这条主线。

## 2. 先区分三种完成含义

PUT 的难点之一，是“函数返回”容易被误读成“远端已经可以消费数据”。NVSHMEM 的公开 RMA 语义区分源缓冲区可复用与数据在目标 PE 上完成传输：blocking PUT 返回时，调用者可以按接口保证复用源缓冲区，但远端传输完成仍应由 `nvshmem_quiet` 建立；NBI 请求同样需要相应的 quiet 来完成。远端消费者何时可以安全读取，还需要与 signal/wait、barrier 或其他发布协议组合，不能仅从发起端函数返回推导。

固定 commit 中，`nvshmemi_ibgda_rma(..., nbi=false)` 的 thread 实现会在请求提交后调用该 QP 上的 `ibgda_quiet(qp)`。这说明当前内部路径可能等待到比 blocking PUT 的最低 API 保证更强的队列边界。然而这只是特定实现、特定 transport 和特定 QP 上的行为，不应提升为公开语义。尤其要避免两个错误推论：第一，不能因为内部调用了 quiet，就认为所有 transport 的 blocking PUT 都承诺远端完成；第二，不能把 `ibgda_quiet(qp)` 描述成“等待这条请求独占的一条 CQE”，因为 collapsed CQ 允许一个 CQE 推进一段已完成 WQE 边界。

后文使用以下术语：**请求已编码**表示 WQE 字节已经写入 WQ；**请求已发布**表示连续 ready 前缀已经推进；**请求已 doorbell**表示 NIC 已获知新的 producer boundary；**请求已完成**表示软件根据 CQ 观察到的完成边界确认该 WQE 不再占用队列容量。这四个时刻可能相邻，但不是同一个事件。

## 3. 从公开 API 进入 IBGDA 实现

### 3.1 device API 先做路径选择

[`nvshmem_putmem`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/device/nvshmem_defines.h#L200-L205) 把无类型字节复制转给 `nvshmemi_put<char, NVSHMEMI_THREADGROUP_THREAD>`。在 [`nvshmemi_put`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/common/nvshmemi_common_device.cuh#L1106-L1136) 中，代码先读取目标 PE 的 `peer_heap_base_p2p`。如果目标对称堆可由当前 GPU 直接访问，就采用 peer/TMA copy；否则再检查 logical endpoint，最后才调用 `nvshmemi_transfer_rma<SCOPE, NVSHMEMI_OP_PUT>`。

[`nvshmemi_transfer_rma`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in#L150-L177) 根据 `selected_device_transport` 分派。IBGDA 分支调用 `nvshmemi_ibgda_rma`，proxy transport 则进入 proxy 实现。由此可见，应用层的“device API”、数据面的“GPU memory 到远端 memory”，以及提交面的“GPU 还是 CPU 敲 NIC doorbell”是三项不同属性。仅看到 `nvshmem_putmem` 出现在 kernel 中，无法证明运行时经过了 IBGDA，更无法证明使用了 GPU NIC handler。

可以把代表性路径压缩为：

```text
nvshmem_putmem
  → nvshmemi_put<THREAD>
      ├─ peer/TMA copy                    （本文排除）
      ├─ logical endpoint                 （本文排除）
      └─ nvshmemi_transfer_rma<PUT>
           ├─ proxy transport             （本文排除）
           └─ nvshmemi_ibgda_rma
                → nvshmemi_ibgda_rma_thread
                    → address/key lookup
                    → reserve and write WQE
                    → publish ready prefix
                    → ring doorbell
                    → poll completion boundary
```

### 3.2 wrapper 固定 proxy PE、协作方式和 QP

[`nvshmemi_ibgda_rma`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L2793-L2828) 先取得目标对应的 proxy PE，再依据 threadgroup scope 选择 thread 或 cooperative 实现，并把 blocking 版本的 `nbi=false` 传入。thread 实现随后通过 `ibgda_get_qp` 选择实际 QP。这里的 proxy PE 是寻址和 endpoint 映射的一部分，不等同于“CPU proxy 负责提交”；是否由 CPU handler 敲 doorbell 是稍后的另一个分支。

若多个 lane 同时调用同一 PE，代码只在 active mask 是完整 warp、所有 lane 的 PE/QP 条件一致时启用 warp coalescing。合并后的工作仍表示各 lane 的独立操作，只是批量预留和发布 WQE，并不是把 thread API 自动改写成一次块级 collective。本文先追踪单 lane；D02 再处理多生产者与合并带来的空洞、回绕和背压。

## 4. 地址与 key：把 PGAS 参数翻译成 NIC 参数

调用点只给出 `dest`、`source`、字节数和目标 PE，mlx5 RDMA WRITE WQE 却需要本地虚拟地址、`lkey`、远端虚拟地址和 `rkey`。这一步是 NVSHMEM 对称寻址模型与 RNIC 访问权限模型的交界。

对于源地址，[`ibgda_get_lkey`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L1832-L1885) 在设备侧 registration metadata 中定位覆盖当前 local pointer 的内存区间，并取得相应 `lkey`。`lkey` 不是由虚拟地址现场计算出的权限码，而是初始化阶段注册内存后下发给 device state 的 NIC key。若源指针移动到另一个注册区间，下一段 WQE 可能需要新的 `lkey`。

对于目标地址，[`ibgda_get_raddr_rkey`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L1887-L1924) 根据目标 PE、proxy PE 和远端 registration metadata 取得 NIC 可使用的 remote address 与 `rkey`。教学中常写成“remote heap base + symmetric offset”，这可以解释对称对象为何可由本地指针和 PE 定位，却不是所有 transport、所有内存类型都必须采用的内部公式；固定源码中的 metadata 查找才是本文能够核实的实现事实。

单个 WQE 的长度还受四个上界共同约束：本次剩余字节数、IBGDA 最大传输大小、本地注册区间剩余长度、远端注册区间剩余长度。可以写成

\[
L_{wqe}=\min(L_{remain},L_{max},L_{local\_chunk},L_{remote\_chunk}).
\]

因此，“一次 NVSHMEM PUT”与“一条 RDMA WRITE WQE”并非一一对应。本文假设各上界都不截断请求，所以只产生一条数据 WQE；真实的大请求或跨 registration chunk 请求会循环查询 key、预留 WQE、推进源/目标指针，直至 `remaining_bytes` 归零。

## 5. WQE 预留与编码：先拥有槽位，再写硬件格式

### 5.1 逻辑索引与物理环形槽位

发送队列是有限深度的环。IBGDA 使用单调递增的逻辑索引描述请求边界，再通过掩码映射到物理槽位。对深度为 `nwqes` 的二次幂队列，[`ibgda_get_wqe_ptr`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L423-L428) 的核心关系是

\[
physical\_slot = logical\_wqe\_index \mathbin{\&} (nwqes-1).
\]

物理地址再按 `MLX5_SEND_WQE_SHIFT` 偏移。逻辑索引必须持续增长，因为仅有取模后的槽号无法区分“上一轮的槽 5”和“下一轮的槽 5”；完成边界与容量判断依赖这种代际信息。

[`ibgda_reserve_wqe_slots`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L1926-L1951) 原子推进 `resv_head`，为生产者分配互不重叠的逻辑区间。预留只表示生产者拥有这些槽位，并不表示内容已写完，也不能通知 NIC。若 `resv_head - cons_idx` 接近队列容量，生产者必须等待完成侧回收空间，否则会覆盖仍由 NIC 使用的旧 WQE。

### 5.2 RC RDMA WRITE 的字段组成

[`ibgda_write_rdma_write_wqe`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L645-L711) 将软件参数编码成 mlx5 WQE。以 RC 为例，一条 WRITE 使用三个 16-byte data segment，合计 48 bytes，放在一个 64-byte WQEBB 中：

| segment | 主要字段 | 本次 PUT 中的来源 |
| --- | --- | --- |
| control segment | opcode、WQE 逻辑索引、QP number、DS 数、CQ update 等控制位 | QP state、预留结果与提交策略 |
| remote-address segment | remote virtual address、`rkey` | `ibgda_get_raddr_rkey` |
| data segment | byte count、`lkey`、local virtual address | 分段长度与 `ibgda_get_lkey` |

RC 不需要在每条 WQE 中携带目的端地址向量，因为 QP 已绑定对端，因而 `ds=3`、opcode 为 `MLX5_OPCODE_RDMA_WRITE`，一个 WQEBB 足够。长度、地址和部分控制字段在写入时转换成 NIC 要求的 big-endian 格式；`lkey/rkey` metadata 已按实现约定保存。这里的端序转换是硬件 ABI 的要求，不改变 CUDA 指针本身的数值语义。

若 QP 是 DCI，WQE 还要携带 address vector。half-AV 情况增加一个 16-byte segment，`ds=4`，仍能放入一个 WQEBB；full-AV 情况的 `ds=6` 会跨两个 WQEBB。固定实现还会在 full-AV 数据 WQE 后追加一条 NOP，并把 CQ update 请求放在 NOP 上，所以单个代表性 PUT 会预留三个 WQEBB。这个额外 WQE 是布局与完成标记策略的结果，不是多传输了一份 payload。

对于源位于 system memory 的 PUT，thread 实现在提交前执行 `__threadfence_system`，使 GPU 对源数据的先前写入在 NIC 读取前达到所需的系统可见边界。它解决的是“NIC 读到新源数据”的本地发布问题，不代替远端完成或目标消费者同步。

## 6. 从 ready 发布到 doorbell：何时 NIC 才能看见请求

### 6.1 `resv_head` 与 `ready_head` 不能合并

多个 GPU 生产者可能按顺序预留，却乱序完成 WQE 填充。例如线程 A 获得槽位 100 后被延迟，线程 B 已写好槽位 101。B 不能把 `ready_head` 直接推进到 102，否则 NIC 可能读取尚未完成的槽位 100。因此，[`ibgda_submit_requests`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L1641-L1686) 只在自己的区间与当前连续 ready 前缀衔接时推进 `ready_head`；出现空洞时，后继生产者必须等待前驱发布。

这形成四个不可互换的边界：`resv_head` 是已分配但可能未写好的右边界，`ready_head` 是所有字节均已完成写入的连续右边界，`prod_idx` 是已经通知 NIC 的右边界，`cons_idx` 是已由完成侧回收的右边界。正常情况下应保持：

\[
cons\_idx \le prod\_idx \le ready\_head \le resv\_head.
\]

并发瞬间中某些边界可能相等，不能因此将它们实现为一个计数器。D02 将专门分析各边界的所有权和内存序。

### 6.2 GPU NIC handler：DBREC 与 UAR/BF 各自做什么

当当前线程取得可提交的连续前缀时，[`ibgda_post_send`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L1562-L1619) 先更新 doorbell record（DBREC），把新的 SQ producer index 写入 NIC 可见内存；随后把 control segment 的 doorbell 数据写到 UAR/BlueFlame mapping。DBREC 是内存中的队列进度记录，UAR/BF write 是促使 NIC 及时取走新 WQE 的 MMIO doorbell，两者不能互相替代。

doorbell 使用的是“下一个空闲位置”的 producer boundary，而 WQE control segment 携带的是当前 WQE 的逻辑索引。对于一条占一个 WQEBB 的请求，WQE 编号为 100，提交后的 producer boundary 是 101。这一个 off-by-one 是读源码和日志时最常见的混淆来源。

### 6.3 CPU handler 只替换最终通知者

如果实际选择 CPU NIC handler，GPU 仍然完成 address/key 查找、WQE 构造和 ready 发布；不同之处是 [`ibgda_proxy_post_send`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L1621-L1639) 把待提交 producer boundary 发布给 host progress engine，再由 host 侧 [`nvshmemt_ibgda_progress`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/modules/transport/ibgda/ibgda.cpp#L469-L600) 更新 DBREC/UAR。因而 CPU handler fallback 不等于 payload 经 CPU staging；payload 仍可由 NIC 在 GPU memory 与远端 memory 之间直接搬运，变化的是 doorbell agent 和控制路径延迟。

## 7. 完成与回收：CQE 表示边界，不必逐请求对应

[`ibgda_poll_cq`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L465-L575) 检查 CQE owner bit、状态和 `wqe_counter`，把硬件报告的已完成 WQE 编号转换成软件的 next-boundary，再推进 `cons_idx`。队列采用 collapsed CQ：不是每条 WQE 都必须生成一个 CQE，带 CQ update 的后继 WQE 可以代表其前面一段有序 WQE 已经越过完成边界。软件关注的是 `cons_idx` 是否达到等待目标，而不是“是否收到属于某一 API 调用的唯一 CQE”。

[`ibgda_quiet(qp)`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L1688-L1703) 读取该 QP 当前的 `ready_head` 作为等待边界，并轮询到 `cons_idx` 达到该边界。这里的 quiet 是固定实现内部、针对所选 QP 的队列等待机制；公开 `nvshmem_quiet` 还需覆盖调用语义要求的此前操作集合，不能只凭这个 helper 推导全局 ordering。

### 7.1 一个带具体索引的单 WQE 例子

假设 RC SQ 深度为 128，起始状态为 `resv_head = ready_head = prod_idx = cons_idx = 100`，且没有其他生产者。一次单 WQEBB PUT 的生命周期如下：

1. 预留操作返回逻辑索引 100，并把 `resv_head` 推到 101；物理槽位是 `100 & 127 = 100`。
2. 线程在槽 100 写入 RC RDMA WRITE WQE。control segment 中的 WQE index 是 100，最后一条 WQE设置 `MLX5_WQE_CTRL_CQ_UPDATE`。
3. WQE 写完后，线程把连续 `ready_head` 从 100 推到 101。因为不存在前驱空洞，它可以立即提交。
4. `ibgda_post_send` 令 `prod_idx=101`，DBREC 记录 next-empty index 101，并用包含 producer index 101 与 QP number 的 doorbell 数据通知 NIC。
5. NIC 完成编号 100 的 WQE 后，collapsed CQE 的 `wqe_counter` 对应已完成 WQE 100；polling 逻辑把它转换成软件边界 101，于是 `cons_idx=101`。

因此，日志中同时出现 100 和 101 并不矛盾：100 标识被执行的 WQE，101 标识其后的边界。若一次 DCI full-AV 请求占用三个 WQEBB，则初始逻辑索引仍为 100，但提交和等待的右边界会变成 103。

## 8. 现实变化：分段、NBI 与 warp 合并

当请求跨越本地或远端 registration chunk 时，thread 实现每轮重新取得 `lkey/raddr/rkey`，计算本轮长度并预留对应 WQEBB。一个 API 请求由多条 WQE 表示，最后一次循环结束才完成整个字节区间。诊断“为什么 WQE 数量多于 API 次数”时，应先排除最大传输长度和 registration 边界，而不是立即认定发生重复发送。

NBI PUT 调用同一核心实现但传入 `nbi=true`，提交后不在这里执行 blocking 路径的 `ibgda_quiet(qp)`；调用者必须用后续同步操作建立所需完成边界。NBI 的价值是允许发起与后续计算或其他通信并行推进，它只提供更早返回的机会，并不自动保证重叠：是否真正重叠仍取决于队列容量、progress、依赖关系和硬件并发。

整 warp 合并满足严格条件时，多个 lane 可以一次性预留一段 WQE 区间，并由协作逻辑减少原子操作和 doorbell 次数。它优化的是 control overhead，不改变每个 lane 的 local pointer、remote address、key 和长度，也不改变各请求必须位于连续 ready 前缀中才能提交的正确性条件。

## 9. 与 GET 比较：方向相反，完成要求也不同

GET 与 PUT 共享路径选择、QP 选择、key lookup、WQE 队列和 doorbell 机制，但数据方向相反。[`ibgda_write_rdma_read_wqe`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L1142-L1208) 使用 `MLX5_OPCODE_RDMA_READ`：remote-address segment 描述读取源，data segment 中的 local address 与 `lkey` 描述 NIC 写回的本地目标缓冲区。

blocking GET 返回时，调用者随后就可能读取本地目标，因此实现必须保证 RDMA READ 对 GPU 可见的结果已经准备好。`nvshmemi_ibgda_rma_thread` 对不能跳过 consistency 操作的 blocking GET，在 READ 后追加带 NIC fence 的 DUMP WQE，并把 CQ update 放到 DUMP 上；然后通过 quiet 等待这组 WQE 的完成边界。NIC fence 约束 DUMP 不越过此前的 READ/atomic，DUMP 的完成因而可作为前序结果已经到达本地一致性边界的证据。

NBI GET 不能在每次调用内立即追加同样的阻塞等待，否则会失去 NBI 的意义。固定实现记录需要 consistency 的 `get_head`，把处理延后到更高层的 `ibgda_quiet_with_cst`。这里再次说明 quiet 等待的是一段队列状态并可能补充 consistency WQE，而不是按 API 调用查找某条专属 CQE。

PUT 通常不需要同样的本地结果一致性步骤，因为 NIC 从源缓冲区读数据，调用者等待的是源可复用或远端传输边界；GET 则让 NIC 写本地目标，函数返回前必须处理“GPU 接下来能否正确观察结果”的额外问题。二者 WQE 外观相似，completion contract 却不能互换。

## 10. 与 fetching AMO 比较：结果先落到运行时内部槽位

fetching AMO 既修改远端对象，又把旧值返回给调用者，所以它同时具有远端原子执行和本地结果交付两项要求。[`nvshmemi_ibgda_amo_fetch_impl`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L2930-L3044) 先选择 QP 并解析远端 address/`rkey`，随后不仅预留 WQE 槽位，还从内部 buffer（`ibuf`）预留结果槽。atomic WQE 的本地 data segment 指向该内部槽位，并使用内部 buffer 的 `lkey`；NIC 返回的旧值不会直接写入用户栈上的普通临时变量。

实现根据操作和数据类型在普通 fetch-add/compare-swap 与 masked atomic opcode 之间选择，并可能在 atomic 后追加带 NIC fence 的 DUMP，或因 WQE 跨段布局追加 NOP。提交后先 quiet 到相应边界，再从 `ibuf` 读取结果、进行需要的端序转换，最后释放内部结果槽。结果槽不能在 NIC 完成写回前复用，因此 fetching AMO 除了 SQ 容量，还引入独立的 result-buffer ownership 与背压问题。

本文选择 fetching AMO 只是为了比较请求生命周期，不据此声称所有 NVSHMEM AMO 都由原生 mlx5 atomic 执行。数据类型、操作、NIC capability、QP 类型和 transport 都可能触发不同 opcode 或 fallback；non-fetching AMO、signal、put-with-signal、fence 与完整 ordering 组合由 D04 固定到同一 commit 后逐项分析。

| 维度 | PUT | GET | fetching AMO |
| --- | --- | --- | --- |
| NIC 主要动作 | 读本地源，写远端目标 | 读远端源，写本地目标 | 原子更新远端，并返回旧值 |
| remote segment | 目标地址与 `rkey` | 源地址与 `rkey` | 原子对象地址与 `rkey` |
| local data segment | 用户源地址与 `lkey` | 用户目标地址与 `lkey` | 运行时 `ibuf` 结果槽与内部 `lkey` |
| 代表性 opcode | RDMA WRITE | RDMA READ | atomic opcode，受操作与能力影响 |
| 额外 consistency 工作 | system-memory 源可能先做 system fence | blocking 路径可能追加 fenced DUMP | fetching 路径可能追加 fenced DUMP |
| 返回前必须处理的本地对象 | 源缓冲区复用边界 | 用户目标中的读取结果 | `ibuf` 结果读取、转换与释放 |
| 主要额外资源 | SQ WQEBB | SQ WQEBB，可能再加 DUMP | SQ WQEBB 与内部结果槽 |

## 11. 如何用这条链定位问题

请求失败或性能异常时，应沿生命周期逐层缩小范围，而不是把所有现象都归因于“RDMA 慢”或“IBGDA 没生效”。首先确认路由：目标是否走 P2P/TMA、logical endpoint、proxy transport 还是 IBGDA。其次确认地址与注册：源/目标是否被正确 metadata 覆盖，`lkey/rkey` 与 chunk 边界是否匹配。然后检查 WQE：opcode、DS、address、key、length、endian 和 CQ update 应与 QP 类型一致。最后检查队列边界：`resv_head` 是否因容量阻塞，`ready_head` 是否被前驱空洞卡住，`prod_idx` 是否真正 doorbell，`cons_idx` 是否因 CQ error 或缺少带完成标记的后继 WQE而停滞。

若配置声称启用了 IBGDA，但只有 CPU handler 能工作，应比较 GPU 和 host 侧最后一跳的 DBREC/UAR 更新，而不是把差异解释成 payload staging。若 PUT 调用已经返回但远端消费者读到旧值，应优先检查应用是否建立 remote completion 与发布/消费协议，而不是依赖固定版本 blocking helper 的更强内部等待。若 GET 或 fetching AMO 返回值异常，则应进一步检查 fenced DUMP、consistency head、结果 buffer 的 key、端序与释放时机。

可以用三个问题检查是否真正掌握了诊断边界。第一，线程 B 已写好槽位 101，为什么不能直接 doorbell 到 102？因为槽位 100 可能仍未完成，NIC 只能消费连续初始化完毕的前缀，越过空洞会读取不完整 WQE。第二，为什么 CQ 中没有“本次 PUT 专属 CQE”仍可能正常完成？因为 collapsed CQ 用带完成标记的后继 WQE 报告有序完成边界，软件等待 `cons_idx` 达到目标即可。第三，为什么同一个 `nvshmem_putmem` 有时根本看不到 mlx5 WRITE WQE？因为 device API 先按拓扑和运行时状态选择 P2P/TMA、logical endpoint 或 transport，只有选中 IBGDA 的远端路径才执行本文追踪的编码逻辑。

本文所有例子都是静态源码推演，没有编译 NVSHMEM，也没有在 GPU、ConnectX 或 InfiniBand 环境中抓取 WQE、CQE、doorbell 或性能数据。数字索引例子用于说明状态不变量，不是一次真实运行记录。实际核验应在保留固定 commit、构建参数、CUDA/driver/OFED、GPU/NIC BDF、QP 类型、handler 与日志的前提下进行，并将“源码预期”与“运行观测”分栏记录。

## 12. 结论与掌握标准

一次 IBGDA PUT 不是从 CUDA 调用直接跳到 NIC 数据搬运，而是依次经过运行时路由、proxy PE 与 QP 选择、local/remote registration metadata 查询、WQEBB 数量计算、逻辑槽位预留、mlx5 WQE 编码、连续 ready 前缀发布、DBREC/UAR doorbell，以及 CQ 驱动的完成边界回收。`resv_head`、`ready_head`、`prod_idx` 和 `cons_idx` 分别保护所有权、初始化完整性、NIC 可见进度和槽位复用，任何两个都不应因“通常相等”而混为一谈。

读者完成本篇后，应能从 `nvshmem_putmem` 指出 P2P、logical endpoint、proxy transport 与 IBGDA 的分叉；能解释 `dest/source/pe` 如何形成 remote address、`rkey`、local address 与 `lkey`；能画出 RC WRITE 的三个 segment，并说明 DCI AV 为什么改变 WQEBB 数；能用 WQE 100、boundary 101 解释 doorbell 与 CQ 的 off-by-one；能说明 collapsed CQ 下 quiet 等待的是完成边界；也能准确指出 GET 的本地一致性处理和 fetching AMO 的内部结果槽为何使其生命周期不同于 PUT。

下一篇 D02 将保持相同 commit，把这里略去的多生产者细节展开：原子预留后为何会形成 ready 空洞、warp 合并如何减少控制开销、环形回绕怎样与 `cons_idx` 配合、队列满时谁负责 progress，以及哪些错误状态会演变为活锁或死锁。

## 参考资料

- [NVIDIA NVSHMEM v3.7.2-0 固定源码](https://github.com/NVIDIA/nvshmem/tree/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4)
- [NVIDIA NVSHMEM API：Remote Memory Access](https://docs.nvidia.com/nvshmem/api/latest/gen/api/rma.html)
- [NVIDIA NVSHMEM API：Memory Ordering](https://docs.nvidia.com/nvshmem/api/latest/gen/api/ordering.html)
- [NVIDIA NVSHMEM FAQ](https://docs.nvidia.com/nvshmem/api/latest/faq.html)
- [D00：IBGDA 实现总览](d00-ibgda-implementation-overview.md)
- [B02：NVSHMEM 完成、排序与可见性](b02-completeness-ordering-and-visibility.md)
- [A03：RDMA 请求、完成与缓冲区协议](a03-rdma-requests-and-buffer-protocol.md)
