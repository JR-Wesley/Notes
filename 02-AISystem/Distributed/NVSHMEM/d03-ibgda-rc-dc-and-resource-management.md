# D03：IBGDA 的 RC、DCI/DCT 与资源管理

D00 建立了 IBGDA 对象总图，D01 追踪单个请求，D02 则证明了共享 QP 上的队列并发协议。本篇把视角再向外扩展：运行时为什么同时创建 RC、DCI 和 DCT，不同 transport object 分别保存什么连接状态，GPU 发起请求时如何从目标 PE 选择具体 QP，以及增加 QP 数量为什么既可能降低队列竞争，也会增加 NIC memory、连接建立和 ordering 管理成本。

正文固定 NVIDIA 官方 NVSHMEM `v3.7.2-0`、commit [`3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4`](https://github.com/NVIDIA/nvshmem/tree/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4)。公开配置含义参考 NVIDIA NVSHMEM 当前环境变量与性能指南，但资源数量、handle 交换和 device selector 以固定 commit 为准。文中的数量公式用于解释增长阶，不代表硬件实测占用；没有编译 NVSHMEM，也没有在 NIC 上创建或测量真实 QP。

## 1. 先建立 RC 与 DC 的连接模型

### 1.1 RC：发送 QP 预先绑定一个远端连接上下文

Reliable Connected（RC）QP 在进入 RTS 前已经通过双方交换的 QP number、LID/GID、PSN 等连接信息绑定到具体对端。请求 WQE 因而不需要每次重复描述“本次要连接哪一个远端 transport endpoint”；D01 所示 RC RDMA WRITE 只需 control、remote-address 和 data segment，一个 64-byte WQEBB 即可容纳。RC 的可靠性、顺序与重传由该连接上下文维护，低延迟的代价是每个通信对及每条并行连接都要在 NIC 中保留专用状态。

在固定实现中，[`ibgda_setup_rc_endpoints`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/modules/transport/ibgda/ibgda.cpp#L3339-L3424) 为每个选中 device、每个 RC ordinal 和每个远端 PE 分配 endpoint；本 PE 的 loopback 位置保留但不创建实际 RC。各 PE 先生成 local RC handle，再通过 bootstrap `alltoall` 取得与自己一一配对的 peer handle，之后逐个执行 RESET→INIT→RTR→RTS。这里必须使用 all-to-all 配对，因为 PE A 上“面向 B 的第 i 条 RC”需要的是 B 上对应返回给 A 的连接信息，而不是所有 PE 共用的一份目标描述。

### 1.2 DC：少量 initiator 动态面向许多 target

Dynamically Connected（DC）把角色拆成 Dynamic Connection Initiator（DCI）和 Dynamic Connection Target（DCT）。DCI 是本地发送端 QP，可以在不同请求中面向不同远端 DCT；DCT 是被寻址的目标 endpoint。NVIDIA RDMA 文档将 DC 描述为从 DCI 到 DCT 的可靠传输，用动态连接减少系统所需 QP 数，尤其适合大规模、通信较稀疏的场景。

这种复用并非没有代价。DCI WQE 需要携带由远端 DCT handle 构成的 address vector（AV），固定源码根据硬件能力使用 half-AV 或 full-AV：half-AV 的普通 RMA 可保持一个 WQEBB，full-AV 会使数据 WQE跨两个 WQEBB，并按 D01 所述追加完成标记 WQE。性能指南还指出 DCI 切换到不同 DCT 时可能产生重新建立动态连接的延迟。因此，DC 节省的是长期连接状态，不保证每次请求都比 RC 更短。

可以把两种模型概括为：

```text
RC:
local RC QP ───────── fixed connection context ───────── remote peer RC QP

DC:
                         ┌──────── remote PE 1 / DCT 0
local DCI ── per-WQE AV ─┼──────── remote PE 2 / DCT 1
                         └──────── remote PE k / DCT j
```

图中的“fixed”表示 QP 连接上下文固定到对端，不表示应用只能访问一个远端内存对象；remote address 和 `rkey` 仍随每条 RMA 请求变化。“dynamic”也不表示没有初始化控制面：DCT 必须事先创建，其 handle 必须分发给发起端，DCI 也必须完成本地状态转换后才能提交。

## 2. 初始化阶段如何创建并交换对象

### 2.1 DCT：每个 PE 创建目标，所有 PE收集其 handle

[`ibgda_setup_dct_endpoints`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/modules/transport/ibgda/ibgda.cpp#L2867-L2903) 在每个选中 HCA/device 上创建本地 DCT，提取 DCT number、路径与 AV 所需的 handle，然后通过 bootstrap `allgather` 收集所有 PE 的 DCT handle。与 RC 不同，一个 DCT 的目标描述可以被许多发起 PE 使用，因此把每个 PE 的 handle 集合广播式汇总，正好形成 device 端按 `<target PE, DCT ordinal, device>` 索引的远端表。

固定源码随后把前一部分 DCT AV metadata 放入 constant memory，超出 [`NVSHMEMI_IBGDA_MAX_CONST_DCTS`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/modules/transport/ibgda/ibgda.cpp#L2926-L2988) 的部分分配到 device global memory。由此可见，DC 将本地 QP 数量从按 peer 增长转化为“少量 DCI/DCT + 按 peer 保存目标 metadata”；连接状态扩展性改善了，但远端 DCT 描述仍然随 PE 数增长，不能把 DC 误解为完全没有 per-peer state。

### 2.2 DCI：本地创建，不与单一远端配对

[`ibgda_setup_dci_endpoints`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/modules/transport/ibgda/ibgda.cpp#L3071-L3100) 为本 PE 的每个选中 device 创建设定数量的 DCI，并依次执行 RESET→INIT→RTR→RTS。它没有像 RC 一样为每个远端执行 peer handle all-to-all，因为 DCI 的目的端在请求 WQE 的 AV 中选择；远端可达信息已经由 DCT handle 汇总提供。

每个 DCI 仍是一条完整发送资源：device state 中需要 QP number、WQ、DBREC、doorbell mapping、CQ association、队列管理变量以及 fetching operation 使用的内部 buffer。减少 DCI 数会让更多 GPU 执行者共享这些对象并增加 D02 所述 reservation、ready CAS 和 post-send 竞争；增加 DCI 数则分散竞争，但消耗更多 NIC 与 GPU-visible control memory。

### 2.3 RC：对象数量按远端 PE 数增长

固定实现为每个远端 PE 创建 `NVSHMEM_IBGDA_NUM_RC_PER_PE` 条 RC，并为每个选中 device 重复这组资源。设 job 中有 $P$ 个 PE，每个 PE 选择 $H$ 个 HCA/device，每个远端配置 $R$ 条 RC，则单个 PE 实际创建的非 loopback RC endpoint 约为

$$
N_{RC,local}=H\times R\times(P-1),
$$

全 job 的 endpoint 数约为

$$
N_{RC,job}=H\times R\times P\times(P-1).
$$

公式计算的是该源码布局下的 endpoint 增长阶，不等于 NIC 厂商给出的精确字节数，也没有扣除不同设备或自定义 QP API 带来的变化。关键结论是 RC 的本地硬件对象随 $P$ 线性增长、全局总量近似二次增长；DCI/DCT 的本地 QP 数主要由配置和 HCA 数决定，而 DCT metadata 表随 $P$ 线性增长。

## 3. device state 如何表达资源集合

[`nvshmemi_ibgda_device_state_t`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/device_host_transport/nvshmem_common_ibgda.h#L285-L340) 保存 `num_shared_dcis`、`num_exclusive_dcis`、`dci_map_type`、`ndcts_per_pe`、`num_rc_per_pe`、`num_default_rc_per_pe`、选中 device 数和指向 DCI、RC、DCT、CQ 与 `qp_group_switches` 的指针。host 初始化完成后，[`ibgda_post_gpu_device_state`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/modules/transport/ibgda/ibgda.cpp#L3699-L3744) 根据实际 handle 数量计算这些字段，而不是让 kernel 再查询 host verbs 对象。

DCI 与 RC 的每个 device QP descriptor 又包含 QP type、QPN、device index、fetch `ibuf`、WQ base、WQ 深度、DBREC、BF/UAR、CQ pointer 和 D02 的 management variables。DCT 在 device 端则压缩为 mlx5 AV 数据，因为它主要供 DCI WQE复制远端目标描述。这种不同结构正反映了角色差异：DCI/RC 是主动产生 WQE 的发送队列，DCT metadata 是被 WQE引用的目的端信息。

资源不能只按“QP 个数”估算。增加一条 DCI 或 RC 往往还意味着增加或扩大 SQ、CQ、DBREC、doorbell mapping、management state 和内部 fetch slots；增加 DCT 会增加目标对象、共享接收侧资源与所有 PE 保存的 AV metadata。本文只确认这些对象之间的依赖，未根据结构体大小推算 NIC SRAM 或系统内存的真实占用，因为大量资源存在设备固件侧状态、对齐、page mapping 与实现共享，单靠 C 结构体大小会得到误导性的数字。

## 4. 运行时首先在 RC 与 DCI 之间选择

[`ibgda_get_qp`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L1822-L1830) 的主分支很直接：只要 `num_rc_per_pe > 0` 且目标不是本 PE，就调用 `ibgda_get_rc`；否则调用 `ibgda_get_dci`。因此在该固定 commit 中，只要为远端 PE 启用了 RC，普通远端 RMA/AMO 的默认 QP 选择会优先 RC，而不会按消息大小在 RC 与 DC 间动态切换。

即使 RC 已启用，DCI 并未完全消失。本 PE 目标以及某些 consistency 操作仍需要 DCI；当前官方环境变量说明也明确指出，正数 `NVSHMEM_IBGDA_NUM_RC_PER_PE` 使普通连接使用 RC，而 DCI 保留用于 consistency。D04 将详细分析 GET/atomic consistency、fence 与 quiet；本篇只需看到资源配置会扩大需要扫描和协调的 QP 集合。

将 `NVSHMEM_IBGDA_NUM_RC_PER_PE=0` 会使 `ibgda_is_rc_enabled()` 为假，远端请求转而使用 DCI/DCT。它是明确的传输资源选择，不等于“关闭可靠性”：DC 本身仍提供可靠传输，只是连接状态由固定 peer-bound RC 改为动态 DCI→DCT。

## 5. DCI 的 exclusive、shared 与映射策略

### 5.1 derived ID 决定优先候选

[`ibgda_get_dci`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L1741-L1793) 先按 `dci_map_type` 生成执行者 ID：

| `IBGDA_DCI_MAP_BY` | derived ID 的基础 | 共享域含义 |
| --- | --- | --- |
| `cta` | 线性 CTA ID | 不同 CTA 尽量映射不同 exclusive DCI；资源不足时进入 shared pool |
| `sm` | 当前 SM ID | 同一 SM 可驻留多个 CTA，所以源码直接标记跨 CTA 共享 |
| `warp` | CTA ID 与 block 内 warp ordinal 组合 | warp 粒度区分更细，但需要更多 DCI 才能保持 exclusive |
| `dct` | warp group 与目标 DCT 分组组合 | 试图保持 DCI 与一组 DCT 的关系，但源码标记为跨 CTA 共享 |

多 device 情形还会通过 `qp_group_switches` 轮转 `dev_offset`，把相应 group 的请求分散到已初始化 devices。随后，如果 derived ID 落在 `num_exclusive_dcis` 范围内，直接选择对应 DCI；否则把 ID 对 `num_shared_dcis` 取模，落到 shared pool，并把 `out_shared_among_ctas` 设为 true。

“exclusive”应理解为映射算法为某个 ID 提供的专用候选，不应扩大成某个 CUDA thread 对 QP 的永久所有权。kernel 数量、CTA ID 空间、自动配置、device 轮转和资源不足都可能使后续执行者走 shared pool。真正决定 D02 应使用 GPU-scope 还是 block-scope 原子与 fence 的，是 `out_shared_among_ctas`，而不是环境变量名称本身。

### 5.2 shared pool 是正确性后备，也是竞争汇聚点

固定配置要求至少存在 shared DCI；当 derived ID 超出 exclusive 数量时，modulo 映射保证仍有可用发送队列，而不是因 GPU 并发规模大于预估值直接失败。这为任意 CTA/warp 提供功能后备，但所有回退执行者会竞争少量 shared DCI 的 `resv_head`、`ready_head`、post-send lock、SQ depth 和 fetch slots。

因此，增加 exclusive DCI 可能降低高并发 kernel 的热点，却不一定提升稀疏小消息延迟；更多 DCI 会增加资源，并可能让针对同一 PE 的请求分布到多个 ordering domain。减少 shared DCI 又可能把大量溢出映射压到一个 QP。应根据实际 CTA/warp 并发、目标分布和消息模式测量，而不能只看 SM 数决定最优值。

### 5.3 `dct` mapping 并非把 DC 变成 RC

性能指南指出，按 DCT 映射可让 DCI 更稳定地对应特定 DCT 或较小 DCT 子集，从而减少频繁切换动态连接的开销。它仍然使用 DCI WQE 与 DCT AV，仍需要 DC 目标 metadata，也可能因 DCI 数少于 DCT group 数而共享；因此不会获得 RC 的完整 per-peer fixed connection resource model。

固定源码的 `dct` 分支用 target PE 对应的 `dct_id`、warp group、`num_dct_groups` 和 device 数构造映射 ID，并明确把 QP 标记为跨 CTA 共享。这个实现事实比“一个 DCI 永远锁定一个 DCT”的简化说法更精确：实际一对一程度取决于 DCI、DCT、group 和 device 数量。

## 6. RC selector 与 `RC_MAP_BY` 的固定版本边界

[`ibgda_get_rc`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L1795-L1820) 对默认或 `NVSHMEMX_QP_ANY` handle 使用 `qp_group_switches` 递增计数，在允许的 RC ordinal 中轮转，再按 `ordinal * npes + target_pe` 定位对应 peer 的 RC。函数无条件把 `out_shared_among_ctas=true`，所以 D02 的 queue protocol 必须采用跨 CTA 安全的原子和 flush。

值得特别保留一个源码边界：该 commit 的 host 初始化会解析 `NVSHMEM_IBGDA_RC_MAP_BY=cta|sm|warp`，验证并把枚举保存进 device state；但默认 device selector `ibgda_get_rc` 本身没有读取 `rc_map_type`，而是使用上述共享切换计数器。当前官方文档把 RC mapping 选项描述为 CTA/SM/warp 映射，然而仅凭此固定 selector，不能证明每次默认请求严格按相应 CUDA ID 选择 RC。若要断言某版本的实际映射效果，应继续核对完整调用路径、custom QP handle 行为和运行 trace，而不能从变量名推导。

`num_default_rc_per_pe` 与 `num_rc_per_pe` 也有区别。前者记录环境配置创建的默认 RC 数，后者可包含后续 QP-specific API 增加的总 RC 数。默认 handle 在前者范围轮转，`NVSHMEMX_QP_ANY` 可在总集合内选择，显式 handle 则指向指定资源。应用若使用 custom QP，就必须把其生命周期和 quiet/fence 范围纳入分析，不能继续假设所有请求都落在默认 QP 集合。

## 7. DCT 选择与 WQE 目的端编码

当选中 DCI 后，[`ibgda_get_dct_id`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L1731-L1739) 根据目标 PE、CTA ID、每 PE 的 DCT 数和当前 device index 选择 DCT。可将其索引结构理解为

$$
dct\_id = target\_pe\times(KH)+(cta\_id\bmod K)\times H+device\_idx,
$$

其中 $K$ 是每 PE、每 device 的 DCT 数，$H$ 是选中 device 数。随后 DCI WQE writer 从 constant/global metadata 表取得该 DCT 的 AV segment，并与 remote address、`rkey`、local address 和 `lkey` 一起写入 WQE。

RC WQE 不走这一步，因为目标 transport endpoint 已在 RC QP context 中；它仍然需要 remote address 与 `rkey` 来指出对端注册内存。DCT ID 与 `rkey` 解决的是两层不同问题：前者选择远端 DC transport endpoint 和路径，后者授权 NIC 访问特定远端 memory region。把 DCT number 当成内存权限，或把 `rkey` 当成连接地址，都会混淆 transport routing 与 memory protection。

## 8. 多 QP 会改变排序与完成成本

单个 QP 内的 WQE 有自然顺序，D02 的 `ready_head/prod_idx/cons_idx` 也都是 per-QP 状态。两个请求若落到不同 RC 或 DCI，它们不再共享同一 SQ 顺序；增加 QP 能扩大可并行提交资源，却同时把操作拆到多个 ordering domain。固定源码在 [`nvshmemi_ibgda_fence`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L3479-L3513) 中明确注释：此前多个 QP 可能面向同一 PE，所以需要处理这些 QP；当总相关 QP 数不超过一条时，顺序自然由该 QP 保证。

同样，quiet 不能只等待“最近一次选择的 QP”。[`nvshmemi_ibgda_qp_quiet`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L3411-L3475) 根据 ALL、DEFAULT、显式 handles 与 PE hint 遍历相应 DCI/RC 集合，并对每条相关 QP 执行 consistency-aware quiet。QP 越多，潜在扫描和同步集合越大；这正是“降低数据路径竞争”与“增加控制/ordering 成本”之间的另一层权衡。完整 fence/quiet API 语义留给 D04，本篇只确认资源分片会改变实现工作量。

## 9. 如何估算资源增长与选择策略

设每 PE 选择 $H$ 个 devices，每 device 创建 $I$ 个 DCI、$K$ 个 DCT，每远端 PE 创建 $R$ 个 RC。忽略 loopback array 空位和 custom QP 后，单 PE 的硬件 endpoint 数量级为：

| 模式 | 本地主动/目标 endpoint 数量级 | per-peer metadata | 主要竞争与代价 |
| --- | --- | --- | --- |
| RC | $HR(P-1)$ 个实际 RC | RC peer handles/连接上下文 | 连接和 NIC state 随 PE 增长；单请求 WQE 更紧凑、通常低延迟 |
| DC | $HI$ 个 DCI + $HK$ 个本地 DCT | 约 $HPK$ 份 DCT AV metadata | DCI 被更多执行者/目标复用；WQE 带 AV，可能有动态切换延迟 |
| RC + DC | 普通远端请求主要用 RC，DCI/DCT 仍为 self/consistency 等路径保留 | 两类 metadata 都存在 | 获得 RC 数据路径，同时承担两套资源与跨 QP ordering 管理 |

这个表用于判断增长方向，不能替代测量。NIC firmware 为一条 RC、DCI 或 DCT 分配多少 SRAM/cache，CQ 是否独立、WQ/doorbell page 如何映射、multi-port 如何分摊，都取决于硬件、driver 和固定构建。可靠实验应记录实际 PE 数、选中 HCA、每类 QP 配置、创建日志、device state 数量、初始化时间、GPU/NIC memory 占用、请求到达模式和延迟/带宽，而不是只报告一个环境变量。

对于小到中等、全连接且延迟敏感的 job，RC 的专用连接通常更有吸引力；对于 PE 很多但每个时刻只访问少量对端的稀疏通信，DC 的资源扩展性可能更重要。若大量 CTA 集中访问少数 PE，则 shared DCI/RC 的队列竞争可能成为瓶颈；若目标高度分散，过少 DCI 又可能频繁切换 DCT。以上是由连接模型推出的选择原则，实际拐点必须在目标 GPU、ConnectX、CUDA、OFED 与网络拓扑上测得。

## 10. 诊断问题与常见误解

如果启用了 RC，为什么日志中仍然看到 DCI/DCT 被创建？因为固定 selector 只让远端普通请求优先 RC，self target 和 consistency 等内部路径仍可能使用 DCI；运行时不能因此省略 DC 资源。看到对象存在也不等于某条 payload 请求实际走过它，应结合 QP type、QPN 或 WQE AV 观测。

如果把 DCI 数增加到与远端 PE 数相同，是否就等价于 RC？不等价。DCI 仍通过 DCT AV 指定目标，远端仍是 DCT，连接与 WQE 格式仍属于 DC；一对一映射最多减少目标切换，不会自动变成 peer-bound RC context。

如果增加 RC 或 DCI 后吞吐没有提升，应先判断瓶颈是否真的在共享 QP。若 `ready_head` 与 `prod_idx` 没有竞争性停滞，限制可能来自链路、PCIe、memory bandwidth、消息粒度、batch 或目标热点；更多 QP 只会增加资源。反之，如果同一 QP 的 reservation/CAS/doorbell lock 明显集中，才有理由尝试扩大 QP 集合或改变映射。

如果 `NVSHMEM_IBGDA_RC_MAP_BY=sm`，能否直接断言固定 commit 中每个 SM 使用独立 RC？不能。该选项在 host 侧被解析和保存，但已核对的默认 `ibgda_get_rc` 使用共享轮转计数器且标记跨 CTA 共享；必须用完整源码与运行 trace 证明最终选择，不能把配置意图冒充执行证据。

## 11. 结论与掌握标准

RC 用 per-peer 连接状态换取紧凑 WQE 和较低动态连接开销，其本地 endpoint 数随远端 PE 数增长；DC 则用少量可复用 DCI 面向每个 PE 的 DCT，把主要硬件 QP 数从 peer 数中解耦，但仍需保存 per-peer DCT AV metadata，并承担更大的共享竞争、AV WQE 和目标切换成本。NVSHMEM 固定实现通过 DCT allgather、RC alltoall、device state 下发和 `ibgda_get_qp` 把初始化资源映射到稳态请求。

读者完成本篇后，应能说明 RC handle 为什么需要 peer 配对而 DCT handle 适合 allgather；能计算 RC 与 DC 的 endpoint 增长阶；能区分 DCI、DCT、remote address 和 `rkey` 的职责；能解释 CTA/SM/warp/DCT mapping、exclusive 与 shared pool 如何改变 D02 的竞争域；能说明多 QP 为什么既提高并发机会又扩大 fence/quiet 的处理集合；也能指出固定 commit 中 `RC_MAP_BY` 配置描述与默认 selector 证据之间仍需运行核验的边界。

下一篇 D04 将继续使用同一 commit，分别追踪 fetching/non-fetching AMO、put-with-signal、signal-op、GET consistency、fence 和 quiet，把“选中了哪条 QP”进一步连接到 opcode、额外 WQE、完成记录与跨 QP ordering。

## 参考资料

- [NVIDIA NVSHMEM v3.7.2-0 固定源码](https://github.com/NVIDIA/nvshmem/tree/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4)
- [NVIDIA NVSHMEM：Environment Variables](https://docs.nvidia.com/nvshmem/api/latest/gen/env.html)
- [NVIDIA NVSHMEM：Performance Best Practices](https://docs.nvidia.com/nvshmem/release-notes-install-guide/best-practice-guide/performance.html)
- [NVIDIA RDMA Aware Networks Programming User Manual](https://docs.nvidia.com/rdma-aware-networks-programming-user-manual-1-7.pdf)
- [D00：IBGDA 实现总览](d00-ibgda-implementation-overview.md)
- [D01：从一次 PUT 追踪 IBGDA 请求生命周期](d01-ibgda-request-lifecycle.md)
- [D02：IBGDA 多生产者并发与发送队列管理](d02-ibgda-concurrency-and-queue-management.md)
