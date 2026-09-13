# D00 IBGDA 实现总览：从 device API 到 mlx5 工作队列

## 本课要解决的问题

C01 已经把 API caller、NIC work submitter 和 payload data mover 分成三条独立轴，C02 则说明如何用运行证据确认实际 transport。本课开始进入固定版本源码，回答一个更具体的问题：当 CUDA thread 发起跨节点 NVSHMEM RMA 时，IBGDA 如何把 `<对称地址, 目标 PE>` 变成 NIC 能执行的地址、key 和 WQE，又如何在许多 GPU 线程共享有限 QP 的条件下安全地提交、通知 NIC 并判断完成？

本文的目标是建立源码地图和因果总图，而不是穷举每个 opcode。读完后，读者应当能区分 host 初始化期与 device 稳态期，能指出 QP、WQ、WQE、DBREC、UAR、CQ、CQE、lkey/rkey、DCI/DCT 和 RC 在同一条请求路径中的位置，也能解释为什么“写完 WQE”“推进 producer index”“敲 doorbell”和“收到完成”是四个不同事件。一次 PUT 的逐语句追踪留给 D01，多生产者算法的证明留给 D02，RC 与 DC 的资源规模分析留给 D03，AMO、signal、fence 和 quiet 的完整实现边界留给 D04。

## 固定版本、证据范围与阅读约定

本文于 2026-09-13 静态核对 NVIDIA 官方仓库 `NVIDIA/nvshmem` 的 release tag [`v3.7.2-0`](https://github.com/NVIDIA/nvshmem/tree/v3.7.2-0)，固定 commit 为 [`3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4`](https://github.com/NVIDIA/nvshmem/commit/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4)。正文中“该版本”“源码中”均特指这一提交；源码链接也固定到完整 commit，避免后续分支变化改变语义。公开 API 契约仍以 NVIDIA 文档为准，本文从内部符号得出的结论不构成跨版本 ABI 或实现承诺。

本次检查没有编译 NVSHMEM，没有在 GPU、ConnectX NIC 或 InfiniBand/RoCE 网络上运行程序，也没有采集 WQE、CQE、doorbell 或性能 trace。因此，“调用”“写入”“轮询”等措辞是在描述固定源码所实现的控制流，不是本机实验观测。NVSHMEM 3.7.2 release notes 还明确提示：从 3.5.21 开始 RC-connected QP 的内部布局发生变化，启用 IBGDA 时会造成 ABI 兼容性破坏。这正是实现笔记必须同时固定 host library、device library、transport plugin 和源码版本的原因。

阅读本文前应掌握 A01—A03 中的 RDMA 基础对象和 B00—B03 中的 NVSHMEM/CUDA 执行模型。这里先压缩复习一次：PD 是一组网络资源与 memory keys 的保护边界；MR 把一段内存登记给 RNIC 并产生本地访问用的 lkey、远端授权用的 rkey；QP 包含工作队列，发送侧把硬件可读的 WQE 放入 SQ；CQ 是完成记录所在的队列，CQE 是其中的硬件记录。IBGDA 没有取消这些对象，而是改变了谁能填充 WQE、谁能推进提交位置以及谁能轮询 CQ。

## 一、先把完整实现分成控制面和稳态数据面

IBGDA 的“GPU initiated”不表示所有初始化也在 GPU 上完成。固定源码呈现的是明显的两阶段结构：host 负责装载 plugin、发现 HCA、检查 mlx5 能力、创建 PD/CQ/QP、注册或映射控制缓冲区、交换连接信息和 memory keys，并构造一份 GPU 可消费的 transport state；kernel 中的 device code 才在稳态请求上选择 QP、查地址和 key、填 WQE、推进队列并触发 doorbell 或通知异步 CPU handler。

```text
初始化与建连阶段（host）

环境变量 / build capability
        |
        v
装载 nvshmem_transport_ibgda.so -> nvshmemt_init
        |
        +-> 枚举 mlx5 HCA、PD 与端口，检查 atomic / UAR mapping capability
        +-> 选择 NIC handler：GPU 或 CPU fallback
        +-> connect_endpoints：创建 DCT、DCI/RC、CQ、WQ、DBREC、UAR、internal buffer
        +-> 登记 heap/local buffer，汇集各 PE 的 lkey/rkey 与 peer metadata
        `-> 形成 nvshmemi_ibgda_device_state_t 并复制到 device constant symbol

稳态请求阶段（device）

NVSHMEM device API
        |
        v
transfer dispatch -> nvshmemi_ibgda_rma / amo / ordering routine
        |
        +-> 选择 RC 或 DCI，DCI WQE 再携带目标 DCT 信息
        +-> symmetric offset -> remote address；chunk/device/PE -> lkey/rkey
        +-> reserve WQ slots -> fill WQE -> publish ready_head
        +-> update prod_idx -> DBREC -> UAR/BF doorbell
        `-> poll collapsed CQ -> advance cons_idx -> slot becomes reusable
```

这张图刻意把“payload data mover”放在 NIC 而不是 GPU：GPU 写的是 WQE 和控制寄存器，NIC 读取 WQE 后才依据 lkey/rkey 对本地或远端内存执行 DMA。若 NIC handler 退回 CPU，GPU 仍然生成或发布工作，但由 CPU progress path 根据 GPU 可见的 producer index 更新 DBREC 并敲 doorbell；因此 IBGDA transport 被选中，并不充分证明每个 doorbell 都由 GPU SM 直接写入。

## 二、host 如何把 IBGDA 安装为 device transport

核心 transport 初始化在 [`nvshmemi_transport_init`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/host/transport/transport.cpp#L61-L312) 中进行。当 `NVSHMEM_IB_ENABLE_IBGDA` 打开且构建包含 IBGDA 支持时，代码动态装载 `nvshmem_transport_ibgda.so.<major>`，通过 `dlsym` 取得 plugin 导出的 `nvshmemt_init`，再传入 CUDA symbol table 和 transport interface version。plugin 初始化成功后，core 把 transport-specific host state 指向 `nvshmemi_ibgda_device_state`，并把公共 device state 的 `selected_device_transport` 设为 `NVSHMEMI_DEVICE_TRANSPORT_TYPE_IBGDA`。若同一运行同时请求 IBGDA 和 GPUNetIO GDAKI，该路径直接报错，因为 device 侧只能选择一个对应的 transport dispatch。

plugin 自己的 [`nvshmemt_init`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/modules/transport/ibgda/ibgda.cpp#L4670-L5063) 还不是“创建所有对端连接”的同义词。它先检查 plugin major version，解析 QP depth、batch size、fetch slots 和 NIC handler 等参数，枚举可用 IB devices，过滤非 mlx5 或缺少必要能力的设备，并安装 `connect_endpoints`、memory registration、progress 和 finalize 等 host callbacks。值得注意的是，其 host-side `rma`、`amo`、`fence`、`quiet` callbacks 都被置为 `NULL`：这并非表示 IBGDA 不支持这些操作，而是说明其主要 RMA/AMO/ordering 实现不通过传统 host transport operation callback，而在编译进 device library 的 inline 路径中完成。

真正创建连接资源的是 [`nvshmemt_ibgda_connect_endpoints`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/modules/transport/ibgda/ibgda.cpp#L4234-L4316)。首次调用按 global setup、per-device calculation、resource creation、endpoint connection 和 GPU state setup 推进；后续调用可以走 RC-only 扩展路径。这里创建或整理的不是单一“endpoint”，而是一组相互引用的对象：DCT 接收目标、DCI 或 RC 发送 QP、每个发送 QP 对应的 CQ、SQ backing memory、DBREC、UAR 映射、fetch/internal buffer，以及 GPU 侧用于选择 QP 和定位这些对象的数组。

### 2.1 CQ、WQ、DBREC 与 UAR 为什么必须同时准备

[`ibgda_create_cq`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/modules/transport/ibgda/ibgda.cpp#L1415-L1574) 分配 CQ buffer 和 CQ doorbell record，将它们映射给 NIC，并通过 DEVX 创建 collapsed CQ。CQ buffer 同时具有 GPU mapping，因此 device code 可以直接读取 CQE；CQ 自己虽然需要一个 UAR 完成创建，但注释明确说 IBGDA 不使用这个 CQ UAR，所以不必映射给 GPU。这里的 CQ UAR 与发送 QP 用来敲 doorbell 的 UAR 不是同一个角色。

[`ibgda_create_qp`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/modules/transport/ibgda/ibgda.cpp#L2250-L2419) 通过 DEVX 创建 DCI 或 RC QP。它把 WQ backing memory、发送 CQ number、PD number、DBREC umem、SQ depth 和 UAR page 写入 QP context；发送 QP 的 UAR 会根据最终 handler 映射到 GPU 或留给 CPU。由此可以把四个常混淆的对象分开：WQ memory 存放 WQE 内容，DBREC 存放“下一个空 WQEBB”的软件记录，UAR/BF 是通知 NIC 有新工作的 MMIO 映射，CQ buffer 则由 NIC 写入完成进度。

### 2.2 host state 如何变成 device 可用状态

公共头文件中的 [`nvshmemi_ibgda_device_state_t`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/device_host_transport/nvshmem_common_ibgda.h#L224-L340) 是 host 与 device 之间的关键交接对象。它保存 QP 数量和映射策略、batch 大小、memory granularity、handler 形态，以及 DCI、RC、DCT、CQ、lkey 和 rkey 的指针或小型缓存。前一部分 keys/DCTs 放在结构的 `constmem` 数组中，超过固定容量的部分由 `globalmem` 指针引用；这里的命名表达访问布局，不能误解成所有结构字段都自动具有 CUDA constant-memory 的独立生命周期。

host 端的 [`ibgda_setup_gpu_state`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/modules/transport/ibgda/ibgda.cpp#L3864-L3954) 建立 DCT、DCI、RC、CQ 和 QP-switch arrays 的 GPU 视图，`ibgda_post_gpu_device_state` 再写入数量、映射方式和 `use_async_postsend` 等标志。core 的 [`nvshmemi_update_device_state`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/host/init/init.cu#L209-L277) 最终把 host 侧 `nvshmemi_ibgda_device_state` 复制到已登记的 device transport state symbols。device 代码通过 `nvshmemi_ibgda_device_state_d` 取得这份状态，而不是在每次 PUT 时回到 host 查询 QP 或 rkey。

## 三、device dispatch：公开 API 怎样进入 IBGDA 内联路径

[`transfer_device.cuh.in`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in#L96-L175) 是最适合开始追踪的分流点。以 RMA 为例，`nvshmemi_transfer_rma_p`、`nvshmemi_transfer_rma_g` 和通用 `nvshmemi_transfer_rma` 都检查 `nvshmemi_device_state_d.selected_device_transport`：IBGDA 分支调用 `nvshmemi_ibgda_*`，GPUNetIO 分支调用对应 GDAKI 实现，默认则进入 proxy。它说明 device API 是统一入口，具体提交机制由运行时写入的 device state 选择；不能仅凭调用了 `nvshmem_*` device function 就断言 IBGDA 生效。

固定源码的大部分稳态逻辑集中在 [`ibgda_device.cuh`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh)。这不是一个独立运行的 `.cu` transport worker，而是被 device transfer path 包含的实现头。模板参数把 thread/warp/block scope、PUT/GET、blocking/NBI、是否支持 half AV segment 等差异编译进具体路径，所以阅读时应先选定一个实例，不能把所有模板分支同时当成一次请求实际执行的步骤。

## 四、第一步不是写 WQE，而是选 QP 并准备地址与 key

### 4.1 RC、DCI 和 DCT 分别解决什么问题

RC QP 是与特定远端连接绑定的可靠连接资源；它通常提供更直接的目标关系，但每个 PE 对多个远端、多个 NIC 或多个并发组创建 RC 会放大资源规模。DC 模式把发起端 DCI 与目标端 DCT 分开：一个 DCI 可以在不同 WQE 中指定不同 DCT，从而用较少发起资源覆盖许多目标，但每个请求需要携带 address-vector/DCT 信息，并承担连接切换和共享竞争成本。

device helper [`ibgda_get_qp`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L1731-L1830) 给出了该版本的实际选择规则：若 `num_rc_per_pe > 0` 且目标不是本 PE，选择 RC；否则选择 DCI。DCI 可按 CTA、SM、warp 或 DCT group 映射，并区分 exclusive 与 shared pool；RC 数组则依据 QP handle、目标 PE、每 PE 的 RC 数量和设备数定位。这里的 DCT 不作为发起 SQ 使用，而是在 DCI WQE 的 address vector 中标识远端动态连接目标。

这个规则还有一个容易漏掉的边界：即使配置启用 RC，代码对本 PE 的某些一致性操作仍可选择 DCI；官方环境变量说明也指出，`NVSHMEM_IBGDA_NUM_RC_PER_PE` 为正时 DCI 仍用于 enforcing consistency。因此“RC enabled”不能简化成“DCI/DCT 完全不存在”。D03 将进一步量化两类资源与 mapping policy。

### 4.2 symmetric address 如何变成远端 NIC 地址

API 层传入的是本 PE 上的 symmetric address 和 `dst_pe`。[`ibgda_get_raddr_rkey`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L1887-L1924) 先以本地 heap base 计算 `roffset`，再用 memory granularity、目标或 proxy PE 以及 NIC index 找到对应 rkey；远端 NIC address 则由 `peer_heap_base_remote[proxy_pe] + roffset` 构造，并在 rail optimization 的间接目标情形下加入节点内 PE 的 heap displacement。这个公式是固定实现的内部路径，不应倒推为 NVSHMEM API 对所有 transport 的地址公式。

本地源地址也需要 lkey。[`ibgda_get_lkey`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L1832-L1885) 对 symmetric heap 以 chunk 和 NIC index 查表；对已注册但不属于 symmetric heap 的 local-only buffer，则遍历 local-only memory handle list。两条路径都返回当前 key 覆盖到哪里，RMA routine 取本地 chunk、远端 chunk 与剩余长度的最小值作为本次 WQE 的传输长度。于是一次逻辑 PUT 可能跨 registration chunk 被拆成多个 WQE；“一个 API 调用”等于“一个 WQE”并不成立。

lkey 与 rkey 不能互换。lkey 授权本地 NIC 读取或写入本地 SGE 指向的内存，rkey 授权远端操作访问目标 MR；remote address 给出位置，rkey 给出权限域。host callback [`nvshmemt_ibgda_add_device_remote_mem_handles`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/modules/transport/ibgda/ibgda.cpp#L4457-L4553) 按 heap chunk、PE 和 device 整理远端 rkeys，并优先把前若干项放进 device state 的小型缓存，其余放入 GPU global memory。这解释了为何 bootstrap/registration 属于前置控制面：若远端 base 或 rkey 没有先交换到本 PE，GPU 无法仅凭一个 CUDA pointer 构造合法 RDMA WQE。

## 五、WQE 从“占到位置”到“对 NIC 可见”经历四个阶段

`nvshmemi_ibgda_device_qp_t` 不只保存 qpn 和 WQE buffer pointer，还包含一组管理变量。固定源码对发送队列维护四个单调逻辑索引：

| 索引 | 源码注释中的含义 | 本课中的状态解释 |
| --- | --- | --- |
| `resv_head` | 最后已预留 WQE index + 1 | 生产者已取得槽位所有权，但槽内可能尚未填完 |
| `ready_head` | 最后已就绪 WQE index + 1 | 到该位置之前不存在 publication hole，可考虑提交 |
| `prod_idx` | 已 post 的 WQE index + 1 | 已向 doorbell path 公布的提交边界 |
| `cons_idx` | 已 poll 的 WQE index + 1 | 硬件完成已被观察，旧槽可据此回收 |

这四者对应的正常偏序是：

\[
\texttt{cons\_idx} \le \texttt{prod\_idx} \le \texttt{ready\_head} \le \texttt{resv\_head}.
\]

这里是对源码状态含义的教学归纳，不是源码中的显式 assert。并发执行时等号可以断开：线程 A 先预留前面的槽后暂停，线程 B 可以填完后面的槽，但 `ready_head` 不能越过 A 留下的空洞；若它越过，NIC 可能读取半写 WQE。B04 中的 publication hole 在这里变成了实际的 WQ 管理问题。

### 5.1 reserve：取得槽位但尚未发布

[`ibgda_reserve_wqe_slots`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L1926-L1951) 通过 GPU-scope 或 CTA-scope atomic add 推进 `resv_head`，返回调用者独占的起始逻辑 index。随后 `ibgda_wait_for_slot_availability` 根据环形 SQ 深度检查较早 WQE 的完成进度，防止 producer 绕回覆盖 NIC 尚未消费的槽。逻辑 index 使用 64 bit 单调计数，真正定位 WQ buffer 时再按有限队列深度回绕；这比直接只维护一个模 N 下标更容易区分不同世代。

### 5.2 fill：按 opcode 写入硬件布局

通用 cooperative RMA routine [`ibgda_rma`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L2244-L2425) 先计算跨 registration chunks 的 WQE 数量，由 warp leader 一次预留连续槽位，再由参与 lanes 分别写 RDMA WRITE 或 RDMA READ WQE。RC WQE 与 DCI WQE 的段数可能不同；若 DCI 不支持 half AV segment，一个逻辑 command 可占两个 WQEBBs。代码只在这一组最后的 WQE 上请求 CQ update，并可能附加 NOP 或用于 consistency 的 DUMP WQE，所以 CQE 数量既不等于 API 数量，也不一定等于 WQE 数量。

### 5.3 publish：填好自己的槽还不够

[`ibgda_submit_requests`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L1641-L1686) 在必要的 memory fence 后，以 compare-and-swap 要求 `ready_head` 恰好等于本次 `base_wqe_idx`，才能把它推进到 `base + count`。这相当于按预留顺序提交连续区间：较后的生产者可以先完成填充，但必须等待之前的空洞闭合。若 QP 跨 CTA 共享，源码采用更强的 GPU-scope flush，因为最终敲 doorbell 的可能是另一个 CTA。

### 5.4 post：批量边界决定何时通知 NIC

`ready_head` 推进后并不必然立刻触发 doorbell。源码在没有并发尾随请求、跨过 `num_requests_in_batch` 边界，或本次请求数量达到 batch threshold 时执行 post-send；否则把若干 ready WQEs 合并到后续通知中。环境变量 `NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH` 因而调整的是通知聚合策略，不是改变每个 API 的 NVSHMEM completion contract。值较大可能减少 doorbell 次数，却也可能增加孤立小请求等待被提交的时间；实际性能结论必须由 E01 的受控测量给出。

## 六、GPU handler 与 CPU handler 在 doorbell 处真正分叉

当最终 handler 是 GPU 时，host 把 QP 的 DBREC 与 UAR/BF 地址填入 device QP view。[`ibgda_post_send`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L1562-L1619) 先在 `post_send_lock` 下用 `atomicMax` 推进 `prod_idx`，执行内存栅栏，更新 DBREC，再向 BF/UAR 写入包含 producer index 与 qpn 的 control segment。顺序的因果关系是：WQE stores 必须先对设备可见，DBREC 才能声明新的队列边界，最后的 MMIO store 才能通知 NIC 取走工作。只看到了 UAR 写入而没有验证之前的可见性协议，不能证明实现正确。

当 handler 是 CPU/GDRCopy 或 CPU/host-memory 时，device QP 的 `bf` 字段并不直接指向 NIC UAR，而指向一个 producer-index notification location。device 侧 [`ibgda_proxy_post_send`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L1621-L1639) 以 system-scope atomic 发布新的 index；host plugin 安装的 [`nvshmemt_ibgda_progress`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/modules/transport/ibgda/ibgda.cpp#L469-L600) 读取这些 indexes，再由 CPU 更新 DBREC 和 UAR。`nvshmemt_init` 也只在 GPU handler 下令 `progress = NULL`、`no_proxy = true`。因此以下三个命题强度不同：

1. IBGDA plugin 成功初始化；
2. device dispatch 选择了 IBGDA；
3. 最终 NIC handler 是 GPU，GPU 直接写 DBREC/UAR。

前两个不能自动推出第三个。官方环境变量文档把 `NVSHMEM_IBGDA_NIC_HANDLER=auto` 定义为优先 GPU、条件不满足时回退 CPU；固定源码也确实检查 GPU mapping NIC UAR 是否成功，再在 GDRCopy 与 host-memory CPU handler 之间选择。C02 中要求同时收集 startup log、handler selection 与 operation observation，正是为了识别这个分叉。

## 七、CQE 既是完成依据，也承担环形队列回收

device QP 引用一个 collapsed CQ view，其中保存 CQ buffer、qpn、队列容量以及与 QP management variables 相连的 `prod_idx`、`cons_idx` 等指针。[`ibgda_poll_cq`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L465-L575) 不是简单“取下一条 CQE”：它先确认目标 logical index 已经提交，读取 collapsed CQE 中的 16-bit hardware WQE counter，处理 counter wraparound，再用 `atomicMax` 推进 64-bit software `cons_idx`。错误 CQE 被视为 fatal，而 debug/timeout 分支还附加诊断。

[`ibgda_quiet(qp)`](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh#L1688-L1722) 读取的是当前 `ready_head`，然后让 `ibgda_poll_cq` 等待 CQ 的 hardware WQE progress 覆盖该 logical index。由于一组请求只在末尾 WQE 请求 CQ update，并使用 collapsed CQ 记录进度，把它描述成“quiet 等待每条请求各自的 CQE”是错误的。更准确的说法是：在这个 helper 层，quiet 以目标 producer boundary 为 ticket，等待 CQ 所反映的硬件消费进度达到该边界；更高层 fence/quiet 对多个 QP、GET consistency 和目标可见性的处理仍需在 D04 分别追踪。

同一 CQ 进度还被 `ibgda_wait_for_slot_availability` 用于背压：当即将预留的 logical index 已超出 SQ 容量时，它等待足够老的 WQEs 完成。完成不只是给调用者一个“操作结束”信号，也关闭了队列槽位的生命周期。若只有 `prod_idx` 而没有可信的 `cons_idx`，生产者可以继续提交，却无法证明回绕后的槽位可安全覆盖。

## 八、把核心对象放回同一张关系表

| 对象 | 主要创建/准备者 | 稳态读写者 | 在请求中的职责 | 不能与什么混淆 |
| --- | --- | --- | --- | --- |
| PD / MR | host + verbs/DEVX | RNIC 验证 keys | 约束注册内存和访问权限 | PD 不是地址翻译表，MR 也不是消息队列 |
| lkey / rkey | host 注册后写入 device state | GPU 填 WQE，RNIC 校验 | 分别授权本地 SGE 与远端目标 | key 不是虚拟地址，也不代表数据 ready |
| RC QP | host 创建并与特定 peer 建连 | GPU 选取、填 SQ；RNIC 执行 | 可靠的 peer-specific 发送上下文 | 不等于每个 CUDA thread 独占一个 QP |
| DCI / DCT | host 创建并交换目标信息 | GPU 选 DCI、在 WQE 指定 DCT | 用较少 initiators 覆盖多个动态 targets | DCT 不是一条由 GPU提交的发送 SQ |
| WQ / WQE | host 分配 WQ；GPU 填 WQE | GPU 写，RNIC 读 | 描述 opcode、地址、key、长度和目标 | WQE 写完不等于已经 doorbell |
| DBREC | host 分配并映射给 NIC | GPU 或 CPU handler 写，RNIC 读 | 记录新的 SQ producer boundary | DBREC 不是 UAR，也不是 CQE |
| UAR / BF | host 通过 mlx5 DEVX 分配映射 | GPU 或 CPU handler MMIO write | 通知 NIC 检查新工作 | doorbell 不搬运 payload |
| CQ / CQE | host 创建 CQ；RNIC 写 CQE | GPU poll | 报告硬件 WQE progress 与错误 | CQE 数量不必等于 API/WQE 数量 |
| internal buffer | host 分配并注册 | GPU/NIC | 接收 fetch result、辅助 consistency 等 | 不是用户 symmetric object |
| `resv/ready/prod/cons` | host 清零管理状态 | 多个 GPU producers、GPU/CPU submitter | 表达预留、连续就绪、已通知、已完成边界 | 四个 index 不能合并成一个“队尾” |

这张表也揭示一个重要的实现事实：IBGDA 并不是“GPU 绕过 verbs，直接把数据发到网卡”的魔法捷径。host 仍使用 verbs/DEVX 建立受保护资源和映射；GPU 只是被授予了对特定 WQ/control memory 与 UAR 的可用视图，从而能在初始化完成后自行产生 NIC 工作。资源的创建权、请求的生产权和 payload 的搬运者属于不同执行实体。

## 九、一次 PUT 的总览时序

下面的时序只描述跨节点、IBGDA device dispatch 已选择、目标不是本 PE 的一般骨架；RC/DC、blocking/NBI、handler 和 registration chunk 会让具体分支变化。

```text
CUDA caller             shared QP state            NIC / remote memory
     |                         |                            |
     | select QP               |                            |
     | lookup lkey/raddr/rkey  |                            |
     | atomicAdd(resv_head) -->|  reserve [b, e)           |
     | wait old cons_idx if ring would wrap                |
     | write RDMA WRITE WQE(s) ---------------------------> WQ memory only
     | fence + CAS ready_head -> e                         |
     |                         |                            |
     | if post condition:      |                            |
     |   atomicMax(prod_idx) ->|                            |
     |   update DBREC          |                            |
     |   ring UAR/BF ----------+--------------------------> NIC fetches WQE
     |                         |                            | DMA payload
     |                         |                            +--> remote MR
     | poll collapsed CQ <-----+---------------------------- CQ progress
     | atomicMax(cons_idx) ----|  old WQ slots reusable     |
     |                         |                            |
```

如果选择 CPU handler，图中 “update DBREC / ring UAR” 两步移到 host progress path，GPU 在相同位置只发布 producer index notification。若是 NBI PUT，API path 可以在 CQ progress 到达之前返回；若是 blocking/cooperative RMA，该固定 helper 可在提交后调用 `ibgda_quiet(qp)`。但 NVSHMEM API 层的 local/remote completion、ordering 和 visibility 仍应以 B02 的契约为准，不能从一段内部 poll 代码直接推广到所有 API 变体。

## 十、建议的源码阅读顺序与定位入口

| 顺序 | 文件与 symbol | 先回答的问题 |
| ---: | --- | --- |
| 1 | `src/host/transport/transport.cpp`：`nvshmemi_transport_init` | plugin 如何被装载，device transport 何时被选中？ |
| 2 | `src/modules/transport/ibgda/ibgda.cpp`：`nvshmemt_init` | 哪些 HCA capability、handler 与 callbacks 构成可用 transport？ |
| 3 | 同文件：`nvshmemt_ibgda_connect_endpoints`、`ibgda_create_qp`、`ibgda_create_cq` | QP/CQ/WQ/DBREC/UAR 在何处创建和映射？ |
| 4 | `src/include/device_host_transport/nvshmem_common_ibgda.h` | host 向 device 交付了哪些指针、keys、数量和 indexes？ |
| 5 | `src/host/init/init.cu`：`nvshmemi_update_device_state` | host transport state 如何复制到 device symbol？ |
| 6 | `src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in` | 统一 device API 如何在 IBGDA、GDAKI 和 proxy 之间分流？ |
| 7 | `ibgda_device.cuh`：`ibgda_get_qp`、`ibgda_get_lkey`、`ibgda_get_raddr_rkey` | QP、地址和 key 如何选择？ |
| 8 | 同文件：`ibgda_reserve_wqe_slots`、WQE writers、`ibgda_submit_requests` | 多生产者怎样预留、填充并闭合 ready 空洞？ |
| 9 | 同文件：`ibgda_post_send` / `ibgda_proxy_post_send` | GPU 与 CPU handler 在哪里分叉？ |
| 10 | 同文件：`ibgda_poll_cq`、`ibgda_quiet` | 完成进度怎样映射回 64-bit logical queue index？ |

实际追踪时应在每一步记录五项内容：输入参数、读取的共享状态、写入的内存位置、使状态对另一执行实体可见的同步动作，以及失败或等待条件。只记录函数调用列表很容易遗漏真正的协议边界。例如，`ibgda_write_rdma_write_wqe` 解释 WQE 内容，但请求何时能被 NIC 读取取决于其后的 ready protocol、memory fence 与 doorbell；三者必须连起来读。

## 十一、常见误解、适用限制与待验证项

“IBGDA 完全没有 CPU”是不准确的。初始化、HCA discovery、PD/MR/QP/CQ creation、endpoint exchange、key distribution 和 teardown 都由 host 完成；在 CPU handler fallback 下，稳态 doorbell 也由 CPU progress path 完成。更准确的结论是：在支持且选中 GPU handler 的稳态路径，GPU 可以生成 WQE 并直接更新 DBREC/UAR，使每个 operation 不必通过传统 CPU proxy 才开始 NIC 工作。

“每个 GPU thread 有自己的 QP”同样不成立。固定源码支持按 CTA、SM、warp 或 DCT group 映射有限 DCI，也支持多个 RC per peer；shared QP 正是 `resv_head/ready_head/prod_idx/cons_idx`、locks、scope-aware atomics 和 fences 存在的原因。QP 增多可以降低共享竞争，却会增加 host/NIC memory、连接规模以及 fence/quiet 遍历成本，不能脱离 workload 和规模给出单调优劣结论。

“doorbell 后就完成”混淆了提交与执行。doorbell 只通知 NIC 有新 WQEs；NIC 仍需读取 WQE、执行 DMA、处理网络可靠性并更新 CQ。反过来，“看到 CQ progress”也不自动等于目标应用已经观察或消费数据。CQ 回答 transport/hardware progress，signal/wait、quiet、fence、barrier 和应用 ack 的语义边界仍由 B02、B04 和 D04 分别说明。

该实现依赖 mlx5 DEVX、GPU/NIC control-buffer mapping、合适的 GPU 与 NIC/driver/CUDA/OFED 组合。官方 3.7.2 release notes 列出的测试矩阵和限制并不能替代目标机器验证；例如文档明确列出 IBGDA 在 CX-4 Ethernet/RoCE 场景下不工作。本文也没有核验不同 GPU architecture 的生成指令、PCIe ordering、BAR/UAR mapping 性能，或 CPU/GDRCopy 与 CPU host-memory handler 的实际延迟。

## 诊断题与参考推理

### 问题一：GPU 已把 RDMA WRITE WQE 的所有字段写完，为什么还不能立即把 `prod_idx` 推到它后面？

因为共享 QP 中可能存在更早预留但尚未填完的槽。NIC 按 SQ 顺序读取，若 producer boundary 越过这个 publication hole，就可能消费不完整 WQE。该版本用 `ready_head` 的有序 CAS 证明前缀连续就绪，再决定是否推进 `prod_idx` 和 doorbell。

### 问题二：为什么一次逻辑 PUT 可能查多次 lkey/rkey 并产生多个 WQE？

本地与远端注册 key 按 memory chunk 和 NIC/PE 组织，一个请求可能跨越 key coverage boundary。实现每次取剩余长度、本地 chunk 剩余长度和远端 chunk 剩余长度的最小值，完成一段后继续查下一段；另外 DCI address-vector 布局也可能让一个 command 占多个 WQEBBs。

### 问题三：日志显示 “Successfully initialized the transport: IBGDA”，能否据此断言 GPU 直接敲 NIC doorbell？

不能。它证明 plugin 初始化并被用于 device-side APIs，但 `auto` handler 还可能因 GPU 无法映射 NIC UAR 而退回 CPU/GDRCopy 或 CPU/host-memory。还需检查 “NIC handler will be ...” 的实际选择，并结合 C02 的运行证据确认。

### 问题四：为什么 `quiet` 不应描述成“等待每个 PUT 对应的一条 CQE”？

该实现只在一组末尾 WQE 请求 CQ update，并使用 collapsed CQ 的 WQE counter 表达累计进度。`ibgda_quiet(qp)` 以当前 ready boundary 为目标，等待 CQ hardware progress 覆盖它；API 与 WQE、WQE 与 CQE 都不是一一对应关系。

### 问题五：如果只看到 `cons_idx` 增长，能否证明远端 kernel 已经消费 payload？

不能。`cons_idx` 表示本地观察到 NIC 对发送 WQE 的完成进度，并使本地 WQ slots 可回收。远端应用是否已经观察、解析并消费 payload，需要 signal/wait、应用状态机或消费 ack 等更高层协议证明。

## 掌握标准与结论

完成本课后，应能从固定 commit 中定位 transport 装载、endpoint 建立、device state 复制和 device dispatch 四个入口；能画出 `<symmetric address, PE>` 到 `<remote NIC address, rkey>` 的实现转换；能解释 RC 与 DCI/DCT 的发起/目标关系；能用 `resv_head -> ready_head -> prod_idx -> cons_idx` 描述共享 SQ 的生命周期；并能分别指出 WQE memory、DBREC、UAR/BF 和 CQ buffer 的生产者与消费者。

IBGDA 的核心不是简单地“让 GPU 调用网络 API”，而是把一套原本由 CPU verbs submission path 管理的生产协议安全地暴露给 GPU：host 先创建并授权资源，device 生产有序的 WQEs，memory fences 建立可见性，doorbell 把连续 ready prefix 交给 NIC，CQ progress 再关闭槽位生命周期。D01 将在同一 commit 上选择一条具体 PUT，把本课的总图压缩成逐函数、逐字段、逐事件的请求轨迹。

## 来源

- NVIDIA NVSHMEM 3.7.2 release tag 与固定源码：[`v3.7.2-0`](https://github.com/NVIDIA/nvshmem/tree/v3.7.2-0)，commit [`3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4`](https://github.com/NVIDIA/nvshmem/commit/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4)，静态核对日期 2026-09-13。
- [NVIDIA NVSHMEM 3.7.2 Release Notes](https://docs.nvidia.com/nvshmem/release-notes-install-guide/release-notes/release-3720.html)：版本兼容性、限制和 IBGDA RC QP ABI known issue。
- [NVSHMEM Environment Variables](https://docs.nvidia.com/nvshmem/api/latest/gen/env.html)：IBGDA enable、NIC handler、RC/DCI/DCT、mapping、batch 与 fetch-slot 选项；该页为滚动文档，使用时仍须与固定 tag 的 `src/modules/transport/common/env_defs.h` 对照。
- [NVSHMEM Performance：Tuning the queue-pair Type and Configuration for IBGDA](https://docs.nvidia.com/nvshmem/release-notes-install-guide/best-practice-guide/performance.html#tuning-the-queue-pair-type-and-configuration-for-ibgda)：RC 与动态连接的资源/性能权衡；性能建议需在目标环境复测。
- 本知识库前置笔记：[A01：从零开始理解 RDMA 通信模型](a01-rdma-communication-model.md)、[A02：RDMA 资源、内存注册与建连生命周期](a02-rdma-resources-and-memory-registration.md)、[A03：RDMA 请求、完成与缓冲区协议](a03-rdma-requests-and-buffer-protocol.md)、[B01：NVSHMEM 对称对象与远端寻址](b01-symmetric-objects-and-remote-addressing.md)、[B02：完成、排序与可见性](b02-completeness-ordering-and-visibility.md)、[B04：数据发布、缓冲区所有权与有界流水协议](b04-data-publishing-and-buffer-ownership.md)、[C01：从 host RDMA 到 GPUDirect RDMA 与 IBGDA](c01-host-rdma-to-gpudirect-and-ibgda.md)与 [C02：Transport 与 GPU–NIC 拓扑核验](c02-transport-verification-and-topology.md)。
