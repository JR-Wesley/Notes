# 从 Halo Exchange 到 MoE Dispatch：用 NVSHMEM 构造规则与稀疏数据交换协议

Halo/stencil 与 Mixture of Experts（MoE）都需要在 GPU 之间移动中间数据，但二者的通信结构几乎相反。规则网格的邻居、边界位置和每轮数据量通常预先已知，可以建立稳定的 point-to-point 通道；MoE 的目标专家和各目标数据量由本轮 token routing 决定，需要先形成 metadata，再分配空间并执行稀疏重分发。前者最难的是迭代代际与边界 buffer 复用，后者最难的是动态容量、负载不均和输出逆置换。

本篇不提供一个声称适用于所有框架的“最优实现”，而是建立从依赖图到通信原语的推导方法。协议基础来自 [B04：数据发布与缓冲区所有权](b04-data-publishing-and-buffer-ownership.md)、[B05：远程原子操作与分布式协调](b05-atomic-operations-and-distributed-coordination.md)、[B06：Team 与集合通信](b06-teams-and-collective-communication.md)以及 [E01：性能测量与调优](e01-performance-measurement-and-tuning.md)。模型侧 Expert Parallelism 背景可参阅现有 [EP 笔记](../../algorithm-and-model/MoE/EP.md)，但本文不以该笔记作为实现事实的主要证据。

## 1. 选择原语之前先画数据依赖图

通信阶段至少要标出五件事：谁产生数据、谁决定目标、谁拥有目标写槽、哪个事件表示数据可读、哪个事件允许槽位重用。RMA PUT 只解决生产者向消费者地址写数据，GET 只解决消费者从生产者地址拉数据，collective 只定义参与者共同完成的数据变换；它们都不会自动回答 buffer generation、消费确认和容量溢出。

```text
EMPTY(g)
  -- producer writes payload --> PUBLISHED(g)
  -- consumer reads payload  --> CONSUMED(g)
  -- acknowledgement         --> EMPTY(g+1)
```

`g` 是 generation。ready signal 建立 payload 从生产者到消费者的发布边，acknowledgement 建立消费者到下一次生产者写入的回收边。只有布尔 ready 而没有 generation 或 ack 时，快速生产者可能覆盖慢消费者尚未读完的数据；`quiet` 只能闭合相应通信完成，不能证明目标应用线程已经消费。

## 2. Halo/stencil：固定邻居上的代际流水

### 2.1 从域分解得到通信对象

以二维五点 stencil 为例，全局网格按行或二维块分给多个 PE。每个 PE 拥有 interior cells，并保存相邻 PE 所需的 top/bottom/left/right boundary；同时为邻居数据预留 ghost/halo cells。第 \(g\) 轮的 interior 更新只依赖本地旧数据，靠近分区边界的更新还依赖邻居发布的第 \(g\) 代 halo。

若按行分解，PE \(p\) 的上边界发送给 \(p-1\)，下边界发送给 \(p+1\)。物理域首尾没有对应邻居，应使用边界条件而不是对不存在的 PE 通信；周期边界才把首尾 PE 连成环。邻居集合应在启动阶段确定，否则一个看似方便的模运算可能改变数学问题。

对称堆中的接收布局可以是 `recv_from_up[K][width]` 和 `recv_from_down[K][width]`。索引 `slot=g mod K` 让不同代使用不同槽；双缓冲允许消费第 \(g\) 代时准备第 \(g+1\) 代，但只把可容忍距离扩为两代。若生产者可能领先更多，仍需 ack 或有界窗口证明不会覆盖。

### 2.2 Push、pull 与 put-with-signal

PUSH 中，边界生产者知道固定目标 PE 和目标槽，可将 boundary PUT 到邻居的 symmetric receive buffer，再更新 ready generation。put-with-signal 直接表达 `payload → ready`，目标 PE 等待正确 generation 后读取 halo。目标参数仍是本 PE 上相应对称对象的地址加目标 PE，不是对方进程泄露的裸指针。

PULL 中，消费者在需要数据时从邻居发布 buffer GET。它减少远端写槽协调，却要求生产者先证明源 generation 已准备好，并在全部 GET 结束前不复用源槽；消费者还必须完成 GET 后才能使用结果。PULL 没有消除同步，只是把“目标槽可写”换成“源槽可读且不可覆盖”。选择依据是 ownership 与依赖方向，性能需在实际 transport 上测量。

### 2.3 一轮可证明的时序

一轮 halo exchange 可以按以下因果关系组织：

1. 确认发送源属于稳定的第 \(g\) 代，并通过邻居 ack 确认 `slot=g mod K` 可覆盖。
2. 计算或 pack 边界。连续行通常可直接发送，列边界可能需要 pack 成连续 buffer。
3. 对每个有效邻居执行 put-with-signal，让 signal 携带 generation，而不是恒置 1。
4. 同时计算不依赖新 halo 的 interior；只有这一部分天然允许与通信重叠。
5. 等待全部必要邻居的 generation，再计算 boundary region。
6. 本 PE 不再读取 halo slot 后向生产者发布 ack，允许未来复用。

第 4 步能否缩短迭代取决于 interior 是否足够长，以及通信是否与计算竞争 SM/HBM。时间线相交但 iteration time 不降，不算有效 overlap。strong scaling 时 local subdomain 变薄，halo/interior 比上升且可隐藏计算减少，最终会出现扩展拐点。

### 2.4 邻居同步不等于全局 barrier

Halo 通常只依赖少量几何邻居。每轮全局 barrier 会把无关 PE 的抖动传播到所有参与者，而每邻居 generation signal 可以表达精确依赖；代价是必须为各方向定义 signal 槽、generation 编码和回收规则。全局 barrier 仍适合初始化、全局收敛判定或真正的阶段切换，原则是同步参与集合与真实依赖一致，而不是机械地拒绝 collective。

## 3. MoE：数据依赖的稀疏重分发

### 3.1 Dispatch 为什么不是固定消息矩阵

Expert Parallelism 中，每个源 PE 持有 token hidden states，router 选择 top-k experts，expert placement 再映射到目标 PE。若源 PE \(s\) 有 \(N_s\) 个 token、hidden width 为 \(H\)、元素宽度为 \(b\)，则忽略 metadata 的逻辑 dispatch payload 约为

\[
B_s=N_s\times k\times H\times b.
\]

发往目标 \(d\) 的 token 数 \(c_{s,d}\) 每轮变化，且 \(\sum_d c_{s,d}=N_s k\)。因此 MoE 必须处理 counts、offsets、capacity 和 token identity：先对 `(target PE, local expert)` 计数并计算 send offsets，再 pack token，让目标获得各源 count 和互不重叠的接收位置，传输 hidden state、routing weight 与恢复原序所需的 metadata，最后发布完成。expert 输出还要按 source/token identity 返回并 combine。

### 3.2 Collective all-to-all 的适用条件

NVSHMEM 标准 typed `alltoall` 为每个目标传送相同 `nelems` 的块，适合规则等长布局；动态 MoE 的 \(c_{s,d}\) 通常不相等，不能未经转换就假定 fixed-count all-to-all 天然匹配。按 capacity padding 到等长块可简化 collective 和 offset，但会增加预留空间与无效区域；容量不足仍需定义 drop、reroute 或 overflow，而这些选择都会影响模型语义或尾延迟。

另一方案是先交换 counts，再以 RMA 传真实长度。它减少 padding，却增加 metadata phase、远端空间规划和细粒度 completion。NVSHMEM collective、NCCL collective、框架 fused MoE communication 和自定义 RMA 都可能合理；应先写出变长语义、参与 group、数据布局与完成要求，再比较库能力，而不是先验认定某个库必然更快。

### 3.3 RMA 的 reservation—publish 协议

RMA 方案可在目标 PE 上为每个 local expert 建立 bounded receive arena。源 PE 对目标 counter 执行 atomic fetch-add，预留 `count` 个连续槽，返回值作为 base offset；随后 PUT token 与 metadata，最后发布相应 ready。fetch-add 只保证 reservation 区间唯一，不保证 payload 已写完，更不保证所有较小 offset 已按序发布。

若各 source 使用固定 segment，可以避免原子热点，但需要按最大 capacity 预留；共享 compact arena 节省静态 padding，却在热门 expert 上产生 atomic contention 和 publication hole。后取得大 offset 的源可能先完成，而小 offset 源仍在传输；目标不能看到总 counter 就消费整个前缀，必须使用 per-reservation ready、连续发布边界，或等待所有预期 source 完成。

一种易证明但同步较强的两阶段设计是：

```text
count phase:
  route -> counts -> exchange counts -> compute disjoint receive offsets
payload phase:
  pack -> PUT nonempty segments -> publish done[src,dst]
  target waits expected sources -> expert compute
return phase:
  pack outputs -> return PUT/signal -> combine by token identity
```

counts phase 已计算互不重叠的 offset，payload phase 无需远端 allocator；代价是多一个 group-level metadata phase。按 chunk/expert 流水可以提前计算，但每个 chunk 都要独立定义 ownership 和 publication，不能只是删除 barrier。

### 3.4 Combine 必须保存原始身份

Dispatch 按 expert/target 重排 token 后，expert 输出顺序与源 batch 不同；top-k 时一个 token 还有多个结果。返回 metadata 至少要确定 source PE、source token index、expert branch 和 routing weight。combine 可先写入不重叠分支槽再归并，也可原子累加，但后者要额外证明数据类型支持、累加次序边界和多写者完成判定。

如果只返回 hidden vectors 而丢失 permutation，通信可能完全成功但模型结果错位。性能测试必须同时校验 token identity、重复/drop 和数值；收发总字节相等不足以证明正确。

## 4. 负载不均、容量与尾部完成

定义目标 PE \(d\) 接收 token 数 \(L_d=\sum_s c_{s,d}\)，简单的设备级 imbalance ratio 为

\[
I=\frac{\max_d L_d}{\frac{1}{P}\sum_d L_d}.
\]

它没有表达不同 expert 计算成本、链路位置和本地/远端比例，但比平均 token 数更能暴露 straggler。热门 expert 会同时增加 inbound bytes、reservation contention、expert compute 和 return traffic，step time 通常受最慢目标而不是平均目标控制。

capacity factor 为 expert 提供高于平均负载的空间，可降低 overflow 概率，却增加 symmetric arena 与 padding。容量不足时必须显式选择：drop 改变模型语义，reroute 改变 expert 选择，overflow queue 增加协议和尾延迟；通信层不能静默截断或越界。复制热门 expert、分层 routing 或调整 placement 也会引入参数/梯度同步成本，本篇不把任何一种策略写成普适最优。

## 5. 两类交换的共同点与差异

| 维度 | Halo/stencil | MoE dispatch/combine |
| --- | --- | --- |
| 通信图 | 固定、小度数邻居 | routing 决定，可能接近 all-to-all |
| 消息大小 | 边界形状预知 | source→target count 动态变化 |
| 目标布局 | 固定方向与 generation 槽 | counts/offsets/capacity 决定 segment |
| 核心状态 | ready generation 与 reuse ack | count、publication、identity 与 overflow |
| 主要不均衡 | 子域形状、边界比例、慢邻居 | 热门 expert、placement、原子热点与 straggler |
| 自然 overlap | interior compute 与 halo | dispatch、分块 expert compute、return/combine |

两者都必须闭合 payload publication 和 buffer reclamation，都不能把 ready 当作消费完成，也不能把 NBI 返回当作远端数据可用。区别是 halo 能在启动时预计算大部分地址和容量，而 MoE offset 与工作量是每轮输入的一部分；把固定双缓冲直接复制到 MoE 会遗漏 count exchange、overflow 与 permutation。

## 6. 保留应用结构的性能测量

Halo 应记录 boundary bytes、邻居数、local domain shape、pack/unpack、通信完成、interior/boundary compute 和 iteration time。一个大连续 PUT 只能测链路，不能代表列 pack、多邻居小消息和局部同步。strong-scaling 还应保持全局问题规模固定，并报告每 PE 的 halo/interior 比。

MoE 应记录每个 expert/PE 的 token histogram、非空 source→target pair、payload/metadata bytes、padding/overflow、pack、dispatch、expert compute、return、combine 和 critical path。aggregate bandwidth 很高也可能 step 很慢，因为最热 PE 决定尾部。两类 overlap 都应按 E01 比较同工作量的串行、并发和控制基线；NBI 或 profiler 区间重叠本身不是收益证据。

## 7. 常见错误与诊断

Halo 偶发读到旧边界时，应检查 signal generation、目标 ack、slot 索引以及 PUT→signal 发布顺序。全局 barrier 能掩盖问题，并不证明原协议正确。MoE 出现 token 丢失或错位时，应核对 `sum(send_counts)==sum(recv_counts)`、reservation 不重叠且不越 capacity、每个 published segment 的 payload/metadata 完整、return identity 唯一，以及 top-k 分支数与 combine 次数一致。

遇到 hang，应列出每个 PE 等待的 signal/count 和唯一能推进它的生产者。halo 常见环路是双方等对方 ready 后才发送；MoE 常见环路是目标等待所有 source signal，而零长度 source 没有定义是否发送 completion。协议必须明确 empty message：显式发布 count=0，或由 counts phase 精确给出非空 source 集合，不能由两端分别猜测。

## 8. 掌握标准、结论与边界

完成本课后，应能从域分解画出 halo buffer 的 producer、consumer、generation、ready 和 ack；解释 PUSH/PULL 只是改变 ownership 方向而非消除同步；为 MoE 写出 count、offset、payload、publication 和 combine identity；判断 fixed-count collective、padding 与变长 RMA 的前提；并用 token histogram、capacity、尾部负载和 step time 评价方案。

统一方法是“先证明协议，再选择原语”：对称寻址决定远端对象如何命名，RMA/collective 决定数据如何移动，signal 决定何时可消费，ack/generation 决定何时可复用。NVSHMEM 能把这些机制带入 GPU kernel 与 stream，但不会替应用定义依赖、容量政策、身份映射和失败处理。

本文没有编译或运行 Jacobi、stencil、MoE 或 NVSHMEM 示例，没有测量性能，也没有核对具体训练框架的 fused dispatch。实际选用 NVSHMEM、NCCL 或框架专用组件前，仍需按目标版本验证 API、transport、collective 能力和数值正确性。

## 参考资料

- [NVIDIA NVSHMEM Developer Page：Halo Exchange](https://developer.nvidia.com/nvshmem)
- [NVIDIA multi-gpu-programming-models：NVSHMEM Jacobi](https://github.com/NVIDIA/multi-gpu-programming-models/tree/master/nvshmem)
- [NVIDIA HOTI 2025 GPU Communications Tutorial](https://github.com/NVIDIA/hoti-2025-gpu-comms-tutorial)
- [NVIDIA NVSHMEM RMA API](https://docs.nvidia.com/nvshmem/api/latest/gen/api/rma.html)
- [NVIDIA NVSHMEM Signaling API](https://docs.nvidia.com/nvshmem/api/latest/gen/api/signal.html)
- [NVIDIA NVSHMEM Collective Communication API](https://docs.nvidia.com/nvshmem/api/latest/gen/api/collectives.html)
