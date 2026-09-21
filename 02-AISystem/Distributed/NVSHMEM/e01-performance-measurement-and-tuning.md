# NVSHMEM 性能测量与调优：从微基准口径到端到端重叠证据

NVSHMEM 性能分析的难点并不是得到一个 GB/s 数字，而是确认这个数字代表哪条执行路径、包含哪些同步成本、由多少线程和多少 PE 共同产生，以及它是否解释了真实应用的端到端时间。小消息延迟、持续带宽、消息率、同步开销和计算通信重叠描述的是不同问题；把其中一个指标当成整体性能，往往会得到互相矛盾但各自“看似正确”的结论。

本篇建立可复现的测量与调优方法，前置内容是 [B02：完成、排序与可见性](b02-completeness-ordering-and-visibility.md)、[B03：CUDA 执行域、协作通信与前进性](b03-cuda-execution-and-cooperation.md)、[C02：Transport 与 GPU–NIC 拓扑核验](c02-transport-verification-and-topology.md)以及 D00—D04 的 IBGDA 实现分析。本文不提供本机实测数据：示例命令和实验表格都是待执行方案，不能当作任何 GPU、NIC 或 NVSHMEM 版本的性能结论。

## 1. 性能问题必须先写成可证伪的问题

“IBGDA 是否更快”不是合格问题，因为“更快”没有指定工作负载和评价指标。更可执行的问题是：在两节点、每节点一个 PE、固定 GPU–NIC affinity 和相同消息尺寸下，device-initiated 4-byte PUT 的完成延迟是否低于 proxy transport；或者，在 128 个 CTA 并发发出 1 KiB PUT 时，IBGDA 是否以更小 working set 达到更高 aggregate message rate。前者强调单次关键路径，后者强调并发注入能力，答案完全可能不同。

一次性能实验至少需要固定六类变量：硬件与 topology、软件栈、通信语义、消息形态、并发结构和计时边界。若只写“消息大小 1 MiB”，却没有说明单向还是双向、一个 PE 还是多个 PE 同时注入、thread/warp/block API、blocking 还是 NBI+quiet、目标是本地 NVLink 还是跨节点 InfiniBand，那么结果无法复现，也不能与另一个 1 MiB 数字直接比较。

## 2. 延迟、带宽和消息率各自回答什么

### 2.1 延迟不是“字节数除以带宽”

单操作延迟应描述从一个明确起点到一个明确终点的时间。对 blocking PUT，计时终点仍需核对该 API 对 source reuse 和 remote completion 的保证；对 NBI PUT，函数返回仅代表操作已发起，若要测完整通信必须把相应 `quiet` 或协议 acknowledgement 纳入终点。对 signal ping-pong，测得的是发送、远端 signal 更新、对端观察、回复以及本端观察的组合 round trip，除以二只能作为对称路径假设下的单程估计，不能自动等同于单个 PUT 的 latency。

可把观测延迟粗略分解为：

$$
T_{obs}=T_{launch}+T_{issue}+T_{queue}+T_{network}+T_{remote}+T_{completion}+T_{sync}.
$$

不同 benchmark 会省略或放大其中某些项。kernel 内循环可以摊薄 `T_launch`，但也可能测到稳态流水而不是冷启动；每轮 barrier 能建立严格迭代边界，却把 collective synchronization 混入结果；一次发很多 NBI 再 quiet 测到的是批处理吞吐，不能命名为单操作 latency。

### 2.2 带宽必须说明有效字节和聚合范围

若在测量窗口 $T$ 内完成 $N$ 次、每次 payload 为 $S$ 字节的单向传输，应用有效带宽可写为

$$
BW_{payload}=\frac{N\times S}{T}.
$$

双向同时传输时，有的工具报告两个方向 payload 总和，有的报告单方向值；多 PE 测试也可能输出 per-PE、per-pair 或 aggregate bandwidth。报告中必须写明分子如何计算，不能仅保留单位。协议 header、WQE、PCIe transaction 和链路编码不在 payload 字节中，因此应用 GB/s 与物理 wire rate 不是同一个量。

带宽曲线通常随消息增大经历 latency-dominated、并发/流水增长和饱和区域，但拐点由 transport、线程协作、registration chunk、QP 数量和 topology 共同决定。大消息峰值不能预测 8-byte AMO 性能，小消息 latency 也不能预测大量 GPU 线程同时发请求时的 message rate。

### 2.3 消息率揭示细粒度通信能力

对于固定小消息，$R=N/T$ 次每秒比 GB/s 更容易反映 WQE 生成、doorbell、QP contention 和 NIC packet processing。IBGDA 的价值之一是让许多 GPU 线程直接并行提交请求，从而提高细粒度注入能力；但单个 GPU thread 填写 WQE 未必比 CPU 快，更多 QP 也会增加资源占用以及 fence/quiet 遍历成本。因此“单消息 latency 较高、并发 message rate 或带宽较高”并不矛盾，它们测量的是不同工作点。

## 3. 正确设计微基准的计时边界

微基准首先要区分 issue time、completion time 和 protocol time。issue benchmark 只包围 API 发起区间，用来研究提交吞吐；completion benchmark 在窗口末尾加入与被测操作匹配的 quiet、wait 或 acknowledgement；protocol benchmark 则测生产者发布、消费者观察并回复的完整闭环。三种结果都可以有价值，但名称和解释必须与计时边界一致。

GPU 计时可使用 CUDA event，但 event 只对其所在 stream 的工作建立时间关系；kernel 内部 device API、多个 stream、host proxy 和远端 PE 的活动是否被窗口覆盖，需要结合程序依赖检查。CPU wall-clock 可以覆盖多进程阶段，却会混入 launcher scheduling、host synchronization 和阻塞等待。Nsight Systems/NVTX 适合验证时间线和空洞，不应只凭 profiler 上两个彩色区间重叠就计算应用收益。

一次可靠 sweep 应包含 warmup 与 measured iterations。warmup 用于越过首次 kernel/module 初始化、connection/cache 冷态和频率爬升，但不能把持续存在的抖动“预热掉”；迭代数应随消息大小调整，使计时窗口足够长，同时避免大消息测试产生不合理总流量。除了均值，至少保留中位数、尾部百分位或跨重复 run 的离散程度。只报告最佳一次会隐藏 contention、路由和系统噪声。

固定源码 `v3.7.2-0` 的 `perftest` 提供 `--min_size`、`--max_size`、`--iters`、`--warmup_iters`、CTA/thread 数、bidirectional 和 message-rate 等参数，并包含 device point-to-point、atomic、signal、collective 与 host/on-stream 测试。使用安装包中的同名二进制时仍应记录其实际版本和 `--help` 输出，因为当前文档、安装包与本文固定源码可能已经不同。

下面只是待执行命令骨架，路径、launcher 和参数必须按实际安装修改：

```bash
NVSHMEM_DEBUG=INFO \
mpirun -np 2 "$NVSHMEM_HOME/bin/perftest/device/pt-to-pt/shmem_put_bw" \
  --min_size 8 --max_size 1M --warmup_iters 20 --iters 1000
```

它不能单独证明 IBGDA 被使用，也不能自动给出公平对照。执行前应按 C02 的证据链记录 transport、HCA、GPU–NIC 距离、PE mapping 和 fallback；执行后还要保存完整启动命令、原始 stdout/stderr、退出码和结果文件，而不是只抄峰值行。本文未运行该命令。

## 4. 建立从硬件上限到应用时间的分层基线

性能定位宜使用四层基线。第一层是硬件路径基线，例如 GPU memory bandwidth、GPU peer bandwidth 以及网络 verbs benchmark，用于判断链路是否健康；它们不包含 NVSHMEM device API 和对称堆协议。第二层是 NVSHMEM 单操作微基准，分别覆盖 PUT/GET、AMO、signal、quiet/fence 和 collective。第三层是通信 pattern benchmark，例如多邻居交换、热点 atomic 或 all-to-all。第四层才是包含真实计算、buffer ownership 和迭代同步的应用端到端时间。

这四层不能互相替代。如果 `ib_write_bw` 正常而 NVSHMEM PUT 异常，问题更可能位于 NVSHMEM transport、GPU submission、mapping 或同步方式；如果 NVSHMEM 微基准正常而应用慢，应检查消息碎片化、串行依赖、过度 quiet、load imbalance 和计算占用；如果底层链路本身未达到合理范围，先调 API batching 往往只是在掩盖 topology 或部署问题。

公平对照还要求一次只改变一个主变量。例如比较 proxy 与 IBGDA 时，保持二进制版本、消息序列、GPU/NIC binding、节点、时钟策略和后台负载一致；比较 RC 与 DCI 时，记录 QP 数量、PE 数和目标切换模式，因为 RC 的 per-peer 资源与 DCI 的动态连接开销不是同一条件。若一次同时改变 transport、线程块数和 batching，就无法归因提升来自哪里。

## 5. Overlap 必须以端到端缩短为证据

计算 kernel 与通信 kernel 在 profiler 上时间区间相交，只说明它们被并发调度，不说明通信被有效隐藏。设独立测得计算时间为 $T_c$，通信完成时间为 $T_m$，组合执行时间为 $T_{both}$。一个便于解释的隐藏量是

$$
T_{hidden}=T_c+T_m-T_{both},
$$

并可用

$$
\eta_{overlap}=\frac{T_{hidden}}{\min(T_c,T_m)}
$$

描述较短阶段被隐藏的比例。该比值只有在三次测量处理同一工作量、相同 completion 语义和相近稳态条件时才有意义；资源竞争可能使 $T_{both}$ 大于 $T_c+T_m$，测得负收益也应保留，而不是截断为零。

有效 overlap 需要满足依赖独立与资源可并行两个条件。数据尚未产生时不能提前发送，接收数据尚未完成时不能提前消费；即使依赖允许，计算和通信仍可能竞争 SM issue slots、HBM bandwidth、L2、copy/transport resources 或 NIC injection。IBGDA 把 WQE 生成放到 GPU 上，减少 CPU proxy 依赖的同时，也可能让通信提交与计算争夺 GPU 执行资源。因此调大通信 CTA 数并不单调改善 overlap。

端到端实验应至少比较三个版本：串行 `compute→communication`，合法的并发/流水版本，以及去除实际数据移动但保留控制结构的开销基线。计时必须覆盖最后的 completion 与消费依赖；若 NBI 调用很快返回，而 quiet 被移到计时窗口外，就只是移动了等待位置。对于多迭代流水，还要分别报告启动、稳态和 drain，避免短运行被 pipeline fill/drain 主导。

## 6. 从曲线形状定位瓶颈

性能曲线比单点更有诊断价值。小消息 latency 高且随并发明显下降，可能说明提交/固定开销可被流水隐藏；带宽随 CTA 增长后平台化，说明已接近某个共享资源上限；继续增加并发反而下降，可能来自 QP contention、SQ/CQ 压力、SM 竞争或 memory traffic。这里只能形成候选假设，必须通过控制变量和 profiler/transport 证据排除其他原因。

常见症状与下一步证据可以这样对应：

| 现象 | 可能机制 | 应补的对照证据 |
| --- | --- | --- |
| 同节点快、跨节点异常慢 | transport fallback、NIC affinity、GDR 路径或链路问题 | 启动日志、HCA mapping、topology、verbs 基线 |
| 小消息单次慢但高并发消息率好 | WQE 生成固定成本被并行提交摊薄 | latency 与 message-rate 两套测试，扫 CTA/warp |
| 大消息早早平台化 | NIC、PCIe/NVLink、HBM 或单线程复制达到瓶颈 | 线程协作 API、底层链路和 memory bandwidth 对照 |
| NBI 看似极快但应用不变 | 只测 issue，等待被移到 quiet/barrier | 把 completion 纳入窗口，检查时间线 critical path |
| QP 增多后吞吐升、quiet 变慢 | 注入并行与完成遍历/资源开销的权衡 | 分开测 steady issue rate 与 fence/quiet latency |
| run 间波动大 | affinity、频率、共享资源、路由或热状态不稳定 | 多次独立 run、环境遥测和尾延迟 |

## 7. 调优顺序：先保证路径和语义，再改并发参数

第一步应验证正确 transport 与 topology，而不是直接修改几十个环境变量。确认 PE 与 GPU 一一映射、GPU–NIC 亲和性、P2P/IBGDA 是否实际启用，并保存 `NVSHMEM_INFO` 或适当 debug 输出。功能开关如禁用 P2P、NVLS 或 GDRCopy 更适合做诊断对照，不能因为某次对照更快就不加分析地作为生产默认。

第二步匹配 API 粒度与线程协作。当前官方指南指出，16-byte aligned buffer 有利于内部采用更宽写入；大块 P2P 数据可能从 warp/block API 的线程协作中获益，而小消息会受到同步开销影响。选择 `thread`、`warp` 还是 `block` 版本必须按消息大小、transport 和调用控制流扫描，不能把“更多线程”视为普适优化。

第三步才调整 batching、QP 类型与数量。批处理能够摊薄 doorbell 和 completion 成本，却增加排队时间并可能推迟可见性；更多 DCI/RC 可以提高并发或降低目标切换影响，却消耗 memory/NIC resources，并扩大 fence/quiet 需要处理的队列集合。调参时每次只改变一项，绘制 latency、message rate、bandwidth、quiet/fence latency 和端到端时间的联合结果，否则容易用一个指标的改善交换另一个关键指标的退化。

最后检查应用协议。频繁 barrier、每个小 PUT 后 quiet、全局同步代替邻居 signal、单槽位导致生产者等待消费者，都可能让底层 transport 优化失去意义。反过来，删除同步以获得漂亮数字但破坏 completion、visibility 或 ownership，不属于性能优化。正确性测试应与性能测试使用同一消息形态和并发范围，并在结果中明确数据校验是否通过。

## 8. 可复现实验记录模板

每个数据点应能从记录中重新构造，而不是只留下图。建议为一次实验保存以下信息：

| 类别 | 必须记录的内容 |
| --- | --- |
| 硬件 | 节点、GPU/NIC 型号、GPU–NIC/NUMA topology、链路速率、是否同节点 |
| 软件 | OS、driver、CUDA、NVSHMEM 版本/commit、OFED/rdma-core、firmware、构建选项 |
| 启动 | launcher、节点/PE 数、PE–GPU–NIC binding、完整环境变量 |
| 工作负载 | API、数据类型、消息大小、单/双向、peer pattern、thread/warp/block、CTA 数 |
| 语义边界 | blocking/NBI、completion 操作、barrier/signal/ack 所在位置、校验方法 |
| 统计 | warmup、迭代数、独立 run 数、原始样本、汇总统计和异常值策略 |
| 证据 | 完整命令、stdout/stderr、退出码、NVTX/profiler 文件、结果处理脚本版本 |

结果图的横轴通常使用消息字节数或并发度，纵轴按问题选择 latency、GB/s、Mops/s 或端到端时间；误差带应来自独立重复或清楚定义的样本集合。图注需要写清 per-PE/aggregate、单向/双向和计时终点。理论峰值线只有在推导口径与路径相符时才添加，不能把 GPU HBM、PCIe、NVLink 与 InfiniBand 的不同峰值混成一个“硬件上限”。

## 9. 诊断题与掌握标准

如果某个 NBI PUT benchmark 只计 API 调用、不计 quiet，却报告 2 ns “网络延迟”，应指出它测到的是 issue cost，远端完成不在窗口内。如果 IBGDA 的 ping-pong latency 高于 proxy，但 128 CTA 小消息 message rate 更高，应先承认二者可以同时成立，再检查 WQE 生成、并发注入和 completion 边界，而不是挑选其中一个结果否定另一个。如果 profiler 显示 compute 与 communication 重叠 80%，但端到端时间不降，应检查共享 SM/HBM 竞争、关键路径依赖以及最后的 quiet/drain 是否被遗漏。

完成本课后，应能为一个 NVSHMEM 性能问题写出可证伪假设；区分 issue、completion、protocol latency；明确 payload bandwidth 与 aggregate 口径；设计包含 warmup、重复、校验和控制变量的消息尺寸/并发 sweep；用底层链路、NVSHMEM 微基准、pattern 和应用四层基线定位问题；并且只在端到端时间确实缩短时声称获得有效 overlap。

## 10. 结论与边界

NVSHMEM 性能不是 transport 名称的单变量函数，而是通信语义、请求粒度、GPU 并发、QP/队列、topology 和同步协议共同形成的结果。延迟、带宽和消息率必须分别测量，overlap 必须用同工作量下的端到端差值证明。调优应从实际路径与正确性开始，再逐步扫描线程协作、batching、QP 和流水深度，并保存足以复现每个数据点的环境与原始证据。

本文没有运行 perftest、Nsight Systems、verbs benchmark 或任何 NVSHMEM workload，没有产生可用于硬件比较的数值。当前官方性能指南中的配置项可能随版本变化，实际实验必须以所安装版本的文档、`--help`、启动日志和运行配置为准。

## 参考资料

- [NVIDIA NVSHMEM Performance Best Practices](https://docs.nvidia.com/nvshmem/release-notes-install-guide/best-practice-guide/performance.html)
- [NVIDIA NVSHMEM Device API Best Practices](https://docs.nvidia.com/nvshmem/release-notes-install-guide/best-practice-guide/apis.html)
- [NVIDIA NVSHMEM API：Using NVSHMEM](https://docs.nvidia.com/nvshmem/api/latest/using.html)
- [NVIDIA NVSHMEM 官方仓库与安装验证示例](https://github.com/NVIDIA/nvshmem)
- [固定源码：NVSHMEM `v3.7.2-0` perftest README](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/perftest/README.md)
- [固定源码：perftest 通用参数实现](https://github.com/NVIDIA/nvshmem/blob/3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4/perftest/common/utils.cu)
