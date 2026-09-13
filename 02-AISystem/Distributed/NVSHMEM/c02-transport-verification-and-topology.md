# C02 NVSHMEM Transport 与 GPU–NIC 拓扑核验：从“可能支持”到“实际生效”

## 本课要解决的问题

C00 解释了 runtime 在第一个通信操作前需要建立哪些控制面条件，C01 则把 API caller、NIC work submitter 和 payload data mover 分开。本课继续回答更实际的问题：一台机器拥有 NVIDIA GPU 和 ConnectX NIC，程序也成功调用了 NVSHMEM，怎样证明同节点通信走 GPU P2P、跨节点 payload 走 GPUDirect RDMA，以及 device API 是否真正由 IBGDA GPU handler 提交，而不是 CPU proxy 或其他回退路径？

单一证据通常不足以回答这些问题。`nvidia-smi topo -m` 只描述硬件连接，环境变量只表达配置意图，初始化成功只证明某条可用路径被建立，CPU 利用率低也不能唯一指向 GPU doorbell。可靠结论需要把静态能力、软件前提、配置、runtime selection、单次操作观测和受控对照实验串成证据链，并明确每项证据能支持到哪一级结论。

本课提供核验方法而不报告本机结果。所有命令都是 Linux/GPU 集群环境中的采集模板，本任务没有执行它们，也没有访问实际 GPU、NIC、交换网络或作业调度系统。因此，正文中的 `PASS`、设备名和数值仅表示记录格式，不能当作当前环境的实验结论。

## 资料版本与适用边界

本文于 2026-09-13 依据 NVIDIA NVSHMEM 3.7.2 发布周期的滚动 API、Best Practice 与 Installation 文档，以及 CUDA 13.4 GPUDirect RDMA Guide 和当前 `nvidia-smi` 文档核对。环境变量、transport 列表、NIC handler 选项和默认值会随 NVSHMEM 版本变化；执行前必须以实际安装版本的文档和 `NVSHMEM_INFO` 输出复核。

本文只建立观测方法，不固定 NVSHMEM 源码 commit，也不把日志文本格式、内部 transport 名称、proxy thread 名称或 profiler event 数量当成稳定 ABI。若日志没有直接给出某项结论，应记录“未确认”，不能用性能差异替代实现证据。

## 一、先定义要证明的命题，而不是先收集大量输出

核验开始前应把问题写成可证伪的命题。例如，“GPU0 与 mlx5_0 的 PCIe 路径是 PIX”是拓扑命题；“mlx5_0 能对 GPU0 memory 完成 RDMA WRITE”是 GPUDirect RDMA 能力命题；“本次 PE 0 的跨节点 PUT 使用 mlx5_0”是 runtime mapping 命题；“该 PUT 的 NIC doorbell 由 GPU SM 而非 CPU proxy 触发”是 IBGDA handler 命题。四者互相关联，却不能互相替代。

证据可以按强度分为五层：

| 层级 | 证据类型 | 能回答什么 | 不能单独回答什么 |
| --- | --- | --- | --- |
| E0 | 型号、版本、拓扑、模块、port state | 理论能力和明显缺失条件 | NVSHMEM 实际选择 |
| E1 | 构建选项、链接库、环境变量 | 用户意图和二进制可能包含的能力 | 初始化是否接受、是否回退 |
| E2 | NVSHMEM startup/debug 日志 | runtime 发现和选定的 transport、NIC、handler | 每条 operation 的动态行为 |
| E3 | GPU/NIC/CPU profiler、counters、trace | 一次运行中哪些执行实体活动 | 单凭相关性证明唯一因果 |
| E4 | 只改变一个因素的 A/B 实验 | 某 feature 对正确性或性能的因果影响 | 跨机器、跨版本普遍规律 |

强结论通常需要相邻多层一致。例如，证明“IBGDA GPU handler 实际生效”至少需要构建包含 IBGDA、运行时启用 IBGDA、日志确认 handler 不是 `auto` 回退后的 CPU，再辅以 proxy activity、CPU profile 或 feature toggle 对照。只设置 `NVSHMEM_IB_ENABLE_IBGDA=1` 仍停留在 E1。

## 二、建立不可缺省的环境清单

第一轮采集应只读且不改变系统状态，覆盖作业、GPU、CUDA、NVSHMEM、kernel、RDMA stack、NIC firmware 和网络模式。版本记录应来自实际运行节点，而不是登录节点；多节点作业还要检查节点间是否一致。容器环境需要同时记录容器内可见版本与宿主 driver/module，因为 CUDA user-space library 与 kernel driver 分处两侧。

```bash
# 采集模板：未在本任务中执行。
uname -a
nvidia-smi
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version --format=csv
nvcc --version

ibv_devices
ibv_devinfo
rdma link show
lspci -nn

# 路径按实际安装调整；不要假设所有节点使用同一前缀。
ldd /path/to/application
env | grep '^NVSHMEM_' | sort
```

至少应保存 host name、job ID、world PE、node-relative PE、PID、CUDA visible device ordinal、GPU UUID/BDF、HCA/port、link layer、GID/LID、NUMA node 和 CPU affinity。GPU ordinal 容易受 `CUDA_VISIBLE_DEVICES` 改写，跨日志关联时优先使用稳定的 GPU UUID 或 PCI BDF；NIC 同样同时记录 `mlx5_*` 名称、netdev 和 PCI BDF，避免只用可能变化的枚举编号。

`ldd` 只能证明运行时装载了哪些 shared libraries，不能完整证明静态链接的 feature 或构建开关。若使用源码构建，应保留 source commit、CMake cache、安装 manifest 和构建日志；若使用 package，应记录完整包版本。看到库中存在 `IBGDA` 字符串只是弱证据，不能替代构建记录和运行日志。

## 三、把 GPU、NIC、PCIe 和 NUMA 放进同一张拓扑图

`nvidia-smi topo -m` 给出 GPU–GPU、GPU–NIC 的连接矩阵及 CPU/memory affinity。当前官方图例中，PIX 表示路径最多经过一个 PCIe switch，PXB 表示经过多个 PCIe switches 而不经过 PCIe host bridge，PHB 表示经过 PCIe host bridge，NODE 表示还穿过同一 NUMA node 内不同 host bridges，SYS 表示穿过 NUMA nodes 之间的 SMP interconnect；`NV#` 表示若干 bonded NVLinks。一般而言，GPU–NIC 的 PIX/PXB 路径比跨 PHB/NODE/SYS 更有利于 peer DMA，但这只是风险排序，不是带宽保证。

```bash
# 采集 GPU–GPU、GPU–NIC 和 PCIe 树。
nvidia-smi topo -m
nvidia-smi topo -nic
nvidia-smi topo -p2p r
nvidia-smi topo -p2p w
lspci -t
```

较新驱动提供拆分后的 `topo -nic`、`-gpu` 和 `-p2p` 子命令；旧版本可能只有 `topo -m`，因此脚本应先检查 `nvidia-smi topo -h`，不要把“不支持某个显示子命令”误判为“不支持 P2P”。CUDA 层的 `cudaDeviceCanAccessPeer` 或 CUDA Samples 中的 P2P 测试能补充功能验证，但仍需 NVSHMEM runtime 证据才能说明应用实际使用 P2P transport。

GPU–NIC 拓扑至少应转写为一张明确映射表：

| local PE | GPU ordinal | GPU UUID/BDF | node rank | 候选 HCA:port | GPU–NIC path | CPU/NUMA affinity | NVSHMEM 实际映射 |
| ---: | ---: | --- | ---: | --- | --- | --- | --- |
| 0 | 待采集 | 待采集 | 0 | 待采集 | 待采集 | 待采集 | 待日志确认 |

前七列是系统与作业事实，最后一列必须来自 NVSHMEM runtime selection 或足够直接的运行证据。不能仅因 mlx5_0 在 GPU0 的 PIX 路径上，就写成 PE0 一定使用 mlx5_0；默认策略还会在多个 local PEs、ports 和外部 NVSHMEM jobs 之间考虑平衡，用户配置也可能覆盖 proximity。

## 四、验证 RDMA fabric 本身，再验证 GPU memory

跨节点 NVSHMEM 依赖底层网络可用。InfiniBand 环境应确认 HCA port 为 ACTIVE、link layer、LID/GID、MTU、P_Key 与 subnet manager；RoCE 环境则要确认 netdev 状态、IP/GID 映射、路由、RoCE version，以及集群采用的 lossless/拥塞控制配置。`ibv_devinfo` 显示设备存在不等于两节点之间的 RDMA 流量可达，至少还要运行 host-memory RDMA baseline。

推荐先用 perftest 等工具建立 host-memory SEND/WRITE/READ 的连通性、延迟和带宽基线，再使用该 perftest 构建支持的 CUDA/DMA-BUF 选项，把数据 buffer 放到明确的 GPU 上。不同 perftest 版本的参数可能是 `--use_cuda=<ordinal>`、`--use_cuda_bus_id=<BDF>` 或额外的 DMA-BUF/Data Direct 选项，应以本机 `ib_write_bw --help` 为准，不复制其他平台命令后直接解释失败。

GPU-memory perftest 成功能够证明指定 HCA、GPU、peer-memory/DMA-BUF path 和 RDMA operation 在该测试中工作；它仍不能证明 NVSHMEM 使用相同 HCA 或相同提交机制。相反，host-memory baseline 成功而 GPU-memory 测试在 registration 阶段失败，通常把问题缩小到 GPU driver、DMA-BUF/peer-memory、IOMMU/topology 或 perftest build，而不是交换网络基本连通性。

测试还需区分方向。NIC 读取 GPU memory 对应发送侧 DMA read，NIC 写入 GPU memory 对应接收侧 DMA write；某些架构上 read/write 性能受 host bridge 和 ordering 的影响不同。只运行一个方向的 bandwidth test，不足以覆盖 PUT、GET 和 fetching AMO 的全部要求。

## 五、GPUDirect RDMA 需要软件与 PCIe 条件同时成立

CUDA GPUDirect RDMA Guide 说明，GPU 与第三方 PCIe device 的理想路径只经过 PCIe switches；经过单一 CPU/IOH 可能可用但性能降低，跨 CPU/NUMA interconnect 可能严重受限甚至不可靠。传统 GPUDirect RDMA 还要求各 PCIe devices 对物理地址具有一致视图，因此 IOMMU 需要关闭或使用 1:1/pass-through；具体 Grace、Blackwell、虚拟化或平台专用路径应遵循对应平台指南，不能机械套用通用 x86 结论。

当前 NVSHMEM GDAKI 文档列出的基础软件前提包括 Mellanox HCA/NIC、MOFED 5.0 或更新版本、满足版本要求的 NVIDIA driver，并至少具备 DMA-BUF、`nvidia_peermem` 或旧 `nv_peer_mem` 路径之一。系统检查可以记录 module 和 kernel 状态，但“module 已加载”仍只是条件证据：

```bash
# 只读检查模板；输出需结合实际发行版和 driver 模式解释。
lsmod | grep -E 'nvidia_peermem|nv_peer_mem'
modinfo nvidia-peermem
cat /proc/cmdline
find /sys/kernel/iommu_groups -maxdepth 1 -type d
```

Open GPU Kernel Modules 与较新 Linux 可使用 DMA-BUF，而某些环境使用 `nvidia-peermem`。二者不是要求同时加载；`NVSHMEM_IB_DISABLE_DMABUF=1` 会改变 IB transport 的选择，诊断时应记录其值。不要为了“修复”测试而随意加载 module、关闭 IOMMU 或修改 ACS，这些是系统级、高影响操作，必须由平台管理员依据硬件指南执行。

ACS、PCIe switch routing、BAR sizing、firmware 和虚拟机 passthrough 也可能影响 peer DMA。`lspci -t` 与 topology matrix 能发现明显的跨 root-complex 风险，却不能穷尽 BIOS/firmware 配置；最终仍应由 GPU-memory RDMA test 和 NVSHMEM run 形成闭环。

## 六、NVSHMEM 配置意图必须与运行时选择分开记录

当前 `NVSHMEM_REMOTE_TRANSPORT` 可选择 `ibrc`、`ucx`、`libfabric`、`ibdevx`、`gpunetio` 或 `none`，但某值只有在安装构建包含相应 support 且平台满足依赖时才可用。IBGDA 在 InfiniBand path 上由 `NVSHMEM_IB_ENABLE_IBGDA=1` 启用；GPUNetIO GDAKI 则要求 `NVSHMEM_REMOTE_TRANSPORT=gpunetio` 与 `NVSHMEM_GPUNETIO_ENABLE_GDAKI=1`。这些变量表达请求，不是成功证明。

多 NIC 映射有两类模式。`NVSHMEM_ENABLE_NIC_PE_MAPPING=0` 时 runtime 使用自动策略，当前文档说明会按距离和本地 PE 做选择；设为 1 后，可用 `NVSHMEM_HCA_LIST` 限定 HCA/port，或用 `NVSHMEM_HCA_PE_MAPPING` 显式声明若干 PEs 到 ports 的分配。后两个变量只有在启用 NIC–PE mapping 时才生效，若忘记开关，环境中虽能看到字符串，runtime 可能不采用预期映射。

建议为每次 run 保存实际环境，而不是只保存启动脚本默认值：

```bash
export NVSHMEM_VERSION=1
export NVSHMEM_INFO=1
export NVSHMEM_DEBUG=INFO
export NVSHMEM_DEBUG_FILE='nvshmem.%h.%p.log'

# 以下是待验证的配置意图示例，不代表已经生效。
export NVSHMEM_REMOTE_TRANSPORT=ibrc
export NVSHMEM_IB_ENABLE_IBGDA=1
export NVSHMEM_IBGDA_NIC_HANDLER=gpu
```

`NVSHMEM_INFO` 用于打印生效的环境选项，`NVSHMEM_DEBUG`/`NVSHMEM_DEBUG_FILE` 提供诊断日志。日志应逐 PE 保存，文件名加入 `%h` 和 `%p` 可避免多进程覆盖。最终报告要同时列出 requested value、runtime accepted/selected value 与证据行；若日志格式没有给出最终 handler，只能标为未确认，不能把 requested value 复制到“实际路径”列。

## 七、同节点 P2P 与跨节点 remote transport 必须分开测试

一个两节点、每节点多 GPU 的作业中，目标 PE 的位置决定候选 path。同节点 peer 可能走 NVLink/NVSwitch/PCIe P2P，跨节点 peer 才需要 remote transport；对本 PE 的操作又可能退化为本地访问。如果 benchmark 将三类 target 混在同一平均值中，结果无法用于验证 transport。

最小测试矩阵应至少包含：单 PE 本地基线；同节点、不同 GPU 的 PUT/GET；跨节点、拓扑较近 NIC 的 PUT/GET；跨节点、拓扑较远或不同 NIC mapping 的对照。每个 case 记录 source/destination PE、GPU UUID、HCA/port、message size、API scope、blocking/NBI、completion point 和 validation method。数据正确性应先于性能，目标 buffer 要用已知 pattern 与序号校验，避免只看程序未报错。

`NVSHMEM_DISABLE_P2P=1` 是诊断开关，可在支持其他可行 path 的环境中与默认配置做 A/B 对照。若同节点性能或 runtime 日志随开关发生预期变化，可以增强 P2P 因果证据；但禁用 P2P 可能使某些 local-only 配置无法通信，不能假定一定有网络 fallback。类似的 `NVSHMEM_DISABLE_NVLS`、`NVSHMEM_DISABLE_GDRCOPY` 只针对各自能力，不应一次同时修改多个开关。

## 八、证明 IBGDA 时必须核对 NIC handler

IBGDA 的证据链至少包含四步。第一，安装或构建记录显示包含 `NVSHMEM_IBGDA_SUPPORT=1` 对应能力；第二，平台满足 GDAKI 的 HCA、MOFED、driver 和 DMA-BUF/peer-memory 前提；第三，运行配置启用 IBGDA；第四，runtime 证据确认实际 handler。当前 `NVSHMEM_IBGDA_NIC_HANDLER` 可取 `auto`、`gpu`、`cpu` 或 `cpu_cuda_memory`，其中 `auto` 会优先 GPU、但不支持时回退 CPU。

因此，“IBGDA enabled”与“GPU rang the NIC doorbell”仍不是同一句话。若为了验证 GPU handler 而显式设置 `gpu`，初始化或通信失败是有效的否定证据，不能改回 `auto` 后把成功运行解释为 GPU path。相反，显式 `cpu` 可以作为功能对照，帮助分离 IBGDA transport 的其他资源与 doorbell processor；对照时必须保持 message size、GPU/NIC mapping、QP/batch 参数和 workload 一致。

Profiler 可提供辅助证据。`NVSHMEM_NVTX` 的 `proxy`、`rma_blocking`、`rma_nonblocking`、`memorder` 等组可以让 Nsight Systems 观察 NVSHMEM 活动；CPU profile 能看到 proxy 线程是否持续处理请求，NIC counters 能显示 port traffic。但没有 CPU hotspot 不等于 GPU handler，NVTX event 也不等于一个底层 WQE；它们应与日志和强制 handler A/B 一起解释。

## 九、用受控对照建立因果关系

每次对照只改变一个因素。验证 P2P 时只改变 `NVSHMEM_DISABLE_P2P`；验证 NIC mapping 时只改变 HCA mapping；验证 IBGDA handler 时只在 `gpu` 与 `cpu` 间切换；验证 batch 时只改变 batch size。若同时切换 transport、QP 类型、message size 和 GPU mapping，即使吞吐发生变化，也无法判断原因。

建议为每个假设设计“预期变化”和“否定条件”。例如，假设默认 PE0 使用 mlx5_0，则显式把 PE0 映射到 mlx5_1 后，runtime 日志和相应 port counter 应共同迁移；如果 counter 不变，应先怀疑映射变量未生效或流量不是目标 operation，而不是直接宣布 topology 不重要。假设 GPU handler 生效，则强制 `cpu` 后 proxy activity 应增加或行为特征改变；若完全不变，说明观测粒度不足或原假设需要重查。

性能差异只能作为因果证据之一。一次运行更快可能来自 clock、NUMA placement、network contention、warm-up、registration cache 或消息批处理。正式比较应固定 GPU clocks/应用频率策略（若平台允许且有权限）、CPU affinity、message sequence、迭代次数和同步边界，记录分布而非单个最好值。E01 将进一步建立性能测量方法；C02 只要求测试足以辨认路径。

## 十、建立可审计的路径判定表

最终报告应对每一种目标路径给出结论强度，而不是统一写“NVSHMEM 正常”。下面的表格可以直接用于核验记录：

| 命题 | 最低必要证据 | 当前状态 | 不足时如何表述 |
| --- | --- | --- | --- |
| GPU–GPU 支持 P2P | topology/P2P capability + CUDA 功能测试 | 未测试 | 硬件关系已知，P2P 功能未确认 |
| NVSHMEM 同节点实际走 P2P | runtime 日志或 P2P toggle A/B + operation 正确性 | 未测试 | P2P 可用，实际选择未确认 |
| HCA 可直接访问 GPU memory | GPU-memory RDMA perftest 成功 + module/DMA-BUF 证据 | 未测试 | 具备软件前提，GDR 功能未确认 |
| NVSHMEM 跨节点使用指定 HCA | runtime mapping + 指定 port traffic + 正确性 | 未测试 | remote 通信成功，HCA 映射未确认 |
| IBGDA transport 已启用 | 构建记录 + accepted config + runtime transport 日志 | 未测试 | 已请求 IBGDA，是否启用未确认 |
| GPU handler 实际提交 | handler=gpu 的 runtime 证据 + 无回退 + 对照/trace | 未测试 | IBGDA 可用，doorbell processor 未确认 |
| 某路径性能更好 | 固定条件的重复 A/B 与统计结果 | 未测试 | 功能路径已确认，性能优劣未知 |

“未测试”和“失败”必须分开。未运行 GPU-memory perftest 不能写成 GPUDirect RDMA 不支持；显式 GPU handler 初始化失败则可记录该配置在当前环境失败，但仍需日志判断是 build、driver、mapping 还是 handler capability。报告还应附原始输出路径和命令，而不是只保存人工归纳。

## 十一、一个推荐的分阶段核验流程

第一阶段冻结软件和硬件清单，验证所有节点的 driver、CUDA、NVSHMEM、MOFED/RDMA stack、GPU/NIC firmware 与可见设备。第二阶段画 GPU–NIC/NUMA 拓扑，建立 PE→GPU UUID→HCA:port 映射。第三阶段先跑 host-memory RDMA baseline，再跑 GPU-memory RDMA baseline，分离 fabric 与 GPUDirect RDMA 问题。

第四阶段运行最小 NVSHMEM correctness workload，分别选择同 PE、同节点 peer 和跨节点 peer，保存每 PE 的 INFO/DEBUG 日志。第五阶段才启用或强制目标 transport/handler，并用一次一个 feature toggle 的方法建立因果证据。第六阶段加入 NVTX、Nsight Systems 和 NIC counters，核对 runtime 结论与实际活动。最后才进行可重复的 latency/bandwidth/message-rate 测量。

```text
版本与设备清单
  -> topology 与 PE/GPU/NIC 映射
  -> host-memory RDMA baseline
  -> GPU-memory RDMA baseline
  -> NVSHMEM local/P2P/remote correctness
  -> transport + handler runtime confirmation
  -> profiler/counter 交叉验证
  -> 受控性能比较
```

如果某阶段失败，应停在该层收集证据。例如 host-memory RDMA 不通时不继续归咎于 IBGDA；GPU-memory perftest 失败时不先调 NVSHMEM QP 数；NVSHMEM remote correctness 尚未通过时不讨论 message rate。分层顺序能显著减少“多个变量一起改，偶然跑通却不知道原因”的情况。

## 十二、常见误判及其根因

1. **把 PIX/PXB 当作实际数据路径证明。** 它只描述候选 PCIe 关系，实际 HCA mapping 和 transport 仍需运行时证据。
2. **把 `nvidia_peermem` 已加载当作 GPUDirect RDMA 已工作。** Module 是前提之一，必须用 GPU-memory RDMA 操作验证。
3. **把 `NVSHMEM_INFO` 中的 requested variable 当作 selected path。** INFO 主要说明配置，最终选择和回退要看日志及运行观测。
4. **把 `auto` handler 当作 GPU handler。** `auto` 明确允许回退 CPU，必须确认最终 processor。
5. **把 CPU proxy 活跃当作 host staging。** Proxy 可能只提交控制请求，payload 仍由 NIC 直接访问 GPU memory。
6. **把跨节点通信成功当作 IBGDA 成功。** Proxy-based IB/UCX 等 path 同样可以正确完成。
7. **把单次带宽变化当作路径证明。** NUMA、clock、拥塞、warm-up 和 batch 都可能造成相关变化。
8. **同时修改多个环境变量。** 结果失去可归因性，也难以复现。
9. **忽略目标 PE 位置。** local、同节点 P2P 和跨节点 remote 样本混合后无法判断 transport。
10. **直接修改 IOMMU、ACS 或 kernel module。** 这属于平台级高风险配置，不是普通应用诊断的第一步。

## 十三、掌握标准与后续路线

完成本课后，读者应能为“支持 P2P”“支持 GPUDirect RDMA”“NVSHMEM 使用指定 HCA”“IBGDA transport 已启用”和“GPU handler 实际提交”分别列出不同的最低证据，能够解释为什么 topology、environment、startup log、profiler 与 A/B test 必须相互交叉，而不是任选其一。

读者还应能在真实环境中按阶段采集版本、GPU/NIC BDF、NUMA、RDMA port、peer-memory/DMA-BUF、PE mapping 和 runtime logs，先验证 host RDMA，再验证 GPU memory，最后验证 NVSHMEM transport 与 handler；遇到失败时应保留“未确认”，不从硬件型号或性能猜实现。

C00—C02 至此完成从 runtime control plane、稳态 data/submit path 到部署证据链的过渡。下一篇 D00“IBGDA 实现总览”尚未创建；进入 D 模块前必须选择一个实际 NVSHMEM source repository 与 commit，使 QP、WQE、doorbell、completion、RC/DCI/DCT 和 device metadata 的结论都能定位到具体文件与 symbol。

## 主要来源

- NVIDIA, [NVSHMEM Environment Variables](https://docs.nvidia.com/nvshmem/api/latest/gen/env.html)：transport、P2P、NIC mapping、IBGDA handler、DEBUG/INFO 与 NVTX 选项。
- NVIDIA, [Using NVSHMEM](https://docs.nvidia.com/nvshmem/api/latest/using.html)：GDAKI/IBGDA/GPUNetIO 的软件前提和启用方式。
- NVIDIA, [NVSHMEM Performance](https://docs.nvidia.com/nvshmem/release-notes-install-guide/best-practice-guide/performance.html)：transport/NIC 选择、feature toggles、profiling 与一次改变一个设置的原则。
- NVIDIA, [NVSHMEM Device APIs](https://docs.nvidia.com/nvshmem/release-notes-install-guide/best-practice-guide/apis.html)：P2P、proxy-based 与 GDAKI device path 的差异。
- NVIDIA, [NVSHMEM Installation](https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/nvshmem-install-proc.html)：IBGDA、UCX、libfabric、GPUNetIO 等构建能力及依赖。
- NVIDIA, [CUDA GPUDirect RDMA Guide 13.4](https://docs.nvidia.com/cuda/gpudirect-rdma/)：PCIe root complex、IOMMU、BAR/peer-memory 与 `nvidia-peermem` 边界。
- NVIDIA, [`nvidia-smi` Documentation](https://docs.nvidia.com/deploy/nvidia-smi/)：`topo -m`、GPU/NIC matrix、P2P capability 和 PIX/PXB/PHB/NODE/SYS 图例。
- NVIDIA, [NVSHMEM Troubleshooting and FAQs](https://docs.nvidia.com/nvshmem/api/latest/faq.html)：runtime debugging、bootstrap plugin 与 GPU memory registration 故障。
- NVIDIA, [NVSHMEM 3.7.2 Release Notes](https://docs.nvidia.com/nvshmem/release-notes-install-guide/release-notes/release-3720.html)：本文核对时的发布周期；滚动页面不代表固定源码 commit。
