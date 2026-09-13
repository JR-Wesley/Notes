# 物理服务器、NUMA 与 PCIe 根层次

本篇对应 [[02-AISystem/cluster-and-hardware/00-Overview-GPU服务器：从物理硬件到Linux设备系统|GPU 服务器知识地图]] 的第 1 阶段。目标是看到一张服务器框图或 Linux 拓扑时，能回答三个问题：计算线程在哪里执行，数据存在哪块内存，设备通过哪条主机 IO 路径接入。先建立这些关系，下一篇再深入 BDF、Bridge 和枚举。

参考平台沿用 DGX H100，讨论范围以双路 x86、离散 GPU、主机 DRAM 和 PCIe IO 为主。本篇不把 Grace Hopper 等 CPU–GPU 一致性互联系统的内存语义直接套入该模型。以下精简拓扑与输出均为教学示意；未访问用户服务器、运行命令或测量性能。

## 1. 一台服务器同时存在几种连接关系

服务器不是 CPU 下挂一串同类设备。CPU 核通过片上互联与缓存层次访问内存控制器，内存控制器再通过内存通道连接 DIMM。GPU 则有自己的执行单元和 HBM；NIC 连接外部网络；NVMe 控制器提供存储访问。这些部件中的许多通过 PCIe 接入主机，但 CPU 内存通道、GPU NVLink fabric 和板级管理接口并不因此变成 PCIe。

以 DGX H100 为例，官方给出双 Intel Xeon 8480C、8 张 H100、4 颗 NVSwitch，并配置 ConnectX 网络设备、系统盘与数据盘。实际设备归属应查官方整机拓扑，不能只凭“8 GPU”推测 PCIe Switch 数量或 GPU/NIC 配对。[DGX H100 系统组成与拓扑](https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html)

| 组件 | 在系统中承担的工作 | Debug 时需要区分的对象 |
| --- | --- | --- |
| CPU socket/package、core、hardware thread | 执行 OS、驱动和应用线程；经缓存、内存控制器和 IO 接口访问外部资源 | Linux CPU 编号通常标识逻辑 CPU，不是整颗处理器编号 |
| 内存控制器、channel、DIMM/DRAM | 为主机物理内存提供存储与访问通路 | NUMA node 不是一根 DIMM；容量、通道填充和带宽是不同信息 |
| GPU 与 HBM | 执行 GPU 工作并提供 GPU 本地内存 | 主机 NUMA node 关联不是 GPU HBM 的容量或所属编号 |
| NIC/HCA | 连接 Ethernet/InfiniBand 等网络，依具体设备提供 RDMA | PCI function、网络端口、Linux netdev、RDMA device 不保证一一对应 |
| NVMe controller 与存储介质 | 接收存储命令、访问 namespace 背后的介质 | 控制器 BDF、namespace、块设备名属于不同层次 |
| PCIe Switch | 在多个 PCIe 端口之间转发事务 | 一个实体 Switch 会在 PCI 层次中表现为多个端口/Bridge，不能按单行数实体芯片 |
| Retimer、连接器、riser、线缆与背板 | 组成或改善电气传输路径，部分组件可能带管理逻辑 | 实际信号路径不一定全部作为 PCI Bridge 出现在 `lspci` 中 |
| BMC、板级 MCU/CPLD | 监测、管理或实现平台控制逻辑；细节由板级设计决定 | 管理记录、供电状态、复位状态与主机枚举结果分别取证 |

下面的 A/B 是局部性区域标签，不是 Linux 编号；图不声明 DGX H100 的实际支路分配：

```text
              主机内存与一致性互联（简化）
 DIMM ─ 内存控制器 ─ CPU package A ═ CPU互联 ═ CPU package B ─ 内存控制器 ─ DIMM
                          │                       │
                    主机 PCIe 接口           主机 PCIe 接口
                          │                       │
                     Root Port(s)            Root Port(s)
                          │                       │
                     PCIe Switch             PCIe / Switch
                       /      \                  /      \
                    GPU-A     NIC-A           GPU-B     NVMe
                      │         │               │
                     HBM      外部网络          HBM
                      └──── NVLink / NVSwitch ───┘

 BMC ─ 管理接口 ─ 传感器 / 电源控制 / 板级 MCU、CPLD 等
```

这张图同时描述内存局部性、主机 IO 和 GPU 高速互联，管理关系单独列出。画实际系统时应保留这种区分：一根 NVLink 边不能替代一根 PCIe 边，BMC 能读取某传感器也不能证明主机能通过 PCIe 访问目标设备。

## 2. NUMA：共享地址空间不意味着访问代价相同

在典型双路 x86 服务器中，两个 CPU package 都能访问主机内存，但内存控制器和 DRAM 分布在不同位置。A 上运行的线程访问 A 侧内存与访问 B 侧内存，经过的互联资源不同。NUMA（Non-Uniform Memory Access）描述这种访问非均匀性；“远端”并不表示不能访问，也不等于每次访问都必然慢一个固定倍数。缓存命中、争用、内存带宽与实际负载会影响测量。

需要分别识别四种编号：package/socket 表示处理器物理组织，core 表示执行核心，logical CPU 是 Linux 调度使用的 CPU 标识，NUMA node 表示内核使用的局部性组织。开启 SMT 时一个 core 可对应多个 logical CPU；同一 package 也可能被配置为多个 NUMA node。不要用 CPU 编号连续性推断 core 或 socket 归属，应读取拓扑属性。[Linux CPU topology](https://docs.kernel.org/admin-guide/cputopology.html)

Linux 的 NUMA 拓扑也可能有无 CPU 的内存节点，不能要求每个 node 都对应一颗插槽中的处理器。`node*/cpulist`、`meminfo` 与 `distance` 分别提供节点 CPU、内存及距离信息。distance 是相对距离描述，不是运行时延迟测试，数值不能直接标成 ns 或据此预测应用耗时。[Linux node sysfs ABI](https://www.kernel.org/doc/Documentation/ABI/stable/sysfs-devices-node)

### 线程位置、页面位置和设备位置要分别观察

假设 GPU-A 与 A 侧 IO 接口相连。一个测试线程即使正在 A 上运行，其传输缓冲区也可能位于 B 侧 DRAM。对于经过主机内存的传输，线程绑到 A 并没有自动把已经分配的页面搬到 A，更不会改变 GPU 的物理连接。此时至少需要记录：

| 位置 | 问题 | 可用证据 |
| --- | --- | --- |
| CPU 执行位置 | 线程允许在哪些 logical CPU 上运行，当前/最近在哪运行 | affinity、cpuset、CPU topology；不要把允许集合当作执行历史 |
| 主机页面位置 | 测试 buffer 的物理页分配到了哪些 node | `/proc/<pid>/numa_maps`，结合分配方式和 memory policy |
| PCIe 设备位置 | GPU/NIC 经哪个 root 层次接入，其 NUMA 关联是什么 | sysfs 父路径、`numa_node`、平台拓扑 |

默认 NUMA 内存分配通常偏向分配动作所在 CPU 的本地节点。对于常见的延迟分配匿名内存，实际触发页面分配的访问很重要，不能仅凭 `malloc` 调用时线程所在位置断定全部页面位置。内存策略、可用内存、cpuset 限制和迁移都可能改变结果。设置新的策略通常作用于之后分配的页面，不能视为已有页面已迁移。[Linux NUMA Memory Policy](https://docs.kernel.org/admin-guide/mm/numa_memory_policy.html)

在本篇 H100 离散 GPU 模型中，主机 node 的 DRAM 与 GPU HBM 分开理解。GPU BDF 的 `numa_node=0` 描述主机侧局部性关联，不表示“GPU 显存就是 node 0 的系统内存”。统一虚拟地址、可访问性、物理存储位置和一致性是不同概念，后续在传输与驱动主题中展开。

## 3. PCIe 根层次：CPU 如何接入 IO 树

Root Complex（RC）是连接主机系统与 PCIe 层次的一组逻辑功能，不应简单理解为 CPU 上唯一一条编号固定的 PCI 设备。Host Bridge 连接 CPU/主机地址域与 PCI 层次；Root Port 是根侧向下游伸出 PCIe 链路的端口，在软件层次中表现为 Bridge。RC 还可以包含集成 endpoint。一个处理器 package 可有多个 IO 根层次或端口，socket 数量不能推导 root bus 数量。[Linux PCI Host Controller 文档](https://docs.kernel.org/PCI/controller/pci-controller-drivers.html)

```text
主机 CPU / 内存地址域
          │
     Host Bridge
          │
       Root Bus
        /    \
 Root Port  Root Port
     │          │
 Endpoint    Switch upstream port
                   │
               Switch 内部层次
                 /       \
          downstream   downstream
               │           │
              GPU         NIC
```

Root Bus 是软件 PCI 层次的起点，Root Port 是该层次里的端口对象，二者不是同一个东西。Switch 的 upstream/downstream 是相对根的方向；NIC 和 GPU 可以在同一个 Switch 下成为不同分支，不必一张卡直接接一个 Root Port。

PCIe 枚举能发现下游 function，但 Host Bridge 本身的发现还需要平台信息。对采用 ACPI 的 x86 平台，firmware 提供 Host Bridge、配置空间访问方法与地址窗口等描述，Linux 才能在相应根下继续发现设备。因此“用 PCI 扫描找出包括根在内的一切”不是完整模型。[Linux 6.14：PCI Host Bridge 的 ACPI 信息](https://docs.kernel.org/6.14/PCI/acpi-info.html)

`/sys/devices/pci0000:00` 中的 `0000:00` 表示 domain 和 root bus 标识，不能解释为 socket 0、node 0。多个 root bus 也可能共享一个 PCI domain。对设备执行 `readlink -f /sys/bus/pci/devices/<BDF>`，可以沿路径识别其 root 和上游 Bridge；把这个结果再与 NUMA 属性、平台框图对应，而不是由 bus 数字大小猜插槽位置。

PCI 设备 `numa_node` 是内核记录的关联节点，Linux v6.12 ABI 文档说明它通常来自 firmware 提供的 proximity 信息，未知值为 `-1`，并允许特定管理写入覆盖。因此它是定位线索，不是独立测量出来的布线证明。本轮只读，不通过修改该值“修复”物理拓扑。[Linux v6.12 PCI sysfs ABI](https://raw.githubusercontent.com/torvalds/linux/v6.12/Documentation/ABI/testing/sysfs-bus-pci)

## 4. 为什么 GPU 通信仍然需要理解 CPU 和 PCIe

理解通信路径时，要先分清“CPU 参与控制”与“数据是否经过主机 DRAM”。应用线程和驱动可能在 CPU 上建立队列、分配资源和提交工作，但大块数据可由设备 DMA 搬运，未必由 CPU 指令逐字节复制。反过来，GPU 间存在高速 fabric 也不保证某次通信一定走该 fabric。

| 场景 | 需要核对的路径 | 这一层能提供什么判断 |
| --- | --- | --- |
| 主机 buffer 与 GPU 传输 | buffer 所在 DRAM node → 主机 IO 根 → GPU | CPU affinity 与 buffer placement 都可能相关；线程位置本身不足以描述数据路径 |
| GPU 与 GPU 传输 | 可能走 NVLink/NVSwitch、PCIe P2P 或经主机内存的路径 | PCIe 近邻和 NVLink 邻接分别核查；存在物理通路不证明软件已选用 |
| GPU 与 NIC 的 GPUDirect RDMA | NIC 与 GPU 间的可达路径及 peer memory 支持 | 相同 NUMA node 不是充分条件；还涉及 Bridge/RC、ACS、IOMMU、驱动与平台支持 |
| 两设备同时向主机传输 | 各自下游链路与共享 upstream 链路 | 下游都为 x16 不意味着可同时独占各自峰值，需检查共享瓶颈与负载 |

NVIDIA 的 GPUDirect RDMA 文档讨论 PCIe 拓扑与不同路径的限制。因此“GPU 和 NIC 都在 node 0”最多是初筛条件，不能单凭它判定 GDR 必然成功或性能最优。[GPUDirect RDMA 官方文档](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html)

这些判断是后续 [[02-AISystem/Distributed/RDMA/RDMA|RDMA]] 与 [[02-AISystem/Distributed/NCCL/NCCL|NCCL]] 学习的前提。本篇不根据纸面拓扑承诺传输速率，也不执行通信 benchmark。

## 5. 管理组件为什么属于排障链条

BMC 提供与主机 OS 不同的管理视角。DGX H100 的 BMC 文档明确支持传感器监控、事件与远程控制等功能。因此主机没有启动或无法登录时，BMC 仍可能提供硬件线索，但可用性依赖管理控制器自身供电、网络和运行状态。[DGX H100 BMC 文档](https://docs.nvidia.com/dgx/dgxh100-user-guide/bmc.html)

MCU、CPLD、Retimer 或背板管理逻辑的职责需要查具体服务器服务手册。不能从 MCU 三个字判断它一定控制哪组 GPU，也不能认定 CPLD 必然暴露一个 PCI BDF。后续处理“整个分支消失”时，需要寻找设备之间共享的供电、时钟、复位和上游链路；它们可能比软件树中的共同父节点覆盖更大或更小的范围。

例如，BMC 清单中仍有某块 GPU 板，而主机看不到其 BDF，只能证明两个观察视角不一致。库存信息可能并非本次 PCIe 探测结果，应核对更新时间、告警和硬件状态。相反，主机能枚举也不能证明板上每项管理功能或所有 GPU 链路正常。这是证据边界，不是预先认定某一侧错误。

## 6. 在 Linux 中做一次只读拓扑核对

以下三组命令面向宿主机 Linux，要求相应工具存在；工具版本可能影响可用字段。容器或 VM 所见拓扑、权限与 cpuset 可能受限制，先记录环境，不能将视图缺失直接解释成硬件缺失。下面无模拟实测结果，所有命令均未在用户服务器运行。

先核对 CPU 与 node，而不是先按经验给线程绑核：

```bash
uname -r
lscpu -e=CPU,NODE,SOCKET,CORE,ONLINE
numactl --hardware

for node_dir in /sys/devices/system/node/node[0-9]*; do
    [ -d "$node_dir" ] || continue
    printf '\n%s\n' "$node_dir"
    cat "$node_dir/cpulist"
    cat "$node_dir/distance"
done
```

`lscpu` 的每一行对应一个 logical CPU，SOCKET/CORE/NODE 字段要联合看。`numactl --hardware` 展示可用节点、CPU 和内存信息；它不是内存延迟 benchmark。[numactl 官方手册](https://github.com/numactl/numactl/blob/master/numactl.8)

再选择一个实际 GPU 或 NIC BDF，核对其根路径与 NUMA 关联：

```bash
lspci -D -t
lspci -D -nnk

# 将示例替换为上述输出中的实际 BDF
bdf=0000:03:00.0
dev_dir="/sys/bus/pci/devices/$bdf"
if [ -d "$dev_dir" ]; then
    readlink -f "$dev_dir"
    for attr in numa_node local_cpulist; do
        if [ -r "$dev_dir/$attr" ]; then
            printf '%s: ' "$attr"
            cat "$dev_dir/$attr"
        fi
    done
    lspci -D -s "$bdf" -nnk
else
    printf '%s\n' '当前视图没有此 BDF：请核对地址、环境和设备身份'
fi
```

父路径回答“挂在哪个内核 IO 层次”，NUMA 属性回答“内核记录的局部性关联”，两者配合使用。缺少属性或得到 `-1` 时保留“未知”，不要自行补成 0。此时也不需要写 sysfs、更改 BIOS 或执行 PCI rescan。

最后核对当前 shell 的允许资源与页面分布，理解进程视图；检查真实测试进程时需换成其 PID，并考虑各线程的 affinity：

```bash
taskset -pc "$$"
grep -E 'Cpus_allowed_list|Mems_allowed_list' "/proc/$$/status"
head -n 12 "/proc/$$/numa_maps"
```

这里展示的是 shell，不是 GPU workload；`numa_maps` 也不是 GPU HBM 分布图。允许 CPU/node 集合不说明进程已经使用了所有这些资源，少量页面样本不能代表整个测试 buffer。后续实验再控制 buffer 初始化与内存策略，本轮不靠调整绑定来追求未经测量的优化。

练习结束应能形成一条证据链：选择一个实际设备 → 保留 BDF 与稳定身份 → 解析根路径 → 读取 node 关联 → 找到该 node 的 CPU 集合 → 对照进程允许资源。遇到未知就记录未知，不把示意服务器编号填进实机记录。

## 7. 用本篇知识缩小 Debug 范围

| 现象 | 本篇支持的下一步 | 暂时不能得出的结论 |
| --- | --- | --- |
| 双路机器显示四个 NUMA node | 对照 CPU 的 SOCKET/NODE 映射与平台配置 | 不能断言装了四颗 CPU 或 topology 错误 |
| GPU 和 NIC 的 `numa_node` 相同，通信仍慢 | 比较 PCI 父路径、共享链路、实际传输方式与 buffer placement | 不能认定必然走最短 P2P 路径 |
| 一组 endpoint 同时不见，另一组正常 | 找共同上游，再查平台供电/复位关系和时间线 | 不能只凭同时消失断定某一颗 Switch 损坏 |
| CPU affinity 看起来正确，主机到 GPU 传输仍慢 | 看页面分配位置、共享资源与测试条件 | 不能把“绑核成功”当作“内存已本地化” |
| sysfs node 为 `-1` | 查 firmware 信息、平台资料与其他拓扑证据 | 不能解释成 node 0，也不能据此判断设备失效 |
| 管理清单存在而 OS 无 BDF | 对齐设备身份、时间与管理/主机视角 | 不能认为 BMC 清单等价于当前枚举结果 |

已有 [[02-AISystem/cluster-and-hardware/单机拓扑分析|单机拓扑分析]] 的 XML 适合作为后续练习材料，但其中 `numaid` 不是物理 socket 数量的充分证据，通信库树可能经过抽象。`gdr=1` 或链路速率字段也不能单独证明所有 P2P 路径可用、实测达到该带宽或不存在共享瓶颈。这里限定其证据用途，未改动原文。

另外，[[02-AISystem/cluster-and-hardware/集群架构|集群架构]] 中涉及 UMA 与 NUMA 的内容可作为历史关联入口；本篇使用“非均匀内存访问”的准确含义。UMA 的访问均匀性与 NVSHMEM 等语境中的 symmetric memory 不是同一个概念，不能从术语相似推出相同内存模型。后续若需要修订旧文，另行限定范围。

## 8. 来源与下一篇

本轮于 2026-09-13 查阅并使用文内链接的一手来源：NVIDIA DGX H100 系统/BMC 文档、Linux CPU topology 与 node ABI、NUMA Memory Policy、PCI Host Controller、Linux 6.14 ACPI Host Bridge 文档、Linux v6.12 PCI sysfs ABI、numactl 手册和 NVIDIA GPUDirect RDMA 文档。固定版本 ABI 用于说明属性语义，不代表用户运行这些 kernel 版本；滚动文档的代码细节未作为源码分析。

本篇提供事实性机制说明与教学推理；平台局部示意图不是实际 DGX 布线图，Debug 分支不是已确认故障。没有运行观察命令、修改服务器设置或执行性能测试。实际服务器型号、BIOS NUMA 配置、版本以及现有 XML 来源仍待确认。

下一篇见 [[02-AISystem/cluster-and-hardware/02-PCIe拓扑与BDF：从设备地址追踪上游|PCIe 拓扑与 BDF：从设备地址追踪上游]]，对应知识地图第 2 阶段，解释 Root Port/Bridge/Endpoint、BDF 与 sysfs 树的对应，重点解决“从一个地址沿上游追踪”和“总线号、实体卡、端口数量为什么不等价”。
