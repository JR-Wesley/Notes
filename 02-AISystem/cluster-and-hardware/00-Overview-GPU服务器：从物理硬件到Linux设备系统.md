# GPU 服务器：从物理硬件到 Linux 设备系统

本文是服务器架构与工程排障的学习入口，围绕“一台现代多 GPU 服务器如何从物理硬件成为 Linux 中可见、可驱动、可使用的设备系统”组织知识。读者已有 Linux、CUDA 与多 GPU 通信基础，重点补齐服务器硬件、PCIe 与 Linux device model，再连接到 NVLink/NVSwitch、RDMA 和 NCCL。本文只建立整体地图与参考模型；各主题的实现细节和实验将按顺序逐篇展开。

## 1. 主线：先确定在哪一层看到了设备

“设备可见”不是单一状态。BMC 清单记录设备、PCI 子系统发现 function、驱动初始化成功、CUDA 可以使用、NCCL 可以通信，是不同层次的证据。排障首先应说清楚：哪个观察者，在什么环境、什么时间，看到了什么对象。

```text
物理硬件：供电、时钟、复位、布线、芯片与板卡
                         ↓
平台与设备 firmware 初始化、PCIe 链路建立
                         ↓
Linux PCI 子系统发现 function、处理总线与地址资源
                         ↓
内核设备对象与 sysfs 中的设备层次
                         ↓
驱动匹配 → probe → 设备初始化与接口建立
                         ↓
用户态驱动、管理工具、CUDA / RDMA 接口
                         ↓
NCCL 等通信软件选择并使用实际传输路径
```

这是依赖关系图，不是精确启动时序。平台 firmware 与 OS 都可能参与 PCI 资源处理，设备 firmware 可能已在设备启动时运行，也可能需要驱动加载。运行期间还会发生复位、热插拔、软件移除和重新发现。后续必须把“启动时未发现”和“运行中丢失”分开分析。

Linux 将设备与驱动分开建模。设备对象存在不要求功能驱动已经成功绑定；匹配只是候选关系，probe 才尝试初始化设备。绑定成功也不能证明之后每次数据传输都正常。[Linux Driver Binding](https://docs.kernel.org/driver-api/driver-model/binding.html)

## 2. 在知识库中的位置与关联

本系列统一保存在 `02-AISystem/cluster-and-hardware/`，从 [[02-AISystem/cluster-and-hardware/cluster-and-hardware|集群与硬件目录入口]] 进入本 Overview。按用户明确调整，原 SoC/PCIE 的全部笔记及 DMA/IOMMU 正文已迁入同一目录；服务器主线中的 PCIe、Linux 机制和后续通信衔接篇也在这里连续维护。体系结构、OS、GPU、RDMA/NCCL 的既有专题仍通过链接提供基础和深入资料，不复制或移动这些邻近内容。

文件前缀表示阅读顺序，不等于下文八个学习阶段的编号：第 4 阶段拆为 04—07 多篇。00 为总览，01—10 已成文；11 是后续拟定顺序，尚未创建文件。

| 文件顺序 | 主题与依赖 | 对应学习阶段 / 状态 |
| --- | --- | --- |
| 00 | 本 Overview：范围、参考模型与知识关联 | 导航，已成文 |
| 01 | [[02-AISystem/cluster-and-hardware/01-物理服务器、NUMA与PCIe根层次|物理服务器、NUMA 与 PCIe 根层次]] | 阶段 1，已成文 |
| 02 | [[02-AISystem/cluster-and-hardware/02-PCIe拓扑与BDF：从设备地址追踪上游|PCIe 拓扑与 BDF]]；依赖物理连接模型 | 阶段 2，已成文 |
| 03 | [[02-AISystem/cluster-and-hardware/03-从上电到设备枚举：Firmware与Linux的职责边界|从上电到设备枚举]]；依赖拓扑与身份 | 阶段 3，已成文 |
| 04 | [[02-AISystem/cluster-and-hardware/04-配置空间、BAR与MMIO：从设备身份到地址资源|配置空间、BAR 与 MMIO]]；从发现走向资源 | 阶段 4，已成文 |
| 05 | [[02-AISystem/cluster-and-hardware/05-DMA与IOMMU：设备如何访问内存|DMA 与 IOMMU]]；从资源访问走向数据搬运 | 阶段 4，已成文 |
| 06 | [[02-AISystem/cluster-and-hardware/06-MSI与MSI-X：从数据完成到中断通知|MSI 与 MSI-X：从数据完成到中断通知]]；设备完成、IRQ、队列与 CPU affinity | 阶段 4，已成文 |
| 07 | [[02-AISystem/cluster-and-hardware/07-PCIe链路、AER与错误恢复：从可见设备到运行中稳定性|PCIe 链路、AER 与错误恢复]]；从静态可见性走向运行中稳定性 | 阶段 4，已成文 |
| 08 | [[02-AISystem/cluster-and-hardware/08-Linux设备模型、驱动与sysfs：把BDF映射到内核对象|Linux device/driver/class 与 sysfs]]；把 BDF 映射到内核对象和功能视图 | 阶段 5，已成文 |
| 09 | [[02-AISystem/cluster-and-hardware/09-KMD、用户态驱动与设备可见性：从内核绑定到GPU通信|KMD、用户态驱动与设备可见性]]；从内核绑定、设备节点到 CUDA/RDMA/NCCL | 阶段 6，已成文 |
| 10 | [[02-AISystem/cluster-and-hardware/10-NVLink与NVSwitch、RDMA、NUMA和NCCL：从拓扑到通信路径|NVLink/NVSwitch、RDMA、NUMA 与 NCCL]]；从物理拓扑到通信库选路 | 阶段 7，已成文 |
| 11 | [[02-AISystem/cluster-and-hardware/11-GPU服务器综合排障：从现象到证据链与最小验证|GPU 服务器综合排障]]；把各层现象串成证据链和最小验证 | 阶段 8，已成文 |

`PCIE.md` 保留为迁入的 PCIe 专题导航，`深入浅出pcie.md` 保留为来源入口，二者不占主线编号。`cluster-and-hardware.md` 同时导航本系列与目录里的其他既有主题；本文负责解释学习顺序与机制依赖，不把整个目录合并成一个大文件。后续如确需分篇，再同步调整计划编号和导航。

| 已有位置 | 在本学习体系中的作用 | 内容边界与衔接方式 |
| --- | --- | --- |
| [[01-ComputerScience/Architecture/Architecture|体系结构]] | CPU、内存层次、处理器互联的基础 | 为 NUMA 与 IO 访问路径提供前提；本入口不复制教材结构 |
| [[02-AISystem/cluster-and-hardware/PCIE|PCIe]] | 拓扑、配置空间、枚举、BAR、链路和事务 | 已整体迁入本目录；《深入浅出pcie》仍主要是来源入口，不视为已完成协议知识 |
| [[01-ComputerScience/operating-system/operating-system|操作系统]] | 页表、内存、进程与通用 IO 基础 | 保留跨领域关联；本服务器系列的 DMA、device model 等正文集中在当前目录 |
| [[02-AISystem/cluster-and-hardware/单机拓扑分析|单机拓扑分析]] | 后续真实材料解读候选 | 已有含 BDF、NUMA、GPU/NIC 的 XML；来源与硬件环境待确认，不能直接当成完整 PCIe 树或 DGX H100 实测 |
| [[02-AISystem/cluster-and-hardware/接口硬件模块|接口硬件模块]] | 接口与互联硬件资料入口 | 目前含外部来源链接；后续按问题核查，不当作已验证结论 |
| [[02-AISystem/cluster-and-hardware/集群架构|集群架构]] | 从单机扩展到跨节点 | 在掌握单机设备、NIC 与网络路径后继续 |
| [[02-AISystem/GPU/GPU|GPU]] | CUDA、GPU 架构和数据传输 | GPU 内部执行机制与主机设备生命周期分别维护，再通过传输路径关联 |
| [[02-AISystem/Distributed/RDMA/RDMA|RDMA]] | NIC、verbs、内存注册与跨机数据通路 | 先建立 PCI function、驱动与 RDMA device 的对应，再进入通信机制 |
| [[02-AISystem/Distributed/NCCL/NCCL|NCCL]] | 集合通信、拓扑发现和传输选择 | 结合已验证底层拓扑理解日志，不直接把通信失败归因于算法 |
| [[02-AISystem/Distributed/分布式基础/分布式互联技术总览|分布式互联技术总览]] | 互联技术的横向视野 | 作为对照入口；PCIe、NVLink 与网络不强行排成唯一线性协议栈 |
| [[00-ToolKit/System/Linux系统分析|Linux 系统分析工具]] | 通用工具使用方法 | 工具用法与解释设备机制的知识正文分开维护 |

附件延续主题已有的 `assets` 布局。现有拓扑附件位于 `assets/单机拓扑分析.assets/`，本轮未更改或重新解释该图片；本文使用 ASCII 图，不新增附件。未来笔记的拟定标题不是已完成产物，只有文件实际存在后才建立对应 WikiLink。

## 3. 推荐学习顺序与交付边界

每个主题依次解释系统位置、前提与核心机制、Linux 观察方法、Debug 分支，并给出少量可执行的观察方法。下表是学习顺序，不是主题完成清单。

| 顺序 | 主题与需要回答的问题 | 主要观察入口 | 推荐落点与完成证据 |
| --- | --- | --- | --- |
| 1 | 物理服务器与 NUMA：CPU socket、内存控制器、DIMM、GPU/HBM、NIC、NVMe、PCIe Switch 如何连接；BMC/MCU/CPLD 位于哪里 | 平台框图、`lscpu`、`numactl -H` | `cluster-and-hardware`；能画出主机内存、IO、GPU fabric 与管理关系，解释 socket 不一定等于 NUMA node |
| 2 | PCIe 拓扑与身份：Root Complex、Root Port、Bridge、Endpoint、domain、BDF；Switch 端口如何表示 | `lspci -D -t`、sysfs 父路径 | 本目录 02；能从 endpoint 追踪上游并区分 BDF、槽位与设备身份 |
| 3 | 从上电到枚举：BIOS/UEFI、ACPI、BMC 与设备 firmware 的职责；链路、扫描和资源处理的依赖 | 启动日志、平台配置、BMC 事件 | `cluster-and-hardware`；能区分未枚举与驱动未初始化，提出可验证的下一步 |
| 4 | PCIe 资源与传输：配置空间、BAR、MMIO、DMA、IOMMU、MSI/MSI-X、速率/宽度、AER | `lspci -vv`、`resource`、`/proc/iomem`、内核日志 | 本目录 04—07；能解释枚举成功为何不代表资源和数据通路可用 |
| 5 | Linux device model：bus/device/driver/class，匹配、probe、绑定、模块与驱动的区别 | `/sys/devices`、`/sys/bus/pci`、`driver`、`lspci -k` | 本目录 08，关联 OS；能区分不存在、未绑定、probe 失败与运行期故障 |
| 6 | 用户态可见性：KMD、用户态驱动、CUDA/NVML、设备节点、权限、容器与编号 | `/dev`、`/proc`、`nvidia-smi`、class 路径 | 本目录 09，关联 GPU/OS；能追踪宿主机可见而应用不可见的原因 |
| 7 | 多 GPU 和跨机通信：NVLink/NVSwitch、FM、RDMA、GPUDirect RDMA、NCCL 与 NUMA/PCIe 路径 | GPU 拓扑、RDMA 信息、FM/NCCL 日志 | 本目录 10，链接既有 RDMA/NCCL 专题；能区分物理路径与软件选路 |
| 8 | 综合排障：未枚举、BDF 消失、链路退化、probe 失败、GPU/NIC 不可见、拓扑不符 | 正常/异常快照、日志时间线、平台资料 | 本目录 11；形成证据 → 假设 → 验证 → 缩小范围的案例，而非只罗列命令 |

第一遍不展开 PCIe 包格式、完整内核调用链或 NCCL 全部算法。源码研究进入具体主题后再固定 kernel/driver 版本和符号；实验完成状态必须由真实结果支持。

## 4. 参考模型：DGX H100 与教学用 PCIe 地址

选用 NVIDIA DGX H100 作为贯穿案例。官方配置包含双 Intel Xeon CPU、8 张 H100 GPU、4 颗 NVSwitch，以及 ConnectX 网络设备和 NVMe 存储，并提供整机拓扑图。固定使用 H100 代际，不能把 H200、B200 或不同 OEM 的 HGX 布线和软件要求直接混入。[NVIDIA DGX H100/H200 系统介绍与拓扑](https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html)

下面是教学抽象，省略实际端口数量、设备归属和支路，不代替官方整机布线图：

```text
主机内存与 PCIe IO
 DRAM ─ CPU0 ═══ CPU互联 ═══ CPU1 ─ DRAM
         │                     │
      PCIe 根层次           PCIe 根层次
         │                     │
       Switch / GPU / NIC / NVMe 等分支

GPU 高速互联
 GPU0 ... GPU7 ── NVLink ── NVSwitch fabric

平台管理关系（不是 GPU 数据传输链）
 BMC ── 板级管理接口 ── 传感器、供电控制、MCU/CPLD 等
```

CPU socket 与 NUMA node 不保证一一对应，取决于处理器与配置。PCIe 树也不能完整描述 NVLink 路径或管理连接。实际 Debug 至少要区分主机 IO/内存关系、GPU fabric 和管理关系，再将它们通过同一设备身份对齐。

BIOS/UEFI 是主机平台 firmware，负责启动与平台初始化并向 OS 提供相关描述；BMC 是运行自身 firmware 的管理控制器。MCU 是微控制器，CPLD 是可编程逻辑器件，可能承担供电时序、复位或板级状态控制，但具体职责必须查目标平台资料。设备 firmware 运行于设备侧，其生命周期不等于主机 OS 生命周期。

KMD 指 kernel-mode driver，是驱动的内核态部分；用户态 CUDA 驱动库、管理库与应用不因此变成内核组件。FM 在本学习模型中暂指 NVIDIA Fabric Manager，它是管理 NVSwitch fabric 的用户态服务，不是 firmware 的缩写，也不代表所有厂商所说的 FM。它与内核驱动及设备协作，具体要求随平台代际变化。[NVIDIA Fabric Manager](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/)

## 5. 从物理分支到 BDF、sysfs、driver 和用户态

以下全部 BDF、GPU index 与接口编号均为教学假设，不是 DGX 固定编号，也不是服务器实测。为理解 Switch 的软件表示，仅取一个示意分支：

```text
PCI root bus 0000:00
└─ 0000:00:01.0  Root Port
   └─ 0000:01:00.0  Switch upstream port
      ├─ 0000:02:00.0  downstream port
      │  └─ 0000:03:00.0  GPU function
      └─ 0000:02:01.0  downstream port
         └─ 0000:04:00.0  NIC function
```

`0000:03:00.0` 表示 domain `0000`、bus `03`、device `00`、function `0`。BDF 通常指 bus/device/function，Linux 常展示带 domain 的完整地址。它定位 PCI function，不是物理卡序列号，也不能单凭该值确定机箱槽位。一个物理设备可能暴露多个 function；地址分配也可能因配置或枚举变化而改变。

该 GPU 的查找入口与真实 sysfs 父路径可以对应为：

```text
/sys/bus/pci/devices/0000:03:00.0
  → /sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0/0000:02:00.0/0000:03:00.0

/sys/bus/pci/devices/0000:03:00.0/driver
  → /sys/bus/pci/drivers/nvidia       # 假定成功绑定 NVIDIA 驱动
```

`/sys/bus/pci/devices` 提供按 BDF 查找的入口；解析链接后的 `/sys/devices` 路径表达内核设备父子层次。它不是机箱机械装配图。设备目录中的 `vendor`、`device` 是设备标识，`config` 暴露配置空间，`resource` 描述资源，不能把这些信息都理解成 GPU 显存内容。[Linux PCI sysfs](https://docs.kernel.org/PCI/sysfs-pci.html)

| 物理/总线对象 | 内核关系 | 用户态映射与边界 |
| --- | --- | --- |
| GPU function | PCI device → `nvidia` 驱动 | 用 `nvidia-smi` 查询 UUID、BDF、index；`/dev/nvidiaN` 是访问接口，不能由 BDF 猜 N，也不能直接把 N 当 CUDA ordinal |
| NIC function | PCI device → `mlx5_core`；RDMA 功能还涉及 `mlx5_ib` 等组件 | 从 `/sys/class/net/<接口>/device` 和 `/sys/class/infiniband/<设备>/device` 反查；实际名称、层次和数量以系统为准 |
| NVMe function | PCI controller → `nvme` 驱动 → namespace/block device | `/dev/nvme0n1` 是 namespace 对应的块设备；一张卡、一个 function、一个块设备不保证一一对应 |
| PCIe Bridge | PCI device，有自身 BDF | 不要求对应应用可打开的 `/dev` 节点；无设备节点不等于枚举失败 |

GPU index 可能变化，跨启动对照应保留 UUID 与 BDF；CUDA 可见设备过滤还可能改变应用中的编号。具体对应通过工具输出验证。[NVIDIA-SMI](https://docs.nvidia.com/deploy/nvidia-smi/index.html)

`/sys` 是设备对象与属性视图；`/proc` 提供进程与内核运行信息；`/dev` 中的节点是字符/块设备接口，不是硬件清单。dmesg 读取内核日志缓冲区，kernel journal 在有相应日志服务时提供另一种事件入口。两者的保留范围与权限不同，日志没有记录不能直接证明事件未发生。

## 6. 第一轮观察：完成一次身份追踪

下列命令供 Linux 服务器执行，本轮未运行，没有生成实测结果。需要相应工具；读取内核日志或完整 PCI 信息可能受权限限制。先记录主机/容器环境、启动时间和版本，再使用本机实际 BDF 替换示例。仅观察，不执行 reset、remove、rescan、unbind 或 firmware 更新。

```bash
uname -r
lspci -D -t
lspci -D -nnk

# 替换为本机 lspci 中实际存在的完整 BDF
bdf=0000:03:00.0
if [ -d "/sys/bus/pci/devices/$bdf" ]; then
    readlink -f "/sys/bus/pci/devices/$bdf"
    if [ -L "/sys/bus/pci/devices/$bdf/driver" ]; then
        readlink -f "/sys/bus/pci/devices/$bdf/driver"
    else
        printf '%s\n' '该设备当前没有 driver 链接'
    fi
    cat "/sys/bus/pci/devices/$bdf/numa_node"
else
    printf '%s\n' '当前环境中未找到该 BDF，请先核对地址与宿主机视图'
fi

nvidia-smi --query-gpu=index,uuid,pci.bus_id,name --format=csv
journalctl -k -b
```

`numa_node` 为 `-1` 表示关联信息未知，不是 node 0。NVIDIA 工具可能用更宽的 domain 字段显示 PCI 地址，比较前应统一表示。练习产物是一条由实际证据支持的“设备身份 → BDF → 父路径 → driver → 用户态标识”记录；没有 NVIDIA GPU 或工具不可用时，不伪造对应输出。

## 7. 从症状定位边界

排障从最后一个已经有证据成立的边界向前推进，并保留时间线。下面是候选检查方向，不是仅凭一条症状即可确定的根因。

| 症状 | 可以支持的判断 | 下一步缩小范围 |
| --- | --- | --- |
| BDF 不在 `/sys/bus/pci/devices` | 当前内核视图没有该地址的 PCI 对象 | 核对宿主机/容器、地址变化与设备身份；沿上游 Bridge 看缺失范围，对比启动和运行期日志 |
| 多个相邻 endpoint 同时消失 | 可能存在共同上游故障，仍需证据 | 查共同 Switch、Root Port、供电/复位域；不能只重装各 endpoint 驱动 |
| BDF 存在，无 `driver` 链接 | 当前未绑定驱动；不能证明 probe 曾运行 | 查 `lspci -k`、驱动匹配、模块状态、override 与 probe 日志 |
| probe 失败 | 已进入驱动初始化尝试 | 对照错误时间、资源分配、firmware、版本及设备状态，不把所有错误归为 PCIe 物理故障 |
| 驱动已绑定，GPU/NIC 工具不可用 | 绑定不足以证明功能可用 | 查运行期初始化、用户态库、设备接口、权限与容器暴露 |
| 链路宽度/速率异常 | 需要对照能力、当前状态及平台预期 | 同时检查 endpoint 与上游端口，结合负载/电源状态、AER 与平台布线判断 |
| 设备可用但多 GPU 通信失败或慢 | 问题进入数据路径或软件选择阶段，也可能暴露底层不稳定 | 检查 NVLink/fabric、FM、RDMA 网络、P2P 条件与 NUMA 亲和性，再看 NCCL 选路 |

NCCL 运行依赖底层设备、拓扑和系统配置，不能用“CUDA 单卡正常”推导全部通信路径正常。[NCCL Troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html)

## 8. 来源、适用范围与后续

本轮于 2026-09-13 整理，使用前一轮已查阅的 Linux 与 NVIDIA 官方文档。这里记录来源支持范围，不代表已核查所有软件版本的实现。Linux 在线文档和 NVIDIA 文档会更新；进入源码、安装兼容性或平台操作主题后，再固定相应版本。

| 实际使用来源 | 本文使用范围 |
| --- | --- |
| [DGX H100/H200 系统介绍](https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html) | H100 参考平台组成与官方拓扑入口；不支持本文教学 BDF 为实机编号 |
| [Linux PCI sysfs](https://docs.kernel.org/PCI/sysfs-pci.html) | PCI 设备层次、属性与资源接口 |
| [Linux Driver Binding](https://docs.kernel.org/driver-api/driver-model/binding.html) | 设备与驱动分离、匹配和绑定概念；不作为固定版本源码调用链 |
| [Fabric Manager](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/) | NVIDIA NVSwitch fabric 管理及平台差异 |
| [NVIDIA-SMI](https://docs.nvidia.com/deploy/nvidia-smi/index.html) | GPU 标识查询与编号边界 |
| [NCCL Troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html) | 通信故障与底层系统配置的关系；未执行 NCCL 测试 |

仓库已有笔记用于定位与衔接，不因被链接就视为内容全部经过核查。本文的目录分工、教学图和 Debug 顺序属于教学组织与工程推理；硬件配置来自官方资料；示意地址不是事实记录。

尚未确认用户实际服务器型号、OS/kernel/driver/firmware 版本、FM 的实际指代，以及已有拓扑 XML 的采集来源。这些信息不阻塞基础学习，但会决定后续平台实例与排障结论的适用范围。

已成文主题按学习顺序衔接：

1. [[02-AISystem/cluster-and-hardware/01-物理服务器、NUMA与PCIe根层次|物理服务器、NUMA 与 PCIe 根层次]]：主机内存、IO、GPU fabric 与管理关系。
2. [[02-AISystem/cluster-and-hardware/02-PCIe拓扑与BDF：从设备地址追踪上游|PCIe 拓扑与 BDF]]：地址、Bridge 总线范围和 sysfs 父路径。
3. [[02-AISystem/cluster-and-hardware/03-从上电到设备枚举：Firmware与Linux的职责边界|从上电到设备枚举]]：firmware、平台描述、PCI 发现与功能驱动接管。

4. 第 4 阶段首篇 [[02-AISystem/cluster-and-hardware/04-配置空间、BAR与MMIO：从设备身份到地址资源|配置空间、BAR 与 MMIO]]：从设备身份到地址资源、上游窗口与资源失败。

第 4 阶段第二篇 [[02-AISystem/cluster-and-hardware/05-DMA与IOMMU：设备如何访问内存|DMA 与 IOMMU]] 补齐设备到内存的地址与生命周期机制，依赖上一篇 BAR/MMIO，并衔接 RDMA 注册与 GPU peer memory。

第 4 阶段第三篇 [[02-AISystem/cluster-and-hardware/06-MSI与MSI-X：从数据完成到中断通知|MSI 与 MSI-X]] 补齐完成通知、Linux IRQ、队列与 CPU affinity，依赖 DMA/IOMMU 的 buffer 生命周期。

主线基础正文至此完成。下一步应以真实服务器快照、版本信息和受控通信测试补充第 1—11 篇中的实机验证。跨轮状态保存在 [[AI-Workspace/03-Projects/2026-09-GPU-Server-System/README|GPU 服务器系统学习项目]]；正文成文不代表完成实机验证。
