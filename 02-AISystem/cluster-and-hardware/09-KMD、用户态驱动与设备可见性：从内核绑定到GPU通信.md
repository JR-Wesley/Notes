---
tags:
  - AI
  - Linux
  - GPU
  - RDMA
  - Debug
---

# 09 · KMD、用户态驱动与设备可见性：从内核绑定到 GPU 通信

本文位于 [[02-AISystem/cluster-and-hardware/00-Overview-GPU服务器：从物理硬件到Linux设备系统|GPU 服务器多层架构]] 的 Kernel/KMD—用户态边界，承接 [[02-AISystem/cluster-and-hardware/08-Linux设备模型、驱动与sysfs：把BDF映射到内核对象|Linux 设备模型、驱动与 sysfs]]。KMD（Kernel Mode Driver）是运行在内核中的设备驱动部分；用户态驱动、runtime 和应用通过稳定的用户态 API 与它协作。看到 PCI BDF、内核模块、`/dev` 节点、CUDA device 或 NCCL rank，实际上是在观察同一硬件对象的不同投影。

## 1. 从 PCI function 到可用计算设备

一张 GPU 或一块 NIC 经过的层次可以写成：

```text
PCIe Endpoint / BDF
  → Linux pci_dev 与 device model
  → KMD 匹配、probe、资源和 firmware 初始化
  → 字符设备/DRM render node 或 RDMA uverbs、netdev
  → 用户态库（CUDA runtime/driver API、libibverbs、libfabric 等）
  → 应用与 NCCL/RDMA 通信库
```

KMD 负责必须受内核保护或需要硬件特权的工作，例如 PCI 配置和 BAR、DMA/IOMMU 映射、IRQ、复位、电源管理、内存隔离、队列创建以及错误恢复。用户态部分负责 API 封装、上下文管理、内核 ioctl 调用、缓存和调度策略；在某些高性能路径上，内核创建并验证资源后，用户态可以通过 `mmap()` 访问 doorbell、队列或共享缓冲区，从而减少系统调用。具体边界由驱动架构决定，不能把“用户态驱动”理解成完全绕过 KMD。

## 2. GPU 的几种可见性

对 GPU 至少要区分四个观察面：

| 层次 | 常见证据 | 说明 |
| --- | --- | --- |
| PCI 层 | `lspci`, `/sys/bus/pci/devices/<BDF>` | function 被枚举，身份、资源和父拓扑可见 |
| KMD 层 | `lspci -k`, `driver` 链接，`dmesg` | 驱动是否匹配、probe 是否成功、固件和资源是否初始化 |
| 设备接口层 | `/dev/nvidia*`、`/dev/dri/renderD*`、sysfs 属性 | 用户态可打开的字符设备或 DRM 接口是否建立 |
| CUDA 层 | `nvidia-smi`、CUDA device query | runtime/用户态驱动能否初始化并返回计算设备 |

这些证据是递进关系而非等价关系。PCI 可见但没有 KMD，通常不能得到 CUDA 设备；KMD 已绑定但 `/dev` 节点缺失，可能是功能注册、权限或 udev 问题；`nvidia-smi` 能列出设备，也不能证明 NCCL 选出的 PCIe、NVLink 或 NIC 路径性能正常。

DRM 驱动通常提供 `cardN` 和面向非显示渲染的 `renderDN` 节点；render node 允许非特权渲染客户端使用受限 ioctl，不承担 modesetting。专有 GPU 栈可能使用自身的 `/dev/nvidia*` 接口，不能用 DRM 节点名称推断所有 GPU 驱动实现。

## 3. NIC/RDMA 的另一条用户态路径

NIC 经过 PCI probe 后，驱动可能注册 netdev（如 `ens5f0`）和 RDMA device。普通 socket 通过网络协议栈使用 netdev；RDMA verbs 则通过 `libibverbs` 访问 `/dev/infiniband/uverbsN` 完成慢路径资源管理，数据面可以把经过内核验证的队列和缓冲区映射到用户态。于是下面对象不能混为一谈：

```text
PCI BDF 0000:05:00.0
  ├─ PF/VF 的 PCI function
  ├─ netdev：ens5f0
  └─ RDMA device：mlx5_0 → /dev/infiniband/uverbsN
```

PF、VF、representor、netdev 和 RDMA device 可能有不同名称和生命周期；一个 BDF 也可能对应多个用户态入口。排查 GPU-NIC P2P 或 NCCL RDMA 问题时，应把 `lspci`、`/sys`、`ip link`、`rdma link`/`ibdev2netdev` 和用户态库输出放在同一时间线中。

## 4. 用户态 API、ioctl 与 mmap 的边界

用户态通常通过 `open()` 打开设备节点，再用 ioctl 请求创建上下文、分配内存、提交命令或查询事件。KMD 在 ioctl 中检查权限、句柄、地址范围和对象生命周期；DMA buffer、fence、doorbell、IOMMU 映射等资源必须由内核或硬件安全边界保护。成功打开设备节点只代表文件接口存在，不代表每个 ioctl、固件状态或后续 DMA 都成功。

GPU DRM uAPI 的基本规则是内核先定义可兼容的用户态接口；驱动特定 ioctl、sysfs 属性和 mmap 区域共同构成用户态驱动的契约。RDMA verbs 也把资源管理和快速数据路径分开。升级 KMD、用户态库或 firmware 时，应把版本兼容作为整体检查，而不是只确认某个模块文件已加载。

## 5. 用证据链定位“应用看不到设备”

```bash
BDF=0000:04:00.0
lspci -s "$BDF" -nnk
readlink -f /sys/bus/pci/devices/$BDF
ls -l /sys/bus/pci/devices/$BDF/driver 2>/dev/null
ls -l /dev/dri/renderD* /dev/nvidia* 2>/dev/null
lsmod | rg 'nvidia|amdgpu|mlx|ib_'
dmesg -T | rg -i 'probe|firmware|gpu|nvidia|amdgpu|rdma|uverbs|failed|timeout'
```

这些命令只读状态，名称需按实际硬件调整。建议按以下顺序解释结果：

1. 没有 BDF：回到 firmware、PCIe 链路、上游端口和枚举。
2. 有 BDF、无 `driver`：检查 ID 匹配、模块、黑名单和 probe 日志。
3. 有 driver、无设备节点：检查功能注册、udev、权限和驱动初始化后半段。
4. 有设备节点、CUDA/RDMA 不可见：检查用户态库版本、环境变量、权限、容器设备映射和设备过滤。
5. 上层可见但通信失败：回到 NUMA、PCIe/NVLink 拓扑、DMA/IOMMU、IRQ、firmware 和 NCCL/RDMA 路径。

## 6. 复位、热拔插与生命周期

设备被 PCI error recovery、热拔插、unbind 或驱动卸载后，`/sys` 目录、设备节点、用户态文件描述符和上下文的生命周期可能不同。DRM 文档规定设备消失后新的打开操作可能返回 `ENXIO`，已有任务可能收到 `ENODEV` 或设备丢失错误；其他 GPU/NIC 驱动有自己的语义。遇到“BDF 重新出现但应用仍报错”，要检查旧进程是否持有失效 fd、驱动是否重建了 BAR/DMA/IRQ、用户态是否重新创建上下文。

事实是某层接口报告了存在或错误；“驱动版本不兼容”“固件导致设备不可见”等是待验证假设。必须结合 `dmesg` 时间戳、模块/库版本、容器边界和可复现实验确认，不能只依据最后一个上层错误字符串下结论。

## 来源

- [Linux DRM Userland Interfaces](https://docs.kernel.org/gpu/drm-uapi.html)：DRM 设备节点、render node、ioctl 与设备消失语义。
- [Linux Userspace Verbs Access](https://docs.kernel.org/infiniband/user_verbs.html)：`libibverbs`、`/dev/infiniband/uverbsN` 及慢/快速路径边界。
- [The Linux Kernel Device Model](https://docs.kernel.org/driver-api/driver-model/overview.html)：设备、驱动和 sysfs 的统一模型。
- [Driver Binding](https://docs.kernel.org/driver-api/driver-model/binding.html)：匹配、`probe()` 与绑定关系。

下一篇进入 NVLink/NVSwitch、RDMA、NUMA 与 NCCL，把用户态可见设备进一步连接到实际通信路径和拓扑选择。
