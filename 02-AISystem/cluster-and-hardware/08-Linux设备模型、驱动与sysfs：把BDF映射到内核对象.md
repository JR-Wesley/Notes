---
tags:
  - AI
  - Linux
  - PCIe
  - Debug
---

# 08 · Linux 设备模型、驱动与 sysfs：把 BDF 映射到内核对象

本文位于 [[02-AISystem/cluster-and-hardware/00-Overview-GPU服务器：从物理硬件到Linux设备系统|GPU 服务器多层架构]] 的 Kernel/KMD 与 OS 观察接口交界，承接 [[02-AISystem/cluster-and-hardware/07-PCIe链路、AER与错误恢复：从可见设备到运行中稳定性|PCIe 链路、AER 与错误恢复]]。前面的“设备”主要按硬件和 PCIe function 讨论；本篇转向 Linux 内核如何把它们建模为 `struct device`、总线设备、驱动和 class，并解释为什么一个 BDF 可以存在于 `/sys`，却没有绑定驱动，更不能据此断言 GPU 或 NIC 已经可用。

## 1. device model 的对象关系

Linux device model 用统一对象描述设备、驱动、总线和 class。PCI 子系统扫描到一个 function 后，会创建对应的 `pci_dev`（其中包含通用 `struct device`），把它挂到 PCI 总线和真实父路径上；Root Port、Switch Port 和 Endpoint 因而形成内核设备树。驱动注册时提供匹配表和操作回调，PCI bus 的匹配逻辑找到候选驱动后调用 `probe()`。

```text
PCI function 0000:04:00.0
        │  pci_dev / struct device
        ├── parent: .../0000:00:01.0/.../0000:04:00.0
        ├── bus: pci
        ├── driver: <成功绑定后才有>
        └── class/subsystem: 由具体子系统注册
```

`probe()` 是驱动真正验证和初始化硬件的边界。它可能申请 BAR、设置 DMA/IOMMU、分配 MSI-X、下载设备 firmware、创建字符设备或网络接口；任何一步失败都可能使 device 对象保留而 driver 绑定失败。因而排障要分别记录“发现”“匹配”“probe 成功”“用户态接口出现”四个状态。

## 2. `/sys` 的路径如何对应真实层次

`/sys` 是 sysfs，一个导出内核对象和属性的虚拟文件系统。PCI 设备的物理层次通常可从 `/sys/bus/pci/devices/<BDF>` 进入；该目录常是指向 `/sys/devices/pci...` 真实设备树位置的符号链接。目录中的 `parent` 关系、`resource`、`config`、`enable`、`power`、`iommu_group` 和 `driver` 等入口分别反映设备树、资源、配置访问、电源、隔离组和绑定状态，具体文件会随内核配置、平台和驱动而变化。

```bash
BDF=0000:04:00.0
readlink -f /sys/bus/pci/devices/$BDF
ls -l /sys/bus/pci/devices/$BDF/driver 2>/dev/null
cat /sys/bus/pci/devices/$BDF/vendor
cat /sys/bus/pci/devices/$BDF/device
cat /sys/bus/pci/devices/$BDF/class
cat /sys/bus/pci/devices/$BDF/enable
readlink -f /sys/bus/pci/devices/$BDF/iommu_group 2>/dev/null
```

`readlink -f` 得到的长路径用于沿父级查找上游 Root Port 和 Switch；`driver` 符号链接存在表示当前 device 已绑定某个驱动，指向 `/sys/bus/pci/drivers/<name>`。绑定关系不是“驱动文件存在”的同义词：模块已加载、设备已枚举、驱动已匹配和 probe 已成功是不同证据。

## 3. `/sys`、`/dev` 与 `/proc` 的分工

| 位置 | 主要对象 | 排障时回答的问题 |
| --- | --- | --- |
| `/sys` | 内核 device/driver/bus/class 的结构和属性 | 设备是否被内核发现？父路径和 driver 绑定是什么？ |
| `/dev` | 用户态可打开的设备节点，由 devtmpfs/udev 等维护 | 应用是否有一个可访问的字符或块设备接口？ |
| `/proc` | 进程、内核运行状态和统计接口 | IRQ、内存、进程和其他运行时状态怎样？ |

例如 PCI BDF 是 `/sys` 中 PCI function 的身份；`/dev/nvidia0` 是 NVIDIA 驱动创建的用户态入口；`/proc/interrupts` 展示 Linux IRQ 统计。三者没有一一对应关系：BDF 存在不保证 `/dev` 节点存在，`/dev` 节点存在也不证明 GPU 初始化和通信路径正常。

## 4. class 与 subsystem 视图

class 按功能而不是按物理总线组织设备。例如网卡可从 `/sys/class/net/<name>` 观察，块设备可从 `/sys/class/block` 观察；这些 class 目录中的条目通常是指向物理 device 树的符号链接。同一 PCI Endpoint 可以同时出现在 PCI 物理树和某个功能 class 视图中。PCIe 拓扑排障从 `/sys/bus/pci/devices` 开始，网络、块设备或 GPU 子系统排障再转到相应 class 和驱动接口。

```text
/sys/devices/.../0000:04:00.0        物理/父子层次
        ↑ symlink
/sys/bus/pci/devices/0000:04:00.0   按 PCI 总线索引
        ↑ symlink（驱动绑定后）
/sys/bus/pci/drivers/<driver>/0000:04:00.0
        ↑ symlink（成功注册功能后可能存在）
/sys/class/net/ens5f0 或 /dev/nvidia0  功能/用户态视图
```

最后两类入口由具体驱动和子系统决定，不应把某个发行版中的名称当作硬件标准。GPU 可能暴露多个 `/dev` 节点、MIG 设备或管理接口；NIC 还会有 PF、VF、representor 与 netdev 的多层关系。

## 5. 绑定、解绑与“可见性”判读

`lspci -nnk` 同时显示 PCI 身份、内核驱动和可用模块，是从硬件视角核对绑定的便捷入口；`/sys/bus/pci/devices/<BDF>/driver` 是内核当前绑定关系的直接证据。没有 `driver` 链接时，可能是没有匹配驱动、probe 失败、设备被手动解绑，或设备正处于错误恢复/移除状态。应结合 `dmesg -T`、模块状态、上游链路和 `/sys` 目录是否刚被创建或删除来判断。

写入 `bind`、`unbind`、`remove`、`driver_override` 或 `enable` 会改变内核状态，可能触发设备复位、I/O 中断或数据丢失；本篇只给只读观察方法，不把这些动作当作通用排障第一步。

## 6. 连接到多 GPU/NIC 排障

可以把证据链写成：

```text
BDF 存在
 → sysfs 物理父路径正确
 → vendor/device/class 身份符合预期
 → driver symlink 存在
 → dmesg 显示 probe 完成
 → /dev、netdev 或 DRM/计算接口出现
 → CUDA/RDMA/NCCL 能枚举并建立通信
```

链条中断的位置决定下一步：BDF 不存在回到 firmware、链路和枚举；driver 缺失检查匹配与 probe；`/dev` 缺失检查驱动的功能注册和 udev；CUDA 或 NCCL 不可见则继续检查用户态库、权限、设备过滤、拓扑和上层 runtime。这个顺序把“Linux 看不到设备”和“应用看不到设备”区分开，避免用 `nvidia-smi` 的结果反推 PCIe 枚举结论。

## 来源

- [The Linux Kernel Device Model](https://docs.kernel.org/driver-api/driver-model/overview.html)：统一 device、bus、driver 模型与 sysfs 目的。
- [Driver Binding](https://docs.kernel.org/driver-api/driver-model/binding.html)：匹配、`probe()`、绑定关系和 sysfs 符号链接。
- [sysfs - The filesystem for exporting kernel objects](https://docs.kernel.org/5.15/filesystems/sysfs.html)：sysfs 层次、设备目录与 `/sys/dev` 入口。
- [Accessing PCI device resources through sysfs](https://docs.kernel.org/PCI/sysfs-pci.html)：PCI 设备 sysfs 资源和移除语义。

下一篇进入 KMD、用户态驱动和 GPU/NIC 可见性，继续说明“内核已绑定”如何变成用户态可使用接口。
