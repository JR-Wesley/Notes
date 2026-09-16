---
tags:
  - AI
---

# 集群与硬件

> AI 集群硬件体系：以 Host、Device、Management 三个世界为坐标，连接 PCIe、GPU scale-up、网络 scale-out、Linux/KMD 与用户态通信软件。

## GPU 服务器学习主线

先从 Overview 建立三世界、多 Fabric、三路径的总坐标，再按 Level 0—10 的能力路线学习；01—11 是各级使用的知识载体，不与 Level 一一对应，也不代表硬件启动的严格时间顺序。

- [[02-AISystem/cluster-and-hardware/00-Overview-GPU服务器：从物理硬件到Linux设备系统|GPU 服务器：从物理硬件到 Linux 设备系统]]：以三个世界、多套 Fabric 和三条路径为总坐标，提供从整机 Inventory 到 PCIe/Linux、GPU Fabric、RDMA/NCCL 和综合诊断的 Level 0—10 分级路线
- [[02-AISystem/cluster-and-hardware/01-物理服务器、NUMA与PCIe根层次|物理服务器、NUMA 与 PCIe 根层次]]：区分线程、页面与设备位置，建立主机内存、PCIe IO 和管理连接的物理模型
- [[02-AISystem/cluster-and-hardware/02-PCIe拓扑与BDF：从设备地址追踪上游|02 · PCIe 拓扑与 BDF]]：设备身份、Bridge 与父路径
- [[02-AISystem/cluster-and-hardware/03-从上电到设备枚举：Firmware与Linux的职责边界|从上电到设备枚举：Firmware 与 Linux 的职责边界]]：启动依赖、平台描述、PCI 发现、驱动接管与日志时间线
- [[02-AISystem/cluster-and-hardware/04-配置空间、BAR与MMIO：从设备身份到地址资源|04 · 配置空间、BAR 与 MMIO]]：设备资源与上游地址窗口
- [[02-AISystem/cluster-and-hardware/05-DMA与IOMMU：设备如何访问内存|05 · DMA 与 IOMMU]]：设备访问内存及映射生命周期
- [[02-AISystem/cluster-and-hardware/06-MSI与MSI-X：从数据完成到中断通知|06 · MSI 与 MSI-X：从数据完成到中断通知]]：设备完成、Linux IRQ、队列与 CPU affinity 的关系
- [[02-AISystem/cluster-and-hardware/07-PCIe链路、AER与错误恢复：从可见设备到运行中稳定性|07 · PCIe 链路、AER 与错误恢复]]：链路状态、错误计数、隔离、复位与驱动恢复
- [[02-AISystem/cluster-and-hardware/08-Linux设备模型、驱动与sysfs：把BDF映射到内核对象|08 · Linux 设备模型、驱动与 sysfs]]：BDF、device/driver/class、`/sys`、`/dev` 与 `/proc` 的对应关系
- [[02-AISystem/cluster-and-hardware/09-KMD、用户态驱动与设备可见性：从内核绑定到GPU通信|09 · KMD、用户态驱动与设备可见性]]：从内核绑定、设备节点到 CUDA/RDMA/NCCL
- [[02-AISystem/cluster-and-hardware/10-NVLink与NVSwitch、RDMA、NUMA和NCCL：从拓扑到通信路径|10 · NVLink/NVSwitch、RDMA、NUMA 与 NCCL]]：从物理拓扑到通信库选路
- [[02-AISystem/cluster-and-hardware/11-GPU服务器综合排障：从现象到证据链与最小验证|11 · GPU 服务器综合排障]]：从现象、分层证据到最小验证

主线正文集中在本目录，00 为 Overview，01—11 按上述顺序阅读。主线基础正文已完成；后续以真实服务器快照和受控实验补充验证，不预建空白笔记。

## 专题入口与既有资料

- [[02-AISystem/cluster-and-hardware/PCIE|PCIe 专题入口]]：由原 SoC/PCIE 整体迁入的导航
- [[02-AISystem/cluster-and-hardware/深入浅出pcie|深入浅出 PCIe]]：保留的来源入口，不占主线编号
- [[02-AISystem/cluster-and-hardware/集群架构|集群架构]]：AI 训练/推理集群拓扑
- [[02-AISystem/cluster-and-hardware/单机拓扑分析|单机拓扑分析]]：单机内互连拓扑
- [[02-AISystem/cluster-and-hardware/AI芯片硬件发展|AI芯片硬件发展]]：AI 芯片演进
- [[02-AISystem/cluster-and-hardware/半导体工艺|半导体工艺]]：制程基础
- [[12-现代GPU服务器互联|接口硬件模块]]：硬件接口模块
