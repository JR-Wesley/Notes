---
tags:
  - AI
  - GPU
  - NVLink
  - RDMA
  - NCCL
---

# 10 · NVLink 与 NVSwitch、RDMA、NUMA 和 NCCL：从拓扑到通信路径

本文承接 [[02-AISystem/cluster-and-hardware/09-KMD、用户态驱动与设备可见性：从内核绑定到GPU通信|KMD、用户态驱动与设备可见性]]。目标不是背诵某台机器的拓扑矩阵，而是建立一个判断方法：当 NCCL、CUDA P2P 或 GPUDirect RDMA 性能异常时，先确认实际硬件路径，再确认软件是否有能力发现并选择这条路径。

## 1. 一台典型多 GPU 节点的多层互连

```text
CPU socket 0 / NUMA 0                         CPU socket 1 / NUMA 1
      │ PCIe Root Complex                            │ PCIe Root Complex
      ├─ PCIe Switch ─ GPU0/GPU1 ─┐        ┌─ PCIe Switch ─ GPU2/GPU3
      └─ NIC0/NVMe                  │        │                  └─ NIC1/NVMe
                                    └─ NVLink/NVSwitch ──────────┘
```

图中有三种容易混淆的关系：PCIe 是主机访问 Endpoint 的总线层次；NVLink/NVSwitch 是 GPU 间或 GPU fabric 内的高速互连；NIC 的 RDMA 路径仍以 NIC 的 PCIe 位置、DMA/IOMMU 和 GPU peer-memory 支持为前提。一个 GPU 可以同时拥有 PCIe BDF、NVLink fabric 端点、CUDA ordinal 和 NCCL rank，这些编号不必相同。

## 2. NVLink/NVSwitch 与 PCIe 的职责

NVLink 提供 GPU 间的高速 peer-to-peer 通路；NVSwitch 把多个 GPU 的链路组织成更规则的 fabric，使不相邻 GPU 也能获得可预测的交换路径。PCIe 仍负责主机枚举、配置、驱动接管以及 CPU/设备之间的控制和数据访问。NVLink 可用不等于 PCIe 正常，PCIe 可用也不等于 NVLink fabric、链路训练或 NVSwitch 管理状态正常。

`nvidia-smi topo -m`、`nvidia-smi nvlink -s` 或厂商拓扑工具给出的是软件观察结果；它们不替代 `lspci -t`、BDF 父路径和平台布线资料。DCGM 文档也把拓扑视为已知 CPU、PCIe 和 direct-NVLink 关系的描述，不把拓扑查询当成带宽测试。

## 3. RDMA 与 GPUDirect RDMA 的真实数据路径

普通跨节点 GPU 通信可能经历 GPU → 主机内存 → NIC；GPUDirect RDMA 的目标是让 NIC DMA 直接进入或离开 GPU memory，减少 CPU 内存 bounce buffer。是否能走这条路径，取决于 GPU/NIC 的 peer-memory 支持、KMD 与用户态库、IOMMU/PCIe 平台限制、注册权限以及具体拓扑。

```text
GPU memory
   │ peer-memory / DMA mapping
   ▼
NIC DMA engine ─ PCIe Root Port/Switch ─ NIC link ─ remote NIC
```

GPU 与 NIC 位于同一 PCIe Switch 或同一 NUMA/Root Complex，通常更有利于减少跨 socket 路径，但“更近”只是性能假设，仍需用拓扑和带宽测试验证。NVIDIA NCCL troubleshooting 文档明确指出，某些 Linux bare-metal 配置下启用 IOMMU 会使 PCIe P2P 不受支持；这类平台约束应记录为环境事实，不能从单次 NCCL 日志推断。

## 4. NUMA 是路径成本的一部分

NUMA 不只影响 CPU 线程和页面，也影响 GPU/NIC 所连接的 Root Complex、PCIe switch 和主机内存。CPU 线程、控制面内存、NIC 和 GPU 若分散在不同 NUMA 节点，可能引入跨 socket UPI/Infinity Fabric 路径，增加延迟并降低有效带宽。`nvidia-smi topo -m` 的 CPU affinity、`lspci` 父路径、`numactl -H` 和 `/sys/bus/pci/devices/<BDF>/numa_node` 应结合阅读。

```bash
nvidia-smi topo -m
nvidia-smi topo -p2p r
numactl -H
for bdf in 0000:04:00.0 0000:05:00.0; do
  echo "$bdf"
  cat /sys/bus/pci/devices/$bdf/numa_node 2>/dev/null
  readlink -f /sys/bus/pci/devices/$bdf
done
```

这些命令只读状态；GPU 型号、驱动版本和拓扑能力不同会导致输出字段不同。NUMA locality 是性能和选路的输入，不是“跨 NUMA 必然失败”的证明。

## 5. NCCL 如何使用拓扑

NCCL 是 topology-aware 的集合通信库，会综合 CUDA 报告的 GPU P2P 能力、NVLink/NVSwitch、PCIe 层次、CPU/NUMA 和网络设备信息，为 collective 选择 ring、tree、通信通道及传输后端。它的选择结果是软件策略，不能反向证明硬件一定存在某条物理链路。环境变量或自定义 topology 文件也可能改变选路，因此复现问题时应保存相关环境。

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET ./your_collective_test
```

日志中的 `P2P`, `SHM`, `NET/IB`, `NET/Socket`, `NVL` 等表示 NCCL 选择或尝试的传输层；它们是上层事实，仍应与 `nvidia-smi topo -m`、RDMA 设备和 BDF 路径互相核对。性能下降可能来自链路降速、P2P 被禁用、GPUDirect RDMA 未建立、错误回退到 host staging、IRQ/NUMA 不匹配或消息规模算法变化。

## 6. 用“路径假设”排障

| 现象 | 先验证的路径事实 | 再检查的软件层 |
| --- | --- | --- |
| GPU 间带宽远低于预期 | NVLink 状态、PCIe 上游路径、P2P capability | CUDA P2P、NCCL GRAPH/transport |
| NIC 可见但 RDMA 性能低 | NIC/GPU NUMA、PCIe Switch 共享关系、链路速率 | RDMA link、peer-memory、NCCL NET/IB |
| NCCL 退回 Socket/SHM | GPU/NIC 是否被 runtime 看见，P2P/IOMMU 条件 | 用户态库、环境变量、容器设备映射 |
| 只有跨 socket 的 GPU 对异常 | CPU affinity、`numa_node`、Root Complex 路径 | 进程绑定、内存策略、NCCL channel |

每个判断都应保留三类证据：硬件/firmware 视图（拓扑与链路）、内核视图（BDF、驱动、DMA/IOMMU）、用户态视图（CUDA、RDMA、NCCL 日志）。三者不一致时，优先定位“哪一层丢失了信息或能力”，不要直接调整 NCCL 参数掩盖底层问题。

## 来源

- [NVIDIA NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)：NCCL 的 topology-aware 通信定位。
- [NVIDIA GPUDirect RDMA Overview](https://docs.nvidia.com/cuda/gpudirect-rdma/)：GPU 与 NIC 直接 DMA 的前提和 PCIe 拓扑观察方法。
- [NVIDIA NCCL GPU Troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html)：GPU P2P、IOMMU 与故障排查边界。
- [NVIDIA DCGM Topology and NVLink](https://docs.nvidia.com/datacenter/dcgm/latest/learn/core-services/topology-and-links.html)：CPU、PCIe、NVLink 关系和拓扑查询边界。

下一篇进入综合排障案例：把未枚举、BDF 消失、链路退化、probe 失败、GPU/NIC 不可见和 NCCL 路径回退串成统一证据流程。
