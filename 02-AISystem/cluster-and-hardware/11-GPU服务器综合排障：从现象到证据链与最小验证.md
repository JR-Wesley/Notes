---
tags:
  - AI
  - GPU
  - PCIe
  - Debug
---

# 11 · GPU 服务器综合排障：从现象到证据链与最小验证

本文把 01—10 篇连接成一套排障方法。核心原则是先定位现象所在层，再沿上下游收集证据；不要用某个上层工具的结果替代硬件、PCIe、内核或用户态的直接证据。每次记录都应包含 BDF、时间戳、主机/容器边界、驱动与 firmware 版本，以及正常节点的对照结果。

## 1. 统一证据链

```text
物理供电/布线/管理事件
  → firmware、链路训练与 PCIe 枚举
  → BDF、配置空间、BAR、PCIe 父路径
  → DMA/IOMMU、MSI-X、AER、错误恢复
  → device/driver/probe 与 sysfs
  → KMD、/dev、netdev、RDMA uverbs
  → CUDA/NVML、libibverbs、NCCL
  → 业务通信性能与错误
```

“不存在”必须带限定词：BMC 看不到、`lspci` 看不到、`/sys` 没有、驱动未绑定、设备节点缺失、CUDA 不可见或 NCCL 未选中，分别对应不同层。上层故障可以由下层异常引起，也可能只是权限、容器映射、版本或配置问题。

## 2. 最小只读快照

```bash
date -Is
uname -a
cat /etc/os-release
lspci -D -nn
lspci -D -t
lspci -D -nnk
lsmod
dmesg -T | tail -n 300
numactl -H
nvidia-smi -L 2>/dev/null
nvidia-smi topo -m 2>/dev/null
ip -br link
rdma link 2>/dev/null
```

针对一个 BDF，再保存：

```bash
BDF=0000:04:00.0
lspci -s "$BDF" -vv
readlink -f /sys/bus/pci/devices/$BDF
ls -l /sys/bus/pci/devices/$BDF/driver 2>/dev/null
cat /sys/bus/pci/devices/$BDF/{vendor,device,class,enable,numa_node} 2>/dev/null
for f in aer_dev_correctable aer_dev_nonfatal aer_dev_fatal; do
  test -r /sys/bus/pci/devices/$BDF/$f && cat /sys/bus/pci/devices/$BDF/$f
done
```

这些命令只读状态，不保证所有发行版都有相同文件。不要在尚未保存证据时执行 `remove`、`unbind`、改变 IOMMU/ACS 或重置设备；这些动作会改变现场。

## 3. 案例一：设备启动后没有枚举

现象：BMC 或机箱资料显示插有 GPU/NIC，但 `lspci`、`/sys/bus/pci/devices` 都没有对应 function。

事实边界：此时尚不能说“驱动坏了”，因为 Linux 尚未拥有一个可匹配的 PCI device。按顺序检查电源/插槽与 BMC 事件、firmware 的 PCIe 插槽和资源设置、Root Port 的链路状态、Switch 上游/下游端口，以及启动日志中的 PCI 扫描和资源分配。

```text
插槽/供电/复位
  → Root Port 链路训练
  → Switch 下游端口
  → 配置空间可读性
  → Linux bus scan 创建 BDF
```

若上游端口存在但下游设备没有，优先假设链路、复位、插槽或设备 firmware 问题；若 BDF 存在但没有 driver，才转入 probe 和模块匹配。若重启后偶发出现，记录每次启动的 `lspci -D -t`、`dmesg` 和 BMC 事件，避免把一次成功启动当作稳定性证明。

## 4. 案例二：BDF 运行中消失或 `/sys` 目录被删除

现象：设备曾经可用，运行中出现 AER、link down、surprise down 或 I/O timeout，随后 BDF 或设备节点消失。

先确认是 PCI 层对象消失，还是用户态句柄失效：检查 `/sys`、`lspci`、Root Port AER 计数和 dmesg 时间线，再看 GPU/NIC 驱动的恢复日志。Root Port 可能比 Endpoint 更早增加 AER 计数；计数最高者不自动等于根因。

若发生 DPC、link reset 或 slot reset，同一 Switch 子树中的多个设备可能同时受影响。恢复后要重新确认 BAR、DMA/IOMMU、MSI-X、driver symlink、`/dev`/netdev 和用户态上下文；旧进程持有的 fd 可能收到 `ENODEV`，不能继续当作有效设备使用。

## 5. 案例三：BDF 存在但链路降速或降宽

现象：`lspci -vv` 显示 Endpoint 仍在，但 `LnkSta` 低于 `LnkCap`，或 GPU/NIC 带宽异常。

沿 BDF 的真实 sysfs 父路径逐级读取 Endpoint、Switch Port 和 Root Port 的链路能力/状态。比较同型号正常节点，记录速率、宽度、retrain、AER 和负载状态。降速可能来自信号完整性、插槽、线缆/转接、固件策略、电源状态或训练失败后的降档；不要仅通过调整 NCCL 参数掩盖它。

## 6. 案例四：BDF 存在但 driver probe 失败

现象：`lspci -nnk` 能看到设备和可用模块，但没有 `Kernel driver in use`，或 dmesg 显示 probe、firmware、BAR、DMA、IRQ、timeout 错误。

把问题拆成匹配和初始化：模块是否加载、ID 表是否匹配、设备是否被 blacklist/override；随后检查 probe 是否在 BAR 映射、固件下载、IOMMU/DMA、MSI-X 或设备自检处失败。`/sys/bus/pci/devices/<BDF>` 仍存在说明 PCI device 尚在，不代表 probe 已完成。

## 7. 案例五：GPU/NIC 在宿主机可见，应用或容器不可见

先在宿主机验证 BDF、driver symlink、`/dev`、netdev/RDMA，再在容器内比较设备节点、权限、库版本和环境变量。CUDA ordinal、NCCL rank、Linux BDF 和 `/dev` 编号不是同一命名空间；容器的 `NVIDIA_VISIBLE_DEVICES`、cgroup 设备规则或 Kubernetes device plugin 可能只暴露子集。

```text
宿主机 PCI/KMD 正常
  → 设备节点与权限
  → 容器设备映射
  → 用户态库/ABI
  → CUDA、RDMA、NCCL 枚举
```

“`nvidia-smi` 能看到”只证明该工具所在环境能通过其用户态栈初始化设备；它不证明目标进程具有相同的节点、权限、库路径或通信拓扑。

## 8. 案例六：NCCL 可运行但路径回退或性能异常

先保存 `nvidia-smi topo -m`、P2P 能力、NUMA affinity、NIC/RDMA 状态和 NCCL `INIT,GRAPH,NET` 日志。日志显示 `SHM`、`Socket`、`NET/IB` 或 P2P/NVL 是软件选择事实；再用 PCIe 父路径、NVLink 状态和 GPUDirect RDMA 条件验证物理可能性。

常见分支包括：NVLink 未建立而回退 PCIe；GPU/NIC P2P 因平台 IOMMU 或驱动条件不可用；NIC 与 GPU 跨 NUMA；容器缺失 RDMA 或 peer-memory 设备；链路降宽导致有效带宽下降。每个分支都应先做最小对照测试，再改变一个变量。

## 9. 记录格式：事实、假设、验证

建议每个故障条目保持如下结构：

| 项目 | 内容 |
| --- | --- |
| 现象 | 哪个命令、哪个进程、哪个 BDF、何时失败 |
| 已知事实 | 原始输出、日志时间线、版本和拓扑快照 |
| 当前假设 | 明确属于链路、firmware、内核、驱动、权限或通信选路的哪一层 |
| 最小验证 | 一条只读命令、一个对照节点或一次受控复现 |
| 结果与下一步 | 假设被支持/排除，继续向上游或下游收敛 |

个人理解和推论应明确标注；没有实机输出时，只能给出观察方法和条件化判断，不能把教学 BDF、DGX 拓扑或典型路径当作用户服务器事实。

## 来源

- [Linux PCI Error Recovery](https://docs.kernel.org/PCI/pci-error-recovery.html)：PCI 错误事件与驱动恢复状态机。
- [Linux PCIe AER Driver Guide](https://docs.kernel.org/next/PCI/pcieaer-howto.html)：AER 报告、控制权和错误处理。
- [Linux Driver Binding](https://docs.kernel.org/driver-api/driver-model/binding.html)：设备匹配、probe 与绑定。
- [NVIDIA NCCL GPU Troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html)：P2P、IOMMU 与 GPU 通信排障。
- [NVIDIA GPUDirect RDMA Overview](https://docs.nvidia.com/cuda/gpudirect-rdma/)：GPU/NIC 拓扑与直接 DMA 的前提。

至此主线知识地图和 01—11 篇基础正文已建立；后续工作应以用户提供的真实服务器快照为输入，逐项完成实机验证和案例补充。
