---
tags:
  - AI
  - PCIe
  - Debug
---

# 07 · PCIe 链路、AER 与错误恢复：从可见设备到运行中稳定性

本文承接 [[02-AISystem/cluster-and-hardware/06-MSI与MSI-X：从数据完成到中断通知|MSI/MSI-X]]。前几篇回答了“设备怎样被发现、映射和传输数据”；本篇回答“链路怎样保持可用，以及发生错误后系统如何记录、隔离、复位和恢复”。因此，“BDF 仍在 `/sys`”只能证明设备对象尚在，不能证明链路、事务和驱动数据面仍然健康。

## 1. 链路状态与设备状态是两件事

PCIe 链路连接相邻端口（Root Port、Switch Port、Endpoint），每一端都维护协商结果和错误状态。常见观察量包括 negotiated speed（例如 16 GT/s）、negotiated width（例如 x16）、链路是否 active，以及是否发生 retrain。`lspci -vv` 展示的是配置空间中 PCI Express capability 的解码结果；它是某个时刻的快照，不是整个历史。

```text
Root Complex ─ Root Port ─ Switch Upstream ─ Switch Downstream ─ GPU/NIC
       每一跳都有独立的链路状态、能力上限和错误计数
```

链路宽度降到 x8、速率降档或反复 retrain，可能仍能枚举并绑定驱动，但会改变带宽、延迟和多 GPU/NIC 通信路径。反过来，链路参数正常也不能排除设备内部 firmware、驱动状态机或 DMA/IOMMU 问题。

## 2. AER 记录什么，不能证明什么

Advanced Error Reporting（AER）由 PCIe capability 提供更细的错误分类。Linux AER 驱动通常挂在支持 AER 的 Root Port 或 RCEC 上，收集并报告 correctable、non-fatal、fatal 等错误，再把需要恢复的事件交给 PCI 错误恢复框架。固件通过 ACPI `_OSC` 把控制权交给 OS 后，Linux 才应接管这些事件；固件和 Linux 同时处理会造成不可预测行为。

错误的观察位置很重要：Endpoint 可能是故障源，但 Root Port 或链路伙伴才真正“看到”错误，因此 `/sys/bus/pci/devices/<BDF>/aer_*` 的计数可能首先增长在上游端口。计数增加说明该对象报告或观察到了错误，不能单独证明该对象就是根因。

```text
correctable  → 硬件自行纠正，仍需关注是否持续增长
non-fatal    → 当前事务/功能受影响，层次通常仍可继续工作
fatal        → 子层次可能被冻结或不可访问，需要复位/恢复
```

## 3. Linux 错误恢复的核心状态机

Linux PCI error recovery 把驱动回调与平台动作分开。错误事件到达后，驱动先停止新 I/O、保存必要状态并在 `error_detected()` 报告能否恢复；随后平台可能重新启用 I/O、执行 link reset 或 slot reset。复位后，驱动在 `slot_reset()` 中重新初始化设备（通常需要恢复配置空间、重新下载设备 firmware、重建 DMA 和中断资源），最后在 `resume()` 恢复正常 I/O。若驱动报告永久失败，设备会被视为不可恢复。

```text
硬件检测错误
  → AER/PCI core 记录并通知
  → driver error_detected()
  → MMIO enable 或 link reset / slot reset
  → driver slot_reset()：恢复配置、BAR、DMA、IRQ、固件状态
  → driver resume()：重新开放数据面
```

这是“可恢复”的框架，不是每个 GPU/NIC 驱动都完整支持所有回调，也不是发生 fatal 后 BDF 必然保留。复位可能让设备回到类似上电初始状态；旧的 BAR 内容、MSI-X 表、队列和 DMA 映射不能假定仍有效，驱动必须按自身契约重建。

## 4. 面向排障的证据链

先用 `lspci -s <BDF> -vv` 对比链路能力与当前状态，再沿父链检查 Root Port 和 Switch Port；同时看 `dmesg -T` 中 AER、link down、completion timeout、unsupported request、surprise down 等关键词。若系统提供 AER 统计，可读取：

```bash
BDF=0000:04:00.0
lspci -s "$BDF" -vv
readlink -f "/sys/bus/pci/devices/$BDF"
for f in aer_dev_correctable aer_dev_nonfatal aer_dev_fatal; do
  test -r "/sys/bus/pci/devices/$BDF/$f" && cat "/sys/bus/pci/devices/$BDF/$f"
done
dmesg -T | rg -i 'pcie|aer|link down|completion timeout|surprise down|fatal|non-fatal'
```

上述命令只读系统状态；不同内核、固件和发行版不一定暴露全部 AER 文件。调试时至少保留故障前后两次快照，并记录 BDF、上游端口、链路速率/宽度、错误计数和时间戳。不要仅凭一次 `lspci` 输出判断“链路稳定”。

| 现象 | 优先观察 | 可能层次 |
| --- | --- | --- |
| BDF 启动后就不存在 | 上游端口、链路训练、固件日志、资源分配 | 板卡/链路/firmware/枚举 |
| BDF 存在但速率或宽度异常 | 端点与每级上游 Port 的 `LnkSta`、retrain 记录 | 信号完整性、插槽、拓扑配置 |
| 运行中出现 AER 且 I/O 停顿 | Root Port 与端点两侧计数、dmesg 时间线、驱动恢复回调 | 链路/事务/驱动恢复 |
| 复位后设备回到 `/sys` 但 GPU/NIC 不可用 | `driver` 链接、BAR、DMA/IOMMU、MSI-X、firmware 重新加载 | 驱动初始化与数据面 |

## 5. 与多 GPU 通信的连接

PCIe AER 只描述 PCIe 层观察到的错误；NCCL、RDMA 或 CUDA 报错是更上层的结果。一个 Root Port 下的 Switch 子树被隔离时，可能同时影响多张 GPU 和 NIC；单个 Endpoint 的复位也可能让其 CUDA context、RDMA queue pair 或 peer-memory 映射失效。因此应沿“故障 BDF → 上游端口 → 同一 Switch 子树 → GPU/NIC 通信路径”检查，而不是只看应用最后一行错误。

事实与推论要分开：AER 日志和计数是内核/硬件报告的事实；“某端口是根因”需要结合时间顺序、链路伙伴计数、插拔/降速复现和硬件替换验证；仅凭计数最高的对象做根因结论属于未经证实的推论。

## 来源

- [Linux PCI Error Recovery](https://docs.kernel.org/PCI/pci-error-recovery.html)：错误恢复步骤、驱动回调与复位后的重新初始化。
- [The PCI Express AER Driver Guide HOWTO](https://docs.kernel.org/next/PCI/pcieaer-howto.html)：AER 驱动职责及 ACPI `_OSC` 控制权边界。
- [Linux PCIe AER sysfs ABI](https://docs.kernel.org/7.1/admin-guide/abi-testing-files.html)：AER 统计文件及“链路伙伴可能报告错误”的说明。

下一篇进入 Linux device/driver/class 与 sysfs，把本篇的 BDF、端口和恢复对象映射到内核对象模型。
