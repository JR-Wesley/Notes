# MSI 与 MSI-X：从数据完成到中断通知

本篇属于 [[02-AISystem/cluster-and-hardware/00-Overview-GPU服务器：从物理硬件到Linux设备系统|GPU 服务器学习地图]] 第 4 阶段，文件顺序为 06，承接 [[02-AISystem/cluster-and-hardware/05-DMA与IOMMU：设备如何访问内存|DMA 与 IOMMU]]。DMA 解释设备如何搬运数据；本篇解释设备完成工作后，Linux 怎样获得通知、识别来源并推进队列。PCIe 资源、中断控制器、驱动队列和应用完成语义共同构成这条路径，因此“没有中断”与“应用没有拿到完成”不能混为一个问题。

范围以 PCIe GPU/NIC/NVMe 等高性能设备的 MSI/MSI-X 为主。以下 IRQ 号、队列数、CPU 和日志均为教学示意，未连接用户服务器，也没有修改 IRQ affinity、irqbalance、MSI 配置或设备寄存器。

## 1. 数据完成、设备通知和应用可见是三个事件

设想 NIC 的一个 receive queue 已借助 DMA 将数据写进主机 buffer。设备随后可能更新 completion queue 或状态位，再发出 MSI/MSI-X；CPU 响应这个 Linux IRQ 后，驱动读取完成状态、回收描述符并把工作交给网络栈或用户态接口。应用最终从 socket、RDMA completion queue 或其他 API 获得“完成”，还会受到调度、轮询和批处理影响。

```text
设备 DMA 写入 buffer / completion 记录
                  ↓
设备按队列状态发起 MSI 或 MSI-X message
                  ↓
PCIe / 中断基础设施投递为 Linux IRQ
                  ↓
CPU 上的 IRQ handler 确认来源、收割完成项
                  ↓
驱动后续处理、网络栈或用户态 completion
                  ↓
应用线程被唤醒或下一次轮询观察到完成
```

这不是每个驱动都逐项执行的固定时间线。设备可以合并多个完成再通知，驱动可以在一次 IRQ 中收割多个 queue，也可转入 NAPI/线程化处理；高性能路径还可能采用 polling。图的价值在于把“设备已有完成”“CPU 收到 IRQ”“驱动处理完成”“应用看见完成”拆开。任何一处延迟、屏蔽、队列停滞或软件同步问题，都可能表现为用户态超时。

中断是通知机制，不是数据搬运机制。MSI/MSI-X 通知 CPU 重新检查设备状态，通常不承载传输 payload；DMA 仍走上一节的设备到内存路径。反过来，完成记录已经可见也不表示应用一定立即运行，CPU 亲和性、软中断、线程调度和应用消费速度都会改变端到端时延。

## 2. INTx、MSI 与 MSI-X 的差别

传统 INTx 使用共享的、通常是电平触发的中断引脚语义。多个设备可能共享一条线，OS 收到中断后需要确认是哪一个设备请求服务。PCIe 从一开始就支持 message signaled interrupt（MSI）：设备发出一个特殊的 Memory Write message 来请求中断，避免依赖传统物理引脚路由。MSI-X 是对 MSI 的扩展，提供独立表项与更灵活的每向量配置，适合多队列设备。

| 模式 | 设备通知方式 | 工程含义 |
| --- | --- | --- |
| INTx | 传统中断引脚/电平语义，经平台路由 | 可共享，通常只适合作为兼容或回退路径；不能由 `INTx+` 单独判断功能正常 |
| MSI | 设备发出中断消息 | 不走传统引脚；设备可支持一个或多个向量，但配置方式比 MSI-X 受限 |
| MSI-X | 每个表项对应可单独配置、屏蔽的消息向量 | 多 queue NIC、NVMe 等可让多个 queue 使用不同向量，仍不保证“一 queue 恰好一个 IRQ” |

MSI/MSI-X 都是 PCIe 层的中断通知能力，设备支持 capability、平台/MSI domain 支持、内核配置、上游 bridge 条件和驱动请求方式必须同时满足。`lspci -vv` 显示 MSI/MSI-X capability 只证明硬件或功能声明了这种能力；`Enable+` 才是当前该 capability 已启用的相关证据。即便已启用，也要继续看实际 IRQ 计数与队列状态。[Linux MSI Driver Guide](https://docs.kernel.org/PCI/msi-howto.html)

驱动通常通过 `pci_alloc_irq_vectors()` 请求可接受范围的向量，并按能力优先尝试 MSI-X、MSI、INTx；实际分配数量可能小于期望值，少于最小值则失败。Linux IRQ number 是内核分配的标识，不能拿 PCIe BDF、MSI-X table index 或 x86 CPU vector 当作同一个编号。具体映射还经过 Linux irq domain，因此 `/proc/interrupts` 的一行是当前内核处理视图，不能反推出固定硬件编号。[Linux PCI Driver 文档](https://docs.kernel.org/PCI/pci.html)、[irq_domain 文档](https://docs.kernel.org/core-api/irq/irq-domain.html)

### 一条 MSI-X 记录涉及多种编号

以教学 NIC `0000:04:00.0` 为例，假设驱动为接收队列请求 MSI-X table index `3`，Linux 分配 IRQ `142`，并将其许可 CPU 集设为 `16-23`。这些值表达的是不同关系：

```text
BDF 0000:04:00.0  ── 标识 PCI function
MSI-X index 3     ── 该 function 内 MSI-X table 的条目
Linux IRQ 142     ── 内核 IRQ 子系统的当前编号
CPU 16-23         ── IRQ 142 的允许投递 CPU 集合
RX queue ?        ── 驱动/设备队列关系，需由该驱动实际定义
```

不能仅据 `142` 对应某行 `/proc/interrupts` 就声称“这是第 3 个 receive queue”。有的驱动把向量用于多个 queue，有的为管理事件保留专用向量，有的在不同版本或配置下重排 queue/vector；IRQ 号和 CPU affinity 也可随重载、CPU online/offline、驱动重置或服务策略变化。

## 3. 从设备消息到 Linux handler 的边界

设备需要按其协议保证 completion/state 的可见性，再写入 MSI/MSI-X message；驱动在 handler 中还要确认、acknowledge 或收割相应队列，避免遗漏或重复处理。具体寄存器顺序和内存屏障由设备驱动协议决定。MMIO 的 posted write、设备读写排序和 CPU 内存顺序不能按普通变量赋值推理；Linux 的 device IO API 提供对应访问与 ordering 语义。[Linux device IO 文档](https://docs.kernel.org/driver-api/device-io.html)

硬中断上下文应当短小，较重的包处理、批量收割或用户态通知可能在后续上下文执行。已有 [[01-ComputerScience/operating-system/IO|IO]] 笔记介绍 hardirq、softirq 和 `ksoftirqd` 的通用概念；本篇只补充其与 PCIe 完成队列的关系。不要把 Linux 里的 softirq 与“软件产生的 CPU 异常”混作同一分类，也不要从 CPU 有 IRQ 计数直接推论用户态操作已经完成。

驱动可以根据工作负载采用 interrupt moderation/coalescing：设备延迟一段时间或积累一定数量完成后再通知。它往往减少 IRQ 压力、提高吞吐，却可能增加低负载延迟。具体时间阈值、包数阈值和管理接口依设备驱动而异，不能在没有 baseline 的情况下通过任意调节参数解释或解决 GPU/NIC 通信超时。

## 4. IRQ affinity、NUMA 与队列亲和性

IRQ affinity 限制一个 IRQ 可以投递到哪些 logical CPU。它表达中断处理的 CPU 位置，不等于 application thread affinity，也不保证设备 DMA buffer 处于同一个 NUMA node。高性能服务常尝试让设备的 IRQ、处理线程与其主机局部性靠近，以减少跨 NUMA 访问；最终效果仍取决于 buffer placement、queue 绑定、RSS/NAPI、CPU 负载及应用模型。

`/proc/irq/<IRQ>/smp_affinity_list` 是允许 CPU 列表，`/proc/interrupts` 是每 CPU 已处理计数。前者不能说明 IRQ 实际只由某一 CPU 处理，后者是累积计数，也不能仅凭一次快照衡量负载。某些中断控制器不支持 affinity 时，写入设置不会按预期改变；内核文档还明确禁止把 IRQ 的允许 CPU 集全部置空。[Linux SMP IRQ affinity](https://docs.kernel.org/core-api/irq/irq-affinity.html)

```text
NIC PCI function ── NUMA node 1（设备局部性线索）
        │
        ├─ IRQ 142 ── allowed CPUs 16-23（IRQ 投递许可）
        ├─ RX queue / NAPI（驱动工作划分）
        └─ host buffer pages（实际内存分配位置）

application thread ── CPU affinity / cpuset（调度约束）
```

这几条关系要分开测量和记录。将 IRQ 强行绑到“看上去同 NUMA”的 CPU 不一定改进端到端通信，特别是在该 CPU 已繁忙、队列处理与应用不在同一侧、或设备存在专门服务策略时。先从当前允许集、实际计数、NUMA 关联和 workload 时间窗口建立事实，再评估是否需要受控实验。

## 5. 只读观察：先把设备、向量和计数对齐

以下命令只读取当前 Linux 视图，未在用户服务器运行。`/proc/interrupts` 标签是驱动和内核输出，不保证包含 BDF；先由 `lspci`、driver 和 `/sys` 建立设备身份，再用驱动名称、接口名称或 IRQ 号做关联。容器可能看不到宿主机完整中断和 PCI 信息。

```bash
# 将 BDF 替换为当前存在的实际设备，例如 NIC 或 GPU
bdf=0000:04:00.0
dev_dir="/sys/bus/pci/devices/$bdf"

if [ -d "$dev_dir" ]; then
    lspci -D -s "$bdf" -vv
    lspci -D -s "$bdf" -nnk
    readlink -f "$dev_dir"
else
    printf '%s\n' '当前 BDF 不存在；先回到枚举、拓扑与 driver 状态检查'
fi

cat /proc/interrupts
```

在 `lspci -vv` 中阅读 `MSI:`、`MSI-X:` capability 与 `Enable` 状态；不要对全配置空间做扫描。随后从真实 `/proc/interrupts` 行中选择一个 IRQ 号，再只读查看它的允许 CPU 集：

```bash
# 用 /proc/interrupts 中真实存在的 IRQ 替换
irq=142
irq_dir="/proc/irq/$irq"

if [ -d "$irq_dir" ]; then
    cat "$irq_dir/smp_affinity_list"
    cat "$irq_dir/smp_affinity"
    grep -E "^[[:space:]]*$irq:" /proc/interrupts
else
    printf '%s\n' '该 IRQ 当前不存在或不可见；不要使用教学编号修改系统设置'
fi
```

为了比较计数增长，取两次同格式样本并记录 workload、时间与 CPU online 状态；不要把一次绝对计数当作速率。下面只是定位某个 IRQ 的方法，不执行持续负载。

```bash
irq=142
grep -E "^[[:space:]]*$irq:" /proc/interrupts
date -Is
# 在明确、受控的 workload 窗口后再次读取相同两项
```

若设备支持多个队列，驱动可能公开额外的 ethtool、RDMA 或 debugfs 信息，但路径和权限高度依设备及驱动变化。本篇不假定存在特定厂商工具，也不在没有平台基线时写 `/proc/irq/*/smp_affinity_list`、设置 irqbalance 或修改 coalescing。

## 6. 用症状缩小范围

| 观察组合 | 先检查的边界 | 不能直接得出的结论 |
| --- | --- | --- |
| DMA buffer 内容已更新，应用迟迟无完成 | completion queue、设备通知、IRQ/NAPI、应用消费路径 | DMA 成功不自动表示应用会立即收到完成 |
| `MSI-X: Enable+`，IRQ 计数不增长 | 是否有实际工作、队列选择、设备状态和 driver 日志 | capability 启用不等于每个向量持续产生中断 |
| IRQ 计数增长，吞吐或应用进度不动 | handler 后续处理、queue backlog、softirq/线程、用户态消费 | 有 IRQ 不等于完成项被正确消费 |
| 单个 IRQ 集中在一颗繁忙 CPU | 当前 affinity、队列映射、irqbalance 和负载 | 不能只因“未平均分布”就认定故障 |
| MSI/MSI-X 未启用或向量不足 | 平台/bridge/内核支持、驱动申请和回退路径 | 不应立即全局使用 `pci=nomsi` 或改 bridge 配置 |
| 重置或恢复后通信异常 | MSI state、driver 重新初始化、队列与 IRQ 重建 | 不能按旧 IRQ 号或旧 affinity 假设状态仍在 |
| GPU/NIC 通信超时且 IRQ 正常 | DMA/IOMMU、queue、远端网络、fabric、软件同步 | IRQ 正常不能排除数据路径或协议问题 |

MSI 也可能因平台全局限制、上游 bridge 或单设备问题未被启用。Linux 文档建议先检查 dmesg、驱动能力及完整上游路径；对 bridge 的 `msi_bus` 等写入会影响其下游设备，不能把它当成普通排障命令。[Linux MSI Driver Guide](https://docs.kernel.org/PCI/msi-howto.html)

## 来源、范围与下一篇

本轮于 2026-09-13 使用 Linux MSI Driver Guide、PCI Driver、irq affinity、irq_domain 和 device IO 官方文档。它们支持 MSI/MSI-X、向量分配、Linux IRQ 映射、允许 CPU 集与 MMIO/ordering 的机制边界。已有 IO 笔记仅作为通用中断概念的关联入口，未改写其正文。

本文未检查特定 NIC/GPU 驱动源码、设备 datasheet、IRQ routing 表或用户服务器输出。教学 IRQ/CPU/队列关系不是实机映射，所有命令未运行。实际平台的 MSI 支持、interrupt moderation 与工作队列结构仍需结合目标设备、内核和驱动版本核对。

下一篇进入 `07 · PCIe 链路、AER 与错误恢复`：从速率/宽度、LTSSM 与错误上报理解为什么设备仍在 sysfs 中却传输异常，或为什么运行期错误可能使设备离开内核视图。该篇已独立成文；再下一篇进入 Linux device/driver/class 与 sysfs。
