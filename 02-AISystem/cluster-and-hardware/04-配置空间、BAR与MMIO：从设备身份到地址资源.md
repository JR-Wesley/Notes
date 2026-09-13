# 配置空间、BAR 与 MMIO：从设备身份到地址资源

本篇是 [[02-AISystem/cluster-and-hardware/00-Overview-GPU服务器：从物理硬件到Linux设备系统|GPU 服务器学习地图]] 第 4 阶段的第一篇，承接 [[02-AISystem/cluster-and-hardware/03-从上电到设备枚举：Firmware与Linux的职责边界|从上电到设备枚举]]。核心问题是：Linux 已有某个 GPU/NIC 的 BDF，为什么驱动仍可能无法使用它？本篇从设备身份走到地址资源，解释配置空间、BAR、MMIO 和上游资源窗口。DMA/IOMMU、中断、链路与错误恢复后续分别展开，不把本篇成文视为第 4 阶段全部完成。

参考场景仍是 x86 多 GPU 服务器；以下 BDF、BAR 大小、寄存器偏移和地址范围全部为教学构造，不是 H100 实际寄存器表、DGX 固定配置或用户服务器输出。

## 1. 找到设备，不等于已经能访问它的功能

BDF 标识一个 PCI function；配置空间让系统读取其身份、类别、能力和配置；BAR 则是配置空间中用于描述设备地址资源的寄存器。设备真正的控制寄存器、队列通知接口或可映射内存通常位于 BAR 所描述的区域，而不是把全部业务寄存器直接放在 PCI 配置空间里。

因此存在两种不同访问：系统通过 BDF 和配置偏移访问某 function 的配置空间；驱动通过映射后的地址访问设备 MMIO 区域。即使 ECAM 将配置访问呈现为 CPU 内存地址访问，它最终仍是配置空间访问，不等同于对 BAR 区域的普通 PCIe Memory Request。ECAM 的平台基址等信息由 ACPI MCFG 等机制提供，Host Bridge 资源窗口则另有描述。[Linux 6.14 PCI/ACPI 文档](https://docs.kernel.org/6.14/PCI/acpi-info.html)

```text
配置路径：BDF + 配置偏移 → function 配置空间 → BAR 中记录的地址与类型

功能路径：驱动 MMIO 映射 + 寄存器偏移
                       ↓
             主机地址域 → PCI 地址域
                       ↓
             上游 Bridge 的地址转发
                       ↓
             endpoint BAR 地址译码
                       ↓
               设备控制寄存器/窗口
```

一张卡可以成功返回 vendor/device ID，却没有获得所需的 BAR 地址范围；也可能资源已分配而 Memory Space Enable 尚未打开，或驱动映射失败。`lspci` 有记录只说明设备已经进入可观察的 PCI 视图，不能一步推到“寄存器访问、DMA 与 GPU 计算都正常”。

## 2. 配置空间中的 BAR 实际描述什么

PCIe function 的配置空间最多为 4 KiB，前 256 字节延续传统 PCI 配置空间布局，扩展能力可位于后续区域；实际可读范围还取决于设备和访问机制。标准 header 提供身份、Command/Status、header type 等字段。常见 endpoint 使用 Type 0 header，其中有六个 32-bit BAR 寄存器槽位；Type 1 Bridge header 有不同布局，不能把 endpoint 的六 BAR 模型直接套到 Bridge 上。[Linux v6.12 `pci_regs.h`](https://raw.githubusercontent.com/torvalds/linux/v6.12/include/uapi/linux/pci_regs.h)

BAR 可以描述 I/O port 或 memory 资源，本篇重点讨论现代 GPU/NIC 常见的 memory BAR。低位包含类型、是否 prefetchable 等属性，其余地址位记录分配后的基址；一个 64-bit memory BAR 占用相邻两个 32-bit 槽位，后一槽提供高位地址，不能再当成独立 BAR。例如 BAR0 为 64-bit 时，下一个独立资源可能从 BAR2 开始，编号不连续不一定是缺失。

| 容易混淆的属性 | 实际含义 | 不能推出的结论 |
| --- | --- | --- |
| 64-bit memory BAR | 可表达 64-bit 形式的基址，使用两个配置槽位 | 不等于容量是 64 bit，也不保证被分配在 4 GiB 以上 |
| BAR size | 设备需要的地址 aperture 大小 | 不等于整张 GPU 的 HBM 容量 |
| Prefetchable | 该区域满足相应预取/访问合并语义条件 | 不自动表示按普通 RAM 的 write-back cache 属性映射 |
| BAR base | 分配给该 aperture 的 PCI 地址基址 | 不是 BDF、CUDA 指针、设备序列号或 DMA buffer 地址 |
| Memory Space Enable | 允许 function 对相应 memory 地址访问进行译码 | 不等于 Bus Master Enable，也不证明设备完整初始化成功 |

BAR 大小不是简单从当前基址数值读出来的。传统大小探测利用设备对 BAR 可写地址位的响应，软件在受控条件下临时探测、读回掩码并恢复。Linux v6.12 `drivers/pci/probe.c` 的 `__pci_read_base()` 展示了读写 BAR、处理 64-bit 地址和恢复配置的局部过程，`pci_size()` 计算相应范围。这是内核管理设备时的机制说明，不是让用户对运行中的卡手工写入全 1 探测大小。[Linux v6.12 PCI 探测源码](https://raw.githubusercontent.com/torvalds/linux/v6.12/drivers/pci/probe.c)

常规 BAR aperture 通常按其大小对齐，地址分配不仅要有足够总量，还要满足连续范围、类型和对齐约束。Resizable BAR 允许在设备支持的大小集合中选择 aperture；“支持 64-bit 地址”“允许在 4 GiB 以上布局”和“支持改变 BAR 大小”是三个不同条件。实际 BIOS 菜单、设备支持和驱动策略需要按平台核对，不能仅凭名称相似一起开关。

## 3. 一个地址示例：从 CPU 映射到 endpoint

沿用教学 GPU `0000:03:00.0`，假设其 BAR0 是 64-bit、non-prefetchable memory BAR，大小为 16 MiB。为了只讲地址层次，本例先假设主机地址与 PCI bus address 数值相同，分配如下：

```text
BAR0 base       = 0x80000000
BAR0 size       = 0x01000000  = 16 MiB
BAR0 last byte  = 0x80ffffff

虚构寄存器 offset = 0x00001000
相应总线地址       = 0x80001000
```

末地址按包含端点的区间计算：`end = base + size - 1`；大小则为 `end - start + 1`。本例 `0x80000000` 按 16 MiB 对齐，`0x1000` 位于窗口内。它只证明地址算术一致，不说明该偏移在 H100 上是合法寄存器，更不能直接执行读写。

驱动不会把 `0x80001000` 强转成普通 C 指针就跨平台访问设备。它需要使用内核提供的资源信息和映射 API，得到适合 MMIO 的映射，再使用对应 IO accessor。设备 IO 的访问顺序、字节序、posted write 和可能的副作用由硬件/架构与 API 约束，不能按普通 DRAM 数组理解。[Linux Bus-Independent Device Accesses](https://docs.kernel.org/driver-api/device-io.html)

若平台进行 host-to-PCI 地址转换，CPU 资源地址与 BAR 中的总线地址可能不同。比如另一个纯教学平台将 CPU 区间 `0x480000000–0x480ffffff` 转换到本例 PCI 区间 `0x80000000–0x80ffffff`，则驱动映射的是内核报告的主机资源地址。对这两个域做数值比较之前，必须先确认转换关系。Linux PCI 驱动文档明确要求使用 PCI resource API，而不是自己把配置空间 BAR 值当作 CPU 地址。[Linux PCI Driver 文档](https://docs.kernel.org/PCI/pci.html)

这里还有第四种容易混入的地址：设备 DMA 使用的 DMA address/IOVA。它由 DMA API 和可能存在的 IOMMU 映射建立，用于设备访问内存，不能因同样是十六进制就与 CPU virtual address、CPU resource address 或 BAR base 互换。[Linux DMA mapping Guide](https://docs.kernel.org/core-api/dma-api-howto.html) 下一篇专门解释这一反向的数据访问路径。

## 4. Endpoint 有 BAR，还需要整条上游路径容纳它

前一篇的 Bridge Primary/Secondary/Subordinate 字段描述 bus number 范围；Bridge memory window 描述可向下游转发的地址范围。这两类资源不是同一件事。一个分支可能拥有足够总线号，可以枚举 endpoint，却没有合适的 memory window 容纳 endpoint 的 BAR。

继续采用主机地址与 PCI 地址数值相同的教学假设，为 GPU BAR0 设置以下嵌套关系。只展示一个 memory 资源类别，省略其他设备、prefetchable 窗口及预留：

```text
Host Bridge 可转发的 memory 范围       0x80000000–0x8fffffff
└─ Root Port 0000:00:01.0 窗口        0x80000000–0x83ffffff
   └─ Switch upstream 01:00.0 窗口   0x80000000–0x83ffffff
      └─ downstream 02:00.0 窗口     0x80000000–0x80ffffff
         └─ GPU 03:00.0 BAR0         0x80000000–0x80ffffff
```

在这个常规层次中，BAR 范围需要被沿途相应窗口覆盖；兄弟 endpoint 的独立资源通常要避免地址重叠。Bridge window 是转发 aperture，不是 Switch 里新安装了这么多 DRAM。`/proc/iomem` 中父窗口包含子设备范围也是正常层次关系，不能看到区间包含就认定资源冲突。

若 GPU 需要 16 MiB 对齐的 aperture，而相关上游只有零散小空间或只剩 8 MiB 合适范围，那么“总机器内存很多”无助于这个分配。资源分配针对 IO 地址空间，不是在为 GPU 从系统 RAM 划出同容量的数据 buffer。Host Bridge 的 `_CRS`、Bridge 窗口、BAR 类型和对齐共同约束结果。BAR 的地址宽度也不自动突破每一级 Bridge 对窗口的支持范围。

在多 GPU 大 BAR 系统上，地址空间需求可能很大，但不能把“Above 4G Decoding”理解成增大 HBM，或把“Resizable BAR”理解成增加显存容量。NVIDIA GPUDirect RDMA 文档讨论了大 BAR 对平台配置的要求及 GPU BAR1 aperture 的用途；具体设备的 BAR 编号与大小仍需实测和对应代际文档确认。[NVIDIA GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html)

## 5. Linux 中的几个文件分别是什么

设备目录中的 `config`、`resource`、`resource0` 名字接近，但含义不同。`config` 是配置空间接口；`resource` 是资源范围与 flags 的文本描述；`resourceN` 对应可存在的资源访问接口，可能支持 mmap。读取资源描述与访问设备内容是两种动作。[Linux PCI sysfs 文档](https://docs.kernel.org/PCI/sysfs-pci.html)

| 观察位置 | 用来回答什么 | 阅读边界 |
| --- | --- | --- |
| `lspci -D -s <BDF> -vv` | BAR 区域、大小、属性、Command 状态、能力等 | 权限不足时信息可能不完整，显示格式也随版本变化 |
| `/sys/bus/pci/devices/<BDF>/resource` | 内核登记的起始地址、末地址和 flags | 不应把 flags 当大小；正常有效区间大小为 `end-start+1` |
| `/sys/bus/pci/devices/<BDF>/resourceN` | 具体资源的访问接口 | 不是普通文本文件，不使用 `cat`/`dd` 当成无副作用查看 |
| `/sys/bus/pci/devices/<BDF>/config` | function 的二进制配置空间 | 本篇不 dump 全空间、不写入配置寄存器 |
| `/proc/iomem` | 系统资源地址范围及占用层次 | 可能因权限显示遮蔽地址；父子包含不等于冲突，不能简单逐行求和 |

对普通 endpoint，`resource` 前六项对应标准 BAR 槽位，但 64-bit BAR 消耗两个配置槽位，其高半部分不意味着另一个独立 aperture；Bridge 与 ROM 等资源还有额外语义。不要因为某行全零就计算出“1 字节 BAR”。Linux v6.12 `drivers/pci/pci-sysfs.c` 的 `resource_show()` 输出 start/end/flags，说明这里是内核资源对象的展示，不是原始 BAR 寄存器转储。[Linux v6.12 PCI sysfs 源码](https://raw.githubusercontent.com/torvalds/linux/v6.12/drivers/pci/pci-sysfs.c)

驱动侧还要区分分配、申请占用、映射和启用。系统可能已经为 BAR 分配地址，驱动随后仍需申请资源、建立 MMIO 映射并初始化功能。`lspci` 的 `Mem+` 主要表达 memory decode enable 状态，`BusMaster+` 主要表达总线主控能力已启用；都不是整张卡的健康证明。DMA 成功还需要其他条件，不能由 `BusMaster+` 单独推出。

## 6. 只读核对与故障判断

先选当前存在的实机 BDF，查看资源描述与上游路径。以下命令未在用户服务器运行，要求 Linux 和 pciutils；只读取描述与日志，不直接读写 BAR 映射区。pciutils 手册也说明某些设备对全配置空间读取可能有问题，因此本练习不使用 `-xxx/-xxxx` 扫描整块配置空间。[pciutils lspci 手册](https://raw.githubusercontent.com/pciutils/pciutils/master/lspci.man)

```bash
# 替换为本机实际 BDF
bdf=0000:03:00.0
dev_dir="/sys/bus/pci/devices/$bdf"
if [ -d "$dev_dir" ]; then
    readlink -f "$dev_dir"
    lspci -D -s "$bdf" -vv
    cat "$dev_dir/resource"
else
    printf '%s\n' '当前 BDF 不存在，先回到设备身份与枚举检查'
fi

# 查看资源树；普通用户可能看到地址被遮蔽
cat /proc/iomem
```

再依据父路径选择真实上游端口；不能直接照用教学地址。对比 endpoint 的 `Region` 与 Bridge 的 memory/prefetchable window 时，要确认资源类型及地址域相同。初学阶段先人工读完整目标输出，不编写依赖空格格式的自动结论脚本。

```bash
# 替换为刚才确认的直接父 Bridge
parent_bdf=0000:02:00.0
lspci -D -s "$parent_bdf" -vv

# 检索只是定位入口，随后阅读该时段完整上下文
journalctl -k -b --no-pager | grep -F -- "$bdf"
journalctl -k -b --no-pager | grep -Ei 'BAR|no space|failed to assign|resource'
```

`no space`、`failed to assign` 等词只有放在所属设备、资源类型与时间中才有意义，部分启动信息还可能随后被重分配或恢复。不能从一次过滤结果直接决定改 BIOS。先确认最终资源状态、错误是否属于目标设备，以及失败发生在分配、驱动占用还是功能初始化阶段。

| 现象 | 优先区分什么 | 下一步证据 |
| --- | --- | --- |
| BDF 存在但 BAR 分配报错 | 身份发现成功，资源是否最终可用未知 | BAR size/type、沿途窗口、同次启动后续日志 |
| 64-bit BAR 分配在 4 GiB 以下 | 这本身可以合法 | 是否满足大小、对齐和窗口，不按地址低就判错 |
| `resource` 地址和原始 BAR 数值不同 | 地址域转换、属性位或展示方式 | 内核资源信息、平台转换；先别直接重写 BAR |
| 内核有资源范围，驱动申请失败 | 资源分配与驱动占用不同 | 占用者、冲突、驱动错误位置与时间 |
| `Mem-` 或未映射 | 生命周期、资源与驱动初始化状态 | 当前驱动绑定及完整日志，不手工切 Command 位 |
| `Mem+` 但初始化超时 | decode enable 不等于设备协议正常 | firmware/设备状态、寄存器访问错误及驱动上下文 |
| GPU HBM 很大，BAR aperture 较小 | 物理显存与主机窗口不是同一容量 | GPU 型号与 BAR 用途，不能认定显存被系统吃掉 |

本篇练习的有效产物是“同一 BDF 的 BAR 类型/范围、父窗口、当前驱动状态与相关日志”的对应记录。未分配、未实现、权限不足和地址被遮蔽分别标注，不能都填成 0 或解释成硬件不存在。

## 来源与下一篇

本轮于 2026-09-13 查阅 Linux PCI sysfs、PCI Driver、device IO、DMA mapping、Linux 6.14 PCI/ACPI 文档；对 Linux v6.12 的 `pci_regs.h`、`drivers/pci/probe.c` 和 `drivers/pci/pci-sysfs.c` 仅核查相应字段与局部函数。pciutils 滚动手册用于命令语义，NVIDIA GPUDirect RDMA 文档用于 GPU BAR aperture 与平台要求。正文链接说明实际来源，不代表用户运行这些版本，也不声称通读 PCI-SIG 完整规范。

教学地址计算和窗口包含关系已作一致性检查；没有设备寄存器读写、BAR 探测写入、BIOS 修改或实机实验。实际平台转换、GPU BAR 布局、版本及故障日志仍待用户服务器信息确定。

下一篇见 [[02-AISystem/cluster-and-hardware/05-DMA与IOMMU：设备如何访问内存|DMA 与 IOMMU：设备如何访问内存]]，继续第 4 阶段，区分 CPU 虚拟地址、物理地址、DMA address/IOVA 和 GPU 内存，并解释映射生命周期。本篇说明 CPU 到设备资源的访问，下一篇补齐设备到内存的数据路径，再连接中断及链路错误。
