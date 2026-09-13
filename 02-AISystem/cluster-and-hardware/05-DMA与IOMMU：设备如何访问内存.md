# DMA 与 IOMMU：设备如何访问内存

本篇属于 [[02-AISystem/cluster-and-hardware/00-Overview-GPU服务器：从物理硬件到Linux设备系统|GPU 服务器学习地图]] 第 4 阶段，文件顺序为 05，承接 [[02-AISystem/cluster-and-hardware/04-配置空间、BAR与MMIO：从设备身份到地址资源|配置空间、BAR 与 MMIO]]。上一篇说明 CPU 如何访问设备资源，本篇解释设备如何访问内存，以及驱动如何为这次访问建立地址、权限和生命周期约束。本文与服务器系列集中保存在 cluster-and-hardware，关联 [[01-ComputerScience/operating-system/虚拟内存|虚拟内存]]、[[01-ComputerScience/operating-system/IO|IO]] 与后续 [[02-AISystem/Distributed/RDMA/RDMA|RDMA]]，不复制这些既有基础专题。

范围以主机 DRAM、PCIe GPU/NIC 和普通 Linux DMA API 模型为主。地址和场景均为教学构造，没有实机映射表、IOMMU fault 日志或性能结果。GPU 内部页表、SVA、ODP 等只说明边界，不用它们替代基础模型。

## 1. DMA 传输仍需要 CPU 和驱动建立条件

DMA（Direct Memory Access）让设备或 DMA engine 搬运数据，CPU 不必逐字节执行搬运指令。现代 PCIe NIC、NVMe 和 GPU 可以具备自身发起内存访问的能力，因此 DMA 不要求主板上另有一颗统一的独立 DMAC。已有 IO 笔记中的独立控制器例子适用于某些体系，不能作为现代服务器的唯一结构。

典型过程是驱动准备 buffer 和描述符，建立设备可用的 DMA 映射，再通知设备开始处理；设备读取描述符、执行传输并更新完成状态，CPU 或其他软件随后检查结果。启动工作可能通过 BAR 中的 doorbell 等 MMIO 接口完成，但描述符引用的数据地址是 DMA 地址，不是那个 doorbell 的 BAR 地址。完成通知可以用中断，也可以轮询，DMA 与中断不是同一动作。

“不由 CPU 搬运每个字节”也不等于绕过 OS、缓存一致性、权限或所有数据复制。驱动仍要管理内存、设备队列和错误恢复；有些平台路径还会使用 bounce buffer。理解 DMA 的关键不是背下“零拷贝”，而是确定谁发起访问、使用什么地址、映射何时有效，以及完成之前谁能修改或释放 buffer。

## 2. 同一块内存有不同观察者的地址

CPU 执行 load/store 时使用虚拟地址，经 CPU 页表和 MMU 转换为物理地址。普通 DMA 请求不通过当前进程的 CPU 页表；驱动应把 DMA API 返回的 `dma_addr_t` 交给设备。在存在 DMA 重映射的系统中，这个设备地址通常处于 IOVA（I/O Virtual Address）空间，由 IOMMU 转换；没有此类重映射时也不能跨平台假定 DMA 地址必然等于 CPU 物理地址，Host Bridge 等仍可能参与转换。[Linux DMA mapping Guide](https://docs.kernel.org/core-api/dma-api-howto.html)

以下例子假设 4 KiB 页面、启用普通 IOMMU 映射，省略 CPU cache 和设备队列，只画同一个 buffer 页：

```text
CPU 访问路径
进程 VA 0x00007f0012345000 ── CPU 页表/MMU ──┐
                                           ↓
                                  DRAM PA 0x0000000120000000
                                           ↑
设备 0000:04:00.0 ── DMA IOVA 0x40000000 ─ IOMMU
                     （特定设备/域的映射）
```

若访问页内偏移 `0x180`，本例 CPU VA 为 `0x00007f0012345180`，设备使用 `0x40000180`，目标 PA 为 `0x0000000120000180`。数值不同仍指向同一位置；这个对应只在各自映射有效时成立。`0x40000000` 本身没有“属于某页”的全局含义，在另一 IOMMU domain 中可能映射到不同 PA，不能仅凭日志里相同数值合并为同一 buffer。

| 地址/标识 | 谁用它 | 与其他字段的区别 |
| --- | --- | --- |
| BDF | PCI 核心、驱动和工具标识 function | 是设备身份位置，不是内存地址 |
| CPU user/kernel VA | 进程或内核执行代码 | 属于对应 CPU 地址空间；用户指针不能直接当 DMA 地址 |
| CPU PA | CPU 物理地址域中的资源位置 | 可指 DRAM 或 MMIO 等资源，不等于任意设备都能直接访问 |
| DMA address / IOVA | 设备访问内存时使用；普通驱动从 DMA API 获取 | 有设备、映射域、权限、长度和生命周期语境 |
| BAR bus base | endpoint 译码其 MMIO aperture 的基址 | 描述设备资源，不是主机传输 buffer 的地址 |
| GPU VA | GPU 内存管理体系中的虚拟地址 | 不自动等于主机 PA 或 NIC 可用的 DMA 地址 |

IOMMU 是设备侧地址转换与访问控制机制，不是 CPU MMU 的另一个名字。CPU 进程页表允许访问某页，不足以授予某个 NIC 相同权限。反过来，设备获得 DMA 映射也不表示某个用户进程拥有该页的 CPU 映射。x86 上常见 Intel VT-d/AMD IOMMU 由平台描述、内核配置和运行模式共同决定具体行为。[Linux x86 IOMMU 文档](https://docs.kernel.org/arch/x86/iommu.html)

前述基础模型有明确例外边界：SVA 等机制可以在硬件/驱动支持下让设备与进程共享虚拟地址语义，并借助 PASID 等区分地址空间。但“硬件可能支持 SVA”不意味着任意设备都能直接使用任意 `malloc` 指针，更不能把 CUDA 统一地址概念自动解释为 NIC 共享 CPU 页表。[Linux x86 SVA 文档](https://docs.kernel.org/arch/x86/sva.html)

## 3. 映射是带方向和生命周期的资源

驱动首先要知道设备 DMA 寻址能力。DMA mask 表达设备能够使用的 DMA 地址位范围，与 BAR 是 32-bit 还是 64-bit 分开设置。64-bit BAR 不证明设备所有 DMA engine 都能使用 64-bit 数据地址。映射还可能因资源不足或约束不满足而失败，驱动需要检查 DMA API 的返回结果，不能把失败值提交给设备。

Linux 提供 coherent allocation 和 streaming mapping 两类常用方式。前者常用于设备与 CPU 共享的控制数据，后者常用于传输 buffer 的阶段性交接。这里的 coherent 主要解决相应内存的 CPU/设备可见性问题，不等于 CPU 与设备之间无需内存屏障、队列协议或完成同步。写好了描述符还需要按设备协议发布，不能让设备先看到“有效”标志却读到未准备好的内容。[Linux DMA API](https://docs.kernel.org/core-api/dma-api.html)

方向按照设备与主机内存之间的数据流理解，不能只看应用函数名中的 send/receive：

| DMA API 方向 | 主机内存的数据流 | 示例语境 |
| --- | --- | --- |
| `DMA_TO_DEVICE` | 设备读取主机准备的数据 | NIC 读取发送 buffer |
| `DMA_FROM_DEVICE` | 设备写入主机内存 | NIC 将接收数据写入 buffer |
| `DMA_BIDIRECTIONAL` | 两个方向都会访问 | 双向共享 buffer；仍需要正确同步，不能用来掩盖未知方向 |

以一次 streaming 接收为例，buffer 在提交前需要具备有效映射，设备写入期间 CPU 不能随意处理未完成内容；确认完成后，驱动按 API 要求同步回 CPU 或解除映射，再消费数据。对于保持映射并复用的 buffer，`dma_sync_*_for_cpu()` 与 `dma_sync_*_for_device()` 表达相应阶段的可见性转换；对于结束使用的映射，则需在设备停止访问后解除。缓存一致性平台可能不需要相同的实际 cache 操作，但驱动仍须遵守 API 和设备协议。

```text
准备有效内存 → 为设备建立 DMA 映射并检查结果
                     ↓
          发布描述符/通知设备（保持正确顺序）
                     ↓
           设备传输；buffer 与映射保持有效
                     ↓
       确认完成或可靠停止 DMA，处理必要同步
                     ↓
      解除映射/回收资源 → 最后才能释放内存
```

这个顺序解释了一类重要故障：CPU 已经释放或 unmap buffer，设备仍持有旧描述符继续 DMA。IOMMU 可能因无有效映射而报 fault；若旧 IOVA 已被重用，也可能访问到非预期资源，未必每次都有显眼错误。映射调用成功只证明当时建图成功，不证明长度、权限、使用时机和停止顺序都正确。

虚拟连续也不意味着物理连续。分散的主机页面可用 scatter-gather 描述；DMA API 可能合并条目，驱动提交给硬件的条目数与原始条目数不能混用，解除映射又须遵循 API 参数要求。本篇不提供可加载驱动代码，以免用缺少设备队列、错误处理与缓存边界的伪实现代替真正可运行例子。

## 4. 页面固定、bounce buffer 与 IOMMU group

设备异步访问用户内存时，内核必须协调页面生命周期，避免 CPU 地址空间变化使设备继续引用无效位置。传统注册式 DMA/RDMA 路径常需要 pin 用户页面并建立相应映射；pin 与 DMA map 分别处理页面生命周期和设备访问关系，不能互相代替。长期 pin 还有更严格约束。支持按需缺页或 MMU notifier 协调的实现可以采取其他方案，不能把“所有 RDMA 必须永久 pin 全部页面”作为普遍规律。[Linux pin_user_pages 文档](https://docs.kernel.org/core-api/pin_user_pages.html)

同样，`mlock`、CUDA pinned host memory 与 RDMA memory registration 不是同一个 API 契约。应用锁住页面不表示它已经取得可供任意设备使用的 IOVA；RDMA 注册还涉及设备访问权限和注册资源；GPU 内存又需要相应驱动与导出/注册支持。后续看一个注册失败时，应先判断失败的是主机页面管理、DMA 映射、设备注册，还是 GPU peer memory 路径。

某些设备不能访问原始内存地址，或平台要求隔离、解密等特殊处理时，Linux 可能使用 SWIOTLB bounce buffer。数据经设备可达的中间 buffer 搬运，再由软件在原 buffer 与中间 buffer 间复制。因此 DMA 不必然等于端到端零 CPU 拷贝；bounce pool 耗尽也可能导致映射失败，而非 PCIe 没枚举。[Linux SWIOTLB 文档](https://docs.kernel.org/core-api/swiotlb.html)

IOMMU group 则描述隔离粒度，不是 NUMA node、PCIe Switch 型号或 DMA 映射表。同组设备可能因为拓扑与隔离能力无法安全地彼此独立分配；group 与当前使用的 IOMMU domain 也不是同一个概念。存在 `iommu_group` 链接不证明某个 buffer 已建立映射，不证明该设备当前一定使用翻译模式，更不证明 P2P 性能最优。[Linux VFIO 文档](https://docs.kernel.org/driver-api/vfio.html)

## 5. 接到 GPU/NIC 通信时，哪些基础仍然成立

若 NIC 访问主机 DRAM，前述 buffer/mapping 模型直接适用。若 NIC 通过 GPUDirect RDMA 访问 GPU 内存，还要考虑 GPU 内存的导出、注册、peer 可达性和访问生命周期。此时不能把 GPU VA 直接当 NIC 地址，也不能把普通 DRAM 的 IOMMU 映射图原封不动套上去。NVIDIA GPUDirect RDMA 文档讨论了拓扑、IOMMU 和内存同步限制；具体支持边界需结合设备代际、驱动以及使用的注册机制确认，不将其中某一传统路径的限制推广成所有新平台的统一结论。[NVIDIA GPUDirect RDMA 文档](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html)

排障时，GPU/NIC 已有 BDF、driver 已绑定而通信仍失败，是进入映射/注册与实际数据路径的合理起点。IOMMU fault 值得查，但也不能先假设所有通信超时都是 IOMMU。设备执行队列、链路、远端网络、completion 通知和软件同步都可能产生相似表象。

如果日志包含 requester/source ID、fault address 和访问权限信息，应先保留其原始格式，按实际平台解释 requester 与 BDF 的对应，再与设备提交队列、buffer 注册和释放时间对齐。fault address 通常属于设备侧地址语境，不能拿去直接与进程 VA 比较，也不能通过 `/proc/iomem` 查不到就断言地址随机损坏。设备报告故障时的请求发起者与被访问目标可能不是同一颗设备。

## 6. Linux 中先做只读观察

本轮不启停 IOMMU、不改启动参数、不写设备寄存器、不执行 DMA benchmark。下面命令用于 Linux 宿主机，未在用户服务器运行；容器/VM 视图、日志权限及内核配置可能限制结果。启动参数是意图线索，不是最终模式证明，应与启动日志及设备信息一起阅读。

```bash
uname -r
cat /proc/sys/kernel/random/boot_id
cat /proc/cmdline

# 过滤用于找位置；随后阅读同次启动的完整上下文
journalctl -k -b -o short-monotonic --no-pager |
    grep -Ei 'DMAR|AMD-Vi|IOMMU|IO_PAGE_FAULT|swiotlb|DMA.*(fault|fail|error)'
```

`DMAR` 或 `AMD-Vi` 出现在日志中只说明存在相关信息，不自动证明每个设备使用相同映射模式。没有匹配行也不能证明 IOMMU 不存在。日志中的“权限失败”“缺失映射”“地址宽度问题”等需按平台字段解释，不由一个关键词直接选择关闭 IOMMU。

接着对目标 NIC/GPU 检查身份、驱动、NUMA 与 group。它们是四种关系，本例只打印当前元数据，不枚举 DMA page table：

```bash
# 替换为本机实际 BDF
bdf=0000:04:00.0
dev_dir="/sys/bus/pci/devices/$bdf"
if [ -d "$dev_dir" ]; then
    readlink -f "$dev_dir"
    lspci -D -s "$bdf" -nnk
    if [ -r "$dev_dir/numa_node" ]; then
        printf 'NUMA node: '
        cat "$dev_dir/numa_node"
    fi
    if [ -L "$dev_dir/iommu_group" ]; then
        group_dir=$(readlink -f "$dev_dir/iommu_group")
        printf 'IOMMU group: %s\n' "$group_dir"
        for member in "$group_dir"/devices/*; do
            [ -e "$member" ] || continue
            printf '%s\n' "${member##*/}"
        done
    else
        printf '%s\n' '当前视图没有 iommu_group 链接，具体原因待核对'
    fi
else
    printf '%s\n' '设备未在当前 PCI 视图中，先回到枚举与地址身份检查'
fi
```

Linux 没有一个通用 `lspci` 选项可以还原所有应用 buffer 的 IOVA 映射和所有设备队列。用户态工具显示出一个 group 只是隔离拓扑的证据，不能当作某次 DMA 成功的证据。进一步定位往往需要目标驱动已有日志或专门 tracing，以及明确的 buffer 地址类型和时间；本篇不擅自启用全局调试或建立测试负载。

| 现象 | 优先检查 | 不能直接推出 |
| --- | --- | --- |
| BDF、BAR 正常，DMA mapping 失败 | DMA mask、映射长度/类型、IOVA 或 bounce 资源及驱动返回值 | 不是“没枚举”，也不必然是卡损坏 |
| DMA 后出现 IOMMU fault | 请求设备、访问方向/权限、地址、map/unmap 与复位时间 | 不足以认定 IOMMU 应被关闭 |
| 只在高并发或退出时 fault | in-flight 请求与资源释放/复用顺序 | 不足以证明带宽瓶颈或物理链路坏 |
| 数据偶发陈旧或损坏，没有 fault | ownership、cache sync、内存顺序、长度与设备协议 | 没 fault 不代表映射和同步完全正确 |
| 大 buffer 或长期运行后失败 | 注册/pin、映射与 bounce 资源，是否泄漏或受限 | 不应直接解释成 GPU HBM 不足 |
| 同组或同 NUMA 的 GPU/NIC 仍通信失败 | 实际 peer 路径、注册机制、版本、completion 与网络 | 隔离/局部性关系不等于 GDR 可用性 |

## 来源、范围与下一篇

本轮于 2026-09-13 使用文内链接的 Linux DMA mapping Guide、DMA API、SWIOTLB、pin_user_pages、VFIO、x86 IOMMU/SVA 官方文档，以及 NVIDIA GPUDirect RDMA 文档。它们支持地址、映射、页面生命周期与平台边界；未进行固定版本 IOMMU driver 源码分析，未将滚动文档中的最新功能视为用户平台已支持。

本篇地址图与排障场景是教学推理，没有从用户机器读取 VA/PA/IOVA 或提交 DMA 请求。实际硬件、IOMMU 模式、驱动注册方式与故障日志仍未知。已有 IO/虚拟内存笔记保持原文，本篇限定独立 DMA 控制器模型的适用范围而不批量修订旧内容。

下一篇继续第 4 阶段的 MSI/MSI-X 与完成通知，把“数据已经搬完”和“CPU/应用得知完成”分开，再解释 `/proc/interrupts`、中断向量、队列及 CPU affinity 如何对应。其后继续 PCIe 链路和错误处理；本阶段尚未全部成文。
