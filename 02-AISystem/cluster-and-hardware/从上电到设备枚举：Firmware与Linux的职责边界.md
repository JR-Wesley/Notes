# 从上电到设备枚举：Firmware 与 Linux 的职责边界

本篇对应 [[02-AISystem/cluster-and-hardware/GPU服务器：从物理硬件到Linux设备系统|GPU 服务器知识地图]] 第 3 阶段，承接 [[01-ComputerScience/SoC/PCIE/PCIe拓扑与BDF：从设备地址追踪上游|PCIe 拓扑与 BDF]]。上一阶段解决“一个已知设备挂在哪里”，本篇进一步解释“它如何被发现，以及发现之后还要完成什么”。范围以 DGX H100 所代表的 x86/ACPI、多 GPU 服务器为参考，不声称掌握某块主板的实际电源时序、BIOS 实现或所有设备 firmware 的启动协议。

工程上最重要的区分是：物理设备准备好、PCI 配置访问成功、内核登记设备、资源可用、功能驱动初始化成功、用户态可使用，是不同阶段。下游阶段失败不自动否定上游阶段曾成功；反过来，曾成功也不能证明运行期仍健康。本文用启动依赖、职责边界和时间线组织判断，而不是把所有设备不可用统称为“没枚举”。

## 1. 启动是一组相互依赖的过程

主机上电之前，平台可能已经有待机电源供给 BMC 和部分管理电路；主机上电后，板级控制逻辑按照设计处理电源、时钟和复位，CPU 开始执行平台 firmware，设备侧也有自己的启动过程。具体先后和并行关系依平台而异，不能把“BMC → BIOS → 每颗设备 firmware → Linux”画成普遍严格的串行调用链。

下面是为了定位故障而构造的依赖图，不是实测时间线。箭头表示后面的观察通常依赖前面的条件，省略平台内部并行、重试和复位动作：

```text
平台电源/时钟/复位条件 ───────→ endpoint 的 PCIe 接口可工作
          │                               │
          └→ CPU 执行平台 firmware         └→ 下游链路建立、配置访问可响应
                    │                                  │
             初始化主机平台与 IO                        │
             准备启动与平台描述                         │
                    │                                  │
              OS loader / Linux                        │
                    └→ 建立 PCI 根与配置访问方法 ────────┘
                                  │
                          发现 function、处理资源
                                  │
                          登记设备、匹配功能驱动
                                  │
                      probe：资源、firmware、设备初始化
                                  │
                          用户态接口与服务准备
```

在常见 PCIe 下游设备上，链路训练需要适当的供电、时钟和复位条件。LTSSM 是硬件链路训练状态机，链路能够进入工作状态与 GPU 计算功能就绪不是同一件事。在 x86/ACPI 平台，根侧常由 firmware 预先配置；部分 Devicetree 平台则由专用 host controller driver 处理 PHY、时钟、电源域等。因此不能把某 SoC 驱动里的初始化顺序直接当作 DGX BIOS 的实现。[Linux PCI Host Controller 文档](https://docs.kernel.org/PCI/controller/pci-controller-drivers.html)

设备的基础 PCIe 接口可能已能返回配置空间，而它的计算、网络或管理功能仍等待后续初始化。故障也可能更早发生，使配置访问无法成功。仅凭“firmware 有问题”不能定位阶段，需要知道是哪一份 firmware、运行在哪个处理器或控制器上、失败发生在何时，以及哪些接口当时仍可访问。

## 2. Firmware 不是单一软件层

firmware 通常指与硬件平台或设备紧密相关的软件，但“firmware”三个字本身不告诉你它由谁执行、保存在何处、由谁更新，或由谁加载。服务器中的 BIOS/UEFI、BMC firmware 和 GPU/NIC firmware 是不同对象；CPLD 配置也不能直接等同于由 Linux 加载的可执行 firmware 文件。

| 对象 | 运行位置与主要职责 | 与 Linux 的边界 |
| --- | --- | --- |
| 平台 BIOS/UEFI firmware | 主机 CPU 执行的平台启动软件，初始化必要硬件并建立启动环境 | 不等于 Linux kernel；向 OS 提供平台描述和约定的服务 |
| BMC firmware | BMC 自身运行的软件，负责其监控、事件和管理功能 | 管理视角独立于主机 PCI 枚举；BMC 可用不代表主机设备可用 |
| 设备 firmware | GPU、NIC、存储控制器或其他设备侧的软件 | 可能从设备非易失存储启动，也可能依赖主机驱动加载；方式以设备为准 |
| 板级 MCU/CPLD | 执行控制程序或实现可编程逻辑，可能参与电源/复位/状态管理 | 不保证作为 PCI function 枚举；需查板级设计 |
| KMD | Linux 内核中的设备驱动部分 | 使用内核接口管理设备，可与设备 firmware 交互；不是设备 firmware 本身 |
| NVIDIA Fabric Manager | 主机用户态服务，参与 NVSwitch fabric 管理 | 在本参考模型中简称 FM；不能用 FM 服务状态代替 PCI 枚举结果 |

DGX H100 的 BMC 提供传感器、事件及远程管理功能。其库存和状态数据有自己的采集方式与更新时间，不应直接当作当前 Linux 扫描结果。[DGX H100 BMC 文档](https://docs.nvidia.com/dgx/dgxh100-user-guide/bmc.html) NVIDIA FM 文档则面向 NVSwitch 系统的 fabric 管理，其依赖与行为有代际差异；这里先定位其职责，后续 GPU fabric 主题再深入。[Fabric Manager 文档](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/)

还要区分“加载 firmware”和“刷写 firmware”。Linux `request_firmware()` 等 API 为驱动获取 firmware 内容，驱动通常还需要把内容传送给设备并完成后续协议；这个 API 的成功不等于设备已经启动成功，也不必然意味着修改了设备 flash。文件不存在、传输失败、设备拒绝版本和设备侧启动失败，需要分别解释。[Linux request_firmware API](https://docs.kernel.org/driver-api/firmware/request_firmware.html)

### UEFI 退出启动服务，不代表平台软件消失

UEFI 环境中的 Boot Services 与 Runtime Services 有明确生命周期差别。OS loader 在接管阶段调用 `ExitBootServices()` 结束启动服务；之后仍存在规范定义的运行时服务。这与 Linux 设备驱动管理设备是不同层面的约定，不能把它理解成 firmware 把每个硬件寄存器都移交给某个 KMD 的单一瞬间。[UEFI 2.10 Runtime Services](https://uefi.org/specs/UEFI/2.10/08_Services_Runtime_Services.html)

平台控制还可能按具体功能协商。以 PCIe AER 为例，Linux 文档说明 OS 是否接管其处理与 firmware 通过 ACPI `_OSC` 授予控制相关。因此 dmesg 中没有 AER 消息，不能独立证明硬件从未报告错误，也不能先假定所有错误都由 Linux 记录。[Linux PCIe AER 文档](https://docs.kernel.org/PCI/pcieaer-howto.html)

## 3. Linux 如何获得根，再发现下游设备

Linux 不能仅靠扫描 PCI 下游就发现整个主机平台。对于 ACPI 平台，Host Bridge 及其配置访问方式、可转发的地址窗口等需要平台描述；在这些信息基础上，PCI 子系统才能访问相应层次。MCFG 提供 ECAM 相关信息，Host Bridge 的 `_CRS` 描述资源；它们不是 GPU 驱动，也不是已完成 GPU 初始化的清单。下游 PCI 设备通常能够通过标准配置访问发现，不要求 ACPI 为每个 GPU/NIC 复制一份完整设备清单。[Linux 6.14 PCI/ACPI 文档](https://docs.kernel.org/6.14/PCI/acpi-info.html)

配置空间提供 vendor/device ID、类别、header/capability 等发现所需信息。PCI 子系统据此识别 function，遇到 Bridge 时结合总线范围继续处理下游，形成前一篇中的层次。平台 firmware 可能已经做过一轮枚举和资源配置，Linux 仍要建立自己的设备模型，并按平台、配置及资源情况保留或调整相应设置。不能简单断言“BDF 永远由 BIOS 决定”，也不能反过来说“Linux 总是从零重编号”。

可以用一个很小的固定版本源码片段验证“发现与功能驱动分开”的机制。在 Linux v6.12、`drivers/pci/probe.c` 中，新设备扫描路径的 `pci_scan_single_device()` 调用 `pci_scan_device()`；后者先经 `pci_bus_read_dev_vendor_id()` 检查标识读取，再分配并设置 `struct pci_dev`，扫描成功后由前者调用 `pci_device_add()`。这是局部发现路径，不是从内核入口到 sysfs 和驱动绑定的完整调用链，也没有覆盖所有平台和错误恢复分支。[Linux v6.12 `drivers/pci/probe.c`](https://raw.githubusercontent.com/torvalds/linux/v6.12/drivers/pci/probe.c)

这说明发现 PCI function 不要求 GPU/NIC 的功能驱动先初始化成功。不要看到 `probe.c` 文件名就把其中 PCI 核心扫描误认为 NVIDIA 或 mlx5 的 `probe()` 回调。前者是在总线层发现设备，后者是在已有设备基础上尝试管理功能；两者都可能被工程口语称为 probe，却处于不同边界。

资源处理同样不能与设备身份合并成一个布尔状态。内核可能已经识别 BDF 并保留设备对象，但 BAR 或上游窗口仍未获得可用分配，从而使后续驱动初始化失败。`lspci` 出现一行设备记录证明不了 MMIO、DMA、中断或完整业务功能已经可用；BAR 与地址窗口的细节在下一阶段展开。

## 4. 从设备对象到驱动接管，还差哪些步骤

内核登记的设备与注册的驱动分别存在，匹配后才会尝试调用功能驱动的 `probe()`。驱动可以先于设备注册，也可以在设备发现后由模块加载进入内核；驱动也可能编入内核，因此不能用 `lsmod` 列表作为全部驱动的清单。Linux 的设备/驱动匹配机制允许这两种到达顺序。[Linux Driver Binding](https://docs.kernel.org/driver-api/driver-model/binding.html)

典型 PCI 功能驱动需要启用设备资源、申请和映射寄存器区、设置 DMA 能力与中断，并按设备协议初始化硬件；具体步骤、顺序和用户接口注册方式由驱动实现决定。`probe()` 返回失败通常使该次绑定不成功，但 PCI 设备对象可以继续存在，所以“有 BDF，没有 driver 链接”是合理状态。[Linux PCI Driver 文档](https://docs.kernel.org/PCI/pci.html)

然而，不能把这个常见行为推成“驱动永远不可能导致设备消失”。初始化可能触发复位或暴露运行期故障，热插拔/错误处理或管理操作也可能移除内核设备对象。sysfs 的 `remove` 就会将设备从内核列表中移除，且不等同于物理断电。[Linux PCI sysfs](https://docs.kernel.org/PCI/sysfs-pci.html) 因而需要分别描述：普通 probe 失败、驱动操作之后发生的链路或设备故障，以及内核设备移除。

驱动绑定成功之后，CUDA/NVML、网络/RDMA 接口、权限、用户态库及相关服务仍可能有独立的准备过程。在 NVSwitch 服务器上，还要核查 GPU fabric 的状态。FM 报错与 GPU BDF 缺失可能同时出现，但 FM 报错本身不能证明它造成了 PCI 枚举失败；应比较时间和依赖关系。

## 5. 用时间线判断设备停在哪个边界

沿用教学地址 `0000:03:00.0`，假设启动时内核曾打印其标识与资源信息，稍后功能驱动报告初始化失败，最后查询 sysfs 仍找到设备但没有 driver 链接。这里有证据支持“设备曾被发现，当前未成功绑定”，排查应进入资源、firmware、版本与初始化条件，不能概括为“没枚举”。如果设备目录后来消失，则需要找两个时刻之间的移除、复位、链路或平台事件，不能用最后一次查询覆盖此前的发现事实。

| 观察阶段 | 有证据支持什么 | 下一步应找什么 |
| --- | --- | --- |
| 主机未进入 Linux | Linux 当前日志不足以解释此前失败 | 平台控制台、BMC 事件、主机上电/POST 状态；先核对供电与启动范围 |
| Linux 根层次就与预期不同 | 问题可能在平台描述、根初始化或观察环境 | ACPI/root 相关日志、配置变化、虚拟化视图与正常拓扑 |
| Root Port 可见，endpoint 未见 | 根侧对象已存在，不等于下游已准备好 | 上游链路状态、下游供电/复位、配置访问、是否曾登记后移除 |
| BDF 存在，出现资源分配问题 | 身份发现与资源可用性不同步 | BAR、上游窗口、平台资源配置与具体错误，下一篇深入 |
| BDF 存在，无 driver 链接 | 当前未绑定，不证明 probe 曾运行 | 匹配、模块是否可用、绑定到其他驱动的历史、probe 日志 |
| 驱动报告 firmware 加载或启动错误 | 进入了相关初始化路径 | 区分文件获取、版本验证、传输和设备侧启动；核对设备身份与时间 |
| 绑定成功而工具/通信不可用 | 问题仍可能位于多个后续边界 | 用户态库、设备接口、权限、fabric 和运行期日志 |
| 曾可用，之后 BDF 消失 | 运行期变化，不能当成从未枚举 | 最后正常时刻、复位/热插拔/移除/错误事件及共同上游 |

“冷启动正常、重启异常”只说明不同复位/保留状态值得检查，不直接证明某个 firmware bug。主机重启、设备功能复位、总线复位、主机断电和 BMC 重启通常影响不同范围。后续实际处理必须依据平台说明和维护窗口决定动作，本轮只建立观察方法，不尝试用反复重启或刷写替代定位。

### 将当前状态与同一次启动的日志放在一起

下面命令用于 Linux 宿主机，未在用户服务器运行。`journalctl` 需要相应日志环境与权限；若不存在则使用 `dmesg`，但不能假设它保留完整历史。boot ID 区分启动实例，单次启动内相对时间有助于排序；跨 BMC、主机和外部日志比较时，还需核对时钟和时区偏差。

```bash
uname -r
cat /proc/sys/kernel/random/boot_id
cat /proc/cmdline

# 查看本次启动的内核完整上下文
journalctl -k -b -o short-monotonic --no-pager

# 没有 journalctl 时，按环境使用内核环形日志
dmesg
```

先保留或查看完整上下文，再按实际 BDF 与已确认上游 BDF 定位。下例只演示局部查询，不能把过滤后的空结果当作“没有发生过”：

```bash
# 替换成实机地址；BDF 可能随启动变化
bdf=0000:03:00.0
dev_dir="/sys/bus/pci/devices/$bdf"
journalctl -k -b -o short-monotonic --no-pager | grep -F -- "$bdf"

if [ -d "$dev_dir" ]; then
    readlink -f "$dev_dir"
    for attr in vendor device class; do
        if [ -r "$dev_dir/$attr" ]; then
            printf '%s: ' "$attr"
            cat "$dev_dir/$attr"
        fi
    done
    if [ -L "$dev_dir/driver" ]; then
        readlink -f "$dev_dir/driver"
    else
        printf '%s\n' '当前未发现 driver 链接'
    fi
    lspci -D -s "$bdf" -nnk
else
    printf '%s\n' '当前路径不存在；请核对设备身份、其他地址和此前日志'
fi
```

某些日志使用缩写 BDF、父端口或驱动名，精确过滤可能漏掉关联事件。读取 `vendor` 等属性也主要是在观察内核记录，不能把它当成实时链路测试。最少的实机记录应包含 boot ID、稳定设备身份、当时 BDF、父路径、driver 状态、首次相关错误及其上下文；若未找到某项就标记缺失，不补写假定结果。

对于 UEFI 启动视图，可以只读检查下面两个位置，但它们都不能证明表内容或 firmware 行为正确：

```bash
if [ -d /sys/firmware/efi ]; then
    printf '%s\n' '当前内核暴露 EFI 接口目录'
else
    printf '%s\n' '当前视图未暴露 EFI 目录，需结合启动方式和环境判断'
fi

if [ -d /sys/firmware/acpi/tables ]; then
    ls /sys/firmware/acpi/tables
fi
```

这里不读取全部二进制 ACPI 表，也不建议随意改变启动参数、ACPI 设置、BAR 配置或设备 firmware。只有明确了失败边界，下一轮才有依据选择更深入的字段和实验。

## 来源与后续

本轮于 2026-09-13 使用文内链接的 Linux PCI Host Controller、PCI Driver、Driver Binding、PCI sysfs、firmware API、AER 文档，以及 Linux 6.14 PCI/ACPI 文档和 Linux v6.12 `drivers/pci/probe.c`。源码只核查新设备扫描的局部符号与关系，不代表用户内核实现已核对。NVIDIA BMC/FM 文档用于参考平台职责；UEFI 2.10 Runtime Services 官方检索内容用于启动/运行时服务边界。UEFI Overview 全文获取返回 403，未作为已通读来源。

所有示例均为教学推理，没有真实启动日志、故障复现、复位、刷写或性能测试。用户服务器型号、BIOS/kernel/driver/设备 firmware 版本、启动方式及原有 XML 来源仍未知。本文没有修改旧知识正文，也未将任何假设登记为已确认根因。

下一篇见 [[01-ComputerScience/SoC/PCIE/配置空间、BAR与MMIO：从设备身份到地址资源|配置空间、BAR 与 MMIO：从设备身份到地址资源]]，先解释为什么 BDF 已经存在，驱动仍可能因为地址资源不可用而失败，随后继续 DMA/IOMMU、中断和链路错误。
