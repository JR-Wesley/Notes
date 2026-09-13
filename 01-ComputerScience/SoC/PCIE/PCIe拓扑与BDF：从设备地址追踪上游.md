# PCIe 拓扑与 BDF：从设备地址追踪上游

本篇对应 [[02-AISystem/cluster-and-hardware/GPU服务器：从物理硬件到Linux设备系统|GPU 服务器学习地图]] 第 2 阶段，承接 [[02-AISystem/cluster-and-hardware/物理服务器、NUMA与PCIe根层次|物理服务器、NUMA 与 PCIe 根层次]]。目标是拿到一个 PCI BDF 后，能确定它属于哪种 function、位于哪条分支、经过哪些上游 Bridge，以及当前证据是否足以指向某个故障范围。完整上电时序、BAR 分配和驱动 probe 留给后续主题。

贯穿平台仍是多 GPU 服务器，但本文所有 BDF、总线范围和输出片段都是人工构造的教学示意，不是 DGX H100 固定地址或用户服务器实测。物理服务器提供应用背景，PCIe 机制归入本目录，避免将通用知识重复写入 GPU 和 NCCL 笔记。

## 1. BDF 定位的是 function，不是物理板卡

Linux 常用 `domain:bus:device.function` 标识 PCI function，例如 `0000:03:00.0`。严格说 BDF 是后三部分，日常工程交流常把带 domain 的完整地址也称为 BDF。传统编码下 bus 占 8 bit，device 占 5 bit，function 占 3 bit；常见文本显示均使用十六进制。`03` 不是第三张 GPU，device 字段也不是机箱第三个插槽。

| 示例字段 | 表示什么 | 不能直接推出什么 |
| --- | --- | --- |
| `0000` | Linux PCI domain，区分总线地址空间 | socket 0、NUMA node 0 或某一颗唯一的 Root Complex |
| `03` | 该 domain 内的 bus number | 物理线缆编号或链路跳数 |
| `00` | bus 上的 device 编号 | GPU 序列号、机箱槽位号 |
| `.0` | function 编号 | 第一个网口或应用中的 GPU 0 |

PCI function 是配置和驱动管理的重要单位，有自己的配置空间；一个 function 可对应一个 GPU 功能，也可以是 Bridge、网络控制器或其他功能。一个实体芯片/板卡可暴露多个 function，一个网络 function 也可能提供多个端口。因此 `lspci` 的记录数量、板卡数、网口数与用户态设备数之间需要建立映射，不能直接画等号。

传统 device/function 编码可容纳每个 bus 32 个 device、每个 device 8 个 function，但不要把它推成“一张 PCIe 卡最多有 8 个软件功能”的通用硬件结论。ARI 改变了该 8 bit 字段的解释，使 function 编号可以扩展；Linux 的 `devfn` 及相关宏保留了这组编码关系。这里仅说明边界，不展开 ARI 路由。[Linux v6.12 `include/linux/pci.h`](https://raw.githubusercontent.com/torvalds/linux/v6.12/include/linux/pci.h)

BDF 是当前位置，vendor/device ID 是设备类型标识，都不等于稳定的单机物理身份。两张同型号卡可以有相同 vendor/device ID，却拥有不同 BDF。重启、平台配置、热插拔或虚拟功能配置变化后，地址还可能重新分配。跨时间比较应同时保存 GPU UUID、设备序列号或平台槽位信息；哪些标识实际可用，取决于设备和管理接口。

## 2. Root Port、Bridge 和 Switch 如何组成树

从主机看，Host Bridge 将主机系统连接到 PCI 层次，root bus 是这一层次的起点。Root Port 位于根侧，在配置模型中表现为 PCI-to-PCI Bridge；它向下游连接 endpoint，或连接 Switch 的 upstream port。Switch 再通过多个 downstream port 扩展分支。Root Complex 是根侧逻辑整体，不能从 `lspci` 中找一条记录就假设那一条代表整颗 RC。[Linux PCI Host Controller 文档](https://docs.kernel.org/PCI/controller/pci-controller-drivers.html)

下面用一个 Root Port、一颗具有两个下游端口的 Switch、一个 GPU function 和一个 NIC function 解释软件层次。图中的内部 bus 是 Switch 的逻辑表示，不是多块板卡共享一根传统并行总线：

```text
Host Bridge / root bus 0000:00
└─ 0000:00:01.0  Root Port
   │  外部 PCIe link
   └─ 0000:01:00.0  Switch upstream port
      │  Switch 内部逻辑 bus 02
      ├─ 0000:02:00.0  Switch downstream port A
      │  │  外部 PCIe link
      │  └─ 0000:03:00.0  GPU function
      └─ 0000:02:01.0  Switch downstream port B
         │  外部 PCIe link
         └─ 0000:04:00.0  NIC function
```

同一颗 Switch 在这里至少贡献一个 upstream port 和两个 downstream port 的 Bridge function。不要把它们数成三颗实体 Switch，也不要把软件树中的每次缩进都理解为板外的一段链路。Retimer、背板和连接器可能位于真实信号路径上，但不作为中间 PCI Bridge 出现在这棵树中。

`PCI bridge` 是类别信息，无法仅凭这两个词区别 Root Port 与 Switch downstream port。要结合父路径和 PCI Express capability 中的 port type。GPU 的 function 类别、Bridge 的类别和具体端口类型属于不同字段，只有同时阅读才知道某个 BDF 在树中承担什么角色。Linux v6.12 定义了 Endpoint、Root Port、Upstream Port、Downstream Port 等 PCIe 类型及 Type 1 Bridge 配置字段。[Linux v6.12 `pci_regs.h`](https://raw.githubusercontent.com/torvalds/linux/v6.12/include/uapi/linux/pci_regs.h)

### Bridge 的地址与它管理的 bus 范围不同

Bridge 自身的 BDF 位于它的上游 bus；它还保存 Primary、Secondary、Subordinate bus number。Primary 是该 Bridge 上游所在 bus，Secondary 是紧接下游的 bus，Subordinate 是该 Bridge 下游可覆盖的最大 bus number。它们帮助软件和硬件确定总线层次及配置访问范围，不是 BAR 的内存地址窗口。

为上面的图设置以下自洽范围，所有数值均为十六进制：

| Bridge 自身 BDF | 类型 | Primary | Secondary | Subordinate |
| --- | --- | --- | --- | --- |
| `0000:00:01.0` | Root Port | `00` | `01` | `04` |
| `0000:01:00.0` | Switch upstream | `01` | `02` | `04` |
| `0000:02:00.0` | downstream A | `02` | `03` | `03` |
| `0000:02:01.0` | downstream B | `02` | `04` | `04` |

GPU 的 bus `03` 落在 Root Port 的 `01–04` 和 upstream port 的 `02–04` 范围内，最终由 downstream A 接到 bus `03`。NIC 的 bus `04` 则进入另一个分支。假如在上游看到 `secondary=01, subordinate=04`，只能说该分支覆盖这些总线号，不能推出每个号码当前都有 endpoint；平台可能预留 bus number，实际树也可能有空隙。

这些寄存器给出枚举结果的一部分，但本文没有假设 firmware 和 Linux 各自在哪一步配置它们。枚举时总线编号如何建立、何时保留或重分配，是下一阶段“从上电到枚举”的问题。当前先掌握：**Bridge 的 BDF 用于定位 Bridge 本身，Secondary/Subordinate 描述它通向的下游范围。**

## 3. 将树、sysfs 路径与驱动视图对齐

对上面同一棵树，`lspci -D -t` 可呈现类似的精简结构。以下是人工整理的示意，具体空格、命名与布局随 pciutils 版本变化，不作为程序解析格式：

```text
-[0000:00]---01.0-[01-04]----00.0-[02-04]--+-00.0-[03]----00.0
                                          \-01.0-[04]----00.0
```

最左侧 `[0000:00]` 给出根的 domain/bus。接着的 `01.0` 在 bus `00`，所以完整地址是 `0000:00:01.0`；它后面的 `[01-04]` 描述下游总线范围。下一层 `00.0` 在 bus `01`，即 `0000:01:00.0`。最末端两个 `00.0` 分别位于 bus `03` 与 `04`，完整 BDF 不同，不能把缩写相同理解成同一个设备。

在 sysfs 中，GPU 的设备入口解析为：

```text
/sys/bus/pci/devices/0000:03:00.0
  -> /sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0/0000:02:00.0/0000:03:00.0

/sys/bus/pci/devices/0000:04:00.0
  -> /sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0/0000:02:01.0/0000:04:00.0
```

`/sys/bus/pci/devices` 按 BDF 提供入口，解析后的 `/sys/devices` 路径表达内核设备父子关系。GPU 和 NIC 的路径共享到 `0000:01:00.0` 为止，之后分别进入两个 downstream port。这比“它们都属于 node 0”提供了更具体的共同上游信息。路径中的 `pci0000:00` 是根层次标识，不是某个 GPU 的 BDF。[Linux PCI sysfs](https://docs.kernel.org/PCI/sysfs-pci.html)

设备目录下的 `driver` 链接表达另一种关系：谁正在管理该 function。它不改变设备在 PCI 树中的父路径。`lspci -k` 中的 `Kernel driver in use` 与 `Kernel modules` 也不能等同，前者描述当前处理驱动，后者列出可能处理该设备的模块。模块候选出现不证明已绑定，更不证明 GPU 功能已初始化成功。[pciutils `lspci` 手册](https://raw.githubusercontent.com/pciutils/pciutils/master/lspci.man)

sysfs 是内核视图，不是设备健康证书。设备对象仍存在时，实际链路或配置访问也可能已经出错；反之，某 BDF 路径不存在可能涉及地址变化、软件移除、虚拟化视图或未枚举。必须结合时间和上下文判断，不能把路径存在与物理健康互相替代。

## 4. 用少量命令完成一次向上追踪

下面只读观察 Linux 宿主机，不执行 reset、remove、rescan、unbind 或配置空间写入。需要 pciutils；完整 capability 读取可能受权限限制。先确认目标属于当前宿主机视图，再将示例 BDF 替换为实际地址。命令未在用户服务器执行。

```bash
lspci --version
lspci -D -t
lspci -D -nnk

bdf=0000:03:00.0
dev_dir="/sys/bus/pci/devices/$bdf"
if [ -d "$dev_dir" ]; then
    readlink -f "$dev_dir"
    lspci -D -s "$bdf" -nnk
    lspci -D -PP -s "$bdf" -nn
else
    printf '%s\n' '该 BDF 不在当前内核视图中，请先核对地址与环境'
fi
```

`-D` 保留 domain，`-nn` 保留数字 ID 与名称，`-PP` 按经过的 Bridge 路径展示设备并保留 bus 信息。`-s` 使用完整 BDF 可减少误选；例如省略 domain 或 bus 可能匹配多个对象。树形输出便于人阅读，自动化采集应选择机器可读格式或 sysfs 属性，而不是依赖 ASCII 树的空格布局。

随后只对需要核查的上游端口读取详细信息。本教学树的 GPU 直接父端口是 `0000:02:00.0`；实际机器应使用刚才解析出的父 BDF：

```bash
# 替换成实际父 Bridge，不要直接照用教学地址
parent_bdf=0000:02:00.0
lspci -D -s "$parent_bdf" -vv
```

先看 port type 与 `Bus: primary=..., secondary=..., subordinate=...`，再看 Link capability/status。`LnkCap` 的支持能力与 `LnkSta` 的协商状态不同，链路速率/宽度还要同对端和平台设计对照。本篇只引入阅读位置，不展开省电、训练和错误恢复细节。若显示 `<access denied>`，应记录信息不完整，而不是把缺失字段当成链路不存在。

若要一次列出当前 endpoint 到根方向的 PCI function，可在 Bash 中使用下面的只读循环。它依据解析后的设备父路径逐级列出已有 PCI 对象，不尝试发现内核没有登记的硬件：

```bash
# Bash；替换为当前设备的实际 BDF
bdf=0000:03:00.0
dev_dir="/sys/bus/pci/devices/$bdf"
if [ -d "$dev_dir" ]; then
    pci_path=$(readlink -f -- "$dev_dir")
    while [[ "$pci_path" == /sys/devices/* ]]; do
        component=${pci_path##*/}
        if [[ "$component" =~ ^[[:xdigit:]]{4}:[[:xdigit:]]{2}:[[:xdigit:]]{2}\.[0-7]$ ]]; then
            printf '\n%s\n' "$component"
            lspci -D -s "$component" -nnk
        fi
        pci_path=${pci_path%/*}
    done
else
    printf '%s\n' '当前设备路径不存在，无法由它还原已经丢失的父路径'
fi
```

当设备已消失，当前目录无法恢复它过去的祖先。此时正常状态的 BDF/父路径记录、同分支仍存活设备、内核日志和平台图比反复查询一个不存在地址更有用。命令执行期间也可能发生移除，单次多命令结果不保证是原子快照，应保留采集时间并对矛盾信息复查。

## 5. 为什么 function 数量和通信拓扑不能直接当硬件清单

多功能设备可以在相同 bus/device 下暴露 `.0`、`.1` 等 function；这通常意味着多个可独立配置的功能，不代表多张板卡。SR-IOV 又引入 PF（Physical Function）和 VF（Virtual Function）：启用 VF 后 Linux 可将其作为独立 PCI 设备处理，数量变化不需要插入新的物理卡。PF/VF 关联可以通过 `physfn`、`virtfn*` 等 sysfs 链接核对；不要假定 VF 总是同一 device 下简单递增 function，也不要在本观察练习中写入 `sriov_numvfs`。[Linux SR-IOV Howto](https://docs.kernel.org/PCI/pci-iov-howto.html)

```bash
# 仅查看可能存在的 PF/VF 关联；dev_dir 使用上一段实际设备路径
if [ -d "$dev_dir" ]; then
    for related in "$dev_dir"/physfn "$dev_dir"/virtfn*; do
        [ -L "$related" ] || continue
        printf '%s -> ' "${related##*/}"
        readlink -f -- "$related"
    done
fi
```

NVLink/NVSwitch 构成的 GPU fabric 也不是 PCIe 树的另一种打印格式。GPU 可以在 PCI 树上相距较远，却通过 NVLink 相连；同一 Switch 下的 GPU 与 NIC 则仍需检查 P2P 可达性和软件选择。PCI 最近共同祖先能帮助识别共享路径，却不足以保证所有 peer traffic 都在这个祖先处直接转发，ACS、Root Complex 和平台支持等条件会影响结果，后续在资源与传输主题中展开。

因此 [[02-AISystem/cluster-and-hardware/单机拓扑分析|已有单机拓扑 XML]] 应与 `lspci`、sysfs 和实际平台信息交叉验证。通信软件为了建模可能省略部分 Bridge、合并节点或引入 GPU 间专用互联边。该 XML 的采集程序与环境尚未知，本篇不将其端口数、物理芯片数或链路能力推断当作已验证事实，也不改写原文。

## 6. 用共同上游缩小故障范围

沿教学树假设 GPU `03:00.0` 不可见，而 NIC `04:00.0`、upstream `01:00.0` 和两个 downstream Bridge 都仍可见。这个组合将关注点缩小到 GPU 分支，但不能仅凭 NIC 正常就证明共享 Switch 内部全部正常。应进一步看 GPU 直接父端口的链路状态、相应时间段日志、供电/复位及设备身份变化。如果两个 endpoint 同时缺失，而 upstream 仍在，则需检查更广的下游共同条件；如果整个 root 分支都缺失，则先核对根层次、环境视图和平台描述，不能从末端 GPU 驱动开始猜。

| 观察组合 | 优先检查的边界 | 仍不能断言 |
| --- | --- | --- |
| 原 BDF 不见，但稳定身份出现在新 BDF | 总线重新编号、配置或环境变化；比较新旧父路径 | 不能仅按旧地址判定硬件丢失 |
| endpoint 目录存在，driver 不存在 | 功能驱动匹配/绑定阶段 | 不能认为没枚举，也不能认定 probe 已尝试 |
| 父 Bridge 存在，单个子分支缺失 | 子链路、设备准备状态、移除事件或配置访问 | 不能仅由父存在证明链路健康 |
| 多个 sibling 同时缺失 | 共同上游及共享供电/复位域，比较正常快照 | 不能确定唯一损坏器件 |
| 端口宽度或速率低于预期 | 对端、平台布线、负载/电源状态与链路日志 | 不能只由 `LnkCap` 推定当前可用吞吐 |
| GPU 与 NIC 共同祖先很近，通信仍失败 | 驱动/fabric/P2P 与选路，结合日志 | 不能由 PCI 树单独证明 GDR 或 NCCL 正常 |

对已知 BDF 可先查看本次启动内核日志，再沿已确认祖先 BDF 搜索。日志可能只记录父端口而未记录目标 endpoint；时间窗口和完整上下文比只过滤一行更重要。没有正常快照时，应把“预期这里有设备”的依据写成平台图或历史信息，不能从当前缺失路径本身推导设备原来接在哪里。

## 来源、适用范围与后续

2026-09-13 实际使用：Linux PCI Host Controller 与 PCI sysfs 文档支持根层次和设备视图；Linux v6.12 `pci_regs.h`、`include/linux/pci.h` 支持 Bridge/port type 与地址编码；pciutils `master` 的 `lspci.man` 支持工具参数与显示边界；Linux SR-IOV Howto 支持 PF/VF 概念及关联。本轮尝试获取 pciutils v3.13.0 手册未成功，因此不声称已核查该固定版本，采用文内明确链接的滚动手册；命令可用性以实际安装版本为准。

本文没有下载或逐章核查 PCI-SIG 完整规范，也没有开展固定版本枚举源码调用链分析。总线表、ASCII 树和 Debug 场景是教学构造，真实服务器、driver/firmware 版本与历史 XML 来源仍未知。本文所有 Linux 命令未实机运行；正文自检不代表平台验证。

下一篇见 [[02-AISystem/cluster-and-hardware/从上电到设备枚举：Firmware与Linux的职责边界|从上电到设备枚举：Firmware 与 Linux 的职责边界]]，对应第 3 阶段：串起供电/复位、BIOS/UEFI、设备 firmware、ACPI、PCI 核心与驱动接管，解释为什么设备可能停在不同的启动边界。BAR 和传输机制随后进入第 4 阶段。
