---
dateCreated: 2025-04-04
dateModified: 2025-05-19
---

参考：<a href=" https://fducslg.github.io/Arch-2022Spring-FDU/%E5%AE%9E%E9%AA%8C%E7%8E%AF%E5%A2%83/">FDU arch</a>

MIPS CPU<a href="https://cjinfdu.github.io/ics24/">FDU ics</a>

pulp riscv core

https://cnrv.gitbooks.io/riscv-soc-book/content/ch8/sec1-PULP_overview.html

龙芯 cpu 设计

https://bookdown.org/loongson/_book3/

入门 pulp cv 32 e 40 p

cs 152 \mit 6.5900

pulp cva 6 简单乱序，scoreboard，带分支预测

cmu 18643

https://www.zhihu.com/people/li-zhi-rui-75/posts

# rvFPGA

rvfpga-el 2-v 3.0

[LinuxFoundationX：采用工业 RISC-V 内核的计算机架构 [RVfpga] |edX]( https://www.edx.org/es/learn/computer-programming/the-linux-foundation-computer-architecture-with-an-industrial-risc-v-core )

[Verilator User’s Guide — Verilator Devel 5.029 documentation](https://verilator.org/guide/latest/index.html)

- riscv 工具链和 openOCD 待完成

# ISA

RISC-V 指令集中的 **Privileged ISA（特权指令集架构）** 和 **Unprivileged ISA（非特权指令集架构）** 是两种不同权限级别的指令集，分别服务于用户级程序和系统级软件，其核心区别体现在功能定位、权限范围和适用场景上。以下是具体分析：

---

### **1. 功能定位与权限范围**
- **Unprivileged ISA（非特权指令集）**
  - **功能**：提供用户模式下应用程序执行所需的基础指令，包括算术运算（如 `ADD`、`SUB`）、逻辑操作（如 `AND`、`OR`）、数据传输（如 `LW`、`SW`）和控制流指令（如 `BEQ`、`JAL`）。
  - **权限限制**：无法直接访问硬件资源（如内存管理单元、中断控制器），也不能修改特权寄存器或执行影响系统全局状态的操作，确保操作系统的安全隔离。
- **Privileged ISA（特权指令集）**
  - **功能**：为操作系统内核或虚拟机监控程序提供底层硬件控制能力，包括中断管理、内存保护（如页表配置）、特权寄存器操作（如 `mstatus`、`mepc`）以及系统调用（如 `ECALL` 触发陷阱）。
  - **权限级别**：支持多级特权模式（如 Machine Mode、Supervisor Mode），允许在更高权限下直接操作硬件资源。

---

### **2. 设计目标与安全性**
- **Unprivileged ISA**
  - **目标**：为应用程序提供安全、稳定的执行环境，通过限制对硬件的直接访问，防止用户程序破坏系统稳定性。
  - **安全性机制**：依赖操作系统通过特权指令实现内存隔离和权限检查，例如用户程序无法绕过虚拟内存保护。
- **Privileged ISA**
  - **目标**：支持操作系统实现资源管理（如进程调度、内存分配）和硬件抽象（如设备驱动），同时提供虚拟化支持（如 Hypervisor 扩展）。
  - **安全性机制**：通过特权级切换（如从用户模式切换到内核模式）和硬件保护机制（如 TLB 控制），确保特权指令仅由可信代码执行。

---

### **3. 典型指令对比**

| **类别**      | **Unprivileged ISA 示例**    | **Privileged ISA 示例**       |
| ----------- | -------------------------- | --------------------------- |
| **算术/逻辑指令** | `ADD`, `SUB`, `XOR`, `AND` | 无（由非特权指令完成）                 |
| **内存管理指令**  | `LW`（加载数据）                 | `SFENCE.VMA`（刷新 TLB）        |
| **控制流指令**   | `JAL`（跳转并链接）               | `MRET`（从陷阱返回）               |
| **系统操作指令**  | 无                          | `ECALL`（触发系统调用）、`WFI`（等待中断） |
| **中断/异常处理** | 无                          | `CSRRW`（读写控制状态寄存器）          |

---

### **4. 应用场景**
- **Unprivileged ISA**
  - 用户应用程序开发（如算法实现、数据处理）。
  - 嵌入式系统中无操作系统的直接硬件控制（需依赖特定硬件扩展）。
- **Privileged ISA**
  - 操作系统内核开发（如内存管理、进程调度）。
  - 虚拟机监控程序（Hypervisor）实现多租户资源隔离。
  - 安全关键系统（如实时操作系统 RTOS）的中断响应和硬件抽象。

---

### **5. 架构扩展与模块化设计**

RISC-V 通过模块化设计允许 **可选扩展**：

- **Unprivileged ISA** 可扩展自定义指令（如向量指令 `V` 扩展），但需遵循用户模式权限规则。
- **Privileged ISA** 支持分层扩展（如 `S` 扩展支持 Supervisor Mode，`H` 扩展支持 Hypervisor），适应不同系统复杂度需求。

---

### **总结**
- **核心区别**：Privileged ISA 提供硬件直接控制能力，服务于系统软件；Unprivileged ISA 限制权限，专注于用户程序功能实现。
- **协同关系**：二者共同构成完整的 RISC-V 生态，非特权指令依赖特权指令实现资源管理和安全隔离。
- **发展前景**：RISC-V 通过模块化设计平衡灵活性与安全性，适用于从嵌入式设备到数据中心服务器的多样化场景。

如需进一步了解具体指令格式或特权架构实现细节，可参考 RISC-V 官方文档中的 [Unprivileged ISA](https://riscv.org/specifications/) 和 [Privileged ISA](https://riscv.org/specifications/privileged-isa/)。
