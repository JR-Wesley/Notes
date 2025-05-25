---
dateCreated: 2025-05-20
dateModified: 2025-05-21
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

ysyx 笔记：https://www.yizishun.com/?paged=4

# rvFPGA

rvfpga-el 2-v 3.0

[LinuxFoundationX：采用工业 RISC-V 内核的计算机架构 [RVfpga] |edX]( https://www.edx.org/es/learn/computer-programming/the-linux-foundation-computer-architecture-with-an-industrial-risc-v-core )

[Verilator User’s Guide — Verilator Devel 5.029 documentation](https://verilator.org/guide/latest/index.html)

- riscv 工具链和 openOCD 待完成

# 中科大

计算机程序的构造：

https://acsa.ustc.edu.cn/ics/

体系结构课：https://soc.ustc.edu.cn/CECS/

https://zhuanlan.zhihu.com/p/4096184482

组成原理课程：

https://soc.ustc.edu.cn/COD/

## 配置一览

- Verilator
- GTKWave
- RISC-V：riscv-unknown-linux-gnu 工具链
- x86：GCC
- 硬件：SystemVerilog
- 软件：C/C++, RISC-V Assembly

# 组成原理

<a href= "https://riscv-programming.org/book/riscv-book.html#pff">RISCV 编程教程</a>。

<a href=" https://soc.ustc.edu.cn/COD/lab1/src/rars1_6.jar">一个基于 Java 的 RISC-V 架构的汇编综合实验平台</a>程序提供了汇编器、仿真器（参考 ICS LabA&S）等功能，同时也包含了方便的外设接口与信息查询接口，被广泛用于 RISC-V 汇编程序的编写与测试之中。

# 体系结构
## 编译

riscv64-unknown-linux-gnu 交叉编译工具链是基于 GCC 的跨平台编译工具

### 测试 1

在 software/mytest 下，存放有一个冒泡排序 C 程序。该程序没有显式调用外部库，因此没有输入输出。

> [!note] 为何强调“显式调用外部库”？
事实上，即使不主动使用任何外部库函数，也不代表程序不需要链接任何库。在本课程开发过程中曾遇到过的一个问题就是很好的反例：
在指定架构为 rv32i（即基本的 RISC-V 整型指令集，没有乘/除法扩展）时编译带除法运算的 C 程序时，有时会出现找不到 libdiv 的链接错误。也就是说，尽管没有调用任何库，在某些情况下编译器也会链接到外部的库函数。
事实上，这也是我们要求手动编译安装特定版本 riscv64-unknown-linux-gnu 编译工具链的主要原因。

> [!note] 可以使用以下命令将该程序编译为 RISC-V 汇编代码
`riscv64-unknown-linux-gnu-gcc -S -march=rv32g -mabi=ilp32 -o test.s test.c`

其中的编译选项如下：

- `-S` 表示只输出汇编代码；
- `-march=rv32g` 指定了使用带全部扩展的 RISC-V 32 位整型指令集（以后简称为 RV32M 指令集）进行编译；
- `-mabi=ilp32` 指定了数据模型：整型（i）、长整型（l）和指针（p）均为 32 位，这是与前面的指令集相适应的数据模型；
- `-o` 指定了输出文件名为其后面的 test. s。

扩展名为 .S 的汇编文件支持预处理，而扩展名为 .s 的汇编文件不支持。一般地，由人工编写的汇编程序使用 .S 作为后缀，而由编译器或反汇编器生成的汇编程序使用 .s 作为后缀。

> [!note] riscv-unknown-linux-gnu 工具链提供了汇编工具，可以将汇编代码编译为目标文件（object）。
> `riscv64-unknown-linux-gnu-as -march=rv32g -mabi=ilp32 -o test. o test. s `

汇编器（Assembler, AS）将汇编文件翻译成机器语言指令，把这些指令打包成可重定位目标程序的格式，并将结果保存在目标文件中。输出的目标文件是一个二进制文件，可以直接被链接器使用，与其他目标文件、库文件一起链接成可执行文件。

> [!note] 如果你并不关心汇编代码，可以使用 GCC 将 C 程序直接编译为目标文件：
`riscv64-unknown-linux-gnu-gcc -c -march=rv32g -mabi=ilp32 -o test. o test. c `

其中 `-c` 选项表示只编译不链接。该指令同时也适用于将汇编文件编译为目标文件。

#### 命令解析

`riscv64-unknown-linux-gnu-gcc -S -march=rv32g -mabi=ilp32 -o test.s test.c` 是一条针对 RISC-V 架构的 C 语言编译命令，下面来详细解析。

- **版本**：`riscv64-unknown-linux-gnu-gcc` 属于 RISC-V 交叉编译器工具链。这里的 `64` 表明该工具链默认是面向 64 位架构的。不过，借助编译选项，它也能够支持 32 位架构。工具链的具体版本可以通过 `-v` 选项查看，像 `riscv64-unknown-linux-gnu-gcc -v` 这样。
- **用途**：此命令主要用于把 C 语言源代码（test. c）编译成 RISC-V 架构的汇编代码（test. s）。
编译选项详解
 - **`-S`**：此选项的作用是让编译器在生成汇编代码后就停止编译流程，不会继续进行汇编和链接操作。最终会生成一个扩展名为 `.s` 的汇编文件。
 - **`-march=rv32g`**：`-march` 用于指定目标处理器的架构。`rv32g` 代表 32 位的 RISC-V 架构，并且包含以下指令集：
	 - `I`：基础整数指令集，这是 RISC-V 架构必备的部分。
	 - `M`：整数乘除法扩展指令集，为处理器增加了乘除法运算能力。
	 - `F`：单精度浮点扩展指令集，支持单精度浮点数的运算。
	 - `D`：双精度浮点扩展指令集，在单精度浮点的基础上，进一步支持双精度浮点数运算。
	 - `G`：是 `IMFD` 的组合，意味着该架构支持上述所有扩展指令集。
- **`-mabi=ilp32`**：
	- `-mabi` 用于设定应用二进制接口（ABI）。
	- `ilp32` 适用于 32 位架构，具体含义如下：
		- `i`：表示 int 类型为 32 位。
		- `l`：代表 long 类型为 32 位。
		- `p`：指 pointer 类型为 32 位。该 ABI 使用寄存器 `a0-a7` 来传递函数参数，栈指针采用 `sp` 寄存器。
- **`-o test.s`**：`-o` 用于指定输出文件的名称。在这里，编译生成的汇编代码会被保存到 `test.s` 文件中。
- **`test.c`**：这是输入的 C 语言源代码文件。

特殊注意事项

- **工具链与架构的一致性**：这里存在一个值得注意的地方，工具链名称中的 `riscv64` 表示默认是 64 位环境，然而编译选项 `-march=rv32g -mabi=ilp32` 却指定了 32 位架构。这种组合在部分情况下可能会引发问题，比如在处理系统库依赖时。要是遇到这类问题，你可以考虑使用专门的 32 位工具链，像 `riscv32-unknown-linux-gnu-gcc`。- **应用场景**：该命令常用于开发需要运行在 32 位 RISC-V 处理器上的程序，或者用于教学，帮助大家理解 C 语言到汇编语言的转换过程。

#### 编译器

`riscv64-unknown-linux-gnu-gcc` 和 `riscv64-unknown-elf-gcc` 是两种不同的 RISC-V 交叉编译器，它们主要在目标环境、支持的系统库以及应用场景等方面存在差异，下面为你详细介绍：

### 1. 目标环境不同

- **riscv64-unknown-linux-gnu-gcc**：
    - 它是为运行 Linux 操作系统的 RISC-V 架构设备设计的编译器。
    - 该编译器生成的代码依赖于 Linux 系统提供的系统调用和 C 库（如 glibc）。
    - 适用于开发需要在完整 Linux 环境中运行的应用程序，像服务器软件、桌面应用等。
- **riscv64-unknown-elf-gcc**：
    - 这是用于编译裸机环境（没有操作系统）或 RTOS（实时操作系统）的编译器。
    - 生成的代码遵循 ELF 格式，但不依赖于 Linux 系统库，通常需要配合轻量级的 C 库（如 Newlib）使用。
    - 主要用于开发嵌入式系统、固件、引导加载程序等。

### 2. 系统库支持不同

- **riscv64-unknown-linux-gnu-gcc**：
    - 链接的是 GNU C 库（glibc），这是一个功能完整但体积较大的 C 库，提供了如文件操作、网络通信、内存管理等丰富的功能。
    - 支持动态链接（.so 文件）和系统调用（如 open、read、write 等）。
- **riscv64-unknown-elf-gcc**：
    - 通常链接轻量级的 C 库（如 Newlib）或不使用任何 C 库（-nostdlib 选项）。
    - 仅提供基本的标准库功能，如字符串处理函数，体积更小，适合资源受限的环境。

### 3. 应用场景不同

- **riscv64-unknown-linux-gnu-gcc**：
    - 适用于开发运行在 RISC-V Linux 系统上的应用程序，比如：
        - 服务器端应用。
        - 桌面应用程序。
        - 需要完整操作系统支持的复杂应用。
- **riscv64-unknown-elf-gcc**：
    - 适用于开发以下场景的软件：
        - 嵌入式系统（如开发板、微控制器）。
        - 操作系统内核、引导加载程序。
        - 实时系统（RTOS）。
        - 对体积和性能要求较高的固件。

### 4. 编译选项差异

- **riscv64-unknown-linux-gnu-gcc**：
    - 可以使用与 Linux 相关的编译选项，例如：
        - `-static`：静态链接 glibc 库。
        - `-fPIC`：生成位置无关代码，适用于共享库。
- **riscv64-unknown-elf-gcc**：
    - 通常需要使用与裸机环境相关的选项，例如：
        - `-ffreestanding`：编译独立环境代码（没有标准库支持）。
        - `-nostdlib`：不使用标准库。
        - `-T linker.ld`：指定链接脚本，用于控制内存布局。

### 5. 文件格式和依赖

- **riscv64-unknown-linux-gnu-gcc**：
    - 生成的可执行文件依赖于 Linux 内核和动态链接器（ld-linux-riscv64.so）。
    - 无法在没有 Linux 环境的系统上运行。
- **riscv64-unknown-elf-gcc**：
    - 生成的 ELF 文件可以直接加载到硬件上运行，或者通过调试工具（如 GDB）进行调试。
    - 可以在模拟器（如 QEMU）或实际硬件上运行，但需要适当的加载机制。

### 总结对比表

|特性|riscv64-unknown-linux-gnu-gcc|riscv64-unknown-elf-gcc|
|---|---|---|
|**目标环境**|Linux 操作系统|裸机或 RTOS|
|**C 库支持**|glibc（完整功能）|Newlib（轻量级）或无库|
|**动态链接**|支持|不支持|
|**系统调用**|支持|不支持|
|**应用场景**|Linux 应用程序|嵌入式系统、固件|
|**文件依赖**|需要 Linux 内核|无需操作系统|

### 选择建议

- 如果你要开发运行在 RISC-V Linux 系统上的应用程序，就选择 `riscv64-unknown-linux-gnu-gcc`。
- 如果你要开发嵌入式系统、操作系统内核或无操作系统环境下的软件，那么选择 `riscv64-unknown-elf-gcc`。

例如，在开发 RISC-V 开发板（如 SiFive Unleashed）上的 Linux 应用时，应使用 `-linux-gnu` 工具链；而开发同一块开发板的引导加载程序时，则应使用 `-elf` 工具链。
