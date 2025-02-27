
参考资料
RISCV 相关资料
https://blog.csdn.net/qq_43858116/article/details/123193844



参考笔记：
[从零开始写RISC-V处理器【1】前言 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/183294586#:~:text=%E4%BB%8E%E9%82%A3%E4%B9%8B%E5%90%8E%E4%B8%80%E4%B8%AA%E2%80%9C%E4%BB%8E%E9%9B%B6%E5%BC%80)

[从零开始写RISC-V处理器 | liangkangnan的博客 (scncyxf.github.io)](https://scncyxf.github.io/2020/04/29/%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%86%99RISC-V%E5%A4%84%E7%90%86%E5%99%A8/)
[深入浅出RISC-V调试 | liangkangnan的博客 (scncyxf.github.io)](https://scncyxf.github.io/2020/03/21/%E6%B7%B1%E5%85%A5%E6%B5%85%E5%87%BARISC-V%E8%B0%83%E8%AF%95/#:~:text=%E6%B7%B1%E5%85%A5%E6%B5%85%E5%87%BARISC-V)

# tinyriscv
tinyriscv SOC输入输出信号有两部分，一部分是系统时钟clk和复位信号rst，另一部分是JTAG调试信号，TCK、TMS、TDI和TDO。

## 架构

![[arch.jpg]]
### 概述
CPU执行一条指令一般有五个过程：取指（Instruction Fetch）、译码（Instruction Decode）、执行（Instruction Execute）、访存（Memory Access）、写回（Write-Back）
1. 取指：程序在编译烧录后会作为二进制的指令存放在rom中。CPU开始工作时PC寄存器（pc_reg.v）会产生指向rom中存放指令的地址(pc_o)。if_id.v作为取指到译码的中间模块，将指令信号打一拍后送到译码模块。
2. 译码：id.v根据指令内容，[解析指令](https://zhida.zhihu.com/search?content_id=211452731&content_type=Article&match_order=1&q=%E8%A7%A3%E6%9E%90%E6%8C%87%E4%BB%A4&zhida_source=entity)得到需要进行的操作，知道接下来该在哪里做什么事（如是否读寄存器、得到寄存器的值、地址），再将解析指令后得到的信号传递给译码与执行的中间模块id_ex.v，打一拍后输出到执行模块。
3. 执行：该阶段就是根据译码的结果执行相关操作了（如对通用寄存器的值做一些运算）。
4. 访存：如果译码识别出世内存访问指令，则将向总线发出访问内存请求，在执行阶段就会得到对应的数据了。
5. 写回：将执行的计算结果写回通用寄存器，作为临时数据存储。
### 取指
**pc_reg.v**：对指令寄存器的地址信号进行**复位、跳转、暂停、递增**操作，即是对地址进行处理，产生的值pc_o将被用作指令寄存器的地址信号从ROM中读取指令内容。

hold_flag_i是一个3位的暂停标志，由控制模块ctrl.v按优先级判断处理顺序后输出。这个暂停标志还会被if_id和id_ex模块使用，如果仅仅需要暂停PC寄存器的话，那么if_id模块和id_ex模块是不需要暂停的。当需要暂停if_id模块时，PC寄存器也会同时被暂停。当需要暂停id_ex模块时，那么整条流水线都会被暂停。

**if_id.v:** 将指令内容、指令地址、外设中断输入信号打一拍后输出。

### 译码
![[riscv指令字段.png]]
R-type:用于寄存器-寄存器操作。I-type:用于短立即数和访存load操作。S-type:用于访存store操作。B-type:用于条件跳转操作。U-type:用于长立即数。J-type:用于无条件跳转。
**id.v**: 在译码时首先根据ISA的设计，将32位的指令码解析为6个部分；接下来根据操作码opcode判断指令的类型，根据funct3、funct7判断具体是什么指令。以I-type的addi为例（将地址为rs1的源寄存器的值与立即数相加再写入目标寄存器）。当根据opcode、funct3判断为I类型的addi、slti、sltiu等执行相同的语句时，输出写通用寄存器信号、用到的寄存器的地址、不需要用到的信号赋默认值。

### 执行

**ex.v**: 执行模块做的工作主要是根据译码结果进行相应操作，包括逻辑运算、读写寄存器、输出跳转、暂停的标志与地址等。

在判断完具体指令为addi后，将无用的信号赋默认值，使能写内存标志、输出写寄存器数据为op1、op2的和，op1、op2即是译码阶段所取的源寄存器1的值和指令中的立即数。

RV32M是一个拓展指令集，添加了整数乘法和除法，因为在某些场合整数乘法和除法极少用到，这样做可以简化低端硬件实现。tinyriscv中乘除法都额外处理。

## 中断

RISC-V中断分为两种类型，一种是同步中断，即ECALL、EBREAK等指令所产生的中断，另一种是异步中断，即GPIO、UART等外设产生的中断。两者除了触发方式不同，处理方式也有区别。由ECALL、EBREAK指令引起的同步中断，如果执行阶段的指令为[除法指令](https://zhida.zhihu.com/search?content_id=211452731&content_type=Article&match_order=1&q=%E9%99%A4%E6%B3%95%E6%8C%87%E4%BB%A4&zhida_source=entity)，则先不处理同步终端，等除法指令执行完再处理，而有外设引起的异步中断可以打断除法指令执行。

**clint.v**：这是中断的核心管理模块，负责对中断输入信号进行仲裁，判断中断类型、切换写CSR寄存器状态、发送中断信号、流水线暂停标志等。
中断状态int_state一共分为四种状态:

```verilog
// 中断状态定义
localparam S_INT_IDLE            = 4'b0001;     //空闲中断
localparam S_INT_SYNC_ASSERT     = 4'b0010;     //同步中断
localparam S_INT_ASYNC_ASSERT    = 4'b0100;     //异步中断
localparam S_INT_MRET            = 4'b1000;     //中断返回
```

在机器模式下处理中断有八个CSR寄存器（Control and Status Register）：

- mtvec(Machine Trap Vector)：保存发生异常时处理器需要跳转到的地址。
- mcause(Machine Exception Cause)：发生异常的种类。
- mepc(Machine Exception PC)：指向发生异常的指令。
- mstatus(Machine Status)：保存全局中断使能，以及其他的状态。mstatus寄存器的详细信息可以查看RISC-V -Reade Chinese的第101页图10.4。
- mscratch(Machine Scratch)：它暂时存放一个字大小的数据。
- mie(Machine Interrupt Enable)：它指出处理器目前能处理和必须忽略的中断。
- mip(Machine Interrupt Pending)：它列出目前正准备处理的中断。
- mtval(Machine Trap Val)：它保存了陷入(trap)的附加信息：地址例外中出错的地址、发生非法指令例外的指令本身，对于其他异常它的值为0。

在clint.v中有四个always语句块，它们分别干了这些事情：

1. 中断仲裁，判断中断类型int_state，决定是否打断执行阶段的除法指令。
2. 写CSR寄存器状态切换。
3. 在发出中断信号前，先写几个必要的CSR寄存器。
4. 写完必要的CSR寄存器后，发送中断标志、中断入口地址给ex.v暂停读写寄存器，跳转到中断入口地址。
### 总线

总线负责数据的传输，从内核与外设的通信到访存请求的传递等信号都需要经过总线。假设各外设独占一条数据线一条地址线，那么结构会十分冗杂，总线的设计使内核只需要一条地址线和一条数据总线。目前有不少成熟、标准的总线，比如AMBA、wishbone、AXI等，tinyriscv自主设计了一种名为RIB的总线，支持多主多从但是同一时刻只能一主一从通信。RIB总线上的各个主设备之间采用固定优先级仲裁机制。


### 模块化

jtag_top：调试模块的顶层模块，主要有三大类型的信号，第一种是读写内存的信号，第二种是读写寄存器的信号，第三种是控制信号，比如复位MCU，暂停MCU等。

pc_reg：PC寄存器模块，用于产生PC寄存器的值，该值会被用作指令存储器的地址信号。

if_id：取指到译码之间的模块，用于将指令存储器输出的指令打一拍后送到译码模块。

id：译码模块，纯组合逻辑电路，根据if_id模块送进来的指令进行译码。当译码出具体的指令(比如add指令)后，产生是否写寄存器信号，读寄存器信号等。由于寄存器采用的是异步读方式，因此只要送出读寄存器信号后，会马上得到对应的寄存器数据，这个数据会和写寄存器信号一起送到id_ex模块。

id_ex：译码到执行之间的模块，用于将是否写寄存器的信号和寄存器数据打一拍后送到执行模块。

ex：执行模块，纯组合逻辑电路，根据具体的指令进行相应的操作，比如add指令就执行加法操作等。此外，如果是lw等访存指令的话，则会进行读内存操作，读内存也是采用异步读方式。最后将是否需要写寄存器、写寄存器地址，写寄存器数据信号送给regs模块，将是否需要写内存、写内存地址、写内存数据信号送给rib总线，由总线来分配访问的模块。

div：除法模块，采用试商法实现，因此至少需要32个时钟才能完成一次除法操作。

ctrl：控制模块，产生暂停流水线、跳转等控制信号。

clint：核心本地中断模块，对输入的中断请求信号进行总裁，产生最终的中断信号。

rom：程序存储器模块，用于存储程序(bin)文件。

ram：数据存储器模块，用于存储程序中的数据。

timer：定时器模块，用于计时和产生定时中断信号。目前支持RTOS时需要用到该定时器。

uart_tx：串口发送模块，主要用于调试打印。

gpio：简单的IO口模块，主要用于点灯调试。

spi：目前只有master角色，用于访问spi从机，比如spi norflash。
## pc_reg



# RVfpga


# 玄铁架构
[(43 封私信 / 81 条消息) Taurus - 知乎 (zhihu.com)](https://www.zhihu.com/people/taurus-54-7/posts)