
# Ch5 指令集体系

**指令集体系 （Instruction Set Architecture ISA）** 是规定处理器的外在行为的一系列内容的统称，包括基本数据类型 （data types）、指令（instructions）、寄存器 （registers）、寻址模式（addressing mode）、存储体系（memory architecture）、中断（interrupt）、异常（exception）、外部 IO（input/output）等内容。对一个 ISA 的硬件实现方式称为微架构（microarchitecture），如 Intel P 6/AMD K 7。
## 5.1 CISC/RISC
指令集从本质可以分为复杂指令集（Complex IS Computer, CISC）和精简指令集（Simple IS Computer, RISC）。CISC 的特点是能在一条指令内完成很多事。当时的年代，内存有限】寄存器少，倾向于使用高度编码、多操作数、长度不等的指令使一条指令做更多事。IBM 等公司发现，尽管 CISC 的特性让代码编写更便捷，这些复杂特性的指令需要好几个周期才能完成，而且大部分复杂指令都没有用到，同时寄存器的数量太少，导致处理器需要经常访问存储器，这会使效率变低。为了克服这些缺点，RISC 指令集降低处理器的复杂度，有更多面积放通用寄存器。它只需要包括常用的指令，减少面积、利于流水。RISC 使用了数量丰富的通用寄存器，所有操作都是在通用寄存器完成的，要和存储器交互需要使用专门访存的 load/store 指令。RISC 指令一般等长，简化解码电路设计，但相比 CISC 需要更多指令实现相同功能，导致占用更多程序存储器和 Cache 缺失。

## 5.2 RISC 介绍

### MIPS

### ARM
## 5.3 load/store 指令
## 5.4 计算指令

## 5.7 异常




# Ch6 指令解码


