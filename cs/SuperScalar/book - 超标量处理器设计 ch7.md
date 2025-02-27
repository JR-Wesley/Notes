---
dateCreated: 2025-02-24
dateModified: 2025-02-26
---
# Ch7 寄存器重命名
## 7.1 概述

相关性（dependency）指一条指令的执行依赖于另一条指令的结果，可分为几类：

（1）数据相关性：

1. Output dependency/ write after write
2. Anti dependency/ write after read
3. True dependency/ read after write
（2）存储器数据的相关性（Memory data dependency），表示访问存储器的指令之间的相关性，也有以上三种。
（3）控制相关性（Control dependency），由于分支指令引起的相关性，使用分支预测解决。
（4）结构相关性（Structure dependency），指令必须等到处理器中某些部件可以使用时才能继续，如需要等发射队列（Issue queue）、重排序缓存（ROB）、功能单元（FU）空闲。
![](assets/ch7%20寄存器重命名/解决相关性.png)
通过更换寄存器可以解决 WAW/WAR 相关性，这两种不算真相关性，出现的原因可能有以下原因：
4. 有限数量寄存器。
5. 程序中的循环体，若反复写入同一寄存器，会出现大量 WAW，而有限的寄存器会导致某个时刻 WAW 不可避免，占用存储和 I-Cache 缺失升高。
6. 代码重用，一些小函数被频繁调用，即存在 WAW，虽然可以将函数嵌入调用程序中（inline）来解决，但仍然会又上述问题。
简单的增加存储器会导致处理器不兼容，也无法解决代码宠用产生的 WAW，最好的方法是使用硬件管理的寄存器重命名（Register Renaming）。处理器中实际存在的寄存器个数要多余 ISA 中定义的通用寄存器个数，这些内部实际存在的寄存器称为物理寄存器（Physical register），ISA 中定义的寄存器称为逻辑寄存器（Logical register, architecture register）。如 MIPS 定义 32 个通用逻辑寄存器，而经过重命名后使用的可以有 128 个物理寄存器，处理器动态地将前者映射到后者以解决假相关性。
![](assets/ch7%20寄存器重命名/使用重命名.png)
**重命名映射表（Register Renaming Table, Register Alias Table RAT）** 用来保存已有的映射关系。它是一个表格，可以基于 SRAM/CAM 实现。
**空闲寄存器列表（Free Register List）** 用来记录哪些物理寄存器是空闲的。

## 7.2 寄存器重命名的方法

实现方式有很多，概括有三种：

1. 将 ARF 扩展
2. 使用同一个 PRF
3. 使用 ROB
