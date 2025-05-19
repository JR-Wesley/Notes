[Diving into JTAG — Boundary Scan (Part 3) | Interrupt (memfault.com)](https://interrupt.memfault.com/blog/diving-into-jtag-part-3)
[JTAG 标准IEEE STD 1149.1-2013学习笔记（一）Test logic architecture_ieee std 1149.1-2013菊花链-CSDN博客](https://blog.csdn.net/qq_44447544/article/details/121925740#:~:text=JTAG%E6%98%AF%E8%8B%B1%E6%96%87%E2%80%9CJo)

# JTAG
## IEEE标准
### logic architecture
测试逻辑架构必须包含：
- 一个 TAP 控制器
- 一个指令寄存器 IR
- 一组测试数据寄存器 DR
![[Figure 5-1—Conceptual schematic of the on-chip test logic.png]]

下面简单介绍下此示意图：
（1）TAP 控制器接收TCK，TMS和TRST（可选）信号，产生 IR、DR和其他组件所需的时钟和控制信号，控制所要执行的操作，如复位、移位、捕获和更新等。
（2）IR 指令解码选择所要进行操作的DR
（3）TMP控制器是可选组件，可接收指令解码信号，用于修改TAP控制器产生的一些控制信号。

所以此测试逻辑，就是通过JTAG接口根据指令去对DR进行读操作或者写操作，写数据从TDI输入，读数据从TDO输出。

## IR 电路结构

指令寄存器的电路结构与全扫描类似，同样是采用了移位寄存器链。其扫描单元如图3所示：
![[IR.png]]


![[Figure 7-2—Instruction register with decoder between shift and update stages.png]]