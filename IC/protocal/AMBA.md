ARM document
https://developer.arm.com/Architectures/AMBA
博客
https://www.zhihu.com/column/c_1663245806869291008
# 介绍
**总线**（Bus）是指计算机组件间规范化的交换数据（data）的方式，即以一种通用的方式为各组件提供数据传送和控制逻辑。大致可以将其分为片上总线和片外总线。其中**片外总线**一般指的是两颗芯片或者两个设备之间的数据交换传输方式，包括UART、I2C、CAN、SPI等。而AMBA总线为**片上总线**，即同一个芯片上不同模块之间的一种规范化交换数据的方式。
对于总线而言，有以下比较重要的性能指标：
- **总线带宽**：指的是单位时间内总线上传送的数据量；其大小为总线位宽*工作频率（单位为bit，但通常用Byte表示，此时需要除以8）。
- **总线位宽**：指的是总线有多少比特，即通常所说的32位，64位总线。
- **总线时钟工作频率**：以MHz或者GHz为单位。
- **总线延迟**：一笔传输从发起到结束的时间。在突发传输中，通常指的是第一笔的发起和结束时间之间的延迟（什么事是突发传输后面再讲）。

# AMBA
AMBA的全称为Advanced Microcontroller Bus Architecture。
![[AMBADev.jpeg]]

# APB(Advanced Peripheral Bus)
## 介绍
>The APB protocol is a low-cost interface, optimized for minimal power consumption and reduced interface complexity. The APB interface is not pipelined and is a simple, synchronous protocol. Every transfer takes at least two cycles to complete.

APB用于访问外设的可编程控制信号寄存器。APB与主存之间通过**APB bridge**连接，APB的传输也由主存启动。**APB bridge**可连接多个APB外设，作为 **Requester**；**APB peripheral**相应请求，作为**Completer**。
## 信号描述
具体参考 AMBA APB Protocol Specification 2.1 AMBA APB signals。
- 单个地址总线 PADDR，用于读和写，字节地址，可以不对齐（但结果无法预测）
- 两个独立总线 PRDATA 读 PWDATA 写。可以是8、16、32比特，二者需要宽度一致。二者不能并行因为没有独立的握手信号。

## 传输
### 写
#### 无等待
PSEL拉高，第一个周期准备，其他信号准备，PENABLE为低；PENABLE拉高，其他信号保持，PREADY为高，传输；最后PSEL拉低，PENABLE拉低，结束。
#### 等待
PENABLE拉高时，PREADY为低，则等待，其他信号保持直到PREADY拉搞，传输结束。
### 写掩码
一个掩码对应一字节。PSTRB[n] 对应PWDATA[(8n + 7):(8n)]。
PSTRB是可选的、可兼容的。

### 读
读传输同写，读数据必须在传输结束前提供。
### 错误反应

**PSLVERR**


## QA


## RTL

https://developer.arm.com/documentation/ddi0479/d/apb-components/apb-example-slaves
