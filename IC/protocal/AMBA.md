---
dateCreated: 2024-10-20
dateModified: 2025-05-13
---

ARM document

https://developer.arm.com/Architectures/AMBA

博客

https://www.zhihu.com/column/c_1663245806869291008

https://blog.csdn.net/weixin_43698385/article/details/125895057

# 介绍
**总线**（Bus）是指计算机组件间规范化的交换数据（data）的方式，即以一种通用的方式为各组件提供数据传送和控制逻辑。大致可以将其分为片上总线和片外总线。其中**片外总线**一般指的是两颗芯片或者两个设备之间的数据交换传输方式，包括 UART、I2C、CAN、SPI 等。而 AMBA 总线为**片上总线**，即同一个芯片上不同模块之间的一种规范化交换数据的方式。
对于总线而言，有以下比较重要的性能指标：
- **总线带宽**：指的是单位时间内总线上传送的数据量；其大小为总线位宽 * 工作频率（单位为 bit，但通常用 Byte 表示，此时需要除以 8）。
- **总线位宽**：指的是总线有多少比特，即通常所说的 32 位，64 位总线。
- **总线时钟工作频率**：以 MHz 或者 GHz 为单位。
- **总线延迟**：一笔传输从发起到结束的时间。在突发传输中，通常指的是第一笔的发起和结束时间之间的延迟（什么事是突发传输后面再讲）。

# AMBA

AMBA 的全称为 Advanced Microcontroller Bus Architecture。

![[assets/AMBADev.jpeg]]

# APB(Advanced Peripheral Bus)
## 介绍

> The APB protocol is a low-cost interface, optimized for minimal power consumption and reduced interface complexity. The APB interface is not pipelined and is a simple, synchronous protocol. Every transfer takes at least two cycles to complete.

APB 用于访问外设的可编程控制信号寄存器。APB 与主存之间通过**APB bridge**连接，APB 的传输也由主存启动。**APB bridge**可连接多个 APB 外设，作为 **Requester**；**APB peripheral**相应请求，作为**Completer**。

## 信号描述

具体参考 AMBA APB Protocol Specification 2.1 AMBA APB signals。

- 单个地址总线 PADDR，用于读和写，字节地址，可以不对齐（但结果无法预测）
- 两个独立总线 PRDATA 读 PWDATA 写。可以是 8、16、32 比特，二者需要宽度一致。二者不能并行因为没有独立的握手信号。

## 传输
### 写
#### 无等待

PSEL 拉高，第一个周期准备，其他信号准备，PENABLE 为低；PENABLE 拉高，其他信号保持，PREADY 为高，传输；最后 PSEL 拉低，PENABLE 拉低，结束。

#### 等待

PENABLE 拉高时，PREADY 为低，则等待，其他信号保持直到 PREADY 拉搞，传输结束。

### 写掩码

一个掩码对应一字节。PSTRB[n] 对应 PWDATA[(8n + 7):(8n)]。

PSTRB 是可选的、可兼容的。

### 读

读传输同写，读数据必须在传输结束前提供。

### 错误反应

**PSLVERR**

## QA

## RTL

https://developer.arm.com/documentation/ddi0479/d/apb-components/apb-example-slaves

# AHB

https://blog.csdn.net/weixin_46022434/article/details/104987905

# AXI

https://blog.csdn.net/qq_57502075/article/details/130470954
