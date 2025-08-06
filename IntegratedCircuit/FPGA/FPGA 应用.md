---
dateCreated: 2021-01-26
dateModified: 2025-04-09
---
# 图像处理

## 1. 图像基础

采样、量化、处理（空域、频域、编码压缩、增强复原、分割、描述、识别）

原始图像获取：CCD 以 MOS 电容器为基础；CMOS 使用光电二极管。CMOS 耗电低，成本低，灵敏度低，分辨率较差，噪声多。

色彩空间：RGB- 工业标准，包含人类视力；CMYK，相减混色，用于打印；His，通过色调、饱和、亮度三种特征来感知颜色；YUV 欧洲电视编码；YCbCr- 世界数字组织，亮 - 蓝色 - 红色

图像压缩：JPG/GIF/PNG/H.261/MPEG-1/H.262/AVS

## 3. 视频流与接口

video 时序生成。

不直接采用视频信号：

1. 异步时钟域转换：本地处理和视频源往往不是一个时钟驱动，输入像素时钟往往不足够功率来驱动本地逻辑，采用 PLL 生成时钟驱动能更强。
2. 图像格式解析，对视频数据合并。
3. 色度重采样。
4. 处理辅助信息。

## 4. 直方图

灰度直方图描述了图像的灰度统计信息。

均衡：通过灰度映射使输入图像在每一级都有近似相同的灰度。解决曝光不足或过度。

拉伸：增强灰度对比度，增强效果。

## 5. 线性、非线性滤波器

根据输入输出之间是否有唯一确定的传输函数，可以分为线性、非线性滤波器。

线性：均值、高斯等平滑滤波器；sobel/laplas/梯度锐化滤波器

处理方法通常是利用一个指定尺寸的 mask/filter/kernel/template/window 卷积；也可以频域直接相乘

### 均值

邻域均值，减小灰度的尖锐变化以减小噪声，但是也会导致边缘细节丢失。

高斯滤波：$h(x,y)=\frac{1}{2\pi\sigma^2} e^{-(x^2+y^2)/{2\sigma^2}}$，将窗口大小取为奇数，滤波器宽度由参数 $\sigma$ 决定，越大频带越宽，平滑程度越好

- 均值的平滑力度更大，保留细节的能力不如高斯。

sobel 算子，用于提取边缘：

$$
G_x = 
\left[
\begin{matrix}
-1 & 0 & +1 \\
-2 & 0 & +2 \\
-1 & 0 & +1
\end{matrix} \tag{sobel}
\right],
G_y = G_x^T\\
\theta = arctan(g_y/g_x)
$$

滤波 +sobel，消除了很多伪边缘

离散傅里叶变换也是线性变换：

$$
g(u,v)=1/MN \sum_M \sum_N f(x,y)e^{-2\pi i(ux/M + vy/N)}
$$

高频代表细节和纹理，低频描述了轮廓。傅里叶变换可用于空域模板滤波。FFT 变换：$(-1)^{x+y}$ 进行中心变换，FFT 后，滤波器函数相乘，反 FFT 取实部。

滤波器划分，有高斯、巴特沃斯、切比雪夫。

## 7. 图像分割

基于阈值分割：

全局，使用同一阈值划分，对噪声敏感

最大类间方差分割 OTSU 自适应分割，按照灰度特性，将图像分为背景和目标

# 数字信号处理

## 计算方法

1. 数字表示：无符号整数、有符号数、二进制补码、二进制反码、减一（比补码都 -1）、有符号数字（SD，非零元素最少的为正则有符号数 CSD）、浮点数、加法器、乘法器、除法器、MAC（分布式算法 DA，转换成各位相乘的结果，用 LUT 累加实现）、CORDIC、多项式的逼近、快速幅度逼近（通过 x,y 逼近 $r=\sqrt{x^2+y^2}$）
2. 数字滤波器：FIR、IIR、CIC
3. FFT
4. 通信系统：编码、调制
5. 自适应：LMS 和 RLS 最常用调整算法
6. 微处理器：
7. 图像和视频：边沿检测 SOBEL、高斯拉普拉斯算子、CANNY 算子

# FPGA 结构

基于 LUT，包括可编程 逻辑块 CLB，可编程输入输出 IOB，时钟管理模块 DCM，嵌入式 RAM。

CLB 由若干查找表和附加逻辑组成，可以实现组合、时序逻辑，可以配置为分布式 RAM/ROM。采用 SRAM 工艺实现。

IOB：根据不同电气特性划分为组。

DCM 包括数字频率合成 DFS，数字移相器 DPS，数字频率扩展器 DSS

# 深入浅出

## FPGA 结构

FPGA&ASIC：ASIC 定制

FPGA&CPLD：CPLD 基于 ROM，掉电不丢失；FPGA 基于 RAM，需要配置 ROM。CPLD 硬件相对复杂；CPLD 资源不可能大，适合简单开发。

altera cla 由 LE 构成，xilinx CLB 由 LC（logic cell）构成，IOB（I/O Bank）是逻辑与外部的接口，从基本的 LVTTL/LVCMOS 到 PCI。FPGA 内部也会扩展 PLL、RAM、乘法器之类的块。

## 代码

同步与异步复位：

异步不增加器件；同步降低了亚稳态出现的概率，但需要专门的端口。

异步复位同步释放：

PLL 复位后，locked 与 rst_n 作为整个系统复位

- 面积和速度互换原则

乒乓操作：输入数据选择 2 选 1，数据缓冲 *2，输出数据 2 选 1，后续

串并转换：选用多个存储器或移位寄存器，因此 n 个移位需要 n 个周期，n 位宽输出。

流水线：原本一个周期完成的切割为多周期

逻辑复制、模块复用：

模块化设计：

始终设计：1 内部逻辑产生时钟；2 分频时钟与使能时钟；3 门控时钟

# FPGA 并行编程

对我们的输入程序作出以下规范：

- 不使用动态内存分配（不使用 malloc(),free(),new 和 delete()）
- 减少使用指针对指针的操作
- 不使用系统调用（例如 abort(),exit(),printf()），我们可以在其他代码例如测试平台上使用这些指令，但是综合的时候这些指令会被无视（或直接删掉）
- 减少使用其他标准库里的内容（支持 math.h 里常用的内容，但还是有一些不兼容）
- 减少使用 C++ 中的函数指针和虚拟函数
- 不使用递归方程
- 精准的表达我们的交互接口

片上晶管的繁多也丰富了我们的预配资源，片上的处理器其实就是一个很好的代表。现如今的高端 FPGA 会含有 4 个甚至更多的微处理器（比如 ARM 核心），小型的 FPGA 上附有一个处理器也变得很常见。处理器使芯片有了运行操作系统（比如 Linux）的能力，它可以通过驱动和外部设备交流，可以运行更大的软件包比如 OpenCV，可以运行更高级的语言（比如 python）并以更快的速度运行。处理器经常成为了整个系统的控制者，协调了各方之间的数据转移，也协调了各个 IP 核心（包括用 HLS 自定义的 IP 核和第三方 IP 核）和板上资源的关系。

## 卷积神经网络加速

计算延迟主要集中在卷积层，设计专用加速器

$$
Out[cho][r][c]=
$$

# FPGA 开发

zynq Soc arm 双核 cortex-A9 处理器和 Xillinx 7 系列 FPGA

xc7z010 lc 28k

FPGA 是 CLB”可配置逻辑模块，IOB“输入输出单元”，PIM“分布式可编程互联矩阵”。基于 SRAM 工艺，基本可编程单元几乎由 LUT，register 组成

## 1. 开发

xc7z010clg400-1

分析 (Elaborated)：顶层原理图视图

综合 (Synthesis)：

约束输入：xdc 文件

设计实现 (Implementation)：

下载比特流，硬件下载

在线逻辑分析仪：

功能仿真：

## 2. GPIO

按键输入，消抖。

使用定时器查询方法

## 3. IP 核

### MMCM/PLL

PLL，锁相环，是一种反馈控制电路，具有时钟倍频、分频、相位偏移、可编程占空比功能，特点是利用外部输入的参考信号控制环路内部震荡信号的频率和相位。因为可以实现信号频率的自动跟踪，所以 PLL 常用于闭环跟踪电路。

Xilinx 7 系列有时钟管理单元 CMT（10-2，20-4 个），每个 CMT 由 MMCM 和 PLL 各一个组成。

MMCM 是 PLL 超集，有更强大的相移功能，MMCM 主要用于驱动器件逻辑（CLB、DSP、ARM）。PLL 主要用于为内存接口生成所需时钟。

PLL 组成：前置分频计数器（D 计数器）、相位 - 频率检测器（PFD，Phase-Frequency Detecter）电路、电荷泵（Charge Pump）、环路滤波器（Loop Filter）、压控振荡器（VCO，Voltage Controlled Oscillator）、反馈乘法器计数器（M 计数器）和后置计数器（O1-O6 计数器）。

工作时，PFD 检测参考频率 Fref 和反馈频率 Feedback 之间的相位差和频率差，控制电荷泵和 VCO 将相位差转化为控制电压；VCO 根据不同的控制电压产生不同的震荡频率，从而影响 Feedback 信号的相位和频率。

反馈路径加入 M 计数器会使 VCO 震荡频率是 Fref 信号频率的 M 倍，Fref 等于 Fin 除于 D。$F_{ref}=F_{in}/{D},F_{VCO}=F_{in}*M/D,F_{OUT}=(F_{IN}*M/(N*O))$。

​	全局和区域 IO 和时钟资源来规划，clock management tiles(CMT) 提供时钟合成 (clock frequency synthesis)，倾斜校正 (deskew)，过滤抖动（jitter filtering）

​	每个 CMTs 包含一个 MMCM(mixed-mode clock manager) 和一个 PLL。如下图所示，CMT 的输 入可以是 BUFR，IBUFG，BUFG，GT，BUFH，本地布线（不推荐使用），输出需要接到 BUFG 或者 BUFH 后再使用

​	混合模式时钟管理器 (MMCM) MMCM 用于在与给定输入时钟有设定的相位和频率关系的情况下，生成不同的时钟信 号。MMCM 提供了广泛而强大的时钟管理功能，

​	数字锁相环 (PLL) 锁相环（PLL）主要用于频率综合。使用一个 PLL 可以从一个输入时钟信号生成多个时钟信号。

​	程序中添加了一个 ODDR 原语,使得 clk_wiz_0 的 BUFG 输出的时钟信号能够输出到 FPGA 的普通 IO。通过 ODDR 把两路单端的数据合并到一路上输出，上下沿同时输出数据，上沿 输出 a 路下沿输出 b 路；如果两路输入信号一路恒定为 1，一路恒定为 0，那么输出的信号实际上 就是输入的时钟信号 pll_clk_o。直接输出到普通 IO，直接通过普通逻辑资源连接。但这样 Clock 输出的时延 和抖动（Jitter）都会变差。

### RAM

BMG IP(Block Ram Generator)，可配置 RAM 和 ROM。真双口（True Dual-Port ram,TDP），两个端口都可以独立地对 BRAM 读写，伪真双口（Simple Dual-Port ram,SDP），其中一个只能读，一个只能写，单口 RAM，只能通过一个端口读写。

### ROM

.coe 例化

### FIFO

### XFFT

### DDR

double data rate SDRAM

MIG IP

## 4. 通信协议

### UART

@ug585 TRM

全双工异步收发控制器，ZYNQ 内包含两个，每个 UART 控制器支持可编程波特率发生器、64 字节接收 FIFO 和发送 FIFO、产生中断、RXD 和 TXD 信号环回设置和可配置的数据为长度、停止位、校验方式。

- Modem control signals: CTS, RTS, DSR, DTR, RI and DCD are available only on the EMIO interface
- includes control bits for the UART clocks,resets and MIO-EMIO signal mapping.
- Software accesses the UART controller registers using the APB 32-bit slave interface attached to the PS AXI interconnect.
- The IRQ from each controller is connectedto the PS interrupt controller and routed to the PL.

#### Mode Config

normal, automatic echo, local loopback and remote loopback.

如果只是用来打印，不需要任何配置即可。

### IIC

SCL、SDA 上拉到 3.3，标准速率 100kbits/s，快速 400bits/s，支持多机，多主控，但同一时刻只允许有一个主控。由数据线 SDA 和时钟 SCL 构成串行总线；每个电路和模块都有唯一的地 址。

http://opencores.org/

## 5. 外设

### Ov5640

### HDMI

TMDS 差分编码传输，与 DVI 相同，发送端收到 RGB，

TMDS 每个链路包含三个传输通道和 1 个时钟信号通道，通过编码把 8 位视频、音频信号转换成最小化、直流平衡的 10 位数据。前八位原始，9 位指示运算，10 平衡。

1. 传输最小化 8 位数据经过编码和直流平衡得到 10 位最小化数据，这仿佛增加了冗余位，对传输链路的带 宽要求更高，但事实上，通过这种算法得到的 10 位数据在更长的同轴电缆中传输的可靠性增强了。下图是一个例子，说明对一个 8 位的并行 RED 数据编码、并/串转换。
2. 直流平衡。直流平衡（DC-balanced）就是指在编码过程中保证信道中直流偏移为零。方法是在原来的 9 位数据的后面加上第 10 位数据，这样，传输的数据趋于直流平衡，使信号对传输线的电磁干扰减 少，提高信号传输的可靠性。
3. 差分信号 TMDS 差分传动技术是一种利用 2 个引脚间电压差来传送信号的技术。传输数据的数值（“0” 或者“1”）由两脚间电压正负极性和大小决定。即，采用 2 根线来传输信号，一根线上传输原来 的信号，另一根线上传输与原来信号相反的信号。这样接收端就可以通过让一根线上的信号减去 另一根线上的信号的方式来屏蔽电磁干扰，从而得到正确的信号。

​	另外，还有一个显示数据通道（DDC），是用于读取表示接收端显示器的清晰度等显示能力 的扩展显示标识数据 (EDID) 的信号线。搭载 HDCP（High-bandwidth Digital Content Protection，高带 宽数字内容保护技术）的发送、接收设备之间也利用 DDC 线进行密码键的认证。

显示时序标准，坐上到右下。行频率，场频率。

### TFTLCD

### RGBLCD

# 嵌入式开发

开发流程：

1. 创建 vivado 工程；
2. 使用 IP integrator 创建 processor system；
3. 生成顶层 HDL；
4. 生成 bitsream 导入 SDK；
5. 在 SDK 创建应用工程；
6. 板级验证

## 1. IO

### 1.1 介绍

ZYNQ7 processing system:

1. PS-PL configuration: 配置 PS-PL 接口，包括 AXI、HP 和 ACP 总线接口。
2. peripheral IO pins: 不同 IO 选择 MIO/EMIO。
3. MIO configuration: 具体配置 MIO/EMIO。
4. clock configuration: 配置 PS 输入时钟、外设时钟，DDR、CPU 时钟等。
5. DDR configuration:DDR 控制器的配置信息
6. SMC timing calculation:SMC 时序
7. interrupts: 中断接口。

ZYNQ 分 PS 和 PL 部分，PS 通过 MIO 连接到 PS 引脚，通过 EMIO 连接 PL 引脚，分为 4 bank，引脚相对固定。

zynq7000 有 54 个 MIO，PS 外设可以用 MIO 访问，PS 通过 APB 总线对 GPIO 驱动，下·

### 1.2 硬件设计

PS-PL configuration：配置 UART 波特率，

peripheral IO pins：选择 UART 接口，GPIO MIO 接口，EMIO GPIO 接口。

​	PS 通过 MIO，EMIO 使用 PL 的 IP 核

MIO configuration：具体配置 MIO/EMIO。

clock configuration: 配置 PS 输入时钟、外设时钟，DDR、CPU 时钟等。

DDR configuration: memory part

CPU 666.666Mhz,DDR 533.33Mhz

generate output products; create HDL wrapper

使用 PL 的资源，需要导出 bitsream，I/O ports 对 PL 管脚分配。

mss:microprocessor software specification

### 1.4 板级验证

## 2. 中断

可屏蔽中断（IRQ）

不可屏蔽中断（NMI）

处理器间中断（IPI）

zynqPS 基于 Cortex-A9 处理器和 GIC pl390 中断控制器

GP AXI

AXI interconnect IP- 用于 AXI 存储器映射的主器件连接从器件，

PS reset IP- 接收 PS 输出异步复位信号，产生 PL 的复位信号，

## 3. AXI

### 分类

AXI-full：高性能（最大 256）

AXI-lite：简化版，低吞吐率

AXI-stream：高速流数据

（前两者）memory mapped: 主机在对从机进行读写操作时，指定一个目标地址，这个地址对应系统存储空间，进行读写操作

AXI 总线的 master 和 slave 的端口分为 5 个双向流量控制的通道，如下图所示。所谓双向流量控制是指发送端用 valid 表示数据是有效的，接收端用 ready 表示可以接受数据；只有在 valid 和 ready 同时为 1 时，数据才成功被传送。valid/ready 机制可以使发送接收双方都有能力控制传输速率。

无论是读写操作，AXI 总线支持，或者说基于突发传输。**单次 burst 传输中的数据，其地址不能跨越 4KB 边界**

- AXI4-Lite 有轻量级，结构简单的特点，适合小批量数据、简单控制场合。不支持批量传输，读写时一次只能读写一个字（32bit）,主要用于访问一些低速外设和外设的控制。
- AXI4 接口和 AXI-Lite 差不多，只是增加了一项功能就是批量传输，可以连续对一片地址进行一次性读写。也就是说具有数据读写的 burst 功能。
- AXI4-Stream 是一种连续流接口，不需要地址线（类似 FIFO），对于这类 IP，ARM 不能通过上面的内存映射方式控制（FIFO 没有地址线），必须有一个转换装置，例如 AXI-DMA 模块来实现内存映射到流式接口的转换。AXI4-Stream 本质都是针对数据流构建的数据通路。

AXI4-Lite 和 AXI4 均采用内存映射控制方式，即 ARM 将用户自定义 IP 编入某一地址空间进行访问，读写时就像在读写自己的片内 RAM，编程也方便，开发难度较低。代价就是资源占用过多，需要额外的读写地址线、读写数据线以及应答信号等。

AXI4-Stream 是一种连续流接口，不需要地址线（类似 FIFO），对于这类 IP，ARM 不能通过上面的内存映射方式控制（FIFO 没有地址线），必须有一个转换装置，例如 AXI-DMA 模块来实现内存映射到流式接口的转换。AXI4-Stream 本质都是针对数据流构建的数据通路。

### Axi 接口

三种 AXI 接口分别是：

AXI-GP 接口（4 个）：是通用的 AXI 接口，包括两个 32 位主设备接口和两个 32 位从设备接口，用过改接口可以访问 PS 中的片内外设。

AXI-HP 接口（4 个）：是高性能/带宽的标准的接口，PL 模块作为主设备连接（从下图中箭头可以看出）。主要用于 PL 访问 PS 上的存储器（DDR 和 On-Chip RAM）

AXI-ACP 接口（1 个）：是 ARM 多核架构下定义的一种接口，中文翻译为加速器一致性端口，用来管理 DMA 之类的不带缓存的 AXI 外设，PS 端是 Slave 接口。

PS-PL 接口只支持 AXI-lite 和 AXI，PL 的 AXI-Stream 不能直接与 PS 对接，需要经过 AXI4 转换。

几个常用的 AXI_stream 接口的 IP 介绍：

AXI-DMA：实现从 PS 内存到 PL 高速传输高速通道 AXI-HP<---->AXI-Stream 的转换

AXI-FIFO-MM2S：实现从 PS 内存到 PL 通用传输通道 AXI-GP<----->AXI-Stream 的转换

AXI-Datamover：实现从 PS 内存到 PL 高速传输高速通道 AXI-HP<---->AXI-Stream 的转换，只不过这次是完全由 PL 控制的，PS 是完全被动的。

AXI-VDMA：实现从 PS 内存到 PL 高速传输高速通道 AXI-HP<---->AXI-Stream 的转换，只不过是专门针对视频、图像等二维数据的。

## 4. DMA

### Pl Dma

AXI DMA：为内存和 AXI4-stream 外设之间提供了高带宽的直接内存访问，其可选的 S/G 功能可以将 CPU 从数据搬运功能中解放。

zynq 的 AXI_DMA 模块，该模块用到了三种总线，AXI4_Lite 用于对寄存器进行配置，AXI4 Memory Map 用于与内存交互，在此模块又分立出了 AXI4 Memory Map Read 和 AXI4 Memory Map Write 两个接口，又分别叫做 M_AXI_MM2S 和 M_AXI_S2MM。AXI4_Stream 接口用于对用户逻辑进行通信，其中 AXI4 Stream Master(MM2S) 是 PS to PL 方向，AMI4 Stream Slave（S2MM）是 PL to PS 方向。

direct register mode

编程顺序：

只需要较少的 FPGA 资源，通过访问 DMACR、原地址、目的地址和长度寄存发起 DMA 传输。

传输完成后，若使能产生了中断输出，DMASR 寄存器通道

开启/使能 MM2S 通道

使能中断

写一个有效的源地址得到 MM2SSA 寄存器。

写传输的字节数到 MM2S_LENGTH 寄存器

S/G 模式

传输的基本参数，存储在内存中。参数成为 BD

工作时，通过 SG 接口

### Axi-stream 实现

PS 开启 HP0 和 GP0 接口。

axi_dma 不具备数据缓冲的能力，高速数据传输时 PL 很难完全配合 PS 发送 DMA 指令的时机，因此需要使用 FIFO 进行数据缓冲，AXI DMA 和 AXI4stream data FIFO 在 PL 实现。

- 由于 PS 模块的 AXI 接口是地址映射接口，如果 PL 需要通过 axi_stream 与 PS 进行数据传输，则需要经由 dma 模块进行 axi_stream 与 axi_full 传输协议的相互转化
- 在 AXI 术语中，TVALID 指示有效的 TDATA 被正确响应表示一次 Burst，多个 Burst 组成 1 个 Packet，Packet 中用 TLAST 表示最后 1 个 Burst 对应的 TVALID 位置

### Ps Dma

微处理器不参与数据传输控制，但是让出总线，由 DMA 控制器完成。

DMA 控制器具有以下的特点：

n 8 个独立的通道，4 个可用于 PL—PS 间数据管理，每个通道有 1024Byte 的 MFIFO；

n 使用 CPU_2x 时钟搬运数据，CPU_2x =（CPU frq/6）*2；

n 执行自定义内存区域内的 DMA 指令运行 DMA；

n AHB 控制寄存器支持安全和非安全模式；

n 每个通道内置 4 字 Cache；

n 可以访问 SoC 的以下映射物理地址：

## 5. UART

全双工异步收发控制器，

每个 UART 包含波特率控制器、64 字节接收发送 FIFO、产生中断、RXD 和 TXD 环回模式和可配置数据位长度、停止位和校验方式，采用独立接受发送路径，中断支持轮询处理和中断驱动处理。

模式切换控制信号连接，总共有四种：正常、回音、本地环回、远程环回。

## 6. RAM、DDR

数据量少，地址不连续，长度不规则，可适用 BRAM

是 PL 的存储器阵列，PL 通过输出时钟、地址、读写控制；PS 通过 BRAM 控制器读写，支持 32 位，当使能 ECC，ECC 允许 AXI 主接口检测和纠正 BRAM 块中的单位、双位错误。AXI BRAM 支持单次传输、突发传输。

PS 通过 M_AXI_GP0 与 BRAM 控制器、PL 读取 BRAM 连接。

### DDR

PL 端的 IP 核作为主设备，通过 HP 接口，与 DDR 控制器通信，实现对 DDR3 的读写。

## 7. Timer

每个 Cortex-A9 都有各自独立的 32 位私有定时器和 32 位看门狗定时器，两个 CPU 共享一个 64 位全局定时器（GT），此外还有 24 位看门狗定时器和两个 TTC。TTC 用来计算来自 EMIO 的信号脉冲宽度，每个 TTC 都有三个独立的定时器。

私有定时器：CPU 频率的一半 -333.333Mhz。

#### Axi Timer

## 8 PS-XADC

## 9 PS-QSPI FLASH

## 10 SD 卡

## 11 双核 AMP

# Trouble Shooting

## 导出内存数据

```matlab
connect
target
target 2
cd D:/prj/
mrd -bin -file (0xfff)  (count)

//地址+长度

clc
data=fopen('D:\');
[wave,count]=fread(data,'uint16');
```

## 固化

sd 卡/FLASH

sdio coresight

xiffs 库

clean prj

# 疑难问题

- ordered port connections cannot be mixed with named port connections

检查括号内是否少/多了‘,’和‘.’

invalid use of undefined type ‘volatile struct sc_ctr_info’

描述

编译时报错

invalid use of undefined type ‘volatile struct sc_ctr_info’

找到相关的 struct 定义，不要再.c 中定义，将其放到.h 文件中定义，并且包含到使用到它的.c 文件中

- 程序运行时不断回到 main 从头开始运行
  描述

问题原因

函数指针未赋值，指向不明位置

- 卡死在 Xil_Assert
   程序卡死在 Xil_Assert
     或者 使用 XSCT 发送 stop 指令时，返回 cpu time out

问题原因

断言错误，出现可能

1、在连接中断句柄前就使能了中断

2、向自定义 FPGA 模块的寄存器读值，

另外，向未定义寄存器写值会导致程序卡死，因为 axi 总线一直不能成功

断言使用场景

同 C 断言，用于判断程序是否具备继续运行的条件，通常用于调试或者问题定位，以免程序运行时出现不必要的错误。

问题 5 FreeRTOS 修改 configure.h

在配置处直接修改，否则添加文件会在重新生成 BSP 时被修改。

- Bus Interface property FREQ_HZ does not match between /processing_system7_0/S_AXI_HP0(200000000) and /S_AXI_HP0(100000000)

在 external interface properties 选择 properties，修改 FREQ_HZ

# 数字通信同步

## 同步

同步技术：

1. 载波同步 - 接收端获取与调制载波同频同相的信号，以实现相干解调，分有载频和无载频；
2. 位同步 - 接收端提供一个作为取样判决用的位定时脉冲，重复频率与码元速率相同、相位与最佳判决时刻一致，一种需要满足最佳判决和定时，一种只需要定位脉冲（不涉及最佳取样时刻，码元中间）；
3. 帧同步 - 发送端提供帧起止，接收端获取标志位，提高性能需要搜索、校核、同步检查；
4. 网同步 - 通信网络保证低速与高速之间协调；
5. 扩频的伪码同步，

同步的实现：

1. 外同步：发送端发送导频，接收端提取导频作为同步信号，导频要在信号谱为零处插入，采用正交，避免对信号解调产生影响，便于提取，频域正交、双导频、时域
2. 自同步：发送端不发送专门信息。

   典型 - 从抑制载波的调制信号恢复载波，常用平方变换、平方环、同相正交环。

   位同步，主要有滤波、包络陷落、锁相环。滤波对不含同步信息的基带微分和全波整流，变成归零单极性脉冲。

## 数字滤波器

更高的精度和信噪比、但受到系统采样率限制。可分为经典滤波器和现代滤波器，现代滤波器把信号视为随机信号，利用统计特征推导估值算法，包括维纳滤波器、卡尔曼滤波器、线性预测滤波器、自适应滤波器。可份额为 IIR（无限脉冲响应）和 FIR（有限脉冲响应）。

$$
H(z)=\sum_{n=0}^{N-1} h(z)Z^{-n}\\\

H(z)=\frac{\sum_{i=0}^M b_iZ^{-i}}{1-\sum_{i=1}^N a_iZ^{-i}}
$$

FIR 不存在反馈，IIR 存在反馈；FIR 严格线性相位，IIR 无法实现线性相位，而且频率选择性越好，相位非线性越严重。

频率滤波器，变换域滤波

![滤波器参数](D:\study\ee\FPGA\滤波器参数.png)

⽬前，FPGA 的发明者——Xilinx 公司已推出 20 nm 的 UltraScale 器件，UltraScale 器件是 Xilinx 公司 Virtex、KintexFPGA 以及 3D IC 系列的扩展器件，不但可提供前所未有的系统集成度，同时还⽀持 ASIC 的系统级性能。Xilinx 正在开发第⼆代 SoC 和 3DIC 技术，以及下⼀代 FPGA 技术，其中包括 FPGA 性能/⽡的突破，以及与其下⼀代 Vivado 设计套件“协同优化”的器件。Xilinx 在系统中重新定义了⾼性能收发器的设计和优化，从⽽可以更有效地把 20nm 的附加价值引⼊已经验证的 28mn 技术之中，相信 FPGA 的应⽤会得到更⼤的发展。FPGA 的演进历程⽰意图如图 1-3 所⽰

FPGA 更适合于触发器丰富的结构，适合完成时序逻辑，因此在数字信号处理领域多使⽤ FPGA 器件。⽬前主流的 FPGA 仍是基于查找表技术的，但已经远远超出了先前版本的基本性能，并且整合了常⽤功能（如 RAM、时钟管理和 DSP）的硬核模块。如图 1-4 所⽰（图 1-4 只是⼀个⽰意图，实际上每⼀个系

列的 FPGA 都有其相应的内部结构），FPGA 芯⽚主要由 6 部分完成，分别为可编程输⼊/输出单元（Input/Output Block，IOB）、基本可编程逻辑块（Configurable Logic Block，CLB）、数字时钟管理模块（Digital Clock Manager，DCM）、嵌⼊式块 RAM（BlockRAM，BRAM）、丰富的布线资源、内嵌的底层功能单元和内嵌专⽤硬件模块。

### 数

定点数、浮点数，尽量用 IP 核

有效数据位：尽量减少无效位，N 位加法需要 N+1 位，M 位和 N 位乘法，需要 M+N 位

有限字长：AD 量化、有限位二进制、防止溢出和压缩电平。

​	AD 变换，e(n) 平稳随机序列；滤波器系数有限

常用运算模块：加法器 add/sub、乘法器 mul、复数乘法 complex multi、除法 div

### 滤波器

FIR 滤波器只在原点有几点，全局稳定，FIR 是一个抽头延迟加法器和乘法器集合，乘法器系数就是 FIR 系数，也被称为抽头延迟线。

相位：只有单位取样相应满足对称，才有线性相位。奇对称除了 M/2 群延时，还有 90° 相移，称为正交变换网络。

直接型、级联型、频率取样、快速卷积

窗函数法

IIR 具有高滤波效率，所需阶数低，但不具有严格线性相位。存在不为零的零点与基点，保证极点在单位圆内。

IIR 容易受有限字长效应，要保证稳定，容易震荡。FIR 通过卷积，更快速，IIR 可以用模拟滤波，但工作量大。

直接 I、直接 II、级联、并联。

巴特沃斯、切比雪夫、椭圆。

## 多速率

改变频率，抽取插值、低通。常用有多速率 FIR（少）、积分梳状 CIC、半带滤波，通常用 CIC 一级，抽取低通滤波，二级 FIR 半带滤波，节省资源。

多速率发送处理器：数字 _>RCF 可编程插值 FIR 滤波器 ->固定系数 FIR 滤波器 ->高速 CIC 插值滤波器 ->数控频率振荡器 NCO->DAC。RCF 一般采样 1-16 倍 256 阶，FFIR2 倍重采样，CIC2-5 阶，NCO 包括产生载波频率，完成数据调制分数乘法器。

多速率接收处理器：ADC->NCO->高速 CIC 抽取 ->FIR HB->FFIR->AGC->数字信号，

### CIC

用于数字下变频（DDC）、上变频（DUC），基于零点相消的 FIR 滤波器。

$$
H(z)=\frac{(1-z^{-M})}{1-z^{-1}}
$$

当 M 远大于 1，第一旁瓣相对于主瓣差值几乎固定 13.46dB，所以可以级联，增加旁瓣的衰减。

半带滤波器

## 自适应滤波器

根据输入信号的统计特征自动变化调整结构参数，可分为自适应算法与参数可调滤波器，可分为开环、闭环系统，可以是 FIR、IIR、格型。

1. 自适应干扰抵消：减去对噪声的相关估计，得到对有用信号的最佳估计。需要避免不相关的叠加，避免有用信号漏入滤波器被抵消。
2. 自适应预测：

## 变换域滤波器

时域无法滤波。

一个域的离散导致另一个域的周期延拓。

DFT：

$$
X(k)=\sum^{N-1}_{n=0} x(n)W^{kn}_N,1\leqslant k\leqslant N-1\\
x(n)=\frac{1}{N} \sum^{N-1}_{l=0} X(k)W^{-kn}_N,1\leqslant n\leqslant N-1
$$

存在的问题：

1. 栅栏效应和序列补零：DFT 只能给出频谱的 $\omega_k=2\pi k/N$ 分量，频谱采样值，不可能得到连续谱，称为栅栏效应。若序列较小，可以在序列后补零，以满足抽样间隔，使谱线加密，但是窗函数宽度不能变，要根据有效长度——而不是补零后的长度——选择窗函数。
2. 频谱泄露和混叠失真：时域会进行截短，频域会进行展宽，甚至导致超过奈奎斯特频率。减少泄露需要加窗函数，但是必须对数据重叠处理以补偿窗函数边缘对数据的衰减，例 hamming 进行 50% 重叠。
3. 分辨率与 DFT 参数选择。长度 N 的 DFT 变换，分辨率 $\Delta f=f_s/N$，N 为有效长度，例如果有 $f_1,f_2$ 两个信号，截断时分辨，需要满足 $2f_s/N<|f_1-f_2|$，补零不能提高分辨率。
