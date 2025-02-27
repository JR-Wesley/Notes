# 图像处理

## 1. 图像基础

采样、量化、处理（空域、频域、编码压缩、增强复原、分割、描述、识别）

原始图像获取：CCD以MOS电容器为基础；CMOS使用光电二极管。CMOS耗电低，成本低，灵敏度低，分辨率较差，噪声多。

色彩空间：RGB-工业标准，包含人类视力；CMYK，相减混色，用于打印；HSI，通过色调、饱和、亮度三种特征来感知颜色；YUV欧洲电视编码；YCbCr-世界数字组织，亮-蓝色-红色

图像压缩：JPG/GIF/PNG/H.261/MPEG-1/H.262/AVS

## 2. 映射技术

计算：cordic

行缓存：buffer

## 3. 视频流与接口

video时序生成。

不直接采用视频信号：

1. 异步时钟域转换：本地处理和视频源往往不是一个时钟驱动，输入像素时钟往往不足够功率来驱动本地逻辑，采用PLL生成时钟驱动能更强。
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

处理方法通常是利用一个指定尺寸的mask/filter/kernel/template/window卷积；也可以频域直接相乘

### 均值

邻域均值，减小灰度的尖锐变化以减小噪声，但是也会导致边缘细节丢失。	

高斯滤波：$h(x,y)=\frac{1}{2\pi\sigma^2} e^{-(x^2+y^2)/{2\sigma^2}}$，将窗口大小取为奇数，滤波器宽度由参数$\sigma$决定，越大频带越宽，平滑程度越好

- 均值的平滑力度更大，保留细节的能力不如高斯。

sobel算子，用于提取边缘：
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
滤波+sobel，消除了很多伪边缘

离散傅里叶变换也是线性变换：
$$
g(u,v)=1/MN \sum_M \sum_N f(x,y)e^{-2\pi i(ux/M + vy/N)}
$$
高频代表细节和纹理，低频描述了轮廓。傅里叶变换可用于空域模板滤波。FFT变换：$(-1)^{x+y}$进行中心变换，FFT后，滤波器函数相乘，反FFT取实部。

滤波器划分，有高斯、巴特沃斯、切比雪夫。

### FPGA实现

一维：5个连续数据求和+增量更新

二维：5行缓存求和

sobel算子：行缓存，基于CORDIC

## 6. 形态学分割

## 7. 图像分割

基于阈值分割：

全局，使用同一阈值划分，对噪声敏感

最大类间方差分割OTSU自适应分割，按照灰度特性，将图像分为背景和目标



# 数字信号处理

## 计算方法

1. 数字表示：无符号整数、有符号数、二进制补码、二进制反码、减一（比补码都-1）、有符号数字（SD，非零元素最少的为正则有符号数CSD）、浮点数、加法器、乘法器、除法器、MAC（分布式算法DA，转换成各位相乘的结果，用LUT累加实现）、CORDIC、多项式的逼近、快速幅度逼近（通过x,y 逼近$r=\sqrt{x^2+y^2}$）
2. 数字滤波器：FIR、IIR、CIC
3. FFT
4. 通信系统：编码、调制
5. 自适应：LMS和RLS最常用调整算法
6. 微处理器：
7. 图像和视频：边沿检测SOBEL、高斯拉普拉斯算子、CANNY算子



# FPGA结构

基于LUT，包括可编程 逻辑块CLB，可编程输入输出IOB，时钟管理模块DCM，嵌入式RAM。

CLB由若干查找表和附加逻辑组成，可以实现组合、时序逻辑，可以配置为分布式RAM/ROM。采用SRAM工艺实现。

IOB：根据不同电气特性划分为组。

DCM包括数字频率合成DFS，数字移相器DPS，数字频率扩展器DSS







# 深入浅出

## FPGA结构

FPGA&ASIC：ASIC定制

FPGA&CPLD：CPLD基于ROM，掉电不丢失；FPGA基于RAM，需要配置ROM。CPLD硬件相对复杂；CPLD资源不可能大，适合简单开发。

altera cla由LE构成，xilinx CLB由LC（logic cell）构成，IOB（I/O Bank）是逻辑与外部的接口，从基本的LVTTL/LVCMOS到PCI。FPGA内部也会扩展PLL、RAM、乘法器之类的块。

软件无线电，SoC

## 代码

同步与异步复位：

异步不增加器件；同步降低了亚稳态出现的概率，但需要专门的端口。

异步复位同步释放：

PLL复位后，locked与rst_n作为整个系统复位



- 面积和速度互换原则

乒乓操作：输入数据选择2选1，数据缓冲*2，输出数据2选1，后续

串并转换：选用多个存储器或移位寄存器，因此n个移位需要n个周期，n位宽输出。

流水线：原本一个周期完成的切割为多周期

逻辑复制、模块复用：

模块化设计：

始终设计：1 内部逻辑产生时钟；2 分频时钟与使能时钟；3 门控时钟



FPGA跨时钟域处理

！！！！



测试文件：

？？？



## 时序分析

STA	

建立时间、保持时间。时钟抖动、偏斜

起点：源寄存器，终点：目的寄存器





## 系统架构





# FPGA并行编程

对我们的输入程序作出以下规范：

- 不使用动态内存分配（不使用malloc(),free(),new和delete()）

- 减少使用指针对指针的操作

- 不使用系统调用（例如abort(),exit(),printf()），我们可以在其他代码例如测试平台上使用这些指令，但是综合的时候这些指令会被无视（或直接删掉）

- 减少使用其他标准库里的内容（支持math.h里常用的内容，但还是有一些不兼容）

- 减少使用C++中的函数指针和虚拟函数

- 不使用递归方程

- 精准的表达我们的交互接口

片上晶管的繁多也丰富了我们的预配资源，片上的处理器其实就是一个很好的代表。现如今的高端FPGA会含有4个甚至更多的微处理器（比如ARM核心），小型的FPGA上附有一个处理器也变得很常见。处理器使芯片有了运行操作系统（比如Linux）的能力，它可以通过驱动和外部设备交流，可以运行更大的软件包比如OpenCV，可以运行更高级的语言（比如python）并以更快的速度运行。处理器经常成为了整个系统的控制者，协调了各方之间的数据转移，也协调了各个IP核心（包括用HLS自定义的IP核和第三方IP核）和板上资源的关系。

185308667

## DSA

领域专用架构设计（domain-spedific arrchitecture ）

- 根据应用的计算、访存特性设计专用硬件架构
- 提升性能、能效
- 面向一类应用而非某一个特定（ASIC）
- 机器学习神经网络加速器、图像处理GPU

DSA优势

- 更合理、高效的并行方案

- 访存带宽的专门优化
- 数值精度的合理选取
- 领域专用编程

目标领域——深度学习：计算量大、高度并发、模式简单、应用广泛

目标平台——FPGA：高灵活度、开发周期短、能效高

1960s perception->1980s ANN(MLP)->1990 SVM->1998 CNN->2006 DBN

Numerous DNN models

- ANN
- CNN
- RNN
- new variants:deconvolution, lstm, residual net, GAN

mainly two phases: training & inference

involvde operations:

- convolution: 6 nested loop(can be converted into MM)
- fully-connected: MV
- cativation: non-linear functions
- pooling: comparson or mean

requerements for DNN architecture

- latency/throughput/performance
- energy-dfficiency
- scalability
- generality



FPGA用于ASIC、domain-spefific design原型开发验证，嵌入式、云计算场景中提供低功耗、高能效的算力。

## pragma

### 硬件资源指定

```c
#pragma HLS RESOURCE variable=<temp>core=<Xilinx core>

a = b + c;
#pragma HLS RESOURCE variable=a core=AddSub_DSP
int A[1024];
#pragma HLS RESOURCE variable=A core=RAM_T2P_BRAM

```

1. 数组：

- 存储与片上存储单元buffer/scratchpad memory
- 设计利用数据局部性提高重复访问数据的访问效率
- FF:flip-flop
  - 单周期并行访问多地址
  - 单周期完成读写
  - 容量有限：typical 10 kbytes
- Block ram
  - 高容量：typical Mbytes
  - 访问性能受限，端口有限

```c
#pragma HLS array_partition variable = <name><type>factor= <int>dim=<int>
type: complete, cyclic, block
  dim=1, 2, 3, ...(dim=0)
//complete 完全散；block 分块； cyclic 取模分块
```

2. 数据精度

- 以数据类型实现

- 支持任意精度、任意位宽的有/无符号数据类型

  ap_uint, ap_int, ap_ufixed, ap_fixed

- 精度考虑乘、加法

3. 循环边界

- 一般为定值

  layency计量单位cycle

```c
#pragma HLS loop_tripcount min=<int>max=<int>avg=<int>
```

4. 循环展开

将循环体内部的代码复制多分（factor）

```c
for(int i = N - 1; i > 1; i = i - 2 )
{
  shift_reg[i] = shift_reg[i - 1];
  shift_reg[i - 1] = shift_reg[i - 2];
}
// 手工展开或使用
#pragma HLS UNROLL (factor = <int>)
```

一般和数组划分配合使用，提升架构并行度，同时占用更多硬件资源。

可能遇到问题：unroll代码很长或者份数多，造成运行时间长

5. 循环流水化

- 循环中多个iteration可以并发执行
- #pragma HLS PIPELINE(II=<int>)
- 优化iteration latency, loop latency, initiation interval
- 指定的facotr II可能无法实现

6. 数据流，也称任务及流水化

- 比pipeline粒度更大
- dataflow作用于不同的子函数
- dataflow后面间的代码只能包括函数调用和中间变量的声明

```c
void top(a, b, c, d)
{
  #pragma HLS dataflow
  func_A(a, b, i1);
  func_B(c, i1, i2);
  func_C(i2, d);
}
// 会被识别为buffer，按照乒乓的方式进行读写操作
```

7. stream

- 数据流中不同任务模块间的数据通路
- hls::stream<stream_type>
- 物理：以高性能FIFO代替高资源占用的double-buffer RAM

```cpp
#pragma HLS stream variable=<variable>depth=<int>dim=<int>
// 数据类型为stream，使用流操作符操作
// 函数参数配置时传引用&
// 必须有一个producer和consumer
```

8. 函数内联

- 等同直接嵌入代码

```c
#pragma HLS INLINE (off)
```

9. 模块接口

- 接口类型包括：函数级（函数调用传参）的接口实现方式、控制协议
- IP级别（顶层函数参数）的接口类型、控制协议

ap_fifo:标准fifo

m_axi:AXI4总线的master

s_axilite:axi4-lite的slave

10. 书写

- 不可使用动态内存分配

- 系统函数不可综合

- 标准库不支持

- 不要使用函数指针、虚函数（尽量C）

- 不可以递归函数调用

- 接口类型需明确定义、循环体优化，减少片外内存访问。

- 设计应将功能划分为多个子模块，分别优化

  与RTL自底向上一致，逐层优化

- dataflow切模块需要考虑不同stage的latency平衡优化，优化throughput



## 卷积神经网络加速

计算延迟主要集中在卷积层，设计专用加速器
$$
Out[cho][r][c]=
$$




# FPGA开发

zynq Soc arm双核cortex-A9处理器和Xillinx 7系列FPGA

xc7z010 lc 28k



FPGA是 CLB”可配置逻辑模块，IOB“输入输出单元”，PIM“分布式可编程互联矩阵”。基于SRAM工艺，基本可编程单元几乎由LUT，register组成

## 1. 开发

xc7z010clg400-1

分析(Elaborated)：顶层原理图视图

综合(Synthesis)：

约束输入：xdc文件

设计实现(Implementation)：

下载比特流，硬件下载

在线逻辑分析仪：

功能仿真：



## 2. GPIO

按键输入，消抖。

使用定时器查询方法

## 3. IP核

### MMCM/PLL

PLL，锁相环，是一种反馈控制电路，具有时钟倍频、分频、相位偏移、可编程占空比功能，特点是利用外部输入的参考信号控制环路内部震荡信号的频率和相位。因为可以实现信号频率的自动跟踪，所以PLL常用于闭环跟踪电路。

Xilinx 7系列有时钟管理单元CMT（10-2，20-4个），每个CMT由MMCM和PLL各一个组成。

MMCM是PLL超集，有更强大的相移功能，MMCM主要用于驱动器件逻辑（CLB、DSP、ARM）。PLL主要用于为内存接口生成所需时钟。

PLL组成：前置分频计数器（D计数器）、相位-频率检测器（PFD，Phase-Frequency Detecter）电路、电荷泵（Charge Pump）、环路滤波器（Loop Filter）、压控振荡器（VCO，Voltage Controlled Oscillator）、反馈乘法器计数器（M计数器）和后置计数器（O1-O6计数器）。

工作时，PFD检测参考频率Fref和反馈频率Feedback之间的相位差和频率差，控制电荷泵和VCO将相位差转化为控制电压；VCO根据不同的控制电压产生不同的震荡频率，从而影响Feedback信号的相位和频率。

反馈路径加入M计数器会使VCO震荡频率是Fref信号频率的M倍，Fref等于Fin除于D。$F_{ref}=F_{in}/{D},F_{VCO}=F_{in}*M/D,F_{OUT}=(F_{IN}*M/(N*O))$。



​	全局和区域IO和时钟资源来规划，clock management tiles(CMT)提供时钟合成(clock frequency synthesis)，倾斜校正(deskew)，过滤抖动（jitter filtering）

​	每个 CMTs 包含一个 MMCM(mixed-mode clock manager)和一个 PLL。如下图所示，CMT 的输 入可以是 BUFR，IBUFG，BUFG，GT，BUFH，本地布线（不推荐使用），输出需要接到 BUFG 或者 BUFH 后再使用

​	混合模式时钟管理器(MMCM) MMCM 用于在与给定输入时钟有设定的相位和频率关系的情况下，生成不同的时钟信 号。 MMCM 提供了广泛而强大的时钟管理功能，

​	数字锁相环(PLL) 锁相环（PLL）主要用于频率综合。使用一个 PLL 可以从一个输入时钟信号生成多个时钟信号。

​	程序中添加了一个 ODDR 原语,使得 clk_wiz_0 的 BUFG 输出的时钟信号能够输出到 FPGA 的普通 IO。通过 ODDR 把两路单端的数据合并到一路上输出，上下沿同时输出数据，上沿 输出 a 路下沿输出 b 路；如果两路输入信号一路恒定为 1，一路恒定为 0，那么输出的信号实际上 就是输入的时钟信号 pll_clk_o。直接输出到普通 IO，直接通过普通逻辑资源连接。但这样 Clock 输出的时延 和抖动（Jitter）都会变差。

## 

### RAM

BMG IP(Block Ram Generator)，可配置RAM和ROM。真双口（True Dual-Port ram,TDP），两个端口都可以独立地对BRAM读写，伪真双口（Simple Dual-Port ram,SDP），其中一个只能读，一个只能写，单口RAM，只能通过一个端口读写。



### ROM

.coe例化



### FIFO





### XFFT





### DDR

double data rate SDRAM

MIG IP 





## 4. 通信协议

### UART

@ug585 TRM

全双工异步收发控制器，ZYNQ内包含两个，每个UART控制器支持可编程波特率发生器、64字节接收FIFO和发送FIFO、产生中断、RXD和TXD信号环回设置和可配置的数据为长度、停止位、校验方式。

- Modem control signals: CTS, RTS, DSR, DTR, RI and DCD are available only on the EMIO interface
- includes control bits for the UART clocks,resets and MIO-EMIO signal mapping. 
- Software accesses the UART controller registers using the APB 32-bit slave interface attached to the PS AXI interconnect. 
- The IRQ from each controller is connectedto the PS interrupt controller and routed to the PL.



#### mode config

normal, automatic echo, local loopback and remote loopback.

如果只是用来打印，不需要任何配置即可。



### IIC

SCL、SDA上拉到3.3，标准速率100kbits/s，快速400bits/s，支持多机，多主控，但同一时刻只允许有一个主控。由数据线 SDA 和时钟 SCL 构成串行总线； 每个电路和模块都有唯一的地 址。

http://opencores.org/





## 5. 外设

### ov5640



### HDMI

TMDS差分编码传输，与DVI相同，发送端收到RGB，

TMDS每个链路包含三个传输通道和1个时钟信号通道，通过编码把8位视频、音频信号转换成最小化、直流平衡的10位数据。前八位原始，9位指示运算，10平衡。

1. 传输最小化 8 位数据经过编码和直流平衡得到 10 位最小化数据，这仿佛增加了冗余位，对传输链路的带 宽要求更高，但事实上，通过这种算法得到的 10 位数据在更长的同轴电缆中传输的可靠性增强了。 下图是一个例子，说明对一个 8 位的并行 RED 数据编码、并/串转换。

2. 直流平衡。直流平衡（DC-balanced）就是指在编码过程中保证信道中直流偏移为零。方法是在原来的 9 位数据的后面加上第 10 位数据，这样，传输的数据趋于直流平衡，使信号对传输线的电磁干扰减 少，提高信号传输的可靠性。

3. 差分信号 TMDS 差分传动技术是一种利用 2 个引脚间电压差来传送信号的技术。传输数据的数值（“0” 或者“1”）由两脚间电压正负极性和大小决定。即，采用 2 根线来传输信号，一根线上传输原来 的信号，另一根线上传输与原来信号相反的信号。这样接收端就可以通过让一根线上的信号减去 另一根线上的信号的方式来屏蔽电磁干扰，从而得到正确的信号。

​	另外，还有一个显示数据通道（DDC），是用于读取表示接收端显示器的清晰度等显示能力 的扩展显示标识数据(EDID)的信号线。搭载 HDCP（High-bandwidth Digital Content Protection，高带 宽数字内容保护技术）的发送、接收设备之间也利用 DDC 线进行密码键的认证。

显示时序标准，坐上到右下。行频率，场频率。





### TFTLCD





### RGBLCD













# 嵌入式开发

开发流程：

1. 创建vivado工程；
2. 使用IP integrator创建processor system；
3. 生成顶层HDL；
4. 生成bitsream导入SDK；
5. 在SDK创建应用工程；
6. 板级验证




## 1. IO

### 1.1 介绍

ZYNQ7 processing system:

1. PS-PL configuration:配置PS-PL接口，包括AXI、HP和ACP总线接口。
2. peripheral IO pins:不同IO选择MIO/EMIO。
3. MIO configuration:具体配置MIO/EMIO。
4. clock configuration:配置PS输入时钟、外设时钟，DDR、CPU时钟等。
5. DDR configuration:DDR控制器的配置信息
6. SMC timing calculation:SMC时序
7. interrupts:中断接口。

ZYNQ分PS和PL部分，PS通过MIO连接到PS引脚，通过EMIO连接PL引脚，分为4 bank，引脚相对固定。

zynq7000有54个MIO，PS外设可以用MIO访问，PS通过APB总线对GPIO驱动，下·



### 1.2 硬件设计

PS-PL configuration：配置UART波特率，

peripheral IO pins：选择UART接口，GPIO MIO接口，EMIO GPIO接口。

​	PS通过MIO，EMIO使用PL的IP核

MIO configuration：具体配置MIO/EMIO。

clock configuration:配置PS输入时钟、外设时钟，DDR、CPU时钟等。

DDR configuration: memory part

CPU 666.666Mhz,DDR 533.33Mhz



generate output products; create HDL wrapper

使用PL的资源，需要导出bitsream，I/O ports 对PL管脚分配。



mss:microprocessor software specification



### 1.3 软件设计

```c
init_platform();//使能caches和初始化uart
cleanup_platform();//取消使能caches
//针对特定平台，如microblaze

#include "xparameters.h" //器件参数信息
#include "xstatus.h"     //包含XST_FAILURE和XST_SUCCESS的宏定义
#include "xil_printf.h"  //包含print()函数
#include "xgpiops.h"     //包含PS GPIO的函数
#include "sleep.h"       //包含sleep()函数

//宏定义GPIO_DEVICE_ID
#define GPIO_DEVICE_ID      XPAR_XGPIOPS_0_DEVICE_ID
//连接到MIO的LED
#define MIOLED0    7     //连接到MIO7
XGpioPs Gpio;            // GPIO设备的驱动程序实例

int main()
{
    int Status;
    XGpioPs_Config *ConfigPtr;

    print("MIO Test! \n\r");
    ConfigPtr = XGpioPs_LookupConfig(GPIO_DEVICE_ID);
    //根据器件ID查找配置信息
    Status = XGpioPs_CfgInitialize(&Gpio, ConfigPtr,
                    ConfigPtr->BaseAddr);
    //初始化器件驱动
    if (Status != XST_SUCCESS){
        return XST_FAILURE;
    }
    //设置指定引脚的方向：0输入，1输出
    XGpioPs_SetDirectionPin(&Gpio, MIOLED0, 1);
    //使能指定引脚输出：0禁止输出使能，1使能输出
    XGpioPs_SetOutputEnablePin(&Gpio, MIOLED0, 1);

    while (1) {
        XGpioPs_WritePin(&Gpio, MIOLED0, 0x0); //向指定引脚写入数据：0或1
        sleep(1);                              //延时1秒
        XGpioPs_WritePin(&Gpio, MIOLED0, 0x1);
        sleep(1);
    }
    return XST_SUCCESS;
}
```

### 1.4 板级验证







## 2. 中断

可屏蔽中断（IRQ）

不可屏蔽中断（NMI）

处理器间中断（IPI）

zynqPS基于Cortex-A9处理器和GIC pl390中断控制器

GP AXI

AXI interconnect IP-用于AXI存储器映射的主器件连接从器件，

PS reset IP-接收PS输出异步复位信号，产生PL的复位信号，

## 3. AXI

### 分类

AXI-full：高性能（最大256）

AXI-lite：简化版，低吞吐率

AXI-stream：高速流数据

（前两者）memory mapped:主机在对从机进行读写操作时，指定一个目标地址，这个地址对应系统存储空间，进行读写操作

AXI总线的master和slave的端口分为5个双向流量控制的通道，如下图所示。所谓双向流量控制是指发送端用valid表示数据是有效的，接收端用ready表示可以接受数据；只有在vaild和ready同时为1时，数据才成功被传送。vaild/ready机制可以使发送接收双方都有能力控制传输速率。
无论是读写操作，AXI 总线支持，或者说基于突发传输。**单次 burst 传输中的数据，其地址不能跨越 4KB 边界**


- AXI4-Lite有轻量级，结构简单的特点，适合小批量数据、简单控制场合。不支持批量传输，读写时一次只能读写一个字（32bit）,主要用于访问一些低速外设和外设的控制。

- AXI4接口和AXI-Lite差不多，只是增加了一项功能就是批量传输，可以连续对一片地址进行一次性读写。也就是说具有数据读写的burst 功能。


- AXI4-Stream是一种连续流接口，不需要地址线（类似FIFO），对于这类IP，ARM不能通过上面的内存映射方式控制（FIFO没有地址线），必须有一个转换装置，例如AXI-DMA模块来实现内存映射到流式接口的转换。AXI4-Stream本质都是针对数据流构建的数据通路。

AXI4-Lite和AXI4均采用内存映射控制方式，即ARM将用户自定义IP编入某一地址空间进行访问，读写时就像在读写自己的片内RAM，编程也方便，开发难度较低。代价就是资源占用过多，需要额外的读写地址线、读写数据线以及应答信号等。

AXI4-Stream是一种连续流接口，不需要地址线（类似FIFO），对于这类IP，ARM不能通过上面的内存映射方式控制（FIFO没有地址线），必须有一个转换装置，例如AXI-DMA模块来实现内存映射到流式接口的转换。AXI4-Stream本质都是针对数据流构建的数据通路。



### axi接口

三种AXI接口分别是：

AXI-GP接口（4个）：是通用的AXI接口，包括两个32位主设备接口和两个32位从设备接口，用过改接口可以访问PS中的片内外设。

AXI-HP接口（4个）：是高性能/带宽的标准的接口，PL模块作为主设备连接（从下图中箭头可以看出）。主要用于PL访问PS上的存储器（DDR和On-Chip RAM）

AXI-ACP接口（1个）：是ARM多核架构下定义的一种接口，中文翻译为加速器一致性端口，用来管理DMA之类的不带缓存的AXI外设，PS端是Slave接口。

PS-PL接口只支持AXI-lite和AXI，PL的AXI-Stream不能直接与PS对接，需要经过AXI4转换。


几个常用的AXI_stream接口的IP介绍：

AXI-DMA：实现从PS内存到PL高速传输高速通道AXI-HP<---->AXI-Stream的转换
AXI-FIFO-MM2S：实现从PS内存到PL通用传输通道AXI-GP<----->AXI-Stream的转换
AXI-Datamover：实现从PS内存到PL高速传输高速通道AXI-HP<---->AXI-Stream的转换，只不过这次是完全由PL控制的，PS是完全被动的。
AXI-VDMA：实现从PS内存到PL高速传输高速通道AXI-HP<---->AXI-Stream的转换，只不过是专门针对视频、图像等二维数据的。





## 4. DMA

### pl dma	

AXI DMA ：为内存和AXI4-stream外设之间提供了高带宽的直接内存访问，其可选的S/G功能可以将CPU从数据搬运功能中解放。

zynq的AXI_DMA模块，该模块用到了三种总线，AXI4_Lite用于对寄存器进行配置，AXI4 Memory Map用于与内存交互，在此模块又分立出了AXI4 Memory Map Read和AXI4 Memory Map Write两个接口，又分别叫做M_AXI_MM2S和M_AXI_S2MM。AXI4_Stream接口用于对用户逻辑进行通信，其中AXI4 Stream Master(MM2S)是PS to PL方向，AMI4 Stream Slave（S2MM）是PL to PS方向。

direct register mode

编程顺序：

只需要较少的FPGA资源，通过访问DMACR、原地址、目的地址和长度寄存发起DMA传输。

传输完成后，若使能产生了中断输出，DMASR寄存器通道

开启/使能MM2S通道

使能中断

写一个有效的源地址得到MM2SSA寄存器。

写传输的字节数到MM2S_LENGTH寄存器



S/G 模式

传输的基本参数，存储在内存中。参数成为BD

工作时，通过SG接口



### axi-stream 实现

PS开启HP0和GP0接口。

axi_dma不具备数据缓冲的能力，高速数据传输时PL很难完全配合PS发送DMA指令的时机，因此需要使用FIFO进行数据缓冲，AXI DMA 和AXI4stream data FIFO 在PL实现。

- 由于PS模块的AXI接口是地址映射接口，如果PL需要通过axi_stream与PS进行数据传输，则需要经由dma模块进行axi_stream与axi_full传输协议的相互转化
- 在AXI术语中，TVALID指示有效的TDATA被正确响应表示一次Burst，多个Burst组成1个Packet，Packet中用TLAST表示最后1个Burst对应的TVALID位置
  

### ps dma

微处理器不参与数据传输控制，但是让出总线，由DMA控制器完成。

DMA控制器具有以下的特点：

n      8个独立的通道，4个可用于PL—PS间数据管理，每个通道有1024Byte的MFIFO；

n      使用CPU_2x 时钟搬运数据，CPU_2x = （CPU frq/6）*2；

n      执行自定义内存区域内的DMA指令运行DMA；

n      AHB控制寄存器支持安全和非安全模式；

n      每个通道内置4字Cache；

n      可以访问SoC的以下映射物理地址：





## 5. UART

全双工异步收发控制器，

每个UART包含波特率控制器、64字节接收发送FIFO、产生中断、RXD和TXD环回模式和可配置数据位长度、停止位和校验方式，采用独立接受发送路径，中断支持轮询处理和中断驱动处理。

模式切换控制信号连接，总共有四种：正常、回音、本地环回、远程环回。





## 6. RAM、DDR

数据量少，地址不连续，长度不规则，可适用BRAM

是PL的存储器阵列，PL通过输出时钟、地址、读写控制；PS通过BRAM控制器读写，支持32位，当使能ECC，ECC允许AXI主接口检测和纠正BRAM块中的单位、双位错误。AXI BRAM支持单次传输、突发传输。

PS通过M_AXI_GP0与BRAM控制器、PL读取BRAM连接。





### DDR

PL端的IP核作为主设备，通过HP接口，与DDR控制器通信，实现对DDR3的读写。





## 7. timer

每个Cortex-A9都有各自独立的32位私有定时器和32位看门狗定时器，两个CPU共享一个64位全局定时器（GT），此外还有24位看门狗定时器和两个TTC。TTC用来计算来自EMIO的信号脉冲宽度，每个TTC都有三个独立的定时器。

私有定时器：CPU频率的一半-333.333Mhz。

#### axi timer



## 8 PS-XADC



## 9 PS-QSPI FLASH



## 10 SD卡



## 11 双核AMP



# problem solving



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

sd卡/FLASH

sdio coresight

xiffs库

clean prj



# 疑难问题

- ordered port connections cannot be mixed with named port connections

检查括号内是否少/多了‘,’和‘.’



invalid use of undefined type ‘volatile struct sc_ctr_info’
描述
编译时报错
invalid use of undefined type ‘volatile struct sc_ctr_info’

找到相关的struct 定义，不要再.c 中定义，将其放到.h文件中定义，并且包含到使用到它的.c文件中



- 程序运行时不断回到main从头开始运行
  描述

问题原因
函数指针未赋值，指向不明位置



- 卡死在Xil_Assert
   程序卡死在Xil_Assert
     或者 使用XSCT 发送stop指令时，返回cpu time out

问题原因
断言错误，出现可能
1、在连接中断句柄前就使能了中断
2、向自定义FPGA模块的寄存器读值，
另外，向未定义寄存器写值会导致程序卡死，因为axi总线一直不能成功

断言使用场景
同C断言，用于判断程序是否具备继续运行的条件，通常用于调试或者问题定位，以免程序运行时出现不必要的错误。

问题5 FreeRTOS修改configure.h
在配置处直接修改，否则添加文件会在重新生成BSP时被修改。



- Bus Interface property FREQ_HZ does not match between /processing_system7_0/S_AXI_HP0(200000000) and /S_AXI_HP0(100000000)

在external interface properties选择properties，修改FREQ_HZ



- address editor





# 数字通信同步

## 同步

同步技术：

1. 载波同步-接收端获取与调制载波同频同相的信号，以实现相干解调，分有载频和无载频；
2. 位同步-接收端提供一个作为取样判决用的位定时脉冲，重复频率与码元速率相同、相位与最佳判决时刻一致，一种需要满足最佳判决和定时，一种只需要定位脉冲（不涉及最佳取样时刻，码元中间）；
3. 帧同步-发送端提供帧起止，接收端获取标志位，提高性能需要搜索、校核、同步检查；
4. 网同步-通信网络保证低速与高速之间协调；
5. 扩频的伪码同步，

同步的实现：

1. 外同步：发送端发送导频，接收端提取导频作为同步信号，导频要在信号谱为零处插入，采用正交，避免对信号解调产生影响，便于提取，频域正交、双导频、时域

2. 自同步：发送端不发送专门信息。

   典型-从抑制载波的调制信号恢复载波，常用平方变换、平方环、同相正交环。

   位同步，主要有滤波、包络陷落、锁相环。滤波对不含同步信息的基带微分和全波整流，变成归零单极性脉冲。

## 数字滤波器

更高的精度和信噪比、但受到系统采样率限制。可分为经典滤波器和现代滤波器，现代滤波器把信号视为随机信号，利用统计特征推导估值算法，包括维纳滤波器、卡尔曼滤波器、线性预测滤波器、自适应滤波器。可份额为IIR（无限脉冲响应）和FIR（有限脉冲响应）。
$$
H(z)=\sum_{n=0}^{N-1} h(z)Z^{-n}\\\

H(z)=\frac{\sum_{i=0}^M b_iZ^{-i}}{1-\sum_{i=1}^N a_iZ^{-i}}
$$
FIR不存在反馈，IIR存在反馈；FIR严格线性相位，IIR无法实现线性相位，而且频率选择性越好，相位非线性越严重。

频率滤波器，变换域滤波

![滤波器参数](D:\study\ee\FPGA\滤波器参数.png)

​	如图所⽰，低通滤波器的通带截⽌频率为ω P ，通带容限为α
1 ，阻带截⽌频率为ω S ，阻带容限为α 2 。通带定义为|ω |≤ω P ，
1−α 1 ≤|H (ejω )|≤1；阻带定义为ω S ≤|ω |≤π，|H (ejω )|≤α 2 ；
过渡带定义为ω P ≤ω ≤ω S 。通带内和阻带内允许的衰减⼀般⽤dB
来表⽰，通带内允许的最⼤衰减⽤α P 表⽰，阻带内允许的最⼩衰减
⽤α S 表⽰，α P 和α S 1分别定义为：
式中，|H (ejω 0 )|归⼀化为1。当 时，α P =3 dB，称
此时的ω P 为低通滤波器的3 dB通带截⽌频率。



FPGA技术与ASIC、DSP及CPU技术不断融合，
FPGA器件中已成功以硬核的形式嵌⼊ASIC、PowerPC处理器、ARM
处理器，以HDL的形式嵌⼊越来越多的标准数字处理单元，如PCI控制
器、以太⽹控制器、MicroBlaze处理器、Nios以及Nios Ⅱ处理器等。
新技术的发展不仅实现了软硬件设计的完美结合，也实现了灵活性与
速度设计的完美结合，使得可编程逻辑器件超越了传统意义上的FPGA
概念，并以此发展形成了现在流⾏的系统级芯⽚（System on
Chip，SOC）及⽚上可编程系统（System on a Programmable
Chip，SOPC）设计技术，其应⽤领域扩展到了系统级，涵盖了实时
数字信号处理技术、⾼速数据收发器、复杂计算以及嵌⼊式系统设计
技术的全部内容。



⽬前， FPGA 的发明者——Xilinx 公司已推出20 nm 的UltraScale器件，UltraScale器件是Xilinx公司Virtex、KintexFPGA以及3D IC系列的扩展器件，不但可提供前所未有的系统集成度，同时还⽀持ASIC的系统级性能。Xilinx正在开发第⼆代SoC和3DIC技术，以及下⼀代FPGA技术，其中包括FPGA性能/⽡的突破，以及与其下⼀代Vivado设计套件“协同优化”的器件。Xilinx在系统中重新定义了⾼性能收发器的设计和优化，从⽽可以更有效地把20nm的附加价值引⼊已经验证的28mn技术之中，相信FPGA的应⽤会得到更⼤的发展。FPGA的演进历程⽰意图如图1-3所⽰

FPGA更适合于触发器丰富的结构，适合完成时序逻辑，因此在数字信号处理领域多使⽤FPGA器件。⽬前主流的FPGA仍是基于查找表技术的，但已经远远超出了先前版本的基本性能，并且整合了常⽤功能（如RAM、时钟管理和DSP）的硬核模块。如图1-4所⽰（图1-4只是⼀个⽰意图，实际上每⼀个系
列的FPGA都有其相应的内部结构），FPGA芯⽚主要由6部分完成，分别为可编程输⼊/输出单元（Input/Output Block，IOB）、基本可编程逻辑块（Configurable Logic Block，CLB）、数字时钟管理模块（Digital Clock Manager，DCM）、嵌⼊式块RAM（BlockRAM，BRAM）、丰富的布线资源、内嵌的底层功能单元和内嵌专⽤硬件模块。



### 数

定点数、浮点数，尽量用IP核

有效数据位：尽量减少无效位，N位加法需要N+1位，M位和N位乘法，需要M+N位

有限字长：AD量化、有限位二进制、防止溢出和压缩电平。

​	AD变换，e(n)平稳随机序列；滤波器系数有限

常用运算模块：加法器add/sub、乘法器mul、复数乘法complex multi、除法div

### 滤波器

FIR滤波器只在原点有几点，全局稳定，FIR是一个抽头延迟加法器和乘法器集合，乘法器系数就是FIR系数，也被称为抽头延迟线。

相位：只有单位取样相应满足对称，才有线性相位。奇对称除了M/2群延时，还有90°相移，称为正交变换网络。

直接型、级联型、频率取样、快速卷积

窗函数法



IIR具有高滤波效率，所需阶数低，但不具有严格线性相位。存在不为零的零点与基点，保证极点在单位圆内。

IIR容易受有限字长效应，要保证稳定，容易震荡。FIR通过卷积，更快速，IIR可以用模拟滤波，但工作量大。

直接I、直接II、级联、并联。

巴特沃斯、切比雪夫、椭圆。


## 多速率

改变频率，抽取插值、低通。常用有多速率FIR（少）、积分梳状CIC、半带滤波，通常用CIC一级，抽取低通滤波，二级FIR半带滤波，节省资源。

多速率发送处理器：数字_>RCF可编程插值FIR滤波器->固定系数FIR滤波器->高速CIC插值滤波器->数控频率振荡器NCO->DAC。RCF一般采样1-16倍256阶，FFIR2倍重采样，CIC2-5阶，NCO包括产生载波频率，完成数据调制分数乘法器。

多速率接收处理器：ADC->NCO->高速CIC抽取->FIR HB->FFIR->AGC->数字信号，

### CIC

用于数字下变频（DDC）、上变频（DUC），基于零点相消的FIR滤波器。
$$
H(z)=\frac{(1-z^{-M})}{1-z^{-1}}
$$
当M远大于1，第一旁瓣相对于主瓣差值几乎固定13.46dB，所以可以级联，增加旁瓣的衰减。

半带滤波器

## 自适应滤波器

根据输入信号的统计特征自动变化调整结构参数，可分为自适应算法与参数可调滤波器，可分为开环、闭环系统，可以是FIR、IIR、格型。

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

1. 栅栏效应和序列补零：DFT只能给出频谱的$\omega_k=2\pi k/N$分量，频谱采样值，不可能得到连续谱，称为栅栏效应。若序列较小，可以在序列后补零，以满足抽样间隔，使谱线加密，但是窗函数宽度不能变，要根据有效长度——而不是补零后的长度——选择窗函数。
2. 频谱泄露和混叠失真：时域会进行截短，频域会进行展宽，甚至导致超过奈奎斯特频率。减少泄露需要加窗函数，但是必须对数据重叠处理以补偿窗函数边缘对数据的衰减，例hamming进行50%重叠。
3. 分辨率与DFT参数选择。长度N的DFT变换，分辨率$\Delta f=f_s/N$，N为有效长度，例如果有$f_1,f_2$两个信号，截断时分辨，需要满足$2f_s/N<|f_1-f_2|$，补零不能提高分辨率。
