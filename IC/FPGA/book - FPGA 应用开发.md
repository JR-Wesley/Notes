---
dateCreated: 2024-07-12
dateModified: 2025-02-27
---
# Wong
# 数字电路

$$
Val = \sum_{i=-F}^{l-1} d_i \cdot 10^i
$$

两个位宽同为 W 位的二进制数 $a, b$ 的和为模即 $2^W$，称 a 和 b 互为补码

W 位二进制有符号数可表达的范围是

$$
[-2^{W-1}, 2^{W-1}-1]
$$

W 位二进制无符号数可表达的范围是

$$
[0, 2^{W}-1]
$$

因此，对于加减法运算，为了保证不溢出，结果必须扩展一位。对于乘除法情况要复杂一些。一般需要 2W 位完整表示

## 组合逻辑
### 编码器、译码器、数据选择器

独热码转二进制码为编码，通常有 4-2/8-3。实际常用的是优先编码器，即只要最低的第 n 位为 1，无论高位，输出为 n。数据选择从多个输入选择一个输出。

### 延迟和竞争冒险

从输入信号变化到输出信号变化的时间延迟为传输延迟。组合逻辑电路因为路径延迟导致输出出现暂时的不正确值的现象称为竞争冒险。竞争冒险可以通过在逻辑中分配冗余项、使用模拟滤波器等方法消除。但在 FPGA 中,基本只会使用同步时序逻辑电路。正常的同步时序逻辑电路天然地不受竞争冒险影响。

### 加法器、乘法器

异或门可以是一个一位半加器

一位全加器可以表示为：

$$
\begin{matrix}
Y = (A\oplus B)\oplus C_i\\
C_o = AB + (A\oplus B)C_i
\end{matrix}
$$

使用多个一位全加器可以构成多位全加器。减法器可以通过将减数求补码再与被减数相加来实现，而有符号加减与无符号加减法电路一致。

多位全加器的构成可以有多种形式。如纹波进位、超前进位。

乘法器可类似笔算顺序。

### 锁存器

组合逻辑的输出只与当前的输入值有关，而锁存器和触发器则能够记忆输入值。锁存器电平敏感。

### 触发器

时钟边沿触发。如果在时钟上升时刻的附近输入 D 正在发生变化,则主锁存器将长时间不能稳定到高电平或低电平，称为“亚稳态”。时钟有效沿前后的 $T_{SU},T_H$

## 时序逻辑
### 移位寄存器和串并转换

![[assets/数字电路/4位移位寄存器.png]]

n 个多位触发器即为延迟链。n 阶延迟链实现 z 域函数 $z^{-n}$

### 分频器

D 触发器可构成 2 分频器也即 T 触发器，n 级可构成 $2^n$ 分频

![[assets/数字电路/分频器.png]]

### 计数器

使用计数器使用加法器和触发器实现。进位输出将在计数器输出溢出的前一个周期出现。

![[assets/数字电路/进位计数器.png]]

![[assets/数字电路/进位计数器波形.png]]

有时需要计数器计到特定值后回到 0

![[assets/数字电路/模计数器.png]]

### 累加器

加法器的输入改为多位可构成累加器。

累加器在 DSP 中就是积分器，实现 z 域传输函数 $\frac{1}{z-1}$

## 存储器

存储器是用来存储数据的器件或电路单元，D 锁存器或触发器本身就是 1 位存储器，但通常所说的存储器都是指大量数据的存储单元。

![[assets/数字电路/二进制词头.png]]

存储器按掉电 (即撤去供电) 后数据是否丢失可分为易失性和非易失性两种,易失性意为掉电数据丢失,非易失性为掉电数据依然留存。按存取能力可分为 ROM(只读存储器) 和 RAM(随机访问存储器) 两种。ROM 并非如字面意义上完全不能写入,因为凡是不能直接进行写入,而是要先进行块擦除才能写入的存储器也被分类到 ROM。而 RAM 则可随时读或写任意字或字节。

![[assets/数字电路/常见存储器.png]]

（注：图中 Synchronous 为 static）

NAND Flash 广泛用于固态硬盘、存储卡、闪存盘 (俗称“U 盘”)，而 SDRAM 则主要用于计算机和嵌入式计算

设备的运行内存,SDRAM 根据每个时钟周期读写数据的次数又分为 SDR(Single Data Rate)、DDR(Dual Data Rate) 和 QDR(Quad Data Rate)。

而在 FPGA 中最为常用的则是 SSRAM,主要原因是其速度快、访问灵活,适合数字逻辑中复杂算法数据的存取。CPU 中的高速缓存也是由 SSRAM 构成。高速缓存是 CPU 中重要的组成角色,很多 CPU 中的高速缓存

甚至会占到硅片中一半的面积。

### SRAM

如图是一个 8 位宽、4 字深的 SRAM，行由 2-4 译码器驱动，列由 8 个读写单元控制

![[assets/数字电路/SRAM.png]]

每个单元由两个非门构成保持环，WL(Word Line) 为高，BL(Bit Line) 为低的一个触发保持环新值（两个非门驱动力较弱），若 BL 未被驱动，可以通过 BL 读取保持环的值。

读写电路中，右侧 Q 驱动差分放大器，WR 为低时，BL 不被驱动，Q 输出被地址译码选中的值，WR 为高时，存储单元被更新。

![[assets/数字电路/SRAM单元.png]]

不需要时钟参与的为异步 SRAM(ASRAM)。

A 称为 SRAM 的地址，地址通常拆分为行列地址。SRAM 的容量为 $Capacity=2^{AW}*DW$。对于多字节位宽的 SRAM，每个字地址对应着多字节，SRAM 会额外提供字节使能输入。

存储器中存储 16 位及更宽的数据时，可将数据的低字节放置在存储器的低字节地址上，称为小端模式，也可将数据的高字节放置在存储器 的低字节地址上，称为大端模式。如果存储数据的起始地址可以被数据的字节数整除，则称为对齐存储，否则称为非对齐存储。对齐存储可以提高数据访问速率，但可能会浪费一些存储空间。

## 小数

## 同步时序逻辑

整个电路使用同一个时钟，触发器的工作与否由使能控制的电路称为同步时序电路。同步时序电路时钟单一，触发器的工作不受到组合逻辑竞争冒险的影响,在 FPGA 中,基本只会使用同步时序逻辑。为使同步时序逻辑电路正常工作，必须保证每一个触发器的数据和时钟都满足建立时间和保持时间的要求。

# Verilog 基本应用
## 组合逻辑

编码器和译码器

```verilog
    always_comb begin
        out = '0;
        for(integer i = 2**OUTW - 1; i >= 0; i--) begin
            if(in[i]) out = OUTW'(i);
        end
    end
```

该逻辑会从高位到低位判断是否为 1，意为只要较低的第 n 位为 1，则无论高位如何，输出为 n。如 $1010->01, 011->0$。

## 时序逻辑

## 跨时钟域

## 存储器

## FIFO
## 按键
使用触发器在一般 10 ms 的周期下间歇地查询输入实现去抖，抖动时间一般在 10 ms 左右。
![](assets/4%20抖动.png)
## PWM

脉冲宽度调制（Pulse Width Modulation）是输出矩形波占空比与待调信号瞬时值呈线性关系的调制。可用信号与锯齿波比较得到。

![](assets/4%20PWM.png)

PWM 常用于功率元件的驱动，如开关电源变换器中的开关器件、控制系统中的电机等，也可在低要求场合做数模转换，相当于采样率为 PWM 频率的 DAC。PWM 输出至片外经过低通重构滤波器可以得到与占空比呈正比的输出电压。

### 单端 PWM

输出占空比：$\eta = data/M, data\in [0, M-1]$

输出频率：$f_{PWM}=f_{clk}/M$

输入可能是有符号或无符号。

### 差分 PWM

功率元件常常需要全桥驱动，以便在但功率电源时获得双极性的驱动电压。全桥驱动需要差分 PWM，由 P 和 N 两个信号构成。表达占空比范围为 [-1, 1], 形式也有很多，如差动时间、固定 P 相、固定 N 相、固定低电平和固定高电平。

## FSM

# IO 规范与外部总线

## 信号

## UART

## SPI

## I2C

## I2S

# 片上互联

## 简单存储器映射

## 流水线

## 握手

“valid” 和“ready”握手机制——AXI 总线使用的机制。

valid 信号由主接口或源接口产生，ready 信号则由从接口或汇接口产生，一般定义为高电平有效。

在 valid-ready 握手机制中

- valid 为“数据有效”之意，一旦数据有效便置位，然后持续到握手成功才可能会清除。
- ready 为“可以接收”之意，具备接收数据的条件时置位。
- valid 和 ready 同时置位，则两者传递一次数据（在即将到来的时钟有效沿）称为一次握手。
- 握手后如果下一个周期数据仍是有效数据，valid 可继续置位，否则应清除。
- 握手后如果下一个周期仍然可以接收数据，ready 可继续置位，否则应清除。
- valid 置位不得依赖于 ready 的状态，任何时刻数据有效，valid 应被置位，而不能等待 ready。
- ready 置位可依赖于 valid 的状态，接收端可以等待 valid 置位后置位 ready。
- valid 清除依赖于 ready 的状态，valid 清除只能发生在握手成功之后，以确保有效数据被接收。
- ready 清除不依赖于 valid 的状态，任何时候不具备接收数据的条件，ready 可被清除。

## AXI

# 数字通信应用
# DDR3

# PCIE
