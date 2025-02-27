# 1. **stm32综述**

## 1.1 架构

 [STM8和STM32产品选型手册.pdf](官方资料\STM8和STM32产品选型手册.pdf)  

[ARM+Cortex-M3与Cortex-M4权威指南.pdf](官方资料\ARM+Cortex-M3与Cortex-M4权威指南.pdf) 

 [STM32F10xxx Cortex-M3编程手册-英文版.pdf](官方资料\STM32F10xxx Cortex-M3编程手册-英文版.pdf) 

 [stm32参考手册英文.pdf](官方资料\stm32参考手册英文.pdf) 

Cortex-M3内核 ARM设计；其他 ST设计

具体引脚由规格书定义。



## 1.2 文件管理

| 名称    |                               | 内容和作用 |
| ------- | ----------------------------- | ---------- |
| Listing |                               |            |
| Objects |                               |            |
| user    | main.c                        |            |
|         | stm32f10x_it.h,stm32f10x_it.c |            |
|         | stm32f10x_conf.h              |            |
| startup | startup_stm32f10x_hd.s        |            |
| Library | CMSIS：core_cm3.c             |            |
|         | FWLB                          |            |

启动文件：汇编指令，建立环境

​	1.作用：初始化堆栈指针、初始化PC指针、初始化中断向量表、配置系统时钟、调用C库函数_main初始化用户堆栈。

2. 查找ARM汇编指令。

3. Stack-局部变量、函数调用、函数形参，不超过SRAM

   Heap-动态内存分配，例malloc。

头文件：寄存器映射，地址定义

## 1.3 固件库

CMSIS标准（Cortex MicroController Software Interface Standard）包括了内核函数层】中间件访问层、设备外设访问层。 

1-汇编编写的启动文件
startup_stm32f10x_md.s:设置堆栈指针、设置PC指针、初始化中断向量表、配置系统时钟、对用C库函数_main最终去到C的世界

2-时钟配置文件
system_stm32f10x.c：把外部时钟HSE=8M，经过PLL倍频为72M。

3-外设相关的
stm32f10x.h：实现了内核之外的外设的寄存器映射
xxx：GPIO、USRAT、I2C、SPI、FSMC
stm32f10x_xx.c：外设的驱动函数库文件
stm32f10x_xx.h：存放外设的初始化结构体，外设初始化结构体成员的参数列表，外设固件库函数的声明

4-内核相关的
CMSIS - Cortex 微控制器软件接口标准
core_cm3.h：实现了内核里面外设的寄存器映射
core_cm3.c：内核外设的驱动固件库

NVIC(嵌套向量中断控制器)、SysTick(系统滴答定时器)
misc.h
misc.c

5-头文件的配置文件
stm32f10x_conf.h：头文件的头文件
//stm32f10x_usart.h
//stm32f10x_i2c.h
//stm32f10x_spi.h
//stm32f10x_adc.h
//stm32f10x_fsmc.h
......

6-专门存放中断服务函数的C文件
stm32f10x_it.c
stm32f10x_it.h

中断服务函数你可以随意放在其他的地方，并不是一定要放在stm32f10x_it.c

#include "stm32f10x.h"   // 相当于51单片机中的  #include <reg51.h>

int main(void)
{
	// 来到这里的时候，系统的时钟已经被配置成72M。
}



## 1.4 MDK编译环境

德国Keil，RealView MDK

### 1.4.1 编译过程

C/C++代码->.o->code、date、bebug->二进制，十六进制文件

汇编代码->.o->

1. 编译。使用armcc和armasm编译器，生成.o文件
2. 链接。armlilnk将.o文件链接成.axf或.elf
3. 格式转换。系统使用elf，将其转换成.bin或.hex下载到FLASH或ROM，在单片机上运行。

### 1.4.2 程序组成、存储、运行

- Code：代码区，机器指令，存储到ROM。
- RO-data：只读，ROM
- RW-data：可读写区域，RAM，初始化非零值
- ZI-data：0初始化数据，RAM
- ZI-data的stack和Heap：局部变量为栈，包括malloc，全局变量为堆。
- 程序大小是code和ro-data之和

### 1.4.3 编译工具链

1. 添加路径到PATH环境变量
2. armcc,armasm,armlink
3. armar,fromelf
4. uvprojx，记录工程结构；uvoptx，记录工程配置；uvguix记录GUI布局
5. output:lib第三方库；.dep,.d以来文件；.crf交叉引用文件，找到跳转位置；机器码

 



### 1.3.4 略



# 2. GPIO

## 2.1介绍

通用输入输出端口，ZET6有7组，每组16个引脚

![GPIO](D:\study\32\stm32相关资料\GPIO.JPG)结构：

1. 上、下拉电阻和保护电阻。防止过高过低电压输入。
2. P-MOS管和N-MOS管。推挽输出；开漏输出，线与功能。
   3. 输出数据寄存器。GPIOx_ODR
4. 复用功能输出。
5. 输入数据寄存器。GPIOx_IDR
6. 复用功能输入。
7. 模拟输入输出。
8. 输出模式。



## 2.2 配置流程

1. 开启GPIO时钟

```c
void RCC_APB2PeriphClockCmd(uint32_t RCC_APB2Periph, FunctionalState NewState)
    //可用|开启同时配置多个时钟
```

2. 初始化引脚，模式，参数

```c
typedef struct
{
  uint16_t GPIO_Pin; 
    //设置引脚
  GPIOSpeed_TypeDef GPIO_Speed;
    //一般50MHz
  GPIOMode_TypeDef GPIO_Mode;
    //复用推挽输出，浮空输入。。。
}GPIO_InitTypeDef;
```

3. 编写测试程序

# 3. 中断

## 3.1 中断介绍

1. 在内核搭载的异常响应系统：内核-系统异常，8个；外设-外部中断，60个。定义在IRQn_Type结构体中——IRQn_Type结构体包含中断

2. NVIC：嵌套向量中断控制器，是内核的外设，32中包含一个子集。

   ```c
   typedef struct{
       ISER[8];//中断使能
       ICER[8];//中断清除
       IP[240];//中断优先级
       ...
   } NVIC_Type;
   ```

3. 优先级。

   NVIC_IPRx中断优先级寄存器，配置外部中断的优先级，8位中32使用4位，分为抢占和子优先级。

   分为五组。主优先级=抢占优先级。

## 3.2 中断控制器

EXTI是外部中断控制器，管理20个中断线/事件线，挂载在APB2上。

功能框图：

![EXTI](D:\study\32\stm32相关资料\EXTI.JPG)

EXTI功能有产生中断—输入NVIC，运行中断函数，和产生事件—传输一个脉冲给其他外设。

每个信号线都有20个。有20个中断/事件线。每个GPIO都可以被设置为输出入线，占用EXTI0-EXTI15



## 3.3 配置流程

1. 初始化用来产生中断的GPIO，开启GPIO时钟，使能某个中断
2. 开启EXTI的AFIO时钟，初始化EXTI、NVIC结构体，使能中断请求

```c
typedef struct
{
  uint32_t EXTI_Line;					//中断、事件线
  EXTIMode_TypeDef EXTI_Mode;			//模式：中断或事件
  EXTITrigger_TypeDef EXTI_Trigger;		//触发类型
  FunctionalState EXTI_LineCmd;			//使能
}EXTI_InitTypeDef;

typedef struct
{
  uint8_t NVIC_IRQChannel;					//中断源，IRQn
  uint8_t NVIC_IRQChannelPreemptionPriority; //抢占
  uint8_t NVIC_IRQChannelSubPriority;		//子优先级
  FunctionalState NVIC_IRQChannelCmd;		//使能或失能
} NVIC_InitTypeDef;
```

3. 编写中断服务函数 

# 4. 通讯协议

串行、并行；全双工、半双工、单工；同步、异步；

速率：比特率(bit/s)每秒传输的二进制的位数；波特率：单位时间传输的码元。

## 4.1 USART-串口通讯

### 4.1.1 介绍

1. 物理层：规定通讯系统中具有机械、电子功能部分的特性，确保原始数据的传输。

RS-232标准，规定信号用途，通讯接口，以及电平标准。最初用于计算器、路由器、调制调解器通讯。

通讯设备的COM接口——DB9接口通过串口信号线传输，使用RS-232标准传输，信号经过电平转换芯片转换成刚TTL标准电平。

| 通讯标准 | 电平标准（发送端）              |
| -------- | ------------------------------- |
| 5V TTL   | 逻辑1：2.4V—5V；逻辑0：0—0.5V   |
| RS-232   | 逻辑1:-15V—-3V；逻辑0：+3V-+15V |

2. 协议层：规定通讯逻辑，用以手法双方的数据打包，解包标准。

由起始位、数据位、校验位、停止位组成。常见的波特率有4800，9600，115200

### 4.1.2 stm32的USART

Universal Synchronous Asynchronous Receiver and Transmitter 通用同步异步收发器，串行通信，全双工；UART为异步。

### 4.1.3 功能框图

![USART](D:\study\32\stm32相关资料\USART.JPG)

1. 功能引脚

   TX：发送数据输出

   RX：接收数据输入

   SW RX：数据接收。只用于单卡，智能卡，无外部

   nRTS：请求发送（Request To Send）。硬件流控制，使能，USART准备好接收，变为低，接收寄存器满，设置为高。

   nCTS：清除以发送（Clear To Send）。硬件流控制，使能CTS，发送器发送下一帧检测，低可发送，高停止。

   SCLK：时钟发送，同步模式。

   USART1-APB2;USART2345-APB1

2. 数据寄存器

   USART_DR低9位有效，第九位取决于控制寄存器1USART_CR1的M位，M=0,8位字长；M=1，九位字长，一般8位。包括已经发送或接受的数据，包含两个寄存器，专门可写TDR，专门可读RDR，二者介于系统总线和移位寄存器之间，都是一个一个位传输的。支持DMA传输。

3. 控制器。控制发送器和接收器。USART_CR1的UE位置1使能USART，M位控制字长。TE位置1，启动数据发送，TX输出，低位前，高位后，同步输出时钟。停止位时间通过USART_CR2的STOP[1:0]控制，可选0.5,1,1.5,2个，默认1个。USART_CR1的RE置1，使能接收

4. 小数波特率生成（bit/s;bps)

   $ Tx/Rx=f_ck/(16*USARTDIV)$

5. 校验控制。

6. 中断控制

### 4.1.4 配置流程

硬件设计：选择CH340G芯片，用于USB转USART的IC。CH340G的TXD引脚与USART1的RX引脚相连，RXD引脚与USART的TX引脚相连。

1. 使能RX、TX引脚GPIO时钟和USART时钟。

```c
RCC_APB2PeriphClockCmd(RCC_APB2Periph_USART1 | RCC_APB2Periph_GPIOA, ENABLE);
```

2. 初始化GPIO，并将GPIO复用到USART上；

​	USART Tx的GPIO配置为推挽复用模式；USART RX的GPIO配置为浮空输出模式。

3. USART初始化，配置USART参数

```c
void USART_Init(USART_TypeDef* USARTx, USART_InitTypeDef* USART_InitStruct)；
typedef struct
{
  uint32_t USART_BaudRate;
    //波特率，一般为2400、9600、19200、115200。设置USART_BRR的值。
  uint16_t USART_WordLength; 
    //数据帧字长，8位无奇偶校验，或9位有奇偶校验。
  uint16_t USART_StopBits; 
    //停止位设置，0.5/1/1.5/2个，一般1个
  uint16_t USART_Parity;
    //奇偶校验位，无、奇、偶
  uint16_t USART_Mode;
    //USART_Rx|USART_Tx
  uint16_t USART_HardwareFlowControl;
    //硬件流控制。使能RTC，使能CTS，同时使能RTS和CTS，不使能
} USART_InitTypeDef;
```

4. 配置中断控制器并使能USART接受中断

```c
// 配置USART为中断源
 NVIC_InitStructure.NVIC_IRQChannel = USART1_IRQn;
//使能串口接收中断
void USART_ITConfig(USART_TypeDef* USARTx, uint16_t USART_IT, FunctionalState NewState)
```

5. 使能USART

```c
void USART_Cmd(USART_TypeDef* USARTx, FunctionalState NewState)；
```

6. 编写USART中断服务函数，实现数据接受和发送

```c
// 串口中断服务函数
void USART1_IRQHandler(void)
{
  uint8_t ucTemp;
	if(USART_GetITStatus(USART1,USART_IT_RXNE)!=RESET)
	{		
		ucTemp = USART_ReceiveData(USART1);
    USART_SendData(USART1,ucTemp);    
	}	 
}
//USART_GetITStatus用来获取标志位转台，专门获取中断事件标志，判断是否真的产生USART数据接收这个中断事件，有的话就接收。
```

5. 字符发送函数，重定向printf和scanf函数

   ！！！重定向函数需要开启 use microLIB

```c
/*****************  发送一个字节 **********************/
void Usart_SendByte( USART_TypeDef * pUSARTx, uint8_t ch)
{
	/* 发送一个字节数据到USART */
	USART_SendData(pUSARTx,ch);
	/* 等待发送数据寄存器为空 */
	while (USART_GetFlagStatus(pUSARTx, USART_FLAG_TXE) == RESET);	
}
void Usart_SendArray( USART_TypeDef * pUSARTx, uint8_t *array, uint16_t num)

 /*****************  发送字符串 **********************/
void Usart_SendString( USART_TypeDef * pUSARTx, char *str)

//重定向c库函数printf到串口，重定向后可使用printf函数
int fputc(int ch, FILE *f)
{
		/* 发送一个字节数据到串口 */
		USART_SendData(DEBUG_USARTx, (uint8_t) ch);
		/* 等待发送完毕 */
		while (USART_GetFlagStatus(DEBUG_USARTx, USART_FLAG_TXE) == RESET);		
		return (ch);
}
//重定向c库函数scanf到串口，重写向后可使用scanf、getchar等函数
int fgetc(FILE *f)
{
		/* 等待串口输入数据 */
		while (USART_GetFlagStatus(DEBUG_USARTx, USART_FLAG_RXNE) == RESET);
		return (int)USART_ReceiveData(DEBUG_USARTx);
}
```



## 4.2 CAN

### 4.2.1 介绍

控制器局域网络（Controller Area Network），BOSCH，现场总线，半双工。

### 4.2.2 物理层

异步通讯，只有CAN_High和CAN_Low两条信号，构成差分信号线，以差分信号通讯。

1. 物理形式。有两种，闭环总线网络，两段要有120欧电阻，和开环总线网络，每跟总线串联2.2千欧电阻。

2. 通讯节点。总线上可以挂载多个通讯节点，不对节点进行地址编码而是数据编码。

   由一个控制器和收发器组成。控制器和收发器之间通过CAN_Tx和CAN_Rx相连，收发器和总线之间使用CAN_High和CAN_Low相连。前者使用TTL，后者是差分信号线。前者传输二进制编码，后者转换成差分信号。

3. 差分信号。总线必须必须处于隐形电平或显性电平。

   

### 4.2.3 协议层

1. 波特率和位同步



### 4.2.4 STM32的CAN特性及架构



### 4.2.5 配置流程

硬件设计：TJA1050芯片作为CAN收发器，将CAN控制器的TTL电平转换成差分信号。






## 4.3 SPI

### 4.3.1 介绍                             

(Serial Peripheral interface)，高速全双工。

### 4.3.2 物理层

三条总线SCK,MOSI,MISO，片选线nSS

### 4.3.3 协议层

时钟极性CPOL和时钟相位CPHA

CPOL：SPI空闲时的电平信号，0-低，1-高。

CPHA：SCK采样的时刻，0-奇数边，1-偶数边。

### 4.3.4 STM32的SPI特性及架构

其外设可以做通讯主机或者从机。SCK最高$f_{pclk}/2$，支持四种模式，数据帧长度可以设为8位或16位，可MSB先行或LSB先行，支持双线全双工、双线单向、单线模式。

![SPI](D:\study\32\stm32相关资料\SPI.JPG)

1. 通信引脚

| 引脚 | SPI  |      |                  |
| ---- | ---- | ---- | ---------------- |
|      | SPI1 | SPI2 | SPI3             |
| nSS  | PA4  | PB12 | PA15下载口的TDI  |
| CLK  | PA5  | PB13 | PB3下载口的TD0   |
| MISO | PA6  | PB14 | PB4下载口的NTRST |
| MOSI | PA7  | PB15 | PB5              |

SPI1是APB2上的，最高36Mbit/s，SPI2、SPI3是APB1的设备，最高18Mbit/s。SPI3用到了下载引脚，默认下载，第二功能为IO，首先需禁用下载，一般不会。

2. 时钟控制

   SCK线的信号，由SPI所在的APB总线频率分频得到。由控制寄存器的BR[0:2]控制分频因子

3. 数据控制

   一般会选择MSB优先，读写数据寄存器DR，DFF位配置8或16位。

4. 总体控制

   一般不用SPI外设的NSS信号线，可以用GPIO控制电平，产生通讯起始停止位。

### 4.3.5 配置流程（以读写串行FLASH为例）

硬件设计：FLASH芯片

1. 初始化通讯使用的引脚。使能引脚时钟，SPI外设时钟。
3. 配置SPI外设，并使能。

```c
typedef struct
{
  uint16_t SPI_Direction;
    //双线全双工、双线只接收、单线只接收、单线只发送。
  uint16_t SPI_Mode;
    //从机或主机模式
  uint16_t SPI_DataSize;
    //8位或16位
  uint16_t SPI_CPOL;
  uint16_t SPI_CPHA;
    //硬件模式自动产生、软件模式软件调试
  uint16_t SPI_NSS;
    //硬件模式自动产生、软件模式软件调试  
  uint16_t SPI_BaudRatePrescaler;
    //分频因子
  uint16_t SPI_FirstBit;
    //MSB或LSB
  uint16_t SPI_CRCPolynomial;
    //校验多现实
}SPI_InitTypeDef;
void SPI_Cmd(SPI_TypeDef* SPIx, FunctionalState NewState);
```

1. 编写SPI按字节收发的函数。
2. 编写对FLASH擦除、读写的函数。
3. 编写测试程序。 



### 4.4.6 串行FLASH文件系统FstFs

1. 文件系统。对存储介质格式化之后新建文件分配表和目录，即可记录数据存放的物理地址。FatFs是面向小型嵌入式系统的通用FAT文件系统，



## 4.4 IIC

### 4.4.1 介绍

IIC(Inter-Intergrated Circuit)，由Phiilps开发，

### 4.4.2 物理层

1. 支持多设备的总线。可以连接多个主机和从机
2. 使用SCL和SDA总线，数据和时钟。
3. 每个总线设备都有独立地址。
4. 总线通过上拉电阻连接电源。空闲输出高阻态。
5. 使用仲裁方式。
6. 三种传输模式，100k、400k、3.4M（大多数不支持）。
7. 连接相同总线的IC数量受最大电容400pF限制。

### 4.4.3 协议层

基本读写过程：参考协议pdf

SCL高，SDA拉低开始；SCL高，SDA拉高停止；SCL高，SDA有效；一般由主机产生。

第九个时钟从机应答，SDA高NACK，低ACK。



### 4.4.4 stm32的IIC特性与架构

直接控制两个GPIO引脚，模拟时序，为软件模拟协议；也可设置IIC片上外设，为硬件协议方式。

后者支持100Kbit/s和400Kbit/s速率，支持7位、10位设备地址，DMA数据传输，SMBus2.0协议。

![IIC](D:\study\32\stm32相关资料\IIC.JPG)

1. 引脚复用

| 引脚 | IIC1              | IIC2 |
| ---- | ----------------- | ---- |
| SCL  | PB5/PB8（重映射） | PB10 |
| SDA  | PB6/PB9           | PB11 |

​	SMBA用于SMBUS警告，IIC未用到。

2. 时钟控制。CCR控制时钟频率，IIC挂载在APB1上

3. 数据控制逻辑。SDA连接移位寄存器上，来源目标即是数据寄存器DR、地址寄存器OAR，PEC寄存器，SDA，可支持两个IIC设备，分别存储在OAR1和OAR2中。

4. 整体控制逻辑。

   ![IIC master](D:\study\32\stm32相关资料\IIC master.JPG)

   ![IIC master rec](D:\study\32\stm32相关资料\IIC master rec.JPG)

   可选择时钟占空比，需要配置CCR值。
   
   



### 4.4.5 配置流程（以读写EEPROM为例）

硬件设计：EEPROM为AT24C02，通信引脚上拉，

1. 配置通讯使用的引脚为开漏模式
2. 使能IIC外设时钟，配置IIC外设参数，并使能

```c
typedef struct
{
  uint32_t I2C_ClockSpeed; 
    //SCL时钟频率，要低于400000
  uint16_t I2C_Mode; 
    //指定工作模式，IIC或SMBUS
  uint16_t I2C_DutyCycle; 
    //指定时钟占空比，可选low:high=2:1,16:9模式
  uint16_t I2C_OwnAddress1;
    //自身地址
  uint16_t I2C_Ack;
    //使能，一般需要使能。
  uint16_t I2C_AcknowledgedAddress;
    //指定地址长度，7/10
}I2C_InitTypeDef;
void I2C_Init(I2C_TypeDef* I2Cx, I2C_InitTypeDef* I2C_InitStruct);
void I2C_Cmd(I2C_TypeDef* I2Cx, FunctionalState NewState);    
```

3. 编写IIC基本按字节收发函数

```c

```

4. 编写读写EEPROM存储内容函数
5. 编写测试程序



## 4.5 RS-485通讯 

### 4.5.1 介绍



### 4.5.2 物理层



### 4.5.3 协议层



# 5. 时钟

## 5.1 RCC

### 5.1.1 介绍
RCC：reset clock control 复位和时钟控制器。
RCC时钟部分作用，库函数的标准配置：PCLK2=HLCK=SYSCLK=PLLCLK=72M，PCLK1-HCLK/2=36M。

### 5.1.2 RCC框图：

1. HSE高速外部时钟信号。HSE由晶振提供，4-16MHz不等。
2. PLL时钟源。来源可为HSE和HSI/2，HSI为内部高速时钟，8M，一般选取HSE。
3. PLL时钟PLLCLK。设置PLL倍频因子，一般PLLCLK=8M*9=72M，最高128M。
4. SYSCLK系统时钟。来源HSI、PLLCLK、HSE，一般SYSTICK=PLLCLK=72M。
5. AHB总线时钟HCLK，AHB预分频得到，设为1，HCLK=SYSCLK=72M。
6. APB2总线时钟HCLK2，PCLK2由HCLK经过高速APB2预分频，设为1，PCLK2=HCLK=72M，片上高速外设挂载。
7. APB1总线时钟HCLK1，PCLK1由HCLK经过低速APB1预分频，设为2，PCLK1=HCLK/2=36M，片上低速外设挂载。
8. 设置系统时钟库。

```c
static void SetSysClockTo72(void);
```

其他时钟：

1. USB。PLLCLK经过USB预分频得到，一般PLLCLK=72M，USBCLK=48M，PLLCLK只能由HSE倍频得到。

2. Cortex系统时钟。用于驱动内核系统定时器Systick。

3. ADC时钟。由PCLK2经过ADC预分频得到，取28M和56M。

4. RTC时钟、独立看门狗时钟。

   RTC由HSE/128分频，或LSE，32.768Hz，或HIS提供。独立看门狗由LSI提供，一般40KHz。
   
5. MCO时钟输出。

   MCO(microcontroller clock output)，PA8复用，对外提供时钟，来源可以使PLLCLK/2、HSI、HSE、SYSCLK。

### 5.1.3 配置流程

- 使用HSE



- 使用HSI



- MCO输出





## 5.2  SysTick

### 5.2.1 介绍

Systick-系统定时器是属于CM3内核中的外设，内嵌在NVIC中，24bit向下递减计数器。

| 寄存器 | 描述                    |
| ------ | ----------------------- |
| CTRL   | Systick控制及状态寄存器 |
| LOAD   | Systick重装载数值寄存器 |
| VAL    | Systick当前数值寄存器   |
| CALIB  | Systick校准数值寄存器   |

### 5.2.2 配置流程

1. 设置重装载寄存器。
2. 清楚当前数值寄存器。
3. 配置控制与状态寄存器。

```c
static __INLINE uint32_t SysTick_Config(uint32_t ticks)
{ 
  if (ticks > SYSTICK_MAXCOUNT)  return (1);
  SysTick->LOAD  =  (ticks & SYSTICK_MAXCOUNT) - 1;
  NVIC_SetPriority (SysTick_IRQn, (1<<__NVIC_PRIO_BITS) - 1); 
  SysTick->VAL   =  (0x00);
  SysTick->CTRL = (1 << SYSTICK_CLKSOURCE) | 
	                (1 << SYSTICK_ENABLE) | 
									(1 << SYSTICK_TICKINT); 
  return (0);
}
```
4. 中断服务函数。

```c
void SysTick_Handler(void)
{
	TimingDelay_Decrement();	
}
void TimingDelay_Decrement(void)
{
  if (TimingDelay != 0x00)
  { 
    TimingDelay--;
  }
}
```
5. 定时编程。

````c
void Delay_us(__IO u32 nTime)
{ 
  TimingDelay = nTime;

  while(TimingDelay != 0);
}
````

6. 中断时间计算。

   Systick是向下递减计数的，计数一次时间$T_{DEC}=1/CLK_{AHB}$，$CLK_{AHB}=72MHz$重装载寄存器的值减到0产生中断
   
   SysTick形参设为SystemCoreClock/100000，则$T_{INT}=10us$。
   
   delay函数的形参，即为$T_{INT}$的个数。

```c
//简洁形式
void SysTick_Delay_us(_IO uint32_t us)
{
    uint32_t i;
    SysTick_Config(SystemCoreClock/1000000);
    //毫秒，即为SystemCoreCLock/1000.
    for(i=0;i<us;i++)
    {
        while(!((SysTick->CTRL)&(1<<16)));
    }
    SysTick->CTRL &=~SysTick_CTRL_ENABLE_Msk;
}
```






可编程预分频器（PSC）驱动的16位自动装载计数器（CNT）构成。

1） 16位向上、向下自动装载计数器（TIMx_CNT）

2）16位可编程预分频器（TIMx_PSC），分频系数为1~65535

3）4个独立通道

A. 输入捕获	B. 输出比较	C.PWM生成	D. 单脉冲模式输出

4）外部信号（TIMx_ETR）控制定时器和定时器相连的同步电路。

5）通过某些事件产生中断/DMA

## 5.3 通用定时器

### 5.3.1 介绍

TIM1,8 高级定时器，16位可向上向下计数，可定时比较捕获，三相；

TIM2-5 通用定时，16位可向上向下计数，可定时比较捕获；

TIM6,7基本定时器，16位向上计数，只能定时，无外部IO。

每个定时器有8个外部IO，除了TIM6,7，都可以用来产生PWM输出，高级可以产生7路输出。

### 5.3.2 结构框图

![基本定时器](D:\study\ee\stm32\stm32相关资料\基本定时器.JPG)

1. 时钟源：TIMxCLK，内部时钟CK_INT，经过APB1预分频后提供，库函数为2，TIMxCLK=36*2=72M。
2. 计数器时钟：PSC16位预分频之后CK_INT，驱动定时器计数，进行1~65536之间分频，CK_INT=TIMxCLK/(PSC+1)。
3. 计数器：CNT是16位计数器，向上计数，最大65535，到达重装寄存器更新，清零。
4. 自动重装寄存器：ARR为16位，计数到达值，使能中断后，产生溢出中断。
5. 定时时间：电梯时期的定时时间为计数器中断周期乘中断次数，记一次数时间$1/(TIMxCLK/(PSC+1))$，产生一次中断：$1/(CK_CLK*ARR)$，设置一个time，定时时间为$1/CK_CLK*(ARR+!)*time$。




### 5.3.4 配置流程

1. TIM3使能，挂载在APB1下

```c
RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM3,ENABLE);
```

2. 初始化参数

```c
void TIM_TimeBaseInit(TIM_TypeDef* TIMx, TIM_TimeBaseInitTypeDef* TIM_TimeBaseInitStruct);
typedef struct
{
  uint16_t TIM_Prescaler;
    //预分频，设置TIMx_PSC，0~65535
  uint16_t TIM_CounterMode; 
    //计数模式，基本只能向上
  uint16_t TIM_Period;  
    //定时器周期，即重载寄存器，时间生成更新到影子寄存器，0~65535
  uint16_t TIM_ClockDivision;
    //时钟分频，基本无功能
  uint8_t TIM_RepetitionCounter;
    //重复计数，基本无。
} TIM_TimeBaseInitTypeDef;
```

3. 设置TIM3_DIER允许更新中断

4. TIM3 中断优先级设置

设置NVIC中断优先级，相关寄存器。

```c
#define xx TIMx_IQRn
RCC_APB1Periph_TIMx
```

5. 允许TIM3工作，使能

```c
void TIM_TimeBaseInit(TIM_TypeDef* TIMx, TIM_TimeBaseInitTypeDef* TIM_TimeBaseInitStruct);
 // 初始化定时器
void TIM_ClearFlag(TIM_TypeDef* TIMx, uint16_t TIM_FLAG);
// 清除计数器中断标志位
void TIM_ITConfig(TIM_TypeDef* TIMx, uint16_t TIM_IT, FunctionalState NewState);
// 开启计数器中断
void TIM_Cmd(TIM_TypeDef* TIMx, FunctionalState NewState);
// 使能计数器
```

6. 编写中断服务函数，清除中断位

```c
void TIMx_IQRHandler(void);
void TIM_ClearITPendingBit(TIM_TypeDef* TIMx, uint16_t TIM_IT);
```



## 5.4 高级定时器

### 5.4.1 高级控制定时器

引入外部引脚，实现输出捕获，输入比较，增加了可编程死区互补输出、重复计数器、带刹车功能。

包含16位自动重装载寄存器ARR，16位计数器CNT可向上向下计数，16位可编程预分频PSC，预分频时钟源可有内外部时钟，8位重复计数器PCR，最高40位可编程。

### 5.4.2 功能框图

![ad TIM](D:\study\ee\stm32\stm32参考资料\ad TIM.JPG)

一、时钟源：有四种时钟源可选

   1. 内部时钟CK_INT：72M，TIMx_SMCR的SMS位000.

   2. 外部时钟模式1TLx：
      ![外部时钟](D:\study\ee\stm32\stm32参考资料\外部时钟.JPG)
   - 引脚4个通道，TIMxCCMRx的CCxS[1:0]选择，x=1控制1/2，x=2控制3/4。
   - 滤波器，TIMx_CCMRx位ICxF[3:0]。
   - 边沿检测，决定上升沿或下降沿有效，TIMx_CCER的CCxP和CCxNP。
   - 触发选择，触发源有两个，一个是滤波后的定时器输入1（TL1FP1)和滤波后的定时器输入2（TL2FP2)，由TIMxSMCRTS的位TS[2:0]控制。   
   - 从模式选择：，TIMx_SMCR的SMS[2:0]为000选择1。 
   - 使能定时器。TIMx_CR1的位CEN控制。

![外部时钟模式2](D:\study\ee\stm32\stm32参考资料\外部时钟模式2.JPG)

   3. 外部时钟模式2：
   - 引脚，来自TIMx_ETR，特定输入通道。
   - 外部触发极性。TIMx_SMCR控制，上升或下降有效。
   - 外部触发预分频,ETRP的信号频率不能超过TIMx_CLK(72M)的1/4，有时候需要分频器，有TIMx_SMCR的位ETR[1:0]配置。
   - 滤波器，TIMx_SMCR的ETF[3:0]配置，fDTS为内部时钟CK_INT分频
   - 从模式，TIMx_SMCR的ECE为1 配置模式2.
   - 使能计数器，TIMx_CR1的CEN配置。   
	4. 内部触发输入
	使用一个定时器作为另一个定时器的预分频，可实现定时器同步或级联。

二、控制器

三、时基单元

时基单元包括四个寄存器，计数（CNT），预分频（PSC），自动重装（ARR），重复计数（PCR）。重复计数只高级定时器独有，前三者16位有效，TIMx_PCR8位有效。

PSC：输入时钟CK_PSC即为时钟源，输出CK_CNT用来驱动CNT计数，$f_{CK_INT}=f_{CK_PSC}/(PSC[15:0]+1)$，得到不同的CK_INT。

CNT：递增计数、递减计数、递增/递减（中心对齐），三种模式。

- 递增。若禁用重复计数，生成上溢时间即生成更新时间（UEV），若使能重复计数器，每生成一次上溢事情，重复计数减一，直到为0生成更新事件。
- 递减。从ARR开始同上。
- 中心对齐。从零递增到ARR-1，然后从ARR递减到1，下溢，从零开始，循环。

ARR：存放于CNT比较的值，相等则递减重复，TIMx_CR1寄存器的ARPE位控制自动重载影子寄存器，ARPE置1，有效，只有事件更新才把TIMx_ARR值赋给影子寄存器。若0，TIMx_ARR马上有效。

RCR：高级定时，发生N+1个溢出事件产生更新事件。

四、输入捕获和输出比较

![TIM_io](D:\study\ee\stm32\stm32参考资料\TIM_io.JPG)

- 输入捕获可以对信号跳边沿捕获，常用测量信号脉宽和PWM输入信号频率占空比两种。

捕获原理：捕捉到跳边沿，CNT锁存到捕获寄存器CCR，前后两次相减，算出脉宽或频率，若超出周期则溢出。

输入通道：TLx

输入滤波器、边沿检测器：采样频率大于二倍信号频率，CR173.com位CKD[1:0]和CCMR1/2位ICxF[3:0]控制，f~sample~由f~CK_INT~和f~DTS~分频提供，f~DTS~是f~CK_INT~分频得到，边沿检测由CCER寄存器的CCXP和CCxNP。

捕获通道：ICx对应CCRx，一个输入可以占用多个捕获，

预分频器：决定多少时间进行一次捕获，CCMRx的IC小PSC配置，每个边沿则不分频。

捕获寄存器：ICxPS是最终捕获的信号，发生捕获，CNT锁存到CCR中，产生CC1I中断，响应中断位CCxIF（SR寄存）置位，软件读取CCR中的值可以将CCxIF清零。第二次捕获，捕获溢出CCxOF置位，只能软件清零。

- 输出比较：通过定时器外部引脚对外输出控制信号，8中模式：冻结、将通道x设置为匹配时输出有效电平，通道x设置为匹配时输出无效电平、翻转、强制变成无效电平、强制变成有效电平、PWM1、PWM2。由CCMRx位OCxM[2:0]配置。

比较寄存：CNT和CCR相同，OCxREF信号极性变化，1有效，0无效，产生CCxI比较中断，标志位CCxIF(SR)置位，OCxREF成为真正的OCx和OCxN。

死区发生器：插入死区，生成两路互补输出OCx，OCxN，死区由BDTR位DTG[7:0]配置

输出控制：

![output deadtime](D:\study\ee\stm32\stm32参考资料\output deadtime.JPG)

OCxREF经过死区发生器后产生两路带死区互补信号OCx_DT和OCxN_DT（123有4无），两路带死区互补信号进入输出控制电路。进入输出控制的信号有原始信号和反向信号，有CCER的位CCxP和CCxNP控制，极性选择的信号是否输出到外部引脚CHx/CHxN由CxEheCxNE配置

输出引脚：定时器外部IO，分为CH1/2/3/4，前三个还有互补输出CN1/2/3N。



五、断路功能

电机刹车功能，使能断路，修改信号电平。断路源可以使时钟故障，可以内部时钟安全系统CSS，可以外部断路输入IO。

系统复位启动默认关闭断路，将断路和死区寄存器（TIMx_BDTR）的BKE置1，使能断路。TIMx_BDTR的BKP设置断路引脚有效，1 BRK高有效，否则低有效。

断路产生：

TIMx_BDTR寄存器主输出模式MOE清零，输出无效、空闲、复位。

根据相关控制位状态控制输出通道引脚电平；使能互补，自动控制输出通道电平。

TIMx_SR寄存器的BIF置1，产生中断和DMA传输请求。

TIMx_BDTR中的自动输出使能AOE置1，下一个UEV再次置1。



### 5.4.3 输入捕获应用

一、脉冲跳变延时间测量：在捕获通道TLx中，每次计数器CNT锁存到捕获寄存器CCR。

二、PWM输入测量：只能使用通道1、2，不能使用3、4，一个通道占用两个捕获寄存器，一个定时器最多使用另个输入通道。

选定输入通道，确定触发信号，设置触发极性，另一路信号由硬件配置，不需要软件。

模式控制器配置为复位模式，SMCR的SMS[2:0]，启动触发信号，同时把CNT清零。（寄存器的值需要+1，因为从0开始）



### 5.4.4 输出比较应用

输出比较有8种，由CCMRx的位OCxM[2:0]配置。

PWM输出模式，分两种，PWM1和PWM2，

| 模式 | 计数器CNT计算方式 | 说明                          |
| ---- | ----------------- | ----------------------------- |
| PWM1 | 递增              | CNT<CRR，通道CH有效，否则无效 |
|      | 递减              | CNT<CRR，通道CH无效，否则有效 |
| PWM2 | 递增              | CNT<CRR，通道CH有效，否则无效 |
|      | 递减              | CNT>CRR，通道CH有效，否则无效 |

1. PWM边沿对齐模式，
2. PWM中心对齐模式



### 5.4.5 配置流程（互补输出和输入捕获）

选择通道对应的输出引脚，增加断路需要用到TIMx_BKIN引脚。

1. 初始化需要用到的GPIO

   输出比较通道、互补通道、刹车通道的GPIO，BKIN引脚先默认低电平。

2. 定时器时基结构体TIM_TimtBaseInitTypeDef初始化

```c
typedef struct
{
  uint16_t TIM_Prescaler;
    //预分频器,可实现1-65536分频
  uint16_t TIM_CounterMode;
    //计数模式，
  uint16_t TIM_Period;
    //定时器周期，设置自动重装载ARR的值，0-65535
  uint16_t TIM_ClockDivision;
    //时钟分频，1/2/4分频
  uint8_t TIM_RepetitionCounter;
    //重复计数器，8位，高级定时器
} TIM_TimeBaseInitTypeDef;
//时基结构体
void TIM_TimeBaseInit(TIM_TypeDef* TIMx, TIM_TimeBaseInitTypeDef* TIM_TimeBaseInitStruct);
```

1. 定时器输出比较结构体TIM_OCInitTypeDef初始化

```c
typedef struct
{
  uint16_t TIM_OCMode;
    //比较输出模式，8种，常用PWM1/PWM2,
  uint16_t TIM_OutputState;
    //比较使能
  uint16_t TIM_OutputNState;
    //比较互补使能
  uint16_t TIM_Pulse;
    //比较输出脉冲宽度，0-65535
  uint16_t TIM_OCPolarity;
    //比较输出极性，可设OCx为高或低，设定TIMx_CEER的CCxP值。
  uint16_t TIM_OCNPolarity;
    //比较互补输出极性，可设OCx为高或低，设定TIMx_CEER的CCxNP值
  uint16_t TIM_OCIdleState;
    //空闲状态通道电平设置，1或0，即空闲BDTR_MOE为0，死区时间后，定时器互补通道输出高或低电平，设定CR2的OISx的值。
  uint16_t TIM_OCNIdleState; 
    //空闲状态互补通道电平设置，1或0，与OCIdleState相反，CR2的OISxN的值。
} TIM_OCInitTypeDef;
//输出比较结构体
void TIM_OCxInit(TIM_TypeDef* TIMx, TIM_OCInitTypeDef* TIM_OCInitStruct);
void TIM_OCxPreloadConfig(TIM_TypeDef* TIMx, uint16_t TIM_OCPreload);
```

1. 输入捕获结构体TIM_INInitTypeDef结构体初始化，设置PWM输入模式等。

```c
typedef struct
{
  uint16_t TIM_Channel;
    //捕获频道ICx选择，可选TIM_Channel_x，设定CCMRx的CCxS值
  uint16_t TIM_ICPolarity;
    //输入捕获边沿触发选择，上升下降跳变。CCER的CCxP和CCxNP值
  uint16_t TIM_ICSelection;
    //输入通道选择，可来自三个通道，TIM_IXSelection_DirectTI、TIM_ICSelection_IndierctTI或TIM_ICSelection_TRC，普通输入捕获都可以用，PWM输入只能用通道1、2.
  uint16_t TIM_ICPrescaler;
    //输入通道预分频，1/2/4/8分频，CCMRx寄存器的ICxPSC[1:0]的值，捕获每个边沿，则为1.
  uint16_t TIM_ICFilter;
    //输入捕获滤波器社会猪，0x0-0x0F，设定ICxF[3:0]，一般不用，设为0
} TIM_ICInitTypeDef;
void TIM_PWMIConfig(TIM_TypeDef* TIMx, TIM_ICInitTypeDef* TIM_ICInitStruct);
void TIM_SelectInputTrigger(TIM_TypeDef* TIMx, uint16_t TIM_InputTriggerSource);
void TIM_SelectSlaveMode(TIM_TypeDef* TIMx, uint16_t TIM_SlaveMode);
void TIM_SelectMasterSlaveMode(TIM_TypeDef* TIMx, uint16_t TIM_MasterSlaveMode);
void TIM_ClearITPendingBit(TIM_TypeDef* TIMx, uint16_t TIM_IT);
```

1. 定时器刹车和死区结构体TIM_BDTRInitTypeDef初始化

```c
typedef struct
{
  uint16_t TIM_OSSRState;
    //运行模式的关闭状态选择，设定BDTR的OSSR值
  uint16_t TIM_OSSIState;
    //空闲模式的关闭状态选择，设定BDTR的OSSI值
  uint16_t TIM_LOCKLevel;
    //锁存级别配置，设定BDTR的LOCK[1:0]
  uint16_t TIM_DeadTime;
    //配置死区发生器，定义死区持续时间，0x0-0xFF，设定BSTR的DTG[7:0]的值
  uint16_t TIM_Break;
    //断路输入功能选择，使能或禁止。设定BDTR的BKE值
  uint16_t TIM_BreakPolarity;
    //断路输入通道BRK极性选择，高或低有效，BDTR的BKP值
  uint16_t TIM_AutomaticOutput;
    //自动输出使能，使能或禁止，BDTR的AOE
} TIM_BDTRInitTypeDef;
//断路和死区结构体。
void TIM_BDTRConfig(TIM_TypeDef* TIMx, TIM_BDTRInitTypeDef *TIM_BDTRInitStruct);
```

1. 最后还有主使能输出

```c
void TIM_CtrlPWMOutputs(TIM_TypeDef* TIMx, FunctionalState NewState);
//通用定时器不需要
```

6. 编写中断服务程序，测试程序，读取捕获值，计算脉宽，输出就不需要。




## 5.5 RTC

### 5.5.1 介绍

V~DD~掉电后还可运行，V~BAT~引脚供电，数据保存在RTC模块寄存器，系统复位、电源复位时也不可复位。

32位计数器，只能向上，三种时钟源：HSE/128，LSI、LSE。前二者掉电后会影响，一般使用LSE低速外部时钟，常用2^15=32768。

### 5.5.2 结构框图

![RTC](D:\study\ee\stm32\stm32参考资料\RTC.JPG)

灰色为备份区域，掉电可驱动，包括分频器、计数器、闹钟控制器，若VDD电源有效，可触发RTC_Second,RTC_Overflow,TRC_Alarm。定时器溢出无法配置为中断，待机时由RTC_Alarm和WKUP事件让它推出待机模式。RTC_Alarm在RTC_CNT和RTC_ALR相等时触发。

备份域的寄存器都是16位的，通常把RTCCLK分频得到TR_CLK=TRCCLK/32768=1Hz，计数周期1s，每秒计数器RTC_CNT加1。

系统复位之后，默认禁止访问后背寄存器和RTC：

1）设置RCC_APB1ENR寄存器的PWREN和BKPEN位来使能电源和后备接口时钟。

2）设置PWR_CR寄存器的DBP位使能后备寄存器和RTC访问。

设置后备寄存器可访问之后，第一次通过APB1接口访问TRC时，时钟频率有差异，需等待同步，确保RTC寄存器值正确。若内核要对RTC进行写操作，内核发出指令后，RTC模块在3个RTCCLK后，开始操作。又有RTCCLK的频率比内核低很多，每次需要检查RTC关闭操作标志位RTOFF，置1后，才完成操作。

### 5.5.3 UNIX时间戳

大多数操作系统都是利用时间戳和计时元年计算当前时间的，取统一标准——UNIX时间戳和UNIX计时元年。

后者被设置为1970年1月1日0时-分0秒，前者就是相对于后者经过的秒数。

### 5.5.4 RTC相关库函数

```c
void RTC_WaitForSynchro(void);
//等待时钟同步和操作完成，时钟同步
void RTC_WaitForLastTask(void);
//如果修改了RTC寄存器，需要确保数据已经写入

void PWR_BackupAccessCmd(FunctionalState NewState);
//使能备份域的访问
void RTC_EnterConfigMode(void);
//进入TRC配置模式
void RTC_ExitConfigMode(void);
//退出TRC配置模式

void RTC_SetPrescaler(uint32_t PrescalerValue);
//设置RTC分频配置
void RTC_SetCounter(uint32_t CounterValue);
//设置RTC计数器的值
void RTC_SetAlarm(uint32_t AlarmValue);
//设置RTC闹钟的值
uint32_t  RTC_GetCounter(void);
//获得RTC计数器的值
```



### 5.5.5 配置流程（获取北京时间）

硬件设计：有备份电源，纽扣电池提供；LSE晶振电路得到RTC时钟。

1. 初始化RTC外设

```c
//开启PWR和Bakcup时钟
//允许访问Backup区域
//复位Backup区域
//使用LSE外部时钟或LSI内部时钟，等待准备好
//使能RTC时钟
//对备份域和APB同步，等待同步完成
//使能秒中断，确保上一次操作完成
//分频配置，设置RTC时钟频率为1Hz，确保上一次操作完成
```

2. 设置时间以及添加配置标志

```c
struct rtc_time {
	int tm_sec;
	int tm_min;
	int tm_hour;
	int tm_mday;
	int tm_mon;
	int tm_year;
	int tm_wday;
};//时间管理结构体，方便查看
//c语言中有tm结构体
//进行时间格式转换，将UNIX时间戳转换为常用时间

//配置时间
//RTC配置，确保完成
//写入寄存器，确保完成

//检查配置RTC
```

3. 获取时间

```c
//转换，输出时间

//中断服务函数
```









## 5.6 DWT内核定时器

### 5.6.1 介绍

Cortex-M3有一个外设DWT（Data Watchpoint and Trace)，系统调试和跟踪，有一个32位寄存器CYCCNT，向上计数器，记录内核时钟运行格式，内核时钟跳动一次，计数器加1，F103精度1/72M=14ns

### 5.6.2 寄存器



### 5.6.3 配置流程

1. 使能DWT外设

寄存器清零

使能Cortex-M DWT CYCCNT寄存器



# 6. 存储

## 6.1 DMA

### 6.1 介绍

直接存储器存取，主要功能是搬移数据而不占用CPU的外设，这里的存储器可以是SRAM或FLASH。DMA控制器包括：DMA1有7个通道，和DMA2有5个通道。DMA2只存在于大容量和互联型产品中。

### 6.2 功能框图

![DMA](D:\study\ee\stm32\stm32相关资料\DMA.JPG)

1. DMA请求。外设需要通过DMA传输，需要给DMA控制器发送DMA请求，收到后控制器给外设应答，外设应答后且DMA控制器收到应答信号后，启动DMA传输。

2. 通道。每个通道对应不同的外设DMA请求，每个通道可以接收多个外设的请求，但是同一时间只能接收一个。

3. 仲裁器。发生多个DMA请求时，仲裁器管理顺序。软件DMA_CCRx寄存器设置，硬件取决于通道编号，越低优先级越高。

4. 数据配置。

   数据传输有三个方向。由DMA_CCR位4DIR配置，0外设到存储器，1存储器到外设。外设地址由DMA_CPAR，存储器地址由DMA_CMAR配置。

   ​	外设到存储器：外设寄存器对应ADC数据寄存器地址，存储器地址为我们定义的变量用于存储采集的数据。

   ​	存储到外设：外设对应数据寄存器的地址，存储器地址为自己定义的变量，相当于一个缓冲区。
   
   ​	存储器到存储器：同上，这时候启动DMA_CRR位14：MEM2MEN
   
   数据传输数量和单位。DMA_CNDTR配置，最多65535个数据。源和目标地址数据宽度必须一致，外设数据宽度由DMA_CRR的PSIZE[1:0]配置，8/16/32位，存储器数据宽度由DMA_CRR的MSIZE[1:0]配置，8/16/32位。同时设置两边数据指针的增量模式，外设地址指针DMA——CCRx的PINC配置，存储器由MINC，由具体情况决定。
   
   传输完成。可以产生中断，查询标志位或中断。传输完成分一次传输或循环完成，一次传输就停止，再传输需要关闭使能再重新配置，由DMA_CCR寄存器的CIRC循环模式控制。
   
   ![DMA1 map](D:\study\ee\stm32\stm32相关资料\DMA1 map.jpg)
   
   ![DMA2 map](D:\study\ee\stm32\stm32相关资料\DMA2 map.jpg)

### 6.3 配置流程

1. 使能DMA时钟

```c
RCC_AHBPeriphClockCmd(RCC_AHBPeriph_DMA1, ENABLE);
```

2. 初始化DMA，配置参数。

```c
typedef struct
{
  uint32_t DMA_PeripheralBaseAddr;
    //外设地址，一般设为其数据寄存器地址。
  uint32_t DMA_MemoryBaseAddr;
    //存储器地址，设为自定义存储区的首地址。
  uint32_t DMA_DIR;
    //方向
  uint32_t DMA_BufferSize; 
    //传输数据数目
  uint32_t DMA_PeripheralInc; 
    //外部一般只有一个寄存器，一般不使能
  uint32_t DMA_MemoryInc;
    //存储器地址递增，一般需要使能
  uint32_t DMA_PeripheralDataSize;
    //外设数据宽度，可选字节8、半字16、字32
  uint32_t DMA_MemoryDataSize;
    //存储器数据宽度，可选字节8、半字16、字32
  uint32_t DMA_Mode;
    //传输模式，一次或循环，使用循环。
  uint32_t DMA_Priority;
    //优先级，单个随意，多个有意义。
  uint32_t DMA_M2M;
    //存储器到存储器设置
}DMA_InitTypeDef;   
```

3. 使能DMA，进行传输。

```c
void DMA_Cmd(DMA_Channel_TypeDef* DMAy_Channelx, FunctionalState NewState)
//DMA1_Channelx
```

4. 等待传输完成





## 6.2 FSMC

### 6.2.1 SRAM

STM32内部有SRAM和FLASH存储，反内存不够时需扩展，扩展一般用SRAM和SDRAM，F1仅支持FSMC扩展SRAM，ZE以上才可扩展外部SRAM。

以IS62WV51216为例，

| 信号线     | 类型 | 说明                                        |
| ---------- | ---- | ------------------------------------------- |
| A0-A18     | I    | 地址输入                                    |
| I/O0-I/O7  | I/O  | 数据输入输出，低字节                        |
| I/O8-I/)15 | I/O  | 数据输入输出，高字节                        |
| CS和CS1#   | I    | 片选信号，CS2高电平有效，CS1#低电平有效     |
| OE#        | I    | 输出使能信号                                |
| WE#        | I    | 写入使能                                    |
| UB#        | I    | 数据掩码,upper byte，高字节允许访问，低有效 |
| LB#        | I    | 数据掩码,lower byte，低字节允许访问，低有效 |

掩码可以选择输出高或低位，全部16位数据。

存储器矩阵，512K*16.指定行地址和列地址就可以找到单元格——存储单元。

地址译码器将N跟地址线转换成2^N跟信号线，每跟对应一行、一列单元。

控制电路包含片选、读写使能、宽度控制信号UB#和LB#，利用CS2或CS1#片选，可以把多个SRAM芯片组成一个大容量内存条，OE#和WE#可以控制读写使能。

SRAM读写时序：

 [IS62WV51216.pdf](..\..\chip\板载IC数据手册\IS62WV51216.pdf) 



### 6.2.2 FSMC

(Flexible Static Memory Controller)灵活静态存储控制器，可以驱动包括SRAM、NORFLASH和NAND FLASH类型存储器，不可驱动SDRAM动态存储器，F4中有FMC外设支持SDRAM。

![FSMC](D:\study\ee\stm32\stm32参考资料\FSMC.JPG)

引脚：（一般用GPIOF，GPIOG，144引脚以上芯片）

| 引脚          | 对应SRAM | 说明         |
| ------------- | -------- | ------------ |
| FSMC_NBL[1:0] |          | 数据掩码信号 |
| FSMC_A[18:0]  |          | 行地址线     |
| FSMC_D[15:0]  |          | 数据线       |
| FSMC_NWE      |          | 写入使能     |
| FSMC_NOE      |          | 输出使能     |
| FSMC_NE[1:4]  |          | 片选         |

控制SRAM的有以下三种，各有4个

FSMC_BCR控制寄存器：可配置要控制的存储器类型、数据线宽度、信号有效极性能参数。

FSMC_BRT时序寄存器：配置SRAM访问的时间延时。

FSMC_BWTR写时序寄存器：与上相同，控制写时序的时间参数

时钟控制：挂载在AHB总线上，HCLK-72MHz，异步类型不起作用。

地址映射：其存储单元是映射到STM32内部寻址空间，定义一个指向地址的指针就可以修改该存储单元内容。

![FSMC map](D:\study\ee\stm32\stm32参考资料\FSMC map.jpg)

FSMC将external RAM分成了四个Bank，分配地址范围和使用的存储器类型，每个Bank分成4小块，对应控制引脚连接片选信号。

FSMC控制SRAM时序有地址建立周期、数据建立周期、HCLK周期。



## 6.3 SD卡



## 6.4 内部FLASH







# 7. I/O设备

## 7.1 显示器

### 7.1.1 介绍

TFT-LCD，液晶，控制红绿蓝强度，混合输出不同色彩，液晶本身不发光，需要背光。OLED使用有机发光二极管，不需背光。

参数包括像素，分辨率，色彩深度(bit)，尺寸，点距。电容触摸，16位并口驱动

80并口信号线：

CS:TFTLCD片选信号线

WR:写入数据

RD:读取数据

D[15:0]:16位双向信号线

RST:硬复位TFTLCD

RS:命令/数据标志（0命令，1读写）

并口驱动时序：

模块的8080并口读/写的过程为：

先根据要写入/读取的数据的类型，设置RS为高（数据）/低（命令），然后拉低片选，选中ILI9341，接着我们根据是读数据，还是要写数据置RD/WR为低，然后：

1.读数据：在RD的上升沿， 读取数据线上的数据（D[15:0]）;

2.写数据：在WR的上升沿，使数据写入到ILI9341里面

![image-20210306145829491](C:\Users\86130\AppData\Roaming\Typora\typora-user-images\image-20210306145829491.png)



驱动芯片有很多类型：ILI9341。

例：自带显存，16位模式，D1-5:B;D6-11:G;D13-17:R。四个重要指令，0XD3,0X36,0x2A,0X2B,0X2C,0X2E。

（空缺）

ID指LCD的ID号，FM指帧缓存，即GRAM。



FSMC：

驱动的控制有，地址线、数据线、写信号、读信号、片选信号

TFTLCD通过RS信号决定传输数据或命令，可以理解为地址信号，将TFTLCD当做一个SRAM使用，



源码：

LCD地址结构体，

```c
typedef struct
{
    vu16 LCD_REG,
    vu16 LCD_RAM
}LCD_TypeDef,
#define LCD_BASE
#define LCD ((LCD_TypeDef *)LCD_BASE)
```

LCD_BASE的地址根据电路决定，





## 7.2 电容触摸



## 7.3 触摸屏



## 7.4 ADC-电压采样

### 7.4.1 介绍

STM32F103有3个ADC，精度12位；通道：ADC1、ADC2有16个，ADC3一般8个。

### 7.4.2 功能框图

![ADC](D:\study\32\stm32相关资料\ADC.JPG)

1. 电压输入范围：$V_{REF-}<=V_{IN}<=V_{REF+}$，由$V_{REF-},V_{REF+},V_{DDA},V_{SSA}$引脚决定，一般后SSA、REF-接地，DDA、REF+接3.3V，得到0~3.3V输入电压范围
2. 输入通道：多达18个，16个为ADCx_IN0-15，对应不同IO，ADC1/2/3有内部通道，

![ZET6 ADC IO](D:\study\32\stm32相关资料\ZET6 ADC IO.png)

规则通道：最多16路。

注入通道，可以插入，只有在规则通道存在时出现。

3. 转换顺序。

   规则顺序寄存器有3个，SRQ1、SRQ2、SRQ3，3控制1-6个转换，2控制7-12个转换，3控制13-16转换，

   通道x在第k个转换，就在SQk写x

   注入序列

   JSQR只有一个，最多支持4通道，由其JL[2:0]控制，JL小于4，JSQR和SQR不一样，第一次转换的是JCQRx[4:0]，x=4-JL

4. 触发源

5. 转换时间

6. ADC输入时钟ADC_CLK由PCLK2分频，最大14M，分频由RCC_CFGR位15:14ADCPRE[1:0]设置，2/4/6/8，一般PCLK=HCLK=72M。

   采样时间，ADC_SMPR1和ADC_SMPR2的SMP[2:0]设置，后者控制0~9，前者控制10-17，每个通道可以不同时间采样，最小1.5个

   换换时间T~Conv~=采样时间+12.5周期。一般ADC预分频时钟12M，1.5周期，转换时间1,17us。

   数据寄存器

规则组，ADC_DR，只有一个，32位寄存器，低16位单ADC使用，高16位双模式，ADC精度只有12位，左右对齐由ADC_CR2的11位ALIGN设置。规则通道有16个，多通道时需要转换，或者开启DMA。

注入组，JDRx，4个通道，4个寄存器，32位，低16有效，高16位保留，左右对齐由ADC_CR2的11位ALIGN设置。

7. 中断

数据转换结束后，产生中断，有3种：规则通道转换结束中断，注入转换通道转换而结束中断，模拟看门狗中断。

模拟看门狗中断，ADC转换的模拟电压低于低阈值(ADC_LTR)或高于高阈值(ADC_HTR)。

DMA请求，把转换好的数据存在内存中，只有ADC1和ADC3可以产生。

3. 电压转换

   ADC转换后为12位数，可以转换为0-3.3V。

   

### 7.4.3 配置流程（独立单通道，独立多通道，双重同步）

1. 初始化ADC用到的GPIO。

   GPIO用户AD转换别用配置为模拟输入模式，且用作ADC的IO不可复用。

2. 开启ADC时钟，设置ADC工作参数初始化，设置转换通道顺序及采样时间，

```c
typedef struct
{
  uint32_t ADC_Mode; 
    //只使用一个，独立模式；
  FunctionalState ADC_ScanConvMode;
    //禁止扫描，单通道不需要；
  FunctionalState ADC_ContinuousConvMode;
    //连续转换，ENABLE
  uint32_t ADC_ExternalTrigConv;
    //不用外部触发，软件开启，
  uint32_t ADC_DataAlign;
    //左右对齐，一般右
  uint8_t ADC_NbrOfChannel; 
    //转换通道数量，1个
}ADC_InitTypeDef;
void ADC_Init(ADC_TypeDef* ADCx, ADC_InitTypeDef* ADC_InitStruct);

```

1. 配置使能ADC转化完成中断，中断服务函数，中断内转换数据。使用DMA需要开启DMA接口。
2. 多通道，需要配置时钟分频，配置转换顺序和采样时间。

```c
void RCC_ADCCLKConfig(uint32_t RCC_PCLK2);
void ADC_RegularChannelConfig(ADC_TypeDef* ADCx, uint8_t ADC_Channel, uint8_t Rank, uint8_t ADC_SampleTime);
//SampleTime eg:ADC_SampleTime_55Cycles5
```

1. 使能ADC，使能软件触发ADC转换

```c
void ADC_ITConfig(ADC_TypeDef* ADCx, uint16_t ADC_IT, FunctionalState NewState);
//转换结束产生使能。
void ADC_Cmd(ADC_TypeDef* ADCx, FunctionalState NewState);
void ADC_ResetCalibration(ADC_TypeDef* ADCx);
//初始化ADC校验寄存器，并等待完成
void ADC_SoftwareStartConvCmd(ADC_TypeDef* ADCx, FunctionalState NewState);
//使用软件触发ADC转换
```







## 7.5 DAC-输出正弦波

### 7.5.1 介绍

DAC把输入的数字编码转换为对应模拟电压输出。外设有2个DAC输出通道，可配置为8位、12位数字输入信号，每个通道都可使用DMA、出错检测、外部触发。


### 7.5.2 功能框图

![DAC](D:\study\ee\stm32\stm32参考资料\DAC.JPG)

左侧，V~DDA~接3.3V，V~SSA~接地，V~ref+~接3.3V，规定了参考电压2.4V-3.3V，输入为DORx数字编码，输出为DAC_OUTx，可配置触发源为外部中断源、定时器触发、软件控制触发。DAC每个通道连到特定管脚：PA4-通道1，PA5-通道2，为避免干扰，引脚需要设置为模拟输入功能。

使用DAC时，不能直接写入DORx寄存器，必须写入DHRx，然后根据触发配置进行处理

### 7.5.3 配置流程

硬件设计，PA4，PA5输出通道不可连接其他元件

1. 获得数据表，根据数据计算周期
2. 初始化用到的GPIO，初始化DAC输出通道，初始化DAC工作模式。

```c
typedef struct
{
  uint32_t DAC_Trigger;
    //可设置TIM等为触发源
  uint32_t DAC_WaveGeneration;
    //不使用波形发生器
  uint32_t DAC_LFSRUnmask_TriangleAmplitude; 
    //
  uint32_t DAC_OutputBuffer;
    //不使用DAC缓冲，使能之后可以减小输出阻抗，适合直接驱动外部负载。
}DAC_InitTypeDef;
```

2. 配置DAC用的定时器

```c
void DAC_Init(uint32_t DAC_Channel, DAC_InitTypeDef* DAC_InitStruct);
void DAC_Cmd(uint32_t DAC_Channel, FunctionalState NewState);
void DAC_DMACmd(uint32_t DAC_Channel, FunctionalState NewState);
//使能DAC的DMA请求
```

3. 配置DMA自动转运数据表。





## 7.6 传感器



## 7.7 摄像头驱动



# 8. 看门狗

## 8.1 IWDG-独立看门狗

###  8.1.1 介绍

stm32有独立看门狗和窗口看门狗，前者为12位递减计数器，到0产生复位信号，IWDG RESET，没到零而刷新，则不产生，VDD供电，停机、待机仍能工作。

### 8.1.2 功能框图

![IWDG](D:\study\ee\stm32\stm32参考资料\IWDG.JPG)

1. 时钟由独立RC震荡LSI提供，30-60Hz。
2. 计数器时钟，LSI8位预分频，IWDG_PR设置。计数器时钟CK_INT=40/4*2^PRV
3. 计数器，12位递减，最大0XFFF，减到0产生IWDG_RESET，重启动。
4. 重装载寄存器，12位，计算器要刷新的值。$Tout=(4*2^{prv})/40*rlv(s)$
5. 键寄存器

| 键值   | 作用               |
| ------ | ------------------ |
| 0XAAAA | 把RLR的值重装到CNT |
| 0X5555 | PR和RLR寄存器可写  |
| 0XCCCC | 启动IWDG           |

0XCCCC开启属于软件启动，不可关闭，除非复位。

6. 状态寄存器，位0：PVU和位1：RVU有效，硬件操作

### 8.1.3 配置流程

1. 配置IWDG

```c
void IWDG_WriteAccessCmd(uint16_t IWDG_WriteAccess);
//使能预分频寄存器PR和重载寄存器RLR可写
#define IWDG_WriteAccess_Enable     ((uint16_t)0x5555)
void IWDG_SetPrescaler(uint8_t IWDG_Prescaler);
//设置预分频值，IWDG_Prescaler_x
void IWDG_SetReload(uint16_t Reload);
//设置重装载寄存器的值
void IWDG_ReloadCounter(void);
//把重装载的值放入计数器
void IWDG_Enable(void);
//使能IWDG
//例，64预分频和625重装载，则溢出时间，60/40*625=1s
```

2. 喂狗函数，一般在主函数中使用

```c
void IWDG_Feed(void)
{
	// 把重装载寄存器的值放到计数器中，喂狗，防止IWDG复位
	// 当计数器的值减到0的时候会产生系统复位
	IWDG_ReloadCounter();
};
```

## 8.2 WWDG-窗口看门狗

### 8.2.1 介绍

窗口看门狗，递减计数器减到某个特定值时复位，减到某个数之前喂狗也会复位，分别为窗口上下限，上限可设置，下限0X40。



### 8.2.2 功能框图

![WWDG](D:\study\ee\stm32\stm32参考资料\WWDG.JPG)

1. 时钟来自PCLK1，最大36M。
2. 计数器时钟经过预分频，CFR的位8:7 WDGTB配置，0/1/2/3，CK计时器时钟=PCLK1/4096，所以CNT_CLK=PCLK1/4096/2^WDGTB，T=1/CNT_CLK。
3. 计数器，7位递减，0X40是可以低贱的最小值，只能是0X40-0X7F，减到0X40时不会立刻产生，若使能提前唤醒中断安：CFR位9EWI置1，产生提前唤醒中断。
4. 窗口值，上窗值根据需求设定。

![WWDG t](D:\study\ee\stm32\stm32参考资料\WWDG t.JPG)

### 8.2.3 配置流程

WWDG用来监测，外部干扰等造成程序背离正常产生故障。

1. 开启时钟，配置参数

```c
void WWDG_SetCounter(uint8_t Counter);
// 设置递减计数器的值
void WWDG_SetPrescaler(uint32_t WWDG_Prescaler);
// 设置预分频器的值
void WWDG_SetWindowValue(uint8_t WindowValue);
// 设置上窗口值
void WWDG_Enable(uint8_t Counter);
//使能WWDG
```

2. 使能WWDG，清楚提前唤醒中断位，配置NVIC中断，开启中断

```c
void WWDG_ClearFlag(void);
// 清除提前唤醒中断标志位
//配置WWDG中断
void WWDG_EnableIT(void);
//开WWDG中断
```

3. 提前唤醒函数，死前中断，必须在再次减一、复位前完成中断服务函数需要做的事。

```c
void WWDG_IRQHandler(void)
{
	// 清除中断标志位
	WWDG_ClearFlag();

	//真正使用的时候，这里应该是做最重要的事情
}
```



# 9. 电源管理

## 9.1介绍







# 10. 电机

- 电机(motor)把电能转换为机械能的电气设备。


## 介绍

- 分类

1. 直流电机

   分普通直流、直流减速，有刷、无刷。

2. 步进电机

   将脉冲信号转换为角位移或线位移的开环控制电机。分为反应式、永磁式、混合式。

3. 伺服电机

   伺服系统中被控制的电机。

4. 舵机

另外有驱动机。

## stm32定时器

电机控制分为电压控制和电流控制。利用MCU精准定时控制，MCU的定时器输出PWM驱动模块，驱动电机



## 直流有刷电机

Brushed DC motor









## 控制系统与电机的关系





## PID

### 介绍

Proportional(比例)、Integral(积分)、Diffenrential(微分)，闭环控制算法。

连续理想的PID控制规律：

$u(t)=K_p(e(t)+\frac{1}{T_t}\int ^t_0 e(t)dt+T_D\frac{de(t)}{dt})$

Kp——比例增益，与比例度成倒数关系

Tt——积分时间常数

TD——微分时间常数

u(t)——PID控制输出信号

e(t)——给定r(t)与测量值误差









## arm cortex

### coresight

