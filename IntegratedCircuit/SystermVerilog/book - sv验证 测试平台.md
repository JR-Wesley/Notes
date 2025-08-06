SystemVerilog验证平台
<a href="https://www.zhihu.com/column/c_1730321185194160128">对绿皮书的</a>

# 一、验证
验证方法学

# 二、数据类型

SV的改进数据结构，具有的有点：



## 1. 内建数据类型
基本类型` wire, net` 有四种取值`0, 1, z, x` RTL使用变量来存放组合和时序值，变量可以是单比特或多比特的无符号数`reg[7:0]m`，32比特有符号数`integer`，64位无符号数`time`或浮点数`real`。若干变量可以一起存放在定宽数组。所有存储都是静态的，在整个仿真过程中存活，子程序`routine`不能通过堆栈来保存形参和局部参数。`wire`通常用来连接不同部分。

### `logic`
任何使用线网的地方都可以使用`logic`但不能有多个结构性驱动，如双向总线需要定义为`wire`，SV会对多个数据来源进行解析以后确定最终值。

> 多个驱动的连接不能使用logic，必须使用连线类型wire

### 双状态数据类型
相比四状态，双状态有利于提高仿真器性能并减少内存使用。
最简单的有`bit`无符号，带符号的有`byte, shortint, int, longint`
![[assets/SV 验证测试平台/2.1 双状态.png]]
## 2. 定宽数组

### 声明和初始化

要求声明中必须给出上下界，可以在变量名后指定维度创建多维定宽数组.

若访问越界，将会返回数组元素类型的缺省值（`logic`是`x`，双状态是`0`），这适用于所有数组类型，包括定宽数组、动态数组、关联数组和队列，也适用于地址中含有`x/z`，线网未驱动是`Z`。

仿真器通常在存放数组元素时使用32比特的字边界，所以`byte, shortint, int`都是存放在一个字中，而`longint`存放在两个字。

非合并数组，字的低位用来存放数据，高位不用。

仿真器通常使用两个或两个以上的连续的字来存放`logic, integer`等四状态类型，这比双状态变量多占用一倍空间。
![[assets/SV 验证测试平台/2.1 非合并.png]]


### 常量数组

```verilog
int ascend[4] = '{0, 1, 2, 3};
int desend[5];

descend = '{4, 3, 2, 1, 0};
descend[0 : 2] = '{5, 6, 7};
ascend = '{4{8}};				// 4个值全为8
descend = '{9, 8, default:1};	// 指定缺省值
```
- 数组操作`for foreach`

```verilog
initial begin
    bit[31:0] src[5], dst[5];
    for(int i=0; i < $size(src); i++)
        src[i] = i;
    foreach (dst[j])
        dst[j]=
end
```

### 动态数组





### 队列





### 关联数组





# 三、过程语句与子程序

## 1.  过程语句

SV从C/C++中引入了很多操作符和语句，





## 2. 任务、函数及 void 函数





## 3. 任务与函数









## 7. 时间值

### 时间单位和精度

当依赖`timescale`时，编译文件时就必须按照适当的顺序以确保所有时延都采用适宜的量程和精度。`timeunit/timeprecision`声明语句可以明确地为每个模块指明时间值，以免含糊不清。如果使用这些代替，就必须把他们放到每个带有时延的模块里。

### 时间参数

SV允许使用数值和单位来明确指定一个时间值。只要使用







# 四、连接设计和测试平台

## 4.1 将测试平台和设计分开

使用模块来保存测试平台经常引起驱动和采样时的时序问题，SV引入了`program block`程序块，从逻辑和时间上来分开测试平台。

模块之间的连接也很复杂，使用接口可以代表一捆连线，智能同步和连接功能。一个接口可以像模块那样例化，也可以像信号一样连接端口。

### 与端口的通信

```verilog
// 使用端口的DUT和测试平台、顶层网表
module arb_port(output logic [1:0] grant,
                input logic [1:0] request,
               input logic rst,
                input logic clk);
    ...
    always@(posedge clk or posedge rst) begin
        if(rst)
            grant <= 2'b00;
        else
            ...        
    end 
endmodule
            
module test(input logic [1:0] grant,
            output logic [1:0] request,
            output logic rst,
            input logic clk);
    initial begin
        @(posedge clk) request <= 2'b01;
        $display("@%0t: Drove req=01", $time);
        repeat(2) @(posedge clk);
        if(grant!= 2'b01)
            $display("@%0t: a1:grant!= 2'b01", $time);
        ...
        $finish;
    end
    ...
endmodule
            
module top;
    logic [1:0] grant, request;
    bit clk, rst;
    always #5 clk = ~clk;
    
    arb_port a1(grant, request, rst, clk);
    test t1(grant, request, rst, clk);
endmodule
```

## 4.2 接口

SV使用接口为块之间的通信建模，接口可以看作一捆智能的连线。接口包含了连接、同步、甚至两个或者更多块之间的通信功能，连接了设计块和测试平台，也有设计级的接口。

### 使用接口来简化连接

对仲裁器的一个改进就是将连接捆绑成接口，接口扩展到了这两个块中，包含了测试平台和DUT的驱动和接收。时钟可以是接口的一部分或者一个独立端口。

最简单的接口仅仅是一组双向信号的组合。这些信号使用logic数据类型，可以使用过程语句驱动。

```verilog
// 使用接口的仲裁器
interface arb_if(input bit clk);
    logic [1:0] grant, request;
    logic rst;
endinterface

module arb(arb_if arbif);
    ...
    always@(posedge arbif.clk or posedge arbif.rst)
        begin
            if(arbif.rst)
                arbif.grant <= 2'b00;
            else
                arbif.grant <= next_grant;
            ...            
        end
endmodule

// 使用接口，简化测试平台

// 使用接口的顶层，需要在模块和程序之外声明接口变量。
module top;
    bit clk;
    always #5 clk = ~clk;
    
    arbif arif(clk);
    arb a1(arbif);
    test t1(arbif);
endmodule : top
```

使用接口，需要在模块和程序之外声明接口变量。有些编译器不支持在模块中定义接口；即使允许，接口只是局部变量，对其他设计不可见。



## 4.8 断言

使用SVA在设计中创建时序断言。断言的例化和其他设计块的例化相似，而且在整个仿真过程有效。仿真器会跟踪哪些断言被激活，在此基础上收集功能覆盖率的数据。

### 1 立即断言 Immediate Assertion

测试平台的过程代码可以检查待测设计的信号值和测试平台的信号值，并在存在问题时采取响应的行动。如，如果产生了总线请求，期望在两个时钟周期后产生应答。

```verilog
// 使用if语句检查一个信号
bus.cb.request <= 1;
repeat(2) @bus.cb;
if(bus.cb.grant != 2'b01)
    $display("Error, grant != 1");
```

断言比if更紧凑，但断言逻辑跟if语句里的比较条件相反。

```verilog
bus.cb.request <= 1;
repeat(2) @bus.cb;
a1: assert (bus.cb.grant == 2'b01);
```

正确产生了grant信号，则继续；表达式为假时输出错误信息：

```shell
"test.sv",7:top.t1.a1: started at 55ns failed at 55ns
offending '(bus.cb.grant == 2'b1)'
```

该消息指出，在test.sv文件的第7行，断言top.t1.a1在55ns开始检查信号，但是检查出现了错误。

断言是声明性的代码，其执行过程和过程代码有很大差异。使用几行断言可以验证复杂的时序关系；等价的过程代码可能远比这些断言要复杂冗长。

### 2. 定制断言行为





# 五、面向对象编程基础

发生器generator创建事务并且将它们传给下一级，驱动器driver和设计进行会话，设计返回的事务会被监视器monitor捕获，计分板scoreboard会将不会的结果和预期的结果比较。

## 5.1 编写类

```verilog
class Transaction;
    bit [31:0] addr, crc, data[8];
    
    function void display;
        $display("Transaction:%h", addr);
    endfunction : display
    
    function void calc_crc;
        crc = addr ^ data.xor;
    endfunction : calc_crc
endclass : Transaction
```

可以把类定义在program, module, package中，或者这些块之外的任何地方。类可以在程序和模块中使用。可以将程序块当作一个包含了测试代码的模块，含有一条测试、组成测试平台的对象及创建、初始化并运行测试的初始化块。使用package可以将一组相关的类和类型的定义绑在一起。

`class`包含变量和子程序的基本构建块，Verilog中对应的module。

`object`类的实例，需要实例化一个模块才能使用。

`handle`指向对象的指针，Verilog中通过实例名在模块外部引用信号和方法。一个OOP句柄就像一个对象的地址，但是保存在一个指向单一数据类型的指针中。

`property`存储数据的变量。Verilog中为reg/wire类型信号。

`method`任务或者函数中操作变量的程序性代码。Verilog模块除了initial always块外，还有任务和函数

`prototype`程序的头，包括程序名、返回类型、参数列表，程序体包含了执行代码。
