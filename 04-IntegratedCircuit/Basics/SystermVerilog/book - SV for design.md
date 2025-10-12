---
dateCreated: 2024-10-25
dateModified: 2025-04-03
---
# Ch1 介绍

> Systemverilog 是 Verilog-2005 的扩展
> 并且成为 Accellera 标准，它基于已证明的技术

> Systemverilog 扩展了建模和验证（3.1a）能力，是第三代 Verilog。
> IEEE 1800-2005 是官方 SV 标准，IEEE 1364-2005 是其基本语言

SV for verification 是增强的断言和测试验证方法。

IEEE Std 1800-2005 SV LRM 是 SV 的 IEEE 标准。

IEEE Std 1364-2005 V LRM 是 Verilog 的 IEEE 标准。

2002 IEEE Std for Verilog RL Synthesis 基于 RTL 的可综合 Verilog HDL 标准语法和语义是官方说明。

Writing Testbenchs Using SV 解释了很多 SV 为验证加入的测试平台扩展。

A Practical Guide for SV Assertions 是断言的介绍。

SV Assertion Handbook 利用断言的技术。

Assertion Based Design 2 对验证断言更多。

The Verilog Hardware Description Language 5 th 完整关于 Verilog 的表述。

Verilog Quickstart, A practical guide to simulation and synthesis 3 rd 是基础 Verilog 书。

Verilog 2001, A Guide to the New Features of the Verilog HDL 2002 是对 2001 扩展的整体介绍。

其改进包括：

1. 设计内部的封装通信和协议检查的接口，编程语句增强。
2. 类似 C 语言中的数据类型，如 int；++, --, += 等其他赋值操作
3. 枚举、结构体、联合；用户定义类型，使用 typedef；类型转换
4. 可被多个设计块共享的定义包 package
5. 外部编译单元区域 scope 声名
6. 显式过程块
7. priority unique 修饰符
8. 通过引用传送到任务、函数、模块。

# Ch2 声名的位置

用到四种新类型：

1. `logic`，四态变量，类似 `reg`，可以声明为任意大小的向量。
2. `enum`，由符号表示的值的枚举型 `net` 或变量，类似 C 的枚举，但是有为硬件建模的附加语法和语义。
3. `typedef`，用户定义的数据类型，由内建类型或者用户定义类型构成。
4. `struct`，可单独或同时引用的变量的集合体

## Package

Verilog 要求变量、net、任务、函数的声名都在模块内部，即相对模块局部声名。也支持命名块 `begin end/fork join`、任务、函数内定义局部变量，只能在模块内定义，为了综合，只能在模块内访问。也不支持全局声名，如一个函数要用到必须在每个块内声名。

### 2.1.1 定义

> [!note] System Verilog 可以使用 `typedef` 定义用户定义类型。
> 为了使多个模块共享用户定义类型的定义，增加了 `Package `，包中可包含和可综合的结构有：
> - parameter/local parameter/const 变量/typedef 用户定义类型
> - auto task/auto function/从其他包 Import/操作符重载定义
> 包是独立的声名空间，不需要包含在 Verilog 模块中。

包中可以进行全局变量声名、静态任务定义、静态函数定义，都是不可综合的。

```verilog
package defs:
	parameter V = "1.1";
	typedef enum {ADD, SUB, MUL} opcodes_t;
	typedef struct {
		logic [31 : 0] a, b;
		opcodes_t opcde;
	} instruction_t;
	function automatic [31 : 0] mul(input [31 : 0] a, b);
		return a * b;
	endfunction
endpackage
```

`module` 的每个实例可以对 `parameter` 重定义，但不能对 `localparameter` 直接重定义。`package` 中的 `parameter` 不能重定义因为它不是模块实例的一部分，其中的 `parameter/local parameter` 是相同的。

### 2.1.2 引用包

模块和接口可以用四种方式引用包中的定义和声名：

1. 范围解析操作符直接引用。
2. 将包中特定子项导入。
3. 用通配符导入包中的子项导入。
4. 将包中子项导入 `$unit$` 声名域。

```verilog
module ALU (
	input defs::instruction_t IW,
	input logic clk,
	output logic [31 : 0] res
);
always_ff @(posedge clk) begin
	case (IW.opcode)
		defs::ADD: result = IW.a + IW.b;
		defs::SUB: result = IW.a - IW.b;
		defs::MUL: result = defs::mul(IW.a, IW.b);
	endcase
end
endmodule
```

> [!note] 显式包引用有助于源代码的可读性。

也可以导入特定子项，如 `import defs::ADD`，使得包中子项局部可见，然后就可以直接使用 `ADD`。注意，导入 `opcode_t` 使得其在模块可见，但是不会使其内部的枚举元素可见，还需要导入每个枚举元素。

使用通配符 `import defs::*` 可使包中所有子项可见，不过也不能自动导入包中所有内容，只有实际用到的子项才会被真正导入。

模块或接口内的局部定义或声名**优先于**通配符导入。包中指定子项名称的导入也**优先于**通配符导入。通配符导入只是简单地将包添加到标识符的搜索规则中，软件工具先搜索局部声名然后在导入到包中搜索，最后在 `$unit` 声名域中搜索。

### 2.1.3 综合指导

为了综合，**包中定义的函数和任务必须是声明为 `automatic`，且不能包含静态变量**。因为自动任务或函数的存储区在每次调用时才会分配。因此引用包中的自动任务或函数的每个模块看到的是不被其他模块共享的该任务或函数存储区的唯一副本。保证了综合前对包中任务或函数的引用的仿真行为与综合后的行为相同。综合后，这些任务或函数的功能就在引用的一个或多个模块中实现。

由于类似原因，**综合不支持包中变量声名**。仿真时，包中的变量会被导入该变量的所有模块共享。一个模块向变量写值，另一个模块看到的就是新值，这类不通过模块端口传递数据的模块间通信时不可综合的。

## 2.2 `$unit` 编译单元声名

## 未命名语句块中的声名

## 仿真时间单位和精度

# Ch3 文本值和数据类型

> SV 扩展了 Verilog 的数据类型，增强了指定文本值的方法

## 3.1 增强的文本值赋值

Verilog 中一个向量可以赋值全 0、全 X（不定）、全 Z（高阻态）。赋值可扩展，即赋值扩展到变量的位宽，但是赋全 1 需要额外的技巧，如

```verilog
parameter SIZE = 64;
reg [SIZE - 1 : 0] data;
data = ~0; // 1 的补码
data = -1; // 2 的补码
```

> [!NOTE] SV 给一个向量赋予特殊的文本值
> SV 有更简单的语法，只要指定值，不用指定进制；而且可以是逻辑 1。
> 用 `'1 '0 'z 'x` 表示，文本值随左手向量宽度扩展。

## 3.2 \`define 增强

>SV 扩展了 Verilog 宏文本替换，可以包含特殊字符。

Verilog 中 `define` 宏中使用 `""` 双引号内部的文本是文本串。如下，双引号内的变量不会被替换。
```verilog
`define print(v)\
	$display("var v = %h", v);
`print(data);

$display("var v = %h", data);
```
>[!note] SV 允许双引号内的宏变量替换，需要加 `
```systemverilog
`define print(v)\
	$display(`"var v = %h`", v);
`print(data);

$display("var data = %h", data);
```
verilog 中为了不影响字符串的双引号，字符串内嵌的引号必须加转义符 `\`，字符串包含变量替换的文本替换宏时，嵌入的引号要使用 \`\ \`"
>在宏中，SV 可以使用两个连续的重音符号 ``，使得两个或多个文本宏连接
```verilog
bit d00_bit; wand d00_net = d00_bit;
...
bit d63_bit; wand d63_net = d63_bit;

`define TWO_STATE_NET(name) bit name``_bit;\
	wand name``_net = name``_bit;
`TWO_STATE_NET(d00)
...
`TWO_STATE_NET(d63)
```

# Ch4 用户自定义和枚举数据类型

# Ch5 数据、结构体、联合体
## 5.1 结构体

结构体使用 `struct` 关键字声名，结构体内的成员可以是任何数据类型，包括用户自定义类型和其他结构体类型，例：

```verilog
struct {
	int a, b;
	opcode_t opcode;
	logic [23 : 0] addr;
	bit error;
} Instr_Wrod;
```

结构体是一个名称下的变量和/或常量的集合。整个集合用结构体名进行引用。结构体内的每个成员有一个名称用来选择。

### 5.1.1 结构体声名

> [!note] 结构体可以是变量或线网。

## 5.3 数组
### 5.3.1 非压缩数组

Verilog 数组的声名如下，且允许多维数组、变量和线网的数组。

```verilog
reg[15 : 0] RAM [0 : 4095];
```

但是限制一次访问只能访问数组的**一个元素**或者一个元素的一位或部分位，试图访问多个元素是错误地。对于**非压缩数组**，SV 参考 Verilog 的数组声名风格，其虽然有相同的数组名称，但数组的各个元素的存储是**独立**的。

> [!note] SV 允许任何数据类型的非压缩数组。
> 可以使用 `event` 数据类型，以及所有 SV 数据类型，包括 ` logic, bit, byte, int, longint, shortreal, real ` 以及使用 ` typedef ` 声名用户自定义类型和 ` struct, enum ` 类型非压缩数组。

> [!note] SV 可以引用整个数组或者其中一段。
> 但等号左边和右边必须有相同的架构和类型。

- 非压缩数组声名的简化
Verilog 数组要求指定地址范围。而 SV 允许非压缩数组只指定维度，不用指定起始地址。

```verilog
int array[64 : 83];
logic [31 : 0] data [1024]; // 地址从零到 （宽度-1）结束
```

### 5.3.2 压缩数组

Verilog 允许创建超出一位的向量如 `wire reg`，压缩数组的向量范围在信号名称前面，非压缩数组的范围在信号名称后面。

SV 将向量声名看成压缩数组，一个 Verilog 向量就无处一个一维压缩数组。SV 可以声名多维压缩数组。

```verilog
wire [3 : 0]sel;
logic [3 : 0][7 : 0] data; //SV 二维压缩数组
```

> [!note] SV 压缩数组整个数组的存储是位连续的，没有间隙。
> 上面的压缩数组作为临近元素存储：
> 31 `data[3][7 : 0]` 23 `data[2][7 : 0]` 15 `data[1][7 : 0]` 7 `data[0][7 : 0]` 0

# Ch6 过程块、任务、函数

> Verilog 中的过程块 always 是通用的，可以描述各种硬件和验证，但是意图不明显。
> SV 增加了专用过程块，减少不确定性，还对任务函数做了很多改进。

## 6.1 Verilog 通用目的 Always 过程块

Verilog 的 always 过程块是一个能重复执行块中语句的无限循环。

- 敏感表
过程块开头所写的边沿敏感事件控制条件认为是这个过程块的敏感表，只有满足该事件条件下，过程块的语句才能被执行。

> 边沿事件控制可用作敏感表。

- 一般用法
RTL 级，always 可以用作组合逻辑、锁存逻辑、时序逻辑的建模。

> always 可以表示任何逻辑类型

- 从过程块推断具体实现
为了减少通用 always 过程块在硬件推断方面的不确定性，综合编译工具对 always 块的使用设定了一系列限制和指导原则（IEEE 1364.1）。
组合逻辑：必须接边沿敏感事件控制、不能包含 edge；必须列出所有输入；过程块不能包括其他事件控制；所有赋值变量必须随可能的输入组合变化而更新；此过程块的变量不能在其他过程块中再次赋值。
锁存逻辑：必须接边沿敏感事件控制、不能包含 edge；过程块不能包括其他事件控制；过程块中的变量至少有一个不能被某些输入条件更新；此过程块的变量不能在其他过程块中再次赋值。
时序逻辑：必须接边沿敏感事件控制、必须包含 edge；过程块不能包括其他事件控制；此过程块的变量不能在其他过程块中再次赋值。

> 工具必须从过程块内容中推断设计意图。但对于通用目的的过程块而言，建模指导规则不能被强制执行。

## 6.2 SV 特有的过程块

SV 增加了特有的过程块以减少建模的不确定性，这些块都是无限循环、可综合的，即 `always_comh, always_latch, always_ff`。特有的过程块反映了设计意图。

### 6.2.1 组合逻辑

`always_comb` 不需要指明敏感表，是自动推断的。推断的敏感表包含了所有被过程块读取（出现在表达式右边或作为条件语句的条件表达式）并在块外复制的信号，还包括调用函数的所有信号，但在次过程块中被阻塞赋值且只在该过程块读取的临时变量不包括在敏感表中，被函数赋值和读取的临时变量也除外。

> `always_comb` 仍要求被赋值的变量不能再次出现在其他过程块被赋值。
> 禁止出现共享变量。

内容和过程块类型的匹配检查是可选的，有些工具会执行而有些不会。

- 零时刻自动求值
在所有 initial always 块启动后，`always_comb` 块会在仿真零时刻自动触发，不管敏感表的信号是否发生变化。这种语义确保了组合逻辑在零时刻产生与输入相对应的输出结果，尤其是使用默认值为逻辑 0 的两态变量建模时，复位信号可能不会引起组合逻辑的敏感表中的信号变化，而没有变化，通用 `always` 过程块不会被触发，输出也不会变化。
- `always_conb` 和 `always@*` 的比较
组合逻辑块会调用函数，但 `always@*` 对调用函数中读取的信号不能推断为敏感值，可能推断出不完整的敏感表。
`always_comb` 对块内读取的信号和块内调用的函数读取的信号都敏感，

### 6.2.2 锁存逻辑

`always_latch` 与 `always_comb` 语义相同，工具也会检查内容需要表示锁存逻辑

### 6.2.3 时序逻辑

> `always_ff` 表述时序逻辑

其敏感表必须明确列出，以判断复位/置位是同步还是异步。还会强制生成可综合敏感表。

### 6.2.4 综合指导

> 应该使用特有的可综合的过程块。以避免潜在的错误。

## 6.3 Function and Task 的改进

### 6.3.1 隐式语句组
- Verilog 需要 `begin end` 写多条语句，任务也可以使用 `fork join`。
- SV 不需要 `begin end` 来打包，其中的语句顺序执行。

> begin… end 将多条语句分组
> SystemVerilog 推断 begin… end

### 6.3.2 函数返回值
- Verilog 中，函数名本身是一个与该函数类型相同的变量。函数的返回值通过对函数名赋值产生。执行到函数末端退出函数，最后赋给函数名的值就是函数的返回值。
- SV 增加了 return 语句。

> function 创建一个与其名称和类型一样的隐含变量

> [!NOTE] return 的优先级高于返回函数名的值
> 二者都可以使用，但 return 优先级更高，即使使用了 return，函数名仍是一个变量，可以当局部变量使用。

### 6.3.3

# Ch7 过程语句

## 7.9 改进的 case 语句

Verilog 中的 `case casex casez` 允许多个选项中选择一个逻辑分支。关键字后的表达式叫**条件表达式**，用来和条件表达式匹配的表达式叫做**条件选项**。Verilog 规定 `case ` 语句必须按照列举顺序计算条件，选项间存在优先级，如果编译器判断所有选项是互斥，通常会优化多余优先级判断。SV 添加了 `unique priority` 修饰符。

> 仿真和综合工具可能对 case 语句的解释不一样

### 7.9.1 Unique case 条件判断

`unique case` 语句指定：

- 只有一个条件选项与条件表达式匹配。
- 必须有一个条件选项和条件表达式匹配。

> [!note] unique case 可以并行求值
> 条件选项的顺序并不重要，可以并行求值；条件选项必须完整，表达式与一个且仅一个选项匹配；每个选项必须互斥。

> [!note] 在 always_comb 中使用 unique case
> always_comb 保证仿真时产生组合逻辑行为。

### 7.9.2 Priority case 语句

# Ch8 FSM 建模
## 8.1 使用枚举类型建立状态机模型

> [!note] 枚举类型有固定数值，可以有显式基类、显式值
> 枚举类型提供了一种定义一个具有有限合法数值集合的变量的方法。数值用标签而不是数字逻辑值表示
> 枚举类型支持高抽象层次的建模，能描述精确、可综合的硬件。四态类型如 logic 可作为基类。

```verilog
module traffic_light(
	output logic g_l, y_l, r_l,
	input sensor,
	input [15 : 0] g_cnt, y_cnt,
	input clk, rst_n
);
enum bit [2 : 0] {
RED = 3'b001,
GREEN = 3'b010, 
YELLOW = 3'b100
} state, next;
always_ff @(posedge clk, negedge rst_n)
	if (!rst_n) state <= RED;
	else state <= next;

always_comb begin : set_next_state
	next = state;
	unique case (state)
		RED: if (sensor) next = GREEN;
		GREEN: if (g_cnt == 0) next = YELLOW;
		YELLOW: if (y_cnt == 0) next = RED;
	endcase
end : set_next_state

always_comb begin: set_output
	{g_l, y_l, r_l} = '0;
	unique case (state)
		RED: r_l = 1'b1;
		GREEN: g_l = 1'b1;
		YELLOW: y_l = 1'b1;
	endcase
end: set_output
endmodule 
```

默认，枚举基类 `int`，每个枚举值标签（0，1，2）默认值，可能不能精确反应硬件行为。` int ` 是一个 32 位两态类型，实际硬件有三个状态，只需要 2 位或 3 位向量。实际的门级有四态，两态仿真的默认初值会掩盖设计问题，标签的默认值可能导致 RTL 和门级仿真不一致，四态类型默认初始值是 X 而不是 0。

> [!note] one-hot 状态机可以使用反向 case 语句
> 条件表达式是要匹配的文本，如 1 位 1，条件选项是状态变量的每一位。

```verilog
enum {
R_BIT = 0,
G_BIT = 1,
Y_BIT = 2} state_bit;
always_comb begin
	next = state;
	unique case (1'b1)
		state[R_BIT]: ...
		state[G_BIT]: ...
		state[Y_BIT]: ...
	endcase
end
```

为 one-hot 码状态机的状态寄存器 case 语句添加 unique，可以简化逻辑、避免错误。

### 未使用的值

变量可能会有枚举列表中未定义的逻辑值。使用枚举类型和 uniquecase 语句结合，就不需要特定的编码需求了，如给 default 定义 xxx，给枚举基类定义 xxx。枚举类型将变量的值限定在其定义的值得集合中，case 中列出这些数值，加上 unique case 得语法检查，就能确保 RTL 模型和综合得门级模型仿真一致。

> [!note] 枚举类型变量只能被赋值其类型集合中的值
> 不能直接赋值文本值，不能对某一位单独赋值。如果要文本赋值，需要静态强制类型转换或动态强制类型转换。

## 使用 2 态数据类型

使用两态数据类型时，可能出现变量默认为 0 而不是 X，这样即使没有插入复位或者复位逻辑有错误，设计好像仍然复位了。另一种情况是，变量使用了缺省值，使得状态没有变化，状态无法转移。

这种状态锁定问题可以通过两种方法解决。第一，使用四态基类如 `logic` 显式声明枚举变量，这样仿真开始时变量有非初始值 X，应用复位后，变量从 X 变为初始复位值。第二，使用默认基类和标签枚举，always_comb 块，即使敏感值没有变化，仿真开始过程块也会执行。

# Ch9 层次化设计
## 9.1 模块原型

？

## 9.2 命名的结束语句

SV 允许在关键字 `endmodule` 后指定模块的名字，这个名称必须与匹配的模块名称一致。SV 也支持对其他命名代码块指定结尾名称，包括 `interface/task/function`

## 9.3 嵌套模块

？

## 9.4 简化的模块实例网表

Verilog 提供了两种连接模块实例的代码风格：按端口次序和按端口名称。端口名称连接方式事更好的语法风格但是连接方式结果冗长。

# Ch10 接口
