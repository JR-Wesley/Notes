
[`#历史内容集合 (qq.com)](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=Mzg5NzU5NDExNA==&action=getalbum&album_id=2893185282286862338#wechat_redirect)



# 仲裁器
在使用**多主设备的总线过程**中，需要考虑到不同主设备申请**总线控制权的优先级问题**。常见的仲裁策略有三种：

| 固定优先级仲裁器                                    | 循环仲裁器                                | 循环优先级仲裁器                                     |
| ------------------------------------------- | ------------------------------------ | -------------------------------------------- |
| 给每个主机分配固定的优先级，在多个设备发起请求时，始终将总线权限分配给高优先级的主机  | 通过一个指针对各个主机的请求进行轮询                   | 根据上一次仲裁的结果进行优先级的重新分配，并应用于下一次的仲裁              |
| 其特点在于原理、结构和实现都很简单，但是可能出现低优先级主机迟迟无法获得总线权限的情况 | 各个端口获得总线的机会基本上是平等的，但是但主机数较多时，遍历的效率较低 | 每个主机获得总线的机会基本平等，而且只要有请求，在下一个周期必然可以输出仲裁结果，效率高 |
| 使用组合电路即可实现                                  | 使用一个计数器即可实现                          |                                              |

## 固定优先级仲裁器
以优先级排序为A>B>C>D为例，input 请求情况request 的四位二进制，从高到低也分别代表主设备A、B、C、D的总线控制请求，output的grant输出one hot编码，即“1000，0100，0010，0001”四种情况中的一种，给出的案例如下，按照四个周期依次输入控制请求，仲裁器在按照固定优先级算法的条件下会依次响应设备A，D，A，B。

| 周期  | 请求情况(request) | 优先级排序 | 响应情况(grant) |
| --- | ------------- | ----- | ----------- |
| 1   | 1010          | ABCD  | 1000(A)     |
| 2   | 0001          | ABCD  | 0001(D)     |
| 3   | 1111          | ABCD  | 1000(A)     |
| 4   | 0011          | ABCD  | 0010©       |
对于**仲裁器**来讲，它是一个**纯组合逻辑电路**。
### 实现
case/if
使用if去判断从高位到地位去判断request。这样的实现有判断优先级。靠前的逻辑少、路径短，靠后的逻辑多、路径长。
使用case语句既可以并行，没有优先级，也可以实现优先编码。
```verilog
module fixed_prio_arb(
	input [3:0]         request,
	output logic [3:0]  grant
);

reg    [3:0] grant_reg;

always_comb begin
	case(1'b1)
		request[3] : grant_reg = 4'b1000;
		request[2] : grant_reg = 4'b0100;
		request[1] : grant_reg = 4'b0010;
		request[0] : grant_reg = 4'b0001;
		default: grant_reg = 4'b0000;
	endcase
end

assign grant = grant_reg;

endmodule : fixed_prio_arb
```
若要实现参数化（这里优先级从低位往高位有限）：
```systemverilog
module prior_arb #(
	parameter REQ_WIDTH = 16
)(
	input  logic  [REQ_WIDTH-1:0] req,
	output logic  [REQ_WIDTH-1:0] grant
);

logic [REQ_WIDTH-1 : 0] pre_req;

always_comb begin
	pre_req[0] = req[0];
	grant[0] = req[0];
	for (int i = 1; i < REQ_WIDTH; i = i + 1) begin
		grant[i] = req[i] & !pre_req[i-1];  // current req & no higher priority request
		pre_req[i] = req[i] | pre_req[i-1]; // or all higher priority requests
	end
end
// req     00110
// pre_req 11110
// grant   00010

endmodule
```
有另一种简洁的实现方法：
本质上，我们要做的是找req这个信号里从低到高第一个出现的1，那么我们给req减去1会得到什么？假设req的第i位是1，第0到第i-1位都是0，那么减去1之后我们知道低位不够减，得要向高位借位，直到哪一位可以借到呢？就是第一次出现1的位，即从第i位借位，第0到i-1位都变成了1，而第i位变为了0，更高位不变。然后我们再给减1之后的结果取反，然后把结果再和req本身按位与，可以得出，只有第i位在取反之后又变成了1，而其余位都是和req本身相反的，按位与之后是0，这样就提取出来了第一个为1的那一位，也就是我们需要的grant。再考虑一下特殊情况req全0，很明显，按位与之后gnt依然都是全0，没有任何问题。
```verilog
module prior_arb #(
	parameter REQ_WIDTH = 16
) (
	input  [REQ_WIDTH-1:0]     req,
	output [REQ_WIDTH-1:0]     gnt
);

	assign gnt = req & (~(req-1));
endmodule
```
### 仿真
```verilog
`timescale 1ns / 1ps
module fixed_arb_tb();
reg  [3:0] request;
wire [3:0] grant;

fixed_arb u1(request,grant);

always #5 request = $random;

initial
begin
request = 4'b0000;
#200;
$stop;
end

endmodule
```

## 轮询仲裁器？？
循环优先级仲裁器（Round Robin arbiter）是一种【尽量均匀的将总线分配给不同主机】的策略。其基本思想如下：
1. 在初始情况下，最低位代表的主机有着最高的优先级，且向左优先级依次递减。以A为最高优先级，D为最低优先级，则可以表示为 DCBA 。
2. 若在一次仲裁中，某一位代表的主机获取了总线权限，则在下一次仲裁中，其左侧相邻位优先级变为最高，并向左优先级依次降低。

Round Robin就是考虑到公平性的一种仲裁算法。其基本思路是，当一个requestor 得到了grant许可之后，它的优先级在接下来的仲裁中就变成了最低，也就是说每个requestor的优先级不是固定的，而是会在最高（获得了grant)之后变为最低，并且根据其他requestor的许可情况进行相应的调整。这样当有多个requestor的时候，grant可以依次给每个requestor，即使之前高优先级的requestor再次有新的request，也会等前面的requestor都grant之后再轮到它。

我们以4个requestor为例来说明，下面这个表格Req[3:0]列表示实际的request，为1表示产生了request；RR Priority这一列为当前的优先级，为0表示优先级最高，为3表示优先级最低；RR Grant这一列表示根据当前Round Robin的优先级和request给出的许可；Fixed Grant表示如果是固定优先级，即按照3210，给出的grant值。

| cycle | req[3 : 0] | RR priority | RR Grant[3 : 0] | Fixed Grant[3 : 0] |
| ----- | ---------- | ----------- | --------------- | ------------------ |
| 1     | 0101       | 3210        | 0001            | 0001               |
| 2     | 0101       | 2103        | 0100            | 0001               |
| 3     | 0011       | 0321        | 0001            | 0001               |
| 4     | 0010       | 2103        | 0010            | 0010               |
| 5     | 1000       | 1032        | 1000            | 1000               |
第一个周期，初始状态，我们假设req[0]的优先级最高，req[1]其次，req[3]最低，当req[2]和req[0]同时为1的时候，根据优先级，req[0]优先级高于req[2]，grant = 0001。

第二个周期，因为req[2]在前一个周期并没有获得grant，那么它继续为1，而这个时候req[0]又来了一个新的request，这个时候就能够看出round robin和fixed priority的差别了。对于fixed priority， grant依然给0，即0001。但是round robin算法的要求是：因为上一个周期req[0]已经被grant了，那么它的优先级变为最低的3，相应的，req[1]的优先级变为最高，因为它本来就是第二高的优先级，那么当req[0]优先级变为最低了之后它自然递补到最高，那么这个时候产生的许可grant就不能给到req[0]，而是要给到req[2]。

同理，第三个周期，req[2]因为在前一个周期grant过，它的优先级变为3最低，req[3]的优先级变为最高。后面的周期大家可以自己顺着分析下来。

换句话说，因为被grant的那一路优先级在下一个周期变为最低，这样让其他路request都会依次被grant到，而不会出现其中某一路在其他路有request的情况下连续被grant的情况，所以round-robin在中文中也被翻译成“轮询调度”。

### 实现
首先看第一种思路，即优先级是变化的，回想一下我们之前讲的Fixed Priority Design，我们都假定了从LSB到MSB优先级是由高到低排列的。那么我们有没有办法先设计一个fixed priority arbiter，它的优先级是一个输入呢？看下面的RTL

```verilog
module arbiter_base #(parameter NUM_REQ = 4)
(
   input [NUM_REQ-1:0]    req,
   input [NUM_REQ-1:0]    base,
   output [NUM_REQ-1:0]    gnt
);

wire[2*NUM_REQ-1:0] double_req = {req,req};

wire[2*NUM_REQ-1:0] double_gnt = double_req & ~(double_req - base);

assign gnt = double_gnt[NUM_REQ-1:0] | double_gnt[2*NUM_REQ-1:NUM_REQ];
endmodule
```

在这个模块中，base是一个onehot的信号，它为1的那一位表示这一位的优先级最高，然后其次是它的高位即左边的位，直到最高位后回到第0位绕回来，优先级依次降低，直到为1那一位右边的这位为最低。咱们以4位为例，如果base = 4'b0100, 那么优先级是bit[2] > bit[3] > bit[0] > bit[1]。

double\_req & ~(double\_req-base)其实就是利用减法的借位去找出base以上第一个为1的那一位，只不过由于base值可能比req值要大，不够减，所以要扩展为{req, req}来去减。当base=4‘b0001的时候就是咱们上一篇里面的最后的算法。当然base=4'b0001的时候不存在req不够减的问题，所以不用扩展。然后gnt取高位扩展和低位的并即可。

每次grant之后，我把我的优先级调整一下就可以了呗。而且这个设计妙就妙在，base要求是一个onehot signal，而且为1的那一位优先级最高。我们前面说过，grant一定是onehot，grant之后被grant的那一路优先级变为最低，它的高1位优先级变为最高，所以，我只需要一个history_reg，来去记录之前最后grant的值，然后只需要将grant的值左移一下就变成了下一个周期的base。比如说，假设我上一个周期grant为4'b0010，那么bit[2]要变为最高优先级，那只需要base是grant的左移即可。RTL代码如下
```verilog
module round_robin_arbiter #(parameter NUM_REQ = 4)
(
  input                      clk,
  input                      rstn,
  input [NUM_REQ-1:0]        req,
  output [NUM_REQ-1:0]       gnt 
);

logic [NUM_REQ-1:0]          hist_q, hist_d;

always_ff@(posedge clk) begin
  if(!rstn) 
    hist_q <= {{NUM_REQ-1{1'b0}}, 1'b1};
  else
    if(|req)
      hist_q <= {gnt[NUM_REQ-2:0, gnt[NUM_REQ-1]}; 
end

arbiter_base #(
  .NUM_REQ(NUM_REQ)
) arbiter(
  .req      (req),
  .gnt      (gnt),
  .base     (hist_q)
);

endmodule
```
和Fixed Priority Arbiter不同，Round robin arbiter不再是纯的组合逻辑电路，而是要有时钟和复位信号，因为里面必须要有个寄存器来记录之前grant的状态。

但是这个设计也有缺点，即在面积和timing上的优化不够好。相比于我们接下来要介绍的设计，在request位数大(比如64位）的时候timing和area都要差一些。

前面的思路是换优先级，而request不变，另一个思路是优先级不变，但是我们从request入手：当某一路request已经grant之后，我们人为地把进入fixed priority arbiter的这一路req给屏蔽掉，这样相当于只允许之前没有grant的那些路去参与仲裁，grant一路之后就屏蔽一路，等到剩余的request都依次处理完了再把屏蔽放开，重新来过。这就是利用屏蔽mask的办法来实现round robin的思路。



//=============
针对于轮询仲裁器而言，我们需要引入时钟信号了，即时序逻辑电路的部分，这是因为，此时的优先级排序，不仅与初态有关，也与前一个状态的响应有关。

其次，对于举例中，我们对于相应情况四位独热码的grant的分析，我们可以发现它和下一个周期优先级排序之间的关系：即若abcd为grant，c为1，即grant为0010，下次的优先级为1032，若b为1，即优先级为0321。因此固定优先级相当于优先级总为ABCD，前一个状态总为1000。
因此可以和减1取反类似实现。
```verilog
module round_robin_arb(
	input wire clk, rst_n,
	input wire [3 : 0] request,
	output logic [3 : 0] grant
	);
logic [3:0] pre_state;
logic [3:0] pre_grant;

always_ff @(posedge clk or negedge rst_n) begin
	if(!rst_n)
		pre_state <= 4'h1;
	else
		pre_state <= {pre_grant[2],pre_grant[1],pre_grant[0],pre_grant[3]};
end

assign pre_grant = {1'b1,request} & ~({1'b1,request} - 1'b1);

assign grant = {1'b1,request} & ~({1'b1,request} - pre_state);

endmodule
```
仿真：
```verilog
`timescale 1ns / 1ps
module round_robin_arb_tb();
reg clk;
reg rst_n;
reg [3:0] request;
wire [3:0] grant;

round_robin_arb u1(clk,rst_n,request,grant);

initial clk = 0;
always #5 clk = !clk;

initial begin
rst_n= 0;
request = 4'h0;
#19 rst_n = 1;
request = 4'b1101;
#10
request = 4'b0101;
#10
request = 4'b0010;
#10
request = 4'b0000;
#100;
$stop;

end

endmodule
```


# 双向握手



# booth乘法器

https://www.bilibili.com/video/BV1U44y1o7sT?spm_id_from=333.788.videopod.sections&vd_source=bc07d988d4ccb4ab77470cec6bb87b69


# RAM
## 单口 RAM 与伪双口 RAM、真双口 RAM 的区别在于：

　　+ 单口 RAM 只有一个时钟（clka）（时钟上升沿到来时对数据进行写入或者读出）、一组输入输出数据线（dina & douta）、一组地址线（addra）、一个使能端（ena）（“ena == 1”时可进行读或写的操作，“ena == 0”时无法进行读或写的操作）、一个写使能端（wea）（在“ena == 1”的情况下：“wea == 1”时只写不读，“wea == 0”时只读不写）。单口读、写无法同时进行，只能或读或写。

　　+ 伪双口 RAM 有两个时钟（clka & clkb）、一组输入输出数据线（dina & doutb）、两组地址线（addra & addrb），两个使能端（ena & enb）、一个写使能端（wea）。一个端口只读（Port a），另一个端口只写（Port b）。整体上，读、写可以同时进行。

　　+ 真双口 RAM 有两个时钟（clka & clkb）、两组输入输出数据线（dina & douta & dinb & doutb）、两组地址线（addra & addrb），两个使能端（ena & enb）、两个写使能端（wea & web）。两个端口都可以进行读写操作（Port a 和 Port b 可以一起读或者一起写或者一个读一个写）。整体上，读、写可以同时进行。

## 单口 ROM 与双口 ROM 的区别在于：

　　+ 单口 ROM 只有一个时钟（clka）、一组输出数据线（douta）、一组地址线（addra）、一个使能端（ena）。只能进行读操作，且一个时钟只能读出某个地址上的一组数据。

　　+ 双口 ROM 有两个时钟（clka & clkb）、两组输出数据线（douta & doutb）、两组地址线（addra & addrb）、两个使能端（ena & enb）。也是只能进行读操作，且每个端口中，一个时钟只能读出某个地址上的一组数据。其实和单口 ROM 没什么区别，基本上可以当成是两个单口 ROM 拼接而成的罢了，只是存储的数据是共享的。



# 归并排序
https://www.zhihu.com/people/han-yao-dong-74/posts

- **“归并排序**（Merge Sort）是建立在**归并**操作上的一种有效，稳定的**排序**算法，该算法是采用分治法（Divide and Conquer）的一个非常典型的应用。 将已有序的子序列合并，得到完全有序的序列；即先使每个子序列有序，再使子序列段间有序。 若将两个有序表合并成一个有序表，称为二路**归并**。”----百度百科
- 排序算法是非常常见且种类繁多的算法，但是大多数算法设计之初是以软件实现为基础进行设计的，本文主要讲解一种硬件友好型排序算法，也就是十大排序算法之一的归并排序，从资源和时序的角度来讲解归并排序的具体实现。本文以二路归并为例，即输入一段连续数据，将其分为2份进行1归并，获得2份有序数据后进行2归并，再进行4归并......直到最后获得TopK排序结果。

# Karatsuba大数乘法
https://zhuanlan.zhihu.com/p/442689186

- 进行乘法运算$a \times b$时，对于FPGA/ASIC硬件而言，有2种实现方法，1是用DSP进行，2是用LUT查找表的方式进行查找（只适合小位宽乘法）。但是，当乘法位宽较大时，单纯用DSP会消耗很多资源，并且时序不好收敛。所以，本文介绍一种针对大数据位宽乘法计算的方法。

以64-bit计算为例，输入A,B位宽为64-bit，可以分割为2个32-bit数，即
$$
\begin{align}
A&=\{AH,AL\}，B=\{BH,BL\};\\
A*B &= (2**32 * AH + AL) * (2**32 * BH + BL) \\
&=(2**64 * AH * BH) + (2**32 (AH*BL + AL*BH)) + (AL*BL) ;
\end{align}
$$
上式需要4个乘法器和三个加法器，下一步思考如何优化？
$AH*BH$和$AL*BL$高低位对应相乘是不可或缺的，所以需要思考如何优化掉$AH*BL+AL*BH$。
$$
\begin{align}
AH*BL+AL*BH&=(AH+AL)*(BH+BL)-AH*BH-AL*BL\\
pphh &= AH*BH(64bit)\\
ppll &= AL*BL(64bit)\\
pphl &= (AH+AL)*(BH+BL) (66 bit)\\
product = {pphh,ppll} + (pphl - pphh - ppll) << 32
\end{align}
$$

值得注意的是，乘法结果一般需要延迟1拍以上才能拿到，那么pphl就一定会和pphh,ppll不同步；另外，注意到最后加减法都是64bit以上的大位宽，所以不应该用简单的组合电路计算输出，这样在布局布线阶段会比较困难，影响最终时序优化。




# 大位宽加法
https://zhuanlan.zhihu.com/p/441764982

### 问题
如何实现数据位宽较大的加法？
### 分析
- 如果是小位宽加法，比如8bit加法，那么直接c=a+b即可，如果再小的位宽比如3bit加法，那么用查找表LUT的方式来计算会比用加法器进位链更省资源，timing更好。
- 所以，之所以需要单独设计大位宽加法器，原因在于位宽较大时直接c=a+b，会导致timing较差，布局布线较困难。
- 进一步的，如果可以在加法器中间加1级寄存器，可以更好地优化时序，实现更高的时钟频率。
### 设计
1. 将原始数据位宽拆分为高低2部分，低位部分直接相加即可：ls_adder <= {1'b0,a[LS_WIDTH-1:0]} + {1'b0,b[LS_WIDTH-1:0]}，加完寄存1拍，通过pipeline的方式计算，不插入bubble的同时可以获得更好的timing;
2. 进位比特即为ls_adder最高比特位：cross_carry = ls_adder[LS_WIDTH]；
3. 将高位部分寄存后相加，相加过程中低位加法的进位bit放在最后，即：ms_adder ={ms_data_a,cross_carry} + {ms_data_b,cross_carry};

# 握手和反压
> 握手参考FPGA应用开发

流水线中的握手与反压 - 不坠青云之志的文章 - 知乎
https://zhuanlan.zhihu.com/p/647594414

数字芯片设计——握手与反压 - 0431大小回的文章 - 知乎
https://zhuanlan.zhihu.com/p/359330607

当入口流量大于出口流量，这时候就需要**反压**，或者，当后级未准备好时，如果本级进行数据传递，那么它就需要反压前级，所以此时前级需要将数据保持不动，直到握手成功才能更新数据。而反压在多级流水线中就变得稍显复杂，原因在于，比如我们采用三级流水设计，如果我们收到后级反压信号，我们理所当然想反压本级输出信号的寄存器，但是如果只反压最后一级寄存器，那么会面临一个问题，就是最后一级寄存器数据会被前两级流水冲毁，导致数据丢失，引出数据安全问题，所以我们此时需要考虑反压设计。

常用的反压方法有三种：
## 不带存储体的反压
也就是后级反压信号*对本级模块中所有流水寄存器都进行控制*，由于不包含存储体，为了保证数据安全性，后级反压信号可以同时反压本模块中所有流水寄存器。
- 优点：节省面积资源
- 缺点：寄存器端口控制复杂
- 适用情况：流水线深度较大时

## 带存储体的逐级反压
如果流水级数不深，可以*在每一需要握手交互模块增加存储体*，原理上相当于，如果后级发出反压信号，可以直接对本级流水线数据源头进行反压，其余中间级不需控制，但后级需要包含RAM或FIFO等存储体，可以接收流水，并需设置水线（water line)，确定反压时间，防止数据溢出，保证数据安全性。

优点：各级流水寄存器端口控制简单
缺点：需要额外存储体
适用情况：流水线深度较小，每一模块都包含存储体时

## 带存储体的跨级反压
很多时候在具体设计过程中，颗粒度划分不精细，反压这时候是对模块而言，而不是说模块内部有多少级流水。此外，并不是每一模块都带有存储体。比如，其中可能 a 模块没有存储体，b 模块没有存储体，但 a b 模块内部还有多级流水，如果 c 模块有存储体，并且需要反压前级模块，这时候可以选择反压 a 模块的源头输入数据，然后将 a b 的流水都存储到带有存储体的 c 模块，但是如果 a b 不是都没有存储体的话，就不应该跨级反压，而应该逐级反压，具体原因后续会讲。

优点：控制简单方便
缺点：需要额外存储体，模块间耦合度高
适用情况：某些模块带存储体，某些模块不带存储体时

## 带存储体的反压

如上文所述，很多时候我们不喜欢对每一级细分流水都进行反压，所以可以选择带存储体的反压，也就是增加RAM或者FIFO，在反压上级模块的同时，本级有足够的深度来存储上一级的流水数据。

### 举个例子——字节的问题

问题：设计一个并行6输入32比特加法器，输出1个带截断的32比特加法结果，要求用三级流水设计，带前后反压。

主要输入：
1. 6个32bit数据
2. 上一级的 valid_i
3. 下一级的 ready_i

输出：
1. 1个32bit结果
2. 给上一级的 ready_o
3. 给下一级的 valid_o

- 分析
其实在多级流水设计中，如果每一级只有一个寄存器，并且都在一个模块中，也就是说当颗粒度划分的很细的时候，一般使用带存储体的反压，比如六级流水，那么就设计好水线，在FIFO未满时提前发出反压信号，一般水线设为FIFO_DEPTH - 流水级数，下面是我设计的代码。

核心思想就是如果FIFO未达到水线（WATERLINE）时，给上一级的反压信号ready_o就持续拉高，否则拉低；FIFO非空时就可以给下一级valid_o拉高，然后下一级的反压信号ready_i可以作为FIFO的读使能信号，具体请参考下文代码。


# 状态机

状态机有没有考虑出错的情况



[CRC、LFSR电路](https://note.youdao.com/ynoteshare1/index.html?id=a99dd6686501a06c8ed39cd2c40f9aed&type=note)

[环形、扭环、LFSR计数器](https://blog.csdn.net/Reborn_Lee/article/details/102172111?utm_source=app)

[线性反馈移位寄存器(Linear Feedback Shift Register, LFSR)](https://blog.csdn.net/qq_44113393/article/details/89852994)

[循环冗余校验(CRC)算法入门](https://www.cnblogs.com/sinferwu/p/7904279.html)

[CRC算法的硬件电路实现：串行电路和并行电路](https://zhuanlan.zhihu.com/p/59666086)

[使用Verilog实现CRC-8的串行计算](https://blog.csdn.net/zhangningning1996/article/details/106795689)

[数字IC笔试题_CRC并行计算](https://zhuanlan.zhihu.com/p/69969288)

[数字IC笔试——乐鑫提前批笔试编程题源码](https://blog.csdn.net/qq_41844618/article/details/106822610)

两个在线生成并行 CRC Verilog 代码工具：[CRC在线工具](https://www.easics.com/webtools/crctool) [CRC generator](http://outputlogic.com/?page_id=321)

包含各种基本单元的设计： https://www.zhihu.com/people/zhishangtanxin/posts?page=2
