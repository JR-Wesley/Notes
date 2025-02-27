# ch1 与UVM接触

## 1.1 UVM
![[IC设计流程.bmp]]


## 1.2 UVM作用

·如何用UVM搭建验证平台，包括如何使用sequence机制、factory机制、callback机制、寄存器模型（register model）等。
·一些验证的基本常识，将会散落在各个章节之间。
·UVM的一些高级功能，如何灵活地使用sequence机制、factory机制等。
·如何编写代码才能保证可重用性。可重用性是目前IC界提及最多的几个词汇之一，它包含很多层次。对于个人来说，如何
保证自己在这个项目写的代码在下一个项目中依然可以使用，如何保证自己写出来的东西别人能够重用，如何保证子系统级的代
码在系统级别依然可以使用；对于同一公司来说，如何保证下一代的产品在验证过程中能最大程度使用前一代产品的代码。
·同样的一件事情有多种实现方式，这多种方式之间分别都有哪些优点和缺点，在权衡利弊之下哪种是最合理的。
·一些OVM用法的遗留问题。

# ch2 一个简单的UVM验证平台

## 2.1 验证平台构成

- 验证平台要模拟DUT的各种真实使用情况，这意味着要给DUT施加各种激励，有正常的激励，也有异常的激励；有这种模式的激励，也有那种模式的激励。激励的功能是由driver来实现的。
- 验证平台要能够根据DUT的输出来判断DUT的行为是否与预期相符合，完成这个功能的是记分板（scoreboard，也被称为checker，本书统一以scoreboard来称呼）。既然是判断，那么牵扯到两个方面：一是判断什么，需要把什么拿来判断，这里很明显是DUT的输出；二是判断的标准是什么。
- 验证平台要收集DUT的输出并把它们传递给scoreboard，完成这个功能的是monitor。
- 验证平台要能够给出预期结果。在记分板中提到了判断的标准，判断的标准通常就是预期。假设DUT是一个加法器，那么当在它的加数和被加数中分别输入1，即输入1+1时，期望DUT输出2。当DUT在计算1+1的结果时，验证平台也必须相应完成同样的过程，也计算一次1+1。在验证平台中，完成这个过程的是参考模型（reference model）。

![[2.1验证平台.bmp]]

UVM中引入了agent和sequence的概念
![[2-2典型.bmp]]

## 2.2 只有Driver的验证平台

1. 最简单的验证平台
driver是验证平台最基本的组件，是整个验证平台数据流的源泉。本节以一个简单的DUT为例，说明一个只有driver的UVM验证平台是如何搭建的。

基于`ch2 dut.sv`，功能是通过rxd接收再通过txd发送，rx_dv为接收数据有效指示，tx_en是发送数据有效指示。

UVM是一个库，这个库中，几乎所有的东西都是使用class来实现的。`driver, monitor, reference model, scoreboard`等组成部分都是类。class有function, task，通过这些函数和任务可以完成driver的输出激励功能、完成monitor的监测功能、完成参考模型的计算功能、完成scoreborad的比较功能。class中可以有成员变量，这些成员变量可以控制类的行为，如控制monitor的行为等。当要实现一个功能时先应该从UVM的某个类派生出一个新的类，在这个类中实现所希望的功能。使用UVM的第一条原则是：验证平台中的所有组件都应该派生自UVM中的类。

```verilog
`ifndef MY_DRIVER__SV
`define MY_DRIVER__SV
class my_driver extends uvm_driver;

   function new(string name = "my_driver", uvm_component parent = null);
      super.new(name, parent);
   endfunction
   extern virtual task main_phase(uvm_phase phase);
endclass

task my_driver::main_phase(uvm_phase phase);
   top_tb.rxd <= 8'b0; 
   top_tb.rx_dv <= 1'b0;
   while(!top_tb.rst_n)
      @(posedge top_tb.clk);
   for(int i = 0; i < 256; i++)begin
      @(posedge top_tb.clk);
      top_tb.rxd <= $urandom_range(0, 255);
      top_tb.rx_dv <= 1'b1;
      `uvm_info("my_driver", "data is drived", UVM_LOW)
   end
   @(posedge top_tb.clk);
   top_tb.rx_dv <= 1'b0;
endtask
`endif
```

这个driver的功能，向rxd上发送256个随机数据，并将rx_dv信号置为高电平。当数据发送完毕后，将rx_dv信号置为低电平。

- 所有派生自uvm_driver的类的new函数有两个参数，一个是string类型的name，一个是uvm_component类型的parent。关于name参数，比较好理解，就是名字而已；至于parent则比较难以理解，读者可暂且放在一边，下文会有介绍。事实上，这两个参数是由uvm_component要求的，每一个派生自uvm_component或其派生类的类在其new函数中要指明两个参数：name和parent，这是uvm_component类的一大特征。而uvm_driver是一个派生自uvm_component的类，所以也会有这两个参数。

- driver所做的事情几乎都在main_phase中完成。UVM由phase来管理验证平台的运行，这些phase统一以xxxx_phase来命名，且都有一个类型为uvm_phase、名字为phase的参数。main_phase是uvm_driver中预先定义好的一个任务。因此几乎可以简单地认为，实现一个driver等于实现其main_phase。

上述代码出现了`uvm_info`宏，这个宏功能与Verilog中的display语句类似，但是比display更强大。三个参数分别是：字符串，用于把打印信息归类；字符串，是具体需要打印的信息；冗余级别，在验证平台中，某些信息非常关键，可以设置为UVM_LOW，有些可有可无，设置为UVM_HIGH，介于之间的就是UVM_MEDIUM。UVM默认只显示UVM_MEDIUM和UVM_LOW的信息。本例打印结果如下：

>UVM_INFO my_driver.sv
>
>(20
>
>) @48500000
>
>:drv[my_driver] data is drived

打印的结果有几项：

- UVM_INFO关键字：表明这是一个uvm_info宏打印的结果。除了uvm_info宏外，还有uvm_error宏、uvm_warning宏，后文中将会介绍。
- my_driver.sv（20）：指明此条打印信息的来源，其中括号里的数字表示原始的uvm_info打印语句在my_driver.sv中的行号。
- 48500000：表明此条信息的打印时间。
- drv：这是driver在UVM树中的路径索引。UVM采用树形结构，对于树中任何一个结点，都有一个与其相应的字符串类型的路径索引。路径索引可以通过get_full_name函数来获取，把下列代码加入任何UVM树的结点中就可以得知当前结点的路径索引：

```verilog
$display("the full name of current component is: %s", get_full_name());
```

- [my_driver]：方括号中显示的信息即调用uvm_info宏时传递的第一个参数。
- data is drived：表明宏最终打印的信息。

可见，UVM_INFO宏非常强大，它包含了打印信息的物理文件来源、逻辑节点信息（在UVM树中的路径索引）、打印时间、对信息的分类组织及打印的信息。读者在搭建验证平台时应该尽量使用uvm_info宏取代display语句。定义my_driver后需要将其实例化。

第2行把uvm_macros.svh文件通过include语句包含进来。这是UVM中的一个文件，里面包含了众多的宏定义，只需要包含一次。

第4行通过import语句将整个uvm_pkg导入验证平台中。只有导入了这个库，编译器在编译my_driver.sv文件时才会认识其中的uvm_driver等类名。

第24和25行定义一个my_driver的实例并将其实例化。注意这里调用new函数时，其传入的名字参数为drv，前文介绍uvm_info宏的打印信息时出现的代表路径索引的drv就是在这里传入的参数drv。另外传入的parent参数为null，在真正的验证平台中，这个参数一般不是null，这里暂且使用null。

第26行显式地调用my_driver的main_phase。在main_phase的声明中，有一个uvm_phase类型的参数phase，在真正的验证平台中，这个参数是不需要用户理会的。本节的验证平台还算不上一个完整的UVM验证平台，所以暂且传入null。

第27行调用finish函数结束整个仿真，这是一个Verilog中提供的函数。

运行这个例子，可以看到“data is drived”被输出了256次。

2. 加入factory机制







3. 加入







## 2.3 为验证平台加入各个组件

1. transaction



2. env



3. monitor



4. agent





5. reference model







6. scoreboard



7. field_automation





## 2.4 UVM终极：sequence





## 2.5 建造测试用例



# ch3 UVM基础






# 附录A SystemVerilog

## 封装继承
当一个变量被设置为local类型后,那么这个变量就会具有两大特点：
- 此变量只能在类的内部由类的函数/任务进行访问。 
- 在类外部使用直接引用的方式进行访问会提示出错。
函数/任务也可以被定义为local类型。这种情况通常用于某些底层函数
如果父类中某成员变量是local类型，那么子类不可以使用这些变量。变量声明为protected，可以被子类访问而不会被外部访问。
## 多态
父类向子类的类型转换，这种类型转换必须通过cast来完成，子类转为父类可以自动完成。
同样调用一个函数，但是输出结果不同，表现为多态，依赖于虚函数实现。

## A6 randomize constraint

SystemVerilog为所有类定义了randomize方法；

```verilog
class animal;
    bit [10:0] kind;
    rand bit[5:0] data;
    rand int addr;
endclass
initial begin
    animal aml;
    aml = new();
    assert(aml.randomize());
end
```

在一个类中只有定义为rand类型的字段才会在调用randomize方法时进行随机化。上面的定义中，data和addr会随机化为一个随机值，而kind在randomize被调用后，依然是默认值0。

与randomize对应的是constraint。在不加任何约束的情况下，上述animal中的data经过随机化后，其值为0～'h3F中的任一值。可以定义一个constraint对其值进行约束：
```verilog
class animal;
    rand bit[5:0] data;
    constraint data_cons{
        data > 10;
        data < 30;
    }
endclass
```
经过上述约束，data在随机后，其值将介于10-30之间。
除了在类的定义时对数据进行约束外，还可以在调用randomize时对数据进行约束：

```verilog
initial begin
    animal aml;
    aml = new();
    assert(aml.randomize() with {data > 10; data < 30;});
end
```

