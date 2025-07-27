---
tags: define, pragma
dateCreated: 2021-10-31
dateModified: 2025-07-27
---
# Gdb

## 1. 概述

 GDB 全称“GNU symbolic debugger”，从名称上不难看出，它诞生于 GNU 计划（同时诞生的还有 GCC、Emacs 等），是 Linux 下常用的程序调试器。发展至今，GDB 已经迭代了诸多个版本，当下的 GDB 支持调试多种编程语言编写的程序，包括 C、C++、Go、Objective-C、OpenCL、Ada 等。实际场景中，GDB 更常用来调试 C 和 C++ 程序。一般来说，GDB 主要帮助我们完成以下四个方面的功能：

1. 启动你的程序，可以按照你的自定义的要求随心所欲的运行程序。
2. 在某个指定的地方或条件下暂停程序。
3. 当程序被停住时，可以检查此时你的程序中所发生的事。
4. 在程序执行过程中修改程序中的变量或条件，将一个 bug 产生的影响修正从而测试其他 bug。

使用 GDB 调试程序，有以下两点需要注意：

1. 要使用 GDB 调试某个程序，该程序编译时必须加上编译选项 **`-g`**，否则该程序是不包含调试信息的；
2. GCC 编译器支持 **`-O`** 和 **`-g`** 一起参与编译。GCC 编译过程对进行优化的程度可分为 5 个等级，分别为：

- **-O/-O0**：不做任何优化，这是默认的编译选项；
- **-O1**：使用能减少目标文件大小以及执行时间并且不会使编译时间明显增加的优化。该模式在编译大型程序的时候会花费更多的时间和内存。在 -O1 下：编译会尝试减少代 码体积和代码运行时间，但是并不执行会花费大量时间的优化操作。
- **-O2**：包含 -O1 的优化并增加了不需要在目标文件大小和执行速度上进行折衷的优化。GCC 执行几乎所有支持的操作但不包括空间和速度之间权衡的优化，编译器不执行循环 展开以及函数内联。这是推荐的优化等级，除非你有特殊的需求。-O2 会比 -O1 启用多 一些标记。与 -O1 比较该优化 -O2 将会花费更多的编译时间当然也会生成性能更好的代 码。
- **-O3**：打开所有 -O2 的优化选项并且增加 -finline-functions, -funswitch-loops,-fpredictive-commoning, -fgcse-after-reload and -ftree-vectorize 优化选项。这是最高最危险 的优化等级。用这个选项会延长编译代码的时间，并且在使用 gcc4.x 的系统里不应全局 启用。自从 3.x 版本以来 gcc 的行为已经有了极大地改变。在 3.x，，-O3 生成的代码也只 是比 -O2 快一点点而已，而 gcc4.x 中还未必更快。用 -O3 来编译所有的 软件包将产生更 大体积更耗内存的二进制文件，大大增加编译失败的机会或不可预知的程序行为（包括 错误）。这样做将得不偿失，记住过犹不及。在 gcc 4.x.中使用 -O3 是不推荐的。
- **-Os**：专门优化目标文件大小 ,执行所有的不增加目标文件大小的 -O2 优化选项。同时 -Os 还会执行更加优化程序空间的选项。这对于磁盘空间极其紧张或者 CPU 缓存较小的 机器非常有用。但也可能产生些许问题，因此软件树中的大部分 ebuild 都过滤掉这个等 级的优化。使用 -Os 是不推荐的。

## 2. 启用 GDB 调试

GDB 调试主要有三种方式：

1. 直接调试目标程序：gdb ./hello_server
2. 附加进程 id：gdb attach pid
3. 调试 core 文件：gdb filename corename

## 3. 退出 GDB

- 可以用命令：**q（quit 的缩写）或者 Ctr + d** 退出 GDB。
- 如果 GDB attach 某个进程，退出 GDB 之前要用命令 **detach** 解除附加进程。

## 4. 常用命令

| 命令名称    | 命令缩写  | 命令说明                                         |
| ----------- | --------- | ------------------------------------------------ |
| run         | r         | 运行一个待调试的程序                             |
| continue    | c         | 让暂停的程序继续运行                             |
| next        | n         | 运行到下一行                                     |
| step        | s         | 单步执行，遇到函数会进入                         |
| until       | u         | 运行到指定行停下来                               |
| finish      | fi        | 结束当前调用函数，回到上一层调用函数处           |
| return      | return    | 结束当前调用函数并返回指定值，到上一层函数调用处 |
| jump        | j         | 将当前程序执行流跳转到指定行或地址               |
| print       | p         | 打印变量或寄存器值                               |
| backtrace   | bt        | 查看当前线程的调用堆栈                           |
| frame       | f         | 切换到当前调用线程的指定堆栈                     |
| thread      | thread    | 切换到指定线程                                   |
| break       | b         | 添加断点                                         |
| tbreak      | tb        | 添加临时断点                                     |
| delete      | d         | 删除断点                                         |
| enable      | enable    | 启用某个断点                                     |
| disable     | disable   | 禁用某个断点                                     |
| watch       | watch     | 监视某一个变量或内存地址的值是否发生变化         |
| list        | l         | 显示源码                                         |
| info        | i         | 查看断点 / 线程等信息                            |
| ptype       | ptype     | 查看变量类型                                     |
| disassemble | dis       | 查看汇编代码                                     |
| set args    | set args  | 设置程序启动命令行参数                           |
| show args   | show args | 查看设置的命令行参数                             |

gprof 是一款 GNU profile 工具，可以运行于 linux、AIX、Sun 等操作系统进行 C、C++、Pascal、Fortran 程序的性能分析，用于程序的性能优化以及程序瓶颈问题的查找和解决。

# Gprof 介绍

gprof(GNU profiler) 是 GNU binutils 工具集中的一个工具，linux 系统当中会自带这个工具。它可以分析程序的性能，能给出函数调用时间、调用次数和调用关系，找出程序的瓶颈所在。在编译和链接选项中都加入 -pg 之后，gcc 会在每个函数中插入代码片段，用于记录函数间的调用关系和调用次数，并采集函数的调用时间。

## Gprof 安装

gprof 是 gcc 自带的工具，无需额外安装步骤。

## Gprof 使用步骤

**1. 用 gcc、g++、xlC 编译程序时，使用 -pg 参数**

​ 如：`g++ -pg -o test.exe test.cpp`

​ 编译器会自动在目标代码中插入用于性能测试的代码片断，这些代码在程序运行时采集并记录函数的调用关系和调用次数，并记录函数自身执行时间和被调用函数的执行时间。

**2. 执行编译后的可执行程序，生成文件 gmon.out**

如：`./test.exe`

​ 该步骤运行程序的时间会稍慢于正常编译的可执行程序的运行时间。程序运行结束后，会在程序所在路径下生成一个缺省文件名为 gmon.out 的文件，这个文件就是记录程序运行的性能、调用关系、调用次数等信息的数据文件。

**3. 使用 gprof 命令来分析记录程序运行信息的 gmon.out 文件**

如：`gprof test.exe gmon.out`

​ 可以在显示器上看到函数调用相关的统计、分析信息。上述信息也可以采用 `gprof test.exe gmon.out> gprofresult.txt` 重定向到文本文件以便于后续分析。

## 实战一：用 Gprof 测试基本函数调用及控制流

### 测试代码

```c
#include <stdio.h>

void loop(int n){
    int m = 0;
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            m++;    
        }   
    }
}

void fun2(){
    return;
}

void fun1(){
    fun2();
}

int main(){
    loop(10000);

    //fun1callfun2
    fun1(); 

    return 0; 
}
```

### 操作步骤

```text
liboxuan@ubuntu:~/Desktop$ vim test.c
liboxuan@ubuntu:~/Desktop$ gcc -pg -o test_gprof test.c 
liboxuan@ubuntu:~/Desktop$ ./test_gprof 
liboxuan@ubuntu:~/Desktop$ gprof ./test_gprof gmon.out
# 报告逻辑是数据表 + 表项解释
Flat profile:

# 1.第一张表是各个函数的执行和性能报告。
Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total           
 time   seconds   seconds    calls  ms/call  ms/call  name    
101.20      0.12     0.12        1   121.45   121.45  loop
  0.00      0.12     0.00        1     0.00     0.00  fun1
  0.00      0.12     0.00        1     0.00     0.00  fun2

 %         the percentage of the total running time of the
time       program used by this function.

cumulative a running sum of the number of seconds accounted
 seconds   for by this function and those listed above it.

 self      the number of seconds accounted for by this
seconds    function alone.  This is the major sort for this
           listing.

calls      the number of times this function was invoked, if
           this function is profiled, else blank.

 self      the average number of milliseconds spent in this
ms/call    function per call, if this function is profiled,
       else blank.

 total     the average number of milliseconds spent in this
ms/call    function and its descendents per call, if this
       function is profiled, else blank.

name       the name of the function.  This is the minor sort
           for this listing. The index shows the location of
       the function in the gprof listing. If the index is
       in parenthesis it shows where it would appear in
       the gprof listing if it were to be printed.


Copyright (C) 2012-2015 Free Software Foundation, Inc.

Copying and distribution of this file, with or without modification,
are permitted in any medium without royalty provided the copyright
notice and this notice are preserved.


# 2.第二张表是程序运行时的
             Call graph (explanation follows)


granularity: each sample hit covers 2 byte(s) for 8.23% of 0.12 seconds

index % time    self  children    called     name
                0.12    0.00       1/1           main [2]
[1]    100.0    0.12    0.00       1         loop [1]
-----------------------------------------------
                                                 <spontaneous>
[2]    100.0    0.00    0.12                 main [2]
                0.12    0.00       1/1           loop [1]
                0.00    0.00       1/1           fun1 [3]
-----------------------------------------------
                0.00    0.00       1/1           main [2]
[3]      0.0    0.00    0.00       1         fun1 [3]
                0.00    0.00       1/1           fun2 [4]
-----------------------------------------------
                0.00    0.00       1/1           fun1 [3]
[4]      0.0    0.00    0.00       1         fun2 [4]
-----------------------------------------------

 This table describes the call tree of the program, and was sorted by
 the total amount of time spent in each function and its children.

 Each entry in this table consists of several lines.  The line with the
 index number at the left hand margin lists the current function.
 The lines above it list the functions that called this function,
 and the lines below it list the functions this one called.
 This line lists:
     index  A unique number given to each element of the table.
        Index numbers are sorted numerically.
        The index number is printed next to every function name so
        it is easier to look up where the function is in the table.

     % time This is the percentage of the `total' time that was spent
        in this function and its children.  Note that due to
        different viewpoints, functions excluded by options, etc,
        these numbers will NOT add up to 100%.

     self   This is the total amount of time spent in this function.

     children   This is the total amount of time propagated into this
        function by its children.

     called This is the number of times the function was called.
        If the function called itself recursively, the number
        only includes non-recursive calls, and is followed by
        a `+' and the number of recursive calls.

     name   The name of the current function.  The index number is
        printed after it.  If the function is a member of a
        cycle, the cycle number is printed between the
        function's name and the index number.


 For the function's parents, the fields have the following meanings:

     self   This is the amount of time that was propagated directly
        from the function into this parent.

     children   This is the amount of time that was propagated from
        the function's children into this parent.

     called This is the number of times this parent called the
        function `/' the total number of times the function
        was called.  Recursive calls to the function are not
        included in the number after the `/'.

     name   This is the name of the parent.  The parent's index
        number is printed after it.  If the parent is a
        member of a cycle, the cycle number is printed between
        the name and the index number.

 If the parents of the function cannot be determined, the word
 `<spontaneous>' is printed in the `name' field, and all the other
 fields are blank.

 For the function's children, the fields have the following meanings:

     self   This is the amount of time that was propagated directly
        from the child into the function.

     children   This is the amount of time that was propagated from the
        child's children to the function.

     called This is the number of times the function called
        this child `/' the total number of times the child
        was called.  Recursive calls by the child are not
        listed in the number after the `/'.

     name   This is the name of the child.  The child's index
        number is printed after it.  If the child is a
        member of a cycle, the cycle number is printed
        between the name and the index number.

 If there are any cycles (circles) in the call graph, there is an
 entry for the cycle-as-a-whole.  This entry shows who called the
 cycle (as parents) and the members of the cycle (as children.)
 The `+' recursive calls entry shows the number of function calls that
 were internal to the cycle, and the calls entry for each member shows,
 for that member, how many times it was called from other members of
 the cycle.


Copyright (C) 2012-2015 Free Software Foundation, Inc.

Copying and distribution of this file, with or without modification,
are permitted in any medium without royalty provided the copyright
notice and this notice are preserved.
# 第三张表是函数与其在报告中序号的对应表

Index by function name

   [3] fun1                    [4] fun2                    [1] loop
```

# Volatile

## 1. Why

volatile 与 const 对应，用于建立语言级别的 memory barrier

volatile: A volatile specifier is a hint to a compiler that an object may change its value in ways not specified by the language so that aggressive optimizations must be avoided.

类型修饰符，用它声明的类型变量表示可以被某些编译器位置因素修改，比如操作系统、线程等，编译器对访问该变量的代码不再优化，提供对特殊地址的稳定访问。

```c
volatile int i=10;
int a = i;
...
// 其他代码，并未明确告诉编译器，对 i 进行过操作
int b = i;
```

而优化做法是，由于编译器发现两次从 i 读数据的代码之间的代码没有对 i 进行过操作，它会自动把上次读的数据放在 b 中。而不是重新从 i 里面读。这样如果 i 是寄存器变量或者一个端口数据就容易出错。

​	其实不只是“内嵌汇编操纵栈”这种方式属于编译无法识别的变量改变，另外更多的可能是多线程并发访问共享变量时，一个线程改变了变量的值，怎样让改变后的值对其它线程 visible。一般说来，volatile 用在如下的几个地方：

1. 中断服务程序中修改的供其它程序检测的变量需要加 volatile；
2. 多任务环境下各任务间共享的标志应该加 volatile；
3. 存储器映射的硬件寄存器通常也要加 volatile 说明，因为每次对它的读写都可能由不同意义；

## 2. Volatile Pointer

```c
const char* cpch;
volatile char* vpch;
//修饰指针指向的对象、数据是const或者volatile

char* const pchc;
char* volatile pchv;
//指针自身的值是const或者volatile
```

1. 可以把非 volatile int 赋给 volatile int，但是不能把非 volatile 对象赋给 volatile 对象。
2. 除了基本类型，用户定义类型也可以用 volatile 类型修饰
3. C++ 中一个有 volatile 标识符的类只能访问它接口的子集，一个由类的实现者控制的子集。用户只能用 const_cast 来获得对类型接口的完全访问。此外，volatile 向 const 一样会从类传递到它的成员。

## 3. 多线程下的 Volatile

有些变量是用 volatile 关键字声明的。当两个线程都要用到某一个变量且该变量的值会被改变时，应该用 volatile 声明，该关键字的作用是防止优化编译器把变量从内存装入 CPU 寄存器中。如果变量被装入寄存器，那么两个线程有可能一个使用内存中的变量，一个使用寄存器中的变量，这会造成程序的错误执行。volatile 的意思是让编译器每次操作该变量时一定要从内存中真正取出，而不是使用已经存在寄存器中的值，如下：

 volatile BOOL bStop = FALSE;

  (1) 在一个线程中：

 while( !bStop ) { … }

 bStop = FALSE;

 return;

  (2) 在另外一个线程中，要终止上面的线程循环：

 bStop = TRUE;

 while( bStop ); //等待上面的线程终止，如果 bStop 不使用 volatile 申明，那么这个循环将是一个死循环，因为 bStop 已经读取到了寄存器中，寄存器中 bStop 的值永远不会变成 FALSE，加上 volatile，程序在执行时，每次均从内存中读出 bStop 的值，就不会死循环了。

  这个关键字是用来设定某个对象的存储位置在内存中，而不是寄存器中。因为一般的对象编译器可能会将其的拷贝放在寄存器中用以加快指令的执行速度，例如下段代码中：

 …

 int nMyCounter = 0;

 for(; nMyCounter<100;nMyCounter++)

 {

 …

 }

 …

  在此段代码中，nMyCounter 的拷贝可能存放到某个寄存器中（循环中，对 nMyCounter 的测试及操作总是对此寄存器中的值进行），但是另外又有段代码执行了这样的操作：nMyCounter -= 1; 这个操作中，对 nMyCounter 的改变是对内存中的 nMyCounter 进行操作，于是出现了这样一个现象：nMyCounter 的改变不同步。

# 指针

```c
// 整型变量
int p;

// 指向int p的指针
int *p;

// 先与[]结合，说明p是int数组
int p[3];

// 先与[]结合，p是数组，且是指针类型，指针指向整型
int *p[3];

// 先与*结合，p是指针，指向整型数组
int (*p)[3];

// 二级指针
int **p;

// 整型参数，返回int的函数
int p(int);

// 返回*p的函数
int (*p)(int);

// 返回整型指针变量组成的数组的指针变量，的函数
int *(*p(int))[3];
```

指针运算

```c
char a[20];
int *ptr = (int*) a;
ptr++;
// ptr的值加上了sizeof(int),
```

数组和指针

```c
// array代表数组本身，类型是int[10]，也是指针，类型int*
int array[10] = {1,...}, value;
value = array[3];

// 数组的大小
sizeof(array);


// 字符串相当于数组
// 结构里单元未必是连续排列
```

## 函数与指针

如果在程序中定义了一个函数，那么在编译时系统就会为这个函数代码分配一段存储空间，这段存储空间的首地址称为这个函数的地址。而且函数名表示的就是这个地址。既然是地址我们就可以定义一个指针变量来存放，这个指针变量就叫作函数指针变量，简称函数指针。

```c
int fun1(char *, int);
int (*pfun1)(char *, int);

pfun1 = fun1;
int a = (*pfun1)("abc", 7);
// 通过函数指针调用函数
```

这个语句就定义了一个指向函数的指针变量 pfun1。首先它是一个指针变量，所以要有一个“\*”，即（\*pfun1）；其次前面的 int 表示这个指针变量可以指向返回值类型为 int 型的函数；后面括号中的 char\*, int 表示这个指针变量可以指向有两个参数且都是 int 型的函数。所以合起来这个语句的意思就是：定义了一个指针变量 pfun1，该指针变量可以指向返回值类型为 int 型，且有 char\*, int 参数的函数。pfun1 的类型为 int(\*)(char\*，int)。

所以函数指针的定义方式为：

> 函数返回值类型 (* 指针变量名) (函数参数列表);

“函数返回值类型”表示该指针变量可以指向具有什么返回值类型的函数；“函数参数列表”表示该指针变量可以指向具有什么参数列表的函数。这个参数列表中只需要写函数的参数类型即可。

我们看到，函数指针的定义就是将“函数声明”中的“函数名”改成“（* 指针变量名）”。但是这里需要注意的是：“（* 指针变量名）”两端的括号不能省略，括号改变了运算符的优先级。如果省略了括号，就不是定义函数指针而是一个函数声明了，即声明了一个返回值类型为指针型的函数。

那么怎么判断一个指针变量是指向变量的指针变量还是指向函数的指针变量呢？首先看变量名前面有没有“*”，如果有“*”说明是指针变量；其次看变量名的后面有没有带有形参类型的圆括号，如果有就是指向函数的指针变量，即函数指针，如果没有就是指向变量的指针变量。

**最后需要注意的是，指向函数的指针变量没有 ++ 和 -- 运算。**

# C Problem

1. 使用 malloc 获取单元。一个指针并不创造结构，而是给出空间容纳可能会使用的地址，创建未被声明的结构需要 malloc，不再需要时要 free。

# C 深度

## 关键字

C 的关键字共有 32 个，根据关键字的作用，可分其为数据类型关键字、控制语句关键字、存储类型关键字和其它关键字四类。

### 数据类型关键字

| 关键字     | 描述                                                         |
| ---------- | ------------------------------------------------------------ |
| *char*     | 声明字符型变量或函数。1 byte                                 |
| *double*   | 声明双精度变量或函数。8 byte                                 |
| *enum*     | 声明枚举类型。|
| *float*    | 声明浮点型变量或函数。4 byte                                 |
| *int*      | 声明整型变量或函数。4 byte                                   |
| *long*     | 声明长整型变量或函数。4 byte                                 |
| *short*    | 声明短整型变量或函数。2 byte                                 |
| *signed*   | 声明有符号类型变量或函数。|
| *struct*   | 声明结构体变量或函数。|
| *union*    | 声明共用体（联合）数据类型。|
| *unsigned* | 声明无符号类型变量或函数。|
| *void*     | 声明函数无返回值或无参数，声明无类型指针（基本上就这三个作用）。|

### 控制语句关键字

循环语句

| 关键字     | 描述                           |
| ---------- | ------------------------------ |
| *for*      | 一种循环语句。|
| *do*       | 循环语句的循环体。|
| *while*    | 循环语句的循环条件。|
| *break*    | 跳出当前循环。|
| *continue* | 结束当前循环，开始下一轮循环。|

 条件语句

| 关键字 | 描述                             |
| ------ | -------------------------------- |
| *if*   | 条件语句。|
| *else* | 条件语句否定分支（与 if 连用）。|
| *goto* | 无条件跳转语句。|

 开关语句

| 关键字    | 描述                       |
| --------- | -------------------------- |
| *switch*  | 用于开关语句。|
| *case*    | 开关语句分支。|
| *default* | 开关语句中的 “其他” 分支。|

 返回语句

| 关键字   | 描述                                       |
| -------- | ------------------------------------------ |
| *return* | 子程序返回语句（可以带参数，也看不带参数）|

### 存储类型关键字

| 关键字     | 描述                                                 |
| ---------- | ---------------------------------------------------- |
| *auto*     | 声明自动变量，一般不使用。|
| *extern*   | 声明变量是在其他文件正声明（也可以看做是引用变量）。|
| *register* | 声明寄存器变量。|
| *static*   | 声明静态变量。|

### 其他关键字

| 关键字     | 描述                                       |
| ---------- | ------------------------------------------ |
| *const*    | 声明只读变量。|
| *sizeof*   | 计算数据类型长度。|
| *typedef*  | 用以给数据类型取别名（当然还有其他作用）。|
| *volatile* | 说明变量在程序执行中可被隐含地改变。|

### 定义

```c
// 定义
int i;
// 声明
extern int i;
```

定义，（编译器）创建一个对象，分配一块内存并取一个名字。一个变量或对象在一定区域内只能被定义一次，否则会重复。

声明，告诉编译器某个名字已经和内存匹配，可以多次声明；告诉编译器内存不许被用作其他变量或对象名。但并没有分配内存。

### Register

最快，但必须是 CPU 寄存器所接受的类型，一个单个值，长度小于等于整型，不存放在内存，不可获取地址。

### Static

1. 修饰变量。变量分局部和全局，都存在内存的静态区。

静态全局，作用域仅限变量被定义的文件中；静态局部变量，函数体中定义，只能在函数中使用，静态值不会被销毁，函数下次仍使用这个值。

1. 修饰函数。静态函数，作用域只局限于本文件，不用担心定义的函数与其他文件的函数重名。
2. C++

### Sizeof

sizeof+ 类型不能省略括号，sizeof+ 变量可以省略

语句

```c++
// 布尔变量
if(flag);

// 非布尔
if(value == 0);

// 浮点，允许误差进度epsinon
if((x >= -EPSINON) && (x <= EPSINON));

// 指针
if(p == NULL);
```

长循环放在最内层，效率高；for 循环取值半开半闭区间；不在循环内修改循环变量；

函数

无返回值默认为 int；

```c
(char *)pvoid++; //ANSI：正确；GNU：正确
(char *)pvoid += 1; //ANSI：错误；GNU：正确
// 如果函数的参数可以是任意类型指针，应声明为void *
void *memcpy(void dest, const void *src, size_t len);
void * memeset(void* buffer, int c, size_t num);
```

可变数组

```c
typedef struct st_type
{
int i;
int a[0];
}type_a;

type_a *p = (type_a*)malloc(sizeof(type_a)+100*sizeof(int));
printf("%d",sizeof(type_a));
free(p);
// 4，但是结构体大小是可变的。
```

### Const

```c
// const只读变量，不可变
// .c中会报错，认为const修饰的为变量但只读；而.cpp可以运行
const int MAX = 100;
int array[MAX];
// 也不能放在case关键词后
```

编译器通常不为普通 const 只读变量分配存储空间，而是将它们保存在符号表中，这使 得它成为一个编译期间的值，没有了存储与读内存的操作，使得它的效率也很高。例如：

```c
 #define M 3 //宏常量 
const int N=5; //此时并未将 N 放入内存中 ...... 
int i=N; //此时为 N 分配内存，以后不再分配！ 
int I=M; //预编译期间进行宏替换，分配内存 
int j=N; //没有内存分配 
int J=M; //再进行宏替换，又一次分配内存！ 
```

const 定义的只读变量从汇编的角度来看，只是给出了对应的内存地址，而不是象#define 一样给出的是立即数，所以，const 定义的只读变量在程序运行过程中只有一份拷贝（因为 它是全局的只读变量，存放在静态区），而#define 定义的宏常量在内存中有若干个拷贝。#define 宏是在预编译阶段进行替换，而 const 修饰的只读变量是在编译的时候确定其值。#define 宏没有类型，而 const 修饰的只读变量具有特定的类型

```c
const int *p; // p 可变，p 指向的对象不可变
int const *p; // p 可变，p 指向的对象不可变
int *const p; // p 不可变，p 指向的对象可变
const int *const p; //指针 p 和 p 指向的对象都不可变

//const 修饰符也可以修饰函数的参数，当不希望这个参数值被函数体内意外改变时使用。
void Fun(const int i);
```

### Volatile

volatile 关键字和 const 一样是一种类型修饰符，用它修饰的变量表示可以被某些编译器 未知的因素更改，比如操作系统、硬件或者其它线程等。遇到这个关键字声明的变量，编 译器对访问该变量的代码就不再进行优化，从而可以提供对特殊地址的稳定访问。

volatile 关键字告诉编译器 i 是随时可能发生变化的，每次使用它的时候必须从内存中取出 i 的值，因而编译器生成的汇编代码会重新从 i 的地址处读取数据放在 k 中。

如果 i 是一个寄存器变量或者表示一个端口数据或者是多个线程的共享数 据，就容易出错，所以说 volatile 可以保证对特殊地址的稳定访问

### Enum

1），#define 宏常量是在预编译阶段进行简单替换。枚举常量则是在编译的时候确定其值。

2），一般在编译器里，可以调试枚举常量，但是不能调试宏常量。

3），枚举可以一次定义大量相关的常量，而#define 宏一次只能定义一个

### Typedef

```c
typedef struct st_type
{
int i;
int a[0];
}type_a, *type_pst;

// 没有区别
struct st_type s1;
type_a s2;
// 没有区别
struct student *s3;
type_pst s4;
type_a *s5;

// define只是文本的替换
// typedef是类型的重命名
#define PCHAR char*
typedef char* pchar;
```

## 符号

编译器会用空格替代原来的注释。

\反斜杠，可以做接续符，可以做转义字符开始标志

||和&&为逻辑运算，逻辑运算前者满足条件后者就不再计算。

|和&是按位。<<和>>按位左移右移，但是位数不能大于数据长度不能小于 0.

```c
// 交换二者的值
x ^= y;
y ^= x;
x ^= y;
```

/和% 要注意存在负数时的商和

优先级：

1. [] () . ->；从左到右
2. -, (强制转换), ++, --, *, &, !, ~, sizeof；从右到左，单目
3. /, *, %
4. +, -
5. <<, >>
6. <=, >, <, >=
7. ==, !=
8. &
9. ^
10. |
11. &&
12. ||
13. ?=
14. 运算符 +=
15. ,

容易出错：

1. .高于 *，->消除问题，例如 (*p).f
2. [] 高于 *，int(*ap)[]
3. 函数 () 高于 *，int (*fp)()，函数指针，所指函数返回 int
4. ==高于!=，((val&mask) != 0)

## 预处理

宏定义是实参代换形参，而不是值传递

\#include 其中，filename 为要包含的文件名称，用尖括号括起来，也称为头文件，表示预处理到 系统规定的路径中去获得这个文件（即 C 编译系统所提供的并存放在指定的子目录下的头 文件）。找到文件后，用文件内容替换该语句。

\#include “filename” 其中，filename 为要包含的文件名称。双引号表示预处理应在当前目录中查找文件名为 filename 的文件，若没有找到，则按系统指定的路径信息，搜索其他目录。找到文件后，用 文件内容替换该语句

### #pragma

它的作用是设定编译器的 状态或者是指示编译器完成一些特定的动作。#pragma 指令对每个编译器给出了一个方法, 在保持与 C 和 C ++ 语言完全兼容的情况下,给出主机或操作系统专有的特征。依据定义,编译 指示是机器或操作系统专有的,且对于每个编译器都是不同的。

\#pragma message(“消息文本”) 当编译器遇到这条指令时就在编译输出窗口中将消息文本打印出来。

\#pragma code_seg( ["section-name"[,"section-class"] ] ) 它能够设置程序中函数代码存放的代码段，当我们开发驱动程序的时候就会使用到它。

\#pragma once (比较常用）只要在头文件的最开始加入这条指令就能够保证头文件被编译一次

\#pragma hdrstop 表示预编译头文件到此为止，后面的头文件不进行预编译。BCB 可以 预编译头文件以加快链接的速度，但如果所有头文件都进行预编译又可能占太多磁盘空间，所以使用这个选项排除一些头文件

\#pragma resource "*.dfm" 表示把*.dfm 文件中的资源加入工程。*.dfm 中包括窗体 外观的定义

\#pragma warning

\#pragma comment(…) 该指令将一个注释记录放入一个对象文件或可执行文件中。

论内存对齐的问题和#pragma pack（）

## 指针和数组

```c
int *p = (int *)0x12ff7c;
*p = 0x100;
*(int *)0x12ff7c = 0x100;

int a[5];
```

数组名左值和右值：x=y

左值：编译时确定，在特定区域保存该地址。x 含义是 x 所代表地址，编译器认为左边符号位所代表的地址内容是一定可以被修改的。只能给非只读变量赋值。

右值：运行时知道，y 是 y 所代表地址里的内容

a 作为右值，与&a[0] 一样，代表数组首元素的首地址，并未分配内存来存该地址。a 不能作为左值

以指针形式和下标形式访问指针数字没有区别。

数组和指针本质是不同的，声明和定义时会有区分。

| 指针                                                         | 数组                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 保存数据的地址，存入指针变量 p 的数据被当作地址。p 本身的地址由编译器存储 | 保存数据，数组名 a 代表的是数组首元素地址而不是数组的首地址。&a 才是整个数组的首地址，a 本身的地址由编译器存储。|
| 间接访问数据，首先取得 p 的内容作为地址，从该地址提取数据或写入。指针可以以指针形式 *(p+i)，也可以以数组形式 p[i]，本质都是 i*sizeof(类型) 个 byte 作为数据真实地址 | 直接访问数据，a 是整个数组的名字，数组内每个元素并没有名字。只能通过“具 名 + 匿名”的方式来访问其某个元素，不能把 数组当一个整体来进行读写操作。数组可以 以指针的形式访问 *(a+i)；也可以以下标的形 式访问 a[i]。但其本质都是 a 所代表的数组首 元素的首地址加上 i*sizeof(类型) 个 byte 作为 数据的真正地址 |
| 通常用于动态数据结构                                         | 通常用于存储固定数目且数据类型相同的元 素。|
| 相关的函数为 malloc 和 free。| 隐式分配和删除                                               |
| 通常指向匿名数据（当然也可指向具名数据）| 自身即为数组名                                               |

## 内存管理

静态区、栈 stack、堆 heap
