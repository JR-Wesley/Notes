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
