---
tags:
  - GPU
---


# Chapter 5: Memory architecture and data locality

In this chapter we will focus on the on-chip memory architecture of the GPU and begin to study how one can organize and position data for efficient access by a massive number of threads.

本章节系统介绍了 CUDA 中的内存架构，阐释了访存效率对于计算效率的影响，使用分块技术对矩乘进行改进，实现了 16x 的性能提升。

## 5.1 Importance of memory access efficiency 访存效率的重要性

以 [3.4 Matrix multiplication 矩阵乘法](https://www.zhouxin.space/notes/note-on-programming-massively-parallel-processors-a-hands-on-approach-4th-edition-part-1/#3.4-matrix-multiplication-%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95) 中给出的矩阵乘法为例，每个线程负责结果中一个元素的计算，即计算两个向量的内积。

|   |
|---|
||

|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```c<br>__global__ void MatrixMulKernel(float* M, float* N,<br>                                float* P, int Width) {<br>    int row = blockIdx.y * blockDim.y + threadIdx.y;<br>    int col = blockIdx.x * blockDim.x + threadIdx.x;<br>    if ((row < Width) && (col < Width)) {<br>        float Pvalue = 0;<br>        for (int k = 0; k < Width; ++k) {<br>            Pvalue += M[row * Width + k] * N[k * Width + col];<br>        }<br>        P[row * Width + col] = Pvalue;<br>    }<br>}<br>``` |

如上所示，在计算内积的循环中，每次都要访问两次全局内存，以及 2 次浮点计算。这里引入一个指标“计算密度” computational intensity，计算公式为浮点操作数 FLOP 比上全局访存字节数，在上述代码中，计算密度为 2FLOP/8B = 0.25FLOP/B。

计算密度这一指标能够指示该 CUDA 程序是否充分利用了核心的计算能力。例如，在 A100 张中，全局访存带宽为 1555GB/s，将其与计算密度相乘，可以得到该程序所需的浮点计算能力为 389GFLOPS，远低于 A100 实际具有的浮点计算能力 19500GFLOPS，遑论 A100 中还具有专门的 tensor core，具有 156000GFLOPS 的浮点算力。

这类被内存带宽拖累的程序被称为内存瓶颈程序。根据 A100 的全局带宽和浮点计算能力，我们可以计算出至少需要 19500/1555=12.5FLOP/B 的计算密度才能充分发挥其计算性能。

## 5.2 CUDA memory types CUDA 内存类型

CUDA 设备提供了多种内存类型以提高计算密度，如下所示。  
![image.png](https://pics.zhouxin.space/202409231026587.png?x-oss-process=image/quality,q_90/format,webp)

最底层的为全局和常量内存，host 可以对其读写，device 可以对全局内存读写，可以对常量内存以低延迟和高带宽进行读，而不可写。

还有一部分是局部内存，其实际上是全局内存的一部分，与全局内存具有类似的延迟和带宽。每个线程会在全局内存中分配一段仅其自己可读写的内存作为局部内存，用于存放寄存器放不下的变量，例如静态数组、溢出的寄存器以及线程的函数调用栈。

寄存器和共享内存是片上内存，其中的变量能够并行地以非常高的速率被访问。寄存器仅对该线程自己可见，用于保存线程经常使用到的一些仅自己可见的变量，共享内存则由一个 Block 内的所有变量共享。

通过使用不同的内存类型，程序员可以控制不同变量的访问速度和可见性。

除了本身的延迟和带宽，访问寄存器更快还有一个原因是指令数量。将两个寄存器中的浮点数相加只需要一条浮点数加法指令，而如果两个不在寄存器中的浮点数加起来则需要额外指令将数据加载到寄存器、将结果搬运回内存。执行这些额外指令本身也会消耗更长的时间。

尽管寄存器和共享内存都是片上内存，但共享内存是内存体系的一部分，其中数据也需要读取到寄存器中再操作，因此相比寄存器其具有更高的延迟和更低的吞吐量。术语 scratchpad memory 指的就是这一部分板上内存。

声明的不同类型的变量其保存的位置、作用域和声明周期各不相同，具体对应关系如下表所示：

|变量声明|内存|作用域|生命周期|
|---|---|---|---|
|非数组的自动变量|寄存器|thread|网格|
|自动数组变量|本地|thread|网格|
|`__device__ __shared__ int SharedVar;`|共享|block|网格|
|`__device__ int GlobalVar;`|全局|grid|应用程序|
|`__device__ __constant__ int ConstVar;`|常量|grid|应用程序|

## 5.3 Tiling for reduced memory traffic 通过分块减少访存

将数据划分为在共享内存中放得下的小块可以减少对全局内存的访问，数据分块的前提是每一块都可以独立地进行计算，不是所有的数据结构、也不是所有的核函数都可以进行分块处理。

![image.png](https://pics.zhouxin.space/202409251414411.png?x-oss-process=image/quality,q_90/format,webp)

如上图所示，正在之前实现的矩阵乘法中，每个线程独立计算一个元素，第一个 block 由四个线程组成，这些线程之间有重复读取全局内存的过程，例如 P00 和 P01 均读取了 M 的第一行。可以通过将这些元素读入共享内存来实现对全局内存的减半访问。

在矩乘中，实际减少的访存次数取决于 block 的 size，具体来说，如果 block 中的线程以 n×n 的规格组织，则能够将访存次数减少到 1/n。

需要注意的是，共享内存的大小是有限的，如果一个 block 中的线程数过多或者矩乘中的维度过大，共享内存可能存不下分块后需要用到的数据，此时可以将其划分为更小的块以便读入共享内存中。

例如，按照 2×2 对 M 和 N 进行分块，4×4 的矩乘将由两阶段完成。对于 block00 来说，第一阶段将 `M[0:2, 0:2]` 和 `N[0:2, 0:2]` 读入共享内存计算矩乘；第二阶段将 `M[0:2 2:4]` 和 `N[2:4, 0:2]` 读入共享内存，计算矩乘并累加到前面的结果中。各线程完成的任务如下表所示：  
![image.png](https://pics.zhouxin.space/202409251455300.png?x-oss-process=image/quality,q_90/format,webp)

## 5.4 A tiled matrix multiplication kernel 分块矩乘核函数

分块矩乘核函数如下：

|   |
|---|
||

|   |
|---|
|```c<br>#define TILE_WIDTH 16<br>__global__ void matrixMulKernel(float* M, float* N, float* P, int Width)<br>{<br>    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];<br>    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];<br><br>    int bx = blockIdx.x; int by = blockIdx.y;<br>    int tx = threadIdx.x; int ty = threadIdx.y;<br><br>    // Identify the row and column of the P element to work on<br>    int Row = by * TILE_WIDTH + ty;<br>    int Col = bx * TILE_WIDTH + tx;<br><br>    // Loop over the M and N tiles required to compute P element<br>    float Pvalue = 0;<br>    for (int ph = 0; ph < Width/TILE_WIDTH; ++ph) {<br>        // Collaborative loading of M and N tiles into shared memory<br>        Mds[ty][tx] = M[Row*Width + ph*TILE_WIDTH + tx];<br>        Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*Width + Col];<br>        __syncthreads();<br><br>        for (int k = 0; k < TILE_WIDTH; ++k) {<br>            Pvalue += Mds[ty][k] * Nds[k][tx];<br>        }<br>        __syncthreads();<br>    }<br>    P[Row*Width + Col] = Pvalue;<br>}<br>```|

与之前的分析类似，首先声明两个共享内存用于存放当前阶段计算需要用到的数据。在阶段的循环中，首先 co-fetch 数据到共享结存中然后进行矩乘计算。这里使用了两次同步，第一次是防止数据还没有加载完就进行读取，第二次是防止计算还没完成就写入下一阶段的数据。

16-26 行演示一种被称为 strip-mining 的技术，即将原始很长的循环划分为多个阶段进行，每个阶段内部有一个嵌套循环负责执行原循环中连续的一小部分。

通过分块，我们将矩乘核函数的计算密度从 0.25 OP/B 提升到了 4 OP/B，这是 16 倍的提升。当然，离 A100 12.5 OP/B 还有很远的距离。更多的优化方法将在后文中继续讨论。

## 5.5 Boundary check 边界检查

这节私以为没有单独拎出来的必要，核心内容就是在加载数据和计算时都要进行边界检查，不要越界。此节跳过。

## 5.6 Impact of memory usage on occupancy 内存使用对使用率的影响

正如第四章所提到的，寄存器和共享内存的过度使用将成为制约每个 SM 中分配到的线程数的负面因素。例如，在 A100 中，每个 SM 共享内存大小为 164KB，按照按照最大线程数 2048 计算，一个 block 中平均每个线程使用的共享内存大小不能超过 164KB/2048 = 82B。而在我们之前的矩乘中，每个线程平均加载了 2 个浮点数，即 8B，小于 82B。这说明在之前的核函数中，内存使用不会成为瓶颈。

可以针对不同的硬件平台，使用不同大小的共享内存。这涉及到了 CUDA 中动态分配共享内存技术，使用关键字 `extern __shared__` 来声明一个动态共享内存：

|   |
|---|
||

|   |
|---|
|```c<br>extern __shared__ Mds_Nds[];<br>```|

该动态数组只有一个，如果由多个变量共享，需要由程序员控制不同变量之间的边界。

在调用核函数时，使用第三个参数动态传入共享内存的大小。还可以在核函数的参数中增加相应的字段用于表示共享内存中不同变量的长度。
