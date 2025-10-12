![[gemm-hierarchy-with-epilogue-no-labels.png]]

本系列记录了一个完整的 cutlass/cute 入门文档，对 GEMM 加速库提供一个整体的认识。

# Resources

- 作为手册查询的文档，详尽地罗列了特性，需要保持更新。
	- <a href="https://docs.nvidia.com/cutlass/index.html">cutlass 官方文档</a>
	- [CUTLASS GitHub Repository](https://github.com/NVIDIA/cutlass)
	- [CUTLASS Documentation](https://nvidia.github.io/cutlass/index.html)
- 重点：入门，了解 GEMM 优化和性能分析、cutlass 的分层、CuTe Layout 等核心概念。官方的文档默认读者了解底层优化与实现的，所以并不适合入门。
	- nv blog 2017 [CUTLASS: Fast Linear Algebra in CUDA C++](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)：强烈推荐，内容对应 GTC2018；详细讲解了在 cutlass 背景下针对 GEMM 优化的分块、内外积转换、缓存，以及相关基础概念和分层设计。
	- <a href="https://siboehm.com/articles/22/CUDA-MMM">GEMM 优化博客</a>：强烈推荐，结合性能指标分析从一个最原始的 GEMM 开始优化，注重性能分析，建议结合 NCU 动手实践，不过有些内容可能会有点让人疑惑。
	- <a href="https://developer.nvidia.com/blog/cutlass-principled-abstractions-for-handling-multidimensional-data-through-tensors-and-spatial-microkernels/">nv blog 2025 cutlass</a> 强烈推荐，内容对应 GTC 2023，介绍 cute，结合官方文档的<a href="https://docs.nvidia.com/cutlass/media/docs/cpp/cute/index.html#">CuTe</a>系列理解 Layout 设计抽象。
	- <a href="https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/">nv 2025 blog cutlass 3 介绍</a> 3.x 系列引入的特性和抽象分层，对应 GTC 2023，是上文的补充第二部分。
	- 现代C++，CUTLASS大量用到了现代C++的特性如模板元编程。
- GTC 官方演讲，cutlass 从 2018 年起每年都会在 GTC 演讲发表最新进展：
	- GTC 2018：[CUTLASS: Software Primitives for Dense Linear Algebra at All Levels and Scales within CUDA](https://www.nvidia.com/en-us/on-demand/session/gtcsiliconvalley2018-s8854/)，初次介绍 cutlass
	- GTC 2019：[PROGRAMMING TENSOR CORES: NATIVE VOLTA TENSOR CORES WITH CUTLASS](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9593-cutensor-high-performance-tensor-operations-in-cuda-v2.pdf)
	- GTC 2020：[Developing CUDA Kernels to Push Tensor Cores to the Absolute Limit on NVIDIA A100](https://www.nvidia.com/en-us/on-demand/session/gtcsj20-s21745/)  ；ppt https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21745-developing-cuda-kernels-to-push-tensor-cores-to-the-absolute-limit-on-nvidia-a100.pdf ；B 站视频：[Ampere arch cutlass 2.2 【Developing CUDA kernels to push Tensor Cores to the Absolute Limit on NVIDIA A10】](https://www.bilibili.com/video/BV1tZ421J7M9/?share_source=copy_web&vd_source=fd37be71d17f708cc53476cbd29e590f)。
	- GTC 2021：[Accelerating Convolution with Tensor Cores in CUTLASS](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31883/)，卷积支持。
	- GTC 2022：[Accelerating Backward Data Gradient by Increasing Tensor Core Utilization in CUTLASS](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41996/)
	- GTC 2022：[CUTLASS: Python API, Enhancements, and NVIDIA Hopper](https://www.nvidia.com/en-us/on-demand/session/gtcfall22-a41131/)
	- GTC 2023：[Developing Optimal CUDA Kernels on Hopper Tensor Cores](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51413/)，cutlass 3.x 引入 cute 抽象
	- GTC 2024：[CUTLASS: A Performant, Flexible, and Portable Way to Target Hopper Tensor Cores](https://www.nvidia.com/en-us/on-demand/session/gtc24-s61198/)，卷积支持、epilog tree。
	- GTC 2025：[Programming Blackwell Tensor Cores with CUTLASS)](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72720/)，TBC，blackwell 特性。
- 典型应用：
	- <a href="https://tridao.me/blog/2024/flash3/">tri dao 对 flash-attention 3 的介绍</a>
	- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
	- [FlashMLA](https://github.com/deepseek-ai/FlashMLA)
	- [flash-attention](https://github.com/Dao-AILab/flash-attention) v3
	- 视频 [GTC Silicon Valley-2019 ID:S9593,cuTENSOR: High-performance Tensor Operations in CUDA](https://developer.nvidia.com/gtc/2019/video/S9593/video)
- 注意：
	- 一些可能有时效性的信息，以及中文搬运或翻译内容有混乱和错误。
	- 有一些特性可能是架构特有的。
	- cutlass 2.x 3.x的抽象和使用是有区分的。
	- Ampere GPU 新增了`LDGSTS`指令，数据块从 Global Memory 到 Shared Memory 的过程不需要经过中间寄存器，可以进一步的优化 SGEMM 的性能。以前的博客可能会提到要经过积存的共享内存读写，这一步其实现在是不存在的了


**PyTorch**

- [Deep Dive on CUTLASS Ping-Pong GEMM Kernel](https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/)


其他博客
- **Colfax**：他们的系列教程是比较深入的，会用到很多深层次的特性和前沿架构。
	- [CUTLASS Tutorial: Fast Matrix-Multiplication with WGMMA on NVIDIA® Hopper™ GPUs](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/)
	- [Tutorial: Matrix Transpose in CUTLASS](https://research.colfax-intl.com/tutorial-matrix-transpose-in-cutlass/)
	- [CUTLASS Tutorial: Persistent Kernels and Stream-K](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/)
	- [CUTLASS Tutorial: Mastering the NVIDIA® Tensor Memory Accelerator (TMA)](https://research.colfax-intl.com/tutorial-hopper-tma/)
	- [Developing CUDA Kernels for GEMM on NVIDIA Hopper Architecture using CUTLASS](https://research.colfax-intl.com/nvidia-hopper-gemm-cutlass/)
	- [A note on the algebra of CuTe Layouts](https://research.colfax-intl.com/a-note-on-the-algebra-of-cute-layouts/)
- CUTE 系列，除了 reed之外的博客基本都是把传统 GEMM优化用 CUTE实现了一遍，没有什么新东西。
	- 这些内容都是完全使用CUTE的实现GEMM优化的，CUTE本质是把CUDA编程模型的优化方法封装起来了，我觉得这些系列一下子丢出来太多理论优化点，没有讲 CUDA 基础内容很难知道在说什么，而且缺乏对于底层细节比如合并访存、bank 冲突的
	- [reed](https://www.zhihu.com/people/reed-84-49) [CUTE系列](https://www.zhihu.com/column/c_1696937812497235968)：从基础的张量和layout讲解CUTE抽象。
	- [杨远航](https://www.zhihu.com/people/yang-yuan-hang-83) [CUTE 系列笔记](https://www.zhihu.com/column/c_1938664963049763058)：使用 CUTE 抽象，实现最简单的全局内存访问模式的GEMM、然后到多层tile的GEMM。
	- https://zhuanlan.zhihu.com/c_1946580149119202264
 [写给大家看的 CuTe 教程：tiled mma](https://zhuanlan.zhihu.com/p/1937145378446226159)

GEMM 逐步优化指南：https://linn-ylz.com/Computer-Science/CUDA/CUDA-SGEMM-optimization-notes/


**Miscellaneous**

- [Build and Develop CUTLASS CUDA Kernels](https://leimao.github.io/blog/Build-Develop-CUTLASS-CUDA-Kernels/) (How to create a CUDA Docker container for CUTLASS kernel development)
- [learn-cutlass](https://gty111.github.io/2023/03/21/learn-cutlass-1/)
- [Lecture 15: CUTLASS (GPU MODE)](https://www.youtube.com/watch?v=G6q719ck7ww&ab_channel=GPUMODE)
- [CUTLASS: A CUDA C++ Template Library for Accelerating Deep Learning Computations (The Linux Foundation)](https://www.youtube.com/watch?v=PWWOGrLZtZg&ab_channel=TheLinuxFoundation)
- [Lecture 36: CUTLASS and Flash Attention 3 (GPU MODE)](https://www.youtube.com/watch?v=JwUcZwPOCpA&t=2831s&ab_channel=GPUMODE)
- <a href="https://www.bilibili.com/video/BV1kToTY6Eh5/?p=4&share_source=copy_web&vd_source=fd37be71d17f708cc53476cbd29e590f">B 站讲解视频 【【CUDA 进阶】Cutlass 软件抽象分层与源码浅析（已完结）</a>，建议了解原理后学习 cutlass 2.x 示例代码。

mma tensor core: https://georgelyu.github.io/p/tensor_core_ptx/

pragramming tc vlta: https://developer.download.nvidia.cn/video/gputechconf/gtc/2019/presentation/s9593-cutensor-high-performance-tensor-operations-in-cuda-v2.pdf

# CUTLASS 与 GEMM 入门

## 介绍

> [!note] CUTLASS
> CUTLASS is **a collection of abstractions for implementing high-performance matrix-matrix multiplication (GEMM) and related computations at all levels and scales** within CUDA. It incorporates strategies for hierarchical decomposition and data movement. CUTLASS decomposes these “moving parts” into reusable, modular software components and abstractions.
> Primitives for different levels of a conceptual parallelization hierarchy can be specialized and tuned via custom tiling sizes, data types, and other algorithmic policy. The resulting flexibility simplifies their use as building blocks within custom kernels and applications.

CUTLASS provides:

- Threadblock-level abstractions for matrix multiply-accumulate operations
- Warp-level primitives for matrix multiply-accumulate operations
- Epilogue components for various activation functions and tensor operations
- Utilities for efficiently loading and storing tensors in memory

总结一下就是  CUTLASS 是一套模板库，为GEMM层次分解、数据抽象、计算提供了模块化的组件，支持定制优化、灵活性扩展。

## Cutlass 3.x VS 2.x

> 参考：[NV AI 技术开放日](https://www.bilibili.com/video/BV1XH4y1c7JZ/?share_source=copy_web&vd_source=fd37be71d17f708cc53476cbd29e590f)
> [cutlass issue](https://github.com/NVIDIA/cutlass/issues/792)

很多博客似乎没有讲到 CUTLASS 2.x 与 3.x 的一些区别，实际上这两代在抽象和实现上有很多进展和不同的，另外有一些优化方法是硬件架构特有的，需要注意区分。CUTLASS 3.x 和 CUTE就是现在最新的使用方法，提供了更高的灵活性，但是它封装了过多的底层优化，可能一开始有点难以理解。

2.x

- **矩阵配置**：详细定义了矩阵 A、B、C（或 D ）的元素类型（`ElementA` 、`ElementB` 、`ElementC` ）、布局（`LayoutA` 为行主序，`LayoutB` 、`LayoutC` 为列主序 ）和对齐方式（`AlignmentA` 、`AlignmentB` 、`AlignmentC` ） 。
- **核心内核配置**：指定了累加器元素类型（`ElementAccumulator` ）、架构标签（`ArchTag` ）、操作符类（`OperatorClass`  ）、线程块形状（`TileShape` ）、集群形状（`ClusterShape` ），以及阶段数类型（`StageCountType` ）和内核调度（`KernelSchedule` ）自动配置 。整体配置较为细致，涉及众多底层参数。

3.x

> [!note] CuTe
> CUTLASS 3.0 introduced a new core library, CuTe, to describe and manipulate tensors of threads and data. CuTe is **a collection of C++ CUDA template abstractions for defining and operating on hierarchically multidimensional layouts of threads and data**. CuTe provides `Layout` and `Tensor` objects that compactly package the type, shape, memory space, and layout of data, while performing the complicated indexing for the user. This lets programmers focus on the logical descriptions of their algorithms while CuTe does the mechanical bookkeeping for them. With these tools, we can quickly design, implement, and modify all dense linear algebra operations.

CUTE was introduced in CUTLASS 3.0 and represents a significant evolution in NVIDIA's approach to tensor computing. CUTE introduces:

- A unified tensor abstraction that works across different hardware levels
- Powerful layout mapping capabilities for tensors
- Composable building blocks for tensor algorithms
- A more intuitive programming model for complex tensor operations

作为对比：

- **CUDA** is the base programming model and platform for NVIDIA GPUs. It provides the fundamental parallel computing architecture and programming interface.
- **CUTLASS** is a library built on top of CUDA that provides optimized implementations of matrix operations.
- **CUTE** is a higher-level abstraction built on top of CUTLASS that simplifies tensor programming.

 Key Differences

|Feature|CUDA|CUTLASS|CUTE|
|---|---|---|---|
|Level of Abstraction|Low-level GPU programming|Matrix operation templates|High-level tensor abstractions|
|Focus|General GPU computing|Matrix multiplication primitives|Flexible tensor operations|
|Programming Model|Explicit thread/block management|Threadblock/warp abstractions|Layout-focused tensor abstractions|
|Optimization Control|Manual|Template-based|Layout-driven|

CUTLASS 3.x 是在 2.x 基础上的重大升级，相比 2.x，3.x 在架构、配置方式、对 GPU 架构的支持等方面都有变化，具体区别如下：

- **GPU 架构支持**：CUTLASS 2.x 支持多种 Pre - Hopper 架构的 GPU，如 Ada（sm_89）、Ampere（sm_8x）、Turing（sm_75）和 Volta（sm_70）等。CUTLASS 3.x 保留了 2.x 的所有特性，额外增加了对 Hopper 架构（sm_90a）的支持，还支持 sm_90/sm_90a 的新特性，如 TMA、wgmma、集群配置等。
- **配置方式**：CUTLASS 2.x 需要明确指定更多配置参数，包括输入输出和计算的数据类型、矩阵布局、使用张量核心还是 SIMT 核心、CUDA SM 架构版本、线程块的 tile 大小、Warp 的 tile 大小、MMA 指令大小、线程块调度方式等，尤其强调要指定 WarpShape 和 InstShape。CUTLASS 3.x 配置更加简化，主要需指定矩阵元素类型、布局和对齐方式，核心 kernel 配置（包括累加器类型、架构标签、操作类等），TileShape（BlockShape）和 ClusterShape，无需明确指定 WarpShape 和 InstShape。
- **代码风格与抽象层次**：CUTLASS 3.x 采用了 CuTe 抽象（CUDA Templates），引入了新的核心库 CuTe 来描述和操作线程与数据张量，提供了 `Layout` 和 `Tensor` 对象，简化了设计，提高了代码的可组合性和可读性。CUTLASS 2.x 未采用此抽象，代码风格相对更传统，抽象层次没有 3.x 高。
- **并行计算策略**：CUTLASS 3.x 定义了更清晰的问题形状描述 `ProblemShape`，引入了更灵活的瓦片调度器 `Tile Scheduler`，官方实现的调度器支持 `Split - K` 模式，对并行计算策略进行了更清晰和灵活的抽象。2.x 版本在这方面的抽象不如 3.x 清晰灵活。

在选择使用 CUTLASS 版本时，如果是在 Hopper 架构的 GPU 上工作，并且想要充分利用芯片的新特性和性能，建议选择 CUTLASS 3.x。如果是在 Pre - Hopper 架构的 GPU 上开发，或者项目对代码改动兼容性要求较高，不想引入过多新特性和变化，使用 CUTLASS 2.x 即可，因为 2.x 的特性在 Hopper 芯片上也仍然受支持。

## 安装与编译

参考官方编译方法和示例： https://docs.nvidia.com/cutlass/media/docs/cpp/quickstart.html# 

CUTLASS 是纯头文件组成的库，所以只需要在编译路径下包含库的头文件即可，如：

```cmake
set(CUTLASS_PATH "your-path/cutlass/include/")
set(CUTLASS_UTIL_PATH "your-pathc/cutlass/tools/util/include")
# ...
```

# Efficient Matrix Multiplication on GPUs

> 本章来源： https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/ ，也即 2018 GTC 的内容，同时包括。
> 本文主要讲参考文章中的理论分析部分，代码分析可见 `cutlass/gemm`。

## GEMM Introduction

> [!note] GEMM 优化
> 注：这节讲述了两点，一个是内积转外积，一个是对输出 C 矩阵分块，现有很多教程会直接把这两点揉到一起，结合 shared memory 讲述。

GEMM computes **C** = _alpha_ **A * B +** _beta_ **C**, where **A**, **B**, and **C** are matrices.  **A** is an _M_-by-_K_ matrix, **B** is a _K_-by-_N_ matrix, and **C** is an _M_-by-_N_ matrix. For simplicity, let us assume scalars _alpha=beta=1_ in the following examples. Later, we will show how to implement custom element-wise operations with CUTLASS supporting arbitrary scaling functions.

> 图示可见 [[#Thread Block Tile|Thread Block Tile]]

```c
for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
       for (int k = 0; k < K; ++k) 
            C[i][j] += A[i][k] * B[k][j]; // dot product of row in A and col in B
```

The element of **C** at position (_i, j)_ is the _K_-element **dot product** of the _i_-th row of **A** and the _j_-th column of **B**. Ideally, performance should be limited by the arithmetic throughput of the processor. Indeed, for large square matrices where _M=N=K_, the number of math operations in a product of matrices is _O(N__3__)_ while the amount of data needed is _O(N__2__),_ yielding a compute intensity on the order of _N_. However, taking advantage of the theoretical compute intensity requires reusing every element _O(N)_ times. Unfortunately, the above “inner product” algorithm depends on holding a large working set in fast on-chip caches, which results in thrashing as _M, N,_ and _K_ grow.

> 理论计算强度为 $N$，即一次读写计算 $N$ 次即可发挥硬件算力。

A better formulation _permutes_ the loop nest by structuring the loop over the _K_ dimension as the outermost loop. This form of the computation loads a column of **A** and a row of **B** once, computes its _outer product_, and _accumulates_ the result of this outer product in the matrix **C**. Afterward, this column of _A_ and row of _B_ are never used again.

> 内积：向量转标量；外积：向量转矩阵

```c
for (int k = 0; k < K; ++k)     // K dimension now outer-most loop
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            C[i][j] += A[i][k] * B[k][j];
```

> [!tip] 解决缓存复用问题
> - 原始顺序是内积：固定 A 的第 `i` 行，C 的第 `j` 列，遍历 `k` 完成点积。A 行连续访问，B 是跨列访问。
> - B 是 “跳跃式访问” ，每次读一个 `B[k][j]`，都要从主存加载一整行到缓存（cache line），但只用了一个元素，造成**缓存未命中**！`M N K` 较大时无妨放入完整行列 → **缓存颠簸（thrashing）**
> - 转换后内积转外积：从 “**点积积累**”（A 行 ×B 列的 K 维点积）转变为 “**外积累加**”（A 的第 k 列 × B 的第 k 行，得到一个 M×N 的外积矩阵，再累加到 C。
> - B 和 C 的访问是**高度局部化和连续的**，A 是列访问，但是 `A[i][k]` 缓存一次复用 `j` 次，而且之后不需要再使用，

> 性能差异的本质：计算强度与内存带宽的匹配
> GEMM 的性能瓶颈由 “**计算强度**”（每传输 1 字节数据需执行的浮点运算次数，FLOPs/Byte）决定：
> - 理论上，矩阵乘法的计算强度为 **K/2**（对 M=N=K 的方阵，总运算量 O (K³)，总数据量 O (K²)，即每字节数据需执行 K 次运算）。
> - 但实际性能能否达到理论值，取决于 “数据是否能在缓存中被复用 K 次”—— 如果数据每次都要从内存重新读取，计算强度会暴跌到接近 0（每字节仅执行 1 次运算），性能被内存带宽卡死。

One concern with this approach is that it requires all _M_-by-_N_ elements of **C** to be live to store the results of each multiply-accumulate instruction, ideally in memory that can be written as fast as the multiply-accumulate instruction can be computed. We can reduce the working set size of **C** by partitioning it into tiles of size _Mtile_-by-_Ntile_ that are guaranteed to fit into on-chip memory. Then we apply the “outer product” formulation to each tile. This leads to the following loop nest.

```c
// 外层：遍历C矩阵的“瓷砖块”（按Mtile、Ntile步长拆分）
for (int m = 0; m < M; m += Mtile)                // iterate over M dimension
    for (int n = 0; n < N; n += Ntile)            // iterate over N dimension
		// 中层：外积的k维度（与之前一致，遍历点积维度）
        for (int k = 0; k < K; ++k)
			// 内层：计算当前瓷砖块的所有元素（Mtile×Ntile）
            for (int i = 0; i < Mtile; ++i)       // compute one tile
	            for (int j = 0; j < Ntile; ++j) {
                    int row = m + i;
                    int col = n + j;
                    C[row][col] += A[row][k] * B[k][col];
                }
```

For each tile of **C**, tiles of **A** and **B** are fetched exactly once, which achieves _O(N)_ compute intensity. The size of each tile of **C** may be chosen to match the capacity of the L1 cache or registers of the target processor, and the outer loops of the nest may be trivially parallelized. This is a great improvement!

> 整个计算过程的**计算强度（FLOPs/Byte）达到 O (K)**（每传输 1 字节 A/B 数据，执行 K 次乘累加运算），完全逼近 GEMM 的理论计算强度（K/2）

Further restructuring offers additional opportunities to exploit both locality and parallelism. Rather than exclusively accumulate _vector_ outer products, we can accumulate the products of _matrices_ by stepping through the _K_ dimension in blocks. We refer to this concept generally as **accumulating matrix products**.

> 最后，在 “瓷砖分块” 的基础上进一步优化 ——**将 “k 维度的逐元素遍历” 改为 “k 维度的块遍历”**，即从 “向量外积累加” 升级为 “矩阵外积累加”，进一步提升效率。如：
> ```c

	// 新增：k维度按Ktile分块
	for (int k = 0; k < K; k += Ktile)
	  // 内层k循环：遍历当前k块
	  for (int k_inner = 0; k_inner < Ktile; ++k_inner)
	    // 后续C/A/B的瓷砖循环...

> ```

> [!tip] 对 C 分块
> 上面方法需要存储整个 C 矩阵，因此通过分块进一步提升缓存复用和计算效率。
> - 未分块时：C 的每个元素需要从 DRAM 读取→更新→写回 DRAM，每次访问耗时数十时钟周期；
> - 分块后：当前瓷砖块的所有 `C [row][col]`（Mtile×Ntile 个元素）会被一次性加载到 L1 缓存中，在整个 k 循环（K 轮）中，所有对该瓷砖的更新都在缓存内完成，仅需在 “处理完整个瓷砖” 后写回 DRAM 一次。
> 
> A/B 矩阵复用：
> - A 的一列、B 的一行会在缓存中复用。
> 
> 同时可支持并行化：
>- 分块后的外层循环（m 和 n 循环，即 “遍历 C 的瓷砖块”）是**无依赖的**：不同瓷砖块的计算完全独立。瓷砖大大小根据硬件属性来决定。

## Hierarchical GEMM Structure

CUTLASS applies the tiling structure to implement GEMM efficiently for GPUs by decomposing the computation into a hierarchy of **thread block tiles**, **warp tiles**, and **thread tiles** and applying the strategy of accumulating matrix products. This hierarchy closely mirrors the [NVIDIA CUDA programming model](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model), as Figure 1 shows. Here, you can see data movement from global memory to shared memory (matrix to thread block tile), from shared memory to the register file (thread block tile to warp tile), and from the register file to the CUDA cores for computation (warp tile to thread tile).

![[fig-09-complete-hierarchy-1.png]]

> Figure 1. The complete GEMM hierarchy transfers data from slower memory to faster memory where it is reused in many math operations.
> 注意原文图中的格子数量不完全等于维度信息。

> [!tip] CUTLASS 分层
> 1. **分层分块与 GPU 硬件架构的 “镜像匹配”**：CUTLASS 的分块层级完全对应 GPU 的 “全局内存→共享内存→寄存器→CUDA 核心” 的存储 / 计算层级，通过 “数据逐级搬运 + 高频复用” 突破内存带宽瓶颈；
> 2. **线程块瓷砖是分层计算的 “核心中间层”**：作为连接 “全局内存（低速）” 和 “warp / 线程计算（高速）” 的桥梁，线程块瓷砖的设计直接决定了数据复用效率和并行计算粒度。

## Thread Block Tile

Each thread block computes its part of the output GEMM by iteratively loading blocks of matrix data from the input matrices and computing an accumulated matrix product (**C** += **A** * **B**). Figure 2 shows the computation performed by a single thread block and highlights the blocks of data used in one iteration of its main loop.

> GEMM 的输出矩阵 C（M×N）会被均匀拆分为多个**不重叠的 “线程块瓷砖”**，每个线程块（Thread Block）被分配一个瓷砖，负责计算该瓷砖内所有 C 元素的值。实现任务级并行，block 间无依赖。
> 每个 block 如何计算这部分的输出：从输入 A/B 矩阵迭代加载子块到共享内存，得到这个完整的矩阵，迭代累加。

![[fig-03-gemm-tile-structure.png]]

> Figure 2. A GEMM problem decomposed into the computation performed by a single thread block. The submatrix of C shown in green is computed by the matrix product of a tile of A and a submatrix of B. This is performed by looping over the K dimension, partitioned into tiles, and accumulating the results of matrix products of each tile.

The CUDA thread block tile structure is further partitioned into warps (groups of threads that execute together in SIMT fashion).

> "Warps provide a helpful organization for the GEMM computation and are an explicit part of the WMMA API, as we shall discuss shortly."
> 这里分层后可以有不同的执行方式，现在不建议使用 WMMA。

Figure 3 shows a detailed view of the structure of one block-level matrix product. Tiles of **A** and **B** are loaded from global memory and stored into shared memory accessible by all warps. The thread block’s output tile is spatially partitioned across warps as Figure 3 shows. We refer to storage for this output tile as _accumulators_ because it stores the result of accumulated matrix products. Each accumulator is updated once per math operation, so it needs to reside in the fastest memory in the SM: the register file.

> Warp - **指令同步执行**：Warp 内的 32 个线程**执行完全相同的指令**（比如同时加载数据、同时做乘累加），但操作各自的数据 —— 这种 “同指令、异数据” 的特性，让 Warp 成为 GPU 最高效的 “并行计算最小单元”；

C tile 进一步拆分为 warp tile：

- **“空间划分” 的含义**：C 线程块瓷砖（如 128×128）会被均匀拆分为**不重叠的 2D Warp 瓷砖**，每个 Warp 负责计算一个 Warp 瓷砖 —— 结合文本例子：256 线程的线程块拆分为 8 个 Warp，128×128 的 C 瓷砖可拆分为 8 个 32×64 的 Warp 瓷砖（8×32×64=128×128），每个 Warp 专门处理自己的 32×64 瓷砖；
- **为什么要拆分？**：Warp 是 GPU 的 “最小执行单元”，若让多个 Warp 处理同一块 C 瓷砖，会出现 “数据竞争”（多个线程写同一个 C 元素）；而 “空间划分” 让每个 Warp 负责独立的区域，既避免竞争，又能让 8 个 Warp 完全并行计算，最大化线程块的并行效率。

> 输出瓦片的存储称为累加器，因为它存储累积矩阵乘积的结果。每个累加器每进行一次数学运算就会更新一次，因此它需要位于流式多处理器（SM）中最快的内存——寄存器文件中。

![[fig-04-cta-structure.png]]

> Figure 3. The thread block structure partitions the tile of C across several warps, with each warp storing a non-overlapping 2D tile. Each warp stores its accumulator elements in registers. Tiles of A and B are stored in shared memory accessible to all of the warps in the thread block.

The parameters _Block__Items{X,Y,K}_ are compile-time constants that the programmer specifies to tune the GEMM computation for the target processor and the aspect ratio of the specific GEMM configuration (e.g. _M_, _N_, _K_, data type, etc.). In the figure, we illustrate an eight-warp, 256-thread thread block which is typical for the large SGEMM (FP32 GEMM) tile size implemented in CUTLASS.

## Warp Tile

Once data is stored in shared memory, each warp computes a sequence of accumulated matrix products by iterating over the _K_ dimension of the thread block tile, loading submatrices (or _fragments_) from shared memory, and computing an accumulated outer product. Figure 4 shows a detailed view. The sizes of the fragments are typically very small in the _K_ dimension to maximize the compute intensity relative to the amount of data loaded from shared memory, thereby avoiding shared memory bandwidth as a bottleneck.

> **线程块瓷砖（Thread Block Tile）在 K 维度上的进一步拆分**—— 每个 Warp 负责处理 Thread Block Tile 中的一个 “小瓷砖”，这个小瓷砖就是 Warp Tile；其数据需从共享内存加载到 Warp 内线程的寄存器，再通过外积累加完成计算。
> 为避免共享内存带宽瓶颈，Warp 不会一次性加载 Thread Block Tile 中 K 维度的所有数据（如 64 个元素），而是拆分为更小的 “片段（fragment）”（如 K 维度仅 8 个元素），即 Warp Tile 的 K 维度大小远小于 Thread Block Tile 的 K 维度；
> 共享内存中的 Warp Tile 片段 → 加载到 Warp 内线程的寄存器 → 线程执行外积计算 → 结果累加到 C 的 Warp Tile 中。

![[warp-tile-structure.png]]

> Figure 4. An individual warp computes an accumulated matrix product by iteratively loading fragments of A and B from the corresponding shared memory (SMEM) tiles into registers (RF) and computing an outer product.

Figure 4 also depicts data sharing from shared memory among several warps. Warps in the same row of the thread block load the same fragments of **A**, and warps in the same column load the same fragments of **B**.

> **行方向 Warp 共享 A 片段**；**列方向 Warp 共享 B 片段**。

We note that the warp-centric organization of the GEMM structure is effective in implementing an efficient GEMM kernel but does **not** rely on implicit warp-synchronous execution for synchronization. CUTLASS GEMM kernels are well-synchronized with calls to `__syncthreads()` as appropriate.

> **Warp 内的 “隐式同步”**：Warp 内的 32 个线程执行指令时天然同步；**线程块内的 “显式同步”**：不同 Warp 之间（如同一行 / 列的 Warp）共享共享内存数据时，需要通过 `__syncthreads()` 显式同步。

| CUTLASS 分块层级                | 对应 GPU 硬件 / 编程模型组件                     | 核心作用：数据存储与复用                                             |
| --------------------------- | -------------------------------------- | -------------------------------------------------------- |
| 1. 线程块瓷砖（Thread Block Tile） | 共享内存（Shared Memory）+ 线程块（Thread Block） | 从全局内存加载 “大瓷砖” 到共享内存，供整个线程块（含多个 Warp）复用，解决 “全局内存访问瓶颈”；    |
| 2. Warp 瓷砖（Warp Tile）       | 寄存器（Register File）+ Warp（32 个线程）       | 从共享内存加载 “中瓷砖” 到 Warp 内线程的寄存器，供 32 个线程协同计算，解决 “共享内存访问延迟”； |
| 3. 线程瓷砖（Thread Tile）        | 寄存器（Register）+ 单个 CUDA 线程              | 线程从自身寄存器中读取 “小瓷砖” 数据，在 CUDA 核心完成乘累加（MAC）计算，最大化单线程效率；     |

## Thread Tile

> 注：这一节使用 thread 来进一步划分和运算的，CUDA 也提供了 `mma` 指令用 tensor core 计算。

The CUDA Programming Model is defined in terms of thread blocks and individual threads. Consequently, the warp structure is mapped onto operations performed by individual threads. Threads cannot access each other’s registers, so we must choose an organization that enables values held in registers to be reused for multiple math instructions executed by the same thread. This leads to a 2D tiled structure within a thread as the detailed view in Figure 5 shows. Each thread issues a sequence of independent math instructions to the CUDA cores and computes an accumulated outer product.

![[fig-06-warp-tile-structure.png]]

> Figure 5. An individual thread (right) participates in a warp-level matrix product (left) by computing an outer product of a fragment of A and a fragment of B held in registers. The warp’s accumulators in green are partitioned among the threads within the warp and typically arranged as a set of 2D tiles.

In Figure 5, the upper left quadrant of the warp is shaded in grey. The 32 cells correspond to the 32 threads within a warp. This arrangement leads to multiple threads within the same row or the same column fetching the same elements of the **A** and **B** fragments, respectively. To maximize compute intensity, this basic structure can be replicated to form the full warp-level accumulator tile, yielding an 8-by-8 overall thread tile computed from an outer product of 8-by-1 and 1-by-8 fragments. This is illustrated by the four accumulator tiles shown in green.

> 2D warp tile 继续划分适配寄存器复用与计算并行。A/B 片段加载后重复计算 thread tile 内的累加结果，如右图 16 个数据加载计算了 64 次乘法。

## MMA

在 warp tile 可以使用 CUDA 9 提出的 WMMA，WMMA 要求矩阵的大小固定是 `16x16x16`，`cutlass/gemm/block_task_wmma.h` 给出了 cutlass 的实现，不过 WMMA 现在已不推荐使用。

下面是使用 WMMA 时的分层，也可以使用 MMA。

![[wmma-warp-tile-structure.png]]

## Efficient GEMM in CUDA

最后对上面讲到的分层设计做一个总结，参考 [cutlass 文档介绍](https://docs.nvidia.com/cutlass/media/docs/cpp/efficient_gemm.html)，这里的介绍和 [[#CUTLASS GEMM Model]] 的内容类似。

The basic triple loop nest computing matrix multiply may be blocked and tiled to match concurrency in hardware, memory locality, and parallel programming models. In CUTLASS, GEMM is mapped to NVIDIA GPUs with the structure illustrated by the following loop nest.

For brevity, address and index calculations are omitted here but are explained in the CUTLASS source code.

```c
for (int cta_n = 0; cta_n < GemmN; cta_n += CtaTileN) {                     // for each threadblock_y           } threadblock-level concurrency
  for (int cta_m = 0; cta_m < GemmM; cta_m += CtaTileM) {                   //    for each threadblock_x        }

    for (int cta_k = 0; cta_k < GemmK; cta_k += CtaTileK) {                 //       "GEMM mainloop" - no unrolling
                                                                            //                       - one iteration of this loop is one "stage"
                                                                            //
      for (int warp_n = 0; warp_n < CtaTileN; warp_n += WarpTileN) {        // for each warp_y                  } warp-level parallelism
        for (int warp_m = 0; warp_m < CtaTileM; warp_m += WarpTileM) {      //    for each warp_x               }
                                                                            //
          for (int warp_k = 0; warp_k < CtaTileK; warp_k += WarpTileK) {    //       fully unroll across CtaTileK
                                                                            //         - one iteration of this loop is one "k Group"
                                                                            //
            for (int mma_k = 0; mma_k < WarpTileK; mma_k += MmaK) {         // for each mma instruction         } instruction-level parallelism
              for (int mma_n = 0; mma_n < WarpTileN; mma_n += MmaN) {       //    for each mma instruction      }
                for (int mma_m = 0; mma_m < WarpTileM; mma_m += MmaM) {     //        for each mma instruction  }
                                                                            //
                  mma_instruction(d, a, b, c);                              //            TensorCore matrix computation

                }   // for mma_m
              }   // for mma_n
            }   // for mma_k

          }   // for warp_k
        }   // for warp_m
      }   // for warp_n

    }   // for cta_k
  }   // for cta_m
}   // for cta_n
```

All loops except the outermost “main” loop have constant iteration counts and can be fully unrolled by the compiler.

This tiled loop nest targets concurrency among

- threadblocks,
- warps, and
- CUDA and Tensor Cores.

It takes advantage of memory locality within

- shared memory and
- registers.

   CUTLASS 将 GEMM 计算映射到 GPU 时，采用了七级嵌套循环结构，从外层到内层依次为：

   - 线程块级（cta_n、cta_m、cta_k）：处理线程块级并发，其中 cta_k 是 "GEMM 主循环 "
   - Warp 级（warp_n、warp_m、warp_k）：处理 Warp 级并行，warp_k 循环会被完全展开
   - 矩阵乘加指令级（mma_k、mma_n、mma_m）：对应 Tensor Core 的指令级并行

The figure below illustrates the flow of data within this structure. This is the hierarchical GEMM computation embodied by CUTLASS. Each stage depicts a nested level of tiling which corresponds to a layer of concurrency within the CUDA execution model and to a level within the memory hierarchy, becoming increasingly finer moving left to right.

![[file-20251006170311038.png]]

### Threadblock-level GEMM

Each threadblock computes its portion of the output GEMM by iteratively loading tiles of input matrices and computing an accumulated matrix product.

> 每个线程块通过迭代加载输入矩阵的分片（tiles）并计算累加的矩阵乘积，来完成其负责的那部分输出计算，即上文说到的外积累加。

At the threadblock level, data are loaded from global memory. The blocking strategy in general is key to achieving efficiency. However, the programmer must balance multiple conflicting goals. A larger threadblock means fewer fetches from global memory, thereby ensuring that DRAM bandwidth does not become a bottleneck. However, large threadblock tiles may not match the dimensions of the problem well. If either the GEMM _M_ or _N_ dimension is small, some threads within the threadblock may not perform meaningful work, as the threadblock may be partially outside the bounds of the problem. If both _M_ and _N_ are small while _K_ is large, this scheme may launch relatively few threadblocks and fail to make full use of all multiprocessors within the GPU. Strategies to optimize performance for this case, as described in the section [Parallelized Reductions](https://docs.nvidia.com/cutlass/media/docs/cpp/efficient_gemm.html#parallelized-reductions), partition the GEMM K dimension across multiple threadblocks or multiple warps. These threadblocks or warps compute matrix products in parallel; the products are then reduced to compute the result.

> 需要权衡 block 的大小，后文也有一种并行规约的方法

In CUTLASS, the dimensions of the threadblock tile are specified as `ThreadblockShape::{kM, kN, kK}` and may be tuned to specialize the GEMM computation for the target processor and dimensions of the GEMM problem.

### Warp-level GEMM

The warp-level GEMM maps to the warp-level parallelism within the CUDA execution model. Multiple warps within a threadblock fetch data from shared memory into registers and perform computations. Warp-level GEMMs may be implemented either by TensorCores issuing [mma.sync](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma) or [wmma](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-wmma-mma) instructions, or by thread-level matrix computations issued to CUDA cores. For maximum performance, access to shared memory should be bank conflict free. To maximize data reuse within the warp, a large warp-level GEMM tile should be chosen.

- 多个 warp 会先从**共享内存（shared memory，CUDA 中线程块内共享的高速缓存）** 读取数据，加载到各自线程的**寄存器（register，线程私有的最快存储）** ，再执行矩阵乘法运算，这是 CUDA 中高效计算的典型数据流向。可以用 Tensor Core 也可以用 CUDA core 计算。

为让 warp 级 GEMM 达到最高性能，需关注：1. bank conflict；2. 大 warp tile，增加数据使用，即一次加载完成更多计算，增加算数密度。

### Thread-level GEMM

At the lowest level of blocking, each thread is responsible for processing a certain number of elements. Threads cannot access each other’s registers, so we choose an organization that enables reuse of values held in registers for multiple math instructions. This results in a 2D tiled structure within a thread, in which each thread issues a sequence of independent math instructions to the CUDA cores and computes an accumulated outer product.

SGEMM, IGEMM, HGEMM, and DGEMM are computed by SIMT math instructions issued by thread-level matrix multiply procedures.

- 每个线程负责处理固定数量的矩阵元素，由于线程间无法访问彼此的寄存器（寄存器是线程私有的高速存储），设计时需优先“复用寄存器中的数据”——让同一批寄存器值支撑多条数学指令，避免频繁从内存读取数据，这是提升效率的核心逻辑。
- 为实现寄存器复用，每个线程会采用“2D 分块”结构：线程不会零散处理单个元素，而是针对一小块矩阵数据，连续发出**独立的数学指令**给 CUDA 核心，最终计算出一个“累积的外积（accumulated outer product）”——外积是矩阵乘法的基础运算单元，累积则是逐步叠加结果以完成最终矩阵相乘。
- SGEMM（单精度浮点矩阵乘法，float）、IGEMM（整数矩阵乘法，int）、HGEMM（半精度浮点，half）、DGEMM（双精度浮点，double），这些不同数据精度的矩阵乘法，本质都是由“线程级矩阵乘法程序”发出的**SIMT 数学指令**完成的。

## Epilogue

The above code focuses only on the matrix multiply computation **C = AB** whose result is held in the registers of each thread within the threadblock. The mapping of logical elements in the output tile to each thread is chosen to maximize performance of the matrix multiply computation but does not result in efficient, coalesced loads and stores to global memory.

上面的代码聚焦矩阵乘法计算阶段的特点与局限

- **核心任务**：聚焦于矩阵乘法运算 `C = AB`，且计算结果仅暂存在线程块（threadblock）内每个线程的**寄存器**中（寄存器是 GPU 中速度最快的存储单元，适合暂存即时计算数据）。
- **设计侧重**：为了最大化矩阵乘法的计算性能，采用了特定的“输出块（output tile，即分块计算的矩阵子块）逻辑元素→线程”映射方式——这种映射能让线程高效并行执行乘法累加（GEMM 核心操作），但该阶段不支持对**全局内存（global memory，GPU 中容量大但访问速度较慢的主存）** 的“合并访问（coalesced loads/stores）”。

The epilogue is a separate phase in which threads exchange data through shared memory then cooperatively access global memory using efficient striped access patterns. It is also the phase in which linear scaling and other elementwise operations may be conveniently computed using the matrix product results as inputs.

下面还有收尾阶段（Epilogue）作为独立于矩阵乘法的后续阶段，专门解决计算阶段的内存访问问题，并扩展额外操作。

- **核心作用（两大点）**：
  1. **优化全局内存访问**：线程先通过**共享内存（shared memory，线程块内线程共享的高速缓存）** 交换数据，再以“条带化访问模式（striped access patterns）”协同访问全局内存——这种模式能实现高效的合并访问，弥补计算阶段的内存效率缺陷。
  2. **支持元素级扩展操作**：可利用矩阵乘法的结果（即寄存器中暂存的 `C`）作为输入，便捷执行“线性缩放（如 `C = alpha*C + beta`，GEMM 常见需求）”“元素级运算（如逐元素加减乘除）”等操作，无需额外启动新的计算核函数。

CUTLASS defines several typical epilogue operations such as linear scaling and clamping, but other device-side function call operators may be used to perform custom operations.

 同时这一阶段还有一定的**灵活性**：CUTLASS 已内置多种典型的 Epilogue 操作（如线性缩放、数值裁剪（clamping）），同时支持用户通过“设备端函数调用算子（device-side function call operators）”自定义操作，满足个性化需求。

## Optimizations

The hierarchical structure described above yields an efficient mapping to the CUDA execution model and CUDA/TensorCores in NVIDIA GPUs. The following sections describe strategies for obtaining peak performance for all corners of the design space, maximizing parallelism and exploiting data locality wherever possible.

### Pipelining

The blocked structure demands a large storage allocation within the registers of each CUDA thread. The accumulator elements typically occupy at least half a thread’s total register budget. Consequently, occupancy – the number of concurrent threads, warps, and threadblocks – is relatively low compared to other classes of GPU workloads. This limits the GPU’s ability to hide memory latency and other stalls by context switching to other concurrent threads within an SM.

这种分块结构要求在每个 CUDA 线程的寄存器中分配大量存储空间。累加器元素通常至少占据线程总寄存器预算的一半。因此，与其他类型的 GPU 工作负载相比，占用率（即并发线程、 warp 和线程块的数量）相对较低。这限制了 GPU 通过上下文切换到 SM 内的其他并发线程来隐藏内存延迟和其他停顿的能力。

To mitigate the effects of memory latency, CUTLASS uses _software pipelining_ to overlap memory accesses with other computation within a thread. CUTLASS accomplishes this by double buffering at the following scopes.

- **Threadblock-scoped shared memory tiles:** two tiles are allocated in shared memory. One is used to **load data for the current matrix operation**, while the other tile is used to **buffer data loaded from global memory** for the next mainloop iteration.
- **Warp-scoped matrix fragments:** two fragments are allocated within registers. One fragment is passed to CUDA and TensorCores during the current matrix computation, while the other is used to receive shared memory fetch returns for the next warp-level matrix operation.

The following diagram illustrates the efficient, pipelined mainloop body used in CUTLASS GEMMs.

![[file-20251006171708220.png]]

### Threadblock Rasterization

To maximize reuse of data held in the last level cache, CUTLASS defines several functions to affect **the mapping of threadblocks to logical partitions of the GEMM problem**. These map consecutively launched threadblocks to packed two-dimensional regions of the partitioned GEMM problem to increase the probability that these will access the same tiles of global memory at approximately the same time.

Several functions are defined in [cutlass/gemm/threadblock_swizzle.h](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/threadblock/threadblock_swizzle.h).

### Parallelized Reductions

**Split K - reduction across threadblocks**

Matrix product computations expose parallelism among _O(MN)_ independent inner product computations. For sufficiently large problem sizes, a GEMM kernel in CUTLASS may approach the theoretical maximum computational throughput. For small problems, however, there are too few threadblocks to efficiently occupy the entire GPU.

As a recourse, parallelizing the reduction performed during the inner product computation enables more threadblocks to execute concurrently while still taking advantage of the throughput benefits of large threadblock-level GEMM tiles.

CUTLASS implements parallel reductions across threadblocks by partitioning the GEMM _K_ dimension and launching an additional set of threadblocks for each partition. Consequently, we refer to this strategy within CUTLASS as “parallel reduction splitK.” The “parallel reduction splitK” strategy requires the execution of 2 kernels: **partitionedK GEMM, and batched reduction**.

PartitionedK GEMM resembles one flavor of batched strided GEMM. Instead of requiring users to specify the problem size of each batch, partitionedK GEMM asks for the overall problem size and the number of partitions that will be applied along the K dimension for operands A and B. For example, parameters of m=128, n=128, k=4096 and partition=16 will result in 16 batched strided GEMMs with each batch of m=128, n=128, k=256. PartitionedK also allows scenario where k is not divisible by the partition count.

For example, parameters of m=128, n=128, k=4096 and partition=20 will result in 20 batched strided GEMMs. The first 19 batches will have m=128, n=128, and k=4096/20=204, and the last batch will have m=128, n=128, and k=220.

The batched reduction kernel takes as input the output (C) of partitionedK GEMM, and performs a reduction along the K-dimension. Users must manage workspace memory to store this intermediate result.

**Sliced K - reduction across warps**

Similar to the split-k scenario, sliced-k aims at improving the efficiency of kernels with smaller M and N dimensions, but large K dimension. At the thread-block level, the parameters CtaTileN and CtaTileM expose parallelism by partitioning the work among warps. Larger warpTiles expose better instruction-level parallelism (ILP) and reuse, but also limit the number of warps running per threadblock, which reduces efficiency.

In order to improve efficiency in such scenarios, partitioning the warpTiles also along ctaTileK helps use the hardware more efficiently by allowing more warps to run concurrently in a CTA. Sliced-k kernels break down a threadblock’s computation among participating warps not just among the CtaTileN, CtaTileM dimension, but also the CtaTileK dimension. Thus, sliced-k entails a small cost in form of a reduction which has to happen at the end among the participating warps. This is because each warp computes using only a “slice” of CtaTileK, so each warp only has a partial sum before the reduction.

### Hopper Warp Specialization

https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#spatial-partitioning-also-known-as-warp-specialization

https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp

https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp

# 动手优化 GEMM

> 参考： https://siboehm.com/articles/22/CUDA-MMM
> 这篇文章的内容主要还是传统的 GEMM 优化方法，但是注重性能分析，到最后结合 warp tile 时，就可以和现在的 tensor core 对接起来了。

# Cutlass 使用

> 参考官方仓库 `cutlass/examples/08_turing_tensorop_gemm`。

# CuTe an Introduction

> 本章来源：2025 nv blog https://developer.nvidia.com/blog/cutlass-principled-abstractions-for-handling-multidimensional-data-through-tensors-and-spatial-microkernels/ ，即 2023 年提出的 CuTe 这一级抽象。

## 概述

- CUTLASS 3.x introduces **CuTe**, a library that **simplifies thread-data organization by representing tensors of threads and data using a hierarchical layout representation**, enabling developers to write high-performance CUDA code.
- CuTe's **layout** algebra allows users to build complicated layouts from simple known layouts or partition one layout across another, eliminating the need for hand-implemented complicated post-partitioned iteration schemes and supporting features like WGMMA on NVIDIA Hopper H100 and UMMA on NVIDIA Blackwell B200.
- CuTe provides **a unified interface for dense linear algebra** on modern NVIDIA GPUs, abstracting away low-level details of tensor layout and thread mapping, and is used in CUTLASS 3.x to simplify the programming model and improve performance on NVIDIA GPUs.

It is now entering the next phase of development with a new Python interface. The fundamental abstractions introduced with the CUTLASS 3.x redesign are exposed directly in Python with CUTLASS 4.0. In this post, we discussed the design principles underlying CUTLASS 3.x, its core backend library, **CUDA Tensors and Spatial Microkernels (CuTe)**, and optimization examples leveraging CuTe’s key features.

CUTLASS 3 introduced CuTe, a new library premised on the layout concept as a uniform and composable abstraction for describing and manipulating threads and data. By elevating layouts to a first-class citizen of the programming model, usage of CuTe greatly simplifies thread-data organization. CuTe reveals indexing logic to developers in an understandable and statically checkable way, while retaining the same high level of performance and Tensor Core operation coverage as in CUTLASS 2.x.

Beyond this more meaningful approach to layouts, CUTLASS 3 shares the same goals as all prior versions of CUTLASS — to help CUDA developers author high-performance GPU linear algebra kernels by developing an intuitive programming model around the latest hardware features. With this new major iteration, we emphasized the following:

- The ability to customize any layer in the design of the library while preserving composability with other layers for developer productivity and cleaner separation of moving parts
- **Compile-time checks** to ensure the correctness of kernel constructions. This guarantees that _if it compiles, it will run correctly,_ with actionable static assert messages otherwise.
- Reduce API surface area with fewer named types and a flatter learning curve with single points of entry that are also customization hooks.
- Great performance on NVIDIA Hopper H100 and NVIDIA Blackwell B200 to use features such as WGMMA (for Hopper) or UMMA (for Blackwell), Tensor Memory Accelerator for Hopper ([TMA](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)), and threadblock clusters.

> [!tip] CuTe Core
> At the heart of CUTLASS 3.x is [CuTe](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cpp/cute), a new library to describe and manipulate tensors of threads and data.
> CuTe is made of two parts: a powerful layout representation and an algebra of operations acting on those layouts.

### 特点

CuTe’s layout representation is **natively hierarchical, naturally supports static and dynamic information, and is used to represent multidimensional tensors**. The same layout representation is used to describe tensors of data and tensors of threads. Using the same vocabulary type across multiple independent resources shows the broad applicability of the CuTe Layout concept. 

Building on this representational power, CuTe provides **a formalized algebra of layouts that enable users to build complicated layouts** from simple known layouts or to partition one layout across another layout. This lets programmers focus on the logical descriptions of their algorithms while CuTe does the mechanical bookkeeping for them. With these tools, users can quickly design, implement, and modify dense linear algebra algorithms.

Unlike any previous GPU programming model, the functional composition of threads and data tensors eliminates one of the most complex hurdles in GPU programming, which is that of consistently mapping a large set of threads to the data they operate upon. Once thread layouts have been described independently of the layouts of data they’ll be operating on, CuTe’s layout algebra can partition data across threads instead of having to hand implement complicated post-partitioned iteration schemes.

> [!tip] Layout
> Layout：层次的、支持动态和静态地表示多维张量，可以表示数据和现成的张量。基于此用形式代数表示更丰富的 layout。
> 线程与数据张量的函数组合消除了将大量线程一致地映射到它们所操作的数据上这一复杂操作。

## Layouts and Tensors

### 定义

More CuTe documentation on layouts and tensors can be found in its [dedicated documentation directory](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/00_quickstart.md).

CuTe provides `Layout` and `Tensor` objects that compactly package the type, shape, memory space, and layout of data, while performing the complicated indexing for the user.

- `Layout<Shape,Stride>` provides a map between logical coordinates within `Shape` and indices computed with `Stride`. (See Figure 1 as an example)
    - `Shape` defines one or more coordinate spaces and maps between them.
    - `Stride` defines the index map that converts coordinates to indices.
- `Tensor<Engine,Layout>` provides the composition of a `Layout` with an iterator. The iterator may be a pointer to data in global memory, shared memory, register memory, or anything else that provides random access offset and dereference.

![[Multiple-matrix-types-png.webp]]

> _Figure 1. Multiple matrix types that can be manipulated by `Shape` and `Stride` functions to create indexes_

It’s worth highlighting that layouts in CuTe are hierarchical and inspired by folding tensor operations in tensor algebra. As shown in the figure, the hierarchical Shape and Stride enable representations of layouts that go far beyond simple row-major and column-major. At the same time, hierarchical layouts can still be accessed just like a normal tensor (e.g., the logical 2-D coordinate shown), so these more advanced data layouts are abstracted over in algorithmic development.

CUTLASS 3.x uses a single vocabulary type (`cute::Layout`), resulting in a simplified, formalized, and uniform layout representation to help users write extremely fast kernels with great ease.

> [!note] Layout and Tensor
> - Layout 建立了**索引**和**逻辑坐标**之间的关系。注意它是层次化的，索引可以提供更为复杂的数据排布方式。`Layout` 确定了坐标怎么映射到索引
> - Tensor 则由一个指针（确定它的地址）和 Layout 确定。`Tensor` 确定了索引怎么找到实际数据。
> 重点：CuTe 通过 `Layout<Shape, Stride>` 定义“逻辑坐标怎么转成索引”，再通过 `Tensor<Engine, Layout>` 把“索引和实际数据存储”绑定，最终实现了“用逻辑坐标访问数据，底层自动处理复杂索引计算和内存适配”。

### Layout<Shape, Stride>：“坐标→索引”的映射规则（核心是“怎么算位置”）

`Layout` 是 CuTe 的“坐标翻译器”，它不直接存储数据，只定义**“如何把‘逻辑坐标’（比如矩阵的行号列号）转换成‘内存索引’（比如从数据起始地址跳过多少个元素）”**。其两个模板参数 `Shape` 和 `Stride` 分别负责“定义坐标范围”和“计算索引步长”，二者配合完成映射。

#### （1）Shape：定义“坐标的规则与范围”

`Shape` 的核心作用是**划定逻辑坐标的“可用空间”，并定义坐标之间的层级/转换关系**，而非简单的“行数×列数”。

- 比如一个“4 行 8 列”的矩阵，基础 `Shape` 是 `Shape<4,8>`，它定义了逻辑坐标是二维的 `(m,n)`（m∈0-3，n∈0-7），即“第一个坐标对应行，第二个对应列”；
- 但 `Shape` 支持“层级划分”（这是后续“高级布局”的基础）：比如把 `Shape<4,8>` 拆成 `Shape<Shape<2,2>, Shape<4,2>>`，此时逻辑坐标变成 `((m1,m2), (n1,n2))`，对应“先按 2×2 分组行，再按 4×2 分组列”——这种层级划分正是 CuTe 支持复杂布局的关键。

简单说：`Shape` 回答了“你能用什么样的坐标（比如是 1D/2D/层级坐标）去定位数据”。

#### （2）Stride：定义“从坐标到索引的计算方法”

`Stride` 是“索引计算的步长表”，核心作用是**把逻辑坐标的每个维度，转换成“在内存中需要跳过的元素个数”**，最终算出“从数据起始位置到目标元素的总偏移（即索引）”。

计算逻辑很直接：假设逻辑坐标是 `(c0, c1, ..., cn-1)`，对应的 `Stride` 是 `(s0, s1, ..., sn-1)`，则最终内存索引 = `c0×s0 + c1×s1 + ... + cn-1×sn-1`。

举个直观例子，假设我们有一个 2×3 的矩阵（`Shape<2,3>`），要计算坐标 `(1,2)` 的索引：

| 0   | 2   | 4   |
| --- | --- | --- |
| 1   | 3   | 5   |

- 若 `Stride<3,1>`（行优先布局）：索引 = 1×3 + 2×1 = 5（内存中按“行 1 所有元素→行 2 所有元素”存储，行内每个元素隔 1 个位置）；
- 若 `Stride<1,2>`（列优先布局）：索引 = 1×1 + 2×2 = 5（此时内存中元素排列是 `(0,0)→(1,0)→(0,1)→(1,1)→(0,2)→(1,2)`，行维度步长 1，列维度步长 2）。

简单说：`Stride` 回答了“每个坐标维度，在内存中对应多少个元素的偏移”。

#### （3）Layout 的本质：Shape + Stride = 完整的“坐标→索引”映射

把 `Shape` 和 `Stride` 结合，就得到了 `Layout`：

比如 `Layout<Shape<2,3>, Stride<3,1>>`，它完整定义了“用二维坐标 `(m,n)`（m=0-1，n=0-2），通过 `m×3 + n×1` 的公式，计算出内存索引”的规则——不同 Shape/Stride 可通过这两个函数生成索引。

>[!tip] Layouts are functions from integers to integers.

### Tensor<Engine, Layout>：“索引→实际数据”的访问载体（核心是“找到数据”）

`Tensor` 是 CuTe 的“数据访问器”，它把 `Layout`（坐标→索引）和 `Engine`（迭代器）结合，实现“从逻辑坐标直接拿到物理数据”的完整链路——使用时不用关心“索引怎么算、数据存在哪里”，只需用逻辑坐标访问即可。

#### （1）Engine：数据的“存储位置与访问接口”

`Engine` 本质是“数据的迭代器（iterator）”，它负责两件事：

- 明确数据的**存储位置**：可以是 GPU 的全局内存（global memory）、共享内存（shared memory）、寄存器（register memory），也可以是 CPU 内存（只要支持“按偏移访问 + 解引用”）；
- 提供**索引→数据的访问能力**：知道“从数据起始地址开始，偏移 `k` 个索引后，如何读取/写入对应元素”（比如指针 `ptr`，偏移 `k` 就是 `*(ptr + k)`）。

#### （2）Tensor 的本质：Layout + Engine = 完整的“坐标→数据”访问

`Tensor` 的工作流程可以拆解为 3 步：

1. 用户输入逻辑坐标（比如 `(1,2)`）；
2. 内部调用 `Layout`，将坐标转换成内存索引（比如通过 `Stride` 算出索引=5）；
3. 内部调用 `Engine`，按索引访问实际数据（比如通过指针 `ptr`，读取 `*(ptr + 5)`）。

这 3 步是完全封装好的——只需要写 `tensor(1,2)`，就能拿到对应数据，不用手动计算索引、不用关心数据存在 GPU 的哪个内存区域。

### 层级化

“CuTe 的布局是层级化的（hierarchical），灵感来自张量代数的折叠操作”，这是 CuTe 区别于普通矩阵库的核心优势，需要重点理解：

“层级化”指 `Shape` 和 `Stride` 都支持“嵌套定义”，可以把一个大的坐标空间拆成多个小的子空间，形成“父坐标→子坐标”的层级关系。  比如一个 `8×8` 的矩阵，普通布局的 `Shape` 是 `Shape<8,8>`（二维坐标 `(m,n)`），而层级化 `Shape` 可以是 `Shape<Shape<2,4>, Shape<4,2>>`（四维坐标 `((m1,m2), (n1,n2))`），对应“先把行分成 2 组×4 行，列分成 4 组×2 列”的逻辑。  此时 `Stride` 也会对应层级化（比如 `Stride<Shape<32,8>, Shape<4,1>>`），索引计算会先算“父坐标的偏移”，再算“子坐标的偏移”，最终得到总索引。

普通矩阵库只支持“行优先（如 C 语言数组）”或“列优先（如 Fortran 数组）”，而 CuTe 的层级化布局可以实现更灵活的结构：

- 比如“块矩阵（Block Matrix）”：把 `16×16` 矩阵拆成 `4×4` 的块，每个块内部是 `4×4` 的子矩阵，此时用层级化 `Shape<Shape<4,4>, Shape<4,4>>` 和对应的 `Stride`，就能直接用“块坐标 + 块内坐标”访问，无需手动计算块的起始索引；
- 再比如“交错布局（Interleaved Layout）”：常用于 GPU 优化（如纹理内存访问），通过层级化 `Stride` 让相邻线程访问的内存地址更连续，提升带宽利用率。

虽然布局是层级化的，但用户访问时完全不用关心层级——依然可以用普通的“扁平坐标”（比如 `(m,n)`）访问，CuTe 会自动处理层级间的坐标转换。  这意味着：算法开发者可以用“高级布局”优化性能（比如适配 GPU 内存特性），但写代码时依然保持“像用普通张量一样简单”，不用因为布局复杂而增加代码复杂度。

## CuTe Layouts to Transform and Partition TODO

### Layout 的功能组合

CuTe Layouts support **functional composition** as a core operation. Functional composition can be used to transform the shape and order of another layout. If we have a layout of data with coordinates (`m,n`) and we want to use coordinates (`thread_idx,value_idx`) instead, then we compose the data layout with a layout that describes the mapping (`thread_idx,value_idx`) -> (`m,n`).  The result is a layout of data with coordinates (`thread_idx,value_idx`), which we can use to access each value of each thread very easily!

As an example, consider a 4×8 layout of data. Further, suppose that we want to assign threads and values to each coordinate of that 4×8 data. We write a “TV layout” that records the particular partitioning pattern, then perform a functional composition between the data layout and the TV layout.

As shown, the composition permutes and reshapes the data such that each thread’s values are arranged across each row of the result. Simply slicing the result with our thread index completes the partitioning.

![[4x8-layout--png.webp]]

> _Figure 3. An example of how a 4×8 layout of data can be assigned a thread and value pair to help coordinate access to the 4×8 data. This is known as a “TV layout”_

A more intuitive view of the partitioning pattern is the inverse of the TV layout.

![[4x8-matrix-png.webp]]

> _Figure 4. Another 4×8 matrix representing how the original data can be mapped, the inverse of the TV layout_

This layout shows the map from each coordinate within the 4×8 data layout to the thread and value. Arbitrary partitioning patterns can be recorded and applied to arbitrary data layouts. Additional documentation on [CuTe Layout Algebra](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/02_layout_algebra.md) can be found on GitHub.

> [!note] Funcional Coposition
> CuTe Layout 核心操作支持函数组合。函数组合可用于改变另一种布局的形状和顺序。TV Layout 让每个线程能快速找到自己需要处理的数据。

### 核心逻辑

CuTe 设计 “函数组合” 作为核心操作，本质是解决 **“数据坐标与线程需求不匹配”** 的问题 ——假设原始数据用坐标 `(m,n)` 定位（比如 4×8 矩阵），但 GPU 编程中，我们需要让每个线程 `thread_idx` 处理一组数据（用 `value_idx` 区分同一线程的不同数据）。此时原始的 `(m,n)` 坐标无法直接对应 `(thread_idx, value_idx)`，就需要通过**函数组合**实现 “坐标转换”：

>[!tip] Layouts are functions from integers to integers.
>而 TV layout 是一个（线程索引，值索引）-> 数据中坐标（m, n) 的映射函数

1. 第一步：定义一个 TV Layout，其映射规则是 **(thread_idx, value_idx) → (m,n)**（告诉程序 “某个线程的某个值，对应原始数据的哪一行哪一列”）。
2. 第二步：将 “TV Layout” 与 “原始 Data Layout” 进行函数组合。
    原始 Data Layout 的规则是 **(m,n) → 数据索引**（通过 (m,n) 找数据），组合后新规则变成：
    **(thread_idx, value_idx) → (m,n) → 数据索引**
    最终等效于 **(thread_idx, value_idx) → 数据索引**。
3. 结果：我们得到了一个 “以 (thread_idx, value_idx) 为坐标” 的新 Layout。此时只需用线程自己的 thread_idx “切片”（比如固定 thread_idx=0，遍历 value_idx），就能快速拿到该线程要处理的所有数据，无需再手动转换原始 (m,n) 坐标。

下面分 “正过程” 和 “逆过程” 理解：

#### 1. 正过程：用 TV Layout 组合出 “线程友好型” Layout

原始数据是 4 行 8 列的矩阵（Data Layout：`(m,n)`→数据，`m=0-3, n=0-7`），我们希望给它分配线程和值索引，比如设计这样的 TV Layout 规则（假设需求：8 个线程，每个线程处理 4 个值）：

- TV Layout 映射：`(thread_idx, value_idx) → (m,n)`，其中 `thread_idx=0-7`，`value_idx=0-3`。
- 对于上图，线程 0 的即对应 TV Layout 第一行，线程 0 所需要的 0-3 个数据，它们的一维索引分别是 `0 4 16 20`。

将 TV Layout 与原始 Data Layout 组合后，新 Layout 的坐标变成 `(thread_idx, value_idx)`：

- 线程 0（`thread_idx=0`）对应的所有数据，就是原始 `m=0` 行的所有 `n` 值（`value_idx=0-7`），刚好排成新 Layout 的第 0 行；
- 线程 1（`thread_idx=1`）对应的所有数据，就是原始 `m=1` 行的所有 `n` 值，排成新 Layout 的第 1 行；
- 以此类推。

此时要给线程分配数据，只需 “切片” 新 Layout：比如线程 `k` 只需取 `thread_idx=k` 的所有 `value_idx` 对应的元素，一步就能完成数据划分，非常高效。

#### 2. 逆过程：用 TV Layout 的逆布局理解 “原始坐标→线程 / 值”

 “更直观的划分方式是 TV Layout 的逆布局”，这里的 “逆” 指**映射规则反过来**：

原始 TV Layout 是 “`(thread_idx, value_idx)→(m,n)`”，逆布局就是 “`(m,n)→(thread_idx, value_idx)`”。

对于 4×8 矩阵，逆布局的作用是 “标注原始数据的每个 `(m,n)` 坐标，对应哪个线程和哪个值”：

- 原始 `(m=0, n=5)` → 对应 `(thread_idx=0, value_idx=5)`（线程 0 的第 5 个值）；
- 原始 `(m=3, n=2)` → 对应 `(thread_idx=3, value_idx=2)`（线程 3 的第 2 个值）。

这种映射关系，能让我们一眼看清 “原始数据如何分配给线程”，进一步理解 TV Layout 的设计逻辑 ——**无论原始数据是何种 Shape（比如 4×8、8×4、2×16），只要定义对应的 TV Layout，就能通过函数组合将其转换为 “线程可直接访问” 的布局**。

最终目的是**降低 GPU 线程访问数据的复杂度**：让每个线程只需用自己的索引，就能快速定位到要处理的数据，无需手动处理复杂的坐标转换。

## CuTe Matrix Multiply-accumulate Atoms

An atom is the **smallest collection of threads and data** that must cooperatively participate in the execution of a hardware-accelerated math or copy operation.

CuTe MMA Atom 是 CuTe 中的**最小协同计算单元**——指完成一次硬件加速的“矩阵乘累加（MMA，Matrix Multiply-Accumulate）”或数据拷贝操作时，必须协同工作的线程组与数据的最小集合。  简单说：它是 GPU 硬件执行 MMA 计算的“最小功能块”，不能再拆分，所有参与该计算的线程和数据必须按此单元的规则配合。

An Atom combines a PTX instruction with metadata about the shape and arrangement of threads and values that must participate in that instruction. This metadata is expressed as CuTe TV layouts that can then be used to partition arbitrary tensors of input and output data. A user should in general, not have to extend this layer, as we’ll provide implementations of CuTe atoms for new architectures.

每个 CuTe MMA 原子由两部分绑定组成：

- **PTX 指令**：GPU 硬件可直接执行的底层机器指令（如文中提到的 `SM70_8x8x4_F32F16F16F32_NT`），定义了“具体做什么计算”（如 8x8x4 尺寸的矩阵乘，输入为 F16 精度、输出为 F32 精度，NT 表示矩阵布局）。
- **元数据（Metadata）**：用 CuTe 特有的“TV 布局（Tensor View Layout）”描述的规则，定义了“怎么协同做”——包括线程的分组方式、数据在线程间的分配规则（即 `(thread_id, value_id) 与数据坐标 (coord)` 的映射关系）。
这个层次的意义：
- **屏蔽硬件细节**：用户无需关心不同 GPU 架构（如 SM70、SM80）的 MMA 指令差异——框架会提供各架构的原子实现，用户只需基于原子组装高层计算逻辑，不用手动扩展或适配硬件。
- **统一数据分区**：原子的 TV 布局元数据可直接用于“分割任意输入/输出张量”，确保线程组能精准获取计算所需的数据，避免数据分配混乱。

![[MMA-Traits-png.webp]]

The above image shows the SM70_8x8x4_F32F16F16F32_NT instruction and its associated `MMA_Traits` metadata. On the left, the TV layouts mapping `(thread_id,value_id) -> coord` are recorded in the traits, and on the right, the traits are visualized with the `inverse coord -> (thread_id,value_id)` mapping. The image on the right can be generated with `print_latex(make_tiled_mma(cute::SM70_8x8x4_F32F16F16F32_NT{}))`

以 `SM70_8x8x4_F32F16F16F32_NT` 原子为例：

- **PTX 指令含义**：针对 SM70 架构（如 Volta 系列 GPU），执行“8x8x4 尺寸的 MMA”，输入矩阵 A/B 为 F16 精度，累加矩阵 C 和输出矩阵 D 为 F32 精度，“NT”表示矩阵 A 按“非转置（Non-Transposed）”布局、矩阵 B 按“转置（Transposed）”布局。
- **元数据可视化**：
  - 左侧：记录 `(thread_id, value_id) → 数据坐标 (coord)` 的映射（即“哪个线程的哪个数据，对应张量的哪个位置”）；
  - 右侧：反向展示 `coord → (thread_id, value_id)` 的映射（更直观呈现数据如何分配给线程）。

> Additional CuTe documentation on matrix multiply-accumulate ([MMA) atoms](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/0t_mma_atom.md) is on GitHub.

## CuTe Tiled MMAs

Tiled MMA and tiled copy are tilings of MMA atoms and copy atoms, respectively. We call this level “tiled” because it builds larger operations on top of the atoms as if fitting together individual tiles to build a reusable component of a mosaic. The tilings reproduce atoms across threads and data, with possible permutations and interleaving of the atoms as well.

This layer is most analogous to the warp-level tiling of MMA instructions in CUTLASS 2.x; however, it views the tiling from the perspective of all threads participating in the operation and generalizes the concept to copy operations as well. The purpose of this layer is to build composable GPU micro-kernels out of a plethora of hardware-accelerated math and data movement operations, each potentially with their own intrinsic layouts of threads and data. The tiled MMA and tiled Copy types present all these various hardware-accelerated CuTe atoms with a single, consistent API for partitioning data.

For example, CuTe might provide an MMA atom that users can call on a single warp, for fixed M, N, and K dimensions. We can then use CuTe operations `make_tiled_mma` to turn this atom into an operation that works on an entire thread block, for larger M, N, and K dimensions. We’ve already seen one example of a Tiled MMA in the previous section, the 1x1x1 tiling of `SM70_8x8x4_F32F16F16F32_NT`.

![[Tiled-MMAs-png.webp]]

> _Figure 6. The above image shows two more tiled MMAs using the same SM70_8x8x4_F32F16F16F32_NT atom_

This image shows two more tiled MMAs using the same `SM70_8x8x4_F32F16F16F32_NT` atom. On the left, four of these atoms are combined in a 2×2 row-major layout to produce a one-warp 16x16x4 MMA. On the right, four of these atoms are 2×2 row-major layouts to produce a one-warp 16x16x4 MMA, and then the rows (M) and the columns (N) are permuted to interleave the atoms. Both of these produce partitioning patterns that can be applied to any data layout, as demonstrated in the following section.

MMA Atom 是 CuTe 中**最小粒度的硬件加速矩阵乘法单元**，特点如下：

- **硬件绑定**：针对特定 GPU 架构（如 SM70，即 Volta 架构）设计，直接调用硬件原生加速指令；
- **尺寸固定**：M、N、K 维度（矩阵乘法的核心尺寸参数）固定，例如“SM70_8x8x4_F32F16F16F32_NT”代表：
  - 架构（SM70）、矩阵尺寸（M=8, N=8, K=4）；
  - 数据类型（输入 A=F32、输入 B=F16、输入 C=F16、输出 D=F32）；
  - 布局（NT，即 A 矩阵按“行优先”、B 矩阵按“列优先”存储）；
- **线程束级调用**：仅能在单个 GPU warp（线程束，通常含 32 个线程）上执行，是构建更大运算的“基本瓷砖”。

“Tiled”（分块）的本质是**将多个 MMA Atom 或 Copy Atom（基础数据拷贝单元）按规则组合，构建更大粒度、可复用的运算单元**，类似用小瓷砖拼出马赛克组件，关键设计包括：

- **跨线程/数据扩展**：将单个 warp 执行的 MMA Atom，扩展到整个 thread block（线程块，含多个 warp）执行，支持更大的 M、N、K 维度（例如 4 个 8x8x4 的 Atom 拼成 16x16x4 的运算）；
- **灵活排列**：组合时可对 Atom 进行“排列（permutation）”或“交错（interleaving）”，例如文中示例：
  - 左图：4 个 SM70_8x8x4 Atom 按 2×2 行优先排列，组成 1 个 warp 执行的 16x16x4 MMA；
  - 右图：在 2×2 排列基础上，对 M（行）、N（列）维度进行排列，实现 Atom 交错，适配不同数据布局需求；
- **统一接口**：无论底层 MMA Atom 或 Copy Atom 的硬件布局、数据类型如何，Tiled 层都提供一致的 API 来划分数据，降低上层使用复杂度。

## CuTe GEMMs and Mainloops

With the architecture agnostic tiled API, users can build a consistent interface to GEMM outer loops, with inner loops from the atom layer.

```c

Tensor gA = . . . // Tile of 64x16 gmem for A
Tensor gB = . . . // Tile of 96x16 gmem for B
Tensor gC = . . . // Tile of 64x96 gmem for C

// 64x16 static-layout padded row-major smem for A
Tensor sA = make_tensor(make_smem_ptr<TA>(smemAptr),
                        Layout<Shape <    _64,_16>,
                               Stride<Int<17>, _1>>{});
// 96x16 static-layout interleaved col-major smem for B
Tensor sB = make_tensor(make_smem_ptr<TB>(smemBptr),
                        Layout<Shape <Shape <_32,  _3>,_16>,
                               Stride<Stride< _1,_512>,_32>>{});

// Partition tensors across threads according to the TiledMMA
ThrMMA thr_mma = tiled_mma.get_slice(thread_idx);
Tensor tCsA = thr_mma.partition_A(sA);        // (MMA, MMA_M, MMA_K) smem
Tensor tCsB = thr_mma.partition_B(sB);        // (MMA, MMA_N, MMA_K) smem
Tensor tCgC = thr_mma.partition_C(gC);        // (MMA, MMA_M, MMA_N) gmem

// Make register tensors the same shape/layout as above
Tensor tCrA = thr_mma.make_fragment_A(tCsA);  // (MMA, MMA_M, MMA_K) rmem
Tensor tCrB = thr_mma.make_fragment_B(tCsB);  // (MMA, MMA_N, MMA_K) rmem
Tensor tCrC = thr_mma.make_fragment_C(tCgC);  // (MMA, MMA_M, MMA_N) rmem

// COPY from smem to rmem thread-level partitions
cute::copy(tCsA, tCrA);
cute::copy(tCsB, tCrB);
// CLEAR rmem thread-level partition (accumulators)
cute::clear(tCrC);

// GEMM on rmem: (V,M,K) x (V,N,K) => (V,M,N)
cute::gemm(tiled_mma, tCrA, tCrB, tCrC);
// Equivalent to
// for(int k = 0; k < size<2>(tCrA); ++k)
//   for(int m = 0; m < size<1>(tCrC); ++m)
//     for(int n = 0; n < size<2>(tCrC); ++n)
//       tiled_mma.call(tCrA(_,m,k), tCrB(_,n,k), tCrC(_,m,n));

// AXPBY from rmem to gmem thread-level partitions
cute::axpby(alpha, tCrC, beta, tCgC);
// Equivalent to
// for(int i = 0; i < size(tCrC); ++i)
//   tCgC(i) = alpha * tCrC(i) + beta * tCgC(i)
```

There are now many decisions to be made for the above code regarding the temporal interleaving of compute and copy instructions

- Allocate rmem as only `A: (MMA,MMA_M)` and `B: (MMA,MMA_N)` and `C: (MMA,MMA_M,MMA_N)` Tensors and copy to it on each k-block iteration.
- Account for multiple k-tiles of gmem and copy to smem on each k-tile iteration.
- Overlap the above copy stages with compute stages asynchronously.
- Optimize by finding better layouts of smem that improve the access patterns for the smem -> rmem copy.
- Optimize by finding efficient TiledCopy partitioning patterns for the gmem -> smem copy.

These concerns are considered part of the “temporal micro-kernels” rather than the “spatial micro-kernels” that CuTe provides. In general, decisions regarding the pipelining and execution of instructions over CuTe Tensors are left to the CUTLASS level and will be discussed in the next part of this series.

## **Summary**

In summary, CuTe empowers developers to write more readable, maintainable, and high-performance CUDA code by abstracting away the low-level details of tensor layout and thread mapping, and providing a unified, algebraic interface for dense linear algebra on modern NVIDIA GPUs.

# CuTe Documentation

> https://docs.nvidia.com/cutlass/media/docs/cpp/cute/00_quickstart.html

- 核心：CuTe is a collection of C++ CUDA template abstractions for defining and operating on **hierarchically multidimensional layouts of threads and data**.
- 组成：CuTe provides `Layout` and `Tensor` objects that compactly packages the type, shape, memory space, and layout of data, while performing the complicated indexing for the user.
- 优势：This lets programmers focus on the logical descriptions of their algorithms while CuTe does the mechanical bookkeeping for them. With these tools, we can quickly design, implement, and modify all dense linear algebra operations.
- 扩展：Layout: The core abstraction of CuTe are the **hierarchically multidimensional layouts** which can be composed with data arrays to represent tensors. The representation of layouts is powerful enough to represent nearly everything we need to implement efficient dense linear algebra. Layouts can also be combined and manipulated via **functional composition**, on which we build a large set of common operations such as tiling and partitioning.
- 需求：CuTe shares CUTLASS 3.x’s software requirements, including NVCC with a C++17 host compiler.

The `cute::print` function has overloads for almost all CuTe types, including Pointers, Integers, Strides, Shapes, Layouts, and Tensors. When in doubt, try calling `print` on it.

## 编译调试

The `cute::print` function has overloads for almost all CuTe types, including Pointers, Integers, Strides, Shapes, Layouts, and Tensors. When in doubt, try calling `print` on it.

> 也可以利用重载的 `cout`

CuTe’s print functions work on either host or device. Note that on device, printing is expensive.

You might also only want to print on thread 0 of each threadblock, or threadblock 0 of the grid. The `thread0()` function returns true only for global thread 0 of the kernel, that is, for thread 0 of threadblock 0. A common idiom for printing CuTe objects to print only on global thread 0.

```c
if (thread0()) {
  print(some_cute_object);
}
```

Some algorithms depend on some thread or threadblock, so you may need to print on threads or threadblocks other than zero. The header file [`cute/util/debug.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/util/debug.hpp), among other utilities, includes the function `bool thread(int tid, int bid)` that returns `true` if running on thread `tid` and threadblock `bid`.

Some CuTe types have special printing functions that use a different output format.

The `cute::print_layout` function will display any rank-2 layout in a plain text table. This is excellent for visualizing the map from coordinates to indices.

The `cute::print_tensor` function will display any rank-1, rank-2, rank-3, or rank-4 tensor in a plain text multidimensional table. The values of the tensor are printed so you can verify the tile of data is what you expect after a copy, for example.

The `cute::print_latex` function will print LaTeX commands that you can use to build a nicely formatted and colored tables via `pdflatex`. This work for `Layout`, `TiledCopy`, and `TiledMMA`, which can be very useful to get a sense of layout patterns and partitioning patterns within CuTe.

## Library Organization

CuTe is a header-only C++ library, so there is no source code that needs building. Library headers are contained within the top level [`include/cute`](https://github.com/NVIDIA/cutlass/tree/main/include/cute) directory, with components of the library grouped by directories that represent their semantics.

|Directory|Contents|
|---|---|
|[`include/cute`](https://github.com/NVIDIA/cutlass/tree/main/include/cute)|Each header in the top level corresponds to one of the fundamental building blocks of CuTe, such as [`Layout`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/layout.hpp) and [`Tensor`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/tensor.hpp).|
|[`include/cute/container`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/container)|Implementations of STL-like objects, such as tuple, array, and aligned array.|
|[`include/cute/numeric`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/numeric)|Fundamental numeric data types that include nonstandard floating-point types, nonstandard integer types, complex numbers, and integer sequence.|
|[`include/cute/algorithm`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm)|Implementations of utility algorithms such as copy, fill, and clear that automatically leverage architecture-specific features if available.|
|[`include/cute/arch`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/arch)|Wrappers for architecture-specific matrix-matrix multiply and copy instructions.|
|[`include/cute/atom`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/atom)|Meta-information for instructions in `arch` and utilities like partitioning and tiling.|

库文件组织是 CuTe 的功能实现，下面将介绍 CuTe 提供的核心功能。

## Layout

> [!note] Fundamentally, a `Layout` maps from coordinate space(s) to an index space.

`Layout`s present a common interface to multidimensional array access that abstracts away the details of how the array’s elements are organized in memory. CuTe also provides an “algebra of `Layout`s.” `Layout`s can be combined and manipulated to construct more complicated layouts and to tile layouts across other layouts.

### Integers

CuTe makes great use of **dynamic** (known only at run-time) and **static** (known at compile-time) integers.

- Dynamic integers (or “run-time integers”) are just ordinary integral types like `int` or `size_t` or `uint16_t`. Anything that is accepted by `std::is_integral<T>` is considered a dynamic integer in CuTe.
- Static integers (or “compile-time integers”) are instantiations of types like `std::integral_constant<Value>`. These types encode the value as a `static constexpr` member. They also support casting to their underlying dynamic types, so they can be used in expressions with dynamic integers. CuTe defines its own CUDA-compatibe static integer types `cute::C<Value>` along with overloaded math operators so that math on static integers results in static integers. CuTe defines shortcut aliases `Int<1>`, `Int<2>`, `Int<3>` and `_1`, `_2`, `_3` as conveniences, which you should see often within examples.

CuTe attempts to handle static and dynamic integers identically. In the examples that follow, all dynamic integers could be replaced with static integers and vice versa. When we say “integer” in CuTe, we almost always mean a static OR dynamic integer.

CuTe provides a number of traits to work with integers.

- `cute::is_integral<T>`: Checks whether `T` is a static or dynamic integer type.
- `cute::is_std_integral<T>`: Checks whether `T` is a dynamic integer type. Equivalent to `std::is_integral<T>`.
- `cute::is_static<T>`: Checks whether `T` is an empty type (so instantiations cannot depend on any dynamic information). Equivalent to `std::is_empty`.
- `cute::is_constant<N,T>`: Checks that `T` is a static integer AND its value is equivalent to `N`.

> CuTe 引入了动态整数和静态整数。
> [`integral_constant` implementations](https://github.com/NVIDIA/cutlass/tree/main/include/cute/numeric/integral_constant.hpp) 是这 Integers 实现的细节。

### Tuple

A tuple is a finite ordered list of zero or more elements. The [`cute::tuple` class](https://github.com/NVIDIA/cutlass/tree/main/include/cute/container/tuple.hpp) behaves like `std::tuple`, but works on device and host. It imposes restrictions on its template arguments and strips down the implementation for performance and simplicity.

> CuTe 提供在 host device 都支持的 tuple 类型，并做了限制以简化和性能。

### IntTuple

CuTe defines the IntTuple concept as either an integer, or a tuple of IntTuples. Note the recursive definition. In C++, we define [operations on `IntTuple`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/int_tuple.hpp).

Examples of `IntTuple`s include:

- `int{2}`, the dynamic integer 2.
- `Int<3>{}`, the static integer 3.
- `make_tuple(int{2}, Int<3>{})`, the tuple of dynamic-2, and static-3.
- `make_tuple(uint16_t{42}, make_tuple(Int<1>{}, int32_t{3}), Int<17>{})`, the tuple of dynamic-42, tuple of static-1 and dynamic-3, and static-17.

> CuTe 中定义的 IntTuple 是一个递归概念，指的是整数或由 IntTuple 组成的元组。它既可以是单个整数（包括动态整数如 int{2}、静态整数如 Int<3>{}），也可以是由这些整数或其他 IntTuple 嵌套组成的元组（如包含动态整数和静态整数的元组，或更深层次嵌套的元组）。

CuTe reuses the `IntTuple` concept for many different things, including Shape, Stride, Step, and Coord (see [`include/cute/layout.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/layout.hpp)).

Operations defined on `IntTuple`s include the following.

- `rank(IntTuple)`: The number of elements in an `IntTuple`. A single integer has rank 1, and a tuple has rank `tuple_size`.
- `get<I>(IntTuple)`: The `I`th element of the `IntTuple`, with `I < rank`. For single integers, `get<0>` is just that integer.
- `depth(IntTuple)`: The number of hierarchical `IntTuple`s. A single integer has depth 0, a tuple of integers has depth 1, a tuple that contains a tuple of integers has depth 2, etc.
- `size(IntTuple)`: The product of all elements of the `IntTuple`.

We write `IntTuple`s with parentheses to denote the hierarchy. For example, `6`, `(2)`, `(4,3)`, and `(3,(6,2),8)` are all `IntTuple`s.

`int{2}`、`uint16_t{42}` 是 C++11 引入的**列表初始化（list initialization）** 语法，属于标准 C++ 语法的一部分，并非 CuTe 特有的扩展。 这种语法的含义是： - `int{2}` 表示创建一个 `int` 类型的变量（或临时对象），并用值 `2` 初始化它； - `uint16_t{42}` 表示创建一个 `uint16_t` 类型的变量（或临时对象），并用值 `42` 初始化它。 ### 这种语法的特点： 1. **明确类型**：直接指定变量类型，同时完成初始化，语义清晰； 2. **禁止窄化转换**：相比传统的 `int(2)` 或 `(int)2`，列表初始化更严格，例如 `int{3.14}` 会编译报错（阻止浮点数到整数的窄化转换），而 `int(3.14)` 则会默默截断为 `3`； 3. **适用于临时对象**：在需要临时整数对象的场景（如函数参数）中非常方便，例如 `make_tuple(int{2}, uint16_t{42})` 明确指定了元组中元素的类型。 在 CuTe 的示例中频繁使用这种语法，主要是为了**显式强调整数的类型**（尤其是区分不同宽度的动态整数，如 `int`、`uint16_t`、`int32_t` 等），使代码的类型意图更清晰。

> `int{2}`、`uint16_t{42}` 是 C++11 引入的**列表初始化（list initialization）** 语法。`int{2}` 表示创建一个 `int` 类型的变量（或临时对象），并用值 `2` 初始化它；

### Shapes and Strides

Both `Shape` and `Stride` are `IntTuple` concepts.

### Layout

A `Layout` is a tuple of (`Shape`, `Stride`). Semantically, it implements a mapping from any coordinate within the Shape to an index via the Stride.

### Tensor

A `Layout` can be composed with data – e.g., a pointer or an array – to create a `Tensor`. The index generated by the `Layout` is used to subscript an iterator to retrieve the appropriate data.

> 后文会再介绍，For details on `Tensor`, please refer to the [`Tensor` section of the tutorial](https://docs.nvidia.com/cutlass/media/docs/cpp/cute/03_tensor.html).

## Layout Creation and Use

A `Layout` is a pair of `IntTuple`s: the `Shape` and the `Stride`. The first element defines the abstract _shape_ of the `Layout`, and the second element defines the _strides_, which map from coordinates within the shape to the index space.

We define many operations on `Layout`s analogous to those defined on `IntTuple`.

- `rank(Layout)`: The number of modes in a `Layout`. Equivalent to the tuple size of the `Layout`’s shape.
- `get<I>(Layout)`: The `I`th sub-layout of the `Layout`, with `I < rank`.
- `depth(Layout)`: The depth of the `Layout`’s shape. A single integer has depth 0, a tuple of integers has depth 1, a tuple of tuples of integers has depth 2, etc.
- `shape(Layout)`: The shape of the `Layout`.
- `stride(Layout)`: The stride of the `Layout`.
- `size(Layout)`: The size of the `Layout` function’s domain. Equivalent to `size(shape(Layout))`.
- `cosize(Layout)`: The size of the `Layout` function’s codomain (not necessarily the range). Equivalent to `A(size(A) - 1) + 1`.

### Hierarchical Access Functions

`IntTuple`s and `Layout`s can be arbitrarily nested. For convenience, we define versions of some of the above functions that take a sequence of integers, instead of just one integer. This makes it possible to access elements inside of nested `IntTuple` or `Layout` more easily. For example, we permit `get<I...>(x)`, where `I...` is a “C++ parameter pack” that denotes zero or more (integer) template arguments. These hierarchical access functions include the following.

- `get<I0,I1,...,IN>(x) := get<IN>(...(get<I1>(get<I0>(x)))...)`. Extract the `IN`th of the … of the `I1`st of the `I0`th element of `x`.
- `rank<I...>(x)  := rank(get<I...>(x))`. The rank of the `I...`th element of `x`.
- `depth<I...>(x) := depth(get<I...>(x))`. The depth of the `I...`th element of `x`.
- `shape<I...>(x)  := shape(get<I...>(x))`. The shape of the `I...`th element of `x`.
- `size<I...>(x)  := size(get<I...>(x))`. The size of the `I...`th element of `x`.

In the following examples, you’ll see use of `size<0>` and `size<1>` to determine loops bounds for the 0th and 1st mode of a layout or tensor.

### Constructing a Layout

> 代码见：

A `Layout` can be constructed in many different ways. It can include any combination of compile-time (static) integers or run-time (dynamic) integers.

The `make_layout` function returns a `Layout`. It deduces the types of the function’s arguments and returns a `Layout` with the appropriate template arguments. Similarly, the `make_shape` and `make_stride` functions return a `Shape` resp. `Stride`. CuTe often uses these `make_*` functions due to restrictions around constructor template argument deduction (CTAD) and to avoid having to repeat static or dynamic integer types.

When the `Stride` argument is omitted, it is generated from the provided `Shape` with `LayoutLeft` as default. The `LayoutLeft` tag constructs strides as an exclusive prefix product of the `Shape` from left to right, without regard to the `Shape`’s hierarchy. This can be considered a “generalized column-major stride generation”. The `LayoutRight` tag constructs strides as an exclusive prefix product of the `Shape` from right to left, without regard to the `Shape`’s hierarchy. For shapes of depth one, this can be considered a “row-major stride generation”, but for hierarchical shapes the resulting strides may be surprising. For example, the strides of `s2xh4` above could be generated with `LayoutRight`.

The `Shape:Stride` notation is used quite often for `Layout`. The `_N` notation is shorthand for a static integer while other integers are dynamic integers. Observe that both `Shape` and `Stride` may be composed of both static and dynamic integers.

Also note that the `Shape` and `Stride` are assumed to be _congruent_. That is, `Shape` and `Stride` have the same tuple profiles. For every integer in `Shape`, there is a corresponding integer in `Stride`. This can be asserted with

```c
static_assert(congruent(my_shape, my_stride));
```

## GEMM

# Conceptual GEMM Hierarchy

> 参考： <a href="https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/">nv 2025 blog cutlass 3 介绍</a>，[cutlass 3 分层设计](https://docs.nvidia.com/cutlass/media/docs/cpp/gemm_api_3x.html)
> CUTLASS 3.x: Orthogonal, Reusable, and Composable Abstractions for GEMM Kernel Design

本节就结合多年 GTC 的内容汇总介绍一下现在 cutlass 的总体架构。

GEMM optimization on GPUs is a **modular** problem. Performant implementations need to specify **hyperparameters** such as tile shapes, math and copy instructions, and warp-specialization schemes. These hyperparameters are, to a large extent, independent from each other; moreover, the best choices may vary significantly based on hardware, problem shape, or other user needs.

With the 3.x redesign, CUTLASS aimed to maximize coverage of the space of GEMM implementations through a hierarchical system of composable, orthogonal building blocks, while also improving code readability and extending support to later NVIDIA architectures such as Hopper and Blackwell. As this design philosophy is linked to the hierarchical hardware design of the GPU, it can also be a good choice for other GPU applications – for example, [FlashAttention-3](https://github.com/Dao-AILab/flash-attention/tree/main/hopper) uses familiar CUTLASS abstractions in its design.

CUTLASS presents a uniform programming model for matrix multiply-accumulate (MMA) operations at different levels of the GPU system hierarchy. CUTLASS 3.0 has GEMM APIs corresponding to the following levels in order of highest to the lowest level.

1. Device
2. Kernel
3. Collective
4. Tiled MMA and Copy
5. Atom

这篇博客讲解 CUTLASS 3.x 中 GEMM 分层系统背后的设计原则，主要设计前 3 哥层次，并且会用到低级 CuTe 抽象来构建 GEMM 内核，关于下面两个层次见下文 [[#CuTe an Introduction]] 以及对应的博客 [part 1](https://developer.nvidia.com/blog/cutlass-principled-abstractions-for-handling-multidimensional-data-through-tensors-and-spatial-microkernels)。

## CUTLASS GEMM Model

CUTLASS implements algorithms that express the classical “triply nested loop” GEMM algorithm with a tiled structure mirroring the above hierarchy.

The following pseudocode describes the model for a GEMM kernel targeting a warp-synchronous matrix multiply instruction like `mma.sync.` The entire operation is referred to as “Gemm,” as it is assumed that an epilogue operation performs the general matrix update similar to BLAS.

```c
// cutlass::gemm::kernel::GemmUniversal: ClusterTileM and ClusterTileN loops
//   are either rasterized by the hardware or scheduled by the kernel in persistent kernels.
// Parallelism over thread block clusters
for (int cluster_m = 0; cluster_m < GemmM; cluster_m += ClusterTileM) {
  for (int cluster_n = 0; cluster_n < GemmN; cluster_n += ClusterTileN) {

    // cutlass::gemm::collective::CollectiveMma: mainloop that iterates over all k-tiles
    // No loop unrolling is performed at this stage
    for (int k_tile = 0; k_tile < size<2>(gmem_tensor_A); k_tile++) {

      // loops inside cute::gemm(tiled_mma, a, b, c); Dispatch 5: (V,M,K) x (V,N,K) => (V,M,N)
      // TiledMma uses the hardware instruction provided through its Mma_Atom
      // TiledMma's atom layout, value layout, and permutations define the iteration order
      for (int tiled_mma_k = 0; tiled_mma_k < size<2>(A); tiled_mma_k++) {
        for (int tiled_mma_m = 0; tiled_mma_m < size<1>(A); tiled_mma_m++) {
          for (int tiled_mma_n = 0; tiled_mma_n < size<1>(B); tiled_mma_n++) {

            // TiledMma's vector mode dispatches to the underlying instruction.
            mma.call(d, a, b, c);
          } // tiled_mma_n
        } // tiled_mma_m
      } // tiled_mma_k
    } // k_tile mainloop
  } // cluster_m
} // cluster_n
```

The first three nested `for` loops correspond to parallelism over thread block clusters. The code does not actually express them as explicit `for` loops. Instead, the parallelization scheme over tiles is implied by CUDA grid launch semantics. However, for persistent kernels, these three loops are expressed in the source code as a single `while` loop that queries the [work tile scheduler](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp) for problem tiles on which to compute.

Inside the three nested `for` loops, one finds code that pulls matrix tiles from global memory into more “local” memory (like shared memory or registers) and computes MMAs. These tiled copy and tiled mma iterations are generally fully static and get fully unrolled.

前三个嵌套的 for 循环对应于线程块集群上的并行性。代码实际上并没有将它们表示为显式的 for 循环。相反，基于瓦片的并行化方案是由 CUDA 网格启动语义暗示的。然而，对于持久内核，这三个循环在源代码中被表示为一个单独的 while 循环，该循环会向工作瓦片调度器查询要在其上进行计算的问题瓦片。

在这三个嵌套的 for 循环内部，可以看到将矩阵瓦片从全局内存提取到更“本地”的内存（如共享内存或寄存器）并计算矩阵乘法累加运算（MMAs）的代码。这些分块复制和分块矩阵乘法累加迭代通常是完全静态的，并会被完全展开。

## A New Conceptual GEMM Hierarchy in CUTLASS 3.x

CUTLASS 3.x develops a [conceptual GEMM hierarchy](https://docs.nvidia.com/cutlass/media/docs/cpp/gemm_api_3x.html) that’s independent of specific hardware features. It is structured into five layers:

![[file-20251006173619177.png]]

> _Figure 1. Conceptual diagram of the CUTLASS GEMM hierarchy independent of hardware_

CUTLASS expresses the above loop nest with the following components which are specialized for data type, layout, and math instruction.

| API level            |                                                                                                                                                                         | API Class and/or function names                                                                                                                 |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Device               | Host-side setup and interface                                                                                                                                           | `cutlass::gemm::device::GemmUniversalAdapter`                                                                                                   |
| Kernel               | Device code for executing a kernel over a grid of threadblocks/clusters                                                                                                 | `cutlass::gemm::kernel::GemmUniversal`                                                                                                          |
| Collective           | Temporal micro-kernels that use architecture-specific synchronization to orchestrate the execution of one or more spatial micro-kernels to compute a single output tile | `cutlass::gemm::collective::CollectiveMma`  <br>`cutlass::epilogue::collective::DefaultEpilogue`  <br>`cutlass::epilogue::collective::Epilogue` |
| Tiled (MMA and Copy) | Spatial micro-kernels that allow for arbitrary interleaving and tiling of architecture specific atoms                                                                   | `cute::TiledMma` and `cute::TiledCopy`  <br>`cute::gemm()` and `cute::copy()`                                                                   |
| Atom                 | Architecture-specific instructions, and associated meta-information                                                                                                     | `cute::Mma_Atom` and `cute::Copy_Atom`                                                                                                          |

Each layer serves as a composition point for abstractions from the previous layer, which can be highly customized using template parameters. Users can either stick to the highest layers, trusting CUTLASS’s compile-time logic to provide a performant GEMM implementation, or opt in to advanced modifications exposed by lower levels of the hierarchy. The spatial micro-kernels provided by the Atom and Tiled MMA/Copy layers are the domain of CuTe and were discussed in part 1. The rest of this post will cover the temporal and kernel-level organization of GEMM made available in the higher layers.

![[file-20251006161349425.png]]

> GTC 2024

In CUTLASS 3.0, we assemble kernels by first composing a collective mainloop and collective epilogue together at the kernel layer, and then wrapping them with a host-side adapter to form a GEMM handle to that kernel.

The following sections describe these components in the order a user should instantiate them in order to assemble a kernel. This order is

1. assemble the required collective mainloop and epilogues,
2. compose them together to build a kernel type, and
3. wrap up the kernel with a device layer adapter.

This order is also reflected in the [CUTLASS 3.0 Hopper kernel examples](https://github.com/NVIDIA/cutlass/tree/main/examples/48_hopper_warp_specialized_gemm) as seen in the excerpt below.

```c
// Step 1: Generate the required collective layer mainloop specialization
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TilesShape, ClusterShape,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

// Step 2: Specify the collective layer epilogue type
using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
    ElementC,
    cutlass::gemm::TagToStrideC_t<LayoutC>,
    cutlass::gemm::TagToStrideC_t<LayoutC>,
    cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>>;

// Step 3: Compose the mainloop and epilogue together at the kernel layer
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>, // ProblemShape [M,N,K,L]
    CollectiveMainloop,
    CollectiveEpilogue
>;

// Step 4: Wrap up the kernel::GemmUniversal kernel class
// with the device adapter to obtain a host-side handle to the kernel
using GemmHandle = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

```

## Collective

## Kernel

## Device API
