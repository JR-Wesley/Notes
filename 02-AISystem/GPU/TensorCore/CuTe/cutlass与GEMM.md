![[gemm-hierarchy-with-epilogue-no-labels.png]]

本系列记录了一个完整的 cutlass/cute 入门文档，对这个库提供了一个整体的认识。

# Resources

- 作为手册查询的文档，仍然注意，官方的文档详尽地罗列了特性，但是它默认读者了解底层优化与实现的，所以并不适合入门：
	- <a href="https://docs.nvidia.com/cutlass/index.html">cutlass 官方文档</a>
	- [CUTLASS GitHub Repository](https://github.com/NVIDIA/cutlass)
	- [CUTLASS Wiki Documentation](https://github.com/NVIDIA/cutlass/wiki/Documentation) (great markdowns with examples and images)
	- [CUTLASS Documentation](https://nvidia.github.io/cutlass/index.html)
	- [CUTE Documentation](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cute)
	- [CUTE Tutorial](https://github.com/NVIDIA/cutlass/tree/main/examples/cute/tutorial)
- 重点：入门，了解 cutlass 的分层、Cute Layout 核心概念：
	- <a href="https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/">nv blog 2017</a>：强烈推荐，非常清晰地讲解了在 cutlass 背景下针对 GEMM 优化的分块、内外积转换、缓存，以及相关基础概念和分层设计。
	- <a href="https://siboehm.com/articles/22/CUDA-MMM">GEMM 优化博客</a>：强烈推荐，结合性能指标分析从一个最原始的 GEMM 开始优化，注重性能分析，建议结合NCU动手实践，不过有些讲解可能会有点让人疑惑。
	- <a href="https://developer.nvidia.com/blog/cutlass-principled-abstractions-for-handling-multidimensional-data-through-tensors-and-spatial-microkernels/">nv blog 2025 cutlass</a> 强烈推荐，介绍 cute，结合官方文档的<a href="https://docs.nvidia.com/cutlass/media/docs/cpp/cute/index.html#">CuTe</a>系列理解 Layout 设计抽象。
	- <a href="https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/">nv blog cutlass 3 介绍</a> TODO
	- <a href="https://www.bilibili.com/video/BV1kToTY6Eh5/?p=4&share_source=copy_web&vd_source=fd37be71d17f708cc53476cbd29e590f">B 站讲解视频 【【CUDA 进阶】Cutlass 软件抽象分层与源码浅析（已完结）</a>，建议了解原理后学习cutlass示例代码。
- 官方推荐：
	- We have also described the structure of an efficient GEMM in our talk at the [GPU Technology Conference 2018](http://on-demand.gputechconf.com/gtc/2018/presentation/s8854-cutlass-software-primitives-for-dense-linear-algebra-at-all-levels-and-scales-within-cuda.pdf).
	- [CUTLASS: Software Primitives for Dense Linear Algebra at All Levels and Scales within CUDA](https://www.nvidia.com/en-us/on-demand/session/gtcsiliconvalley2018-s8854/)
	- [Developing CUDA Kernels to Push Tensor Cores to the Absolute Limit on NVIDIA A100](https://www.nvidia.com/en-us/on-demand/session/gtcsj20-s21745/)
	- [Accelerating Convolution with Tensor Cores in CUTLASS](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31883/)
	- [Accelerating Backward Data Gradient by Increasing Tensor Core Utilization in CUTLASS](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41996/)
	- [CUTLASS: Python API, Enhancements, and NVIDIA Hopper](https://www.nvidia.com/en-us/on-demand/session/gtcfall22-a41131/)
- 应用：
	- <a href="https://tridao.me/blog/2024/flash3/">tri dao 对 flash-attention 3 的介绍</a>
	- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
	- [FlashMLA](https://github.com/deepseek-ai/FlashMLA)
	- [flash-attention](https://github.com/Dao-AILab/flash-attention) v3
- 注意：
	- cutlass 和 cute 是不同的抽象，需要略做区分。
	- 注意一些可能有时效性的信息。
	- 很多中文博客是搬运的英文资料，建议直接阅读推荐英文原文。
- GTC
	- [CUTLASS: CUDA TEMPLATE LIBRARY FOR DENSE LINEAR ALGEBRA AT ALL LEVELS AND SCALES (GTC 2018)](https://github.com/MekkCyber/CutlassAcademy/blob/main/s8854-cutlass-software-primitives-for-dense-linear-algebra-at-all-levels-and-scales-within-cuda.pdf)
	- [PROGRAMMING TENSOR CORES: NATIVE VOLTA TENSOR CORES WITH CUTLASS (GTC 2019)](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9593-cutensor-high-performance-tensor-operations-in-cuda-v2.pdf)
	- [Developing CUDA Kernels to Push Tensor Cores to the Absolute Limit on NVIDIA A100 (GTC 2020)](https://www.nvidia.com/en-us/on-demand/session/gtcsj20-s21745/)
	- [Accelerating Convolution with Tensor Cores in CUTLASS (GTC 2021)](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31883/)
	- [Accelerating Backward Data Gradient by Increasing Tensor Core Utilization in CUTLASS (GTC 2022)](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41996/)
	- [CUTLASS: Python API, Enhancements, and NVIDIA Hopper (GTC 2022)](https://www.nvidia.com/en-us/on-demand/session/gtcfall22-a41131/)
	- [Developing Optimal CUDA Kernels on Hopper Tensor Cores (GTC 2023)](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51413/)
	- [CUTLASS: A Performant, Flexible, and Portable Way to Target Hopper Tensor Cores (GTC 2024)](https://www.nvidia.com/en-us/on-demand/session/gtc24-s61198/)

## Articles

 [写给大家看的 CuTe 教程：tiled mma](https://zhuanlan.zhihu.com/p/1937145378446226159) https://zhuanlan.zhihu.com/p/1930389542784964333

**PyTorch**

- [Deep Dive on CUTLASS Ping-Pong GEMM Kernel](https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/)

**Nvidia**

- [Implementing High Performance Matrix Multiplication Using CUTLASS v2.8](https://developer.nvidia.com/blog/implementing-high-performance-matrix-multiplication-using-cutlass-v2-8/)
- [CUTLASS: Fast Linear Algebra in CUDA C++](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)

**Colfax**

- [CUTLASS Tutorial: Fast Matrix-Multiplication with WGMMA on NVIDIA® Hopper™ GPUs](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/)
- [Tutorial: Matrix Transpose in CUTLASS](https://research.colfax-intl.com/tutorial-matrix-transpose-in-cutlass/)
- [CUTLASS Tutorial: Persistent Kernels and Stream-K](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/)
- [CUTLASS Tutorial: Mastering the NVIDIA® Tensor Memory Accelerator (TMA)](https://research.colfax-intl.com/tutorial-hopper-tma/)
- [Developing CUDA Kernels for GEMM on NVIDIA Hopper Architecture using CUTLASS](https://research.colfax-intl.com/nvidia-hopper-gemm-cutlass/)
- [A note on the algebra of CuTe Layouts](https://research.colfax-intl.com/a-note-on-the-algebra-of-cute-layouts/)

**Miscellaneous**

- [Build and Develop CUTLASS CUDA Kernels](https://leimao.github.io/blog/Build-Develop-CUTLASS-CUDA-Kernels/) (How to create a CUDA Docker container for CUTLASS kernel development)
- [learn-cutlass](https://gty111.github.io/2023/03/21/learn-cutlass-1/)

## Videos

- [Lecture 15: CUTLASS (GPU MODE)](https://www.youtube.com/watch?v=G6q719ck7ww&ab_channel=GPUMODE)
- [CUTLASS: A CUDA C++ Template Library for Accelerating Deep Learning Computations (The Linux Foundation)](https://www.youtube.com/watch?v=PWWOGrLZtZg&ab_channel=TheLinuxFoundation)
- [Lecture 36: CUTLASS and Flash Attention 3 (GPU MODE)](https://www.youtube.com/watch?v=JwUcZwPOCpA&t=2831s&ab_channel=GPUMODE)
- [GTC 2024 : CUTLASS: A Performant, Flexible, and Portable Way to Target Hopper Tensor Cores](https://www.nvidia.com/en-us/on-demand/session/gtc24-s61198/)

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

## 安装与编译

https://docs.nvidia.com/cutlass/media/docs/cpp/quickstart.html# 官方给出了编译方法和一些示例。

CUTLASS 是纯头文件组成的库，所以只需要在编译路径下包含库的头文件即可，如：

```cmake
set(CUTLASS_PATH "your-path/cutlass/include/")
set(CUTLASS_UTIL_PATH "your-pathc/cutlass/tools/util/include")
```

## Efficient Matrix Multiplication on GPUs

> 本章来源： https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/
> 注意原文图中的格子数量不完全等于维度信息。

### GEMM Introduction

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

### Hierarchical GEMM Structure

CUTLASS applies the tiling structure to implement GEMM efficiently for GPUs by decomposing the computation into a hierarchy of **thread block tiles**, **warp tiles**, and **thread tiles** and applying the strategy of accumulating matrix products. This hierarchy closely mirrors the [NVIDIA CUDA programming model](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model), as Figure 1 shows. Here, you can see data movement from global memory to shared memory (matrix to thread block tile), from shared memory to the register file (thread block tile to warp tile), and from the register file to the CUDA cores for computation (warp tile to thread tile).

![[fig-09-complete-hierarchy-1.png]]

> Figure 1. The complete GEMM hierarchy transfers data from slower memory to faster memory where it is reused in many math operations.

> [!tip] CUTLASS 分层
> 1. **分层分块与 GPU 硬件架构的 “镜像匹配”**：CUTLASS 的分块层级完全对应 GPU 的 “全局内存→共享内存→寄存器→CUDA 核心” 的存储 / 计算层级，通过 “数据逐级搬运 + 高频复用” 突破内存带宽瓶颈；
> 2. **线程块瓷砖是分层计算的 “核心中间层”**：作为连接 “全局内存（低速）” 和 “warp / 线程计算（高速）” 的桥梁，线程块瓷砖的设计直接决定了数据复用效率和并行计算粒度。

### Thread Block Tile

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

### Warp Tile

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

### Thread Tile

> 注：这一节使用 thread 来进一步划分和运算的，CUDA 也提供了 `mma` 指令用 tensor core 计算。

The CUDA Programming Model is defined in terms of thread blocks and individual threads. Consequently, the warp structure is mapped onto operations performed by individual threads. Threads cannot access each other’s registers, so we must choose an organization that enables values held in registers to be reused for multiple math instructions executed by the same thread. This leads to a 2D tiled structure within a thread as the detailed view in Figure 5 shows. Each thread issues a sequence of independent math instructions to the CUDA cores and computes an accumulated outer product.

![[fig-06-warp-tile-structure.png]]

> Figure 5. An individual thread (right) participates in a warp-level matrix product (left) by computing an outer product of a fragment of A and a fragment of B held in registers. The warp’s accumulators in green are partitioned among the threads within the warp and typically arranged as a set of 2D tiles.

In Figure 5, the upper left quadrant of the warp is shaded in grey. The 32 cells correspond to the 32 threads within a warp. This arrangement leads to multiple threads within the same row or the same column fetching the same elements of the **A** and **B** fragments, respectively. To maximize compute intensity, this basic structure can be replicated to form the full warp-level accumulator tile, yielding an 8-by-8 overall thread tile computed from an outer product of 8-by-1 and 1-by-8 fragments. This is illustrated by the four accumulator tiles shown in green.

> 2D warp tile 继续划分适配寄存器复用与计算并行。A/B 片段加载后重复计算 thread tile 内的累加结果，如右图 16 个数据加载计算了 64 次乘法。


## 动手优化 GEMM

如果基于上一节给出的分级，在warp tile可以使用CUDA 9提出的WMMA，WMMA 要求矩阵的大小固定是 `16x16x16`，`cutlass/gemm/block_task_wmma.h`给出了cutlass的实现，不过WMMA现在已不推荐使用，后续上文的 wmma/complete gemm 代码等就不看了。

> 参考： https://siboehm.com/articles/22/CUDA-MMM

有了上一节的优化方法概论，这一节从一个视角，结合代码定量分析，逐步优化。


## Cutlass 使用

> 直接参考官方仓库 `cutlass/examples/08_turing_tensorop_gemm`。

# CuTe TODO

> 本章来源： https://developer.nvidia.com/blog/cutlass-principled-abstractions-for-handling-multidimensional-data-through-tensors-and-spatial-microkernels/

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

An atom is the smallest collection of threads and data that must cooperatively participate in the execution of a hardware-accelerated math or copy operation.

An Atom combines a PTX instruction with metadata about the shape and arrangement of threads and values that must participate in that instruction. This metadata is expressed as CuTe TV layouts that can then be used to partition arbitrary tensors of input and output data. A user should in general, not have to extend this layer, as we’ll provide implementations of CuTe atoms for new architectures.

![[MMA-Traits-png.webp]]

The above image shows the SM70_8x8x4_F32F16F16F32_NT instruction and its associated `MMA_Traits` metadata. On the left, the TV layouts mapping `(thread_id,value_id) -> coord` are recorded in the traits, and on the right, the traits are visualized with the `inverse coord -> (thread_id,value_id)` mapping. The image on the right can be generated with `print_latex(make_tiled_mma(cute::SM70_8x8x4_F32F16F16F32_NT{}))`
