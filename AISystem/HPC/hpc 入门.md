---
dateCreated: 2025-07-11
dateModified: 2025-07-13
---

清华最好的 HPC 系统课程：https://lab.cs.tsinghua.edu.cn/hpc/doc/

浙大 HPC 小学期课程：https://docs.zjusct.io/optimization/parallel/gpu/cuda/ 和关于优化的一些资料：https://docs.zjusct.io/optimization/

上交 HPC 入门指南：https://xflops.sjtu.edu.cn/hpc-start-guide/

知乎请问高性能计算的学习路线：https://www.zhihu.com/question/33576416

个人博客： https://www.haibinlaiblog.top/


- 并行程序设计导论：
    - MPI可以参考mpi4py Python库，可以快速验证一些标准概念诸如（通信子，rank,数据收发，broadcast，gather，scatter等通信原语）。 [mpi4py-examples](https://github.com/jbornschein/mpi4py-examples).
    - pthread可以参考C++11的thread多线程并发功能和Java多线程。
    - openMP可以参考 [openMP资源](https://www.openmp.org/resources/)和 [openMP简易教程-海康流行版本](http://read.pudn.com/downloads632/ebook/2565497/OpenMP%E7%AE%80%E6%98%93%E6%95%99%E7%A8%8B.pdf)
    - 高性能计算-SIMD指令集：当前流行的avx2和arm-NEON指令。

- GPU编程优化-大众高性能计算
	- 理论篇：向量机和阵列机结构区别，英伟达GPU代次的计算能力单元硬件结构和功能差异。
	- 入门篇：总共四个范例，对应优达学城parallel-map,reduce,stencil,shared-memory例子，对应优达学城03节课。
	- 提高篇：讲述卷积（conv)，规约（scan),归并、双调，奇偶排序，图像处理等高阶例子。 对应优达学城04节课 。
	- 核心篇：讲述GPU存储器体系（类似存储器山，openMP-MPI-Multi-Cuda)编程，可以和《并行编程导论》参考阅读。
	- 关于动态并行，stream/event 书中未涉及。

优达学城GPU编程：最好的GPU编程教学视频，另一个是周斌的GPU编程参考GPU编程资料中04优达城GPU编程

- 链接: [https://pan.baidu.com/s/1NZt1ZW1qenXlOOPt07ZPfA](https://pan.baidu.com/s/1NZt1ZW1qenXlOOPt07ZPfA) 提取码: j69p


# 并行计算系统

系统学习并行计算需要构建一套涵盖**硬件基础、理论模型、编程框架、算法设计、性能优化**的完整知识体系，同时结合实践掌握核心技能。以下是具体的知识体系和需要掌握的理论、技能：

### 一、基础理论与核心概念

并行计算的理论基础是理解 “为什么并行”“并行的边界在哪里” 的关键，是后续学习的前提。

1. **并行性的本质与来源**

    - 并行性的定义：通过同时执行多个计算任务提升效率的能力。
    - 并行性的来源：数据并行（同一操作作用于多组数据，如向量运算）、任务并行（多个独立任务同时执行，如多线程处理不同请求）、指令级并行（CPU 流水线、超标量执行）、内存并行（多通道内存、缓存并行访问）。
2. **并行计算的核心定律**

    - **Amdahl 定律**：加速比受串行部分限制（`加速比 = 1/(串行比例 + 并行比例/处理器数)`），揭示 “串行瓶颈” 的重要性。
    - **Gustafson 定律**：当问题规模随处理器数增加而扩大时，加速比可接近处理器数（`加速比 = 串行比例 + 处理器数×并行比例`），更贴近实际大规模并行场景。
    - **Karp-Flatt 度量**：评估并行化的 “有效并行度”，判断性能损失是否来自串行部分或并行开销。
3. **并行计算模型（理论抽象）**

    - **PRAM 模型**（并行随机访问机）：抽象的共享内存并行模型，分为 EREW（独占读独占写）、CREW（并发读独占写）、CRCW（并发读写），用于理论算法复杂度分析。
    - **BSP 模型**（Bulk Synchronous Parallel）：分 “局部计算 - 全局通信 - 同步” 三阶段的分布式模型，量化通信和同步开销，更贴近实际分布式系统。
    - **LogP 模型**：通过 `L`（通信延迟）、`o`（通信开销）、`g`（带宽限制）、`P`（处理器数）四个参数建模分布式系统的通信成本，指导算法设计。

### 二、硬件基础：并行计算的 “载体”

并行计算依赖硬件架构，理解硬件是设计高效并行程序的前提。

1. **并行计算的硬件体系结构**

    - **共享内存架构**：多处理器 / 多核 CPU 共享同一块物理内存（如 SMP 对称多处理机），通过缓存一致性协议（MESI 等）保证内存视图一致。
    - **分布式内存架构**：多个独立节点通过网络连接，每个节点有私有内存（如集群、超级计算机），节点间通过消息传递通信。
    - **异构架构**：CPU+GPU/FPGA/AI 芯片组成的混合架构（如 PCIE 连接的 CPU+GPU），CPU 负责逻辑控制，GPU/FPGA 负责高密度并行计算（算力密集型任务）。
2. **关键硬件组件与特性**

    - 多核 CPU：核心数、缓存层次（L1/L2/L3）、指令集（AVX2/AVX512 等向量指令，支持数据并行）、超线程技术。
    - GPU：流处理器（SP）、线程块（Block）、warp 调度、全局内存 / 共享内存 / 寄存器（GPU 内存层次的并行访问特性）。
    - 网络与存储：分布式系统中的通信硬件（InfiniBand/RDMA 用于低延迟通信）、并行文件系统（如 Lustre）。

### 三、并行编程模型与框架

编程模型是 “人 - 硬件” 的接口，不同模型对应不同硬件架构和应用场景，需掌握主流框架的使用与原理。

1. **共享内存编程模型**（适用于多核 CPU、SMP 架构）

    - **线程级并行**：
        - POSIX Threads（Pthreads）：底层线程 API，手动管理线程创建、同步（锁、条件变量）、销毁，适合细粒度控制。
        - OpenMP：基于编译制导（`#pragma`）的共享内存并行框架，支持循环并行（`#pragma omp for`）、任务并行（`#pragma omp task`），简单易用，适合快速并行化串行代码。
    - **核心技能**：线程同步（互斥锁、读写锁、原子操作）、负载均衡（动态调度 `schedule(dynamic)`）、避免竞态条件。
2. **分布式内存编程模型**（适用于集群、超级计算机）

    - **消息传递接口（MPI）**：分布式并行的事实标准，支持进程间通信（点对点通信 `MPI_Send/MPI_Recv`、集体通信 `MPI_Bcast/MPI_Allreduce`），可跨节点扩展到数万进程。
    - 核心概念：进程组（`MPI_Comm`）、_rank_（进程标识）、通信域、非阻塞通信（`MPI_Isend/MPI_Irecv`）提升效率。
    - **核心技能**：设计无死锁的通信逻辑、优化集体通信开销（如调整数据分块大小）、利用 MPI-IO 进行并行 IO。
3. **异构计算编程模型**（适用于 CPU+GPU/FPGA）

    - **CUDA**（NVIDIA GPU 专属）：基于 C/C++ 扩展，通过 `__global__` 函数定义核函数，线程层次（Grid->Block->Thread）映射到 GPU 硬件，支持共享内存（`__shared__`）、原子操作、异步内存拷贝。
    - **OpenCL**（跨平台异构计算）：支持 CPU/GPU/FPGA，通过 “平台 - 设备 - 上下文 - 命令队列” 模型抽象硬件，内核函数（Kernel）在设备上执行，适合多厂商硬件。
    - **OpenACC**：类似 OpenMP 的制导式编程，通过 `#pragma acc` 自动将代码映射到 GPU，降低异构编程门槛。
    - **核心技能**：GPU 内存层次优化（全局内存合并访问、共享内存减少延迟）、线程块划分（避免分支 divergence）、异步计算与通信重叠。
4. **高级并行框架**（面向特定场景的封装）

    - 分布式计算：MapReduce（Hadoop）、Spark（内存计算）、Dask（Python 并行框架）。
    - 机器学习并行：Parameter Server（分布式参数同步）、PyTorch Distributed（多卡训练）、Horovod（跨框架分布式训练）。

### 四、并行算法设计

并行算法是并行计算的 “灵魂”，需掌握如何将串行算法改造成高效的并行版本，核心是**减少通信开销**和**实现负载均衡**。

1. **并行算法设计原则**

    - 分治策略：将问题拆解为独立子问题（如并行排序中的归并排序、快速排序并行化）。
    - 数据划分：通过 “块划分”“循环划分”“带状划分” 将数据分配给不同处理器（如矩阵乘法中按块拆分矩阵，减少跨节点通信）。
    - 局部性利用：最大化本地计算，最小化跨节点 / 核通信（如 GPU 算法中优先使用共享内存而非全局内存）。
2. **经典并行算法案例**

    - 线性代数：并行矩阵乘法（2D/3D 分块）、并行 LU 分解、并行 FFT（Cooley-Tukey 算法的并行化）。
    - 图算法：并行 BFS（分层遍历的任务并行）、并行 PageRank（数据并行更新节点分数）。
    - 排序算法：并行归并排序、基数排序（数据并行分桶）、GPU 上的 Thrust 库排序。
3. **算法复杂度分析**

    - 并行时间复杂度：计算 “关键路径” 上的操作数（而非总操作数）。
    - 通信复杂度：量化数据传输的次数和数据量（如分布式算法中 `O(n/p)` 的通信量，`p` 为处理器数）。

### 五、性能分析与优化

并行程序的性能往往受限于**通信延迟、负载不均、缓存失效、指令并行性不足**等问题，需掌握系统的优化方法。

1. **性能评估指标**

    - 加速比（Speedup）：`串行时间/并行时间`，衡量并行效率提升。
    - 效率（Efficiency）：`加速比/处理器数`，反映资源利用率（理想为 1）。
    - 可扩展性（Scalability）：增加处理器数时，效率是否保持稳定（强可扩展性：固定问题规模；弱可扩展性：问题规模随处理器数增加）。
2. **性能分析工具**

    - 共享内存：`perf`（CPU 指令级剖析）、`gprof`（函数级耗时统计）、`Intel VTune`（缓存命中率、指令并行度）。
    - 分布式系统：`MPI Profiler`（如 `mpiP`、`TAU`）、`Intel Trace Analyzer`（通信轨迹分析）。
    - 异构计算：`nvprof`/`nvidia-smi`（GPU 核函数耗时、内存带宽、SM 利用率）、`Nsight`（GPU 调试与性能剖析）。
3. **核心优化技术**

    - **减少通信开销**：
        - 通信合并：将小数据块合并为大块传输（如 MPI 中用 `MPI_Gatherv` 代替多次 `MPI_Send`）。
        - 通信与计算重叠：利用非阻塞通信（如 MPI 的 `MPI_Irecv`+`MPI_Wait`，GPU 的 `cudaMemcpyAsync`）。
    - **提升数据局部性**：
        - 缓存优化：按缓存行对齐数据、循环分块（Loop Tiling）减少缓存失效（如将大矩阵拆分为缓存大小的块）。
        - 内存访问模式：GPU 中保证全局内存 “合并访问”（线程 ID 与内存地址对齐），避免非对齐访问导致的带宽损失。
    - **负载均衡**：
        - 动态任务调度：OpenMP 的 `schedule(dynamic)`、MPI 的 “主从模式” 动态分配任务。
        - 数据划分优化：按处理器性能分配数据量（如异构系统中给 GPU 分配更多计算密集型任务）。
    - **向量化与指令级并行**：
        - 利用 SIMD 指令（如 AVX2/AVX512）手动或通过编译器（`-O3 -mavx2`）实现数据并行（如前面的矩阵乘法向量化）。
        - 避免控制流分支（如 GPU 中减少 `if-else`，用掩码操作替代）。

### 六、应用场景与实践

并行计算的价值体现在解决实际问题中，需结合具体领域理解其应用。

1. **科学与工程计算**

    - 计算流体力学（CFD）：通过分布式并行求解 Navier-Stokes 方程。
    - 量子化学：用 GPU 加速电子结构计算（如 Gaussian 软件的 GPU 版本）。
2. **人工智能与机器学习**

    - 分布式训练：数据并行（多卡拆分样本）、模型并行（多卡拆分网络层）、混合并行（如 Megatron-LM）。
    - 推理加速：GPU/TPU 的批处理并行、模型量化与并行计算结合。
3. **大数据与云计算**

    - 分布式存储与计算：HDFS+MapReduce 处理 PB 级数据，Spark 的 RDD 并行计算模型。
    - 实时计算：Flink 的流并行处理（按数据分片并行消费）。

### 七、核心技能总结

理论需结合实践，以下是必须掌握的核心技能：

1. **编程能力**：
    - 熟练掌握 C/C++（并行编程的主要语言），了解 Python（快速验证并行框架）。
    - 精通至少 2 种并行框架：如 OpenMP（共享内存）+ MPI（分布式），或 CUDA（GPU）+ OpenMP。
2. **算法设计与实现**：
    - 能将串行算法改造成并行版本（如将串行矩阵乘法拆分为块并行）。
    - 掌握数据划分、任务调度的基本方法，解决负载均衡问题。
3. **性能调优**：
    - 能用性能工具定位瓶颈（如缓存未命中、通信延迟过高）。
    - 熟练应用缓存优化、向量化、通信重叠等技术提升性能。
4. **系统部署与调试**：
    - 能在集群 / 多核 CPU/GPU 环境部署并行程序，解决环境配置问题（如 MPI 版本兼容、GPU 驱动匹配）。
    - 掌握并行程序调试技巧（如多线程调试工具 `gdb`+`thread`，MPI 调试工具 `TotalView`）。

### 学习路径建议

1. **入门**：先掌握计算机体系结构（多核、缓存）和并行基本概念（Amdahl 定律），用 OpenMP 实现简单并行（如并行求和、循环）。
2. **进阶**：学习 MPI 分布式编程（实现分布式矩阵乘法）、GPU 编程（CUDA 基础），理解并行算法设计（分治、数据划分）。
3. **深入**：研究性能优化（缓存、向量化、通信优化），结合具体领域（如机器学习分布式训练）实践，分析大规模并行系统的可扩展性。

通过理论 + 工具 + 实践的结合，逐步构建完整的并行计算知识体系。

# 高性能计算学习路线

## 入门指南

### Easy：一两星期就够了！

- C/C++ 语言
- 多线程与锁（std::thread, pthread）
- SIMD 指令的使用
- OpenMP 的使用

习题：使用多线程（可以使用 openmp，std::thread, pthread 中的任意一种）和 SIMD 优化以下程序

```
int sum_array(int *arr, int len) {
  int sum = 0;
  for(int i = 0; i < len; ++i) {
    sum += arr[i];
  }
  return sum;
}

int dot_product(int *a, int *b, int len) {
  int sum = 0;
  for(int i = 0; i < len; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}
```

### Hard：来点有意思的

- Intel VTune Profiler 和 Intel Advisor 的使用
- 火焰图
- 阿姆达尔定律
- 存储器层次结构，高速缓存
- 原子操作
- MPI 编程

习题：优化矩阵乘法

```
void matrix_mul(int *a, int *b, int *c, int len) {
  for(int i = 0; i < len; ++i) {
    for(int j = 0; j < len; ++j) {
      int sum = 0;
      for(int k = 0; k < len; ++k) {
        sum += a[i * len + k] * b[k * len + j];
      }
      c[i * len + j] = sum;
    }
  }
}
```

### Lunatic：大的要来了

- 异构编程（GPGPU，太湖之光，etc）
- 现代处理器体系结构
- 熟练使用 Linux
- 运维，装机

~~习题：完整参与一次比赛~~

## 高性能计算 = 高性能的算法 + 高性能的软件系统 + 高性能的硬件

HPC 是一个比较综合的方向，涉及算法、体系结构、编程语言、操作系统、计算机网络等，还涉及专业的学科知识譬如生物信息学等，这也正是它的趣味性所在。High level 地想一想，要以最高效的方式来对一个给定问题求解，我们必然需要有高效的算法设计（上层）、高效的编程模型和代码生成（中层）、以及高效的计算机体系结构来执行机器码（下层）。要实现极致的效率，三者必须协作。

### 学习资源

- [Intel Developer Zone (opens new window)](https://software.intel.com/content/www/us/en/develop/home.html) 中有 vtune, advisor 等性能分析工具的使用指南，还有 intel 的各种高性能计算库的文档。
- [CUDA docs (opens new window)](https://docs.nvidia.com/cuda/index.html) 中有 CUDA 的文档和入门教程。
- [MPI tutorial (opens new window)](https://mpitutorial.com/tutorials/)MPI 的入门资料
- [intel Intrinsics Guide (opens new window)](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#cats=Arithmetic&expand=3904,3913,4011,4014,4602,4011&techs=MMX,SSE,SSE2,SSE3,SSSE3,SSE4_1,SSE4_2,AVX,AVX2,FMA,AVX_512,AMX,SVML,Other)intel 的 SIMD 文档
- [Linux perf (opens new window)](http://www.brendangregg.com/linuxperf.html) 介绍了对 Linux 进行性能分析与调优的各种工具
- OpenMP 和 MPI：《并行程序设计导论》
- CUDA :《CUDA 并行程序设计》《GPU 编程指南》第 5、6、9 章
- 运维：《Linux 命令行与 shell 脚本编程大全》(“Linux Command Line and Shell Scripting Bible”)
- 编译、链接：《程序员的自我修养: 装载、链接与库》《深入理解计算机系统》第 7 章
- 内存与 Cache：[What Every Programmer Should Know About Memory(pdf)(opens new window)](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf)

利用**提问**的方式来梳理下一些知识，大部分都是开放式问题，启发你自己去查找资料、思考。

## 基础篇

- 什么是超算？
- 什么是并行计算？什么是并行分布式计算？为什么需要并行分布式计算？
- 什么是线程？什么是进程？

### 程序性能分析

- Intel vtune profiler 如何使用？
- 如何测试程序性能？什么是热点分析？
- 程序性能分析报告怎么看？
- 什么是火焰图？
- 什么是 ebpf？
- 有哪些常用的 profiler？
- 我应该优化程序哪一部分？
- 性能优化有哪些策略：
    1. 并行度优化
    2. 数据传输优化
    3. 存储器访问优化
    4. 向量化优化
    5. 负载均衡优化
    6. 多线程扩展性优化

### 学习 OpenMP

- 什么是 OpenMP？有什么作用？
- 如何使用 OpenMP 来加速我的代码？在 Linux 环境下如何配置？
- 我该在哪并行？如何选择并行区域？
- 什么是数据依赖？什么是数据冲突？如何解决？
- 什么是原子操作？为什么需要原子操作？
- 我该选择多少线程来运行呢？线程数量越多越好吗？
- 我的代码运行正确吗？如何检验优化后代码运行的正确性？
- 我的代码优化成功了吗？加速比如何计算？
- OpenMP 实战：多线程矩阵乘法，矩阵分块乘法。

### 学习 MPI

- 什么是 MPI？
- MPI 有哪些实现（openmpi、intel-mpi、 mpich2、platform-mpi）
- 什么是主进程 master，什么是从进程 slave？
- 什么是进程间通信？为什么进程间需要通信？如何进行进程间通信？
- 如何发送数据和接收数据？都有哪些方法？每种方法之间又何不同？
- 什么是同步？为什么需要同步？
- 什么是死锁？如何避免死锁？我该如何编译和运行 MPI 程序？
- 为什么我在一个节点上运行这么多进程和我在多个节点上运行这么多进程时间不一样？
- MPI 实战：矩阵乘法，矩阵分块乘法。

### 学习使用高性能集群

- 如何远程登录（SSH）？节点之间如何做到无密码访问？
- 集群是怎样运作的？什么是管理节点？什么是计算节点？
- 为什么所有节点都能看到同样的目录？什么是共享存储？
- 了解你的集群。集群拓扑结构是怎样的？配置是什么？使用的是什么网络？
- Linux 基本命令？如何上传和下载文件？
- 如何从源码编译？什么是软件依赖？什么是动态链接库？什么是动态链接？什么是静态链接？
- 什么是脚本？掌握 Shell 脚本的使用。
- 什么是编译器？什么是编译工具链？什么是 make 和 cmake？如何编写 MakeFile 和 CMakeList？
- 编译报错怎么办？
- 如何提交作业？PBS 作业调度系统如何使用？

## 提高篇

### 计算机体系结构

推荐 [现代微处理器架构 90 分钟指南(opens new window)](https://www.starduster.me/2020/11/05/modern-microprocessors-a-90-minute-guide/)

- CPU 的频率、寄存器、缓存、内存的读写时延大概是一个什么比例？
- 内存，PCIE，显存的带宽大概有多大？
- 缓存的结构是什么？缓存里面有哪些设计指标？如何影响缓存的效果？
- 多核的 CPU 并行执行，缓存是如何保持一致的？
- CPU 的流水线是什么？
- 分支预测是怎么做的？
- ILP(Instruction-level parallelism) 是怎么做到的？
- 什么是 NUMA 节点？什么是 SMT？
- CPU 的向量指令集（SIMD）是干嘛的？如何写程序用上 SIMD？
- GPU 和 CPU 的设计区别在哪？GPU 适用于什么场合？
- GPU 的编程范式是什么？GPU 程序怎么写？
- GPU 的摩尔定律终结没有？

### 操作系统、编译器、运行时系统、算法

推荐 [高速缓存与内存一致性专栏(opens new window)](https://zhuanlan.zhihu.com/p/136300660)

- 虚拟内存、TLB 都是啥？为什么使用虚拟内存？
- 进程的内存分布是啥样的？
- OS 眼中的线程、进程是如何被映射到多个 CPU 核心上执行的？线程之间是如何调度的？
- 最主要的一些编译优化都是啥？尤其是循环优化。
- 如何做性能分析？都有些啥工具？
- 如何用各种 hardware performance counter？
- pthread 怎么用？它是怎么实现的？
- OpenMP 怎么用？它是怎么实现的？和 pthread 的区别与联系？
- OpenMP 里面的各种细节，譬如如何同步、如何共享数据等
- MPI 怎么用？它是怎么实现的？和 OpenMP 分别代表了什么模型？
- OpenMP 里的各种细节，各种 collect，如何同步之类的
- SIMD、SPMD、SIMT 之间的区别是什么？
- 线程之间如何同步？
- 并行编程和串行编程有啥区别？
- 并行编程都有哪些模型？
- 如何把一个算法并行化？这个过程中都有哪些讲究？
- 如何把一个算法在 GPU 上高性能地实现？
- 如何实现 lock-free 数据结构和算法？
- 如何设计 cache-oblivious algorithms？

## 拓展阅读

- [Extreme HTTP Performance Tuning: 1.2M API req/s on a 4 vCPU EC2 Instance (opens new window)](https://talawah.io/blog/extreme-http-performance-tuning-one-point-two-million/) 展示了性能调优和热点分析的各种工具的使用。

内容收集主要来自 [高性能计算学习路线 (opens new window)](https://www.zhihu.com/question/33576416) 和 [华农队长的备赛指南(opens new window)](https://mp.weixin.qq.com/s?__biz=MzI5NzUxMDAxOQ==&mid=2247484464&idx=1&sn=060fe6e547468103352485ef56c4b386&chksm=ecb2b1dcdbc538cafbe97f8602c662b5f2001fd63b4a1a10f615fbf9fff0061079efbcc6c713)
