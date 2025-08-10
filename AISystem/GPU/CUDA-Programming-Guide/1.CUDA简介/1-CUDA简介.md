---
dateCreated: 2025-07-05
dateModified: 2025-08-10
---

对应 [https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html](https://link.zhihu.com/?target=https%3A//docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) 第 1-4 章

# 1. CUDA 简介

## 为什么要使用 GPU

GPU（Graphics Processing Unit）在相同的价格和功率范围内，比 CPU 提供更高的指令吞吐量和内存带宽。许多应用程序利用这些更高的能力，使得自己在 GPU 上比在 CPU 上运行得更快。其他计算设备，如 FPGA，也非常节能，但提供的编程灵活性要比 GPU 少得多。

GPU 和 CPU 之间的主要区别在于设计思想的不同。CPU 的设计初衷是为了实现在执行一系列操作（称之为一个 thread）时达到尽可能高的性能，同时可能只能实现其中数十个线程的并行化，GPU 的设计初衷是为了实现在在并行执行数千个线程时达到尽可能高的性能（通过分摊较慢的单线程程序以实现更高的吞吐量）。

为了能够实现更高强度的并行计算，GPU 将更多的晶体管用于数据计算而不是数据缓存或控制流。下图显示了 CPU 与 GPU 的芯片资源分布示例。

![The GPU Devotes More Transistors to Data Processing](gpu-devotes-more-transistors-to-data-processing.png)

一般来说，应用程序有并行和串行部分，所以系统可以利用 GPU 和 CPU 的混搭来获得更高的整体性能。对于并行度高的程序也可以利用 GPU 的大规模并行特性来实现比 CPU 更高的性能。

## CUDA：通用并行计算平台和程序模型

CUDA（Compute Unified Device Architecture）是 NVIDIA 推出的通用并行计算平台和程序模型，它利用 NVIDIA GPU 中的并行计算引擎以比 CPU 更有效的方式加速计算密集型任务（如图形处理、深度学习、科学计算）。

CUDA 可以直接链接到 GPU 的虚拟指令集和并行计算单元，从而在 GPU 中完成内核函数的计算。CUDA 提供 C/C++/Fortran 接口，也有许多高性能计算或深度学习库提供包装后的 Python 接口。如下图所示，支持其他语言、应用程序编程接口或基于指令的方法，例如 FORTRAN、DirectCompute、OpenACC。

![gpu-computing-applications.png](gpu-computing-applications.png)

- **核心优势**：
    - 利用 GPU 数千个核心实现并行计算，速度比 CPU 快数十倍甚至百倍。
    - 支持 C/C++、Python（通过 PyCUDA、Numba）等编程语言，兼容主流深度学习框架（TensorFlow、PyTorch）。

## A Scalable Programming Model

多核 CPU 和超多核 (manycore) GPU 的出现，意味着主流处理器进入并行时代。当下开发应用程序的挑战在于能够利用不断增加的处理器核数实现对于程序并行性透明地扩展，例如 3D 图像应用可以透明地拓展其并行性来适应内核数量不同的 GPUs 硬件。

CUDA 并行程序模型主要为克服这一挑战而设计，其对于程序员具有较小的学习难度，因为其使用了标准编程语言如 C。其核心是三个关键抽象——**线程组的层次结构、共享内存和屏障同步**——它们只是作为最小的语言扩展集向程序员公开。

这些抽象提供了细粒度的**数据并行性和线程并行性**，并将嵌套在粗粒度的**数据并行和任务并行**中。它们指导程序员将主问题拆解为可以线程块独立并行解决的粗粒度子问题，同时每个子问题可以被进一步细分为更小的组成部分，其可以被每个线程块中的线程通过并行合作的方式解决。

这种分解通过允许线程在求解每个子问题时进行协作，保留了语言的表达能力，同时实现了自动可扩展性。实际上，每个线程块可被调度到 GPU 内任何可用的多处理器上，调度顺序、方式（并发或串行）均不受限制。因此，如图 3 所示，已编译的 CUDA 程序能够在任意数量的多处理器上执行，且仅需运行时系统知晓物理多处理器的数量。

这种可扩展的编程模型使得 GPU 架构能够通过简单调整多处理器和内存分区的数量来覆盖广泛的市场范围。

![automatic-scalability.png](automatic-scalability.png)

> [!note] SM
> 注意：GPU 是围绕一系列流式多处理器 (SM: Streaming Multiprocessors) 构建的（有关详细信息，请参 [阅硬件实现](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation)）。多线程程序被划分为彼此独立执行的线程块，因此具有更多多处理器的 GPU 将比具有更少多处理器的 GPU 在更短的时间内完成程序执行。

## GPU 架构

GPU 并不是一个独立运行的计算平台，而需要与 CPU 协同工作，可以看成是 CPU 的协处理器，因此当我们在说 GPU 并行计算时，其实是指的基于 CPU+GPU 的异构计算架构。在异构计算架构中，GPU 与 CPU 通过 [PCIe总线](https://zhida.zhihu.com/search?content_id=6024941&content_type=Article&match_order=1&q=PCIe%E6%80%BB%E7%BA%BF&zhida_source=entity) 连接在一起来协同工作，CPU 所在位置称为为主机端（host），而 GPU 所在位置称为设备端（device），如下图所示。CPU 起控制作用，一般称为主机 host，GPU 看作 CPU 协处理器，一般称为设备 device，主机和设备之间内存访问一般通过 PCIe 总线链接。

![](CPU+GPU.png)

基于 CPU+GPU 的异构计算. 来源：Preofessional CUDA® C Programming

可以看到 GPU 包括更多的运算核心，其特别适合数据并行的计算密集型任务，如大型矩阵运算，而 CPU 的运算核心较少，但是其可以实现复杂的逻辑运算，因此其适合控制密集型任务。另外，CPU 上的线程是重量级的，上下文切换开销大，但是 GPU 由于存在很多核心，其线程是轻量级的。因此，基于 CPU+GPU 的异构计算平台可以优势互补，CPU 负责处理逻辑复杂的串行程序，而 GPU 重点处理数据密集型的并行计算程序，从而发挥最大功效。

![](CPU+GPU异构计算.png)

基于 CPU+GPU 的异构计算应用执行逻辑. 来源：Preofessional CUDA® C Programming

CUDA 是 NVIDIA 公司所开发的 GPU 编程模型，它提供了 GPU 编程的简易接口，基于 CUDA 编程可以构建基于 GPU 计算的应用程序。CUDA 提供了对其它编程语言的支持，如 C/C++，Python，Fortran 等语言

# Summary
### **1. CUDA Overview**

CUDA, developed by NVIDIA, is a parallel computing platform and programming model that harnesses GPU power to accelerate compute-intensive applications written in C, C++, or Fortran. Widely used in deep learning, scientific computing, and HPC, it bridges the gap between CPU sequential processing and GPU parallel throughput.

### **2. Key Components & Benefits**

- **GPU Architecture**: Optimized for parallelism, GPUs allocate more transistors to data processing than CPUs, enabling thousands of concurrent threads.
- **CUDA Programming Model**: Uses abstractions like thread blocks, shared memory, and synchronization to partition problems into parallel sub-tasks. This design ensures scalability across GPUs with varying core counts.
- **Software Ecosystem**: Supports multiple languages (e.g., FORTRAN, OpenACC) and scales from high-end to mainstream GPUs, making it accessible for diverse applications.

### **3. Scalability & Market Impact**

CUDA’s programming model enables automatic scalability: compiled programs run efficiently on any GPU by dynamically scheduling thread blocks across available multiprocessors. This flexibility allows NVIDIA to target a broad market, from enthusiast-grade GeForce GPUs to professional Quadro/Tesla products, by simply adjusting hardware components like multiprocessor counts.

# Installation

## Driver Toolkit

https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=runfile_local

可以在路径

```text
/usr/local/cuda-10.1/extras/demo_suite
```

路径下找到一些样例程序。deviceQuery 将输出 CUDA 的相关信息。CUDA 的各种特性：纹理内存 (texture memory)、常量内存 (constant memory)、共享内存 (shared memory)、块 (block)、线程 (thread)、统一寻址 (unified addressing) 都包含在以上信息中。了解这些特性是进行 CUDA C/C++ 编程的基础。

# GPU 性能指标

1. 核心数
2. GPU 显存容量
3. GPU 计算峰值
4. 显存带宽

A 100 架构解析：https://zhuanlan.zhihu.com/p/1908285912053453831

# **NVIDIA 主流架构演进**

| **架构名称**         | **发布时间** | **核心参数**                                                                      | **特点和优势**                                                                                                                                                                                                  | **算力等级** | **代表型号**                                      |
| ---------------- | -------- | ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | --------------------------------------------- |
| **Fermi**        | 2010 年   | 晶体管：30 亿  <br>CUDA 核心：512  <br>SM 单元：16  <br>制程：40nm                          | 首次引入统一计算架构，支持 ECC 内存和动态并行计算，推动 GPGPU 应用从科学计算向通用计算扩展。但受限于制程和架构设计，能效比偏低。| 2.0      | GeForce GTX 580  <br>Tesla C2050              |
| **Kepler**       | 2012 年   | 晶体管：43 亿  <br>CUDA 核心：2304  <br>SM 单元：15  <br>制程：28nm                         | 引入 GPU Boost 动态超频技术，支持动态并行计算和单精度浮点（FP32）性能提升，能效比相比 Fermi 提升 50% 以上。GK110 核心首次实现完整双精度浮点（FP64）计算能力，推动 HPC 领域发展。| 3.0      | GeForce GTX 780  <br>Tesla K40                |
| **Maxwell**      | 2014 年   | 晶体管：29 亿  <br>CUDA 核心：2048  <br>SM 单元：16  <br>制程：28nm                         | 革命性优化能效比（较 Kepler 提升 3 倍），引入多分辨率渲染（MFAA）和动态超分辨率（DSR），支持 DirectX 12 和 OpenGL 4.5。首次在消费级显卡中实现完整的异步计算和多线程处理，为 VR 应用奠定基础。| 5.0/5.2  | GeForce GTX 980  <br>Tesla M40                |
| **Pascal**       | 2016 年   | 晶体管：72 亿  <br>CUDA 核心：3584  <br>SM 单元：28  <br>制程：16nm                         | 首次引入 16nm FinFET 制程，支持 **HBM2** 显存（带宽提升 3 倍），并推出首个专为深度学习设计的 **Tensor Core**（P100）。消费级显卡（如 GTX 1080）首次支持实时光线追踪（需软件支持），同时多 GPU 互联技术 SLI 升级至更高效的 NVLink。| 6.0/6.1  | GeForce GTX 1080  <br>Tesla P100              |
| **Volta**        | 2017 年   | 晶体管：211 亿  <br>CUDA 核心：5120  <br>SM 单元：80  <br>制程：12nm                        | 革命性 Tensor Core 支持**混合精度计算（FP16/FP32/INT8）**，AI 性能提升 50 倍以上。首次实现**结构化稀疏（Structured Sparsity）技术**，同时引入 **GDDR5X** 显存和 **NVLink 2.0**（带宽 300GB/s）。Volta 架构成为 AI 训练和推理的里程碑，V100 GPU 至今仍广泛应用于数据中心。| 7.0      | Tesla V100  <br>Quadro GV100                  |
| **Turing**       | 2018 年   | 晶体管：186 亿  <br>CUDA 核心：2560  <br>RT Core：32  <br>Tensor Core：256  <br>制程：12nm | 首次集成专用**光线追踪核心（RT Core）**，支持实时光线追踪加速（较 CPU 快 100 倍），并推出 **DLSS** 1.0（深度学习超采样）。引入 RTX 平台，将光线追踪、深度学习和栅格化技术深度融合，重新定义游戏和专业图形渲染标准。| 7.5      | GeForce RTX 2080  <br>Tesla T4                |
| **Ampere**       | 2020 年   | 晶体管：540 亿  <br>CUDA 核心：5376  <br>第三代 Tensor Core：432  <br>制程：7nm              | 第三代 Tensor Core 支持 **TF32** 精度（性能提升 20 倍），第二代 RT Core（光线追踪性能翻倍），并引入**多实例 GPU（MIG）技术**，支持 GPU 资源细粒度分割。第三代 NVLink 带宽达 600GB/s，A100 GPU 成为超算和 AI 训练的标杆。消费级显卡（如 RTX 3090）首次实现 24GB GDDR6X 显存，推动 8K 游戏和内容创作。| 8.0/8.6  | GeForce RTX 3090  <br>A100 Tensor Core GPU    |
| **Ada Lovelace** | 2022 年   | 晶体管：760 亿  <br>CUDA 核心：16384  <br>第四代 Tensor Core：512  <br>制程：4nm             | 第四代 Tensor Core 支持 **FP8** 精度（AI 推理性能提升 4 倍），第三代 RT Core（光线追踪性能提升 2 倍），并推出 DLSS 3.0（结合光线重建技术）。AV1 编码加速引擎支持 8K 视频实时处理，同时 Ada 架构首次在消费级显卡中实现 12 层光追计算，推动影视渲染和虚拟制作进入实时时代。| 8.9      | GeForce RTX 4090  <br>RTX 6000 Ada Generation |
| **Hopper**       | 2022 年   | 晶体管：800 亿  <br>CUDA 核心：6080  <br>第四代 Tensor Core：608  <br>制程：4nm              | 第四代 Tensor Core 支持 **Transformer** 引擎（AI 训练速度提升 30 倍），DPX 指令（动态编程加速 40 倍），并引入机密计算（保护数据隐私）。第四代 NVLink 带宽达 900GB/s，H100 GPU 首次实现 900GB/s 显存带宽，成为百亿亿次超算和万亿参数大模型的核心。| 9.0      | H100 Tensor Core GPU  <br>H200 NVL            |
| **Blackwell**    | 2024 年   | 晶体管：2080 亿  <br>CUDA 核心：28160  <br>第五代 Tensor Core：2240  <br>制程：4nm           | 第二代 Transformer 引擎支持 **FP4** 精度（AI 算力达 20 PetaFLOPS），第五代 NVLink 带宽达 1.8TB/s，支持多 GPU 集群无缝互联。新增解压缩引擎（数据库查询加速 5 倍）、RAS 引擎（故障预测与修复），并首次实现芯片级机密计算。GB200 超级芯片（双 B200+Grace CPU）推理性能较 H100 提升 30 倍，成本和能耗降低至 1/25。| 9.6      | B200 GPU  <br>GB200 Superchip                 |
