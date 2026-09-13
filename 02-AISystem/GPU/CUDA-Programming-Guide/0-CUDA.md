---
tags:
  - GPU
category: Summary
---

本目录基于 Professional C programming 和 [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/contents.html#)  (v12.9->v13.3) 整理。旨在提供一个系统的入门和实践指南，并且会注意一些编程特性的更新。本系列会尽量表明所有内容的参考和出处，尽量保证信息来源可信且有时效性。

# 推荐资源

- **官方文档**：
	- 官方编程指南：官方在 v13 后更新了一个 [新版全面的编程指南](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)。
	- 查阅与跟进——官方文档：注意 CUDA 和 GPU 架构一直在不断进展，需要跟进 [NVIDIA CUDA C++ Programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) 以同步最新编程范式和方法。官方文档是一个详尽的编程指南，但是没有给出具体的指导方法或者构建整个体系，没把各个用法的关联组织起来。- 是官方详尽的 CUDA 指南，建议反复理解，同时注意会不断更新，建议阅读网页版保持最新进展。但是讲的很宽泛，有一定的跳跃性，前 7 章是入门核心的内容。
    - [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)：包含编程指南、API 参考等的 CUDA 完全工具导航。
	- [run-time API](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
	- [driver API](https://docs.nvidia.com/cuda/cuda-driver-api/index.html)
	- [nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)
	- https://docs.nvidia.com/cuda/archive/9.1/pdf/CUDA_C_Programming_Guide.pdf
		- 2018
	- Professional CUDA： https://www.cs.utexas.edu/~rossbach/cs380p/papers/cuda-programming.pdf
		- 2014， 通过一些简单的例子，解释 CUDA 的基本原理和基本加速概念
    - CUDA C++ Best Practices Guide  [https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html](https://link.zhihu.com/?target=https%3A//docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
	    - 主要从理论上给出了一些性能优化的方法，如何最大化利用 GPU 特性提升性能。需要掌握一些上面教程的基本概念。
	- https://developer.nvidia.com/cuda-books-archive
- 高级特性
	- 强烈推荐——性能优化和 tensor core 英文博客，见 [[cutlass与GEMM]]。
- [NVIDIA Developer](https://developer.nvidia.com/)：官方专题教程、示例代码和白皮书。
	- https://developer.download.nvidia.cn/CUDA/training/StreamsAndConcurrencyWebinar.pdf
- **性能优化与进阶指南**
	- 在有了一定上面的基础变成后，推荐性能分析——C++ best Practice, NCU profiling guide，以及深入了解 CUDA toolkit 或 PTX/SASS。
    - [NVIDIA Performance Guide](https://developer.nvidia.com/performance-guides)
    - [NVIDIA Code Examples](https://github.com/NVIDIA/cuda-samples)
    - [CUDA Optimization Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- **在线课程和书籍**：
	- 强烈推荐入门与系统——Professional C programming：一个详尽的入门指导书，有助于掌握核心的概念和编程模型，需要注意一些特性和工具链的更新，与官方更新的文档相互对照。
	- 视频课程推荐——Programming Massively Parallel Processors A Hands-on Approach 及 CMPS224 课程。
	- 【推荐】一个详尽的总结性的 现代 GPU 架构和高效 GEMM 加速： https://www.aleksagordic.com/blog/matmul
    - 【【精译⚡GPU 计算】贝鲁特美国大学•CMPS224•2021】https://www.bilibili.com/video/BV1Rx4y147Dp/?p=5&share_source=copy_web&vd_source=fd37be71d17f708cc53476cbd29e590f
	    - 基于 Programming Massively Parallel Processors A Hands-on Approach 4th Edition 的 GPU 课程，老师的讲解很深入有见解，PPT 可以见 https://www.elsevier.com/books-and-journals/book-companion/9780323912310。
	- Programming Massively Parallel Processors A Hands-on Approach
		- 这本书详尽的讲述了 CUDA 架构和编程模型，并详细介绍了常见算子的优化，值得阅读。
	- https://www.bilibili.com/video/BV1c64y1Q7Kt/?vd_source=bc07d988d4ccb4ab77470cec6bb87b69
		- https://www.olcf.ornl.gov/calendar/cuda-shared-memory/ **美国橡树岭领导计算设施 (Oak Ridge Leadership Computing Facility, OLCF)** 发布的关于 **CUDA 培训系列**
		- https://www.olcf.ornl.gov/cuda-training-series/
	- 一个基础的 GPU 介绍课程：https://cvw.cac.cornell.edu/gpu-architecture
- **开源项目**：
	- NV 6G GPU https://docs.nvidia.com/aerial/cuda-accelerated-ran/latest/index.html
	- leetgpu
	- [https://github.com/bytedance/lightseq](https://link.zhihu.com/?target=https%3A//github.com/bytedance/lightseq)
		- 字节跳动开源的生成模型推理加速引擎，BERT、GPT、VAE 等等全都支持，速度也是目前业界最快的之一。
	- [https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer](https://link.zhihu.com/?target=https%3A//github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer)
		- 英伟达开源的 Transformer 推理加速引擎。
	- [https://github.com/Tencent/TurboTransformers](https://link.zhihu.com/?target=https%3A//github.com/Tencent/TurboTransformers)
		- 腾讯开源的 Transformer 推理加速引擎。
	- [https://github.com/microsoft/DeepSpeed](https://link.zhihu.com/?target=https%3A//github.com/microsoft/DeepSpeed)
		- DeepSpeed 微软开源的深度学习分布式训练加速引擎。

 选择看其他书籍或者个人博客，注意有一些旧的资料可能会有一些过时的内容，要和官方文档同步。不推荐看照搬英文文章的中文博客。

https://people.maths.ox.ac.uk/gilesm/cuda/index.html

[NVIDIA CUDA Tutorial 1: Introduction](https://www.youtube.com/watch?v=m0nhePeHwFs&list=PLKK11Ligqititws0ZOoGk3SW-TZCar4dK)

# 其他参考

https://christianjmills.com/series/notes/cuda-mode-notes.html

https://shichaoxin.com/tags/

chen tianqi：DLSYS https://dlsys.cs.washington.edu/

## Preofessional CUDA® C Programming

https://github.com/mapengfei-nwpu/ProfessionalCUDACProgramming

参考博客：https://jinbridge.dev/docs/hpc/cuda-programming-101/

📖 本书内容已整合进对应章节笔记：[[2-1执行模型]]（Ch.3）、[[2-2存储模型]]（Ch.4–5）、[[4-4并发与流]]（Ch.6）、[[3-2指令集]]（Ch.7）、[[4-性能优化与高级主题]]（Ch.9 多 GPU）、[[4-2最佳实践]]（Ch.10）。（Ch.8 CUDA 库与 OpenACC 已移除）

CUDA C Programming Guide 解读：https://zhuanlan.zhihu.com/p/53773183

- **书籍**：
    - 《CUDA C 编程权威指南》Professional CUDA C Programming：全面介绍 CUDA 编程模型与优化技巧。
    - 《GPU 高性能编程 CUDA 实战》：通过案例学习 CUDA 并行编程。《CUDA by Example》（CUDA 编程入门经典）
    - 《高性能 CUDA 应用设计与开发》（深入优化）
- **在线课程**：
    - Coursera《GPU 计算基础》（NVIDIA 官方课程）。
    - Udemy《CUDA 并行编程实战》：结合项目实践。
    - https://people.maths.ox.ac.uk/~gilesm/cuda/：该课程每天约有 3 小时的讲座和 4 小时的实践课。课程目标是，在课程结束时，你将能够编写相对简单的程序，并且有信心、有能力通过学习英伟达在 GitHub 上提供的 CUDA 代码示例继续学习。
    - https://tschmidt23.github.io/cse599i/
    - Coursera: [GPU Programming for Science and Engineering](https://www.coursera.org/learn/gpu-programming)
    - Udemy: [CUDA C++ High Performance Parallel Programming](https://www.udemy.com/course/cuda-c-programming/)
    - 《CUDA 高性能编程：GPU 编程实战》
    - 《GPU 高性能编程 CUDA 实战》

HPC 方向主要需要了解 HPC SDK 等较上层的模块，如何使用。涉及运维、功耗控制等方面时，也会涉及驱动中的 NVML 等模块。下面挑选常用的模块作一些介绍：

- [HPC SDK](https://developer.nvidia.com/hpc-sdk)：其实就是把 HPC 常用的子模块打包到了一起。
    - 分析部分包括 Profiles（Nsight）和 Debugger（cuda-gdb）。
- Nsight：有几个子产品：
    - System：综合分析 CPU、GPU 的性能
    - Compute：kernel profiler，专门调试核函数
    - Graphics：调试、分析 Windows 和 Linux 平台图形应用的性能
- [NVTX (Tools Extension Library)](https://github.com/NVIDIA/NVTX)：C 语言 API，提供 C++ 和 Python 接口。Nsight 等性能分析工具通过该 API 进行测量。我们也可以在程序中使用该 API 进行事件记录等。和 MPI 的 PMPI 有些类似。
- [CUPTI (Profiling Tools Interface)](https://developer.nvidia.com/cupti)：和上面那个功能类似，允许各种测量和性能检测的 API。
- [NVML (NVIDIA Management Library)](https://developer.nvidia.com/nvidia-management-library-nvml)：C 语言 API，监控和管理 NVIDIA GPU 设备。API 分为五个模块：初始化和清理、查询、控制、事件处理、错误报告。库文件 `libnvidia-ml.so`，链接参数 `-lnvidia-ml`。
- [NCCL (NVIDIA Collective Communications Library)](https://developer.nvidia.com/nccl)：C 语言 API，MPI 的替代品。提供多 GPU、多节点通信原语。适用硬件：NVLink、Mellanox Network。

# 在一切开始之前——安装

个人环境：

- win11 下安装 wsl2+archlinux 系统，windows 下装有驱动。
- linux 下安装 cuda-tools，包含 ncu，完成后可在 windows 下启动图形界面，arch 下通过 yay 安装。
	- ==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0. For instructions on enabling permissions and to get more information see https://developer.nvidia.com/ERR_NVGPUCTRPERM
	- To allow access for any user, create a file with the .conf extension containing `options nvidia NVreg_RestrictProfilingToAdminUsers=0` in /etc/modprobe.d.
	- 开启 windows 下控制面板性能分析权限，支持 ncu profile。
- 硬件与驱动：NVIDIA GeForce RTX 4060 | NVIDIA-SMI 580.82.09              Driver Version: 581.29         CUDA Version: 13.0     |
- vscode ssh 远程连接与调试
	- 安装 Nsight Visual Studio Code Edition，launch.json 添加调试信息，支持 gdb 调试
	- WARNING: Debug interface is not enabled. Please see https://docs.nvidia.com/cuda/cuda-gdb/index.html#supported-platforms for more details.
	- vscode 添加配置 c_cpp_properties，配置 IntelliSense

# CUDA 核心知识提纲

CUDA 编程的核心知识体系可分为**基础语法**、**并行策略**、**内存优化**、**高级技术**四个递进层次。

## CUDA 编程核心

1. **硬件模型**
    - GPU 架构层次：SM（流式多处理器）、CUDA Core、Tensor Core
    - 内存层次：寄存器、共享内存、全局内存、常量内存、纹理内存
    - 线程调度：Warp（32 线程）、调度器、指令发射单元
2. **编程模型**
    - 主机 - 设备分离：CPU（主机）控制，GPU（设备）执行计算
    - Kernel 函数：用 `__global__` 修饰，并行执行的函数
    - 线程组织：网格（Grid）→ 线程块（Block）→ 线程（Thread）
    - 线程索引计算：`blockIdx`、`threadIdx`、`blockDim`
3. **内存管理**
    - 内存分配：`cudaMalloc`、`cudaFree`
    - 数据传输：`cudaMemcpy`（同步）、`cudaMemcpyAsync`（异步）
    - 统一内存（Unified Memory）：`cudaMallocManaged`，自动内存迁移
4. **线程同步**
    - 块内同步：`__syncthreads()`，确保所有线程执行到该点再继续
    - 原子操作：`atomicAdd`、`atomicCAS`，实现线程安全的内存操作
5. **CUDA 流（Stream）**
    - 异步执行：任务在流中排队，支持计算与数据传输重叠
    - 流同步：`cudaStreamSynchronize`、事件（`cudaEvent`）

## 性能优化核心

见 [[4-性能优化与高级主题]]

1. **内存优化**
    - 全局内存合并访问：确保 Warp 内线程连续访问内存
    - 共享内存 tiling：减少全局内存访问（如矩阵乘分块）
    - 内存带宽利用率计算：实际带宽 / 理论峰值带宽
	- **优化内存带宽**：合并访问、对齐数据**合并访问**：相邻线程访问连续内存地址
	- **内存对齐**：数据大小为 4/8/16 字节倍数
	- **减少全局内存访问**：尽量在寄存器和共享内存中计算
	- **减少全局内存访问**：每 100 次计算对应 1 次内存访问
2. **计算优化**
    - Tensor Core 利用：使用 `wmma` 库实现高效矩阵乘（FP16/BF16/INT8）
    - 向量化编程：用 `float4` 等类型提高内存访问效率
    - 指令级并行：减少分支发散，提高 Warp 执行效率
	- **最大化并行度**：充分利用 SM 资源
	- **避免线程发散**：减少 warp 内分支差异
3. **资源利用率**
    - 线程块调度：调整块大小以最大化 SM 占用率（Occupancy）
    - 寄存器压力：通过 `nvcc --ptxas-options=-v` 查看寄存器使用

## 调试与性能分析

1. **基准测试**：用 nvprof 确定热点函数
2. **分析瓶颈**：
    - 计算瓶颈：低占有率（Occupancy）
    - 内存瓶颈：低内存带宽利用率
3. **针对性优化**：
    - 计算密集型：增加并行度、展开循环
    - 内存密集型：优化内存访问模式、使用共享内存
4. **调试工具**
    - CUDA-GDB：GPU 内核调试
    - Nsight Compute：详细分析内核性能指标
    - Nsight Systems：系统级性能追踪
5. **关键性能指标**
    - 计算指标：SM 利用率、Tensor Core 利用率
    - 内存指标：全局内存带宽、共享内存 Bank 冲突
    - 指令指标：分支发散率、寄存器压力

# 相关生态系统

1. **CUDA Runtime API**：CUDA 的核心运行时接口，提供设备初始化、内存管理（如 `cudaMalloc`）、核函数启动等基础操作，是 CUDA 编程的入口。
2. **CUDA Driver API**：比 Runtime 更底层的驱动接口，需显式加载 CUDA 驱动，支持动态版本适配，常用于需要细粒度控制驱动交互的场景。
3. **NVRTC**：CUDA 运行时编译库，可在程序运行时动态编译 CUDA 核函数，支持动态生成计算逻辑（如根据输入动态调整算子）。
4. **CUDA Math Library (cuMath)**：CUDA 内置的数学函数库，包含基础算术、三角函数、指数函数等，已针对 GPU 架构优化。
