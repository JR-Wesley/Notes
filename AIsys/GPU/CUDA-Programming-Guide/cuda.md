---
dateCreated: 2025-07-03
dateModified: 2025-08-02
---

本目录基于 https://docs.nvidia.com/cuda/cuda-c-programming-guide/contents.html v12.9 整理。

# 推荐资源

- **官方文档**，主要是两个 Guide：
    - [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)：包含编程指南、API 参考等。
    - [NVIDIA Developer](https://developer.nvidia.com/)：提供教程、示例代码和白皮书。
    - NVIDIA CUDA C++ Programming Guide [https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html](https://link.zhihu.com/?target=https%3A//docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
	    - 基础教程，但是讲的很宽泛，也缺乏细节，有一定的跳跃性。前 7 章是比较核心的内容。
    - CUDA C++ Best Practices Guide  [https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html](https://link.zhihu.com/?target=https%3A//docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
	    - 主要从理论上给出了一些性能优化的方法，如何最大化利用 GPU 特性提升性能。需要掌握一些上面教程的基本概念。
- **在线课程**：
    - 【【精译⚡GPU 计算】贝鲁特美国大学•CMPS224•2021】https://www.bilibili.com/video/BV1Rx4y147Dp/?p=5&share_source=copy_web&vd_source=fd37be71d17f708cc53476cbd29e590f
	    - 基于 Programming Massively Parallel Processors A Hands-on Approach 4th Edition 的 GPU 课程，老师的讲解很深入有见解，PPT 可以见 https://www.elsevier.com/books-and-journals/book-companion/9780323912310。
- **开源项目**：
	- leetgpu
	- [https://github.com/bytedance/lightseq](https://link.zhihu.com/?target=https%3A//github.com/bytedance/lightseq)
		- 字节跳动开源的生成模型推理加速引擎，BERT、GPT、VAE 等等全都支持，速度也是目前业界最快的之一。
	- [https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer](https://link.zhihu.com/?target=https%3A//github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer)
		- 英伟达开源的 Transformer 推理加速引擎。
	- [https://github.com/Tencent/TurboTransformers](https://link.zhihu.com/?target=https%3A//github.com/Tencent/TurboTransformers)
		- 腾讯开源的 Transformer 推理加速引擎。
	- [https://github.com/microsoft/DeepSpeed](https://link.zhihu.com/?target=https%3A//github.com/microsoft/DeepSpeed)
		- DeepSpeed 微软开源的深度学习分布式训练加速引擎。

# 其他参考

https://shichaoxin.com/tags/

chen tianqi：DLSYS https://dlsys.cs.washington.edu/

## Preofessional CUDA® C Programming

- [x] CUDA 编程入门极简教程 https://zhuanlan.zhihu.com/p/34587739
谭升的博客：[https://face2ai.com/program-blog/#GPU编程（CUDA）](https://link.zhihu.com/?target=https%3A//face2ai.com/program-blog/%23GPU%25E7%25BC%2596%25E7%25A8%258B%25EF%25BC%2588CUDA%25EF%25BC%2589)

https://zhuanlan.zhihu.com/p/690779388

https://github.com/mapengfei-nwpu/ProfessionalCUDACProgramming

参考博客：https://jinbridge.dev/docs/hpc/cuda-programming-101/

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
1. **性能优化指南**
    - [NVIDIA Performance Guide](https://developer.nvidia.com/performance-guides)
    - [NVIDIA Code Examples](https://github.com/NVIDIA/cuda-samples)
    - [CUDA Optimization Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

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

# CUDA 核心知识提纲

CUDA 编程的核心知识体系可分为**基础语法**、**并行策略**、**内存优化**、**高级技术**四个递进层次。

### **一、CUDA 基础架构**

1. **硬件模型**
    - GPU 架构层次：SM（流式多处理器）、CUDA Core、Tensor Core
    - 内存层次：寄存器、共享内存、全局内存、常量内存、纹理内存
    - 线程调度：Warp（32 线程）、调度器、指令发射单元
2. **编程模型**
    - 主机 - 设备分离：CPU（主机）控制，GPU（设备）执行计算
    - Kernel 函数：用 `__global__` 修饰，并行执行的函数
    - 线程组织：网格（Grid）→ 线程块（Block）→ 线程（Thread）
    - 线程索引计算：`blockIdx`、`threadIdx`、`blockDim`

### **二、CUDA 编程核心**
1. **内存管理**
    - 内存分配：`cudaMalloc`、`cudaFree`
    - 数据传输：`cudaMemcpy`（同步）、`cudaMemcpyAsync`（异步）
    - 统一内存（Unified Memory）：`cudaMallocManaged`，自动内存迁移
2. **线程同步**
    - 块内同步：`__syncthreads()`，确保所有线程执行到该点再继续
    - 原子操作：`atomicAdd`、`atomicCAS`，实现线程安全的内存操作
3. **CUDA 流（Stream）**
    - 异步执行：任务在流中排队，支持计算与数据传输重叠
    - 流同步：`cudaStreamSynchronize`、事件（`cudaEvent`）

### **三、性能优化核心**

1. **内存优化**
    - 全局内存合并访问：确保 Warp 内线程连续访问内存
    - 共享内存 tiling：减少全局内存访问（如矩阵乘分块）
    - 内存带宽利用率计算：实际带宽 / 理论峰值带宽
2. **计算优化**
    - Tensor Core 利用：使用 `wmma` 库实现高效矩阵乘（FP16/BF16/INT8）
    - 向量化编程：用 `float4` 等类型提高内存访问效率
    - 指令级并行：减少分支发散，提高 Warp 执行效率
3. **资源利用率**
    - 线程块调度：调整块大小以最大化 SM 占用率（Occupancy）
    - 寄存器压力：通过 `nvcc --ptxas-options=-v` 查看寄存器使用

#### 2. **性能优化步骤**

1. **基准测试**：用 nvprof 确定热点函数
2. **分析瓶颈**：
    - 计算瓶颈：低占有率（Occupancy）
    - 内存瓶颈：低内存带宽利用率
3. **针对性优化**：
    - 计算密集型：增加并行度、展开循环
    - 内存密集型：优化内存访问模式、使用共享内存

#### 3. **性能优化黄金法则**

- **减少全局内存访问**：每 100 次计算对应 1 次内存访问
- **最大化并行度**：充分利用 SM 资源
- **避免线程发散**：减少 warp 内分支差异
- **优化内存带宽**：合并访问、对齐数据**合并访问**：相邻线程访问连续内存地址
- **内存对齐**：数据大小为 4/8/16 字节倍数
- **减少全局内存访问**：尽量在寄存器和共享内存中计算

### **四、高级特性**
1. **动态并行**
    - 内核中启动新内核（`cudaLaunchKernel`），适合递归算法
2. **多 GPU 编程**
    - 设备管理：`cudaSetDevice`、`cudaGetDeviceCount`
    - 进程间通信（IPC）：共享 GPU 内存
3. **CUDA 与其他技术结合**
    - CUDA + MPI：分布式多节点 GPU 计算
    - CUDA + OpenMP：混合 CPU-GPU 并行

### **五、调试与性能分析**
1. **调试工具**
    - CUDA-GDB：GPU 内核调试
    - Nsight Compute：详细分析内核性能指标
    - Nsight Systems：系统级性能追踪
2. **关键性能指标**
    - 计算指标：SM 利用率、Tensor Core 利用率
    - 内存指标：全局内存带宽、共享内存 Bank 冲突
    - 指令指标：分支发散率、寄存器压力

### **六、实战案例**
1. **矩阵乘法优化**
    - 朴素实现 → 共享内存 tiling → Tensor Core 优化
2. **卷积加速**
    - 直接卷积 → Im2col + 矩阵乘 → cuDNN 调用
3. **深度学习算子实现**
    - Softmax、BatchNorm、注意力机制的 GPU 优化

### **七、相关生态系统**

#### **一、基础计算与核心库（CUDA 生态基础）**

1. **CUDA Runtime API**：CUDA 的核心运行时接口，提供设备初始化、内存管理（如 `cudaMalloc`）、核函数启动等基础操作，是 CUDA 编程的入口。
2. **CUDA Driver API**：比 Runtime 更底层的驱动接口，需显式加载 CUDA 驱动，支持动态版本适配，常用于需要细粒度控制驱动交互的场景。
3. **NVRTC**：CUDA 运行时编译库，可在程序运行时动态编译 CUDA 核函数，支持动态生成计算逻辑（如根据输入动态调整算子）。
4. **CUDA Math Library (cuMath)**：CUDA 内置的数学函数库，包含基础算术、三角函数、指数函数等，已针对 GPU 架构优化。

#### **二、线性代数与矩阵计算**

1. **cuBLAS**：NVIDIA 优化的线性代数库，支持 dense 矩阵的乘法（GEMM）、向量 - 矩阵运算等，是深度学习和科学计算的基础依赖（如 PyTorch/TensorFlow 底层调用）。
2. **cuSPARSE**：稀疏矩阵计算库，支持稀疏矩阵存储（如 CSR、COO 格式）及稀疏 - 稠密矩阵乘法、稀疏线性方程组求解，适用于高稀疏数据场景（如推荐系统）。
3. **cuSOLVER**：线性代数求解库，基于 cuBLAS 和 cuSPARSE，支持矩阵分解（LU、Cholesky）、特征值求解、最小二乘问题等，面向科学计算和工程仿真。
4. **cuTENSOR**：张量计算库，支持高维张量（如 3D/4D）的收缩、点积、广播等操作，优化了张量内存布局和并行访问，适配深度学习中的张量运算。

#### **三、深度学习专用库**

1. **cuDNN（CUDA Deep Neural Network library）**：深度学习核心加速库，针对卷积（Conv）、池化（Pooling）、激活函数（如 ReLU）、LSTM 等算子做了极致优化，是 PyTorch/TensorFlow 的必选依赖。
2. **TensorRT**：深度学习推理优化库，通过模型量化（INT8/FP16）、层融合、内核自动调优等方式加速推理，支持 C++/Python 接口，常用于生产环境部署。
3. **cuML**：GPU 加速的机器学习库，提供分类（如随机森林）、回归、聚类等算法，兼容 scikit-learn 接口，适合大规模数据集训练。
4. **cuGraph**：GPU 加速的图计算库，支持图遍历（BFS/DFS）、社区发现、图神经网络（GNN）算子等，适配社交网络、推荐系统等图数据场景。

#### **四、并行算法与数据结构**

1. **Thrust**：基于 CUDA 的并行算法库，接口类似 C++ STL，提供排序（sort）、扫描（prefix_sum）、归约（reduce）等算法，自动优化并行粒度，降低并行编程门槛。
2. **CUB（CUDA Unbound）**：更底层的并行原语库，包含线程级 / 块级协作的内存访问、数据重组等工具，供开发者手动优化高性能算子（如 cuDNN 内部使用）。
3. **Moderngpu**：开源并行算法库，专注于高吞吐量的内存密集型操作（如散列、稀疏数据处理），提供可复用的并行模式。

#### **五、信号与图像处理**

1. **cuFFT（CUDA Fast Fourier Transform）**：GPU 加速的快速傅里叶变换库，支持 1D/2D/3D FFT 及逆变换，性能远超 CPU 实现，用于雷达信号处理、图像滤波等。
2. **NPP（NVIDIA Performance Primitives）**：图像处理与信号处理算子库，提供像素级操作（如 resize、滤波）、视频编解码辅助、直方图计算等，适用于实时视觉系统。
3. **cuCVD（CUDA Computer Vision Data Structures）**：计算机视觉基础库，提供图像金字塔、特征点匹配等底层数据结构和算子，常与 NPP 配合使用。

#### **六、科学计算与工程仿真**

1. **cuRAND**：GPU 加速的随机数生成库，支持均匀分布、正态分布等多种随机数类型，性能和随机性优于 CPU 实现，用于蒙特卡洛模拟、深度学习初始化等。
2. **cuSparseX**：cuSPARSE 的扩展，支持更复杂的稀疏矩阵格式（如块稀疏）和高阶运算，面向大规模科学计算（如有限元分析）。
3. **MAGMA（Matrix Algebra on GPU and Multicore Architectures）**：混合 CPU-GPU 的线性代数库，自动分配 CPU/GPU 计算任务，兼容 LAPACK 接口，适合异构计算场景。

### **常见问题与避坑指南**

1. **线程同步错误**：
    - 避免在 `__syncthreads()` 前后出现条件分支（如 `if` 语句），否则可能导致死锁。
2. **内存访问优化**：
    - 全局内存访问需按 32 字节对齐，否则会降低带宽利用率。
3. **调试困难**：
    - 使用 `printf()` 在 Kernel 中打印调试信息，但注意大量打印会严重影响性能。
4. **硬件限制**：
    - 不同 GPU 架构（如 Pascal、Volta、Ampere）的特性（如共享内存大小、Tensor Core 支持）不同，需针对性优化。
5. **线程索引越界**：必须检查 `idx < n`
6. **内存泄漏**：确保 `cudaMalloc` 与 `cudaFree` 成对出现
7. **同步陷阱**：核函数异步执行，需显式同步（如 `cudaDeviceSynchronize`）
