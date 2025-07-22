---
dateCreated: 2025-07-03
dateModified: 2025-07-11
---
# 参考

- [x] CUDA 编程入门极简教程 https://zhuanlan.zhihu.com/p/34587739

- **官方文档**：
    - [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)：包含编程指南、API 参考等。
    - [NVIDIA Developer](https://developer.nvidia.com/)：提供教程、示例代码和白皮书。
    - [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
    - NVIDIA CUDA C++ Programming Guide [https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html](https://link.zhihu.com/?target=https%3A//docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)：这是英伟达官方的 CUDA 编程教程，很多细节没讲，有一定的跳跃性。
    - CUDA C++ Best Practices Guide  [https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html](https://link.zhihu.com/?target=https%3A//docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) 侧重点在实践方面，比如如何编程才能最大化利用 GPU 特性提升性能，建议基础打好之后再来看这个。
- **书籍**：
    - 《CUDA C 编程权威指南》：全面介绍 CUDA 编程模型与优化技巧。
    - 《GPU 高性能编程 CUDA 实战》：通过案例学习 CUDA 并行编程。
- **在线课程**：
    - Coursera《GPU 计算基础》（NVIDIA 官方课程）。
    - Udemy《CUDA 并行编程实战》：结合项目实践。
    - 《CUDA 高性能编程：GPU 编程实战》
    - 《GPU 高性能编程 CUDA 实战》
1. **性能优化指南**
    - [NVIDIA Performance Guide](https://developer.nvidia.com/performance-guides)
2. **官方文档**：
    - [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
3. **书籍**：
    - 《CUDA by Example》（CUDA 编程入门经典）
    - 《高性能 CUDA 应用设计与开发》（深入优化）
4. **在线课程**：
    - Coursera: [GPU Programming for Science and Engineering](https://www.coursera.org/learn/gpu-programming)
    - Udemy: [CUDA C++ High Performance Parallel Programming](https://www.udemy.com/course/cuda-c-programming/)
5. **实战案例**：
    - [NVIDIA Code Examples](https://github.com/NVIDIA/cuda-samples)
    - [CUDA Optimization Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

CUDA 在实际应用中的整体结构层次如上图所示。HPC 方向主要需要了解 HPC SDK 等较上层的模块，如何使用。涉及运维、功耗控制等方面时，也会涉及驱动中的 NVML 等模块。下面挑选常用的模块作一些介绍：

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
- [cuBLAS (Basic Linear Algebra)](https://developer.nvidia.com/cublas)：提供科学计算和 AI 有关的扩展 API，符合 BLAS 和 GEMM API。
- [cuDNN (Deep Neural Network)](https://developer.nvidia.com/cudnn)：C 语言 API，提供 C++ 接口，提供深度神经网络的相关原语。

https://github.com/mapengfei-nwpu/ProfessionalCUDACProgramming

参考：* https://zhuanlan.zhihu.com/p/346910129

CUDA C 解读： https://zhuanlan.zhihu.com/p/53773183

leetgpu

参考博客： https://jinbridge.dev/docs/hpc/cuda-programming-101/

# 核心知识提纲

CUDA 编程的核心知识体系可分为**基础语法**、**并行策略**、**内存优化**、**高级技术**四个递进层次。

以下是 CUDA 的**核心知识体系提纲**，涵盖从基础到高级的完整脉络，帮助你系统掌握 CUDA 编程与优化：

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
1. **NVIDIA 库**
    - cuBLAS：基础线性代数子程序
    - cuDNN：深度学习加速库
    - cuFFT：快速傅里叶变换库
2. **高级抽象框架**
    - PyTorch/TensorFlow：自动生成 CUDA 代码
    - TVM/Halide：自动算子优化

#### **四、常见问题与避坑指南**

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
