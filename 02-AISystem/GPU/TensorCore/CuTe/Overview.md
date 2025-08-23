---
dateCreated: 2025-08-06
dateModified: 2025-08-06
---
# Cute

[(8 条消息) cutlass cute 101 - 知乎]([https://zhuanlan.zhihu.com/p/660379052](https://zhuanlan.zhihu.com/p/660379052))

[(10 条消息) nvidia cute 简述 (一)--layout - 知乎]([https://zhuanlan.zhihu.com/p/24603200660](https://zhuanlan.zhihu.com/p/24603200660))

[项目简介与总体架构 | NVIDIA/cutlass - KoalaWiki]([https://opendeep.wiki/NVIDIA/cutlass/introduction](https://opendeep.wiki/NVIDIA/cutlass/introduction))

https://zhuanlan.zhihu.com/p/353208013

https://zhuanlan.zhihu.com/p/620185229

# CuTe 解析

CuTe（CUDA Tensor Templates）是 CUTLASS 库的**底层核心组件**，是支撑 CUTLASS 实现高性能线性代数运算的 “骨架”。简单来说，CuTe 为 CUTLASS 提供了**通用的张量（tensor）抽象、索引与布局管理、硬件无关的并行操作原语**，而 CUTLASS 则基于 CuTe 构建了更高层的线性代数运算（如 GEMM、卷积等），并针对特定 GPU 硬件（如 Tensor Core）进行了深度优化。

### 具体关系可以从三个层面理解

#### 1. 定位与职责：底层工具 Vs 高层实现

- **CuTe**：是一套**张量操作的基础模板库**，专注于 “如何描述和操作张量”。它提供了：
    - 张量的通用抽象（如 `Tensor` 类型，可表示任意维度、任意布局的张量）；
    - 灵活的索引与布局系统（如支持行优先、列优先、块布局等，自动处理内存地址计算）；
    - 线程级并行操作的原语（如张量分片、迭代、组合等，与硬件架构解耦）。
        它的核心目标是**提供一套灵活、通用的 “张量操作语言”**，让上层库（如 CUTLASS）可以专注于算法逻辑，而非底层的内存布局或索引计算。
- **CUTLASS**：是**基于 CuTe 的高层线性代数库**，专注于 “如何高效实现具体的线性代数运算”。它利用 CuTe 的张量抽象，实现了：
    - 优化的 GEMM（矩阵乘法）、卷积、转置等运算；
    - 针对不同 GPU 架构（如 Ampere、Hopper）的硬件特性（如 Tensor Core、共享内存）的适配；
    - 支持多种数据类型（fp16、bf16、int8 等）和混合精度计算。
        它的核心目标是**提供即用型的高性能线性代数接口**，开发者无需关心底层张量操作细节，直接调用即可。

#### 2. 依赖关系：CUTLASS “依赖” CuTe

CUTLASS 的代码实现高度依赖 CuTe 的基础组件。例如：

- CUTLASS 中描述矩阵（二维张量）的布局（如 `RowMajor`），本质上是 CuTe 张量布局的特例；
- CUTLASS 中线程块对矩阵的分块计算（如将 1024x1024 矩阵划分为 128x128 子块），依赖 CuTe 的张量分片（tensor slicing）原语；
- CUTLASS 对 Tensor Core 的调用（如 16x16x16 矩阵乘法指令），其输入输出的张量索引计算由 CuTe 自动处理。

可以说，**CuTe 是 CUTLASS 的 “基础设施”**，没有 CuTe 提供的通用张量操作能力，CUTLASS 难以在保持灵活性的同时实现对多种硬件和数据类型的适配。

#### 3. 设计目标：通用性 Vs 高性能

- CuTe 的设计目标是**通用性和灵活性**：它不绑定特定硬件或运算类型，而是提供一套抽象接口，让开发者可以用统一的方式描述和操作任意张量（从 1D 向量到 4D 卷积输入）。例如，同样的 CuTe 代码可以描述矩阵乘法中的 2D 张量，也可以描述卷积中的 4D 张量（NCHW 格式）。
- CUTLASS 的设计目标是**高性能和专用性**：它基于 CuTe 的通用能力，针对线性代数中最核心的运算（如 GEMM、卷积）进行深度优化，包括硬件指令选择（如 Tensor Core vs CUDA Core）、内存访问模式（如合并访问、共享内存复用）、线程布局（如 Warp 级并行）等，最终提供接近硬件理论峰值的性能。

### 总结

CuTe 与 CUTLASS 是 “**底层基础**” 与 “**高层应用**” 的关系：

- CuTe 提供 “通用张量操作能力”，解决 “如何描述和操作任意维度、布局的张量”；
- CUTLASS 基于 CuTe，解决 “如何用这些能力高效实现线性代数运算（如 GEMM、卷积）”。

对于开发者而言：

- 若需直接使用高性能线性代数运算（如调用矩阵乘法），直接用 CUTLASS 即可；
- 若需定制特殊张量操作（如非标准布局的矩阵运算、融合了自定义逻辑的线性代数核），则需要深入了解 CuTe 的张量抽象。

# CUTE 核心

要深入理解 **CuTe**（CUDA Tensor Expression），我们需要从其核心设计理念出发：**通过 DSL（Domain-Specific Language）抽象张量操作**，简化多维张量的建模、内存管理、数据搬运和计算。以下是 CuTe 的关键概念和学习路径：

---

### **一、CuTe 的核心概念**

#### **1. 张量建模：Shape 与 Stride**

CuTe 使用 `Shape` 和 `Stride` 来定义张量的维度和内存布局：

- **Shape**：描述张量的维度（如 `Shape<_128, _64, _32>` 表示 3D 张量）。
- **Stride**：描述每个维度的步长（如 `Stride<_1, _128, _128*64>` 表示列优先布局）。
- **示例**：

    cpp

    深色版本

- ```
    using Layout = cutlass::layout::ColumnMajor;  // 列优先布局
    using Tensor = cute::Tensor<float, Layout<Shape<_128, _64>>>;
    ```

#### **2. 内存管理：多级内存抽象**

CuTe 支持多级内存（GMEM、SMEM、Register）的统一抽象：

- **GMEM**（Global Memory）：全局内存，用于存储大张量。
- **SMEM**（Shared Memory）：共享内存，用于线程块内暂存数据。
- **Register**：寄存器内存，用于线程级计算。
- **TMA**（Tensor Memory Accelerator）：Hopper 架构引入的硬件加速器，优化数据搬运。
- **示例**：

    cpp

    深色版本

- ```
    using MemLayout = cutlass::layout::TensorOpMultiplicand<…>;  // 自定义布局
    ```

#### **3. 数据搬运：Tiling 与 Partitioning**

CuTe 提供了灵活的数据搬运策略：

- **Tiling**：将张量拆分为小块（Tile），匹配 GPU 计算单元。
- **Partitioning**：将数据分配到线程或线程束（Warp）。
- **示例**：

    cpp

    深色版本

- ```
    using TiledCopy = cute::TiledCopy<…>;  // 定义数据搬运策略
    ```

#### **4. 计算：通用张量运算**

CuTe 支持通用张量运算（如 GEMM、卷积、广播）：

- **元素级操作**：逐元素加法、乘法。
- **广播**：低维张量扩展到高维。
- **收缩**：张量收缩（如矩阵乘法）。
- **示例**：

    cpp

    深色版本

- ```
    auto result = cute::elementwise(op)(A, B);  // 逐元素操作
    ```

---

### **二、CuTe 的核心接口与设计模式**

#### **1. 张量布局（Layout）**

- **RowMajor/ColumnMajor**：行优先/列优先布局。
- **TensorNHWC/TensorNCHW**：深度学习中常见的张量布局。
- **自定义布局**：通过 `Shape` 和 `Stride` 定义复杂布局。
- **示例**：

    cpp

    深色版本

- ```
    using Layout = cutlass::layout::TensorNHWC<Shape<_N, _H, _W, _C>>;
    ```

#### **2. 数据搬运（TiledCopy）**

- **GMEM ↔ SMEM**：通过 `TiledCopy` 实现高效的数据搬运。
- **SMEM ↔ Register**：利用 `Partitioning` 将数据分配到寄存器。
- **示例**：

    cpp

    深色版本

- ```
    using Copy = cute::TiledCopy<…>;  // 定义数据搬运策略
    ```

#### **3. 计算内核（CollectiveBuilder）**

CuTe 提供了 `CollectiveBuilder` 用于构建自定义计算内核：

- **GEMM**：矩阵乘法（支持 Tensor Core 加速）。
- **Convolution**：卷积操作（隐式 GEMM）。
- **示例**：

    cpp

    深色版本

- ```
    using Collective = cute::CollectiveBuilder<…>;  // 构建计算内核
    ```

---

### **三、学习路径与实践建议**

#### **阶段 1：掌握 CuTe 的核心抽象**

1. **熟悉 Shape 与 Stride**：
    - 学习如何定义多维张量的布局。
    - 实践：修改 `Shape` 和 `Stride` 参数，观察内存布局变化。
2. **理解多级内存模型**：
    - 掌握 GMEM、SMEM、Register 的用途。
    - 实践：通过 `cutlass_profiler` 分析不同内存层次的性能差异。

#### **阶段 2：数据搬运与 Tiling**

1. **Tiling 策略**：
    - 学习如何拆分张量为 Tile，匹配 GPU 计算单元。
    - 实践：调整 Tile Size，观察对性能的影响。
2. **Partitioning**：
    - 学习如何将数据分配到线程或线程束。
    - 实践：使用 `cute::Partition` 工具优化线程分配。

#### **阶段 3：通用张量运算**

1. **元素级操作**：
    - 实现逐元素加法、乘法。
    - 实践：使用 `cute::elementwise` 编写自定义操作。
2. **广播与收缩**：
    - 实现广播（Broadcast）和张量收缩（Contraction）。
    - 实践：结合 CuTe 的 DSL 实现矩阵乘法。

#### **阶段 4：高性能内核开发**

1. **GEMM 内核**：
    - 使用 `CollectiveBuilder` 构建自定义 GEMM 内核。
    - 实践：优化 Tile Size 和数据搬运策略。
2. **卷积内核**：
    - 实现隐式 GEMM 卷积。
    - 实践：结合 CuTe 的布局转换优化卷积性能。

---

### **四、资源推荐**

#### **1. 官方文档与示例**

- **CuTe 文档**：CUTLASS 3.x CuTe 文档
- **示例代码**：CUTLASS 3.x 的 `tutorial` 文件夹中的完整示例（如 `tiled_copy.cu`）。

#### **2. 实践项目**

- **CUTLASS 3.x 示例**：运行 `examples/` 和 `test/` 文件夹中的代码，理解 CuTe 的实际应用。
- **自定义内核**：使用 `CollectiveBuilder` 构建自己的 GEMM 或卷积内核。

#### **3. 社区与工具**

- **NVIDIA 开发者论坛**：CUTLASS 讨论区
- **性能分析工具**：`Nsight Systems`、`nvprof` 分析内核性能瓶颈。

---

### **五、总结**

CuTe 通过 **DSL 抽象** 简化了张量操作，使开发者能够专注于算法设计而非底层实现。掌握其核心概念（如 Shape、Stride、Tiling）和接口（如 `CollectiveBuilder`）是高效开发的关键。通过实践和调优，可以充分发挥 CuTe 在 GPU 加速计算中的性能潜力。

# 张量支持

NVIDIA 为 CUDA 提供了多个库和 API，用于支持张量操作和其他高性能计算任务。以下是一些主要的库：

1. **cuTENSOR**：

- NVIDIA cuTENSOR 是一个用于张量运算的高性能库。它提供了对一维、二维、三维以及更高维度张量的高效处理函数，包括张量收缩（tensor contraction）、元素级操作（element-wise operations）、广播（broadcasting）等。
- 它旨在简化复杂张量操作的实现，并且优化以利用最新的 GPU 架构。

1. **cuBLAS**：

- cuBLAS 是一个提供 BLAS（Basic Linear Algebra Subprograms）接口的 GPU 加速库。虽然它主要是为矩阵运算设计的，但是由于张量可以被视为多维数组或一系列矩阵，因此对于某些张量操作来说，cuBLAS 也可以是一个有用的工具。

1. **cuSPARSE**：

- 对于稀疏张量的操作，cuSPARSE 库提供了针对稀疏矩阵的线性代数子程序，这些子程序同样可以在处理具有稀疏结构的张量时发挥作用。

1. **TensorRT**：

- 虽然 TensorRT 更多地被认为是一个用于深度学习推理优化的 SDK，但它也包含了高度优化的张量操作实现，适用于构建高效的深度学习模型推理流程。

1. **CUDA C++ 编程指南中的底层 API**：

- 如果你需要更细粒度的控制或者特定的张量操作不在上述库的支持范围内，你可以直接使用 CUDA C++ 进行编程。通过编写自定义的 CUDA 内核，你可以实现任何所需的张量操作。这需要对 GPU 架构有深入的理解，但同时也提供了最大的灵活性。

选择哪个库或方法取决于你的具体需求，比如性能要求、是否需要支持稀疏数据结构、以及你希望投入的时间来开发和优化代码。对于大多数常见的张量操作，使用像 cuTENSOR 这样的高级库通常会更加高效和便捷。

## Tensor Core 支持

NVIDIA CUDA 提供了几种方式来编程 GPU Tensor Core，以实现高效的矩阵运算和深度学习应用。Tensor Cores 最初是在 Volta 架构中引入的，并在随后的架构如 Turing 和 Ampere 中得到了进一步增强。以下是几种主要的方式：

1. **CUDA Warp-Level Matrix Operations (WMMA)**：

- NVIDIA 在 CUDA 9.0 中引入了实验性的 Warp-Level Matrix Operations API（也称为 WMMA API），它允许开发者通过 CUDA C++ 直接利用 Tensor Cores 进行矩阵乘法和累加操作。
- 使用 WMMA API，你可以执行半精度浮点数（FP16）的矩阵乘法，并且可以得到全精度（FP32）的结果。不过需要注意的是，从 CUDA 11 开始，WMMA API 已经被标记为不推荐使用，建议转而使用新的 Cooperative Groups 或者直接使用 CUTLASS 库。

1. **cuBLAS**：

- cuBLAS 是一个包含 BLAS（Basic Linear Algebra Subprograms）功能的 GPU 加速库，支持 Tensor Core 操作。例如，`cublasGemmEx` 函数可以用来执行基于 Tensor Cores 的 GEMM（General Matrix Multiply）操作。
- 通过选择适当的计算类型（如 `CUBLAS_COMPUTE_16F` 对于 FP16 矩阵乘法），你可以在调用这些函数时启用 Tensor Cores。

1. **cuDNN**：

- cuDNN 是一个用于深度神经网络的 GPU 加速库，其中包含了对 Tensor Core 的优化。许多常见的深度学习层，如卷积层、全连接层等，在 cuDNN 中都经过了优化以利用 Tensor Cores 提高性能。
- 当你在构建深度学习模型时，如果底层框架（如 TensorFlow, PyTorch 等）使用了 cuDNN 来加速计算，则自动会受益于 Tensor Cores 的性能提升。

1. **CUTLASS (CUDA Templates for Linear Algebra Subroutines and Solvers)**：

- CUTLASS 是 NVIDIA 开发的一个开源库，提供了高效的线性代数子程序模板，专门设计用于充分利用 Tensor Cores 的能力。
- 它不仅支持基本的 GEMM 操作，还包括了更多复杂的张量操作，使得开发者能够编写高度优化的 CUDA 内核来利用 Tensor Cores。

1. **Direct CUDA Programming with Tensor Core Instructions**：

- 对于那些希望完全控制硬件特性的高级用户，可以直接在 CUDA 内核中使用 PTX（Parallel Thread Execution）指令集中的 Tensor Core 相关指令进行编程。这种方式虽然灵活，但需要深入了解 GPU 架构以及 PTX 编程。


