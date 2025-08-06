---
dateCreated: 2025-08-06
dateModified: 2025-08-06
---

参考：https://www.zhihu.com/question/451127498

# Tensor Core 支持

CUDA 提供了多种指令、API、接口和库，可以抽象 tensor/matrix 数据类型或直接使用 **Tensor Core** 进行高效的基于 tensor 的操作。以下是详细的分类和说明：

---

### **1. CUDA Libraries（函数库）**

这些库提供了高度优化的接口，直接支持 tensor/matrix 操作，并可能利用 Tensor Core 加速计算。

#### **(1) cuBLAS**
- **功能**：用于矩阵和向量运算的库，支持 BLAS 接口（如 GEMM 矩阵乘法）。
- **Tensor Core 支持**：
  - 在支持 Tensor Core 的 GPU 上，`cublasGemmEx()` 和 `cublasHgemm()` 等函数会自动利用 Tensor Core 加速 FP 16/FP 32/INT 8 计算。
  - 示例：`cublasGemmEx()` 可指定数据类型（如 `CUDA_R_16F`）和计算模式，自动调用 Tensor Core。
- **使用场景**：通用矩阵乘法、线性代数运算。

#### **(2) cuDNN**
- **功能**：深度神经网络加速库，提供卷积、池化、归一化等操作。
- **Tensor Core 支持**：
  - 卷积（Convolution）、激活（Activation）、归一化（Normalization）等操作在支持 Tensor Core 的 GPU 上会自动优化。
  - 示例：`cudnnConvolutionForward()` 会根据硬件自动选择是否使用 Tensor Core。
- **使用场景**：深度学习模型的前向/后向传播。

#### **(3) CUTLASS**
- **功能**：基于模板的 GEMM 库，支持自定义矩阵乘法（GEMM）和 Tensor Core 优化。
- **Tensor Core 支持**：
  - 提供 warp-level 和 thread-level 的矩阵乘法模板，可直接调用 `mma.sync` 指令。
  - 示例：通过 `cutlass::gemm::device::Gemm` 模板类配置 Tensor Core 的数据类型和矩阵分块。
- **使用场景**：需要自定义 GEMM 操作的高性能计算（如自定义深度学习层）。

#### **(4) cuFFT**
- **功能**：快速傅里叶变换（FFT）库。
- **Tensor Core 支持**：不直接依赖 Tensor Core，但可用于处理与 tensor 相关的频域计算。

#### **(5) cuSPARSE**
- **功能**：稀疏矩阵运算库，支持稀疏矩阵 - 向量乘法（SpMV）等。
- **Tensor Core 支持**：不直接依赖 Tensor Core，但可与其他库结合使用。

---

### **2. WMMA API（Warp Matrix Multiply-Accumulate API）**

WMMA 是 CUDA 提供的底层 API，允许开发者直接使用 Tensor Core 进行 warp-level 的矩阵乘法累加操作。

#### **核心步骤**
1. **定义矩阵分块（Fragment）**：
   - 使用 `wmma::fragment` 抽象矩阵块（tile），例如：

     ```cpp
     wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
     wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
     wmma::fragment<wmma::accumulator, M, N, K, float> acc_frag;
     ```

2. **加载数据到 Fragment**：
   - 使用 `__ldmatrix_sync` 将全局内存中的数据加载到寄存器。
3. **执行矩阵乘法**：
   - 使用 `wmma::mma_sync` 进行 `D = A * B + C` 操作：

     ```cpp
     wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
     ```

4. **存储结果**：
   - 使用 `__stmatrix_sync` 将结果写回共享内存或全局内存。

#### **使用场景**
- 需要精细控制 Tensor Core 的底层操作（如自定义 GEMM、混合精度计算）。
- 适用于 Volta/Turing/Ampere 架构的 GPU（需检查 GPU 是否支持 Tensor Core）。

---

### **3. CUDA Runtime API 和 Driver API**
#### **(1) CUDA Runtime API**
- **功能**：提供高级接口，简化内存管理、核函数调用等。
- **Tensor Core 支持**：
  - 通过调用 cuBLAS、cuDNN 等库间接使用 Tensor Core。
  - 示例：`cudaMalloc` 分配显存，`cudaMemcpy` 拷贝数据，`cudaLaunchKernel` 启动核函数。

#### **(2) CUDA Driver API**
- **功能**：更底层的接口，提供对 CUDA 硬件的直接控制。
- **Tensor Core 支持**：通常与 cuBLAS/cuDNN 结合使用，不直接操作 Tensor Core。

---

### **4. Tensor Core 的底层指令**
#### **(1) `mma.sync` 指令**
- **功能**：在 warp 内执行矩阵乘法累加（`D = A * B + C`）。
- **使用方式**：
  - 通过 WMMA API 或 CUTLASS 库调用。
  - 示例：

    ```cpp
    __half2 a[16][16], b[16][16], c[16][16];
    __mma_sync(c, a, b, c, 16, 16, 16);
    ```

- **要求**：
  - 矩阵分块尺寸必须符合 Tensor Core 的要求（如 16 x 16 x 16）。
  - 数据类型需匹配（如 FP 16、FP 32、INT 8）。

#### **(2) `__ldmatrix_sync` 和 `__stmatrix_sync`**
- **功能**：将数据从全局内存加载到寄存器，或将结果存储回内存。
- **示例**：

  ```cpp
  __half2 a[8][8];
  __ldmatrix_sync(a, global_memory_ptr, ...);
  ```

---

### **5. 其他相关库**
#### **(1) CUB**
- **功能**：并行算法库，提供排序、扫描、归约等操作。
- **Tensor Core 支持**：不直接涉及 Tensor Core，但可与矩阵操作结合使用。

#### **(2) Thrust**
- **功能**：高级并行算法库（类似 C++ STL）。
- **Tensor Core 支持**：不直接涉及 Tensor Core。

---

### **6. 使用 Tensor Core 的注意事项**
1. **GPU 架构要求**：
   - 必须使用支持 Tensor Core 的 GPU（Volta、Turing、Ampere、Ada 等架构）。
   - 示例：Tesla V 100、RTX 2080、A 100、H 100。
2. **数据类型要求**：
   - 支持的数据类型包括 FP 16、FP 32、INT 8 等。
3. **矩阵尺寸要求**：
   - 矩阵分块尺寸必须符合 Tensor Core 的限制（如 16 x 16 x 16）。
4. **编译器设置**：
   - 编译时需指定目标架构，例如：

     ```bash
     nvcc -arch=sm_70 my_code.cu  # sm_70 对应 Volta 架构
     ```

---

# 支持 Tensor Core 的接口详细对比

以下是 **cuBLAS**、**cuDNN**、**CUTLASS**、**WMMA API** 和 **MMA**（含 Hopper 架构的 WGMMA）的详细对比，涵盖实现方式、特点、用途及差异：

---

### **1. cuBLAS**
#### **定义与实现**
- **核心功能**：提供基础线性代数子程序（BLAS）的 GPU 加速实现，包括矩阵乘法（GEMM）、向量运算等。
- **实现方式**：
  - 基于 CUDA 内核，高度优化，利用 Tensor Core 的低精度/混合精度计算（如 FP16、FP32）。
  - 支持多 GPU 扩展（cuBLASMg）和并发执行（CUDA 流）。
- **特点**：
  - **标准化接口**：遵循 BLAS 标准 API，兼容性强。
  - **高性能优化**：针对 NVIDIA GPU 架构（Volta、Turing、Ampere）优化，自动适配新硬件。
  - **多精度支持**：支持 FP16、FP32、INT8 等数据类型。
  - **易用性**：开发者无需手动管理底层细节，直接调用 API 即可。
- **用途**：
  - 科学计算（如物理模拟、金融建模）。
  - 深度学习中的通用矩阵乘法（如全连接层）。
  - 高性能计算（HPC）中的线性代数运算。

#### **代码示例**

```cpp
cublasHandle_t handle;
cublasCreate(&handle);
float alpha = 1.0f, beta = 0.0f;
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
```

---

### **2. cuDNN**
#### **定义与实现**
- **核心功能**：为深度学习提供高度优化的卷积、池化、归一化等操作。
- **实现方式**：
  - 基于 cuBLAS 和 Tensor Core，针对深度学习模型（如 CNN、Transformer）设计。
  - 提供自动算法选择（`cudnnGetConvolutionForwardAlgorithm`）和混合精度支持。
- **特点**：
  - **深度学习专用**：优化卷积、注意力机制、矩阵乘法（如 `matmul`）等常见操作。
  - **灵活性**：支持多种数据布局（NCHW、NHWC）和自定义网络结构。
  - **易集成**：与 PyTorch、TensorFlow 等框架无缝衔接。
  - **性能调优**：通过 `cudnn.benchmark` 自动选择最优算法。
- **用途**：
  - 深度学习模型训练与推理（如图像分类、目标检测）。
  - 自定义网络层的加速（如自定义卷积核）。

#### **代码示例**

```python
import torch
from torch.backends import cudnn
cudnn.benchmark = True  # 自动选择最优算法
model = torch.nn.Linear(1000, 1000).cuda()
input = torch.randn(100, 1000).cuda()
output = model(input)  # 自动调用 cuDNN 优化的 GEMM
```

---

### **3. CUTLASS**
#### **定义与实现**
- **核心功能**：基于模板的 CUDA 线性代数库，支持自定义 GEMM 内核开发。
- **实现方式**：
  - 提供模板化 API（如 `cutlass::gemm::device::Gemm`），允许开发者自定义矩阵分块、数据布局、精度等。
  - 支持 **CollectiveMma** 和 **Epilogue** 抽象，实现操作融合（如激活函数与 GEMM 融合）。
  - 与 CuTe 布局结合，支持复杂数据排列（如分层张量布局）。
- **特点**：
  - **高度可定制**：通过模板参数控制内核行为（如分块尺寸、数据类型）。
  - **操作融合**：减少内存访问，提升性能（如 GEMM + ReLU 融合）。
  - **跨架构兼容**：支持 Volta、Turing、Ampere 等架构。
- **用途**：
  - 自定义深度学习算子（如稀疏矩阵乘法、混合精度训练）。
  - 高性能计算中的复杂矩阵运算（如非标准分块）。

#### **代码示例**

```cpp
using Gemm = cutlass::gemm::device::Gemm<
  float, cutlass::layout::RowMajor,
  float, cutlass::layout::ColumnMajor,
  float, cutlass::layout::RowMajor,
  float>;
Gemm gemm_op;
gemm_op.initialize(M, N, K, A, B, C);
gemm_op();
```

---

### **4. WMMA API**
#### **定义与实现**
- **核心功能**：提供 warp-level 的矩阵乘法累加操作（`D = A * B + C`），封装 Tensor Core 的底层指令。
- **实现方式**：
  - 通过 `wmma::fragment` 抽象矩阵分块（如 `16x16x16`）。
  - 使用 `load_matrix_sync`、`mma_sync`、`store_matrix_sync` 管理数据加载、计算和存储。
- **特点**：
  - **高级抽象**：隐藏 Tensor Core 的寄存器管理和数据流细节。
  - **易用性**：适合快速实现标准 GEMM，但灵活性有限。
  - **性能**：接近硬件上限，但需合理设计分块和内存布局。
- **用途**：
  - 快速实现标准矩阵乘法（如 GEMM）。
  - 深度学习框架中的底层加速（如 cuDNN 内部调用 WMMA）。

#### **代码示例**

```cpp
using namespace wmma;
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> acc_frag;

load_matrix_sync(a_frag, A, K, row_major);
load_matrix_sync(b_frag, B, K, col_major);
mma_sync(acc_frag, a_frag, b_frag, acc_frag);
store_matrix_sync(C, acc_frag, N, row_major);
```

---

### **5. MMA（含 Hopper 的 WGMMA）**
#### **定义与实现**
- **核心功能**：底层 PTX 指令，直接操作 Tensor Core 的寄存器和硬件资源。
- **实现方式**：
  - 使用 `__ldmatrix_sync`、`__mma_sync`、`__stmatrix_sync` 等指令手动管理数据流。
  - **Hopper 的 WGMMA**：支持异步计算、共享内存直接读取，进一步提升性能。
- **特点**：
  - **极低抽象层级**：需手动管理寄存器、数据加载/存储和分块策略。
  - **极致灵活性**：可自定义非标准分块（如 `8x8x16`）和数据布局。
  - **性能潜力大**：适合深度定制化优化，但开发难度高。
- **用途**：
  - 自定义算法融合（如 GEMM + 归约）。
  - 非标准矩阵操作（如稀疏矩阵乘法）。
  - Hopper 架构下的高性能计算（如实时 AI 推理）。

#### **代码示例**

```cpp
__half2 a[8][8], b[8][8], c[8][8];
__ldmatrix_sync(a, global_memory_ptr_a, …); // 手动加载数据
__ldmatrix_sync(b, global_memory_ptr_b, …);
__mma_sync(c, a, b, c, 8, 8, 8); // 执行矩阵乘法
__stmatrix_sync(global_memory_ptr_c, c, …); // 存储结果
```

---

### **6. 对比总结**

| **特性**               | **cuBLAS**                          | **cuDNN**                          | **CUTLASS**                        | **WMMA API**                       | **MMA/WGMMA**                      |
|------------------------|-------------------------------------|-------------------------------------|------------------------------------|------------------------------------|------------------------------------|
| **抽象层级**           | 高（BLAS 标准接口）                 | 高（深度学习专用接口）              | 中（模板化 API）                   | 高（封装 fragment）                | 低（PTX 指令）                     |
| **灵活性**             | 低（固定接口）                      | 中（支持自定义网络结构）            | 高（模板参数控制）                 | 低（固定分块和布局）               | 极高（完全自定义）                 |
| **性能优化潜力**       | 高（自动优化）                      | 高（自动算法选择）                  | 高（可深度定制）                   | 中（依赖开发者设计）               | 极高（手动优化）                   |
| **开发难度**           | 低（直接调用 API）                  | 低（框架集成）                      | 中（模板编程）                     | 低（简化 fragment 操作）           | 高（需手动管理寄存器和内存）       |
| **适用场景**           | 通用线性代数（科学计算、HPC）| 深度学习模型（CNN、Transformer）| 自定义算子（稀疏计算、操作融合）| 快速实现 GEMM（如深度学习框架后端）| 自定义算法、非标准矩阵操作         |
| **数据类型支持**       | FP16/FP32/INT8                      | FP16/FP32/INT8                      | FP16/FP32/INT8                     | FP16/FP32                          | FP16/FP32/INT8                     |
| **架构支持**           | Volta+                              | Volta+                              | Volta+                             | Volta+                             | Hopper（WGMMA 异步计算）|

---

### **7. 选择建议**
- **使用 cuBLAS**：
  - 如果需要标准线性代数操作（如 GEMM），且希望快速实现。
  - 适用于科学计算和通用高性能计算。
- **使用 cuDNN**：
  - 如果开发深度学习模型（如 CNN、Transformer），且希望自动优化。
  - 与 PyTorch/TensorFlow 等框架无缝集成。
- **使用 CUTLASS**：
  - 如果需要自定义 GEMM 内核（如操作融合、非标准分块）。
  - 适用于深度学习自定义算子和高性能计算。
- **使用 WMMA API**：
  - 如果需要快速实现 warp-level 矩阵乘法，但不想手动管理底层细节。
  - 适合深度学习框架后端开发（如 cuDNN 内部调用）。
- **使用 MMA/WGMMA**：
  - 如果需要极致性能优化（如异步计算、共享内存直接读取）。
  - 适合 Hopper 架构下的实时 AI 推理和非标准矩阵操作。

---

### **8. 未来趋势**
- **Hopper 的 WGMMA**：通过异步计算和共享内存直接读取，进一步降低延迟，提升性能。
- **库与框架的深度集成**：cuBLAS/cuDNN/CUTLASS 将更紧密地与 PyTorch/TensorFlow 结合，推动实时 AI 应用（如自动驾驶、医疗影像）。
- **自动代码生成**：工具链（如 Triton、TVM）可能自动生成 MMA/WGMMA 代码，降低开发门槛。
