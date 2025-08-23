---
dateCreated: 2025-08-06
dateModified: 2025-08-11
---

参考：https://zhuanlan.zhihu.com/p/1906775725278737888

# MMA 使用

根据 NVIDIA 官方的最新技术路线图和开发者文档，**MMA（Matrix Multiply-Accumulate）和 WMMA（Warp Matrix Multiply-Accumulate）** 的地位和使用方式已经发生了显著变化，以下是详细分析：

---

### **1. MMA 与 WMMA 的现状**
#### **(1) MMA（Tensor Core 指令）**
- **核心地位**：
  MMA 是 NVIDIA Tensor Core 的底层指令，**仍然是当前及未来 GPU 架构（如 Hopper、Ada）的核心计算单元**。
  - **Hopper 架构**（H 100）引入了 **FP 8 和 BF 16 支持**，进一步扩展了 MMA 的适用范围。
  - **Ada 架构**（RTX 40 系列）增强了对 **DLSS 3 和光线追踪** 的支持，MMA 在 AI 推理和图形计算中仍然关键。
- **推荐使用方式**：
  - **通过 CUTLASS 3. x 或 cuTENSOR**：NVIDIA 推荐使用更高层的库（如 CUTLASS 3. x）来封装 MMA 操作，而非直接调用底层指令。
  - **FP 8 支持**：Hopper 的 MMA 指令支持 FP 8，能显著提升大模型训练和推理的吞吐量（如 LLM 和扩散模型）。

#### **(2) WMMA（Warp-Level Matrix Multiply-Accumulate）**
- **历史背景**：
  - **Volta/Turing 架构**（如 V 100、T 4）中，WMMA 是 Warp 级的矩阵乘法指令，用于加速小规模矩阵运算（如 GEMV）。
  - **局限性**：WMMA 的 tile 大小固定（如 16 x 16 x 16），灵活性较低，且不支持最新的 FP 8 数据类型。
- **当前状态**：
  - **NVIDIA 已逐步弃用 WMMA**，推荐开发者使用 **CUTLASS 2. x/3. x** 或 **Triton** 来替代。
  - **新架构（如 Hopper）不再支持 WMMA**，仅保留对旧架构的兼容性。

---

### **2. 是否可以继续使用 MMA 和 WMMA？**
#### **(1) MMA 可以继续使用，但需升级方式**
- **推荐做法**：
  - **使用 CUTLASS 3. x + CuTe**：通过 `cutlass::gemm::threadblock::Mma` 和 `cuTe::Shape` 抽象分块逻辑，自动适配不同架构（如 Hopper 的 FP 8）。
  - **FP 8 支持**：在 Hopper 上利用 MMA 的 FP 8 指令，显著提升大模型训练效率（如 Mamba-MoE 架构）。
  - **cuTENSOR**：对于张量计算（如卷积、矩阵分解），使用 cuTENSOR 库可自动优化 MMA 调度。
- **示例代码（CUTLASS 3. x）**：

  ```cpp
  using MMA = cutlass::gemm::threadblock::Mma<
      cutlass::gemm::GemmShape<128, 128, 32>,  // Tile shape
      cutlass::gemm::GemmShape<32, 32, 8>,     // Warp shape
      cutlass::gemm::GemmShape<16, 16, 4>,     // Instruction shape (MMA micro-tile)
      float, float, float,                    // Data types
      cutlass::arch::OpClassTensorOp,         // Use Tensor Core
      cutlass::arch::Sm90                       // Target Hopper architecture
  >;
  ```

#### **(2) WMMA 不推荐继续使用**
- **原因**：
  - **性能瓶颈**：WMMA 的 tile 大小固定（如 8 x 8 x 4），无法适配现代大模型的计算需求（如 128 x 128 x 32 的 tile）。
  - **缺乏新特性支持**：不支持 FP 8、混合精度等新数据类型。
  - **兼容性问题**：Hopper 架构已移除对 WMMA 的支持。
- **替代方案**：
  - **Triton**：通过 Python 编写内核，自动调度 MMA 指令（如 [Triton 的矩阵乘法](https://triton-lang.org/main/getting-started/tutorials/01-matrix-multiplication.html)）。
  - **CUTLASS 3. x**：通过模板化配置生成高效的 MMA 内核。

---

### **3. 更高效的方法：NVIDIA 的推荐方向**
#### **(1) 使用 CUTLASS 3. X + CuTe**
- **优势**：
  - **多级分块抽象**：通过 `cuTe::Shape` 和 `cuTe::Stride` 定义灵活的 tile 布局，适配不同硬件（如 Hopper 的 16 x 16 x 4 微块）。
  - **自动优化**：CUTLASS 3. x 能根据硬件特性（如寄存器数、共享内存大小）自动调整 tile 大小。
  - **支持 FP 8/BF 16**：Hopper 的 FP 8 MMA 指令可显著降低内存带宽需求（如 LLM 推理中每个 token 的计算量减少 50%）。
- **示例（FP 8 支持）**：

  ```cpp
  using ElementA = cutlass::float8_t;
  using ElementB = cutlass::float8_t;
  using ElementC = float;
  using MMA = cutlass::gemm::threadblock::Mma<
      cutlass::gemm::GemmShape<128, 128, 32>,
      cutlass::gemm::GemmShape<32, 32, 8>,
      cutlass::gemm::GemmShape<16, 16, 4>,
      ElementA, ElementB, ElementC,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm90
  >;
  ```

#### **(2) 使用 Triton**
- **优势**：
  - **Python 编写，C++ 性能**：通过高级语言定义计算逻辑，Triton 自动编译为高效的 CUDA 代码。
  - **动态 tile 分块**：支持运行时调整 tile 大小，适应不同输入维度。
  - **集成 FP 8 支持**：Triton 2.0 已支持 Hopper 的 FP 8 MMA 指令。
- **示例（Triton 矩阵乘法）**：

  ```python
  import triton
  import triton.language as tl

  @triton.jit
  def matmul_kernel(A, B, C, M, N, K, BLOCK_SIZE: tl.constexpr):
      pid = tl.program_id(0)
      num_pid_m = tl.cdiv(M, BLOCK_SIZE)
      num_pid_n = tl.cdiv(N, BLOCK_SIZE)
      pid_m = pid // num_pid_n
      pid_n = pid % num_pid_n
      rm = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
      rn = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
      rk = tl.arange(0, BLOCK_SIZE)
      # Load A and B tiles
      a = tl.load(A + rm[:, None] * K + rk[None, :])
      b = tl.load(B + rk[:, None] * N + rn[None, :])
      # Compute C tile
      c = tl.dot(a, b)
      # Store C tile
      tl.store(C + rm[:, None] * N + rn[None, :], c)
  ```

#### **(3) 使用 cuTENSOR**
- **适用场景**：
  - **张量分解**（如 Tucker 分解）、**卷积**、**张量核函数**。
  - cuTENSOR 会自动选择最优的 MMA 配置（如 tile 大小、数据布局）。
- **示例**：

  ```cpp
  #include <cute/tensor.hpp>
  #include <cute/algorithm/gemm.hpp>

  using namespace cute;
  using LayoutA = Layout<Shape<_128, _128>, Stride<_128, _1>>;
  using LayoutB = Layout<Shape<_128, _128>, Stride<_1, _128>>;
  Tensor A = make_tensor<LayoutA>(...);
  Tensor B = make_tensor<LayoutB>(...);
  Tensor C = gemm(A, B);  // 自动调用 MMA 指令
  ```

---

### **4. 总结：是否继续使用 MMA/WMMA？**

| **技术** | **是否推荐** | **原因** | **替代方案** |
|----------|--------------|----------|--------------|
| **MMA**（Tensor Core） | ✅ **推荐** | 现代 GPU 架构的核心指令，支持 FP 8/BF 16。 | 通过 CUTLASS 3. x、Triton 或 cuTENSOR 使用。 |
| **WMMA**（Warp-Level） | ❌ **不推荐** | 固定 tile 大小，性能落后于 MMA，Hopper 不再支持。 | 使用 CUTLASS 3. x 或 Triton 替代。 |

---

### **5. 最佳实践建议**
1. **选择目标架构**：
   - 如果使用 **Hopper（H 100）**，优先使用 **FP 8 MMA** 和 **CUTLASS 3. x**。
   - 如果使用 **Ampere（A 100）**，可继续使用 **FP 16 MMA**，但避免 WMMA。

2. **避免手动调用底层指令**：
   - 通过 **CUTLASS 3. x** 或 **Triton** 抽象分块逻辑，自动适配硬件特性（如寄存器数、共享内存大小）。

3. **关注 NVIDIA 官方文档**：
   - [CUTLASS 3.x 文档](https://docs.nvidia.com/cuda/cutlass/)
   - [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/01-matrix-multiplication.html)
   - [Hopper 架构白皮书](https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/ai-enterprise/hopper-architecture-whitepaper.pdf)

通过上述方法，您可以充分利用 NVIDIA 最新的硬件特性（如 FP 8 和 Hopper MMA），同时避免因使用过时技术（如 WMMA）导致的性能瓶颈。

# WMMA 和 MMA 对比

在 CUDA 中，**WMMA**（Warp Matrix Multiply-Accumulate）和 **MMA**（Matrix Multiply-Accumulate）是两种不同的接口/指令，均用于利用 GPU 的 **Tensor Core** 进行高效的矩阵计算。它们的主要区别在于抽象层级、灵活性和适用场景。以下是详细对比：

---

### **1. WMMA（Warp Matrix Multiply-Accumulate API）**
#### **定义**

WMMA 是 NVIDIA 提供的 **高级抽象 API**，封装了 Tensor Core 的底层操作（如数据加载、矩阵乘法、结果存储），简化了开发者对 Tensor Core 的使用。它通过 warp-level 的 fragment 操作，提供更简洁的编程接口。

#### **核心功能**
- **数据加载**：`load_matrix_sync`（从全局内存或共享内存加载数据到 fragment）。
- **矩阵乘法**：`mma_sync`（执行 `D = A * B + C` 操作）。
- **结果存储**：`store_matrix_sync`（将结果写回全局内存或共享内存）。
- **Fragment 抽象**：通过 `wmma::fragment` 定义矩阵分块（如 `16x16x16`）。

#### **特点**
- **高抽象层级**：隐藏了底层寄存器管理和数据加载/存储的细节。
- **易用性**：开发者只需调用 API 函数，无需手动编写 PTX 指令。
- **适用场景**：适合快速实现 GEMM（通用矩阵乘法）等标准操作，如深度学习中的卷积、全连接层等。

#### **示例代码**

```cpp
using namespace wmma;
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> acc_frag;

load_matrix_sync(a_frag, A_global, K, row_major);
load_matrix_sync(b_frag, B_global, K, col_major);
mma_sync(acc_frag, a_frag, b_frag, acc_frag);
store_matrix_sync(C_global, acc_frag, N, row_major);
```

---

### **2. MMA（Matrix Multiply-Accumulate PTX 指令）**
#### **定义**

MMA 是 **底层 PTX 指令**，直接调用 GPU 的 Tensor Core 硬件资源，提供更细粒度的控制。开发者需要手动管理数据加载、矩阵分块和寄存器分配。

#### **核心功能**
- **数据加载**：`__ldmatrix_sync`（从全局内存或共享内存加载数据到寄存器）。
- **矩阵乘法**：`__mma_sync`（执行 `D = A * B + C` 操作）。
- **结果存储**：`__stmatrix_sync`（将结果写回内存）。
- **线程级控制**：每个线程需明确负责的数据和计算任务。

#### **特点**
- **低抽象层级**：直接操作硬件资源，需手动管理数据布局和寄存器。
- **灵活性高**：可自定义数据加载/存储方式、矩阵分块策略（如非标准分块尺寸）。
- **适用场景**：适合需要极致性能优化的场景（如自定义算法融合、非标准矩阵尺寸）。

#### **示例代码**

```cpp
__half2 a[8][8], b[8][8], c[8][8];
__ldmatrix_sync(a, global_memory_ptr_a, …); // 手动加载数据
__ldmatrix_sync(b, global_memory_ptr_b, …);
__mma_sync(c, a, b, c, 8, 8, 8); // 执行矩阵乘法
__stmatrix_sync(global_memory_ptr_c, c, …); // 存储结果
```

---

### **3. 核心区别对比**

| **特性**               | **WMMA**                                | **MMA**                                 |
|------------------------|-----------------------------------------|-----------------------------------------|
| **抽象层级**           | 高级 API（封装底层细节）| 低级 PTX 指令（直接操作硬件）|
| **编程复杂度**         | 低（只需调用函数）| 高（需手动管理数据和寄存器）|
| **灵活性**             | 低（固定分块和布局）| 高（可自定义分块和数据流）|
| **性能优化潜力**       | 一般（依赖库优化）| 高（可深度定制）|
| **适用场景**           | 标准矩阵乘法（如 GEMM）| 自定义算法、非标准矩阵操作              |
| **架构支持**           | Volta/Turing/Ampere/Hopper              | Volta/Turing/Ampere/Hopper              |
| **典型用途**           | 深度学习框架（如 cuBLAS、cuDNN）| 高性能计算、自定义内核开发              |

---

### **4. 使用建议**
- **使用 WMMA**：
  - 如果你需要快速实现标准矩阵乘法（如 GEMM）。
  - 开发者希望简化代码逻辑，避免底层细节（如寄存器管理）。
  - 适用于大多数深度学习框架（如 TensorFlow、PyTorch）的后端实现。
- **使用 MMA**：
  - 如果你需要极致性能优化（如自定义数据布局、算法融合）。
  - 需要处理非标准矩阵尺寸或特殊数据流（如稀疏矩阵）。
  - 开发者熟悉 PTX 指令和 GPU 架构（如 SM 架构、寄存器分配）。

---

### **5. Hopper 架构的 WGMMA**

在 **Hopper 架构**（NVIDIA H100）中，NVIDIA 引入了 **Warpgroup Matrix Multiply-Accumulate (WGMMA)**，进一步扩展了 MMA 的能力：

- **异步计算**：支持异步执行，减少指令流水线阻塞。
- **直接读取共享内存**：无需先加载到寄存器，可直接从共享内存（SMEM）进行计算。
- **性能提升**：通过 Warpgroup 并行化，接近 Tensor Core 的理论算力上限。

---

### **总结**
- **WMMA** 是 **高级抽象 API**，适合快速实现标准矩阵操作，简化开发流程。
- **MMA** 是 **底层 PTX 指令**，适合需要极致性能和灵活性的场景。
- **Hopper 的 WGMMA** 在 MMA 基础上进一步优化，支持异步计算和共享内存直接读取，是未来高性能计算的首选。

# WMMA
### **WMMA API 详解**

**WMMA（Warp Matrix Multiply-Accumulate）** 是 NVIDIA 提供的底层 API，允许开发者直接使用 **Tensor Core** 进行 warp-level 的矩阵乘法累加操作（`D = A * B + C`）。WMMA 是 CUDA 提供的高层 C++ 接口，封装了 Tensor Core 的底层指令（如 `mma.sync`），专门用于在 warp 级别高效执行矩阵乘加操作。
它通过抽象矩阵分块（tile）、数据加载/存储和计算指令，帮助开发者高效利用 Tensor Core 的硬件加速能力。它是实现高性能矩阵运算（如 GEMM、卷积）的基础，比直接操作 PTX 指令更易用，同时保留了接近硬件峰值的性能。

---

### **1. 核心组件**

WMMA API 定义在 `nvcuda::wmma` 命名空间中，围绕 “**warp 级矩阵块处理**” 设计（一个 warp 包含 32 个线程，共同协作处理一个矩阵块）。核心组件包括：

- **Fragment**：矩阵分块的数据结构，用于存储矩阵 A、B 和累加器（accumulator）。
- **Load/Store 操作**：从全局内存或共享内存加载数据到 fragment，或存储结果。
- **Matrix Multiply-Accumulate (MMA)**：执行矩阵乘法累加操作。

#### 片段（Fragment）：矩阵数据的容器

Fragment 是 WMMA 的核心数据结构，它抽象了矩阵的分块（tile），并封装了 Tensor Core 的寄存器布局，代表一个由 warp 处理的矩阵块（或其部分），是 Tensor Core 可直接操作的数据格式。

“片段” 是 WMMA 中最核心的概念片段的类型由以下属性定义：

- **矩阵角色**：`matrix_a`（输入矩阵 A）、`matrix_b`（输入矩阵 B）、`accumulator`（累加矩阵 C/D）。
- **尺寸**：支持固定的小矩阵尺寸（如 16x16x16、32x8x16 等，随 GPU 架构扩展），对应 Tensor Core 的硬件处理单元。
- **数据类型**：输入矩阵（A/B）支持 `half`（fp16）、`nv_bfloat16`（bf16）、`int8` 等；累加矩阵（C/D）通常为 `float`（fp32）或 `int32`，保证计算精度。
- **布局**：内存中的数据布局（`row_major` 行优先 / `col_major` 列优先）。

片段类型需通过模板参数显式定义，例如：

```cpp
// 定义A矩阵的片段：16x16x16，fp16，行优先
using FragA = nvcuda::wmma::fragment<
nvcuda::wmma::matrix_a,    // 角色：A矩阵
16, 16, 16,                // 尺寸：m=16, n=16, k=16（A为m×k，B为k×n，C为m×n）
half,                      // 数据类型：fp16
nvcuda::wmma::row_major    // 内存布局
> ;

// 定义累加器片段：16x16x16，fp32（累加结果）
using FragC = nvcuda::wmma::fragment<
nvcuda::wmma::accumulator, 
16, 16, 16, 
float
> ;
```

常见的 fragment 类型包括：

- `wmma::fragment<wmma::matrix_a, M, N, K, T, Layout>`：矩阵 A 的分块（row-major）。
- `wmma::fragment<wmma::matrix_b, M, N, K, T, Layout>`：矩阵 B 的分块（col-major）。
- `wmma::fragment<wmma::accumulator, M, N, K, T>`：累加器（结果矩阵 C 的分块）。

**参数说明**：
- `M, N, K`：矩阵的维度（必须符合 Tensor Core 的约束，如 16x16x16）。
- `T`：数据类型（如 `__half`, `float`）。
- `Layout`：内存布局（`row_major` 或 `col_major`）。

**示例**：

```cpp
using namespace wmma;
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> acc_frag;
```

#### 数据加载：`load_matrix_sync`

将全局内存或共享内存中的矩阵数据加载到 fragment 中，自动处理内存布局到片段格式的转换（如对齐、维度适配）。

**函数原型**：

```cpp
template <typename FragmentType>
void load_matrix_sync(
FragmentType& frag,        // 输出：加载到的片段
const typename FragmentType::element_type* ptr,  // 输入：内存中矩阵的起始地址
int ldm,                   // 输入：矩阵的领先维度（内存中每行/列的实际长度）
nvcuda::wmma::layout_t layout = FragmentType::layout  // 输入：内存布局
);
```

 - 支持 `row_major` 或 `col_major` 布局。
 - **示例**：从全局内存加载 A 矩阵到片段：

```cpp
half* A_global;  // 全局内存中A矩阵（16x16，行优先）的起始地址
FragA a_frag;    // A矩阵的片段

// 加载A矩阵到片段（ldm=16表示A矩阵每行实际长度为16）
nvcuda::wmma::load_matrix_sync(a_frag, A_global, 16);

load_matrix_sync(a_frag, A_global, K, row_major);
load_matrix_sync(b_frag, B_global, K, col_major);
```

#### 数据存储：`store_matrix_sync`

将片段中的计算结果（如累加后的矩阵 D）存储回全局内存或共享内存，转换为常规布局（如行优先）。

- **函数原型**：

```cpp
template <typename FragmentType>
void store_matrix_sync(
typename FragmentType::element_type* ptr,  // 输出：存储到的内存地址
const FragmentType& frag,                  // 输入：要存储的片段
int ldm,                                   // 输入：矩阵的领先维度
nvcuda::wmma::layout_t layout = FragmentType::layout  // 输入：内存布局
);
```

- **`store_matrix_sync`**：将 fragment 中的结果存储回全局内存或共享内存。
- **示例**：将累加器片段存储到全局内存：

```cpp
float* D_global;  // 全局内存中存储结果D的地址
FragC d_frag;     // 计算完成的累加器片段

// 存储片段到全局内存（行优先，ldm=16）
nvcuda::wmma::store_matrix_sync(D_global, d_frag, 16);
store_matrix_sync(C_global, acc_frag, N, row_major);
```

#### 核心乘加：`mma_sync`

WMMA 的核心函数，执行矩阵块乘加操作：C=A×B+C，其中 A、B 是输入片段，C 是累加片段（输入输出两用）。

- **`mma_sync`**：执行 `acc = A * B + acc` 操作。
- **函数原型**：

```cpp
template <typename AccumulatorFragment, typename MatrixAFragment, typename MatrixBFragment>
void mma_sync(
AccumulatorFragment& c,    // 输入输出：累加矩阵（C = A×B + C）
const MatrixAFragment& a,  // 输入：矩阵A
const MatrixBFragment& b,  // 输入：矩阵B
const AccumulatorFragment& c_initial  // 输入：初始C矩阵（通常与c相同，用于累加）
);
```

- **示例**：执行 16x16x16 的矩阵乘加：

```cpp
// 假设a_frag和b_frag已加载数据，c_frag已初始化
nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);  // c_frag = a_frag × b_frag + c_frag
mma_sync(acc_frag, a_frag, b_frag, acc_frag);
```

#### 片段初始化：`fill_fragment`

初始化累加器片段（如填充 0），作为 `mma_sync` 的初始值（避免未定义行为）。

- **函数原型**：

```cpp
template <typename FragmentType>
void fill_fragment(
FragmentType& frag,                // 输出：要初始化的片段
const typename FragmentType::element_type& value  // 输入：初始值
);
```

- **示例**：初始化累加器片段为 0：

```cpp
FragC c_frag;
nvcuda::wmma::fill_fragment(c_frag, 0.0f);  // 累加器初始值为0
```

---

### **2. 使用流程**

以 **半精度矩阵乘法（HGEMM）** 为例（`C = A * B`，`A: M×K`, `B: K×N`, `C: M×N`）：

#### **3.1 初始化**
- 确定矩阵分块尺寸（如 `16x16x16`）。
- 分配全局内存并初始化矩阵 `A`、`B`、`C`。

#### **3.2 Kernel 函数**
- **线程块设计**：每个 warp 负责一个 `16x16` 的 tile。
- **步骤**：
  1. **加载数据**：从全局内存加载 `A` 和 `B` 的分块到 fragment。
  2. **执行 MMA**：调用 `mma_sync` 进行矩阵乘法累加。
  3. **存储结果**：将结果写回全局内存。

**示例代码**：

```cpp
__global__ void wmma_hgemm(const half *A, const half *B, half *C, int M, int N, int K) {
    // 定义 fragment
    using namespace wmma;
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> acc_frag;

    // 计算当前 warp 的 tile 坐标
    int warp_id = threadIdx.x / 32; // 每个 warp 有 32 threads
    int warp_row = warp_id / (N / 16); // 行索引
    int warp_col = warp_id % (N / 16); // 列索引

    // 加载 A 和 B 的 tile
    load_matrix_sync(a_frag, A + warp_row * K * 16, K, row_major);
    load_matrix_sync(b_frag, B + warp_col * K * 16, K, col_major);

    // 初始化累加器
    fill_fragment(acc_frag, 0.0f);

    // 执行矩阵乘法
    mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    // 存储结果到全局内存
    store_matrix_sync(C + (warp_row * 16) * N + warp_col * 16, acc_frag, N, row_major);
}
```

#### **3.3 主机代码调用**

```cpp
// 分配内存并初始化 A, B, C
// …

// 启动 kernel
int block_size = 256; // 每个 block 有 16 warps
int grid_size = (M * N) / (16 * 16) / 16; // 根据矩阵尺寸调整
wmma_hgemm<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);
```

### WMMA 的使用流程（以 GEMM 为例）

WMMA 的核心是 “warp 级协作”（一个 warp 处理一个矩阵块），使用时需遵循固定流程。以下是一个完整示例：用 WMMA 实现 16x16x16 的矩阵乘法 C=A×B（A 和 B 为 fp16，C 为 fp32）。

#### 步骤 1：定义矩阵尺寸与片段类型

cpp

运行

```cpp
#include <mma.h>
#include <cuda_fp16.h>  // 用于half类型
using namespace nvcuda;

// 矩阵尺寸（需与WMMA支持的尺寸匹配，如16x16x16）
const int M = 16;  // C的行数 = A的行数
const int N = 16;  // C的列数 = B的列数
const int K = 16;  // A的列数 = B的行数

// 定义片段类型
using FragA = wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major>;
using FragC = wmma::fragment<wmma::accumulator, M, N, K, float>;
```

#### 步骤 2：编写核函数（warp 级处理）

一个 warp（32 线程）协作处理一个 16x16x16 的矩阵块：

cpp

运行

```cpp
__global__ void wmma_gemm_kernel(const half* A, const half* B, float* C) {
// 1. 初始化累加器片段（C的初始值为0）
FragC c_frag;
wmma::fill_fragment(c_frag, 0.0f);

// 2. 加载A和B矩阵到片段（假设A是行优先，B是列优先）
FragA a_frag;
FragB b_frag;
wmma::load_matrix_sync(a_frag, A, K);  // A的领先维度为K（每行长度）
wmma::load_matrix_sync(b_frag, B, N);  // B的领先维度为N（每列长度）

// 3. 执行矩阵乘加：c_frag = a_frag × b_frag + c_frag
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

// 4. 将结果存储到全局内存C
wmma::store_matrix_sync(C, c_frag, M, wmma::row_major);  // C的领先维度为M
}
```

#### 步骤 3：主机端调用

cpp

运行

```cpp
#include <iostream>

int main() {
// 1. 分配主机内存并初始化
half* h_A = new half[M * K];
half* h_B = new half[K * N];
float* h_C = new float[M * N];
// 初始化A和B（示例：A=1, B=1，预期C=16×1=16）
for (int i = 0; i < M*K; ++i) h_A[i] = 1.0f;
for (int i = 0; i < K*N; ++i) h_B[i] = 1.0f;

// 2. 分配设备内存
half *d_A, *d_B;
float *d_C;
cudaMalloc(&d_A, M*K*sizeof(half));
cudaMalloc(&d_B, K*N*sizeof(half));
cudaMalloc(&d_C, M*N*sizeof(float));

// 3. 复制数据到设备
cudaMemcpy(d_A, h_A, M*K*sizeof(half), cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, K*N*sizeof(half), cudaMemcpyHostToDevice);

// 4. 启动核函数（1个block，1个warp=32线程）
wmma_gemm_kernel<<<1, 32>>>(d_A, d_B, d_C);
cudaDeviceSynchronize();

// 5. 复制结果回主机并验证
cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
std::cout << "C[0][0] = " << h_C[0] << "（预期16.0）" << std::endl;  // 输出16.0

// 6. 释放内存
delete[] h_A; delete[] h_B; delete[] h_C;
cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
return 0;
}
```

---

### **4. 关键特性**
#### **4.1 Warp-Level 并行**
- 每个 warp 处理一个 `16x16` 的 tile，充分利用 Tensor Core 的硬件资源。
- 多个 warp 并行处理不同 tile，实现高吞吐量。

#### **4.2 数据类型支持**
- 支持 FP16（`__half`）、FP32（`float`）、INT8（`int8_t`）等数据类型。
- 示例：`__half2` 用于 FP16 计算，`int4` 用于 INT8 计算。

#### **4.3 内存优化**
- **全局内存对齐**：确保数据地址对齐到 128 字节（Tensor Core 要求）。
- **共享内存中转**：对于复杂场景，可通过共享内存暂存数据，减少全局内存访问延迟。

---

### **5. 注意事项**
1. **GPU 架构要求**：
   - 支持 Tensor Core 的 GPU（Volta、Turing、Ampere 等）。
   - 示例：Tesla V100（sm_70）、RTX 2080（sm_75）、A100（sm_80）。

2. **矩阵分块约束**：
   - 分块尺寸必须符合 Tensor Core 的要求（如 `16x16x16`）。
   - 矩阵总尺寸需能被分块尺寸整除（否则需处理边界）。

3. **编译器设置**：
   - 编译时指定目标架构：

```bash
nvcc -arch=sm_70 wmma_kernel.cu
```

1. **性能调优**：
   - 合理设计线程块和网格，避免资源争用。
   - 利用共享内存减少全局内存访问。

2. **矩阵尺寸限制**：WMMA 仅支持固定尺寸（如 16x16x16、32x8x16 等），具体取决于 GPU 架构：

- Volta（Sm70）：仅支持 16x16x16。
- Ampere（Sm80+）：支持 16x16x16、32x8x16、8x32x16 等。
larger 矩阵需手动分块（如 1024x1024 矩阵分为 64 个 16x16 子块）。
1. **线程映射**：一个 warp（32 线程）必须完整处理一个片段，核函数的线程块大小需为 32 的倍数（如 32、64 等）。
2. **内存对齐**：输入矩阵的内存地址需按数据类型对齐（如 fp16 对齐到 2 字节，bf16 对齐到 2 字节），否则会导致未定义行为。
3. **数据类型组合**：不同架构支持的类型组合不同（如 Hopper 支持 fp8 输入），需参考 CUDA 文档确认兼容性。

---

### **7. 与 cuBLAS/CUTLASS 的对比**

| **特性**         | **WMMA API**                  | **cuBLAS**                  | **CUTLASS**                |
|------------------|-------------------------------|-----------------------------|----------------------------|
| **抽象层级**     | 底层（warp-level）| 高级（GEMM 接口）| 中级（模板化 GEMM）|
| **灵活性**       | 高（可自定义分块和内存布局）| 低（固定接口）| 中（模板参数配置）|
| **开发难度**     | 高（需手动管理内存和分块）| 低（直接调用函数）| 中（模板编程）|
| **性能**         | 高（接近硬件上限）| 高（高度优化）| 高（可定制优化）|

---

### **总结**

WMMA API 是直接操作 Tensor Core 的强大工具，适合需要精细控制矩阵计算的场景。通过 fragment 抽象和 warp-level 的并行设计，开发者可以充分发挥 Tensor Core 的性能潜力。然而，其使用门槛较高，需结合内存优化和分块策略，推荐在 cuBLAS/CUTLASS 无法满足需求时使用。

CUDA 的 WMMA API 是基于 Tensor Core 的核心矩阵操作接口，通过 “片段” 抽象封装了矩阵加载、乘加、存储等流程，适合在 warp 级别实现高性能矩阵运算。使用时需遵循 “定义片段→加载数据→执行乘加→存储结果” 的流程，并注意尺寸限制和线程协作。对于大多数场景，推荐优先使用 CUTLASS（基于 WMMA）以降低开发成本；若需定制特殊逻辑，则可直接使用 WMMA。

# MMA API

在 CUDA 中，**MMA（Matrix Multiply-Add，矩阵乘加）** 是一类针对矩阵块运算的硬件加速接口，主要通过 GPU 的**Tensor Core**（从 Volta 架构引入的专用矩阵运算单元）实现高效的矩阵乘法与累加操作。这些接口是高性能线性代数计算（如 GEMM、卷积等）的核心基础，直接映射到硬件指令，能显著提升计算密集型任务的效率。

### 核心配套 API 分类与功能

#### 1. 数据加载：`load_matrix_sync`

`load_matrix_sync` 用于将**全局内存或共享内存中的数据**加载到 MMA 专用的 “片段（fragment）” 中。片段是 Tensor Core 处理的基本数据单元（类似硬件可识别的矩阵块容器），需按 Tensor Core 的对齐和布局要求组织数据。

**作用**：
将内存中的矩阵数据（如全局内存中的 A、B 矩阵）转换为 Tensor Core 可直接运算的片段格式，处理布局转换（如行优先→硬件优化布局）和对齐。

**函数原型（简化版）**：

```cpp
template <typename Fragment, typename T, int Layout>
void load_matrix_sync(
  Fragment& frag,        // 输出：加载到的片段
  const T* ptr,          // 输入：内存中矩阵的起始地址
  int ldm,               // 输入：矩阵的领先维度（内存中每行/列的实际长度）
  unsigned int mask = 0  // 输入：掩码（用于部分加载，通常为0）
);
```

**关键参数**：

- `Fragment`：片段类型，需指定矩阵角色（A/B/ 累加器）、尺寸、数据类型等，如 `__mma::fragment<__mma::matrix_a, 16, 16, 16, __half, __mma::row_major>`。
    - `matrix_a`/`matrix_b`：表示该片段是 A 矩阵或 B 矩阵的输入。
    - `accumulator`：表示该片段是累加器（如 C/D 矩阵）。
- `Layout`：内存中矩阵的布局（`__mma::row_major` 或 `__mma::col_major`）。
- `ldm`：领先维度（Leading Dimension），确保从内存正确读取矩阵的行 / 列。

**示例**：
加载 16x16 的 fp16 矩阵 A（行优先）到片段：

```cpp
#include <mma.h>
using namespace nvcuda;

// 定义A矩阵的片段类型（16x16x16，fp16，行优先）
using FragmentA = __mma::fragment<
  __mma::matrix_a,    // 角色：A矩阵
  16, 16, 16,         // 尺寸：m=16, n=16, k=16
  __half,             // 数据类型：fp16
  __mma::row_major    // 内存布局：行优先
>;

__half* A_global;  // 全局内存中的A矩阵（16x16）
FragmentA a_frag;  // A矩阵的片段

// 从全局内存加载A矩阵到片段
load_matrix_sync(a_frag, A_global, 16);  // ldm=16（A矩阵的实际行长度）
```

#### 2. 数据存储：`store_matrix_sync`

`store_matrix_sync` 用于将 MMA 片段中的结果（如计算完成的 D 矩阵）从片段**存储回全局内存或共享内存**，并转换为常规内存布局（如行优先 / 列优先）。

**作用**：
将 Tensor Core 输出的片段格式转换为通用内存布局，供后续计算或主机端读取。

**函数原型（简化版）**：

```cpp
template <typename Fragment, typename T, int Layout>
void store_matrix_sync(
  T* ptr,                // 输出：存储到的内存地址
  const Fragment& frag,  // 输入：要存储的片段
  int ldm,               // 输入：矩阵的领先维度
  unsigned int mask = 0  // 输入：掩码（用于部分存储，通常为0）
);
```

**示例**：
将累加器片段（D 矩阵，fp32）存储到全局内存：

```cpp
// 定义累加器片段类型（16x16x16，fp32）
using FragmentC = __mma::fragment<
  __mma::accumulator,  // 角色：累加器（C/D矩阵）
  16, 16, 16,          // 尺寸：m=16, n=16, k=16
  float,               // 数据类型：fp32
  __mma::row_major
>;

float* D_global;    // 全局内存中存储D矩阵的地址
FragmentC d_frag;   // 计算完成的D矩阵片段

// 将片段存储到全局内存（行优先）
store_matrix_sync(D_global, d_frag, 16);  // ldm=16
```

#### 3. 片段初始化：`fill_fragment`

`fill_fragment` 用于初始化 MMA 片段（尤其是累加器片段），通常将其填充为初始值（如 0），作为 `__mma_sync` 运算的起点（即 D=A×B+C 中的初始 C）。

**作用**：
确保累加器片段在运算前有已知的初始值，避免未定义行为。

**函数原型**：

```cpp
template <typename Fragment, typename T>
void fill_fragment(Fragment& frag, T value);
```

**示例**：
初始化累加器片段为 0：

```cpp
FragmentC c_frag;  // 累加器片段（C矩阵）
fill_fragment(c_frag, 0.0f);  // 填充为0
```

#### 4. 元素级操作：片段的访问与修改

MMA 片段本质上是一个包含矩阵元素的容器，虽然设计为配合 `__mma_sync` 使用，但也支持通过索引访问元素（只读或读写，取决于片段类型），用于微调或 debug。

**示例**：
访问累加器片段的元素：

```cpp
// 访问c_frag中第i行第j列的元素（仅累加器片段支持写操作）
c_frag[i][j] = 1.0f;  // 手动修改初始值
```

> 注意：输入片段（matrix_a/matrix_b）通常是只读的，修改可能导致未定义行为；累加器片段（accumulator）支持读写。

### MMA 完整工作流示例

这些 API 需配合 `__mma_sync` 形成完整的矩阵乘加流程，例如 16x16x16 的 fp16→fp32 运算：

```cpp
#include <mma.h>
using namespace nvcuda;

// 定义片段类型
using FragA = __mma::fragment<__mma::matrix_a, 16, 16, 16, __half, __mma::row_major>;
using FragB = __mma::fragment<__mma::matrix_b, 16, 16, 16, __half, __mma::col_major>;
using FragC = __mma::fragment<__mma::accumulator, 16, 16, 16, float>;

__global__ void mma_kernel(__half* A, __half* B, float* C, int lda, int ldb, int ldc) {
  // 1. 初始化累加器片段（C矩阵初始值）
  FragC c_frag;
  fill_fragment(c_frag, 0.0f);  // 或从内存加载已有C矩阵

  // 2. 加载A、B矩阵到片段
  FragA a_frag;
  FragB b_frag;
  load_matrix_sync(a_frag, A + threadIdx.x * 16, lda);  // 假设线程块内分块
  load_matrix_sync(b_frag, B + threadIdx.y * 16, ldb);

  // 3. 执行矩阵乘加：D = A * B + C
  __mma_sync(c_frag, a_frag, b_frag, c_frag);  // 结果存回c_frag

  // 4. 将结果存储回全局内存
  store_matrix_sync(C + threadIdx.x * 16, c_frag, ldc);
}
```

### 总结

CUDA 的 MMA 相关 API 围绕 “**片段（fragment）**” 这一核心概念设计，形成完整的 “加载→运算→存储” 流水线：

- `load_matrix_sync`：将内存数据加载为片段（适配 Tensor Core 格式）；
- `__mma_sync`：执行核心矩阵乘加（D=A×B+C）；
- `store_matrix_sync`：将片段结果存储回内存（转换为通用格式）；
- `fill_fragment`：初始化片段（如累加器初始值）。

这些 API 共同支撑了 Tensor Core 的高效利用，是实现高性能 GEMM、卷积等运算的基础（CUTLASS 等库的底层核心就是这些 API 的封装）。

# MMA
### CUDA 提供的 MMA 接口层次

CUDA 对 MMA 的支持分为**硬件级 PTX 指令**和**CUDA C++ intrinsic 函数**两个层次，前者是底层汇编指令，后者是 C++ 级别的封装（更易用）。

#### 1. 硬件级 PTX 指令：`mma.sync`

PTX（Parallel Thread Execution）是 CUDA 的中间汇编语言，`mma.sync` 是 Tensor Core 的核心 MMA 指令，直接控制硬件执行矩阵乘加操作。其基本功能是：

对两个小型矩阵（如 16x16x16）执行乘法，再与第三个矩阵累加，即 D=A×B+C。

`mma.sync` 的指令格式随 GPU 架构演进（支持的数据类型和矩阵尺寸逐渐扩展），以最常见的**Ampere 架构（Sm80）** 为例，指令格式如下（简化版）：

```ptx
mma.sync.aligned.m16n16k16.row.col.f16.f16.f32.f32 d, a, b, c;
``` 

- **参数含义**：
    - `m16n16k16`：矩阵尺寸，即 A(16×16)×B(16×16)（实际维度为 m×k×k×n，此处 m=16,n=16,k=16）。
    - `row.col`：矩阵 A、B 的布局（行优先 / 列优先）。
    - `f16.f16.f32.f32`：数据类型，A 和 B 为 fp16，累加结果 C 和输出 D 为 fp32。
    - `d, a, b, c`：寄存器中的矩阵数据（需按硬件要求对齐）。
- **支持的架构与扩展**：
    - Volta（Sm70）：首次引入，仅支持 fp16 输入、fp32 累加，矩阵尺寸固定为 16x16x16。
    - Ampere（Sm80）：新增 tf32（Tensor float32）、bf16（脑浮点）支持，扩展矩阵尺寸（如 32x8x16）。
    - Hopper（Sm90）：支持 fp8（8 位浮点）、int4 等低精度类型，进一步提升 AI 推理效率。

#### 2. CUDA C++ Intrinsic 函数：`__mma_sync`

为简化开发，CUDA 提供了 C++ 级别的 `__mma_sync` intrinsic 函数，封装了 `mma.sync` PTX 指令，开发者无需直接编写 PTX。这些函数按数据类型和矩阵尺寸分为多个变体，核心功能与底层指令一致，但更易集成到 C++ 代码中。

##### 常用 `__mma_sync` 函数示例

- **半精度输入，单精度累加（最常用）**：
    计算 D=A×B+C，其中 A(16×16)、B(16×16) 为 fp16，、 为 fp32。

    ```cpp
    #include <mma.h>
    using namespace nvcuda;
    
    // 定义矩阵片段（片段是Tensor Core的输入/输出格式，需按硬件要求组织数据）
    __half a_frag[16][16];  // A矩阵（fp16）
    __half b_frag[16][16];  // B矩阵（fp16）
    float  c_frag[16][16];  // C矩阵（fp32，累加输入）
    float  d_frag[16][16];  // D矩阵（fp32，输出）
    
    // 执行MMA操作（16x16x16）
    __mma_sync(
      d_frag,    // 输出矩阵D
      a_frag,    // 输入矩阵A
      b_frag,    // 输入矩阵B
      c_frag,    // 累加矩阵C
      0, 0, 0    // 对齐参数（通常为0）
    );
    ```

- **其他数据类型变体**：
    - `__mma_sync` 支持 bf16（`__nv_bfloat16`）、tf32（`__tf32`）、int8 等类型，函数名和参数类型相应调整，例如：

        ```cpp
        // bf16输入，fp32累加
        __nv_bfloat16 a_bf16[16][16];
        __nv_bfloat16 b_bf16[16][16];
        __mma_sync(d_frag, a_bf16, b_bf16, c_frag, 0, 0, 0);
        ```

- **矩阵尺寸变体**：
    除 16x16x16 外，Ampere 及以上架构支持更大或更灵活的尺寸（如 32x8x16），通过函数模板参数指定，例如：

    ```cpp
    // 32x8x16尺寸的MMA（A:32x16, B:16x8, C/D:32x8）
    __mma::fragment<__mma::matrix_a, 32, 16, 16, __half, __mma::row_major> a_frag;
    __mma::fragment<__mma::matrix_b, 16, 8, 16, __half, __mma::col_major> b_frag;
    __mma::fragment<__mma::accumulator, 32, 8, 16, float> c_frag, d_frag;
    
    __mma_sync(d_frag, a_frag, b_frag, c_frag);
    ```

### MMA 接口的核心功能

MMA 接口的核心是**高效执行 “矩阵块乘加”**，这是绝大多数线性代数运算的基础，具体应用场景包括：

1. **GEMM（通用矩阵乘法）**
    大矩阵乘法 C=α⋅A⋅B+β⋅C 可分解为无数个小矩阵块的 MMA 操作。通过 MMA 接口，能充分利用 Tensor Core 的算力（如 Ampere Tensor Core 的算力是同架构 CUDA Core 的 8 倍以上）。

2. **卷积运算**
    深度学习中的卷积操作可通过 “im2col” 转化为 GEMM，进而依赖 MMA 加速。例如，3x3 卷积层的计算可分解为多个 16x16x16 的矩阵块乘加。

3. **深度学习层计算**
    全连接层、Transformer 中的注意力机制等，本质上都是矩阵运算，MMA 接口是这些层高性能实现的核心（如 PyTorch、TensorFlow 的 GPU 加速底层依赖）。

4. **低精度与混合精度计算**
    MMA 接口支持 fp16、bf16、int8 等低精度类型，在 AI 推理（如模型量化）中可显著降低内存带宽需求，同时保持较高精度。

### 总结

CUDA 的 MMA 接口（`mma.sync` PTX 指令和 `__mma_sync` intrinsic 函数）是 Tensor Core 的直接编程接口，核心功能是高效执行小矩阵块的乘加操作。它们是实现高性能 GEMM、卷积等线性代数运算的基础，也是 CUTLASS 等优化库的底层依赖。

对于开发者：

- 若需极致性能（如编写自定义 GEMM kernel），可直接使用 `__mma_sync` intrinsic 函数；
- 若追求易用性，可基于 CUTLASS 等库（其内部封装了 MMA 接口）快速开发，无需直接操作底层指令。

在 CUDA 的 MMA（矩阵乘加）编程模型中，除了核心的 `__mma_sync`（矩阵乘加运算），还包含一系列配套的 API 用于**数据加载（load）、数据存储（store）、片段（fragment）操作**等，这些 API 共同构成了完整的 MMA 工作流。它们主要定义在 `nvcuda::mma` 命名空间中，用于处理 Tensor Core 的输入输出数据格式、内存交互和中间状态管理。
