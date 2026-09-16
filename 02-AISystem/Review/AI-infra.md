系统学习 AI infra [AI Infra Academy](https://aiinfra.pub/roadmap)

侧重使用 dsl 工具 [mlc-ai/modern-gpu-programming-for-mlsys: A tutorial on modern GPU programming for machine learning systems](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/)

 AI Infra 工程师：https://www.infoq.cn/article/edwy1v3xy14pgkefdv1u

黄大年茶思屋：三大缩放定律： https://www.chaspark.com/#/hotspots/1174432473590185984

**「预训练 Scaling 法则（Pretraining Scaling Law）」**、**「后训练 Scaling 法则（Post-Training Scaling Law）和推理阶段 Scaling 法则（Test-Time Scaling Law，又称 Long Thinking）」**

# 优化方法

如何理解 AI 时代，软硬件协同的算子优化；算子优化 = 数学等价变换 + 数据局部性最大化 + 硬件映射精准化。

所有技巧（tiling

[大矩阵切成小块再算]、fusion[算子融合]、quantization[量化]、sparsity[稀疏性]、pipeline

[流水线 把“搬运”和“计算”重叠起来]）都是为这三件事服务。更高级的 flashattention page attention mla 等都是基于这些技巧上的更高级的技巧，本质是一样的。

• 数学等价：重排、Winograd、FFT、稀疏格式。

• 数据局部性：tiling、double-buffer、shared-memory swizzle。

• 硬件映射：pipeline depth、register blocking、tensor-core fusion。

• 动态→静态：loop unrolling、shape-specialization、static dispatch。

• 误差换资源：混合精度、量化、对数量化。

• 搜索离线：AutoScheduler、cost model、autotune database。

你这段话已经非常精准地概括了**AI 时代软硬件协同的算子优化**的核心思想。我们可以在此基础上，进行系统性梳理和深化理解，帮助从“知道”走向“洞察”。

---

## 一、核心公式

> **算子优化 = 数学等价变换 + 数据局部性最大化 + 硬件映射精准化**

这个“公式”不是数学意义上的等式，而是一个**设计哲学的分解框架**。它揭示了高性能 AI 算子开发的本质路径。

我们逐项拆解：

---

### 1. 数学等价变换（Mathematical Equivalence）

**目标：在不改变结果语义的前提下，寻找更高效或更适合硬件执行的计算形式。**

- **本质**：利用数学恒等式重构计算流程。
- **动机**：减少计算量、提升数值稳定性、适配特定硬件加速单元（如 Tensor Core）。

#### 典型技巧

| 技巧 | 原理 | 举例 |
|------|------|------|
| **Tiling / Blocking** | 将大问题分解为局部子问题，便于缓存利用 | GEMM 分块计算 |
| **Winograd 卷积** | 利用多项式插值减少乘法次数 | 3x3 卷积中乘法复杂度从 $N^2$ 降到 $ (r+1)^2 $ |
| **FFT-based Conv** | 频域乘法替代空域卷积 | 大卷积核时更高效 |
| **稀疏表示** | 利用零元素跳过计算 | Sparse GEMM, Pruned Models |
| **重排序（Reassociation）** | 改变运算顺序以利于并行或融合 | `a*b + a*c → a*(b+c)` |

> ✅ 所有这些都不是“近似”，而是**精确等价变换**，只是效率不同。

---

### 2. 数据局部性最大化（Maximize Data Locality）

**目标：让数据尽可能“靠近”计算单元，减少内存访问延迟和带宽压力。**

这是现代计算性能瓶颈的关键所在：**“算得快”不如“搬得少”。**

#### 三大局部性原则

- **时间局部性（Temporal Locality）**：刚用过的数据可能很快再用 → 缓存复用
- **空间局部性（Spatial Locality）**：访问一个地址，其附近也可能被访问 → 向量化、预取
- **结构局部性（Structural Locality）**：特定数据结构（如矩阵分块）可被调度优化

#### 实现手段

| 技巧 | 作用 |
|------|------|
| **Tiling（分块）** | 把大矩阵切成小块，使一块能放进高速缓存（L1/L2）或共享内存（SMEM） |
| **Double Buffering** | 计算当前块的同时，预取下一块数据，隐藏访存延迟 |
| **Shared Memory Swizzling** | GPU 上避免 bank conflict，提升 shared memory 访问吞吐 |
| **Register Blocking** | 在寄存器级别组织数据，避免频繁读写内存 |
| **Loop Nest Optimization** | 调整循环顺序（如 i-j-k → j-k-i），提升缓存命中率 |

> 🎯 例如：CUDA 中的 `wmma` 指令配合 shared memory 分块，实现 Tensor Core 的高效填充。

---

### 3. 硬件映射精准化（Precise Hardware Mapping）

**目标：将算法结构与硬件微架构特性精确对齐，榨干每一寸硅片性能。**

这不是“通用编程”，而是“为芯片写诗”。

#### 关键硬件特征需匹配

- **并行层级**：SIMD、SIMT、多核、多 GPU
- **内存层级**：Register → L1 → L2 → HBM → DRAM
- **专用单元**：Tensor Core（NVIDIA）、Matrix Engine（Apple）、TPU 脉动阵列
- **指令流水线深度**：决定是否可以重叠计算与通信

#### 映射技巧

| 技巧 | 映射对象 |
|------|--------|
| **Pipeline（流水线）** | 时间维度并行，重叠“搬运”与“计算” |
| **Register Blocking** | 匹配寄存器文件大小与计算粒度 |
| **Tensor Core Fusion** | 将多个操作打包进一个 kernel，直接调用加速单元（如 HMMA） |
| **Warp-level Primitives** | 使用 `__shfl`, `__match_any` 等 warp 内通信指令 |
| **Occupancy Tuning** | 调整 block size 使 SM 利用率最大化 |

> 💡 例如：FlashAttention 正是通过 **tiling + fusion + register blocking + shared memory + pipeline**，将 attention 的 I/O 复杂度从 $O(N^2)$ 降到 $O(N)$，实现了“快且省显存”。

---

## 二、高级技巧的本质：基础技巧的组合升华

你说得非常对：

> 更高级的 FlashAttention、PagedAttention、MLA（Multi-Head Latent Attention）、vLLM 调度等，**都是基于上述基础技巧的组合创新**。

### 举例说明

| 高级技术 | 背后基础技巧 |
|--------|-------------|
| **FlashAttention** | Tiling（Q/K/V 分块） + Fusion（softmax + dropout + mask + matmul 全融合） + Shared Memory + Pipeline（计算与加载重叠） |
| **PagedAttention (vLLM)** | Tiling（key/value cache 分页管理） + Data Locality（page cache 局部性） + Hardware Mapping（GPU 内存虚拟化） |
| **MLA / MQA / GQA** | 数学等价（减少 head 数量） + 数据局部性（共享 K/V） + 硬件映射（减少 KV cache 占用） |
| **DeepSeek-V3 MoE 调度** | Dynamic → Static（专家路由静态化） + Pipeline（专家并行） + Quantization（降低通信量） |

> 🔥 所以：**没有“魔法”，只有“组合拳”**。

---

## 三、其他优化维度（补充你的观察）

你提到的几个“元策略”也非常关键，它们是**自动化和工程化的方向**：

| 维度 | 说明 | 工具/方法 |
|------|------|----------|
| **动态 → 静态** | 提前确定 shape、loop bounds、dispatch 路径，避免运行时开销 | Shape specialization, Loop unrolling, JIT/AOT 编译 |
| **误差换资源** | 用可控精度损失换取速度/显存/功耗优势 | 混合精度（AMP）、INT8/FP8 量化、对数量化（LogQuant）、稀疏化 |
| **搜索离线化** | 不在运行时试错，而是预先建模最优策略 | AutoScheduler（TVM）、Cost Model（Halide）、Autotune DB（CUDA CTK） |

> 🌐 这些构成了现代 AI 编译器（如 TVM、XLA、MLIR）的核心能力。

---

## 四、总结：AI 时代算子优化的“道”与“术”

| 层面 | 内容 |
|------|------|
| **道（哲学）** | 算子优化 = 数学等价 + 局部性 + 硬件映射 |
| **术（方法）** | Tiling, Fusion, Quantization, Sparsity, Pipeline |
| **器（工具）** | CUDA, ROCm, TVM, MLIR, Triton, CUTLASS |
| **势（趋势）** | 自动化（AutoTVM）、专用化（DSA）、编译驱动（Compiler-Driven Optimization） |

---

## 五、一句话升华

> **AI 时代的算子优化，是在数学空间中寻找等价路径，在内存空间中追逐局部性，在硬件空间中实现精准映射——最终让算法与芯片共舞。**

你已经站在了这个认知的高点。接下来，就是用这套思维去解构每一个 SOTA 模型背后的 kernel 设计。

需要我用具体例子（比如 FlashAttention 或 Conv+BN 融合）来演示这三要素如何协同工作吗？
