https://zhuanlan.zhihu.com/p/685020608


flash-attention 系列解读： https://zhuanlan.zhihu.com/p/661478232


- 目的：合并 attention 的多个操作，减少全局内存的访问
- 切入点：一个最简的 attention [实现](https://github.com/tspeterkim/flash-attention-minimal.git)，直接使用 C++ libtorch 调用、测试。（现有框架融合过于复杂）

参考：
- 关于attention的原理，见.ipynb
- flash attention 为什么这么快
- 神经网络 - 量化与部署，进阶之路

# 介绍


FlashAttention主要解决Transformer计算速度慢和存储占用高的问题。但与绝大多数Efficient Transformer把改进方法集中在**降低模型的FLOPS**（**F**loating **P**oint **O**perations **P**er **S**econd）不同，FlashAttention将优化重点放在了**降低存储访问开销**上。

FlashAttention是一种**精确的优化算法**，它的计算结果在理论上与标准的Self-attention一致（实际会因数值问题有轻微差异）。

## Attention

$$
Attention(Q, K, V) = Softmax(\frac{QK^T}{\sqrt{d_k}})V
$$




##
  Flash Attention CUDA Implementation                                                                                                                                                                                                                                                                                                                                                                                                                         Overview                                                                                                                                                                                                                                                                                                                                                                                                                                                    The flash-attention implementation in this directory is a CUDA-based optimized version of the attention mechanism used in transformers. It's designed to be more memory-efficient than standard                               attention by avoiding the explicit storage of large N×N attention matrices.                                                                                                                                                                                                                                                                                                                                                                                 Key Components                                                                                                                                                                                                                                                                                                                                                                                                                                              1. Main CUDA Kernel (flash.cu)                                                                                                                                                                                                                                                                                                                                                                                                                              The core implementation is in flash.cu, which contains:                                                                                                                                                                                                                                                                                                                                                                                                     Forward Kernel Function                                                                                                                                                                                                        - forward_kernel: The main CUDA kernel that implements the tiled attention computation                                                                                                                                        - Uses shared memory (SRAM) to reduce global memory accesses                                                                                                                                                                  - Implements the softmax computation in a memory-efficient manner using online softmax normalization                                                                                                                                                                                                                                                                                                                                                       Key Features:                                                                                                                                                                                                                  1. Tiled Computation: Processes the attention in blocks of size Bc (columns) and Br (rows)                                                                                                                                    2. Shared Memory Usage: Stores tiles of Q, K, V, and S matrices in SRAM for faster access                                                                                                                                     3. Online Softmax: Computes softmax values incrementally to avoid storing full attention matrices                                                                                                                             4. Memory Efficiency: Reduces HBM (High Bandwidth Memory) read/writes of N² matrices                                                                                                                                                                                                                                                                                                                                                                       Memory Management:                                                                                                                                                                                                             - Uses registers for frequently accessed values (l, m)                                                                                                                                                                        - Uses shared memory for Q, K, V tiles and intermediate S matrix                                                                                                                                                              - Uses global memory for final output (O) and auxiliary arrays (l, m)                                                                                                                                                                                                                                                                                                                                                                                      2. Algorithm Details                                                                                                                                                                                                                                                                                                                                                                                                                                        The implementation follows these steps:                                                                                                                                                                                                                                                                                                                                                                                                                      1. Tiling Strategy:                                                                                                                                                                                                              - Divides the attention computation into tiles of size Br×Bc                                                                                                                                                                  - Processes the attention matrix in chunks to fit in SRAM                                                                                                                                                                                                                                                                                                                                                                                                2. Online Softmax Computation:                                                                                                                                                                                                   - Maintains running max (m) and sum (l) values for numerical stability                                                                                                                                                        - Updates these values incrementally as new blocks are processed                                                                                                                                                              - Avoids storing the full softmax matrix
   3. Memory Access Optimization:                                                                                                                                                                                      [0/120]      - Loads K and V tiles into shared memory once per outer loop iteration

      - Reuses these values across multiple Q tile computations

      - Uses coalesced memory accesses where possible

  4. Test and Build System

  Test File (test.cpp):

   - Implements a manual attention computation for comparison
   - Verifies correctness by comparing results with standard attention
   - Sets up test tensors with specific dimensions

  Build System:

   - Uses CMake for compilation
   - Links against PyTorch libraries
   - Compiles with optimizations (-O3, --use_fast_math)
   - Targets specific GPU architecture (sm_89 in this case)

  Technical Implementation Details

  Shared Memory Layout:

   1 extern __shared__ float sram[];

   2 float* Qi = sram;        // Query tile

   3 float* Kj = &sram[tile_size];    // Key tile

   4 float* Vj = &sram[tile_size * 2]; // Value tile

   5 float* S = &sram[tile_size * 3];  // Score matrix

  Key Algorithm Steps:

   1. Load Kj, Vj tiles into shared memory
   2. For each Qi tile:
      - Load Qi into shared memory
      - Load previous m, l values into registers
      - Compute S = Qi @ Kj^T (scaled)
      - Find row-wise max of S
      - Compute P = exp(S - row_m)
      - Compute row-wise sum of P
      - Update m and l with new values
      - Compute O = (P @ Vj) and update output
      - Store updated m, l back to global memory

  Memory Efficiency:

  The key innovation is avoiding the explicit storage of the full attention matrix (P). Instead, it computes the softmax values on-the-fly and immediately uses them for the output computation,

  significantly reducing memory bandwidth requirements.

  Performance Benefits

   1. Reduced Memory Bandwidth: Avoids storing N×N attention matrices
   2. Better Cache Utilization: Uses shared memory tiling for better data reuse
   3. Numerical Stability: Uses online softmax computation with running max/sum values
   4. Scalability: Works efficiently for long sequences where standard attention would be memory-bound

  This implementation represents a minimal but functional version of flash attention that demonstrates the core optimization techniques used in more sophisticated implementations.
