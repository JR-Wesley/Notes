---
dateCreated: 2025-07-03
dateModified: 2025-07-05
---

参考：https://zhuanlan.zhihu.com/p/346910129

# 简介



其它程序如 bandwidthTest, vectorAdd 等也将对 CUDA 的性能进行测试。

示例：

- cudaMalloc: 为指针申请 GPU 中的内存
- cudaMemcpy: CPU 和 GPU 之间的内存拷贝
- cudaFree: 释放指针指向的 GPU 内存


# CUDA 快速入门指南：系统学习路径与实践策略

**示例代码（向量加法）**：

cuda

```cuda
#include <stdio.h>

// 核函数：每个线程计算一个元素
__global__ void vectorAdd(float* A, float* B, float* C, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int n = 1000;
    size_t size = n * sizeof(float);
    
    // 分配主机内存
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    
    // 初始化数据
    for (int i = 0; i < n; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }
    
    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // 从主机复制数据到设备
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // 定义线程块和网格维度
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // 执行核函数
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    
    // 从设备复制结果到主机
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // 验证结果
    for (int i = 0; i < 5; i++) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }
    
    // 释放内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
```

##### **2. 并行编程模型与优化（2-3 周）**

- **线程组织策略**：
    - 合理设计 Block 和 Grid 维度，例如处理二维矩阵时使用二维线程块 `dim3 block(16, 16)`。
- **内存优化**：
    - 使用 Shared Memory 减少 Global Memory 访问（如矩阵乘法中的分块算法）。
    - 合并 Global Memory 访问，利用 GPU 的内存事务机制提高带宽利用率。
- **同步与原子操作**：
    - 使用 `__syncthreads()` 实现 Block 内线程同步。
    - 原子操作（如 `atomicAdd()`）用于处理线程间竞争。

**示例：使用 Shared Memory 优化矩阵乘法**

cuda

```cuda
__global__ void matrixMultiply(float* A, float* B, float* C, int N) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // 分块计算
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 加载数据到Shared Memory
        if (row < N && t * TILE_SIZE + tx < N)
            s_A[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        else
            s_A[ty][tx] = 0.0f;
            
        if (t * TILE_SIZE + ty < N && col < N)
            s_B[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            s_B[ty][tx] = 0.0f;
        
        // 同步确保所有数据加载完成
        __syncthreads();
        
        // 计算部分和
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += s_A[ty][k] * s_B[k][tx];
        }
        
        // 同步确保所有线程完成计算再加载下一阶段数据
        __syncthreads();
    }
    
    // 写入结果
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
```

##### **3. 工具链与性能分析（1-2 周）**

- **编译与调试工具**：
    - `nvcc`：NVIDIA CUDA 编译器，支持将 `.cu` 文件编译为可执行文件。
    - `cuda-gdb`：GPU 调试器，用于定位 Kernel 中的错误。
- **性能分析工具**：
    - **Nsight Compute**：分析 Kernel 性能瓶颈（如内存带宽、计算效率）。
    - **Nsight Systems**：系统级性能分析，监控 CPU-GPU 交互和内存传输。

**性能分析示例**：

bash

```bash
# 编译时添加调试信息
nvcc -g -G vectorAdd.cu -o vectorAdd

# 使用cuda-gdb调试
cuda-gdb vectorAdd

# 使用Nsight Compute分析性能
nv-nsight-cu-cli ./vectorAdd
```

##### **4. 高级主题与应用（2-4 周）**

- **CUDA 库**：
    - cuBLAS（线性代数）、cuDNN（深度学习）、cuFFT（快速傅里叶变换）等，直接调用预优化库加速常见计算。
- **多 GPU 编程**：
    - 使用 `cudaSetDevice()` 管理多个 GPU，通过 `nccl`（NVIDIA Collective Communications Library）实现 GPU 间通信。
- **Python 接口**：
    - **PyCUDA**：直接在 Python 中调用 CUDA 核函数。
    - **Numba**：通过 JIT 编译 Python 函数为 CUDA 代码，简化开发。

**Python + Numba 示例**：

python

运行

```python
import numpy as np
from numba import cuda

@cuda.jit
def vector_add(a, b, c):
    idx = cuda.grid(1)
    if idx < a.size:
        c[idx] = a[idx] + b[idx]

def main():
    n = 1000
    a = np.arange(n).astype(np.float32)
    b = np.arange(n).astype(np.float32) * 2
    c = np.zeros_like(a)
    
    # 定义线程块和网格
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    
    # 执行核函数
    vector_add[blocks_per_grid, threads_per_block](a, b, c)
    
    print(c[:5])  # 验证结果

if __name__ == "__main__":
    main()
```

#### **三、实战项目与资源推荐**

##### **1. 实战项目**

- **基础项目**：
    - 实现矩阵乘法、快速傅里叶变换（FFT）等算法，对比 CPU 和 GPU 性能差异。
    - 图像滤波（如高斯模糊），利用 GPU 并行处理像素。
- **进阶项目**：
    - 基于 CUDA 实现简单的神经网络（如 MNIST 分类）。
    - 优化深度学习框架中的算子（如卷积、Softmax）。

##### **2. 推荐学习资源**

- **官方文档**：
    - [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)：包含编程指南、API 参考等。
    - [NVIDIA Developer](https://developer.nvidia.com/)：提供教程、示例代码和白皮书。
- **书籍**：
    - 《CUDA C 编程权威指南》：全面介绍 CUDA 编程模型与优化技巧。
    - 《GPU 高性能编程 CUDA 实战》：通过案例学习 CUDA 并行编程。
- **在线课程**：
    - Coursera《GPU 计算基础》（NVIDIA 官方课程）。
    - Udemy《CUDA 并行编程实战》：结合项目实践。

#### **四、常见问题与避坑指南**

1. **线程同步错误**：
    - 避免在 `__syncthreads()` 前后出现条件分支（如 `if` 语句），否则可能导致死锁。
2. **内存访问优化**：
    - 全局内存访问需按 32 字节对齐，否则会降低带宽利用率。
3. **调试困难**：
    - 使用 `printf()` 在 Kernel 中打印调试信息，但注意大量打印会严重影响性能。
4. **硬件限制**：
    - 不同 GPU 架构（如 Pascal、Volta、Ampere）的特性（如共享内存大小、Tensor Core 支持）不同，需针对性优化。

#### **五、学习路线时间规划（参考）**

- **第 1 周**：环境搭建，掌握 CUDA 基础语法（核函数、线程组织）。
- **第 2-3 周**：深入理解并行编程模型，实践内存优化（如 Shared Memory）。
- **第 4 周**：学习性能分析工具（Nsight），优化现有项目。
- **第 5-6 周**：探索 CUDA 库（如 cuBLAS）和 Python 接口（Numba），完成综合项目。

# CUDA 核心知识

CUDA 编程的核心知识体系可分为**基础语法**、**并行策略**、**内存优化**、**高级技术**四个递进层次。以下是由浅入深的核心要点梳理，结合实战场景和避坑指南：

#### 2. **基础代码模板**

```cpp
// 核函数定义
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 主机端调用
int main() {
    // 内存分配与数据初始化
    float *h_a, *h_b, *h_c;  // 主机内存
    float *d_a, *d_b, *d_c;  // 设备内存
    
    // 分配主机内存并初始化
    h_a = (float*)malloc(n * sizeof(float));
    h_b = (float*)malloc(n * sizeof(float));
    h_c = (float*)malloc(n * sizeof(float));
    
    // 分配设备内存
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));
    
    // 数据从主机复制到设备
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // 配置并启动核函数
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    
    // 数据从设备复制到主机
    cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 释放内存
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    
    return 0;
}
```

#### 3. **必知 API**

- **内存管理**：`cudaMalloc`、`cudaFree`、`cudaMemcpy`
- **错误检查**：`cudaGetLastError`、`cudaDeviceSynchronize`
- **设备信息**：`cudaGetDeviceCount`、`cudaGetDeviceProperties`

#### 4. **避坑指南**

- **线程索引越界**：必须检查 `idx < n`
- **内存泄漏**：确保 `cudaMalloc` 与 `cudaFree` 成对出现
- **同步陷阱**：核函数异步执行，需显式同步（如 `cudaDeviceSynchronize`）

### 二、**内存模型与优化（进阶）**

**目标**：理解 CUDA 内存层次，掌握高性能内存访问模式

#### 1. **内存层次结构**

|内存类型|速度|作用域|用途|
|---|---|---|---|
|寄存器|最快|线程私有|局部变量|
|共享内存|快|块内共享|数据缓存、块内通信|
|全局内存|慢|全局可见|主要数据存储|
|常量内存|较快|全局只读|不变参数|
|纹理内存|较快|全局只读|空间局部性数据|

#### 2. **共享内存优化**

- **矩阵乘法示例**：

    ```cpp
    __global__ void matrixMultiplyShared(float* A, float* B, float* C, int N) {
        __shared__ float sA[16][16];  // 块内共享内存
        __shared__ float sB[16][16];
        
        int bx = blockIdx.x, by = blockIdx.y;
        int tx = threadIdx.x, ty = threadIdx.y;
        int row = by * 16 + ty;
        int col = bx * 16 + tx;
        
        float sum = 0;
        for (int t = 0; t < (N + 15) / 16; t++) {
            // 加载数据到共享内存
            if (row < N && t*16+tx < N) sA[ty][tx] = A[row*N + t*16+tx];
            if (t*16+ty < N && col < N) sB[ty][tx] = B[(t*16+ty)*N + col];
            
            __syncthreads();  // 同步，确保所有线程加载完成
            
            // 计算部分积
            for (int k = 0; k < 16; k++) {
                sum += sA[ty][k] * sB[k][tx];
            }
            
            __syncthreads();  // 同步，确保所有线程使用完共享内存
        }
        
        if (row < N && col < N) {
            C[row*N + col] = sum;
        }
    }
    ```

#### 3. **内存优化技巧**

- **合并访问**：相邻线程访问连续内存地址
- **内存对齐**：数据大小为 4/8/16 字节倍数
- **减少全局内存访问**：尽量在寄存器和共享内存中计算

#### 4. **性能分析工具**

- **Nsight Compute**：分析内存吞吐量、占有率
- **CUDA-MEMCHECK**：检测内存越界、悬空指针
- **nvprof**：统计核函数执行时间、内存带宽利用率

### 三、**高级并行技术（精通）**

**目标**：掌握多 GPU 协作、流并行、动态并行等复杂技术

#### 1. **多 GPU 编程**

- **设备管理**：`cudaSetDevice`、`cudaGetDeviceCount`
- **P2P 内存访问**：

    cpp

    运行

    ```cpp
    // 启用GPU 0到GPU 1的P2P访问
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);
    
    // 直接在GPU间复制数据
    cudaMemcpyPeer(dst_ptr_gpu1, 1, src_ptr_gpu0, 0, size);
    ```

- **任务划分策略**：按数据分块、按功能分阶段

#### 2. **异步流并行**

- **流（Stream）**：独立的执行队列，实现计算与数据传输重叠

    cpp

    运行

    ```cpp
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // 异步操作
    cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream);
    kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c);
    cudaMemcpyAsync(h_c, d_c, size, cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    ```

#### 3. **动态并行**

- **GPU 内递归计算**：

    cpp

    运行

    ```cpp
    __global__ void recursiveKernel(int depth) {
        if (depth > 0) {
            // 子核函数调用
            recursiveKernel<<<1, 1>>>(depth - 1);
        }
        // 处理数据
    }
    ```

#### 4. **CUDA 与其他技术结合**

- **OpenMP+CUDA**：混合 CPU-GPU 并行
- **CUDA+MPI**：分布式多节点 GPU 计算
- **CUDA 与深度学习框架**：自定义算子开发

### 四、**调试与性能优化（实战）**

**目标**：掌握 CUDA 程序调试方法，能定位性能瓶颈并优化

#### 1. **调试技巧**

- **printf 调试**：在核函数中使用 `printf`（注意性能影响）
- **条件断点**：在 Nsight Compute 中设置基于变量值的断点
- **内存错误检测**：`cuda-memcheck --leak-check full ./program`

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
- **优化内存带宽**：合并访问、对齐数据

### 五、**实战项目推荐**

1. **基础项目**：

    - 并行向量加法 / 乘法
    - 并行前缀和（Scan）算法
2. **进阶项目**：

    - 基于共享内存优化的矩阵乘法
    - CUDA 实现快速傅里叶变换（FFT）
3. **高级项目**：

    - 多 GPU 协作的粒子模拟
    - 基于 CUDA 的深度学习算子开发（如卷积、BatchNorm）

### 六、**推荐学习资源**

1. **官方文档**：
    - [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
    - [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
2. **书籍**：
    - 《CUDA by Example》（CUDA 编程入门经典）
    - 《高性能 CUDA 应用设计与开发》（深入优化）
3. **在线课程**：
    - Coursera: [GPU Programming for Science and Engineering](https://www.coursera.org/learn/gpu-programming)
    - Udemy: [CUDA C++ High Performance Parallel Programming](https://www.udemy.com/course/cuda-c-programming/)
4. **实战案例**：
    - [NVIDIA Code Examples](https://github.com/NVIDIA/cuda-samples)
    - [CUDA Optimization Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

通过这四个层次的学习，配合大量实战练习，你将从 CUDA 初学者逐步成长为能够开发高性能并行程序的专家。记住：CUDA 编程的核心竞争力在于对硬件架构的理解和对内存访问模式的优化。
