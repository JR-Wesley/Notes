---
dateCreated: 2025-07-31
dateModified: 2025-07-31
---
# 高性能计算、模型部署、自定义算子开发

## 🧱 一、你需要掌握的核心知识体系

### 1. **C++14/17 基础（必须）**

LibTorch 是 C++ 库，你需要扎实的现代 C++ 基础。

- ✅ 基础语法：变量、循环、函数、指针、引用
- ✅ 面向对象：类、继承、多态
- ✅ 模板（Templates）：函数模板、类模板（非常重要！）
- ✅ STL：`vector`, `string`, `memory`（智能指针 `shared_ptr`, `unique_ptr`）
- ✅ C++11/14/17 特性：`auto`, `lambda`, `move semantics`, `rvalue references`

---

### 2. **PyTorch 的 Python API（熟悉）**

你最终是要用 Python 调用 C++，所以必须理解 PyTorch 的语义。

- ✅ `torch.Tensor` 的创建、操作、设备（CPU/GPU）
- ✅ `torch.nn.Module`, `forward` 方法
- ✅ `torch.jit.trace`, `torch.jit.script`（用于模型序列化）
- ✅ `torch.utils.cpp_extension` 的使用（如 `load_inline`, `CUDAExtension`）

---

### 3. **LibTorch C++ API（核心）**

这是你要重点学习的部分。

| 模块 | 内容 |
|------|------|
| `#include <torch/torch.h>` | 主头文件 |
| `torch::Tensor` | C++ 中的张量类型，对应 Python 的 `torch.Tensor` |
| `torch::nn::Module` | 定义神经网络模块 |
| `torch::nn::Linear`, `torch::nn::Conv2d` 等 | 常见层 |
| `torch::optim::SGD`, `Adam` | 优化器 |
| `torch::save()`, `torch::load()` | 模型保存与加载 |
| `torch::jit::load()` | 加载 Python 导出的 `.pt` 模型 |

> 📚 官方文档：[LibTorch C++ API](https://pytorch.org/cppdocs/)

---

### 4. **如何让 C++ 函数被 Python 调用？**

有两种主流方式：

#### ✅ 方法 1：使用 `pybind11` + `torch::Tensor`（推荐）

- 将 C++ 函数用 `pybind11` 封装为 Python 模块
- 输入输出使用 `torch::Tensor`，自动与 NumPy/PyTorch 兼容
- 可以配合 `torch.utils.cpp_extension.CUDAExtension` 编译 CUDA 代码

```cpp
// binding.cpp
#include <pybind11/pybind11.h>
#include <torch/extension.h>

torch::Tensor add_tensors(torch::Tensor a, torch::Tensor b) {
    return a + b;
}

PYBIND11_MODULE(my_ops, m) {
    m.def("add_tensors", &add_tensors, "Add two tensors");
}
```

#### ✅ 方法 2：使用 TorchScript（适用于模型部署）

- 在 Python 中用 `@torch.jit.script` 或 `torch.jit.trace` 导出模型为 `.pt`
- 在 C++ 中用 `torch::jit::load("model.pt")` 加载并推理

```cpp
auto module = torch::jit::load("model.pt");
auto output = module.forward({input_tensor});
```

> 适用场景：**部署训练好的模型**，而不是开发新算子。

---

### 5. **构建系统（Build System）**

你需要学会如何编译 LibTorch 项目。

| 工具 | 说明 |
|------|------|
| `CMake` | 最常用，LibTorch 官方推荐 |
| `setuptools` + `CUDAExtension` | 适合 Python 扩展，自动处理编译 |
| `make` | 简单项目可用，但不推荐复杂项目 |

> 📚 推荐：使用 `CMake` + `FindTorch.cmake`

---

## 🧪 四、一个完整的小项目练习建议

### 目标：实现一个 `fused_relu` 函数
- 输入：`torch.Tensor`
- 输出：`input.clamp(min=0)`（即 ReLU）
- 用 C++ 实现，通过 `pybind11` 暴露给 Python

#### 步骤
1. 写 `relu_op.cpp`：

   ```cpp
   torch::Tensor fused_relu(torch::Tensor x) {
       return torch::clamp(x, 0);
   }
   ```

2. 用 `pybind11` 绑定
3. 用 `CUDAExtension` 编译
4. Python 中调用并验证结果

---

## ✅ 总结：你要学什么？

| 类别 | 内容 |
|------|------|
| **语言基础** | C++14/17（模板、智能指针、lambda）|
| **核心库** | LibTorch C++ API（`torch::Tensor`, `torch::nn`）|
| **绑定技术** | `pybind11`（让 C++ 函数被 Python 调用）|
| **构建工具** | `CMake` 或 `setuptools` + `CUDAExtension` |
| **部署方式** | TorchScript（模型部署）或自定义算子（扩展）|

---

## 💡 小贴士

- 从 **CPU 版本开始**，再扩展到 CUDA
- 使用 `torch::without_grad()` 减少内存开销
- 打印 tensor 用 `std::cout << tensor << std::endl;`
- 调试时用 `assert(tensor.defined())` 检查空指针

---

# Cuda 加速

## 🧱 实现一个“可调用”的 CUDA 算子（基础版）

目标：把你的 CUDA kernel 封装成 `torch.nn.Module` 或函数，能在 PyTorch 模型中调用。

### 项目结构

```
my_extension/
├── my_op.cu          # CUDA kernel
├── my_op.cpp         # C++ 绑定代码
├── setup.py          # 编译脚本
└── test.py           # 测试模型调用
```

---

### 1. `my_op.cu`（示例：Fused ReLU + Scale）

```cpp
// my_op.cu
#include <cuda_runtime.h>

__global__ void fused_relu_scale_kernel(float* input, float* output, int N, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = fmaxf(input[idx], 0.0f) * alpha;
    }
}

extern "C" void launch_fused_relu_scale(float* d_input, float* d_output, int N, float alpha) {
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    fused_relu_scale_kernel<<<grid_size, block_size>>>(d_input, d_output, N, alpha);
    cudaDeviceSynchronize();
}
```

---

### 2. `my_op.cpp`（C++ 绑定 + PyTorch 集成）

```cpp
// my_op.cpp
#include <torch/extension.h>

void launch_fused_relu_scale(float* d_input, float* d_output, int N, float alpha);

torch::Tensor fused_relu_scale(torch::Tensor input, float alpha) {
    auto output = torch::empty_like(input);
    launch_fused_relu_scale(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel(),
        alpha
    );
    return output;
}

// 注册为 PyTorch 函数
PYBIND11_MODULE(my_ops, m) {
    m.def("fused_relu_scale", &fused_relu_scale, "Fused ReLU + Scale");
}
```

---

### 3. `setup.py`

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_ops',
    ext_modules=[
        CUDAExtension('my_ops', [
            'my_op.cu',
            'my_op.cpp',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

---

### 4. 编译

```bash
python setup.py build_ext --inplace
```

---

### 5. `test.py`（在 PyTorch 模型中使用）

```python
import torch
from my_ops import fused_relu_scale

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 100)

    def forward(self, x):
        x = self.linear(x)
        x = fused_relu_scale(x, alpha=1.5)  # 调用你的 CUDA 算子
        return x

# 测试
model = MyModel().cuda()
x = torch.randn(32, 100, device='cuda')
y = model(x)
print(y.shape)
```

---

## ⚡ 阶段 3：性能优化（比 PyTorch 原生更快）

你现在能“调用”了，下一步是“更快”。

### ✅ 优化方向

| 优化点 | 方法 | 工具/技巧 |
|--------|------|-----------|
| **内存访问优化** | 合并多个操作（fused kernel）| 把 `ReLU + Scale + Add` 合并 |
| **减少内存拷贝** | 原地操作（in-place）| `input.clamp_min_(0)` → 但 CUDA 中需小心 |
| **提高并行度** | 使用 Shared Memory、Coalesced Access | 手动管理 `__shared__` |
| **使用 Tensor Core** | FP 16 + 1688 MMA 指令 | `__half`, `wmma` API（Volta+）|
| **减少 launch 开销** | 合并小 kernel | 用一个 kernel 做多个事 |

---

### 🔥 示例：Fused Bias + GeLU（比 `torch.nn.Linear + GELU` 更快）

```cpp
__global__ void fused_linear_gelu(float* input, float* weight, float* bias, float* output, int B, int I, int O) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * O) return;

    int b = idx / O;
    int o = idx % O;

    float sum = bias[o];
    for (int i = 0; i < I; i++) {
        sum += input[b * I + i] * weight[o * I + i];
    }

    // GELU 近似
    float x = sum;
    float gelu = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * x * (1.0f + 0.044715f * x * x)));
    output[idx] = gelu;
}
```

> 这个 kernel 把 **Linear + Bias + GELU** 三步融合，减少内存读写次数，速度可提升 2-3 倍。

---

## 📈 阶段 4：性能对比与验证

### 写一个 Benchmark 脚本

```python
import torch
import time
from my_ops import fused_linear_gelu

# 原生实现
class NativeModel(torch.nn.Module):
    def __init__(self, I, O):
        super().__init__()
        self.linear = torch.nn.Linear(I, O)
        self.gelu = torch.nn.GELU()

    def forward(self, x):
        return self.gelu(self.linear(x))

# 自定义实现（假设已封装）
def custom_forward(input, weight, bias):
    return fused_linear_gelu(input, weight, bias)

# 测试
B, I, O = 32, 768, 3072
x = torch.randn(B, I, device='cuda')
model = NativeModel(I, O).cuda()

# 原生
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    y1 = model(x)
torch.cuda.synchronize()
t1 = time.time()

# 自定义
t2 = time.time()
for _ in range(100):
    y2 = custom_forward(x, model.linear.weight, model.linear.bias)
torch.cuda.synchronize()
t3 = time.time()

print(f"Native: {(t1-t0)*1000:.2f} ms")
print(f"Custom: {(t3-t2)*1000:.2f} ms")
print(f"Speedup: {(t1-t0)/(t3-t2):.2f}x")
```

---

## 🧠 阶段 5：进阶技巧（真正超越原生）

| 技巧 | 说明 |
|------|------|
| **使用 CUTLASS / CTK** | NVIDIA 官方的线性代数库，支持 Tensor Core |
| **使用 CUDA Graphs** | 减少 kernel launch 开销，适合固定计算图 |
| **Kernel Fusion 自动化** | 借鉴 TorchDynamo + Inductor 思路 |
| **Memory Pool 优化** | 使用 `cudaMallocAsync` / `cudaFreeAsync`（CUDA 11.2+）|
| **Profile 驱动优化** | 用 `nsight-systems` 或 `nvprof` 找瓶颈 |

---

## 📚 推荐学习资源

| 资源                                                                                          | 说明                         |     |
| ------------------------------------------------------------------------------------------- | -------------------------- | --- |
| [PyTorch C++ Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)         | 官方教程                       |     |
| [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | CUDA 权威文档                  |     |
| [CUTLASS](https://github.com/NVIDIA/cutlass)                                                | 高性能 GEMM 库                 |     |
| [FlashAttention](https://github.com/HazyResearch/flash-attention)                           | 实战参考（融合 Attention + IO 优化）|     |
| [Triton](https://github.com/openai/triton)                                                  | 可选：用 Python 写高性能 kernel    |     |
| [PyTorch C++ API 文档](https://pytorch.org/cppdocs/)                                          | 官方文档，必看                    |     |
| [pybind11 官方文档](https://pybind11.readthedocs.io/)                                           | 学会如何绑定 C++ 和 Python        |     |
| [LibTorch Examples](https://github.com/pytorch/examples/tree/master/cpp)                    | 官方 C++ 示例                  |     |
| [pytorch/cpp-demo](https://github.com/pytorch/cpp-demo)                                     | 简单的 C++ 推理 demo            |     |
| [torchani/cpp](https://github.com/aiqm/torchani/tree/master/cpp)                            | 实际项目参考                     |     |
|                                                                                             |                            |     |

---

## 💡 小建议

- 从 **小算子** 开始：ReLU、GELU、LayerNorm
- 用 **`torch.allclose()`** 验证数值正确性
- 用 **`torch.cuda.synchronize()`** 准确计时
- 关注 **memory bandwidth bound** vs **compute bound**

---

如果你告诉我你想优化的具体算子（比如 LayerNorm、Attention、Softmax），我可以给你写一个完整的 **高性能 CUDA 实现 + PyTorch 集成 demo**。欢迎继续提问！
