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


# 项目管理

### 1. **PyBind11**（首选推荐）

- **作用**：将 C++（包括 CUDA 代码）封装成 Python 可调用的模块。
- **优点**：
    - 轻量级，头文件库，无需复杂构建系统。
    - 支持 NumPy 数组无缝传递（`py::array_t<float>`）。
    - 编译后生成 `.so` 文件（Linux）或 `.pyd`（Windows），可直接 `import`。
- **官网**：https://github.com/pybind/pybind11

> ✅ 特别适合封装 CUDA kernel 并暴露给 Python。


使用 `CMakeLists.txt` 管理编译流程，自动调用 `nvcc` 编译 `.cu` 文件，并链接生成 Python 扩展模块。

如果你使用 PyTorch：强烈推荐 `torch.utils.cpp_extension`

这是 PyTorch 官方提供的工具，可以**直接编译 CUDA + C++ 代码并加载为 Python 模块**。
# 技术栈

- ✅ 使用 **CUDA** 编写 kernel
- ✅ 使用 **LibTorch + PyBind11** 封装 C++ 接口
- ✅ 需要支持多个算子（如 matmul、bitonic sort、MoE 等）
- ✅ 未来会频繁增删算子
- ✅ 希望编译方式**简单、灵活、可维护**

---

## ✅ 最终结论：**推荐使用 `torch.utils.cpp_extension` 而不是纯 CMake**

### 🔚 一句话总结：
> **用 `torch.utils.cpp_extension.load`（开发阶段） + `setup.py` + `CUDAExtension`（发布阶段）**，完全替代手动 CMake，更简单、更集成、更适合 PyTorch 生态。

---

## 🤔 为什么不用纯 CMake？

| 优点 | 缺点 |
|------|------|
| 灵活、强大、工业级构建系统 | 配置复杂，需手动处理： |
| | - LibTorch 路径查找 |
| | - CUDA 编译器 (`nvcc`) 设置 |
| | - PyBind11 与 Python 头文件对接 |
| | - 生成 `.so` 并确保能 `import` |
| | - 不同平台兼容性（Linux/macOS/Windows） |

👉 对于 **PyTorch + CUDA 扩展开发**，CMake 是“杀鸡用牛刀”，而 `torch.utils.cpp_extension` 是“量身定制”。

---

## ✅ 推荐方案：分阶段使用 `torch.utils.cpp_extension`

### 🧪 阶段 1：开发调试阶段 → 使用 `load()`（即时编译）

```python
# compile_dev.py
from torch.utils.cpp_extension import load
import os

# 动态列出所有算子目录
op_sources = {
    'matmul': ['src/kernels/matmul.cu', 'src/bindings/matmul.cpp'],
    'bitonic_sort': ['src/kernels/bitonic_sort.cu', 'src/bindings/bitonic_sort.cpp'],
    'moe': ['src/kernels/moe.cu', 'src/bindings/moe.cpp'],
}

# 动态编译并加载
compiled_ops = {}
for op_name, sources in op_sources.items():
    # 检查文件是否存在，便于增删
    if all(os.path.exists(s) for s in sources):
        compiled_ops[op_name] = load(
            name=f"cuda_op_{op_name}",
            sources=sources,
            verbose=True,
            with_cuda=True,
            extra_include_paths=["src/utils"],  # 如有头文件
            extra_cflags=['-O3'],
            extra_cuda_cflags=['-O3', '--use_fast_math']
        )
        print(f"✅ {op_name} 加载成功")
```

✅ **优点**：
- 修改代码后，下次运行自动重新编译
- 无需安装，`import` 即用
- 支持热重载（适合 Jupyter/Notebook）
- 增删算子只需修改 `op_sources` 字典

🔧 使用：
```python
x = torch.randn(100, 100, device='cuda')
y = compiled_ops['matmul'].matmul_forward(x, x.T)
```

---

### 📦 阶段 2：稳定集成阶段 → 使用 `setup.py`（正式安装）

```python
# setup.py
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# 自动扫描算子
def find_cuda_extensions():
    extensions = []
    op_dir = 'src/bindings'
    if not os.path.exists(op_dir):
        return extensions

    for fname in os.listdir(op_dir):
        if fname.endswith('.cpp'):
            op_name = fname[:-4]  # remove .cpp
            cpp_file = f'src/bindings/{op_name}.cpp'
            cu_file = f'src/kernels/{op_name}.cu'
            sources = [cpp_file]
            if os.path.exists(cu_file):
                sources.append(cu_file)

            extensions.append(
                CUDAExtension(
                    name=f'cuda_ops.{op_name}',
                    sources=sources,
                    include_dirs=['src/utils'],
                    extra_compile_args={
                        'cxx': ['-O3'],
                        'nvcc': ['-O3', '--use_fast_math']
                    }
                )
            )
    return extensions

setup(
    name='cuda_ops',
    version='0.1.0',
    packages=find_packages(),
    ext_modules=find_cuda_extensions(),
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
```

📦 安装：
```bash
pip install -e .
```

✅ 使用：
```python
import cuda_ops.matmul
import cuda_ops.moe

y = cuda_ops.matmul.matmul_forward(x, w)
```

---

## ✅ 工程结构建议（支持动态增删）

```
cuda_ops_project/
├── src/
│   ├── kernels/
│   │   ├── matmul.cu        # 新增：复制模板即可
│   │   ├── bitonic_sort.cu
│   │   └── moe.cu
│   ├── bindings/
│   │   ├── matmul.cpp       # 绑定文件，命名一致
│   │   ├── bitonic_sort.cpp
│   │   └── moe.cpp
│   └── utils/
│       └── cuda_helpers.h
├── tests/
│   ├── test_matmul.py
│   └── …
├── benchmarks/
├── compile_dev.py           # 开发时用 load()
├── setup.py                 # 发布时用
└── README.md
```

---

## ✅ 增删算子有多简单？

### ➕ 新增一个算子 `gelu_approx.cu`

1. 创建 `src/kernels/gelu_approx.cu`
2. 创建 `src/bindings/gelu_approx.cpp`
3. 运行 `python compile_dev.py` → 自动编译加载 ✅

### ➖ 删除 `bitonic_sort`

1. 删除 `src/kernels/bitonic_sort.cu`
2. 删除 `src/bindings/bitonic_sort.cpp`
3. `compile_dev.py` 和 `setup.py` 会自动跳过（文件不存在）✅

---

## ✅ 高级技巧

### 1. 共享头文件
在 `src/utils/common.h` 中定义：
```cpp
#pragma once
#include <torch/extension.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be on GPU")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
```

在 `.cpp` 文件中使用：
```cpp
torch::Tensor my_op(torch::Tensor x) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    // …
}
```

### 2. 支持 `torch.compile`
确保你的函数是 pure function，可被 `torch.compile` 识别：
```cpp
// 在 .cpp 中避免全局状态
// 使用 TORCH_LIBRARY() 定义自定义算子（高级用法）
```

---

## ✅ 总结：你应该怎么做？

| 场景 | 推荐方式 | 命令 |
|------|----------|------|
| **开发调试** | `torch.utils.cpp_extension.load` | `python compile_dev.py` |
| **正式集成** | `setup.py` + `CUDAExtension` | `pip install -e .` |
| **增删算子** | 只需增删 `.cu` 和 `.cpp` 文件 | 自动识别 ✅ |
| **CMake** | ❌ 不推荐（除非已有大型 C++ 项目） | - |

---

🎯 **最佳实践**：
1. 用 `load()` 快速迭代开发
2. 用 `setup.py` 打包发布
3. 文件名统一（如 `op_name.cu` + `op_name.cpp`）
4. 利用 Python 脚本自动扫描，避免硬编码

如果你希望，我可以为你生成一个完整的模板仓库（包括 `compile_dev.py` 自动生成脚本），你只需往 `kernels/` 和 `bindings/` 丢文件就能用。欢迎继续提问！