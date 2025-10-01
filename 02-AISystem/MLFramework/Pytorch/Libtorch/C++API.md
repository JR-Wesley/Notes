---
dateCreated: 2025-07-31
dateModified: 2025-07-31
---
# 基于Pytorch C++的高性能计算、自定义算子开发
## 介绍

> 📚 官方文档：[LibTorch C++ API](https://pytorch.org/cppdocs/)

1. **官方文档**
   - C++ API 文档：https://pytorch.org/cppdocs/
   - 教程：https://pytorch.org/tutorials/advanced/cpp_frontend.html

2. **示例项目**
   - LibTorch examples 目录中的 `mnist`, `imagenet`, `cpp-frontend`


## 🧱 一、掌握的核心知识体系

### 1. **C++14/17 基础（必须）**

   - LibTorch 要求 C++14 以上，推荐使用 C++17。
   - 熟悉智能指针（`std::shared_ptr`, `std::unique_ptr`）、RAII、模板、STL 容器。
- ✅ 基础语法：变量、循环、函数、指针、引用
- ✅ 面向对象：类、继承、多态
- ✅ 模板（Templates）：函数模板、类模板（非常重要！）
- ✅ STL：`vector`, `string`, `memory`（智能指针 `shared_ptr`, `unique_ptr`）
- ✅ C++11/14/17 特性：`auto`, `lambda`, `move semantics`, `rvalue references`

---

### 2. **PyTorch 的 Python API（熟悉）**

   - Tensor、Autograd、Module（nn::Module）、Optimizer、Dataset、DataLoader
   - 模型训练/推理流程
- ✅ `torch.Tensor` 的创建、操作、设备（CPU/GPU）
- ✅ `torch.nn.Module`, `forward` 方法
- ✅ `torch.jit.trace`, `torch.jit.script`（用于模型序列化）
- ✅ `torch.utils.cpp_extension` 的使用（如 `load_inline`, `CUDAExtension`）

---

### 3. **LibTorch C++ API（核心）**

| 模块                                         | 内容                                    |
| ------------------------------------------ | ------------------------------------- |
| `#include <torch/torch.h>`                 | 主头文件                                  |
| `torch::Tensor`                            | C++ 中的张量类型，对应 Python 的 `torch.Tensor` |
| `torch::nn::Module`                        | 定义神经网络模块                              |
| `torch::nn::Linear`, `torch::nn::Conv2d` 等 | 常见层                                   |
| `torch::optim::SGD`, `Adam`                | 优化器                                   |
| `torch::save()`, `torch::load()`           | 模型保存与加载                               |
| `torch::jit::load()`                       | 加载 Python 导出的 `.pt` 模型                |

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

## 💡 小贴士

- 从 **CPU 版本开始**，再扩展到 CUDA
- 使用 `torch::without_grad()` 减少内存开销
- 打印 tensor 用 `std::cout << tensor << std::endl;`
- 调试时用 `assert(tensor.defined())` 检查空指针

---
# 核心用法


要学习 PyTorch 的 C++ 前端（即 **LibTorch**），并熟练使用 C++ 的 Torch 库进行深度学习模型的部署与开发，你需要系统掌握以下几个方面的核心内容和 API。LibTorch 是 PyTorch 的官方 C++ 接口，提供了与 Python 前端几乎对等的功能，适合高性能、低延迟的生产环境。

---
## 核心模块
### LibTorch 核心模块与 API

#### 1. **Tensor 操作（`torch::Tensor`）**

这是最核心的数据结构，类似于 NumPy 或 Python 版 PyTorch 中的 `torch.Tensor`。

- **创建 Tensor**
  ```cpp
  torch::Tensor t = torch::rand({2, 3});
  torch::Tensor zeros = torch::zeros({3, 4}, torch::kFloat32);
  torch::Tensor from_data = torch::from_blob(data_ptr, {batch, channels, h, w}, torch::kFloat32);
  ```

- **常用操作**
  - 数学运算：`t.add()`, `t.mul()`, `t.pow()`, `torch::sigmoid(t)`
  - 形状操作：`t.view()`, `t.reshape()`, `t.transpose()`, `t.permute()`
  - 设备管理：`t.to(torch::kCUDA)`, `t.to(torch::kCPU)`
  - 数据访问：`t.accessor<float, 2>()`, `t.data_ptr<float>()`

- **类型与设备**
  - `torch::kFloat32`, `torch::kInt64`, `torch::kCUDA`, `torch::kCPU`

#### 2. **神经网络模块（`torch::nn`）**

使用 `nn::Module` 构建自定义网络。

- **常用层**
  ```cpp
  struct Net : torch::nn::Module {
      Net() {
          fc1 = register_module("fc1", torch::nn::Linear(784, 64));
          fc2 = register_module("fc2", torch::nn::Linear(64, 10));
          conv1 = register_module("conv1", torch::nn::Conv2d(1, 32, 3));
      }
      torch::Tensor forward(torch::Tensor x) {
          x = torch::relu(fc1->forward(x));
          x = fc2->forward(x);
          return torch::log_softmax(x, /*dim=*/1);
      }
      torch::nn::Linear fc1{nullptr}, fc2{nullptr};
      torch::nn::Conv2d conv1{nullptr};
  };
  ```

- **常见模块**
  - `torch::nn::Linear`, `Conv1d/2d/3d`, `BatchNorm`, `Dropout`, `ReLU`, `Sigmoid`, `MaxPool`, `AdaptiveAvgPool`

- **容器**
  - `torch::nn::Sequential`, `torch::nn::ModuleList`

#### 3. **模型训练（Optimizer, Loss, Autograd）**

- **损失函数**
  ```cpp
  auto loss = torch::nll_loss(output, target);
  auto mse_loss = torch::mse_loss(pred, label);
  ```

- **优化器**
  ```cpp
  torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(0.001));
  // 或 SGD
  torch::optim::SGD optimizer(net->parameters(), 0.01);
  ```

- **训练循环**
  ```cpp
  for (int epoch = 1; epoch <= 10; ++epoch) {
      for (auto &batch : data_loader) {
          optimizer.zero_grad();
          auto output = net->forward(batch.data);
          auto loss = torch::nll_loss(output, batch.target);
          loss.backward();
          optimizer.step();
      }
  }
  ```

#### 4. **数据加载（Dataset & DataLoader）**

LibTorch 提供了 C++ 版本的 `Dataset` 和 `DataLoader`。

- 自定义 Dataset：
  ```cpp
  class MyDataset : public torch::data::Dataset<MyDataset> {
  public:
      torch::data::Example<> get(size_t index) override;
      torch::optional<size_t> size() const override;
  };
  ```

- 使用 DataLoader：
  ```cpp
  auto data_loader = torch::data::make_data_loader(
      std::move(dataset), 
      torch::data::DataLoaderOptions().batch_size(32).workers(2)
  );
  ```

#### 5. **模型保存与加载**

- **保存模型参数**
  ```cpp
  torch::save(net, "model.pt");
  // 或只保存参数
  torch::save(net->state_dict(), "weights.pt");
  ```

- **加载模型**
  ```cpp
  torch::load(net, "model.pt");
  // 或
  torch::load(net->state_dict(), "weights.pt");
  ```

#### 6. **模型部署（TorchScript）**

这是 C++ 部署的关键！Python 训练好的模型需转换为 TorchScript。

- **在 Python 中导出 Script Module**
  ```python
  model = MyModel()
  model.eval()
  example = torch.rand(1, 3, 224, 224)
  traced_script_module = torch.jit.trace(model, example)
  traced_script_module.save("model.pt")
  ```

- **C++ 中加载并推理**
  ```cpp
  torch::jit::script::Module module = torch::jit::load("model.pt");
  module.to(torch::kCUDA);  // 如果使用 GPU
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::randn({1, 3, 224, 224}).to(torch::kCUDA));
  at::Tensor output = module.forward(inputs).toTensor();
  ```


---

### 总结：你需要掌握的核心 API

| 类别 | 关键 API / 类 |
|------|-------------|
| Tensor | `torch::Tensor`, `torch::rand`, `view`, `to`, `data_ptr`, `accessor` |
| 网络 | `torch::nn::Module`, `Linear`, `Conv2d`, `Sequential`, `register_module` |
| 训练 | `torch::optim::Adam/SGD`, `backward()`, `zero_grad()`, `step()` |
| 损失 | `torch::nll_loss`, `torch::mse_loss`, `torch::cross_entropy` |
| 数据 | `torch::data::Dataset`, `DataLoader` |
| 模型 | `torch::save`, `torch::load`, `torch::jit::load`, `forward()` |
| 部署 | `torch::jit::script::Module`, `IValue`, `trace`, `script` |

---

✅ **建议学习路径**：
1. 先跑通一个 LibTorch 示例（如 MNIST）
2. 学会加载 TorchScript 模型并推理
3. 尝试用 C++ 从头定义并训练一个简单网络
4. 集成到实际项目（如 OpenCV 图像分类）

掌握这些内容后，你就可以在 C++ 环境中高效使用 PyTorch 进行深度学习开发与部署了。

# Tensor
`torch::Tensor` 是 LibTorch C++ 前端中最核心的数据结构，它表示一个多维数组（张量），支持 GPU 加速、自动求导（autograd）以及丰富的数学和逻辑操作。其 API 设计与 Python 版 `torch.Tensor` 高度一致，但在 C++ 中以函数式或方法式调用。

下面详细介绍 `torch::Tensor` 提供的主要操作类别、常用 API 及其**输出格式/返回类型**。

---

## 一、基本属性查询（返回张量元信息）

这些 API 返回的是**标量值或 `c10::IntArrayRef` 等非张量类型**，用于获取 Tensor 的元数据。

| API | 说明 | 返回类型 | 示例输出 |
|-----|------|----------|---------|
| `.sizes()` | 返回各维度大小 | `c10::IntArrayRef` (类似 `std::vector<int64_t>`) | `{3, 224, 224}` |
| `.size(dim)` | 返回第 `dim` 维的大小 | `int64_t` | `224` |
| `.dim()` / `.ndimension()` | 返回维度数 | `int64_t` | `3` |
| `.numel()` | 返回总元素个数 | `int64_t` | `3*224*224 = 150528` |
| `.defined()` | 是否为有效 Tensor | `bool` | `true` |
| `.dtype()` | 数据类型 | `c10::ScalarType` | `kFloat`, `kLong`, `kInt` 等 |
| `.device()` | 设备类型 | `c10::Device` | `cpu`, `cuda:0` |
| `.requires_grad()` | 是否需要梯度 | `bool` | `true` |
| `.is_cuda()` | 是否在 CUDA 上 | `bool` | `true` |

---

## 二、创建操作（返回新 `torch::Tensor`）

这些函数返回一个**新的 `torch::Tensor` 对象**。

| API | 说明 | 输出格式（返回 Tensor 的形状/类型） |
|------|------|-----------------------------|
| `torch::rand({H, W})` | 生成 [0,1) 均匀分布随机数 | 形状 `{H, W}`，类型 `kFloat32` |
| `torch::randn({B,C,H,W})` | 标准正态分布 | 形状 `{B,C,H,W}`，类型 `kFloat32` |
| `torch::zeros({2,3})` | 全零张量 | 形状 `{2,3}`，类型 `kFloat32` |
| `torch::ones({N})` | 全一张量 | 形状 `{N}`，类型 `kFloat32` |
| `torch::full({2,2}, 5.0)` | 填充指定值 | 形状 `{2,2}`，值全为 `5.0` |
| `torch::arange(0, 10, 2)` | 等差序列 | 形状 `{5}`，值 `[0,2,4,6,8]`，类型 `kInt64` |
| `torch::eye(3)` | 单位矩阵 | 形状 `{3,3}`，对角为 1，其余为 0 |

> ✅ 所有创建函数都可指定 `dtype` 和 `device`：
> ```cpp
> torch::zeros({2,2}, torch::kFloat64).to(torch::kCUDA);
> ```

---

## 三、数学运算（返回新 `torch::Tensor`）

返回结果为**同类型、广播后形状的新张量**。

### 1. 基础算术
| API | 说明 | 输出格式 |
|------|------|---------|
| `a.add(b)` 或 `a + b` | 加法 | 广播后形状，类型与输入一致 |
| `a.sub(b)` 或 `a - b` | 减法 | 同上 |
| `a.mul(b)` 或 `a * b` | 逐元素乘法 | 同上 |
| `a.div(b)` 或 `a / b` | 逐元素除法 | 同上 |
| `a.pow(2)` | 幂运算 | 形状不变，类型不变 |

### 2. 激活函数
| API | 说明 | 输出格式 |
|------|------|---------|
| `torch::relu(x)` | ReLU | 形状同 `x`，类型同 `x` |
| `torch::sigmoid(x)` | Sigmoid | 同上 |
| `torch::tanh(x)` | Tanh | 同上 |
| `torch::softmax(x, dim)` | Softmax 沿 `dim` 轴 | 形状同 `x` |
| `torch::log_softmax(x, dim)` | Log-Softmax | 同上 |

### 3. 线性代数
| API | 说明 | 输出格式 |
|------|------|---------|
| `torch::matmul(a, b)` | 矩阵乘法 | `(M×N) @ (N×P) → (M×P)` |
| `torch::mm(a, b)` | 仅限 2D 矩阵乘 | `(M×N) × (N×P) → (M×P)` |
| `torch::bmm(a, b)` | 批量矩阵乘 | `(B×N×M) × (B×M×P) → (B×N×P)` |
| `torch::dot(a, b)` | 向量点积 | 标量（0D Tensor） |

---

## 四、形状变换操作（返回新 `torch::Tensor`）

返回**视图或新内存张量**，形状改变，但数据不变（除非 `clone()`）。

| API | 说明 | 输出格式 |
|------|------|---------|
| `x.view({a,b,c})` | 重塑形状（共享内存） | 新形状 `{a,b,c}`，总元素数不变 |
| `x.reshape({a,b})` | 类似 `view`，必要时复制 | 新形状 `{a,b}` |
| `x.transpose(0,1)` | 交换两个维度 | 形状中 `dim0` 和 `dim1` 互换 |
| `x.permute({2,0,1})` | 重排所有维度 | 按指定顺序排列维度 |
| `x.unsqueeze(0)` | 插入长度为 1 的维度 | 形状增加一维，如 `(H,W) → (1,H,W)` |
| `x.squeeze()` | 移除长度为 1 的维度 | 如 `(1,3,1,4) → (3,4)` |
| `x.flatten(start, end)` | 展平指定维度范围 | 如 `flatten(1,2)` 将第1~2维合并 |

> ⚠️ `view` 要求张量是**连续的（contiguous）**，否则需先调用 `.contiguous()`

---

## 五、索引与切片操作

返回**新张量或可写视图**。

| API | 说明 | 返回类型 |
|------|------|----------|
| `x.index({0})` | 索引第0个元素（支持多维） | `torch::Tensor` |
| `x.slice(dim, start, end, step)` | 切片操作 | 新张量（可共享内存） |
| `x[0]` | 运算符重载切片 | `torch::Tensor` |
| `x.at({0,1})` | 访问单个元素值（慢） | `at::Tensor` 内部值（如 `float`） |

示例：
```cpp
auto row = x.slice(0, 0, 1);     // 第0行
auto col = x.index({Slice(), 1}); // 第1列
```

---

## 六、类型与设备转换

返回**新 Tensor**，可能位于不同设备或具有不同数据类型。

| API | 说明 | 返回类型 |
|------|------|----------|
| `x.to(torch::kFloat64)` | 转换数据类型 | 新 `Tensor`，类型为 `kFloat64` |
| `x.to(torch::kCUDA)` | 移动到 GPU | 新 `Tensor` 在 CUDA 上 |
| `x.cpu()` | 移回 CPU | CPU 上的 `Tensor` |
| `x.cuda()` | 移到 CUDA | CUDA 上的 `Tensor` |
| `x.type_as(another)` | 转为与 another 相同类型 | 类型匹配的新 Tensor |

> ✅ 可链式调用：`x.to(torch::kFloat32).to(torch::kCUDA)`

---

## 七、聚合操作（Reduction）

返回**降维或标量 Tensor**。

| API | 说明 | 输出格式 |
|------|------|---------|
| `x.sum()` | 所有元素和 | 0D Tensor（标量） |
| `x.sum(dim)` | 沿 `dim` 求和 | 形状中 `dim` 维被移除 |
| `x.mean()` | 平均值 | 0D Tensor |
| `x.mean(dim)` | 沿 `dim` 平均 | 形状减少一维 |
| `x.max()` | 最大值 | 0D Tensor |
| `x.max(dim)` | 返回最大值和索引 | `std::tuple<torch::Tensor, torch::Tensor>` |
| `x.argmax(dim)` | 最大值索引 | 形状减少一维，类型 `kInt64` |
| `x.norm()` | L2 范数 | 0D Tensor |

示例：
```cpp
auto s = x.sum(0); // shape: (H,W) → (W,) if dim=0
```

---

## 八、数据访问与互操作

用于与 C++ 原生数据交互。

| API | 说明 | 返回类型 |
|------|------|----------|
| `x.data_ptr<T>()` | 获取原始数据指针 | `T*`，如 `float*` |
| `x.accessor<T, Dims>()` | 安全多维索引访问 | `PackedTensorAccessor<T, Dims>` |
| `torch::from_blob(ptr, sizes, options)` | 从内存块创建 Tensor | `torch::Tensor`（不拥有内存） |

示例：
```cpp
float* data = tensor.data_ptr<float>();
auto acc = tensor.accessor<float, 2>(); // 2D 访问
float val = acc[0][1]; // 安全访问 (0,1)
```

---

## 九、自动求导相关

| API | 说明 | 返回类型 |
|------|------|----------|
| `x.requires_grad_(true)` | 开启梯度记录 | `torch::Tensor`（in-place 修改） |
| `x.grad()` | 获取梯度 | `torch::Tensor` 或 `undefined` |
| `x.backward()` | 反向传播 | `void`（触发计算图反向） |

---

## 十、其他常用操作

| API | 说明 | 输出格式 |
|------|------|----------|
| `x.clone()` | 深拷贝 | 新内存，形状类型相同 |
| `x.detach()` | 分离计算图 | 不参与梯度的新 Tensor |
| `torch::cat({a,b}, dim)` | 拼接张量 | 沿 `dim` 扩展的新形状 |
| `torch::stack({a,b}, dim)` | 堆叠（新增维度） | 多一维的新张量 |
| `x.is_contiguous()` | 是否连续存储 | `bool` |
| `x.contiguous()` | 确保连续 | 新 Tensor（若不连续则复制） |

---

## 总结：API 输出格式规律

| 操作类型 | 返回值类型 | 是否共享内存 | 常见用途 |
|---------|-----------|--------------|----------|
| **创建** | `torch::Tensor` | 否 | 初始化 |
| **数学/激活** | `torch::Tensor` | 否（新结果） | 计算 |
| **形状变换** | `torch::Tensor` | 通常是（视图） | Reshape |
| **索引/切片** | `torch::Tensor` | 是（视图） | 子张量 |
| **Reduction** | `torch::Tensor`（0D 或降维） | 否 | 统计 |
| **类型/设备转换** | `torch::Tensor` | 否（新对象） | 部署 |
| **属性查询** | 基本类型（`int64_t`, `bool` 等） | — | 调试 |

---

✅ **最佳实践建议**：

1. 使用 `auto` 接收返回值，避免手动指定类型。
2. 注意 `in-place` 操作（如 `add_()`）会修改原 Tensor。0
3. 推理时使用 `torch::NoGradGuard no_grad;` 禁用梯度以提升性能。
