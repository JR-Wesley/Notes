# FP8 格式转换

```python
[utils.py](http://utils.py/) 46-52 ​
```

## 函数定义与输入验证

```

def per_token_cast_to_fp8(x: torch.Tensor):

assert x.dim() == 2 and x.size(1) % 128 == 0

```

- **输入验证**：`assert` 语句确保输入满足两个条件：

1. 张量必须是 2 维的 (`x.dim() == 2`)

2. 第二维度大小必须是 128 的倍数 (`x.size(1) % 128 == 0`)，这是后续按 128 元素分组处理的基础

## 张量重塑与分组

```

m, n = x.shape

x_view = x.view(m, -1, 128)

```

- **语法解析**：使用张量的 `.shape` 属性获取维度信息，通过 `.view()` 方法重塑张量形状

- **实现逻辑**：

- 将输入张量 `(m, n)` 重塑为 `(m, k, 128)` 的三维结构，其中 `k = n/128`

- 这种分组方式与 NVIDIA 的 TensorRT-LLM 中 fp8 量化的 128 元素块处理策略一致

## 动态范围计算

```

x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)

```

- **语法解析**：链式调用多个张量方法，涉及维度操作和数值处理

- **关键步骤**：

1. `.abs()`：计算每个元素的绝对值

2. `.float()`：转换为 float32 类型以保证计算精度

3. `.amax(dim=2)`：沿最后一个维度（128 元素组内）计算最大值

4. `.view(m, -1)`：调整形状为 `(m, k)`

5. `.clamp(1e-4)`：限制最小值为 1e-4，避免后续除法中出现零或极小值

## FP8 转换核心实现

```

return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)

```

- **语法解析**：使用张量广播机制进行缩放计算，然后转换数据类型并重塑回原始形状

- **FP8 转换关键逻辑**：

1. **缩放因子计算**：`448.0 / x_amax.unsqueeze(2)`

- `.unsqueeze(2)` 增加维度使形状从 `(m,k)` 变为 `(m,k,1)`，实现与 `(m,k,128)` 的广播运算

- 448 是 float8_e4m3fn 格式的最大正值（2^7 - 2^-3 = 127.875？此处可能是特定实现的缩放系数）

2. **数据缩放**：`x_view * (缩放因子)` 将每个 128 元素组归一化到 [-448, 448] 范围

3. **类型转换**：`.to(torch.float8_e4m3fn)` 将数据转换为 fp8 类型，具体使用：

- `float8_e4m3fn` 格式：4 位指数、3 位尾数的无符号 fp8 类型

- 这种格式在 NVIDIA GPU 中硬件加速支持较好

4. **形状恢复**：`.view(m, n)` 将三维张量重塑回原始二维形状

5. **返回缩放参数**：同时返回 `(x_amax / 448.0)` 作为反量化时使用的缩放因子

## 整体转换流程总结

1. 输入验证确保数据满足 128 元素分组要求

2. 按 128 元素分组计算每组的动态范围 (amax)

3. 将每组数据缩放到 fp8 格式的表示范围内

4. 执行类型转换并恢复原始张量形状

5. 返回 fp8 张量和对应的缩放因子用于后续反量化

这种实现采用了 per-group（每 128 元素组）的动态量化策略，相比 per-tensor 量化能更好地保留局部动态范围，是 LLM 推理中常用的 fp8 转换方案。

# Utils

## `init_dist`

### **1. `init_dist(local_rank, num_local_ranks)` 函数作用**

`init_dist` 是 DeepEP 库自定义的一个**分布式环境初始化函数**，主要负责初始化多进程通信所需的基础组件，为后续分布式操作（如数据同步、集体通信）做准备。具体作用包括：

- **初始化通信后端**：启动分布式通信所需的后端（如 NCCL 或 Gloo，通常用于 GPU 间通信）。
- **设置进程标识**：确定当前进程的全局 `rank`（唯一标识）和总进程数 `num_ranks`（world size）。
- **绑定设备**：根据 `local_rank` 将当前进程绑定到指定 GPU（避免多进程抢占同一 GPU）。
- **创建通信组**：构建进程间的通信组 `group`，用于后续集体通信操作（如 `dist.all_reduce`、`dist.all_gather`）。

### **2. 返回值 `group` 的作用**

`group` 是 `torch.distributed.ProcessGroup` 类型的对象，代表**一组参与通信的进程集合**，是分布式通信的核心协调者。其具体作用如下：

#### **（1）限定集体通信的作用范围**

`group` 定义了哪些进程参与集体通信操作（如 `dist.all_reduce`、`dist.all_gather`）。例如：

```python

# 使用 group 限定 all_reduce 仅在该组内的进程间执行

gbl_num_tokens_per_expert = num_tokens_per_expert.clone()

dist.all_reduce(gbl_num_tokens_per_expert, group=group) # 仅 group 内的进程参与数据同步

```

若不指定 `group`，PyTorch 会默认使用全局通信组（包含所有进程），但 `init_dist` 返回的 `group` 可能是根据测试需求定制的子通信组（如仅包含当前节点内的进程，符合 `test_intranode.py` 的“节点内测试”场景）。

#### **（2）作为 `deep_ep.Buffer` 的通信句柄**

在创建 `deep_ep.Buffer`（专家并行通信缓冲区）时，`group` 被作为参数传入：

```python

buffer = deep_ep.Buffer(group, int(2e9), num_rdma_bytes, …)

```

`Buffer` 类需要通过 `group` 获取进程间的通信上下文（如通信后端、进程拓扑），以实现跨进程的数据分发（dispatch）和聚合（combine），这是专家并行（EP）中“跨进程路由 token 到对应专家”的核心依赖。

#### **（3）确保分布式操作的一致性**

后续所有依赖分布式协调的操作（如性能调优时的配置同步、测试结果验证）均基于 `group` 进行。例如，在收集最佳配置时：

```python

# 基于 group 收集所有进程的最佳配置

dist.all_gather(all_best_fp8_results_list, best_dispatch_results, group=group)

```

`group` 确保了不同进程间的数据交换仅在预设的通信范围内生效，避免跨组干扰，同时保证通信效率。

### 3. Torch 进程组

`torch.distributed.ProcessGroup` 是 PyTorch 中实现分布式训练的核心抽象概念，它定义了一组参与通信的进程（processes）以及它们之间的通信方式。通过 `ProcessGroup`，PyTorch 能够在多机多卡环境中高效协调计算资源，实现数据并行、模型并行等多种分布式训练策略。一个分布式作业里可以有多个进程组，每个进程组包含不同的进程子集，各进程组相互独立。**核心作用**如下：

#### **(1) 通信范围划分**

- 在分布式训练中，进程组定义了通信的范围。将多个进程划分为不同的组（例如，将 8 个进程分为 2 个组，每组 4 个进程），每个组形成一个独立的通信域。不同组的进程可以独立进行通信，互不干扰。
- 例如：
- **默认进程组**：通过 `torch.distributed.init_process_group` 初始化的全局进程组，通常包含所有参与训练的进程。
- **自定义进程组**：通过 `torch.distributed.new_group` 创建的子组，用于特定的通信需求（如混合并行（数据并行 + 模型并行）中，不同层的模型参数在独立的组内同步）。

#### **(2) 支持通信操作**

- 封装底层通信实现（如 NCCL、GLOO、MPI），提供一致的 API（如 `all_reduce`、`broadcast`、`send/recv`），使代码不依赖于具体的通信后端。
- **NCCL**：NVIDIA GPU 间的高性能通信，支持 GPUDirect RDMA。
- **GLOO**：跨平台（CPU/GPU）通信，适合小规模集群和快速原型。
- **MPI**：支持异构环境和复杂网络拓扑。
- 所有集合通信操作（如 `all_reduce`、`broadcast`、`all_gather` 等）都依赖于 `ProcessGroup` 来指定通信的进程范围。
- 例如，`dist.all_reduce(tensor, group=group)` 会将 `tensor` 在指定的 `group` 中进行归约操作。
- 点对点通信操作 **Send/Recv**：进程间直接发送和接收数据。

_应用_：流水线并行中不同阶段之间的数据传输。

#### **(3) 灵活的分布式策略**

- 通过划分不同的进程组，可以实现更复杂的分布式策略，例如：
- **数据并行**：所有进程属于同一组，同步梯度。
- **模型并行**：不同组处理模型的不同部分（如不同层），独立通信。

## `per_token_cast_to_fp8`

### **1. 输入输出概述**

- **输入**：`x: torch.Tensor`（2D 张量，形状 `[num_tokens, hidden]`，`hidden % 128 == 0`，数据类型为 BF16/FP32）。
- **输出**：元组 `(fp8_data_tensor, scales_tensor)`，其中：
- `fp8_data_tensor`：量化后的 FP8 数据（E4M3 格式），形状与输入 `x` 一致 `[num_tokens, hidden]`。
- `scales_tensor`：每个量化块的缩放因子，形状 `[num_tokens, num_groups]`（`num_groups = hidden / 128`）。

### **2. 核心变量形状与变换逻辑**

（`m=num_tokens`, `n=hidden`, `g=num_groups=n/128`）

- `x`: `(m, n)`
- 输入张量，需满足 `n % 128 == 0`（按 128 元素分块量化的前提）。
- `x_view`: `(m, g, 128)`
- 将 `x` 按隐藏维度分块：`x.view(m, -1, 128)`，其中 `-1` 自动计算为 `g = n/128`。例如 `n=512` 时 `g=4`，`x_view` 形状为 `(m, 4, 128)`。
- `x`: `(m, n)`
- 输入张量，需满足 `n % 128 == 0`（按 128 元素分块量化的前提）。
- `x_view`: `(m, g, 128)`
- 将 `x` 按隐藏维度分块：`x.view(m, -1, 128)`，其中 `-1` 自动计算为 `g = n/128`。例如 `n=512` 时 `g=4`，`x_view` 形状为 `(m, 4, 128)`。
- `x_amax`: `(m, g)`
- 计算每个 128 元素块的绝对值最大值（`amax`）
- `x_view.abs().float()`：转 FP32 避免精度损失；
- `amax(dim=2)`：沿第 2 维（128 元素块）取最大值，得到 `(m, g)`；
- `clamp(1e-4)`：限制最小值为 `1e-4`，避免后续除零错误。
- `fp8_data_tensor`: `(m, n)`
- 量化后的数据
- `x_view * (448.0 / x_amax.unsqueeze(2))`：将每个块缩放到 E4M3 范围（[-448, 448]），`x_amax.unsqueeze(2)` 扩展为 `(m, g, 1)` 以广播到 `x_view` 的 `(m, g, 128)`；
- `.to(torch.float8_e4m3fn)`：转换为 E4M3 FP8 格式；
- `.view(m, n)`：恢复原始形状。
- `scales_tensor`: `(m, g)`
- 缩放因子（用于反量化）：
- `(x_amax / 448.0).view(m, -1)`：计算量化时的缩放系数倒数（`1/scale`），形状保持 `(m, g)`。

### **3. 为何返回元组？**

FP8 量化是**有损压缩**，需同时存储：

- **量化后的数据**（`fp8_data_tensor`）：用 1 字节/元素存储，相比 BF16（2 字节）节省 50% 内存。
- **缩放因子**（`scales_tensor`）：记录每个 128 元素块的动态范围（`x_amax / 448.0`），用于反量化时恢复原始数据精度（通过 `per_token_cast_back` 函数）。

二者缺一不可，因此返回元组 `(fp8_data_tensor, scales_tensor)`。

### **4. 关键设计细节**

- **分块量化（128 元素/块）**：隐藏维度按 128 元素分块（`x_view`），平衡量化精度与计算效率（块太小则缩放因子存储开销大，块太大则精度损失严重）。
- **E4M3 格式适配**：E4M3 FP8 的动态范围为 `[-448, 448]`，因此通过 `448.0 / x_amax` 将每个块的最大值归一化到 448，确保数据能被 FP8 精确表示。
- **数值稳定性**：`clamp(1e-4)` 避免 `x_amax` 过小导致的除零错误，`float()` 转换确保 `amax` 计算精度。

### **总结**

该函数通过**分块量化**将高 Precision 张量（BF16/FP32）压缩为 FP8（E4M3）格式，同时记录缩放因子，实现内存高效存储与后续精确恢复。返回元组是为了同时保留量化数据和反量化所需的动态范围信息。

---

# 参考

1. [理解DeepEP源码和节点通信逻辑](https://zhuanlan.zhihu.com/p/1890067712996270654)
