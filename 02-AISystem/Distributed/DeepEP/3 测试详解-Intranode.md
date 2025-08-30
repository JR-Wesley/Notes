# `intranode` 测试详解

> 本文对应版本 [f0d34aa](https://github.com/deepseek-ai/DeepEP/commit/f0d34aabcb7bdcb3a05d022e7d11b3bf4ccf8ee8)

下面基于 `tests/test_intranode.py` 讲解测试通信库的步骤，这个测试文件就调用了 `Buffer` 提供的相关接口进行对比和性能测试。关于 `Buffer` 的具体实现后续讲解。

## 一、实验设置

### 参数配置

```python
# tests\test_intranode.py
@@ -261, 10
```

这一段是测试开始前的一些参数。

1. **`'--num-processes', type=int, default=8`**
	- 要启动的进程数量。每个进程通常运行在不同的 GPU 或计算资源上，一般一个 GPU 一个进程。
2. **`'--num-tokens', type=int, default=4096`**
	- ：输入的令牌数量。令牌是模型处理的基本单位。
3. **`'--hidden', type=int, default=7168`**
	- 隐藏层的维度大小。隐藏层是输入层和输出层之间的中间层，决定了隐藏层中神经元的数量。
4. **`'--num-topk', type=int, default=8`**
	- 每个令牌对应的 Top-K 专家数量，每个令牌会根据得分选择与之最相关的 K 个专家进行处理。
5. **`'--num-experts', type=int, default=256`**
	- 模型中专家的总数。每个专家是一个独立的子模型 FFN。

#### 数据分配

根据上面的参数，可以得到我们需要处理的输入数据的格式，以及会生成一些用于 router 的信息的格式。

- **输入张量 `x`**：形状为 `(num_tokens, hidden)`，即 `(4096, 7168)`。每个进程会生成一个值为当前进程排名的 `x` 张量，其数据类型为 `torch.bfloat16`。
- **得分矩阵 `scores`**：形状为 `(num_tokens, num_experts)`，即 `(4096, 256)`。每个进程生成一个随机的得分矩阵，用于确定每个令牌对应的 Top-K 专家。
- **Top-K 索引 `topk_idx`**：通过对 `scores` 矩阵进行 `torch.topk` 操作得到，形状为 `(num_tokens, num_topk)`，即 `(4096, 8)`。它表示每个令牌对应的 Top-K 专家的索引。

#### 专家分配

### 创建多进程节点内专家并行 EP 测试

```python
# tests\test_intranode.py
@@ -275, 1
torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
```

`torch.multiprocessing.spawn` 用于在多进程环境下启动多个子进程并执行指定的函数（使用见 [[分布式]]）。

在实际执行时，`test_loop` 函数接收到的参数情况如下：

```python
# tests/test_intranode.py
@@ -233, 24

def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
```

- `local_rank`：由 `spawn` 函数自动传入的子进程全局排名。
- `num_local_ranks`：从 `args` 元组中获取的 `num_processes`。
- `args`：从 `args` 元组中获取的命令行参数解析结果。

`init_dist` 函数初始化分布式环境，获取当前进程的全局排名 `rank`、总进程数 `num_ranks` 以及通信组 `group`。

- `test_ll_compatibility`：一个布尔变量，用于控制是否进行低延迟功能测试，当前设置为 `False`，即不进行测试。
- `num_rdma_bytes`：RDMA 缓冲区所需的字节数，初始化为 0。若进行低延迟测试，调用 `deep_ep.Buffer.get_low_latency_rdma_size_hint` 函数计算所需的 RDMA 缓冲区大小。

> 关于初始化分布式环境见 [[5 Utils]]。

#### 创建 `Buffer` 实例

在这个例子中，由于设置了不进行低延迟相关测试，则剩余部分只包括：创建 `Buffer` 实例、设置随机种子、执行主测试函数、释放资源。

```python
buffer = deep_ep.Buffer(group, int(2e9), num_rdma_bytes, low_latency_mode=test_ll_compatibility,

num_qps_per_rank=(ll_num_experts // num_ranks if test_ll_compatibility else 1), explicitly_destroy=True)
```

创建 `deep_ep.Buffer` 实例，参数含义如下：

- `group`：通信组。
- `int(2e9)`：NVLink 缓冲区的字节数。
- `num_rdma_bytes`：RDMA 缓冲区的字节数。
- `low_latency_mode`：是否启用低延迟模式。
- `num_qps_per_rank`：每个进程的队列对数量，低延迟模式下根据专家数量计算，否则为 1。
- `explicitly_destroy`：是否需要显式调用 `destroy` 方法释放资源，设置为 `True`。

`Buffer` 的具体组成具体见 [[6 Runtime-Buffer]]，`intranode` 测试只有 `num_nvl_bytes = int(2e9)` 会对后面的测试起作用，RDMA 相关的参数可以不用考虑。

#### 执行主测试函数

```python
for i in (24, ):
	test_main(args, i, local_rank, num_ranks, rank, buffer, group)

if local_rank == 0:
	print('', flush=True)
```

- 遍历 `(24, )` 这个元组，将 `i` 作为多处理器数量传入 `test_main` 函数进行测试。本地排名为 0 的进程，打印一个空行。

## 二、测试配置与数据生成

上面做好了实验基本的参数配置和分布式进程初始化，在 `test_main` 函数中就是 DeepEP 主要测试功能函数的实现。这一部分定义了 `test_main` 函数，该函数的主要功能是对 `deep_ep.Buffer` 的分发（dispatch）和合并（combine）操作进行全面测试，同时对这些操作的性能进行调优。

下面解读 dispatch layout 计算，并和 `buffer` 实现对比，测试性能。

```python
# tests/test_intranode.py
@@ -15, 55
```

### 函数定义与初始化配置

```python
def test_main(args: argparse.Namespace, num_sms: int, local_rank: int, num_ranks: int, rank: int,
buffer: deep_ep.Buffer, group: dist.ProcessGroup):
```

- `args`：命令行参数解析后的命名空间对象。
- `num_sms`：流多处理器（SM）的数量，这个测试中在调用时指定了 `num_sums=24`。
- `local_rank`：当前进程在本地节点的排名。
- `num_ranks`：总进程数。
- `rank`：当前进程的全局排名。
- `buffer`：`deep_ep.Buffer` 实例，用于通信操作。
- `group`：分布式进程组。

```python
assert num_experts % num_ranks == 0
if local_rank == 0:
print(f'[config] num_tokens={num_tokens}, hidden={hidden}, num_topk={num_topk}', flush=True)
```

从命令行参数中获取令牌数量、隐藏维度大小、Top-K 专家数量和专家总数。确保专家总数能被进程数整除，并打印配置信息。

- 要求 `num_experts` 能被 `num_processes` 整除，在默认情况下 `256 % 8 == 0` 满足条件。每个进程负责一部分专家，专家数量为 `num_experts // num_processes = 256 // 8 = 32` 个。

### 数据生成

```python

# tests/test_intranode.py
@@ -25, 11
# Random data
```

这一部分，每一个 rank 都会各自生成各种输入数据，包括全 1 张量、随机张量、FP8 格式的张量、得分矩阵、Top-K 索引和权重等，并计算每个令牌对应的进程排名。

#### 1. 原始数据

-  `x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * rank``
	- **数据说明**：生成一个形状为 `(num_tokens, hidden)` 的张量，数据类型为 `torch.bfloat16`，放置在 CUDA 设备上。张量的所有元素初始值为 1，再乘以当前进程的全局排名 `rank`。
	- 其元素值与进程排名相关，方便后续验证数据分发和合并的正确性。
1. `x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
	- **数据说明**：生成一个形状为 `(num_tokens, hidden)` 的张量，数据类型为 `torch.bfloat16`，放置在 CUDA 设备上。张量元素服从标准正态分布。
	- 输入数据，用于测试分发和合并操作在随机数据下的表现。
2. `x_e4m3 = per_token_cast_to_fp8(x) if deep_ep.Buffer.is_sm90_compiled() else None x_e4m3 = (x_e4m3[0], x_e4m3[1].T.contiguous().T) if x_e4m3 is not None else None`
	- **数据说明**：首先检查 `deep_ep.Buffer.is_sm90_compiled()` 是否为 `True`，若为 `True`，调用 DeepEP 库定义的 `per_token_cast_to_fp8` 函数将 `x` 转换为 FP8（E4M3 格式）；否则，`x_e4m3` 为 `None`。若 `x_e4m3` 不为 `None`，对其第二个元素进行转置再转置操作，确保内存连续。
	- **作用**：作为 FP8 格式的输入数据，用于测试在低精度计算下分发和合并操作的正确性。确保仅在兼容的设备上启用 FP8，避免不支持硬件的错误。
	- **硬件支持**：SM90（如 H100 GPU）引入了专门的 FP8 Tensor Core，对 E4M3 格式提供原生支持，计算速度比 BF16 快得多。
	- **性能对比**：在 H100 上，FP8 的矩阵乘法吞吐量是 BF16 的 2 倍，能效比更高。

> 关于 e4m3 格式转换，详细见 [[5 Utils]]。

#### 2. 计分统计

```python
scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
```

- **数据说明**：生成一个形状为 `(num_tokens, num_experts)` 的张量，数据类型为 `torch.float32`，放置在 CUDA 设备上。张量元素服从标准正态分布，取绝对值后加 1，避免负值影响后续排序。
- **作用**：作为每个令牌对应每个专家的得分，用于确定每个令牌对应的 Top-K 专家。

```python
topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
```

- **数据说明**：对 `scores` 张量在最后一个维度（即每一行）上取前 `num_topk` 个最大值，返回这些值的索引。形状为 `(num_tokens, num_topk)`。
- **作用**：表示每个令牌对应的 Top-K 专家的索引。

```python

topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device='cuda') * rank

```

- **数据说明**：生成一个形状为 `(num_tokens, num_topk)` 的张量，数据类型为 `torch.float32`，放置在 CUDA 设备上。张量元素初始值为 1，再乘以当前进程的全局排名 `rank`。
- **作用**：作为每个令牌对应的 Top-K 专家的权重，用于测试分发和合并操作中权重的处理。

```python

topk_weights_pure_rand = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda')

```

- **数据说明**：生成一个形状为 `(num_tokens, num_topk)` 的张量，数据类型为 `torch.float32`，放置在 CUDA 设备上。张量元素服从标准正态分布。
- **作用**：作为纯随机的 Top-K 专家权重，用于测试在随机权重下分发和合并操作的表现。

#### 3. 专家索引到计算设备

这一步计算就把 token 到 expert 的选择关系 `topk_idx` 转化为了 **token 到 rank/device 的选择关系**。

```python

rank_idx = topk_idx // (num_experts // num_ranks)

rank_idx.masked_fill_(topk_idx == -1, -1)

inplace_unique(rank_idx, num_ranks)

```

- **数据说明**：
- 第一行：计算每个 Top-K 专家所在的进程排名。
- 第二行：将 `topk_idx` 中值为 -1 的位置对应的 `rank_idx` 元素也设为 -1。
- 第三行：调用 DeepEP 库定义的 `inplace_unique` 函数对 `rank_idx` 进行原地去重操作，确保每个令牌对应的进程排名唯一。
- **作用**：表示每个令牌对应的 Top-K 专家所在的进程排名，用于后续计算每个进程的令牌数量和布局信息。
- `rank_idx` 表示每个 Top-K 专家所在的进程排名，通过 `topk_idx // (num_experts // num_processes)` 计算得到。后续会根据 `rank_idx` 进行数据的分发和合并操作，确保每个令牌能被正确分配到对应的进程和专家进行处理。
- 这里的专家映射情况就是直接的线性映射，如：
- 8 个专家（`num_experts = 8`），4 个计算设备（`num_ranks = 4`），因此每个 rank 负责 2 个专家（`8 // 4 = 2`）

```

专家索引 | 对应rank

0 1 | 0

2 3 | 1

4 5 | 2

6 7 | 3

```

### 全局元数据计算 Layout

这一部分主要是调用了 `deep_ep.cpp` 接口的 `buffer.get_dispatch_layout`。

```python

# tests/test_intranode.py

@@ -38, 20

# Expert meta

# Rank layout meta

```

#### 令牌计算

每个 rank 都各自计算每个专家和每个进程的令牌数量，并通过 `dist.all_reduce` 进行全局同步。

1. **计算每个专家的令牌数量**

```python

num_tokens_per_expert = torch.zeros((num_experts, ), [dtype=torch.int](http://dtype=torch.int/), device='cuda')

for i in range(num_experts):

num_tokens_per_expert[i] = (topk_idx == i).sum()

```

创建一个形状为 `(num_experts,)` 的全零张量，数据类型为 `torch.int`，放置在 CUDA 设备上，用于存储每个专家对应的令牌数量。

遍历每个专家，对于每个专家 `i`，统计 `topk_idx` 中值等于 `i` 的元素数量，即该专家对应的令牌数量。`topk_idx` 是一个形状为 `(num_tokens, num_topk)` 的张量，表示每个令牌对应的 Top-K 专家的索引。

1. **全局同步**

```python

gbl_num_tokens_per_expert = num_tokens_per_expert.clone()

dist.all_reduce(gbl_num_tokens_per_expert, group=group)

```

先克隆 `num_tokens_per_expert` 张量得到 `gbl_num_tokens_per_expert`，然后使用 `dist.all_reduce` 函数对 `gbl_num_tokens_per_expert` 进行全局归约操作，将所有进程中每个专家的令牌数量累加起来，得到全局每个专家的令牌数量。`group` 是进程组，确保归约操作在指定的进程组内进行。

#### Layout 计算

随后计算每个进程的令牌数量、令牌在进程中的索引，以及判断每个令牌是否在某个进程中，并进行全局同步。

1. **初始化张量**

```python

num_tokens_per_rank = torch.empty((num_ranks, ), [dtype=torch.int](http://dtype=torch.int/), device='cuda')

token_idx_in_rank = torch.full((num_ranks, num_tokens), -1, dtype=torch.long, device='cuda')

```

- `num_tokens_per_rank`：创建一个形状为 `(num_ranks,)` 的空张量，数据类型为 `torch.int`，放置在 CUDA 设备上，用于存储每个进程的令牌数量。
- `token_idx_in_rank`：创建一个形状为 `(num_ranks, num_tokens)` 的张量，数据类型为 `torch.long`，放置在 CUDA 设备上，初始值全为 -1，用于存储每个令牌在进程中的索引。

1. **计算每个进程的令牌数量和令牌索引**

```python

for i in range(num_ranks):

num_tokens_per_rank[i] = (rank_idx == i).sum()

token_sel = (rank_idx == i).max(dim=-1)[0]

count = token_sel.sum().item()

tokens = torch.sort(token_sel.to([torch.int](http://torch.int/)), descending=True)[1]

tokens[:count] = torch.sort(tokens[:count])[0]

token_idx_in_rank[i][tokens[:count]] = torch.arange(count, dtype=torch.long, device='cuda')

```

遍历每个进程，对于每个进程 `i`：

- `num_tokens_per_rank[i] = (rank_idx == i).sum()`：统计 `rank_idx` 中值等于 `i` 的元素数量，即该进程对应的令牌数量。`rank_idx` 是一个形状为 `(num_tokens, num_topk)` 的张量，表示每个令牌对应的 Top-K 专家所在的进程排名。
- `token_sel = (rank_idx == i).max(dim=-1)[0]`：对于每个令牌，判断其是否有对应的 Top-K 专家在进程 `i` 中，得到一个形状为 `(num_tokens,)` 的布尔张量。
- `rank_idx` 维度 `(#tokens, #topk)` 表示每个 tokens 选择的专家索引。
- 第一步比较生成 `bool` 类型，`True` 表示 token 选择了 `rank[i]`，布尔类型在 `max` 计算时会被转为 0、1，所以等价于在 `num_topk` 维度上取 `or` 操作。
- 结果是一个元组 `(values, indices)`，其中 `values` 是每个 token 在 `num_topk` 维度上的最大值（`1` 或 `0`），形状为 `(num_tokens,)`。
- **`[0]`**：取元组的第一个元素（即 `values`），得到形状为 `(num_tokens,)` 的布尔张量（`1` 对应 `True`，`0` 对应 `False`）。
- `token_sel` 维度 `(#tokens, )`，每一个元素表示，对 `rand_idx` 每一行/每一个 `token` 判断该行的 `topk` 中是否有 `rank i`。
- `count = token_sel.sum().item()`：统计 `token_sel` 中 `True` 的数量，即该进程对应的令牌数量。
- `tokens = torch.sort(token_sel.to([torch.int](http://torch.int/)), descending=True)[1]`：将 `token_sel` 转换为整数类型，然后进行降序排序，得到排序后的索引。
- 排序后返回一个元组 `(sorted_values, sorted_indices)`：
- `sorted_values`：排序后的值（例如 `[1,1,0,0]`）；
- `sorted_indices`：原始张量中元素的索引（即 “哪些 token 是 1，哪些是 0”）。
- `[1]`：取元组的第二个元素（即 `sorted_indices`），得到形状为 `(num_tokens,)` 的索引张量。
- `tokens[:count] = torch.sort(tokens[:count])[0]`：对前 `count` 个索引进行升序排序。
- `token_idx_in_rank[i][tokens[:count]] = torch.arange(count, dtype=torch.long, device='cuda')`：利用 python **索引**语法将排序后的前 `count` 个令牌的索引依次赋值给 `token_idx_in_rank` 中对应进程的位置。
- `token_idx_in_rank` 是形状为 `(num_ranks, num_tokens)` 的张量，这里给第 `i` 行（对应 rank `i`）中 “需要处理的 token” 的位置，赋值为连续的本地索引。

1. **调整 `token_idx_in_rank` 张量**

```python

token_idx_in_rank = token_idx_in_rank.T.contiguous().to([torch.int](http://torch.int/))

```

对 `token_idx_in_rank` 进行转置操作，使其形状变为 `(num_tokens, num_ranks)`，然后调用 `contiguous` 方法确保张量在内存中是 “连续存储” 的，最后将数据类型转换为 `torch.int`。

1. **判断每个令牌是否在某个进程中**

```python

is_token_in_rank = token_idx_in_rank >= 0

```

创建一个布尔张量 `is_token_in_rank`，判断 `token_idx_in_rank` 中每个元素是否大于等于 0，即判断每个令牌是否在某个进程中。

1. **全局同步**

```python

gbl_num_tokens_per_rank = num_tokens_per_rank.clone()

dist.all_reduce(gbl_num_tokens_per_rank, group=group)

```

先克隆 `num_tokens_per_rank` 张量得到 `gbl_num_tokens_per_rank`，然后使用 `dist.all_reduce` 函数对 `gbl_num_tokens_per_rank` 进行全局归约操作，将所有进程中每个进程的令牌数量累加起来，得到全局每个进程的令牌数量。

#### 步骤可视化

下面以一个可视化例子解释这个循环的步骤。假设我们有：

- 4 个 token（`num_tokens = 4`）
- 3 个 rank（`num_ranks = 3`）
- `rank_idx` 内容如下：

```python

rank_idx = torch.tensor([

[0, 1], # Token 0选择了rank 0和1上的专家

[1, 2], # Token 1选择了rank 1和2上的专家

[0, 2], # Token 2选择了rank 0和2上的专家

[1, 0] # Token 3选择了rank 1和0上的专家

])

```

##### 1. 统计每个 Rank 负责的 Token 数量

```plaintext

对于rank 0：token 0、2、3 → num_tokens_per_rank[0] = 3

对于rank 1：token 0、1、3 → num_tokens_per_rank[1] = 3

对于rank 2：token 1、2 → num_tokens_per_rank[2] = 2

```

##### 2. 构建 Token 在 Rank 内的索引映射

```python

# 初始token_idx_in_rank（全-1）

token_idx_in_rank = [

[-1, -1, -1, -1], # rank 0

[-1, -1, -1, -1], # rank 1

[-1, -1, -1, -1] # rank 2

]

  

# 处理rank 0

token_sel = [True, False, True, True]

tokens = [0, 2, 3, 1] # 排序后的token索引

count = 3

token_idx_in_rank[0][[0, 2, 3]] = [0, 1, 2] # 设置为0,1,2

  

# 处理rank 1

token_sel = [True, True, False, True]

tokens = [0, 1, 3, 2]

count = 3

token_idx_in_rank[1][[0, 1, 3]] = [0, 1, 2]

  

# 处理rank 2

token_sel = [False, True, True, False]

tokens = [1, 2, 0, 3]

count = 2

token_idx_in_rank[2][[1, 2]] = [0, 1]

  

# 转置后

token_idx_in_rank = [

[0, -1, -1], # token 0

[-1, 0, 1], # token 1

[1, -1, 0], # token 2

[2, 1, -1] # token 3

]

  

# 有效token掩码

is_token_in_rank = [

[True, False, False],

[False, True, True],

[True, False, True],

[True, True, False]

]

```

#### Layout 验证

验证 `buffer.get_dispatch_layout` 方法计算得到的布局信息是否与手动计算的布局信息一致，同时测量该方法的执行性能。

1. 调用 `get_dispatch_layout` 方法获取参考布局信息

```python

ref_num_tokens_per_rank, _, ref_num_tokens_per_expert, ref_is_token_in_rank, _ = \

buffer.get_dispatch_layout(topk_idx, num_experts)

```

输入参数：

- `buffer.get_dispatch_layout(topk_idx, num_experts)`：调用 `deep_ep.Buffer` 实例 `buffer` 的 `get_dispatch_layout` 方法，传入 `topk_idx`（每个令牌对应的 Top-K 专家索引）和 `num_experts`（专家总数）作为参数，该方法会返回一系列布局相关的信息。

intranode 的输出只有三个有效：

- `ref_num_tokens_per_rank`：参考的每个进程的令牌数量。
- `ref_num_tokens_per_expert`：参考的每个专家的令牌数量。
- `ref_is_token_in_rank`：参考的每个令牌是否在某个进程中的布尔张量。
- `_`：表示忽略该位置返回的值。

1. 验证参考布局信息与手动计算的布局信息是否一致

```python

assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)

assert torch.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)

assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)

```

- `torch.allclose`：用于比较两个张量的所有元素是否接近。如果两个张量对应位置的元素差值在一定的容忍范围内，则认为它们接近。
- 这三个 `assert` 语句分别验证参考的每个进程的令牌数量、每个专家的令牌数量以及每个令牌是否在某个进程中的布尔张量是否与手动计算的结果一致。如果不一致，程序会抛出 `AssertionError` 异常，表明 `get_dispatch_layout` 方法的计算结果可能存在问题。

1. 测量 `get_dispatch_layout` 方法的执行性能并打印

```python

t = bench(lambda: buffer.get_dispatch_layout(topk_idx, num_experts))[0]

```

- `bench`：这是一个 DeepEP 自定义的性能测试函数，用于测量传入的函数的执行时间。
- `lambda: buffer.get_dispatch_layout(topk_idx, num_experts)`：定义了一个匿名函数，该函数调用 `buffer.get_dispatch_layout` 方法。
- `t`：获取 `bench` 函数返回结果的第一个元素，即 `get_dispatch_layout` 方法的执行时间。

最后调用进程组 `group` 的 `barrier` 方法，该方法会阻塞当前进程，直到进程组内的所有进程都调用了该方法，确保所有进程在这一步完成同步，然后短暂等待。

### 配置对象设置

```python

# Config

nvl_buffer_size = 256

config = deep_ep.Config(num_sms, 8, nvl_buffer_size)

```

调用 `csrc\config.hpp` 初始化配置对象，设置 SM 数量、NVL 块大小和 NVL 缓冲区大小。

## 三、dispatch 测试

上面生成数据和配置后，下面就调用了 `deep_ep.cpp` 接口的 `buffer.dispatch` 来进行测试分发和合并。下面解读三个不同情形下的合并测试。

### 循环测试

```python

# tests/test_intranode.py

@@ -86, 85

for previous_mode in (False, True):

for async_mode in (False, True):

for current_x in filter(lambda elem: elem is not None, (x_pure_rand, x, x_e4m3)):

for with_topk in (False, True):

# … 分发操作 …

# … 数据检查 …

# … 合并操作 …

# … 数据检查 …

```

这一部分通过四重循环遍历不同的测试模式，包括 _ 是否使用之前的事件、是否异步执行、不同的数据类型以及是否包含 Top-K 信息 _，对 `buffer.dispatch` 和 `buffer.combine` 方法进行测试，并对结果进行检查。

这里使用了四重嵌套循环，每个循环代表一个测试维度，组合起来能覆盖多种测试场景：

- `previous_mode`：布尔值，代表是否使用之前捕获的事件，用于测试 `dispatch` 方法在依赖先前事件时的表现。
- `async_mode`：布尔值，代表是否使用异步模式调用 `dispatch` 方法，测试异步和同步模式下的功能。
- `current_x`：输入数据，可能是纯随机张量 `x_pure_rand`、与进程排名相关的全 1 张量 `x` 或 FP8 格式的张量 `x_e4m3`（若支持），测试不同数据类型和特征下的 `dispatch` 功能。
- `with_topk`：布尔值，代表是否在 `dispatch` 调用中包含 `topk_idx` 和 `topk_weights` 参数，测试有无 Top-K 信息时的功能。

循环开始先打印信息。

语法解释：

1. `filter(lambda elem: elem is not None, (x_pure_rand, x, x_e4m3))`

这是一个**过滤迭代器**，用于筛选出非 `None` 的输入张量。

- **`(x_pure_rand, x, x_e4m3)`**：一个元组，包含三个可能的输入张量：
- `x_pure_rand`：纯随机初始化的张量（bfloat16 格式）；
- `x`：固定值初始化的张量（bfloat16 格式）；
- `x_e4m3`：FP8 格式的张量（可能为 `None`，例如当 GPU 不支持 SM90 架构时）。
- **`lambda elem: elem is not None`**：一个匿名函数（lambda 表达式），作为过滤条件：
- 输入 `elem` 为元组中的每个元素；
- 返回 `True` 如果 `elem` 不是 `None`，否则返回 `False`。
- **`filter(…)`**：Python 内置函数，根据 lambda 的返回值过滤元组：
- 保留所有 `elem is not None` 的元素；
- 返回一个迭代器，遍历所有非 `None` 的张量。

**作用**：只处理有效的输入张量（跳过 `x_e4m3` 为 `None` 的情况），避免后续代码报错。

1. `recv_x, … = buffer.dispatch(** dispatch_args)`

这是**函数调用与参数解包**，调用 `buffer.dispatch` 方法并接收返回值。

- `**dispatch_args`：字典解包语法，将 `dispatch_args` 中的键值对转换为函数的关键字参数，等价于：

```python

buffer.dispatch(x=current_x, num_tokens_per_rank=num_tokens_per_rank, is_token_in_rank=is_token_in_rank)

```

1. `event.current_stream_wait() if async_mode else ()`

这是一个**三元表达式**，等价于：

```python

if async_mode:

event.current_stream_wait()

else:

pass # 空元组 () 表示不执行任何操作

```

1. **`*recv_x`**：元组解包语法，将元组中的元素展开作为参数传递给函数。例如：

```python

# 若 recv_x = (data, scale)

per_token_cast_back(*recv_x) # 等价于 per_token_cast_back(data, scale)

```

### 构建 `dispatch` 方法的参数

```python

dispatch_args = {'x': current_x, 'num_tokens_per_rank': num_tokens_per_rank, 'is_token_in_rank': is_token_in_rank,

'num_tokens_per_expert': num_tokens_per_expert, 'config': config, 'async_finish': async_mode}

if with_topk:

dispatch_args.update({'topk_idx': topk_idx, 'topk_weights': topk_weights_pure_rand if current_x is x_pure_rand else topk_weights})

if previous_mode:

dispatch_args.update({'previous_event': buffer.capture()})

```

- 构建 `dispatch` 方法的基本参数，包含输入数据、每个进程的令牌数量、令牌是否在进程中的信息、每个专家的令牌数量、配置对象以及是否异步执行。
- 若 `with_topk` 为 `True`，添加 `topk_idx` 和 `topk_weights` 参数。
- 若 `previous_mode` 为 `True`，添加之前捕获的事件参数。

### 调用 `dispatch` 方法

```python

recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle, event = buffer.dispatch(**dispatch_args)

event.current_stream_wait() if async_mode else ()

recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x

```

- 调用 `buffer.dispatch` 方法进行数据分发操作，获取返回的接收数据、接收的 Top-K 索引、接收的 Top-K 权重、每个专家接收的令牌数量列表、句柄和事件。
- 若 `async_mode` 为 `True`，等待事件完成。
- 若 `recv_x` 是元组，调用 `per_token_cast_back` 函数将其转换回原始数据类型。

#### 接收检查

这段代码的主要功能是对 `buffer.dispatch` 方法返回的接收数据、Top-K 索引和 Top-K 权重进行了全面的检查，确保分发操作的正确性和数据的一致性。

1. 提取前缀矩阵

```python

rank_prefix_matrix = handle[0]

```

从 `dispatch` 方法返回的 `handle` 元组中提取第一个元素作为 `rank_prefix_matrix`，这个矩阵后续会用于验证数据的一致性。

1. 验证接收数据的数量

```python

assert gbl_num_tokens_per_rank[rank].item() == recv_x.size(0), f'{gbl_num_tokens_per_rank[rank].item()} != {recv_x.size(0)}'

```

- `gbl_num_tokens_per_rank[rank].item()`：获取当前进程的全局令牌数量。
- `recv_x.size(0)`：获取 `dispatch` 方法返回的接收数据 `recv_x` 的第一维大小，即接收的令牌数量。
- `assert` 语句检查这两个值是否相等，如果不相等，会抛出 `AssertionError` 异常，并打印具体的错误信息。

1. 验证每个专家接收的令牌数量

```python

assert gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist() == recv_num_tokens_per_expert_list

```

- `gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist()`：将全局每个专家的令牌数量张量 `gbl_num_tokens_per_expert` 重塑为 `(num_ranks, -1)` 的形状，然后提取当前进程对应的部分并转换为列表。
- `recv_num_tokens_per_expert_list`：`dispatch` 方法返回的当前进程中每个专家接收的令牌数量列表。
- `assert` 语句检查这两个列表是否相等，确保每个专家接收的令牌数量正确。

1. 检查接收数据的一致性

```python

if current_x is not x_pure_rand:

check_data(recv_x, rank_prefix_matrix)

```

- 如果当前输入数据 `current_x` 不是纯随机数据 `x_pure_rand`，则调用 `check_data` 函数对接收数据 `recv_x` 进行检查。
- `check_data` 函数用于验证 `recv_x` 的一致性，确保每个进程的数据符合预期。

1. 初始化克隆的 Top-K 权重

```python

recv_topk_weights_clone = None

```

初始化 `recv_topk_weights_clone` 为 `None`，后续在需要时会对 `recv_topk_weights` 进行克隆。

1. 检查 Top-K 索引

```python

if with_topk:

# Check `topk_idx`

assert (recv_topk_idx.eq(-1) | ((recv_topk_idx >= 0) & (recv_topk_idx < (num_experts // num_ranks)))).sum().item() == recv_topk_idx.numel()

for i, count in enumerate(recv_num_tokens_per_expert_list):

assert recv_topk_idx.eq(i).sum().item() == count

```

- **第一个 `assert` 语句**：
- `recv_topk_idx.eq(-1)`：检查 `recv_topk_idx` 中值为 -1 的元素。
- `(recv_topk_idx >= 0) & (recv_topk_idx < (num_experts // num_ranks))`：检查 `recv_topk_idx` 中值在有效范围内（大于等于 0 且小于每个进程的专家数量）的元素。
- `(recv_topk_idx.eq(-1) | ((recv_topk_idx >= 0) & (recv_topk_idx < (num_experts // num_ranks))))`：对上述两个条件进行逻辑或运算，得到所有有效元素。
- `(recv_topk_idx.eq(-1) | ((recv_topk_idx >= 0) & (recv_topk_idx < (num_experts // num_ranks)))).sum().item()`：计算有效元素的数量。
- `recv_topk_idx.numel()`：计算 `recv_topk_idx` 中元素的总数。
- 该 `assert` 语句确保 `recv_topk_idx` 中的所有元素都是有效元素。
- **第二个 `assert` 语句**：
- 遍历 `recv_num_tokens_per_expert_list`，对于每个专家 `i`，检查 `recv_topk_idx` 中值等于 `i` 的元素数量是否等于该专家接收的令牌数量 `count`。

1. 检查 Top-K 权重

```python

# Check `topk_weights`

recv_topk_weights_clone = recv_topk_weights.clone()

if current_x is not x_pure_rand:

recv_topk_weights[recv_topk_idx.eq(-1)] = recv_topk_weights.amax(dim=1, keepdim=True).expand_as(recv_topk_weights)[recv_topk_idx.eq(-1)]

check_data(recv_topk_weights, rank_prefix_matrix)

```

- **克隆 Top-K 权重**：对 `recv_topk_weights` 进行克隆，保存原始数据。
- **处理无效索引对应的权重**：
- 如果当前输入数据 `current_x` 不是纯随机数据 `x_pure_rand`，则将 `recv_topk_idx` 中值为 -1 的元素对应的 `recv_topk_weights` 替换为该行的最大值。
- `recv_topk_weights.amax(dim=1, keepdim=True)`：计算 `recv_topk_weights` 每行的最大值。
- `expand_as(recv_topk_weights)`：将最大值扩展为与 `recv_topk_weights` 相同的形状。
- `recv_topk_weights[recv_topk_idx.eq(-1)] = …`：将无效索引对应的权重替换为最大值。
- **检查处理后的 Top-K 权重**：调用 `check_data` 函数对处理后的 `recv_topk_weights` 进行检查，确保其一致性。

### Dispatch 不同情景检查

这两段代码分别对不同场景下 `buffer.dispatch` 方法的功能进行测试。

- `# Test num_worst_tokens != 0` 部分在包含 Top-K 信息的场景下，测试 `dispatch` 方法处理 `num_worst_tokens` 参数的功能。
- `# Test cached dispatch (must without top-k staffs)` 部分在不包含 Top-K 信息的场景下，测试 `dispatch` 方法的缓存分发功能。

#### `# Test num_worst_tokens != 0` 部分

```python

# tests/test_intranode.py

@@ -121, 14

# Test `num_worst_tokens != 0

```

此部分代码在 `with_topk` 为 `True` 的情况下，测试 `dispatch` 方法在传入 `num_worst_tokens` 参数时的行为，以此验证 `dispatch` 方法处理最差令牌数量的能力。

1. **设置 `num_worst_tokens` 参数**：

```python

num_worst_tokens = num_tokens * num_ranks

dispatch_args.update({'num_worst_tokens': num_worst_tokens})

```

计算 `num_worst_tokens` 的值并将其添加到 `dispatch_args` 字典中，后续会将其作为参数传递给 `dispatch` 方法。

1. **调用 `dispatch` 方法**：

```python

recv_worst_x, recv_worst_topk_idx, recv_worst_topk_weights, empty_list, _, event = buffer.dispatch(**dispatch_args)

```

调用 `buffer.dispatch` 方法，传入更新后的参数，获取接收数据、Top-K 索引、Top-K 权重等返回值。

1. **处理异步操作**：

```python

event.current_stream_wait() if async_mode else ()

```

若处于异步模式，等待事件完成，确保数据处理完成。

1. **数据类型转换**：

```python

recv_worst_x = per_token_cast_back(*recv_worst_x) if isinstance(recv_worst_x, tuple) else recv_worst_x

```

若 `recv_worst_x` 是元组，调用 `per_token_cast_back` 函数将其转换回原始数据类型。

1. **结果验证**：

```python

assert len(empty_list) == 0

assert num_worst_tokens == recv_worst_x.size(0)

assert num_worst_tokens == recv_worst_topk_idx.size(0)

assert num_worst_tokens == recv_worst_topk_weights.size(0)

assert torch.equal(recv_x, recv_worst_x[:recv_x.size(0)])

assert torch.equal(recv_topk_idx, recv_worst_topk_idx[:recv_x.size(0)])

assert torch.equal(recv_topk_weights_clone, recv_worst_topk_weights[:recv_x.size(0)])

assert torch.all(recv_worst_topk_idx[recv_x.size(0):] == -1).item()

```

通过一系列 `assert` 语句验证返回结果的正确性，包括 `empty_list` 是否为空、接收数据的大小是否符合预期，以及前 `recv_x.size(0)` 个元素是否与之前的结果一致等。

#### `# Test cached dispatch (must without top-k staffs)` 部分

```python

# tests/test_intranode.py

@@ -137, 9

# Test cached dispatch (must without top-k staffs)

```

这部分代码在 `with_topk` 为 `False` 的情况下，测试 `dispatch` 方法的缓存分发功能，即不使用 Top-K 相关参数时的行为。

1. **构建参数**：

```python

dispatch_args = {'x': current_x, 'handle': handle, 'config': config, 'async_finish': async_mode}

if previous_mode:

dispatch_args.update({'previous_event': buffer.capture()})

```

构建调用 `dispatch` 方法所需的参数，若 `previous_mode` 为 `True`，添加之前捕获的事件参数。

1. **调用 `dispatch` 方法**：

```python

recv_x, _, _, _, _, event = buffer.dispatch(**dispatch_args)

```

调用 `buffer.dispatch` 方法，传入构建好的参数，获取接收数据和事件。

1. **处理异步操作**：

```python

event.current_stream_wait() if async_mode else ()

```

若处于异步模式，等待事件完成，确保数据处理完成。

## 四、combine 测试

```python

# tests/test_intranode.py

@@ -148, 14

# Test combine

```

在同样的循环里面，这段代码通过构建不同参数组合调用 `combine` 方法，该方法用于将之前 `dispatch` 分发出去的数据合并回来，对合并后的数据和 Top-K 权重进行处理，并与参考数据进行比较，验证 `combine` 方法的正确性。通过设置不同的测试条件（如是否包含 Top-K 信息、是否启用 `previous_mode` 等），确保合并操作的正确性。

### 1. 构建 `combine` 方法的参数

```python

combine_args = {'x': recv_x, 'handle': handle, 'config': config, 'async_finish': async_mode}

if with_topk:

combine_args.update({'topk_weights': recv_topk_weights})

if previous_mode:

combine_args.update({'previous_event': buffer.capture()})

```

- `combine_args`：构建一个字典，包含 `combine` 方法的基本参数。
- `x`：需要合并的数据，即之前 `dispatch` 方法返回的 `recv_x`。
- `handle`：`dispatch` 方法返回的句柄，包含分发操作的布局信息。
- `config`：配置对象，用于指定合并操作的配置参数。
- `async_finish`：布尔值，指定是否以异步模式执行合并操作。
- `if with_topk`：如果在测试中包含 Top-K 信息，将 `recv_topk_weights`（`dispatch` 方法返回的 Top-K 权重）添加到参数中。
- `if previous_mode`：如果启用了 `previous_mode`，调用 `buffer.capture()` 捕获当前事件，并将其添加到参数中，用于事件同步。

### 2. 调用 `combine` 方法

```python

combined_x, combined_topk_weights, event = buffer.combine(**combine_args)

event.current_stream_wait() if async_mode else ()

```

- `buffer.combine(**combine_args)`：调用 `combine` 方法进行数据合并操作，返回合并后的数据 `combined_x`、合并后的 Top-K 权重 `combined_topk_weights` 以及一个事件对象 `event`。
- `event.current_stream_wait() if async_mode else ()`：如果以异步模式执行合并操作，等待事件完成，确保数据合并操作已经结束。

### 3. 验证合并后的数据

```python

check_x = combined_x.float() / is_token_in_rank.sum(dim=1).unsqueeze(1)

ref_x = x_pure_rand if current_x is x_pure_rand else x

assert calc_diff(check_x, ref_x) < 5e-6

```

- `check_x`：对合并后的数据 `combined_x` 进行处理，将其转换为 `float` 类型，并除以每个令牌所属进程的数量（`is_token_in_rank.sum(dim=1).unsqueeze(1)`），得到用于验证的数据。
- `ref_x`：根据当前输入数据 `current_x` 的类型，选择参考数据。如果 `current_x` 是纯随机数据 `x_pure_rand`，则参考数据为 `x_pure_rand`；否则，参考数据为 `x`。
- `assert calc_diff(check_x, ref_x) < 5e-6`：调用 `calc_diff` 函数计算 `check_x` 和 `ref_x` 之间的差异，并使用 `assert` 语句确保差异小于 `5e-6`。如果差异超过该阈值，测试将失败。

### 4. 验证合并后的 Top-K 权重（如果包含 Top-K 信息）

```python

if with_topk:

check_topk_weights = combined_topk_weights if (current_x is x_pure_rand) else (combined_topk_weights / is_token_in_rank.sum(dim=1).unsqueeze(1))

ref_topk_weights = topk_weights_pure_rand if current_x is x_pure_rand else topk_weights

assert calc_diff(check_topk_weights, ref_topk_weights) < 1e-9

```

- `check_topk_weights`：根据当前输入数据 `current_x` 的类型，对合并后的 Top-K 权重 `combined_topk_weights` 进行处理。如果 `current_x` 是纯随机数据 `x_pure_rand`，则直接使用 `combined_topk_weights`；否则，将其除以每个令牌所属进程的数量。
- `ref_topk_weights`：根据当前输入数据 `current_x` 的类型，选择参考的 Top-K 权重。如果 `current_x` 是纯随机数据 `x_pure_rand`，则参考权重为 `topk_weights_pure_rand`；否则，参考权重为 `topk_weights`。
- `assert calc_diff(check_topk_weights, ref_topk_weights) < 1e-9`：调用 `calc_diff` 函数计算 `check_topk_weights` 和 `ref_topk_weights` 之间的差异，并使用 `assert` 语句确保差异小于 `1e-9`。如果差异超过该阈值，测试将失败。

## 五、Tune 处理

```python

# For later tuning

dispatch_bf16_nvl_recv_bytes = recv_x.numel() * 2

combine_bf16_nvl_send_bytes = dispatch_bf16_nvl_recv_bytes

```

在分布式训练里，了解数据传输量对于性能调优至关重要。此代码计算分发和合并操作期间通过 NVLink（NVIDIA 高速互联技术）传输的数据字节数，为后续性能优化提供依据。

- `recv_x.numel()`：该方法返回 `recv_x` 张量中的元素总数。`recv_x` 是分发操作后接收到的数据。
- `* 2`：由于 `recv_x` 的数据类型是 `torch.bfloat16`，每个 `bfloat16` 数据元素在内存中占用 2 字节，将元素总数乘以 2 就得到了分发操作期间通过 NVLink 接收的总字节数。
- `dispatch_bf16_nvl_recv_bytes`：这个变量存储计算得到的接收字节数。
- `combine_bf16_nvl_send_bytes`：该变量表示合并操作期间通过 NVLink 发送的字节数。
- 在理想情况下，合并操作发送的数据量应与分发操作接收的数据量相等。所以，这行代码直接将 `dispatch_bf16_nvl_recv_bytes` 的值赋给 `combine_bf16_nvl_send_bytes`。

这两个变量会在后续的性能调优代码里用于计算分发和合并操作的数据传输速率（GB/s），帮助开发者找到最优的配置参数。例如：

```python

# …

t = bench(lambda: buffer.dispatch(**tune_args))[0]

if local_rank == 0:

print(f'[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size if nvl_chunk_size else "default"}: '

f'{dispatch_bf16_nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL), avg_t: {t * 1e6:.2f} us', flush=True)

# …

```

## 六、分发和合并性能调优

```python

# Tune dispatch performance

best_dispatch_results = None

fp8_factor = (1 + 4 / 128) / 2

for current_x in filter(lambda elem: elem is not None, (x_e4m3, x)):

best_time, best_results = 1e10, None

for nvl_chunk_size in tuple(range(4, 33, 2)) + (0, ):

# … 测试不同配置下的分发性能 …

# … 记录最佳配置 …

  

# Tune combine performance

best_time, best_results = 1e10, None

for nvl_chunk_size in tuple(range(1, 17, 1)) + (0, ):

# … 测试不同配置下的合并性能 …

# … 记录最佳配置 …

```

分别对分发和合并操作进行性能调优，尝试不同的 NVL 块大小，记录最佳性能配置并打印结果。

---
