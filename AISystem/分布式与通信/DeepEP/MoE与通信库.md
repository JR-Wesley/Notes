---
dateCreated: 2025-08-08
dateModified: 2025-08-08
---


# MoE 与通信库的协同：以 DeepEP 为例

MoE 的稀疏激活依赖高效的跨设备通信（专家分布在不同 GPU 上，Token 需分发到目标专家，结果需回传合并），因此需要专门的通信库支持。例如 DeepEP 作为适配 MoE 的通信库，通过优化 All-to-All 通信、动态路由适配等能力，解决了 MoE 的通信瓶颈，使稀疏激活的效率优势得以落地。

## hidden_size（隐藏层维度）

- **定义**：

`hidden_size` 表示**每个 token 的特征向量维度**，即模型隐藏层的维度。例如：

- GPT-3 的 `hidden_size=12288`，意味着每个 token 用 12288 维向量表示。
- **作用**：

决定了模型的表示能力，维度越高，模型能捕获的信息越丰富，但计算量也越大。

## Token 的定义、输入顺序与分发布局

1. Token 的本质与输入特性

- **Token**：
- 是输入文本序列经过分词器（Tokenizer）处理后的编码单元
- 例如："Hello world!" → ["Hello", "▁world", "!"], 对应 ID 可能为 [101, 2026, 999]
- **输入顺序**：
- 通常按序列顺序输入（如文本中的先后顺序）
- 在推理时，可按生成顺序逐个处理（自回归生成）

1. num_tokens（令牌数量）

- **定义**：

`num_tokens` 表示**一批输入数据中的 “令牌” 总数**。在 NLP 中，一个 token 通常是一个词元（如通过 BPE 分词后的子词）；在视觉任务中，可能是一个图像块（patch）。

- **数据格式**：

一个 token 本身不是单一的 `bf16` 值，而是一个**向量**。例如：

- 若 `hidden_size=1024` 且使用 `bf16`（2 字节），则一个 token 占用 `1024×2=2048` 字节。
- `num_tokens` 是这批输入中所有 token 的数量，例如一个批次有 64 个序列，每个序列平均长度为 128，则 `num_tokens=64×128=8192`。

## 存储

- **本地 Buffer**：

输入数据（tokens）通常存储在 GPU 显存的 `sendBuf` 中，格式为 `[num_tokens, hidden_size]`。例如：

```plaintext

sendBuf shape: [8192, 1024] # 8192个token，每个1024维

```

- **分块存储**：

为优化内存访问，数据按 `HGran`/`WGran` 分块（如之前分析的 `HGran=4`，`WGran=32`），匹配 GPU SM 的处理能力。

## Token 到专家的分发机制

Token 会根据路由结果发送到多个专家：

- **Top-K 路由**：每个 Token 选择得分最高的 K 个专家

```python

# 伪代码示例

expert_scores = router_network(input_token) # 路由网络输出专家得分

topk_experts = torch.topk(expert_scores, k=2).indices # 选择Top-2专家

```

- **分发布局确定流程**：

1. **路由计算**（通常在发起 GPU 上）：

```python

# 路由器输出示例 (batch_size=4, num_experts=1024, k=2)

topk_experts = [[128, 256], [256, 512], [0, 128], [768, 128]]

```

1. **布局计算**（CPU/GPU 协同）：

```python

# 计算每个GPU需要接收的Token数量

tokens_per_gpu = [0] * num_gpus

for token_experts in topk_experts:

for expert_id in token_experts:

target_gpu = expert_id % num_gpus

tokens_per_gpu[target_gpu] += 1

```

1. **生成映射表**：

```python

# 生成Token到目标GPU的映射

dispatch_map = {gpu_id: [] for gpu_id in range(num_gpus)}

for token_idx, token_experts in enumerate(topk_experts):

for expert_id in token_experts:

target_gpu = expert_id % num_gpus

dispatch_map[target_gpu].append(token_idx)

```

# Buffer

## 通信 Buffer 的作用与设计

通信 Buffer 是 DeepEP 中实现高效 MoE 通信的关键组件，其主要功能包括：

1. **数据暂存**：作为数据发送 / 接收的临时存储区域，避免频繁访问原始数据
2. **通信协调**：管理不同 GPU 之间的数据流动，优化传输顺序和批量大小
3. **重叠计算**：支持通信与计算的异步执行，提高硬件利用率

流程链路是：`buffer初始化` → `计算token分布` → `通过buffer dispatch（分发token到目标rank）` → `专家处理` → `通过buffer combine（合并各专家结果）`

### 每个 GPU Rank 都有独立的 Buffer 实例

```python

# 每个进程（对应一个 GPU）创建自己的 Buffer

buffer = Buffer(

rank=local_rank, # 当前 GPU 的 ID

group_size=world_size, # 总 GPU 数量

num_nvl_bytes=1024*1024, # NVLink 缓冲区大小（1MB）

num_rdma_bytes=4096*1024 # RDMA 缓冲区大小（4MB）

)

```

- **Buffer 结构**：

```plaintext

Buffer for GPU 0:

├── NVLink Buffer (1MB) # 用于节点内通信

├── RDMA Buffer (4MB) # 用于节点间通信

├── Layout Metadata # 存储分发布局信息

└── Event Handles # 用于同步 CUDA 流

```

### 存储内容与数据流程

Buffer 主要存储三类数据：

1. **待发送的 Tokens**：

```python

# 在 dispatch 阶段

buffer.nvlink_buffer[target_gpu_offset:target_gpu_offset+size] = tokens_to_send

```

1. **已接收的 Tokens**：

```python

# 在 receive 阶段

received_tokens = buffer.rdma_buffer[my_offset:my_offset+received_size]

```

1. **布局元数据**：

```python

# 例如，每个 GPU 需要接收的 Token 数量

buffer.layout.num_tokens_per_gpu = [128, 256, 192, …]

```

## Buffer 与 CUDA Stream 的关系

### CUDA Stream 的作用

CUDA Stream 是 GPU 上的执行队列，允许异步执行多个操作：

- 不同 Stream 中的操作可以并行执行
- 同一 Stream 中的操作按顺序执行

### Buffer 与 Stream 的协作模式

1. **计算流（Compute Stream）**：

```python

compute_stream = [torch.cuda.Stream](http://torch.cuda.stream/)() # 创建计算流

  

# 在计算流中执行专家计算

with [torch.cuda.stream](http://torch.cuda.stream/)(compute_stream):

expert_output = expert_model(input_tokens)

```

1. **通信流（Communication Stream）**：

```python

comm_stream = buffer.get_comm_stream() # 获取 Buffer 的通信流

  

# 在通信流中执行数据发送

with [torch.cuda.stream](http://torch.cuda.stream/)(comm_stream):

buffer.dispatch(tokens, expert_indices)

```

1. **同步机制**：

```python

# 使用事件同步计算和通信

compute_done = torch.cuda.Event(enable_timing=True)

compute_stream.record_event(compute_done)

  

# 通信流等待计算完成

comm_stream.wait_event(compute_done)

  

# 在通信完成后继续计算

comm_done = torch.cuda.Event(enable_timing=True)

comm_stream.record_event(comm_done)

compute_stream.wait_event(comm_done)

```

## 关键优化点

### 1. 通信与计算重叠

```python

# 伪代码：异步执行通信与计算

with [torch.cuda.stream](http://torch.cuda.stream/)(comm_stream):

# 1. 将数据从计算缓冲区复制到通信 Buffer

buffer.prepare_for_dispatch(input_tokens)

# 2. 启动通信（NVLink/RDMA）

buffer.dispatch_async()

  

with [torch.cuda.stream](http://torch.cuda.stream/)(compute_stream):

# 3. 执行不需要通信结果的本地计算

partial_result = local_computation()

# 4. 等待通信完成

torch.cuda.current_stream().wait_event(buffer.get_dispatch_done_event())

# 5. 使用通信结果继续计算

final_result = combine(partial_result, received_data)

```

### 2. Buffer 内存优化

- **动态调整大小**：根据实际 Token 分布情况调整 Buffer 大小

```python

# 根据历史数据估计所需 Buffer 大小

estimated_size = estimate_buffer_size(batch_size, num_experts, k=2)

buffer.resize(estimated_size)

```

- **内存复用**：在不同批次间复用 Buffer 内存

```python

# 每批处理后重置 Buffer 指针，避免频繁内存分配

buffer.reset()

```

## 总结

通信 Buffer 在 DeepEP 中的核心角色：

1. **数据中转站**：作为 GPU 间数据传输的临时存储，减少原始数据访问
2. **异步执行协调者**：通过与 CUDA Stream 协作，实现通信与计算的重叠
3. **内存优化器**：动态管理内存使用，避免不必要的内存分配

---

# 更多优化

专家并行与剪枝的核心共性：动态筛选有效参数。未参与计算的参数占据显存，但通常显存用不满

- **未激活专家的参数仍占用显存**：专家并行中，所有专家参数需分布在不同 GPU（如 6 个专家分布在 3 个 GPU，每个 GPU 存 2 个专家），无论是否被当前 Token 激活，专家参数都需常驻显存（以便后续 Token 可能激活）—— 这部分未参与计算的参数确实会占用显存。
- **显存通常用不满的现实基础**：大模型训练时，GPU 显存规划会预留冗余（避免 OOM），且专家参数的分布是 “按 GPU 拆分” 的（而非集中在单卡）。例如，8 卡 GPU 训练时，每个 GPU 仅存储部分专家参数，剩余显存可用于激活值、梯度等临时数据；同时，MoE 的稀疏激活（仅用部分专家）降低了计算时的显存峰值，因此即使有未激活的专家参数，整体显存通常不会被占满。
