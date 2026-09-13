结合 [[2-组件概述和加速方法]] 中讲解了总体调用 Python API 的流程，下面分析各个 API 具体做了什么，是如何调用 C++ 接口的。

# Buffer Python 构造

## 类的整体结构：字段 + Runtime + 方法

先看类头部：

- `num_sms = 20` ：默认 SM 数
- `__init__(...)` ：创建 runtime
- `destroy()`
- `get_low_latency_rdma_size_hint(...)`
- `get_comm_stream()`
- `get_local_buffer_tensor(...)`
- `clean_low_latency_buffer(...)`
- `low_latency_dispatch(...)`
- `low_latency_combine(...)`
- `get_next_low_latency_combine_buffer(...)`

这个结构非常清晰：它是一个“通信 buffer 管理器”。

真正重要的是 `self.runtime`，它是底层 C++ 对象。Python 层几乎所有高层 API 最终都会走到它。

---

## 4. `destroy()`：清理 Runtime

```python
def destroy(self):
    assert self.explicitly_destroy, '`explicitly_destroy` flag must be set'
    self.runtime.destroy()
    self.runtime = None
```

这里很直观：

- 如果你创建 buffer 时设置了 `explicitly_destroy=True`
- 就必须手动释放
- 否则资源会泄漏或在 Python GC 时造成异常/卡住

这也是 Python 层做“生命周期管理”的典型方式。

---

## 5. `get_low_latency_rdma_size_hint`: 计算 Buffer 大小

这个函数很重要，因为测试里会先调用：

```python
num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, num_ranks, num_experts)
```

它的实现很简单：

```python
return _C.get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts)
```

含义是：

- 低延迟路径要求一个最小的 RDMA buffer 空间
- 这个 buffer 满足“每个 rank 最多 dispatch 的 token 数 + hidden 维度 + expert 数”这些需求
- 它不是任意给值，而是由底层 C++ 计算出来的“最小合法尺寸”

所以这一步本质上是“合法化资源分配”。

---

## 6. `low_latency_dispatch`：把 Token 按 Expert 分发

这个是测试中最重要的调用之一，位于 legacy.py 里。

### 6.1 参数检查

```python
check_torch_deterministic()
assert self.nvshmem_qp_depth >= (num_max_dispatch_tokens_per_rank + 1) * 2
```

这里在强调：

- 低延迟 kernel 需要确定性执行
- 需要 QP 深度足够大，否则通信并发不安全

---

### 6.2 真正调用底层 Runtime

```python
packed_recv_x, packed_recv_x_scales, packed_recv_count, packed_recv_src_info, packed_recv_layout_range, event, hook = \
    self.runtime.low_latency_dispatch(
        x, topk_idx,
        cumulative_local_expert_recv_stats,
        dispatch_wait_recv_cost_stats,
        num_max_dispatch_tokens_per_rank, num_experts,
        use_fp8, round_scale, use_ue8m0,
        async_finish, return_recv_hook)
```

这个返回值非常关键：

- `packed_recv_x`：收到的 token 数据
- `packed_recv_x_scales`：FP8 scaling 因子（如果开启 FP8）
- `packed_recv_count`：每个 local expert 收到的 token 个数
- `packed_recv_src_info`：每个 token 的来源 rank / 偏移信息
- `packed_recv_layout_range`：每个 sender rank 的布局区间信息
- `event`：异步通信完成事件
- `hook`：接收钩子，用于异步等待

---

### 6.3 组装 `handle`

```python
handle = (packed_recv_src_info, packed_recv_layout_range, num_max_dispatch_tokens_per_rank, x.size(1), num_experts)
```

这里非常重要：这个 `handle` 是后续 `low_latency_combine` 必须携带的“通信上下文”。

它包含的不是 token 数据，而是“combine 时需要知道怎么把这些数据还原/聚合”的信息：

- `src_info`：来自哪个 rank
- `layout_range`：每个 rank 对应的 token 区间
- `num_max_dispatch_tokens_per_rank`：最大每 rank token 数
- `hidden`：token 长度
- `num_experts`：total expert count

也就是说，dispatch 不是单纯“send”，它会把通信的资源布局和元信息一并交给 combine。

---

### 6.4 为什么要返回 `EventOverlap`

这一段很精巧：

```python
return (packed_recv_x, packed_recv_x_scales) if use_fp8 else packed_recv_x, packed_recv_count, handle, \
    EventOverlap(event, tensors_to_record if async_finish else None), hook
```

这里 `EventOverlap` 是非常典型的 Python 层“异步事件包装”。

含义是：

- 如果 `async_finish=True`，则把发送/接收涉及的 tensor 记录下来
- 之后 `event.current_stream_wait()` 或 hook 触发时，确保这些 tensor 的生命周期/同步安全

这是为了支持 overlap 和 async kernel pipeline。

---

## 7. `low_latency_combine`: Reduce + 归并返回

combine 的函数体在 legacy.py 里。

### 7.1 先核对 Handle

```python
src_info, layout_range, num_max_dispatch_tokens_per_rank, hidden, num_experts = handle
assert self.nvshmem_qp_depth >= (num_max_dispatch_tokens_per_rank + 1) * 2
```

这里说明 combine 依赖的是 dispatch 返回的 handle，而不是重新计算。

---

### 7.2 真正调用底层 Kernel

```python
combined_x, event, hook = self.runtime.low_latency_combine(
    x, topk_idx, topk_weights,
    src_info, layout_range,
    combine_wait_recv_cost_stats,
    num_max_dispatch_tokens_per_rank,
    num_experts,
    use_logfmt, zero_copy,
    async_finish, return_recv_hook, out)
```

这里的 input：

- `x`：本地 expert 的计算结果
- `topk_idx`：每个 token 选中的 expert
- `topk_weights`：对应权重
- `src_info` / `layout_range`：来自 dispatch 的通信布局
- `zero_copy`：是否直接使用下一轮 combine buffer，不再额外 copy
- `out`：可选输出 buffer，直接写入

本质上，这一步是：

- 取出每个 rank 投递来的 token
- 根据 `topk_idx` 定位目标 expert
- 用 `topk_weights` 对这些 token 做 reduce
- 输出到 `combined_x`

---

### 7.3 为什么 `zero_copy` 很关键

代码中有：

```python
if zero_copy:
    buffer.get_next_low_latency_combine_buffer(handle)[:, :, :] = simulated_gemm_x
```

意思是：

- 直接把本地计算结果放到下一轮 combine 的 RDMA buffer
- 后面 combine kernel 直接读这块待发送区域
- 跳过一次 `copy`

这是低延迟路径非常典型的优化：减少一轮显存拷贝，减少 latency。

---

## 8. `get_next_low_latency_combine_buffer`: 直接拿 Raw Buffer

```python
src_info, layout_range, num_max_dispatch_tokens_per_rank, hidden, num_experts = handle
return self.runtime.get_next_low_latency_combine_buffer(num_max_dispatch_tokens_per_rank, hidden, num_experts)
```

这个函数返回的是：

- 一个底层 RDMA buffer 的 view
- 本质上是一个 BF16 tensor
- 形状通常是 `[num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden]`

这个 buffer 是“下一次 combine 可直接写入和直接读取”的区域。
因此 zero-copy combine 的本质，不是“算完再拷贝”，而是“直接把结果写到 RDMA 共享区，后续通信核直接用”。

---

## 9. 整体理解：这个类到底在做什么

你可以把 `Buffer` 理解为下面这个流程：

1. 初始化
   - 所有 rank 知道彼此的 GPU / IPC / NVSHMEM 标识
   - 底层 runtime ready

2. dispatch
   - 先做 token 分发，把每个 token 发给对应 expert
   - 记录 src 和 layout metadata

3. combine
   - 按 `handle` 还原出别的 rank 发来的 token 分布
   - 用 `topk_idx` + `topk_weights` 做 reduce
   - 返回最终输出

4. 低延迟优化
   - 直接使用 RDMA buffer
   - zero-copy
   - async_finish / hook
   - 预先申请 QP / 共享状态

---

## 10. 这套设计的本质

这个 `Buffer` 类最核心的设计思想是：

- Python 层不承担真正的高性能 CUDA kernel
- 它负责“编排 + 同步 + 语义封装”
- C++ runtime 承担“真正的 kernel 调度、RDMA、shared memory、NVSHMEM”
- `handle` 是把“通信结构信息”作为状态对象传递，保证 combine 能知道 dispatch 时的布局

也就是：

- dispatch 负责“分发并记录拓扑”
- combine 负责“按拓扑反向归并”

这是 low-latency MoE EP 的关键。

---

如果你愿意，我下一步可以继续展开成“按调用栈继续往 C++ 扩展 `_C.Buffer` 走一层”，讲解：

- `get_local_nvshmem_unique_id`
- `low_latency_dispatch` 在 C++ 里如何组织 `src_info/layout_range`
- `combine` 是怎么做 gate / mask / reduce 的

这样会更接近底层实现。

# Buffer Init

## `__init__` 的作用

`__init__` 在 legacy.py 里，是整个 `Buffer` 对象的构造函数。它的职责不是“简单赋值”，而是：

1. 确定当前进程在 EP group 里的身份
2. 初始化底层 C++ runtime
3. 让所有 rank 同步各自 GPU / IPC / NVSHMEM 元信息
4. 开启 low-latency 模式下所需的 IBGDA + NVSHMEM 配置
5. 让 runtime 进入可用状态，后续才能调用 dispatch/combine

一句话：它是“把 Python 层的 buffer 连接到底层 GPU 通信 runtime，并准备好全局通信状态”的入口。

---

## 逐行解析

### 1. 参数定义

```python
def __init__(self,
             group: Optional[dist.ProcessGroup],
             num_nvl_bytes: int = 0,
             num_rdma_bytes: int = 0,
             low_latency_mode: bool = False,
             num_qps_per_rank: int = 24,
             allow_nvlink_for_low_latency_mode: bool = True,
             allow_mnnvl: bool = False,
             explicitly_destroy: bool = False,
             enable_shrink: bool = False,
             comm: Optional["mpi4py.MPI.Comm"] = None) -> None:
```

这个参数列表已经基本说明了初始化目标：

- `group`：
  - torch distributed 的 process group
  - 这是通信组的核心标识
- `num_nvl_bytes`：
  - NVLink 通信 buffer 大小
- `num_rdma_bytes`：
  - RDMA 通信 buffer 大小
- `low_latency_mode`：
  - 是否开启低延迟路径
- `num_qps_per_rank`：
  - 每 rank 开多少个 QP（Queue Pair）
- `allow_nvlink_for_low_latency_mode`：
  - 是否允许 low-latency 模式走 NVLink
- `allow_mnnvl`：
  - 是否允许 MNNVL（multi-node NVLink）
- `explicitly_destroy`：
  - 是否手动释放资源
- `enable_shrink`：
  - 是否开启 shrink 模式，支持 mask 某些 rank
- `comm`：
  - 如果不传 `group`，也可以传 MPI comm

这不是普通对象初始化，而是“通信运行时初始化”。

---

### 2. 检查 NVLink 连接

```python
check_nvlink_connections(group)
```

这是构造前的硬性检查。

它会检查当前通信组下的 GPU 连接拓扑是否满足 DeepEP 对 NVLink/RDMA 的假设。
如果当前环境不支持本项目的通信前提，构造直接 fail，避免后面通信出现隐性错误。

这等价于“启动前先确认基础硬件拓扑正确”。

---

### 3. 取当前 Rank 和 Group Size

```python
if group is not None:
    self.rank = group.rank()
    self.group = group
    self.group_size = group.size()

    def all_gather_object(obj):
        object_list = [None] * self.group_size
        dist.all_gather_object(object_list, obj, group)
        return object_list
```

这段是最核心的分布式基础信息获取。

#### 作用

- `self.rank`：当前 rank
- `self.group`：当前通信 group
- `self.group_size`：总 rank 数

#### `all_gather_object`

这个函数非常关键：

```python
def all_gather_object(obj):
    object_list = [None] * self.group_size
    dist.all_gather_object(object_list, obj, group)
    return object_list
```

它的行为是：

- 当前 rank 把一个对象 `obj` 发给所有其他 rank
- 每个 rank 都得到整个 group 的同一份列表
- 例如 `obj` 是 device id / IPC handle / unique id

所以它用于“全局同步状态”。

这是分布式初始化里最常见的模式：每个 rank 先 local 生成信息，再 gather 到全组。

---

### 4. 如果没有 group，则支持 MPI Comm

```python
elif comm is not None:
    self.rank = comm.Get_rank()
    self.group = comm
    self.group_size = comm.Get_size()

    def all_gather_object(obj):
        return comm.allgather(obj)
```

这里是另一种初始化路径：

- 如果不是 torch distributed 的 `ProcessGroup`
- 而是 MPI comm
- 也支持初始化

这个分支的逻辑和上面基本等价，只是底层 allgather API 不同。

---

### 5. 否则报错

```python
else:
    raise ValueError("Either 'group' or 'comm' must be provided.")
```

意思很直接：

- 不能同时没有通信上下文
- 必须传 `group` 或 `comm`

否则无法建立跨 rank 的共享通信状态。

---

### 6. 保存 Buffer 参数

```python
self.num_nvl_bytes = num_nvl_bytes
self.num_rdma_bytes = num_rdma_bytes
self.low_latency_mode = low_latency_mode
self.explicitly_destroy = explicitly_destroy
self.enable_shrink = enable_shrink
```

这些字段只是把初始化参数记到对象上，方便后续方法访问。

例如：

- `self.num_rdma_bytes` 后面会被拿来计算 buffer 大小
- `self.low_latency_mode` 会决定是否启用 IBGDA / low-latency API
- `self.enable_shrink` 影响 mask buffer 机制

---

### 7. 创建底层 C++ Runtime

```python
self.runtime = _C.Buffer(self.rank, self.group_size, num_nvl_bytes, num_rdma_bytes, low_latency_mode,
                         explicitly_destroy, enable_shrink, allow_mnnvl)
```

这是整个 `__init__` 的真正核心。

#### 它做了什么

这里调用的是底层扩展模块 `deep_ep._C.Buffer`，即 C++/CUDA runtime。

传入参数包括：

- 当前 rank
- group size
- NVLink buffer bytes
- RDMA buffer bytes
- 是否 low-latency
- 是否 explicit destroy
- 是否 shrink
- 是否 allow MNNVL

#### 意义

Python 层的 `Buffer` 只是“封装层”，真正的通信状态、内存分配、QV/IBGDA 连接都在 C++ runtime 中。

所以这一步是把 Python 对象变成“可工作的通信 runtime”。

---

### 8. 同步本地 GPU Device Id

```python
# Synchronize device IDs
local_device_id = self.runtime.get_local_device_id()
device_ids = all_gather_object(local_device_id)
```

#### 语义

- `self.runtime.get_local_device_id()`：拿当前 rank 当前 GPU id
- `all_gather_object`：把它 broadcast 给所有 rank

#### 目的

让每个 rank 知道：

- 哪个 GPU 属于哪个 rank
- 所有 rank 的 GPU 编号映射关系

这是跨进程通信的底层前提。
后续 NVLink / RDMA 都依赖 GPU 之间的“谁和谁连接”的全局信息。

---

### 9. 同步 IPC Handles

```python
# Synchronize IPC handles
local_ipc_handle = self.runtime.get_local_ipc_handle()
ipc_handles = all_gather_object(local_ipc_handle)
```

#### IPC Handle 是什么

IPC = Inter-Process Communication，常用于 GPU 显存共享或跨进程访问。

这里每个 rank 获取自己的 IPC handle，再全局同步给所有 rank。这样其他 rank 就能正确访问这个 rank 的 CUDA buffer。

#### 作用

本质上是在建立“跨 rank 共享显存 / 共享 buffer”的能力。
DeepEP 的 dispatch/combine 依赖这些 handle 来建立 high-throughput / low-latency 的数据通路。

---

### 10. 有条件地初始化 NVSHMEM / IBGDA

```python
# Synchronize NVSHMEM unique IDs
root_unique_id = None
if self.runtime.get_num_rdma_ranks() > 1 or low_latency_mode:
```

这里的核心条件是：

- either there are multiple RDMA ranks
- or low-latency mode is on

也就是说，只要需要跨 rank 的 RDMA 语义，或者 low-latency 模式打算走 RDMA，就要初始化 NVSHMEM 相关东西。

#### 1）为什么要判断 `get_num_rdma_ranks() > 1 or low_latency_mode`

因为：

- 低延迟路径必须能跨 rank 直接访问 RDMA buffer
- 即使是单机，也有可能走 low-latency 模式
- 如果是多机 RDMA 通信，也必须走这套初始化

#### 2）为什么设置 `NVSHMEM_IB_ENABLE_IBGDA`

IBGDA 是低延迟 RDMA 通信关键路径。它让 NVSHMEM 走更直接的 IBGDA 方案而非普通路径。

#### 3）`NVSHMEM_IBGDA_NUM_RC_PER_PE`

这是每个 PE（Process Element）/ rank 的 QP 数。

`num_qps_per_rank` 传的是 `num_experts // num_ranks`，也就是“每个 rank 多少个本地 expert 对应多少个 QP”。

这说明 low-latency 路径的通信并发度和“expert 并行度”绑定得很深。

#### 4）为什么设 `NVSHMEM_QP_DEPTH`

它要求“QP 深度必须大于最大 in-flight WR 数”，代码中甚至注释写了：

> Make sure QP depth is always larger than the number of on-flight WRs, so that we can skip WQ slot check

也就是说，低延迟 kernel 对通信并发有很高要求，需要足够大 QP depth，以避免 WQ slot 检查卡住。

#### 5）为什么 `NVSHMEM_CUMEM_GRANULARITY = 2**29`

这是一个内存对齐/管理参数，注释写了：

> NVSHMEM initialization requires at least 256 MiB

也就是 NVSHMEM 初始化要求最小内存粒度，防止分配不稳定。

#### 6）为什么要同步 `root_unique_id`

`root_unique_id` 是 NVSHMEM 的唯一标识，类似“启动通信组的根节点信息”。

只有 root 节点生成它，然后通过 `all_gather_object` 让所有 rank 都拿到。

这是不少分布式通信框架的典型做法：只让某一个 rank 先生成全局通信 ID，然后每个 rank 都拿到一份。

---

### 11. 设置 IBGDA 和 NVSHMEM 环境变量

```python
    # Enable IBGDA
    assert num_qps_per_rank > 0
    os.environ['NVSHMEM_DISABLE_P2P'] = '0' if allow_nvlink_for_low_latency_mode else '1'
    os.environ['NVSHMEM_IB_ENABLE_IBGDA'] = '1'
    os.environ['NVSHMEM_IBGDA_NUM_RC_PER_PE'] = f'{num_qps_per_rank}'
```

这是低延迟模式的关键配置。

#### 解释

- `NVSHMEM_DISABLE_P2P`
  - 控制是否关闭 peer-to-peer
  - 如果允许 NVLink，就设置为 `0`，允许直接 P2P
- `NVSHMEM_IB_ENABLE_IBGDA`
  - 打开 IBGDA（InfiniBand GPUDirect Acceleration）模式
- `NVSHMEM_IBGDA_NUM_RC_PER_PE`
  - 每个 PE（rank）有多少个 RC QP（可靠连接）

这个设置直接表明：low-latency 路径不是普通 NCCL 通信，而是基于 NVSHMEM + IBGDA 的 RDMA 低时延数据移动方式。

---

### 12. 设置 QP Depth

```python
    # Make sure QP depth is always larger than the number of on-flight WRs, so that we can skip WQ slot check
    self.nvshmem_qp_depth = int(os.environ.get('NVSHMEM_QP_DEPTH', '1024'))
    os.environ['NVSHMEM_QP_DEPTH'] = str(self.nvshmem_qp_depth)
```

这里说明：

- `NVSHMEM_QP_DEPTH` 是 queue pair depth
- 这个值必须至少大于 in-flight WR 数量
- 这样可以“跳过 WQ slot check”

#### 这意味着什么

在低延迟路径里，通信非常频繁，必须让 QP 里有足够深度的 in-flight 请求，避免每次都“等 slot”。
这是为了减少延迟开销，提升低时延特性。

---

### 13. 降低 GPU Memory usage，并关闭某些非必须功能

```python
    # Reduce gpu memory usage
    # 6 default teams + 1 extra team
    os.environ['NVSHMEM_MAX_TEAMS'] = '7'
    # Disable NVLink SHARP
    os.environ['NVSHMEM_DISABLE_NVLS'] = '1'
    # NOTES: NVSHMEM initialization requires at least 256 MiB
    os.environ['NVSHMEM_CUMEM_GRANULARITY'] = f'{2 ** 29}'

    if not allow_mnnvl:
        # Disable multi-node NVLink detection
        os.environ['NVSHMEM_DISABLE_MNNVL'] = '1'
```

这几行都是针对“低延迟 + RDMA 通信”做的资源控制。

#### 关键含义

- `MAX_TEAMS = 7`：
  - 限制 NVSHMEM team 数，控制内存开销
- `DISABLE_NVLS = 1`：
  - 关闭 NVLink SHARP，不用复杂 NVLink routing
- `CUMEM_GRANULARITY = 2**29`：
  - 设定显存粒度，保证 NVSHMEM 初始化能申请足够内存
- `DISABLE_MNNVL`：
  - 如果不允许 MNNVL，就关闭那条路径

它们共同目标是：在低延迟通信场景下，把资源占用控制到最小，同时避免不必要的高级拓扑逻辑。

---

### 14. 生成并同步 NVSHMEM Unique Id

```python
    # Synchronize using the root ID
    if (low_latency_mode and self.rank == 0) or (not low_latency_mode and self.runtime.get_rdma_rank() == 0):
        root_unique_id = self.runtime.get_local_nvshmem_unique_id()
    nvshmem_unique_ids = all_gather_object(root_unique_id)
    root_unique_id = nvshmem_unique_ids[0 if low_latency_mode else self.runtime.get_root_rdma_rank(True)]
```

这是 NVSHMEM 初始化里非常典型的“root 节点选举 + 广播”。

#### 意义

- 只有 root rank 生成 `root_unique_id`
- 其他 rank 通过 `all_gather_object` 拿到全部信息
- 最后每个 rank 都拿到“真正要用的 root unique id”

#### 为什么分成两个路径

- `low_latency_mode`：root 是 rank 0
- 非 low_latency：root 是 RDMA root rank

这说明不同模式下，真正的“启动者”可能不是同一个 rank。

---

### 15. 最后同步整个 Runtime

```python
# Make CPP runtime available
self.runtime.sync(device_ids, ipc_handles, root_unique_id)
assert self.runtime.is_available()
```

这是整个构造函数完结的关键动作。

#### `self.runtime.sync(...)`

把前面收集到的这些信息全部喂给底层 runtime：

- device_ids
- ipc_handles
- root_unique_id

#### `is_available()`

最后断言 runtime 已经 ready。

这意味着：

- 没有这些数据同步，就不能真的开始 dispatch/combine
- 只有完成这一步，底层通信系统才算“可用”

---

## 这个 `__init__` 在架构上扮演的角色

可以把它概括成：

- Python 层负责：
  - 识别 rank
  - 组织 group
  - 收集全局元数据
  - 设置环境变量
  - 调用 runtime
- C++ runtime 负责：
  - 分配真实 buffer
  - 建立 NVSHMEM / IBGDA / RDMA 连接
  - 执行低延迟 dispatch/combine kernels

所以它本质上是“Runtime bootstrap”，不是单纯的对象初始化。

---

## 一句话总结

`Buffer.__init__` 的核心职责就是：

“在进程组中把每个 rank 的 GPU、IPC、NVSHMEM 标识统一收集起来，并初始化底层 RDMA/NVSHMEM runtime，使后续 low-latency dispatch/combine 能在全局可见状态下运行。”

如果你愿意，我下一步可以继续沿着这个 `__init__` 往下讲：

1. `self.runtime.low_latency_dispatch` 在 C++ 侧到底收到哪些参数
2. `handle` 里 `src_info` 和 `layout_range` 分别代表什么
3. 为什么 `combine` 一定要依赖 `handle`，而不是直接用原始 `topk_idx` 计算

这样会更容易把“Python 层封装”和“底层 runtime 真正做什么”串起来。

# Python API

## 构造函数

初始化中，最重要部分是创建 `Buffer`，接口定义与 `deep_ep/buffer.py`，构造函数创建 `runtime` 调用 `csrc/deep_ep.cpp` 里的 `Buffer` 的构造函数。

```python

# deep_ep/buffer.py

@@ -32, 79

```

### **输入参数解析**

1. 通信组与基础配置

- `group`: PyTorch 分布式通信组（`ProcessGroup`），定义参与通信的 rank 集合。
- `explicitly_destroy`: 是否需要显式调用 `destroy()` 释放资源（默认由析构函数释放，避免 Python 异常时挂起）。

1. 缓冲区大小配置

- `num_nvl_bytes`: 节点内 NVLink 通信缓冲区大小（字节），用于高吞吐量节点内通信。
- `num_rdma_bytes`: 节点间 RDMA 通信缓冲区大小（字节），用于低延迟模式或跨节点通信。

1. 通信模式与硬件控

- `low_latency_mode`: 是否启用低延迟模式（优化通信延迟，依赖 RDMA 和 IBGDA 技术）。
- `allow_nvlink_for_low_latency_mode`: 低延迟模式下是否允许 NVLink 通信（需注意与 hook 机制的兼容性）。
- `allow_mnnvl`: 是否允许多节点 NVLink 检测（禁用可减少节点间通信复杂度）。

1. RDMA 与 QP 配置

- `num_qps_per_rank`: RDMA 通信的队列对（QP）数量，低延迟模式下需等于本地专家数（影响通信并行度）。

### 主要流程

1. 硬件连接检查

```python

check_nvlink_connections(group)

```

首先调用 `check_nvlink_connections` 验证通信组内所有 rank 之间的 NVLink 连接是否可用（节点内通信依赖 NVLink，节点间依赖 RDMA），避免因硬件连接问题导致通信失败。

1. 基础属性初始化

包括以下参数：

```python

self.rank = group.rank() # 当前 rank 编号

self.group_size = group.size() # 通信组内总 rank 数

[self.group](http://self.group/) = group # 通信组对象（如 PyTorch 分布式 ProcessGroup）

self.num_nvl_bytes = num_nvl_bytes # NVLink 缓冲区大小（字节）

self.num_rdma_bytes = num_rdma_bytes # RDMA 缓冲区大小（字节）

self.low_latency_mode = low_latency_mode # 是否启用低延迟模式

self.explicitly_destroy = explicitly_destroy # 是否需要显式释放资源

```

记录通信组信息、缓冲区大小、模式配置等基础参数，为后续资源分配和模式切换提供依据。

1. C++ 运行时初始化

```python

self.runtime = deep_ep_cpp.Buffer(…)

```

创建 C++ 底层运行时实例（`deep_ep_cpp.Buffer`），封装了实际的通信逻辑（如 NVLink/RDMA 操作）。Python 层通过调用 `self.runtime` 的方法间接操作底层通信。

1. 分布式信息同步

为确保所有 rank 协同工作，需要同步三类关键信息：

- **设备 ID 同步**：

每个 rank 获取本地设备 ID（GPU 编号），并通过 `dist.all_gather_object` 收集所有 rank 的设备 ID，确保跨设备通信时的设备一致性。

- **IPC 句柄同步**：

收集所有 rank 的进程间通信（IPC）句柄，用于节点内多进程共享内存访问（如同一节点内不同进程的 GPU 间通信）。

- **NVSHMEM 配置与 ID 同步**：

若启用低延迟模式或存在多 RDMA rank（节点间通信），则配置 NVSHMEM（NVIDIA 分布式共享内存库）环境变量（如 QP 数量、缓冲区粒度、禁用多节点 NVLink 检测等），并同步 root 节点的 NVSHMEM 唯一 ID，确保所有 rank 能通过 RDMA 建立连接。

1. 运行时就绪确认

```python

self.runtime.sync(device_ids, ipc_handles, root_unique_id)

assert self.runtime.is_available()

```

调用 `sync` 方法将同步后的设备 ID、IPC 句柄、NVSHMEM ID 传入 C++ 运行时，完成最终初始化，并断言运行时就绪（确保后续通信操作可用）。

### Distributed

同步 `device IDs, IPC handles, NVSHMEM unique IDs` 使用的是 `dist.ProcessGroup`，它是 PyTorch 分布式计算库 `torch.distributed` 里的一个核心类，它代表了一组参与分布式计算的进程。在分布式训练或者计算场景中，通常会有多个进程同时运行，这些进程需要相互通信和协作，`ProcessGroup` 就为这些进程提供了一个逻辑上的分组，方便管理和控制进程间的通信。

### 作用

在当前 `Buffer` 类的 `__init__` 方法里，`dist.ProcessGroup` 对象 `group` 主要有以下几个用途：

#### 1. 获取进程信息

```python

self.rank = group.rank()

self.group_size = group.size()

```

- `group.rank()`：返回当前进程在该进程组里的唯一标识符，即进程编号。`rank` 是一个非负整数，范围从 0 到 `group.size() - 1`。不同进程的 `rank` 不同，可用于区分不同进程，在分布式通信里，不同 `rank` 的进程可能承担不同任务。
- `group.size()`：返回进程组里的进程总数，也就是参与分布式计算的进程数量。

#### 2. 进程间通信同步

```python

dist.all_gather_object(device_ids, local_device_id, group)

dist.all_gather_object(ipc_handles, local_ipc_handle, group)

dist.all_gather_object(nvshmem_unique_ids, root_unique_id, group)

```

- `dist.all_gather_object` 是 PyTorch 分布式库提供的一个集体通信函数，作用是把每个进程的某个对象收集到所有进程中。这里通过传入 `group` 参数，指定在哪个进程组内进行通信同步操作。借助这个函数，所有进程能获取到其他进程的设备 ID、IPC 句柄以及 NVSHMEM 唯一 ID 等信息。

## Dispatch API

Buffer 有一个 `runtime` 成员实例化 C++ 接口的实现，在 [[2-组件概述和加速方法]] 中可以看到 Buffer 提供了的几个 API 里调用 `runtime` 来实现 dispatch 和 combine 功能。

`dispatch` 是 Buffer 类的核心方法，负责将输入的 tokens **分发到不同的分布式 rank**（支持节点内 NVLink 和节点间 RDMA 通信），是 MoE 模型中专家并行（EP）的关键步骤。其核心逻辑是根据 token 选择的专家索引（`topk_idx`），将 tokens 路由到对应专家所在的 rank，并返回接收端的 tokens 及通信元信息。

### 输入参数解析

1. 输入数据（待分发的 tokens）

- `x`: 待分发的 tokens 张量或 FP8 元组
- 单张量模式：`torch.bfloat16` 类型，形状 `[num_tokens, hidden]`
- FP8 元组模式：`(x_e4m3, x_scales)`，其中 `x_e4m3` 为 `torch.float8_e4m3fn` 类型（`[num_tokens, hidden]`），`x_scales` 为 `torch.float` 类型（`[num_tokens, hidden//128]`，需满足 `hidden` 可被 128 整除）

1. 缓存模式（复用布局信息，`handle≠None` 时）

- `handle`: 元组，缓存的通信布局信息（如前缀矩阵、接收索引等），由非缓存模式首次调用返回，用于后续分发时跳过布局计算，提升效率。缓存模式下需设置 `topk_idx=None`

1. 通信布局（非缓存模式必需，`handle=None` 时）

- `num_tokens_per_rank`: 张量 `[num_ranks]`（`torch.int`），每个 rank 需接收的 tokens 数量
- `num_tokens_per_rdma_rank`: 张量 `[num_rdma_ranks]`（`torch.int`），每个 rdma rank 需接收的 tokens 数量，intranode 为 `None`
- `is_token_in_rank`: 张量 `[num_tokens, num_ranks]`（`torch.bool`），标记每个 token 是否发送到对应 rank
- `num_tokens_per_expert`: 张量 `[num_experts]`（`torch.int`），每个专家需接收的 tokens 数量

1. 专家选择信息

- `topk_idx`: 张量 `[num_tokens, num_topk]`（`torch.int64`），每个 token 选择的专家索引（`-1` 表示未选择），非缓存模式必需
- `topk_weights`: 张量 `[num_tokens, num_topk]`（`torch.float`），每个 token 对所选专家的权重，用于后续聚合加权

1. 性能与内存配置

- `expert_alignment`: 整数，接收的 tokens 数量需对齐到该值（如 16/32），优化内存访问效率，默认 1
- `num_worst_tokens`: 整数，指定接收 tokens 的最大可能数量（仅节点内模式支持），避免 CPU-GPU 同步，提升 CUDA Graph 兼容性，默认 0
- `config`: `Config` 对象，分发内核的性能调优参数（如 SM 数量、块大小等），默认通过 `get_dispatch_config` 根据 rank 数量自动选择

1. 通信同步与流控制

- `previous_event`: `EventOverlap` 对象，分发开始前需等待的前置事件（如上游计算完成）
- `async_finish`: 布尔值，若为 `True`，当前 CUDA 流不等待分发内核完成，直接返回（需通过返回的 `event` 手动同步），默认 `False`
- `allocate_on_comm_stream`: 布尔值，若为 `True`，分配的输出张量所有权归通信流（而非默认计算流），优化异步通信效率

### 输出参数解析

返回一个元组，包含接收的 tokens、专家信息、通信句柄及同步事件，具体如下：

- `recv_x`: 接收的 tokens 数据，类型与输入 `x` 一致
- 单张量模式：`torch.bfloat16` 类型，形状 `[num_recv_tokens, hidden]`
- FP8 元组模式：`(recv_x_e4m3, recv_x_scales)`，其中 `recv_x_e4m3` 为 `torch.float8_e4m3fn` 类型，`recv_x_scales` 为 `torch.float` 类型
- `recv_topk_idx`: 接收的专家索引张量（仅非缓存模式返回，缓存模式为 `None`）
- 形状 `[num_recv_tokens, num_topk]`，`torch.int64` 类型，与输入 `topk_idx` 对应
- `recv_topk_weights`: 接收的专家权重张量（仅非缓存模式返回，缓存模式为 `None`）
- 形状 `[num_recv_tokens, num_topk]`，`torch.float` 类型，与输入 `topk_weights` 对应
- `num_recv_tokens_per_expert_list`: 本地专家接收 tokens 数量列表（仅非缓存模式返回，缓存模式为 `None`）
- Python 列表 `[num_local_experts]`，每个元素为对应专家接收的 tokens 数（已按 `expert_alignment` 对齐）；若 `num_worst_tokens > 0`，返回空列表
- `handle`: 通信布局句柄（仅非缓存模式返回，缓存模式为 `None`）
- 元组，包含前缀矩阵、接收索引等布局信息，用于后续缓存模式分发（`handle≠None` 时复用）
- `event`: 分发内核完成事件（仅 `async_finish=True` 时有效）
- `EventOverlap` 对象，用于同步异步通信操作的完成状态

### **关键逻辑分支**

1. **节点间/节点内通信切换**：

- 若 `runtime.get_num_rdma_ranks() > 1`（存在多节点 RDMA 通信），调用 `internode_dispatch`；
- 否则为节点内通信，调用 `intranode_dispatch`（依赖 NVLink）。

1. **缓存/非缓存模式**：

- **缓存模式**（`handle is not None`）：复用 `handle` 中的布局信息，直接执行分发，跳过布局计算，减少重复计算，提升效率；
- **非缓存模式**（`handle is None`）：需提供 `num_tokens_per_rank`/`is_token_in_rank`/`num_tokens_per_expert`，计算布局并返回新 `handle`。

1. `async_finish` 与 `event`：异步通信与同步控制

两者配合实现通信过程的异步执行与显式同步，避免阻塞当前 CUDA 流，提升计算与通信的重叠效率。

**`async_finish`：控制是否异步执行**

- **类型**：布尔值（默认 `False`）。
- **作用**：
- 当 `async_finish=False`（默认）：当前 CUDA 流会等待通信内核执行完成后再继续，确保数据就绪。
- 当 `async_finish=True`：当前 CUDA 流不等待通信完成，直接返回，允许后续计算与通信并行执行（需通过 `event` 手动同步）。

**`event`：通信完成的同步标记**

- **类型**：`EventOverlap` 对象（封装 CUDA 事件）。
- **作用**：

当 `async_finish=True` 时，`event` 记录通信内核的完成状态。用户可通过 `event.wait()` 等操作显式等待通信完成，避免后续计算访问未就绪的数据。

1. 调用 C++ 接口：

节点内就直接调用 C++ runtime `dispatch intranode`。

节点间再次处理 python 接口的 `dispatch internode`。

### **总结**

`dispatch` 方法通过灵活的参数配置，支持 MoE 模型中 tokens 的高效分布式分发，兼顾高吞吐量（节点内 NVLink）和低延迟（节点间 RDMA）场景。其核心价值在于：

- 支持 BF16/FP8 数据类型，平衡精度与带宽；
- 提供缓存机制复用布局信息，减少重复计算；
- 通过事件同步和流控制，支持通信与计算重叠，提升整体性能。

---
