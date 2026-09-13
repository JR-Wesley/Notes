# Low Latency Buffer

我把默认参数代入到真实代码里，按 `test_loop -> Buffer(...) -> C++ Buffer::Buffer(...) -> sync(...)` 的顺序串起来，解释每一个值是如何落进 runtime 的。

## 默认执行配置是怎么落到 `Buffer` 里的

从 test_low_latency.py 看，默认配置在 `__main__` 里是：

- `--num-processes 8`
- `--num-tokens 128`
- `--hidden 7168`
- `--num-topk 8`
- `--num-experts 288`
- `--allow-mnnvl` 不开
- `--disable-nvlink` 不开
- `--use-logfmt` 不开
- `--pressure-test` 不开
- `--shrink-test` 不开

然后 `test_loop(local_rank, num_local_ranks, args)` 里执行：

```python
rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
```

在默认 8 进程情况下，通常是：

- `local_rank` ∈ {0,1,2,3,4,5,6,7}
- `num_local_ranks = 8`
- `num_ranks = 8`
- `group` 是全局通信组

接着：

```python
num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, num_ranks, num_experts)
```

这里实际代入值是：

- `num_max_dispatch_tokens_per_rank = 128`
- `hidden = 7168`
- `num_ranks = 8`
- `num_experts = 288`

然后创建 buffer：

```python
buffer = deep_ep.Buffer(
    group,
    num_rdma_bytes=num_rdma_bytes,
    low_latency_mode=True,
    num_qps_per_rank=num_experts // num_ranks,
    allow_nvlink_for_low_latency_mode=not args.disable_nvlink,
    explicitly_destroy=True,
    allow_mnnvl=args.allow_mnnvl,
    enable_shrink=args.shrink_test)
```

所以这里得到的具体值是：

- `num_nvl_bytes = 0`（默认不传，等价于 0）
- `num_rdma_bytes = 计算得到的低延迟 RDMA buffer size`
- `low_latency_mode = True`
- `num_qps_per_rank = 288 // 8 = 36`
- `allow_nvlink_for_low_latency_mode = True`
- `explicitly_destroy = True`
- `allow_mnnvl = False`
- `enable_shrink = False`

这就是默认测试的真实启动参数。

---

## Python 层如何进入 C++ Runtime

在 legacy.py 的 `Buffer.__init__` 里，真正创建底层 runtime 的语句是：

```python
self.runtime = _C.Buffer(
    self.rank,
    self.group_size,
    num_nvl_bytes,
    num_rdma_bytes,
    low_latency_mode,
    explicitly_destroy,
    enable_shrink,
    allow_mnnvl
)
```

也就是：

- Python 侧把参数传给 C++ `Buffer`
- C++ runtime 负责真正分配共享 memory 和初始化通信环境

这里 `num_nvl_bytes` 是 0，所以在默认低延迟配置下，C++ 侧几乎所有“真正的初始化”都走 RDMA/NVSHMEM 分支，而不是 NVLink 的 local IPC 分支。

---

## C++ Runtime 构造函数的真实执行顺序

对应的构造函数在 buffer.hpp：

```cpp
Buffer(int rank,
       int num_ranks,
       int64_t num_nvl_bytes,
       int64_t num_rdma_bytes,
       bool low_latency_mode,
       bool explicitly_destroy,
       bool enable_shrink,
       bool use_fabric)
```

下面按真实代码一段一段解释。

---

## 1. 成员初始化

```cpp
: rank(rank),
  num_ranks(num_ranks),
  num_nvl_bytes(num_nvl_bytes),
  num_rdma_bytes(num_rdma_bytes),
  enable_shrink(enable_shrink),
  low_latency_mode(low_latency_mode),
  explicitly_destroy(explicitly_destroy),
  comm_stream(at::cuda::getStreamFromPool(true)),
  shared_memory_allocator(use_fabric)
```

这一步执行的是：

- 保存全局 rank / 总 rank 数
- 保存通信 buffer 大小
- 保存低延迟模式和 shrink 模式 flag
- 设置通信专用 stream
- 初始化 shared memory allocator

这里 `comm_stream` 是后面所有 dispatch/combine 通信会用到的 stream，目的是把“通信”和“计算”分离，避免互相卡住。

---

## 2. 计算 Metadata 区域大小

```cpp
int64_t barrier_signal_bytes = LEGACY_NUM_MAX_NVL_PEERS * sizeof(int);
int64_t buffer_ptr_bytes = LEGACY_NUM_MAX_NVL_PEERS * sizeof(void*);
int64_t barrier_signal_ptr_bytes = LEGACY_NUM_MAX_NVL_PEERS * sizeof(int*);
```

这个意思是：

- `barrier_signal_bytes`：每个 peer 的 barrier signal 需要多少字节
- `buffer_ptr_bytes`：peer buffer ptr 数组需要多少字节
- `barrier_signal_ptr_bytes`：peer barrier ptr 数组需要多少字节

它们不是随便放的，而是为了把“共享区头信息”一起放进同一块 shared memory 里，供 GPU kernel 读取。

> 这也是为什么在 NVLink buffer 分配时，内存布局是：
> `[data region | barrier signal | buffer ptr array | barrier ptr array]`

---

## 3. 参数合法性检查

```cpp
EP_STATIC_ASSERT(LEGACY_NUM_BUFFER_ALIGNMENT_BYTES % sizeof(int4) == 0, "Invalid alignment");
EP_HOST_ASSERT(num_nvl_bytes % LEGACY_NUM_BUFFER_ALIGNMENT_BYTES == 0 and
               (num_nvl_bytes <= std::numeric_limits<int>::max() or num_rdma_bytes == 0));
EP_HOST_ASSERT(num_rdma_bytes % LEGACY_NUM_BUFFER_ALIGNMENT_BYTES == 0 and
               (low_latency_mode or num_rdma_bytes <= std::numeric_limits<int>::max()));
EP_HOST_ASSERT(num_nvl_bytes / sizeof(int4) < std::numeric_limits<int>::max());
EP_HOST_ASSERT(num_rdma_bytes / sizeof(int4) < std::numeric_limits<int>::max());
EP_HOST_ASSERT(0 <= rank and rank < num_ranks and (num_ranks <= LEGACY_NUM_MAX_NVL_PEERS * LEGACY_NUM_MAX_RDMA_PEERS or low_latency_mode));
EP_HOST_ASSERT(num_ranks < LEGACY_NUM_MAX_NVL_PEERS or num_ranks % LEGACY_NUM_MAX_NVL_PEERS == 0);
if (num_rdma_bytes > 0)
    EP_HOST_ASSERT(num_ranks > LEGACY_NUM_MAX_NVL_PEERS or low_latency_mode);
```

对你这个默认配置，关键点是：

- `num_nvl_bytes = 0`
- `num_rdma_bytes > 0`
- `low_latency_mode = True`

所以这些检查会放宽到允许 low-latency 场景。

这里最关键的两条是：

```cpp
EP_HOST_ASSERT(0 <= rank and rank < num_ranks ...)
EP_HOST_ASSERT(num_ranks < LEGACY_NUM_MAX_NVL_PEERS or num_ranks % LEGACY_NUM_MAX_NVL_PEERS == 0)
```

因为默认 `num_ranks = 8`，而 `LEGACY_NUM_MAX_NVL_PEERS = 8`，所以：

- `8 % 8 == 0` 成立
- 这说明当前默认配置正好匹配一组 NVLink peer

---

## 4. 计算 `rdma_rank` 和 `nvl_rank`

```cpp
CUDA_RUNTIME_CHECK(cudaGetDevice(&device_id));
rdma_rank = rank / LEGACY_NUM_MAX_NVL_PEERS, nvl_rank = rank % LEGACY_NUM_MAX_NVL_PEERS;
num_rdma_ranks = std::max(1, num_ranks / LEGACY_NUM_MAX_NVL_PEERS), num_nvl_ranks = std::min(num_ranks, LEGACY_NUM_MAX_NVL_PEERS);
```

对默认 `rank` 和 `num_ranks = 8`：

- `rdma_rank = rank / 8`
- `nvl_rank = rank % 8`

因此：

- rank 0 -> `(rdma_rank=0, nvl_rank=0)`
- rank 1 -> `(0,1)`
- ...
- rank 7 -> `(0,7)`

也就是说：

- 8 个 rank 组成一个 NVLink group
- 这组内部的 peer count 正好等于 `8`

`num_rdma_ranks`：

```cpp
std::max(1, 8 / 8) = 1
```

`num_nvl_ranks`：

```cpp
std::min(8, 8) = 8
```

这意味着：

- 在默认低延迟测试中，它本质上是“一个 RDMA group + 一个 NVLink group”，并且规模刚好等于 8

---

## 5. 获取 GPU 的 SM 数量

```cpp
cudaDeviceProp device_prop = {};
CUDA_RUNTIME_CHECK(cudaGetDeviceProperties(&device_prop, device_id));
num_device_sms = device_prop.multiProcessorCount;
```

这里我们拿到当前 GPU 的 SM 数，比如 H100 可能是 132，A100 可能是 108，等。

这个值被后面的 kernel 配置用到，尤其是：

- channel 数量
- grid/block 分配
- per-channel buffer size 上限

因为后面 `dispatch / combine` 里会做：

```cpp
int num_channels = config.num_sms / 2;
```

所以 `num_device_sms` 直接影响 kernel 的并行度和通信分块。

---

## 6. 进一步检查每个 Channel 的 Bytes 不能太大

```cpp
EP_HOST_ASSERT(ceil_div<int64_t>(num_nvl_bytes, num_device_sms / 2) < std::numeric_limits<int>::max());
EP_HOST_ASSERT(ceil_div<int64_t>(num_rdma_bytes, num_device_sms / 2) < std::numeric_limits<int>::max());
```

这表明：

- 每个 channel 对应的 buffer 负载不能超过 `int` 的上限
- 否则 index / offset / size 会溢出

对默认测试中：

- `num_nvl_bytes = 0`
- `num_rdma_bytes` 是一个大值，但它已由 `get_low_latency_rdma_size_hint(...)` 计算出“足够大的合法值”

所以这一步大概率会通过。

---

## 7. NVLink Buffer 分支被跳过

```cpp
if (num_nvl_bytes > 0) {
    ...
}
```

默认参数里 `num_nvl_bytes = 0`，所以这整段不会执行。

也就是说，默认低延迟路径中：

- 不会申请 NVLink local shared buffer
- 不会做本地 IPC 句柄注册
- 不会进入 local NVLink peer map 流程

因为低延迟模式的重点是 RDMA/NVSHMEM，不是 NVLink local high-throughput。

这一点和你原来“old summary 里认为通用 buffer 都分配 NVLink + RDMA”不完全一样，当前真实代码下默认低延迟场景是“NVLink path 关闭，RDMA path 启动”。

---

## 8. RDMA Buffer 的真正初始化：`sync()` 才真正发生

这里很关键：构造函数本身不直接初始化 NVSHMEM，也不直接生成所有 rank 的共享状态。

真正的全局初始化发生在 buffer.hpp 里的 `sync(...)`：

```cpp
void sync(const std::vector<int>& device_ids,
          const std::vector<std::optional<pybind11::bytearray>>& all_gathered_handles,
          const std::optional<pybind11::bytearray>& root_unique_id_opt)
```

在 Python 侧的 legacy.py 里，构造函数最后执行：

```python
self.runtime.sync(device_ids, ipc_handles, root_unique_id)
assert self.runtime.is_available()
```

所以，C++ 构造函数本体只是“准备了基本成员/资源”，而真正让 runtime usable 的是 `sync()`。

这个层次非常关键，因为它说明：

- 构造函数 = 资源准备阶段
- `sync()` = 分布式通信环境就绪阶段

---

## 9. 默认低延迟测试下，真正发生的 C++ Runtime 初始化

在默认值下，`sync()` 会走到这条逻辑：

```cpp
if (num_rdma_bytes > 0) {
    EP_HOST_ASSERT(root_unique_id_opt.has_value());
    auto nvshmem_rank = low_latency_mode ? rank : rdma_rank;
    auto num_nvshmem_ranks = low_latency_mode ? num_ranks : num_rdma_ranks;
    EP_HOST_ASSERT(nvshmem_rank == nvshmem::init(root_unique_id, nvshmem_rank, num_nvshmem_ranks,
                                                 low_latency_mode ? LEGACY_NUM_MAX_NVL_PEERS : 0));
    rdma_buffer_ptr = nvshmem::alloc(num_rdma_bytes, LEGACY_NUM_BUFFER_ALIGNMENT_BYTES);
    CUDA_RUNTIME_CHECK(cudaMemset(rdma_buffer_ptr, 0, num_rdma_bytes));
    nvshmem::barrier(true);
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
}
```

对默认测试：

- `low_latency_mode = True`
- `rank = 0..7`
- `num_ranks = 8`

所以：

- `nvshmem_rank = rank`
- `num_nvshmem_ranks = 8`
- `low_latency_mode ? LEGACY_NUM_MAX_NVL_PEERS : 0 = 8`

这意味着：

- 8 个 rank 全部参与 NVSHMEM 的低延迟通信组
- 每个 rank 分别初始化自己的 `nvshmem_rank`
- 再 allocate 一块 `rdma_buffer_ptr` 作为共享 RDMA buffer

然后：

```cpp
CUDA_RUNTIME_CHECK(cudaMemset(rdma_buffer_ptr, 0, num_rdma_bytes));
```

把这块共享 buffer 清零，保证低延迟路径第一次运行时不带历史脏数据。

---

## 10. 结论：默认低延迟测试时，C++ Runtime 构造的真实状态

对默认执行配置来说，构造函数的真实状态是：

- `num_nvl_bytes = 0`
- `num_rdma_bytes > 0`
- `low_latency_mode = True`
- `num_ranks = 8`
- `rdma_rank = rank / 8`
- `nvl_rank = rank % 8`
- `num_rdma_ranks = 1`
- `num_nvl_ranks = 8`

也就是说：

- 它不是普通“全局 NVLink + RDMA buffer 都初始化”
- 它是一个“低延迟 8-rank NVSHMEM 组”的初始化，重点在 RDMA buffer 和通信状态同步
- 真实的资源准备和全局 ready 是在 `sync()` 里完成的，不是单纯在构造函数体内结束

---

## 最简版一句话总结

默认低延迟测试的 `Buffer` 构造顺序可以概括成：

“把 rank / 设备 / buffer 大小保存下来，按 8-peer NVLink 拓扑划分 rank，然后在低延迟 mode 下跳过 NVLink IPC 分支，进入 NVSHMEM/RDMA 初始化，并在 `sync()` 中完成所有 peer 的 handle 同步和 shared buffer ready，最后设置 `available = true`，从而准备好后续 `low_latency_dispatch` / `low_latency_combine`。”

# Dispatch

下面以 `test_low_latency.py` 的默认规模为例，分析最简路径：

- `num_processes = num_ranks = 8`
- `num_tokens = 128`
- `hidden = 7168`
- `num_topk = 8`
- `num_experts = 288`
- `use_fp8 = false`
- `use_logfmt = false`
- `zero_copy = false`
- `return_recv_hook = false`
- `async_finish = false`
- `shrink_test = false`

注意：测试脚本本身会循环测试多种配置；这里选择其中最简单的一次。

---

## 1. Python 测试入口

测试构造输入：

```python
x = torch.ones((128, 7168), dtype=torch.bfloat16) * (rank - 128)
x[:, -128:] = torch.arange(128).view(-1, 1)
```

因此，对 rank `r`：

```text
x[token, 0:7040] = r - 128
x[token, 7040:7168] = token
```

例如 rank 3、token 5：

```text
x[5, 0:7040] = -125
x[5, 7040:7168] = 5
```

`topk_idx` 的形状是：

```text
[128, 8]
```

每个 token 选择 8 个 expert。由于 `topk_idx` 是随机生成的，某个例子可以是：

```text
token 5:
topk_idx[5]     = [7, 38, 101, 145, 190, 211, 250, -1]
topk_weights[5] = [0.20, 0.15, 0.10, 0.18, 0.12, 0.10, 0.15, 0.08]
```

`-1` 表示该位置不参与通信。

---

## 2. `Buffer.low_latency_dispatch`

Python wrapper 最终调用：

```python
self.runtime.low_latency_dispatch(...)
```

对应 `buffer.hpp` 中的：

```cpp
void low_latency_dispatch(...)
```

首先检查：

```cpp
EP_HOST_ASSERT(low_latency_mode);
```

当前 Buffer 必须使用 low-latency 模式。

接着检查输入：

```cpp
x.dim() == 2
x.shape == [128, 7168]
x.dtype == BF16
```

以及：

```cpp
num_experts % num_ranks == 0
```

这里：

```text
288 % 8 = 0
```

每个 rank 的 local expert 数量：

```cpp
num_local_experts = num_experts / num_ranks
                   = 288 / 8
                   = 36
```

---

## 3. 两个 RDMA Buffer

Low-latency 使用双 buffer：

```cpp
auto buffer = layout.buffers[low_latency_buffer_idx];
auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];
```

初始时：

```text
low_latency_buffer_idx = 0
buffer     = buffers[0]
next_buffer = buffers[1]
```

下一次调用会切换为：

```text
buffer     = buffers[1]
next_buffer = buffers[0]
```

这样可以让当前通信使用一个 buffer，同时清理另一个 buffer。

---

## 4. 分配接收张量

关闭 FP8 后：

```cpp
packed_recv_x =
torch::empty(
    {num_local_experts,
     num_ranks * num_max_dispatch_tokens_per_rank,
     hidden},
    BF16)
```

假设：

```text
num_max_dispatch_tokens_per_rank = 128
```

则：

```text
packed_recv_x.shape = [36, 8 * 128, 7168]
                    = [36, 1024, 7168]
```

它的逻辑结构是：

```text
packed_recv_x[local_expert][source_rank * 128 + slot][hidden]
```

例如：

```text
packed_recv_x[12][3 * 128 + 5]
```

表示：

```text
本 rank 的 local expert 12
从 source rank 3 收到的第 5 个 token
```

同时分配：

```cpp
packed_recv_src_info
```

形状：

```text
[36, 1024]
```

保存每个接收 token 的原始 source index。

```cpp
packed_recv_layout_range
```

形状：

```text
[36, 8]
```

每个元素编码：

```text
高 32 位：该 source rank 的起始位置
低 32 位：该 source rank 收到的 token 数量
```

```cpp
packed_recv_count
```

形状：

```text
[36]
```

保存每个 local expert 最终收到的 token 总数。

---

## 5. 创建 `launcher`

代码构造了一个 lambda：

```cpp
auto launcher = [=, this](int phases) {
    internode_ll::dispatch(..., phases);
};
```

最简调用：

```cpp
launcher(LEGACY_LOW_LATENCY_SEND_PHASE |
         LEGACY_LOW_LATENCY_RECV_PHASE);
```

因此本次 kernel 同时执行：

```text
SEND phase
RECV phase
```

如果使用 `return_recv_hook=true`，第一次只执行 SEND，之后 hook 再执行 RECV；当前最简配置关闭 hook，所以一次完成两阶段。

---

# 6. 进入 `internode_ll::dispatch`

入口位于 `internode_ll.cu`：

```cpp
void dispatch(
    void* packed_recv_x,
    ...
    int num_tokens,
    int hidden,
    int num_max_dispatch_tokens_per_rank,
    int num_topk,
    int num_experts,
    int rank,
    int num_ranks,
    ...
)
```

实际传入值：

| 变量 | 值 |
|---|---:|
| `num_tokens` | 128 |
| `hidden` | 7168 |
| `num_max_dispatch_tokens_per_rank` | 128 |
| `num_topk` | 8 |
| `num_experts` | 288 |
| `num_ranks` | 8 |
| `rank` | 当前 rank，例如 3 |
| `use_fp8` | false |
| `use_ue8m0` | false |
| `round_scale` | false |

---

## 7. 计算 Warp 分组

```cpp
const int num_warp_groups = ceil_div(num_experts, num_device_sms);
```

`num_device_sms` 取自 GPU 实际 SM 数。

例如假设 GPU 有 80 个 SM：

```text
num_warp_groups = ceil_div(288, 80)
                = 4
```

然后：

```cpp
const int num_warps_per_group = 32 / num_warp_groups;
                           = 32 / 4
                           = 8
```

```cpp
const auto num_warps = num_warp_groups * num_warps_per_group;
                       = 4 * 8
                       = 32
```

每个 block 的线程数：

```text
num_warps * 32 = 32 * 32 = 1024
```

kernel 的组织形式近似为：

```text
grid.x  = 72
block.x = 1024
```

因为：

```cpp
num_sms = ceil_div(num_experts, num_warp_groups)
        = ceil_div(288, 4)
        = 72
```

这里的 `num_sms` 是 kernel 实际使用的 block 数，不一定等于物理 SM 数。

---

## 8. Expert 如何分配给 Warp Group

kernel 内部：

```cpp
const auto warp_id = thread_id / 32;
const auto warp_group_id = warp_id / num_warps_per_group;
const auto sub_warp_id = warp_id % num_warps_per_group;
const auto responsible_expert_idx =
    sm_id * num_warp_groups + warp_group_id;
```

假设：

```text
num_warp_groups = 4
num_warps_per_group = 8
sm_id = 10
warp_id = 19
```

则：

```text
warp_group_id = 19 / 8 = 2
sub_warp_id   = 19 % 8 = 3
```

负责的全局 expert：

```text
responsible_expert_idx
= 10 * 4 + 2
= 42
```

expert 42 属于哪个 rank？

```cpp
dst_rank = responsible_expert_idx / num_local_experts;
         = 42 / 36
         = 1
```

它是 rank 1 上的 local expert：

```cpp
local_expert_idx = 42 % 36 = 6
```

所以这个 warp group 的职责是：

```text
处理 global expert 42
即 rank 1 的 local expert 6
```

---

# 9. Dispatch 的消息大小

关闭 FP8 时，dispatch 消息大小是：

```cpp
num_bytes_per_msg =
    sizeof(int4) + hidden * sizeof(nv_bfloat16)
```

代入：

```text
sizeof(int4) = 16
sizeof(BF16) = 2
hidden = 7168
```

因此：

```text
num_bytes_per_msg
= 16 + 7168 * 2
= 14352 bytes
```

消息布局：

```text
offset 0       : 16 bytes，source index/control 信息
offset 16      : 14336 bytes，7168 个 BF16
```

当前 token 的消息大致是：

```text
[source_idx][BF16 hidden vector]
```

关闭 FP8 后，不需要 scale buffer，也不做 FP8 转换。

---

# 10. Dispatch 发送阶段

kernel 首先统计每个 expert 有多少 token：

```cpp
topk_idx[token_idx][k]
```

对于每一个有效 expert：

```text
if topk_idx[token][k] == -1:
    跳过
else:
    计算目标 expert 和目标 rank
```

例如：

```text
topk_idx[5] = [7, 38, 101, 145, 190, 211, 250, -1]
```

各 expert 的归属：

| expert | owner rank | local expert |
|---:|---:|---:|
| 7 | `7 / 36 = 0` | 7 |
| 38 | `38 / 36 = 1` | 2 |
| 101 | `101 / 36 = 2` | 29 |
| 145 | `145 / 36 = 4` | 1 |
| 190 | `190 / 36 = 5` | 10 |
| 211 | `211 / 36 = 5` | 31 |
| 250 | `250 / 36 = 6` | 34 |
| -1 | 无 | 跳过 |

当前 rank 会根据 top-k 结果，把 token 5 发给这些 expert 的 owner rank。

---

## 11. 发送地址

kernel 中类似：

```cpp
const auto dst_ptr =
    reinterpret_cast<uint64_t>(rdma_recv_x) +
    (global_expert_idx * num_max_dispatch_tokens_per_rank + src_idx)
        * num_bytes_per_msg;
```

假设：

```text
global_expert_idx = 38
num_max_dispatch_tokens_per_rank = 128
src_idx = 5
num_bytes_per_msg = 14352
```

则目标 slot 偏移：

```text
(38 * 128 + 5) * 14352
= 4869 * 14352
= 69,? bytes
```

重要的是公式，而不是具体字节数：

```text
目标地址 =
接收 buffer
+ (expert_id * capacity + 原始 token index) * 消息大小
```

这样远端收到数据后，可以根据：

```text
expert_id
source token index
```

直接定位。

之后通过：

```cpp
nvshmemi_ibgda_put_nbi_warp(...)
```

执行非阻塞 IBGDA put。

---

# 12. Dispatch 接收阶段

kernel 进入：

```cpp
LOW_LATENCY_DISPATCH_RECV:
```

对于负责的 expert：

```cpp
const auto src_rank =
    responsible_expert_idx / num_local_experts;

const auto local_expert_idx =
    responsible_expert_idx % num_local_experts;
```

如果当前负责：

```text
responsible_expert_idx = 42
```

则：

```text
src_rank = 42 / 36 = 1
local_expert_idx = 42 % 36 = 6
```

接收地址：

```cpp
rdma_recv_x_uint8 =
    rdma_recv_x + src_rank * num_max_dispatch_tokens_per_rank
                  * num_bytes_per_msg;
```

也就是从 source rank 1 的区域读取。

本地 packed 输出地址：

```cpp
recv_x_int4 =
    packed_recv_x
    + local_expert_idx
      * num_ranks
      * num_max_dispatch_tokens_per_rank
      * hidden_int4;
```

BF16 的 `hidden_int4`：

```text
hidden_int4 = hidden * sizeof(BF16) / sizeof(int4)
            = 7168 * 2 / 16
            = 896
```

所以一个 local expert 的数据区域包含：

```text
8 个 source rank
* 128 个 token slot
* 896 个 int4
```

---

## 13. 接收数量和布局

接收 warp 会等待：

```cpp
rdma_recv_count
```

这个计数表示某个 source rank 发来了多少 token。

收到后：

```cpp
recv_token_begin_idx =
    atomicAdd(packed_recv_count + local_expert_idx,
              num_recv_tokens);
```

例如 local expert 6：

```text
原来 packed_recv_count[6] = 20
本次 source rank 1 收到 3 个 token
```

则：

```text
recv_token_begin_idx = 20
packed_recv_count[6] = 23
```

这 3 个 token 写入：

```text
packed_recv_x[6][20:23]
```

同时：

```cpp
recv_range[src_rank] =
    pack2(num_recv_tokens, recv_token_begin_idx);
```

例如：

```text
recv_range[1] =
    pack2(3, 20)
```

逻辑上表示：

```text
source rank 1:
begin = 20
count = 3
```

---

## 14. Copy 到 Packed Buffer

关闭 FP8 时，数据复制路径是：

```cpp
const auto src_data =
    reinterpret_cast<int4*>(
        reinterpret_cast<uint8_t*>(src_src_idx)
        + sizeof(int4));

const auto dst_data =
    recv_x_int4
    + (recv_token_begin_idx + i) * hidden_int4;
```

含义：

```text
src_src_idx
```

指向远端消息开头。

跳过前 16 字节控制字段：

```cpp
src_data = message + 16
```

然后把：

```text
7168 个 BF16
```

以：

```text
896 个 int4
```

为单位复制到 packed buffer。

FP8 分支在当前配置中不会执行。

---

# 15. 返回给 Python 的结果

最终 `low_latency_dispatch` 返回：

```text
packed_recv_x
packed_recv_count
handle
event
hook
```

当前最简配置下：

```text
packed_recv_x.shape = [36, 1024, 7168]
packed_recv_x.dtype = BF16
packed_recv_count.shape = [36]
```

handle 主要包含：

```python
packed_recv_src_info
packed_recv_layout_range
num_max_dispatch_tokens_per_rank
hidden
num_experts
```

这些信息会在后续 `low_latency_combine` 中使用。

---

## 16. 整体数据流

```text
x[128, 7168]
    |
    | 根据 topk_idx[128, 8] 统计 expert
    v
每个 token -> 一个或多个目标 expert
    |
    | IBGDA put
    v
远端 rank 的 rdma_recv_x
    |
    | 依据 rdma_recv_count 等待数据
    v
packed_recv_x[36, 1024, 7168]
    |
    | packed_recv_count/layout_range/src_info
    v
后续 expert 计算
    |
    v
low_latency_combine
```

最简配置的核心原则是：

```text
不做 FP8 转换
不做 LogFMT 压缩
不做 zero-copy
直接以 BF16 消息进行 IBGDA 通信
```

因此单个 dispatch token 的有效通信负载约为：

```text
7168 * 2 = 14336 bytes
```

加上 16 字节控制信息后：

```text
14352 bytes/token
```

这就是该默认测试在关闭优化选项后的基本实现路径。

# Runtime Combine

## low_latency_combine 的作用

`low_latency_combine` 在低延迟 MoE 路径里扮演“把分发后的张量按原始 expert 方案重新合回去”的角色。它位于 buffer.hpp。

从设计上看，它和 `low_latency_dispatch` 是一对：

- `low_latency_dispatch`：把输入按 expert / rank / token 进行分发
- `low_latency_combine`：把这些已分发的结果恢复/合并成最终输出

也就是说，这个函数不是普通的“加法合并”，而是“按照低延迟通信协议中的布局信息，把分散的 expert 结果重新还原为原始 token 视角”。

---

## 1. 函数签名拆解

```cpp
std::tuple<torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>>
low_latency_combine(
    const torch::Tensor& x,
    const torch::Tensor& topk_idx,
    const torch::Tensor& topk_weights,
    const torch::Tensor& src_info,
    const torch::Tensor& layout_range,
    const std::optional<torch::Tensor>& combine_wait_recv_cost_stats,
    int num_max_dispatch_tokens_per_rank,
    int num_experts,
    bool use_logfmt,
    bool zero_copy,
    bool async,
    bool return_recv_hook,
    const std::optional<torch::Tensor>& out = std::nullopt)
```

参数含义：

- `x`
  - 一个已经被分发并按 expert-layout 存放的输入张量
  - 形状通常是 `[num_experts / num_ranks, num_ranks * num_max_dispatch_tokens_per_rank, hidden]`
  - 这表示：每个 local expert 都收到来自各个 rank 的若干 token

- `topk_idx`
  - 每个 token 的 top-k expert 选择结果
  - 在 combine 时用于决定“哪些 expert 的 token 应该被汇总到哪里”

- `topk_weights`
  - top-k 的权重
  - 参与最终组合时的加权合并

- `src_info`
  - 源信息，告诉 combine 过程每个 received token 来自哪一个 rank / 哪个来源位置

- `layout_range`
  - 布局范围信息，用于恢复分发前的 chunk 布局，决定哪些 token 对应哪一段区间

- `combine_wait_recv_cost_stats`
  - 统计 combine 等待接收的开销，可选，用于 debug / profiling

- `num_max_dispatch_tokens_per_rank`
  - 每个 rank 在 dispatch 阶段最多处理的 token 数

- `num_experts`
  - 总专家数

- `use_logfmt`
  - 是否使用 logfmt 形式输出日志/统计

- `zero_copy`
  - 是否采用 zero-copy 路径，减少复制开销

- `async`
  - 是否异步执行

- `return_recv_hook`
  - 是否返回接收阶段 hook，用于延迟触发 recv phase

- `out`
  - 可选输出缓冲区，复用用户提供的 tensor

---

## 2. 前置校验非常严格

函数开头有大量 `EP_HOST_ASSERT`，这是在保护协议正确性：

```cpp
EP_HOST_ASSERT(x.dim() == 3 and x.is_contiguous() and x.scalar_type() == torch::kBFloat16);
EP_HOST_ASSERT(x.size(0) == num_experts / num_ranks);
EP_HOST_ASSERT(x.size(1) == num_ranks * num_max_dispatch_tokens_per_rank);
EP_HOST_ASSERT(x.size(2) % sizeof(int4) == 0 and x.size(2) % 128 == 0);
```

这些条件大致说明：

- `x` 必须是连续的 BF16 tensor
- 每个 local expert 的维度必须匹配
- hidden 维必须满足 int4 对齐和 128 倍对齐
- 这和 CUDA kernel 的 TMA / vectorized 访存要求一致

后面的断言还校验：

- `topk_idx` 和 `topk_weights` 维度和大小一致
- `src_info` 是 int32 且与 expert 分组维度匹配
- `layout_range` 是 int64 且 shape 对应 `[num_experts / num_ranks, num_ranks]`

这说明这个 combine 不是“纯数据搬运”，而是“强约束下的 structured combine”，需要严格保证布局一致。

---

## 3. low_latency Mode 的 Buffer 语义

```cpp
LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
EP_HOST_ASSERT(layout.total_bytes <= num_rdma_bytes);
auto buffer = layout.buffers[low_latency_buffer_idx];
auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];
```

这里构造了 `LowLatencyLayout`，本质上是对低延迟通信 buffer 的“布局描述对象”。

它会根据：

- RDMA buffer 起始地址
- 每 rank 最大 dispatch token 数
- hidden 大小
- rank 数
- expert 数

把共享缓冲区切成若干个 buffer segment，例如：

- `combine_rdma_recv_data_buffer`
- `combine_rdma_recv_flag_buffer`
- `combine_rdma_send_buffer`
- 以及各类 meta / flag 区

`low_latency_buffer_idx` 是当前活跃 buffer 的 index，`low_latency_buffer_idx ^= 1` 用来切换到 next buffer，形成 ping-pong / 双缓冲模式。

这也是低延迟设计的关键：

“每轮使用一块 buffer，下一轮切换另一块，避免覆盖上一轮尚未消费完的数据”。

---

## 4. 流同步逻辑

```cpp
auto compute_stream = at::cuda::getCurrentCUDAStream();
auto launch_stream = return_recv_hook ? compute_stream : comm_stream;
EP_HOST_ASSERT(not(async and return_recv_hook));
if (not return_recv_hook)
    stream_wait(launch_stream, compute_stream);
```

这里的意思是：

- 如果不需要 recv hook，则用 `comm_stream` 来发起 launch
- 如果需要 recv hook，则使用 compute stream
- 这个设计是为了让用户可在某个时刻“手动触发 recv phase”
- 但 `async && return_recv_hook` 被禁止，是因为这两者语义冲突

`stream_wait(launch_stream, compute_stream)` 的作用是：

- 让 communication stream 等待 compute stream 上已有的操作完成
- 这样确保上轮计算和当前 combine 之间不会乱序

这就是低延迟通信最典型的“先等待再发起”的串行约束。

---

## 5. 输出张量的分配

```cpp
torch::Tensor combined_x;
if (out.has_value()) {
    EP_HOST_ASSERT(out->dim() == 2 and out->is_contiguous());
    EP_HOST_ASSERT(out->size(0) == num_combined_tokens and out->size(1) == hidden);
    EP_HOST_ASSERT(out->scalar_type() == x.scalar_type());
    combined_x = out.value();
} else {
    combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
}
```

这里的 output 形状通常是：

- `[num_combined_tokens, hidden]`

`num_combined_tokens` 由 `topk_weights.size(0)` 计算得出，也就是 combine 前后 token 数目。

它不是简单地等于原始输入 token 数，而是按 top-k 的合并结果确定的。

如果用户提供了 `out`，则复用该输出 buffer，减少一次分配。

否则就新建一个 tensor。

---

## 6. Kernel launch：真正的重组逻辑

```cpp
auto next_clean_meta = next_buffer.clean_meta();
auto launcher = [=, this](int phases) {
    internode_ll::combine(combined_x.data_ptr(),
                          buffer.combine_rdma_recv_data_buffer,
                          buffer.combine_rdma_recv_flag_buffer,
                          buffer.combine_rdma_send_buffer,
                          x.data_ptr(),
                          topk_idx.data_ptr<topk_idx_t>(),
                          topk_weights.data_ptr<float>(),
                          src_info.data_ptr<int>(),
                          layout_range.data_ptr<int64_t>(),
                          mask_buffer_ptr,
                          combine_wait_recv_cost_stats.has_value() ? combine_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,
                          next_clean_meta.first,
                          next_clean_meta.second,
                          num_combined_tokens,
                          hidden,
                          num_max_dispatch_tokens_per_rank,
                          num_topk,
                          num_experts,
                          rank,
                          num_ranks,
                          use_logfmt,
                          workspace,
                          num_device_sms,
                          launch_stream,
                          phases,
                          zero_copy);
};
```

这里最关键的是：

- 调用 `internode_ll::combine(...)`
- 传入所有“已经分发好的中间 buffer”
- 以及“重构所需的 metadata”

### 各个参数的职责

- `combined_x.data_ptr()`
  - 输出结果缓冲区

- `buffer.combine_rdma_recv_data_buffer`
  - 收到的 RDMA 数据缓冲区

- `buffer.combine_rdma_recv_flag_buffer`
  - 接收标志缓冲区，用于判断哪个 token / 哪个块已收到

- `buffer.combine_rdma_send_buffer`
  - 发送侧 buffer，参与 combine 交换

- `x.data_ptr()`
  - 已分发状态的输入数据块

- `topk_idx.data_ptr<topk_idx_t>()`
  - TOP-K expert 索引

- `topk_weights.data_ptr<float>()`
  - 权重

- `src_info.data_ptr<int>()`
  - 源地址/源 rank 信息

- `layout_range.data_ptr<int64_t>()`
  - 提供 chunk/布局范围

- `mask_buffer_ptr`
  - shrink / mask 相关控制字段

- `next_clean_meta`
  - 下一个 buffer 的清理元信息，用于避免 stale 数据污染

- `phases`
  - 表示执行 send phase / recv phase / 两者都执行

---

## 7. Phases 的含义：send + Recv 两阶段

```cpp
launcher(return_recv_hook ? LEGACY_LOW_LATENCY_SEND_PHASE :
                            (LEGACY_LOW_LATENCY_SEND_PHASE | LEGACY_LOW_LATENCY_RECV_PHASE));
```

这里的 `phases` 是位掩码：

- `LEGACY_LOW_LATENCY_SEND_PHASE`
  - 发送 phase
- `LEGACY_LOW_LATENCY_RECV_PHASE`
  - 接收 phase

如果 `return_recv_hook == true`，说明这次只发起 send phase，recv phase 由用户后续手动触发；否则一次性把两阶段一起执行。

这是一种经典的“异步接收回调”设计：

- 用户先提交 send / 预取逻辑
- 之后再在合适时机触发 receive
- 这样可以把通信和计算重叠起来，减少等待延迟

---

## 8. Async / Event / recv_hook 语义

```cpp
std::optional<EventHandle> event;
if (async) {
    event = EventHandle(launch_stream);
} else if (not return_recv_hook) {
    stream_wait(compute_stream, launch_stream);
}
```

这段非常重要：它决定了 combine 是否是异步的。

### Async == True

- 返回一个 `EventHandle`
- 允许外部用它来同步
- 同时保留所有张量生命周期，避免异步执行时对象被提前销毁

### Async == False

- 直接 `stream_wait(compute_stream, launch_stream)`
- 也就是把 communication stream 和 compute stream 同步起来

### return_recv_hook == True

- 不在这里做同步等待
- 因为接收阶段是显式回调触发的

---

## 9. Receiver Hook

```cpp
std::optional<std::function<void()>> recv_hook = std::nullopt;
if (return_recv_hook)
    recv_hook = [=]() { launcher(LEGACY_LOW_LATENCY_RECV_PHASE); };
```

这里返回一个 `recv_hook`，它是一个闭包：

- 当调用这个函数时
- 只执行 `LEGACY_LOW_LATENCY_RECV_PHASE`

它的优点是：

- 允许上层在更合适的时机再接收数据
- 让通信和计算更灵活地 overlap
- 这是低延迟路径非常典型的“hook-based pipeline”

---

## 10. 返回值有哪些

```cpp
return {combined_x, event, recv_hook};
```

返回值是三元组：

1. `combined_x`
   - 最终合并后的输出张量

2. `event`
   - 可选的异步同步事件，用于 stream / event 交互

3. `recv_hook`
   - 可选的回调，用于延迟触发接收 phase

也就是说，函数既能：

- 直接返回结果
- 也能给你一个异步事件
- 也能给你一个接收 hook

---

## 11. 它和 low_latency_dispatch 的配合关系

最关键的是理解它们是一套 pipeline：

1. `low_latency_dispatch`
   - 输入 `x` 和 `topk_idx`
   - 先把 token 按 expert / rank 分发到各个 expert buffer
   - 产出 packed_x, packed_src_info, layout_range 等

2. `low_latency_combine`
   - 接收这些已分发数据
   - 结合 `topk_weights`, `src_info`, `layout_range`
   - 还原最终 combine 后的输出

可以抽象为：

- dispatch: “把一个大 tensor 拆成很多专家维度的数据块”
- combine: “从这些数据块中按照 top-k 和源信息重新组装”

它本质上是低延迟 MoE 的“逆操作”过程。

---

## 12. 一句话总结

`low_latency_combine` 的核心任务是：

> 在低延迟通信模式下，利用 RDMA/NVLink 的双缓冲布局和原始 top-k/source metadata，把已经被分发到专家组上的 token 重新合并回最终输出 tensor，并提供异步 / hook / event 机制来最大化通信 - 计算重叠。

我会按“默认测试配置 + 关键参数取 false”的实际执行路径，剥离掉所有不走的分支和断言，直接追踪 `low_latency_combine` 在 C++ 侧的逐行执行序列。

## 实际执行路径：`self.runtime.low_latency_combine` 在“默认测试配置 + 多个特性取 false”时的 C++ 侧执行流

我按如下默认测试语义来理解这条路径：

- `async = false`
- `return_recv_hook = false`
- `use_logfmt = false`
- `zero_copy = false`
- `combine_wait_recv_cost_stats = None`
- `out = None`
- `low_latency_mode` 已经处于开启状态
- 进入的是低延迟 combine 的正常双缓冲路径，不走 callback / delayed recv / async 事件分支

函数最终返回：

```cpp
return {combined_x, event, recv_hook};
```

在你的默认测试配置下，最终值实际上等于：

```cpp
{ combined_x, nullopt, nullopt }
```

即：

- `combined_x`：最终合并后的张量
- `event`：没有，因为 `async=false`
- `recv_hook`：没有，因为 `return_recv_hook=false`

按你给的默认测试配置，`low_latency_combine` 的 C++ 实际执行顺序可以归纳成这个伪代码：

```cpp
EP_HOST_ASSERT(low_latency_mode);

layout = LowLatencyLayout(...);
buffer = layout.buffers[low_latency_buffer_idx];
next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];

compute_stream = cudaGetCurrentCUDAStream();
launch_stream = comm_stream;

stream_wait(launch_stream, compute_stream);

combined_x = torch::empty({num_combined_tokens, hidden}, x.options());

next_clean_meta = next_buffer.clean_meta();

launcher = [&](int phases) {
    internode_ll::combine(
        combined_x.data_ptr(),
        buffer.combine_rdma_recv_data_buffer,
        buffer.combine_rdma_recv_flag_buffer,
        buffer.combine_rdma_send_buffer,
        x.data_ptr(),
        topk_idx.data_ptr<topk_idx_t>(),
        topk_weights.data_ptr<float>(),
        src_info.data_ptr<int>(),
        layout_range.data_ptr<int64_t>(),
        mask_buffer_ptr,
        nullptr,
        next_clean_meta.first,
        next_clean_meta.second,
        num_combined_tokens,
        hidden,
        num_max_dispatch_tokens_per_rank,
        num_topk,
        num_experts,
        rank,
        num_ranks,
        false,  // use_logfmt
        workspace,
        num_device_sms,
        launch_stream,
        LEGACY_LOW_LATENCY_SEND_PHASE | LEGACY_LOW_LATENCY_RECV_PHASE,
        false   // zero_copy
    );
};

launcher(...);

stream_wait(compute_stream, launch_stream);

return {combined_x, nullopt, nullopt};
```

# Launch

## 这段代码的真正含义

这段是外层的 `combine` 调度函数，定义在 internode_ll.cu。它不是直接做“数值计算”，而是：

1. 先算出这次 kernel 需要的 SM / warp / shared-memory 规模；
2. 再根据 `hidden` 选择编译期模板实例，比如 `combine<true, 512, 11, 4>`；
3. 最后发起真正的 CUDA kernel launch。

也就是说，这一层负责“部署和调度”，真正的合并逻辑在模板化的 `combine<...>` 里。

---

## 1) 入口参数：每个变量的作用

```cpp
void combine(void* combined_x,
             void* rdma_recv_x,
             int* rdma_recv_flag,
             void* rdma_send_x,
             const void* x,
             const topk_idx_t* topk_idx,
             const float* topk_weights,
             const int* src_info,
             const int64_t* layout_range,
             int* mask_buffer_ptr,
             int64_t* combine_wait_recv_cost_stats,
             int* next_clean,
             int num_next_clean_int,
             int num_combined_tokens,
             int hidden,
             int num_max_dispatch_tokens_per_rank,
             int num_topk,
             int num_experts,
             int rank,
             int num_ranks,
             bool use_logfmt,
             void* workspace,
             int num_device_sms,
             cudaStream_t stream,
             int phases,
             bool zero_copy)
```

逐项解释：

- `combined_x`：输出结果，最终合并后的 token 数据
- `rdma_recv_x`：远端接收到的数据缓冲区
- `rdma_recv_flag`：接收完成标志，表示某个 expert 的数据已经收到
- `rdma_send_x`：本地发出去的数据缓冲区
- `x`：当前 rank 本地的输入张量
- `topk_idx`：每个 token 的 top-k expert 索引，形状通常是 `[num_combined_tokens, num_topk]`
- `topk_weights`：对应 top-k 的权重
- `src_info`：每个 token 在 source 侧的 index / 位置
- `layout_range`：每个 expert 的布局信息，比如起始偏移和数量
- `mask_buffer_ptr`：mask / timeout 相关的 rank 屏蔽表
- `combine_wait_recv_cost_stats`：接收等待耗时统计
- `next_clean`：下一轮清理缓存
- `num_next_clean_int`：`next_clean` 的长度
- `num_combined_tokens`：本次需要合并的 token 数
- `hidden`：每个 token 的 hidden size
- `num_max_dispatch_tokens_per_rank`：每个 rank 最多参与的 token 数
- `num_topk`：每个 token 取前 k 个 expert
- `num_experts`：总 expert 数
- `rank`：当前 rank
- `num_ranks`：总 rank 数
- `use_logfmt`：是否启用 LogFMT 压缩
- `workspace`：临时工作区，比如清理旗标
- `num_device_sms`：当前 GPU 上可用的 SM 数
- `stream`：CUDA stream
- `phases`：当前执行 send / recv 哪个阶段
- `zero_copy`：是否采用零拷贝

---

## 2) 第 1 步：计算 Warp 分组

```cpp
constexpr int kNumMaxTopk = 11;
const int num_warp_groups = ceil_div(num_experts, num_device_sms);
const int num_warps_per_group = 32 / num_warp_groups;
const int num_recv_per_sm = ceil_div(num_combined_tokens, num_device_sms);
EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0 and num_recv_per_sm >= 0);
```

这里的核心思想是：

- 把 `num_experts` 按 SM 划分：
  - `num_warp_groups = ceil_div(num_experts, num_device_sms)`
- 每个 warp group 里有多少个 warp：
  - `num_warps_per_group = 32 / num_warp_groups`
- 每个 SM 平均要处理多少个 combined token：
  - `num_recv_per_sm = ceil_div(num_combined_tokens, num_device_sms)`

### 用例子算一遍

假设参数是：

- `num_experts = 8`
- `num_device_sms = 4`
- `num_combined_tokens = 6`

则：

- `num_warp_groups = ceil_div(8, 4) = 2`
- `num_warps_per_group = 32 / 2 = 16`
- `num_recv_per_sm = ceil_div(6, 4) = 2`

所以：

- 一共 2 个 warp group
- 每个 group 16 个 warp
- 一个 SM 平均处理 2 个 token

---

## 3) 第 2 步：计算总 Launch 数量

```cpp
const auto num_warps = num_warp_groups * num_warps_per_group;
const auto num_sms =
    max(ceil_div(num_experts, num_warp_groups), num_recv_per_sm == 0 ? 1 : ceil_div(num_combined_tokens, num_recv_per_sm));
```

这行其实是在选最终 launch 的 SM 数。

- `num_warps = num_warp_groups * num_warps_per_group`
- `num_sms` 取两个量中的较大者：
  - `ceil_div(num_experts, num_warp_groups)`
  - `ceil_div(num_combined_tokens, num_recv_per_sm)`

也就是：

- 先按 expert 维度切；
- 再按 token 数量维度切；
- 取更大值，避免某一维不够。

### 例子

继续上面的例子：

- `num_warps = 2 * 16 = 32`
- `ceil_div(8, 2) = 4`
- `ceil_div(6, 2) = 3`

所以：

- `num_sms = max(4, 3) = 4`

这里就说明：最终会起 4 个 SM 去并发处理。

---

## 4) 第 3 步：workspace 和约束检查

```cpp
auto atomic_clean_flag = static_cast<int*>(workspace);
EP_HOST_ASSERT(sizeof(int) <= LEGACY_NUM_WORKSPACE_BYTES);
EP_HOST_ASSERT(num_topk <= kNumMaxTopk);
```

这一段是安全检查：

- `workspace` 被当作 `int*`，也就是临时标志位；
- `sizeof(int) <= LEGACY_NUM_WORKSPACE_BYTES`：要求工作区足够大；
- `num_topk <= 11`：因为 `kNumMaxTopk = 11`，这是上限；
- 也就是说，这个 kernel 不是无限制地支持任意 `topk`，而是要求 `topk <= 11`。

### 例子

如果：

- `num_topk = 3`

那么满足：

- `3 <= 11` ✅

---

## 5) 第 4 步：零拷贝和 LogFMT 互斥

```cpp
EP_HOST_ASSERT(not(zero_copy and use_logfmt));
```

这是一个非常关键的约束：

- `zero_copy == true` 且 `use_logfmt == true` 不允许同时成立；
- 原因是 online cast / LogFMT 压缩时，需要改写和重排数据，不能完全零拷贝。

### 例子

- `zero_copy = false, use_logfmt = true`：允许
- `zero_copy = true, use_logfmt = true`：直接断言失败

---

## 6) 第 5 步：共享内存大小估算

```cpp
constexpr int kNumStages = 3;
constexpr int kNumMaxUnrolls = 4;
constexpr int kMaxNumGroups = 2;

// Send buffer size
const int num_meta_bytes = hidden / 128 * 4;
const int num_send_tma_bytes = 32 * sizeof(int4) * kNumMaxUnrolls + 16;
const int smem_send_size = num_warps * (kNumStages * num_send_tma_bytes + num_meta_bytes);

// Receive buffer size
const int num_recv_tma_bytes = 16 + hidden * 2;
const int smem_recv_size = kMaxNumGroups * (kNumStages * num_recv_tma_bytes + hidden * 2 + kNumStages * num_meta_bytes * 3);

// Total requirement
const int smem_size = max(smem_send_size, smem_recv_size);
```

这一段在估算 shared memory 需求，目的是确保 kernel 启动前有足够的共享内存。

### 解释

#### 发送侧

```cpp
num_meta_bytes = hidden / 128 * 4;
```

这里含义是：每 128 维一个 mini header，4 bytes 统计元信息。

#### 接收侧

```cpp
num_recv_tma_bytes = 16 + hidden * 2;
```

说明每个接收 slot 需要：

- 16 bytes 头部控制信息
- `hidden * 2` bytes 的数据存储（因为 BF16 2 bytes/值）

### 具体例子

设：

- `hidden = 512`
- `num_warps = 32`

则：

- `num_meta_bytes = 512 / 128 * 4 = 4 * 4 = 16`
- `num_send_tma_bytes = 32 * sizeof(int4) * 4 + 16`
- `sizeof(int4) = 16`
- 所以：
  - `num_send_tma_bytes = 32 * 16 * 4 + 16 = 2048 + 16 = 2064`

发送 shared memory：

- `smem_send_size = 32 * (3 * 2064 + 16)`
- `= 32 * (6192 + 16)`
- `= 32 * 6208 = 198656`

接收 shared memory：

- `num_recv_tma_bytes = 16 + 512 * 2 = 16 + 1024 = 1040`
- `smem_recv_size = 2 * (3 * 1040 + 512 * 2 + 3 * 16 * 3)`
- `= 2 * (3120 + 1024 + 144)`
- `= 2 * 4288 = 8576`

最终：

- `smem_size = max(198656, 8576) = 198656`

说明发侧 shared memory 更大，所以它决定了 total shared memory 需求。

---

## 7) 第 6 步：真正 Launch 一个模板化的 Kernel

```cpp
#define COMBINE_LAUNCH_CASE(hidden)                                                                                                \
    {                                                                                                                              \
        auto combine_func =                                                                                                        \
            use_logfmt ? combine<true, hidden, kNumMaxTopk, kNumMaxUnrolls> : combine<false, hidden, kNumMaxTopk, kNumMaxUnrolls>; \
        SET_SHARED_MEMORY_FOR_TMA(combine_func);                                                                                   \
        LAUNCH_KERNEL(&cfg,                                                                                                        \
                      combine_func,                                                                                                \
                      combined_x,                                                                                                  \
                      rdma_recv_x,                                                                                                 \
                      rdma_recv_flag,                                                                                              \
                      rdma_send_x,                                                                                                 \
                      x,                                                                                                           \
                      topk_idx,                                                                                                    \
                      topk_weights,                                                                                                \
                      src_info,                                                                                                    \
                      layout_range,                                                                                                \
                      mask_buffer_ptr,                                                                                             \
                      combine_wait_recv_cost_stats,                                                                                \
                      next_clean,                                                                                                  \
                      num_next_clean_int,                                                                                          \
                      atomic_clean_flag,                                                                                           \
                      num_combined_tokens,                                                                                         \
                      hidden,                                                                                                      \
                      num_topk,                                                                                                    \
                      num_max_dispatch_tokens_per_rank,                                                                            \
                      num_experts,                                                                                                 \
                      rank,                                                                                                        \
                      num_ranks,                                                                                                   \
                      num_warp_groups,                                                                                             \
                      num_warps_per_group,                                                                                         \
                      phases,                                                                                                      \
                      zero_copy);                                                                                                  \
    }                                                                                                                              \
    break
```

这一段非常关键：它在根据 `hidden` 生成真正的 kernel 实例。

### 这行代码的意思

```cpp
use_logfmt ? combine<true, hidden, kNumMaxTopk, kNumMaxUnrolls>
          : combine<false, hidden, kNumMaxTopk, kNumMaxUnrolls>;
```

表示：

- 若 `use_logfmt == true`：用 `combine<true, hidden, 11, 4>` 实例化
- 若 `use_logfmt == false`：用 `combine<false, hidden, 11, 4>` 实例化

其中：

- `true/false`：是否使用 LogFMT 压缩
- `hidden`：编译期已知的 hidden size
- `kNumMaxTopk = 11`：最大 top-k
- `kNumMaxUnrolls = 4`：最大 unroll 级别

---

## 8) 最后发起 Kernel

```cpp
SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
SWITCH_HIDDEN(COMBINE_LAUNCH_CASE);
#undef COMBINE_LAUNCH_CASE
```

这表示：

- grid 大小 = `num_sms`
- 每个 block 线程数 = `num_warps * 32`

### 在例子中

如果：

- `num_sms = 4`
- `num_warps = 32`

那么最终启动是：

- `grid = (4, 1, 1)`
- `block = (1024, 1, 1)`

也就是 4 个 SM，每个 SM 一个 1024-thread block。

---

## 9) 一次完整例子的“最终值”

我们用一个完整参数例子：

```cpp
num_experts = 8
num_device_sms = 4
num_combined_tokens = 6
hidden = 512
num_topk = 3
num_max_dispatch_tokens_per_rank = 8
num_experts = 8
rank = 0
num_ranks = 2
use_logfmt = true
zero_copy = false
phases = SEND | RECV
```

那么最终是：

- `kNumMaxTopk = 11`
- `num_warp_groups = ceil_div(8, 4) = 2`
- `num_warps_per_group = 32 / 2 = 16`
- `num_recv_per_sm = ceil_div(6, 4) = 2`
- `num_warps = 2 * 16 = 32`
- `num_sms = max(ceil_div(8,2)=4, ceil_div(6,2)=3) = 4`

shared memory 估算：

- `num_meta_bytes = 512 / 128 * 4 = 16`
- `num_send_tma_bytes = 32 * 16 * 4 + 16 = 2064`
- `smem_send_size = 32 * (3 * 2064 + 16) = 198656`
- `num_recv_tma_bytes = 16 + 512 * 2 = 1040`
- `smem_recv_size = 2 * (3 * 1040 + 512 * 2 + 3 * 16 * 3) = 8576`
- `smem_size = max(198656, 8576) = 198656`

最终 kernel 实例：

```cpp
combine<true, 512, 11, 4>
```

并且：

```cpp
SETUP_LAUNCH_CONFIG(4, 32 * 32, stream);
```

相当于：

- gridDim.x = 4
- blockDim.x = 1024

---

## 10) 总结一句话

这个外层 `combine` 函数做的事情，本质上就是：

- 根据专家数、token 数、SM 数、hidden size 计算并发规模；
- 申请/检查 shared memory；
- 选定真正的模板化 kernel；
- 调用 `combine<...>` 进行 top-k 聚合和 RDMA/LogFMT 传输。

如果你愿意，我可以继续把这段代码按“实际调用路径”再展开成：

- 一个“参数流向图”
- 或者按 “调用产生的 kernel launch 实例” 画出一张时序示意图。
