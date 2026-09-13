# Combine Low Latency Kernel

这个函数是低延迟 MoE 的“重组内核”，它负责把已经按 expert/rank 分发好的数据重新组合回最终输出 `combined_x`。

它既有“发送阶段”（把数据发到目标 rank / expert），也有“接收阶段”（等待远端完成并把多个 top-k 候选做加权合并）。

它的核心思想可以概括成一句话：

> 接收端在 `rdma_recv_x` 中拿到各 expert 的 payload，然后按 `topk_idx` 和 `topk_weights` 还原出每个 token 的最终结果，写回 `combined_x`。

---

## 1. 模板参数和 Launch 限制

```cpp
template <bool kUseLogFMT, int kHidden, int kNumMaxTopk, int kNumMaxUnrolls>
__global__ __launch_bounds__(1024, 1) void combine(...)
```

这几部分的含义是：

- `kUseLogFMT`
  - 是否启用 LogFMT 压缩格式
  - 如果为 true，数据在发送前会做 LogFMT 编码，接收端再解码
- `kHidden`
  - 当前 hidden size
  - 例如 4096 / 8192 等
- `kNumMaxTopk`
  - 最大 top-k 数
  - 这里的上限在上层代码中被限制为 11
- `kNumMaxUnrolls`
  - 开发时用于 TMA / 向量化的循环展开上限
- `__global__`
  - GPU kernel
- `__launch_bounds__(1024, 1)`
  - 一个 block 最多 1024 线程；occupancy 由 1 控制

---

## 2. 函数参数：每个参数都对应一类共享状态

```cpp
void* combined_x,
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
int* atomic_clean_flag,
int num_combined_tokens,
int hidden,
int num_topk,
int num_max_dispatch_tokens_per_rank,
int num_experts,
int rank,
int num_ranks,
int num_warp_groups,
int num_warps_per_group,
int phases,
bool zero_copy
```

逐个理解：

- `combined_x`
  - 最终输出，写回这里
- `rdma_recv_x`
  - 远端收到的数据缓冲区
- `rdma_recv_flag`
  - 每个 expert 是否已经收到远端数据完成的标志
- `rdma_send_x`
  - 发送端缓存，准备发给远端的数据
- `x`
  - 当前 rank 本地已经分发好的 packed 数据
- `topk_idx`
  - 每个 token 的 top-k expert 索引
- `topk_weights`
  - 每个 token 对应 expert 的权重
- `src_info`
  - 每个 token 的来源信息，说明它来自哪个 rank / 位置
- `layout_range`
  - 每个 expert / rank 的 token 布局区间
- `mask_buffer_ptr`
  - timeout 或失败 rank 的 mask 缓冲区
- `combine_wait_recv_cost_stats`
  - 接收等待时间统计
- `next_clean`
  - 下一轮 buffer 清理信息
- `num_next_clean_int`
  - 下一轮 buffer 清理长度
- `atomic_clean_flag`
  - 清理标志，用于同步 buffer reuse
- `num_combined_tokens`
  - 最终要 combine 的 token 数
- `hidden`
  - hidden size
- `num_topk`
  - top-k 数
- `num_max_dispatch_tokens_per_rank`
  - 每个 rank 上最大分发 token 数
- `num_experts`
  - 总 expert 数
- `rank`
  - 当前 rank
- `num_ranks`
  - rank 总数
- `num_warp_groups`
  - 每个 SM 分成多少 warp groups
- `num_warps_per_group`
  - 每个 warp group 内有几个 warps
- `phases`
  - 当前执行的是哪个阶段：send / recv / both
- `zero_copy`
  - 是否启用 zero-copy 发送路径

---

## 3. 线程和组的划分

假设我们取一个最简单但足够真实的例子：

参数	值

num_experts	8

num_ranks	2

rank	0

hidden	512

num_topk	3

num_combined_tokens	6

num_max_dispatch_tokens_per_rank	8

num_device_sms	4

phases	SEND + RECV 两阶段同时开启

zero_copy	false

use_logfmt	true

则：

num_local_experts = 8 / 2 = 4

num_warp_groups = ceil_div(8, 4) = 2

num_warps_per_group = 32 / 2 = 16

num_warps = 2 * 16 = 32

如果当前 block 是 blockIdx.x = 1，那么：

sm_id = 1

warp_group_id 在 0..1 中间变化

例如 warp_id = 7：

warp_group_id = 7 / 16 = 0

sub_warp_id = 7 % 16 = 7

responsible_expert_idx = 1 * 2 + 0 = 2

接着：

dst_rank = 2 / 4 = 0

local_expert_idx = 2 % 4 = 2

global_expert_idx = rank * num_local_experts + local_expert_idx = 0 * 4 + 2 = 2

意思是：

当前 SM 上第 1 个 block 的第 7 个 warp，负责处理的“全局 expert”是 2；

它对应到本地 rank 0 上的 expert 2；

发送目标 rank 是 0（本地），或者在其他组合中会按 responsible_expert_idx 变成远端 rank。

```cpp
const auto num_local_experts = num_experts / num_ranks;
const auto warp_group_id = warp_id / num_warps_per_group;
const auto sub_warp_id = warp_id % num_warps_per_group;
const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;
```

这里是 GPU 内核里最关键的“分工”代码。

重点解释：

- `sm_id`
  - 当前 SM 的编号
- `num_sms`
  - 总共使用多少个 SM
- `thread_id`
  - 当前线程 id
- `warp_id / lane_id`
  - warp 和 lane 级别的线程索引
- `num_local_experts = num_experts / num_ranks`
  - 每个 rank 有多少个 local expert
- `warp_group_id`
  - 这个 warp 属于哪个 group
- `sub_warp_id`
  - 同一 warp group 内的子 warp 编号
- `responsible_expert_idx`
  - 当前 warp group 负责处理哪个 expert

这一层逻辑非常重要：

它让不同 SM / warp group 分工处理不同 expert，而不是让所有线程都做同样的事情。

---

## 4. Shared Memory 只是一个大中转缓冲区

```cpp
extern __shared__ __align__(1024) uint8_t smem_buffer[];
```

`smem_buffer` 是一个共享内存区域，用来临时存放：

- TMA load buffer
- TMA store buffer
- mbarrier
- 编码/解码 metadata
- `LogFMT` 相关的 amax/amin 统计

---

## 5. 数据类型与消息布局

```cpp
constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
constexpr int64_t hidden_bf16_int4 = kHidden / kNumElemsPerInt4;
```

这是把 hidden 维映射成 int4 单元数量的计算。

因为 BF16 与 int4 之间有固定关系：

- `sizeof(int4) = 16 bytes`
- `sizeof(nv_bfloat16) = 2 bytes`
- 所以一个 `int4` 里能装 8 个 bf16 元素

---

```cpp
constexpr int kNumSendUnrolls = kHidden % (32 * 4 * sizeof(int4) / sizeof(nv_bfloat16)) == 0 ? 4 : 2;
constexpr int kNumRecvUnrolls = 2;
constexpr int hidden_bf16_int4_pad = align_up(static_cast<int>(hidden_bf16_int4), 32 * kNumSendUnrolls);
```

这在设计上区分了：

- send stage 的 unroll 因子
- recv stage 的 unroll 因子

为什么不一样？

- send 阶段更偏 TMA 访存和大块复制
- recv 阶段更偏按 top-k 还原与累加

这里的 `padding` 让 hidden size 对齐，避免尾部处理复杂。

---

```cpp
constexpr int kNumDivisions = kHidden / 128;
constexpr int kNumMetaBytes = kNumDivisions * sizeof(nv_bfloat162);
constexpr size_t num_bytes_per_slot = kHidden * sizeof(nv_bfloat16) + kNumMetaBytes;
EP_STATIC_ASSERT(num_bytes_per_slot % sizeof(int4) == 0, "Invalid vectorization");
```

这里定义了“一条消息”的大小：

- `kHidden * sizeof(nv_bfloat16)`：原始数据本体
- `kNumMetaBytes`：每 128 个通道一段元信息，主要用于 LogFMT 的 min/max 等辅助信息

所以一个消息 slot 的总大小就是：

- 真正的 hidden data
- 再加上一个小的 metadata 区

这为低延迟通信提供了“带压缩状态的 payload”布局。

---

## 6. Send phase：把本地结果发给目标 Expert

```cpp
if ((phases & LEGACY_LOW_LATENCY_SEND_PHASE) == 0)
    goto LOW_LATENCY_COMBINE_RECV;
```

这说明如果当前不需要 send phase，则直接进入 recv phase。

在默认同步配置里，两阶段通常都执行，因此不会跳过。

---

### 6.1 清理 next Buffer

```cpp
if (sm_id == 0 and warp_group_id == 0 and sub_warp_id == 0) {
    for (int i = lane_id; i < num_next_clean_int; i += 32)
        next_clean[i] = 0;

    __syncwarp();
    if (lane_id == 0)
        atomic_add_release_global(atomic_clean_flag, num_experts);
}
```

这段做的事是：

- 清理 next buffer
- 让后续发送/接收不会看到旧数据残留
- 对 `atomic_clean_flag` 做加法，表示当前轮开始进入清理状态

这里本质上是双缓冲轮换的“准备阶段”。

---

### 6.2 负责处理的 Expert

```cpp
if (responsible_expert_idx < num_experts) {
    const auto dst_rank = responsible_expert_idx / num_local_experts;
    const auto local_expert_idx = responsible_expert_idx % num_local_experts;
    const auto global_expert_idx = rank * num_local_experts + local_expert_idx;
    const auto layout = __ldg(layout_range + local_expert_idx * num_ranks + dst_rank);
    const auto local_x =
        static_cast<const int4*>(x) + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_bf16_int4;
    const auto local_src_info = src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
    const auto rdma_send_x_vec =
        static_cast<uint8_t*>(rdma_send_x) + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_slot;
```

这里是每个部分最关键的“定位”代码：

- `dst_rank`
  - 当前 expert 的目标 rank
- `local_expert_idx`
  - 当前 rank 内部的 expert 索引
- `global_expert_idx`
  - 全局 expert 索引
- `layout`
  - 当前 expert 对应的 token 区间
- `local_x`
  - 当前 expert 对应的本地数据起点
- `local_src_info`
  - 该 expert 对应的 source info 起点
- `rdma_send_x_vec`
  - 当前 expert 对应的 send buffer 起点

它把“本地 expert 中的数据”映射到了“目标 rank 的存储位置”。

---

### 6.3 把 Layout 解码成 Offset / Count

```cpp
int offset, num_tokens_to_send;
unpack2(layout, num_tokens_to_send, offset);
```

这个 `layout` 是一个压缩起来的整数 pair，解出来就是：

- `offset`：这一段数据在 `x` 中的起点
- `num_tokens_to_send`：这一段实际要发多少 token

这一步就是把“packed layout metadata”恢复成真正的区间参数。

---

### 6.4 初始化 TMA Stage 和 Barrier

```cpp
constexpr int kNumTMABufferBytes = sizeof(int4) * 32 * kNumSendUnrolls;
constexpr int kNumStages = 3;
constexpr int kNumPrefetch = 1;
...
auto smem_ptr = smem_buffer + warp_id * (...);
...
if (lane_id < kNumStages) {
    mbarrier_init(full_barriers[lane_id], 1);
    fence_barrier_init();
}
__syncwarp();
```

这里开始使用 TMA 相关的 shared-memory stage 模式。

- `full_barriers`
  - 每个 stage 的 mbarrier
- `tma_load_and_arrive`
  - 从 global memory load data 到 shared memory，并通知 barrier
- `mbarrier_wait`
  - 等待 stage ready

这里是 GPU 高性能 pipeline 的典型手法：

“异步拉取 + barrier 同步 + 继续处理下一段数据”。

---

### 6.5 真正的 Send 行为：copy / Put

最关键的代码在这里：

```cpp
if (not is_rank_masked<true>(mask_buffer_ptr, dst_rank)) {
    for (int token_idx = offset + sub_warp_id; token_idx < offset + num_tokens_to_send; token_idx += num_warps_per_group) {
        ...
        const auto src_idx = __shfl_sync(0xffffffff, __ldg(local_src_info + token_idx), 0);
        const auto buf_ptr = reinterpret_cast<int64_t>(rdma_send_x_vec_row);
        const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) +
            (global_expert_idx * num_max_dispatch_tokens_per_rank + src_idx) * num_bytes_per_slot;
        const auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);

        int num_send_bytes = hidden * sizeof(nv_bfloat16);
        ...
        if (not zero_copy or dst_p2p_ptr != 0) {
            ...
        }

        if (dst_p2p_ptr == 0)
            nvshmemi_ibgda_put_nbi_warp(dst_ptr, buf_ptr, num_send_bytes, dst_rank, local_expert_idx, lane_id, token_idx - offset);
    }
}
```

这段非常关键：它做了三件事情：

1. 找到 token 的 source idx
   - `src_idx = ...`
2. 计算目标地址
   - `dst_ptr = rdma_recv_x + (global_expert_idx * ...) + src_idx * slot`
3. 发消息
   - `nvshmemi_ibgda_put_nbi_warp`

因此，send phase 实际上是：

> 把当前本地 expert 的 token 通过 RDMA/IBGDA 发到目标 rank 指定 expert buffer 中。

---

### 6.6 Zero-copy 分支

```cpp
if (not zero_copy or dst_p2p_ptr != 0) {
    const auto cpy_src_int4_ptr = zero_copy ? reinterpret_cast<int4*>(buf_ptr) : x_int4;
    const auto cpy_dst_int4_ptr =
        dst_p2p_ptr == 0 ? reinterpret_cast<int4*>(buf_ptr) : reinterpret_cast<int4*>(dst_p2p_ptr);
    ...
}
```

它说明：

- `zero_copy = false`：先从 `x` 读入，写到 `rdma_send_x` buffer，再发
- `zero_copy = true`：直接从 send buffer 读、目标地址直接写，不再额外复制

如果 `dst_p2p_ptr == 0`，说明不能直接对端写，就需要先在本地 buffer 暂存，再 call RDMA put。

---

### 6.7 Send Phase 结束后，写 Flag

```cpp
if (sub_warp_id == 1 and lane_id == 0) {
    while (ld_acquire_global(atomic_clean_flag) == 0)
        ;
    auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_flag + global_expert_idx);
    ...
    if (dst_p2p_ptr == 0) {
        nvshmemi_ibgda_amo_nonfetch_add(reinterpret_cast<int*>(dst_ptr), 1, dst_rank, local_expert_idx);
    } else {
        st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr), 1);
    }
    atomic_add_release_global(atomic_clean_flag, -1);
}
```

这表示：

- 当前 expert 的发送任务已经全部完成
- 通过 `rdma_recv_flag` 告诉对端：“这个 expert 的数据已经准备好”

这个 flag 是后续 receive phase 阻塞等待的关键信号。

---

## 7. 接收 phase：等待远端完毕，然后还原最终结果

这个部分从标签开始：

```cpp
LOW_LATENCY_COMBINE_RECV:
    if ((phases & LEGACY_LOW_LATENCY_RECV_PHASE) == 0)
        return;
```

如果当前没有 recv phase，就直接 return。

---

### 7.1 等待远端 `rdma_recv_flag`

```cpp
if (responsible_expert_idx < num_experts) {
    if (sub_warp_id == 0 and lane_id == 0) {
        const auto src_rank = responsible_expert_idx / num_local_experts;
        auto start_time = clock64();
        uint64_t wait_recv_cost = 0;
        if (not is_rank_masked(mask_buffer_ptr, src_rank)) {
            while (ld_acquire_sys_global(rdma_recv_flag + responsible_expert_idx) == 0
                   && (wait_recv_cost = clock64() - start_time) <= LEGACY_NUM_TIMEOUT_CYCLES)
                ;
        }
        ...
    }
}
cg::this_grid().sync();
```

这里在等：

- `rdma_recv_flag[responsible_expert_idx]`
- 也就是对应的 source rank / expert 已经发完

如果一直等不到：

- 超时记录
- 把这个 rank 标记为 mask
- 避免 kernel 卡死

这部分是 combine 鲁棒性的关键。

---

### 7.2 Grid Sync

```cpp
cg::this_grid().sync();
```

这是接收 phase 的关键 fence。

它确保所有 SM 都准备好了，再进入后面的 decode / reduce。

---

## 8. 重组织 warp：decode Warp 和 Reduction Warp

```cpp
constexpr int kMaxNumGroups = 2;
const int num_decode_warps = hidden_bf16_int4_pad / (kNumRecvUnrolls * 32);
const int num_groups = min(kMaxNumGroups, (num_threads / 32) / (num_decode_warps + 1));
const int decode_warp_idx = __shfl_sync(0xffffffff, warp_id % (num_decode_warps + 1), 0);
const int group_idx = __shfl_sync(0xffffffff, warp_id / (num_decode_warps + 1), 0);
```

这里把所有 warp 重新分配为两类：

- decode warp：拉数据并 decode
- reduction warp：扛 reduction / accumulate

这就是“接收端的并行化”。

---

## 9. `group_idx < num_groups` 这段：核心 Combine 逻辑

这里才是真正的“重组阶段”。

### 9.1 先初始化 Shared Memory Stage 和 Barrier

```cpp
if (decode_warp_idx == num_decode_warps)
    tma_phase = (1 << kNumStages) - 1;
...
if (decode_warp_idx == num_decode_warps and lane_id < kNumStages) {
    mbarrier_init(full_barriers[lane_id], 1);
    mbarrier_init(empty_barriers[lane_id], num_decode_warps);
}
asm volatile("bar.sync %0, %1;" ::"r"(group_idx + 1), "r"((num_decode_warps + 1) * 32));
```

这里是把一组 stage synchronization 建立起来，用于：

- TMA load
- decode
- reduce
- store result

---

### 9.2 TMA Load warp：把远端数据拉回本地 Buffer

```cpp
if (decode_warp_idx == num_decode_warps) {
    for (int token_idx = sm_id + num_sms * group_idx; token_idx < num_combined_tokens; token_idx += num_sms * num_groups) {
        if (lane_id < num_topk)
            topk_idx_by_lane = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + lane_id));
        for (int i = 0; i < num_topk; ++i) {
            int topk_idx_reg = __shfl_sync(0xffffffff, topk_idx_by_lane, i);
            if (topk_idx_reg < 0)
                continue;
            if (is_rank_masked(mask_buffer_ptr, topk_idx_reg / num_local_experts))
                continue;

            mbarrier_wait<true>(empty_barriers[stage_idx], tma_phase, stage_idx);
            auto buffer = static_cast<uint8_t*>(rdma_recv_x) +
                (topk_idx_reg * num_max_dispatch_tokens_per_rank + token_idx) * num_bytes_per_slot;
            ...
            tma_load_1d(..., buffer + ..., full_barriers[stage_idx], num_tma_bytes);
            mbarrier_arrive_and_expect_tx(full_barriers[stage_idx], num_tma_bytes);
            ...
            stage_idx = (stage_idx + 1) % kNumStages;
        }
    }
}
```

这里做的是：

- For each token
- Look at its top-k experts
- If expert valid and unmasked
- Pull the data from `rdma_recv_x` into shared memory staging
- Use `mbarrier` to track stage availability

这就是 combine 中“从远端取回 payload”的阶段。

---

### 9.3 Reduction warp：按 Top-k 权重做累积

```cpp
else {
    for (int token_idx = sm_id + num_sms * group_idx; token_idx < num_combined_tokens; token_idx += num_sms * num_groups) {
        if (lane_id < num_topk) {
            topk_idx_by_lane = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + lane_id));
            topk_weights_by_lane = __ldg(topk_weights + token_idx * num_topk + lane_id);
        }
        __syncwarp();

        float combined_values[kNumElemsPerInt4 * kNumRecvUnrolls] = {0.0f};
        for (int i = 0; i < num_topk; ++i) {
            int topk_idx_reg = __shfl_sync(0xffffffff, topk_idx_by_lane, i);
            if (topk_idx_reg < 0)
                continue;
            if (is_rank_masked(mask_buffer_ptr, topk_idx_reg / num_local_experts))
                continue;
            const auto topk_weight = __shfl_sync(0xffffffff, topk_weights_by_lane, i);

            mbarrier_wait<true>(full_barriers[stage_idx], tma_phase, stage_idx);
            ...
            decode_and_accumulate<kNumRecvUnrolls>(..., combined_values, ..., topk_weight);
            ...
            stage_idx = (stage_idx + 1) % kNumStages;
        }
        ...
    }
}
```

这是整个 combine 内核最核心的一段。

它的逻辑是：

- 对每个 token
- 所有 top-k expert 逐个处理
- 如果该 expert 合法
- 把它对应的 data decode 出来
- 再乘上 `topk_weight`
- 累加到 `combined_values`
- 最终得到 token 的最终 combine 结果

所以这一步就是：

> 把不同 expert 的候选结果按 top-k 权重做线性融合。

---

## 10. `decode_and_accumulate` 的逻辑

这个函数是“decode + accumulate”的核心。

它会根据是否开启 LogFMT 做不同处理。

### 10.1 LogFMT 模式

```cpp
if (enable_cast) {
    ...
    decode(...)
} else {
    ...
}
```

如果启用 LogFMT，就会：

- 从编码后的 buffer 里还原 log-scale 值
- 乘以权重
- 累加进 `combined_values`

它本质上是“压缩表示的还原”。

### 10.2 非 LogFMT 模式

```cpp
else {
    for (int k = 0; k < kNumRecvUnrolls * 4; ++k) {
        auto bf16_pack = *reinterpret_cast<__nv_bfloat162*>(ld_buffer + k);
        accum[k * 2 + 0] += static_cast<float>(bf16_pack.x) * weight;
        accum[k * 2 + 1] += static_cast<float>(bf16_pack.y) * weight;
    }
}
```

这里是直接把 BF16 读出并累积。

这表示普通路径下没有压缩，不需要额外 decode 逻辑。

---

## 11. 最终写回 `combined_x`

这段代码最后把 `combined_values` 按 chunk 写回输出：

```cpp
#pragma unroll
for (int k = 0; k < kNumRecvUnrolls * 4; ++k) {
    auto combined_pack = __nv_bfloat162(combined_values[k * 2], combined_values[k * 2 + 1]);
    tma_st_buffers[decode_warp_idx][kNumRecvUnrolls * 4 * lane_id + k] = *reinterpret_cast<uint32_t*>(&combined_pack);
}
...
if (elect_one_sync()) {
    tma_store_1d(tma_st_buffers[decode_warp_idx],
                 static_cast<int4*>(combined_x) + token_idx * hidden_bf16_int4 + decode_warp_idx * kNumRecvUnrolls * 32,
                 kNumBF16PerWarpBytes);
}
```

它的含义是：

- 把每个 token 的 combine 结果从 `combined_values` 取出来
- 重新按 BF16 / int4 形式重排
- 写入 `combined_x[token_idx]`

这就是最终输出。

---

## 12. 一句话总结

这个内核的总流程可以概括成：

1. 给每个 expert 分配处理份额
2. send phase 发送 each expert payload 到目标 rank
3. recv phase 等待远端 flag
4. 按 `topk_idx` 遍历候选 expert
5. 读取它们的 payload
6. 如果启用 LogFMT 就 decode，否则直接 BF16 读取
7. 乘上 `topk_weights`
8. 累加到 `combined_values`
9. 最终写回 `combined_x`

也就是说，它不是简单“复制”，而是：

> 根据 MoE 的 top-k 语义，把多个专家的候选输出重新合成一个最终 token 表示。

如果你愿意，我下一步可以继续给你展开：

- `decode_and_accumulate` 的具体数值过程
- 或者画一张 “dispatch → send → flag → recv → combine → output” 的时序图。
