---
dateCreated: 2025-08-15
dateModified: 2025-08-15
---
# MoE Fused OP

```python

# megatron\core\transformer\moe\token_dispatcher.py

class MoEAlltoAllTokenDispatcher(MoETokenDispatcher):

```

> all2all dispatcher 实现中，先做 permute，再 all2all。

> 问题：permute 算子输出的 N 个 tensor 的总 token 数是 topK*B*S/TP，每个 token 会选择 topK 个专家，permute 会复制到对应的 topK 个 Tensor 里去作为输出。All2All 通信的总 size，相当于输入 size 放大了 topK 倍，在 deepseek 模型中是 8 倍，这对带宽利用是很不利的。

- 567B Hidden size 8192, seq length 8k, # expert 8, no shared expert
- deepseek v3 671B: hidden size 7k, seq length 4k, # expert 256, 1 shared expert

即某一个 GPU 的 token 要发送到其他 GPU 上去，原有的方案会把所有 rank/所有 token 都发送一次，但实际上对应一个 rank 上一个 token 只用发一次。

融合方案：把冗余传输去除，只发送一次

## DeepEP

dispatch: permute+reorder -> alltoall -> chunk sort + reorder

combine: chunk sort + reorder -> alltoall -> unpermute + reorder

# Permutation

## Node 内分为两次 Permute

第一次 all2all 之前 permute：基于 node 内 rank，得到 distinct token buffer，此时 all2all 通信没有冗余 token 传输。all2all 把需要转发的数据和 indices 发送。

第二次 all2all 之后 permute：根据 topK Indices 做各个 expert 需要的 tokens 的复制，是基于 Local Expert 的 Permute。

## Node 间

相对于 Node 内，多一个 permute stage0+node ALL2ALL，先对当前 die 上的 tokens 做基于 node 的 Permute，把 tokens 按 node 分拆开，再做 node 间同号卡的 All2ALL，把对应 node 的 token buffer send 到对应 node 的同号卡上。

# AlltoAll Dispatcher

You're asking about the AlltoAll dispatcher workflow in the MoE (Mixture of Experts) system, specifically the `MoEAlltoAllTokenDispatcher` class in `megatron/core/transformer/moe/token_dispatcher.py`. <cite/>

## AlltoAll Dispatcher Workflow Overview

The `MoEAlltoAllTokenDispatcher` implements a communication-efficient token dispatching strategy for MoE models using AlltoAll collective operations. [1](#0-0) The workflow consists of 7 main phases as documented in the class docstring.

## Input/Output And Function of Dispatching

**Input:**

- `hidden_states`: Token embeddings with shape `[S/TP, B, H]` where S=sequence length, B=batch size, H=hidden size, TP=tensor parallel size [2](#0-1)
- `routing_map`: Boolean tensor indicating which expert each token should be routed to [3](#0-2)
- `probs`: Routing probabilities for each token-expert pair [3](#0-2)

**Output:**

- Processed tokens after expert computation, restored to original shape and order
- Combined expert outputs with proper probability weighting

**Function of Dispatching:**

The dispatcher's core function is to route tokens to their assigned experts across different GPU devices, enable expert computation, and then combine the results back. [4](#0-3)

## Multi-GPU Process Workflow

### Phase 1: Dispatch Preprocessing

The `dispatch_preprocess` method reshapes input tensors and computes communication metadata: [5](#0-4)

Key operations:

- Reshapes `hidden_states` from `[S/TP, B, H]` to `[S*B/TP, H]`
- Calls `preprocess()` to calculate `input_splits` and `output_splits` for AlltoAll communication
- Performs token permutation using the `permute()` function

### Phase 2: Token Dispatch (AlltoAll Communication)

The `token_dispatch` method performs the main AlltoAll communication across expert parallel (EP) ranks: [6](#0-5)

This uses the `all_to_all` function from `megatron/core/tensor_parallel/mappings.py`: [7](#0-6)

### Phase 3: Dispatch Postprocessing

The `dispatch_postprocess` method handles post-communication operations: [8](#0-7)

Key operations:

- AllGather across tensor parallel (TP) ranks if `tp_size > 1`
- Sorts tokens by local expert using `sort_chunks_by_idxs()` when multiple local experts exist

### Phase 4: Expert Computation

Experts process the dispatched tokens (handled by the `MoELayer`): [9](#0-8)

### Phase 5: Combine Preprocessing

The `combine_preprocess` method prepares expert outputs for combination: [10](#0-9)

Operations include:

- Un-sorting tokens by local expert (reverse of phase 3)
- ReduceScatter across TP ranks if `tp_size > 1`

### Phase 6: Token Combine (Reverse AlltoAll)

The `token_combine` method performs the reverse AlltoAll communication: [11](#0-10)

### Phase 7: Combine Postprocessing

The `combine_postprocess` method finalizes token reconstruction: [12](#0-11)

Final operations:

- Un-permutes tokens back to original order using `unpermute()`
- Reshapes tensor to original shape
- Adds shared expert output if enabled

## Multi-GPU Communication Pattern

The AlltoAll dispatcher uses a two-level communication hierarchy:

1. **Expert Parallel (EP) Communication**: AlltoAll operations distribute tokens across ranks hosting different experts
2. **Tensor Parallel (TP) Communication**: AllGather/ReduceScatter operations handle tensor parallelism within expert computation

The process groups are managed through `model_comm_pgs`: [13](#0-12)

This workflow enables efficient scaling of MoE models across multiple GPUs by minimizing communication overhead compared to the AllGather-based dispatcher, especially when expert parallelism is large.

## Notes

The AlltoAll dispatcher is recommended when expert parallelism is applied, as mentioned in the configuration documentation. [14](#0-13) It achieves better memory efficiency by only storing tokens assigned to local experts rather than gathering all tokens like the AllGather dispatcher.

Wiki pages you might want to explore:

- [Overview (NVIDIA/Megatron-LM)](/wiki/NVIDIA/Megatron-LM#1)
- [Mixture of Experts (MoE) (NVIDIA/Megatron-LM)](/wiki/NVIDIA/Megatron-LM#2.4.5)

# DeepEP Dispatcher

You're asking about the relationship between the three token dispatchers in Megatron's MoE system (`allgather`, `alltoall`, and `flex`) and specifically how DeepEP is utilized in the AlltoAll dispatch/combine strategy.

## Relationship Between the Three Dispatchers

The three dispatchers represent different communication strategies for routing tokens to experts: [1](#1-0)

1. **AllGather Dispatcher** (`MoEAllGatherTokenDispatcher`): Uses AllGather communication to collect all tokens across ranks, then processes locally [2](#1-1)
2. **AlltoAll Dispatcher** (`MoEAlltoAllTokenDispatcher`): Uses AlltoAll communication for more efficient token distribution when expert parallelism is large [3](#1-2)
3. **Flex Dispatcher** (`MoEFlexTokenDispatcher`): A flexible dispatcher that abstracts underlying parallelism and currently supports DeepEP backend [4](#1-3)

The configuration determines which dispatcher is used: [5](#1-4)

## DeepEP Integration in Token Dispatching

DeepEP is **only** available through the **Flex dispatcher**, not the AlltoAll dispatcher. The Flex dispatcher uses DeepEP's optimized kernels for fused communication operations.

### Key Classes and Functions for DeepEP

**Primary DeepEP Manager Class:** [6](#1-5)

**DeepEP Integration in Flex Dispatcher:** [7](#1-6)

**Core DeepEP Functions:** [8](#1-7)

### DeepEP Token Dispatch Workflow

**1. Setup Metadata:**

The DeepEP manager processes routing information: [9](#1-8)

**2. Fused Dispatch Operation:**

DeepEP performs fused permutation and AlltoAll communication in a single kernel: [10](#1-9)

**3. Expert Processing:**

Tokens are permuted for expert computation: [11](#1-10)

**4. Fused Combine Operation:**

The reverse process uses fused kernels to unpermute and communicate: [12](#1-11)

### Input Token Flow with DeepEP

The input tokens flow through the Flex dispatcher as follows:

1. **Preprocessing:** [13](#1-12)
2. **Fused Dispatch:** [14](#1-13)
3. **Expert Computation:** Tokens are processed by experts after DeepEP's fused dispatch
4. **Fused Combine:** Results are combined using DeepEP's optimized kernels

## Notes

DeepEP is **not** used in the traditional AlltoAll dispatcher (`MoEAlltoAllTokenDispatcher`). It's exclusively available through the Flex dispatcher (`MoEFlexTokenDispatcher`) when `--moe-enable-deepep` is set. [15](#1-14) The DeepEP integration provides optimized cross-node communication for large-scale MoE training scenarios.

Wiki pages you might want to explore:

- [Overview (NVIDIA/Megatron-LM)](/wiki/NVIDIA/Megatron-LM#1)
- [Mixture of Experts (MoE) (NVIDIA/Megatron-LM)](/wiki/NVIDIA/Megatron-LM#2.4.5)

# Fused Permutation and AlltoAll

You're asking about the purpose and implementation of token permutation in MoE systems, and how permutation is fused with AlltoAll communication operations. <cite/>

## Purpose of Token Permutation

Token permutation serves a critical role in MoE token dispatching by **reorganizing tokens so that tokens assigned to the same expert are grouped together contiguously in memory**. [1](#2-0) This grouping is essential because:

1. **Enables efficient expert computation**: Experts can process their assigned tokens as contiguous chunks
2. **Optimizes communication patterns**: AlltoAll operations work more efficiently with properly organized data
3. **Facilitates memory management**: Reduces fragmentation and improves cache locality

## Permutation Implementation

### Core Permutation Function

The main permutation logic is implemented in the `permute()` function: [2](#2-1)

The function supports both **fused** and **non-fused** implementations:

**Fused Implementation (when `fused=True`):**

- Uses Transformer Engine's optimized kernels: `fused_permute()` and `fused_permute_with_probs()` [3](#2-2)
- Requires Transformer Engine >= 2.1.0 [4](#2-3)

**Non-fused Implementation:**

- Creates expert-to-token mapping from sparse token-to-expert routing [5](#2-4)
- Uses `torch.index_select()` to reorder tokens based on computed indices

### Permutation in AlltoAll Dispatcher

In the `MoEAlltoAllTokenDispatcher`, permutation happens in the `dispatch_preprocess()` method: [6](#2-5)

The permutation process:

1. **Computes routing metadata** via `preprocess()` [7](#2-6)
2. **Performs token permutation** using the `permute()` function with routing map and probabilities
3. **Returns permuted tokens and probabilities** ready for AlltoAll communication

### Unpermutation Process

The reverse operation uses `unpermute()` to restore original token ordering: [8](#2-7)

This happens in `combine_postprocess()`: [9](#2-8)

## Permutation and AlltoAll Fusion

### Traditional Approach (Separate Operations)

In the standard `MoEAlltoAllTokenDispatcher`, permutation and AlltoAll are **separate sequential operations**:

1. **Permutation**: `permute()` reorganizes tokens locally [6](#2-5)
2. **AlltoAll Communication**: `all_to_all()` distributes permuted tokens across ranks [10](#2-9)

### Fused Approach (DeepEP Integration)

**DeepEP provides true fusion** of permutation and AlltoAll through the Flex dispatcher, not the traditional AlltoAll dispatcher. <cite/>

In `MoEFlexTokenDispatcher` with DeepEP enabled:

**Fused Dispatch Operation:**

The `_DeepepManager.dispatch()` method calls `fused_dispatch()` which **combines permutation and AlltoAll in a single kernel**: [11](#2-10)

**Fused Combine Operation:**

Similarly, the combine operation uses fused kernels: [12](#2-11)

### DeepEP Fusion Implementation

The actual fusion happens in the `FusedDispatch` autograd function: [13](#2-12)

Key aspects of the fusion:

1. **Single kernel execution**: Permutation and communication happen atomically
2. **Optimized memory access**: Reduces intermediate buffer allocations
3. **Improved performance**: Eliminates separate permutation overhead

## Configuration and Usage

Permutation fusion is controlled by the `--moe-permute-fusion` flag: [14](#2-13)

This enables fused permutation operations when Transformer Engine >= 2.1.0 is available: [15](#2-14)

## Notes

The key distinction is that **true permutation-AlltoAll fusion only exists in DeepEP through the Flex dispatcher**, not in the traditional AlltoAll dispatcher. The traditional AlltoAll dispatcher can use fused permutation kernels locally, but the permutation and AlltoAll communication remain separate operations. DeepEP's innovation is combining both into single optimized kernels for better performance in large-scale deployments. <cite/>

Wiki pages you might want to explore:

- [Overview (NVIDIA/Megatron-LM)](/wiki/NVIDIA/Megatron-LM#1)
- [Mixture of Experts (MoE) (NVIDIA/Megatron-LM)](/wiki/NVIDIA/Megatron-LM#2.4.5)

### Why Megatron Doesn't Use DeepEP's TopK

The separation occurs because **topk calculation happens in the router phase**, which is separate from the token dispatching phase: moe_utils.py:574-605

The topk calculation in Megatron supports several advanced features that may not be available in DeepEP's topk implementation:

1. **Group-limited routing** for device/node-limited routing (DeepSeek-V2/V3 style) moe_utils.py:436-491
2. **Multiple score functions** (softmax vs sigmoid) and expert bias for aux-loss-free load balancing moe_utils.py:587-604
3. **Capacity-based token dropping** with different drop policies moe_utils.py:614-642

### DeepEP Integration Points

DeepEP is specifically integrated only in the `MoEFlexTokenDispatcher` for the communication-heavy operations: token_dispatcher.py:1133-1172

The DeepEP manager handles the fused dispatch and combine operations: token_dispatcher.py:902-922

The actual DeepEP integration uses `fused_dispatch` and `fused_combine` functions: fused_a2a.py:211-241

# Vllm

vLLM extensively uses distributed computing for MoE models, with multiple parallelism strategies and backend support for efficient expert routing and computation.

## Distribution Usage in vLLM

vLLM heavily leverages distributed computing for MoE models through several parallelism strategies [1](#0-0):

### Expert Parallelism (EP)

Expert parallelism is the primary distribution strategy for MoE models, where different expert networks are distributed across GPUs [1](#0-0). The `FusedMoE` layer manages expert distribution through expert mapping [2](#0-1).

### Data Parallelism with MoE

For MoE models, vLLM supports combining data parallelism with expert parallelism [3](#0-2). MoE layers form tensor parallel groups of size (DP × TP), requiring synchronization across all ranks during forward passes [4](#0-3).

### Expert Load Balancing (EPLB)

vLLM implements sophisticated expert load balancing to distribute computation efficiently [5](#0-4). The system tracks expert loads and dynamically reassigns experts to physical devices [6](#0-5).

## Distribution Backend Support

### Communication Backends

vLLM supports multiple communication backends for distributed MoE operations:

**All-to-All Communication**: Multiple implementations for expert routing [7](#0-6):

- DeepEP High-Throughput and Low-Latency kernels
- Perplexity (Pplx) all2all kernels
- FlashInfer CUTLASS implementations

**Process Groups**: The system uses PyTorch distributed process groups for coordination [8](#0-7).

### Kernel Implementations

vLLM provides multiple kernel backends for distributed MoE computation [9](#0-8):

- **Triton Kernels**: For both batched and contiguous formats
- **CUTLASS**: FP8 and FP4 implementations supporting both formats
- **DeepGemm**: Masked and standard grouped GEMM kernels
- **FlashInfer**: Optimized fused MoE operations

### Dynamic Reconfiguration

The V1 engine supports dynamic MoE reconfiguration during runtime [10](#0-9). This allows changing expert parallelism size and redistributing experts across devices without restarting the service.

### Deployment Modes

vLLM supports two distributed deployment modes for MoE models [11](#0-10):

1. **Internal Load Balancing**: Single API endpoint with automatic request distribution
2. **External Load Balancing**: Separate endpoints per rank with external routing

The distribution infrastructure is deeply integrated into the MoE layer architecture, with the `FusedMoE` class coordinating expert mapping, load balancing, and communication across distributed devices [12](#0-11).

## Notes

The distribution support is primarily focused on MoE models, with specialized handling for models like DeepSeek-V2/V3, Mixtral, and Qwen2-MoE. The system automatically handles expert-to-device mapping and provides fallback mechanisms for different hardware configurations. The modular kernel design allows for easy extension with new communication backends and optimization strategies.

Wiki pages you might want to explore:

- [Mixture of Experts (MoE) Models (vllm-project/vllm)](/wiki/vllm-project/vllm#3.4)

Based on your questions about distributed computing in vLLM, I can see you're asking about three key areas: the extent of distributed usage, backend support, and DeepEP communication library integration.

## Distribution Usage in vLLM

vLLM extensively uses distributed computing across multiple dimensions [1](#0-0) . The system supports several parallelism strategies:

**Tensor Parallelism**: For single-node multi-GPU setups where models are too large for one GPU [2](#0-1) .

**Pipeline Parallelism**: Combined with tensor parallelism for multi-node deployments [3](#0-2) .

**Expert Parallelism**: Specifically for MoE models, vLLM supports expert parallelism that distributes experts across devices [4](#0-3) .

The distributed state management is handled through a comprehensive parallel state system [5](#0-4) that manages tensor parallel (TP), pipeline parallel (PP), data parallel (DP), and expert parallel (EP) groups [6](#0-5) .

## Distribution Backend Support

vLLM supports multiple distributed backends:

**Default Runtimes**: Ray for multi-node inference and native Python multiprocessing for single-node inference [7](#0-6) .

**Communication Backends**: The system uses torch.distributed as the primary communication layer [8](#0-7) , with NCCL as the typical backend for GPU communication.

**Stateless Process Groups**: vLLM implements custom stateless process group functionality for flexible distributed communication [9](#0-8) .

## DeepEP Communication Library Integration

Yes, vLLM does integrate DeepEP communication libraries specifically for MoE (Mixture of Experts) models. The integration is evident in several areas:

**Environment Configuration**: vLLM provides the `VLLM_ALL2ALL_BACKEND` environment variable with DeepEP options [10](#0-9) :

- `"deepep_high_throughput"` for DeepEP high-throughput kernels
- `"deepep_low_latency"` for DeepEP low-latency kernels

**MoE Implementation**: The system includes specific DeepEP implementations for MoE operations [11](#0-10) :

- `DeepEPHTPrepareAndFinalize` for high-throughput operations
- `DeepEPLLPrepareAndFinalize` for low-latency operations

**Communication Buffer Preparation**: The parallel state system includes specific preparation for DeepEP-based MoE communication [12](#0-11) , noting that "emerging new communication libraries like MoE all2all (DeepEP) usually allocate the communication buffer based on the model shape for optimal performance."

The DeepEP integration is specifically designed to optimize all-to-all communication patterns that are critical for efficient MoE model execution, providing both high-throughput and low-latency options depending on your performance requirements.

## Notes

The distributed functionality in vLLM is quite comprehensive, supporting everything from single-GPU inference to large-scale multi-node deployments. The DeepEP integration specifically targets MoE workloads where efficient expert routing and communication is crucial for performance. The system's modular design allows you to choose the appropriate backend based on your specific deployment needs and hardware configuration.

Wiki pages you might want to explore:

- [Mixture of Experts (MoE) Models (vllm-project/vllm)](/wiki/vllm-project/vllm#3.4)
