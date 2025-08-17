---
dateCreated: 2025-08-15
dateModified: 2025-08-15
---
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

## 3 MoE

The Three MoE Dispatcher Classes

The three dispatcher classes all inherit from MoETokenDispatcher and implement both dispatch and combine operations token_dispatcher.py:135-188 :

###

1. MoEAllGatherTokenDispatcher

Implementation Method: Uses AllGather communication pattern token_dispatcher.py:196-200

Key Operations:

- Dispatch: Gathers tokens from all TP×EP ranks using AllGather token_dispatcher.py:240-262

- Combine: Uses Reduce-Scatter to aggregate expert outputs back to original ranks token_dispatcher.py:312-325

Best Use Case: Small expert parallelism scenarios, but has higher memory overhead as it stores all tokens

###

1. MoEAlltoAllTokenDispatcher

Implementation Method: Uses AlltoAll communication pattern with a 7-step workflow token_dispatcher.py:332-344

Key Operations:

- Dispatch: AlltoAll communication for expert parallelism, followed by AllGather for tensor parallelism token_dispatcher.py:602-628

- Combine: Reverse AlltoAll to redistribute tokens back to original ranks token_dispatcher.py:742-768

Best Use Case: Large expert parallelism with lower memory overhead as it only stores assigned tokens

###

1. MoEFlexTokenDispatcher

Implementation Method: Uses fused DeepEP kernels that combine permutation and communication token_dispatcher.py:1133-1137

Key Operations:

- Dispatch: Fused permutation and AlltoAll in a single optimized operation token_dispatcher.py:1232-1259

- Combine: Fused un-permutation and communication using DeepEP kernels token_dispatcher.py:1289-1307

Best Use Case: Large-scale training with optimized cross-node communication

##

Backend Implementation Differences

###

1. MoEFlexTokenDispatcher - DeepEP with CUDA Kernels

The Flex dispatcher uses DeepEP's fused CUDA kernels for optimized communication token_dispatcher.py:1161-1162 . It leverages the _DeepepManager which provides fused dispatch and combine operations token_dispatcher.py:902-922 .

Key Backend Features:

- Fused CUDA kernels: Uses fused_dispatch and fused_combine from DeepEP token_dispatcher.py:989-999

- Single-step operations: Combines permutation and AlltoAll communication in one kernel call token_dispatcher.py:1240-1245

- Cross-node optimization: Specifically designed for efficient cross-node token communication README.md:460-462

The fused kernels are implemented in FusedDispatch and FusedCombine autograd functions fused_a2a.py:68-82 .

###

1. MoEAlltoAllTokenDispatcher - Standard PyTorch AlltoAll

This dispatcher uses PyTorch's native distributed communication primitives without custom CUDA kernels token_dispatcher.py:621-627 .

Key Backend Features:

- Standard AlltoAll: Uses all_to_all from megatron.core.tensor_parallel token_dispatcher.py:765-767

- Separate permutation: Token permutation and communication are separate operations using permute and unpermute functions token_dispatcher.py:588-599

- Multi-step workflow: 7-step process with explicit preprocess/postprocess phases token_dispatcher.py:336-344

###

1. MoEAllGatherTokenDispatcher - Standard PyTorch AllGather/ReduceScatter

This dispatcher uses the simplest backend with standard PyTorch collective operations token_dispatcher.py:258-260 .

Key Backend Features:

- AllGather/ReduceScatter: Uses gather_from_sequence_parallel_region and reduce_scatter_to_sequence_parallel_region token_dispatcher.py:322-324

- No custom kernels: Relies entirely on PyTorch's native distributed primitives

- Higher memory usage: Stores all tokens across TP×EP ranks README.md:441

##

Performance and Optimization Hierarchy

The three dispatchers represent different levels of optimization:

1. AllGather (simplest): Standard PyTorch collectives, highest memory usage

2. AlltoAll (intermediate): Standard PyTorch with optimized communication pattern, lower memory usage

3. Flex (most advanced): Custom CUDA kernels with fused operations, optimized for large-scale cross-node scenarios

The Flex dispatcher requires DeepEP installation and is enabled via --moe-enable-deepep arguments.py:2806-2807 , while the other two use standard PyTorch distributed communication without additional dependencies.
