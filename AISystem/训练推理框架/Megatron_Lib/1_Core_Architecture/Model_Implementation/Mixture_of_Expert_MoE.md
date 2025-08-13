---
dateCreated: 2025-08-09
dateModified: 2025-08-10
---

This document covers the Mixture of Experts (MoE) implementation in Megatron-Core, including the routing mechanisms, token dispatching strategies, expert architectures, and parallelism approaches. MoE enables scaling model capacity while maintaining computational efficiency by routing tokens to a subset of specialized expert networks.

For information about general transformer architectures, see [Model Implementations](https://deepwiki.com/NVIDIA/Megatron-LM/2.4-model-implementations). For distributed training strategies beyond expert parallelism, see [Parallelism Strategies](https://deepwiki.com/NVIDIA/Megatron-LM/3-parallelism-strategies).

## MoE Architecture Overview

The MoE system in Megatron-Core replaces traditional dense MLP layers with a sparse mixture of expert networks. Each token is dynamically routed to a subset of experts based on learned routing probabilities, enabling the model to scale capacity without proportionally increasing computation.

### Core MoE Flow

![](assets/Mixture_of_Expert_MoE.assets/Core%20MoE%20Flow.png)

**MoE Processing Pipeline**

- **Router**: Computes routing probabilities and selects top-k experts per token
- **Token Dispatcher**: Handles communication and token permutation across devices
- **Expert Networks**: Process tokens through specialized MLP networks
- **Token Combiner**: Aggregates expert outputs and restores token ordering
- **Auxiliary Losses**: Encourage load balancing across experts

Sources:[](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/moe_layer.py#L225-L270)

[megatron/core/transformer/moe/moe_layer.py225-270](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/moe_layer.py#L225-L270)[](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/README.md#L1-L63)

[megatron/core/transformer/moe/README.md1-63](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/README.md#L1-L63)

## Core Components Architecture

![](assets/Mixture_of_Expert_MoE.assets/Core%20Components%20Architecture.png)

**Component Relationships** The `MoELayer` orchestrates all components through a four-stage process: routing & preprocessing, token dispatch, expert computation, and token combination.

Sources:[](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/moe_layer.py#L92-L98)

[megatron/core/transformer/moe/moe_layer.py92-98](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/moe_layer.py#L92-L98) [](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/router.py#L25-L26)[megatron/core/transformer/moe/router.py25-26](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/router.py#L25-L26) [](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/token_dispatcher.py#L46-L49)[megatron/core/transformer/moe/token_dispatcher.py46-49](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/token_dispatcher.py#L46-L49)[](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/experts.py#L97-L101)

[megatron/core/transformer/moe/experts.py97-101](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/experts.py#L97-L101)

## Routing Mechanisms

The routing system determines which experts process each token through the `Router` class hierarchy.

### TopK Router Implementation

|Router Method|Purpose|Load Balancing Strategy|
|---|---|---|
|`sinkhorn_load_balancing()`|Sinkhorn algorithm routing|Optimal transport-based balancing|
|`aux_loss_load_balancing()`|Auxiliary loss routing|Switch/GShard-style load balancing|
|`seq_aux_loss_load_balancing()`|Sequence-level aux loss|Per-sample load balancing (DeepSeek-V2 style)|
|`routing()` with "none"|Naive top-k|No load balancing|

The `TopKRouter` supports several routing configurations:

![](assets/Mixture_of_Expert_MoE.assets/TopK%20Router%20Implementation.png)

**Expert Bias for Aux-Loss-Free Load Balancing** The router supports dynamic expert bias adjustment through `moe_router_enable_expert_bias`, which updates per-expert bias based on token assignment counts to encourage load balancing without auxiliary losses.

Sources:[](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/router.py#L394-L441)

[megatron/core/transformer/moe/router.py394-441](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/router.py#L394-L441) [](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/router.py#L156-L187)[megatron/core/transformer/moe/router.py156-187](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/router.py#L156-L187)[](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/moe_utils.py#L781-L799)

[megatron/core/transformer/moe/moe_utils.py781-799](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/moe_utils.py#L781-L799)

## Token Dispatching Strategies

Token dispatchers handle the communication and permutation required to route tokens to their assigned experts across devices.

### Dispatcher Comparison

|Dispatcher|Communication Pattern|Best Use Case|Memory Overhead|
|---|---|---|---|
|`MoEAllGatherTokenDispatcher`|AllGather → Local Processing → ReduceScatter|Small expert parallelism|Higher (stores all tokens)|
|`MoEAlltoAllTokenDispatcher`|AlltoAll → Local Processing → AlltoAll|Large expert parallelism|Lower (stores assigned tokens)|
|`MoEFlexTokenDispatcher`|Flexible with DeepEP optimization|Large-scale training|Optimized|

### AlltoAll Dispatcher Workflow

![](assets/Mixture_of_Expert_MoE.assets/AlltoAll%20Dispatcher%20Workflow.png)

**Token Permutation and Capacity Management** The dispatcher supports both dropless training and capacity-based token dropping. When `moe_expert_capacity_factor` is set, tokens exceeding expert capacity are dropped based on routing probabilities or position.

Sources:[](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/token_dispatcher.py#L332-L344)

[megatron/core/transformer/moe/token_dispatcher.py332-344](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/token_dispatcher.py#L332-L344) [](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/token_dispatcher.py#L424-L545)[megatron/core/transformer/moe/token_dispatcher.py424-545](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/token_dispatcher.py#L424-L545)[](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/moe_utils.py#L529-L642)

[megatron/core/transformer/moe/moe_utils.py529-642](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/moe_utils.py#L529-L642)

## Expert Network Implementations

Expert networks process tokens through specialized MLP architectures with different optimization strategies.

### Expert Types Comparison

|Expert Type|Optimization Strategy|Memory Layout|Best Performance|
|---|---|---|---|
|`SequentialMLP`|Process experts one by one|Standard linear layers|Small number of experts|
|`GroupedMLP`|Grouped GEMM operations|Concatenated weight matrices|Multiple local experts|
|`TEGroupedMLP`|Transformer Engine optimizations|TE-optimized layout|Large-scale training|

### GroupedMLP Architecture

![](assets/Mixture_of_Expert_MoE.assets/GroupedMLP%20Architecture.png)

**Grouped GEMM Optimization** `GroupedMLP` reshapes weight matrices to enable parallel processing of multiple experts in a single grouped GEMM operation, significantly improving throughput when multiple experts are co-located on the same device.

Sources:[](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/experts.py#L242-L301)

[megatron/core/transformer/moe/experts.py242-301](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/experts.py#L242-L301) [](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/experts.py#L266-L267)[megatron/core/transformer/moe/experts.py266-267](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/experts.py#L266-L267)[](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/experts.py#L269-L282)

[megatron/core/transformer/moe/experts.py269-282](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/experts.py#L269-L282)

## Load Balancing and Auxiliary Losses

MoE models require load balancing mechanisms to prevent expert collapse and ensure efficient utilization.

### Auxiliary Loss Functions

![](assets/Mixture_of_Expert_MoE.assets/Auxiliary%20Loss%20Functions.png)

### Load Balancing Strategies

|Strategy|Configuration|Behavior|
|---|---|---|
|Switch/GShard Style|`moe_router_load_balancing_type="aux_loss"`|Penalizes uneven expert usage across all tokens|
|Sequence-level|`moe_router_load_balancing_type="seq_aux_loss"`|Computes loss per individual sequence|
|Sinkhorn|`moe_router_load_balancing_type="sinkhorn"`|Optimal transport-based assignment|
|Aux-loss-free|`moe_router_enable_expert_bias=True`|Dynamic expert bias without auxiliary losses|

**Loss Tracking and Logging** The system provides comprehensive auxiliary loss tracking through `save_to_aux_losses_tracker()` and `track_moe_metrics()`, enabling per-layer monitoring and debugging.

Sources:[](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/moe_utils.py#L33-L78)

[megatron/core/transformer/moe/moe_utils.py33-78](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/moe_utils.py#L33-L78) [](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/moe_utils.py#L81-L131)[megatron/core/transformer/moe/moe_utils.py81-131](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/moe_utils.py#L81-L131)[](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/moe_utils.py#L645-L778)

[megatron/core/transformer/moe/moe_utils.py645-778](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/moe_utils.py#L645-L778)

## Parallelism Integration

MoE integrates with Megatron's parallelism strategies through expert parallelism (EP) and supports combination with tensor, data, pipeline, and context parallelism.

### Expert Parallelism Patterns

### MoE Parallel Folding

Megatron-Core supports different parallelism configurations for attention and MoE components:

|Component|Tensor Parallel|Expert Parallel|Notes|
|---|---|---|---|
|Attention|`tensor_model_parallel_size`|N/A|Standard TP for attention layers|
|MoE Experts|`expert_tensor_parallel_size`|`expert_model_parallel_size`|Independent TP/EP configuration|

**Sequence Parallelism Requirement** When combining MoE with tensor parallelism (`tp_size > 1`), sequence parallelism must be enabled to maintain correctness during training.

Sources:[](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/moe_layer.py#L240-L244)

[megatron/core/transformer/moe/moe_layer.py240-244](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/moe_layer.py#L240-L244) [](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/moe_utils.py#L9-L9)[megatron/core/transformer/moe/moe_utils.py9](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/moe_utils.py#L9-L9)[](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/token_dispatcher.py#L64-L71)

[megatron/core/transformer/moe/token_dispatcher.py64-71](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/token_dispatcher.py#L64-L71)

## Performance Optimizations

### Token Permutation Fusion

The system supports fused token permutation operations through `moe_permute_fusion`, which combines permutation and probability application into optimized kernels when Transformer Engine >= 2.1.0 is available.

### Shared Expert Integration

![](assets/Mixture_of_Expert_MoE.assets/Shared%20Expert%20Integration.png)

**Communication Overlap** With `moe_shared_expert_overlap=True`, shared expert computation can be overlapped with MoE token dispatching communication for improved performance.

### Advanced Features

|Feature|Configuration|Benefits|
|---|---|---|
|FP8 Training|`fp8="hybrid"` + `moe_router_padding_for_fp8=True`|Memory and compute efficiency|
|CUDA Graph Support|`enable_cuda_graph=True`|Reduced kernel launch overhead|
|DeepEP Integration|`moe_token_dispatcher_type="flex"` + `moe_enable_deepep=True`|Optimized cross-node communication|
|Router Dtype Control|`moe_router_dtype="fp32"`|Improved numerical stability|

Sources: [megatron/core/transformer/moe/shared_experts.py26-33](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/shared_experts.py#L26-L33) [megatron/core/transformer/moe/README.md180-191](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/README.md#L180-L191) [megatron/core/transformer/moe/README.md175-178](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/moe/README.md#L175-L178)
