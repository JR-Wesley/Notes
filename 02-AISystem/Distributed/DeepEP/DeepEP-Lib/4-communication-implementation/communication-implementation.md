This document covers the core communication implementation for DeepEP's internode operations, focusing on the RDMA and NVLink-based data dispatch and combine mechanisms. This implementation handles the orchestration of token communication across multiple RDMA ranks and NVLink peers in distributed expert-parallel workloads.

  

For intranode-only communication within a single node, see [Intranode Communication]([https://deepwiki.com/deepseek-ai/DeepEP/5.1-intranode-communication](https://deepwiki.com/deepseek-ai/DeepEP/5.1-intranode-communication)). For low-latency optimized communication modes, see [Low-Latency Mode]([https://deepwiki.com/deepseek-ai/DeepEP/5.3-low-latency-mode](https://deepwiki.com/deepseek-ai/DeepEP/5.3-low-latency-mode)). For higher-level communication patterns and the dispatch-combine model, see [Communication Model]([Communication Model | deepseek-ai/DeepEP | DeepWiki](https://deepwiki.com/deepseek-ai/DeepEP/3.2-communication-model)).

  

## Communication Architecture Overview

  

The internode communication system implements a two-phase dispatch-combine pattern that coordinates data movement across both RDMA ranks (inter-node) and NVLink peers (intra-node). The implementation uses a hierarchical approach where data flows through multiple communication layers.

  

![[Communication Architecture Overview.png]]

  

Sources: [csrc/kernels/internode.cu84-303]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L84-L303](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L84-L303)) [csrc/kernels/internode.cu357-992]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L357-L992](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L357-L992)) [csrc/kernels/internode.cu1368-1814]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L1368-L1814](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L1368-L1814))

  

## Core Data Structures and Metadata

  

The communication system relies on several key data structures for tracking token sources, managing buffer layouts, and coordinating multi-rank operations.

  

### SourceMeta Structure

  

The `SourceMeta` structure encodes critical routing information for each token, including source RDMA rank and NVLink peer membership.

  

![[Core Data Structures and Metadata.png]]

  

Sources: [csrc/kernels/internode.cu17-36]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L17-L36](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L17-L36)) [csrc/kernels/internode.cu530-541]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L530-L541](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L530-L541)) [csrc/kernels/internode.cu932-934]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L932-L934](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L932-L934))

  

### Buffer Layout Management

  

The system uses symmetric and asymmetric buffer layouts to optimize data placement and access patterns across different communication scenarios.

  

|Buffer Type|Structure|Purpose|Key Methods|

|---|---|---|---|

|`SymBuffer<T>`|Symmetric layout|RDMA rank communication|`send_buffer()`, `recv_buffer()`|

|`AsymBuffer<T>`|Asymmetric layout|NVLink peer communication|`buffer()`, `advance()`|

|Channel-based|Multi-channel partitioning|Parallel processing|`get_channel_task_range()`|

  

Sources: [csrc/kernels/internode.cu415-418]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L415-L418](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L415-L418)) [csrc/kernels/internode.cu430-434]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L430-L434](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L430-L434)) [csrc/kernels/internode.cu1434-1436]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L1434-L1436](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L1434-L1436))

  

## Dispatch Phase Implementation

  

The dispatch phase coordinates the forwarding of tokens from source ranks to destination ranks through a multi-stage process involving notification, data movement, and synchronization.

  

### Notification and Coordination

  

The `notify_dispatch` kernel establishes communication channels and coordinates token count information across all participating ranks.

![[Notification and Coordination.png]]

Sources: [csrc/kernels/internode.cu84-303]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L84-L303](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L84-L303)) [csrc/kernels/internode.cu305-348]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L305-L348](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L305-L348))

  

### Token Dispatch Kernel

  

The main `dispatch` kernel implements a complex multi-warp coordination system with specialized roles for different aspects of the communication pipeline.

  

#### Warp Role Assignments

![[Warp Role Assignments.png]]

Sources: [csrc/kernels/internode.cu370-405]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L370-L405](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L370-L405)) [csrc/kernels/internode.cu462-614]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L462-L614](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L462-L614)) [csrc/kernels/internode.cu690-834]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L690-L834](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L690-L834))

  

#### TMA Integration and Memory Management

![[TMA Integration and Memory Management.png]]

The dispatch kernel uses Tensor Memory Accelerator (TMA) operations for efficient data copying, particularly in the NVL forwarding and receiving paths.

  

Sources: [csrc/kernels/internode.cu445-455]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L445-L455](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L445-L455)) [csrc/kernels/internode.cu802-818]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L802-L818](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L802-L818)) [csrc/kernels/internode.cu940-984]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L940-L984](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L940-L984))

  

## Combine Phase Implementation

  

The combine phase aggregates results from expert processing back to the originating ranks, using a coordinated multi-stage approach that mirrors the dispatch phase structure.

  

### Cached Notification System

  

The `cached_notify` kernel prepares metadata for the combine phase, handling both buffer cleanup and head pointer management for cached execution scenarios.

![[Cached Notification System.png]]

Sources: [csrc/kernels/internode.cu1043-1185]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L1043-L1185](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L1043-L1185)) [csrc/kernels/internode.cu1187-1222]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L1187-L1222](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L1187-L1222))

  

### Combine Kernel Architecture

  

The `combine` kernel implements the final aggregation phase with specialized warp roles for handling different aspects of the reduction process.

  

#### Multi-Stage Token Combination

![[Multi-Stage Token Combination.png]]

Sources: [csrc/kernels/internode.cu1224-1357]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L1224-L1357](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L1224-L1357)) [csrc/kernels/internode.cu1668-1677]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L1668-L1677](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L1668-L1677)) [csrc/kernels/internode.cu1747-1760]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L1747-L1760](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L1747-L1760))

  

## Buffer Management and Synchronization

  

The communication implementation relies on sophisticated buffer management and synchronization mechanisms to coordinate multi-rank operations safely and efficiently.

  

### Queue Management and Flow Control

![[Queue Management and Flow Control.png]]

Sources: [csrc/kernels/internode.cu588-613]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L588-L613](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L588-L613)) [csrc/kernels/internode.cu735-758]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L735-L758](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L735-L758)) [csrc/kernels/internode.cu1629-1642]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L1629-L1642](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L1629-L1642))

  

### IBGDA Integration

  

The implementation integrates closely with InfiniBand GPU Direct Async (IBGDA) for efficient RDMA operations.

  

|IBGDA Function|Purpose|Usage Context|

|---|---|---|

|`nvshmemi_ibgda_put_nbi_warp`|Non-blocking warp-level PUT|Data dispatch, result forwarding|

|`nvshmemi_ibgda_amo_nonfetch_add`|Atomic memory operation|Tail pointer updates|

|`nvshmemi_ibgda_quiet`|Wait for operation completion|Synchronization barriers|

|`translate_dst_rdma_rank`|Low-latency rank translation|Address space mapping|

  

Sources: [csrc/kernels/internode.cu149-153]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L149-L153](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L149-L153)) [csrc/kernels/internode.cu672-674]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L672-L674](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L672-L674)) [csrc/kernels/internode.cu684-686]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L684-L686](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L684-L686))

  

## Performance Optimizations

  

The implementation includes several performance optimizations designed to maximize throughput and minimize latency in distributed communication scenarios.

  

### Memory Access Patterns

![[Memory Access Pattern.png]]

Sources: [csrc/kernels/internode.cu551-554]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L551-L554](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L551-L554)) [csrc/kernels/internode.cu1262-1294]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L1262-L1294](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L1262-L1294)) [csrc/kernels/internode.cu1588-1592]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L1588-L1592](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L1588-L1592))