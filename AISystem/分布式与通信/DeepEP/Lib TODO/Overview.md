---
dateCreated: 2025-08-08
dateModified: 2025-08-08
---
## Purpose and Scope

DeepEP is a high-performance communication library specifically designed for Mixture-of-Experts (MoE) models and expert parallelism (EP) workloads. It provides optimized GPU kernels for the fundamental MoE operations: **dispatch** (routing tokens to experts) and **combine** (aggregating expert outputs). The library supports both high-throughput operations for training and inference prefilling, as well as low-latency operations optimized for inference decoding scenarios.

This document provides a high-level overview of the DeepEP system architecture, core concepts, and main components. For detailed installation instructions, see [Installation]([Installation | deepseek-ai/DeepEP | DeepWiki](https://deepwiki.com/deepseek-ai/DeepEP/2.1-installation)). For comprehensive API documentation, see [Python API]([Python API | deepseek-ai/DeepEP | DeepWiki](https://deepwiki.com/deepseek-ai/DeepEP/4-python-api)). For implementation details of specific communication modes, see [Communication Implementation]([Communication Implementation | deepseek-ai/DeepEP | DeepWiki](https://deepwiki.com/deepseek-ai/DeepEP/5-communication-implementation)).

## Core MoE Communication Pattern

DeepEP implements the dispatch-combine communication pattern that is fundamental to MoE architectures. This pattern involves two primary phases:

![[Core MoE Communication Pattern.png]]

The dispatch phase takes input tokens and routing information (`topk_idx`, `topk_weights`) and redistributes tokens to the appropriate expert ranks. The combine phase performs the inverse operation, aggregating expert outputs back to the original token layout using the routing metadata.

Sources: [README.md1-10]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/README.md#L1-L10](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/README.md#L1-L10)) [README.md158-184]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/README.md#L158-L184](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/README.md#L158-L184))

## System Architecture

DeepEP is structured in multiple layers, from high-level Python APIs down to hardware-specific CUDA kernels:

![[DeepWiki/DeepEP/assets/System Architecture.png]]

The architecture provides multiple abstraction layers that allow users to access optimized communication primitives through a simple Python interface while leveraging low-level hardware capabilities.

Sources: [README.md122-151]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/README.md#L122-L151](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/README.md#L122-L151)) [README.md235-288]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/README.md#L235-L288](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/README.md#L235-L288))

## Communication Modes

DeepEP supports three distinct communication modes, each optimized for different scenarios and hardware topologies:

### Intranode Communication

Uses NVLink for high-bandwidth communication within a single node (typically 8 GPUs). Implemented in `intranode.cu` kernels with bandwidth reaching ~160 GB/s. Suitable for single-node MoE deployments.

### Internode Communication

Combines NVLink and RDMA for multi-node scaling. Uses asymmetric bandwidth forwarding where tokens are first aggregated via NVLink within nodes, then communicated across nodes via RDMA (~50 GB/s), and finally distributed via NVLink on the destination node. Implemented in `internode.cu` kernels.

### Low-Latency Mode

Optimized for inference decoding with pure RDMA communication to minimize latency. Uses specialized kernels in `internode_ll.cu` and supports communication-computation overlap through hook-based mechanisms without occupying Streaming Multiprocessors (SMs).

Sources: [README.md13-23]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/README.md#L13-L23](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/README.md#L13-L23)) [README.md26-38]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/README.md#L26-L38](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/README.md#L26-L38)) [README.md232-288]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/README.md#L232-L288](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/README.md#L232-L288))

## Key Components

### Buffer System

The `deep_ep.Buffer` class serves as the main interface for MoE communication operations. It manages memory allocation for both NVLink (`num_nvl_bytes`) and RDMA (`num_rdma_bytes`) buffers and provides methods for `dispatch`, `combine`, and `get_dispatch_layout` operations.

![[Buffer System.png]]

### Configuration System

The `deep_ep_cpp.Config` class manages communication parameters and provides auto-tuned configurations for different cluster sizes. It includes methods like `get_nvl_buffer_size_hint()` and `get_rdma_buffer_size_hint()` for optimal memory allocation.

### Event Management

The `EventOverlap` class provides CUDA event-based synchronization for communication-computation overlap, enabling asynchronous operations and pipeline optimization.

Sources: [README.md138-150]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/README.md#L138-L150](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/README.md#L138-L150)) [README.md158-224]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/README.md#L158-L224](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/README.md#L158-L224)) [README.md247-259]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/README.md#L247-L259](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/README.md#L247-L259))

## Performance Characteristics

DeepEP delivers high performance across different hardware configurations:

|Communication Type|Expert Count|Bandwidth|Use Case|

|---|---|---|---|

|Intranode (NVLink)|8|~155 GB/s|Single-node training|

|Internode (RDMA)|16-64|43-58 GB/s|Multi-node training|

|Low-Latency (Pure RDMA)|8-256|39-127 GB/s|Inference decoding|

The library supports precision optimizations including FP8 for dispatch operations and BF16 for combine operations, balancing memory bandwidth with numerical precision requirements.

Sources: [README.md11-38]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/README.md#L11-L38](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/README.md#L11-L38))

## Hardware Requirements

DeepEP requires specific hardware capabilities:

- **GPU Architecture**: Ampere (SM80) or Hopper (SM90) GPUs with SM90 PTX ISA support
- **CUDA Version**: CUDA 11.0+ for SM80, CUDA 12.3+ for SM90
- **Intranode**: NVLink connectivity for high-bandwidth GPU-to-GPU communication
- **Internode**: RDMA-capable network (InfiniBand recommended) for multi-node scaling
- **Dependencies**: NVSHMEM runtime for multi-GPU memory programming, PyTorch 2.1+

The system is optimized for DeepSeek-V3 model configurations but supports general MoE architectures with configurable expert counts and token routing strategies.

Sources: [README.md43-56]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/README.md#L43-L56](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/README.md#L43-L56)) [README.md89-115]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/README.md#L89-L115](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/README.md#L89-L115))
