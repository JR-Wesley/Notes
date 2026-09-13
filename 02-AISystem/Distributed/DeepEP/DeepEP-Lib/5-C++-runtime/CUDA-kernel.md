---
dateCreated: 2025-08-08
dateModified: 2025-08-08
---
This document covers the CUDA kernel implementations that form the core computational engine of DeepEP's communication system. These kernels handle token routing, data movement, and synchronization operations across different hardware communication paths.

## Kernel Architecture Overview

DeepEP's CUDA kernels are organized into specialized source files that handle different aspects of MoE communication:

**Kernel Module Architecture**

**Sources:** [csrc/kernels/layout.cu1-136]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/layout.cu#L1-L136](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/layout.cu#L1-L136)) [csrc/kernels/intranode.cu1-806]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/intranode.cu#L1-L806](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/intranode.cu#L1-L806)) [csrc/kernels/internode.cu1-1495]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L1-L1495](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L1-L1495)) [csrc/kernels/internode_ll.cu1-737]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode_ll.cu#L1-L737](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode_ll.cu#L1-L737)) [csrc/kernels/runtime.cu1-94]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/runtime.cu#L1-L94](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/runtime.cu#L1-L94))

## Layout Kernels

The `layout.cu` file provides token distribution calculations that determine how tokens are routed across ranks and experts:

|Function|Purpose|Template Parameters|

|---|---|---|

|`get_dispatch_layout`|Calculate token routing statistics|`kNumThreads=256`, `kNumExpertsPerSM=4`, `kNumRanksPerSM=8`|

**Layout Kernel Processing**

![[Layout Kernel Processing.png]]

The kernel processes `topk_idx` arrays to calculate token distribution statistics using shared memory for efficient per-thread counting, then reduces results across threads.

**Sources:** [csrc/kernels/layout.cu9-116]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/layout.cu#L9-L116](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/layout.cu#L9-L116)) [csrc/kernels/layout.cu118-131]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/layout.cu#L118-L131](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/layout.cu#L118-L131))

## Intranode Kernels

The `intranode.cu` file handles NVLink-based communication within a single node:

|Function|Template Parameters|Purpose|

|---|---|---|

|`notify_dispatch`|`kNumRanks`|Setup rank-to-rank token counts and barriers|

|`cached_notify_dispatch`|`kNumRanks`|Optimized notify using cached data|

|`dispatch`|`kNumRanks`, `kNumThreads=768`, `kNumTMABytesPerWarp=8192`|Token scattering via NVLink|

|`cached_notify_combine`|`kNumRanks`|Setup combine metadata with send_head|

|`combine`|`dtype_t`, `kNumRanks`, `kNumThreads`, `kNumTMABytesPerWarp`|Result aggregation|

**Intranode Dispatch Architecture**

![[Intranode Dispatch Architecture.png]]

The dispatch kernel uses even-numbered SMs for sending and odd-numbered SMs for receiving, with TMA acceleration for efficient memory transfers on SM90 hardware.

**Sources:** [csrc/kernels/intranode.cu11-130]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/intranode.cu#L11-L130](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/intranode.cu#L11-L130)) [csrc/kernels/intranode.cu166-509]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/intranode.cu#L166-L509](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/intranode.cu#L166-L509)) [csrc/kernels/intranode.cu579-805]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/intranode.cu#L579-L805](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/intranode.cu#L579-L805))

## Internode Kernels

The `internode.cu` file manages RDMA/NVSHMEM communication across multiple nodes using a sophisticated dual-path architecture:

|Function|Template Parameters|Purpose|

|---|---|---|

|`notify_dispatch`|`kLowLatencyMode`, `kNumRDMARanks`|Cross-node metadata exchange|

|`dispatch`|`kLowLatencyMode`, `kNumRDMARanks`, `kCachedMode`, `kNumTMABytesPerWarp`, `kNumDispatchRDMASenderWarps`|Cross-node token routing|

|`combine`|`kLowLatencyMode`, `kNumRDMARanks`, `kCachedMode`, `kNumTMABytesPerWarp`, `kNumCombineRDMASenderWarps`|Cross-node result aggregation|

**Internode Buffer Architecture**

![[Internode Buffer Architecture.png]]

**Key Features:**

- **Dual-Path Communication**: Combines RDMA and NVLink for optimal bandwidth
- **Low-Latency Mode**: Direct GPU-to-GPU RDMA using IBGDA
- **Symmetric/Asymmetric Buffers**: Different buffer patterns for RDMA vs NVLink
- **Warp Roles**: Specialized warp roles (kRDMASender, kRDMAAndNVLForwarder, kNVLReceivers)

**Sources:** [csrc/kernels/internode.cu17-58]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L17-L58](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L17-L58)) [csrc/kernels/internode.cu83-303]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L83-L303](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L83-L303)) [csrc/kernels/internode.cu355-1495]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L355-L1495](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode.cu#L355-L1495))

## Low-Latency Kernels

The `internode_ll.cu` file provides specialized kernels optimized for low-latency inference workloads:

|Function|Template Parameters|Purpose|

|---|---|---|

|`clean_low_latency_buffer`|`kNumThreads`|Buffer cleanup with NVSHMEM barriers|

|`dispatch`|`kUseFP8`, `kUseUE8M0`, `kHidden`|Low-latency token dispatch with FP8|

|`combine`|`kUseLogFMT`, `kHidden`, `kNumMaxTopk`|Result combination with LogFMT compression|

**Low-Latency Execution Phases**

![[Low-Latency Execution Phases.png]]

**Key Optimizations:**

- **Phased Execution**: Separate send/receive phases for pipeline optimization
- **FP8 Precision**: Dynamic FP8 casting with local amax calculation
- **LogFMT Compression**: Logarithmic format for ultra-low precision
- **Zero-Copy Mode**: Direct memory access without intermediate buffers
- **TMA Acceleration**: Tensor Memory Accelerator for SM90 hardware

**Sources:** [csrc/kernels/internode_ll.cu10-37]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode_ll.cu#L10-L37](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode_ll.cu#L10-L37)) [csrc/kernels/internode_ll.cu39-392]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode_ll.cu#L39-L392](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode_ll.cu#L39-L392)) [csrc/kernels/internode_ll.cu394-732]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode_ll.cu#L394-L732](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/internode_ll.cu#L394-L732))

## Runtime Utilities

The `runtime.cu` file provides synchronization and NVSHMEM management utilities:

|Function|Template Parameters|Purpose|

|---|---|---|

|`barrier`|`kNumRanks`|Intranode GPU barrier synchronization|

|`get_unique_id`|-|NVSHMEM unique ID generation|

|`init`|-|NVSHMEM initialization with team creation|

|`alloc/free`|-|NVSHMEM memory management|

|`barrier`|-|Global NVSHMEM barrier|

|`finalize`|-|NVSHMEM cleanup|

**NVSHMEM Team Management**

![[NVSHMEM Team Management.png]]

The runtime manages NVSHMEM teams for low-latency mode, where GPU ranks are grouped by RDMA connectivity for optimized communication patterns.

**Sources:** [csrc/kernels/runtime.cu18-31]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/runtime.cu#L18-L31](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/runtime.cu#L18-L31)) [csrc/kernels/runtime.cu37-89]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/runtime.cu#L37-L89](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/runtime.cu#L37-L89))

## Implementation Details

### Memory Access Patterns

CUDA kernels use specific memory access patterns optimized for different communication modes:

**Sources:** [csrc/kernels/layout.cu18-24]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/layout.cu#L18-L24](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/layout.cu#L18-L24)) [csrc/kernels/api.cuh45-50]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/api.cuh#L45-L50](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/api.cuh#L45-L50)) [csrc/kernels/api.cuh142-153]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/api.cuh#L142-L153](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/api.cuh#L142-L153))

### Synchronization Mechanisms

The kernel system uses multiple synchronization primitives:

|Mechanism|Scope|Implementation|

|---|---|---|

|`__syncthreads()`|Thread block|CUDA built-in|

|`barrier()`|Intranode GPUs|Signal-based|

|`notify_dispatch`|Cross-node|RDMA coordination|

|`cached_notify`|Optimized notify|Cached metadata|

**Sources:** [csrc/kernels/api.cuh10]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/api.cuh#L10-L10](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/api.cuh#L10-L10)) [csrc/kernels/api.cuh85-96]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/api.cuh#L85-L96](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/api.cuh#L85-L96)) [csrc/kernels/layout.cu35-96]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/layout.cu#L35-L96](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/layout.cu#L35-L96))

## Hardware Optimization Features

### SM90 Architecture Support

The kernel system leverages SM90 features for enhanced performance:

- **Cooperative Kernels**: Enable cross-SM synchronization
- **Cluster Dimensions**: Group SMs for coordinated execution
- **TMA (Tensor Memory Accelerator)**: Optimized memory transfers
- **Dynamic Shared Memory**: Configurable shared memory allocation

### Multi-Precision Support

Kernels support multiple data types with specialized code paths:

- **BF16**: Primary data type for training workloads
- **FP8**: Experimental support for inference optimization
- **Scale tensors**: Separate scaling factors for quantized data

### Rank Scalability

The system supports various distributed configurations:

- **Intranode**: 2-8 GPUs per node via NVLink
- **Internode**: 2-16 RDMA ranks across nodes
- **Hybrid**: Combined NVLink + RDMA communication

**Sources:** [csrc/kernels/launch.cuh7-18]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/launch.cuh#L7-L18](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/launch.cuh#L7-L18)) [csrc/kernels/launch.cuh79-83]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/launch.cuh#L79-L83](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/launch.cuh#L79-L83)) [csrc/kernels/launch.cuh62-69]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/launch.cuh#L62-L69](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/kernels/launch.cuh#L62-L69))