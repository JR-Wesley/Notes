---
dateCreated: 2025-08-08
dateModified: 2025-08-08
---
This document covers the C++ runtime system that forms the core implementation of DeepEP. The runtime provides the foundational classes, memory management, and communication orchestration that backs the Python API. For detailed information about CUDA kernel implementations, see [CUDA Kernels]([https://deepwiki.com/deepseek-ai/DeepEP/6.2-cuda-kernels](https://deepwiki.com/deepseek-ai/DeepEP/6.2-cuda-kernels)). For device-side utilities and low-level primitives, see [Device Utilities]([https://deepwiki.com/deepseek-ai/DeepEP/6.3-device-utilities](https://deepwiki.com/deepseek-ai/DeepEP/6.3-device-utilities)).

## Core Classes

The C++ runtime is centered around three primary classes that manage the complete lifecycle of expert-parallel communication.

![[Core Classes.png]]

**Sources:** [csrc/deep_ep.cpp15-82]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L15-L82](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L15-L82)) [csrc/deep_ep.cpp1344-1350]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L1344-L1350](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L1344-L1350)) [csrc/deep_ep.cpp1353-1355]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L1353-L1355](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L1353-L1355))

### Buffer Class

The `Buffer` class serves as the primary runtime object, managing all communication resources and orchestrating data movement between GPUs. It encapsulates:

- **Rank Management**: Global rank (`rank`), RDMA rank (`rdma_rank`), and NVLink rank (`nvl_rank`) for hierarchical communication
- **Memory Allocation**: NVLink buffers (`buffer_ptrs`) and RDMA buffers (`rdma_buffer_ptr`) with alignment guarantees
- **Stream Management**: Dedicated communication stream (`comm_stream`) separate from compute streams
- **Synchronization**: Host-mapped counters (`moe_recv_counter`, `moe_recv_expert_counter`) for CPU-GPU coordination

**Sources:** [csrc/deep_ep.cpp15-82]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L15-L82](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L15-L82))

### Config Class

The `Config` class provides tuning parameters for communication performance, controlling chunk sizes and buffer utilization across different communication modes.

**Sources:** [csrc/deep_ep.cpp1344-1350]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L1344-L1350](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L1344-L1350))

### EventHandle Class

The `EventHandle` class wraps CUDA events for precise stream synchronization, enabling asynchronous communication patterns without blocking compute streams.

**Sources:** [csrc/deep_ep.cpp1353-1355]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L1353-L1355](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L1353-L1355))

## Runtime Lifecycle

The `Buffer` runtime follows a strict three-phase lifecycle with explicit resource management and distributed synchronization.

![[Runtime Lifecycle.png]]

**Sources:** [csrc/deep_ep.cpp15-82]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L15-L82](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L15-L82)) [csrc/deep_ep.cpp185-240]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L185-L240](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L185-L240)) [csrc/deep_ep.cpp143-183]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L143-L183](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L143-L183))

### Construction Phase

During construction, the `Buffer` allocates local memory resources and prepares for distributed coordination:

```

// Memory layout calculation and allocation

int64_t barrier_signal_bytes = NUM_MAX_NVL_PEERS * sizeof(int);

int64_t buffer_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(void*);

// Allocate contiguous block for buffer + metadata

cudaMalloc(&buffer_ptrs[nvl_rank], num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes);

```

**Sources:** [csrc/deep_ep.cpp21-82]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L21-L82](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L21-L82))

### Synchronization Phase

The `sync()` method coordinates distributed initialization by exchanging IPC handles and establishing NVSHMEM connections. This phase transforms the buffer from unavailable to ready for communication.

**Sources:** [csrc/deep_ep.cpp185-240]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L185-L240](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L185-L240))

### Destruction Phase

The `destroy()` method performs careful cleanup with distributed barriers to ensure all ranks complete operations before releasing shared resources.

**Sources:** [csrc/deep_ep.cpp143-183]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L143-L183](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L143-L183))

## Communication Operations

The runtime provides three distinct communication modes, each optimized for different hardware topologies and performance requirements.

![[Communication Operations.png]]

**Sources:** [csrc/deep_ep.cpp242-303]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L242-L303](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L242-L303)) [csrc/deep_ep.cpp305-540]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L305-L540](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L305-L540)) [csrc/deep_ep.cpp653-931]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L653-L931](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L653-L931)) [csrc/deep_ep.cpp1090-1206]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L1090-L1206](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L1090-L1206))

### Dispatch-Combine Pattern

All communication operations follow a two-phase pattern:

1. **Dispatch Phase**: Route tokens to appropriate expert ranks based on expert assignments
2. **Combine Phase**: Aggregate processed results back to original token locations

Each phase involves complex memory management, stream coordination, and distributed synchronization handled by the runtime.

**Sources:** [csrc/deep_ep.cpp305-540]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L305-L540](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L305-L540)) [csrc/deep_ep.cpp542-651]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L542-L651](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L542-L651))

## Memory Management

The runtime implements sophisticated memory management with multiple buffer types and strict alignment requirements.

![[Memory Management.png]]

**Sources:** [csrc/deep_ep.cpp47-82]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L47-L82](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L47-L82))

### Buffer Alignment

All buffers maintain strict alignment requirements (`NUM_BUFFER_ALIGNMENT_BYTES`) to ensure optimal memory access patterns and hardware compatibility.

**Sources:** [csrc/deep_ep.cpp27-28]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L27-L28](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L27-L28))

### Host-Mapped Counters

The runtime uses host-mapped memory for CPU-GPU coordination, allowing the CPU to poll completion status without expensive device synchronization.

**Sources:** [csrc/deep_ep.cpp65-81]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L65-L81](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L65-L81))

## Stream Management

The runtime maintains careful separation between compute and communication streams to enable overlapping computation with data movement.

![[Stream Management.png]]

**Sources:** [csrc/deep_ep.cpp20]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L20-L20](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L20-L20)) [csrc/deep_ep.cpp254-262]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L254-L262](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L254-L262)) [csrc/deep_ep.cpp394-400]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L394-L400](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L394-L400))

### Asynchronous Operation Support

Methods support both synchronous and asynchronous execution modes through the `async` parameter, with optional tensor allocation on communication streams for zero-copy operations.

**Sources:** [csrc/deep_ep.cpp282-300]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L282-L300](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L282-L300)) [csrc/deep_ep.cpp517-536]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L517-L536](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L517-L536))

## Python Bindings

The runtime exposes its functionality to Python through pybind11 bindings that preserve the full C++ interface while providing Pythonic tensor integration.

![[Python Bindings.png]]

**Sources:** [csrc/deep_ep.cpp1341-1381]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L1341-L1381](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L1341-L1381))

The bindings preserve method signatures while automatically handling PyTorch tensor integration, CUDA stream management, and Python object lifetime management. All communication methods return tuples containing result tensors and optional `EventHandle` objects for asynchronous coordination.

**Sources:** [csrc/deep_ep.cpp1357-1378]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L1357-L1378](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L1357-L1378))
