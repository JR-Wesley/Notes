This page covers the core C++ implementation layer of DeepEP, including the main `Buffer` class, configuration system, event management, and Python bindings via pybind11. This layer serves as the runtime foundation that manages memory, coordinates communication operations, and provides the interface between Python and CUDA kernel implementations.

  

For details about the specific CUDA kernel implementations, see [6.2]([CUDA Kernels | deepseek-ai/DeepEP | DeepWiki](https://deepwiki.com/deepseek-ai/DeepEP/6.2-cuda-kernels)). For hardware integration specifics, see [7.1]([https://deepwiki.com/deepseek-ai/DeepEP/7.1-nvshmem-integration](https://deepwiki.com/deepseek-ai/DeepEP/7.1-nvshmem-integration)).

  

## Core Architecture Overview

  

The core implementation consists of three primary classes that work together to provide the DeepEP runtime system:

![[Core Architecture Overview.png]]

**Sources:** [csrc/deep_ep.hpp23-166]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.hpp#L23-L166](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.hpp#L23-L166)) [csrc/deep_ep.cpp1341-1381]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L1341-L1381](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L1341-L1381))

  

## Buffer Class Implementation

  

The `Buffer` class is the central component that manages all communication operations and memory resources. It maintains both NVLink-based intranode buffers and NVSHMEM-based internode buffers.

  

### Memory Management Architecture

![[Memory Management Architecture.png]]

**Sources:** [csrc/deep_ep.hpp25-78]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.hpp#L25-L78](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.hpp#L25-L78)) [csrc/deep_ep.cpp15-82]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L15-L82](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L15-L82))

  

### Core Methods and Lifecycle

  

The `Buffer` class provides methods organized into several categories:

  

|Method Category|Key Methods|Purpose|

|---|---|---|

|**Initialization**|`Buffer()`, `sync()`, `destroy()`|Setup and teardown|

|**Layout Planning**|`get_dispatch_layout()`|Token routing calculation|

|**Intranode Ops**|`intranode_dispatch()`, `intranode_combine()`|NVLink communication|

|**Internode Ops**|`internode_dispatch()`, `internode_combine()`|NVSHMEM communication|

|**Low-Latency Ops**|`low_latency_dispatch()`, `low_latency_combine()`|Inference-optimized paths|

|**Utilities**|`get_local_buffer_tensor()`, `get_comm_stream()`|Resource access|

  

**Sources:** [csrc/deep_ep.hpp80-164]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.hpp#L80-L164](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.hpp#L80-L164)) [csrc/deep_ep.cpp84-1329]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L84-L1329](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L84-L1329))

  

### Runtime State Management

![[Runtime State Management.png]]

**Sources:** [csrc/deep_ep.cpp84-183]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L84-L183](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L84-L183)) [csrc/deep_ep.cpp185-240]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L185-L240](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L185-L240))

  

## Configuration System

![[Configuration System.png]]

The `Config` class encapsulates performance tuning parameters for communication operations:

  

### Configuration Parameters

  

**Sources:** [csrc/deep_ep.cpp1344-1350]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L1344-L1350](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L1344-L1350))

  

## Event Management

![[Event Management.png]]

The `EventHandle` class provides CUDA event synchronization capabilities:

  

### Event Handling Flow

  

**Sources:** [csrc/deep_ep.cpp1353-1355]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L1353-L1355](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L1353-L1355))

  

## Python Bindings Architecture

  

The pybind11 integration exposes the C++ classes to Python with full method binding:

  

### Binding Structure

![[Binding Structure.png]]

**Sources:** [csrc/deep_ep.cpp1341-1381]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L1341-L1381](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L1341-L1381))

  

## Runtime Lifecycle Management

  

The complete runtime lifecycle involves careful coordination of memory resources and synchronization:

![[Runtime Lifecycle Management.png]]

### Initialization and Synchronization Flow

  

**Sources:** [csrc/deep_ep.cpp15-82]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L15-L82](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L15-L82)) [csrc/deep_ep.cpp185-240]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L185-L240](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L185-L240)) [csrc/deep_ep.cpp143-183]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L143-L183](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L143-L183))

  

### Memory Alignment and Validation

  

The implementation enforces strict alignment requirements and performs comprehensive validation:

  

- **Buffer Alignment**: All buffers must be aligned to `NUM_BUFFER_ALIGNMENT_BYTES`

- **Token Size Constraints**: Hidden dimensions must be multiples of `sizeof(int4)` for vectorized operations

- **Rank Validation**: Ensures proper mapping between global ranks, RDMA ranks, and NVLink ranks

- **Stream Management**: Maintains separate communication and compute streams with proper synchronization

  

**Sources:** [csrc/deep_ep.cpp27-32]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L27-L32](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L27-L32)) [csrc/deep_ep.cpp336-338]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L336-L338](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L336-L338)) [csrc/deep_ep.cpp495-505]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L495-L505](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L495-L505))