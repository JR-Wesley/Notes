DeepEP is a high-performance communication library designed for expert-parallel workloads in Mixture-of-Experts (MoE) models. The system provides a unified interface for efficient token dispatch and combine operations across different hardware topologies, from single-node NVLink configurations to multi-node RDMA clusters.

  

This page provides a high-level architectural overview of the DeepEP system. For detailed implementation information, see the subsections on [System Overview]([https://deepwiki.com/deepseek-ai/DeepEP/3.1-system-overview](https://deepwiki.com/deepseek-ai/DeepEP/3.1-system-overview)), [Communication Model]([https://deepwiki.com/deepseek-ai/DeepEP/3.2-communication-model](https://deepwiki.com/deepseek-ai/DeepEP/3.2-communication-model)), [Buffer System]([https://deepwiki.com/deepseek-ai/DeepEP/3.3-buffer-system](https://deepwiki.com/deepseek-ai/DeepEP/3.3-buffer-system)), and [Configuration System]([https://deepwiki.com/deepseek-ai/DeepEP/3.4-configuration-system](https://deepwiki.com/deepseek-ai/DeepEP/3.4-configuration-system)).

  

## System Layers

  

DeepEP is organized into four distinct architectural layers that provide efficient expert-parallel communication across different hardware configurations:

  

**DeepEP System Architecture**

![[DeepEP System Architecture.png]]

The system supports three communication modes:

  

- **Intranode**: High-throughput NVLink-based communication within a single node

- **Internode**: RDMA + NVLink communication across multiple nodes

- **Low-latency**: IBGDA-optimized communication for inference workloads

  

Sources: [deep_ep/buffer.py13-28]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/deep_ep/buffer.py#L13-L28](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/deep_ep/buffer.py#L13-L28)) [deep_ep/buffer.py6-9]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/deep_ep/buffer.py#L6-L9](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/deep_ep/buffer.py#L6-L9)) [deep_ep/utils.py33-51]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/deep_ep/utils.py#L33-L51](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/deep_ep/utils.py#L33-L51))

  

## Core Components

  

DeepEP's architecture centers around several key components that work together to provide expert-parallel communication:

  

### Buffer Class

  

The `deep_ep.Buffer` class serves as the primary interface for all communication operations:

  

|Component|Purpose|Key Methods|

|---|---|---|

|`Buffer.__init__`|Initialize communication infrastructure|Setup IPC handles, NVSHMEM coordination|

|`Buffer.dispatch`|Route tokens to experts|`intranode_dispatch`, `internode_dispatch`, `low_latency_dispatch`|

|`Buffer.combine`|Aggregate expert outputs|`intranode_combine`, `internode_combine`, `low_latency_combine`|

|`Buffer.get_dispatch_layout`|Calculate token routing|Layout computation for communication patterns|

  

**Buffer System Integration**

![[Buffer System Integration.png]]

Sources: [deep_ep/buffer.py32-67]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/deep_ep/buffer.py#L32-L67](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/deep_ep/buffer.py#L32-L67)) [deep_ep/buffer.py177-194]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/deep_ep/buffer.py#L177-L194](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/deep_ep/buffer.py#L177-L194)) [deep_ep/buffer.py261-288]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/deep_ep/buffer.py#L261-L288](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/deep_ep/buffer.py#L261-L288))

  

## Communication Pattern

  

DeepEP implements a **dispatch-combine** communication pattern that efficiently handles token routing for MoE workloads:

  

**Dispatch-Combine Flow**

![[Dispatch-Combine Flow.png]]

The system automatically selects the appropriate communication mode based on hardware topology:

  

- Single-node setups use `intranode_dispatch`/`intranode_combine` with NVLink

- Multi-node setups use `internode_dispatch`/`internode_combine` with RDMA+NVLink

- Low-latency workloads use `low_latency_dispatch`/`low_latency_combine` with IBGDA

  

Sources: [deep_ep/buffer.py290-417]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/deep_ep/buffer.py#L290-L417](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/deep_ep/buffer.py#L290-L417)) [deep_ep/buffer.py515-630]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/deep_ep/buffer.py#L515-L630](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/deep_ep/buffer.py#L515-L630))

  

## Hardware Abstraction

  

DeepEP provides a unified interface across different hardware communication mechanisms through a layered abstraction:

  

**Hardware Communication Stack**

![[Hardware Communication Stack.png]]

  

### Configuration System

  

The system uses performance-tuned configurations based on the number of ranks:

  

|Ranks|Dispatch Config|Combine Config|Use Case|

|---|---|---|---|

|2-8|`Config(20, 6-24, 256, 6, 128)`|`Config(20, 4-10, 256, 6, 128)`|Small scale|

|16-32|`Config(20, 32-36, 288, 20-32, 128)`|`Config(20, 1-4, 288, 8-12, 128)`|Medium scale|

|64-160|`Config(20, 20-32, 560-720, 12-32, 128)`|`Config(20, 1-2, 560-720, 8-20, 128)`|Large scale|

  

The `deep_ep_cpp.Config` class contains performance parameters:

  

- `num_sms`: Number of streaming multiprocessors to use

- `num_blocks_per_sm`: CUDA blocks per SM

- `num_threads_per_block`: Threads per CUDA block

- `chunk_size`: Data chunk size for processing

- `num_buffers`: Number of communication buffers

  

Sources: [deep_ep/buffer.py206-231]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/deep_ep/buffer.py#L206-L231](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/deep_ep/buffer.py#L206-L231)) [deep_ep/buffer.py234-259]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/deep_ep/buffer.py#L234-L259](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/deep_ep/buffer.py#L234-L259)) [deep_ep/buffer.py81-100]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/deep_ep/buffer.py#L81-L100](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/deep_ep/buffer.py#L81-L100))

  

## Event Management and Synchronization

  

DeepEP implements sophisticated event management to handle asynchronous operations and stream synchronization:

  

### Event System Architecture

  

![[Event System Architecture.png]]

  

Sources: [deep_ep/buffer.py126-133]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/deep_ep/buffer.py#L126-L133](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/deep_ep/buffer.py#L126-L133)) [csrc/deep_ep.cpp266-281]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L266-L281](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/csrc/deep_ep.cpp#L266-L281)) [deep_ep/utils.py1-50]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/deep_ep/utils.py#L1-L50](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/deep_ep/utils.py#L1-L50))

  

This architecture enables DeepEP to efficiently handle the complex communication patterns required for MoE workloads while maintaining high performance across different hardware configurations and scales.