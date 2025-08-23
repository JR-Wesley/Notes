---
dateCreated: 2025-08-08
dateModified: 2025-08-09
---
## Purpose and Scope

This document provides an overview of NCCL (NVIDIA Collective Communications Library), a high-performance library designed for optimized inter-GPU communication in distributed computing environments. NCCL implements standard collective communication primitives and point-to-point operations, supporting both single-node and multi-node GPU clusters across various interconnect technologies.

For detailed information about specific architectural components, see [Architecture]([https://deepwiki.com/NVIDIA/nccl/2-core-architecture](https://deepwiki.com/NVIDIA/nccl/2-core-architecture)). For building and packaging details, see [Building and Packaging]([CUDA Integration | NVIDIA/nccl | DeepWiki](https://deepwiki.com/NVIDIA/nccl/6-cuda-integration)). For plugin system integration, see [Plugin System]([Device-Side Operations | NVIDIA/nccl | DeepWiki](https://deepwiki.com/NVIDIA/nccl/5-device-side-operations)).

## What is NCCL

NCCL (pronounced "Nickel") is a stand-alone library that provides optimized communication primitives for GPU-to-GPU data exchange. The library implements a comprehensive set of collective operations including:

| Operation             | Description                                             |
| --------------------- | ------------------------------------------------------- |
| `ncclAllReduce`       | Combines data from all GPUs using a reduction operation |
| `ncclAllGather`       | Gathers data from all GPUs to all GPUs                  |
| `ncclReduce`          | Combines data from all GPUs to a single GPU             |
| `ncclBroadcast`       | Distributes data from one GPU to all GPUs               |
| `ncclReduceScatter`   | Combines and distributes data across GPUs               |
| `ncclSend`/`ncclRecv` | Point-to-point communication patterns                   |

The library is optimized for various interconnect technologies including PCIe, NVLink, NVswitch, InfiniBand Verbs, and TCP/IP sockets, supporting arbitrary numbers of GPUs in single-process or multi-process (MPI) applications.

**Sources:** [README.md3-7]([https://github.com/NVIDIA/nccl/blob/7c12c627/README.md#L3-L7](https://github.com/NVIDIA/nccl/blob/7c12c627/README.md#L3-L7))

## High-Level Architecture

NCCL follows a layered architecture that abstracts hardware complexity while maximizing performance across different interconnect technologies.

### Core System Architecture

![](Core%20System%20Architecture.png)

**Sources:** [src/init.cc]([https://github.com/NVIDIA/nccl/blob/7c12c627/src/init.cc](https://github.com/NVIDIA/nccl/blob/7c12c627/src/init.cc)) [src/group.cc]([https://github.com/NVIDIA/nccl/blob/7c12c627/src/group.cc](https://github.com/NVIDIA/nccl/blob/7c12c627/src/group.cc)) [src/enqueue.cc]([https://github.com/NVIDIA/nccl/blob/7c12c627/src/enqueue.cc](https://github.com/NVIDIA/nccl/blob/7c12c627/src/enqueue.cc)) [src/graph/topo.cc]([https://github.com/NVIDIA/nccl/blob/7c12c627/src/graph/topo.cc](https://github.com/NVIDIA/nccl/blob/7c12c627/src/graph/topo.cc)) [src/transport/net_ib.cc]([https://github.com/NVIDIA/nccl/blob/7c12c627/src/transport/net_ib.cc](https://github.com/NVIDIA/nccl/blob/7c12c627/src/transport/net_ib.cc)) [src/proxy.cc]([https://github.com/NVIDIA/nccl/blob/7c12c627/src/proxy.cc](https://github.com/NVIDIA/nccl/blob/7c12c627/src/proxy.cc))

## Core Component Interactions

![](Core%20Component%20Interactions.png)

The following diagram illustrates how NCCL's major components interact during a typical collective operation lifecycle:

### Operation Flow and Component Coordination

**Sources:** [src/init.cc]([https://github.com/NVIDIA/nccl/blob/7c12c627/src/init.cc](https://github.com/NVIDIA/nccl/blob/7c12c627/src/init.cc)) [src/group.cc]([https://github.com/NVIDIA/nccl/blob/7c12c627/src/group.cc](https://github.com/NVIDIA/nccl/blob/7c12c627/src/group.cc)) [src/enqueue.cc]([https://github.com/NVIDIA/nccl/blob/7c12c627/src/enqueue.cc](https://github.com/NVIDIA/nccl/blob/7c12c627/src/enqueue.cc)) [src/proxy.cc]([https://github.com/NVIDIA/nccl/blob/7c12c627/src/proxy.cc](https://github.com/NVIDIA/nccl/blob/7c12c627/src/proxy.cc)) [src/device/common.h]([https://github.com/NVIDIA/nccl/blob/7c12c627/src/device/common.h](https://github.com/NVIDIA/nccl/blob/7c12c627/src/device/common.h))

## Key System Capabilities

NCCL provides several critical capabilities that enable high-performance distributed GPU computing:

### Hardware Abstraction and Optimization

- **Topology Discovery**: Automatic detection of GPU and network hardware topology through `ncclTopoGetSystem` in [src/graph/topo.cc]([https://github.com/NVIDIA/nccl/blob/7c12c627/src/graph/topo.cc](https://github.com/NVIDIA/nccl/blob/7c12c627/src/graph/topo.cc))
- **Path Optimization**: Intelligent routing computation via `ncclTopoComputePaths` for optimal bandwidth utilization
- **Algorithm Selection**: Dynamic selection of communication algorithms (ring, tree, NVLS, CollNet) based on hardware capabilities

### Transport Layer Flexibility

- **Multi-transport Support**: Unified interface supporting InfiniBand (`ncclIbInit`), TCP sockets (`ncclSocketInit`), and GPU-direct P2P (`ncclP2pSetup`)
- **Plugin Architecture**: Extensible network plugin system through `ncclNet_t` interface in [src/include/net.h]([https://github.com/NVIDIA/nccl/blob/7c12c627/src/include/net.h](https://github.com/NVIDIA/nccl/blob/7c12c627/src/include/net.h))
- **Proxy Service**: Asynchronous network operation handling via `ncclProxyMain` for CPU offload

### Performance Optimization Features

- **Strong Stream Management**: CUDA Graph capture support through `ncclStrongStreamAcquire` in [src/misc/strongstream.cc]([https://github.com/NVIDIA/nccl/blob/7c12c627/src/misc/strongstream.cc](https://github.com/NVIDIA/nccl/blob/7c12c627/src/misc/strongstream.cc))
- **Memory Management**: Optimized GPU memory registration and IPC mechanisms
- **Kernel Fusion**: Device-side operation batching through `ncclDevKernel` structures

**Sources:** [src/graph/topo.cc]([https://github.com/NVIDIA/nccl/blob/7c12c627/src/graph/topo.cc](https://github.com/NVIDIA/nccl/blob/7c12c627/src/graph/topo.cc)) [src/graph/paths.cc]([https://github.com/NVIDIA/nccl/blob/7c12c627/src/graph/paths.cc](https://github.com/NVIDIA/nccl/blob/7c12c627/src/graph/paths.cc)) [src/include/net.h]([https://github.com/NVIDIA/nccl/blob/7c12c627/src/include/net.h](https://github.com/NVIDIA/nccl/blob/7c12c627/src/include/net.h)) [src/misc/strongstream.cc]([https://github.com/NVIDIA/nccl/blob/7c12c627/src/misc/strongstream.cc](https://github.com/NVIDIA/nccl/blob/7c12c627/src/misc/strongstream.cc)) [src/device/common.h](https://github.com/NVIDIA/nccl/blob/7c12c627/src/device/common.h)
