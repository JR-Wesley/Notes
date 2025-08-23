---
dateCreated: 2025-08-08
dateModified: 2025-08-08
---
This page provides an introduction to DeepEP and guides you through the basic setup and usage concepts. DeepEP is a high-performance communication library designed for expert-parallel workloads and Mixture of Experts (MoE) models, supporting both intranode and internode GPU communication.

For detailed installation instructions, see [Installation]([Installation | deepseek-ai/DeepEP | DeepWiki](https://deepwiki.com/deepseek-ai/DeepEP/2.1-installation)). For information about the build system and compilation options, see [Build System]([https://deepwiki.com/deepseek-ai/DeepEP/2.2-build-system](https://deepwiki.com/deepseek-ai/DeepEP/2.2-build-system)).

## Overview

DeepEP provides a Python interface to optimized CUDA kernels that implement efficient dispatch-combine communication patterns across GPU clusters. The library is designed to handle the complex communication requirements of expert-parallel training and inference workloads.

### Core Components

The system consists of four main layers:

|Layer|Components|Purpose|

|---|---|---|

|**Python Interface**|`deep_ep.Buffer`, `deep_ep.utils`|High-level API and utilities|

|**C++ Runtime**|`deep_ep_cpp.Buffer`, `deep_ep_cpp.Config`, `deep_ep_cpp.EventHandle`|Core implementation and configuration|

|**CUDA Kernels**|`intranode.cu`, `internode.cu`, `internode_ll.cu`, `layout.cu`|Communication primitives|

|**Hardware Layer**|NVLink, RDMA/InfiniBand, NVSHMEM|Physical communication infrastructure|

## System Architecture

The following diagram illustrates how the main code entities relate to the system architecture:

![[DeepWiki/DeepEP/1_Getting_Started/assets/System Architecture.png]]

Sources: [setup.py36-47]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/setup.py#L36-L47](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/setup.py#L36-L47)) [setup.py112-121]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/setup.py#L112-L121](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/setup.py#L112-L121))

## Communication Workflow

DeepEP implements a dispatch-combine pattern optimized for different hardware topologies:

![[Communication Workflow.png]]

Sources: [setup.py36]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/setup.py#L36-L36](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/setup.py#L36-L36)) [setup.py47]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/setup.py#L47-L47](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/setup.py#L47-L47))

## Build Process Overview

The build system automatically detects your environment and configures the appropriate features:

![[Build Process Overview.png]]

Sources: [setup.py15-29]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/setup.py#L15-L29](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/setup.py#L15-L29)) [setup.py42-52]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/setup.py#L42-L52](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/setup.py#L42-L52)) [setup.py53-66]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/setup.py#L53-L66](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/setup.py#L53-L66))

## Quick Start

### Prerequisites

Before installation, ensure you have:

- CUDA Toolkit (version 12+ recommended)
- PyTorch with CUDA support
- Python development headers
- C++ compiler supporting C++17

### Optional Dependencies

- **NVSHMEM**: Required for internode and low-latency communication modes
- **InfiniBand/RDMA**: Required for multi-node deployments

### Basic Installation

```shell

# Clone the repository

git clone [https://github.com/deepseek-ai/DeepEP.git](https://github.com/deepseek-ai/DeepEP.git)

cd DeepEP

  

# Install with default configuration

python [setup.py](http://setup.py/) install

```

The build system will automatically:

- Detect available NVSHMEM installation
- Configure appropriate CUDA architecture targets
- Enable/disable features based on your hardware
- Compile and link the necessary components

### Environment Variables

Key environment variables that control the build:

|Variable|Purpose|Default|

|---|---|---|

|`NVSHMEM_DIR`|Path to NVSHMEM installation|Auto-detected|

|`TORCH_CUDA_ARCH_LIST`|Target GPU architectures|`9.0` or `8.0`|

|`DISABLE_SM90_FEATURES`|Disable H100-specific features|`0`|

|`DISABLE_AGGRESSIVE_PTX_INSTRS`|Disable advanced PTX instructions|`1`|

Sources: [setup.py17-18]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/setup.py#L17-L18](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/setup.py#L17-L18)) [setup.py53-66]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/setup.py#L53-L66](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/setup.py#L53-L66)) [setup.py70-78]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/setup.py#L70-L78](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/setup.py#L70-L78))

## Next Steps

After installation:

1. **Detailed Installation**: See [Installation]([Installation | deepseek-ai/DeepEP | DeepWiki](https://deepwiki.com/deepseek-ai/DeepEP/2.1-installation)) for comprehensive setup instructions
2. **Build Configuration**: See [Build System]([https://deepwiki.com/deepseek-ai/DeepEP/2.2-build-system](https://deepwiki.com/deepseek-ai/DeepEP/2.2-build-system)) for advanced build options
3. **System Architecture**: See [Architecture]([Architecture | deepseek-ai/DeepEP | DeepWiki](https://deepwiki.com/deepseek-ai/DeepEP/3-architecture)) for deep technical details
4. **Python API**: See [Python API]([Python API | deepseek-ai/DeepEP | DeepWiki](https://deepwiki.com/deepseek-ai/DeepEP/4-python-api)) for usage examples
5. **Testing**: See [Testing and Validation]([Testing and Validation | deepseek-ai/DeepEP | DeepWiki](https://deepwiki.com/deepseek-ai/DeepEP/8-testing-and-validation)) for running tests

The following sections provide increasingly detailed information about specific aspects of the system, from high-level architecture to low-level CUDA kernel implementations.

Sources: [setup.py1-126]([https://github.com/deepseek-ai/DeepEP/blob/4b67064d/setup.py#L1-L126](https://github.com/deepseek-ai/DeepEP/blob/4b67064d/setup.py#L1-L126))
