- deepep metrics

- cuda async

- sccl metrics

  

  

- test deepep dispatcher megatron

- deepep implementation (inter intra ll)

- sccl test

- kernel reduce/gemm/cutlass

- cuda stream/graph

- project quant

  

# a sneak peak

  

第一部分《AI 系统概述》：AI 基础知识和 AI 系统的全栈概述的 AI 系统概述，以及 AI 系统的系统性设计和方法论，主要是整体了解 AI 训练和推理全栈的体系结构内容。

  

- 经典模型演进、模型量化压缩： [https://infrasys-ai.github.io/aisystem-docs/02Hardware01Foundation/02ArchSlim.html#、https://infrasys-ai.github.io/aisystem-docs/02Hardware01Foundation/06BitWidth.html#](https://infrasys-ai.github.io/aisystem-docs/02Hardware01Foundation/02ArchSlim.html#%E3%80%81https://infrasys-ai.github.io/aisystem-docs/02Hardware01Foundation/06BitWidth.html#)

- 轻量化与分布式： [https://infrasys-ai.github.io/aisystem-docs/02Hardware01Foundation/03MobileParallel.html#](https://infrasys-ai.github.io/aisystem-docs/02Hardware01Foundation/03MobileParallel.html#)

- GEMM优化： [https://infrasys-ai.github.io/aisystem-docs/02Hardware01Foundation/05Matrix.html#](https://infrasys-ai.github.io/aisystem-docs/02Hardware01Foundation/05Matrix.html#)

  

- ISA： [https://infrasys-ai.github.io/aisystem-docs/02Hardware02ChipBase/02CPUISA.html#](https://infrasys-ai.github.io/aisystem-docs/02Hardware02ChipBase/02CPUISA.html#)

- [GPU原理]([https://infrasys-ai.github.io/aisystem-docs/02Hardware03GPUBase/01Works.html#](https://infrasys-ai.github.io/aisystem-docs/02Hardware03GPUBase/01Works.html#)) [GPU架构]([https://infrasys-ai.github.io/aisystem-docs/02Hardware03GPUBase/04History.html#](https://infrasys-ai.github.io/aisystem-docs/02Hardware03GPUBase/04History.html#))

- [GPU详解，包括tensor core, 分布式]([https://infrasys-ai.github.io/aisystem-docs/02Hardware04NVIDIA/README.html](https://infrasys-ai.github.io/aisystem-docs/02Hardware04NVIDIA/README.html))

  

  

第二部分《AI 硬件体系架构》：硬核篇介绍 AI 芯片概况，从芯片基础到 AI 芯片的范围都会涉及，芯片设计需要考虑上层 AI 框架的前端、后端编译、以及针对 AI 算法的实现角度等相关技术综合设计符合 AI 范式的芯片架构。

  

第三部分《AI 编程与计算架构》：进阶篇介绍 AI 编译器原理，将站在系统设计的角度，思考在设计现代机器学习系统中需要考虑的编译器问题，特别是针对 AI 计算图的中间表达乃至后端优化。

  

第四部分《AI 推理系统与引擎》：实际应用推理系统与推理引擎，AI 系统领域众多，技术点也非常的多，但本质还是得回归到业务本质，让行业、企业能够真正应用起来，而推理系统涉及一些核心算法是真正在部署与推理端，帮助 AI 业务进行落地。

  

第五部分《AI 框架核心模块》：介绍 AI 框架核心技术，首先介绍任何一个 AI 框架都离不开的自动微分技术，通过自动微分功能后就会产生表示神经网络的图和算子，然后介绍 AI 框架前端的优化，还有最近大模型分布式训练在 AI 框架中的关键技术。

  

  

# AIsys

# Overview

  

## Course Modules

  

The curriculum is organized into five main modules, each covering a critical aspect of AI systems:

  

| Module | Title | Description |

| ------ | ------------------------------ | ------------------------------------------------------------------------- |

| 1 | AI System Introduction | Overview of AI systems and their architectural components |

| 2 | AI Chip Architecture | Hardware fundamentals including CPU, GPU, and specialized AI accelerators |

| 3 | AI Compiler Principles | Traditional and AI-specific compilation techniques and optimizations |

| 4 | AI Inference Systems | Model conversion, optimization, and deployment for inference |

| 5 | AI Framework Core Technologies | Fundamental techniques in AI frameworks like automatic differentiation |

  

  

![[Pasted image 20250820161439.png]]

  

  

  

  

## What is an AI System?

  

> [!note] AIsys arch

> [https://deepwiki.com/chenzomi12/AISystem/2-ai-system-architecture](https://deepwiki.com/chenzomi12/AISystem/2-ai-system-architecture)

  

An AI system is the collection of software and hardware infrastructure that connects AI hardware accelerators to higher-level applications. Similar to how operating systems provide abstractions over computer hardware for traditional applications, AI systems abstract the complexities of specialized AI hardware, providing high-level programming models and tools that allow developers to focus on algorithmic design rather than hardware-specific details.

  

AI systems serve as middleware that handles critical tasks including:

  

- Providing programming languages and APIs for model development

- Translating high-level model descriptions into efficient hardware instructions

- Managing resource allocation and scheduling for AI workloads

- Optimizing execution for performance, efficiency, and scalability

- Facilitating model deployment across diverse hardware environments

  

  

  

  

# Distributed Training

  

## Distributed Training Architectures

  

Distributed training methods can be categorized into several main approaches, each addressing different scaling challenges:

  

### Data Parallelism

  

Data parallelism is the most widely used approach for distributed training. In this strategy:

  

- The complete model is replicated on each device

- Each device processes different mini-batches of data (forward and compute loss)

- Gradients are synchronized across devices to maintain model consistency

- Model parameters are updated using the aggregated gradients

  

> [!note] DP

> Data parallelism is ideal when the model can fit in a single device's memory but needs to process large datasets more efficiently.

> Gradient synchronization using `AllReduce`.

  

#### Data Parallelism Variants

  

1. **DP (Data Parallelism)** - Single-process, multi-threaded implementation where data is distributed across devices, but limited by Python GIL

2. **DDP (Distributed Data Parallel)** - Multi-process implementation with optimized communication and better scaling across machines

3. **FSDP (Fully Sharded Data Parallel)** - Memory-optimized variant where model parameters are sharded across devices

4. **Asynchronous Data Parallelism** - Workers update parameters independently without waiting for synchronization

  

### Model Parallelism

  

Model parallelism partitions a model across multiple devices when the model is too large to fit in a single device's memory. There are two primary approaches:

  

1. **Pipeline Parallelism** - Divides the model by layers across devices, with activations flowing between devices in a pipelined fashion

2. **Tensor Parallelism** - Splits individual operations (like matrix multiplications) across devices

### Hybrid Parallelism

  

Hybrid parallelism combines multiple parallelism strategies to optimize training efficiency and resource utilization. Common combinations include:

  

- **3D Parallelism** - Combines data, pipeline, and tensor parallelism

- **ZeRO-Powered Data Parallelism** - Combines data parallelism with memory optimizations like parameter sharding

  

Hybrid approaches are particularly valuable for training very large models like transformers that require both model and data parallelism.

## PyTorch Distributed Architecture

  

PyTorch provides a comprehensive framework for distributed training through multiple components:

### Key Components

  

1. **Distributed Data Parallel (DDP)** - Implements efficient data parallel training with optimized gradient communication

2. **RPC-based Distributed Training** - Supports general training structures that don't fit data parallelism, like pipeline parallelism

3. **C10d Communication Library** - Provides low-level tensor communication primitives (collectives and point-to-point)

  

## DDP Implementation Detail

  

Distributed Data Parallel (DDP) is the most widely used distributed training approach in PyTorch. Let's examine its implementation details:

  

TOADD: **Diagram: DDP Implementation Workflow**

  

### Computation-Communication Overlap

  

One of DDP's key optimizations is overlapping gradient computation with communication:

  

Key aspects of the implementation:

  

1. **Parameter Bucketing** - DDP organizes parameters into buckets for efficient communication

2. **Autograd Hooks** - Registers hooks on parameters to trigger communication when gradients are ready

3. **Asynchronous AllReduce** - Starts communication for a bucket as soon as all its gradients are computed

4. **Buffer Synchronization** - Ensures consistent non-parameter states (like BatchNorm statistics)

  

### DDP Data Loading

  

Efficient data loading is crucial for distributed training. DDP uses:

  

The key components are:

  

1. **DistributedSampler** - Partitions the dataset among workers based on their rank

2. **MultiProcessingDataLoaderIter** - Manages worker processes for parallel data loading

3. **Pin Memory** - Optimizes data transfer to GPU memory

4. **Prefetching** - Loads the next batch while the current one is being processed

## Elastic Training

  

Elastic training extends distributed training with the ability to adapt to dynamic changes in resource availability and handle failures gracefully.

### Elastic Agent

  

The Elastic Agent is the control plane responsible for:

  

1. **Worker Process Management** - Starting, monitoring, and restarting worker processes

2. **Fault Detection** - Identifying failed or unhealthy workers

3. **Resource Adaptation** - Responding to membership changes by restarting workers

### Rendezvous Mechanism

  

The Rendezvous mechanism provides distributed synchronization and node discovery:

  

1. **Barrier Operation** - Blocks until reaching minimum node count or timeout

2. **Rank Assignment** - Assigns unique ranks to participating nodes

3. **Consistent State** - Ensures all members agree on job membership and roles

4. **Shared Key-Value Store** - Facilitates information exchange for job initialization

  

Rendezvous follows a state machine with phases:

  

- Version Counter → Active Version → Setup → Join Phase → Confirm Phase → Final State

  

### Implementation Example

  

Elastic training in PyTorch can be launched using `torchrun` with:

  

```

torchrun --nnodes=MIN_SIZE:MAX_SIZE \

--nproc-per-node=TRAINERS_PER_NODE \

--max-restarts=NUM_ALLOWED_FAILURES \

--rdzv-id=JOB_ID \

--rdzv-backend=c10d \

--rdzv-endpoint=HOST_NODE_ADDR \

YOUR_TRAINING_SCRIPT.py

```

  

The training script must implement checkpoint saving and loading to support restarts:

  

```

def main():

args = parse_args(sys.argv[1:])

state = load_checkpoint(args.checkpoint_path)

initialize(state)

  

torch.distributed.init_process_group(backend=args.backend)

  

for i in range(state.epoch, state.total_num_epochs):

for batch in iter(state.dataset):

train(batch, state.model)

  

state.epoch += 1

save_checkpoint(state)

```

  

## Asynchronous Data Parallelism

  

While synchronous data parallelism (like DDP) ensures model consistency, asynchronous data parallelism offers potential benefits in heterogeneous environments:

  

Key characteristics:

  

1. Devices update parameters independently without waiting for others

2. Faster devices can process more batches

3. May lead to parameter inconsistency but can increase hardware utilization

4. Suitable for heterogeneous computing environments

## Performance Considerations

  

When implementing distributed training, several performance factors should be considered:

  

|Factor|Description|Optimization Techniques|

|---|---|---|

|Communication Overhead|Time spent synchronizing gradients or activations|Gradient compression, bucketing, using high-speed interconnects|

|Computation-Communication Overlap|Ability to perform communication while computing|Asynchronous communication operations, proper bucketing|

|Batch Size|Total batch size across all devices|Linear scaling with warmup, LARS/LAMB optimizers|

|Memory Efficiency|Maximizing model size that can be trained|Parameter sharding, activation checkpointing, mixed precision|

|Scaling Efficiency|How performance scales with more devices|Optimized communication algorithms (Ring-AllReduce)|

  

For DDP specifically, using profiling tools reveals that computation-communication overlap can significantly improve efficiency:

  

```

| Configuration | GPU Summary |

| --- | --- |

| Number of Worker(s): 2 | Name: Tesla V100-SXM2-16GB |

| Device Type: GPU | Compute Capability: 7.0 |

```

  

## Summary

  

Distributed training is essential for training large models and scaling AI workloads. Key approaches include:

  

1. **Data Parallelism** - Most common and suitable when the model fits on a single device

2. **Model Parallelism** - Used when model is too large for a single device

3. **Hybrid Parallelism** - Combining strategies for optimal performance

4. **Elastic Training** - Adding fault tolerance and dynamic resource adaptation

  

Each approach has specific implementation details and performance considerations. PyTorch's distributed training ecosystem provides a comprehensive set of tools for implementing these approaches efficiently.

# Data Parallelism

  

## Introduction to Data Parallelism

****

> [!note] DP

Data parallelism is a distributed training approach that divides the training dataset into smaller subsets and processes them concurrently across multiple computational nodes. Each node maintains a complete copy of the model but processes different data subsets. After each training iteration, gradients from all nodes are aggregated to update the model parameters consistently across all nodes.

  

Data parallelism can be classified in several ways:

  

1. By synchronization method: **Synchronous** vs **Asynchronous** data parallelism

2. By implementation approach: **Data Parallel (DP)**, **Distributed Data Parallel (DDP)**, **Fully Sharded Data Parallel (FSDP)**, and other variants

  

## Data Parallel (DP) vs Distributed Data Parallel (DDP)

  

### Data Parallel (DP)

  

PyTorch's `torch.nn.DataParallel` implements the basic data parallelism approach. DP operates on a single process with multiple threads, distributing data across multiple GPUs on a single machine.

  

The workflow is as follows:

  

1. **Forward pass**: The mini-batch is split evenly across GPUs

2. **Loss calculation and backward pass**: Each GPU calculates loss and computes gradients

3. **Gradient aggregation**: Gradients are transferred to a single GPU for accumulation

4. **Parameter update**: The model parameters and optimizer state are updated on one GPU and then copied back to all GPUs

  

**Limitations of DP**:

  

- Python's Global Interpreter Lock (GIL) limits multi-threading performance

- Gradient accumulation and parameter updates on a single GPU create imbalanced GPU utilization

- Inefficient for small batch sizes due to communication overhead

  

### Distributed Data Parallel (DDP)

  

PyTorch's `torch.nn.parallel.DistributedDataParallel` enhances data parallelism with several optimizations. DDP uses a multi-process approach, with each process managing a single GPU, which circumvents Python's GIL limitations.

  

Key features of DDP:

  

1. **Multi-process implementation**: Avoids GIL limitations and supports multi-node training

2. **Optimized communication**: Uses Ring AllReduce algorithm for efficient gradient aggregation

3. **Computation-communication overlap**: Begins gradient communication as soon as partial gradients are available

4. **Balanced GPU load**: All GPUs perform the same operations

  

## DDP Implementation Analysis

  

Let's examine a simplified example of DDP implementation in PyTorch:

  

```

# Initialize process group

dist.init_process_group("nccl", rank=rank, world_size=world_size)

# Create local model

model = nn.Linear(10, 10).to(rank)

# Construct DDP model

ddp_model = DDP(model, device_ids=[rank])

# Define loss function and optimizer

loss_fn = nn.MSELoss()

optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

  

# Forward pass

outputs = ddp_model(torch.randn(20, 10).to(rank))

labels = torch.randn(20, 10).to(rank)

# Backward pass

loss_fn(outputs, labels).backward()

# Update parameters

optimizer.step()

```

  

### Forward Propagation and Model Consistency

  

DDP ensures model consistency by synchronizing model parameters and buffers across all processes. This happens during initialization and at the beginning of each forward pass.

  

In PyTorch, this synchronization is handled by the `_sync_module_states` method:

  

During initialization, DDP synchronizes all model parameters to ensure each process starts with the same model:

  

```

_sync_module_states(

module=self.module,

process_group=self.process_group,

broadcast_bucket_size=self.broadcast_bucket_size,

src=0,

params_and_buffers_to_ignore=self.parameters_to_ignore,

)

```

  

Before each forward pass, DDP synchronizes the model buffers (like BatchNorm statistics) to maintain consistency:

  

```

_sync_module_states(

module=self.module,

process_group=self.process_group,

broadcast_bucket_size=self.broadcast_bucket_size,

src=0,

params_and_buffers_to_ignore=self.parameters_to_ignore,

)

```

  

### Computation-Communication Overlap

  

One of DDP's key optimizations is overlapping gradient computation with communication. Instead of waiting for all gradients to be computed before starting communication, DDP initiates communication as soon as individual gradients become available.

  

This is implemented using PyTorch's autograd hooks system. The process works as follows:

  

1. During initialization, DDP organizes parameters into "buckets" for efficient communication

2. DDP registers autograd hooks for each parameter

3. When a parameter's gradient is computed, its hook is triggered

4. When all parameters in a bucket have gradients ready, an AllReduce operation starts for that bucket

5. Computation continues while communication happens in the background

  

### Distributed Data Loading

  

Efficient data loading is crucial for distributed training. PyTorch provides specialized tools for this purpose:

  

1. `torch.utils.data.distributed.DistributedSampler`: Partitions the dataset among processes

2. `torch.utils.data.DataLoader` with `num_workers`: Enables multi-process data loading

  

The `DistributedSampler` divides the dataset so that each process gets a distinct subset:

  

```

# Partition the dataset

indices = indices[self.rank:self.total_size:self.num_replicas]

```

  

The worker processes fetch data based on these indices:

  

```

# Worker fetches data

data = fetcher.fetch(index)

```

  

## Asynchronous Data Parallelism

  

While synchronous data parallelism (like DDP) ensures all processes stay in sync, asynchronous data parallelism allows processes to progress at their own pace.

  

### Key characteristics of asynchronous data parallelism:

  

1. **Independent progress**: Each node works at its own pace

2. **Parameter updates**: Nodes can update parameters as soon as they complete their computation

3. **Improved GPU utilization**: Faster nodes are not slowed down by slower ones

4. **Potential convergence issues**: Model may converge differently due to stale gradients

  

The workflow is as follows:

  

1. **Forward pass**: Each GPU processes its data independently

2. **Backward pass**: Each GPU computes gradients independently

3. **Parameter update**: Each GPU sends its gradients to a parameter server or master process

4. **Parameter retrieval**: GPUs get the latest parameters for the next iteration

  

Sources: [05Framework/04Parallel/02DataParallel.md:421-447]

  

## Elastic Data Parallelism

  

Elastic training extends distributed training to handle dynamic resources and fault tolerance. PyTorch provides this capability through the Torchelastic component.

  

### Elastic Agent

  

The Elastic Agent manages the worker processes and handles:

  

1. **Process management**: Starts and monitors worker processes

2. **Fault recovery**: Detects and recovers from worker failures

3. **Elastic scaling**: Adjusts to changes in available resources

  

### Rendezvous Mechanism

  

Rendezvous is a key component that enables node discovery and synchronization:

  

1. **Barrier functionality**: Ensures a minimum number of nodes are available

2. **Consistency**: Guarantees all nodes have the same view of membership

3. **Fault tolerance**: Handles node failures during discovery

4. **Dynamic membership**: Supports nodes joining and leaving

  

### Elastic Data Parallel Implementation

  

Implementing elastic data parallel training with PyTorch involves:

  

1. Setting up checkpointing logic to handle restarts

2. Using `torchrun` to launch the training job

3. Implementing elastic-friendly training code

  

Example launch command:

  

```

torchrun --nnodes=MIN_SIZE:MAX_SIZE \

--nproc-per-node=TRAINERS_PER_NODE \

--max-restarts=NUM_ALLOWED_FAILURES_OR_MEMBERSHIP_CHANGES \

--rdzv-id=JOB_ID \

--rdzv-backend=c10d \

--rdzv-endpoint=HOST_NODE_ADDR \

YOUR_TRAINING_SCRIPT.py

```

  

Training script example:

  

```

def main():

args = parse_args(sys.argv[1:])

state = load_checkpoint(args.checkpoint_path)

initialize(state)

  

torch.distributed.init_process_group(backend=args.backend)

  

for i in range(state.epoch, state.total_num_epochs):

for batch in iter(state.dataset):

train(batch, state.model)

  

state.epoch += 1

save_checkpoint(state)

```

  

## Performance Analysis of DDP

  

Performance analysis of DDP using PyTorch's profiler demonstrates the effectiveness of computation-communication overlap. The `torch.profiler.profile` tool can provide detailed insights into the execution timeline.

  

Key observations from profiling:

  

1. Communication operations overlap with backward computation

2. Almost no wait time for communication after backward pass completes

3. Bucketization of parameters optimizes communication efficiency

  

Sources: [05Framework/04Parallel/02DataParallel.md:401-418]

  

## Summary

  

- Data parallelism divides the dataset across multiple computing devices, with each device maintaining a complete model copy.

- Distributed Data Parallel (DDP) is superior to basic Data Parallel (DP) due to its multi-process implementation, optimized communication, and computation-communication overlap.

- Asynchronous data parallelism allows nodes to progress independently but may introduce convergence challenges due to parameter staleness.

- Elastic data parallelism adds fault tolerance and dynamic resource scaling through components like Elastic Agent and the Rendezvous mechanism.

- Performance analysis shows that DDP's computation-communication overlap significantly improves training efficiency.

  

# Model Parallelism

  

Model parallelism is a distributed training technique that partitions model parameters across multiple devices to overcome memory limitations when working with large neural networks. Unlike data parallelism which copies the entire model to each device and splits the data, model parallelism splits the model itself. This approach is particularly crucial for modern large language models and transformer architectures whose parameter count exceeds the memory capacity of individual accelerators. For information about data parallelism approaches, see [Data Parallelism]([https://deepwiki.com/chenzomi12/AISystem/5.1-data-parallelism](https://deepwiki.com/chenzomi12/AISystem/5.1-data-parallelism)).

  

## Types of Model Parallelism

  

Model parallelism can be broadly categorized into two main strategies: tensor parallelism and pipeline parallelism. Each addresses different challenges in distributed model training.

  

### Tensor Parallelism

  

Tensor parallelism involves decomposing tensor operations into multiple sub-tensor operations that can be executed in parallel across different devices. This approach is particularly effective for matrix multiplication operations, which are fundamental to neural network computations.

  

  

In tensor parallelism, a matrix multiplication operation is split by partitioning one of the matrices along its columns. Each device holds a complete copy of the input tensor X but only a portion of matrix A. After multiplication, an all-gather communication primitive combines the partial results to form the complete output.

### Pipeline Parallelism

  

Pipeline parallelism divides the model's layers into sequential stages, with each stage assigned to a different device. This approach is suitable for deep models with many sequential layers.

  

In pipeline parallelism, a deep neural network is divided into segments that are placed on different devices. Each device processes its assigned layers and passes the activations to the next device in the pipeline. To reduce idle time (bubbles), inputs are typically split into micro-batches.

  

## Communication Primitives for Model Parallelism

  

Effective model parallelism relies heavily on efficient communication between devices. Several key communication primitives support this distributed computation paradigm.

  

These communication primitives are essential building blocks for implementing model parallelism strategies:

  

| Primitive | Description | Primary Use |

| -------------- | ----------------------------------------------------- | -------------------- |

| All-gather | Collects data from all devices into a complete tensor | Tensor Parallelism |

| Reduce-scatter | Performs reduction operation then distributes results | Tensor Parallelism |

| Point-to-Point | Direct communication between specific devices | Pipeline Parallelism |

| Broadcast | Sends same data from one device to many devices | Both |

  

## Hardware Support for Model Parallelism

  

Modern GPU architectures have evolved to better support model parallelism with specialized interconnect technologies.

  

### NVLink and NVSwitch

  

NVIDIA's NVLink technology facilitates high-bandwidth, low-latency connections between GPUs, enabling efficient model parallelism implementations. NVLink has evolved through multiple generations:

  

|NVLink Generation|First Appearance|Bandwidth|Key Features|

|---|---|---|---|

|NVLink 1.0|Pascal (2016)|160 GB/s|First generation, 5x PCIe bandwidth|

|NVLink 2.0|Volta (2017)|Higher|6 links per GPU (vs 4 in 1.0)|

|NVLink 3.0|Ampere (2020)|Further improved|Enhanced multi-node scaling|

|NVLink 4.0|Hopper (2022)|Significantly higher|Support for up to 256 GPUs|

|NVLink 5.0|Blackwell (2024)|1.8 TB/s per GPU|Support for up to 576 GPUs|

  

NVSwitch complements NVLink by enabling an all-to-all GPU connectivity topology, which is critical for efficient tensor parallelism where multiple GPUs need to frequently exchange partial results.

  

## Model Parallelism vs. Data Parallelism

  

Understanding when to use model parallelism versus data parallelism is crucial for efficient distributed training.

  

|Aspect|Data Parallelism|Model Parallelism|

|---|---|---|

|Partitioning|Divides data batch across devices|Divides model parameters across devices|

|Model Size|Limited to single device memory|Can exceed single device memory|

|Communication|Gradient synchronization (AllReduce)|Activations/gradients between layers|

|Scaling Efficiency|Near-linear with batch size|Depends on model architecture|

|Memory Utilization|High duplication (model copied)|Efficient for parameters (no duplication)|

|Ideal Use Case|Small/medium models, large datasets|Large models that don't fit in device memory|

  

## Tensor Parallelism Implementation

  

Tensor parallelism is particularly effective for distributed training of transformer-based models, as it can parallelize the computation of large matrix multiplications found in attention mechanisms and feed-forward networks.

  

In this example of tensor parallelism:

  

1. Matrix A is split column-wise across devices

2. Input X is replicated on all devices

3. Each device computes a partial matrix multiplication

4. All-Gather communication collects partial results

5. The complete output Y is assembled from partial results

  

This approach is commonly used in frameworks like NVIDIA's Megatron-LM for training large language models.

  

## Pipeline Parallelism Implementation

  

Pipeline parallelism divides models sequentially across devices, allowing efficient training of very deep networks that wouldn't fit on a single device.

  

In pipeline parallelism:

  

1. The model is divided into sequential stages across devices

2. Input batches are further divided into micro-batches

3. Each device processes its stage for a micro-batch, then passes results to the next device

4. This creates a pipeline where multiple micro-batches are processed simultaneously at different stages

5. The timeline shows how micro-batches flow through the system, with both forward (F) and backward (B) passes

  

The empty spaces in the timeline represent "bubbles" where devices are idle. To reduce these bubbles and improve device utilization, techniques like gradient accumulation and micro-batch scheduling are used.

  

## Computing Considerations for Model Parallelism

  

Successfully implementing model parallelism requires attention to several key aspects:

  

### Inter-Device Communication

  

The efficiency of model parallelism heavily depends on the communication bandwidth between devices. High-speed interconnects like NVLink significantly reduce the overhead of transferring activations and gradients between devices.

  

### Memory Optimization

  

Even with model parallelism, memory efficiency remains critical:

  

|Technique|Description|Application|

|---|---|---|

|Activation Checkpointing|Store only selected activations; recompute others during backward pass|Reduces memory at cost of computation|

|Mixed Precision Training|Use lower precision (FP16/BF16) for most operations|Reduces memory usage and speeds computation|

|Gradient Accumulation|Update parameters after multiple micro-batches|Complements pipeline parallelism|

|Offloading|Move parameters/optimizer states to CPU when not in use|Extends effective memory for very large models|

  

## Model Parallelism in AI System Design

  

When designing AI systems that leverage model parallelism, several architectural considerations become important:

  

1. **Hardware selection**: GPUs with high-bandwidth interconnects (NVLink) are essential for efficient tensor parallelism.

2. **Network topology**: Network design impacts the efficiency of communication primitives; all-to-all networks benefit tensor parallelism while ring topologies may be sufficient for pipeline parallelism.

3. **Software frameworks**: Libraries like PyTorch's distributed module provide primitives for implementing model parallelism strategies.

4. **Model architecture**: Some neural network architectures are more amenable to parallelization than others. Transformer-based models are particularly well-suited for tensor parallelism due to their large matrix operations.

5. **Heterogeneous computing**: Combining model parallelism with other techniques like offloading to CPU memory can further expand the size of trainable models.

  

## Hybrid Parallelism Approaches

  

In practice, model parallelism is often combined with data parallelism to create hybrid parallelism strategies. This approach leverages the strengths of both techniques to maximize training efficiency.

  

In hybrid parallelism:

  

1. Data parallelism handles batch splitting

2. Tensor parallelism handles large operators (like matrix multiplications)

3. Pipeline parallelism handles sequential model partitioning

  

This comprehensive approach allows training models that would be impossible to train with any single technique alone, while maintaining reasonable training efficiency.

  

## Conclusion

  

Model parallelism is a crucial technique for training large neural networks that exceed the memory capacity of individual accelerators. By partitioning models across multiple devices through tensor parallelism, pipeline parallelism, or combinations of these approaches, researchers and practitioners can train increasingly complex models. The continued development of hardware interconnects like NVLink and software frameworks with built-in support for distributed training has made model parallelism more accessible and efficient, enabling breakthroughs in large language models and other AI applications.

  

When designing AI systems that require model parallelism, careful consideration of hardware capabilities, network topology, and model architecture is essential to achieve optimal performance and training efficiency.

  

  

  

  

  

# Elastic Training TODO

  

  

  

  

  

  

  

  

  

  

  

  

# Pytorch DP的使用

  

### 一、怎么理解DP？

  

#### 1. **DP (Data Parallelism)**

  

指的是 **数据并行**：将一个 batch 的数据拆分成多个子 batch，分发到不同的 GPU 上进行前向和反向计算，最后在主 GPU 上汇总梯度并更新模型参数。

  

#### 2. **Single-process, multi-threaded implementation**

  

`DataParallel` 是 **单进程、多线程** 的实现。

  

- 所有 GPU 的计算都在 **同一个 Python 进程** 中进行。

- 每个 GPU 的计算由一个线程驱动（通过 Python 的 threading 模块）。

  

#### 3. **Data is distributed across devices**

  

这是数据并行的核心：输入数据被 `DataParallel` 自动 split 到多个 GPU 上，每个 GPU 拥有模型的完整副本，独立计算。

  

#### 4. **Limited by Python GIL**

  

这是关键限制：**Python 的全局解释器锁（GIL）** 会阻止多个线程真正并行执行 Python 代码。

  

- 虽然 PyTorch 的底层计算（如 CUDA kernel）是 C++ 实现的，可以绕过 GIL 并真正并行。

- 但在主进程中，主线程仍需负责协调、拼接梯度、同步操作等，这些 Python 层的操作受 GIL 限制，导致 CPU 成为瓶颈，尤其在模型较大或数据较复杂时。

  

---

  

### 二、`DataParallel` 的主要问题

  

|问题|说明|

|---|---|

|**GIL 瓶颈**|多线程受 GIL 限制，无法充分利用多核 CPU，影响整体吞吐|

|**主 GPU 显存压力大**|梯度汇总、参数更新都在主 GPU（device 0）上完成，导致其显存占用远高于其他 GPU|

|**扩展性差**|通常只适用于 1~4 个 GPU，跨节点不支持|

|**性能不稳定**|在大模型或复杂数据 pipeline 下容易出现负载不均衡|

  

---

  

### 三、✅ 推荐使用 `DistributedDataParallel` (DDP)

  

现代 PyTorch 推荐使用 **`torch.nn.parallel.DistributedDataParallel`（DDP）**，它是：

  

- **多进程、每个进程一个 GPU**

- 每个 GPU 运行在独立的 Python 进程中，**绕过 GIL**

- 使用分布式通信后端（如 NCCL、Gloo）进行梯度 All-Reduce

- 显存使用更均衡，性能更好，扩展性强（支持多机多卡）

  

#### ✅ 使用 DDP 的基本流程：

  

```python

import torch

import torch.distributed as dist

import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data.distributed import DistributedSampler

  

def train(rank, world_size):

# 1. 初始化进程组

dist.init_process_group("nccl", rank=rank, world_size=world_size)

  

# 2. 构建模型并放到对应 GPU

model = YourModel().to(rank)

ddp_model = DDP(model, device_ids=[rank])

  

# 3. 使用 DistributedSampler

dataset = YourDataset()

sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

dataloader = DataLoader(dataset, batch_size=..., sampler=sampler)

  

# 4. 训练循环

optimizer = torch.optim.Adam(ddp_model.parameters())

for data, target in dataloader:

data, target = [data.to](http://data.to/)(rank), [target.to](http://target.to/)(rank)

optimizer.zero_grad()

output = ddp_model(data)

loss = loss_fn(output, target)

loss.backward()

optimizer.step()

  

def main():

world_size = 4 # 使用 4 个 GPU

mp.spawn(train, args=(world_size,), nprocs=world_size)

  

if __name__ == "__main__":

main()

```

  

---

  

### 四、DDP 相比 DP 的优势

  

|特性|DataParallel (DP)|DistributedDataParallel (DDP)|

|---|---|---|

|进程模型|单进程多线程|多进程（每个 GPU 一个进程）|

|GIL 影响|受限|完全绕过|

|显存使用|主 GPU 压力大|各 GPU 均衡|

|性能|一般，易瓶颈|高，接近线性加速|

|扩展性|单机多卡|支持单机/多机多卡|

|易用性|简单（API 简洁）|稍复杂（需初始化进程组）|

  

---

  

### 五、什么时候还能用 `DataParallel`？

  

虽然 DDP 是主流，但 `DataParallel` 仍有适用场景：

  

- **快速原型开发**：单机单节点、GPU 数量少（≤2），想快速测试。

- **资源受限**：无法使用 `torchrun` 或 `mp.spawn` 的环境。

- **模型较小、数据 pipeline 简单**：GIL 影响不明显。

  

但 **正式训练、大模型、高性能需求场景，一律推荐 DDP**。

  

随着模型越来越大，还有更高级的并行策略：

  

- **Fully Sharded Data Parallel (FSDP)**：分片模型参数、梯度、优化器状态，极大减少显存占用（适合大模型）

- **DeepSpeed**：微软推出，支持 ZeRO 优化、混合精度、offload 等

- **Megatron-LM**：NVIDIA 推出的大模型训练框架，支持 tensor parallelism