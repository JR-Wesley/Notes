---
dateCreated: 2025-08-03
dateModified: 2025-08-03
---
## Review

1. Chapter 1: CPUs are designed to minimize the latency of instruction execution and that GPUs are designed to maximize the throughput of executing instructions.
2. Chapters 2 & 3: the core features of the CUDA programming interface for creating and calling kernels to launch and execute threads.

In the next three chapters we will discuss the architecture of modern GPUs, both the compute architecture and the memory architecture, and the performance optimization techniques stemming from the understanding of this architecture.

1. We will start by showing a high-level, simplified view of the compute architecture and explore the concepts of flexible resource assignment, scheduling of blocks, and occupancy.
2. We will then advance into thread scheduling, latency tolerance, control divergence, and synchronization.
3. We will finish the chapter with a description of the API functions that can be used to query the resources that are available in the GPU and the tools to help estimate the occupancy of the GPU when executing a kernel.

In the following two chapters, we will present the core concepts and programming considerations of the GPU memory architecture. In particular, Chapter 5, Memory Architecture and Data Locality, focuses on the on-chip memory architecture, and Chapter 6, Performance Considerations, briefly covers the off-chip memory architecture then elaborates on various performance considerations of the GPU architecture as a whole.

## 4.1 Architecture of a Modern GPU

![](assets/4%20Compute%20architecture%20and%20scheduling.assets/Fig4.1.png)

Fig. 4.1 shows a high-level, CUDA C programmer’s view of the architecture of a typical CUDA-capable GPU. It is organized into **an array of highly threaded streaming multiprocessors (SMs)**. Each SM has several processing units called **streaming processors or CUDA cores** (hereinafter referred to as just cores for brevity), shown as small tiles inside the SMs in Fig. 4.1, that share control logic and memory resources. For example, the Ampere A100 GPU has 108 SMs with 64 cores each, totaling 6912 cores in the entire GPU.

The SMs also come with different on-chip memory structures collectively labeled as “Memory” in Fig. 4.1. These **on-chip memory** structures will be the topic of Chapter 5, Memory Architecture and Data Locality. GPUs also come with gigabytes of off-chip device memory, referred to as “**Global Memory**” in Fig. 4.1. While older GPUs used graphics double data rate synchronous DRAM, more recent GPUs starting with NVIDIA’s Pascal architecture may use **HBM** (high-bandwidth memory) or HBM2, which consist of DRAM (dynamic random access memory) modules tightly integrated with the GPU in the same package. For brevity we will broadly refer to all these types of memory as DRAM for the rest of the book. We will discuss the most important concepts involved in accessing GPU DRAMs in Chapter 6, Performance Considerations.

> [!note] GPU is organized into an array of SMs and a memory hierarchy.

## 4.2 Block Scheduling

When a kernel is called, the CUDA runtime system launches **a grid of threads** that execute the kernel code. These threads are assigned to SMs **on a block-by-block basis**.

> [!note] All threads in a block are simultaneously assigned to the same SM.

![](assets/4%20Compute%20architecture%20and%20scheduling.assets/Fig4.2.png)

Fig. 4.2 illustrates the assignment of blocks to SMs. Multiple blocks are likely to be simultaneously assigned to the same SM. For example, in Fig. 4.2, three blocks are assigned to each SM. However, blocks need to reserve hardware resources to execute, so only a limited number of blocks can be simultaneously assigned to a given SM. The limit on the number of blocks depends on a variety of factors that are discussed in Section 4.6.

With a limited number of SMs and a limited number of blocks that can be simultaneously assigned to each SM, there is a limit on the total number of blocks that can be simultaneously executing in a CUDA device. Most grids contain many more blocks than this number. To ensure that all blocks in a grid get executed, the runtime system maintains a list of blocks that need to execute and assigns new blocks to SMs when previously assigned blocks complete execution.

> [!note] There is a limited number of SMs and a limited number of blocks that can be simultaneously assigned to each SM

The assignment of threads to SMs on a block-by-block basis guarantees that **threads in the same block are scheduled simultaneously on the same SM**. This guarantee makes it possible for **threads in the same block to interact with each other** in ways that threads across different blocks cannot. 1 This includes barrier synchronization, which is discussed in Section 4.3. It also includes accessing a low-latency shared memory that resides on the SM, which is discussed in Chapter 5, Memory Architecture and Data Locality.

## 4.3 Synchronization and Transparent Scalability

### Usage of `__syncthreads()`

> [!note] CUDA allows threads in the same block to coordinate their activities using the barrier synchronization function `__syncthreads ()`.

When a thread calls `__syncthreads ()`, it will be held at the program location of the call until every thread in the same block reaches that location. This ensures that all threads in a block have completed a phase of their execution before any of them can move on to the next phase.

Barrier synchronization is a simple and popular method for coordinating parallel activities. In real life, we often use barrier synchronization to coordinate parallel activities of multiple people. For example, assume that four friends go to a shopping mall in a car. They can all go to different stores to shop for their own clothes. This is a parallel activity and is much more efficient than if they all remain as a group and sequentially visit all the stores of interest. However, barrier synchronization is needed before they leave the mall. They must wait until all four friends have returned to the car before they can leave. The ones who finish earlier than the others must wait for those who finish later. Without the barrier synchronization, one or more individuals can be left in the mall when the car leaves, which could seriously damage their friendship!

Fig. 4.3 illustrates the execution of barrier synchronization. There are $N$ threads in the block. Time goes from left to right. Some of the threads reach the barrier synchronization statement early, and some reach it much later. The ones that reach the barrier early will wait for those that arrive late. When the latest one arrives at the barrier, all threads can continue their execution. With barrier synchronization, “no one is left behind.”

![](assets/4%20Compute%20architecture%20and%20scheduling.assets/Fig4.3.png)

### Incorrect Usage of the Barriers

In CUDA, if a `__syncthreads()` statement is present, it must be executed by all threads in a block. When a `__syncthreads ()` statement is placed in an if statement, either all threads in a block execute the path that includes the `__syncthreads ()` or none of them does. For an if-then-else statement, if each path has a ` __syncthreads ()` statement, either all threads in a block execute the then-path or all of them execute the else-path. The two `__syncthreads ()` are different barrier synchronization points. For example, in Fig. 4.4, two `__syncthreads ()` are used in the if statement starting in line 04. All threads with even threadIdx. x values execute the then-path while the remaining threads execute the else-path. The `__syncthreads ()` calls at line 06 and line 10 define **two** different barriers. Since not all threads in a block are guaranteed to execute either of the barriers, the code violates the rules for using `__syncthreads ()` and will result in undefined execution behavior.

> [!note ]
> In general, incorrect usage of barrier synchronization can result in incorrect result, or in threads waiting for each other forever, which is referred to as a **deadlock**. It is the responsibility of the programmer to avoid such inappropriate use of barrier synchronization.

![](assets/4%20Compute%20architecture%20and%20scheduling.assets/Fig4.4.png)

# References

1. CUDA Occupancy Calculator, 2021. https://docs.nvidia.com/cuda/cuda-occupancy-calculator/ index. html.
2. NVIDIA (2017). NVIDIA Tesla V 100 GPU Architecture. Version WP-08608-001_v 1.1.
3. Ryoo, S., Rodrigues, C., Stone, S., Baghsorkhi, S., Ueng, S., Stratton, J., et al., Program optimization space pruning for a multithreaded GPU. In: Proceedings of the Sixth ACM/IEEE International Symposium on Code Generation and Optimization, April 6 9, 2008.
