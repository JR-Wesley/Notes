## 3. PTX Machine Model

PTX machine model 从虚拟 ISA 的视角说明 NVIDIA GPU 如何组织执行资源。它关注 thread block 如何分配到 Streaming Multiprocessor、线程如何组成 warp 并以 SIMT 方式执行、分支发散如何影响效率，以及寄存器和 shared memory 等片上资源如何约束并发执行规模。

### 3.1 A Set of SIMT Multiprocessors

#### English Original

> The NVIDIA GPU architecture is built around a scalable array of multithreaded Streaming Multiprocessors (SMs). When a host program invokes a kernel grid, the blocks of the grid are enumerated and distributed to multiprocessors with available execution capacity. The threads of a thread block execute concurrently on one multiprocessor. As thread blocks terminate, new blocks are launched on the vacated multiprocessors.
>
> A multiprocessor consists of multiple Scalar Processor (SP) cores, a multithreaded instruction unit, and on-chip shared memory. The multiprocessor creates, manages, and executes concurrent threads in hardware with zero scheduling overhead. It implements a single-instruction barrier synchronization. Fast barrier synchronization together with lightweight thread creation and zero-overhead thread scheduling efficiently support very fine-grained parallelism, allowing, for example, a low granularity decomposition of problems by assigning one thread to each data element (such as a pixel in an image, a voxel in a volume, a cell in a grid-based computation).
>
> To manage hundreds of threads running several different programs, the multiprocessor employs an architecture we call SIMT (single-instruction, multiple-thread). The multiprocessor maps each thread to one scalar processor core, and each scalar thread executes independently with its own instruction address and register state. The multiprocessor SIMT unit creates, manages, schedules, and executes threads in groups of parallel threads called warps. (This term originates from weaving, the first parallel thread technology.) Individual threads composing a SIMT warp start together at the same program address but are otherwise free to branch and execute independently.
>
> When a multiprocessor is given one or more thread blocks to execute, it splits them into warps that get scheduled by the SIMT unit. The way a block is split into warps is always the same; each warp contains threads of consecutive, increasing thread IDs with the first warp containing thread 0.
>
> At every instruction issue time, the SIMT unit selects a warp that is ready to execute and issues the next instruction to the active threads of the warp. A warp executes one common instruction at a time, so full efficiency is realized when all threads of a warp agree on their execution path. If threads of a warp diverge via a data-dependent conditional branch, the warp serially executes each branch path taken, disabling threads that are not on that path, and when all paths complete, the threads converge back to the same execution path. Branch divergence occurs only within a warp; different warps execute independently regardless of whether they are executing common or disjointed code paths.
>
> SIMT architecture is akin to SIMD (Single Instruction, Multiple Data) vector organizations in that a single instruction controls multiple processing elements. A key difference is that SIMD vector organizations expose the SIMD width to the software, whereas SIMT instructions specify the execution and branching behavior of a single thread. In contrast with SIMD vector machines, SIMT enables programmers to write thread-level parallel code for independent, scalar threads, as well as data-parallel code for coordinated threads. For the purposes of correctness, the programmer can essentially ignore the SIMT behavior; however, substantial performance improvements can be realized by taking care that the code seldom requires threads in a warp to diverge. In practice, this is analogous to the role of cache lines in traditional code: Cache line size can be safely ignored when designing for correctness but must be considered in the code structure when designing for peak performance. Vector architectures, on the other hand, require the software to coalesce loads into vectors and manage divergence manually.
>
> How many blocks a multiprocessor can process at once depends on how many registers per thread and how much shared memory per block are required for a given kernel since the multiprocessor’s registers and shared memory are split among all the threads of the batch of blocks. If there are not enough registers or shared memory available per multiprocessor to process at least one block, the kernel will fail to launch.
>
> `_images/hardware-model.png`
>
> Figure 4 Hardware Model
>
> A set of SIMT multiprocessors with on-chip shared memory.

#### 中文翻译

NVIDIA GPU 架构建立在可扩展的多线程 Streaming Multiprocessor（SM）阵列之上。当 host 程序调用一个 kernel grid 时，grid 中的 block 会被依次列举，并分配给仍有执行容量的 multiprocessor。一个 thread block 中的线程在同一个 multiprocessor 上并发执行。当某些 thread block 执行结束后，新的 block 会被启动到由它们腾出的 multiprocessor 上。

一个 multiprocessor 由多个 Scalar Processor（SP）core、一个多线程指令单元和片上 shared memory 构成。Multiprocessor 在硬件中以零调度开销创建、管理和执行并发线程，并实现单指令 barrier synchronization。快速 barrier synchronization、轻量级线程创建和零开销线程调度共同高效支持非常细粒度的并行。例如，可以采用低粒度方式分解问题，为每个数据元素分配一个线程；这些数据元素可以是图像中的像素、体数据中的 voxel，或基于网格计算中的 cell。

为了管理运行若干不同程序的数百个线程，multiprocessor 使用一种称为 SIMT（single-instruction, multiple-thread，单指令多线程）的架构。Multiprocessor 把每个线程映射到一个 scalar processor core，每个标量线程都拥有自己的指令地址和寄存器状态，并可独立执行。Multiprocessor 中的 SIMT 单元以 warp 这种并行线程组为单位创建、管理、调度和执行线程。（warp 一词源自纺织；最早的并行线程技术采用了这一名称。）构成 SIMT warp 的各个线程从同一个程序地址开始执行，但之后可以自由地发生分支并独立执行。

当一个 multiprocessor 接收到一个或多个 thread block 后，它会把这些 block 划分成 warp，再由 SIMT 单元调度。Block 划分为 warp 的方式始终相同：每个 warp 包含 thread ID 连续递增的线程，第一个 warp 从 thread 0 开始。

每到一次指令发射时刻，SIMT 单元都会选择一个已就绪的 warp，并向该 warp 中的 active thread 发射下一条指令。一个 warp 每次执行一条共同指令，因此当 warp 中所有线程采用相同执行路径时，可以实现完整效率。如果 warp 中的线程因为依赖数据的条件分支而发生发散，warp 会串行执行实际被选择的每一条分支路径，并禁用不在当前路径上的线程；所有路径完成后，线程再汇合到同一条执行路径。分支发散只发生在 warp 内，不同 warp 相互独立执行，而不论它们采用相同还是不同的代码路径。

SIMT 架构与 SIMD（Single Instruction, Multiple Data，单指令多数据）向量组织方式相似，因为两者都使用一条指令控制多个处理单元。关键区别在于，SIMD 向量组织会把 SIMD 宽度暴露给软件，而 SIMT 指令描述的是单个线程的执行和分支行为。与 SIMD 向量机不同，SIMT 允许程序员为相互独立的标量线程编写线程级并行代码，也允许为相互协调的线程编写数据并行代码。就正确性而言，程序员基本可以忽略底层 SIMT 行为；但若在代码设计中尽量避免 warp 内线程发散，则可能获得显著性能提升。实际中，这类似于 cache line 在传统代码中的作用：设计正确程序时可以忽略 cache line 大小，但为了达到峰值性能，必须在代码结构中考虑它。另一方面，向量架构要求软件把 load 合并成向量，并手工管理发散。

一个 multiprocessor 能同时处理多少个 block，取决于给定 kernel 的每线程寄存器需求和每 block shared memory 需求，因为 multiprocessor 的寄存器和 shared memory 会在这一批 block 的全部线程之间分配。如果每个 multiprocessor 上没有足够的寄存器或 shared memory 来容纳至少一个 block，kernel 将启动失败。

`_images/hardware-model.png`

图 4：硬件模型。

一组具有片上 shared memory 的 SIMT multiprocessor。

#### 重点解读

Block 调度与 warp 调度处于不同层次。一个 block 只会驻留并执行在一个 SM 上，但一个 SM 可以同时驻留多个 block；block 被进一步划分为 warp，warp 才是 SM 选择并发射指令的主要执行分组。当某个 block 结束并释放寄存器、shared memory 等资源后，SM 才能接纳新的 block。

SIMT 与 SIMD 的共同点和差别可概括如下：

| 维度 | SIMT | SIMD |
| --- | --- | --- |
| 软件看到的基本实体 | 独立的标量 thread | 固定宽度的 vector/lane |
| 指令描述对象 | 单个线程的操作与分支行为 | 一条向量指令作用于多个数据元素 |
| 执行分组 | 硬件把线程组织成 warp | 软件显式使用向量宽度 |
| 发散处理 | 硬件通过 active thread/mask 执行不同路径 | 通常需要软件显式组织 mask 或控制流 |
| 正确性与性能 | 正确性通常可按独立线程推理，但性能必须考虑 warp 行为 | 程序从一开始就必须显式考虑向量组织 |

Warp divergence 的代价来自路径串行化。例如，同一 warp 一半线程执行 `if` 分支、另一半执行 `else` 分支时，硬件需要分别执行两条路径，并在每条路径上屏蔽不参与的线程。分支本身不一定昂贵；真正影响利用率的是同一 warp 内线程选择不同路径。不同 warp 选择不同路径不会构成这里所说的 warp divergence。

寄存器和 shared memory 还是 SM 上的驻留资源。每线程寄存器数乘以线程数、每 block shared memory 乘以驻留 block 数，会共同限制一个 SM 能同时容纳的 block 和 warp 数，即 occupancy。减少资源用量有时能增加并发驻留量，但 occupancy 并非越高越快；若为了提高 occupancy 造成寄存器溢出或额外访存，性能反而可能下降。

### 3.2 Independent Thread Scheduling

#### English Original

> On architectures prior to Volta, warps used a single program counter shared amongst all 32 threads in the warp together with an active mask specifying the active threads of the warp. As a result, threads from the same warp in divergent regions or different states of execution cannot signal each other or exchange data, and algorithms requiring fine-grained sharing of data guarded by locks or mutexes can easily lead to deadlock, depending on which warp the contending threads come from.
>
> Starting with the Volta architecture, Independent Thread Scheduling allows full concurrency between threads, regardless of warp. With Independent Thread Scheduling, the GPU maintains execution state per thread, including a program counter and call stack, and can yield execution at a per-thread granularity, either to make better use of execution resources or to allow one thread to wait for data to be produced by another. A schedule optimizer determines how to group active threads from the same warp together into SIMT units. This retains the high throughput of SIMT execution as in prior NVIDIA GPUs, but with much more flexibility: threads can now diverge and reconverge at sub-warp granularity.
>
> Independent Thread Scheduling can lead to a rather different set of threads participating in the executed code than intended if the developer made assumptions about warp-synchronicity of previous hardware architectures. In particular, any warp-synchronous code (such as synchronization-free, intra-warp reductions) should be revisited to ensure compatibility with Volta and beyond. See the section on Compute Capability 7.x in the Cuda Programming Guide for further details.

#### 中文翻译

在 Volta 之前的架构上，一个 warp 使用由全部 32 个线程共享的单一 program counter，同时使用 active mask 指明 warp 中的 active thread。因此，同一 warp 中处于发散区域或不同执行状态的线程无法相互发信号或交换数据；依赖 lock 或 mutex 保护细粒度数据共享的算法很容易发生死锁，具体取决于相互竞争的线程来自哪个 warp。

从 Volta 架构开始，Independent Thread Scheduling 允许线程之间完全并发，而不受 warp 边界限制。在 Independent Thread Scheduling 下，GPU 为每个线程维护执行状态，包括 program counter 和 call stack，并且可以在线程粒度上让出执行权，从而更充分地利用执行资源，或允许一个线程等待另一个线程产生数据。调度优化器决定如何把同一 warp 中的 active thread 组合为 SIMT unit。这保留了早期 NVIDIA GPU 上 SIMT 执行的高吞吐量，同时提供更大的灵活性：线程现在可以在 sub-warp 粒度上发散和重新汇合。

如果开发者假设旧硬件架构中的 warp-synchronicity，Independent Thread Scheduling 可能使实际参与某段代码执行的线程集合与预期明显不同。尤其是，任何 warp-synchronous code，例如不使用同步的 warp 内 reduction，都应重新检查，以确保与 Volta 及后续架构兼容。更多细节可参阅 Cuda Programming Guide 中 Compute Capability 7.x 相关章节。

#### 重点解读

Independent Thread Scheduling 改变的是调度灵活性，不是让 warp 这一执行组织消失。Volta 及以后仍以 SIMT 方式组合并发射同一 warp 中的 active thread，但 GPU 保存每线程 program counter 和 call stack，可以在更细粒度上发散、暂停和重新组合线程。

因此，不能再依赖“同一 warp 的线程自然锁步前进”来实现线程间通信。旧式 warp-synchronous 技巧如果省略显式同步，可能在 Volta 及以后得到错误参与者集合或读取尚未就绪的数据。凡是 lane 之间交换数据、共享生产者—消费者状态或执行 warp 内归约，都应使用具有明确参与 mask 和同步语义的 warp 原语，而不是把历史硬件行为当作正确性保证。

### 3.3 On-chip Shared Memory

#### English Original

> As illustrated by Figure 4, each multiprocessor has on-chip memory of the four following types:
>
> - One set of local 32-bit registers per processor,
> - A parallel data cache or shared memory that is shared by all scalar processor cores and is where the shared memory space resides,
> - A read-only constant cache that is shared by all scalar processor cores and speeds up reads from the constant memory space, which is a read-only region of device memory,
> - A read-only texture cache that is shared by all scalar processor cores and speeds up reads from the texture memory space, which is a read-only region of device memory; each multiprocessor accesses the texture cache via a texture unit that implements the various addressing modes and data filtering.
>
> The local and global memory spaces are read-write regions of device memory.

#### 中文翻译

如图 4 所示，每个 multiprocessor 都具有以下四类片上内存：

- 每个 processor 各自拥有的一组 local 32-bit register；
- 由所有 scalar processor core 共享的并行 data cache 或 shared memory，shared memory 状态空间驻留于此；
- 由所有 scalar processor core 共享的只读 constant cache，用于加速对 constant memory 状态空间的读取；constant memory 是 device memory 中的只读区域；
- 由所有 scalar processor core 共享的只读 texture cache，用于加速对 texture memory 状态空间的读取；texture memory 是 device memory 中的只读区域。每个 multiprocessor 通过 texture unit 访问 texture cache，该单元实现不同的寻址模式和数据过滤功能。

Local memory 和 global memory 状态空间是 device memory 中的可读写区域。

#### 重点解读

这里需要严格区分 register 与 PTX `.local` 状态空间。原文中的 “local 32-bit registers per processor” 表示每个处理单元拥有本地寄存器集合；最后一句的 local memory space 则指 PTX 中每线程私有、但位于 device memory 的 `.local` 状态空间。`.local` 并不意味着它与寄存器一样位于片上，编译器发生 register spilling 时，溢出值就可能进入 local memory。

| 存储资源 | 物理/逻辑位置 | 可见范围 | 主要作用 |
| --- | --- | --- | --- |
| register | SM 片上寄存器资源 | 单个 thread | 保存线程运算状态和临时值；也是限制 block 驻留数量的重要资源。 |
| shared memory | SM 片上共享资源 | CTA；支持的 cluster 中可扩展到 peer CTA | 线程显式管理的低延迟共享数据空间，也会限制 block 驻留数量。 |
| constant cache | SM 片上只读 cache | SM 上执行的线程共享 cache | 加速对 device constant memory 的读取。 |
| texture cache | SM 片上只读 cache | SM 上执行的线程共享 cache | 加速 texture memory 访问，并配合 texture unit 提供寻址和过滤。 |
| local memory | device memory 中的线程私有状态空间 | 单个 thread | 保存无法放入寄存器的线程私有数据；可能经过 cache，但语义上不同于 register。 |
| global memory | device memory 中的全局状态空间 | 所有线程 | 保存跨线程、跨 CTA 以及跨 kernel 使用的主要数据。 |

