# NVSHMEM 简介

[NVSHMEM](https://developer.nvidia.com/nvshmem) implements the [OpenSHMEM](http://openshmem.org/) parallel programming model for clusters of NVIDIA ® GPUs. The NVSHMEM Partitioned Global Address Space (PGAS) spans the memory across GPUs and includes an API for fine-grained GPU-GPU data movement from within a CUDA kernel, on CUDA streams, and from the CPU.

NVSHMEM 是 NVIDIA 面向 GPU 集群实现的 OpenSHMEM 风格通信库。它把多个 GPU 上按相同方式分配的显存组织成分区全局地址空间（Partitioned Global Address Space，PGAS），允许 CPU 线程或 CUDA kernel 中的 GPU 线程直接发起跨 GPU 的单边通信、原子操作、同步与集合通信。

它解决的核心问题不是“如何替代所有 MPI 或 NCCL 调用”，而是：当通信模式细粒度、不规则、依赖数据，或通信必须嵌入一个长时间运行的 kernel 时，如何减少 CPU 调度、kernel 边界和主机—设备同步带来的开销。

## 核心抽象

### Processing Element

一个 NVSHMEM 作业由多个 Processing Element（PE）组成。每个 PE 是执行同一份程序的进程，拥有从 `0` 到 `npes - 1` 的编号。程序通常采用 SPMD（Single Program, Multiple Data）方式运行：各 PE 执行相同代码，再根据 PE 编号处理不同数据。

PE 通过 `nvshmem_my_pe()` 获取自身编号，通过 `nvshmem_n_pes()` 获取 PE 总数。常见部署是一进程对应一块 GPU；设备选择、初始化和对称内存分配之间还存在明确的调用约束，见 [B00：NVSHMEM 编程模型](b00-nvshmem-programming-model.md)。

### 对称堆与对称对象

每个 PE 都有位于 GPU 显存中的 symmetric heap。所有 PE 以相同顺序、相同大小调用 `nvshmem_malloc`、`nvshmem_calloc` 等集合式分配接口后，会分别得到形状一致的内存区域。所有 PE 上这一组对应区域共同构成一个 symmetric object。

远端对象由二元组确定：

```text
<当前 PE 上的 symmetric address, 目标 PE>
```

这里的 symmetric address 只应在产生它的 PE 内使用，不能先把本地指针值传给另一 PE 再解引用。NVSHMEM runtime 会根据本地对称地址和目标 PE 完成远端地址转换。

对称地址同时也是本地 GPU 上的有效地址，因此当前 PE 可以用普通 CUDA load/store 或 CUDA API 访问自己的那一份对象；访问远端 PE 时则使用 NVSHMEM API。

### 单边通信

NVSHMEM 的 Remote Memory Access（RMA）主要包括：

- `put`：把本地数据写入目标 PE 的对称对象；
- `get`：把目标 PE 的对称对象读到本地；
- Atomic Memory Operation（AMO）：对远端对称对象执行原子更新或读取；
- put-with-signal：写入数据后更新远端 signal，便于接收侧等待数据就绪。

“单边”表示发起方描述本地地址、远端对称地址和目标 PE 即可发起访问，不要求目标 PE 同时配对调用 `recv`。目标 PE 仍然需要用 wait/test、barrier 或应用协议判断数据何时可安全使用。

## 内存模型

[![../_images/mem_model.png](https://docs.nvidia.com/nvshmem/api/latest/_images/mem_model.png)](https://docs.nvidia.com/nvshmem/api/latest/_images/mem_model.png)

NVSHMEM 程序由私有数据对象和所有 PE 均可远程访问的数据对象组成。私有数据对象存储在每个 PE 的本地内存中，只能由该 PE 自身访问；其他 PE 无法通过 NVSHMEM 例程访问这些数据对象。私有数据对象遵循 _C 语言 _ 的内存模型。而远程可访问的数据对象则可以通过 NVSHMEM 例程被远程 PE 访问。远程可访问的数据对象称为*对称数据对象*。每个对称数据对象在所有可通过 NVSHMEMAPI [1](https://docs.nvidia.com/nvshmem/api/latest/gen/mem-model.html#id2).访问的 PE 上都有一个名称、类型和大小相同的对应对象。

在 NVSHMEM 中，由 NVSHMEM 内存管理例程分配的 GPU 内存是对称的。 有关分配对称内存的信息，请参阅 [“内存管理”部分。](https://docs.nvidia.com/nvshmem/api/latest/gen/api/memory.html#sec-memory-management)

NVSHMEM 动态内存分配例程（例如 `nvshmem_malloc`）允许将*对称数据对象*集中分配到称为*对称堆*的特殊内存区域。对称堆在程序执行期间创建，其内存位置由 NVSHMEM 库确定。对称堆可以位于不同 PE 上的不同内存区域。图 [NVSHMEM 内存模型](https://docs.nvidia.com/nvshmem/api/latest/gen/mem-model.html#fig-mem-model) 展示了一个 NVSHMEM 内存布局示例，说明了远程访问的对称对象和私有数据对象的位置。

### 指向对称对象的指针

**在 NVSHMEM 操作中，对称数据对象通过指向所需远程可访问对象的本地指针进行引用**。此指针中包含的地址称为*对称地址*。每个对称地址也是一个可用于直接内存访问的*本地地址*；但是，并非所有本地地址都是对称的。只要生成的本地指针仍然位于同一对称分配或对象内，就允许对传递给 NVSHMEM 例程的对称地址进行操作，包括指针运算、数组索引以及访问结构体或联合体成员。对称地址仅在生成它们的 PE 上有效；使用由不同 PE 生成的对称地址进行直接内存访问或将其作为 NVSHMEM 例程的参数会导致未定义行为。

提供给类型化接口的对称地址必须根据其类型和底层架构的任何要求进行自然对齐。提供给固定大小 NVSHMEM 接口（例如 `NVSHMEM` `nvshmem_put32`）的对称地址也必须按给定大小对齐。提供给固定大小 NVSHMEM 接口的对称对象的存储大小必须等于给定操作的位宽。由于 C/C++ 结构体可能包含实现定义的填充，因此不应将固定大小接口与 C/C++ 结构体一起使用。“mem”接口（例如 `NVSHMEM` `nvshmem_putmem`）没有对齐要求。

该 `nvshmem_ptr` 例程允许程序员查询指定 PE 上远程可访问数据对象的本地地址。返回的指针可用于直接内存访问；但是，如果将此地址作为参数传递给需要对称地址的 NVSHMEM 例程，则会导致未定义行为。

### 操作顺序

在 NVSHMEM 中，读取数据的阻塞操作（例如，get 或原子取加操作）应按照操作执行的顺序返回数据。例如，考虑一个对值执行原子取加操作的程序。1 对于对称变量𝑥在 PE 0 上。

```c

a = nvshmem_int_fadd(x, 1, 0);
b = nvshmem_int_fadd(x, 1, 0);
```

在这个例子中，OpenSHMEM 规范保证了 $𝑏 >𝑎$ 然而，这种强顺序性在弱顺序架构上会带来显著的开销，因为它要求在任何此类操作返回之前执行内存屏障。NVSHMEM 放宽了这一要求，从而在 NVIDIA GPU 上提供更高效的实现。因此，NVSHMEM 不保证 $𝑏 >𝑎$。

如果需要这种排序，程序员可以使用 `get` `nvshmem_fence` 操作来强制阻塞操作（例如，上述两个语句之间）的顺序。非阻塞操作的顺序并非由 `get` 调用决定 `nvshmem_fence`，而是必须使用 `get` 操作来完成 `nvshmem_quiet`。获取操作的完成语义与 OpenSHMEM 规范保持一致：`get` 或 `AMO` 的结果对于其后出现的任何依赖操作都可用，并按程序顺序执行。

### 原子性保证

NVSHMEM 包含多个对对称数据对象执行原子操作的例程，这些例程在 [“原子内存操作”](https://docs.nvidia.com/nvshmem/api/latest/gen/api/amo.html#sec-amo) 一节中定义。这些原子例程保证，任何例程对同一位置使用相同数据类型（在 [“标准 AMO 类型和名称”](https://docs.nvidia.com/nvshmem/api/latest/gen/api/amo.html#stdamotypes) 和 [“扩展 AMO 类型和名称](https://docs.nvidia.com/nvshmem/api/latest/gen/api/amo.html#extamotypes)”表中指定）的并发访问将是互斥的。当目标 PE 对与一个或多个原子操作相同的位置和数据类型执行等待或测试操作时，也保证了互斥性。

在以下情况下，NVSHMEM 原子操作不能保证排他性，所有这些情况都会导致未定义行为。

1. 当使用不同数据类型通过 NVSHMEM 原子操作对同一位置进行并发访问时。
2. 当使用原子操作和非原子操作同时访问同一位置时。
3. 当使用 NVSHMEM 原子操作和非 NVSHMEM 操作（例如，加载和存储操作）并发访问同一位置时。

### NVSHMEM 和 OpenSHMEM 的区别

#### 阻塞式获取操作的顺序

OpenSHMEM 中的阻塞式数据读取操作（例如 get 或原子取加操作）应按照操作执行顺序返回数据。例如，考虑一个程序，该程序执行原子集合操作来更新 PE 0 上的对称变量 x 和 y ：

// Let: v_N represent symmetric variable v at PE N

// Input: x_0 = 0, y_0 = 0, i = 0, j = 0

if (nvshmem_my_pe() == 0) {

    a = nvshmem_int_atomic_set(x, 1, 0);

    nvshmem_quiet();

    b = nvshmem_int_atomic_set(y, 1, 0);

}

i = nvshmem_int_atomic_fetch(y, 0);

j = nvshmem_int_atomic_fetch(x, 0);

// Allowed output: i = 1, j = 0

在这个例子中，OpenSHMEM 规范保证了如果 i == 1，则 j == 1。然而，这种强顺序性在弱顺序架构上会带来显著的开销，因为它要求在任何取指操作返回之前执行内存屏障。NVSHMEM 放宽了这一要求，从而提供了一种更高效的实现，使其与 NVIDIA GPU 内存模型保持一致。因此，NVSHMEM 不再保证如果 i == 1，则 j == 1。

如果需要这种排序，程序员可以使用 `nvshmem_fence` 操作来强制阻塞操作（例如， `nvshmem_int_atomic_fetch` 上述两个语句之间）的顺序。非阻塞操作的顺序并非由调用决定 `nvshmem_fence`。相反，这些操作必须通过操作来完成 `nvshmem_quiet` 。获取操作的完成语义与规范保持一致，即 get 或 AMO 的结果对于其后出现的任何依赖操作都可用，并按程序顺序执行。

#### 可见性保证

在同时具备 NVLink 和 InfiniBand 的系统中，NVSHMEM 同步操作 `nvshmem_barrier`（包括 `nvshmem_barrier_all`` NVSHMEM_Ssync_get_re `nvshmem_quiet`... `nvshmem_wait_until_*``nvshmem_test_*`

[ [1](https://docs.nvidia.com/nvshmem/api/latest/gen/mem-model.html#id1) ]

出于效率考虑，对称数据对象在所有处理单元 (PE) 上可以使用相同的偏移量（从任意内存地址开始）。有关对称堆布局和实现效率的更多讨论，请参见 [NVSHMEM_MALLOC、NVSHMEM_FREE 和 NVSHMEM_ALIGN章节。](https://docs.nvidia.com/nvshmem/api/latest/gen/api/memory.html#subsec-shfree)

## 执行模型

NVSHMEM 程序由一组称为 PE 的 NVSHMEM 进程组成。虽然 NVSHMEM 本身并不要求，但在典型使用中，PE 会采用单程序多数据 (SPMD) 模型执行。SPMD 要求每个 PE 使用相同的可执行文件；然而，PE 可以遵循不同的控制路径。PE 使用操作系统进程实现，并且在启用线程支持时，PE 可以创建额外的线程。

PE 执行是**松耦合**的，依赖于 NVSHMEM 操作在执行的 PE 之间进行通信和同步。程序中的 NVSHMEM 阶段以调用初始化例程 `nvshmem_init` 或结束函数开始 `nvshmem_init_thread`，该调用必须在调用任何其他 NVSHMEM 库例程之前执行。当所有 PE 都调用结束函数 `nvshmem_finalize` 或任何 PE 调用结束函数 `nvshmem_global_exit` 时，NVSHMEM 程序结束对 NVSHMEM 库的使用。在调用结束函数期间 `nvshmem_finalize`，NVSHMEM 库必须完成所有待处理的通信，并使用跨 PE 的隐式集体同步释放与该库关联的所有资源。在初始化之前或之后调用任何 NVSHMEM 例程 `nvshmem_finalize` 都会导致未定义行为。在结束函数之后，后续的初始化调用也会导致未定义行为。

NVSHMEM 程序的 PE 由唯一的整数标识。这些标识符是按单调递增方式赋值的整数，从零到比 PE 总数少一的值。PE 标识符用于 NVSHMEM 调用（例如，指定对对称数据对象执行 _put_ 或 _get 操作、集体同步调用），或使用 __C_ 语言结构来定义 PE 的控制流。这些标识符在程序的 NVSHMEM 阶段期间保持不变。

## NVSHMEM 操作的进行

NVSHMEM 模型假设计算和通信自然重叠。NVSHMEM 程序应展现出通信的进行，无论是否调用 NVSHMEM。考虑一个执行计算的 PE，此时没有调用任何 NVSHMEM。其他 PE 应该能够与该执行计算的 PE 进行通信（例如，_put_、_get_、 _atomic_ 等），并在该 PE 不发出任何显式 NVSHMEM 调用的情况下完成通信操作。涉及该 PE 的单向 NVSHMEM 通信调用应该能够进行，而与该 PE 何时再次调用 NVSHMEM 无关。

## 调用 NVSHMEM 操作

指向非 `const` 数据对象的 NVSHMEM 例程的指针参数在内存中不得与其他 NVSHMEM 操作的参数重叠，但 [NVSHMEM_REDUCTIONS](https://docs.nvidia.com/nvshmem/api/latest/gen/api/collectives.html#subsec-shmem-reductions) 节中所述的就地归约除外 。否则，行为未定义。如果两个参数的任何数据元素位于相同的物理内存位置，则这两个参数在内存中重叠。

For example, consider an address 𝑎 returned by the `nvshmem_ptr` operation for symmetric object 𝐴 on PE 𝑖. Providing the local address 𝑎 and the symmetric address of object 𝐴 to an NVSHMEM operation targeting PE 𝑖 results in undefined behavior.

提供给 NVSHMEM 例程的缓冲区会一直*处于使用状态*，直到调用 PE 上的相应 NVSHMEM 操作完成为止。对正在使用的缓冲区进行更新（包括通过本地和远程 NVSHMEM 操作执行的更新）会导致未定义行为。类似地，只有当缓冲区作为 `const` 限定参数提供给正在使用它的 NVSHMEM 例程时，才允许从正在使用的缓冲区读取数据。否则，行为未定义。对于被 AMO 使用的缓冲区，将进行例外处理，如 [“原子性保证”](https://docs.nvidia.com/nvshmem/api/latest/gen/mem-model.html#subsec-amo-guarantees) 一节所述。有关 NVSHMEM 操作完成的信息，请参阅 [“内存排序”](https://docs.nvidia.com/nvshmem/api/latest/gen/api/ordering.html#subsec-memory-order) 一节。

具有多个对称对象参数的 NVSHMEM 例程并不要求这些对称对象位于同一个对称内存段中。例如，位于对称数据段中的对象和位于对称堆中的对象都可以作为参数传递给同一个 NVSHMEM 操作。

## 为什么允许 GPU 发起通信

传统的 host-staged 流程可能需要 kernel 返回 CPU，由 CPU 发起通信，再启动下一个 kernel。NVSHMEM 的 device API 允许 CUDA 线程在 kernel 内执行 put/get、原子操作及同步，因此可以把如下循环保留在 GPU 上：

```text
计算局部数据 -> 与邻居交换 -> 等待依赖 -> 继续计算
```

这对 stencil、图计算、稀疏计算、动态负载均衡以及其他邻域通信或细粒度通信尤其有价值。收益来自减少 CPU 介入、避免频繁 kernel launch，并让通信粒度与 CUDA 线程层级匹配；是否更快仍取决于拓扑、传输层、消息大小、访问合并和同步设计。

## API 能力

| 能力 | 典型接口 | 说明 |
| --- | --- | --- |
| 初始化与查询 | `nvshmem_init`、`nvshmem_finalize`、`nvshmem_my_pe` | 建立并查询 PE 世界 |
| 对称内存 | `nvshmem_malloc`、`nvshmem_calloc`、`nvshmem_free` | 所有 PE 必须按兼容顺序参与 |
| RMA | `nvshmem_<type>_put/get`、`*_nbi` | 支持阻塞与非阻塞变体 |
| 原子操作 | `nvshmem_<type>_atomic_*` | 用于计数器、队列和细粒度协调 |
| 点对点同步 | `nvshmem_wait_until`、`nvshmem_test`、signal API | 等待或检测本地对称对象上的远端更新 |
| 内存序 | `nvshmem_fence`、`nvshmem_quiet` | 分别侧重顺序与完成性 |
| 集合同步与通信 | barrier、sync、broadcast、reduce、all-to-all 等 | 可在 team 范围内执行 |
| CUDA 扩展 | `nvshmemx_*_on_stream`、`*_warp`、`*_block` | 与 stream、warp、thread block 集成 |

内存序是 NVSHMEM 正确性的关键，不能把一次 `put` 返回误解成远端数据已经对消费者可见，详见 [B02：完成、排序与可见性](b02-completeness-ordering-and-visibility.md)。

## 与 MPI、NCCL 的关系

| 维度 | NVSHMEM | MPI | NCCL |
| --- | --- | --- | --- |
| 主要抽象 | PGAS、单边 RMA、AMO、同步、集合通信 | 消息传递、RMA、集合通信 | GPU 集合通信与点对点通信 |
| 发起位置 | CPU 或 CUDA kernel | 通常由 CPU 进程调用 | 通常由 CPU 把操作排入 CUDA stream |
| 地址模型 | 对称对象加目标 PE | 发送/接收缓冲区或 MPI window | communicator 中的 send/recv buffer |
| 典型优势 | kernel 内细粒度或不规则通信 | 通用分布式编程与成熟生态 | 拓扑优化的大吞吐 GPU 集合通信 |

三者不是互斥替代关系。MPI 可用于启动和管理进程，也可与 NVSHMEM 混合使用；NVSHMEM 构建时可选择 NCCL 支持，以加速部分 host-initiated collectives。选择应基于通信模式和执行位置，而不是只按“消息大小”做结论。参见 [NCCL API 使用详解](../NCCL/NCCL-API-使用详解.md)。

## 通信路径与硬件

NVSHMEM 不限于 DGX。节点内可以通过 NVLink 或 PCIe P2P 访问其他 GPU；节点间可利用 GPUDirect RDMA，经 InfiniBand/RoCE 等网络传输。当前安装指南还列出 Slingshot-11 和 Amazon EFA 等支持路径。具体能力由 GPU、NIC、驱动、CUDA、所选 transport 及构建选项共同决定。

跨 NVLink 或支持 GPUDirect RDMA 的网络执行原子操作还可能需要 GDRCopy 等组件，因此“普通 RMA 可运行”不代表所有 AMO 都已具备正确的硬件与软件支持。

## 适用场景与限制

适合优先考虑 NVSHMEM 的场景：

- 通信由 GPU 上的数据依赖动态决定；
- 希望在持久化 kernel 中交替进行计算、通信和同步；
- 需要远端原子、远端队列或邻居 signal；
- 通信模式稀疏、不规则，难以直接表达成固定集合通信；
- 希望使用 warp/block 共同发起通信以改善访问合并。

需要谨慎评估的场景：

- 工作负载主要是规则的 all-reduce、all-gather 等大吞吐集合通信，此时 NCCL 往往更自然；
- 程序不能保证对称分配顺序或清晰的 ownership；
- kernel 内存在阻塞等待，却没有保证相关 PE 和线程可以同时取得执行资源；
- 代码依赖 CUDA 默认的强顺序直觉，未显式设计 `fence`、`quiet` 和 wait/test；
- 小而离散的线程级远端访问无法合并，网络事务开销可能抵消 GPU 发起通信的收益。

## 学习路径

1. 先掌握 PE、对称堆与 `<symmetric address, PE>` 寻址：[B00：NVSHMEM 编程模型](b00-nvshmem-programming-model.md)。
2. 再理解 put/get、signal、`fence`、`quiet` 与 barrier 的边界：[B02：完成、排序与可见性](b02-completeness-ordering-and-visibility.md)。
3. 最后通过程序骨架建立初始化、分配、kernel、同步和释放的完整顺序：[实验 B01：对称寻址与 PUT](experiments/experiment-b01-symmetric-addressing-and-put.md)。

## 参考资料

- [NVSHMEM API Introduction](https://docs.nvidia.com/nvshmem/api/latest/introduction.html)
- [Using NVSHMEM](https://docs.nvidia.com/nvshmem/api/latest/using.html)
- [NVSHMEM API Overview](https://docs.nvidia.com/nvshmem/api/latest/api/overview.html)
- [NVSHMEM Installation Guide](https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/abstract.html)
