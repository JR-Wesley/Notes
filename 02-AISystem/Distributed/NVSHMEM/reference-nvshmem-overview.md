# NVSHMEM 简介

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
