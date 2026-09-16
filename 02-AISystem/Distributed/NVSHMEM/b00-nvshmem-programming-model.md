# B00 NVSHMEM 编程模型：从多个独立 GPU 进程到可远端访问的对称对象

## 本课要解决的问题

完成 A01—A04 后，我们已经知道，传统 RDMA 程序必须处理本地与远端内存注册、地址与 key 交换、QP 等传输资源以及请求完成。NVSHMEM 没有消灭这些底层事实，而是在它们之上提供了一种面向 GPU 集群的 Partitioned Global Address Space（PGAS）编程模型：程序用 processing element（PE）标识通信参与者，用 symmetric object 表示每个 PE 上逻辑对应的对象，再通过 PUT、GET、atomic、signal 和 collective 等操作访问或协调这些对象。应用因此不必在每次通信时显式传递远端虚拟地址和 `rkey`，但仍然必须管理对象生命周期、并发访问、完成条件与消费者同步。

本课建立后续 NVSHMEM 学习所需的总体坐标。读完后，应该能够回答五个问题：PE 与 GPU、进程、CUDA thread 分别是什么关系；PGAS 为什么仍然由多份物理内存构成；“对称”究竟约束什么；one-sided communication 为什么不等于自动同步；host、CUDA stream 和 device kernel 发起的操作为什么不能混用同一个完成判断。B01—B04 将分别深入远端寻址、完成与可见性、CUDA 执行与前进性，以及数据发布与缓冲区所有权，本课只建立这些问题之间的边界。

## 资料版本与论述边界

本文于 2026-09-12 依据 NVIDIA 官方 NVSHMEM API Guide 的 `latest` 页面核对，并以 [NVSHMEM 3.7.2 Release Notes](https://docs.nvidia.com/nvshmem/release-notes-install-guide/release-notes/release-3720.html) 标识所处发布周期。`latest` 是滚动文档，未来版本可能增加 API 或调整限制；本文没有绑定 NVSHMEM 源码 commit，因此只讨论公开编程契约，不把某个 transport、地址换算公式、QP 布局或 progress engine 写成所有版本都成立的实现事实。

文中的 C++/CUDA 片段用于解释控制关系，没有在本任务中编译或运行，也没有 GPU、NVLink、InfiniBand 或 IBGDA 实测结果。为保持主线清晰，示例省略错误检查；真实程序必须检查 CUDA 与 NVSHMEM 初始化、内存分配和 kernel launch 的返回状态。

## 一、先分清四层实体：作业、PE、进程与 CUDA 执行者

一个 NVSHMEM job 由若干 PE 组成。常见的 C/CUDA 使用方式中，每个 PE 是由进程管理器启动的一个操作系统进程，各 PE 执行同一份可执行程序，形成 Single Program, Multiple Data（SPMD）结构。`nvshmem_my_pe()` 返回当前 PE 在 world 中的编号，范围为 $[0, N-1]$；`nvshmem_n_pes()` 返回 PE 总数。所有 PE 执行同一程序不意味着控制流必须完全相同，例如 PE 0 可以负责生成输入，其他 PE 可以负责消费，但凡涉及初始化、对称内存管理或 collective，参与者和调用顺序必须满足对应的集合契约。

PE 是 NVSHMEM 的分布式身份，不是 CUDA thread、warp、thread block 或 streaming multiprocessor（SM）。一个 PE 选定一张 GPU 后，可以在该 GPU 上启动许多 kernel；每个 kernel 又包含许多 CUDA 执行线程。这两套编号回答不同问题：PE ID 决定“访问哪个远端参与者”，CUDA thread/block 索引决定“本 GPU 上由哪个执行者处理哪个元素”。如果把两者混为一谈，就容易误以为“PE 1”表示第二个 CUDA thread，或错误地用 `threadIdx.x` 直接构造跨节点目标。

Team 是 PE 的子集及其通信上下文，也不等于 CUDA thread block。`NVSHMEM_TEAM_WORLD` 表示整个 job，`NVSHMEMX_TEAM_NODE` 表示同一节点内的 PE。一个 PE 在 world team 和 node team 中可能有不同编号；后者常用于把节点内 PE 映射到本地 GPU，前者常用于全局点对点目标。后续 B06 会展开 Team 的创建、编号翻译与 collective 规则。

可以把这些实体的包含关系先写成：

```text
NVSHMEM job
├── PE 0：OS process 0 -> selected GPU -> kernels -> CUDA threads
├── PE 1：OS process 1 -> selected GPU -> kernels -> CUDA threads
└── PE N-1：OS process N-1 -> selected GPU -> kernels -> CUDA threads

Team：从上述 PE 集合中选择一个有独立编号空间的子集
```

## 二、初始化把独立进程组织成一个 NVSHMEM 作业

NVSHMEM 阶段从初始化开始。`nvshmem_init()` 是所有 PE 必须匹配参与的 collective；初始化成功后，runtime 才能提供 PE 查询、对称内存、通信和同步服务。结束时，各 PE 用 `nvshmem_finalize()` 匹配初始化并释放 runtime 资源。若 host 端存在多个线程可能调用 NVSHMEM，应使用 `nvshmem_init_thread(requested, &provided)` 请求线程支持级别，并以实际返回的 `provided` 为准；不能仅因 C++ 程序创建了多个线程，就假设 host API 自动支持任意并发调用。

在常见的一 PE 对一 GPU 模式中，程序初始化后查询节点内 PE 编号，再选择对应 GPU。官方使用指南要求 PE 在第一次对称分配、同步、通信、collective launch 或 device API 之前完成 GPU 选择，并在 NVSHMEM 生命周期内围绕所选 GPU 工作。多 PE 共享一张 GPU 属于 Multi-Process GPU（MPG）高级场景，是否支持、支持到何种程度以及是否需要 MPS 取决于版本和运行配置；初学阶段不应把它当作默认部署模型。

```cpp
// 教学骨架：未编译、未运行，省略错误检查。
nvshmem_init();

int world_pe = nvshmem_my_pe();
int world_size = nvshmem_n_pes();
int node_pe = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
cudaSetDevice(node_pe);

// 对称分配、kernel、通信与同步位于此处。

nvshmem_finalize();
```

从系统实现角度看，初始化背后通常还涉及 bootstrap、PE 信息交换、GPU/NIC 拓扑发现、symmetric heap 建立、内存注册、transport 选择和 endpoint 元数据准备。但这些是理解运行时职责的概念分解，不是 `nvshmem_init()` 对外承诺的固定内部步骤或顺序。C00 将专门讨论 control plane；D 模块只有在固定源码 commit 后才追踪具体数据结构和调用路径。

## 三、PGAS 不是一块共享物理内存，而是可统一寻址的分区集合

不同 PE 通常是不同进程，并各自管理本地 GPU 内存。普通 `cudaMalloc()` 得到的对象默认是分配 PE 的私有对象，其他 PE 不能仅凭获得它的裸指针值，就把它当作 NVSHMEM 远端操作数。NVSHMEM 在每个 PE 上建立 symmetric heap，其中可放置能够被其他 PE 通过 NVSHMEM 语义访问的对象。所有 PE 上这些 remotely accessible 分区的集合构成 PGAS。

“全局地址空间”描述的是编程层如何定位数据，不表示硬件上只有一份内存，也不表示任意 GPU load/store 都可以跨节点透明执行。数据仍然位于某个 PE 的本地分区；远端访问必须同时说明对象位置和目标 PE。位置是地址语义的一部分，因此 NVSHMEM 的核心寻址形式是：

$$
\langle \text{symmetric address},\ \text{target PE}\rangle .
$$

这个二元组比“全局唯一裸指针”更准确。调用 PE 提供自己地址空间中生成的 symmetric address，runtime 再结合目标 PE 解析对方的对应对象。不同 PE 上对应对象的虚拟地址数值不必相同，而且一个 PE 生成的 symmetric address 不能发送给另一个 PE 后作为后者的本地 symmetric address 使用。B01 会用对象内偏移建立教学模型，并解释为什么该模型不能被误写成 runtime 必然使用的固定公式。

## 四、对称对象约束分配关系，而不要求内容相同

当所有 PE 以兼容顺序和相同参数调用 `nvshmem_malloc(size)` 时，每个 PE 都从自己的 symmetric heap 得到一块对应分配。所谓 symmetric，主要指这些对象具有匹配的分配次序、大小和逻辑布局；它不要求虚拟地址数值相同，也不要求对象内容相同。PE 0 可以在自己的数组中保存 0，PE 1 在对应数组中保存 1，它们仍然是同一个逻辑 symmetric object 在不同 PE 上的分区。

```cpp
// 所有 PE 都必须以兼容顺序执行，bytes 必须一致。
int *mailbox = static_cast<int *>(nvshmem_malloc(bytes));
```

动态对称内存管理本身是 collective。对非零分配，`nvshmem_malloc()` 和 `nvshmem_align()` 在退出前包含语义等价于 `nvshmem_barrier_all` 的过程；`nvshmem_free()` 在入口包含相应过程。由此可以保证某个 PE 从分配返回时，其他 PE 的对应内存已经建立，但这不意味着之后所有对该对象的读写都自动同步。所有 PE 仍须使用一致参数并维持兼容的分配/释放顺序，否则对应关系被破坏，后续行为可能未定义或出现永久等待。

对称对象同时具有本地和远端两种观察方式。`mailbox[i]` 是当前 PE 可通过 CUDA 本地访问的 GPU 对象；`< &mailbox[i], q >` 则表示 PE `q` 上对应对象的对应元素。这里的本地指针承担“描述逻辑位置”的角色，并不把 PE `q` 的虚拟地址暴露给应用。对称性也不会自动提供边界检查：数组越界、释放后使用、不同 PE 使用不同结构布局，仍然是应用错误。

## 五、one-sided RMA 将数据移动与接收方同步解耦

消息传递通常要求发送和接收操作建立匹配关系。NVSHMEM 的 Remote Memory Access（RMA）是 one-sided：origin PE 在一次 PUT 或 GET 中提供源、目标、长度和远端 PE，destination PE 不必同时执行一个匹配的 receive。PUT 把 origin 的本地数据写入 destination PE 的 symmetric object；GET 把 destination PE 的 symmetric object 读回 origin 的本地缓冲区。

```text
PUT：origin local source  ─────> destination PE 的 symmetric destination
GET：origin local dest    <───── destination PE 的 symmetric source
```

```cpp
// 教学片段：在发起 PE 上执行，未编译、未运行。
nvshmem_int_p(&mailbox[index], value, target_pe);  // 单元素 PUT
int value2 = nvshmem_int_g(&mailbox[index], target_pe);  // 单元素 GET
```

one-sided 只表示目标 PE 不需要以匹配调用参与这次数据移动，不能推出目标 PE 已经准备好消费数据。官方 RMA 文档明确把通信与同步分开：PUT/GET 负责搬运，应用协议还要决定源缓冲区何时能复用、操作何时达到所需完成点、写入以什么顺序发布，以及消费者通过 wait、signal、barrier 或其他状态变量在何时观察结果。也就是说，“无需 `recv`”不等于“无需同步”。

NVSHMEM 的基本能力可以按职责分为几类。RMA（PUT/GET）移动数据；Atomic Memory Operations（AMO）对远端对象执行不可分割的读改写；signal 及 wait/test 适合表达通知与条件等待；memory ordering API 建立排序或完成条件；collective 在一个 Team 内协调多个 PE。应用需要根据协议组合这些能力，而不能把任意一个原语当成完整的生产者—消费者协议。B02 深入 completion、ordering 与 visibility，B04 则把这些概念落实为数据槽位的 ownership 状态机。

## 六、远端可寻址并不等于共享缓存一致性

初学者容易把 PGAS 理解成“多张 GPU 共享一个普通数组”。更准确的说法是：NVSHMEM 让每个 PE 能以统一形式引用其他 PE 的对应对象，并为跨 PE 操作规定一致性与同步语义。GPU 架构本身是弱一致性的，远端写入、目标 GPU 的普通 load、CUDA stream 中的任务以及 host 线程之间不会仅凭程序文本顺序就自动形成所需的 happens-before 关系。

因此，看到下面的逻辑时不能只问“PUT 发出了吗”，还必须区分多个阶段：producer 是否已完成本地数据生成；PUT 是否已经不再依赖源缓冲区；payload 是否达到远端所需完成点；flag 是否在 payload 之后发布；consumer 的 wait 满足后，是否允许安全读取 payload；producer 又在何时可以覆盖下一轮使用的槽位。这些问题分别涉及 CUDA 顺序、NVSHMEM ordering/completion、远端可见性和应用 ownership，不能用一个含糊的“同步完成”代替。

```text
producer PE                              consumer PE
生成 payload
    ├── PUT payload ───────────────────> inbox
    ├── 建立必要的顺序
    └── 更新 signal ───────────────────> wait/test 满足
                                             └── 消费 inbox
```

本课只给出判断框架：地址有效性回答“访问哪个对象”，完成语义回答“通信推进到哪里”，同步回答“执行者何时可以继续”，ownership 回答“谁在何时可以读写或复用”。四个问题互相依赖，但不是同一个问题。

## 七、三种发起域与三套编号必须分开

NVSHMEM 操作可以从 host、CUDA stream 和 device kernel 发起。Host API 由 CPU 线程调用；on-stream API 由 CPU 将 NVSHMEM 工作排入一个 CUDA stream；device API 则由正在执行的 CUDA thread 在 kernel 中调用。这三类接口可能完成相似的数据移动，但其程序顺序来源不同：host 调用受 CPU 线程顺序约束，on-stream 操作要等到它到达 stream 队首才执行，device 操作处于 kernel 与 CUDA thread 的执行上下文中。

| 坐标系统 | 典型标识 | 主要回答的问题 |
| --- | --- | --- |
| 分布式参与域 | world PE、Team-relative PE | 操作针对哪个 PE，哪些 PE 参加 collective |
| CUDA 执行域 | thread、warp、block/CTA、grid | 本 GPU 上哪些线程共同发起或等待 |
| 工作提交与顺序域 | host thread、CUDA stream、device kernel | 操作何时被提交，与哪些前后任务有顺序关系 |

这些坐标不能相互替代。某个 block 的 `threadIdx.x == 0` 只选出了当前 block 的一个线程，没有选出 world PE 0；两个操作位于不同 CUDA stream，也不会因为它们由同一个 PE 提交就自动串行；host 上的完成操作与 device 发起的通信也不能未经 API 契约证明就视为同一个 completion domain。跨 stream 依赖通常需要 CUDA event 或明确同步，跨 NVSHMEM 发起域的完成关系则必须查对应 API 契约。B03 将系统分析这些组合。

## 八、thread、warp 与 Block API 表示协作发起范围

device 端既有由单个 CUDA thread 调用的接口，也有带 `_warp` 或 `_block` 后缀的 thread-group 接口。后两者表示一个 warp 或整个 thread block 共同参与同一逻辑 NVSHMEM 操作，runtime 可以让多个线程协作搬运较大的连续数据。它们不是“任意线程都可独立调用的更快版本”：规定 group 内所有线程参与的接口必须由所有成员调用，并为共同参数传入一致值；若只有部分线程进入，可能产生未定义行为或死锁。

协作粒度也不是越大越好。单元素控制更新通常不需要整个 block 搬运；较大连续消息可能从 coalesced access 与协作复制中受益；实际收益还取决于消息大小、对齐、拓扑和所选 transport。性能判断应由后续 microbenchmark 验证，不能仅从 API 后缀推出。

CUDA kernel 若调用 NVSHMEM wait、barrier 或其他 synchronization/collective API，必须使用 `nvshmemx_collective_launch()`。该函数在所有 PE 上集合调用，并通过 CUDA cooperative launch 限制可启动网格，使所需线程有机会并发驻留。普通只做 one-sided 数据移动、不调用同步或 collective 的 kernel 不强制使用该接口。即使使用 collective launch，也只解决特定驻留与启动条件，不会修复 Team 调用不匹配、错误的 stream 依赖或应用层循环等待。

## 九、Team 定义集合通信的参与域

点对点 PUT/GET 通过目标 PE 指定一个远端参与者；collective 则必须先确定一组共同参与的 PE。Team 为这组 PE 提供句柄、Team 内编号以及实现 collective 所需的上下文。预定义 Team 适合 world、node 等常见范围，应用也可以从父 Team 拆分出子 Team。

Team 的关键不是“给 PE 分组”这一句定义，而是参与契约：同一个 collective 实例必须由 Team 中所要求的 PE 以匹配参数和兼容顺序调用，Team 外 PE 不参与；同一 PE 在多个 Team 中的相对编号可能不同；同一 Team 上不允许并发的 collective 不能被任意提交到不同 stream 后期待 runtime 自动排序。B06 会区分 synchronization collective 与 data collective，并讨论 broadcast、reduce、collect/fcollect 和 all-to-all 的数据布局。

## 十、一个最小 Ring PUT 程序如何贯穿这些抽象

下面的教学示例让每个 PE 把自己的 world PE 编号写到下一个 PE 的 `inbox`。它展示的是生命周期与角色关系，不用于证明特定版本的性能或完成细节。

```cpp
// 教学示例：未编译、未运行，省略错误检查与头文件。
__global__ void ring_put(int *inbox) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int me = nvshmem_my_pe();
        int npes = nvshmem_n_pes();
        int peer = (me + 1) % npes;
        nvshmem_int_p(inbox, me, peer);
    }
}

int main() {
    nvshmem_init();
    int node_pe = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    cudaSetDevice(node_pe);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    int *inbox = static_cast<int *>(nvshmem_malloc(sizeof(int)));
    cudaMemsetAsync(inbox, 0, sizeof(int), stream);

    ring_put<<<1, 1, 0, stream>>>(inbox);
    nvshmemx_barrier_all_on_stream(stream);

    int received = -1;
    cudaMemcpyAsync(&received, inbox, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    nvshmem_free(inbox);
    cudaStreamDestroy(stream);
    nvshmem_finalize();
}
```

对这个程序可以按层解释。进程管理器提供多个 PE；node-relative PE 编号帮助每个 PE 选 GPU；collective `nvshmem_malloc()` 在各 PE 上建立对应 `inbox`；kernel 中唯一活跃的 CUDA thread 用 `<inbox, peer>` 标识远端对象；`nvshmem_int_p` 执行 one-sided 标量 PUT，远端没有匹配 receive；stream 上的 barrier 让所有 PE 共同到达后续阶段并满足该 API 规定的更新完成语义；CUDA stream 同步后，host 才读取本地结果并进入 collective free。

示例仍有明确边界。它使用全局 barrier，扩展性和并发性通常不如为真实 producer-consumer 设计的 signal/wait 协议；它假设节点内 PE 编号可以合法映射到设备编号；也没有处理初始化失败或设备数量不足。其价值在于展示“PE 目标、symmetric object、CUDA 执行者和同步点”是如何组合的，而不是提供可直接部署的完整程序。

## 十一、NVSHMEM 隐藏什么，又没有隐藏什么

与手写 verbs/RDMA 程序相比，NVSHMEM runtime 隐藏了大量控制面工作。应用在 PUT 中通常只传入本地源、symmetric destination、元素数量和目标 PE，不再显式携带远端 virtual address、`rkey` 与 QP。runtime 可以根据拓扑选择节点内 P2P、网络 transport、proxy 或 IBGDA 等路径，并解析远端对象所需的元数据。具体选择受构建、平台、配置和版本影响，不能由 API 名字单独推出。

被隐藏的是传输实现细节，不是并发正确性责任。应用仍须保证 symmetric address 和范围合法、各 PE 的 collective 匹配、源数据在通信读取期间稳定、消费者只在协议允许后读数据、不同 stream 和发起域建立了必要依赖，以及对象在所有潜在访问完成前不被释放。换言之，NVSHMEM 把“怎样驱动具体传输设备”提升为库职责，却保留“程序想表达什么数据依赖”这一应用职责。

底层差异还会重新表现为性能差异。相同 PUT 可能走 NVLink P2P、host-assisted proxy 或 GPU-initiated network path；消息粒度、访存合并、GPU/NIC 亲和性和并发资源都会影响延迟与带宽。因此编程模型提供可移植语义，不承诺不同 transport 上具有相同成本。C01 将区分 payload data path、请求提交 path 与 progress path，D 模块再进入固定版本的 IBGDA 实现。

## 十二、与 MPI、NCCL 的关系应按核心抽象比较

MPI、NCCL 与 NVSHMEM 可以出现在同一个应用中，不是简单的三选一。MPI 的核心抽象是 rank 间消息传递，虽然 MPI 也具有 RMA；NCCL 主要面向 GPU collective；NVSHMEM 的特色是以 symmetric memory 为基础的 one-sided、细粒度且可由 device 发起的通信。对比应聚焦主抽象，而不能据此断言某个库完全不支持表中其他能力。

| 维度 | MPI | NCCL | NVSHMEM |
| --- | --- | --- | --- |
| 主要编程抽象 | rank 与 message passing | GPU collective | PE、PGAS 与 symmetric object |
| 典型操作 | Send/Recv，另有 RMA | AllReduce、AllGather 等 | PUT/GET、AMO、signal/wait、collective |
| 接收方匹配 | 典型 point-to-point 需要匹配接收 | 所有参与 rank 匹配 collective | RMA 目标 PE 不需匹配 receive，但同步另行设计 |
| kernel 内细粒度通信 | 不是传统主路径 | 不是主要抽象 | device API 是核心能力之一 |
| 常见适用问题 | 通用分布式消息协议 | 规则的大规模张量集合通信 | 不规则访问、细粒度通知、persistent kernel 与 GPU 发起通信 |

实际选型还要看算法结构、消息大小、硬件支持、软件集成和团队维护成本。例如训练中的大张量 AllReduce 往往适合 NCCL；不规则图遍历或 GPU resident work queue 可能更适合 NVSHMEM；控制面或既有分布式框架又可能继续使用 MPI。E 模块会用 workload 和测量结果讨论，而不是仅凭抽象标签判断性能。

## 十三、常见错误及其根因

1. **把 PE 当成 CUDA thread。** PE 是分布式进程身份，CUDA thread 是某张 GPU 上的执行实体。前者决定远端参与者，后者决定本地并行工作。
2. **把 PGAS 当成硬件缓存一致的共享内存。** PGAS 提供全局寻址与通信语义，不会让所有 GPU 对普通 load/store 自动保持强一致。
3. **把 symmetric object 当成所有 PE 共用的一份物理对象。** 实际上每个 PE 有自己的对应分区；“对称”约束逻辑对应关系，不要求内容或指针数值相同。
4. **只让部分 PE 调用对称分配或 collective。** SPMD 允许一般控制流分化，但集合操作必须满足参与域和匹配规则。
5. **认为 one-sided 意味着不需要同步。** PUT/GET 不需要匹配 receive，却仍需要完成、排序、通知与 ownership 协议。
6. **把 API 返回、kernel 完成与远端消费确认视为同一事件。** 它们属于不同阶段，必须依据具体接口和执行域证明。
7. **交换另一个 PE 的裸 symmetric pointer。** symmetric address 只在生成它的 PE 上作为本地地址和 NVSHMEM 参数有效；远端由本地 symmetric address 与目标 PE 共同表达。
8. **认为 collective launch 能消除所有死锁。** 它提供 cooperative residency 条件，但无法修复不匹配 collective、跨 stream 循环依赖或错误的应用协议。

## 十四、把本课接到后续路线

本课的主线可以压缩为：job 由多个 PE 构成，每个 PE 通常控制一张 GPU；每个 PE 在自己的 symmetric heap 中持有逻辑对应的对象；`<symmetric address, target PE>` 标识远端对象；RMA、AMO 和 signal 等原语操作这些对象；Team 确定集合参与域；host、stream 和 device API 决定操作在哪个提交与执行域中发生；completion、ordering、visibility 与 ownership 决定结果何时可以被安全使用。

接下来应依次阅读：

- [B01：对称对象与远端寻址](b01-symmetric-objects-and-remote-addressing.md)，证明 `<symmetric address, PE>` 如何定位对象并区分 UVA、direct pointer 与 registration；
- [B02：完成、排序与可见性](b02-completeness-ordering-and-visibility.md)，区分 blocking/NBI、fence/quiet、signal/wait 与不同完成阶段；
- [B03：CUDA 执行域、协作通信与前进性](b03-cuda-execution-and-cooperation.md)，处理 thread-group、stream、collective launch 和等待环；
- [B04：数据发布、缓冲区所有权与有界流水协议](b04-data-publishing-and-buffer-ownership.md)，把抽象语义落实成可审查的 producer-consumer 协议。

完成上述内容后，再学习 B05 原子协调、B06 Team/data collectives 与 C00 初始化控制面。到那时，读者应能解释：NVSHMEM runtime 为什么可以隐藏 RDMA 地址/key 交换，却不能替应用决定数据何时可覆盖；为什么 GPU 发起通信降低了 host 参与，却没有取消 CUDA 与网络两套顺序关系；以及为什么同一个 API 语义可以由不同 transport 实现而表现出不同成本。

## 参考资料

- NVIDIA, [Using NVSHMEM](https://docs.nvidia.com/nvshmem/api/latest/using.html)：PE/SPMD、PGAS、symmetric address、GPU 映射与总体使用流程。
- NVIDIA, [Library Setup, Exit, and Query](https://docs.nvidia.com/nvshmem/api/latest/gen/api/setup.html)：初始化、finalize、PE 查询与 host thread support。
- NVIDIA, [Memory Management](https://docs.nvidia.com/nvshmem/api/latest/gen/api/memory.html)：对称分配、参数一致性、隐含 barrier 与 heap 策略。
- NVIDIA, [Remote Memory Access](https://docs.nvidia.com/nvshmem/api/latest/gen/api/rma.html)：PUT/GET、one-sided 操作及通信与同步的分离。
- NVIDIA, [NVSHMEM and the CUDA Model](https://docs.nvidia.com/nvshmem/api/latest/cuda-interactions.html)：stream、device operation、nonlocal dependency 与 CUDA forward progress。
- NVIDIA, [Kernel Launch Routines](https://docs.nvidia.com/nvshmem/api/latest/api/launch.html)：`nvshmemx_collective_launch` 的适用条件与 grid 限制。
- NVIDIA, [Collective Communication](https://docs.nvidia.com/nvshmem/api/latest/gen/api/collectives.html)：Team-based/implicit collective 与并发访问边界。
