# C00 NVSHMEM 运行时初始化与控制平面：从独立进程到可通信的 PE

## 本课要解决的问题

前面的 B00—B06 已经从应用视角解释 PE、symmetric object、RMA、完成语义、CUDA 执行域、原子协调和 Team collective，但这些抽象并不会在进程启动时自然存在。操作系统最初看到的只是若干相互独立的进程，CUDA runtime 也只知道本机可见的 GPU；在第一个 PUT 能够成立以前，系统至少要回答：哪些进程属于同一个作业，每个进程的 PE 编号是什么，它绑定哪张 GPU，节点内哪些 PE 可以直接访问彼此显存，跨节点采用哪个 transport，对称对象如何获得远端可达性，以及 device code 从哪里取得这些运行时信息。

这些准备工作统称为 control plane。它建立成员关系、能力、资源与元数据；真正搬运应用 payload 的 PUT、GET、AMO 或 collective 属于 data plane。二者不能按“CPU 路径”和“GPU 路径”简单划分：IBGDA 可以让稳态网络请求由 GPU 提交，但初始化阶段仍需进程管理器、CPU、CUDA driver、NIC driver 和 NVSHMEM runtime 创建并交换资源。反过来，使用 MPI 或 PMI bootstrap 也不表示后续 payload 必然经过 MPI。

本课从公开 API 可观察的状态机出发，解释 `nvshmem_init`、`nvshmem_init_thread`、`nvshmemx_init_attr`、bootstrap、CUDA device selection、deferred device initialization、symmetric heap 与 transport 准备、错误处理和 `nvshmem_finalize`。读完后，应该能够诊断“程序卡在 init”“PE 选错 GPU”“对称分配失败”“MPI 已初始化却 NVSHMEM 无法通信”等问题，并能区分规范保证、运行时职责与需要固定源码才能证实的实现细节。

## 资料版本与论述边界

官方 Execution Model 对 `nvshmem_init`/`nvshmem_init_thread`、collective `nvshmem_finalize`、`nvshmem_global_exit` 和 finalize 后不得重新初始化的英文原文与中文翻译，见 [双语核对笔记：NVSHMEM Execution Model](official-docs/r04-execution-model-and-progress.md)。本课在该生命周期契约上继续展开 bootstrap、heap、transport 和 reverse cleanup。

官方 Using NVSHMEM 页面关于 MPI communicator 初始化、PE→GPU 选择、host/device libraries、PMI/PMIx 与 launcher 的完整双语材料，见 [双语核对笔记：Using NVSHMEM](official-docs/r05-using-nvshmem.md)。其中 launcher/bootstrap 与 payload transport 是不同层次，不能由 `mpirun` 或 `srun` 反推数据路径。

本文于 2026-09-13 依据 NVIDIA NVSHMEM API Guide 的滚动 `latest` 页面核对，并以 [NVSHMEM 3.7.2 Release Notes](https://docs.nvidia.com/nvshmem/release-notes-install-guide/release-notes/release-3720.html)标识发布周期。API 状态、默认环境变量和可选 transport 可能随版本变化；本文没有固定 NVSHMEM 源码 commit，因此不会把某个版本的内部函数名、初始化线程、endpoint 数量、QP 建立顺序或 key 数据结构写成长期契约。

文中把初始化拆成多个概念阶段，其中“bootstrapping 与 device initialization 两阶段”是当前官方文档明确给出的状态模型；“拓扑发现、内存注册、peer metadata/endpoint 准备、device state 发布”则是根据后续通信必须拥有的信息所作的系统性分解。后者用于解释职责和排错，不代表公开 API 保证按此顺序逐项执行，也不保证每种 transport 都创建相同资源。

代码片段是教学骨架，未编译、未运行，没有 MPI、Slurm、GPU、InfiniBand、RoCE、NVLink 或 IBGDA 环境验证。真实部署必须以安装版本的启动日志、构建选项、进程管理器、GPU/NIC 拓扑和 transport 状态为准。

## 一、先区分 launcher、bootstrap、runtime 与 transport

Launcher 负责创建进程并把它们放进同一个作业环境，例如 `srun`、`mpirun` 或 NVSHMEM 随附的 Hydra launcher。Bootstrap 负责让这些进程在初始化早期发现成员、获得 rank/size，并执行少量控制信息交换；当前文档列出的默认 bootstrap 选择包括 PMI、MPI、SHMEM 和 plugin，PMI 又可选择 PMI、PMI-2 或 PMIx。NVSHMEM runtime 在 bootstrap 建立的成员集合上创建 PE、Team、对称内存和通信状态，而 transport 才负责后续节点内或跨节点数据操作。

这四层经常组合出现，却回答不同问题。`mpirun -n 8 app` 可以只是用 MPI launcher 启动八个进程，NVSHMEM 仍可能通过 PMI bootstrap；也可以用 `nvshmemx_init_attr` 显式传入一个 MPI communicator，使该 communicator 的 ranks 成为 NVSHMEM world PEs。无论哪种方式，后续远程 PUT 可能选择 IB RC、UCX、libfabric、IBGDA 或其他当前构建支持的 transport，不能由 launcher 名字反推 data path。

```text
launcher
  └─ 创建进程、提供作业环境
       └─ bootstrap
            ├─ 建立成员集合、rank/size
            └─ 提供初始化期控制交换
                 └─ NVSHMEM runtime
                      ├─ PE/Team 与 CUDA device state
                      ├─ symmetric heap 与远端元数据
                      └─ transport resources
                           └─ data plane：PUT/GET/AMO/collective
```

如果错误发生在 `nvshmem_bootstrap_mpi.so` 加载阶段，首先应检查 bootstrap plugin 和动态库搜索路径，而不是检查 QP 数据路径；如果 bootstrap 成功后 GPU memory registration 失败，问题已进入 device/transport 准备阶段。按层定位比笼统地说“NVSHMEM 初始化失败”更有诊断价值。

## 二、公开状态机不是简单的未初始化/已初始化二值

当前 NVSHMEM 文档把初始化明确分成 bootstrapping 与 device initialization 两阶段。初始状态是 `NVSHMEM_STATUS_NOT_INITIALIZED`；完成成员发现、但尚未完成 CUDA device 相关准备时是 `NVSHMEM_STATUS_IS_BOOTSTRAPPED`；完成 device initialization 后，根据 GPU 是否由单个 PE 独占或处于不同级别的 Multi-Process GPU（MPG）共享模式，状态可能是 `NVSHMEM_STATUS_IS_INITIALIZED`、`NVSHMEM_STATUS_LIMITED_MPG` 或 `NVSHMEM_STATUS_FULL_MPG`。应用可用 `nvshmemx_init_status()` 查询当前组件观察到的状态。

如果调用初始化 API 前尚未设置 CUDA device，runtime 可以只完成 bootstrap，把 device initialization 延迟到后续某个会触发集合初始化的操作。官方列举的触发点包括 `nvshmem_malloc/calloc/align`、`nvshmem_barrier_all`、`nvshmem_sync_all` 及其 on-stream world 版本。在 bootstrapped 状态下，host 可调用 PE 数量、world PE 和 node Team 相对编号等查询来选择 GPU；但在 device initialization 完成前，device code 不能调用包括 PE query 在内的 NVSHMEM device API。

```text
NOT_INITIALIZED
      │ collective init
      ▼
BOOTSTRAPPED ── host PE/node queries ──> cudaSetDevice(...)
      │                                   │
      └──── first triggering operation ───┘
                         ▼
        INITIALIZED / LIMITED_MPG / FULL_MPG
                         │
                         │ nvshmem_finalize
                         ▼
                  BOOTSTRAPPED state
```

图中的 finalize 回到 bootstrapped state 是当前文档对内部状态的描述，不表示应用可以忽略 init/finalize 配对随意重启。多次初始化虽然被允许，但必须由与首次相同的进程集合调用，每次初始化都要有匹配的 finalize；动态组件还可能具有各自观察到的初始化状态。一般应用应采用一次清晰的 NVSHMEM 生命周期，避免把引用计数式多重初始化作为常规控制流。

## 三、GPU 选择位于 bootstrap 与 device initialization 之间

最常见的一 PE 一 GPU 部署中，各 PE 先初始化并取得 node-relative PE 编号，再用该编号选择本地 GPU。关键条件不是“在 `nvshmem_init` 之前必须无条件调用 `cudaSetDevice`”，而是 CUDA device 必须在任何会触发 device initialization 的其他 NVSHMEM 操作之前确定。官方示例因此允许先 `nvshmem_init()`，在 bootstrapped 状态查询 `NVSHMEMX_TEAM_NODE` 内的编号，再执行 `cudaSetDevice(node_pe)`，随后进行第一个对称分配或同步。

```cpp
// 教学骨架：未编译、未运行，省略完整错误收集。
nvshmem_init();

int node_pe = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
int device_count = 0;
cudaGetDeviceCount(&device_count);
if (node_pe < 0 || node_pe >= device_count) {
    nvshmem_global_exit(1);
}
cudaSetDevice(node_pe);

// 第一个对称分配可触发延迟的 device initialization。
int *mailbox = static_cast<int *>(nvshmem_calloc(1, sizeof(int)));
if (mailbox == nullptr) {
    nvshmem_global_exit(2);
}

// 后续 kernel、RMA、collective 位于完整初始化阶段。
nvshmem_free(mailbox);
nvshmem_finalize();
```

`node_pe -> CUDA device ordinal` 是常见映射，不是对所有容器、MIG、GPU visibility 或多 PE 共享 GPU 环境都成立的定理。`CUDA_VISIBLE_DEVICES` 会改变进程看到的 device ordinal；调度器可能已按进程裁剪设备可见集；MPG 场景则可能故意让多个 PE 共享同一 GPU。因此生产程序应同时核对 local PE 数量、可见 device 数量、调度器赋值和目标 MPG 模式，不应在越界后让某些 PE 局部返回，因为其他 PE 可能继续等待 collective。

## 四、三类初始化入口解决不同的作业集成问题

`nvshmem_init()` 是最简单的入口：所有 PEs collective 调用，失败时程序以非零状态退出。`nvshmem_init_thread(requested, &provided)` 在相同初始化基础上请求 host/on-stream API 的线程支持级别，并以 `provided` 返回实际级别。当前文档说明 host 与 stream APIs 支持 `NVSHMEM_THREAD_SERIALIZED`，device APIs 支持 `NVSHMEM_THREAD_MULTIPLE`；应用仍应以运行时返回值为准，尤其不能在请求 `MULTIPLE` 后不检查 `provided` 就并发调用 host collective。

四个线程级别从严格到宽松依次是 SINGLE、FUNNELED、SERIALIZED 和 MULTIPLE。FUNNELED 允许进程有多个 host threads，但只有执行初始化的主线程调用 NVSHMEM；SERIALIZED 允许不同线程调用，但应用必须保证调用不并发；MULTIPLE 才允许任意线程并发调用。线程安全也不取消 collective matching：多个 host threads 即使被 API 允许调用，也仍要为同一 Team 建立一致顺序。

`nvshmemx_init_attr(flags, &attr)` 用于复用已有 MPI communicator、已有 OpenSHMEM job，或使用 launcher-agnostic Unique ID（UID）建立作业。MPI 模式中，传入 communicator 的各 rank 构成 NVSHMEM world Team；NVSHMEM 必须在该 communicator 被销毁之前 finalize。OpenSHMEM 模式同理要求先结束 NVSHMEM，再结束外层 OpenSHMEM。UID 模式由一个参与者取得 opaque UID，应用先通过某种外部机制把同一 token、各自 rank 和总 rank 数分发给所有进程，再 collective 调用 attribute init。

UID 只解决“如何把这些进程聚成同一个 NVSHMEM job”，不替代生产部署中的进程创建、故障管理和安全分发。若两个不相关作业误用同一 session 信息，或 ranks/nranks 不一致，初始化无法形成正确成员集合。MPI communicator 也只是 membership/bootstrap 边界，不保证 NVSHMEM 后续使用 MPI transport 搬 payload。

## 五、bootstrap 的最小职责是建立共同身份和控制交换能力

从 API 视角看，bootstrap 完成后，runtime 已能回答 `nvshmem_my_pe()`、`nvshmem_n_pes()` 和 node Team 查询。这说明所有参与进程必须对成员集合、唯一 PE 编号、world 大小以及至少一部分节点归属达成一致。初始化后续还需要跨 PE 交换配置和远端访问元数据，因此 bootstrap 必须提供某种控制信息交换和 collective 协调能力，但应用不应假设它的具体消息协议。

PMI/PMIx 常由 Slurm 或 Hydra 等进程管理环境提供；MPI/OpenSHMEM plugin 复用已有通信环境；UID bootstrap 使用 IP-based networking，并可通过 `NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME` 和 address family/session 设置选择控制网络。这里的 socket/interface 只属于 UID bootstrap 控制面，不能据此判定 payload 走以太网、RoCE 还是 InfiniBand。控制面接口与数据面 NIC 可能相同，也可能完全不同。

`NVSHMEM_BOOTSTRAP`、`NVSHMEM_BOOTSTRAP_PMI` 与 plugin 路径必须在所有 PEs 上兼容。某个 PE 加载了不同 MPI ABI、缺少 bootstrap shared object 或看到了不同环境变量时，故障可能表现为一部分进程提前退出、另一部分卡在 init。排查时应先确认每个节点的库文件和环境一致，再进入 GPU/NIC 层。

## 六、device initialization 把 PE 绑定到具体 CUDA 与通信资源

完成 `cudaSetDevice` 后，runtime 才能围绕选定 GPU 建立 device-dependent 状态。公开契约不枚举内部步骤，但为了让 device API、对称内存与 transport 可用，运行时在概念上必须完成若干职责：识别本地 GPU 及 peer accessibility，判断节点内 P2P 能力，发现可用远程 transport 和 NIC，为 symmetric heap 建立 CUDA/transport 可访问性，准备访问远端 PE 所需的地址与凭据，并让 device code 能读取必要的 PE、Team 和通信元数据。

这些职责不是固定顺序。使用动态 CUDA VMM heap 与静态预分配 heap 的内存建立方式不同；节点内 P2P 可能不需要网络 endpoint；CPU proxy transport、IB RC、UCX 与 IBGDA 对队列和 device-visible metadata 的需求也不同。即使两个版本都能执行同一个 `nvshmem_putmem`，初始化阶段创建的内部资源仍可能完全不同。

可以把“第一个 PUT 前必须成立”的逻辑条件整理为：调用 PE 已有稳定身份；目标 PE 属于同一 job；本地 GPU 已确定；操作数位于合法且仍存活的内存范围；所选 path 能访问该内存；runtime 能把 `<symmetric address, target PE>` 解析成该 path 所需的远端定位信息；提交和完成机制已经建立。这个条件表比猜测“init 一定先建 QP 再注册 MR”更稳定。

## 七、symmetric heap 同时是地址契约和 transport 资源

从应用看，symmetric heap 是所有 PEs 以一致顺序和参数分配对应对象的空间；从运行时看，这些对象还必须能被本地 CUDA 和所选通信 path 正确访问。当前 NVSHMEM 默认支持基于 CUDA VMM 的动态 symmetric heap；设置 `NVSHMEM_DISABLE_CUDA_VMM` 会改用静态 heap，此时总容量在 job 启动阶段确定，可用 `NVSHMEM_SYMMETRIC_SIZE` 配置每 PE heap 大小。当前滚动文档给出的默认值是每 PE 1 GiB，但它属于版本化默认值，不应写死为应用长期假设。

静态 heap 的容量计划必须包括应用对象、对齐与 runtime 可能保留的空间，而不是只把数组 payload 大小相加。动态 VMM 降低了必须预估整个 heap 的压力，却不取消物理显存、虚拟地址空间、registration、映射粒度和 transport capability 限制。设置很大的 `NVSHMEM_SYMMETRIC_SIZE` 也不会凭空创造 GPU memory；设置非法值则会在初始化期间报错或导致程序终止。

`nvshmem_malloc/calloc/align/free` 是 world collective，并以一致参数维持各 PE 上对象的对应关系。非零 allocation 在退出前包含与 `barrier_all` 等价的过程，free 在入口包含相应过程；这保证分配返回时远端对应对象已建立，却不表示对象内容已被应用初始化，也不表示 future RMA 自动同步。第一个 allocation 还可能触发 deferred device initialization，因此“卡在 malloc”有时不是 allocator 本身，而是此前延迟的 GPU、heap 或 transport 初始化失败。

普通 buffer 经 `nvshmemx_buffer_register` 注册后可以作为本地 RMA/AMO operand，但不因此成为远端 symmetric object；NVSHMEM heap 本来就由 runtime 注册，不应再次调用该接口。这个区别说明“内存已注册”和“属于 PGAS 对称寻址空间”是两个条件。

## 八、transport 选择是能力、配置与拓扑共同作用的结果

当前环境变量页面提供 `NVSHMEM_REMOTE_TRANSPORT` 等选项，并列出 ibrc、ucx、libfabric、ibdevx、gpunetio、none 等当前可选值；实际可用集合取决于 library 构建和平台。`none` 表示禁用 remote transport，只适用于不需要跨节点远端 path 的配置。节点内 peer path、跨节点 remote transport、collective backend 和 bootstrap 仍是不同选择层，不能把一个变量解释成完整通信拓扑。

初始化时 runtime 需要把 PE 映射到可用 NIC 或 transport resource。当前默认 NIC 策略会考虑 proximity 并在本地 PEs 间平衡，也可以通过 NIC–PE mapping、HCA list 等选项干预。这里“最近”是 runtime 在可见 topology 上的选择结果，不保证等于用户从主板图纸推测的最短路径，也不证明某次操作已使用该 NIC。C02 需要结合 `nvidia-smi topo -m`、RDMA device 信息、启动日志和受控运行验证实际映射。

IBGDA 尤其说明控制面不会因 GPU-initiated fast path 消失。CPU/runtime 仍须创建 GPU 可访问的 NIC work queues、doorbell/completion 状态和远端访问元数据，稳态 GPU thread 才可能直接提交。若构建不包含目标 transport、GPUDirect RDMA registration 失败、NIC handler 能力不足或 topology 不满足，初始化可能失败或选择另一条支持路径；具体 fallback 必须以该版本日志和配置为证据。

## 九、peer metadata 与 device state 应理解为逻辑义务

一次远端 RMA 最终必须定位目标内存并满足访问权限。传统 verbs 程序显式交换 remote address、rkey 和 QP 信息；NVSHMEM 把这些细节隐藏在初始化与运行时元数据中。因而可以合理推论：针对需要远端寻址的 transport，各 PEs 必须以某种形式获得或可查询 peer heap 定位、访问凭据和 endpoint/connection 状态；device API 还需要把足够的信息放到 GPU 可访问的状态中。

这段结论应停留在逻辑层。不能在未固定源码时断言所有 PEs 都保存完整 $N\times N$ endpoint 表，也不能断言 symmetric remote address 一律由 `remote_heap_base + local_offset` 计算。RC、DCI/DCT、UCX、libfabric、节点内 direct pointer 和按需连接可能采用不同压缩、缓存或查找结构。是否 eager 建立所有连接、何时注册新 VMM segment、device state 中保存哪些字段，都属于 D00—D03 的源码问题。

同样，初始化成功只证明 runtime 已建立公开操作所需的基础状态，不证明某个特定性能路径已启用。要声称“本作业使用 IBGDA GPU handler”，还需核对构建、remote transport、handler 最终选择、日志与实际 GPU–NIC 环境；只看到 `nvshmem_init` 返回不构成证据。

## 十、正常退出必须先结束应用工作，再释放控制面

`nvshmem_finalize()` 是 world collective，包含隐式全局 barrier，用于在所有 PEs 进入后完成 pending communications，并释放 NVSHMEM resources，包括 symmetric heap 和由 `nvshmem_ptr` 获得的相关指针状态。它必须是 NVSHMEM 阶段的最后一个 library call；返回后 OS processes 仍可继续执行普通程序代码，但不能再访问已释放的 NVSHMEM 资源。

隐式 barrier 不能替应用“召回”尚未到达 finalize 的 CUDA kernel。若 stream 中仍排有会继续发起 NVSHMEM 操作的 kernel，host 直接进入 finalize 会让其他 PE 的退出顺序和资源生命周期难以成立。稳妥的关闭顺序是先停止产生新工作，等待所有相关 CUDA streams/kernels 完成，排空业务层 ready/ack 或 work queue，再按 collective 顺序销毁自定义 Teams、注销外部 registered buffers、释放 symmetric allocations，最后由所有 PEs 调用 finalize。

```text
停止新请求
  -> 等待 kernels/streams 结束
  -> 完成通信并排空应用协议
  -> destroy application Teams
  -> unregister local registered buffers
  -> collective free symmetric objects
  -> collective nvshmem_finalize
  -> 再销毁外层 MPI/OpenSHMEM bootstrap 对象
```

若 NVSHMEM 基于用户提供的 MPI communicator 初始化，必须先 `nvshmem_finalize`，再销毁 communicator 或调用 `MPI_Finalize`。普通成功退出和故障终止也不同：`nvshmem_global_exit(status)` 是任一 PE 可调用的非集合终止接口，会要求整个 NVSHMEM program 立即终止；它适合无法安全进入 collective cleanup 的全局致命故障，但不提供正常 finalize 的资源与业务协议保证。

## 十一、初始化故障应按阶段建立证据链

初始化卡住首先检查所有预期 PEs 是否都启动、是否调用了相同初始化入口、环境变量和 bootstrap plugin 是否一致。如果日志显示 plugin 无法加载，应核对 NVSHMEM library 路径、`LD_LIBRARY_PATH` 或显式 plugin 配置，以及 plugin 编译所用 MPI/OpenSHMEM ABI。此时调整 NIC 或 QP 数量通常无关。

若 PE/rank 查询成功但首个 `nvshmem_malloc` 或 barrier 失败，应想到 deferred device initialization：检查每个 PE 的 `cudaSetDevice` 是否在触发点之前成功、local PE 到 visible device 的映射是否越界、是否意外多 PE 共享 GPU，以及 symmetric heap 配置是否合法。随后再检查 GPUDirect RDMA module/DMA-BUF、GPU memory registration、remote transport 构建和 NIC 可见性。

建议在受控诊断中记录 `NVSHMEM_VERSION=1`、`NVSHMEM_INFO=1` 与适当的 `NVSHMEM_DEBUG` 输出，同时保留 launcher 命令、每节点 PE 数、host name、world/node rank、CUDA device ordinal/UUID、bootstrap 选择、remote transport 与 HCA mapping。日志可能包含系统路径和拓扑信息，分享前应按环境安全要求处理；也不应在性能正式测量中默认保留 TRACE 级日志。

一个 PE 上的 CUDA 错误不能简单 `return`，因为其他 PEs 可能正等待下一次 collective。可恢复错误需要通过已有控制面让所有 PEs 走一致的清理分支；无法恢复且无法保证集合清理时，使用全局终止机制通常比让部分 PEs 悬挂更明确。具体生产容错仍取决于 launcher、MPI/PMI 和集群策略，NVSHMEM 初始化本身不等于通用故障恢复框架。

## 十二、从启动到退出的完整教学骨架

下面的顺序适用于普通一 PE 一 GPU 作业，用于表达生命周期而非展示完整错误处理。其重要之处在于：只有 bootstrapped-state queries 位于 `cudaSetDevice` 之前；所有可能触发 device initialization 的对称分配、同步和通信都位于 device selection 之后；退出前明确同步 stream 并释放对象。

```cpp
// 教学示例：未编译、未运行。
int provided = NVSHMEM_THREAD_SINGLE;
int rc = nvshmem_init_thread(NVSHMEM_THREAD_SERIALIZED, &provided);
if (rc != 0) {
    // 初始化未成功，后续 NVSHMEM 调用的行为没有保证。
    return 1;
}
if (provided < NVSHMEM_THREAD_SERIALIZED) {
    // 初始化已经成功，因此先由所有 PEs 匹配 finalize。
    nvshmem_finalize();
    return 1;
}

int world_pe = nvshmem_my_pe();
int world_npes = nvshmem_n_pes();
int node_pe = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

int ndevices = 0;
cudaError_t ce = cudaGetDeviceCount(&ndevices);
if (ce != cudaSuccess || node_pe < 0 || node_pe >= ndevices) {
    nvshmem_global_exit(2);
}
cudaSetDevice(node_pe);

int *inbox = static_cast<int *>(nvshmem_calloc(1, sizeof(int)));
if (inbox == nullptr) {
    nvshmem_global_exit(3);
}

cudaStream_t stream;
cudaStreamCreate(&stream);

// launch application kernels and NVSHMEM operations on the selected GPU
// ...

nvshmemx_barrier_all_on_stream(stream);
cudaStreamSynchronize(stream);

nvshmem_free(inbox);
cudaStreamDestroy(stream);
nvshmem_finalize();

// world_pe/world_npes 只是普通整数仍可打印；NVSHMEM API 不可再调用。
```

示例中的错误分支也有边界。`nvshmem_init_thread` 自身失败后，任何后续 NVSHMEM 调用的行为都没有保证，所以示例直接返回；若初始化成功但线程级别不足，所有 PEs 应匹配 finalize。初始化成功后，某个 PE 又发现 device mapping 或 allocation 致命错误时，示例才使用 `nvshmem_global_exit` 避免其他 PEs 永久等待。生产程序应结合作业管理器统一上报错误，而不是照抄简化分支。

## 十三、常见误解与纠正

1. **“`mpirun` 启动，所以 payload 一定走 MPI。”** Launcher/bootstrap 与 data transport 是独立层；必须查看实际 remote transport。
2. **“`nvshmem_init` 返回后，任何 GPU 都可以再选。”** 未设置 device 时可能只完成 bootstrap；必须在第一个触发 device initialization 的其他操作前调用 `cudaSetDevice`。
3. **“卡在第一个 malloc，所以只是显存不足。”** 首个 world collective allocation 可能触发延迟 device/transport 初始化，还应检查 PE–GPU 映射、heap 与 registration。
4. **“symmetric heap 就是一块所有 GPU 缓存一致的共享显存。”** 它是各 PE 对应分区及远端访问契约，仍需 NVSHMEM ordering/completion 和应用同步。
5. **“MPI communicator 传给 init 后可立即释放。”** 它定义 NVSHMEM world membership，必须存活到 NVSHMEM finalize 之后。
6. **“UID 模式不需要任何外部协调。”** 应用仍需分发同一 UID、唯一 ranks 和一致 nranks，并负责创建进程。
7. **“启用 IBGDA 后初始化不再使用 CPU。”** GPU fast path 仍依赖 CPU/runtime/driver 建立控制面资源。
8. **“finalize 的 barrier 会自动等待尚未启动完的所有 kernel。”** 应用必须先停止并同步可能继续使用 NVSHMEM 的异步 CUDA 工作。
9. **“初始化成功证明性能路径生效。”** 它只证明当前配置建立了可用运行时；具体 transport/handler 需要日志和实验。

## 十四、掌握标准与后续路线

完成本课后，读者应能把一次启动画成 launcher → bootstrap → PE/device binding → heap/transport preparation → data plane，并说明每一层的输入、输出和典型故障。面对 `nvshmem_init` 后再 `cudaSetDevice` 的代码，应能解释它为何在 bootstrapped 两阶段模型中合法，以及为什么 `nvshmem_malloc` 之前必须完成 device selection。

读者还应能选择 `init`、`init_thread` 或 attribute init，说明 MPI/OpenSHMEM/UID 对外部对象生命周期的约束；能够比较动态 VMM heap 与静态 heap，解释为什么第一个 allocation 可能暴露延迟初始化错误；能够设计从停止 kernel 到 finalize 的反向释放顺序，而不把 global exit 当作正常清理。

C00 与 C01 共同完成“控制面—数据路径”的分层：本文回答第一个通信操作之前 runtime 必须准备什么，[C01：从 host RDMA 到 GPUDirect RDMA 与 IBGDA](c01-host-rdma-to-gpudirect-and-ibgda.md)回答稳态请求由谁提交、payload 经哪里移动。下一篇 C02 尚未创建，将把 bootstrap、device、heap、transport、NIC mapping 和 handler 变成实际环境核验清单；本文不为不存在的文件建立链接。

## 主要来源

- NVIDIA, [Library Setup, Exit, and Query](https://docs.nvidia.com/nvshmem/api/latest/gen/api/setup.html)：初始化入口、MPI/OpenSHMEM/UID attribute、两阶段状态、线程级别、finalize 与 global exit。
- NVIDIA, [Using NVSHMEM](https://docs.nvidia.com/nvshmem/api/latest/using.html)：GPU 选择时序、PMI/Hydra/mpirun/srun 启动、编译链接与通信模型。
- NVIDIA, [Environment Variables](https://docs.nvidia.com/nvshmem/api/latest/gen/env.html)：bootstrap、symmetric heap、调试、remote transport 与 NIC mapping 的当前选项和默认值。
- NVIDIA, [Memory Management](https://docs.nvidia.com/nvshmem/api/latest/gen/api/memory.html)：collective symmetric allocation、动态 VMM/static heap、barrier 位置与普通 buffer registration 边界。
- NVIDIA, [Troubleshooting and FAQs](https://docs.nvidia.com/nvshmem/api/latest/faq.html)：bootstrap plugin 加载、GPU memory registration 与常见运行环境故障。
- NVIDIA, [NVSHMEM 3.7.2 Release Notes](https://docs.nvidia.com/nvshmem/release-notes-install-guide/release-notes/release-3720.html)：本文核对时的发布周期；滚动 API 页面不等同于固定源码 commit。
