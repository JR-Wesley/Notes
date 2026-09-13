# B06 NVSHMEM Team 与集合通信：参与域、数据布局和并发匹配

## 本课要解决的问题

PUT、GET、AMO 和 signal 以一个目标 PE 为主要对象；collective communication 则要求一组 PE 共同完成同一个同步或数据变换。程序不仅要知道“数据发给谁”，还要定义哪些 PE 属于同一参与域、各 PE 在这个域中的编号、所有参与者正在匹配哪一次 collective，以及每个 PE 的 source/dest 应按什么布局解释。Team 正是承载这些信息的通信上下文。

初学者常把 collective 简化成“大家都执行同一个函数”。这个描述不够严格：world PE 0 不一定是子 Team 的 PE 0；不同 PE 若以不同顺序调用 broadcast 和 reduce，会把不同实例错误匹配；两个 CUDA streams 若同时在同一 Team 上提交 collective，会违反并发规则；即使函数返回成功，也只说明当前 PE 的本地返回边界，不必然说明其他 PE 已同时返回。对于 all-to-all 和 fcollect，参数完全匹配仍可能因缓冲区大小或 block 顺序错误而越界。

本课围绕三条主线展开。第一条是 Team 生命周期与编号翻译；第二条是 collective matching、执行域与前进性；第三条是 barrier/sync、broadcast、fcollect、all-to-all 和 reduction 的数据与完成语义。完成本课后，应该能够仅凭 Team 成员表和每个 PE 的 source 内容，手工计算 collective 后各 PE 的 dest，而不是把 collective 当作不可分析的黑盒。

## 资料版本与论述边界

本文于 2026-09-13 依据 NVIDIA 官方 NVSHMEM API Guide 的滚动 `latest` 页面核对，并以 [NVSHMEM 3.7.2 Release Notes](https://docs.nvidia.com/nvshmem/release-notes-install-guide/release-notes/release-3720.html)标识发布周期。主要依据是 Team Management、Collective Communication、NVSHMEM and the CUDA Model、Kernel Launch Routines 与 Memory Ordering。

本文讨论公开 API 契约，不假定 broadcast、reduce 或 all-to-all 必然由某组 PUT/GET、signal 或 AMO 实现，也不猜测具体算法是 ring、tree、recursive doubling、NVLS 还是 transport-specific protocol。算法、临时资源和网络路径会随版本、消息规模、Team、拓扑与配置变化；固定实现必须在后续源码课中核对。

代码和布局示例用于纸面推导，未编译、未运行，也没有 GPU、NVLink、InfiniBand 或 IBGDA 实测结果。示例省略错误处理，真实程序必须检查 Team 创建、CUDA 调用、内存容量和 collective 返回状态。

## 一、Team 是 PE 集合、编号空间和 collective 资源上下文

一个 Team 至少包含一个 PE，并通过不透明的 `nvshmem_team_t` handle 引用。Team handle 不是 symmetric object，不能作为远端内存传输；它只在本地程序中标识一组成员及该组 collective 使用的上下文。预定义的 `NVSHMEM_TEAM_WORLD` 包含整个 NVSHMEM job，PE 在 world team 中的编号与 `nvshmem_my_pe()` 相同。

`NVSHMEMX_TEAM_NODE` 是常用的预定义扩展 Team，包含同一节点内的 PE。它的 PE 0 表示该节点子组中的第一个成员，不等于全局 world PE 0。类似地，`NVSHMEM_TEAM_SHARED` 描述成员间 symmetric heap 都可通过 `nvshmem_ptr` 直接访问的集合；它是可达性语义，不应未经查询就假设等于 node team。

应用定义的 Team 可以由父 Team split 得到。创建成功后，成员在新 Team 中从 0 连续编号；未被选入的父 Team 成员取得 `NVSHMEM_TEAM_INVALID`。因此，一个 OS process 同时拥有 world rank、node-relative rank 和若干 application-team rank 是正常现象。任何函数参数中的 PE 编号都必须结合它所属的 Team 解释。

```text
WORLD members:       [0, 1, 2, 3, 4, 5]
even Team members:   [0,    2,    4   ]
even Team rank:       0     1     2

world PE 2 在 even Team 中是 PE 1；
even Team 的 root=0 对应 world PE 0；
even Team 的 target=2 对应 world PE 4。
```

## 二、Team-relative 编号不能直接当作 world PE 编号

`nvshmem_team_my_pe(team)` 返回调用者在指定 Team 中的相对编号，`nvshmem_team_n_pes(team)` 返回 Team 大小。若 handle 恰为 `NVSHMEM_TEAM_INVALID`，这两个查询返回 -1；对其他损坏或失效 handle 的行为不能依赖。Team collective 的 `PE_root` 等参数使用 Team-relative 编号，而普通不带 Team 参数的 point-to-point RMA 通常使用 world PE 编号。

需要跨编号空间时，应使用 `nvshmem_team_translate_pe(src_team, src_pe, dest_team)`。它把 `src_team` 中的相对编号翻译成同一成员在 `dest_team` 中的编号；若该 PE 不同时属于两个 Team，则返回 -1。把 `NVSHMEM_TEAM_WORLD` 作为目标 Team，可以得到对应的全局 PE 编号。

例如 even Team 中的相对 PE 2 要执行普通 `nvshmem_putmem(..., pe)`，目标 world 编号不是 2，而应翻译为 4：

```cpp
// 教学片段：未编译、未运行。
int team_target = 2;
int world_target = nvshmem_team_translate_pe(
    even_team, team_target, NVSHMEM_TEAM_WORLD);
if (world_target >= 0) {
    nvshmem_putmem(remote_dest, local_source, bytes, world_target);
}
```

编号混淆通常不会被类型系统发现，因为两种编号都是 `int`。最稳妥的代码风格是用 `world_pe`、`team_pe`、`root_in_team` 等名字表达命名空间，并把转换集中在边界处。

## 三、Team split 本身就是 parent Team 上的 collective

`nvshmem_team_split_strided(parent, start, stride, size, ...)` 用父 Team 中的等差编号集合创建子 Team。父 Team 的所有 PE 都必须参与，并为 `start/stride/size` 提供相同值。成员得到有效 handle，非成员得到 `NVSHMEM_TEAM_INVALID`。调用者不能只让预期进入子 Team 的偶数 PE 调用 split，因为在创建完成前，子 Team 还不存在，匹配域仍是 parent Team。

设 parent size 为 $N$，新 Team 的第 $i$ 个成员在 parent 中的编号为：

$$
p_i=\text{start}+i\cdot\text{stride},\qquad 0\le i<\text{size},
$$

且每个 $p_i$ 都必须落在 $[0,N-1]$。新 Team 保留成员在 parent 中的相对顺序。例如 `start=0,stride=2,size=3` 从六个 world PEs 中得到成员 $[0,2,4]$，其新编号依次为 $[0,1,2]$。

`nvshmem_team_split_2d` 把 parent Team 映射到二维坐标并产生行/列 Team，适合网格、矩阵分块和多维 stencil。它方便表达拓扑关系，但不意味着物理网络一定按该二维结构布线；逻辑 Team 与 GPU/NIC topology 仍是两个层次。

Team 创建事件也依靠 parent Team 上的调用顺序匹配。若 PE 0 先创建 row team 再创建 column team，而 PE 1 以相反顺序调用，即使两次参数各自合法，也会破坏匹配。创建成功后，parent 和 child 可立即用于后续操作，无需额外同步。

## 四、Team 生命周期必须晚于所有在途使用

应用创建的 Team 在 `nvshmem_team_destroy(team)` 返回后失效。destroy 是该 Team 上的 collective，必须由其成员匹配参与；预定义 Team 不能销毁。调用 destroy 前，应用必须确保没有 host thread、CUDA stream 或 device kernel 仍可能使用该 handle，也不能让某个 stream 中尚未到达队首的 collective 在 destroy 后才开始执行。

这与内存释放的 ownership 问题类似：API 的 host 调用已经返回，不代表所有异步执行域都不再引用资源。若 Team 被多个 CUDA streams 使用，应先用 event/stream synchronization 建立明确的结束顺序，再由 Team 成员销毁。把 handle 变量设为 `NVSHMEM_TEAM_INVALID` 只能帮助本地防止误用，不能替其他 PE 完成 collective destroy。

预定义 Team 的生命周期覆盖整个 NVSHMEM 阶段，应用不需要也不能销毁。自定义 Team 则应有清晰的 owner：谁负责创建、哪些阶段可以使用、在哪个 collective phase 统一销毁。没有这份生命周期设计，动态频繁创建 Team 很容易引入资源泄漏或跨 stream use-after-destroy。

## 五、collective 通过 Team 和程序顺序匹配实例

Team-based collective 要求 Team 中所有 PE 参与，并在所有成员上以相同程序顺序发生。匹配并不依靠用户提供显式 operation ID，而是依靠“这是该 Team 上的第几个 collective”。因此，下列控制流会产生不匹配：

```cpp
// 错误模型：同一 Team 上的 collective 顺序不一致。
if (world_pe == 0) {
    nvshmem_int_broadcast(team, dest, source, n, root);
    nvshmem_int_sum_reduce(team, dest2, source2, n);
} else {
    nvshmem_int_sum_reduce(team, dest2, source2, n);
    nvshmem_int_broadcast(team, dest, source, n, root);
}
```

即使所有 PE 最终都各调用一次 broadcast 和 reduce，它们也没有在相同次序上匹配。结果可能是永久等待或未定义行为。正确条件不是“调用次数相同”，而是 Team、collective 类型、参数约束和程序顺序共同一致。

不同 PE 可以在进入和退出 collective 时存在时间差。某个 PE 的 team-based collective 返回成功，只说明该 PE 达到该接口规定的本地完成边界，不表示其他所有 PE 已在同一时刻返回。因此，collective 返回值通常不应被当作全局事务提交状态；官方文档也提示，不同 PE 的返回状态不必完全相同，team creation 等特定操作另有一致返回要求。

## 六、同一 Team 每个 PE 同时只能贡献一个 collective 实例

对一个 Team，单个 PE 同一时刻不能让多个 host threads、CUDA streams、warps 或 blocks 并发执行多个 collective 实例。普通 thread-scoped device collective 要由每个 PE 上恰好一个 CUDA thread 调用；warp-scoped 版本由恰好一个 warp 共同调用；block-scoped 版本由恰好一个 block 共同调用。warp/block 内所有 threads 必须传入一致参数。

“恰好一个实例”与“恰好 thread 0”不同。程序可以选择任意满足控制条件的 thread、warp 或 block，但每个 PE 上只能有一个与其他 PE 匹配的实例。若两个 blocks 都在 `NVSHMEM_TEAM_WORLD` 上调用同一个 barrier，即使使用 collective launch，也产生两个并发实例，违反 Team 使用规则。

不同 Team 上的 collective 可以并发，因为它们拥有不同匹配域与资源上下文。但“API 允许”不等于 GPU 一定有前进性。如果 Team A 的 kernel 持有全部驻留资源并等待 Team B，而 Team B 的工作尚未调度，仍可能死锁。B03 的等待图方法仍然适用。

## 七、host、on-stream 与 device collective 的阻塞对象不同

标准 Team collective 被定义为 blocking routine，但“谁被阻塞”取决于发起域。Host collective 由 CPU thread 调用，调用者等待本地 collective 完成；on-stream API 的 host 调用先把工作排入 CUDA stream 并返回，当操作到达队首后，它会阻塞该 stream 的后续任务直到 collective 完成；device collective 则让调用的 CUDA thread/group 等待，并同时占用相应 GPU 执行资源。

两个 on-stream collective 位于同一 stream 时受 FIFO 顺序约束；位于不同 streams 时可能并发。若它们使用同一 Team，应用必须用 CUDA event 或其他依赖把它们串行化，不能指望 Team 自动替 stream 排序。Host 上的 symmetric allocation 也隐式使用 world team，所以它不能与另一 stream 上的 world-team barrier 并发。

device kernel 中调用 synchronization 或 collective API 时，必须通过 `nvshmemx_collective_launch` 启动。该接口在 world PEs 上集合调用，并使用 cooperative launch 检查 grid 是否能够满足所需驻留条件。它解决的是安全启动与执行资源前提，不解决 Team 实例数量、参数匹配或跨 stream 循环依赖。

## 八、`sync` 与 `barrier` 都会合，但完成语义不同

`nvshmem_sync(team)` 和 `nvshmem_team_sync(team)` 让 Team 成员在一个同步点会合，并完成/可见化此前的 memory stores；它们不负责完成此前通过 NVSHMEM RMA/AMO 发起的 remote memory updates。若程序希望先完成远端 PUT，再只用 sync 会合，应在发起端先执行与发起域匹配的 `quiet`，否则不能由 sync 单独推出远端更新已完成。

`nvshmem_barrier(team)` 除了让所有 Team 成员会合，还完成这些成员此前发起的 stores 和 remote memory updates，包括相应 RMA 与 AMO。world 版本 `nvshmem_barrier_all()` 隐式作用于全部 PEs。二者不能仅按“快/慢”记忆；正确选择取决于进入同步点前是否存在需要 collective 一并完成的远端更新。

| 操作 | 参与范围 | 会合 | 完成此前 remote RMA/AMO | 典型用途 |
| --- | --- | --- | --- | --- |
| `team_sync/sync` | 指定 Team | 是 | 否 | 已经另行 quiet 后的阶段会合，或只协调 stores |
| `sync_all` | world | 是 | 否 | 全作业轻量阶段会合 |
| `barrier(team)` | 指定 Team | 是 | 是 | Team 内通信阶段排空并会合 |
| `barrier_all` | world | 是 | 是 | 全作业 remote updates 排空并会合 |

完成范围仍受发起域限制。CPU 上执行的 barrier 不应未经契约证明就用于完成 GPU device 发起的通信；官方文档要求从 CPU 确保 GPU-side 操作完成时，在 GPU 侧执行 quiet 并等待 kernel，或使用 stream-based quiet。B02 已详细解释这类域边界。

## 九、broadcast：root 使用 Team-relative 编号

Broadcast 把 root PE 的 source 复制到 Team 所有成员的 dest，包括 root 自己的 dest。所有成员必须传入匹配的 Team、source/dest symmetric address、`nelems` 和 `PE_root`；root 是 Team-relative 的零基编号。调用前，每个 PE 的 dest 必须已经可以接收数据。

设 Team 大小为 $P$，root 为 $r$，每个 source/dest 含 $n$ 个元素，则结果为：

$$
\forall p\in[0,P),\ \forall e\in[0,n):\quad
\operatorname{dest}_p[e]=\operatorname{source}_r[e].
$$

如果 even Team 成员是 world $[0,2,4]$，传 `PE_root=1` 表示 world PE 2，而不是 world PE 1。world PE 1 不在 Team 中。调用返回时，本地 dest 已更新，source 可复用；这仍是当前 PE 的返回条件，不意味着所有成员同时离开函数。

Broadcast 不是任意控制流分支的替代品。root 必须先完成 source 的本地生产并与 collective 发起域建立顺序；其他 PE 不能在 dest 尚被前一阶段使用时进入 collective。若 source 与 dest 是否允许重叠影响设计，必须查该具体 API/算法的契约，不能把 reduction 的原地规则直接套用到 broadcast。

## 十、fcollect：按 Team-relative PE 顺序拼接等长贡献

`fcollect` 让每个 PE 贡献相同数量 $n$ 的元素，并把所有贡献按 Team-relative PE 编号顺序拼接到每个成员的 dest。若 Team 大小为 $P$，dest 至少需要 $P\times n$ 个元素，且所有 PE 的 `nelems` 必须相同。

$$
\operatorname{dest}_p[r\cdot n+e]
=\operatorname{source}_r[e],
\quad 0\le p,r<P,\ 0\le e<n.
$$

结果中的第 $r$ 个 block 来自 Team PE $r$，不是 world PE $r$。对于成员 $[0,2,4]$ 的 even Team，三个 block 分别来自 world PE 0、2、4。所有成员最后得到相同的拼接结果。

```text
Team PE 0 source: [A0 A1]
Team PE 1 source: [B0 B1]
Team PE 2 source: [C0 C1]

每个 PE 的 dest:
[A0 A1 | B0 B1 | C0 C1]
```

`fcollect` 中的“f”对应固定、相同的贡献长度。它适合 all-gather-like 的等长收集；如果实际每个 PE 长度不同，就需要先交换长度、计算 displacement 并选择支持相应语义的方案，不能向 fcollect 传不同 `nelems`。

## 十一、all-to-all：source 按目标分块，dest 按来源分块

All-to-all 让每个 PE 向 Team 中每个 PE 发送 $n$ 个元素。每个 PE 的 source 和 dest 都需要容纳 $P\times n$ 个元素。Source 的第 $j$ 个 block 表示“发给 Team PE $j$”，dest 的第 $i$ 个 block 表示“从 Team PE $i$ 收到”。

$$
\operatorname{dest}_j[i\cdot n+e]
=\operatorname{source}_i[j\cdot n+e],
\quad 0\le i,j<P,\ 0\le e<n.
$$

这个公式是理解 all-to-all 的核心。固定 source PE $i$，它的 block $j$ 发往目标 $j$；固定目标 PE $j$，它的 dest 按 source rank $i$ 的顺序收集各个 block。若把 source 排列成矩阵，all-to-all 相当于在 PE 维度重新分发矩阵块，但不能简单把实现写成一定执行某种矩阵转置算法。

```text
P=3, n=1
PE 0 source: [x00 x01 x02]
PE 1 source: [x10 x11 x12]
PE 2 source: [x20 x21 x22]

PE 0 dest:   [x00 x10 x20]
PE 1 dest:   [x01 x11 x21]
PE 2 dest:   [x02 x12 x22]
```

常见错误是只为 dest 分配 $n$ 个元素，或把 `nelems` 误认为 source 总长度 $P n$。API 中 `nelems` 表示“每个对端的元素数”，所以每个 source/dest 总容量都是 $P\times nelems$。

## 十二、reduction：逐元素组合所有 PE 的贡献

Reduction 对所有 PE 的对应 source 元素应用 associative binary operation，例如 sum、product、min、max 或受支持整数类型上的 bitwise operation，并把结果写入所有参与 PE 的 dest。设运算为 $\oplus$，则 all-reduction 的元素关系为：

$$
\forall p,e:\quad
\operatorname{dest}_p[e]
=\operatorname{source}_0[e]\oplus\operatorname{source}_1[e]
\oplus\cdots\oplus\operatorname{source}_{P-1}[e].
$$

所有 PE 必须使用相同的 `nreduce`，source/dest 至少容纳该数量的元素。当前文档要求 source 与 dest 要么是完全相同的 symmetric address，要么是两个完全不重叠的 symmetric buffers；部分重叠不合法。同时文档提示，特定算法未必直接支持 in-place reduction，所以不能只看到“相同地址允许”就忽略目标版本和算法限制。

“associative”在数学模型和浮点机器运算中要区分。Floating-point addition 由于舍入不满足严格结合律；不同 reduction tree 或并行顺序可能产生末位差异。NVSHMEM collective 不应被默认视为跨算法、跨规模 bitwise deterministic，数值验收需设置合理容差或使用明确的可复现算法。

## 十三、collective 缓冲区必须有明确容量与访问窗口

标准 Team 数据 collective 的 source/dest 是 symmetric address。所有 PE 应从匹配的对称对象计算参数，并保证 dest 在进入 collective 前不再被上一阶段使用。collective 与任何其他方式并发访问同一 symmetric memory、且至少一方写入时，可能构成未定义冲突；不能让一个 stream reduction dest 的同时，另一个 kernel 普通读取它。

| Collective | 每个 PE source 容量 | 每个 PE dest 容量 | 结果分布 |
| --- | ---: | ---: | --- |
| broadcast | root 至少 $n$ | 至少 $n$ | 所有 PE 得到 root 数据 |
| fcollect | 至少 $n$ | 至少 $P n$ | 所有 PE 得到按 Team rank 拼接结果 |
| all-to-all | 至少 $P n$ | 至少 $P n$ | 每个 PE 得到来自所有 source 的目标块 |
| reduction | 至少 $n$ | 至少 $n$ | 所有 PE 得到逐元素 reduction 结果 |

表中是元素容量，不是字节数；`*mem` 形式的 `nelems` 才按字节解释。类型化接口按元素类型计数。设计时应使用 checked multiplication 验证 $P\times n\times sizeof(T)$ 不溢出 `size_t`，并在 symmetric allocation 之前确定一致大小。

## 十四、一个偶数 PE Team 的 broadcast 生命周期

下面的教学程序从 world 中创建偶数 PE Team，让该 Team 的相对 PE 0 广播一个整数。它用于展示 split、invalid handle、Team-relative root 和 destroy 的关系，省略工程级错误处理。

```cpp
// 教学示例：未编译、未运行。
nvshmem_init();
int node_pe = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
cudaSetDevice(node_pe);

int npes = nvshmem_n_pes();
int even_count = (npes + 1) / 2;

int *source = static_cast<int *>(nvshmem_malloc(sizeof(int)));
int *dest   = static_cast<int *>(nvshmem_malloc(sizeof(int)));

nvshmem_team_t even_team = NVSHMEM_TEAM_INVALID;
int rc = nvshmem_team_split_strided(
    NVSHMEM_TEAM_WORLD, 0, 2, even_count,
    nullptr, 0, &even_team);

if (rc == 0 && even_team != NVSHMEM_TEAM_INVALID) {
    int team_pe = nvshmem_team_my_pe(even_team);
    if (team_pe == 0) {
        int value = 42;
        cudaMemcpy(source, &value, sizeof(int), cudaMemcpyHostToDevice);
    }

    nvshmem_int_broadcast(even_team, dest, source, 1, 0);

    int result = -1;
    cudaMemcpy(&result, dest, sizeof(int), cudaMemcpyDeviceToHost);
    nvshmem_team_destroy(even_team);
}

nvshmem_free(dest);
nvshmem_free(source);
nvshmem_finalize();
```

所有 world PEs 都调用 `team_split_strided`，但只有偶数成员得到有效 Team 并进入 broadcast；broadcast 的 root 0 是 even Team PE 0。所有成员按同一顺序先 broadcast、再 destroy。非成员跳过这两次子 Team collective，随后仍与成员共同进入 world-scoped symmetric free；成员必须先完成子 Team 工作才能到达 free，因而不会在 broadcast 使用期间释放缓冲区。

示例假设同步 `cudaMemcpy` 在 root 调用 broadcast 前完成 source 初始化。若改成 `cudaMemcpyAsync`，必须让 collective 所在 stream 或 host 路径等待初始化完成；不能仅因为 broadcast 会等待所有 PEs，就推断它自动等待 root 的任意其他 CUDA stream。

## 十五、算法选择属于性能实现，不属于基础语义

Broadcast 可以用线性发送、树形扩散或硬件 multicast；reduction 可以用树、ring、分层归约或 NVLink SHARP；all-to-all 可以批处理、分阶段或按拓扑调度。应用首先应按 API 契约给出正确 Team、布局和同步，之后才能根据实际系统选择算法。不能从“collective 很高级”推断一定比手写点对点快，也不能从小规模实验推断跨节点扩展。

Team 设计本身也影响性能。world collective 会让所有 PEs 参与，即使算法只有节点内数据依赖；node team 能缩小参与范围，分层 Team 可以先节点内聚合再跨节点通信。不过，多建 Team 会占用 runtime 资源，动态创建/销毁也有成本。应将与算法阶段稳定对应的 Team 复用，而不是每个小消息临时 split。

性能实验至少要记录 Team 大小和成员分布、collective 类型、datatype、元素数、in-place/out-of-place、host/on-stream/device 作用域、warp/block 粒度、拓扑、transport、预热与同步计时边界。只记录 kernel 时间而忽略 stream collective 的等待，或在每轮加入 world barrier，都会改变被测协议。

Tile-based collective、NVLS 和用户可选算法是 3.7.2 文档中的扩展能力，但需要额外的 tensor layout、硬件与算法约束。本基础课不把它们与标准 contiguous-buffer collective 混写；后续若使用，应独立核对 tile completion、算法 flags、设备支持与并发 Team 要求。

## 十六、常见错误及其定位方式

1. **把 Team root 当作 world root。** 先列成员表，再把 `PE_root` 从 Team-relative 编号翻译到 world。
2. **只有子 Team 成员调用 split。** Split 的参与域是 parent Team；非成员也必须调用并接收 invalid handle。
3. **不同 PE 以不同顺序调用 collective。** 为每个 Team 单独列出 collective 序列号，检查类型和参数匹配。
4. **两个 streams 并发使用同一 Team。** 用 CUDA event 明确串行化，或使用独立 Team 并另行证明前进性。
5. **一个 grid 中多个 blocks 在同一 Team 上调用 collective。** 每个 PE 同时只能贡献一个匹配实例；collective launch 不会修复实例重复。
6. **把 sync 当成会完成 RMA。** 若此前存在 PUT/AMO，使用 quiet 后 sync，或选择具有相应完成语义的 barrier。
7. **低估 fcollect/all-to-all 的 dest 容量。** 根据 Team size 计算 $P n$，并区分每对端元素数与总元素数。
8. **让 collective dest 与其他 kernel 并发访问。** 先建立 stream/event 顺序，避免一方写入时的冲突。
9. **销毁仍在 stream 中使用的 Team。** 先等待所有引用该 handle 的工作完成，再由成员 collective destroy。
10. **把 reduction 当作 bitwise deterministic。** 对浮点结果考虑运算顺序与容差，不假设固定 tree。

## 十七、掌握标准与后续路线

完成本课后，读者应能从 world 成员表构造 strided Team，写出每个成员的 Team-relative 编号，并用 `nvshmem_team_translate_pe` 判断 Team 编号如何映射回 world。给定多段控制流，读者应能按每个 Team 独立列出 collective 调用序列，发现顺序不匹配、同 Team 并发和重复 device instance。

对于数据 collective，读者应能写出 broadcast、fcollect、all-to-all 和 reduction 的下标公式，计算 source/dest 最小容量，并说明函数返回、本地 dest 可用与所有 PE 同时返回并非同一事实。还应能准确区分 sync 与 barrier 是否完成先前 remote updates，以及 host、on-stream、device 三种发起域中真正被阻塞的执行者。

B00—B06 至此覆盖 NVSHMEM 的基础语义层：PE/PGAS、远端寻址、完成与可见性、CUDA 前进性、buffer ownership、atomic coordination 和 Team collectives。下一篇 C00“运行时初始化与控制平面”尚未创建，因此本文不建立失效链接；它将回答第一个通信操作前，runtime 如何从 bootstrap 逐步准备 GPU、symmetric heap、transport、peer metadata 和 device state。

## 主要来源

- NVIDIA, [Team Management](https://docs.nvidia.com/nvshmem/api/latest/gen/api/teams.html)：Team handle、编号查询/翻译、split、并发、调用顺序与 destroy 生命周期。
- NVIDIA, [Collective Communication](https://docs.nvidia.com/nvshmem/api/latest/gen/api/collectives.html)：collective matching、sync/barrier、broadcast、fcollect、all-to-all、reduction、缓冲区和返回边界。
- NVIDIA, [NVSHMEM and the CUDA Model](https://docs.nvidia.com/nvshmem/api/latest/cuda-interactions.html)：on-stream collective、同 Team 并发限制、循环依赖与前进性。
- NVIDIA, [Kernel Launch Routines](https://docs.nvidia.com/nvshmem/api/latest/api/launch.html)：device synchronization/collective 的 collective launch 与 grid 驻留约束。
- NVIDIA, [Memory Ordering](https://docs.nvidia.com/nvshmem/api/latest/gen/api/ordering.html)：host/device 完成域与 quiet 的非集合语义。
- NVIDIA, [NVSHMEM 3.7.2 Release Notes](https://docs.nvidia.com/nvshmem/release-notes-install-guide/release-notes/release-3720.html)：本文核对时的发布周期；API Guide `latest` 仍是滚动文档。

