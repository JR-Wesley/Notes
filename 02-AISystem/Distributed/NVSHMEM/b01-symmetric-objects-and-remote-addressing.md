# B01 NVSHMEM 对称对象与远端寻址

## 本课要解决的核心问题

[A04：从 rping 源码追踪一轮完整 RDMA 通信](a04-rping-full-rdma-flow.md)中，客户端必须显式发送 `{addr, rkey, size}`，服务端才能构造 RDMA READ 或 WRITE。NVSHMEM 的接口表面上只需要一个指针、元素数量和目标 PE，例如 `nvshmem_float_put(dest, source, count, pe)`。减少的参数并没有消失：远端实际地址、传输权限、注册信息和 transport 路径由 NVSHMEM runtime 根据对称对象与 PE 进行解析。应用仍然必须保证传入的是正确对象、正确对象内位置、正确目标 PE 和合法长度。

本课围绕 `<symmetric address, target PE>` 解释这种寻址模型。重点不是记住 `nvshmem_malloc()` 的函数签名，而是弄清“对称”究竟对称什么、为什么不同 PE 不需要相同的指针数值、对象内偏移怎样定位远端对应成员，以及 UVA、`nvshmem_ptr()`、普通 buffer registration 和 symmetric registration 各自解决什么问题。寻址正确只证明 runtime 能确定目标位置，不证明数据已经写入、消费者已经得到通知或缓冲区已经可以复用；这些完成与所有权问题由 [B02：完成、排序与可见性](b02-completeness-ordering-and-visibility.md)继续处理。

## 资料版本与论述边界

本文在 2026-09-12 核对 NVIDIA 官方 NVSHMEM API Guide 的 `latest` 页面，并用 [NVSHMEM 3.7.2 Release Notes](https://docs.nvidia.com/nvshmem/release-notes-install-guide/release-notes/release-3720.html)确认当前发布周期。`latest` 是会随发布更新的滚动 URL，不是不可变归档，因此本文将 API 结论标记为“3.7.2 发布周期所见契约”，不声称已经绑定 NVSHMEM 源码 commit。后续 D 模块分析实现时必须另外选择固定源码版本。

本文的代码用于说明地址关系，未编译、未运行，也没有 GPU、NVLink、InfiniBand 或 IBGDA 实测结果。诸如“runtime 可概念化为按对象内偏移寻找远端对应位置”的表达是教学模型；官方 API 保证的是 symmetric address 与目标 PE 的组合能够标识远端对应对象，不公开承诺内部一定采用某个固定地址表、算术公式、MR 或 key 布局。

## 从每个 PE 的本地对象构成 PGAS

NVSHMEM 作业由多个 processing element（PE）组成，每个 PE 通常是独立操作系统进程，并在选定 GPU 上拥有自己的本地内存。每个 PE 都执行同一份程序，但可以根据 `nvshmem_my_pe()` 返回的 world PE 编号承担不同工作。普通 `cudaMalloc()` 创建的分配默认只属于调用 PE；其他 PE 不能仅凭获知这个指针数值，就把它当作 NVSHMEM 远端对象。

当所有 PE 以兼容顺序调用 `nvshmem_malloc(size)` 时，每个 PE 的 symmetric heap 中都得到一个与该次分配对应的本地内存块。所有这些分区合起来形成 partitioned global address space（PGAS）。所谓 symmetric object，不是“一块物理内存被所有 GPU 同时拥有”，而是“每个 PE 都有一份在逻辑布局、分配次序和大小上相对应的对象”。对象内容可以不同，物理 GPU 可以不同，进程虚拟地址也不要求相等。

假设两个 PE 都执行：

```cpp
// 教学片段，未编译、未运行。
constexpr std::size_t N = 1024;
float *x = static_cast<float *>(nvshmem_malloc(N * sizeof(float)));
```

PE 0 上的 `x` 指向 PE 0 本地 GPU 内存，PE 1 上的 `x` 指向 PE 1 本地 GPU 内存。PE 0 可用普通 CUDA load/store 访问自己的 `x[i]`；若它调用 `nvshmem_float_p(&x[i], value, 1)`，同一个本地 symmetric address `&x[i]` 与目标 PE 1 组合后，表示“PE 1 上本次对应分配的第 `i` 个元素”。这里没有把 PE 1 的原始指针传给 PE 0，也没有要求两边 `x` 的数值相等。

## 对称的是分配关系和对象内位置，不是裸指针值

可以用一个只用于推理的模型表达远端定位。设第 $k$ 次对称分配在 PE $p$ 上的本地基址为 $B_{p,k}$，某个合法地址 `ptr` 位于该对象内，其字节偏移为

$$
\Delta = \operatorname{byte}(ptr) - \operatorname{byte}(B_{p,k}).
$$

当 PE $p$ 以 `ptr` 和目标 PE $q$ 发起 NVSHMEM 操作时，语义目标是 PE $q$ 上第 $k$ 个对应对象内相同偏移 $\Delta$ 的位置。可以把远端实际位置概念化为 $B_{q,k}+\Delta$，但 $B_{q,k}$ 是 runtime 管理的目标映射，不是应用通过交换裸指针得到的 API 参数。官方文档明确说明 symmetric address 只在返回它的本地 PE 上有效，不能把它发送给其他 PE 后当作对方的 symmetric address 使用。

这个模型解释了数组下标和结构成员为何自然工作。若各 PE 都分配相同布局的 `Tile` 数组，那么 `&tiles[s].payload[i]` 相对本地分配基址的偏移，在所有 PE 上都对应同一个 slot、同一个成员和同一个元素：

```cpp
// 教学片段，未编译、未运行。
struct Tile {
    std::uint64_t ready;
    float payload[256];
};

Tile *tiles = static_cast<Tile *>(nvshmem_malloc(slots * sizeof(Tile)));
float *remote_element = &tiles[slot].payload[index];
nvshmem_float_p(remote_element, value, peer);
```

成立的前提是 `slot < slots`、`index < 256`，所有 PE 对这次分配使用相同 `slots` 和 ABI 兼容的 `Tile` 布局，并且对象仍处于有效生命周期。对称性不会修复越界指针：如果本地地址已经超出该 symmetric object，runtime 不会因为目标 PE 上恰好还有可映射内存就把操作变成合法访问。类型对齐、元素数量乘法溢出和起始地址加长度后的范围都仍需由应用检查。

## 为什么不能交换另一个 PE 的 symmetric pointer

下面的思路沿用了手写 RDMA 控制面，却不符合 NVSHMEM symmetric addressing：PE 1 把本地 `x` 转成整数发送给 PE 0，PE 0 再把这个数值强转为目标地址。即使两个进程打印出的地址碰巧相同，这个数值也不是 PE 0 上可合法使用的 symmetric address；即使数值不同，把 PE 1 的进程地址放进 PE 0 的指针变量也不会自动建立映射。

```cpp
// 错误模型：不要交换并复用其他 PE 的裸指针值。
std::uintptr_t peer_reported_x = exchange_pointer_value_somehow();
float *wrong = reinterpret_cast<float *>(peer_reported_x);
nvshmem_float_p(&wrong[i], value, peer);  // 不满足 symmetric-address 契约
```

正确做法是每个 PE 保存自己由同一次 collective allocation 返回的本地基址，并在发起操作的 PE 上计算对象内地址：

```cpp
float *local_symmetric_addr = &x[i];
nvshmem_float_p(local_symmetric_addr, value, peer);
```

与 A04 的 rping 相比，应用层不再交换远端 virtual address 和 `rkey`；runtime 在初始化和内存管理期间建立远端映射与访问能力。仍然保留的参数是对象内位置、目标 PE、类型或字节长度。换言之，NVSHMEM 消除了应用显式管理地址/key 控制包的责任，却没有消除地址范围和协议阶段的正确性责任。

## 分配和释放为什么必须是集合式操作

`nvshmem_malloc()`、`nvshmem_align()` 和 `nvshmem_free()` 都是需要所有 PE 参与的 collective operation。非零大小的 `nvshmem_malloc()` 与 `nvshmem_align()` 在退出前包含语义等价于 `nvshmem_barrier_all` 的过程，使调用返回时其他 PE 的对应内存已经可用于远端访问；`nvshmem_free()` 则在入口包含相应 barrier，避免某个 PE 在其他 PE 仍可能使用对象时率先释放。`nvshmem_calloc()` 同样集合分配，并把本地对象初始化为全零位。

所有 PE 不仅要传入相同大小，还要以相同顺序执行 symmetric heap 操作。下面的分支会破坏第 $k$ 次分配的对应关系，并可能使某些 PE 永久等待 collective，或使后续 NVSHMEM 行为未定义：

```cpp
// 错误：只有部分 PE 参与本次对称分配。
if (nvshmem_my_pe() == 0) {
    buffer = nvshmem_malloc(bytes);
}
```

允许不同 PE 根据编号执行不同计算，并不意味着可以任意分化对称内存生命周期。若某个可选对象只在 PE 0 上有业务数据，所有 PE 仍需共同完成对应分配；未使用该对象的 PE 可以不填充内容，但不能跳过这次 collective。释放时，每个 PE 应传入自己那份对应分配返回的 local symmetric address，并保持释放顺序兼容。

零大小是容易忽视的边界。当前 Memory Management 契约规定 `size == 0` 时 `nvshmem_malloc()` 不执行分配、返回 null，且不执行 barrier。因此不能把一次零大小调用当作同步点；如果不同 PE 计算出不同的 size，行为也不是“部分 PE 分配、部分 PE 跳过”，而是违反相同参数要求。

## 本地 symmetric address 在 RMA 参数中的角色

NVSHMEM RMA 的远端 operand 必须由 symmetric address 表达。对于 PUT，`dest` 是发起 PE 本地持有的 symmetric address，`pe` 指定要访问哪一份远端对应对象；`source` 是本地读取源。对于 GET，`source` 是本地持有的 symmetric address，`pe` 指定要从哪一份远端对应对象读取；`dest` 是本地写入目标。参数名中的 `dest` 或 `source` 描述数据流方向，不表示该指针数值来自目标 PE。

```text
PUT:  local source  ─────>  <local symmetric dest address, target PE>
GET:  <local symmetric source address, target PE>  ─────>  local dest
```

当前 RMA API 允许某些本地 operand 来自 symmetric heap，或来自通过 `nvshmemx_buffer_register()` 注册的 host/device buffer；文档还列出 peer-to-peer transport 下 device shared memory 等特定例外。无论本地 operand 允许哪些来源，远端 operand 的定位规则仍是 symmetric object 加目标 PE。应用不应从“某个 topology 上 PUT 的本地 source 可以是 `cudaMalloc` 内存”推出“远端 dest 也可以是任意 `cudaMalloc` 指针”。跨节点 InfiniBand 尤其会暴露这种混淆：官方 FAQ 将非 symmetric remote address 列为 Remote Protection Error 的常见原因。

RMA 调用还需要合法的目标 PE 编号和完整范围。若 `dest = &x[N-2]`、传输 `4` 个 float，即使起点本身位于对象内，尾部仍越过分配边界。对 strided `iput/iget`，还要将元素步长和最后一个元素位置纳入范围证明。symmetric address 解决“对应哪一份对象”，不会替应用保存数组长度或执行 C++ 边界检查。

## UVA 与 symmetric addressing 是两个维度

CUDA Unified Virtual Addressing（UVA）让一个操作系统进程中的 CPU 内存和各 GPU 内存位于统一虚拟地址空间，使 CUDA 可以根据指针属性判断内存位置，并在支持的 peer access 场景中使用统一形式的指针。UVA 讨论的是一个进程及其 CUDA contexts 如何解释虚拟地址；NVSHMEM symmetric addressing 讨论的是多个 PE 的逻辑对应对象如何通过 `<symmetric address, PE>` 被定位。两者可能同时被 runtime 利用，但不能相互替代。

NVSHMEM 的不同 PE 通常是不同进程，各进程有独立虚拟地址空间。地址 `0x...` 在 PE 0 与 PE 1 中数值相同，不足以证明它们指向对应对象；数值不同也不妨碍它们在 NVSHMEM 中对称。CUDA peer access 可使同一节点上的 GPU 直接 load/store 某个 peer allocation，但它取决于设备拓扑、peer capability 和映射配置，不能由 UVA 本身推出，更不能扩展为跨节点裸指针可解引用。

UVA 也不同于 Unified Memory。前者统一虚拟地址的表示与属性查询，后者涉及 managed allocation、迁移或系统级一致性机制。当前 `nvshmemx_buffer_register()` 明确不支持 CUDA managed memory；因此“一个指针能被 CPU 和 GPU 表示”并不证明它能作为 NVSHMEM 注册 buffer 或远端 symmetric object。

## `nvshmem_ptr()` 返回的是直接访问指针，不是新的 symmetric address

`nvshmem_ptr(dest, pe)` 的输入 `dest` 仍是调用 PE 上的 local symmetric address。若目标 PE 的对应对象可以通过普通 load/store 直接访问，该函数返回一个指向目标对象的本地可用映射地址；若当前 topology 或 transport 不支持直接访问，则返回 null。它更接近“为这个特定 peer 查询一条直接指针路径”，不是绕过 symmetric heap 的通用地址转换服务。

```cpp
// 教学片段，未编译、未运行。
float *direct = static_cast<float *>(nvshmem_ptr(&x[i], peer));
if (direct != nullptr) {
    *direct = value;                 // 普通 direct store，需遵守 NVSHMEM/CUDA 内存模型
} else {
    nvshmem_float_p(&x[i], value, peer);  // 使用 NVSHMEM RMA 路径
}
```

最关键的类型边界是：`nvshmem_ptr()` 返回的 direct pointer 不能再传给要求 symmetric address 的 NVSHMEM API，否则行为未定义。调用 `nvshmem_float_p(direct, value, peer2)` 是错误的，因为 `direct` 是针对某个 peer 的本地直接映射地址，不再是“当前 PE 上对应对象的位置”。同理，不能把它保存到 wire message 后交给另一 PE 使用。

direct load/store 是否完成、与 NVSHMEM RMA 如何排序、其他 PE 何时看见结果，仍需按内存模型和同步 API 建立关系。非 null 只证明地址可被当前执行环境直接引用，不证明访问具有原子性，也不自动通知远端消费者。`NVSHMEM_TEAM_SHARED` 可用于描述成员之间 symmetric heap 均可经 `nvshmem_ptr` 直接访问的 team，但具体可达集合仍来自运行时拓扑，不能假设等于 world team。

## 普通注册与 symmetric registration 不能混为一谈

当前 Memory Management API 提供两类名称相近但语义不同的扩展。`nvshmemx_buffer_register(addr, length)` 把已有 buffer 注册到远端 transport，并在传入未注册 host memory 时同时处理 CUDA 注册，使它能够作为 host/device RMA 或 AMO 的本地 operand。它没有把该 buffer 变成远端 symmetric object；官方契约明确说，将这种 registered buffer 用作任何 NVSHMEM API 的远端 operand 不受支持。

普通注册还有明确边界：NVSHMEM heap 已由 runtime 注册，不应再次调用该函数；CUDA managed memory 不受支持；一次 RMA 的 buffer 不能跨越多个 registration；释放底层内存前必须先用起始地址执行 `nvshmemx_buffer_unregister()`。这些约束对应 A02 的本地 MR 范围与生命周期，只是具体 transport registration 被 NVSHMEM API 封装。

`nvshmemx_buffer_register_symmetric(user_buffer, size, flags)` 则是另一条路径：所有 PE 集合调用，把各自通过 CUDA VMM 分配且满足 granularity 要求的 user buffer 映射进 NVSHMEM symmetric heap。成功返回的是可用于 NVSHMEM 通信的 symmetric address；后续应使用这个返回值表达远端 operand，而不是默认原始 `user_buffer` 数值已经取得相同语义。各 PE 必须传入相同 size，平台必须支持 CUDA VMM，并通过对应 symmetric unregister API 解除映射。

`nvshmemx_buffer_register_symmetric_at_preferred_address()` 允许请求在 symmetric heap 中使用 preferred address，但官方契约说明该地址不可用时可以改映射到其他位置。因此 preferred 不等于 guaranteed，更不能据此建立“所有 PE 裸地址必须相等”的协议。应用始终应保存 API 实际返回的 symmetric address，并把它与原始 physical allocation 的管理句柄和生命周期区分开。

| 内存来源或接口 | 可作为远端 symmetric operand | 可作为本地 operand | 是否集合操作 | 关键限制 |
| --- | --- | --- | --- | --- |
| `nvshmem_malloc/calloc/align` | 是 | 是 | 是 | 各 PE 参数与调用顺序兼容；使用返回的本地 symmetric address |
| 普通 `cudaMalloc` | 默认否 | 取决于路径；不能笼统保证 | 否 | 不是 symmetric object，peer 可达不等于跨节点可达 |
| `nvshmemx_buffer_register` | 否 | 在文档支持的 RMA/AMO 位置可用 | 否 | 不支持 managed memory、跨 registration 和先 free 后 unregister |
| `nvshmemx_buffer_register_symmetric` | 成功映射后是 | 是 | 是 | 仅 CUDA VMM buffer、相同 size、满足 heap granularity |
| `nvshmem_ptr` 返回值 | 否 | 仅作当前 PE 的 direct load/store pointer | 否 | 可能为 null；不得传回要求 symmetric address 的 API |

## 对称结构中的裸指针字段为什么危险

对象本身位于 symmetric heap，不代表对象内部保存的每个指针值都会自动翻译。若各 PE 在 `Node::payload` 中保存本地 `cudaMalloc()` 或本地 symmetric allocation 的裸指针，远端读取这个字段后得到的只是写入者地址空间中的数值。对另一个 PE 而言，它既不一定可解引用，也不一定能作为本地 symmetric address。

```cpp
// 容易误用：Node 对称，不代表 payload 指针值自动对称。
struct Node {
    float *payload;
    std::uint64_t length;
};
```

可迁移的表示通常是数组索引、对象 ID 或相对于某个已知 symmetric allocation 基址的整数偏移。每个发起 PE 用自己的本地基址加偏移，重新构造 local symmetric address，再与目标 PE 组合。若必须传递直接访问 handle，则需由明确的 CUDA IPC、VMM、transport registration 或库 API 定义其作用域与生命周期，不能把普通 C++ pointer 当作分布式全局指针。

## 五个必须分开检查的条件

对称寻址只解决正确性链条的一部分。一次远端访问至少要分别检查以下条件：

| 条件 | 要回答的问题 | 典型失败 |
| --- | --- | --- |
| 地址有效 | 指针是否位于仍存活的正确 symmetric object 内，范围是否越界 | 传入 private pointer、悬空指针、错误 offset |
| 远端可达 | 指定 PE 与对象是否能通过所选 transport 或 direct mapping 访问 | `nvshmem_ptr()` 返回 null、transport 不支持某路径 |
| 权限/注册 | runtime 是否为该方向准备了远端访问能力，本地非 heap buffer 是否已合法注册 | Remote/Local Protection Error |
| 数据就绪 | 生产者写入本地 source 是否已在发起 RMA 前对正确执行域可见 | GPU/stream 顺序错误，发送旧数据 |
| 完成与消费 | 操作何时完成，目标 PE 通过什么通知消费，何时允许覆盖槽位 | 把调用返回当成远端消费确认 |

这五项不能互相证明。一个地址可以是合法 symmetric address，但 `nvshmem_ptr()` 因 peer 不可直接映射而返回 null；一个 registered local buffer 可以被 transport 读取，却不能当远端 dest；PUT 已经正确寻址并完成，也不表示消费者已经使用完目标槽位。诊断时按这几个维度逐项排除，比笼统归因于“NVSHMEM 地址有问题”更有效。

## 常见错误与定位方法

出现 InfiniBand Remote Protection Error 时，首先检查 NVSHMEM API 的远端 operand 是否来自 symmetric heap 或 symmetric registration 返回值，是否误传了普通 `cudaMalloc` 地址、另一 PE 打印出的裸指针或 `nvshmem_ptr()` 返回值。出现 Local Protection Error 时，检查本地 operand 是否位于 symmetric heap，或是否已通过 `nvshmemx_buffer_register()` 覆盖完整范围。官方 FAQ 将这两类地址来源错误列为常见原因，但具体 status 仍应结合 transport 日志和失败请求分析。

若错误只在数组尾部、某些 stride 或较大消息出现，应重新计算 `[start, start + bytes)` 或最后一个 strided element 是否仍在同一分配内。若程序在分配处挂起，检查所有 PE 是否以相同顺序、相同 size 进入 collective allocation，而不是先怀疑网络带宽。若同节点可运行、跨节点失败，检查是否无意中依赖了 peer direct access 或普通 `cudaMalloc` 本地/远端例外；跨节点路径通常更严格地暴露 symmetric remote operand 与 registration 要求。

还应记录实际使用的 NVSHMEM 版本、CUDA 版本、PE→GPU 绑定、transport 和拓扑。`nvshmem-info -a` 可提供构建和运行配置线索，但配置输出不能替代对象生命周期审计。地址打印只能辅助确认每个 PE 的本地分配和偏移，不能用“两个地址相同”作为对称性证明。

## 诊断题与参考推理

### 问题一：两个 PE 的 `nvshmem_malloc()` 返回值不同，程序一定错误吗

不一定。API 语义要求各 PE 对称参与同一次分配并用本地返回的 symmetric address，不要求应用比较出的虚拟地址数值相同。应检查分配顺序、大小和对象内 offset，而不是比较裸地址。反过来，两个数值相同也不能证明它们来自同一次对应分配。

### 问题二：PE 0 能否把 PE 1 发来的 `x` 指针直接传给 `nvshmem_put`

不能。symmetric address 只在返回它的 PE 上有效。PE 0 应使用自己的 `x` 计算 `&x[i]`，再把目标参数设为 PE 1。若双方需要共享动态位置，应交换 index/offset 等逻辑位置，而不是交换目标进程的裸指针。

### 问题三：`nvshmemx_buffer_register()` 成功后，为什么仍不能把该 buffer 作为远端 PUT 目标

因为普通 registration 只使 buffer 成为合法本地 operand，并未把所有 PE 上的对应 buffer 纳入 symmetric heap。要让 user-managed buffer 成为远端 symmetric operand，需要满足 current API 的 CUDA VMM、collective、same-size 和 granularity 要求，调用 symmetric registration，并使用其返回的 symmetric address。

### 问题四：`nvshmem_ptr(&x[i], peer)` 返回非 null 后，能否把返回值传给另一个 `nvshmem_float_p`

不能。返回值是当前 PE 用于直接 load/store 该 peer 对象的本地映射指针，不是 symmetric address。把它传给要求 symmetric address 的 NVSHMEM routine 属于未定义行为。直接 store 的排序和目标通知也必须另外设计。

### 问题五：PUT 的地址和长度都正确，目标为什么仍可能读到旧数据

寻址正确与数据就绪、操作完成、远端通知是独立条件。源数据可能尚未在发起 RMA 的 stream/device 顺序域中就绪，PUT 可能只完成了发起或本地源读取，目标也可能没有等待 signal/barrier。应继续按 B02 检查调用域、fence/quiet、signal/wait 和消费者协议。

## 掌握标准与后续阅读

完成本课后，应能画出两个 PE 上对应分配而基址可能不同的图，并用“本地 symmetric address 的对象身份与偏移 + target PE”解释远端定位；应能为数组元素和结构成员证明范围合法，识别对称结构中的裸指针陷阱；应能区分 UVA、peer direct pointer、普通 registered local buffer 和 symmetric registered buffer；还应能说明为什么地址有效不等于可达、获权、就绪、完成或消费。

配套的 [实验 B01：两 PE 对称寻址与单元素 PUT](experiments/experiment-b01-symmetric-addressing-and-put.md)提供一个环形写入程序。该实验当前仍是未编译、未运行的初稿，阅读时可先纸面标出每个 PE 的 local symmetric address、目标 PE、实际接收者和 barrier 作用，再在具备环境后记录真实版本与结果。下一步应阅读 B02，将本课的“写到哪里”扩展为“何时写完、谁能观察、何时复用”。

## 主要来源

- [Using NVSHMEM](https://docs.nvidia.com/nvshmem/api/latest/using.html)：PE、PGAS、symmetric allocation、`<symmetric_address, destination_PE>` 及地址只在本地 PE 有效的契约。
- [Memory Management](https://docs.nvidia.com/nvshmem/api/latest/gen/api/memory.html)：collective allocation/free、barrier 位置、普通 registration、symmetric registration 及生命周期限制。
- [Library Setup, Exit, and Query：`nvshmem_ptr`](https://docs.nvidia.com/nvshmem/api/latest/gen/api/setup.html#nvshmem-ptr)：direct pointer 的输入、null 返回与不得作为 symmetric address 回传的限制。
- [Remote Memory Access](https://docs.nvidia.com/nvshmem/api/latest/gen/api/rma.html)：PUT/GET 远端 symmetric operand、本地 registered operand、元素数量和完成语义。
- [Troubleshooting and FAQs](https://docs.nvidia.com/nvshmem/api/latest/faq.html)：非 symmetric 远端地址与未注册本地地址导致 protection error 的官方诊断提示。
- [CUDA Programming Guide：Unified Virtual Address Space](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#unified-virtual-address-space)：UVA 的进程内地址空间含义，用于与 NVSHMEM 跨 PE 对称寻址区分。
