## 8. Memory Consistency Model

### English Original

> In multi-threaded executions, the side-effects of memory operations performed by each thread become visible to other threads in a partial and non-identical order. This means that any two operations may appear to happen in no order, or in different orders, to different threads. The axioms introduced by the memory consistency model specify exactly which contradictions are forbidden between the orders observed by different threads.
>
> In the absence of any constraint, each read operation returns the value committed by some write operation to the same memory location, including the initial write to that memory location. The memory consistency model effectively constrains the set of such candidate writes from which a read operation can return a value.

### 中文翻译

在 multi-threaded execution 中，每个 thread 执行的 memory operation 所产生的 side effect，会以部分且彼此不同的顺序对其他 thread 可见。因此，对不同 thread 而言，任意两个 operation 可能看起来没有先后顺序，或者以不同顺序发生。Memory consistency model 引入的 axiom 精确规定不同 thread 所观察到的顺序之间不允许出现哪些矛盾。

若完全没有 constraint，每个 read operation 都可以返回同一 memory location 上某个 write operation 提交的 value，包括该 location 的 initial write。Memory consistency model 实际上就是约束 read 可以从哪些 candidate write 中取得 value。

### 重点解读

该模型不是为所有 thread 构造一个统一全序，而是定义多种局部关系，并禁止它们形成特定矛盾。理解后续章节时，应持续区分“source-code 顺序”“跨 thread 的因果顺序”与“某次 read 实际观察到哪个 write”这三件事。

### 8.1 Scope and Applicability of the Model

#### English Original

> The constraints specified under this model apply to PTX programs with any PTX ISA version number, running on `sm_70` or later architectures.
>
> The memory consistency model does not apply to texture (including `ld.global.nc`) and surface accesses.

#### 中文翻译

该模型的 constraint 适用于运行在 `sm_70+` architecture 上的 PTX program，与其 PTX ISA version number 无关。

Memory consistency model 不适用于 texture access（包括 `ld.global.nc`）和 surface access。

#### 8.1.1 Limitations on Atomicity at System Scope

##### English Original

> When communicating with the host CPU, certain strong operations with system scope may not be performed atomically on some systems. For more details on atomicity guarantees to host memory, see the CUDA Atomicity Requirements.

##### 中文翻译

与 host CPU 通信时，在某些 system 上，部分 system-scope strong operation 可能无法以 atomic 方式执行。关于 host memory 的 atomicity guarantee，参阅 CUDA Atomicity Requirements。

### 8.2 Memory Operations

#### English Original

> The fundamental storage unit in the PTX memory model is a byte, consisting of 8 bits. Each state space available to a PTX program is a sequence of contiguous bytes in memory. Every byte in a PTX state space has a unique address relative to all threads that have access to the same state space.
>
> Each PTX memory instruction specifies an address operand and a data type. The address operand contains a virtual address that gets converted to a physical address during memory access. The physical address and the size of the data type together define a physical memory location, which is the range of bytes starting from the physical address and extending up to the size of the data type in bytes. Analogously, the address operand together with the size of the data type define the range of virtual memory addresses accessed by the memory operation.
>
> The memory consistency model specification uses the terms “address” or “memory address” to indicate a virtual address, and the term “memory location” to indicate a physical memory location.
>
> Each PTX memory instruction also specifies the operation — either a read, a write or an atomic read-modify-write — to be performed on all the bytes in the corresponding memory location.

#### 中文翻译

PTX memory model 的基本 storage unit 是由 8 bit 构成的 byte。PTX program 可用的每个 state space 都是 memory 中一段连续 byte；相对于所有能访问同一 state space 的 thread，其中每个 byte 都具有唯一 address。

每条 PTX memory instruction 都指定 address operand 和 data type。Address operand 包含 virtual address，在 memory access 时被转换为 physical address。Physical address 与 data-type size 共同定义 physical memory location，也就是从该 physical address 开始、长度等于 data type byte 数的 byte range。类似地，address operand 与 data-type size 共同定义 memory operation 所访问的 virtual-address range。

在本规范中，“address”或“memory address”表示 virtual address，“memory location”表示 physical memory location。

每条 PTX memory instruction 还指定对相应 memory location 中全部 byte 执行的 operation：read、write 或 atomic read-modify-write。

#### 8.2.1 Overlap

##### English Original

> Two memory locations are said to overlap when the starting address of one location is within the range of bytes constituting the other location. Two memory operations are said to overlap when the range of virtual addresses accessed by the two operations intersect. The overlap is said to be complete when both memory locations are identical, and it is said to be partial otherwise.

##### 中文翻译

如果一个 memory location 的 starting address 位于另一个 location 所构成的 byte range 内，则两个 memory location overlap。如果两个 memory operation 访问的 virtual-address range 相交，则两个 operation overlap。两个 memory location 完全相同时称为 complete overlap，否则称为 partial overlap。

#### 8.2.2 Aliases

##### English Original

> Two distinct virtual addresses are said to be aliases if they map to the same memory location.

##### 中文翻译

如果两个不同的 virtual address 映射到同一个 memory location，则二者互为 alias。

#### 8.2.3 Multimem Addresses

##### English Original

> A multimem address is a virtual address which points to multiple distinct memory locations across devices.
>
> Only `multimem.*` operations are valid on multimem addresses. That is, the behavior of accessing a multimem address in any other memory operation is undefined.

##### 中文翻译

Multimem address 是指向多个 device 上不同 memory location 的 virtual address。只有 `multimem.*` operation 可以合法访问 multimem address；使用其他 memory operation 访问它的行为未定义。

#### 8.2.4 Memory Operations on Vector Data Types

##### English Original

> The memory consistency model relates operations executed on memory locations with scalar data types, which have a maximum size and alignment of 64 bits. Memory operations with a vector data type are modelled as a set of equivalent memory operations with a scalar data type, executed in an unspecified order on the elements in the vector.

##### 中文翻译

Memory consistency model 直接描述在 scalar-data-type memory location 上执行的 operation；其最大 size 与 alignment 为 64 bit。使用 vector data type 的 memory operation 被建模为一组等价 scalar memory operation，并以未指定顺序作用于 vector 中的各 element。

#### 8.2.5 Memory Operations on Packed Data Types

##### English Original

> A packed data type consists of two values of the same scalar data type, as described in Packed Data Types. These values are accessed in adjacent memory locations. A memory operation on a packed data type is modelled as a pair of equivalent memory operations on the scalar data type, executed in an unspecified order on each element of the packed data.

##### 中文翻译

Packed data type 由两个相同 scalar data type 的 value 构成，它们位于相邻 memory location。对 packed data type 的一次 memory operation，被建模为两次等价 scalar memory operation，并以未指定顺序分别作用于 packed data 的两个 element。

#### 8.2.6 Initialization

##### English Original

> Each byte in memory is initialized by a hypothetical write W0 executed before starting any thread in the program. If the byte is included in a program variable, and that variable has an initial value, then W0 writes the corresponding initial value for that byte; else W0 is assumed to have written an unknown but constant value to the byte.

##### 中文翻译

Memory 中的每个 byte 都由一个假想 write W0 初始化，W0 在 program 的任何 thread 启动之前执行。如果该 byte 属于一个具有 initial value 的 program variable，W0 写入对应初值；否则，假定 W0 向该 byte 写入一个未知但保持不变的 value。

#### 重点解读

Vector 与 packed access 在模型中会拆成多个 scalar operation，且 element 间执行顺序未指定。因此，不能仅凭“一条 vector instruction”推导整个 vector 具有单一不可分割的 memory-model atomicity。模型中的初始值也统一来自 W0，使 initial state 能和普通 write 一起参与后续关系推理。

### 8.3 State Spaces

#### English Original

> The relations defined in the memory consistency model are independent of state spaces. In particular, causality order closes over all memory operations across all the state spaces. But the side-effect of a memory operation in one state space can be observed directly only by operations that also have access to the same state space. This further constrains the synchronizing effect of a memory operation in addition to scope. For example, the synchronizing effect of the PTX instruction `ld.relaxed.shared.sys` is identical to that of `ld.relaxed.shared.cluster`, since no thread outside the same cluster can execute an operation that accesses the same memory location.

#### 中文翻译

Memory consistency model 定义的 relation 独立于 state space；特别是 causality order 会闭包覆盖所有 state space 中的全部 memory operation。但是，一个 state space 中 memory operation 的 side effect，只能被同样有权访问该 state space 的 operation 直接观察。因此，除 scope 之外，state-space accessibility 还会进一步限制 memory operation 的 synchronization effect。

例如，`ld.relaxed.shared.sys` 与 `ld.relaxed.shared.cluster` 的 synchronization effect 相同，因为同一 cluster 之外的 thread 无法执行访问同一 shared-memory location 的 operation。

### 8.4 Operation Types

#### English Original

> For simplicity, the rest of the document refers to the following operation types, instead of mentioning specific instructions that give rise to them.
>
> Table 20 Operation Types

| Operation Type | Instruction/Operation |
|---|---|
| atomic operation | `atom` or `red` instruction. |
| read operation | All variants of `ld` instruction and `atom` instruction (but not `red` instruction). |
| write operation | All variants of `st` instruction, and atomic operations if they result in a write. |
| memory operation | A read or write operation. |
| volatile operation | An instruction with `.volatile` qualifier. |
| acquire operation | A memory operation with `.acquire` or `.acq_rel` qualifier. |
| release operation | A memory operation with `.release` or `.acq_rel` qualifier. |
| mmio operation | An `ld` or `st` instruction with `.mmio` qualifier. |
| memory fence operation | A `membar`, `fence.sc` or `fence.acq_rel` instruction. |
| proxy fence operation | A `fence.proxy` or a `membar.proxy` instruction. |
| strong operation | A memory fence operation, or a memory operation with a `.relaxed`, `.acquire`, `.release`, `.acq_rel`, `.volatile`, or `.mmio` qualifier. |
| weak operation | An `ld` or `st` instruction with a `.weak` qualifier. |
| synchronizing operation | A barrier instruction, fence operation, release operation or acquire operation. |

#### 中文翻译

为简化表述，后文使用 Table 20 中的 operation type，而不反复列出产生这些 operation 的具体 instruction。表格完整保留英文原文，不另作中文副表。

其中需要特别注意：`atom` 同时属于 atomic、read，并在产生 write 时也属于 write；`red` 属于 atomic，但不属于这里定义的 read。Strong operation 包括 memory fence，或带 `.relaxed`、`.acquire`、`.release`、`.acq_rel`、`.volatile`、`.mmio` qualifier 的 memory operation；`.weak` load/store 则属于 weak operation。

#### 8.4.1 `mmio` Operation

##### English Original

> An mmio operation is a memory operation with `.mmio` qualifier specified. It is usually performed on a memory location which is mapped to the control registers of peer I/O devices. It can also be used for communication between threads but has poor performance relative to non-mmio operations.
>
> The semantic meaning of mmio operations cannot be defined precisely as it is defined by the underlying I/O device. For formal specification of semantics of mmio operation from Memory Consistency Model perspective, it is equivalent to the semantics of a strong operation. But it follows a few implementation-specific properties, if it meets the CUDA atomicity requirements at the specified scope:
>
> - Writes are always performed and are never combined within the scope specified.
> - Reads are always performed, and are not forwarded, prefetched, combined, or allowed to hit any cache within the scope specified.
> - As an exception, in some implementations, the surrounding locations may also be loaded. In such cases the amount of data loaded is implementation specific and varies between 32 and 128 bytes in size.

##### 中文翻译

Mmio operation 是带 `.mmio` qualifier 的 memory operation，通常作用于映射到 peer I/O device control register 的 memory location。它也可用于 thread 间通信，但性能明显低于 non-mmio operation。

Mmio 的精确语义由 underlying I/O device 定义，无法由 PTX 统一规定；从 Memory Consistency Model 的形式化视角看，它等价于 strong operation。如果在指定 scope 上满足 CUDA atomicity requirement，还具有以下 implementation-specific property：write 一定实际执行且不会在该 scope 内合并；read 一定实际执行，不会被 forwarding、prefetch、combine，也不允许命中该 scope 内的 cache。部分 implementation 可能额外 load 周围 location，额外范围由实现决定，大小在 32–128 byte 之间。

#### 8.4.2 `volatile` Operation

##### English Original

> A volatile operation is a memory operation with `.volatile` qualifier specified. The semantics of volatile operations are equivalent to a relaxed memory operation with system-scope but with the following extra implementation-specific constraints:
>
> - The number of volatile instructions (not operations) executed by a program is preserved. Hardware may combine and merge volatile operations issued by multiple different volatile instructions, that is, the number of volatile operations in the program is not preserved.
> - Volatile instructions are not re-ordered around other volatile instructions, but the memory operations performed by those instructions may be re-ordered around each other.
>
> Note
>
> PTX volatile operations are intended for compilers to lower volatile read and write operations from CUDA C++, and other programming languages sharing CUDA C++ volatile semantics, to PTX.
>
> Since volatile operations are relaxed at system-scope with extra constraints, prefer using other strong read or write operations (e.g. `ld.relaxed.sys` or `st.relaxed.sys`) for Inter-Thread Synchronization instead, which may deliver better performance.
>
> PTX volatile operations are not suited for Memory Mapped IO (MMIO) because volatile operations do not preserve the number of memory operations performed, and may perform more or less operations than requested in a non-deterministic way. Use `.mmio` operations instead, which strictly preserve the number of operations performed.

##### 中文翻译

Volatile operation 是带 `.volatile` qualifier 的 memory operation。其语义等价于 system-scope relaxed memory operation，但额外具有 implementation-specific constraint：program 执行的 volatile instruction 数量保持不变，但 hardware 可以合并由不同 volatile instruction 发出的 volatile operation，因此 operation 数量不保证不变；volatile instruction 不会越过其他 volatile instruction 重排，但这些 instruction 实际执行的 memory operation 仍可能彼此重排。

PTX volatile operation 主要供 compiler 将 CUDA C++ 以及共享 CUDA C++ volatile 语义的语言中的 volatile read/write lowering 为 PTX。

由于 volatile 本质是带额外约束的 system-scope relaxed operation，进行 inter-thread synchronization 时应优先使用其他 strong read/write，例如 `ld.relaxed.sys`、`st.relaxed.sys`，它们可能具有更好性能。

PTX volatile 不适合 MMIO，因为它不保留实际 memory-operation 数量，可能以非确定方式执行多于或少于请求数量的 operation。MMIO 应使用严格保留 operation 数量的 `.mmio` operation。

#### 重点解读

`volatile`、`.mmio` 与 atomic/synchronization 是三类不同诉求。`volatile` 主要约束 compiler lowering 与 instruction 可见性；`.mmio` 强调每次设备访问必须真实发生；跨 thread 同步则应使用明确的 strong operation、scope 与 fence。把 `volatile` 当作通用同步或 MMIO 手段都会产生错误。

### 8.5 Scope

#### English Original

> Each strong operation must specify a scope, which is the set of threads that may interact directly with that operation and establish any of the relations described in the memory consistency model. There are four scopes:
>
> Table 21 Scopes

| Scope | Description |
|---|---|
| `.cta` | The set of all threads executing in the same CTA as the current thread. |
| `.cluster` | The set of all threads executing in the same cluster as the current thread. |
| `.gpu` | The set of all threads in the current program executing on the same compute device as the current thread. This also includes other kernel grids invoked by the host program on the same compute device. |
| `.sys` | The set of all threads in the current program, including all kernel grids invoked by the host program on all compute devices, and all threads constituting the host program itself. |

> Note that the warp is not a scope; the CTA is the smallest collection of threads that qualifies as a scope in the memory consistency model.

#### 中文翻译

每个 strong operation 都必须指定 scope，即能够直接与该 operation 交互并建立 memory consistency model 中各类 relation 的 thread set。四种 scope 的完整定义见 Table 21；表格只保留英文原文。

`.cta` 覆盖同一 CTA，`.cluster` 覆盖同一 cluster，`.gpu` 覆盖同一 compute device 上当前 program 的全部 thread（包括 host 在同一 device 上启动的其他 kernel grid），`.sys` 则覆盖当前 program 在所有 compute device 上的全部 kernel grid，以及 host program 自身的 thread。

Warp 不是 memory consistency scope；CTA 是符合该模型 scope 定义的最小 thread collection。

### 8.6 Proxies

#### English Original

> A memory proxy, or a proxy is an abstract label applied to a method of memory access. When two memory operations use distinct methods of memory access, they are said to be different proxies.
>
> A proxy fence is required to synchronize memory operations across different proxies. Although virtual aliases use the generic method of memory access, since using distinct virtual addresses behaves as if using different proxies, they require a proxy fence to establish memory ordering.
>
> Unless otherwise specified, memory operations as defined in Operation types use generic method of memory access, i.e. a generic proxy. Operations using methods of access distinct from the generic method include:
>
> - textures and surface accesses,
> - accesses to the same location via the same proxy using distinct virtual memory addresses,
> - async-proxy and tensormap-proxy accesses by `.async.bulk` operations,
> - fabric-proxy accesses by fabric operations,
> - readonly-proxy accesses by `ld.proxy::readonly` operations.
>
> The readonly proxy may be used to load read-only data. Refer to Data Movement and Conversion Instructions: `ld` for details.

#### 中文翻译

Memory proxy 是施加在某种 memory-access method 上的抽象 label。两个 memory operation 使用不同访问方法时，就属于不同 proxy。

不同 proxy 之间的 memory operation 需要 proxy fence 才能同步。Virtual alias 虽然都使用 generic access method，但通过不同 virtual address 访问时，其行为等同于不同 proxy，因此也需要 proxy fence 建立 memory ordering。

除非另有说明，Operation Types 中定义的 memory operation 使用 generic method，即 generic proxy。不同于 generic proxy 的访问包括 texture/surface access；通过不同 virtual address 访问同一 location；`.async.bulk` operation 的 async-proxy 与 tensormap-proxy access；fabric operation 的 fabric-proxy access；以及 `ld.proxy::readonly` 的 readonly-proxy access。Readonly proxy 可用于 load read-only data，详见 `ld` instruction。

#### 8.6.1 Strong Proxy Accesses

##### English Original

> Strong modifications through
>
> - generic-proxy,
> - async-proxy,
> - tensormap-proxy,
> - fabric-proxy,
>
> eventually become observable by strong accesses to the same location performed via a different proxy in that list, if modification and access are:
>
> - system-scope — even if these used distinct memory addresses — or
> - gpu-scope if the different proxies involved are in the following list:
>   - async-proxy,
>   - generic-proxy.

##### 中文翻译

通过 generic、async、tensormap 或 fabric proxy 完成的 strong modification，在满足以下条件时，最终会被上述列表中另一 proxy 对同一 location 的 strong access 观察到：两端都是 system scope，此时即使使用不同 memory address 也成立；或者两端是 gpu scope，且涉及的不同 proxy 仅为 async-proxy 与 generic-proxy。

#### 重点解读

Scope 解决“哪些 thread 被同步覆盖”，proxy 解决“通过哪条访问路径观察 memory”。即使 scope 足够大，如果 producer 和 consumer 通过不同 proxy 或不同 alias address 访问，也可能仍需对应 proxy fence。二者必须同时满足。

### 8.7 Morally Strong Operations

#### English Original

> Two operations are said to be morally strong relative to each other if they satisfy all of the following conditions:
>
> - The operations are related in program order (i.e, they are both executed by the same thread), or each operation is strong and specifies a scope that includes the thread executing the other operation.
> - Both operations are performed via the same proxy.
> - If both are memory operations, then they overlap completely.
>
> Most (but not all) of the axioms in the memory consistency model depend on relations between morally strong operations.

#### 中文翻译

如果两个 operation 同时满足以下条件，则称它们彼此 morally strong：二者由 program order 关联，也就是由同一 thread 执行；或者二者都是 strong operation，且各自 scope 都包含执行另一个 operation 的 thread。二者还必须通过同一 proxy 执行；如果二者都是 memory operation，则还必须 complete overlap。

Memory consistency model 中的大多数（但不是全部）axiom 都依赖 morally strong operation 之间的 relation。

#### 8.7.1 Conflict and Data-races

##### English Original

> Two overlapping memory operations are said to conflict when at least one of them is a write.
>
> Two conflicting memory operations are said to be in a data-race if they are not related in causality order and they are not morally strong.

##### 中文翻译

两个 overlapping memory operation 中至少一个为 write 时，二者 conflict。如果两个 conflicting memory operation 既没有 causality-order relation，也不 morally strong，则二者构成 data race。

#### 8.7.2 Limitations on Mixed-size Data-races

##### English Original

> A data-race between operations that overlap completely is called a uniform-size data-race, while a data-race between operations that overlap partially is called a mixed-size data-race.
>
> The axioms in the memory consistency model do not apply if a PTX program contains one or more mixed-size data-races. But these axioms are sufficient to describe the behavior of a PTX program with only uniform-size data-races.
>
> **Atomicity of mixed-size RMW operations**
>
> In any program with or without mixed-size data-races, the following property holds for every pair of overlapping atomic operations A1 and A2 such that each specifies a scope that includes the other: Either the read-modify-write operation specified by A1 is performed completely before A2 is initiated, or vice versa. This property holds irrespective of whether the two operations A1 and A2 overlap partially or completely.

##### 中文翻译

Complete-overlap operation 之间的 data race 称为 uniform-size data race；partial-overlap operation 之间的 data race 称为 mixed-size data race。

如果 PTX program 包含一个或多个 mixed-size data race，memory consistency model 的 axiom 不适用；但如果 program 只包含 uniform-size data race，这些 axiom 足以描述其行为。

无论 program 是否包含 mixed-size data race，对于任意两个 overlapping atomic operation A1、A2，只要双方 scope 都包含对方，就保证其中一个 RMW 在另一个开始之前完整执行。该属性与 A1、A2 是 partial overlap 还是 complete overlap 无关。

##### 重点解读

“Morally strong”不是普通语言中的强弱评价，而是模型中的精确定义：scope、proxy 和完整 overlap 缺一不可。Mixed-size race 会让大部分 axiom 整体失去适用性，因此在同一 byte region 上混用不同宽度的非同步访问，比同宽 race 更难推理；例外是满足双向 scope 覆盖的 overlapping atomic RMW 仍保留不可交错性。

### 8.8 Release and Acquire Patterns

#### English Original

> Some sequences of instructions give rise to patterns that participate in memory synchronization as described later. The release pattern makes prior operations from the current thread¹ visible to some operations from other threads. The acquire pattern makes some operations from other threads visible to later operations from the current thread.
>
> A release pattern on a location M consists of one of the following:
>
> - A release operation on M.
>
>   E.g.: `st.release [M];` or `atom.release [M];` or `mbarrier.arrive.release [M];`
>
> - Or a release or acquire-release operation on M followed by a strong write on M in program order.
>
>   E.g.: `st.release [M]; st.relaxed [M];`
>
> - Or a release or acquire-release memory fence followed by a strong write on M in program order.
>
>   E.g.: `fence.release; st.relaxed [M];` or `fence.release; atom.relaxed [M];`
>
> - Or a release or acquire-release memory fence followed in program order by an asynchronous operation that performs a strong write on M.
>
>   E.g.: `fence.release; cp.async.bulk.global.shared.relaxed.sys.b128 [M];`
>
> - Or a release asynchronous operation that performs a strong write on M.
>
>   E.g.: `st.async.release.sys [M];`
>
> Any memory synchronization established by a release pattern only affects operations occurring in program order before the first instruction in that pattern.
>
> An acquire pattern on a location M consists of one of the following:
>
> - An acquire operation on M.
>
>   E.g.: `ld.acquire [M];` or `atom.acquire [M];` or `mbarrier.test_wait.acquire [M];`
>
> - Or a strong read on M followed by an acquire operation on M in program order.
>
>   E.g.: `ld.relaxed [M]; ld.acquire [M];`
>
> - Or a strong read on M followed by an acquire memory fence in program order.
>
>   E.g.: `ld.relaxed [M]; fence.acquire;` or `atom.relaxed [M]; fence.acquire;`
>
> - Or first observing completion of an asynchronous operation that performs a strong read on M and then this observation is followed by an acquire memory fence in program order.

```ptx
cp.async.bulk.mbarrier::complete_tx::bytes.relaxed.sys.b128
    [dst], [M], size, [barrier];             // strong read on M
mbarrier.try_wait.relaxed p, [barrier];     // observes completion of async op
                                             // that performs strong read on M
@p fence.acquire;                            // acquire fence in program order
                                             // after observing completion
```

> Any memory synchronization established by an acquire pattern only affects operations occurring in program order after the last instruction in that pattern.
>
> Note that while atomic reductions conceptually perform a strong read as part of its read-modify-write sequence, this strong read does not form an acquire pattern.
>
> E.g.: `red.add [M], 1; fence.acquire;` is not an acquire pattern.
>
> ¹ For both release and acquire patterns, this effect is further extended to operations in other threads through the transitive nature of causality order.

#### 中文翻译

某些 instruction sequence 会形成参与 memory synchronization 的 pattern。Release pattern 使当前 thread 中较早的 operation 对其他 thread 的部分 operation 可见；acquire pattern 则使其他 thread 的部分 operation 对当前 thread 中较晚的 operation 可见。由于 causality order 具有传递性，release/acquire 的影响还可进一步扩展到其他 thread 的 operation。

Location M 上的 release pattern 可以是：直接对 M 执行 release operation；对 M 执行 release/acquire-release operation 后，再按 program order 对 M 执行 strong write；release/acquire-release fence 后按 program order strong-write M；release/acquire-release fence 后按 program order 启动一个 strong-write M 的 asynchronous operation；或直接执行 strong-write M 的 release asynchronous operation。原文列出的典型 instruction sequence 已完整保留。

Release pattern 建立的 memory synchronization 只影响 pattern 第一条 instruction 之前、按 program order 出现的 operation。

Location M 上的 acquire pattern 可以是：直接对 M 执行 acquire operation；strong-read M 后按 program order 对 M 执行 acquire operation；strong-read M 后执行 acquire fence；或先观察到某个 strong-read M 的 asynchronous operation 已完成，再按 program order 执行 acquire fence。

Acquire pattern 建立的 synchronization 只影响 pattern 最后一条 instruction 之后、按 program order 出现的 operation。

Atomic reduction 在概念上会在 RMW 中执行 strong read，但这个 read 不构成 acquire pattern；因此 `red.add [M], 1; fence.acquire;` 不是 acquire pattern。

#### 重点解读

Release 管“之前”，acquire 管“之后”：release pattern 把当前 thread 在 pattern 之前的效果推出去，acquire pattern 把其他 thread 的效果拉进来并约束 pattern 之后的操作。二者只有通过 observation relation 等条件真正配对时，才形成跨 thread synchronization。

### 8.9 Ordering of Memory Operations

#### English Original

> The sequence of operations performed by each thread is captured as program order while memory synchronization across threads is captured as causality order. The visibility of the side-effects of memory operations to other memory operations is captured as communication order. The memory consistency model defines contradictions that are disallowed between communication order on the one hand, and causality order and program order on the other.

#### 中文翻译

每个 thread 内的 operation sequence 由 program order 表示；跨 thread 的 memory synchronization 由 causality order 表示；memory operation side effect 对其他 memory operation 的可见性由 communication order 表示。Memory consistency model 定义 communication order 与 causality/program order 之间不允许出现的矛盾。

#### 8.9.1 Program Order

##### English Original

> The program order relates all operations performed by a thread to the order in which a sequential processor will execute instructions in the corresponding PTX source. It is a transitive relation that forms a total order over the operations performed by the thread, but does not relate operations from different threads.

##### 中文翻译

Program order 按照 sequential processor 执行相应 PTX source instruction 的顺序，关联同一 thread 执行的全部 operation。它是传递关系，在单个 thread 的 operation 上形成 total order，但不关联不同 thread 的 operation。

##### 8.9.1.1 Asynchronous Operations

###### English Original

> Some PTX instructions (all variants of `cp.async`, `cp.async.bulk`, `cp.reduce.async.bulk`, `wgmma.mma_async`) perform operations that are asynchronous to the thread that executed the instruction. These asynchronous operations are ordered after prior instructions in the same thread (except in the case of `wgmma.mma_async`), but they are not part of the program order for that thread. Instead, they provide weaker ordering guarantees as documented in the instruction description.
>
> For example, the loads and stores performed as part of a `cp.async` are ordered with respect to each other, but not to those of any other `cp.async` instructions initiated by the same thread, nor any other instruction subsequently issued by the thread with the exception of `cp.async.commit_group` or `cp.async.mbarrier.arrive`. The asynchronous mbarrier arrive-on operation performed by a `cp.async.mbarrier.arrive` instruction is ordered with respect to the memory operations performed by all prior `cp.async` operations initiated by the same thread, but not to those of any other instruction issued by the thread. The implicit mbarrier complete-tx operation that is part of all variants of `cp.async.bulk` and `cp.reduce.async.bulk` instructions is ordered only with respect to the memory operations performed by the same asynchronous instruction, and in particular it does not transitively establish ordering with respect to prior instructions from the issuing thread.

###### 中文翻译

部分 PTX instruction——包括全部 `cp.async`、`cp.async.bulk`、`cp.reduce.async.bulk`、`wgmma.mma_async` variant——执行相对于 issuing thread 异步的 operation。除 `wgmma.mma_async` 外，这些 asynchronous operation 排在同一 thread 的 prior instruction 之后，但它们并不属于该 thread 的 program order，只提供各 instruction description 中规定的较弱 ordering guarantee。

例如，同一 `cp.async` 内部的 load 与 store 彼此有序，但不与同一 thread 发起的其他 `cp.async`，也不与该 thread 随后发出的其他 instruction 建立顺序；例外是 `cp.async.commit_group` 与 `cp.async.mbarrier.arrive`。

`cp.async.mbarrier.arrive` 执行的 asynchronous mbarrier arrive-on operation，与同一 thread 先前启动的全部 `cp.async` memory operation 有序，但不与该 thread 发出的其他 instruction 有序。所有 `cp.async.bulk`、`cp.reduce.async.bulk` variant 隐含的 mbarrier complete-tx operation，只与同一 asynchronous instruction 执行的 memory operation 有序；特别是，它不会传递性地为 issuing thread 的更早 instruction 建立 ordering。

###### 重点解读

“Instruction 在 program order 中出现”不等于“它启动的 asynchronous memory operation 也进入同一 program order”。必须依赖该 async instruction 明确提供的 commit、barrier 或 completion mechanism 建立顺序，不能从普通 source-code 先后关系推导完整同步。

#### 8.9.2 Observation Order

##### English Original

> Observation order relates a write W to a read R through an optional sequence of atomic read-modify-write operations.
>
> A write W precedes a read R in observation order if:
>
> - R and W are morally strong and R reads the value written by W, or
> - For some atomic operation Z, W precedes Z and Z precedes R in observation order.

##### 中文翻译

Observation order 通过可选的 atomic RMW sequence，把 write W 关联到 read R。如果 R 与 W morally strong 且 R 读取 W 写入的 value，则 W 在 observation order 中先于 R；或者存在 atomic operation Z，使 W 先于 Z 且 Z 先于 R，也可形成该 relation。

#### 8.9.3 Fence-SC Order

##### English Original

> The Fence-SC order is an acyclic partial order, determined at runtime, that relates every pair of morally strong `fence.sc` operations.

##### 中文翻译

Fence-SC order 是 runtime 确定的无环 partial order，它关联每一对 morally strong `fence.sc` operation。

#### 8.9.4 Memory Synchronization

##### English Original

> Synchronizing operations performed by different threads synchronize with each other at runtime as described here. The effect of such synchronization is to establish causality order across threads.
>
> A `fence.sc` operation X synchronizes with a `fence.sc` operation Y if X precedes Y in the Fence-SC order.
>
> A `bar{.cta}.sync` or `bar{.cta}.red` or `bar{.cta}.arrive` operation synchronizes with a `bar{.cta}.sync` or `bar{.cta}.red` operation executed on the same barrier.
>
> A `barrier.cluster.arrive` operation synchronizes with a `barrier.cluster.wait` operation.
>
> A release pattern X synchronizes with an acquire pattern Y, if a write operation in X precedes a read operation in Y in observation order, and the first operation in X and the last operation in Y are morally strong.
>
> **API synchronization**
>
> A synchronizes relation can also be established by certain CUDA APIs.
>
> Completion of a task enqueued in a CUDA stream synchronizes with the start of the following task in the same stream, if any.
>
> For purposes of the above, recording or waiting on a CUDA event in a stream, or causing a cross-stream barrier to be inserted due to `cudaStreamLegacy`, enqueues tasks in the associated streams even if there are no direct side effects. An event record task synchronizes with matching event wait tasks, and a barrier arrival task synchronizes with matching barrier wait tasks.
>
> Start of a CUDA kernel synchronizes with start of all threads in the kernel. End of all threads in a kernel synchronize with end of the kernel.
>
> Start of a CUDA graph synchronizes with start of all source nodes in the graph. Completion of all sink nodes in a CUDA graph synchronizes with completion of the graph. Completion of a graph node synchronizes with start of all nodes with a direct dependency.
>
> Start of a CUDA API call to enqueue a task synchronizes with start of the task.
>
> Completion of the last task queued to a stream, if any, synchronizes with return from `cudaStreamSynchronize`. Completion of the most recently queued matching event record task, if any, synchronizes with return from `cudaEventSynchronize`. Synchronizing a CUDA device or context behaves as if synchronizing all streams in the context, including ones that have been destroyed.
>
> Returning `cudaSuccess` from an API to query a CUDA handle, such as a stream or event, behaves the same as return from the matching synchronization API.
>
> In addition to establishing a synchronizes relation, the CUDA API synchronization mechanisms above also participate in proxy-preserved base causality order except for the tensormap-proxy which is not acquired from generic-proxy at CUDA Kernel start and must therefore be acquired explicitly using `fence.proxy.tensormap::generic.acquire` when needed.

##### 中文翻译

不同 thread 执行的 synchronizing operation 会按本节规则在 runtime 彼此 synchronize，从而建立跨 thread causality order。

如果 `fence.sc` X 在 Fence-SC order 中先于 `fence.sc` Y，则 X synchronizes with Y。同一 barrier 上的 `bar{.cta}.sync`、`bar{.cta}.red` 或 `bar{.cta}.arrive`，会与该 barrier 上的 `bar{.cta}.sync` 或 `bar{.cta}.red` 同步。`barrier.cluster.arrive` 会与 `barrier.cluster.wait` 同步。

如果 release pattern X 中的 write 在 observation order 中先于 acquire pattern Y 中的 read，并且 X 的第一条 operation 与 Y 的最后一条 operation morally strong，则 X synchronizes with Y。

部分 CUDA API 也能建立 synchronizes relation：

- CUDA stream 中一个 enqueued task 完成后，与同一 stream 中下一 task 的开始同步。
- 在 stream 中 record/wait CUDA event，或因 `cudaStreamLegacy` 插入 cross-stream barrier，即使没有直接 side effect，也视为在相关 stream 中 enqueue task。Event-record task 与匹配的 event-wait task 同步；barrier-arrival task 与匹配的 barrier-wait task 同步。
- CUDA kernel 的开始与其中所有 thread 的开始同步；所有 kernel thread 的结束与 kernel 的结束同步。
- CUDA graph 的开始与所有 source node 的开始同步；全部 sink node 完成与 graph 完成同步；graph node 完成与所有直接依赖它的 node 开始同步。
- 用于 enqueue task 的 CUDA API call 开始，与 task 开始同步。
- Stream 中最后一个 queued task 完成，与 `cudaStreamSynchronize` 返回同步；最近一个匹配的 event-record task 完成，与 `cudaEventSynchronize` 返回同步。同步 CUDA device/context 等价于同步该 context 的全部 stream，包括已销毁的 stream。
- 查询 stream/event 等 CUDA handle 的 API 返回 `cudaSuccess`，其行为等同于相应 synchronization API 返回。

上述 CUDA API synchronization 除建立 synchronizes relation 外，也参与 proxy-preserved base causality order。例外是 tensormap-proxy：CUDA kernel 开始时不会自动从 generic-proxy acquire；需要时必须显式使用 `fence.proxy.tensormap::generic.acquire`。

#### 8.9.5 Causality Order

##### English Original

> Causality order captures how memory operations become visible across threads through synchronizing operations. The axiom “Causality” uses this order to constrain the set of write operations from which a read operation may read a value.
>
> Relations in the causality order primarily consist of relations in Base causality order¹, which is a transitive order, determined at runtime.
>
> **Base causality order**
>
> An operation X precedes an operation Y in base causality order if:
>
> - X precedes Y in program order, or
> - X synchronizes with Y, or
> - For some operation Z,
>   - X precedes Z in program order and Z precedes Y in base causality order, or
>   - X precedes Z in base causality order and Z precedes Y in program order, or
>   - X precedes Z in base causality order and Z precedes Y in base causality order.
>
> **Proxy-preserved base causality order**
>
> A memory operation X precedes a memory operation Y in proxy-preserved base causality order if X precedes Y in base causality order, and:
>
> - X and Y are performed to the same address, using the generic proxy, or
> - X and Y are performed to the same address, using the same proxy, and by the same thread block, or
> - X and Y are aliases and there is an alias proxy fence along the base causality path from X to Y.
>
> **Causality order**
>
> Causality order combines base causality order with some non-transitive relations as follows:
>
> An operation X precedes an operation Y in causality order if:
>
> - X precedes Y in proxy-preserved base causality order, or
> - For some operation Z, X precedes Z in observation order, and Z precedes Y in proxy-preserved base causality order.
>
> ¹ The transitivity of base causality order accounts for the “cumulativity” of synchronizing operations.

##### 中文翻译

Causality order 描述 memory operation 如何通过 synchronizing operation 跨 thread 变得可见。Causality axiom 使用该 order 约束 read 可以从哪些 write operation 读取 value。

Causality order 的 relation 主要来自 runtime 确定、具有传递性的 base causality order。X 在 base causality order 中先于 Y 的条件包括：X 在 program order 中先于 Y；X synchronizes with Y；或者通过某个 Z，将 program order、base causality order 或二者组合传递连接起来。Base causality order 的传递性体现 synchronizing operation 的“cumulativity”。

若 X 已在 base causality order 中先于 Y，并且满足以下任一 proxy-preservation 条件，则 X 在 proxy-preserved base causality order 中先于 Y：二者用 generic proxy 访问同一 address；二者由同一 thread block 使用同一 proxy 访问同一 address；或者二者是 alias，且 X 到 Y 的 base-causality path 上存在 alias proxy fence。

Causality order 把 base causality 与部分非传递 relation 结合起来：如果 X 在 proxy-preserved base causality order 中先于 Y，则 X causally precedes Y；或者存在 Z，使 X 在 observation order 中先于 Z，且 Z 在 proxy-preserved base causality order 中先于 Y，也成立。

##### 重点解读

跨 thread 的“因果链”不是单纯把 program order 拼接起来：它首先需要 synchronization，然后还要经过 proxy-preservation 过滤。Alias 或不同 proxy 可能切断本来存在的 base causality path，除非在路径上加入正确的 proxy fence。

#### 8.9.6 Coherence Order

##### English Original

> There exists a partial transitive order that relates overlapping write operations, determined at runtime, called the coherence order¹. Two overlapping write operations are related in coherence order if they are morally strong or if they are related in causality order. Two overlapping writes are unrelated in coherence order if they are in a data-race, which gives rise to the partial nature of coherence order.
>
> ¹ Coherence order cannot be observed directly since it consists entirely of write operations. It may be observed indirectly by its use in constraining the set of candidate writes that a read operation may read from.

##### 中文翻译

Runtime 会确定一个关联 overlapping write operation 的传递 partial order，称为 coherence order。如果两个 overlapping write morally strong，或在 causality order 中相关，则二者在 coherence order 中相关；如果两个 overlapping write 构成 data race，则它们在 coherence order 中不相关，这也是该 order 呈 partial 的原因。

Coherence order 完全由 write operation 构成，无法直接观察；但它会约束 read 可选择的 candidate write，因此可以被间接观察。

#### 8.9.7 Communication Order

##### English Original

> The communication order is a non-transitive order, determined at runtime, that relates write operations to other overlapping memory operations.
>
> A write W precedes an overlapping read R in communication order if R returns the value of any byte that was written by W.
>
> A write W precedes a write W’ in communication order if W precedes W’ in coherence order.
>
> A read R precedes an overlapping write W in communication order if, for any byte accessed by both R and W, R returns the value written by a write W’ that precedes W in coherence order.
>
> Communication order captures the visibility of memory operations — when a memory operation X1 precedes a memory operation X2 in communication order, X1 is said to be visible to X2.

##### 中文翻译

Communication order 是 runtime 确定的非传递 order，用于把 write operation 与其他 overlapping memory operation 关联起来。

如果 overlapping read R 返回了 write W 写入的任意 byte value，则 W 在 communication order 中先于 R。如果 W 在 coherence order 中先于 W’，则 W 在 communication order 中也先于 W’。如果对于 R 与 W 共同访问的任意 byte，R 返回的 value 来自 coherence order 中先于 W 的某个 W’，则 R 在 communication order 中先于 W。

Communication order 描述 memory-operation visibility：当 X1 在 communication order 中先于 X2 时，就说 X1 对 X2 可见。

##### 重点解读

三类 order 的分工可以概括为：program order 描述单 thread 的执行次序；causality order 描述同步产生的跨 thread 因果关系；communication order 描述实际 value/side effect 的可见方向；coherence order 则给 overlapping write 提供部分排序。后续 axiom 的核心，就是禁止这些 order 之间产生矛盾。

### 8.10 Axioms

#### 8.10.1 Coherence

##### English Original

> If a write W precedes an overlapping write W’ in causality order, then W must precede W’ in coherence order.

##### 中文翻译

如果 write W 在 causality order 中先于 overlapping write W’，则 W 必须在 coherence order 中先于 W’。

#### 8.10.2 Fence-SC

##### English Original

> Fence-SC order cannot contradict causality order. For a pair of morally strong `fence.sc` operations F1 and F2, if F1 precedes F2 in causality order, then F1 must precede F2 in Fence-SC order.

##### 中文翻译

Fence-SC order 不能与 causality order 矛盾。对于一对 morally strong `fence.sc` operation F1、F2，如果 F1 在 causality order 中先于 F2，则 F1 也必须在 Fence-SC order 中先于 F2。

#### 8.10.3 Atomicity

##### English Original

> **Single-Copy Atomicity**
>
> Conflicting morally strong operations are performed with single-copy atomicity. When a read R and a write W are morally strong, then the following two communications cannot both exist in the same execution, for the set of bytes accessed by both R and W:
>
> - R reads any byte from W.
> - R reads any byte from any write W’ which precedes W in coherence order.
>
> **Atomicity of read-modify-write (RMW) operations**
>
> When an atomic operation A and a write W overlap and are morally strong, then the following two communications cannot both exist in the same execution, for the set of bytes accessed by both A and W:
>
> - A reads any byte from a write W’ that precedes W in coherence order.
> - A follows W in coherence order.
>
> **Litmus Test 1**

```ptx
.global .u32 x = 0;

// T1                              // T2
A1: atom.sys.inc.u32 %r0, [x];     A2: atom.sys.inc.u32 %r0, [x];

// FINAL STATE: x == 2
```

> Atomicity is guaranteed when the operations are morally strong.
>
> **Litmus Test 2**

```ptx
.global .u32 x = 0;

// T1                              // T2 (In a different CTA)
A1: atom.cta.inc.u32 %r0, [x];     A2: atom.gpu.inc.u32 %r0, [x];

// FINAL STATE: x == 1 OR x == 2
```

> Atomicity is not guaranteed if the operations are not morally strong.

##### 中文翻译

**Single-Copy Atomicity：**conflicting 且彼此 morally strong 的 operation 以 single-copy atomicity 执行。当 read R 与 write W morally strong 时，对二者共同访问的 byte set 而言，以下两种 communication 不能同时存在：R 从 W 读取某个 byte；同时 R 又从 coherence order 中先于 W 的某个 W’ 读取某个 byte。

**RMW Atomicity：**当 atomic operation A 与 write W overlap 且 morally strong 时，对二者共同访问的 byte set 而言，以下两种 communication 不能同时存在：A 从 coherence order 中先于 W 的某个 W’ 读取 byte；同时 A 又在 coherence order 中位于 W 之后。

Litmus Test 1 中，两个 system-scope atomic increment morally strong，因此保证 final `x == 2`。Litmus Test 2 的 T1、T2 位于不同 CTA；`.cta` operation 的 scope 不包含另一 CTA，两个 operation 不 morally strong，因此 atomicity 不受该 axiom 保证，final state 可以是 `x == 1` 或 `x == 2`。

##### 重点解读

使用 atomic opcode 本身并不足以保证两个 operation 彼此 atomic。它们还必须在 scope 上互相覆盖、使用同一 proxy，并满足 complete overlap，才能 morally strong 并获得这里的 single-copy/RMW atomicity guarantee。

#### 8.10.4 No Thin Air

##### English Original

> Values may not appear “out of thin air”: an execution cannot speculatively produce a value in such a way that the speculation becomes self-satisfying through chains of instruction dependencies and inter-thread communication. This matches both programmer intuition and hardware reality, but is necessary to state explicitly when performing formal analysis.
>
> **Litmus Test: Load Buffering with true dependencies**

```ptx
.global .u32 x = 0;
.global .u32 y = 0;

// T1                              // T2
A1: ld.global.u32 %r0, [x];        A2: ld.global.u32 %r1, [y];
B1: st.global.u32 [y], %r0;        B2: st.global.u32 [x], %r1;

// FINAL STATE: x == 0 AND y == 0
```

> The litmus test known as “LB+deps” (Load Buffering with dependencies) checks such forbidden values that may arise out of thin air. Two threads T1 and T2 each read from a first variable and copy the observed result into a second variable, with the first and second variable exchanged between the threads. If each variable is initially zero, the final result shall also be zero. If A1 reads from B2 and A2 reads from B1, then values passing through the memory operations in this example form a cycle: A1->B1->A2->B2->A1. Only the values `x == 0` and `y == 0` are allowed to satisfy this cycle. If any of the memory operations in this example were to speculatively associate a different value with the corresponding memory location, then such a speculation would become self-fulfilling, and hence forbidden.
>
> **Litmus Test: Load Buffering without dependencies**

```ptx
.global .u32 x = 0;
.global .u32 y = 0;

// T1                              // T2
A1: ld.global.u32 %r0, [x];        A2: ld.global.u32 %r1, [y];
B1: st.global.u32 [y], 1;          B2: st.global.u32 [x], 1;

// FINAL STATE: x == 1 AND y == 1
```

> This litmus test differs from the one above in that it unconditionally stores 1 to x and y. In this litmus test a final state of `x == 1` and `y == 1` is permitted. This execution does not contradict the requirement demonstrated by the previous litmus test. Here there is no self-fulfilling cycle – the litmus test will always and unconditionally store 1 to x and y, so here the cycle is not self-fulfilled and the speculation is valid.
>
> Here the lack of dependencies is plain, but the implementation may perform any chain of reasoning to determine that a store is not dependent on a prior load, and thus break self-fulfilling cycles which would otherwise apparently be forbidden by the No-Thin-Air axiom.
>
> This form of load buffering is deliberately permitted in the PTX memory consistency model.

##### 中文翻译

Value 不能“凭空出现”：execution 不能推测性地产生某个 value，再让该 speculation 通过 instruction dependency 与 inter-thread communication chain 自我满足。该要求符合 programmer intuition 与 hardware reality，但进行形式化分析时必须显式写出。

第一个 litmus test 称为 “LB+deps”（带 dependency 的 Load Buffering）。T1、T2 各自读取一个 variable，再把读到的结果写入另一个 variable，两条 thread 的 source/destination 对调。若 `x`、`y` 初始均为 0，最终也必须均为 0。

若 A1 从 B2 读取、A2 从 B1 读取，则 value 在 memory operation 间形成环 `A1->B1->A2->B2->A1`。只有 `x == 0`、`y == 0` 能合法满足该环。若任一 operation 推测性地为 location 关联其他 value，该 speculation 会通过这个环自我实现，因此被禁止。

第二个 test 没有 dependency：B1、B2 无条件分别向 `y`、`x` 写入 1。因此 `x == 1 AND y == 1` 合法，不违反前一个 test；这些 value 无论 read 结果如何都会被写入，不存在 self-fulfilling cycle。

尽管示例中的无 dependency 很明显，implementation 可以使用任意 reasoning chain 判断 store 不依赖 prior load，从而打破看起来会被 No-Thin-Air axiom 禁止的 self-fulfilling cycle。PTX memory consistency model 有意允许这种 load buffering。

##### 重点解读

No-Thin-Air 禁止的是“值仅因假设其存在而存在”的循环推测，不是禁止所有 load buffering 或 speculative execution。只要 store value 能独立于 prior load 确定，dependency cycle 就可以被实现打破。

#### 8.10.5 Sequential Consistency Per Location

##### English Original

> Within any set of overlapping memory operations that are pairwise morally strong, communication order cannot contradict program order, i.e., a concatenation of program order between overlapping operations and morally strong relations in communication order cannot result in a cycle. This ensures that each program slice of overlapping pairwise morally strong operations is strictly sequentially-consistent.
>
> **Litmus Test: CoRR**

```ptx
.global .u32 x = 0;

// T1                                      // T2
W1: st.global.relaxed.sys.u32 [x], 1;      R1: ld.global.relaxed.sys.u32 %r0, [x];
                                          R2: ld.global.relaxed.sys.u32 %r1, [x];

// IF %r0 == 1 THEN %r1 == 1
```

> The litmus test “CoRR” (Coherent Read-Read), demonstrates one consequence of this guarantee. A thread T1 executes a write W1 on a location x, and a thread T2 executes two (or an infinite sequence of) reads R1 and R2 on the same location x. No other writes are executed on x, except the one modelling the initial value. The operations W1, R1 and R2 are pairwise morally strong. If R1 reads from W1, then the subsequent read R2 must also observe the same value. If R2 observed the initial value of x instead, then this would form a sequence of morally-strong relations R2->W1->R1 in communication order that contradicts the program order R1->R2 in thread T2. Hence R2 cannot read the initial value of x in such an execution.

##### 中文翻译

在任意一组彼此 pairwise morally strong 的 overlapping memory operation 中，communication order 不能与 program order 矛盾。也就是说，把 overlapping operation 间的 program-order relation 与 communication order 中的 morally-strong relation 串接起来，不能形成环。这保证每个由 pairwise morally strong overlapping operation 构成的 program slice 都严格 sequentially consistent。

“CoRR”（Coherent Read-Read）litmus test 展示该保证的一个结果：T1 对 `x` 执行 W1，T2 对同一 `x` 依次执行 R1、R2；除 initial write 外没有其他 write。W1、R1、R2 两两 morally strong。如果 R1 读到 W1，后续 R2 必须观察到同一 value。

若 R2 反而读到 `x` 的 initial value，communication order 会形成 morally-strong chain `R2->W1->R1`，与 T2 中的 program order `R1->R2` 矛盾。因此这种 execution 中 R2 不能读到 initial value。

#### 8.10.6 Causality

##### English Original

> Relations in communication order cannot contradict causality order. This constrains the set of candidate write operations that a read operation may read from:
>
> - If a read R precedes an overlapping write W in causality order, then R cannot read from W.
> - If a write W precedes an overlapping read R in causality order, then for any byte accessed by both R and W, R cannot read from any write W’ that precedes W in coherence order.
>
> **Litmus Test: Message Passing**

```ptx
.global .u32 data = 0;
.global .u32 flag = 0;

// T1                                      // T2
W1: st.global.u32 [data], 1;               R1: ld.global.relaxed.sys.u32 %r0, [flag];
F1: fence.sys;                             F2: fence.sys;
W2: st.global.relaxed.sys.u32 [flag], 1;    R2: ld.global.u32 %r1, [data];

// IF %r0 == 1 THEN %r1 == 1
```

> The litmus test known as “MP” (Message Passing) represents the essence of typical synchronization algorithms. A vast majority of useful programs can be reduced to sequenced applications of this pattern.
>
> Thread T1 first writes to a data variable and then to a flag variable while a second thread T2 first reads from the flag variable and then from the data variable. The operations on the flag are morally strong and the memory operations in each thread are separated by a fence, and these fences are morally strong.
>
> If R1 observes W2, then the release pattern “F1; W2” synchronizes with the acquire pattern “R1; F2”. This establishes the causality order `W1 -> F1 -> W2 -> R1 -> F2 -> R2`. Then axiom causality guarantees that R2 cannot read from any write that precedes W1 in coherence order. In the absence of any other writes in this example, R2 must read from W1.
>
> **Litmus Test: CoWR**

```ptx
// These addresses are aliases
.global .u32 data_alias_1;
.global .u32 data_alias_2;

// T1
W1: st.global.u32 [data_alias_1], 1;
F1: fence.proxy.alias;
R1: ld.global.u32 %r1, [data_alias_2];

// %r1 == 1
```

> Virtual aliases require an alias proxy fence along the synchronization path.
>
> **Litmus Test: Store Buffering**
>
> The litmus test known as “SB” (Store Buffering) demonstrates the sequential consistency enforced by the `fence.sc`. A thread T1 writes to a first variable, and then reads the value of a second variable, while a second thread T2 writes to the second variable and then reads the value of the first variable. The memory operations in each thread are separated by `fence.sc` instructions, and these fences are morally strong.

```ptx
.global .u32 x = 0;
.global .u32 y = 0;

// T1                              // T2
W1: st.global.u32 [x], 1;          W2: st.global.u32 [y], 1;
F1: fence.sc.sys;                  F2: fence.sc.sys;
R1: ld.global.u32 %r0, [y];        R2: ld.global.u32 %r1, [x];

// %r0 == 1 OR %r1 == 1
```

> In any execution, either F1 precedes F2 in Fence-SC order, or vice versa. If F1 precedes F2 in Fence-SC order, then F1 synchronizes with F2. This establishes the causality order in `W1 -> F1 -> F2 -> R2`. Axiom causality ensures that R2 cannot read from any write that precedes W1 in coherence order. In the absence of any other write to that variable, R2 must read from W1. Similarly, in the case where F2 precedes F1 in Fence-SC order, R1 must read from W2. If each `fence.sc` in this example were replaced by a `fence.acq_rel` instruction, then this outcome is not guaranteed. There may be an execution where the write from each thread remains unobserved from the other thread, i.e., an execution is possible, where both R1 and R2 return the initial value “0” for variables y and x respectively.

##### 中文翻译

Communication-order relation 不能与 causality order 矛盾，因此 read 可选择的 candidate write 受到以下限制：若 read R 在 causality order 中先于 overlapping write W，则 R 不能从 W 读取；若 write W 在 causality order 中先于 overlapping read R，则对二者共同访问的任意 byte，R 都不能从 coherence order 中早于 W 的 W’ 读取。

**Message Passing：**“MP” 是典型 synchronization algorithm 的核心，大量有用 program 都可归约为该 pattern 的顺序组合。T1 先写 `data`，再写 `flag`；T2 先读 `flag`，再读 `data`。Flag operation morally strong，且每个 thread 内的 memory operation 由 morally strong fence 分隔。

若 R1 观察到 W2，release pattern `F1; W2` 就与 acquire pattern `R1; F2` 同步，建立 `W1 -> F1 -> W2 -> R1 -> F2 -> R2` 的 causality order。Causality axiom 保证 R2 不能读取 coherence order 中早于 W1 的 write；示例没有其他 write，因此 R2 必须读到 W1。也就是说，只要 `%r0 == 1`，就必须有 `%r1 == 1`。

**CoWR：**`data_alias_1`、`data_alias_2` 是指向同一 location 的 virtual alias。Synchronization path 上必须有 `fence.proxy.alias`，才能保证从第二个 alias 读取到经第一个 alias 写入的 value。

**Store Buffering：**“SB” 展示 `fence.sc` 强制的 sequential consistency。T1、T2 各自先写一个 variable，再读另一个 variable，且每个 thread 的 write/read 之间都有 morally strong `fence.sc`。

任意 execution 中，Fence-SC order 必须让 F1、F2 中一个先于另一个。若 F1 先于 F2，F1 synchronizes with F2，建立 `W1 -> F1 -> F2 -> R2`，因此 R2 必须读到 W1。反向时 R1 必须读到 W2，所以至少 `%r0`、`%r1` 中一个为 1。

如果把两条 `fence.sc` 都替换为 `fence.acq_rel`，则不再保证该结果；可能存在双方 write 都未被另一 thread 观察到的 execution，使 R1、R2 都返回对应 variable 的 initial value 0。

##### 重点解读

Message Passing 依赖 release/acquire pattern 经 observation order 配对；Store Buffering 的更强结论则依赖 Fence-SC order 在两条 `fence.sc` 之间选定方向。`fence.acq_rel` 不会自动建立这种全局 Fence-SC 排序，所以不能作为 `fence.sc` 的等价替代。

### 8.11 Special Cases

#### 8.11.1 Reductions Do Not Form Acquire Patterns

##### English Original

> Atomic reduction operations like `red` do not form acquire patterns with acquire fences.
>
> **Litmus Test: Message Passing with a Red Instruction**

```ptx
.global .u32 x = 0;
.global .u32 flag = 0;

// T1                                  // T2
W1: st.u32 [x], 42;                    RMW1: red.sys.global.add.u32 [flag], 1;
W2: st.release.gpu.u32 [flag], 1;      F2: fence.acquire.gpu;
                                      R2: ld.weak.u32 %r1, [x];

// %r1 == 0 AND flag == 2
```

> The litmus test known as “MP” (Message Passing) demonstrates the consequence of reductions being excluded from acquire patterns. It is possible to observe the outcome where R2 reads the value 0 from x and flag has the final value of 2. This outcome is possible since the release pattern in T1 does not synchronize with any acquire pattern in T2. Using the `atom` instruction instead of `red` forbids this outcome.

##### 中文翻译

`red` 等 atomic reduction operation 不会与 acquire fence 共同形成 acquire pattern。

该 Message Passing test 展示 reduction 被排除在 acquire pattern 之外的后果：R2 可能从 `x` 读到 0，同时 `flag` 的 final value 为 2。原因是 T1 的 release pattern 没有与 T2 中任何 acquire pattern 同步。若用 `atom` 替代 `red`，则会禁止该结果。

##### 重点解读

`red` 虽是 atomic RMW，但不会向程序暴露 read result，因此规范明确不把其内部概念性 strong read 用来构造 acquire pattern。需要通过 RMW value 建立 acquire synchronization 时，应使用符合条件的 `atom`，而不是在 `red` 后简单追加 acquire fence。

