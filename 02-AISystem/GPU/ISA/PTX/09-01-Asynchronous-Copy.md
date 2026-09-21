### 9.7.10.28 Data Movement and Conversion Instructions: Asynchronous Copy

#### English Original

> An asynchronous copy operation performs the underlying operation asynchronously in the background, thus allowing the issuing threads to perform subsequent tasks.
>
> An asynchronous copy operation can be a bulk operation that operates on a large amount of data, or a non-bulk operation that operates on smaller sized data. The amount of data handled by a bulk asynchronous operation must be a multiple of 16 bytes.
>
> An asynchronous copy operation typically includes the following sequence:
>
> - Optionally, reading from the tensormap.
> - Reading data from the source location(s).
> - Writing data to the destination location(s).
> - Writes being made visible to the executing thread or other threads.

#### 中文翻译

Asynchronous copy operation 在后台异步执行底层操作，使发起该操作的 thread 能继续处理后续任务。

异步复制可以是处理大量数据的 bulk operation，也可以是处理较小数据量的 non-bulk operation。Bulk asynchronous operation 处理的数据量必须是 16 byte 的整数倍。

一次异步复制通常依次包含以下阶段：可选地读取 tensormap；从 source location 读取数据；向 destination location 写入数据；最后使这些写入对执行 thread 或其他 thread 可见。

#### 重点解读

“发起指令返回”与“复制完成并可安全读取”是不同事件。性能收益来自在两者之间安排独立计算；若立即等待，虽然语义正确，却难以隐藏搬运延迟。Bulk/non-bulk 不仅区别于数据量，后续的 completion group 也必须分别管理。

#### 9.7.10.28.1 Completion Mechanisms for Asynchronous Copy Operations

##### English Original

> A thread must explicitly wait for the completion of an asynchronous copy operation in order to access the result of the operation. Once an asynchronous copy operation is initiated, modifying the source memory location or tensor descriptor or reading from the destination memory location before the asynchronous operation completes, exhibits undefined behavior.
>
> This section describes two asynchronous copy operation completion mechanisms supported in PTX: Async-group mechanism and mbarrier-based mechanism.
>
> Asynchronous operations may be tracked by either of the completion mechanisms or both mechanisms. The tracking mechanism is instruction/instruction-variant specific.

##### 中文翻译

Thread 必须显式等待 asynchronous copy operation 完成，才能访问其结果。异步复制一旦发起，在其完成之前修改 source memory location 或 tensor descriptor，或者读取 destination memory location，都会导致未定义行为。

PTX 支持两种异步复制完成机制：async-group mechanism 和 mbarrier-based mechanism。Asynchronous operation 可以由其中一种机制跟踪，也可以同时由两种机制跟踪；具体采用哪一种取决于 instruction 及其 variant。

##### 重点解读

等待完成是数据正确性的前提，并不只是优化流水线的性能选项。尤其是 tensor copy，tensor descriptor 也必须在操作未完成期间保持不变；不能只保护 source/destination buffer，而忽略描述符的生命周期。

##### 9.7.10.28.1.1 Async-Group Mechanism

###### English Original

> When using the async-group completion mechanism, the issuing thread specifies a group of asynchronous operations, called async-group, using a commit operation and tracks the completion of this group using a wait operation. The thread issuing the asynchronous operation must create separate async-groups for bulk and non-bulk asynchronous operations.
>
> A commit operation creates a per-thread async-group containing all prior asynchronous operations tracked by async-group completion and initiated by the executing thread but none of the asynchronous operations following the commit operation. A committed asynchronous operation belongs to a single async-group.
>
> When an async-group completes, all the asynchronous operations belonging to that group are complete and the executing thread that initiated the asynchronous operations can read the result of the asynchronous operations. All async-groups committed by an executing thread always complete in the order in which they were committed. There is no ordering between asynchronous operations within an async-group.
>
> A typical pattern of using async-group as the completion mechanism is as follows:
>
> - Initiate the asynchronous operations.
> - Group the asynchronous operations into an async-group using a commit operation.
> - Wait for the completion of the async-group using the wait operation.
> - Once the async-group completes, access the results of all asynchronous operations in that async-group.

###### 中文翻译

使用 async-group completion mechanism 时，发起 thread 通过 commit operation 把若干 asynchronous operation 组成一个 async-group，再用 wait operation 跟踪该 group 的完成情况。发起异步操作的 thread 必须分别为 bulk 与 non-bulk asynchronous operation 建立 async-group。

一次 commit 会创建当前 thread 专属的 async-group：它包含该 thread 此前发起、由 async-group completion 跟踪、且尚未被归入 group 的 asynchronous operation；不包含 commit 之后发起的 operation。每个已经 commit 的 asynchronous operation 只属于一个 async-group。

Async-group 完成意味着其中所有 asynchronous operation 都已完成，发起 thread 随后可以读取它们的结果。同一 thread commit 的各个 async-group 总是按 commit 顺序完成；但同一个 async-group 内的 asynchronous operation 之间不保证顺序。

典型用法是先发起一批异步操作，再用 commit 将其组成 async-group，随后 wait 该 group 完成，最后访问 group 中所有操作的结果。

###### 重点解读

Async-group 是 per-thread 的完成跟踪单元，不是 CTA-wide barrier。不同 group 之间有 commit 顺序约束，同一 group 内却没有逐操作顺序保证；如果后一次复制依赖前一次复制的结果，不能仅因它们同在一个 group 就认为依赖已满足。

##### 9.7.10.28.1.2 Mbarrier-Based Mechanism

###### English Original

> A thread can track the completion of one or more asynchronous operations using the current phase of an mbarrier object. When the current phase of the mbarrier object is complete, it implies that all asynchronous operations tracked by this phase are complete, and all threads participating in that mbarrier object can access the result of the asynchronous operations.
>
> The mbarrier object to be used for tracking the completion of an asynchronous operation can be either specified along with the asynchronous operation as part of its syntax, or as a separate operation. For bulk asynchronous operations and fabric operations, the mbarrier object must be specified in the asynchronous operation, whereas for non-bulk operations, it can be specified after the asynchronous operation.
>
> A typical pattern of using mbarrier-based completion mechanism is as follows:
>
> - Initiate the asynchronous operations.
> - Set up an mbarrier object to track the asynchronous operations in its current phase, either as part of the asynchronous operation or as a separate operation.
> - Wait for the mbarrier object to complete its current phase using `mbarrier.test_wait` or `mbarrier.try_wait`.
> - Once the `mbarrier.test_wait` or `mbarrier.try_wait` operation returns `True`, access the results of the asynchronous operations tracked by the mbarrier object.

###### 中文翻译

Thread 可以使用 mbarrier object 的 current phase 跟踪一个或多个 asynchronous operation。当该 phase 完成时，表示它跟踪的所有 asynchronous operation 均已完成，参与该 mbarrier object 的所有 thread 都可以访问这些操作的结果。

用于跟踪异步操作的 mbarrier object，可以作为异步操作语法的一部分同时指定，也可以由单独的 operation 指定。对于 bulk asynchronous operation 和 fabric operation，必须在异步操作中指定 mbarrier object；对于 non-bulk operation，则可以在异步操作之后指定。

典型流程是发起异步操作，将其纳入 mbarrier object 的 current phase（可在指令中或通过单独操作完成），使用 `mbarrier.test_wait` 或 `mbarrier.try_wait` 等待该 phase 完成；只有 wait 返回 `True` 后，才能访问该 mbarrier 跟踪的异步操作结果。

###### 重点解读

Async-group 主要由发起 thread 自己 commit/wait；mbarrier 则把“完成”绑定到 barrier 的 phase，使参与该 barrier 的多个 thread 能在同一阶段完成后安全消费结果。因此在生产者搬运、消费者计算分工的流水线中，mbarrier 比 per-thread group 更适合作为跨 thread 的交接点。这里的“可访问”仍以参与正确 phase 且等待成功为条件。

##### 9.7.10.28.1.3 Report Mechanisms for Asynchronous Copy Operations

###### English Original

> Asynchronous copy operations with copy-direction from `.global` to `.shared::cta` or `.shared::cluster` support reporting data validity through an mbarrier object. When the qualifier `.report_mechanism` is specified, the copy operation inspects the data being copied and signals a [report-on](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-report-on) operation on the mbarrier if the data matches the pattern as specified in the qualifier. To avoid data inspection, the report mechanism must be set to `.mbarrier::report::disabled`. When the report mechanism is unspecified, it defaults to `.mbarrier::report::disabled`. The granularity of the data inspection is also specified as part of the qualifier. Note that when the `.report_mechanism` is specified for data inspection, the associated mbarrier operand must have been initialized with the layout `.layout::v1`, otherwise the behavior is undefined. For more details on mbarrier layout, refer [Layouts of the mbarrier object](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-object-layout).
>
> [Table 33](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#report-mechanisms-for-asynchronous-copy-operations) lists all supported report mechanisms.
>
> Table 33 Report mechanisms for asynchronous copy operations

| Report mechanism `.mbarrier::report::validity` | Elmemt format | Matching bit pattern | Granularity of the source data that is written into | Details of data inspection | Result of inspection |
|---|---|---|---|---|---|
| `::disabled` | N/A | N/A | N/A | No inspection | N/A |
| `::per_16bytes::80000000` | FP32 | -0 | 16 bytes of 4 FP32 elements | A sampled element from the aligned 16 byte chunk of the source global memory is matched with the Matching bit-pattern | If the pattern matches, issues a [report-on](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-report-on) operation on the mbarrier to update the payload report to a non-zero value. In particular, the report predicate is modified to non-zero value; whereas report value is altered in an idempotent way (atomically OR’ed with 0) |
| `::per_16bytes::8000` | FP16 | -0 | 16 bytes of 8 FP16 elements | A sampled element from the aligned 16 byte chunk of the source global memory is matched with the Matching bit-pattern | If the pattern matches, issues a report-on operation on the mbarrier to update the payload report to a non-zero value. In particular, the report predicate is modified to non-zero value; whereas report value is altered in an idempotent way (atomically OR’ed with 0) |
| `::per_16bytes::80` | FP8 | -0 | 16 bytes of 16 FP8 elements | A sampled element from the aligned 16 byte chunk of the source global memory is matched with the Matching bit-pattern | If the pattern matches, issues a report-on operation on the mbarrier to update the payload report to a non-zero value. In particular, the report predicate is modified to non-zero value; whereas report value is altered in an idempotent way (atomically OR’ed with 0) |
| `::per_16bytes::8` | FP4 | -0 | 16 bytes of 32 FP4 elements | A sampled element from the aligned 16 byte chunk of the source global memory is matched with the Matching bit-pattern | If the pattern matches, issues a report-on operation on the mbarrier to update the payload report to a non-zero value. In particular, the report predicate is modified to non-zero value; whereas report value is altered in an idempotent way (atomically OR’ed with 0) |
| `::per_element::ff` | 8bit/ 4bit | NaN / 0xF | 1 byte | Each element is matched with the Matching bit-pattern | If the pattern matches, issues a report-on operation on the mbarrier to update the payload report to a non-zero value. In particular, the report predicate is modified to non-zero value; whereas report value is altered in an idempotent way (atomically OR’ed with 0) |

###### 中文翻译

从 `.global` 复制到 `.shared::cta` 或 `.shared::cluster` 的 asynchronous copy operation，可以通过 mbarrier object 报告 data validity。指定 `.report_mechanism` qualifier 时，copy operation 会检查正在复制的数据；若数据与 qualifier 中指定的 pattern 匹配，就在 mbarrier 上触发一次 `report-on` operation。

若不需要检查数据，应将 report mechanism 设为 `.mbarrier::report::disabled`；未指定时也默认采用该值。检查粒度由 qualifier 一同指定。如果启用了用于 data inspection 的 `.report_mechanism`，相关 mbarrier operand 必须以 `.layout::v1` 初始化，否则行为未定义。Mbarrier layout 的细节见原文链接。Table 33 完整列出支持的 report mechanism；为适配 Markdown，原表合并单元格所涵盖的内容在各行展开，表格本身不另作中文翻译。

###### 重点解读

Report mechanism 是“复制过程中顺带检查数据，并把匹配结果写入 barrier payload”，不是判断 copy 是否完成的替代机制。消费者仍需按 mbarrier phase 正确等待；只有 wait 成功后，才能安全读取复制结果和相应 report。`::per_16bytes` 对每个对齐的 16-byte chunk 抽样匹配，`::per_element` 则按 element 匹配，两者的检查粒度不同。

#### 9.7.10.28.2 Async Proxy

##### English Original

> The `cp{.reduce}.async.bulk` operations are performed in the *asynchronous proxy* (or *async proxy*).
>
> Accessing the same memory location across multiple proxies needs a cross-proxy fence. For the async proxy, `fence.proxy.async` should be used to synchronize memory between *generic proxy* and the *async proxy*.
>
> The completion of a `cp{.reduce}.async.bulk` operation is followed by an implicit *generic-async* proxy fence. So the result of the asynchronous operation is made visible to the generic proxy as soon as its completion is observed. *Async-group* OR *mbarrier-based* completion mechanism must be used to wait for the completion of the `cp{.reduce}.async.bulk` instructions.

##### 中文翻译

`cp{.reduce}.async.bulk` 操作在 asynchronous proxy（简称 async proxy）中执行。

跨多个 proxy 访问同一内存位置时，需要 cross-proxy fence。对于 async proxy，应使用 `fence.proxy.async` 在 generic proxy 与 async proxy 之间同步内存。

`cp{.reduce}.async.bulk` 完成后，会隐式执行一次 generic–async proxy fence。因此，一旦观察到异步操作完成，其结果就对 generic proxy 可见。必须使用 async-group 或 mbarrier-based completion mechanism，等待 `cp{.reduce}.async.bulk` 指令完成。

##### 重点解读

这里有两个不同方向的要求：发起 bulk copy 前，如果 generic proxy 先写了异步操作要读取的位置，需用显式跨 proxy fence 建立可见性；bulk copy 完成后则自带隐式 generic–async proxy fence，但仍须通过对应 completion mechanism 确认“已经完成”。下文的非 bulk `cp.async` 是 generic proxy 中的 weak memory operation，不要把这一段的 async-proxy 规则直接套给它。

#### 9.7.10.28.3 Data Movement and Conversion Instructions: Non-bulk Copy

##### 9.7.10.28.3.1 `cp.async`

###### English Original

> `cp.async`
>
> Initiates an asynchronous copy operation from one state space to another.

**Syntax**

```ptx
cp.async.ca.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}
                         [dst], [src], cp-size{, src-size}{, cache_policy} ;
cp.async.cg.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}
                         [dst], [src], 16{, src-size}{, cache_policy} ;
cp.async.ca.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}
                         [dst], [src], cp-size{, ignore-src}{, cache_policy} ;
cp.async.cg.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}
                         [dst], [src], 16{, ignore-src}{, cache_policy} ;

.level::cache_hint =     { .L2::cache_hint }
.level::prefetch_size =  { .L2::64B, .L2::128B, .L2::256B }
cp-size =                { 4, 8, 16 }
```

**Description**

> `cp.async` is a non-blocking instruction which initiates an asynchronous copy operation of data from the location specified by source address operand `src` to the location specified by destination address operand `dst`. Operand `src` specifies a location in the global state space and `dst` specifies a location in the shared state space.
>
> Operand `cp-size` is an integer constant which specifies the size of data in bytes to be copied to the destination `dst`. `cp-size` can only be 4, 8 and 16.
>
> Instruction `cp.async` allows optionally specifying a 32-bit integer operand `src-size`. Operand `src-size` represents the size of the data in bytes to be copied from `src` to `dst` and must be less than `cp-size`. In such case, remaining bytes in destination `dst` are filled with zeros. Specifying `src-size` larger than `cp-size` results in undefined behavior.
>
> The optional and non-immediate predicate argument `ignore-src` specifies whether the data from the source location `src` should be ignored completely. If the source data is ignored then zeros will be copied to destination `dst`. If the argument `ignore-src` is not specified then it defaults to `False`.
>
> Supported alignment requirements and addressing modes for operand `src` and `dst` are described in [Addresses as Operands](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#addresses-as-operands).
>
> The mandatory `.async` qualifier indicates that the `cp` instruction will initiate the memory copy operation asynchronously and control will return to the executing thread before the copy operation is complete. The executing thread can then use [async-group based completion mechanism](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms-async-group) or the [mbarrier based completion mechanism](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms-mbarrier) to wait for completion of the asynchronous copy operation. No other synchronization mechanism guarantees the completion of the asynchronous copy operations.
>
> There is no ordering guarantee between two `cp.async` operations if they are not explicitly synchronized using `cp.async.wait_all` or `cp.async.wait_group` or [mbarrier instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier).
>
> As described in [Cache Operators](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators), the `.cg` qualifier indicates caching of data only at global level cache L2 and not at L1 whereas `.ca` qualifier indicates caching of data at all levels including L1 cache. Cache operator are treated as performance hints only.
>
> `cp.async` is treated as a weak memory operation performed in the generic proxy in the [Memory Consistency Model](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-consistency-model).
>
> The `.level::prefetch_size` qualifier is a hint to fetch additional data of the specified size into the respective cache level.The sub-qualifier `prefetch_size` can be set to either of `64B`, `128B`, `256B` thereby allowing the prefetch size to be 64 Bytes, 128 Bytes or 256 Bytes respectively.
>
> The qualifier `.level::prefetch_size` may only be used with `.global` state space and with generic addressing where the address points to `.global` state space. If the generic address does not fall within the address window of the global memory, then the prefetching behavior is undefined.
>
> The `.level::prefetch_size` qualifier is treated as a performance hint only.
>
> When the optional argument `cache_policy` is specified, the qualifier `.level::cache_hint` is required. The 64-bit operand `cache_policy` specifies the cache eviction policy that may be used during the memory access.
>
> The qualifier `.level::cache_hint` is only supported for `.global` state space and for generic addressing where the address points to the `.global` state space.
>
> `cache_policy` is a hint to the cache subsystem and may not always be respected. It is treated as a performance hint only, and does not change the memory consistency behavior of the program.

**PTX ISA Notes**

> Introduced in PTX ISA version 7.0.
>
> Support for `.level::cache_hint` and `.level::prefetch_size` qualifiers introduced in PTX ISA version 7.4.
>
> Support for `ignore-src` operand introduced in PTX ISA version 7.5.
>
> Support for sub-qualifier `::cta` introduced in PTX ISA version 7.8.

**Target ISA Notes**

> Requires `sm_80` or higher.
>
> Sub-qualifier `::cta` requires `sm_30` or higher.

**Examples**

```ptx
cp.async.ca.shared.global  [shrd],    [gbl + 4], 4;
cp.async.ca.shared::cta.global  [%r0 + 8], [%r1],     8;
cp.async.cg.shared.global  [%r2],     [%r3],     16;

cp.async.cg.shared.global.L2::64B   [%r2],      [%r3],     16;
cp.async.cg.shared.global.L2::128B  [%r0 + 16], [%r1],     16;
cp.async.cg.shared.global.L2::256B  [%r2 + 32], [%r3],     16;

createpolicy.fractional.L2::evict_last.L2::evict_unchanged.b64 cache_policy, 0.25;
cp.async.ca.shared.global.L2::cache_hint [%r2], [%r1], 4, cache_policy;

cp.async.ca.shared.global                   [shrd], [gbl], 4, p;
cp.async.cg.shared.global.L2::cache_hint   [%r0], [%r2], 16, q, cache_policy;
```

###### 中文翻译

`cp.async`：发起一次从一个 state space 到另一个 state space 的异步复制。这里实际指定的是 `src` 位于 global state space，`dst` 位于 shared state space。上面的 Syntax 和 Examples 保留 PTX 原样，不翻译指令拼写与操作数。

`cp.async` 是 non-blocking instruction：它发起从 `src` 到 `dst` 的数据复制，但执行 thread 无须等复制完成就可以继续运行。`cp-size` 是立即数，表示写入目标的字节数，只能取 4、8 或 16；`.cg` 变体的语法限定为 16 字节。

可以额外指定 32-bit 的 `src-size`，表示实际从源端读取并复制的字节数，且必须小于 `cp-size`；目标端剩余字节补零。若 `src-size` 大于 `cp-size`，行为未定义。也可以指定非立即数 predicate `ignore-src`：为真时完全忽略源数据，在目标端写入零；省略时默认为 `False`。`src`、`dst` 的对齐要求与寻址模式见原文的 Addresses as Operands。

必选的 `.async` qualifier 表示复制尚未完成时控制权便返回执行 thread。该 thread 必须使用 async-group-based 或 mbarrier-based completion mechanism 等待完成，其他同步机制不保证异步复制完成。若没有通过 `cp.async.wait_all`、`cp.async.wait_group` 或 mbarrier 指令显式同步，两次 `cp.async` 之间没有顺序保证。

`.cg` 表示只在 L2 层缓存、不经过 L1；`.ca` 表示包括 L1 在内的所有缓存层都可缓存。这些 cache operator 仅是性能提示。按照 Memory Consistency Model，`cp.async` 是在 generic proxy 中执行的 weak memory operation。

`.level::prefetch_size` 提示在相应缓存层额外预取 64、128 或 256 字节，只能用于 `.global`，或指向 `.global` 的 generic address。若 generic address 不在 global memory 地址窗口内，预取行为未定义；该 qualifier 本身仅是性能提示。

指定可选的 `cache_policy` 时，必须同时带 `.level::cache_hint`。64-bit `cache_policy` 表示内存访问可采用的缓存驱逐策略；`.level::cache_hint` 仅支持 `.global`，或指向 `.global` 的 generic address。`cache_policy` 也是可能不被执行的性能提示，不改变程序的 memory consistency 行为。

**版本与目标架构说明**：该指令在 PTX ISA 7.0 引入；`.level::cache_hint` 和 `.level::prefetch_size` 在 7.4 引入，`ignore-src` 在 7.5 引入，`::cta` 在 7.8 引入。目标 ISA 要求 `sm_80` 或更高；原文另列出 `::cta` sub-qualifier 要求 `sm_30` 或更高，实际使用时仍需满足整条指令的 `sm_80` 要求。

###### 重点解读

`src-size` 与 `ignore-src` 都支持在尾部或整块数据不应从源地址读取时对 shared memory 填零，但语义不同：前者复制有效前缀并补零，后者在 predicate 为真时整块补零。Cache qualifier、预取大小和驱逐策略都不提供完成保证，也不改变内存一致性；正确性仍依赖后续 wait 或 mbarrier。

##### 9.7.10.28.3.2 `cp.async.commit_group`

###### English Original

> `cp.async.commit_group`
>
> Commits all prior initiated but uncommitted `cp.async` instructions into a `cp.async-group`.

**Syntax**

```ptx
cp.async.commit_group ;
```

**Description**

> `cp.async.commit_group` instruction creates a new `cp.async-group` per thread and batches all prior `cp.async` instructions initiated by the executing thread but not committed to any `cp.async-group` into the new `cp.async-group`. If there are no uncommitted `cp.async` instructions then `cp.async.commit_group` results in an empty `cp.async-group`.
>
> An executing thread can wait for the completion of all `cp.async` operations in a `cp.async-group` using `cp.async.wait_group`.
>
> There is no memory ordering guarantee provided between any two `cp.async` operations within the same `cp.async-group`. So two or more `cp.async` operations within a `cp.async-group` copying data to the same location results in undefined behavior.

**PTX ISA Notes**

> Introduced in PTX ISA version 7.0.

**Target ISA Notes**

> Requires `sm_80` or higher.

**Examples**

```ptx
// Example 1:
cp.async.ca.shared.global [shrd], [gbl], 4;
cp.async.commit_group ; // Marks the end of a cp.async group

// Example 2:
cp.async.ca.shared.global [shrd1],   [gbl1],   8;
cp.async.ca.shared.global [shrd1+8], [gbl1+8], 8;
cp.async.commit_group ; // Marks the end of cp.async group 1

cp.async.ca.shared.global [shrd2],    [gbl2],    16;
cp.async.cg.shared.global [shrd2+16], [gbl2+16], 16;
cp.async.commit_group ; // Marks the end of cp.async group 2
```

###### 中文翻译

`cp.async.commit_group` 把此前已经发起、但尚未 commit 的所有 `cp.async` 指令归入一个 `cp.async-group`。该指令按 thread 创建新 group，只收集当前执行 thread 发起的未归组操作；如果没有这类操作，也会创建一个空 group。执行 thread 可以用 `cp.async.wait_group` 等待一个 group 中所有 `cp.async` 完成。

同一 `cp.async-group` 内的不同 `cp.async` 之间没有 memory ordering 保证。因此，如果同一个 group 内两个或更多 `cp.async` 向同一位置复制数据，行为未定义。该指令在 PTX ISA 7.0 引入，目标 ISA 要求 `sm_80` 或更高。上方示例一展示单次复制后 commit；示例二把两批复制分别归入 group 1 和 group 2。

###### 重点解读

`commit_group` 是“划定待等待的一批操作”，不是“等待这一批完成”。一个 group 可以包含多次独立复制，但它们的目标区间不应互相重叠；若操作之间存在顺序依赖，需要通过完成等待把它们分到适当阶段，而不是指望 group 内自然按指令顺序执行。

##### 9.7.10.28.3.3 `cp.async.wait_group`, `cp.async.wait_all`

###### English Original

> `cp.async.wait_group`, `cp.async.wait_all`
>
> Wait for completion of prior asynchronous copy operations.

**Syntax**

```ptx
cp.async.wait_group N;
cp.async.wait_all ;
```

**Description**

> `cp.async.wait_group` instruction will cause executing thread to wait till only `N` or fewer of the most recent `cp.async-groups` are pending and all the prior `cp.async-groups` committed by the executing threads are complete. For example, when `N` is 0, the executing thread waits on all the prior `cp.async-groups` to complete. Operand `N` is an integer constant.
>
> `cp.async.wait_all` is equivalent to:

```ptx
cp.async.commit_group;
cp.async.wait_group 0;
```

> An empty `cp.async-group` is considered to be trivially complete.
>
> Writes performed by `cp.async` operations are made visible to the executing thread only after:
>
> 1. The completion of `cp.async.wait_all` or
> 2. The completion of `cp.async.wait_group` on the `cp.async-group` in which the `cp.async` belongs to or
> 3. [mbarrier.test_wait](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait) and [mbarrier.try_wait](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait) returns `True` on an `mbarrier object` which is tracking the completion of the `cp.async` operation.
>
> There is no ordering between two `cp.async` operations that are not synchronized with `cp.async.wait_all` or `cp.async.wait_group` or [mbarrier objects](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier).
>
> `cp.async.wait_group` and `cp.async.wait_all` does not provide any ordering and visibility guarantees for any other memory operation apart from `cp.async`.

**PTX ISA Notes**

> Introduced in PTX ISA version 7.0.

**Target ISA Notes**

> Requires `sm_80` or higher.

**Examples**

```ptx
// Example of .wait_all:
cp.async.ca.shared.global [shrd1], [gbl1], 4;
cp.async.cg.shared.global [shrd2], [gbl2], 16;
cp.async.wait_all;  // waits for all prior cp.async to complete

// Example of .wait_group :
cp.async.ca.shared.global [shrd3], [gbl3], 8;
cp.async.commit_group;  // End of group 1

cp.async.cg.shared.global [shrd4], [gbl4], 16;
cp.async.commit_group;  // End of group 2

cp.async.cg.shared.global [shrd5], [gbl5], 16;
cp.async.commit_group;  // End of group 3

cp.async.wait_group 1;  // waits for group 1 and group 2 to complete
```

###### 中文翻译

`cp.async.wait_group` 与 `cp.async.wait_all` 用于等待此前的异步复制完成。`cp.async.wait_group N` 使执行 thread 等到至多还有最近的 `N` 个 `cp.async-group` 未完成，而更早 commit 的 group 全部完成；`N` 必须是整数立即数。`N = 0` 时等待此前所有 group 完成。

`cp.async.wait_all` 等价于先执行 `cp.async.commit_group`，再执行 `cp.async.wait_group 0`。空 group 视为立即完成。

`cp.async` 写入的内容，只有在 `cp.async.wait_all` 完成、对所属 group 的 `cp.async.wait_group` 完成，或跟踪该操作的 mbarrier object 上 `mbarrier.test_wait` / `mbarrier.try_wait` 返回 `True` 后，才对执行 thread 可见。没有借助这些机制同步的两个 `cp.async` 操作之间，不存在顺序保证。

`cp.async.wait_group` 和 `cp.async.wait_all` 只对 `cp.async` 提供完成、排序与可见性保证；它们不为其他内存操作提供这些保证。这两条指令在 PTX ISA 7.0 引入，目标 ISA 要求 `sm_80` 或更高。上方示例中，`wait_all` 等待此前的两次复制；`wait_group 1` 等待 group 1 和 group 2 完成，但允许最新的 group 3 仍未完成。

###### 重点解读

`wait_group N` 的 `N` 是“允许仍在途的最近 group 数”，不是要等待的 group 编号。`wait_group 1` 可用于流水线：消费较早批次，同时保留最新一批继续搬运。若下一步要读取最新批次的 shared-memory 结果，还需等待该批次完成。`wait_all` 更直接，却通常减少复制与计算重叠的空间。

#### 9.7.10.28.4 Data Movement and Conversion Instructions: Bulk Copy

##### 9.7.10.28.4.1 Data Movement and Conversion Instructions: `cp.async.bulk`

###### English Original

cp.async.bulk

Initiates an asynchronous copy operation from one state space to another.

**Syntax**

```ptx
// global -> shared::cta
cp.async.bulk{.sem}.dst.src.completion_mechanism{.level::cache_hint}{.ignore_oob}
                      [dstMem], [srcMem], size{, ignoreBytesLeft, ignoreBytesRight}, [mbar] {, cache_policy}

.sem =                  { .weak }
.dst =                  { .shared::cta }
.src =                  { .global }
.completion_mechanism = { .mbarrier::complete_tx::bytes{.report_mechanism} }
.level::cache_hint =    { .L2::cache_hint }
.report_mechanism =     { .mbarrier::report::disabled,
                          .mbarrier::report::validity::per_16bytes::80000000,
                          .mbarrier::report::validity::per_16bytes::8000,
                          .mbarrier::report::validity::per_16bytes::80,
                          .mbarrier::report::validity::per_16bytes::8,
                          .mbarrier::report::validity::per_element::ff
                        }

cp.async.bulk.sem.scope.dst.src.completion_mechanism{.level::cache_hint}{.ignore_oob}.type
                      [dstMem], [srcMem], size{, ignoreBytesLeft, ignoreBytesRight}, [mbar] {, cache_policy};

.sem =                  { .relaxed }
.scope =                { .cta, .cluster, .gpu, .sys }
.dst =                  { .shared::cta }
.src =                  { .global }
.completion_mechanism = { .mbarrier::complete_tx::bytes{.report_mechanism} }
.level::cache_hint =    { .L2::cache_hint }
.type =                 { .b128 }
.report_mechanism =     { .mbarrier::report::disabled,
                          .mbarrier::report::validity::per_16bytes::80000000,
                          .mbarrier::report::validity::per_16bytes::8000,
                          .mbarrier::report::validity::per_16bytes::80,
                          .mbarrier::report::validity::per_16bytes::8,
                          .mbarrier::report::validity::per_element::ff
                        }


// global -> shared::cluster
cp.async.bulk{.sem}.dst.src.completion_mechanism{.multicast}{.level::cache_hint}
                      [dstMem], [srcMem], size, [mbar] {, ctaMask} {, cache_policy}

.sem =                  { .weak }
.dst =                  { .shared::cluster }
.src =                  { .global }
.completion_mechanism = { .mbarrier::complete_tx::bytes{.report_mechanism} }
.level::cache_hint =    { .L2::cache_hint }
.multicast =            { .multicast::cluster{::16b, ::32b} }
.report_mechanism =     { .mbarrier::report::disabled,
                          .mbarrier::report::validity::per_16bytes::80000000,
                          .mbarrier::report::validity::per_16bytes::8000,
                          .mbarrier::report::validity::per_16bytes::80,
                          .mbarrier::report::validity::per_16bytes::8,
                          .mbarrier::report::validity::per_element::ff
                        }

cp.async.bulk.sem.scope.dst.src.completion_mechanism{.multicast}{.level::cache_hint}.type
                      [dstMem], [srcMem], size, [mbar] {, ctaMask} {, cache_policy};

.sem =                  { .relaxed }
.scope =                { .cta, .cluster, .gpu, .sys }
.dst =                  { .shared::cluster }
.src =                  { .global }
.completion_mechanism = { .mbarrier::complete_tx::bytes{.report_mechanism} }
.level::cache_hint =    { .L2::cache_hint }
.type =                 { .b128 }
.multicast =            { .multicast::cluster{::16b, ::32b} }
.report_mechanism =     { .mbarrier::report::disabled,
                          .mbarrier::report::validity::per_16bytes::80000000,
                          .mbarrier::report::validity::per_16bytes::8000,
                          .mbarrier::report::validity::per_16bytes::80,
                          .mbarrier::report::validity::per_16bytes::8,
                          .mbarrier::report::validity::per_element::ff
                        }


// shared::cta -> shared::cluster
cp.async.bulk{.sem}.dst.src.completion_mechanism [dstMem], [srcMem], size, [mbar]

.sem =                  { .weak }
.dst =                  { .shared::cluster }
.src =                  { .shared::cta }
.completion_mechanism = { .mbarrier::complete_tx::bytes }

cp.async.bulk.sem.scope.dst.src.completion_mechanism.type [dstMem], [srcMem], size, [mbar];

.sem =                  { .relaxed }
.scope =                { .cta, .cluster }
.dst =                  { .shared::cluster }
.src =                  { .shared::cta }
.completion_mechanism = { .mbarrier::complete_tx::bytes }
.type =                 { .b128 }


// shared::cta -> global
cp.async.bulk{.sem}.dst.src.completion_mechanism{.level::cache_hint}{.cp_mask}
                      [dstMem], [srcMem], size {, cache_policy} {, byteMask}

.sem =                  { .weak }
.dst =                  { .global }
.src =                  { .shared::cta }
.completion_mechanism = { .bulk_group }
.level::cache_hint =    { .L2::cache_hint }

cp.async.bulk.sem.scope.dst.src.completion_mechanism{.level::cache_hint}{.cp_mask}.type
                      [dstMem], [srcMem], size {, cache_policy} {, byteMask};

.sem =                  { .relaxed }
.scope =                { .cta, .cluster, .gpu, .sys }
.dst =                  { .global }
.src =                  { .shared::cta }
.completion_mechanism = { .bulk_group }
.level::cache_hint =    { .L2::cache_hint }
.type =                 { .b128 }
```

**Description**

cp.async.bulk is a non-blocking instruction which initiates an asynchronous bulk-copy operation from the location specified by source address operand srcMem to the location specified by destination address operand dstMem.

The direction of bulk-copy is from the state space specified by the .src modifier to the state space specified by the .dst modifiers.

The 32-bit operand size specifies the amount of memory to be copied, in terms of number of bytes. size must be a multiple of 16. If the value is not a multiple of 16, then the behavior is undefined. The memory range [dstMem, dstMem + size - 1] must not overflow the destination memory space and the memory range [srcMem, srcMem + size - 1] must not overflow the source memory space. Otherwise, the behavior is undefined. The addresses dstMem and srcMem must be aligned to 16 bytes.

The optional qualifier .ignore_oob specifies that up to 15 bytes at the beginning or ending of [srcMem .. srcMem+size) may be out-of-bounds of a global memory allocation, and the value of the corresponding bytes in destination shared memory [dstMem .. dstMem+size) is indeterminate. The 32-bit operands ignoreBytesLeft and ignoreBytesRight are used to specify the bytes from beginning and ending of the copy-chunk specified by size that may go out of bounds. The only valid values for ignoreBytesLeft and ignoreBytesRight are [0..15], and any other value may result in undefined behavior. The srcMem and dstMem addresses must be aligned to 16 bytes, and the size operand must be a multiple of 16 even with .ignore_oob qualifier. The qualifier .ignore_oob is only available for the global to .shared::cta copy direction. In presence of .ignore_oob, the qualifier .report_mechanism, if specified, must be set to .mbarrier::report::disabled.

When the destination of the copy is .shared::cta the destination address has to be in the shared memory of the executing CTA within the cluster, otherwise the behavior is undefined.

When the source of the copy is .shared::cta and the destination is .shared::cluster, the destination has to be in the shared memory of a different CTA within the cluster.

The modifier .completion_mechanism specifies the completion mechanism that is supported on the instruction variant. The completion mechanisms that are supported for different variants are summarized in the following table:

| `.completion-mechanism` | `.dst` | `.src` | Completion mechanism |
|---|---|---|---|
| `.mbarrier::...` | `.shared::cta` | `.global` | mbarrier based |
| `.mbarrier::...` | `.shared::cluster` | `.global` | mbarrier based |
| `.mbarrier::...` | `.shared::cluster` | `.shared::cta` | mbarrier based |
| `.bulk_group` | `.global` | `.shared::cta` | Bulk async-group based |


The modifier .mbarrier::complete_tx::bytes specifies that the cp.async.bulk variant uses mbarrier based completion mechanism. The complete-tx operation, with completeCount argument equal to amount of data copied in bytes, will be performed on the mbarrier object specified by the operand mbar. This instruction accesses its mbarrier operand using generic-proxy.

The modifier .bulk_group specifies that the cp.async.bulk variant uses bulk async-group based completion mechanism.

The optional qualifier .multicast::cluster allows copying of data from global memory to shared memory of multiple CTAs in the cluster. Operand ctaMask specifies the destination CTAs in the cluster such that each bit position in the 16-bit or otherwise 32-bit ctaMask operand corresponds to the %cluster_ctarank of the destination CTA. The additional sub-qualifier ::16b or ::32b can be specified to correspond to the width of the ctaMask argument. By default, ctaMask is assumed to be of 16-bit width. The source data is multicast to the same CTA-relative offset as dstMem in the shared memory of each destination CTA. The mbarrier signal is also multicast to the same CTA-relative offset as mbar in the shared memory of the destination CTA.

When the optional argument cache_policy is specified, the qualifier .level::cache_hint is required. The 64-bit operand cache_policy specifies the cache eviction policy that may be used during the memory access.

cache_policy is a hint to the cache subsystem and may not always be respected. It is treated as a performance hint only, and does not change the memory consistency behavior of the program. The qualifier .level::cache_hint is only supported when at least one of the .src or .dst statespaces is .global state space.

The optional qualifier .report_mechanism specifies the mbarrier reporting mechanism as outlined in the Report Mechanisms for Asynchronous Copy Operations.

When the optional qualifier .cp_mask is specified, the argument byteMask is required. The i-th bit in the 16-bit wide byteMask operand specifies whether the i-th byte of each 16-byte wide chunk of source data is copied to the destination. If the bit is set, the byte is copied.

The optional .sem modifier specifies the memory ordering semantics as described in the Memory Consistency Model.

When .sem is not specified, it defaults to .weak.

When .sem is .weak, the data copy operation accesses memory with weak memory operations.

When .sem is .relaxed, the data copy accesses memory with naturally-aligned strong memory operations with element-wise atomic size specified by .type and thread scope specified by .scope.

The complete-tx operation on the mbarrier has .release semantics at the .cluster scope as described in the Memory Consistency Model.

**Notes**

The copy operation with shared::cluster.global is optimized for target architecture sm_90a/sm_100f/sm_100a/ sm_103f/sm_103a/sm_107a/sm_107f/sm_110f/sm_110a and may have substantially reduced performance on other targets and hence such operation is advised to be used with .target sm_90a/sm_100f/sm_100a/sm_103f/sm_103a/sm_107a/sm_107f/sm_110f/sm_110a.

**PTX ISA Notes**

Introduced in PTX ISA version 8.0.

Support for .shared::cta as destination state space is introduced in PTX ISA version 8.6.

Support for .cp_mask qualifier introduced in PTX ISA version 8.6.

Support for .ignore_oob qualifier introduced in PTX ISA version 9.2.

Support for .weak and .relaxed semantics, .scope and .type qualifiers are introduced in PTX ISA version 9.3.

Support for .multicast::cluster::16b and .multicast::cluster::32b qualifiers introduced in PTX ISA version 9.4.

Support for .report_mechanism qualifier introduced in PTX ISA version 9.4.

**Target ISA Notes**

Requires sm_90 or higher.

.multicast::cluster{::16b} qualifier advised to be used with .target sm_90a or sm_100f or sm_100a or sm_103f or sm_103a or sm_107a or sm_107f or sm_110f or sm_110a.

Support for .cp_mask qualifier requires sm_100 or higher.

Qualifiers .weak, .relaxed, .scope and .type supported on following architectures:

sm_90a

sm_100f or higher in the same family

sm_110f or higher in the same family

Qualifier .multicast::cluster::32b is supported on following family-specific architectures:

sm_107f or higher in the same family

Qualifier .report_mechanism is supported on following family-specific architectures:

sm_107f or higher in the same family

**Examples**

```ptx
// .global -> .shared::cta (strictly non-remote):
cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [dstMem], [srcMem], size, [mbar];

cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint
                                             [dstMem], [srcMem], size, [mbar], cache_policy;

// .global -> .shared::cluster:
cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [dstMem], [srcMem], size, [mbar];

cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster
                                             [dstMem], [srcMem], size, [mbar], ctaMask;

cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint
                                             [dstMem], [srcMem], size, [mbar], cache_policy;

cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster::32b
                                             [dstMem], [srcMem], size, [mbar], ctaMask32;

// Validity check using 0x8000 pattern on every 16B
cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.mbarrier::report::validity::per_16bytes::8000
                                             [dstMem], [srcMem], size, [mbar];

// .shared::cta -> .shared::cluster (strictly remote):
cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [dstMem], [srcMem], size, [mbar];

// .shared::cta -> .global:
cp.async.bulk.global.shared::cta.bulk_group [dstMem], [srcMem], size;

cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint} [dstMem], [srcMem], size, cache_policy;

// .shared::cta -> .global with .cp_mask:
cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint.cp_mask [dstMem], [srcMem], size, cache_policy, byteMask;

// ignore_oob
cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.ignore_oob [dstMem], [srcMem], size, ignoreBytesLeft, ignoreBytesRight, [mbar];

// .global -> .shared::cta with .relaxed scope and .b128 type
cp.async.bulk.relaxed.shared::cta.global.mbarrier::complete_tx::bytes.b128 [dstMem], [srcMem], size, [mbar];

cp.async.bulk.relaxed.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint.b128
                                             [dstMem], [srcMem], size, [mbar], cache_policy;
```

###### 中文翻译

`cp.async.bulk` 是 non-blocking bulk copy 指令，按 `.src` 和 `.dst` 指定的 state space，从 `srcMem` 向 `dstMem` 异步复制数据。语法列出四种方向：`.global → .shared::cta`、`.global → .shared::cluster`、`.shared::cta → .shared::cluster`、`.shared::cta → .global`；前面三种使用 mbarrier-based completion，最后一种使用 bulk async-group。

32-bit `size` 以字节为单位，必须是 16 的倍数；`srcMem`、`dstMem` 都须 16-byte 对齐，源与目标区间均不得超出各自 memory space，否则行为未定义。`.shared::cta` 作为目标时，地址必须属于当前执行 CTA 的 shared memory；从 `.shared::cta` 复制到 `.shared::cluster` 时，目标必须属于 cluster 内另一 CTA。

仅在 `.global → .shared::cta` 方向可用的 `.ignore_oob`，允许源区间的开头或结尾最多各有 15 字节超出 global allocation。`ignoreBytesLeft` 与 `ignoreBytesRight` 分别指定两端可越界的字节数，合法范围均为 `[0..15]`；相应目标字节的值不确定，不是补零。该 qualifier 不放宽 16-byte 对齐和 `size` 的倍数要求。启用时，若写出 `.report_mechanism`，它必须是 `.mbarrier::report::disabled`。

完成机制由指令变体决定。`.mbarrier::complete_tx::bytes` 在 `mbar` 指定的 barrier 上执行 complete-tx，`completeCount` 等于复制字节数；该指令以 generic proxy 访问 mbarrier operand。`.bulk_group` 则使用 bulk async-group completion。原文表格列出各复制方向对应的机制，表格保持英文。

可选 `.multicast::cluster` 把 global 数据复制到 cluster 内多个 CTA 的 shared memory。`ctaMask` 的各 bit 对应目标 CTA 的 `%cluster_ctarank`；可用 `::16b` 或 `::32b` 指定 mask 宽度，默认 16 bit。各目标 CTA 接收的数据与 mbarrier signal，分别落在相对于本 CTA 与 `dstMem`、`mbar` 相同的偏移处。

传入 64-bit `cache_policy` 时必须带 `.level::cache_hint`；该 hint 只适用于源或目标至少有一侧是 `.global` 的变体。缓存驱逐策略仅为性能提示，可能不被采纳，也不改变 memory consistency。可选 `.report_mechanism` 指定前文介绍的 mbarrier 数据有效性报告机制。

带 `.cp_mask` 时必须提供 16-bit `byteMask`；其第 `i` 位决定每个 16-byte 源数据块的第 `i` 字节是否复制。`.sem` 省略时默认 `.weak`，以 weak memory operations 访问数据；`.relaxed` 则按 `.type` 指定的 element-wise atomic size 与 `.scope` 指定的 thread scope，使用 naturally-aligned strong memory operations。对 mbarrier 的 complete-tx 在 `.cluster` scope 具有 `.release` 语义。

原文 Notes 提醒：`.shared::cluster.global` 方向针对列出的 architecture-specific / family-specific targets 优化，其他目标上性能可能显著下降。目标 ISA 至少要求 `sm_90`；`.cp_mask` 至少需要 `sm_100`，`.multicast::cluster::32b` 与 `.report_mechanism` 还受文中列出的 family-specific architecture 限制。英文原文保留其完整架构清单与 PTX ISA 引入版本。

###### 重点解读

四种方向不能套用同一种 wait：前三种用 mbarrier，`.shared::cta → .global` 用 bulk group。`size` 与两端地址的 16-byte 约束是正确性要求；`.ignore_oob` 只允许指定的边界字节越界，目标对应内容不确定，不可当作零填充。`.multicast::cluster` 还要求正确配置目标 CTA 的 barrier 与 mask。官方示例中的 `L2::cache_hint}` 多出一个 `}`；这里按原文保留，复制到代码前应核对语法。

##### 9.7.10.28.4.2 Data Movement and Conversion Instructions: `cp.reduce.async.bulk`

###### English Original

cp.reduce.async.bulk

Initiates an asynchronous reduction operation.

**Syntax**

```ptx
cp.reduce.async.bulk{.sem.scope}.dst.src.completion_mechanism.redOp.type
              [dstMem], [srcMem], size, [mbar]

.sem =                  { .relaxed }
.scope =                { .cta, .cluster }
.dst =                  { .shared::cluster }
.src =                  { .shared::cta }
.completion_mechanism = { .mbarrier::complete_tx::bytes }
.redOp=                 { .and, .or, .xor,
                          .add, .inc, .dec,
                          .min, .max }
.type =                 { .b32, .u32, .s32, .b64, .u64 }


cp.reduce.async.bulk{.sem.scope}.dst.src.completion_mechanism{.level::cache_hint}.redOp.type
               [dstMem], [srcMem], size{, cache_policy}

.sem =                  { .relaxed }
.scope =                { .cta, .cluster, .gpu, .sys }
.dst =                  { .global      }
.src =                  { .shared::cta }
.completion_mechanism = { .bulk_group }
.level::cache_hint    = { .L2::cache_hint }
.redOp=                 { .and, .or, .xor,
                          .add, .inc, .dec,
                          .min, .max }
.type =                 { .f16, .bf16, .b32, .u32, .s32, .b64, .u64, .s64, .f32, .f64 }


cp.reduce.async.bulk{.sem.scope}.dst.src.completion_mechanism{.level::cache_hint}.add.noftz.type
               [dstMem], [srcMem], size{, cache_policy}

.sem =                  { .relaxed }
.scope =                { .cta, .cluster, .gpu, .sys }
.dst  =                 { .global }
.src  =                 { .shared::cta }
.completion_mechanism = { .bulk_group }
.type =                 { .f16, .bf16, .f32 }
```

**Description**

cp.reduce.async.bulk is a non-blocking instruction which initiates an asynchronous reduction operation on an array of memory locations specified by the destination address operand dstMem with the source array whose location is specified by the source address operand srcMem. The size of the source and the destination array must be the same and is specified by the operand size.

Each data element in the destination array is reduced inline with the corresponding data element in the source array with the reduction operation specified by the modifier .redOp. The type of each data element in the source and the destination array is specified by the modifier .type.

The source address operand srcMem is located in the state space specified by .src and the destination address operand dstMem is located in the state specified by the .dst.

The 32-bit operand size specifies the amount of memory to be copied from the source location and used in the reduction operation, in terms of number of bytes. size must be a multiple of 16. If the value is not a multiple of 16, then the behavior is undefined. The memory range [dstMem, dstMem + size - 1] must not overflow the destination memory space and the memory range [srcMem, srcMem + size - 1] must not overflow the source memory space. Otherwise, the behavior is undefined. The addresses dstMem and srcMem must be aligned to 16 bytes.

The operations supported by .redOp are classified as follows:

The bit-size operations are .and, .or, and .xor.

The integer operations are .add, .inc, .dec, .min, and .max. The .inc and .dec operations return a result in the range [0..x] where x is the value at the source state space.

The floating point operation .add rounds to the nearest even. The cp.reduce.async.bulk.add.f16 and cp.reduce.async.bulk.add.bf16 operations require .noftz qualifier. It preserves input and result subnormals, and does not flush them to zero. The default behavior for cp.reduce.async.bulk.add.f32 is equivalent to specifying optional .noftz qualifier which preserves input and result subnormals, and does not flush them to zero.

The following table describes the valid combinations of .redOp and element type:

| `.dst` | `.redOp` | Element type |
|---|---|---|
| `.shared::cluster` | `.add` | `.u32`, `.s32`, `.u64` |
| `.shared::cluster` | `.min`, `.max` | `.u32`, `.s32` |
| `.shared::cluster` | `.inc`, `.dec` | `.u32` |
| `.shared::cluster` | `.and`, `.or`, `.xor` | `.b32` |
| `.global` | `.add` | `.u32`, `.s32`, `.u64`, `.f32`, `.f64`, `.f16`, `.bf16` |
| `.global` | `.min`, `.max` | `.u32`, `.s32`, `.u64`, `.s64`, `.f16`, `.bf16` |
| `.global` | `.inc`, `.dec` | `.u32` |
| `.global` | `.and`, `.or`, `.xor` | `.b32`, `.b64` |


The modifier .completion_mechanism specifies the completion mechanism that is supported on the instruction variant. The completion mechanisms that are supported for different variants are summarized in the following table:

| `.completion-mechanism` | `.dst` | `.src` | Completion mechanism |
|---|---|---|---|
| `.mbarrier::...` | `.shared::cluster` | `.global` | mbarrier based |
| `.mbarrier::...` | `.shared::cluster` | `.shared::cta` | mbarrier based |
| `.bulk_group` | `.global` | `.shared::cta` | Bulk async-group based |


The modifier .mbarrier::complete_tx::bytes specifies that the cp.reduce.async.bulk variant uses mbarrier based completion mechanism. The complete-tx operation, with completeCount argument equal to amount of data copied in bytes, will be performed on the mbarrier object specified by the operand mbar. This instruction accesses its mbarrier operand using generic-proxy.

The modifier .bulk_group specifies that the cp.reduce.async.bulk variant uses bulk async-group based completion mechanism.

When the optional argument cache_policy is specified, the qualifier .level::cache_hint is required. The 64-bit operand cache_policy specifies the cache eviction policy that may be used during the memory access.

cache_policy is a hint to the cache subsystem and may not always be respected. It is treated as a performance hint only, and does not change the memory consistency behavior of the program. The qualifier .level::cache_hint is only supported when at least one of the .src or .dst statespaces is .global state space.

The optional .sem.scope qualifier specifies the memory ordering semantics of each individual element-wise reduction in dstMem performed by this operation as described in the Memory Consistency Model. These are element-wise atomic at the specified scope. Source location accesses are .weak, that is, not atomic. When .sem.scope is not specified, it defaults to .relaxed.sys.

The load operations from srcMem in cp.reduce.async.bulk are treated as weak memory operation and the complete-tx operation on the mbarrier has .release semantics at the .cluster scope as described in the Memory Consistency Model. The memory operations are performed in the async proxy.

**PTX ISA Notes**

Introduced in PTX ISA version 8.0.

Support for .sem.scope qualifier is introduced in PTX ISA version 9.3.

Qualifier .noftz with .f32 type introduced in PTX ISA version 9.4.

**Target ISA Notes**

Requires sm_90 or higher.

**Examples**

```ptx
cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.add.u64
                                                                  [dstMem], [srcMem], size, [mbar];

cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.min.s32
                                                                  [dstMem], [srcMem], size, [mbar];

cp.reduce.async.bulk.global.shared::cta.bulk_group.min.f16 [dstMem], [srcMem], size;

cp.reduce.async.bulk.global.shared::cta.bulk_group.L2::cache_hint.xor.s32 [dstMem], [srcMem], size, policy;

cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f16 [dstMem], [srcMem], size;

cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f32 [dstMem], [srcMem], size;

cp.reduce.async.bulk.relaxed.cta.global.bulk_group.add.u64 [dstMem], [srcMem], size, [mbar];
```

###### 中文翻译

`cp.reduce.async.bulk` 不只是搬运数据，而是在目标数组的每个元素上，以对应的源元素执行 `.redOp` 指定的异步归约；两数组大小相同，元素类型由 `.type` 指定，源和目标 state space 分别由 `.src`、`.dst` 指定。`size` 是参与归约的源数据字节数，必须为 16 的倍数；两端地址须 16-byte 对齐，访问区间不得溢出各自 memory space，否则行为未定义。

可选运算包括 bitwise 的 `.and`、`.or`、`.xor`，整数的 `.add`、`.inc`、`.dec`、`.min`、`.max`。`.inc` 与 `.dec` 的结果位于 `[0..x]`，其中 `x` 是源元素值。浮点 `.add` 采用 nearest-even 舍入；`.f16`、`.bf16` 的 add 需要 `.noftz`，保留输入与结果的 subnormal；`.f32` 默认行为等价于指定可选 `.noftz`。具体 `.dst`、`.redOp` 和元素类型的合法组合见保留的英文表格。

完成机制由方向决定：使用 `.mbarrier::complete_tx::bytes` 的变体在 `mbar` 上按复制字节数执行 complete-tx，并以 generic proxy 访问 mbarrier operand；`.bulk_group` 变体则由 bulk async-group 跟踪。各方向对应关系保留在原文表格。

若提供 64-bit `cache_policy`，必须同时指定 `.level::cache_hint`；该选项仅用于至少一侧为 `.global` 的方向，且只是可能不被采纳的性能提示，不改变一致性语义。

可选 `.sem.scope` 规定目标端每个 element-wise reduction 的 memory ordering 与原子作用域；它们在所设 scope 下逐元素 atomic。源端读取始终是 `.weak`、非 atomic。省略 `.sem.scope` 时默认为 `.relaxed.sys`。`srcMem` 的 load 属于 weak memory operation；mbarrier 上的 complete-tx 在 `.cluster` scope 具有 `.release` 语义；数据内存操作发生在 async proxy。目标 ISA 至少要求 `sm_90`。

###### 重点解读

归约的目标端逐元素 atomic，并不意味着源端 load 也 atomic；`.sem.scope` 主要界定目标归约的强度和可见范围。区分 `.mbarrier` 与 `.bulk_group` 变体，尤其不要用非 bulk 的 `cp.async.wait_group` 等待这里的 bulk group。原文的 completion 表还列出 `.global → .shared::cluster`，但本节 Syntax 未列对应形式；最后一个 `.relaxed` 示例带 `[mbar]`，也与相应 `.bulk_group` 语法所列操作数不一致。这两处按原文保留，实际编写指令前需核对汇编器和目标版本。

##### 9.7.10.28.4.3 Data Movement and Conversion Instructions: `cp.async.bulk.prefetch`

###### English Original

cp.async.bulk.prefetch

Provides a hint to the system to initiate the asynchronous prefetch of data to the cache.

**Syntax**

```ptx
cp.async.bulk.prefetch.L2.src{.level::cache_hint}         [srcMem], size {, cache_policy};
cp.async.bulk.prefetch.L2.src{.level::eviction_priority}  [srcMem], size;

.src =                { .global }
.level::cache_hint =  { .L2::cache_hint }
.level::eviction_priority = { .L2::evict_last }
```

**Description**

cp.async.bulk.prefetch is a non-blocking instruction which may initiate an asynchronous prefetch of data from the location specified by source address operand srcMem, in .src statespace, to the L2 cache.

The 32-bit operand size specifies the amount of memory to be prefetched in terms of number of bytes. size must be a multiple of 16. If the value is not a multiple of 16, then the behavior is undefined. The address srcMem must be aligned to 16 bytes.

When the optional argument cache_policy is specified, the qualifier .level::cache_hint is required. The 64-bit operand cache_policy specifies the cache eviction policy that may be used during the memory access.

cache_policy is a hint to the cache subsystem and may not always be respected. It is treated as a performance hint only, and does not change the memory consistency behavior of the program.

cp.async.bulk.prefetch is treated as a weak memory operation in the Memory Consistency Model.

Qualifier .level::eviction_priority specifies the eviction priority to be applied for the prefetched cache line.

**PTX ISA Notes**

Introduced in PTX ISA version 8.0.

Support for .level::eviction_priority qualifier introduced in PTX ISA version 9.4.

**Target ISA Notes**

Requires sm_90 or higher.

Qualifier .level::eviction_priority is supported on the following family-specific architectures:

sm_107f or higher in the same family

**Examples**

```ptx
cp.async.bulk.prefetch.L2.global                 [srcMem], size;

cp.async.bulk.prefetch.L2.global.L2::cache_hint  [srcMem], size, policy;

cp.async.bulk.prefetch.L2.global.L2::evict_last  [srcMem], size;
```

###### 中文翻译

`cp.async.bulk.prefetch` 是非阻塞预取提示：系统可以把 `.global` 中 `srcMem` 指向的数据异步预取到 L2，但不保证一定执行。32-bit `size` 表示预取字节数，必须是 16 的倍数；`srcMem` 必须 16-byte 对齐，违反尺寸要求会产生未定义行为。

使用 64-bit `cache_policy` 时必须配 `.level::cache_hint`。该缓存驱逐策略可能不被采纳，仅是性能提示，不改变 memory consistency；`cp.async.bulk.prefetch` 在一致性模型中被视为 weak memory operation。`.level::eviction_priority` 用于指定预取 cache line 的驱逐优先级。

目标 ISA 至少要求 `sm_90`；`.level::eviction_priority` 仅在原文列出的 `sm_107f` 同族及更高目标上受支持。英文部分保留了具体 PTX ISA 引入版本与全部指令示例。

###### 重点解读

Prefetch 是“可能执行”的 cache hint，并非真实数据搬运完成的凭据。它适合提前提示 L2 访问模式，但不能替代 load/copy，也不能用作同步。

##### 9.7.10.28.4.4 Data Movement and Conversion Instructions: `multimem.cp.async.bulk`

###### English Original

multimem.cp.async.bulk

Initiates an asynchronous copy operation to a multimem address range.

**Syntax**

```ptx
multimem.cp.async.bulk{.sem}.dst.src.completion_mechanism{.cp_mask}
    [dstMem], [srcMem], size{, byteMask};

    .sem                  = { .weak }
    .dst                  = { .global }
    .src                  = { .shared::cta }
    .completion_mechanism = { .bulk_group }

multimem.cp.async.bulk.sem.scope.dst.src.completion_mechanism{.cp_mask}.type
    [dstMem], [srcMem], size{, byteMask};

    .sem                  = { .relaxed }
    .scope                = { .cta, .cluster, .gpu, .sys }
    .dst                  = { .global }
    .src                  = { .shared::cta }
    .completion_mechanism = { .bulk_group }
    .type                 = { .b128 }
```

**Description**

Instruction multimem.cp.async.bulk initiates an asynchronous bulk-copy operation from source address range [srcMem, srcMem + size) to memory locations residing on each GPU’s memory referred to by the destination multimem address range [dstMem, dstMem + size). The direction of bulk-copy is from the state space specified by the .src modifier to the state space specified by the .dst modifiers.

The 32-bit operand size specifies the amount of memory to be copied, in terms of number of bytes. Operand size must be a multiple of 16. The memory range [dstMem, dstMem + size) must not overflow the destination multimem memory space. The memory range [srcMem, srcMem + size) must not overflow the source memory space. The addresses dstMem and srcMem must be aligned to 16 bytes. If any of these pre-conditions is not met, the behavior is undefined.

The modifier .completion_mechanism specifies the completion mechanism that is supported by the instruction. The modifier .bulk_group specifies that the multimem.cp.async.bulk instruction uses bulk async-group based completion mechanism.

When the optional modifier .cp_mask is specified, the argument byteMask is required. The i-th bit in the 16-bit wide byteMask operand specifies whether the i-th byte of each 16-byte wide chunk of source data is copied to the destination. If the bit is set, the byte is copied.

The optional .sem qualifier specifies the memory ordering semantics as described in the Memory Consistency Model.

When .sem is not specified, it defaults to .weak.

When .sem is .weak, the data copy operation accesses memory with weak memory operations.

When .sem is .relaxed, the data copy accesses memory with naturally-aligned strong memory operations with element-wise atomic size specified by .type and thread scope specified by .scope.

**PTX ISA Notes**

Introduced in PTX ISA version 9.1.

Support for .weak and .relaxed semantics, .scope and .type qualifiers are introduced in PTX ISA version 9.3.

**Target ISA Notes**

Requires sm_90 or higher.

Support for .cp_mask qualifier requires sm_100 or higher.

Qualifiers .weak, .relaxed, .scope and .type supported on following architectures:

sm_90a

sm_100f or higher in the same family

sm_110f or higher in the same family

**Examples**

```ptx
multimem.cp.async.bulk.global.shared::cta.bulk_group [dstMem], [srcMem], size;

multimem.cp.async.bulk.global.shared::cta.bulk_group [dstMem], [srcMem], 512;

multimem.cp.async.bulk.global.shared::cta.bulk_group.cp_mask [dstMem], [srcMem], size, byteMask;

multimem.cp.async.bulk.relaxed.cta.global.bulk_group.b128 [dstMem], [srcMem], size;
```

###### 中文翻译

`multimem.cp.async.bulk` 从当前 `.shared::cta` 的源区间 `[srcMem, srcMem + size)`，异步复制到 destination multimem 地址区间所引用的每个 GPU 的内存位置。目标为 `.global`，使用 `.bulk_group` completion。32-bit `size` 必须为 16 的倍数，源与目标地址须 16-byte 对齐，两个区间均不能越过各自 memory space；否则行为未定义。

可选 `.cp_mask` 要求传入 16-bit `byteMask`。其第 `i` 位若置位，则每个 16-byte 源块的第 `i` 字节会被复制；未置位的字节不由这次操作复制。

`.sem` 省略时默认为 `.weak`，以 weak memory operations 访问数据；`.relaxed` 变体使用 naturally-aligned strong memory operations，并由 `.type` 决定逐元素原子大小、由 `.scope` 决定 thread scope。目标 ISA 至少要求 `sm_90`，`.cp_mask` 需要 `sm_100` 或更高；较新的 ordering qualifier 仅适用于英文原文列出的特定架构族。

###### 重点解读

这里的 multimem 目标不是单一 GPU 上的普通 global 指针；一次 bulk copy 覆盖 multimem 地址映射到的每个 GPU。`.cp_mask` 是按每个 16-byte chunk 重复应用的逐字节 mask，不是只作用于整个复制区间的前 16 字节。

##### 9.7.10.28.4.5 Data Movement and Conversion Instructions: `multimem.cp.reduce.async.bulk`

###### English Original

multimem.cp.reduce.async.bulk

Initiates an asynchronous reduction operation to a multimem address range.

**Syntax**

```ptx
multimem.cp.reduce.async.bulk{.sem.scope}.dst.src.completion_mechanism.redOp.type  [dstMem], [srcMem], size;

    .sem                  =  { .relaxed }
    .scope                = { .cta, .cluster, .gpu, .sys }
    .dst                  = { .global }
    .src                  = { .shared::cta }
    .completion_mechanism = { .bulk_group }
    .redOp                = { .and, .or, .xor,
                              .add, .inc, .dec,
                              .min, .max }
    .type                 = { .f16, .bf16,
                              .b32, .u32, .s32,
                              .b64, .u64, .s64,
                              .f32, .f64 }

multimem.cp.reduce.async.bulk{.sem.scope}.dst.src.completion_mechanism.add.noftz.type  [dstMem], [srcMem], size;

    .sem                  = { .relaxed }
    .scope                = { .cta, .cluster, .gpu, .sys }
    .dst                  = { .global }
    .src                  = { .shared::cta }
    .completion_mechanism = { .bulk_group }
    .type                 = { .f16, .bf16, .f32 }
```

**Description**

Instruction multimem.cp.reduce.async.bulk initiates an element-wise asynchronous reduction operation with elements from source memory address range [srcMem, srcMem + size) to memory locations residing on each GPU’s memory referred to by the multimem destination address range [dstMem, dstMem + size).

Each data element in the destination array is reduced inline with the corresponding data element in the source array with the reduction operation specified by the modifier .redOp. The type of each data element in the source and the destination array is specified by the modifier .type.

The source address operand srcMem is in the state space specified by .src and the destination address operand dstMem is in the state specified by the .dst.

The 32-bit operand size specifies the amount of memory to be copied from the source location and used in the reduction operation, in terms of number of bytes. Operand size must be a multiple of 16. The memory range [dstMem, dstMem + size) must not overflow the destination multimem memory space. The memory range [srcMem, srcMem + size) must not overflow the source memory space. The addresses dstMem and srcMem must be aligned to 16 bytes. If any of these preconditions is not met, the behavior is undefined.

The operations supported by .redOp are classified as follows:

The bit-size operations are .and, .or, and .xor.

The integer operations are .add, .inc, .dec, .min, and .max. The .inc and .dec operations return a result in the range [0..x] where x is the value at the source state space.

The floating point operation .add rounds to the nearest even, preserve input and result subnormals, and does not flush them to zero. The multimem.cp.reduce.async.bulk.add.f16 and multimem.cp.reduce.async.bulk.add.bf16 operations require .noftz qualifier. It preserves input and result subnormals, and does not flush them to zero. The default behavior for multimem.cp.reduce.async.bulk.add.f32 is equivalent to specifying optional .noftz qualifier which preserves input and result subnormals, and does not flush them to zero.

The following table describes the valid combinations of .redOp and element type:

| `.redOp` | Element type |
|---|---|
| `.add` | `.u32`, `.s32`, `.u64`, `.f32`, `.f64`, `.f16`, `.bf16` |
| `.min`, `.max` | `.u32`, `.s32`, `.u64`, `.s64`, `.f16`, `.bf16` |
| `.inc`, `.dec` | `.u32` |
| `.and`, `.or`, `.xor` | `.b32`, `.b64` |


The modifier .completion_mechanism specifies the completion mechanism that is supported by the instruction. The modifier .bulk_group specifies that the multimem.cp.reduce.async.bulk uses bulk async-group based completion mechanism.

The optional .sem.scope qualifier specifies the memory ordering semantics of each individual element-wise reduction in dstMem performed by this operation as described in the Memory Consistency Model. These are element-wise atomic at the specified scope. Source location accesses are .weak, that is, not atomic. When .sem.scope is not specified, it defaults to .relaxed.sys.

The load operations from srcMem in multimem.cp.reduce.async.bulk are treated as weak memory operations as described in the Memory Consistency Model.

**PTX ISA Notes**

Introduced in PTX ISA version 9.1.

Support for .sem.scope qualifier is introduced in PTX ISA version 9.3.

Qualifier .noftz with .f32 type introduced in PTX ISA version 9.4.

**Target ISA Notes**

Requires sm_90 or higher.

**Examples**

```ptx
multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32 [dstMem], [srcMem], size;

multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.xor.b64 [dstMem], [srcMem], size;

multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.inc.u32 [dstMem], [srcMem], size;

multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.dec.u32 [dstMem], [srcMem], size;

multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.max.s32 [dstMem], [srcMem], size;

multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f16 [dstMem], [srcMem], size;

multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.min.bf16 [dstMem], [srcMem], size;

multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 [dstMem], [srcMem], size;

multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f32 [dstMem], [srcMem], size;

multimem.cp.reduce.async.bulk.relaxed.cta.global.bulk_group.add.u64 [dstMem], [srcMem], size, [mbar];
```

###### 中文翻译

`multimem.cp.reduce.async.bulk` 对 multimem 目标区间所引用的每个 GPU 的内存位置，使用 `.shared::cta` 源区间的对应元素执行异步逐元素归约。源与目标元素类型由 `.type` 指定，归约操作由 `.redOp` 指定。32-bit `size` 必须是 16 的倍数；源、目标地址须 16-byte 对齐，两个区间不能越过相应 memory space，否则行为未定义。

运算分为 bitwise（`.and`、`.or`、`.xor`）、整数（`.add`、`.inc`、`.dec`、`.min`、`.max`）和浮点 `.add`。`.inc`、`.dec` 结果位于 `[0..x]`，`x` 为源元素值。浮点 add 采用 nearest-even 舍入，保留输入和结果的 subnormal，不 flush to zero；`.f16` 与 `.bf16` 的 add 必须带 `.noftz`，`.f32` 默认等价于可选 `.noftz`。合法的 `.redOp` 与元素类型组合见英文表格。

指令使用 `.bulk_group`，即 bulk async-group completion。可选 `.sem.scope` 规定目标端每个 element-wise reduction 的 memory ordering 与 atomic scope；源端 load 是 `.weak`、非 atomic。省略 `.sem.scope` 时默认为 `.relaxed.sys`。源端 load 在一致性模型中属于 weak memory operation。目标 ISA 至少要求 `sm_90`，原文保留各功能的 PTX ISA 引入版本。

###### 重点解读

与普通 bulk reduction 相比，本指令把逐元素归约扩展到 multimem 地址映射的多个 GPU；但异步 completion 和目标端原子语义仍要分别理解。原文最后一个 `.relaxed` 示例也带 `[mbar]`，与其 `.bulk_group` 语法不一致，保留原文但不建议直接照抄。

#### 9.7.10.28.5 Data Movement and Conversion Instructions: Tensor Copy

##### 9.7.10.28.5.1 Restriction on Tensor Copy instructions

###### English Original

Following are the restrictions on the types .b4x16, .b4x16_p64, .b6x16_p32 and .b6p2x16:

1. cp.reduce.async.bulk doesn’t support the types .b4x16, .b4x16_p64, .b6x16_p32 and .b6p2x16.
1. cp.async.bulk.tensor with the direction .global.shared::cta doesn’t support the type .b4x16_p64.
1. cp.async.bulk.tensor with the direction .shared::cluster.global doesn’t support the sub-byte types on sm_120a.
1. OOB-NaN fill mode doesn’t support the types .b4x16, .b4x16_p64, .b6x16_p32 and .b6p2x16.
1. Box-Size[0] must be exactly:
   1. For b6x16_p32 and .b6p2x16:
      1. 128B for .shared::cluster.global and .shared::cta.global directions on sm_107f.
      1. 64B or 128B for .global.shared::cta direction on sm_107f.
      1. 96B on sm_100f.
   1. 64B for b4x16_p64.
1. Tensor-Size[0] must be a multiple of:
   1. For b6x16_p32 and .b6p2x16:
      1. 128B for .shared::cluster.global and .shared::cta.global directions on sm_107f.
      1. 64B or 128B for .global.shared::cta direction on sm_107f.
      1. 96B on sm_100f.
   1. 64B for b4x16_p64.
1. For .b4x16_p64, .b6x16_p32 and .b6p2x16, the first coordinate in the tensorCoords argument vector must be a multiple of 128.
1. For .b4x16_p64, .b6x16_p32 and .b6p2x16, the global memory address must be 32B aligned. Additionally, tensor stride in every dimension must be 32B aligned.
1. For .b6x16_p32 and .b6p2x16:
   1. On sm_107f, for .shared::cluster.global and .shared::cta.global directions, the global memory address must be 32B aligned. Additionally, tensor stride in every dimension must be 32B aligned.
   1. On sm_100f, the global memory address must be 32B aligned. Additionally, tensor stride in every dimension must be 32B aligned.
1. For .b6x16_p32 and .b6p2x16:
   1. On sm_107f, for .global.shared::cta direction, the global memory address must be 16B aligned. Additionally, tensor stride in every dimension must be 16B aligned on.
   1. On sm_100f, the global memory address must be 32B aligned. Additionally, tensor stride in every dimension must be 32B aligned.
1. .b4x16_p64 supports the following swizzling modes:
   1. None.
   1. 128B (With all potential swizzle atomicity values except: 32B with 8B flip)
1. .b6x16_p32 and .b6p2x16 supports the following swizzling modes:
   1. None.
   1. 64-Byte with 16B atomic swizzling on sm_107f.
   1. 128-Byte with 16B, 32B and 64B atomic swizzling on sm_107f
   1. 128-Byte (With all potential swizzle atomicity values except: 32B with 8B flip) on sm_100f
1. .b6x16_p32 and .b6p2x16 doesn’t support the following swizzling modes:
   1. 32-Byte with 16B atomic swizzling
   1. 128-Byte with 32B atomic swizzling with 8B flip


Following are the restrictions on the 96B swizzle mode:

1. The .swizzle_atomicity must be 16B.
1. The .interleave_layout must not be set.
1. Box-Size[0] must be less than or equal to 96B.
1. The type must not be among following: .b4x16_p64, .b6x16_p32 and .b6p2x16.
1. The .load_mode must not be set to .im2col::w::128.


Following are the restrictions on the 128-byte swizzle mode having 32-byte atomicity with 8-byte flip as sub-mode:

1. The copy direction must be .shared::cluster.global or otherwise .shared::cta.global.
1. The .load_mode must not be set to .im2col::w, .im2col::w::128.


Following are the restrictions on the .global.shared::cta direction:

1. Starting co-ordinates for Bounding Box (tensorCoords) must be non-negative with the following exception:
   1. For sm_107f, starting co-ordinates for Bounding Box (tensorCoords) can be negative for .tile::scatter4 mode.
1. The bounding box along the D, W and H dimensions must stay within the tensor boundaries. This implies:
   1. Bounding-Box Lower-Corner must be non-negative.
   1. Bounding-Box Upper-Corner must be non-positive.


Following are the restrictions for sm_120a:

1. cp.async.bulk.tensor with the direction .shared::cluster.global doesn’t support:
   1. the sub-byte types
   1. the qualifier .swizzle_atomicity


Following are the restrictions for sm_103a while using type .b6p2x16 on cp.async.bulk.tensor with the direction .global.shared::cta:

1. Box-Size[0] must be exactly either of 48B or 96B.
1. The global memory address must be 16B aligned.
1. Tensor Stride in every dimension must be 16B aligned.
1. The first coordinate in the tensorCoords argument vector must be a multiple of 64.
1. Tensor-Size[0] must be a multiple of 48B.
1. The following swizzle modes are supported:
   1. None.
   1. 128B (With all potential swizzle atomicity values except: 32B with 8B flip)
   1. 64B swizzle with 16B swizzle atomicity

###### 中文翻译

这部分列出 tensor copy 的类型与布局约束。对 `.b4x16`、`.b4x16_p64`、`.b6x16_p32`、`.b6p2x16`，`cp.reduce.async.bulk` 均不支持；`.global.shared::cta` 方向的 `cp.async.bulk.tensor` 不支持 `.b4x16_p64`，`sm_120a` 上 `.shared::cluster.global` 方向不支持 sub-byte 类型。OOB-NaN fill mode 也不支持上述四种类型。

对 `.b6x16_p32`、`.b6p2x16`，`Box-Size[0]` 的精确值、`Tensor-Size[0]` 的倍数均随目标和方向变化：`sm_107f` 的 `.shared::cluster.global`、`.shared::cta.global` 为 128B；其 `.global.shared::cta` 为 64B 或 128B；`sm_100f` 为 96B。对 `.b4x16_p64`，前者必须恰好 64B，后者必须是 64B 的倍数。

对 `.b4x16_p64`、`.b6x16_p32`、`.b6p2x16`，`tensorCoords` 的第一坐标须为 128 的倍数；原文还给出 global memory 地址与各维 tensor stride 的 32B 对齐约束。对于后两种 6-bit 类型，`sm_107f` 的 shared-to-global 方向和 `sm_100f` 均要求 32B 对齐；`sm_107f` 的 `.global.shared::cta` 方向列出 16B 对齐，而 `sm_100f` 仍为 32B。具体条件应按目标架构和 copy 方向逐项核对。

`.b4x16_p64` 支持无 swizzle 或 128B swizzle，但后者排除 32B atomicity 配合 8B flip。`.b6x16_p32` 与 `.b6p2x16` 支持无 swizzle；在 `sm_107f` 可用 64B/16B atomic swizzle，或 128B 配 16B、32B、64B atomic swizzle；在 `sm_100f` 可用除 32B atomicity 配 8B flip 之外的 128B swizzle。它们不支持 32B/16B atomic swizzle，也不支持 128B/32B atomicity 加 8B flip。

96B swizzle 要求 `.swizzle_atomicity = 16B`、不设置 `.interleave_layout`、`Box-Size[0] ≤ 96B`、类型不能是 `.b4x16_p64` / `.b6x16_p32` / `.b6p2x16`，且 `.load_mode` 不能是 `.im2col::w::128`。对“128B swizzle、32B atomicity、8B flip”这个子模式，复制方向只能是 `.shared::cluster.global` 或 `.shared::cta.global`，并且不能用 `.im2col::w` 或 `.im2col::w::128`。

`.global.shared::cta` 方向的 bounding box 起始坐标原则上不能为负；例外是 `sm_107f` 的 `.tile::scatter4`。D、W、H 维的 bounding box 必须留在 tensor 边界内；原文分别表述为 lower corner 非负、upper corner 非正。这里照译原文约束，不擅自改动该 upper-corner 表述。

`sm_120a` 的 `.shared::cluster.global` 方向不支持 sub-byte 类型，也不支持 `.swizzle_atomicity`。在 `sm_103a`、类型 `.b6p2x16`、方向 `.global.shared::cta` 的特定组合下，`Box-Size[0]` 必须是 48B 或 96B；global 地址和各维 stride 要 16B 对齐；首坐标为 64 的倍数；`Tensor-Size[0]` 为 48B 的倍数；支持无 swizzle、排除 32B atomicity/8B flip 的 128B swizzle，或 64B/16B atomic swizzle。

###### 重点解读

这一节的约束不是单一“类型是否支持”的开关，而是类型、复制方向、目标架构、box/tensor size、地址与 stride 对齐、swizzle 子模式共同决定是否合法。特别是 `sm_103a` 的例外与其他目标数值不同，不能把某个尺寸或对齐值泛化给全部 6-bit tensor。

##### 9.7.10.28.5.2 Overriding tensor property value

###### English Original

PTX instructions operating on tensor data require an opaque tensor-map object. The tensor-map specifies various properties associated with the tensor. Certain PTX instructions, when qualified with .override::* qualifier, ignore a specific property from the tensor-map; instead, the explicit instruction operand’s value is used for that property. The following subsection describes semantics around overriding various properties from the tensor-map object.

###### 中文翻译

处理 tensor 数据的 PTX 指令需要 opaque tensor-map object，其中保存 tensor 的各种属性。部分指令使用 `.override::*` 时，会忽略 tensor map 中的特定属性，改用显式指令 operand 的值。下两个小节分别讨论 tensor base address 和 tensor attribute 的覆盖。

###### 重点解读

Tensor map 是默认属性来源，override 改变的是本次指令采用的特定属性值；不能理解为修改了 descriptor 本身。

###### 9.7.10.28.5.2.1 Tensor Base Address Override

**English Original**

The optional qualifier .override::global_address allows overriding of the tensor base address. When .override::global_address is specified, the global memory address present in the opaque tensorMap object is ignored and instead the address specified by the explicit 64-bit operand gAddrToOverride is used. The gAddrToOverride address must be 16B aligned; otherwise, a runtime error is raised. Additionally, the memory address range [gAddrToOverride, gAddrToOverride + 128 KiB) must be allocated and accessible during execution; otherwise, the behavior is undefined.

**中文翻译**

`.override::global_address` 允许覆盖 tensor base address：忽略 `tensorMap` 内的 global 地址，改用显式的 64-bit `gAddrToOverride`。该地址必须 16B 对齐，否则产生 runtime error；执行期间，`[gAddrToOverride, gAddrToOverride + 128 KiB)` 必须已经分配且可访问，否则行为未定义。

**重点解读**

这里的 16B 地址对齐是 runtime error 条件，128 KiB 可访问范围则是不满足时的 undefined behavior；两种失败方式不同。

###### 9.7.10.28.5.2.2 Tensor Attribute Override

**English Original**

The optional qualifier .override_attribute allows overriding of tensor attributes. The only attributes that are allowed to be overridden are global dimension (global_dim) and global stride (global_stride). When the qualifier .override_attribute is specified, attribute values from the tensorMap object are ignored and values specified via the additional operand attributeOverrideInfo are used. The following conditions must be satisfied for valid usage of .override_attribute qualifier:

As part of tensor attributes override, the tensor base address override must also naturally follow. In other words, while using .override_attribute qualifier, it is mandatory to also specify .override::global_address qualifier.

The tensor data access mode must be tile. i.e. The qualifier .load_mode if specified must be set to .tile.

If specified, the optional .multicast qualifier must be .multicast::cluster::32b.

Tensor start coordinates as specified by the operand tensorCoords must be zero, otherwise the behavior is undefined.

The .override_attribute qualifier can take the following two values: .override::global_dim and .override::global_dim_stride. For .1d tensors, since stride is absent, only .override::global_dim is valid. For tensors with dimensionality of at least 2, both dimension and stride must be overridden together, so only .override::global_dim_stride is valid. The form of the attributeOverrideInfo operand depends on the value used for the .override_attribute qualifier and is summarized in the Table 34.

Table 34 Nature of `attributeOverrideInfo`

| `.dim` | valid `.override_attribute` qualifier | Nature of `attributeOverrideInfo` argument |
|---|---|---|
| `.1d` | `.override::global_dim` | `tensorSizeToOverride` |
| `.2d` to `.5d` | `.override::global_dim_stride` | `tensorSizeToOverride`, `tensorLowerStrideToOverride`, `tensorUpperStrideToOverride` |

When .override::global_dim is used, the operand tensorSizeToOverride specifies the new value for dimension. In this case, the operand tensorSizeToOverride is a brace-enclosed vector expression comprising singleton 8-bit or otherwise 16-bit register. Note that, when .b16 register is specified, 8-bit value must be wrapped inside lower half of the register operand. When .override::global_dim_stride is specified the operand tensorSizeToOverride must be a vector expression with number of 8-bit or otherwise 16-bit register elements exactly matching to the tensor dimensionality as specified by the qualifier .dim. Note that, when vector comprising .b16 register is specified, 8-bit value must be wrapped inside lower half of each of the register operand. In this case, additional operands tensorLowerStrideToOverride and tensorUpperStrideToOverride specify the new values to be used for tensor lower stride and tensor upper stride respectively. The tensorLowerStrideToOverride is a brace-enclosed vector expression comprising of 32-bit registers. The vector size for tensorLowerStrideToOverride must be one less than the tensor dimensionality as given by .dim. The tensorUpperStrideToOverride operand must be single .b16 register holding at most four 4-bit tensor upper stride values. The total number of 4-bit tensor upper stride values packed within the tensorUpperStrideToOverride operand must be one less than the tensor dimensionality given by .dim otherwise the behavior is undefined. Any unused bits from tensorUpperStrideToOverride can be populated with zero. As an example, when .dim is set to .4d, lower 12-bits from tensorUpperStrideToOverride operand specify tensor upper stride values whereas upper 4-bits can be set to zero.

Table 35 Operand type and size requirements for each component of `attributeOverrideInfo`

| Component of `attributeOverrideInfo` | Nature of the operand |
|---|---|
| `tensorSizeToOverride` | Vector expression comprising `.dim` many elements of type `.b8` or `.b16`. For `.b16` register, the value must reside within the lower 8-bits of operand. |
| `tensorLowerStrideToOverride` | Vector expression of size `.dim - 1` where each element is of type `.b32`. |
| `tensorUpperStrideToOverride` | Single `.b16` packed register storing `.dim - 1` values. Each value is of type `.b4`. Unused bits must be set to 0. |

For each tensor dimension, the values specified for tensorLowerStrideToOverride and tensorUpperStrideToOverride are operated as shown in the below equation to compute the effective global stride (global_stride) value to be used instead of the one already present in the tensorMap object.

tensorGlobalStride[i] = ((tensorLowerStrideToOverride[i] + (tensorUpperStrideToOverride[i] << 32)) << 4);

**中文翻译**

`.override_attribute` 仅允许覆盖 global dimension（`global_dim`）和 global stride（`global_stride`），替换值来自额外 operand `attributeOverrideInfo`，而非 `tensorMap`。使用时必须同时指定 `.override::global_address`；访问模式必须是 tile（若指定 `.load_mode`，只能为 `.tile`）；若指定 `.multicast`，只能是 `.multicast::cluster::32b`；`tensorCoords` 的起始坐标必须全为零，否则行为未定义。

`.1d` tensor 没有 stride，所以只能用 `.override::global_dim`；`.2d` 到 `.5d` 必须同时覆盖 dimension 和 stride，只能用 `.override::global_dim_stride`。Table 34 与 Table 35 保留原文，分别列出 operand 组成以及类型、向量长度约束，不另行翻译表格。

使用 `.override::global_dim` 时，`tensorSizeToOverride` 是花括号包围的单元素 8-bit 或 16-bit register 向量；用 `.b16` register 时，8-bit 值位于低半部。使用 `.override::global_dim_stride` 时，该向量必须恰有 `.dim` 个 8-bit 或 16-bit register 元素，`.b16` 的有效 8 bit 同样放在各元素的低半部。

此外，`tensorLowerStrideToOverride` 是由 `.dim - 1` 个 32-bit register 构成的花括号向量；`tensorUpperStrideToOverride` 是单个 `.b16` register，其中打包 `.dim - 1` 个 4-bit upper-stride 值，多余 bit 置零。例如 `.4d` 使用低 12 bit 表示三个 upper-stride 值，高 4 bit 可置零。每维的 effective global stride 按原文公式计算：`tensorGlobalStride[i] = ((tensorLowerStrideToOverride[i] + (tensorUpperStrideToOverride[i] << 32)) << 4)`。

**重点解读**

覆盖属性与仅覆盖基地址不同：前者还强制覆盖基地址、tile 模式和零起点。上位 stride 与下位 stride 被拼接后再整体左移 4 位，所以有效 stride 具有 16-byte 粒度；`.dim - 1` 也反映首维 stride 不需要单独提供。

##### 9.7.10.28.5.3 Data Movement and Conversion Instructions: `cp.async.bulk.tensor`

###### English Original

cp.async.bulk.tensor

Initiates an asynchronous copy operation on the tensor data from one state space to another.

**Syntax**

```ptx
// global -> shared::cta
cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.cta_group}{.level::cache_hint}{.override::global_address}{.override_attribute}
                                   [dstMem], [tensorMap{, gAddrToOverride}{, attributeOverrideInfo}, tensorCoords], [mbar]{, im2colInfo} {, cache_policy}

.dst =                  { .shared::cta }
.src =                  { .global }
.dim =                  { .1d, .2d, .3d, .4d, .5d }
.completion_mechanism = { .mbarrier::complete_tx::bytes{.report_mechanism} }
.cta_group =            { .cta_group::1, .cta_group::2 }
.load_mode =            { .tile, .tile::gather4, .im2col, .im2col::w, .im2col::w::128 }
.level::cache_hint =    { .L2::cache_hint }
.override_attribute =   { .override::global_dim, .override::global_dim_stride }
.report_mechanism =     { .mbarrier::report::disabled,
                          .mbarrier::report::validity::per_16bytes::80000000,
                          .mbarrier::report::validity::per_16bytes::8000,
                          .mbarrier::report::validity::per_16bytes::80,
                          .mbarrier::report::validity::per_16bytes::8,
                          .mbarrier::report::validity::per_element::ff
                        }


// global -> shared::cluster
cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.multicast}{.cta_group}{.level::cache_hint}{.override::global_address}{.override_attribute}
                                   [dstMem], [tensorMap{, gAddrToOverride}{, attributeOverrideInfo}, tensorCoords], [mbar]{, im2colInfo}
                                   {, ctaMask} {, cache_policy}

.dst =                  { .shared::cluster }
.src =                  { .global }
.dim =                  { .1d, .2d, .3d, .4d, .5d }
.completion_mechanism = { .mbarrier::complete_tx::bytes{.report_mechanism} }
.cta_group =            { .cta_group::1, .cta_group::2 }
.load_mode =            { .tile, .tile::gather4, .im2col, .im2col::w, .im2col::w::128 }
.level::cache_hint =    { .L2::cache_hint }
.multicast =            { .multicast::cluster{::16b, ::32b} }
.override_attribute =   { .override::global_dim, .override::global_dim_stride }
.report_mechanism =     { .mbarrier::report::disabled,
                          .mbarrier::report::validity::per_16bytes::80000000,
                          .mbarrier::report::validity::per_16bytes::8000,
                          .mbarrier::report::validity::per_16bytes::80,
                          .mbarrier::report::validity::per_16bytes::8,
                          .mbarrier::report::validity::per_element::ff
                        }


// shared::cta -> global
cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.level::cache_hint}{.override::global_address}{.override_attribute}
                                   [tensorMap{, gAddrToOverride}{, attributeOverrideInfo}, tensorCoords], [srcMem] {, cache_policy}

.dst =                  { .global }
.src =                  { .shared::cta }
.dim =                  { .1d, .2d, .3d, .4d, .5d }
.completion_mechanism = { .bulk_group }
.load_mode =            { .tile, .tile::scatter4, .im2col_no_offs, .im2col_no_offs::w }
.level::cache_hint =    { .L2::cache_hint }
.override_attribute =   { .override::global_dim, .override::global_dim_stride }
```

**Description**

cp.async.bulk.tensor is a non-blocking instruction which initiates an asynchronous copy operation of tensor data from the location in .src state space to the location in the .dst state space.

The operand dstMem specifies the location in the .dst state space into which the tensor data has to be copied and srcMem specifies the location in the .src state space from which the tensor data has to be copied.

When .dst is specified as .shared::cta, the address dstMem must be in the shared memory of the executing CTA within the cluster, otherwise the behavior is undefined. When .dst is .shared::cta, the mbarrier object mbar must also reside in the shared memory of the executing CTA, otherwise behavior is undefined. If the mbarrier object mbar resides in the shared memory of a different CTA within the cluster (a remote barrier), .dst must be specified as .shared::cluster, otherwise behavior is undefined.

When .dst is specified as .shared::cluster, the address dstMem can be in the shared memory of any of the CTAs within the current cluster.

The operand tensorMap is the generic address of the opaque tensor-map object which resides in .param space or .const space or .global space. The operand tensorMap specifies the properties of the tensor copy operation, as described in Tensor-map. The tensorMap is accessed in tensormap proxy. Refer to the CUDA programming guide for creating the tensor-map objects on the host side.

The dimension of the tensor data is specified by the .dim modifier.

The vector operand tensorCoords specifies the starting coordinates in the tensor data in the global memory from or to which the copy operation has to be performed. The individual tensor coordinates in tensorCoords are of type .s32. The format of vector argument tensorCoords is dependent on .load_mode specified and is as follows:
| `.load_mode` | `tensorCoords` | Semantics |
|---|---|---|
| `.tile::scatter4` | `{col_idx, row_idx0, row_idx1, row_idx2, row_idx3}` | Fixed length vector of size 5. The five elements together specify the start co-ordinates of the four rows. |
| `.tile::gather4` | `{col_idx, row_idx0, row_idx1, row_idx2, row_idx3}` | Fixed length vector of size 5. The five elements together specify the start co-ordinates of the four rows. |
| Rest all | `{d0, .., dn}` for `n = .dim` | Vector of `n` elements where `n = .dim`. The elements indicate the offset in each of the dimension. |


The modifier .completion_mechanism specifies the completion mechanism that is supported on the instruction variant. The completion mechanisms that are supported for different variants are summarized in the following table:
| `.completion-mechanism` | `.dst` | `.src` | Completion mechanism needed for entire Async operation | Optional completion of reading tensormap object |
|---|---|---|---|---|
| `.mbarrier::...` | `.shared::cta` | `.global` | mbarrier based | Bulk async-group based |
| `.mbarrier::...` | `.shared::cluster` | `.global` | mbarrier based | Bulk async-group based |
| `.bulk_group` | `.global` | `.shared::cta` | Bulk async-group based | |


The modifier .mbarrier::complete_tx::bytes specifies that the cp.async.bulk.tensor variant uses mbarrier based completion mechanism. Upon the completion of the asynchronous copy operation, the complete-tx operation, with completeCount argument equal to amount of data copied in bytes, will be performed on the mbarrier object specified by the operand mbar. This instruction accesses its mbarrier operand using generic-proxy. When the mbarrier object mbar resides in the shared memory of a different CTA within the cluster (a remote barrier), .dst must be specified as .shared::cluster; using .shared::cta as .dst in this case will not signal the remote barrier and results in undefined behavior.

The modifier .cta_group can only be specified with the mbarrier based completion mechanism. The modifier .cta_group is used to signal either the odd numbered CTA or the even numbered CTA among the CTA-Pair. When .cta_group::1 is specified, the mbarrier object mbar that is specified must be in the shared memory of the same CTA as the shared memory destination dstMem. When .cta_group::2 is specified, the mbarrier object mbar can be in shared memory of either the same CTA as the shared memory destination dstMem or in its peer-CTA. If .cta_group is not specified, then it defaults to .cta_group::1.

The modifier .bulk_group specifies that the cp.async.bulk.tensor variant uses bulk async-group based completion mechanism.

The qualifier .load_mode specifies how the data in the source location is copied into the destination location. If .load_mode is not specified, it defaults to .tile.

In .tile mode, the multi-dimensional layout of the source tensor is preserved at the destination. In .tile::gather4 mode, four rows in 2-dimensional source tensor are combined to form a single 2-dimensional destination tensor. In .tile::scatter4 mode, single 2-dimensional source tensor is divided into four rows in 2-dimensional destination tensor. Details of .tile::scatter4/.tile::gather4 modes are described in .tile::scatter4 and .tile::gather4 modes.

In .im2col, .im2col::*, .im2col_no_offs and .im2col_no_offs::w modes, some dimensions of the source tensors are unrolled in a single dimensional column at the destination. Details of various im2col modes are described in im2col mode. In each of these modes, the tensor has to be at least 3-dimensional. The vector operand im2colInfo can be specified only when .load_mode is .im2col or .im2col::w or .im2col::w::128. The format of the vector argument im2colInfo is dependent on the exact im2col mode and is as follows:
| Exact im2col mode | `im2colInfo` argument | Semantics |
|---|---|---|
| `.im2col` | `{ i2cOffW , i2cOffH , i2cOffD }` for `.dim = .5d` | A vector of im2col offsets whose vector size is two less than number of dimensions `.dim`. |
| `.im2col::w` | `{ wHalo, wOffset }` | A vector of 2 arguments containing `wHalo` and `wOffset` arguments. |
| `.im2col::w::128` | `{ wHalo, wOffset }` | A vector of 2 arguments containing `wHalo` and `wOffset` arguments. |
| `.im2col_no_offs` | im2colInfo is not applicable. | im2colInfo is not applicable. |
| `.im2col_no_offs::w` | im2colInfo is not applicable. | im2colInfo is not applicable. `wHalo` and `wOffset` values default to 0. |


Argument wHalo is a 16bit unsigned integer whose valid set of values differs on the load-mode and is as follows: - Im2col::w mode : valid range is [0, 512). - Im2col::w::128 mode : valid range is [0, 32).

Argument wOffset is a 16bit unsigned integer whose valid range of values is [0, 32).

The optional qualifier .multicast::cluster allows copying of data from global memory to shared memory of multiple CTAs in the cluster. Operand ctaMask specifies the destination CTAs in the cluster such that each bit position in the 16-bit or otherwise 32-bit ctaMask operand corresponds to the %cluster_ctarank of the destination CTA. The additional sub-qualifier ::16b or ::32b can be specified to correspond to the width of the ctaMask argument. By default, ctaMask is assumed to be of 16-bit width. The source data is multicast to the same offset as dstMem in the shared memory of each destination CTA. When .cta_group is specified as:

.cta_group::1 : The mbarrier signal is multicast to the same shared memory offset as mbar in the destination CTA.

.cta_group::2 : The mbarrier signal is multicast either to all the odd-numbered CTAs or even-numbered CTAs for each possible CTA-Pair within a cluster. In particular, for each destination CTA specified in the ctaMask, the mbarrier signal is sent either to the destination CTA or its peer-CTA based on the %cluster_ctarank parity of the shared memory where the mbarrier object mbar resides.

The optional modifiers .override::global_address and .override_attribute support value override functionality for various tensor properties present in the tensorMap object. For more details refer to the Overriding tensor property value section.

When the optional argument cache_policy is specified, the qualifier .level::cache_hint is required. The 64-bit operand cache_policy specifies the cache eviction policy that may be used during the memory access.

cache_policy is a hint to the cache subsystem and may not always be respected. It is treated as a performance hint only, and does not change the memory consistency behavior of the program.

The optional qualifier .report_mechanism specifies the mbarrier reporting mechanism as outlined in the Report Mechanisms for Asynchronous Copy Operations.

The copy operation in cp.async.bulk.tensor is treated as a weak memory operation and the complete-tx operation on the mbarrier has .release semantics at the .cluster scope as described in the Memory Consistency Model.

**Notes**

The copy operation with shared::cluster.global is optimized for target architecture sm_90a/sm_100f/sm_100a/ sm_103f/sm_103a/sm_107a/sm_107f/sm_110f/sm_110a and may have substantially reduced performance on other targets and hence such operation is advised to be used with .target sm_90a/sm_100f/sm_100a/sm_103f/sm_103a/sm_107a/sm_107f/sm_110f/sm_110a.

**PTX ISA Notes**

Introduced in PTX ISA version 8.0.

Support for .shared::cta as destination state space is introduced in PTX ISA version 8.6.

Support for qualifiers .tile::gather4 and .tile::scatter4 introduced in PTX ISA version 8.6.

Support for qualifiers .im2col::w and .im2col::w::128 introduced in PTX ISA version 8.6.

Support for qualifier .cta_group introduced in PTX ISA version 8.6.

Support for .multicast::cluster::16b and .multicast::cluster::32b qualifiers introduced in PTX ISA version 9.4.

Support for qualifiers .override::global_address and .override_attribute introduced in PTX ISA version 9.4.

Support for qualifier .im2col_no_offs::w introduced in PTX ISA version 9.4.

Support for qualifier .report_mechanism introduced in PTX ISA version 9.4.

**Target ISA Notes**

Requires sm_90 or higher.

.multicast::cluster qualifier advised to be used with .target sm_90a or sm_100f or sm_100a or sm_103f or sm_103a or sm_110f or sm_110a.

Qualifiers .tile::gather4 and .im2col::w require:

sm_100a when destination state space is .shared::cluster and is supported on sm_100f from PTX ISA version 8.8.

sm_100 or higher when destination state space is .shared::cta.

Qualifier .tile::scatter4 is supported on following architectures:

sm_100a

sm_101a (Renamed to sm_110a from PTX ISA version 9.0)

And is supported on following family-specific architectures from PTX ISA version 8.8:

sm_100f or higher in the same family

sm_101f or higher in the same family (Renamed to sm_110f from PTX ISA version 9.0)

sm_110f or higher in the same family

Qualifier .im2col::w::128 is supported on following architectures:

sm_100a

sm_101a (Renamed to sm_110a from PTX ISA version 9.0)

And is supported on following family-specific architectures from PTX ISA version 8.8:

sm_100f or higher in the same family

sm_101f or higher in the same family (Renamed to sm_110f from PTX ISA version 9.0)

sm_110f or higher in the same family

Qualifier .cta_group is supported on following architectures:

sm_100a

sm_101a (Renamed to sm_110a from PTX ISA version 9.0)

And is supported on following family-specific architectures from PTX ISA version 8.8:

sm_100f or higher in the same family

sm_101f or higher in the same family (Renamed to sm_110f from PTX ISA version 9.0)

sm_110f or higher in the same family

Qualifier .multicast::cluster::32b is supported on following family-specific architectures:

sm_107f or higher in the same family

Qualifiers .override::global_address and .override_attribute are supported on following family-specific architectures:

sm_107f or higher in the same family

Qualifier .im2col_no_offs::w is supported on following family-specific architectures:

sm_107f or higher in the same family

Qualifier .report_mechanism is supported on following family-specific architectures:

sm_107f or higher in the same family

**Examples**

```ptx
.reg .b16 ctaMask;
.reg .u16 i2cOffW, i2cOffH, i2cOffD;
.reg .b64 l2CachePolicy;
.reg .b8  newTensorSize_<5>;
.reg .b32 newTensorLowerStrd_<4>;
.reg .b16 tensorUpperStrd;
.reg .u64 gAddrToOverride<3>;

cp.async.bulk.tensor.1d.shared::cta.global.mbarrier::complete_tx::bytes.tile  [sMem0], [tensorMap0, {tc0}], [mbar0];

@p cp.async.bulk.tensor.5d.shared::cta.global.im2col.mbarrier::complete_tx::bytes
                     [sMem2], [tensorMap2, {tc0, tc1, tc2, tc3, tc4}], [mbar2], {i2cOffW, i2cOffH, i2cOffD};

cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.tile  [sMem0], [tensorMap0, {tc0}], [mbar0];

@p cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster
                     [sMem1], [tensorMap1, {tc0, tc1}], [mbar2], ctaMask;

@p cp.async.bulk.tensor.5d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes
                     [sMem2], [tensorMap2, {tc0, tc1, tc2, tc3, tc4}], [mbar2], {i2cOffW, i2cOffH, i2cOffD};

@p cp.async.bulk.tensor.3d.im2col.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint
                     [sMem3], [tensorMap3, {tc0, tc1, tc2}], [mbar3], {i2cOffW}, policy;

@p cp.async.bulk.tensor.1d.global.shared::cta.bulk_group  [tensorMap3, {tc0}], [sMem3];

cp.async.bulk.tensor.2d.tile::gather4.shared::cluster.global.mbarrier::complete_tx::bytes
                     [sMem5], [tensorMap6, {x0, y0, y1, y2, y3}], [mbar5];

cp.async.bulk.tensor.3d.im2col::w.shared::cluster.global.mbarrier::complete_tx::bytes
                     [sMem4], [tensorMap5, {t0, t1, t2}], [mbar4], {im2colwHalo, im2colOff};

cp.async.bulk.tensor.1d.shared::cluster.global.tile.cta_group::2
                     [sMem6], [tensorMap7, {tc0}], [peerMbar];

cp.async.bulk.tensor.mbarrier::complete_tx::bytes.shared::cta.global.1d.override::global_address
                     [sMem7], [tensorMap8, gAddrToOverride1, {tc0}], [peerMbar];

cp.async.bulk.tensor.mbarrier::complete_tx::bytes.shared::cta.global.5d.tile.override::global_address.cta_group::1.override::global_dim_stride
                     [sMem8], [tensorMap9, gAddrToOverride2, {newTensorSize_0, newTensorSize_1, newTensorSize_2, newTensorSize_3, newTensorSize_4}, {newTensorLowerStrd_0, newTensorLowerStrd_1, newTensorLowerStrd_2, newTensorLowerStrd_3}, tensorUpperStrd, {0, 0, 0, 0, 0}], [mbar];

cp.async.bulk.tensor.global.shared::cta.1d.bulk_group.tile.override::global_address.override::global_dim
                     [tensorMap10, gAddrToOverride3, {newTensorSize_0}, {0}], [sMem9];

cp.async.bulk.tensor.mbarrier::complete_tx::bytes.shared::cluster.global.4d.im2col::w::128.override::global_address.cta_group::1
                     [sMem10], [tensorMap11, gAddrToOverride4, {tc0, tc1, tc2, tc3}], [mbar5], {im2col_0, im2col_1};

cp.async.bulk.tensor.3d.global.shared::cta.bulk_group.im2col_no_offs::w  [tensorMap6, {tc0, tc1, tc2}], [sMem11];

// Validity check using 0x8 pattern on every 16B
cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.mbarrier::report::validity::per_16bytes::8 [sMem], [tensorDesc, {u}], [mbar];

// Validity check using 0xff pattern on per element
cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.mbarrier::report::validity::per_element::ff
                     [sMem], [tensorDesc, {u, v, w, x, y}], [mbar];
```

###### 中文翻译

`cp.async.bulk.tensor` 使用 tensor map 指定张量属性，异步把 tensor 数据从 `.src` 复制到 `.dst`。语法涵盖 `.global → .shared::cta`、`.global → .shared::cluster` 与 `.shared::cta → .global`。`dstMem` 和 `srcMem` 分别是目标和源的 shared-memory 地址；若目标为 `.shared::cta`，`dstMem` 必须属于执行 CTA。此时 `mbar` 也必须位于执行 CTA 的 shared memory；若 barrier 属于 cluster 中另一 CTA，必须使用 `.shared::cluster` 目标，否则不发出远端 barrier signal，行为未定义。`.shared::cluster` 目标地址则可以属于当前 cluster 的任一 CTA。

`tensorMap` 是 opaque tensor-map object 的 generic address，可位于 `.param`、`.const` 或 `.global`，通过 tensormap proxy 访问；它提供复制的 tensor 属性，host 端创建方式参见 CUDA Programming Guide。`.dim` 指定 1D–5D 维度。`.s32` 的 `tensorCoords` 向量指定 global tensor 中的起始坐标；其形状由 `.load_mode` 决定：scatter4/gather4 使用五元素的列坐标和四个行坐标，其余模式按维度数提供坐标。原文表格保留完整形式，不另译表格。

完成机制具有两个不同关注点：`.mbarrier::complete_tx::bytes` 负责 global-to-shared 复制整体完成，复制完后按实际复制字节数对 `mbar` 执行 complete-tx，barrier operand 通过 generic proxy 访问；该变体还可选择通过 bulk async-group 跟踪 tensor map 对象的读取完成。shared-to-global 的 `.bulk_group` 则由 bulk async-group 负责完整操作。不要把“tensor map 已读完”和“目标数据已写完”当成同一事件。

`.cta_group` 只适用于 mbarrier-based 变体，默认 `.cta_group::1`。它在 CTA-Pair 中决定向奇数或偶数 CTA 发出 signal：`::1` 要求 `mbar` 与 `dstMem` 在同一 CTA 的 shared memory；`::2` 允许 `mbar` 位于目标 CTA 或其 peer-CTA。

`.load_mode` 省略时为 `.tile`，保持多维布局。`.tile::gather4` 将二维源的四行组合到二维目标，`.tile::scatter4` 将二维源分散为目标的四行。`.im2col`、`.im2col::*`、`.im2col_no_offs` 等模式把部分维度展成目标的一维列，要求 tensor 至少三维；仅 `.im2col`、`.im2col::w`、`.im2col::w::128` 接收额外 `im2colInfo`。原文表格列出各模式的 operand 形式；`wHalo` 对 `.im2col::w` 的范围是 `[0,512)`，对 `.im2col::w::128` 是 `[0,32)`，`wOffset` 的范围是 `[0,32)`，均为 16-bit unsigned。

`.multicast::cluster` 可以把 global tensor 数据复制到 cluster 内多个 CTA，由 16-bit（默认）或 32-bit 的 `ctaMask` 按 `%cluster_ctarank` 选取目标。各 CTA 的目标 shared 地址具有相同 offset；`.cta_group::1` 时，mbarrier signal 广播到各目标 CTA 对应 offset；`::2` 时则依 barrier 所在 CTA 的 rank parity，发送到目标 CTA 或其 peer-CTA 中相应的奇/偶 CTA。

可选 `.override::global_address`、`.override_attribute` 用显式 operand 覆盖 tensor map 属性，约束见上一小节。指定 `cache_policy` 时必须有 `.level::cache_hint`，该驱逐策略仅是可能不被采纳的性能提示，不改变一致性。`.report_mechanism` 可在 mbarrier 上附加数据有效性报告。复制本身是 weak memory operation，mbarrier complete-tx 在 `.cluster` scope 有 `.release` 语义。

原文 Notes 与 Target ISA Notes 记录了 `.shared::cluster.global` 的目标架构性能建议、基础 `sm_90` 要求，以及 gather/scatter、im2col、CTA group、multicast、override、report 等 qualifier 各自的架构条件；这些列表与示例均保留英文原文。

###### 重点解读

这条指令同时涉及三个对象：tensor map 给出布局和边界等属性，`tensorCoords` 给出本次 tile 的位置，mbarrier 或 bulk group 给出完成跟踪。global-to-shared 的 mbarrier 保证复制结果，而可选的 bulk async-group 只可用于判断 descriptor 读取完成；后者不能替代等待目标 shared 数据完成。远端 barrier、`.cta_group::2` 和 multicast 叠加时，必须分别核对目标 CTA 与收到 signal 的 CTA。

##### 9.7.10.28.5.4 Data Movement and Conversion Instructions: `cp.reduce.async.bulk.tensor`

###### English Original

cp.reduce.async.bulk.tensor

Initiates an asynchronous reduction operation on the tensor data.

**Syntax**

```ptx
// shared::cta -> global:
cp.reduce.async.bulk.tensor.dim.dst.src.redOp{.load_mode}.completion_mechanism{.level::cache_hint}{.override::global_address}{.override_attribute}
                                          [tensorMap{, gAddrToOverride}{, attributeOverrideInfo}, tensorCoords], [srcMem] {,cache_policy}

.dst =                  { .global }
.src =                  { .shared::cta }
.dim =                  { .1d, .2d, .3d, .4d, .5d }
.completion_mechanism = { .bulk_group }
.load_mode =            { .tile, .im2col_no_offs, .im2col_no_offs::w }
.redOp =                { .add, .min, .max, .inc, .dec, .and, .or, .xor}
.override_attribute =   { .override::global_dim, .override::global_dim_stride }
```

**Description**

cp.reduce.async.bulk.tensor is a non-blocking instruction which initiates an asynchronous reduction operation of tensor data in the .dst state space with tensor data in the .src state space.

The operand srcMem specifies the location of the tensor data in the .src state space using which the reduction operation has to be performed.

The operand tensorMap is the generic address of the opaque tensor-map object which resides in .param space or .const space or .global space. The operand tensorMap specifies the properties of the tensor copy operation, as described in Tensor-map. The tensorMap is accessed in tensormap proxy. Refer to the CUDA programming guide for creating the tensor-map objects on the host side.

Each element of the tensor data in the .dst state space is reduced inline with the corresponding element from the tensor data in the .src state space. The modifier .redOp specifies the reduction operation used for the inline reduction. The type of each tensor data element in the source and the destination tensor is specified in Tensor-map.

The dimension of the tensor is specified by the .dim modifier.

The vector operand tensorCoords specifies the starting coordinates of the tensor data in the global memory on which the reduce operation is to be performed. The number of tensor coordinates in the vector argument tensorCoords should be equal to the dimension specified by the modifier .dim. The individual tensor coordinates are of the type .s32.

The following table describes the valid combinations of .redOp and element type:
| `.redOp` | Element type |
|---|---|
| `.add` | `.u32`, `.s32`, `.u64`, `.f32`, `.f16`, `.bf16` |
| `.min`, `.max` | `.u32`, `.s32`, `.u64`, `.s64`, `.f16`, `.bf16` |
| `.inc`, `.dec` | `.u32` |
| `.and`, `.or`, `.xor` | `.b32`, `.b64` |


The modifier .completion_mechanism specifies the completion mechanism that is supported on the instruction variant. Value .bulk_group of the modifier .completion_mechanism specifies that cp.reduce.async.bulk.tensor instruction uses bulk async-group based completion mechanism.

The qualifier .load_mode specifies how the data in the source location is copied into the destination location. If .load_mode is not specified, it defaults to .tile. In .tile mode, the multi-dimensional layout of the source tensor is preserved at the destination. In .im2col_no_offs and .im2col_no_offs::w modes, some dimensions of the source tensors are unrolled in a single dimensional column at the destination. Details of the im2col mode are described in im2col mode. In .im2col mode, the tensor has to be at least 3-dimensional.

The optional qualifiers .override::global_address and .override_attribute support value override functionality for various tensor properties present in the tensorMap object. For more details refer to the Overriding tensor property value section.

When the optional argument cache_policy is specified, the qualifier .level::cache_hint is required. The 64-bit operand cache_policy specifies the cache eviction policy that may be used during the memory access.

cache_policy is a hint to the cache subsystem and may not always be respected. It is treated as a performance hint only, and does not change the memory consistency behavior of the program. The qualifier .level::cache_hint is only supported when at least one of the .src or .dst statespaces is .global state space.

Each reduction operation performed by cp.reduce.async.bulk.tensor has individually .relaxed.gpu memory ordering semantics and are element-wise atomic. The load operations in cp.reduce.async.bulk.tensor are treated as weak memory operations and the complete-tx operation on the mbarrier has .release semantics at the .cluster scope as described in the Memory Consistency Model.

**PTX ISA Notes**

Introduced in PTX ISA version 8.0.

Support for qualifiers .override::global_address and .override_attribute introduced in PTX ISA version 9.4.

Support for qualifier .im2col_no_offs::w introduced in PTX ISA version 9.4.

**Target ISA Notes**

Requires sm_90 or higher.

Qualifiers .override::global_address and .override_attribute are supported on following architectures:

sm_107f or higher in the same family

Qualifier .im2col_no_offs::w is supported on following family-specific architectures:

sm_107f or higher in the same family

**Examples**

```ptx
.reg .b8  newTensorSize_<2>;
.reg .u64 gAddrToOverride<3>;

cp.reduce.async.bulk.tensor.1d.global.shared::cta.add.tile.bulk_group
                                             [tensorMap0, {tc0}], [sMem0];

cp.reduce.async.bulk.tensor.2d.global.shared::cta.and.bulk_group.L2::cache_hint
                                             [tensorMap1, {tc0, tc1}], [sMem1] , policy;

cp.reduce.async.bulk.tensor.3d.global.shared::cta.xor.im2col.bulk_group
                                             [tensorMap2, {tc0, tc1, tc2}], [sMem2];

cp.reduce.async.bulk.tensor.global.shared::cta.bulk_group.2d.tile.min.override::global_address
                                             [tensorMap3, gAddrToOverride1, {tc0, tc1}], [sMem3];

cp.reduce.async.bulk.tensor.global.shared::cta.bulk_group.1d.max.L2::cache_hint.override::global_address.override::global_dim
                                             [tensorMap4, gAddrToOverride2, {newTensorSize_0}, {0}], [sMem4], policy;

cp.reduce.async.bulk.tensor.3d.global.shared::cta.or.im2col_no_offs::w.bulk_group
                                             [tensorMap5, {tc0, tc1, tc2}], [sMem5];
```

###### 中文翻译

`cp.reduce.async.bulk.tensor` 以 `.shared::cta` 中的源 tensor 元素，对 `.global` 目标 tensor 的对应元素执行异步逐元素归约；`.redOp` 选择运算，元素类型由 `tensorMap` 提供。`srcMem` 指向源，`tensorMap` 可位于 `.param`、`.const` 或 `.global`，并通过 tensormap proxy 访问。`.dim` 指定维度，`.s32` 的 `tensorCoords` 长度应等于 `.dim`，指出 global tensor 中开始归约的位置。

合法 `.redOp`/元素类型组合完整列于英文表格。`.bulk_group` 指定 bulk async-group completion。`.load_mode` 默认 `.tile`，保持多维布局；`.im2col_no_offs` 和 `.im2col_no_offs::w` 则把源的部分维度展成目标的一维列，相关 im2col 模式要求至少三维。可用 `.override::global_address` 与 `.override_attribute` 覆盖 tensor map 属性，约束见上文。

指定可选的 64-bit `cache_policy` 时需要 `.level::cache_hint`；该 hint 仅用于至少一侧为 `.global` 的方向，是可能不被采纳的性能提示，不改变 memory consistency。每个目标端 reduction 具有 `.relaxed.gpu` 逐元素原子语义；源端 load 是 weak memory operation。原文还说 mbarrier 的 complete-tx 在 `.cluster` scope 具有 `.release` 语义。目标 ISA 至少要求 `sm_90`，override 与较新的 im2col 模式另有 `sm_107f` 同族限制。

###### 重点解读

这是“带 tensor map 的异步归约”，不是简单 tensor copy。逐元素 atomic 指的是目标端归约；源端 load 仍为 weak。原文最后的 mbarrier release 句与本指令仅列 `.bulk_group` 的语法不完全对应，笔记按原文保留；实现时按具体变体核对完成协议。

##### 9.7.10.28.5.5 Data Movement and Conversion Instructions: `cp.async.bulk.prefetch.tensor`

###### English Original

cp.async.bulk.prefetch.tensor

Provides a hint to the system to initiate the asynchronous prefetch of tensor data to the cache.

**Syntax**

```ptx
// global -> L2:
cp.async.bulk.prefetch.tensor.dim.L2.src{.load_mode}{.level::cache_hint}{.override::global_address}{.override_attribute}
                                                             [tensorMap{, gAddrToOverride}{, attributeOverrideInfo}, tensorCoords]
                                                             {, im2colInfo } {, cache_policy}

cp.async.bulk.prefetch.tensor.dim.L2.src{.load_mode}{.level::eviction_priority}{.override::global_address}{.override_attribute}
                                                             [tensorMap{, gAddrToOverride}{, attributeOverrideInfo}, tensorCoords]
                                                             {, im2colInfo }

.src =                { .global }
.dim =                { .1d, .2d, .3d, .4d, .5d }
.load_mode =          { .tile, .tile::gather4, .im2col, .im2col::w, .im2col::w::128 }
.level::cache_hint =  { .L2::cache_hint }
.level::eviction_priority = { .L2::evict_last }
.override_attribute =  { .override::global_dim, .override::global_dim_stride }
```

**Description**

cp.async.bulk.prefetch.tensor is a non-blocking instruction which may initiate an asynchronous prefetch of tensor data from the location in .src statespace to the L2 cache.

The operand tensorMap is the generic address of the opaque tensor-map object which resides in .param space or .const space or .global space. The operand tensorMap specifies the properties of the tensor copy operation, as described in Tensor-map. The tensorMap is accessed in tensormap proxy. Refer to the CUDA programming guide for creating the tensor-map objects on the host side.

The dimension of the tensor data is specified by the .dim modifier.

The vector operand tensorCoords specifies the starting coordinates in the tensor data in the global memory from which the copy operation has to be performed. The individual tensor coordinates in tensorCoords are of type .s32. The format of vector argument tensorCoords is dependent on .load_mode specified and is as follows:
| `.load_mode` | `tensorCoords` | Semantics |
|---|---|---|
| `.tile::gather4` | `{col_idx, row_idx0, row_idx1, row_idx2, row_idx3}` | Fixed length vector of size 5. The five elements together specify the start co-ordinates of the four rows. |
| Rest all | `{d0, .., dn}` for `n = .dim` | Vector of `n` elements where `n = .dim`. The elements indicate the offset in each of the dimension. |


The qualifier .load_mode specifies how the data in the source location is copied into the destination location. If .load_mode is not specified, it defaults to .tile.

In .tile mode, the multi-dimensional layout of the source tensor is preserved at the destination. In .tile::gather4 mode, four rows in the 2-dimensional source tensor are fetched to L2 cache. Details of .tile::gather4 modes are described in .tile::scatter4 and .tile::gather4 modes.

In .im2col and .im2col::* modes, some dimensions of the source tensors are unrolled in a single dimensional column at the destination. Details of the im2col and .im2col::* modes are described in im2col mode and im2col::w, im2col_no_offs::w and im2col::w::128 modes respectively. In .im2col and .im2col::* modes, the tensor has to be at least 3-dimensional. The vector operand im2colInfo can be specified only when .load_mode is .im2col or .im2col::w or .im2col::w::128. The format of the vector argument im2colInfo is dependent on the exact im2col mode and is as follows:
| Exact im2col mode | `im2colInfo` argument | Semantics |
|---|---|---|
| `.im2col` | `{ i2cOffW , i2cOffH , i2cOffD }` for `.dim = .5d` | A vector of im2col offsets whose vector size is two less than number of dimensions `.dim`. |
| `.im2col::w` | `{ wHalo, wOffset }` | A vector of 2 arguments containing `wHalo` and `wOffset` arguments. |
| `.im2col::w::128` | `{ wHalo, wOffset }` | A vector of 2 arguments containing `wHalo` and `wOffset` arguments. |


The optional qualifiers .override::global_address and .override_attribute support value override functionality for various tensor properties present in the tensorMap object. For more details refer to the Overriding tensor property value section.

When the optional argument cache_policy is specified, the qualifier .level::cache_hint is required. The 64-bit operand cache_policy specifies the cache eviction policy that may be used during the memory access.

cache_policy is a hint to the cache subsystem and may not always be respected. It is treated as a performance hint only, and does not change the memory consistency behavior of the program.

Qualifier .level::eviction_priority specifies the eviction priority to be applied for the prefetched cache line.

cp.async.bulk.prefetch.tensor is treated as a weak memory operation in the Memory Consistency Model.

**PTX ISA Notes**

Introduced in PTX ISA version 8.0.

Support for qualifier .tile::gather4 introduced in PTX ISA version 8.6.

Support for qualifiers .im2col::w and .im2col::w::128 introduced in PTX ISA version 8.6.

Support for qualifier .level::eviction_priority introduced in PTX ISA version 9.4.

Support for qualifiers .override::global_address and .override_attribute introduced in PTX ISA version 9.4.

**Target ISA Notes**

Requires sm_90 or higher.

Qualifier .tile::gather4 is supported on following architectures:

sm_100a

sm_101a (Renamed to sm_110a from PTX ISA version 9.0)

And is supported on following family-specific architectures from PTX ISA version 8.8:

sm_100f or higher in the same family

sm_101f or higher in the same family (Renamed to sm_110f from PTX ISA version 9.0)

sm_110f or higher in the same family

Qualifiers .im2col::w and .im2col::w::128 are supported on following architectures:

sm_100a

sm_101a (Renamed to sm_110a from PTX ISA version 9.0)

And are supported on following family-specific architectures from PTX ISA version 8.8:

sm_100f or higher in the same family

sm_101f or higher in the same family (Renamed to sm_110f from PTX ISA version 9.0)

sm_110f or higher in the same family

Qualifier .level::eviction_priority is supported on following family-specific architectures:

sm_107f or higher in the same family

Qualifiers .override::global_address and .override_attribute are supported on following architectures:

sm_107f or higher in the same family

**Examples**

```ptx
.reg .b16 ctaMask, im2colwHalo, im2colOff;
.reg .u16 i2cOffW, i2cOffH, i2cOffD;
.reg .b64 l2CachePolicy;
.reg .b8  newTensorSize_<5>;
.reg .b32 newTensorLowerStrd_<4>;
.reg .b16 tensorUpperStrd;
.reg .u64 gAddrToOverride<3>;

cp.async.bulk.prefetch.tensor.1d.L2.global.tile  [tensorMap0, {tc0}];

@p cp.async.bulk.prefetch.tensor.2d.L2.global    [tensorMap1, {tc0, tc1}];

@p cp.async.bulk.prefetch.tensor.5d.L2.global.im2col
                      [tensorMap2, {tc0, tc1, tc2, tc3, tc4}], {i2cOffW, i2cOffH, i2cOffD};

@p cp.async.bulk.prefetch.tensor.3d.L2.global.im2col.L2::cache_hint
                      [tensorMap3, {tc0, tc1, tc2}], {i2cOffW}, policy;

cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4 [tensorMap5, {col_idx, row_idx0, row_idx1, row_idx2, row_idx3}];

cp.async.bulk.prefetch.tensor.4d.L2.global.im2col::w::128
                      [tensorMap4, {t0, t1, t2, t3}], {im2colwHalo, im2colOff};

cp.async.bulk.prefetch.tensor.3d.L2.global.tile::gather4.L2::evict_last
                      [tensorMap5, {t0, t1, t2}], {i2cOffW};

cp.async.bulk.prefetch.tensor.2d.L2.global.tile.override::global_address
                      [tensorMap6, gAddrToOverride1, {t0, t1}];

cp.async.bulk.prefetch.tensor.3d.L2.global.override::global_address.override::global_dim_stride.L2::cache_hint
                      [tensorMap7, gAddrToOverride2, {newTensorSize_0, newTensorSize_1, newTensorSize_2}, {newTensorLowerStrd_0, newTensorLowerStrd_1}, tensorUpperStrd, {0, 0, 0}], policy;
```

###### 中文翻译

`cp.async.bulk.prefetch.tensor` 是 non-blocking tensor 预取提示：系统可能依据 `.global` 中的 tensor 数据与 tensor map 属性，异步把相应数据预取到 L2。`tensorMap` 是位于 `.param`、`.const` 或 `.global` 的 opaque object，通过 tensormap proxy 访问；`.dim` 指定维度，`.s32` 的 `tensorCoords` 指出 global tensor 的起点。`.tile::gather4` 使用五元素坐标，其余模式使用与 `.dim` 一致的坐标向量，具体形式见原文表格。

`.load_mode` 默认 `.tile`，保持多维布局；`.tile::gather4` 把二维 tensor 的四行预取至 L2。`.im2col` 和 `.im2col::*` 则以列展开方式选择数据，要求 tensor 至少三维；只有 `.im2col`、`.im2col::w`、`.im2col::w::128` 可附加 `im2colInfo`，参数形式保留在英文表格。

可选 override qualifier 按前述规则覆盖 tensor map 的属性。指定 64-bit `cache_policy` 时必须带 `.level::cache_hint`；该策略可能不被执行，不改变一致性。`.level::eviction_priority` 指定预取 cache line 的驱逐优先级。该指令在 Memory Consistency Model 中属于 weak memory operation。基础目标 ISA 至少是 `sm_90`，gather、im2col、eviction priority 和 override 各自有更具体的架构条件，详见保留的英文原文。

###### 重点解读

Prefetch 的 `may initiate` 表明它只是可选缓存提示，不能用它推断 tensor 数据已经抵达 L2，也不能替代真正的 copy 或 reduction。override 改变本次访问采用的属性，仍需满足上一小节关于地址、维度和模式的约束。

#### 9.7.10.28.6 Data Movement and Conversion Instructions: Bulk and Tensor Copy Completion Instructions

##### 9.7.10.28.6.1 Data Movement and Conversion Instructions: `cp.async.bulk.commit_group`

###### English Original

> `cp.async.bulk.commit_group`
>
> Commits all prior initiated but uncommitted `cp.async.bulk` instructions into a *cp.async.bulk-group*.

**Syntax**

```ptx
cp.async.bulk.commit_group;
```

**Description**

> `cp.async.bulk.commit_group` instruction creates a new per-thread *bulk async-group* and batches all prior `cp{.reduce}.async.bulk{.prefetch}{.tensor}`/`applypriority.async.bulk{.tensor}` instructions satisfying the following conditions into the new *bulk async-group*:

- The prior `cp{.reduce}.async.bulk{.prefetch}{.tensor}`/`applypriority.async.bulk{.tensor}` instructions use *bulk_group* based completion mechanism, and
- They are initiated by the executing thread but not committed to any *bulk async-group*.

> If there are no uncommitted `cp{.reduce}.async.bulk{.prefetch}{.tensor}`/`applypriority.async.bulk{.tensor}` instructions then `cp.async.bulk.commit_group` results in an empty *bulk async-group*.
>
> An executing thread can wait for the completion of all `cp{.reduce}.async.bulk{.prefetch}{.tensor}`/`applypriority.async.bulk{.tensor}` operations in a *bulk async-group* using `cp.async.bulk.wait_group`.
>
> There is no memory ordering guarantee provided between any two `cp{.reduce}.async.bulk{.prefetch}{.tensor}`/`applypriority.async.bulk{.tensor}` operations within the same *bulk async-group*.

**PTX ISA Notes**

> Introduced in PTX ISA version 8.0.

**Target ISA Notes**

> Requires `sm_90` or higher.

**Examples**

```ptx
cp.async.bulk.commit_group;
```

###### 中文翻译

`cp.async.bulk.commit_group` 将此前已发起、尚未提交的 `cp.async.bulk` 指令归入一个 `cp.async.bulk-group`。

该指令为当前 thread 创建新的 bulk async-group，并收集此前发起的 `cp{.reduce}.async.bulk{.prefetch}{.tensor}` / `applypriority.async.bulk{.tensor}` 操作，但必须同时满足两个条件：操作采用 `.bulk_group` completion mechanism，且由当前执行 thread 发起、尚未归入任何 bulk async-group。如果不存在符合条件的操作，也会创建一个空 group。

执行 thread 可以用 `cp.async.bulk.wait_group` 等待 group 内全部操作完成。同一 bulk async-group 中的两个操作之间不保证 memory ordering。该指令于 PTX ISA 8.0 引入，目标 ISA 要求 `sm_90` 或更高。

###### 重点解读

标题中的 `cp.async.bulk` 是简称；实际收集范围由“是否使用 `.bulk_group` completion”决定，也包括符合条件的 reduction、tensor、prefetch 与 `applypriority` 变体。`commit_group` 只是划定当前 thread 的批次，不表示完成，也不在同一批次的操作之间建立顺序。它与前述非 bulk 的 `cp.async.commit_group` 属于两套不同的 group。

##### 9.7.10.28.6.2 Data Movement and Conversion Instructions: `cp.async.bulk.wait_group`

###### English Original

> `cp.async.bulk.wait_group`
>
> Wait for completion of *bulk async-groups*.

**Syntax**

```ptx
cp.async.bulk.wait_group{.read} N;
```

**Description**

> `cp.async.bulk.wait_group` instruction will cause the executing thread to wait until only N or fewer of the most recent *bulk async-groups* are pending and all the prior *bulk async-groups* committed by the executing threads are complete. For example, when N is 0, the executing thread waits on all the prior *bulk async-groups* to complete. Operand N is an integer constant.
>
> By default, `cp.async.bulk.wait_group` instruction will cause the executing thread to wait until completion of all the bulk async operations in the specified *bulk async-group*. A bulk async operation includes the following:

- Optionally, reading from the tensormap.
- Reading from the source locations.
- Writing to their respective destination locations.
- Writes being made visible to the executing thread.

> The optional `.read` modifier indicates that the waiting has to be done until all the bulk async operations in the specified *bulk async-group* have completed:

1. reading from the tensormap
2. the reading from their source locations.

**PTX ISA Notes**

> Introduced in PTX ISA version 8.0.

**Target ISA Notes**

> Requires `sm_90` or higher.

**Examples**

```ptx
cp.async.bulk.wait_group.read   0;
cp.async.bulk.wait_group        2;
```

###### 中文翻译

`cp.async.bulk.wait_group` 等待 bulk async-group 的完成。`cp.async.bulk.wait_group N` 使执行 thread 等到最多只剩最近的 `N` 个 group 仍未完成，而更早提交的 group 全部完成。`N` 是整数立即数；`N = 0` 时，等待此前所有 group 完成。

默认形式等待指定 group 内的 bulk async operation 完整结束，包括可选地读取 tensor map、读取源位置、写入目标位置，以及使写入对执行 thread 可见。

可选的 `.read` 形式只需等到这些操作完成 tensor map 读取和源位置读取，不要求目标写入及其可见性已经完成。该指令于 PTX ISA 8.0 引入，目标 ISA 要求 `sm_90` 或更高。示例分别展示 `wait_group.read 0` 与默认形式的 `wait_group 2`。

###### 重点解读

`N` 是允许继续在途的“最近 group 数量”，不是 group 编号。默认 `wait_group 0` 可作为后续读取目标结果的完成点；`wait_group.read 0` 只确认此前各 group 的读取阶段结束，不能据此读取尚未完成写入的目标数据。`.read` 适合区分输入资源的读取生命周期与输出结果的完成时刻。和 `commit_group` 一样，这里只跟踪 bulk async-group，不能替代非 bulk 的 `cp.async.wait_group`，也不是跨 thread barrier。

