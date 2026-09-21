# NVSHMEM Execution Model：官方英文、中文对照与技术解读

## 文档范围与版本

本文整理 NVIDIA NVSHMEM API Guide 的 **Execution Model** 页面，覆盖 PE/SPMD、NVSHMEM phase、初始化与结束、PE identifier、communication progress、参数重叠和 buffer in-use 规则。英文部分保留官方原文；中文部分提供逐段翻译；“技术解读”负责把规范连接到 B00、B03、B04 与 C00，不属于 NVIDIA 原文。

用户粘贴内容中 `nvshmem_ptr` 示例的数学变量在复制时丢失。本文于 2026-09-18 对照当前滚动 `latest` 页面核验后，恢复为官方页面显示的 `a`、`A` 和 `i`；除此之外不改写英文语义。滚动页面说明 API 契约，不固定某个 proxy、transport、GPU handler 或 progress engine 实现。

来源：[NVIDIA NVSHMEM API Guide — Execution Model](https://docs.nvidia.com/nvshmem/api/latest/gen/exec-model.html)。

## Execution Model

### Official English

> An NVSHMEM program consists of a set of NVSHMEM processes called PEs. While not required by NVSHMEM, in typical usage, PEs are executed using a single program, multiple data (SPMD) model. SPMD requires each PE to use the same executable; however, PEs are able to follow divergent control paths. PEs are implemented using OS processes and PEs are permitted to create additional threads, when threading support is enabled.
>
> PE execution is loosely coupled, relying on NVSHMEM operations to communicate and synchronize among executing PEs. The NVSHMEM phase in a program begins with a call to the initialization routine `nvshmem_init` or `nvshmem_init_thread`, which must be performed before using any of the other NVSHMEM library routines. An NVSHMEM program concludes its use of the NVSHMEM library when all PEs call `nvshmem_finalize` or any PE calls `nvshmem_global_exit`. During a call to `nvshmem_finalize`, the NVSHMEM library must complete all pending communication and release all the resources associated to the library using an implicit collective synchronization across PEs. Calling any NVSHMEM routine before initialization or after `nvshmem_finalize` leads to undefined behavior. After finalization, a subsequent initialization call also leads to undefined behavior.
>
> The PEs of the NVSHMEM program are identified by unique integers. The identifiers are integers assigned in a monotonically increasing manner from zero to one less than the total number of PEs. PE identifiers are used for NVSHMEM calls (for example, to specify put or get routines on symmetric data objects, collective synchronization calls) or to dictate a control flow for PEs using constructs of C. The identifiers are fixed for the duration of the NVSHMEM phase of a program.

### 中文对照

> 一个 NVSHMEM 程序由一组称为 PE 的 NVSHMEM 进程组成。NVSHMEM 并不强制要求 SPMD，但典型用法采用 single program, multiple data 模型：每个 PE 使用同一个可执行文件，同时允许各 PE 沿不同控制路径执行。PE 由操作系统进程实现；启用线程支持时，PE 可以创建额外线程。
>
> 各 PE 的执行是松耦合的，运行中的 PE 依靠 NVSHMEM 操作进行通信和同步。程序的 NVSHMEM phase 从调用 `nvshmem_init` 或 `nvshmem_init_thread` 开始；在此之前不得使用其他 NVSHMEM library routines。当所有 PE 调用 `nvshmem_finalize`，或任意 PE 调用 `nvshmem_global_exit` 时，程序结束对 NVSHMEM 库的使用。`nvshmem_finalize` 调用期间，NVSHMEM 库必须借助 PE 间隐式集合同步完成所有 pending communication，并释放与库关联的全部资源。在初始化之前或 `nvshmem_finalize` 之后调用任何 NVSHMEM routine 都会导致未定义行为；finalize 后再次初始化同样属于未定义行为。
>
> NVSHMEM 程序中的 PE 由唯一整数标识，编号从 0 单调排列到 PE 总数减 1。PE identifier 用于 NVSHMEM 调用，例如指定 symmetric data object 上 put/get 的目标或参与 collective synchronization，也可以用于 C 控制结构，使不同 PE 进入不同控制路径。PE identifier 在整个 NVSHMEM phase 内保持不变。

### 技术解读：SPMD 相同程序不等于相同控制流

PE 首先是 NVSHMEM 进程级通信参与者，不是 CUDA thread、warp 或 block。SPMD 表示各 PE 通常启动同一个 executable，但它们可以按 PE id、输入数据或算法状态走不同分支。分支本身合法；只有当分支破坏 collective matching、对称分配顺序或必须由所有 PE 参与的生命周期调用时，才产生错误。

NVSHMEM phase 是严格生命周期边界。初始化不仅取得 PE id，还可能准备 bootstrap、GPU binding、symmetric heap、transport 和 device state；finalize 具有隐式 collective synchronization，并负责完成 pending communication 和释放资源。因此不能把 finalize 当成单 PE 的普通析构，也不能在部分 PE 已 finalize 后继续从其他 PE 发出 NVSHMEM 操作。`global_exit` 与正常 collective finalize 的控制语义不同，不能用它证明 pending work 已按正常协议排空。

## Progress of NVSHMEM Operations

### Official English

> The NVSHMEM model assumes that computation and communication are naturally overlapped. NVSHMEM programs are expected to exhibit progression of communication both with and without NVSHMEM calls. Consider a PE that is engaged in a computation with no NVSHMEM calls. Other PEs should be able to communicate (for example, put, get, atomic, and so on) and complete communication operations with that computationally-bound PE without that PE issuing any explicit NVSHMEM calls. One-sided NVSHMEM communication calls involving that PE should progress regardless of when that PE next engages in an NVSHMEM call.

### 中文对照

> NVSHMEM 模型假定计算和通信可以自然重叠。无论程序是否正在调用 NVSHMEM，通信都应能够取得进展。设某个 PE 正在执行计算且没有调用任何 NVSHMEM routine，其他 PE 仍应能够与这个忙于计算的 PE 通信，例如执行 put、get 或 atomic，并完成相关通信；计算中的目标 PE 不需要为此显式调用 NVSHMEM。涉及该 PE 的 one-sided NVSHMEM communication 应当取得进展，而不取决于该 PE 下一次何时进入 NVSHMEM 调用。

### 技术解读：目标 PE 不主动调用，不等于所有发起方式都自动前进

这段契约强调 one-sided communication 不要求目标应用代码配对接收或周期性进入 NVSHMEM。它不能被扩大为“任何 CUDA scheduling、stream dependency 和通信路径都会无条件前进”。如果发起侧 device kernel 因 residency 或循环等待无法执行到提交点，或者两个 streams/PEs 形成依赖环，操作尚未真正发起，目标无须参与的 progress 保证也无法解除该死锁。

“naturally overlapped”描述模型允许 computation-bound target 与远端通信并存，不等于每个程序都会获得性能收益。实际 overlap 还取决于 transport：CPU proxy 需要 host progress resource，IBGDA 需要 GPU 构造请求并驱动 NIC，同节点 P2P 又可能占用 SM/L2/HBM。是否缩短端到端时间必须按 E01 测量。

## Invoking NVSHMEM Operations：参数不得非法重叠

### Official English

> Pointer arguments to NVSHMEM routines that point to non-`const` data must not overlap in memory with other arguments to the same NVSHMEM operation, with the exception of in-place reductions as described in Section NVSHMEM_REDUCTIONS. Otherwise, the behavior is undefined. Two arguments overlap in memory if any of their data elements are contained in the same physical memory locations. For example, consider an address $a$ returned by the `nvshmem_ptr` operation for symmetric object $A$ on PE $i$. Providing the local address $a$ and the symmetric address of object $A$ to an NVSHMEM operation targeting PE $i$ results in undefined behavior.

### 中文对照

> NVSHMEM routine 中指向非 `const` 数据的 pointer argument，不得与同一次 NVSHMEM operation 的其他参数发生内存重叠；`NVSHMEM_REDUCTIONS` 所规定的 in-place reduction 是例外。否则行为未定义。只要两个参数的任意数据元素落在相同物理内存位置，就构成 overlap。例如，设 `nvshmem_ptr` 为 PE $i$ 上的 symmetric object $A$ 返回本地地址 $a$。若某个以 PE $i$ 为目标的 NVSHMEM operation 同时接收本地地址 $a$ 和对象 $A$ 的 symmetric address，则行为未定义。

### 技术解读：不同指针数值也可能别名到相同物理内存

重叠判断依据是 physical memory，而不是两个 pointer value 是否相等。`nvshmem_ptr(A, i)` 返回的 peer direct pointer 与本地 symmetric address `A` 可能以不同虚拟地址映射同一远端物理对象；把二者作为同一 operation 的读写参数会形成未定义 alias。审查 in-place 操作时应首先确认 API 是否明确允许，而不是因为 source/destination 看起来处于不同地址空间就自行推断安全。

## Invoking NVSHMEM Operations：buffer in-use 生命周期

### Official English

> Buffers provided to NVSHMEM routines are in-use until the corresponding NVSHMEM operation has completed at the calling PE. Updates to a buffer that is in-use, including updates performed through locally and remotely issued NVSHMEM operations, result in undefined behavior. Similarly, reads from a buffer that is in-use are allowed only when the buffer was provided as a `const`-qualified argument to the NVSHMEM routine for which it is in-use. Otherwise, the behavior is undefined. Exceptions are made for buffers that are in-use by AMOs, as described in Section Atomicity Guarantees. For information regarding the completion of NVSHMEM operations, see Section Memory Ordering.

### 中文对照

> 传给 NVSHMEM routine 的 buffer 在相应操作于 calling PE 完成之前都处于 **in-use** 状态。更新一个仍处于 in-use 状态的 buffer 会导致未定义行为；这种更新既包括本地发起的操作，也包括远端发起的 NVSHMEM 操作。类似地，只有当 buffer 作为相应 NVSHMEM routine 的 `const`-qualified 参数传入时，才允许在 in-use 期间读取它；否则读取同样是未定义行为。AMO 正在使用的 buffer 适用 Atomicity Guarantees 所规定的例外。关于 NVSHMEM operation 的完成条件，应查阅 Memory Ordering。

### 技术解读：in-use 是访问纪律，不是一个统一时间点

调用者必须按具体操作的 local completion 契约判断 buffer 何时退出 in-use。Blocking PUT 返回、NBI PUT 经适当 flush/quiet、NBI GET 经 quiet，可能对应不同边界；不能把“API 已返回”普遍视为 buffer 可重写。更重要的是，这条规则也覆盖远端发起的更新：某对象作为本次 operation 的非 `const` 参数仍在使用时，另一个 PE 并发 PUT 到同一物理区域同样可能造成未定义行为。

`const` 规则描述操作期间允许的访问方向。例如 PUT source 作为只读输入，应用可否并发读取仍要符合该 routine 的参数限定和线程模型；但任何写入都会改变通信层正读取的数据。目标端 slot 的应用消费生命周期还要更长：calling PE 上的 operation completion 不证明远端 consumer 已经读完，因此 B04 的 ready/ack ownership 仍然必要。

## Invoking NVSHMEM Operations：多个 symmetric segment

### Official English

> NVSHMEM routines with multiple symmetric object arguments do not require these symmetric objects to be located within the same symmetric memory segment. For example, objects located in the symmetric data segment and objects located in the symmetric heap can be provided as arguments to the same NVSHMEM operation.

### 中文对照

> 对于具有多个 symmetric object 参数的 NVSHMEM routine，这些 symmetric objects 不要求位于同一个 symmetric memory segment。例如，位于 symmetric data segment 的对象和位于 symmetric heap 的对象可以同时作为同一次 NVSHMEM operation 的参数。

### 技术解读：对象资格一致，不要求底层 segment 相同

应用要证明的是每个参数各自具有合法 symmetric-object 身份、范围、对齐和生命周期，而不是证明它们来自同一次 heap allocation。Runtime 可以为不同 segment 维护不同注册或地址 metadata；该事实对应用透明，也不能据此推断多 segment operation 的具体性能或底层 WQE 数量。

## 与课程正文的对应关系

| 官方章节 | 课程正文 | 应带走的边界 |
| --- | --- | --- |
| PE、SPMD 与 identifier | [B00：NVSHMEM 编程模型](../b00-nvshmem-programming-model.md) | PE 是 OS process 级参与者；同一 executable 允许分支，但 collective 必须匹配 |
| NVSHMEM phase 与 finalize | [C00：初始化与控制平面](../c00-runtime-initialization-and-control-plane.md) | init/finalize 定义合法调用生命周期；finalize 是隐式集合操作并完成 pending communication |
| Progress of Operations | [B03：CUDA 执行与前进性](../b03-cuda-execution-and-cooperation.md) | 目标应用无须配对调用，但发起、调度和依赖环仍需单独证明 |
| Argument overlap 与 buffer in-use | [B04：发布与缓冲区所有权](../b04-data-publishing-and-buffer-ownership.md) | 物理别名可能发生在不同指针值之间；退出 in-use 依赖具体 completion 契约 |
| Multiple symmetric segments | [B01：对称对象与远端寻址](../b01-symmetric-objects-and-remote-addressing.md) | 多参数可来自不同 symmetric segment，每个对象仍需独立满足合法性条件 |

## 自检边界

本文只完成官方英文保存、中文翻译与静态语义解读，没有运行多进程、线程支持、progress、alias 或 buffer-race 实验。用户粘贴中缺失的 `a/A/i` 变量依据 2026-09-18 当前官方页面恢复并明确记录，没有补造其他原文内容。
