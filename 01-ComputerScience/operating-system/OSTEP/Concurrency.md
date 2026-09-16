# 对话

并发是操作系统核心模块，多线程共享内存时若无访问协调会产生运行错误；串行访问可保证结果正确但效率低下，最优方案需兼顾正确性与性能。操作系统层面学习并发有两点意义：一是提供锁、条件变量等机制支持多线程程序，二是操作系统本身属于并发程序，共享内存管控失误会引发严重故障。

Concurrency is a core module of operating systems. Uncoordinated access to shared memory by multiple threads leads to program errors. Serial access ensures correctness at the cost of efficiency, so an optimal solution needs both accuracy and high performance. We study concurrency in OS for two reasons: OS supplies primitives like locks and condition variables for multi-threaded apps, and OS itself is concurrent; mismanaged shared memory will cause serious system failures.

# 26 Concurrency: An Introduction 并发线程导论知识笔记

## 一、线程基础核心概念

**线程**是进程内部新增的执行抽象，单进程可承载多条独立执行流，同一进程内全部线程共享统一地址空间，能够直接读写同一份共享数据。
线程拥有**私有独立资源**：专属程序计数器 PC、独立通用寄存器组、专属线程栈，栈中存放局部变量、函数入参与返回地址；进程之间资源完全隔离，线程仅栈与寄存器为私有数据。
**TCB 线程控制块**用于存储单一线程完整上下文；进程切换时需要更换页表，而同进程内部线程切换无需修改页表，仅完成寄存器状态的保存与恢复。
地址空间布局存在明显区分：单线程进程仅分配单一栈空间，多线程进程为每一条线程单独划分栈区域，会持续占用进程地址空间，重度递归场景下极易出现栈空间耗尽问题。

Thread is a new execution abstraction inside a process. A single process can carry multiple independent execution flows, and all threads within the same process share a unified address space to directly read and write the same shared data.

Threads own private independent resources: exclusive Program Counter (PC), independent general register groups, dedicated thread stacks storing local variables, function parameters and return addresses. Resources are fully isolated between processes, while only stacks and registers are private among threads.

**TCB Thread Control Block** stores the complete context of a single thread. Page tables must be replaced during process context switches, whereas thread switches within one process do not modify page tables and only save and restore register states.

Clear differences exist in address space layout: a single-threaded process is assigned only one stack, while multi-threaded processes allocate separate stack regions for each thread, continuously occupying the process address space and facing high risks of stack exhaustion under heavy recursion.

![[Fig26.1.png]]

## 二、线程使用两大场景

**并行计算**场景下，多核 CPU 环境可将计算任务拆分至多线程执行，充分利用多核硬件提升整体运算速度，该拆分优化过程称为并行化；多进程间共享数据存在高额开销，共享地址空间的线程更适配数据并行类业务。
**IO 阻塞优化**是线程另一核心用途，单一线程等待磁盘读写、网络收发、缺页异常等 IO 操作时会陷入阻塞；多线程程序中阻塞线程主动让出 CPU，调度器调度处于就绪状态的线程持续执行，实现 IO 操作与计算任务并行执行，该机制被广泛应用于各类服务器程序。
任务选型存在明确标准：逻辑独立、内存数据共享需求极低的业务选用多进程实现；需要高频读写共享内存数据的业务优先采用多线程架构。
In **parallel computing** scenarios, computing tasks can be split into multiple threads on multi-core CPUs to fully leverage multi-core hardware and boost overall computing speed, a splitting optimization named parallelization. Inter-process data sharing brings high overhead, so threads with shared address space better fit data parallel businesses.
**IO blocking mitigation** serves as another core thread usage. A single thread gets blocked waiting for IO operations including disk read/write, network transmission and page faults. Blocked threads yield CPU in multi-threaded programs, and the scheduler runs ready threads to overlap IO operations and computing tasks, a mechanism widely deployed in all kinds of server software.
Clear standards exist for task selection: adopt multi-process architecture for logically independent businesses with minimal memory data sharing demands; prioritize multi-thread architecture for businesses requiring frequent read and write on shared memory data.

## 三、Pthread 线程创建与执行调度

**pthread_t** 作为线程唯一标识，`pthread_create()` 负责生成新线程，配置线程入口函数与传入参数；`pthread_join()` 具备阻塞特性，会挂起主线程直至目标线程执行完毕。
线程调度具备**无序性**特征，线程创建完成后没有固定执行优先级，运行时序完全由操作系统调度器决定，先创建的线程无法保证优先运行；多核硬件平台中多条线程能够同时并行执行。
主线程本身属于独立执行线程，完整多线程程序运行期间至少包含主线程与若干工作线程两类执行流。
**pthread_t** acts as the unique identifier of threads. `pthread_create()` spawns new threads and configures thread entry functions and input parameters. `pthread_join()` has blocking features and suspends the main thread until the target thread finishes execution.
Thread scheduling features **unordered execution**. No fixed priority exists after thread creation, and the running sequence is fully determined by OS scheduler. Earlier created threads cannot guarantee prior execution, and multiple threads can run simultaneously on multi-core hardware platforms.
The main thread itself is an independent execution thread. A complete multi-thread program contains at least two types of execution flows: the main thread and several worker threads.

## 四、共享数据引发的并发核心问题

变量自增操作 `counter = counter + 1` 编译后拆解为三条独立硬件指令，依次为内存读取变量、寄存器数值自增、寄存器数据写回内存；操作系统定时器中断能够在任意两条指令之间触发线程上下文切换。

**竞态条件（数据竞争）** 是并发核心故障根源，多条线程同步运行临界区代码读写共享变量，中断与上下文切换的随机时机造成共享变量计算结果每次运行均不相同，程序输出具备不可确定性。

**临界区**指代访问全局变量、共享数据结构等共享资源的代码片段，同一时间仅允许单一线程进入执行。

程序必须满足**互斥访问**约束，保证临界区代码同一时刻至多仅有一条线程执行，从根源消除竞态条件。

开发层面存在**原子性执行**诉求，希望一组关联指令整体执行，仅存在全部执行完成或完全未启动两种对外可见状态，不存在中间暴露的执行阶段；通用硬件不支持复杂逻辑的单条原子指令，同步原语需要依托硬件基础原子指令搭建实现。

The variable increment operation `counter = counter + 1` compiles into three independent hardware instructions: load variable from memory, increment register value, write register data back to memory. OS timer interrupts can trigger thread context switches between any two instructions.

**Race condition (data race)** stands as the core root of concurrent failures. Multiple threads run critical section code to read and write shared variables concurrently, and random timing of interrupts and context switches leads to inconsistent calculation results of shared variables across runs, bringing non-deterministic program output.

**Critical section** refers to code segments accessing shared resources such as global variables and shared data structures, where only one thread is allowed to execute at the same time.

Programs must follow **mutual exclusion** constraints to ensure at most one thread runs critical section code at any moment and eliminate race conditions fundamentally.

Development has demands for **atomic execution**: a group of correlated instructions should execute as a whole, with only two externally visible states: fully completed or never started, without exposed intermediate execution stages. General hardware does not support single atomic instructions for complex logic, so synchronization primitives must be built based on basic hardware atomic instructions.

## 五、并发程序另一类同步需求

系统存在线程等待唤醒同步场景，部分业务逻辑下线程需要主动阻塞，等待其他线程完成指定操作后再恢复运行，典型场景包含磁盘 IO 任务结束、前置工作线程任务完成；后续内容会介绍条件变量机制，完成线程休眠与唤醒的同步控制。

Thread wait-wake synchronization scenarios exist in systems. Under certain business logic, threads need to block actively and resume execution only after other threads complete designated operations, including typical cases like disk IO completion and finish of preposed worker thread tasks. Condition variable mechanisms will be introduced later to realize synchronous control of thread sleep and wakeup.

## 六、关键并发术语标准定义

**临界区 Critical Section**：访问变量、数据结构等共享资源的代码段。
**竞态条件 Race Condition**：多线程同步进入临界区修改共享数据，生成不符合预期的错误运算结果。
**不确定性程序 Indeterminate Program**：内部包含竞态条件，多次运行程序会输出差异化结果。
**互斥 Mutual Exclusion**：同步原语提供的底层保障机制，限制同一时刻仅单一线程执行临界区代码，彻底消除数据竞争。
**Critical Section**: Code segments accessing shared resources including variables and data structures.
**Race Condition**: Multiple threads enter critical sections concurrently to modify shared data and generate unexpected incorrect calculation results.
**Indeterminate Program**: Programs containing race conditions that produce differentiated outputs across multiple runs.
**Mutual Exclusion**: Underlying guarantee mechanism provided by synchronization primitives, restricting only one thread to execute critical section code at the same time and completely eliminate data races.

## 七、操作系统课程学习并发的意义

操作系统是最早大规模落地并发逻辑的软件，硬件中断机制天然引入并发冲突，内核内部全部共享数据结构，涵盖页表、进程链表、文件系统 inode、内存分配位图等，访问逻辑均属于临界区，必须依靠同步原语完成保护。

多线程进程技术普及后，上层应用程序同样需要处理并发同步逻辑，操作系统提供硬件层与软件层双重同步底层支撑，是应用层并发编程的底层基础。

Operating systems are the first software with large-scale deployed concurrent logic. Hardware interrupt mechanisms inherently introduce concurrent conflicts. All internal kernel shared data structures including page tables, process lists, filesystem inodes and memory allocation bitmaps involve critical section access logic that must be protected by synchronization primitives.

After the popularization of multi-threaded process technology, upper-layer application programs also need to handle concurrent synchronization logic. Operating systems provide dual underlying synchronization support on hardware and software layers, serving as the foundation of application-layer concurrent programming.

## 八、工具补充知识点

**objdump** 属于二进制反汇编工具，能够输出程序编译生成的汇编指令，用于分析变量更新操作底层执行流程。
配套仿真工具 **x86.py** 可模拟多线程 x86 指令执行、中断调度、共享内存读写流程，支持自定义中断间隔、线程数量、随机调度种子参数，直观观测竞态条件生成过程与临界区运行行为。
**objdump** is a binary disassembly tool that outputs assembly instructions compiled from programs to analyze low-level execution flow of variable update operations.
The supporting simulator **x86.py** simulates multi-thread x86 instruction execution, interrupt scheduling and shared memory read-write processes. It supports configurable parameters including interrupt intervals, thread counts and random scheduler seeds to directly observe race condition generation and critical section runtime behavior.

# 27 Interlude: Thread API POSIX Thread API 知识笔记

## 一、线程创建接口

**pthread_create** 是 POSIX 标准创建线程的核心函数，函数接收四类参数，参数类型与作用存在明确规范。第一个参数为 `pthread_t` 类型指针，用于存储线程标识，后续等待、操作线程均依赖该标识；第二个参数是线程属性结构体指针，可配置栈大小、调度优先级等特性，传入 NULL 代表使用系统默认属性，自定义属性需提前调用 `pthread_attr_init` 完成初始化；第三个参数为函数指针，规定线程入口函数格式：接收 `void*` 单参数、返回 `void*` 类型指针，void 通用指针的设计允许线程接收任意类型入参、返回任意类型结果；第四个参数是传递给入口函数的实参指针。
多参数传递需要自定义结构体封装多个数值，将结构体地址作为入参传入线程；仅传递单个数字时可直接强制转换 void 指针，无需封装结构体。线程创建成功后会拥有独立调用栈，与进程内其余线程共享同一地址空间。
**pthread_create** is the core POSIX function for spawning threads, which accepts four parameters with fixed definitions. The first parameter is a pointer of `pthread_t` to store thread identifier for subsequent thread operations. The second is an attribute structure pointer to configure stack size, scheduling priority and other features; passing NULL enables default settings, and custom attributes require prior initialization via `pthread_attr_init`. The third is a function pointer defining thread entry rules: the target function takes one `void*` argument and returns a `void*` pointer. Generic void pointers support arbitrary input argument types and return data types. The fourth parameter is the argument pointer passed to the entry function.
Multiple arguments need to be wrapped into a custom structure whose address is passed to threads. Single numeric values can be cast directly to void pointers without struct packaging. A newly created thread owns an independent call stack and shares the address space with all other threads within the same process.

## 二、线程等待与返回值获取

**pthread_join** 实现主线程阻塞等待目标线程执行结束，函数包含两个参数。第一个参数为待等待线程的 `pthread_t` 标识；第二个参数为二级 void 指针，用于接收线程返回数据的地址，不关心返回结果时直接传入 NULL。
线程返回数据存在关键安全约束，禁止返回线程栈上局部变量的指针，线程执行结束后栈内存自动回收，外部读取该指针会产生非法内存访问错误；正确方案是在堆内存分配存储返回值的结构体，在线程函数内分配、主线程 join 完成后手动释放堆内存。
程序分为两类使用场景，并行计算类程序通常调用 join 等待全部工作线程完成后再退出；长期运行的服务程序（如多线程 Web 服务器）持续创建工作线程处理请求，一般不会调用 join。配套封装函数会对 API 返回值做断言校验，快速捕获线程创建、等待过程中的异常。
**pthread_join** blocks the caller until the target thread terminates and takes two parameters. The first is the `pthread_t` ID of the waiting thread; the second is a double void pointer to capture the address of thread return data, and NULL can be passed if return values are irrelevant.
Critical safety rules apply to thread return values: pointers to stack-allocated local variables must never be returned, as stack memory is reclaimed after thread exit and leads to invalid memory access. The standard safe practice is allocating return storage on heap inside the thread and freeing the heap memory after join completes in the main thread.
Two typical usage patterns exist: parallel computing programs invoke join to wait for all worker threads before exit, while long-running server programs such as multi-threaded web servers spawn persistent worker threads without calling join. Wrapper functions wrap pthread APIs with assertion checks to detect runtime failures instantly.

## 三、互斥锁 Mutex

互斥锁用于保护临界区代码，实现线程间互斥访问共享资源，基础操作接口为 `pthread_mutex_lock` 与 `pthread_mutex_unlock`。调用 lock 时若锁未被占用，当前线程获取锁并进入临界区；若锁已被其他线程持有，调用线程阻塞直至锁释放；仅有持有锁的线程允许执行 unlock 释放操作。

锁存在两种合法初始化方式，静态初始化直接使用宏 `PTHREAD_MUTEX_INITIALIZER` 赋值；动态运行时初始化调用 `pthread_mutex_init`，支持自定义属性，属性传 NULL 使用默认配置，锁使用完毕后必须调用 `pthread_mutex_destroy` 回收资源。

标准锁接口存在两处常见缺陷，一是缺少初始化步骤会导致锁状态随机，引发不可复现的并发 bug；二是未校验 API 返回错误码，锁操作失败会静默失效，造成多线程同时进入临界区。工程中可封装锁函数，增加返回值断言校验，简化错误排查。

库内提供两类非阻塞锁拓展接口，`pthread_mutex_trylock` 获取锁失败时直接返回错误而不阻塞；`pthread_mutex_timedlock` 设置超时时间，超时未获取锁自动退出等待。两类接口适用于死锁规避场景，常规业务代码优先使用基础 lock/unlock。

Mutex locks protect critical sections and enforce mutual exclusion over shared resources, with core primitives `pthread_mutex_lock` and `pthread_mutex_unlock`. If the lock is free upon lock invocation, the caller acquires it and enters critical section; if held by another thread, the caller blocks until release. Only the lock owner can execute unlock.

Two valid initialization methods are available: static initialization via macro `PTHREAD_MUTEX_INITIALIZER`, and runtime dynamic initialization with `pthread_mutex_init` which supports custom attributes (NULL for default settings). Unused mutexes must be destroyed with `pthread_mutex_destroy`.

Two common pitfalls exist for mutex usage: missing initialization leads to undefined lock state and unreproducible concurrency bugs; unchecked return codes hide lock failures silently and break mutual exclusion. Wrapper functions with assertion checks simplify error detection in development.

Two non-blocking extended lock APIs are provided: `pthread_mutex_trylock` returns immediately on lock contention without blocking; `pthread_mutex_timedlock` aborts waiting after a specified timeout. These are mainly used for deadlock avoidance, while basic lock/unlock is preferred for general scenarios.

## 四、条件变量 Condition Variable

条件变量用于线程间同步等待与信号通知，解决线程需要等待其他线程修改状态后再执行的场景，配套核心接口为 `pthread_cond_wait`、`pthread_cond_signal`，使用时必须绑定一把互斥锁。

`pthread_cond_wait` 执行逻辑分为三步，调用时线程主动释放绑定的互斥锁并进入休眠等待；收到其他线程 signal 唤醒后，函数内部重新抢占获取互斥锁，函数返回时线程始终持有锁。信号发送接口 `pthread_cond_signal` 仅接收条件变量参数，发送信号前必须持有关联互斥锁，避免产生竞态条件。

等待逻辑必须使用 while 循环判断条件，不可使用 if 判断，原因是系统可能产生虚假唤醒，线程被唤醒后状态未必发生变化，循环可重新校验状态保证逻辑正确。

禁止采用自旋 flag 轮询替代条件变量，自旋持续占用 CPU 资源造成性能损耗，且手动 flag 同步极易引入并发漏洞，学术研究显示半数以上手动自旋同步代码存在逻辑缺陷。

条件变量支持静态宏 `PTHREAD_COND_INITIALIZER` 初始化，也可动态调用 `pthread_cond_init` 创建，生命周期结束调用 `pthread_cond_destroy` 释放资源。

Condition variables enable thread waiting and signaling for scenarios where one thread waits for state changes made by others, with core primitives `pthread_cond_wait` and `pthread_cond_signal`. Every condition variable must be paired with a dedicated mutex.

`pthread_cond_wait` follows three steps: release the paired mutex and sleep on invocation; re-acquire the mutex automatically after being woken by signal; return to caller with mutex held. `pthread_cond_signal` only takes a condition variable as argument and must be called while holding the associated mutex to eliminate race conditions.

Condition checks must be wrapped in while loops instead of single if statements, to handle spurious wakeups where threads wake without actual state changes, and recheck the predicate for correctness.

Spin-loop flag polling is strongly discouraged as a replacement for condition variables: busy waiting wastes CPU cycles, and manual flag synchronization introduces high concurrency bug rates. Research shows over half of ad-hoc spin synchronization implementations contain defects.

Condition variables can be statically initialized via macro `PTHREAD_COND_INITIALIZER`, or dynamically created with `pthread_cond_init` and destroyed with `pthread_cond_destroy` after use.

## 五、编译运行规范

使用 pthread 库编写程序存在固定编译要求，代码头部必须引入头文件 `pthread.h`；编译链接阶段增加 `-pthread` 编译参数，该参数同时完成头文件关联与线程库链接，标准编译命令格式为 `gcc -o 输出文件名 源文件.c -Wall -pthread`。

Fixed compilation rules apply to pthread programs: source files must include header `pthread.h`; the `-pthread` flag is required during compilation and linking to link the pthread library, with standard command template `gcc -o output source.c -Wall -pthread`.

## 六、POSIX 线程开发通用准则

代码逻辑尽量简化，复杂线程交互极易隐藏并发漏洞；尽可能减少线程间数据交互路径，所有同步逻辑采用成熟标准方案。锁、条件变量必须完整初始化，未初始化同步原语会触发随机异常。所有库函数返回码强制校验，忽略错误会导致难以复现的诡异运行行为。

线程传参与返回值严格规避栈内存指针，线程私有栈空间数据仅自身可访问，跨线程共享数据必须分配在堆或全局地址空间。线程间状态通知统一使用条件变量，杜绝自旋轮询实现同步。充分查阅 Linux 系统 pthread 手册页，手册包含大量接口细节与边界场景说明。

Simplify concurrent logic as much as possible, since complicated thread interactions hide concurrency bugs. Minimize cross-thread data interaction and adopt standard mature synchronization patterns. All mutexes and condition variables require full initialization to avoid random runtime errors. All pthread API return codes must be checked, as unhandled failures produce non-deterministic abnormal behaviors.

Never pass pointers to stack-allocated variables as thread arguments or return values. Each thread owns a private stack inaccessible to other threads; shared data must reside on heap or global memory. Use condition variables exclusively for cross-thread signaling and avoid busy spin waiting. Refer to Linux pthread man pages for detailed edge-case specifications of all interfaces.

## 七、配套调试工具与作业知识点

valgrind 工具集内的 helgrind 是并发专用检测工具，能够自动识别代码中数据竞争、死锁等并发问题，精准定位出错代码行并输出详细上下文信息。作业覆盖多类典型并发缺陷场景，包含无锁共享变量的数据竞争、双锁顺序不当引发的死锁、自旋 flag 低效同步、条件变量标准同步实现对比，通过工具输出区分正确与错误并发代码的行为差异。

helgrind, a subtool of valgrind, is dedicated to concurrency bug detection, which automatically identifies data races and deadlocks and reports exact faulty code lines with context details. The coding homework covers typical concurrency defect cases: unprotected shared variable data races, deadlock caused by disordered mutex acquisition, inefficient spin-flag synchronization, and standard condition variable implementations, helping distinguish correct and defective concurrent logic via tool output.


# OSTEP Chapter28 Locks 中英双语知识笔记
## 一、锁基础概念与Pthread互斥锁
### 1. 锁核心作用
并发编程核心痛点：多线程/多核、中断会拆分临界区指令，引发数据竞争。锁包裹临界区，保证临界区代码**原子执行**，同一时间仅一个线程进入临界区，实现互斥。
锁本质是状态变量：两种状态——空闲（unlocked/free）、持有（locked/acquired）；配套`lock()`获取锁、`unlock()`释放锁。
### 2. Pthread标准互斥锁（mutex）
POSIX库将锁命名为mutex（mutual exclusion互斥），标准使用范式：
```c
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
Pthread_mutex_lock(&lock);
balance = balance + 1; // 临界区
Pthread_mutex_unlock(&lock);
```
- 粗粒度锁：全局一把锁保护所有共享数据，并发度低；
- 细粒度锁：不同共享数据分配独立mutex，多线程可并行操作无关联数据，提升并发性能。
### Core Concept & Pthread Mutex
The fundamental pain point of concurrent programming: interrupts and multi-core execution split critical section instructions and trigger data races. Locks wrap critical sections to guarantee atomic execution; only one thread can enter a critical section at any time to achieve mutual exclusion.
A lock is essentially a state variable with two states: unlocked (free) and locked (held). Two core APIs: `lock()` to acquire, `unlock()` to release.
POSIX threads name locks mutex (short for mutual exclusion), standard usage:
```c
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
Pthread_mutex_lock(&lock);
balance = balance + 1; // critical section
Pthread_mutex_unlock(&lock);
```
- Coarse-grained lock: single global lock for all shared data, low concurrency;
- Fine-grained lock: separate mutex for different shared data, enabling parallel access to unrelated data and boosting concurrency.

## 二、锁的三大评价标准
### 1. 互斥正确性（Mutual Exclusion）
最基础指标：锁是否能阻止多线程同时进入临界区，杜绝数据竞争。
### 2. 公平性（Fairness）
锁释放后，等待线程是否均等获得获取机会；极端场景是否存在**饥饿（starvation）**：线程永久自旋等待，永远无法抢到锁。
### 3. 性能（Performance）
统计锁带来的CPU开销，分三类场景评估：
1. 无竞争：单线程反复加解锁，基础调用开销；
2. 单CPU多线程竞争：自旋、上下文切换损耗；
3. 多核CPU多线程竞争：跨核缓存、空转损耗。
### Three Evaluation Metrics for Locks
#### 1. Mutual Exclusion (Correctness)
The most fundamental standard: whether the lock prevents multiple threads from entering critical sections simultaneously to eliminate data races.
#### 2. Fairness
Whether waiting threads get equal chances to acquire the lock after release. Extreme case: starvation, where a thread spins indefinitely and never obtains the lock.
#### 3. Performance
Measure CPU overhead introduced by locks in three scenarios:
1. No contention: overhead of repeated lock/unlock by a single thread;
2. Single-CPU multi-thread contention: cost of spin-waiting and context switches;
3. Multi-CPU multi-thread contention: cross-core cache overhead and idle spinning.

## 三、早期互斥方案：关闭中断（Controlling Interrupts）
### 实现逻辑
单处理器系统早期方案：进入临界区关闭硬件中断，退出后恢复中断；无中断调度切换，临界区天然原子执行。
```c
void lock() { DisableInterrupts(); }
void unlock() { EnableInterrupts(); }
```
### 致命缺陷
1. 信任风险：关闭中断是特权指令，用户程序滥用会独占CPU、死循环卡死整机；仅适合内核内部使用；
2. 不支持多核：多CPU场景下，其他核心线程不受本地中断屏蔽影响，仍可同时进入临界区；
3. 丢失中断：长时间屏蔽中断会遗漏磁盘、网络硬件中断，引发系统IO异常。
### Early Mutual Exclusion: Disabling Interrupts
#### Implementation Logic
A primitive single-processor solution: disable hardware interrupts before entering critical sections and re-enable after exit. Without interrupt-driven context switches, critical sections run atomically.
```c
void lock() { DisableInterrupts(); }
void unlock() { EnableInterrupts(); }
```
#### Critical Drawbacks
1. Trust risk: interrupt masking is a privileged operation; malicious/buggy user programs can monopolize CPU or hang the whole system. Only safe for OS kernel internal use;
2. Multi-CPU incompatibility: threads on other cores ignore local interrupt masking and may concurrently enter critical sections;
3. Lost interrupts: long interrupt masking drops disk/network hardware interrupts and causes severe IO failures.

## 四、纯普通读写的失败锁方案：单flag标记
### 代码逻辑
仅用普通load/store读写flag变量，flag=0空闲、flag=1持有：
```c
void lock(lock_t *mutex) {
    while (mutex->flag == 1); // 循环检测
    mutex->flag = 1;
}
void unlock(lock_t *mutex) { mutex->flag = 0; }
```
### 核心问题
1. 正确性失效：检测flag和赋值flag分为两条非原子指令，线程调度中断可穿插执行，两个线程同时将flag置1，破坏互斥；
2. 性能缺陷：获取锁失败时持续循环读取flag，**自旋等待（spin-wait）** 空耗CPU。
### Failed Lock: Single Flag with Normal Load/Store
#### Logic
A naive implementation using regular memory read/write on a flag variable (0 = free, 1 = held):
```c
void lock(lock_t *mutex) {
    while (mutex->flag == 1); // polling loop
    mutex->flag = 1;
}
void unlock(lock_t *mutex) { mutex->flag = 0; }
```
#### Critical Flaws
1. Broken correctness: flag check and assignment are two non-atomic instructions. Thread preemption interleaves execution, allowing two threads to set flag=1 simultaneously and break mutual exclusion;
2. Poor performance: spin-waiting (endless polling) wastes CPU cycles when the lock is held.

## 五、硬件原子指令与自旋锁实现
### 5.1 Test-And-Set（TAS，测试并置位）
#### 原子语义
原子完成两件事：读取内存旧值、写入新值；两条操作不可分割，硬件保证原子性。
```c
int TestAndSet(int *old_ptr, int new) {
    int old = *old_ptr;
    *old_ptr = new;
    return old;
}
```
#### 自旋锁实现
```c
void lock(lock_t *lock) {
    while (TestAndSet(&lock->flag, 1) == 1); // 自旋直到获取锁
}
void unlock(lock_t *lock) { lock->flag = 0; }
```
#### 特性
- 正确：满足互斥；
- 不公平：无等待队列，存在饥饿；
- 性能：多核短临界区场景表现尚可；单CPU下线程被抢占后，自旋会耗尽时间片。

### 5.2 Compare-And-Swap（CAS，比较交换）
#### 原子语义
原子判断内存值是否等于预期值，相等则更新为新值；返回原始内存值。比TAS功能更强，可实现无锁编程。
```c
int CompareAndSwap(int *ptr, int expected, int new) {
    int original = *ptr;
    if (original == expected) *ptr = new;
    return original;
}
```
#### 自旋锁改造
```c
void lock(lock_t *lock) {
    while (CompareAndSwap(&lock->flag, 0, 1) == 1);
}
```

### 5.3 Load-Linked / Store-Conditional（LL/SC，加载链接+条件存储）
MIPS/ARM/PowerPC架构配套双指令：
1. LoadLinked：读取内存并标记硬件预留；
2. StoreConditional：仅当LL到SC之间内存无修改时，才写入新值并返回成功；否则写入失败。
#### 锁实现逻辑
先循环LL检测flag是否空闲；SC尝试原子抢占锁，失败则重新循环。

### 5.4 Fetch-And-Add（FAA，取值并自增）
原子读取旧值并对内存变量+1，用于实现公平Ticket锁。
```c
int FetchAndAdd(int *ptr) {
    int old = *ptr;
    *ptr = old + 1;
    return old;
}
```
#### Ticket锁（公平锁）
维护两个变量`ticket`（取票号）、`turn`（当前放行号）：
1. 线程加锁时FAA获取专属myturn票据；
2. 循环自旋直到`lock->turn == myturn`；
3. 解锁仅自增turn，有序放行等待线程。
#### 优势：天然公平，彻底杜绝饥饿。

### Hardware Atomic Instructions & Spin Locks
#### 5.1 Test-And-Set (TAS)
#### Atomic Semantics
Performs two operations indivisibly: read the old memory value and write a new value; hardware enforces atomicity.
```c
int TestAndSet(int *old_ptr, int new) {
    int old = *old_ptr;
    *old_ptr = new;
    return old;
}
```
#### Spin Lock Implementation
```c
void lock(lock_t *lock) {
    while (TestAndSet(&lock->flag, 1) == 1); // spin until lock acquired
}
void unlock(lock_t *lock) { lock->flag = 0; }
```
#### Properties
- Correct: guarantees mutual exclusion;
- Unfair: no waiting queue, starvation risk exists;
- Performance: decent on multi-core with short critical sections; severe waste on single CPU if lock holder is preempted.

#### 5.2 Compare-And-Swap (CAS)
#### Atomic Semantics
Atomically check if memory matches expected value; update to new value if match, return original value. More powerful than TAS, supports lock-free algorithms.
```c
int CompareAndSwap(int *ptr, int expected, int new) {
    int original = *ptr;
    if (original == expected) *ptr = new;
    return original;
}
```
#### Spin Lock Rewrite
```c
void lock(lock_t *lock) {
    while (CompareAndSwap(&lock->flag, 0, 1) == 1);
}
```

#### 5.3 Load-Linked / Store-Conditional (LL/SC)
Paired instructions for MIPS/ARM/PowerPC architectures:
1. LoadLinked: read memory and set hardware reservation tag;
2. StoreConditional: write new value only if no memory modification occurred between LL and SC, return success/failure.
#### Lock Logic
Loop LL to poll for free flag; use SC to atomically seize the lock, restart loop on SC failure.

#### 5.4 Fetch-And-Add (FAA)
Atomically read old value and increment memory variable by 1, foundation for fair ticket locks.
```c
int FetchAndAdd(int *ptr) {
    int old = *ptr;
    *ptr = old + 1;
    return old;
}
```
#### Ticket Lock (Fair Lock)
Maintain two shared variables: `ticket` (issue number) and `turn` (current allowed thread):
1. Thread calls FAA to get unique `myturn` ticket when acquiring lock;
2. Spin until `lock->turn == myturn`;
3. Unlock simply increments turn to release next waiting thread in order.
#### Advantage: naturally fair, completely eliminates starvation.

## 六、优化自旋浪费：从yield阻塞到内核休眠锁
### 6.1 自旋+Yield让出CPU
自旋获取锁失败时调用`yield()`主动放弃CPU，切换就绪线程执行；
- 优点：单CPU两线程场景消除无限空转；
- 缺陷：大量线程竞争时频繁上下文切换，开销巨大，仍存在饥饿。

### 6.2 队列休眠锁（Solaris park/unpark）
引入等待队列+OS休眠原语`park()`（线程睡眠）、`unpark(tid)`（唤醒指定线程）：
1. 获取锁失败时，线程加入等待队列，调用park阻塞；
2. 解锁时从队列取出线程unpark唤醒，锁直接移交唤醒线程，无需重置flag为0；
3. 引入`guard`自旋锁保护队列、flag修改，规避并发操作竞争；
4. 解决**唤醒丢失竞态**：使用`setpark()`标记即将睡眠，提前释放guard，防止线程永久休眠。
#### 优势：无CPU空转、有序唤醒，消除饥饿；
#### 缺陷：少量短自旋保护队列操作。

### 6.3 Linux Futex 混合锁（两阶段锁Two-Phase Lock）
现代系统标准实现，分两阶段混合自旋+内核阻塞：
1. 第一阶段（用户态自旋）：无竞争时仅执行原子位操作，无内核切换开销；短临界区自旋等待锁释放；
2. 第二阶段（内核阻塞）：自旋多次仍无法获取锁，调用`futex_wait`进入内核休眠；解锁时`futex_wake`唤醒等待线程。
#### Futex核心设计：单个int变量同时存储锁持有标记（最高位）、等待线程计数；区分快速无竞争路径、慢速阻塞路径。
#### 两阶段锁核心思想：先自旋尝试避免上下文切换，自旋失败再休眠，兼顾长短临界区性能。

### Mitigate Spin Waste: Yield, Blocking Queue Locks & Futex Hybrid Locks
#### 6.1 Spin + Yield
Call `yield()` to voluntarily release CPU when lock acquisition fails;
- Advantage: eliminates infinite idle spinning for two threads on single CPU;
- Drawback: massive context switch overhead under heavy multi-thread contention, starvation still possible.

#### 6.2 Queue-Based Blocking Lock (Solaris park/unpark)
Utilize waiting queue + OS sleep primitives `park()` (block thread) and `unpark(tid)` (wake target thread):
1. If lock acquisition fails, thread enqueues itself and calls park to sleep;
2. During unlock, dequeue a thread and unpark it; lock ownership transfers directly without resetting flag to 0;
3. Auxiliary `guard` spin lock protects concurrent modification of queue and flag;
4. Fix lost wakeup race condition: `setpark()` marks incoming sleep before releasing guard to prevent permanent thread blocking.
#### Advantages: zero CPU spinning, ordered wakeup, starvation-free;
#### Drawback: tiny short spin section to guard queue operations.

#### 6.3 Linux Futex & Two-Phase Locks
Standard modern lock implementation hybridizing spin and kernel blocking:
1. Phase 1 (user-space spin): atomic bit operations only for no-contention fast path, no kernel switch overhead; short critical sections spin briefly waiting for lock release;
2. Phase 2 (kernel blocking): after repeated spin failures, invoke `futex_wait` to sleep in kernel; unlock calls `futex_wake` to wake waiting threads.
#### Futex design: single integer stores both lock-held flag (highest bit) and waiter count, separating fast uncontended path and slow blocking path.
#### Two-phase lock core idea: spin first to avoid expensive context switches; fall back to sleep after spin timeout, balancing performance for both short and long critical sections.

## 七、衍生问题：优先级反转（Priority Inversion）
### 场景
调度器支持线程优先级：低优先级线程持有自旋锁，高优先级线程竞争锁持续自旋；低优先级线程因调度无法运行释放锁，高优先级线程永久卡死。
### 解决方案
1. 规避自旋锁，改用阻塞休眠锁；
2. 优先级继承：等待锁的高优先级线程临时提升锁持有者优先级；
3. 统一所有线程调度优先级。
### Derived Issue: Priority Inversion
#### Scenario
Priority-based scheduler: a low-priority thread holds a spin lock, a high-priority thread spins indefinitely contending for the lock. The low-priority thread cannot be scheduled to release the lock, permanently blocking high-priority work.
#### Mitigations
1. Replace spin locks with blocking sleep locks;
2. Priority inheritance: temporarily boost the lock holder’s priority to match waiting high-priority threads;
3. Assign identical scheduling priority to all threads.

## 八、软件纯读写算法（仅理论）
Dekker算法、Peterson算法：仅依靠普通load/store实现双线程互斥，无硬件原子指令；
### 局限
1. 仅支持2线程，无法扩展多线程；
2. 现代CPU宽松内存一致性模型下失效；
3. 性能差，无工程落地价值，仅并发理论教学使用。
### Pure Software Mutual Exclusion Algorithms (Theoretical Only)
Dekker’s algorithm and Peterson’s algorithm implement two-thread mutual exclusion with only regular load/store, no hardware atomic instructions.
#### Limitations
1. Only supports two threads, not scalable to multi-thread scenarios;
2. Broken under modern relaxed memory consistency models;
3. Poor performance with no practical engineering value, only used for concurrent theory teaching.

## 九、锁方案整体对比总结
1. 关中断：仅内核单CPU临时使用，不通用；
2. 基础flag：完全错误，无法互斥；
3. TAS/CAS自旋锁：正确、多核短临界区友好，不公平、单CPU自旋浪费；
4. Ticket锁（FAA）：自旋锁，公平无饥饿，仍空耗CPU；
5. Yield自旋锁：缓解单CPU自旋，大量线程切换开销高；
6. 队列休眠锁：无自旋浪费、公平，少量短自旋保护队列；
7. Futex两阶段锁：工业标准，无竞争极快，兼顾自旋与休眠，平衡性能。
### Summary of All Lock Schemes
1. Interrupt disable: only temporary kernel single-CPU usage, not general-purpose;
2. Basic single flag: completely incorrect, fails mutual exclusion;
3. TAS/CAS spin lock: correct, efficient for multi-core short critical sections, unfair with CPU waste on single CPU;
4. Ticket lock (FAA): fair spin lock without starvation, still consumes CPU via spinning;
5. Spin + Yield: mitigate single-CPU idle spin, heavy context switch cost with many threads;
6. Queue-based blocking lock: zero idle spin, fair, tiny short spin to guard queue metadata;
7. Futex two-phase lock: industrial standard, ultra-fast no-contention path, balances spin and sleep for overall performance.


# OSTEP Chapter29 Lock-based Concurrent Data Structures 锁基础并发数据结构知识笔记

## 一、章节主题与核心挑战

在锁的基础上，本章讨论如何把锁应用到常见数据结构中。给数据结构加锁使其可被多线程安全使用，即**线程安全（thread safe）**；加锁的具体方式同时决定数据结构的**正确性与性能**。核心挑战是：给定一种数据结构，如何加锁才能既保证正确，又让多条线程同时访问同一结构并保持高性能。并发数据结构已有多年研究积累（相关论文数以千计），本章只提供所需思维方式的基础引导，深入材料推荐 Moir 与 Shavit 的综述 [MS04]。

Building on locks, this chapter shows how to apply locks to common data structures. Adding locks to a structure so threads can use it safely makes it **thread safe**; exactly how the locks are added determines both the **correctness and the performance** of the structure. The core challenge: given a data structure, how to add locks so it works correctly while enabling many threads to access it concurrently with high performance. Concurrent data structures have been studied for years, with literally thousands of research papers; this chapter offers a sufficient introduction to the required thinking, pointing to Moir and Shavit's survey [MS04] for deeper study.

## 二、并发计数器 Concurrent Counters

### 1. 简单加锁计数器与性能瓶颈

计数器是最简单的数据结构，接口仅含 init/increment/decrement/get（图 29.1 为无锁版本，代码极少）。图 29.2 给出最基本加锁方案：**单一锁**，在操作数据结构的例程进入时获取、返回时释放，与 monitor 风格一致——调用对象方法时锁自动加解锁。该方案简单且正确，如果性能足够，任务就此完成，无需更花哨的设计；只有结构确实太慢时才需要进一步优化。

The counter is one of the simplest data structures, with a minimal interface of init/increment/decrement/get (Fig 29.1 shows the unsynchronized version with tiny code). Fig 29.2 presents the most basic locking approach: a **single lock**, acquired when calling a routine that manipulates the structure and released when returning, matching the monitor style where locks are acquired and released automatically on method calls. It is simple and correct; if performance is adequate, you are done—only when the structure is too slow do further optimizations become necessary.

基准测试揭示性能问题：在配备四颗 2.7 GHz Intel i5 的 iMac 上，各线程把共享计数器自增一百万次，并改变线程数（图 29.5 上线 'Precise'）。单线程约 0.03 秒完成；两个线程并发各更新一百万次反而耗时超过 5 秒，线程越多情况越糟。理想情况是多处理器上多线程应与单线程一样快地完成，称为**完美缩放（perfect scaling）**：总工作量虽然增加，但并行执行使完成任务的时间不增加。

Benchmarks reveal the performance problem: on an iMac with four 2.7 GHz Intel i5 CPUs, each thread updates a shared counter one million times while the thread count varies (top line 'Precise' in Fig 29.5). One thread finishes in roughly 0.03 seconds; two threads updating concurrently take over 5 seconds—worse with more threads. Ideally, many threads on multiple processors complete as fast as a single thread on one—**perfect scaling**: more work is done, but in parallel, so the time to finish does not increase.

### 2. 可扩展计数：近似计数器 Approximate Counter

研究者早已深入研究更可扩展的计数器 [MS04]，并且可扩展计数确实重要：Linux 多核性能分析 [B+10] 显示，缺少可扩展计数会使部分工作负载在多核机器上出现严重的扩展问题。本章介绍**近似计数器（approximate counter）**[C06]：把单个逻辑计数器表示为**每 CPU 一个本地物理计数器**加**一个全局计数器**；每个本地计数器配一把本地锁，全局计数器配一把全局锁。

Scalable counters have been studied for years [MS04], and they genuinely matter: Linux scalability analysis [B+10] shows workloads suffer serious scalability problems on multicore machines without scalable counting. The **approximate counter** [C06] represents one logical counter via numerous **local physical counters, one per CPU core**, plus a **single global counter**, with one lock per local counter and one for the global counter.

更新流程：某核心上的线程要自增时只增自己的本地计数器，用本地锁同步。因为每 CPU 有独立计数器，跨 CPU 的线程更新互不争用，更新因此可扩展。为让全局计数保持更新（供读取使用），本地值周期性转移到全局：获取全局锁、把本地值加进全局计数、再将本地清零。转移频率由**阈值 S** 决定：S 越小行为越接近不可扩展计数器；S 越大越可扩展，但全局值与真实计数偏差越大。若要精确值，可按规定顺序获取全部本地锁与全局锁（避免死锁），但那不可扩展。

Update flow: a thread on a given core only increments its local counter, synchronized by the local lock. Because each CPU has its own counter, threads across CPUs update without contention—updates are scalable. To keep the global counter fresh, local values are periodically transferred: acquire the global lock, add the local value to the global count, then reset the local counter to zero. The transfer frequency is set by **threshold S**: the smaller S, the more the counter resembles the non-scalable one; the larger S, the more scalable, but the further the global value drifts from the actual count. An exact value could be obtained by acquiring all local locks and the global lock in a specified order to avoid deadlock, but that is not scalable.

图 29.3 用 S=5 给出追踪示例：四个 CPU 各有本地计数器 L1..L4，时间向下推进，每步可能自增一个本地计数器；本地值达到阈值 S 即转移到全局并清零。图 29.5 下线 'Approximate' 显示 S=1024 时性能优异：四处理器更新四百万次的时间几乎不高于单处理器更新一百万次。图 29.6 展示阈值 S 的作用：S 低则性能差但全局计数始终相当准确；S 高则性能优秀，但全局计数滞后（最多滞后 CPU 数×S）。这个精度/性能权衡正是近似计数器所提供的。

Fig 29.3 traces the S=5 example: four CPUs with local counters L1..L4, time increasing downward, each step may increment a local counter; when a local value reaches threshold S, it transfers to the global counter and resets. Fig 29.5 bottom line 'Approximate' shows excellent performance at S=1024: four million updates on four processors take hardly more time than one million on one processor. Fig 29.6 shows the role of threshold S: low S gives poor performance but an always-quite-accurate global count; high S gives excellent performance but a lagging global count (lagging by at most CPUs × S). This accuracy/performance tradeoff is what approximate counters enable.

实现要点（图 29.4）：结构含全局计数 global 与全局锁 glock、每 CPU 计数 local[NUMCPUS] 与本地锁 llock[NUMCPUS]、阈值 threshold；update() 用 threadID 映射到 CPU，平时只锁本地并累加，本地达到阈值才在全局锁保护下转移并清零；get() 只返回全局值（近似）。本地锁存在的假设是每个核心上可能运行多条线程，若每核只有一条线程则无需本地锁。

Implementation (Fig 29.4): the structure holds global with glock, per-CPU local[NUMCPUS] with llock[NUMCPUS], and threshold; update() maps threadID to a CPU, normally locking only the local counter, transferring under the global lock once the local reaches the threshold; get() just returns the global value—only approximate. Local locks exist because more than one thread may run on each core; if exactly one thread ran per core, no local lock would be needed.

```c
typedef struct __counter_t {
    int global;                     // global count
    pthread_mutex_t glock;          // global lock
    int local[NUMCPUS];             // per-CPU count
    pthread_mutex_t llock[NUMCPUS]; // ... and locks
    int threshold;                  // update frequency
} counter_t;

// update: usually grab local lock and update local amount;
// once local count has risen 'threshold', grab global lock
// and transfer local values to it
void update(counter_t *c, int threadID, int amt) {
    int cpu = threadID % NUMCPUS;
    pthread_mutex_lock(&c->llock[cpu]);
    c->local[cpu] += amt;
    if (c->local[cpu] >= c->threshold) {
        pthread_mutex_lock(&c->glock);
        c->global += c->local[cpu];
        pthread_mutex_unlock(&c->glock);
        c->local[cpu] = 0;
    }
    pthread_mutex_unlock(&c->llock[cpu]);
}

// get: just return global amount (approximate)
int get(counter_t *c) {
    pthread_mutex_lock(&c->glock);
    int val = c->global;
    pthread_mutex_unlock(&c->glock);
    return val;
}
```

## 三、并发链表 Concurrent Linked Lists

### 1. 基础加锁方案与异常控制流问题

基础并发链表（图 29.7）采用单锁：insert 例程进入时获取锁、退出时释放锁。一个棘手细节是 malloc() 可能失败（罕见情况），此时代码必须在失败返回前释放锁。此类**异常控制流**被证明极易出错：对 Linux 内核补丁的近期研究发现，近 40% 的 bug 位于这类罕见路径上（该观察也催生了作者自己的一项研究：从 Linux 文件系统中移除内存失败路径，得到更健壮的系统 [S+11]）。

The basic concurrent linked list (Fig 29.7) uses a single lock: insert acquires the lock on entry and releases it on exit. One tricky issue arises if malloc() fails (a rare case): the code must release the lock before failing. Such **exceptional control flow** is quite error prone; a recent study of Linux kernel patches found nearly 40% of bugs on such rarely-taken paths (this observation sparked the authors' own research that removed memory-failing paths from a Linux file system, yielding a more robust system [S+11]).

由此产生一个挑战：能否重写 insert 与 lookup，在并发插入下保持正确，同时避免失败路径也需要调用 unlock？答案是肯定的：把加锁范围缩小到只包裹 insert 中真正的临界区，并让 lookup 使用单一公共返回路径。insert 中 malloc 部分无需加锁——假设 malloc 本身线程安全，每条线程都可以调用它而无竞态之忧；只有更新共享链表时才需要持锁。lookup 则通过简单变换跳出主搜索循环、汇入单一 return 路径，从而减少代码中锁的获取/释放点，降低意外引入 bug（如返回前忘记解锁）的概率。

This raises a challenge: can we rewrite insert and lookup to remain correct under concurrent insert while avoiding failure paths that also require an unlock call? The answer is yes: shrink the lock to surround only the actual critical section in insert, and give lookup a single common exit path. Part of insert needs no lock—assuming malloc is thread-safe, each thread can call it without race concerns; only updating the shared list requires holding the lock. Lookup uses a simple transformation to jump from the main search loop to a single return path, reducing the number of lock acquire/release points and thus the chance of accidentally introducing bugs such as forgetting to unlock before returning.

### 2. 扩展性：交接锁 Hand-over-Hand Locking

基础链表同样扩展性不佳。研究者探索过**交接锁（hand-over-hand locking，又名 lock coupling）**[MS04]：链表每个节点一把锁，遍历时先获取下一节点的锁，再释放当前节点的锁（因此得名 "hand-over-hand"）。概念上它允许链表操作获得很高的并发度；但实践中很难快过简单单锁方案——遍历每个节点都获取释放锁的开销过高。即使链表很大、线程很多，允许多个并发遍历带来的收益也不大可能超过"拿一把锁、执行一次操作、释放"的简单方案。或许"每隔若干节点获取一次新锁"的混合方案值得研究。

The basic list again scales poorly. Researchers explored **hand-over-hand locking (a.k.a. lock coupling)** [MS04]: one lock per node; traversal grabs the next node's lock, then releases the current node's lock (hence the name). Conceptually it enables a high degree of concurrency, but in practice it is hard to make such a structure faster than the simple single-lock approach, because the overhead of acquiring and releasing a lock for every node in a traversal is prohibitive. Even with very large lists and many threads, the concurrency from multiple ongoing traversals is unlikely to beat simply grabbing a single lock, performing the operation, and releasing it. A hybrid—grabbing a new lock every so many nodes—might be worth investigating.

## 四、并发队列 Concurrent Queues

除"加一把大锁"的标准做法（作者假定读者自己能实现）外，本章介绍 Michael 与 Scott 设计的并发队列 [MS98]（图 29.9）。队列有两把锁：**head 锁与 tail 锁**，目的是让入队与出队操作并发执行。常见情况下，入队例程只访问 tail 锁，出队只访问 head 锁。关键技巧是在初始化时分配一个**哑节点（dummy node）**，使 head 与 tail 操作得以分离。队列在多线程应用中很常见；但纯锁队列常不能完全满足需求，支持空/满等待的完整有界队列是下一章条件变量的主题。

Beyond the standard "add a big lock" approach (assumed to be obvious to the reader), this chapter presents the Michael and Scott queue [MS98] (Fig 29.9). It has two locks—a **head lock and a tail lock**—to enable concurrency between enqueue and dequeue. In the common case, enqueue touches only the tail lock and dequeue only the head lock. The key trick is a **dummy node** allocated at initialization, which separates head and tail operations. Queues are common in multithreaded applications; however, a pure lock-based queue often does not fully meet their needs—a complete bounded queue that lets threads wait when empty or full is the subject of the next chapter on condition variables.

```c
typedef struct __queue_t {
    node_t *head;
    node_t *tail;
    pthread_mutex_t head_lock, tail_lock;
} queue_t;

// enqueue: only the tail lock; append new node at tail
void Queue_Enqueue(queue_t *q, int value) {
    node_t *tmp = malloc(sizeof(node_t));
    assert(tmp != NULL);
    tmp->value = value;
    tmp->next = NULL;
    pthread_mutex_lock(&q->tail_lock);
    q->tail->next = tmp;
    q->tail = tmp;
    pthread_mutex_unlock(&q->tail_lock);
}

// dequeue: only the head lock; empty queue -> new_head == NULL
int Queue_Dequeue(queue_t *q, int *value) {
    pthread_mutex_lock(&q->head_lock);
    node_t *tmp = q->head;
    node_t *new_head = tmp->next;
    if (new_head == NULL) {
        pthread_mutex_unlock(&q->head_lock);
        return -1; // queue was empty
    }
    *value = new_head->value;
    q->head = new_head;
    pthread_mutex_unlock(&q->head_lock);
    free(tmp);
    return 0;
}
```

## 五、并发哈希表 Concurrent Hash Table

本章以哈希表收尾（图 29.10）：一个**不 resize** 的简单哈希表（resize 处理需要更多工作，留给读者练习），直接构建在前文开发的并发链表之上——`BUCKETS=101` 个链表，Hash_Insert/Hash_Lookup 通过 `key % BUCKETS` 定位桶并调用对应链表的插入/查找。性能出色的原因不是整个结构一把锁，而是**每桶一把锁**（每桶由一个链表表示），使大量并发操作可以同时进行。图 29.11 显示四线程各执行 1 万至 5 万次并发更新时的性能：这个简单并发哈希表缩放极好，而单锁链表几乎不扩展。

The chapter ends with a hash table (Fig 29.10): a simple **non-resizing** table (resizing requires more work and is left as an exercise), built directly on the concurrent lists developed earlier—`BUCKETS=101` lists; Hash_Insert/Hash_Lookup use `key % BUCKETS` to pick a bucket and call the corresponding list routine. Its excellent performance comes not from one lock for the whole structure but from **one lock per bucket** (each bucket being a list), enabling many concurrent operations. Fig 29.11 shows performance under concurrent updates (10,000–50,000 per thread, four threads): this simple concurrent hash table scales magnificently, while the single-lock linked list does not.

## 六、总结与关键教训

本章抽样介绍了从计数器、链表、队列到哈希表的并发数据结构，得到几条重要教训：注意锁围绕**控制流变化**（函数返回、退出、错误等）的获取与释放；**更多并发不必然更快**；**性能问题应在确实存在时才修复**。最后一点——避免过早优化——对任何注重性能的开发者都至关重要：如果加速不会改善应用整体性能，就没有价值。

This chapter samples concurrent data structures from counters, to lists and queues, and finally to the ubiquitous hash table. Key lessons: be careful with lock acquisition and release around **control flow changes** (returns, exits, errors); **more concurrency does not necessarily increase performance**; **performance problems should be remedied only once they exist**. The last point—avoiding premature optimization—is central to any performance-minded developer: there is no value in making something faster if doing so will not improve the application's overall performance.

历史佐证：许多操作系统转向多处理器时最初使用单一锁，包括 Sun OS 与 Linux（后者该锁名为 **big kernel lock, BKL**）。多年间这种简单方案是好的选择，但当多 CPU 系统成为常态，内核同一时刻只允许一个活跃线程变成性能瓶颈。Linux 采取更直接的路径：一锁换多锁；Sun 做出更激进的决定：从头构建新操作系统 Solaris，从第一天起就更根本地融入并发。

Historical evidence: many operating systems used a single lock when first transitioning to multiprocessors, including Sun OS and Linux (in the latter the lock was named the **big kernel lock, BKL**). For many years this simple approach was a good one, but when multi-CPU systems became the norm, allowing only one active thread in the kernel became a performance bottleneck. Linux took the more straightforward path—replace one lock with many; Sun made a more radical decision—build a brand-new operating system, Solaris, incorporating concurrency more fundamentally from day one.

本章只是高性能数据结构研究的开端：深入可参考 Moir 与 Shavit 的综述 [MS04]；B-tree 等其他结构需要数据库课程的知识；不依赖传统锁的**无锁（non-blocking）数据结构**会在常见并发 bug 章节初尝，但那是需要更多研究的整个知识领域。

This chapter only scratches the surface of high-performance structures. See Moir and Shavit's excellent survey [MS04] for more; other structures such as B-trees are best learned in a database class; **non-blocking data structures** that avoid traditional locks are tasted in the common concurrency bugs chapter but constitute an entire area requiring more study.

## 七、三条 TIP 与配套作业

- **更多并发不必然更快**：若方案因频繁加解锁引入大量开销（而不是一次性获取），更高并发可能并不重要；简单方案往往表现良好，尤其当它很少调用昂贵例程时。增加锁与复杂度可能成为败笔。唯一确定方法是把简单与复杂两个方案都实现并实测——性能上无法作弊。
- **警惕锁与控制流**：许多函数以获取锁、分配内存等有状态操作开始；出错时必须在返回前撤销全部状态，极易出错。应尽量重构代码以减少这种模式。
- **避免过早优化（Knuth 定律）**：构建并发数据结构先采用最基础方案——加一把大锁提供同步访问，先得到正确实现；若之后发现性能问题再精化，只在必要时才把它变快。Knuth 名言："Premature optimization is the root of all evil"（过早优化是万恶之源）。

- **More concurrency isn't necessarily faster**: if your design adds heavy overhead (e.g., acquiring and releasing locks frequently instead of once), higher concurrency may not matter; simple schemes tend to work well, especially if they rarely use costly routines. Adding more locks and complexity can be your downfall. The only real way to know is to build both alternatives and measure—you cannot cheat on performance.
- **Be wary of locks and control flow**: many functions begin with stateful operations like acquiring a lock or allocating memory; when errors arise the code must undo all state before returning, which is error prone. Structure code to minimize this pattern.
- **Avoid premature optimization (Knuth's law)**: start with the most basic approach—add a single big lock for synchronized access—and get a correct implementation first; refine only if performance problems appear. As Knuth famously stated, "Premature optimization is the root of all evil."

配套作业要点：用 gettimeofday() 测量时间，考察其精度与最小可测间隔（也可研究 x86 的 rdtsc 周期计数器）；构建简单并发计数器，随线程数增加测量耗时并考察可用 CPU 数的影响；构建 sloppy counter（近似计数器），随线程数与阈值测量并与章节数据对照；实现 hand-over-hand 链表并测量，找出它何时优于标准链表；选一个偏好数据结构（如 B-tree）先以单锁实现测量，再设计更有趣的加锁策略对比。

Homework highlights: measure time with gettimeofday() and study its accuracy and smallest measurable interval (also consider the rdtsc cycle counter on x86); build a simple concurrent counter and measure as thread count increases and against the available CPU count; build a version of the sloppy counter, measuring versus thread count and threshold, and compare with chapter data; implement a hand-over-hand linked list and measure when it beats the standard list; pick a favorite structure (e.g., B-tree), implement it with a single lock, measure, then design a more interesting locking strategy and compare.

来源：OSTEP（Operating Systems: Three Easy Pieces）Chapter 29 "Lock-based Concurrent Data Structures"，VERSION 1.01。原文：https://pages.cs.wisc.edu/~remzi/OSTEP/threads-locks-usage.pdf

主要参考文献：[MS04] Moir & Shavit, "Concurrent Data Structures", Handbook of Data Structures and Applications, 2004；[MS98] Michael & Scott, "Nonblocking Algorithms and Preemption-safe Locking on Multiprogrammed Shared-memory Multiprocessors", JPDC Vol.51, 1998；[B+10] Boyd-Wickizer et al., "An Analysis of Linux Scalability to Many Cores", OSDI '10；[C06] Corbet, "The Search For Fast, Scalable Counters", LWN, 2006；[S+11] Sundararaman et al., "Making the Common Case the Only Case with Anticipatory Memory Allocation", FAST '11。