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