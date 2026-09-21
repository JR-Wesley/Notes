# C++并发编程：从线程到并行算法

C++ 并发编程围绕执行流、共享状态和任务协作展开。线程提供执行载体，同步机制约束访问顺序，异步设施传递结果，并行算法提供更高层的计算接口。

## 1. 为什么要并发编程

应用往往需要同时处理多类工作，例如响应界面操作、读取数据和执行计算。如果让耗时工作占住唯一的执行流程，其他工作就只能等待。并发设计把这些工作组织成可以分别推进的任务；当硬件提供多个执行资源时，其中相互独立的计算还可以并行执行。其主要价值包括改善交互响应，以及利用多核缩短计算时间，但需要付出同步、调度和程序复杂度的代价。[Paul：为什么要并发编程](https://paul.pub/cpp-concurrency/#id-为什么要并发编程)

以图像编辑器为例，滤镜计算期间仍能处理用户的取消操作，体现的是响应能力；把图像分块交给多个核心计算，则可能缩短滤镜处理时间。两种收益应分别判断：界面保持响应，并不意味着计算本身更快。

### 1.1 并发与并行

**并发（Concurrency）描述多个任务如何被组织和推进；并行（Parallelism）描述多个计算是否在同一时刻实际执行。** 两者分别关注程序结构与执行方式。[Go 官方：Concurrency is not parallelism](https://go.dev/blog/waza-talk)

| 维度 | 并发 | 并行 |
| --- | --- | --- |
| 关注点 | 多个任务的进展与协作 | 多个计算在时间上重叠执行 |
| 执行方式 | 可以交替推进，也可以同时推进 | 需要能够同时执行计算的资源 |
| 示例 | 一个执行资源轮流处理 A、B | 两个执行资源同时处理 A、B |

下面的时间片示意只表达两者的区别，不代表固定调度顺序，也不假定每个任务在一个时间片内完成：

| 执行资源 | 时间片 1 | 时间片 2 | 时间片 3 | 时间片 4 |
| --- | --- | --- | --- | --- |
| 交替执行：资源 0 | A | B | A | B |
| 同时执行：资源 0 | A | A | A | A |
| 同时执行：资源 1 | B | B | B | B |

并发结构允许任务在单个执行资源上交替推进，也允许它们在多个执行资源上并行运行。因此，程序中存在多个任务或线程，并不能单独证明它们正在并行执行。

### 1.2 进程与线程

在常见操作系统的进程与线程模型中，**进程（Process）提供程序运行所需的资源环境，线程（Thread）承载其中可被调度的执行流。** 一个进程可以包含多个线程；这些线程共享进程地址空间，同时各自保存执行上下文。[Microsoft：About Processes and Threads](https://learn.microsoft.com/en-us/windows/win32/procthread/about-processes-and-threads)

| 维度 | 进程 | 线程 |
| --- | --- | --- |
| 主要职责 | 管理地址空间及相关系统资源 | 执行指令并接受调度 |
| 内存关系 | 通常具有独立的虚拟地址空间 | 同一进程内共享地址空间 |
| 执行状态 | 容纳一个或多个执行线程 | 各自具有寄存器上下文、栈和线程局部存储等 |

共享地址空间让线程间传递数据更直接，也意味着多个线程可能访问同一个对象。线程有自己的栈，不等于栈中对象绝对不可被其他线程访问：若其地址被传出，仍需要考虑共享访问和对象生命周期。

操作系统决定可运行线程何时获得执行资源。程序不能依赖“先创建的线程先完成”等调度假设；存在先后依赖的任务，应通过同步机制表达顺序。[Paul：进程与线程](https://paul.pub/cpp-concurrency/#id-进程与线程)

### 1.3 并发系统的性能

多线程带来的收益取决于任务能否独立执行，以及节省的时间能否覆盖额外开销。线程创建、任务切换和协调都会消耗资源，因此线程数增加不保证性能提升。[Paul：并发系统的性能](https://paul.pub/cpp-concurrency/#id-并发系统的性能)

#### Amdahl 定律：串行部分限制加速比

对于固定规模的任务，设单执行资源耗时为 $T_1$，其中可并行部分的时间比例为 $p$，可用执行资源数为 $N$。在均匀分配并行工作、忽略额外开销的理想模型下：

$$
T_N = T_1\left((1-p)+\frac{p}{N}\right),\qquad
S(N)=\frac{T_1}{T_N}=\frac{1}{(1-p)+p/N}
$$

当 $p<1$ 时，即使不断增加执行资源，加速比仍受串行部分限制：

$$
\lim_{N\to\infty} S(N)=\frac{1}{1-p}
$$

这里的 $p$ 是执行时间比例，不是代码行数比例。该模型讨论固定工作量的加速，不能直接用于工作量随资源数增长的情形。[LLNL：Amdahl's Law](https://hpc.llnl.gov/documentation/tutorials/introduction-parallel-computing-tutorial#Amdahl)

例如，按公式计算，一个原本耗时 100 秒、其中 10 秒必须串行执行的任务，使用 4 个理想执行资源时耗时为 $10+90/4=32.5$ 秒，加速比约为 $3.08$；无限增加资源也只能趋近 10 倍。这是模型推算值。

#### 实际开销与任务粒度

实际执行还受到同步、通信、资源竞争和负载不均衡影响。任务划分过细时，协调成本可能超过计算收益；划分不均时，其他执行资源需要等待最慢的任务。优化应同时关注可并行比例、任务粒度和硬件瓶颈。[LLNL：并行开销与任务划分](https://hpc.llnl.gov/documentation/tutorials/introduction-parallel-computing-tutorial)

评估时应明确目标：响应性关注交互是否及时，延迟关注单个任务耗时，吞吐量关注单位时间完成的工作量。对同一输入和正确结果比较串行与并发版本，才能判断并发设计是否带来了所需收益。

## 2. C++ 与并发编程

### 2.1 从平台接口到标准库

C++11 之前，C++ 标准没有统一的线程与同步设施。开发者可以使用 POSIX Threads、Windows 线程 API 或第三方库编写多线程程序，但需要处理平台之间的接口差异。C++11 将线程、同步和异步结果等机制纳入标准，为跨平台代码提供共同接口；C++14、C++17 又继续扩展这些设施。[Paul：C++ 与并发编程](https://paul.pub/cpp-concurrency/)

| 标准 | 代表性并发设施 | 主要作用 |
| --- | --- | --- |
| C++11 | `thread`、`mutex`、`condition_variable`、`future`、`atomic` | 建立线程管理、同步、结果传递与原子操作基础 |
| C++14 | `shared_timed_mutex`、`shared_lock` | 提供共享锁定及其所有权管理 |
| C++17 | `shared_mutex`、`scoped_lock`、并行算法 | 扩展读写互斥、多锁管理和算法执行策略 |

这里列出的是代表性设施。标准库的实现状态需要按具体版本和功能核对。[libstdc++：标准库实现状态](https://gcc.gnu.org/onlinedocs/libstdc++/manual/status.html)

### 2.2 编译器、标准库与语言模式

使用某个标准中的功能，需要区分三个条件：编译器能否处理相关语言特性、标准库是否实现所需接口，以及构建配置是否启用了对应语言模式。GCC 分别提供语言特性与 libstdc++ 的支持表，不能用单一的编译器版本号代替全部检查。[GCC：C++ Standards Support](https://gcc.gnu.org/projects/cxx-status.html)、[libstdc++：实现状态](https://gcc.gnu.org/onlinedocs/libstdc++/manual/status.html)

在 GCC 中，可通过 `-std=c++17` 明确选择 C++17 模式，例如以下命令编译已有的源文件：

```sh
g++ -std=c++17 example.cpp -o example
```

这个参数指定语言模式，不负责安装缺失的标准库或运行时依赖；线程与并行算法所需的链接配置还取决于具体工具链。显式指定语言模式也能避免构建依赖编译器版本各异的默认设置。[GCC：C++ Standards Support](https://gcc.gnu.org/projects/cxx-status.html)

## 3. 测试环境与构建依赖

### 3.1 原文的环境背景

原文示例通过 CMake 构建，并介绍了 macOS 与 Ubuntu 环境；其中并行算法示例采用 GCC 9 和 Intel TBB。Ubuntu 配置涉及 Ubuntu 16.04 与较新的 TBB 软件包，macOS 配置包含本机 TBB 安装路径。这些属于 2019 年的环境记录，具体软件包版本、路径及编译器限制应结合该历史背景理解。[Paul：测试环境](https://paul.pub/cpp-concurrency/#id-测试环境)

### 3.2 构建链中的不同职责

| 组成 | 职责 | 需要确认的内容 |
| --- | --- | --- |
| 编译器与语言模式 | 编译 C++ 源代码 | 编译器版本、所启用的 C++ 标准 |
| C++ 标准库 | 提供并发接口的实现 | 标准库种类、版本和目标功能的支持情况 |
| 系统线程支持 | 提供底层线程能力 | 平台需要的编译选项与链接依赖 |
| 并行算法后端 | 为算法实现提供执行支持 | 所选标准库及其配置要求的后端依赖 |
| CMake | 组织目标、语言标准与依赖 | 目标属性、依赖发现及链接设置 |

例如，CMake 的 `FindThreads` 模块可检测平台线程支持，并通过 `Threads::Threads` 目标携带所需的编译与链接要求。它不负责启用 C++ 语言标准，也不负责提供并行算法后端。[CMake：FindThreads](https://cmake.org/cmake/help/latest/module/FindThreads.html)

### 3.3 用 CMake 表达标准与线程依赖

在已经声明 C++ 项目且存在 `main.cpp` 的 CMake 工程中，可以为线程示例目标设置：

```cmake
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

add_executable(thread_example main.cpp)
set_target_properties(thread_example PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
target_link_libraries(thread_example PRIVATE Threads::Threads)
```

`CXX_STANDARD` 指定目标使用的标准级别；`CXX_STANDARD_REQUIRED` 禁止在无法满足要求时降到较低标准；`CXX_EXTENSIONS NO` 请求不使用编译器特有的扩展模式。CMake 从 3.8 开始识别 `CXX_STANDARD` 的值 `17`，但是否能够编译相应代码仍由工具链能力决定。[CMake：CXX_STANDARD](https://cmake.org/cmake/help/latest/prop_tgt/CXX_STANDARD.html)

`THREADS_PREFER_PTHREAD_FLAG` 表示在适用时优先选择 `-pthread` 方式；实际要求由 `FindThreads` 检测并传给目标。这段配置表达的是普通线程示例的构建要求，并行算法还需根据标准库实现补充相应依赖。[CMake：FindThreads](https://cmake.org/cmake/help/latest/module/FindThreads.html)

## 4. 线程与生命周期

线程管理需要区分三个对象：C++ 中的 `std::thread` 对象、它所代表的执行线程，以及线程访问的数据。三者的生命周期并不相同；正确使用线程，需要明确谁负责等待结束、谁拥有数据，以及数据何时可以销毁。[Paul：线程](https://paul.pub/cpp-concurrency/#id-线程)

### 4.1 创建线程与指定任务

`std::thread` 定义在 `<thread>` 中。使用可调用对象构造线程时，新执行线程会调用该对象；入口可以是普通函数、lambda 或函数对象。默认构造的 `std::thread` 则不启动线程。[C++ 标准草案：线程构造](https://eel.is/c++draft/thread.thread.constr)

下面是一个使用 lambda 的完整示例：

```cpp
#include <iostream>
#include <thread>

int main() {
    std::thread worker([] {
        std::cout << "worker is running\n";
    });

    worker.join();
}
```

构造 `worker` 时就启动了执行线程，`join()` 用于等待它结束。不能把 `join()` 理解成启动指令，也不能根据代码书写顺序推断新线程与主线程接下来谁先执行。[Paul：创建线程](https://paul.pub/cpp-concurrency/#id-创建线程)

线程入口的返回值不会由 `std::thread` 保存；入口抛出的异常若没有在线程内被捕获，会触发 `std::terminate()`，不会自动传播到调用 `join()` 的线程。需要返回结果或传递异常时，可以使用后文的 `future` 等设施。[C++ 标准草案：线程构造](https://eel.is/c++draft/thread.thread.constr)

### 4.2 参数传递与数据生命周期

对于 `std::thread(f, args...)`，可调用对象及参数通常以衰减后的值保存，并通过复制或移动构造。传入普通左值不等于让新线程引用原对象；需要按引用传递时，应明确使用 `std::ref` 或 `std::cref`。lambda 的引用捕获同样会让任务依赖外部对象。[C++ 标准草案：线程构造](https://eel.is/c++draft/thread.thread.constr)

```cpp
#include <functional>
#include <iostream>
#include <thread>

void calculate(int& result) {
    result = 6 * 7;
}

int main() {
    int result = 0;
    std::thread worker(calculate, std::ref(result));

    worker.join();
    std::cout << result << '\n';
}
```

这里 `result` 活到 `join()` 返回之后，主线程也只在工作线程结束后读取它。线程完成与成功的 `join()` 返回之间存在同步关系，因此可以在此后读取工作线程写入的结果。[C++ 标准草案：join](https://eel.is/c++draft/thread.thread.member)

若改为 `detach()`，就需要另外保证 `result` 的存活时间和访问顺序。引用包装器、指针和引用捕获本身都不会延长被引用对象的生命周期。[Paul：参数传递](https://paul.pub/cpp-concurrency/#id-创建线程)

### 4.3 join、detach 与 joinable

| 接口 | 作用 | 成功调用后的状态 |
| --- | --- | --- |
| `join()` | 阻塞调用线程，直到目标线程结束 | 目标已结束，线程对象不再关联执行线程 |
| `detach()` | 解除线程对象与执行线程的关联 | 执行线程可能继续运行，对象不再管理它 |
| `joinable()` | 查询对象是否仍关联执行线程 | 只查询，不等待、不改变状态 |

`joinable()` 不是“线程正在运行”的检测函数。线程函数已经返回，但尚未 `join()` 或 `detach()` 时，对象仍然是 joinable；默认构造、被移走关联关系，以及成功 join 或 detach 后的对象都不是 joinable。`join()` 和 `detach()` 要求对象处于 joinable 状态，不能在成功调用后重复执行。[C++ 标准草案：线程成员函数](https://eel.is/c++draft/thread.thread.member)

**`std::thread` 析构时若仍然 joinable，会调用 `std::terminate()`。** 它不会自动等待，也不会自动分离。因此，正常返回、提前返回和异常退出等路径都必须妥善处理线程关联关系；把 `join()` 写在函数末尾并不自动保证所有路径安全。[C++ 标准草案：线程析构](https://eel.is/c++draft/thread.thread.destr)

`detach()` 之后无法再通过原对象等待该线程，但仍可以通过共享状态和同步设施通信。它也不会把任务变成独立于进程的后台服务。分离只改变管理关联，数据生命周期和任务完成协议仍需明确设计。

### 4.4 管理当前线程

`std::this_thread` 中的函数作用于调用它们的线程，而不是某个指定的 `std::thread` 对象。

| 接口 | 语义 | 典型用途 |
| --- | --- | --- |
| `get_id()` | 获取当前执行线程的标识 | 日志与线程身份识别 |
| `yield()` | 给实现一次重新调度的机会 | 主动让出执行机会 |
| `sleep_for(duration)` | 按相对时长阻塞当前线程 | 延时执行 |
| `sleep_until(time_point)` | 按指定时间点阻塞当前线程 | 等待截止时间 |

例如，以下片段让调用线程暂停指定时长：

```cpp
// 需要 <chrono> 和 <thread>
std::this_thread::sleep_for(std::chrono::milliseconds(100));
```

`yield()` 不保证另一个特定线程接着执行；睡眠结束也不意味着线程立即获得处理器。更关键的是，`yield()`、`sleep_for()` 和 `sleep_until()` 都不建立线程间同步关系。不能用“先睡一会儿”证明另一个线程已完成数据写入；应使用 `join()`、条件变量或其他明确的同步机制。[C++ 标准草案：this_thread](https://eel.is/c++draft/thread.thread.this)

### 4.5 一次性初始化：call_once 与 once_flag

多个线程可能依赖同一份初始化结果。`std::call_once` 与 `std::once_flag` 定义在 `<mutex>` 中，用来协调这类一次性工作。**一次性的范围由共享的 `once_flag` 对象决定。** 如果每个线程各自创建一个普通局部 flag，就会形成互不相关的初始化过程。[Paul：一次调用](https://paul.pub/cpp-concurrency/#id-一次调用)

对于同一个 flag，初始化函数至多有一次正常返回。其他调用在正常返回后可以观察到初始化结果。如果初始化抛出异常，该次调用把异常传给调用者，不会将初始化标记为成功；后续调用可以再次尝试。[C++ 标准草案：call_once](https://eel.is/c++draft/thread.once.callonce)

下面的示例让主线程和工作线程共同使用一个初始化结果：

```cpp
#include <iostream>
#include <mutex>
#include <thread>

std::once_flag config_flag;
int config_value = 0;

void initialize_config() {
    config_value = 42;
}

int read_config() {
    std::call_once(config_flag, initialize_config);
    return config_value;
}

int main() {
    int worker_result = 0;
    std::thread worker([&worker_result] {
        worker_result = read_config();
    });

    const int main_result = read_config();
    worker.join();
    std::cout << main_result << ' ' << worker_result << '\n';
}
```

哪个线程执行初始化取决于运行时的调用次序；两次 `read_config()` 正常返回时，初始化都已成功完成。此后示例只读取 `config_value`。`call_once` 保护的是初始化及其结果发布，不会自动保护初始化之后对数据的任意修改。[C++ 标准草案：call_once](https://eel.is/c++draft/thread.once.callonce)

## 5. 并发任务、共享状态与正确性

### 5.1 从串行计算到任务划分

以区间内整数的平方根求和为例，各项的平方根可以独立计算，但最终需要汇总为一个结果。并发改造包含两个不同的问题：如何划分输入，以及如何合并输出。输入互不重叠，并不意味着输出没有共享访问。[Paul：并发任务](https://paul.pub/cpp-concurrency/#id-并发任务)

为明确边界，以下采用半开区间 $[0,M)$，目标为：

$$
S=\sum_{i=0}^{M-1}\sqrt{i}
$$

对应的串行计算函数使用局部累加器：

```cpp
#include <cmath>
#include <cstddef>

double sum_roots(std::size_t begin, std::size_t end) {
    double result = 0.0;
    for (std::size_t i = begin; i < end; ++i) {
        result += std::sqrt(static_cast<double>(i));
    }
    return result;
}
```

调用 `sum_roots(0, M)` 处理整个区间；分块后，每个任务可调用同一函数处理自己的区间。局部变量属于各次调用，函数本身没有写入共享的总和。

### 5.2 线程数量与区间边界

`std::thread::hardware_concurrency()` 返回实现提供的硬件并发度提示，无法确定时可以返回 `0`。它不是最优线程数，也不是为当前程序保留的资源数量，因此不能未经检查就用作除数。[C++ 标准草案：hardware_concurrency](https://eel.is/c++draft/thread.thread.static)

对于非空输入，可以先将 `0` 回退为 `1`，再把任务数限制为不超过元素数量；这只是初始划分策略，实际并发度仍需结合任务成本确定。空输入 $M=0$ 可直接返回零，无须创建线程。

一种保证覆盖完整、没有重复的划分方法是：对 $1\leq P\leq M$ 个任务，令

$$
q=\left\lfloor\frac{M}{P}\right\rfloor,\qquad r=M\bmod P
$$

前 $r$ 个任务各处理 $q+1$ 项，其余任务各处理 $q$ 项。第 $k$ 个任务（$0\leq k<P$）的边界为：

$$
b_k=kq+\min(k,r),\qquad
e_k=b_k+q+\begin{cases}1,&k<r\\0,&k\geq r\end{cases}
$$

任务处理 $[b_k,e_k)$。例如 $M=10,P=3$ 时得到 $[0,4)$、$[4,7)$、$[7,10)$。这是一种按商和余数推导的划分方案；若只让每个任务处理 $\lfloor M/P\rfloor$ 项，就会遗漏余数部分。

### 5.3 共享累加为什么出错

如果各任务将每一项直接加到同一个普通 `double sum` 中，即使输入区间互不重叠，也会并发读写同一个累加器。形如 `sum += value` 的表达式包含读取旧值、计算新值和写回，不能视为不可分割的一步。[Paul：竞争条件与临界区](https://paul.pub/cpp-concurrency/#id-竞争条件与临界区)

下面用简化的交错模型说明“更新丢失”。假设初值为 10，A 要加 2，B 要加 3：

| 步骤 | 任务 A | 任务 B | 共享值 |
| --- | --- | --- | --- |
| 1 | 读取 10 | | 10 |
| 2 | | 读取 10 | 10 |
| 3 | 计算并写回 12 | | 12 |
| 4 | | 根据先前的 10 写回 13 | 13 |

预期结果为 15，而这一抽象交错只留下 13。该表用于说明复合更新为何需要协调；真实 C++ 程序中的数据竞争属于未定义行为，不能据此断言结果只会落在某几个数值之间。[C++ 标准草案：数据竞争](https://eel.is/c++draft/intro.races)

### 5.4 竞争条件、数据竞争与临界区

| 概念 | 含义 | 在求和任务中的对应问题 |
| --- | --- | --- |
| 竞争条件（Race condition） | 正确性依赖于任务之间不受控制的相对执行顺序 | 多次更新交错后破坏求和结果 |
| 数据竞争（Data race） | C++ 内存模型中缺乏必要顺序的冲突访问 | 多个线程无同步地读写普通 `sum` |
| 临界区（Critical section） | 操作共享状态、需要协调访问的代码范围 | 对共享总和的完整读—改—写过程 |

在普通线程场景下，若两个潜在并发的访问发生冲突、至少一个不是原子操作，且彼此不存在 happens-before 顺序，就构成数据竞争。**happens-before 是程序建立的顺序关系，不是某次运行中碰巧观察到的先后顺序。** 因此，不能仅用缓存是否刷新、处理器是否单核或输出是否偶尔正确来判断安全性。[C++ 标准草案：数据竞争](https://eel.is/c++draft/intro.races)

竞争条件还可以出现在没有数据竞争的程序中。例如，若读取和写入各自都受锁保护，但两者之间释放了锁，上表中的交错仍可能发生：每次内存访问受到保护，完整的“读取后加上增量再写回”却没有作为整体受到保护。这说明同步范围应由要维护的不变式决定，而不只是给单条访问语句加锁。

### 5.5 正确性约束与汇总方式

对上述求和问题，可以直接从数学定义和对象生命周期得到四项约束：

1. **完整覆盖**：每个输入元素恰好属于一个任务。
2. **更新完整**：每个任务的贡献恰好计入最终结果一次。
3. **读取有序**：读取任务结果前，必须建立任务完成或结果就绪的同步关系。
4. **对象有效**：线程访问的输入、输出和同步对象必须保持存活。

主线程在最后逐个 `join()`，只能保证它在工作线程结束后继续；并不能消除工作线程运行期间彼此对 `sum` 的数据竞争。

沿着这些约束，可以形成两种汇总结构：让任务以相同的互斥协议更新共享总和；或者让任务各自形成局部结果，再在完成同步后统一汇总。前者需要确定临界区，后者需要明确结果归属与就绪时机。两者都必须先保证正确性，才能比较耗时。[Paul：互斥体与锁](https://paul.pub/cpp-concurrency/#id-互斥体与锁)

## 6. 互斥与锁管理

### 6.1 互斥体与同步协议

互斥（Mutual exclusion）让同一时刻至多一个线程拥有某个互斥体的独占所有权。线程先获取所有权，再访问受保护的数据，最后释放所有权。互斥体释放与后续成功获取同一个互斥体之间建立同步关系，使临界区中的更新能够按该关系被观察到。[C++ 标准草案：互斥体要求](https://eel.is/c++draft/thread.mutex.requirements.mutex)

| 操作 | 行为 | 返回结果 |
| --- | --- | --- |
| `lock()` | 等待直到获得所有权 | 成功返回时持有锁 |
| `try_lock()` | 尝试获取，不等待锁变为可用 | 成功为 `true`，失败为 `false` |
| `unlock()` | 释放当前线程拥有的锁 | 不再拥有锁 |

`try_lock()` 允许虚假失败，因此返回 `false` 不能证明其他线程一定持有锁。对普通 `mutex`，线程不能再次锁定自己已经持有的同一个互斥体；`unlock()` 也必须由拥有锁的线程执行。[C++ 标准草案：互斥体要求](https://eel.is/c++draft/thread.mutex.requirements.mutex)

**锁保护数据依赖共同遵守的协议。** 互斥体不会自动拦截对变量的访问；如果一个线程加锁写入，另一个线程绕过该锁读取，仍可能发生数据竞争。每个线程分别创建一把局部锁，也不能保护共同的累加器，因为它们没有竞争同一个互斥体。

### 6.2 互斥体的类型

| 类型 | 标准 | 扩展能力 |
| --- | --- | --- |
| `mutex` | C++11 | 基本独占互斥 |
| `timed_mutex` | C++11 | 支持限时获取 |
| `recursive_mutex` | C++11 | 同一线程可以重复获取 |
| `recursive_timed_mutex` | C++11 | 同时支持递归与限时获取 |
| `shared_timed_mutex` | C++14 | 共享、独占及限时获取 |
| `shared_mutex` | C++17 | 共享与独占获取 |

带超时的类型提供 `try_lock_for` 和 `try_lock_until`，分别按时长和时间点尝试获取。递归互斥体允许同一线程多次获取，但需要相同次数的释放才能完全解除占有；它并不能解决多个线程互相等待不同锁的问题。[Paul：mutex](https://paul.pub/cpp-concurrency/#id-mutex)

共享互斥体是**同一个对象的两种所有权模式**：多个线程可以同时以共享模式持有它；独占模式则排斥其他共享及独占持有者。共享接口包括 `lock_shared()`、`try_lock_shared()`、`unlock_shared()`。读写锁协议通常让读者获取共享所有权、写者获取独占所有权；共享模式本身不会禁止代码写入数据。[C++ 标准草案：共享互斥体](https://eel.is/c++draft/thread.sharedmutex.requirements)

### 6.3 锁的粒度与持有时间

求和任务若对每一项都先加锁，再计算平方根并更新总和，会把主要计算放进串行临界区，同时增加反复获取锁的成本。更合适的结构是各任务先算出局部结果，只在合并结果时持锁。[Paul：互斥体与锁](https://paul.pub/cpp-concurrency/#id-互斥体与锁)

下面的片段复用第 5 节的 `sum_roots`，并使用下一小节介绍的锁守卫：

```cpp
#include <mutex>

double total = 0.0;
std::mutex total_mutex;

void accumulate_block(std::size_t begin, std::size_t end) {
    const double local = sum_roots(begin, end);
    {
        std::lock_guard<std::mutex> guard(total_mutex);
        total += local;
    }
}
```

对于 $M$ 项、$P$ 个任务，这种改写把汇总时获取锁的次数从每项一次变为每任务一次，即从 $M$ 次减为 $P$ 次。这个计数来自程序结构，并不代表固定的性能提升倍数。`total` 应在启动任务前初始化；运行期间的访问遵守同一协议，或在所有任务结束并完成同步后读取。

锁的设计同时涉及保护的数据范围与持锁时间。缩短临界区时，必须保留完整的不变式，不能把需要整体执行的检查与更新拆开。独立计算可以移出锁；依赖受保护状态的计算则需要先证明移出后仍然正确。

### 6.4 RAII 与锁所有权

手动配对 `lock()` 和 `unlock()` 容易在提前返回或异常路径中遗漏释放。RAII 将资源管理绑定到对象生命周期：锁包装器负责持有互斥体所有权，并在正常离开作用域或异常栈展开时释放它。[Paul：通用互斥管理](https://paul.pub/cpp-concurrency/#id-通用互斥管理)

| 包装器 | 所有权与能力 | 适用场景 |
| --- | --- | --- |
| `lock_guard` | 构造时获取或接管所有权，析构时释放；不能移动 | 单个互斥体的固定作用域保护 |
| `unique_lock` | 可移动，可延迟获取、尝试获取、主动解锁及重新锁定 | 需要灵活控制持锁状态，或配合条件变量 |
| `shared_lock` | 可移动，以共享模式管理所有权 | 共享互斥体上的读者访问 |
| `scoped_lock` | 管理一个或多个互斥体；多锁时采用死锁避免算法 | 同时保护多个对象，C++17 起可用 |

互斥体提供锁定操作，包装器保存自己的所有权状态。特别是 `unique_lock` 析构时只在 `owns_lock()` 为真时解锁；构造了包装器，不等于已经持有锁。[C++ 标准草案：lock_guard](https://eel.is/c++draft/thread.lock.guard)、[unique_lock](https://eel.is/c++draft/thread.lock.unique)、[shared_lock](https://eel.is/c++draft/thread.lock.shared)、[scoped_lock](https://eel.is/c++draft/thread.lock.scoped)

| 构造标签 | 含义 | 使用条件 |
| --- | --- | --- |
| `defer_lock` | 关联互斥体，暂不获取所有权 | 稍后通过包装器获取锁 |
| `try_to_lock` | 尝试获取所有权 | 使用前检查是否成功 |
| `adopt_lock` | 接管当前线程已经获取的所有权 | 必须事先确实拥有相应锁 |

这些标签并非所有包装器都支持。例如 `lock_guard` 支持接管已持有的锁，但不提供延迟锁定模式。锁包装器还必须有足够长的生命周期；没有保存下来的临时守卫会在完整表达式结束时析构，无法保护后续语句。[C++ 标准草案：lock_guard](https://eel.is/c++draft/thread.lock.guard)、[unique_lock](https://eel.is/c++draft/thread.lock.unique)

### 6.5 多锁与死锁

转账同时涉及转出账户 A 和转入账户 B。若两个方向的转账各自按“先转出、后转入”加锁，就可能出现：

| 线程 | 已持有 | 正在等待 |
| --- | --- | --- |
| A 向 B 转账 | A 的锁 | B 的锁 |
| B 向 A 转账 | B 的锁 | A 的锁 |

双方都等待对方释放资源，而释放又依赖于等待结束，于是形成循环等待。统一锁定顺序可以避免这种倒序获取造成的循环；也可以把需要共同获取的锁交给 `std::lock` 或 `std::scoped_lock`。[Paul：死锁与通用锁定算法](https://paul.pub/cpp-concurrency/#id-死锁)

`std::lock(a, b, ...)` 获取全部参数对象的锁，并通过死锁避免算法协调获取过程。`std::try_lock(a, b, ...)` 则按参数顺序尝试；全部成功时返回 `-1`，否则返回获取失败的参数下标，并释放此次尝试中已经获取的锁。两者负责获取过程，不代替调用者管理成功获取后的释放。[C++ 标准草案：通用锁定算法](https://eel.is/c++draft/thread.lock.algorithm)

假设 `mutex_a`、`mutex_b` 是两个不同的互斥体，C++17 可用以下片段共同管理它们：

```cpp
{
    std::scoped_lock guard(mutex_a, mutex_b);
    // 在这里检查并更新两个对象的共同不变式。
}
```

如果两个参数实际上指向同一个普通互斥体，例如转出和转入是同一账户，就需要先处理该业务分支，不能直接把同一把锁传入两次。`scoped_lock` 解决其管理的多锁获取问题，并不保证包含其他锁、线程等待或回调的整个程序都没有死锁。[C++ 标准草案：scoped_lock](https://eel.is/c++draft/thread.lock.scoped)

### 6.6 延迟锁定与接管锁定

使用 `unique_lock` 延迟获取时，应将包装器传给 `std::lock`，使底层锁定与包装器的所有权状态一起更新：

```cpp
{
    std::unique_lock<std::mutex> lock_a(mutex_a, std::defer_lock);
    std::unique_lock<std::mutex> lock_b(mutex_b, std::defer_lock);
    std::lock(lock_a, lock_b);
    // 两个包装器现在都拥有锁，离开作用域时自动释放。
}
```

若这里改成 `std::lock(mutex_a, mutex_b)`，就绕过了包装器：底层互斥体被锁定，但两个 `unique_lock` 仍认为自己不拥有锁，析构时不会释放。这是由所有权状态推导出的区别。[C++ 标准草案：unique_lock](https://eel.is/c++draft/thread.lock.unique)

另一种方式是先获取底层锁，再用 `adopt_lock` 交给守卫管理：

```cpp
{
    std::lock(mutex_a, mutex_b);
    std::lock_guard<std::mutex> lock_a(mutex_a, std::adopt_lock);
    std::lock_guard<std::mutex> lock_b(mutex_b, std::adopt_lock);
    // 两个守卫接管已有所有权，不重复加锁。
}
```

这里 `adopt_lock` 是调用方作出的前置条件承诺，不是探测锁是否已经持有，也不是忽略锁状态的开关。[C++ 标准草案：lock_guard](https://eel.is/c++draft/thread.lock.guard)

## 7. 条件等待与线程协作

### 7.1 等待的是条件，通知是重新检查的机会

互斥体解决共享状态的访问冲突；条件变量解决“当前状态还不允许继续”的等待问题。例如消费者需要等待队列非空，有界队列中的生产者需要等待剩余容量。[Paul：条件变量](https://paul.pub/cpp-concurrency/#id-条件变量)

一个完整协议由共享状态、保护状态的互斥体、条件谓词和条件变量共同组成。`std::condition_variable` 定义在 `<condition_variable>` 中，其等待接口使用 `std::unique_lock<std::mutex>`：

```cpp
// lock 已持有保护共享状态的互斥体。
cv.wait(lock, [&] { return ready; });
```

带谓词的等待相当于反复执行 `while (!ready) cv.wait(lock);`。检查条件时持锁；需要等待时，释放锁与进入等待构成原子步骤；醒来后重新获取锁，再检查条件。等待可能虚假唤醒，通知后条件也可能被别的线程抢先改变，所以不能用一次 `if` 检查替代循环。[C++ 标准草案：condition_variable](https://eel.is/c++draft/thread.condition.condvar)

条件变量不保存“通知次数”。在等待线程尚未进入等待时发出的通知，不会成为未来可消费的事件。正确协议把事实保存在共享状态中：若 `ready` 已经为真，之后到来的线程直接通过谓词，无须收到那次通知。

### 7.2 一次结果发布示例

以下完整程序通过条件变量发布一个整数结果。生产者与消费者都在同一互斥体下访问 `ready` 和 `value`：

```cpp
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

int main() {
    std::mutex mutex;
    std::condition_variable cv;
    bool ready = false;
    int value = 0;

    std::thread producer([&] {
        {
            std::lock_guard<std::mutex> guard(mutex);
            value = 42;
            ready = true;
        }
        cv.notify_one();
    });

    int result;
    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return ready; });
        result = value;
    }
    producer.join();
    std::cout << result << '\n';
}
```

生产者即使先完成发布，消费者也能根据 `ready` 继续。消费者进入等待后会释放互斥体，让生产者有机会修改状态。示例中的所有共享对象存活到线程结束；`join()` 还确保通知操作已经完成后才销毁条件变量。

### 7.3 通知、超时与退出

| 接口或类型 | 作用与边界 |
| --- | --- |
| `notify_one()` | 唤醒一个等待者，不保证选中某个指定线程 |
| `notify_all()` | 唤醒所有等待者，各自仍需重新获取锁并检查条件 |
| `wait_for` / `wait_until` | 按相对时长或截止时间等待；带谓词版本返回条件是否满足 |
| `cv_status` | 无谓词限时等待的超时状态，不能单独证明业务条件成立 |
| `condition_variable_any` | 支持满足所需锁定要求的其他锁类型 |
| `notify_all_at_thread_exit` | 将解锁与通知安排在线程退出时，在线程局部对象析构后执行 |

通知可以在持锁时发出，也可以在解锁后发出；后者可避免等待者刚醒来就争用尚未释放的锁，但必须保证条件变量仍然存活。通知本身不代替共享状态的同步。[C++ 标准草案：条件变量](https://eel.is/c++draft/thread.condition.condvar)、[通用条件变量](https://eel.is/c++draft/thread.condition.condvarany)、[线程退出通知](https://eel.is/c++draft/thread.condition.nonmember)

对可关闭的队列，等待谓词可以设计为 `closed || !queue.empty()`。醒来后仍有数据就消费；队列为空且已关闭就退出。关闭时修改状态并通知所有等待者，才能避免结束阶段仍有人永久等待。这是把退出条件纳入同一状态协议的设计方式。

### 7.4 账户等待与转账原子性

按原文账户示例作静态分析，余额不足时等待到账，只解决了“何时允许扣款”。它还需要分别满足以下约束：

- 若允许余额变为零，扣款条件应允许余额等于扣款额，不能只接受严格大于。
- 余额读取也必须与写入同步；仅保护日志输出的锁，不能保护使用另一把锁修改的账户数据。
- 分别锁定后扣款、入账，不会自动使两个步骤成为不可分割的转账；即使消除单字段的数据竞争，读取总额仍需要一致的快照协议。
- 若所有参与者都等待其他参与者先提供资金，条件变量不会凭空产生进展；业务仍需定义失败、超时或取消方式。

因此，原文的总额异常不能仅解释为“读到转账中间状态”：未同步的余额读取还可能构成数据竞争，必须先消除未定义行为，再讨论一致性。[Paul：条件变量示例](https://paul.pub/cpp-concurrency/#id-条件变量)、[C++ 标准草案：数据竞争](https://eel.is/c++draft/intro.races)

## 8. 异步任务与结果传递

### 8.1 共享状态连接生产者与消费者

`<future>` 中的设施把任务执行与结果获取分开：生产者将值或异常写入共享状态，消费者通过 future 等待就绪并取出结果。共享状态既包含结果，也包含它是否就绪的信息。[C++ 标准草案：异步共享状态](https://eel.is/c++draft/futures.state)

| 设施 | 职责 |
| --- | --- |
| `async` | 按启动策略调用任务并返回 future |
| `packaged_task` | 将可调用对象包装为能发布结果或异常的任务 |
| `promise` | 由调用方显式发布值或异常 |
| `future` | 持有共享状态的独占消费接口，可移动、不可复制 |
| `shared_future` | 通过多个句柄共享结果，可复制 |

“共享状态”是库内部的结果通道，并不意味着任务访问的所有外部变量都因此安全。引用参数和捕获对象仍需满足生命周期与同步要求。

### 8.2 async 与启动策略

| 策略 | 执行方式 |
| --- | --- |
| `std::launch::async` | 在新的执行线程中调用任务 |
| `std::launch::deferred` | 首次非限时等待（如 `wait()` 或 `get()`）时，在调用该等待的线程中执行任务 |
| 默认策略 | 允许实现从 async 与 deferred 中选择 |

若需要明确的异步执行，应显式传入 `std::launch::async`。以下完整程序同时启动两个独立任务，再取得结果：

```cpp
#include <future>
#include <iostream>

int main() {
    auto left = std::async(std::launch::async, [] { return 6 * 7; });
    auto right = std::async(std::launch::async, [] { return 8 * 9; });
    const int result = left.get() + right.get();
    std::cout << result << '\n';
}
```

先取得某个 future 不会阻止已经启动的另一个任务执行。相反，若每启动一个任务就立即等待，再启动下一个，任务之间就失去了重叠执行的机会。

由 async 策略产生的共享状态，在最后一个引用释放时可能等待关联线程完成。因此，丢弃返回的临时 future 可能让表达式结束处发生等待；销毁 future 也不等于取消任务。延迟任务若始终没有被非限时等待触发，可以一直不执行。[C++ 标准草案：async](https://eel.is/c++draft/futures.async)

普通函数、lambda、函数对象和成员函数都可作为任务入口。通过指针或引用传入成员函数对象时，原对象必须存活到任务不再访问它；传入对象值则涉及复制或移动后的对象。[Paul：async](https://paul.pub/cpp-concurrency/#id-async)

### 8.3 等待、取值与异常

| 接口 | 语义 |
| --- | --- |
| `valid()` | 是否关联共享状态，不表示结果已经就绪 |
| `wait()` | 等待就绪，不取走结果，也不重新抛出任务保存的异常 |
| `wait_for` / `wait_until` | 返回 `ready`、`timeout` 或 `deferred`；限时等待不执行延迟任务 |
| `get()` | 等待并获取结果，或重新抛出共享状态中保存的异常 |

普通 `future` 的 `get()` 消费其共享状态关联，之后 `valid()` 为假，不能重复取值。future 可以移动到另一个线程后再消费，关键是所有权与访问协议，而不是只能由创建它的线程读取。[C++ 标准草案：future](https://eel.is/c++draft/futures.unique.future)

`shared_future` 可以由 `future.share()` 得到，原 future 随之失去关联。多个线程宜各自持有一份 shared_future 副本，调用 `get()` 读取同一结果；普通值类型的 `get()` 返回 `const T&`，可重复读取。共享读取能力不会自动保护结果所指向的可变对象。[C++ 标准草案：shared_future](https://eel.is/c++draft/futures.shared.future)

### 8.4 packaged_task：包装任务，分离调度

`packaged_task<R(Args...)>` 保存可调用对象。调用包装器时，它执行任务，并将返回值或任务抛出的异常放入关联共享状态。创建包装器本身不启动线程；执行位置由直接调用、工作线程或任务队列决定。[C++ 标准草案：packaged_task](https://eel.is/c++draft/futures.task.members)

```cpp
#include <future>
#include <iostream>
#include <thread>
#include <utility>

int main() {
    std::packaged_task<int(int)> task([](int x) { return x * x; });
    auto result = task.get_future();
    std::thread worker(std::move(task), 12);
    worker.join();
    std::cout << result.get() << '\n';
}
```

示例将任务移动到工作线程，并先 join 再 get，使取值抛出异常时也没有尚待回收的线程对象。线程池则让长期存活的工作线程从队列领取任务，避免为每个任务单独创建线程；队列的互斥、等待和关闭属于独立的调度协议。

`packaged_task` 可移动但不可复制，而 C++11～C++17 的 `std::function` 要求所存目标可复制，因此不能直接按值保存一个 packaged_task。使用以 `std::function` 为元素的队列时，需要额外的可复制间接包装，或采用支持移动任务的队列设计。[C++ 标准草案：packaged_task](https://eel.is/c++draft/futures.task.members)、[function 构造约束](https://eel.is/c++draft/func.wrap.func.con)

### 8.5 promise：显式发布值或异常

`promise` 让生产者自主决定结果何时就绪。`set_value()` 发布值，`set_exception()` 发布异常；同一共享状态不能重复满足，`get_future()` 也只能为该状态获取一次 future。生产者放弃未就绪的共享状态时，消费者可通过 `broken_promise` 错误获知结果无法正常提供。[C++ 标准草案：promise](https://eel.is/c++draft/futures.promise)

```cpp
#include <exception>
#include <future>
#include <iostream>
#include <thread>
#include <utility>

int main() {
    std::promise<int> producer;
    auto result = producer.get_future();
    std::thread worker([p = std::move(producer)]() mutable {
        try {
            const int value = 42;  // 可替换为可能抛出异常的计算。
            p.set_value(value);
        } catch (...) {
            p.set_exception(std::current_exception());
        }
    });
    worker.join();
    std::cout << result.get() << '\n';
}
```

结果发布与执行线程结束是不同事件：生产者可以先 `set_value()`，再进行后续工作。消费结果不授权销毁那些后续工作仍会使用的对象。上例选择 join 后读取；需要更早消费时，仍须在所有退出路径妥善回收工作线程。

### 8.6 从共享累加到结果归约

求和任务可让每个分块返回一个局部和，由主线程依次取得 future 的结果并累加。这样工作线程不再共同更新总和，同步集中在结果就绪处；区间覆盖与异常处理仍需要正确设计。

浮点加法不满足结合律，分块归约改变累加次序后，结果可能与串行逐项累加略有差异。数值验证应区分允许的舍入误差与漏算、重复计算或数据竞争，并选择符合任务精度要求的误差界。

## 9. 并行算法与性能

### 9.1 执行策略与算法接口

C++17 为部分标准算法增加接受执行策略的重载。算法仍表达“做什么”，策略指定“允许怎样执行”；它们不是三个 `sequenced_policy` 取值，而是不同策略类型的对象，定义在 `<execution>` 中。

| 策略 | 类型 | 元素操作的执行方式 |
| --- | --- | --- |
| `std::execution::seq` | `sequenced_policy` | 在调用线程中顺序执行 |
| `std::execution::par` | `parallel_policy` | 允许多个线程并行，同一线程内的调用不交错执行 |
| `std::execution::par_unseq` | `parallel_unsequenced_policy` | 允许并行及同一线程内无顺序交错执行，便于向量化 |

允许并行不等于保证使用多个线程，也不保证比串行快。实际执行受到实现、可用资源和输入规模影响。策略名称不是任务开始顺序、线程数或加速比的承诺。[C++ 标准草案：并行算法执行](https://eel.is/c++draft/algorithms.parallel.exec)

### 9.2 回调的正确性约束

并行算法中的比较器或操作函数必须满足算法本身的契约，并避免共享数据竞争。例如排序比较器不能随调用顺序改变其比较规则。

在 `par` 下，即使共享更新通过互斥体避免数据竞争，也可能因锁竞争失去性能；更不能让一个回调等待另一个回调“必须并行运行”，因为实现可以选择串行执行。在 `par_unseq` 下，同一线程内的操作允许交错，回调不能执行标准所禁止的向量化不安全操作，例如以互斥体阻塞等待。[C++ 标准草案：执行约束](https://eel.is/c++draft/algorithms.parallel.exec)

使用这些标准执行策略时，元素操作抛出且未捕获的异常会调用 `std::terminate()`，包括 `seq` 策略；这与不传执行策略的普通算法重载不同。算法自身无法分配所需临时内存时可以抛出 `std::bad_alloc`。[C++ 标准草案：并行算法异常](https://eel.is/c++draft/algorithms.parallel.exceptions)

### 9.3 用同一输入比较三种排序策略

以下完整示例仅演示接口与结果校验，不构成性能基准：

```cpp
#include <algorithm>
#include <execution>
#include <iostream>
#include <vector>

int main() {
    const std::vector<int> input{9, 2, 7, 2, 5, 1};
    auto sequential = input;
    auto parallel = input;
    auto vectorized = input;

    std::sort(std::execution::seq, sequential.begin(), sequential.end());
    std::sort(std::execution::par, parallel.begin(), parallel.end());
    std::sort(std::execution::par_unseq, vectorized.begin(), vectorized.end());

    const bool correct = std::is_sorted(sequential.begin(), sequential.end())
        && sequential == parallel && sequential == vectorized;
    std::cout << std::boolalpha << correct << '\n';
}
```

按原文代码作静态检查，第三次排序再次操作第二份已排序数据，未使用预先创建的第三份副本，因此三次计时的输入状态不同，不能直接据此评价策略速度。公平比较应让每种策略从同一原始输入的独立副本开始。[Paul：并行算法](https://paul.pub/cpp-concurrency/#id-并行算法)

计时还应明确是否包含数据生成、复制和线程运行时初始化；使用相同优化设置，进行多轮测量，并记录输入规模、分布、工具链与硬件。算法耗时和端到端耗时回答不同问题，两者都可以测量，但不能混用口径。原文的历史耗时不代表其他环境的预期结果。

## 10. 抽象层次与内存模型

| 需求 | 核心抽象 | 仍需由程序保证的内容 |
| --- | --- | --- |
| 管理执行线程 | `thread` | 结束回收、异常路径与数据存活 |
| 保护共享不变式 | 互斥体与 RAII 锁 | 统一访问协议、正确的临界区范围 |
| 等待可继续的状态 | 条件变量与谓词 | 状态发布、重新检查、退出与进展 |
| 获取任务结果 | future 共享状态 | 任务输入存活、异常消费、完成语义 |
| 对数据执行并行计算 | 标准算法与执行策略 | 回调约束、输入有效性、效果验证 |

这些抽象逐步减少手动协调，但都建立在对象生命周期与内存访问规则之上。内存模型进一步解释 happens-before、原子操作及内存序如何约束可观察结果；原文没有系统展开这一主题。原子操作本身也不等于整个算法无锁，更不自动保证复合业务操作正确。[C++ 标准草案：内存访问与数据竞争](https://eel.is/c++draft/intro.races)

## 概念关系

| 关系 | 含义 |
| --- | --- |
| 线程 → 生命周期 | 执行期间访问的对象必须保持有效 |
| 共享状态 → 同步 | 冲突访问需要明确的顺序约束 |
| 互斥 → RAII | 将锁的释放关联到所有权对象的生命周期 |
| 条件等待 → 共享状态与互斥 | 等待依据是状态谓词，检查与修改需要协调 |
| 异步任务 → 结果通道 | 任务执行与结果获取通过共享状态连接 |
| 并行算法 → 正确性与性能 | 执行策略的选择同时受到访问安全和计算开销约束 |

## 参考资料

- [Paul：C++ 并发编程（从C++11到C++17）](https://paul.pub/cpp-concurrency/)
- [[01-ComputerScience/Programming/concurrent-programming/并发编程核心|并发编程核心]]
- [[01-ComputerScience/Programming/concurrent-programming/C++并发编程实战|C++并发编程实战]]
