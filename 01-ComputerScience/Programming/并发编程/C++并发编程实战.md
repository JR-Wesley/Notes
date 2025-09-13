
> 参考

https://www.bilibili.com/video/BV1FP411x73X/?spm_id_from=333.337.search-card.all.click&vd_source=bc07d988d4ccb4ab77470cec6bb87b69

这本书对于 C++ 并发编程做了详尽的介绍，但是这本书有一些讲解和设计还是有缺陷的。推荐阅读英文版，中文翻译质量低。

- 中文翻译：https://nj.gitbooks.io/c/content/
- 本书源码下载地址：manning
- 不错的笔记：https://github.com/downdemo/Cpp-Concurrency-in-Action-2ed

# 第 1 章 你好，并发世界

主要内容

定义并发和多线程

使用并发和多线程

C++ 的并发史

简单的 C++ 多线程

## 1.1 何谓并发

# 2 Managing Threads

This chapter covers

- Starting threads, and various ways of specifying code to run on a new thread
- Waiting for a thread to finish versus leaving it to run
- Uniquely identifying threads

he C++ Standard Library makes most thread-management tasks relatively easy, with almost everything managed through the `std::thread` object associated with a given thread, as you’ll see.

## 2.1 Basic Thread Management

Every C++ program has at least one thread, which is started by the C++ runtime: the thread running main(). Your program can then launch additional threads that have another function as the entry point. These threads then run concurrently with each other and with the initial thread. In the same way that the program exits when it returns from main(), when the specified entry point function returns, the thread exits.

> 每个 C++ 程序至少有一个线程，由 C++ 运行时启动，即执行 `main()` 函数的主线程。
> 启动：程序可以启动额外的线程，这些线程以其他函数作为入口点。
> 执行：这些线程会与主线程及其他线程并发运行。
> 返回：如同程序从 `main()` 返回时退出一样，当线程的入口点函数返回时，该线程就会退出。

### 2.1.1 Launching a Thread

starting a thread using the C++ Standard Library always boils down to constructing a `std::thread` object:

```c
void do_some_work(); std::thread my_thread(do_some_work);
```

>线程的启动本质是**构造 `std::thread` 对象**——通过该对象指定线程要执行的“任务”，无论任务简单或复杂，这是 C++ 标准库启动线程的统一入口，与线程的具体功能、启动位置无关。
>- 无参、返回值为 `void` 的普通函数。函数在独立线程中执行，直到函数返回，线程就结束。
>- 带额外参数的“函数对象”（可理解为能像函数一样调用的类实例）。这类任务可能在运行中通过“消息系统”执行一系列独立操作，且仅当通过消息系统收到“停止信号”时，线程才结束。

As with much of the C++ Standard Library, `std::thread` works with any callable type, so you can pass an instance of a class with a function call operator to the `std::thread` constructor instead:

```cpp
class background_task {  
public: void operator()() const {  do_something(); do_something_else(); } 
}; 
background_task f; 
std::thread my_thread(f);
```

> `std::thread` 支持所有“可调用类型”——除了普通函数，还能传入带“函数调用运算符”的类实例（即函数对象），拓展了任务的复杂度。

In this case, the supplied function object is **copied** into the storage belonging to the newly created thread of execution and invoked from there. It’s therefore essential that the copy behaves **equivalently** to the original, or the result may not be what’s expected.

> **多线程编程中函数对象的传递与调用规则**，创建新线程传入的函数对象会被复制，必须保证副本与原对象逻辑等效。

One thing to consider when passing a function object to the thread constructor is to avoid what’s dubbed “C++’s most vexing parse.” If you pass a temporary rather than a named variable, the syntax can be the same as that of a function declaration, in which case the compiler interprets it as such, rather than **an object definition**. For example,

```cpp
std::thread my_thread(background_task());
```

declares a my_thread function that takes a single parameter (of type pointer-to-afunction-taking-no-parameters-and-returning-a-background_task-object) and  returns a std::thread object, rather than launching a new thread. You can avoid this by naming your function object as shown previously, by using an extra set of parentheses, or by using the new uniform initialization syntax; for example:

```cpp
std::thread my_thread((background_task()));
std::thread my_thread{background_task()};
```

One type of callable object that avoids this problem is a *lambda expression*. This is a new feature from C++11 which allows you to write a local function, possibly capturing some local variables and avoiding the need to pass additional arguments.

```cpp
std::thread my_thread([]{
do_something(); do_something_else();
});
```

> C++ 一个语法解析的漏洞，见 [[#C++ 解析解释]]

Once you’ve started your thread, you need to explicitly decide **whether to wait for it to finish (by joining with it—see section 2.1.2) or leave it to run on its own (by detaching it—see section 2.1.3)**. If you don’t decide before the `std::thread` object is destroyed, then your program is terminated (the `std::thread` destructor calls `std::terminate()`). It’s therefore imperative that you ensure that the thread is correctly joined or detached, even in the presence of exceptions. See section 2.1.3 for a technique to handle this scenario.

> 要求开发者显式处理线程生命周期使用，避免资源泄露或程序异常终止。

Note that you only have to make this decision before the `std::thread` object is destroyed—the thread itself may well have finished long before you join with it or detach it, and if you detach it, then if the thread is still running, it will continue to do so, and may continue running long after the `std::thread `object is destroyed; it will only stop running when it finally returns from the thread function.

> 你只需要在 `std::thread` 对象被销毁前决定是调用 `join()` 还是 `detach()`。线程本身可能在你调用这些方法之前就已经执行完毕。

如果选择 `detach()`：

- 即使线程仍在运行，它也会继续执行
- 即使 `std::thread` 对象已经被销毁，线程也可能继续长时间运行
- 线程只有在从其线程函数中最终返回时才会停止运行

简单来说，`std::thread` 对象的生命周期和它所代表的线程的实际执行周期可以是分离的，关键是要在对象销毁前做出 `join` 或 `detach` 的决定。

If you don’t wait for your thread to finish, you need to ensure that **the data accessed by the thread is valid until the thread has finished with it**.One situation in which you can encounter such problems is when the thread function holds pointers or references to local variables and the thread hasn’t finished when the function exits. The following listing shows an example of such a scenario.

>如果不等待线程完成，就必须确保线程所访问的数据在其运行期间内保持有效。这并非线程特有的新问题——即使在单线程代码中，访问已销毁的对象也是未定义行为。但线程的使用增加了遇到此类生命周期问题的可能性。

![[file-20250909140926984.png]]

![[file-20250909141526118.png]]

![[file-20250909141534150.png]]

One common way to handle this scenario is to **make the thread function self-contained and copy the data into the thread rather than sharing the data**. If you use a callable object for your thread function, that object is copied into the thread, so the original object can be destroyed immediately. But you still need to be wary of objects containing pointers or references, such as in listing 2.1. In particular, it’s a bad idea to create a thread within a function that has access to the local variables in that function, unless the thread is guaranteed to finish before the function exits. Alternatively, you can ensure that the thread has completed execution before the function exits by *joining* with the thread.

> 注意处理线程数据共享问题。让线程函数独立封装，将数据复制到线程中而非共享数据。需警惕包含指针或引用的对象。

#### C++ 解析解释

这段话，是 C++ 多线程编程中一个非常经典、但初学者容易困惑的问题：**“C++ 的最令人烦恼的解析”（Most Vexing Parse）**。

---

##### 🎯 核心问题：`std::thread my_thread(background_task());` 真的创建了一个线程吗？

**答案：不是！**
这行代码看起来像是在用 `background_task()` 这个函数对象启动一个新线程，但实际上，**它被编译器解析成了一条函数声明**，而不是对象定义。

---

##### 🔍 为什么会这样？——“Most Vexing Parse”的本质

C++ 的语法有一个规则：**如果一个语句既可以被解释为对象定义，也可以被解释为函数声明，那么编译器必须把它当作函数声明**。

来看这行代码：

```cpp
std::thread my_thread(background_task());
```

我们本意是：

> 创建一个名为 `my_thread` 的 `std::thread` 对象，并用一个临时的 `background_task()` 对象作为参数来构造它。

但编译器看到的是：

> 这看起来像一个函数声明！

具体来说，编译器可能会这样理解：

```cpp
std::thread my_thread(background_task());
```

等价于：

```cpp
std::thread my_thread(background_task (*param)());
```

即：

- 函数名：`my_thread`
- 返回类型：`std::thread`
- 参数：一个名为 `param` 的参数，它的类型是 **“指向一个无参、返回 `background_task` 对象的函数的指针”**

因为 `background_task()` 的写法和函数指针类型的语法非常相似，所以编译器选择了“函数声明”这个解释。

这就导致：

❌ 没有创建线程

❌ 只是声明了一个函数

❌ 你的 `background_task` 根本没有执行！

---

##### ✅ 如何避免这个问题？

有几种方式可以“打破”这种歧义，让编译器明白你是在定义对象，而不是声明函数。

1：加一层括号（Extra Parentheses）

```cpp
std::thread my_thread((background_task()));
```

注意这里有两个括号：`(background_task())`。这告诉编译器：这不是一个类型，而是一个表达式，因此不能是函数声明。

> ✅ 这样就能正确创建线程。

⚠️ 注意：这只是“历史写法”，C++11 之后更推荐下面的方法。

---

2：使用花括号初始化（Uniform Initialization，C++11 起）

```cpp
std::thread my_thread{background_task()};
```

花括号 `{}` **不会被解释为函数声明**，因为它不是 C 风格的语法。所以编译器不会产生歧义。

> ✅ 安全、清晰、现代 C++ 推荐写法。

---

3：使用变量（Named Variable）

```cpp
background_task task;
std::thread my_thread(task);
```

或者：

```cpp
std::thread my_thread((task)); // 也可以
```

当你使用一个**已命名的对象**时，就不会出现“函数指针类型”的歧义了。

---

##### 💡 建议

- 在现代 C++ 中，**优先使用花括号 `{}` 初始化**，既安全又清晰。
- 遇到“Most Vexing Parse”时，记住：**编译器总是倾向于“函数声明”**。
- 这个问题不仅出现在 `std::thread`，也出现在其他模板类构造中（如 `std::vector`, `std::make_pair` 等）。

---

### 2.1.2 Waiting for a Thread to Complete

If you need to wait for a thread to complete, you can do this by calling `join()` on the associated `std::thread `instance.

`join()` is a simple and brute-force technique—either you wait for a thread to finish or you don’t. If you need more fine-grained control over waiting for a thread, such as to check whether a thread is finished, or to wait only a certain period of time, then you have to use alternative mechanisms such as *condition variables and futures*, which we’ll look at in chapter 4.

The act of calling join() also **cleans up any storage associated with the thread**, so the std::thread object is no longer associated with the nowfinished thread; it isn’t associated with any thread. This means that you can call `join()` only once for a given thread; once you’ve called `join()`, the `std::thread` object is no longer joinable, and `joinable()` will return false.

> `join()` 用于等待线程完成并清理相关存储。

### 2.1.3 Waiting in Exceptional Circumstances

As mentioned earlier, you need to ensure that you’ve called either `join()` or `detach()` before a std::thread object is destroyed.

- If you’re detaching a thread, you can usually call `detach()` immediately after the thread has been started, so this isn’t a problem.
- But if you’re intending to wait for the thread, you need to carefully pick the place in the code where you call `join()`. This means that the call to `join()` is liable to be skipped if an exception is thrown after the thread has been started but before the call to `join()`.

To avoid your application being terminated when an exception is thrown, you therefore need to make a decision about what to do in this case. In general, if you were intending to call `join()` in a non-exceptional case, you also need to call `join()` in the presence of an exception to avoid accidental lifetime problems.

If you don’t need to wait for a thread to finish, you can avoid this exception-safety issue by *detaching* it. This breaks the association of the thread with the `std::thread` object and ensures that `std::terminate()` won’t be called when the `std::thread` object is destroyed, even though the thread is still running in the background.

> [!important] C++ 中 `std::thread` 的**生命周期管理规则**与潜在风险
> C++ 标准明确规定：**`std::thread` 对象被销毁前，必须先调用 `join()` 或 `detach()`**。  若不遵守，程序会触发“异常终止”（crash），因为线程的“归属权”未被正确处理（既没等待它结束，也没让它脱离对象独立运行）。

- `detach()` 的作用是让线程“脱离”`std::thread` 对象的控制，成为**后台独立运行的线程**（其生命周期不再与原 `std::thread` 对象绑定）。通常在“启动线程后立即调用 `detach()`”（比如 `std::thread t(func); t.detach();`），此时不会有问题——因为对象销毁时，线程已脱离控制，无需额外处理。
- `join()` 的作用是“阻塞当前线程（通常是主线程），等待目标线程执行完毕”，必须在 `std::thread` 对象销毁前调用。但它存在一个关键风险：**若线程启动后、`join()` 调用前抛出异常，`join()` 会被跳过，导致对象销毁时违反规则**。
- 总结：管理 `std::thread` 时，`detach()` 要注意避免访问已销毁的局部资源，`join()` 要警惕异常导致调用遗漏，两者都需确保在对象销毁前执行。

![[file-20250909143355452.png]]

The code in listing 2.2 uses a try/catch block to ensure that a thread with access to local state is finished before the function exits, whether the function exits normally, or by an exception. The use of try/catch blocks is verbose, and it’s easy to get the scope slightly wrong, so this isn’t an ideal scenario.

One way of doing this is to use the standard **Resource Acquisition Is Initialization (RAII)** idiom and provide a class that does the join() in its destructor.

>使用标准的 RAII（资源获取即初始化）惯用法，通过创建一个类，在其析构函数中执行 join() 操作。RAII 的核心思想是将资源的获取与对象的创建绑定，将资源的释放与对象的销毁绑定。

---

#### 📚 代码解读

```cpp
// Listing 2.3
class thread_guard {
    std::thread& t;

public:
    explicit thread_guard(std::thread& t_) : t(t_) {}

    ~thread_guard() {
        if (t.joinable()) { // make sure join is only called once
            t.join();
        }
    }

    thread_guard(thread_guard const&) = delete;
    thread_guard& operator=(thread_guard const&) = delete;
};

struct func;
void f() {
    int some_local_state = 0;
    func my_func(some_local_state);
    std::thread t(my_func);
    thread_guard g(t);  // 守护线程 t
    do_something_in_current_thread();
} // g 析构，自动调用 t.join()
```

---

##### 🔍 设计解读

C++ 中的 `std::thread` 如果在析构前没有被 `join()` 或 `detach()`，程序会**调用 `std::terminate()` 直接终止**（非常危险）。

这个类的唯一目的就是：

> 在其生命周期结束时，**自动等待（join）所管理的线程**，防止资源泄漏或程序崩溃。

---

##### ✅ **使用 RAII 原则：构造获取资源，析构释放资源**

- **构造函数**：接收一个 `std::thread&`，表示“我将守护这个线程”。
- **析构函数**：检查线程是否可连接（`joinable()`），如果是，则调用 `join()`。
- 这样，无论函数如何退出（正常返回、异常抛出），只要 `g` 在栈上，它的析构函数一定会被调用。

> ✅ 这是 RAII 的完美体现：**用对象的生命周期管理资源（这里是线程的 join 状态）**。

---

##### ✅ 3. **禁止拷贝：防止误用**

```cpp
thread_guard(thread_guard const&) = delete;
thread_guard& operator=(thread_guard const&) = delete;
```

为什么？

- 如果允许拷贝，可能会出现两个 `thread_guard` 都试图 `join` 同一个线程，而 `join()` 只能调用一次。
- 或者拷贝后原对象和副本都去 `join`，导致未定义行为。

> 🔒 禁止拷贝是安全且必要的设计。

---

##### ✅ 4. **引用成员：绑定到已有线程**

它持有 `std::thread&` 而不是 `std::thread`，说明它**不拥有线程对象，只是守护它**。

这在某些场景下是合理的，比如你已经创建了线程，现在想确保它被正确 `join`。

---

##### ⚠️ 潜在问题与局限性

虽然这个设计很经典，但也有一些**值得讨论的局限性**：

1. **依赖外部线程对象的生命周期**

```cpp
std::thread t(my_func);
thread_guard g(t);
```

如果 `t` 的生命周期比 `g` 短（比如 `t` 是临时对象或提前销毁），`g` 持有的引用就变成**悬空引用（dangling reference）**，析构时访问非法内存。

> 💡 但在 `f()` 这个函数中，`t` 和 `g` 都在栈上，`t` 先构造，后析构，所以 `g` 析构时 `t` 还活着 —— **安全**。

但这个类本身**无法保证**这一点，依赖使用者正确组织代码。

2. **不能移动，不够灵活**

现代 C++ 推崇移动语义，但这个类因为持有引用，**无法安全地移动或转移所有权**。

相比之下，更好的做法是**持有 `std::thread` 的值（即拥有线程对象）**。

---

##### ✅ 更现代的改进版本（推荐）

```cpp
class thread_guard {
    std::thread t;

public:
    explicit thread_guard(std::thread t_) : t(std::move(t_)) {}

    ~thread_guard() {
        if (t.joinable()) {
            t.join();
        }
    }

    thread_guard(const thread_guard&) = delete;
    thread_guard& operator=(const thread_guard&) = delete;

    thread_guard(thread_guard&&) = default;
    thread_guard& operator=(thread_guard&&) = default;
};
```

使用方式：

```cpp
void f() {
    int some_local_state = 0;
    func my_func(some_local_state);
    thread_guard g{std::thread(my_func)}; // 直接传入线程对象
    do_something_in_current_thread();
} // 自动 join
```

优点：

- 拥有线程对象，生命周期完全可控；
- 支持移动，更灵活；
- 避免悬空引用问题。

- **但在现代 C++ 中，更推荐“拥有线程对象”的版本**，更安全、更灵活。
- 使用 C++20，可以直接用 `std::jthread`，它内置了自动 `join()` 和协作中断机制，是 `thread_guard` 的官方实现 😊

---

### 2.1.4 Running Threads in the Background

Calling detach() on a std::thread object leaves the thread to run in the background, with no direct means of communicating with it. It’s no longer possible to wait for that thread to complete; if a thread becomes detached, it isn’t possible to obtain a std::thread object that references it, so it can no longer be joined. Detached threads truly run in the background; ownership and control are passed over to the C++ Runtime Library, which ensures that the resources associated with the thread are correctly reclaimed when the thread exits.

>`detach()` 使线程独立运行并由 C++ 运行时库自动管理生命周期，之后程序就无法再控制或获取该线程的状态了。

Detached threads are often called **daemon threads** after the UNIX concept of a **daemon process** that runs in the background without any explicit user interface. Such threads are typically **long-running**; they run for almost the entire lifetime of the application, performing a background task such as monitoring the filesystem, clearing unused entries out of object caches, or optimizing data structures. At the other extreme, it may make sense to use a detached thread where there’s another mechanism for identifying when the thread has completed or where the thread is used for a fire-and-forget task.

> **分离线程（Detached threads）** 常被称为“守护线程（daemon threads）”。可用于长期后台任务或无需关注结果的短期任务。

In order to detach the thread from a `std::thread` object, there must be a thread to detach: you can’t call `detach()` on a `std::thread` object *with no associated thread of execution*. This is exactly the same requirement as for join(), and you can check it in exactly the same way—you can only call `t.detach()` for a `std::thread` object t when  `t.joinable()` returns true.

> `detach() join()` 都只能用于关联了线程的对象

Consider an application such as a word processor that can edit multiple documents at once. There are many ways to handle this, both at the UI level and internally. One way that’s increasingly common at the moment is to have multiple, independent, top-level windows, one for each document being edited. Although these windows appear to be completely independent, each with its own menus, they’re running within the same instance of the application. One way to handle this internally is to run each document-editing window in its own thread; **each thread runs the same code but with different data** relating to the document being edited and the corresponding window properties. Opening a new document therefore requires starting a new thread. The thread handling the request isn’t going to care about waiting for that other thread to finish, because it’s working on an unrelated document, so this makes it a prime candidate for running a detached thread. The following listing shows a simple code outline for this approach.

> 一个多文档处理应用，每个线程执行相同代码，但是不同数据，打开不同窗口，并且这些线程为互不依赖的分离线程。
> TODO: 没有理解 listing 2.4 想做什么

# 3 Sharing Data between Threads

This chapter covers

- Problems with sharing data between threads
- Protecting data with mutexes
- Alternative facilities for protecting shared data

One of the key benefits of using threads for concurrency is the potential to easily and directly share data between them, so now that we’ve covered starting and managing threads, let’s look at the issues surrounding shared data.

If you’re sharing data between threads, you need to have rules for which thread can access which bit of data when, and how any updates are communicated to the other threads that care about that data. The ease with which data can be shared between multiple threads in a single process is not only a benefitit can also be a big drawback. Incorrect use of shared data is one of the biggest causes of concurrency-related bugs, and the consequences can be far worse than sausageflavored cakes. This chapter is about sharing data safely between threads in C++, avoiding the potential problems that can arise, and maximizing the benefits.

## 3.1 Problems with Sharing Data between Threads
