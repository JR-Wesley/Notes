---
category: Summary
---

> [!note] 参考
> <a href="https://cntransgroup.github.io/EffectiveModernCppChinese/Introduction.html">高效现代 C++ 中文 </a>
> <a href="https://learn.microsoft.com/zh-cn/cpp/cpp/welcome-back-to-cpp-modern-cpp?view=msvc-170">微软现代 C++ 中文 </a>
> <a href="https://en.cppreference.com/w/cpp/23.html"> Cpp reference 23 </a>

# 现代 C++ 语法特性

---

## 三、标准库（STL）

C++ STL（Standard Template Library，标准模板库）是 C++ 标准库的核心组成部分，基于泛型编程思想实现，提供了一系列**通用数据结构（容器）** 和**算法**，旨在提高代码复用性、效率和规范性。STL 的设计遵循 “**数据与操作分离**” 原则：容器存储数据，算法通过迭代器操作容器数据，二者通过迭代器无缝协作。

### STL 的六大核心组件

STL 由以下六大组件构成，彼此协同工作：

1. **容器（Containers）**：存储数据的模板类（如动态数组、链表、哈希表等）。
2. **迭代器（Iterators）**：连接容器与算法的 “桥梁”，提供访问容器元素的统一接口。
3. **算法（Algorithms）**：通用操作函数（如排序、查找、复制等），通过迭代器操作容器。
4. **函数对象（Functors）**：重载 `()` 运算符的类 / 结构体，可作为算法的参数（如自定义比较规则）。
5. **适配器（Adapters）**：转换已有组件的接口（如将容器转为栈 / 队列，或调整函数参数）。
6. **分配器（Allocators）**：负责容器的内存管理（默认无需手动干预）。

### 一、容器（Containers）

容器是 STL 中最常用的组件，用于存储同类型元素，分为三大类：**序列容器**、**关联容器**、**无序关联容器**。

#### 1. 序列容器（Sequence Containers）

**特点**：元素按插入顺序排列，可通过位置访问，不自动排序。

| 容器             | 底层结构   | 核心特性                                                   | 适用场景                      |
| -------------- | ------ | ------------------------------------------------------ | ------------------------- |
| `vector`       | 动态数组   | 连续内存，随机访问快（O (1)）；尾部插入 / 删除快（O (1)），中间插入 / 删除慢（O (n)）。| 频繁随机访问、尾部操作，如存储列表数据。|
| `list`         | 双向链表   | 非连续内存，随机访问慢（O (n)）；任意位置插入 / 删除快（O (1)）。| 频繁插入 / 删除（尤其是中间位置），如链表操作。|
| `deque`        | 分段连续数组 | 双端队列，头部 / 尾部插入 / 删除快（O (1)），随机访问较快（O (1)）。| 双端频繁操作，如队列、缓冲区。|
| `array`        | 固定大小数组 | 编译期确定大小，连续内存，效率与 C 数组相当，更安全（支持边界检查）。| 已知固定大小的数组，替代 C 风格数组。|
| `forward_list` | 单向链表   | 比 `list` 更节省空间，仅支持单向遍历，插入 / 删除效率同 `list`。| 内存受限场景，单向遍历为主。|

#### 2. 关联容器（Associative Containers）

**特点**：元素按**键（Key）** 排序（默认升序），基于**红黑树**实现，查找效率高（O (log n)）。

|容器|存储内容|核心特性|适用场景|
|---|---|---|---|
|`set`|键（Key）即值（Value）|键唯一，自动排序，支持快速查找、插入、删除。|存储不重复元素，需排序和快速查找（如字典）。|
|`multiset`|键即值，可重复|与 `set` 类似，但键可重复。|存储可重复元素，需排序（如成绩统计）。|
|`map`|键值对（Key-Value）|键唯一，按键排序，通过键快速访问值（类似字典）。|存储键值映射（如 ID→用户信息）。|
|`multimap`|键值对，键可重复|与 `map` 类似，键可重复（一个键对应多个值）。|一对多映射（如班级→学生列表）。|

#### 3. 无序关联容器（Unordered Associative Containers）

**特点**：元素无序，基于**哈希表**实现，平均插入 / 查找 / 删除效率为 O (1)（最坏 O (n)），C++11 新增。

|容器|存储内容|核心特性|适用场景|
|---|---|---|---|
|`unordered_set`|键即值，唯一|无序，哈希表存储，查找速度通常快于 `set`（无排序开销）。|无需排序，仅需快速查找 / 去重（如黑名单）。|
|`unordered_multiset`|键即值，可重复|无序，键可重复，哈希表存储。|无需排序，允许重复元素（如频率统计）。|
|`unordered_map`|键值对，键唯一|无序，哈希表存储，查找速度通常快于 `map`。|无需排序的键值映射（如缓存）。|
|`unordered_multimap`|键值对，键可重复|无序，键可重复，哈希表存储。|无需排序的一对多映射。|

### 二、迭代器（Iterators）

迭代器是**容器元素的 “指针抽象”**，提供统一的接口（如 `++`、`*`）访问容器元素，使算法可独立于容器类型。

- 迭代器分类（按功能强弱）：

|迭代器类型|支持操作|适用容器|
|---|---|---|
|输入迭代器|`++`、`*`（只读）|输入流（如 `istream_iterator`）|
|输出迭代器|`++`、`*`（只写）|输出流（如 `ostream_iterator`）|
|前向迭代器|`++`、`*`（读写），单向遍历|`forward_list`|
|双向迭代器|`++`、`--`、`*`（读写），双向遍历|`list`、`set`、`map`|
|随机访问迭代器|支持双向迭代器所有操作 + 随机访问（`+n`、`[]`）|`vector`、`deque`、`array`|

### 三、算法（Algorithms）

STL 提供了约 100 个通用算法（定义在 `<algorithm>` 中），按功能可分为：**排序与查找**、**修改与复制**、**数值计算**等。算法通过迭代器操作容器，不依赖具体容器类型。

- **排序与查找**：`std::sort`（排序）、`std::binary_search`（二分查找）、`std::find`（线性查找）。
- **修改与转换**：`std::for_each`（遍历元素）、`std::transform`（转换元素）、`std::copy`（复制元素）。
- **集合操作**：`std::set_union`（并集）、`std::set_intersection`（交集）、`std::remove`（移除元素）。

1. **排序算法**：`std::sort`（快速排序变种，平均 O (n log n)）

```cpp
#include <vector>
#include <algorithm>

int main() {
	std::vector<int> nums = {3, 1, 4, 1, 5};
	std::sort(nums.begin(), nums.end());  // 升序排序（默认）
	// 结果：1,1,3,4,5

	// 自定义排序（降序）
	std::sort(nums.begin(), nums.end(), std::greater<int>());  // 用标准函数对象
	// 结果：5,4,3,1,1
	return 0;
}
```

1. **查找算法**：`std::find`（线性查找）、`std::binary_search`（二分查找，需先排序）

```cpp

#include <vector>

#include <algorithm>

int main() {

std::vector<int> nums = {1, 3, 5, 7};

// 线性查找

auto it = std::find(nums.begin(), nums.end(), 3); // 找到3，返回迭代器

// 二分查找（需先排序）
bool has_5 = std::binary_search(nums.begin(), nums.end(), 5);  // true
return 0;

}

```

3. **遍历与修改**：`std::for_each`（遍历元素并执行操作）、`std::transform`（转换元素）

```cpp
#include <vector>
#include <algorithm>
#include <iostream>

int main() {
	std::vector<int> nums = {1, 2, 3};
	// 遍历并打印（用lambda作为操作）
	std::for_each(nums.begin(), nums.end(),
		[](int x) { std::cout << x << " "; });  // 1 2 3

	// 转换元素（每个数乘2）
	std::transform(nums.begin(), nums.end(), nums.begin(),
		[](int x) { return x * 2; });  // nums变为{2,4,6}
	return 0;
}
```

### 四、函数对象（Functors）

函数对象（仿函数）是**重载 `()` 运算符的类 / 结构体**，可像函数一样被调用。其优势是：可存储状态（成员变量），比普通函数更灵活，常用于算法的自定义参数（如比较规则、转换逻辑）。

#### 示例：自定义函数对象

```cpp
#include <vector>
#include <algorithm>

// 自定义函数对象：比较两个数的平方
struct SquareCompare {
    bool operator()(int a, int b) const {
        return a*a < b*b;  // 按平方值升序
    }
};

int main() {
    std::vector<int> nums = {3, -2, 1};
    // 用自定义函数对象排序
    std::sort(nums.begin(), nums.end(), SquareCompare());
    // 结果：1, -2, 3（平方分别为1,4,9）
    return 0;
}
```

#### 标准函数对象（定义在 `<functional>` 中）

STL 提供了常用函数对象，如 `std::plus`（加法）、`std::less`（小于）、`std::greater`（大于）等，可直接用于算法：

```cpp
#include <functional>  // 标准函数对象
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> nums = {3, 1, 4};
    // 用std::greater排序（降序）
    std::sort(nums.begin(), nums.end(), std::greater<int>());  // 4,3,1
    return 0;
}
```

### 五、适配器（Adapters）

适配器用于**转换已有组件的接口**，使其满足新的需求，分为三类：

#### 1. 容器适配器（Container Adapters）

基于现有容器实现特定数据结构，隐藏底层容器的部分接口，只暴露特定功能。

|适配器|底层默认容器|核心功能|适用场景|
|---|---|---|---|
|`stack`|`deque`|后进先出（LIFO）|栈操作（如表达式求值）|
|`queue`|`deque`|先进先出（FIFO）|队列操作（如任务调度）|
|`priority_queue`|`vector`|优先级队列（最大元素先出）|按优先级处理任务（如堆排序）|

#### 2. 迭代器适配器（Iterator Adapters）

转换迭代器的行为，如反向遍历、插入元素等。常见的有：

- `reverse_iterator`：反转迭代器方向（`rbegin()`、`rend()`）；
- `insert_iterator`：将赋值操作转为插入操作（如 `back_inserter`）。

#### 3. 函数适配器（Function Adapters）

调整函数的参数或行为，如绑定参数、否定逻辑等。C++11 后常用 `std::bind`（替代旧版 `bind1st`/`bind2nd`）。

**示例**：用 `std::bind` 绑定函数参数

```cpp
#include <functional>
#include <iostream>

// 原函数：a + b * c
int func(int a, int b, int c) {
    return a + b * c;
}

int main() {
    // 绑定b=2，c=3，只保留a作为参数：func(a, 2, 3) = a + 6
    auto bound_func = std::bind(func, std::placeholders::_1, 2, 3);
    std::cout << bound_func(4) << std::endl;  // 4 + 2*3 = 10
    return 0;
}
```

### 六、分配器（Allocators）

分配器负责容器的**内存分配与释放**，是容器模板的可选参数（默认使用 `std::allocator`）。用户通常无需关注，但可自定义分配器优化特定场景（如内存池、共享内存）。

**示例**：默认分配器与容器

```cpp
#include <vector>
#include <memory>  // std::allocator

int main() {
    // 显式指定分配器（默认就是std::allocator<int>）
    std::vector<int, std::allocator<int>> vec;
    vec.push_back(1);
    return 0;
}
```

### 其他库

> 多线程与并发见 [多线程](01-ComputerScience/Programming/现代C++/多线程.md)
> `<memory>` 见内存管理

- **`<utility>`**：包含 `std::pair`（键值对）、`std::swap`（交换）、`std::move`（移动语义）等基础工具。
- **`<tuple>`**：元组类型（`std::tuple`），可存储多个不同类型的值（类似结构体的轻量替代）。
- **`<optional>`**（C++17）：`std::optional` 表示 “可能存在的值”，避免用特殊值（如 `-1`）表示 “无结果”。
- **`<variant>`**（C++17）：`std::variant` 表示 “多类型中的一种”，类型安全的联合体。
- **`<chrono>`**（C++17）：时间库，用于时间点、时间段的计算（如秒、毫秒级计时）。
- 输入输出（`<iostream>`, `<fstream>`, `<sstream>`）
	- **`<iostream>`**：标准输入输出（`std::cin` / `std::cout`），用于控制台交互。
	- **`<fstream>`**：文件输入输出（`std::ifstream` / `std::ofstream`），支持文件读写。
	- **`<sstream>`**：字符串流（`std::stringstream`），用于字符串与其他类型的转换（如数字转字符串）。
- 数值工具（`<cmath>`, `<numeric>`, `<random>`）
	- **`<cmath>`**：数学函数库，包含三角函数（`sin`/`cos`）、指数对数（`exp`/`log`）、取整（`ceil`/`floor`）等。
	- **`<numeric>`**：数值算法，如 `std::accumulate`（累加）、`std::gcd`（最大公约数，C++17）。
	- **`<random>`**：高质量随机数生成器，替代 C 风格的 `rand()`（分布不均匀）。
- 字符串处理（`<string>`, `<string_view>`）
	- **`<string>`**：动态字符串类，封装了字符串的创建、拼接、查找、替换等操作，比 C 风格字符串（`char*`）更安全易用。
	- **`<string_view>`**（C++17）：轻量级字符串视图，用于 “只读访问字符串”，避免不必要的字符串拷贝（性能优化）。
- **文件系统（C++17）**：
    - `std::filesystem::path`, `create_directories`, `exists`。
- **范围库（C++20）**：
    - `std::views::filter`, `std::views::transform`。
- **协程（C++20）**：
    - 异步编程的新范式（如 `co_await`, `co_yield`）。
- **概念（Concepts, C++20）**：

```cpp
template<typename T>
concept Addable = requires(T a, T b) {
    { a + b } -> std::convertible_to<T>;
};
```

---

## 内存管理

在 C++ 中，**内存管理**是开发的核心挑战之一，而**RAII（Resource Acquisition Is Initialization，资源获取即初始化）** 是 C++ 解决资源（包括内存）管理问题的核心机制。RAII 通过对象的生命周期自动管理资源，从根本上避免了手动管理资源的风险（如内存泄漏、悬垂指针等）。

### 一、RAII：资源管理的核心思想

RAII 的核心逻辑是：**将资源的获取与对象的初始化绑定，将资源的释放与对象的析构绑定**。

- **资源获取**：当创建对象时（初始化），同时获取资源（如分配内存、打开文件、锁定互斥量等）。
- **资源释放**：当对象超出作用域（生命周期结束）时，自动调用析构函数，在析构函数中释放资源。

由于 C++ 中栈对象的生命周期是确定的（由作用域控制），这种机制能确保：**无论程序是正常退出还是因异常退出，资源都能被可靠释放**。

#### RAII 的简单示例（手动实现）

假设需要管理一块动态内存，传统手动管理方式存在泄漏风险，而 RAII 可通过封装解决：

```cpp
// 1. 传统手动管理（风险高）
void bad_usage() {
    int* ptr = new int(10);  // 获取资源（分配内存）
    // … 业务逻辑（若中途return或抛异常，delete不会执行）
    delete ptr;  // 释放资源（可能被跳过）
}

// 2. RAII封装（安全）
class IntPtr {  // 封装内存资源的RAII类
private:
    int* ptr;
public:
    // 构造函数：获取资源（初始化时分配内存）
    IntPtr(int value) : ptr(new int(value)) {}
    
    // 析构函数：释放资源（对象销毁时自动调用）
    ~IntPtr() {
        delete ptr;  // 确保释放，无论程序如何退出
    }
    
    // 提供访问资源的接口
    int& get() { return *ptr; }
};

void good_usage() {
    IntPtr raii_obj(10);  // 创建对象时获取资源（分配内存）
    raii_obj.get() = 20;  // 使用资源
    // … 业务逻辑（即使中途return或抛异常）
}  // raii_obj超出作用域，自动调用析构函数释放内存
```

**关键**：`IntPtr` 对象的生命周期完全由作用域控制，其析构函数必然会执行，从而确保内存被释放。

### 二、C++ 内存管理的核心工具（基于 RAII）

C++ 标准库提供了多个基于 RAII 的工具，彻底替代了手动 `new/delete`，从根源上避免内存问题。

#### 1. 智能指针（Smart Pointers）（C++11）

智能指针是 RAII 在内存管理中的典型应用，它们封装了原始指针，在析构函数中自动释放内存。C++11 起提供三种核心智能指针：

（1）`std::unique_ptr`：独占所有权

- 特点：**同一时间只能有一个 `unique_ptr` 指向资源**，所有权不可共享（禁止复制，仅允许移动）。
- 适用场景：管理单个对象的独占所有权（如局部动态对象、工厂函数返回值）。

```cpp
#include <memory>

void use_unique_ptr() {
    // 创建unique_ptr（获取资源），指向一个int对象
    std::unique_ptr<int> uptr(new int(10));  // C++11
    // 或更安全的方式（C++14起推荐）：
    auto uptr = std::make_unique<int>(10);   // 避免裸new，更安全
    
    *uptr = 20;  // 访问资源（重载了*和->运算符）
    
    // 所有权转移（通过移动语义）
    std::unique_ptr<int> uptr2 = std::move(uptr);  // uptr变为nullptr，uptr2拥有所有权
}  // uptr2超出作用域，自动释放内存（调用delete）
```

（2）`std::shared_ptr`：共享所有权

- 特点：**多个 `shared_ptr` 可共享同一资源的所有权**，通过 “引用计数” 跟踪所有者数量，当最后一个 `shared_ptr` 销毁时，释放资源。
- 适用场景：资源需要被多个对象共享（如容器中存储的动态对象、跨模块传递的资源）。

```cpp
#include <memory>
#include <vector>

void use_shared_ptr() {
    // 创建shared_ptr（引用计数初始化为1）
    auto sptr = std::make_shared<int>(100);  // 推荐使用make_shared，更高效
    
    {
        auto sptr2 = sptr;  // 复制，引用计数变为2
        *sptr2 = 200;
    }  // sptr2销毁，引用计数减为1
    
    std::vector<std::shared_ptr<int>> vec;
    vec.push_back(sptr);  // 引用计数变为2
    vec.push_back(sptr);  // 引用计数变为3
}  // sptr销毁，vec中元素也销毁，引用计数减为0 → 释放内存
```

（3）`std::weak_ptr`：弱引用（解决循环引用）

- 特点：**不增加引用计数**，仅作为 `shared_ptr` 的 “观察者”，可用于打破 `shared_ptr` 的循环引用（避免内存泄漏）。
- 用法：通过 `lock()` 方法获取 `shared_ptr`（若资源已释放，返回空）。

```cpp
#include <memory>

struct Node {
    std::shared_ptr<Node> next;  // 若两个Node相互指向，会形成循环引用
    // std::weak_ptr<Node> next;  // 改用weak_ptr可打破循环
};

void avoid_cycle() {
    auto node1 = std::make_shared<Node>();
    auto node2 = std::make_shared<Node>();
    node1->next = node2;  // 引用计数：node2变为2
    node2->next = node1;  // 引用计数：node1变为2 → 循环引用！
    
    // 函数结束时，node1和node2的引用计数各减1（变为1），资源不会释放（内存泄漏）
    // 若next是weak_ptr，node2->next = node1不会增加node1的引用计数，无循环
}
```

#### 2. 标准容器（自动管理内存）

STL 容器（如 `std::vector`、`std::string`）内部也基于 RAII 管理内存：

- 容器初始化时自动分配内存，元素添加时动态扩容。
- 容器销毁时（超出作用域），自动释放所有内部内存（包括元素的内存）。

```cpp
#include <vector>
#include <string>

void container_raii() {
    std::vector<int> vec;  // 初始化时分配内部内存
    vec.push_back(1);
    vec.push_back(2);  // 自动扩容
    
    std::string str = "hello";  // 管理字符串的动态内存
    
}  // vec和str销毁，自动释放所有内部内存（无需手动操作）
```

### 三、C++ 内存管理的常见问题与 RAII 的解决

手动管理内存（`new/delete`）容易引发三类问题，而 RAII 从根本上避免了这些问题：

|**问题**|**手动管理的风险**|**RAII 的解决方式**|
|---|---|---|
|**内存泄漏**|忘记调用 `delete`，或异常导致 `delete` 被跳过（如 `bad_usage()` 示例）。|析构函数自动执行，无论程序正常退出还是异常退出，资源必被释放。|
|**double free**|对同一指针多次调用 `delete`（如复制指针后分别释放）。|智能指针管理所有权（`unique_ptr` 禁止复制，`shared_ptr` 通过引用计数确保仅释放一次）。|
|**悬垂指针（野指针）**|指针指向的内存已释放，但指针未置空，后续访问导致未定义行为。|智能指针在资源释放后自动变为 `nullptr`（如 `unique_ptr` 移动后变为空）。|

### 四、RAII 的扩展：管理非内存资源

RAII 不仅用于内存管理，还可管理所有 “需手动释放的资源”，例如：

- 文件句柄（`std::fstream` 内部使用 RAII，离开作用域自动关闭文件）；
- 互斥锁（`std::lock_guard` 获取锁，析构时自动释放，避免死锁）；
- 网络连接、数据库连接等。

```cpp
#include <mutex>
#include <thread>

std::mutex mtx;  // 全局互斥锁

void safe_thread() {
    std::lock_guard<std::mutex> lock(mtx);  // RAII：获取锁（初始化）
    // … 临界区操作（即使抛异常）
}  // lock析构，自动释放锁（避免死锁）
```

### 总结

- **RAII 是 C++ 资源管理的灵魂**：通过对象生命周期绑定资源的获取与释放，确保资源安全。
- **内存管理的最佳实践**：完全避免手动 `new/delete`，优先使用 `std::unique_ptr`（独占）、`std::shared_ptr`（共享）和标准容器，它们都是 RAII 的完美实现。
- **核心价值**：将资源管理逻辑与业务逻辑分离，减少人为错误，提升代码可靠性。
- **内存管理**：
    - `new/delete` 与 `malloc/free` 的区别。
    - 内存泄漏、浅拷贝与深拷贝。
- **RAII 模式**：
    - 通过对象生命周期管理资源（如文件句柄、锁）。
- **智能指针**：
    - `std::unique_ptr`, `std::shared_ptr`, `std::weak_ptr`。

## 性能优化

### 优先使用连续内存容器，减少缓存未命中

- **原理**：CPU 缓存以 “缓存行”（通常 64 字节）为单位加载数据，连续内存（如 `vector`）能最大化缓存利用率，而离散内存（如 `list`）会导致频繁缓存失效。
- **实践**：用 `vector` 替代 `list`、`deque`（除非尾部操作），用 `array` 替代 C 风格数组（边界安全且性能相当）。

### 避免不必要的拷贝：用引用和移动语义

- **原理**：大对象（如 `std::string`、自定义类）的拷贝会触发内存分配和数据复制，开销大。
- **实践**：
    - 传递参数时用 `const T&`（常量引用）替代值传递。
    - 返回大对象时用**移动语义**（`std::move`）或依赖用返回值优化（RVO，编译器自动触发）。
    - 容器插入时用 `emplace_back`（直接在容器内构造对象）替代 `push_back`（先构造临时对象再拷贝）。

    ```cpp
    // 定义一个大对象
    struct BigObj { std::array<int, 1000> data; };
    
    // 差：值传递导致拷贝（1000个int复制）
    void bad_func(BigObj obj) {}
    
    // 优：引用传递，无拷贝
    void good_func(const BigObj& obj) {}
    
    // 容器插入：emplace_back直接构造，比push_back少一次拷贝
    std::vector<BigObj> vec;
    vec.emplace_back();  // 直接在容器内构造
    // 等价于 vec.push_back(BigObj());  // 先构造临时对象，再拷贝（或移动）
    ```

### 数据对齐与紧凑布局

- **原理**：CPU 访问未对齐的数据（如跨缓存行的 `int`）需要额外周期，而紧凑的数据结构能减少内存占用和缓存行消耗。
- **实践**：
    - 按 “字节由小到大” 排序类成员（利用内存对齐规则，减少填充字节）。
    - 用 `alignas` 指定对齐方式（如对齐到缓存行，避免伪共享）。

    ```cpp
    // 差：成员顺序导致大量填充（假设64位系统，int占4字节，double占8字节）
    struct BadLayout {
        char c;    // 1字节 + 3字节填充（对齐到int）
        int i;     // 4字节
        double d;  // 8字节（总大小：1+3+4+8=16字节）
    };
    
    // 优：按大小排序，无填充
    struct GoodLayout {
        char c;    // 1字节
        int i;     // 4字节（+3填充对齐到double）
        double d;  // 8字节（总大小：1+3+4+8=16字节，与BadLayout相同，但逻辑更紧凑）
    };
    
    // 避免伪共享（多线程访问同一缓存行的不同变量，导致缓存失效）
    struct alignas(64) CacheLineAligned {  // 对齐到64字节缓存行
        int value;
    };
    ```

### 减少函数调用开销：内联与避免过度封装

- **内联函数**：对短小高频的函数（如 getter/setter）用 `inline` 修饰，编译器会将函数体直接嵌入调用处，减少栈帧创建 / 销毁的开销。

    ```cpp
    // 高频调用的小函数，建议内联
    inline int add(int a, int b) { return a + b; }
    ```

    注意：`inline` 是建议，编译器可能忽略（如函数体过大）；类内定义的成员函数默认内联。

- **避免过度封装**：性能关键路径上，避免多层函数调用（如 `obj.getA().getB().calc()`），减少间接跳转开销。

### 编译期计算：用 `constexpr` 减少运行时开销

将能在编译期确定的计算（如常量、简单算法）用 `constexpr` 实现，避免运行时重复计算 （见前）。

## C++11/14/17/20/23 Misc

- **`if constexpr`（C++17）**：
	- 编译时条件分支，用于模板编程。
- **`std::optional`/`std::variant`（C++17）**：
    - 替代 `nullptr` 和 `int*` 的安全方式。
    - `std::variant` 用于类型安全的联合体。
- **`std::string_view`（C++17）**：
    - 轻量级字符串视图，避免不必要的拷贝。
- **`[[nodiscard]]`（C++17）**：
    - 强制检查函数返回值是否被丢弃。
- **`consteval`/`constexpr if`（C++20）**：
    - 编译期计算的强制要求。
- 模板与元编程
- **模板参数推导（C++17）**：

```
template<typename T>
class MyVector {};
MyVector v = {1, 2, 3}; // 编译器自动推导 T=vector<int>
```

- **SFINAE（C++11/17）**：
    - 使用 `std::enable_if` 实现条件编译。
- **模板特化**：
    - 全特化与偏特化。
- **错误处理**
- **异常安全（C++11/17）**：
    - `noexcept`、强异常安全保证。
- **错误码（C++11/17）**：
    - `std::error_code`, `std::system_error`。

---
