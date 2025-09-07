---
category: Book
---

# （二）C++ 基础语法

- **从 C 到 C++ 的过渡**
    
    - C++ 对 C 的扩展：命名空间、引用、函数重载
    - 输入输出流：iostream 库的使用
    - 内存管理：new/delete 操作符
    - 类型安全：强类型检查
    - 编译时计算：const、constexpr
- **基本语法特性**
    
    - 变量声明：auto 关键字、类型推导
    - 引用类型：左值引用、右值引用（C++11）
    - 函数重载：同名函数的不同参数
    - 默认参数：函数参数的默认值
    - 内联函数：inline 关键字的使用
- **命名空间**
    
    - namespace 定义：避免命名冲突
    - using 声明：简化命名空间使用
    - 匿名命名空间：文件内部链接
    - 命名空间别名：简化长命名空间名

# （三）面向对象编程基础

- **类与对象**
    
    - 类的定义：成员变量、成员函数
    - 对象的创建：栈对象、堆对象
    - 访问控制：public、private、protected
    - 构造函数：默认构造、参数构造、拷贝构造
    - 析构函数：资源清理、RAII 原则
- **封装特性**
    
    - 数据隐藏：私有成员的保护
    - 接口设计：公有接口的设计原则
    - getter/setter：属性访问方法
    - 友元：friend 关键字的使用
    - 静态成员：类级别的数据和方法
- **继承机制**
    
    - 单继承：基类和派生类
    - 多继承：多个基类的继承
    - 继承方式：public、protected、private 继承
    - 虚继承：解决菱形继承问题
    - 构造和析构顺序：继承链中的调用顺序
- **多态性**
    
    - 函数重载：编译时多态
    - 虚函数：运行时多态
    - 纯虚函数：抽象基类
    - 虚析构函数：正确的多态析构
    - 动态绑定：虚函数表机制

# 二、C++ 核心特性（3-4 个月）

## （一）内存管理与 RAII

- **动态内存管理**
    
    - new/delete：单个对象的分配释放
    - new[]/delete[]：数组的分配释放
    - 内存泄漏：常见问题和检测工具
    - 悬空指针：野指针的避免
    - 内存对齐：数据结构的内存布局
- **RAII 原则**
    
    - 资源获取即初始化：构造函数获取资源
    - 自动资源管理：析构函数释放资源
    - 异常安全：异常情况下的资源管理
    - 智能指针：自动内存管理
    - 作用域管理：栈对象的自动清理
- **智能指针（C++11）**
    
    - unique_ptr：独占所有权指针
    - shared_ptr：共享所有权指针
    - weak_ptr：弱引用指针，解决循环引用
    - 自定义删除器：特殊资源的清理
    - make_unique/make_shared：安全的对象创建

## （二）运算符重载

- **运算符重载基础**
    
    - 可重载运算符：算术、比较、逻辑运算符
    - 重载形式：成员函数、友元函数、全局函数
    - 返回类型：引用返回、值返回的选择
    - 参数传递：const 引用的使用
    - 重载原则：保持运算符的语义
- **特殊运算符重载**
    
    - 赋值运算符：operator=的实现
    - 下标运算符：operator[] 的重载
    - 函数调用运算符：operator() 的使用
    - 类型转换运算符：隐式和显式转换
    - 流运算符：operator<<和 operator>>

## （三）模板编程基础

- **函数模板**
    
    - 模板定义：template 关键字
    - 类型参数：typename 和 class 关键字
    - 模板实例化：显式和隐式实例化
    - 模板特化：全特化和偏特化
    - 模板参数推导：auto 和 decltype
- **类模板**
    
    - 类模板定义：模板类的声明和实现
    - 模板成员函数：类外定义的语法
    - 模板继承：模板类的继承关系
    - 模板友元：友元函数和友元类
    - 模板别名：using 和 typedef
- **模板元编程入门**
    
    - 编译时计算：constexpr 和模板递归
    - 类型萃取：type_traits 库的使用
    - SFINAE：替换失败不是错误
    - 模板约束：概念和约束（C++20）

## （四）异常处理

- **异常机制**
    
    - try-catch-throw：异常的抛出和捕获
    - 异常类型：标准异常类层次
    - 异常规范：noexcept 关键字（C++11）
    - 异常安全：基本保证、强保证、无抛出保证
    - 栈展开：异常传播过程中的对象析构
- **异常处理最佳实践**
    
    - RAII 与异常：资源管理的异常安全
    - 异常中立：库代码的异常处理
    - 异常性能：异常的性能影响
    - 错误码 vs 异常：错误处理策略选择

# 三、STL 标准模板库（2-3 个月）

## （一）容器类

- **序列容器**
    
    - vector：动态数组，连续内存存储
    - deque：双端队列，分段连续存储
    - list：双向链表，非连续存储
    - forward_list：单向链表（C++11）
    - array：固定大小数组（C++11）
- **关联容器**
    
    - set/multiset：有序集合，基于红黑树
    - map/multimap：有序映射，键值对存储
    - unordered_set/unordered_multiset：哈希集合（C++11）
    - unordered_map/unordered_multimap：哈希映射（C++11）
- **容器适配器**
    
    - stack：栈适配器，LIFO 结构
    - queue：队列适配器，FIFO 结构
    - priority_queue：优先队列，堆结构

## （二）迭代器

- **迭代器类型**
    
    - 输入迭代器：只读，单向遍历
    - 输出迭代器：只写，单向遍历
    - 前向迭代器：读写，单向遍历
    - 双向迭代器：读写，双向遍历
    - 随机访问迭代器：读写，随机访问
- **迭代器使用**
    
    - begin/end：容器的迭代器范围
    - 迭代器运算：递增、递减、距离计算
    - 迭代器失效：容器操作对迭代器的影响
    - 反向迭代器：rbegin/rend
    - 常量迭代器：const_iterator

## （三）算法库

- **非修改算法**
    
    - find/find_if：查找元素
    - count/count_if：计数元素
    - for_each：遍历执行操作
    - all_of/any_of/none_of：条件判断（C++11）
- **修改算法**
    
    - copy/copy_if：复制元素
    - transform：变换元素
    - replace/replace_if：替换元素
    - remove/remove_if：移除元素
    - unique：去除重复元素
- **排序算法**
    
    - sort：快速排序
    - stable_sort：稳定排序
    - partial_sort：部分排序
    - nth_element：第 n 个元素
    - binary_search：二分查找

## （四）函数对象与 Lambda

- **函数对象**
    
    - 函数指针：传统的函数传递方式
    - 函数对象类：重载 operator() 的类
    - 标准函数对象：less、greater、plus 等
    - 函数适配器：bind1st、bind2nd（已废弃）
- **Lambda 表达式（C++11）**
    
    - Lambda 语法：[capture](https://segmentfault.com/a/parameters) -> return_type { body }
    - 捕获方式：值捕获、引用捕获、混合捕获
    - 泛型 Lambda：auto 参数（C++14）
    - 初始化捕获：广义捕获（C++14）

# 四、现代 C++ 特性（3-4 个月）

## （一）C++11 核心特性

- **类型推导**
    
    - auto 关键字：自动类型推导
    - decltype：表达式类型推导
    - 尾置返回类型：函数返回类型推导
    - 类型别名：using 声明
- **移动语义**
    
    - 右值引用：&&语法
    - 移动构造函数：高效的对象转移
    - 移动赋值运算符：避免不必要的拷贝
    - std::move：强制移动语义
    - 完美转发：std::forward
- **统一初始化**
    
    - 列表初始化：{}语法
    - 初始化列表：std::initializer_list
    - 聚合初始化：结构体和数组
    - 直接初始化 vs 拷贝初始化

## （二）C++14/17/20 新特性

- **C++14 特性**
    
    - 泛型 Lambda：auto 参数
    - 变量模板：template 变量
    - 二进制字面量：0b 前缀
    - 数字分隔符：单引号分隔
- **C++17 特性**
    
    - 结构化绑定：auto [a, b] = tuple
    - if constexpr：编译时条件判断
    - 折叠表达式：可变参数模板简化
    - std::optional：可选值类型
    - std::variant：类型安全的联合体
- **C++20 特性**
    
    - 概念（Concepts）：模板约束
    - 协程（Coroutines）：异步编程
    - 模块（Modules）：替代头文件
    - 范围（Ranges）：算法库增强

## （三）并发编程

- **线程库（C++11）**
    
    - std::thread：线程创建和管理
    - std::mutex：互斥锁
    - std::lock_guard：自动锁管理
    - std::unique_lock：灵活的锁管理
    - std::condition_variable：条件变量
- **原子操作**
    
    - std::atomic：原子类型
    - 内存序：memory_order 枚举
    - 无锁编程：lock-free 数据结构
    - 内存模型：happens-before 关系
- **异步编程**
    
    - std::async：异步任务执行
    - std::future/std::promise：异步结果获取
    - std::packaged_task：任务包装
    - 线程池：自定义线程池实现

# 五、高级 C++ 编程（4-5 个月）

## （一）模板元编程

- **模板特化**
    
    - 全特化：完全指定模板参数
    - 偏特化：部分指定模板参数
    - 函数模板特化：特定类型的优化
    - 类模板特化：特定类型的实现
- **SFINAE 技术**
    
    - 替换失败不是错误：模板重载决议
    - enable_if：条件启用模板
    - 类型检测：has_member 等技术
    - 概念模拟：C++20 之前的约束
- **模板元编程技巧**
    
    - 递归模板：编译时计算
    - 类型列表：typelist 技术
    - 策略模式：policy-based design
    - 表达式模板：延迟计算

## （二）设计模式

- **创建型模式**
    
    - 单例模式：线程安全的实现
    - 工厂模式：对象创建的抽象
    - 建造者模式：复杂对象的构建
    - 原型模式：对象的克隆
- **结构型模式**
    
    - 适配器模式：接口适配
    - 装饰器模式：功能扩展
    - 外观模式：接口简化
    - 代理模式：访问控制
- **行为型模式**
    
    - 观察者模式：事件通知
    - 策略模式：算法封装
    - 命令模式：操作封装
    - 状态模式：状态管理

## （三）性能优化

- **编译器优化**
    
    - 内联函数：减少函数调用开销
    - 模板实例化：减少代码膨胀
    - 链接时优化：LTO 技术
    - 编译器指令：likely/unlikely（C++20）
- **运行时优化**
    
    - 内存局部性：缓存友好的数据结构
    - 分支预测：减少分支误预测
    - 向量化：SIMD 指令的使用
    - 内存池：减少内存分配开销
- **性能分析**
    
    - 性能分析工具：gprof、Valgrind、Intel VTune
    - 基准测试：Google Benchmark
    - 内存分析：内存泄漏检测
    - 热点分析：CPU 使用率分析

# 六、实际应用开发（持续进行）

## （一）图形界面开发

- **Qt 框架**
    
    - Qt Core：核心功能模块
    - Qt GUI：图形用户界面
    - Qt Widgets：传统桌面控件
    - Qt Quick：现代 UI 框架
    - 信号槽机制：事件处理
- **其他 GUI 框架**
    
    - GTK+：跨平台 GUI 工具包
    - wxWidgets：原生外观的 GUI
    - FLTK：轻量级 GUI 库
    - Dear ImGui：即时模式 GUI

## （二）游戏开发

- **游戏引擎**
    
    - Unreal Engine：虚幻引擎 C++ 开发
    - 自定义引擎：游戏引擎架构
    - 图形渲染：OpenGL、DirectX、Vulkan
    - 物理引擎：Bullet、Box2D
- **游戏编程技术**
    
    - 实体组件系统：ECS 架构
    - 游戏循环：渲染循环和逻辑更新
    - 资源管理：纹理、模型、音频
    - 网络编程：多人游戏同步

## （三）系统编程

- **操作系统接口**
    
    - POSIX API：跨平台系统调用
    - Windows API：Windows 系统编程
    - 文件系统：文件操作和目录遍历
    - 进程间通信：管道、共享内存、消息队列
- **网络编程**
    
    - Socket 编程：TCP/UDP 通信
    - 异步 I/O：epoll、IOCP
    - 网络库：Boost.Asio、libuv
    - HTTP 服务器：Web 服务开发

## （四）科学计算

- **数值计算库**
    
    - Eigen：线性代数库
    - BLAS/LAPACK：基础线性代数
    - GSL：GNU 科学计算库
    - Intel MKL：高性能数学库
- **并行计算**
    
    - OpenMP：共享内存并行
    - MPI：分布式内存并行
    - CUDA：GPU 并行计算
    - OpenCL：异构并行计算

# 七、工程化与最佳实践（2 个月）

## （一）代码质量管理

- **编码规范**
    
    - Google C++ Style Guide：谷歌编码规范
    - 命名约定：变量、函数、类的命名
    - 代码格式：缩进、空格、换行
    - 注释规范：文档注释、行内注释
- **静态分析**
    
    - Clang Static Analyzer：静态代码分析
    - PVS-Studio：商业静态分析工具
    - Cppcheck：开源静态分析
    - 编译器警告：-Wall、-Wextra 等选项

## （二）测试与调试

- **单元测试**
    
    - Google Test：C++ 测试框架
    - Catch2：现代 C++ 测试框架
    - 测试驱动开发：TDD 方法论
    - 模拟对象：Google Mock
- **调试技术**
    
    - GDB：GNU 调试器
    - Visual Studio 调试器：Windows 平台调试
    - Valgrind：内存错误检测
    - AddressSanitizer：地址消毒器

## （三）构建与部署

- **构建系统**
    
    - CMake：跨平台构建配置
    - Makefile：传统构建脚本
    - Ninja：高速构建执行器
    - 包管理：vcpkg、Conan
- **持续集成**
    
    - GitHub Actions：自动化构建测试
    - Jenkins：企业级 CI/CD
    - Docker：容器化部署
    - 跨平台编译：多目标平台支持

# 八、学习资源与职业发展

## （一）学习资源推荐

- **经典书籍**
    
    - 《C++ Primer》：C++ 入门经典
    - 《Effective C++》：C++ 最佳实践
    - 《More Effective C++》：进阶技巧
    - 《Effective Modern C++》：现代 C++ 特性
    - 《C++ 模板元编程》：模板高级技术
- **在线资源**
    
    - cppreference.com：C++ 标准库参考
    - ISO C++ 标准：官方语言标准
    - CppCon：C++ 年度大会视频
    - Compiler Explorer：在线编译器
    - C++ Core Guidelines：核心指导原则

# Ch2 变量和类型

# 2.3 复合类型

compound type 指基于其他类型定义的类型，引用和指针即属于符合类型。

一条声明语句是一个数据类型和变量名列表组成，更通用地说，是一个 base type + declarator 列表组成。每个声明符定义了一个变量并指定该变量为与基本类型有关的某种类型。

## 1 引用

reference 为变量起另一个名字，引用类型引用 refers to 是另一种类型。通过将声明符写为&d 的形式来定义引用类型，其中 d 是声明的变量名。

```c++
int ival = 1024;
int &refVal = ival;
```

初始化变量时，初始值会拷贝到新建对象中，定义引用时，引用和它的初始值绑定，而不是拷贝，因此引用必须初始化。

引用并非对象，它是已经存在的对象的另一个名字。对其的操作都是在与之绑定的对象上进行的。也不能定义引用的引用。引用一般需要和与之绑定的对象类型绑定。且引用只能绑定对象，不能与字面值绑定。

# 2.4 Const 限定符

使用 const 关键字对变量类型加以限定，不可对该变量赋值，同时必须初始化当然可以用非 const 类型初始化。

```c++
const int i = 32;

int j = 32;
const int k = j;
int h = i;
```

const 变量仅在文件内有效。多个文件内同名的 const 变量等同于不同文件中分别的独立的变量。

有时需要 const 变量在不同文件中有效，因此对其不管是声明还是定义都添加 extern 关键字

```c++
// file1.cc 可被其他文件访问
extern const int bufSize = fcn();
// file1.h 与.cc中定义的是同一个
extern const int bufSize; 
```

## 1 Const 引用

可以把引用绑定到 const 对象上，称为对常量的引用。与普通引用不同的是，对常量引用不能被用作修改它绑定的对象。

```c++
const int ci = 1024;
const int &r1 = ci;

// r1 = 42; 常量引用不可修改
// int &r2 = ci; 非常量引用不能指向常量
```

- 初始化和对 const 引用
初始化常量引用允许使用任意表达式作为初始值，这是一个初始化引用的例外，只要改表达式的结果能转化为引用类型。允许常量引用绑定非常量的对象、字面值或表达式。

```C++
int i = 42;
const int &r1 = i;
const int &r2 = 42;
const int &r3  = r2 * 2;
// int &r4 = r1 * 2; 非常量引用不可绑定字面值。

double dval = 3.14;
const int &ri = dval;
// 实际上创建了一个临时量，用来和ri绑定
const int temp = dval;
const int &ri = temp;
```

可知，若非常量引用来绑定一个 dval 是错误的。

- const 变量可能引用一个非 const 对象
常量引用仅对可参与的操作限定，对引用的对象本身是否是常量不限定。因为对象可能是非常量，可以通过其他途径改变值。

```c++
int i = 42;
int &r1 = i;
const int &r2 = i;
r1 = 0;
```

# Ch3 字符串、向量、数组

# 3.3 标准库类型 Vector

vector 表示对象集合，也称容器。所有对象类型相同，每个元素都有一个

索引，索引用于访问对象。

vector 是一个类模板，编译器根据模板创建类也即实例化。使用模板需要指出实例化为何种类型。vector 能容纳绝大多类型的对象作为元素，但引用不是对象，所以不可以包含引用。

## 定义初始化

|                           | 初始化方法              |
| ------------------------- | ------------------ |
| vector\<T\> v1;           | T 类型空对象，默认初始化       |
| vector\<T\> v2(v1);       | v2 包含 v1 的所有副本        |
| vector\<T\> v2=v1         | 与上等价               |
| vector\<T\> v3(n, val)    | v3 包含 n 个重复的元素，每个值 val |
| vector\<T\>v4(n)          | v4 包含 n 个重复执行了值初始化的对象 |
| vector\<T\>v5{a, b, …}  | v5 每个元素被赋予相应初始值。|
| vector\<T\>v5={a, b, …} | 与上等价               |

拷贝初始化 (=) 只能提供一个初始值；提供类内初始值只能使用拷贝初始化或花括号初始化；提供列表初始值不能用圆括号而是花括号。圆括号是指定数量和值的初始化。当然若花括号中的类型不能用于列表初始化，编译器会尝试值初始化。

## 添加元素

## 其他操作

# 表达式、语句、函数

# 类

# II C++ 标准库

# Ch 8 IO 库

## 8.2 文件输入输出

`fstream` 定义了一个三个类型来支持文件 IO:` ifstream/ofstreawm/ fstream`

这些类型提供的操作和 `cin/ cout` 一样，可以用 `<< >>` 来读写，也可以用 `getline` 从 `ifstream` 读取。

除了继承自 `iostream` 的行为，`fstream` 还有其他操作

|                   | fstream 特有的操作                                                             |
| ----------------- | ------------------------------------------------------------------------ |
| fstream fstrm;    | 创建一个未绑定的文件流                                                              |
| fstream fstrm(s); | 创建并打开 s，s 可以是 string 类型，也可以是指向 C 字符串的指针。这些构造函数都是 explicit 的，默认的文件模式依赖于 fstream 类型 |

# Ch 9 顺序容器

容器就是一些特定类型对象的集合。顺序容器有控制元素和访问顺序的能力，而不依赖元素值，与元素加入的位置对应。

泛型算法

# Ch 11 关联容器

关联容器中的元素**按照关键字**保存和访问。顺序容器中的元素是按它们在容器中的**位置**来顺序保存和访问的。关联容器支持高效关键字查找和访问。两个主要的类型是 `map, set`。map 中的元素是一些关键字 - 值（key-value）对：关键字起到索引的作用，值则表示与索引相关联的数据。set 中每个元素只包含一个关键字；set 支持高效的关键字查询操作——检查一个给定关键字是否在 set 中。

| 关联容器       |                      |
| ---------- | -------------------- |
| map        | 关键数组，保存 key-value 对。|
| set        | 关键字即值，只保存关键字。|
| multi-     | 关键字可重复出现             |
| unordered_ | 用哈希函数组织。不保持关键字按顺序存储。|

例子：

```C++
map<string, size_t> word_count;
string word;
while(cin >> word)
	++word_count[word];
for(const auto &w: word_count)
	cout << w.first <<"occurs"<< w.second << "time(s)" << endl;

map<string, size_t> word_count;
set<string> exlucde ={"The", "an"};
string word;
while(cin >> word)
	if(exclude.find(word) == exclude.end())
		++word_count[word];
```

## 概述

- 类似顺序容器，关联容器也是模板，必须指定类型。
- 支持**普通容器**操作，不支持**顺序容器相关操作**如 push_front, push_back。
- 支持一些关联操作和类型别名，可以提供一些调整哈希性能的操作。
- 迭代器都是双向的。
定义容器，可以创建空容器，初始化为拷贝，从一个值范围初始化，只要这些值可以转为为容器类型。也可以进行值初始化。

### 关键字类型要求

对于有序容器（map, multimap, set, multiset），关键字类型必须定义元素比较的方法，默认标准库使用关键字类型的<运算符来比较。

有序容器，可以提供一个自定义的比较操作，必须定义是严格弱序，可看作“小于等于”——两个关键字不能同时小于等于除非等价（即看作相同）、具有传递性。

比较函数也是**类型**的一部分，如。

```c++
bool compareIsbn(const Sales_data &lhs, const Sales_data &rhs){
return lhs.isbn() < rhs.isbn();
}

multiset<Sales_data, decltype(compareIsbn)*> bookstore(compareIsbn);
```

比较操作类型——是函数指针，利用 decltype 指出定义操作类型，且需要加上 *。也可以使用 `&compareIsbn`

### Pair

包含在 utility 中。一个 pair 保存两个数据，是用来生成特定类型的模板。默认构造函数对数据成员值初始化，也可以提供初始化器。

```c++
pair<string, string> author{"James", "Joyce"};
cout << w.first << w.second << endl;
```

pair 的数据成员是 public 的，两个成员分别命名为 `first, second`，可以用成员访问符号访问

| pair<T1, T2> p;           | 定义                    |
| ------------------------- | --------------------- |
| pair<T1, T2> p(v1, v2))   | 用 v1, v2 初始化            |
| pair<T1, T2> p = {v1, v2} | 与上等价                  |
| make_pair(v1, v2)         | 返回通过推理类型的 pair         |
| p.first/p.second          |                       |
| p1 relop p2               | 关系运算符（< > <= >=），按字典序 |
| p1 ==/ != p2              | 判断两个成员是否都相等           |

## 操作

set 的 key 即为 value，map 中元素是 key-value 对，而其 key 又是 const 无法改变。

| 额外的类型别名     |                                                           |
| ----------- | --------------------------------------------------------- |
| key_type    | 此容器类型的关键字类型                                               |
| mapped_type | 每个关键字关联的类型；只适用于 map                                        |
| value_type  | 对 set，与 key_type<br>对 map，为 pair<const key_type, mapped_type> |

```c++
map<string, int>::mapped_type v;
set<string>::value_type v;
```

### 1 迭代器

### 2 添加

使用 `insert` 插入，若不包含重复关键字，则无影响。

### 3 删除

|               | 删除元素                                     |
| ------------- | ---------------------------------------- |
| c.erase(k)    | 删除关键字为 k 的元素，返回 size_type 值，指出删除元素的数量        |
| c.erase(p)    | 删除迭代器 p 指定的元素，p 不能指向 c.end()，返回一个指向 p 之后元素的迭代器 |
| c.erase(b, e) | 删除迭代器 b, e 范围中的元素，返回 e                      |

### 4 下标

### 5 访问

## 无序

动态内存

## 类设计

拷贝构造

重载

OOP

# Ch10 泛型算法

标准库定义了一些通用算法，可作用于不同类型容器和不同类型元素。

## 10.1 概述

大多定义在 `algorithm` 中，`nuumeric` 还定义了一组数值泛型算法。一般这些算法不直接操作容器，而是遍历由两个迭代器指定的一个元素范围。

```c++
// 
int val = 42;
auto result = find(vec.begin(), vec.end(), val);

cou << val << (result == vec.end() ? " is not present" : " is present") << endl;
```

如 find 函数返回第一个找到的等于给定值的迭代器，若无则返回第二个参数代表失败。

由于操作的是迭代器，我们可以用同样的函数在任何容器包括数组上使用。

## 10.2 认识

## 17.3 正则表达式

regular expression 是一种描述字符序列的方法。RE 库定义在头文件 regex 中。
