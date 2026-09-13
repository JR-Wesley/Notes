https://zhuanlan.zhihu.com/p/613356779

# 句柄和指针

我们把问题拆成三块：  
1. 句柄 / 指针本身是什么、存在哪；  
2. 它们指向的内存空间在哪里；  
3. 如果直接取内容会怎样、为什么错。  

下面按这三块给出简洁回答。

---

### 1. 句柄和指针本身是什么、存在哪  
| 名称 | 本质 | 典型存储位置（CPU 端） |
|---|---|---|
| 设备指针（如 CUDA `float* d_data`） | 64 位虚拟地址，由驱动填入 CPU 可寻址的统一虚拟地址空间 | 局部变量 → CPU 栈；全局变量 → CPU 数据段；动态分配 → CPU 堆 |
| 句柄（如 CUDA `cudaEvent_t`、`cl_mem`） | 由运行时系统分配的小整数或结构体，用于在驱动 / 内核中索引资源 | 同上，存在 CPU 的栈 / 堆 / 数据段 |

---

### 2. 它们指向的内存空间在哪里  
- **设备指针**  
  - 指向 GPU 的全局内存（Global Memory），位于 GPU 显存芯片，通过页表映射到统一虚拟地址空间。  
- **句柄**  
  - 间接指向 GPU 上的缓冲区、事件、上下文等对象，对象实际存储在 GPU 显存或驱动的管理结构中。

---

### 3. 如果直接取内容会怎样、为什么错  
- **在 CPU 上直接解引用设备指针**  
  - 若系统不支持统一寻址（如旧架构），会触发段错误；若支持统一寻址，会触发缺页异常，由驱动把对应页迁移到 CPU 可访问的物理页，再继续执行。  
- **在 CPU 上直接解引用句柄**  
  - 句柄不是指针，直接当地址用会触发段错误；必须通过 API（如 `clEnqueueReadBuffer`）把句柄对应的 GPU 数据显式拷回 CPU 内存后才能访问。

# Effective C++

1. 视 C++ 为一个语言联邦（C、Object-Oriented C++、Template C++、STL）
2. 宁可以编译器替换预处理器（尽量以 `const`、`enum`、`inline` 替换 `#define`）
3. 尽可能使用 const
4. 确定对象被使用前已先被初始化（构造时赋值（copy 构造函数）比 default 构造后赋值（copy assignment）效率高）
5. 了解 C++ 默默编写并调用哪些函数（编译器暗自为 class 创建 default 构造函数、copy 构造函数、copy assignment 操作符、析构函数）
6. 若不想使用编译器自动生成的函数，就应该明确拒绝（将不想使用的成员函数声明为 private，并且不予实现）
7. 为多态基类声明 virtual 析构函数（如果 class 带有任何 virtual 函数，它就应该拥有一个 virtual 析构函数）
8. 别让异常逃离析构函数（析构函数应该吞下不传播异常，或者结束程序，而不是吐出异常；如果要处理异常应该在非析构的普通函数处理）
9. 绝不在构造和析构过程中调用 virtual 函数（因为这类调用从不下降至 derived class）
10. 令 `operator=` 返回一个 `reference to *this` （用于连锁赋值）
11. 在 `operator=` 中处理 “自我赋值”
12. 赋值对象时应确保复制 “对象内的所有成员变量” 及 “所有 base class 成分”（调用基类复制构造函数）
13. 以对象管理资源（资源在构造函数获得，在析构函数释放，建议使用智能指针，资源取得时机便是初始化时机（Resource Acquisition Is Initialization，RAII））
14. 在资源管理类中小心 copying 行为（普遍的 RAII class copying 行为是：抑制 copying、引用计数、深度拷贝、转移底部资源拥有权（类似 auto_ptr））
15. 在资源管理类中提供对原始资源（raw resources）的访问（对原始资源的访问可能经过显式转换或隐式转换，一般而言显示转换比较安全，隐式转换对客户比较方便）
16. 成对使用 new 和 delete 时要采取相同形式（`new` 中使用 `[]` 则 `delete []`，`new` 中不使用 `[]` 则 `delete`）
17. 以独立语句将 newed 对象存储于（置入）智能指针（如果不这样做，可能会因为编译器优化，导致难以察觉的资源泄漏）
18. 让接口容易被正确使用，不易被误用（促进正常使用的办法：接口的一致性、内置类型的行为兼容；阻止误用的办法：建立新类型，限制类型上的操作，约束对象值、消除客户的资源管理责任）
19. 设计 class 犹如设计 type，需要考虑对象创建、销毁、初始化、赋值、值传递、合法值、继承关系、转换、一般化等等。
20. 宁以 pass-by-reference-to-const 替换 pass-by-value （前者通常更高效、避免切割问题（slicing problem），但不适用于内置类型、STL 迭代器、函数对象）
21. 必须返回对象时，别妄想返回其 reference（绝不返回 pointer 或 reference 指向一个 local stack 对象，或返回 reference 指向一个 heap-allocated 对象，或返回 pointer 或 reference 指向一个 local static 对象而有可能同时需要多个这样的对象。）
22. 将成员变量声明为 private（为了封装、一致性、对其读写精确控制等）
23. 宁以 non-member、non-friend 替换 member 函数（可增加封装性、包裹弹性（packaging flexibility）、机能扩充性）
24. 若所有参数（包括被 this 指针所指的那个隐喻参数）皆须要类型转换，请为此采用 non-member 函数
25. 考虑写一个不抛异常的 swap 函数
26. 尽可能延后变量定义式的出现时间（可增加程序清晰度并改善程序效率）
27. 尽量少做转型动作（旧式：`(T)expression`、`T(expression)`；新式：`const_cast<T>(expression)`、`dynamic_cast<T>(expression)`、`reinterpret_cast<T>(expression)`、`static_cast<T>(expression)`、；尽量避免转型、注重效率避免 dynamic_casts、尽量设计成无需转型、可把转型封装成函数、宁可用新式转型）
28. 避免使用 handles（包括 引用、指针、迭代器）指向对象内部（以增加封装性、使 const 成员函数的行为更像 const、降低 “虚吊号码牌”（dangling handles，如悬空指针等）的可能性）
29. 为 “异常安全” 而努力是值得的（异常安全函数（Exception-safe functions）即使发生异常也不会泄露资源或允许任何数据结构败坏，分为三种可能的保证：基本型、强列型、不抛异常型）
30. 透彻了解 inlining 的里里外外（inlining 在大多数 C++ 程序中是编译期的行为；inline 函数是否真正 inline，取决于编译器；大部分编译器拒绝太过复杂（如带有循环或递归）的函数 inlining，而所有对 virtual 函数的调用（除非是最平淡无奇的）也都会使 inlining 落空；inline 造成的代码膨胀可能带来效率损失；inline 函数无法随着程序库的升级而升级）
31. 将文件间的编译依存关系降至最低（如果使用 object references 或 object pointers 可以完成任务，就不要使用 objects；如果能够，尽量以 class 声明式替换 class 定义式；为声明式和定义式提供不同的头文件）
32. 确定你的 public 继承塑模出 is-a（是一种）关系（适用于 base classes 身上的每一件事情一定适用于 derived classes 身上，因为每一个 derived class 对象也都是一个 base class 对象）
33. 避免遮掩继承而来的名字（可使用 using 声明式或转交函数（forwarding functions）来让被遮掩的名字再见天日）
34. 区分接口继承和实现继承（在 public 继承之下，derived classes 总是继承 base class 的接口；pure virtual 函数只具体指定接口继承；非纯 impure virtual 函数具体指定接口继承及缺省实现继承；non-virtual 函数具体指定接口继承以及强制性实现继承）
35. 考虑 virtual 函数以外的其他选择（如 Template Method 设计模式的 non-virtual interface（NVI）手法，将 virtual 函数替换为 “函数指针成员变量”，以 `tr1::function` 成员变量替换 virtual 函数，将继承体系内的 virtual 函数替换为另一个继承体系内的 virtual 函数）
36. 绝不重新定义继承而来的 non-virtual 函数
37. 绝不重新定义继承而来的缺省参数值，因为缺省参数值是静态绑定（statically bound），而 virtual 函数却是动态绑定（dynamically bound）
38. 通过复合塑模 has-a（有一个）或 “根据某物实现出”（在应用域（application domain），复合意味 has-a（有一个）；在实现域（implementation domain），复合意味着 is-implemented-in-terms-of（根据某物实现出））
39. 明智而审慎地使用 private 继承（private 继承意味着 is-implemented-in-terms-of（根据某物实现出），尽可能使用复合，当 derived class 需要访问 protected base class 的成员，或需要重新定义继承而来的时候 virtual 函数，或需要 empty base 最优化时，才使用 private 继承）
40. 明智而审慎地使用多重继承（多继承比单一继承复杂，可能导致新的歧义性，以及对 virtual 继承的需要，但确有正当用途，如 “public 继承某个 interface class” 和 “private 继承某个协助实现的 class”；virtual 继承可解决多继承下菱形继承的二义性问题，但会增加大小、速度、初始化及赋值的复杂度等等成本）
41. 了解隐式接口和编译期多态（class 和 templates 都支持接口（interfaces）和多态（polymorphism）；class 的接口是以签名为中心的显式的（explicit），多态则是通过 virtual 函数发生于运行期；template 的接口是奠基于有效表达式的隐式的（implicit），多态则是通过 template 具现化和函数重载解析（function overloading resolution）发生于编译期）
42. 了解 typename 的双重意义（声明 template 类型参数是，前缀关键字 class 和 typename 的意义完全相同；请使用关键字 typename 标识嵌套从属类型名称，但不得在基类列（base class lists）或成员初值列（member initialization list）内以它作为 base class 修饰符）
43. 学习处理模板化基类内的名称（可在 derived class templates 内通过 `this->` 指涉 base class templates 内的成员名称，或藉由一个明白写出的 “base class 资格修饰符” 完成）
44. 将与参数无关的代码抽离 templates（因类型模板参数（non-type template parameters）而造成代码膨胀往往可以通过函数参数或 class 成员变量替换 template 参数来消除；因类型参数（type parameters）而造成的代码膨胀往往可以通过让带有完全相同二进制表述（binary representations）的实现类型（instantiation types）共享实现码）
45. 运用成员函数模板接受所有兼容类型（请使用成员函数模板（member function templates）生成 “可接受所有兼容类型” 的函数；声明 member templates 用于 “泛化 copy 构造” 或 “泛化 assignment 操作” 时还需要声明正常的 copy 构造函数和 copy assignment 操作符）
46. 需要类型转换时请为模板定义非成员函数（当我们编写一个 class template，而它所提供之 “与此 template 相关的” 函数支持 “所有参数之隐式类型转换” 时，请将那些函数定义为 “class template 内部的 friend 函数”）
47. 请使用 traits classes 表现类型信息（traits classes 通过 templates 和 “templates 特化” 使得 “类型相关信息” 在编译期可用，通过重载技术（overloading）实现在编译期对类型执行 if…else 测试）
48. 认识 template 元编程（模板元编程（TMP，template metaprogramming）可将工作由运行期移往编译期，因此得以实现早期错误侦测和更高的执行效率；TMP 可被用来生成 “给予政策选择组合”（based on combinations of policy choices）的客户定制代码，也可用来避免生成对某些特殊类型并不适合的代码）
49. 了解 new-handler 的行为（set_new_handler 允许客户指定一个在内存分配无法获得满足时被调用的函数；nothrow new 是一个颇具局限的工具，因为它只适用于内存分配（operator new），后继的构造函数调用还是可能抛出异常）
50. 了解 new 和 delete 的合理替换时机（为了检测运用错误、收集动态分配内存之使用统计信息、增加分配和归还速度、降低缺省内存管理器带来的空间额外开销、弥补缺省分配器中的非最佳齐位、将相关对象成簇集中、获得非传统的行为）
51. 编写 new 和 delete 时需固守常规（operator new 应该内涵一个无穷循环，并在其中尝试分配内存，如果它无法满足内存需求，就应该调用 new-handler，它也应该有能力处理 0 bytes 申请，class 专属版本则还应该处理 “比正确大小更大的（错误）申请”；operator delete 应该在收到 null 指针时不做任何事，class 专属版本则还应该处理 “比正确大小更大的（错误）申请”）
52. 写了 placement new 也要写 placement delete（当你写一个 placement operator new，请确定也写出了对应的 placement operator delete，否则可能会发生隐微而时断时续的内存泄漏；当你声明 placement new 和 placement delete，请确定不要无意识（非故意）地遮掩了它们地正常版本）
53. 不要轻忽编译器的警告
54. 让自己熟悉包括 TR1 在内的标准程序库（TR1，C++ Technical Report 1，C++11 标准的草稿文件）
55. 让自己熟悉 Boost（准标准库）

# More Effective c++[](https://kalosaner.github.io/2025/04/07/Effective-C++-%E5%92%8C-More-Effective-C++-%E6%80%BB%E7%BB%93/#more-effective-c)

1. 仔细区别 pointers 和 references（当你知道你需要指向某个东西，而且绝不会改变指向其他东西，或是当你实现一个操作符而其语法需求无法由 pointers 达成，你就应该选择 references；任何其他时候，请采用 pointers）
2. 最好使用 C++ 转型操作符（`static_cast`、`const_cast`、`dynamic_cast`、`reinterpret_cast`）
3. 绝不要以多态（polymorphically）方式处理数组（多态（polymorphism）和指针算术不能混用；数组对象几乎总是会涉及指针的算术运算，所以数组和多态不要混用）
4. 非必要不提供 default constructor（避免对象中的字段被无意义地初始化）
5. 对定制的 “类型转换函数” 保持警觉（单自变量 constructors 可通过简易法（explicit 关键字）或代理类（proxy classes）来避免编译器误用；隐式类型转换操作符可改为显式的 member function 来避免非预期行为）
6. 区别 increment/decrement 操作符的前置（prefix）和后置（postfix）形式（前置式累加后取出，返回一个 reference；后置式取出后累加，返回一个 const 对象；处理用户定制类型时，应该尽可能使用前置式 increment；后置式的实现应以其前置式兄弟为基础）
7. 千万不要重载 `&&`，`||` 和 `,` 操作符（`&&` 与 `||` 的重载会用 “函数调用语义” 取代 “骤死式语义”；`,` 的重载导致不能保证左侧表达式一定比右侧表达式更早被评估）
8. 了解各种不同意义的 new 和 delete（`new operator`、`operator new`、`placement new`、`operator new[]`；`delete operator`、`operator delete`、`destructor`、`operator delete[]`）
9. 利用 destructors 避免泄漏资源（在 destructors 释放资源可以避免异常时的资源泄漏）
10. 在 constructors 内阻止资源泄漏（由于 C++ 只会析构已构造完成的对象，因此在构造函数可以使用 try…catch 或者 auto_ptr（以及与之相似的 classes） 处理异常时资源泄露问题）
11. 禁止异常流出 destructors 之外（原因：一、避免 terminate 函数在 exception 传播过程的栈展开（stack-unwinding）机制种被调用；二、协助确保 destructors 完成其应该完成的所有事情）
12. 了解 “抛出一个 exception” 与 “传递一个参数” 或 “调用一个虚函数” 之间的差异（第一，exception objects 总是会被复制（by pointer 除外），如果以 by value 方式捕捉甚至被复制两次，而传递给函数参数的对象则不一定得复制；第二，“被抛出成为 exceptions” 的对象，其被允许的类型转换动作比 “被传递到函数去” 的对象少；第三，catch 子句以其 “出现于源代码的顺序” 被编译器检验对比，其中第一个匹配成功者便执行，而调用一个虚函数，被选中执行的是那个 “与对象类型最佳吻合” 的函数）
13. 以 by reference 方式捕获 exceptions（可避免对象删除问题、exception objects 的切割问题，可保留捕捉标准 exceptions 的能力，可约束 exception object 需要复制的次数）
14. 明智运用 exception specifications（exception specifications 对 “函数希望抛出什么样的 exceptions” 提供了卓越的说明；也有一些缺点，包括编译器只对它们做局部性检验而很容易不经意地违反，与可能会妨碍更上层的 exception 处理函数处理未预期的 exceptions）
15. 了解异常处理的成本（粗略估计，如果使用 try 语句块，代码大约整体膨胀 5%-10%，执行速度亦大约下降这个数；因此请将你对 try 语句块和 exception specifications 的使用限制于非用不可的地点，并且在真正异常的情况下才抛出 exceptions）
16. 谨记 80-20 法则（软件的整体性能几乎总是由其构成要素（代码）的一小部分决定的，可使用程序分析器（program profiler）识别出消耗资源的代码）
17. 考虑使用 lazy evaluation（缓式评估）（可应用于：Reference Counting（引用计数）来避免非必要的对象复制、区分 operator[] 的读和写动作来做不同的事情、Lazy Fetching（缓式取出）来避免非必要的数据库读取动作、Lazy Expression Evaluation（表达式缓评估）来避免非必要的数值计算动作）
18. 分期摊还预期的计算成本（当你必须支持某些运算而其结构几乎总是被需要，或其结果常常被多次需要的时候，over-eager evaluation（超急评估）可以改善程序效率）

# typename


```cpp
template<typename... Args>
// vs
template<class... Ts>
```

在大多数情况下，`typename` 和 `class` 在模板参数声明中**可以互换使用**，它们的作用是等价的。但这背后有一些历史、语义和细微差别的考量。

我们来详细解析：

---

## 一、基本等价性：`typename` 和 `class` 可以互换

### ✅ 语法上完全等价

```cpp
template<typename T>      struct A1 { };
template<class T>         struct A2 { };

template<typename... Ts>  struct B1 { };
template<class... Ts>     struct B2 { };
```

上面四种写法在**模板参数声明**中是完全等价的。编译器把它们视为相同。

### 为什么？
因为 C++ 标准规定：在模板参数列表中，`typename` 和 `class` 都表示“这里是一个类型占位符”，不区分是内置类型（如 `int`）、类类型（如 `std::string`）还是其他类型。

---

## 二、为什么会有两个关键字？历史原因

### 1. **`class` 先出现**
- C++ 最早引入模板时（1980s），只用了 `class` 关键字。
- 例如：
  ```cpp
  template<class T>
  T max(T a, T b) { return a > b ? a : b; }
  ```

### 2. **`typename` 后来引入**
- 随着模板复杂度增加，出现了嵌套依赖类型的问题。
- 例如：
  ```cpp
  template<class T>
  void foo() {
      T::iterator * p;  // 这是声明一个指针？还是乘法？
  }
  ```
  编译器不知道 `T::iterator` 是不是一个类型，所以需要一个关键字来**明确告诉编译器：“这是一个类型”**。

- 于是 C++98 引入了 `typename` 用于**在模板中声明某个嵌套名是一个类型**：
  ```cpp
  template<class T>
  void foo() {
      typename T::iterator * p;  // 明确告诉编译器 T::iterator 是一个类型
  }
  ```

### 3. **但 `class` 已经被广泛使用**
- 为了向后兼容，C++ 标准保留了 `class` 作为模板参数的声明方式。
- 所以现在有两种方式声明类型模板参数。

---

## 三、`typename` vs `class`：使用建议

虽然等价，但社区和现代 C++ 风格倾向于：

| 情况 | 推荐 | 原因 |
|------|------|------|
| **模板参数声明** | ✅ `typename` | 语义更清晰，表示“任意类型”，不限于 `class` 类型 |
| **嵌套依赖类型** | ✅ 必须用 `typename` | `class` 不能用于此场景 |

### ✅ 推荐写法（现代 C++ 风格）

```cpp
template<typename T>
using MyVector = std::vector<T>;

template<typename... Args>
auto make_tuple(Args&&... args) {
    return std::make_tuple(std::forward<Args>(args)...);
}
```

### ❌ 不推荐（但合法）

```cpp
template<class T>
using MyVector = std::vector<T>;  // 虽然合法，但语义稍弱
```

---

## 四、为什么 `typename` 语义更好？

### 1. `class` 容易误导
- `class` 这个词让人以为只能用于“类类型”（如 `std::string`）。
- 但实际上，模板可以接受**任何类型**，包括：
  - 内置类型：`int`, `double`, `char*`
  - 基本类型：`bool`, `void*`
  - 函数指针、数组等

```cpp
max(3, 5);        // T = int，不是 class
max(3.14, 2.71);  // T = double，不是 class
```

用 `typename` 更准确地表达了“这是一个类型占位符”的含义。

### 2. `typename` 更一致
- `typename` 也用于嵌套依赖类型，统一使用 `typename` 可以减少认知负担。
- 例如：
  ```cpp
  template<typename T>
  void foo() {
      typename T::value_type x;  // 这里必须用 typename
  }
  ```

---

## 五、`class` 何时更合适？（极少情况）

只有在你**明确想限制模板只接受类类型**时，用 `class` 有文档作用，但实际并不能限制（因为 `int` 也可以传入）。

所以本质上，这只是**风格选择**。

---

## 六、总结

| 问题 | 回答 |
|------|------|
| `typename` 和 `class` 可以互换吗？ | ✅ **是的**，在模板参数声明中完全等价 |
| 为什么有两个关键字？ | 📜 历史原因：`class` 先引入，`typename` 后来为解决嵌套类型问题引入 |
| 推荐用哪个？ | ✅ **推荐 `typename`**，语义更清晰，表示“任意类型” |
| `class` 有什么问题？ | ⚠️ 容易误导，以为只能用于类类型，但实际上不是限制 |
| `typename` 用在哪些地方？ | 1. 模板参数声明<br>2. 嵌套依赖类型（如 `typename T::iterator`） |

---

### 🎯 最终建议

在现代 C++ 中，**统一使用 `typename`**：

```cpp
template<typename T>
struct MyClass { };

template<typename... Args>
auto make_something(Args&&... args);
```

这不仅是主流风格（如 Google C++ Style Guide、Effective C++ 推荐），也更符合“模板是关于类型的通用编程”这一思想。

您看到的代码中同时出现 `typename... Args` 和 `class... Ts`，可能是不同开发者风格不同，或者库的历史原因，但从语义上讲，**推荐统一使用 `typename`**。
