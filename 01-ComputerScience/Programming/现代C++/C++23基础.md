---
dateCreated: 2025-07-10
dateModified: 2025-07-11
---

<a href="git@github.com : Extra-Creativity/Modern-Cpp-Basics. git">北大学生的现代 C++ 教程</a>

# 基础
## size_t

在 C++ 里，`size_t` 属于无符号整数类型，它的设计目的是用于表示对象或者数据结构的大小。下面为你详细介绍它的相关内容：

### 基本情况

- **定义出处**：`size_t` 在标准库的多个头文件（像 `<cstddef>`、`<cstdio>`、`<cstring>`、`<vector>` 等）中均有定义。
- **类型本质**：它是无符号整数类型，这意味着其取值范围是从 0 开始的正整数。
- **具体用途**：在 C++ 里，`size_t` 主要用于以下场景：
    - 表示数组、字符串或者容器的大小。
    - 作为循环计数器，尤其是在遍历数组或容器时。
    - 用于内存分配函数（例如 `malloc`）和字符串处理函数（例如 `strlen`）的返回值类型。

### 关键特性

- **平台适配性**：`size_t` 的宽度会依据不同平台而有所变化。在 32 位系统中，它通常等同于 `unsigned int`；在 64 位系统中，则一般和 `unsigned long long` 一样。这种特性保证了它能够适应不同平台的内存寻址需求。
- **防止负值问题**：由于 `size_t` 是无符号类型，使用它可以避免在表示大小时出现负值的情况。
- **标准兼容性**：C++ 标准规定，`size_t` 必须能够表示任何对象的最大可能大小。

## 右值引用

在 C++ 中，**右值引用（Rvalue Reference）** 是 C++11 引入的核心特性之一，它主要用于实现 **移动语义（Move Semantics）** 和 **完美转发（Perfect Forwarding）**，从而大幅提升程序性能并简化模板编程。以下是对右值引用及其相关内容的详细介绍：

### **一、基本概念：左值 Vs 右值**

#### 1. **左值（Lvalue）**

- **定义**：可寻址、有持久存储位置的表达式。
- **特点**：可出现在赋值语句左侧，生命周期超出当前表达式。
- **示例**：

    ```cpp
    int x = 10;       // x是左值
    std::string s;
    s = "hello";      // s是左值
    ```

#### 2. **右值（Rvalue）**

- **定义**：不可寻址、临时的表达式（如字面量、函数返回的临时对象）。
- **特点**：只能出现在赋值语句右侧，生命周期仅限于当前表达式。
- **示例**：

    ```cpp
    42;               // 字面量是右值
    std::string("hi");// 临时对象是右值
    x + 5;            // 表达式结果是右值
    ```

#### 3. **C++11 对右值的细分**

- **纯右值（Prvalue）**：传统意义上的右值（如字面量、临时对象）。
- **将亡值（xvalue）**：通过 `std::move` 或 `static_cast<T&&>` 显式转换的对象，即将被移动的资源。

### **二、右值引用的语法**

右值引用使用 **`&&`** 声明，专门绑定到右值（临时对象）：

```cpp
int&& rref = 42;          // 绑定到纯右值
std::string s = "hello";
std::string&& rref_s = std::move(s);  // 绑定到将亡值（通过std::move转换）
```

#### 关键特性

1. **延长临时对象生命周期**：

```cpp
    std::vector<int>&& rref = std::vector<int>{1, 2, 3};  // rref延长了临时vector的生命周期
    ```

2. **不能直接绑定到左值**：

```cpp
    int x = 10;
    int&& rref = x;  // 错误：右值引用不能绑定到左值
    int&& rref = std::move(x);  // 正确：通过std::move将左值转为右值引用
    ```

### **三、移动语义（Move Semantics）**

#### 1. **动机**

传统的拷贝构造函数在处理临时对象时会进行不必要的深拷贝，而移动语义允许直接转移资源所有权（如内存、文件句柄），避免拷贝开销。

#### 2. **移动构造函数**

语法：`ClassName(ClassName&& other) noexcept`

```cpp
class MyVector {
private:
    int* data;
    size_t size;
public:
    // 移动构造函数
    MyVector(MyVector&& other) noexcept : data(other.data), size(other.size) {
        other.data = nullptr;  // 释放原对象的资源
        other.size = 0;
    }
};
```

#### 3. **移动赋值运算符**

语法：`ClassName& operator=(ClassName&& other) noexcept`

```cpp
MyVector& operator=(MyVector&& other) noexcept {
    if (this != &other) {
        delete[] data;        // 释放当前资源
        data = other.data;    // 转移资源所有权
        size = other.size;
        other.data = nullptr; // 清空原对象
        other.size = 0;
    }
    return *this;
}
```

#### 4. **移动语义的触发条件**

- **临时对象作为右值**：

    ```cpp
    MyVector createVector() { return MyVector(); }
    MyVector v = createVector();  // 调用移动构造函数
    ```

- **显式使用 `std::move`**：

    ```cpp
    MyVector v1, v2;
    v1 = std::move(v2);  // 调用移动赋值运算符
    ```

### **四、`std::move` 和 `std::forward`**

#### 1. **`std::move`**

- **作用**：将左值强制转换为右值引用，用于触发移动语义。
- **本质**：`static_cast<T&&>(x)` 的包装。
- **示例**：

    ```cpp
    std::vector<int> v1 = {1, 2, 3};
    std::vector<int> v2 = std::move(v1);  // 转移v1的资源到v2
    ```

#### 2. **`std::forward`（完美转发）**

- **作用**：在模板函数中保持参数的原始值类别（左值 / 右值）。
- **典型场景**：转发构造参数到另一个函数。
- **示例**：

    ```cpp
    template<typename T, typename… Args>
    std::unique_ptr<T> make_unique(Args&&… args) {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)…));
    }
    ```

### **五、引用折叠与万能引用（Universal Reference）**

#### 1. **引用折叠规则**

- `T& &` → `T&`
- `T& &&` → `T&`
- `T&& &` → `T&`
- `T&& &&` → `T&&`

#### 2. **万能引用（转发引用）**

当模板参数推导为 `T&&` 时，可接受任意类型的参数（左值或右值）：

```cpp
template<typename T>
void f(T&& arg) {  // arg是万能引用
    g(std::forward<T>(arg));  // 完美转发arg到g
}
```

**区分万能引用与右值引用**：

- 万能引用：`T&&` + 模板参数推导（如 `auto&&`、`template<typename T> void f(T&&)`）。
- 右值引用：明确类型的 `&&`（如 `int&&`、`std::vector<int>&&`）。

### **六、移动语义的应用场景**

#### 1. **容器优化**

`std::vector`、`std::string` 等容器在插入临时对象时使用移动语义：

```cpp
std::vector<std::string> vec;
vec.push_back("hello");  // 直接移动临时字符串，避免拷贝
```

#### 2. **智能指针转移所有权**

```cpp
std::unique_ptr<int> ptr1 = std::make_unique<int>(42);
std::unique_ptr<int> ptr2 = std::move(ptr1);  // 转移所有权
```

#### 3. **资源管理类**

自定义 RAII 类通过移动语义高效管理资源：

```cpp
class FileHandle {
public:
    FileHandle(FileHandle&& other) noexcept { … }  // 移动构造函数
};
```

### **七、右值引用的注意事项**

1. **移动后对象的状态**
    移动后的对象必须处于有效但未指定的状态（通常为空）。

2. **`noexcept` 声明**
    移动构造函数应标记为 `noexcept`，以允许容器在重新分配内存时安全使用。

3. **拷贝与移动的自动生成**

    - 若未定义移动构造函数，编译器会默认生成拷贝构造函数处理右值。
    - 若定义了移动构造函数或移动赋值运算符，编译器不会自动生成拷贝构造函数。

### **八、C++17 与 C++20 的增强**

#### 1. **`std::move_if_noexcept`**

优先使用移动语义，若移动构造函数非 `noexcept` 则回退到拷贝：

```cpp
T a;
T b = std::move_if_noexcept(a);  // 安全转移资源
```

#### 2. **临时对象的自动续命（C++17）**

```cpp
const auto& ref = std::vector<int>{1, 2, 3};  // 临时对象的生命周期延长到ref的作用域结束
```

#### 3. **右值引用的约束（C++20 Concepts）**

```cpp
template<typename T>
requires std::movable<T>
void process(T&& obj) { … }  // 限制T必须是可移动的
```

### **九、总结**

右值引用是现代 C++ 性能优化的核心工具，通过移动语义避免不必要的拷贝，通过完美转发简化模板编程。掌握右值引用需要理解：

1. **值类别**：左值、纯右值、将亡值的区别。
2. **移动语义**：移动构造函数和移动赋值运算符的实现。
3. **关键工具**：`std::move` 触发移动，`std::forward` 保持值类别。
4. **引用折叠**：理解万能引用的推导规则。

合理应用右值引用可显著提升代码性能，尤其是在处理大型对象和容器操作时。

## Lambda

在 C++ 中，lambda 表达式是一种匿名函数对象，允许你在需要的地方定义轻量级的可调用对象，无需显式编写函数或函数对象类。它在 C++11 中引入，之后不断增强（如 C++14、C++17、C++20 的扩展），现已成为现代 C++ 中不可或缺的特性。

### **1. 基本语法**

```cpp
[capture](parameters) -> return_type { body }
```

- **`[capture]`**：捕获列表，用于捕获外部变量。
- **`(parameters)`**：参数列表，与普通函数相同。
- **`-> return_type`**：返回类型，可省略（编译器自动推导）。
- **`{ body }`**：函数体，包含具体实现。

### **2. 捕获列表（Capture List）**

捕获列表允许 lambda 访问外部作用域的变量，有以下几种方式：

#### 1 值捕获（By Value）

```cpp
int x = 10;
auto lambda = [x]() { return x * 2; };  // 捕获x的值
std::cout << lambda() << std::endl;     // 输出：20
```

#### 2 引用捕获（By Reference）

```cpp
int x = 10;
auto lambda = [&x]() { x *= 2; };  // 捕获x的引用
lambda();
std::cout << x << std::endl;      // 输出：20
```

#### 3 隐式捕获

- `[=]`：以值方式捕获所有外部变量。
- `[&]`：以引用方式捕获所有外部变量。

```cpp
int a = 5, b = 10;
auto sum = [=]() { return a + b; };  // 值捕获a和b
auto multiply = [&]() { a *= b; };    // 引用捕获a和b
```

#### 4 混合捕获

```cpp
int a = 5, b = 10, c = 15;
auto lambda = [a, &b, &c]() {
    b += a;
    c -= a;
};  // 值捕获a，引用捕获b和c
```

### **3. 参数列表与返回类型**

#### 1 参数列表

与普通函数类似，但不能有默认参数：

```cpp
auto add = [](int a, int b) { return a + b; };
std::cout << add(3, 4) << std::endl;  // 输出：7
```

#### 2 返回类型

通常可省略，由编译器自动推导。如需显式指定，使用尾置返回类型：

```cpp
auto divide = [](double a, double b) -> double {
    return a / b;
};
```

### **4. 可变 lambda（Mutable Lambda）**

默认情况下，值捕获的变量在 lambda 内部是只读的。使用 `mutable` 关键字可修改它们：

```cpp
int x = 10;
auto lambda = [x]() mutable {
    x += 5;  // 修改值捕获的x
    std::cout << x << std::endl;  // 输出：15
};
lambda();
std::cout << x << std::endl;      // 外部x仍为10（值捕获不影响原变量）
```

### **5. 泛型 lambda（C++14 及以后）**

使用 `auto` 作为参数类型，创建模板化的 lambda：

```cpp
auto print = [](auto value) {
    std::cout << value << std::endl;
};

print(42);      // 输出：42
print("Hello"); // 输出：Hello
```

### **6. 应用场景**

#### 1 作为函数参数（如 STL 算法）

```cpp
#include <algorithm>
#include <vector>

std::vector<int> nums = {1, 2, 3, 4, 5};
int sum = 0;

// 计算总和
std::for_each(nums.begin(), nums.end(), [&sum](int num) {
    sum += num;
});

// 筛选偶数
auto even = std::find_if(nums.begin(), nums.end(), [](int num) {
    return num % 2 == 0;
});
```

#### 2 延迟执行

```cpp
auto delayedPrint = [](const std::string& msg) {
    return [msg]() { std::cout << msg << std::endl; };
};

auto printer = delayedPrint("延迟执行");
printer();  // 稍后调用
```

#### 3 自定义排序

```cpp
std::vector<std::pair<int, std::string>> people = {
    {25, "Alice"},
    {20, "Bob"},
    {30, "Charlie"}
};

// 按年龄排序
std::sort(people.begin(), people.end(), 
          [](const auto& a, const auto& b) {
              return a.first < b.first;
          });
```

### **7. C++14 扩展**

#### 1 初始化捕获（Init Capture）

允许在捕获列表中初始化新变量，甚至移动资源：

```cpp
auto lambda = [value = 42]() {
    return value;
};
std::cout << lambda() << std::endl;  // 输出：42

// 移动捕获（适用于不可复制的对象）
std::unique_ptr<int> ptr = std::make_unique<int>(100);
auto lambda2 = [ptr = std::move(ptr)]() {
    return *ptr;
};
```

### **8. C++17 扩展**

#### 1 `constexpr` Lambda

允许在编译时执行 lambda：

```cpp
constexpr auto square = [](int x) { return x * x; };
static_assert(square(5) == 25);  // 编译时检查
```

#### 2 结构化绑定捕获

```cpp
std::pair<int, std::string> p = {42, "Hello"};
auto lambda = [&[a, b] = p]() {
    a = 100;  // 修改p.first
};
```

### **9. C++20 扩展**

#### 模板参数列表

更灵活的泛型 lambda：

```cpp
auto lambda = []<typename T>(const T& value) {
    return value;
};
```

#### 概念约束（Concepts）

```cpp
#include <concepts>

auto lambda = []<std::integral T>(T value) {
    return value * 2;
};
```

### **10. 实现原理**

lambda 表达式本质上是编译器生成的匿名函数对象（functor）。例如：

```cpp
auto add = [](int a, int b) { return a + b; };
```

等价于：

```cpp
struct __lambda_ {
    int operator()(int a, int b) const {
        return a + b;
    }
};

auto add = __lambda_{};
```

### **11. 注意事项**

1. **生命周期风险**：引用捕获时需确保被引用的对象在 lambda 执行时仍然有效。
2. **性能考量**：值捕获会复制对象，大对象建议使用引用捕获。
3. **移动语义**：使用 `std::move` 捕获不可复制的资源（如 `std::unique_ptr`）。

### **总结**

Lambda 表达式是 C++ 中最强大的特性之一，它使代码更简洁、灵活，尤其在函数式编程和 STL 算法中表现出色。掌握 lambda 的捕获机制、泛型特性及各版本扩展，是成为现代 C++ 程序员的关键一步。
