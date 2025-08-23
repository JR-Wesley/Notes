---
dateCreated: 2025-08-16
dateModified: 2025-08-17
---

> [!note] 参考
> <a href="https://cntransgroup.github.io/EffectiveModernCppChinese/Introduction.html">高效现代 C++ 中文 </a>
> <a href="https://learn.microsoft.com/zh-cn/cpp/cpp/welcome-back-to-cpp-modern-cpp?view=msvc-170">微软现代 C++ 中文 </a>
> <a href="https://en.cppreference.com/w/cpp/23.html"> Cpp reference 23 </a>

# 现代 C++ 语法特性

## 一、基础语法

> 基础语法注重与 C 不同的特性。包括基础的数据类型、对函数的增强、一些其他特性。

### 数据类型
#### 强类型

C++ 的 “强类型” 特性体现在对类型的严格区分和对隐式转换的限制上，C 语言允许宽松的隐式转换（弱类型特征）。这种设计的核心目标是**在编译期发现类型不匹配的错误**，减少运行时异常，提升代码安全性。其核心思路是：**除非开发者明确指示（显式转换），否则编译器默认禁止可能不安全的类型混用**。

> [!important] 强类型语言的核心特征
> **不同类型之间的操作受到严格限制，类型转换（尤其是隐式转换）需要满足明确的规则，不允许随意的、可能导致歧义的类型混用**。

C++ 的强类型体现在：

- 每种类型有明确的语义和操作限制（如 `int` 和 `float` 不能随意混用）；
- 隐式转换仅在 “安全且无歧义” 的场景下允许（如 `int`→`long`）；
- 多数跨类型操作需要显式转换（如 `void*` → `int*`）。

##### 1. `void*` 的转换

- **C 语言**：`void*` 可以隐式转换为任何指针类型，无需显式转换
- **C++ 语言**：`void*` 不能隐式转换为其他指针类型，必须显式转换：

```cpp
void* ptr = new int[10];
// int* int_ptr = ptr;  // C++编译错误：必须允许void*隐式转换
int* int_ptr = static_cast<int*>(ptr);  // 必须显式转换
```

##### 2. 整数与指针的转换

- **C 语言**：整数和指针可以随意隐式转换（风险极高）：
- **C++ 语言**：禁止整数与指针的隐式转换，必须显式转换（且不推荐）：

```cpp
int x = 0x7fffffff;
// int* ptr = x;  // C++编译错误：整数不能隐式转为指针
int* ptr = reinterpret_cast<int*>(x);  // 必须显式转换（不推荐）
```

##### 3. 枚举类型的转换

- **C 语言**：枚举常量本质是 `int`，可与整数随意隐式转换：
- **C++ 语言**：枚举是独立类型，与整数的转换需显式进行：

```cpp
enum Color { RED, GREEN };
// Color c = 1;  // C++编译错误：整数不能隐式转为枚举
Color c = static_cast<Color>(1);  // 必须显式转换

// int x = RED;  // C++11前允许（兼容C），但现代C++建议显式转换
int x = static_cast<int>(RED);
```

更严格的是 C++11 引入的 `enum class`（强类型枚举），完全禁止与整数隐式转换：

```cpp
enum class Color { RED, GREEN };
// int x = Color::RED;  // 编译错误：强类型枚举不能隐式转为整数
```

##### 4. 布尔类型的转换

- **C 语言**：任何整数 / 指针都可隐式视为 “布尔值”（0 为假，非 0 为真）：
- **C++ 语言**：虽然也允许整数 / 指针隐式转为 `bool`（兼容 C 的常见场景），但限制更严格：
	- 仅允许 “零值→`false`，非零值→`true`” 的转换；
	- 反之，`bool` 转为整数时，`true` 固定为 1（C 中可能因实现不同而变化）：

	```cpp
	bool b = true;
	int x = b;  // C++中x必为1（C中可能是其他非零值）
	```

##### 5. 窄化转换（Narrowing Conversion）

- **C 语言**：允许隐式窄化转换（如 `double`→`int` 可能丢失小数部分）：
- **C++ 语言**：禁止隐式窄化转换（编译错误）：

```cpp
double d = 3.14;
// int x = d;  // C++编译错误：不允许double隐式转为int（窄化）
int x = static_cast<int>(d);  // 必须显式转换（明确告知编译器接受截断）
```

注意：列表初始化（`{}`）对窄化转换的检查更严格，完全禁止任何可能丢失信息的转换。

#### 自动类型推导（C++11）

C++11 及后续标准引入了**自动类型推导**机制，核心工具包括 `auto`、`decltype` 以及 C++14 新增的 `decltype(auto)`。这些特性大幅简化了代码编写（尤其是模板和泛型编程中），同时保持了类型安全性。

##### `auto`：根据初始化表达式推导变量类型

`auto` 的核心功能是**让编译器根据变量的初始化表达式自动推导其类型**，无需显式声明。其设计初衷是简化复杂类型（如模板迭代器、lambda 表达式类型）的声明。

- 基本用法与推导规则

`auto` 必须结合初始化表达式使用（变量必须初始化），编译器会根据表达式的类型推导变量类型，并忽略**顶层 const/volatile 和引用**。

```cpp
// 基础类型推导
auto a = 10;         // a的类型：int（推导自整数字面量10）
auto b = 3.14;       // b的类型：double（推导自浮点字面量3.14）
auto c = 'a';        // c的类型：char
auto d = true;       // d的类型：bool

// 忽略顶层const和引用
const int x = 5;
auto x1 = x;         // x1的类型：int（顶层const被忽略）
auto& x2 = x;        // x2的类型：const int&（加&后保留底层const）

int y = 10;
int& ry = y;
auto z = ry;         // z的类型：int（引用被忽略，推导为被引用类型）
auto& rz = ry;       // rz的类型：int&（加&后保留引用）
```

**关键规则**：

- `auto` 推导时会 “退化” 类型（类似数组名退化为指针、函数名退化为函数指针）：

    ```cpp
    int arr[5] = {1,2,3,4,5};
    auto arr1 = arr;    // arr1的类型：int*（数组名退化）
    auto& arr2 = arr;   // arr2的类型：int(&)[5]（加&后保留数组类型）
    ```

- 若初始化表达式是引用，`auto` 会推导为被引用的类型（而非引用本身），除非显式添加 `&`。
- C++11 中 `auto` 仅用于变量声明，C++14 扩展了其适用场景：
- **函数返回类型推导**：

    ```cpp
    // C++14起支持，编译器根据return语句推导返回类型
    auto add(int a, double b) {
        return a + b;  // 返回类型：double（int隐式转换为double）
    }
    ```

    注意：若函数有多个 return 语句，所有返回类型必须可推导为同一类型，否则编译错误。

- **lambda 表达式参数**：

    ```cpp
    // C++14起，lambda参数可使用auto（本质是泛型lambda）
    auto sum = [](auto a, auto b) { return a + b; };
    sum(1, 2);       // 推导为int+int，返回3
    sum(3.14, 2.7);  // 推导为double+double，返回5.84
    ```

- 3. `auto` 的限制
- 不能用于未初始化的变量：`auto x;`（编译错误，无初始化表达式无法推导）。
- 不能用于函数参数（非 lambda）：`void func(auto x) {}`（C++17 前不支持，C++20 起可用于模板函数简化）。
- 不能推导数组类型（除非用引用）：如上述 `arr1` 会退化为指针。

##### `decltype`：推导表达式的精确类型

`decltype`（“declare type” 的缩写）用于**推导表达式的类型**，但不执行表达式（仅分析类型）。与 `auto` 不同，它会完整保留表达式的类型信息（包括 const、volatile、引用等），且无需初始化变量。

###### 1. 基本用法

语法：`decltype(表达式) 变量名;`（变量可初始化也可不初始化）。

```cpp
int x = 10;
const int& rx = x;

// 推导变量类型
decltype(x) a;       // a的类型：int（x是int）
decltype(rx) b = x;  // b的类型：const int&（rx是const int&，保留引用和const）

// 推导表达式类型
decltype(x + 3) c;   // c的类型：int（x+3是int类型的表达式）
decltype(x * 1.5) d; // d的类型：double（int*double结果为double）
```

###### 2. 特殊规则：表达式的值类别影响推导结果

`decltype` 的推导结果与表达式的 “值类别”（lvalue/xvalue/prvalue）密切相关：

- 若表达式是**左值**（可取地址的对象），`decltype(表达式)` 推导为 “左值引用类型”。
- 若表达式是**纯右值**（临时对象），`decltype(表达式)` 推导为表达式的类型本身。

最典型的例子是 “变量名加括号” 的情况：

```cpp
int y = 20;

// 情况1：表达式是变量名（左值，但decltype对变量名特殊处理）
decltype(y) e;       // e的类型：int（变量名直接推导为其类型）

// 情况2：表达式是带括号的变量名（视为左值表达式）
decltype((y)) f = y; // f的类型：int&（(y)是左值表达式，推导为引用）
```

其他例子：

```cpp
int arr[5];
decltype(arr) g;     // g的类型：int[5]（数组名是左值，但变量名直接推导为数组类型）
decltype((arr)) h;   // h的类型：int(&)[5]（(arr)是左值表达式，推导为数组引用）
```

###### 3. 适用场景

`decltype` 的核心价值在于**保留类型的精确信息**，适合以下场景：

- **声明与表达式类型一致的变量**（无需初始化）：

    ```cpp
    vector<int> v;
    decltype(v.begin()) it;  // it的类型：vector<int>::iterator（与v.begin()返回类型一致）
    ```

- **模板中推导复杂类型**：

    ```cpp
    template <typename T, typename U>
    void func(T t, U u) {
        decltype(t + u) result;  // 推导t+u的类型，无需知道T和U具体是什么
        result = t + u;
    }
    ```

- **定义函数返回类型**（配合尾置返回类型，C++11 起）：

    ```cpp
    // 尾置返回类型：用decltype推导返回类型（依赖参数类型）
    template <typename T, typename U>
    auto add(T t, U u) -> decltype(t + u) {
        return t + u;
    }
    ```

##### 三、`decltype(auto)`：结合 `auto` 与 `decltype` 的优势

C++14 引入 `decltype(auto)`，它的行为是：**用 `auto` 的语法（根据初始化推导），但采用 `decltype` 的推导规则（保留精确类型）**。主要用于函数返回类型推导，解决 `auto` 在转发场景中丢失引用 /const 的问题。

###### 1. 基本用法

- **变量声明**：与 `decltype` 类似，但必须初始化（因为带 `auto`）。

    ```cpp
    int x = 10;
    int& rx = x;
    
    decltype(auto) a = x;    // a的类型：int（同decltype(x)）
    decltype(auto) b = rx;   // b的类型：int&（同decltype(rx)，保留引用）
    decltype(auto) c = (x);  // c的类型：int&（同decltype((x))，左值表达式推导为引用）
    ```

- **函数返回类型**：完美转发返回值的类型（保留引用、const 等）。
    例如，实现一个 “转发函数”，返回另一个函数的结果，且保留其类型：

    ```cpp
    int global = 100;
    
    int& get_ref() { return global; }  // 返回int&
    int get_val() { return global; }   // 返回int
    
    // 用decltype(auto)推导返回类型，保留原函数的返回类型
    decltype(auto) forward_ref() { return get_ref(); }  // 返回int&
    decltype(auto) forward_val() { return get_val(); }  // 返回int
    
    int main() {
        forward_ref() = 200;  // 合法：forward_ref返回int&，可赋值
        // forward_val() = 300;  // 错误：forward_val返回int（右值），不可赋值
        return 0;
    }
    ```

###### 2. 与 `auto` 的关键区别

`auto` 推导会忽略引用和顶层 const，而 `decltype(auto)` 会完整保留：

```cpp
int x = 5;
const int& rx = x;

auto a = rx;          // a的类型：int（忽略引用和const）
decltype(auto) b = rx;// b的类型：const int&（保留引用和const）
```

##### 四、`auto`、`decltype`、`decltype(auto)` 的对比总结

|特性|`auto`|`decltype(表达式)`|`decltype(auto)`|
|---|---|---|---|
|核心功能|推导变量类型（忽略顶层 const / 引用）|推导表达式类型（保留所有类型信息）|用 `auto` 语法，按 `decltype` 规则推导|
|是否需要初始化|必须（变量初始化）|可选（可仅声明类型）|必须（带 `auto` 特性）|
|引用 /const 处理|忽略顶层，需显式 `&` 保留|完整保留（依赖表达式值类别）|完整保留（同 `decltype`）|
|典型场景|简化变量声明（如迭代器、lambda）|模板类型推导、函数返回类型定义|转发函数返回类型（保留原类型）|
|示例|`auto it = v.begin();`|`decltype(v.begin()) it;`|`decltype(auto) func() { return x; }`|

##### 五、使用建议

1. 日常变量声明优先用 `auto`，简化代码（如 `auto result = compute();`）。
2. 需保留精确类型（如引用、const、数组）时用 `decltype`（如模板中定义关联类型）。
3. 函数返回值需要 “原样转发” 时用 `decltype(auto)`（如包装函数、转发器）。
4. 避免过度使用自动推导：复杂场景下显式类型更易读（如 `int` 比 `auto` 更清晰时）。

这些工具是 C++ 类型系统的重要补充，尤其在泛型编程和现代 C++ 开发中不可或缺，合理使用可大幅提升代码的简洁性和可维护性。

#### 常量表达式

C++ 中的**常量表达式（Constant Expression）** 是指在**编译期就能确定值**的表达式，它允许程序在编译阶段完成计算、内存分配等操作，从而提升运行时效率并增强类型安全性。C++11 引入了 `constexpr` 关键字来显式声明常量表达式，后续标准（C++14/C++17/C++20）不断扩展其能力，使其成为现代 C++ 的核心特性之一。

##### 一、常量表达式的核心价值

1. **编译期计算**：将部分运行时的计算提前到编译期完成，减少程序运行时的开销。
2. **类型安全**：编译期即可可以验证表达式的有效性，避免运行时错误。
3. **优化机会**：编译器可基于常量表达式进行更深度的优化（如常量折叠、死代码消除）。
4. **支持编译期内存分配**：如 `constexpr` 数组可在编译期确定大小并分配内存。

##### 二、`constexpr` 的基本用法

`constexpr` 可用于修饰**变量**、**函数**和**构造函数**，表明它们的值或返回结果可以在编译期确定。

###### 1. `constexpr` 变量

用 `constexpr` 声明的变量必须在编译期初始化，且其值必须是常量表达式。

```cpp
// 基础类型常量表达式
constexpr int max_size = 1024;  // 编译期确定值为1024
constexpr double pi = 3.1415926;

// 表达式初始化（需为编译期可计算）
constexpr int a = 10 + 20;       // 合法：10+20是编译期常量
constexpr int b = max_size / 2;  // 合法：基于其他constexpr变量

// 错误示例：运行时才能确定的值不能初始化constexpr变量
int x = 5;
// constexpr int c = x;  // 编译错误：x是运行时变量
```

**与 `const` 的区别**：

- `const` 变量仅表示 “只读”，其值可能在运行时初始化（如 `const int d = rand();`，值在运行时确定）。
- `constexpr` 变量必须在编译期确定值，是 “编译期常量” 的强化版本。

###### 2. `constexpr` 函数

`constexpr` 函数是**可以在编译期被调用并计算结果**的函数。其返回值在编译期调用时是常量表达式，在运行时调用时与普通函数一致。

###### C++11 中的限制

- 函数体只能有一条 `return` 语句（不允许复杂逻辑）。
- 参数和返回值必须是 “字面类型”（可在编译期构造的类型，如基础类型、数组、某些结构体等）。

```cpp
// C++11：简单constexpr函数
constexpr int add(int a, int b) {
    return a + b;  // 仅一条return语句
}

constexpr int sum = add(10, 20);  // 编译期调用，sum=30（常量表达式）
int x = 5, y = 6;
int runtime_sum = add(x, y);      // 运行时调用，与普通函数一致
```

###### C++14 及以后的扩展

- 允许函数体包含多条语句（条件分支、循环等）。
- 支持局部变量（但必须是 `constexpr` 或编译期可初始化）。

```cpp
// C++14：复杂constexpr函数（支持循环和分支）
constexpr int factorial(int n) {
    if (n <= 1) return 1;
    int result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

constexpr int f5 = factorial(5);  // 编译期计算：f5=120
```

###### 3. `constexpr` 构造函数与常量对象

对于自定义类型，`constexpr` 构造函数允许在编译期创建对象（即 “常量对象”）。

```cpp
class Point {
private:
    int x, y;
public:
    // constexpr构造函数（C++11起）
    constexpr Point(int x_, int y_) : x(x_), y(y_) {}

    // constexpr成员函数（返回成员变量）
    constexpr int getX() const { return x; }
    constexpr int getY() const { return y; }
};

// 编译期创建Point对象（常量对象）
constexpr Point origin(0, 0);
constexpr int x = origin.getX();  // 编译期获取x值（0）
```

C++20 进一步允许 `constexpr` 构造函数中包含更复杂的逻辑（如循环、条件判断）。

##### 三、常量表达式的应用场景

1. **数组大小定义**：
    数组大小必须是编译期常量，`constexpr` 可动态计算大小：

    ```cpp
    constexpr int n = 5;
    int arr[factorial(n)];  // 数组大小为120（编译期确定）
    ```

2. **模板参数**：
    模板参数必须是编译期常量，`constexpr` 函数的返回值可直接作为参数：

    ```cpp
    template <int N>
    struct Buffer { char data[N]; };
    
    Buffer<add(3, 7)> buf;  // 等价于Buffer<10>，编译期确定N=10
    ```

3. **`std::array` 初始化**：
    `std::array` 的大小需编译期确定，结合 `constexpr` 可实现编译期初始化：

    ```cpp
    #include <array>
    constexpr int size = 4;
    std::array<int, size> arr = {1, 2, 3, 4};  // 编译期确定大小
    ```

4. **编译期算法**：
    复杂逻辑（如排序、查找）可通过 `constexpr` 函数在编译期完成：

    ```cpp
    // 编译期计算斐波那契数列
    constexpr int fib(int n) {
        return (n <= 1) ? n : fib(n-1) + fib(n-2);
    }
    constexpr int fib10 = fib(10);  // 编译期计算：55
    ```

##### 四、常量表达式的限制

1. **字面类型要求**：参与常量表达式的类型必须是 “字面类型”（Literal Type），即：

    - 基础类型（`int`、`double` 等）、引用、指针。
    - 不含虚函数或虚基类的类，且其所有成员和基类都是字面类型。
2. **函数副作用**：`constexpr` 函数不能有副作用（如修改全局变量、I/O 操作），因为编译期执行无法产生运行时副作用。
3. **动态内存**：C++11/14 中 `constexpr` 函数不允许使用动态内存（`new`/`delete`），C++20 起允许但限制严格（编译期分配的内存必须在编译期释放）。

##### 五、总结

常量表达式（`constexpr`）是 C++ 编译期编程的核心工具，它通过以下方式增强程序：

- **性能**：将计算从运行时提前到编译期，减少运行开销。
- **安全性**：编译期验证表达式有效性，避免运行时错误。
- **灵活性**：支持编译期动态计算（如数组大小、模板参数），突破了传统常量的限制。

随着 C++ 标准的演进，`constexpr` 的能力不断扩展（从简单函数到复杂逻辑），已成为现代 C++ 中编写高效、安全代码的重要手段。

### 函数
#### **函数重载（Function Overloading）**

   - **函数重载**允许在同一作用域内定义多个同名函数，根据**参数列表（参数类型、数量或顺序）** 区分。
   - 返回类型不同不能作为重载依据（编译器无法仅通过返回类型区分调用）。

#### **默认参数（Default Arguments）**

   - C++ 允许为函数参数指定默认值，调用时若省略该参数，编译器会自动填入默认值。
   - 默认参数必须从右向左定义（若某个参数有默认值，其右侧所有参数都必须有默认值）。

#### 引用参数（Reference Parameters）

C++ 引入**引用（&）** 作为函数参数，允许函数直接操作实参本身（而非副本），替代 C 语言的指针传递，更安全且语法简洁。

- 避免大对象的拷贝（提升性能）。
- 直接修改实实参（无需指针的 `*` 和 `&` 操作）。
- 常量引用：不修改实参，避免拷贝

#### Lambda 表达式（匿名函数）（C++11）

C++11 引入的**lambda 表达式**（匿名函数）是一种在代码中**就地定义匿名函数**的语法，主要用于简化 “临时性功能代码” 的编写，尤其适合作为算法的回调函数或短小的函数对象。它的核心价值是**减少代码冗余**，让逻辑更紧凑。

##### 一、lambda 表达式的基本语法

lambda 表达式的完整语法结构如下：

```cpp
[capture-list] (parameter-list) mutable noexcept -> return-type {
    // 函数体
}
```

各部分含义：

- **`[capture-list]`（捕获列表）**：定义 lambda 外部的变量如何被内部访问（核心特性，后文详解）。
- **`(parameter-list)`（参数列表）**：与普通函数的参数列表一致（可省略，若为空）。
- **`mutable`（可选）**：允许 lambda 内部修改按值捕获的变量（默认不可修改）。
- **`noexcept`（可选）**：声明 lambda 不会抛出异常。
- **`-> return-type`（返回类型）**：指定返回类型（可省略，由编译器自动推导）。
- **`{函数体}`**：lambda 的执行逻辑。

##### 二、最简单的 Lambda 表达式

最简化的 lambda 可以省略参数列表、返回类型，仅保留捕获列表（可为空）和函数体：

```cpp
#include <iostream>

int main() {
    // 无参数、无捕获、无返回值的lambda
    auto hello = []() { 
        std::cout << "Hello, Lambda!" << std::endl; 
    };
    
    hello();  // 调用lambda，输出：Hello, Lambda!
    return 0;
}
```

- `auto` 用于存储 lambda 表达式（lambda 的类型是编译器生成的匿名类型，无法显式写出）。
- 可直接调用，无需命名：`[]() { std::cout << "直接调用"; }();`

##### 三、核心部分：捕获列表（`[capture-list]`）

捕获列表控制 lambda 如何访问**定义它的作用域中的变量**，是 lambda 与普通函数的关键区别。常见捕获方式：

|捕获方式|含义|
|---|---|
|`[]`|不捕获任何外部变量|
|`[var]`|按值捕获变量 `var`（副本，不可修改）|
|`[&var]`|按引用捕获变量 `var`（可修改原变量）|
|`[=]`|按值捕获所有使用到的外部变量|
|`[&]`|按引用捕获所有使用到的外部变量|
|`[=, &var]`|默认按值捕获，仅 `var` 按引用捕获|
|`[&, var]`|默认按引用捕获，仅 `var` 按值捕获|
|`[this]`|在类中捕获当前对象（`*this`）|

- 示例：不同捕获方式的效果

```cpp
#include <iostream>

int main() {
    int a = 10, b = 20;
    
    // 1. 按值捕获a，按引用捕获b
    auto func1 = [a, &b]() {
        // a = 100;  // 错误：按值捕获的变量不可修改（除非加mutable）
        b = 200;    // 正确：按引用捕获可修改原变量
        std::cout << "a=" << a << ", b=" << b << std::endl;  // a=10, b=200
    };
    func1();
    std::cout << "外部b=" << b << std::endl;  // 外部b被修改为200
    
    // 2. 按值捕获所有变量（a和b），并允许修改副本（mutable）
    auto func2 = [=]() mutable {
        a = 100;  // 允许修改副本（不影响外部a）
        b = 200;
        std::cout << "内部a=" << a << ", 内部b=" << b << std::endl;  // 100, 200
    };
    func2();
    std::cout << "外部a=" << a << ", 外部b=" << b << std::endl;  // 10, 200（无变化）
    
    // 3. 按引用捕获所有变量
    auto func3 = [&]() {
        a = 30;
        b = 40;
    };
    func3();
    std::cout << "外部a=" << a << ", 外部b=" << b << std::endl;  // 30, 40（被修改）
    
    return 0;
}
```

##### 四、参数列表与返回类型

lambda 的参数列表和返回类型与普通函数类似，但更灵活：

1. 参数列表

- 支持普通参数、默认参数（C++14 起）、可变参数等：

    ```cpp
    // 带参数和默认参数的lambda
    auto add = [](int x, int y = 5) { 
        return x + y; 
    };
    std::cout << add(3);    // 3+5=8
    std::cout << add(3, 4); // 3+4=7
    ```

- C++14 起支持 `auto` 作为参数类型（泛型 lambda）：

    ```cpp
    // 泛型lambda（参数类型自动推导）
    auto sum = [](auto a, auto b) { 
        return a + b; 
    };
    sum(1, 2);       // 3（int+int）
    sum(3.14, 2.7);  // 5.84（double+double）
    ```

1. 返回类型

- 若函数体仅有一条 `return` 语句，返回类型可省略（编译器自动推导）：

    ```cpp
    auto multiply = [](int x, int y) { 
        return x * y;  // 自动推导返回类型为int
    };
    ```

- 若函数体有复杂逻辑（如分支返回不同类型），需显式指定返回类型：

    ```cpp
    auto divide = [](double x, double y) -> double {  // 显式指定返回double
        if (y == 0) return 0;
        return x / y;
    };
    ```

##### 五、lambda 在 STL 中的典型应用

lambda 最常用的场景是作为 STL 算法的回调函数（如排序、遍历、条件判断），替代繁琐的函数对象或全局函数。

- 配合 STL 算法（如 `sort`、`for_each`）实现简洁的回调逻辑。

```cpp
#include <algorithm>
#include <vector>

int main() {
    vector<int> nums = {3, 1, 4, 1, 5};

    // 用lambda作为排序规则（降序）
    sort(nums.begin(), nums.end(), 
         [](int a, int b) { return a > b; });  // 匿名函数：比较a和b

    // 用lambda遍历输出
    for_each(nums.begin(), nums.end(), 
             [](int x) { cout << x << " "; });  // 输出：5 4 3 1 1
    return 0;
}
```

##### 六、lambda 的类型与存储

- lambda 的类型是**编译器生成的匿名非 union 类类型**（称为 “闭包类型”），因此必须用 `auto` 或模板参数接收。
- 可将 lambda 存储在 `std::function`（C++11 起）中，实现更灵活的回调管理：

    ```cpp
    #include <functional>
    
    int main() {
        // 存储lambda到std::function
        std::function<int(int, int)> func = [](int a, int b) {
            return a + b;
        };
        func(2, 3);  // 5
        return 0;
    }
    ```

- 无捕获的 lambda 可隐式转换为函数指针，有捕获的则不能：

    ```cpp
    // 无捕获lambda → 函数指针
    auto lambda = [](int x) { return x * 2; };
    int (*func_ptr)(int) = lambda;  // 合法
    func_ptr(5);  // 10
    ```

##### 七、注意事项

1. **生命周期问题**：按引用捕获的变量必须确保在 lambda 调用时仍有效（避免悬垂引用）：

    ```cpp
    auto get_lambda() {
        int x = 10;
        return [&x]() { return x; };  // 危险：x在函数返回后销毁，引用失效
    }
    ```

2. **`mutable` 的使用**：仅允许修改按值捕获的副本，不影响外部变量：

    ```cpp
    int x = 5;
    auto func = [x]() mutable { x++; return x; };
    func();  // 返回6（副本被修改）
    std::cout << x;  // 仍为5（外部x不变）
    ```

3. **性能**：lambda 通常与手写函数效率相同（编译器优化），但复杂捕获可能引入微小开销。

#### Const 成员函数

在类的成员函数后加 `const`，表示该函数**不会修改类的成员变量**，是对成员函数的 “只读” 约束。

- 增强代码可读性（明确函数不修改对象状态）。
- 确保 `const` 对象只能调用 `const` 成员函数（安全性）。

### 其他编程范式
#### 命名空间（`namespace`）

   - **C++ 特性**：避免全局命名冲突。
   - **示例**：

 ```cpp
 namespace MyLib {
	 int func() { return 42; }
 }
 int main() {
	 std::cout << MyLib::func(); // 使用命名空间
 }
 ```

#### 右值引用与移动表达式（C++11）

C++11 引入**右值引用（`&&`）** 和移动语义，通过 `std::move` 将左值转为右值，允许表达式中的资源（如动态内存）被 “移动” 而非 “拷贝”，大幅优化性能。

避免表达式中临时对象的冗余拷贝（如 `a = b + c` 中，`b + c` 的结果是临时对象，可被移动到 `a`）。

- 示例：移动表达式优化

```cpp
#include <vector>
#include <utility>  // std::move

int main() {
    std::vector<int> v1(10000, 1);  // 大容器，含10000个元素

    // 差：拷贝构造，复制10000个元素（开销大）
    std::vector<int> v2 = v1;

    // 优：移动构造，仅转移资源所有权（无元素复制）
    std::vector<int> v3 = std::move(v1);  // v1变为空，v3拥有原资源
    return 0;
}
```

- 关键表达式：
- `std::move(x)`：将左值 `x` 转为右值引用，触发移动操作；
- 移动赋值：`a = std::move(b)`，资源从 `b` 移动到 `a`；
- 临时对象表达式（如 `a + b`）天然是右值，自动触发移动（若类型支持）。

#### 初始化列表表达式（Initializer List）（C++11）

C++11 引入**初始化列表（`{}`）**，作为一种通用的表达式语法，统一了不同场景的初始化方式（如变量、容器、函数参数）。

- 用简洁的 `{}` 语法表示 “值的集合”，使表达式更直观，支持多种类型的初始化。
- 示例：初始化列表表达式的应用

```cpp
#include <vector>
#include <map>

// 接受初始化列表作为参数的函数
void print(std::initializer_list<int> list) {
    for (int x : list) {
        std::cout << x << " ";
    }
}

int main() {
    // 初始化容器
    std::vector<int> vec = {1, 2, 3};  // 列表初始化
    std::map<int, std::string> map = {{1, "one"}, {2, "two"}};

    // 作为函数参数
    print({4, 5, 6});  // 输出：4 5 6

    // 直接作为表达式使用
    auto list = {7, 8, 9};  // list类型为std::initializer_list<int>
    return 0;
}
```

#### `nullptr` 表达式（C++11）

C++11 引入 `nullptr` 作为空指针常量，替代 C 中的 `NULL`（`NULL` 本质是 `0`），避免整数与指针的混淆：

```cpp
void func(int x) {}
void func(void* p) {}

int main() {
func(NULL);    // 调用func(int)（C中歧义）
func(nullptr); // 调用func(void*)（明确指向指针重载）
}
```

#### 范围 for 表达式（C++11）

C++11 简化了容器遍历的表达式，直接迭代元素而非通过索引或迭代器：

```cpp
std::vector<int> nums = {1, 2, 3};
for (int x : nums) {  // 范围for，等价于遍历nums的每个元素
std::cout << x;
}
```

#### 类型转换表达式增强

C++ 提供更安全的类型转换表达式（替代 C 的强制转换）：

- `static_cast`：编译期类型转换（如 `int`→`double`）；
- `dynamic_cast`：运行时多态类型转换（带类型检查）；
- `reinterpret_cast`：底层二进制转换（如指针→整数），比 C 的强制转换更明确。

#### 结构化绑定（C++17）

 ```
std::pair<int, std::string> p = {42, "hello"};
auto [id, name] = p; // 解包
```

#### 折叠表达式（C++17）

```
template<typename… Args>
void print(Args… args) {
(std::cout << … << args) << '\n'; // 折叠表达式
}
```

#### 泛型编程

泛型编程（Generic Programming）是一种**独立于具体数据类型的编程范式**，核心思想是：**编写与类型无关的通用代码，通过 “参数化类型” 实现代码复用**，同时保证类型安全和高效性。C++ 是泛型编程的典型实现者，其核心工具是**模板（Template）**，标准模板库（STL）是泛型编程的经典应用。C++ 通过**函数模板**和**类模板**实现泛型，模板是 “代码生成器”—— 编译器根据传入的具体类型，自动生成对应版本的函数或类。

C++ 标准模板库（STL）核心组件（容器、算法、迭代器）全基于模板实现：

- **容器（Containers）**：如 `vector<T>`、`list<T>`、`map<K, V>`，通过类模板支持任意元素类型。
- **算法（Algorithms）**：如 `sort`、`find`，通过函数模板可操作任意容器（依赖迭代器）。
- **迭代器（Iterators）**：作为容器与算法的 “桥梁”，模拟指针行为，使算法独立于具体容器类型。

---

## 二、面向对象编程（OOP）

> [!important] 封装继承多态
> C++ 面向对象的核心是通过**封装**实现数据安全，**继承**实现代码复用，**多态**实现接口灵活。

- **类与对象**：
    - 构造函数/析构函数、拷贝构造函数、赋值运算符。
    - `this` 指针、常量成员函数、友元函数。
    - 静态成员（`static`）、单例模式。
- **继承与多态**：
    - 公有/私有继承、虚函数、虚析构函数。
    - 抽象类、纯虚函数。

### 1. **类与对象（Class & Object）**

   - **C++ 核心特性**：封装数据和行为。
	- `public`：外部可直接访问（暴露的接口）。
	- `private`：仅类内部可访问（隐藏的实现细节）。
	- `protected`：类内部和子类可访问（用于继承场景）。

### 2. **构造函数与析构函数**

   - **C++ 特性**：
     - 构造函数：自动初始化对象。
     - 析构函数：自动释放资源。
   - **示例**：

     ```cpp
     class File {
     public:
         File(const char* path) { /* 打开文件 */ }
         ~File() { /* 关闭文件 */ }
     };
     ```

- **委派构造函数（C++11）**：允许一个构造函数调用同一类的其他构造函数，减少代码重复。

### 3. 继承

C++ 通过 `class 子类 : 继承方式 父类` 语法实现继承，继承方式（`public`/`private`/`protected`）控制父类成员在子类中的访问权限（默认 `private`）。

```cpp
// 父类：通用动物
class Animal {
protected:  // 受保护成员：子类可访问
	string name;
public:
	Animal(string n) : name(n) {}
	void eat() {  // 通用行为
		cout << name << " is eating." << endl;
	}
	virtual void makeSound() {  // 虚函数：允许子类重写
		cout << name << " makes a sound." << endl;
	}
};

// 子类：狗（继承自动物）
class Dog : public Animal {
public:
	// 继承父类构造函数
	Dog(string n) : Animal(n) {}
	
	// 重写父类方法（实现狗的特有行为）
	void makeSound() override {
		cout << name << " barks: Woof!" << endl;
	}
	
	// 新增子类特有方法
	void fetch() {
		cout << name << " is fetching the ball." << endl;
	}
};
```

### 4. 多态（Polymorphism）：接口复用与动态行为

多态是指**同一接口（如父类指针 / 引用）可以表现出不同的行为，具体行为由实际对象类型决定**，分为 “编译期多态”（函数重载）和 “运行期多态”（虚函数）。而**虚函数（Virtual function）** 和**抽象类（Abstract Class）** 是实现多态的关键技术手段，三者紧密关联：虚函数是多态的基础机制，抽象类则是多态的一种高级应用形式（定义接口规范）。

- **接口统一**：用统一的父类接口操作不同子类对象，无需关心具体类型。
- **动态绑定**：运行时根据对象实际类型执行对应方法，提高代码灵活性。

#### 一、多态：同一接口，不同实现

多态的核心思想是：**用统一的父类接口（如指针或引用）操作不同的子类对象时，程序会根据对象的实际类型执行对应的方法**，而不是接口的静态类型。

例如，“动物” 是一个父类，“狗” 和 “猫” 是子类：

```cpp
// 父类
class Animal {
public:
    // 虚函数：关键！为多态提供支持
    virtual void makeSound() { 
        cout << "动物发出声音" << endl; 
    }
};

// 子类1
class Dog : public Animal {
public:
    // 重写父类的虚函数
    void makeSound() override { 
        cout << "狗汪汪叫" << endl; 
    }
};

// 子类2
class Cat : public Animal {
public:
    // 重写父类的虚函数
    void makeSound() override { 
        cout << "猫喵喵叫" << endl; 
    }
};
```

多态的体现：用父类指针指向不同引用指向子类对象，调用方法时会执行子类的实现：

```cpp
int main() {
    Animal* animal1 = new Dog();  // 父类指针指向Dog对象
    Animal* animal2 = new Cat();  // 父类指针指向Cat对象
    
    animal1->makeSound();  // 输出：狗汪汪叫（实际执行Dog的方法）
    animal2->makeSound();  // 输出：猫喵喵叫（实际执行Cat的方法）
    
    delete animal1;
    delete animal2;
    return 0;
}
```

**本质**：多态通过 “动态绑定” 实现 —— 程序在运行时才确定要调用的具体方法（而非编译期根据指针类型确定）。

#### 二、虚函数：多态的 “开关”

虚函数是**被 `virtual` 关键字修饰的成员函数**，它是 C++ 实现多态的基础。其核心作用是：**允许子类重写（override）父类的方法，并在运行时根据对象实际类型调用对应的重写版本**。

1. **父类声明**：在父类中用 `virtual` 声明函数（如 `virtual void makeSound()`）。
2. **子类重写**：子类用 `override` 关键字显式重写（C++11 起推荐，确保重写正确），函数签名（返回类型、参数列表）必须与父类完全一致。
3. **动态绑定**：只有通过**父类指针或引用**调用虚函数时，才会触发多态（动态绑定）；直接用子类对象调用时，行为与普通函数相同。

如果没有 `virtual`，函数调用会在编译期根据指针类型（而非对象类型）确定，无法实现多态：

```cpp
class Animal {
public:
    void makeSound() {  // 非虚函数
        cout << "动物发出声音" << endl; 
    }
};

// …（Dog和Cat类定义同上）

int main() {
    Animal* animal = new Dog();
    animal->makeSound();  // 输出：动物发出声音（编译期绑定，调用父类方法）
    return 0;
}
```

#### 三、抽象类：多态的 “接口规范”

抽象类是**包含纯虚函数（Pure Virtual Function）的类**，它不能实例化对象，只能作为父类被继承。其核心作用是：**定义 “必须实现的接口”，强制子类提供具体实现，从而规范多态行为**。

- 纯虚函数的定义：表示该函数没有默认实现，必须由子类重写：

```cpp
class Animal {  // 抽象类（因为包含纯虚函数）
public:
    // 纯虚函数：只有声明，没有实现
    virtual void makeSound() = 0; 
};
```

- 抽象类的特性
	1. **不能实例化**：`Animal animal;` 或 `new Animal();` 都会编译错误（抽象类是 “接口”，不是 “具体对象”）。
	2. **强制重写**：子类必须重写所有纯虚函数，否则子类也会成为抽象类（无法实例化）。
	3. **定义接口**：抽象类本质是 “接口规范”，例如 `Animal` 规定 “所有动物必须能发出声音”，但不关心具体怎么叫（由子类实现）。

```cpp
class Dog : public Animal {
public:
    void makeSound() override {  // 必须重写，否则Dog也是抽象类
        cout << "狗汪汪叫" << endl;
    }
};
```

### 运算符重载（Operator Overloading）

   - **C++ 特性**：自定义类的运算符行为。
   - **示例**：

 ```cpp
 class Vector {
 public:
	 Vector operator+(const Vector& other) {
		 return Vector(x + other.x, y + other.y);
	 }
 };
 ```

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
