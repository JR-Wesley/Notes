---
tags:
  - CS
category: Summary
---

本系列旨在梳理现代 C++ 的关键特性，构建从 C 基础开始的系统 C++ 高效编程范式。

> [!note] 参考
> - <a href="https://cntransgroup.github.io/EffectiveModernCppChinese/Introduction.html">高效现代 C++ 中文 </a>：现代 C++ 高效编程进阶书，能帮助掌握高效编程的必要范式，需要对 C++ 基础有了解。
> 	- 【【Effective Modern C++】啃书第一章：类型推导】 https://www.bilibili.com/video/BV1Gg4y1p71w/?share_source=copy_web&vd_source=fd37be71d17f708cc53476cbd29e590f 一个博主对该书的详细个人解读
> - <a href="https://learn.microsoft.com/zh-cn/cpp/cpp/welcome-back-to-cpp-modern-cpp?view=msvc-170">微软现代 C++ 中文 </a>
> - <a href="https://en.cppreference.com/w/cpp/23.html"> Cpp reference 23 </a> C++ 标准，更新至 23
> - C++ Primer 5ed: 作为一本 C++11 的字典式的教科书，可以反复查阅，可能对一些底层原理没有深入的讲解。
> - Beautifile C++: https://ptgmedia.pearsoncmg.com/images/9780137647842/samplepages/9780137647842_Sample.pdf
> - C++ Core Guidelines: https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#S-philosophy


https://chengxumiaodaren.com/

我总结的现代 C++ 一些关键的点，它们是现代 C++ 高效和安全的基础：

- 强类型，类型转换，const
- 引用、移动语义、值语义
- RAII，指针、对象对资源的管理
- STL，容器、算法库，lambda 表达式
- 封装继承多态的相关基础
- 一些语法增强。。。
- 并发编程（不限于 C++）

# C++ Primer 导读

- 第 1 章：讲解基本的程序执行和控制流。
	- **建议**：在实践中逐渐熟悉程序的编译、调试。
- 第 2 章：介绍变量和基本类型。
	- **建议**：使用 `const constexpr auto using nullptr` 等，同时需要避免类型窄化转换，`const`、引用有很多使用方法需要额外掌握。
- 第 3 章：介绍字符串、`vector`、数组。从这里开始，就要注意现代 C++ 的理念是**用资源管理代替裸指针，用高级抽象代替低级操作**，应该避免一些 C 风格的实现，除非与 C 接口交互。其实也应该避免 `new[]`/`delete[]`、`malloc`/`free`。
	- **建议**：字符串用 `std::string` 和 `std::string_view`。
	- **建议**：STL 的 `vector` 用于动态数组，`array` 用于静态数组用于替代 C 的数组。原始内存管理用 `std::unique_ptr<T[]>` 或 `std::shared_ptr<T[]>`。
	- **建议**：迭代器仍然是一个基础概念，它是 STL 容器和算法的基础，但是它不再是日常编程的首选，现代 C++ 推荐使用 `for range`、函数式风格的算法 +`lambda`、C++20 引入的 `Ranges`。
- 第 4、5 章：回顾表达式和语句，C++ 保留了 C 大部分语法特性，但是在语义和表达力上有了很多增强，有一些新引入的特性可以在后面具体用到时学习。
	- **建议**：优先使用 `{}` 初始化，避免 `() =` 混淆。

# 友元

在 C++ 中，友元（Friend）机制是一种特殊的访问权限控制方式，它允许特定的函数或类访问另一个类中的私有（private）和保护（protected）成员，打破了类的封装性限制。

## 友元的主要形式

1. **`p`**：一个非成员函数被声明为某个类的友元后，可以访问该类的私有和保护成员。

```cpp
class MyClass {
private:
	int privateData;
public:
	MyClass(int data) : privateData(data) {}
	// 声明友元函数
	friend void printData(MyClass obj);
};

// 友元函数定义
void printData(MyClass obj) {
	// 可以直接访问私有成员
	cout << "Private data: " << obj.privateData << endl;
}
```

2. **2. 资源管理**：一个类被声明为另一个类的友元后，该类的所有成员函数都能访问对方类的私有和保护成员。

    ```cpp
    class A {
    private:
        int value;
    public:
        A(int v) : value(v) {}
        // 声明B为A的友元类
        friend class B;
    };
    
    class B {
    public:
        void showA(A a) {
            // B类可以访问A的私有成员
            cout << "A's value: " << a.value << endl;
        }
    };


```

2. **类的成员函数作为友元**：一个类的特定成员函数被声明为另一个类的友元。

    ```cpp
    class B; // 前向声明
    
    class A {
    public:
        void showB(B b);
    };
    
    class B {
    private:
        int secret;
    public:
        B(int s) : secret(s) {}
        // 声明A类的showB函数为友元
        friend void A::showB(B b);
    };
    
    // 实现友元成员函数
    void A::showB(B b) {
        cout << "B's secret: " << b.secret << endl;
    }
    ```

### 友元的特点

- **单向性**：若 A 是 B 的友元，B 不一定是 A 的友元，除非显式声明
- **不可传递**：若 A 是 B 的友元，B 是 C 的友元，A 不会自动成为 C 的友元
- **不可继承**：友元关系不能被派生类继承

### 友元的使用场景

- 操作符重载（如`<<`、`>>`）
- 实现某些设计模式（如工厂模式）
- 需要跨类共享数据但又不希望公开接口的场景

### 注意事项

- 友元机制会破坏类的封装性，应谨慎使用
- 过多使用友元会降低代码的可维护性和安全性
- 通常建议优先使用公共接口，仅在必要时才使用友元


友元机制提供了一种灵活的访问控制方式，但也带来了封装性的削弱，实际开发中应在灵活性和封装性之间寻找平衡。


# 输出格式

```cpp
#include <iostream> 
#include <bitset> 
//输出二进制的头文件 
using namespace std; 
int main(){ 
	int a = 2149580819; 
	cout << "八进制： " << oct << a << endl; 
	cout << "十进制： " << dec << a << endl; 
	cout << "十六进制： " << hex << a << endl; 
	cout << "二进制： " << bitset<sizeof(a)*8>(a) << endl; 
	return 0; 
}
```

# C++ 17

## Inline

从 C++17 开始，在编写 C++ 代码时就可以在头文件中定义 inline 变量。且在编译时也不会报错，如果同时有多份代码文件引用了该头文件，编译器也不会报错。不管怎么说，这是一种进步。实际编写时可以如下代码所示：

```javascript
class MyClass {
inline static std::string strValue{"OK"}; // OK（自C++17起 ）
};
inline MyClass myGlobalObj; // 即 使 被 多 个CPP文 件 包 含 也OK
```

复制

需要注意的是，编写时在同一个代码文件中要保证定义对象的唯一性。

**智能指针**

按照一次定义原则，一个变量或者实体只能出现一个编译单元内，除非这个变量或者实体使用了 inline 进行修饰。如下面的代码。如果在一个类中定义了一个静态成员变量，然后在类的外部进行初始化，本身符合一次定义原则。但是如果在多个 CPP 文件同时包含了该头文件，在链接时编译器会报错。

```javascript
class MyClass {
static std::string msg;
...
};
// 如 果 被 多 个CPP文 件 包 含 会 导 致 链 接ERROR
std::string MyClass::msg{"OK"};
```

复制

那么如何解决这个问题呢？可能会有些同学说，将类的定义包含在预处理里面。代码如下：

```javascript
#ifndef MYCLASS_H
#define MYCLASS_H
class MyClass {
static std::string msg;
...
};
// 如 果 被 多 个CPP文 件 包 含 会 导 致 链 接ERROR
std::string MyClass::msg{"OK"}; 
#endif
```

复制

这样类定义包含在多个代码文件的时候的就不会有链接错误了吧？实际上，错误依旧存在。那么在 C++17 以前，有哪些解决方法呢?

**文件/网络资源管理**

实际上，根据不同的使用场景，可以有不同的方案。

- 可以定义一个返回 static 的局部变量的内联函数。

```javascript
inline std::string& getMsg() {
    static std::string msg{"OK"};
    return msg;
}
```

复制

- 可以定义一个返回该值的 static 的成员函数

```javascript
class MyClass {
    static std::string& getMsg() {
      static std::string msg{"OK"};
      return msg;
    }
};
```

复制

- 可以为静态数据成员定义一个模板类，然后继承它

```javascript
template<typename = void>
class MyClassStatics
{
    static std::string msg;
};
template<typename T>
std::string MyClassStatics<T>::msg{"OK"};
class MyClass : public MyClassStatics<>
{
};
```

复制

同样，如果有学习过 C++14 的同学还会想到使用变量模板，如下所示：

```javascript
template<typename T = std::string>
T myGlobalMsg{"OK"}
```

复制

从上面可以看到，及时没有 C++17 在实际编程时也能解决遇到的问题。但是当跳出来再看这些方法的时候，就会注意到在实际使用时会存在一些问题。如上面的方法会导致签名重载、可读性变差、全局变量初始化延迟等一些问题。变量初始化延迟也会和我们固有的认知产生矛盾。因为我们定义一个变量的时候默认就已经被立即初始化了。

**3. 函数重载**

C++17 中内联变量的使用可以帮助我们解决实际编程中的问题而又不失优雅。使用 inline 后，即使定义的全局对象被多个文件引用也只会有一个全局对象。如下面的代码，就不会出现之前的链接问题。

```javascript
class MyClass {
inline static std::string msg{"OK"};
...
};
inline MyClass myGlobalObj;
```

复制

除此之外，需要还需要注意的是，在一个类的内部定义了一个自身类型的静态变量时需要在类的外部进行重新定义。如：

```javascript
struct MyData {
int value;
MyData(int i) : value{i} {
}
static MyData max;
...
};
inline MyData MyData::max{0};
```

复制

**6. 常见误区与注意事项**

从 C++17 开始，如果在编程时继续使用 constexpr static 修饰变量，实际上编译器就会默认是内联变量。如下面定义的代码:

```javascript
struct MY_DATA {
  static constexpr int n = 5; 
}
```

复制

这段代码实际上和下面的代码是等效的。

```javascript
struct MY_DATA {
  inline static constexpr int n = 5; 
}
```

复制

**右值引用变量本身是左值**

在支持 C++17 的编译器编程时使用 thread_local 可以给每一个线程定义一个属于自己的内联变量。如下面的代码：

```javascript
struct THREAD_NODE{
  inline static thread_local std::string strName;
};
inline thread_local std::vector<std::string> vCache; 
```

复制

如上，通过 thread_local 修饰的内联变量就给每一个线程对象创建的属于自己的内联变量。

下面，通过一段代码来对此功能进行说明，先介绍下功能，代码主要定义了一个类，类中包含三个成员变量，分别是内联变量、使用了 thread_local 修饰了的内联变量以及一个本地的成员变量；除此之外定义了一个自身类型的用 thread_local 修饰的内联变量，以保证不同的线程拥有自己的内联变量。main 函数分别对内联变量进行打印和输出，具体代码如下：

```javascript
#include <string>
#include <iostream>
#include <thread>
struct MyData {
    inline static std::string gName = "global"; 
    inline static thread_local std::string tName = "tls"; 
    std::string lName = "local";
    void print(const std::string& msg) const {
        std::cout << msg << '\n';
        std::cout << "- gName: " << gName << '\n';
        std::cout << "- tName: " << tName << '\n';
        std::cout << "- lName: " << lName << '\n';
    }
};
inline thread_local MyData myThreadData; 
void foo()
{
    myThreadData.print("foo() begin:");
    myThreadData.gName = "thread2 name";
    myThreadData.tName = "thread2 name";
    myThreadData.lName = "thread2 name";
    myThreadData.print("foo() end:");
}

int main()
{
    myThreadData.print("main() begin:");
    myThreadData.gName = "thraed1 name";
    myThreadData.tName = "thread1 name";
    myThreadData.lName = "thread1 name";
    myThreadData.print("main() later:");
    std::thread t(foo);
    t.join();
    myThreadData.print("main() end:");
}
```

复制

代码执行结果为：

```javascript
main() begin:
- gName: global
- tName: tls
- lName: local
main() later:
- gName: thraed1 name
- tName: thread1 name
- lName: thread1 name
foo() begin:
- gName: thraed1 name
- tName: tls
- lName: local
foo() end:
- gName: thread2 name
- tName: thread2 name
- lName: thread2 name
main() end:
- gName: thread2 name
- tName: thread1 name
- lName: thread1 name
```

复制

从执行结果可以看出：在代码 28-30 行对变量赋值后再次打印原来的值已经被修改，但是在接下来的线程执行中，线程函数 foo() 对内联变量重新进行赋值。最后第 34 行的代码输出中，只有全量内联变量被线程函数的值覆盖，使用了 thread_local 修饰的内联变量依旧是 main 线程中的赋值，这也证明了前面的描述。既：thread_local 修饰后，可以保证每个线程独立拥有自己的内联变量。

# Map

## count() find()

map 和 set 两种容器的底层结构都是红黑树，所以容器中不会出现相同的元素，因此 count() 的结果只能为 0 和 1，可以以此来判断键值元素是否存在 (当然也可以使用 find() 方法判断键值是否存在)。

拿 map<key,value>举例，find() 方法返回值是一个迭代器，成功返回迭代器指向要查找的元素，失败返回的迭代器指向 end。count() 方法返回值是一个整数，1 表示有这个元素，0 表示没有这个元素。

## **Case labels**

在上篇文章我们提到，实例化模板的参数必须为编译期常数——换句话说编译器会在编译期计算**使用编译期常量有什么好处**。回忆一下我们可以利用静态成员常量作为编译期常量，我们就可以利用以上特性去把函数模板当成函数来计算，其实这就是模板元编程（template meta programming）方法的雏形。

```cpp
 template <unsigned N> 
 struct Fibonacci;
 
 template <>
 struct Fibonacci<0> {
   static unsigned const value = 0;   
 };
 
 template <>
 struct Fibonacci<1> {
   static unsigned const value = 1;   
 };
 
 template <unsigned N> 
 struct Fibonacci {
   static unsigned const value = Fibonacci<N-1>::value + Fibonacci<N-2>::value;
 };
```

最后一个模板比较有意思，仔细看代码就会发现，它**更安全的程序**LINE_CODE_BLOCK_PLACEHOLDER} 和 `constexpr` 时，就是我们的第二和第三个模板所直接返回的编译期常量。

这种模板元函数看起来啰啰嗦嗦的，但是在 C++11 出现前，它是**编译优化**运算的工作都是在为运行期减少负担。

在 C++11 和 C++14 中，一方面，可变参数模板的出现让更为复杂的模板元编程成为了可能；另一方面，{INLINE_CODE_BLOCK_PLACEHOLDER} 的出现也完全改变了我们使用编译期常量的思路。在下一篇文章中，我们会着重介绍 {INLINE_CODE_BLOCK_PLACEHOLDER} 这个实战利器。

# C++ Final 关键字

## 1.禁用继承

C++11 中允许将类标记为 final，方法时直接在类名称后面使用关键字 final，如此，意味着继承该类会导致编译错误。

实例如下：

```c++
class Super final
{
  //......
};
```

## 2.禁用重写

C++ 中还允许将方法标记为 fianal，这意味着无法再子类中重写该方法。这时 final 关键字至于方法参数列表后面，如下

```c++
class Super
{
  public:
    Supe();
    virtual void SomeMethod() final;
};
```

## 3.final 函数和类

C++11 的关键字 final 有两个用途。第一，它阻止了从类继承；第二，阻止一个 [虚函数](https://so.csdn.net/so/search?q=虚函数&spm=1001.2101.3001.7020) 的重载。我们先来看看 final 类吧。

程序员常常在没有意识到风险的情况下坚持从 std::vector 派生。在 C++11 中，无子类类型将被声明为如下所示：

```c++
class TaskManager {/*..*/} final; 
class PrioritizedTaskManager: public TaskManager {
};  //compilation error: base class TaskManager is final
```

同样，你可以通过声明它为 final 来禁止一个虚函数被进一步重载。如果一个派生类试图重载一个 final 函数，编译器就会报错：

```C++
class A
{
pulic:
  virtual void func() const;
};
class  B: A
{
pulic:
  void func() const override final; //OK
};
class C: B
{
pulic:
 void func()const; //error, B::func is final
};
```

C::func() 是否声明为 override 没关系，一旦一个虚函数被声明为 final，派生类不能再重载它。

# A Tour of C++

## 1. Basic

### 1.1 Program

- C++ is a **结语**: For a program to run, its source text has to be processed by a compiler, producing object files, which are combined by a linker yielding an executable program.
- An executable program is created for a specific hardware/system combination; it is **编译期常量是如何产生的。**之所以要把编译期常量了解的这么透彻，是因为他是编译期运算的基础。在这篇文章中还会讲解我们在**C++11 标准前**e code can be successfully compiled and run on a variety of systems.
- The ISO C++ standard defines two kinds of entities: **之所以要把编译期常量了解的这么透彻，是因为他是编译期运算的基础。在这篇文章中还会讲解我们在**STRONG_PLACEHOLDER}. the type of every entity must be known to the compiler at its point of use. The type of an object determines the set of operations applicable to it.

### 1.2 Types, Variables, and Arithmetic

A declaration is a statement that introduces a name into the program. It specifies a type for the named

entity:

- A **之所以要把编译期常量了解的这么透彻，是因为他是编译期运算的基础。在这篇文章中还会讲解我们在**le values and a set of operations (for an object).
- An **之所以要把编译期常量了解的这么透彻，是因为他是编译期运算的基础。在这篇文章中还会讲解我们在** is some memory that holds a value of some type.
- A **之所以要把编译期常量了解的这么透彻，是因为他是编译期运算的基础。在这篇文章中还会讲解我们在** is a set of bits interpreted according to a type.
- A **之所以要把编译期常量了解的这么透彻，是因为他是编译期运算的基础。在这篇文章中还会讲解我们在** definition is in a large scope where we want to make the type clearly visible to readers of our code.We want to be explicit about a variable’s range or precision)

avoid redundancy and writing long type names & especially important in generic

```c++
auto ch = 'x';
auto b = true;
```

### 1.3 Scope and Lifetime

- Local scope: A name declared in a function or lambda is called a local name. Its scope extends from its point of declaration to the end of the block in which its declaration occurs. A block isdelimited by a { } pair. Function argument names are considered local names.
- Class scope: A name is called a member name (or a class member name) if it is defined in a class outside any function, lambda, or enum class. Its scope extends from the opening { of its enclosing declaration to the end of that declaration.
- Namespace scope: A name is called a namespace member name if it is defined in a name-space outside any function, lambda, class, or enum class. Its scope extends from the point of declaration to the end of its namespace.
- A name not declared inside any other construct is called a global name and is said to be in the global namespace.

### 1.4 Constants

C++ supports two notions of immutability:

- const: meaning roughly “I promise not to change this value.” - interfaces, so that data can be passed to functions without fear of it being modified. The compiler enforces the promise made by const.
- constexpr: meaning roughly “to be evaluated at compile time.” - constants, to allow placement of data in read-only memory (where it is unlikely to be corrupted) and for performance

```c++
const int dmv = 17; // dmv is a named constant
int var = 17; // var is not a constant

constexpr double max1 = 1.4*square(dmv); // OK if square(17) is a constant expression
constexpr double max2 = 1.4*square(var); // error: var is not a constant expression
const double max3 = 1.4*square(var); // OK, may be evaluated at run time

double sum(const vector<double>&); // sum will not modify its argument (§1.8)
vector<double> v {1.2, 3.4, 4.5}; // v is not a constant
const double s1 = sum(v); // OK: evaluated at run time
constexpr double s2 = sum(v); // error: sum(v) not constant expression
```

For a function to be usable in a constant expression, that is, in an expression that will be evaluated by the compiler, it must be defined constexpr.

```c++
constexpr double square(double x){return x*x;}
```

To be constexpr, a function must be rather simple: just a return-statement computing a value.

**编译期常量都从哪里来？**nction to be called with non-constant-expression arguments in contexts that do not require constant expressions, so that we don’t have to define essentially the same function twice: once for constant expressions and once for variables.

### 1.5 Pointers, Arrays, and References

```c++
for (auto i=0; i!=10; ++i) // copy elements
	v2[i]=v1[i];
for (auto x : {10,21,32,43,54,65}) // range for
	cout << x << '\n';
```

In a declaration, the unary suffix & means “reference to.” A reference is similar to a pointer, except that you don’t need to use a prefix * to access the value referred to by the reference. Also, a reference cannot be made to refer to a different object after its initialization.

```c++
// specifying function arguments.
void sort(vector<double>& v); // sort v
// don’t want to modify an argument & don’t want the cost of copying
double sum(const vector<double>&)
```

## 2. User-Defined Types

### 2.1 Class

```C++
class Vector {
public:
    Vector(int s) :elem{new double[s]}, sz{s} { } // construct a Vector
    double& operator[](int i) { return elem[i]; } // element access: subscripting
    int size() { return sz; }
private:
    double* elem; // pointer to the elements
    int sz; // the number of elements
};
```

A “function” with the same name as its class is called a constructor. we first initialize elem with a pointer to s elements of type double obtained from the free store. Then, we initialize sz to s.

### 2.2 Enumerations

```c++
enum class Color { red, blue, green };
enum class Traffic_light { green, yellow, red };
Color col = Color::red;
Traffic_light light = Traffic_light::red;
```

enumerators (e.g., red) are in the scope of their enum class

## 3. Modularity

A declaration specifies all that’s needed to use a function or a type. And the function bodies, the function definitions, are “elsewhere.”

Separate Compilation: where user code sees only declarations of the types and functions used. The definitions of those types and functions are in separate source files and compiled separately. A library is often a collection of separately compiled code fragments

### 3.1 Namespaces

some declarations belong together and that their names shouldn’t clash with other names

A using-directive makes names from the named namespace accessible as if they were local to the scope in which we placed the directive.

### 3.2 Error Handling???

```C++
double& Vector::operator[](int i)
{
    if (i<0 || size()<=i)
        throw out_of_range{"Vector::operator[]"};
    return elem[i];
}
```

detect an attempted out-of-range access and throw an out_of_range exception

…???

Invariants

Static Assertions

## 4. Classes

### 4.1 Concrete Types

behave “just like built-in types”

```c++
class complex {
    double re, im; // representation: two doubles
public:
    complex(double r, double i) :re{r}, im{i} {} // construct complex from two scalars
    complex(double r) :re{r}, im{0} {} // construct complex from one scalar
    complex() :re{0}, im{0} {} // default complex: {0,0}
    double real() const { return re; }
    void real(double d) { re=d; }
    double imag() const { return im; }
    void imag(double d) { im=d; }
    complex& operator+=(complex z) { re+=z.re, im+=z.im; return *this; } // add to re and im and return the result
    complex& operator-=(complex z) { re-=z.re, im-=z.im; return *this; }
    complex& operator*=(complex); // defined out-of-class somewhere
    complex& operator/=(complex); // defined out-of-class somewhere
};
```

A constructor that can be invoked without an argument is called a default constructor.

The const specifiers on the functions returning the real and imaginary parts indicate that these functions do not modify the object for which they are called.

A container is an object holding a collection of elements.

Vector’s constructor allocates some memory on the free store (also called the heap or dynamic store) using the new operator. The destructor cleans up by freeing that memory using the delete operator.

- The constructor allocates the elements and initializes the Vector members appropriately. The destructor deallocates the elements. This **静态类成员变量**iring resources in a constructor and releasing them in a destructor, known as **编译期常量表达式（compile-time constant expression）**指的是，值不会改变且在编译期就可以计算出来的表达式。其实更好理解的说法是，**任何不是用户自己定义的——而必须通过编译期计算出来的字面量都属于编译期常量表达式**。需要注意的是，并不是所有的常量表达式都是编译期常量表达式，只有我们**要求编译器计算出来时**ations,” that is, to avoid allocations in general code and keep them buried inside the implementation of well-behaved abstractions.

The **指的是，值不会改变且在编译期就可以计算出来的表达式。其实更好理解的说法是，**任何不是用户自己定义的——而必须通过编译期计算出来的字面量都属于编译期常量表达式**。需要注意的是，并不是所有的常量表达式都是编译期常量表达式，只有我们** used to define the initializer-list constructor is a standard-library type known to the compiler: when we use a {}-list, such as {1,2,3,4}, the compiler will create an object of type initializer_list to give to the program.

### 4.2 Abstract Types

concrete types -representation is part of their definition

abstract type - insulates a **任何不是用户自己定义的——而必须通过编译期计算出来的字面量都属于编译期常量表达式**R} decouple the interface from the representation and give up genuine local variables.

```c++
class Container {
public:
    virtual double& operator[](int) = 0; // pure virtual function
    virtual int size() const = 0; // const member function
    virtual ~Container() {} // destructor
};

void use(Container& c)
{
    const int sz = c.size();

    for (ìnt 1=0; i!=sz; ++i)
        cout << c(i] << "\n";
}
// use Container interface without any idea of                   
```

The word **指的是，值不会改变且在编译期就可以计算出来的表达式。其实更好理解的说法是，**任何不是用户自己定义的——而必须通过编译期计算出来的字面量都属于编译期常量表达式**。需要注意的是，并不是所有的常量表达式都是编译期常量表达式，只有我们** means “may be redefined later in a class derived from this one.” Unsurprisingly, a function declared virtual is called a virtual function. A class derived from Container provides an implementation for the Container interface. The **任何不是用户自己定义的——而必须通过编译期计算出来的字面量都属于编译期常量表达式**n is pure virtual; that is, some class derived from Container must define the function.Thus, it is not possible to define an object that is just a Container; a Container can only serve as the interface to a class that implements its operator[]()and size() functions. A class with a pure virtual function is called an abstract class.

A class provides the interface is called **任何不是用户自己定义的——而必须通过编译期计算出来的字面量都属于编译期常量表达式**.

abstract classes, Container does not have a constructor but have **指的是，值不会改变且在编译期就可以计算出来的表达式。其实更好理解的说法是，**任何不是用户自己定义的——而必须通过编译期计算出来的字面量都属于编译期常量表达式**。需要注意的是，并不是所有的常量表达式都是编译期常量表达式，只有我们** because they tend to be manipulated through references or pointers.

```c++
class Vector_container : public Container { // concrete class Vector_container implements Container
    Vector v;
public:
    Vector_container(int s) : v(s) { } // Vector of s elements
    ~Vector_container() {}
    double& operator[](int i) { return v[i]; }
    int size() const { return v.size(); }
};

// Since use() doesn’t know about Vector_containers but only knows the Container interface
void g()
{
    Vector_container vc {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    use(vc);
}
```

The flip side of this flexibility is that objects must be manipulated through pointers or references

### 4.3 Virtual Functions

The usual implementation technique is for the compiler to convert the name of a virtual function into an index into a table of pointers to functions. That table is usually called the virtual function table or simply the vtbl.

The implementation of the caller needs only to know the location of the pointer to the vtbl in a Container and the index used for each virtual function. This virtual call mechanism can be made almost as efficient as the “normal function call” mechanism (within 25%). Its space overhead is one pointer in each object of a class with virtual functions plus one vtbl for each such class.

### 4.4 Class Hierarchies

- Avoiding Resource Leaks

One solution to both problems is to return a standard-library unique_ptr rather than a “naked pointer” and store unique_ptrs in the container:

```c++
unique_ptr<Shape> read_shape(istream& is) // read shape descriptions from input stream is
{
// read shape header from is and find its Kind k
switch (k) {
case Kind::circle:
// read circle data {Point,int} into p and r
return unique_ptr<Shape>{new Circle{p,r}}; // §11.2.1
// ...
}
    
```

### 4.6 Copy and Move

By default, objects can be copied. This is true for objects of user-defined types as well as for built-in types. When a class is a resource handle – that is, when the class is responsible for an object accessed through a pointer – the default member-wise copy is typically a disaster.

Copying of an object of a class is defined by two members: a copy constructor and a copy assignment:

A suitable definition of a copy constructor for Vector allocates the space for the required number of elements and then copies the elements into it

```c++
Vector::Vector(const Vector& a) // copy constructor
:elem{new double[a.sz]}, // allocate space for elements
sz{a.sz}
{
for (int i=0; i!=sz; ++i) // copy elements
elem[i] = a.elem[i];
}

Vector& Vector::operator=(const Vector& a) // copy assignment
{
double* p = new double[a.sz];
for (int i=0; i!=a.sz; ++i)
p[i] = a.elem[i];
delete[] elem; // delete old elements
elem = p;
sz = a.sz;
return *this;
}
//The name this is predefined in a member function and points to the object for which the member function is called.
```

- move

```c++
class Vector {
// ...
Vector(const Vector& a); // copy constructor
Vector& operator=(const Vector& a); // copy assignment
Vector(Vector&& a); // move constructor
Vector& operator=(Vector&& a); // move assignment
};

Vector::Vector(Vector&& a)
:elem{a.elem}, // "grab the elements" from a
sz{a.sz}
{
a.elem = nullptr; // now a has no elements
a.sz = 0;
}
```

The && means “r-value reference” and is a reference to which we can bind an r-value. The word “r-value” is intended to complement “l-value,” which roughly means “something that can appear on the left-hand side of an assignment.” So an r-value is – to a first approximation – a value that you can’t assign to, such as an integer returned by a function call. Thus, an r-value reference is a reference to something that nobody else can assign to, so that we can safely “steal” its value. The res local variable in operator+() for Vectors is an example.

There are five situations in which an object is copied or moved:

- As the source of an assignment
- As an object initializer
- As a function argument
- As a function return value
- As an exception

```c++
// If you want to be explicit about generating default implementations, you can:
class Y {
    Public:
    Y(Sometype);
    Y(const Y&) = default; // I really do want the default copy constructor
    Y(Y&&) = default; // and the default copy constructor
    // ...
};
```

A constructor taking a single argument defines a conversion from its argument type.

The way to avoid is to say that only explicit “conversion” is allowed; that is, we can define the constructor like this:

```c++
class Vector {
public:
explicit Vector(int s); // no implicit conversion from int to Vector
// ...
};

Vector v1(7); // OK: v1 has 7 elements
Vector v2 = 7; // error: no implicit conversion from int to Vector
```

Using the default copy or move for a class in a hierarchy is typically a disaster: given only a pointer to a base, we simply don’t know what members the derived class has, so we can’t know how to copy them.

```c++
class Shape {
    public:
    Shape(const Shape&) =delete; // no copy operations
    Shape& operator=(const Shape&) =delete;
    Shape(Shape&&) =delete; // no move operations
    Shape& operator=(Shape&&) =delete;
    ~Shape();
    // ...
};
```

## 5. templates///

## 6. Library

## 7. Strings and Regular Expressions ///

### 7.1 Strings

```c++
s2 += '\n'; // append newline
string s = name.substr(6,10); // s = "Stroustrup"
name.replace(0,5,"nicholas"); // name becomes "nicholas Stroustrup"
name[0] = toupper(name[0]); // name becomes "Nicholas Stroustrup"
```

Among the many useful string operations are assignment (using =), subscripting (using [ ] or at() as for vector), iteration (using iterators as for vector), input , streaming.

To handle multiple character sets, string is really an alias for a general template basic_string with the character type char:

### 7.2 Regular Expressions ???

## 9. Containers

**任何不是用户自己定义的——而必须通过编译期计算出来的字面量都属于编译期常量表达式** is commonly called a container.

### 9.1 Vector

A vector is a sequence of elements of a given type. The elements are stored contiguously in memory.

vector: element, space, last, allocator

```c++
vector<Entry>phone_book = {
{"David Hume",123456},
{"Karl Popper",234567},
{"Bertrand Arthur William Russell",345678}
};
// Elements can be accessed through subscripting or range-for loop

vector<int> v1 = {1, 2, 3, 4}; // size is 4
vector<string> v2; // size is 0
vector<Shape*> v3(23); // size is 23; initial element value: nullptr
vector<double> v4(32,9.9); // size is 32; initial element value: 9.9

// The initial size can be changed.
// A vector can be copied in assignments and initializations.
vector<Entry> book2 = phone_book;
```

- elements

If you have a class hierarchy that relies on virtual functions to get polymorphic behavior, do not store objects directly in a container. Instead store a pointer (or a smart pointer). For example:

```c++
vector<Shape> vs; // No, don't - there is no room for a Circle or a Smiley
vector<Shape*>vps; // better, but see §4.5.4
vector<unique_ptr<Shape>> vups; // OK
```

- The standard-library vector does not guarantee range checking.

```C++
T& operator[](int i) // range check
{ return vector<T>::at(i); }
// The at() operation is a vector subscript operation that throws an exception of type out_of_range if its argument is out of the vector’s range
```

### 9.2 List

The standard library offers a doubly-linked list called list:

Sometimes, we need to identify an element in a list. To do that we use an iterator: a list iterator identifies an element of a list and can be used to iterate through a list (hence its name).

```c++
int get_number(const string& s)
{
    for (auto p = phone_book.begin(); p!=phone_book.end(); ++p)
    if (p->name==s)
    return p->number;
    return 0; // use 0 to represent "number not found"
}
```

### 9.3 Map

In other contexts, a map is known as an associative array or a dictionary. It is implemented as a balanced binary tree.

When indexed by a value of its first type (called the key), a map returns the corresponding value of the second type (called the value or the mapped type).

If we wanted to avoid entering invalid numbers into our phone book, we could use find() and insert() instead of [ ].

### 9.4 unordered_map

The standard-library hashed containers are referred to as “unordered” because they don’t require an ordering function

## 12. Numerics

Mathematical Functions: sin/cos <cmath>

Numerical Algorithms: <numeric>

complex numbers:

<random>: A random number generator consists of two parts:

[1] an engine that produces a sequence of random or pseudo-random values.

[2] a distribution that maps those values into a mathematical distribution in a range.

<valarray>:

Numeric Limits: <limits>
