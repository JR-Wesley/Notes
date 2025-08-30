---
dateCreated: 2023-01-15
dateModified: 2025-07-27
---

# Static

在 C++ 中，`static` 关键字是一个多功能的关键字，用于控制变量、函数和类成员的**生命周期**、**作用域**和**访问权限**。以下是 `static` 在 C++ 中的详细用法和示例：

---

## **一、静态局部变量**

### **定义**

在函数内部使用 `static` 修饰的变量称为**静态局部变量**。

### **特点**

1. **生命周期**：从程序开始到结束，变量的值在函数调用之间保留。
2. **作用域**：仍局限于函数内部。
3. **存储位置**：静态存储区（全局数据区），而非栈区。
4. **初始化**：静态局部变量只初始化一次，后续调用函数时不再重新初始化。

### **示例**

```
#include <iostream>
using namespace std;

void counter() {
    static int count = 0; // 静态局部变量
    count++;
    cout << "Count: " << count << endl;
}

int main() {
    for (int i = 0; i < 5; ++i) {
        counter(); // 每次调用count的值保留
    }
    return 0;
}
```

**输出**：

深色版本

```
Count: 1
Count: 2
Count: 3
Count: 4
Count: 5
```

### **注意事项**

- 如果不使用 `static`，普通局部变量每次函数调用后会被销毁，重新初始化为 0。
- 静态局部变量适用于需要跨函数调用保留状态的场景（如计数器、缓存等）。

---

## **二、静态全局变量**

### **定义**

在文件作用域（函数外部）使用 `static` 修饰的全局变量或函数称为**静态全局变量**和**静态函数**。

### **特点**

1. **作用域**：仅限于定义它的文件（内部链接），其他文件无法访问。
2. **链接属性**：静态全局变量具有**内部链接**（Internal Linkage），而非外部链接（External Linkage）。
3. **目的**：避免命名冲突，实现文件级别的封装。

### **示例**


```
// file1.cpp
#include <iostream>
using namespace std;

static int file1_var = 10; // 静态全局变量，仅在file1.cpp中可见

void printFile1Var() {
    cout << "file1_var: " << file1_var << endl;
}

// file2.cpp
#include <iostream>
using namespace std;

int file2_var = 20; // 普通全局变量，具有外部链接

extern int file1_var; // 错误！无法访问file1.cpp中的静态全局变量
void printFile2Var() {
    cout << "file2_var: " << file2_var << endl;
}
```

### **注意事项**

- 如果全局变量不需要跨文件访问，使用 `static` 可以避免命名冲突。
- 静态函数的作用域同样限于定义它的文件。

---

## **三、静态成员变量**

### **定义**

在类中使用 `static` 修饰的成员变量称为**静态成员变量**。

### **特点**

1. **归属**：属于类本身，而非类的某个对象。
2. **共享性**：所有对象共享同一个静态成员变量。
3. **访问权限**：受类的访问修饰符（`public`/`private`）控制。
4. **定义与初始化**：必须在类外定义并初始化（C++17 之前）。

### **示例**

```
#include <iostream>
using namespace std;

class Box {
public:
    static int objectCount; // 声明静态成员变量
    Box() { objectCount++; } // 构造函数中增加计数
    ~Box() { objectCount--; } // 析构函数中减少计数
};

// 类外定义并初始化静态成员变量
int Box::objectCount = 0;

int main() {
    Box b1, b2;
    cout << "Number of Box objects: " << Box::objectCount << endl; // 输出2
    return 0;
}
```

### **注意事项**

- 静态成员变量不能直接通过对象访问（除非是 `public`），推荐通过 `类名::静态成员变量` 访问。
- 静态成员变量的初始化必须在类外完成。

---

## **四、静态成员函数**

### **定义**

在类中使用 `static` 修饰的成员函数称为**静态成员函数**。

### **特点**

1. **归属**：属于类本身，而非类的某个对象。
2. **访问权限**：受类的访问修饰符控制。
3. **限制**：
    - 不能访问非静态成员变量（因为没有 `this` 指针）。
    - 只能访问静态成员变量和其他静态成员函数。

### **示例**

```
#include <iostream>
using namespace std;

class MathUtils {
public:
    static int add(int a, int b) { // 静态成员函数
        return a + b;
    }
};

int main() {
    int result = MathUtils::add(3, 4); // 直接通过类名调用
    cout << "Result: " << result << endl; // 输出7
    return 0;
}
```

### **注意事项**

- 静态成员函数通常用于提供与类相关的工具函数（如数学计算、工厂方法等）。

---

## **五、静态初始化顺序问题**

### **问题描述**

当多个翻译单元（源文件）中的静态变量相互依赖时，可能出现**静态初始化顺序问题**（Static Initialization Order Problem）。

### **示例**

cpp

深色版本

```
// file1.cpp
#include <iostream>
using namespace std;

static int globalVar1 = getGlobalVar2(); // 依赖globalVar2的初始化

// file2.cpp
#include <iostream>
using namespace std;

static int globalVar2 = 100; // 初始化后才被globalVar1使用

int getGlobalVar2() {
    return globalVar2;
}
```

**问题**：如果 `globalVar1` 在 `globalVar2` 初始化之前使用，会导致未定义行为。

### **解决方案**

- 使用函数返回静态局部变量（Meyers' Singleton 模式）：

    cpp

    深色版本

- ```
    int& getGlobalVar2() {
        static int globalVar2 = 100; // 延迟初始化
        return globalVar2;
    }
    ```

---

## **六、总结对比**

|用法|作用域|生命周期|存储位置|典型用途|
|---|---|---|---|---|
|静态局部变量|函数内部|程序运行期间|静态存储区|跨调用保留状态|
|静态全局变量|定义文件|程序运行期间|静态存储区|文件级封装，避免命名冲突|
|静态成员变量|类|程序运行期间|静态存储区|类级别的共享数据|
|静态成员函数|类|无（函数代码）|代码区|操作静态成员或类相关功能|

---

## **七、关键点**

1. **静态局部变量**：生命周期长，作用域小。
2. **静态全局变量**：作用域小，生命周期长。
3. **静态成员变量/函数**：属于类，共享性高。
4. **初始化顺序问题**：跨文件依赖时需谨慎处理。

通过合理使用 `static`，可以有效管理资源、避免命名冲突，并提高代码的封装性和可维护性。

# 引用

在 C++ 中，**左值（lvalue）**和**右值（rvalue）**是表达式的分类，它们决定了对象的生命周期、可寻址性以及如何参与赋值和资源管理。结合引用类型（`&` 和 `&&`），它们是现代 C++ 中实现**移动语义**。
- **和**：
    - 可以被多次使用。
    - 生命周期超出当前表达式。
    - 示例：

        cpp

        深色版本

- - ```
        int x = 10;          // x 是左值
        int* p = &x;         // 可以取地址
        x = 20;              // 左值可以赋值
        int arr[3];          // arr[0] 是左值
        int& ref = x;        // 左值引用
        ```
        





## **是表达式的分类，它们决定了对象的生命周期、可寻址性以及如何参与赋值和资源管理。结合引用类型（`&` 和 `&x`），它们是现代 C++ 中实现**

- **和**：临时存在的值，通常是字面量、临时对象或表达式结果，没有持久内存地址，不能取地址。
- **是表达式的分类，它们决定了对象的生命周期、可寻址性以及如何参与赋值和资源管理。结合引用类型（`&10` 和 `x = 10`），它们是现代 C++ 中实现**：
    - 生命周期仅限于当前表达式。
    - 通常不可修改（除非通过右值引用）。
    - 示例：


- - ```
        int y = 42;          // 42 是右值（字面量）
        int z = x + 5;       // x + 5 是右值（临时结果）
        std::string s = "hello"; // "hello" 是右值（临时字符串）
        ```
        





## **完美转发**

|特性|左值（lvalue）|右值（rvalue）|
|---|---|---|
|是否可取地址|✅ 可以（如 `10 = x`）|❌ 不可以（如 `&` 非法）|
|是否可赋值|✅ 可以（如 `&&`）|❌ 不可以（如 `T&` 非法）|
|生命周期|持久（超出当前表达式）|临时（仅当前表达式）|
|典型例子|变量、数组元素、解引用指针|字面量、临时对象、表达式结果|

---

# **1. 左值（lvalue）与右值（rvalue）**

## **左值（lvalue）**

- **定义**：只能绑定到**特点**。
- **右值（rvalue）**：
    - 避免拷贝，直接操作原始对象。
    - 修改左值的内容。
- **定义**：


- ```
    int x = 10;
    int& ref = x;       // 左值引用绑定左值
    ref = 20;           // 修改 x 的值
    ```

## **特点**

- **左值 vs 右值对比表**：只能绑定到**2. 引用类型：`T&&` 和 `std::move`**或通过 `T&` 转换的左值。
- **左值引用（lvalue Reference, `int x = 10; int& ref = x;`）**：
    - 实现**绑定目标**（资源转移，而非拷贝）。
    - 支持**左值**（保留参数的值类别）。
- **用途**：

    cpp

    深色版本

- ```
    int&& rref = 42;           // 右值引用绑定右值
    rref = 50;                 // 修改 rref 的值
    std::vector<int> v1 = {1,2,3};
    std::vector<int> v2 = std::move(v1); // 将 v1 转为右值，触发移动构造
    ```

## **示例**

|引用类型|可绑定对象|示例|
|---|---|---|
|`const T&`|左值|`int&& temp = 10; const int& cref = temp;`|
|`T&&`|左值或右值（延长临时对象寿命）|`int&& rref = x + 5;`|
|`std::move`|右值|`std::forward`|

---

# **右值引用（rvalue Reference, `std::vector`）**

## **绑定目标**

通过右值引用实现资源转移，避免深拷贝开销：

cpp

深色版本

```
class Buffer {
public:
    Buffer(Buffer&& other) noexcept {  // 移动构造函数
        data_ = other.data_;           // 直接“窃取”资源
        other.data_ = nullptr;         // 释放原资源所有权
    }
private:
    int* data_;
};
```

## **右值**

将左值显式转换为右值，触发移动语义：

cpp

深色版本

```
std::vector<int> v1 = {1,2,3};
std::vector<int> v2 = std::move(v1);  // v1 被“移动”，v2 接管资源
```

---

# **用途**

## **移动语义**

通过右值引用和 `emplace_back` 保留参数的值类别：

cpp

深色版本

```
template <typename T>
void wrapper(T&& arg) {
    target(std::forward<T>(arg));  // 保持 arg 的左/右值属性
}
```

## **完美转发**

cpp

深色版本

```
template <typename T, typename... Args>
std::unique_ptr<T> create(Args&&... args) {
    return std::make_unique<T>(std::forward<Args>(args)...);
}
```

---

# **示例**

## **绑定规则总结**

- **3. 移动语义与右值引用**：直接构造元素，避免临时对象。
- **移动构造函数**：在容器扩容时通过移动元素减少拷贝。

## **`std::move_iterator` 的作用**

- **4. 完美转发与右值引用**：`std::unique_ptr` 和 `std::shared_ptr` 利用移动语义实现所有权转移。
- **模板中的引用折叠**：通过右值引用实现资源接管。

## **示例：工厂函数**

根据参数类型选择不同的函数：

cpp

深色版本

```
void process(int& lval) { std::cout << "Lvalue\n"; }
void process(int&& rval) { std::cout << "Rvalue\n"; }

int main() {
    int x = 10;
    process(x);      // 调用 process(int&)
    process(20);     // 调用 process(int&&)
}
```

---

# **5. 典型应用场景**

1. **1. 容器操作优化**
    右值引用变量有名称和地址，因此它本身是左值：

    cpp

    深色版本

2. ```
    int&& rref = 42;
    int& lref = rref;  // 合法！rref 是左值
    ```

3. **`std::move` 的 `std::move`**
    `std::unique_ptr` 仅用于显式将左值转为右值，通常在资源转移时使用，而非随意调用。
    
4. **`&`**
    右值引用常与智能指针（如 `&&`）结合，确保资源安全。
    
5. **2. 资源管理**
    右值引用绑定的临时对象生命周期仅限于当前表达式，需谨慎处理。

---

# **智能指针**

|概念|核心要点|
|---|---|
|左值（lvalue）|有持久内存地址，可取地址，可赋值，如变量、数组元素。|
|右值（rvalue）|临时值，无持久地址，如字面量、表达式结果，需通过右值引用绑定。|
|左值引用 `std::vector`|绑定左值，用于避免拷贝和修改原始对象。|
|右值引用 `std::forward`|绑定右值，实现移动语义和完美转发，提升性能。|
|移动语义|通过右值引用转移资源，避免深拷贝，如 `someStruct<42ul, 'e', GREEN> theStruct;` 的移动构造函数。|
|完美转发|通过 `someStruct` 保留参数的值类别，常用于模板函数和工厂函数。|

通过合理使用左值、右值及引用类型，可以显著提升 C++ 程序的性能和安全性，尤其是在处理大型对象、资源管理及模板编程时。

# 友元

在 C++ 中，友元（Friend）机制是一种特殊的访问权限控制方式，它允许特定的函数或类访问另一个类中的私有（private）和保护（protected）成员，打破了类的封装性限制。

## 友元的主要形式

1. **文件/网络资源管理**：一个非成员函数被声明为某个类的友元后，可以访问该类的私有和保护成员。

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

2. **3. 函数重载**：一个类被声明为另一个类的友元后，该类的所有成员函数都能访问对方类的私有和保护成员。

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

**6. 常见误区与注意事项**

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

**右值引用变量本身是左值**

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

**`sizeof` 的使用场景**

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

**避免混用裸指针与智能指针**

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

**生命周期管理**

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

# =delete

C++11 中，当我们定义一个类的成员函数时，如果后面使用 "=delete" 去修饰，那么就表示这个函数被定义为 deleted，也就意味着这个成员函数不能再被调用，否则就会出错，编译时直接报错。

## 巧妙用法

这里说个=delete 的巧妙用法，在 C++ 里会有很多隐式类型转换，如下代码，

当我们把 100.0 传给 obj.func() 时，发生了隐式类型转换，由 double 转为了 int，有时我们不希望发生这样的转换，我们就是希望传进来的参数和规定的类型一致，那么此时可以使用=delete 来达到这个目的，如下，

```c++
#include <cstdio>

class TestClass
{
public:

    void func(int data) { printf("data: %d\n", data); }
    void func(double data)=delete;

};


int main(void)
{

    TestClass obj;
    obj.func(100);
    obj.func(100.0);
    
    return 0;

}
```

我们把参数类型是 double 的重载函数加上=delete 进行修饰，表示这个函数被删除，那么用户就不能使用这个函数了，这样再编译就会出错，

# Map

## count() find()

map 和 set 两种容器的底层结构都是红黑树，所以容器中不会出现相同的元素，因此 count() 的结果只能为 0 和 1，可以以此来判断键值元素是否存在 (当然也可以使用 find() 方法判断键值是否存在)。

拿 map<key,value>举例，find() 方法返回值是一个迭代器，成功返回迭代器指向要查找的元素，失败返回的迭代器指向 end。count() 方法返回值是一个整数，1 表示有这个元素，0 表示没有这个元素。

# Constexpr

C++20 都支持虚函数的 constexpr 了，我打算用三篇读文章讲清楚编译期常量和 constexpr 这个东西和编译期常量的关系，即为什么需要他来辅助解决这个问题。最后帮助读者在实际编码过程中能够有意识地去运用他们，这才是终极目标。这篇文章中会讲到隐藏在日常编程中的各种编译期常量，以及他们存在的意义。

## **7. 总结**

想要用编译期常量就要首先知道它们是什么，一般出现在哪里和运行期常量有什么区别，因此我打算用第一篇文章重点分析编译期常量以及使用他们有什么好处。

编译期常量 (Compile-time constants) 是 C++ 中相当重要的一部分，整体而言他们有助提高**友元函数**，并提高程序的性能。这篇文章中出现的编译期常量都是在 C++11 之前就可以使用的，constexpr 是 C++11 的新特性，所以各位不要有心理包袱。

总有些东西是编译器要求编译期间就要确定的，除了变量的类型外，最频繁出现的地方就是数组、switch 的 case 标签和模板了。

### **友元类**

如果我们想要创建一个不是动态分配内存的数组，那么我们就必须给他设定一个 size——这个 size 必须在编译期间就知道，因此静态数组的大小是编译期常量。

```cpp
 int someArray[520];
```

只有这么做，编译器才能准确地解算出到底要分配给这个数组多少内存。如果这个数组在函数中，数组的内存就会被预留在该函数的栈帧中；如果这个数组是类的一个成员，那么编译器要确定数组的大小以确定这个类成员的大小——无论哪种情况，编译器都要知道这个数组具体的 size。

有些时候我们不用显示得指明数组的大小，我们用字符串或花括号来初始化数组的时候，编译器会实现帮我们数好这个数组的大小。

```cpp
 int someArray[] = {5, 2, 0};
 char charArray[] = "Ich liebe dich.";
```

### **1 内联变量的缘起**

除了类型以外，数字也可以作为模板的参数。这些数值变量包括 int，long，short，bool，char 和弱枚举 enum 等。

```cpp
 enum Color {RED, GREEN, BLUE};
 
 template<unsigned long N, char ID, Color C>
 struct someStruct {};
 
 someStruct<42ul, 'e', GREEN> theStruct;
```

### **编程秘籍**

既然编译器在初始化模板的时候必须知道模板的类型，那么这些模板的参数也必须是编译期常量。

switch 语句的分支判断也必须是编译期常量，和上边模板的情况非常类似。

```cpp
 void comment(int phrase) {
   switch(phrase) {
   case 42:
   std::cout << "You are right!" << std::endl;
   break;
   case BLUE:
   std::cout << "Don't be upset!" << std::endl;
   break;
   case 'z':
   std::cout << "You are the last one!" << std::endl;
   break;
   default:
   std::cout << "This is beyond what I can handle..." << std::endl;
   }
 }
```

## **2 内联变量的使用**

如果编译期常量的使用方法只有上边呈现的几种，那你大概会感觉有些无聊了。事实上，关于编译期常量我们能做的事情还有许多，他们能帮助我们去实现更高效的程序。

### **3 Constexpr Static 和 inline**

编译期常量能让我们写出更有逻辑的代码——在编译期就体现出逻辑。比如矩阵相乘：

```cpp
 class Matrix{
   unsigned rowCount;
   unsigned columnCount;
   //...
 };
```

我们都知道，两个矩阵相乘，当且仅当左矩阵的列数等于右矩阵的行数，如果不满足这个规则的话，那就完蛋了，所以针对上边矩阵的乘法，我们在函数中要做一些判断：

```cpp
 Matrix operator*(Matrix const& lhs, Matrix const& rhs) {
   if(lhs.getColumnCount() != rhs.getRowCount()) {
     throw OhWeHaveAProblem(); 
   }
   
   //...
 }
```

但是如果我们在编译期就知道了矩阵的 size，那么我们就可以把上边的判断放在模板中完成——这样的话不同 size 的矩阵一下子就成了不同类型的变量了。这样我们的矩阵乘法也相应变得简单了一些：

```cpp
 template <unsigned Rows, unsigned Columns>
 class Matrix {
   /* ... */
 };
 
 template <unsigned N, unsigned M, unsigned P>
 Matrix<N, P> operator*(Matrix<N, M> const& lhs, Matrix<M, P> const& rhs) {
   /* ... */
 }
 
 Matrix<1, 2> m12 = /* ... */;
 Matrix<2, 3> m23 = /* ... */;
 auto m13 = m12 * m23; // OK
 auto mX = m23 * m13;  // Compile Error!
```

在这个例子中，编译器本身就阻止了错误的发生，还有很多其他的例子——更复杂的例子在编译期间使用模板。从 C++11 后有一堆这样的模板都定义在了标准库 STL 中，这个之后再说。所以大家不要觉得上边这种做法是脱裤子放屁，相当于我们把运行时的条件判断交给了编译期来做，前提就是矩阵的类型必须是编译期常量。你可能会问，除了像上边直接用常数来实例化矩阵，有没有其他方法来告诉编译器这是个编译期常量呢？请往下看。

### **4 内联变量和 thread_local**

编译器能根据编译期常量来实现各种不同的优化。比如，如果在一个 if 判断语句中，其中一个条件是编译期常量，编译器知道在这个判断句中一定会走某一条路，那么编译器就会把这个 if 语句优化掉，留下只会走的那一条路。

```cpp
 if (sizeof(void*) == 4) {
   std::cout << "This is a 32-bit system!" << std::endl;
 } else {
   std::cout << "This is a 64-bit system!" << std::endl;
 }
```

在上例中，编译器就会直接利用其中某一个 cout 语句来替换掉整个 if 代码块——反正运行代码的机器是 32 还是 64 位的又不会变。另一个可以优化的地方在空间优化。总体来说，如果我们的对象利用编译期常数来存储数值，那么我们就不用在这个对象中再占用内存存储这些数。就拿本文之前的例子来举例：

- someStruct 结构中包含一个‘unsigned long’，一个‘char’，和一个‘color’，尽管如此他的实例对象却只占用一个 byte 左右的空间。
- 矩阵相乘的时候，我们在矩阵中也没必要花费空间去存储矩阵的行数和列数了。

**从编译期常量谈起**

这一篇文章只讲到了编译期常量，为了使编译器在编译期间计算出常量，我们在 C++11 标准之前和之后都采用了不同的方法去实现它。在第二篇文章中，我会将主要精力放在 C++11 标准之前的编译期计算的问题，通过展现一系列蹩脚的方法来引出我们的主角——constexpr。

在第一篇文章中，我把主要精力放在了什么是编译期常量，以及编译期常量有什么作用上。在这一篇文章中，我将更详细地介绍**程序的正确性** calculations），通过了解这些比较原始的方法，我们能够更好地理解 C++11 标准为编译期运算方面所做的工作。

作者：小天狼星不来客

链接：https://zhuanlan.zhihu.com/p/256416683

来源：知乎

著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

## **数组中的编译期常量**

在我们的经验中，大部分编译期常量的来源还是字面常量（literals）以及枚举量（enumerations）。比如上一篇文章我写的 `p` 中 `p` 的三个模板参数都是常量——分别是整形字面量、char 型字面量和枚举常量。

比较典型的编译期常量的来源就是内置的 `p` 操作符。编译器必须在编译期就知道一个变量占据了多少内存，所以它的值也可以被用作编译期常量。

```cpp
 class SomeClass {
   //...
 };
 int const count = 10;  //作为数组的size，编译期常量
 SomeClass theMovie[count] = { /* ... */}; //常量表达式，在编译期计算
 int const otherConst = 26; //只是常量，但不是编译期常量
 
 int i = 419;
 unsigned char buffer[sizeof(i)] = {};   //常量表达式，在编译期计算
```

另一个经常出现编译期常量最常出现的地方就是**模板中的编译期常量**（static class member variables），而枚举常量常常作为它的替换也出现在类中。

```cpp
 struct SomeStruct{
   static unsigned const size1 = 44;  //编译期常量
   enum { size2 = 45 };  //编译期常量
   int someIntegers[size1];  //常量表达式，在编译期计算
   double someDoubles[size2]; //常量表达式，在编译期计算
 };
```

与编译期常量对应的概念**Case labels**R} 的值，即 `switch` 此时变成了编译期常量表达式。

```cpp
const int i = 100;        
const int j = i * 200;    //常量表达式，但不是编译期常量表达式

const int k = 100;        
const int p = k * 200;    //是编译期常量表达式，由下边数组确定
unsigned char helper[p] = {}; //要求p是编译期常量表达式，在编译期就需确定
```

## **使用编译期常量有什么好处**

从上边的例子可以看出，有时我们可以**更安全的程序**可以做各种各样的编译期运算，实现在编译期就确定一个常量表达式的目的。事实上，由最简单的运算表达式出发，我们可以做到各种各样的编译期运算。比如非常简单：

```cpp
 int const doubleCount = 10;
 unsigned char doubleBuffer[doubleCount * sizeof(double)] = {};
```

除此之外，我们也可以用许多其他的操作，比如考虑下边并没有什么意义的代码：

```cpp
 std::string nonsense(char input) {
   switch(input) {
   case "some"[(sizeof(void*) == 4) ? 0 : 1]:
     return "Aachen";
   default:
     return "Wuhan";
   }
 }
```

上边的代码并没有什么实际的意义，但是我还是想解释一下。在上一篇文章我们解释过了，`sizeof(void*) == 4` 语句的每一个 case label 必须是编译期常量，表达式 `sizeof` 的意思是当前系统是不是一个 32 位系统，这个表达式由于 `0` 的原因是常量表达式，判断结果作为三元运算符的第一个参数，最后的 case label 由当前系统的位数分别是 "some" 的 "s"（是 32 位系统）或 "o"（不是 32 位系统）。返回的两个字符串分别是我的两个学校的城市。

尽管上边的例子是无意义的，我们仍然可以看出由这种方法写出的常量表达式很难读。我们可以改进可读性，将上边例子改写成：

```cpp
 std::string nonsense(char input) {
   auto const index = (sizeof(void*) == 4) ? 0 : 1;
   auto const someLabel = "some"[index];
   switch(input) {
   case someLabel:
     return "Aachen";
   default:
     return "Wuhan";
   }
 }
```

## **编译优化**

在上篇文章我们提到，实例化模板的参数必须为编译期常数——换句话说编译器会在编译期计算**结语**。回忆一下我们可以利用静态成员常量作为编译期常量，我们就可以利用以上特性去把函数模板当成函数来计算，其实这就是模板元编程（template meta programming）方法的雏形。

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

最后一个模板比较有意思，仔细看代码就会发现，它**编译期常量是如何产生的。**之所以要把编译期常量了解的这么透彻，是因为他是编译期运算的基础。在这篇文章中还会讲解我们在**C++11 标准前**去实例化参数为 N 的的模板，递归终止在模板参数为 `1` 和 `constexpr` 时，就是我们的第二和第三个模板所直接返回的编译期常量。

这种模板元函数看起来啰啰嗦嗦的，但是在 C++11 出现前，它是**之所以要把编译期常量了解的这么透彻，是因为他是编译期运算的基础。在这篇文章中还会讲解我们在**运算的工作都是在为运行期减少负担。

在 C++11 和 C++14 中，一方面，可变参数模板的出现让更为复杂的模板元编程成为了可能；另一方面，`constexpr` 的出现也完全改变了我们使用编译期常量的思路。在下一篇文章中，我们会着重介绍 {INLINE_CODE_BLOCK_PLACEHOLDER} 这个实战利器。

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

- C++ is a **之所以要把编译期常量了解的这么透彻，是因为他是编译期运算的基础。在这篇文章中还会讲解我们在**: For a program to run, its source text has to be processed by a compiler, producing
  object files, which are combined by a linker yielding an executable program.
- An executable program is created for a specific hardware/system combination; it is **编译期常量都从哪里来？**. we usually mean **静态类成员变量**; that is, the source code can be successfully compiled and run on a variety of systems.
- The ISO C++ standard defines two kinds of entities: **编译期常量表达式（compile-time constant expression）**指的是，值不会改变且在编译期就可以计算出来的表达式。其实更好理解的说法是，**任何不是用户自己定义的——而必须通过编译期计算出来的字面量都属于编译期常量表达式**。需要注意的是，并不是所有的常量表达式都是编译期常量表达式，只有我们**要求编译器计算出来时**(built-in types and loops) and loops.
- The **指的是，值不会改变且在编译期就可以计算出来的表达式。其实更好理解的说法是，**任何不是用户自己定义的——而必须通过编译期计算出来的字面量都属于编译期常量表达式**。需要注意的是，并不是所有的常量表达式都是编译期常量表达式，只有我们** are perfectly ordinary C++ code provided by every C++ implementation.
- C++ is a **任何不是用户自己定义的——而必须通过编译期计算出来的字面量都属于编译期常量表达式**. the type of every entity must be known to the compiler at its point of use. The type of an object determines the set of operations applicable to it.

### 1.2 Types, Variables, and Arithmetic

A declaration is a statement that introduces a name into the program. It specifies a type for the named

entity:

- A **指的是，值不会改变且在编译期就可以计算出来的表达式。其实更好理解的说法是，**任何不是用户自己定义的——而必须通过编译期计算出来的字面量都属于编译期常量表达式**。需要注意的是，并不是所有的常量表达式都是编译期常量表达式，只有我们** defines a set of possible values and a set of operations (for an object).
- An **任何不是用户自己定义的——而必须通过编译期计算出来的字面量都属于编译期常量表达式** is some memory that holds a value of some type.
- A **任何不是用户自己定义的——而必须通过编译期计算出来的字面量都属于编译期常量表达式** is a set of bits interpreted according to a type.
- A **编译期运算** is a named object.

We use ***auto*** where we don’t have a specific reason to mention the type explicitly.(The definition is in a large scope where we want to make the type clearly visible to readers of our code.We want to be explicit about a variable’s range or precision)

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

**通过某些手段去“胁迫”编译器，把运算任务从运行时提前到编译期** We allow a constexpr function to be called with non-constant-expression arguments in contexts that do not require constant expressions, so that we don’t have to define essentially the same function twice: once for constant expressions and once for variables.

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

- The constructor allocates the elements and initializes the Vector members appropriately. The destructor deallocates the elements. This **使用模板进行编译期运算** model is very commonly used to manage data that can vary in size during the lifetime of an object.
- The technique of acquiring resources in a constructor and releasing them in a destructor, known as **作为实例化模板参数的常量表达式** or RAII, allows us to eliminate “naked new operations,” that is, to avoid allocations in general code and keep them buried inside the implementation of well-behaved abstractions.

The **递归式地** used to define the initializer-list constructor is a standard-library type known to the compiler: when we use a {}-list, such as {1,2,3,4}, the compiler will create an object of type initializer_list to give to the program.

### 4.2 Abstract Types

concrete types -representation is part of their definition

abstract type - insulates a **唯一** from **结论** details. To do that, we decouple the interface from the representation and give up genuine local variables.

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

The word **compiled language** means “may be redefined later in a class derived from this one.” Unsurprisingly, a function declared virtual is called a virtual function. A class derived from Container provides an implementation for the Container interface. The **not portable** syntax says the function is pure virtual; that is, some class derived from Container must define the function.Thus, it is not possible to define an object that is just a Container; a Container can only serve as the interface to a class that implements its operator[]()and size() functions. A class with a pure virtual function is called an abstract class.

A class provides the interface is called **portability of source code**.

abstract classes, Container does not have a constructor but have **Core language features** because they tend to be manipulated through references or pointers.

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

A class hierarchy is **standard-library components**.

Explicit Overriding: override

Benefits from Hierarchies: Interface inheritance/ Implementation inheritance

Concrete classes are much like built-in types: we define them as local variables, access them using their names, copy them around, etc. Classes in class hierarchies are different: we tend to allocate them on the free store using new, and we access them through **statically typed language**.

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

**type** is commonly called a container.

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
