---
dateCreated: 2024-03-16
dateModified: 2025-07-27
---
## Ch2 变量和类型

## 2.3 复合类型

compound type 指基于其他类型定义的类型，引用和指针即属于符合类型。

一条声明语句是一个数据类型和变量名列表组成，更通用地说，是一个 base type + declarator 列表组成。每个声明符定义了一个变量并指定该变量为与基本类型有关的某种类型。

### 1 引用

reference 为变量起另一个名字，引用类型引用 refers to 是另一种类型。通过将声明符写为&d 的形式来定义引用类型，其中 d 是声明的变量名。

```c++
int ival = 1024;
int &refVal = ival;
```

初始化变量时，初始值会拷贝到新建对象中，定义引用时，引用和它的初始值绑定，而不是拷贝，因此引用必须初始化。

引用并非对象，它是已经存在的对象的另一个名字。对其的操作都是在与之绑定的对象上进行的。也不能定义引用的引用。引用一般需要和与之绑定的对象类型绑定。且引用只能绑定对象，不能与字面值绑定。

## 2.4 Const 限定符

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

### 1 Const 引用

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

## Ch3 字符串、向量、数组

## 3.3 标准库类型 Vector

vector 表示对象集合，也称容器。所有对象类型相同，每个元素都有一个

索引，索引用于访问对象。

vector 是一个类模板，编译器根据模板创建类也即实例化。使用模板需要指出实例化为何种类型。vector 能容纳绝大多类型的对象作为元素，但引用不是对象，所以不可以包含引用。

### 定义初始化

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

### 添加元素
### 其他操作

## 表达式、语句、函数

## 类

## II C++ 标准库

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
