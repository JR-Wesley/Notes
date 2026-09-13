---
tags:
  - CS
---

# 现代 Python 3 整理

## 基础语法

> [!note] 基础语法重点
> 基础数据类型的表示 `int`, `float`, `str`, `bool`, `list`, `dict`, `set`, `tuple`、操作。
> 重点：基础**容器数据类型**的用法 `list, tuple, dict, set`。
> 函数、包的基础使用

### 基础数据类型

   - **变量是动态类型，无需声明类型。**
	 - **数字（Number）**：
		- **整型（int）**：用于表示整数，例如 `5`、`-10` 、`100000` 等。Python 3 中的 `int` 类型理论上可以表示无限大的整数，如 `12345678901234567890` 也能正常处理。
		- **浮点型（float）**：用于表示带有小数部分的数值，如 `3.14`、`-2.5` 。特殊的浮点值还有 `float('inf')`（正无穷）、`float('-inf')`（负无穷）和 `float('nan')`（非数字，如 `0 / 0` 运算的结果）。
		- **复数（complex）**：由实部和虚部组成，形式为 `a + bj`，其中 `a` 是实部，`b` 是虚部，例如 `3 + 4j`。
	- **字符串（str）**：用于表示文本数据，是由字符组成的有序序列。可以使用单引号（`'`）、双引号（`"`）或三引号（`'''` 或 `"""`）来创建字符串。如 `'Hello,'`、`"World!"` 、`'''Python'''`。字符串支持多种操作，比如拼接（`'Hello' + ' World'`）、切片（`'Hello'[1:3]` 结果为 `'el'`）、查找（`'Hello'.find('l')` 返回 `2`）等。
	- **布尔值（bool）**：只有两个取值，`True` 和 `False`，用于逻辑判断。例如 `3 > 2` 的结果为 `True`，`3 == 2` 的结果为 `False`。布尔值常和逻辑运算符（`and`、`or`、`not`）一起使用。

### 列表（list）

- **是一种有序、可变的数据集合，可以包含不同类型的元素**
- **创建与初始化**：使用方括号 `[]` 或者 `list()` 函数来创建列表。可以在创建时直接初始化元素，例如 `my_list = [1, 'apple', 3.14]`，也可以创建空列表 `empty_list = []` 或 `empty_list = list()`。
- **元素访问**：通过索引访问元素，索引从 `0` 开始，如 `my_list[0]` 获取第一个元素；也支持负索引，`my_list[-1]` 获取最后一个元素。还能使用切片操作获取子列表，例如 `my_list[1:3:1]` 获取索引为 `1` 和 `2` 的元素组成的新列表，取值间隔步进默认为 1。
- **增删改操作**：
    - **添加元素**：`append(x)` 方法在列表末尾添加单个元素；`extend(l)` 方法用于合并另一个可迭代对象到当前列表；`insert(i, x)` 方法在指定索引位置插入元素。
    - **删除元素**：`pop(i)` 方法根据索引删除并返回对应元素，默认删除最后一个；`remove(x)` 方法根据元素值删除首次出现的该元素；`del` 语句可以删除指定索引或切片的元素。
    - **修改元素**：直接通过索引赋值来修改，如 `my_list[1] = 10`。
- **常用方法与操作**：
    - **排序**：`sort()` 方法对列表进行原地排序；`sorted()` 函数返回排序后的新列表，原列表不变。
    - **反转**：`reverse()` 方法原地反转列表；也可以通过切片 `[::-1]` 获取反转后的新列表。
    - **获取长度**：使用 `len()` 函数获取列表中元素的个数。

### 元组（tuple）

- **有序的数据集合，元组是不可变的**
- **创建与初始化**：使用圆括号 `()` 或者 `tuple()` 函数创建元组，可以在创建时初始化元素，如 `my_tuple = (1, 'apple', 3.14)`，也能创建空元组 `empty_tuple = ()` 或 `empty_tuple = tuple()` ；如果元组只有一个元素，需要在元素后面加逗号，例如 `single_element_tuple = (1,)` ，否则会被当作普通数据类型。
- **元素访问**：和列表类似，通过索引（正索引或负索引）以及切片来访问元组中的元素，如 `my_tuple[1]` ，但元组是不可变的，不能直接修改元素。虽然元组本身不可变，但如果元组中包含可变元素（如列表），可变元素内部是可以修改的。
- **特性**：元组的不可变性使得它在一些场景下更安全，比如作为字典的键（列表不能作为字典键，因为列表可变）；也常被用于函数返回多个值，比如 `def get_info(): return "Alice", 25` ，实际上返回的是一个元组。

### 集合（set）

- **是一个无序且不重复元素的集合**
- **创建与初始化**：使用花括号 `{}` 或者 `set()` 函数创建集合，注意创建空集合只能用 `set()` ，因为 `{}` 表示空字典；创建非空集合时，`my_set = {1, 2, 3}` ，集合中的元素是无序且唯一的。
- **集合操作**：
    - **添加元素**：使用 `add()` 方法添加单个元素，`update()` 方法可以添加多个元素（参数为可迭代对象）。
    - **删除元素**：`remove()` 方法删除指定元素，如果元素不存在会报错；`discard()` 方法删除指定元素，元素不存在时不报错；`pop()` 方法随机删除并返回一个元素；`clear()` 方法清空集合。
    - **集合运算**：支持并集（`|` 或 `union()` 方法）、交集（`&` 或 `intersection()` 方法）、差集（`-` 或 `difference()` 方法）、对称差集（`^` 或 `symmetric_difference()` 方法）等操作。`issubset(other_set)`：判断当前集合是否为另一个集合的子集，`issuperset(other_set)`：判断当前集合是否为另一个集合的超集
- **应用场景**：常用于去重，比如将列表转换为集合再转换回列表，`list(set([1, 2, 2, 3]))` ；也用于判断元素是否存在，集合的查找效率比列表高很多。

### 字典（dict）

- **是一种无序的键值对（key-value）数据结构**
- **创建与初始化**：使用花括号 `{}` 或者 `dict()` 函数创建字典，可以在创建时初始化键值对，如 `my_dict = {'name': 'Alice', 'age': 25}` ，也能创建空字典 `empty_dict = {}` 或 `empty_dict = dict()` ；还可以通过其他方式初始化，例如 `dict([('key1', 'value1'), ('key2', 'value2')])` 。
- **元素访问与修改**：通过键来访问对应的值，如 `my_dict['name']` ；如果键不存在会报错，可使用 `get()` 方法，键不存在时返回默认值（默认为 `None` ）；直接通过键赋值来修改或添加键值对，如 `my_dict['age'] = 26` （修改），`my_dict['city'] = 'Beijing'` （添加）。
- **常用方法与操作**：
    - **删除键值对**：使用 `pop()` 方法根据键删除并返回对应的值；`del` 语句删除指定键的键值对；`popitem()` 方法随机删除并返回一个键值对（Python 3.7+ 中按插入顺序删除）；`clear()` 方法清空字典。
    - **获取键、值、键值对**：`keys()` 方法返回所有键的视图，`values()` 方法返回所有值的视图，`items()` 方法返回所有键值对的视图，这些视图会随着字典的变化而动态更新；可以使用 `list()` 函数将视图转换为列表。

### 基础结构

> 控制语句，需要掌握遍历可迭代对象（列表、字符串、字典、`range` 等）、循环结合 `input` 、范围循环、循环控制语句的使用。

   - **控制结构**
     - 条件语句：`if-elif-else`。
     - 循环语句：`for`、`while`、`for-else`、`while-else`。
   - **函数定义**
     - 使用 `def` 定义函数，定义时必须遵循 `位置参数 → 默认参数 → *args → **kwargs`，支持关键字参数、默认参数。
     - 可返回单个值、多个值（本质是元组）、`None`。
     - **可变参数**：接收任意数量的位置参数，用 `*args` 表示（本质是元组）：
     - **关键字可变参数**：接收任意数量的关键字参数，用 `**kwargs` 表示（本质是字典）。
     - 可赋值给变量、作为参数传递、作为返回值，更多见高级用法。
   - **模块与导入**
     - 使用 `import` 导入模块或包，支持别名。
 - `in` 用于判断某个元素是否**属于**一个容器（如列表、字符串、字典、集合等），返回布尔值 `True` 或 `False`。
 - `is` 用于判断两个变量是否**指向同一个内存地址的对象**（即是否为同一个对象），返回布尔值 `True` 或 `False`。

---

## 面向对象编程（OOP）

> 重点在于**类与对象的设计**、**三大特性（封装、继承、多态）** 及 Python 特有的实现方式

### 基础

   - **类与对象**
	   - **类（Class）**：是对一类事物的抽象模板，定义了该类对象共有的属性（数据）和方法（行为）。
	   - **对象（Object）**：是类的具体实例，通过类创建。

		```python
		class Person:  # 类名通常用 PascalCase（首字母大写）
			# 类属性（所有实例共享）
			species = "Homo sapiens"
			
			# 构造方法：初始化实例属性
			def __init__(self, name, age):  # self 代表实例本身
				self.name = name  # 实例属性（每个对象独有的数据）
				self.age = age
			
			# 实例方法：操作实例属性的函数
			def greet(self):
				return f"Hello, I'm {self.name}"
		```

- **封装实现**：不同于其他语言的 `public` / `private` 关键字，Python 通过**命名约定**实现封装：
	- 公有属性 / 方法：默认命名（如 `name`、`greet()`），外部可直接访问。
	- 受保护属性 / 方法：前缀单下划线 `_`（如 `_age`），表示 “建议私有”，外部可访问但不推荐（仅为约定，无强制限制）。
	- 私有属性 / 方法：前缀双下划线 `__`（如 `__salary`），Python 会对其进行 “名字修饰”（实际存储为 `_类名__属性名`），阻止外部直接访问（伪私有，仍可通过修饰后的名字访问，但不建议）。
- **继承与多态**
     - 支持单继承和多继承，使用 `super()` 调用父类方法。
     - 多态指**不同类的对象对同一方法调用可产生不同行为**，实现 “接口复用”。
     - 示例：

	```python
	class Animal:
	   def speak(self):
		   print("Animal sound")
	
	class Dog(Animal):
	   def speak(self):
		   print("Woof!")
	
	dog = Dog()
	dog.speak()  # 输出: Woof!
	```

- **魔术方法**
	- 以双下划线 `__` 开头和结尾的方法（如 `__init__`）是特殊方法，用于定义类的 “内置行为”（如打印、运算、比较等），让类的实例支持 Python 内置操作。
	- 常用特殊方法：
		- `__init__`：构造方法（初始化实例）。
		- `__str__`：定义 `print(obj)` 或 `str(obj)` 时的字符串表示（可读性优先）。
		- `__repr__`：定义 `repr(obj)` 时的字符串表示（调试优先，应尽量完整）。
		- `__add__`：定义 `obj1 + obj2` 时的行为。
		- `__len__`：定义 `len(obj)` 时的返回值。

### 类方法（@classmethod）

绑定到类本身，第一个参数为 `cls`（代表类），用于操作类属性（所有实例共享的数据）。

```python
class Person:
	count = 0  # 类属性：记录实例数量
	
	def __init__(self, name):
		self.name = name
		Person.count += 1  # 每次实例化，计数+1
	
	@classmethod
	def get_count(cls):  # 类方法：访问类属性
		return f"Total instances: {cls.count}"

p1 = Person("Alice")
p2 = Person("Bob")
print(Person.get_count())  # 调用类方法 → "Total instances: 2"
```

### 静态方法（@staticmethod）

不绑定到类或实例，无默认参数（既没有 `self` 也没有 `cls`），更像 “类的命名空间中的普通函数”，用于实现与类相关但不依赖类 / 实例属性的功能。

```python
class MathUtils:
	@staticmethod
	def add(a, b):  # 静态方法：与类/实例属性无关
		return a + b

print(MathUtils.add(2, 3))  # 直接通过类调用 → 5
```

### 抽象类（abc 模块）

抽象类是**包含抽象方法（未实现的方法）的类**，不能被实例化，只能作为父类被继承，强制子类实现抽象方法（定义接口规范）。

- 实现方式：通过 `abc` 模块的 `ABC` 类和 `abstractmethod` 装饰器。

    ```python
    from abc import ABC, abstractmethod
    
    class Shape(ABC):  # 抽象类（继承 ABC）
        @abstractmethod  # 抽象方法（必须被子类实现）
        def area(self):
            pass  # 不实现具体逻辑
    
    class Circle(Shape):
        def __init__(self, radius):
            self.radius = radius
        
        def area(self):  # 必须实现父类的抽象方法
            return 3.14 * self.radius **2
    
    # s = Shape()  # 报错：抽象类不能实例化
    c = Circle(2)
    print(c.area())  # 正确：实现了抽象方法 → 12.56
    ```

---

## 高效访问操作

Python 中还有许多高效访问可迭代对象（如列表、元组、集合、字典、生成器等）的方法，这些方法通常基于内置函数、迭代工具或语法特性，能大幅提升处理效率和代码简洁性。

> [!note] 高效访问可迭代对象的核心思路
> 1. 优先使用**内置函数**（`enumerate`、`zip` 等）和**标准库工具**（`itertools`），利用底层优化提升效率；
> 2. 对序列类型（列表、字符串等）善用**切片**批量操作；
> 3. 用**推导式**简化 “访问 + 处理” 逻辑，用**惰性迭代**处理大数据或无限流，减少内存占用。

### 1. 基于内置函数的高效访问

1.**`enumerate()`**：同时获取元素的**索引和值**，避免手动维护索引变量

```python
fruits = ["apple", "banana", "cherry"]
for idx, fruit in enumerate(fruits, start=1):  # start指定起始索引（默认0）
    print(f"第{idx}个水果：{fruit}")
# 输出：第1个水果：apple；第2个水果：banana；第3个水果：cherry
```

2.**`zip()`**：并行遍历多个可迭代对象，将对应位置元素打包为元组

```python
names = ["Alice", "Bob"]
ages = [25, 30]
for name, age in zip(names, ages):  # 长度不一致时，以最短的为准
    print(f"{name} is {age} years old")
# 输出：Alice is 25 years old；Bob is 30 years old
```

- 扩展：`itertools.zip_longest()` 可处理长度不一致的情况（用默认值填充）。

3.**`map()`**：对可迭代对象的每个元素应用函数，返回迭代器（惰性计算）

```python
numbers = [1, 2, 3]
squared = map(lambda x: x** 2, numbers)  # 计算每个元素的平方
print(list(squared))  # [1, 4, 9]（转为列表查看结果）
```

1. **`filter()`**：筛选出满足条件（函数返回 `True`）的元素，返回迭代器

```python
numbers = [1, 2, 3, 4, 5]
evens = filter(lambda x: x % 2 == 0, numbers)  # 筛选偶数
print(list(evens))  # [2, 4]
```

1. **`sum()`、`max()`、`min()`**：直接对可迭代对象进行聚合计算

```python
numbers = [3, 1, 4, 1, 5]
print(sum(numbers))  # 14（求和）
print(max(numbers))  # 5（最大值）
print(min(numbers))  # 1（最小值）
```

### 2. 基于切片的高效访问（序列类型专属，如列表、元组、字符串）

切片通过 `[start:end:step]` 语法实现**批量访问、反转、跳步**等操作，返回新对象（不修改原对象），效率远高于循环逐个访问。

```python
lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 1. 访问子序列（左闭右开）
print(lst[2:5])  # [2, 3, 4]（索引2到4的元素）

# 2. 跳步访问（间隔step个元素）
print(lst[::2])  # [0, 2, 4, 6, 8]（每隔1个取一个）

# 3. 反转序列
print(lst[::-1])  # [9, 8, 7, …, 0]

# 4. 从末尾开始访问
print(lst[-3:])  # [7, 8, 9]（最后3个元素）
```

### 3. 基于迭代工具库 `itertools` 的高级访问

`itertools` 模块提供了大量高效处理可迭代对象的工具，尤其适合大数据流或复杂迭代场景：

1. **`itertools.islice()`**：对可迭代对象进行切片（支持生成器等不可直接切片的对象）

    ```python

from itertools import islice

generator = (x for x in range(100)) # 生成器（不可直接切片）

sliced = islice(generator, 5, 10) # 取索引 5 到 9 的元素

print(list(sliced)) # [5, 6, 7, 8, 9]

```

2. **`itertools.chain()`**：将多个可迭代对象 “串联” 为一个迭代器

```python
from itertools import chain
list1 = [1, 2]
list2 = [3, 4]
combined = chain(list1, list2)  # 等价于 list1 + list2，但更高效（惰性）
print(list(combined))  # [1, 2, 3, 4]
```

1. **`itertools.groupby()`**：按指定键对元素分组（需先排序相同键的元素）

```python
from itertools import groupby
words = ["apple", "banana", "ant", "ball", "cat"]
words_sorted = sorted(words, key=lambda x: x[0])  # 按首字母排序
for key, group in groupby(words_sorted, key=lambda x: x[0]):
	print(f"首字母 {key}: {list(group)}")
# 输出：首字母 a: ['apple', 'ant']；首字母 b: ['banana', 'ball']；首字母 c: ['cat']
```

1. **`itertools.product()`**：生成多个可迭代对象的 “笛卡尔积”（类似嵌套循环）

```python
from itertools import product
colors = ["红", "蓝"]
sizes = ["S", "M"]
for color, size in product(colors, sizes):
	print(f"{color}{size}")  # 红S；红M；蓝S；蓝M
```

### 4. 基于推导式的高效访问与转换

推导式（列表、字典、集合推导式）通过简洁语法实现 “访问 + 处理 + 生成新对象” 的一站式操作，效率高于循环（底层优化）。

1. **列表推导式**：`[表达式 for 元素 in 可迭代对象 if 条件]`

```python
numbers = [1, 2, 3, 4, 5]
# 访问并筛选偶数，同时计算平方
even_squares = [x** 2 for x in numbers if x % 2 == 0]
print(even_squares)  # [4, 16]
```

1. **字典推导式**：`{键表达式: 值表达式 for 元素 in 可迭代对象 if 条件}`

```python
names = ["Alice", "Bob"]
# 访问列表元素，生成“名字: 长度”的字典
name_lengths = {name: len(name) for name in names}
print(name_lengths)  # {'Alice': 5, 'Bob': 3}
```

1. **生成器表达式**：`(表达式 for 元素 in 可迭代对象 if 条件)`（惰性计算，节省内存）

```python
numbers = [1, 2, 3, 4, 5]
even_generator = (x for x in numbers if x % 2 == 0)  # 不立即计算，迭代时才生成
print(next(even_generator))  # 2（逐个获取）
print(list(even_generator))  # [4]（剩余元素）
```

### 5. 基于 “惰性迭代” 的高效访问

对于大型或无限可迭代对象（如生成器、文件流），使用**惰性迭代**（每次只生成 / 处理一个元素）可避免一次性加载所有数据到内存，大幅提升效率：

- 生成器表达式
- `map()`、`filter()`、`itertools` 工具返回的迭代器
- 文件对象的迭代（逐行读取）：

---

## 函数高级功能

### 嵌套函数（nested function）

嵌套函数的核心价值是**将函数的作用域限制在外部函数内部**，避免全局命名空间污染，同时可以访问外部函数的变量和参数。这种用法主要用于封装逻辑、实现闭包、简化代码结构等。常见用法包括：

#### 1. 封装内部辅助逻辑

当某个函数只需要在外部函数内部使用（作为 “工具函数”）时，可将其定义为嵌套函数，避免暴露在全局作用域中。

```python
def calculate_statistics(data):
    # 嵌套函数：仅用于内部计算平均值
    def mean():
        return sum(data) / len(data) if data else 0
    
    # 嵌套函数：仅用于内部计算标准差
    def std_dev():
        m = mean()
        return (sum((x - m)**2 for x in data) / len(data))** 0.5 if data else 0
    
    # 外部函数使用内部函数
    return {
        "mean": mean(),
        "std_dev": std_dev()
    }

stats = calculate_statistics([1, 2, 3, 4, 5])
print(stats)  # 输出: {'mean': 3.0, 'std_dev': 1.4142…}
```

这里 `mean()` 和 `std_dev()` 仅为 `calculate_statistics()` 服务，无需暴露给外部，增强了代码的封装性。

#### 2. 实现闭包（Closure）

嵌套函数可以 “捕获” 外部函数的变量（即使外部函数执行完毕，这些变量仍能被嵌套函数访问），这种机制称为**闭包**。闭包常用于保留状态或实现 “函数工厂”（动态生成函数）。

- 嵌套函数中，内层函数引用外层函数的变量，且外层函数返回内层函数。
- 作用：封装私有变量、实现数据持久化（变量不会随外层函数结束而销毁）：

示例：实现一个计数器（保留计数状态）

```python
def make_counter():
    count = 0  # 外部函数的变量，被内部函数捕获
    
    def counter():
        nonlocal count  # 声明修改外部函数的变量
        count += 1
        return count
    
    return counter  # 返回嵌套函数

# 创建两个独立的计数器（状态互不干扰）
counter1 = make_counter()
counter2 = make_counter()

print(counter1())  # 1
print(counter1())  # 2
print(counter2())  # 1（counter2的状态独立）
```

这里 `counter()` 捕获了 `make_counter()` 中的 `count` 变量，每次调用 `counter1()` 或 `counter2()` 时，会分别维护自己的 `count` 状态。

#### 3. 实现装饰器（Decorator）

装饰器是 Python 的高级特性，本质是 “接收函数并返回新函数的函数”，其核心实现依赖嵌套函数。装饰器用于给函数动态添加功能（如日志、计时、权限校验等）。

- 用于在不修改原函数代码的前提下，为函数添加额外功能（如日志、权限验证、缓存）。
- 本质是 “函数嵌套 + 闭包”，语法用 `@装饰器名` 放在函数定义前：

示例：实现一个计时装饰器

```python
import time

def timer(func):  # 外部函数：接收被装饰的函数
    def wrapper(*args, **kwargs):  # 嵌套函数：包装原函数
        start = time.time()
        result = func(*args, **kwargs)  # 调用原函数
        end = time.time()
        print(f"{func.__name__} 耗时: {end - start:.2f}秒")
        return result  # 返回原函数结果
    
    return wrapper  # 返回嵌套函数（包装后的函数）

# 使用装饰器
@timer
def slow_function():
    time.sleep(1)  # 模拟耗时操作

slow_function()  # 输出: slow_function 耗时: 1.00秒
```  

这里 `wrapper()` 是嵌套函数，它捕获了外部函数 `timer()` 的参数 `func`，并在调用 `func` 前后添加了计时逻辑。

进阶：带参数的装饰器（需额外一层函数嵌套）、类装饰器等。

#### 4. 限制变量作用域

嵌套函数只能在外部函数内部被调用（除非被返回），其内部定义的变量不会污染全局或外部作用域，避免命名冲突。

```python
def outer():
    x = 10  # 外部函数的局部变量
    
    def inner():
        y = 20  # 嵌套函数的局部变量
        print(x + y)  # 可访问外部函数的x
    
    inner()  # 内部调用嵌套函数

outer()  # 输出: 30
# inner()  # 报错：全局作用域中无inner
# print(y)  # 报错：全局作用域中无y
```

#### 嵌套函数的作用域规则

1. 嵌套函数可以访问外部函数的变量和参数，但默认不能修改（需用 `nonlocal` 声明）。
2. 若嵌套函数定义了与外部函数同名的变量，则内部变量会 “遮蔽” 外部变量（局部优先）。
3. 嵌套函数无法直接访问全局变量，除非用 `global` 声明。

### 匿名函数（`lambda`）

- 用 `lambda` 定义简单的单行函数，语法：`lambda 参数: 表达式`（返回表达式结果）。
- 适用于临时使用的简单逻辑，常配合 `map`、`sorted` 等函数：

    ```python
    add = lambda a, b: a + b  # 等价于def add(a,b): return a+b
    sorted([(2, 3), (1, 4)], key=lambda x: x[1])  # 按元组第二个元素排序
    ```

### 生成器函数（`yield`）

- 用 `yield` 关键字返回值，函数调用时返回生成器（可迭代对象），具有 “惰性计算” 特性（按需生成值，节省内存）。
- 适用于处理大数据流或无限序列：

    ```python
    def fibonacci(n):
        a, b = 0, 1
        for _ in range(n):
            yield a  # 每次调用next()时返回a，并暂停
            a, b = b, a + b
    
    for num in fibonacci(5):  # 生成0, 1, 1, 2, 3
        print(num)
    ```

### 函数式编程工具

- **`map(func, iterable)`**：用 `func` 处理可迭代对象的每个元素，返回迭代器：

```python
list(map(lambda x: x*2, [1, 2, 3]))  # [2, 4, 6]
```

- **`filter(func, iterable)`**：筛选出 `func` 返回 `True` 的元素：

```python
list(filter(lambda x: x % 2 == 0, [1, 2, 3, 4]))  # [2, 4]
```

- **`functools.reduce(func, iterable)`**：累积处理元素（从左到右）：

```python
from functools import reduce
reduce(lambda a, b: a + b, [1, 2, 3, 4])  # 10（等价于((1+2)+3)+4）

```

---

## 并发编程

> 这里不包括并行计算框架或张量编程框架，使用 python 本身调用线程或进程。

### **1. 多线程（Threading）**

- **适用场景**：I/O 密集型任务（如网络请求、文件读写），因为线程在等待 I/O 时会释放 GIL。
- **注意事项**：
    - Python 的 GIL 限制了多线程在 CPU 密集型任务中的性能。
    - 使用 `threading` 模块或 `concurrent.futures.ThreadPoolExecutor`。

### **2. 多进程（Multiprocessing）**

- **适用场景**：CPU 密集型任务（如数值计算、图像处理），每个进程有独立的 GIL，可绕过 GIL 限制。
- **注意事项**：
    - 进程间通信（IPC）开销较高，需避免频繁共享数据。
    - 使用 `multiprocessing` 模块或 `concurrent.futures.ProcessPoolExecutor`。

### **3. 协程（Async/Await）**

- **适用场景**：I/O 密集型任务（如异步网络请求、事件驱动），通过单线程切换任务减少上下文切换开销。
- **注意事项**：
    - 使用 `asyncio` 库，协程需配合异步 I/O 操作（如 `aiohttp`、`asyncpg`）。

### **4. 线程池与进程池（concurrent.futures）**

- **适用场景**：简化并发任务管理，复用线程/进程资源。
- **优势**：
    - 提供统一的 API（`map`、`submit`），支持任务超时、取消等操作。

### **5. 绕过 GIL 的 C 扩展**

- **适用场景**：CPU 密集型任务需要极致性能，或需调用外部库。
- **方法**：
    - **Cython**：将 Python 代码编译为 C 扩展，支持类型注解。
    - **Numba**：即时编译（JIT）数值计算代码。
    - **C/C++ 扩展**：通过 `ctypes`、`cffi` 或 `CPython API` 调用原生代码。

---

## 常用功能

   - **异常处理**：使用 `try-except-finally` 捕获和处理异常，避免 “裸 `except`”（`except:`），明确捕获特定异常（如 `except ValueError`），防止掩盖未知错误。

```python
try:
	result = 10 / 0
except ZeroDivisionError:
	print("不能除以零")
finally:
	print("清理资源")
```

   - **元编程**：使用 `type()` 或元类（Metaclass）动态创建类。

```python
class Meta(type):
def __new__(cls, name, bases, attrs):
attrs['added_by_meta'] = True
return super().__new__(cls, name, bases, attrs)

class MyClass(metaclass=Meta):
pass
```

   - **上下文管理器**：使用 `with` 自动管理资源（文件、锁等）。

```python
with open("file.txt", "r") as f:
   data = f.read()
```

   - **类型注解**：使用 `:` 和 `->` 指定变量/函数返回值类型。

```python
def add(a: int, b: int) -> int:
   return a + b
```

   - **海象运算符（:=，3.8+）**：在表达式中赋值，简化条件判断中的赋值：

```python
if (n := len(data)) > 10:
   print(f"Data is too long ({n} elements)")
```

   - **f-string 格式化（3.6+）**

```python
name = "Alice"
print(f"Hello, {name}!")
```

   - **解包操作符**

```python
nums = [1, 2, 3]
print(*nums)  # 输出: 1 2 3
```

- ****`pytest` 替代 `unittest`**：`pytest` 支持函数式测试（无需继承 `TestCase`）、参数化测试（`@pytest.mark.parametrize`）、 fixture（测试资源复用），比 `unittest` 更灵活。

---

## 常用包

文件与 IO 操作：

-  `pathlib`（Python 3.4+），用 `pathlib`（3.4+）替代 `os.path` 处理路径，更面向对象：
- `os` 与 `sys`：系统交互
- `json`：JSON 数据处理：内置的 JSON 序列化 / 反序列化工具，支持 Python 基本类型（`dict`/`list`/`str` 等）与 JSON 格式的转换。
-  `csv`：CSV 文件处理：用于读写逗号分隔值（CSV）文件，支持自定义分隔符、引号规则等，适合处理表格数据。

函数式编程：

-  `functools`：函数增强工具
	- **`lru_cache`**：缓存函数结果，优化重复计算（如递归、高频调用）。
	- **`partial`**：固定函数部分参数，生成新函数（偏函数）。
-  `itertools`：高效迭代工具。提供生成迭代器的函数，用于高效循环（内存友好，适合大数据流）。
	- `chain`：拼接多个迭代器。
	- `islice`：切片迭代器（无需生成完整列表）。
	- `product`：计算笛卡尔积（如嵌套循环的替代）。

数据结构：

-  `collections`：提供扩展数据结构，弥补内置类型的不足。
	- **`defaultdict`**：避免 `KeyError`，为不存在的键自动生成默认值（如空列表、0 等）。
	- **`deque`**：双端队列，支持 O (1) 时间复杂度的头部 / 尾部插入 / 删除（比 `list` 的 `appendleft` 高效），适合 “滑动窗口”“队列 / 栈” 场景。
	- **`namedtuple`**：带字段名的元组，兼具 `tuple` 的不可变性和类的可读性，适合存储简单数据（如坐标、点）。
	- `Counter`：计数工具
-  `dataclasses`：数据类（3.7+）。自动为 “数据存储类” 生成 `__init__`/`__repr__`/`__eq__` 等魔术方法，替代手动编写冗余代码（详见前文 “OOP 强化” 部分）。

网络：

-  `urllib`：HTTP 客户端与 URL 处理。
-  `socket`：底层网络编程。提供 TCP/UDP 套接字接口，用于实现自定义网络协议（如简易服务器 / 客户端）。

其他常用包：

- `typing`：类型提示工具。Python 3.5 + 引入，用于为变量、函数参数和返回值添加**类型标注**，配合静态检查工具（如 `mypy`）提升代码可读性和健壮性。
- `random`：生成随机数（`random.randint(1,10)` 生成 1-10 随机整数）
- `logging`：日志记录（比 `print` 更灵活，支持分级、输出到文件）
- `argparse`：解析命令行参数（快速构建命令行工具）
- **`re`**：正则表达式，用于字符串匹配、提取、替换（如验证邮箱、手机号）。
- **`math`/`statistics`**：数学与统计工具，`math` 提供三角函数、对数等，`statistics` 提供均值、中位数等统计量。
- **`random`**：随机数生成，如 `random.randint(1,10)` 生成 1-10 的随机整数，`random.shuffle` 打乱列表。
- **`hashlib`**：加密哈希算法，如 `md5`、`sha256`（用于密码加密、文件校验）。
- **`enum`**：枚举类型（3.4+），定义命名常量集合（如订单状态、颜色）。

其他包：

- 数据处理：
	- **`pandas`**：提供 `DataFrame` 数据结构，支持清洗、转换、聚合等操作。现代版本（2.0+）引入了 Apache Arrow 作为后端，大幅提升了 IO 性能和内存效率。
	- **`polars`**：用 Rust 实现的高性能数据处理库，API 设计类似 `pandas`，但并行计算能力更强（比 `pandas` 快 10-100 倍），适合处理 GB 级甚至 TB 级数据。无全局锁、自动利用多核 CPU，内存占用更低。替代 `pandas` 处理大型数据集，尤其是需要高性能的 ETL 流程。
	- **`numpy`**：数值计算基础库，提供多维数组（`ndarray`）和线性代数、傅里叶变换等底层工具。是 `pandas`、`scikit-learn` 等库的依赖基础。
- 可视化：
	- **`matplotlib`**：最基础的可视化库，支持线图、柱状图、散点图等，可高度定制样式。缺点是 API 较繁琐，适合静态图。
	- **`seaborn`**：基于 `matplotlib` 的高级封装，内置美观的主题和统计图表（如热力图、小提琴图），一行代码即可生成复杂可视化。
- 机器学习
	- **`scikit-learn`**：机器学习入门必备库，包含分类、回归、聚类、降维等算法，以及数据预处理（标准化、编码）、模型评估工具。API 统一（`fit`/`predict`），适合快速验证想法。
	- **`xgboost`/`lightgbm`**：高性能梯度提升树库，在 Kaggle 竞赛和工业界广泛使用，支持并行训练和缺失值处理，精度和速度优于传统决策树。
	-  **`PyTorch`**：Facebook 开发的动态图深度学习框架，灵活性高，调试方便，适合研究和快速迭代。支持自动微分、GPU 加速，生态丰富（如 `torchvision`、`torchaudio` 处理多模态数据）。
	- **`Hugging Face Transformers`**：NLP 领域的 “基础设施”，提供预训练模型（如 BERT、GPT、LLaMA）的统一接口，支持文本分类、翻译、生成等任务，一行代码即可调用大模型。

## 代码与项目管理

代码质量自动化：工具链前置拦截

通过**预提交钩子（pre-commit）** 在代码提交前自动运行检查，提前发现问题：

```yaml
# .pre-commit-config.yaml 核心工具示例
repos:
  - repo: https://github.com/psf/black          # 代码格式化（强制统一风格）
    rev: 24.4.2
    hooks: [ { id: black } ]
  - repo: https://github.com/astral-sh/ruff     # Lint（检查未使用变量、语法错误等）
    rev: v0.4.10
    hooks: [ { id: ruff, args: ["--fix"] } ]    # 自动修复可修复问题
  - repo: https://github.com/pre-commit/mirrors-mypy  # 静态类型检查
    rev: v1.10.0
    hooks: [ { id: mypy } ]
```  

- 核心工具作用：
  - `black`：消除格式争论（如缩进、换行），强制统一风格；
  - `ruff`：替代 `flake8`/`isort`，快速检查并自动修复导入排序、变量未使用等问题；
  - `mypy`/`pyright`：基于类型提示检查类型错误（如传字符串给需整数的函数）。

### 编程范式：多范式融合与现代特性

现代 Python 不再局限于单一范式，而是根据场景灵活组合，同时充分利用新版本语法特性提升效率。

#### 1. 多范式融合：按需选择

- **面向对象（OOP）**：用于复杂状态管理（如业务模型），优先“组合优于继承”，用 `Protocol` 定义接口（显式约束），用 `dataclasses`/`pydantic` 简化数据类（减少样板代码）。
- **函数式编程**：用于无状态逻辑（如数据转换、工具函数），用 `functools.lru_cache` 缓存结果，`itertools` 处理迭代器，避免副作用（如修改入参）。
- **异步编程**：IO 密集型场景（如 API 调用、数据库操作）必用 `asyncio`+`async/await`，配合异步库（`httpx`、`asyncpg`）提升吞吐量，避免阻塞。
- **数据导向**：核心业务数据用 `pydantic` 模型定义（自动验证 + 类型提示），替代松散的 `dict`，减少运行时错误。

#### 2. 类型驱动开发：显式标注提升可维护性

- **类型提示全覆盖**：函数参数、返回值、类属性、变量必须添加类型标注（3.9+ 原生泛型 `list[int]`，3.10+ 联合类型 `int | str`），例如：

  ```python
  from typing import List, Optional

  def filter_even(numbers: List[int]) -> List[int]:
      """过滤列表中的偶数"""
      return [n for n in numbers if n % 2 == 0]
  ```  

- **工具配合**：IDE（VS Code+Pyright、PyCharm）利用类型提示提供自动补全和实时错误提示；`mypy` 在 CI 中强制检查，避免类型不匹配（如“传 `None` 给非可选参数”）。

#### 3. 避免反模式：现代 Python 的“禁忌”

- 禁用旧式语法：如 `print "hello"`（Python 2）、`xrange`（改用 `range`）、`typing.List`（改用 `list`）。
- 减少动态特性滥用：避免 `eval`/`exec`（安全风险 + 可读性差）、`__getattr__`（调试困难），除非必要。
- 慎用继承：优先通过“依赖注入”组合组件，而非多层继承（避免“继承地狱”）。

### 书写规范：可读性优先，工具保障

基于 PEP 8 但更强调“自动化落地”，核心是“让代码读起来像自然语言”。

#### 1. 格式规范：工具强制统一

- **缩进与换行**：4 个空格缩进（不用 Tab），单行长度不超过 88 字符（`black` 默认），长表达式用括号换行：

  ```python
  # 推荐
  result = (
      very_long_function_name(
          arg1, arg2,
          arg3, arg4
      )
      + another_long_value
  )
  ```  

- **导入排序**：按“标准库→第三方库→本地库”分组，组内按字母排序（`isort`/`ruff` 自动处理）：

  ```python
  # 推荐
  import os  # 标准库
  import sys

  import requests  # 第三方库
  from pydantic import BaseModel

  from myproject import utils  # 本地库
  ```  

#### 2. 命名与注释：清晰无歧义

- **命名风格**：
  - 变量/函数/方法：蛇形小写（`user_name`、`calculate_total`）；
  - 类/协议：帕斯卡式（`UserModel`、`Serializable`）；
  - 常量：全大写蛇形（`MAX_RETRY`、`API_URL`）；
  - 避免单字母命名（除 `i`/`j` 循环变量、`x`/`y` 坐标等约定场景）。
- **注释原则**：“解释为什么，而非是什么”（代码本身应清晰表达“是什么”）：

  ```python
  # 不推荐（冗余：代码已说明是计算和）
  def add(a: int, b: int) -> int:
      # 计算a和b的和
      return a + b

  # 推荐（解释特殊逻辑）
  def calculate_discount(price: float, user_level: int) -> float:
      # VIP用户额外9折（业务规则说明）
      discount = 0.9 if user_level >= 3 else 1.0
      return price * discount
  ```  

- **文档字符串**：公共函数/类必须有文档字符串，推荐 Google 风格（清晰易读）：

  ```python
  def fetch_data(url: str, timeout: int = 10) -> dict:
      """从URL获取JSON数据并解析。

      Args:
          url: 数据请求地址
          timeout: 超时时间（秒），默认10秒

      Returns:
          解析后的JSON字典

      Raises:
          requests.Timeout: 超时异常
          ValueError: 响应不是JSON格式
      """
      ...
  ```  

#### 3. 代码简洁性：利用现代语法糖

- **f-string 替代传统格式化**：`f"Hello, {name}!"` 替代 `"%s" % name` 或 `str.format`。
- **海象运算符（:=）简化条件判断**：

  ```python
  # 推荐
  if (n := len(items)) > 10:
      print(f"Too many items: {n}")

  # 替代（重复计算len(items)）
  n = len(items)
  if n > 10:
      print(f"Too many items: {n}")
  ```  

- **路径处理用 `pathlib`**：`Path("data") / "file.txt"` 替代 `os.path.join`。
- **上下文管理器管理资源**：`with open(...)` 自动关闭文件，`async with` 管理异步资源（如 HTTP 会话）。

### 项目管理：结构化与可扩展性

大型项目的核心是“让新人能快速上手”，通过标准化结构、自动化流程和完善文档实现。

采用 `src` 布局（避免导入混乱），典型结构：

```
myproject/
├── pyproject.toml          # 依赖管理（poetry/uv配置）
├── README.md               # 项目说明（安装、启动、核心功能）
├── src/                    # 源代码根目录
│   └── myproject/          # 项目包
│       ├── __init__.py     # 包标识（可空）
│       ├── api/            # API层（如FastAPI路由）
│       ├── core/           # 核心逻辑（业务模型、配置）
│       │   ├── models.py   # Pydantic/Dataclass模型
│       │   └── config.py   # 配置管理
│       ├── db/             # 数据库层（ORM、迁移）
│       └── utils/          # 工具函数
├── tests/                  # 测试目录（与src结构对应）
│   ├── unit/               # 单元测试
│   └── integration/        # 集成测试
├── docs/                   # 文档（API文档、设计说明）
└── .github/workflows/      # CI/CD配置（如GitHub Actions）
```  

 测试策略：自动化覆盖

- **测试框架**：用 `pytest` 替代 `unittest`，通过 `fixture` 复用测试资源（如数据库连接），`parametrize` 实现参数化测试。
- **覆盖率要求**：核心业务逻辑测试覆盖率≥80%（用 `pytest-cov` 监控），CI 中设置最低覆盖率门槛（未达标则阻断合并）。
- **测试分层**：单元测试（测独立函数/类）→ 集成测试（测模块间交互）→ E2E 测试（模拟用户流程，如用 `playwright`）。

## 实现数据结构

### 1. 数组（Array）

Python 的内置 `list` 本质上是动态数组，支持随机访问、动态扩容，可直接作为数组使用：

### 2. 栈（Stack）

栈遵循 " 后进先出（LIFO）"，可用 `list` 实现（append 添加到尾部，pop 从尾部删除）：

```python
class Stack:
    def __init__(self):
        self.stack = []
    
    def push(self, item):  # 入栈
        self.stack.append(item)
    
    def pop(self):  # 出栈
        if not self.is_empty():
            return self.stack.pop()
        raise IndexError("栈为空")
    
    def peek(self):  # 查看栈顶
        if not self.is_empty():
            return self.stack[-1]
        raise IndexError("栈为空")
    
    def is_empty(self):
        return len(self.stack) == 0
```

### 3. 队列（Queue）

队列遵循 " 先进先出（FIFO）"，用 `collections.deque` 效率更高（避免 `list.pop(0)` 的 O (n) 开销）：

### 4. 链表（Linked List）

Python 无内置链表，需自定义节点类实现，常见有单链表、双链表：

### 5. 二叉树（Binary Tree）

通过节点类实现，每个节点包含左子树、右子树和值：

### 6. 哈希表（Hash Table）

Python 的 `dict` 本质是哈希表，支持 O (1) 平均复杂度的插入、查找、删除：

```python
hash_map = {"name": "Alice", "age": 20}
hash_map["gender"] = "female"  # 插入
print(hash_map["name"])  # 查找，输出: Alice
del hash_map["age"]  # 删除
```

如需自定义哈希逻辑，可重写类的 `__hash__` 和 `__eq__` 方法。

### 7. 堆（Heap）

利用 `heapq` 模块实现最小堆（默认），最大堆可通过取负值实现：

```python
import heapq

# 最小堆
min_heap = []
heapq.heappush(min_heap, 3)
heapq.heappush(min_heap, 1)
heapq.heappush(min_heap, 2)
print(heapq.heappop(min_heap))  # 输出: 1（弹出最小值）

# 最大堆（存入负值）
max_heap = []
heapq.heappush(max_heap, -3)
heapq.heappush(max_heap, -1)
heapq.heappush(max_heap, -2)
print(-heapq.heappop(max_heap))  # 输出: 3（弹出最大值）
```

### 8. 图（Graph）

常用邻接表（字典 + 列表）或邻接矩阵（二维列表）表示，邻接表更省空间：

```python
# 邻接表表示无向图
class Graph:
    def __init__(self):
        self.adj = {}  # key: 节点，value: 相邻节点列表
    
    def add_edge(self, u, v):  # 添加边
        if u not in self.adj:
            self.adj[u] = []
        if v not in self.adj:
            self.adj[v] = []
        self.adj[u].append(v)
        self.adj[v].append(u)  # 无向图双向添加
    
    def bfs(self, start):  # 广度优先遍历
        visited = set()
        queue = [start]
        visited.add(start)
        while queue:
            node = queue.pop(0)
            print(node, end=" ")
            for neighbor in self.adj.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

# 示例
graph = Graph()
graph.add_edge(0, 1)
graph.add_edge(0, 2)
graph.add_edge(1, 3)
graph.bfs(0)  # 输出: 0 1 2 3
```

# 参考资料

https://huccihuang.github.io/

https://python-cookbook.readthedocs.io/zh-cn/latest/

fluent python
