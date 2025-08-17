---
dateCreated: 2025-08-16
dateModified: 2025-08-16
---
# Lua 语法整理

### **1. 注释**
- **单行注释**：使用 `--` 开头。

  ```lua
  -- 这是一个单行注释
  ```

- **多行注释**：使用 `--[[ … ]]` 包裹。

  ```lua
  --[[ 
  这是一个多行注释
  可以跨越多行
  ]]
  ```

---

### **2. 变量与数据类型**
- **变量声明**：
  - **局部变量**：使用 `local` 关键字声明（推荐，避免污染全局作用域）。

    ```lua
    local a = 10
    ```

  - **全局变量**：未加 `local` 的变量默认是全局的。

    ```lua
    b = "Hello Lua"
    ```

- **数据类型**（8 种）：
  1. **nil**：空值（未定义或删除变量）。
  2. **boolean**：`true` 或 `false`（注意：Lua 中 `0` 和空字符串是 `true`）。
  3. **number**：数字（默认是双精度浮点数，Lua 5.3+ 支持整数类型）。
  4. **string**：字符串（单引号 `'` 或双引号 `"`，支持 `[[…]]` 多行字符串）。
  5. **table**：唯一的数据结构（用于数组、字典、对象等）。
  6. **function**：函数（一等公民，可作为参数传递或赋值）。
  7. **userdata**：C 扩展数据（由 C/C++ 编写的库使用）。
  8. **thread**：协程（coroutine）。

---

### **3. 运算符**
- **算术运算符**：`+`, `-`, `*`, `/`, `%`, `^`（幂运算）。
- **比较运算符**：`==`, `~=`（不等于）, `<`, `>`, `<=`, `>=`。
- **逻辑运算符**：`and`, `or`, `not`。
- **字符串运算符**：
  - **拼接**：`..`（例：`"Hello" .. " World"`）。
  - **长度**：`#`（例：`#s` 返回字符串或表的长度）。
- **其他运算符**：
  - **表索引**：`[]`（如 `t[key]`）或 `.`（如 `t.key`）。

---

### **4. 流程控制**
- **条件语句**：

  ```lua
  if condition then
    -- 代码块
  elseif condition2 then
    -- 代码块
  else
    -- 代码块
  end
  ```

- **循环语句**：
  - **while 循环**：

    ```lua
    while condition do
      -- 代码块
    end
    ```

  - **for 数值循环**（固定范围）：

    ```lua
    for i = 1, 10 do
      print(i)  -- 输出 1 到 10
    end
    ```

  - **for 泛型循环**（遍历表）：

    ```lua
    for key, value in pairs(t) do
      print(key, value)
    end
    ```

  - **repeat-until 循环**（先执行后判断）：

    ```lua
    repeat
      -- 代码块
    until condition
    ```

---

### **5. 函数**
- **定义函数**：

  ```lua
  function factorial(n)
    if n == 0 then return 1 end
    return n * factorial(n - 1)
  end
  ```

- **匿名函数**：

  ```lua
  local add = function(a, b) return a + b end
  print(add(2, 3))  -- 输出 5
  ```

- **可变参数**：使用 `…` 接收任意数量的参数。

  ```lua
  function sum(…)
    local args = {…}
    local total = 0
    for _, v in ipairs(args) do
      total = total + v
    end
    return total
  end
  print(sum(1, 2, 3))  -- 输出 6
  ```

- **多返回值**：函数可以返回多个值。

  ```lua
  function min_max(t)
    return table.min(t), table.max(t)
  end
  local m, M = min_max({3, 1, 4})
  print(m, M)  -- 输出 1 4
  ```

---

### **6. 表（Table）**
- **创建表**：

  ```lua
  local t1 = {}  -- 空表
  local t2 = {1, 2, 3}  -- 数组（索引从 1 开始）
  local t3 = {key1 = "value1", key2 = "value2"}  -- 字典
  ```

- **表操作**：
  - **访问元素**：`t[key]` 或 `t.key`。

    ```lua
    t = {a = 1, b = 2}
    print(t["a"])  -- 输出 1
    print(t.b)     -- 输出 2
    ```

  - **修改元素**：

    ```lua
    t.new_key = "new_value"
    ```

  - **遍历表**：

    ```lua
    for k, v in pairs(t) do
      print(k, v)
    end
    ```

- **元表（Metatable）与元方法**：
  - 元表是赋予表特殊行为的机制，例如重载运算符。
  - 常用元方法：`__add`（加法）、`__index`（访问不存在的键）、`__newindex`（设置新键）等。

  ```lua
  local mt = {
    __add = function(t1, t2)
      return t1.x + t2.x
    end
  }
  local t1 = {x = 1}
  local t2 = {x = 2}
  setmetatable(t1, mt)
  print(t1 + t2)  -- 输出 3
  ```

---

### **7. 协程（Coroutine）**
- **创建协程**：`coroutine.create(func)`。
- **挂起和恢复**：使用 `coroutine.yield()` 挂起，`coroutine.resume(co)` 恢复。
- **示例**：

  ```lua
  local co = coroutine.create(function()
    print("Start")
    coroutine.yield()  -- 挂起
    print("End")
  end)
  coroutine.resume(co)  -- 输出 Start
  coroutine.resume(co)  -- 输出 End
  ```

---

### **8. 模块与包管理**
- **创建模块**：使用 `module` 或返回一个表。

  ```lua
  -- math_utils.lua
  local M = {}
  function M.add(a, b)
    return a + b
  end
  return M
  ```

- **使用模块**：通过 `require` 加载。

  ```lua
  local math_utils = require("math_utils")
  print(math_utils.add(2, 3))  -- 输出 5
  ```

---

### **9. 错误处理**
- **异常处理**：使用 `pcall`（保护调用）捕获错误。

  ```lua
  local status, result = pcall(function()
    error("An error occurred!")
  end)
  if not status then
    print("Error:", result)  -- 输出 Error: An error occurred!
  end
  ```

---

### **10. 标准库**
- **字符串处理**：`string.len`, `string.sub`, `string.upper`, `string.match`。
- **数学计算**：`math.sin`, `math.random`, `math.floor`。
- **输入输出**：`io.read`, `io.write`, `file:open`, `file:close`。
- **表操作**：`table.insert`, `table.remove`, `table.sort`。

---

# 应用

1. **嵌入式脚本**：Lua 常用于嵌入到 C/C++ 程序中，扩展应用程序功能。
2. **游戏开发**：Lua 是 Unity 和 Roblox 的脚本语言，常用于游戏逻辑。
3. **自动化脚本**：利用 Lua 编写自动化任务脚本（如文件处理、数据转换）。
