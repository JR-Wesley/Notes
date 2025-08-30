---
tags:
  - Tool
---


# Shell

| 语法 | 作用 |

| ------------ | --------------------------------------------- |

| `$val` | 引用变量 `val` 的值。 |

| `${val}` | 与 `$val` 相同，但明确界定变量名边界（用于避免歧义，如 `${val}123`）。 |

| `$(command)` | 执行命令并将结果替换到当前位置（命令替换）。 |

| 赋值加引号 | 仅在值包含空格或特殊字符时需要。 |

## **1. `date +%s` 的含义**

`date` 是 Linux/macOS 系统中用于显示或设置系统时间的命令，`+%s` 是 `date` 的格式化选项，表示**将当前时间转换为 Unix 时间戳**（即从 1970 年 1 月 1 日 00:00:00 UTC 到当前时刻的秒数）。

### **示例**

```bash

$ date +%s

1689253200 # 输出当前时间戳（会随执行时间变化）

```

### **在脚本中的作用**

在你的脚本中，`date +%s` 用于生成唯一的日志目录名（如 `log_1689253200`），确保每次运行脚本时创建的日志目录不会冲突。

## **2. `2>&1` 的含义**

`2>&1` 是 Shell 中的**重定向符号**，用于将**标准错误输出（stderr）合并到标准输出（stdout）**。具体解释：

- **`1`**：代表标准输出（stdout），是命令的正常输出。

- **`2`**：代表标准错误（stderr），是命令执行过程中的错误信息。

- **`>`**：重定向操作符，用于将输出导向到文件或其他位置。

- **`&1`**：表示 “指向标准输出的位置”。

- **作用**：将错误信息和正常输出合并，方便统一处理（如写入同一个日志文件）。

### **示例对比**

```bash

# 仅将标准输出重定向到文件（错误信息仍会显示在终端）

ls /nonexistent > output.txt

  

# 将标准输出和标准错误都重定向到文件

ls /nonexistent > output.txt 2>&1

  

# 执行命令，正常输出写入 log.txt，错误输出单独写入 error.txt

make > log.txt 2> error.txt

```

- **示例**：

```bash

# 执行 make 并将所有输出（正常日志+错误）写入 build.log

make -j16 > build.log 2>&1

```

- `> build.log`：将 `stdout` 写入 `build.log`。

- `2>&1`：将 `stderr` 重定向到 `stdout`（因此也会写入 `build.log`）。

## **1. `mkdir -p` 的含义**

`mkdir` 是用于创建目录的命令，`-p` 是其常用选项，作用是 **“递归创建目录，且忽略已存在的目录（不报错）”**。

### 具体作用

- **递归创建多级目录**：比如 `mkdir -p a/b/c`，会一次性创建 `a`、`a/b`、`a/b/c` 三级目录（如果它们不存在）。

- **避免 “目录已存在” 错误**：如果目录已经存在，`mkdir -p` 不会像普通 `mkdir` 那样报错（普通 `mkdir` 会提示 `File exists`）。

### 示例

```bash

# 假设当前没有任何目录

mkdir -p build/block_1 # 成功创建 build 和 build/block_1

mkdir -p build/block_1 # 再次执行不报错（因为目录已存在）

```

在你之前的脚本中，`mkdir -p "$LOG_DIR"` 和 `mkdir -p "$BUILD_SUBDIR"` 就是为了确保日志目录和编译子目录能被正确创建，即使上级目录不存在或目录已存在也不会中断脚本。

## `find -name -o`：文件搜索与条件逻辑

`find` 是 Shell 中用于搜索文件 / 目录的命令，`-name` 和 `-o` 是其核心参数。

### 1. 基本语法

```bash

find [搜索路径] [搜索条件]

```

### 2. 参数解释

- **`-name "pattern"`**：按文件名匹配搜索（支持通配符 `*`、`?` 等），区分大小写。

- 示例：`find ./src -name "*.cpp"` → 在 `./src` 目录下搜索所有 `.cpp` 后缀的文件。

- **`-o`**：逻辑 “或”（OR）运算符，用于连接多个搜索条件，满足任意一个条件即匹配。

- 注意：`find` 中默认条件是 “与”（AND），`-o` 显式指定 “或”。

### 3. 常见用法

```bash

# 搜索 ./src 目录下的 .cpp 文件 或 .c 文件（-o 连接两个 -name 条件）

find ./src -name "*.cpp" -o -name "*.c"

  

# 搜索 ./include 目录下的 .h 或 .hpp 文件，且排除 ./include/old 子目录

find ./include -path "./include/old" -prune -o -name "*.h" -o -name "*.hpp"

```

### 4. 注意事项

- 多个 `-o` 条件需要用括号包裹（需转义 ``\(` `\)``），否则可能逻辑错误：

正确：`find . \( -name "*.h" -o -name "*.cpp" \) -print`

- `-iname` 是 `-name` 的不区分大小写版本（如 `-iname "*.CPP"` 会匹配 `.cpp` `.CPP` 等）。

# Make

## `make -C`：切换目录并执行 Make

`make` 是构建工具，用于执行 Makefile 中的编译规则；`-C` 用于指定执行目录。

- **参数解释**：`-C <dir>` → 先切换到 `<dir>` 目录，再执行 `make`（等效于 `cd <dir> && make`）。

- **示例**：

```bash

# 进入 ./build 目录并执行 make（编译项目）

make -C ./build

# 进入 ./build 目录并并行编译（-j16 表示16线程）

make -C ./build -j16

```

# `echo -e`：解析转义字符

`echo` 用于输出字符串；`-e` 用于启用转义字符解析（如换行 `\n`、制表符 `\t` 等）。

## 1. 参数解释

`-e`：让 `echo` 识别并解析字符串中的转义序列（默认不解析）。

## 2. 常用转义字符

- `\n`：换行

- `\t`：制表符（Tab）

- `\r`：回车（光标回到行首）

## 3. 示例

```bash

# 不使用 -e：转义字符被当作普通字符输出

echo "Line1\nLine2" # 输出：Line1\nLine2

  

# 使用 -e：解析 \n 为换行

echo -e "Line1\nLine2"

# 输出：

# Line1

# Line2

  

# 结合制表符

echo -e "Name\tAge\nTom\t20"

# 输出：

# Name Age

# Tom 20

```

## 4. 注意

- 某些 Shell（如 `bash`）中，`echo` 不加 `-e` 也可能解析转义字符，但为了兼容性（如 `sh`），建议显式加 `-e`。

- 若要输出 `-e` 本身，可加 `--` 标记：`echo -- -e` → 输出 `-e`。

# `@` 的含义

- 在 Makefile 中，默认会先打印执行的命令，再显示命令输出。

- 加上 `@` 后，会**隐藏命令本身的打印**，只显示 `echo` 的输出内容。

## 示例（Makefile 中）

```makefile

# 不带 @ 的情况

test1:

echo "Hello 1" # 执行时会先显示 "echo "Hello 1"，再显示 "Hello 1"

  

# 带 @ 的情况

test2:

@echo "Hello 2" # 执行时只显示 "Hello 2"，不显示命令本身

```

执行 `make test1` 输出：

```plaintext

echo "Hello 1"

Hello 1

```

执行 `make test2` 输出：

```plaintext

Hello 2

```

## 注意

`@` 是 Makefile 的特殊符号，在 Shell 脚本中使用 `@echo` 会报错（Shell 会把 `@` 当作普通字符，提示 “命令未找到”）。

# `cmake -B`：指定构建目录（现代 CMake 用法）

`cmake` 是跨平台构建工具，用于生成 Makefile、VS 项目等构建文件，`-B` 是指定构建目录的参数（CMake 3.13+ 支持）。

## 1. 基本语法

```bash

cmake -B <构建目录> [源代码目录]

```

## 2. 参数解释

- **`-B <build_dir>`**：指定构建目录（存放生成的 Makefile、中间文件等），目录不存在时会自动创建。

- 后续的 `[源代码目录]`：指定 CMakeLists.txt 所在的源代码根目录（通常是项目根目录）。

## 3. 作用与优势

- 实现 “**源代码目录与构建目录分离**”（推荐做法）：避免构建产物污染源代码目录。

- 无需提前进入构建目录，直接在命令行指定即可。

## 4. 示例

```bash

# 在当前目录（源代码目录）生成构建文件到 ./build 目录

cmake -B ./build

  

# 明确指定源代码目录（当在其他目录执行cmake时）

cmake -B ./build ~/projects/my_prj # ~/projects/my_prj 是源代码根目录（含CMakeLists.txt）

```

# Gdb

| 命令 | 简写 | 含义 |

| ----------------- | ----- | ----------------------------------------------- |

| list | l | 列出 10 行代码 |

| break | b | 设置断点 |

| break if | b if | 设置条件断点 |

| delete [break id] | d | 删除断点 047 (按照 break id) 删除，没有 break id, 删除所有段 6 |

| disable | | 禁用断点 |

| enable | | 允许断点 |

| info | i | 显示程序状态. info b (列出断点), info regs (列出寄存器) 等 |

| run [args] | r | 开始运行程序，可带参数 |

| display | disp | 跟踪查看那某个变量，每次停下来都显示其值 |

| print | p | 打印内部变量值 |

| watch | | 监视变量值新旧的变化 |

| step | s | 执行下一条语句，如果该语句为函数调用，则进入函数执行第一条语句 |

| next | n | 执行下一条语句，如果该语句为函数调用，不会进入函数内部执行 (即不会一步步地调试函数内部语句） |

| continue | c | 继续程序的运行，直到遇到下一个断点 |

| finish | | 如果进入了某个函数，返回到调用调用它的函数，jump out |

| set var name = v | | 设置变量的值 |

| backtrace | bt | 查看函数调用信息（堆栈） |

| start | st | 开始执行程序，在 main 函数中的第一条语句前停下 |

| frame | f | 查看栈帧，比如 frame 1 查看 1 号栈帧 |

| up | | 查看上一个栈帧 |

| down | | 查看那下一个栈帧 |

| quit | q | 离开 gdb |

| edit | | 在 gdb 中进行编辑 |

| whatis | | 查看变量的类型 |

| search | | 搜索源文件中的文本 |

| file | | 装入需要调试的程序 |

| kill | k | 终止正在调试的程序 |

| layout | | 改变当前布局 (必备命令) |

| examine | x | 查看内存空间 (必备命令) |

| checkpoint | ch | debug 快照，需要反复调试某一段代码时，非常有用 |

| disassemble | disas | 反汇编 |

| stepi | si | 下一行指令 (遇到函数，进入函数) |

| nexti | ni | 下一行指令 |

|命令|解释|示例|

|---|---|---|

|file <文件名>|加载被调试的可执行程序文件。 <br>因为一般都在被调试程序所在目录下执行 GDB，因而文本名不需要带路径。|(gdb) file gdb-sample|

|r|Run 的简写，运行被调试的程序。 <br>如果此前没有下过断点，则执行完整个程序；如果有断点，则程序暂停在第一个可用断点处。|(gdb) r|

|c|Continue 的简写，继续执行被调试程序，直至下一个断点或程序结束。|(gdb) c|

|b <行号> <br>b <函数名称> <br>b *<函数名称> <br>b *<代码地址> d [编号]|b: Breakpoint 的简写，设置断点。两可以使用 “行号”“函数名称”“执行地址” 等方式指定断点位置。 <br>其中在函数名称前面加 “*” 符号表示将断点设置在“由编译器生成的 prolog 代码处”。如果不了解汇编，可以不予理会此用法。 d: Delete breakpoint 的简写，删除指定编号的某个断点，或删除所有断点。断点编号从 1 开始递增。|(gdb) b 8 <br>(gdb) b main <br>(gdb) b *main <br>(gdb) b *0x804835c (gdb) d|

|s, n|s: 执行一行源程序代码，如果此行代码中有函数调用，则进入该函数； <br>n: 执行一行源程序代码，此行代码中的函数调用也一并执行。 s 相当于其它调试器中的 “Step Into (单步跟踪进入)”； <br>n 相当于其它调试器中的 “Step Over (单步跟踪)”。 这两个命令必须在有源代码调试信息的情况下才可以使用（GCC 编译时使用“-g” 参数）。|(gdb) s <br>(gdb) n|

|si, ni|si 命令类似于 s 命令，ni 命令类似于 n 命令。所不同的是，这两个命令（si/ni）所针对的是汇编指令，而 s/n 针对的是源代码。|(gdb) si <br>(gdb) ni|

|p <变量名称>|Print 的简写，显示指定变量（临时变量或全局变量）的值。|(gdb) p i <br>(gdb) p nGlobalVar|

|display ... undisplay <编号>|display，设置程序中断后欲显示的数据及其格式。 <br>例如，如果希望每次程序中断后可以看到即将被执行的下一条汇编指令，可以使用命令 <br>“display /i $pc” <br>其中 $pc 代表当前汇编指令，/i 表示以十六进行显示。当需要关心汇编代码时，此命令相当有用。 undispaly，取消先前的 display 设置，编号从 1 开始递增。|(gdb) display /i $pc (gdb) undisplay 1|

|i|info 的简写，用于显示各类信息，详情请查阅 “help i”。|(gdb) i r|

|q|Quit 的简写，退出 GDB 调试环境。|(gdb) q|

|help [命令名称]|GDB 帮助命令，提供对 GDB 名种命令的解释说明。 <br>如果指定了 “命令名称” 参数，则显示该命令的详细说明；如果没有指定参数，则分类显示所有 GDB 命令，供用户进一步浏览和查询。|(gdb) help|

| 命令名称 | 命令缩写 | 命令说明 |

| ------------ | --------- | ------------------------ |

| **run** | r | 运行一个待调试的程序 |

| **continue** | c | 让暂停的程序继续运行 |

| **next** | n | 运行到下一行 |

| **step** | s | 单步执行，遇到函数会进入 |

| **until** | u | 运行到指定行停下来 |

| **finish** | fi | 结束当前调用函数，回到上一层调用函数处 |

| return | return | 结束当前调用函数并返回指定值，到上一层函数调用处 |

| jump | j | 将当前程序执行流跳转到指定行或地址 |

| print | p | 打印变量或寄存器值 |

| backtrace | bt | 查看当前线程的调用堆栈 |

| frame | f | 切换到当前调用线程的指定堆栈 |

| thread | thread | 切换到指定线程 |

| break | b | 添加断点 |

| tbreak | tb | 添加临时断点 |

| delete | d | 删除断点 |

| enable | enable | 启用某个断点 |

| disable | disable | 禁用某个断点 |

| watch | watch | 监视某一个变量或内存地址的值是否发生变化 |

| list | l | 显示源码 |

| info | i | 查看断点 / 线程等信息 |

| ptype | ptype | 查看变量类型 |

| disassemble | dis | 查看汇编代码 |

| set args | set args | 设置程序启动命令行参数 |

| show args | show args | 查看设置的命令行参数 |

## 查看变量的方法

在 GDB 中，除了常规的打印和类型查看命令外，还有多种高级技巧可以更高效地查看变量、监控内存状态及分析复杂数据结构。以下是补充方法：

### **一、使用 GDB 的变量自动显示（Display）**

设置后每次程序暂停时自动打印变量，避免重复输入：

```bash

(gdb) display x # 每次程序暂停时自动打印变量x

(gdb) display/i $pc # 自动显示当前执行的汇编指令

(gdb) info display # 查看所有自动显示设置

(gdb) undisplay 1 # 删除编号为1的自动显示设置

```

### **二、内存区域可视化**

1. **连续内存块查看**

```bash

(gdb) x/10xw buffer # 以16进制格式查看buffer开始的10个word（4字节）

(gdb) x/20b array # 以字节为单位查看array的20个元素

(gdb) x/5i $pc # 查看当前指令地址开始的5条汇编指令

```

格式说明：`x/[数量][格式][单位] 地址`，常用格式有 `x`（16 进制）、`d`（十进制）、`s`（字符串）、`i`（指令）。

2. **字符串与宽字符查看**

```bash

(gdb) print (char*)buffer # 打印以null结尾的字符串

(gdb) print/c *buffer@10 # 以字符形式打印buffer的前10个元素

(gdb) print/wcs L"宽字符串" # 打印宽字符字符串

```

### **三、复杂数据结构分析**

1. **结构体与数组的组合**

```bash

(gdb) print *(struct Point (*)[10])array # 将array解释为Point[10]数组

(gdb) print (*(MyClass*)obj)->method() # 调用对象的方法（需程序处于暂停状态）

```

2. **处理多级指针**

```bash

(gdb) print **pptr # 打印二级指针pptr指向的对象

(gdb) print (*pptr)->x # 打印二级指针指向对象的成员

```

3. **STL 容器深度解析（需安装 Python Pretty Printers）**

```bash

(gdb) p my_vector.size() # 打印vector的大小

(gdb) p my_map["key"] # 打印map中特定键的值

(gdb) p *my_list.begin() # 打印list的第一个元素

```

### **四、动态类型识别（RTTI）增强**

1. **多态对象的真实类型判断**

```bash

(gdb) p *(Derived*)base_ptr # 强制转换为派生类类型（需手动指定可能的类型）

(gdb) python print(gdb.parse_and_eval('base_ptr').dynamic_type) # 动态获取真实类型

```

2. **虚拟函数表查看**

```bash

(gdb) p *(void***)obj # 打印对象的虚函数表指针

(gdb) p (void(*)())*(void***)obj[0] # 打印第一个虚函数的地址

```

### **五、条件表达式与自定义函数**

1. **计算表达式**

```bash

(gdb) print arr[0] + arr[1] # 计算表达式值

(gdb) print strlen(name) # 调用函数计算（需程序未退出）

```

2. **定义临时变量**

```bash

(gdb) set $sum = 0 # 定义临时变量$sum

(gdb) set $sum = $sum + arr[i] # 累加计算

(gdb) print $sum # 打印临时变量结果

```

3. **自定义 GDB 函数（Python）**

```python

(gdb) python

>def print_array(name, length):

> arr = gdb.parse_and_eval(name)

> for i in range(length):

> print(f"{name}[{i}] = {arr[i]}")

>end

(gdb) python print_array("my_array", 10)

```

### **六、内存变化监控**

1. **硬件观察点（Watchpoint）**

```bash

(gdb) watch var # 变量var被修改时触发断点

(gdb) rwatch var # 变量var被读取时触发断点

(gdb) awatch var # 变量var被读取或修改时触发断点

```

2. **内存范围监控**

```bash

(gdb) watch *array@10 # 监控array开始的10个元素的变化

```

### **七、可视化插件与工具**

1. **DDD（Data Display Debugger）**

图形化前端，支持变量可视化：

```bash

ddd --gdb ./program # 启动DDD并连接GDB

```

2. **GDB Dashboard**

增强 GDB 界面，自动显示源代码、寄存器、堆栈等信息：

```bash

git clone [https://github.com/cyrus-and/gdb-dashboard.git](https://github.com/cyrus-and/gdb-dashboard.git)

echo "source ~/gdb-dashboard/.gdbinit" >> ~/.gdbinit

```

3. **VSCode 的 GDB 插件**

通过图形界面查看变量，支持自动补全和格式化显示。

### **八、性能分析与变量关联**

1. **统计变量变化频率**

```bash

(gdb) break func

(gdb) commands

> silent

> set $counter = $counter + 1

> if $counter % 100 == 0

> print x

> end

> continue

> end

```

2. **时间序列分析**

使用 Python 脚本记录变量随时间的变化：

```python

(gdb) python

>values = []

>def record_x():

> x = gdb.parse_and_eval('x')

> values.append(int(x))

> if len(values) % 10 == 0:

> print(f"Collected {len(values)} samples")

>end

(gdb) commands 1

> python record_x()

> continue

> end

```

### **九、远程调试与分布式环境**

1. **跨平台调试**

```bash

# 目标设备（ARM）

gdbserver :1234 ./program

# 开发机（x86）

gdb-multiarch -ex "target remote <IP>:1234" ./program

```

2. **分布式调试**

使用 GDBSERVER 和多个 GDB 实例调试分布式系统：

```bash

# 节点1

gdbserver :1234 ./server

# 节点2

gdbserver :1235 ./client

# 开发机

gdb -ex "target remote <节点1>:1234" -ex "target remote <节点2>:1235"

```

## Info

在 GDB 中，`info` 命令是一个强大的信息查询工具，可用于查看调试过程中的各种状态信息。

### **一、断点与观察点信息**

```bash

(gdb) info breakpoints # 查看所有断点和观察点信息

(gdb) info break 1 # 查看编号为1的断点详情

(gdb) info watchpoints # 仅查看观察点信息

```

### **二、线程与进程信息**

```bash

(gdb) info threads # 查看所有线程状态（ID、名称、当前函数等）

(gdb) info inferiors # 查看多进程调试中的所有进程

(gdb) info registers # 查看所有寄存器的值

(gdb) info frame # 查看当前栈帧的详细信息

```

### **三、程序与符号信息**

```bash

(gdb) info program # 查看程序当前状态（运行中、已停止等）

(gdb) info sharedlibrary # 查看已加载的共享库

(gdb) info functions # 查看所有函数符号

(gdb) info variables # 查看所有全局和静态变量

(gdb) info locals # 查看当前栈帧的局部变量

```

### **四、内存与映射信息**

```bash

(gdb) info proc mappings # 查看进程的内存映射（类似pmap命令）

(gdb) info files # 查看程序文件和符号表信息

(gdb) info address var # 查看变量var的内存地址

```

### **五、源文件与行号信息**

```bash

(gdb) info sources # 查看程序的源文件列表

(gdb) info line 10 # 查看第10行对应的函数和地址

(gdb) info line func # 查看函数func的起始行号

```

### **六、信号与异常信息**

```bash

(gdb) info signals # 查看GDB如何处理各种信号（如SIGINT、SIGSEGV）

(gdb) info handle # 查看信号处理设置的详细信息

```

### **七、调试会话信息**

```bash

(gdb) info history # 查看GDB命令历史

(gdb) info display # 查看自动显示的变量（使用display命令设置）

(gdb) info macros # 查看定义的GDB宏命令

```

### **八、高级用法示例**

1. **查看线程详细信息**

```bash

(gdb) info threads

Id Target Id Frame

1 Thread 0x7ffff7fc5700 (LWP 2809) "program" main () at main.c:10

2 Thread 0x7ffff77c4700 (LWP 2810) "worker" worker_thread () at worker.c:25

```

2. **查看内存映射**

```bash

(gdb) info proc mappings

process 2809

Mapped address spaces:

Start Addr End Addr Size Offset objfile

0x400000 0x401000 0x1000 0x0 /home/user/program

0x600000 0x601000 0x1000 0x0 /home/user/program

0x7ffff7a0d000 0x7ffff7bcd000 0x2c0000 0x0 /lib/x86_64-linux-gnu/libc-2.27.so

```

3. **查看信号处理设置**

```bash

(gdb) info signals

Signal Stop Print Pass to program Description

SIGINT Yes Yes No Interrupt

SIGQUIT Yes Yes No Quit

SIGILL Yes Yes No Illegal instruction

SIGTRAP Yes Yes No Trace/breakpoint trap

```

## 技巧

### **一、快速打印变量的方法**

1. **使用历史命令（最直接）**

- 在 GDB 中按 **↑键** 可快速召回上一条命令，重复按可浏览历史命令。

- 使用 `history` 命令查看所有历史输入。

2. **设置别名（Alias）**

```bash

(gdb) alias pn = print node->next # 为复杂表达式创建别名

(gdb) pn # 直接使用别名打印

```

3. **使用 GDB 的自动补全**

- 输入变量名前缀后按 **Tab 键**，GDB 会自动补全变量名。

- 例如：输入 `pri` 后按 Tab，GDB 会补全为 `print`。

4. **保存常用命令到 GDBinit 文件**

在用户目录创建 `.gdbinit` 文件，添加常用命令：

```bash

alias pn = print node->next

alias ps = print *stack

```

启动 GDB 时会自动加载这些设置。

### **二、查看变量类型**

使用 `whatis` 和 `ptype` 命令：

```bash

(gdb) whatis x # 查看变量x的基本类型（如int、struct）

(gdb) ptype x # 查看变量x的完整类型定义（包括结构体成员）

(gdb) ptype *(node) # 查看指针node指向对象的类型

```

# Gdb 多线程调试

mpi

[debugging - MPI并行程序的调试技巧 - galois - SegmentFault 思否]([https://segmentfault.com/a/1190000000447786](https://segmentfault.com/a/1190000000447786))

[程序调试 — 中国科大超级计算中心用户使用手册 ：2024-05-18 版 文档]([程序调试 - 中国科大超级计算中心用户使用手册 ：2024-05-18版 文档](https://scc.ustc.edu.cn/zlsc/user_doc/html/debug/debug.html#id12))
