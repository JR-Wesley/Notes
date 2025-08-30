# 错误

# C/C++ 程序常见错误提示及 GDB 调试指南

## 一、常见错误和异常提示

### 1. 编译时错误（Compile-time Errors）

#### 语法错误

```bash
// 缺少分号
error: expected ';' before '}' token
error: expected unqualified-id before 'return'

// 变量未声明
error: 'undefined_variable' was not declared in this scope

// 类型不匹配
error: cannot convert 'int*' to 'double*' in assignment
error: invalid conversion from 'const char*' to 'char'

// 函数未定义
error: 'function_name' was not declared in this scope
undefined reference to `function_name'
```

#### 头文件相关

```bash
// 头文件未找到
fatal error: header_file.h: No such file or directory
compilation terminated.

// 重复包含
warning: #pragma once in main file
```

### 2. 运行时错误（Runtime Errors）

#### 段错误（Segmentation Fault）

```bash
Segmentation fault (core dumped)
```

**常见原因：**
- 空指针解引用
- 数组越界
- 野指针使用
- 栈溢出

#### 总线错误（Bus Error）

```bash
Bus error (core dumped)
```

**常见原因：**
- 内存对齐问题
- 访问无效的内存地址

#### 中止错误（Aborted）

```bash
Aborted (core dumped)
```

**常见原因：**
- `assert()` 失败
- `free()` 重复释放内存
- `malloc()` 内存不足

#### 浮点异常

```bash
Floating point exception (core dumped)
```

**常见原因：**
- 除以零
- 无效的浮点运算

## 二、GDB 调试实战

### 1. 基本 GDB 调试流程

```bash
# 编译时添加调试信息
gcc -g -o program program.c
g++ -g -o program program.cpp

# 启动GDB
gdb ./program

# 运行程序
(gdb) run
# 或带参数运行
(gdb) run arg1 arg2
```

### 2. 常见错误的 GDB 调试方法

#### 场景 1：段错误调试

```c
// segfault_example.c
#include <stdio.h>

void bad_function() {
    int *ptr = NULL;
    *ptr = 42;  // 段错误
}

int main() {
    printf("程序开始\n");
    bad_function();
    printf("程序结束\n");
    return 0;
}
```

**GDB 调试步骤：**

```gdb
# 启动GDB
gdb ./segfault_example

# 运行程序
(gdb) run

# 程序崩溃后，查看调用栈
(gdb) bt
# 输出：
# #0  0x0000555555555156 in bad_function () at segfault_example.c:5
# #1  0x000055555555517a in main () at segfault_example.c:10

# 查看当前栈帧的源代码
(gdb) list
# 输出：
#    1    #include <stdio.h>
#    2    
#    3    void bad_function() {
#    4        int *ptr = NULL;
#    5        *ptr = 42;  // 段错误
#    6    }
#    7    
#    8    int main() {
#    9        printf("程序开始\n");
#   10        bad_function();

# 查看变量值
(gdb) print ptr
# $1 = (int *) 0x0

# 查看寄存器
(gdb) info registers
```

#### 场景 2：数组越界调试

```c
// array_overflow.c
#include <stdio.h>

int main() {
    int arr[5] = {1, 2, 3, 4, 5};
    
    // 数组越界
    for (int i = 0; i <= 5; i++) {  // 错误：应该是 i < 5
        printf("arr[%d] = %d\n", i, arr[i]);
    }
    
    return 0;
}
```

**GDB 调试步骤：**

```gdb
# 设置断点
(gdb) break array_overflow.c:8

# 运行程序
(gdb) run

# 单步执行
(gdb) next
(gdb) next

# 查看变量
(gdb) print i
(gdb) print arr[i]
(gdb) print arr

# 继续执行
(gdb) continue
```

#### 场景 3：内存泄漏检测

```c
// memory_leak.c
#include <stdio.h>
#include <stdlib.h>

void memory_leak_function() {
    int *ptr = (int*)malloc(sizeof(int) * 10);
    // 忘记释放内存
    // free(ptr);
}

int main() {
    memory_leak_function();
    printf("内存泄漏测试\n");
    return 0;
}
```

**使用 Valgrind 检测内存泄漏：**

```bash
valgrind --tool=memcheck --leak-check=full ./memory_leak

# 输出：
# ==12345== HEAP SUMMARY:
# ==12345==     in use at exit: 40 bytes in 1 blocks
# ==12345==   total heap usage: 1 allocs, 0 frees, 40 bytes allocated
# ==12345== 
# ==12345== 40 bytes in 1 blocks are definitely lost in loss record 1 of 1
# ==12345==    at 0x4C2E0EF: malloc (in /usr/lib/valgrind/vgpreload_memcheck-amd64-linux.so)
# ==12345==    by 0x108728: memory_leak_function (memory_leak.c:6)
# ==12345==    by 0x108745: main (memory_leak.c:11)
```

### 3. GDB 高级调试技巧

#### 3.1 条件断点

```gdb
# 在特定条件下停止
(gdb) break file.c:15 if x > 10

# 仅在第100次执行时停止
(gdb) break file.c:20
(gdb) command 1
> silent
> printf "x = %d, y = %d\n", x, y
> continue
> end
```

#### 3.2 监视点（Watchpoint）

```gdb
# 当变量值改变时停止
(gdb) watch variable_name

# 当内存地址被修改时停止
(gdb) watch *0x7ffffffeed00

# 条件监视点
(gdb) watch x if x == 0
```

#### 3.3 函数调用调试

```gdb
# 查看函数参数
(gdb) info args

# 查看局部变量
(gdb) info locals

# 打印变量值
(gdb) print variable_name
(gdb) p/x ptr      # 以十六进制显示
(gdb) p/d count    # 以十进制显示
(gdb) p array[0]@5 # 显示数组前5个元素
```

#### 3.4 多线程调试

```gdb
# 查看所有线程
(gdb) info threads

# 切换线程
(gdb) thread 2

# 查看特定线程的调用栈
(gdb) thread apply 2 bt

# 设置线程特定断点
(gdb) break file.c:15 thread 2
```

### 4. 实用 GDB 命令速查

| 命令 | 说明 |
|------|------|
| `run` | 运行程序 |
| `break` 或 `b` | 设置断点 |
| `next` 或 `n` | 单步执行（不进入函数） |
| `step` 或 `s` | 单步执行（进入函数） |
| `continue` 或 `c` | 继续执行 |
| `print` 或 `p` | 打印变量值 |
| `backtrace` 或 `bt` | 显示调用栈 |
| `info locals` | 显示局部变量 |
| `info args` | 显示函数参数 |
| `list` 或 `l` | 显示源代码 |
| `watch` | 设置监视点 |
| `delete` | 删除断点 |
| `quit` 或 `q` | 退出 GDB |

### 5. 调试配置文件

创建 `.gdbinit` 文件进行个性化配置：

```gdb
# ~/.gdbinit
set confirm off
set pagination off
set print pretty on
set print array on
set print array-indexes on
set print elements 0
set history save on
set history filename ~/.gdb_history
set history size 1000

# 自动加载符号
set auto-solib-add on

# 显示源代码行数
set listsize 20

# 颜色支持
set style enabled on
```

### 6. 常见问题解决

#### 问题 1：GDB 无法显示源代码

```gdb
# 确保编译时包含调试信息
gcc -g -o program program.c

# 检查源文件路径
(gdb) show directories

# 添加源文件搜索路径
(gdb) directory /path/to/source
```

#### 问题 2：Core 文件调试

```bash
# 使用core文件调试
gdb ./program core

# 或
gdb --core=core ./program

# 查看程序崩溃时的状态
(gdb) bt
(gdb) info registers
(gdb) info sharedlibrary
```

#### 问题 3：动态库调试

```gdb
# 确保加载了动态库的调试符号
(gdb) info sharedlibrary

# 如果缺少调试符号，安装对应的debug包
# Ubuntu/Debian: sudo apt-get install libxxx-dev libxxx-dbg
# CentOS/RHEL: sudo yum install libxxx-devel debuginfo-install libxxx
```

通过掌握这些常见的错误提示和 GDB 调试技巧，您可以更有效地定位和解决 C/C++ 程序中的各种问题。记住，良好的调试习惯包括：编译时添加调试信息、使用断点逐步排查、善用监视点和条件断点，以及结合 Valgrind 等工具进行内存分析。

# Core Dump

# C/C++ 程序内核崩溃及 GDB 调试指南

## 1. 内核崩溃（Core Dump）是什么？

内核崩溃（Core Dump）是当程序异常终止时，操作系统将程序的内存映像保存到磁盘文件的过程。这个文件包含了程序崩溃时的完整内存状态，包括：

- 程序的内存映像
- 寄存器状态
- 调用栈信息
- 变量值
- 堆栈内容

### 常见导致内核崩溃的原因

```c
// 1. 空指针解引用
int *ptr = NULL;
*ptr = 10;  // SIGSEGV (段错误)

// 2. 数组越界
int arr[5];
arr[10] = 100;  // 可能导致SIGSEGV

// 3. 野指针
int *p;
*p = 42;  // 使用未初始化的指针

// 4. 内存重复释放
int *ptr = malloc(sizeof(int));
free(ptr);
free(ptr);  // 重复释放，SIGABRT

// 5. 栈溢出
void infinite_recursion() {
    infinite_recursion();  // 无限递归导致栈溢出
}

// 6. 缓冲区溢出
char buffer[10];
strcpy(buffer, "This string is too long!");  // 缓冲区溢出
```

## 2. 启用 Core Dump 功能

### 2.1 检查和设置 Core Dump 限制

```bash
# 查看当前core dump大小限制
ulimit -c

# 设置core dump大小为无限制
ulimit -c unlimited

# 或者设置具体大小（如1GB）
ulimit -c 1073741824

# 永久设置：编辑 /etc/security/limits.conf
# * soft core unlimited
# * hard core unlimited
```

### 2.2 配置 Core 文件命名和位置

```bash
# 查看当前core文件命名模式
cat /proc/sys/kernel/core_pattern

# 设置core文件命名模式
echo "core.%e.%p.%h.%t" | sudo tee /proc/sys/kernel/core_pattern

# 常用的core文件命名模式：
# %e - 可执行文件名
# %p - 进程PID
# %h - 主机名
# %t - 时间戳（UNIX时间）
# %u - 用户ID
# %g - 组ID
```

### 2.3 创建 Core Dump 测试程序

```c
// crash_test.c
#include <stdio.h>
#include <stdlib.h>

void bad_function() {
    int *ptr = NULL;
    printf("即将访问空指针...\n");
    *ptr = 42;  // 这将导致段错误
}

int main() {
    printf("程序开始执行\n");
    printf("PID: %d\n", getpid());
    
    bad_function();
    
    printf("这行不会被执行\n");
    return 0;
}
```

编译并运行：

```bash
gcc -g -o crash_test crash_test.c
./crash_test
```

## 3. 使用 GDB 调试 Core 文件

### 3.1 基本 GDB 调试命令

```bash
# 使用GDB加载程序和core文件
gdb ./crash_test core

# 或者
gdb --core=core ./crash_test
```

### 3.2 GDB 常用调试命令

```gdb
# 启动GDB后常用命令：

# 显示程序崩溃时的调用栈
(gdb) bt
(gdb) backtrace

# 显示带源代码行号的调用栈
(gdb) bt full

# 切换到特定栈帧
(gdb) frame 0
(gdb) frame 1

# 查看当前栈帧的局部变量
(gdb) info locals

# 查看特定变量的值
(gdb) print variable_name
(gdb) p ptr
(gdb) p *ptr

# 查看寄存器内容
(gdb) info registers

# 查看程序崩溃时的汇编代码
(gdb) disassemble

# 查看特定函数的汇编代码
(gdb) disassemble function_name

# 继续执行程序（通常不会成功）
(gdb) continue

# 退出GDB
(gdb) quit
```

### 3.3 实际调试示例

假设我们有以下程序：

```c
// example.c
#include <stdio.h>
#include <string.h>

void process_string(char *str) {
    char buffer[50];
    
    // 模拟缓冲区溢出
    strcpy(buffer, str);
    
    // 处理字符串
    for (int i = 0; i < strlen(buffer); i++) {
        buffer[i] = toupper(buffer[i]);
    }
    
    printf("Processed: %s\n", buffer);
}

void analyze_data(int *data, int count) {
    int sum = 0;
    
    // 数组越界访问
    for (int i = 0; i <= count; i++) {  // 注意：应该是 i < count
        sum += data[i];
    }
    
    printf("Sum: %d\n", sum);
}

int main(int argc, char *argv[]) {
    // 测试缓冲区溢出
    char long_string[100] = "This is a very long string that will cause buffer overflow";
    process_string(long_string);
    
    // 测试数组越界
    int numbers[5] = {1, 2, 3, 4, 5};
    analyze_data(numbers, 5);
    
    return 0;
}
```

编译时包含调试信息：

```bash
gcc -g -fsanitize=address -o example example.c
```

运行程序产生 core 文件，然后使用 GDB 调试：

```gdb
# 加载程序和core文件
gdb ./example core

# 查看调用栈
(gdb) bt
# 输出可能类似：
# #0  0x00007f8a1b2c5337 in __GI_raise (sig=sig@entry=6) at ../nptl/sysdeps/unix/sysv/linux/raise.c:56
# #1  0x00007f8a1b2c8a28 in __GI_abort () at abort.c:89
# #2  0x00007f8a1b3062a4 in __libc_message (do_abort=do_abort@entry=1, fmt=fmt@entry=0x7f8a1b415210 "*** %s ***: %s terminated\n") at ../sysdeps/posix/libc_fatal.c:175
# #3  0x00007f8a1b3afbe8 in __GI___fortify_fail (msg=<optimized out>) at fortify_fail.c:37
# #4  0x00007f8a1b3afbb9 in __GI___fortify_fail_abort (need_backtrace=need_backtrace@entry=0, msg=msg@entry=0x7f8a1b4150d8 "stack smashing detected") at fortify_fail.c:43
# #5  0x00007f8a1b3afb66 in __stack_chk_fail () at stack_chk_fail.c:28
# #6  0x00000000004006b6 in process_string (str=0x7ffca5e8b9f0 "This is a very long string that will cause buffer overflow") at example.c:8
# #7  0x000000000040075d in main (argc=1, argv=0x7ffca5e8baa8) at example.c:27

# 切换到问题函数的栈帧
(gdb) frame 6
# #6  0x00000000004006b6 in process_string (str=0x7ffca5e8b9f0 "This is a very long string that will cause buffer overflow") at example.c:8

# 查看源代码
(gdb) list
# 输出源代码，显示问题行

# 查看局部变量
(gdb) info locals
# buffer = "This is a very long string that will cause buffer ove\000\060\252\202\232|\177\000\000\360Y\252\232|\177\000"

# 查看传入的参数
(gdb) print str
```

## 4. 高级调试技巧

### 4.1 使用 GDB 脚本自动化调试

```gdb
# debug_script.gdb
set confirm off
set pagination off

# 加载core文件
target core core

# 显示调用栈
backtrace full

# 显示寄存器
info registers

# 显示所有线程的调用栈（多线程程序）
thread apply all backtrace

# 保存调试信息到文件
set logging on
set logging file debug_output.txt

# 执行常用命令
backtrace
info locals
info args

set logging off

quit
```

使用脚本：

```bash
gdb -x debug_script.gdb ./program core
```

### 4.2 分析多线程程序的 Core 文件

```c
// multithread_crash.c
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

void* thread_function(void* arg) {
    int thread_id = *(int*)arg;
    
    printf("线程 %d 开始执行\n", thread_id);
    
    // 模拟线程崩溃
    if (thread_id == 2) {
        int *ptr = NULL;
        *ptr = 42;  // 线程2将崩溃
    }
    
    sleep(5);  // 其他线程继续运行
    return NULL;
}

int main() {
    pthread_t threads[3];
    int thread_ids[3] = {1, 2, 3};
    
    // 创建多个线程
    for (int i = 0; i < 3; i++) {
        pthread_create(&threads[i], NULL, thread_function, &thread_ids[i]);
    }
    
    // 等待线程结束
    for (int i = 0; i < 3; i++) {
        pthread_join(threads[i], NULL);
    }
    
    return 0;
}
```

调试多线程 core 文件：

```gdb
# 查看所有线程
(gdb) info threads

# 切换到特定线程
(gdb) thread 2

# 查看当前线程的调用栈
(gdb) bt

# 查看所有线程的调用栈
(gdb) thread apply all bt
```

### 4.3 内存泄漏检测（结合 Valgrind）

```bash
# 使用Valgrind检测内存问题
valgrind --tool=memcheck --leak-check=full ./program

# Valgrind会报告：
# - 内存泄漏
# - 无效内存访问
# - 未初始化内存使用
# - 内存管理错误
```

## 5. 预防和最佳实践

### 5.1 编译时启用安全检查

```bash
# 启用各种安全检查
gcc -g -O2 \
    -fsanitize=address \
    -fsanitize=undefined \
    -fstack-protector-all \
    -D_FORTIFY_SOURCE=2 \
    -Wformat-security \
    -Werror \
    program.c -o program
```

### 5.2 代码中的预防措施

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 安全的字符串复制
char* safe_strcpy(char *dest, size_t dest_size, const char *src) {
    if (!dest || !src || dest_size == 0) {
        return NULL;
    }
    
    size_t src_len = strlen(src);
    if (src_len >= dest_size) {
        return NULL;  // 目标缓冲区太小
    }
    
    strcpy(dest, src);
    return dest;
}

// 安全的内存访问
int safe_array_access(int *array, size_t size, size_t index) {
    if (!array || index >= size) {
        fprintf(stderr, "数组访问越界: index=%zu, size=%zu\n", index, size);
        return -1;
    }
    return array[index];
}

// 智能指针模式（C语言模拟）
typedef struct {
    void *ptr;
    size_t size;
} SafePointer;

SafePointer* safe_malloc(size_t size) {
    SafePointer *sp = malloc(sizeof(SafePointer));
    if (!sp) return NULL;
    
    sp->ptr = malloc(size);
    sp->size = size;
    
    if (!sp->ptr) {
        free(sp);
        return NULL;
    }
    
    return sp;
}

void safe_free(SafePointer *sp) {
    if (sp) {
        if (sp->ptr) {
            free(sp->ptr);
            sp->ptr = NULL;
        }
        free(sp);
    }
}
```

## 6. 常见问题排查

### 6.1 Core 文件未生成

```bash
# 检查core dump是否启用
ulimit -c

# 检查文件系统是否有足够空间
df -h

# 检查目录权限
ls -ld /path/to/core/directory

# 检查core_pattern设置
cat /proc/sys/kernel/core_pattern

# 检查系统日志
dmesg | grep -i core
```

### 6.2 GDB 无法加载符号

```gdb
# 确保程序编译时包含调试信息
(gdb) info sources
(gdb) info functions

# 如果没有符号信息，重新编译
gcc -g -o program program.c
```

通过以上方法，您可以有效地调试 C/C++ 程序的内核崩溃问题，快速定位和修复程序中的内存错误。
