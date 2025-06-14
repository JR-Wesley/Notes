---
dateCreated: 2023-08-02
dateModified: 2025-06-03
---

<a href=" https://www.gnu.org/software/make/manual/">GNU Make Manual 官方文档</a>

<a href=" https://makefiletutorial.com/">Makefile Tutorial</a>；<a href=" https://cppcheatsheet.com/notes/c_make.html">Makefile cheatsheet</a>

# 介绍

makefile 定义了一系列规则来指定编译规则，make 是解释 makefile 中指令的命令工具。例子：来自 GNU 的 make 手册，指导编译、链接。

1. 如果这个工程没有编译过，那么我们的所有 C 文件都要编译并被链接。
2. 如果这个工程的某几个 C 文件被修改，那么我们只编译被修改的 C 文件，并链接目标程。
3. 如果这个工程的头文件被改变了，那么我们需要编译引用了这几个头文件的 C 文件，并链接目标程序。

## Makefile 规则
**核心概念**
   - **目标（Target）**：需要生成的文件或执行的命令标签。
   - **依赖（Dependencies/prerequisites）**：目标构建所需的文件或其他目标。
   - **命令（Recipe）**：生成目标的具体 Shell 命令。
   - **规则（Rule）**：目标、依赖和命令的组合，格式为：

```makefile
target ... : prerequisites ...
	command ...
```

其中包含文件依赖关系，即 target 目标文件依赖于 prerequisites 中的文件，生成规则定义在 command 中，也即 prerequisites 中若有一个以上文件比 target 文件要新的话，command 定义命令就被执行。

```ma
objects = main.o kbd.o command.o display.o \
insert.o search.o files.o utils.o

# 必须以 tab 开头
edit : $(objects)
	cc -o edit $(objects)

# 隐式规则
$(objects) : defs.h
kbd.o command.o files.o : command.h
display.o insert.o search.o files.o : buffer.h
	...

# 伪目标
.PHONY = clean
clean :
	rm edit main.o $(objects)
```

make 找第一个目标文件 target，作为最终目标文件。make 会一层层地寻找文件依赖关系，直到编译出第一个目标文件，若出现错误，则直接退出报错。

## Makefile 的组成

1. 显示规则：说明如何生成一个或多个目标文件
2. 隐晦规则：make 有自动推导功能，可以简略地写 Makefile
3. 变量定义：变量一般为字符串
4. 文件指示：在一个 Makefile 中引用另一个 Makefile，类似 include；根据某些情况指定 Makefile 有效部分，类似 if；定义一个多行命令
5. 注释：行注释#

|     |                                             |
| --- | ------------------------------------------- |
| `-` | Ignore errors                               |
| `@` | Don’t print command                         |
| `+` | Run even if Make is in ‘don’t execute’ mode |

## 特殊目标

### **1. `.PHONY`：声明伪目标**

告诉 Make 某个目标**不对应实际文件**，而是代表一组命令的名称。即使存在同名文件，Make 也会强制执行对应的命令。

```makefile
.PHONY: 目标1 目标2 …
```

```makefile
.PHONY: clean all

all: program
clean:
    rm -f program *.o
```

- `all` 和 `clean` 是伪目标，不对应实际文件。
- 执行 `make clean` 时，无论是否存在名为 `clean` 的文件，都会执行 `rm` 命令。
- **避免冲突**：若目录中存在名为 `clean` 的文件，`make clean` 会认为目标已更新，导致命令不执行。
- **提高效率**：伪目标不检查文件时间戳，直接执行命令。

### **2. `.DEFAULT_GOAL`：指定默认目标**

当用户直接运行 `make` 而不指定目标时，明确告诉 Make 应该执行哪个目标。

```makefile
.DEFAULT_GOAL: 目标名称
```

```makefile
.DEFAULT_GOAL: all

all: program
    @echo "Building program…"
```

- 直接执行 `make` 时，等同于执行 `make all`。

传统上，Make 默认执行 Makefile 中**第一个**目标。但使用 `.DEFAULT_GOAL` 可以：

- 显式指定默认目标，提高可读性。
- 将默认目标放在文件任意位置（不必是第一个）。

```makefile
.DEFAULT_GOAL: all
.PHONY: all clean

all: program
    @echo "Building program…"

clean:
    @echo "Cleaning up…"
    rm -f program *.o
```

- 直接运行 `make` 时，默认执行 `all`。
- `all` 和 `clean` 始终被视为未更新，确保命令执行。

### **注意事项**

1. **兼容性**
    `.DEFAULT_GOAL` 是 GNU Make 的特性，部分旧版 Make 可能不支持。传统项目通常通过将 `all` 作为第一个目标实现相同效果。

2. **伪目标命名**
    伪目标应避免与实际文件名冲突，常见命名如：`all`, `clean`, `install`, `test`, `check` 等。

3. **执行顺序**
    `.DEFAULT_GOAL` 的声明位置不影响默认目标的执行顺序，它只决定 “哪个目标是默认的”。

### **总结**

- **`.PHONY`**：确保目标命令始终执行，避免与文件冲突。
- **`.DEFAULT_GOAL`**：明确指定默认目标，增强 Makefile 的可维护性。

## Make 运行

make 命令执行后有三个推出码：

- 0 - 成功执行
- 1 - make 运行出现错误，返回 1.
- 2 - 如果使用了 `-q` 选项，并且 make 使得一些目标不需要更新
make 寻找默认的 Makefile 执行，也可以指定文件 `make -f xx.mk`，指定目标。

## 检查规则
- `-n --just-print --dry-run --recon` 不执行，只打印，不管目标是否更新，把规则和连带规则下的命令打印不执行。
- `-t --touch` 把目标文件的时间更新，但不更改目标文件。make 假装编译目标，但不真正编译，只是把目标变成已编译的状态。
- `-q --question` 找目标，如果目标存在则不输出也不编译；若不存在，则打印出错信息。
- `-W <file> --what-if=<file> --assume-new=<file> --new-file=<file>` 这个参数需要指定一个文件。一般是是源文件 (或依赖文件)，Make 会根据规则推导来运行依赖于这个文件的命令，一般来说，可以和 `-n` 参数一同使用，来查看这个依赖文件所发生的规则命令。
- `-b -m` 忽视版本兼容性
- `-B --always-make` 认为所有目标都需要更新
- `-C <dir> --directory=<dir>` 指定读取 makefile 目录，如果有多个，则前后叠加。
- `--debug[=<options>]`
输出 make 的调试信息。它有几种不同的级别可供选择，如果没有参数，那就是输出最简单的调试信息。下面是 `<options>` 的取值:
a — 也就是 all,输出所有的调试信息。(会非常的多)
b —— 也就是 basic,只输出简单的调试信息。即输出不需要重编译的目标。
v —— 也就是 verbose, 在 b 选项的级别之上。输出的信息包括哪个 makefile 被解析，不需要被重编译的依赖文件 (或是依赖目标) 等。
i —— 也就是 implicit，输出所以的隐含规则。
j —— 也就是 jobs,输出执行规则中命令的详细信息,如命令的 PID、返回码等。
m —— 也就是 makefile,输出 make 读取 makefile,更新 makefile,执行 makefile 的信息。
- `-d --debug=a`
- `-e --environment-overrides` 环境变量覆盖 makefile 中定义的变量值
- `-f=<file> --file=<file> --makefile=<file>` 指定需要执行的 makefile
- `-h --help`
- `-i --ignore-errors` 执行时忽视所有错误
- `-I <dir> --include-dir=<dir>` 指定一个倍包含 makefile 的搜索目标，可以有多个
- `-j [<jobsnum] --jobs[=<jobsnum>]` 指定同时运行命令的个数，加速运行
- `-k --keep-going` 出错了不停止
- `-l <load> --load-average[=<load>] --max-load[=<load>]` 指定 make 运行命令的负载
- `-o <file> --old-file=<file> --assume-old=<file>` 不重新生成指定 file，即使这个目标依赖文件新于它。
- `-p --print-data-base` 输出 makefile 所有数据，包括所有规则和变量。
- `-q --question` 不运行命令也不输出，仅检查指定目标是否需要更新，0 为需要，2 为有错误。
- `-r --no-builtin-rules` 进制使用任何隐含规则
- `-R --no-builtin-variables` 进制 make 使用任何作用于变量上的隐含规则
- `-s --silent --quiet` 运行时不输出命令的输出
- `-S --no-keep-going --stop` 取消 `-k` 选项
- `-w --print-directory` 输出运行 makefile 之前后之后的信息
- `--no-print-directory` 禁止 `-w` 选项
- `-W <file> --what-if=<file> --new-file=<file> --assume-file=<file>` 假定目标 file 需要更新
- `--warn-undefined-variables` 只要 make 发现未定义的变量，输出警告。

# 规则

规则包含**依赖关系和生成目标的方法**，Makefile 只有一个最终目标，一般第一个目标为最终目标。

## 通配符

make 支持三种 `*, ?, […]`，和 UNIX 的 B-Shell 相同。

## 自动生成依赖

大多数 C/C++ 编译器都支持一个“-M”选项，即自动寻找源文件中包含的头文件，生成依赖关系。不过使用 GNU 的 C/C++ 编译器，需要用“-MM”参数，不然，“-M”参数会把一些标准库的头文件也包含进来。

```makefile
cc -MM main.c
main.o : main.c defs.h
```

GNU 组织建议把编译器为每一个源文件的自动生成的依赖关系放到一个文件中，为每一个“name. c”的文件都生成一个“name. d”的 Makefile 文件。[. d] 文件中就存放对应 [. c] 文件的依赖关系。于是，我们可以写出 [. c] 文件和 [. d] 文件的依赖关系，并让 make 自动更新或自成 [. d] 文件，并把其包含在我们的主 Makefile 中。如下：

```makefile
%.d: %.c
	@mkdir -p $(dir $@); \
	rm -f $@; \
	$(CC) -MM $< >$@.tmp; \
	sed 's,\($*\)\.o[ :]*,1.o $@ : ,g' < $@.tmp > $@; \
	rm -f $@.tmp
	
	# $@.xxxx 表示随即编号，这里$@.tmp可以任意替代
	
main.o main.d: main.c defs.h
```

这个规则的意义是，所有 `[.d]` 依赖于 `[.c]`，首先删除目标，然后为每个目标 `$<` 生成用 `$@` 命名的文件。然后在编译器生成的依赖中加入 `[.d]`，`[.d]` 文件会自动更新、生成。使用 `include` 添加其他文件。

```makefile
sources = foo.c bar.c
include $(sources:.c=.d)
```

`$(@:.o=.d)` 是一个**变量替换表达式**，用于将目标文件（`.o`）的名称转换为对应的依赖文件（`.d`）的名称。

- **`$@`**：自动变量，表示当前规则的目标文件（如 `build/main.o`）。
- **`:%.o=%.d`**：替换模式，将 `.o` 后缀替换为 `.d`。
**示例**：

- 若 `$@` 为 `build/main.o`，则 `$(@:.o=.d)` 会变为 `build/main.d`。
在编译 C/C++ 代码时，`.d` 文件用于记录源文件的头文件依赖关系，确保头文件修改时能触发重新编译。可以 **使用 `patsubst` 替代**

## 模式规则

模式规则类似一个一般的规则，只是规则中，目标定义需要 `%` 字符，它可以表达一个或多个任意字符。依赖中也可以包含。

```makefile
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
```

# 命令

每条规则中的命令和操作系统 Shell 的命令行是一致的。make 的 命 令 默 认 是 被“/bin/sh”——UNIX 的标准 Shell 解释执行的。除非特别指定一个其它的 Shell。

make 会按顺序执行命令，每条命令的开头必须以 [Tab] 键开头，除非命令是紧跟在依赖规则后面的分号后的。在命令行之间中的空格或是空行会被忽略，但是如果该空格或空行是以 Tab 键开头的，make 会认为其是一个空命令。`make` 针对每条命令，都会创建一个独立的 Shell 环境。如果要让上一条命令的结果应用在下一条命令时，应该把这两条命令写在一行上，使用分号分隔这两条命令。

## Define

为相同的命令序列定义一个变量。`define` 的第一行执行程序、第二行更改名称。

```makefile
define run-yacc
yacc $(firstword $^)
mv y.tab.c $@
endef

foo.c : foo.y
	$(run-yacc)
```

## 条件

```makefile
ifeq ($(CC), gcc)
…
else
…
endif
```

使用 `ifeq ifneq` 判断。

`ifdef ` 判断变量是否为空，

- `$(if CONDITION, THEN-PART, ELSE-PART)` 是 Makefile 的条件函数。

# 变量

变量的命名字可以包含字符、数字、下划线 (可以是数字开头)，但不应该含有“:”、“#”、“=”或是空字符 (空格、回车等)。变量是大小写敏感的。还有一些特殊的自动化变量。

变量声明需要赋初值，使用时加上 `$`，最好用小括号或花括号包括。如果要使用 `$` 字符，需要用 `$$` 表示。变量会在使用它的地方精确地展开

## 定义
1. `a=`，变量可以嵌套，而且没有定义顺序

```makefile
foo = $(bar)
bar = $(ugh)
ugh = Huh?
all:
	echo $(foo)

> Huh?
```

1. 使用 `:=` 立即赋值，如果使用了其他变量必须已经定义。
2. `?=` 表示如果没有定义则使用这个定义，否则不执行
3. `+=` 可以给变量追加值。若变量未定义，则 `+=` 变为 `a=`，若用 `a=` 定义，则继承；若用 `:=` 定义，则同样 `:=` 赋值
4. `define ` 定义，变量的值可以包含函数、命令、文字, 或是其它变量。而且命令同样需要 `Tab` 开头。
5. `!=` 右边为一条 shell 命令，返回赋值。如 `var != echo "hello"`。
6. 如果有变量是通常 make 的命令行参数设置的，那么 Makefile 中对这个变量的赋值会被忽略。如果你想在 Makefile 中设置这类参数的值, 那么, 你可以使用“override”指示符。

```makefile
override <var> = <val>
override <var> := <val>
override <var> += <val>
override define foo
bar
endef
```

## 高级用法
1. 替换
`$(var:a=b)`，将变量 `var` 中所有以 `a` 字符结尾的 `a` 替换成 `b`
`$(var:%.o=%.*c*)` 使用静态模式，要求模式中有一个 `%` 匹配。
2. 变量可以看成变量

```makefile
x =	$(y)
y = z
z = hello
a := $($(x))
```

这里替换后 `a:=$($(y))=$(z)=hello`。

这种替换可以结合函数、字符的替换，

## 环境变量

make 运行时的系统环境变量可以在 make 开始运行时被载入 Makefile 中，但是若 Makefile 已经定义类这个变量，或者这个变量由 make 命令行带入，系统的环境变量的值将被覆盖。（make 指定了 `-e`，系统环境变量会覆盖 Makefile）

当 make 嵌套调用时，上层 Makefile 定义的变量会以系统环境变量的方式传递给下层 Makefile。默认只有命令行设置的变量会被传递。定义在文件中的变量，向下层 Makefile 传递，需要使用 export 关键字声明。

## 自动化变量
- `$@` 规则中的目标文件集
- `$%` 仅当目标是函数库文件时，表示规则中的目标成员名
- `$<` 依赖目标中的第一个目标名
- `$?` 所有比目标新的依赖目标集合
- `$^` 所有依赖目标的集合，去除重复目标
- `$+` 类似 `$^` 也是依赖目标集合，不去除重复的目标
- `$*` 表示目标模式中 `%` 及其之前的部分
- `$(@D) $(@F)` 表示 `$@` 的目录部分和文件部分，如 `dir/foo.o` 中分别是 `dir foo.o`。同样有 `$(*D) $(%D) $(<D) $(^D) $(+D) $(?D)`

# 函数

调用：`(<func> <arg>)`

## 字符串处理
1. `$(wildcard *.c)` 获取所有 `.c` 文件
2. `$(subst <from>, <to>, <text>)` 把字符串中进行替换，返回被替换的字符串。
3. `$(pathsubst <pattern>, <replacement>, <text>)` 查找并替换。可以包含 `%` 通配符，如果替换前后都包含，则匹配的部分不变。
4. `$(strip <string>)` 去除开头结尾空格。
5. `$(findstring <find>, <in>)` 查找字符串，入如果找到则返回，否则返回空。
6. `$(filter <pattern …>, <text>)` 以模式过滤，保留符合模式的单词
7. `$(filter-out <pattern …>, <text>)` 反过滤，去除符合模式的单词
8. `$(sort <list>)` 升序排列，并且去掉相同的单词
9. `$(word <n>, <text>)` 取单词中第 n 个单词（从 1 开始），若超出索引返回空
10. `$(wordlist <s>, <e>, <text>)` 从字符串中取出 s 到 e 的子列表
11. `$(words <text>)` 统计单词个数
12. `$(firstword <text>)` 取首个单词，也可以用 word 实现

## 文件名操作

下面的每个参数字符串都会当成一个或一系列文件名

1. `$(dir <names…>)` 取目录名
2. `$(notdir <names…>)` 取文件名，非目录部分
3. `$(suffix <names…>)` 取文件名后缀
4. `$(basename <names…>)` 取出各个文件名前缀，即取出后缀
5. `$(addsuffix <suffix>, <names…>)` 添加后缀
6. `$(addprefix <prefix>, <names…>)` 添加前缀
7. `$(join <list1>, <list2>)` 把两个字符串依次添加

## 循环
1. `$(foreach <var>, <list>, <text>)` 把 `list` 中的单词逐一取出放到 `var` 指定变量，然后执行 `text`，返回的字符串以空格分隔。注意 `var` 是一个临时变量
2. `$(if <condition>, <then-part>, <else-part>)` 返回两个部分

## 其它
- `$(call <expr>, <param1>, <param2>,…)` 用来创建新的参数化的函数。`expr` 读取变量如 `$(1), $(2)`，返回值是 `call` 的返回值。
- `$(origin <var>)` 返回变量的来源
- `$(shell <expr>)` 执行 shell 命令
- `$(error <text>)` 产生错误，返回错误信息
- `$(warning <text>)` 输出错误信息，并不退出
- `$(eval <text>)` 允许此函数新定义一个 makefile 下的结构，包含变量、目标、隐式或者显示的规则。eval 函数的参数会被展开，然后再由 makeifle 进行解析。也就是说，eval 函数会被 make 解析两次，第一次是对 eval 函数的解析，第二次是 make 对 eval 参数的解析。
**示例：**

```makefile
1 define func
2 $1:
3     cc $2 -o $$@
4 endef
5 
6 $(eval $(call func,main,main.c))
```

执行结果：

```shell
cc main.c -o main
```
