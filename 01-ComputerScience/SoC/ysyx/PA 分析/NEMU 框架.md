---
dateCreated: 2025-06-02
dateModified: 2025-06-06
---
参考：
https://blog.csdn.net/weixin_63603830/article/details/133933645?spm=1001.2014.3001.5502
https://note.tonycrane.cc/cs/
https://imessiy.github.io/YSYX/PA2/

将在 NEMU 中模拟的计算机称为 " 客户 (guest) 计算机 "，在 NEMU 中运行的程序称为 " 客户程序 "。

```shell
ics2024
├── abstract-machine   # 抽象计算机
├── am-kernels         # 基于抽象计算机开发的应用程序
├── fceux-am           # 红白机模拟器
├── init.sh            # 初始化脚本
├── Makefile           # 用于工程打包提交
├── nemu               # NEMU
└── README.md
```

## NEMU

NEMU 主要由 4 个模块构成：**monitor, CPU, memory, 设备**。

Monitor (监视器) 模块是为了方便地监控客户计算机的运行状态而引入的. 它除了负责与 GNU/Linux 进行交互 (例如读入客户程序) 之外, 还带有调试器的功能, 为 NEMU 的调试提供了方便的途径. 从概念上来说, monitor 并不属于一个计算机的必要组成部分, 但对 NEMU 来说, 它是必要的基础设施. 如果缺少 monitor 模块, 对 NEMU 的调试将会变得十分困难.

> [!note] 系统 shell 配置脚本中加入了环境变量
> `NEMU_HOME AM_HOME NPC_HOME NVBOARD_HOMW`

### 文件组织

```shell
nemu
├── configs                    # 预先提供的一些配置文件
├── include                    # 存放全局使用的头文件
│   ├── common.h               # 公用的头文件
│   ├── config                 # 配置系统生成的头文件, 用于维护配置选项更新的时间戳
│   ├── cpu
│   │   ├── cpu.h
│   │   ├── decode.h           # 译码相关
│   │   ├── difftest.h
│   │   └── ifetch.h           # 取指相关
│   ├── debug.h                # 一些方便调试用的宏
│   ├── device                 # 设备相关
│   ├── difftest-def.h
│   ├── generated
│   │   └── autoconf.h         # 配置系统生成的头文件, 用于根据配置信息定义相关的宏
│   ├── isa.h                  # ISA相关
│   ├── macro.h                # 一些方便的宏定义
│   ├── memory                 # 访问内存相关
│   └── utils.h
├── Kconfig                    # 配置信息管理的规则
├── Makefile                   # Makefile构建脚本
├── README.md
├── resource                   # 一些辅助资源
├── scripts                    # Makefile构建脚本
│   ├── build.mk
│   ├── config.mk
│   ├── git.mk                 # git版本控制相关
│   └── native.mk
├── src                        # 源文件
│   ├── cpu
│   │   └── cpu-exec.c         # 指令执行的主循环
│   ├── device                 # 设备相关
│   ├── engine
│   │   └── interpreter        # 解释器的实现
│   ├── filelist.mk
│   ├── isa                    # ISA相关的实现
│   │   ├── mips32
│   │   ├── riscv32
│   │   ├── riscv64
│   │   └── x86
│   ├── memory                 # 内存访问的实现
│   ├── monitor
│   │   ├── monitor.c
│   │   └── sdb                # 简易调试器
│   │       ├── expr.c         # 表达式求值的实现
│   │       ├── sdb.c          # 简易调试器的命令处理
│   │       └── watchpoint.c   # 监视点的实现
│   ├── nemu-main.c            # 你知道的…
│   └── utils                  # 一些公共的功能
│       ├── log.c              # 日志文件相关
│       ├── rand.c
│       ├── state.c
│       └── timer.c
└── tools                      # 一些工具
    ├── fixdep                 # 依赖修复, 配合配置系统进行使用
    ├── gen-expr
    ├── kconfig                # 配置系统
    ├── kvm-diff
    ├── qemu-diff
    └── spike-diff
```

为了支持不同的 ISA, 框架代码把 NEMU 分成两部分: ISA 无关的基本框架和 ISA 相关的具体实现. NEMU 把 ISA 相关的代码专门放在 `nemu/src/isa/` 目录下, 并通过 `nemu/include/isa.h` 提供 ISA 相关 API 的声明. 这样以后, `nemu/src/isa/` 之外的其它代码就展示了 NEMU 的基本框架. 这样做有两点好处:

- 有助于我们认识不同 ISA 的共同点: 无论是哪种 ISA 的客户计算机, 它们都具有相同的基本框架
- 体现抽象的思想: 框架代码将 ISA 之间的差异抽象成 API, 基本框架会调用这些 API, 从而无需关心 ISA 的具体细节. 如果你将来打算选择一个不同的 ISA 来进行二周目的攻略, 你就能明显体会到抽象的好处了: 基本框架的代码完全不用修改!

<a href=" https://ysyx.oscc.cc/docs/ics-pa/nemu-isa-api.html#%E5%AF%84%E5%AD%98%E5%99%A8%E7%9B%B8%E5%85%B3">NEMU ISA API</a>

### 项目构建
#### 配置系统 Kconfig

NEMU 中的配置系统位于 `nemu/tools/kconfig`, 它来源于 GNU/Linux 项目中的 kconfig, 我们进行了少量简化. kconfig 定义了一套简单的语言, 开发者可以使用这套语言来编写 " 配置描述文件 ". 在 " 配置描述文件 " 中, 开发者可以描述:

- 配置选项的属性, 包括类型, 默认值等
- 不同配置选项之间的关系
- 配置选项的层次关系

#### Make Menuconfig

在 NEMU 项目中, " 配置描述文件 " 的文件名都为 `Kconfig`, 如 `nemu/Kconfig`. 当你键入 `make menuconfig` 的时候, 背后其实发生了若干时间

目前我们只需要关心配置系统生成的如下文件:

- `nemu/include/generated/autoconf.h`, 阅读 C 代码时使用
- `nemu/include/config/auto.conf`, 阅读 Makefile 时使用

# Makefile 解析

通过包含 `nemu/include/config/auto.conf`, 与 kconfig 生成的变量进行关联. 因此在通过 menuconfig 更新配置选项后, Makefile 的行为可能也会有所变化.

通过文件列表 (filelist) 决定最终参与编译的源文件. 在 `nemu/src` 及其子目录下存在一些名为 `filelist.mk` 的文件, 它们会根据 menuconfig 的配置对如下 4 个变量进行维护:

- `SRCS-y` - 参与编译的源文件的候选集合
- `SRCS-BLACKLIST-y` - 不参与编译的源文件的黑名单集合
- `DIRS-y` - 参与编译的目录集合, 该目录下的所有文件都会被加入到 `SRCS-y` 中
- `DIRS-BLACKLIST-y` - 不参与编译的目录集合, 该目录下的所有文件都会被加入到 `SRCS-BLACKLIST-y` 中

## 编译和链接 `make`

键入 `make -nB`, 它会让 `make` 程序以 " 只输出命令但不执行 " 的方式强制构建目标.运行后, 你可以看到很多形如

```
echo + CC src/nemu-main.c
mkdir -p /home/maria/ysyx/ysyx-workbench/nemu/build/obj-riscv32-nemu-interpreter/src/
gcc -O2 -MMD -Wall -Werror -I/home/maria/ysyx/ysyx-workbench/nemu/include -I/home/maria/ysyx/ysyx-workbench/nemu/src/engine/interpreter -I/home/maria/ysyx/ysyx-workbench/nemu/src/isa/riscv32/include -I tools/capstone/repo/include -O2    -DITRACE_COND=true -D__GUEST_ISA__=riscv32 -c -o /home/maria/ysyx/ysyx-workbench/nemu/build/obj-riscv32-nemu-interpreter/src/nemu-main.o src/nemu-main.c
/home/maria/ysyx/ysyx-workbench/nemu/tools/fixdep/build/fixdep  /home/maria/ysyx/ysyx-workbench/nemu/build/obj-riscv32-nemu-interpreter/src/nemu-main.d  /home/maria/ysyx/ysyx-workbench/nemu/build/obj-riscv32-nemu-interpreter/src/nemu-main.o unused >  /home/maria/ysyx/ysyx-workbench/nemu/build/obj-riscv32-nemu-interpreter/src/nemu-main.d.tmp
mv  /home/maria/ysyx/ysyx-workbench/nemu/build/obj-riscv32-nemu-interpreter/src/nemu-main.d.tmp  /home/maria/ysyx/ysyx-workbench/nemu/build/obj-riscv32-nemu-interpreter/src/nemu-main.d

...
flock /home/maria/ysyx/ysyx-workbench/nemu/../.git/ make -C /home/maria/ysyx/ysyx-workbench/nemu/…git_commit MSG=' "compile NEMU"'
sync /home/maria/ysyx/ysyx-workbench/nemu/../.git/
```

### 首先检查该路径是否配置

```
# Sanity check
ifeq ($(wildcard $(NEMU_HOME)/src/nemu-main.c),)
endif
```

`$(wildcard PATTERN) ` 是一个内置函数，用于文件路径的模式匹配。它的核心作用是根据指定的通配符模式，返回当前目录下匹配的文件列表。

**PATTERN**：支持通配符的文件路径模式，常见通配符包括：
- `*`：匹配任意数量（包括零个）的任意字符。
- `?`：匹配单个任意字符。
- `[]`：匹配方括号内指定的任意一个字符（如 `[abc]` 匹配 `a`、`b` 或 `c`）。

在 Makefile 中，`ifeq` 是条件判断的关键字，用于根据变量值或表达式结果执行不同的操作。

```makefile
ifeq (ARG1, ARG2)
    # 当 ARG1 等于 ARG2 时执行的命令
else
    # 否则执行的命令
endif
```

- **`ifeq`**：判断两个参数是否相等。
- **`else`**：可选分支，条件不成立时执行。
- **`endif`**：结束条件判断。

### 导入规则

然后会导入 ` menuconfig` 生成的 `auto. conf` 相关的变量和规则。

动态生成 NEMU 可执行文件的名称，使其包含（使用 `?=`，如果已经定义则不执行）：

- **目标架构**（`GUEST_ISA`）：表明模拟器支持的指令集。
- **后端引擎**（`ENGINE`）：表明模拟器使用的执行引擎类型（解释器或即时编译器）。

这种命名方式便于区分不同配置的 NEMU 二进制文件，例如：

- `riscv64-nemu-interpret`：基于解释器的 RISC-V 模拟器。
- `x86_64-nemu-tcg`：基于 TCG（即时编译）的 x86 模拟器。

`remove_quote = $(patsubst "%",%,$(1))` 将匹配到的 `"xxx"` 替换为 `xxx`，即移除双引号，用于如 `CONFIG_ISA="riscv32"`。

- `patsubst` 是 Makefile 内置函数，用于模式替换，语法为：`$(patsubst 模式, 替换文本, 文本)`。它会将 **文本** 中匹配 **模式** 的部分替换为 **替换文本**。
- **模式匹配**： `"%` 匹配以双引号开头的字符串，`%"` 匹配以双引号结尾的字符串。
	- `%` 在 `patsubst` 中是通配符，表示任意字符串。
	- 因此，`"%` 匹配 `"xxx`，`%"` 匹配 `xxx"`。

### 导入文件

`FILELIST_MK = $(shell find -L ./src -name "filelist.mk")`：如果在 `./src` *目录或其任意子目录*中找到了 `filelist.mk`，`FILELIST_MK` 变量就会被赋值为这个文件的路径。要是没找到，变量就为空。

- `-L` 选项：查找时会跟随符号链接，能处理链接指向的文件。

**include 指令** 会让 Makefile 在当前位置包含另一个 Makefile 文件的内容，就好像这些内容原本就在这里一样。本项目包含：

```shell
./src/device/filelist.mk
./src/utils/filelist.mk
./src/engine/filelist.mk
./src/isa/filelist.mk
./src/filelist.mk
```

// TODO

然后根据各子文件中定义确定最终的源文件。

### 编译选项

### 加入 `config.mk` Menuconfig 相关

### 添加 `native.mk` Nemu 编译指令

`-include $(NEMU_HOME)/../Makefile` 管理 git

## `build.mk` 编译命令

Makefile 的编译规则在 `include $(NEMU_HOME)/scripts/build.mk` 中定义:

`OBJS = $(SRCS:%.c=$(OBJ_DIR)/%.o) $(CXXSRC:%.cc=$(OBJ_DIR)/%.o)` 这行 Makefile 代码的作用是把源文件列表（`SRCS` 和 `CXXSRC`）转换为对应的目标文件列表（`OBJS`），并自动将目标文件放到指定的输出目录 `$(OBJ_DIR)` 中。

```shell
$(OBJ_DIR)/%.o: %.c
  @echo + CC $<
  @mkdir -p $(dir $@)
  @$(CC) $(CFLAGS) -c -o $@ $<
  $(call call_fixdep, $(@:.o=.d), $@)
```

这段 Makefile 代码定义了一个**模式规则**，用于将 `.c` 源文件编译为 `.o` 目标文件，并自动生成依赖文件（`.d`）。以下是详细解析：

```makefile
@echo + CC $<
```

- `@`：禁止打印命令本身，仅显示 `echo` 的内容。
- 效果：编译时显示 `+ CC main.c`。

```makefile
@mkdir -p $(dir $@)
```

- `mkdir -p`：递归创建目录（如 `build/`），避免因目录不存在导致编译失败。`-p` 选项的作用是**递归创建目录**，并且**在目录已存在时不报错**。

```makefile
@$(CC) $(CFLAGS) -c -o $@ $<
```

- `$(CC)`：编译器（如 `gcc`）。
- `$(CFLAGS)`：编译选项（如 `-Wall -O2`）。
- `-c`：只编译不链接。

### 生成依赖

```makefile
$(call call_fixdep, $(@:.o=.d), $@)
```

- `$(@:.o=.d)`：将 `.o` 替换为 `.d`（如 `build/main.d`）。
- `call_fixdep`：调用自定义函数生成依赖文件，记录 `.c` 文件包含的头文件。
例如，`main.d` 可能包含：

```makefile
build/main.o: main.c header.h util.h
```

假设：

- `OBJ_DIR := build`
- `SRCS := main.c util.c`
- `OBJS := build/main.o build/util.o`

执行 `make` 时，此规则会自动处理：

1. `main.c` → `build/main.o`
2. `util.c` → `build/util.o`
3. 生成 `build/main.d` 和 `build/util.d` 记录依赖关系。

这条规则是 Makefile 中编译 C 源文件的**标准写法**，通过模式匹配和自动变量实现了：

- 自动创建输出目录。
- 自动编译所有 `.c` 文件。
- 自动追踪头文件依赖，确保增量编译正确性。

合理使用此规则可大幅简化大型项目的构建配置。

在 `$(NEMU_HOME)/scripts/config.mk` 中定义了一个名为 ` call_fixdep ` 的**函数**，用于生成和处理依赖文件（`. d `）。` call_fixdep ` 的核心作用是：

```shell
define call_fixdep
	@$(FIXDEP) $(1) $(2) unused > $(1).tmp
	@mv $(1).tmp $(1) 
endef
```

1. 调用 `$(FIXDEP)` 工具生成依赖信息。
2. 临时保存结果到 `.tmp` 文件。
3. 将临时文件重命名为最终的依赖文件（如 `main.d`）。

- **`$(1)`**：第一个参数，表示目标依赖文件（如 `build/main.d`）。
- **`$(2)`**：第二个参数，表示目标对象文件（如 `build/main.o`）。

 1. **生成依赖信息**

```makefile
@$(FIXDEP) $(1) $(2) unused > $(1).tmp
```

- **`$(FIXDEP)`**：依赖生成工具（可能是自定义脚本或 GCC 的 `-MM` 选项）。
- **参数传递**：
    - `$(1)`：输出的依赖文件路径。
    - `$(2)`：目标对象文件路径。
    - `unused`：占位参数，某些工具可能需要但实际不使用。
- **输出重定向**：结果先写入临时文件（如 `build/main.d.tmp`）。

1. **替换最终文件**

```makefile
@mv $(1).tmp $(1)
```

- 将临时文件重命名为正式的依赖文件，确保文件原子性更新（避免中途被 Make 读取）。

### **为什么需要临时文件？**

生成的 `.d` 文件通常包含：

makefile

```makefile
build/main.o: main.c header.h util.h
```

- 表示 `main.o` 依赖于 `main.c` 及其包含的所有头文件。

直接写入 `.d` 文件可能导致：

1. **部分更新**：生成过程中 Make 可能读取不完整的依赖文件。
2. **时间戳问题**：即使内容未变，文件修改时间也会更新，触发不必要的重新编译。

通过临时文件确保：

- 依赖文件要么完全生成，要么不存在。
- 内容不变时，时间戳保持一致。

`call_fixdep` 是一个**安全生成依赖文件的函数**，通过临时文件机制避免了生成过程中的竞争问题，确保 Make 能够正确识别文件依赖关系，实现高效的增量编译。

### 链接

```makefile
$(BINARY):: $(OBJS) $(ARCHIVES)    # 目标: 依赖
    @echo + LD $@                  # 打印链接信息
    @$(LD) -o $@ $(OBJS) $(LDFLAGS) $(ARCHIVES) $(LIBS)  # 链接命令
```

- `$(LD)`：链接器（通常是 `ld` 或 `gcc`）。
- `-o $@`：指定输出文件为 `$(BINARY)`。
- `$(OBJS)`：目标文件列表。
- `$(LDFLAGS)`：链接选项（如 `-L/path` 指定库搜索路径）。
- `$(ARCHIVES)`：静态库文件（如 `libdevice.a`）。
- `$(LIBS)`：动态库链接选项（如 `-lm` 链接数学库）。

1. **双冒号 `::` 的作用**

    - 与单冒号 `:` 类似，但允许定义**多个同名规则**。
    - 多个 `$(BINARY)::` 规则会按顺序执行，而单冒号规则会被合并。
2. **依赖关系**

    - `$(BINARY)`：最终输出的可执行文件或库（如 `nemu`）。
    - `$(OBJS)`：目标文件列表（如 `main.o util.o`）。
    - `$(ARCHIVES)`：静态库列表（如 `libdevice.a`）。
3. **自动变量**

    - `$@`：当前目标（即 `$(BINARY)`）。
    - `$(OBJS)` 和 `$(ARCHIVES)`：所有依赖文件。

 **典型应用场景**举例：

```makefile
BINARY := nemu
OBJS := main.o device.o
ARCHIVES := libdevice.a
LDFLAGS := -L./libs
LIBS := -lpthread

$(BINARY):: $(OBJS) $(ARCHIVES)
    @$(LD) -o $@ $(OBJS) $(LDFLAGS) $(ARCHIVES) $(LIBS)
```

```bash
ld -o nemu main.o device.o -L./libs libdevice.a -lpthread
```

##### **注意事项**

1. **链接顺序**
    目标文件和库的顺序很重要，例如：

    - 被依赖的库应放在依赖它的目标文件之后。
    - 静态库（`$(ARCHIVES)`）需在动态库（`$(LIBS)`）之前。
2. **双冒号 `::` vs 单冒号 `:`**

    - 双冒号允许多个规则，但通常在需要**拆分链接步骤**时使用（如先生成部分链接的 `.o`）。
    - 大多数情况下使用单冒号即可。
3. **静态库与动态库**

    - `$(ARCHIVES)`：静态库（`.a`），链接时直接嵌入可执行文件。
    - `$(LIBS)`：动态库（`.so`），运行时动态加载。

这条规则是 Makefile 中**链接阶段的标准写法**，通过组合目标文件和库文件生成最终二进制文件。合理使用此规则可确保链接过程正确处理依赖关系和编译选项。

## Difftest

## 调试 `gdb`

# 以第一个客户程序为例

## 运行 `run`

在 `nemu/` 目录下编译并运行 NEMU：`make run`。对应的 Make 指令为：

```shell
IMG ?=
NEMU_EXEC := $(BINARY) $(ARGS) $(IMG)

run-env: $(BINARY) $(DIFF_REF_SO)

run: run-env
	$(call git_commit, "run NEMU")
	$(NEMU_EXEC)
```

我们已经知道, NEMU 是一个用来执行客户程序的程序, 但客户程序一开始并不存在于客户计算机中. 我们需要将客户程序读入到客户计算机中, 这件事是 monitor 来负责的. 于是 NEMU 在开始运行的时候, 首先会调用 `init_monitor()` 函数 (在 `nemu/src/monitor/monitor.c` 中定义) 来进行一些和 monitor 相关的初始化工作.

```c
// nemu-main.c

int main(int argc, char *argv[]) {
  /* Initialize the monitor. */
#ifdef CONFIG_TARGET_AM
  am_init_monitor();
#else
  init_monitor(argc, argv);
#endif

  /* Start engine. */
  engine_start();

  return is_exit_status_bad();
}
```

### Kconfig 生成的宏与条件编译

我们已经在上文提到过, kconfig 会根据配置选项的结果在 `nemu/include/generated/autoconf.h` 中定义一些形如 `CONFIG_xxx` 的宏, 我们可以在 C 代码中通过条件编译的功能对这些宏进行测试, 来判断是否编译某些代码. 例如, 当 `CONFIG_DEVICE` 这个宏没有定义时, 设备相关的代码就无需进行编译.

为了编写更紧凑的代码, 我们在 `nemu/include/macro.h` 中定义了一些专门用来对宏进行测试的宏. 例如 `IFDEF(CONFIG_DEVICE, init_device());` 表示, 如果定义了 `CONFIG_DEVICE`, 才会调用 `init_device()` 函数; 而 `MUXDEF(CONFIG_TRACE, "ON", "OFF")` 则表示, 如果定义了 `CONFIG_TRACE`, 则预处理结果为 `"ON"` (`"OFF"` 在预处理后会消失), 否则预处理结果为 `"OFF"`.

## Monitor 函数

`parse_args()`, `init_rand()`, `init_log()` 和 `init_mem()`

### Isa

### 寄存器和内存

## 运行客户程序

## 调试用代码

# 简易调试器

# 表达式求值
