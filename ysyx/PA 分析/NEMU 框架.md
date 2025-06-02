---
dateCreated: 2025-06-02
dateModified: 2025-06-02
---

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

为了支持不同的 ISA, 框架代码把 NEMU 分成两部分: ISA 无关的基本框架和 ISA 相关的具体实现. NEMU 把 ISA 相关的代码专门放在 `nemu/src/isa/` 目录下, 并通过 `nemu/include/isa.h` 提供 ISA 相关 API 的声明. 这样以后, `nemu/src/isa/` 之外的其它代码就展示了 NEMU 的基本框架.

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

#### Kconfig 生成宏和条件编译

#### Makefile

通过包含 `nemu/include/config/auto.conf`, 与 kconfig 生成的变量进行关联. 因此在通过 menuconfig 更新配置选项后, Makefile 的行为可能也会有所变化.

通过文件列表 (filelist) 决定最终参与编译的源文件. 在 `nemu/src` 及其子目录下存在一些名为 `filelist.mk` 的文件, 它们会根据 menuconfig 的配置对如下 4 个变量进行维护:

- `SRCS-y` - 参与编译的源文件的候选集合
- `SRCS-BLACKLIST-y` - 不参与编译的源文件的黑名单集合
- `DIRS-y` - 参与编译的目录集合, 该目录下的所有文件都会被加入到 `SRCS-y` 中
- `DIRS-BLACKLIST-y` - 不参与编译的目录集合, 该目录下的所有文件都会被加入到 `SRCS-BLACKLIST-y` 中

#### 编译和链接

Makefile 的编译规则在 `nemu/scripts/build.mk` 中定义:

```shell
$(OBJ_DIR)/%.o: %.c
  @echo + CC $<
  @mkdir -p $(dir $@)
  @$(CC) $(CFLAGS) -c -o $@ $<
  $(call call_fixdep, $(@:.o=.d), $@)
```

### 以第一个客户程序为例

我们已经知道, NEMU 是一个用来执行客户程序的程序, 但客户程序一开始并不存在于客户计算机中. 我们需要将客户程序读入到客户计算机中, 这件事是 monitor 来负责的. 于是 NEMU 在开始运行的时候, 首先会调用 `init_monitor()` 函数 (在 `nemu/src/monitor/monitor.c` 中定义) 来进行一些和 monitor 相关的初始化工作.

### Monitor 函数

#### Isa

#### 寄存器和内存

### 运行客户程序

### 调试用代码

# 简易调试器

# 表达式求值
