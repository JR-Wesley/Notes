---
dateCreated: 2025-03-10
dateModified: 2025-05-24
---
# 讲义内容
- [ ] 在 [FCEUX](https://github.com/NJU-ProjectN/fceux-am) 下运行 ROM

> [!note] 使用 ccache 加速编译过程|运行 make 时通过多 CPU 加速
> `ccache` 是一个 `compiler cache`

## 环境部署

基于 riscv 32 ISA：

|              |                                                                                                        |
| ------------ | ------------------------------------------------------------------------------------------------------ |
| riscv32 (64) | The RISC-V Instruction Set Manual[ABI for riscv](https://github.com/riscv-non-isa/riscv-elf-psabi-doc) |

这时一段利用寄存器计算 1 到 100 的和的汇编程序。

```assembly
// PC: instruction    | // label: statement
0: mov  r1, 0         |  pc0: r1 = 0;
1: mov  r2, 0         |  pc1: r2 = 0;
2: addi r2, r2, 1     |  pc2: r2 = r2 + 1;
3: add  r1, r1, r2    |  pc3: r1 = r1 + r2;
4: blt  r2, 100, 2    |  pc4: if (r2 < 100) goto pc2;   // branch if less than
5: jmp 5              |  pc5: goto pc5;
```

以存储程序为核心的图灵机模型就是计算机的核心思想，而程序就是一个状态机。

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

NEMU 主要由 4 个模块构成：monitor, CPU, memory, 设备。

Monitor (监视器) 模块是为了方便地监控客户计算机的运行状态而引入的. 它除了负责与 GNU/Linux 进行交互 (例如读入客户程序) 之外, 还带有调试器的功能, 为 NEMU 的调试提供了方便的途径. 从概念上来说, monitor 并不属于一个计算机的必要组成部分, 但对 NEMU 来说, 它是必要的基础设施. 如果缺少 monitor 模块, 对 NEMU 的调试将会变得十分困难.

<a href=" https://ysyx.oscc.cc/docs/ics-pa/nemu-isa-api.html#%E5%AF%84%E5%AD%98%E5%99%A8%E7%9B%B8%E5%85%B3">NEMU ISA API</a>

NEMU 中的配置系统位于 `nemu/tools/kconfig`, 它来源于 GNU/Linux 项目中的 kconfig。目前我们只需要关心配置系统生成的如下文件:

- `nemu/include/generated/autoconf.h`, 阅读 C 代码时使用
- `nemu/include/config/auto.conf`, 阅读 Makefile 时使用
