---
dateCreated: 2025-03-10
dateModified: 2025-06-02
category: Note
---
# 讲义内容

对应学习讲义预学习“完成 PA 1”1 节和 PA 1 部分。

- [x] 在 [FCEUX](https://github.com/NJU-ProjectN/fceux-am) 下运行 ROM
- [ ] 检查画面、按键、声音
- [ ] 阅读框架代码

> [!note] 使用 ccache 加速编译过程|运行 make 时通过多 CPU 加速
> `ccache` 是一个 `compiler cache`。通过配置更改 `gcc` 为 `ccache`。` ccache ` 跳过完全重复的编译过程。如果和多线程编译共同使用，编译速度还能进一步加快!
> 通过 `lscpu` 命令来查询你的系统中有多少个 CPU，然后在运行 `make` 的时候添加一个 `-j?` 的参数, 其中 `?` 为 CPU 数量。为了查看编译加速的效果，你可以在编译的命令前面添加 `time` 命令，它将会对紧跟在其后的命令的执行时间进行统计。

## 环境

PA 的目的是要实现 NEMU, 一款经过简化的全系统模拟器.

基于 riscv 32 ISA：

|              |                                                                                                        |
| ------------ | ------------------------------------------------------------------------------------------------------ |
| riscv32 (64) | The RISC-V Instruction Set Manual[ABI for riscv](https://github.com/riscv-non-isa/riscv-elf-psabi-doc) |

## 认识到程序是个状态机

这是一段利用寄存器计算 1 到 100 的和的汇编程序。

```assembly
// PC: instruction    | // label: statement
0: mov  r1, 0         |  pc0: r1 = 0;
1: mov  r2, 0         |  pc1: r2 = 0;
2: addi r2, r2, 1     |  pc2: r2 = r2 + 1;
3: add  r1, r1, r2    |  pc3: r1 = r1 + r2;
4: blt  r2, 100, 2    |  pc4: if (r2 < 100) goto pc2;   // branch if less than
5: jmp 5              |  pc5: goto pc5;
```

以存储程序为核心的 " 图灵机 "(Turing Machine, TRM) 模型就是计算机的核心思想，而程序就是一个状态机。
