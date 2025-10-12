---
tags:
  - Tool
---

# Verilator

`Verilator` is a tool that compiles `Verilog` and `SystemVerilog` sources to highly optimized (and optionally multithreaded) cycle-accurate `C++` or `SystemC` code. The converted modules can be instantiated and used in a C++ or a SystemC testbench, for verification and/or modelling purposes.

More information can be found at [the official Verilator website](https://www.veripool.org/verilator/) and [the official manual](https://verilator.org/guide/latest/).

Verilator is a [cycle-based](https://www.asic-world.com/verilog/verifaq3.html) simulator, which means it does not evaluate time within a single clock cycle, and does not simulate exact circuit timing. Instead, the circuit state is typically evaluated once per clock-cycle, so any intra-period glitches cannot be observed, and timed signal delays are not supported. This has both benefits and drawbacks when comparing Verilator to other simulators.

# IC/embed 开发

## iverilog/ Gtkwave

[5.1 iverilog · FPGA使用笔记 · 看云 (kancloud.cn)](https://www.kancloud.cn/dlover/fpga/1327817)

| 选项              | 说明          |
| --------------- | ----------- |
| -D macro[=def ] | 定义宏         |
| -I incdir       | 等同于 -incdir |
| -o filename     | 指定输出的可执行文件名 |
| -s topmodule    | 等同于 -top    |
| -y libdir       | 等同于 -y      |
|                 |             |

[verilog - 伊卡洛斯 Verilog 警告 $readmemh : Standard inconsistency, 以下 1364-2005 - IT工具网 (coder.work)](https://www.coder.work/article/8352598#google_vignette)

## 无法处理单元素 Array

<a href="https://github.com/steveicarus/iverilog/pull/1115">iverilog 库中的 pull</a>

## Wavedrom

时序波形绘制

## Voga

综合性串口调试工具

## Modelsim (linux 安装失败)

[Ubuntu 20.04 LTS安装Modelsim SE 2020.4_modelsim ubuntu-CSDN博客](https://blog.csdn.net/weixin_43245577/article/details/140839616)

安装 32 位支持库

将 apt-get install lib32ncurses5 改为 apt-get install lib32ncurses5-dev

https://blog.csdn.net/m0_64037204/article/details/132336249
