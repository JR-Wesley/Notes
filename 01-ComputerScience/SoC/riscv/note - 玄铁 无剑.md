---
dateCreated: 2025-05-20
dateModified: 2025-05-25
---
# Installation

参考：https://www.zhihu.com/people/hong-qiang-zi/posts

1. 下载无剑代码，安装必要的工具。
2. 在 toolchain 下载 RISC-V+Toolchain-V1.2.2，解压。
3. 添加环境变量 `TOOL_PATH dir/wujian100_open/riscv_toolchain wujian100_open_PATH dir/wujian100_open`（不需要执行原本的 csh 脚本，根据自己的终端环境来）。
4. `tools/Srec 2 vmem. py ` 修改第一行 `python3`。
5. 执行 `../tools/run_case -sim_tool iverilog ../case/timer/timer_test.c` 完成。
