- [x] PPT
- [x] VIDEO
- [x] DOC
- [ ] 自己搭建

>[!note]
>尝试用编程取代重复劳动（正则表达式/Shell/Vim）

## 提升编译效率
1. 并行编译 -  `make -j $(nproc)`。可以将指令加入环境变量 `export MAKEFLAGS="-j $(nproc)"/alias make = make -j $(nproc) / .bashrc`。
2. 分布式编译 - `icecream`。把源文件分派给其他机器。
3. 编译缓存 - `ccache`。记录 `.c + 编译选项 -> .o` 的关系，如果`.c`之前用相同的选项编译过, 就直接取出`.o`, 跳过`gcc`的编译
## 提升运行效率
1. 使用 `tmux` 管理多个窗口，写个Makefile, 键入 `make run` 自动编译运行。
2. 给编辑器绑定快捷键, 实现 “一键”编译运行
3. 监视源文件, 更新时自动触发指定脚本, 实现保存后 “零键”编译运行，如 `inotifywait`。

## Differential Testing
1. 防御性编程 - `assert`  将预期的正确行为直接写到程序中
2. Differential Testing 核心思想: 对于符合相同规范的两种实现, 给定有定义的相同输入, 两者行为应当一致