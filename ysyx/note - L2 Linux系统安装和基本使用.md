---
dateCreated: 2024-11-19
dateModified: 2025-03-11
banner: "[[../../300-以影像之/Genshin/OST/沉玉沐芳 Jadeite Redolence.jpg]]"
---
# 讲义内容
- [x] 安装 linux，了解用法。
- [x] ysyx 代码安装配置。

<a href=" https://ysyx.oscc.cc/docs/ics-pa/PA0.html">ysyx PA 0</a>, <a href="https://nju-projectn.github.io/ics-pa-gitbook/ics2024/">NJU PA 0</a>

- 其他资源
	- PA 习题课 why @b
	- ICS 理论课袁春风 @MOOC
	- ICS 计算机组成构造，@b
	- OS 课，jyy @b

工具：<a href="https://dir.scmor.com/">镜像和各种搜索引擎</a>

x 86/ riscv 32 (64)/ ISA 无关的手册

<a href=" https://nju-projectn.github.io/ics-pa-gitbook/ics2024/blank.html">杂项各种</a>

> [!warning]
PA 的某些特性会依赖于 64 位平台和图形显示.

## 推荐阅读
- <a href=" https://101.ustclug.org/">中国科学技术大学 Linux 用户协会发起 linux 101</a>
- <a href="总结了很多常用的命令行工具">总结了很多常用的命令行工具 The Art of Command Line</a>
- <a href=" https://missing-semester-cn.github.io/">MIT The Missing Semester of Your CS Education</a>：shell vim 数据整理命令行 git
- <a href=" https://ysyx.oscc.cc/docs/ics-pa/PA0.html">MIT Academic Integrity</a>

## 总结

安装、配置、熟悉使用了 wsl。关于 linux/man/shell/regular expression 还有很多需要学习的东西。

> [!note] 持续学习
[GNU diff format](http://www.gnu.org/software/diffutils/manual/html_node/Unified-Format.html)
<a href=" https://nju-projectn.github.io/ics-pa-gitbook/ics2024/0.4.html">讲义还给出了更多 vim 配置，结合其他资料选择</a>。
[Harley Hahn's Guide to Unix and Linux](http://www.harley.com/books/sg3.html).
[鸟哥的Linux私房菜](http://linux.vbird.org/linux_basic)
> <a href=" https://linuxconfig.org/gdb-debugging-tutorial-for-beginners">GNU 教程、还有其他资料</a>

## Linux 教程

<a href=" https://ysyx.oscc.cc/docs/ics-pa/linux.html#%E5%9C%A8linux%E4%B8%8B%E7%BC%96%E5%86%99hello-world%E7%A8%8B%E5%BA%8F">PA 给出的教程</a>

Different GNU/Linux distribution has different package manager. In Ubuntu, the package manager is called `apt`.

命令行格式：

```shell
命令名称 参数1 参数2 参数3 …
```

`reboot` 为重启命令。

`poweroff` 为关机命令。

`mount` 用于挂载一个文件系统

`umount` 与 `mount` 相反，是卸载一个挂载点，即取消该入口。

`whereis` 用于查找文件、手册等。

`whoami` 显示自身的用户名称，本指令相当于执行 "id -un" 指令。


[note - L2 Linux系统安装和基本使用](note%20-%20L2%20Linux系统安装和基本使用.md)