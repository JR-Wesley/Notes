---
dateCreated: 2023-07-31
dateModified: 2025-05-24
---

https://vim.wxnacy.com/#docs/get_started

https://www.runoob.com/linux/linux-vim.html

https://csguide.cn/

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

## 文件系统

用户根目录是:	/home/usrname，Linux 为每个用户都创建了根目录

| 名          | 内容                                                         |
| ----------- | ------------------------------------------------------------ |
| /bin        | 存储一些二进制可执行文件，/usr/bin 也存放了基于用户的命令文件 |
| /sbin       | 存储了系统命令，/usr/sbin 也存储了系统命令                    |
| /root       | 超级用户 root 的根目录文件                                     |
| /home       | 普通用户默认目录，该目录下每个用户都有一个以用户名命名的文件夹 |
| /boot       | 存放 Ubuntu 系统内核和系统启动文件                             |
| /mnt        | 通常包括系统引导后被挂在的文件系统的挂载点                   |
| /dev        | 存放设备文件                                                 |
| /etc        | 存放系统管理所需的配置文件和目录                             |
| /lib        | 保存系统程序所需的库文件，/usr/lib 下存放了一些用于普通用户的库文件 |
| /lost+found | 一般为空，当系统非正常关机后，这里会保存一些零散文件         |
| /var        | 存储一些不断变化的文件，比如日志                             |
| /usr        | 包括与系统用户直接有关的文件和目录，比如应用程序和所需的库文件 |
| /media      | 存放 Ubuntu 系统自动挂载的设备文件                             |
| /proc       | 虚拟目录，不实际存在磁盘上，用来保存系统信息和进程信息       |
| /tmp        | 存储系统和用户的临时文件，该文件夹对所有用户都提供读写权限   |
| /opt        | 可选文件和程序的存放目录                                     |
| /sys        | 系统设备和文件层次结构，并向用户提供详细的内核信息           |
|             |                                                              |
|             |                                                              |

文件操作命令

```shell
touch [参数][文件名]	创建新文件
-a	只更改存取事件
-c	不建立任何文件
-d<日期>	使用指定的日期
-t<时间>	使用指定的时间

-mkdir [参数][文件夹名目录名]	文件夹创建
-p	若创建的目录上层目录还未创建，则仪器创建上层目录

-rm	[参数][目的文件或文件夹目录名] 文件及目录删除名
-d	直接把要删除的目录的硬链接数据删成0，删除该目录
-f	强制删除文件或文件夹（目录）
-i	删除文件或文件夹（目录）前先询问用户
-r	递归删除，指定文件夹（目录）下的所有文件和子文件都删除
-v	显示删除过程

-rmdir

-cp
```

磁盘管理

## Tee

```shell
tee [OPTION]... [FILE]...
# 从标准输入中复制到每一个文件，并输出到标准输出
```

### Q1、如何在 Linux 上使用这个命令？

假设因为某些原因，你正在使用 `ping` 命令。

```text
ping google.com
```

然后同时，你想要输出的信息也同时能写入文件。这个时候，`tee` 命令就有其用武之地了。

```text
ping google.com | tee output.txt
```

这个输出内容不仅被写入 `output.txt` 文件，也被显示在标准输出中。

### Q2、如何确保 Tee 命令追加信息到文件中？

默认情况下，在同一个文件下再次使用 `tee` 命令会覆盖之前的信息。如果你想的话，可以通过 `-a` 命令选项改变默认设置。

```text
[command] | tee -a [file]
```

基本上，`-a` 选项强制 `tee` 命令追加信息到文件。

### **Q3、如何让 Tee 写入多个文件？**

这非常之简单。你仅仅只需要写明文件名即可。

```text
[command] | tee [file1] [file2] [file3]
```

比如：

```text
ping google.com | tee output1.txt output2.txt output3.txt
```

### **Q4. 如何让 Tee 命令的输出内容直接作为另一个命令的输入内容？**

使用 `tee` 命令，你不仅可以将输出内容写入文件，还可以把输出内容作为另一个命令的输入内容。比如说，下面的命令不仅会将文件名存入 `output.txt` 文件中，还会通过 `wc` 命令让你知道输入到 `output.txt` 中的文件数目。

```text
ls file* | tee output.txt | wc -l
```

### **Q5. 如何使用 Tee 命令提升文件写入权限？**

假如你使用 [Vim 编辑器](https://link.zhihu.com/?target=https%3A//www.howtoforge.com/vim-basics) 打开文件，并且做了很多更改，然后当你尝试保存修改时，你得到一个报错，让你意识到那是一个 root 所拥有的文件，这意味着你需要使用 `sudo` 权限保存修改。

如此情况下，你可以（在 Vim 内）使用 `tee` 命令来提高权限。

```text
:w !sudo tee %
```

上述命令会向你索要 root 密码，然后就能让你保存修改了。

### **Q6. 如何让 Tee 命令忽视中断？**

`-i` 命令行选项使 `tee` 命令忽视通常由 `ctrl+c` 组合键发起的中断信号（`SIGINT`）。

```text
[command] | tee -i [file]
```

当你想要使用 `ctrl+c` 中断该命令，同时让 `tee` 命令优雅的退出，这个选项尤为实用。

# Petalinux

Petalinux 工具提供在 Xillinx 处理系统上定制、构建、调配嵌入式 Linux 解决方案所需的所有组件。

# 系统配置

在 Linux 系统中，`arm64` 和 `x86_64` 指的是两种不同的处理器架构：

1. **arm64**：这是 ARM 架构的 64 位版本，通常用于嵌入式系统、移动设备（如智能手机和平板电脑）以及一些服务器。ARM 架构以其低功耗和高效能而闻名。
2. **x86_64**：这是 x86 架构的 64 位版本，也称为 AMD64。它是最常见的个人电脑和服务器架构，由 AMD 和 Intel 等公司生产。

### 如何查看你的 PC 是哪个架构

在 Linux 系统中，你可以通过几种不同的方法来查看你的系统是基于哪种架构：

#### 方法 1：使用 `uname` 命令

打开终端，输入以下命令：

```sh
uname -m
```

或者：

```sh
uname --machine
```

这将显示你的系统架构。如果输出是 `x86_64`，那么你的系统是基于 x86 架构的 64 位系统。如果输出是 `aarch64`，那么你的系统是基于 ARM 架构的 64 位系统（在 Linux 中，ARM64 通常表示为 `aarch64`）。

#### 方法 2：查看 `/proc/cpuinfo` 文件

你也可以查看 `/proc/cpuinfo` 文件来获取架构信息：

```sh
cat /proc/cpuinfo
```

在输出中查找 `architecture` 行，它将告诉你系统的架构。

#### 方法 3：使用 `lscpu` 命令

`lscpu` 命令提供了关于 CPU 架构和特性的详细信息：

```sh
lscpu
```

在输出中查找 `Architecture` 行，它将显示你的系统架构。

#### 方法 4：使用 `dmidecode` 命令

`dmidecode` 工具可以读取硬件信息，包括系统架构：

```sh
dmidecode -t baseboard
```

这将显示主板信息，包括系统架构。

通过以上任一方法，你可以确定你的 Linux PC 是基于 `arm64` 还是 `x86_64` 架构。

# Linux 环境工具
## 命令行

apt-get update：是同步/etc/apt/sources. list 和/etc/apt/sources. list. d 中列出的软件源的软件包版本，这样才能获取到最新的软件包。

apt-get upgrade：是更新已安装的所有或者指定软件包，升级之后的版本就是本地索引里的，因此，在执行 upgrade 之前一般要执行 update，这样安装的才是最新的版本。

## 开发管理工具

temux 多窗口

https://www.cnblogs.com/niuben/p/15983908.html

### Tmux
#### 会话与进程

- `session`，会话（任务）
- `windows`，窗口
- `pane`，窗格

命令行的典型使用方式是，打开一个终端窗口（terminal window，以下简称 " 窗口 "），在里面输入命令。**用户与计算机的这种临时的交互，称为一次 " 会话 "（session）** 。会话的一个重要特点是，窗口与其中启动的进程是连在一起的。打开窗口，会话开始；关闭窗口，会话结束，会话内部的进程也会随之终止，不管有没有运行完。

一个典型的例子就是，SSH 登录远程计算机，打开一个远程窗口执行命令。这时，网络突然断线，再次登录的时候，是找不回上一次执行的命令的。因为上一次 SSH 会话已经终止了，里面的进程也随之消失了。为了解决这个问题，会话与窗口可以 " 解绑 "：窗口关闭时，会话并不终止，而是继续运行，等到以后需要的时候，再让会话 " 绑定 " 其他窗口。

对于**window 可以理解为一个工作区，一个窗口**。对于一个 session，可以创建好几个 window，对于每一个 window 窗口，都可以将其分解为几个 pane 小窗格。与 shell 交互的地方是一个 pane，默认这个 pane 占满整个屏幕。

#### Tmux 的作用

**Tmux 就是会话与窗口的 " 解绑 " 工具，将它们彻底分离。**
1. 它允许在单个窗口中，同时访问多个会话。这对于同时运行多个命令行程序很有用。
2. 它可以让新窗口 " 接入 " 已经存在的会话。
3. 它允许每个会话有多个连接窗口，因此可以多人实时共享会话。
4. 它还支持窗口任意的垂直和水平拆分。

类似的终端复用器还有 GNU Screen。Tmux 与它功能相似，但是更易用，也更强大。

#### 基本使用
- 输入 `tmux` 进入
- 按下 `Ctrl+d` 或者显式输入 `exit` 命令退出

---

session

- 新建窗口、指定名称（编号默认从 0 开始）
`tmux new -s <session-name>`
- 分离：退出当前 Tmux 窗口，但是会话和里面的进程仍然在后台运行。
`tmux detach` 或 `ctrl+b d
`tmux ls` 命令可以查看当前所有的 Tmux 会话。
- 重新接入某个已存在的会话。
`tmux attach -t 0 | tmux attach -t <session-name>` 
- 杀死会话
`tmux kill-session -t 0 | tmux kill-session -t <session-name>` 
- 切换会话
`tmux switch -t 0 | tmux switch -t <session-name>` 
- 重命名会话
`tmux rename-session -t 0 <new-name>` 
---

window

- 划分窗格

```shell
# 划分上下两个窗格 $ 
tmux split-window 
# 划分左右两个窗格 $ 
tmux split-window -h
```

- 移动光标

```shell
# 光标切换到上方窗格 $ 
tmux select-pane -U 
# 光标切换到下方窗格 $ 
tmux select-pane -D 
# 光标切换到左边窗格 $ 
tmux select-pane -L 
# 光标切换到右边窗格 $ 
tmux select-pane -R
```

- 交换窗格位置

```shell
# 当前窗格上移 $ 
tmux swap-pane -U 
# 当前窗格下移 $ 
tmux swap-pane -D
```

#### 快捷键

Tmux 窗口有大量的快捷键。所有快捷键都要通过前缀键唤起。默认的前缀键是 `Ctrl+b`，即先按下 `Ctrl+b`，快捷键才会生效。举例来说，帮助命令的快捷键是 `Ctrl+b ?`。它的用法是，在 Tmux 窗口中，先按下 `Ctrl+b`，再按下 `?`，就会显示帮助信息。然后，按下 ESC 键或 `q` 键，就可以退出帮助。

- `<prefix> :`：进入 tmux 命令行，即后面输入的命令为 `tmux <…>`。
- `?` 列出所有快捷键；按 q 返回

会话：

- Ctrl+b d：分离当前会话。
- Ctrl+b s：列出所有会话。
- Ctrl+b $：重命名当前会话。
- `<prefix> D`：选择要断开的会话。
窗口：
- `c` 新建窗口，此时当前窗口会切换至新窗口，不影响原有窗口的状态
- `p` 切换至上一窗口
- `n` 切换至下一窗口
- `w` 窗口列表选择，注意 macOS 下使用 `⌃p` 和 `⌃n` 进行上下选择
- `&` 关闭当前窗口
- `,` 重命名窗口，可以使用中文，重命名后能在 tmux 状态栏更快速的识别窗口 id
- `0` 切换至 0 号窗口，使用其他数字 id 切换至对应窗口
- `f` 根据窗口名搜索选择窗口，可模糊匹配

窗格：

- Ctrl+b %：划分左右两个窗格。
- Ctrl+b "：划分上下两个窗格。
- `Ctrl+b <方向>`：光标切换到其他窗格。是指向要切换到的窗格的方向键，比如切换到下方窗格，就按方向键↓。
- Ctrl+b ;：光标切换到上一个窗格。
- Ctrl+b o：光标切换到下一个窗格。
- Ctrl+b {：当前窗格与上一个窗格交换位置。
- Ctrl+b }：当前窗格与下一个窗格交换位置。
- Ctrl+b Ctrl+o：所有窗格向前移动一个位置，第一个窗格变成最后一个窗格。
- Ctrl+b Alt+o：所有窗格向后移动一个位置，最后一个窗格变成第一个窗格。
- Ctrl+b x：关闭当前窗格。
- Ctrl+b !：将当前窗格拆分为一个独立窗口。
- Ctrl+b z：当前窗格全屏显示，再使用一次会变回原来大小。
- `Ctrl+b Ctrl+ <>`：按箭头方向调整窗格大小。
- Ctrl+b q：显示窗格编号。

#### .tmux/ Oh My Tmux

https://github.com/gpakosz/.tmux

oh my tmux 使用 `C-a` 作为 `C-b` 的第二个前缀键。

- `<prefix>` means you have to either hit Ctrl + a or Ctrl + b
- `<prefix> c` means you have to hit Ctrl + a or Ctrl + b followed by c
- `<prefix> C-c` means you have to hit Ctrl + a or Ctrl + b followed by Ctrl + c

配置

- `<prefix> e` opens the `.local` customization file copy with the editor defined by the `$EDITOR` environment variable (defaults to `vim` when empty)
- `<prefix> r` reloads the configuration
- `C-l` clears both the screen and the tmux history
**session**
- `<prefix> C-c` creates a new session
- `<prefix> C-f` lets you switch to another session by name
**window**
- `<prefix> C-h` and `<prefix> C-l` let you navigate windows (default `<prefix> n` and `<prefix> p` are unbound)
- `<prefix> Tab` brings you to the last active window
**pane**
- `<prefix> -` splits the current pane vertically
- `<prefix> _` splits the current pane horizontally
- `<prefix> h`, `<prefix> j`, `<prefix> k` and `<prefix> l` let you navigate panes ala Vim
- `<prefix> H`, `<prefix> J`, `<prefix> K`, `<prefix> L` let you resize panes
- `<prefix> <` and `<prefix> >` let you swap panes
- `<prefix> +` maximizes the current pane to a new window
- `<prefix> m` toggles mouse mode on or off
- `<prefix> U` launches Urlview (if available)
- `<prefix> F` launches Facebook PathPicker (if available)
- `<prefix> Enter` enters copy-mode
- `<prefix> b` lists the paste-buffers
- `<prefix> p` pastes from the top paste-buffer
- `<prefix> P` lets you choose the paste-buffer to paste from

Additionally, `copy-mode-vi` matches [my own Vim configuration](https://github.com/gpakosz/.vim.git)

Bindings for `copy-mode-vi`:

- `v` begins selection / visual mode
- `C-v` toggles between blockwise visual mode and visual mode
- `H` jumps to the start of line
- `L` jumps to the end of line
- `y` copies the selection to the top paste-buffer
- `Escape` cancels the current operation

#### Oh My Tmux 配置

<a href=" https://zhuanlan.zhihu.com/p/112426848">自己配置美化</a>

安装在 `~/.config/tmux` 下，只更改自己的配置 `.tmux.conf.local`。

bar 中：status_left、status__right、window_status_

### Ranger

<a href=" https://zhuanlan.zhihu.com/p/476289339">ranger 文件查看器</a>

https://blog.virtualfuture.top/posts/file-manager-ranger/

通过改写 `~/.config/ranger` 下的文件，调整设置。

`EDITOR=code` 通过 code 打开。

### 显示系统配置信息 Neofetch

<a href="https://zhuanlan.zhihu.com/p/690584480">neofetch 使用</a>

## Fish Shell

https://www.cnblogs.com/aaroncoding/p/17118251.html

> [!warning]
> fish 终端下使用 conda，会出现环境管理的错误

## Windows Ternimal

https://learn.microsoft.com/zh-cn/windows/terminal/

### Wsl 2 管理

[python - 在 Ubuntu(WSL1 和 WSL2)中显示 matplotlib 图(和其他 GUI) - IT工具网 (coder.work)](https://www.coder.work/article/21956)

[python - Show matplotlib plots (and other GUI) in Ubuntu (WSL1 & WSL2) - Stack Overflow](https://stackoverflow.com/questions/43397162/show-matplotlib-plots-and-other-gui-in-ubuntu-wsl1-wsl2)

解决：[How to use GUI apps in WSL2 (forwarding X server) | Aitor Alonso (aalonso.dev)](https://aalonso.dev/blog/2021/how-to-use-gui-apps-in-wsl2-forwarding-x-server-cdj)

linux ~/. bashrc 修改环境变量 DISPLAY

使用 xtiming 转发图形

迁移、安装

[WSL2 子系统迁移（docker&ubuntu） - 简书 (jianshu.com)](https://www.jianshu.com/p/636f2c47792e) 计算机\HKEY_USERS\S-1-5-21-3044130876-3969153343-4187468819-1001\Software\Microsoft\Windows\CurrentVersion\Lxss\{5177 a 3 b 1-5404-4 d 72-bbcd-8460 fc 2 ff 41 a}

## 指令

```shell
sudo dpkg -i hello.deb
// 安装程序
sudo dpkg -l | grep “a”
// 卸载

tar -zxvf .tar
// 解压缩

# 如果刚创建.sh文件，使用./ 或者绝对路径执行不了时，很可能是因为权限不够。此时你可以使用chmod命令来给shell文件授权。之后就能正常运行了。
chmod +x helloworld.sh

locate
// plocate
```

## Ubuntu 安装

Prob: 网络问题

Sol: 虚拟机网络配置，初始化

Prob: 环境源

Sol: 国内服务器更换

[ubuntu | 镜像站使用帮助 | 清华大学开源软件镜像站 | Tsinghua Open Source Mirror](https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/)

[Ubuntu 20.04换国内源 清华源 阿里源 中科大源 163源 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/421178143)

## 驱动 Libsub

[Linux libusb USB开发（二）—— libusb安装与调试_libusb-1.0.so-CSDN博客](https://blog.csdn.net/jiguangfan/article/details/86492698)
