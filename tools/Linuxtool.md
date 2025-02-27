---
dateCreated: 2023-07-31
dateModified: 2024-12-06
---
https://vim.wxnacy.com/#docs/get_started

https://www.runoob.com/linux/linux-vim.html

https://csguide.cn/

# 常用操作

```shell
ls *.tar.gz | xargs -n1 tar xzvf

```

用 tar 命令批量解压某个文件夹下所有的 tar.gz 文件

for file in *_1; do

  mv "$file" "${file/_1/:1}"

done

# 程序运行

# Linux

linux 分底层与应用

Ubuntu 密码 Eric zhy72538490

## Shell

```shell
用户名@机器名:~$ ls
command -option [argument]

# 1. ls	显示指定目录下的内容
-a	显示所有文件及子目录，包括"."开头的隐藏文件
-l	显示文件的详细信息
-t	将文件按创建时间排序列出
-A	和-a相同，但不列出".”和".."子目录和父目录
-R	递归列出所有文件，包括子目录的文件
# 参数可任意组合

# 2. cd 目录切换
cd /	进入/目录下
cd /usr	进入"/usr"里面
cd ..		进入上级目录
cd ~		切换到当前用户主目录

# 3. pwd 当前路径显示

# 4. uname 系统信息查看
-r	列出当前系统具体内核版本号
-s	列出系统内核名称
-o	列出系统信息

# 5. clear 清屏

# 6. sudo 切换用户执行身份命令
-h	显示帮助信息
-l	列出当前用户可执行与不可执行的命令
-p	改变询问密码的提示符

# 7. adduser 添加用户命令，需要sudo
-system	添加系统用户
-home DIR	DIR表示用户的主目录路径
-uid ID		ID表示用户的uid
-ingroup GRP	表示用户所属组名


# 8. deluser 删除用户命令，需要sudo
-system				当用户是一个系统用户时才能删除
-remove-home	删除用户主目录
-remove-all-files	删除与用户有关的所有文件
-backup		备份用户信息

# 9. su 切换用户命令
-c -command		执行指定的命令，执行完后回复用户身份
-login	改变用户身份，同时改变工作目录和PATH环境变量
-m		改变用户身份的时候不改变环境变量
-h		显示帮助信息

# 10. cat 显示文件内容命令
# linux下类似记事本gedit
-n 由1开始对所有输出的行进行编号
-b 和n类似，但是不对空白行编号
-s 遇到连续两个空行以上合并成一个空白行

# 11. ifconfig 显示和配置网络属性命令
interface	网络接口，比如eth0
up	开启网络涉笔
down	关闭网络设备
add	IP地址，设置网络IP
netmask add	子网编码

# 12. man 系统帮助

# 13. reboot 系统重启

# 14. poweroff 系统关闭

# 15. install 软件安装

```

## 工具

- APT 下载工具

```shell
sudo apt-get update
sudo apt-get check
sudo apt-get install package-name
sudo apt-get upgrade package-name
sudo apt-get remove package-name

```

- gedit/vi/vim

```shell
i	当前光标字符前，转为输入
I	当前光标字符行首，转为输入
a	当前光标字符后，转为输入
A	当前光标所在行行尾，转为输入
o	当前光标所在行下方，新建一行，转为输入
O	当前光标所在行上方，新建一行，转为输入
s	删除光标所在字符
r	替换光标处字符
```

Ctrl+S 是短暂停止该终端，Ctrl+Q 重新打开

ESC

移动光标指令：

h/i/j/k,nG,n+,n-,Ctrl+f,Ctrl+b,cc,dd,ndd,x,X,nyy,p

底行命令：

```shell
x	保存文件并退出
q	退出
w	保存文档
q!	退出VI/VIM，不保存文档
/	搜索后面的文字
```

## 文件系统

用户根目录是:	/home/usrname，Linux 为每个用户都创建了根目录

```shell
cd /	//进入根目录
```

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

cmv


```

文件压缩和解压

```shell
```

```shell
find [路径][参数][关键字]
-



grep [参数] 关键字 文件列表


```

用户权限系统

Ubuntu 是多用户系统，

- 初次创建的用户。
- root 用户，系统管理员
- 普通用户。

```shell
chmod [参数][文件名、目录名]

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
