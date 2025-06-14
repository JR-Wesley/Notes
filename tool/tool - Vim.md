---
dateCreated: 2024-11-13
dateModified: 2025-06-03
---

<a href=" https://devhints.io/vim">cheatsheet vim</a>

# Vim 核心使用
- move
`h (left)       j (down)       k (up)       l (right)`
- operator
`a(append) A(append at the end of the line)`
`i(insert) I (insert at the beginning of the line) `
`d(delete) x(delete the cursor) c(delete and insert)`
- dd (a whole line) dw (a word) de (to the end) D (to the end)
`o(append a new line) O(append before the line)`
`r(replace a single character)`
`c(change)` 接 `e w $` 删除并开始添加，相当于 `d i`。
`R(enter replace mode until <ESC>)`
`u(undo) U(undo on a line)<ctrl>+R(undo the undo)`
`y(copy yank)`
`p(paste)`

- motion
`operator   [number]   motion`: repeat a motion a number times
`w(the next word) e(end of the word) b(the previous word)`
`0(begin) $(end) ^(第一个非空字符)`

`G` moves to the end of the file.

`<num> G` moves to that line number.

`gg` moves to the first line.

- search and substitute
`/(search after cursor) ?(search before cursor)`
After searching, `n(find next occurrence) N(opposite direction)`

`%(find matching )`

查找回车应当用 `\n`，而替换为回车应当用 `\r`（相当于 `<CR>`）。查找模式，末尾加入 `\c` 表示大小写不敏感查找，`\C` 表示大小写敏感查找。Vim 默认采用大小写敏感的查找。

在 normal 模式下按下 `*` 即可查找光标所在单词（word），要求每次出现的前后为空白字符或标点符号。例如当前为 `foo`，可以匹配 `foo bar` 中的 `foo`，但不可匹配 `foobar` 中的 `foo`。这在查找函数名、变量名时非常有用。按下 `g*` 即可查找光标所在单词的字符序列，每次出现前后字符无要求。即 foo bar 和 foobar 中的 foo 均可被匹配到。

`:set incsearch` 可以在敲键的同时搜索，按下回车把移动光标移动到匹配的词；按下 Esc 取消搜索。

`:set wrapscan` 用来设置到文件尾部后是否重新从文件头开始搜索。

语法格式：`:{作用范围}s/{查找}/{替换}/{替换标志}`

1. To substitute new for the **first** old in a line type `:s/old/new`
2. To substitute new for all 'old's on a **line** type `:s/old/new/g`
3. To substitute all occurrences in the **file** type `:%s/old/new/g

支持在 visual 模式下选取替换 `:'<,'>s/{}/{}/g`。选定行：`:2,11s/old/new/g`。另外，当前行与后续两行 `:.,+2s/old/new/g`

标志：

1. 大小敏感 `i`：`:%s/old/new/i `
2. 逐个询问 `c` : ` :%s/old/new/gc`

- command
`<ctrl>+G(show message)`
`:!` execute an external command
`<EDC> :q! <ENTER>` trash all changes. 更快捷 `ZQ`
`<ESC> :wq <ENTER>` to save the changes. 更快捷 `ZZ`
`:help <F1> <HELP>` open a help window
`:set xxx`" sets the option "xxx". 直接改变一些设定。
When typing a `:` command, press ` CTRL-D ` to see possible completions. Press `<TAB>` to use one completion.
- 读写
  1. `:w FILENAME` writes the current Vim file to disk with name FILENAME.
  2. `v  motion  :w FILENAME` saves the Visually selected lines in file FILENAME.
  3. `:r FILENAME` retrieves disk file FILENAME and puts it below the cursor position.
  4. `:r !dir` reads the output of the dir command and puts it below the cursor position.

## 跳转

`:10 :+10` 跳转到行号和相对行号。`nG` 移动光标到当前文件的第 n 行

`*` 移动光标到 [匹配] 光标当前所在单词的下一个单词。`#` 移动光标到 [匹配] 光标当前所在单词的上一个单词

`}` 移动光标到当前段落的末尾。`{` 移到光标到当前段落的开头。

`H` 移动光标到屏幕的第一行

`M` 移动光标到屏幕的中间一行

`L` 移动光标到屏幕的最后一行

`Ctrl + f` 向前滚动一页

`Ctrl + d` 向前滚动半页

`Ctrl + u` 向后滚动半页

`Ctrl + b` 向后滚动一页

Ctrl + E 和 Ctrl + Y 快捷键，它们分别以一行一行的方式上下移动屏幕，而不用移动光标。

- zz- 将当前行移动到屏幕的中 间
- （小心 zz，如果碰巧碰巧 Caps Lock 意外，你会保存并退出 vim！）
- zt - 将当前行移动到屏幕的顶部
- zb - 将当前行移动到屏幕底部

使用标记功能可以在文件中的任意位置设置标记，在普通模式下，使用 `m <标记>` 设置标记。然后通过 ` <标记>` 键跳转回该标记位置。`:delmark <标记>` 删除。

## 操作

可视模式下，`>` 用于增加缩进，`<` 减少缩进，`~` 用于转换大小写

<a href=" https://www.zhihu.com/people/vicyuan/posts">vim</a>

<a href="https://www.zhihu.com/column/c_1733808828066566145">practical vim</a>

## 文件

Vim 附带了文件浏览器（netrw），可以通过 `:Explore` 打开。在文件浏览器中，你可以直接点击文件名或使用 `j` 和 `k` 键进行上下导航，然后按 `Enter` 键打开文件并跳转到指定行。

搜索

https://www.quanxiaoha.com/vim-command/vim-search.html

在普通模式下，使用以下命令可以复制和粘贴文本：

- `"ayy`：复制当前行到寄存器 `a`
- `"ap`：粘贴寄存器 `a` 的内容

## 多窗口分屏

Vim 允许在一个窗口中分割显示多个文件或多个部分，使用以下命令：

- `:vsp`：垂直分屏
- `:sp`：水平分屏
- `Ctrl + w + 箭头键`：切换分屏焦点

## 联动操作

很多命令都可以和 Vim 教程网总结的 [vim光标移动命令](https://link.zhihu.com/?target=https%3A//vimjc.com/vim-cursor.html) 连动

基本命令模式为：`<start_position><command><end_position>`

例如，Vim 命令 `0y$` 拆开分别表示：`0` 移动光标到当前行首；`y` 复制；`$` 当前行尾。所以，命令 `0y$` 意味着复制光标当前所在行所有内容

例如，Vim 命令 `ye`，表示从当前位置拷贝到当前所在单词的最后一个字符。

## 基本配置，一些通过 Lazyvim 预设

<a href=" https://www.quanxiaoha.com/vim-command/vim-save-exit.html">vim</a>

## 按键映射

打开 PowerToys 的设置，在键盘管理页面将 Caps Lock 键映射为 Esc 键。

# Neovim 配置

<a href="https://soda.dnggentle.art/%E4%B8%BA%E9%9B%84%E5%BF%83%E5%8B%83%E5%8B%83%E7%9A%84%E5%BC%80%E5%8F%91%E8%80%85%E8%80%8C%E6%89%93%E9%80%A0%E7%9A%84lazyvim%E6%95%99%E7%A8%8B/0-%E5%89%8D%E8%A8%80/">lazyvim 中文翻译，使用教程</a>

## Lazyvim 安装

<a href=" https://lazyvim-ambitious-devs.phillips.codes/">关于 lazyvim 配置的教程</a>

需要：

- <a href="https://blog.csdn.net/m0_60670525/article/details/136329707">nvim 需要从网站下载最新版本</a>才能使用，apt-get 并非最新。
- nerd font 字体。字体需要在 windows 下安装，然后更改外观设置。

由于初始配置需要拉取 lazy. vim，可以修改 git 网址通过 ssh 拉取。注意：<a href=" https://lazy.folke.io/">lazy. vim</a>是原始的管理器，而 lazyvim 是一个基于此的已经配置好的环境。

### Lazyvim 基本按键

`<>` 代表需要用到 `<Ctrl><Alt><leader><Shift>` 键

- default `<leader>` is `<space>`
- default `<localleader>` is `\`

直接通过 `<Ctrl>` 控制窗口。

|             |                        |       |
| ----------- | ---------------------- | ----- |
| `<C-h>`     | Go to Left Window      | **n** |
| `<C-j>`     | Go to Lower Window     | **n** |
| `<C-k>`     | Go to Upper Window     | **n** |
| `<C-l>`     | Go to Right Window     | **n** |
| `<C-Up>`    | Increase Window Height | **n** |
| `<C-Down>`  | Decrease Window Height | **n** |
| `<C-Left>`  | Decrease Window Width  | **n** |
| `<C-Right>` | Increase Window Width  | **n** |

控制当前光标所在的行整体移动。

|         |           |                     |
| ------- | --------- | ------------------- |
| `<A-j>` | Move Down | **n**, **i**, **v** |
| `<A-k>` | Move Up   | **n**, **i**, **v** |

直接添加注释，或把当前行转换为注释。

|       |                   |       |
| ----- | ----------------- | ----- |
| `gco` | Add Comment Below | **n** |
| `gcO` | Add Comment Above | **n** |
| `gcc` | toggle comment    | n     |

控制窗口。

|                |                    |       |
| -------------- | ------------------ | ----- |
| `   <leader>w` | Windows            | **n** |
| `<leader>-`    | Split Window Below | **n** |
| `<leader>\|`   | Split Window Right | **n** |
| `<leader>wd`   | Delete Window      | **n** |
| `<leader>wm`   | Toggle Maximize    | **n** |

## Neo-tree

显示文件树。

|             |                             |       |
| ----------- | --------------------------- | ----- |
| `<leader>e` | Explorer NeoTree (Root Dir) | **n** |
| `<leader>E` | Explorer NeoTree (cwd)      | **n** |

## Bufferline

显示开启的多个窗口

| `<S-h>` | Prev Buffer | **n** |
| ------- | ----------- | ----- |
| `<S-l>` | Next Buffer | **n** |

| Key          | Description                 | Mode  |
| ------------ | --------------------------- | ----- |
| `<leader>bl` | Delete Buffers to the Left  | **n** |
| `<leader>bp` | Toggle Pin                  | **n** |
| `<leader>bP` | Delete Non-Pinned Buffers   | **n** |
| `<leader>br` | Delete Buffers to the Right | **n** |
| `[b`         | Prev Buffer                 | **n** |
| `[B`         | Move buffer prev            | **n** |
| `]b`         | Next Buffer                 | **n** |
| `]B`         | Move buffer next            | **n** |

## Hop

方便快速跳转。在 `options.lua` 设置快捷键。

> [!我的配置]
> `<leader>h` 开启跳转，
> `w e l ` 选择跳转的方式：单词、单词结尾、行

## Conform. Nvim

|              |                       |              |
| ------------ | --------------------- | ------------ |
| `<leader>cF` | Format Injected Langs | **n**, **v** |

可以查看有哪些支持的格式化工具，通过参数添加。

对于 C 和 C++，常用的格式化工具有 `clang-format`、`astyle`、`clang-tidy` 等。你需要确保至少安装了其中一个工具。例如，安装 `clang-format`：

### Verilog 格式化

<a href=" https://vlieo.com/post/verible-verilog-format-use-guide/">提供的指令</a>

## Mason

`:Mason` 打开管理器，主要用于安装管理文本格式化工具，如 verible, gofmt。

## Treesitter

设置语法高亮。

`:TSInstall {language}`

`:TSInstallInfo`: List information about currently installed parsers

## Ale

异步语法检查。

调用 verilator 检查 verilog。pyright 安装在 python 虚拟环境下，启动 nvim 时需要激活虚拟环境。

`:ALEInfo` 显示 ALE 的配置和状态信息

## Mini. Align

定制化的对齐功能。<a href=" https://github.com/echasnovski/mini.align/blob/main/doc/mini-align.txt">详细功能</a>

## Vim/ Neovim

https://www.bilibili.com/video/BV1y4411C7pE/?spm_id_from=333.999.0.0&vd_source=bc07d988d4ccb4ab77470cec6bb87b69

https://think.leftshadow.com/docs/vim/%E9%80%9F%E6%9F%A5/

<a href="https://www.cnblogs.com/youngxhui/p/17730419.html">从头开始配置</a>

管理插件

nvim-treesitter-context 把函数名/switch 行等 pin 住, 滚动代码时知道光标在哪个代码块里

symbols-outline 括号【包括小括号、中括号、大括号等】展示不同的颜色

### Noice. Nvim

显示信息

### Flash 快速向下搜索的

nvim-ts-rainbow2

lsp_signature 显示函数的参数的名称和种类

### Todo-comments

<a href="**[todo-comments.nvim](https://github.com/folke/todo-comments.nvim)**">高亮显示、搜索 TODO 注释</a>

PERF/ HACK/ TODO/ NOTE/ FIX/ WARNING

### Vim-wakatime **款工作统计软件**

noice 将提示、消息、命令都显示为独立的命令框

### Nvim-ufo 折叠相关的插件

nvim 中的 copilot 插件
