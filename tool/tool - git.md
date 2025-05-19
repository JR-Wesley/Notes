---
dateCreated: 2025-04-02
dateModified: 2025-04-02
---
# Git

<a href=" https://www.runoob.com/git/git-tutorial.html">runoob git 教程</a> <a href=" https://blog.csdn.net/m0_63230155/article/details/134607239">常用命令</a>

[Visualizing Git Concepts with D3](http://onlywei.github.io/explain-git-with-d3)

<a href="https://git-scm.com/docs">git 官方文档</a>

<a href=" https://www.runoob.com/manual/git-guide/">简明指南</a>

特点

- Git 是一个开源的分布式版本控制系统，用于敏捷高效地处理任何或小或大的项目。
- Git 是 Linus Torvalds 为了帮助管理 Linux 内核开发而开发的一个开放源码的版本控制软件。
- Git 与常用的版本控制工具 CVS, Subversion 等不同，它采用了分布式版本库的方式，不必服务器端软件支持。

```shell
# 设置
git config --global user.name "Your Name" $ 
git config --global user.email "email@example.com"

### …or create a new repository on the command line
echo "# Tools" >> README.md
# 创建新仓库
git init
# 你可以提出更改（把它们添加到暂存区）
git add README.md
# 实际提交改动
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/JR-Wesley/Tools.git
# 将这些改动提交到远端仓库
git push -u origin main

### …or push an existing repository from the command line
# 将你的仓库连接到某个远程服务器
git remote add origin https://github.com/JR-Wesley/Tools.git
# 修改 url
git remote set-url origin git@github.com:someaccount/someproject.git

git branch -M main # 这个 main 对应分支的名字
git push -u origin main
```

`git config --global url.ssh://git@github.com/.insteadOf https://github.com/` 把默认 htpps 改成 ssh。<a href="https://stackoverflow.com/questions/11200237/how-do-i-get-git-to-default-to-ssh-and-not-https-for-new-repositories">参考</a>

## 检出仓库

执行如下命令以创建一个本地仓库的克隆版本：

`git clone /path/to/repository`

如果是远端服务器上的仓库，你的命令会是这个样子：

`git clone username@host:/path/to/repository`

## 分支

```shell
# 查看当前分支
git branch
# 创建分支 $ 
git branch dev
# 切换分支
git checkout dev
# 创建一个叫做 dev 的分支，并切换过去
git checkout -b dev

# 合并分支
git merge dev
# 删除分支 $ 
git branch -d dev
# 重命名当前分支为 main
git branch -M main
```

除非你将分支推送到远端仓库，不然该分支就是 _不为他人所见的_：

## 更新与合并

要更新你的本地仓库至最新改动，执行：

`git pull`

以在你的工作目录中 _获取（fetch）_ 并 _合并（merge）_ 远端的改动。

要合并其他分支到你的当前分支（例如 master），执行：

`git merge <branch>`

在这两种情况下，git 都会尝试去自动合并改动。遗憾的是，这可能并非每次都成功，并可能出现 _ 冲突（conflicts）_。这时候就需要你修改这些文件来手动合并这些 _ 冲突（conflicts）_。改完之后，你需要执行如下命令以将它们标记为合并成功：

`git add <filename>`

在合并改动之前，你可以使用如下命令预览差异：

`git diff <source_branch> <target_branch>`

## 替换本地改动

假如你操作失误（当然，这最好永远不要发生），你可以使用如下命令替换掉本地改动：

`git checkout -- <filename>`

此命令会使用 HEAD 中的最新内容替换掉你的工作目录中的文件。已添加到暂存区的改动以及新文件都不会受到影响。

假如你想丢弃你在本地的所有改动与提交，可以到服务器上获取最新的版本历史，并将你本地主分支指向它：

`git fetch origin`

`git reset --hard origin/master`

## 忽视

<a href=" https://blog.csdn.net/m0_63230155/article/details/134471033">通过 `. gitignore` 忽视指定文件</a>

## Lazygit

<a href=" https://github.com/jesseduffield/lazygit/blob/master/docs/Config.md">lazygit 默认配置</a>

Default path for the global config file:

- Linux: `~/.config/lazygit/config.yml
关闭 `autoFetch`
