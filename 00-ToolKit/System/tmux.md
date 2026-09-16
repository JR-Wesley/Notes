# tmux：持久化终端工作区

tmux 是 terminal multiplexer（终端复用器）：它在一个 tmux server 中管理多个 session，每个 session 包含多个 window，每个 window 又可拆成多个 pane。它尤其适合通过 SSH 进行服务器开发，因为终端程序连接到 tmux 提供的伪终端，而不是直接依赖当前 SSH 客户端；客户端 detach 或网络断开后，只要远端主机、tmux server、对应 session 和其中的进程仍在运行，就可以再次 attach 并恢复现场。tmux 不是任务持久化或容灾系统：重启主机、终止 tmux server/session，或程序自身退出，都会使相应工作消失。

## 核心模型与按键规则

tmux 的层级可概括为 `server → session → window → pane`。session 是一个可分离、可重连的工作空间；window 类似终端标签页；pane 是 window 内的分屏。客户端 attach 到 session 后，普通按键会发给当前 pane 中的程序；tmux 自己的默认命令以 Prefix `C-b`（`Ctrl-b`）开头。写作 `C-b c` 时，应先按 `Ctrl-b`，松开后再按 `c`，而不是同时按三个键。

```text
tmux server
├── session: dev
│   ├── window 0: code
│   │   ├── pane 0
│   │   └── pane 1
│   └── window 1: logs
└── session: test
    └── window 0
```

## 建立与恢复工作区

为 session 命名比依赖数字编号更容易辨认。`new-session` 的 `-A` 选项可在同名 session 已存在时直接 attach，不存在时创建，因此适合作为日常入口。

```bash
# 创建名为 dev 的 session
tmux new-session -s dev

# 已存在则 attach，不存在则创建
tmux new-session -A -s dev

# 查看 session
tmux list-sessions

# 连接已有 session
tmux attach-session -t dev

# 终止 session；其中的 window、pane 及其程序也会被终止
tmux kill-session -t dev
```

常见简写分别是 `tmux new -s dev`、`tmux ls` 和 `tmux a -t dev`。在 tmux 内按 `C-b d` 是 detach：当前客户端返回外层 shell，session 和其中仍在运行的程序继续存在。它不同于在 pane 中执行 `exit`；后者会结束该 pane 的 shell，最后一个 pane 退出后，相应 window 会消失，最后一个 session 结束后 tmux server 通常也会退出。

服务器开发可以固定采用如下流程：

```bash
ssh server
tmux new-session -A -s dev
```

结束连接前按 `C-b d`，下次登录后重复同一条 `tmux new-session -A -s dev` 即可。即使 SSH 意外断线，通常也能重新 attach；但关键服务仍应由 systemd、容器编排器或作业调度系统管理，而不应只依赖 tmux。

## Window 与 pane

window 适合分隔 code、build、test、monitor、logs 等长期上下文，pane 适合同时观察相关命令。以下均为默认绑定；配置可能覆盖它们，可用 `C-b ?` 或 `tmux list-keys -T prefix -N` 查看当前有效绑定。

| 操作 | 默认按键 |
| --- | --- |
| detach 当前客户端 | `C-b d` |
| 新建 window | `C-b c` |
| 上一个／下一个 window | `C-b p`／`C-b n` |
| 选择编号为 0–9 的 window | `C-b 0` … `C-b 9` |
| 显示 session/window/pane 树 | `C-b w` |
| 重命名当前 window | `C-b ,` |
| 左右分屏 | `C-b %` |
| 上下分屏 | `C-b "` |
| 按方向选择 pane | `C-b ←/→/↑/↓` |
| 最大化／恢复当前 pane | `C-b z` |
| 关闭当前 pane（需确认） | `C-b x` |
| 显示按键帮助 | `C-b ?` |

`C-b x` 会终止 pane，不等同于 detach。对于仍在运行的构建、测试或日志任务，应先确认当前 pane 内容再操作。一个实用布局是将编译放在主 pane，将 `watch -n 1 nvidia-smi`、`htop` 或日志跟踪放在旁边；需要集中查看某个 pane 时按 `C-b z`，再次按下即可恢复布局。

## 历史输出、复制与系统剪贴板

按 `C-b [` 进入 copy mode 后，pane 的画面被冻结，可以使用方向键和 `PageUp`、`PageDown` 浏览历史，按 `q` 退出。tmux 维护自己的 paste buffer；选中文本后的具体复制键取决于 copy mode 使用 Emacs 还是 vi 绑定，最近一次复制的内容可用 `C-b ]` 粘贴到当前 pane。

`setw -g mode-keys vi` 会启用 vi 风格 copy mode，但 tmux buffer 与 macOS/Linux 的系统剪贴板不是天然等价。能否同步到系统剪贴板取决于 tmux 的 `set-clipboard`、外层终端对相应转义序列的支持，以及 SSH/本机工具环境；不要仅凭启用 vi mode 或鼠标就假定跨机器剪贴板已经可用。

## 一份克制的配置

tmux server 启动时会读取 `~/.tmux.conf`；之后创建新 session 不会自动重读它。下面的配置开启鼠标、扩大滚动历史、从 1 编号，并让新 window/pane 继承当前 pane 的工作目录。`bind` 默认写入 prefix key table，因此这些按键仍需先按 `C-b`。

```tmux
# ~/.tmux.conf
set -g mouse on
set -g history-limit 100000

set -g base-index 1
setw -g pane-base-index 1
set -g renumber-windows on

setw -g mode-keys vi

bind c new-window -c "#{pane_current_path}"
bind | split-window -h -c "#{pane_current_path}"
bind - split-window -v -c "#{pane_current_path}"

bind r source-file ~/.tmux.conf \; display-message "tmux.conf reloaded"
```

保存后，可在现有 tmux 客户端中按 `C-b r` 重载。只有这段配置已经生效时，`C-b |` 和 `C-b -` 才分别表示左右和上下分屏；默认分屏键仍是 `%` 与 `"`。

不要在未检查 terminfo 的情况下机械设置 `default-terminal "tmux-256color"`。可先执行 `infocmp tmux-256color`；若本机缺少该条目，程序可能出现颜色或按键能力判断错误。终端能力与 SSH 跨主机传播还涉及外层 `$TERM`、远端 terminfo 数据库和 tmux 版本，应按实际环境验证。

## 安装与适用边界

tmux 在 macOS 与主流 Linux 上的核心命令和默认按键基本一致，差异主要来自包管理器、tmux 版本、终端模拟器、terminfo 与剪贴板环境。

```bash
# Debian / Ubuntu
sudo apt install tmux

# Fedora / RHEL 系发行版
sudo dnf install tmux

# macOS（Homebrew）
brew install tmux
```

日常使用先掌握 `tmux new-session -A -s dev`、`C-b d`、window 切换、pane 分屏、`C-b z` 和 `C-b [` 即可。插件、会话布局恢复和系统剪贴板集成应在理解默认行为后按需增加；其中 tmux-resurrect 一类插件可以保存部分会话信息，但仍不能把任意进程的内存状态变成可恢复快照。

## 来源

- [tmux 官方 Getting Started](https://github.com/tmux/tmux/wiki/Getting-Started)：层级模型、Prefix、attach/detach、默认绑定、copy mode 与配置加载行为。
- [tmux 官方 Installing](https://github.com/tmux/tmux/wiki/Installing)：支持平台、包管理器与源码安装说明。
- [tmux manual page（OpenBSD）](https://man.openbsd.org/tmux)：命令、选项、目标语法和默认按键的完整参考。

