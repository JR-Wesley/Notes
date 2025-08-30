# 🧱 使用 `uv` 创建和管理多个虚拟环境

## 1. 创建虚拟环境（默认在当前目录）

```
uv venv .venv
```

这会在当前目录创建一个名为 `.venv` 的虚拟环境。

> 💡 可以指定路径：
> ```
> uv venv ~/envs/myproject_venv
> ```

---

## 2. 激活虚拟环境

```
source .venv/bin/activate  # Linux/macOS
```

---

## 3. 在环境中安装包

```
uv pip install requests numpy
```

> ⚠️ 注意：即使激活了环境，也建议使用 `uv pip` 而不是 `pip`，以确保使用 `uv` 的高速解析。

---

## 4. 创建多个独立环境（推荐做法）

为不同项目创建不同的环境：

```
# 项目 A
cd project-a
uv venv .venv
source .venv/bin/activate
uv pip install flask requests

# 项目 B
cd project-b
uv venv .venv
source .venv/bin/activate
uv pip install django pandas
```

---

## 6. 列出所有环境？（⚠️ `uv` 本身不管理“全局环境列表”）

`uv` 不像 `conda` 那样有 `conda env list` 的功能。它**不维护全局环境注册表**。

但你可以：

### 手动管理环境目录，例如

```
~/envs/
├── project-a-venv
├── project-b-venv
└── data-analysis-venv
```

然后创建时指定路径：

```
uv venv ~/envs/project-a-venv
```

---

## 7. 删除环境

直接删除目录即可（`uv` 不提供删除命令）：

```
rm -rf .venv
# 或
rm -rf ~/envs/project-a-venv
```

---

## 8. 使用 `requirements.txt` 管理依赖

```
# 生成依赖文件
uv pip freeze > requirements.txt

# 安装依赖
uv pip install -r requirements.txt
```

---

## 9. 创建并安装依赖（一步完成）

```
uv venv .venv && source .venv/bin/activate && uv pip install -r requirements.txt
```

---

## 10. 查看环境中的包

```
uv pip list
uv pip show requests
```

# 🔄 与 `pip` + `venv` 对比

|功能|`pip` + `venv`|`uv`|
|---|---|---|
|创建虚拟环境|`python -m venv .venv`|`uv venv .venv` ✅ 更快|
|安装包|`pip install pkg`|`uv pip install pkg` ✅ 极快|
|安装依赖|`pip install -r req.txt`|`uv pip install -r req.txt` ✅ 更快|
|依赖解析|慢|⚡ 超快（Rust 实现）|
|全局环境管理|❌ 无|❌ 无（需手动）|

# ✅ 总结

| 目标      | 命令                                                 |
| ------- | -------------------------------------------------- |
| 安装 `uv` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| 创建环境    | `uv venv .venv`                                    |
| 激活环境    | `source .venv/bin/activate`                        |
| 安装包     | `uv pip install 包名`                                |
| 安装依赖    | `uv pip install -r requirements.txt`               |
| 删除环境    | `rm -rf .venv`                                     |
| 列出包     | `uv pip list`                                      |

> 💡 提示：`uv` 生成的环境是标准的 `venv` 环境，你可以用 `python -m venv` 激活，也可以用 `uv` 安装包，完全兼容。

---

`uv` 是目前**最快、最现代**的 Python 环境和包管理工具之一，特别适合需要频繁创建环境和安装依赖的开发场景。推荐替代 `pip` 和 `virtualenv`！
