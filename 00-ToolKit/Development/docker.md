- 安装： docker.cnb.cool/fudaneda/docker/chipyard
- 需要添加 sudo 以使用 vsocd 插件：[(71 封私信 / 86 条消息) 用vscode 安装docker显示不出有容器和镜像 ？ - 知乎](https://zhuanlan.zhihu.com/p/1938195768633718492)

# Docker

## 框架

Docker 是一款开源的容器化平台，通过虚拟化技术将应用及其依赖打包成标准化容器，实现“一次构建，到处运行”。以下是 Docker 的核心概念及常用操作详解：

### **一、核心概念**

1. **镜像（Image）**
   - 容器的“模板”，包含运行应用所需的代码、依赖、配置等（只读）。
   - 可从 Docker Hub（公共仓库）拉取，或自行构建。

2. **容器（Container）**
   - 镜像的运行实例，是独立的可执行单元（可读写层）。
   - 容器间相互隔离，拥有独立的网络和文件系统。

3. **仓库（Repository）**
   - 存储和分发镜像的平台（如 Docker Hub、私有仓库）。

```bash
docker --version  # 查看版本
docker run hello-world  # 运行测试镜像
```

## 常用命令

拉取的镜像本质是一个“模板”，需先创建容器才能运行其中的应用。关键是理解镜像的用途（如是否是 Web 服务、数据库、工具等），通常镜像作者会在仓库文档中说明使用方式（如端口、环境变量、启动命令等）。

### 1. **查看镜像信息**

先确认拉取的镜像名称和标签，以及是否有文档说明：

```bash
docker images  # 列出本地镜像，找到刚拉取的镜像（如他人的 `xxx/app:latest`）
```

- 建议访问镜像的仓库页面（如 Docker Hub 的对应地址），查看作者提供的 `README`，了解启动参数、默认配置等（这是“合适使用”的关键，避免因参数缺失导致容器无法正常运行）。

### 2. **创建并启动容器**

根据镜像用途，使用 `docker run` 命令创建容器，核心参数根据需求调整：

```bash
docker run [选项] [镜像名:标签] [可选命令]
```

**常用场景示例**：
- **如果是 Web 应用（如 Nginx、Node.js 服务）**：需要映射端口（`-p`）让主机能访问：

  ```bash
  # 示例：运行一个Web镜像，将容器的80端口映射到主机的8080端口，后台运行
  docker run -d -p 8080:80 --name myapp 他人的镜像名:标签
  ```

  此时访问 `http://localhost:8080` 即可打开应用。

- **如果是数据库（如 MySQL、PostgreSQL）**：需要设置环境变量（`-e`）配置密码，挂载数据卷（`-v`）持久化数据：

  ```bash
  # 示例：运行他人的MySQL镜像，设置root密码，持久化数据
  docker run -d -p 3306:3306 \
    -e MYSQL_ROOT_PASSWORD=你的密码 \
    -v mysql-data:/var/lib/mysql \  # 数据卷持久化数据
    --name mydb \
    他人的镜像名:标签
  ```

- **如果是交互式工具（如 Python、Ubuntu 终端）**：需要 `-it` 参数进入交互模式：

  ```bash
  # 示例：进入他人的Ubuntu镜像的bash终端
  docker run -it --name myubuntu 他人的镜像名:标签 /bin/bash
  ```

### 3. **验证容器是否正常运行**

```bash
docker ps  # 查看运行中的容器，确认状态为 Up
docker logs 容器名/ID  # 查看日志，排查启动失败问题（如参数错误、端口冲突）
```

## **二、容器的日常管理**

容器启动后，需根据需求进行状态监控、配置修改、数据备份等管理操作。

### 1. **查看容器状态**

```bash
docker ps  # 运行中的容器
docker ps -a  # 所有容器（包括已停止的）
docker inspect 容器名/ID  # 查看容器详细信息（IP、配置、挂载等）
```

### 2. **进入运行中的容器（修改配置/调试）**

如果需要临时修改容器内的文件或执行命令，用 `docker exec` 进入交互终端（**推荐，不影响容器运行**）：

```bash
docker exec -it 容器名/ID /bin/bash  # 进入bash（若容器内无bash，可尝试 /bin/sh）
```

- 退出终端时，用 `exit` 命令，容器会继续后台运行（不会因退出终端而停止）。

### 3. **启动/停止/重启容器**

```bash
docker stop 容器名/ID  # 停止容器（优雅关闭）
docker start 容器名/ID  # 启动已停止的容器
docker restart 容器名/ID  # 重启容器（配置生效常用）
```

### 4. **数据管理（避免数据丢失）**

- 如果拉取的镜像未使用数据卷（`-v`），容器内的数据会随容器删除而丢失。建议：
  - 若容器已运行且有重要数据，可先通过 `docker cp` 复制到主机：

    ```bash
    docker cp 容器名/ID:容器内路径 主机路径  # 如复制日志：docker cp myapp:/app/logs ./logs
    ```

  - 后续重启容器时，通过 `-v` 挂载数据卷或主机目录，确保数据持久化。

### 5. **备份与迁移容器**

- 若需保存当前容器状态为新镜像（可迁移到其他机器）：

  ```bash
  docker commit 容器名/ID 新镜像名:标签  # 将容器状态提交为镜像
  docker save -o 备份文件名.tar 新镜像名:标签  # 导出镜像为tar包
  # 其他机器导入：docker load -i 备份文件名.tar
  ```

## **三、退出容器的正确方式**

“退出容器”分两种场景：**退出容器内的终端**（容器继续运行）和**停止容器**（容器不再运行）。

### 1. **退出容器内的交互终端（容器继续运行）**

当你通过 `docker exec -it` 或 `docker run -it` 进入容器的终端后，退出终端但保持容器运行：

- 直接输入 `exit` 命令，或按 `Ctrl+D`，终端退出，容器仍在后台运行（可通过 `docker ps` 确认）。

### 2. **停止容器（彻底退出运行）**

若需终止容器运行（如应用不再使用）：

```bash
docker stop 容器名/ID  # 优雅停止（发送 SIGTERM 信号，允许应用保存状态）
# 若容器无响应，强制停止：
docker kill 容器名/ID  # 发送 SIGKILL 信号，立即终止
```

### 3. **删除不再需要的容器**

容器停止后仍会占用磁盘空间，可删除无用容器：

```bash
docker rm 容器名/ID  # 删除已停止的容器
docker rm -f 容器名/ID  # 强制删除运行中的容器（不推荐，建议先stop）
```

## **3. 数据管理**

- **数据卷（Volume）**：Docker 管理的持久化存储，独立于容器生命周期。

  ```bash
  docker volume create myvol  # 创建卷
  docker run -d -v myvol:/app/data myapp  # 挂载卷到容器 /app/data
  docker volume ls  # 查看所有卷
  docker volume rm myvol  # 删除卷
  ```

- **绑定挂载**：直接挂载主机目录到容器（需指定绝对路径）。

  ```bash
  docker run -d -v /host/path:/container/path myapp
  ```

## **4. 网络管理**

- **创建自定义网络**（容器间通过名称通信）：

  ```bash
  docker network create mynet  # 创建桥接网络
  docker run -d --name app1 --network mynet myapp  # 加入网络
  docker run -d --name app2 --network mynet myapp  # app2 可通过 app1 访问 app1
  ```

## **Dockerfile 构建镜像**

Dockerfile 是镜像的构建脚本，示例如下（构建一个 Python 应用）：

```dockerfile
# 基础镜像
FROM python:3.9

# 工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["python", "app.py"]
```

构建并运行：

```bash
docker build -t mypythonapp:v1 .
docker run -d -p 5000:5000 mypythonapp:v1
```

## **Docker Hub**

1. 登录 Docker Hub：

   ```bash
   docker login  # 输入用户名和密码
   ```

2. 给镜像打标签（格式：用户名/仓库名: 标签）：

   ```bash
   docker tag myapp:v1 username/myapp:v1
   ```

3. 推送镜像：

   ```bash
   docker push username/myapp:v1
   ```
