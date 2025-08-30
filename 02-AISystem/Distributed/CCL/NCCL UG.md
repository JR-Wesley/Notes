NCCL 基础解读：https://aijishu.com/a/1060000000483892

# NCCL UG

https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html

---

## **核心功能**

1. **集合通信操作**
    
    - **AllReduce**：跨多个设备/节点聚合数据（如梯度同步）。
    - **Broadcast**：从一个设备向所有设备广播数据。
    - **Reduce**：汇总多个设备的数据到目标设备。
    - **AllGather**：收集所有设备的数据到每个设备。
    - **ReduceScatter**：分片汇总数据后分发到各设备。
2. **点对点通信**
    
    - **Send/Recv**：直接在设备间传输数据。
    - **Scatter/Gather**：分发/收集数据到多个设备。
    - **All-to-all**：全互连通信模式。
3. **多 GPU 管理**
    
    - 支持单线程管理多个 GPU。
    - 可创建多个通信器（communicators）并行运行。
    - 支持 CUDA 流（CUDA Stream）和 CUDA Graphs 集成。
4. **容错与错误处理**
    
    - 异步错误检测（如 `ncclCommGetAsyncError`）。
    - 通信器销毁和异常终止（`ncclCommAbort`）。

---

## **关键 API**

1. **通信器管理**
    
    - `ncclGetUniqueId`：生成唯一通信器 ID。
    - `ncclCommInitRank`：初始化通信器（指定设备、ID 和进程排名）。
    - `ncclCommFinalize` / `ncclCommDestroy`：销毁通信器。
2. **集体通信函数**
    
    - `ncclAllReduce` / `ncclBroadcast` / `ncclReduce` / `ncclAllGather` / `ncclReduceScatter`。
3. **组操作（Group Calls）**
    
    - `ncclGroupStart` / `ncclGroupEnd`：组合多个操作为原子操作。
4. **内存管理**
    
    - `ncclMemAlloc` / `ncclMemFree`：分配/释放内存（支持 NVLink、IB 等优化）。

---

## **环境变量**

1. **网络配置**
    
    - `NCCL_SOCKET_IFNAME`：指定网络接口（如 `eth0`）。
    - `NCCL_IB_HCA`：InfiniBand HCA 设备选择（如 `mlx5_0`）。
    - `NCCL_IB_TIMEOUT` / `NCCL_IB_RETRY_CNT`：InfiniBand 超时与重试策略。
2. **性能优化**
    
    - `NCCL_ALGO` / `NCCL_PROTO`：指定通信算法（环形/树形）和协议（LL/LL128）。
    - `NCCL_NET_GDR_LEVEL`：控制 GPU Direct RDMA 级别。
3. **调试与日志**
    
    - `NCCL_DEBUG=INFO`：启用详细日志输出。
    - `NCCL_DEBUG_FILE`：指定日志文件路径。
4. **其他配置**
    
    - `NCCL_IGNORE_CPU_AFFINITY`：忽略 CPU 亲和性设置。
    - `NCCL_P2P_DISABLE`：禁用 P2P 通信（用于调试）。

---

## **与 MPI 集成**

- **多设备支持**：在 MPI 程序中结合 NCCL 实现多 GPU 通信。
- **混合模式**：NCCL 处理设备间通信，MPI 处理跨节点通信（通过 CUDA-aware MPI）。
- **示例**：使用 `ncclCommInitRank` 在每个进程中初始化 NCCL 通信器。

---

## **常见问题与调试**

1. **GPU Direct 问题**
    
    - 检查驱动版本、PCIe 拓扑（`NCCL_TOPO_DUMP_FILE`）。
    - 禁用 P2P（`NCCL_P2P_DISABLE=1`）排查问题。
2. **网络问题**
    
    - InfiniBand 配置（`NCCL_IB_HCA`、`NCCL_IB_SL`）。
    - RoCE/以太网适配（`NCCL_SOCKET_FAMILY`）。
3. **性能瓶颈**
    
    - 使用 `NCCL_DEBUG=INFO` 分析通信路径。
    - 调整 `NCCL_NET_GDR_READ` 优化内存传输。
4. **容器环境**
    
    - Docker 需启用 `--gpus` 和共享 IPC（`--ipc=host`）。

---

## **版本迁移**

- **从 NCCL 1 到 2 的差异**：
    - 通信器初始化方式变化（如 `ncclCommInitRank`）。
    - 集合操作参数顺序调整（如 `AllGather`）。
    - 新增非阻塞组操作（2.2+）。

---

## **文档资源**

- **官方链接**：NCCL User Guide
- **版本**：2.23.4（当前总结基于此版本）。

---

此总结覆盖了 NCCL 的核心功能、API、配置选项及常见调试方法，适用于分布式训练、多 GPU 通信场景的开发与优化。
