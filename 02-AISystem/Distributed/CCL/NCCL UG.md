# NCCL UG

https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html

---

### **核心功能**

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
3. **多GPU管理**
    
    - 支持单线程管理多个GPU。
    - 可创建多个通信器（communicators）并行运行。
    - 支持CUDA流（CUDA Stream）和CUDA Graphs集成。
4. **容错与错误处理**
    
    - 异步错误检测（如`ncclCommGetAsyncError`）。
    - 通信器销毁和异常终止（`ncclCommAbort`）。

---

### **关键API**

1. **通信器管理**
    
    - `ncclGetUniqueId`：生成唯一通信器ID。
    - `ncclCommInitRank`：初始化通信器（指定设备、ID和进程排名）。
    - `ncclCommFinalize` / `ncclCommDestroy`：销毁通信器。
2. **集体通信函数**
    
    - `ncclAllReduce` / `ncclBroadcast` / `ncclReduce` / `ncclAllGather` / `ncclReduceScatter`。
3. **组操作（Group Calls）**
    
    - `ncclGroupStart` / `ncclGroupEnd`：组合多个操作为原子操作。
4. **内存管理**
    
    - `ncclMemAlloc` / `ncclMemFree`：分配/释放内存（支持NVLink、IB等优化）。

---

### **环境变量**

1. **网络配置**
    
    - `NCCL_SOCKET_IFNAME`：指定网络接口（如`eth0`）。
    - `NCCL_IB_HCA`：InfiniBand HCA设备选择（如`mlx5_0`）。
    - `NCCL_IB_TIMEOUT` / `NCCL_IB_RETRY_CNT`：InfiniBand超时与重试策略。
2. **性能优化**
    
    - `NCCL_ALGO` / `NCCL_PROTO`：指定通信算法（环形/树形）和协议（LL/LL128）。
    - `NCCL_NET_GDR_LEVEL`：控制GPU Direct RDMA级别。
3. **调试与日志**
    
    - `NCCL_DEBUG=INFO`：启用详细日志输出。
    - `NCCL_DEBUG_FILE`：指定日志文件路径。
4. **其他配置**
    
    - `NCCL_IGNORE_CPU_AFFINITY`：忽略CPU亲和性设置。
    - `NCCL_P2P_DISABLE`：禁用P2P通信（用于调试）。

---

### **与MPI集成**

- **多设备支持**：在MPI程序中结合NCCL实现多GPU通信。
- **混合模式**：NCCL处理设备间通信，MPI处理跨节点通信（通过CUDA-aware MPI）。
- **示例**：使用`ncclCommInitRank`在每个进程中初始化NCCL通信器。

---

### **常见问题与调试**

1. **GPU Direct问题**
    
    - 检查驱动版本、PCIe拓扑（`NCCL_TOPO_DUMP_FILE`）。
    - 禁用P2P（`NCCL_P2P_DISABLE=1`）排查问题。
2. **网络问题**
    
    - InfiniBand配置（`NCCL_IB_HCA`、`NCCL_IB_SL`）。
    - RoCE/以太网适配（`NCCL_SOCKET_FAMILY`）。
3. **性能瓶颈**
    
    - 使用`NCCL_DEBUG=INFO`分析通信路径。
    - 调整`NCCL_NET_GDR_READ`优化内存传输。
4. **容器环境**
    
    - Docker需启用`--gpus`和共享IPC（`--ipc=host`）。

---

### **版本迁移**

- **从NCCL 1到2的差异**：
    - 通信器初始化方式变化（如`ncclCommInitRank`）。
    - 集合操作参数顺序调整（如`AllGather`）。
    - 新增非阻塞组操作（2.2+）。

---

### **文档资源**

- **官方链接**：NCCL User Guide
- **版本**：2.23.4（当前总结基于此版本）。

---

此总结覆盖了NCCL的核心功能、API、配置选项及常见调试方法，适用于分布式训练、多GPU通信场景的开发与优化。