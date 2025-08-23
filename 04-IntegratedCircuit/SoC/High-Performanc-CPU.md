---
dateCreated: 2025-02-27
dateModified: 2025-04-11
---
# SOC

> [DarkRISC-V开源代码](https://github.com/darklife/darkriscv)
> [tinyriscv](https://gitee.com/liangkangnan/tinyriscv)

此 repo 实现了一个简单的 MIPS 五级流水 CPU：https://crpboy.github.io/p/nscscc-2024-report/

实现简单 MIPS 五级流水 CPU 对应视频：[教你写一个简单的CPU](https://www.bilibili.com/video/BV1pK4y1C7es)

[计算机组成原理实验与参考实现](https://github.com/lvyufeng/step_into_mips)

- 多发射
- 乱序
- cache 大小
- tlb
- miss 处理
- cachemiss wave front 调度
[[SOC]]
## CPU

高性能 CPU 设计涉及多个关键部件和技术，旨在提升指令执行效率、并行性和能效。以下是主要部件及其功能的详细介绍：

---

### **1. 分支预测（Branch Prediction）**
- **作用**：预测条件分支（如 `if-else`）的执行路径，减少流水线停顿。
- **实现**：
  - **静态预测**：基于指令类型简单预测（如总预测分支不跳转）。
  - **动态预测**：基于历史行为调整预测（如两位饱和计数器）。
  - **高级算法**：TAGE（Tagged Geometric History Length）、神经分支预测器。
- **重要性**：错误预测会导致流水线清空（惩罚周期），准确预测可提升 10-30% 性能。

---

### **2. 指令流水线（Pipeline）**
- **阶段划分**：取指（Fetch）、译码（Decode）、执行（Execute）、访存（Memory）、写回（Writeback）。
- **优化技术**：
  - **深度流水线**：增加阶段数以提升频率（如 Intel Pentium 4 的 20 级流水线），但分支惩罚增大。
  - **流水线冒险处理**：通过旁路（Bypassing）和流水线停顿（Stalling）解决数据/控制依赖。

---

### **3. 超标量架构（Superscalar）**
- **核心思想**：每个时钟周期发射多条指令到不同执行单元。
- **关键部件**：
  - **发射队列（Issue Queue）**：动态调度指令，识别可并行执行的指令。
  - **保留站（Reservation Stations）**：管理指令依赖，分配执行资源。
- **示例**：Intel 的 Skylake 架构每个周期可发射 6 条指令。

---

### **4. 乱序执行（Out-of-Order Execution, OoOE）**
- **机制**：动态重排指令顺序，绕过数据依赖和资源冲突。
- **依赖技术**：
  - **寄存器重命名**：消除假依赖（如 WAR/WAW 冲突），使用物理寄存器堆动态映射。
  - **重排序缓冲区（ROB）**：跟踪指令状态，确保最终结果按程序顺序提交。
- **优势**：提高执行单元利用率，减少空闲周期。

---

### **5. 执行单元（Execution Units）**
- **类型**：
  - **整数单元（ALU）**：处理算术逻辑运算。
  - **浮点单元（FPU）**：加速浮点计算。
  - **向量单元（SIMD）**：如 SSE、AVX，单指令处理多数据。
  - **内存单元**：处理加载（Load）和存储（Store）操作。
- **多端口设计**：支持同时执行多个同类操作（如 2 个整数单元 +1 个浮点单元）。

---

### **6. 高速缓存（Cache）**
- **层级结构**：
  - **L 1 Cache**：分指令/数据缓存（如 32-64 KB），1-3 周期延迟。
  - **L 2 Cache**：统一缓存（256 KB-1 MB），10-20 周期延迟。
  - **L 3 Cache**：共享缓存（数 MB 到数十 MB），20-50 周期延迟。
- **优化策略**：
  - **预取（Prefetching）**：预测未来数据并提前加载。
  - **替换算法**：LRU（最近最少使用）、随机替换。

---

### **7. 内存子系统**
- **内存控制器（IMC）**：管理 DRAM 访问，支持多通道（如 DDR 4/DDR 5）。
- **虚拟内存支持**：TLB（Translation Lookaside Buffer）加速虚拟地址到物理地址转换。

---

### **8. 多核与多线程**
- **多核（Multi-Core）**：集成多个独立处理核心，共享 L 3 缓存。
- **同步技术**：MESI 协议维护缓存一致性。
- **超线程（Hyper-Threading）**：单个物理核心模拟多个逻辑核心，共享执行资源。

---

### **9. 功耗与散热管理**
- **动态电压频率调整（DVFS）**：根据负载调整电压和频率。
- **时钟门控（Clock Gating）**：关闭空闲模块的时钟信号以省电。
- **高级制程技术**：FinFET、GAA 晶体管降低漏电流。

---

### **10. 高级优化技术**
- **推测执行（Speculative Execution）**：提前执行可能需要的指令，若预测错误则回滚。
- **值预测（Value Prediction）**：预测指令结果以减少依赖等待。
- **近内存计算（Near-Memory Computing）**：在内存附近集成计算单元，减少数据搬运开销。

---

### **示例架构：Apple M 1 Ultra**
- **核心设计**：16 个高性能 Firestorm 核心 + 4 个能效 Icestorm 核心。
- **关键技术**：
  - 8 宽解码/发射，乱序执行窗口超过 600 条指令。
  - 统一内存架构（UMA），192 GB/s 带宽。
  - 定制 AMX 矩阵加速单元，专攻 AI 计算。

---

高性能 CPU 通过分支预测降低控制冒险、超标量乱序执行提升并行性、多级缓存减少内存延迟、多核/多线程扩展吞吐量，结合先进制程和功耗管理，实现每秒万亿次计算的效率。未来趋势包括异构计算（CPU+GPU/FPGA）、Chiplet 集成和量子计算启发的新架构。
