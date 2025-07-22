---
dateCreated: 2025-02-11
dateModified: 2025-07-13
---

相关资源推荐：https://zhuanlan.zhihu.com/p/20076957712

AI 相关大学课程严忻恺：https://www.zhihu.com/people/yan-xin-kai-38/posts

综述：

- https://zhuanlan.zhihu.com/p/101544149
- https://blog.csdn.net/qq_42722197/article/details/119814538
- https://zhuanlan.zhihu.com/p/616138047
- <a href="https://zhuanlan.zhihu.com/p/33876622">一天搞懂</a>
- <a href=" https://zhuanlan.zhihu.com/p/20076957712?utm_psn=1872615359586111488">知乎总结</a>
- <a href=" https://chenzomi12.github.io/index.html">ZOMI AI 系统</a><img src=" https://chenzomi12.github.io/_images/03Architecture031.png" alt="系统框图">
- <a ref=" https://openmlsys.github.io/index.html">open ML sys </a>
- <a href="https://novel.ict.ac.cn/aics/">中科大智能计算系统课程和书</a>

https://eyeriss.mit.edu/tutorial-previous.html eyeriss tutorial

AL chip:

- <a href=" https://nycu-caslab.github.io/AAML2024/labs/lab_2.html">CSIC 30066 台湾课程</a>
- CSCS 10014: Computer Organization ( https://nycu-caslab.github.io/CO2024/index.html# )
- <a href=" https://hanlab.mit.edu/courses/2024-fall-65940">TinyML and Efficient Deep Learning Computing</a>by MIT hansong, 如何基于已有的硬件进行优化。
- <a href=" https://csg.csail.mit.edu/6.5930/index.html">6.5930/1 Hardware Architecture for Deep Learning</a>如何设计更好的面向深度学习的硬件
- <a href=" https://people.cs.nycu.edu.tw/~ttyeh/course/2024_Fall/IOC5009/outline.html">AAML2024 台湾课程</a>
- <a href="https://nycu-caslab.github.io/AAML2024/index.html"> AAML2024 实验网站</a>
- EE 290 笔记 https://www.zhihu.com/people/zfeng-xin-zw/posts
- ece 5545 https://www.bilibili.com/video/BV1kn4y1o7Eu/?spm_id_from=333.1387.favlist.content.click&vd_source=bc07d988d4ccb4ab77470cec6bb87b69 https://zhuanlan.zhihu.com/p/668411397

利用了谷歌的 CFU 平台<a href=" https://cfu-playground.readthedocs.io/en/latest/index.html">一个介绍的网站</a>

CS 217 MIT 课程整理：https://www.zhihu.com/people/yan-xin-kai-38/posts

# EE 290
# DNN

# Quantization

NVIDIA Ampere Architecture features the Third-generation Tensor Cores:

- Acceleration for all data types including FP 16, BF 16, TF 32, FP 64, INT 8, INT 4, and Binary.

## Floating-Point Arithmetic
## Fixed-Point Arithmetic
## Hardware Implication
## DNN Quantization

# 介绍

通俗来讲，机器学习是指从数据中学习出有用知识的技术。以学习模式分类，机器学习可以分为监督学习（Supervised Learning）、无监督学习（Unsupervised Learning）和强化学习（Reinforcement Learning）等。

- 监督学习是已知输入和输出的对应关系下的机器学习场景。比如给定输入图像和它对应的离散标签。
- 无监督学习是只有输入数据但不知道输出标签下的机器学习场景。比如给定一堆猫和狗的图像，自主学会猫和狗的分类，这种无监督分类也称为聚类（Clustering）。
- 强化学习则是给定一个学习环境和任务目标，算法自主地去不断改进自己以实现任务目标。比如 AlphaGo 围棋就是用强化学习实现的，给定的环境是围棋的规则，而目标则是胜利得分。

# 分布式
### 一、岗位核心技能要求

从岗位职责和任职要求提炼，该岗位需掌握以下核心技能：

#### 1. **基础技术栈**

- **编程语言**：熟练掌握 **C/C++**（系统级编程、内存管理、性能优化），熟悉 **Linux 编程**（用户态工具链、系统调用），深入理解 **Linux Kernel**（设备驱动、进程调度、内存管理，为驱动开发打基础）。
- **体系架构**：理解 **计算机体系结构**（CPU/GPU 协同、总线 / 缓存机制），掌握 **分布式并行计算**（如 MPI 模型）和 **异构计算框架**（CUDA 为核心，可拓展 OpenCL/HIP）。

#### 2. **多 GPU 专项技能**

- **驱动与软件设计**：设计、开发多 GPU 驱动程序（含用户态 / 内核态交互），支持 **多 GPU 互联**（南向 / 北向互联，如 PCIe Peer2Peer、GPU Direct）。
- **通信与加速库**：熟悉 **NCCL**（多 GPU 通信加速，用于分布式训练）、**DOCA**（NVIDIA 数据中心 GPU 框架）、**NVSHMEM**（GPU 间共享内存），了解 **RDMA**（远程直接内存访问，实现跨节点 GPU 通信）。
- **硬件编程**：掌握芯片互联（PCIe 架构）、网络传输编程（RDMA 协议、verbs API），具备 GPU 硬件特性适配经验（如显存分配、并行任务调度）。

### 二、系统学习路径（分阶段）

#### **阶段 1：基础夯实（2-3 个月）**

- **语言与系统**：
    - 深入学习 **C/C++**：聚焦指针、内存管理、多线程（如 POSIX Threads），推荐《Effective C++》《C++ 并发编程实战》。
    - 精通 **Linux 编程**：学习进程 / 线程管理、IO 操作、系统调用，实践《Linux 程序设计》案例；通过 **Linux Kernel 源码**（重点看 `drivers/gpu` 目录）理解驱动架构。
- **并行计算入门**：
    - 学习 **CUDA 基础**：掌握线程模型（Grid/Block/Thread）、内存管理（全局 / 共享内存），完成 NVIDIA 官方示例（如向量加法、矩阵乘法），参考《CUDA C Programming Guide》。

#### **阶段 2：多 GPU 深化（3-6 个月）**

- **多 GPU 通信与互联**：
    - **NCCL 实战**：学习集合通信（AllReduce、Broadcast）、点对点通信 API，结合 PyTorch/TensorFlow 的多 GPU 训练（如 `torch.nn.parallel.DistributedDataParallel`），复现官方示例。
    - **硬件互联**：
        - 研究 **PCIe Peer2Peer**：通过 `nvidia-smi` 验证硬件支持，编写代码测试 GPU 间直接内存访问（参考 Linux 内核 PCIe 子系统文档）。
        - 探索 **RDMA**：搭建 InfiniBand 环境，学习 verbs API，实现跨节点 GPU 内存远程访问。
- **驱动开发进阶**：
    - 从 **用户态驱动** 入手：基于 CUDA Runtime 编写多 GPU 管理工具（枚举设备、分配显存、同步任务），模拟驱动核心逻辑。
    - 尝试 **内核态驱动**：学习 Linux 字符设备驱动框架（`file_operations`、`ioctl`），实现简单的多 GPU 资源调度（如显存池管理），参考《Linux 设备驱动开发详解》。

#### **阶段 3：项目整合与实战（持续迭代）**

- **开源项目参与**：
    - 克隆 **NVIDIA multi-gpu programming models**（GitHub 或 CSDN 示例），学习 MPI+NCCL 等多 GPU 编程模型，修改并扩展示例（如多 GPU 图像处理）。
    - 贡献 **Horovod**（分布式训练框架）社区，优化 NCCL 通信性能，理解工业级多 GPU 训练流程。
- **硬件级实践**：
    - 搭建 **多 GPU 测试环境**：利用本地多 GPU 主机（如 NVIDIA RTX 6000 双卡）或云实例（AWS p3.8xlarge），测试 PCIe P2P、RDMA 功能。
    - 基于 **DOCA 框架**：开发多 GPU 数据加速应用（如网络分流、存储卸载），参考 DOCA 官方文档的多 GPU 示例。

### 三、可上手的实战项目推荐

#### **项目 1：多 GPU 矩阵乘法（NCCL 加速）**

- **目标**：实现多 GPU 协同的矩阵乘法，用 NCCL 聚合结果，对比单 GPU 性能。
- **步骤**：
    1. 拆分矩阵到多个 GPU，每个 GPU 计算子矩阵乘法。
    2. 通过 `ncclAllReduce` 合并结果，输出最终矩阵。
    3. 优化：调整分块策略、NCCL 通信参数（如线程数、缓冲区大小）。
- **技术栈**：CUDA、NCCL、C++，参考 NVIDIA 多 GPU 编程示例。

#### **项目 2：Linux 多 GPU 驱动原型（用户态）**

- **目标**：编写用户态工具，管理多 GPU 设备（枚举、显存分配、任务调度）。
- **步骤**：
    1. 调用 CUDA Runtime API（`cudaGetDeviceCount`、`cudaMalloc`）枚举 GPU、分配显存。
    2. 实现多线程调度：每个线程控制一个 GPU 执行计算任务（如向量加法）。
    3. 扩展：模拟多 GPU 互联，通过共享内存实现数据交换。
- **技术栈**：C++、CUDA Runtime、Linux 多线程。

#### **项目 3：多 GPU 分布式训练（PyTorch/TensorFlow）**

- **目标**：基于深度学习框架，实现多 GPU 同步训练，优化通信效率。
- **步骤**：
    1. 用 `torch.nn.parallel.DistributedDataParallel`（PyTorch）或 `tf.distribute.MirroredStrategy`（TensorFlow）配置多 GPU。
    2. 训练 ResNet-50 等模型，对比单 GPU 和多 GPU 的训练速度、显存占用。
    3. 优化：调整 batch size、使用混合精度（AMP）、定制 NCCL 通信策略。
- **技术栈**：PyTorch/TensorFlow、NCCL、深度学习模型。

#### **项目 4：PCIe Peer2Peer 通信测试**

- **目标**：验证双 GPU 间的直接内存访问，分析性能差异。
- **步骤**：
    1. 用 `nvidia-smi dmon` 检查 PCIe 带宽，确认 P2P 支持（`nvidia-smi topo -m`）。
    2. 编写 CUDA 程序：GPU0 分配内存，GPU1 直接读取并修改，验证数据一致性。
    3. 对比 P2P 与 CPU 中转的性能（时间、带宽）。
- **技术栈**：CUDA、Linux 性能分析（`nvidia-smi`、`nvprof`）。

### 四、学习资源与工具

- **官方文档**：NVIDIA CUDA/NCCL/DOCA 文档、Linux Kernel 文档（`Documentation/gpu/` 目录）。
- **书籍推荐**：《深入理解 Linux 内核》《CUDA Programming: A Developer's Guide》《并行计算：结构、算法和编程》。
- **实践工具**：`nvidia-smi`（硬件状态）、`nvprof`（性能分析）、`gdb`（调试）、云平台（AWS/GCP 多 GPU 实例）。

通过 **“理论学习→小项目实践→开源参与→硬件验证”** 的闭环，逐步掌握多 GPU 驱动与软件设计的核心能力！

# LLM 分布式 HPC

面向人工智能（尤其是 LLM）、分布式训练与高性能计算的系统知识和技能体系，需要融合**计算机体系结构、并行计算、分布式系统、AI 框架原理**等多领域知识，核心目标是高效处理 “大规模数据 + 超大模型” 的计算需求（高吞吐量、低延迟、高资源利用率）。以下是具体的知识框架：

### **一、基础理论层：构建底层认知**

#### 1. 数学与数值基础

- **线性代数**：矩阵运算（乘法、转置、分解）、向量计算（内积、外积）、张量操作（高维数据的切片 / 拼接 / 广播）—— 是 LLM 中注意力机制、线性层计算的核心。
- **数值分析**：浮点精度（FP32/FP16/BF16/INT8 的误差与稳定性）、数值优化（梯度下降的并行化实现）、数值稳定性（避免分布式计算中的精度丢失）。
- **概率论与统计**：理解模型训练中的随机过程（如采样、dropout 的并行实现）、分布式场景下的统计量聚合（如均值、方差的并行计算）。

#### 2. 计算机体系结构

- **硬件基础**：
    - CPU：缓存层次（L1/L2/L3）、指令集（x86 的 AVX2/512，ARM 的 NEON）、多核调度（超线程、核心绑定）。
    - GPU：CUDA 架构（SM 流多处理器、线程束 Warp、共享内存 Shared Memory、全局内存 Global Memory）、AMD ROCm 架构、张量核心（Tensor Core，用于混合精度计算）。
    - 加速器：TPU（脉动阵列 Systolic Array）、NPU（如昇腾）、FPGA（定制化计算）的核心原理。
    - 内存与存储：DRAM（带宽瓶颈）、HBM（高带宽内存，GPU 专用）、NVLink（GPU 间高速通信）、CXL（CPU 与加速器的缓存一致性协议）、NVMe（高性能存储）。
- **并行计算模型**：
    - SIMD（单指令多数据，如 CPU 向量化、GPU 线程束）、MIMD（多指令多数据，如分布式节点）、SPMD（单程序多数据，分布式训练的核心模式）。
    - 内存模型：共享内存（OpenMP）、分布式内存（MPI）、分布式共享内存（DSM）。

#### 3. 分布式系统理论

- **核心理论**：CAP 定理（一致性、可用性、分区容错的权衡）、一致性模型（强一致性、最终一致性，影响参数同步策略）、Paxos/Raft（分布式共识，用于元数据管理）。
- **通信模型**：
    - 消息传递：同步通信（阻塞 MPI_Send/Recv）、异步通信（非阻塞 MPI_Isend/Irecv）、集体通信（MPI_Allreduce/ReduceScatter，参数平均的核心）。
    - 通信开销：延迟（Latency）、带宽（Bandwidth）、拓扑依赖（如胖树 Fat-Tree vs torus 拓扑对通信效率的影响）。
- **调度与容错**：任务调度算法（FCFS、优先级调度、抢占式调度）、负载均衡（数据 / 任务分片策略）、容错机制（Checkpointing、弹性训练、故障检测与恢复）。

### **二、核心技术层：AI 系统与高性能计算的交叉**

#### 1. LLM 与深度学习系统基础

- **LLM 核心计算特性**：
    - Transformer 架构：多头注意力（QKV 矩阵乘法、Softmax 的并行化难点）、Feed-Forward 网络（大矩阵乘法）、层归一化（LayerNorm 的并行实现）。
    - 计算密集型 vs 通信密集型：注意力层是通信瓶颈（全局依赖），线性层是计算瓶颈（大矩阵乘法）。
- **并行策略**：
    - 数据并行（DP）：多设备复制模型，拆分数据（适用于小模型，通信量随设备数线性增长）。
    - 模型并行（MP）：拆分模型层或张量（如 Tensor Parallelism，拆分矩阵为块，降低单设备内存压力）。
    - 流水线并行（PP）：按层拆分模型，设备间流水线执行（隐藏层间依赖的延迟，如 Megatron-LM 的 1F1B 调度）。
    - 混合并行：ZeRO（DeepSpeed）、Megatron-LM 的 3D 并行（数据 + 张量 + 流水线），平衡内存与通信。
- **计算图与自动微分**：计算图的拆分与并行执行（静态图 TensorFlow vs 动态图 PyTorch）、反向传播的并行化（梯度计算的依赖关系、梯度同步策略）。

#### 2. 高性能计算（HPC）核心技术

- **并行编程模型**：
    - 共享内存：OpenMP（#pragma omp parallel for）、TBB（C++ 线程库）。
    - 分布式内存：MPI（Message Passing Interface，如 OpenMPI、MPICH）。
    - 异构计算：CUDA（GPU 核函数、线程配置、内存优化）、OpenACC（directives 指导异构并行）、HIP（跨 AMD/NVIDIA 的统一接口）。
- **高性能库**：
    - 线性代数：BLAS（Level 1/2/3，如 OpenBLAS）、LAPACK（矩阵分解）、cuBLAS（GPU 加速 BLAS）。
    - 深度学习原语：cuDNN（GPU 加速卷积、池化）、FlashAttention（优化注意力计算的库）。
    - 通信优化：NCCL（GPU 间集体通信，比 MPI 更高效）、RDMA（远程直接内存访问，低延迟通信）。

#### 3. 分布式训练系统

- **框架原理**：
    - PyTorch Distributed：后端（Gloo/NCCL/MPI）、通信原语（all_reduce、broadcast）、分布式数据加载（DistributedSampler）。
    - DeepSpeed：ZeRO（内存优化，分片优化器 / 梯度 / 参数）、混合精度训练（FP16/BF16）、MoE（混合专家模型）支持。
    - Megatron-LM：张量并行、流水线并行的实现细节，大批次训练的优化。
- **关键技术**：
    - 混合精度训练：FP16/BF16 的数值稳定性处理（损失缩放 Loss Scaling）、Tensor Core 利用。
    - 量化与压缩：INT8/INT4 量化（降低内存与计算量）、稀疏化（剪枝不重要的权重 / 激活）、知识蒸馏（小模型模仿大模型）。
    - 检查点（Checkpoint）：高效存储（如使用 Zarr/HDF5 压缩）、分布式 checkpoint（避免单节点 IO 瓶颈）、增量 checkpoint（只存变化的参数）。

### **三、工具与工程实践层：落地能力**

#### 1. 核心工具与框架

- **编程语言**：
    - 系统层：C/C++（高性能库开发）、CUDA C++（GPU 内核编写）、Rust（内存安全的系统工具）。
    - 应用层：Python（AI 框架接口）、Julia（高性能科学计算）。
- **AI 框架**：PyTorch（动态调试友好）、TensorFlow（静态图优化）、Megatron-LM（LLM 专用）、DeepSpeed（优化器）。
- **HPC 工具**：
    - 编译：GCC/Clang（CPU 优化）、NVCC（CUDA 编译）、MLIR（多级别中间表示，用于编译优化）。
    - 性能分析：perf（CPU 性能计数器）、nvprof/Nsight（GPU profiling）、nsys（系统级性能分析）、TAU（分布式程序分析）。
    - 集群管理：Slurm（HPC 集群调度）、Kubernetes（容器化集群，云原生场景）、Docker/Singularity（容器化部署）。

#### 2. 性能优化技能

- **计算优化**：
    - 数据局部性：缓存友好的数据布局（如矩阵转置优化缓存命中率）、循环展开（减少分支开销）、向量化（AVX2/512 指令手动 / 自动生成）。
    - GPU 优化：线程块（Block）与网格（Grid）配置（匹配 SM 数量）、共享内存复用（减少全局内存访问）、指令级并行（隐藏内存延迟）。
- **通信优化**：
    - 重叠计算与通信（非阻塞通信）、消息聚合（减少小消息数量）、通信拓扑优化（按网络结构调整数据分片）。
    - 利用硬件加速：NVLink（GPU 间高速互联）、InfiniBand（低延迟网络）、RDMA（绕过 CPU 直接访问内存）。
- **内存优化**：
    - 内存池（减少动态内存分配开销）、零拷贝（避免数据冗余复制）、内存对齐（提升访问效率）。
    - 显存优化：激活 checkpoint（反向传播时重计算激活值，换显存）、模型参数分片（ZeRO 的核心）。

#### 3. LLM 部署与推理系统

- **推理并行策略**：
    - 张量并行（拆分注意力 / 线性层，降低单设备计算量）、流水线并行（按层拆分，提高吞吐量）。
    - 动态批处理（vLLM 的 PagedAttention）、连续批处理（Triton Inference Server），提升 GPU 利用率。
- **服务化框架**：Triton Inference Server（多模型管理）、vLLM（高吞吐量 LLM 推理）、FastTransformer（优化 Transformer 推理）。
- **部署优化**：模型量化（GPTQ/AWQ）、模型蒸馏（减小模型大小）、边缘部署（TensorRT 优化、ONNX Runtime）。

### **四、前沿与交叉层：把握方向**

- **AI 系统研究热点**：
    - 自动并行（Auto-Parallelism，如 Alpa/AutoDist 自动选择并行策略）。
    - 编译优化（TVM/TensorRT，自动生成高性能内核）、MLIR（统一 AI 框架的中间表示）。
    - 硬件 - 软件协同设计（如 TPU 与 JAX 的协同优化）。
- **LLM 特有的挑战**：
    - 长上下文处理（如 100k+ tokens，内存高效的注意力实现）。
    - 多模态融合（文本 + 图像 + 语音的异构计算优化）。
    - 低资源训练 / 推理（联邦学习、边缘 LLM）。
- **交叉领域**：科学机器学习（AI+HPC，如用 LLM 求解偏微分方程）、云原生 AI（Kubernetes+AI 训练 / 推理）。

### **学习路径建议**

1. **打基础**：先掌握 C/C++、数据结构与算法，理解计算机体系结构（重点 CPU/GPU 缓存、指令集）。
2. **练并行**：从 OpenMP（共享内存）到 CUDA（GPU 编程），再到 MPI（分布式），用小案例（如矩阵乘法）实践并行优化。
3. **懂框架**：读 PyTorch Distributed 源码，理解数据并行实现；跑通 DeepSpeed/Megatron-LM 的示例，分析并行策略。
4. **做项目**：复现 LLM 训练（如用 16GB GPU 训练小模型，尝试模型并行），优化推理服务（如用 vLLM 部署开源模型）。
5. **追前沿**：读顶会论文（OSDI/SOSP/MLSys），关注 FAIR/DeepMind/Google Brain 的工程博客，理解工业界实践。

核心是 “理论 + 工具 + 实践” 结合，尤其注重**硬件特性与软件优化的匹配**（如 GPU 的 Tensor Core 必须用 FP16 矩阵乘法才能发挥性能），以及**分布式场景下的通信与内存瓶颈解决**。
