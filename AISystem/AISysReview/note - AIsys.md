---
dateCreated: 2025-02-11
dateModified: 2025-08-08
---

相关资源推荐：https://zhuanlan.zhihu.com/p/20076957712

AI 相关大学课程严忻恺：https://www.zhihu.com/people/yan-xin-kai-38/posts

https://www.mlsysbook.ai/

个人博客：https://shichaoxin.com/

https://mlsys-learner-resources.github.io/Awesome-MLSys-Blogger/

https://hao-ai-lab.github.io/cse234-w25/

https://goodcucumber.github.io/x40paraguide/x40.html

https://github.com/ForceInjection/AI-fundermentals/tree/main

综述：

- https://zhuanlan.zhihu.com/p/101544149
- https://blog.csdn.net/qq_42722197/article/details/119814538
- https://zhuanlan.zhihu.com/p/616138047
- <a href="https://zhuanlan.zhihu.com/p/33876622">一天搞懂</a>
- <a href=" https://zhuanlan.zhihu.com/p/20076957712?utm_psn=1872615359586111488">知乎总结</a>
- <a href=" https://chenzomi12.github.io/index.html">ZOMI AI 系统</a>
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

# LLM Infra

大模型（亿级及以上参数）的知识体系是硬件、软件、算法、工程优化的深度融合，需要从 “底层硬件支撑” 到 “上层模型应用” 全链路贯通。以下从核心模块拆解知识体系，并附构建路径：

### 一、硬件基础：大模型的 “物理载体”

大模型的硬件基础核心是 “算力 - 存储 - 互联” 三位一体的支撑体系，决定了模型能否训练 / 推理（能否跑起来）、效率（跑多快）、成本（跑多便宜）。

#### 1. 核心计算硬件

- **加速芯片**：大模型的 “核心算力源”
    - GPU：主流选择（NVIDIA A100/H100，AMD MI250），优势是生态完善（CUDA）、算力密度高（H100 单卡 FP8 算力达 4PFlops）
    - 专用 ASIC：如 Google TPU（针对 Transformer 优化）、AWS Trainium/Inferentia、国内昇腾 910/310，优势是性价比（针对特定场景定制）
    - CPU：辅助角色（控制流、小批量预处理），依赖多核高主频（如 Intel Xeon、AMD EPYC）
- **算力集群架构**：单卡算力有限，需通过集群扩展（万卡级是大模型训练常态）
    - 单机多卡：通过 NVLink（GPU 间高速互联，H100 支持 900GB/s）实现卡内通信
    - 多机集群：通过 Infiniband（400Gbps+）或 RoCE（基于以太网的高速互联）实现跨机通信

#### 2. 存储体系

- **内存 / 显存**：模型参数、中间激活值的 “临时存储”
    - 高带宽显存（HBM）：GPU 核心存储（H100 HBM3 带宽达 5TB/s），决定单卡能承载的模型分片大小
    - 主机内存（DDR5）：CPU 侧缓存，用于数据预处理、参数临时加载
- **持久化存储**：训练数据、模型权重、日志的 “长期存储”
    - 分布式文件系统：如 Ceph、Lustre（支持 PB 级容量，满足千亿级参数模型权重存储，单模型权重可达 TB 级）
    - 对象存储：如 S3、OSS（用于训练数据（文本、图像）的海量存储，支持高并发读取）

#### 3. 网络互联

- **核心需求**：大模型训练 / 推理依赖分布式计算，跨设备 / 跨节点通信延迟和带宽是瓶颈
- **关键技术**：
    - 节点内：NVLink（GPU 间）、PCIe 5.0（CPU 与 GPU）
    - 节点间：Infiniband HDR/EDR（延迟 < 1us，带宽 400Gbps）、RoCE（成本更低，适合中小集群）
    - 网络拓扑：Fat-Tree（胖树）架构，避免跨节点通信拥塞

### 二、软件框架：大模型的 “操作系统”

软件框架是连接硬件与模型的 “中间层”，负责算力调度、分布式协同、内存管理等核心功能。

#### 1. 底层计算框架

- **核心框架**：模型训练 / 推理的 “基础引擎”
    - PyTorch/TensorFlow：主流深度学习框架，提供自动微分、算子库（如 PyTorch 的 TorchScript）
    - 专用分布式框架：针对大模型优化，解决单卡放不下、单节点算力不足的问题
        - DeepSpeed（微软）：支持 ZeRO 内存优化、混合精度训练、推理加速
        - Megatron-LM（NVIDIA）：专注 Transformer 分布式训练，支持张量并行、管道并行
        - Colossal-AI（国内）：整合多种并行策略，适配国产硬件

#### 2. 编译与优化工具

- **作用**：将模型代码 “翻译” 为硬件可高效执行的指令，提升计算效率
    - 算子优化：TensorRT（NVIDIA）、TVM（跨硬件）、Ascend C（昇腾），通过算子融合、低精度计算加速
    - 自动并行：Alpa（谷歌）、Mesh TensorFlow，自动划分模型到分布式集群，减少人工调参
    - 编译优化：XLA（TensorFlow）、TorchInductor（PyTorch 2.0+），将动态图转为静态图编译执行

#### 3. 集群与资源管理

- **集群调度**：管理万级节点的算力分配（避免资源浪费）
    - Slurm：高性能计算（HPC）常用调度器，支持任务排队、资源隔离
    - Kubernetes（K8s）：容器化调度，适合云原生场景（弹性扩缩容）
- **监控与运维**：确保集群稳定运行
    - 监控工具：Prometheus（指标采集）、Grafana（可视化）、Nvidia DCGM（GPU 状态监控）
    - 故障处理：自动容错（如 DeepSpeed 的 Checkpoint 恢复）、节点健康检测

### 三、模型架构：大模型的 “骨架设计”

模型架构是大模型的 “先天基因”，决定了参数量、计算量、表达能力，也影响后续优化难度。

#### 1. 基础架构范式

- **Transformer 主导**：当前大模型（LLM、多模态）的核心架构
    - 核心组件：多头自注意力（计算密集，O (n²) 复杂度）、FeedForward 网络（FFN）、LayerNorm、残差连接
    - 变体优化：
        - 注意力简化：如 FlashAttention（减少内存访问）、SwiGLU（替换 FFN 的激活函数，提升效率）
        - 稀疏化：如 Longformer（局部注意力）、GPT-4（可能用的混合注意力），降低长文本计算量
- **MoE（混合专家模型）**：提升参数量但控制计算量的 “性价比架构”
    - 设计：将 FFN 替换为 “专家网络 + 路由层”，输入仅路由到部分专家（如 1/16），参数量可扩展到万亿级但计算量不变
    - 代表：GLaM、Switch Transformer、GPT-4（疑似用 MoE）

#### 2. 模型压缩与轻量化

- **核心目标**：在精度损失可接受的前提下，降低参数量 / 计算量（便于部署）
    - 量化：INT8/INT4/FP4（如 GPTQ、AWQ），将 32 位参数转为低精度，显存占用减少 4-8 倍
    - 剪枝：移除冗余参数（如注意力头剪枝、神经元剪枝），保留核心结构
    - 知识蒸馏：用大模型 “教” 小模型（如 DistilBERT），继承能力但缩小体积

### 四、计算优化：大模型的 “效率引擎”

大模型计算成本极高（训练一次千亿模型成本千万级），计算优化是核心竞争力，目标是 “用更少资源跑更快”。

#### 1. 并行计算策略

- **核心问题**：单卡放不下模型（参数 + 激活值）、单节点算力不足，需拆分模型到多设备
- **三大并行范式**：
    - 数据并行（DP）：多卡复制相同模型，各处理部分数据，适合数据量大但模型小的场景（低效：参数重复存储）
    - 模型并行（MP）：
        - 张量并行（TP）：拆分单一层的权重矩阵到多卡（如将 1024 维权重拆到 2 卡，各处理 512 维），适合大层（如注意力、FFN）
        - 管道并行（PP）：按层拆分模型（如层 1-10 在卡 1，层 11-20 在卡 2），各卡依次计算，适合层数多的模型
    - 专家并行（EP）：MoE 架构专用，专家网络分布在不同卡，路由层协调调用
- **混合并行**：结合 TP+PP+EP（如 Megatron-LM + MoE），支持万亿级模型

#### 2. 内存优化

- **核心问题**：训练时激活值（中间结果）占用显存可能是参数的 10 倍以上（如 BERT-large 激活值占 10GB+）
- **优化手段**：
    - 激活检查点（Checkpointing）：只保存部分层的激活值，反向传播时重新计算其他层（时间换空间）
    - ZeRO（DeepSpeed）：按阶段拆分优化器状态、梯度、参数，单卡仅存 1/N（N 为卡数），显存占用降低 10 倍
    - 内存复用：动态释放无用中间变量（如 PyTorch 的 autograd 内存池）

#### 3. 低精度计算

- **核心逻辑**：用低精度（FP16/BF16/FP8/INT8）替代 FP32，减少计算量和内存访问
- **训练优化**：
    - 混合精度：FP16 计算 + FP32 保存参数（避免精度丢失），如 Apex（NVIDIA）
    - FP8 训练：H100 支持 FP8，算力是 FP16 的 2 倍，精度损失可控（需调参补偿）
- **推理优化**：INT8/INT4（如 TensorRT-LLM），计算效率提升 4-8 倍，配合量化校准（如 KL 散度校准）保证精度

### 五、推理训练优化：大模型的 “落地保障”

训练优化目标是 “稳定训完、成本可控”，推理优化目标是 “低延迟、高吞吐、高并发”（满足实际应用）。

#### 1. 训练优化

- **稳定性保障**：
    - 梯度裁剪：避免梯度爆炸（大模型易出现）
    - 混合精度训练：减少数值溢出（FP16 易下溢，用 BF16 更稳定）
    - 检查点（Checkpoint）：定期保存模型状态，中断后可恢复（支持增量保存，减少 IO）
- **效率提升**：
    - 梯度累积：小批量累加梯度再更新，模拟大 batch（避免单 batch 太大撑爆显存）
    - 动态 batch：根据显存自动调整 batch size（如 DeepSpeed 的 AutoTP）

#### 2. 推理优化

- **核心指标**：延迟（单请求响应时间）、吞吐（单位时间处理请求数）、显存占用
- **关键技术**：
    - KV 缓存：推理时缓存注意力的 KV 对（避免重复计算，显存换速度）
    - 动态批处理：合并多个请求为一个 batch（如 Triton Inference Server），提升 GPU 利用率
    - 推理引擎：TensorRT-LLM、vLLM（PagedAttention）、FastTransformer，优化算子和内存管理
    - 量化推理：INT8/INT4（如 AWQ，量化后精度损失 < 1%），显存和速度双重优化
- **服务化部署**：
    - 动态扩缩容：根据请求量自动调整 GPU 资源（K8s + 推理服务）
    - 负载均衡：多实例 / 多节点分摊请求（避免单点过载）

### 六、核心挑战

1. **成本壁垒**：硬件（单 H100 卡 20 万 +）、电力（万卡集群年电费千万级）、人力（优化工程师稀缺）
2. **效率瓶颈**：通信开销（分布式计算中跨节点通信占比可达 30%+）、内存墙（激活值 / 参数存储限制模型规模）
3. **稳定性**：训练中硬件故障（如 GPU 掉卡）、数值不稳定（大模型易发散）
4. **工程复杂度**：并行策略调参（TP/PP 拆分比例）、硬件适配（国产芯片生态不完善）

### 七、常用工具链

|类别|核心工具 / 框架|作用|
|---|---|---|
|训练框架|PyTorch + DeepSpeed/Megatron-LM|分布式训练核心引擎|
|推理引擎|vLLM、TensorRT-LLM、FastTransformer|推理加速与部署|
|编译优化|TensorRT、TVM、TorchInductor|算子优化与低精度计算|
|集群管理|Slurm、Kubernetes + Volcano|算力调度与资源管理|
|监控运维|Prometheus、Grafana、DCGM|硬件 / 任务状态监控|
|量化工具|AWQ、GPTQ、TensorRT Quantization|模型低精度量化|

### 八、知识体系构建路径

1. **打基础**：从 “硬件 + 框架” 切入

    - 硬件：理解 GPU 架构（SM、HBM、NVLink）、分布式集群基本原理
    - 框架：熟悉 PyTorch 基础（张量、自动微分），再学 DeepSpeed（跑一个小模型的分布式训练）
2. **学架构**：掌握 Transformer 核心（注意力计算、残差连接），对比 MoE 与传统架构的差异，理解模型设计对计算的影响（如注意力的 O (n²) 复杂度为何是瓶颈）
3. **练优化**：从单卡优化到分布式

    - 单卡：学习量化（用 GPTQ 量化一个模型）、算子优化（用 TensorRT 加速推理）
    - 分布式：用 Megatron-LM 实现张量并行，对比不同并行策略的效率（如 TP vs PP）
4. **看实践**：跟踪工业界案例（如 OpenAI 的训练基建、阿里云 PAI 的大模型方案），理解工程落地中的权衡（如成本与速度的平衡）
5. **抓核心**：始终围绕 “资源（硬件）- 效率（优化）- 目标（训练 / 推理）” 的关系，比如 “为什么 MoE 适合超大模型？”“低精度推理为什么在推理中更常用而训练中谨慎？”

大模型 infra 是 “硬科技 + 软创新” 的结合，既需要硬件底层知识（如 GPU 显存带宽），也需要算法优化能力（如并行策略设计），更需要工程落地经验（如集群稳定性保障）。从具体问题出发（如 “如何用 8 卡训练一个 100 亿参数模型”），逐步串联各模块知识，是最有效的学习方式。

# LLM Infra 知识构建

（核心是打通 “硬件设计→系统软件→模型算法→部署落地” 的全链条），建议构建一套 **“垂直穿透 + 横向协同”** 的知识体系，既覆盖各环节的技术细节，又突出 “跨层优化” 这一 AI 系统的核心逻辑。

### **1. LLM 模型基础与核心技术**

（算法层：理解模型本质，是系统优化的前提）

├── LLM 核心原理

│ ├── Transformer 架构细节（注意力机制、FFN、位置编码）

│ ├── 预训练技术（数据清洗、tokenization、训练目标如 MLM）

│ ├── 微调与对齐（LoRA、RLHF、DPO 等参数高效微调方法）

│ └── 多模态扩展（LLaVA、GPT-4V 等跨模态融合机制）

├── LLM 模型优化算法

│ ├── 量化技术（INT4/8、AWQ、GPTQ 等量化方法与精度恢复）

│ ├── 剪枝与稀疏化（结构化 / 非结构化剪枝、动态稀疏训练）

│ ├── 模型压缩（知识蒸馏、MoE 架构设计与路由优化）

│ └── 长上下文扩展（FlashAttention、RoPE 扩展、窗口注意力等）

### **2. AI 系统软件栈（LLM Infra 核心）**

（系统层：连接模型与硬件的桥梁，决定部署效率）

├── 训练框架与调度

│ ├── 分布式训练框架（PyTorch Distributed、DeepSpeed、Megatron-LM）

│ ├── 并行策略（数据并行、模型并行、张量并行、流水线并行）

│ ├── 训练调度（ZeRO 优化、内存高效训练技巧、混合精度训练）

│ └── checkpoint 管理与训练中断恢复

├── 推理框架与优化

│ ├── 推理引擎（vLLM、TensorRT-LLM、ONNX Runtime、FasterTransformer）

│ ├── 推理优化（KV 缓存管理、批处理调度、连续批处理（Continuous Batching））

│ ├── 动态请求处理（多用户请求调度、优先级机制、推理延迟优化）

│ └── 多模态推理管线（文本 - 图像 / 语音的联合推理流程）

├── AI 编译器与中间表示

│ ├── 编译器架构（TVM、MLIR、XLA 的 IR 设计与优化流程）

│ ├── 算子优化（LLM 核心算子（如 attention）的自动代码生成与调优）

│ ├── 硬件感知编译（针对 GPU/ASIC 的算子映射与指令选择）

│ └── 动态形状处理（LLM 变长输入场景的编译器适配）

### **3. AI 硬件加速与架构设计**

（硬件层：系统性能的物理基础，需理解硬件特性以做协同优化）

├── 通用加速硬件

│ ├── GPU 架构与 LLM 适配（NVIDIA Hopper/Ampere 的 Tensor Core、共享内存优化）

│ ├── 数据中心级 AI 芯片（如 AMD MI300、寒武纪思元、特斯拉 D1 的架构特点）

│ └── 内存与存储架构（HBM 带宽优化、NVLink/PCIe 通信效率、存储墙突破）

├── 专用 AI 加速芯片（ASIC）

│ ├── LLM 专用加速器设计（如 Google TPU v4、Graphcore IPU 的架构目标）

│ ├── 计算单元设计（ systolic array 、向量单元在 LLM 算子中的效率）

│ └── 片上网络（NoC）与多芯片互联（针对分布式 LLM 的通信优化）

├── 边缘与低功耗硬件

│ ├── 移动端 LLM 加速（如 Qualcomm NPU、苹果 Neural Engine 的部署限制与优化）

│ └── RISC-V 架构在 AI 加速中的应用（自定义指令集、向量扩展与 LLM 适配）

### **4. LLM 训练与推理工程实践**

（工程层：将技术落地的核心能力，覆盖全流程工具与问题解决）

├── 训练工程

│ ├── 超大规模集群搭建（节点配置、网络拓扑、GPU 集群调度）

│ ├── 训练稳定性保障（梯度爆炸 / 消失处理、NaN 检测与恢复、容错机制）

│ └── 训练效率指标（TFLOPS 利用率、内存占用优化、千卡级集群调优）

├── 推理部署工程

│ ├── 云原生部署（Kubernetes 调度 LLM 服务、容器化优化、资源隔离）

│ ├── 服务化架构（API 设计、负载均衡、弹性扩缩容、多模型服务编排）

│ ├── 边缘部署（模型裁剪适配、低功耗模式、本地推理延迟优化）

│ └── 性能监控与调优（ latency/P99 指标、吞吐量优化、瓶颈定位工具）

### **5. 跨层协同与系统优化**

（核心竞争力：突破 “硬件 - 软件 - 模型” 的孤岛，实现端到端效率）

├── 软硬件协同设计

│ ├── 算子与硬件特性匹配（如将 attention 映射到 Tensor Core 的最佳策略）

│ ├── 内存层级优化（寄存器→L1→HBM 的数据 locality 设计）

│ └── 编译时与运行时协同（静态优化 + 动态调度的混合策略）

├── 模型 - 系统协同优化

│ ├── 模型结构与推理效率适配（如设计更 “硬件友好” 的 LLM 变体）

│ ├── 量化与硬件计算精度协同（如 INT4 计算单元与模型量化的误差补偿）

│ └── 动态请求与资源分配协同（根据输入长度动态调整计算 / 存储资源）

### **6. 工具链与前沿研究**

（辅助层：提升效率的工具与跟踪技术演进）

├── 开源工具与生态

│ ├── LLM 开发工具（Hugging Face 生态、vLLM/TensorRT-LLM 实践）

│ ├── 性能分析工具（NVIDIA NSight、TensorBoard Profiler、PerfDog）

│ └── 硬件仿真与验证工具（Gem5、Xcelium 在 AI 加速芯片验证中的应用）

├── 前沿方向跟踪

│ ├── 下一代 AI 加速架构（存算一体、光计算在 LLM 中的潜力）

│ ├── LLM 能效优化（低比特训练、绿色 AI 计算框架）

│ └── 去中心化 LLM（联邦学习、分布式推理的安全与效率平衡）

### 体系特点

1. **垂直穿透**：从 “模型算法”（最上层）到 “硬件架构”（最底层），每层知识都需理解 “上层需求如何驱动下层设计，下层特性如何约束上层优化”。
2. **聚焦 LLM 特性**：突出 LLM 的 “大模型、长上下文、高算力需求” 特点，比如专门强化 “KV 缓存优化”“分布式并行策略”“超长文本推理” 等场景。
3. **工程导向**：加入大量 “训练 / 推理工程实践” 内容，避免纯理论，强调 “问题 - 方案 - 工具” 的落地逻辑（如 “如何定位推理延迟瓶颈”“用什么工具调优 GPU 利用率”）。

这套体系能帮你在工作中快速定位问题（比如推理延迟高，可能需要从 “推理框架调度”“硬件内存带宽”“模型量化策略” 三个层面排查），并形成 “系统思维” 而非孤立看待某一环节。
