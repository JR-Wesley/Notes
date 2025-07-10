---
dateCreated: 2025-07-08
dateModified: 2025-07-09
---

一面

项目，算子开发，cuda

静态链接，动态链接

红黑树，具体带系数的时间复杂度

内存泄漏，怎么解决

模板特化，偏特化，模板实例化是在哪个阶段，模板怎么拒绝一个类型

智能指针，shared_ptr 是线程安全的吗？

多线程和多进程，应用场景

协程

进程间通信，应用场景

二面

python list 去重

python 装饰器，作用

python 内存管理

linux 查看文件大小，查看网络状态

https://cppguide.cn/

ailab 高性能算子 -1 面

项目拷打 1，重点和部署量化流程和 gridsample 算子的优化以及算子的底层定义

项目拷打 234，重点是模型结构拷打

介绍一下访存密集型算子

介绍一下 fp32fp16int8 是怎么存储的

介绍一下量化原理以及过程

介绍一下 gpu 和 cpu 的结构以及适合计算什

udp 和 tcp 协议适合什么各自的优缺点

linux 查找磁盘使用指令

算法：链表判断有环➕cuda 实现向量加

ailab2 面

项目拷打重点同上

介绍一下锁的概念

介绍一下交叉熵的理解

介绍一下 cuda 编程的并行性和并发性

介绍一下 gan 和 cvae

手撕一下 svm

手撕一下 transformer

沐曦集成—ai 系统架构

项目拷打重点同上

介绍一下 cuda 编程的并行性和并发性

介绍一下 c++ 编程的三大特性

介绍一下 map 和 unorderedmap 底层实现

介绍一下 new，malloc，智能指针

介绍一下 new，malloc 的底层原理

介绍一下 lambda

手撕一下 transformer

算法：两数之和还有一个是 hash 具体啥忘了

随着大模型与生成式 AI 的爆发式增长，AI 基础设施正面临前所未有的性能、规模与效率挑战。该岗位致力于培养构建下一代 AI 系统底座的领军人才，具备软硬协同、跨层优化的知识面和技术深度，支撑集团核心 AI 业务的训练推理提效、集群资源调度及异构算力协同优化，推动 AI 技术的边界突破。核心问题包括但不限于：1. 极致性能优化：探索算法、训推引擎和基础设施的 co-design 协同突破效率瓶颈，最大化算力、网络和存储等硬件性能。2. 高性能网络：负责设计、实现、维护 AI 和高性能计算所需要的高性能网络通信框架和大模型推理场景的性能优化，聚焦模型通信场景的能力建设，完善集合通信、点对点通信等通信方式与推理框架的联合方案设计，推动提升推理性能。3. 智能资源调度：针对大规模分布式的 LLM/多模态理解生成训练推理等新兴计算场景，优化多集群多地域的异构调度编排能力，实现分钟级模型分发、训推任务弹性伸缩等。4. 其他随着 AI 模型、训推范式、算力硬件等迭代演进而出现的 AI 系统优化工程挑战和业界难题。

职位要求

1. 分布式系统、计算机体系结构、编译优化或通信与计算协同设计方向的硕/博士研究生。2. 具备 AI 训推计算性能分析与优化的经验，能深入分析 AI 模型在 GPU 平台上的性能瓶颈，提出并实施优化方案。针对分布式训练和推理系统，进行性能调优，提升系统的吞吐量和效率。3. 熟悉业界常见的优化栈（cuda/rocm/cutlass/ck/triton 等），在高效的内存管理、通信优化（NvLink/Infiniband/RoCEv2 等）关键技术上有实操经验。4. 分布式系统研发经验是加分项：设计和实现高效的分布式训练和推理框架，解决大规模分布式系统中的通信、同步和负载均衡问题。探索新型的分布式架构，提升系统的可扩展性和容错性。5. 前沿技术研究：跟踪 AI Infra 领域的最新研究进展，探索新的硬件架构、算法和系统优化技术。发表高水平学术论文，参与国际顶级会议（如 ISCA、MICRO、OSDI、SOSP、ATC、NSDI 等）。

2,熟悉 [后向误差传播算法](https://zhida.zhihu.com/search?content_id=153124779&content_type=Answer&match_order=1&q=%E5%90%8E%E5%90%91%E8%AF%AF%E5%B7%AE%E4%BC%A0%E6%92%AD%E7%AE%97%E6%B3%95&zhida_source=entity)（BP），完成从标量求导到矩阵求导思维方式的转换，熟悉常见算子的梯度推导（矩阵乘，卷积，池化，Relu，如果会 batch normalization 就一步到位了）；

3，熟悉 [autograd](https://zhida.zhihu.com/search?content_id=153124779&content_type=Answer&match_order=1&q=autograd&zhida_source=entity) 的基本原理，能自己手撸一个最好；

4，熟悉 cuda 编程（举一反三），熟悉 cuda 高阶用法，event, stream, 异步/同步，会优化常见 cuda kernel, element-wise, reduce, broadcast，MatMul, conv, pooling 等；

5，熟悉 c++ 和 python, 对 c++ 高级用法感到舒服，各种模式，惯用法，模板；熟悉 vim, [gdb](https://zhida.zhihu.com/search?content_id=153124779&content_type=Answer&match_order=1&q=gdb&zhida_source=entity) 程序调试；

6，熟悉 socket, [RDMA编程](https://zhida.zhihu.com/search?content_id=153124779&content_type=Answer&match_order=1&q=RDMA%E7%BC%96%E7%A8%8B&zhida_source=entity)，熟悉常见 [collective operation](https://zhida.zhihu.com/search?content_id=153124779&content_type=Answer&match_order=1&q=collective+operation&zhida_source=entity) 代价分析，譬如 ring allreduce, tree allreduce 代价分析；

7，熟悉多线程编程，熟悉锁，条件变量，内核线程，用户级线程，对 actor, CSP(coroutine) 各种技术熟悉；

8，熟悉 [编译器基本原理](https://zhida.zhihu.com/search?content_id=153124779&content_type=Answer&match_order=1&q=%E7%BC%96%E8%AF%91%E5%99%A8%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86&zhida_source=entity)，parser 什么的不重要，主要是 [dataflow分析](https://zhida.zhihu.com/search?content_id=153124779&content_type=Answer&match_order=1&q=dataflow%E5%88%86%E6%9E%90&zhida_source=entity)，灵活运用；熟悉多重循环程序优化技巧，譬如 polyhedral 模型；

9，熟悉常见 [分布式系统原理](https://zhida.zhihu.com/search?content_id=153124779&content_type=Answer&match_order=1&q=%E5%88%86%E5%B8%83%E5%BC%8F%E7%B3%BB%E7%BB%9F%E5%8E%9F%E7%90%86&zhida_source=entity)，mapreduce, spark, flink, tensorflow 等；

10，熟悉 [计算机体系机构](https://zhida.zhihu.com/search?content_id=153124779&content_type=Answer&match_order=1&q=%E8%AE%A1%E7%AE%97%E6%9C%BA%E4%BD%93%E7%B3%BB%E6%9C%BA%E6%9E%84&zhida_source=entity)，量化分析方法，Amdahl' Law, Roofline Model, 流水线分析（譬如 David Patterson 那本书）；

11，熟悉操作系统原理及常用系统诊断工具，譬如各种资源利用率分析；

12，programming language 原理，命令式编程，函数式编程，逻辑编程，入门书《程序的构造与解释》？

13，熟悉项目构建原理，compiler, assembler, linker，loader 之类，有一本书《程序员的自我修养》有比较全面覆盖。

https://ucbrise.github.io/cs294-ai-sys-fa19/
https://zhuanlan.zhihu.com/p/608318764?share_code=6kgRiae9U4sA&utm_psn=1926408052585793087

# 知识体系

- 计算机
1. 操作系统
2. C++、C 等编程语言
3. 软件架构设计和优化、性能分析和调优
4. 数据结构与算法
5. linux
6. Qemu 模拟器 gem 5 / gpgpusim 等仿真工具框架
7. 集合通信原语（如 AllReduce, AllGather）和底层原理，RDMA 等高速网络通信技术

- 体系结构
1. CPU 流水线、缓存、分支预测等微架构原理
2. GPU
3. 可编程芯片（NPU/TPU）架构
4. AI 加速器
5. QEMU/GEM5 仿真模拟器

- 深度学习与框架
1. Python
2. LLVM/MLIR/TVM
3. 深度学习模型和框架（如 TensorFlow/Pytorch/Megatron/DeepSpeed）
4. ONNX PTX SASS
5. 主流模型 NLP/CV 模型架构与算法，MLP、CNN、LSTM、MHA、MLA、MOE、NLP (Bert), LLM (Transformer), diffusers

- 分布式训练或 HPC (高性能计算)
1. 分布式并行策略及其挑战（如数据并行、模型并行、流水线并行、、张量并行、序列并行、Zero 冗余优化器等）、内核级优化（如算子融合、内存管理优化、通信优化）
2. 主流框架（如 PyTorch 生态下的框架）适配到新型硬件（如 GPGPU/NPU/加速卡）。
3. 主流分布式训练/推理框架（如 DeepSpeed, Megatron-LM, Colossal-AI, FSDP, vLLM, SGLang, Hugging Face Accelerate/Transformers 等）千亿参数级别大模型训练或实战经验、推理优化
4. 高效的内存管理、通信优化（NvLink/Infiniband/RoCEv2 等）
5. 分析性能瓶颈（如通信开销、计算效率、内存限制）和可能的精度问题

- 算法
1. 数值计算、线性代数相关算法有深刻的理解
2. 卷积、矩阵乘、矩阵分解、BatchNorm、flash attention 密集型算子优化

- 算子并行优化
1. 并行编程基础：有 CUDA/OpenCL/OpenMP
2. GPU 高性能算子开发与优化，工具 Nsight Systems compute, DLProf, PyTorch Profiler, TensorBoard
3. Triton、TVM、MLIR 等深度学习专用编译器或编译器组件、编译器技术（如 TVM, MLIR, LLVM）在深度学习的应用。
4. NCCL，NCCL、NVSHMEM 或其他分布式计算相关，MPI 开发、RDMA

- 数字集成电路
1. 互联通信协议
2. 算法硬件实现
3. 电路面积、时序、功耗优化
4. 硬件测试验证流程和工具（vcs, verdi, verilator, etc.）
5. SOC 和 IP 的架构/微架构设计探索 + 性能模型建模，包含不限于核心并行计算处理器、NOC、Cache、MMU、Memory、ESL/RDMA、die-to-die、一致性协议、DMA、etc。
6. 3 D 堆叠，chiplet

# AI Sys

在 AI 芯片公司，大语言模型（LLM）从算法定义到最终在自研硬件上部署，需要经历**7 个核心系统层次**的协同设计，每个层次都需适配芯片特性（如计算单元架构、存储层次、互联方式），同时解决性能、能效和兼容性问题。以下是各层次的具体设计要点：

### **一、算法层：模型结构与训练目标定义**

- **核心任务**：确定 LLM 的基础架构（如 Transformer 变体）、参数量（如 7B/70B / 千亿级）、训练目标（如预训练 / 微调）和推理场景（如对话 / 生成）。
- **设计要点**：
    - 针对芯片计算特性调整模型结构（如若芯片支持稀疏计算，可设计稀疏注意力机制）；
    - 确定量化策略（如 FP16/INT8/FP8），平衡精度与算力利用率（自研芯片可能有专用低精度计算单元）；
    - 优化序列长度（如支持动态上下文窗口），适配芯片的片上存储容量（如 HBM 带宽）。

### **二、框架层：分布式训练 / 推理框架适配**

- **核心任务**：将 LLM 模型代码（如基于 PyTorch/TensorFlow）适配到自研芯片，通过分布式框架实现大规模训练 / 推理。
- **设计要点**：
    - **框架移植**：修改主流框架（如 Megatron-LM、vLLM）的硬件接口，将模型计算映射到芯片的计算核（如替换 CUDA 调用为芯片专用 API）；
    - **并行策略**：设计混合并行方案（数据并行 + 张量并行 + 流水线并行），例如：
        - 用张量并行拆分 Transformer 层的 QKV 计算，适配芯片的多核集群架构；
        - 用流水线并行处理超长序列，避免单芯片内存溢出；
    - **通信适配**：将框架的集合通信（如 AllReduce）绑定到芯片的互联协议（如 PCIe/NVLink 类似的自研链路），优化跨芯片数据传输。

### **三、算子层：高性能计算内核开发**

- **核心任务**：为 LLM 的关键算子（如注意力、矩阵乘法、激活函数）开发芯片专用实现，最大化硬件利用率。
- **设计要点**：
    - **算子映射**：将 Transformer 的核心计算（如 MatMul、Softmax）拆解为芯片支持的指令集（如张量计算单元 TCU 的专用指令）；
    - **算子优化**：
        - 利用芯片的存储层次（如片上 SRAM 缓存）减少访存延迟（如矩阵分块适配缓存大小）；
        - 算子融合（如 QKV 计算 + 注意力掩码融合），减少中间数据读写；
        - 稀疏计算优化（如跳过零值特征），适配芯片的稀疏加速单元；
    - **性能调优**：通过芯片性能计数器（如计算单元利用率、内存带宽）调整算子实现（如线程块大小、数据布局）。

### **四、编译层：模型到硬件指令的转换**

- **核心任务**：将 LLM 的计算图（由框架生成）编译为芯片可执行的机器码，完成优化（如指令重排、内存分配）。
- **设计要点**：
    - **计算图优化**：基于芯片架构进行图剪枝、算子合并（如将多层 BN+ReLU 合并为单指令）；
    - **指令生成**：通过编译器（如基于 TVM/MLIR 定制）将算子转换为芯片的微指令流，利用指令级并行（ILP）提升效率；
    - **内存调度**：优化数据在片上 / 片外存储的分配与搬运（如预取策略），避免计算单元空闲；
    - **硬件适配**：针对芯片的特殊功能（如动态电压调节、多精度计算）生成适配指令（如自动切换 FP16/INT8 计算模式）。

### **五、运行时（Runtime）层：任务调度与资源管理**

- **核心任务**：管理芯片的计算资源、内存和通信，协调多芯片 / 多节点的协同执行。
- **设计要点**：
    - **任务调度**：将编译后的指令分发到芯片的计算核心，支持多流并行（如计算与数据传输重叠）；
    - **内存管理**：
        - 分配芯片的 HBM/SRAM 资源（如为注意力权重分配高带宽存储）；
        - 实现内存池复用，减少动态分配开销；
    - **通信管理**：封装芯片间的互联接口（如自研高速链路），提供集合通信 API（如 AllReduce、Broadcast），支持分布式训练的梯度同步；
    - **故障处理**：检测芯片错误（如计算超时、内存错误），实现任务重试或故障节点隔离。

### **六、驱动层：硬件抽象与控制**

- **核心任务**：作为 Runtime 与硬件的接口，将高层指令转换为芯片的物理操作（如寄存器配置、时钟控制）。
- **设计要点**：
    - **硬件抽象**：封装芯片的底层寄存器、计算单元、存储控制器，提供统一的软件调用接口（如初始化、启动 / 停止计算）；
    - **资源隔离**：控制多进程 / 多任务对芯片资源的访问（如通过 PCIe BAR 空间隔离），避免冲突；
    - **性能监控**：读取芯片的传感器数据（如温度、功耗），反馈给 Runtime 进行动态调频（如高负载时提升核心频率）；
    - **兼容性**：适配 Linux 内核驱动框架（如 PCIe 设备驱动模型），确保芯片可被操作系统识别和管理。

### **七、硬件层：芯片物理实现与部署**

- **核心任务**：将上述软件层的需求映射到芯片的物理设计，最终部署为单机 / 集群系统。
- **设计要点**：
    - **计算单元**：根据 LLM 算子特性设计专用加速核（如 Transformer 计算引擎、注意力专用单元）；
    - **存储层次**：配置 HBM 容量（如 128GB / 芯片）和带宽（如 800GB/s），匹配 LLM 的访存需求；
    - **互联设计**：支持多芯片组网（如片间 NVLink-like 总线、RDMA 网络），满足分布式训练的通信带宽（如单机 8 卡总带宽 2TB/s）；
    - **功耗与散热**：根据软件层的计算强度（如峰值算力 3PFlops）设计电源和散热方案（如液冷），确保稳定运行。

### **总结：各层次的协同关系**

- **自顶向下**：算法层定义模型需求→框架层确定分布式策略→算子层适配计算核心→编译层优化指令与内存→Runtime 层调度资源→驱动层控制硬件→硬件层提供物理支撑。
- **自底向上**：硬件特性（如存储带宽）约束算子设计→编译层需匹配硬件指令集→Runtime 层需利用硬件互联特性→框架层的并行策略需适配硬件拓扑。

以 “千亿参数 LLM 推理” 为例，完整流程为：

1. 算法层确定用 INT4 量化压缩模型；
2. 框架层（vLLM）采用 PagedAttention 优化内存；
3. 算子层开发 INT4 注意力核，适配芯片的量化计算单元；
4. 编译层将算子合并为指令流，优化 HBM 访存；
5. Runtime 层调度多芯片分片执行，通过高速互联传输中间结果；
6. 驱动层控制芯片工作在低功耗模式，匹配推理场景；
7. 硬件层通过 8 卡集群提供足够算力，完成高吞吐推理。

每个层次的设计都需紧密围绕 “芯片特性” 与 “LLM 需求” 的匹配，最终实现模型在硬件上的高效部署。
