---
tags:
  - AI
  - HPC
  - summary
---

# 参考

八股： https://moyutianzun.cn/archives/llm-infra-ba-gu-quan-ji#llm-inference

https://www.zhihu.com/column/c_1903483612977927815

https://zhuanlan.zhihu.com/p/1907536883430437857

https://www.nowcoder.com/feed/main/detail/c77ef218ce5d40d281c1aea6c906de1c

https://zhuanlan.zhihu.com/p/1920946738270810330

https://blog.csdn.net/sinat_37574187/article/details/149797468

https://zhuanlan.zhihu.com/p/672836957

https://zhuanlan.zhihu.com/p/678602674

CUDA 手撕：https://zhuanlan.zhihu.com/p/12661298743

[AccumulateMore/CV: ✔（已完结）最全面的 深度学习 笔记【土堆 Pytorch】【李沐 动手学深度学习】【吴恩达 深度学习】](https://github.com/AccumulateMore/CV)

Paper: [[https://arxiv.org/pdf/1909.08053.pdf](https://arxiv.org/pdf/1909.08053.pdf)]([https://arxiv.org/pdf/1909.08053.pdf](https://arxiv.org/pdf/1909.08053.pdf))

GTC viedeo: [GTC 2020: Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism | NVIDIA Developer]([https://developer.nvidia.com/gtc/2020/video/s21496](https://developer.nvidia.com/gtc/2020/video/s21496))

[isocpp/CppCoreGuidelines: The C++ Core Guidelines are a set of tried-and-true guidelines, rules, and best practices about coding in C++]([https://github.com/isocpp/CppCoreGuidelines](https://github.com/isocpp/CppCoreGuidelines))

[C++ Core Guidelines]([https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines))

[https://developer.nvidia.com/deep-learning-performance-training-inference](https://developer.nvidia.com/deep-learning-performance-training-inference)

genesis framewokr

cppcon

https://ucbrise.github.io/cs294-ai-sys-fa19/

https://zhuanlan.zhihu.com/p/608318764?share_code=6kgRiae9U4sA&utm_psn=1926408052585793087

- Horovod

岗位职责：1. 研发 GPGPU 集合通信库；2. 定位和解决应用中的分布式通信问题；3. 分析优化分布式计算中的单机内/多机间集合通信性能。任职要求：1. 熟悉 C/C++ 编程；2. 熟悉分布式常用的集合通信操作，了解常用的集合通信库，如 OpenMPI、Gloo、NCCL；3. 熟悉网络通信、RDMA 技术，了解 ibverbs 编程接口；4. 熟悉分布式训练框架，如 PyTorch、Horovod；5. 了解 GPU 体系架构和 CUDA 编程者优先；6. 有类 NCCL 通信库开发经验者优先。

https://zhuanlan.zhihu.com/p/608318764?share_code=6kgRiae9U4sA&utm_psn=1926408052585793087

https://www.infoq.cn/article/edwy1v3xy14pgkefdv1u

**

- **岗位匹配度**：★★★★★
- **技能关联**：
    - DeepEP 的核心目标是优化 MoE（Mixture of Experts）模型的通信效率，这与 HPC 中的分布式计算、低延迟通信（如 RDMA、NVLink）密切相关。
    - CUDA/FPGA/MPI 技能是 HPC 领域的核心能力，涉及多节点通信、负载均衡、资源调度等。
- **典型工作内容**：
    - 开发和优化分布式通信库（如 AllReduce、AllToAll）。
    - 设计并行算法，解决大规模计算中的性能瓶颈。
    - 支持企业级 HPC 集群的部署与调优（如超算中心、云计算平台）。
- **适合领域**：超算中心、云计算公司（阿里云、AWS）、半导体厂商（NVIDIA、AMD）。
**

- **岗位匹配度**：★★★★★
- **技能关联**：
    - DeepEP 直接服务于大语言模型（LLM）的专家并行（Expert Parallelism），这是当前 LLM 训练和推理的主流优化方向。
    - 异构编程能力（CUDA/FPGA）可用于加速模型计算，而 MPI 能力可支持多节点分布式训练。
- **典型工作内容**：
    - 优化大模型的训练和推理效率（如 MoE、KV Cache 压缩）。
    - 开发底层框架组件（如自定义算子、分布式通信层）。
    - 与硬件团队协作，设计 AI 芯片的软件栈（如 NVIDIA 的 TensorRT、华为的 CANN）。
- **适合领域**：AI 大厂（Meta、Google、百度）、AI 芯片公司（NVIDIA、寒武纪）、开源社区（PyTorch、TensorFlow）。

---

**

- **岗位匹配度**：★★★★☆
- **技能关联**：
    - DeepEP 的底层实现涉及通信库的开发（如 RDMA、NVLink 协议），这与系统软件（如操作系统、运行时）的设计逻辑高度相关。
    - 异构编程能力（CUDA/FPGA）需要与硬件抽象层（如 PTX、OpenCL）结合，编译器优化经验可加分。
- **典型工作内容**：
    - 开发支持异构计算的编译器或运行时（如 TVM、Halide）。
    - 优化硬件资源调度（如 GPU/FPGA 的内存管理、任务分发）。
    - 参与开源系统软件项目（如 Linux 内核、LLVM）。
- **适合领域**：开源社区、芯片厂商（NVIDIA、Intel）、云计算平台。
**

- **岗位匹配度**：★★★★☆
- **技能关联**：
    - DeepEP 中提到的“SM-free kernels”（无需占用 GPU SM 的通信优化）与 FPGA 的硬件加速理念一致。
    - 你的 FPGA 能力可直接用于设计专用加速器（如 MoE 的路由、通信卸载）。
- **典型工作内容**：
    - 使用 Verilog/VHDL 或高层次综合（HLS）开发 AI 加速器。
    - 优化硬件逻辑以降低功耗和延迟（如 MoE 的路由表压缩）。
    - 与算法团队协作，将 AI 模型部署到 FPGA 上。
- **适合领域**：FPGA 厂商（Xilinx、Intel）、AI 芯片初创公司（如 SambaNova、Graphcore）。

- **岗位匹配度**：★★★☆☆
- **技能关联**：
    - DeepEP 的通信优化能力可支持云平台的大规模模型训练服务（如 Model Parallelism）。
    - 异构编程能力可帮助云厂商设计弹性计算资源（如 GPU/FPGA 实例）。
- **典型工作内容**：
    - 设计云平台的分布式 AI 训练架构（如弹性扩展、资源隔离）。
    - 开发云原生 AI 工具链（如 Kubernetes 调度器、Serverless 推理服务）。
    - 优化云平台的硬件利用率（如 GPU 利用率、网络带宽分配）。
- **适合领域**：云计算巨头（AWS、Azure、阿里云）、AI SaaS 平台。

---

# 技能与掌握水平

## 知识能力矩阵

2,熟悉 [后向误差传播算法](https://zhida.zhihu.com/search?content_id=153124779&content_type=Answer&match_order=1&q=%E5%90%8E%E5%90%91%E8%AF%AF%E5%B7%AE%E4%BC%A0%E6%92%AD%E7%AE%97%E6%B3%95&zhida_source=entity)（BP），完成从标量求导到矩阵求导思维方式的转换，熟悉常见算子的梯度推导（矩阵乘，卷积，池化，Relu，如果会 batch normalization 就一步到位了）；

3，熟悉 [autograd](https://zhida.zhihu.com/search?content_id=153124779&content_type=Answer&match_order=1&q=autograd&zhida_source=entity) 的基本原理，能自己手撸一个最好；

8，熟悉 [编译器基本原理](https://zhida.zhihu.com/search?content_id=153124779&content_type=Answer&match_order=1&q=%E7%BC%96%E8%AF%91%E5%99%A8%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86&zhida_source=entity)，parser 什么的不重要，主要是 [dataflow分析](https://zhida.zhihu.com/search?content_id=153124779&content_type=Answer&match_order=1&q=dataflow%E5%88%86%E6%9E%90&zhida_source=entity)，灵活运用；熟悉多重循环程序优化技巧，譬如 polyhedral 模型；

9，熟悉常见 [分布式系统原理](https://zhida.zhihu.com/search?content_id=153124779&content_type=Answer&match_order=1&q=%E5%88%86%E5%B8%83%E5%BC%8F%E7%B3%BB%E7%BB%9F%E5%8E%9F%E7%90%86&zhida_source=entity)，mapreduce, spark, flink, tensorflow 等；

12，programming language 原理，命令式编程，函数式编程，逻辑编程，入门书《程序的构造与解释》？

13，熟悉项目构建原理，，有一本书《程序员的自我修养》有比较全面覆盖。

| 学科分类      | 各个主题                                               | 主题内各个技术点和知识点                                                                                                     |
| --------- | -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 计算机       | 操作系统                                               | 进程管理、、文件系统、I/O 系统、安全与保护机制                                                                                        |
|           |                                                    | 内存管理                                                                                                             |
|           | 现代 C++                                             | •C++17/20 新特性、智能指针、并发编程、模板元编程、STL 库使用                                                                            |
|           | 计算机基础                                              | compiler, assembler, linker，loader                                                                               |
|           | 软件架构设计                                             | •设计模式、架构风格、代码重构、性能瓶颈分析、性能测试工具                                                                                    |
|           | 编程基础                                               | 命令行、函数式子，C++ python 用法                                                                                           |
|           | 数据结构与算法                                            | •基本数据结构（数组、链表等）、高级数据结构（树、图等）、经典算法（排序、查找等）                                                                        |
|           | linux                                              | •命令行基础、shell 脚本编写、用户权限管理、网络配置与管理                                                                                 |
|           | Qemu 模拟器 gem 5 / gpgpusim 等仿真工具框架                  | •虚拟化原理、QEMU 工作流程、GEM5 仿真模型、GPU 模拟方法                                                                              |
|           | 分析调试工具优化、性能分析和调优                                   | valgrind/callgrind/Vtune，资源利用率分析                                                                                 |
|           | 网络                                                 | socket                                                                                                           |
| 体系结构      | CPU 流水线、缓存、分支预测等微架构原理                              | •流水线技术、多级缓存设计、分支预测策略、超标量处理器                                                                                      |
|           | 量化分析方法                                             | Amdahl' Law, Roofline Model,                                                                                     |
|           | GPU 编程模型                                           | 线程模型，存储模型，优化方法                                                                                                   |
|           |                                                    | 高阶用法，event, stream, 异步/同步                                                                                        |
|           | GPU 指令集 PTX SASS                                   |                                                                                                                  |
|           | Mem                                                | (L0/L1/L1.5/L1.75/L2/HBM, Layout, memory arch)                                                                   |
|           | 可编程芯片（NPU/TPU）架构                                   | •神经网络处理器架构、TPU 工作原理、专用硬件加速技术                                                                                     |
|           | AI 加速器                                             | •AI 加速器类型、应用场景、性能对比、未来发展趋势                                                                                       |
|           | QEMU/GEM5 仿真模拟器                                    | •仿真器在体系结构研究中的应用、搭建仿真环境、案例分析                                                                                      |
| 深度学习框架    | Python                                             | •Python 基础语法、高级特性、科学计算库（NumPy, Pandas 等）                                                                         |
|           | 编译器：LLVM/MLIR/TVM                                  | •编译原理、中间表示、TVM 自动调优、MLIR 模块化设计                                                                                   |
|           | 深度学习模型和框架（如 TensorFlow/Pytorch/Megatron/DeepSpeed） | •主流框架特点、模型训练技巧、分布式训练支持                                                                                           |
|           | 模型表示 ONNX ggml                                     | •模型格式转换                                                                                                          |
|           | 主流模型 NLP/CV 模型架构与算法                                | ，MLP、CNN、LSTM、MHA、MLA、MOE、NLP (Bert), LLM (Transformer), diffusers 各类模型的工作原理、应用场景、实现细节                           |
| 分布式训练 HPC | 集合通信原语和底层原理                                        | （如 AllReduce, AllGather）RDMA 技术优势、应用场景,代价分析                                                                      |
|           | 分布式并行策略及其挑战                                        | 如数据并行、模型并行、流水线并行、张量并行、序列并行、Zero 冗余优化器等                                                                           |
|           | 内核级优化（如算子融合、内存管理优化、通信优化）                           |                                                                                                                  |
|           | 框架 Pytorch 端到端使用和性能分析                              |                                                                                                                  |
|           | 主流框架适配到新型硬件。                                       | （如 PyTorch 生态下的框架）框架适配方法（如 GPGPU/NPU/加速卡）、性能评估指标、优化建议                                                            |
|           | 主流分布式训练/推理框架                                       | （如 DeepSpeed, Megatron-LM, Colossal-AI, FSDP, vLLM, TensorRT-LLM SGLang, Hugging Face Accelerate/Transformers 等） |
|           | 高效的内存管理、通信优化                                       | •内存管理策略、高效通信协议（NvLink/Infiniband/RoCEv2 等）                                                                       |
|           | 分析性能瓶颈和可能的精度问题                                     | （如通信开销、计算效率、内存限制）瓶颈定位方法、解决策略                                                                                     |
|           | RDMA 等高速网络通信技术                                     |                                                                                                                  |
|           | NCCL、NVSHMEM 或其他分布式计算相关                            |                                                                                                                  |
| 算法        | 数值计算、线性代数相关算法有深刻的理解                                | •数值稳定性、矩阵运算、求解线性方程组                                                                                              |
|           | 密集型算子优化                                            | 卷积、矩阵乘、矩阵分解、BatchNorm、flash attention 优化技巧、实现细节、性能比较                                                             |
|           | 大模型核心技术                                            | FlashAttention、PagedAttention、MoE、Chunked Prefill                                                                |
|           | 大模型量化算法及量化算子的实现；                                   | •量化算法原理、实现步骤、效果评估（如 AWQ、GPTQ、SmoothQuant 等）                                                                      |
| 并行优化      | 并行编程基础：有 CUDA/OpenCL/OpenMP                        | •并行编程概念、CUDA/OpenCL/OpenMP 基础                                                                                    |
|           | GPU 高性能算子开发与优化，工具                                  | Nsight Systems compute, DLProf, PyTorch Profiler, TensorBoard                                                    |
|           | 深度学习专用编译器或编译器组件、编译器技术                              | （如 TVM, MLIR, LLVM）Triton、TVM、MLIR 等                                                                             |
|           | 高性能传统工具，                                           | •分布式计算原理、MPI 开发入门                                                                                                |
| ISA       | 包括在 Cmodel, Zebu/PLD 上调试功能                         | -hang, mismatch, assertion), 汇编/反汇编，指令 cycle 计算 Wave/PFC analysis 能力 Jtag 调试 Hang 问题                             |
|           |                                                    |                                                                                                                  |
|           |                                                    |                                                                                                                  |
|           | 软硬件协同编程，性能优化基础                                     | 基于 HW Top(SPEC) 性能评估，分析和建模能力 RR（Rough Roof）                                                                      |
|           |                                                    | 基于 HW Micro Arch 性能评估，分析和建模能力 UB（Up Bound）                                                                       |
|           |                                                    | 基于软件栈 +HW 硬件微架构的性能评估分析和建模能力（Realistic Evaluation）                                                                |
|           |                                                    | 运用 BR 性能分析工具分析算子端到端瓶颈                                                                                            |
|           |                                                    |                                                                                                                  |

- 数字集成电路
1. 互联通信协议
2. 算法硬件实现
3. 电路面积、时序、功耗优化
4. 硬件测试验证流程和工具（vcs, verdi, verilator, etc.）
5. SOC 和 IP 的架构/微架构设计探索 + 性能模型建模，包含不限于核心并行计算处理器、NOC、Cache、MMU、Memory、ESL/RDMA、die-to-die、一致性协议、DMA、etc。
6. 3 D 堆叠，chiplet

---

## 能力维度划分与评分标准（每项满分 10 分）

我们定义五个核心能力维度，每个维度下针对具体技术点进行打分，再加权汇总：

| 维度 | 定义 | 权重建议 |
|------|------|--------|
| **1. 知识广度 (Breadth)** | 对该领域相关技术点的覆盖程度：是否知道关键概念、主流工具、典型架构 | 20% |
| **2. 理解深度 (Depth)** | 是否理解原理、机制、数学推导、性能瓶颈与设计权衡 | 30% |
| **3. 实践能力 (Hands-on)** | 是否有实际项目经验，能否独立实现、调试、调优 | 30% |
| **4. 架构思维 (Architecture Thinking)** | 能否从系统层面思考问题，设计可扩展、高效、健壮的解决方案 | 15% |
| **5. 学习迁移能力 (Learning & Adaptability)** | 面对新技术/硬件能否快速上手，举一反三，解决新问题 | 5% |

> 💡 **权重说明**：对于底层系统/AI 编译器/大模型训练方向，**深度 + 实践**是核心，因此占比最高；架构思维对高级工程师/研究员尤其重要。

评分等级建议（S-A-B-C-D 制）

| 分数范围     | 等级  | 描述                                                 |
| -------- | --- | -------------------------------------------------- |
| 8.0–10.0 | S   | 可主导创新项目，发表顶会论文或开源核心组件，可完全独立负责该领域的工作，并具备指导他人完成工作的能力 |
| 6.0–7.9  | A   | 能独立负责模块设计与优化，具有系统观念，很熟练可以很快上手，也可以和团队合作分工           |
| 4.0–5.9  | B   | 熟悉原理，有动手经历，能在指导下完成开发任务，具备发展潜力                      |
| 2.0–3.9  | C   | 了解基本概念和作用，缺乏实战经验                                   |
| <2.0     | D   | 仅听说或者不知道，未参与工作                                     |

| 技术点 | 知识广度 | 理解深度 | 实践能力 | 架构思维 | 学习迁移 | 加权得分 |
|--------|----------|----------|----------|----------|------------|-----------|
| FlashAttention | 8 | 7 | 6 | 5 | 7 | `= 8×0.2 + 7×0.3 + 6×0.3 + 5×0.15 + 7×0.05 = 6.9` |

- **知识广度 8**：了解其动机（减少显存访问）、基本思想（分块 + 重计算）、支持平台（HuggingFace, vLLM）
- **理解深度 7**：理解 IO 复杂度分析、Tiling 策略、与传统 Attention 的对比
- **实践能力 6**：能调用 FlashAttention 接口，但未手动实现内核
- **架构思维 5**：能说出它如何影响推理吞吐，但未参与整体 KV Cache 设计
- **学习迁移 7**：能类比到 PagedAttention 的设计思路

---

## 总评分数计算方式

```text
总评分 = Σ(各主题平均分 × 主题权重) / 总权重
```

| 学科分类 | 建议权重 |
|---------|----------|
| 分布式训练/HPC | 25% |
| 算子并行优化 | 20% |
| 深度学习与框架 | 20% |
| 体系结构 | 15% |
| 计算机基础 | 10% |
| 算法 | 10% |

> 💬 说明：若岗位偏重“大模型训练系统”，则前两项权重应更高。

场景 1：招聘 AI 编译器工程师

- 重点关注：**TVM/MLIR、CUDA、算子优化、自动调优**
- 提高“理解深度”和“实践能力”权重
- 要求相关主题平均分 ≥ 8.0

场景 2：组建大模型训练团队

- 关键能力：**分布式并行、通信优化、DeepSpeed/Megatron、内存管理**
- “架构思维”权重提升至 20%
- 要求“分布式训练”主题得分 ≥ 8.5

场景 3：评估实习生潜力

- 降低“实践能力”要求，提高“学习迁移能力”权重
- 关注“知识广度”和成长性

# **大语言模型（LLM）全栈知识体系：从算法到硅**

大语言模型的成功部署与高效运行，绝非单一技术之功，而是跨越七个紧密耦合、层层递进的系统层次，实现“软件定义硬件，硬件赋能软件”的极致协同。这一体系从抽象的数学原理出发，最终落地于物理芯片，每一层都为上一层提供支撑，并受下一层能力的约束。

## 第零层：基础

### **计算机体系结构与并行计算**

- **核心理论**：
    - **异构计算原理**：掌握 CPU/GPU/NPU 的架构差异（如英伟达 GPU 的 SM 单元、昇腾 NPU 的达芬奇架构），理解冯・诺依曼架构与哈佛架构的内存访问特性 1。
    - **并行计算模型**：深入理解 Amdahl 定律、Gustafson 定律，掌握多线程（OpenMP）、分布式（MPI）、向量化（SIMD）的协同优化策略。
    - **存储层次**：熟悉多级缓存（L1/L2/L3）、高带宽内存（HBM）、片上存储（SRAM）的性能差异，掌握数据局部性优化策略。
    - **Chiplet 设计**：学习 UCIe 标准与芯粒互连技术，理解分解式计算如何提升芯片能效比（如 Arm CSA 架构的模块化设计）5。
- **实践工具**：
    - 体系结构仿真：GEM5、QEMU（支持 Chiplet 建模）
    - 并行性能分析：Intel VTune、NVIDIA Nsight Compute

### **数学与算法基础**

- **核心内容**：
    - **线性代数**：精通矩阵运算（矩阵乘法优化、低秩分解、Strassen 算法）、特征值分解在模型压缩中的应用。
    - **数值计算**：掌握浮点运算特性（如舍入误差、溢出），理解 FP8/BF16 量化对数值稳定性的影响。掌握量化误差分析（如 KL 散度量化）、定点化技术（如 QAT 量化感知训练）。
    - **算法设计**：熟悉贪心算法（剪枝策略）、动态规划（算子调度）在模型优化中的应用。
- **学习资源**：
    - 书籍：《线性代数及其应用》《数值分析》
    - 工具：MATLAB/Python 的 NumPy 库实践矩阵运算优化
    - 量化分析：PyTorch QAT 工具链、TensorFlow Lite Micro

## **第一层：算法与模型层 (Algorithm & Model Layer)**

- **核心内容**：
    - **模型结构**：掌握 Transformer 架构（自注意力机制、位置编码）、MoE（专家混合模型）、多模态模型（如 Flamingo）的计算特性 815。
    - **训练机制**：理解分布式训练（数据并行 / 模型并行 / 流水并行）、混合精度训练（FP16/BF16/INT8）的原理与实现 15。
    - **推理优化**：熟悉推理引擎（TensorRT/ONNX Runtime）的工作流程，掌握动态 batch、算子融合、内存优化等技术 614。
- **学习资源**：
    - 论文：《Attention Is All You Need》《Scaling Laws for Neural Language Models》
    - 实战：基于 Hugging Face 的 LLaMA-2 微调与推理优化模型压缩：Hugging Face Optimum、TensorRT-LLM
    - 动态推理框架：Hugging Face TGI（支持连续批处理与 PagedAttention）
- **核心任务**：确定 LLM 的基础架构（如 Transformer 变体）、参数量（如 7B/70B / 千亿级）、训练目标（如预训练 / 微调）和推理场景（如对话 / 生成）。
- **设计要点**：
    - 针对芯片计算特性调整模型结构（如若芯片支持稀疏计算，可设计稀疏注意力机制）；
    - 确定量化策略（如 FP16/INT8/FP8），平衡精度与算力利用率（自研芯片可能有专用低精度计算单元）；
    - 优化序列长度（如支持动态上下文窗口），适配芯片的片上存储容量（如 HBM 带宽）。

这是整个体系的起点，定义了“要计算什么”。

*   **核心任务**：
    1.  **模型架构设计**：确定基础结构（如 Transformer 及其变体），选择关键组件（如多头注意力 MHA、MLP、MoE 专家混合模型）。
    2.  **训练目标与策略**：定义预训练/微调的目标函数，选择优化器（AdamW 等），并决定训练范式（如指令微调、强化学习对齐）。
    3.  **推理模式设计**：规划生成方式（自回归、流式输出）、批处理策略（连续批处理）和上下文管理（KV Cache）。
*   **关键技术**：
    *   **位置编码**：RoPE（旋转位置编码）、ALiBi 等，解决长序列建模问题。
    *   **稀疏化技术**：MoE（Mixture of Experts）通过门控机制仅激活部分参数，突破算力墙。
    *   **参数高效微调（PEFT）**：LoRA、Prefix Tuning 等，在冻结主干网络的情况下高效适配下游任务。
*   **关键考量**：模型的设计必须前瞻性地考虑后续各层的实现成本与可行性。例如，选择支持动态长度的架构以适应分页内存管理。

- **核心技能**：
    - **量化技术**：掌握 PTQ（训练后量化）、QAT（量化感知训练）的原理，熟悉 INT8/FP8 混合精度部署。
    - **剪枝策略**：实现结构化剪枝（通道剪枝）与非结构化剪枝（权重稀疏化），结合硬件稀疏计算单元优化。
    - **模型蒸馏**：设计知识蒸馏方案（如教师 - 学生模型），优化端侧推理性能。
- **前沿技术**：
    - **动态推理优化**：使用 Hugging Face TGI 的连续批处理与流式输出，提升长序列生成效率。
    - **自我奖励机制**：学习 LaTRO 框架的隐变量推理优化，通过自我评估提升多步骤推理准确率。
    - 使用 TensorRT 对 BERT 模型进行 INT8 量化，对比精度与速度的平衡点
    - 基于 SepLLM 框架实现长文本推理的稀疏注意力优化

大模型架构设计是 Infra 构建的基础，直接影响计算效率和资源需求。目前主流的架构优化方法主要包括参数建模、位置编码和 MoE 算法等。

**参数建模**是大模型设计的核心，主要解决模型参数量与计算量呈指数级增长的挑战。以 GPT-3 为例，其 1750 亿参数模型在 NVIDIA A100 上训练，单卡需要 32 年，千卡集群优化后仍需 34 天 。参数建模方法主要包括：

1. **低秩适应 (LoRA)**：通过低秩矩阵分解冻结原模型参数，仅训练增量矩阵，显著减少显存占用。例如，LoRA 将 GPT-3 的 1.2TB 显存需求降至 350GB，且支持多任务场景下快速切换不同 LoRA 模块。
2. **动态参数共享**：根据任务和实例条件动态决定参数共享策略，提升多任务学习效率。如 DynaShare 提出分层门控策略，结合任务级和实例级参数选择，实现更灵活的参数共享。
3. **混合专家模型 (MoE)**：通过稀疏门控机制激活部分专家网络，而非全部参数，提高计算效率。如 Switch Transformer 采用单专家路由，使模型参数量达到万亿级别，但仍能保持高效训练。

**位置编码**是解决长文本推理的关键技术，直接影响模型对序列内词汇关系的理解能力。主流位置编码方法包括：

1. **旋转位置编码 (RoPE)**：通过复数旋转矩阵将绝对位置编码转化为相对位置感知，计算高效但外推能力有限。例如，RoPE 在 LLaMA 等模型中广泛应用，支持长文本推理但需要提前知道最大序列长度。
2. **ALiBi 位置编码**：采用可学习线性变换，自适应地融合位置信息，具有更好的外推能力。如 Bloom 模型使用 ALiBi，使模型在处理远超训练序列长度的上下文时表现更稳定。
3. **More 架构**：通过动态路由减少 KV Cache 内存占用，提升推理速度。例如，谷歌 More 架构通过路由机制让模型在自适应推理过程中突破固定思考深度限制，实现参数效率与自适应计算的统一。

**MoE 算法**是解决大模型算力墙的有效手段，其核心是门控机制和负载均衡策略 。主流 MoE 架构包括：

1. **GShard**：采用 Top-2 Gating，通过本地分组和容量约束防止专家过载，但存在 token 丢弃问题。例如，GShard 在训练过程中引入辅助负载均衡损失，鼓励各专家负载更加均衡。
2. **Switch Transformer**：单专家路由 (Top-1)，降低通信开销但需精细调参容量因子。如 Switch Transformer 通过单专家路由使训练速度提升，但仍需依赖辅助损失来保持负载均衡。
3. **GLaM**：回归 Top-2 Gating，增加残差旁路减少 overflow，提升零样本性能。例如，GLaM 采用精心设计的辅助损失函数和容量约束，在保持负载均衡的同时减少 token 丢失。

大模型架构与计算优化方法的选择，需根据具体场景需求（如训练/推理、长文本处理、多任务学习等）进行权衡，形成最适合的模型设计。

## **第二层：框架与编程接口层 (Framework & API Layer)**

这是连接算法与系统的桥梁，定义了“如何描述计算”。

*   **核心任务**：
    1.  **高层 API 封装**：提供用户友好的接口（如 PyTorch, TensorFlow）来构建和操作模型。
    2.  **计算图构建**：将模型代码解析为有向无环图（DAG），明确算子间的依赖关系。
    3.  **分布式抽象**：集成或实现分布式训练/推理框架（如 DeepSpeed, Megatron-LM, Horovod, vLLM），隐藏复杂的并行细节。
*   **关键技术**：
    *   **自动微分（Autograd）**：实现反向传播算法（BP），自动计算梯度，是训练的核心。
    *   **并行策略**：数据并行（DP）、张量并行（TP）、流水线并行（PP）、ZeRO 优化等，用于拆分巨大模型。
    *   **插件系统**：vLLM 等推理引擎采用双轨插件，将硬件差异封装，实现框架与后端解耦。
*   **关键考量**：框架需要具备良好的可扩展性和灵活性，能够对接不同的编译器、运行时和硬件后端。

- **核心任务**：将 LLM 模型代码（如基于 PyTorch/TensorFlow）适配到自研芯片，通过分布式框架实现大规模训练 / 推理。
- **设计要点**：
    - **框架移植**：修改主流框架（如 Megatron-LM、vLLM）的硬件接口，将模型计算映射到芯片的计算核（如替换 CUDA 调用为芯片专用 API）；
    - **并行策略**：设计混合并行方案（数据并行 + 张量并行 + 流水线并行），例如：
        - 用张量并行拆分 Transformer 层的 QKV 计算，适配芯片的多核集群架构；
        - 用流水线并行处理超长序列，避免单芯片内存溢出；
    - **通信适配**：将框架的集合通信（如 AllReduce）绑定到芯片的互联协议（如 PCIe/NVLink 类似的自研链路），优化跨芯片数据传输。
    - 

训练优化与推理加速是大模型 Infra 的核心技术挑战，需要通过算法创新和工具链支持来实现性能提升。

**训练优化技术**主要包括分布式训练框架和算子优化：

1. **分布式训练框架**：
   - **Megatron-LM**：NVIDIA 开发的模型并行框架，侧重多节点预训练，支持 Transformer 架构的高效训练。
   - **DeepSpeed**：微软开发的优化库，通过 3D 并行 (数据/模型/流水线) 和 ZeRO 技术解决显存不足问题，支持单 GPU 训练大模型。
   - **Colossal-AI**：上海交大开发的框架，支持多级并行，动态内存管理提升显存利用率，自适应混合 Adam 优化器在异构训练中更灵活。
   - **Alpa**：基于 Python 的并行计算库，通过自动并行化和零拷贝通信实现跨硬件 (CPU/GPU/TPU) 的高效训练，尤其在 MoE 模型训练中性能显著优于 DeepSpeed。

2. **算子优化**：
   - **算子融合**：如 vLLM 的分页注意力 (PagedAttention) 和 LightLLM 的细粒度 KV Cache 管理，减少内存碎片和通信开销。
   - **量化技术**：如 8 位/4 位量化压缩 KV Cache，降低存储需求。
   - **内存管理**：如 NPUDirect 算法小包通信时延降低 90%，适合 MoE 模型推理。

**推理加速技术**主要包括批处理策略和 KV Cache 优化：

1. **批处理策略**：
   - **Orca 连续批处理**：vLLM 采用的动态批处理策略，通过预测生成长度上界减少 KV Cache 浪费，提升吞吐量。
   - **Prefix Prompt Cache**：SGLang 设计的前缀提示缓存机制，预计算并缓存固定前缀的 KV Cache，减少重复计算。

2. **KV Cache 优化**：
   - **分页注意力 (PagedAttention)**：借鉴计算机分页内存管理，将 KV 缓存映射到不连续的 GPU 内存区域，避免内存碎片问题。
   - **缓存清理策略**：如 LRU（最近最少使用）和 LFU（最少使用频率），根据需求清理部分缓存释放内存。
   - **缓存合并**：将相邻时间步的 KV 进行合并，降低缓存规模。

训练优化与推理加速技术的选择，需考虑硬件特性（如 GPU/TPU/NPU）、模型规模（如百亿/千亿/万亿参数）和应用场景（如训练/推理、长文本处理、多任务学习等），形成最适合的优化方案。

## **第三层：算子与内核层 (Operator & Kernel Layer)**

这是性能优化的基石，定义了“计算的具体实现”。

*   **核心任务**：
    1.  **高性能算子开发**：为模型中的关键运算（如 MatMul, Conv, Softmax, Attention）编写高度优化的底层代码。
    2.  **算子融合（Kernel Fusion）**：将多个小算子合并成一个大的 CUDA/NPU 内核，减少访存次数和启动开销（如 Fused Bias+Add+LayerNorm）。
    3.  **特定硬件适配**：利用专用硬件单元（如 NVIDIA Tensor Core, 华为昇腾 Cube Unit）进行加速。
*   **关键技术**：
    *   **GPU 编程**：精通 CUDA，掌握 `stream`, `event`, 内存层次（Global, Shared, L1/L2 Cache）优化。
    *   **领域专用库**：使用 cuBLAS, cuDNN, CUTLASS, CK, Triton 等库来实现高效的基础运算。
    *   **手写汇编/低级代码**：在极致优化场景下，直接编写汇编或利用 TVM/MLIR 生成最优代码。
*   **关键考量**：此层的性能直接决定了单个节点的算力上限，是“榨干”硬件潜力的关键。

- **核心技能**：
    - **架构设计**：掌握张量计算单元（如 Tensor Core）、片上网络（NoC）、Chiplet 设计的原理，理解华为昇腾、摩尔线程等国产芯片的架构差异。
    - **编程模型**：精通 CUDA（核函数优化、shared memory 使用）、OpenCL，熟悉国产芯片的编程框架（如昇腾 Can、寒武纪 MLU-OPS）。
    - **性能调优**：使用 Nsight Compute、HPCG 等工具分析计算瓶颈，优化访存带宽与计算密度。
- **前沿技术**：
    - **全自动芯片设计**：学习中科院「启蒙」系统的 AI 驱动芯片设计流程，掌握硬件代码自动生成（CodeV 系列）与操作系统内核优化（AutoOS）2。
    - **异构计算优化**：基于 LLVM/MLIR 实现跨芯片算子适配（如 FlagGems 支持 180 + 算子）10。

- **核心技能**：
    - **并行策略**：设计多级并行方案（节点间 MPI + 节点内 OpenMP+SIMD 向量化），解决负载不均衡问题。
    - **通信优化**：掌握 RDMA、NVLink 等高速互联技术，优化 AllReduce、Gather/Scatter 等集体通信操作。
    - **编译器优化**：使用 LLVM/MLIR 进行循环展开、自动向量化，理解算子融合的底层实现。
- **前沿技术**：
    - **云超算标准化**：学习 GB/T 45400-2025 国家标准，掌握弹性高性能计算（E-HPC V2.0）的资源调度与成本优化 11。
    - **量子 - 经典混合计算**：探索量子自动学习（QAL）、张量网络建模在 HPC 中的应用。
    - 在超算集群上优化分子动力学模拟程序（如 NAMD）的并行效率
    - 基于 MLIR 实现矩阵乘法的自动分块与访存优化

- **核心任务**：为 LLM 的关键算子（如注意力、矩阵乘法、激活函数）开发芯片专用实现，最大化硬件利用率。
- **设计要点**：
    - **算子映射**：将 Transformer 的核心计算（如 MatMul、Softmax）拆解为芯片支持的指令集（如张量计算单元 TCU 的专用指令）；
    - **算子优化**：
        - 利用芯片的存储层次（如片上 SRAM 缓存）减少访存延迟（如矩阵分块适配缓存大小）；
        - 算子融合（如 QKV 计算 + 注意力掩码融合），减少中间数据读写；
        - 稀疏计算优化（如跳过零值特征），适配芯片的稀疏加速单元；
    - **性能调优**：通过芯片性能计数器（如计算单元利用率、内存带宽）调整算子实现（如线程块大小、数据布局）。

## **第四层：编译与优化层 (Compiler & Optimization Layer)**

这是智能化的调度中心，定义了“如何高效地执行计算”。

*   **核心任务**：
    1.  **计算图优化**：基于数据流分析（Dataflow Analysis）进行常量折叠、死代码消除、算子重排等。
    2.  **内存规划**：智能分配和复用内存，最小化内存占用和碎片（如 TVM Relay, MLIR）。
    3.  **代码生成**：将高级计算图编译为目标硬件的机器码或中间表示（IR）。
*   **关键技术**：
    *   **深度学习编译器**：TVM, MLIR, XLA 等，实现跨平台、跨硬件的代码生成与优化。
    *   **Polyhedral 模型**：用于循环嵌套的复杂变换与优化（如 tiling, fusion, parallelization）。
    *   **自动调优（Auto-tuning）**：使用搜索算法（如网格搜索、贝叶斯优化）找到最优的算子实现参数。
*   **关键考量**：编译器是实现“一次编写，到处高效运行”的关键，能显著降低在新硬件上移植模型的成本。
- **核心任务**：将 LLM 的计算图（由框架生成）编译为芯片可执行的机器码，完成优化（如指令重排、内存分配）。
- **设计要点**：
    - **计算图优化**：基于芯片架构进行图剪枝、算子合并（如将多层 BN+ReLU 合并为单指令）；
    - **指令生成**：通过编译器（如基于 TVM/MLIR 定制）将算子转换为芯片的微指令流，利用指令级并行（ILP）提升效率；
    - **内存调度**：优化数据在片上 / 片外存储的分配与搬运（如预取策略），避免计算单元空闲；
    - **硬件适配**：针对芯片的特殊功能（如动态电压调节、多精度计算）生成适配指令（如自动切换 FP16/INT8 计算模式）。

## **第五层：运行时与资源管理层 (Runtime & Resource Management Layer)**

这是系统的“操作系统”，定义了“何时何地执行计算”。

*   **核心任务**：
    1.  **任务调度**：协调 CPU、GPU/NPU、内存和通信资源，调度计算任务和数据传输。
    2.  **内存管理**：管理设备内存池，实现高效的内存分配、回收和迁移。
    3.  **通信管理**：提供集合通信原语（Collective Operations）的 API（如 AllReduce, AllGather），并管理底层通信。
*   **关键技术**：
    *   **异步编程**：利用 `stream` 和 `event` 实现计算与通信的重叠（Overlap）。
    4.  **Actor/CSP 模型**：用于构建高并发、低延迟的服务系统。
    5.  **虚拟内存与分页**：借鉴 OS 思想，vLLM 的 PagedAttention 技术将不连续的物理内存块映射给请求，解决 KV Cache 碎片问题。
*   **关键考量**：此层的效率决定了多卡/多机集群的整体吞吐量和响应延迟。

- **核心任务**：管理芯片的计算资源、内存和通信，协调多芯片 / 多节点的协同执行。
- **设计要点**：
    - **任务调度**：将编译后的指令分发到芯片的计算核心，支持多流并行（如计算与数据传输重叠）；
    - **内存管理**：
        - 分配芯片的 HBM/SRAM 资源（如为注意力权重分配高带宽存储）；
        - 实现内存池复用，减少动态分配开销；
    - **通信管理**：封装芯片间的互联接口（如自研高速链路），提供集合通信 API（如 AllReduce、Broadcast），支持分布式训练的梯度同步；
    - **故障处理**：检测芯片错误（如计算超时、内存错误），实现任务重试或故障节点隔离。

## **第六层：驱动与通信层 (Driver & Communication Layer)**

这是连接软件与硬件的“神经末梢”，定义了“如何控制硬件”。

*   **核心任务**：
    1.  **硬件抽象**：提供统一的软件接口（如 CUDA Driver API, HCCL），屏蔽不同 GPU/NPU 的硬件差异。
    2.  **高速通信实现**：实现高效的集合通信库（如 NCCL, HCCL, Gloo），利用高速互连技术。
    3.  **性能监控**：读取硬件计数器，监控温度、功耗、利用率等指标。
*   **关键技术**：
    *   **RDMA 编程**：使用 `ibverbs` 等接口，实现远程直接内存访问，绕过 CPU，降低通信延迟。
    *   **高速互连**：NVLink（芯片间）、InfiniBand/RoCEv2（节点间）提供超高带宽和低延迟。
    *   **国产化通信优化**：华为 NPUDirect 等技术，通过精简同步步骤，将小包通信延迟降低 90%。
*   **关键考量**：通信已成为分布式训练的主要瓶颈，此层的优化对整体性能提升至关重要。

- **核心任务**：作为 Runtime 与硬件的接口，将高层指令转换为芯片的物理操作（如寄存器配置、时钟控制）。
- **设计要点**：
    - **硬件抽象**：封装芯片的底层寄存器、计算单元、存储控制器，提供统一的软件调用接口（如初始化、启动 / 停止计算）；
    - **资源隔离**：控制多进程 / 多任务对芯片资源的访问（如通过 PCIe BAR 空间隔离），避免冲突；
    - **性能监控**：读取芯片的传感器数据（如温度、功耗），反馈给 Runtime 进行动态调频（如高负载时提升核心频率）；
    - **兼容性**：适配 Linux 内核驱动框架（如 PCIe 设备驱动模型），确保芯片可被操作系统识别和管理。

## **第七层：硬件与体系结构层 (Hardware & Architecture Layer)**

这是整个体系的物理基础，定义了“计算发生的场所”。

*   **核心任务**：
    1.  **芯片设计**：设计专用的 AI 加速器（如 NVIDIA GPU, Google TPU, 华为昇腾 NPU），包含计算单元、存储层次和互联网络。
    2.  **系统集成**：构建服务器和超算集群，配置 HBM 内存、PCIe 总线、NVLink 和 InfiniBand 网络。
    3.  **功耗与散热**：设计电源和冷却方案，确保系统稳定运行。
*   **关键技术**：
    *   **计算机体系结构**：理解 CPU/GPU/NPU 的微架构（流水线、缓存、分支预测）。
    *   **Chiplet 设计**：采用 UCIe 等标准，将大型芯片分解为多个芯粒，提升良率和灵活性。
    *   **性能建模**：使用 Amdahl 定律、Roofline 模型量化分析系统瓶颈。
*   **关键考量**：硬件的能力设定了整个系统性能的理论上限，而软硬件协同设计（Co-design）是突破这一上限的唯一途径。

- **设计要点**：
    - **计算单元**：根据 LLM 算子特性设计专用加速核（如 Transformer 计算引擎、注意力专用单元）；
    - **存储层次**：配置 HBM 容量（如 128GB / 芯片）和带宽（如 800GB/s），匹配 LLM 的访存需求；
    - **互联设计**：支持多芯片组网（如片间 NVLink-like 总线、RDMA 网络），满足分布式训练的通信带宽（如单机 8 卡总带宽 2TB/s）；
    - **功耗与散热**：根据软件层的计算强度（如峰值算力 3PFlops）设计电源和散热方案（如液冷），确保稳定运行。

![](image-20250813222055263.png)

## **总结：协同演进的生态系统**

这七个层次并非孤立存在，而是一个**自顶向下需求传导，自底向上能力支撑**的有机整体。

*   **从算法到硬件**：一个新型的稀疏注意力算法（第一层）会推动框架增加新的并行策略（第二层），催生新的融合算子（第三层），要求编译器进行特殊优化（第四层），并可能最终影响下一代 NPU 的架构设计（第七层）。
*   **从硬件到算法**：新一代 GPU 引入 FP8 精度（第七层），促使编译器和算子库支持该格式（第三、四层），进而让研究人员探索在 FP8 下保持模型精度的训练方法（第一层）。

因此，构建面向未来的 AI 人才知识体系，必须打破传统学科的壁垒，培养横跨算法、框架、编译、系统、硬件的**全栈视野**。唯有如此，才能真正驾驭大语言模型这一复杂巨系统，推动 AI 技术的持续创新与落地。

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

### 四、大模型 Infra 的核心挑战与解决方案

大模型 Infra 面临的主要挑战包括算力墙、存储墙、通信瓶颈、软件生态碎片化和成本高昂等问题，需要通过技术创新和工具链支持来解决。

**算力墙挑战**：随着模型参数量的指数级增长，算力需求急剧上升。例如，GPT-3 需要 3640 PF-days 算力，GPT-5 预计参数量达 18 万亿，需 3 万 -5 万张 H100 GPU 训练 200 多天 。解决方案包括：

1. **分布式训练**：通过数据并行、模型并行和流水线并行等技术，将计算任务分配到多个 GPU 上并行执行。
2. **混合专家模型 (MoE)**：通过稀疏门控机制激活部分专家网络，而非全部参数，提高计算效率。
3. **参数高效微调 (PEFT)**：如 LoRA、Prefix Tuning 等技术，仅训练少量参数即可适配下游任务。

**存储墙挑战**：KV Cache 和模型参数占用大量显存/内存，限制模型规模和上下文长度 。解决方案包括：

1. **KV Cache 分页管理**：如 vLLM 的分页注意力和 LightLLM 的细粒度 KV Cache 管理，减少内存碎片。
2. **量化技术**：如 8 位/4 位量化压缩 KV Cache，降低存储需求。
3. **动态参数管理**：如 NPUDirect 算法动态切分物理内存适配虚拟地址，提升内存利用率 20% 以上。

**通信瓶颈挑战**：分布式训练中 GPU 间数据传输延迟高，限制训练速度和扩展性 。解决方案包括：

1. **通信算法优化**：如 NCCL 2.4 双向二叉树在千卡规模下性能优于传统 Ring All-Reduce，Blink 协议在异构 GPU 间吞吐量达 26.4GB/s，优于 NCCL 的 4.8GB/s。
2. **网络硬件升级**：如 InfiniBand 等高速网络降低通信延迟，NVLink 和 PCIe 混合使用提高带宽。
3. **通信与计算重叠**：如 NPUDirect 实现 " 单消息一次同步 " 机制，使小包通信耗时降低 90%，整网通信时延减少 50%。

**软件生态碎片化挑战**：多芯片适配困难，需工具链解耦硬件与框架 。解决方案包括：

1. **中间件抽象**：如昇腾 Can 和寒武纪 Cambricon Tookit 提供统一的编程接口，支持不同框架在不同硬件上运行。
2. **插件系统**：如 vLLM 的双轨插件体系，将硬件差异封装在平台插件内部，实现核心框架与硬件解耦。
3. **开源社区共建**：如昇腾社区由 6000+ 认证开发者组成，推动算子开发和算法优化。

**成本高昂挑战**：硬件采购与维护成本高，如万卡集群训练一次费用超 200 万元 。解决方案包括：

1. **弹性算力调度**：如阿里云 ECS 弹性伸缩按需创建或释放资源，综合算力成本最高可降 55%。
2. **混合实例类型**：如阿里云 SpotMax 方案结合抢占式实例和按量付费实例，平衡成本与可用性。
3. **国产芯片替代**：如华为昇腾 910B 的 FP16 算力达 320TFLOPS，与 NVIDIA A100 接近，推动算力成本下降。

大模型 Infra 的核心挑战与解决方案的选择，需根据具体场景需求（如模型规模、硬件环境、成本预算等）进行权衡，形成最适合的系统架构。

### 五、大模型 Infra 的常用工具链

大模型 Infra 的常用工具链主要包括训练框架、推理框架、数据工程工具和监控管理工具等，形成完整的开发、训练、部署和应用流程。

**训练框架**是大模型开发的基础，主要包括：

1. **Megatron-LM**：NVIDIA 开发的模型并行框架，侧重多节点预训练，支持 Transformer 架构的高效训练。
2. **DeepSpeed**：微软开发的优化库，通过 3D 并行和 ZeRO 技术解决显存不足问题，支持单 GPU 训练大模型。
3. **Colossal-AI**：上海交大开发的框架，支持多级并行，动态内存管理提升显存利用率，自适应混合 Adam 优化器在异构训练中更灵活。
4. **Alpa**：基于 Python 的并行计算库，通过自动并行化和零拷贝通信实现跨硬件高效训练，尤其在 MoE 模型训练中性能优异。

**推理框架**是大模型应用的关键，主要包括：

1. **vLLM**：支持动态批处理 (Orca 策略) 和显存优化，高吞吐推理。
2. **TensorRT-LLM**：NVIDIA 开发的推理加速框架，针对 NVIDIA 硬件优化，提供端到端推理加速。
3. **FasterTransformer**：基于 NVIDIA 的插件化设计，支持 TensorRT 加速，适用于 CPU/GPU 混合部署。
4. **SGLang**：支持 Prefix Prompt Cache 设计，预计算并缓存固定前缀的 KV Cache，减少重复计算。

**数据工程工具**支撑大模型的数据准备和处理，主要包括：

1. **MLFlow**：模型版本管理，跟踪实验参数和结果。
2. **DVC**：数据与模型版本控制，支持分布式数据集管理。
3. **Label Studio**：自动化标注工具，支持大规模数据标注和管理。
4. **Prometheus**：监控工具，实时跟踪模型性能和资源使用情况。

**国产工具链**是大模型 Infra 的重要组成部分，主要包括：

1. **昇腾 Can**：华为开发的 AI 计算架构，深度兼容 PyTorch（通过 TorchAir 扩展库），支持 Ascend C 开发 260+ 算子，且与 vLLM 通过插件系统集成。
2. **寒武纪 Cambricon Tookit**：寒武纪开发的 AI 计算工具包，提供算子库和编程接口，支持模型在寒武纪芯片上运行。
3. **MindSpore**：华为开发的深度学习框架，与昇腾 Can 深度集成，支持大模型训练和推理。
4. **PaddlePaddle**：百度开发的深度学习平台，支持模型在多种硬件上运行，包括百度昆仑芯片。

**云服务工具链**提供弹性算力和分布式训练环境，主要包括：

1. **阿里云 ECS 弹性伸缩**：支持按需创建或释放资源，综合算力成本最高可降 55%，且容器服务 ACS 支持秒级热变配。
2. **华为昇腾 LLMDataDist**：提供分布式 KV Cache 管理，支持与 vLLM、MindSpore 等框架集成，优化跨集群通信。
3. **腾讯云 AI 平台**：提供大规模 GPU 集群和分布式训练框架，支持大模型训练和推理。
4. **百度智能云磐久 AI**：提供 AI 算法预测 GPU 故障，准确率达 92%，稳定连接超过 10 万个 GPU，提升训练性能。

整体要求：掌握 python、docker、推理框架、nsys、ncu 等基础设施或工具的使用，熟悉 LLM Engine 和 Inference pipline，能够进行 perf test、profiling、trace capture/analysis 等，理解 LLM Inference Perf 各项 metrics，对 LLM Inference 有一定程度的了解，熟悉上述模型架构，能够顺利完成 Inference Benchmark。

TensorRT-llm 源码本地编译，跑通了 llama3-70B，Qwen2-57B，DS-R1 FP8 三个模型，可以测试离线推理指标。

sglang 源码本地编译，跑通了 llama3-70B，Qwen2-57B 这两个模型，可以测试线上推理指标。

阅读 FlashAttention 论文，了解了 GPU 结构。

在 TensorRT-llm 框架上完成 Deepseek-R1-0528（4 layers）的线上推理 benchmar

k。

sglang 和 vllm 框架也基本完成 Deepseek-R1-0528（4 layers）线上推理。

阅读了 TensorRT-llm 的 benchmark_serving 源码和 modeling_deepseekv3 权重加载部分源码。

得到 TensorRT-llm 框架上 Deepseek-R1-0528（2 layers）的 online benchmark 性能指标。

阅读 Deepseek 原论文，学习掌握了 Deepseek 结构和整个推理执行流程。

学习阅读 TensorRT 部分源码，学习在 nsys 上抓取模型推理 trace，认识各种算子。

完成了 TensorRT-llm 框架上 Deepseek-R1-0528（2 layers）prefill 和 decode 阶段各个算子 trace 的抓取。

学习了 chunked prefill 原理，阅读了 PagedAttention 论文。

继续完善 TensorRT-llm 框架上 Deepseek-R1-0528（2 layers）prefill 和 decode 阶段各个算子 trace 的抓取。

继续学习 cuda 编程，实现了基本的向量加、sgemm、softmax 以及 gelu 算子。

开始了解学习 Qwen2.5 VL 模型架构。

学习 Rmsnorm AddRmsnorm 算子。

Communication-Efficient Distributed Deep

Learning: A Comprehensive SurveyTwo-Tree Algorithms for Full Bandwidth

Broadcast, Reduction and Scan 1

Peter Sanders and Jochen Speck

Universit¨at Karlsruhe, D-76128 Karlsruhe, GermanyCollective communication:

theory, practice, and

experienceExhaustive Study of Hierarchical AllReduce

Patterns for Large Messages Between GPUs

Yuichiro UenoMGPUSim: Enabling Multi-GPU Performance

Modeling and Optimization

Yifan Sun1 Trinayan Baruah1 Saiful A. Mojumder2 Shi Dong1 Xiang Gong1 Shane Treadway1

Yuhui Bao1 Spencer Hance1 Carter McCardwell1 Vincent Zhao1 Harrison Barclay1

Amir Kavyan Ziabari3 Zhongliang Chen3 Rafael Ubal1 José L. Abellán4 John Kim5 Ajay Joshi2

David Kaeli1

(yifansun, tbaruah, shidong, xgong, streadwa, ybao, shance, cmccaLogGPO: An accurate communication model for

performance prediction of MPI programsPerformance Analysis of MPI Collective

Operations ?

Jelena Pjeˇsivac-Grbovi´c1, Thara Angskun1, George Bosilca1,

Graham E. Fagg1, Edgar Gabriel2, and Jack J. Dongarra13Optimization of Collective Communication Operations in MPICHSynthesizing Optimal Collective Algorithms

Zixian Cai∗Optimized Large-Message Broadcast for Deep Learning Workloads:

MPI, MPI+NCCL, or NCCL2?

Copy

Compute Express LinkTM (CXLTM)Optimizing Communication for Clusters of GPUs

by

Michael Wayne LeBeaneGPU Triggered Networking for Intra-Kernel Communications

Michael LeBeaneNVIDIA Magnum IO GPUDirect Storage

Overview GuideNVIDIA Magnum IO GPUDirect Storage

cuFile API

API Reference-a----         2021/3/13     17:08        4423050 106_-AsynchronousPeer_LRomanovsky.pdf

-a----          2021/3/4     14:48         181690 2013-hips-holk-rust.pdf

-a----         2021/2/25     16:31         280023 An_evaluation_of_Open_MPIs_matching_transport_lay.pdf

-a----          2021/3/9      9:49       20891683 ATPESC_2019_Track-2_1_7-30_830am_Guo-Raffenetti-Thakur-MPI_for_Scalab

                                                  le_Computing.pdf

-a----         2021/3/12     19:53        1781524 BR100-MultiGPU_Architecture Design Document v0.1.docx

-a----        2020/10/21     11:19        4014209 bureddy-mug-18.pdf

-a----         2023/4/26     14:17        1671353 bureddy-mug-20.pdf

-a----         2021/3/24     10:34           3097 debug.log

-a----        2022/10/13     14:02        1827092 dgx-1-multinode-scaling-whitepaper.pdf

-a----         2021/1/14      9:01         283995 dotguide.pdf

-a----        2020/10/21     11:19        1332801 Efficient_Large_Message_Broadcast_using_NCCL_and_C.pdf

-a----         2021/3/26     18:00         917943 gem5 simulator.pdf

-a----         2021/5/24     15:15         386498 GPGPUTransparentVirtualizatio.pdf

-a----         2021/5/16     18:57         513044 GPU PaaS computation model in Aneka computing environments.pdf

-a----        2020/10/21     11:19        1562581 GPUDirect_RDMA.pdf

-a----         2021/3/18     14:15        3497521 GPU_Direct_IO_with_HDF5-_John_Ravi.pdf

-a----         2021/2/27     11:39        1689690 Highly Scalable Deep Learning Training System with Mixed-Precision Tr

                                                  aining ImageNet in Four Minutes.pdf

-a----         2021/2/27     11:40         536443 ImageNet ResNet-50 Training in 224 Seconds.pdf

-a----         2021/2/22     14:02         321429 Intel_PSM2_PG_H76473_v1_0.pdf

-a----         2023/4/13     21:39        6106074 introduction of nvswitch+pcie+nvlink.pdf

-a----          2021/3/1     20:51         319045 IntroToRUST.pdf

-a----        2020/10/21     11:19        2429374 In_network_computing.pdf

-a----          2021/3/2     17:22         419580 Massively Distributed SGD.pdf

-a----         2021/3/18     11:22        1092402 Michael_dissertation_final.pdf

-a----          2021/5/8     15:05         691789 MPI_virtual topology.pdf

-a----        2022/12/26      9:58        4586302 Multi Node Multi GPU Programming.pdf

-a----         2021/2/22     16:57        1403632 NCCL-Woolley.pdf

-a----        2020/10/21     11:19         719011 NCCL2.0.pdf

-a----         2021/2/20     15:15       13173508 nodecart-final.pdf

-a----          2023/2/9     16:37        7872487 nvidia-h100-tensor-core-hopper-whitepaper.pdf

-a----         2023/4/17     16:04        4321830 NVSwitch HotChips 2022 r5.pdf

-a----          2021/4/9      9:46         775632 ofv_presentation_GPU.pdf

-a----          2021/5/8     15:04         457355 ompi cartesion topology.pdf

-a----          2021/3/1     20:57        1482186 ompi one side communication.pdf

-a----         2021/2/23      9:53         179963 ompi topology.pdf

-a----          2021/5/8      9:56         938930 ompi tree match.pdf

-a----         2021/2/26     16:52        2530594 OpenMPI Arch Design (002).pptx

-a----         2021/2/20     15:41         609589 openmpi.pdf

-a----         2021/3/25     16:53        1178136 OpenSHMEM-1.5.pdf

-a----        2020/11/11     10:13         270598 openucx-readthedocs-io-en-master.pdf

-a----         2021/3/28      0:04         372067 Patent_ 基于地址扩展的多节点 GPU All Reduce 性能优化方案.docx

-a----         2021/2/18     11:29        7231839 PMIx-BoF.pdf

-a----          2021/4/7     15:23        3814164 rdma_guide_mlnx-lnvgy_dd_nic_cx.ib-5.1-0.6.6.0-0_sles15_x86-64.pdf

-a----        2020/10/29     15:28        4340733 rocmdocs-amd-com-en-latest.pdf

-a----          2021/3/6     12:00        4013398 S4236-multi-gpu-programming-mpi.pdf

-a----         2021/2/23     10:50        5139769 Saxena_G_Computing_PhD_2018.pdf

-a----          2021/4/8     10:44        3152832 Scaling-DL-on-Summit.pdf

-a----          2021/2/5     20:06         956426 StreamsAndConcurrencyWebinar.pdf

-a----        2020/10/21     11:19         849441 Summit-NCCL.pdf

-a----        2020/12/29     17:47         933885 tree based graph search by Amazon.pdf

-a----         2023/5/10     20:24       12427444 trip-report-cgo2023.pdf

-a----        2020/11/23     17:42        9111226 UCX_HOTI_Tutorial-Final_v1.pdf

-a----         2021/2/27     14:40        1470253 Volltext (pdf).pdf

-a----         2021/2/20     13:42        1539076 wed_01_pt2pt.pdf

0、背景

        在日常使用大模型时，试想以下几种情况下大模型推理系统能否应对：

你的问题比较复杂 prompt 较长，且一次问答没有完全解答清楚，你开始不断的进行连续对话补充细节修正讨论内容。推理服务能否正常工作？大模型是否保持较好的回答质量没有丢失过往问答中的关键信息？TTFT？

输入一部《三体》(单部约 20~36 万字，tokenize 后约 180K tokens)，大模型能否接受这种规模的输入？能否正确回答相关提问？TTFT？一次输入三部《三体》(约 90 万字，tokenize 后约 600K tokens) 后又会怎样？

实际上作为参考，一个 8B LLM 在单个 A100 GPU 上 300K token prefilll 未经针对长文本的优化时，TTFT 需要 6 分钟， 1M 输入则需要 30 分钟，attn 占绝对大头 。且如果模型没有经过长文本训练，回答质量会暴跌。

        上述这两种情形是典型的超长文本推理prefill优化，其输入规模可接近或超过"M"级，由于大模型推理计算中，只有attention是存在token间依赖的，其余router、ffn、layernorm、permute、unpermute等均是token-wise的，因此，对于一个既定模型面对这种“M”规模的超长文本推理优化时，推理性能上在compute和memory上的挑战各不一样：

compute workload 方面，prefill 阶段，O(L^2) 级的标准 attention 计算复杂度增长，O(L) 级的其余 op 计算复杂度增长 (主要是 Moe/FFN)，decode 阶段不受影响；

memory workload 方面，prefill 阶段 KV-cache、activate 的 mem consumption & access bytes 线性增长，decode 阶段则只有 KV-cache；

显然亟待优化的主要是以下几项，对应措施也显而易见：

prefill attention 计算，采用 sparse attention 由平方增长降到 O(k * L) 线性增长， algorithm level 降低 compute workload；

更激进的量化，压缩 KV-cache 和 activate，降低 device mem consumption & access；

优化 chunk prefill 组织调度，降低 pipline bubble；

算子硬件效率的进一步优化；

前两项优化都是数学上不等价的有损近似优化，平衡不好就会使模型掉点无法接受，而这次阿里云的超长文本推理优化在稀疏注意力、kv cache 低 bit 量化、prefill chunk 调度等方面综合处理的比较好，总结来看较好地解决了以下两方面的诸多工程问题：

GTC proposals	效果

推理服务稳定性	能否在超长文本输入时正常工作

超长文本时 kv-cache 也线性增长，是否有压缩、offload 等机制避免 OOM；

各 op 是否支持，框架是否支持 chunk prefill；

Streaming LLM 后已基本解决	run through

回答质量	能否在超长文本输入时正确回答

早期大部分模型都没有经过超长文本训练 (position_ids 过大)；

模型架构设计时没有考虑超长文本输入 (如位置编码长度外推问题)；

DCA(GTC 未提及，但也是相关 feature 之一)	train-free 正常推理

推理性能	prefill attention op latency

分钟级 latency

O(L^2) 级的标准 attention 计算复杂度增长

sparse attention， FP5 kv-cache

attn 最大 27x 加速

O(k * L) 线性增长

chunk prefill	prefill 时不同 chunk 的 workload 不一致导致 latency 差异凸显，使得 PP 时出现 bubble	Dynamic Chunked Prefill	bubble time 减少 80%

1、Attention 算子优化

1.1 Sparse attention

        不考虑ring-attention等多卡attention方案，单卡上最有效的注意力workload优化是稀疏注意力，常见的如slide window attention，这种缩小上下文scope的sparse pattern很好理解也很有效，通过将token间的上下文依赖限制在一定窗口范围内，可以显著降低计算复杂度，且对模型性能影响有限。但针对超长文本场景这种单一固定pattern的稀疏方式局限性较大，常规的window size代价太大，较大的window size则效果有限。

        作者引入了微软提出的MInference——一种针对超长文本推理的Dynamic Sparse Attention，MInference针对每一个Attention Head的特点，通过识别三种 sparse pattern 模式（A-shape, Vertical-Slash, Block-Sparse），动态构建稀疏索引，并使用作者手撸的CUDA/Triton kernel 来显著减少prefill latency。其稀疏pattern、方式如下图、表所示。

稀疏对象	qk，即 attn weight，(head_num，seql, seql)

稀疏粒度	per head，即 (seql, seql)

block size	1x64(vertical-slash)，64x64(A shape，block-sparse)

如何确定每个 head 用哪种稀疏方式

稀疏方式选择：每个 head 分别用三种 pattern 进行稀疏，基于系数结果求 socre 并取最小者，这里的 score 综合考量化了计算量、匹配误差、稀疏率、稀疏分布等多方面因素。

具体稀疏过程

总的来说，稀疏化都是设计算法以 block 为粒度计算 qk 相似度或类似的指标，取 top-k 或者用阈值等方式截断，这里作者设计了多种机制：

A-shape：这种稀疏 pattern 识别是最简单的，和 StreamingLLM 一致，即句首 +slidewindow，与输入无关。

Vertical-Slash：A-shape 的一种泛化，由于垂直线和斜线的连续性，作者基于最后一个 Query 向量和 Key 向量估计稀疏注意力索引矩阵，然后使用这些稀疏索引进行块稀疏计算。作者估计的复杂度代价是 O(n)。

Block-Sparse：引入了一个均值池化来进行聚类，以获得 稀疏后的和 。

具体执行稀疏计算操作时则是通过如下方式，基于块索引矩阵进行稀疏计算，跳过其余区域，从而达到一个加速效果。

稀疏效果

        注意力稀疏是一种有损优化，因此在讨论加速效果前，我们首先应该验证稀疏后能够保持了原版注意力的多少信息，作者用attention recall这一指标来做了量化，如下图所示，可以看到当正确选择了稀疏模式时，recall能够较好保持，但错误的系数模式将导致急剧恶化，这也从侧面证明每个head独立选择系数模式的必要性。

性能方面，如论文所述，总体来说，在一百万的超长 prompt 预处理上，相比 FlashAttention-2 快了十倍

1.2 Dual chunk attention
纯算法层面的设计，仅针对超长文本的生成质量，对性能优化没有帮助，甚至有一些额外开销，因此简单介绍，位置索引的计算公式等细节见论文，本篇不再赘述。

        除了超长文本引起的attention workload暴增，基于旋转位置编码的大型语言模型在输入token数量超过其预训练长度时，生成质量也会断崖式下降。这主要是由于在计算attention weight时，Query和Key的相对位置距离过大，在训练过程中未曾见过。为了解决这一问题，阿里引入了Dual chunk attention (DCA)，该方法通过将过大的相对位置，重新映射为较小的值，从而train-free解决了这一问题，因此该GTC的工作中也引入了DCA，其主要设计有两点：

1.重用原始位置索引以及其嵌入向量，但是重新设计了相对位置矩阵的构造，使其不超过训练长度。

2.将完整 sequence 分切分为 chun，设置每个 chunk 都小于训练窗口长度，并且设计 intra-chunk attention、inter-chunk attention、successive-chunk attention 三种位置索引构造方式如下图，分别负责 chunk 内注意力、chunk 间注意力、连续 chunk 注意力的提取。
Intra-Chunk Attention（块内注意力）：每个块内部计执行标准 attention，捕获局部依赖。

Inter-Chunk Attention（块间注意力）：块与块之间通过稀疏注意力（如滑动窗口）交互，捕获全局信息。

successive-Chunk Attention（连续块注意力）：特殊的 Inter-Chunk Attention，

        qwen系列模型的实验来看，不论模型预训练阶段是否有过超长文本语料的训练，当面对其从未见过长度的的超长输入时，DCA能够帮助模型保持准确率不掉点，且latency和显存开销基本持平常规flash-attn。

1.3 FP5 KV-cache
        decode 阶段则需要读取整个序列的 KV-cache，超长文本输入使得这一访存量&显存开销是十分惊人的。更激进的低 bit 量化是有效手段，但模型数据集验证性能角度来看通常 kv cache 量化相比于 weight 更敏感，因此 kv-cache 的并不是量化的第一选择，FP8/Int8 是目前保证模型数据集性能的最激进方案，FP4/Int4 则掉点明显难以接受，作者因此折衷设计了 FP5 来压缩 KV-cache，并针对这种硬件并不原生支持的类型设计了高效的 pack、unpack、dequant 方式，以尽可能缩小由此带来的 overhead，保证整体性能收益。

pack：store efficiency

        GPRs上基于32bit变量操作时，将6个FP5 pack到一个32bit中，以保证符号位与值位是1：1均匀的，减少不必要的读写，但每32bit最后第30、31两个bit会被浪费；FP5的浮点格式没有透露，可能为S1E3M1，相比于FP4增加1bit来扩展值域，减小量化带来的精度损失。将kv-cache存出到HBM时，bit的浪费不可接受，因此是将单bit的符号位pack到一起，另4bit的值pack到一起，分别以32bit存出。

unpack：load efficiency

load 时 (signed 32bit，value 32bit) →FP5→dequant 的过程，但 slide 里这部分只有这个图，没看懂表达的内容，也没开源代码提 issue 问，暂 bypass。

convert：dequant efficiency

FP5 的反量化需要将 32bit 以 4bit 的粒度进行位运算，作者介绍了这一过程中的两个指令优化例子，如下图所示。

首先是将取 32bit 某 4bit 需要的两个位运算替换为结果等价的一条 dp4a 指令，位运算 latency 通常 1cycle，指令带宽也极大，dp4a latency 没有公开数据，一些论文的测试结果来看在 4cycle 左右，两条位运算替换为一条 dp4a 收益存疑；

调整计算次序合并编译期常量计算，减少运行时的计算，这一优化消除了一次整数乘法 (实质上是左移位运算)，但我怀疑这种编译期常量的 instruction fusion 非常简单，编译器的优化 pass 能够处理，应无需开发者来做；

作者宣称这里减少了一半的指令数，但基于此的性能收益没有说明，个人感觉应该很有限。

结合了以上 sparse attention、DCA、FP5 kv-cache 等手段后，面对 1M 输入，attention 也能够正常工作且保证回答质量，prefill latency 相比于 flash-attn 达到了 27.8x 的加速。

2、Moe FFN 算子 decode 优化

这里 Moe FFN decode 的优化其实和长文本没有联系，且只有两页 slide，没有开源细节，存在个人猜测成分。

作者认为的痛点问题

decode 阶段 bs 一般较小， FC1 activate(bs， 7168) 远小于 weight(7168， 2048+2048)，FC2 activate(bs， 2048) 同样远小于 weight(2048， 7168)，因此 load weight 是瓶颈。

业界现有 FusedMoe(vllm triton impl) 带宽利用率低，作者称只有 50%；

decode 阶段 M 较小，以 FC2 为例，常规 slice-k 只能切 N = 7168(如 TP 则 N 更小) 不合适，通常会采用 split-k 的方式去切 N = 7168 和 K = 2048，作者认为这个 K 太小，切 split-k 不方便；

均含 epilogue，FC1 有 swiglu+dequant，FC2 有 dequant，开销不容忽视；

针对优化

warp specialization & persistent & pingpong：SM90 常规手段，不再赘述

large tile with TMA

更大、更少的内存 instrs；

更好的数据局部性，cache/TLB 减少冲突提升 hit；

更少的 TMA sync；

参数模板化，参数空间 (tile/TMA shape，multi-stage num，is_bypass_L2，block_swizzle, ……etc)，针对当前模型几组典型 shape 区间，离线或在推理服务 warm up 阶段进行 kernel tuning 搜索参数空间得到最佳参数组合，执行时根据 shape 调度到最佳的 kernel

和 vllm FusedMoe triton Impl 相比，提升如下图。

3、框架层优化

        超长文本输入时，主流推理框架均基于chunk prefill进行处理。然而chunk大小通常是固定的，这导致处理不同的chunk时workload极度不均衡，latency因此出现差异。例如，第一个chunk是纯prefill，无past KV-cache，计算密集型，但随着chunk后移past sequence length增大，需要读取的past KV-cache越来越多，计算workload变化不大但访存wokload持续增长，逐渐向访存密集型过渡。框架在PP并行下调度处理这些chunk时因此在pipline上产生很多bubble。stage越多chunk距离越大latency差异就越大，stage3 chunk0应该和stage0 chunk3、stage1 chunk2、stage1 chunk1进行overlap，但chunk3显著慢于chunk0，这一prefill latency差异导致了bubble time。

        BladeLLM针对长文本prefill设计了Dynamic Chunked Pipline Parallelism(DCPP)机制，即采用不一致的chunk size，使得PP在不同stage调度处理不同chunk时latency更加一致，从而减少pipline bubble。

如下表，作者展示了 DCPP 在 64K、128K 下的 prefill 加速效果，可以看到 bubble time 得到大幅度的压缩，使得 DCPP 相比于其他并行方式 prefill latency 最佳。

4、总结

        综上，GTC阿里云团队主要分享了三方面的工作：

MIInference、DCA 等 attention 优化在稀疏化、长文本扩展、位置编码外推等方面的工作，使得模型能够无需额外训练就能够应对长文本输入，并显著降低 attention 的计算 workload，结合这些工作，阿里云团队基于 bladellm 推理引擎开发了 FP5 kv-cache 量化进一步压缩了显存开销和访存 workload，使得整个 attention 面对长文本输入时大幅加速；

针对 Moe FFN decode load B matrix 的访存瓶颈，将带宽效率优化至 90+%，优于 vllm；

针对长文本 PP 时的 bubble，框架层引入了一种动态 chunk shape 的 DCPP chunk prefill，以平衡 chunk 间的 prefill latency，使得 PP pipline 消除了 80+% 的 bubble time，优于其他并行策略；

以上工作中，1 已经应用到 qwen2.5、qwen3 则有自己的实现 (见 qwen 技术报告)，开源社区支持慢一些，也已经有 vllm PR 了；2 仍然在闭源 blade-nn 算子库中；3 也在闭源 blade-llm 推理框架中；

5、参考文献

GTC：slide

Dual chunked attention：paper | code | vllm PR

Dynamic sparse attention：paper | code | vllm PR

阿里云 BladeLLM 推理引擎简介

vllm FusedMoe Impl(Triton)

qwen 团队技术博客

Mode                 LastWriteTime         Length Name

----                 -------------         ------ ----

d-----         2025/8/22     13:53                DGX arch

d-----         2025/8/22     13:53                GTC

d-----         2025/8/22     13:53                GTC2025

d-----         2025/8/22     13:55                HC2024

d-----         2025/8/22     13:56                NVIDIA DPU 资料包

d-----         2025/8/22     13:57                NVSwitch

-a----         2024/6/17     17:20         315920 amd-ai-networking-direction-and-strategy.pdf

-a----         2023/4/26     14:17        1671353 bureddy-mug-20.pdf

-a----         2023/4/27     17:46        3859709 CUDA_C_Programming_Guide.pdf

-a----         2023/4/27     17:47        4368780 CUDA_Driver_API.pdf

-a----         2021/9/20     13:23        1651799 CUDA_Multi_Process_Service_Overview.pdf

-a----          2025/1/3     10:41         498349 gaudi-3-ai-accelerator-32-node-cluster-reference-design-white-paper.p

                                                  df

-a----          2024/6/3     16:44        2364166 GTC_2024_S61368_MGPU.pdf

-a----         2024/8/30     20:54      151469948 HC2024.zip

-a----         2023/4/13     21:39        6106074 introduction of nvswitch+pcie+nvlink.pdf

-a----        2022/12/26      9:58        4586302 Multi Node Multi GPU Programming.pdf

-a----         2021/7/19      8:58       28907063 NVIDIA DPU 资料包.zip

-a----         2024/4/23     16:25         438508 nvidia sharp.pdf

-a----         2021/9/20     15:02        7979890 nvidia-ampere-architecture-whitepaper.pdf

-a----          2023/2/9     16:37        7872487 nvidia-h100-tensor-core-hopper-whitepaper.pdf

-a----         2024/5/23     10:18       29309021 NVLink C2C 2.pdf

-a----         2024/5/22     20:19       14092916 NVLink C2C.pdf

-a----         2024/5/28     17:10        1587985 nvswitch fabric-manager-user-guide.pdf

-a----         2023/4/17     16:04        4321830 NVSwitch HotChips 2022 r5.pdf

-a----         2024/8/13     15:10         491529 nvswitch-technical-overview.pdf

-a----          2023/8/8     11:53        6460143 ptx_isa_8.2.pdf

-a----         2024/7/17     19:01         438508 sharp.pdf

-a----         2022/5/24     10:15         849441 Summit-NCCL.pdf

-a----         2024/8/16     15:10        3933820 Turing white paper.pdf

    目录: O:\软件生态部\Base SW\Multi GPU\SCCL1.0 & 2.0 shared\nvidia 资料\DGX arch

Mode                 LastWriteTime         Length Name

----                 -------------         ------ ----

-a----         2023/10/5     10:19        4466331 dgx1-v100-system-architecture-whitepaper.pdf
-a----         2023/10/5     10:19        2738857 dgxa100-system-architecture-white-paper.pdf
-a----         2023/10/8     15:21        7979890 nvidia-ampere-architecture-whitepaper.pdf
-a----         2023/10/8     14:36        2393089 nvidia-dgx-superpod-a100.pdf
-a----         2023/10/5     10:24        7880278 nvidia-h100-tensor-core-hopper-whitepaper.pdf
-a----         2024/1/18     17:50         198997 nvswitch2.png
-a----         2024/1/18     17:51         227420 nvswitch3.PNG

    目录: O:\软件生态部\Base SW\Multi GPU\SCCL1.0 & 2.0 shared\nvidia 资料\GTC

Mode                 LastWriteTime         Length Name

----                 -------------         ------ ----

-a----         2023/6/27     10:46         657258 CWES52010_Inter-GPU Communication Techniques and Libraries.pdf
-a----         2023/6/27     10:47        1191328 S31880-INTER-GPU COMMUNICATION FOR LARGE-SCALE TRAINING.pdf
-a----         2023/6/27     10:47        1340764 S51111 - Scaling Deep Learning Training Fast Inter-GPU Communication
                                                  with NCCL_1679417502845001oKEq.pdf
-a----         2023/6/27     10:47        2946436 S51142 - Accelerating Data Movement Between GPUs and Storage or Memor
                                                  y_1679457043806001GCFN.pdf
-a----         2023/6/27     10:47        8079792 S51421_Investigating Hidden Bottlenecks for Multi-Node Workloads.pdf
-a----         2023/6/27     10:47        6493749 S51705_How to Streamline Shared Memory Space with the NVSHMEM.pdf
-a----         2023/6/27     10:47        2935838 S51892 - High-Performance and Scalable Data Science using MPI4Dask_16
                                                  79377846749001dtLC.pdf
-a----         2023/6/27     10:48        1774764 S51971 - Large-Scale Deep Learning using Hybrid_Multi-Cloud MLOps, GP
                                                  Us, OpenMPI, and DeepSpeed.pdf
-a----         2023/6/27     10:48        5288675 S52202 - Accelerate Spark With RAPIDS For Cost Savings_16793690033700
                                                  01fLYx.pdf
-a----         2023/6/27     10:48         463917 s8462-multi-gpu-training-with-nccl.pdf
-a----         2023/6/27     10:48        1649326 s91043-rapids-cuda-dataframe-internals-for-c++-developers.pdf

    目录: O:\软件生态部\Base SW\Multi GPU\SCCL1.0 & 2.0 shared\nvidia 资料\GTC2025

Mode                 LastWriteTime         Length Name

----                 -------------         ------ ----

-a----         2025/4/28     10:22        3141370 S72576.pdf
-a----         2025/4/28     10:22        2999806 S72579.pdf
-a----         2025/4/12     15:35        4360874 S72683_CUDA Techniques to Maximize Memory Bandwidth and Hide Latency.
                                                  pdf
-a----         2025/4/12     15:35        8575289 S72685_CUDA Techniques to Maximize Compute And Througput.pdf

    目录: O:\软件生态部\Base SW\Multi GPU\SCCL1.0 & 2.0 shared\nvidia 资料\HC2024

Mode                 LastWriteTime         Length Name

----                 -------------         ------ ----

-a----         2024/8/27     12:35        5007046 04_HC2024.IBM.CBerry.final.pdf
-a----         2024/8/27     12:35         269364 1-HC2024.ucsd.BryanChin.v02.pdf
-a----         2024/8/27     12:35        2304429 11_HC2024.SKhynix.GuhyunKim.rev920240822.pdf
-a----         2024/8/27     12:35        1380254 14_HC2024.Intel.Xeon_6_SoC.Praveen.Mosur.pdf
-a----         2024/8/27     12:35         768108 16_HC2024.Poster.UCLA.BlaiseTine.v01.pdf
-a----         2024/8/27     12:35        1853916 2-HC2024.nvidia.MarkRen.Intro.v04.pdf
-a----         2024/8/27     12:35         974075 22_HC2024.Poster.KAIST.Ryu.v3.pdf
-a----         2024/8/27     12:35        1884975 23_HC2024.AMD.MI300X.ASmith(MI300X).v1.Final.20240817.pdf
-a----         2024/8/27     12:35        3213795 25_HC2024.Qualcomm.GWilliams.pdf
-a----         2024/8/27     12:35         705136 28_HC2024.Poster.KAIST.Park.v2.pdf
-a----         2024/8/27     12:35        2443785 3-HC24.synopsys.SteliosDiamantidis.v03.pdf
-a----         2024/8/27     12:35         363351 35_HC2024.Poster.KAIST.SangyeobKim.v01.pdf
-a----         2024/8/27     12:35       25279571 3_HC2024.Poster.KAIST.SungyeobYoo.v02 merged.pdf
-a----         2024/8/27     12:35        1678815 4-HC24.PrimisAI.Hans_Bouwmeester.v4.pdf
-a----         2024/8/27     12:35        1051246 42.HC2024.Poster.NationalChungChengUniversityTaiwan.JinnShyanWang.v02
                                                  .pdf
-a----         2024/8/27     12:35        8646312 45_HC2024.PosterSlides.UNIST.JueunJung.v02.pdf
-a----         2024/8/27     12:35        1295227 48_HC2024.Sambanova.Prabhakar.final-withoutvideo.pdf
-a----         2024/8/27     12:35        1920658 49_HC2024.Poster.UEC.BinhKieuDoNguyen.v2.pdf
-a----         2024/8/27     12:35        3187256 5-HC24.ucsd.HanxianHuang.v02.pdf
-a----         2024/8/27     12:35        2135489 50_HC2024.Poster.CMU.Tang.pdf
-a----         2024/8/27     12:35        3524360 52_HC2024.Poster.ASTAR-NUS.Nambiar.pdf
-a----         2024/8/27     12:35        8288684 56_HC2024.Intel.AGihon.pdf
-a----         2024/8/27     12:35        1685089 6-HC2024.nvidia.MarkRen.agent.v04.pdf
-a----         2024/8/27     12:35        1765420 60_HC2024.Intel.RomanKaplan.Gaudi3-0826.pdf
-a----         2024/8/27     12:35        3068344 61_HC2024.Broadcom.ManishMehta.v2-NO-VIDEO.pdf
-a----         2024/8/27     12:35        9784900 64_HC2024.NVIDIA.TirumalaWong.pdf
-a----         2024/8/27     12:35        1315806 67_HC2024.Poster.NUS.Gupta.v4.pdf
-a----         2024/8/27     12:35        1258165 75_HC2024.Poster.UEC.KhaiDuyNguyen.v3.pdf
-a----         2024/8/27     12:35       31975293 76_HC2024.Poster.Berkeley.Schmulbach.v2.pdf
-a----         2024/8/27     12:35        2631078 83_HC2024.Furiosa.JunePaik.Final.pdf
-a----         2024/8/27     12:35         706771 86_HC2024.Poster.Purdue.AnnusZulfiqar.v2.0.pdf
-a----         2024/8/27     12:35        2057075 88_HC2024.Tenstorrent.Jasmina.Davor.v7.pdf
-a----         2024/8/27     12:35        1732500 HC2024.T2.Frore.Sathyamurthy.final.pdf
-a----         2024/8/27     12:35       18262836 HC2024.T2.Nvidia.Heydari.final.v1.pdf
-a----         2024/8/27     12:35        4676079 HC2024.T2.Phononic.Edwards.final.v1.pdf
-a----         2024/8/27     12:35        3707753 HC2024.T2.Qualcomm.NaderNikfar.final-0824.pdf
-a----         2024/8/27     12:35        2962277 HC2024.T2.Supermicro.Garvens.final.pdf
-a----         2024/8/27     12:35        1844294 HotChips - 2024-08-26.pdf
-a----         2024/8/27     12:35         697737 HotChips2024_GC_remarks.pdf

    目录: O:\软件生态部\Base SW\Multi GPU\SCCL1.0 & 2.0 shared\nvidia 资料\NVIDIA DPU资料包

Mode                 LastWriteTime         Length Name

----                 -------------         ------ ----

d-----         2025/8/22     13:56                NVIDIA DOCA 开发者专场

d-----         2025/8/22     13:56                NVIDIA DPU 公开课

-a----         2021/7/18     21:53        6640279 智东西公开课 - 数据中心处理器 DPU 公开课 NVIDIA 专场课件 - 面向数据中心的巨大

                                                  突破 - NVIDIA DPU 集数据中心于芯片-NVIDIA 网络事业部亚太区市场开发高

                                                  级总监宋庆春.pdf

    目录: O:\软件生态部\Base SW\Multi GPU\SCCL1.0 & 2.0 shared\nvidia 资料\NVIDIA DPU资料包\NVIDIA DOCA开发者专场

Mode                 LastWriteTime         Length Name

----                 -------------         ------ ----

-a----         2021/7/18     21:53        2227264 智东西公开课 - NVIDIA DOCA 开发者专场课件 -DOCA SDK 特性解读和基于 DOCA 的 D
                                                  PU 应用开发指南 -NVIDIA 网络事业部软件开发经理王加伟.pdf
-a----         2021/7/18     21:53        1458978 智东西公开课 - NVIDIA DOCA 开发者专场课件 -DPU 的出现是现代数据中心的必
                                                  然趋势 - NVIDIA 系统软件开发部研发总监马洪岩 .pdf
-a----         2021/7/18     21:53        3083870 智东西公开课 - NVIDIA DOCA 开发者专场课件 -NVIDIA DPU 在现代数据中心中的
                                                  应用及未来技术演进 -NVIDIA 网络事业部亚太区市场开发高级总监宋庆春.pdf

    目录: O:\软件生态部\Base SW\Multi GPU\SCCL1.0 & 2.0 shared\nvidia 资料\NVIDIA DPU资料包\NVIDIA DPU公开课

Mode                 LastWriteTime         Length Name

----                 -------------         ------ ----

-a----         2021/7/18     21:53        1438702 智东西公开课 - NVIDIA DPU 公开课课件 -DPU 及其在裸金属云中的关键技术 - NVI
                                                  DIA 网卡首席架构师陈志辉.pdf
-a----         2021/7/18     21:53       16234476 智东西公开课 -NVIDIA DPU 公开课课件 -DPU 在裸金属云中的关键价值 -UCloud 资
                                                  深技术专家马彦青.pdf

    目录: O:\软件生态部\Base SW\Multi GPU\SCCL1.0 & 2.0 shared\nvidia 资料\NVSwitch

Mode                 LastWriteTime         Length Name

----                 -------------         ------ ----

-a----          2024/9/5     10:56          48646 port-logic.webp
