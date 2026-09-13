---
type: plan
domain: 02-AISystem
topic: AI-sys-review
status: active
tags: [AI, AI-System, learning-map]
---

# AI System 知识体系与项目导航

> 本知识地图按知识分类和项目入口组织，适合根据实际项目选择学习内容。

## 目标与使用方式

本目录服务于打好计算机系统基础、增强 AI 系统项目能力，并形成理解前沿技术的整体框架。学习不按时间线性推进：遇到实际项目时，先定位知识域，再沿“前置基础 → 系统机制 → 当前笔记 → 参考资料 → 实验或源码证据”补齐。阅读完成不等于掌握；当前反馈证据：尚无。

## 总体知识树

AI System
├── A 系统基础与性能模型
├── B 模型与数据工作负载
├── C 机器学习框架
├── D 单节点执行与算子优化
├── E 通信、存储与集群
├── F 分布式训练
├── G 推理系统
└── H AI 编译器与前沿案例

## A. 系统基础与性能模型

| 子领域 | 核心知识点 | 当前笔记 | 参考资料 | 完成证据 |
|---|---|---|---|---|
| 并发与执行 | 进程、线程、同步、任务/算子并行 | [[hpc-basics/并行计算基础理论]]、[[GPU/CPU和GPU的并行与并发]] | 01-ComputerScience 中 OS/并发笔记 | 解释同步开销并运行并行程序 |
| 内存与 I/O | Cache、带宽、局部性、向量化、I/O | [[hpc-basics/向量化]]、[[hpc-basics/IO]] | [[GPU/CUDA-Programming-Guide/2-2存储模型]] | 测量计算受限和访存受限 |
| 通信性能 | Latency、Bandwidth、Throughput、扩展效率 | [[hpc-basics/通信]] | [[AI-sys-review/ZOMI-infra/2-store-communication/2-store-communication]] | 记录消息规模与并发结果 |
| 性能方法 | 基线、对照实验、瓶颈定位 | [[GPU/Operator/性能分析]] | AI-infra 的算子优化框架 | 给出瓶颈证据而非只报加速比 |

## B. 模型与数据工作负载

| 子领域 | 核心知识点 | 当前笔记 | 参考资料 | 完成证据 |
|---|---|---|---|---|
| Transformer | Tensor、Embedding、Attention、数据流 | [[AI-sys-review/ZOMI-infra/6-algorithm-data/1-basic/Transformer]]、[[AI-sys-review/ZOMI-infra/6-algorithm-data/1-basic/Embedding]] | [[AI-sys-review/ZOMI-infra/6-algorithm-data/1-basic/LLM]] | 画张量形状与算子 |
| 注意力变体 | MQA/GQA/MLA、长序列、KV 复用 | [[Inference/Attention]]、[[Inference/kv-cache]] | [[AI-sys-review/ZOMI-infra/6-algorithm-data/1-basic/4-attention]]、a-visual-guide/NLP-and-LLM.md | 解释显存、带宽和质量取舍 |
| MoE 与路由 | Expert、Router、负载均衡、专家并行 | [[AI-sys-review/ZOMI-infra/6-algorithm-data/2-MoE/Overview]]、[[AI-sys-review/ZOMI-infra/6-algorithm-data/DeepSeek/DeepSeek]] | [[AI-sys-review/a-visual-guide/MOE]] | 连接路由、通信和调度 |
| 数据与规模 | Tokenizer、参数量、Scaling Law、训练数据 | [[AI-sys-review/ZOMI-infra/6-algorithm-data/1-basic/Tokenizer]]、[[AI-sys-review/ZOMI-infra/6-algorithm-data/1-basic/7-parameter]] | [[AI-sys-review/ZOMI-infra/0-summary/1-scaling-law/pretraining-scaling]] | 估算计算、显存和数据需求 |

## C. 机器学习框架

| 子领域 | 核心知识点 | 当前笔记 | 参考资料 | 完成证据 |
|---|---|---|---|---|
| 编程接口 | Python/C++、Tensor、Operator、Dispatcher | [[Framework/Pytorch/Pytorch]]、[[AI-sys-review/openMLsys]] | openMLsys 编程接口章节 | 从 API 追踪到设备算子 |
| 自动微分与计算图 | 前向/反向、依赖、动态图/静态图、循环展开 | [[AI-sys-review/openMLsys]] | openMLsys 计算图章节 | 画小模型计算图 |
| 设备后端 | CPU/CUDA/MPS、算子分发、线性代数库 | [[Framework/Pytorch/Pytorch-Lib/device-backend/Overview]]、[[Framework/Pytorch/Pytorch-Lib/device-backend/CUDA-backend]] | [[Framework/Pytorch/Pytorch-Lib/Overview]] | 固定版本记录后端调用路径 |
| 运行时 | 内存分配、调度、异步执行、同步 | [[Framework/Pytorch/Pytorch-Lib/device-backend/device-backend]] | openMLsys 后端与运行时章节 | profiler 或源码解释调度 |

## D. 单节点执行与算子优化

| 子领域 | 核心知识点 | 当前笔记 | 参考资料 | 完成证据 |
|---|---|---|---|---|
| CUDA 执行模型 | Grid、Block、Warp、线程层级、同步 | [[GPU/CUDA-Programming-Guide/2-1执行模型]]、[[GPU/PMPP/1-Fundamental-Concepts/4-compute-architecture-and-scheduling]] | [[GPU/CUDA-Programming-Guide/CUDA-Programming-Guide]] | 向量加法/归约及线程映射 |
| GPU 内存与架构 | Register、Shared Memory、L1/L2、HBM、Bank Conflict | [[GPU/CUDA-Programming-Guide/2-2存储模型]]、[[GPU/GPU-arch/GPU-arch]] | [[GPU/PMPP/1-Fundamental-Concepts/5-memory-architecture-and-data-locality]] | 对比访存方案与 profiler |
| Tensor Core 与矩阵 | MMA、WMMA、GEMM、CuTe、布局 | [[GPU/tensor-core/tensor-core]]、[[GPU/tensor-core/MMA]]、[[GPU/tensor-core/CuTe/cutlass与GEMM]] | [[GPU/tensor-core/CuTe/Overview]] | 矩阵乘 baseline/优化对比 |
| 优化方法 | Tiling、Fusion、Pipeline、量化、稀疏 | [[GPU/Operator/算子融合]]、[[Inference/Quantization/Quantization]]、[[AI-sys-review/AI-infra]] | [[GPU/Operator/典型算法分析]] | 说明数学、局部性、硬件映射 |

## E. 通信、存储与集群

| 子领域 | 核心知识点 | 当前笔记 | 参考资料 | 完成证据 |
|---|---|---|---|---|
| 通信原语 | Send/Recv、Broadcast、Reduce、AllReduce、All-to-All | [[hpc-basics/MPI]]、[[AI-sys-review/ZOMI-infra/2-store-communication/1-collective-communication/Overview]] | [[AI-sys-review/ZOMI-infra/2-store-communication/1-collective-communication/2-cc-overview]] | point-to-point 与 collective 基准 |
| 通信库 | MPI、NCCL、Gloo、API 与拓扑 | [[AI-sys-review/ZOMI-infra/2-store-communication/2-comm-lib/Overview]]、[[AI-sys-review/ZOMI-infra/2-store-communication/2-comm-lib/3-NCCL]] | [[AI-sys-review/ZOMI-infra/2-store-communication/2-comm-lib/4-NCCL-API]] | 比较接口、性能和边界 |
| RDMA 与存储 | Zero-copy、Memory Registration、RDMA、数据路径 | [[AI-sys-review/ZOMI-infra/2-store-communication/RDMA-intro]] | 2026-09-NVSHMEM-Knowledge-Notes 相关交付 | 记录复制次数和瓶颈 |
| 集群与拓扑 | Scale-up/Scale-out、节点、交换机、故障 | [[AI-sys-review/ZOMI-infra/1-AI-cluster/1-AI-cluster]]、[[AI-sys-review/ZOMI-infra/1-AI-cluster/4-performance/4-performance]] | [[AI-sys-review/ZOMI-infra/2-store-communication/scaleup-and-out]] | 画拓扑并解释带宽/延迟 |

## F. 分布式训练

| 子领域 | 核心知识点 | 当前笔记 | 参考资料 | 完成证据 |
|---|---|---|---|---|
| 训练总览 | 算力/显存瓶颈、training step、同步、checkpoint | [[AI-sys-review/LLM分布式训练系统/LLM分布式训练系统]] | [[AI-sys-review/ZOMI-infra/4-train/4-train]] | 分解计算、通信、等待和内存 |
| 并行策略 | Data、Model/Tensor、Pipeline、混合并行 | [[AI-sys-review/LLM分布式训练系统/4-并行策略]]、[[AI-sys-review/ZOMI-infra/4-train/1-parallel-begin/3-data-parallel]] | [[AI-sys-review/ZOMI-infra/4-train/2-parallel-adv/10-pipeline]] | 比较切分、通信、显存和扩展 |
| 参数与显存 | ZeRO、参数/梯度/优化器状态切分 | [[AI-sys-review/ZOMI-infra/4-train/2-parallel-adv/5-zero]] | [[AI-sys-review/ZOMI-infra/4-train/2-parallel-adv/2-DeepSpeed-zeros-TODO]] | 估算不同策略显存 |
| 运行时 | 调度、集合通信、容错、负载均衡 | [[AI-sys-review/LLM分布式训练系统/6-运行时]]、[[AI-sys-review/LLM分布式训练系统/5-集合通信]] | [[AI-sys-review/ZOMI-infra/4-train/4-train]] | 源码追踪或多进程实验 |

## G. 推理系统

| 子领域 | 核心知识点 | 当前笔记 | 参考资料 | 完成证据 |
|---|---|---|---|---|
| 推理流程 | Prefill、Decode、Sampling、Latency/Throughput | [[Inference/解码]]、[[Inference/Inference]] | [[AI-sys-review/ZOMI-infra/5-inference/1-foundation/Overview]] | 按 batch/序列长度测量 |
| KV Cache | Cache 组织、显存占用、复用、分页 | [[Inference/kv-cache]] | [[Framework/vLLM/框架分析]]、[[AI-sys-review/ZOMI-infra/5-inference/2-infer-speed-up/Overview]] | 解释序列长度与显存 |
| 压缩与算子 | Quantization、Sparsity、Fusion、Flash/Paged Attention | [[Inference/Quantization/Quantization]]、[[Inference/融合算子/融合算子]]、[[Inference/Quantization/NN稀疏]] | [[Inference/推理优化参考资料]] | 比较质量、显存和速度 |
| 服务运行时 | Continuous Batching、调度、资源隔离 | [[Framework/vLLM/框架分析]]、[[Inference/推理加速方法论]] | [[AI-sys-review/ZOMI-infra/5-inference/5-inference]] | 端到端服务基准 |

## H. AI 编译器与前沿案例

| 子领域 | 核心知识点 | 当前笔记 | 参考资料 | 完成证据 |
|---|---|---|---|---|
| 编译器前端 | IR、类型/形状、自动微分、图优化、Lowering | [[AI-compiler/AI-compiler]]、[[AI-sys-review/openMLsys]] | openMLsys AI 编译器章节 | 画模型到 IR 的转换 |
| 编译器后端 | 算子选择、Codegen、内存规划、调度 | [[Framework/Pytorch/Pytorch-Lib/device-backend/Overview]]、[[AI-sys-review/openMLsys]] | [[AI-sys-review/AI-infra]] | 对比图或 kernel 结果 |
| 自动优化 | Shape Specialization、Autotuning、Cost Model | [[GPU/Operator/性能分析]]、[[AI-sys-review/AI-infra]] | [[AI-sys-review/Sys4AI]] | 记录搜索空间与收益 |
| 纵向案例 | vLLM、MoE、DeepSeek、完整 AI infra | [[Framework/vLLM/框架分析]]、[[AI-sys-review/ZOMI-infra/ZOMI-infra]] | [[AI-sys-review/a-visual-guide/a-visual-guide]]、[[AI-sys-review/Sys4AI]] | 写清瓶颈、依赖、代价、来源 |

## 项目驱动取用规则

| 项目问题 | 首先定位 | 再补关联域 | 最小证据 |
|---|---|---|---|
| 单算子慢/GPU 利用率低 | D | A、C | baseline、优化版、profiler、解释 |
| 多卡训练扩展差 | E、F | A、B | step 分解、通信基准、策略比较 |
| 推理延迟或显存不足 | G | B、C、D | 端到端基准、KV Cache/调度解释 |
| 框架调用路径不清楚 | C | D、H | 固定 commit 调用链、架构图 |
| 新论文/新系统难判断 | H | A–G 相关域 | 来源、依赖、代价、边界、待验证问题 |

## 学习范围与参考资料

### 学习范围

| 方向 | 学习重点 | 暂不作为主线 |
|---|---|---|
| 张量与框架 | PyTorch 张量计算、线性代数建模、代表性模型结构 | 模型准确率比较和大范围算法综述 |
| 异构与高性能计算 | GPU、CUDA、Tensor Core、并行编程和性能分析 | 与当前项目无关的硬件细节 |
| 分布式系统 | 并行策略、集合通信、NCCL、MPI、RDMA、训练运行时 | 泛化的 MLOps 全生命周期 |
| 推理与编译 | KV Cache、批处理、算子优化、编译器、运行时 | 暂不独立展开数据分析与处理 |

### 课程、教材与系统综述

| 用途 | 参考资料 |
|---|---|
| 机器学习系统教材 | [openMLsys](https://openmlsys.github.io/)、[机器学习系统设计与实现](http://link.zhihu.com/?target=https%3A//openmlsys.github.io/) |
| AI 系统课程 | [UC Berkeley AI Systems](http://link.zhihu.com/?target=https%3A//ucbrise.github.io/cs294-ai-sys-sp22/)、[CSE 599W](http://link.zhihu.com/?target=http%3A//dlsys.cs.washington.edu/)、[Deep Learning Systems](http://link.zhihu.com/?target=https%3A//dlsyscourse.org/)、[ML Compilation](http://link.zhihu.com/?target=https%3A//mlc.ai/summer22-zh/) |
| 系统综述 | [MLSys 方向综述](https://zhuanlan.zhihu.com/p/104444471)、[MLSys 入门指南](http://link.zhihu.com/?target=https%3A//fazzie-key.cool/2023/02/21/MLsys/)、[MLSys Book](https://www.mlsysbook.ai/)、[Awesome MLSys Bloggers](https://mlsys-learner-resources.github.io/Awesome-MLSys-Blogger/) |
| AI 系统实现 | [AISystem GitHub](http://link.zhihu.com/?target=https%3A//github.com/chenzomi12/AISystem)、[Microsoft System for AI](http://link.zhihu.com/?target=https%3A//github.com/microsoft/AI-System)、[Stanford MLSys Seminar](http://link.zhihu.com/?target=https%3A//www.youtube.com/playlist%3Flist%3DPLSrTvUm384I9PV10koj_cqit9OfbJXEkq) |
| 编译器学习 | [Dive into Deep Learning Compiler](http://link.zhihu.com/?target=https%3A//tvm.d2l.ai/)、[MLC course](http://link.zhihu.com/?target=https%3A//mlc.ai/summer22-zh/)、[中科院智能计算系统课程](https://novel.ict.ac.cn/aics/) |

### 模型与 LLM

| 用途 | 参考资料 |
|---|---|
| LLM 基础与实践 | 动手学深度学习前 5 章；[CS336](https://cs336.stanford.edu/)、[Happy-LLM](https://github.com/datawhalechina/happy-llm)、[LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) |
| LLM 结构与部署 | [Llama2 流程](https://zhuanlan.zhihu.com/p/651248009)、[大模型部署工程](https://zhuanlan.zhihu.com/c_1905912040196931924)、[byhand.ai](https://www.byhand.ai/)、[spaces.ac.cn](https://spaces.ac.cn/) |
| 讲解与复习 | [StatQuest](https://www.youtube.com/@statquest/videos)、[freeCodeCamp](https://www.youtube.com/@freecodecamp/videos)、[图解大模型](https://www.zhihu.com/people/aaron-73-65/posts) |

### GPU、并行计算与硬件

| 用途 | 参考资料 |
|---|---|
| CUDA 与 GPU 编程 | [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)、[CUDA 博客合集](https://github.com/caiwanxianhust/CUDA-BLOG)、[CUDA 视频课程](https://www.youtube.com/watch?v=Sdjn9FOkhnA&list=PL5B692fm6--vWLhYPqLcEu6RF3hXjEyJr&index=1) |
| 并行理论 | 《并行算法设计与性能优化》、[并行计算课程与资料](https://goodcucumber.github.io/x40paraguide/x40.html) |
| AI 硬件架构 | [Eyeriss Tutorial](https://eyeriss.mit.edu/tutorial-previous.html)、[Hardware Architecture for Deep Learning](https://csg.csail.mit.edu/6.5930/index.html)、[TinyML and Efficient Deep Learning Computing](https://hanlab.mit.edu/courses/2024-fall-65940)、[AAML2024](https://nycu-caslab.github.io/AAML2024/index.html)、[Computer Organization](https://nycu-caslab.github.io/CO2024/index.html) |
| 加速器与验证 | [CFU Playground](https://cfu-playground.readthedocs.io/en/latest/index.html)、[AAML2024 Labs](https://nycu-caslab.github.io/AAML2024/labs/lab_2.html) |

### 社区与专题参考

| 用途 | 参考资料 |
|---|---|
| vLLM 与推理 | [vLLM PD](https://www.zhihu.com/people/52-34-86-1/posts)、[从零实现 vLLM](https://zhuanlan.zhihu.com/c_1930254288103403625)、[LLM 推理框架](https://zhuanlan.zhihu.com/c_1916901019268391457) |
| 分布式训练 | [分布式训练专题](https://www.cnblogs.com/sunstrikes/collections/17032)、[AI纵贯线](https://www.zhihu.com/column/c_1777819405453787137) |
| AI 系统综述 | [AI System 综述](https://zhuanlan.zhihu.com/p/20076957712)、[AI 系统课程资料](https://www.zhihu.com/people/yan-xin-kai-38/posts)、[个人博客](https://shichaoxin.com/) |

## 目录整理与复盘边界

未来可将资料归入 00-overview、01-foundations、02-single-node、03-distributed、04-inference、05-compiler、06-case-studies、07-reading-pool。本次只重写导航，不执行迁移；迁移前须建立旧路径—新路径映射、附件清单和哈希基线，再分批验证 WikiLink、Canvas/Base、插件索引和自动区块。

复盘以项目子问题完成为触发点：检查机制解释、运行/源码证据、未验证假设和需要补的知识域。若持续只摘录不解释，减少资料入口；若实验无法解释，回到相邻基础域；若多个项目反复触及同一缺口，再建立独立概念笔记或 MOC。
