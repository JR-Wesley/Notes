---
tags: 
- AI
---

# AI 系统

> AI 基础设施与系统软件领域：覆盖 GPU/硬件、并行与分布式训练、推理优化、AI 编译器、主流框架（Megatron/PyTorch/vLLM/llama.cpp）及大模型算法基础。以系统视角串联从芯片到模型部署的全栈知识。

## 主题地图

| 子主题 | 内容概述 | 进度 |
| --- | --- | --- |
| [[02-AISystem/AI-compiler/AI-compiler|AICompiler]] | AI 编译器：TVM、LLVM 与 AI 编译器原理 | 起步 |
| [[AI-sys-review|AISysReview]] | AI 系统综述：openMLsys、Sys4AI、ZOMI infra、LLM 分布式训练系统 | 进行中 |
| [[02-AISystem/algorithm-and-model/algorithm-and-model|AlgorithmAndModel]] | 算法与模型：机器学习基础、矩阵论、LLM/MoE/RL 模型侧知识 | 进行中 |
| [[02-AISystem/cluster-and-hardware/cluster-and-hardware|ClusterAndHardware]] | 集群与硬件：AI 芯片、半导体工艺、单机拓扑、集群架构 | 进行中 |
| [[02-AISystem/Distributed/Distributed|Distributed]] | 分布式：NCCL、NVSHMEM、DeepEP、并行策略、RDMA、分布式训练 | 进行中 |
| [[02-AISystem/Framework/Framework|Framework]] | 训练/推理框架：Megatron、PyTorch、vLLM、llama.cpp、HF | 进行中 |
| [[02-AISystem/GPU/GPU|GPU]] | GPU 体系：CUDA 编程指南、GPU 架构/ISA、tensor-core、PMPP | 进行中 |
| [[02-AISystem/hpc-basics/hpc-basics|HPCBasics]] | HPC 基础：MPI/OpenMP、向量化、并行计算理论、IO | 进行中 |
| [[02-AISystem/Inference/Inference|Inference]] | 推理：Attention/kv-cache、量化、推理加速方法论、解码 | 进行中 |
| [[02-AISystem/job-interview/job-interview|JobInterview]] | AI 系统求职：知识体系、行业现状、面试八股 | 进行中 |

## 关键笔记

- [[02-AISystem/AI-sys-review|AI System 知识体系与项目导航]]：AI 系统知识全景与项目入口

## 学习路径

1. [[02-AISystem/hpc-basics/hpc-basics|HPCBasics]] → [[02-AISystem/GPU/GPU|GPU]] 建立硬件与并行基础
2. [[02-AISystem/Distributed/Distributed|Distributed]]（NCCL/并行策略）→ [[02-AISystem/Framework/Framework|Framework]]（Megatron/PyTorch）
3. [[02-AISystem/Inference/Inference|Inference]] → [[02-AISystem/AI-compiler/AI-compiler|AICompiler]] 覆盖推理与编译优化
4. 配合 [[02-AISystem/algorithm-and-model/algorithm-and-model|AlgorithmAndModel]] 补齐模型侧知识
