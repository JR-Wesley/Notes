---
dateCreated: 2025-07-31
dateModified: 2025-08-03
---

ZOMI 整理

# LLM Infra

这个开源项目英文名字叫做 **AIInfra**，中文名字叫做 **AI 基础设施**。大模型是基于 AI 集群的全栈软硬件性能优化，通过最小的每一块 AI 芯片组成的 AI 集群，编译器使能到上层的 AI 框架，训练过程需要分布式并行、集群通信等算法支持，而且在大模型领域最近持续演进如智能体等新技术。

本开源课程主要是跟大家一起探讨和学习人工智能、深度学习的系统设计，而整个系统是围绕着 ZOMI 在工作当中所积累、梳理、构建 AI 大模型系统的基础软硬件栈，因此成为 AI 基础设施。希望跟所有关注 AI 开源课程的好朋友一起探讨研究，共同促进学习讨论。

与 **AISystem**[[https://github.com/Infrasys-AI/AISystem](https://github.com/Infrasys-AI/AISystem)] 项目最大的区别就是 **AIInfra** 项目主要针对大模型，特别是大模型在分布式集群、分布式架构、分布式训练、大模型算法等相关领域进行深度展开。

![](assets/ZOMI-AIInfra.assets/aifoundation01.jpg)

## 课程内容大纲

课程主要包括以下模块，内容陆续更新中，欢迎贡献：

| 序列  | 教程内容                                                                                                                                               | 简介                                                                                                                              | 地址                                                                       |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| 00  | [大模型系统概述](https://github.com/Infrasys-AI/AIInfra/blob/main/README.md#00-%E5%A4%A7%E6%A8%A1%E5%9E%8B%E7%B3%BB%E7%BB%9F%E6%A6%82%E8%BF%B0)           | 系统梳理了大模型关键技术点，涵盖 Scaling Law 的多场景应用、训练与推理全流程技术栈、AI 系统与大模型系统的差异，以及未来趋势如智能体、多模态、轻量化架构和算力升级。| [Slides](https://github.com/Infrasys-AI/AIInfra/blob/main/00Summary)     |
| 01  | [AI 计算集群](https://github.com/Infrasys-AI/AIInfra/blob/main/README.md#01-ai-%E8%AE%A1%E7%AE%97%E9%9B%86%E7%BE%A4)                                   | 大模型虽然已经慢慢在端测设备开始落地，但是总体对云端的依赖仍然很重很重，AI 集群会介绍集群运维管理、集群性能、训练推理一体化拓扑流程等内容。| [Slides](https://github.com/Infrasys-AI/AIInfra/blob/main/01AICluster)   |
| 02  | [通信与存储](https://github.com/Infrasys-AI/AIInfra/blob/main/README.md#02-%E9%80%9A%E4%BF%A1%E4%B8%8E%E5%AD%98%E5%82%A8)                               | 大模型训练和推理的过程中都严重依赖于网络通信，因此会重点介绍通信原理、网络拓扑、组网方案、高速互联通信的内容。存储则是会从节点内的存储到存储 POD 进行介绍。| [Slides](https://github.com/Infrasys-AI/AIInfra/blob/main/02StorComm)    |
| 03  | [集群容器与云原生](https://github.com/Infrasys-AI/AIInfra/blob/main/README.md#03-%E9%9B%86%E7%BE%A4%E5%AE%B9%E5%99%A8%E4%B8%8E%E4%BA%91%E5%8E%9F%E7%94%9F) | 讲解容器与 K8S 技术原理及 AI 模型部署实践，涵盖容器基础、Docker 与 K8S 核心概念、集群搭建、AI 应用部署、任务调度、资源管理、可观测性、高可靠设计等云原生与大模型结合的关键技术点。| [Slides](https://github.com/Infrasys-AI/AIInfra/blob/main/03DockCloud)   |
| 04  | [分布式训练](https://github.com/Infrasys-AI/AIInfra/blob/main/README.md#04-%E5%A4%A7%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83)                               | 大模型训练是通过大量数据和计算资源，利用 Transformer 架构优化模型参数，使其能够理解和生成自然语言、图像等内容，广泛应用于对话系统、文本生成、图像识别等领域。| [Slides](https://github.com/Infrasys-AI/AIInfra/blob/main/04Train)       |
| 05  | [分布式推理](https://github.com/Infrasys-AI/AIInfra/blob/main/README.md#05-%E5%A4%A7%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)                               | 大模型推理核心工作是优化模型推理，实现推理加速，其中模型推理最核心的部分是 Transformer Block。本节会重点探讨大模型推理的算法、调度策略和输出采样等相关算法。| [Slides](https://github.com/Infrasys-AI/AIInfra/blob/main/05Infer)       |
| 06  | [大模型算法与数据](https://github.com/Infrasys-AI/AIInfra/blob/main/README.md#06-%E5%A4%A7%E6%A8%A1%E5%9E%8B%E7%AE%97%E6%B3%95%E4%B8%8E%E6%95%B0%E6%8D%AE) | Transformer 起源于 NLP 领域，近期统治了 CV/NLP/多模态的大模型，我们将深入地探讨 Scaling Law 背后的原理。在大模型算法背后数据和算法的评估也是核心的内容之一，如何实现 Prompt 和通过 Prompt 提升模型效果。| [Slides](https://github.com/Infrasys-AI/AIInfra/blob/main/06AlgoData)    |
| 07  | [大模型应用](https://github.com/Infrasys-AI/AIInfra/blob/main/README.md#07-%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%BA%94%E7%94%A8)                               | 当前大模型技术已进入快速迭代期。这一时期的显著特点就是技术的更新换代速度极快，新算法、新模型层出不穷。因此本节内容将会紧跟大模型的时事内容，进行深度技术分析。| [Slides](https://github.com/Infrasys-AI/AIInfra/blob/main/07Application) |

![](assets/ZOMI-AIInfra.assets/aifoundation02.png)

# 课程细节


### **[00. 大模型系统概述](https://github.com/Infrasys-AI/AIInfra/blob/main/00Summary)**


大模型系统概述、Scaling Law 解读、训练推理流程、系统区别及未来趋势。

|编号|名称|具体内容|
|:-:|:--|:--|
|1|[Scaling Law 解读](https://github.com/Infrasys-AI/AIInfra/blob/main/00Summary/01ScalingLaw)|Scaling Law 在不同场景下的应用与演进|
|2|[训练推理全流程](https://github.com/Infrasys-AI/AIInfra/blob/main/00Summary/02TrainInfer)|大模型训练与推理全流程及软硬件优化|
|3|[与 AI 系统区别](https://github.com/Infrasys-AI/AIInfra/blob/main/00Summary/03Different)|AI 系统与大模型系统的通用性、资源与软件栈差异|
|3|[大模型系统发展](https://github.com/Infrasys-AI/AIInfra/blob/main/00Summary/04Develop)|大模型系统未来趋势：技术演进、场景应用与算力生态升级|

### **[01. AI 计算集群](https://github.com/Infrasys-AI/AIInfra/blob/main/01AICluster)**

AI 集群架构演进、万卡集群方案、性能建模与优化，GPU/NPU 精度差异及定位方法。

|编号|名称|具体内容|
|:-:|:--|:--|
|1|[计算集群之路](https://github.com/Infrasys-AI/AIInfra/blob/main/01AICluster/01Roadmap)|高性能计算集群发展与万卡 AI 集群建设及机房基础设施挑战|
|2|[集群建设之巅](https://github.com/Infrasys-AI/AIInfra/blob/main/01AICluster/02TypicalRepresent)|超节点计算集群架构演进与昇腾集群组网方案解析|
|3|[集群性能分析](https://github.com/Infrasys-AI/AIInfra/blob/main/01AICluster/03Analysis)|集群性能指标分析、建模与常见问题定位方法解析|

### **[02. 通信与存储](https://github.com/Infrasys-AI/AIInfra/blob/main/02StorComm)**

通信与存储篇：AI 集群组网技术、高速互联方案、集合通信原理与优化、存储系统设计及大模型挑战。

|编号|名称|具体内容|
|:-:|:--|:--|
|1|[集群组网之路](https://github.com/Infrasys-AI/AIInfra/blob/main/02StorComm/01Roadmap)|AI 集群组网架构设计与高速互联技术解析|
|2|[网络通信进阶](https://github.com/Infrasys-AI/AIInfra/blob/main/02StorComm/02NetworkComm)|网络通信技术进阶：高速互联、拓扑算法与拥塞控制解析|
|3|[集合通信原理](https://github.com/Infrasys-AI/AIInfra/blob/main/02StorComm/03CollectComm)|通信域、通信算法、集合通信原语|
|4|[集合通信库](https://github.com/Infrasys-AI/AIInfra/blob/main/02StorComm/04CommLibrary)|集合通信库技术解析：MPI、NCCL 与 HCCL 架构及算法原理|
|5|[集群存储之路](https://github.com/Infrasys-AI/AIInfra/blob/main/02StorComm/05StorforAI)|数据存储、CheckPoint 梯度检查点等存储与大模型结合的相关技术|

### **[03. 集群容器与云原生](https://github.com/Infrasys-AI/AIInfra/blob/main/03DockCloud)**


AI 集群云原生篇：容器技术、K8S 编排、AI 云平台与任务调度，提升集群资源管理与应用部署效率。

|编号|名称|具体内容|
|:-:|:--|:--|
|1|[容器时代](https://github.com/Infrasys-AI/AIInfra/blob/main/03DockCloud/01Roadmap)|容器技术基础与云原生架构解析，结合分布式训练应用实践|
|2|[容器初体验](https://github.com/Infrasys-AI/AIInfra/blob/main/03DockCloud/02DockerK8s)|Docker 与 K8S 基础原理及实战，涵盖容器技术与集群管理架构解析|
|3|[深入 K8S](https://github.com/Infrasys-AI/AIInfra/blob/main/03DockCloud/03DiveintoK8s)|K8S 核心机制深度解析：编排、存储、网络、调度与监控实践|
|4|[AI 云平台](https://github.com/Infrasys-AI/AIInfra/blob/main/03DockCloud/04CloudforAI)|AI 云平台演进与云原生架构解析，涵盖持续交付与智能化运维实践|

### **[04. 分布式训练](https://github.com/Infrasys-AI/AIInfra/blob/main/04Train)**

大模型训练全解析：并行策略、加速算法、微调与评估，覆盖训练到优化的完整流程。

|编号|名称|具体内容|
|:-:|:--|:--|
|1|[分布式并行基础](https://github.com/Infrasys-AI/AIInfra/blob/main/04Train/01ParallelBegin)|分布式并行的策略分类、模型适配与硬件资源优化对比|
|2|[大模型并行进阶](https://github.com/Infrasys-AI/AIInfra/blob/main/04Train/02ParallelAdv)|Megatron、DeepSeed 架构解析、MoE 扩展与高效训练策略|
|3|[大模型训练加速](https://github.com/Infrasys-AI/AIInfra/blob/main/04Train/03TrainAcceler)|大模型训练加速在算法优化、内存管理与通算融合策略解析|
|4|[后训练与强化学习](https://github.com/Infrasys-AI/AIInfra/blob/main/04Train/04PostTrainRL)|后训练与强化学习算法对比、框架解析与工程实践|
|5|[大模型微调 SFT](https://github.com/Infrasys-AI/AIInfra/blob/main/04Train/05FineTune)|大模型微调算法原理、变体优化与多模态实践|
|6|[大模型验证评估](https://github.com/Infrasys-AI/AIInfra/blob/main/04Train/06VerifValid)|大模型评估、基准测试与统一框架解析|

### **[05. 分布式推理](https://github.com/Infrasys-AI/AIInfra/blob/main/05Infer)**

大模型推理全解析：加速技术、架构优化、长序列处理与压缩方案，覆盖推理全流程与实战实践。

|编号|名称|具体内容|
|:-:|:--|:--|
|1|[基本概念](https://github.com/Infrasys-AI/AIInfra/blob/main/05Infer/01Foundation)|大模型推理流程、框架对比与性能指标解析|
|2|[大模型推理加速](https://github.com/Infrasys-AI/AIInfra/blob/main/05Infer/02InferSpeedUp)|大模型推理加速中 KV 缓存优化、算子改进与高效引擎解析|
|3|[架构调度加速](https://github.com/Infrasys-AI/AIInfra/blob/main/05Infer/03SchedSpeedUp)|架构调度加速中缓存优化、批处理与分布式系统调度解析|
|4|[长序列推理](https://github.com/Infrasys-AI/AIInfra/blob/main/05Infer/04LongInfer)|长序列推理算法优化、并行策略与高效生成方法解析|
|5|[输出采样](https://github.com/Infrasys-AI/AIInfra/blob/main/05Infer/05OutputSamp)|推理输出采样的基础方法、加速策略与 MOE 推理优化|
|6|[大模型压缩](https://github.com/Infrasys-AI/AIInfra/blob/main/05Infer/06CompDistill)|低精度量化、知识蒸馏与高效推理优化解析|
|7|[推理框架架构](https://github.com/Infrasys-AI/AIInfra/blob/main/05Infer/07Framework)|主流推理框架 vLLM、SGLang 等核心技术与部署实践|
|8|[DeepSeek 开源](https://github.com/Infrasys-AI/AIInfra/blob/main/05Infer/08DeepSeek)|DeepSeek 推理 FlashMLA、DeepEP 与高效算子加速解析|

### **[06. 大模型算法与数据](https://github.com/Infrasys-AI/AIInfra/blob/main/06AlgoData)**


大模型算法与数据全览：Transformer 架构、MoE 创新、多模态模型与数据工程全流程实践。

|编号|名称|具体内容|
|:-:|:--|:--|
|1|[Transformer 架构](https://github.com/Infrasys-AI/AIInfra/blob/main/06AlgoData/01Basic)|Transformer 架构原理深度介绍|
|2|[MoE 架构](https://github.com/Infrasys-AI/AIInfra/blob/main/06AlgoData/02MoE)|MoE(Mixture of Experts) 混合专家模型架构原理与细节实现|
|3|[创新架构](https://github.com/Infrasys-AI/AIInfra/blob/main/06AlgoData/03NewArch)|SSM、MMABA、RWKV、Linear Transformer、JPEA 等新大模型结构|
|4|[图文生成与理解](https://github.com/Infrasys-AI/AIInfra/blob/main/06AlgoData/04ImageTextGenerat)|多模态对齐、生成、理解及统一多模态架构解析|
|5|[视频大模型](https://github.com/Infrasys-AI/AIInfra/blob/main/06AlgoData/05VideoGenerat)|视频多模态理解与生成方法演进及 Flow Matching 应用|
|6|[语音大模型](https://github.com/Infrasys-AI/AIInfra/blob/main/06AlgoData/06AudioGenerat)|语音多模态识别、合成与端到端模型演进及推理应用|
|7|[数据工程](https://github.com/Infrasys-AI/AIInfra/blob/main/06AlgoData/07DataEngineer)|数据工程、Prompt Engine 等相关技术|

### **[07. 大模型应用](https://github.com/Infrasys-AI/AIInfra/blob/main/07Application)**

大模型应用篇：AI Agent 技术、RAG 检索增强生成与 GraphRAG，推动智能体与知识增强应用落地。

| 编号  | 名称                                                                                     | 具体内容                                    |
| :-: | :------------------------------------------------------------------------------------- | :-------------------------------------- |
| 00  | [大模型热点](https://github.com/Infrasys-AI/AIInfra/blob/main/07Application/00Others)       | OpenAI、WWDC、GTC 等大会技术洞察                 |
| 01  | [Agent 简单概念](https://github.com/Infrasys-AI/AIInfra/blob/main/07Application/01Sample)  | AI Agent 智能体的原理、架构                      |
| 02  | [Agent 核心技术](https://github.com/Infrasys-AI/AIInfra/blob/main/07Application/02AIAgent) | 深入 AI Agent 原理和核心                       |
| 03  | [检索增强生成(RAG)](https://github.com/Infrasys-AI/AIInfra/blob/main/07Application/03RAG)    | 检索增强生成技术的介绍                             |
| 04  | [自动驾驶](https://github.com/Infrasys-AI/AIInfra/blob/main/07Application/04AutoDrive)     | 端到端自动驾驶技术原理解析，萝卜快跑对产业带来的变化              |
| 05  | [具身智能](https://github.com/Infrasys-AI/AIInfra/blob/main/07Application/05Embodied)      | 关于对具身智能的技术原理、具身架构和产业思考                  |
| 06  | [生成推荐](https://github.com/Infrasys-AI/AIInfra/blob/main/07Application/06Remmcon)       | 推荐领域的革命发展历程，大模型迎来了生成式推荐新的增长             |
| 07  | [AI 安全](https://github.com/Infrasys-AI/AIInfra/blob/main/07Application/07Safe)         | 隐私计算发展过程，隐私计算未来发展如何？                    |
| 08  | [AI 历史十年](https://github.com/Infrasys-AI/AIInfra/blob/main/07News/06History)           | 过去十年 AI 大事件回顾，2012 到 2025 从模型、算法、芯片硬件发展 |