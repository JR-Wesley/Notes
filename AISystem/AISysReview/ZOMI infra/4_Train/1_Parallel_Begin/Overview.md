---
dateCreated: 2025-08-09
dateModified: 2025-08-09
---
# 分布式并行基础

分布式训练可以将模型训练任务分配到多个计算节点上,从而加速训练过程并处理更大的数据集。模型是一个有机的整体，简单增加机器数量并不能提升算力，需要有并行策略和通信设计，才能实现高效的并行训练。本节将会重点打开业界主流的分布式并行框架 DeepSpeed、Megatron-LM 的核心多维并行的特性来进行原理介绍。

## 内容大纲

| 大纲    | 小节                               | 链接                                                                                                                                                    |
| :---- | :------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------- |
| 分布式并行 | 01 分布式并行框架介绍                     | [PPT](https://github.com/Infrasys-AI/AIInfra/blob/main/04Train/01ParallelBegin/01Introduction.pdf), [视频](https://www.bilibili.com/video/BV1op421C7wp) |
| 分布式并行 | 02 DeepSpeed 介绍                  | [PPT](https://github.com/Infrasys-AI/AIInfra/blob/main/04Train/01ParallelBegin/02DeepSpeed.pdf), [视频](https://www.bilibili.com/video/BV1tH4y1J7bm)    |
| 分布式并行 | 03 优化器并行 ZeRO1/2/3 原理            | [PPT](https://github.com/Infrasys-AI/AIInfra/blob/main/04Train/01ParallelBegin/03DSZero.pdf), [视频](https://www.bilibili.com/video/BV1fb421t7KN)       |
| 分布式并行 | 04 Megatron-LM 代码概览              | [PPT](https://github.com/Infrasys-AI/AIInfra/blob/main/04Train/01ParallelBegin/04Megatron.pdf), [视频](https://www.bilibili.com/video/BV12J4m1K78y)     |
| 分布式并行 | 05 大模型并行与 GPU 集群配置               | [PPT](https://github.com/Infrasys-AI/AIInfra/blob/main/04Train/01ParallelBegin/05MGConfig.pdf), [视频](https://www.bilibili.com/video/BV1NH4y1g7w4)     |
| 分布式并行 | 06 大模型并行与 GPU 集群配置               | [PPT](https://github.com/Infrasys-AI/AIInfra/blob/main/04Train/01ParallelBegin/06MGTPPrinc.pdf), [视频](https://www.bilibili.com/video/BV1ji421C7jH)    |
| 分布式并行 | 07 Megatron-LM TP 原理             | [PPT](https://github.com/Infrasys-AI/AIInfra/blob/main/04Train/01ParallelBegin/07MGTPCode.pdf), [视频](https://www.bilibili.com/video/BV1yw4m1S71Y)     |
| 分布式并行 | 08 Megatron-LM TP 代码解析           | [PPT](https://github.com/Infrasys-AI/AIInfra/blob/main/04Train/01ParallelBegin/07MGTPCode.pdf), [视频](https://www.bilibili.com/video/BV1cy411Y7B9)     |
| 分布式并行 | 09 Megatron-LM SP 代码解析           | [PPT](https://github.com/Infrasys-AI/AIInfra/blob/main/04Train/01ParallelBegin/08MGSPPrinc.pdf), [视频](https://www.bilibili.com/video/BV1EM4m1r7tm)    |
| 分布式并行 | 10 Megatron-LM PP 基本原理           | [PPT](https://github.com/Infrasys-AI/AIInfra/blob/main/04Train/01ParallelBegin/10MGPPCode.pdf), [视频](https://www.bilibili.com/video/BV18f42197Sx)     |
| 分布式并行 | 11 流水并行 1F1B/1F1B Interleaved 原理 | [PPT](https://github.com/Infrasys-AI/AIInfra/blob/main/04Train/01ParallelBegin/10MGPPCode.pdf), [视频](https://www.bilibili.com/video/BV1aD421g7yZ)     |
| 分布式并行 | 12 Megatron-LM 流水并行 PP 代码解析      | [PPT](https://github.com/Infrasys-AI/AIInfra/blob/main/04Train/01ParallelBegin/10MGPPCode), [视频](https://www.bilibili.com/video/BV1hs421g7vN)         |
