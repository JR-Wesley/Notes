---
dateCreated: 2025-08-09
dateModified: 2025-08-09
---
# MoE 混合专家

[](https://github.com/Infrasys-AI/AIInfra/blob/main/06AlgoData/02MoE/README.md#moe-混合专家)

MoE（Mixture of Experts）架构，即专家混合架构，是一种通过多个专家模块并行处理不同子任务，由门控网络依据输入数据动态分配，决定各专家模块参与度，以实现更高效、灵活处理复杂任务，提升模型表现与泛化能力的技术。

## 内容大纲

[](https://github.com/Infrasys-AI/AIInfra/blob/main/06AlgoData/02MoE/README.md#内容大纲)

> 建议优先下载 PDF 版本，PPT 版本会因为字体缺失等原因导致版本很丑哦~

| 大纲        | 小节                  | 链接                                                                                                                                                |
| :-------- | :------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------ |
| MOE 基本介绍  | 01 MOE 架构剖析         | [PPT](https://github.com/Infrasys-AI/AIInfra/blob/main/06AlgoData/02MoE/01MOEIntroducion.pdf), [视频](https://www.bilibili.com/video/BV17PNtekE3Y/) |
| MOE 前世今生  | 02 MOE 前世今生         | [PPT](https://github.com/Infrasys-AI/AIInfra/blob/main/06AlgoData/02MoE/02MOEHistory.pdf), [视频](https://www.bilibili.com/video/BV1y7wZeeE96/)     |
| MOE 核心论文  | 03 MOE 奠基论文         | [PPT](https://github.com/Infrasys-AI/AIInfra/blob/main/06AlgoData/02MoE/03MOECreate.pdf), [视频](https://www.bilibili.com/video/BV1MiAYeuETj/)      |
| MOE 核心论文  | 04 MOE 初遇 RNN       | [PPT](https://github.com/Infrasys-AI/AIInfra/blob/main/06AlgoData/02MoE/04MOERNN.pdf), [视频](https://www.bilibili.com/video/BV1RYAjeKE3o/)         |
| MOE 核心论文  | 05 GSard 解读         | [PPT](https://github.com/Infrasys-AI/AIInfra/blob/main/06AlgoData/02MoE/05MOEGshard.pdf), [视频](https://www.bilibili.com/video/BV1r8ApeaEyW/)      |
| MOE 核心论文  | 06 Switch Trans 解读  | [PPT](https://github.com/Infrasys-AI/AIInfra/blob/main/06AlgoData/02MoE/06MOESwitch.pdf), [视频](https://www.bilibili.com/video/BV1UsPceJEEQ/)      |
| MOE 核心论文  | 07 GLaM & ST-MOE 解读 | [PPT](https://github.com/Infrasys-AI/AIInfra/blob/main/06AlgoData/02MoE/07MOEGLaM_STMOE.pdf), [视频](https://www.bilibili.com/video/BV1L59qYqEVw/)  |
| MOE 核心论文  | 08 DeepSeek MOE 解读  | [PPT](https://github.com/Infrasys-AI/AIInfra/blob/main/06AlgoData/02MoE/08DeepSeekMoE.pdf), [视频](https://www.bilibili.com/video/BV1tE9HYUEdz/)    |
| MOE 架构原理  | 09 MOE 模型可视化        | [PPT](https://github.com/Infrasys-AI/AIInfra/blob/main/06AlgoData/02MoE/09MoECore.pdf), [视频](https://www.bilibili.com/video/BV1Gj9ZYdE4N/)        |
| 大模型遇 MOE  | 10 MoE 参数与专家        | [PPT](https://github.com/Infrasys-AI/AIInfra/blob/main/06AlgoData/02MoE/10MOELLM.pdf), [视频](https://www.bilibili.com/video/BV1UERNYqEwU/)         |
| 手撕 MOE 代码 | 11 单机单卡 MoE         | [PPT](https://github.com/Infrasys-AI/AIInfra/blob/main/06AlgoData/02MoE/11MOECode.pdf), [视频](https://www.bilibili.com/video/BV1UTRYYUE5o)         |
| 手撕 MOE 代码 | 12 单机多卡 MoE         | [PPT](https://github.com/Infrasys-AI/AIInfra/blob/main/06AlgoData/02MoE/11MOECode.pdf), [视频](https://www.bilibili.com/video/BV1JaR5YSEMN)         |
| 手撕 MOE 代码 | 13 MoE 性能分析         | PPT, 视频                                                                                                                                           |
| 视觉 MoE    | 14 视觉 MoE 模型        | [PPT](https://github.com/Infrasys-AI/AIInfra/blob/main/06AlgoData/02MoE/12MOEFuture.pdf), [视频](https://www.bilibili.com/video/BV1JNQVYBEq7)       |
