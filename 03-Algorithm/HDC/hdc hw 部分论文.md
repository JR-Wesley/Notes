# A Highly Energy-Efficient Hyperdimensional  Computing Processor for Biosignal Classification
## 总结

应用了已经提出的方法对 HDC 硬件的设计进行优化，包括 CA Vector folding，并且评估了不同参数对性能、功耗的影响。
应用：生物信号分类。

CA 实时生成超向量，替代了传统的存储内存，提高能效。
在最大吞吐量下，架构实现了10.7倍的改进，每个通道33.5倍。
还设计了一个优化的 SVM 作为对比，HDC在能量效率上比SVM高出9.5倍。

# A Programmable Hyper-Dimensional Processor  Architecture for Human-Centric IoT

可编程的 HDC 处理器
对 HDC 的操作进行了总结，本文采用 MAP

基准：用了多个例子，建立了 HDC 和传统 ML 算法的对比，在 CPU / GPU 上

- 结构：有一定扩展性
分析数据流：
两种核心模式： n-gram; feature superposition
主要成分： IM/Encode/ AM
根据数据流，设计了基本单元，然后构成层次

其他方法：
特征降维（feature reduction）或特征选择（feature selection）


# A Robust and Energy-Efficient Classifier Using  Brain-Inspired Hyperdimensional Computing
HDC 展示得与脑行为一致。它式高维、全息、伪随机的向量。高能效的计算容忍硬件故障。

方法：基于超维分类的硬件。超高准确率，略低于传统机器学习方法，只需一半的能量。此外可以容忍 8.8 倍的存储单元错误，保持高准确率。
应用：21种欧洲语言数据集
用于高维分类器，包括编码和相似性搜索两个主要模块。
总体算法没介绍

# Accelerating Hyperdimensional Computing on  FPGAs by Exploiting Computational Reuse


# An Edge AI Accelerator Design Based on HDC  Model for Real-time EEG-based Emotion  Recognition System with RISC-V FPGA Platform

基于 HDC 用 FPGA RISCV 平台，设计的实时表情识别系统

- 算法集合了 BPF STFT HDC 等方法，完成 EEG 信号检测，有算法测试和硬件设计
- HDC AI加速器包括顶层控制器、数据存储、超向量映射、空间编码器、时间编码器、联想记忆和汉明距离计算。

其算法框架来源 A. Menon et al., ”A Highly Energy-Efficient Hyperdimensional Computing Processor for Biosignal Classification,”

数据集：
SEEDKMU
LOSOV 验证方法



# EcoFlex-HDP: High-Speed and Low-Power and  Programmable Hyperdimensional-Computing  Platform with CPU Co-processing

提到了之后会做 CGRA 平台


# Hardware Optimizations of Dense Binary Hyperdimensional Computing: Rematerialization of Hypervectors, Binarized Bundling, and Combinational Associative Memory


# Revisiting HyperDimensional Learning for FPGA and Low-Power Architectures


# EcoFlex-HDP: High-Speed and Low-Power and  Programmable Hyperdimensional-Computing  Platform with CPU Co-processing

本文有开源：<a href="https://github.com/yuya-isaka/EcoFlexHDP">EcoFlexHDP</a>

## 项目框架
![](EcoFlex-HDP%20overview.png)

![](EcoFlex-HDP%20HD%20core.png)

