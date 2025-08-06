---
aliases:
  - 随机 HV 生成
---
仅考虑 $\{-1, 1\}^N$ 的向量表示，MAP 的操作

# 随机 HV 生成

Item Memory (IM) 存储了随机生成的 HV

后来提出，可以用 CA 来避免这个存储开销。

## 不同类别的 HV

Hyperdimensional Biosignal Processing: A Case Study for EMG-based Hand Gesture Recognition

提出一种连续级别的随机向量产生方法 CiM

cirlar hv

# 编码表示
## 输入数据的转换

A Programmable Hyper-Dimensional Processor Architecture for Human-Centric IoT

主要是两种：n-gram Sequence Encoding 和 Feature Superposition 相当于结合了时/空编码
对 permutation 可以优化
> [!tip]

## 操作
- MAP
- 性能 - accuracy, precision, recall
对序列预测问题效果好，而且存储、操作开销小很多
对任意预测问题，结合 HDC ML/DL 或许更好

- 开销 - training time, inference time, parameter size

# 具体应用

A Programmable Hyper-Dimensional Processor Architecture for Human-Centric IoT
![[img/benchmark suite.png]]

Hyper-Dimensional Computing Challenges and Opportunities for AI Applications
![[img/summary.png]]

### 目前有的数据集和实验

## 实验
最基本的逻辑推理：USA-DOl -> CHN-RMB

>[!warning]
>python 实验的 bundle 操作直接累加，输出的 HV 元素不再是 {-1, 1} 
>HW 需要考虑量化

## 数据集
ISOLET

https://archive.ics.uci.edu/dataset/54/isolet

## 2 D 图像
MNIST
训练、测试数量：6000, 1000。类别： 0-9 数字，10 种

FASHION-MNIST
训练、测试数量：6000, 1000。类别：不同的服装， 10 种

EMG-hand-gesture


Language Geometry using Random Indexing
21 European Language

<a href="http://dmb.iasi.cnr.it/supbarcodes. php">Empirical</a>
类别：8 种，包括 fungi plant kngdoms。
inga: 
simulated: 


# 数据向量化
## 图像
对一个 channel，输入数据 flatten 为一个长 $size = width * height$ 的序列，每个像素量化到 $0-255$。
构造 HV $\mathbb{X}'$
input stream of symbols: $I (x_t|t\in \mathbb{Z}_{size}),x_t\in  \mathbb{Z}_{255}$，projected to $\mathbb{X}_{val}'$
every position is projected to: $X (p_t|t\in \mathbb (Z)_{size}), p_t\in \mathbb{X}_{pos}'$
AM: 


# 实验

## MNIST/ FashionMNIST
目前的准确率只有 70%左右。

设计硬件，对 MNIST 进行测试。搭建测试环境。
问题：量化和训练。

可以有两种量化方法：
1. O 首先对一个 sample 量化，然后 train （以及可能的 retrain）之后，统一量化。
2. 或者在 train（possible retrain）之后一次量化。

For one sample has $n$ HVs, the training set has $m$ samples.
- Method 1: 



# 可能的改进
## 应用
1. 对于图像识别，HDC 还做不到 yolo 那样的快速、全局，算法和硬件协同改进，能结合多种效果，如同时定位、分类。
2. 或者，专注高效的领域 1 D

few-shot learning

## 问题

1. 对特征的表示，全局
2. 统一的 HDC 框架？HDC 对信息的表示、操作方式是多样的，可能不光是 -1, 1 的表示
3. 实际的任务，分类，如果需要所有的 AM 的需求
4. 生物、few shot learning 的应用
5. 定量的评估 HDC 对误差的容忍率，比如图片有雾；安全，基于超维向量，向量在训练、查询时是不变的，但是有些领域，
6. 硬件实时性在车、航空领域；新兴的计算范式，对比 CNN
7. 理想的全面的 HDC 加速器架构应该是什么样的，支持 HDC 和其他处理，图像和语音、DNA 序列的编码方式不同。

## HW
1. 可重构。更通用的加速器，能实现不同类型任务（DNA/image/language recognition/ExG）的 on-chip learning，需要可重构
2. 实时。对实时敏感的应用，autonomouts vehicle，无法容忍网络延时；security and privacy；网络运输的额外功耗、资源开销。
3. 能效。对 HV 数据表示的存储压缩，使得有更高计算效率


原理上，HDC 和 AI 的本质，探索新的模式

应用上，HDC

目标：

可重构/编程：

	需要更少的训练数据，one-shot learning

	重训练：适应个体差异、场景差异；对性能的提升是明显的

低功耗、实时：边缘场景、实时性要求高

## 目标
- 应用级别的可重构，以满足现实多样的需求，1D 和 2D 不同的编码方式；易于使用，不需要额外编译或 pragma，只需要调用扩展指令。
- 用于边缘计算的 few-shot learning，降低功耗，减少延时。

## 现有工作

Fully Learnable Hyperdimensional Computing  Framework With Ultratiny Accelerator for  Edge-Side Applications 