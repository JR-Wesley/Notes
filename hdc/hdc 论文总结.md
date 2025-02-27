---
tags:
  - summary
dateCreated: 2024-10-12
dateModified: 2024-11-22
---

[IEEE Xplore Search Results](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Index%20Terms%22:Hyperdimensional%20Computing&matchBoolean=true)

包含 HDC一个网站： https://www.hd-computing.com/

[(43 封私信 / 81 条消息) 什么是超维计算，该计算框架存在什么优点，它能革新现有的人工智能方法吗？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/603376219)

一些超维计算的代码例子： https://github.com/HyperdimensionalComputing/collection

<a href=" https://wh-xu.github.io/"> HDC 领域的一个作者</a>。其中的 FSL-HDnn 结合了 CNN 和 HDC 的硬件架构。

## 未来发展

(1) exploit HDC intrinsic characteristics for more classifica844 tion/cognitive tasks in different domains like security, image 845 processing, and real-time applications. (2) focus on develop846 ing an efficient encoding algorithm that handles HDC capac847 ity limitation and would improve data representation for 2D 848 applications. (3) develop more hardware friendly metrics for 849 similarity measurement that would enhance system accuracy. 850 (4) design a unified HD processor that addresses diverse data 851 types and can trade-offs accuracy to efficiency based on the 852 application requirements. (5) investigate the dimension of HD 853 vector that store and manipulate different data representations 854 to reduce the memory footprint and enhance system effi855 ciency. (6) study various methods for integrating the DL/ML 856 techniques with HDC and analyzing system performance 857 effects.

# 重要的作者、工作

Alexandru Nicolau https://ieeexplore.ieee.org/author/37271711300

torchhd 和其他

加速 permute n-gram 编码操作 Accelerating Permute and N-Gram Operations for Hyperdimensional Learning in Embedded Systems

Rahim Rahimi

https://ieeexplore.ieee.org/author/37076618700

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10181701&tag=1

Jan M. Rabaey, University of California Berkeley, USA

https://ieeexplore.ieee.org/author/37276611300

Abbas Rahimi, Johannes Kepler University, Lines, Austria

https://ieeexplore.ieee.org/author/37071093800

Hani Saleh

https://ieeexplore.ieee.org/author/37391220900

有一些硬件优化、加速的工作

Tajana Rosing, University of California San Diego La Jolla, CA, USA

https://ieeexplore.ieee.org/author/37295875800

Weiqiang Liu, Nanjing University of Aeronautics and Astronautics

https://ieeexplore.ieee.org/author/37085338613

## 实验总结

https://github.com/search?q=hyperdimensional+computing&type=repositories

collection of HDC

https://github.com/HyperdimensionalComputing/collection

torchhd 库 uci. edu 组的工具

[abbas-rahimi/HDC-Language-Recognition](https://github.com/abbas-rahimi/HDC-Language-Recognition)

[mmejri3/er-hdc](https://github.com/mmejri3/er-hdc)

[wh-xu/hdc-workload](https://github.com/wh-xu/hdc-workload)

[Orienfish/LifeHD](https://github.com/Orienfish/LifeHD)

[ComeOnGetMe/_hyperdimensional_-_computing_](https://github.com/ComeOnGetMe/hyperdimensional-computing)

# Exploring Embedding Methods in Binary Hyperdimensional Computing: A Case Study for Motor-Imagery Based Brain–Computer Interfaces

https://github.com/MHersche/HDembedding-BCI

# Binary Models for Motor-Imagery Brain-Computer Interfaces: Sparse Random Projection and Binarized SVM

https://ieeexplore.ieee.org/abstract/document/9073968

https://github.com/hanyax?tab=repositories

# 论文总结

> A Survey on Hyperdimensional Computing aka Vector Symbolic Architectures, Part I: Models and Data Transformations

两部分的综述。这篇系统地讲述超维计算和向量符号体系结构 (HDC/VSA)，第一部分讲已有*计算模型*和*输入到高维分布式表示的转换*。第二部分致力于应用、认知计算和体系结构，以及未来工作。

> P. Kanerva. 2009. Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors. Cognitive Computation 1, 2 (2009), 139–159.

> P. Kanerva. 2019. Computing with high-dimensional vectors. IEEE Design & Test 36, 3 (2019), 7–14.

---


# 1. 综述、介绍原理、讲解应用场景

# Hyperdimensional Computing: An Algebra for Computing with Vectors


# 超维计算概念、应用及研究进展

<a href=" https://www.chinaai.org.cn/newsinfo/7131133.html">中文网站的一篇总结</a>

# A Comparison of Vector Symbolic Architectures

VSA 用超长原子向量，远超需要表示的符号数量。其额外的空间引入的冗余。

单比特的错误不影响整体性质；随机的高维向量近似正交；通过向量的距离来判断相关性。

其上的操作也保留了这种相似性。

VSA 结合了矢量空间和对应操作，但是其具体实现有太多。

### VSA 的分类

VSA 根据其底层向量空间和操作的实现方式被分为不同的类别。文件 2 中提到了 11 种不同的 VSA 实现，它们在向量空间和算子的具体实现上有所差异。以下是一些主要的 VSA 实现：

1. **Multiply-Add-Permute (MAP)**：有三种变体，分别是 MAP-C（连续空间）、MAP-B（二进制空间）和 MAP-I（整数空间）。
2. **Binary Spatter Code (BSC)**：使用二进制向量空间。
3. **Binary Sparse Distributed Representation (BSDC)**：有三种变体，BSDC-CDT、BSDC-S 和 BSDC-SEG。
4. **Holographic Reduced Representations (HRR)**：使用实数向量空间。
5. **Frequency Holographic Reduced Representations (FHRR)**：使用复数向量空间。
6. **Vector Derived Binding (VTB)**：基于 HRR，但使用不同的绑定和解绑操作。
7. **Matrix Binding of Additive Terms (MBAT)**：使用矩阵乘法进行绑定。

### 相似性测量方法

1. **余弦相似度（Cosine Similarity）**：
    - 用于实数向量空间（如 MAP-C、HRR、MBAT 和 VTB）的相似性度量。余弦相似度是通过计算两个向量的点积和它们模的乘积的比值来确定的，其结果是一个介于 -1 到 1 之间的标量值，其中 1 表示向量完全相同的方向，-1 表示完全相反的方向，0 表示向量正交。
2. **汉明距离（Hamming Distance）**：
    - 用于二进制密集向量空间（如 BSC）。汉明距离测量两个等长字符串或向量在相同位置上不同元素的数量。
3. **重叠度量（Overlap Measure）**：
    - 用于二进制稀疏向量空间（如 BSDC-CDT、BSDC-S 和 BSDC-SEG）。重叠度量通常是指两个向量中同时为 1 的元素数量，有时可以被归一化到 [0, 1] 范围内，其中 0 表示完全不相似，1 表示完全相似。
4. **角度距离（Angle Distance）**：
    - 用于复数向量空间（如 FHRR）。角度距离是通过计算两个复数向量之间角度的平均差异来确定的。由于复数向量具有单位长度，因此可以直接计算角度的差异。

### 绑定操作的分类

绑定操作是 VSA 中用于连接两个向量的关键操作，其输出再次是同一向量空间中的向量。文件 2 中提出了一个绑定操作的分类体系，将现有的绑定操作分为以下几类：

1. **准正交绑定（Quasi-orthogonal Bindings）**：
    - **自反绑定（Self-inverse Bindings）**：绑定操作本身是其逆操作，例如 BSC 中的 XOR 操作。
    - **非自反绑定（Non self-inverse Bindings）**：需要额外的解绑操作，例如 VTB 中的矩阵乘法。
2. **非准正交绑定（Non-quasi-orthogonal Bindings）**：
    - **BSDC-CDT**：使用逻辑 OR 操作进行绑定，需要上下文依赖的稀疏化（CDT）过程。
3. **基于角度的绑定（Angle-based Bindings）**：
    - **FHRR**：在复数向量空间中，通过角度的加法和减法进行绑定和解绑。
4. **基于矩阵的绑定（Matrix-based Bindings）**：
    - **MBAT**和**VTB**：使用矩阵乘法进行绑定和解绑。
5. **基于位移的绑定（Shift-based Bindings）**：
    - **BSDC-S**和**BSDC-SEG**：通过向量的位移进行绑定。
6. **基于卷积的绑定（Convolution-based Bindings）**：
    - **HRR**：使用循环卷积进行绑定，需要特定的解绑操作（循环相关）。

这个分类体系不仅帮助我们理解不同 VSA 实现之间的差异，还揭示了绑定操作的数学属性，如交换性（Commutative）和结合性（Associative）

## Abstract

Vector Symbolic Architectures combine a high-dimensional vector space with a set of carefully designed operators in order to perform symbolic computations with large numerical vectors.

The available implementations differ in the **underlying vector space and the particular implementations of the VSA operators.**

## Intro

Symbols are encoded in **very large atomic vectors**, much larger than would be required to just distinguish the symbols. VSAs use **the additional space** to introduce redundancy in the representations, usually combined with distributing information across many dimensions of the vector

The operations in VSAs are mathematical operations that create, process and preserve the graded similarity of the representations in a systematic and useful way.

As stated initially, a VSA combines a vector space with a set of operations. Which VSA is the best choice for the task at hand?

# An Introduction to Hyperdimensional Computing for Robotics
## 总结

讲解了超维计算的基本原理，总结了用于机器人方面的应用。

高容量

近邻查询的不稳定性

随机向量近似正交

随机向量之间的噪声对最近邻查询影响较小

1. **采样密度的降低**：在高维空间中，如果固定数据点的数量不变，随着维度的增加，这些数据点之间的采样密度会降低。这意味着在高维空间中，数据点之间的“距离”变得更大，使得确定最近邻变得更加困难。
2. **距离的相对性**：在低维空间中，我们可以通过直观感受来判断两个点之间的距离。然而，在高维空间中，由于数据点之间的距离变大，任意两个点之间的距离都变得相对接近，这使得区分最近邻和次近邻变得不那么明显。
3. **最近邻查询的不稳定性**：由于高维空间中数据点之间的距离变大，对于给定的查询点，其到最近邻点的距离可能与到其他非最近邻点的距离相差无几。在这种情况下，即使是很小的噪声也可能导致最近邻查询结果的变化，从而使最近邻查询变得不稳定。
4. **查询的不确定性**：在高维空间中，由于数据点之间的距离相对接近，对于一个给定的查询点，很难确定哪个数据点是真正的最近邻。这种不确定性使得最近邻查询在高维空间中变得不那么有意义。
5. **维度的指数增长**：随着维度的增加，空间的体积呈指数增长，而数据点的数量通常是有限的。这意味着在高维空间中，数据点之间的间隔变得更大，导致最近邻查询的准确性下降。

# A Survey on Hyperdimensional Computing Aka Vector Symbolic Architectures, Part I: Models and Data Transformations

关于这篇文章的解读： https://cloud.tencent.com/developer/article/2321030

## Abstract
- Definition: **Hyperdimensional Computing and Vector Symbolic Architectures (HDC/VSA)** use **high-dimensional distributed representations** and rely on the algebraic properties of their key operations to incorporate the advantages of **structured symbolic representations and distributed vector representations**.
- Notable models: Tensor Product Representations, Holographic Reduced Representations, Multiply-Add-Permute, Binary Spatter Codes, and Sparse Binary Distributed Representations

## Intro

The two main approaches to AI is **symbolic and connectionist**.
- Symbolic AI solves problems or infers new knowledge through **the processing of these symbols**.
- In the connectionist approach, information is processed in **a network of simple computational units** often called neurons (ANN, artificial neural networks).

HDC/VSA is the umbrella term for a family of computational models that rely on mathematical properties of high-dimensional vector spaces and use high-dimensional distributed representations called hypervectors (HVs) for structured (“symbolic”) representation of data while maintaining the advantages of connectionist distributed vector representations.

# A Survey on Hyperdimensional Computing Aka Vector Symbolic Architectures, Part II: Applications, Cognitive Models, and Challenges

## Open Issue
1. HDC/VSA 源于将人工智能的符号方法的优势（例如组合性和系统性）与人工智能的神经网络方法（连接主义）的优势（例如基于载体的表示、学习和基础）相结合的提议。
要结合 HDC/VSA 和 神经符号 / neural-symbolic computing 需要形成一个接口。
So far, a major advantage of the HDC/VSA models has been their ability to use HVs in a single unified format to represent data of varied types and modalities.
the representation of the input data to be transformed into HVs should be able to specify the compositionality explicitly.

2. 目前的 HDC/VSA 重实现了 AL/ML 用 HV，以一种更适合非传统的计算硬件。
在解决简单分类或相似性搜索问题能比上 SOTA，且更低 energy/computational costs，在“微型机器学习”（Tiny Machine Learning）和“边缘机器学习”（Edge Machine Learning）领域具有潜在应用价值。
目前有把 HV 生成用 NN 替代，用神经网络生成的 HVs 来解决下游任务，但是局限于特定的任务和应用场景，缺乏广泛的通用性。

3. 结合 NN 和 HDC 变得更通用的工作也比较少。
目前的 general AI require a much higher level of generalization

提出了很多问题

# Computing with High-Dimensional Vectors
## 总结

计算从 von Neumann 架构的精准计算（耗能），到人脑启发的 ANN 计算（计算密集，不够灵活），引出超维计算。超维计算用 n 维 {-1, +1} 向量表示，结合操作，赋予了计算能力。健壮性，高维空间的：需要用 associative memory 来识别相似的事物。
然后利用公式推导了 MAP 操作表示的意义。理论上分析了其性质。
## Computing at Large

The genius of the **von Neumann architecture** is that the programs reside in memory and are manipulated with the same operations as the data—to the machine, the programs are data.so computers are designed to work as deterministic machines: identical inputs always produce identical outputs -> numerical analysis

However, reliability at high speeds comes at a cost: it requires large amounts of energy. our brains make it possible for us to understand and react to what is happening around us.The operations refer to vectors even as the vectors are realized with bits and numbers - addition/ multiplication/ permutation of coordinates

## High-dimensional Computing: An Example

identifying languages from their letter-use statistics (not relying on dictionaries). For each language, we compute a high-dimensional profile vector from about a million bytes of text. the letters of the alphabets are assigned **n-dimensional random vectors of +1s and −1s**. The letter vectors are used to make trigram vectors with permutation and multiplication.

## High-dimensional Vectors and Robustness

**concentration of measure**
it is essential for the brain to be able to identify similar sensory stimuli as the same
a system must be able to **identify new inputs with things already stored in the system’s memory**. The function is referred to as **associative memory**, which, in essence, is the nearest-neighbor search among vectors stored in the memory.

## Vector Operations and Their Algebra

**holographic or holistic** , any part of a vector represents the same thing as the entire vector, only less reliably.
What exactly those operations are, depends on the nature of the vectors, whether binary, “bipolar,” integer, real, or complex.

**Similarity of vectors is based on distance.** Dot product A • B Cosine and (Pearson’s) correlation

**Addition** The sum vector is similar to the argument vectors and independent of their order: A + B + C ~ A, B, C. It can therefore be used to represent a set or a multiset.
**Multiplication** multiplication preserves similarity: (X * A) • (X * B) = A • B
**Permutations** provide a means to represent sequences and nested structure.
sequence (a, b, c)

$$
\begin{align}
S3 &= ρ (ρ(A)) + ρ(B) + C  \\
&= ρ2(A) + ρ(B) + C\\
P3 &= ρ2 (A) * ρ(B) * C
\end{align}
$$

(a, b) ρ1 and ρ2 as ρ1(A) + ρ2(B),

The versatility of computing with numbers is partly due to the fact that addition and multiplication form an algebraic structure called a field.

## Second Look at Language Identification: Working out the Math

# Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors

## Hyperdimensional Computer



# What We Mean When We Say “What’s the Dollar of Mexico?”: Prototypes and Mapping in Concept Space
## 总结

前半部分讲解计算机、人脑、认知等方面的联系，从本质上解读了计算的内涵，讲解非常详尽但语言晦涩。

然后推出了高维向量的概念、存储、操作、组合等概念。

讲解了高维向量的表示是如何反应真实的语言，对语言的底层原理也做了解读，以货币和国家为例。

	整体逻辑推导非常基础

## 解读

比喻的语言要求概念空间的映射。

高位空间的映射可以反应概念空间的性质。只用加法、乘法就可以表示映射。

高维空间的性质和认知等想象对应。

正是超高的维度，近似的结果，保证了简单的操作就可以这个映射。

打包和解包操作被视为空间点（概念的代表）之间的映射，这暗示了一种类比机制，类比映射是从示例中计算出来的。

大脑的计算能力、组织方式远比计算机灵活、特别，即使计算机技术提升无数倍。神经科学的发现给了我们启发，要创造模型 能产生神经元行为并且物理可实现。

高级神经系统的一个显著特征是其回路的大小 —— 数量极大，可能说明，大脑的计算不能用小单元分析、不能分割。因此本文继续用高维的向量表示来构建计算的基本单元。对高维向量，用距离作为相似度，

基本单元：单词/模式/向量/点是意义最小的单位。向量的任何部分与整个向量具有相同的意义，只是意义以较低的分辨率表示。这种表示是高度冗余的，与避免冗余的传统表示形成了鲜明对比。这种表现被恰当地称为全息或整体。

存储：联想神经网络：当单词 D 与单词 A 一起存储时，D 可以通过用 A 或类似于 A 的“带噪声”版本对记忆进行寻址来检索。地址 A 也称为记忆线索。

操作：bind/ bund 分别生成不相似和相似的结果

组合：变量和值的显式表示，这个操作使得我们可以通过复合载体分析成其成分

全息矢量的映射保存了距离

## Abstract

We assume that the brain is some kind of a **computer** and look at operations implied by the **figurative use of language**.

*a mapping in concept space -> the nature of concept space in terms of readily computable mappings*

mappings of the appropriate kind are possible in **high-dimensional**

high-dimensional spaces have been shown elsewhere to **correspond to cognitive phenomena such as memory recall.**

## The Brain as a Computer

There is more to the brain’s computing than raw processing power, something **fundamentally different in how the computing is organized.**

of finding models that are capable of producing the behavior and also **suited for building into neural-like physical systems**

## Language as a Window into the Brain’s Computing

A prominent feature of advanced nervous systems is the **size** of their circuits.

Another possible conclusion is that the brain’s computing cannot be analyzed at all in terms of smaller units; it cannot be compartmentalized.

uses very large patterns, or h**igh-dimensional vectorsdimensionality in the tens of thousands**—as basic units on which the computer operates

A word/pattern/vector/point is the least unit with meaning.

vector components take part in the meaning but in themselves are meaningless or, any subset of components—has the same meaning as the entire vector,

## Memory for Wide Words
**when the word D is stored with the word A as the address, D can later be retrieved by addressing the memory with A or with a “noisy” version of it that is similar to A. The address A is also called a memory cue.**

## Basic Operations for Wide Words

Basic operations and their use for computing is the essence of this paper.

**binding** : U ∗ V dissimilar (orthogonal) to the two. for binary, pairwise Exclusive-Or (XOR, addition modulo 2)

**bundling** [U + V + … + W ] maximally similar to them.

## Composing with Wide Words

**A data record** consists of **a set of variables** (attributes, roles) and their **values** (fillers)
The **variables** of a traditional record are implicit—they are implied by their **locations** in the record, called fields—whereas the **values** are **encoded** explicitly in their fields.
holistic encoding: **variable–value pair $x = a$ is encoded by the vector $X ∗ A$**
entire record:

$$
H = [(X ∗ A) + (Y ∗ B) + (Z ∗ C)]
$$

variables and the values explicitly are included, span the entire 10,000-bit vector

The operations have two very important properties:

1. binding is invertible (XOR is its own invese
2. binding (and its inverse) distributes over bundling:
These properties make it possible to analyze a composite vector into its constituents

$$
\begin{align}
X ∗ H &= X ∗ [(X ∗ A) + (Y ∗ B) + (Z ∗ C)] \\
&= [(X ∗ X ∗ A) + (X ∗ Y ∗ B) + (X ∗ Z ∗ C)] \\
&= [A + R1 + R2] \\
&= A′ ≈A
\end{align}
$$

assume that the variables and **the values are represented by approximately orthogonal** vectors as they would be if chosen independently at random

R1 and R2 are approximately orthogonal to each other and to the rest and so they act as random noise added to A.

## Holistic Vectors as Mappings

geometric property of interest is that the mapping preserves distance

$$
d(X ∗ A, X ∗ A′) = d(A, A′)
$$

When the variable X is bound to the value A by X ∗ A, X maps A to another part of the space.

Thus, if a configuration of points—their **distances** from each other—represents **relations** between their respective objects, **binding** them with X moves the configuration “bodily” to **a different part of the space**.

XORing with X serves as a mapping in which the relations are maintained.

## Mapping Between Analogical Structures

the holistic vectors for the United States and Mexico would be encoded by

If we now pair USTATES with MEXICO, we get a bundle that pairs USA with Mexico, Washington DC with Mexico City, and dollar with peso, plus noise:

what in Mexico corresponds to our dollar. We get:

$$
\begin{align}
USTATES &= [(NAM ∗ USA) + (CAP ∗ WDC) + (MON ∗ DOL)]  \\
MEXICO &= [(NAM ∗ MEX) + (CAP ∗ MXC) + (MON ∗ PES)]\\
F_{UM} &= USTATES ∗ MEXICO \\
&= [(USA ∗ MEX) + (WDC ∗ MXC)  +(DOL ∗ PES) + noise]\\
DOL ∗ FUM &= DOL ∗ [(USA ∗ MEX) + (WDC ∗ MXC)  +(DOL ∗ PES) + noise]  \\
&= [(DOL ∗ USA ∗ MEX)  +(DOL ∗ WDC ∗ MXC)  +(DOL ∗ DOL ∗ PES) + (DOL ∗ noise)]  \\
&= [noise1 + noise2 + PES + noise3]  \\
&= [PES + noise4]  \\
&≈ PES
\end{align}
$$

## From Variables to Prototypes

idea that different parts of a concept space can contain similar structures, and that traversing between the structures is by relatively straight-forward mapping.

Language operates with labels that are largely arbitrary, and with structures built of labels, yet refer to and evoke in us images and experiences of real things in the world. This requires e**fficient mapping between superficial language and the inner life,** a mapping that is not found in animals at large.

# An Extension to Basis-Hypervectors for Learning from Circular Data in Hyperdimensional Computing

关注 Basis-Hypervector、输入信息编码方式的讨论。研究了三种不同的基本向量表示，表明 level/cirlar 对性能有提升。

Intro 提到了很多不同的应用，


# Torchhd: An Open Source Python Library to Support Research on Hyperdimensional Computing and Vector Symbolic Architectures

University of California, Irvine

开发了一套易用的工具库，对 HDC 计算优化的工具

超维计算的核心思想时用随机生成的向量表示信息。结合一组操作，HD 可以表示组合的结构。

在算法的执行上优化了较大的加速效果

和 OpenHD HDTorch VSA Toolbox 进行了对比，提供了更多类型的 HDC 的支持

三个对比试验：

EU languages (Rahimi et al., 2016b) Wortschatz Corpora Europarl Parallel Corpus

EMG gestures (Rahimi et al., 2016a) Analysis of Robust Implementation of an EMG Pattern Recognition based Control 自己搭建的数据集？

VoiceHD (Imani et al., 2017)

The central idea is to represent information with randomly generated vectors, called hypervectors.

**[openhd](https://github.com/UCSD-SEELab/openhd)**
对 GPU 进行加速，用 C++ 实现
HDTorch
基于 Pytorch 的优化框架

# 3 HDC 应用 SW

# Hyperdimensional Biosignal Processing: A Case Study for EMG-based Hand Gesture Recognition
## 总结
- **背景：HDC 通过高维、全息、伪随机分布的向量进行计算**
- **方法：提出了时空编码，使得 HDC 实现高准确率，只用了 1/3 的训练数据。**
- **结论：HDC 有快速和准确的学习，具体的性能会受 N / 训练数据的影响**

超维向量表示信息，操作方式允许直接的认知式的操作。

特点：低能量。

- 普遍且完整的
- HDC 中的学习与执行共享其结构，相对轻量级
- 稳健，容忍信号中的噪音，从而导致超低能量计算
- 以内存为中心，通过在内存内操纵和比较大型模式; 操作要么是本地的，要么可以以分布式方式执行

优势：

- 平均准确率高 2%，加入了 temporal 提升更多 8%
- 保持准确性，当数据有重叠，维度下降
- 学习快，数据需求少

使用 MAP 操作 和 cos 相似度

编码 spatial：

- 不同通道随机生成的 4 组正交编码
- 同一时间下，21 个离散的连续的向量。
- 每个时间下的 record 为 4 个通道的组合
- label 为所有的累加，存在 AM，然后通过相似度匹配

编码 tempo-spatio

连续利用 N-gram 的向量共同生成一个判断向量。

为了决定合适的 N，引入了反馈机制，动态地调整

实验：

调整 N 判断对准确性的影响：需要选取一个合适的 N

学习率：SVM 需要更多的训练数据，HDC 学习更快，当然准确率也有上限

HDC 硬件友好，能效、健壮性更优。

肌肉收缩由称为运动神经元的神经细胞的电活动产生。它们的细胞体位于脊髓中，它们的轴突直接连接到目标肌肉。产生肌肉收缩的刺激以被称为动作势（AP）的电压从大脑皮质传播到目标肌肉。AP 是通过钠 + 和钾 + 离子沿着神经细胞膜通过产生的。由于这种离子流，神经冲动向肌肉细胞传播并开始收缩

In this formalism, information is represented in high-dimensional vectors called hypervectors.

Such hypervectors can then be mathematically manipulated to not only classify but also to bind, associate, and perform other types of cognitive operations in a straightforward manner

## III HDC

Hyperdimensional computing has been used for identifying the source language of text samples from a sequence of N consecutive letters (N -grams)

MAP operations

## IV HDC Coding

0 mV to 20 mV -> discretize the channel’s signal to 21 discrete levels

### A. Encoding Spatial Correlation into a Holistic Record
- channel: there are four fields in the record, namely, CH1, CH2, CH3, and CH4
an item memory (**iM**) that assigns a unique but random hypervector to every field: {iM(CH1) ⊥ iM(CH2) ⊥ iM(CH3) ⊥ iM(CH4)}

- level: To represent these 21 discrete levels, we use a method of mapping quantities and dates “continuously” to hypervectors
perform such mapping we consider a continuous item memory (**CiM**).
Such continuous mapping better represents the adjacent levels since their corresponding hypervectors are similar to each other.

- label: GV(Label[t]) += R[t]. range(5)
these five GVs are stored in an associative memory as the learned patterns
When testing, we call the output of the encoder a “query hypervector” since its label is unknown.

### B Spatiotemporal Correlations

For N -grams at large this becomes N -gram[t]= ∏N−1 i=0 ρiR[t−i].

GV(Label[t]) += N gram[t]

**determine the proper size of an N -gram**

### C. Adaptive Encoder Using Feedback

we design an adaptive mechanism to adjust the size of N -grams during classification

# Language Geometry Using Random Indexing

Random Indexing for identifying the language of text samples

Method: encoding letter N -grams into high-dimensional Language Vectors

Experiment: low power and space

## 总结

结果：识别特定的语言，N = 3 以上，到达了 97% 的准确率

## Intro

Application domain: recognize unknown languages.

This is because embedded within each language are certain **features that clearly distinguish one from another**, whether it be accent, rhythm, or pitch patterns.

most language models use **distributional statistics** to explain structural similarities in various specified languages. counting individual letters, letter bigrams, trigrams, tetragrams, etc., and comparing the frequency profiles of different text samples.

More data, more accuracy.

High-dimensional vector models (NLP) are used to capture word meaning from word-use statistics. The vectors are called **semantic vectors or context vectors**.

SVD

Characteristics: it is possible to calculate useful representations **in a single pass** over the dataset with very little computing.

## Random Indexing

Random Indexing represents information by projecting data onto vectors in a high-dimensional space.

MAP coding. D-dimensional space vectors for 26 letters

- Encoding
{-1, 1}, MAP, cos

## Experimental Results

dataset: Wortschatz Corpora/ Europarl Parallel Corpus

Leipzig Corpora Collection

N-gram from 1 to 5 (74.9 - 94.0 -> 97, 97, 97)

### Making and Comparing of Text Vectors

[11] random indexing: **the text vectors** of an unknown text sample is compared for similarity to precomputed text vectors of known language samples (**language vectors**)

Simple language recognition can be done by comparing **letter frequencies** of a given text to known letter frequencies of languages. (**ergodic**)

A more generalized method - letter blocks of different sizes (**N-gram**). For an alphabets of L letters, there is $(L+1)^N$ N-grams (L + 1 Space).

# SpatialHD: Spatial Transformer Fused with Hyperdimensional Computing for AI Applications

融合 Spatial Transformer Networks / HDC

# SynergicLearning: Neural Network-Based Feature Extraction for Highly-Accurate Hyperdimensional Learning

 NN - HDC 混合的方法，NN 作为特征提取，特殊训练以配合 HDC，

和 NN 准确率近似，比 HDC 高 10%，

也有端到端的硬件实现和编译器

能效提升 1.6 比 HD，延迟提升 2.13

