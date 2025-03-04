---
dateCreated: 2024-08-22
dateModified: 2025-03-04
---

https://scholar.google.com/citations?user=5piRzloAAAAJ&hl=en

Automatic Generation of Complete Polynomial InterpolationDesign Space for Hardware Architectures

用标准的硬件结构（含 LUT）生成不同精度的初等函数

Multiplier Optimization via E-Graph Rewriting

乘法器优化为基础运算，可减少 latency。

对于公司的调研任务，我记得 20 号在线下公司讨论的时候，他们想知道 CPLD 这块除了 ABC, 是否还有其他的优化工具。我这里稍微补充下，调研的同学可以参考 CirKit, iMap, openROAD 等，看他们在工艺映射这块是否都用了 ABC。早期的二级优化工具是 Espresso, 它被嵌入到 ABC 的前身，SiS 中去。新版 ABC 中没有 espresso 命令了的。SiS 1.4 的代码在这里：https://ultraespresso.di.univr.it/sis。有 4 位加法器的例子（https://ultraespresso.di.univr.it/full-adder.html）。那里还有一些比较新的文档，论文，以及 2019 年的 DATE2019 培训等可以参考（https://ultraespresso.di.univr.it/workshops/workshops）。这是 2016 年 SiS 的 PDF，https://www.corsi.univr.it/documenti/OccorrenzaIns/matdid/matdid882703.pdf。关于“与或”逻辑优化的论文，2022 年的 IWLS 还有日本的论文，Two-Level Minimization for Partially Defined Logic Functions，https://www.iwls.org/iwls2022/program.php，在 Youtube 上可以看报告。还有 IWLS2021 也有“与或”二级优化的论文，Two-Level Approximate Logic Synthesis, IWLS2023 的论文，Technology Mapping Using Multi-output Library Cells, 可以对加法器映射优化。

# 分组工作

1 2 组

https://www.sciencedirect.com/science/article/pii/S1383762124001802?via%3Dihub#bib1

1 组：

南京大学李丽

# Coarse-grained Reconfigurable Architectures for Radio Baseband Processing: A Survey
## Abstract

Digital baseband processing demands a flexible system

- the key characteristics of hardware platforms for baseband processing
- CGRA is a domain-specific accelerator in baseband processing applications

# 1 Introduction

![[Fig1.png]]

### 1.1 Related Surveys

either provide a general overview/classification of architectures or adopt a domain-specific approach.

![[Tab1.png]]

# 2 Background

The hardware accelerator need to be carefully designed to meet required performance/flexibility/energy metrics.

![[Fig2.png]]

# 3 CGRA Overview

![[Fig3.png]]

# 4 Classification of Hardware Architectures
- ASIC
- Programmable processor
- GPP
- Special purpose processor: DSP, GPU, ASIP
- reconfigurable architectures: data-driven execution style

![[tab2.png]]

### 4.6 CGRA as Domain-specific Accelerator

DSAs in the past have been used for various tasks, however, they

become obsolete very quickly given their non-compatibility to the advancements

in their target domains.

at the core of these updates remain

the same computational kernels with some parametric changes that can

be achieved by incorporating reconfigurability

# 5 Overview of Baseband Functions

baseline reference - OFDM

NC-OFDM receiver: antenna/ radio/intermediate frequency HW, ADDA, digital front end(transform IQ)

![[Fig5.png]]

![[Tab3.png]]

## 1. Time Synchronization

## 2. Frequency Synchronization

## 3. IFFT/FFT

## 4. Channel Estimation Andquealization

## 5. Forward Error Correction

## 6 MIMO Detection

## 7 Accelerating Baseband Processing Blocks with CGRAs

# 6 Systematic Review of CGRAs with Supported Baseband Algorithms

review various relevant works and highlight architectural paradigms for CGRA that can be used in exploring CGRAs for emerging baseband algorithms

Architectural features: PE array size, supported bit-width, operating frequency, power consumption, area, technology.

3 categories: broad-scoped, MIMO, low-power

## Broad-scoped

![[Tab4.png]]

Many are based on highly parameterizable templates

Some adopt a clustered approach toward the PE arrangement within the array -> overall reduction of interconnect wires

Some use techniques to make the configuration more efficient

## MIMO Ps

![[Tab5.png]]

1. 1MIMO supporting CGRA systems are large and consume high power due to the computational-intensive nature
2. diverse bit width computations are executed
3. various data access patterns are utilized during the execution.
4. To improve the computational performance, different architecture many be incorporated

## CGRA for Low Power Application

![[Tab6.png]]

## Analysis and Summarizing Design Parameter Considerations
1. heterogeneity is incorporated including, CORDIC, MAC, square-root, floating-point, quality-scalable PEs
2. PEs are usually arranged in a matrix form. The number of PEs in an array is kept low but deploying multiple arrays to achieve high performance.
3. Different bit-width, or multi-granular support, integer and FP, bitwise operations
4. nearest-neighbor and 2D mesh have been the most commonly adopted styles
5. memory access
6. loosely and tightly coupled paradigms
7. technology/ operating frequency/

![[Fig7.png]]

# 7. Challenges and Potential Research Directions
## Compilers

scalable mapper

highlight the run-time overhead

## Architecture Design

trade-offs

PE heterogeneity: homogeneous PE system offers simple SW and HW design but the flexibility comes at the cost of increased configuration information. The design space search needs to be automated.

interconnect/ memory system/elasticity

## Evaluation

Lack of mature unified frameworks

Baseband workload benchmarks are not completely realistic

![[Tab7.png]]

# 8. Future Directions

![[Tab8.png]]

### 8.1. Irregular Functions

### 8.2. Approximate Computing

### 8.3. Multi-role CGRA

### 8.4. Deep-learning-based Receivers

The functional blocks are implemented on certain assumptions that may not entirely represent the real environment. CGRA-related work target DL for baseband processing algorithm is in infancy

### 8.5. Speculating the Future Radios - a Conceptual Framework

RISC-V + CGRA

![[Fig8.png]]

# 9. Conclusion

# Reference

141-151 CGRA 加速基带信号处理的各种处理单元
