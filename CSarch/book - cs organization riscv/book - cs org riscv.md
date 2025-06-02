---
dateCreated: 2024-09-23
dateModified: 2025-04-12
---
# Overview

CS organization RISCV 版本

DDCA 书的重点：

ch 6

有对高级代码的直接 RISCV 汇编代码表示；对不同调用的表示，内存变化图示；对不同类型机器码的解释，以方便后续 RTL 设计。最后讲解了程序的表示；RISC 指令架构的其他要点、不同变种

ch 7

从头开始设计微架构，包括了单周期、状态机、多周期的设计。最后讲解了高级设计方法

ch8 memory

其他资料：

[前言 :: RISC-V CPU设计实验教程 (fpgaxlab.github.io)](https://fpgaxlab.github.io/jurv-open/site/jurv/v2.0/index.html)

# Cs Org RISCV

# 1 Computer Abstractions and Technology



# 2 The Processor

# 3 Archimetric for Computers

## 3.4 Division

dividend / divisor = quotient - remainder

### A Division Algorithm and Hardware

The hardware mimics grammar school algorithm by iterate the shift and comparison operation.

![[../assets/csorg/Fig3.8.png]]

![[../assets/csorg/Fig3.9.png]]

The signed division makes sure that the dividend and remainder have identical signs

$$
+7 div -2 = -3 remain +1
$$

Other techniques to produce more than one bit of the quotient per step such as SRT division which tries to predict several quotient bits.

RISC-V have instructions for division and remainder: div, divu, rem, remu

> RISC-V divide ignore overflow and division by 0, so software must check the divisor and quotient
> restoring algorithm dont immediately add the divisor back if the remainder is negative; dont save the result of the subtract is nonperforming division algorithm
![[../assets/csorg/Fig3.12.png]]

## 3.5 Floating Point


### Software Optimization via Blocking (TODO)

# 6 Parallel Processors from Client to Cloud
## 6.1 Introduction

# Digital Design and Computer Architecture


## Advanced Architecture
- **Deep pipelines**
- **Micro-Operations**
- **Branch Prediction**
- **Superscalar Processor**
A superscalar processor contains multiple copies of the datapath hardware to execute multiple instructions simultaneously.
- **Out-of-Order Processor**
Out-of-order processors use a table to keep track of instructions waiting to issue. The table, sometimes called a scoreboard, contains information about the dependencies. The size of the table determines how many instructions can be considered for issue.
- **Register Renaming**
- **Multithreading**
A program running on a computer is called a **process**.
Each process consists of one or more **threads** that also run simultaneously.
The degree to which a process can be split into multiple threads that can run simultaneously defines its level of **thread-level parallelism (TLP)**.
When one thread’s turn ends, the OS saves its architectural state, loads the architectural state of the next thread, and starts executing that next thread. This procedure is called **context switching**.
A hardware multithreaded processor contains more than one copy of its architectural state so that more than one thread can be active at a time.
Switching between threads can either be fine-grained or coarse-grained. **Fine-grained multithreading** switches between threads on each instruction and must be supported by hardware multithreading. **Coarse-grained multithreading** switches out a thread only on expensive stalls, such as long memory accesses due to cache misses.
Multithreading doesn't improve the performance of an individual thread(no ILP increase) but improves the overall throughput because multiple threads can use processor resources that would have been idle when executing a single thread. Multithreading is also relatively inexpensive to implement because it replicates only the PC and register file, not the execution units and memories.
- **Multiprocessors**
1. Symmetric multiprocessors include two or more identical processors sharing a single main memory. They are easy to design and programming for.
2. Heterogeneous multiprocessors add complexity in terms of both designing the different heterogeneous elements and the additional programming effort to decide when and how to make use of the varying resources.
3. Processors in clustered multiprocessor systems each have their own local memory system instead of sharing memory.

Western Digital’s SweRV cores, the SweRVolf SoC, and the PULP (Parallel Ultra Low Power) Platform.

SiFive’s Freedom E310 core and Western Digital’s open-source SweRV
