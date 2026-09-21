---
tags: 
- CS
---

# 计算机科学：AI Infra 基础入口

本领域保存 AI Infra 所需的程序设计、算法、体系结构、操作系统、网络与编译基础，并与 `02-AISystem` 的 GPU、框架、通信和推理专题相连。学习与索引主轴是“工作负载如何变成设备和集群上的执行”，详见 [[01-ComputerScience/CS-diy|AI Infra 导向的计算机知识体系与学习路径]]；AI Systems 的具体专题和项目入口见 [[02-AISystem/AI-sys-review|AI System 知识体系与项目导航]]。索引中的材料状态不代表个人掌握程度。

## AI Infra 知识主线

| 执行层与问题 | 计算机基础入口 | AI Systems 中的后续入口 |
| --- | --- | --- |
| 模型和张量产生什么工作负载？ | [[03-Algorithm/basic-algorithm/basic-algorithm\|算法]]、线性代数、数值与概率 | [[02-AISystem/algorithm-and-model/algorithm-and-model\|模型]]、[[../02-AISystem/Acceleration/Acceleration\|推理]] |
| 框架怎样把模型变成执行任务？ | [[01-ComputerScience/Programming/Programming\|编程]]、[[01-ComputerScience/Compiler/Compiler\|编译器]] | [[02-AISystem/Framework/Framework\|框架]]、[[02-AISystem/AI-compiler/AI-compiler\|AI 编译器]] |
| 算子怎样并行并实现加速？ | [[01-ComputerScience/Architecture/Architecture\|体系结构]]、[[01-ComputerScience/system-basics/system-basics\|系统基础]] | [[02-AISystem/hpc-basics/hpc-basics\|并行计算]]、[[02-AISystem/GPU/GPU\|GPU]] |
| 主机如何管理设备与内存？ | [[01-ComputerScience/operating-system/operating-system\|OS]]、[[01-ComputerScience/SoC/SoC\|硬件系统]] | [[02-AISystem/cluster-and-hardware/cluster-and-hardware\|服务器与拓扑]] |
| 数据怎样跨设备和节点传输？ | [[01-ComputerScience/Network/Network\|网络]]、OS 的并发和 I/O | [[02-AISystem/Distributed/Distributed\|通信与分布式]] |
| 训练或推理怎样扩展和保持可靠？ | 算法分析、系统性能、文件与持久化 | [[02-AISystem/Distributed/Parallelism/Parallelism\|并行策略]]、[[../02-AISystem/Acceleration/Acceleration\|推理系统]] |

数学、正确性、性能建模、软件工程、可靠性与安全贯穿整条路径。先修关系和可验证的学习产出见 [[01-ComputerScience/CS-diy#四、按依赖推进的学习路径|阶段路径]]；本科其他知识域以 [[01-ComputerScience/CS-diy#六、本科知识域对照|覆盖对照]]保留，遇到项目直接依赖时再加深。目录仍按笔记主题组织，下表用于定位文件，不承担学习先后顺序。

## 主题地图

下面保留现有目录入口及原材料维护状态。目录按资料积累组织，与本科知识域不是一一对应关系；尤其 CSAPP 同时服务系统基础、体系结构和 OS，SoC 则是硬件方向的综合应用。

| 子主题 | 内容概述 | 在学习架构中的作用 | 材料状态（沿用） |
| --- | --- | --- | --- |
| [[01-ComputerScience/Architecture/Architecture\|Architecture]] | CSAPP、RISC-V 组织、超标量处理器、AI 空间架构、浮点规范 | 先组成与性能模型，再按方向深入微架构 | 进行中 |
| [[01-ComputerScience/Compiler/Compiler\|Compiler]] | LLVM 与编译流程基础 | 与语言语义、自动机、文法和系统执行连接 | 起步 |
| [[01-ComputerScience/Network/Network\|Network]] | 网络基础与高性能网络 | 先协议模型，再服务实现与通信优化 | 起步 |
| [[01-ComputerScience/operating-system/operating-system\|operating-system]] | 核心机制、虚拟内存、进程线程、rCore 与 OS 八股 | 建立虚拟化、并发和持久化模型 | 进行中 |
| [[01-ComputerScience/Programming/Programming\|Programming]] | C/C++/Python/Rust/Golang/Lua/Qt、并发编程与 Web | 以程序设计能力组织，多语言按需要选用 | 进行中 |
| [[01-ComputerScience/SoC/SoC\|SoC]] | RISC-V、AMBA、NoC、cache、Chipyard、ysyx、PCIe | 组成与系统基础之后的硬件方向实践 | 进行中 |
| [[01-ComputerScience/system-basics/system-basics\|system-basics]] | 表示、转换、链接、执行、异常中断与 PA | 串起从源程序到机器执行的共同基础 | 进行中 |

## 关键笔记

- [[01-ComputerScience/CS-diy|CS-diy]]：完整知识地图、先修关系、诊断、阶段安排、主资料与证据标准。
- [[01-ComputerScience/Programming/web/前端|前端]]：HTML/Web 前端入门笔记，可用于平台实践；与 HCI 的用户研究和可用性评估配合。

## 学习路径

先诊断 C/C++ 与 Linux、算法和数学、单机系统的基础，再依次建立并行／GPU、框架与运行时、主机设备路径、通信和分布式执行模型。训练或推理项目把这些层连接起来，用正确性、性能与故障证据检验理解；具体入口和停止条件集中在 [[01-ComputerScience/CS-diy#四、按依赖推进的学习路径|学习路径]]。
