---
type: moc
domain: 01-ComputerScience
topic: knowledge-system-and-resources
status: active
tags: [CS, learning]
---

# 计算机科学知识树与资料地图

本文件按知识体系组织，不规定线性时间表；实际项目触发哪个主题，就从该主题的“知识点—现有笔记—参考资料—证据”链路进入。表中的“现有笔记”是仓库材料，不等于已经完成学习。

## 一、知识树总览

| 一级领域 | 二级子领域 | 关键知识点 | 依赖/连接 |
| --- | --- | --- | --- |
| 编程基础 | C/C++ 与系统编程 | 类型、指针、内存、RAII、STL、调试、构建 | 系统基础、OS、网络项目 |
| 编程基础 | Rust/Go/Python/Web | 所有权、并发、工程工具、脚本与服务 | 项目实现与工具链 |
| 编程基础 | 数据结构与算法 | 复杂度、树图、哈希、并查集、排序、基础算法设计 | 所有实现与分布式项目 |
| 计算机系统 | 程序表示与转换 | 数据表示、汇编、编译、链接、ELF、ABI | 体系结构、编译器、调试 |
| 计算机系统 | 程序执行与存储 | 指令执行、内存布局、缓存、虚拟地址、性能 | OS、体系结构、SoC |
| 计算机系统 | 异常、中断与 IO | 特权级、异常、中断、设备、系统调用、DMA | OS、网络、驱动 |
| 体系结构 | ISA 与组成 | ISA、数据通路、流水线、缓存、分支预测、乱序 | 系统基础、RISC-V、SoC |
| 体系结构 | 性能与模拟 | CPI、局部性、性能计数、模拟器、性能优化 | 编译器、GPU、高性能系统 |
| 操作系统 | 进程、线程与调度 | 进程模型、线程、上下文切换、调度、同步 | 编程、系统基础、网络 |
| 操作系统 | 虚拟内存与文件 | 地址空间、分页、缺页、文件系统、持久化 | 体系结构、存储、数据库 |
| 操作系统 | IO 与内核实践 | 阻塞/非阻塞、系统调用、锁、内核模块、驱动 | 网络、SoC、性能工程 |
| 网络 | 网络基础 | 分层、TCP/IP、路由、可靠性、拥塞、socket | OS、并发、分布式 |
| 网络 | 服务端工程 | TCP 服务、线程池、事件循环、抓包、压测 | C++ 并发、IO、性能 |
| 网络 | 高性能通信 | RDMA、DPDK、OVS、网卡驱动、FPGA 网卡 | 体系结构、OS、存储 |
| 编译器 | 编译基础 | 词法、语法、语义、IR、优化、代码生成 | 编程、系统基础、ISA |
| 编译器 | LLVM 与系统优化 | LLVM IR、pass、后端、深度学习编译器 | 体系结构、GPU、AI 系统 |
| 分布式/数据 | 分布式系统 | 故障模型、一致性、复制、Raft、共识、RPC | 网络、并发、数据结构 |
| 分布式/数据 | 数据库与存储 | KV、事务、日志、恢复、对象存储、纠删码、NVMe/SPDK | OS、网络、分布式 |
| 并行/AI | 并行计算与 GPU | 多核、SIMD、CUDA、kernel、profiling、FlashAttention | 体系结构、编译器、性能 |
| 硬件系统 | SoC/RISC-V/嵌入式 | RTL、AMBA、NoC、缓存、Chipyard、ysyx、PCIe、STM32 | 组成、OS、网络、验证 |

## 二、知识点—笔记—资料关联

### 1. 编程与算法

| 子领域/知识点            | 当前笔记                                                                                                                                                                   | 参考资料                                                                                                                                                                                               | 建议证据                  |     |     |     |     |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- | --- | --- | --- | --- |
| C、内存、调试            | [[01-ComputerScience/Programming/C]]、[[01-ComputerScience/Programming/程序执行的内存分配]]                                                                                      | CSAPP、CMU 15213                                                                                                                                                                                    | 可调试的小型系统程序与内存分析       |     |     |     |     |
| C++、STL、现代特性       | [[01-ComputerScience/Programming/现代C++/现代C++]]、[[01-ComputerScience/Programming/现代C++/系统梳理/系统梳理]]                                                                      | [现代 C++ 教程](https://changkun.de/modern-cpp/zh-cn/00-preface/)                                                                                                                                      | C++ 项目、测试、构建和错误复盘     |     |     |     |     |
| 并发编程               | [[01-ComputerScience/Programming/并发编程/并发编程]]、[[01-ComputerScience/Programming/并发编程/并发编程核心]]                                                                            | 《Linux 多线程服务端编程》、C++ 并发资料                                                                                                                                                                          | 竞态复现、同步修复、基准对比        |     |     |     |     |
| 数据结构与算法            | 目前主要作为 CS 资料入口，具体笔记见算法领域                                                                                                                                               | [Hello 算法](https://www.hello-algo.com/)、[CS 61B](https://sp18.datastructur.es/)、[LeetCode](https://leetcode.cn/)、[代码随想录](https://programmercarl.com/)、[CodeTop](https://codetop.cc/home)           | 能解释复杂度并在项目中正确使用       |     |     |     |     |
| Rust/Go/Python/Web | [[01-ComputerScience/Programming/Rust]]、[[01-ComputerScience/Programming/golang]]、[[01-ComputerScience/Programming/Python]]、[[01-ComputerScience/Programming/web/web]] | [Rust 中文书](https://rustwiki.org/zh-CN/book/title-page.html)、[Go 语言圣经](https://golang-china.github.io/gopl-zh/)、[TypeScript 手册](https://typescript.bootcss.com/)、[ES6](https://es6.ruanyifeng.com/) | 一个可运行工具或服务及 README/测试 |     |     |     |     |

### 2. 系统基础与体系结构

| 子领域/知识点       | 当前笔记                                                                                      | 参考资料                                                                                           | 建议证据        |                                                                                                                                 |                              |                 |
| ------------- | ----------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- | --------------- |
| 表示、编译、汇编、链接 | [[01-ComputerScience/system-basics/一-程序的表示-转换与链接/一-程序的表示-转换与链接]]、[[01-ComputerScience/system-basics/一-程序的表示-转换与链接/程序的链接]] | 《深入理解计算机系统》、[CMU 15-213](https://www.cs.cmu.edu/~213/) | gcc/clang、objdump、readelf 实验 |
| 执行、内存、缓存 | [[01-ComputerScience/system-basics/二-程序的执行和存储访问/二-程序的执行和存储访问]]、[[01-ComputerScience/system-basics/二-程序的执行和存储访问/层次存储结构]] | 《计算机系统基础》、CSAPP | 汇编、缓存或内存访问测量 |
| 异常、中断、IO | [[01-ComputerScience/system-basics/三-异常-中断和输入-输出/三-异常-中断和输入-输出]]、[[01-ComputerScience/system-basics/三-异常-中断和输入-输出/8.IO操作的实现]] | [[01-ComputerScience/system-basics/nju-系统基础]] | 系统调用、缺页、IO 路径说明 |
| 组成、ISA、流水线、缓存 | [[01-ComputerScience/Architecture/Architecture]]、[[01-ComputerScience/Architecture/book-cs-organization-riscv/book-cs-organization-riscv]] | 《计算机组成与设计》、[CS 61C](https://cs61c.org/fa24/)、[ETH DDCA](https://csdiy.wiki/%E4%BD%93%E7%B3%BB%E7%BB%9F%E7%BB%93%E6%9E%84/DDCA/) | 数据通路图、模拟器或性能解释 |
| 超标量与性能 | [[01-ComputerScience/Architecture/book-super-scalar-processor/book-super-scalar-processor]]、[[01-ComputerScience/Architecture/Simulator]] | CSAPP、体系结构课程 | profiling、CPI/缓存/分支数据 |

### 3. 操作系统与 IO

| 子领域/知识点 | 当前笔记 | 参考资料 | 建议证据 |
| --- | --- | --- | --- |
| 进程、线程、调度 | [[01-ComputerScience/operating-system/进程与线程]]、[[01-ComputerScience/operating-system/OSTEP/Concurrency]] | 《操作系统导论》、[MIT 6.S081](https://pdos.csail.mit.edu/6.S081/2021/schedule.html)、[CS 162](https://cs162.org/) | xv6/rCore lab 或并发 bug 复盘 |
| 虚拟内存、地址空间 | [[01-ComputerScience/operating-system/虚拟内存]]、[[01-ComputerScience/operating-system/虚拟地址空间]] | OSTEP、[rCore Tutorial](https://rcore-os.cn/rCore-Tutorial-Book-v3/)、[Writing an OS in Rust](https://os.phil-opp.com/) | 地址转换、page fault、映射实验 |
| IO、文件、内核 | [[01-ComputerScience/operating-system/IO]]、[[01-ComputerScience/operating-system/操作系统核心]]、[[01-ComputerScience/operating-system/rCore]] | [LearningOS](https://github.com/LearningOS)、[JYY OS](https://jyywiki.cn/OS/2023/index.html) | 系统调用/文件系统/阻塞行为验证 |

### 4. 网络、高性能、分布式与存储

| 子领域/知识点 | 当前笔记 | 参考资料 | 建议证据 |
| --- | --- | --- | --- |
| TCP/IP 与网络基础 | [[01-ComputerScience/Network/计算机网络概述]] | 《计算机网络：自顶向下方法》、[CS 144](https://cs144.github.io/)、[中科大课程视频](https://www.bilibili.com/video/BV1JV411t7ow/) | socket/可靠传输实验、抓包记录 |
| 网络服务端 | [[01-ComputerScience/Network/高性能网络]]、[[01-ComputerScience/Programming/web/网络编程]] | 《Unix 网络编程》《Linux 高性能服务器编程》《TCP/IP 网络编程》 | 并发 TCP 服务、压测和故障复盘 |
| RDMA/DPDK/OVS/驱动 | [[01-ComputerScience/Network/高性能网络]] | RDMA、DPDK、OVS、网卡驱动和 FPGA 网卡资料 | 记录协议、拓扑、驱动、拥塞/流控与 TCP 对比 |
| 分布式与 Raft | 当前以资源和项目入口为主 | [MIT 6.5840](https://pdos.csail.mit.edu/6.824/schedule.html) | Raft/KV 项目、故障注入、一致性分析 |
| 数据库与存储 | 当前以待建设子领域为主 | [CMU 15-445](https://15445.courses.cs.cmu.edu/)、NVMe/SPDK、事务、纠删码、对象存储 | workload、恢复、持久化和事务测试 |

### 5. 编译器、并行/GPU、AI 与 SoC

| 子领域/知识点 | 当前笔记 | 参考资料 | 建议证据 |
| --- | --- | --- | --- |
| 编译器基础与 LLVM | [[01-ComputerScience/Compiler/Compiler]]、[[01-ComputerScience/Compiler/LLVM简介]] | [北大编译实践](https://pku-minic.github.io/online-doc/)、[CS143](https://web.stanford.edu/class/cs143/)、[KAIST CS420](https://github.com/kaist-cp/cs420)、[LLVM Tutorial](https://llvm.org/docs/tutorial/)、龙书 | 解释器、IR、LLVM pass 或后端实验 |
| 并行计算与 GPU | [[01-ComputerScience/Architecture/AI-Spatial-Architecture]] | [Stanford CS149](https://gfxcourses.stanford.edu/cs149/fall24)、CUDA、FlashAttention、算子优化 | kernel、正确性测试、profiling 与优化数据 |
| AI 基础与 LLM 工程 | 本领域笔记分布在 AI System | [李宏毅 ML](https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php)、[LLM Cookbook](https://datawhalechina.github.io/llm-cookbook/) | 可运行实验、版本、数据和限制 |
| SoC/RISC-V/嵌入式 | [[01-ComputerScience/SoC/SoC]]、[[01-ComputerScience/SoC/ysyx/ysyx]]、[[01-ComputerScience/SoC/riscv/riscv]]、[[01-ComputerScience/SoC/Embedded/Embedded]] | AMBA、NoC、Chipyard、PCIe、STM32、PULP、RVFPGA 仓库资料 | RTL/仿真、波形、测试、工具链和硬件限制 |

## 三、综合入口与选择规则

| 用途 | 入口 |
| --- | --- |
| 综合路线与课程发现 | [JYY Reading List](https://jyywiki.cn/Reading_List.md)、[CS 自学指南](https://csdiy.wiki/)、[HackWay](https://hackway.org/)、[LearnCS](https://www.learncs.site/) |
| NJU 系统/程序课程 | [NJU 程序构造](http://www.why.ink:8080/CPL/2023/)、[NJU OS ICS Wiki](https://njuics-wiki.github.io/ics-wiki/) |
| 资源使用 | 每个项目优先选一个主教材/课程、一个实现或 lab、一个可运行产出；同类资料只在解释不同或项目需要时并列。 |
| 学习记录 | 记录触发问题、知识点、现有笔记、参考资料、产出证据和未解决问题；不把阅读链接当作完成。 |

## 四、学习记录

当前个人完成证据：尚无。后续每次记录触发问题、对应知识点、关联笔记、参考资料、产出证据和未解决问题；不把阅读链接本身当作完成。
