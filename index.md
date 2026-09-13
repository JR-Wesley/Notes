# 数字公园 · Personal Knowledge Base

![[00-ToolKit/assets/2436x1125-0bc7b5cd302841f5a0463ab8e55c3d02.jpg]]

*Where nanoseconds meet neurons.*

一个持续积累与整理的个人知识库，以计算机系统、AI Systems、GPU/HPC、算法和集成电路为核心，同时保存跨学科学习与个人成长记录。使用 Markdown 编写，以 Obsidian 组织与浏览，通过 Git 追踪版本，并借助 AI 辅助阅读、分析和维护。

这里保存概念理解、系统分析、源码阅读、论文与课程笔记，以及实验、工具和学习过程记录。内容处于不同成熟阶段；课程摘录、待验证推论与个人实践需要结合笔记中的来源和上下文阅读。

[用 AI 操作知识库](#用-ai-操作知识库) · [领域导航](#领域导航) · [学习路径](#学习路径) · [使用方式](#使用方式) · [Digital Garden](https://jr-wesley.github.io/MyDigitalGarden/)

## 领域导航

| 领域                                                                          | 内容与边界                    | 主要主题                                                                                                      |
| --------------------------------------------------------------------------- | ------------------------ | --------------------------------------------------------------------------------------------------------- |
| [00-ToolKit](00-ToolKit/00-tool-kit.md)                                     | 可复用的工具、环境配置与工作流，回答“如何使用” | Development、System、Scripts、Template、AI 协作入口                                                               |
| [01-ComputerScience](01-ComputerScience/01-ComputerScience.md)             | 从程序执行到硬件架构的计算机基础         | Architecture、Compiler、Network、OperatingSystem、Programming、SoC、SystemBasics                                |
| [02-AISystem](02-AISystem/02-AISystem.md)                                   | 模型运行所需的软件、通信和硬件系统        | GPU、Distributed、Framework、Inference、AICompiler、AlgorithmAndModel、ClusterAndHardware、HPCBasics、AISysReview |
| [03-Algorithm](03-Algorithm/03-Algorithm.md)                                | 算法原理、数学基础及应用             | BasicAlgorithm、Compression、Encryption、HDC、ComputerGraphics                                                |
| [04-IntegratedCircuit](04-IntegratedCircuit/04-integrated-circuit.md)       | 数字电路、硬件设计与芯片实现           | Accelerator、AsicFlow、Basics、IP                                                                            |
| [05-PersonalDevelopment](05-PersonalDevelopment/05-PersonalDevelopment.md) | 跨学科知识、职业发展与生活实践          | Career、Language、Economics、Philosophy、Psychology、Literature、Reading、Medicine&Food、Workout、music 等          |
| [Educated](Educated/Educated.md)                                            | 日记、学习进度、任务、规划与复盘         | 日期笔记、战略屋                                                                                                  |

`Educated` 是过程记录空间；可以从中提取值得长期维护的知识，同时保留日记的时间与情境。

AI Systems 的几个常用入口：

- [GPU](02-AISystem/GPU/GPU.md)：CUDA 编程、执行与存储模型、GPU 架构、算子、Tensor Core、CuTe。
- [Distributed](02-AISystem/Distributed/Distributed.md)：Parallelism、RDMA、NCCL、NVSHMEM、DeepEP 与分布式训练。
- [Framework](02-AISystem/Framework/Framework.md)：PyTorch、Megatron、vLLM、llama.cpp 与 Hugging Face 相关资料。
- [Inference](02-AISystem/Inference/Inference.md)：KV Cache、Attention、量化、解码与推理优化。
- [AISysReview](02-AISystem/AI-sys-review.md)：综合学习体系、课程和跨主题总结。

## 知识如何组织

长期目标是把领域、主题、概念与系统、来源、实验和应用连接起来：

```text
Domain → Topic → Concept / System
                       ↕
                 Source / Paper
                       ↕
              Experiment / Practice
                       ↓
                 Knowledge Network
```

| 组织方式 | 职责 |
| --- | --- |
| Folder | 稳定的领域与主题边界 |
| Note | 可独立维护的知识单元，保留必要上下文 |
| Metadata | 笔记类型、状态与来源 |
| WikiLink | 定义、依赖、实现、比较、应用等语义关系 |
| MOC / Index | 主题地图、学习路径或目录导航 |

文件夹不承担全部知识关系。通用概念与具体实现分别理解，例如 AllReduce 的语义和算法，与 NCCL 的协议、拓扑和 kernel 实现属于不同层次。书籍和课程保留章节顺序，再逐步抽取可复用知识。

技术笔记强调定义、架构、数据结构、调用关系、性能模型和实验依据；个人成长笔记保留观点、论证、证据与反思，不机械拆成碎片。

这些是持续维护的方向，尚未在全部历史笔记中统一落实。当前不少目录页由 Zoottelkeeper 自动生成，用于查找文件；人工知识地图则需要进一步说明概念关系和学习顺序。

## 用 AI 操作知识库

本页是知识库用户入口，目标是告诉你“去哪里”就够，不需要在这里看到完整的接口列表。

如果你不确定该从哪开始，先打开 [AI 对话入口总览](AI-Workspace/02-Prompts/README.md)；对于一般需求建议先用 [知识库智能入口](AI-Workspace/02-Prompts/00-Knowledge-Workbench.md)。智能入口会先读取 [工作流路由表](AI-Workspace/02-Prompts/Workflow-Catalog.md)，选出单一主流程并按需继续。

需要快速独立的小任务时，用 [快速任务](AI-Workspace/02-Prompts/00-Quick-Task.md)。需要跨对话、并行或高影响批量变更时，用 [统筹项目](AI-Workspace/02-Prompts/07-Coordinate-Team.md)。

新增功能：当你告诉我某个学习主题时，可以直接走 [学习辅导](AI-Workspace/02-Prompts/26-Learning-Tutor.md)；AI 先定位现有文档、给出理解路径，再补齐关键知识点。若你希望改进现有文档可用 [文档改进](AI-Workspace/02-Prompts/27-Document-Improvement.md)，输出修改建议或按授权直接改写。

凡涉及知识正文成文的任务，默认使用清晰、自然、逻辑连贯且信息密度适当的段落表达。`AI-Workspace` 与领域知识库分离，并在 `.gitignore` 中排除。

## 学习路径

以下是按现有内容组织的浏览顺序，可根据具体问题跨领域阅读。

| 方向 | 建议顺序 |
| --- | --- |
| 计算机系统 | [系统基础](01-ComputerScience/system-basics/system-basics.md) → [体系结构](01-ComputerScience/Architecture/Architecture.md) → [操作系统](01-ComputerScience/operating-system/operating-system.md) → [系统编程](01-ComputerScience/Programming/Programming.md) |
| GPU 编程与优化 | [HPC 基础](02-AISystem/hpc-basics/hpc-basics.md) → [GPU](02-AISystem/GPU/GPU.md) 中的 CUDA / PMPP → 算子与存储优化 → Tensor Core / CuTe |
| 分布式 AI 系统 | [模型基础](02-AISystem/algorithm-and-model/algorithm-and-model.md) → [并行策略](02-AISystem/Distributed/Parallelism/Parallelism.md) → [通信系统](02-AISystem/Distributed/Distributed.md) → [框架实现](02-AISystem/Framework/Framework.md) → [推理优化](02-AISystem/Inference/Inference.md) |
| 数字 IC 与处理器 | [电路与设计基础](04-IntegratedCircuit/Basics/Basics.md) → [SoC](01-ComputerScience/SoC/SoC.md) / [IP](04-IntegratedCircuit/IP/IP.md) → [加速器](04-IntegratedCircuit/Accelerator/Accelerator.md) → [ASIC Flow](04-IntegratedCircuit/asic-flow/asic-flow.md) |

阅读时结合系统整体与实现细节，从基本原理推导，再通过源码和实践检查理解。源码笔记应尽量明确 Repository、Commit、File 和 Symbol；实验记录应说明环境、命令、配置和结果条件，区分实测与估算。

## 使用方式

在 Obsidian 中将仓库根目录打开为 Vault，从本页或各领域索引开始浏览，结合搜索、WikiLink 与 Backlink 跟进相关笔记。

在 GitHub 或普通 Markdown 阅读器中，可通过本页的标准 Markdown 链接导航。部分笔记使用 Obsidian WikiLink、嵌入、Canvas、Base 或插件语法，其展示效果取决于阅读器支持。

附件主要保存在笔记或主题附近的 `assets` 目录，也有历史遗留的 `pic`、`pics`、`images` 等布局。移动文件时需要同时检查引用，不能仅根据“看起来没有使用”删除附件。

## 许可

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
