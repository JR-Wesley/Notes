# 数字公园 · Personal Knowledge Base

![[00-ToolKit/assets/2436x1125-0bc7b5cd302841f5a0463ab8e55c3d02.jpg]]

*Where nanoseconds meet neurons.*

一个持续积累与整理的个人知识库，以计算机系统、AI Systems、GPU/HPC、算法和集成电路为核心，同时保存跨学科学习与个人成长记录。使用 Markdown 编写，以 Obsidian 组织与浏览，通过 Git 追踪版本，并借助 AI 辅助阅读、分析和维护。

这里保存概念理解、系统分析、源码阅读、论文与课程笔记，以及实验、工具和学习过程记录。内容处于不同成熟阶段；课程摘录、待验证推论与个人实践需要结合笔记中的来源和上下文阅读。

[用 AI 操作知识库](#用-ai-操作知识库) · [领域导航](#领域导航) · [学习路径](#学习路径) · [使用方式](#使用方式) · [Digital Garden](https://jr-wesley.github.io/MyDigitalGarden/)

## 用 AI 操作知识库

本 README 是用户入口。每次想让 AI 阅读、生成、整理或维护知识时，从这里选择需求；打开对应提示词文件，复制其中完整的 `text` 代码块，填写 `【】` 中会影响结果的信息，再发送给能够访问本仓库或你所提供材料的 AI 对话。

### 最简单的开始方式

| 你的情况 | 打开这个入口 | 使用方法 |
| --- | --- | --- |
| 不确定该选什么功能 | [知识库智能入口](AI-Workspace/02-Prompts/00-Knowledge-Workbench.md) | 描述目标、材料和允许动作，由 AI 选择工作流并执行 |
| 目标明确，属于一次性小任务 | [快速任务](AI-Workspace/02-Prompts/00-Quick-Task.md) | 填写目标、范围和完成标准 |
| 要按学习列表持续建设一组知识 | [系统知识建设](AI-Workspace/02-Prompts/16-System-Knowledge-Research-And-Write.md) | 提供学习需求、主题列表和材料范围 |
| 工作会跨阶段、多个对话或大量文件 | [统筹项目](AI-Workspace/02-Prompts/07-Coordinate-Team.md) | 提供最终目标、授权范围与完成标准 |

通常从“知识库智能入口”开始即可。AI 会继续使用当前对话中的上下文；不需要每一步重新复制提示词。只有确实更换对话、需要独立审阅或任务可以并行时，才使用项目、交接或专家入口。

### 按日常需求选择

| 我想做什么 | 用户应打开的提示词 |
| --- | --- |
| 根据已有笔记回答问题、比较观点、找缺口 | [向知识库提问](AI-Workspace/02-Prompts/17-Ask-Vault.md) |
| 处理网页、PDF、图片、摘录或临时想法 | [采集与处理材料](AI-Workspace/02-Prompts/18-Capture-And-Process.md) |
| 阅读论文、书籍、课程、规范或源码 | [阅读来源](AI-Workspace/02-Prompts/19-Study-Source.md) |
| 调研一个问题或写一篇新笔记 | [主题研究](AI-Workspace/02-Prompts/08-Research-Topic.md) · [生成知识笔记](AI-Workspace/02-Prompts/09-Write-Knowledge.md) |
| 修改现有笔记或检查内容质量 | [编辑知识](AI-Workspace/02-Prompts/12-Edit-Knowledge.md) · [审阅交付](AI-Workspace/02-Prompts/10-Review-Deliverable.md) |
| 综合多篇笔记、建立 WikiLink 或 MOC | [综合与连接](AI-Workspace/02-Prompts/20-Synthesize-And-Connect.md) |
| 设计实验、运行获准实践、整理实测结果 | [实验与验证](AI-Workspace/02-Prompts/21-Experiment-And-Validate.md) |
| 制定学习路线，或从日记与复盘中提炼下一步 | [学习规划](AI-Workspace/02-Prompts/11-Plan-Learning.md) · [处理日记与复盘](AI-Workspace/02-Prompts/22-Process-Daily-Notes.md) |
| 定期检查知识成熟度、陈旧内容和缺口 | [周期性知识复查](AI-Workspace/02-Prompts/23-Review-Knowledge-System.md) |
| 维护链接、附件、Metadata、MOC、索引、目录或命名 | [知识结构维护](AI-Workspace/02-Prompts/24-Maintain-Knowledge-Structure.md) · [目录整理](AI-Workspace/02-Prompts/02-Organize-Vault.md) |
| 整理单篇 Markdown，或只读检查一批笔记 | [单文件格式](AI-Workspace/02-Prompts/01-Format-One-Note.md) · [只读检查](AI-Workspace/02-Prompts/03-Audit-Notes.md) |
| 从缺口和待办中安排下一批知识工作 | [知识工作排期](AI-Workspace/02-Prompts/25-Plan-Knowledge-Backlog.md) |
| 启动、继续、交接或收尾一个长期项目 | [统筹项目](AI-Workspace/02-Prompts/07-Coordinate-Team.md) · [启动或继续](AI-Workspace/02-Prompts/04-Start-Or-Resume-Project.md) · [对话交接](AI-Workspace/02-Prompts/05-Handoff-Conversation.md) · [项目收尾](AI-Workspace/02-Prompts/06-Close-And-Improve.md) |
| 给长期参与的 AI 对话指定固定职责 | [长期专家接入（可选）](AI-Workspace/02-Prompts/00-Join-As-Expert.md) |
| 查询历史项目或总结当前对话 | [查询项目档案](AI-Workspace/02-Prompts/14-Query-Project-Archives.md) · [总结当前对话](AI-Workspace/02-Prompts/15-Summarize-Current-Conversation.md) |
| 调整或扩展 AI 提示词体系 | [维护提示词与使用指南](AI-Workspace/02-Prompts/13-Extend-Prompts-And-Guide.md) |

[查看完整提示词目录](AI-Workspace/02-Prompts/README.md)

### 用户入口与 AI 对话入口

| 层级 | 面向谁 | 作用 |
| --- | --- | --- |
| 本 README | 用户 | 按目标找到正确功能，打开并复制一份提示词 |
| `AI-Workspace/02-Prompts` 中的提示词 | 收到消息的 AI 对话 | 提供本次任务的输入、边界、执行方式和交付要求 |
| [工作流路由表](AI-Workspace/02-Prompts/Workflow-Catalog.md) | 使用智能入口的 AI | 从用户意图选择一项主工作流，不加载全部提示词 |
| [AGENTS.md](AGENTS.md) | 在仓库中工作的 AI | 提供始终适用的最小边界和按需规则路由 |
| [项目档案](AI-Workspace/03-Projects/README.md) | 用户与长任务 AI | 保存跨阶段任务的状态、成果和证据 |

用户不需要把整个 README、全部提示词或维护规则一起发送。一次选择一个入口即可；AI 根据提示词读取完成任务所需的规则和材料。只有明确允许创建或修改文件时，提示词才授权对应写入。`AI-Workspace` 是本机 AI 操作空间，已与知识领域目录分离并从 Git 版本跟踪中排除。

所有会生成或实质编辑知识笔记的入口共享同一套成文规范：默认使用清晰、自然、逻辑连贯且信息密度合适的段落，标题、列表和表格只在内容关系确实需要时使用。具体语言风格仍由主题、读者和任务类型决定，不把技术写作习惯强加给旅行、生活、创作或其他领域。

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
