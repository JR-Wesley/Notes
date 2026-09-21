## 1. Introduction

### English Original

> This document describes PTX, a low-level parallel thread execution virtual machine and instruction set architecture (ISA). PTX exposes the GPU as a data-parallel computing device.

### 中文翻译

本文描述 PTX。PTX 是一种低级的并行线程执行虚拟机和指令集架构（Instruction Set Architecture，ISA），它将 GPU 呈现为一个数据并行计算设备。

### 重点解读

这里的核心词是“虚拟机”和“数据并行”。PTX 不是某一代 NVIDIA GPU 直接执行的最终机器指令，而是位于 CUDA 等高级语言与目标 GPU 原生指令之间的虚拟 ISA。它描述单个逻辑线程如何执行，同时让大量线程在不同数据元素上运行同一程序。

### 1.1 Scalable Data-Parallel Computing using GPUs

#### English Original

> Driven by the insatiable market demand for real-time, high-definition 3D graphics, the programmable GPU has evolved into a highly parallel, multithreaded, many-core processor with tremendous computational horsepower and very high memory bandwidth. The GPU is especially well-suited to address problems that can be expressed as data-parallel computations - the same program is executed on many data elements in parallel - with high arithmetic intensity - the ratio of arithmetic operations to memory operations. Because the same program is executed for each data element, there is a lower requirement for sophisticated flow control; and because it is executed on many data elements and has high arithmetic intensity, the memory access latency can be hidden with calculations instead of big data caches.
>
> Data-parallel processing maps data elements to parallel processing threads. Many applications that process large data sets can use a data-parallel programming model to speed up the computations. In 3D rendering large sets of pixels and vertices are mapped to parallel threads. Similarly, image and media processing applications such as post-processing of rendered images, video encoding and decoding, image scaling, stereo vision, and pattern recognition can map image blocks and pixels to parallel processing threads. In fact, many algorithms outside the field of image rendering and processing are accelerated by data-parallel processing, from general signal processing or physics simulation to computational finance or computational biology.
>
> PTX defines a virtual machine and ISA for general purpose parallel thread execution. PTX programs are translated at install time to the target hardware instruction set. The PTX-to-GPU translator and driver enable NVIDIA GPUs to be used as programmable parallel computers.

#### 中文翻译

在市场对实时、高分辨率 3D 图形的持续需求推动下，可编程 GPU 已经演化成具有强大计算能力和极高内存带宽的高度并行、多线程、多核处理器。GPU 尤其适合解决能够表示为数据并行计算的问题，即同一个程序并行作用于许多数据元素，并且具有较高的算术强度，即算术操作数量与内存操作数量之比。由于每个数据元素执行的是同一个程序，对复杂流控制的需求较低；又因为程序在大量数据元素上执行且算术强度较高，所以可以利用计算来隐藏内存访问延迟，而不必主要依赖大型数据缓存。

数据并行处理把数据元素映射到并行处理线程。许多处理大型数据集的应用都可以利用数据并行编程模型加速计算。在 3D 渲染中，大量像素和顶点会映射到并行线程。类似地，渲染图像后处理、视频编解码、图像缩放、立体视觉和模式识别等图像与媒体处理应用，可以把图像块和像素映射到并行线程。事实上，图像渲染与图像处理之外的许多算法也能通过数据并行处理得到加速，应用范围从通用信号处理、物理仿真一直延伸到计算金融和计算生物学。

PTX 为通用并行线程执行定义了一套虚拟机和 ISA。PTX 程序会在部署阶段被翻译成目标硬件指令集。PTX-to-GPU 翻译器和驱动程序使 NVIDIA GPU 能够作为可编程并行计算机使用。

#### 重点解读

这一节给出了适合 GPU 的两个必要特征。第一，任务必须能够拆成大量结构相近的数据元素，使许多线程有工作可做；第二，算术强度要足够高，使计算能够覆盖访存等待时间。所谓“隐藏延迟”并不表示访存延迟消失，而是 GPU 在一批线程等待数据时调度其他可运行线程或执行其他计算。

文中的 *install time* 不必狭义理解为操作系统的软件安装过程。工程上，它表示目标 GPU 确定之后的翻译阶段：PTX 可以由 `ptxas` 在构建期间转换，也可以由驱动在模块装载时进行 JIT 编译。总体链路是：高级语言源代码 → PTX → 目标 GPU 原生指令。

### 1.2 Goals of PTX

#### English Original

> PTX provides a stable programming model and instruction set for general purpose parallel programming. It is designed to be efficient on NVIDIA GPUs supporting the computation features defined by the NVIDIA Tesla architecture. High level language compilers for languages such as CUDA and C/C++ generate PTX instructions, which are optimized for and translated to native target-architecture instructions.
>
> The goals for PTX include the following:
>
> - Provide a stable ISA that spans multiple GPU generations.
> - Achieve performance in compiled applications comparable to native GPU performance.
> - Provide a machine-independent ISA for C/C++ and other compilers to target.
> - Provide a code distribution ISA for application and middleware developers.
> - Provide a common source-level ISA for optimizing code generators and translators, which map PTX to specific target machines.
> - Facilitate hand-coding of libraries, performance kernels, and architecture tests.
> - Provide a scalable programming model that spans GPU sizes from a single unit to many parallel units.

#### 中文翻译

PTX 为通用并行编程提供稳定的编程模型和指令集。它被设计为能够在支持 NVIDIA Tesla 架构所定义计算特性的 NVIDIA GPU 上高效运行。CUDA、C/C++ 等高级语言的编译器会生成 PTX 指令，再对这些指令进行优化，并把它们翻译成目标架构的原生指令。

PTX 的设计目标包括：

- 提供一套能够跨越多代 GPU 的稳定 ISA。
- 使编译后的应用程序获得可与 GPU 原生性能相比的性能。
- 为 C/C++ 及其他编译器提供与具体机器无关的目标 ISA。
- 为应用程序和中间件开发者提供用于代码分发的 ISA。
- 为优化型代码生成器和翻译器提供共同的源级 ISA，再由它们把 PTX 映射到特定目标机器。
- 便于手工编写程序库、性能关键 kernel 和架构测试。
- 提供可扩展的编程模型，覆盖从单个并行单元到许多并行单元的不同 GPU 规模。

#### 重点解读

PTX 的定位可以概括为“稳定但可继续优化的虚拟 ISA”。稳定表示它作为编译器接口和代码分发格式能够跨越多代 GPU；可继续优化表示 PTX 指令与最终硬件指令不要求一一对应。后端可以根据目标架构进行指令选择、合并、拆分、寄存器分配和调度。

跨代稳定也不等于任意 PTX 指令都能在任意 GPU 上运行。具体指令仍可能要求最低 PTX 版本、特定 `.target` 或最低 compute capability。手写 PTX 可以表达高级语言难以稳定生成的操作，但不会天然获得更高性能，并会增加架构兼容性和正确性维护成本。

### 1.3 PTX ISA Version 9.4

本节是版本新增特性清单，按要求省略。涉及具体指令时，应再回到对应 PTX 版本和目标架构要求进行核对。

### 1.4 Document Structure

#### English Original

> The information in this document is organized into the following Chapters:
>
> - Programming Model outlines the programming model.
> - PTX Machine Model gives an overview of the PTX virtual machine model.
> - Syntax describes the basic syntax of the PTX language.
> - State Spaces, Types, and Variables describes state spaces, types, and variable declarations.
> - Instruction Operands describes instruction operands.
> - Abstracting the ABI describes the function and call syntax, calling convention, and PTX support for abstracting the Application Binary Interface (ABI).
> - Instruction Set describes the instruction set.
> - Special Registers lists special registers.
> - Directives lists the assembly directives supported in PTX.
> - Release Notes provides release notes for PTX ISA versions 2.x and beyond.

#### 中文翻译

本文档中的信息按照以下章节组织：

- Programming Model 概述 PTX 编程模型。
- PTX Machine Model 概述 PTX 虚拟机模型。
- Syntax 描述 PTX 语言的基本语法。
- State Spaces, Types, and Variables 描述状态空间、类型和变量声明。
- Instruction Operands 描述指令操作数。
- Abstracting the ABI 描述函数与调用语法、调用约定，以及 PTX 对应用二进制接口（Application Binary Interface，ABI）抽象的支持。
- Instruction Set 描述 PTX 指令集。
- Special Registers 列出特殊寄存器。
- Directives 列出 PTX 支持的汇编 directive。
- Release Notes 提供 PTX ISA 2.x 及后续版本的发布说明。

#### References — English Original

> - 754-2008 IEEE Standard for Floating-Point Arithmetic. ISBN 978-0-7381-5752-8, 2008.
>   - http://ieeexplore.ieee.org/servlet/opac?punumber=4610933
> - The OpenCL Specification, Version: 1.1, Document Revision: 44, June 1, 2011.
>   - http://www.khronos.org/registry/cl/specs/opencl-1.1.pdf
> - CUDA Programming Guide.
>   - https://docs.nvidia.com/cuda/cuda-programming-guide/index.html
> - CUDA Dynamic Parallelism Programming Guide.
>   - https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html
> - CUDA Atomicity Requirements.
>   - https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#atomicity
> - PTX Writers Guide to Interoperability.
>   - https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/index.html

#### 参考资料 — 中文翻译

- IEEE 754-2008 浮点算术标准，ISBN 978-0-7381-5752-8，2008 年。
  - <http://ieeexplore.ieee.org/servlet/opac?punumber=4610933>
- OpenCL 规范 1.1 版，文档修订版 44，2011 年 6 月 1 日。
  - <http://www.khronos.org/registry/cl/specs/opencl-1.1.pdf>
- CUDA 编程指南。
  - <https://docs.nvidia.com/cuda/cuda-programming-guide/index.html>
- CUDA 动态并行编程指南。
  - <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html>
- CUDA 原子性要求。
  - <https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#atomicity>
- PTX 编写者互操作指南。
  - <https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/index.html>

