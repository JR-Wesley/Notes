---
dateCreated: 2025-06-27
dateModified: 2025-07-04
---
# LLVM

## 简介

LLVM 是一个开源的模块化编译基础设施（Compiler Infrastructure），最初由 Chris Lattner 在 2000 年开发，如今已成为现代编译技术的核心框架。其名称最初是 “Low Level Virtual Machine” 的缩写，但现在 LLVM 项目已不仅仅是虚拟机，而是涵盖了从前端到后端的完整编译流程。

需重点突出其 **编译器框架设计、中间表示（IR）、优化技术** 等核心概念

- **核心思想**：
    - **三段式设计**：前端（如 Clang）→ 中间优化层（LLVM IR）→ 后端（目标代码生成）。
    - **与 GCC 对比**：LLVM 更模块化，支持动态编译（JIT）、跨平台（ARM/x86/RISC-V 等）。

## 核心架构

1. **模块化设计**

LLVM 将编译过程拆分为独立模块，包括前端（Frontend）、优化器（Optimizer）、后端（Backend），各模块通过中间表示（Intermediate Representation, IR）连接，支持 “一次编译，多处优化 / 生成”。

1. **中间表示（LLVM IR）**

- 是一种低级、静态单赋值（SSA）形式的代码表示，独立于编程语言和目标架构。
- 支持三种形式：文本形式（.ll）、二进制形式（.bc）、内存中的即时表示。
- 例如，C 语言代码 `int add(int a, int b) { return a + b; }` 的 LLVM IR 大致如下：

```llvm
define i32 @add(i32 %a, i32 %b) {
  %sum = add i32 %a, %b
  ret i32 %sum
}
```

1. **主要组件**

- **Clang**：LLVM 的 C/C++/Objective-C 前端，替代传统的 GCC。
- **LLVM Optimizer（opt）**：对 IR 进行优化，支持数百种优化 passes（如常量折叠、循环展开、指令调度等）。
- **LLVM Backend（llc）**：将优化后的 IR 转换为目标机器码，支持 x86、ARM、RISC-V 等架构，甚至 GPU 和自定义硬件。
- **LLVM JIT（Just-In-Time Compiler）**：如 `libLLVM` 提供动态编译能力，用于 V8、Swift 等语言的运行时。

#### 3. **LLVM 核心组件**

| **组件**                | **功能**                        |
| --------------------- | ----------------------------- |
| **Clang**             | C/C++/Objective-C 前端，生成 LLVM IR |
| **LLVM Pass**         | 优化模块（如内联优化、死代码消除）|
| **LLVM CodeGen**      | 将 IR 转换为目标机器码（如 x86 汇编）|
| **LLVM Linker (lld)** | 链接器，支持 ELF/Mach-O/COFF 格式       |
|                       |                               |

## 应用场景

#### 1. **编程语言开发**

- **Rust**：使用 LLVM 作为后端生成高效机器码。
- **Swift**：Swift 编译器将 Swift 代码转换为 LLVM IR 进行优化。
- C/C++（Clang）、Swift、Rust（部分组件）、Dart（Flutter 编译）等。
- 自定义语言开发：通过 LLVM 构建前端，快速实现跨平台编译。

#### 2. **静态分析与安全工具**

- **Clang Static Analyzer**：基于 LLVM IR 的静态漏洞检测工具。
- **Sanitizers**：内存检测（AddressSanitizer）、未定义行为检测（UBSan）。

1. **代码优化与分析**

- 静态分析工具：`clang-tidy`、`scan-build`。
- 编译器优化研究：开发者可自定义优化 passes 提升特定场景性能。

1. **动态编译与虚拟机**

- JavaScript 引擎（如 V8）、游戏引擎（Unity 的 IL2CPP）、Python JIT（如 PyPy）。

1. **硬件与领域特定编译**

- GPU 编译（NVIDIA CUDA、OpenCL）、FPGA 编程、安全领域（代码混淆、加固）。
- **自定义优化 Pass**：实现论文中的新优化算法（如自动并行化）。
- **硬件加速器支持**：为新型处理器（如 TPU）添加后端支持。

## 理论基础

- **编译原理基础**：了解编译流程（词法分析、语法分析、语义分析、中间代码生成、优化、目标代码生成），推荐教材《编译原理及实践》或《Engineering a Compiler》。
- **C/C++ 编程**：LLVM 核心用 C++ 实现，需熟悉模板、STL、内存管理等。
- **数据结构与算法**：掌握图论（控制流图、数据流分析）、树（语法树）、哈希表等。

# 入门与工具使用
- **了解 LLVM 架构**：阅读 [LLVM 官方文档](https://llvm.org/docs/) 中的 _Architecture Manual_，理解 IR 和模块分工。
- **安装与实操 LLVM 工具链**（可以通过清华源下载）：

## IR 生成与转换

```bash
# 将 C 代码转为 LLVM IR（文本形式）
clang -S -emit-llvm hello.c -o hello.ll
# 转为二进制 IR
clang -c -emit-llvm hello.c -o hello.bc
# 优化 IR
opt -O3 hello.bc -o hello_opt.bc
# 生成目标代码
llc hello_opt.bc -o hello.s
```

### **1. 将 C 代码转为 LLVM IR（文本形式）**

```bash
clang -S -emit-llvm hello.c -o hello.ll
```  

- **功能**：将 C 源文件 `hello.c` 转换为 LLVM IR 的文本格式（`.ll` 文件）。
- **参数解析**：
    - `-S`：仅生成汇编代码（此处为 LLVM IR，而非目标机器汇编），不执行链接。
    - `-emit-llvm`：指定输出格式为 LLVM IR（默认输出目标机器汇编）。
    - `-o hello.ll`：指定输出文件名为 `hello.ll`。
- **工作原理**：
    - Clang（LLVM 的 C/C++ 前端）解析 C 代码的语法和语义，构建抽象语法树（AST）。
    - AST 被转换为 LLVM IR（一种平台无关的中间表示，采用静态单赋值形式 SSA）。
- **输出示例（`hello.ll`）**：

##### **2. 将 C 代码转为二进制 LLVM IR**

```bash
clang -c -emit-llvm hello.c -o hello.bc
```

- **功能**：将 C 源文件 `hello.c` 转换为 LLVM IR 的二进制格式（`.bc` 文件，Bitcode）。
- **参数解析**：
    - `-c`：编译但不链接（生成目标文件，此处为二进制 IR）。
    - `-emit-llvm`：指定输出格式为 LLVM IR。
    - `-o hello.bc`：指定输出文件名为 `hello.bc`。
- **工作原理**：
    - 与生成文本 IR 类似，但输出为二进制格式（更紧凑，加载更快）。
    - `.bc` 文件可视为 LLVM IR 的 “字节码”，可被 LLVM 工具链直接处理（如优化、生成目标代码）。
- **应用场景**：
    - 作为编译中间产物，减少后续处理的解析开销（如优化工具 `opt` 优先处理 `.bc` 文件）。
    - 跨平台分发未编译的代码（类似 Java 的 `.class` 文件），由目标平台的 LLVM 后端生成机器码。

##### **3. 优化 LLVM IR**

```bash
opt -O3 hello.bc -o hello_opt.bc
```

- **功能**：对二进制 LLVM IR（`hello.bc`）应用优化，生成优化后的 IR（`hello_opt.bc`）。
- **参数解析**：
    - `-O3`：优化级别（3 为最高级别），等效于启用一系列优化 Pass（如常量折叠、死代码消除、循环展开、内联等）。
    - `-o hello_opt.bc`：指定优化后的输出文件。
- **工作原理**：
    - `opt` 工具加载 `.bc` 文件，按顺序应用优化 Pass（LLVM 的优化模块）。
    - 每个 Pass 分析或转换 IR，例如：
        - `ConstantFolding`：计算编译期可确定的表达式（如 `2+3` 直接替换为 `5`）。
        - `DeadCodeElimination`：移除不影响程序结果的代码（如未使用的变量）。
        - `LoopUnrolling`：展开小循环以减少分支预测失败。
- **输出示例（优化后的 IR）**：
    优化后可能会内联 `printf` 调用（取决于具体实现），或移除冗余的内存分配（如 `alloca` 和 `store` 指令）。

##### **4. 生成目标机器码（汇编）**

```bash
llc hello_opt.bc -o hello.s
```

- **功能**：将优化后的 LLVM IR（`hello_opt.bc`）转换为目标机器的汇编代码（`hello.s`）。
- **参数解析**：
    - `-o hello.s`：指定输出文件名为汇编代码（`.s` 扩展名）。
- **工作原理**：
    - `llc`（LLVM static compiler）是 LLVM 的静态编译器后端，负责将 IR 映射到目标硬件的指令集（如 x86、ARM、CUDA 等）。
    - 主要步骤：
        - **指令选择**：将 LLVM IR 指令（如 `add`、`load`）转换为目标机器的具体指令（如 x86 的 `addl`、`movl`）。
        - **寄存器分配**：将 IR 中的虚拟寄存器映射到物理寄存器（如 x86 的 `eax`、`ebx`）。
        - **指令调度**：重排指令以提高流水线效率，减少数据依赖导致的停顿。
- **输出示例（`hello.s`，x86-64 汇编）**：

```assembly
.section    .rodata
.LC0:
.string    "Hello, World!"
.text
.globl    main
.type    main, @function
main:
.LFB0:
  .cfi_startproc
  subq    $8, %rsp
  .cfi_def_cfa_offset 16
  leaq    .LC0(%rip), %rdi
  call    printf@PLT
  xorl    %eax, %eax
  addq    $8, %rsp
  .cfi_def_cfa_offset 8
  ret
  .cfi_endproc
```

### 扩展与应用场景

1. **完整编译流程**：
上述命令可组合为：

```bash
clang -S -emit-llvm hello.c -o hello.ll  # 生成文本IR（可选）
clang -c -emit-llvm hello.c -o hello.bc  # 生成二进制IR
opt -O3 hello.bc -o hello_opt.bc         # 优化IR
llc hello_opt.bc -o hello.s               # 生成汇编
as hello.s -o hello.o                     # 汇编为目标文件
ld hello.o -o hello                        # 链接为可执行文件
```

实际中，`clang` 可直接完成上述全流程（如 `clang -O3 hello.c -o hello`），但拆分步骤便于调试和定制优化。

1. **自定义优化 Pass**：
若开发了自定义 LLVM Pass（如 `MyPass.so`），可通过 `opt` 加载：

```bash
opt -load-pass-plugin=MyPass.so -passes=my-pass hello.bc -o hello_opt.bc
```

1. **跨平台编译**：
通过 `llc` 的 `-march` 参数指定目标架构：

```bash
llc -march=arm hello_opt.bc -o hello_arm.s  # 生成ARM汇编
```

## IR 结构

- **特点**：
    - **SSA 形式（Static Single Assignment）**：每个变量只赋值一次，便于优化。
    - **强类型系统**：显式标注类型（如 `i32`、`float`、`%struct`）。
    - **元数据支持**：附加调试信息（如 `!dbg`）或优化提示。
- **示例代码**：

```llvm
define i32 @add(i32 %a, i32 %b) {
  %result = add i32 %a, %b
  ret i32 %result
}
```

通过阅读 `.ll` 文件，分析函数、指令、类型系统（如 `i32`、`void*`）。

### 常见挑战与建议

1. **LLVM 代码库庞大**：先聚焦核心模块（如 `lib/IR`、`lib/Transforms/Scalar`），避免一开始通读全部源码。
2. **C++ 模板复杂度**：LLVM 大量使用模板元编程，可从简单 Pass 入手，逐步理解模板设计模式（如 traits、policy-based design）。
3. **缺乏实践场景**：尝试从小项目开始，例如：
    - 开发一个统计代码复杂度的工具。
    - 为 C 语言添加自定义编译指示（`#pragma`）的处理逻辑。
4. **版本兼容性**：LLVM 版本迭代快（如 16. x、17. x），注意文档与代码的版本匹配，优先参考最新稳定版（当前 2025 年可能已到 18+ 版本）。

# LLVM 优化技术 Pass

- **优化器原理**：学习数据流分析（如到达定值、活跃变量）、循环优化（归纳变量消除、循环展开），参考《Optimizing Compilers for Modern Architectures》。
- **Pass 基础**：LLVM 的核心扩展方式是编写 Pass，用于修改 IR。Pass 分为模块级、函数级、基本块级等。
- **入门示例：简单的 IR 转换**
    编写一个 Pass 来统计函数中的指令数量，或重命名变量。参考 LLVM 官方教程 [Writing an LLVM Pass](https://llvm.org/docs/WritingAnLLVMPass.html)。
- **代码实现步骤**：
    1. 使用 LLVM 的 C++ API，包含头文件（如 `llvm/IR/Function.h`）。
    2. 继承 `llvm::Pass` 类，实现 `runOnFunction` 等方法。
    3. 编译 Pass 为动态库，通过 `opt` 工具加载：

```bash
opt -load mypass.so -my-pass input.bc -o output.bc
```

#### 1. **Pass 机制**

- **优化流程**：通过 `PassManager` 管理一系列优化 Pass（如 `-O1`, `-O2`, `-O3`）。
- **Pass 类型**：
    - **Analysis Pass**：收集信息（如控制流图、数据依赖）。
    - **Transformation Pass**：修改 IR（如 `mem2reg` 将栈变量提升为寄存器变量）。
    - **Utility Pass**：辅助工具（如 `-verify` 检查 IR 合法性）。

#### 2. **经典优化技术**

- **内联优化（Inlining）**：将小函数调用替换为函数体，减少开销。

```llvm
; 优化前
%result = call i32 @add(i32 3, i32 5)

; 优化后（内联展开）
%result = add i32 3, 5
```

- **循环优化**：循环展开（Loop Unrolling）、循环不变量外提（LICM）。
- **死代码消除（DCE）**：删除不可达代码或无用计算结果。
- **向量化（Vectorization）**：将标量操作转换为 SIMD 指令（如 SSE/AVX）。

#### 3. **链接时优化（LTO）**

- **原理**：在链接阶段跨模块优化，消除冗余代码和全局变量。
- **应用**：提升大型项目（如 Chrome、Firefox）的性能。

# 前端开发
- **自定义前端（可选）**：若需为新语言开发编译器，可结合 ANTLR/LLVM 构建前端，将语法树转换为 LLVM IR。
- **实战项目**：
    - 开发一个简单的代码混淆器，修改 IR 结构。
    - 为特定领域语言（如数学表达式语言）实现 LLVM 后端。
- **阅读 LLVM 源码**：关注 `lib/IR`（IR 定义）、`lib/Transforms`（优化 Pass）、`lib/Target`（后端）等目录。
- **贡献代码**：在 [LLVM 社区](https://llvm.org/community/) 中参与讨论，提交 bug 修复或新功能，例如优化某个 Pass 的性能。
- **关注前沿技术**：跟踪 LLVM 开发者会议（如 LLVM Developer’s Meeting），了解 GPU 编译、AI 编译器（如 MLIR）等方向。

# LLVM 代码生成

#### 1. **目标平台支持**

- **目标描述**：通过 `TableGen` 定义指令集、寄存器、调用约定。
- **指令选择**：将 IR 指令映射到机器指令（如 `add` → `ADD` 或 `LEA`）。
- **寄存器分配**：将虚拟寄存器分配到物理寄存器（如使用图着色算法）。

#### 2. **SelectionDAG**

- **流程**：IR → SelectionDAG → 机器指令。
- **优化点**：合并冗余指令、消除公共子表达式。

#### 3. **JIT 编译**

- **LLVM JIT 引擎**：动态编译 IR 为机器码（用于解释器或动态语言如 Julia）。
- **示例**：

    ```c++
    LLVMInitializeNativeTarget();
    auto module = llvm::parseIRFile("example.ir");
    auto engine = llvm::EngineBuilder(std::move(module)).create();
    engine->runFunction(module->getFunction("main"), {});
    ```

# 资源推荐

- **官方文档**：
    [LLVM Documentation](https://llvm.org/docs/)（必看，包含开发者手册、编程指南等）。
- **书籍**：
    - 《LLVM in Action》（入门实战，涵盖 Pass 开发和优化）。
    - 《Engineering a Compiler》（编译原理与实践结合，适合进阶）。
    - 《Optimizing Compilers for Modern Architectures》（优化器深度解析）。
- **在线课程**：
    - Coursera《Compiler Construction》（普林斯顿大学课程，涉及 LLVM）。
    - B 站 / YouTube 上的 LLVM 实战教程（如 “LLVM 编译器开发” 系列）。
- **社区与工具**：
    - [LLVM Discourse](https://discourse.llvm.org/)（官方论坛）。
    - [Stack Overflow](https://stackoverflow.com/questions/tagged/llvm)（搜索问题）。
    - [Clang Builtin Reference](https://clang.llvm.org/docs/BuiltinReference.html)（C 扩展内置函数参考）。

# 面试问题

#### **基础问题**

1. **LLVM IR 与 AST 的区别？**

    - AST 是前端生成的语法树（保留语言结构），IR 是平台无关的中间表示（更接近机器指令）。
2. **什么是 Pass？列举常见的 Pass 类型。**

    - Pass 是优化模块，如 `mem2reg`（提升变量到寄存器）、`inline`（函数内联）。

#### **进阶问题**

1. **如何编写自定义 LLVM Pass？**

    - 继承 `Pass` 类，注册 Pass 到 PassManager，使用 `opt` 工具加载。
    - 示例代码：```c++
        struct HelloPass : public FunctionPass {
          bool runOnFunction(Function &F) override {
            errs() << "Function: " << F.getName() << "\n";
            return false; // 未修改 IR
          }
        };
        char HelloPass::ID = 0;
        static RegisterPass<HelloPass> X("hello", "Hello World Pass");

        ```
        

2. **LLVM 如何处理异常（如 C++ 的 try/catch）？**

    - 使用 `invoke` 指令和 `landingpad` 块实现异常控制流，依赖平台特定的异常处理表（如 Windows 的 SEH）。

#### **设计问题**

1. **如何用 LLVM 实现一个简单的 JIT 编译器？**
    - 步骤：解析 IR → 创建 ExecutionEngine → 调用函数指针执行。
- **结合项目经验**：
    “在某个项目中，我通过添加自定义 Pass（如循环展开因子调整），将性能提升了 15%。”
- **引用技术细节**：
    “LLVM 的 `mem2reg` Pass 通过将栈变量提升为 SSA 寄存器，减少内存访问开销。”
- **对比其他工具**：
    “相比于 GCC，LLVM 的模块化设计使得添加新后端（如 RISC-V）更高效。”

熟悉 LLVM，ai 编译器和传统编译器都是潜在的工作职位，我个人是 tvm 和 llvm 都工作过一段时间。不考虑应届毕业生的情况下，想找一份 llvm 的工作，得考虑以下几点

1，对 LLVM 整体架构和一些基础概念熟悉，包括 [clang](https://zhida.zhihu.com/search?content_id=392137466&content_type=Answer&match_order=1&q=clang&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NTE3ODYwMjMsInEiOiJjbGFuZyIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjM5MjEzNzQ2NiwiY29udGVudF90eXBlIjoiQW5zd2VyIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.lAigBQk5STnQlkr5vouTTCld4CEMwrkkhH1m1UMNFvs&zhida_source=entity)，[llc](https://zhida.zhihu.com/search?content_id=392137466&content_type=Answer&match_order=1&q=llc&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NTE3ODYwMjMsInEiOiJsbGMiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjozOTIxMzc0NjYsImNvbnRlbnRfdHlwZSI6IkFuc3dlciIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.YJwhJXDFB59EeNnv_dhAovlPBsziu_b-wfhqpemvwJ0&zhida_source=entity)，[opt](https://zhida.zhihu.com/search?content_id=392137466&content_type=Answer&match_order=1&q=opt&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NTE3ODYwMjMsInEiOiJvcHQiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjozOTIxMzc0NjYsImNvbnRlbnRfdHlwZSI6IkFuc3dlciIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.F6CMRO9MQjQjhH_Xg9K7DdszmgUt7FoJsMAIBuEjnTM&zhida_source=entity)，[jit](https://zhida.zhihu.com/search?content_id=392137466&content_type=Answer&match_order=1&q=jit&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NTE3ODYwMjMsInEiOiJqaXQiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjozOTIxMzc0NjYsImNvbnRlbnRfdHlwZSI6IkFuc3dlciIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.bQEszFtHHAPc1ANvvufuY-0VX3J4AvaH3ZNWerCiOT4&zhida_source=entity)，[ssa](https://zhida.zhihu.com/search?content_id=392137466&content_type=Answer&match_order=1&q=ssa&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NTE3ODYwMjMsInEiOiJzc2EiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjozOTIxMzc0NjYsImNvbnRlbnRfdHlwZSI6IkFuc3dlciIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.G5cpGxfzEnu2v7f7YLzoYQs27aKUINxwXtcPNehIdK0&zhida_source=entity)，intrinsic 等等，把文档多过几遍应该也差不多熟悉了

2，熟悉以后，确定是想做前端还是后端，前端我不是很熟，说说后端吧，后端主要就是 [ir](https://zhida.zhihu.com/search?content_id=392137466&content_type=Answer&match_order=1&q=ir&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NTE3ODYwMjMsInEiOiJpciIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjM5MjEzNzQ2NiwiY29udGVudF90eXBlIjoiQW5zd2VyIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.n6pdbVE-nCyS8-OiV1mc-r4I9tNdi4C8-q4bf-BSWJ4&zhida_source=entity) 到目标 binary 的过程，先把从 ir 到目标指令如何生成的，当中步骤和 pass 了解的差不多，再可以参考某一平台，譬如官方的 [CPU0Target教程](https://zhida.zhihu.com/search?content_id=392137466&content_type=Answer&match_order=1&q=CPU0Target%E6%95%99%E7%A8%8B&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NTE3ODYwMjMsInEiOiJDUFUwVGFyZ2V05pWZ56iLIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MzkyMTM3NDY2LCJjb250ZW50X3R5cGUiOiJBbnN3ZXIiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.UDcXuzxClNanZ4kqd4dPBvufwhrQA02np35a1UkUIX8&zhida_source=entity)，去完成一个简单点的新后端的添加，包括指令，寄存器，调用约定，下降之类的一些操作的实现。重点！！！一定要有工程经验，知道如何去做。

3，之后就可以去看优化了，包括机器相关和机器无关，如循环优化，指令调度，这方面需要一些 [编译优化](https://zhida.zhihu.com/search?content_id=392137466&content_type=Answer&match_order=1&q=%E7%BC%96%E8%AF%91%E4%BC%98%E5%8C%96&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NTE3ODYwMjMsInEiOiLnvJbor5HkvJjljJYiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjozOTIxMzc0NjYsImNvbnRlbnRfdHlwZSI6IkFuc3dlciIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.xLkPACT2mHZS3p0vmnjdhV94utgNCoGZKijMtsbDjYI&zhida_source=entity) 的基础知识，编译器设计，编译原理，编译器优化，这些书，我是翻了几遍了。可以去抠 llvm 的公用优化 pass，熟悉其怎么做的，原理是什么。

4，关于一些基础知识，例如 c++ 的常用新特性，当然，写代码的时候多熟悉下，数据结构，常见的排序算法，图论算法之类的，会手写更好

突然发现我找工作时得到过 @LanTn 和 @CompilerCoder 的帮助，来回馈一下社会 学习 LLVM 第一步是了解编译器里的基本概念，把 LLVM 中的实现和概念联系起来，基本都不带改名的。之后就是以项目为导向，如果是本科生，可以考虑一下我读研的实验室，也可以考虑软件所 PLCT@小乖他爹 ，或者直接去他们那实习打工。LLVM 的东西太多，没有确定的目标是事倍功半。 做项目中会获得什么？ 在写 Pass 时实现功能时，需要解决各种稀奇古怪和不符合预期的问题，如： analysis 结果咋没更新呀？ 为啥 bb0 不支配/后支配 bb1 为啥这个计算挪不出循环 为啥这俩东西判断为 MayAlias 为啥这个优化没做，我要自己定制 pipeline，一个 pass 搞几遍！ 为啥老子被判断为 divergent？ 这都向量化不了？老子自己复现几篇 diao 炸的 SLP balabala！ LLVM error: unable to legalize/select tablegen 写错几个数字影响指令调度的效果，一把辛酸泪 以前有人问我：“兄弟，我拿你的 pass 去跑了一下，怎么死循环了？”，“可能是 IR 情况没考虑完吧……” … 也就是说，要完成本月的 kpi，不得不先翻课本看看经典的算法，再去结合 LLVM 的实现 可选: 编译期的 bug 比较好调，gdb/lldb 是你最好的老师，最困难的地方在于编出的程序运行错误或性能不符合预期 此时，你得 print-all，从超长的 IR 中推理出到底是哪个环节出错了，pipeline 就是多米诺骨牌，一个人一包烟，一台电脑过一天，最后发现原来是写错了几个数字或者一个 if，在论证猜想的过程中可能得学习很多 pass，我一直觉得，上游的代码也是人写的，指不定是他们写错了！ 以我校招面试经验看，一般来说，做到可选之前再懂点 cpp 就可以了。
