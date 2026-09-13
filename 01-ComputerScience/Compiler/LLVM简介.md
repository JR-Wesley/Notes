# LLVM

LLVM并非单一工具，而是包含**编译器框架、优化器、代码生成器、工具链（如Clang）** 的庞大生态，因此学习需先建立整体认知，再拆解细节。


### 编译原理

#### 1. 编译原理核心概念
LLVM的设计完全遵循编译原理的经典流程，需理解：
- **前端流程**：词法分析（Lexical Analysis）→ 语法分析（Syntax Analysis）→ 语义分析（Semantic Analysis）→ 中间代码生成（IR Generation）。
- **中端优化**：基于中间代码（IR）的跨平台优化（如常量传播、死代码消除、循环优化）。
- **后端生成**：目标平台相关优化（指令选择、寄存器分配、指令调度）→ 目标代码（汇编/机器码）。

#### 2. 现代C++

LLVM源码完全基于**C++17及以上标准**开发，大量使用模板、智能指针、Lambda、STL容器、RAII等特性，需熟练掌握：
- 核心特性：`std::unique_ptr`/`std::shared_ptr`、模板元编程、 constexpr、范围for循环、Lambda表达式。
- 内存管理：避免野指针、理解RAII机制（LLVM中大量对象通过`llvm::OwningPtr`或智能指针管理）。

#### 3. 目标平台基础

若需深入LLVM后端（代码生成），需了解目标平台的基础架构，如：
- x86_64/ARM的寄存器结构、指令集（如x86的`mov`/`add`，ARM的`ldr`/`str`）。
- 目标文件格式（ELF、Mach-O、PE）的基本结构（如节区、符号表、重定位表）。

### LLVM整体认知

此阶段目标是：**理解LLVM的生态组成、核心概念（尤其是IR），并能通过工具链实操感受其工作流程**。

#### 1. 先搞懂：LLVM生态的核心组件

LLVM生态包含多个工具和库，需先明确各组件的定位，避免混淆

| 组件          | 核心作用                                                                 |
|---------------|--------------------------------------------------------------------------|
| **Clang**     | LLVM的C/C++/Objective-C前端（替代GCC前端），负责生成LLVM IR。             |
| **LLVM IR**   | 跨平台中间表示（Intermediate Representation），是LLVM各模块的“通用语言”。 |
| **LLVM Core** | 核心框架（库），提供IR的创建、修改、分析接口（如`Module`/`Function`类）。 |
| **Opt**       | 优化器工具，基于Pass机制对IR进行优化（如`opt -O2 input.ll -o output.ll`）。|
| **LLC**       | 代码生成器，将优化后的IR转换为目标平台的汇编代码（如`llc output.ll -o output.s`）。 |
| **LLDB**      | 基于LLVM的调试器，支持多平台调试（替代GDB）。                             |
| **TableGen**  | LLVM的代码生成工具，用于自动生成后端指令集描述、Pass声明等（核心工具）。  |

#### 2. 实操：用LLVM工具链跑通“源码→可执行文件”流程

通过实际操作理解LLVM的工作链路，步骤如下：
1. **安装LLVM工具链**：
   - 简单方式：通过包管理器安装（如Ubuntu `sudo apt install llvm clang`，macOS `brew install llvm`）。
   - 推荐方式：从[LLVM官网](https://llvm.org/)下载源码，本地编译（需CMake，编译时可开启`-DLLVM_BUILD_EXAMPLES=ON`生成示例代码）。
   
2. **跑通完整流程**（以C代码为例）：
   ```bash
   # 1. 源码→LLVM IR（.ll文本格式）
   clang -S -emit-llvm test.c -o test.ll
   
   # 2. IR优化（-O2级别，生成优化后的IR）
   opt -O2 test.ll -o test_opt.ll
   
   # 3. 优化后的IR→目标平台汇编（.s）
   llc test_opt.ll -o test.s
   
   # 4. 汇编→可执行文件（链接阶段，用clang链接）
   clang test.s -o test
   
   # 运行可执行文件
   ./test
   ```
   关键：查看`test.ll`和`test_opt.ll`的差异，理解优化对IR的影响（如冗余变量被消除、循环被简化）。

#### 3. 核心：深入理解LLVM IR
IR是LLVM的“灵魂”——前端生成IR，中端优化IR，后端消费IR。需掌握IR的**语法、类型系统、核心指令**：
- 语法基础：IR是强类型的中间语言，所有指令需明确类型（如`i32`表示32位整数，`double`表示双精度浮点数）。
  示例IR（实现`add(1,2)`）：
  ```llvm
  ; 定义模块（每个IR文件对应一个Module）
  module asm ".file \"test.c\""
  
  ; 定义函数add，参数为i32 %a、i32 %b，返回i32
  define dso_local i32 @add(i32 %a, i32 %b) #0 {
  entry:
    ; 计算%a + %b，结果存入%add
    %add = add nsw i32 %a, %b
    ; 返回%add
    ret i32 %add
  }
  ```
- 核心概念：
  - `Module`：IR的顶层容器，包含函数、全局变量、类型定义。
  - `Function`：函数定义，包含多个`BasicBlock`（基本块，满足“单入口、单出口”）。
  - `BasicBlock`：包含一系列`Instruction`（指令，如`add`/`ret`/`br`）。
- 学习资料：官方文档《[LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html)》（必看，覆盖IR的所有细节）。


### 三、进阶阶段：深入LLVM核心模块（2-4个月）
此阶段目标是：**基于LLVM Core库进行开发，掌握“IR操作、Pass开发、后端适配”等核心能力**，需结合源码阅读和代码实践。

#### 1. 核心模块一：LLVM Core（IR操作）
LLVM Core提供C++ API用于创建、修改、分析IR，是所有LLVM开发的基础。需掌握以下核心类的使用：
- `llvm::Module`：管理整个IR模块，如创建函数（`Module::getOrInsertFunction`）、添加全局变量（`Module::getOrInsertGlobal`）。
- `llvm::Function`：表示函数，如获取函数的基本块（`Function::begin()`/`end()`）、添加基本块（`BasicBlock::Create`）。
- `llvm::BasicBlock`：表示基本块，如添加指令（`IRBuilder`工具类）、设置分支（`BranchInst::Create`）。
- `llvm::IRBuilder`：便捷创建指令的工具类（如`CreateAdd`生成`add`指令，`CreateRet`生成`ret`指令）。

**实践任务**：用LLVM Core API手动构建一个简单的IR模块（如实现`add(1,2)`），代码示例：
```cpp
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

int main() {
  // 1. 创建Module（上下文默认）
  auto M = std::make_unique<Module>("MyFirstModule", getGlobalContext());
  
  // 2. 定义函数类型：i32 (i32, i32)
  Type *I32Ty = Type::getInt32Ty(getGlobalContext());
  FunctionType *FTy = FunctionType::get(I32Ty, {I32Ty, I32Ty}, false);
  
  // 3. 在Module中创建函数add
  Function *AddFunc = Function::Create(FTy, Function::ExternalLinkage, "add", M.get());
  
  // 4. 为函数参数命名（可选，但便于调试）
  auto ArgIt = AddFunc->arg_begin();
  Argument *A = &*ArgIt++; A->setName("a");
  Argument *B = &*ArgIt++; B->setName("b");
  
  // 5. 创建基本块（entry）并添加到函数
  BasicBlock *EntryBB = BasicBlock::Create(getGlobalContext(), "entry", AddFunc);
  IRBuilder<> Builder(EntryBB);
  
  // 6. 生成指令：%add = add nsw i32 %a, %b
  Value *Add = Builder.CreateAdd(A, B, "add");
  
  // 7. 生成返回指令：ret i32 %add
  Builder.CreateRet(Add);
  
  // 8. 打印IR到标准输出
  M->print(outs(), nullptr);
  return 0;
}
```
编译运行后，会输出与手动编写的`add`函数IR一致的内容，以此验证API的使用。

#### 2. 核心模块二：Pass机制（中端优化）
LLVM的优化（如`-O2`/`-O3`）是通过**Pass**（分析/转换插件）实现的。Pass分为两类：
- **分析Pass（Analysis Pass）**：不修改IR，仅收集信息（如“函数的控制流图”“变量的使用情况”），如`DominatorTreeAnalysis`（支配树分析）。
- **转换Pass（Transformation Pass）**：修改IR以优化代码（如“死代码消除”“常量传播”），如`DeadCodeEliminationPass`。

**实践任务：编写一个简单的分析Pass**（统计函数中的`add`指令数量）：
1. 定义Pass类，继承`FunctionPass`（针对单个函数的Pass）：
   ```cpp
   #include "llvm/Pass.h"
   #include "llvm/IR/Function.h"
   #include "llvm/IR/Instruction.h"
   #include "llvm/Support/raw_ostream.h"

   using namespace llvm;

   namespace {
     // 定义Pass类，名称为"CountAddInst"
     struct CountAddInst : public FunctionPass {
       static char ID; // Pass的唯一标识（必须定义）
       CountAddInst() : FunctionPass(ID) {}

       // 核心方法：对每个函数执行分析
       bool runOnFunction(Function &F) override {
         int AddCount = 0;
         // 遍历函数的所有基本块
         for (BasicBlock &BB : F) {
           // 遍历基本块的所有指令
           for (Instruction &I : BB) {
             // 判断指令是否为AddInst（add指令）
             if (isa<AddInst>(&I)) {
               AddCount++;
             }
           }
         }
         // 输出结果
         errs() << "Function " << F.getName() << " has " << AddCount << " add instructions\n";
         return false; // 分析Pass不修改IR，返回false
       }
     };
   }

   // 初始化Pass的ID（必须）
   char CountAddInst::ID = 0;

   // 注册Pass（让opt工具能识别）
   static RegisterPass<CountAddInst> X("count-add", "Count Add Instructions Pass");
   ```
2. 编译Pass为动态库（需链接LLVM Core库，通过CMake配置）。
3. 用`opt`工具加载Pass并运行：
   ```bash
   opt -load ./libCountAddInst.so -count-add test.ll -o /dev/null
   ```
   运行后会输出每个函数的`add`指令数量，以此掌握Pass的开发流程。

#### 3. 核心模块三：后端开发（可选，偏底层）
若需为新的目标平台（如自定义CPU）适配LLVM，需了解后端的核心流程：
1. **目标平台描述**：用TableGen定义目标架构的寄存器、指令集（如`X86.td`为x86架构的描述文件）。
2. **指令选择**：将IR指令映射为目标平台的机器指令（通过“指令选择模式”匹配）。
3. **寄存器分配**：将IR中的虚拟寄存器分配到物理寄存器（LLVM提供`Greedy`/`PBQP`等分配算法）。
4. **指令调度**：调整指令顺序以优化CPU流水线执行效率。

学习资料：官方文档《[LLVM Backend Developer's Guide](https://llvm.org/docs/Backend/index.html)》、LLVM源码中`llvm/lib/Target`目录下的目标平台实现（如`X86`/`ARM`文件夹）。


### 四、实战阶段：通过项目深化理解（3-6个月）
理论学习后，需通过实际项目巩固能力，推荐以下方向：
1. **自定义语言前端**：为简单语言（如类C迷你语言）编写前端，生成LLVM IR。
   - 步骤：词法分析（用Flex）→ 语法分析（用Bison）→ 语义分析→ 生成IR（调用LLVM Core API）。
   - 参考：LLVM官方示例`Kaleidoscope`（[教程链接](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/index.html)），手把手教你实现一个支持变量、函数、循环的语言前端。

2. **开发实用Pass**：
   - 分析Pass：如“检测函数中的未使用参数”“生成函数的调用关系图”。
   - 转换Pass：如“循环展开优化”“移除冗余的空函数调用”。

3. **LLVM工具二次开发**：
   - 基于Clang开发静态分析工具（如检测C代码中的内存泄漏，利用Clang的AST接口）。
   - 基于LLDB开发自定义调试命令（如针对特定框架的调试辅助命令）。


### 五、学习资源汇总
1. **官方文档（首选）**：
   - [LLVM Documentation](https://llvm.org/docs/)：包含核心概念、API手册、教程。
   - [Clang Documentation](https://clang.llvm.org/docs/)：Clang的使用和开发指南。

2. **书籍**：
   - 《LLVM Cookbook》：实操性强，涵盖Pass开发、后端适配、工具链使用。
   - 《Advanced Compiler Design and Implementation》（鲸书）：深入编译器优化技术，与LLVM中端优化原理相通。

3. **源码阅读**：
   - LLVM源码仓库（[GitHub](https://github.com/llvm/llvm-project)）：重点阅读`llvm/include/llvm/IR`（核心IR类）、`llvm/lib/IR`（IR实现）、`llvm/lib/Transforms`（Pass实现）。
   - 推荐从简单模块入手（如`llvm/lib/Transforms/Scalar/DeadCodeElimination.cpp`），理解Pass的逻辑。

4. **社区与会议**：
   - LLVM邮件列表（[llvm-dev](https://lists.llvm.org/mailman/listinfo/llvm-dev)）：提问、交流的核心渠道。
   - LLVM Dev Meeting：每年举办的技术会议，演讲内容涵盖LLVM最新进展和实践（[视频回放](https://llvm.org/devmtg/)）。


### 六、常见误区与建议
1. **不要跳过编译原理**：若直接学LLVM API，会难以理解“为何要设计`Module`/`BasicBlock`”“Pass的作用是什么”，基础不牢会导致后续学习卡顿。
2. **不要畏惧源码**：LLVM源码量大（数百万行），但核心模块逻辑清晰，建议从“小功能”入手（如阅读`AddInst`的实现），逐步扩展。
3. **多调试少空想**：编写Pass或IR生成代码时，用`llvm::errs()`打印中间结果，或用LLVM的调试工具（如`llc -debug`）查看后端流程，比单纯看文档更有效。
4. **跟踪版本迭代**：LLVM更新频繁（每6个月一个版本），部分API可能变更（如旧Pass管理器逐步被新Pass管理器替代），建议参考对应版本的文档和源码。


总之，学习LLVM的核心是“**理论→实操→源码→项目**”的循环：先理解编译原理，再通过工具链实操建立认知，接着深入核心模块开发，最后通过项目落地巩固。LLVM生态庞大，但只要循序渐进，就能逐步掌握其核心能力。