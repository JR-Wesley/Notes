## 7. Abstracting the ABI

### English Original

> Rather than expose details of a particular calling convention, stack layout, and Application Binary Interface (ABI), PTX provides a slightly higher-level abstraction and supports multiple ABI implementations. In this section, we describe the features of PTX needed to achieve this hiding of the ABI. These include syntax for function definitions, function calls, parameter passing, and memory allocated on the stack (`alloca`).
>
> Refer to PTX Writers Guide to Interoperability for details on generating PTX compliant with Application Binary Interface (ABI) for the CUDA® architecture.

### 中文翻译

PTX 不直接暴露某一种 calling convention、stack layout 和 Application Binary Interface（ABI）的细节，而是提供稍高层次的抽象，并支持多种 ABI implementation。本节介绍 PTX 为隐藏 ABI 而提供的功能，包括 function definition、function call、parameter passing，以及在 stack 上分配 memory 的 `alloca` 语法。

关于如何生成符合 CUDA® architecture ABI 的 PTX，参阅 PTX Writers Guide to Interoperability。

### 重点解读

PTX 的 function interface 描述“传什么、返回什么”，但不承诺这些值最终一定放在 physical register 或 stack 的某个固定位置。具体映射由 backend 按目标 ABI 决定，这正是 PTX 能跨 GPU generation 和 ABI implementation 保持可移植性的关键。

### 7.1 Function Declarations and Definitions

#### English Original

> In PTX, functions are declared and defined using the `.func` directive. A function declaration specifies an optional list of return parameters, the function name, and an optional list of input parameters; together these specify the function’s interface, or prototype. A function definition specifies both the interface and the body of the function. A function must be declared or defined prior to being called.
>
> The simplest function has no parameters or return values, and is represented in PTX as follows:

```ptx
.func foo
{
    ...
    ret;
}

    ...
    call foo;
    ...
```

> Here, execution of the `call` instruction transfers control to `foo`, implicitly saving the return address. Execution of the `ret` instruction within `foo` transfers control to the instruction following the call.
>
> Scalar and vector base-type input and return parameters may be represented simply as register variables. At the call, arguments may be register variables or constants, and return values may be placed directly into register variables. The arguments and return variables at the call must have type and size that match the callee’s corresponding formal parameters.
>
> Example

```ptx
.func (.reg .u32 %res) inc_ptr ( .reg .u32 %ptr, .reg .u32 %inc )
{
    add.u32 %res, %ptr, %inc;
    ret;
}

    ...
    call (%r1), inc_ptr, (%r1,4);
    ...
```

> When using the ABI, `.reg` state space parameters must be at least 32-bits in size. Subword scalar objects in the source language should be promoted to 32-bit registers in PTX, or use `.param` state space byte arrays described next.
>
> Objects such as C structures and unions are flattened into registers or byte arrays in PTX and are represented using `.param` space memory. For example, consider the following C structure, passed by value to a function:

```c
struct {
    double dbl;
    char   c[4];
};
```

> In PTX, this structure will be flattened into a byte array. Since memory accesses are required to be aligned to a multiple of the access size, the structure in this example will be a 12 byte array with 8 byte alignment so that accesses to the `.f64` field are aligned. The `.param` state space is used to pass the structure by value:
>
> Example

```ptx
.func (.reg .s32 out) bar (.reg .s32 x, .param .align 8 .b8 y[12])
{
    .reg .f64 f1;
    .reg .b32 c1, c2, c3, c4;
    ...
    ld.param.f64 f1, [y+0];
    ld.param.b8  c1, [y+8];
    ld.param.b8  c2, [y+9];
    ld.param.b8  c3, [y+10];
    ld.param.b8  c4, [y+11];
    ...
    ... // computation using x,f1,c1,c2,c3,c4;
}

{
     .param .b8 .align 8 py[12];
     ...
     st.param.b64 [py+ 0], %rd;
     st.param.b8  [py+ 8], %rc1;
     st.param.b8  [py+ 9], %rc2;
     st.param.b8  [py+10], %rc1;
     st.param.b8  [py+11], %rc2;
     // scalar args in .reg space, byte array in .param space
     call (%out), bar, (%x, py);
     ...
}
```

> In this example, note that `.param` space variables are used in two ways. First, a `.param` variable `y` is used in function definition bar to represent a formal parameter. Second, a `.param` variable `py` is declared in the body of the calling function and used to set up the structure being passed to bar.
>
> The following is a conceptual way to think about the `.param` state space use in device functions.
>
> For a caller,
>
> - The `.param` state space is used to set values that will be passed to a called function and/or to receive return values from a called function. Typically, a `.param` byte array is used to collect together fields of a structure being passed by value.
>
> For a callee,
>
> - The `.param` state space is used to receive parameter values and/or pass return values back to the caller.
>
> The following restrictions apply to parameter passing.
>
> For a caller,
>
> - Arguments may be `.param` variables, `.reg` variables, or constants.
> - In the case of `.param` space formal parameters that are byte arrays, the argument must also be a `.param` space byte array with matching type, size, and alignment. A `.param` argument must be declared within the local scope of the caller.
> - In the case of `.param` space formal parameters that are base-type scalar or vector variables, the corresponding argument may be either a `.param` or `.reg` space variable with matching type and size, or a constant that can be represented in the type of the formal parameter.
> - In the case of `.reg` space formal parameters, the corresponding argument may be either a `.param` or `.reg` space variable of matching type and size, or a constant that can be represented in the type of the formal parameter.
> - In the case of `.reg` space formal parameters, the register must be at least 32-bits in size.
> - All `st.param` instructions used for passing arguments to function call must immediately precede the corresponding `call` instruction and `ld.param` instruction used for collecting return value must immediately follow the `call` instruction without any control flow alteration. `st.param` and `ld.param` instructions used for argument passing cannot be predicated. This enables compiler optimization and ensures that the `.param` variable does not consume extra space in the caller’s frame beyond that needed by the ABI. The `.param` variable simply allows a mapping to be made at the call site between data that may be in multiple locations (e.g., structure being manipulated by caller is located in registers and memory) to something that can be passed as a parameter or return value to the callee.
>
> For a callee,
>
> - Input and return parameters may be `.param` variables or `.reg` variables.
> - Parameters in `.param` memory must be aligned to a multiple of 1, 2, 4, 8, or 16 bytes.
> - Parameters in the `.reg` state space must be at least 32-bits in size.
> - The `.reg` state space can be used to receive and return base-type scalar and vector values, including sub-word size objects when compiling in non-ABI mode. Supporting the `.reg` state space provides legacy support.
>
> Note that the choice of `.reg` or `.param` state space for parameter passing has no impact on whether the parameter is ultimately passed in physical registers or on the stack. The mapping of parameters to physical registers and stack locations depends on the ABI definition and the order, size, and alignment of parameters.

#### 中文翻译

PTX 使用 `.func` directive 声明和定义 function。Function declaration 指定可选的 return-parameter list、function name 和可选的 input-parameter list，三者共同构成 function interface 或 prototype。Function definition 同时包含 interface 与 function body。Function 在被调用之前必须已经声明或定义。

最简单的 function 没有 parameter 和 return value。执行 `call` 会把控制流转移到 `foo`，同时隐式保存 return address；`foo` 中的 `ret` 则把控制流转回 `call` 后面的 instruction。

Scalar 和 vector base-type input/return parameter 可以直接表示为 register variable。调用时，argument 可以是 register variable 或 constant，return value 可以直接写入 register variable。Call site 的 argument 与 return variable 必须在 type 和 size 上匹配 callee 对应的 formal parameter。`inc_ptr` 示例接收 `%ptr` 和 `%inc`，返回 `%res`；call site 把 `%r1` 与 immediate `4` 传入，并把结果写回 `%r1`。

使用 ABI 时，`.reg` state-space parameter 至少必须为 32 bit。Source language 中小于一个 word 的 scalar object 应提升为 PTX 32-bit register，或者改用 `.param` state-space byte array。

C structure、union 等 object 会在 PTX 中扁平化为 register 或 byte array，并使用 `.param` space memory 表示。示例 structure 由一个 `double` 和四个 `char` 组成，按值传递给 function。在 PTX 中它被展开为 12-byte array；由于 memory access address 必须按 access size 对齐，该 array 采用 8-byte alignment，以保证 `.f64` field 的访问正确对齐。

示例中的 `.param` variable 有两种用途：callee `bar` 定义中的 `y` 是 formal parameter；caller body 中声明的 `py` 用于组装即将传给 `bar` 的 structure。Caller 先使用 `st.param` 把各 field 写入 `py`，再通过 `call (%out), bar, (%x, py)` 传递 scalar argument 与 byte-array argument；callee 使用 `ld.param` 逐 field 读取。

从概念上看，caller 使用 `.param` state space 设置传给 callee 的 value，或接收 callee 返回的 value；按值传递 structure 时，通常使用 `.param` byte array 汇集各 field。Callee 则通过 `.param` 接收 parameter value，或把 return value 传回 caller。

Caller 侧的 parameter-passing restriction 如下：

- Argument 可以是 `.param` variable、`.reg` variable 或 constant。
- 若 formal parameter 是 `.param` byte array，argument 也必须是 type、size、alignment 均匹配的 `.param` byte array，并且必须在 caller local scope 内声明。
- 若 formal parameter 是 `.param` base-type scalar/vector，argument 可以是 type、size 匹配的 `.param`/`.reg` variable，也可以是能够由 formal-parameter type 表示的 constant。
- 若 formal parameter 位于 `.reg` space，argument 同样可以是 type、size 匹配的 `.param`/`.reg` variable，或可由该 type 表示的 constant；该 register 至少为 32 bit。
- 为 function call 传递 argument 的全部 `st.param` 必须紧邻并位于相应 `call` 之前；收集 return value 的 `ld.param` 必须紧邻并位于 `call` 之后，中间不能出现 control-flow alteration。这些 `st.param` 和 `ld.param` 不能被 predicate guard 控制。

上述相邻性与禁止 predication 的限制使 compiler 能够优化，并保证 `.param` variable 不会在 caller frame 中占用超出 ABI 所需的额外空间。`.param` variable 只是 call site 的映射媒介，把可能分散在 register 和 memory 中的数据组织成可传递给 callee 的 parameter 或 return value。

Callee 侧的 restriction 如下：input 和 return parameter 可以是 `.param` 或 `.reg` variable；`.param` memory 中的 parameter 必须按 1、2、4、8 或 16 byte 的整数倍对齐；`.reg` parameter 至少为 32 bit。在 non-ABI mode 中，`.reg` 还可以接收和返回 base-type scalar/vector，包括 sub-word-size object，这主要用于 legacy support。

选择 `.reg` 还是 `.param` 传递 parameter，并不决定它最终位于 physical register 还是 stack。实际映射取决于 ABI definition，以及 parameter 的 order、size 和 alignment。

#### 重点解读

`.param` 在 source-level PTX 中像一块显式 parameter buffer，但它不是对最终物理 stack slot 的承诺。Backend 可以把其中某些值分配到 physical register，也可以放到 stack；`.reg` formal parameter 同样不保证一定使用 physical register。

`st.param → call → ld.param` 的严格邻接关系形成一个不可随意打断的 call-site protocol。若在其中插入 branch、对这些 instruction 加 predicate，或让 byte-array argument 的 type/size/alignment 不匹配，compiler 就无法安全地把逻辑 parameter object 映射到 ABI location。

#### 7.1.1 Changes from PTX ISA Version 1.x

##### English Original

> In PTX ISA version 1.x, formal parameters were restricted to `.reg` state space, and there was no support for array parameters. Objects such as C structures were flattened and passed or returned using multiple registers. PTX ISA version 1.x supports multiple return values for this purpose.
>
> Beginning with PTX ISA version 2.0, formal parameters may be in either `.reg` or `.param` state space, and `.param` space parameters support arrays. For targets `sm_20` or higher, PTX restricts functions to a single return value, and a `.param` byte array should be used to return objects that do not fit into a register. PTX continues to support multiple return registers for `sm_1x` targets.
>
> Note
>
> PTX implements a stack-based ABI only for targets `sm_20` or higher.
>
> PTX ISA versions prior to 3.0 permitted variables in `.reg` and `.local` state spaces to be defined at module scope. When compiling to use the ABI, PTX ISA version 3.0 and later disallows module-scoped `.reg` and `.local` variables and restricts their use to within function scope. When compiling without use of the ABI, module-scoped `.reg` and `.local` variables are supported as before. When compiling legacy PTX code (ISA versions prior to 3.0) containing module-scoped `.reg` or `.local` variables, the compiler silently disables use of the ABI.

##### 中文翻译

在 PTX ISA 1.x 中，formal parameter 只能位于 `.reg` state space，且不支持 array parameter。C structure 等 object 会被扁平化，并通过多个 register 传递或返回；PTX ISA 1.x 为此支持多个 return value。

从 PTX ISA 2.0 开始，formal parameter 可以位于 `.reg` 或 `.param` state space，`.param` parameter 也支持 array。对于 `sm_20+` target，PTX 将 function 限制为单一 return value；无法放入一个 register 的 object 应通过 `.param` byte array 返回。`sm_1x` target 仍支持多个 return register。

PTX 仅为 `sm_20+` target 实现 stack-based ABI。

PTX ISA 3.0 之前允许在 module scope 定义 `.reg`、`.local` variable。PTX ISA 3.0 及以后使用 ABI 编译时，不允许 module-scoped `.reg`/`.local` variable，只能在 function scope 内使用；不使用 ABI 编译时仍支持原有行为。若编译包含 module-scoped `.reg` 或 `.local` variable 的 legacy PTX code（ISA 3.0 之前），compiler 会静默禁用 ABI。

##### 重点解读

这一节虽然按版本说明演进，但直接影响旧 PTX 的兼容行为：module-scoped `.reg`/`.local` 可能让 compiler 自动退出 ABI mode。分析旧代码时，不能只看 `.version`，还要检查这些 variable 的 scope。

### 7.2 Variadic Functions

#### English Original

> Note
>
> Support for variadic functions which was unimplemented has been removed from the spec.
>
> PTX version 6.0 supports passing unsized array parameter to a function which can be used to implement variadic functions.
>
> Refer to [Kernel and Function Directives: `.func`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#kernel-and-function-directives-func) for details.

#### 中文翻译

规范中曾经存在但并未实现的 variadic-function support 已被移除。PTX 6.0 支持向 function 传递 unsized array parameter，可利用这一机制实现 variadic function。详细信息参阅 Kernel and Function Directives: `.func`。

#### 重点解读

这里不是说 PTX 存在类似 C `...` 的原生 variadic calling convention；可行方式是把可变参数显式打包为 unsized array parameter，再由 callee 按约定解析。

### 7.3 Alloca

#### English Original

> PTX provides `alloca` instruction for allocating storage at runtime on the per-thread local memory stack. The allocated stack memory can be accessed with `ld.local` and `st.local` instructions using the pointer returned by `alloca`.
>
> In order to facilitate deallocation of memory allocated with `alloca`, PTX provides two additional instructions: `stacksave` which allows reading the value of stack pointer in a local variable, and `stackrestore` which can restore the stack pointer with the saved value.
>
> `alloca`, `stacksave`, and `stackrestore` instructions are described in [Stack Manipulation Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#stack-manipulation-instructions).

#### 中文翻译

PTX 提供 `alloca` instruction，用于在 runtime 从 per-thread local-memory stack 分配 storage。程序使用 `alloca` 返回的 pointer，通过 `ld.local` 和 `st.local` 访问这块 stack memory。

为了便于释放 `alloca` 分配的 memory，PTX 还提供两条 instruction：`stacksave` 把 stack pointer 的当前值读入 local variable，`stackrestore` 使用保存的值恢复 stack pointer。

`alloca`、`stacksave` 和 `stackrestore` 的详细定义参阅 Stack Manipulation Instructions。

#### 重点解读

`alloca` 分配的是每个 thread 私有的 local-memory stack，不是 CTA shared memory。典型的成对使用方式是在进入临时分配区域前 `stacksave`，完成后 `stackrestore`；它恢复的是 stack pointer，从而一次回收其后的动态分配，而不是逐块释放。

