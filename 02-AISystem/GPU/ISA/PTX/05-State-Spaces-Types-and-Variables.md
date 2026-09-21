## 5. State Spaces, Types, and Variables

#### English Original

> While the specific resources available in a given target GPU will vary, the kinds of resources will be common across platforms, and these resources are abstracted in PTX through state spaces and data types.

#### 中文翻译

虽然不同目标 GPU 提供的具体资源会有所不同，但资源种类在各个平台之间具有共性；PTX 通过 state space 和 data type 对这些资源进行抽象。

#### 重点解读

State space 描述数据位于哪类逻辑存储区域以及谁能访问，data type 描述数据的解释方式和位宽。两者共同构成 PTX 变量和指令操作数的类型系统：同一个 bit pattern 放在不同 state space 中具有不同的作用域与访问指令，使用不同 type 又会产生不同的算术和转换语义。

### 5.1 State Spaces

#### English Original

> A state space is a storage area with particular characteristics. All variables reside in some state space. The characteristics of a state space include its size, addressability, access speed, access rights, and level of sharing between threads.
>
> The state spaces defined in PTX are a byproduct of parallel programming and graphics programming. The list of state spaces is shown in Table 6, and properties of state spaces are shown in Table 7.

**Table 6. State Spaces**

| Name | Description |
| --- | --- |
| `.reg` | Registers, fast. |
| `.sreg` | Special registers. Read-only; pre-defined; platform-specific. |
| `.const` | Shared, read-only memory. |
| `.global` | Global memory, shared by all threads. |
| `.local` | Local memory, private to each thread. |
| `.param` | Kernel parameters, defined per-grid; or Function or local parameters, defined per-thread. |
| `.shared` | Addressable memory, defined per CTA, accessible to all threads in the cluster throughout the lifetime of the CTA that defines it. |
| `.tex` | Global texture memory (deprecated). |

**Table 7. Properties of State Spaces**

| Name | Addressable | Initializable | Access | Sharing |
| --- | --- | --- | --- | --- |
| `.reg` | No | No | R/W | per-thread |
| `.sreg` | No | No | RO | per-CTA |
| `.const` | Yes | Yes¹ | RO | per-grid |
| `.global` | Yes | Yes¹ | R/W | Context |
| `.local` | Yes | No | R/W | per-thread |
| `.param` (as input to kernel) | Yes² | No | RO | per-grid |
| `.param` (used in functions) | Restricted³ | No | R/W | per-thread |
| `.shared` | Yes | No | R/W | per-cluster⁵ |
| `.tex` | No⁴ | Yes, via driver | RO | Context |

> Notes:
>
> 1. Variables in `.const` and `.global` state spaces are initialized to zero by default.
> 2. Accessible only via the `ld.param{::entry}` instruction. Address may be taken via `mov` instruction.
> 3. Accessible via `ld.param{::func}` and `st.param{::func}` instructions. Device function input and return parameters may have their address taken via `mov`; the parameter is then located on the stack frame and its address is in the `.local` state space.
> 4. Accessible only via the `tex` instruction.
> 5. Visible to the owning CTA and other active CTAs in the cluster.

#### 中文翻译

State space 是具有特定性质的存储区域。所有变量都位于某个 state space 中。State space 的性质包括容量、是否可寻址、访问速度、访问权限，以及在线程之间的共享层级。

PTX 中定义的 state space 来源于并行编程和图形编程的需求。表 6 列出 state space，表 7 列出它们的属性；表格及脚注按原文完整保留，不另行翻译。

#### 重点解读

State space 不是单纯的“存储速度分类”。例如 `.reg` 不可取地址，`.local` 虽然线程私有却属于可寻址 memory，`.shared` 的生命周期和共享范围绑定 CTA/cluster，`.param` 的读写权限与共享范围又取决于它用于 kernel 还是 device function。分析 PTX 变量时，应同时检查 addressability、access、sharing 和 lifetime。

#### 5.1.1 Register State Space

##### English Original

> Registers (`.reg` state space) are fast storage locations. The number of registers is limited, and will vary from platform to platform. When the limit is exceeded, register variables will be spilled to memory, causing changes in performance. For each architecture, there is a recommended maximum number of registers to use (see the CUDA Programming Guide for details).
>
> Registers may be typed (signed integer, unsigned integer, floating point, predicate) or untyped. Register size is restricted; aside from predicate registers which are 1-bit, scalar registers have a width of 8-, 16-, 32-, 64-, or 128-bits, and vector registers have a width of 16-, 32-, 64-, or 128-bits. The most common use of 8-bit registers is with `ld`, `st`, and `cvt` instructions, or as elements of vector tuples.
>
> Registers differ from the other state spaces in that they are not fully addressable, i.e., it is not possible to refer to the address of a register. When compiling to use the Application Binary Interface (ABI), register variables are restricted to function scope and may not be declared at module scope. When compiling legacy PTX code (ISA versions prior to 3.0) containing module-scoped `.reg` variables, the compiler silently disables use of the ABI. Registers may have alignment boundaries required by multi-word loads and stores.

##### 中文翻译

Register（`.reg` state space）是高速存储位置。寄存器数量有限，并且随平台而变化。超出限制时，register variable 会 spill 到 memory，从而影响性能。每种架构都有建议使用的最大寄存器数量，详细信息参阅 CUDA Programming Guide。

Register 可以是有类型的，例如 signed integer、unsigned integer、floating point 和 predicate，也可以是 untyped。Register size 受到限制：predicate register 为 1 bit；除此之外，scalar register 的宽度可以是 8、16、32、64 或 128 bit，vector register 的宽度可以是 16、32、64 或 128 bit。8-bit register 最常见于 `ld`、`st`、`cvt` 指令，或作为 vector tuple 的元素。

Register 与其他 state space 的不同之处在于它不能被完整寻址，即不能引用一个 register 的地址。使用 Application Binary Interface（ABI）编译时，register variable 只能在 function scope 中声明，不能在 module scope 中声明。编译包含 module-scoped `.reg` variable 的旧版 PTX 代码（ISA 3.0 之前）时，编译器会静默禁用 ABI。Register 还可能需要满足 multi-word load/store 要求的对齐边界。

##### 重点解读

`.reg` 同时受到容量和 addressability 两类约束。寄存器过多会降低 SM 可同时驻留的线程或 block 数，超过后端可分配范围还可能发生 spilling；寄存器又没有可供普通指针引用的地址，因此“把寄存器地址传给函数”并不是合法 PTX 操作。

#### 5.1.2 Special Register State Space

##### English Original

> The special register (`.sreg`) state space holds predefined, platform-specific registers, such as grid, cluster, CTA, and thread parameters, clock counters, and performance monitoring registers. All special registers are predefined.

##### 中文翻译

Special register（`.sreg`）state space 保存预定义且与平台相关的 register，例如 grid、cluster、CTA 和 thread 参数、时钟计数器及性能监控 register。所有 special register 都是预定义的。

#### 5.1.3 Constant State Space

##### English Original

> The constant (`.const`) state space is a read-only memory initialized by the host. Constant memory is accessed with a `ld.const` instruction. Constant memory is restricted in size, currently limited to 64 KB which can be used to hold statically-sized constant variables. There is an additional 640 KB of constant memory, organized as ten independent 64 KB regions. The driver may allocate and initialize constant buffers in these regions and pass pointers to the buffers as kernel function parameters. Since the ten regions are not contiguous, the driver must ensure that constant buffers are allocated so that each buffer fits entirely within a 64 KB region and does not span a region boundary.
>
> Statically-sized constant variables have an optional variable initializer; constant variables with no explicit initializer are initialized to zero by default. Constant buffers allocated by the driver are initialized by the host, and pointers to such buffers are passed to the kernel as parameters. See the description of kernel parameter attributes in Kernel Function Parameter Attributes for more details on passing pointers to constant buffers as kernel parameters.

##### 中文翻译

Constant（`.const`）state space 是由 host 初始化的只读 memory，通过 `ld.const` 指令访问。Constant memory 容量受限：当前有 64 KB 可用于保存静态大小的 constant variable；另外还有 640 KB constant memory，由十个相互独立的 64 KB 区域构成。Driver 可以在这些区域中分配并初始化 constant buffer，再把 buffer pointer 作为 kernel function parameter 传入。由于十个区域并不连续，driver 必须保证每个 constant buffer 完整落在一个 64 KB 区域内，不能跨越区域边界。

静态大小的 constant variable 可以带有可选 initializer；没有显式 initializer 的 constant variable 默认初始化为零。Driver 分配的 constant buffer 由 host 初始化，指向这些 buffer 的 pointer 作为参数传给 kernel。有关把 constant buffer pointer 作为 kernel parameter 传递的更多信息，参阅 Kernel Function Parameter Attributes。

##### 重点解读

`.const` 的语义重点是 host 初始化、device 只读和容量受限。静态 constant variable 与 driver 分配的 constant buffer 属于两条不同使用路径；后者通过 kernel pointer parameter 传递，不能因为都位于 constant memory 就假定地址布局连续。

##### 5.1.3.1 Banked Constant State Space (deprecated)

###### English Original

> Previous versions of PTX exposed constant memory as a set of eleven 64 KB banks, with explicit bank numbers required for variable declaration and during access.
>
> Prior to PTX ISA version 2.2, the constant memory was organized into fixed size banks. There were eleven 64 KB banks, and banks were specified using the `.const[bank]` modifier, where bank ranged from 0 to 10. If no bank number was given, bank zero was assumed.
>
> By convention, bank zero was used for all statically-sized constant variables. The remaining banks were used to declare incomplete constant arrays (as in C, for example), where the size is not known at compile time. For example, the declaration

```ptx
.extern .const[2] .b32 const_buffer[];
```

> resulted in `const_buffer` pointing to the start of constant bank two. This pointer could then be used to access the entire 64 KB constant bank. Multiple incomplete array variables declared in the same bank were aliased, with each pointing to the start address of the specified constant bank.
>
> To access data in constant banks 1 through 10, the bank number was required in the state space of the load instruction. For example, an incomplete array in bank 2 was accessed as follows:

```ptx
.extern .const[2] .b32 const_buffer[];
ld.const[2].b32  %r1, [const_buffer+4]; // load second word
```

> In PTX ISA version 2.2, we eliminated explicit banks and replaced the incomplete array representation of driver-allocated constant buffers with kernel parameter attributes that allow pointers to constant buffers to be passed as kernel parameters.

###### 中文翻译

旧版 PTX 把 constant memory 暴露为十一组 64 KB bank，变量声明和访问都必须显式指定 bank number。

在 PTX ISA 2.2 之前，constant memory 被组织成固定大小的 bank，共有十一个 64 KB bank。使用 `.const[bank]` modifier 指定 bank，其中 bank number 范围为 0–10；未指定时默认为 bank 0。

按照约定，bank 0 用于所有静态大小的 constant variable，其余 bank 用于声明编译期大小未知的 incomplete constant array。示例声明会使 `const_buffer` 指向 constant bank 2 的起始位置，该 pointer 随后可以访问完整的 64 KB bank。同一 bank 中声明的多个 incomplete array variable 会互相 alias，并且都指向指定 constant bank 的起始地址。

访问 constant bank 1–10 中的数据时，load instruction 的 state space 必须包含 bank number。示例中的 `ld.const[2]` 从 bank 2 的 `const_buffer+4` 处读取第二个 word。

PTX ISA 2.2 删除了显式 bank，并使用 kernel parameter attribute 取代 driver-allocated constant buffer 的 incomplete array 表示法，从而允许把 constant buffer pointer 作为 kernel parameter 传入。

#### 5.1.4 Global State Space

##### English Original

> The global (`.global`) state space is memory that is accessible by all threads in a context. It is the mechanism by which threads in different CTAs, clusters, and grids can communicate. Use `ld.global`, `st.global`, and `atom.global` to access global variables.
>
> Global variables have an optional variable initializer; global variables with no explicit initializer are initialized to zero by default.

##### 中文翻译

Global（`.global`）state space 是一个 context 内所有线程都可以访问的 memory。不同 CTA、cluster 和 grid 中的线程通过它进行通信。使用 `ld.global`、`st.global` 和 `atom.global` 访问 global variable。

Global variable 可以带有可选 initializer；没有显式 initializer 的 global variable 默认初始化为零。

##### 重点解读

`.global` 提供跨线程层级共享的地址空间，但“所有线程可访问”并不自动建立同步、原子性或内存可见顺序。跨 CTA、cluster 或 grid 通信还必须选择与算法相符的 atomic、fence、barrier 或 kernel/grid 依赖机制。

#### 5.1.5 Local State Space

##### English Original

> The local state space (`.local`) is private memory for each thread to keep its own data. It is typically standard memory with cache. The size is limited, as it must be allocated on a per-thread basis. Use `ld.local` and `st.local` to access local variables.
>
> When compiling to use the Application Binary Interface (ABI), `.local` state-space variables must be declared within function scope and are allocated on the stack. In implementations that do not support a stack, all local memory variables are stored at fixed addresses, recursive function calls are not supported, and `.local` variables may be declared at module scope. When compiling legacy PTX code (ISA versions prior to 3.0) containing module-scoped `.local` variables, the compiler silently disables use of the ABI.

##### 中文翻译

Local（`.local`）state space 是每个线程保存自身数据的私有 memory。它通常是带 cache 的普通 memory。由于必须为每个线程单独分配，其容量受到限制。使用 `ld.local` 和 `st.local` 访问 local variable。

使用 Application Binary Interface（ABI）编译时，`.local` state-space variable 必须在 function scope 内声明并分配到 stack。对于不支持 stack 的实现，所有 local memory variable 都存储在固定地址，不支持 recursive function call，并且 `.local` variable 可以在 module scope 声明。编译包含 module-scoped `.local` variable 的旧版 PTX 代码（ISA 3.0 之前）时，编译器会静默禁用 ABI。

##### 重点解读

`.local` 的“local”表示每线程私有的逻辑作用域，不代表物理上位于 SM 片上。它通常用于 stack、无法放入 register 的线程私有对象和 register spill。由于容量按线程分配，大量 local memory 使用会随线程数放大，并可能增加 device-memory traffic。

#### 5.1.6 Parameter State Space

##### English Original

> The parameter (`.param`) state space is used (1) to pass input arguments from the host to the kernel, (2a) to declare formal input and return parameters for device functions called from within kernel execution, and (2b) to declare locally-scoped byte array variables that serve as function call arguments, typically for passing large structures by value to a function. Kernel function parameters differ from device function parameters in terms of access and sharing (read-only versus read-write, per-kernel versus per-thread). Note that PTX ISA versions 1.x supports only kernel function parameters in `.param` space; device function parameters were previously restricted to the register state space. The use of parameter state space for device function parameters was introduced in PTX ISA version 2.0 and requires target architecture `sm_20` or higher. Additional sub-qualifiers `::entry` or `::func` can be specified on instructions with `.param` state space to indicate whether the address refers to kernel function parameter or device function parameter. If no sub-qualifier is specified with the `.param` state space, then the default sub-qualifier is specific to and dependent on the exact instruction. For example, `st.param` is equivalent to `st.param::func` whereas `isspacep.param` is equivalent to `isspacep.param::entry`. Refer to the instruction description for more details on default sub-qualifier assumption.
>
> Note
>
> The location of parameter space is implementation specific. For example, in some implementations kernel parameters reside in global memory. No access protection is provided between parameter and global space in this case. Though the exact location of the kernel parameter space is implementation specific, the kernel parameter space window is always contained within the global space window. Similarly, function parameters are mapped to parameter passing registers and/or stack locations based on the function calling conventions of the Application Binary Interface (ABI). Therefore, PTX code should make no assumptions about the relative locations or ordering of `.param` space variables.

##### 中文翻译

Parameter（`.param`）state space 有三类用途：（1）把 host 的输入参数传给 kernel；（2a）声明 kernel 执行期间调用的 device function 的正式输入参数和返回参数；（2b）声明充当 function-call argument 的局部 byte array variable，通常用于把大型结构按值传给函数。Kernel function parameter 与 device function parameter 的访问和共享语义不同，前者与后者分别涉及只读或读写、per-kernel 或 per-thread。PTX ISA 1.x 的 `.param` space 只支持 kernel function parameter，device function parameter 当时只能使用 register state space。PTX ISA 2.0 开始允许 device function parameter 使用 parameter state space，并要求目标架构为 `sm_20` 或更高。

使用 `.param` state space 的指令还可以指定 `::entry` 或 `::func` sub-qualifier，说明地址引用 kernel function parameter 还是 device function parameter。如果没有指定 sub-qualifier，默认值取决于具体指令。例如，`st.param` 等价于 `st.param::func`，而 `isspacep.param` 等价于 `isspacep.param::entry`。具体默认规则应查阅相应指令说明。

Parameter space 的物理位置由实现决定。例如，某些实现把 kernel parameter 放在 global memory 中，此时 parameter space 与 global space 之间没有访问保护。虽然 kernel parameter space 的确切位置依赖实现，但其地址窗口始终包含在 global space window 内。类似地，function parameter 会按照 ABI function calling convention 映射到 parameter-passing register 和/或 stack location。因此，PTX 代码不能假设 `.param` space variable 之间的相对位置或排列顺序。

##### 重点解读

`.param` 是调用接口抽象，不是固定物理存储。必须先区分 entry parameter 与 function parameter：前者由 host 提供、在 grid 内共享且只读；后者遵循 device ABI、通常按线程存在，并可能通过 register 或 stack 传递。显式写出 `::entry`/`::func` 能减少依赖指令默认规则造成的歧义。

##### 5.1.6.1 Kernel Function Parameters

###### English Original

> Each kernel function definition includes an optional list of parameters. These parameters are addressable, read-only variables declared in the `.param` state space. Values passed from the host to the kernel are accessed through these parameter variables using `ld.param` instructions. The kernel parameter variables are shared across all CTAs from all clusters within a grid.
>
> The address of a kernel parameter may be moved into a register using the `mov` instruction. The resulting address is in the `.param` state space and is accessed using `ld.param` instructions.
>
> Example

```ptx
.entry foo ( .param .b32 N, .param .align 8 .b8 buffer[64] )
{
    .reg .u32 %n;
    .reg .f64 %d;

    ld.param.u32 %n, [N];
    ld.param.f64 %d, [buffer];
    ...
```

> Example

```ptx
.entry bar ( .param .b32 len )
{
    .reg .u32 %ptr, %n;

    mov.u32      %ptr, len;
    ld.param.u32 %n, [%ptr];
    ...
```

> Kernel function parameters may represent normal data values, or they may hold addresses to objects in constant, global, local, or shared state spaces. In the case of pointers, the compiler and runtime system need information about which parameters are pointers, and to which state space they point. Kernel parameter attribute directives are used to provide this information at the PTX level. See Kernel Function Parameter Attributes for a description of kernel parameter attribute directives.
>
> Note
>
> The current implementation does not allow creation of generic pointers to constant variables (`cvta.const`) in programs that have pointers to constant buffers passed as kernel parameters.

###### 中文翻译

每个 kernel function definition 都可以包含一组可选参数。这些参数是在 `.param` state space 中声明、可寻址且只读的变量。Host 传给 kernel 的值通过这些 parameter variable 使用 `ld.param` 读取。Kernel parameter variable 由一个 grid 内全部 cluster 的所有 CTA 共享。

可以使用 `mov` 把 kernel parameter 的地址移动到 register。得到的地址仍属于 `.param` state space，必须使用 `ld.param` 访问。

第一个示例直接使用 parameter symbol `N` 和 `buffer` 加载值；第二个示例先把参数 `len` 的地址移动到 `%ptr`，再通过 `%ptr` 使用 `ld.param` 间接加载。

Kernel function parameter 可以表示普通数据值，也可以保存指向 constant、global、local 或 shared state space 中对象的地址。对于 pointer，compiler 和 runtime system 需要知道哪些 parameter 是 pointer，以及它们指向哪个 state space。Kernel parameter attribute directive 在 PTX 层提供这些信息，详细规则参阅 Kernel Function Parameter Attributes。

当前实现中，如果程序把 constant buffer pointer 作为 kernel parameter 传入，则不允许使用 `cvta.const` 为 constant variable 创建 generic pointer。

###### 重点解读

把 parameter 的地址 `mov` 到 register 不会把它转换成 generic/global address；地址仍在 `.param` state space 中，访问指令仍应是 `ld.param`。若参数本身保存 pointer，则 `.ptr` attribute 描述的是“该参数值指向哪里”，与 parameter variable 自身所在的 `.param` space 是两层不同信息。

##### 5.1.6.2 Kernel Function Parameter Attributes

###### English Original

> Kernel function parameters may be declared with an optional `.ptr` attribute to indicate that a parameter is a pointer to memory, and also indicate the state space and alignment of the memory being pointed to. Kernel Parameter Attribute: `.ptr` describes the `.ptr` kernel parameter attribute.

###### 中文翻译

Kernel function parameter 可以带有可选 `.ptr` attribute，用于表明该参数是 memory pointer，并指出所指 memory 的 state space 和 alignment。下一节具体说明 `.ptr` kernel parameter attribute。

##### 5.1.6.3 Kernel Parameter Attribute: `.ptr`

###### English Original

> `.ptr`
>
> Kernel parameter alignment attribute.
>
> Syntax

```ptx
.param .type .ptr .space .align N  varname
.param .type .ptr        .align N  varname

.space = { .const, .global, .local, .shared };
```

> Description
>
> Used to specify the state space and, optionally, the alignment of memory pointed to by a pointer type kernel parameter. The alignment value `N`, if present, must be a power of two. If no state space is specified, the pointer is assumed to be a generic address pointing to one of const, global, local, or shared memory. If no alignment is specified, the memory pointed to is assumed to be aligned to a 4 byte boundary.
>
> Spaces between `.ptr`, `.space`, and `.align` may be eliminated to improve readability.
>
> PTX ISA Notes
>
> - Introduced in PTX ISA version 2.2.
> - Support for generic addressing of `.const` space added in PTX ISA version 3.1.
>
> Target ISA Notes
>
> - Supported on all target architectures.
>
> Examples

```ptx
.entry foo ( .param .u32 param1,
             .param .u32 .ptr.global.align 16 param2,
             .param .u32 .ptr.const.align 8 param3,
             .param .u32 .ptr.align 16 param4  // generic address
                                               // pointer
) { .. }
```

###### 中文翻译

`.ptr` 是 kernel parameter 的 pointer/alignment attribute。

它用于指定 pointer-type kernel parameter 所指 memory 的 state space，并可选择指定 alignment。Alignment value `N` 如果存在，必须是 2 的幂。未指定 state space 时，pointer 被视为 generic address，可指向 const、global、local 或 shared memory。未指定 alignment 时，默认假定所指 memory 按 4-byte boundary 对齐。

为了提高可读性，可以省略 `.ptr`、`.space` 和 `.align` 之间的空格。

该 attribute 在 PTX ISA 2.2 中引入；PTX ISA 3.1 增加对 `.const` space generic addressing 的支持。所有 target architecture 都支持它。

示例中，`param1` 是没有 pointer attribute 的普通参数；`param2` 指向 16-byte aligned global memory；`param3` 指向 8-byte aligned constant memory；`param4` 是对齐到 16 byte、未指定具体 state space 的 generic pointer。

###### 重点解读

`.ptr` 和 `.align` 为 compiler 提供别名、地址空间与对齐信息，但它们不是动态运行时检查。若调用方传入的地址不满足声明的 state space 或 alignment，代码可能产生错误结果或未定义行为；因此 attribute 必须与实际对象保持一致。

##### 5.1.6.4 Device Function Parameters

###### English Original

> PTX ISA version 2.0 extended the use of parameter space to device function parameters. The most common use is for passing objects by value that do not fit within a PTX register, such as C structures larger than 8 bytes. In this case, a byte array in parameter space is used. Typically, the caller will declare a locally-scoped `.param` byte array variable that represents a flattened C structure or union. This will be passed by value to a callee, which declares a `.param` formal parameter having the same size and alignment as the passed argument.
>
> Example

```ptx
// pass object of type struct { double d; int y; };
.func foo ( .reg .b32 N, .param .align 8 .b8 buffer[12] )
{
    .reg .f64 %d;
    .reg .s32 %y;

    ld.param.f64 %d, [buffer];
    ld.param.s32 %y, [buffer+8];
    ...
}

// code snippet from the caller
// struct { double d; int y; } mystruct; is flattened, passed to foo
    ...
    .reg .f64 dbl;
    .reg .s32 x;
    .param .align 8 .b8 mystruct;
    ...
    st.param.f64 [mystruct+0], dbl;
    st.param.s32 [mystruct+8], x;
    call foo, (4, mystruct);
    ...
```

> See the section on function call syntax for more details.
>
> Function input parameters may be read via `ld.param` and function return parameters may be written using `st.param`; it is illegal to write to an input parameter or read from a return parameter.
>
> Aside from passing structures by value, `.param` space is also required whenever a formal parameter has its address taken within the called function. In PTX, the address of a function input parameter may be moved into a register using the `mov` instruction. Note that the parameter will be copied to the stack if necessary, and so the address will be in the `.local` state space and is accessed via `ld.local` and `st.local` instructions. It is not possible to use `mov` to get the address of or a locally-scoped `.param` space variable. Starting PTX ISA version 6.0, it is possible to use `mov` instruction to get address of return parameter of device function.
>
> Example

```ptx
// pass array of up to eight floating-point values in buffer
.func foo ( .param .b32 N, .param .b32 buffer[32] )
{
    .reg .u32  %n, %r;
    .reg .f32  %f;
    .reg .pred %p;

    ld.param.u32 %n, [N];
    mov.u32      %r, buffer;  // forces buffer to .local state space
Loop:
    setp.eq.u32  %p, %n, 0;
@%p bra         Done;
    ld.local.f32 %f, [%r];
    ...
    add.u32      %r, %r, 4;
    sub.u32      %n, %n, 1;
    bra          Loop;
Done:
    ...
}
```

###### 中文翻译

PTX ISA 2.0 把 parameter space 的用途扩展到 device function parameter。最常见的用途是按值传递无法放入一个 PTX register 的对象，例如大于 8 byte 的 C structure。此时使用 parameter space 中的 byte array。通常 caller 声明一个局部 `.param` byte array variable，表示展开后的 C structure 或 union；该对象按值传给 callee，callee 声明具有相同 size 和 alignment 的 `.param` formal parameter。

第一个示例把 `struct { double d; int y; }` 展开为对齐到 8 byte 的 12-byte buffer。Caller 使用 `st.param` 把两个字段写入局部 parameter buffer，再按值调用 `foo`；callee 使用 `ld.param` 按对应 offset 读取字段。更多细节参阅 function-call syntax。

Function input parameter 可以通过 `ld.param` 读取，function return parameter 可以通过 `st.param` 写入；写 input parameter 或读取 return parameter 都是非法的。

除按值传递 structure 外，只要被调函数内部需要取得 formal parameter 的地址，也必须使用 `.param` space。在 PTX 中，可以用 `mov` 把 function input parameter 的地址移入 register。必要时该 parameter 会被复制到 stack，因此所得地址属于 `.local` state space，应通过 `ld.local` 和 `st.local` 访问。不能使用 `mov` 获取局部 `.param` space variable 的地址。从 PTX ISA 6.0 开始，可以使用 `mov` 获取 device function return parameter 的地址。

第二个示例中，`mov.u32 %r, buffer` 迫使 input parameter `buffer` 进入 `.local` stack storage；之后循环使用 `ld.local` 而不是 `ld.param` 访问数组元素。

###### 重点解读

Device function parameter 在“按值传递”与“取地址”时表现不同。只按值访问可直接使用 `ld.param`/`st.param`；一旦需要稳定地址，ABI 可能把对象实体化到 per-thread stack，使地址转入 `.local` state space。后续 load/store 必须跟随地址所属 state space 改用 `ld.local`/`st.local`。

#### 5.1.7 Shared State Space

##### English Original

> The shared (`.shared`) state space is a memory that is owned by an executing CTA and is accessible to the threads of all the CTAs within a cluster. An address in shared memory can be read and written by any thread in a CTA cluster.
>
> Additional sub-qualifiers `::cta` or `::cluster` can be specified on instructions with `.shared` state space to indicate whether the address belongs to the shared memory window of the executing CTA or of any CTA in the cluster respectively. The addresses in the `.shared::cta` window also fall within the `.shared::cluster` window. If no sub-qualifier is specified with the `.shared` state space, then it defaults to `::cta`. For example, `ld.shared` is equivalent to `ld.shared::cta`.
>
> Variables declared in `.shared` state space refer to the memory addresses in the current CTA. Instruction `mapa` gives the `.shared::cluster` address of the corresponding variable in another CTA in the cluster.
>
> Shared memory typically has some optimizations to support the sharing. One example is broadcast; where all threads read from the same address. Another is sequential access from sequential threads.
>
> Maximum capacity for statically allocated shared memory is 48 KB per CTA. Architecture specific targets such as `sm_90a`, support extended statically allocated shared memory capacity beyond 48 KB per CTA as described below:

| Target architecture | Maximum statically allocated shared memory size |
| --- | --- |
| `sm_90a` | 228 KB |
| `sm_100a`, `sm_103a`, `sm_107a` | 228 KB |
| `sm_110a` | 228 KB |
| `sm_120a`, `sm_121a` | 100 KB |

##### 中文翻译

Shared（`.shared`）state space 是由正在执行的 CTA 所拥有、并可由同一 cluster 内所有 CTA 的线程访问的 memory。CTA cluster 中任意线程都可以读写 shared-memory address。

使用 `.shared` state space 的指令可以额外指定 `::cta` 或 `::cluster` sub-qualifier，分别说明地址属于当前 CTA 的 shared-memory window，还是 cluster 内任意 CTA 的 shared-memory window。`.shared::cta` window 中的地址也属于 `.shared::cluster` window。未指定 sub-qualifier 时默认为 `::cta`，例如 `ld.shared` 等价于 `ld.shared::cta`。

在 `.shared` state space 声明的 variable 指向当前 CTA 的 memory address。`mapa` 指令可以得到 cluster 内另一个 CTA 中对应 variable 的 `.shared::cluster` address。

Shared memory 通常针对共享访问进行优化，例如所有线程读取同一地址时的 broadcast，以及编号连续的线程进行 sequential access。

静态分配 shared memory 的一般最大容量为每 CTA 48 KB。`sm_90a` 等 architecture-specific target 支持超过 48 KB 的扩展静态容量，具体数值按原文表格保留，不另行翻译。

##### 重点解读

`.shared::cta` 与 `.shared::cluster` 描述不同地址窗口，而不是两套独立物理变量。普通 `.shared` variable 首先引用当前 CTA 的对象；访问 peer CTA 对应对象时，需要用 `mapa` 建立 cluster address，并确保 peer CTA 仍存活及同步关系正确。表中的容量是“静态分配”上限，不能直接等同于每个 GPU 的总 shared-memory 容量或动态分配上限。

#### 5.1.8 Texture State Space (deprecated)

##### English Original

> The texture (`.tex`) state space is global memory accessed via the texture instruction. It is shared by all threads in a context. Texture memory is read-only and cached, so accesses to texture memory are not coherent with global memory stores to the texture image.
>
> The GPU hardware has a fixed number of texture bindings that can be accessed within a single kernel (typically 128). The `.tex` directive will bind the named texture memory variable to a hardware texture identifier, where texture identifiers are allocated sequentially beginning with zero. Multiple names may be bound to the same physical texture identifier. An error is generated if the maximum number of physical resources is exceeded. The texture name must be of type `.u32` or `.u64`.
>
> Physical texture resources are allocated on a per-kernel granularity, and `.tex` variables are required to be defined in the global scope.
>
> Texture memory is read-only. A texture’s base address is assumed to be aligned to a 16 byte boundary.
>
> Example

```ptx
.tex .u32 tex_a;         // bound to physical texture 0
.tex .u32 tex_c, tex_d;  // both bound to physical texture 1
.tex .u32 tex_d;         // bound to physical texture 2
.tex .u32 tex_f;         // bound to physical texture 3
```

> Note
>
> Explicit declarations of variables in the texture state space is deprecated, and programs should instead reference texture memory through variables of type `.texref`. The `.tex` directive is retained for backward compatibility, and variables declared in the `.tex` state space are equivalent to module-scoped `.texref` variables in the `.global` state space.
>
> For example, a legacy PTX definitions such as

```ptx
.tex .u32 tex_a;
```

> is equivalent to:

```ptx
.global .texref tex_a;
```

> See Texture Sampler and Surface Types for the description of the `.texref` type and Texture Instructions for its use in texture instructions.

##### 中文翻译

Texture（`.tex`）state space 是通过 texture instruction 访问的 global memory，由一个 context 中的所有线程共享。Texture memory 只读且带 cache，因此 texture memory access 与对相应 texture image 的 global-memory store 不保持一致。

GPU hardware 可供单个 kernel 访问的 texture binding 数量固定，通常为 128。`.tex` directive 把具名 texture-memory variable 绑定到 hardware texture identifier，identifier 从 0 开始顺序分配。多个名称可以绑定到同一个 physical texture identifier；超过物理资源上限会产生错误。Texture name 的类型必须为 `.u32` 或 `.u64`。

Physical texture resource 以 per-kernel 粒度分配，`.tex` variable 必须定义在 global scope。Texture memory 是只读的，并假定 texture base address 按 16-byte boundary 对齐。

示例说明 `.tex` declaration 如何分配或共享 physical texture identifier。

显式声明 texture state-space variable 已弃用，程序应通过 `.texref` 类型的 variable 引用 texture memory。为保持向后兼容，`.tex` directive 仍然保留；在 `.tex` state space 声明的 variable 等价于 `.global` state space 中 module-scoped `.texref` variable。因此旧式 `.tex .u32 tex_a;` 等价于 `.global .texref tex_a;`。关于 `.texref` 及 texture instruction 的用法，参阅 Texture Sampler and Surface Types 与 Texture Instructions。

##### 重点解读

本节描述的是兼容旧 PTX 的 `.tex` declaration。新代码不应继续建立对 deprecated `.tex` state space 的设计依赖，而应使用 `.texref` 等当前接口。无论采用哪种表示，texture cache 都不保证观察同一 kernel 内对底层 global memory 的写入，这是一项正确性约束而不只是性能差异。

### 5.2 Types

#### 5.2.1 Fundamental Types

##### English Original

> In PTX, the fundamental types reflect the native data types supported by the target architectures. A fundamental type specifies both a basic type and a size. Register variables are always of a fundamental type, and instructions operate on these types. The same type-size specifiers are used for both variable definitions and for typing instructions, so their names are intentionally short.
>
> Table 8 lists the fundamental type specifiers for each basic type:

**Table 8. Fundamental Type Specifiers**

| Basic Type | Fundamental Type Specifiers |
| --- | --- |
| Signed integer | `.s8`, `.s16`, `.s32`, `.s64` |
| Unsigned integer | `.u8`, `.u16`, `.u32`, `.u64` |
| Floating-point | `.f16`, `.f16x2`, `.f32`, `.f64` |
| Bits (untyped) | `.b8`, `.b16`, `.b32`, `.b64`, `.b128` |
| Predicate | `.pred` |

> Most instructions have one or more type specifiers, needed to fully specify instruction behavior. Operand types and sizes are checked against instruction types for compatibility.
>
> Two fundamental types are compatible if they have the same basic type and are the same size. Signed and unsigned integer types are compatible if they have the same size. The bit-size type is compatible with any fundamental type having the same size.
>
> In principle, all variables (aside from predicates) could be declared using only bit-size types, but typed variables enhance program readability and allow for better operand type checking.

##### 中文翻译

PTX fundamental type 反映 target architecture 支持的 native data type。Fundamental type 同时指定 basic type 和 size。Register variable 始终使用 fundamental type，instruction 也在这些类型上操作。Variable definition 和 instruction typing 使用相同的 type-size specifier，因此名称被有意设计得很短。

表 8 列出每种 basic type 的 fundamental type specifier，按原文完整保留，不另行翻译。

大多数 instruction 带有一个或多个 type specifier，用于完整确定指令行为。系统会根据 instruction type 检查 operand type 和 size 是否兼容。

两个 fundamental type 具有相同 basic type 和相同 size 时互相兼容。相同 size 的 signed/unsigned integer type 相互兼容。Bit-size type 与任何相同 size 的 fundamental type 兼容。

原则上，除 predicate 外的所有 variable 都可以只用 bit-size type 声明；但 typed variable 能提高可读性，并让 operand type checking 更严格。

##### 重点解读

`.b32` 只声明“32 bit 容器”，不承诺其中是整数还是浮点数，因此适合搬运和保存原始 bit pattern；`.s32`、`.u32`、`.f32` 则携带解释语义。相同位宽带来的 compatibility 允许某些指令复用存储，但具体 instruction type 仍决定算术、比较、舍入和符号扩展行为。

#### 5.2.2 Restricted Use of Sub-Word Sizes

##### English Original

> The `.u8`, `.s8`, and `.b8` instruction types are restricted to `ld`, `st`, `add`, `sub`, `min`, `max`, `neg` and `cvt` instructions. The `.f16` floating-point type is allowed in half precision floating point instructions and texture fetch instructions. The `.f16x2` floating point type is allowed only in half precision floating point arithmetic instructions and texture fetch instructions.
>
> For convenience, `ld`, `st`, and `cvt` instructions permit source and destination data operands to be wider than the instruction-type size, so that narrow values may be loaded, stored, and converted using regular-width registers. For example, 8-bit or 16-bit values may be held directly in 32-bit or 64-bit registers when being loaded, stored, or converted to other types and sizes.

##### 中文翻译

`.u8`、`.s8` 和 `.b8` instruction type 只能用于 `ld`、`st`、`add`、`sub`、`min`、`max`、`neg` 和 `cvt`。`.f16` floating-point type 可用于 half-precision floating-point instruction 和 texture-fetch instruction；`.f16x2` 只能用于 half-precision floating-point arithmetic instruction 和 texture-fetch instruction。

为方便使用，`ld`、`st` 和 `cvt` 允许 source/destination data operand 比 instruction-type size 更宽，从而可以使用常规宽度 register 加载、存储和转换 narrow value。例如，8-bit 或 16-bit value 在 load、store 或转换到其他类型和大小时，可以直接保存在 32-bit 或 64-bit register 中。

##### 重点解读

Register container width 与 instruction 的有效数据宽度可以不同。宽 register 容纳 narrow value 不表示指令自动对高位采用任意语义；高位如何处理取决于具体 `ld`、`st` 或 `cvt` 变体及其 signedness/type rules。

#### 5.2.3 Alternate Floating-Point Data Formats

##### English Original

> The fundamental floating-point types supported in PTX have implicit bit representations that indicate the number of bits used to store exponent and mantissa. For example, the `.f16` type indicates 5 bits reserved for exponent and 10 bits reserved for mantissa. In addition to the floating-point representations assumed by the fundamental types, PTX allows the following alternate floating-point data formats:
>
> **`bf16` data format:**
>
> This data format is a 16-bit floating point format with 8 bits for exponent and 7 bits for mantissa. A register variable containing `bf16` data must be declared with `.b16` type.
>
> **`e4m3` data format:**
>
> This data format is an 8-bit floating point format with 4 bits for exponent and 3 bits for mantissa. The `e4m3` encoding does not support infinity and `NaN` values are limited to `0x7f` and `0xff`. A register variable containing `e4m3` value must be declared using bit-size type.
>
> **`e5m2` data format:**
>
> This data format is an 8-bit floating point format with 5 bits for exponent and 2 bits for mantissa. A register variable containing `e5m2` value must be declared using bit-size type.
>
> **`tf32` data format:**
>
> This data format is a special 32-bit floating point format supported by the matrix multiply-and-accumulate instructions, with the same range as `.f32` and reduced precision (`>=10` bits). The internal layout of `tf32` format is implementation defined. PTX facilitates conversion from single precision `.f32` type to `tf32` format. A register variable containing `tf32` data must be declared with `.b32` type.
>
> **`e2m1` data format:**
>
> This data format is a 4-bit floating point format with 2 bits for exponent and 1 bit for mantissa. The `e2m1` encoding does not support infinity and `NaN`. `e2m1` values must be used in a packed format specified as `e2m1x2`. A register variable containing two `e2m1` values must be declared with `.b8` type.
>
> **`e2m3` data format:**
>
> This data format is a 6-bit floating point format with 2 bits for exponent and 3 bits for mantissa. The `e2m3` encoding does not support infinity and `NaN`. `e2m3` values must be used in a packed format specified as `e2m3x2`. A register variable containing two `e2m3` values must be declared with `.b16` type where each `.b8` element has 6-bit floating point value and 2 MSB bits padded with zeros.
>
> **`e3m2` data format:**
>
> This data format is a 6-bit floating point format with 3 bits for exponent and 2 bits for mantissa. The `e3m2` encoding does not support infinity and `NaN`. `e3m2` values must be used in a packed format specified as `e3m2x2`. A register variable containing two `e3m2` values must be declared with `.b16` type where each `.b8` element has 6-bit floating point value and 2 MSB bits padded with zeros.
>
> **`ue8m0` data format:**
>
> This data format is an 8-bit unsigned floating-point format with 8 bits for exponent and 0 bits for mantissa. The `ue8m0` encoding does not support infinity. `NaN` value is limited to `0xff`. `ue8m0` values must be used in a packed format specified as `ue8m0x2`. A register variable containing two `ue8m0` values must be declared with `.b16` type.
>
> **`ue4m3` data format:**
>
> This data format is a 7-bit unsigned floating-point format with 4 bits for exponent and 3 bits for mantissa. The `ue4m3` encoding does not support infinity. `NaN` value is limited to `0x7f`. A register variable containing single `ue4m3` value must be declared with `.b8` type having MSB bit padded with zero.
>
> **`ue5m3` data format:**
>
> This data format is a 8-bit unsigned floating-point format with 5 bits for exponent and 3 bits for mantissa. The `ue5m3` encoding does not support infinity. `NaN` value is limited to `0xff`. A register variable containing single `ue5m3` value must be declared with `.b8` type.
>
> Alternate data formats cannot be used as fundamental types. They are supported as source or destination formats by certain instructions.

##### 中文翻译

PTX 支持的 fundamental floating-point type 具有隐含 bit representation，规定 exponent 和 mantissa 各占多少 bit。例如，`.f16` 为 exponent 保留 5 bit，为 mantissa 保留 10 bit。除 fundamental type 所假定的浮点表示外，PTX 还支持以下 alternate floating-point data format。

**`bf16`：**16-bit floating-point format，包含 8-bit exponent 和 7-bit mantissa。保存 `bf16` 数据的 register variable 必须声明为 `.b16`。

**`e4m3`：**8-bit floating-point format，包含 4-bit exponent 和 3-bit mantissa。不支持 infinity，`NaN` 仅限 `0x7f` 和 `0xff`。保存 `e4m3` value 的 register variable 必须使用 bit-size type 声明。

**`e5m2`：**8-bit floating-point format，包含 5-bit exponent 和 2-bit mantissa。保存 `e5m2` value 的 register variable 必须使用 bit-size type 声明。

**`tf32`：**由 matrix multiply-and-accumulate instruction 支持的特殊 32-bit floating-point format。它与 `.f32` 具有相同 range，但 precision 降低到至少 10 bit；内部 layout 由实现定义。PTX 支持从 single-precision `.f32` 转换到 `tf32`。保存 `tf32` 数据的 register variable 必须声明为 `.b32`。

**`e2m1`：**4-bit floating-point format，包含 2-bit exponent 和 1-bit mantissa，不支持 infinity 和 `NaN`。`e2m1` 必须以 `e2m1x2` packed format 使用；包含两个 `e2m1` value 的 register variable 必须声明为 `.b8`。

**`e2m3`：**6-bit floating-point format，包含 2-bit exponent 和 3-bit mantissa，不支持 infinity 和 `NaN`。它必须以 `e2m3x2` packed format 使用；两个 `e2m3` value 使用 `.b16` register，其中每个 `.b8` element 保存一个 6-bit value，高 2 bit 补零。

**`e3m2`：**6-bit floating-point format，包含 3-bit exponent 和 2-bit mantissa，不支持 infinity 和 `NaN`。它必须以 `e3m2x2` packed format 使用；两个 `e3m2` value 使用 `.b16` register，其中每个 `.b8` element 保存一个 6-bit value，高 2 bit 补零。

**`ue8m0`：**8-bit unsigned floating-point format，包含 8-bit exponent 和 0-bit mantissa。不支持 infinity，`NaN` 仅限 `0xff`。它必须以 `ue8m0x2` packed format 使用；两个 `ue8m0` value 使用 `.b16` register。

**`ue4m3`：**7-bit unsigned floating-point format，包含 4-bit exponent 和 3-bit mantissa。不支持 infinity，`NaN` 仅限 `0x7f`。单个 `ue4m3` value 使用 `.b8` register，最高位补零。

**`ue5m3`：**8-bit unsigned floating-point format，包含 5-bit exponent 和 3-bit mantissa。不支持 infinity，`NaN` 仅限 `0xff`。单个 `ue5m3` value 使用 `.b8` register。

Alternate data format 不能作为 fundamental type，只能由特定 instruction 用作 source 或 destination format。

##### 重点解读

Alternate format 的语义由使用它的具体 instruction 提供，承载它的 register 则通常声明为同宽度 `.b*` 容器。容器 type 只保证 bit 数足够，不会让普通浮点指令自动把内容识别为 `bf16`、FP8、FP6、FP4 或 `tf32`；必须查阅指令允许的 format、packing order、rounding 和 accumulator type。

#### 5.2.4 Fixed-point Data format

##### English Original

> PTX supports following fixed-point data formats:
>
> **`s2f6` data format:**
>
> This data format is 8-bit signed 2’s complement integer with 2 sign-integer bits and 6 fractional bits with form `xx.xxxxxx`. The `s2f6` encoding does not support infinity and `NaN`.
>
> `s2f6` value = s8 value * 2^(-6)
>
> Positive max representation = `01.111111` = 127 * 2^(-6) = 1.984375
>
> Negative max representation = `10.000000` = -128 * 2^(-6) = -2.0

##### 中文翻译

PTX 支持以下 fixed-point data format。

**`s2f6`：**一种 8-bit signed two’s-complement integer 表示，具有 2 个 sign/integer bit 和 6 个 fractional bit，形式为 `xx.xxxxxx`。`s2f6` 不支持 infinity 和 `NaN`。

其数值等于 `s8 value × 2^-6`。最大正值表示 `01.111111`，即 `127 × 2^-6 = 1.984375`；最小负值表示 `10.000000`，即 `-128 × 2^-6 = -2.0`。

#### 5.2.5 Packed Data Types

##### English Original

> Certain PTX instructions operate on two or more sets of inputs in parallel, and produce two or more sets of outputs. Such instructions can use the data stored in a packed format. PTX supports either two or four values of the same scalar data type to be packed into a single, larger value. The packed value is considered as a value of a packed data type. In this section we describe the packed data types supported in PTX.

##### 中文翻译

某些 PTX instruction 会并行处理两组或更多组 input，并产生两组或更多组 output。这些指令可以使用 packed format 保存的数据。PTX 支持把同一 scalar data type 的两个或四个 value 打包进一个更大的 value；所得 value 被视为 packed data type。本节说明 PTX 支持的 packed data type。

##### 重点解读

Packed type 把多个低位宽元素放入一个 register container，使一条指令能够并行处理多个 element。它描述的是 operand layout 与指令语义，不意味着这些格式都可以像 `.f32` 一样独立声明为 fundamental register type。

##### 5.2.5.1 Packed Floating Point Data Types

###### English Original

> PTX supports various variants of packed floating point data types. Out of them, only `.f16x2` is supported as a fundamental type, while others cannot be used as fundamental types - they are supported as instruction types on certain instructions. When using an instruction with such non-fundamental types, the operand data variables must be of bit type of appropriate size. For example, all of the operand variables must be of type `.b32` for an instruction with instruction type as `.bf16x2`. Table 9 described various variants of packed floating point data types in PTX.

**Table 9. Operand types for packed floating point instruction type**

| Packed floating point type | Number of elements contained in a packed format | Type of each element | Register variable type to be used in the declaration |
| --- | --- | --- | --- |
| `.f16x2` | Two | `.f16` | `.f16x2` or `.b32` |
| `.f32x2` | Two | `.f32` | `.b64` |
| `.bf16x2` | Two | `.bf16` | `.b32` |
| `.e4m3x2` | Two | `.e4m3` | `.b16` |
| `.e5m2x2` | Two | `.e5m2` | `.b16` |
| `.e2m3x2` | Two | `.e2m3` | `.b16` |
| `.e3m2x2` | Two | `.e3m2` | `.b16` |
| `.ue8m0x2` | Two | `.ue8m0` | `.b16` |
| `.ue5m3x2` | Two | `.ue5m3` | `.b16` |
| `.s2f6x2` | Two | `.s2f6` | `.b16` |
| `.e2m1x2` | Two | `.e2m1` | `.b8` |
| `.e4m3x4` | Four | `.e4m3` | `.b32` |
| `.e5m2x4` | Four | `.e5m2` | `.b32` |
| `.e2m3x4` | Four | `.e2m3` | `.b32` |
| `.e3m2x4` | Four | `.e3m2` | `.b32` |
| `.e2m1p4x4` | Four | `.e2m1` wrapped in `.b8` with 4-bit of zero padding at MSB | `.b32` |
| `.ue8m0x4` | Four | `.ue8m0` | `.b32` |
| `.e2m1x4` | Four | `.e2m1` | `.b16` |

###### 中文翻译

PTX 支持多种 packed floating-point data type。其中只有 `.f16x2` 是 fundamental type，其他格式不能作为 fundamental type，只能作为某些 instruction 的 instruction type。使用这些 non-fundamental type 时，operand data variable 必须声明为适当大小的 bit type。例如，instruction type 为 `.bf16x2` 时，所有 operand variable 都必须为 `.b32`。表 9 完整列出相关 packed type、element 数量、element type 与 register declaration type，不另行翻译。

###### 重点解读

表中的 register type 是存储容器约束。例如 `.bf16x2` 包含两个 `bf16` element，总计 32 bit，因此 operand variable 声明为 `.b32`；真正把两个 16-bit segment 解释为 `bf16` 的是使用 `.bf16x2` instruction type 的具体指令。

##### 5.2.5.2 Packed Integer Data Types

###### English Original

> PTX supports four variants of packed integer data types: `.u16x2`, `.s16x2`, `.u8x4`, and `.s8x4`. The `.u16x2`, `.s16x2` packed data types consist of two `.u16` or `.s16` values. The `.u8x4`, `.s8x4` packed data types consist of four `.u8` or `.s8` values. A register variable containing `.u16x2`, `.s16x2`, `.u8x4`, `.s8x4` data must be declared with `.b32` type. Packed integer data types cannot be used as fundamental types. They are supported as instruction types on certain instructions.

###### 中文翻译

PTX 支持四种 packed integer data type：`.u16x2`、`.s16x2`、`.u8x4` 和 `.s8x4`。`.u16x2`、`.s16x2` 分别由两个 `.u16` 或 `.s16` value 构成；`.u8x4`、`.s8x4` 分别由四个 `.u8` 或 `.s8` value 构成。保存这些数据的 register variable 必须声明为 `.b32`。Packed integer data type 不能作为 fundamental type，只能作为某些 instruction 的 instruction type。

##### 5.2.5.3 Packed Fixed-Point Data Types

###### English Original

> PTX supports `.s2f6x2` packed fixed-point data type consisting of two `.s2f6` packed fixed-point values. A register variable containing `.s2f6x2` value must be declared with `.b16` type. Packed fixed-point data type cannot be used as fundamental type and is only supported as instruction type.

###### 中文翻译

PTX 支持 `.s2f6x2` packed fixed-point data type，由两个 `.s2f6` packed fixed-point value 构成。保存 `.s2f6x2` value 的 register variable 必须声明为 `.b16`。Packed fixed-point data type 不能作为 fundamental type，只能作为 instruction type 使用。

### 5.3 Texture Sampler and Surface Types

#### English Original

> PTX includes built-in opaque types for defining texture, sampler, and surface descriptor variables. These types have named fields similar to structures, but all information about layout, field ordering, base address, and overall size is hidden to a PTX program, hence the term opaque. The use of these opaque types is limited to:
>
> - Variable definition within global (module) scope and in kernel entry parameter lists.
> - Static initialization of module-scope variables using comma-delimited static assignment expressions for the named members of the type.
> - Referencing textures, samplers, or surfaces via texture and surface load/store instructions (`tex`, `suld`, `sust`, `sured`).
> - Retrieving the value of a named member via query instructions (`txq`, `suq`).
> - Creating pointers to opaque variables using `mov`, e.g., `mov.u64 reg, opaque_var;`. The resulting pointer may be stored to and loaded from memory, passed as a parameter to functions, and de-referenced by texture and surface load, store, and query instructions, but the pointer cannot otherwise be treated as an address, i.e., accessing the pointer with `ld` and `st` instructions, or performing pointer arithmetic will result in undefined results.
> - Opaque variables may not appear in initializers, e.g., to initialize a pointer to an opaque variable.
>
> Note
>
> Indirect access to textures and surfaces using pointers to opaque variables is supported beginning with PTX ISA version 3.1 and requires target `sm_20` or later.
>
> Indirect access to textures is supported only in unified texture mode (see below).
>
> The three built-in types are `.texref`, `.samplerref`, and `.surfref`. For working with textures and samplers, PTX has two modes of operation. In the unified mode, texture and sampler information is accessed through a single `.texref` handle. In the independent mode, texture and sampler information each have their own handle, allowing them to be defined separately and combined at the site of usage in the program. In independent mode, the fields of the `.texref` type that describe sampler properties are ignored, since these properties are defined by `.samplerref` variables.
>
> Table 10 and Table 11 list the named members of each type for unified and independent texture modes. These members and their values have precise mappings to methods and values defined in the texture `HW` class as well as exposed values via the API.

**Table 10. Opaque Type Fields in Unified Texture Mode**

| Member | `.texref` values | `.surfref` values |
| --- | --- | --- |
| `width` | in elements | in elements |
| `height` | in elements | in elements |
| `depth` | in elements | in elements |
| `channel_data_type` | `enum` type corresponding to source language API | `enum` type corresponding to source language API |
| `channel_order` | `enum` type corresponding to source language API | `enum` type corresponding to source language API |
| `normalized_coords` | `0`, `1` | N/A |
| `filter_mode` | `nearest`, `linear` | N/A |
| `addr_mode_0`, `addr_mode_1`, `addr_mode_2` | `wrap`, `mirror`, `clamp_ogl`, `clamp_to_edge`, `clamp_to_border` | N/A |
| `array_size` | as number of textures in a texture array | as number of surfaces in a surface array |
| `num_mipmap_levels` | as number of levels in a mipmapped texture | N/A |
| `num_samples` | as number of samples in a multi-sample texture | N/A |
| `memory_layout` | N/A | `1` for linear memory layout; `0` otherwise |

#### 中文翻译

PTX 提供用于定义 texture、sampler 和 surface descriptor variable 的内置 opaque type。这些类型像 structure 一样具有具名 field，但 layout、field ordering、base address 和整体 size 对 PTX 程序完全隐藏，因此称为 opaque。其用途仅限于：在 global（module）scope 或 kernel entry parameter list 中定义 variable；通过逗号分隔、对具名 member 赋值的 static assignment expression 初始化 module-scope variable；通过 `tex`、`suld`、`sust`、`sured` 等 texture/surface load-store instruction 引用资源；使用 `txq`、`suq` query instruction 读取具名 member；以及用 `mov` 创建 opaque-variable pointer。

Opaque pointer 可以存入 memory、从 memory 加载、作为 function parameter 传递，也可以由 texture/surface load、store 和 query instruction 解引用，但不能作为普通 address 使用。用 `ld`/`st` 访问它或对它执行 pointer arithmetic 会产生 undefined result。Opaque variable 也不能出现在 initializer 中，例如不能用它初始化指向 opaque variable 的 pointer。

从 PTX ISA 3.1 开始，可以通过 opaque-variable pointer 间接访问 texture 和 surface，并要求 `sm_20+`；texture 的间接访问只支持 unified texture mode。

三种内置类型是 `.texref`、`.samplerref` 和 `.surfref`。Unified mode 使用一个 `.texref` handle 同时访问 texture 与 sampler 信息；independent mode 为 texture 和 sampler 使用独立 handle，并在使用位置组合。Independent mode 下，`.texref` 中描述 sampler property 的 field 会被忽略，因为这些 property 由 `.samplerref` 提供。表 10、11 完整保留两种模式的具名 member，不另行翻译。

#### 重点解读

Opaque handle 的“pointer”只是一种可传递的资源引用，并不是普通 data pointer。它可以被专用 texture/surface/query instruction 解引用，却不能参与普通 `ld`、`st` 或地址运算。统一模式把 texture 与 sampling state 绑定在一个 `.texref` 中；独立模式拆分 `.texref` 与 `.samplerref`，使同一 texture 可以组合不同 sampler state。

#### 5.3.1 Texture and Surface Properties

##### English Original

> Fields `width`, `height`, and `depth` specify the size of the texture or surface in number of elements in each dimension.
>
> The `channel_data_type` and `channel_order` fields specify these properties of the texture or surface using enumeration types corresponding to the source language API. For example, see Channel Data Type and Channel Order Fields for the OpenCL enumeration types currently supported in PTX.

##### 中文翻译

`width`、`height` 和 `depth` field 分别以 element 数量表示 texture 或 surface 各维度的大小。

`channel_data_type` 和 `channel_order` 使用与 source-language API 对应的 enumeration type 描述 texture/surface 的相应属性。PTX 当前支持的 OpenCL enumeration type 参阅 Channel Data Type and Channel Order Fields。

#### 5.3.2 Sampler Properties

##### English Original

> The `normalized_coords` field indicates whether the texture or surface uses normalized coordinates in the range [0.0, 1.0) instead of unnormalized coordinates in the range [0, N). If no value is specified, the default is set by the runtime system based on the source language.
>
> The `filter_mode` field specifies how the values returned by texture reads are computed based on the input texture coordinates.
>
> The `addr_mode_{0,1,2}` fields define the addressing mode in each dimension, which determine how out-of-range coordinates are handled.
>
> See the CUDA C++ Programming Guide for more details of these properties.

**Table 11. Opaque Type Fields in Independent Texture Mode**

| Member | `.samplerref` values | `.texref` values | `.surfref` values |
| --- | --- | --- | --- |
| `width` | N/A | in elements | in elements |
| `height` | N/A | in elements | in elements |
| `depth` | N/A | in elements | in elements |
| `channel_data_type` | N/A | `enum` type corresponding to source language API | `enum` type corresponding to source language API |
| `channel_order` | N/A | `enum` type corresponding to source language AP | `enum` type corresponding to source language AP |
| `normalized_coords` | N/A | `0`, `1` | N/A |
| `force_unnormalized_coords` | `0`, `1` | N/A | N/A |
| `filter_mode` | `nearest`, `linear` | ignored | N/A |
| `addr_mode_0`, `addr_mode_1`, `addr_mode_2` | `wrap`, `mirror`, `clamp_ogl`, `clamp_to_edge`, `clamp_to_border` | N/A | N/A |
| `array_size` | N/A | as number of textures in a texture array | as number of surfaces in a surface array |
| `num_mipmap_levels` | N/A | as number of levels in a mipmapped texture | N/A |
| `num_samples` | N/A | as number of samples in a multi-sample texture | N/A |
| `memory_layout` | N/A | N/A | `1` for linear memory layout; `0` otherwise |

> In independent texture mode, the sampler properties are carried in an independent `.samplerref` variable, and these fields are disabled in the `.texref` variables. One additional sampler property, `force_unnormalized_coords`, is available in independent texture mode.
>
> The `force_unnormalized_coords` field is a property of `.samplerref` variables that allows the sampler to override the texture header `normalized_coords` property. This field is defined only in independent texture mode. When `True`, the texture header setting is overridden and unnormalized coordinates are used; when `False`, the texture header setting is used.
>
> The `force_unnormalized_coords` property is used in compiling OpenCL; in OpenCL, the property of normalized coordinates is carried in sampler headers. To compile OpenCL to PTX, texture headers are always initialized with `normalized_coords` set to True, and the OpenCL sampler-based `normalized_coords` flag maps (negated) to the PTX-level `force_unnormalized_coords` flag.
>
> Variables using these types may be declared at module scope or within kernel entry parameter lists. At module scope, these variables must be in the `.global` state space. As kernel parameters, these variables are declared in the `.param` state space.
>
> Example

```ptx
.global .texref     my_texture_name;
.global .samplerref my_sampler_name;
.global .surfref    my_surface_name;
```

> When declared at module scope, the types may be initialized using a list of static expressions assigning values to the named members.
>
> Example

```ptx
.global .texref tex1;
.global .samplerref tsamp1 = { addr_mode_0 = clamp_to_border,
                               filter_mode = nearest
                             };
```

##### 中文翻译

`normalized_coords` 表明 texture 或 surface 使用范围为 `[0.0, 1.0)` 的 normalized coordinate，还是范围为 `[0, N)` 的 unnormalized coordinate。未指定时，runtime system 根据 source language 设置默认值。`filter_mode` 规定 texture read 如何根据输入 texture coordinate 计算返回值；`addr_mode_{0,1,2}` 定义每个维度的 addressing mode，决定如何处理越界 coordinate。更多细节参阅 CUDA C++ Programming Guide。表 11 按原文完整保留，不另行翻译。

Independent texture mode 使用单独的 `.samplerref` 保存 sampler property，`.texref` 中对应 field 被禁用。该模式额外提供 `force_unnormalized_coords`：为 `True` 时覆盖 texture header 的 `normalized_coords` 并使用 unnormalized coordinate；为 `False` 时沿用 texture header 设置。

编译 OpenCL 时，normalized-coordinate property 位于 sampler header。转换到 PTX 时，texture header 总是把 `normalized_coords` 初始化为 `True`，OpenCL sampler 的 `normalized_coords` flag 取反后映射为 PTX `force_unnormalized_coords`。

这些类型可以在 module scope 或 kernel entry parameter list 中声明。Module scope declaration 必须位于 `.global` state space；作为 kernel parameter 时声明在 `.param` state space。Module-scope opaque type 可以通过对具名 member 赋值的 static-expression list 初始化。

##### 重点解读

OpenCL 到 PTX 的映射存在一次逻辑取反：OpenCL 的 sampler `normalized_coords` 被映射为 PTX `force_unnormalized_coords` 的反值。阅读生成 PTX 时不能只比较 flag 名称，必须考虑其语义方向。

#### 5.3.3 Channel Data Type and Channel Order Fields

##### English Original

> The `channel_data_type` and `channel_order` fields have enumeration types corresponding to the source language API. Currently, OpenCL is the only source language that defines these fields. Table 13 and Table 12 show the enumeration values defined in OpenCL version 1.0 for channel data type and channel order.

**Table 12. OpenCL 1.0 Channel Data Type Definition**

| OpenCL enum | Value |
| --- | --- |
| `CL_SNORM_INT8` | `0x10D0` |
| `CL_SNORM_INT16` | `0x10D1` |
| `CL_UNORM_INT8` | `0x10D2` |
| `CL_UNORM_INT16` | `0x10D3` |
| `CL_UNORM_SHORT_565` | `0x10D4` |
| `CL_UNORM_SHORT_555` | `0x10D5` |
| `CL_UNORM_INT_101010` | `0x10D6` |
| `CL_SIGNED_INT8` | `0x10D7` |
| `CL_SIGNED_INT16` | `0x10D8` |
| `CL_SIGNED_INT32` | `0x10D9` |
| `CL_UNSIGNED_INT8` | `0x10DA` |
| `CL_UNSIGNED_INT16` | `0x10DB` |
| `CL_UNSIGNED_INT32` | `0x10DC` |
| `CL_HALF_FLOAT` | `0x10DD` |
| `CL_FLOAT` | `0x10DE` |

**Table 13. OpenCL 1.0 Channel Order Definition**

| OpenCL enum | Value |
| --- | --- |
| `CL_R` | `0x10B0` |
| `CL_A` | `0x10B1` |
| `CL_RG` | `0x10B2` |
| `CL_RA` | `0x10B3` |
| `CL_RGB` | `0x10B4` |
| `CL_RGBA` | `0x10B5` |
| `CL_BGRA` | `0x10B6` |
| `CL_ARGB` | `0x10B7` |
| `CL_INTENSITY` | `0x10B8` |
| `CL_LUMINANCE` | `0x10B9` |

##### 中文翻译

`channel_data_type` 和 `channel_order` field 使用与 source-language API 对应的 enumeration type。目前只有 OpenCL 定义这些 field。表 12、13 完整保留 OpenCL 1.0 定义的 channel data type 和 channel order 枚举值，不另行翻译。

### 5.4 Variables

#### English Original

> In PTX, a variable declaration describes both the variable’s type and its state space. In addition to fundamental types, PTX supports types for simple aggregate objects such as vectors and arrays.

#### 中文翻译

PTX variable declaration 同时描述 variable 的 type 和 state space。除 fundamental type 外，PTX 还支持 vector 和 array 等简单 aggregate object type。

#### 重点解读

PTX declaration 不是只有“类型 + 名称”。完整语义还包括 state space、位宽、aggregate shape、alignment、initializer、address 及 attribute；任何一项都可能影响合法访问指令、可见范围与最终内存布局。

#### 5.4.1 Variable Declarations

##### English Original

> All storage for data is specified with variable declarations. Every variable must reside in one of the state spaces enumerated in the previous section.
>
> A variable declaration names the space in which the variable resides, its type and size, its name, an optional array size, an optional initializer, and an optional fixed address for the variable.
>
> Predicate variables may only be declared in the register state space.
>
> Examples

```ptx
.global .u32 loc;
.reg    .s32 i;
.const  .f32 bias[] = {-1.0, 1.0};
.global .u8  bg[4] = {0, 0, 0, 0};
.reg    .v4 .f32 accel;
.reg    .pred p, q, r;
```

##### 中文翻译

所有数据存储都通过 variable declaration 指定。每个 variable 都必须位于上一节列出的某个 state space 中。

Variable declaration 指定 variable 所在的 space、type 和 size、name，以及可选 array size、initializer 和 fixed address。

Predicate variable 只能在 register state space 中声明。

示例依次展示 global integer、register integer、自动推导长度并初始化的 constant array、global byte array、register vector 以及 predicate register declaration。

#### 5.4.2 Vectors

##### English Original

> Limited-length vector types are supported. Vectors of length 2 and 4 of any non-predicate fundamental type can be declared by prefixing the type with `.v2` or `.v4`. Vectors must be based on a fundamental type, and they may reside in the register space. Vectors cannot exceed 128-bits in length; for example, `.v4 .f64` is not allowed. Three-element vectors may be handled by using a `.v4` vector, where the fourth element provides padding. This is a common case for three-dimensional grids, textures, etc.
>
> Examples

```ptx
.global .v4 .f32 V;   // a length-4 vector of floats
.shared .v2 .u16 uv;  // a length-2 vector of unsigned ints
.global .v4 .b8  v;   // a length-4 vector of bytes
```

> By default, vector variables are aligned to a multiple of their overall size (vector length times base-type size), to enable vector load and store instructions which require addresses aligned to a multiple of the access size.

##### 中文翻译

PTX 支持有限长度的 vector type。任意 non-predicate fundamental type 都可以添加 `.v2` 或 `.v4` 前缀，声明长度为 2 或 4 的 vector。Vector 必须基于 fundamental type，并且可以位于 register space。Vector 总长度不能超过 128 bit，因此 `.v4 .f64` 不合法。三元素 vector 可以使用 `.v4` 表示，并把第四个 element 作为 padding；这在三维 grid、texture 等场景中很常见。

示例分别声明包含四个 float 的 global vector、包含两个 unsigned integer 的 shared vector，以及包含四个 byte 的 global vector。

默认情况下，vector variable 按其整体 size（vector length × base-type size）的整数倍对齐，从而满足 vector load/store instruction 要求 address 按 access size 对齐的条件。

##### 重点解读

`.v2`/`.v4` 是短定长 aggregate，不等同于任意长度 SIMD vector。总宽度上限为 128 bit，且默认 alignment 等于整体 vector size；三维数据使用 `.v4` 时，padding element 仍会计入存储和 alignment。

#### 5.4.3 Array Declarations

##### English Original

> Array declarations are provided to allow the programmer to reserve space. To declare an array, the variable name is followed with dimensional declarations similar to fixed-size array declarations in C. The size of each dimension is a constant expression.
>
> Examples

```ptx
.local  .u16 kernel[19][19];
.shared .u8  mailbox[128];
```

> The size of the array specifies how many elements should be reserved. For the declaration of array `kernel` above, `19*19 = 361` halfwords are reserved, for a total of 722 bytes.
>
> When declared with an initializer, the first dimension of the array may be omitted. The size of the first array dimension is determined by the number of elements in the array initializer.
>
> Examples

```ptx
.global .u32 index[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
.global .s32 offset[][2] = { {-1, 0}, {0, -1}, {1, 0}, {0, 1} };
```

> Array `index` has eight elements, and array `offset` is a `4x2` array.

##### 中文翻译

Array declaration 用于预留存储空间。声明 array 时，在 variable name 后添加与 C fixed-size array 类似的维度声明；每个维度的 size 都是 constant expression。

Array size 指定要预留的 element 数量。示例中的 `kernel[19][19]` 预留 `19×19=361` 个 halfword，共 722 byte；`mailbox[128]` 在 shared state space 预留 128 个 byte。

带 initializer 声明时，可以省略第一个 array dimension，其大小由 initializer 中的 element 数量决定。示例中 `index` 有 8 个 element，`offset` 是 `4×2` array。

#### 5.4.4 Initializers

##### English Original

> Declared variables may specify an initial value using a syntax similar to C/C++, where the variable name is followed by an equals sign and the initial value or values for the variable. A scalar takes a single value, while vectors and arrays take nested lists of values inside of curly braces (the nesting matches the dimensionality of the declaration).
>
> As in C, array initializers may be incomplete, i.e., the number of initializer elements may be less than the extent of the corresponding array dimension, with remaining array locations initialized to the default value for the specified array type.
>
> Examples

```ptx
.const  .f32 vals[8] = { 0.33, 0.25, 0.125 };
.global .s32 x[3][2] = { {1,2}, {3} };
```

> is equivalent to

```ptx
.const  .f32 vals[8] = { 0.33, 0.25, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0 };
.global .s32 x[3][2] = { {1,2}, {3,0}, {0,0} };
```

> Currently, variable initialization is supported only for constant and global state spaces. Variables in constant and global state spaces with no explicit initializer are initialized to zero by default. Initializers are not allowed in external variable declarations.
>
> Variable names appearing in initializers represent the address of the variable; this can be used to statically initialize a pointer to a variable. Initializers may also contain `var+offset` expressions, where offset is a byte offset added to the address of `var`. Only variables in `.global` or `.const` state spaces may be used in initializers. By default, the resulting address is the offset in the variable’s state space (as is the case when taking the address of a variable with a `mov` instruction). An operator, `generic()`, is provided to create a generic address for variables used in initializers.
>
> Starting PTX ISA version 7.1, an operator `mask()` is provided, where mask is an integer constant. The only allowed expressions in the `mask()` operator are integer constant expression and symbol expression representing address of variable. The `mask()` operator extracts `n` consecutive bits from the expression used in initializers and inserts these bits at the lowest position of the initialized variable. The number `n` and the starting position of the bits to be extracted is specified by the integer constant mask. PTX ISA version 7.1 only supports extracting a single byte starting at byte boundary from the address of the variable. PTX ISA version 7.3 supports Integer constant expression as an operand in the `mask()` operator.
>
> Supported values for mask are: `0xFF`, `0xFF00`, `0XFF0000`, `0xFF000000`, `0xFF00000000`, `0xFF0000000000`, `0xFF000000000000`, `0xFF00000000000000`.
>
> Examples

```ptx
.const  .u32 foo = 42;
.global .u32 bar[] = { 2, 3, 5 };
.global .u32 p1 = foo;          // offset of foo in .const space
.global .u32 p2 = generic(foo); // generic address of foo

// array of generic-address pointers to elements of bar
.global .u32 parr[] = { generic(bar), generic(bar)+4,
generic(bar)+8 };

// examples using mask() operator are pruned for brevity
.global .u8 addr[] = {0xff(foo), 0xff00(foo), 0xff0000(foo), ...};

.global .u8 addr2[] = {0xff(foo+4), 0xff00(foo+4), 0xff0000(foo+4),...}

.global .u8 addr3[] = {0xff(generic(foo)), 0xff00(generic(foo)),...}

.global .u8 addr4[] = {0xff(generic(foo)+4), 0xff00(generic(foo)+4),...}

// mask() operator with integer const expression
.global .u8 addr5[] = { 0xFF(1000 + 546), 0xFF00(131187), ...};
```

> Note
>
> PTX 3.1 redefines the default addressing for global variables in initializers, from generic addresses to offsets in the global state space. Legacy PTX code is treated as having an implicit `generic()` operator for each global variable used in an initializer. PTX 3.1 code should either include explicit `generic()` operators in initializers, use `cvta.global` to form generic addresses at runtime, or load from the non-generic address using `ld.global`.
>
> Device function names appearing in initializers represent the address of the first instruction in the function; this can be used to initialize a table of function pointers to be used with indirect calls. Beginning in PTX ISA version 3.1, kernel function names can be used as initializers e.g. to initialize a table of kernel function pointers, to be used with CUDA Dynamic Parallelism to launch kernels from GPU. See the CUDA Dynamic Parallelism Programming Guide for details.
>
> Labels cannot be used in initializers.
>
> Variables that hold addresses of variables or functions should be of type `.u8` or `.u32` or `.u64`.
>
> Type `.u8` is allowed only if the `mask()` operator is used.
>
> Initializers are allowed for all types except `.f16`, `.f16x2` and `.pred`.
>
> Examples

```ptx
.global .s32 n = 10;
.global .f32 blur_kernel[][3]
               = {{.05,.1,.05},{.1,.4,.1},{.05,.1,.05}};

.global .u32 foo[] = { 2, 3, 5, 7, 9, 11 };
.global .u64 ptr = generic(foo);   // generic address of foo[0]
.global .u64 ptr = generic(foo)+8; // generic address of foo[2]
```

##### 中文翻译

已声明 variable 可以使用类似 C/C++ 的语法指定初始值：在 variable name 后写等号及一个或多个 value。Scalar 接收单个 value；vector 和 array 使用花括号内的 nested value list，其嵌套层数与 declaration dimension 对应。

与 C 相同，array initializer 可以不完整，即 initializer element 少于对应 array dimension 的范围；其余位置使用该 array type 的 default value 初始化。示例中未提供的 `.f32` 和 `.s32` element 都补零。

当前只有 constant 和 global state space 支持 variable initialization。没有显式 initializer 的 constant/global variable 默认初始化为零；external variable declaration 不允许 initializer。

Initializer 中出现的 variable name 表示该 variable 的地址，可用于静态初始化 pointer。Initializer 也可以使用 `var+offset` expression，其中 offset 是加到 `var` 地址上的 byte offset。只有 `.global` 或 `.const` variable 能用于 initializer。默认得到的地址是 variable 在其自身 state space 中的 offset；`generic()` operator 用于显式创建 generic address。

PTX ISA 7.1 引入 `mask()` operator，其中 mask 是 integer constant。`mask()` 中只允许 integer constant expression 或表示 variable address 的 symbol expression。它从 initializer expression 中提取连续的 `n` bit，并把这些 bit 放入被初始化 variable 的最低位；提取数量和起始位置由 mask 指定。PTX ISA 7.1 只支持从 variable address 的 byte boundary 提取单个 byte；PTX ISA 7.3 增加 integer constant expression operand。支持的 mask value 已在英文原文完整列出。

示例区分 state-space offset 与 generic address：`p1 = foo` 保存 `foo` 在 `.const` space 中的 offset，`p2 = generic(foo)` 保存 generic address。`parr` 保存 `bar` 各 element 的 generic-address pointer；后续示例使用 mask 从 symbol address、generic address 或 integer constant expression 中提取指定 byte。

PTX 3.1 把 initializer 中 global variable 的默认 addressing 从 generic address 改为 global-state-space offset。Legacy PTX 会被视为对每个 initializer 中的 global variable 隐式使用 `generic()`。PTX 3.1 代码应显式使用 `generic()`、在运行时用 `cvta.global` 形成 generic address，或直接用 `ld.global` 从 non-generic address 加载。

Initializer 中的 device-function name 表示该函数第一条指令的地址，可用于初始化 indirect call 所需的 function-pointer table。从 PTX ISA 3.1 开始，kernel-function name 也能作为 initializer，例如初始化 CUDA Dynamic Parallelism 从 GPU 启动 kernel 所需的 kernel-function-pointer table。Label 不能用于 initializer。

保存 variable/function address 的 variable 应使用 `.u8`、`.u32` 或 `.u64`；其中 `.u8` 只允许与 `mask()` operator 配合使用。除 `.f16`、`.f16x2` 和 `.pred` 外，其他类型都允许 initializer。

##### 重点解读

Initializer 中最容易混淆的是地址类别。裸 symbol 默认产生其 state space 内的 offset，并不必然是可跨 state space 使用的 generic pointer；需要 generic address 时必须显式写 `generic(symbol)`。这一区别会决定运行时应使用 `ld.global`、`ld.const` 等 state-space-specific instruction，还是先形成 generic address 再访问。

`mask()` 是静态 bit extraction 工具，不是一般函数调用。它主要用于把地址或常量表达式的某个 byte 提取到 `.u8` initializer 中；mask value 同时编码提取宽度和位置。

#### 5.4.5 Alignment

##### English Original

> Byte alignment of storage for all addressable variables can be specified in the variable declaration. Alignment is specified using an optional `.align` byte-count specifier immediately following the state-space specifier. The variable will be aligned to an address which is an integer multiple of byte-count. The alignment value byte-count must be a power of two. For arrays, alignment specifies the address alignment for the starting address of the entire array, not for individual elements.
>
> The default alignment for scalar and array variables is to a multiple of the base-type size. The default alignment for vector variables is to a multiple of the overall vector size.
>
> Examples

```ptx
// allocate array at 4-byte aligned address. Elements are bytes.
.const .align 4 .b8 bar[8] = {0,0,0,0,2,0,0,0};
```

> Note that all PTX instructions that access memory require that the address be aligned to a multiple of the access size. The access size of a memory instruction is the total number of bytes accessed in memory. For example, the access size of `ld.v4.b32` is 16 bytes, while the access size of `atom.f16x2` is 4 bytes.

##### 中文翻译

所有可寻址 variable 的 storage byte alignment 都可以在 declaration 中指定。可选 `.align byte-count` specifier 紧跟在 state-space specifier 后，使 variable address 对齐到 byte-count 的整数倍；byte-count 必须是 2 的幂。对于 array，alignment 约束的是整个 array 的起始地址，而不是各 element 的单独地址。

Scalar/array variable 默认按 base-type size 的整数倍对齐；vector variable 默认按整个 vector size 的整数倍对齐。

示例把由 byte element 构成的 `bar` array 起始地址对齐到 4 byte。

所有访问 memory 的 PTX instruction 都要求 address 按 access size 的整数倍对齐。Memory instruction 的 access size 是一次访问的总 byte 数，例如 `ld.v4.b32` 一次访问 16 byte，`atom.f16x2` 一次访问 4 byte。

##### 重点解读

声明对齐与访问对齐必须同时成立。即使 base element 只有 1 byte，后续若使用 16-byte vector load，实际 address 仍必须满足 16-byte alignment；array 的 `.align` 只保证 base address，带 offset 的访问还要单独证明 offset 保持所需对齐。

#### 5.4.6 Parameterized Variable Names

##### English Original

> Since PTX supports virtual registers, it is quite common for a compiler frontend to generate a large number of register names. Rather than require explicit declaration of every name, PTX supports a syntax for creating a set of variables having a common prefix string appended with integer suffixes.
>
> For example, suppose a program uses a large number, say one hundred, of `.b32` variables, named `%r0`, `%r1`, …, `%r99`. These 100 register variables can be declared as follows:

```ptx
.reg .b32 %r<100>;    // declare %r0, %r1, ..., %r99
```

> This shorthand syntax may be used with any of the fundamental types and with any state space, and may be preceded by an alignment specifier. Array variables cannot be declared this way, nor are initializers permitted.

##### 中文翻译

PTX 支持 virtual register，因此 compiler frontend 经常生成大量 register name。为避免逐个显式声明，PTX 支持用公共 prefix 加 integer suffix 的语法一次创建一组 variable。

例如，`.reg .b32 %r<100>;` 一次声明 `%r0` 到 `%r99` 共 100 个 `.b32` register variable。

这种简写可以用于任意 fundamental type 和任意 state space，并且前面可以带 alignment specifier；但不能用它声明 array，也不能提供 initializer。

#### 5.4.7 Variable Attributes

##### English Original

> Variables may be declared with an optional `.attribute` directive which allows specifying special attributes of variables. Keyword `.attribute` is followed by attribute specification inside parenthesis. Multiple attributes are separated by comma.
>
> Variable and Function Attribute Directive: `.attribute` describes the `.attribute` directive.

##### 中文翻译

Variable declaration 可以带可选 `.attribute` directive，用于指定 variable 的特殊属性。`.attribute` keyword 后面使用括号包围 attribute specification，多个 attribute 以逗号分隔。下一节详细说明 variable/function `.attribute` directive。

#### 5.4.8 Variable and Function Attribute Directive: `.attribute`

##### English Original

> `.attribute`
>
> Variable and function attributes
>
> Description
>
> Used to specify special attributes of a variable or a function.
>
> The following attributes are supported.
>
> **`.managed`**
>
> `.managed` attribute specifies that variable will be allocated at a location in unified virtual memory environment where host and other devices in the system can reference the variable directly. This attribute can only be used with variables in `.global` state space. See the CUDA UVM-Lite Programming Guide for details.
>
> **`.unified`**
>
> `.unified` attribute specifies that function has the same memory address on the host and on other devices in the system. Integer constants `uuid1` and `uuid2` respectively specify upper and lower 64 bits of the unique identifier associated with the function or the variable. This attribute can only be used on device functions or on variables in the `.global` state space. Variables with `.unified` attribute are read-only and must be loaded by specifying `.unified` qualifier on the address operand of `ld` instruction, otherwise the behavior is undefined.
>
> PTX ISA Notes
>
> - Introduced in PTX ISA version 4.0.
> - Support for function attributes introduced in PTX ISA version 8.0.
>
> Target ISA Notes
>
> - `.managed` attribute requires `sm_30` or higher.
> - `.unified` attribute requires `sm_90` or higher.
>
> Examples

```ptx
.global .attribute(.managed) .s32 g;
.global .attribute(.managed) .u64 x;

.global .attribute(.unified(19,95)) .f32 f;

.func .attribute(.unified(0xAB, 0xCD)) bar() { ... }
```

##### 中文翻译

`.attribute` 用于指定 variable 或 function 的特殊属性，当前支持 `.managed` 和 `.unified`。

`.managed` 表示 variable 分配在 unified virtual memory environment 中，使 host 和系统内其他 device 能够直接引用。它只能用于 `.global` state-space variable，详细信息参阅 CUDA UVM-Lite Programming Guide。

`.unified` 表示 function 在 host 与系统内其他 device 上具有相同 memory address。Integer constant `uuid1`、`uuid2` 分别指定与 function 或 variable 关联的 unique identifier 的高、低 64 bit。该 attribute 只能用于 device function 或 `.global` state-space variable。带 `.unified` 的 variable 是只读的，必须在 `ld` instruction 的 address operand 上指定 `.unified` qualifier 后加载，否则行为未定义。

`.attribute` 在 PTX ISA 4.0 中引入，PTX ISA 8.0 开始支持 function attribute。`.managed` 要求 `sm_30+`，`.unified` 要求 `sm_90+`。

示例声明两个 managed global variable、一个带 128-bit unique identifier 的 unified global variable，以及一个 unified device function。

##### 重点解读

`.managed` 与 `.unified` 解决不同问题：`.managed` 描述 variable 位于 host/device 可共同引用的 unified virtual memory 环境；`.unified` 强调跨 host/device 的地址身份一致，并附带只读及专用 load qualifier 要求。不能把二者当作可互换的“统一内存”标记。

### 5.5 Tensors

#### English Original

> A tensor is a multi-dimensional matrix structure in the memory. Tensor is defined by the following properties:
>
> - Dimensionality
> - Dimension sizes across each dimension
> - Individual element types
> - Tensor stride across each dimension
>
> PTX supports instructions which can operate on the tensor data. PTX Tensor instructions include:
>
> - Copying data between global and shared memories
> - Reducing the destination tensor data with the source.
>
> The Tensor data can be operated on by various `wmma.mma`, `mma` and `wgmma.mma_async` instructions.
>
> PTX Tensor instructions treat the tensor data in the global memory as a multi-dimensional structure and treat the data in the shared memory as a linear data.

#### 中文翻译

Tensor 是 memory 中的多维矩阵结构，由 dimensionality、各 dimension 的 size、单个 element type，以及各 dimension 上的 tensor stride 定义。

PTX 提供可以操作 tensor data 的 instruction，包括在 global memory 与 shared memory 之间复制数据，以及使用 source tensor data 对 destination tensor data 执行 reduction。Tensor data 还可由多种 `wmma.mma`、`mma` 和 `wgmma.mma_async` instruction 处理。

PTX tensor instruction 将 global memory 中的 tensor data 视为多维结构，而将 shared memory 中的数据视为线性数据。

#### 重点解读

Tensor copy 的两端并不采用同一种寻址视角：global memory 端由多维坐标、尺寸和 stride 描述，shared memory 端则是线性布局。Tensor-map 的作用就是把多维 global layout、访问窗口以及 shared-memory layout 变换集中编码，使 instruction 不必逐元素计算地址。

#### 5.5.1 Tensor Dimension, Size and Format

##### English Original

> Tensors can have dimensions: 1D, 2D, 3D, 4D or 5D.
>
> Each dimension has a size which represents the number of elements along the dimension. The elements can have one the following types:
>
> - Bit-sized type: `.b32`, `.b64`
> - Sub-byte types: `.b4x16`, `.b4x16_p64`, `.b6x16_p32`, `.b6p2x16`
> - Integer: `.u8`, `.u16`, `.u32`, `.s32`, `.u64`, `.s64`
> - Floating point and alternate floating point: `.f16`, `.bf16`, `.tf32`, `.f32`, `.f64` (rounded to nearest even).
>
> Tensor can have padding at the end in each of the dimensions to provide alignment for the data in the subsequent dimensions. Tensor stride can be used to specify the amount of padding in each dimension.

##### 中文翻译

Tensor 可以是 1D、2D、3D、4D 或 5D。每个 dimension 都有一个 size，表示该 dimension 上的 element 数量。Element 可以采用以下类型：bit-sized type `.b32`、`.b64`；sub-byte type `.b4x16`、`.b4x16_p64`、`.b6x16_p32`、`.b6p2x16`；integer type `.u8`、`.u16`、`.u32`、`.s32`、`.u64`、`.s64`；以及 floating-point 或 alternate floating-point type `.f16`、`.bf16`、`.tf32`、`.f32`、`.f64`，后者使用 round-to-nearest-even。

为了让后续 dimension 的数据满足 alignment，tensor 的每个 dimension 末尾都可以带 padding；tensor stride 用于指定各 dimension 的 padding 量。

##### 重点解读

这里的 stride 不只是逻辑步长，还承载物理 padding。Dimension size 决定有效 element 的范围，stride 决定相邻高维切片在 memory 中相隔多远；二者不能混用。

##### 5.5.1.1 Sub-byte Types

###### 5.5.1.1.1 Padding and Alignment of the Sub-byte Types

###### English Original

> The sub-byte types are expected to packed contiguously in the global memory and the Tensor copy instruction will expand them by appending empty spaces as shown below:
>
> **Type `.b4x16`:** With this type, there is no padding involved and the packed sixteen `.b4` elements in a 64-bits container is copied as is between the shared memory and the global memory.
>
> **Type `.b4x16_p64`:** With this type, sixteen contiguous 4-bits of data is copied from global memory to the shared memory with the append of 64-bits of padding as shown in Figure 5.
>
> `_images/tensor-dimension-size-format-sub-bytes-padding-align-b4-16-p64.png`
>
> Figure 5 Layout for `.b4x16_p64`
>
> The padded region that gets added is un-initialized.
>
> **Type `.b6x16_p32`:** With this type, sixteen 6-bits of data is copied from global memory to the shared memory with an append of 32-bits of padding as shown in Figure 6.
>
> `_images/tensor-dimension-size-format-sub-bytes-padding-align-b6-16-p32.png`
>
> Figure 6 Layout for `.b6x16_p32`
>
> The padded region that gets added is un-initialized.
>
> **Type `.b6p2x16`:** With this type, sixteen elements, each containing 6-bits of data at the LSB and 2-bits of padding at the MSB, are copied from shared memory into the global memory by discarding the 2-bits of padding data and packing the 6-bits data contiguously as shown in Figure 7.
>
> `_images/tensor-dimension-size-format-sub-bytes-padding-align-b6-p2-16.png`
>
> Figure 7 Layout for `.b6p2x16`
>
> In case of `.b6x16_p32` and `.b4x16_p64`, the padded region that gets added is un-initialized.
>
> The types `.b6x16_p32` and `.b6p2x16` share the same encoding value in the descriptor (value 15) as the two types are applicable for different types of tensor copy operations:

| Type | Valid Tensor Copy Direction |
|---|---|
| `.b6x16_p32` | `.shared::cluster.global`, `.shared::cta.global` |
| `.b6p2x16` | `.global.shared::cta` |

###### 中文翻译

Sub-byte type 在 global memory 中应连续紧密打包；tensor copy instruction 会按目标布局追加空位并展开数据。

`.b4x16` 不带 padding：16 个 `.b4` element 紧密打包在一个 64-bit container 中，在 shared memory 与 global memory 之间原样复制。

`.b4x16_p64` 从 global memory 向 shared memory 复制连续的 16 个 4-bit 数据，并在其后追加 64-bit padding，如 Figure 5 所示；新增 padding 区域未初始化。

`.b6x16_p32` 从 global memory 向 shared memory 复制 16 个 6-bit 数据，并追加 32-bit padding，如 Figure 6 所示；新增 padding 区域同样未初始化。

`.b6p2x16` 用于相反方向：shared memory 中的 16 个 element 各以低 6 bit 保存数据、高 2 bit 保存 padding。复制到 global memory 时丢弃每个 element 的高 2 bit padding，并把 6-bit 数据连续打包，如 Figure 7 所示。

`.b6x16_p32` 与 `.b6p2x16` 在 descriptor 中共享编码值 15，因为二者用于不同的 tensor copy direction；上表完整保留原文，不另作中文副表。

###### 重点解读

这些名称同时表达“有效数据宽度、批量 element 数和 shared-memory 展开形式”。其中 `_p64`、`_p32` 是整块末尾 padding，`b6p2` 则是每个 element 内部 6-bit data 加 2-bit padding。Padding 未初始化意味着程序不能把它当作零值或有效数据读取。

#### 5.5.2 Tensor Access Modes

##### English Original

> Tensor data can be accessed in two modes:
>
> **Tiled mode:**
>
> In tiled mode, the source multi-dimensional tensor layout is preserved at the destination.
>
> **Im2col mode:**
>
> In im2col mode, the elements in the Bounding Box of the source tensor are rearranged into columns at the destination. Refer here for more details.

##### 中文翻译

Tensor data 有两种访问模式。Tiled mode 在 destination 保留 source 的多维 tensor layout；im2col mode 则把 source tensor 的 Bounding Box 内的 element 重新排列为 destination 中的列。

#### 5.5.3 Tiled Mode

##### English Original

> This section talks about how Tensor and Tensor access work in tiled mode.

##### 中文翻译

本节说明 tensor 及其访问在 tiled mode 下如何工作。

##### 5.5.3.1 Bounding Box

###### English Original

> A tensor can be accessed in chunks known as Bounding Box. The Bounding Box has the same dimensionality as the tensor they are accessing into. Size of each bounding Box must be a multiple of 16 bytes. The address of the bounding Box must also be aligned to 16 bytes.
>
> Bounding Box has the following access properties:
>
> - Bounding Box dimension sizes
> - Out of boundary access mode
> - Traversal strides
>
> The tensor-coordinates, specified in the PTX tensor instructions, specify the starting offset of the bounding box. Starting offset of the bounding box along with the rest of the bounding box information together are used to determine the elements which are to be accessed.

###### 中文翻译

Tensor 可以按称为 Bounding Box 的数据块访问。Bounding Box 与其访问的 tensor 具有相同 dimensionality；每个 Bounding Box 的 size 必须是 16 byte 的整数倍，其 address 也必须按 16 byte 对齐。

Bounding Box 的访问属性包括各 dimension size、out-of-boundary access mode 和 traversal stride。PTX tensor instruction 中指定的 tensor-coordinate 给出 Bounding Box 的起始 offset；该 offset 与其余 Bounding Box 信息共同决定实际访问哪些 element。

##### 5.5.3.2 Traversal-Stride

###### English Original

> While the Bounding Box is iterating the tensor across a dimension, the traversal stride specifies the exact number of elements to be skipped. If no jump over is required, default value of 1 must be specified.
>
> The traversal stride in dimension 0 can be used for the Interleave layout. For non-interleaved layout, the traversal stride in dimension 0 must always be 1.
>
> Figure 8 illustrates tensor, tensor size, tensor stride, Bounding Box size and traversal stride.
>
> `_images/tensor-tiled-mode-bounding-box-example.png`
>
> Figure 8 Tiled mode bounding box, tensor size and traversal stride

###### 中文翻译

当 Bounding Box 沿某个 dimension 迭代 tensor 时，traversal stride 指定每一步跨越的 element 数；若无需跳过 element，也必须指定默认值 1。

Dimension 0 的 traversal stride 可用于 interleave layout；对于 non-interleaved layout，dimension 0 的 traversal stride 必须始终为 1。Figure 8 展示 tensor、tensor size、tensor stride、Bounding Box size 与 traversal stride 的关系。

##### 5.5.3.3 Out of Boundary Access

###### English Original

> PTX Tensor operation can detect and handle the case when the Bounding Box crosses the tensor boundary in any dimension. There are 2 modes:
>
> **Zero fill mode:**
>
> Elements in the Bounding Box which fall outside of the tensor boundary are set to 0.
>
> **OOB-NaN fill mode:**
>
> Elements in the Bounding Box which fall outside of the tensor boundary are set to a special NaN called OOB-NaN.
>
> Figure 9 shows an example of the out of boundary access.
>
> `_images/tensor-oob-access.png`
>
> Figure 9 Out of boundary access

###### 中文翻译

当 Bounding Box 在任意 dimension 上越过 tensor boundary 时，PTX tensor operation 可以检测并处理这种情况。Zero-fill mode 将 Bounding Box 中越界的 element 置 0；OOB-NaN-fill mode 则将其置为一种特殊 NaN，即 OOB-NaN。Figure 9 给出了 out-of-boundary access 示例。

###### 重点解读

OOB fill 让边界 tile 可以继续使用固定形状，不必由每个 thread 手写边界分支。Zero fill 适合以零为单位元或 padding 值的运算；OOB-NaN 能让错误或无效值在浮点计算中更容易传播和暴露。

##### 5.5.3.4 `.tile::scatter4` and `.tile::gather4` Modes

###### English Original

> These modes are similar to the tiled mode with restriction that these modes work only on 2D tensor data. `Tile::scatter4` and `Tile::gather4` modes are used to access multiple non-contiguous rows of tensor data.
>
> In `Tile::scatter4` mode single 2D source tensor is divided into four rows in the 2D destination tensor. In `Tile::gather4` mode four rows in the source 2D tensor are combined to form single 2D destination tensor.
>
> These modes work on four rows and hence the instruction will take:
>
> - four tensor coordinates across the dimension 0
> - one tensor coordinate across the dimension 1
>
> The interleave layout is not supported for `.tile::scatter4` and `.tile::gather4` modes.
>
> All other constraints and rules of the tile mode apply to these modes as well.

###### 中文翻译

这两种 mode 与 tiled mode 类似，但仅适用于 2D tensor data，用来访问多行不连续的 tensor data。`Tile::scatter4` 将单个 2D source tensor 分到 2D destination tensor 的四行；`Tile::gather4` 则把 source 2D tensor 的四行组合为一个 2D destination tensor。

由于一次处理四行，instruction 接收 dimension 0 上的四个 tensor coordinate，以及 dimension 1 上的一个 tensor coordinate。二者均不支持 interleave layout，其余 tiled-mode constraint 与 rule 仍然适用。

###### 5.5.3.4.1 Bounding Box

###### English Original

> For `Tile::scatter4` and `Tile::gather4` modes, four request coordinates will form four Bounding Boxes in the tensor space.
>
> Figure 10 shows an example of the same with start coordinates `(1, 2)`, `(1, 5)`, `(1, 0)` and `(1, 9)`.
>
> The size of the bounding box in the dimension 0 represents the length of the rows. The size of the bounding box in the dimension 1 must be one.
>
> `_images/tiled-scatter4-gather4-bounding-box.png`
>
> Figure 10 `tiled::scatter4`/`tiled::gather4` mode bounding box example

###### 中文翻译

在 `Tile::scatter4` 与 `Tile::gather4` mode 中，四个 request coordinate 会在 tensor space 中形成四个 Bounding Box。Figure 10 的起始 coordinate 分别为 `(1, 2)`、`(1, 5)`、`(1, 0)` 和 `(1, 9)`。Bounding Box 在 dimension 0 上的 size 表示行长度，在 dimension 1 上的 size 必须为 1。

#### 5.5.4 `im2col` Mode

##### English Original

> Im2col mode supports the following tensor dimensions: 3D, 4D and 5D. In this mode, the tensor data is treated as a batch of images with the following properties:
>
> - N: number of images in the batch
> - D, H, W: size of a 3D image (depth, height and width)
> - C: channels per image element
>
> The above properties are associated with 3D, 4D and 5D tensors as follows:

| Dimension | N/D/H/W/C applicability |
|---|---|
| 3D | NWC |
| 4D | NHWC |
| 5D | NDHWC |

##### 中文翻译

Im2col mode 支持 3D、4D 和 5D tensor。在该 mode 中，tensor data 被视为一批 image：N 表示 batch 中的 image 数；D、H、W 表示 3D image 的 depth、height 和 width；C 表示每个 image element 的 channel 数。不同 dimensionality 与这些属性的对应关系见上表；表格完整保留原文，不另作中文副表。

##### 5.5.4.1 Bounding Box

###### English Original

> In im2col mode, the Bounding Box is defined in DHW space. Boundaries along other dimensions are specified by Pixels-per-Column and Channels-per-Pixel parameters as described below.
>
> The dimensionality of the Bounding Box is two less than the tensor dimensionality.
>
> The following properties describe how to access of the elements in im2col mode:
>
> - Bounding-Box Lower-Corner
> - Bounding-Box Upper-Corner
> - Pixels-per-Column
> - Channels-per-Pixel
>
> Bounding-box Lower-Corner and Bounding-box Upper-Corner specify the two opposite corners of the Bounding Box in the DHW space. Bounding-box Lower-Corner specifies the corner with the smallest coordinate and Bounding-box Upper-Corner specifies the corner with the largest coordinate.
>
> Bounding-box Upper- and Lower-Corners are 16-bit signed values whose limits varies across the dimensions and are as shown below:

|  | 3D | 4D | 5D |
|---|---|---|---|
| Upper- / Lower- Corner sizes | `[-2^15, 2^15-1]` | `[-2^7, 2^7-1]` | `[-2^4, 2^4-1]` |

> Figure 11 and Figure 12 show the Upper-Corners and Lower-Corners.
>
> `_images/tensor-im2col-mode-bounding-box1.png`
>
> Figure 11 im2col mode bounding box example 1
>
> `_images/tensor-im2col-mode-bounding-box2.png`
>
> Figure 12 im2col mode bounding box example 2
>
> The Bounding-box Upper- and Lower-Corners specify only the boundaries and not the number of elements to be accessed. Pixels-per-Column specifies the number of elements to be accessed in the NDHW space.
>
> Channels-per-Pixel specifies the number of elements to access across the C dimension.
>
> The tensor coordinates, specified in the PTX tensor instructions, behaves differently in different dimensions:
>
> - Across N and C dimensions: specify the starting offsets along the dimension, similar to the tiled mode.
> - Across DHW dimensions: specify the location of the convolution filter base in the tensor space. The filter corner location must be within the bounding box.
>
> The im2col offsets, specified in the PTX tensor instructions in im2col mode, are added to the filter base coordinates to determine the starting location in the tensor space from where the elements are accessed.
>
> The size of the im2col offsets varies across the dimensions and their valid ranges are as shown below:

|  | 3D | 4D | 5D |
|---|---|---|---|
| im2col offsets range | `[0, 2^16-1]` | `[0, 2^8-1]` | `[0, 2^5-1]` |

> Following are some examples of the im2col mode accesses:
>
> Example 1 (Figure 13):

```text
Tensor Size[0] = 64
Tensor Size[1] = 9
Tensor Size[2] = 14
Tensor Size[3] = 64
Pixels-per-Column = 64
channels-per-pixel = 8
Bounding-Box Lower-Corner W = -1
Bounding-Box Lower-Corner H = -1
Bounding-Box Upper-Corner W = -1
Bounding-Box Upper-Corner H = -1.

tensor coordinates = (7, 7, 4, 0)
im2col offsets : (0, 0)
```

> `_images/tensor-im2col-mode-example1.png`
>
> Figure 13 im2col mode example 1
>
> Example 2 (Figure 14):

```text
Tensor Size[0] = 64
Tensor Size[1] = 9
Tensor Size[2] = 14
Tensor Size[3] = 64
Pixels-per-Column = 64
channels-per-pixel = 8
Bounding-Box Lower-Corner W = 0
Bounding-Box Lower-Corner H = 0
Bounding-Box Upper-Corner W = -2
Bounding-Box Upper-Corner H = -2

tensor coordinates = (7, 7, 4, 0)
im2col offsets: (2, 2)
```

> `_images/tensor-im2col-mode-example2.png`
>
> Figure 14 im2col mode example 2

###### 中文翻译

在 im2col mode 中，Bounding Box 定义在 DHW space；其他 dimension 的边界由 Pixels-per-Column 与 Channels-per-Pixel parameter 指定。Bounding Box 的 dimensionality 比 tensor dimensionality 少 2。

Im2col element access 由 Bounding-Box Lower-Corner、Bounding-Box Upper-Corner、Pixels-per-Column 和 Channels-per-Pixel 描述。Lower-Corner 与 Upper-Corner 指定 DHW space 中 Bounding Box 的两个对角，前者是 coordinate 最小的角，后者是 coordinate 最大的角。两者都是 16-bit signed value，但不同 tensor dimensionality 下可用 bit 数不同，具体范围见上表；原网页的指数在纯文本附件中丢失，这里恢复为 `2^n` 记法。

Figure 11 与 Figure 12 展示 Upper-Corner 和 Lower-Corner。它们只规定边界，并不规定访问的 element 数量；Pixels-per-Column 指定 NDHW space 中要访问的 element 数，Channels-per-Pixel 指定 C dimension 上要访问的 element 数。

PTX tensor instruction 中的 tensor coordinate 在不同 dimension 上含义不同：N、C dimension 上表示起始 offset，与 tiled mode 类似；DHW dimension 上则指定 tensor space 中 convolution filter base 的位置，而且 filter corner 必须位于 Bounding Box 内。Instruction 指定的 im2col offset 会加到 filter-base coordinate 上，从而得到 tensor space 中开始访问 element 的位置；其有效范围见上表。

Example 1 和 Example 2 分别对应 Figure 13、Figure 14。参数块保留原文，以便逐项对照实现。

###### 重点解读

Bounding Box 定义“卷积窗口允许落入的相对范围”，Pixels-per-Column 定义“最终取多少个空间位置”，Channels-per-Pixel 定义“每个位置取多少个 channel”。因此 corners 并不等同于访问计数；tensor coordinate 与 im2col offset 也分工不同：前者给 filter base，后者在调用时对 base 做偏移。

##### 5.5.4.2 Traversal Stride

###### English Original

> The traversal stride, in im2col mode, does not impact the total number of elements (or pixels) being accessed unlike the tiled mode. Pixels-per-Column determines the total number of elements being accessed, in im2col mode.
>
> The number of elements traversed along the D, H and W dimensions is strided by the traversal stride for that dimension.
>
> The following example with Figure 15 illustrates accesses with traversal-strides:

```text
Tensor Size[0] = 64
Tensor Size[1] = 8
Tensor Size[2] = 14
Tensor Size[3] = 64
Traversal Stride = 2
Pixels-per-Column = 32
channels-per-pixel = 16
Bounding-Box Lower-Corner W = -1
Bounding-Box Lower-Corner H = -1
Bounding-Box Upper-Corner W = -1
Bounding-Box Upper-Corner H = -1.
Tensor coordinates in the instruction = (7, 7, 5, 0)
Im2col offsets in the instruction : (1, 1)
```

> `_images/tensor-im2col-mode-example3.png`
>
> Figure 15 im2col mode traversal stride example

###### 中文翻译

与 tiled mode 不同，im2col mode 的 traversal stride 不改变访问的 element 或 pixel 总数；总数由 Pixels-per-Column 决定。Traversal stride 只改变沿 D、H、W dimension 遍历 element 时每一步的间隔。Figure 15 用上述参数展示带 traversal stride 的访问。

##### 5.5.4.3 Out of Boundary Access

###### English Original

> In im2col mode, when the number of requested pixels in NDHW space specified by Pixels-per-Column exceeds the number of available pixels in the image batch then out-of-bounds access is performed.
>
> Similar to tiled mode, zero fill or OOB-NaN fill can be performed based on the Fill-Mode specified.

###### 中文翻译

在 im2col mode 中，如果 Pixels-per-Column 指定的 NDHW-space requested pixel 数超过 image batch 中可用的 pixel 数，就会发生 out-of-bounds access。与 tiled mode 一样，可根据指定的 Fill-Mode 执行 zero fill 或 OOB-NaN fill。

#### 5.5.5 `im2col::w`, `im2col_no_offs::w` and `im2col::w::128` Modes

##### English Original

> These modes are similar to the im2col mode with the restriction that elements are accessed across the W dimension only while keeping the H and D dimension constant.
>
> All the constraints and rules of the im2col mode apply to these modes as well. Note that for these modes, the legal Swizzling Modes is required. In other words, swizzling mode must not be set to (i) no swizzle and (ii) 128-byte swizzle mode with 32-byte atomicity with 8-byte flip.
>
> The number of elements accessed in the `im2col::w::128` mode is fixed and is equal to 128. The number of elements accessed in the `im2col::w` and `im2col_no_offs::w` modes depend on the Pixels-per-Column field in the TensorMap.

##### 中文翻译

这些 mode 与 im2col mode 类似，但只能沿 W dimension 访问 element，同时保持 H、D dimension 不变。Im2col mode 的全部 constraint 与 rule 仍然适用。

这些 mode 必须使用合法的 swizzling mode：不能选择 no-swizzle，也不能选择带 32-byte atomicity 与 8-byte flip 的 128-byte swizzle。`im2col::w::128` 固定访问 128 个 element；`im2col::w` 与 `im2col_no_offs::w` 的访问数量由 TensorMap 中的 Pixels-per-Column field 决定。

##### 5.5.5.1 Bounding Box

###### English Original

> In these modes, the size of the bounding box in D and H dimensions are 1.
>
> The D and H dimensions in the tensor coordinates argument in the PTX instruction specify the position of the bounding box in the tensor space.
>
> The Bounding-Box Lower-Corner-W and Bounding-Box Upper-Corner-W specify the two opposite corners of the Bounding Box in the W dimension.
>
> The W dimension in the tensor coordinates argument in the PTX instruction specify the location of the first element that is to be accessed in the bounding box.
>
> Number of pixels loaded in `im2col::w` mode and number of pixels stored or otherwise reduced in `im2col_no_offs::w` mode is specified by Pixels-per-Column in the TensorMap. Number of pixels loaded in `im2col::w::128` mode is always 128. So, Pixels-per-Column is ignored in `im2col::w::128` mode.
>
> Figure 16 shows an example of the `im2col::w` and `im2col::w:128` modes.
>
> `_images/tensor-im2col-w-w128-modes-example.png`
>
> Figure 16 `im2col::w` and `im2col::w::128` modes example
>
> The first element can lie outside of the Bounding Box in the W-dimension only and only on the left side of the Bounding Box. Figure 17 shows of an example of this.
>
> `_images/tensor-im2col-w-w128-modes-example2.png`
>
> Figure 17 `im2col::w` and `im2col::w::128` modes first element outside Bounding Box example
>
> For `im2col_no_offs::w` mode, since the tensor data is being stored or otherwise reduced to global memory, the bounding box must always stay within the tensor boundary limit. In other words, the numeric value for Lower-Corner-W must be non-negative while the value for Upper-Corner-W must be non-positive. In this mode, it also naturally follows that the negative values for tensor coordinates would result in illegal out-of-bounds access and runtime error is raised in such cases. Additionally, note that once out of boundary access is detected, subsequent pixel operations would not be performed even if the total count of the pixel operated is less than the Pixels-per-Column value specified in the TensorMap.
>
> Figure 18 shows an example of the `im2col_no_offs::w` mode.
>
> `_images/tensor-w-no-offs-mode-example.png`
>
> Figure 18 `im2col_no_offs::w` mode example

###### 中文翻译

在这些 mode 中，Bounding Box 在 D、H dimension 上的 size 都是 1。PTX instruction 的 tensor-coordinate argument 中，D、H dimension 指定 Bounding Box 在 tensor space 中的位置；Bounding-Box Lower-Corner-W 与 Upper-Corner-W 指定 W dimension 上的两个对角；W coordinate 则指定 Bounding Box 中第一个待访问 element 的位置。

`im2col::w` 的 load pixel 数，以及 `im2col_no_offs::w` 的 store 或其他 reduction pixel 数，由 TensorMap 的 Pixels-per-Column 指定。`im2col::w::128` 始终 load 128 个 pixel，因此忽略 Pixels-per-Column。Figure 16 展示 `im2col::w` 与 `im2col::w::128`。

第一个 element 只能在 W dimension 上、且只能从 Bounding Box 左侧落在其外部，Figure 17 展示这种情况。

对于 `im2col_no_offs::w`，tensor data 会被 store 或 reduction 到 global memory，因此 Bounding Box 必须始终处于 tensor boundary 内：Lower-Corner-W 的数值必须非负，Upper-Corner-W 必须非正。负 tensor coordinate 会造成非法 OOB access，并触发 runtime error。一旦检测到 OOB，后续 pixel operation 不再执行，即使已处理 pixel 总数仍小于 TensorMap 中的 Pixels-per-Column。Figure 18 给出该 mode 的示例。

##### 5.5.5.2 Traversal Stride

###### English Original

> This is similar to im2col mode with the exception of that the number of elements traversed along only the W dimension is strided by the traversal stride as specified in the TensorMap.

###### 中文翻译

其行为与 im2col mode 类似，但 TensorMap 指定的 traversal stride 只作用于 W dimension 上的 element 遍历。

##### 5.5.5.3 `wHalo`

###### English Original

> In `im2col::w` mode, the `wHalo` argument in the PTX instruction specifies how many filter halo elements must be loaded at the end of the image.
>
> In `im2col::w::128` mode, the halo elements are loaded after every 32 elements in the bounding box along the W dimension. The `wHalo` argument in the PTX instruction specifies how many halo elements must be loaded after every 32 elements.
>
> Note that for `im2col_no_offs::w` mode, it is illegal to explicitly specify `wHalo` value and instead it always defaults to 0.
>
> Following is an example of `.im2col::w` mode access:

```text
Tensor Size [0] = 128
Tensor Size [1] = 9
Tensor Size [2] = 7
Tensor Size [3] = 64
Pixels-per-column = 128
Channels-per-pixel = 64
Bounding Box Lower Corner W = 0
Bounding Box Upper Corner W = 0

Tensor Coordinates in the instruction = (7, 2, 3, 0)
wHalo in the instruction = 2 (as 3x3 convolution filter is used)
```

> A tensor copy operation with the above parameters loads 128 pixels and the two halo pixels as shown in Figure 19.
>
> `_images/tensor-im2col-w-w128-modes-example3.png`
>
> Figure 19 tensor copy operation with `im2col::w` mode example
>
> The halo pixels are always loaded in the shared memory next to the main row pixels as shown in Figure 19.
>
> Following is an example of `.im2col::w::128` mode access:

```text
Tensor Size [0] = 128
Tensor Size [1] = 9
Tensor Size [2] = 7
Tensor Size [3] = 64
Channels-per-pixel = 64
Bounding Box Lower Corner W = 0
Bounding Box Upper Corner W = 0

Tensor Coordinates in the instruction = (7, 2, 3, 0)
wHalo in the instruction = 2 (as 3x3 convolution filter is used)
```

> A tensor copy operation with the above parameters loads 128 elements such that after every 32 elements, `wHalo` number of elements are loaded as shown in Figure 20.
>
> `_images/tensor-im2col-w-w128-modes-example4.png`
>
> Figure 20 tensor copy operation with `im2col::w::128` mode example

###### 中文翻译

在 `im2col::w` mode 中，PTX instruction 的 `wHalo` argument 指定 image 末尾需要额外 load 多少个 filter halo element。

在 `im2col::w::128` mode 中，沿 W dimension 每经过 Bounding Box 中的 32 个 element 就 load 一组 halo element；`wHalo` 指定每 32 个 element 后 load 的 halo element 数。对于 `im2col_no_offs::w`，显式指定 `wHalo` 是非法的，其值始终默认为 0。

第一个示例使用 `im2col::w`，按上述参数 load 128 个 pixel 和 2 个 halo pixel，如 Figure 19 所示；halo pixel 在 shared memory 中始终紧邻 main-row pixel。第二个示例使用 `im2col::w::128`，固定 load 128 个 element，并在每 32 个 element 后 load `wHalo` 个 element，如 Figure 20 所示。

##### 5.5.5.4 `wOffset`

###### English Original

> In the convolution calculations, the same elements along the W dimension are reused for different locations within the convolution filter footprint. Based on the number of times a pixel is used, the pixels may be loaded into different shared memory buffers. Each buffer can be loaded by a separate tensor copy operation.
>
> The `wOffset` argument in the tensor copy and prefetch instruction adjusts the source pixel location for each buffer. The exact position of the buffer is adjusted along the W dimension using the following formula:

```text
Bounding Box Lower Corner W += wOffset
Bounding Box Upper Corner W += wOffset
W += wOffset
```

> Note that for `im2col_no_offs::w` mode, it is illegal to explicitly specify `wOffset` value and instead it always defaults to 0.
>
> Following are examples of tensor copy to multiple buffers with various `wHalo` and `wOffset` values:
>
> Example 1:

```text
Tensor Size [0] = 128
Tensor Size [1] = 9
Tensor Size [2] = 67
Tensor Size [3] = 64
Pixels-per-Column = 128
Channels-per-pixel = 64
Bounding Box Lower Corner W = -1
Bounding Box Upper Corner W = 0
Traversal Stride = 2

Tensor Coordinates in the instruction = (7, 2, -1, 0)

Shared memory buffer 1:
   wHalo = 1
   wOffset = 0

Shared memory buffer 2:
   wHalo = 0
   wOffset = 1
```

> `_images/tensor-im2col-w-w128-modes-example5.png`
>
> Figure 21 tensor copy operation to buffer 1 of Example 1
>
> `_images/tensor-im2col-w-w128-modes-example6.png`
>
> Figure 22 tensor copy operation to buffer 2 of Example 1
>
> Example 2:

```text
Tensor Size [0] = 128
Tensor Size [1] = 7
Tensor Size [2] = 7
Tensor Size [3] = 64
Pixels-per-Column = 128
Channels-per-pixel = 64
Bounding Box Lower Corner W = -1
Bounding Box Upper Corner W = -1
Traversal Stride = 3

Tensor Coordinates in the instruction = (7, 2, -1, 0)

Shared memory buffer 1:
   wHalo = 0
   wOffset = 0

Shared memory buffer 2:
   wHalo = 0
   wOffset = 1

Shared memory buffer 3:
   wHalo = 0
   wOffset = 2
```

> `_images/tensor-im2col-w-w128-modes-example7.png`
>
> Figure 23 tensor copy operation to buffer 1 of Example 2
>
> `_images/tensor-im2col-w-w128-modes-example8.png`
>
> Figure 24 tensor copy operation to buffer 2 of Example 2
>
> `_images/tensor-im2col-w-w128-modes-example9.png`
>
> Figure 25 tensor copy operation to buffer 3 of Example 2

###### 中文翻译

在 convolution calculation 中，W dimension 上的同一批 element 会在 convolution-filter footprint 的不同位置重复使用。根据 pixel 的复用次数，可以把它们 load 到不同 shared-memory buffer；每个 buffer 可由独立 tensor copy operation 填充。

Tensor copy 与 prefetch instruction 的 `wOffset` argument 用于调整各 buffer 的 source-pixel location。它会把 Bounding Box Lower Corner W、Bounding Box Upper Corner W 和 W coordinate 同时加上 `wOffset`。对于 `im2col_no_offs::w`，显式指定 `wOffset` 非法，其值始终默认为 0。

Example 1 使用两个 shared-memory buffer：buffer 1 取 `wHalo=1, wOffset=0`，buffer 2 取 `wHalo=0, wOffset=1`，对应 Figure 21、22。Example 2 使用三个 buffer，`wOffset` 依次为 0、1、2，均无 halo，对应 Figure 23–25。参数块与图路径完整保留，便于对照每个 buffer 的 source window。

###### 重点解读

`wHalo` 解决卷积行末额外邻域数据的装载，`wOffset` 解决同一 source row 为不同 filter position 生成不同 shared-memory buffer 的问题。后者同时移动两个 corner 和 W coordinate，因此移动的是整套 W-direction access window，而不是只修改首地址。

#### 5.5.6 Interleave Layout

##### English Original

> Tensor can be interleaved and the following interleave layouts are supported:
>
> - No interleave (NDHWC)
> - 8 byte interleave (NC/8DHWC8): C8 utilizes 16 bytes in memory assuming 2B per channel.
> - 16 byte interleave (NC/16HWC16): C16 utilizes 32 bytes in memory assuming 4B per channel.
>
> The C information is organized in slices where sequential C elements are grouped in 16 byte or 32 byte quantities.
>
> If the total number of channels is not a multiple of the number of channels per slice, then the last slice must be padded with zeros to make it complete 16B or 32B slice.
>
> Interleaved layouts are supported only for the dimensionalities: 3D, 4D and 5D.
>
> The interleave layout is not supported for `.im2col::w` and `.im2col::w::128` modes.

##### 中文翻译

Tensor 支持以下 interleave layout：不交错的 NDHWC；8-byte interleave `NC/8DHWC8`，假定每 channel 为 2B 时 C8 在 memory 中占 16 byte；16-byte interleave `NC/16HWC16`，假定每 channel 为 4B 时 C16 占 32 byte。

C 信息按 slice 组织，连续 C element 被分组为 16-byte 或 32-byte 单元。如果 channel 总数不是每 slice channel 数的整数倍，最后一个 slice 必须用零 padding 补成完整 16B 或 32B slice。

Interleaved layout 仅支持 3D、4D、5D tensor，不支持 `.im2col::w` 与 `.im2col::w::128` mode。

##### 重点解读

Interleave 的目的，是把一组 channel 变成适合固定宽度 memory transaction 与 shared-memory access 的 slice。这里的 `C8`、`C16` 表示每组 channel 数，而实际 byte 数还取决于每 channel 的宽度；最后一个不足整组的 slice 必须显式 zero-pad。

#### 5.5.7 Swizzling Modes

##### English Original

> The layout of the data in the shared memory can be different to that of global memory, for access performance reasons. The following describes various swizzling modes:
>
> **No swizzle mode:**
>
> There is no swizzling in this mode and the destination data layout is exactly similar to the source data layout.

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |

> … Pattern repeats …
>
> **32 byte swizzle mode:**
>
> The following table, where each elements (numbered cell) is 16 byte and the starting address is 256 bytes aligned, shows the pattern of the destination data layout:

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 3 | 2 | 5 | 4 | 7 | 6 |

> … Pattern repeats …
>
> An example of the 32 byte swizzle mode for NC/(32B)HWC(32B) tensor of 1x2x10x10xC16 dimension, with the innermost dimension holding slice of 16 channels with 2 byte/channel, is shown in Figure 26.
>
> `_images/tensor-32B-swizzle.png`
>
> Figure 26 32-byte swizzle mode example
>
> Figure 27 shows the two fragments of the tensor: one for C/(32B) = 0 and another for C/(32B) = 1.
>
> `_images/tensor-32B-swizzle-frag.png`
>
> Figure 27 32-byte swizzle mode fragments
>
> Figure 28 shows the destination data layout with 32 byte swizzling.
>
> `_images/tensor-32B-swizzle-dst.png`
>
> Figure 28 32-byte swizzle mode destination data layout
>
> **64 byte swizzle mode:**
>
> The following table, where each elements (numbered cell) is 16 byte and the starting address is 512 bytes aligned, shows the pattern of the destination data layout:

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 3 | 2 | 5 | 4 | 7 | 6 |
| 2 | 3 | 0 | 1 | 6 | 7 | 4 | 5 |
| 3 | 2 | 1 | 0 | 7 | 6 | 5 | 4 |

> … Pattern repeats …
>
> An example of the 64 byte swizzle mode for NHWC tensor of 1x10x10x64 dimension, with 2 bytes / channel and 32 channels, is shown in Figure 29.
>
> `_images/tensor-64B-swizzle.png`
>
> Figure 29 64-byte swizzle mode example
>
> Each colored cell represents 8 channels. Figure 30 shows the source data layout.
>
> `_images/tensor-64B-swizzle-src.png`
>
> Figure 30 64-byte swizzle mode source data layout
>
> Figure 31 shows the destination data layout with 64 byte swizzling.
>
> `_images/tensor-64B-swizzle-dst.png`
>
> Figure 31 64-byte swizzle mode destination data layout
>
> **96 byte swizzle mode:**
>
> The following table where each element (numbered cell) is 16 byte shows the swizzling pattern at the destination data layout:

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 3 | 2 | 5 | 4 | 7 | 6 |

> … Pattern repeats …
>
> An example of the data layout in global memory and its swizzled data layout in shared memory where each element (colored cell) is 16 bytes and the starting address is 256 bytes aligned is shown in Figure 32.
>
> `_images/tensor-96B-swizzle.png`
>
> Figure 32 96-byte swizzle mode example
>
> **128 byte swizzle mode:**
>
> The 128-byte swizzling mode supports the following sub-modes:
>
> **16-byte atomicity sub-mode:**
>
> In this sub-mode, the 16-byte of data is kept intact while swizzling.
>
> The following table, where each elements (numbered cell) is 16 byte and the starting address is 1024 bytes aligned, shows the pattern of the destination data layout:

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 3 | 2 | 5 | 4 | 7 | 6 |
| 2 | 3 | 0 | 1 | 6 | 7 | 4 | 5 |
| 3 | 2 | 1 | 0 | 7 | 6 | 5 | 4 |
| 4 | 5 | 6 | 7 | 0 | 1 | 2 | 3 |
| 5 | 4 | 7 | 6 | 1 | 0 | 3 | 2 |
| 6 | 7 | 4 | 5 | 2 | 3 | 0 | 1 |
| 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |

> … Pattern repeats …
>
> An example of the 128 byte swizzle mode for NHWC tensor of 1x10x10x64 dimension, with 2 bytes / channel and 64 channels, is shown in Figure 33.
>
> `_images/tensor-128B-swizzle.png`
>
> Figure 33 128-byte swizzle mode example
>
> Each colored cell represents 8 channels. Figure 34 shows the source data layout.
>
> `_images/tensor-128B-swizzle-src.png`
>
> Figure 34 128-byte swizzle mode source data layout
>
> Figure 35 shows the destination data layout with 128 byte swizzling.
>
> `_images/tensor-128B-swizzle-dst.png`
>
> Figure 35 128-byte swizzle mode destination data layout
>
> **32-byte atomicity sub-mode:**
>
> In this sub-mode, the 32-byte of data is kept intact while swizzling.
>
> The following table where each element (numbered cell) is 16 byte shows the swizzling pattern at the destination data layout:

| 0 1 | 2 3 | 4 5 | 6 7 |
|---|---|---|---|
| 2 3 | 0 1 | 6 7 | 4 5 |
| 4 5 | 6 7 | 0 1 | 2 3 |
| 6 7 | 4 5 | 2 3 | 0 1 |

> … Pattern repeats …
>
> This sub-mode requires 32 byte alignment at shared memory.
>
> An example of the data layout in global memory and its swizzled data layout in shared memory where each element (colored cell) is 16 bytes is shown in Figure 36.
>
> `_images/tensor-128B-swizzle-32B-atom.png`
>
> Figure 36 128-byte swizzle mode example with 32-byte atomicity
>
> **32-byte atomicity with 8-byte flip sub-mode:**
>
> The swizzling pattern for this sub-mode is similar to the 32-byte atomicity sub-mode except that there is a flip of adjacent 8-bytes within the 16-byte data at every alternate shared memory line. Note that this mode is legal only when `cp.async.bulk.tensor` specifies the copy direction as `.shared::cluster.global` or otherwise `.shared::cta.global`.
>
> An example of the data layout in global memory and its swizzled data layout in shared memory where each element (colored cell) is 16 bytes (two 8-byte sub-elements for each 16-byte colored cell are shown to show the flip) is shown in Figure 37.
>
> `_images/tensor-128B-swizzle-32B-atom-8B-flip.png`
>
> Figure 37 128-byte swizzle mode example with 32-byte atomicity with 8-byte flip
>
> **64-byte atomicity sub-mode:**
>
> In this sub-mode, the 64-byte of data is kept intact while swizzling.
>
> The following table where each element (numbered cell) is 16 byte shows the swizzling pattern at the destination data layout:

| 0 1 2 3 | 4 5 6 7 |
|---|---|
| 4 5 6 7 | 0 1 2 3 |

> … Pattern repeats …
>
> This sub-mode requires 64-byte alignment at shared memory.
>
> An example of the data layout in global memory and its swizzled data layout in shared memory where each element (colored cell) is 16 bytes is shown in Figure 38.
>
> `_images/tensor-128B-swizzle-64B-atom.png`
>
> Figure 38 128-byte swizzle mode example with 64-byte atomicity
>
> Table 14 lists the valid combination of swizzle-atomicity with the swizzling-mode.
>
> Table 14 Valid combination of swizzle-atomicity with swizzling-mode

| Swizzling Mode | Swizzle-Atomicity |
|---|---|
| No Swizzling | – |
| 32B Swizzling Mode | 16B |
| 64B Swizzling Mode | 16B |
| 96B Swizzling Mode | 16B |
| 128B Swizzling Mode | 16B, 32B, 32B + 8B-flip, 64B |

> The value of swizzle base offset is 0 when the dstMem shared memory address is located at the following boundary:

| Swizzling Mode | Starting address of the repeating pattern |
|---|---|
| 128-Byte swizzle | 1024-Byte boundary |
| 96-Byte swizzle | 256-Byte boundary |
| 64-Byte swizzle | 512-Byte boundary |
| 32-Byte swizzle | 256-Byte boundary |

> Otherwise, the swizzle base offset is a non-zero value, computed using following formula:

| Swizzling Mode | Formula |
|---|---|
| 128-Byte swizzle | `base offset = (dstMem / 128) % 8` |
| 96-Byte swizzle | `base offset = (dstMem / 128) % 2` |
| 64-Byte swizzle | `base offset = (dstMem / 128) % 4` |
| 32-Byte swizzle | `base offset = (dstMem / 128) % 2` |

##### 中文翻译

出于 access performance 的考虑，shared memory 中的数据布局可以不同于 global memory。PTX 为此提供多种 swizzling mode。

**No-swizzle mode：**不执行 swizzle，destination data layout 与 source data layout 完全相同；上表中的编号顺序不变并重复出现。

**32-byte swizzle mode：**每个编号 cell 表示 16 byte，starting address 按 256 byte 对齐。相邻 shared-memory line 将每对 cell 的次序交换，pattern 按表中两行重复。Figure 26 以 `NC/(32B)HWC(32B)`、`1x2x10x10xC16` tensor 为例，最内层 dimension 是 16 个 channel 的 slice，每 channel 2 byte；Figure 27 展示 `C/(32B)=0` 与 `C/(32B)=1` 两个 fragment；Figure 28 展示 32-byte swizzling 后的 destination layout。

**64-byte swizzle mode：**每个编号 cell 为 16 byte，starting address 按 512 byte 对齐；四行 permutation 构成重复 pattern。Figure 29 使用 `1x10x10x64` NHWC tensor，每 channel 2 byte、共 32 channel；每个 colored cell 表示 8 channel。Figure 30、31 分别给出 source layout 与 64-byte-swizzled destination layout。

**96-byte swizzle mode：**每个编号 cell 为 16 byte，采用表中的两行重复 pattern。Figure 32 同时展示 global-memory layout 与 shared-memory swizzled layout，其中 starting address 按 256 byte 对齐。

**128-byte swizzle mode：**支持四种 atomicity sub-mode。

- 16-byte atomicity：swizzle 时每个 16-byte data unit 保持完整。每个编号 cell 为 16 byte，starting address 按 1024 byte 对齐，八行 permutation 构成重复 pattern。Figure 33–35 展示 `1x10x10x64` NHWC tensor 的示例、source layout 和 destination layout；每个 colored cell 表示 8 channel，每 channel 2 byte。
- 32-byte atomicity：每个 32-byte data unit 保持完整，按表中的四行 pattern 重排，并要求 shared-memory address 按 32 byte 对齐。Figure 36 展示 global 与 shared layout。
- 32-byte atomicity with 8-byte flip：总体 pattern 与 32-byte atomicity 相似，但每隔一条 shared-memory line，会在每个 16-byte data 内交换相邻的两个 8-byte sub-element。该 mode 仅在 `cp.async.bulk.tensor` 的 copy direction 为 `.shared::cluster.global` 或 `.shared::cta.global` 时合法，Figure 37 展示具体布局。
- 64-byte atomicity：每个 64-byte data unit 保持完整，按表中两行 pattern 交换，并要求 shared-memory address 按 64 byte 对齐。Figure 38 给出示例。

Table 14、repeating-pattern boundary 表与 base-offset formula 表均完整保留英文原文，不另作中文副表。只有当 dstMem shared-memory address 位于对应 repeating-pattern boundary 时，swizzle base offset 才为 0；否则必须根据最后一张表的公式计算非零 base offset。

##### 重点解读

Swizzling 并不是改变 tensor 的逻辑内容，而是重排 shared-memory address，使并行访问更符合 bank 分布。模式名中的 32B、64B、96B、128B 描述 swizzle 范围，atomicity 则描述重排时必须保持连续、不可拆开的 data unit；两者是不同维度。

实现时需要同时满足三类约束：选定 mode/sub-mode 的合法组合、shared-memory address alignment，以及 repeating pattern 的 base offset。只看“128B swizzle”而忽略 atomicity 和基址位置，会得到错误的 destination address mapping。

#### 5.5.8 Tensor-map

##### English Original

> The tensor-map is a 128-byte opaque object either in `.const` space or `.param` (kernel function parameter) space or `.global` space which describes the tensor properties and the access properties of the tensor data described in previous sections.
>
> Tensor-Map can be created using CUDA APIs. Refer to CUDA programming guide for more details.

##### 中文翻译

Tensor-map 是一个 128-byte opaque object，可以位于 `.const` space、`.param`（kernel function parameter）space 或 `.global` space，用于描述前述 tensor property 与 tensor-data access property。

Tensor-map 可以通过 CUDA API 创建，详细信息参阅 CUDA Programming Guide。

##### 重点解读

Tensor-map 是 tensor instruction 的元数据描述符，而不是 tensor data 本身。它把 dimension、size、stride、element format、Bounding Box、interleave、swizzle 等配置集中到 128-byte object 中；运行时 instruction 再结合 coordinate、offset、`wHalo` 等 operand 完成一次具体访问。

