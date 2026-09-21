## 6. Instruction Operands

### 6.1 Operand Type Information

#### English Original

> All operands in instructions have a known type from their declarations. Each operand type must be compatible with the type determined by the instruction template and instruction type. There is no automatic conversion between types.
>
> The bit-size type is compatible with every type having the same size. Integer types of a common size are compatible with each other. Operands having type different from but compatible with the instruction type are silently cast to the instruction type.

#### 中文翻译

Instruction 中的所有 operand 都通过其 declaration 具有已知 type。每个 operand type 必须与 instruction template 和 instruction type 所确定的 type 兼容；不同 type 之间不会发生自动转换。

Bit-size type 与具有相同 size 的任意 type 兼容；size 相同的 integer type 彼此兼容。如果 operand type 与 instruction type 不同但兼容，该 operand 会被静默 cast 为 instruction type。

#### 重点解读

“兼容”不等于“自动数值转换”。例如相同位宽的 `.b32`、`.u32`、`.s32` 可以作为兼容的 bit pattern 使用，但需要改变位宽或数值表示时，必须显式使用 `cvt` 等 instruction。静默 cast 只改变 instruction 对这些 bit 的解释方式。

### 6.2 Source Operands

#### English Original

> The source operands are denoted in the instruction descriptions by the names a, b, and c. PTX describes a load-store machine, so operands for ALU instructions must all be in variables declared in the `.reg` register state space. For most operations, the sizes of the operands must be consistent.
>
> The `cvt` (convert) instruction takes a variety of operand types and sizes, as its job is to convert from nearly any data type to any other data type (and size).
>
> The `ld`, `st`, `mov`, and `cvt` instructions copy data from one location to another. Instructions `ld` and `st` move data from/to addressable state spaces to/from registers. The `mov` instruction copies data between registers.
>
> Most instructions have an optional predicate guard that controls conditional execution, and a few instructions have additional predicate source operands. Predicate operands are denoted by the names p, q, r, s.

#### 中文翻译

Instruction description 使用 a、b、c 表示 source operand。PTX 描述的是 load-store machine，因此 ALU instruction 的 operand 必须全部位于声明在 `.reg` register state space 中的 variable；对于大多数 operation，operand size 必须一致。

`cvt`（convert）instruction 可以接受多种 operand type 和 size，因为它的职责正是在几乎任意 data type 和 size 之间转换。

`ld`、`st`、`mov`、`cvt` instruction 用于把数据从一个位置复制到另一个位置。`ld` 与 `st` 在 addressable state space 和 register 之间移动数据，`mov` 在 register 之间复制数据。

大多数 instruction 都有控制 conditional execution 的可选 predicate guard，少数 instruction 还具有额外的 predicate source operand。Predicate operand 以 p、q、r、s 表示。

### 6.3 Destination Operands

#### English Original

> PTX instructions that produce a single result store the result in the field denoted by d (for destination) in the instruction descriptions. The result operand is a scalar or vector variable in the register state space.

#### 中文翻译

产生单一结果的 PTX instruction，会把结果写入 instruction description 中以 d（destination）表示的 field。Result operand 是 register state space 中的 scalar 或 vector variable。

### 6.4 Using Addresses, Arrays, and Vectors

#### English Original

> Using scalar variables as operands is straightforward. The interesting capabilities begin with addresses, arrays, and vectors.

#### 中文翻译

使用 scalar variable 作为 operand 很直接；更丰富的能力体现在 address、array 和 vector 上。

### 6.4.1 Addresses as Operands

#### English Original

> All the memory instructions take an address operand that specifies the memory location being accessed. This addressable operand is one of:
>
> **`[var]`**
>
> the name of an addressable variable `var`.
>
> **`[reg]`**
>
> an integer or bit-size type register `reg` containing a byte address.
>
> **`[reg+immOff]`**
>
> a sum of register `reg` containing a byte address plus a constant integer byte offset (signed, 32-bit).
>
> **`[var+immOff]`**
>
> a sum of address of addressable variable `var` containing a byte address plus a constant integer byte offset (signed, 32-bit).
>
> **`[immAddr]`**
>
> an immediate absolute byte address (unsigned, 32-bit).
>
> **`var[immOff]`**
>
> an array element as described in Arrays as Operands.
>
> The register containing an address may be declared as a bit-size type or integer type.
>
> The access size of a memory instruction is the total number of bytes accessed in memory. For example, the access size of `ld.v4.b32` is 16 bytes, while the access size of `atom.f16x2` is 4 bytes.
>
> The address must be naturally aligned to a multiple of the access size. If an address is not properly aligned, the resulting behavior is undefined. For example, among other things, the access may proceed by silently masking off low-order address bits to achieve proper rounding, or the instruction may fault.
>
> The address size may be either 32-bit or 64-bit. 128-bit addresses are not supported. Addresses are zero-extended to the specified width as needed, and truncated if the register width exceeds the state space address width for the target architecture.
>
> Address arithmetic is performed using integer arithmetic and logical instructions. Examples include pointer arithmetic and pointer comparisons. All addresses and address computations are byte-based; there is no support for C-style pointer arithmetic.
>
> The `mov` instruction can be used to move the address of a variable into a pointer. The address is an offset in the state space in which the variable is declared. The state space specified by memory instructions must match the state space of the address operand, otherwise the behavior is undefined. Load and store operations move data between registers and locations in addressable state spaces. The syntax is similar to that used in many assembly languages, where scalar variables are simply named and addresses are de-referenced by enclosing the address expression in square brackets. Address expressions include variable names, address registers, address register plus byte offset, and immediate address expressions which evaluate at compile-time to a constant address.
>
> Here are a few examples:

```ptx
.shared .u16 x;
.reg    .u16 r0;
.global .v4 .f32 V;
.reg    .v4 .f32 W;
.const  .s32 tbl[256];
.reg    .b32 p;
.reg    .s32 q;

ld.shared.u16   r0,[x];
ld.global.v4.f32 W, [V];
ld.const.s32    q, [tbl+12];
mov.u32         p, tbl;
```

#### 中文翻译

所有 memory instruction 都接受一个 address operand，用于指定被访问的 memory location。可用形式如下：

- `[var]`：addressable variable `var` 的名称。
- `[reg]`：保存 byte address 的 integer-type 或 bit-size-type register `reg`。
- `[reg+immOff]`：保存 byte address 的 register `reg` 加一个 signed 32-bit constant integer byte offset。
- `[var+immOff]`：addressable variable `var` 的 byte address 加一个 signed 32-bit constant integer byte offset。
- `[immAddr]`：unsigned 32-bit immediate absolute byte address。
- `var[immOff]`：下一节所述的 array element。

保存 address 的 register 可以声明为 bit-size type 或 integer type。

Memory instruction 的 access size 是一次在 memory 中访问的总 byte 数。例如，`ld.v4.b32` 的 access size 是 16 byte，`atom.f16x2` 的 access size 是 4 byte。

Address 必须自然对齐到 access size 的整数倍。若 address 未正确对齐，行为未定义；访问可能静默清除 address 的低位以完成对齐，也可能触发 instruction fault，实际行为不限于这两种情况。

Address size 可以是 32 bit 或 64 bit，不支持 128-bit address。必要时 address 会 zero-extend 到指定宽度；如果 register width 超过 target architecture 中相应 state-space address width，则会被截断。

Address arithmetic 使用 integer arithmetic 和 logical instruction 完成，包括 pointer arithmetic 与 pointer comparison。所有 address 和 address computation 都以 byte 为单位，不支持 C-style pointer arithmetic。

`mov` 可以把 variable address 写入 pointer。该 address 是 variable 所在 state space 内的 offset。Memory instruction 指定的 state space 必须与 address operand 的 state space 匹配，否则行为未定义。Load/store operation 在 register 与 addressable state-space location 之间移动数据。语法与常见 assembly language 类似：scalar variable 直接以名称表示，address expression 用方括号解引用；expression 可以是 variable name、address register、address register 加 byte offset，或在 compile time 求值为 constant address 的 immediate expression。

示例依次展示 shared scalar load、global vector load、带 constant byte offset 的 constant-memory load，以及把 array base address 移入 register。

#### 重点解读

PTX address arithmetic 始终以 byte 为单位。`tbl+12` 的含义是 base address 加 12 byte，而不是自动根据 `.s32` element size 移动 12 个 element。与 C pointer arithmetic 相同的缩放必须由编译器或程序显式计算。

Alignment 依据整个 instruction 的 access size，而不是单个 element type。例如 `ld.global.v4.f32` 即使每个 component 只有 4 byte，address 仍需满足 16-byte natural alignment。

#### 6.4.1.1 Generic Addressing

##### English Original

> If a memory instruction does not specify a state space, the operation is performed using generic addressing. The state spaces `.const`, Kernel Function Parameters (`.param`), `.local` and `.shared` are modeled as windows within the generic address space. Each window is defined by a window base and a window size that is equal to the size of the corresponding state space. A generic address maps to global memory unless it falls within the window for const, local, or shared memory. The Kernel Function Parameters (`.param`) window is contained within the `.global` window. Within each window, a generic address maps to an address in the underlying state space by subtracting the window base from the generic address.

##### 中文翻译

如果 memory instruction 未指定 state space，operation 就使用 generic addressing。`.const`、Kernel Function Parameters（`.param`）、`.local` 和 `.shared` state space 被建模为 generic address space 中的 window。每个 window 由 window base 和等于相应 state-space size 的 window size 定义。

除非 generic address 落在 const、local 或 shared memory window 内，否则它映射到 global memory。Kernel Function Parameters（`.param`）window 包含在 `.global` window 内。在每个 window 内，用 generic address 减去 window base，即可得到 underlying state space 中的 address。

##### 重点解读

Generic address space 是一层统一编码，不代表所有 state space 具有相同物理存储。硬件或编译链先判断 address 落在哪个 window，再减去对应 base 得到该 state space 内的 offset；未命中特殊 window 时按 global memory 处理。

### 6.4.2 Arrays as Operands

#### English Original

> Arrays of all types can be declared, and the identifier becomes an address constant in the space where the array is declared. The size of the array is a constant in the program.
>
> Array elements can be accessed using an explicitly calculated byte address, or by indexing into the array using square-bracket notation. The expression within square brackets is either a constant integer, a register variable, or a simple register with constant offset expression, where the offset is a constant expression that is either added or subtracted from a register variable. If more complicated indexing is desired, it must be written as an address calculation prior to use. Examples are:

```ptx
ld.global.u32  s, a[0];
ld.global.u32  s, a[N-1];
mov.u32        s, a[1];  // move address of a[1] into s
```

#### 中文翻译

所有 type 都可以声明 array，其 identifier 会成为该 array 所在 state space 中的 address constant；array size 是 program constant。

Array element 可以通过显式计算的 byte address 访问，也可以使用方括号索引。方括号内的 expression 可以是 constant integer、register variable，或 register 加减 constant offset 的简单 expression；若需要更复杂的 indexing，必须先单独写成 address calculation，再用于访问。

示例访问 `a[0]`、`a[N-1]`，并把 `a[1]` 的 address 移入 `s`。

### 6.4.3 Vectors as Operands

#### English Original

> Vector operands can be specified as source and destination operands for instructions. However, when specified as destination operand, all elements in vector expression must be unique, otherwise behavior is undefined. Vectors may also be passed as arguments to called functions.
>
> Vector elements can be extracted from the vector with the suffixes `.x`, `.y`, `.z` and `.w`, as well as the typical color fields `.r`, `.g`, `.b` and `.a`.
>
> A brace-enclosed list is used for pattern matching to pull apart vectors.

```ptx
.reg .v4 .f32 V;
.reg .f32     a, b, c, d;

mov.v4.f32 {a,b,c,d}, V;
```

> Vector loads and stores can be used to implement wide loads and stores, which may improve memory performance. The registers in the load/store operations can be a vector, or a brace-enclosed list of similarly typed scalars. Here are examples:

```ptx
ld.global.v4.f32  {a,b,c,d}, [addr+16];
ld.global.v2.u32  V2, [addr+8];
```

> Elements in a brace-enclosed vector, say `{Ra, Rb, Rc, Rd}`, correspond to extracted elements as follows:

```text
Ra = V.x = V.r
Rb = V.y = V.g
Rc = V.z = V.b
Rd = V.w = V.a
```

#### 中文翻译

Vector operand 可以作为 instruction 的 source operand 或 destination operand。但当它用作 destination 时，vector expression 中的所有 element 必须互不重复，否则行为未定义。Vector 也可以作为 argument 传给被调用的 function。

可以用 `.x`、`.y`、`.z`、`.w` suffix 提取 vector element，也可以使用对应的 color field `.r`、`.g`、`.b`、`.a`。花括号包围的 list 用于 pattern matching，将 vector 拆分为各 component。

Vector load/store 可实现 wide load/store，从而可能改善 memory performance。Load/store 中的 register operand 可以是一个 vector，也可以是由相同 type scalar 组成的 brace-enclosed list。对于 `{Ra, Rb, Rc, Rd}`，其 element 依次对应 `V.x/V.r`、`V.y/V.g`、`V.z/V.b`、`V.w/V.a`。

#### 重点解读

Destination vector 要求 element 唯一，是因为一次 instruction 的多个输出 component 不能写回同一个 register。Vector load/store 的性能价值来自合并成更宽的 memory access，但仍必须满足整次 access 的 alignment 要求。

### 6.4.4 Labels and Function Names as Operands

#### English Original

> Labels and function names can be used only in `bra`/`brx.idx` and `call` instructions respectively. Function names can be used in `mov` instruction to get the address of the function into a register, for use in an indirect call.
>
> Beginning in PTX ISA version 3.1, the `mov` instruction may be used to take the address of kernel functions, to be passed to a system call that initiates a kernel launch from the GPU. This feature is part of the support for CUDA Dynamic Parallelism. See the CUDA Dynamic Parallelism Programming Guide for details.

#### 中文翻译

Label 只能用于 `bra`/`brx.idx` instruction，function name 只能用于 `call` instruction。Function name 还可以用于 `mov`，把 function address 写入 register，以供 indirect call 使用。

从 PTX ISA 3.1 开始，`mov` 可以取得 kernel function address，并将其传给从 GPU 发起 kernel launch 的 system call。该功能属于 CUDA Dynamic Parallelism 支持，详细信息参阅 CUDA Dynamic Parallelism Programming Guide。

### 6.5 Type Conversion

#### English Original

> All operands to all arithmetic, logic, and data movement instruction must be of the same type and size, except for operations where changing the size and/or type is part of the definition of the instruction. Operands of different sizes or types must be converted prior to the operation.

#### 中文翻译

所有 arithmetic、logic 和 data-movement instruction 的 operand 都必须具有相同 type 和 size，除非改变 size 和/或 type 本来就是该 instruction 定义的一部分。不同 size 或 type 的 operand 必须在 operation 之前完成转换。

### 6.5.1 Scalar Conversions

#### English Original

> Table 15 and Table 16 show what precision and format the `cvt` instruction uses given operands of differing types. For example, if a `cvt.s32.u16` instruction is given a `u16` source operand and `s32` as a destination operand, the `u16` is zero-extended to `s32`.
>
> Conversions to floating-point that are beyond the range of floating-point numbers are represented with the maximum floating-point value (IEEE 754 Inf for `f32` and `f64`, and ~131,000 for `f16`).
>
> Table 15 Convert Instruction Precision and Format Table 1

| Source Format \ Destination Format | s8 | s16 | s32 | s64 | u8 | u16 | u32 | u64 | f16 | f32 | f64 | bf16 | tf32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s8 | – | sext | sext | sext | – | sext | sext | sext | s2f | s2f | s2f | s2f | – |
| s16 | chop¹ | – | sext | sext | chop¹ | – | sext | sext | s2f | s2f | s2f | s2f | – |
| s32 | chop¹ | chop¹ | – | sext | chop¹ | chop¹ | – | sext | s2f | s2f | s2f | s2f | – |
| s64 | chop¹ | chop¹ | chop¹ | – | chop¹ | chop¹ | chop¹ | – | s2f | s2f | s2f | s2f | – |
| u8 | – | zext | zext | zext | – | zext | zext | zext | u2f | u2f | u2f | u2f | – |
| u16 | chop¹ | – | zext | zext | chop¹ | – | zext | zext | u2f | u2f | u2f | u2f | – |
| u32 | chop¹ | chop¹ | – | zext | chop¹ | chop¹ | – | zext | u2f | u2f | u2f | u2f | – |
| u64 | chop¹ | chop¹ | chop¹ | – | chop¹ | chop¹ | chop¹ | – | u2f | u2f | u2f | u2f | – |
| f16 | f2s | f2s | f2s | f2s | f2u | f2u | f2u | f2u | – | f2f | f2f | f2f | – |
| f32 | f2s | f2s | f2s | f2s | f2u | f2u | f2u | f2u | f2f | – | f2f | f2f | f2f |
| f64 | f2s | f2s | f2s | f2s | f2u | f2u | f2u | f2u | f2f | f2f | – | f2f | – |
| bf16 | f2s | f2s | f2s | f2s | f2u | f2u | f2u | f2u | f2f | f2f | f2f | f2f | – |
| tf32 | – | – | – | – | – | – | – | – | – | – | – | – | – |

> Table 16 Convert Instruction Precision and Format Table 2

| Source Format \ Destination Format | f16 | f32 | bf16 | e4m3 | e5m2 | e2m3 | e3m2 | e2m1 | ue8m0 | ue5m3 | s2f6 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| f16 | – | f2f | f2f | f2f | f2f | f2f | f2f | f2f | – | f2f | – |
| f32 | f2f | – | f2f | f2f | f2f | f2f | f2f | f2f | f2f | f2f | f2f |
| bf16 | f2f | f2f | – | f2f | f2f | f2f | f2f | f2f | f2f | f2f | f2f |
| e4m3 | f2f | – | f2f | – | – | – | – | – | – | – | – |
| e5m2 | f2f | – | f2f | – | – | – | – | – | – | – | – |
| e2m3 | f2f | – | f2f | – | – | – | – | – | – | – | – |
| e3m2 | f2f | – | f2f | – | – | – | – | – | – | – | – |
| e2m1 | f2f | – | f2f | – | – | – | – | – | – | – | – |
| ue8m0 | – | – | f2f | – | – | – | – | – | – | – | – |
| ue5m3 | – | – | f2f | – | – | – | – | – | – | – | – |
| s2f6 | – | – | f2f | – | – | – | – | – | – | – | – |

> Notes
>
> `sext` = sign-extend; `zext` = zero-extend; `chop` = keep only low bits that fit;
>
> `s2f` = signed-to-float; `f2s` = float-to-signed; `u2f` = unsigned-to-float;
>
> `f2u` = float-to-unsigned; `f2f` = float-to-float.
>
> ¹ If the destination register is wider than the destination format, the result is extended to the destination register width after chopping. The type of extension (sign or zero) is based on the destination format. For example, `cvt.s16.u32` targeting a 32-bit register first chops to 16-bit, then sign-extends to 32-bit.

#### 中文翻译

Table 15 和 Table 16 给出 `cvt` 在 source、destination type 不同时采用的 precision 与 format。例如，`cvt.s32.u16` 接受 `u16` source operand 并生成 `s32` destination operand 时，会把 `u16` zero-extend 为 `s32`。

如果转换结果超出 floating-point number 的范围，则以该 floating-point format 的最大值表示：`f32`、`f64` 使用 IEEE 754 Inf，`f16` 使用约 131,000。

两张转换矩阵完整保留英文原表，不另作中文副表。缩写含义如下：`sext` 是 sign extension；`zext` 是 zero extension；`chop` 表示只保留能够容纳的低位；`s2f`、`f2s`、`u2f`、`f2u`、`f2f` 分别表示 signed-to-float、float-to-signed、unsigned-to-float、float-to-unsigned 和 float-to-float。

如果 destination register 比 destination format 更宽，结果会先截断到 destination format，再按 destination format 的 signedness 扩展到 register width。例如，以 32-bit register 为 destination 的 `cvt.s16.u32`，先截断为 16 bit，再 sign-extend 到 32 bit。

#### 重点解读

`cvt` 的结果由两层宽度共同决定：instruction 中写出的 destination format 决定转换和舍入语义，实际 destination register width 决定最终承载宽度。窄化转换不是直接“写入低位后结束”，而是先 `chop` 到 format width，再按 destination format 做 sign/zero extension。

矩阵中的 `–` 表示该 source/destination 组合不由相应 `cvt` 形式支持，不能把它理解成 bit pattern 原样复制；纯位复制应根据具体类型和 instruction 规则选择 `mov` 或其他合法操作。

### 6.5.2 Rounding Modifiers

#### English Original

> Conversion instructions may specify a rounding modifier. In PTX, there are four integer rounding modifiers and six floating-point rounding modifiers. Table 17 and Table 18 summarize the rounding modifiers.
>
> Table 17 Floating-Point Rounding Modifiers

| Modifier | Description |
|---|---|
| `.rn` | rounds to nearest even |
| `.rna` | rounds to nearest, ties away from zero |
| `.rz` | rounds towards zero |
| `.rm` | rounds towards negative infinity |
| `.rp` | rounds towards positive infinity |
| `.rs` | rounds either towards zero or away from zero based on the carry out of the integer addition of random bits and the discarded bits of mantissa |

> Table 18 Integer Rounding Modifiers

| Modifier | Description |
|---|---|
| `.rni` | round to nearest integer, choosing even integer if source is equidistant between two integers. |
| `.rzi` | round to nearest integer in the direction of zero |
| `.rmi` | round to nearest integer in direction of negative infinity |
| `.rpi` | round to nearest integer in direction of positive infinity |

#### 中文翻译

Conversion instruction 可以指定 rounding modifier。PTX 提供四种 integer rounding modifier 和六种 floating-point rounding modifier，完整定义见 Table 17、Table 18；表格保留英文原文，不另作中文副表。

Floating-point rounding 包括 round-to-nearest-even（`.rn`）、nearest 且 tie away from zero（`.rna`）、toward zero（`.rz`）、toward negative infinity（`.rm`）、toward positive infinity（`.rp`），以及基于 random bit 与被丢弃 mantissa bit 的 integer addition carry，在 toward zero 与 away from zero 之间决定的 `.rs`。

Integer rounding 包括 nearest 且 tie 取偶数（`.rni`）、toward zero（`.rzi`）、toward negative infinity（`.rmi`）和 toward positive infinity（`.rpi`）。

#### 重点解读

Floating-point modifier 与 integer modifier 的名称相似但不能互换；带 `i` 的形式描述转成 integer 时如何选择整数结果。`.rs` 是 stochastic rounding，其方向由随机位与被丢弃 mantissa 位共同决定，适合降低大量低精度运算中系统性舍入偏差，但结果不具备普通确定性 rounding mode 的逐次固定性。

### 6.6 Operand Costs

#### English Original

> Operands from different state spaces affect the speed of an operation. Registers are fastest, while global memory is slowest. Much of the delay to memory can be hidden in a number of ways. The first is to have multiple threads of execution so that the hardware can issue a memory operation and then switch to other execution. Another way to hide latency is to issue the load instructions as early as possible, as execution is not blocked until the desired result is used in a subsequent (in time) instruction. The register in a store operation is available much more quickly. Table 19 gives estimates of the costs of using different kinds of memory.
>
> Table 19 Cost Estimates for Accessing State-Spaces

| Space | Time | Notes |
|---|---:|---|
| Register | 0 |  |
| Shared | 0 |  |
| Constant | 0 | Amortized cost is low, first access is high |
| Local | > 100 clocks |  |
| Parameter | 0 |  |
| Immediate | 0 |  |
| Global | > 100 clocks |  |
| Texture | > 100 clocks |  |
| Surface | > 100 clocks |  |

#### 中文翻译

来自不同 state space 的 operand 会影响 operation speed。Register 最快，global memory 最慢。Memory delay 可以通过多种方式隐藏：其一是保持多个 execution thread，使 hardware 发出 memory operation 后切换去执行其他工作；其二是尽可能提前发出 load，因为只有后续 instruction 真正使用所需结果时，execution 才会因依赖而阻塞。Store operation 中的 source register 会更快恢复可用。

Table 19 给出访问不同 memory 类别的估算成本，保留英文原表，不另作中文副表。

#### 重点解读

表中的 `0` 应理解为抽象成本或可忽略的估算值，而不是现实硬件上的绝对零周期。特别是 constant memory 的首次访问成本较高，只是在访问模式合适时摊销成本很低。

“提前发出 load”依赖 instruction-level parallelism；“切换到其他 execution”依赖足够的 active warp/thread 来隐藏 latency。Local memory 虽名为 local，通常仍由高延迟的 device memory backing，因此成本更接近 global memory，而不是 register 或 shared memory。

