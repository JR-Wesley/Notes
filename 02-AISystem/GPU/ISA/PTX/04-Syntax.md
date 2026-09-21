## 4. Syntax

#### English Original

> PTX programs are a collection of text source modules (files). PTX source modules have an assembly-language style syntax with instruction operation codes and operands. Pseudo-operations specify symbol and addressing management. The `ptxas` optimizing backend compiler optimizes and assembles PTX source modules to produce corresponding binary object files.

#### 中文翻译

PTX 程序由一组文本源模块（文件）构成。PTX 源模块采用汇编语言风格的语法，其中包含指令操作码和操作数。伪操作用于规定符号和地址管理。优化后端编译器 `ptxas` 会优化并汇编 PTX 源模块，生成相应的二进制目标文件。

#### 重点解读

PTX 文件虽然外观类似汇编，但其中同时存在真正描述计算行为的 instruction 和管理模块、符号、类型、地址空间与目标架构的 directive。`ptxas` 不是简单逐行编码，而是会继续优化 PTX 并为指定 GPU 架构生成二进制代码。

### 4.1 Source Format

#### English Original

> Source modules are ASCII text. Lines are separated by the newline character (`\n`).
>
> All whitespace characters are equivalent; whitespace is ignored except for its use in separating tokens in the language.
>
> The C preprocessor `cpp` may be used to process PTX source modules. Lines beginning with `#` are preprocessor directives. The following are common preprocessor directives:
>
> `#include`, `#define`, `#if`, `#ifdef`, `#else`, `#endif`, `#line`, `#file`
>
> C: A Reference Manual by Harbison and Steele provides a good description of the C preprocessor.
>
> PTX is case sensitive and uses lowercase for keywords.
>
> Each PTX module must begin with a `.version` directive specifying the PTX language version, followed by a `.target` directive specifying the target architecture assumed. See PTX Module Directives for a more information on these directives.

#### 中文翻译

源模块是 ASCII 文本，使用换行字符 `\n` 分隔各行。

所有空白字符等价。除用于分隔语言中的 token 外，空白会被忽略。

可以使用 C 预处理器 `cpp` 处理 PTX 源模块。以 `#` 开头的行属于预处理器 directive。常见 directive 包括：

`#include`、`#define`、`#if`、`#ifdef`、`#else`、`#endif`、`#line`、`#file`。

Harbison 和 Steele 所著的 *C: A Reference Manual* 对 C 预处理器有很好的说明。

PTX 区分大小写，并且关键字使用小写形式。

每个 PTX 模块都必须首先使用 `.version` directive 指定 PTX 语言版本，随后使用 `.target` directive 指定所假定的目标架构。关于这些 directive 的更多信息，可参阅 PTX Module Directives。

#### 重点解读

一个 PTX 模块的最小开头顺序是 `.version` 在前、`.target` 在后。这里的版本与架构承担不同职责：`.version` 声明源代码使用哪一版 PTX 语言，`.target` 声明代码所依赖的目标 GPU 特性。某条指令能否使用，通常需要同时满足语言版本和目标架构要求。

### 4.2 Comments

#### English Original

> Comments in PTX follow C/C++ syntax, using non-nested `/*` and `*/` for comments that may span multiple lines, and using `//` to begin a comment that extends up to the next newline character, which terminates the current line. Comments cannot occur within character constants, string literals, or within other comments.
>
> Comments in PTX are treated as whitespace.

#### 中文翻译

PTX 注释遵循 C/C++ 语法：使用不可嵌套的 `/*` 和 `*/` 表示可以跨越多行的注释，使用 `//` 开始一直延伸到下一个换行符的单行注释。注释不能出现在字符常量、字符串字面量或其他注释内部。

PTX 会把注释视为空白字符。

### 4.3 Statements

#### English Original

> A PTX statement is either a directive or an instruction. Statements begin with an optional label and end with a semicolon.
>
> Examples

```ptx
        .reg     .b32 r1, r2;
        .global  .f32  array[N];

start:  mov.b32   r1, %tid.x;
        shl.b32   r1, r1, 2;          // shift thread id by 2 bits
        ld.global.b32 r2, array[r1];  // thread[tid] gets array[tid]
        add.f32   r2, r2, 0.5;        // add 1/2
```

#### 中文翻译

一条 PTX statement 要么是 directive，要么是 instruction。Statement 可以由一个可选 label 开头，并以分号结束。

上例前两行分别声明 32 位寄存器和全局浮点数组；`start:` 是可选 label。后续指令读取线程标识、计算字节偏移、从 global memory 加载数据并执行浮点加法。

#### 重点解读

最常见的指令结构可以抽象为 `label: @predicate opcode.type destination, sources;`，其中 label 和 predicate 都是可选部分。Directive 与 instruction 都以分号结束，但 label 本身以冒号标记，不是独立的可执行指令。

#### 4.3.1 Directive Statements

##### English Original

> Directive keywords begin with a dot, so no conflict is possible with user-defined identifiers. The directives in PTX are listed in Table 1 and described in State Spaces, Types, and Variables and Directives.

**Table 1. PTX Directives**

|  |  |  |  |
| --- | --- | --- | --- |
| `.address_size` | `.explicitcluster` | `.maxnreg` | `.section` |
| `.alias` | `.extern` | `.maxntid` | `.shared` |
| `.align` | `.file` | `.minnctapersm` | `.sreg` |
| `.branchtargets` | `.func` | `.noreturn` | `.target` |
| `.callprototype` | `.global` | `.param` | `.tex` |
| `.calltargets` | `.loc` | `.pragma` | `.version` |
| `.common` | `.local` | `.reg` | `.visible` |
| `.const` | `.maxclusterrank` | `.reqnctapercluster` | `.weak` |
| `.entry` | `.maxnctapersm` | `.reqntid` |  |

##### 中文翻译

Directive 关键字以点号开头，因此不会与用户定义的 identifier 冲突。PTX directive 完整列于表 1，并在 State Spaces, Types, and Variables 以及 Directives 章节中详细说明。

#### 4.3.2 Instruction Statements

##### English Original

> Instructions are formed from an instruction opcode followed by a comma-separated list of zero or more operands, and terminated with a semicolon. Operands may be register variables, constant expressions, address expressions, or label names. Instructions have an optional guard predicate which controls conditional execution. The guard predicate follows the optional label and precedes the opcode, and is written as `@p`, where `p` is a predicate register. The guard predicate may be optionally negated, written as `@!p`.
>
> The destination operand is first, followed by source operands.
>
> Instruction keywords are listed in Table 2. All instruction keywords are reserved tokens in PTX.

**Table 2. Reserved Instruction Keywords**

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| `abs` | `cvta` | `min` | `shf` | `vabsdiff` |
| `activemask` | `discard` | `mma` | `shfl` | `vabsdiff2` |
| `add` | `div` | `mov` | `shl` | `vabsdiff4` |
| `addc` | `dp2a` | `movmatrix` | `shr` | `vadd` |
| `alloca` | `dp4a` | `mul` | `sin` | `vadd2` |
| `and` | `elect` | `mul24` | `slct` | `vadd4` |
| `applypriority` | `ex2` | `multimem` | `spcompress` | `vavrg2` |
| `atom` | `exit` | `nanosleep` | `spdecompress` | `vavrg4` |
| `bar` | `fence` | `neg` | `sqrt` | `vmad` |
| `barrier` | `fma` | `not` | `st` | `vmax` |
| `bfe` | `fns` | `or` | `stackrestore` | `vmax2` |
| `bfi` | `getctarank` | `pmevent` | `stacksave` | `vmax4` |
| `bfind` | `griddepcontrol` | `popc` | `stmatrix` | `vmin` |
| `bmsk` | `isspacep` | `prefetch` | `sub` | `vmin2` |
| `bra` | `istypep` | `prefetchu` | `subc` | `vmin4` |
| `brev` | `ld` | `prmt` | `suld` | `vote` |
| `brkpt` | `ldmatrix` | `rcp` | `suq` | `vset` |
| `brx` | `ldu` | `red` | `sured` | `vset2` |
| `call` | `lg2` | `redux` | `sust` | `vset4` |
| `clmad` | `lop3` | `rem` | `szext` | `vshl` |
| `clz` | `mad` | `ret` | `tanh` | `vshr` |
| `cnot` | `mad24` | `rsqrt` | `tcgen05` | `vsub` |
| `copysign` | `madc` | `sad` | `tensormap` | `vsub2` |
| `cos` | `mapa` | `selp` | `testp` | `vsub4` |
| `clusterlaunchcontrol` | `match` | `set` | `tex` | `wgmma` |
| `cp` | `max` | `selp` | `tld4` | `wmma` |
| `createpolicy` | `mbarrier` | `setmaxnreg` | `trap` | `xor` |
| `cvt` | `membar` | `setp` | `txq` |  |

> [!note]
> NVIDIA 原表中 `selp` 出现两次，本文按原表保留。

##### 中文翻译

Instruction 由 instruction opcode 和其后以逗号分隔的零个或多个 operand 构成，并以分号结束。Operand 可以是 register variable、constant expression、address expression 或 label name。Instruction 可以带有控制条件执行的可选 guard predicate。Guard predicate 位于可选 label 之后、opcode 之前，写作 `@p`，其中 `p` 是 predicate register；也可以写成 `@!p` 对条件取反。

Destination operand 位于最前，其后是 source operand。

Instruction keyword 完整列于表 2。所有 instruction keyword 都是 PTX reserved token。

##### 重点解读

Predicate 是 PTX 表达条件执行的核心形式。`@p` 表示只让 predicate register `p` 为真时的线程执行该指令，`@!p` 则表示只让 `p` 为假的线程执行。它让同一个 warp 中的线程可以根据各自 predicate 状态成为 active 或 inactive thread。

### 4.4 Identifiers

#### English Original

> User-defined identifiers follow extended C++ rules: they either start with a letter followed by zero or more letters, digits, underscore, or dollar characters; or they start with an underscore, dollar, or percentage character followed by one or more letters, digits, underscore, or dollar characters:

```text
followsym:   [a-zA-Z0-9_$]
identifier:  [a-zA-Z]{followsym}* | {[_$%]{followsym}+
```

> PTX does not specify a maximum length for identifiers and suggests that all implementations support a minimum length of at least 1024 characters.
>
> Many high-level languages such as C and C++ follow similar rules for identifier names, except that the percentage sign is not allowed. PTX allows the percentage sign as the first character of an identifier. The percentage sign can be used to avoid name conflicts, e.g., between user-defined variable names and compiler-generated names.
>
> PTX predefines one constant and a small number of special registers that begin with the percentage sign, listed in Table 3.

**Table 3. Predefined Identifiers**

|  |  |  |  |
| --- | --- | --- | --- |
| `%aggr_smem_size` | `%dynamic_smem_size` | `%lanemask_gt` | `%reserved_smem_offset_begin` |
| `%clock` | `%envreg<32>` | `%lanemask_le` | `%reserved_smem_offset_cap` |
| `%clock64` | `%globaltimer` | `%lanemask_lt` | `%reserved_smem_offset_end` |
| `%cluster_ctaid` | `%globaltimer_hi` | `%nclusterid` | `%smid` |
| `%cluster_ctarank` | `%globaltimer_lo` | `%nctaid` | `%tid` |
| `%cluster_nctaid` | `%gridid` | `%nsmid` | `%total_smem_size` |
| `%cluster_nctarank` | `%is_explicit_cluster` | `%ntid` | `%warpid` |
| `%clusterid` | `%laneid` | `%nwarpid` | `WARP_SZ` |
| `%ctaid` | `%lanemask_eq` | `%pm0, ..., %pm7` |  |
| `%current_graph_exec` | `%lanemask_ge` | `%reserved_smem_offset_<2>` |  |

#### 中文翻译

用户定义的 identifier 遵循扩展 C++ 规则：它可以由字母开头，后跟零个或多个字母、数字、下划线或美元符号；也可以由下划线、美元符号或百分号开头，但其后必须至少有一个字母、数字、下划线或美元符号。

PTX 没有规定 identifier 的最大长度，但建议所有实现至少支持 1024 个字符的 identifier。

C 和 C++ 等许多高级语言采用相似的 identifier 命名规则，但不允许百分号。PTX 允许百分号作为 identifier 的第一个字符，可用它避免名称冲突，例如区分用户定义的变量名和编译器生成的名称。

PTX 预定义了一个常量和少量以百分号开头的 special register，完整列于表 3。

#### 重点解读

以 `%` 开头不自动表示“硬件寄存器”。表 3 中的名称是 PTX 预定义 identifier，其中大部分是只读 special register；用户或编译器也可以利用 `%` 前缀构造普通 identifier 以避免命名冲突。`WARP_SZ` 是预定义常量，因此不带 `%`。

### 4.5 Constants

#### English Original

> PTX supports integer and floating-point constants and constant expressions. These constants may be used in data initialization and as operands to instructions. Type checking rules remain the same for integer, floating-point, and bit-size types. For predicate-type data and instructions, integer constants are allowed and are interpreted as in C, i.e., zero values are `False` and non-zero values are `True`.

#### 中文翻译

PTX 支持整数常量、浮点常量和常量表达式。这些常量可以用于数据初始化，也可以作为指令操作数。整数类型、浮点类型和 bit-size 类型采用相同的类型检查规则。对于 predicate 类型的数据和指令，可以使用整数常量，其解释方式与 C 相同：零值为 `False`，非零值为 `True`。

#### 重点解读

PTX 常量在语法中具有固定的初始类型，再根据使用位置转换为指令或数据声明所要求的大小。理解常量时需要分别确认字面量本身的类型、常量表达式的求值类型，以及它最终在使用位置发生的类型或位宽转换。

#### 4.5.1 Integer Constants

##### English Original

> Integer constants are 64-bits in size and are either signed or unsigned, i.e., every integer constant has type `.s64` or `.u64`. The signed/unsigned nature of an integer constant is needed to correctly evaluate constant expressions containing operations such as division and ordered comparisons, where the behavior of the operation depends on the operand types. When used in an instruction or data initialization, each integer constant is converted to the appropriate size based on the data or instruction type at its use.
>
> Integer literals may be written in decimal, hexadecimal, octal, or binary notation. The syntax follows that of C. Integer literals may be followed immediately by the letter `U` to indicate that the literal is unsigned.

```text
hexadecimal literal:  0[xX]{hexdigit}+U?
octal literal:        0{octal digit}+U?
binary literal:       0[bB]{bit}+U?
decimal literal       {nonzero-digit}{digit}*U?
```

> Integer literals are non-negative and have a type determined by their magnitude and optional type suffix as follows: literals are signed (`.s64`) unless the value cannot be fully represented in `.s64` or the unsigned suffix is specified, in which case the literal is unsigned (`.u64`).
>
> The predefined integer constant `WARP_SZ` specifies the number of threads per warp for the target platform; to date, all target architectures have a `WARP_SZ` value of 32.

##### 中文翻译

整数常量的大小为 64 bit，并且分为有符号或无符号两类，即每个整数常量的类型都是 `.s64` 或 `.u64`。整数常量的 signed/unsigned 属性对于正确计算常量表达式是必要的，因为 division、ordered comparison 等操作的行为依赖于 operand type。当整数常量用于 instruction 或数据初始化时，会根据使用位置的数据类型或指令类型转换为适当大小。

整数字面量可以采用十进制、十六进制、八进制或二进制表示法，其语法遵循 C。字面量后可以紧跟字母 `U`，表示该字面量是无符号数。

对应语法依次为：以 `0x` 或 `0X` 开头的十六进制、以 `0` 开头的八进制、以 `0b` 或 `0B` 开头的二进制，以及不以零开头的十进制；四种形式都可以带可选的 `U` 后缀。

整数字面量本身是非负的，其类型由数值大小和可选类型后缀决定。除非数值无法由 `.s64` 完整表示，或者明确指定了 unsigned 后缀，否则字面量默认为有符号 `.s64`；上述两种情况下则为无符号 `.u64`。

预定义整数常量 `WARP_SZ` 表示目标平台每个 warp 的线程数。截至本文档版本，所有目标架构的 `WARP_SZ` 都为 32。

##### 重点解读

负数不是“带负号的整数字面量”，而是对非负整数字面量应用 unary minus 后形成的常量表达式。因此，`-1` 的类型来自字面量 `1` 和一元负号的类型规则；`-1U` 则仍保留 unsigned 类型。这个区别会影响后续 division、comparison、shift 和类型转换行为。

#### 4.5.2 Floating-Point Constants

##### English Original

> Floating-point constants are represented as 64-bit double-precision values, and all floating-point constant expressions are evaluated using 64-bit double precision arithmetic. The only exception is the 32-bit hex notation for expressing an exact single-precision floating-point value; such values retain their exact 32-bit single-precision value and may not be used in constant expressions. Each 64-bit floating-point constant is converted to the appropriate floating-point size based on the data or instruction type at its use.
>
> Floating-point literals may be written with an optional decimal point and an optional signed exponent. Unlike C and C++, there is no suffix letter to specify size; literals are always represented in 64-bit double-precision format.
>
> PTX includes a second representation of floating-point constants for specifying the exact machine representation using a hexadecimal constant. To specify IEEE 754 double-precision floating point values, the constant begins with `0d` or `0D` followed by 16 hex digits. To specify IEEE 754 single-precision floating point values, the constant begins with `0f` or `0F` followed by 8 hex digits.

```text
0[fF]{hexdigit}{8}      // single-precision floating point
0[dD]{hexdigit}{16}     // double-precision floating point
```

> Example

```ptx
mov.f32  $f3, 0F3f800000;       //  1.0
```

##### 中文翻译

浮点常量表示为 64-bit double-precision value，所有浮点常量表达式也使用 64-bit double-precision arithmetic 求值。唯一例外是用 32-bit hexadecimal notation 表示精确 single-precision floating-point value；这种值保留其精确的 32-bit single-precision 表示，但不能用于常量表达式。每个 64-bit 浮点常量都会根据其使用位置的数据类型或指令类型转换成适当的浮点大小。

浮点字面量可以包含可选小数点和可选的有符号 exponent。与 C/C++ 不同，PTX 不使用 suffix letter 指定大小；普通浮点字面量始终表示为 64-bit double-precision 格式。

PTX 还提供第二种浮点常量表示法，使用十六进制常量精确指定机器表示。IEEE 754 double-precision value 以 `0d` 或 `0D` 开头，后跟 16 个十六进制数字；IEEE 754 single-precision value 以 `0f` 或 `0F` 开头，后跟 8 个十六进制数字。

示例中的 `0F3f800000` 按 single-precision bit pattern 精确表示 `1.0`，并被移动到 32-bit floating-point register `$f3`。

##### 重点解读

十六进制浮点形式描述的是 IEEE 754 原始 bit pattern，而不是把十六进制整数按数值转换成浮点数。`0F3f800000` 中的 `0F` 是 single-precision 常量前缀，后面的 `3f800000` 是 `1.0f` 的 32-bit 编码。这种写法适合精确构造 NaN、无穷大、signed zero 或舍入边界值。

#### 4.5.3 Predicate Constants

##### English Original

> In PTX, integer constants may be used as predicates. For predicate-type data initializers and instruction operands, integer constants are interpreted as in C, i.e., zero values are `False` and non-zero values are `True`.

##### 中文翻译

在 PTX 中，整数常量可以用作 predicate。对于 predicate 类型的数据 initializer 和 instruction operand，整数常量按照 C 的规则解释：零值为 `False`，非零值为 `True`。

#### 4.5.4 Constant Expressions

##### English Original

> In PTX, constant expressions are formed using operators as in C and are evaluated using rules similar to those in C, but simplified by restricting types and sizes, removing most casts, and defining full semantics to eliminate cases where expression evaluation in C is implementation dependent.
>
> Constant expressions are formed from constant literals, unary plus and minus, basic arithmetic operators (addition, subtraction, multiplication, division), comparison operators, the conditional ternary operator (`?:`), and parentheses. Integer constant expressions also allow unary logical negation (`!`), bitwise complement (`~`), remainder (`%`), shift operators (`<<` and `>>`), bit-type operators (`&`, `|`, and `^`), and logical operators (`&&`, `||`).
>
> Constant expressions in PTX do not support casts between integer and floating-point.
>
> Constant expressions are evaluated using the same operator precedence as in C. Table 4 gives operator precedence and associativity. Operator precedence is highest for unary operators and decreases with each line in the chart. Operators on the same line have the same precedence and are evaluated right-to-left for unary operators and left-to-right for binary operators.

**Table 4. Operator Precedence**

| Kind | Operator Symbols | Operator Names | Associates |
| --- | --- | --- | --- |
| Primary | `()` | parenthesis | n/a |
| Unary | `+- ! ~` | plus, minus, negation, complement | right |
| Unary | `(.s64)`, `(.u64)` | casts | right |
| Binary | `*/ %` | multiplication, division, remainder | left |
| Binary | `+-` | addition, subtraction | left |
| Binary | `>> <<` | shifts | left |
| Binary | `< > <= >=` | ordered comparisons | left |
| Binary | `== !=` | equal, not equal | left |
| Binary | `&` | bitwise AND | left |
| Binary | `^` | bitwise XOR | left |
| Binary | `\|` | bitwise OR | left |
| Binary | `&&` | logical AND | left |
| Binary | `\|\|` | logical OR | left |
| Ternary | `?:` | conditional | right |

##### 中文翻译

PTX 常量表达式使用与 C 相同的运算符，并按照类似 C 的规则求值；不过，PTX 通过限制类型和大小、移除大多数 cast，并为相关行为定义完整语义，简化了规则并消除了 C 中依赖具体实现的表达式求值情况。

常量表达式可由 constant literal、unary plus/minus、基本算术运算符（加、减、乘、除）、比较运算符、conditional ternary operator `?:` 和括号组成。整数常量表达式还允许 unary logical negation `!`、bitwise complement `~`、remainder `%`、shift operator `<<` 和 `>>`、bit-type operator `&`、`|`、`^`，以及 logical operator `&&` 和 `||`。

PTX 常量表达式不支持整数与浮点数之间的 cast。

常量表达式使用与 C 相同的运算符优先级。表 4 给出优先级和结合性：unary operator 的优先级最高，表中每向下一行优先级降低；位于同一行的运算符具有相同优先级。Unary operator 从右向左求值，binary operator 从左向右求值。

##### 重点解读

这里的 cast 只允许在整数 `.s64` 与 `.u64` 之间调整 signedness，不允许在整数和浮点之间转换。若需要运行时数值类型转换，应使用相应 PTX instruction，例如 `cvt`，而不是把它写成常量表达式 cast。

#### 4.5.5 Integer Constant Expression Evaluation

##### English Original

> Integer constant expressions are evaluated at compile time according to a set of rules that determine the type (signed `.s64` versus unsigned `.u64`) of each sub-expression. These rules are based on the rules in C, but they’ve been simplified to apply only to 64-bit integers, and behavior is fully defined in all cases (specifically, for remainder and shift operators).
>
> - Literals are signed unless unsigned is needed to prevent overflow, or unless the literal uses a `U` suffix. For example:
>   - `42`, `0x1234`, `0123` are signed.
>   - `0xfabc123400000000`, `42U`, `0x1234U` are unsigned.
> - Unary plus and minus preserve the type of the input operand. For example:
>   - `+123`, `-1`, `-(-42)` are signed.
>   - `-1U`, `-0xfabc123400000000` are unsigned.
> - Unary logical negation (`!`) produces a signed result with value `0` or `1`.
> - Unary bitwise complement (`~`) interprets the source operand as unsigned and produces an unsigned result.
> - Some binary operators require normalization of source operands. This normalization is known as the usual arithmetic conversions and simply converts both operands to unsigned type if either operand is unsigned.
> - Addition, subtraction, multiplication, and division perform the usual arithmetic conversions and produce a result with the same type as the converted operands. That is, the operands and result are unsigned if either source operand is unsigned, and is otherwise signed.
> - Remainder (`%`) interprets the operands as unsigned. Note that this differs from C, which allows a negative divisor but defines the behavior to be implementation dependent.
> - Left and right shift interpret the second operand as unsigned and produce a result with the same type as the first operand. Note that the behavior of right-shift is determined by the type of the first operand: right shift of a signed value is arithmetic and preserves the sign, and right shift of an unsigned value is logical and shifts in a zero bit.
> - AND (`&`), OR (`|`), and XOR (`^`) perform the usual arithmetic conversions and produce a result with the same type as the converted operands.
> - AND_OP (`&&`), OR_OP (`||`), Equal (`==`), and Not_Equal (`!=`) produce a signed result. The result value is `0` or `1`.
> - Ordered comparisons (`<`, `<=`, `>`, `>=`) perform the usual arithmetic conversions on source operands and produce a signed result. The result value is `0` or `1`.
> - Casting of expressions to signed or unsigned is supported using `(.s64)` and `(.u64)` casts.
> - For the conditional operator (`? :`), the first operand must be an integer, and the second and third operands are either both integers or both floating-point. The usual arithmetic conversions are performed on the second and third operands, and the result type is the same as the converted type.

##### 中文翻译

整数常量表达式在编译期求值。求值规则会确定每个子表达式的类型，即 signed `.s64` 或 unsigned `.u64`。这些规则以 C 为基础，但简化为只处理 64-bit integer，并且完整定义所有情况的行为，特别是 remainder 和 shift operator。

- 除非为避免 overflow 必须使用 unsigned，或者字面量带有 `U` suffix，否则 literal 为 signed。例如：
  - `42`、`0x1234`、`0123` 为 signed。
  - `0xfabc123400000000`、`42U`、`0x1234U` 为 unsigned。
- Unary plus 和 unary minus 保留 input operand 的类型。例如：
  - `+123`、`-1`、`-(-42)` 为 signed。
  - `-1U`、`-0xfabc123400000000` 为 unsigned。
- Unary logical negation `!` 产生 signed result，其值为 `0` 或 `1`。
- Unary bitwise complement `~` 把 source operand 解释为 unsigned，并产生 unsigned result。
- 某些 binary operator 要求对 source operand 进行归一化。这称为 usual arithmetic conversions：只要任一 operand 为 unsigned，就把两个 operand 都转换为 unsigned。
- Addition、subtraction、multiplication 和 division 执行 usual arithmetic conversions，并产生与转换后 operand 相同类型的结果。也就是说，只要任一 source operand 为 unsigned，两个 operand 和结果都为 unsigned；否则均为 signed。
- Remainder `%` 把 operand 解释为 unsigned。该行为不同于 C；C 允许负 divisor，但相关行为可能依赖具体实现。
- Left/right shift 把第二个 operand 解释为 unsigned，结果类型与第一个 operand 相同。Right shift 的行为由第一个 operand 的类型决定：signed value 的右移是 arithmetic shift，会保留符号；unsigned value 的右移是 logical shift，会移入零 bit。
- AND `&`、OR `|` 和 XOR `^` 执行 usual arithmetic conversions，并产生与转换后 operand 相同类型的结果。
- AND_OP `&&`、OR_OP `||`、Equal `==` 和 Not_Equal `!=` 产生 signed result，结果值为 `0` 或 `1`。
- Ordered comparison `<`、`<=`、`>`、`>=` 对 source operand 执行 usual arithmetic conversions，并产生 signed result，结果值为 `0` 或 `1`。
- 可以使用 `(.s64)` 和 `(.u64)` cast，把表达式转换为 signed 或 unsigned。
- 对 conditional operator `? :`，第一个 operand 必须是 integer；第二、第三个 operand 必须同时为 integer 或同时为 floating-point。第二、第三个 operand 会执行 usual arithmetic conversions，结果类型与转换后的类型相同。

##### 重点解读

PTX 把所有整数常量表达式限制在 `.s64`/`.u64` 范围内，并明确规定 `%` 与 shift 的语义，避免依赖 host C/C++ 编译器的实现差异。最需要警惕的是 signed/unsigned 混合：只要某个参与 usual arithmetic conversions 的 operand 为 unsigned，另一个 operand 也会转成 unsigned，可能改变 division 和 ordered comparison 的结果。

#### 4.5.6 Summary of Constant Expression Evaluation Rules

##### English Original

> Table 5 contains a summary of the constant expression evaluation rules.

**Table 5. Constant Expression Evaluation Rules**

| Kind | Operator | Operand Types | Operand Interpretation | Result Type |
| --- | --- | --- | --- | --- |
| Primary | `()` | any type | same as source | same as source |
| Primary | constant literal | n/a | n/a | `.u64`, `.s64`, or `.f64` |
| Unary | `+-` | any type | same as source | same as source |
| Unary | `!` | integer | zero or non-zero | `.s64` |
| Unary | `~` | integer | `.u64` | `.u64` |
| Cast | `(.u64)` | integer | `.u64` | `.u64` |
| Cast | `(.s64)` | integer | `.s64` | `.s64` |
| Binary | `+- * /` | `.f64` | `.f64` | `.f64` |
| Binary | `+- * /` | integer | use usual conversions | converted type |
| Binary | `< > <= >=` | `.f64` | `.f64` | `.s64` |
| Binary | `< > <= >=` | integer | use usual conversions | `.s64` |
| Binary | `== !=` | `.f64` | `.f64` | `.s64` |
| Binary | `== !=` | integer | use usual conversions | `.s64` |
| Binary | `%` | integer | `.u64` | `.s64` |
| Binary | `>> <<` | integer | 1st unchanged, 2nd is `.u64` | same as 1st operand |
| Binary | `& \| ^` | integer | `.u64` | `.u64` |
| Binary | `&& \|\|` | integer | zero or non-zero | `.s64` |
| Ternary | `?:` | `int ? .f64 : .f64` | same as sources | `.f64` |
| Ternary | `?:` | `int ? int : int` | use usual conversions | converted type |

##### 中文翻译

表 5 汇总了常量表达式求值规则。表格内容按原文完整保留，不另行翻译。

