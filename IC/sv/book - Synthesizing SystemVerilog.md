[【翻译】可综合SystemVerilog教程(1) / Synthesizing SystemVerilog - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/479482290)
这篇介绍了各种面向综合的SystemVerilog特性以，感觉写的很好的所以翻译一遍.可以用于SystemVerilog的入门或参考（针对设计而非验证，并假定读者具备Verilog基础）。如注意到任何错误或错字请指出。

> 原文：
> **Synthesizing SystemVerilog**
> **Bursting the Myth that SystemVerilog is only for Verification**
> *Stuart Sutherland* Sutherland HDL, Inc. stuart@sutherland-hdl.com
> *Don Mills* Microchip Technology, Inc. mills@microchip.chip
> SNUG Silicon Valley **2013**
> 可以在Google等地方搜到。



\1. 摘要和引言 | Abstract and Introduction

\2. 数据类型介绍 | Data types

\3. 参数化模块 | Parameterized models

\4. 共享声明空间：封装与$unit | Shared declaration spaces - packages and $unit

## 摘要

SystemVerilog不仅只能用于验证（Verification）！从SystemVerilog标准设计之初，其主要目标之一就是以更少的代码更精确地构建复杂硬件设计的可综合模型。这一目标确实达到了，Synopsys在Design Complier (DC)和Synplify-Pro中关于SystemVerilog的实现（implementing）都做的很棒。

这篇文章详细阐明了SystemVerilog在ASIC和FPGA设计中可综合的部分，然后介绍了这些构件（constructs）相较传统Verilog的优势。读者能从本文获得新的RTL建模技巧，通过这些技巧确实能减少代码量和潜在的设计错误，达到更高的综合结果质量（Quality of Results, QoR）。

**目标受众：**包括从事RTL设计以及综合的工程师，面向ASIC与FPGA实现。

注：本文信息基于Synopsys Design Complier（也叫HDL Compiler）的2012.06-SP4版本，以及Synopsys Synplify-Pro的2012.09-SP1版本。都是本文撰写时最新发布的版本。

## 1.0 引言 - Verilog vs. SystemVerilog的真相

一个常见的错误观点是，"Verilog"是一个可综合的硬件建模语言，而"SystemVerilog"则是一个不可综合的验证语言。这完全是错的！

Verilog起源于1984，同时用于硬件功能建模与描述硬件测试平台（testbench）Verilog的许多语言构件（language constructs），如if...else语句，既可用于硬件建模，也可用于验证。此外还有大量仅用于验证的构件，如$display语句就在硬件中没有相应的直接表示。综合关心的是语言中硬件建模的部分，因此仅支持原始Verilog语言的一个子集。

1995年，IEEE官方正式地将Verilog语言标准化了，标准号是**1364-1995**，名为**Verilog-1995** [1]。随后IEEE便开始扩展该语言，包括设计和验证方面，并于2001年发布了**1364-2001**标准，通常称作**Verilog-2001** [2]。一年后，IEEE发布了**1364.1-2002 Verilog RTL Synthesis**标准 [3]，其中规定了Verilog-2001中的部分子集应当被视为是可综合的。

IEEE随后也更新了Verilog标准，即1364-2005，又称Verilog-2005 [4]。然而集成电路的功能性，复杂性，以及时钟速度自2000以来发展的是如此之快，以至仅对Verilog标准进行小幅更新不足以跟得上硬件模型与验证测试对语言能力所持续增长的需求。

IEEE所规定用于增强Verilog语言的新特性是如此之多，以至于IEEE为此创建了一个新的标准号：1800-2005，和一个新名称：SystemVerilog [5]，而仅用于描述语言新增扩展内容。**SystemVerilog-2005并非一个独立语言——它只是在Verilog-2005基础之上的一组扩展。**分两个文档的原因之一是为了帮助那些供应Verilog模拟器和综合编译器的公司能专注实现所有新功能。

**然后是混乱的名称变化...** 2009年，IEEE将Verilog 1364-2005和SystemVerilog扩展（1800-2005）合并为了同一文件。出于作者一直不理解的原因，IEEE选择停用了原先的Verilog名称，并将合并后的标准名称改为SystemVerilog。原始的1364 Verilog标准结束了，然后IEEE批准了**1800-2009 SystemVerilog-2009**标准[6] ，作为一个完整的硬件设计与验证语言。在IEEE的术语下，Verilog标准已经不复存在了，现在只有SystemVerilog标准。**自2009年以来，你就再也没用过Verilog... 你一直都是在用SystemVerilog进行设计和综合！**（IEEE随后发布了一个SystemVerilog-2012标准，其中包括了对原始的，现已不复存在的Verilog语言的额外增强）

![img](https://pic1.zhimg.com/80/v2-a8a54da12b5632cb7ac258363a256bb7_1440w.webp?source=d16d100b)

Figure 1. Verilog到SystemVerilog的发展图

值得注意的是，SystemVerilog标准**同时**扩展了Verilog的验证以及硬件建模能力。Figure 1是语言的发展图，虽然并不全面，但也能够说明SystemVerilog对原始Verilog的大量扩展增强了硬件建模能力。本文的重点就是这些构件（constructs）是如何综合的，以及在设计使用中这些SystemVerilog扩展所带来的优势。

本文的目的是提供一个全面的清单，列出所有能用Synopsys **Design Compile**（DC，或HDL Compiler）和/或**Synplify-Pro**综合的东西。重点是SystemVerilog增加的那部分构件，以及用户如何利用这些改进。为完整起见，本文也会提及不同版本Verilog标准中的可综合构件，但不会详细讨论。

应当注意的是，目前还没有正式的SystemVerilog综合标准。为反应SystemVerilog中新增的许多可综合扩展，IEEE选择了不更新1364.1 Verilog。作者觉得这是短视的，是对工程社区的损害，希望本文与旧的1364.1-2002 Verilog综合标准结合使用，能作为一个SystemVerilog可综合子集的非官方标准。

## 2 数据类型 Data types

注：在本文中，短语“值集（value sets）”用于指二态值（0和1）与四态值（0, 1, Z, X）。短语“数据类型（data type）”是所有线网类型（net types），变量类型（variable types）和用户自定义类型（user-defined types）的总称。值集和数据类型这两个术语与在IEEE SystemVerilog官方标准[7]中的使用方式并不相同，该标准主要是为实现软件工具（如模拟器，综合编译器）的公司所编写的。SystemVerilog标准采用了诸如“types”，“objects”和“kinds”等术语，这些术语对于工具的实现者们来说有着特殊意义，但作者认为这些术语对于使用SystemVerilog语言的工程师来说既不通俗也不直观。

### 2.1 值集 Value sets

原始Verilog只有四态值，即向量（vector）的每个位（bit）可以是逻辑0, 1, Z或X。SystemVerilog新增了二态值的表示，即向量中的每位只能是0或1。SystemVerilog在Verilog语言中新增了关键词**bit**和**logic**，分别代表二态值集和四态值集。

SystemVerilog的线网类型（net）仅使用四态值集**logic**，例如**wire**。变量（variables）则有的使用四态值集**logic**，有的使用二态值集**bit**。（对于那些模拟器和综合编译器的实现者来说，bit和logic关键词还有着更多内容，但对于理解如何通过SystemVerilog进行模型设计来说，已经足够了）

关键词bit和logic也可以在不明确定义是net还是variable的情况下使用，这种情况下是net还是variable是由上下文推断出来的。bit总是推断出variable，logic在大多情况下会推断出variable，但诺与模块输入/输入端口声明一起使用，则推断出net。下面这些声明展示了这些推断规则：

![img](https://pica.zhimg.com/80/v2-51d4373218934ba7f82ea2eb81452e96_1440w.webp?source=d16d100b)

***重要提示：\***综合会以同等方式处理bit和logic，二态值集和四态值集都是用于模拟（simulation）的，在综合中无意义。

- **SystemVerilog优点一**：你不必再纠结模块端口该声明为wire还是reg了（或更具体地，net还是variable）。有了SystemVerilog，你可以将所有模块端口和本地信号都声明为logic，语言会为你正确地推断出是net还是variable（可能偶有例外，工程师有时也可能希望明确地使用与推断结果不同的类型，但这种例外很少）。

请注意验证代码（verification code）会有些不同。在testbench中，随机生成的测试值应被声明为bit（二态）而不是logic（四态）。关于在设计与验证代码中二态和四态类型的详细讨论，见Sutherland [20]。

### 2.2 线网类型 Net types

可综合的线网类型为：

1. **wire**和**tri**：[[1\]](#ref_1)允许并支持多驱的互连线网。
2. **supply0**和**supply1**：[[2\]](#ref_2)分别为带有常量0和1的互连线网。
3. **wand**, **triand**, **wor**, **trior**：[[3\]](#ref_3)将多个驱动相与（AND）或者相或（OR）的互连线网。

这里不详细讨论这些线网类型的综合，因为它们一直都是Verilog的一部分，要了解有关这些传统Verilog类型点信息，请参考1364.1 Verilog RTL Synthesis standard [3]或综合编译器（synthesis complier）文档。

- **SystemVerilog优点二**：（或至少应当算一个优点）SystemVerilog还有个的线网类型的**uwire**，对设计非常有利，但目前还不支持综合。本文Section 12就讨论了为何**uwire**在设计工作中是一个重要优点。

### 2.3 变量类型 Variable types

变量用于过程代码（procedural code）中，也被称作always块。Verilog/SystemVerilog要求过程赋值的左侧（left-hand side）必须是一个变量类型。在SystemVerilog中，可综合的变量类型有：

1. **reg**：[[4\]](#ref_4)一个具有用户自定义向量长度的通用四态变量。
2. **integer**：[[5\]](#ref_5)一个32bit的四态变量。
3. **logic**：[[6\]](#ref_6)除模块输入/输出端口处外，将推断出一个具有用户自定义向量大小的通用四态变量。
4. **bit**：[[7\]](#ref_7)推断出一个具有用户自定义向量长度的通用二态变量。
5. **byte**, **shortint**, **int**, **longint**：[[8\]](#ref_8)分别为具有8bit, 16bit, 32bit, 64bit向量长度的二态变量。

reg和integer类型也一直都是Verilog的一部分，因此不作讨论。

logic关键词并非是一个真的变量类型，但在几乎所有情况下，logic都会推断出一个reg变量。因此logic可用于代替reg，让语言来推断出一个变量。

bit, byte, shortint, int和longint类型都只存储二态的值。综合会将它们视作具相同向量长度（vector size）的四态reg变量。**注意：**这里模拟和综合实现之间会存在功能不匹配的风险，因为综合并不会维持二态行为。一个潜在区别是，二态变量在开始模拟时每个位的值都是0，而综合实现时每个位可能是0也可能是1。

- ***建议：\***在声明中尽可能使用logic，让语言根据上下文来推断是线网还是变量类型。在RTL设计中避免使用任何二态类型，这些类型可能会隐藏设计问题（见Sutherland [20]），可以导致仿真与综合的不匹配。一个例外是，在for-loop中用int来作为迭代器变量。

### 2.4 向量声明（压缩数组） Vector declarations (packed arrays)

可以通过在方括号中指定一个比特范围（range of bits）来声明向量（vector），然后跟上向量的名称。范围声明形式为 [最高有效比特位:最低有效比特位]。

![img](https://pica.zhimg.com/80/v2-c3577f0bf4be539940cdfa8e9fab05f5_1440w.webp?source=d16d100b)

向量的声明，位选（bit select）以及部分选择（多位选择）一直是Verilog的一部分，并且可综合。Verilog-2001标准新增了对变量的部分选择（part select），同样也可综合。

SystemVerilog标准会将向量推断为压缩数组（packed array），以表示向量代表一个连续存储的bit数组。SystemVerilog还新增的一个重要功能：允许使用多个范围将向量划分为多个分区（subfields）。例如：

![img](https://pica.zhimg.com/80/v2-75ea41238687b56f774598557a992718_1440w.webp?source=d16d100b)

多维压缩数组，以及多位压缩数组的选择（selections within multidimensional packed arrays）是可综合的。当设计需要经常引用向量的某些分区时，这一特性会很有用。例如在上面的例子中，这一特性就使得我们能够更加容易地从32bit向量中选择字节（byte）。

### 2.5 数组（非压缩数组） Array (unpacked arrays)

SystemVerilog允许声明线网，变量和用户自定义类型（见Section 2.6）的一或多维数组。数组维数在数组明后面声明。下面是两个例子：

![img](https://picx.zhimg.com/80/v2-2b02844ec1ef556d0086a3de4f78d20b_1440w.webp?source=d16d100b)

通过索引号（index number）来选定数组中的某个元素：

![img](https://pica.zhimg.com/80/v2-7ed23a84fb3a42969a5e17135e84a2f7_1440w.webp?source=d16d100b)

Verilog数组和数组的选定都是可综合的。

SystemVerilog对Verilog数组进行了多个方面的扩展，其中一些对复杂设计建模非常重要。这些功能增强将在Section 2.5.1到2.5.6中讨论。

### 2.5.1 C风格的数组声明 C-style array declarations

Verilog数组是以指定数组地址范围的方式来声明的，语法是 [首地址 : 尾地址] ，例如 [0:255]。SystemVerilog也允许通过指定数组大小的方式来声明数组，方法同C语言。比如：

![img](https://picx.zhimg.com/80/v2-525b4e4559794ff30146486fb7501834_1440w.webp?source=d16d100b)

这种语法所声明数组总是从地址0开始寻址的，到最大地址减一结束。这个便利的小改进是可综合的。

### 2.5.2 数组复制 Copying arrays

Verilog只允许访问一个数组中的单个元素。要将数据从一个数组复制到另一个数组就需要写循环，索引每个数组元素。SystemVerilog允许通过单个赋值语句来复制数组。既可以复制整个数组，也可以复制数组的一部分。比如：

![img](https://pic1.zhimg.com/80/v2-e6c154328c7f81f06400ea4457957ce9_1440w.webp?source=d16d100b)

数组复制赋值（assign）要求赋值两边的数组维度（都是n维数组）和每个维度上的元素数都是相同的[[9\]](#ref_9)。元素本身的位数和类型也必须分别是相同和兼容的。数组复制赋值是可综合的，能大大降低将数据块从一个阵列移到另一个阵列时设计代码的复杂度。

### 2.5.3 阵列的数值列表赋值 Assigning value lists to arrays

可以用数值列表来为数组的全或多个元素复制，数值列表用 `{ } 括起。该列表可包含单个或全部数组元素的默认值。

![img](https://picx.zhimg.com/80/v2-f3f3b7d95bb70af0f3a803ed6031b86b_1440w.webp?source=d16d100b)

数组列表是可综合的。数组列表中数值的个数必须与数组维数相等。每个元素的位数也必须与列表中数值的相同。

### 2.5.4 通过模块端口以及任务或者函数传递数组 Passing arrays through module ports and to tasks and functions

对数组多个元素进行赋值（assign）的能力也使得，将数组作为模块端口或者任务/函数参数成为了可能。下例定义了一个用户自定义类型，表示一个由32位元素组成的8*256二维数组，然后将该数组传入和传出了一个函数，并通过了模块端口。Section 2.6会更详细地讨论用户自定义类型，Section 4.1讨论了定义用户定义类型的适当位置。

![img](https://picx.zhimg.com/80/v2-437590e2a30b05bd2599100dc2404463_1440w.webp?source=d16d100b)

注意，SystemVerilog要求通过数组端口或任务/函数参数传递的值具有相同维数，且每个元素都有相同的向量长度和兼容类型。

### 2.5.5 数组查询系统函数 Array query system functions

SystemVerilog提供了一些特殊的系统函数，使得能够在没有对数组大小进行硬编码的情况下，更加容易地处理数组。可综合的数组查询还数有：$left(), $right(), $low(), $high(), $increment(), $size(), $dimensions(), $unpacked_dimentions()。下面是其中一些函数的使用例子：

![img](https://pic1.zhimg.com/80/v2-904efc61a18deb7ace0e7f70de3d2d8a_1440w.webp?source=d16d100b)

***注意：\***这个例子可通过foreach循环得到大大简化。不幸的是DC和Synplify-Pro并不支持foreach。关于foreach的更多细节，请见Section 12.2。

### 2.5.6 非综合数组的增强

SystemVerilog还对Verilog数组的其它几个方面进行了扩展，这些扩展是不可综合的。这些扩展包括foreach数组迭代循环，数组操作函数，数组定位器函数和数组位流转换。

### 2.6 用户自定义类型 User-defined types

原始Verilog只有内置的数据类型。SystemVerilog允许设计和验证工程师创建新的，用户自定义的数据类型。变量（variable）和线网（net）都可被声明位用户定义的类型。如果没有指定var或net类型关键词，那么用户自定义类型就会被认为是变量。可综合的用户自定义类型有：

1. enum：由一个枚举列表指定了合法值的变量或线网，见Section 2.6.1。
2. struct：一个由多个线网或变量组成的结构，见Section 2.6.2。
3. union：一个能在不同时候以不同类型表示的变量，见Section 2.6.3。
4. **typdef**：类型的定义，见Section 2.6.4。

### 2.6.1 枚举类型 Enumerated types

枚举类型允许用特定一组被命名的值来定义变量和网表。本文仅介绍枚举类型的可综合部分。这是声明枚举类型的基本语法：

![img](https://picx.zhimg.com/80/v2-a07da6f78a91f49c784ef23971044888_1440w.webp?source=d16d100b)

枚举类型有一个基本数据类型，默认是int（二态的32bit类型）。在上例中，State是一个int类型，WAITE，LOAD和DONE都会是32bit的int值。枚举列表中的每个标签都是具有相应逻辑值的常量。默认情况下列表中第一个标签的逻辑值为0，随后每下一个标签都递增1。因此在上例中，WAITE是0，LOAD是1，DONE是2。

设计者也可以显式制定基本类型，以更具体地建模硬件。设计者可为枚举列表中的任何或所有标签都显式指定明确的值。例如：

![img](https://picx.zhimg.com/80/v2-2e3458563f2a72b0596c090c427b516d_1440w.webp?source=d16d100b)

枚举类型相较内置的变量和线网有着更加严格的检查规则。这些规则包括：

1. 枚举列表中每个标签的值必须是唯一的（unique）。

2. 变量和标签值必须有一样的长度（size）。

3. 被枚举的变量只能被赋值（assign）为：

4. 1. 来自它的枚举列表中的一个标签。
   2. 来自同一枚举定义中另一个枚举类型的值。

与传统Verilog相比，枚举类型更严格的规则提供了显著的优势。下面两个例子分别是用Verilog和SystemVerilog建模的同一个简单状态机。两个模型都有几处代码错误，会在注释中指出。

![img](https://picx.zhimg.com/80/v2-6c6b3db5172a364abbe04ae9f81b60e3_1440w.webp?source=d16d100b)

上例中的6个bug在语法上都是合法的，模拟器能够编译并运行。但愿验证代码可以捉到这些功能性bug。综合器可能会给一些编码错误提出警告（warning），但其中一些bug还是会最终出现在设计的门级实现中。

下面的例子展示了相同的编码错误，但用的是枚举类型而不是Verilog的parameter和reg变量（这个例子还用了些后面会介绍的其它SystemVerilog构件）。其中的评注显示，在使用SystemVerilog时，传统Verilog中的每个功能性错误都变为了语法错误：编译器会捕捉这些错误，而不是要到后面才检测功能错误，调试问题再修复错误，然后再重新验证功能。

![img](https://picx.zhimg.com/80/v2-a15a4a9681cd103c58d3a316cff475df_1440w.webp?source=d16d100b)

注：DC没能捉到本例中的第二个语法错误，但VCS检测到了它。

SystemVerilog还提供了枚举类型的几种方法（method）。可综合的方法有：**.first**, **.last**, **.next**, **.prev**和**.num**，顺序基于枚举列表中的声明顺序。上例中轮转下个状态的解码器可通过枚举方法更简洁地写出，如下：

![img](https://picx.zhimg.com/80/v2-768bfa03ac7172f359c2411877790143_1440w.webp?source=d16d100b)

尽管某些情况下枚举方法能够简化代码，实际设计中它们的应用却有所局限。作者觉得，使用赋值枚举标签（assign）而使用枚举方法，是种更好的编码风格。标签能让代码更加self-documenting，并在状态机分支里提供更多灵活性（flexibility）。

***注：***直至撰写本文时，Synplify-Pro还不支持枚举方法。

- **SystemVerilog优点三**：枚举类型能预防难以检测和调试的编码错误！当一个变量或线网仅能使用限定的一组合法值时，请使用枚举类型。

### 2.6.2 结构体 Structures

SystemVerilog结构体提供了一种机制，可将多个变量集中于同一个共有名称下。只要结构体中所用的变量类型是可综合的，结构体就是可综合的。

![img](https://pic1.zhimg.com/80/v2-86f6d43f033f8c3b6ce01d591f5a7303_1440w.webp?source=d16d100b)

可通过点操作符（.）来访问结构体中的单独成员。

![img](https://pic1.zhimg.com/80/v2-398f1560df6b09148b61ffddfaa3bb5d_1440w.webp?source=d16d100b)

更有用的是，结构体还可作为一个整体来读取或者写入。可以将整个结构体复制给另一个结构体，只要这两个结构体来自同一个定义。这要用到**typedef**，会在Section 2.6.4中展示。通过**typedef**定义结构体类型也使得，可以通过模块端口或任务/函数参数传递整个结构体。

也可以用数值列表来给一个结构体中的所有成员赋值（assign），用 ’{} 括起来。列表可以还有单个结构体成员的值，或是一或多个成员的默认值。

![img](https://picx.zhimg.com/80/v2-ad356f333dbdd7c24ee0c338d568c23c_1440w.webp?source=d16d100b)

默认情况下，结构体成员会以软件工具所认为最优的方式存储。大多情况下这也意味着最好的模拟和综合质量结果。设计者可通过将结构体声明为压缩式（packed）来管理结构成员的存储方式。压缩结构体会以连续的方式存储所有成员，其中第一个成员存储在最左边（最高有效位most signficant bits）。压缩结构体（packed struct）与压缩共用体（packed union）配合到一起会非常有用（见Section 2.6.3）。

![img](https://picx.zhimg.com/80/v2-7cc32708dd112e0f59de884d20d1e77c_1440w.webp?source=d16d100b)

- **SystemVerilog优点四**：通过结构体将相关的变量集合在一起，可将它们作为一个整体进行赋值（assign）或者传递给其它模块，能够减少代码量并确保一致性。仅当结构体会在共用体（union）中使用时，才使用压缩结构体。

### 2.6.3 共用体 Unions

共用体（union）允许单个存储空间以多种存储格式（format）表示。SystemVerilog有三种类型的共用体：简单共用体（simple union），压缩共用体（packed union），标签共用体（tagged union）。只有压缩共用体是可综合的。

压缩共用体要求共用体内所有表示都是比特数（number of bits）相同的压缩类型。压缩类型包括位向量（bit-vector (packed arrays)），整数类型（integer types）和压缩结构体（packed structures）。由于压缩共用体中所有成员长度（size）都是一样的，因此向其中一个成员（格式）写入数据，然后再从另一个不同成员读回该数据是合法的。

下例是**一个**即可用于存储数据包（data_packet），也可存储指令包（instruction_packet）的64bit硬件寄存器。

![img](https://picx.zhimg.com/80/v2-7fa4ce58f28f2088725bae047b09b5c6_1440w.webp?source=d16d100b)

![img](https://picx.zhimg.com/80/v2-dc74c72ad74ddc8b914eab5847e0b006_1440w.webp?source=d16d100b)

上例中最后三行语句根据条件选择将拼接数据赋值到data_packet结构或者是instruction_packet结构上。这是合法的，功能也是正确的，因为这些结构体成员都是“压缩的（packed）”。

***注：\***将一个值列表（list of values）赋值（assign）给一个结构体在语法上是合法的（见Section 2.1），并且也是推荐的编码风格。然而DC并不支持将值列表赋值给含有共用体成员的结构体（structures containing union members）。

### 2.6.4 类型定义 Type definition (typedef)

typedef使得我们能基于内置类型和其它用户自定义类型来构建新的数据类型，同C语言相似。下面是些简单例子：

![img](https://picx.zhimg.com/80/v2-3a8bff3861a012c4e8076a4f2eb4dd6e_1440w.webp?source=d16d100b)

SystemVerilog同样还提供了一个包（package）构建来对类型定义以及其它定义进行封装。有关面向综合的包的使用细节，请见Section 4.1。

- **SystemVerilog优点五**：可以保证用户自定义类型在整个项目中的声明是一致的，即便是简单的向量声明。采用用户自定义类型也可防止出现意外的长度（size）或类型不匹配。
- **建议：**[[10\]](#ref_10)大方地使用**typedef**。即便是特定长度（size）的简单向量，如地址或数据向量，在整个设计中也应被定义为用户自定义类型。所有的类型声明都应封装于一或多个包（package）中（见Section 4.1）。

## 3 参数化模块 Parameterized models

Verilog可通过参数或者重定义参数来配置或扩展模块。本文不讨论参数化模块，因为它们一直是Verilog的一部分。

SystrmVerilog扩展了Verilog中的参数定义以及参数重定义，从而允许对数据类型进行参数化。例如：

![img](https://picx.zhimg.com/80/v2-ed13753208f23d7306d71850886314c9_1440w.webp?source=d16d100b)

参数化数据类型（Parameterized）是可综合的。注意在SystemVerilog-2009中，模块参数列表 **#(...)** 中的parameter关键词是可选的，但DC仍要求要有parameter关键词。

## 4 共享声明空间：包与$unit Shared declaration spaces and $unit

### 4.1 包 Packages

原始的Verilog语言并没有共享声明空间。每个模块都要包含在模块内所用到的所有声明。这是语言的一个主要局限。如果要在多个模块中使用相同的参数，任务或函数定义，设计者就必须采用笨拙的办法，通常是配合编译器指令‘ifdef和‘include。在SystemVerilog中，由于新增的用户自定义类型，面向对象类（object-oriented class）的定义以及随机约束（randomization constraints），使得共享声明空间的需求非常迫切。

SystemVerilog添加的用户自定义包（user-defined packages）解决了Verilog的不足之处。包能提供一个声明空间，可在任何模块设计以及验证代码中应用。包中可含有的可综合项有：

1. **parameter** 和 **localparam** 常量定义。
2. **const** 变量定义。
3. **typedef** 用户自定义类型。
4. 完全automatic的 **task** 和 **functino** 定义。
5. 对其它package的 **import** 语句。
6.  用于包链（package chaining）的export语句。

下面是一个包（package）的例子：

![img](https://pic1.zhimg.com/80/v2-11fc49df906e26ad867bc9e3c57d67eb_1440w.webp?source=d16d100b)

![img](https://picx.zhimg.com/80/v2-5c2dc79d2a3120fa3b2a3d17009cdd35_1440w.webp?source=d16d100b)

注意已经在包中定义了的parameter不能被重新定义，localparam同理。另外，综合还要求包中定义的task和function被声明为automatic。

### 4.1.1 引用包定义 Referencing package definitions

在设计块（design block（即，一个模块module或是接口interface）），包内的定义有三种可综合的使用方式：

1. 显式包引用
2. 显式import语句
3. 通配import语句

通过 **包名::包项名** 的方式，可对包项（package item）进行**显式引用（Explicit reference）**。例如：

![img](https://picx.zhimg.com/80/v2-a0c9e059704df0d912559fa7dc91adb6_1440w.webp?source=d16d100b)

对包项的显式引用并不能使得该项在该模块内的其它地方可见，因此每次在模块内使用该定义时，都必须显式引用。

通过 import 语句**显式导入（Explicit import）**包项。一次导入后，该项就可在模块中被多次引用。例如：

![img](https://pic1.zhimg.com/80/v2-956510da66c9066a626f5797750c95be_1440w.webp?source=d16d100b)

通配符导入（Wildcard imports）用星号来表示包内所有定义。通配符导入使包中所有项都在模块内可见。例如：

![img](https://pic1.zhimg.com/80/v2-ba499233f2a0e7467c778a4da2f2c782_1440w.webp?source=d16d100b)

- **SystemVerilog优点六**：封包消除了重复的代码，以及在不同设计块中不匹配的风险，避免了维护重复代码的麻烦。
- **建议**：从现在开始使用包（package）机制！包使得任务，函数以及用户自定义类型的定义都能在整个项目被简洁地复用。

### 4.1.2 import语句的位置 Placement of import statements

前两例子中import语句的位置很关键。一个包必须要已被导入（import），才能在模块端口列表中使用它的定义。在SystemVerilog-2005中，import语句只能出现在模块端口列表后面，这太晚了。SystemVerilog-2009的更新允许了将import语句放在模块端口列表前面（并且如果需要的话，也可以放在参数（parameter）列表前）。SystemVerilog-2009已经发布三年了，但Synopsys在实现SystemVerilog标准中这一细微但重要的变化时并不是很迅速。

***注意：***DC支持在端口列表前导入包，但需通过**set hdlin_sverilog_std 2009**来启用该支持。截止本文撰写时，Synplify-Pro还不支持模块端口前的包导入。

### 4.1.3 将一个包导入到另一个包内 Importing a package into another package

包还可以引用另一个包定义，也能从另一包导入定义。将包导入其它包是可综合的。SystemVerilog还支持**包链（package chaining）**，能更简便地使用引用来其它包的项的包。

***注意：\***DC还不支持包链。有关包链的更多细节，请见Section 12.6。

### 4.1.4 包的编译顺序 Package compilation order

SystemVerilog语法规则要求在引用包定义前，必须先将其编译。这意味着在编译包和模块时，将存在文件顺序单依赖关系。也意味着饮用了包的项目无法独立编译；包必须与模块一同编译（除非预编译，如果工具支持增量编译的话）。

### 4.2 $unit

在包机制加入之前，SystemVerilog还提供了种不同的机制来创建多模块共享定义。机制就是一个名为**$unit**的伪全局命名空间（pseudo-global name space）。任何处在有名声明空间（named declaration space）外的声明都会放在$unit包内。在下例中，bool_t的定义就处在两模块外，因此处于$unit声明空间内。

![img](https://pica.zhimg.com/80/v2-27c5a809c5f5e4ae2560e4d41677110c_1440w.webp?source=d16d100b)

[[11\]](#ref_11)$unit可包含的用户自定义类型种类与有名包（named package）是相同的，综合限制也是一样的。

***注意：\***$unit是个危险的共享命名空间（shared name space），充斥着风险（hazards）。简单地说，$unit的使用风险包括：

1. $unit中的定义可能会散落在许多文件中，导致噩梦般的代码维护。
2. 当$unit空间中的定义存在于多个文件中时，这些文件必须以非常特殊的顺序编译，以确保每个定义在引用前都被编译。
3. 每次调用编译起都会开启一个新的$unit空间，该空间并不共享其它$unit空间中的声明。因此同时编译多个文件的编译器（如VCS）将看到单个$unit空间，而独立编译每个文件的编译起（如DC）将看到多个不相连的$unit空间。
4. 在SystemVerilog中，于同一命名空间内多次定义同一名字是非法的。因此如果一个文件在$unit中定义了一个bool_t，而另一个文件也在$unit中定义了一个bool_t，并且这两个文件一起编译的话，就会发生编译错误或中elaboration错误。
5. 有名包（named package）可被导入（import）$unit，但必须注意不要多次导入同一个包。将同一个包多次导入同一名空间是非法的。

- ***建议：应当避免使用$unit！***取而代之地，通过有名包（named package）来共享定义就能避免$unit的所有风险。