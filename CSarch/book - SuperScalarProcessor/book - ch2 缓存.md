---
banner: "[[../../300-以影像之/Clorinde 8205816.jpeg]]"
dateCreated: 2024-11-20
dateModified: 2025-05-25
---
# Ch2 Cache
## 2.1 Cache 的一般设计

在超标量处理器中，有两个部分比较影响性能分别是：**分支预测**和**Cache**。

Cache 存在是因为计算机世界存在如下两个基本现象：

1. **时间相关性**：如果一个数据现在被访问了，那么很可能**之后还会被访问**
2. **空间相关性**：如果一个数据被访问了，那它**周围的数据**也很可能被访问

Cache 是个很广义的概念，例如 DRAM 可以看做是硬盘（disk）或者闪存（flash）的 Cache；L2 Cache 可以看做是 DRAM 的 Cache；L1 Cache 可以看做是 L2 的 Cache。

在当前硅工艺下，DRAM 的访问速度比处理器还要慢**一个甚至几个数量级**。所以催生了和处理器采用相同硅工艺的 L1 Cache L2 Cache，其中 L1 Cache 紧耦合在处理器的流水线中，是影响处理器性能的关键因素。（本文不考虑虚拟存储器）

![[assets/ch2 Cache/Cache.png|SuperScalar/Cache.png]]

现代超标量处理器内核采用哈佛结构，L1 Cache 一般包括两个物理组件：**指令（I-Cache）** 和**数据（D-Cache）**，它们的原理一致，但是 D-Cache 有读和写，I-Cache 一般只有读（后面如果没特殊说明，都是指 D-Cache）。L1 Cache 一般采用 SRAM 来实现，大容量的 SRAM 需要更多的时间来找到特定的地址内容，可能就无法和处理器保持速度相近，所以“快”是 L1 Cache 的主要指标。

L2 Cache 一般是**指令和数据共享**，为了求“全”，保存更多内容，不一定要求和处理器保持一样的速度了，所以一般可以做到以 MB 为单位的大小。在多核处理器上，L2 Cache 有可能是被多核共享的（不过一般都是 L3 Cache 才共享）。但是 L1 Cache 仍然是每个核独享的。

- 超标量处理器对 Cache 的需求
I-Cache 需要每周期读取多条指令，但其延时仍然很大，如有好几个周期，但一般这不会造成性能下降，除非遇到预测跳转的分支指令，这些延迟会对性能造成影响。
D-Cache 需要支持每周期内多条 load/store 指令的访问，即多端口设计。处理器中的其他部件如 IQ，SB，RF，ROB 等本身就很小，多端口设计也不占用太大空间。而 D-Cache 本身容量大，多端口会导致面积延时过大，对性能造成负面影响。L2 Cache 被访问的频率没有 L1 Cache 高，所以并不需要多端口，而延迟也并不特别重要，因为只有发生 L1 Cache miss 时才会访问它。但是 L2 Cache 需要比较高的**命中率**，因为它发生缺失时会去访问物理内存，这个访问时间会很长。

  Cache 主要由两个部分组成：**Tag 部分**和**Data 部分**

- Data 部分保存一片**连续地址**的数据
- Tag 部分保存这片连续数据的**公共地址**
一个 Tag 和它对应的所有数据组成的一行称为 **Cache Line**；而 **Cache Line** 中的数据部分被称为数据块（**Cache Data Block**）；如果一个数据可以存储在 Cache 的多个地方，这些被同一个地址找到的多个 Cache Line 就称为 Cache Set。下面是 Cache 一种可能的结构：
![[assets/ch2 Cache/Cache 结构.png|SuperScalar/Cache 结构.png]]
Cache 有三种主要实现方式：
- **直接映射（direct-mapped）**：物理内存中的一个数在 Cache 中**只有一个地方**容纳它；
- **组相连（set-associative）**：物理内存中的一个数在 Cache 中有**多个地方**可以存放；
- **全相连（fully-associative）**：物理内存中的一个数在 Cache 中**任何地方**都可以存放；
![[assets/ch2 Cache/Cache 组成方式.png|SuperScalar/Cache 组成方式.png]]
可以看出直接映射和全相连是组相连的两个特殊情况。TLB 和 Victim Cache（后面会说这两个是啥）多采用全相连结构，普通的 I-Cache 和 D-Cache 则采用组相连结构。
Cache 只能保存最近使用过的内容，有时候需要找的指令或者数据并不在 Cache 中，这种情况就叫 Cache 缺失（Cache Miss），这种情况的发生频率会直接影响处理器的性能。Cache Miss 的情况概括如下：
- **Compulsory（必然**），因为 Cache 只缓存之前用过的内容，因此第一次访问的内容肯定不会在 Cache 中，这种缺失看上去就是必然的。不过实际可以采用预取（perfetching）来降低这种情况的频率。
- **Capcity（容量限制）**，Cache 容量越大，能容纳的内容就越多发生缺失的情况就会减少。然而考虑到面积和访问速度限制，一般 Cache 不能做的很大。
- **Conflict（访问冲突）**，为了解决多个数据被映射到 Cache 中相同位置的情况，一般会用组相连结构。考虑到面积限制，组相连度不会很高。例如 2-way 的 Cache 访问三个属于同一 Cache Set 的数据。就会频繁发生缺失。这时候会用 Victim Cache 来缓解这个问题。
上面的三个缺失条件称为 3C 定理，一般采用**预取**（perfetching）和 **Victim Cache** 来缓解，不过只是减少缺失频率，不能完全消除缺失。

### 2.1.1 Cache 的组成方式
1. 直接映射
Direct-mapped 是最容易实现的一种，处理器访问存储器的地址被分为三个部分：**Tag，Index，Block Offset**。使用 Index 找到 Cache Line，但是所有 Index 相同的都会找到同一个 Cache Line。因此 Cache Line 中的 Tag 部分就用来和地址进行比较，只有它们相等才表明这个 Cache Line 是想要的哪个。一个 Cache line 中有很多数据，通过 Block Offset 用来找到 Cache Line 中想要的数据，定位到字节。同时在 Cache Line 中还有一个有效位（**Valid**）来标记这个 Cache Line 是否保存着有效数据。只有之前被访问过的存储器地址，它的数据才会在对应的 Cache Line 中，相应的有效位也会置为 1。
![[assets/ch2 Cache/directC.png|SuperScalar/directC.png]]
对于 Index 相同的存储器地址，会寻址到同一个 Cache Line，这就造成了冲突。这种情况是直接相连的一大缺点。*实现简单，不需要替换算法，但效率不高*，所以现代处理器很少使用了。

2. 组相连
一个数据可以放在多个 Cache Line 中，如果一个数据可以放在 n 个位置，则称这个 Cache 是 n 路组相连 Cache（**n-way set-assocaiative Cache**），如下是一个两路组相连 Cache
![[assets/ch2 Cache/setAssC.png|SuperScalar/setAssC.png]]
这种结构依然采用 Index 来选择 Cache Line，此时会得到两个 Cache Line，这就是一个 **Cache Set**。在这两个 Cache Line 中选择想要的哪个就要根据 Tag 比较结果来确定。如果两个的结果都不相等，那就是发生了缺失。
可以看到这种方式需要从多个 Line 中选择一个匹配，*相较直接映射结构会增加部分延迟，所以有时候需要进行流水操作以减少对 cycle time*。但是这样会增加 load 指令的延迟，一定程度影响处理器的效率。但是这种方式的突出优点就是可以*显著减少 Cache 的缺失频率*，因此现代处理器大多使用这种方式。
对于 Tag 和 Data 一般是分开放置，分别称为 Tag SRAM 和 Data SRAM。对于这两个部分可以采用两种访问方式：
- **并行访问**：同时访问 Tag SRAM 和 Data SRAM，如上图所示。在访问 Tag 的时候把对应的 Data 数据也读取出来然后送到一个受 Tag 比较结果控制的多路选择器上，选出对应的 Data Block，然后根据 Block Offset 选出对应的字节（选字节的过程一般称为数据对齐 Data Alignment）
![[assets/ch2 Cache/并行访问Cache.png|SuperScalar/并行访问Cache.png]]
Cache 访问一般是处理器中的关键路径，如果在一个周期内处理完这个访问过程，实际上是比较消耗时间的，所以通常采用流水的方式进行。对于 I-Cache 流水结构并不会有太大影响，仍然可实现每周期读取指令；对于 D-Cache，流水线会增大 load 指令延迟。
![[assets/ch2 Cache/并行访问Cache流水线.png|SuperScalar/并行访问Cache流水线.png]]
Address Calculation 计算出存储器的地址；Disambiguation 对 load/store 指令之间存在的相关性进行检查；Cache Access 直接并行访问 Tag SRAM 和 Data SRAM，并使用 Tag 比较的结果对输出数据进行选择；Result Drive 阶段使用存储器中的 block offset 对 data block 选出需要的数据。通过将 Cache 的访问放到几个流水周期内完成，可以降低处理器的周期时间，从而提升效率。
- **串行访问**：先访问 Tag SRAM，然后根据比较结果再访问 Data SRAM，不需要多路选择器，不需要的那部分 SRAM 可以置使能信号为无效，节省功耗。
![[assets/ch2 Cache/串行Cache.png|SuperScalar/串行Cache.png]]
这种方式串行了 Tag SRAM 和 Data SRAM，延迟会更大，所以还是要采用流水处理方式。*串行访问的流水线降低了访问延迟*，因为此时不需要多路选择器，对降低周期时间有好处，但是*缺点是 Cache 访问增加了一个周期*，增大了 load 指令的延迟，而 load 指令处于相关性的顶端，对处理器执行效率有负面影响。
![[assets/ch2 Cache/访问Cache流水线.png|SuperScalar/访问Cache流水线.png]]
对于 Cache 访问访问，串行和并行是比较难区分好坏的。*并行访问会有较低的时钟频率和较大的功耗，但是访问周期缩短一周期*。考虑在超标量处理器中，我们可以将 Cache 访问的时间通过填充其他指令来掩盖，因此使用串行访问来提升时钟频率，同时不会因为增加了一个访问周期导致性能明显下降；对于顺序执行的处理器来说，无法对指令进行调度，增加一个访问周期可能就会引起性能降低，所以采用并行可能更合适。

1. 全相连
由于数据可以放在任何一个 Cache Line 中，所以不再需要 Index 部分，而是直接将 Tag 进行比较来定位需要的 Cache Line。这相当于直接使用存储器的内容来寻址，从存储器找到匹配的项，也就是内容寻址存储器（Content Address Memory，CAM），实际中的全相连 Cache 都是使用 CAM 来存储 Tag，使用普通的 SRAM 存储数据。全相连有*最大的灵活度，缺失率也最低*，但是由于有大量内容需要比较，所以延迟也是最大的。所以采用这种结构的容量都不会太大例如 TLB。
![[assets/ch1 概述/全相连.png|SuperScalar/全相连.png]]

### 2.1.2 Cache 的写入

在一般的 RISC 处理器中，**I-Cache** 都不会被直接写入内容，即使有自修改（self-modifying）的情况，也是借助 D-Cache 来实现的：将要改写的指令作为数据写入到 D-Cache 中，然后将 D-Cache 的内容写入下一级存储器（例如 L2 Cache，这个存储器一定是指令和数据共享的，这个过程称为 clean），然后将 I-Cache 中的内容置为无效，再次执行就能使用被修改的指令了。

对于 **D-Cache** 它的读写操作并不相同，当执行 store 时，如果只向 D-Cache 中写入数据，并不改变下级存储器的数据，就会导致 D-Cache 和下级存储器对某个地址有不同的数据，这称为不一致（**non-consistent**）。

为了保持一致有两种策略：

- **写通（Write Through）**：当数据写入 D-Cache 的时候，同时也写入下级存储器。由于下级存储访问时间长，而 store 指令出现频率高，可能会导致处理器效率降低。
- **写回（Write Back）**：数据写入 D-Cache 时只对 Cache Line 做标记，不写入下级存储器。只有当这个被标记的 line 被替换才写入下级。这个标记称为脏状态（**dirty**）。可以提升效率，但是会造成 D-Cache 和下级存储器很多数据的不一致，对存储器的管理带来负担。

当上述情况，写数据时发现地址并不在 D-Cache 中，就是写缺失（**Write Miss**）。应对策略也是两种：

- **Non-Write Allocate**：直接将数据写入下级存储器，而不是将数据写入 D-Cache 中。
- **Write Allocate**：首先从下级存储器将缺失地址对应的整个数据块取出来，和将要写入的数据合并，然后将合并数据写入 D-Cache 中。具体要不要写入下级存储就可以选择上面的写通或者写回来决策了。

考虑一下为啥不直接把数据写入空的或者随意一个 Line 中？因为直接从 D-Cache 中找一个 line 来存储这个数据并标记，这个 line 中其他部分与下级不一致，写回时会导致下级的正确数据被篡改。

![[assets/ch1 概述/直接写的问题.png|SuperScalar/直接写的问题.png]]

可以看出，一般 Write Through 和 Non-Write Allocate 一起使用，都是直接将数据更新到下级存储器。

![[assets/ch1 概述/写方法配合1.png|SuperScalar/写方法配合1.png]]

而 Write Back 和 Write Allocate 一起使用。无论是读取还是写入发生缺失，都要从 D-Cache 找一个 line 存放新数据，当替换的 line 是 dirty 状态时，会发生两次对下级存储器的访问：写入 line 数据到下一级，从下级读取缺失地址对应的数据块，写回到刚才找的 Cache line 中。合并数据并标记为 dirty。

![[assets/ch1 概述/写方法配合2.png|SuperScalar/写方法配合2.png]]

**写回配合 Write Allocate 要比 写通配合 Non-Write Allocate 的方式复杂，但是可以减少下级存储器的写入频率，因此可以使处理器有更好的性能。**

### 2.1.3 Cache 的替换策略

当我们需要从 Cache Set 找一个 Line 存放数据，而这个 Cache Set 所有 Line 都被占用的时候，就需要替换掉一个 Line。下面介绍几种替换策略：

1. **近期最少使用法（Least Recently Used，LRU）**
**选择最近被使用次数最少的 Line**，需要跟踪每个 Line 的使用情况，为每个 Line 设置个 age，每次访问就增加对应 Line 的 age 或者减少其它 Line 的 age，这样替换时选择年龄最小的哪个。实际随着 way 数的增加，精准实现这种 LRU 的开销就很大了，一般使用“伪 LRU “方式。即将所有 way 进行分组，每组使用一个 1 位的 age。下面是一个示例（就是一个二分查找的过程）
![[assets/ch1 概述/伪 LRU.png|SuperScalar/伪 LRU.png]]
2. **随机替换**
为了避免硬件复杂性，Random Replacement 方法不再记录每个 way 的年龄信息，而是随机选择一个 way 进行替换。
相比 LRU 发生缺失的频率会高一些，但是随着 Cache 容量的增加，这个差距越来越小。一般不是真随机，而是采用类似**时钟算法**来近似随机。本质是一个计数器，**每周期加 1，宽度由 Cache 的 way 来决定**，例如 8 路组相连就需要 3 位计数器。当需要替换 Line 时就是用计数器的值去替换对应的 Line。（**这里注意计时器是每个周期加 1，不是只有读写 Cache 才会加 1。所以可以实现对 Cache 的伪随机**），不一定最优，但是硬件简单不会损失过多性能，属于折中的策略。

## 2.2 提高 Cache 的性能

写缓存 write buffer、流水线 pipeline Cache、多级结构 multilevel Cache、Victim Cache 和预取 prefetching 等方法可以使无论是无论是顺序还是乱序执行的处理器性能提升。对于乱序处理器，还有如非阻塞 non-blocking Cache、关键字优先 critical word first 和提前开始 early restart 等方法。

### 2.2.1 写缓存

处理器中，无论是 load 还是 store 指令，D-Cache 发生缺失时，需要从下一级存储器读取数据，写到一个 Cache Line 中。如果这个 Cache Line 是脏的状态，则需要将其写入下级存储器。一般的下级存储器，如 L 2 Cache 或物理内存，一般只有一个读写端口，就要求上面的过程是*串行完成*的。即先将脏状态的 Cache line 中的数据写回下级存储器，然后才能读取下级存储器而得到缺失的数据，由于下级存储器的访问时间都比较长，串行访问导致 D-Cache 发生缺失的处理时间变得很长。这时候就可以用写缓存 Write Buffer 来解决，Dirty Cache Line 会**先写入缓存**，然后**等下级存储器空闲时写入**。

![](assets/ch1%20概述/WriteBuffer.png)

对于*写回 write back* 类型的 D-Cache：当一个脏状态的 Cache line 被替换时，这个 line 的数据会首先放到写缓存中，然后就可以从下级存储器中读数据了，写缓存的数据择机写到下级存储器。

对于*写通 write through* 类型的 D-cache：每次数据写到 D-Cache 的同时，就将其放到写缓存中，减少了写操作需要的时间。写通类型便于存储一致性 coherence 的管理，所以多核处理器的 L 1 Cache 多采用这种结构。

写缓存会增加系统复杂度，比如发生读缺失时，除了在下级存储器查找，还需要在写缓存中查找，因此需要在写缓存中加入**地址比较电路 CAM**，显然写缓存的数据最新，需要优先使用。

总结，写缓存相当于是 L 1 Cache 到下级存储器的缓冲，通过它，向下级存储器中写入数据的动作会被隐藏。尤其对于写通类似的 D-Cache 来说很重要。

### 2.2.2 流水线

对于读取 D-Cache 来说，由于 Tag SRAM 和 Data SRAM 可以在同时进行读取，所有当处理器的周期时间要求不是很严格时，可以在一个周期内完成读取的操作；对于写 D-Cache 来说，只能串行地完成。只要通过 Tag 比较，确认要写的地址在 Cache 中后，才可以写 Data SRAM，在主频比较高时，操作难以在一周期内完成，这就是要对写操作流水线。流水线的划分方式有很多，比较典型的是*将 Tag SRAM 地读取和比较放在一个周期，写 Data SRAM 放在下一个周期*。这样对于一条 store 指令来说，即使在 D-Cache 命中的时候，最快也需要两个周期才可以完成写操作。若连续地执行 store，可以获得流水地效果。

![](assets/ch2%20Cache/Cache%20写流水.png)

在上图的实现方式中，load 指令在 D-Cache 命中的情况下，可以在一个周期内完成，store 指令则一周期读取 Tag 一周期选择是否写入 Data。当指令 load 指令时，它需要的数据可能正好在 store 指令的流水线寄存器中（即上图中的 Delayed Store Data），而不在 Data SRAM 中，所以需要一种机制进行判断、转发。

### 2.2.3 多级结构

![](assets/ch2%20Cache/多级.png)

通常，L 1 Cache 容量很小，和处理器内核保持同样的速度等级，L 2 Cache 访问消耗几个处理器时钟周期，容量更大些，现代处理器中 L 1 和 L 2 Cache 通常在同一芯片内。一般在处理器中，L 2 Cache 会使用写回方式，对于 L 1 Cache 来说，写通实现也可以接受，这样可以简化流水线设计，便于在多核环境下管理存储器之间的一致性。

对于多结构得 Cache (multilevel Cache) 还需要了解两个概念 Inclusive/Exclusive，以 L 1/L 2 Cache 为例。

- **Inclusive**: L 2 Cache 包括了 L 1 Cache 中的所有内容。
- **Exclusive**: L 2 Cache 和 L 1 Cache 中的内容互不相同。
![](assets/ch2%20Cache/Inclusive%20exlusive.png)
Inclusive 比较浪费资源，因为它将一份数据保存在了两个地方，但这也带来了明显的好处，首先可以将数据直接写到 L 1 Cache 中，虽然此时会将 Cache line 中原来的数据覆盖，但是在 L 2 Cache 中存有这个数据的备份，所以覆盖不会引起问题（被覆盖的 line 不能是脏状态）；Inclusive 类型简化了一致性 coherence 管理，如在多核处理器中，一个处理器改变了存储器中一个地址的数据时，如果其他处理器的私有 Cache 中也保存了这个数据，那么需要将它们设为无效。对于 Inclusive 类型，只需要检查最低一级的 Cache 即可，这个例子中就是 L 2 Cache，如果没有发现对应的地址，也就不需要更新 L 1 Cache，避免影响处理器流水线；对于 Exclusive 类型，显然需要检查所有 Cache，检查 L 1 Cache 显然影响了处理器流水线，如果处理器要读取的数据不在 L 1 Cache 而是在 L 2 Cache 中，将数据从 L 2 Cache 放到 L 1 Cache 的同时，也需要将 L 1 Cache 中覆盖的 line 写到 L 2 Cache 中，这种交换数据的过程会降低处理器的效率。但是 Exclusive 类型 Cache 避免了硬件的浪费，一定程度上提高了整体性能。现在大多数采用 Inclusive。

### 2.2.4 Victim Cache

有时候，Cache 中被踢出的数据可能马上又要使用，比如一个两路组相联的 D-Cache，有 3 个数据位于同一个 Cache set。增大 way 数又会浪费空间，因为其他 set 未必有这种特性。Victim Cache 就是用来保存被排出的数据，通常采用全相连的方式，容量较小，位置如下。

![](assets/ch2%20Cache/VictimCache1.png)

![](assets/ch2%20Cache/VictimCache2.png)

Victim Cache 的本质相当于增加了 Cache 中 way 的个数，避免多个数据竞争 Cache 中有限的位置，从而降低 Cache 的缺失率。一般情况下 Cache 和 Victim Cache 存在互斥 exclusive 关系。如果在 Victim Cache 中找到了 Cache 中没有的数据，直接使用就可以了，和 Cache 命中效果一样，同时 Victim Cache 的数据会写入 Cache，相当于它们交换了数据，这个过程和 Exclusive Cache 行为一样。

还有一种与之类似的思路，称为 Filter Cache，它是在 Cache 之前，即第一个数据被使用时，不会马上放到 Cache 中，而是放在 Filter Cache 中，等到这个数据再次被使用，才会搬移到 Cache 中，避免那些偶然使用的数据占据 Cache。

![](assets/ch2%20Cache/Filter%20Cache.png)

### 2.2.5 预取

影响 Cache 缺失率的 3 C 定律中，Compulsory 可以通过预取 prefetching 缓解。

#### 硬件预取

对于指令来说，猜测后续会执行什么指令相对是比较容易的，因为程序本身就是串行执行的，因此只需要在访问 I - Cache 中的一个数据块（data block）的时候，将它后面的数据块也取出来放到 I - Cache 中就可以了，当然，由于程序中存在分支指令，所以这种猜测有时候也会出错，导致不会被使用的指令进入了 I - Cache，这一方面降低了 I - Cache 实际可用的容量，一方面又占用了本来可能有用的指令，这称为“Cache 污染”，不仅浪费了时间，还会影响处理器的执行效率，为了避免这种情况的发生，可以将预取的指令放到一个单独的缓存中，如图 2.23 所示。

![](assets/ch2%20Cache/指令硬件预取.png)

图 2.23 所示的方法实际上是在 Alpha 21064 处理器中对指令使用的预取方法，这种方法称为**硬件预取（Hardware Prefetching）**，当 I - Cache 发生缺失时，除了将需要的数据块（data block）从下级存储器（在图 2.23 中即为 L2 Cache）取出来并放到 I - Cache 中，还会将下一个数据块也读取出来，只不过它不会放到 I - Cache 中，而是放到一个称为 Stream Buffer 的地方，在后续执行时，如果在 I - Cache 中发生了缺失，但是在 Stream Buffer 中找到了想要的指令，那么除了使用 Stream Buffer 中读取的指令之外，还会将其中对应的数据块搬移到 I - Cache 中，同时继续从 L2 Cache 中读取下一个数据块（在图 2.23 中，即为 blockᵢ₊₂）放到 Stream Buffer 中，当程序中没有遇到分支指令时，这种方法会一直正确地工作，从而使 I - Cache 的缺失率得到降低，当然，分支指令会导致 Stream Buffer 中的指令变得无效，此时的预取相当于做了无用功，浪费了总线带宽和功耗，事实上，使用预取的方法是一把双刃剑，它可能会减少 Cache 的缺失率，也可能由于错误的预取而浪费功耗和性能，这种情况对于数据的预取尤为明显，需要在设计的时候仔细进行权衡。

不同于指令的预取，对于数据的预取来说，它的规律更难以进行捕捉，一般情况下，当访问 D - Cache 发生缺失时，除了将所需要的数据块从下级存储器中取出来之外，还会将下一个数据块也读取出来，这种方法被广泛地应用于现实当中的处理器中，例如 Intel 的 Core i7 处理器就采用了这种预取下一个数据块的方法，但是，这种方法并不总是很有效，因为程序以后要访问的数据有可能并不在下一个数据块中，这时候的预取就会浪费带宽和功耗，因此有些处理器采用了更激进的方法来实现数据的预取，例如在 Intel Pentium 4 和 IBM Power5 处理器中，采用了一种称为 Strided Prefetching 的方法，它能够使用硬件来观测程序中使用数据的规律，例如，某些数据结构的访问，第一个数据位于地址 a，第二个数据位于地址 a + 128，第三个数据位于地址 a + 256，那么负责预取的硬件就会预测下一个数据位于地址 a + 384，并将这个地址的数据预取出来，甚至会将地址 a + 512、a + 640、a + 768 等数据都进行预取，例如 IBM Power5 处理器就可以根据规律，提前预取 12 个数据块，这种预取的方法是比较激进的，对于一些程序，它有可能大幅提高执行速度，但是也可能对一些其他的程序带来负面的影响，毕竟这种方法只是一种猜测，而且这些预取来的数据需要替换掉 D - Cache 中一些可能在以后使用的数据，因此这种预取的方法是否真的需要使用，是需要根据实际的情况进行权衡的。

#### 软件预取

使用硬件进行数据的预取，很难得到满意的结果，其实，在程序的编译阶段，编译器（compiler）就可以对程序进行分析，进而知道哪些数据是需要进行预取的，如果在指令集中设有预取指令（prefetching instruction），那么编译器就可以直接控制程序进行预取，此时的预取就是比较有针对性的了，例如下面的一段程序。

```c
for (i = 0; i < N; i++) {
	Prefetch(&a[i+P]);
	Prefetch(&b[i+P]);
	SUM += a[i] + b[i];
}
```

为了提高性能，处理器需要在每周期同时执行多条 load/store 指令，这需要**多端口** D-Cache 支持多条指令同时访问。实际上超标量处理器中很多部件都是多端口结构的如 RF, IQ, ROB，但是这些部件本身容量不大，即使采用多端口设计也不会对芯片面积功耗产生太大影响，而 D-Cache 则不行。本节介绍三种实现多端口 D-Cache 的方法。

这个程序要计算两个数组之和，即 `a[i]+b[i]`，在进行计算之前，先将需要使用的数据使用软件预取出来，这样就可以保证在进行计算的时候，直接从 D-Cache 中就可以找到需要的数据，这就是**软件预取 (Software Prefetching)**，但是，这种软件预取的方法有一个前提，那就是预取的时机。如果预取数据的时间太晚，那么当真正需要使用数据时，有可能还没有被预取出来，这样的预取就失去了意义；如果预取的时间太早，那么就有可能踢掉 D-Cache 中一些本来有用的数据，造成 Cache 的“污染”。要选择一个合适的时机进行预取，也就是要决定程序中 P 的取值，不过，这并没有一个准确的答案，需要根据实际的设计情况来决定。

还需要注意的是，使用软件预取的方法，当执行预取指令的时候，处理器需要能够继续执行，也就是继续能够从 D-Cache 中读取数据，而不能够让预取指令阻碍了后面指令的执行，这就要求 D-Cache 是**非阻塞 (non-blocking)** 结构的，关于非阻塞结构的详细内容，将在后文进行介绍。

在实现了虚拟存储器 (Virtual Memory) 的系统中，预取指令有可能会引起一些异常 (exception)，例如发生 Page Fault、虚拟地址错误 (Virtual Address Fault) 或者保护违例 (Protection Violation) 等。此时有两种选择，如果对这些异常进行处理，就称这种预取指令为**处理错误的预取指令 (Faulting Prefetch Instruction)**，反之，如果不对这些异常进行处理，并抛弃掉这条预取指令，就称这种预取指令为**不处理错误的预取指令 (Nonfaulting Prefetch Instruction)**，此时发生异常的预取指令就会变成一条空指令，这种方法符合预取指令的定位，使预取指令“悄无声息”的执行，对于程序员来说是不可见的，现代的很多处理器都采用了这种方式。

## 2.3 多端口 Cache
### True Multi-port

直接多端口实际不现实。以双端口为例，所有的控制通路和数据通路都需要复制，它有两套地址解码器，使两个端口可以同时寻址 Tag SRAM/Data SRAM；有两个 MUX，同时读取两个端口的数据；比较器也翻倍，用于判断两个端口的命中情况；同时需要两个对齐器用来完成字节或半字的读取等。Tag SRAM/Data SRAM 本身不需要复制，但它们的每个 Cell 需要同时支持两个并行的读取操作（对于 SRAM cell，不需要两个写端口）。

![](assets/ch2%20Cache/多端口SRAM.png)

这种方式能提供双端口，但是需要复制很多电路，增大了面积，同时驱动多个读端口需要更长的访问时间，功耗也会增加。

### Multiple Cache Copied

![](assets/ch2%20Cache/MultiCacheCopies.png)

这种方式将 SRAM 复制，和真双口相比，不需要多端口，消除了处理器周期时间的影响，但是浪费了很多面积，而且需要两个之间的同步。

### Multi-banking

这种结构是现实中广泛使用的方法，它将 Cache 分为很多小 bank，每个 bank 都只有一个端口，如果一个周期之内，Cache 的多个端口上的访问地址都位于不同 bank 之中，这样不会引起问题，只有当两个或者多个端口的地址位于同一个 bank 之中，会引起 **bank 冲突 conflict**，这种方法称为 **multi-banking**。

这种方法，一个双端口的 Cache 仍需要两个地址解码器 Address Decoder、两个 MUX、两套比较器和两个对齐器，Data SRAM 不需要多端口结构了，而由于需要判断 Cache 的每个端口是否命中，Tag SRAM 需要多端口同时读取。

![](assets/ch2%20Cache/multibankingCache.png)

影响这种多端口 Cache 性能的关键因素就是 bank 冲突，可以采用更多 bank 数，也可以提高 bank 利用率，避免数据集中在一个 bank。同时优势每个端口都会访问所有的 bank，需要更多布线资源，对版图设计会有一定影响。这个方式总体使得 Cache 总面积降低，而且不会对处理器周期时间产生太大影响。

### AMD Opteron 多端口 Cache（TODO）

## 2.4 超标量处理器的取指令
