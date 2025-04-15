---
dateCreated: 2024-09-23
dateModified: 2025-04-12
---
# Overview

CS organization RISCV 版本

DDCA 书的重点：

ch 6

有对高级代码的直接 RISCV 汇编代码表示；对不同调用的表示，内存变化图示；对不同类型机器码的解释，以方便后续 RTL 设计。最后讲解了程序的表示；RISC 指令架构的其他要点、不同变种

ch 7

从头开始设计微架构，包括了单周期、状态机、多周期的设计。最后讲解了高级设计方法

ch8 memory

其他资料：

[前言 :: RISC-V CPU设计实验教程 (fpgaxlab.github.io)](https://fpgaxlab.github.io/jurv-open/site/jurv/v2.0/index.html)

# Cs Org RISCV

# 1 Computer Abstractions and Technology

# 2 Instructions: Language of the Computer
## 2.1 Introduction
**instruction set** The vocabulary of commands understood by a given architecture.
**stored-program concept** The idea that instructions and data of many types can be stored in memory as numbers and thus be easy to change, leading to the stored-program computer.

## 2.2 Operations of the Computer Hardware

> Design Principle 1: Simplicity favors regularity.

## 2.3 Operands of the Computer Hardware
*The operands must be from registers.*

> Design Principle 2: Smaller is faster.

Data structures (arrays and structures) are kept in memory.

RISC-V must include instructions that transfer data between memory and registers. Such instructions are called data transfer instructions.

To access a word or doubleword in memory, the instruction must supply the memory address.(address A value used to delineate the location of a specific data element within a memory array.)

load - memory -> register; store - register -> memory

Computers divide into those that use the address of the leftmost or “big end” byte as the doubleword address versus those that use the rightmost or “little end” byte. RISC-V belongs to the latter camp, referred to as little-endian.

**alignment restriction** A requirement that data be aligned in memory on natural boundaries. (RISCV and x86 don't align but MIPS does)

## 2.4 Signed and Unsigned Numbers
**binary digit** Also called bit. One of the two numbers in base 2, 0 or 1, that are the components of information.

$$
d *Base^i
$$

**least significant bit** The rightmost bit in an RISC-V doubleword.
**most significant bit** The leftmost bit in an RISC-V doubleword.
sign and magnitude - a single bit to indicate positive or negative
two’s complement representation: using sign bit

$$
(x_{63} * -2^{63})+(x_{62} * x^{62}) + ... + (x_0 * 2^0)
$$

**one’s complement** A notation that represents the most negative value by 10 … 000two and the most positive value by 01 … 11two, leaving an equal number of negatives and positives but ending up with two zeros, one positive (00 … 00two) and one negative (11 … 11two). The term is also used to mean the inversion of every bit in a pattern: 0 to 1 and 1 to 0.
**biased notation** A notation that represents the most negative value by 00 … 000two and the most positive value by 11 … 11two, with 0 typically having the value 10 … 00two, thereby biasing the number such that the number plus the bias has a non-negative representation.

## 2.5 Representing Instructions in the Computer
**instruction format** A form of representation of an instruction composed of fields of binary numbers.
machine language Binary representation used for communication within a computer system.
- RISCV Fields(R type for registers)
![[RISCVfield.png]]

> Design Principle 3: Good design demands good compromises.

I-type and is used by arithmetic operands with one constant operand, including addi, and by load instructions.

RISCV doesn't have subi, because the immediate field represents two's implement, so addi can be used to subtract constants.

12 bits immediate is interpreted as a two's complement value, so it can represent integers from $-2^{11}\ to\ 2^{11}-1$。When used for load double word instr., it can refer to any doubleword within a region of ±211 or 2048 bytes (±28 or 256 doublewords) of the base address in the base register rd.

![[Fig2.5.png]]

- Big picture
Today’s computers are built on two key principles: 1. Instructions are represented as numbers. 2. Programs are stored in memory to be read or written, just like data.
![[Fig2.7.png]]

## 2.6 Logical Operations

## 2.7 Instructions for Making Decisions
**conditional branch** An instruction that tests a value and that allows for a subsequent transfer of control to a new address in the program based on the outcome of the test.
In general, the code will be more efficient using bne than beq.
**basic block** A sequence of instructions without branches (except possibly at the end) and without branch targets or branch labels (except possibly at the beginning).

- Alternative
Set a register based upon the result of the comparison, then branch on the value(MIPS). It makes datapath slightly simpler but takes more instructions.
Keep extra bits that record what occurred during an instruction, called condition codes or flags. One downside to condition codes is that if many instructions always set them, it will create dependencies that will make it difficult for pipelined execution

**branch address table** Also called branch table. A table of addresses of alternative instruction sequences.
RISC-V include an indirect jump instruction, which performs an unconditional branch to the address specified in a register. In RISC-V, the jump-and-link register instruction (jalr) serves this purpose.

## 2.8 Supporting Procedures in Computer Hardware

## 2.10 RISC-V Addressing for Wide Immediates and Addresses
- Wide Immediate Operands
The RISC-V instruction set includes the instruction Load upper immediate (lui) to load a 20-bit constant into bits 12 through 31 of a register.
- Addressing in Branches
The RISC-V branch instructions use the RISC-V instruction format called SBtype.
The unconditional jump-and-link instruction (jal) is the only instruction that uses the UJ-type format.
Alternative: Program counter Register Branch offset
PC-relative addressing An addressing regime in which the address is the sum of the program counter (PC) and a constant in the instruction.

RISC-V uses PC-relative addressing for both conditional branches and unconditional jumps.

Hence, RISC-V allows very long jumps to any 32bit address with a two-instruction sequence: lui writes bits 12 through 31 of the address to a temporary register, and jalr adds the lower 12 bits of the address to the temporary register and jumps to the sum.

Occasionally conditional branches branch far away, farther than can be represented in the 12-bit address in the conditional branch instruction. The assembler comes to the rescue just as it did with large addresses or constants: it inserts an unconditional branch to the branch target, and inverts the condition so that the conditional branch decides whether to skip the unconditional branch.

- RISC-V Addressing Mode Summary
addressing mode One of several addressing regimes delimited by their varied use of operands and/or addresses.
![[Fig2.17.png]]
1. Immediate addressing, where the operand is a constant within the instruction itself.
2. Register addressing, where the operand is a register.
3. Base or displacement addressing, where the operand is at the memory location whose address is the sum of a register and a constant in the instruction.
4. PC-relative addressing, where the branch address is the sum of the PC and a constant in the instruction.
![[Fig2.18.png]]
![[Fig2.19.png]]

## 2.13 A C Sort Example to Put it All Together

# 2 The Processor

# 3 Archimetric for Computers

## 3.4 Division

dividend / divisor = quotient - remainder

### A Division Algorithm and Hardware

The hardware mimics grammar school algorithm by iterate the shift and comparison operation.

![[Fig3.8.png]]

![[Fig3.9.png]]

The signed division makes sure that the dividend and remainder have identical signs

$$
+7 div -2 = -3 remain +1
$$

Other techniques to produce more than one bit of the quotient per step such as SRT division which tries to predict several quotient bits.

RISC-V have instructions for division and remainder: div, divu, rem, remu

> RISC-V divide ignore overflow and division by 0, so software must check the divisor and quotient
> restoring algorithm dont immediately add the divisor back if the remainder is negative; dont save the result of the subtract is nonperforming division algorithm
![[Fig3.12.png]]

## 3.5 Floating Point

# 5 Large and Fast: Exploiting Memory Hierarchy
## 5.1 Introduction
**principle of locality**
- **temporal locality** The locality principle stating that if a data location is referenced then it will tend to be referenced again soon.
- **spatial locality** The locality principle stating that if a data location is referenced, data locations with nearby addresses will tend to be referenced soon.
**memory hierarchy** A structure that uses multiple levels of memories; as the distance from the processor increases, the size of the memories and the access time both increase.
![[Fig5.1.png]]

- **block (or line)** The minimum unit of information that can be either present or not present in a cache.
- **hit rate** The fraction of memory accesses found in a level of the memory hierarchy.
- **miss rate** The fraction of memory accesses not found in a level of the memory hierarchy.
- hit time The time required to access a level of the memory hierarchy, including the time needed to determine whether the access is a hit or a miss.
- **miss penalty** The time required to fetch a block into a level of the memory hierarchy from the lower level, including the time to access the block, transmit it from one level to the other, insert it in the level that experienced the miss, and then pass the block to the requestor.

![[Fig5.2.png]]

## 5.2 Memory Technologies

![[csorg.assets/5memTech.png]]

DRAMs use a two-level decoding structure, and this allows us to refresh an entire row (which shares a word line) with a read cycle followed immediately by a write cycle.

![[Fig5.4.png]]

The advantage of SDRAMs is that the use of a clock eliminates the time for the memory and processor to synchronize. The speed advantage of synchronous DRAMs comes from the ability to transfer the bits in the burst without having to specify additional address bits.

Double Data Rate (DDR) SDRAM. The name means data transfers on both the rising and falling edge of the clock, thereby getting twice as much bandwidth as you might expect based on the clock rate and the data width.

**address interleaving** DRAM can be internally organized to read or write from multiple banks, with each having its own row buffer. For example, with four banks, there is just one access time and then accesses rotate between the four banks to supply four times the bandwidth.
**DIMMs** individual DRAMs, memory for servers is commonly sold on small boards called dual inline memory modules . A DIMM has multiple DRAM chips. Such a subset of chips in EIMM is **memory rank**.

**Flash Memory** is a type of electrically erasable programmable read-only memory (EEPROM).
**wear leveling** Writes can wear out flash memory bits and most flash products include a controller to spread the writes by remapping blocks that have been written many times to less trodden blocks.

**Disk Memory**
**track** One of thousands of concentric circles that make up the surface of a magnetic disk.
**sector** One of the segments that make up a track on a magnetic disk; a sector is the smallest amount of information that is read or written on a disk.

## 5.3 The Basics of Caches
### Accessing a Cache
**direct-mapped cache** A cache structure in which each memory location is mapped to exactly one location in the cache.
![[Fig5.8.png]]
**tag** A field in a table used for a memory hierarchy that contains the address information required to identify whether the associated block in the hierarchy corresponds to a requested word. e.g.

$$
(Block address) modulo (Number of blocks in the cache)
$$

**valid bit** A field in the tables of a memory hierarchy that indicates that the associated block in the hierarchy contains valid data.
![[Fig5.9.png]]
**temporal locality** recently referenced words replace less recently referenced words.
![[Fig5.10.png]]
e.g.
- 64-bit addresses
- A direct-mapped cache
- The cache size is $2^n$ blocks, so n bits are used for the index
- The block size is $2^m$ words ($2^{m+2}$ bytes), so $m$ bits are used for the word within the block, and two bits are used for the byte part of the address
The size of the tag field is $64-(m+n+2)$. The total number of bits in a direct-mapped cache is $2^n × (block\ size + tag\ size + valid\ field\ size) = 2^n * (2^m x 32 +63-m-n)$.

Stated alternatively, spatial locality among the words in a block decreases with a very large block; consequently, the benefits to the miss rate become smaller.

![[Fig5.11.png]]

**early restart** resume execution as soon as the requested word of the block is returned, rather than wait for the entire block

**requested word first or critical word first** the requested word is transferred from the memory to the cache first. The remainder of the block is then transferred, starting with the address after the requested word and wrapping around to the beginning of the block

The two techniques work well for instruction access because instructoin access are largely sequential while the data cache access is less predictable.

### Handling Cache Misses
**cache miss** A request for data from the cache that cannot be filled because the data are not present in
steps to be taken on an instruction cache miss:
1. Send the original PC value to the memory.
2. Instruct main memory to perform a read and wait for the memory to complete its access.
3. Write the cache entry, putting the data from memory in the data portion of the entry, writing the upper bits of the address (from the ALU) into the tag field, and turning the valid bit on.
4. Restart the instruction execution at the first step, which will refetch the instruction, this time finding it in the cache.

- Handling Writes
**write-through** A scheme in which writes always update both the cache and the next lower level of the memory hierarchy, ensuring that data are always consistent between the two.
**write buffer** A queue that holds data while the data are waiting to be written to memory.
		 miss: write allocate - the block is fetched from memory and overwritten; no write allocate - update in memory but not put it in the cache
**write-back** A scheme that handles writes by updating values only to the block in the cache, then writing the modified block to the lower level of the hierarchy when the block is replaced.(better performance but more complex to implement)
		store either requires two cycles(check for a hit and perform the write) or need a writer buffer

- An Example Cache: The Intrinsity FastMATH Processor
**split cache** A scheme in which a level of the memory hierarchy is composed of two independent caches that operate in parallel with each other, with one handling instructions and one handling data.
![[Pasted image 20241008195727.png]]

## 5.4 Measuring and Improving Cache Performance

$$
CPU\ time = (CPU\ execution\ clock\ cycles + Memory-stall\ clock\ cycles\ Clock\ cycle\ time
$$

Assume that the costs of cache accesses that are hits are part of the normal CPU execution cycles

$$
Memory-stall\ clock\ cycles = (Read-stall\ cycles + Write-stall\ cycles)
$$

$$
Read-stall\ cycles\ Reads  = \frac{Read}{Program} × \ miss\ rate × Read\ miss\ penalty
$$

For write-through:

$$
Write-stall\ cycles\ Writes  = \frac{Write}{Program} × \ miss\ rate × Write\ miss \ penalty+Write buffer stalls
$$

Assume that the write buffer stalls are negligible:

$$
Memory-stall\ clock\ cycles\ = \frac{Memory\ accesses }{Program} × Miss\ rate × Miss\ penalty
$$

$$
Memory-stall\ clock\ cycles\ = \frac{Instruction}{Program} \times \frac{misses}{Instruction}\ \times Miss\ penalty
$$

**average memory access time (AMAT)**

$$
AMAT = Time\ for\ a\ hit + Miss\ rate × Miss\ penalty
$$

### Reducing Cache Misses by More Flexible Placement of Blocks
**fully associative cache** A cache structure in which a block can be placed in any location in the cache.
**set-associative cache** A cache that has a fixed number of locations (at least two) where each block can be placed.
n-way set-associative cache - n locations for a block
Each block in the memory maps to a unique set in the cache given by the index field, and a block can be placed in any element of that set.
The set containing a memory block is given by $(Block\ number) modulo (Number\ of\ sets\ in\ the\ cache)$
Since the block may be placed in any element of the set, all the tags of all the elements of the set must be searched.
![[Fig5.15.png]]
The cache access consists of indexing the appropriate set and then searching the tags of the set.
![[Fig5.18.png]]
A Content Addressable Memory (CAM) is a circuit that combines comparison and storage in a single device(eight-way and above built using CAMs in 2013).

- Choosing Which Block to Replace
**least recently used (LRU)** A replacement scheme in which the block replaced is the one that has been unused for the longest time. LRU replacement is implemented by keeping track of when each element in a set was used relative to the other elements in the set.
- Reducing the Miss Penalty Using Multilevel Caches
**multilevel cache** A memory hierarchy with multiple levels of caches, rather than just a cache and main memory

### Software Optimization via Blocking (TODO)

# 6 Parallel Processors from Client to Cloud
## 6.1 Introduction

# Digital Design and Computer Architecture

# Microarchitecture

architectural state: program counter and 32 32-bit regitster

实现部分指令 (RV32I)：

R-type: add, sub, and, or, slt

Memory: lw, sw

Branch: beq

![[RVFig7.1.png]]

Pipeline: Fetch, Decode, Execute, Memory, and Writeback.

## Advanced Architecture
- **Deep pipelines**
- **Micro-Operations**
- **Branch Prediction**
- **Superscalar Processor**
A superscalar processor contains multiple copies of the datapath hardware to execute multiple instructions simultaneously.
- **Out-of-Order Processor**
Out-of-order processors use a table to keep track of instructions waiting to issue. The table, sometimes called a scoreboard, contains information about the dependencies. The size of the table determines how many instructions can be considered for issue.
- **Register Renaming**
- **Multithreading**
A program running on a computer is called a **process**.
Each process consists of one or more **threads** that also run simultaneously.
The degree to which a process can be split into multiple threads that can run simultaneously defines its level of **thread-level parallelism (TLP)**.
When one thread’s turn ends, the OS saves its architectural state, loads the architectural state of the next thread, and starts executing that next thread. This procedure is called **context switching**.
A hardware multithreaded processor contains more than one copy of its architectural state so that more than one thread can be active at a time.
Switching between threads can either be fine-grained or coarse-grained. **Fine-grained multithreading** switches between threads on each instruction and must be supported by hardware multithreading. **Coarse-grained multithreading** switches out a thread only on expensive stalls, such as long memory accesses due to cache misses.
Multithreading doesn't improve the performance of an individual thread(no ILP increase) but improves the overall throughput because multiple threads can use processor resources that would have been idle when executing a single thread. Multithreading is also relatively inexpensive to implement because it replicates only the PC and register file, not the execution units and memories.
- **Multiprocessors**
1. Symmetric multiprocessors include two or more identical processors sharing a single main memory. They are easy to design and programming for.
2. Heterogeneous multiprocessors add complexity in terms of both designing the different heterogeneous elements and the additional programming effort to decide when and how to make use of the varying resources.
3. Processors in clustered multiprocessor systems each have their own local memory system instead of sharing memory.

Western Digital’s SweRV cores, the SweRVolf SoC, and the PULP (Parallel Ultra Low Power) Platform.

SiFive’s Freedom E310 core and Western Digital’s open-source SweRV
