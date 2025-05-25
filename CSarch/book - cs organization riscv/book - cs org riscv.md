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
![[assets/RISCVfield.png]]

> Design Principle 3: Good design demands good compromises.

I-type and is used by arithmetic operands with one constant operand, including addi, and by load instructions.

RISCV doesn't have subi, because the immediate field represents two's implement, so addi can be used to subtract constants.

12 bits immediate is interpreted as a two's complement value, so it can represent integers from $-2^{11}\ to\ 2^{11}-1$。When used for load double word instr., it can refer to any doubleword within a region of ±211 or 2048 bytes (±28 or 256 doublewords) of the base address in the base register rd.

![[assets/Fig2.5.png]]

- Big picture
Today’s computers are built on two key principles: 1. Instructions are represented as numbers. 2. Programs are stored in memory to be read or written, just like data.
![[../../IC/program/assets/Fig2.7.png]]

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
![[assets/Fig2.17.png]]
1. Immediate addressing, where the operand is a constant within the instruction itself.
2. Register addressing, where the operand is a register.
3. Base or displacement addressing, where the operand is at the memory location whose address is the sum of a register and a constant in the instruction.
4. PC-relative addressing, where the branch address is the sum of the PC and a constant in the instruction.
![[../assets/csorg/Fig2.18.png]]
![[../assets/csorg/Fig2.19.png]]

## 2.13 A C Sort Example to Put it All Together

# 2 The Processor

# 3 Archimetric for Computers

## 3.4 Division

dividend / divisor = quotient - remainder

### A Division Algorithm and Hardware

The hardware mimics grammar school algorithm by iterate the shift and comparison operation.

![[../assets/csorg/Fig3.8.png]]

![[../assets/csorg/Fig3.9.png]]

The signed division makes sure that the dividend and remainder have identical signs

$$
+7 div -2 = -3 remain +1
$$

Other techniques to produce more than one bit of the quotient per step such as SRT division which tries to predict several quotient bits.

RISC-V have instructions for division and remainder: div, divu, rem, remu

> RISC-V divide ignore overflow and division by 0, so software must check the divisor and quotient
> restoring algorithm dont immediately add the divisor back if the remainder is negative; dont save the result of the subtract is nonperforming division algorithm
![[../assets/csorg/Fig3.12.png]]

## 3.5 Floating Point


### Software Optimization via Blocking (TODO)

# 6 Parallel Processors from Client to Cloud
## 6.1 Introduction

# Digital Design and Computer Architecture


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
