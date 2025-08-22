---
dateCreated: 2025-05-29
dateModified: 2025-05-29
---
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
(x_{63} * -2^{63})+(x_{62} * x^{62}) + … + (x_0 * 2^0)
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
![[Fig2.17.png]]
1. Immediate addressing, where the operand is a constant within the instruction itself.
2. Register addressing, where the operand is a register.
3. Base or displacement addressing, where the operand is at the memory location whose address is the sum of a register and a constant in the instruction.
4. PC-relative addressing, where the branch address is the sum of the PC and a constant in the instruction.

![](Fig2.18.png)

![](Fig2.19.png)

## 2.13 A C Sort Example to Put it All Together
