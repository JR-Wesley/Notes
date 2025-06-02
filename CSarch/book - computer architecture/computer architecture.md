---
dateCreated: 2024-09-23
dateModified: 2025-05-26
---

# Ch1 Fundamentals of Quantitative Design and Analysis

## 1.1 Introduction

Rapid improvement has come both from **technology** used to build computers & **innovations** in computer design. An increasing fraction of the computer business being based on microprocessors and two changes to succeed with new architecture:

1. the virtual elimination of assembly language programming reduced the need for object-code compatibility
2. the creation of standardized, vendor-independent operating systems, such as UNIX and its clone, Linux, lowered the cost and risk of bringing out a new architecture

RISC focus on two critical performance :

1. the exploitation of **instruction-level parallelism** (initially through pipelining and later through multiple instruction issue)
2. the use of **caches** (initially in simple forms and later using more sophisticated organizations and optimizations).

**The preceding hardware innovations led to a renaissance in computer design which emphasized both architectural innovation and efficient use of technology improvements**
This hardware renaissance allow modern programmers trade performance for productivity.
Alas, 17-year hardware renaissance is over:
1. Dennard scaling ended around 2004 because current and voltage couldn’t keep dropping and still maintain the dependability of integrated circuits.
   -> use multiple efficient processors or cores
   This milestone signaled a historic switch from relying solely on instruction-level parallelism (ILP), the primary focus of the first three editions of this book, to data-level parallelism (DLP) and thread-level parallelism(TLP)
2. Moore’s Law recently ended.
- transistors no longer getting much better because of the slowing of Moore’s Law and the end of Dennard scaling,
- the unchanging power budgets for microprocessors
- the replacement of the single power-hungry processor with several energy efficient processors
- the limits to multiprocessing to achieve Amdahl’s Law
The only path left to improve energy-performance-cost is **specialization**

## 1.2 Classes of Computers

![[csarchGraph/Fig1.2.png]]

- Classes of Parallelism and Parallel Architectures
1. Data-level parallelism (DLP) arises because there are many data items that can be operated on at the same time.
2. Task-level parallelism (TLP) arises because tasks of work are created that can operate independently and largely in parallel.

Computer hardware in turn can exploit these two kinds of application parallelism in four major ways:

1. Instruction-level parallelism exploits data-level parallelism at modest levels with compiler help using ideas like pipelining and at medium levels using ideas like speculative execution.
2. Vector architectures, graphic processor units (GPUs), and multimedia instruction sets exploit data-level parallelism by applying a single instruction to a collection of data in parallel.
3. Thread-level parallelism exploits either data-level parallelism or task-level parallelism in a tightly coupled hardware model that allows for interaction between parallel threads.
4. Request-level parallelism exploits parallelism among largely decoupled tasks specified by the programmer or the operating system.

 - Flynn abbreviations：
1. Single instruction stream, single data stream (SISD)
2. Single instruction stream, multiple data streams (SIMD)
3. Multiple instruction streams, single data stream (MISD)
4. Multiple instruction streams, multiple data streams (MIMD）

## 1.3 Defining Computer Architecture
**computer architecture** instruction set design & implementation

### **Instruction Set Architecture: The Myopic View of Computer Architecture**

(programmer-visible instruction)

RISC-V's good ideas: a large set of registers, easy-to-pipeline instructions, and a lean set of operations

1. **Class of ISA**: general-purpose register architectures, where the operands are either registers or memory locations. The two popular versions of this class are **register-memory** ISA, which can access memory as part of many instructions, and **load-store** ISAs which can access memory only with load or store instructions.
2. **Memory addressing**: Virtually all computers use byte addressing to access memory operands. Some require that objects must be aligned and some not.
3. **Addressing modes**: RISC-V addressing modes are Register, Immediate (for constants), and Displacement.
4. **Types and sizes of operands** - 8-bit , 16-bit, 32-bit, 64-bit, and IEEE 754 floating point in 32-bit and 64-bit.
5. **Operations** - The general categories of operations are data transfer, arithmetic logical, control, and floating point.
6. **Control flow instructions** - Virtually all ISAs support conditional branches, unconditional jumps, procedure calls, and returns
7. **Encoding an ISA** - There are two basic choices on encoding: fixed length and variable length.
![[csarchGraph/Fig1.7.png]]

### **Genuine Computer Architecture: Designing the Organization and Hardware to Meet Goals and Functional Requirements**
**implementation = organization + hardware**
**organization**(microarchitecture) includes high-level aspects of a computer’s design, such as the memory system, the memory interconnect, and the design of the internal processor or CPU
**Hardware** refers to the specifics of a computer, including the detailed logic design and the packaging technology of the computer
**architecture** - instruction set architecture, organization or microarchitecture, and hardware
![[csarchGraph/Fig1.8.png]]

## 1.4 Trends in Technology(waiting)

## 1.5 Trends in Power and Energy in Integrated Circuits(waiting)

## 1.6 Trends in Cost(waiting)

## 1.7 Dependability(waiting)

## 1.8 Measuring, Reporting, and Summarizing Performance
**response time/execution time** - the time between the start and the completion of an event
**throughput** - the total amount of work done in a given time
**wall-clock time, response time, or elapsed time** - the latency to complete a task
**CPU time** - the time the processor is computing, not including the time waiting for I/O or running other programs

- Benchmarks
Attempts at running programs that are much simpler than a real application have led to performance pitfalls, such as kernels, toy programs, synthetic benchmarks
Another issue is the conditions under which the benchmarks are run, such as illegal flags.
**benchmark suites** - collections of benchmark applications

- Reporting Performance Results
**reproducibility**
We could normalize execution times to a reference computer by dividing the time on the reference computer by the time on the computer being rated, yielding a ratio proportional to performance.
mean must be computed using the geometric mean

$$
Geometric\ mean = (\sum_{i=1}^n sample_i)^{1/n}
$$

## 1.9 Quantitative Principles of Computer Design

### Take Advantage of Parallelism

Being able to expand memory and the number of processors and storage devices is called **scalability**.

ILP/DLP

### Principle of Locality

Programs tend to reuse data and instructions they have used recently.

Temporal locality/ Spatial locality

### Focus on the Common Case
**Amdahl’s Law** states that the performance improvement to be gained from using some faster mode of execution is limited by the fraction of the time the faster mode can be used. It depends on two things:
1. The fraction of the computation time in the original computer that can be converted to take advantage of the enhancement - $Fraction_{enhanced}$
2. The improvement gained by the enhanced execution mode, that is, how much faster the task would run if the enhanced mode were used for the entire program - $Speedup_{enhanced}$
The overall speedup is

$$
Speedup_{overall} = \frac{1}{(1-Fraction_{enhanced})+ \frac{Fraction_{enhanced}}{Speedup_{enhanced}}}
$$

Amdahl’s Law expresses the law of diminishing returns: The incremental improvement in speedup gained by an improvement of just a portion of the computation diminishes as improvements are added.

### The Processor Performance Equation

$$
CPU\ time = CPU\ clock\ cycles\ for\ a\ program \times Clock\ cycle\ time = CPU\ clock\ cycles\ for\ a\ program \times Clock\ rate
$$

$$
clock\ cycles\ per\ instruction(CPI) =\frac{CPU\ clock\ cycles\ for\ a\ program  }{Instruction\ count(IC)}
$$

$$
CPU\ time = IC\times CPI \times Clock\ cycle\ time
$$

$$
CPI = \sum_{i=1}^n \frac{IC_i}{IC}\times CPI_i
$$

## 1.10 Putting It All Together: Performance, Price, and Power(waiting)

## 1.11 Fallacies and Pitfalls(waiting)

# Ch2 Memory Hierarchy Design

## 2.1 Introduction

The solution to unlimited amounts of fast memory is a memory hierarchy, which takes advantage of locality and trade-offs in the cost-performance of memory technology. The (spatial and temporal) locality is that most programs do not access code or data uniformly.

A memory hierarchy is organized into several levels—each smaller, faster, and more expensive per byte than the next lower level, which is farther from the processor. The goal is to provide a memory system with a cheapest-level cost per byte and a fastest-level speed.

Inclusion property is that the data contained in a lower level are a superset of the next higher level. It's required for the lowest level (main memory in the case of caches, the second storage (disk or flash) in the case of virtual memory) but not for all the level in all cases.

Although the gap in access time increased significantly for many years, the lack of significant performance improvement in single processors has led to a slowdown in the growth of the gap between processors and DRAM. the gap between CPU memory demand and DRAM bandwidth continues to grow as the numbers of cores grow.

Traditionally, designers of memory hierarchies focused on optimizing average memory access time, which is determined by the cache access time, miss rate, and miss penalty. Power has become a major consideration as static and dynamic power account for a large proportion of total power consumption.

### Basics of Memory Hierarchies: A Quick Review

Each cache block (Multiple words) includes a tag to indicate which memory address it corresponds to. If there are n blocks in a set, the cache placement is called n-way set associative. Finding a block consists of first mapping the block address to the set and then searching the set—usually in parallel—to find the block. The set number equals to $<Block address> \mod <Number of sets in cache>$. The end points of set associativity is direct-mapped (one block per set) and fully associative (a block in anywhere).

Caching data that is only read is easy because the copy in the cache and memory will be identical. There are two main strategies to caching writes. A write-through cache updates the item in the cache and writes through to update main memory. A write-back cache only updates the copy in the cache. Both write strategies can use a write buffer to allow the cache to proceed as soon as the data are placed in the buffer rather than wait for full latency to write the data into memory.

## 2.2 Memory Technology and Optimizations

# Ch3 Instruction-level Parallelism and Its Exploitation

## 3.1 Instruction-Level Parallelism: Concepts and Challenges

The potential overlap among instructions is called **instruction-level parallelism(ILP)**.

First look at the limitation imposed by data and control hazards and then turn to increasing the ability of the compiler and the processor to exploit the parallelism

2 approaches to exploting ILP:

1. **HW to discover and exploit the parallelism dynamically**
2. **SW to find parallelsim statically at compile time**
The limitations on ILP approaches led to the movement toward multicore.
This section will discuss features of both programs and processors that limit the amount of parallelism that can be exploited among instructions, as well as the critical mapping between program structure and hardware structure.

$$
Pipeline\ CPI=Ideal\ pipeline\ CPI+Structural\ stalls+Data\ hazard\ stalls+Control\ stalls
$$

![[csarchGraph/Fig3.1.png]]

### ILP

a basic block - a straight-line code sequence with no branches in except to the entry and no branches out except at the exit

The amount of parallelism within a basic block is small.

- *Loop-level parallelism: every iteration of the loop can overlap*

```c
for(i = 0; i <= 999; i = i + 1)
	x[i] = x[i] + y[i];
```

Techniques for converting such loop-level parallelism into instruction-level parallelsim include unrolling the loop **statically by the compiler or dynamically by the hardware**.

An alternative method is SIMD in vector processors and GPUs. A SIMD instruction exploits data-level parallelism by operating on a small to moderate number of data items in parallel. A vector instruction exploits data-level parallelism by operating on many data items in parallel using both parallel execution units and a deep pipeline.

### Data Dependences and Hazards

If two instructions are parallel, they can execute **simultaneously** in a pipeline of arbitrary depth without causing any stalls, assuming the pipeline has sufficient resources(no structrall hazards)

If two instructions are data-dependent, they must execute **in order**.

#### Data Dependences

3 types:

- *(true)data dependences*
- *name dependences*
- *control dependences*
an instruction j is data-dependent on instruction i if:
- i produces a result that may be used by j
- j depend on k, k depend on i(a chian dependences)

```c
Loop: fld f0,0(x1) //f0=array element
	fadd.d f4,f0,f2 //add scalar in f2
	fsd f4,0(x1) //store result
	
	addi x1,x1,-8 //decrement pointer 8 bytes
	bne x1,x2,Loop //branch x16!=x2
```

A processor with pipeline interlock will **detect a hazard and stall, reducing the overlap(HW)**. A processor without pipeline interlock relies on compiler scheduling

Data dependences are a property of programs. They convert 3 things:

1. the possibility of a hazard
2. the order of the calculation
3. an upper bound on parallelsim
A dependence can be overcome in 2 ways:
4. maintaining the dependence but avoiding a hazard
5. eliminating a dependence by transforming the code(Scheduling both by SW&HW)
A data value may flow between instructions through **registers or memory location**(more difficult to detect)

#### Name Dependecs

Name Dependencs: when two instructions use the same register or memory location, called a *name*, but there is no flow of data. 2 types(instr. i precedes instr. j in program order):

1. *antidependence*: j writes a reg or mem location that i reads
2. *output dependence*: i and j write and same reg or mem location
There is no value transmitted between the instructions and they can execute simultaneously or be reordered. Register renaming can be done by SW or HW

#### Data Hazards

A hazard exists whenever there is a name or data dependence between instructions and they are close enough that the overlap during execution would change the order of access to the operand. We must preserve the program order. The SW and HW techniques to exploit parallelsim by preserving program order only where is affects the outcome of the program. Instructions i and j:

- RAW(read after write) - true dependence
- WAW(write after write) - output dependence. In pipelines that write in more than one pipe stage or allow an instruction to proceed even when a previous instruction is stalled.
- WAR(write after read) - antidependence(name). In most static issue pipelines, all reads are early and writes are late

### Control Dependences

the order of an instruction with respect to a branch instruction

```c
if p1{
	S1;
};
```

S1 is control-dependent on p1. 2 constraints are imposed:

1. An instruction that is control-dependent on a branch cannot be moved before the branch.
2. An instruction that is not control-dependent on a branch cannot be moved after the branch
Control dependences is not the critical property that must be preserved. The two critical properties are:
- **the exception behavior**: any changes in the ordering of instruction execution must not change how exceptions are raised in the program. The reordering of instruction execution must not cause any new exceptions.
- **and the data flow**: branch makes data flow dynamic. An instruction may be data-dependent on more than one predecessor
Speculation overcomes exception and lessen the impact of control problem.
Sometimes violating the control dependence cannot affect the effect of program:

```c
	add x1,x2,x3
	beq x12,x0,skip
	sub x4,x5,x6
	add x5,x4,x9
skip: or x7,x8,x9
```

Suppose sub instruction was unsed after skip(The property ofwhether a value will be used by an upcoming instruction is called liveness). x4 is dead and sub don't generate exception, we could move the sub instr before the branch.

The type of code scheduling is called software speculation. Control dependence is preserved by implementing control hazard detection that causes control stalls.

## 3.2 Basic Compiler Techniques for Exposing ILP

Compiler technology are crucial for processors that use statiic issue or static scheduling

### Basic Pipeline Scheduling and Loop Unrolling

Finding sequences of unrelated instructions that can be overlapped in the pipeline to keep the pipeline full. The execution of dependent instruction must be separated to avoid pipeline stall.

A compiler’s ability to perform this scheduling depends both on **the amount of ILP available** in the program and on **the latencies of the functional units** in the pipeline.

```cpp
for (i=999; i>=0; i=i+1)
	 x[i] = x[i] + s;
```

![[csarchGraph/Fig3.2.png]]

The straightforward RISC-V code, not scheduled for the five-stage pipeline:

```c
Loop:   fld f0,0(x1) //f0=array element 
		// stall
		fadd.d f4,f0,f2 //add scalar in f2 
		// stall 
		// stall
		fsd f4,0(x1) //store result 
		addi x1,x1, -8 //decrement pointer //8 bytes (per DW) 
		bne x1,x2,Loop //branch x1 != x2
```

Without any scheduling, it takes eight cycles. Scheduling the loop can obtain only two stalls.

```c
Loop:   fld f0,0(x1) //f0=array element 
		addi x1,x1, -8 //decrement pointer //8 bytes (per DW) 
		fadd.d f4,f0,f2 //add scalar in f2 
		// stall 
		// stall
		fsd f4,0(x1) //store result 
		bne x1,x2,Loop //branch x161⁄4x2
```

The actual work of operating on the array element takes just three (the load, add, and store) of those seven clock cycles. The remaining four clock cycles consist of loop overhead—the addi and bneand two stalls.

**Loop unrolling** simply replicates the loop body multiple times, adjusting the loop termination code.

**strip mining**: loop bound n -> first executed (n mod k) times and unroll the rest to (n/k) times.

### Summary of the Loop Unrolling and Scheduling

The key to most of these techniques is to know when and how the ordering among instructions may be changed.

- Finding that the loop iterations were independent, except for the loop maintenance code.
- Use different registers to avoid unnecessary constraints
- Eliminate the extra test and branch instructions and adjust the loop termination and iteration code.
- Determine that the loads and stores in the unrolled loop can be interchanged by observing that the loads and stores from different iterations are independent.
- Schedule the code, preserving any dependences needed
Three different effects *limit* the gains from loop unrolling:
(1) a decrease in the amount of overhead amortized with each unroll,
(2) code size limitations(may lead to cache miss or **register pressure**)
(3) compiler limitations.

## 3.3 Reducing Branch Costs With Advanced Branch Prediction(waiting)
### Correlating Branch Predictors

Branch predictors that use the behavior of other branches to make a prediction are called **correlating predictors or two-level predictors**. In the general case, an (m,n) predictor uses the behavior of the last m branches to choose from 2m branch predictors, each of which is an n-bit predictor for a single branch.

[指令级并行技术 | KuangjuX(狂且)](https://blog.kuangjux.top/2022/03/04/ILP/)

## 3.4 Overcoming Data Hazards With Dynamic Scheduling(waiting)

A statically scheduled pipeline fetches an instruction and issues it, unless there is a data dependence between an instruction already in the pipeline and the fetched instruction that cannot be hidden with bypassing or forwarding. Then the hazard detection hardware stalls the pipeline.

- **dynamic scheduling**: hardware reorders the instruction execution to reduce the stalls while remaining data flow and exception behavior
advantages:
1. allow code that was compiled with one pipeline in mind to run efficiently on different pipeline
2. enable handling some cases when dependences are unknown at compile time
3. allow the processor to tolerate unpredictable delays such as cache misses
A dynamically scheduled processor cannot change the data flow, it tries to avoid stalling when dependences are present; Static pipeline scheduling by the compiler tries to minimize stalls by separating dependent instructions so that they will not lead to hazards.

### Dynamic Scheduling: The Idea

Limitation in in-order instruction issue and execution: instructions are issued in programorder, and if an instruction is stalled in the pipeline, no later instructions can proceed.

```c
fdiv.d f0,f2,f4
fadd.d f10,f0,f8
fsub.d f12,f8,f14
```

*fsub.d* cannot execute because of stall due to *fadd.d* but is not data-dependent.
Classical five-stage pipeline: both structral and data hazards could be checked during ID. The issue process is separated into two parts: **checking for any structural hazards and waiting for the absence of a data hazard**. Such pipeline use **in-order instruction issue** and does **out-of-order execution**.
OoO execution introduces the possibility of WAR and WAW hazards, which are avoided by **register renaming**.
Dynamically scheduled processors preserve exception behavior by delaying the notification of an associated exception until the processor knows that the instruction should be the next one completed.
**Imprecise excption**: the processor state when an exception is raised does not look exactly as if the instructions were executed sequentially in strict program order. It can occur because of 2 possibilities:
1. The pipeline may have *already completed* instructions that are *later* in program order than the instruction causing the exception.
2. The pipeline may have *not yet completed* some instructions that are *earlier* in program order than the instruction causing the exception.
ID stage is split into 2 stages for OoO execution:
3. Issue—Decode instructions, check for structural hazards.
4. Read operands—Wait until no data hazards, then read operands.
Having multiple instructions in execution at once requires multiple functional units. All instructions pass through the issue stage in order but can be stalled or bypass in *read operands* stage and enter execution OoO.
Scoreboarding allows instructions to execute OoO(sufficient for simple processors). Tomasulo's algorithm(more sophisticated) handles antidependences and output dependences by effectively renaming the registers dynamically. Additionally, it can be extended to handle speculation.

### Dynamic Scheduling Using Tomasulo’s Approach

Key principals: **dynamically determining when an instruction is ready to execute and renaming registers to avoid unnecessary hazards**.

RAW hazards are avoided by executing an instruction only when its operands are available.

WAR and WAW hazards, which arise from name dependences, are eliminated by register renaming

Register renaming is provided by reservation stations, which buffer the operands of instructions waiting to issue and are associated with the functional units. The basic idea is that a reservation station *fetches and buffers an operand as soon as it is available, eliminating the need to get the operand from a register*.

The use of reservation stations, rather than a centralized register file, leads to two other important properties. 1. hazard detection and execution control are distributed; 2. results are passed directly to functional units from the reservation stations.

The bypass is done with a common real bus - common data bus, or (CDB)

![[csarchGraph/Fig3.10.png]]

There are only three steps an instruction goes through.

1. Issue

## 3.5 Dynamic Scheduling: Examples and the Algorithm(TODO)

```c++
1. fld f6,32(x2) 
2. fld f2,44(x3) 
3. fmul.d f0,f2,f4 
4. fsub.d f8,f2,f6 
5. fdiv.d f0,f0,f6 
6. fadd.d f6,f8,f2
```

![[csarchGraph/Fig3.11.png]]

Tomasulo’s scheme offers two major advantages over earlier and simpler schemes: (1) the distribution of the hazard detection logic, and (2) the elimination of stalls for WAW and WAR hazards.

## 3.7 Exploiting ILP Using Multiple Issue and Static Scheduling(TODO)

## 3.8 Exploiting ILP Using Dynamic Scheduling, Multiple Issue, and Speculation(TODO)

## 3.9 Advanced Techniques for Instruction Delivery and Speculation(TODO)

## 3.11 Multithreading: Exploiting Thread-Level Parallelism to Improve Uniprocessor Throughput(TODO)

# Ch4 Data-level Parallelism in Vector, SIMD and GPU Arch
## 4.1 Introduction

## 4.2 Vector Architecture

## 4.3 SIMD Instruction Set Extensions for Multimedia

## 4.4 Graphics Processing Units

# Ch5 Thread-level Parallelsim

# Ch6 Warehouse-Scale Computers to Exploit Request-Level and Data-Level Parallelism

# Ch7 Domain-Specific Architectures

## 7.1 Introduction

- Moore’s Law: growth & demise - slowing its development of new semiconductor processes

architectures targeted million-line programs; Architects treated such code as black boxes, so compilers cannot even bridge the semantic gap between C or C++ and the architecture of GPUs.

- Dennard scaling ended: more transistors switching now means more power and we have replaced the single inefficient processor with multiple efficient cores. we need to lower the energy per operation.

we need to increase the number of arithmetic operations per

instruction from one to hundreds. So we need a drastic change in computer architecture from general-purpose cores to domain-specific architectures (DSAs).

The new normal is that **a computer will consist of standard processors to run conventional large programs such as operating systems along with domainspecific processors that do only a narrow range of tasks**, but they do them extremely well. computers will be much more **heterogeneous**.

Part of the argument: preceding architecture may not be a good match to some domains(caches, out-of-order execution, etc.).

1. architects should expand their areas of expertise. Domain-specific algorithms are almost always for small compute-intensive kernels of larger systems, such as for object recognition or speech understanding.
2. find a target whose demand is large enough to justify allocating dedicated silicon on an SOC or even a custom chip.

One way is to use reconfigurable chips such as FPGAs;

Another DSA challenge is how to port software to it.

## 7.2 Guidelines for DSAs

1. Use **dedicated memories** to minimize the distance over which data is moved.
2. Invest the resources saved from dropping advanced microarchitectural optimizations into more **arithmetic units or bigger memories**.
3. Use the easiest form of parallelism that matches the domain.
4. **Reduce data size and type** to the simplest needed for the domain.
5. Use a **domain-specific programming language** to port code to the DSA.

## 7.3 Example Domain: Deep Neural Networks

AI instead of building artificial intelligence as a large set of logical rules, the focus switched to machine learning from example data as the path to artificial intelligence.

**The Neurons of DNNs**

## 7.4 Google’s Tensor Processing Unit, an Inference Data Center Accelerator

TPU's domain is the inference phase of DNNs

### TPU Origin

deploying GPUs, FPGAs, or custom ASICs in their data centers

### TPU Architecture

TPU: a coprocessor on the PCIe I/O bus, which allows it to be plugged into existing servers. host server sends instructions over the PCIe bus directly to the TPU for it to execute->closer in spirit to an FPU than it is to a GPU, which fetches instructions from its memory.

- host CPU sends TPU instructions over the PCIe bus into an instruction buffer.
- The internal blocks connected-256-byte-wide (2048-bits) paths.
- Matrix Multiply Unit-heart of the TPU, perform 8-bit multiply-and-adds
- The 16-bit products in 32-bit Accumulators
- Activation-nonlinear functions
- weights-an on-chip Weight FIFO that reads from an off-chip 8 GiB DRAM called Weight Memory (for inference, read-only)
- intermediate results - 24 MiB on-chip Unified Buffer, which can serve as inputs to the Matrix Multiply Unit.

A programmable DMA controller transfers data to or from CPU Host memory and the Unified Buffer

### TPU Instruction Set Architecture

over slow PCIe bus, TPU instructions follow the CISC tradition, including a repeat field.

- Read_Host_Memory: CPU host memory-> Unified Buffer
- Read_Weights:Weight Memory->Weight FIFO
- Matrix Multiply/Convolve: causes the Matrix Multiply Unit to perform a matrix-matrix multiply, a vector-matrix multiply, an element-wise matrix multiply, an element-wise vector multiply, or a convolution from the Unified Buffer into the Accumulators.
- Activate performs the nonlinear function
- Write_Host_Memory: Unified Buffer->CPU host memory.

### TPU Microarchitecture

philosophy: keep the Matrix Multiply Unit busy

The plan is to hide the execution of the other instructions by overlapping their execution with the Matrix Multiply instruction.

four general categories of instructions have separate execution hardware (with read and write host memory combined into the same unit)

To increase instruction parallelism further, the Read_Weights instruction follows the decoupled access/execute philosophy in that they can complete after sending their addresses but before the weights are fetched from Weight Memory. The matrix unit has not-ready signals from the Unified Buffer and the Weight FIFO that will cause the matrix unit to stall if their data are not yet available

a TPU instruction can execute for many clock cycles

Because reading SRAM is much more expensive than arithmetic, the Matrix Multiply Unit uses systolic execution to save energy by reducing reads and writes of the Unified Buffer

### TPU Implementation

Unified Buffer a third of the die, and the Matrix Multiply Unit is a quarter

### TPU Software

TensorFlow

the TPU stack is split into a User Space Driver and a Kernel Driver

Kernel Driver: lightweight and handles only memory management and interrupts

User Space Driver: sets up and controls TPU execution, reformats data into TPU order, and translates API calls into TPU instructions and turns them into an application binary.

compiles a model the first time, the second and following evaluations run at full speed

### Improving the TPU

### Summary: How TPU Follows the Guidelines

1. Unified Buffer, Accumulators, weight FIFO
2. dedicated memory and 65,536 8-bit ALUs
3. two-dimensional SIMD parallelism with a systolic organiazation;

   overlapped execution pipeline of instr

4. computes primarily on 8-bit integers
5. TensorFlow(GPUs rely on CUDA and OpenCL)

## 7.5 Microsoft Catapult, a Flexible Data Center Accelerator

Microsoft: Catapult that placed an FPGA on a PCIe bus board into data center servers.

- It had to preserve homogeneity of servers to enable rapid redeployment of
  machines and to avoid making maintenance and scheduling even more complicated,
  even if that notion is a bit at odds with the concept of DSAs.

- It had to scale to applications that might need more resources than could fit into
  a single accelerator without burdening all applications with multiple
  accelerators.

- power-efficient; couldn’t become a dependability problem by being a single point of failure; fit within the available spare space and power in existing servers; could not hurt data center network performance or reliability; improve the cost-performance of the server.

### Catapult Implementation and Architecture

Altera FPGA in Catapult boards includes mechanisms to detect and correct SEUs inside the FPGA and reduces the chances of SEUs by periodically scrubbing the FPGA configuration state.

separate network - reducing the variability of communication performance->a data center network.

### Catapult Software

biggest difference with TPU: HDL

RTL code is divided into the shell and the role,

### CNNs on Catapult

systolic PE

### Search Acceleration on Catapult

## 7.6 Intel Crest, a Data Center Accelerator for Training

A traditional microprocessor manufacturer like Intel taking this bold step of embracing DSAs.

## 7.7 Pixel Visual Core, a Personal Mobile Device Image Processing Unit

**IPUs solve the inverse problem of GPUs**: they analyze and modify an input image in contrast to generating an output image.

We call them IPUs to signal that, as a DSA, they do not need to do everything well because there will also be CPUs (and GPUs) in the system to perform non-input-vision tasks. IPUs rely on stencil computations mentioned above for CNNs.

The innovations of Pixel Visual Core include replacing the one-dimensional SIMD unit of CPUs with a **two-dimensional array of processing elements** (PEs). They provide a two-dimensional shifting network for the PEs that is aware of the two-dimensional spatial relationship between the elements, and a two dimensional version of buffers that reduces accesses to off-chip memory. This novel hardware makes it easy to perform stencil computations that are central to both vision processing and CNN algorithms.

### ISPs, the Hardwired Predecessors of IPUs

Most PMD have multiple cameras for input -> ISPs for **enhancing** input images.

An ISP processes the input image by calculating a series of cascading algorithms via software configurable hardware building blocks, typically organized as a pipeline to minimize memory traffic.

two downsides: the inflexibility of ISP; only for image-enhancing function

### Pixel Visual Core Software

 generalized the typical hardwired pipeline organization of kernels of an ISP into a directed acyclic graph (DAG) of kernels.

### Pixel Visual Core Architecture Philosophy

## 7.8 Cross-Cutting Issues

### Heterogeneity and System on a Chip (SOC)

- The easy way to incorporate DSAs into a system is over the I/O bus, (data center accelerators). To avoid fetching memory operands over the slow I/O bus, these accelerators have local DRAM.

Amdahl’s Law: the performance of an accelerator is limited by the frequency of shipping data between the host memory and the accelerator memory

-> applications that would benefit from the host CPU and the accelerators to be integrated into the same SOC(Pixel Visual Core and eventually the Intel Crest)

-> IP block must be scalable in area, energy, and performance.

### An Open Instruction Set

One challenge for designers of DSAs is determining how to collaborate with a CPU: choose CPU instruction set; design your own custom RISC processor.

RISC-V-a viable free and open instruction set with plenty of opcode space reserved

## 7.9 Putting It All Together: CPUs Versus GPUs Versus

DNN Accelerators

# Appendix A Instruction Set Principles

## A.9 Putting It All Together: The RISC-V Architecture

RISC-V provides a both a 32-bit and a 64-bit instruction set, as well as a variety of extensions for features like floating point. Like its RISC predecessors, RISC-V emphasizes

- A simple load-store instruction set.
- Design for pipelining efficiency, including a fixed instruction set encoding.
- Efficiency as a compiler target.

### RISC-V Instruction Set Organization

The RISC-V IS is organized as 3 base IS(32/63-bit and optional extensions): e.g. RV64IMAFD(RVG)

![[csarch.assets/Fig A.22.png]]

### Registers for RISC-V

RV64G: 32 64-bit GPRs(x0, …, x31) also integer registers; 32 FPRs(f0, …, f31) which holds 32 single-precision or double-precision values. x0 is always 0.

### Data Types for RISC-V

8-bit bytes, 16-bit half words, 32-bit words, 64-bit double words for integer data and 32-bit single precision and 64-bit double precision for floating point

They are loaded with either 0/sign bit replicated to fill 64 bits.

### Addressing Modes for RISC-V Data Transfers

The only data addressing modes are immediate and displacement, both with 12-bit fields.

# Appendix B Review of Memory Hierarchy

# Appendix F Interconnection Networks

## F.3 Connecting More than Two Devices

## F.4 Network Topology

## F.5 Network Routing, Arbitration, and Switching

### Arbitration

The arbitration algorithm determines when requested network paths are available for packets.

# 改简要的概括

## Chapter1: Fundamentals of Quantitative Design and Analysis

### 1.1 Introduction

在第一台电子计算机被发明以来的 70 年间，计算机技术取得了惊人的进步。更好的计算机体系结构和和更先进的集成电路技术共同推动着微处理器的进步。汇编语言的淘汰和平台无关的操作系统为新的计算机体系结构铺平了道路，促成了 RISC 的诞生。微处理器的发展使用户获得了更多的算力；引入了新的计算机，包括 PC、手机等；取代了其他电子电路设计；催生了软件工业。

但硬件的文艺复兴可能面临终结，因为两个重要的定律不复存在。一是 [Dennard定律](https://zhida.zhihu.com/search?content_id=222471267&content_type=Article&match_order=1&q=Dennard%E5%AE%9A%E5%BE%8B&zhida_source=entity)：得益于更小的晶体管尺寸，相同硅片面积下晶体管数量的增加不会引起功耗的增加。Dennard 定律在 2004 年终结，因为电压不再随着晶体管尺寸的缩小而降低。二是 [Moore定律](https://zhida.zhihu.com/search?content_id=222471267&content_type=Article&match_order=1&q=Moore%E5%AE%9A%E5%BE%8B&zhida_source=entity)：每两年芯片上的晶体管数量会翻倍。Moore 定律将于近期终结。领域特定加速可能有机会破局。

### 1.2 Classes of Computers

**Internet of Things/Embedded Computers**

IoT 设备从真实世界中收集有用的数据并与真实世界进行互动，催生了很多智能应用。在嵌入式领域，价格是关键因素。

**Personal Mobile Device**

PMD 中价格和功耗至关重要，其应用多是面向网络的或是基于媒体的。响应性和可预测性是媒体应用的关键，所以实时处理性能很重要。

**Desktop Computing**

个人电脑市场目前超过一半被笔记本电脑占据，性价比在该领域是关键。

**Servers**

服务器领域关心可靠性和可扩展性，将吞吐量作为性能指标。

**Clusters/Warehouse-Scala Computers**

SaaS 应用的增长催生了机群和仓库级计算机，其注重功耗、性价比和可靠性。

**Classes of Parallelism and Parallel Architecture**

软件中有两种并行性：数据级并行和任务级并行。计算机通过四种方式挖掘软件中的并行性：指令级并行；向量架构、GPU 和多媒体指令；线程级并行；任务级并行。Flynn 将计算机分为四类：SISD，SIMD，MISD 和 MIMD。

### 1.3 Defining Computer Architecture

早些年将计算机体系结构定义为指令集体系结构，这是片面的。

**Instruction Set Architecture: The Myopic View of Computer Architecture**

ISA 是软硬件的界面，本书中采用 [RISC-V](https://zhida.zhihu.com/search?content_id=222471267&content_type=Article&match_order=1&q=RISC-V&zhida_source=entity) 作为示例：

- ISA 的分类：寄存器 - 内存型和 load-store 型。
- [内存索引](https://zhida.zhihu.com/search?content_id=222471267&content_type=Article&match_order=1&q=%E5%86%85%E5%AD%98%E7%B4%A2%E5%BC%95&zhida_source=entity)：按字节索引。
- 索引方式：寄存器、立即数、偏移量等。
- 操作数：字节、半字、字、双字和 IEEE 754 浮点数。
- 操作类型：数据传输、算术运算、控制流和浮点运算。
- 控制流指令：条件跳转、无条件跳转、过程调用和返回。
- 编码方式：定长和变长。

**Genuine Computer Architecture: Designing the Organization and Hardware to Meet Goals and Functional Requirements**

组织架构或微架构指处理器的设计，包括内存系统、处理器核设计等内容，例如同一 ISA 的不同处理器。硬件架构指计算机的规格，例如同一系列的不同处理器。

### 1.4 Trends in Technology

五个实现技术如今正以惊人的速度改变着半导体行业：

1. 集成电路技术
2. 半导体 DRAM
3. 半导体闪存
4. 磁盘技术
5. 网络技术

**Performance Trends: Bandwidth Over Latency**

带宽的增长速度比延迟的降低速度更快

**Scaling of Transistor Performance and Wires**

晶体管密度随晶体管尺寸的缩小按平方增加，晶体管性能随晶体管尺寸的缩小按线性增加，但连线延迟不会随之降低，导致连线延迟如今成为棘手的问题。

### 1.5 Trends in Power and Energy in Integrated Circuits

**Power and Energy: A System Perspective**

对于系统设计者而言，有三个问题值得关注：处理器的最高功耗是多少？处理器的正常功耗（TDP）是多少？处理器的能效比（以消耗的能量计算）是多少？

**Energy and Power Within a Microprocessor**

[动态功耗](https://zhida.zhihu.com/search?content_id=222471267&content_type=Article&match_order=1&q=%E5%8A%A8%E6%80%81%E5%8A%9F%E8%80%97&zhida_source=entity)：

Energydynamic∝1/2×Capacitive load×Voltage2

Powerdynamic∝1/2×Capacitive load×Voltage2×Frequency switched

现代处理器提供多种方式来提升能效比：

- [暗硅](https://zhida.zhihu.com/search?content_id=222471267&content_type=Article&match_order=1&q=%E6%9A%97%E7%A1%85&zhida_source=entity)
- DVFS
- 低功耗模式
- 超频

随着漏电电流的增加，[静态功耗](https://zhida.zhihu.com/search?content_id=222471267&content_type=Article&match_order=1&q=%E9%9D%99%E6%80%81%E5%8A%9F%E8%80%97&zhida_source=entity) 也增加：

Powerstatic∝Currentstatic×Voltage

**The Shift in Computer Architecture Because of Limits of Energy**

能效比的追求使得 DSA 成为未来的希望。

### 1.6 Trends in Cost

**The Impact of Time, Volume, and Commoditization**

电子元件的价格会随着时间下降，因为随着学习曲线，制造的良率会提升。生产量的提升也会造成价格的降低，因为更快的学习曲线，更多的谈判空间和更少的均摊成本。商品化意味着激烈的竞争，使得价格进一步降低。

**Cost of an Integrated Circuit**

集成电路的成本包括晶片成本，测试成本和封装成本。晶片成本与晶圆成本，晶片面积，良率有关。晶片面积的缩小可以有效提高良率，冗余的方法也可以提升良率。对于设计人员来说，只需要关注芯片面积，成本随芯片面积的增加呈平方增加。另外，对于低产量的芯片，掩膜成本也是不可忽视的一部分。

**Cost Versus Price**

商品化导致利润空间缩小。

**Cost of Manufacturing Versus Cost of Operation**

WSC 中的运营成本很高。

### 1.7 Dependability

Module availability=MTTFMTTF+MTTR

冗余可以有效提升系统的可靠性。

### 1.8 Measuring, Reporting, and Summarizing Performance

不同人对性能的定义是不同的，我们坚持性能是运行真实程序的时间。时间可能是真实流逝的时间，也可能是 CPU 时间。

**Benchmarks**

运行简化的程序来测试计算机的性能是不准确的。编译选项、是否允许修改源码也会影响运行基准测试程序的性能。基准测试程序套件包含许多不同的测试程序，避免将鸡蛋放在一个篮子里。SPEC 推出了很多成功的基准测试程序套件。

个人电脑的测试程序分为处理器密集型和图像密集型。SPEC89 是测试处理器性能的程序，已进行了 6 次迭代。最新的 [SPEC2017](https://zhida.zhihu.com/search?content_id=222471267&content_type=Article&match_order=1&q=SPEC2017&zhida_source=entity) 包含 10 个整数测试程序和 17 个浮点测试程序，但其可能不能量化 21 世纪的电脑性能，因为其甚至未使用动态链接。

服务器有不同的用途，也有不同的性能指标。最简单的方式是运行多个 SPEC2017 的拷贝来测试吞吐率，称为 SPECrate。服务器通常还有大量的 IO 和文件操作，SPEC 也提供测试文件系统、Java 性能的测试程序。[TPC](https://zhida.zhihu.com/search?content_id=222471267&content_type=Article&match_order=1&q=TPC&zhida_source=entity) 是一套测试交易处理性能的测试程序。

**Reporting Performance Results**

实验要具备可重复性。

**Summarizing Performance Results**

将待测电脑运行 SPEC 程序的时间除以参考电脑运行 SPEC 程序的时间，得到的比值称为 SPECRatio。比较两台电脑的性能时，参考电脑的选择不会有任何影响。SPEC 分数的计算使用几何平均。

### 1.9 Quantitative Principles of Computer Design

**Take Advantage of Parallelism**

挖掘多个层级的并行性。

**Principle of Locality**

时间相关性和空间相关性。

**Focus on the Common Case**

优先考虑常见的情况。

**Amdahl's Law**

Speedupoverall=1(1−Fractionenhanced+FractionenhancedSpeedupenhanced)

**The Processor Performance Equation**

CPU Time=Instruction count×Cycles per instruction×Clock cycle time

其中，时钟周期和硬件工艺、组成结构有关，CPI 和组成结构、ISA 有关，指令数和 ISA、编译器有关。

### 1.10 Putting It All Together: Performance, Price, and Power

比较了三个戴尔服务器的各项指标。

## Chapter2: Memory Hierarchy Design

### 2.1 Introduction

如今的存储器层次结构一般包含“寄存器 -L1$-L2$-L3$-内存-硬盘/闪存”几部分，大部分情况下下层的存储器的内容是上层的存储器的内容的超集。尽管近年来单核性能的停滞不前使得处理器和内存之间的“剪刀差”不再扩大，但是多核的趋势其实对内存造成了更大的压力。如今的处理器通过流水化的Cache，更多的Cache层次等方式来解决内存访问的瓶颈，预计使用嵌入式DRAM实现的L4$ 将会是未来的一个趋势。传统意义上，设计者只需关注 Cache 的性能，即“内存平均访问时间”，也就是 Cache 的命中延迟、命中率和缺失惩罚，但如今更大的 Cache 造成了可观的功耗，在可移动设备的 CPU 中甚至达到 25%-50%，功耗也成为设计者的重要考虑因素之一。

**Basics of Memory Hierarchies: A Quick Review**

一个 Cache 行由若干个字组成，其包含一个 `tag` 域。最受欢迎的 Cache 组织结构是 [组相联结构](https://zhida.zhihu.com/search?content_id=222471267&content_type=Article&match_order=1&q=%E7%BB%84%E7%9B%B8%E8%81%94%E7%BB%93%E6%9E%84&zhida_source=entity)，直接映射和全相联结构是其特殊情况。Cache 的写策略有写回和写穿透，两者都可以用写缓存来加速。

缺失率是 Cache 性能的重量衡量指标，根据 3C 模型将 Cache 缺失分为三类：强制缺失、容量缺失和冲突缺失。随着超线程和多核技术的广泛应用，我们也可以增加第四个 C：一致性缺失。

用缺失率来衡量 Cache 性能具有误导性，设计者也会使用“每千条指令缺失数”和“内存平均访问时间”作为衡量指标。但由于动态调度和超线程，这些指标也不能和真正的程序运行时间相关联。

以下六种简单的方法可以优化 Cache，但这六种方法都有一定的副作用。

1. 更大的 Cache 行以减小缺失率
2. 更大的 Cache 以减小缺失率
3. 更大的组相联度以减小缺失率
4. 更多的 Cache 层次以减小缺失惩罚
5. 读缺失的优先度高于写缺失以减小缺失惩罚
6. 避免虚实地址翻译后再索引 Cache 以减小命中延迟

### 2.2 Memory Technology and Optimizations

随着突发传输的广泛应用，内存延迟用两个维度来描述：访问时间和周期时间。访问时间指发送读请求到读数据返回的时间；周期时间指两个不相关的读请求之间的最小间隔。所有的计算机都用 DRAM 组成内存，用 SRAM 组成 Cache。

**SRAM Technology**

SRAM 的访问时间和周期时间基本相同，它一般使用 6 个晶体管来存储一个 bit。Cache 的访问时间和 Cache 行数量呈正相关，而功耗和 Cache 大小（静态功耗）和 Cache 行数量（动态功耗）相关。

**DRAM Technology**

访问 DRAM 的地址被分为行地址和列地址，行地址在 RAS 期间被使用，列地址在 CAS 期间被使用。DRAM 在读取数据时读取线必须经过预充电，读取的一行数据被放在行缓冲中，由于读取的过程损坏了行中的数据，所以如果行缓冲中的数据要被替换，需要先将旧的数据写回到对应的行中而不是直接丢弃。由于漏电，DRAM 还需要定期刷新每一行，即将它读出然后重新写入，设计者一般将刷新所用的时间控制在 5% 以内。

**Improving Memory Performance Inside a DRAM Chip: [SDRAMs](https://zhida.zhihu.com/search?content_id=222471267&content_type=Article&match_order=1&q=SDRAMs&zhida_source=entity)**

最早的 DRAM 是异步的，在 90 年代中期设计者加入了时钟，发明了同步 DRAM（SDRAM），还加入了突发传输模式。内存带宽需求的增长使得 DRAM 单次传输的比特数逐渐增长，从 4bits 一直到 16bits，在 00 年代还引入了 DDR 技术，在时钟的上升沿和下降沿都传输数据。最后，SDRAM 还引入了 bank 来控制功耗和减少访问时间。每个 bank 都有单独的行缓冲，可以并行访问不同 bank。

电压和 SDRAM 的功耗紧密相关，从 DDR1 到 DDR4，内存电压不断下降。近期的 SDRAM 还支持 power-down 模式，屏蔽了刷新以外的所有操作。

**Graphics Data RAMs**

[GDDR](https://zhida.zhihu.com/search?content_id=222471267&content_type=Article&match_order=1&q=GDDR&zhida_source=entity) 使用更宽的数据总线和更快的时钟频率来进一步提高带宽，以满足 GPU 的需求。

**Packaging Innovation: Stacked or Embedded DRAMs**

将 DRAM 和处理器一起封装可以进一步降低延迟，增加带宽，这种内存称为 [HBM](https://zhida.zhihu.com/search?content_id=222471267&content_type=Article&match_order=1&q=HBM&zhida_source=entity)。DRAM 可以直接堆叠在处理器上方（3D），也可以和处理器通过衬底相连（2.5D）。

**Flash Memory**

闪存是一种 EEPROM，其与 DRAM 最大的不同在于：

1. 读取闪存是顺序的，一次读取一整页内容，其读取速度在 DRAM 和硬盘之间。
2. 闪存必须先被擦除才能重新写入，擦除以块为单位，其写入速度在 DRAM 和硬盘之间，但比读取速度优势小得多。
3. 闪存是非易失性的，待机功耗很小。
4. 闪存的块只能被写入有限次，所以需要平衡写入负载。
5. 闪存的价格在 SDRAM 和硬盘之间。

DRAM 和闪存芯片都有冗余的块用于替换损坏的块。

**Phase-Change Memory Technology**

PCM 已经被研究了数十年，它使用一个加热装置让介质在晶体和非晶体之间切换，读取数据通过检测两种状态电阻的不同来实现。

**Enhancing Dependability in Memory Systems**

更大的 Cache 和内存使得生产过程中和使用过程中的错误发生的更频繁。电路的变化造成的错误称为 hard errors 或 permanent faults，其可通过重映射空闲的行来解决。存储数据的随机变化称为 soft errors 或 transient faults，其可以通过 ECC 解决，近年来还引入了 Chipkill 技术来进一步增加可靠性，其思想与 RAID 类似。

### 2.3 Ten Advanced Optimizations of Cache Performance

我们可以从以下五个维度来提升 Cache 性能：

1. 减少命中延迟
2. 增加 Cache 带宽
3. 减少缺失惩罚
4. 增加命中率
5. 通过并行来减少缺失惩罚或缺失率

**First Optimization: Small and Simple First-Level Caches to Reduce Hit Time and Power**

高时钟频率和功耗限制的压力使得一级 Cache 的大小一般不大。目前的处理器的一级 Cache 容量基本不再增长，设计者转而追求更高的组相联度。Cache 命中的关键路径分为三部分：使用地址索引 Tag SRAM，Tag 对比和数据选择。CACTI 可以量化 Cache 的命中延迟，将其表示为一个 Cache 大小、组相联度，读写端口数等参数的函数。功耗是 Cache 设计的重要因素，除了组相联度外，更少的 Cache 行（更大的 Cache 行）可以减少功耗，但这也增加了缺失率。另一个方法是将 Cache 分为 bank 实现。

如今的 Cache 设计注重更高的组相联度有以下三个原因：如今的处理器会花费至少 2 个周期来访问 Cache，使得命中延迟不会是关键路径；VIPT 的 Cache 容量受到限制，增大组相联度是增加 Cache 容量的最有效的方式；超线程技术使得冲突缺失变得更加频繁。

**Second Optimization: Way Prediction to Reduce Hit Time**

路预测指在每个 Cache 行中使用额外的 bit 来预测下一个 Cache 访问会命中哪一路，直接输出预测的路的数据，预测失败会引入一周期的惩罚。模拟器结果表明对于 2 路组相联 Cache，预测正确率在 90% 以上；对于 4 路组相联 Cache，预测正确率在 80% 以上；ICache 的预测正确率比 DCache 高。如果只访问预测正确的 Cache 行的数据，虽然会造成预测失败时更大的惩罚，但可以显著降低 Cache 的功耗，在低功耗领域可以得到应用。

**Third Optimization: Pipelined Access and Multibanked Caches to Increase Bandwidth**

流水化和 Multibank 一般应用在 L1 Cache 上，因为这可以增加超标量乱序流水线的指令吞吐率。Multibank 也用于 L2 和 L3 Cache 以减小功耗。ICache 的流水化增加了分支预测错误的惩罚，DCache 的流水化增加了 Load-to-Use，但高时钟频率使现代处理器都需要多级流水访问 Cache。Multibank 使得对于 Cache 的访问可以交叠，允许 Cache 在一拍内处理多个读写请求。

**Fourth Optimization: Nonblocking Caches ot Increase Cache Bandwidth**

"hit under miss" 和 "miss under miss" 的非阻塞 Cache 可以显著降低超标量处理器的缺失惩罚，但优化的幅度难以量化。一般来说，乱序处理器可以弥补 L1 数据 Cache 缺失、L2 数据 Cache 命中造成的惩罚，但是不能弥补更低层次的 Cache 缺失造成的惩罚。

为了实现非阻塞 Cache，我们需要考察两个问题：一是仲裁命中和缺失之间的冲突，二是记录正在进行的缺失。现代处理器使用 MSHR 来记录一个进行中的缺失，包括数据将被写回哪个 Cache 行，哪些指令正在等待该缺失返回的信息。在多核处理器中，非原子的缺失和一致性请求可能会引入死锁。

**Fifth Optimization: Critical Word First and Early Restart to Reduce Miss Penalty**

- 关键字优先：先发出缺失的字的读请求，再处理 Cache 行中的其他字。
- 提前重启：按原来的方式发出读请求，但在缺失的字返回时马上返回给流水线。

该技术取得的性能提升取决于 Cache 行的大小和程序的访存行为。由于空间相关性，可能马上会收到第二个对该 Cache 行的访问，使得关键字优先技术不能提高 Cache 性能。

**Sixth Optimization: Merging Write Buffer to Reduce Miss Penalty**

写缺失会先检查写缓存中已有的 Cache 行，如果地址相同则将写数据合并。但 I/O 地址不能使用该技术。

**Seventh Optimization: Compiler Optimizations to Reduce Miss Rate**

按行优先/列优先数组在内存中的顺序访问数组元素。如算法不允许如此（例如矩阵乘法），也可以用分块的方式减少 Cache 缺失。

**Eighth Optimization: Hardware Prefetching of Instructions and Data to Reduce Miss Penalty or Miss Rate**

在 Cache 缺失时可以取出当前缺失的 Cache 行和下一个 Cache 行，将下一个 Cache 行放入一个 Buffer 中。一般指令 Cache 只需一行 Buffer，数据 Cache 需要多行 Buffer。Intel 尝试过更为激进的预取策略，但性能提升十分有限。

**Ninth Optimization: Compiler-Controlled Prefetching to Reduce Miss Penalty or Miss Rate**

软件预取可分为预取至寄存器/预取至 Cache，可产生例外/不可产生例外，主流的做法是提供不可产生例外的预取至 Cache 的指令。因为预取指令的额外开销，编译器必须判断哪些访存指令容易造成 Cache 缺失，也要考虑预取的数据何时能够到达，这是较为困难的。

**Tenth Optimization: Using HBM to Extend the Memory Hierarchy**

HBM 的大小可达到 128MiB 至 1GiB，必须要足够大的 Cache 行来减少存储 Tag 的开销，但这导致两个问题：一个 Cache 行内的数据可能不都是有用的，导致浪费；Cache 缺失会变得更多。一个解决方案是子 Cache 行——用多位有效位来指示一个 Cache 行，允许其部分有效，但这没有解决第二个问题。

若是将 Tag 也存在 HBM 中则不用担心 Tag 的开销问题。我们可以将每一路的 Tag 和数据存在 SDRAM 的同一行内，首先打开行，读取 Tag 判断命中哪一路，然后发送对应的列选择信号，这可以避免一次 Cache 命中读取多个 SDRAM 的行。也可以直接做成直接映射的 Cache，一次读取一整行，包括 Tag 和数据，实验表明这种实现的性能最好。另一个问题是 Cache 缺失的惩罚过大，我们可以使用缺失预测器来解决这一问题。

**Cache Optimization Summry**

从命中延迟、带宽、缺失惩罚、缺失率、功耗和复杂度几个方面总结了上述技术。

### 2.4 Virtual Memory and Virtual Machines

本节将侧重于保护和隐私方面的内容。

**Protection via Virtual Memory**

包括 TLB 的页式内存管理是隔离不同进程的基本机制。为了隔离用户和内核进程，体系结构必须：

- 提供至少两种模式区分内核和用户
- 提供用户只读，内核可读写的接口
- 提供在用户态和内核态之间切换的机制
- 提供对内存访问的限制

一般内存保护通过对每一页增加权限限制来实现，包括读权限、写权限和执行权限。为了避免一次内存访问需要两倍的时间：一次读取页表项和一次读取数据，我们可以利用相关性，使用 TLB 作为页表的缓存。TLB 由操作系统进行管理，但操作系统不可避免的也有 bug，所以我们需要虚拟机。

**Protection via Virtual Machines**

虚拟机在近些年重新获得热度，主要因为：

- 现代系统中隔离和安全变得更重要
- 标准的操作系统在安全和可靠性方面表现不佳
- 在数据中心和云上许多不同的用户共用一台计算机
- 处理器速度的大幅提高使得虚拟机的额外开销变得可以接受

使用虚拟机系统，多个操作系统可以共享硬件资源。提供虚拟机系统的软件称为 VMM，硬件平台称为主机，它被所有客户虚拟机共享。VMM 的规模比传统的操作系统更小，通常只有万行代码。一般的，用户态指令在虚拟机上运行没有额外开销，IO 指令和特权指令在虚拟机上运行的额外开销较高。

除了安全性，虚拟机在以下两个方面也有优势：

1. 管理软件：虚拟机提供了完整的软件栈的抽象
2. 管理硬件：不同的软件栈共享一套硬件

**Requirements of a Virtual Machine Monitor**

基本的两个要求是：

1. 客户软件在虚拟机上运行时和在原生硬件上运行时应该表现一致
2. 客户软件不能直接改变真实的系统资源

**Instruction Set Architecture Support for Virtual Machines**

可以直接运行虚拟机的体系结构称为可虚拟化的。一般的，VMM 可以捕获客户操作系统执行的特权指令并作出相应的支持。若硬件支持三个特权等级，客户操作系统执行特定特权指令（那些不影响其他客户操作系统的指令）时可以不经过 VMM。

**Impact of Virtual Machines on Virtual Memory and I/O**

可将内存分为三级：虚拟内存 - 物理内存 - 机器内存。另一种实现是引入影子页表（shadow page table），影子页表捕获用户操作系统对其页表的修改，直接将虚拟内存映射到机器内存。VMM 保存了所有客户操作系统的 TLB 内容。IO 设备之间的映射可以是分时的（网口），也可以是分块的（硬盘）。

**Extending the Instruction Set for Efficient Virtualization and Better Security**

近年来的指令集的虚拟化扩展主要关注页表和 TLB 处理的性能和 IO 性能。虚拟内存管理方面，避免了不必要的 TLB 刷新，采用了嵌入页表机制（nested page table）。IO 方面，客户操作系统可以使用 DMA，也可以直接处理中断。

安全方面，VMM 提供了不同客户操作系统之间的隔离。Intel 提出了 SGX 扩展，将代码和数据进行加密。

**An Example VMM: The Xen Virtual Machine**

通过修改操作系统的少量代码，VMM 可以实现的更为简单高效。

### 2.5 Cross-Cutting Issues: The Design of Memory Hierarchy

**Protection, Virtualization, and Instruction Set Architecture**

保护是体系结构和操作系统的共同努力，IBM 主流的硬件采取以下措施提高虚拟机的性能：

1. 减少处理器虚拟化的开销
2. 减少因虚拟化造成的中断的额外开销
3. 通过直接将中断传递给正确的虚拟机来减少中断的开销

**Autonomous Instruction Fetch Units**

乱序处理器一般将取指操作解耦合。取指单元一般从指令 Cache 中取出整个 Cache 行。

**Speculation and Memory Access**

猜测执行可能造成不必要的内存访问，造成 Cache 缺失率虚高。

**Special Instruction Caches**

现代处理器可以使用一个较小的 Cache 存储译码后的微操作。

**Coherency of Cached Data**

多核和 IO 都会造成一致性问题。

### 2.6 Putting It All Together: Memory Hierarchies in the ARM Cortex-A53 and Intel Core I7 6700

**The ARM Cortex-A53**

Cortex-A53 是一个 IP 核，IP 核是可移动设备领域的主流形式，分为硬核和软核，前者对特定代工厂做了优化，是一个黑盒，后者可以为不同代工厂编译并做少量修改。Cortex-A53 每拍可以发射两条指令，支持两级 TLB 和两级 Cache。

**The Intel Core i7 6700**

i7 是一个四核处理器，每个核每拍可以执行 4 条指令，使用多发射、动态调度、超线程技术，16 级流水线。它支持三个内存通道，峰值内存带宽超过 25GB/s。它支持 2 级 TLB，一级 Cache 使用 VIPT 索引，二级和三级 Cache 使用 PIPT 索引。评估该处理器的访存子系统性能是比较困难的，因为非写分配的一级数据 Cache 和激进的预取策略，但数据表明激进的预取策略对其性能有较大帮助。

## Chapter3: Instruction-Level Parallelism and Its Exploitation

### 3.1 Instruction-Level Parallelism: Concepts and Challenges

本章将探讨数据冲突和控制冲突，探讨软硬件增加指令级并行度（ILP）的方式。开发 ILP 有两种主要的途径：一是硬件动态的，二是软件在编译时静态的。这两种方式开发的 ILP 都是有上限的。

**What Is Instruction-Level Parallelism?**

由于 RISC 程序的基本块都很小，而且基本块内的指令很可能有数据相关，我们能在基本块内开发的 ILP 非常有限，必须要在多个基本块间开发 ILP。增加 ILP 最简单的方法是在循环的多次执行中寻找并行性。

**Data Dependences and Hazards**

指令 j 对指令 i 数据相关，如果任意一条成立：

- 指令 j 会用到指令 i 产生的结果
- 指令 j 对指令 k 数据相关，指令 k 对指令 i 数据相关

数据相关的指令不能（完全地）交叠执行。数据相关是程序的固有属性，它限制了 ILP 的开发。数据相关可以是寄存器间的，也可以是内存间的，后者会造成更大的挑战。

名字相关指两条指令使用了相同的寄存器或内存地址，但它们之间没有数据之间的关联。对于指令 i 和指令 j 有两种名字相关，假设指令 i 的程序序在前：

- 反相关指指令 j 写了某一寄存器或内存地址，而指令 i 要读取它
- 输出相关指指令 i 和指令 j 写了同一寄存器或内存地址

当数据相关或名字相关的指令在程序中足够靠近时，它们就会发生数据冲突。我们在开发并行性的时候要避免数据冲突，保持程序序，即保持程序的输出顺序不变。我们将数据冲突分为三种：RAW，WAW 和 WAR。

**Control Dependences**

控制相关提出了两个限制：

1. 一条对分支指令控制相关的指令不能移到该分支指令之前
2. 一条不对分支指令控制相关的指令不能移到该分支指令之后

有两个关键的属性要求我们遵守数据相关和控制相关：例外和数据流。如果执行的指令不影响程序的正确性，我们也可以违反控制相关。

### 3.2 Basic Compiler Techniques for Exposing ILP

编译优化来开发 ILP 对静态发射和静态调度的处理器来说是非常关键的。

**Basic Pipeline Scheduling and Loop Unrolling**

循环展开可以减少循环的额外开销，还有利于指令的调度。

**Summary of Loop Unrolling and Scheduling**

在实践中，进行循环展开时需要做出以下决策：

- 循环间是不相关的
- 展开后使用不同寄存器来消除名字相关
- 消除额外的分支指令并修改循环代码
- 检查循环间的 Load 和 Store 指令的地址以确保它们不相关
- 调度代码以减少阻塞

循环展开有以下限制：

- 边际效用
- 代码大小
- 编译器限制（寄存器数量限制）

### 3.3 Reducing Branch Costs With Advanced Branch Prediction

**Correlating Branch Predictor**

在进行分支预测时不仅需要考察本地历史，还要考察全局历史。一个 (m,n) 预测器使用最近 m 个分支的信息索引 $2^m$ 个预测表，每个预测表有若干项（用分支地址低位索引），每项用 n 位来预测一个分支指令。我们可以连接（或哈希）分支地址的低位和全局历史来索引整个分支预测表。GShare 预测器使用分支地址的低位和全局历史的异或来索引，它的效果非常好，成为了分支预测器的 baseline。

**Tournament Predictors: Adaptively Combining Local and Global Predictors**

全局预测器通过全局历史进行索引，本地预测器使用分支地址进行索引，一个选择器选择它们的预测结果之一作为最终结果。一般的，每个循环都有一个 2 位的计数器来作为选择器。本地预测器可以做成 2 层，第一层使用分支地址索引，记录分支的本地历史信息，用分支的本地历史信息去索引第二层预测表，得到更准确的结果。

**Tagged Hybrid Predictors**

TAGE 预测器使用多个用不同全局历史长度来索引的预测器，首先通过 PC 低位和 i 位全局历史的哈希来索引预测表，然后通过 PC 低位和 i 位全局历史的（另一个）哈希来对比 tag。TAGE 的预测结果由 tag 匹配的全局历史长度最长的预测器提供，第 0 级预测器作为后备预测器没有 tag 域。预测表还有 use 域来记录表项是否最近被使用。当预测表的容量不断增加，预测表的初始化对性能也有较大影响。

**The Evolution of the Intel Core i7 Branch Predictor**

Core i7 920 使用了两级预测器，每一级包含三种不同的预测器：本地历史，全局历史和循环退出，通过竞争选择预测结果。Core i7 6700 可能采用了 TAGE 预测器。

### 3.4 Overcoming Data Hazards With Dynamic Scheduling

动态调度的优点有：

- 不用为每一种微架构编译一个二进制文件
- 可以解决编译时无法确定的依赖关系
- 解决无法预测的延迟，例如 Cache 缺失

**Dynamic Scheduling: The Idea**

动态调度时，指令仍是按序发射的，但是当它的源操作数准备好后就可以马上执行，即指令是乱序执行，乱序结束的。乱序执行引入了 WAR 和 WAW 冲突，乱序结束引入了非精确例外的问题。

为了允许乱序执行，我们将 ID 级拆分为两级：

- 发射：译码，检查结构相关
- 读操作数：等待数据冲突解决，读取源操作数

为发挥动态调度的优势，需要多个流水的功能部件以允许多条指令同时执行。**计分板**是最初的动态调度算法，此处我们重点讨论**Tomasulo 算法**，它能通过重命名寄存器解决输出相关和反相关，也可以引入猜测执行。

**Dynamic Scheduling Using Tomasulo's Approach**

Tomasulo 算法的基本原理是动态判断指令何时准备好执行和使用重命名寄存器来避免不必要的冲突。寄存器重命名会重命名所有目的寄存器，解决 WAW 和 WAR 冲突。Tomasulo 算法的实现中，寄存器重命名通过保留站实现，保留站可以缓存指令的源操作数。指令发射时，需要它的结果的指令的源操作数被标记为它的保留站号，该指令执行完成后将结果写入这些指令的源操作数域中。

在 Tomasulo 算法中，冲突检测和执行控制是分开的，每个功能部件的保留站决定其中的指令是否可以执行；操作数由保留站直接提供给功能部件而不需要经过寄存器堆，意味着需要结果总线将功能部件的输出载入到保留站的源操作数域中。

Tomasulo 算法主要包含三步：

- 发射（分发）：将指令从 IQ 中发送到保留站中（如果有空位），如果源操作数已准备好，则读取寄存器，如果未准备好，则记录相应的保留站号
- 执行：将准备好的指令发送到功能部件
- 写回：将结果发送到结果总线上

Tomasulo 算法导致了相关的指令之间至少有一拍的延迟。

### 3.5 Dynamic Scheduling: Examples and the Algorithm

**Tomasulo's Algorithm: The Details**

算法伪代码。

**Tomasulo's Algorithm: A Loop-Based Example**

如果 Load 和 Store 的地址不同，它们可以安全地乱序执行，但地址相同的 Load 和 Store 之间也会有 RAW，WAR，WAW 冲突。假设 Load 和 Store 的地址按照程序序被计算出来，那么 Load 需要检查 Store Buffer 中所有的地址，Store 需要检查 Load Buffer 和 Store Buffer 中所有的地址。

Tomasulo 算法在 360/91 之后许多年都无人使用，但随着多发射的流行，它被广泛采用，因为：

- 需要减小 Cache 缺失的影响
- 处理器有更多的晶体管
- 不需要编译器对特定流水线结构优化也可以取得高性能

### 3.6 Hardware-Based Speculation

硬件的猜测执行有三个重要思想：

1. 动态的分支预测来选择执行哪些指令
2. 猜测地允许指令在控制相关解决前执行
3. 在多个基本块间动态调度

为了支持猜测执行，我们需要区分指令产生（可前递的）结果和指令真正完成。在指令不再处于猜测状态时，我们才可以让它产生不可撤销的更新。真正地更新寄存器或内存的阶段称为指令提交。为了在指令产生结果和指令提交之间暂存指令的结果，我们引入了 ROB，ROB 可以作为重命名寄存器使用。

指令的执行有四个阶段：

1. 发射（分发）：从 IQ 中获取指令，将其发送到保留站和 ROB 中
2. 执行：等待操作数准备好后发送到功能部件
3. 写回：将结果发送到结果总线上
4. 提交：指令位于 ROB 头且结果已准备好时即可提交，若是分支预测错则取消后续指令

引入 ROB 还可以支持精确例外。为了提高流水线的效率，分支指令预测错会在执行时就清除后续指令，因为分支指令预测错比例外出现的更多。为了维持访存指令的顺序性，我们要求：

- 不允许 Load 指令开始访存，如果有效的 ROB 指令中有相同地址的 Store 指令
- 维持计算访存指令地址的操作是按程序序的

### 3.7 Exploiting ILP Using Multiple Issue and Static Scheduling

多发射处理器主要有三种类型：

1. 静态的超标量处理器
2. VLIW 处理器
3. 动态的超标量处理器

**The Basic VLIW Approach**

VLIW 指将多个操作放入一个非常长的指令中，并要求这些操作满足一些限制。为了尽可能占满 VLIW 的指令槽，需要编译器调度代码。在一个基本块内调度称为本地调度技术，在多个基本块间调度称为全局调度技术。

VLIW 技术的问题包括增加了指令的大小和同步操作的限制。需要大量的循环展开来有效利用指令槽，以及空闲的指令槽占用大量空间。早期的硬件没有冲突检测逻辑，VLIW 指令中一个操作阻塞会造成其他操作同步阻塞。VLIW 技术也有一些应用问题，包括二进制兼容性问题。

### 3.8 Exploiting ILP Using Dynamic Scheduling, Multiple Issue, and Speculation

由于指令组内相关性，指令的多发射（分发）逻辑会变得非常复杂。对于 n 发射的处理器：对于组内的每一条指令预先分配保留站号和 ROB 号，分析组内相关性，若某指令与组内在它之前的指令相关，则使用预先分配的 ROB 号来填写保留站，无相关则用原先的保留站和 ROB 信息来填写保留站。指令的多提交则相对简单。

### 3.9 Advanced Techniques for Instruction Delivery and Speculation

**Increasing Instruction Fetch Bandwidth**

BTB 存储预测跳转的分支指令的目标地址，在取指时，通过对比 PC 判断是否命中 BTB，若 BTB 中没有相应的表项，则 nextPC 是顺序的，若 BTB 中有相应的表项，则 nextPC 是目标地址。一类 BTB 的变种是存储目标地址和目标指令，这样 BTB 不用当拍返回结果，使得 BTB 可以做的更大，并应用**分支折叠**技术。

**Specialized Branch Predictors: Predicting Procedure Returns, Indirect Jumps, and Loop Branches**

返回地址栈在 call 指令执行时将返回地址入栈，在 return 指令执行时将地址出栈，有效预测返回指令的跳转地址。很多处理器也加入了预测间接跳转指令的专门预测器，而循环分支指令的预测工作近年来由 Tage 承担。

最近的设计将取指单元设计成一个单独的模块，拥有以下功能：

- 分支预测单元
- 指令预取单元
- 指令缓存单元

**Speculation: Implementation Issues and Extensions**

除了使用 ROB 进行寄存器重命名外，我们也可以用额外的物理寄存器重命名。体系结构上可见的寄存器（逻辑寄存器）和指令产生的临时值都存储在物理寄存器中，ROB 仅承担指令按序完成的作用。寄存器重命名表承担着映射物理寄存器和逻辑寄存器的任务。相比于 ROB 重命名，物理寄存器重命名的实现可以简化指令提交，它只需要修改寄存器映射表并释放旧的物理寄存器（该逻辑寄存器之前对应的物理寄存器）。同样的，组内相关性检测也是物理寄存器重命名的一个瓶颈。

现代处理器继续提高发射宽度的主要瓶颈在于：更大的发射宽度造成更复杂的组内相关性检查，使得重命名逻辑变得复杂，重命名的过程必须在一拍内完成，更高的频率限制了重命名的复杂度。

猜测执行需要消耗面积和功耗，而错误的猜测执行造成的例外事件（例如 Cache 缺失或 TLB 缺失）会降低性能。很多流水线只允许低开销的例外事件（例如 L1 Cache 缺失）猜测执行，昂贵的例外事件（L2 Cache 缺失或 TLB 缺失）发生时处理器会等待它不再是猜测状态后再执行。

考虑到更高的分支频率，分支指令的聚集性和功能部件的长延迟，一拍内预测多个分支指令可以提高性能，但至 2017 年还没有处理器实现。

猜测执行会增加能耗，因为执行了不需要的指令，并且额外消耗了恢复状态的能量。但考虑到总体上指令执行的更快，如果不需要的指令被执行得比较少，那总体能效比可能是更优的，不过现实情况是 30% 的指令执行最终没有被用到，降低了能效比。

目前处理器也加入了地址预测：预测两个 Store 或者一个 Load 和一个 Store 的地址是不是相同的。值预测指在数据相关解决前预测指令的源操作数，尽管在学术界有诸多研究，但没有出现在真实的处理器中。

### 3.10 Cross-Cutting Issues

**Hardware Versus Software Speculation**

- 为了更有效地猜测执行，我们必须消除内存引用之间的相关性，这在编译时是很难做到的。硬件支持可以让内存访问猜测执行，但需要谨慎设计以免恢复的额外开销超过猜测执行带来的优势。
- 基于硬件的猜测执行在执行流不可预测时更有优势，因为基于硬件的转移猜测比基于软件的转移猜测正确率更高。
- 基于硬件的猜测执行有利于保持精确例外。
- 基于软件的猜测执行可以利用更大的指令窗口。
- 基于硬件的猜测执行对于没有精心设计的（任意的）代码序列也可以进行动态调度。

**Speculative Execution and the Memory System**

猜测执行可能发送错误的内存访问，造成正确性和性能方面的问题。

### 3.11 Multithreading: Exploiting Thread-Level Parallelism to Improve Uniprocessor Throughput

线程有独立的寄存器状态和 PC，但和进程内的其他线程共享内存空间。超线程指多个线程共用一个处理器而不用进行线程切换。由于动态调度难以隐藏过长的内存访问时间，设计者企图通过其他方式提高 ILP。超线程允许多个线程交叠地共享一个处理器内的功能部件，但有着私有的寄存器和 PC。

超线程有三种实现：细粒度超线程，粗粒度超线程和同步超线程。细粒度超线程以轮询的方式每周期切换进程，在面向多线程负载的处理器中有所应用；粗粒度超线程只在长时间阻塞的情况下才会切换线程，但由于启动的额外开销，并没有实际应用。最常见的是同步多线程（SMT），它在寄存器重命名和动态调度的时候不区分来自不同进程的指令。

**Effectiveness of Simultaneous Multithreading on Superscalar Processors**

SMT 带来的性能和能耗比的提升比较有限，但也是宝贵的。

### 3.12 Putting It All Together: The Intel Core I7 6700 and ARM Cortex-A53

**The ARM Cortex-A53**

ARM Cortex-A53 是一款静态双发射处理器核，有八个流水级。共有四种预测器：一项的带有 2 条分支目标指令的分支目标缓存，3072 项的混合预测器，256 项的间接分支指令预测器和 8 项的返回地址栈。分支预测错的惩罚是 8 个时钟周期。

**The Intel Core i7 6700**

i7 是一款激进的乱序超标量处理器核，共有 14 个流水级，每周期最多 4 条微码重命名，6 条微码执行。取指单元配备多层的分支预测器和返回地址栈，每周期取出 16 字节的指令。16 字节的指令被放入预译码缓存中，进行宏融合，然后通过预译码分为单独的 x86 指令并放入指令队列中。三个译码器和一个译码引擎将 x86 指令翻译成 RISC 形式的微码，并放入 64 项的微码缓存中，微码缓存还承担了循环检测和微融合的工作。随后指令被重命名、发射、执行，i7 使用了集中式保留站。由于充足的缓存和队列，性能损失主要来自于分支预测错和 Cache 缺失。

## Chapter4: Data-Level Parallelism in Vector, SIMD, and GPU Architectures

### 4.1 Introduction

不仅在科学计算中面向矩阵的运算具有 DLP，面向媒体的图像和音频处理，机器学习算法也有 DLP。SIMD 架构可以降低功耗，更重要的是它方便程序员以顺序思维进行编程。本章将介绍三种 SIMD 的变种：向量体系结构、多媒体指令扩展和 GPU。

### 4.2 Vector Architecture

向量体系结构将内存中分散的元素取到一个更大的寄存器堆中，对寄存器堆中的数据进行操作并将结果写回内存中。

**RV64V Extension**

RV64V 指 RISC-V 基础指令集加上向量扩展 RVV。RV64V 指令集由以下部分组成：

- 向量寄存器
- 向量功能部件
- 向量访存部件
- 标量寄存器

RV64V 的创新是使用了动态寄存器类型，它可以减少向量指令的类型。动态寄存器类型还可以减少上下文切换的时间，实现隐式类型转换。

**How Vector Processor Work: An Example**

当循环中没有**循环体间相关**时，循环是可向量化的。向量体系结构将元素之间相关数据的前递称为**链接**。最大向量长度（mvl）是硬件确定的，如果向量长度不等于 mvl，可以将向量长度存储在 *vl* 寄存器中。由较窄的数据向较宽的数据的类型转换是自动的。

**Vector Execution Time**

向量操作的执行时间主要由以下三点确定：

1. 向量的长度
2. 结构相关
3. 数据相关

给定向量的长度和执行速率，我们就可以算出一条向量指令执行的时间。现代的处理器的向量功能部件都有多条流水线，每周期可以产生两个甚至更多的结果。为了简化，我们的 RV64V 实现只有一条流水线，每拍产生一个结果，那么一条向量指令的执行时间约等于向量长度。

此处我们引入 convoy 的概念，一个 chovoy 内的指令可以并行地执行，没有结构相关，其中的数据相关由链接解决。现代处理器都实现了灵活链接，一条向量指令可以向任何正在执行的其他向量指令前递结果。一个 convoy 执行的时间称为 chime，它忽略了发射指令的开销，因为处理器一次只能发射一条向量指令。另一类不可忽略的开销是功能部件的启动延迟（但 chime 也忽略了）。

**Multiple Lanes: Beyond One Element per Clock Cycle**

使用多条通道可以加速向量指令的执行，每条通道包括向量寄存器堆的读写端口和每个向量执行部件的一条流水线。每个向量寄存器的 0 号元素都使用通道 0，如此定位可以避免跨通道的通讯（前递），减少连线开销。

**Vector-Length Registers: Handling Loops Not Equal to 32**

*vector-length register (vl)* 控制向量操作的长度。使用 strip mining 来处理向量长度比 mvl 大的情况，*setvl* 指令可以简化这个过程。

**Predicate Registers: Handling IF Statements in Vector Loops**

谓词寄存器用于存储向量操作的掩码，初始为全 1。

**Memory Banks: Supplying Bandwidth for Vector Load/Store Unit**

分 Bank 实现可以增加访存带宽。

**Stride: Handling Multidimensional Arrays in Vector Architectures**

矩阵乘法中 stride 用于访问在内存中间隔一定距离的数组元素。

**Gather-Scatter: Handling Sparse Matrices in Vector Architectures**

gather 操作将索引寄存器中的偏移量和基地址相加，将向量取至寄存器中；scatter 操作则同样将索引寄存器中的偏移量和基地址相加，将向量存回内存中。gather-scatter 操作需要程序员给编译器显式提示，而且执行速度远慢于常规访存。

**Programming Vector Architectures**

使用向量编程的一个优势是，编译器会给出提示，指示为什么代码不能被向量化，程序员可根据提示修改程序。

### 4.3 SIMD Instruction Set Extensions for Multimedia

SIMD 指令源于简单的观察：图像和声音处理系统使用的数据位宽通常小于 32bits。相比向量指令，SIMD 指令使用更小的寄存器堆，没有向量长度寄存器，没有 stride 和 gather-scatter 操作，也没有谓词寄存器。

MMX 指令将 64bits 的浮点寄存器分为 8 个 8bits 或者 4 个 16bits。SSE 使用额外的 128bits 的寄存器（XMM 寄存器）。AVX 再次将寄存器位宽增加至 256bits（YMM 寄存器）。

**Programming Multimedia SIMD Architectures**

SIMD 指令通常在库函数中使用。

**The Roofline Visual Performance Model**

算术密度用于衡量内存访问和浮点运算的比例，在 Roofline 模型中，转折点左侧（低算术密度）表示瓶颈在访存，右侧（高算术密度）表示瓶颈在计算。

### 4.4 Graphics Processing Units

GPU 目前已非常普及，本节将介绍 GPU 如何进行运算。

**Programming the GPU**

英伟达研发了一个类似于 C 语言的编程语言和环境用于解决异构计算和多层次并行的困难，称为 CUDA。OpenCL 是一个类似的编程语言，致力于实现平台无关。英伟达将所有层次的并行性统一为 CUDA 线程，CUDA 的编程模型是单指令流多数据流，多个 CUDA 线程形成线程块（Thread Block），执行线程块的硬件称为多线程 SIMD 处理器（英伟达称为流处理器 SM）。

CUDA 中用 `__device__` 和 `__global__` 来表示 GPU，用 `__host__` 来表示处理器。用于 GPU 的函数定义语法为 `name <<<dimGrid, dimBlock>>>(…parameter list…)`，其中 `dimGrid` 表示代码的维度（使用多少线程块），`dimBlock` 表示线程块的维度（有多少 CUDA 线程）。在代码中，`blockIdx` 表示线程块的 ID，`threadIdx` 表示线程块中的线程 ID，`blockDim` 等于 `dimBlock`。

为简化硬件设计，线程块之间不能有相关，线程块之间的执行顺序是任意的，线程块之间不能直接通信，只能通过对全局内存的原子操作来通信。为了获得良好的性能，CUDA 程序员必须时刻关心 GPU 的硬件结构，但这也造成了程序员的生产力降低。

**NVIDIA GPU Computational Structures**

网格（Grid）指运行在 GPU 上的一段代码，由一组线程块组成。一个线程块包括多个 SIMD 线程（英伟达称为 Wrap）。线程块调度器（英伟达称为 Giga Thread Engine）将一个线程块分配给一个多线程 SIMD 处理器，SIMD 线程调度器（英伟达称为 Wrap Scheduler）在一个多线程 SIMD 处理器内部分配每个时钟周期执行哪个 SIMD 线程。多线程 SIMD 处理器与向量处理器类似，但向量处理器有几个流水化的功能部件，而多线程 SIMD 处理器有许多并行的功能部件。GPU 由一个或多个多线程 SIMD 处理器组成。

每个 SIMD 线程有单独的 PC，SIMD 线程调度器会将准备好的 SIMD 线程发送给分配单元。SIMD 指令的位宽是 32，所以一个 SIMD 线程可以处理 32 个元素。SIMD 处理器需要执行 SIMD 线程，因此有多条 SIMD 通道，类似于向量通道，例如有 16 个通道，那么执行一条 SIMD 指令需要 2 拍。

因为不同的 SIMD 线程之间是独立的，所以 SIMD 线程调度器可以任意选择准备好的 SIMD 线程执行，使用计分板跟踪至多 64 个 SIMD 线程的执行状态。一个 SIMD 处理器有 32K 至 64K 个寄存器，每个 SIMD 通道拥有独立的寄存器（例如每通道 1K 个）。每个 SIMD 线程最多使用 256 个向量寄存器。寄存器是动态分配给线程块的。注意一个 CUDA 线程是 SIMD 线程的投影，是 SIMD 线程中的一个元素的操作。

**NVIDIA GPU Instruction Set Architecture**

PTX 指令集是硬件指令集的一个抽象，其指令描述单个 CUDA 线程的操作，与机器指令通常（但不绝对）是一一对应的。PTX 指令使用无限多的只写一次寄存器，编译器会为这些寄存器分配物理寄存器，优化器会减少物理寄存器的使用数量。

PTX 指令的格式是 `opcode.type d, a, b, c`，其中 d 是目的寄存器（store 指令除外），a、b、c 是源操作数。PTX 使用谓词寄存器来实现分支指令。控制流指令包括函数跳转的 `call` 和 `return`，线程相关的 `exit` 和 `branch`，同步指令 `bar.sync`。GPU 的所有访存都是 gather-scatter 形式的，硬件会识别其空间相关性。

**Conditional Branching in GPUs**

GPU 相比向量处理器对分支指令有更多的硬件支持。在 PTX 指令层面，使用 `branch`、`call`、`return` 和 `exit` 指令和每个线程之间的锁步。在 GPU 机器指令层面，除了常规的分支指令外，每个 SIMD 线程还有分支同步栈。对于 PTX 指令中简单的分支指令，编译器只需使用谓词寄存器；复杂的分支则需要使用分支同步栈，在不同 CUDA 线程分流时压栈，合流时出栈。分支的层数越多，GPU 运算的效率越低，PTX 编译器会对分支指令进行编译优化。需要注意的是，同一线程块内的 CUDA 线程看似是相互独立的，但是在分支中他们是相关的，只有当谓词寄存器中 1 的比例足够大时，程序才有较好的效率。

**NVIDIA GPU Memory Structures**

每一个 SIMD 通道（即 CUDA 线程）有单独的内存，称为私有内存，GPU 将私有内存缓存在 L1 和 L2 Cache 中。每个多线程 SIMD 处理器有本地内存，其容量较小（48KB 左右）但延迟较低，带宽较高；本地内存是线程块动态获取的。GPU 内存是所有的线程块共享的，其还能被系统处理器（CPU）读写。GPU 将大量晶体管用于寄存器而非 L2 和 L3 缓存，并使用多线程来隐藏访存延迟。

**Innovations in the Pascal GPU Architecture**

Pascal GPU 的每个多线程 SIMD 处理器拥有两个 SIMD 线程调度器，每周期可以调度两个线程的 SIMD 指令到两组 16 SIMD 通道的功能部件。Pascal GPU 还有以下创新：

- 更快的单精度、双精度、半精度浮点运算；
- HBM2 实现更高的内存带宽；
- 高速的芯片间互联；
- 统一的虚拟内存和分页支持。

**Similarity and Difference Between Vector Architectures and GPUs**

SIMD 处理器和向量处理器类似，但 SIMD 处理器支持多线程。GPU 有更多的寄存器以支持多线程，也有更多的通道以更快执行一条 SIMD 指令。对于访存，GPU 中所有访存都是隐含的 gather-scatter 形式，而向量处理器则是自定义的。对于隐藏访存延迟，向量处理器的方式是均摊访存延迟至所有向量元素，GPU 的方式是多线程。对于分支指令，两者都使用掩码，但 GPU 使用硬件和编译器处理掩码。对于 GPU 而言，系统处理器通过 PCIe 相连，访问延迟非常高，而对于向量处理器而言，标量处理器易于访问。

**Similarity and Differences Between Multimedia SIMD Computers and GPUs**

GPU 拥有更多的 SIMD 通道，支持更多的线程。

**Summary**

虽然英伟达将所有层次的并行性统一为 CUDA 线程，但程序员必须了解 CUDA 线程是如何被组织成线程块的，32 个 CUDA 线程是一起执行的。

### 4.5 Detecting and Enhancing Loop-Level Parallelism

挖掘循环体的并行性是提高程序执行效率的重点，本节将谈论循环体的编译优化。循环体间相关是阻碍编译器进行编译优化的主要因素，在循环的某一趟执行中产生的数据会在之后的某一趟执行中被用到，导致两趟循环不能并行执行。但是循环体间相关没有形成环时，相关性是偏序的，可以被消除。另外，识别指向同一个地址的两个引用（尤其是循环间的）从而消除不必要的访存指令也是至关重要的。

**Finding Dependences**

指针和数组访问对相关性分析造成了困难。几乎所有相关性分析算法都会假设数组访问是仿射的，即 `b[i]` 的地址是 `a*i+b`，多维数组是仿射的当且仅当每一维都是仿射的，散列访问（`x[y[i]]`）是非仿射的。判断循环中两个对同一数组的访问是否指向同一地址，只需分析仿射函数是否有相同的值即可，即判断是否存在两个 `i` 使得 `a*i+b` 等于 `c*i+d`。在部分程序中，`a`、`b`、`c`、`d` 这些值可能不是常量，导致相关性分析无法进行。但在大多数程序中，这些值是常量，此时若 GCD(c,a) 整除 (d-b) 时即存在循环间相关。GCD 测试可能出现假阳性，因为其没有考虑循环的边界。相关性分析也存在很大的局限性，因为其假设数组访问是仿射的，而且对过程调用中的指针无能为力。

**Eliminating Dependent Computations**

对于累加程序，我们可以将结果进行向量扩展，然后将向量进行合并（reduction）。任何具有组合律的运算都可以进行合并，但计算机中数的运算受精度和范围的限制，可能不具有组合律。

### 4.6 Cross-Cutting Issues

**Energy and DLP: Slow and Wide Versus Fast and Narrow**

通过降低频率和电压，增加更多运算资源，在保持峰值运算性能的基础上可以降低功耗，因此 GPU 的频率通常更低。而且向量运算通常可以简化控制单元，减少译码和相关性检测的逻辑。

**Banked Memory and Graphics Memory**

为提高访存带宽，GPU 通常使用堆叠式内存（HBM），而且访存合并的控制逻辑也更为复杂。

**Strided Accesses and TLB Misses**

在跨步式访问中，每次访存都可能造成 TLB 缺失，造成性能的极大损失。

### 4.7 Putting It All Together: Embedded Versus Server GPUs and Tesla Versus Core I7

由于图形应用的流行，移动终端、台式电脑和服务器中都有不同规格的 GPU。

**Comparison of a GPU and a MIMD With Multimedia SIMD**

GPU 有更高的访存带宽，更高的单精度和双精度浮点运算性能，但 Cache 容量较小，若工作集可以完全放在多核处理器的 Cache 中，GPU 可能处于劣势。另外，GPU 缺少内存一致性模型和一些同步指令，SIMD 指令缺少 gather-scatter 形式的访存。

**Comparison Update**

Intel 最新的架构已加入 gather-scatter 形式的访存，CPU 和 GPU 的速度也有了很大的提升。

## Chapter5: Thread-Level Parallelism

### 5.1 Introduction

多处理器架构的流行反映出以下因素：

- 继续增加硅面积来挖掘 ILP 的效率已经很低，而功耗却成为关键因素；
- 高端服务器的需求旺盛；
- 数据密集型的应用在互联网的驱动下井喷；
- 台式电脑继续增加性能的需求已经降低（图像处理除外）；
- 有效运用多处理器架构的方法论逐渐成熟；
- 多处理器降低开发芯片的成本。

本章我们讨论线程级并行性（TLP），意为多个程序通过 MIMD 的形式开发并行性，主要是通过多处理器。本章讨论的多处理器是多个紧密相关的处理器核，由同一个操作系统控制并共享地址空间。在这样的多处理器上软件有两种方式开发并行性，一是多个线程合作处理同一个任务，称为并行处理；二是多个互不相关的进程同时工作，称为任务级并行。

多处理器虽然共享地址空间，但不意味着只有单一的物理内存。多处理器不仅包括单芯片的多核处理器，也包括多芯片的计算机，每一个芯片都是一个多核处理器。在下一章我们将讨论大量处理器通过网络互联的架构，称为仓库级计算机。

**Multiprocessor Architecture: Issues and Approach**

TLP 相比 ILP 的层次更高，通常由程序员或操作系统来识别。虽然多线程也可以挖掘 DLP，但其要求每个线程的工作量足够大来减少额外开销，比起 SIMD 和 GPU 而言代价更高。

第一种多处理器系统称为对称多处理器或集中式共享内存多处理器（SMP 或 UMA）。这种多处理器系统通常只有 32 核或更少，当前的多核处理器基本都是 SMP，但部分多核处理器对 LLC 的访问是不一致的，称为 NUCA。第二种多处理器系统称为分布式共享内存多处理器（DSM 或 NUMA），其可能由多个多核处理器组成，处理器访问本地内存远快于远程内存。

SMP 和 DSM 都是共享内存的，即所有处理器共享地址空间。下一章将讨论消息传递模型的多处理机系统。

**Challenges of Parallel Processing**

并行处理的第一个难题是程序中有限的并行性，第二个难题是通讯的高额开销。

### 5.2 Centralized Shared-Memory Architectures

多级的、容量较大的 Cache 减少访存带宽的需求是集中式内存多处理器流行的重要原因。但对于共享数据的缓存造成了新的问题：缓存一致性（coherence）。

**What Is Multiprocessor Cache Coherence?**

一个内存系统是一致的，如果

- 处理器 P 向位置 X 写入值后处理器 P 向位置 X 读取值，若中间没有其他处理器向位置 X 写入值，则读取的值是写入的值；
- 一个处理器向位置 X 写入值后另一个处理器向位置 X 读取值，若中间没有其他处理器向位置 X 写入值，且写入和读取时足够分开的，则读取的值是写入的值；
- 写入操作是串行的，即两个写入的顺序对于所有处理器而言都是一致的。

一个处理器写入的值何时才能被其他处理器看到也是一个重要的问题，该问题称为内存一致性（consistency）模型。缓存一致性和内存一致性是互补的，前者定义了对同一内存位置读写的行为，后者定义了对不同内存位置读写的行为。在本节中，假设写操作完成指所有处理器都看到了写操作的效果，处理器不会改变写操作的顺序。

**Basic Schemes for Enforcing Coherence**

共享数据在 Cache 中存在多个拷贝对性能有着很大的好处，所以硬件通过 Cache 一致性协议来解决 Cache 一致性问题。Cache 一致性协议的关键是跟踪共享的数据块的状态，有两类一致性协议：

1. 基于目录的：共享数据块的状态是集中存放在目录中的，目录可以是集中的，也可以是分布的。
2. 基于侦听的：每一个 Cache 行都保存共享数据块的状态，通过总线进行通讯。

**Snooping Coherence Protocols**

有两种方式来确保 Cache 一致性。一是在写操作之前确保对该数据块具有独占的访问权限，称为写无效协议。二是在写操作之后进行广播，更新其他 Cache 中的值，称为写更新协议。写更新协议的总线压力大，在现实中较少使用。

**Basic Implementation Techniques**

当处理器对共享的 Cache 行进行写操作时，必须在总线上广播无效请求。当两个处理器同时广播无效请求时，总线仲裁器会将其串行化。

对于写穿透 Cache 而言，内存中的数据总是最新的，因此缺失的 Cache 请求可以从内存中取回数据。对于写回 Cache 而言，最新的数据可能在某个处理器的 Cache 中，当该 Cache 侦听到读请求时，会提供最新的数据并取消从内存取回的数据。

除了有效位和脏位以外，我们还需要共享位来标记 Cache 行的状态。

**An Example Protocol**

基于侦听的 Cache 一致性协议通常由每个 Cache 行独立的有限状态机实现。本节将讨论 MSI 协议，包含修改、共享和无效三个状态。本节中对于总线操作的原子性假设在现实世界中是不成立的。

**Extensions to the Basic Coherence Protocol**

MESI 协议增加了独占状态，表示处理器拥有该 Cache 行的独占权限但未进行修改。MOESI 协议增加了占有状态，表示处理器拥有该共享 Cache 行的最新拷贝，内存中的相应数据是过时的。

**Limitations in Symmetric Shared-Memory Multiprocessors and Snooping Protocols**

当核数继续增长，每个核的内存带宽需求继续增长，任何集中式的资源都会变成瓶颈，例如总线。以下技术可以增加侦听总线的带宽：

- 将 tag 进行复制
- 将共享 LLC 进行分块
- 在共享 LLC 处部署（分布式）目录

**Implementing Snooping Cache Coherence**

当前的处理器总线不止一条，造成一致性操作是非原子的，增加了额外的复杂度。

### 5.3 Performance of Symmetric Shared-Memory Multiprocessors

基于侦听协议的 Cache 系统的性能取决于单核中固有的 Cache 缺失的开销和一致性开销等诸多因素。一致性开销又分为真共享和假共享。

**A Commercial Workload**

加大 L3 Cache 可以减少单核中固有的 Cache 缺失的开销，但对一致性开销无济于事。核数的增加会导致一致性开销的增加。增加 Cache 行大小可以减少真共享开销，但增加了假共享开销。

**A Multiprogramming and OS Workload**

此例使用的测试集为一个编译程序的某一个阶段。

**Performance of the Multiprogramming and OS Workload**

操作系统对 Cache 的需求更高，增加 Cache 行大小可以有效改善操作系统的 Cache 性能。

### 5.4 Distributed Shared-Memory and Directory-Based Coherence

侦听协议没有集中式的数据结构，降低了开销，也限制了其可扩展性。近些年来访存带宽需求的增加使得分布式内存受到更多欢迎，但除非我们降低侦听协议的广播开销，否则分布式内存中 Cache 的一致性问题会成为累赘。有共享包含式 LLC 的存储系统中，只需要在 LLC 的每个 Cache 行中加入长度为核数的比特指示当前 Cache 行在哪些核的 Cache 中有备份，即可实现集中式目录协议。

要实现分布式目录协议，我们可以将目录分布在 LLC 的每个 bank 中，也可以将目录分布在每个内存中。若目录分布在每个内存中，那么目录需要记录每个内存行的状态，行大小等于 LLC 的行大小，空间复杂度是行数量乘以核数量，对于几百个核的计算机来说都是可以接受的。

**Directory-Based Cache Coherence Protocols: The Basics**

在简单的目录协议中，有以下状态：共享、无效、修改。目录会和发出请求的节点、地址所在的节点和拥有该 Cache 行的节点进行通讯。

**An Example Directory Protocol**

状态转换图和侦听协议基本一致，除了写缺失不是在总线上广播的，而是由目录控制进行点对点的取回和无效的。目录收到某个核的请求时，可能需要改变目录的状态，向其他核发送请求。

### 5.3 Synchronization: The Basics

同步机制是软硬件结合实现的，本节将关注基于锁的同步机制。

**Basic Hardware Primitives**

实现同步的基础是能够原子读 - 修改 - 写的硬件原语，软件开发者一般不直接使用硬件原语，而是依靠系统开发者提供的同步库函数。最简单的硬件原语是原子交换，其他还包括 TAS、原子加等。另一种实现是 LL-SC。

**Implementing Locks Using Coherence**

实现了 TAS 锁和 TTAS 锁。

### 5.6 Models of Memory Consistency: An Introduction

内存一致性是不同处理器对不同位置的读写之间的属性。最直观的内存一致性模型是顺序一致性，对每个处理器而言内存访问是按序的，不同处理器之间的内存访问是任意交叠的。实现顺序一致性的方式是将一个内存访问的完成推迟到其他所有处理器都完成（可能需要的）无效请求。顺序一致性大大影响了性能，体系结构研究者提出了更为宽松的内存一致性协议。

**The Programmer's View**

对于程序员来说，无数据冒险的程序是最符合直觉的。

**Relaxed Consistency Models: The Basics and Release Consistency**

放松一致性模型的核心是允许读写操作乱序完成，但提供强制按序执行的同步原语。根据放松的程度不同，有许多不同的一致性模型。顺序一致性维护所有四种顺序：R->W，R->R，W->W，W->R。

- 放松 W->R 的模型称为 TSO 或处理器一致性模型。
- 放松 W->R 和 W->W 的模型称为 PSO。
- 放松所有四种顺序的模型有很多种，包括弱一致性模型和宽松一致性模型。

宽松一致性模型将对共享变量的同步操作分为获取和释放。

### 5.7 Cross-Cutting Issues

**Compiler Optimization and the Consistency Model**

编译器不能交换对不同变量的读和写之间的顺序，因为这可能改变程序的语义。这极大地限制了编译优化的效果。

**Using Speculation to Hide Latency in Strict Consistency Models**

可以在处理器内部将访存指令乱序执行，按序提交，在不符合内存一致性的操作出现时进行回滚。

**Inclusion and Its Implementation**

包含式的 Cache 可以减少一致性协议的通讯开销，但不同层次的 Cache 有不同的块大小，增加了复杂度。

**Performance Gains From Multiprocessing and Multithreading**

初步分析超线程技术的性能优势。

### 5.8 Putting It All Together: Multicore Processors and Their Performance

**Performance of Multicore-Based Multiprocessors on a Multiprogrammed Workload**

介绍了 IBM Power8、Intel Xeon E7 和 Fujitsu SPARC64 X+。

**Scalability in an Xeon MP With Different Workload**

分析了 Java 服务器、虚拟机和科学计算三种负载在多核下的表现。

**Performance and Energy Efficiency of the Intel i7 920 Multicore**

多核和超线程将能效比的负担更多地转移给了程序员。

## Chapter6: Warehouse-Scale Computers to Expoilt Request-Level and Data-Level Parallelism

### 6.1 Introducion

WSC 的体系结构和服务器的体系结构有很多相似之处，它们都关注性价比、能效比、可靠性、网络带宽和交互式、批处理工作性能。WSC 的体系结构也有不同，例如更充分的并行性（数据并行和请求并行），更关注运营开销、选址、低负载时的能效比和规模问题。WSC 和 HPC、数据中心也有很大不同。

### 6.2 Programming Models and Workloads for Warehouse-Scale Computers

MapReduce 是 WSC 上常用的操作，WSC 对执行 MapReduce 的稳定性做了优化。

### 6.3 Computer Architecture of Warehouse-Scale Computers

一个机柜中有 40-80 台服务器，机柜顶部有交叉开关用以连接不同的机柜。

**Storage**

硬盘可以放在机柜中，不同机柜间通过网络互联访问；也可以使用 NAS 等结构。

**WSC Memory Hierarchy**

访问本地存储设备、机柜内的存储设备和机柜间的存储设备有不同的带宽和延迟。

### 6.4 The Efficiency and Cost of Warehouse-Scale Computers

实践中不会有所有服务器同时处于最高负载的情况，因此可以安全地削减 40% 的电能供应。

**Measuring Efficiency of a WSC**

PUE 是所有设备的功耗除以 IT 装备的功耗，其越大表示 WSC 的能效比越低。虽然 WSC 的设计者更关心带宽，但软件和用户日渐关心延迟。体系结构学者提出了 SLO 的概念，表示绝大部分的请求（例如 99.9%）都要在阈值范围内完成。

**Cost of a WSC**

运营开销称为 OPEX，建造开销称为 CAPEX。

### 6.5 Cloud Computing: The Return of Utility Computing

WSC 的单价比起小规模的数据中心更低，但中小企业无法承担分布式 WSC 的开销，因此云计算成为新趋势。

**Amazon Web Services**

亚马逊云服务能够成功的原因有：虚拟机的支持，低售价，开源软件的支持，无保证的服务等。目前亚马逊提供种类众多的不同性能和优化方向的云服务器实例。使用云服务器最大的优点是按需使用，无需在创业时就将大量资金投入服务器的采购，而是根据企业的发展逐步增加购买的云服务器实例的数量。

**How Big Is the AWS Cloud**

目前微软、亚马逊和谷歌在各个大洲都有大量 WSC。

### 6.6 Cross-Cutting Issues

**Preventing the WSC Network From Being a Bottleneck**

大厂正在自研交换机。

**Using Energy Efficiently Inside the Server**

电源损耗和低功耗时的能效比仍有提升空间。

### 6.7 Putting It All Together: A Google Warehouse-Scale Computer

**Power Distribution in a Google WSC**

WSC 配备 UPS 甚至发电机。

**Cooling in a Google WSC**

利用风扇和水进行散热的系统。

**Racks of a Google WSC**

UPS 是分布式的。

**Networking in a Google WSC**

谷歌的网络拓扑系统称为 Close。

**Servers in a Google WSC**

基于 Intel Haswell 处理器。

**Conclusion**

现在的 WSC 更加节能。

## Chapter7: Domain-Specific Architectures

### 7.1 Introduction

摩尔定律和登纳德缩放定律面临终结，领域特定加速器（DSA）成为继续提升计算机性能的重要途径。

### 7.2 Guidelines for DSAs

- 使用指定的内存区域以减少数据移动。
- 将晶体管更多地用于运算单元和内存。
- 利用该领域最简单的并行性。
- 减少数据类型的大小。
- 使用特定领域的语言以方便代码移植。

### 7.3 Example Domain: Deep Neural Networks

机器学习需要大量的数据和算力，目前 DNN 是机器学习中的热门。

**The Neurons of DNNs**

介绍了神经元、激活函数和隐藏层的概念。

**Training Versus Inference**

介绍了监督学习、反向传播和 SGD。

**Multilayer Perceptron**

MLP 是全连接神经网络。

**Convolutional Neural Network**

CNN 将上一层神经元的输出中临近的部分作为本层神经元的输入，利用空间相关性。

**Recurrent Neural Network**

RNN 在 DNN 中增加了状态机。LSTM 是目前最流行的 RNN。

**Batches**

将多个数据一起输入 DNN 进行训练可以提高效率，这些数据称为一个 Batch。

**Quantization**

将浮点数转为定点数进行操作可以节约资源。

**Summary of DNNs**

对于 DNN 加速的硬件需要矩阵向量乘、矩阵矩阵乘、非线性函数等运算能力。

### 7.4 Google's Tensor Processing Unit, an Inference Data Center Accelerator

TPU 的核心是 65536 个 8 位的 ALU 用于计算矩阵乘法。

**TPU Origin**

TPU 最初起源于语音识别的需求。

**TPU Architecture**

TPU 是 PCIe 通道上的协处理器，有 CPU 发送指令并由 DMA 搬运数据。

**TPU Instruction Set Architecture**

因为指令由 PCIe 通道传输，因此遵循 CISC 风格，没有程序计数器和分支指令，以下五个指令最为常用：

1. `Read_Host_Memory`
2. `Read_Weights`
3. `MatrixMultiply/Convolve`
4. `Activate`
5. `Write_Host_Memory`

**TPU Microarchitecture**

不同的指令有不同的功能部件，可以并行执行。使用脉动运行的方式以减少缓存的读取和写入。

**TPU Implementation**

大部分面积用于缓存和运算单元，而不是控制逻辑。

**TPU Software**

TPU 的软件栈与 CPU 和 GPU 相同，使用 TensorFlow。TPU 和 GPU 一样，使用用户空间驱动和核心驱动，核心驱动只负责内存管理和中断，是轻量级的。用户空间驱动管理 TPU 的运行，负责处理数据，将 API 调用转化为 TPU 指令等，用户空间驱动经常更新。

**Improving the TPU**

提升内存带宽最有效。

**Summary: How TPU Follows the Guidelines**

参考 7.2 节。

### 7.5 Microsoft Catapult, a Flexible Data Center Accelerator

微软将 FPGA 连接至 PCIe 通道，并使用专用网络连接不同的 FPGA。

**Catapult Implementation and Architecture**

数据中心中一半的机柜配备 Catapult 加速板，低速的网络将一个机柜内的 48 个 FPGA 连接起来。

**Catapult Software**

开发者需要编写 RTL 来使用 FPGA。

**CNNs on Catapult**

微软提供了一个可配置的 CNN 加速器作为 Catapult 的应用。

**Search Acceleration on Catapult**

使用 FPGA 可以加速 Bing 搜索引擎的文档排序功能。

**Catapult Version 1 Deployment**

微软先进行了小规模的测试。

**Catapult Version 2**

第二版将 FPGA 放置在 CPU 和 NIC 之间，以利用原有网络拓扑。

**Summary: How Catapult Follows the Guidelines**

参考 7.2 节。

### 7.6 Intel Crest, a Data Center Accelerator for Training

Crest 的目标是加速 DNN 的训练。

### 7.7 Pixel Visual Core, a Personal Mobile Device Image Processing Unit

Pixel 是可编程、可扩展的图像处理和计算视觉加速器。

**ISPs, the Hardwired Predecessors of IPUs**

几乎所有移动设备中都有 ISP，它可以处理拍摄的照片，消除噪点，提升图像质量。但 ISP 的可扩展性不足。

**Pixel Visual Core Software**

Pixel 将 ISP 中的核心流水线转化为了有向无环图，Pixel 程序使用 Halide 语言编写。

**Pixel Visual Core Architecture Philosophy**

Pixel 需要将功耗降到最低，除了 7.2 节提到的原则，以下原则也指导了 Pixel 的设计：

- 二维比一维好。
- 近比远好。

Pixel 的可扩展性体现在以下方面：

- 使用二维的 PE 实现二维的 SIMD 架构。
- 每个 PE 都有单独的缓存。
- NESW 四个方向都可以作为数据输入。

**The Pixel Visual Core Halo**

使用简化的 PE 来处理边缘处的留白。

**A Processor of the Pixel Visual Core**

Pixel 组成的处理器包含 2 个或更多 Pixel 核心，使用 mesh 网络连接，包含访存单元 SHG，标量流水 SCL，单独的指令内存和 DMA。

**Pixel Visual Core Instruction Set Architecture**

和 GPU 一样，Pixel 采用两趟编译的风格，先将目标语言（Halide）编译成 vISA，再将 vISA 编译为 pISA。vISA 是超长指令字风格，对于图像大小、寄存器分配和内存大小没有限制。

**Pixel Visual Core Example**

vISA 代码实例。

**Pixel Visual Core Processing Element**

halo 的额外开销相对较小。

**Two-Dimensional Line Buffers and Their Controller**

LB 负责存储不同 Kernel 之间的中间结果。

**Pixel Visual Core Implementation**

初代的 Pixel 是一个单独的芯片。

**Summary: How Pixel Visual Core Follows the Guidelines**

参考 7.2 节。

### 7.8 Cross-Cutting Issues

**Heterogeneity and System on a Chip (SOC)**

IP 化的趋势。

**An Open Instruction Set**

DSA 通常需要与 CPU 绑定，也就是与特定指令集绑定。

### 7.9 Putting It All Together: CPUs Versus GPUs Versus DNN Accelerators

利用 6 个 DNN 测试集比较 Intel Haswell，NVIDIA K80 和 TPU 的性能。

**Performance: Rooflines, Response Time, and Throughput**

将计算密度重新定义为每次读取权重后的运算次数，可以用 Roofline 模型展示 DNN 的计算性能。由于响应时间的限制，CPU 和 GPU 并不能达到理想的吞吐量。

**Cost-Performance, TCO, and Performance/Watt**

TCO 是衡量服务器芯片价格的较好指标。

**Evaluating Catapult and Pixel Visual Core**

Pixel 性能较低。
