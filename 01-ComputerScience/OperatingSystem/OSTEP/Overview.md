# 课程介绍

有的章节有作业，参考对应章节和 https://pages.cs.wisc.edu/~remzi/OSTEP/Homework/homework.html 。作业都是通过 Python 仿真模拟一个简单的操作系统的功能。

- [x] 4
- [x] 13
- [x] 26
- [ ] 27
- [ ] 36

---

# 4 The Abstraction: The Process

The OS’s core abstraction: the **process** (a running program, distinct from static on-disk programs). To let users run multiple programs simultaneously, the OS uses **CPU virtualization** via time sharing (alternating processes on physical CPUs) to create the illusion of many virtual CPUs, with potential performance costs from sharing. Key OS components for this are **mechanisms** (low-level tools like context switches to switch processes) and **policies** (algorithms for decisions like which process to run next, using history/workload/performance metrics).

A process’s state includes its **address space** (memory for code/data), **registers** (e.g., **program counter (PC)** for next instruction, **stack pointer and associated frame pointe**r to manage the stack for the function parameters, local variables, and return addresses), and **I/O info** (open files). The **Process API** covers creating/destroying processes, waiting for them, suspending/resuming, and checking status.

Process creation involves loading code/static data into memory (eagerly in simple OSes, lazily in modern ones), allocating **run-time stack** (or just stack, for locals/parameters) and heap (for dynamic data), initializing I/O (e.g., 3 default **file descriptors** in UNIX), and starting execution at `main()`.

Processes have 3 core states: **Running** (executing instructions), **Ready** (ready but not running), **Blocked** (waiting for events like I/O, then returning to Ready). The OS uses data structures (e.g., xv6’s `proc` struct) to track processes—storing state, registers, memory, PID, open files, etc.—via a process list (with **Process Control Blocks**).

![[file-20250828201157484.png]]

> [!important] ASIDE: KEY PROCESS TERMS
> - The **process** is the major OS abstraction of a running program. At any point in time, the process can be described by its state: the contents of memory in its **address space**, the contents of CPU registers (including the **program counter and stack pointer**, among others), and information about I/O (such as open files which can be read or written).
> - The **process API** consists of calls programs can make related to processes. Typically, this includes creation, destruction, and other useful calls.
> - Processes exist in one of many different **process states**, including running, ready to run, and blocked. Different events (e.g., getting scheduled or descheduled, or waiting for an I/O to complete) transition a process from one of these states to the other.
> - A **process list** contains information about all processes in the system. Each entry is found in what is sometimes called a **process control block (PCB)**, which is really just a structure that contains information about a specific process.

---

# 13 The Abstraction: Address Spaces

![[file-20250830144320656.png]]

The OS provides **memory virtualization** with **address space** (a running program’s view of its memory) as the core abstraction—address space includes the static **code segment** (instructions of the program), expandable **heap segment** (dynamically-allocated, user-managed memory like `malloc()`), and expandable **stack segment** (to keep track of where the program is in the function call chain as well as to allocate local variables and pass parameters and return values to and from routines.).

![[file-20250830144308810.png]]

The **virtual memory (VM) system** realizes memory virtualization by translating **virtual addresses** (visible to users) to **physical addresses** (real memory locations). The VM system is responsible for providing the illusion of a large, sparse, private address space to each running program; each virtual address space contains all of a program’s instructions and data, which can be referenced by the program via virtual addresses.

VM follows three goals: **transparency** (invisible to programs), **efficiency** (low time/space overhead, relying on hardware like TLBs), and **protection** (preventing cross-process/OS memory access). **Isolation** (key for reliability) is enabled by protection. VM relies on **mechanisms** (e.g., address translation) and **policies** (e.g., memory eviction) to function.

# Concurrency

# 26 Concurrency: An Introduction

Threads is a concurrency abstraction within a single process, which are execution points in a process—multi-threaded processes have multiple PCs, share an address space (enabling easy data sharing), and each has private registers. The state of each thread is stored in **Thread Control Blocks/TCBs**. Thread context switches are lighter than process switches (no page table changes) but require saving/restoring registers. Unlike single-threaded processes (one stack), multi-threaded ones have one stack per thread (thread-local storage).

![[file-20250830150301197.png]]

In this figure, you can see two stacks spread throughout the address space of the process. Thus, any stack-allocated variables, parameters, return values, and other things that we put on the stack will be placed in what is sometimes called thread-local storage, i.e., the stack of the relevant thread.

Two key reasons to use threads are **parallelism** (using multiple CPUs to speed up tasks like array operations) and **avoiding I/O blocking** (letting other threads run while one waits for I/O, critical for servers).

A major issue arises with shared data: threads updating a shared variable (e.g., a counter) often yield incorrect results due to **race conditions**—outcomes depend on execution timing, caused by non-atomic instruction sequences (e.g., loading, incrementing, storing a counter). The core of race conditions are the uncontrolled scheduling of the instructions.

![[file-20250830105132222.png]]

> [!important] ASIDE: KEY CONCURRENCY TERMS CRITICAL SECTION, RACE CONDITION, INDETERMINATE, MUTUAL EXCLUSION
> - A **critical section** is a piece of code that accesses a shared resource, usually a variable or data structure.
> - A **race condition** (or data race [NM92]) arises if multiple threads of execution enter the critical section at roughly the same time; both attempt to update the shared data structure, leading to a surprising (and perhaps undesirable) outcome.
> - An **indeterminate** program consists of one or more race conditions; the output of the program varies from run to run, depending on which threads ran when. The outcome is thus not deterministic, something we usually expect from computer systems.
> - To avoid these problems, threads should use some kind of **mutual exclusion** primitives; doing so guarantees that only a single thread ever enters a critical section, thus avoiding races, and resulting in deterministic program outputs.

---

# 27 Interlude: Thread API

> [!important] ASIDE: THREAD API GUIDELINES
> There are a number of small but important things to remember when you use the POSIX thread library (or really, any thread library) to build a multi-threaded program. They are:
> - **Keep it simple.** Above all else, any code to lock or signal between threads should be as simple as possible. Tricky thread interactions lead to bugs.
> - **Minimize thread interactions.** Try to keep the number of ways in which threads interact to a minimum. Each interaction should be carefully thought out and constructed with tried and true approaches (many of which we will learn about in the coming chapters).
> - **Initialize locks and condition variables.** Failure to do so will lead to code that sometimes works and sometimes fails in very strange ways.
> - **Check your return codes.** Of course, in any C and UNIX programming you do, you should be checking each and every return code, and it’s true here as well. Failure to do so will lead to bizarre and hard to understand behavior, making you likely to (a) scream, (b) pull some of your hair out, or (c) both.
> - **Be careful with how you pass arguments to, and return values from, threads.** In particular, any time you are passing a reference to a variable allocated on the stack, you are probably doing something wrong.
> - **Each thread has its own stack.** As related to the point above, please remember that each thread has its own stack. Thus, if you have a locally-allocated variable inside of some function a thread is executing, it is essentially private to that thread; no other thread can (easily) access it. To share data between threads, the values must be in the **heap** or otherwise some locale that is globally accessible.
> - **Always use condition variables to signal between threads.** While it is often tempting to use a simple flag, don’t do it.
> - **Use the manual pages.** On Linux, in particular, the pthread man pages are highly informative and discuss many of the nuances presented here, often in even more detail. Read them carefully!

# 28 Locks

# 36 I/O Devices
