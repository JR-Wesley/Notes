> code: https://github.com/remzi-arpacidusseau/ostep-code.git
> 本节代码：intro/
> 注： mem.c 程序可能有无法暂停的问题。

# Introduction to Operating Systems

The primary way the OS does this is through a general technique that we call virtualization. That is, the OS takes a physical resource (such as the processor, or memory, or a disk) and transforms it into a more general, powerful, and easy-to-use virtual form of itself. Thus, we sometimes refer to the operating system as a virtual machine.

Professor: They are the three key ideas we’re going to learn about: virtualization, concurrency, and persistence. In learning about these ideas, we’ll learn all about how an operating system works, including how it decides what program to run next on a CPU, how it handles memory overload in a virtual memory system, how virtual machine monitors work, how to manage information on disks, and even a little about how to build a distributed system that works when parts have failed. That sort of stuff.

教授：这是我们将要学习的三个核心概念：虚拟化、并发和持久性。在学习这些概念的过程中，我们将全面了解操作系统的工作原理，包括它如何决定接下来在 CPU 上运行哪个程序、如何处理虚拟内存系统中的内存过载问题、虚拟机监控程序如何工作、如何管理磁盘上的信息，甚至还会涉及一些关于如何构建在部分组件出现故障时仍能正常运行的分布式系统的知识。诸如此类的内容。

# Core Knowledge Points of Operating Systems Introduction

## 1. Fundamental of Program Execution & Von Neumann Model

- A running program **executes instructions** via the cycle of **fetching, decoding and executing** from memory, and proceeds sequentially until completion.
- Modern processors optimize with parallel/out-of-order execution, but programs assume **sequential instruction execution** in a simple model.
- This execution model is the basis of the **Von Neumann model of computing**, which stores both instructions and data in memory.

## 2. Definition & Core Roles of Operating System (OS)

- The OS is a body of software that ensures the system operates **correctly, efficiently and in an easy-to-use manner** (early names: supervisor/master control program).
- Three core roles:
  1. **Virtual machine**: Transforms physical resources into **general, powerful, easy-to-use virtual forms** via **virtualization**.
  2. **Interface provider**: Exports hundreds of **system calls (APIs)** as a **standard library** for applications to run programs, allocate memory and access devices.
  3. **Resource manager**: Manages system resources (CPU, memory, disk) with goals like **efficiency and fairness**, coordinating shared resource access for multiple programs.

## 3. Three Core Themes of OS Research

### 3.1 Virtualization (The Crux: How to Virtualize resources?)

- **CPU Virtualization**: The OS (with hardware support) creates the illusion of **an infinite number of virtual CPUs** from a single/small set of physical CPUs, enabling **seemingly concurrent running** of multiple programs (via rapid scheduling/switching). It raises policy questions (e.g., which program to run first).
- **Memory Virtualization**: Each process has a private **virtual address space** mapped to physical memory by the OS; multiple processes can use the same virtual address with **isolated memory access**, making each process seem to own exclusive physical memory.
- Core research focus: **Mechanisms and policies** for virtualization, **efficiency** and required **hardware support**.

### 3.2 Concurrency (The Crux: How to Build Correct Concurrent programs?)

- Concurrency refers to problems arising when **many tasks execute at once** in the same program, originating from OS multiprogramming and now common in modern **multi-threaded programs**.
- The root cause of concurrency errors: Non-atomic execution of operations (e.g., `counter++` requires **load, increment, store** three instructions), leading to inconsistent shared data (e.g., incorrect counter values with high loop counts).
- Core research focus: **Primitives** from OS, **mechanisms** from hardware, and their application to solve concurrency problems.

### 3.3 Persistence (The Crux: How to Store Data persistently?)

- Memory (DRAM) is **volatile** (data lost on power failure/crash); **persistent storage** relies on I/O hardware (hard drive/SSD) and OS software (**file system**).
- The file system manages persistent data **reliably and efficiently** on disks, and disk is not virtualized as private for each application (files enable **inter-process data sharing**).
- Core file operations via system calls: **open()**, **write()**, **close()**; the OS handles underlying disk I/O via **device drivers**.
- File system optimizations & reliability: **delayed/batched writes** for performance; **journaling/copy-on-write** protocols for crash recovery; uses data structures (lists, btrees) for efficient operations.
- Core research focus: Correct persistent storage techniques, high-performance **mechanisms/policies**, and **reliability** against hardware/software failures.

## 4. OS Design Goals

- **Abstraction**: Build core abstractions (CPU, memory, file) to make the system **convenient and easy to use** (a fundamental computer science concept).
- **High performance**: Minimize OS **overheads** (extra time/space) from virtualization and other features.
- **Protection & Isolation**: Isolate processes from each other and the OS to prevent malicious/accidental bad behavior of one program from harming others.
- **High reliability**: Ensure non-stop operation, as OS failure causes all applications to fail (a major challenge for complex modern OS).
- Other goals: **energy-efficiency**, **security** (extension of protection), **mobility** (for small devices); OS implementation varies by usage scenarios.

## 5. Historical Development of OS

1. **Early OS: Just Libraries**
   - A set of libraries for common functions (e.g., low-level I/O); **batch processing** (one program runs at a time, controlled by human operators) due to high hardware costs, no interactivity.
2. **Beyond Libraries: Protection**
   - Pioneered **system call** (Atlas computing system); distinguished **user mode** (hardware-restricted, no direct I/O/memory access) and **kernel mode** (full hardware access).
   - System call uses **trap** instruction to switch to kernel mode and **return-from-trap** to switch back, enabling controlled OS access.
3. **Era of Multiprogramming (Minicomputer)**
   - **Multiprogramming** becomes mainstream: Load multiple jobs into memory and switch rapidly to improve **CPU utilization** (avoid waste from slow I/O).
   - Drives innovations in **memory protection** and concurrency handling; birth of **UNIX** (Bell Labs, Ken Thompson/Dennis Ritchie) – simplifies excellent ideas from prior systems (Multics/TENEX), C-based kernel, open-source, pipe-based workflow, and BSD distribution (Berkeley) with advanced features.
4. **Modern Era (Personal Computer/PC)**
   - Early PC OS (DOS, old Mac OS) regressed (no memory protection, cooperative scheduling); modern PC OS revives minicomputer OS features (macOS based on UNIX, Windows NT with core improvements).
   - **Linux** (Linus Torvalds): UNIX-based, open-source, combined with GNU tools, dominant in servers/Android; UNIX becomes the core of modern OS (macOS, Linux, Android).

## 6. Learning Scope & Homework Requirements

- **Uncovered content**: Networking, graphics devices, advanced security (to be learned in specialized courses).
- **Covered content**: CPU/memory virtualization, concurrency, persistence (devices, I/O, file systems).
- Two homework types:
  1. **Simulation-based**: Simulate system behavior (e.g., disk scheduling) to explore system characteristics (approximations, need real-world verification).
  2. **Real-world code interaction**: C programming/performance measurement on **UNIX-based systems** (Linux/macOS), requiring gcc, Python and code editors; large-scale projects are recommended to solidify systems skills.
