---
dateCreated: 2025-02-27
dateModified: 2025-08-09
---

# 📜 A High Performance Computing Explorer's Atlas | Ongoing Learning Repository

**Status**: Actively Curating (Knowledge lava cooling into crystallized notes)

> *"Standing on the shoulders of giants and occasionally peeking through their notes"*
> — An evolving handbook for hardware-centric learning

> Recommended to use Git shallow clone:
> `git clone --depth=1 https://github.com/JR-Wesley/Notes`

---

## 📜 Repository Manifesto

> *"Where nanoseconds meet neurons"*
This repository archives my (**not well organized**) learning journey through **VLSI Digital IC design**, incorporating:

- 🧠 **Heterogeneous Computing**: GPU/FPGA/CGRA workload partitioning
- ⚙️ **AI-Tailored Architectures**: Tensor cores to neuromorphic accelerators
- 🔗 **RISC-V Ecosystem**: Custom extension development (Vector, AI/ML)
- 🚨 **VLSI-Scale Verification**: Formal methods for billion-gate designs

**Ethical Note**: Contains reconstructed knowledge from cited sources - strictly for educational purposes.

## Learning in the Era of AI

In the era of LLM, I believe there are severel highlight that should be practices during permenant studying:

1. Systemic View: Bidirectional Insight from Top-Down and Bottom-Up
2. First Principles: The Cornerstone of Critical Thinking
3. Hands-On Practice: Acquiring True Knowledge Through 'Getting Hands Dirty'
4. AI Assistance: Co-Evolution with LLMs


---

## 🌐 Knowledge Matrix

```shell
.
├── 01-ComputerScience
│   ├── Architecture
│   ├── Network
│   ├── OperatingSystem
│   ├── Programming
│   └── SystemBasics
├── 02-AISystem
│   ├── AISysReview
│   ├── AlgorithmAndModel
│   ├── Distributed
│   ├── Framework
│   ├── GPU
│   ├── HardwareAndCompiler
│   ├── HPCBasics
│   └── JobInterview
├── 03-Algorithm
│   ├── BasicAlgorithm
│   ├── Compression
│   ├── Encryption
│   ├── HDC
│   └── Vision
├── 04-IntegratedCircuit
│   ├── Accelerator
│   ├── AsicFlow
│   ├── BasicsAndApplication
│   ├── DSP
│   ├── JobInterview
│   ├── SoC
│   └── StandardAndProtocol
├── 05-SelfDevelopment
│   ├── career
│   ├── language
│   ├── medicine
│   ├── Reading
│   └── recreation
└── ToolKit
    ├── assets
```

![[Knowledge Matrix.png]]

## 🛠️ Usage & Navigation

### Highlight

I've arranged my notes into 6 parts:

- 🗂️ [[01-ComputerScience/01-ComputerScience|01-ComputerScience]]: Foundational pillars of computing, from hardware to software.
- 🗂️ [[02-AISystem/02-AISystem|02-AISystem]]: Deep dive into the architecture and infrastructure powering AI.
- 🗂️ [[03-Algorithm/03-Algorithm|03-Algorithm]]: Core computational methods and specialized techniques for problem-solving.
- 🗂️ [[04-IntegratedCircuit/04-IntegratedCircuit|04-IntegratedCircuit]]: Design, flow, and technology behind modern semiconductor chips.
	- A Guide to RISCV CPU architecture
	- Roadmap of YSYX（一生一芯）
- 🗂️ [[05-SelfDevelopment/05-SelfDevelopment|05-SelfDevelopment]]: Resources for personal growth, career, and well-being.
	- Medicine
- 🗂️ [[00-ToolKit|00-ToolKit]]: Essential tools, scripts, and assets to boost productivity.
	- Enabling high efficiency using Linux, shell, etc.

See Also <a href="https://www.zhihu.com/people/turing-48-20/columns">Zhihu columns</a>.

### Recommended Exploration Paths

- **System of Computer Science**
    1. **Foundation**: Start with [[01-ComputerScience/SystemBasics]] for fundamental concepts.
    2. **Architecture**: Deepen understanding of hardware interaction via [[01-ComputerScience/Architecture]].
    3. **Management**: Explore resource management through [[01-ComputerScience/OperatingSystem]].
    4. **Communication**: Understand data exchange with [[01-ComputerScience/Network]].
    5. **Application**: Apply principles through practical coding in [[01-ComputerScience/Programming]].
- **AI Full-Stack System Optimization**
    1. **Tensor Programming**: Begin with low-level computation in [[02-AISystem/GPU]] and [[03-Algorithm/BasicAlgorithm]], focusing on parallel operations.
    2. **Framework**: Study high-level abstractions and optimization passes in [[02-AISystem/Framework]].
    3. **Integration**: Analyze full-stack performance, bottlenecks, and co-design principles in [[02-AISystem/AISysReview]] and [[02-AISystem/Distributed]].
- **Operator or Backend Acceleration**
    1. **Identify**: Identify core compute-intensive algorithms from [[03-Algorithm]] (e.g., [[03-Algorithm/BasicAlgorithm]], [[03-Algorithm/Vision]]).
    2. **Profile**: Study hardware capabilities and constraints in [[02-AISystem/GPU]] and [[04-IntegratedCircuit/Accelerator]].
    3. **Optimize**: Explore efficient implementation techniques and compiler optimizations in [[02-AISystem/HardwareAndCompiler]].
    4. **Implement**: Design and optimize specific operators leveraging [[02-AISystem/AlgorithmAndModel]] insights.
- **High Performance CPU Design**
    1. **Foundation**: Build foundation with [[04-IntegratedCircuit/BasicsAndApplication]] and [[04-IntegratedCircuit/DSP]].
    2. **Paradigms**: Explore specialized compute paradigms in [[04-IntegratedCircuit/Accelerator]] and reconfigurable architectures (CGRA).
    3. **Requirements**: Study AI-specific hardware requirements and dataflow in [[02-AISystem/AISysReview]], [[02-AISystem/GPU]], and [[02-AISystem/Distributed]].
    4. **Patterns**: Investigate core algorithmic patterns for acceleration, particularly in [[03-Algorithm/HDC]].
    5. **Synthesize**: Synthesize knowledge to design a high-performance CPU microarchitecture.
- **Hardware Accelerator Design**
    1. **Target**: Identify target application domains and algorithms (e.g., from [[03-Algorithm/Vision]], [[03-Algorithm/Encryption]], [[02-AISystem/AlgorithmAndModel]]).
    2. **Define**: Define accelerator specifications and architecture based on computational needs.
    3. **Design**: Design the datapath and control logic, leveraging concepts from [[04-IntegratedCircuit/BasicsAndApplication]] and [[01-ComputerScience/Architecture]].
    4. **Integrate**: Explore integration possibilities within larger systems ([[04-IntegratedCircuit/SoC]]).
- **Silicon Implementation Flow**
    1. **Flow**: Start with the complete design process in [[04-IntegratedCircuit/AsicFlow]].
    2. **Timing**: Master timing analysis and closure techniques in [[04-IntegratedCircuit/AsicFlow]] (assuming STA - Static Timing Analysis is a key part).
    3. **Integration**: Learn system integration and physical design methodologies from [[04-IntegratedCircuit/SoC]], potentially using a case study like [[SoC/pulp_VLSI]].

## 🔐 License Matrix

<a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a>

## 🌟 Roadmap to Enlightenment
