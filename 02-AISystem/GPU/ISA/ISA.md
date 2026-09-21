# PTX ISA

本文按 NVIDIA 官方 [Parallel Thread Execution ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) 的章节顺序整理。每一节优先保留用户提供的英文原文，随后给出中文翻译；解读仅用于说明关键抽象、边界条件和容易混淆之处。按整理范围，1.3 节版本新增清单不收录。

## 章节

- [[02-AISystem/GPU/ISA/PTX/01-Introduction|1. Introduction]]
- [[02-AISystem/GPU/ISA/PTX/02-Programming-Model|2. Programming Model]]
- [[02-AISystem/GPU/ISA/PTX/03-PTX-Machine-Model|3. PTX Machine Model]]
- [[02-AISystem/GPU/ISA/PTX/04-Syntax|4. Syntax]]
- [[02-AISystem/GPU/ISA/PTX/05-State-Spaces-Types-and-Variables|5. State Spaces, Types, and Variables]]
- [[02-AISystem/GPU/ISA/PTX/06-Instruction-Operands|6. Instruction Operands]]
- [[02-AISystem/GPU/ISA/PTX/07-Abstracting-the-ABI|7. Abstracting the ABI]]
- [[02-AISystem/GPU/ISA/PTX/08-Memory-Consistency-Model|8. Memory Consistency Model]]
- [[02-AISystem/GPU/ISA/PTX/09-Instruction-Set|9. Instruction Set（节选）]]

## 延伸资料

- [NVIDIA PTX ISA：1. Introduction](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#introduction)
- [NVIDIA PTX ISA：2. Programming Model](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#programming-model)
- [NVIDIA PTX ISA：3. PTX Machine Model](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#ptx-machine-model)
- [NVIDIA PTX ISA：4. Syntax](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#syntax)
- [NVIDIA PTX ISA：5. State Spaces, Types, and Variables](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#state-spaces-types-and-variables)
- [NVIDIA PTX ISA：6. Instruction Operands](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#instruction-operands)
- [NVIDIA PTX ISA：7. Abstracting the ABI](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#abstracting-the-abi)
- [NVIDIA PTX ISA：8. Memory Consistency Model](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-consistency-model)
- [NVIDIA PTX ISA：9.7.10.28 Asynchronous copy](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-asynchronous-copy)
- [NVIDIA PTX ISA：9.7.10.28.5 Tensor copy](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-tensor-copy)
- [NVIDIA PTX ISA：9.7.10.28.6 Bulk and Tensor copy completion](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-bulk-tensor-copy-completion)
- [NVIDIA PTX ISA：9.7.15 Parallel Synchronization and Communication Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions)
- [NVIDIA PTX ISA：9.7.15.4 membar / fence](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar-fence)
- [NVIDIA PTX ISA：9.7.15.5 atom](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-atom)
- [NVIDIA PTX ISA：9.7.15.6 red](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-red)
- [NVIDIA PTX ISA：9.7.15.7 red.async](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-red-async)
- [NVIDIA PTX ISA：9.7.15.8 multimem.red.async](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-multimem-red-async)
- [NVIDIA PTX ISA：9.7.15.10 vote.sync](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-vote-sync)
- [NVIDIA PTX ISA：9.7.15.11 match.sync](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-match-sync)
- [NVIDIA PTX ISA：9.7.15.12 activemask](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-activemask)
- [NVIDIA PTX ISA：9.7.15.13 redux.sync](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-redux-sync)
- [NVIDIA PTX ISA：9.7.15.14 griddepcontrol](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol)
- [NVIDIA PTX ISA：9.7.15.15 elect.sync](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-elect-sync)
- [NVIDIA PTX ISA：9.7.15.16 mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier)
- [NVIDIA PTX ISA：9.7.15.16.8–16.16 mbarrier phase and instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-phase-completion)
- [NVIDIA PTX ISA：9.7.15.16.17–16.21 mbarrier arrive/drop/wait/query](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive-drop)
- [知乎：PTX 相关文章](https://zhuanlan.zhihu.com/p/27455487044)
- [Top-K CUDA](https://blog.alpindale.net/posts/top_k_cuda/)
