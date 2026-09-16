# GPU 低精度执行机制

低精度格式只有进入匹配的计算与数据移动路径，才会转化为硬件收益。GPU 支持“FP8”至少可能表示四件不同的事：能够存储这种 bit pattern、能够在 CUDA/HIP 中转换这种类型、库中存在接收这种 dtype 的 GEMM、矩阵核心拥有原生 FP8 instruction。四者不能互相替代。

本篇从硬件视角回答：Tensor Core/Matrix Core 实际执行什么；operand、accumulator、instruction shape 与 layout 如何组成计算合同；位宽降低为什么通常提高峰值吞吐；又为什么实际 Kernel 仍可能受 HBM、metadata、转换或 shape 限制。

库内已有 [[02-AISystem/GPU/tensor-core/MMA|MMA]] 和 [[02-AISystem/GPU/tensor-core/CuTe/cutlass与GEMM|CUTLASS 与 GEMM]] 记录具体指令与分块基础；本篇只建立低精度硬件主线，并在需要实现时回链专项笔记，而不复制其全部内容。

## 硬件支持应写成一份完整合同

一条低精度矩阵路径至少由以下元组定义：

$$
\mathcal{H}=
(F_A,F_B,F_{\mathrm{acc}},F_D,
I_{MNK},L_A,L_B,
S_A,S_B,
\mathcal{A}),
$$

其中 $F_A,F_B$ 是输入 element formats，$F_{\mathrm{acc}}$ 是累加格式，$F_D$ 是输出格式，$I_{MNK}$ 是 instruction tile，$L_A,L_B$ 是 operand layout，$S_A,S_B$ 是可选 scaling contract，$\mathcal{A}$ 是 architecture/compute capability。

只说“GPU 支持 INT4”遗漏了最关键的信息：是 INT4×INT4 还是 INT4 storage + FP16 multiply；是否要求 signed/unsigned；累加到 INT32 还是 FP32；支持什么 $M\times N\times K$ shape；operand 从 register、shared memory 还是专用 tensor memory 读取；是否接受 zero-point；scale 是软件提前应用还是 instruction operand。

## Tensor Core 不是一个更快的 CUDA Core

普通 vector/scalar ALU 对线程持有的标量或短向量执行加减乘。Tensor Core、AMD Matrix Core 或 Gaudi MME 面向小矩阵 tile 的 multiply-accumulate：

$$
D=A B+C.
$$

硬件内部并行执行许多乘法与加法，但软件看到的是具有固定 shape、dtype、layout 和协作粒度的指令。Kernel 再把 instruction tile 组合成 warp、warp-group、CTA 和完整 GEMM。

因此矩阵核心的高吞吐来自专用 datapath 和大规模规则并行，而不是任意低精度标量运算都会自动加速。Unpack、scale statistics、softmax、layer norm、indexing 和通信仍可能运行在 vector/core、load-store 或其他单元上。

## 为什么位宽越低，峰值吞吐通常越高

固定芯片面积与功耗下，低位乘法器和加法树更小；同样宽度的数据通路每周期能携带更多 element；register、cache、SMEM 和 HBM 每个 byte也能服务更多操作。因此硬件常为 FP16/BF16、FP8、FP4 或 INT8 提供逐级更高的 matrix throughput。

但这不是数学定律，也不是对所有 instruction 的保证。实际比例取决于硬件是否复制更多乘法 lane、是否复用同一 datapath、accumulator 宽度、sparsity mode、clock/power limit 和具体 shape。FP4 峰值是 FP8 的两倍，只能被视为目标硬件上某类 matrix instruction 的上界，不能直接推导模型速度。

## NVIDIA Tensor Core 的能力谱系

下面只列与低精度知识主线相关的代表能力，不作为所有 SKU 和 instruction 的完整支持矩阵。

| 架构 | 代表性新增能力 | 对量化系统的意义 |
| --- | --- | --- |
| Volta | 第一代 Tensor Core，FP16 matrix multiply，可用 FP32 accumulate | 建立低精度 operand + 高精度累加范式 |
| Turing | 扩展低精度整数 Tensor Core，包括 INT8/更窄整数模式 | 推理量化可进入原生整数矩阵路径 |
| Ampere | BF16、TF32、改进 INT8，并引入结构化稀疏加速路径 | dtype 与 sparsity 成为独立可组合轴 |
| Hopper | FP8 E4M3/E5M2 Tensor Core、warp-group MMA 与异步数据移动增强 | FP8 训练/推理和更大协作粒度 mainloop |
| Blackwell | 第五代 Tensor Core、FP4/FP6/FP8 与 block-scaled/MX instruction、TMEM | shared scale成为显式硬件 operand |

“某代支持某格式”仍需落实到具体 compute capability、PTX version、CUDA library 和 GPU SKU。消费级与数据中心 GPU 即使架构名称相近，也可能在吞吐、容量、instruction exposure 或库支持上不同。

## 从 MMA 到 WGMMA，再到 tcgen05

### Warp-level MMA

传统 mma.sync 由一个 warp 协作执行。每个 lane 持有 A/B/C/D fragment 的一部分，lane-to-element mapping 由 instruction 规定。软件不能把任意 row-major 指针直接当作 fragment；必须通过 load、layout transform 或 ldmatrix 等机制构造正确寄存器分布。

Instruction 名称通常编码 shape、layout、A/B dtype、accumulator dtype 与饱和等属性。相同输入 dtype 在不同 shape 或 accumulation mode 下是不同合同。

### Hopper warp-group MMA

WGMMA 扩大到多个 warp 协作，并能从 shared-memory descriptors 消费 operand。更大协作范围和异步 issue 有利于让 Tensor Core 保持饱和，但也要求 warp specialization、barrier、SMEM layout 和 pipeline stage协调。硬件吞吐提高的同时，producer/consumer 调度变得更显式。

### Blackwell tcgen05 与 CTA-group MMA

Blackwell tcgen05 路径进一步引入 CTA-group 协作和 Tensor Memory（TMEM）等资源。对 block-scaled MMA，data、scale-factor tile 和 accumulator 可位于不同存储层次；scale factor必须先按规定布局进入 TMEM，instruction descriptor 还编码 element format、scale format、major mode 与 tile shape。

这说明“硬件自动处理 scale”并不表示 scale 可以是任意普通数组。硬件只自动执行合同内的乘法；scale dtype、vector size、layout 和地址仍由 quantizer、library 与 Kernel 正确准备。

## Block-scaled MMA 的数值语义

对沿 K 分块的 A、B，硬件目标可抽象为

$$
D_{ij}
=
\sum_g
\left(S^A_{i,g}A^{\mathrm{low}}_{i,g}\right)
\left(S^B_{j,g}B^{\mathrm{low}}_{j,g}\right)
+C_{ij}.
$$

CUTLASS 对 Blackwell tcgen05 的描述中，block-scaled operation 固定使用 FP32 accumulator，并把 A/B scale factor作为独立 operand；每个 scale vector覆盖规定数量的 K elements。MXFP8/6/4 和 NVFP4 可拥有不同 scale format 与 vector-size合同。

与软件 fused dequant相比，原生 block-scaled MMA可减少逐元素 convert/multiply 指令和寄存器中间值，但需要额外 scale operand bandwidth、TMEM staging 与严格 layout。硬件优化的是一种受约束的复合格式，不是任意“FP4 + scale”。

## Operand 从哪里来：低精度算力依赖数据供给

一条代表性数据路径是

$$
\text{HBM}
\rightarrow
\text{L2}
\rightarrow
\text{SMEM}
\rightarrow
\text{register/TMEM}
\rightarrow
\text{matrix unit}.
$$

每层都存在容量、带宽、延迟和对齐约束。HBM 负责大容量模型；L2 提供跨 CTA/重复请求缓存；SMEM 是软件管理的 tile staging；register/TMEM 保存 instruction fragment 与 accumulator。Tensor Core 峰值只有在 operand 持续到达时才能维持。

### 异步 copy 与多 stage pipeline

Ampere/Hopper/Blackwell 提供逐代增强的异步数据移动机制。Kernel 常使用 double/multi buffering：计算当前 tile 时预取下一 tile，通过 barrier 管理 producer 与 consumer。低精度使同一 transaction 带来更多 element，但 metadata、unpack、scale transform 也加入 pipeline。

若 load latency 无法隐藏，Tensor Core 会等待；若 stage 太多，SMEM 占用增大并降低 residency；若 accumulator 或 decode 临时值过多，register pressure 会降低 occupancy。最优 stage 数不是“越多越好”。

### Layout 是硬件带宽合同

Global-memory coalescing、SMEM bank mapping 和 fragment layout必须共同满足。一个在 HBM 中连续的 row-major tile，进入 SMEM 后可能需要 swizzle以避免 bank conflict；一个自然 scale Tensor 还可能需要按 warp-group 消费顺序重排。

Layout transform 可以离线、加载时或在 shared memory 中进行。离线 prepack 降低稳态指令，却绑定 architecture；运行时 transform 更通用，但占用 bandwidth、SMEM 和 latency。

## Storage type、conversion support 与 native MMA

判断硬件/软件“支持”某低位格式时，按以下层次核查：

| 层次 | 判断证据 | 能说明什么 |
| --- | --- | --- |
| Header/dtype 存在 | CUDA/HIP/PyTorch 定义类型 | 能承载 bit pattern |
| Convert intrinsic 存在 | cast/pack/unpack instruction 或 API | 能在设备上转换 |
| Matrix ISA 存在 | PTX/SASS/MFMA 等列出 operand combination | 有原生矩阵计算语义 |
| Library Kernel 存在 | cuBLASLt、CUTLASS、hipBLASLt 等支持 | 有可调用实现 |
| Framework dispatch 存在 | PyTorch/vLLM/TE 等选择该 Kernel | 模型可能使用 |
| End-to-end benchmark | profiler 显示目标 instruction 且性能提升 | 实际 workload 获益 |

例如 CUDA 中出现 FP4/MX basic type，不代表所有 GPU 都能执行 FP4 matrix multiply；旧 GPU 可能只能存储、转换或软件展开。相反，一个 Kernel也可能用 INT4 storage，但解码到 FP16 后调用 FP16 MMA，仍然获得 bandwidth收益，却不属于 native INT4 matrix arithmetic。

## INT8/INT4 的硬件路径

### Native integer MMA

INT8×INT8 通常累加到 INT32。它适合 W8A8，并可在 epilogue 应用 scale、bias 与 output requantization。若使用 asymmetric zero-point，额外 correction 是否被高效支持取决于 library/Kernel，而不是 INT8 Tensor Core 本身自动理解 affine mapping。

某些架构支持更窄整数 matrix modes，但真实 LLM W4A16 往往采用 packed INT4 storage + FP16/BF16 MMA，因为一个 operand仍是 16-bit activation。要确认是哪条路径，应查看生成的 instruction 与 profiler，而不是 checkpoint 名称。

### Packed bandwidth 与解码吞吐

低位 weight减少 HBM/L2 traffic，却把工作转移给 integer/vector pipeline：mask、shift、sign extension、lookup、convert 和 scale。若 Tensor Core 很快而 decode pipeline 不够，矩阵单元会吃不满。优秀 W4A16 Kernel需要平衡 global load、SMEM、vector decode 与 MMA issue，而不是只优化任一单元。

## FP16、BF16 与 TF32：不同的 range/precision折中

FP16 有较多 fraction、较窄 exponent；BF16 保留 FP32 的 8-bit exponent、减少 fraction，因此训练中更不易 overflow/underflow，但局部精度更低。二者都常以 FP32 accumulate进入 Tensor Core。

TF32 是 Ampere 引入的 Tensor Core 计算模式：输入通常从 FP32语义进入，但乘法使用缩短的 significand、保留类似 FP32 的 exponent range，并以 FP32 accumulator工作。它主要是 compute format，不应简单当作 19-bit checkpoint storage format。Storage、multiply precision 和 accumulator再次是三个不同轴。

## FP8：硬件格式仍离不开外部 scaling

Hopper 的 FP8 Tensor Core使 E4M3/E5M2 operands 可进入原生矩阵路径，但 Tensor 范围仍通常通过外部 scale适配。硬件负责有限 codebook 的 multiply-accumulate；amax history、current/delayed scaling、overflow policy和 rowwise/columnwise量化由软件 recipe决定。

FP8 的执行合同必须写完整 variant。不同厂商或历史 API 的 E4M3/E5M2 可能在 finite-only、unsigned zero、NaN 与 bias 上存在差异。即使都占 8 bit，bitstream 未必互操作。跨 NVIDIA/AMD 转换时应核对 OCP-FP8 与 FNUZ/厂商变体，而不是只比较 E/M 位数。

## FP4/MX：scale metadata 进入矩阵 datapath

FP4 element自身动态范围极窄，现代路径通常与 block scaling结合。Blackwell 支持面向 MXFP8/6/4、NVFP4等组合的 block-scaled Tensor Core instruction；AMD CDNA4/MI350 系列官方资料也列出 MXFP4、MXFP6、MXFP8 Matrix Core 支持。

跨厂商共同点是：

$$
\text{narrow element}
+\text{fine-grained scale}
+\text{higher-precision accumulate}.
$$

不同点可能包括 element variant、scale format、block/vector size、layout、instruction shape 与软件 API。OCP MX 定义数值格式基线，不要求 NVIDIA TMEM layout 与 AMD MFMA operand mapping相同。

## AMD Matrix Core 与 MFMA

AMD CDNA GPU使用 Matrix Core/MFMA 指令完成矩阵 fused multiply-add。CDNA3/MI300 系列官方资料列出 FP16、BF16、FP8 和 INT8 matrix能力；ROCm Kernel仍需选择 MFMA shape、wavefront tile、LDS layout 和 accumulator placement。

AMD wavefront通常以不同于 NVIDIA warp的协作规模和 lane mapping工作，因此不能逐字移植 PTX fragment layout。概念层可以对应：

| NVIDIA | AMD | 共同职责 |
| --- | --- | --- |
| Tensor Core MMA | Matrix Core MFMA | 小矩阵乘加 |
| Warp/warp-group | Wavefront/workgroup | 线程协作 |
| SMEM | LDS | 软件管理 tile staging |
| CUDA Core/vector pipe | VALU | 标量、向量、decode 与辅助运算 |
| cuBLASLt/CUTLASS | hipBLASLt/Composable Kernel | 高层 GEMM 与模板 Kernel |

CDNA4进一步加入 MXFP4/6/8支持，表明 microscaling正在从单一厂商格式走向多加速器计算合同。但“都支持 MXFP4”只保证共同数值语义的一部分，不能保证 checkpoint prepack 或 Kernel layout互通。

## Intel Gaudi：同名 dtype 不代表同一执行结构

Gaudi 3 官方资料列出 BF16、FP16、FP32 与 FP8 E4M3/E5M2等能力，并通过 Matrix Multiplication Engine 与 Tensor Processor Core组织矩阵和向量计算。其线程模型、memory hierarchy、编程栈与 CUDA/ROCm不同。

这一对照的意义不是在本篇穷举 Gaudi 指令，而是建立跨平台原则：格式标准化解决 codebook/scale 语义；硬件 ISA与 library解决如何执行；framework backend解决如何 dispatch。模型迁移必须分别验证三层。

## Accumulator 与内部精度

低位输入通常配更宽 accumulator：

| Input multiply | 常见 accumulator | 主要原因 |
| --- | --- | --- |
| INT8×INT8 | INT32 | 防止长 dot product整数溢出 |
| FP16/BF16 | FP32 | 扩大范围并降低累加 rounding |
| FP8 | FP32 | 保护长 reduction |
| FP4/MXFP | FP32 | element/scale很低精度，累加需更稳健 |

“FP32 accumulate”仍需问清：每次 product如何舍入；是否使用 fused multiply-add；内部是否截断；split-K partial结果以什么 dtype存储；epilogue何时 downcast。不同 instruction或 library math mode可能有不同语义。

Accumulator更宽会消耗更多 register/TMEM容量，也可能限制同时驻留 tile数量。硬件设计是在数值稳定性与资源密度之间折中，并非所有中间状态都越宽越好。

## Rounding、saturation 与特殊值由哪一层负责

从 BF16/FP32 转成低位 operand时，会发生 rounding、overflow、underflow和特殊值处理。责任可能位于：

- 独立 quantize Kernel；
- load/conversion intrinsic；
- compiler-generated cast；
- matrix instruction 的 operand解释；
- epilogue output conversion。

同一数据路径可能有多次 rounding：scale量化一次、element encode一次、乘法/累加内部一次、输出 cast再一次。硬件支持一种 dtype不自动规定整个模型使用何种 clipping或 scale selection。

整数 instruction 的饱和模式、浮点格式是否保留 infinity/NaN、subnormal是否支持或 flush-to-zero，都必须查具体 ISA/API。不能用 FP32 的 IEEE直觉替代 FP8/FP4合同。

## Structured sparsity 与低精度是独立轴

2:4 structured sparsity表示每组四个元素中满足规定数量的非零模式，并额外携带稀疏 metadata。它改变乘法数量与 operand编码；FP8/INT8/FP16描述数值格式。硬件可提供“低精度 + sparse”组合吞吐，但二者概念上独立：

$$
\text{scheme}
=
\text{numerical format}
+\text{sparsity pattern}
+\text{sparsity metadata}
+\text{instruction contract}.
$$

峰值表中 sparse throughput常按满足结构约束后的有效 operations计算，不能与 dense数字不加说明地比较。模型还要承担 pruning、metadata、质量恢复和实际稀疏 Kernel覆盖率。

## 如何正确阅读硬件峰值表

一个可信峰值数字至少要附带：

| 字段 | 需要确认 |
| --- | --- |
| Precision | FP8 variant、FP4/MX、INT8、accumulator |
| Dense/sparse | 是否假设结构化稀疏 |
| Operation count | FMA算 1 次还是 2 次 operation |
| Scope | 单 SM/CU、单 GPU、整机或机架 |
| Clock | base、boost、实际受功耗限制频率 |
| Shape | instruction/GEMM 是否足够大且对齐 |
| Data movement | 是否假设 operand已在片上并被充分复用 |

整数常用 TOPS，浮点常用 TFLOPS/PFLOPS；宣传页还可能按 sparsity把有效吞吐翻倍。比较不同厂商时必须统一计数口径，并优先使用实际 GEMM benchmark和 profiler。

## Roofline：算力提高会抬高对数据供给的要求

设峰值 matrix throughput 为 $P$，HBM bandwidth为 $B$，机器平衡点为

$$
I^*=\frac{P}{B}
\quad
\text{operations/byte}.
$$

低位矩阵单元提高 $P$，同时量化降低 operand bytes。两者都可能提升性能，但更高的 $P$ 也意味着要达到 compute-bound需要更高 arithmetic intensity。小 batch decode通常仍受 weight/KV bandwidth限制；大 GEMM更可能接近 matrix-core ceiling。

Kernel还可能受片上瓶颈限制：L2/SMEM bandwidth、bank conflict、register file、TMEM、instruction issue、decode vector pipeline或barrier。HBM Roofline只是第一层上界。

## Shape、tile 与利用率

Matrix instruction处理固定 tile。若 M/N/K不是 tile倍数，Kernel需要 padding、predicate或tail Kernel。有效利用率可粗略写为

$$
\eta_{\mathrm{shape}}
=
\frac{MNK}
{\operatorname{pad}(M)
\operatorname{pad}(N)
\operatorname{pad}(K)}.
$$

小矩阵即使完全对齐，也可能没有足够 CTA填满所有 SM/CU。大 tile提高数据复用，却减少并行 tile数量并增加SMEM/register占用。低精度格式的 group/block boundary若不匹配 K tile，还会增加scale切换与tail处理。

因此硬件优化不仅是“选择最低位宽”，还要让模型维度、batch/sequence、group size和layout匹配可用 tile。

## 从 API 到实际 instruction

使用低精度硬件通常经过多层：

$$
\text{framework op}
\rightarrow
\text{library/compiler dispatch}
\rightarrow
\text{Kernel template/DSL}
\rightarrow
\text{virtual ISA}
\rightarrow
\text{machine instruction}.
$$

在 NVIDIA 栈中可以是 PyTorch/Transformer Engine → cuBLASLt/CUTLASS/Triton → PTX → SASS；AMD 栈可以是 framework → hipBLASLt/Composable Kernel/Triton → LLVM/GCN ISA。高层 dtype请求不保证最终生成目标 matrix instruction，可能发生fallback或隐式转换。

验证路径应包括：库日志/dispatch、Kernel名称、反汇编或 profiler instruction分类、Tensor Core/Matrix Core利用率以及实际数据类型。只看 Python Tensor dtype不足以确认硬件行为。

## 硬件感知的格式选择流程

1. 固定 workload：训练/推理、prefill/decode、M/N/K、batch、sequence和复用。
2. 明确 numerical scheme：element、scale、granularity、accumulator与误差目标。
3. 查询目标 architecture的原生 instruction与合法 operand/layout。
4. 检查 library/framework是否已有稳定 Kernel，而不只看 ISA。
5. 计算 logical/physical bytes、metadata和转换成本。
6. 用 reference验证数值，再 profile HBM、matrix unit、vector/decode和occupancy。
7. 若未达到预期，定位瓶颈后再决定换格式、group、tile或fusion。

硬件原生支持应约束候选方案，但不应替代模型质量验证；算法最优scheme也必须经过硬件可执行性筛选。二者在 Kernel contract处汇合。

## 小结

GPU低精度能力不是一张 dtype清单，而是 element format、scale、instruction shape、layout、accumulator、architecture和software stack组成的合同。Tensor Core/Matrix Core加速规则矩阵乘加，辅助的quantize、decode、softmax、metadata和communication仍由其他单元与memory hierarchy承担。

NVIDIA 的演化从 Volta FP16 Tensor Core，经 Turing INT、Ampere BF16/TF32、Hopper FP8，到 Blackwell FP4/MX block-scaled MMA；AMD CDNA3提供FP8/INT8 MFMA，CDNA4进一步支持MXFP4/6/8。共同趋势是更窄element、更细shared scale和更宽accumulator，但具体layout与ISA仍由厂商定义。

下一篇将把低精度数据路径扩展到多GPU：量化 communication buffer、error feedback、collective、tensor/expert parallel，以及HBM、NVLink/Infinity Fabric和网络bandwidth之间的权衡。

## 自检问题

1. “CUDA里存在FP4 dtype”和“GPU原生执行FP4 MMA”为什么是两种能力？
2. W4A16为什么可能只使用FP16 Tensor Core，却仍从INT4获益？
3. Block-scaled MMA自动应用scale，为什么quantizer仍需生成特殊scale layout？
4. Warp MMA、WGMMA和CTA-group tcgen05在协作与operand位置上有何变化？
5. 更低位matrix throughput提高后，为什么Kernel反而更容易受数据供给限制？
6. FP32 accumulator保护了什么，又不能恢复什么？
7. TF32为什么更接近compute format，而不是普通checkpoint storage dtype？
8. OCP MXFP4数值兼容为什么不保证NVIDIA与AMD prepacked bytes兼容？
9. Sparse FP8峰值为什么不能与dense FP8数字直接比较？
10. 如何证明framework请求的FP8实际落到了目标matrix instruction？

## 来源与适用边界

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 与 [CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/)：用于CUDA数值类型、compute capability、浮点行为和Tensor Core背景；具体低位matrix合同以PTX/library文档为准。
- [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)：用于mma、wgmma、tcgen05、operand dtype/layout、block scaling和architecture要求。PTX文档会随CUDA更新，部署必须固定版本。
- [CUTLASS tcgen05 MMA Programming Guide](https://docs.nvidia.com/cutlass/4.5.2/media/docs/pythonDSL/mma_docs/tcgen05_programming.html) 与 [CUTLASS Blackwell functionality](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)：用于Blackwell CTA-group、TMEM、FP32 accumulator和block-scaled MMA的数据/scale路径。
- [NVIDIA GPU Performance Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html) 与 [Matrix Multiplication Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)：用于Tensor Core历史、算术强度、shape/alignment和峰值解释原则。
- [AMD MI300 Series microarchitecture](https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300.html) 与 [ROCm precision support](https://rocm.docs.amd.com/en/latest/reference/precision-support.html)：用于CDNA3 Matrix Core的FP8/INT8能力和ROCm dtype/library层次。
- [AMD CDNA Architecture](https://www.amd.com/en/technologies/cdna.html) 与 [CDNA 4 Architecture Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf)：用于CDNA4/MI350的MXFP4、MXFP6、MXFP8支持；厂商峰值不作为实际模型性能保证。
- [Intel Gaudi 3 AI Accelerator White Paper](https://cdrdv2-public.intel.com/817486/gaudi-3-ai-accelerator-white-paper.pdf)：用于Gaudi 3的FP8与矩阵/张量处理结构对照；本文未展开其完整ISA。
- [OCP 8-bit Floating Point Specification](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf) 与 [OCP MX v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)：用于区分标准数值语义与厂商执行布局。

本文没有运行GPU microbenchmark、反汇编或profiler。架构表是代表能力谱系，不是完整SKU支持矩阵；任何吞吐、instruction和library可用性都应按目标GPU、driver、CUDA/ROCm和library版本重新核查。
