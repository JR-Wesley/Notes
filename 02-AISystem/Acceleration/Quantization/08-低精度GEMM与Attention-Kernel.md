# 低精度 GEMM 与 Attention Kernel

前一篇把量化 Tensor 拆成 codes、metadata 与 schema。本篇继续追踪这些 bytes 如何进入计算：

$$
\text{load}
\rightarrow
\text{unpack}
\rightarrow
\text{decode/scale}
\rightarrow
\text{MMA}
\rightarrow
\text{accumulate}
\rightarrow
\text{epilogue/store}.
$$

这里最重要的区分是：量化格式描述“数值是什么”，Kernel 描述“怎样移动并计算这些数值”。同一 INT4 scheme 可以由先完整反量化再调用 GEMM 的朴素实现消费，也可以由 fused weight-only Kernel 消费；两者数值近似相同，HBM traffic 和执行效率却完全不同。

通用 GEMM 分块、MMA fragment 与 CUDA pipeline 的基础连接到 [[../../GPU/tensor-core/CuTe/cutlass与GEMM|CUTLASS 与 GEMM]] 和 [[../../GPU/tensor-core/MMA|MMA]]。本篇只解释低精度表示给这些结构增加了哪些数据路径和约束。

## 从数学 GEMM 到量化 GEMM

普通矩阵乘写成

$$
C_{ij}=\sum_{k=0}^{K-1}A_{ik}B_{jk}.
$$

若 A、B 采用 symmetric affine quantization，且 scale 分别按输出行/列共享，

$$
A_{ik}\approx S^A_i q^A_{ik},
\qquad
B_{jk}\approx S^B_j q^B_{jk},
$$

则

$$
C_{ij}
\approx
S^A_iS^B_j
\sum_k q^A_{ik}q^B_{jk}.
$$

因为 scale 不随 k 变化，它可以移到 dot product 外，在 epilogue 中应用。这是 per-row activation、per-output-channel weight 与整数 GEMM 容易配合的原因之一。

若 scale 沿 K 方向每个 block 都不同，

$$
C_{ij}
\approx
\sum_g
S^A_{i,g}S^B_{j,g}
\sum_{k\in g}q^A_{ik}q^B_{jk}.
$$

Scale 不能再整体移到完整 K reduction 之后；Kernel 必须在每个 K block 的局部 dot product 上应用 scale，然后再累加。这会改变 mainloop，而不只是 epilogue。Block-scaled MMA 正是把这种局部 scaling 纳入硬件/指令合同。

## Kernel 的三个数值域

| 数值域 | 存在位置 | 典型 dtype | 主要职责 |
| --- | --- | --- | --- |
| Storage domain | HBM/L2/SMEM 中的 operand | packed INT4、INT8、FP8、FP4、MX codes | 降低容量和 bandwidth |
| Multiply domain | MMA 输入或解码后的寄存器 fragment | INT8、FP8、FP16/BF16、block-scaled FP | 执行乘法 |
| Accumulator domain | MMA accumulator/register/TMEM | INT32、FP32，有时 FP16 | 控制长 reduction 的范围与误差 |

Storage dtype 不必等于 multiply dtype。W4A16 weight-only Kernel 常从 HBM 加载 packed INT4，在寄存器中恢复到 FP16/BF16 fragment，再执行 16-bit Tensor Core MMA；INT4 在这条路径上主要压缩 storage/bandwidth，不表示硬件执行 INT4×FP16 乘法。

原生 INT8 GEMM 则可以让 storage 和 multiply 都是 INT8，并在 INT32 中累加。FP8 GEMM 可让 operands 保持 FP8 codebook 语义进入 Tensor Core，以 FP32 累加。原生 block-scaled FP4/MX 路径还会把 shared scale 作为显式 operand 送入矩阵指令。

## Mainloop：低精度数据路径发生在哪里

高性能 GEMM 把输出矩阵切成 CTA、warp-group/warp 和 instruction tiles，并沿 K 轴迭代。典型 mainloop 同时进行：

1. 从 HBM/L2 异步加载下一块 A、B 和必要 metadata；
2. 将当前 tile 放入适合计算的 shared-memory/register layout；
3. 对 packed code 做 unpack、sign extension、lookup 或 scale；
4. 发出 MMA，并在更宽 accumulator 中累加；
5. 用多 stage pipeline 覆盖 load、decode 与 compute latency。

量化带来的额外工作主要是 unpack/convert、scale load、scale association 和可能的 zero-point correction。优化目标不是让这些工作消失，而是让它们与原本等待 memory 或 Tensor Core 的周期重叠，并避免生成完整的高精度中间 Tensor。

### Prologue、mainloop 与 epilogue 的边界

Prologue 通常指进入主 MMA 循环前的准备，例如动态选择 scale、量化 activation 或 prefetch 第一批 tile。Mainloop 反复加载 K tile 并执行 MMA。Epilogue 将 accumulator 乘 scale、加 bias、执行 activation/residual，并转换或重新量化输出。

不同库对这些名称的具体边界并不完全相同，但分析时可以问：某一步只执行一次、每个 K block 执行一次，还是每个输出 element 执行一次？把本应每个输出 tile 执行的 scale 错放到每个 element，或把随 K block 变化的 scale错误推迟到 epilogue，都会造成性能或数值错误。

## Materialized dequantization 为什么常常浪费 bandwidth

假设 packed INT4 权重占 $N/2$ byte。若先完整恢复为 FP16 权重：

$$
W_{\mathrm{INT4}}
\rightarrow
W_{\mathrm{FP16}}
\rightarrow
\text{HBM}
\rightarrow
\text{GEMM},
$$

除了读取约 $N/2$ byte 与 metadata，还要写入并再次读取 $2N$ byte 的 FP16 Tensor。仅中间 Tensor 就引入约 $4N$ byte HBM traffic，可能完全盖过原本的压缩收益。

Fused dequantization 改为

$$
\text{packed codes from HBM}
\rightarrow
\text{register fragment}
\rightarrow
\text{MMA},
$$

恢复后的值只在寄存器或短暂 shared-memory stage 中存在，不回写为完整 Tensor。Fusion 的核心收益是消除中间 materialization 和 Kernel launch，而不仅是把两段源码写进一个函数。

## Fusion 的四种层次

| 层次 | 做法 | 消除的开销 | 仍存在的成本 |
| --- | --- | --- | --- |
| Separate dequant + GEMM | 先生成高精度 Tensor | 无 | 完整中间读写与额外 launch |
| Fused load/dequant | 加载 packed operand 后在寄存器解码 | 高精度中间 Tensor | unpack、convert、scale load |
| Native low-precision MMA | code 直接作为 MMA operand | 大部分软件 decode | 格式转换、metadata、accumulate |
| Producer-consumer fusion | 前一算子直接生成 consumer 所需低精度 tile | 额外 quantized Tensor pass 的一部分 | 统计、同步、生命周期限制 |

并非 fusion 越多越好。过大的 fused Kernel 可能提高寄存器压力、降低 occupancy、阻碍 compiler scheduling，或无法复用已生成的 quantized activation。是否融合应依据 bytes saved、复用次数、tile 生命周期和资源占用判断。

## Weight-only W4A16：带宽压缩而非纯整数计算

W4A16 通常表示 weight 以 4 bit 存储，activation 使用 FP16/BF16；它没有说明 4-bit codebook、group size、scale dtype、packing 或 accumulator。典型 Kernel 数据路径为：

$$
B_W
\rightarrow
\text{vector load}
\rightarrow
\text{nibble unpack}
\rightarrow
\text{signed/codebook decode}
\rightarrow
\times S_g
\rightarrow
\text{FP16/BF16 fragment}
\rightarrow
\text{MMA}.
$$

这条路径的优势来自每次从 HBM 搬运更多权重，并在使用前就地恢复。代价是 vector/ALU pipeline 需要完成 bit manipulation 和 conversion。Marlin 一类 Kernel 会离线 reshuffle weights 与 group scales，使 global/shared load 和反量化结果直接匹配 Tensor Core fragment，同时用 pipeline 将 dequantization 与 MMA 交错。

### 为什么 decode batch 更适合 weight-only

自回归 decode 每步 token 数较少，线性层对每个加载的 weight 复用次数低，容易受 weight bandwidth 限制。把 BF16 weight 压到 INT4 可显著减少权重 bytes。Prefill 或大 batch 下，同一 weight tile 被更多 activation 行复用，算术强度上升，decode/unpack 的额外指令更容易成为瓶颈，理论 4 倍压缩不再对应 4 倍速度。

因此 W4A16 的性能必须随 batch/token rows 报告。Kernel 在 batch=1 很快，不代表在 prefill 或 continuous batching 下仍是最佳选择。

## W8A8：量化 activation 后进入整数矩阵路径

若 activation 按 row/token scale，weight 按 output channel scale：

$$
A_{ik}\approx S^A_i(q^A_{ik}-Z^A_i),
\qquad
B_{jk}\approx S^B_j(q^B_{jk}-Z^B_j).
$$

在 symmetric $Z=0$ 时，Kernel 可以执行 INT8×INT8→INT32 dot product，再在 epilogue 乘 $S^A_iS^B_j$。与 W4A16 相比，它减少两侧 operand bytes，并使用整数 Tensor Core；但 dynamic activation quantization 需要先求 amax/range、生成 scale 和 INT8 codes。

若 activation 只被一个 GEMM 消费，独立 quantize Kernel 写入 HBM 再读回可能抵消收益。若同一 activation 被 Q/K/V 三个 projection 复用，量化一次后供多个 consumer 读取，materialized INT8 Tensor 可能更值得。系统决策必须把 consumer fan-out 纳入考虑。

## Asymmetric zero-point 为什么让 Kernel 更复杂

整数 dot product 展开为

$$
\sum_k(q^A_k-Z_A)(q^B_k-Z_B)
=
\sum_k q^A_kq^B_k
-Z_B\sum_kq^A_k
-Z_A\sum_kq^B_k
+KZ_AZ_B.
$$

除主 dot product 外，还出现 row/column sums 和常数修正。如果某个 operand 是 static weight，其 sums 可以离线预计算；dynamic activation 的 sums 仍需运行时获得。某些 instruction 或 library 会融合 correction，另一些只优化 symmetric path。

Zero-point 提升偏斜分布的码字利用率，却增加 metadata、correction arithmetic 和实现分支。这不是 asymmetric 一定慢，而是性能结论必须落实到具体 Kernel contract。

## Accumulator precision：低位输入不意味着低位累加

Dot product 会累积 $K$ 项。INT8 operands 的单项最坏幅值约为 $127^2=16129$；若所有项同号，INT32 理论上在

$$
K\lesssim
\frac{2^{31}-1}{16129}
\approx133144
$$

时才不溢出。真实分布通常远未达到该最坏界，但 zero-point correction、split-K reduction 和极端输入仍需纳入范围分析。

FP8/FP4 输入通常使用 FP32 accumulator，以降低长 K reduction 的 rounding error并扩大范围。即使 accumulator 是 FP32，乘法输入已发生的量化误差不会恢复；若硬件采用有限精度的内部乘加路径，也需按具体 instruction 文档核对。

Accumulator 最终还要经过 epilogue 转成 FP16/BF16、INT8 或下一种低精度格式。Output cast/requantization 是新的量化边界，会引入自己的 scale、rounding、saturation 和 metadata。

## Scale 在 Kernel 中的四种应用位置

| Scale granularity | 能否移出 K reduction | 常见位置 |
| --- | --- | --- |
| Per-tensor | 可以 | GEMM 前 normalize 或 epilogue |
| A per-row、B per-column | 可以 | accumulator epilogue 的 outer product |
| Per-K-block | 不可以 | 每个 K block 局部 dot/MMA 前后 |
| Per-element | 通常不可以 | decode path，成本最高 |

对于 block scale，若先把 code 解码为 FP16 fragment再乘 scale，属于软件 fused dequant；若 MMA instruction 同时接收 code 和 scale，属于 native block-scaled MMA。两者可能实现相同 reconstruction 公式，但指令数、寄存器和 scale layout 不同。

## FP8 GEMM：element exponent 与外部 scale同时存在

Tensor-wise或 row-wise FP8 GEMM 可抽象为

$$
C
\approx
S_A S_B
\operatorname{GEMM}
(Q_A^{\mathrm{FP8}},Q_B^{\mathrm{FP8}}),
$$

其中每个 FP8 element 自带 exponent/fraction，外部 scale 把 Tensor 分布移动到适合的 FP8 range。Tensor/row scale 可在 epilogue 应用；若采用 block-wise FP8，scale 可能随 K block 变化并进入 mainloop。

Native FP8 MMA 避免把所有 operand materialize 为 FP16，但仍需要 BF16/FP16→FP8 conversion、scale statistics、可能的 rowwise/columnwise副本和输出转换。训练中还可能同时保留高精度 master weights。评价 FP8 Kernel 时要明确测量的是单个 GEMM、Transformer layer 还是包含 conversion 的完整 step。

## FP4/MX 与 native block-scaled MMA

对 MXFP4，局部计算可写成

$$
C_{ij}
\approx
\sum_g
S^A_{i,g}S^B_{j,g}
\sum_{k\in g}
\operatorname{E2M1}(q^A_{ik})
\operatorname{E2M1}(q^B_{jk}).
$$

Blackwell block-scaled MMA 把 data codes 与 scale vectors 作为不同 operand path，并沿 K 方向按规定 vector size 应用 scale，使用更高精度 accumulator。CUTLASS 还要求 scale 按硬件消费顺序 swizzle。这减少软件逐元素 dequant 指令，却把格式、scale vector size 和 layout 绑定得更紧。

原生支持不等于零转换成本。Producer 可能仍需统计、生成 E8M0/E4M3 scale、编码 FP4、创建 rowwise/columnwise layout，并处理 padding。只有这些准备成本被复用或融合时，低位 MMA 峰值才可能转化为端到端收益。

## Epilogue：量化 GEMM 的下一条数据边界

Epilogue 从 accumulator 生成输出，可能组合：

$$
D=
Q_{\mathrm{out}}
\bigl(
\alpha C+\beta C_0+\mathrm{bias}+\mathrm{residual}
\bigr).
$$

其中 $Q_{\mathrm{out}}$ 可以是 cast，也可以是带新 scale 的量化。把 bias、activation、residual 和 requantization 融入 epilogue，可以避免输出先写高精度 HBM 再由下一 Kernel读取。

但 output scale 若依赖当前 tile/global amax，就需要 reduction。Per-row scale 可以由每行协作产生；per-tensor scale 可能需要跨 CTA reduction或延迟策略。动态量化输出的统计依赖会限制单 pass fusion。

## Attention 不是一个 GEMM，而是一条数值流水线

Scaled dot-product attention 为

$$
O=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d}}
\right)V.
$$

其中至少有四种数值角色：

1. Q/K/V projection 的 GEMM operands；
2. $QK^\top$ 的 score accumulator；
3. softmax 的 max、exp、sum 与 probability；
4. probability 与 V 的乘加以及最终输出。

这些对象的动态范围和误差敏感度不同。Q、K、V 可以低精度存储或乘法；score 通常需要更宽 accumulator；softmax reduction 对 overflow/underflow 敏感，常在 FP32 中维护 max/sum；输出可以再转为模型主 dtype。将整条 attention 标成“FP8”会掩盖这些内部精度边界。

## FlashAttention 的 IO 思路如何与量化结合

朴素 attention 会 materialize $N\times N$ score/probability matrix。FlashAttention 类算法按 Q/K/V tiles 在线维护 softmax statistics，使大矩阵不落 HBM，核心收益来自减少 IO，而不是改变 attention 数学结果。

低精度可以进一步减少 Q/K/V 或 KV cache 的读取 bytes，但不能破坏在线 softmax 的数值不变量。一个 fused quantized attention tile 可能执行：

$$
\text{load packed K/V + scale}
\rightarrow
\text{decode tile}
\rightarrow
QK^\top
\rightarrow
\text{online softmax update}
\rightarrow
PV\text{ accumulation}.
$$

如果先把整个 KV cache 反量化回高精度 HBM，再调用 FlashAttention，就重新引入了本应避免的 IO。理想路径是在 K/V tile 被加载时解码，并让恢复值只存在于 register/SMEM。

## KV cache 量化改变 Attention 的 load path

Decode 阶段，每个新 query 都要读取历史 K/V。随着 context length 增长，KV bytes 与读取流量线性增加，因此 KV quantization 同时影响容量和 bandwidth。对 paged KV cache，逻辑 token 到物理 page/block 的映射之后，还需定位量化 block 和对应 scale：

$$
(\text{sequence},\text{head},\text{token},d)
\rightarrow
\text{page}
\rightarrow
\text{code offset}
\rightarrow
\text{scale offset}.
$$

若 scale 粒度与 page、head dimension tile 或 vector load 不对齐，Kernel 会增加地址计算和不规则 metadata load。KV format 因而需要联合决定 page size、K/V layout、head grouping、scale granularity 与 attention tile。

### K 与 V 为什么可能采用不同策略

K 通过 $QK^\top$ 影响 softmax logits，小 score 误差可能被 softmax 放大或改变注意力排序；V 的误差在 probability 加权后进入输出。二者的敏感方向并不相同。某些方案对 K 使用 per-channel 统计、对 V 使用 per-token 统计，或让两者使用不同 bit/scale；这些选择必须由模型评测和 Kernel layout共同验证。

### Prefill 与 decode 的权衡不同

Prefill 同时处理许多 query token，attention 具有较高并行度，转换/解码指令可能更显眼。Decode 通常只有少量新 query，却读取长历史 KV，更容易受 KV bandwidth 和访问延迟限制。因此同一种 KV format 可能在 decode 显著受益，却在短 prompt/prefill 中收益较小。

## Paged Attention 中 metadata 不能成为随机访问放大器

Paged Attention 已经需要 page table lookup。若每个很小 group 都单独读取 scale，metadata address 和 cache miss 可能放大随机访问成本。常见优化方向是让一个 page/tile 内的 scales 连续、让 scale与 data 的索引可由简单位运算或整除推导，并在 warp 内复用已加载 metadata。

把 scale 与 codes interleave 可能改善单 block locality，但也可能破坏宽向量 data load；SoA 能让 data/scale 分别合并加载，却要求两个地址流。不存在脱离具体 tile 与 cache 行的通用最优布局。

## Kernel dispatch 是 Quantization Scheme 的现实约束

Runtime 选择 Kernel 时至少需要匹配：

| 维度 | 示例 |
| --- | --- |
| Operand formats | W4A16、W8A8、FP8×FP8、MXFP4×MXFP4 |
| Granularity | per-channel、per-token、group 128、block 32 |
| Layout | row/column major、packed order、pre-swizzled scale |
| Shape constraints | K/N alignment、supported tile、tail handling |
| Device | architecture、instruction availability、shared/register capacity |
| Epilogue | output dtype、bias、activation、requantization |

不匹配时，Runtime 可能 fallback 到较慢 Kernel、先转 layout、先 dequantize，甚至拒绝执行。模型文件“可以加载”不代表其量化路径被高效支持。Benchmark 前应确认实际 dispatch 的 Kernel，而不是根据配置名推断。

## Roofline：压缩 bytes 后瓶颈会移动

对某个算子，

$$
T\gtrsim
\max
\left(
\frac{\text{bytes moved}}{\text{memory bandwidth}},
\frac{\text{operations}}{\text{compute throughput}}
\right).
$$

Weight quantization 减少 bytes，但增加 unpack/scale operations；native low-precision MMA 提高 compute throughput，却可能让 quantize、layout transform 或 epilogue 成为新瓶颈。优化后性能不再受原瓶颈限制是正常现象，不代表量化失败。

应把 bytes 分成 weight/activation/KV codes、metadata、中间 Tensor 和输出，把 operations 分成 MMA、decode、statistics、correction 与 reduction。只有这样才能解释同一格式为何随 batch、K、sequence length 和 fusion 状态表现不同。

## 两条完整 Kernel 数据路径

### W4A16 decode Linear

$$
\begin{aligned}
&X_{\mathrm{BF16}}[B,K],\quad
W_{\mathrm{INT4}}[M,K],\quad
S_W[M,\lceil K/G\rceil]\\
&\rightarrow
\text{CTA tile selection}\\
&\rightarrow
\text{async load X, packed W, scales}\\
&\rightarrow
\text{unpack/decode W tile in registers}\\
&\rightarrow
\text{BF16/FP16 MMA with FP32 accumulate}\\
&\rightarrow
\text{bias/activation/output cast in epilogue}.
\end{aligned}
$$

这条路径主要节省 weight bandwidth。检查重点是 weight prepack、scale 与 K group 对齐、decode pipeline 是否遮蔽、寄存器压力和 batch-dependent arithmetic intensity。

### Quantized KV decode Attention

$$
\begin{aligned}
&Q_{\mathrm{main}},\quad
(B_K,M_K),\quad(B_V,M_V),\quad
\text{page table}\\
&\rightarrow
\text{locate K/V pages and metadata}\\
&\rightarrow
\text{load/decode K tile}\\
&\rightarrow
QK^\top\text{ accumulation}\\
&\rightarrow
\text{online softmax in wider precision}\\
&\rightarrow
\text{load/decode V tile}\\
&\rightarrow
PV\text{ accumulation and output cast}.
\end{aligned}
$$

这条路径主要节省长期驻留容量和每步 KV read bandwidth。检查重点是 page/tile/group 对齐、K/V scale association、softmax precision、长 context 下 metadata traffic 与真实 attention quality。

## Benchmark 与正确性检查

### 数值检查

先比较 fused Kernel 与“按同一格式 reference dequantize + high-precision op”，这能隔离 Kernel/layout bug；再比较 quantized result 与原始高精度模型，后者才包含量化算法误差。测试应覆盖非对齐 shape、尾 group、极值、zero、NaN/Inf policy 和不同 batch/sequence。

### 性能检查

至少记录 GPU/architecture、library/commit、dtype 与完整 scheme、M/N/K 或 batch/head/sequence、warmup、workspace、是否包含 quantize/prepack、实际 dispatch Kernel、latency 分位数和 achieved bandwidth/throughput。Static weight prepack 应单独报告一次性成本；dynamic activation/KV quantize 则属于稳态路径。

### 资源检查

用 profiler 观察 HBM/L2/SMEM traffic、Tensor Core 与 vector pipeline利用率、register/SMEM 占用、occupancy、stall reason 和 launch 数。若 fused Kernel 比预期慢，应先判断是 memory、decode、MMA、metadata、epilogue 还是 occupancy 瓶颈，而不是只调整 tile size。

## 小结

低精度 Kernel 的本质是让压缩表示尽可能晚地解码、让高精度中间结果尽可能短命，并让 metadata 与计算 tile 对齐。W4A16 主要通过压缩 weight bandwidth 获益；W8A8/FP8 进一步改变 multiply domain；MX/FP4 原生 block-scaled MMA 把局部 scale纳入指令；Attention 则需要把量化 load 与 online softmax、paged KV layout共同设计。

Fusion 真正消除的是中间 HBM traffic、重复 layout transform 和 Kernel launch，但它会增加寄存器、调度和实现约束。最终性能由 storage bytes、decode/quantize work、MMA throughput、accumulator/epilogue 和 workload shape共同决定，不能由 bit width 或硬件峰值单独推断。

下一篇将进入 GPU 硬件层，按 architecture、instruction、operand format、accumulator、throughput 和数据移动单元，系统说明 Tensor Core/Matrix Core 如何执行 INT8、FP8、FP4/MX。

## 自检问题

1. 为什么 W4A16 常是 4-bit storage，却不是 INT4×FP16 原生乘法？
2. 什么条件下 scale 可以移到完整 K reduction 后的 epilogue？
3. Materialized dequantization 比 fused dequantization 多出哪些 HBM traffic？
4. Dynamic activation quantization 何时适合生成可复用的 INT8 Tensor，何时更适合与 consumer 融合？
5. Asymmetric zero-point 为什么会引入 row/column sum correction？
6. FP32 accumulator 为什么不能消除 FP8/FP4 operand 已产生的量化误差？
7. Native block-scaled MMA 省掉了什么，又增加了哪些 layout/metadata 约束？
8. 为什么 FlashAttention 与 KV quantization 必须在 tile load path 上结合？
9. Prefill 与 decode 为什么可能偏好不同的量化 Kernel？
10. Runtime fallback 到 dequantize-then-GEMM 时，模型仍能运行为什么不代表量化优化成功？

## 来源与适用边界

- [NVIDIA CUTLASS GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html) 与 [CUTLASS Arguments and Operands](https://docs.nvidia.com/cutlass/latest/media/docs/operators/api_reference/arguments.html)：用于 GEMM mainloop/epilogue 分层、scaled operand、scale mode 与 Blackwell scale layout。
- [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)：用于区分 MMA operand、block scaling、accumulator 与 instruction contract；具体合法组合必须固定 PTX/CUDA 版本和 compute capability。
- [Marlin repository](https://github.com/IST-DASLab/marlin) 与 Elias Frantar et al., [Marlin: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models](https://arxiv.org/abs/2408.11743)：用于 W4A16 的 offline reshuffle、fused dequantization、pipeline 和 batch-dependent weight bandwidth分析。
- Tri Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)：用于 attention tiling、online softmax 与避免 materialize score matrix 的 IO 原理；本文不把原论文性能数字外推到量化实现。
- Zirui Liu et al., [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750)：用于 K/V 具有不同量化敏感方向以及量化 KV cache 的研究背景；具体 per-channel/per-token recipe 是论文方案，不是通用标准。
- [OCP Microscaling Formats (MX) Specification v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)：用于 MX element、E8M0 block scale和 block-32 数值语义；GPU mainloop 与 scale swizzle 属于厂商实现合同。

本文没有运行 CUDA Kernel、模型推理或 profiler。公开检索在本轮出现连接失败，因此未写入未经现有一手资料支持的最新框架支持矩阵或性能数字；实现时仍需按目标 GPU、CUDA/library 版本和源码 commit 重新核查。
