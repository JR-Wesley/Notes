# FP8、FP4 与 Microscaling 格式

“FP8”或“FP4”只描述一个低位浮点 element format 家族，不能单独定义一条可执行的 Tensor 数据路径。真正部署时还需要回答：采用哪个 encoding variant，是否有外部 scale，scale 由多少元素共享、用什么 dtype 存储，如何 rounding 和 saturation，数据怎样 packing，以及硬件能否直接把 data 与 scale 送入矩阵指令。Microscaling（MX）进一步把低位 element 与细粒度 shared scale 绑定成规范化复合格式。本篇沿 [[00-低精度量化知识地图]] 的统一模型，区分 element codebook、Tensor scaling recipe、存储布局和硬件执行合同。

## 先区分四个容易混淆的层次

| 层次 | 回答的问题 | 例子 |
| --- | --- | --- |
| Element format | 单个 bit pattern 怎样解码？ | OFP8 E4M3、OFP8 E5M2、FP4 E2M1 |
| Scaling scheme | 一组 element 怎样借助外部 scale 表示更宽值域？ | tensor-wise FP8 scaling、block scaling、MX |
| Tensor/storage format | element、scale、padding 与 layout 怎样共同存储？ | packed FP4 + scale Tensor、MX scale swizzle |
| Compute contract | 指令怎样读取 operand、scale 并累加？ | dequantize-to-BF16 GEMM、FP8 MMA、block-scaled MMA |

OCP OFP8 规范主要定义 element encoding；OCP MX 规范定义 element format、32-element block 和 E8M0 scale 的组合。NVIDIA Transformer Engine 的 FP8/MXFP8/NVFP4 recipe 则进一步规定实际训练中的 scale 计算、rowwise/columnwise 副本和执行流程。Blackwell `tcgen05.mma` 又规定硬件可消费的 scale vector、物理 layout 与 accumulator。把这四层合称“某种 FP4”会丢失绝大多数系统信息。

## 低位浮点仍然是非均匀 codebook

对具有 sign、exponent 和 fraction 的低位浮点 element，正规数仍可写为

$$
v=(-1)^s2^{E-\mathrm{bias}}\left(1+\frac{F}{2^m}\right).
$$

Exponent 让 codebook 的绝对间距随数量级增大，因此它不是传统 affine INT 的等间距 lattice。在同一 binade 内，相邻值间距固定；跨越 exponent 边界后间距成倍变化。更多 exponent bits 通常换来更宽 dynamic range，更多 fraction bits则换来更密的局部 codebook。总位宽很低时，一个 bit 的分配就会显著改变行为。

外部 scale $S_g$ 将 element codebook 放大到 Tensor 的真实单位：

$$
q_i=\operatorname{encode}_{F}(x_i/S_{g(i)}),
\qquad
\hat{x}_i=S_{g(i)}\operatorname{decode}_{F}(q_i).
$$

这里 $F$ 是 element format，$g(i)$ 是分组函数。即使 element 自身是浮点数，整个过程依然属于量化：normalize、有限 codebook encoding、rounding、可能的 saturation，以及依赖 metadata 的 reconstruction 都没有消失。

## OFP8：E4M3 与 E5M2

OCP 8-bit Floating Point（OFP8）Revision 1.0 定义 E4M3 和 E5M2 两种 8-bit encoding。二者都有 1 个 sign bit，但以不同方式在 range 与 precision 之间分配剩余位。

| 属性 | OFP8 E4M3 | OFP8 E5M2 |
| --- | ---: | ---: |
| 字段 | S1E4M3 | S1E5M2 |
| Exponent bias | 7 | 15 |
| Significand precision | 4 bit | 3 bit |
| 最大有限幅值 | 448 | 57,344 |
| 最小正规正数 | $2^{-6}$ | $2^{-14}$ |
| 最小正 subnormal | $2^{-9}$ | $2^{-16}$ |
| Infinity | 不支持 | 支持 |
| NaN | 仅保留少量编码 | 使用 exponent 全 1 的 NaN 编码 |
| 设计倾向 | 更高局部精度 | 更宽动态范围 |

E4M3 为扩大有限范围，取消 infinity，并把大部分 exponent 全 1 的 pattern 用作有限值；最大幅值因此达到 448。E5M2 更接近 IEEE 特殊值约定，保留 infinity 与 NaN，最大有限幅值为 57,344。E5M2 的 exponent 比 E4M3 多 1 bit，覆盖范围宽得多，但 fraction 少 1 bit，在同一数量级内的量化间隔更大。

这解释了常见训练 recipe 为什么倾向于前向采用 E4M3，而对动态范围更宽的梯度采用 E5M2。不过，这只是具体 recipe 的选择，不是 OFP8 标准要求。OFP8 element specification 也不规定 Tensor scale、scale history、accumulator dtype、GEMM layout 或训练收敛策略。

此外，名字相同不保证 encoding 完全相同。PyTorch 等生态还能看到 `e4m3fn`、`e4m3fnuz`、`e5m2fnuz` 等变体，其中 finite-only、unsigned-zero、NaN 与 bias 规则可能不同。必须把完整 dtype 名称或规范版本视为 codebook ID；不能把任意 `E4M3` 的 max value 都默认写成 448。

## Tensor-wise FP8 scaling

模型 Tensor 的数值范围未必自然适配 FP8。常见做法是为整个 Tensor 或某个较大区域选择 scale：

$$
S=\frac{a_{\max}}{v_{\max}},
\qquad q_i=\operatorname{encode}_{\mathrm{FP8}}(x_i/S),
$$

其中 $v_{\max}$ 是目标 FP8 variant 的最大有限幅值。实际实现会加入 margin、power-of-two rounding、epsilon、history 或 saturation policy。Current scaling 使用当前 Tensor statistics；delayed scaling 根据历史 amax 产生后续 scale。Element codebook 相同，但 parameter lifecycle 不同。

Tensor-wise scale 的 metadata 很少，却让整个 Tensor 受同一个最大值约束。若少数 outlier 把 $a_{\max}$ 拉高，大量普通值会被压到 FP8 codebook 靠近零的稀疏区域。E5M2 能用更宽 element range缓和 overflow，却牺牲局部 precision；更细粒度 block scale 则从 granularity 方向解决问题。

普通 FP8 recipe 不等于“裸 FP8 cast”。例如一个系统可能把 BF16 activation 除以 FP32 scale，编码成 E4M3，在 Tensor Core 中执行 FP8 multiply 并用 FP32 accumulate，最后再缩放输出。Data 是 1 byte/element，但还存在 scale、转换 Kernel、可能的高精度 master weight 和输出 dtype。评价收益时必须区分临时 GEMM operand、持久模型存储和训练状态。

## Block scaling、block floating point 与 microscaling

**Block scaling** 是上位概念：一小组 element 共享外部 scale。若 block $g$ 的 scale 为 $S_g$，则

$$
\hat{x}_i=S_gv_i,\qquad i\in g,
$$

其中 $v_i$ 是 block 内 element code。缩小 block 可让 scale 更贴近局部范围，但会增加 metadata、scale bandwidth 和 layout 约束。

**Block Floating Point（BFP）** 通常让一个 block 共享 exponent，而每个 element 保存 sign 与定点式 significand。它可理解为 $S_g=2^{e_g}$ 乘以一个固定点 element codebook；block 内 element 不再各自拥有完整 exponent。

**Microscaling（MX）** 是 OCP 标准化的细粒度 block-scaled 格式族。MX 使用 32 个 element 的 block 和一个 E8M0 shared scale。对 MXFP 格式，block 内每个 element 仍然拥有自己的低位 exponent 与 fraction；因此把 MXFP4 简称为“32 个数共享一个 exponent”并不精确。更准确的说法是：它们共享一个 power-of-two block scale，同时每个 FP4/FP6/FP8 element 仍有局部浮点 codebook。MXINT8 的 element codebook 与 MXFP 又不同。

因此三者关系是：BFP 和 MX 都属于 block scaling；某些 BFP 可表示为 shared power-of-two scale 加定点 element；MXFP 则是 shared power-of-two scale 加低位浮点 element。Shared exponent 是一种实现结构，shared scale 是更一般的数学描述。

## E8M0：MX 的 shared scale

OCP MX v1.0 使用 8-bit E8M0 作为 block scale。它没有 sign 和 fraction，有限 scale 是 2 的整数次幂：

$$
S_g=2^{e_g-127}.
$$

编码 `0x00` 表示 $2^{-127}$，`0xFE` 表示 $2^{127}$，`0xFF` 保留为 NaN；它没有数值零、subnormal 或 infinity。因为没有 fraction，任意理想 scale 都必须量化到 power of two。这使 scale 紧凑，乘 scale 在概念上接近 exponent 调整，并提供很宽范围，但相对任意 FP16/FP32 scale 会产生额外的 scale rounding error。

以 amax-based scaling 为例，若希望 block 内最大值不超过 element 最大幅值 $v_{\max}$，理想 scale 是 $a_{\max}/v_{\max}$。为了避免 overflow，可以把它向上舍入到可表示的 E8M0 power of two：

$$
S_g=2^{\left\lceil\log_2(a_{\max}/v_{\max})\right\rceil}.
$$

这是理解 E8M0 的示意公式；具体 MX conversion 仍要遵循规范规定的 scale selection、rounding、saturation 和特殊值处理。若向上舍入后的 scale 接近理想值的两倍，block 内归一化值就只使用约一半 element range，这正是 power-of-two metadata 的精度代价。

## OCP MX v1.0 的 concrete formats

OCP MX v1.0 将 block size 固定为 32、scale 固定为 E8M0，并列出以下 concrete formats。MXFP8 与 MXFP6 各有两种 element encoding，不能只写位宽而省略 variant。

| 格式 | Element format | Element bits | Block size | Shared scale | 理想有效 bits/element |
| --- | --- | ---: | ---: | --- | ---: |
| MXFP8 | E4M3 或 E5M2 | 8 | 32 | E8M0, 8 bit | $8+8/32=8.25$ |
| MXFP6 | E2M3 或 E3M2 | 6 | 32 | E8M0, 8 bit | $6.25$ |
| MXFP4 | E2M1 | 4 | 32 | E8M0, 8 bit | $4.25$ |
| MXINT8 | 规范定义的 INT8 element | 8 | 32 | E8M0, 8 bit | $8.25$ |

表中有效位宽只计算一个 scale 恰好服务 32 个有效 element 的逻辑成本，不包含 packing padding、尾块、Tensor header 或硬件 swizzle。OCP 合规实现也不必支持表中全部格式；支持某个子集不代表支持所有 variant。

MXFP4 的 E2M1 element 幅值 codebook 为

$$
\{0,0.5,1,1.5,2,3,4,6\}
$$

及其正负号组合，其中正负零占用各自 sign pattern。它没有足够码字同时保留 IEEE 风格 infinity/NaN。E2M1 的最大 element 幅值只有 6，看似动态范围极窄；与每 32 个数一个 E8M0 scale 组合后，Tensor-level range 才得以覆盖许多数量级。代价是一个 block 内最大值与最小重要值仍需竞争 E2M1 的少量局部码字。

MXFP6 提供两种不同取舍。E2M3 的 fraction 较多、局部 codebook 更密，最大幅值为 7.5；E3M2 用一个 fraction bit 换一个 exponent bit，最大幅值为 28。两者都借助同样的 E8M0 block scale扩展整体范围。MXFP8 同理：E4M3 与 E5M2 仍是不同 codebook，只是 block scaling 降低了单个 element format 独自承担全 Tensor 动态范围的压力。

## 为什么 MX 不只是“小 group 的普通 FP8”

把任意 FP8 每 32 个数配一个 FP16 scale，也属于 block-scaled FP8，但不自动成为 OCP MXFP8。MX 合规性同时约束 block size、element encoding、E8M0 scale 和规定的 conversion/operation 语义。换用 FP16 scale、block 128 或自定义 E4M3 variant，虽然可用同一统一公式描述，却是另一个 scheme。

反过来，MX 规范定义数值语义，并不替所有硬件规定同一种物理 memory layout。实现可以为了矩阵指令重新排列 scale。NVIDIA Blackwell 的 block-scaled MMA 要求沿 GEMM reduction/K 维消费 scale vector，并使用特定 swizzled scale layout；这是 NVIDIA instruction/backend contract，而不是由“每 32 个 element 共享 E8M0”这句数学定义自动推出。

MX rowwise 与 columnwise 表示也不能通过简单 transpose bit pattern 得到。假设矩阵 $X[M,K]$ 的 rowwise block 沿 K 每 32 个值统计，而 columnwise block 沿 M 每 32 个值统计；转置会改变哪些 element 共享 scale。若反向 GEMM 需要另一方向连续的 block，系统通常要从高精度源分别量化两份，而不是直接转置已有 MX data。于是更细粒度 scale改善数值范围的同时，可能增加副本、quantize bandwidth 和调度复杂度。

## FP4 不是一个唯一格式

`FP4` 只说明总位宽为 4，无法唯一确定 exponent/fraction 分配、bias、特殊值和外部 scaling。E2M1 是当前 AI 系统中的重要 element format，但“E2M1 data”仍不等于 MXFP4 或 NVFP4：

| 名称 | Element | Local block | Local scale | 额外层次 | 标准/实现属性 |
| --- | --- | ---: | --- | --- | --- |
| 裸 E2M1 | FP4 E2M1 | 无规定 | 无规定 | 无规定 | 只有 element codebook |
| OCP MXFP4 | FP4 E2M1 | 32 | E8M0 power-of-two | 无额外全局 scale（按 MX 组合定义） | OCP MX v1.0 标准格式 |
| NVIDIA NVFP4 recipe | FP4 E2M1 | 通常 16 个连续 element；权重可用 16×16 2D scaling | FP8 E4M3 | 每 Tensor FP32 global scale | NVIDIA Transformer Engine recipe/Blackwell 实现 |

NVFP4 的重建关系可写为

$$
\hat{x}_i=v_i\,S^{\mathrm{block}}_{g(i)}S^{\mathrm{global}},
$$

其中 $v_i$ 是 E2M1，local scale 是 E4M3，global scale 是 FP32。E4M3 local scale带 fraction，比 E8M0 power-of-two scale 更细；但其范围有限，所以增加 global scale覆盖 Tensor 级数量级。NVFP4 使用 16-element 1D block，Transformer Engine 当前训练 recipe 还可对权重使用 16×16 2D scaling，以改善 rowwise 与 columnwise 表示的一致性。以上是厂商 recipe，不应写成 OCP MXFP4 的新版本。

容量上，若只计算 1D local metadata，NVFP4 data + E4M3 scale 为 $4+8/16=4.5$ bit/element，另加摊薄后的 per-tensor FP32 global scale；MXFP4 是 $4+8/32=4.25$ bit/element。NVFP4 花费更多 local metadata，换取更小 block 和带 mantissa 的 local scale。真实容量仍受 2D scaling、双向副本、padding 与 layout 影响。

## 与传统 INT8/INT4 的本质区别

传统 affine INT 的 block reconstruction 为

$$
\hat{x}_i=S_g(q_i-Z_g),
$$

block 内重建点等间距，zero-point 可精确对齐零。Block-scaled FP/MXFP 的典型形式为

$$
\hat{x}_i=S_g\operatorname{decode}_{\mathrm{FP}}(q_i),
$$

block 内仍有 exponent 造成的非均匀间距，通常没有 affine integer zero-point。二者都可使用 per-block scale，差异不在“有没有 scale”，而在 element codebook、zero alignment、rounding/saturation、metadata dtype 和硬件 compute contract。

| 比较维度 | Affine INT8/INT4 | FP8 / block-scaled FP | OCP MXFP |
| --- | --- | --- | --- |
| Element codebook | 等间距整数 lattice | 浮点非均匀 codebook | 指定 FP8/6/4 codebook |
| 零点 | 可用显式 $Z$ | 通常由 element zero 表示 | element zero；E8M0 scale 没有 zero |
| 外层 scale | 常见 FP16/FP32，可任意粒度 | recipe 自定 | E8M0，每 32 element |
| 局部 range/precision | 固定绝对 step | 随 element exponent 变化 | 局部 FP codebook + block power-of-two scale |
| Packing | INT4 常 2/byte | FP8 1 byte；FP4 需 packing | FP6/FP4 需规范兼容 packing/实现布局 |
| Compute | integer dot、fused DQ 或 W4A16 | FP8 MMA 或 DQ | 原生 block-scaled MMA或软件展开 |
| 典型敏感点 | outlier 拉大 uniform step、zero-point correction | format variant、scale history、overflow | block orientation、E8M0 rounding、scale layout |

浮点 codebook 并不天然比 integer codebook 准确。若一组数集中在窄范围，uniform INT 可把全部码字集中在那里；若数值跨多个数量级，浮点 codebook 的相对精度可能更合适。最终质量由数据分布、scale granularity、clipping/rounding 和任务敏感度共同决定。

## 从存储到 block-scaled MMA

以 MXFP4 矩阵 operand 为例，逻辑数据路径为：高精度 Tensor 沿目标 K 方向切成 32-element block；统计 block range；把理想 scale量化成 E8M0；用该 scale normalize；编码为 E2M1；两个 FP4 code pack 到一个 byte；E8M0 scale另存为 metadata Tensor；加载时 data 与 scale按硬件要求进入不同 operand path；MMA 在 block 内应用 scale并以更高精度累加。

NVIDIA PTX `tcgen05.mma` 的 block-scaled形式明确区分 element type、scale type 和 scale vector size。当前 PTX 文档列出的组合中，MX 类使用 UE8M0 scale，NVFP4 类可使用 UE4M3 scale；block-scaled MMA把 scale应用在 GEMM K 维，并使用 FP32 accumulator。CUTLASS 还要求 scale预先变换为硬件需要的 swizzled layout并放入相应存储层次。这证明 scale不是只在模型文件头读取一次的静态常数，而是矩阵 datapath 的显式 operand。

原生 instruction support 也不意味着端到端自动加速。系统仍可能需要 BF16→MX/NVFP quantization、生成 rowwise/columnwise 副本、scale swizzle、padding 和输出转换。小 GEMM、shape 不对齐、非融合转换或额外副本可能吞掉 Tensor Core 的理论吞吐优势。低位计算的端到端收益必须同时计算 data bytes、metadata bytes、conversion passes 和可用 Kernel。

## 用统一模型逐项比较三个方案

下面把教学方案 `INT8_BLOCK128`、一般性 `FP8_BLOCK128` 与标准化 `MXFP4_BLOCK32_ROW` 放入同一数据路径。前两个名称仍是参数化描述，不指向唯一标准实现。

| 阶段 | `INT8_BLOCK128` | `FP8_BLOCK128` | `MXFP4_BLOCK32_ROW` |
| --- | --- | --- | --- |
| Partition | 指定轴每 128 element | 指定轴每 128 element | 每行沿 reduction/K 轴每 32 element |
| Statistics | 常见 block absmax/min-max | 常见 block amax | 按 MX conversion 产生 block scale |
| Scale | 需指定 FP16/FP32 等 | 需指定 dtype 与是否 power-of-two | E8M0 power-of-two |
| Zero-point | 可为 0 或非零 | 通常无 affine $Z$ | 无 affine $Z$ |
| Element codebook | uniform signed INT8 | 必须指定 E4M3/E5M2 variant | E2M1 |
| Normalize/encode | $x/S$ 后 integer rounding | $x/S$ 后 FP encode | $x/S_g$ 后 E2M1 encode |
| Metadata cost | 若 FP16 scale：$16/128$ bit/element | 由 scale dtype 决定 | $8/32=0.25$ bit/element |
| Data storage | 1 byte/element | 1 byte/element | 2 element/byte |
| Decode | $S(q-Z)$ | $S\operatorname{fp8}(q)$ | $S_{E8M0}\operatorname{e2m1}(q)$ |
| Compute | integer dot 或 fused DQ | FP8 MMA 或 fused DQ | block-scaled MMA或软件 decode |
| 关键未知/约束 | axis、对称性、scale layout | FP8 variant、scale recipe、layout | MX规范、row orientation、硬件 scale layout |

这三种方案不能只按 8/8/4 bit 排序。`INT8_BLOCK128` 使用更大的 block但 uniform codebook；`FP8_BLOCK128` 的局部 exponent能覆盖不同数量级，但 scale recipe 未指定；MXFP4 用极少 element bits和更细 block，并承担 E8M0 scale rounding、packing 与更严格硬件 layout。有效选择必须结合 Tensor 分布和目标 Kernel。

## 标准、厂商实现与研究方案的边界

OFP8 与 MX 是开放规范，定义互操作所需的数值格式语义。NVIDIA NVFP4 是厂商 recipe和硬件支持路径，不等于 OCP MXFP4。Transformer Engine 对 MXFP8 的默认 E4M3、训练中双向表示和 scale swizzle 属于具体软件实现；其他 MX 合规实现不必复制全部策略。论文中的自定义 FP4、block size 或 scale 类型又属于研究方案，除非明确声明并验证兼容性，否则不能贴上标准格式名称。

同样要区分 storage type 与 arithmetic support。CUDA header中出现 `__nv_fp4_e2m1` 类型，说明工具链能表示或转换该数据；只有目标 GPU、PTX instruction、library Kernel 和合法 operand layout同时满足时，才构成原生 FP4/MX compute path。后续硬件篇会按 GPU architecture、instruction kind、operand type、accumulator 和吞吐分别核查。

## 小结

FP8/FP4 是 element codebook，block scaling 是参数共享机制，MX 是把低位 element、32-element block 与 E8M0 shared scale组合起来的标准格式族。MXFP 并非简单 shared exponent：block 共享 power-of-two scale，但每个 element 仍有自己的低位 exponent。NVFP4 则采用 E2M1、16-element/E4M3 local scale 和 FP32 global scale的层次化 recipe，与 MXFP4 在 block、metadata 和标准归属上都不同。

本篇建立的关键判断方法是：看到低精度名称时，先问 element encoding，再问 scale granularity和 dtype，然后问 storage layout 与 compute contract。下一篇将转向模型对象，解释 weights、activations、KV cache、gradients 为什么具有不同分布、生命周期、访问模式和误差敏感度，因而不能套用同一量化策略。

## 自检问题

1. OFP8 E4M3 为什么比 E5M2 有更高局部 precision，却有更小 dynamic range？
2. 为什么“E4M3”仍不足以唯一确定 max finite、NaN 和 infinity 行为？
3. MXFP4 与传统 BFP 都有 shared scaling，为何不能简单说 MXFP4 的 32 个 element 共用一个 exponent？
4. E8M0 为什么没有 zero，它的 power-of-two scale 又引入了什么误差？
5. 裸 E2M1、MXFP4 和 NVFP4 在统一模型中分别还包含哪些不同组件？
6. 为什么 rowwise MX Tensor 不能通过普通 transpose 直接得到数值等价的 columnwise MX Tensor？
7. 原生 block-scaled MMA 支持为什么仍不保证端到端模型获得理论加速比？

## 来源与适用边界

- [OCP 8-bit Floating Point Specification (OFP8), Revision 1.0](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf)：E4M3/E5M2 encoding、bias、有限范围、subnormal 和特殊值的标准定义。
- [OCP Microscaling Formats (MX) Specification v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)：MX block、E8M0 scale、MXFP8/6/4 与 MXINT8 concrete formats 的规范基线。
- Paulius Micikevicius et al., [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433), 2022：OFP8 E4M3/E5M2 的设计动机与训练研究背景；实验结果不视为所有模型和实现的质量保证。
- [NVIDIA Transformer Engine: MXFP8](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/mxfp8/mxfp8.html)：当前 NVIDIA MXFP8 recipe、E8M0 scale计算、rowwise/columnwise量化、swizzle 与 Blackwell 支持。它描述厂商实现，不替代 OCP 数值规范。
- [NVIDIA Transformer Engine: NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)：NVFP4 的 E2M1、E4M3 local scale、FP32 global scale、block size 和训练 recipe。
- [NVIDIA PTX ISA 9.1](https://docs.nvidia.com/cuda/pdf/ptx_isa_9.1.pdf) 与 [CUTLASS Blackwell SM100 GEMMs](https://docs.nvidia.com/cutlass/4.3.3/media/docs/cpp/blackwell_functionality.html)：用于核对 `tcgen05.mma` block-scale operand组合、沿 K 的 scale vector、FP32 accumulator 和 scale layout。具体可用性取决于 CUDA、CUTLASS 与 GPU compute capability。

本文没有运行 FP8/FP4 conversion、GEMM benchmark 或训练实验。格式数值来自标准定义；metadata 位宽是未计 padding、swizzle和额外副本的逻辑值。厂商软件页面更新较快，实际工程必须固定文档版本、library version 与 GPU architecture。

