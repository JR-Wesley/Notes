# PTQ、QAT 与 LLM 量化算法

前五篇已经回答“数值怎样编码”“一套 scheme 由哪些维度组成”“不同 Tensor 为什么需要不同策略”。本篇进入算法层，但仍沿用同一数据路径：量化算法并不会创造另一套数值表示，它只是在 partition、statistics、scale、mapping、模型参数或精度分配等位置介入，使最终低精度表示造成的任务损失更小。

理解算法时最重要的不是记住 GPTQ、AWQ、SmoothQuant 的名称，而是回答四个问题：它改变了哪个对象；它优化的是逐元素重建误差、层输出误差还是任务损失；它需要多少 calibration/training 数据与计算；最终导出的 code、metadata 和 Kernel contract 是什么。

## 先建立算法层的统一坐标

设一层线性变换为

$$
Y=XW^\top,
$$

量化后的近似输出为

$$
\hat{Y}=\mathcal{F}(\hat{X},\hat{W}),
$$

其中 $\hat{X}$、$\hat{W}$ 是按照 [[02-量化的统一数学模型]] 编码再恢复的值，$\mathcal{F}$ 还可能包含低精度 accumulator、requantization 或 fused Kernel。算法可以在以下位置介入：

| 介入层次 | 核心动作 | 代表思路 | 直接改变什么 |
| --- | --- | --- | --- |
| Parameter selection | 选择 clipping range、scale、zero-point | min/max、percentile、MSE calibration | quantizer metadata |
| Granularity/codebook | 改变共享范围或码字集合 | per-channel、per-group、mixed codebook | scheme 与存储成本 |
| Equivalent transformation | 在不改高精度函数的前提下重分配幅值 | SmoothQuant、AWQ scaling | 待量化 Tensor 的分布 |
| Error compensation | 量化部分参数后修正其余参数 | GPTQ/OBQ 类方法 | quantized weight values |
| Precision allocation | 对敏感层、通道或对象保留更多精度 | mixed precision、outlier path | 多种 scheme 的组合 |
| Model adaptation | 将量化噪声放入训练闭环 | QAT、quantization-aware fine-tuning/distillation | 可训练模型参数 |

这些方法可以组合。SmoothQuant 后仍要 calibration activation scale；AWQ 最终仍需选择 group-wise weight quantizer；GPTQ 可以使用 mixed precision；QAT 部署时仍需确定 packing 和 Kernel。算法名不能替代完整的 Quantization Scheme descriptor。

## 基线：Round-to-Nearest 先回答“简单方法有多差”

对固定的 symmetric uniform quantizer，最简单的权重量化是

$$
S_g=\frac{\max_{i\in g}|w_i|}{q_{\max}},
\qquad
q_i=\operatorname{clip}\!\left(
\operatorname{round}(w_i/S_g),q_{\min},q_{\max}
\right).
$$

这个 baseline 常称为 RTN（Round-to-Nearest）。它只根据当前 group 的权重值做局部投影，不利用 calibration activation，也不在量化一个权重后补偿其他权重。RTN 的价值不是“足够先进”，而是隔离三个基础变量：element codebook、group size 和 clipping policy。

如果一种复杂方法只与 FP16/BF16 比较，而没有与相同 bit width、group size 和 layout 的 RTN 比较，就无法判断收益究竟来自算法，还是来自更细粒度或不同格式。反过来，RTN 的 reconstruction error 较低也不保证层输出误差较低，因为不同权重会被不同强度的 activation 使用。

## PTQ：在不重新训练完整模型的条件下适配量化

Post-Training Quantization 从已训练模型出发，通过少量或无训练数据估计量化参数、变换权重或补偿误差，最终导出低精度模型。典型 PTQ 流程是：

$$
\text{FP model}
\rightarrow
\text{collect statistics}
\rightarrow
\text{choose scheme/parameters}
\rightarrow
\text{quantize or transform layer}
\rightarrow
\text{validate}
\rightarrow
\text{pack/export}.
$$

“Post-training”只描述模型适应阶段，并不表示所有参数都是 static。权重通常离线量化并静态保存；activation 仍可在运行时 per-token dynamic quantize。PTQ 也不等同于无数据：RTN 可以完全 weight-only，SmoothQuant、GPTQ 和 AWQ 通常使用 calibration samples 来观察 activation 或层输入。

### Calibration dataset 决定算法看见哪种分布

Calibration 数据无需覆盖完整训练集，但应代表部署时的长度、领域、prompt 结构和数值分布。它的作用不是重新学习知识，而是估计 range、outlier、channel importance 或局部 Hessian。数据过少会让统计量噪声大；数据域偏移会使 static scale 或敏感度排序不适合真实请求；只用短序列还可能漏掉长上下文下的 activation/KV 行为。

Calibration 有三个容易混淆的产物。Observer statistics 是 min/max、amax、histogram 或 activation norm；quantizer parameters 是由统计量推导出的 scale、zero-point 和 clipping threshold；algorithm state 还可能包含 GPTQ 的近似 Hessian、AWQ 的 channel scale 搜索结果或 SmoothQuant 的迁移系数。只有部署需要的部分才应进入最终 checkpoint。

### Calibration objective 不只有 min/max

Min/max calibration 选择覆盖观测范围的 scale，避免校准样本中的 saturation，却可能被单个 outlier 拉大 step。Percentile clipping 舍弃分布尾部；MSE calibration 搜索使

$$
\mathcal{L}_{\mathrm{rec}}(\alpha,\beta)
=
\sum_i\left(x_i-Q_{\alpha,\beta}(x_i)\right)^2
$$

最小的范围；histogram/KL 类准则则比较量化前后的分布。不同 objective 保护的性质不同，没有一个 threshold 对所有层和任务都普适。

更接近模型行为的 layer reconstruction 目标是

$$
\mathcal{L}_{\mathrm{layer}}
=
\lVert XW^\top-X\hat{W}^\top\rVert_F^2,
$$

它用 calibration activation $X$ 对权重误差加权。再向上是 block output、logit、distillation 或最终 task loss。目标越接近模型输出，通常越能表达真实敏感度，但计算成本、数据依赖和过拟合风险也更高。

## Outlier 不是一种单一现象

量化困难通常来自“一个共享范围内的幅值差异过大”，但 outlier 至少要区分三类：

| 类型 | 现象 | 常见处理 |
| --- | --- | --- |
| Persistent channel outlier | 某些 hidden channel 在许多 token 中持续较大 | per-channel scale、SmoothQuant/AWQ 等等价缩放 |
| Token-dependent outlier | 大值位置随输入或 token 改变 | per-token dynamic scale、更细 block、保留高精度路径 |
| Isolated weight outlier | 少量权重远离所在 group 分布 | 更小 group、clipping、mixed precision、error compensation |

Outlier handling 也不等于简单删除大值。可以扩大 range 接受更粗的 resolution，可以 clip 尾部接受 saturation，可以把大值迁移到更易量化的对象，可以让少量元素走高精度旁路，也可以借助模型或层输出目标补偿它造成的误差。选择取决于 outlier 的稳定性以及目标 Kernel 是否支持不规则表示。

## Equivalent transformation：函数不变，量化难度重新分配

对 $X[N,K]$、$W[M,K]$ 和正对角矩阵 $D\in\mathbb{R}^{K\times K}$，

$$
XW^\top
=
(XD^{-1})(WD)^\top.
$$

高精度下左右两边完全相等；量化后，$XD^{-1}$ 和 $WD$ 的幅值分布已经改变。这类方法利用等价变换在 activation 与 weight、不同 channel 或不同算子之间重新分配量化难度。它不消灭信息损失，只是让有限 codebook 更有效地覆盖重要数值。

### SmoothQuant：把 activation 难度迁移给 weight

SmoothQuant 面向 LLM W8A8 PTQ。其出发点是 activation outlier 比 weight 更难量化，而 weight 是静态对象，可以离线吸收 channel-wise scaling。论文使用 activation 与 weight 的每个 input channel 最大幅值构造平滑系数，常写为

$$
s_j=
\frac{\max|X_j|^{\alpha}}
{\max|W_j|^{1-\alpha}},
$$

并用 $X'_j=X_j/s_j$、$W'_j=s_jW_j$ 保持线性层函数不变。$\alpha$ 控制难度迁移程度：增大它通常更积极地压低 activation channel 幅值，同时扩大相应 weight channel。

关键机制不是“把 outlier 变小”这么简单，而是 activation 与 weight 的量化误差预算被重新平衡。变换后仍要分别确定 activation 和 weight 的 quantizer；部署收益还要求 W8A8 Kernel 能高效消费最终 layout。若只得到较小 fake-quant error，却需要额外 runtime scaling pass，就可能损失系统收益，因此静态因子通常应 fold 进相邻权重或算子参数。

### AWQ：用 activation 识别重要 weight channel

AWQ 是 activation-aware 的低 bit weight-only PTQ。它用 calibration activation 判断哪些 weight channel 对层输出更重要，再搜索 per-channel scaling，使重要 channel 的权重在 group-wise 低比特量化中获得更好的有效分辨率。其等价缩放同样可 fold 到相邻计算，不要求为少数权重建立不规则 mixed-precision 存储。

“Activation-aware”不表示 activation 也在运行时量化；AWQ 的主要部署对象仍是 weight。它与 SmoothQuant 都使用等价缩放，但目标不同：SmoothQuant 重点缓解 W8A8 中难量化的 activation outlier，AWQ 重点保护 weight-only quantization 中由 activation 暴露出的重要权重方向。

## GPTQ：用二阶信息补偿逐步量化误差

GPTQ 是 one-shot weight-only PTQ，建立在层输出 reconstruction 目标上。若用 $X[K,N]$ 表示收集到的层输入样本，可写成

$$
\min_{\hat{W}}
\lVert WX-\hat{W}X\rVert_F^2,
\qquad
H\propto XX^\top.
$$

$H$ 是该局部二次目标对权重的近似 Hessian，它表达 input feature 的尺度与相关性。RTN 只看某个权重离哪个 code 最近；GPTQ 则关心这个权重误差经过真实层输入后对输出造成多大影响。

算法按列或按 block 逐步量化权重。某个权重被投影到 codebook 后，其误差不会被直接遗忘，而是利用 $H^{-1}$ 中的相关性更新尚未量化的权重，使后续参数补偿已经产生的输出误差。GPTQ 又通过一次处理多个行、延迟批量更新和稳定的矩阵分解，把原本昂贵的二阶过程扩展到大模型。

这里的“二阶”是 calibration layer reconstruction 的局部近似，不是对整个语言模型 task loss 求完整 Hessian；“one-shot”也不表示一次 rounding，而是指不进行常规 end-to-end retraining。最终 checkpoint 仍只是普通的量化 code 与 metadata，推理不需要携带 Hessian。实际速度取决于导出格式、packing 和对应 weight-only GEMM Kernel，而不是 GPTQ 名称本身。

## QAT：让模型在训练中适应离散化

Quantization-Aware Training 在前向中插入 fake quantizer：

$$
\hat{x}=D(E(x)),
$$

前向使用重建后的量化值模拟部署误差，但训练状态通常仍以 FP16/BF16/FP32 保存和更新。由于 rounding 的导数几乎处处为零，常用 Straight-Through Estimator（STE）在反向中用近似梯度穿过 quantizer。一个示意形式是

$$
\frac{\partial \hat{x}}{\partial x}
\approx
\mathbf{1}_{x\in[\alpha,\beta]},
$$

即范围内近似传递梯度、饱和区抑制或修改梯度；具体实现可采用不同 STE、learned scale 和 clipping 参数。

QAT 的优势是模型能共同调整许多层来适应量化噪声，尤其在更低 bit、同时量化 W/A/KV 或 PTQ 难以维持质量时有价值。代价是训练数据、算力、优化稳定性和工程复杂度显著增加。Fake quantizer 还必须尽量匹配真实 backend 的 codebook、rounding、granularity、clipping 和 accumulator；否则训练适应的是一个并不会部署的数值系统。

LLM-QAT 展示了一条针对大模型的路径：用预训练模型生成数据并进行 distillation，在不依赖原始训练集的条件下让模型适应 weight、activation 和 KV cache 的低比特表示。它是一种具体研究方案，不代表所有 QAT 都是 data-free，也不意味着合成数据能覆盖任意部署分布。

## PTQ、QAT 与 fine-tuning 不是简单的二选一

实际方法位于连续谱上：

| 方法 | 是否使用 calibration 数据 | 是否反向传播 | 是否更新模型参数 | 典型成本 |
| --- | --- | --- | --- | --- |
| Weight-only RTN | 可不使用 | 否 | 否 | 最低 |
| Range calibration | 是 | 否 | 否 | 统计与搜索 |
| SmoothQuant/AWQ | 是 | 否 | 离线等价变换 | 中低 |
| GPTQ | 是 | 否 | 局部误差补偿后写入权重 | 中等，逐层处理 |
| Quantization-aware fine-tuning | 是 | 是 | 是 | 低于完整预训练但仍显著 |
| Full QAT/distillation | 是或合成 | 是 | 是 | 最高 |

有些文献把少量参数更新称为 QAT，有些称 quantization-aware fine-tuning；也有方法先做 PTQ 初始化再短暂训练。分类时应记录“反向是否穿过 quantizer、哪些参数被更新、优化什么 loss、使用什么数据”，不要只根据标题归类。

## Mixed precision：把有限高精度预算给敏感位置

Mixed precision 的对象可以是 layer、channel、group、token、算子或 accumulator。最简单的形式是敏感层保持 BF16，其余层 INT8/INT4；更细的方法可以让少数 outlier channel 使用更高 bit。抽象成资源约束优化：

$$
\min_{b_1,\ldots,b_L}
\mathcal{L}(b_1,\ldots,b_L)
\quad
\text{s.t.}
\quad
\sum_{\ell=1}^{L}C_\ell(b_\ell)\le C_{\mathrm{budget}},
$$

其中 $C_\ell$ 可以是模型 bytes、latency、energy 或 metadata/Kernels 成本。只按 reconstruction error 分配 bit 可能忽略真实 latency：backend 若没有某个混合组合的 fused Kernel，理论上节省的 bit 反而引入 dispatch、转换或不规则访存。

因此 mixed precision 必须同时给出 sensitivity metric 与 system cost model。常见敏感度依据包括 layer output error、Hessian/Fisher 近似、activation magnitude、loss perturbation 和小规模任务评测；系统约束则包括支持的 dtype、group size、tile alignment、额外 kernel 数与高精度旁路比例。

## 把四种代表方法放回统一数据路径

| 数据路径阶段 | RTN | SmoothQuant | GPTQ | AWQ |
| --- | --- | --- | --- | --- |
| Calibration input | 不需要或只做验证 | 收集 channel activation range | 收集 layer input，构造二阶统计 | 收集 activation importance |
| Partition | 预先指定 | 通常按 Tensor/channel 配合 W8A8 scheme | 常用 per-channel/per-group weight scheme | 常用 group-wise weight scheme |
| Pre-transform | 无 | activation/weight 等价缩放 | 通常无全局等价迁移 | activation-aware channel scaling |
| Scale/codebook | 直接由局部权重范围决定 | 对变换后 W/A 分别确定 | 在逐步量化过程中使用指定 quantizer | 对缩放后 weight 确定 |
| Error handling | 独立 rounding | 在 W 与 A 间迁移难度 | 用近似二阶相关性补偿剩余权重 | 保护重要 channel 的有效精度 |
| 最终主要对象 | Weight | Weight + activation | Weight | Weight |
| Runtime algorithm state | 无 | 无；变换应被 fold | 无 Hessian | 无搜索状态 |
| Kernel 前提 | 对应 weight format | 高效 W8A8 | 对应 packed W3/W4 等格式 | 对应 packed weight-only format |

这张表说明算法处理阶段和部署格式是两套描述。GPTQ 与 AWQ 可能导出相似的 INT4 group-wise payload，却产生不同 code；SmoothQuant 最终可能导出普通 INT8 Tensor，但其数值分布已经经过离线变换。Checkpoint 标签应同时记录算法 provenance 和真正决定 decoder/Kernel 的 format descriptor。

## 评估顺序：从 bit-level 正确性到任务指标

量化算法至少应经过四层验证：

1. **Format correctness**：pack/unpack round trip、signedness、scale/zero-point shape、tail 与 layout 都正确。
2. **Numerical reconstruction**：报告 weight/activation 的 MSE、max error、relative L2、cosine similarity、saturation rate。
3. **Model quality**：比较 layer output、logit、perplexity 和目标任务；固定 calibration/evaluation dataset 与序列长度。
4. **System performance**：报告实际 checkpoint/HBM bytes、prefill/decode latency、throughput、batch/sequence、GPU、Kernel 和软件版本。

只报告 perplexity 不能证明格式或 Kernel 正确，只报告模型大小不能证明运行更快，只报告 Tensor Core 峰值也不能证明端到端收益。算法比较应固定 codebook、group size 与 backend；系统比较还需把 quantize/dequantize、metadata load、padding 和高精度 fallback 计算在内。

## 推荐实践顺序

先在同一小型线性层上实现 RTN，固定 INT4 symmetric、group size 128，记录 weight reconstruction 与 layer output error。随后只改变一个变量：加入 calibration clipping；把 group size 改为 32；再使用 activation 对误差加权。这样能看到“更细粒度”和“更聪明算法”分别贡献多少。

第二步复现 SmoothQuant 的等价变换恒等式，先验证高精度变换前后输出一致，再量化两侧并扫描 $\alpha$。第三步用一个小矩阵实现顺序量化与误差补偿，比较 RTN 和二阶目标，而不是一开始量化完整 LLM。第四步才加入 fake quant 和 STE，确认训练时 quantizer 与目标部署格式一致。

本篇不直接提供完整 GPTQ/AWQ 实现，因为真正可用的实现还涉及数值稳定的矩阵分解、逐层数据捕获、packing 与 backend-specific Kernel；这些将在算法实验和 Runtime 篇分别落地。

## 小结

PTQ、QAT、calibration、outlier handling 与 mixed precision 描述的是不同问题。PTQ 是训练后适配阶段；calibration 负责从有限样本估计量化参数或敏感度；SmoothQuant 与 AWQ 通过等价缩放重塑待量化分布；GPTQ 用局部二阶信息补偿逐步权重量化误差；QAT 把 fake quantization 放入优化闭环；mixed precision 则在质量与系统预算之间分配不同表示。

学习这些算法时应始终回到两条线：算法究竟改变了 encoder、metadata、Tensor 参数还是训练目标；部署时最终保存什么 code/metadata，并由哪个 Kernel 消费。算法精度收益只有在格式可表示、布局可加载、Kernel 可执行时，才会成为 AI Infra 收益。

## 自检问题

1. 为什么 RTN 是评价 GPTQ、AWQ 等方法时不可缺少的 baseline？
2. PTQ 使用 calibration 数据为什么仍然不等于重新训练模型？
3. Min/max、MSE clipping 与 layer reconstruction 分别优化什么目标？
4. SmoothQuant 中等价变换为什么不改变高精度线性层，却会改变量化误差？
5. AWQ 使用 activation statistics，为什么它仍属于 weight-only quantization？
6. GPTQ 的近似 Hessian 描述什么相关性，为什么推理 checkpoint 不需要保存它？
7. QAT 为什么需要 STE，fake quantizer 与真实 backend 不一致会有什么后果？
8. Mixed precision 为什么不能只根据 layer sensitivity 决定，还必须考虑 Kernel 支持？
9. 两个都标记为 W4A16 的 GPTQ 和 AWQ checkpoint 为什么可能不能由同一 Kernel 直接读取？

## 来源与适用边界

- Benoit Jacob et al., [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html), CVPR 2018：用于 affine quantization、fake quantization 与整数部署路径的基础定义；其 CNN/mobile 场景不能直接外推到现代 LLM。
- Markus Nagel et al., [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295), 2021：用于 PTQ/QAT、range setting、STE 和量化误差的系统性背景。
- Guangxuan Xiao et al., [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://proceedings.mlr.press/v202/xiao23c.html), ICML 2023：用于 activation-to-weight 难度迁移、channel-wise equivalent transformation 与 W8A8 PTQ。
- Elias Frantar et al., [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323), ICLR 2023：用于 one-shot weight quantization、局部二阶信息与误差补偿机制。
- Ji Lin et al., [AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html), MLSys 2024：用于 activation-aware salient weight channel、per-channel scaling 与 weight-only PTQ。
- Zechun Liu et al., [LLM-QAT: Data-Free Quantization Aware Training for Large Language Models](https://arxiv.org/abs/2305.17888), 2023：用于 LLM 的 data-free distillation QAT 以及同时量化 weight、activation、KV cache 的研究案例。

本文解释算法机制，不把论文中的特定模型结果或速度数字外推为普遍结论。没有运行 calibration、模型评测、QAT 或 Kernel benchmark；公式用于建立统一理解，复现时必须核对论文版本、实现细节、数据集、导出格式和目标 backend。
