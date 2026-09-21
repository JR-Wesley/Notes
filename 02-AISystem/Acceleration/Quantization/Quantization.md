# 量化（Quantization）

> 从数值表示、量化数学与现代低精度格式，逐步进入模型算法、Kernel、GPU 和分布式 AI Infra。

## 系统学习入口

- [[00-低精度量化知识地图]]：主线、分类体系、学习阶段与完整数据路径
- [[01-数值表示与误差基础]]：FP/INT 编码、range、precision、ULP、rounding 与 subnormal
- [[02-量化的统一数学模型]]：codebook、mapping、scale、zero-point、clipping 与误差
- [[03-量化方案的分类维度与粒度]]：正交分类轴、group/block 和 metadata 开销
- [[04-FP8、FP4与Microscaling格式]]：OFP8、MXFP8/6/4、E8M0、MXFP4 与 NVFP4
- [[05-模型张量的量化策略]]：weights、activations、KV cache、gradients、optimizer 与通信对象
- [[06-PTQ、QAT与LLM量化算法]]：RTN、calibration、outlier、SmoothQuant、GPTQ、AWQ、QAT 与 mixed precision
- [[07-量化张量的存储、布局与元数据]]：INT4 packing、metadata shape、alignment、checkpoint 与 Kernel prepack
- [[08-低精度GEMM与Attention-Kernel]]：load、decode、scale、MMA、accumulate、epilogue 与量化 KV cache
- [[09-GPU低精度执行机制]]：Tensor Core/Matrix Core、MMA、operand、accumulator、吞吐与 memory hierarchy
- [[10-分布式AI中的低精度数据路径]]：collective algebra、gradient compression、TP/PP/EP 与通信计算重叠
- [[11-低精度格式与系统实现案例]]：从 BF16 到 INT8 block、FP8 block 与 MXFP4 的完整数值、存储和执行合同
- [[12-低精度AI-Infra前沿与资料索引]]：标准、硬件、框架、论文、演进方向与版本边界

00–12 已构成第一阶段完整主线。后续按具体问题深入，不再重复整套课程框架。

## 早期资料与旁支

- [[../../Framework/llama.cpp/repo/量化]]：旧量化概述，含重复和待核查内容，不作为新系列的事实基线
- [[quant]]：早期量化资料与实践线索
- [[Paperreading]]：压缩、稀疏与论文摘录
- [[NN稀疏]]：稀疏化旁支；它与量化可组合，但不是同一压缩维度
