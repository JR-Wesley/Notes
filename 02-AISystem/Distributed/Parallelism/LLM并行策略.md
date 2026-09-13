[LLM并行训练1-流水线并行 - SunStriKE - 博客园](https://www.cnblogs.com/sunstrikes/p/18270017)
[(94 封私信 / 81 条消息) LLM推理并行优化的必备知识 - 知乎](https://zhuanlan.zhihu.com/p/1937449564509545940)
# 并行基础

大模型推理应用中，不同的场景、模型层有着不同的计算特点，并行策略需要根据这些特点进行调整，不但要消除冗余存储、冗余计算，还要最大限度地降低通信开销对计算的影响，保证推理的 SLA（如**TTFT/TPOT**）。

主流的并行策略包括：**DP/TP/SP/EP/CP/PP/ZeRO**，当前的 [MoE模型](https://zhida.zhihu.com/search?content_id=261477567&content_type=Article&match_order=1&q=MoE%E6%A8%A1%E5%9E%8B&zhida_source=entity) 中，比较常用的是**DP/TP/SP 和 EP**。这些策略一般会组合使用，e.g.:

- 在 [Attention层](https://zhida.zhihu.com/search?content_id=261477567&content_type=Article&match_order=1&q=Attention%E5%B1%82&zhida_source=entity) 中采用**TP、SP**，也可以开始**CP**；
- [FFN层](https://zhida.zhihu.com/search?content_id=261477567&content_type=Article&match_order=1&q=FFN%E5%B1%82&zhida_source=entity) 如果是 dense 结构用**TP+SP**，如果是 sparse 结构 (MoE) 常用**EP**；
- **DP**是所有层都适用。

**[ZeRO策略](https://zhida.zhihu.com/search?content_id=261477567&content_type=Article&match_order=1&q=ZeRO%E7%AD%96%E7%95%A5&zhida_source=entity)**(**参数分片/shard**)、**PP**层间的流水线并行相对而言当前的使用频率较低，在一些特定场景中可考虑开启。

![[file-20251106075538014.png]]

训练与推理的原理相同，推理中有个场景需要单独讨论 ---**[PD分离](https://zhida.zhihu.com/search?content_id=261477567&content_type=Article&match_order=1&q=PD%E5%88%86%E7%A6%BB&zhida_source=entity)**：目前推理部署中，会把 P(Prefill) 和 D(Decode) 阶段进行分离部署以解决 compute-bond、memory-bond 场景问题，而且会配置 xPxD（多 P 多 D）。P 和 D 的并行策略可以不一样，比如，P 实例处理请求数一般较少，**DP 设置小**；而 decode 需将并发打上去，**DP 数量设置大**； P 实例的 MoE 层可使**用 TP 并行**，D 实例则一般使用**EP 并行**。

## 1 MoE 模型的并行策略

当前主流的模型采用的 FFN 是独立专家 (MoE)+ 共享专家 (Dense)，这类 MoE 模型的并行策略一般使用**DP/TP/EP**，如下所示是一个 DP=2/TP=2/EP=4 的例子（假设 word_size=4）。

![[file-20251106080338597.png]]

**Attention 层采用 TP 并行**，每个 DP 需要一个 allreduce 操作。数据进行 FFN 层计算，在**MoE 采用了 EP 并行**，会将数据跨 DP 域进行 [allgather](https://zhida.zhihu.com/search?content_id=261477567&content_type=Article&match_order=1&q=allgather&zhida_source=entity) 然后再进行 route 分发，计算完成后进行 [reduce scatter](https://zhida.zhihu.com/search?content_id=261477567&content_type=Article&match_order=1&q=reduce+scatter&zhida_source=entity) 操作。**Dense 层则采用 TP 并行**，计算完成后需要 allreduce 操作。

可以尝试对局部进行调整优化，比如 decode 中：**把 MoE 的 allgather 换成 alltoall，数据量大时**能够提升整体效率。

![[file-20251106080612397.png]]
