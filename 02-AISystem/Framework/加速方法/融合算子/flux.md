---
dateCreated: 2025-08-15
dateModified: 2025-08-15
---

flux 卡间通信，推理 TP 优化，单机 8 卡 TP=8

TP 高 allgather reducescatter

PP 中相邻 stage/gpu p2p

EP 中 moe all2all

DP 无

multigpu llm e2e pd 各类 op 占比排序

1. mma op(vanilla/cutlass/triton)
2. attention op(flashattn paged/ragged…)
3. 卡间,e2e latency 占比根据 prompt 长度浮动，长文本显著
4. rope(attn qk), nom(小),acti(可 fused)

算子优化固定，3 占比更多

存在于 fwd bpw bpa 等 pipeline，训练荣誉 overlap，推理仅 fwd，卡间和前后 op 依赖，主流东没事实现。

none: latency++

less comm workload，改动小，有损，量化有上限

framework-lvevel，仅训练，推理依赖

op-level 粗，有一定，有限，mma 效率下降，launch OH 无法忽视

op-lvl fine-grained，最高，

vllm PR llm 5%

sglan Q4 devel roadmap

multi-stream sync & host-device async

cutstreamwrval/waitval api host/kernel

mma prologue/ epilogue

vanilla

pro

main

epi

allga+ mma

allgather(a) a+b

mma + r-s

flux 基于 cuts3 cute stream-k 对 sm90 sm80，封装严重，

TODO

allgather+mma

1. wait signal 主要调整，否则会 hang/mis，根据 t b tile m-dim index 计算依赖于哪个 chunk，然后 load A 之前等待 sig
2. thread block swizzle 可选

实现了计算与卡件通信的优化，有在 kernel 外部 launch sync，kernel 数量减少，广义 kernel 融合

两种非常规的合并，mma 算子特殊，p/ep 融合差异大，对卡件同新在 threadblock/warp 细粒度

优秀 overlap,latency 小幅恶化，

限制：双方 latency 差异大无法掩盖

效费比低：卡件通信不主流

融合框架耦合

于 GPU 需要 host device 内存操作冷门 api

ep 需要 kernel 内 warp/block 跨卡通信，需要 driver/compiler

多留异步

不支持跨 node

开发复杂
