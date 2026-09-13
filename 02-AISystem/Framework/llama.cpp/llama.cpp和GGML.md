首先我们学习一些 ggml 的核心概念。如果你熟悉 PyTorch 或 TensorFlow，这可能对你来说有比较大的跨度。但由于 ggml 是一个 **低层** 的库，理解这些概念能让你更大幅度地掌控性能。

- [ggml_context](https://github.com/ggerganov/ggml/blob/18703ad600cc68dbdb04d57434c876989a841d12/include/ggml.h#L355): 一个装载各类对象 (如张量、计算图、其他数据) 的“容器”。
- [ggml_cgraph](https://github.com/ggerganov/ggml/blob/18703ad600cc68dbdb04d57434c876989a841d12/include/ggml.h#L652): 计算图的表示，可以理解为将要传给后端的“计算执行顺序”。
- [ggml_backend](https://github.com/ggerganov/ggml/blob/18703ad600cc68dbdb04d57434c876989a841d12/src/ggml-backend-impl.h#L80): 执行计算图的接口，有很多种类型: CPU (默认) 、CUDA、Metal (Apple Silicon) 、Vulkan、RPC 等等。
- [ggml_backend_buffer_type](https://github.com/ggerganov/ggml/blob/18703ad600cc68dbdb04d57434c876989a841d12/src/ggml-backend-impl.h#L18): 表示一种缓存，可以理解为连接到每个 `ggml_backend` 的一个“内存分配器”。比如你要在 GPU 上执行计算，那你就需要通过一个 `buffer_type` (通常缩写为 `buft` ) 去在 GPU 上分配内存。
- [ggml_backend_buffer](https://github.com/ggerganov/ggml/blob/18703ad600cc68dbdb04d57434c876989a841d12/src/ggml-backend-impl.h#L52): 表示一个通过 `buffer_type` 分配的缓存。需要注意的是，一个缓存可以存储多个张量数据。
- [ggml_gallocr](https://github.com/ggerganov/ggml/blob/18703ad600cc68dbdb04d57434c876989a841d12/include/ggml-alloc.h#L46): 表示一个给计算图分配内存的分配器，可以给计算图中的张量进行高效的内存分配。
- [ggml_backend_sched](https://github.com/ggerganov/ggml/blob/18703ad600cc68dbdb04d57434c876989a841d12/include/ggml-backend.h#L169): 一个调度器，使得多种后端可以并发使用，在处理大模型或多 GPU 推理时，实现跨硬件平台地分配计算任务 (如 CPU 加 GPU 混合计算)。该调度器还能自动将 GPU 不支持的算子转移到 CPU 上，来确保最优的资源利用和兼容性。

# 深入理解 GGML

[(57 封私信 / 85 条消息) 深入理解GGML（一）模型和计算图 - 知乎](https://zhuanlan.zhihu.com/p/818059523)

[(57 封私信 / 85 条消息) llama.cpp源码解析--CUDA流程版本 - 知乎](https://zhuanlan.zhihu.com/p/665027154)
