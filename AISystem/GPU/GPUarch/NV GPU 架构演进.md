---
dateCreated: 2025-08-15
dateModified: 2025-08-15
---

[英伟达AI芯片路线图分析与解读](https://mp.weixin.qq.com/s/2p0UMOGNEv2krD9PtF7dDg)

# GPU 性能指标

1. 核心数
2. GPU 显存容量
3. GPU 计算峰值
4. 显存带宽

# Professional Product

|                                                                                                                                                                                                                                                                                                                                                                        |                                               |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| Professional Product                                                                                                                                                                                                                                                                                                                                                   | Graphics Cards                                |
| [Volta (2017)]([ https://en.wikipedia.org/wiki/Volta_\ (microarchitecture\)]( https://en.wikipedia.org/wiki/Volta_%5C\ (microarchitecture%5 C\)) "Volta (microarchitecture)") <br>(Pred. - [Pascal]([ https://en.wikipedia.org/wiki/Pascal_\ (microarchitecture\)]( https://en.wikipedia.org/wiki/Pascal_%5C\ (microarchitecture%5 C\)) "Pascal (microarchitecture)")) | Tesla V <br>Titan V <br>Quadro GV100          |
| [Ampere (2020)]([ https://en.wikipedia.org/wiki/Ampere_\ (microarchitecture\)]( https://en.wikipedia.org/wiki/Ampere_%5C\ (microarchitecture%5 C\)) "Ampere (microarchitecture)")                                                                                                                                                                                      | A 100                                         |
| [Hopper (2022)]([ https://en.wikipedia.org/wiki/Hopper_\ (microarchitecture\)]( https://en.wikipedia.org/wiki/Hopper_%5C\ (microarchitecture%5 C\)) "Hopper (microarchitecture)")                                                                                                                                                                                      | H 100 <br>H 200                               |
| [Blackwell (2024)]([ https://en.wikipedia.org/wiki/Blackwell_\ (microarchitecture\)]( https://en.wikipedia.org/wiki/Blackwell_%5C\ (microarchitecture%5 C\)) "Blackwell (microarchitecture)")                                                                                                                                                                          | B 100 <br>B 200                               |
| [Rubin (2026)]([ https://en.wikipedia.org/wiki/Rubin_\ (microarchitecture\)]( https://en.wikipedia.org/wiki/Rubin_%5C\ (microarchitecture%5 C\)) "Rubin (microarchitecture)")                                                                                                                                                                                          | R 100<br><br>R 200                            |
| [Feynman (2028)]([ https://en.wikipedia.org/wiki/Feynman_\ (microarchitecture\)]( https://en.wikipedia.org/wiki/Feynman_%5C\ (microarchitecture%5 C\)) "Feynman (microarchitecture)")                                                                                                                                                                                  | F 100 (Unconfirmed)<br><br>F 200 (Unconfirmed |

A 100 架构解析：https://zhuanlan.zhihu.com/p/1908285912053453831

# **NVIDIA 主流架构演进**

| **架构名称**         | **发布时间** | **核心参数**                                                                      | **特点和优势**                                                                                                                                                                                                  | **算力等级** | **代表型号**                                      |
| ---------------- | -------- | ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | --------------------------------------------- |
| **Fermi**        | 2010 年   | 晶体管：30 亿  <br>CUDA 核心：512  <br>SM 单元：16  <br>制程：40 nm                          | 首次引入统一计算架构，支持 ECC 内存和动态并行计算，推动 GPGPU 应用从科学计算向通用计算扩展。但受限于制程和架构设计，能效比偏低。| 2.0      | GeForce GTX 580  <br>Tesla C 2050              |
| **Kepler**       | 2012 年   | 晶体管：43 亿  <br>CUDA 核心：2304  <br>SM 单元：15  <br>制程：28 nm                         | 引入 GPU Boost 动态超频技术，支持动态并行计算和单精度浮点（FP 32）性能提升，能效比相比 Fermi 提升 50% 以上。GK 110 核心首次实现完整双精度浮点（FP 64）计算能力，推动 HPC 领域发展。| 3.0      | GeForce GTX 780  <br>Tesla K 40                |
| **Maxwell**      | 2014 年   | 晶体管：29 亿  <br>CUDA 核心：2048  <br>SM 单元：16  <br>制程：28 nm                         | 革命性优化能效比（较 Kepler 提升 3 倍），引入多分辨率渲染（MFAA）和动态超分辨率（DSR），支持 DirectX 12 和 OpenGL 4.5。首次在消费级显卡中实现完整的异步计算和多线程处理，为 VR 应用奠定基础。| 5.0/5.2  | GeForce GTX 980  <br>Tesla M 40                |
| **Pascal**       | 2016 年   | 晶体管：72 亿  <br>CUDA 核心：3584  <br>SM 单元：28  <br>制程：16 nm                         | 首次引入 16 nm FinFET 制程，支持 **HBM 2** 显存（带宽提升 3 倍），并推出首个专为深度学习设计的 **Tensor Core**（P 100）。消费级显卡（如 GTX 1080）首次支持实时光线追踪（需软件支持），同时多 GPU 互联技术 SLI 升级至更高效的 NVLink。| 6.0/6.1  | GeForce GTX 1080  <br>Tesla P 100              |
| **Volta**        | 2017 年   | 晶体管：211 亿  <br>CUDA 核心：5120  <br>SM 单元：80  <br>制程：12 nm                        | 革命性 Tensor Core 支持**混合精度计算（FP 16/FP 32/INT 8）**，AI 性能提升 50 倍以上。首次实现**结构化稀疏（Structured Sparsity）技术**，同时引入 **GDDR 5 X** 显存和 **NVLink 2.0**（带宽 300 GB/s）。Volta 架构成为 AI 训练和推理的里程碑，V 100 GPU 至今仍广泛应用于数据中心。| 7.0      | Tesla V 100  <br>Quadro GV 100                  |
| **Turing**       | 2018 年   | 晶体管：186 亿  <br>CUDA 核心：2560  <br>RT Core：32  <br>Tensor Core：256  <br>制程：12 nm | 首次集成专用**光线追踪核心（RT Core）**，支持实时光线追踪加速（较 CPU 快 100 倍），并推出 **DLSS** 1.0（深度学习超采样）。引入 RTX 平台，将光线追踪、深度学习和栅格化技术深度融合，重新定义游戏和专业图形渲染标准。| 7.5      | GeForce RTX 2080  <br>Tesla T 4                |
| **Ampere**       | 2020 年   | 晶体管：540 亿  <br>CUDA 核心：5376  <br>第三代 Tensor Core：432  <br>制程：7 nm              | 第三代 Tensor Core 支持 **TF 32** 精度（性能提升 20 倍），第二代 RT Core（光线追踪性能翻倍），并引入**多实例 GPU（MIG）技术**，支持 GPU 资源细粒度分割。第三代 NVLink 带宽达 600 GB/s，A 100 GPU 成为超算和 AI 训练的标杆。消费级显卡（如 RTX 3090）首次实现 24 GB GDDR 6 X 显存，推动 8 K 游戏和内容创作。| 8.0/8.6  | GeForce RTX 3090  <br>A 100 Tensor Core GPU    |
| **Ada Lovelace** | 2022 年   | 晶体管：760 亿  <br>CUDA 核心：16384  <br>第四代 Tensor Core：512  <br>制程：4 nm             | 第四代 Tensor Core 支持 **FP 8** 精度（AI 推理性能提升 4 倍），第三代 RT Core（光线追踪性能提升 2 倍），并推出 DLSS 3.0（结合光线重建技术）。AV 1 编码加速引擎支持 8 K 视频实时处理，同时 Ada 架构首次在消费级显卡中实现 12 层光追计算，推动影视渲染和虚拟制作进入实时时代。| 8.9      | GeForce RTX 4090  <br>RTX 6000 Ada Generation |
| **Hopper**       | 2022 年   | 晶体管：800 亿  <br>CUDA 核心：6080  <br>第四代 Tensor Core：608  <br>制程：4 nm              | 第四代 Tensor Core 支持 **Transformer** 引擎（AI 训练速度提升 30 倍），DPX 指令（动态编程加速 40 倍），并引入机密计算（保护数据隐私）。第四代 NVLink 带宽达 900 GB/s，H 100 GPU 首次实现 900 GB/s 显存带宽，成为百亿亿次超算和万亿参数大模型的核心。| 9.0      | H 100 Tensor Core GPU  <br>H 200 NVL            |
| **Blackwell**    | 2024 年   | 晶体管：2080 亿  <br>CUDA 核心：28160  <br>第五代 Tensor Core：2240  <br>制程：4 nm           | 第二代 Transformer 引擎支持 **FP 4** 精度（AI 算力达 20 PetaFLOPS），第五代 NVLink 带宽达 1.8 TB/s，支持多 GPU 集群无缝互联。新增解压缩引擎（数据库查询加速 5 倍）、RAS 引擎（故障预测与修复），并首次实现芯片级机密计算。GB 200 超级芯片（双 B 200+Grace CPU）推理性能较 H 100 提升 30 倍，成本和能耗降低至 1/25。| 9.6      | B 200 GPU  <br>GB 200 Superchip                 |
