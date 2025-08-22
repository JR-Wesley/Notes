---
dateCreated: 2025-08-08
dateModified: 2025-08-09
---

reference: [NVIDIA/Megatron-LM | DeepWiki]([https://deepwiki.com/NVIDIA/Megatron-LM](https://deepwiki.com/NVIDIA/Megatron-LM))

# Overview

## Purpose and Scope

This document provides a comprehensive overview of the Megatron-LM framework, a GPU-optimized system for training large-scale transformer models. Megatron-LM combines cutting-edge research implementations with production-ready training infrastructure, enabling efficient training of models with hundreds of billions of parameters across thousands of GPUs.

The framework consists of two main components:

- **Megatron-LM** (research-oriented training scripts and model implementations) and
- **Megatron-Core** (production-ready library of GPU-optimized techniques).

This documentation covers the entire system architecture, from core transformer implementations to distributed training orchestration and inference deployment.

For detailed information about specific subsystems, see: [Core Architecture]([https://deepwiki.com/NVIDIA/Megatron-LM/2-core-architecture](https://deepwiki.com/NVIDIA/Megatron-LM/2-core-architecture)), [Parallelism Strategies]([https://deepwiki.com/NVIDIA/Megatron-LM/3-parallelism-strategies](https://deepwiki.com/NVIDIA/Megatron-LM/3-parallelism-strategies)), [Training System]([https://deepwiki.com/NVIDIA/Megatron-LM/4-training-system](https://deepwiki.com/NVIDIA/Megatron-LM/4-training-system)), [Data Processing]([https://deepwiki.com/NVIDIA/Megatron-LM/5-data-processing](https://deepwiki.com/NVIDIA/Megatron-LM/5-data-processing)), [Inference System]([https://deepwiki.com/NVIDIA/Megatron-LM/6-inference-system](https://deepwiki.com/NVIDIA/Megatron-LM/6-inference-system)), [Fine-tuning and Evaluation]([https://deepwiki.com/NVIDIA/Megatron-LM/7-fine-tuning-and-evaluation](https://deepwiki.com/NVIDIA/Megatron-LM/7-fine-tuning-and-evaluation)), and [CI/CD and Testing]([https://deepwiki.com/NVIDIA/Megatron-LM/8-cicd-and-testing](https://deepwiki.com/NVIDIA/Megatron-LM/8-cicd-and-testing)).

## System Architecture Overview

Megatron-LM implements a layered architecture centered around **Megatron-Core** as the foundational library, with training orchestration, model implementations, and infrastructure components built on top. The system separates core GPU-optimized techniques from research implementations and production workflows.

**Overall System Architecture**
![](Overall%20System%20Architecture.png)

**Sources:** [README.md71-86]([https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/README.md#L71-L86](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/README.md#L71-L86)) [megatron/training/arguments.py45-82]([https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/training/arguments.py#L45-L82](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/training/arguments.py#L45-L82)) [megatron/core/transformer/transformer_config.py32-38]([https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/transformer_config.py#L32-L38](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/transformer_config.py#L32-L38)) [megatron/training/training.py1-4]([https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/training/training.py#L1-L4](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/training/training.py#L1-L4))

## Core Components

### Megatron-LM Vs Megatron-Core

The framework separates research implementations from production-ready infrastructure:

- **Megatron-LM**: Training scripts (`pretrain_gpt.py`, `pretrain_bert.py`), tools, examples, and experimental features
- **Megatron-Core**: Production library under `megatron.core` with versioned APIs, optimized implementations, and formal support

**Training Pipeline and Model Architecture Flow**
![](Training%20Pipeline%20and%20Model%20Architecture%20Flow.png)
![[Training Pipeline and Model Architecture Flow.png]]

**Sources:** [README.md75-85]([https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/README.md#L75-L85](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/README.md#L75-L85)) [megatron/training/arguments.py84-119]([https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/training/arguments.py#L84-L119](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/training/arguments.py#L84-L119)) [pretrain_gpt.py95-108]([https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/pretrain_gpt.py#L95-L108](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/pretrain_gpt.py#L95-L108)) [megatron/core/models/gpt/gpt_model.py34-75]([https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/models/gpt/gpt_model.py#L34-L75](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/models/gpt/gpt_model.py#L34-L75)) [megatron/training/training.py1-4]([https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/training/training.py#L1-L4](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/training/training.py#L1-L4))

### Supported Model Architectures

Megatron supports multiple transformer-based architectures through the `TransformerConfig` system:

| Architecture | Implementation                      | Key Classes                   | Use Case                                 |
| ------------ | ----------------------------------- | ----------------------------- | ---------------------------------------- |
| **GPT**      | `megatron.core.models.gpt.GPTModel` | `GPTModel`, `LanguageModule`  | Autoregressive language modeling         |
| **BERT**     | `megatron.legacy.model.BertModel`   | `BertModel`, `MegatronModule` | Bidirectional language understanding     |
| **T5**       | `megatron.core.models.t5.T5Model`   | `T5Model`, encoder-decoder    | Text-to-text generation                  |
| **Retro**    | `megatron.core.models.retro`        | `RetroModel`                  | Retrieval-augmented generation           |
| **MoE**      | `megatron.core.transformer.moe`     | `MoELayer`, expert routing    | Mixture of experts scaling               |
| **LLaVA**    | `megatron.core.models.multimodal`   | Multimodal fusion             | Vision-language models                   |
| **Mamba**    | `examples/mamba/`                   | State space models            | Sequence modeling with linear complexity |

**Sources:** [README.md210-380]([https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/README.md#L210-L380](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/README.md#L210-L380)) [megatron/core/models/gpt/gpt_model.py34-75]([https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/models/gpt/gpt_model.py#L34-L75](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/models/gpt/gpt_model.py#L34-L75)) [megatron/core/transformer/transformer_config.py32-38]([https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/transformer_config.py#L32-L38](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/transformer/transformer_config.py#L32-L38))

### Parallelism Strategies

The framework implements comprehensive parallelism support through `megatron.core.parallel_state` and specialized modules:

- **Tensor Parallelism** (`--tensor-model-parallel-size`): `megatron.core.tensor_parallel` splits layers across GPUs
- **Pipeline Parallelism** (`--pipeline-model-parallel-size`): `megatron.core.pipeline_parallel.schedules` distributes transformer blocks
- **Data Parallelism**: `megatron.core.distributed.DistributedDataParallel` replicates models with gradient sync
- **Sequence Parallelism** (`--sequence-parallel`): Distributes sequence dimension in layer norms and dropouts
- **Expert Parallelism** (`--expert-model-parallel-size`): `megatron.core.transformer.moe` distributes MoE experts
- **Context Parallelism** (`--context-parallel-size`): Handles long sequences via `_CONTEXT_PARALLEL_GROUP`

**Sources:** [README.md292-306]([https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/README.md#L292-L306](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/README.md#L292-L306)) [megatron/core/tensor_parallel/layers.py1-4]([https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/tensor_parallel/layers.py#L1-L4](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/tensor_parallel/layers.py#L1-L4)) [megatron/core/pipeline_parallel/schedules.py1-4]([https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/pipeline_parallel/schedules.py#L1-L4](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/pipeline_parallel/schedules.py#L1-L4)) [megatron/core/parallel_state.py22-106]([https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/parallel_state.py#L22-L106](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/core/parallel_state.py#L22-L106))

## Performance and Scalability

### Training Performance

Megatron-LM demonstrates exceptional scaling characteristics across model sizes and hardware configurations:

- **Model Scale**: Supports models from 345M to 462B parameters
- **Hardware Scale**: Tested up to 6,144 H100 GPUs
- **Efficiency**: Achieves 41-48% Model FLOPs Utilization (MFU)
- **Scaling Pattern**: Shows superlinear weak scaling due to improved arithmetic intensity

### Benchmark Results

| Model Size | GPUs | Global Batch Size | MFU | Throughput    |
| ---------- | ---- | ----------------- | --- | ------------- |
| 2B         | 96   | 1152              | 41% | -             |
| 175B       | 1024 | 1536              | 47% | 138 TFLOP/GPU |
| 462B       | 6144 | -                 | 48% | -             |

**Sources:** [README.md87-100]([https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/README.md#L87-L100](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/README.md#L87-L100))

## Getting Started

### Entry Points

The framework provides multiple entry points for different use cases:

**Training Scripts:**

- `pretrain_gpt.py`: GPT model pretraining
- `pretrain_bert.py`: BERT model pretraining
- `pretrain_t5.py`: T5 model pretraining

**Data Processing:**

- `tools/preprocess_data.py`: Convert raw text to training format

**Inference:**

- `tools/run_text_generation_server.py`: REST API server for text generation
- `tools/text_generation_cli.py`: Command-line interface for inference

**Evaluation:**

- `tasks/main.py`: Downstream task evaluation

### Basic Usage Pattern

![](Basic%20Usage%20Pattern.png)

**Sources:** [README.md199-212]([https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/README.md#L199-L212](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/README.md#L199-L212)) [README.md492-509]([https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/README.md#L492-L509](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/README.md#L492-L509)) [pretrain_gpt.py1-10]([https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/pretrain_gpt.py#L1-L10](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/pretrain_gpt.py#L1-L10)) [megatron/training/checkpointing.py1-4]([https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/training/checkpointing.py#L1-L4](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/megatron/training/checkpointing.py#L1-L4))

## Installation and Prerequisites

### Dependencies

- **PyTorch**: Latest stable version
- **CUDA/cuDNN/NCCL**: Latest stable versions
- **Hardware**: NVIDIA Turing generation GPUs or later for best performance
- **FP8 Support**: Available on Hopper, Ada, and Blackwell architectures

### Installation Options

- **PyPI**: `pip install megatron-core[dev]` for latest features
- **Docker**: NVIDIA PyTorch NGC Container (recommended)
- **Source**: Git clone with environment setup via `docker/common/install.sh`

**Sources:** [README.md101-184]([https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/README.md#L101-L184](https://github.com/NVIDIA/Megatron-LM/blob/bbb4c5fb/README.md#L101-L184))
