---
dateCreated: 2025-08-10
dateModified: 2025-08-10
---
## BLAS Library Selection

PyTorch provides mechanisms to select the preferred BLAS library at runtime, allowing users to choose between different implementations based on their workload characteristics.

### BLAS Backend Selection

![](assets/Linear_Algebra_Backends.assets/BLAS%20Backend%20Selection.png)

**BLAS Backend Selection Process**

The BLAS backend selection process allows users to choose between different BLAS libraries at runtime. The `preferred_blas_library()` function can be used to set the preferred backend, and the `blas_library_context()` context manager can be used to temporarily change the backend for a specific code block.

Sources:[](https://github.com/pytorch/pytorch/blob/3f1636eb/test/test_matmul_cuda.py#L65-L72)

[test/test_matmul_cuda.py65-72](https://github.com/pytorch/pytorch/blob/3f1636eb/test/test_matmul_cuda.py#L65-L72) [](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/Context.cpp#L1000-L1050)[aten/src/ATen/Context.cpp1000-1050](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/Context.cpp#L1000-L1050)[](https://github.com/pytorch/pytorch/blob/3f1636eb/torch/backends/cuda/__init__.py#L1-L100)

[torch/backends/cuda/__init__.py1-100](https://github.com/pytorch/pytorch/blob/3f1636eb/torch/backends/cuda/__init__.py#L1-L100)

## Performance Considerations

When working with PyTorch's linear algebra backends, several factors can significantly impact performance:

1. **Data Type Selection**:

    - FP16/BF16 operations can be significantly faster on modern GPUs with tensor cores
    - TF32 mode provides a good balance between precision and performance for FP32 operations
2. **Memory Layout**:

    - Contiguous tensors generally perform better than strided ones
    - Column-major layout may be more efficient for certain operations due to BLAS conventions
3. **Batch Size Optimization**:

    - Using batched operations (bmm, baddbmm) is more efficient than loops of individual operations
    - Very large batch sizes may require workspace size adjustments
4. **Backend Selection**:

    - cuBLASLt/hipBLASLt generally perform better for operations supported by tensor cores
    - Traditional cuBLAS/rocBLAS may be faster for small matrices or unusual shapes
5. **Tuning System**:

    - Enable the tunable operations system for workloads with consistent matrix shapes
    - Pre-tune operations during initialization to avoid runtime tuning overhead

Sources:[](https://github.com/pytorch/pytorch/blob/3f1636eb/test/test_matmul_cuda.py#L82-L143)

[test/test_matmul_cuda.py82-143](https://github.com/pytorch/pytorch/blob/3f1636eb/test/test_matmul_cuda.py#L82-L143) [](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L387-L437) [aten/src/ATen/cuda/CUDABlas.cpp387-437](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L387-L437) [](https://github.com/pytorch/pytorch/blob/3f1636eb/torch/cuda/tunable.py#L1-L298) [torch/cuda/tunable.py1-298](https://github.com/pytorch/pytorch/blob/3f1636eb/torch/cuda/tunable.py#L1-L298)

# Linear Algebra Backends

This page documents the linear algebra backend system in PyTorch, which provides high-performance implementations of matrix and vector operations (such as GEMM, GEMV, and batched GEMM) across supported hardware and BLAS libraries. The system includes backend selection, data type and precision management, and a tunable operations system for automatic performance optimization.

For details on how these backends are invoked from the compilation pipeline, see [TorchInductor](https://deepwiki.com/pytorch/pytorch/2.2-torchinductor). For device-specific execution details, see [Device Backends](https://deepwiki.com/pytorch/pytorch/3-device-backends).

## System Architecture

PyTorch's linear algebra backend system is structured to provide a unified interface for tensor operations, while abstracting over hardware-specific and library-specific optimizations. The system routes high-level operations to the appropriate backend implementation, with support for runtime backend selection and autotuning.

#### Diagram: High-Level Operation Routing to Backend Code Entities

![](assets/Linear_Algebra_Backends.assets/High-Level%20Operation%20Routing%20to%20Backend%20Code%20Entities.png)

**Key Code Entities:**

- `at::cuda::blas::gemm`, `at::cuda::blas::bgemm`, `at::cuda::blas::gemm_and_bias`: Core dispatch points for matrix operations ([](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.h)

[aten/src/ATen/cuda/CUDABlas.h](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.h)[](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/native/cuda/Blas.cpp)- [aten/src/ATen/native/cuda/Blas.cpp](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/native/cuda/Blas.cpp)).

- `BlasBackend` enum: Represents backend choices ([](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/Context.cpp)
- [aten/src/ATen/Context.cpp](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/Context.cpp)).
- `TunableOp System`, `GemmTunableOp`: Tunable operation infrastructure ([](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/tunable/TunableGemm.h)
- [aten/src/ATen/cuda/tunable/TunableGemm.h](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/tunable/TunableGemm.h)).

Sources:[](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/native/cuda/Blas.cpp#L323-L448)

[aten/src/ATen/native/cuda/Blas.cpp323-448](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/native/cuda/Blas.cpp#L323-L448) [](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L367-L563) [aten/src/ATen/cuda/CUDABlas.cpp367-563](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L367-L563) [](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.h#L40-L231) [aten/src/ATen/cuda/CUDABlas.h40-231](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.h#L40-L231) [](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/tunable/TunableGemm.h#L29-L239) [aten/src/ATen/cuda/tunable/TunableGemm.h29-239](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/tunable/TunableGemm.h#L29-L239)

## Backend Implementation Layer

The core linear algebra operations are implemented through a template-based system that supports multiple data types and backend libraries.

### Backend Implementation: CUDA and ROCm

The CUDA backend supports both cuBLAS and cuBLASLt libraries, with backend selection logic based on operation parameters, data types, and hardware capabilities. ROCm backends (rocBLAS, hipBLASLt) are supported on AMD GPUs.

#### Diagram: Backend Selection and Code Entities

![](assets/Linear_Algebra_Backends.assets/Backend%20Selection%20and%20Code%20Entities.png)

**Key Code Entities:**

- `cublasCommonArgs`: Matrix preparation and transpose logic ([](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/native/cuda/Blas.cpp#L142-L216)
- [aten/src/ATen/native/cuda/Blas.cpp142-216](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/native/cuda/Blas.cpp#L142-L216)).
- `bgemm_internal_cublaslt()`, `bgemm_internal_cublas()`: Batched GEMM implementations ([](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L367-L563)
- [aten/src/ATen/cuda/CUDABlas.cpp367-563](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L367-L563)).
- `CuBlasLtWorkspace`: Workspace management for cuBLASLt ([](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L246-L253)
- [aten/src/ATen/cuda/CUDABlas.cpp246-253](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L246-L253)).

Sources:[](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L136-L189)

[aten/src/ATen/cuda/CUDABlas.cpp136-189](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L136-L189) [](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L367-L563)[aten/src/ATen/cuda/CUDABlas.cpp367-563](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L367-L563)[](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/native/cuda/Blas.cpp#L142-L216)

[aten/src/ATen/native/cuda/Blas.cpp142-216](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/native/cuda/Blas.cpp#L142-L216)

### Data Type and Precision Support

The linear algebra backends support a range of data types and precision modes, including mixed precision and hardware-specific formats.

| Data Type               | cuBLAS | cuBLASLt | ROCm BLAS | hipBLASLt | Mixed Precision / Special Modes |
| ----------------------- | ------ | -------- | --------- | --------- | ------------------------------- |
| `float`                 | ✓      | ✓        | ✓         | ✓         | TF32 (Ampere+), FP32            |
| `double`                | ✓      | ✓        | ✓         | Limited   | -                               |
| `at::Half`              | ✓      | ✓        | ✓         | ✓         | Accumulate in FP32              |
| `at::BFloat16`          | ✓      | ✓        | ✓         | ✓         | Accumulate in FP32              |
| `c10::complex<float>`   | ✓      | ✓        | ✓         | Limited   | -                               |
| `c10::complex<double>`  | ✓      | ✓        | ✓         | Limited   | -                               |
| `float8` (experimental) | -      | -        | -         | ✓         | -                               |

- TF32 is available on NVIDIA Ampere and later GPUs for FP32 inputs.
- Mixed precision: Half/BFloat16 inputs can use FP32 accumulation and output.

Sources:[](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L387-L437)

[aten/src/ATen/cuda/CUDABlas.cpp387-437](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L387-L437) [](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/native/cuda/Blas.cpp#L360-L402)[aten/src/ATen/native/cuda/Blas.cpp360-402](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/native/cuda/Blas.cpp#L360-L402) [](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/Context.cpp#L1000-L1050)[aten/src/ATen/Context.cpp1000-1050](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/Context.cpp#L1000-L1050)[](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/tunable/GemmCommon.h#L106-L215)

[aten/src/ATen/cuda/tunable/GemmCommon.h106-215](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/tunable/GemmCommon.h#L106-L215)

## Tunable Operations System

PyTorch provides a tunable operations system for linear algebra, which benchmarks and selects the fastest backend implementation for a given operation signature and input parameters. This system is especially relevant for GEMM and batched GEMM operations on ROCm and CUDA platforms.

#### Diagram: TunableOp System and Code Entities

![](assets/Linear_Algebra_Backends.assets/TunableOp%20System%20and%20Code%20Entities.png)

**Key Code Entities:**

- `GemmParams<T>`: Encapsulates operation parameters ([](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/tunable/GemmCommon.h#L283-L377)
- [aten/src/ATen/cuda/tunable/GemmCommon.h283-377](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/tunable/GemmCommon.h#L283-L377)).
- `TuningContext`: Manages tuning and caching ([](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/tunable/TunableGemm.h#L210-L239)
- [aten/src/ATen/cuda/tunable/TunableGemm.h210-239](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/tunable/TunableGemm.h#L210-L239)).
- `Callable<GemmParams<T>>`: Interface for backend implementations ([](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/tunable/TunableGemm.h#L29-L43)
- [aten/src/ATen/cuda/tunable/TunableGemm.h29-43](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/tunable/TunableGemm.h#L29-L43)).

Sources:[](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/tunable/TunableGemm.h#L206-L239)

[aten/src/ATen/cuda/tunable/TunableGemm.h206-239](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/tunable/TunableGemm.h#L206-L239) [](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/tunable/GemmCommon.h#L31-L377)[aten/src/ATen/cuda/tunable/GemmCommon.h31-377](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/tunable/GemmCommon.h#L31-L377)[](https://github.com/pytorch/pytorch/blob/3f1636eb/torch/cuda/tunable.py#L1-L196)

[torch/cuda/tunable.py1-196](https://github.com/pytorch/pytorch/blob/3f1636eb/torch/cuda/tunable.py#L1-L196)

### Backend Registration and Environment Controls

The tunable operations system supports registration of multiple backend implementations, with selection and tuning controlled by environment variables and runtime APIs.

#### Diagram: Backend Registration and Environment Controls

![](assets/Linear_Algebra_Backends.assets/Backend%20Registration%20and%20Environment%20Contro.png)

- Each backend provides a unique name and callable implementation.
- Environment variables control which backends are enabled and whether tuning is active.

Sources:[](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/tunable/TunableGemm.h#L210-L230)

[aten/src/ATen/cuda/tunable/TunableGemm.h210-230](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/tunable/TunableGemm.h#L210-L230) [](https://github.com/pytorch/pytorch/blob/3f1636eb/torch/cuda/tunable.py#L1-L196)[torch/cuda/tunable.py1-196](https://github.com/pytorch/pytorch/blob/3f1636eb/torch/cuda/tunable.py#L1-L196)[](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/tunable/GemmCommon.h#L31-L375)

[aten/src/ATen/cuda/tunable/GemmCommon.h31-375](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/tunable/GemmCommon.h#L31-L375)

## Backend Selection Logic

Backend selection is performed dynamically based on hardware capabilities, operation parameters, and user or environment configuration.

#### Diagram: Backend Selection Decision Flow

![](assets/Linear_Algebra_Backends.assets/Backend%20Selection%20Decision%20Flow.png)

- The selection logic is implemented in `blasPreferredBackend()` and related helpers.
- Platform checks and parameter evaluation determine the optimal backend for each operation.

Sources:[](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/native/cuda/Blas.cpp#L341-L402)

[aten/src/ATen/native/cuda/Blas.cpp341-402](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/native/cuda/Blas.cpp#L341-L402) [](https://github.com/pytorch/pytorch/blob/3f1636eb/test/test_linalg.py#L55-L70)[test/test_linalg.py55-70](https://github.com/pytorch/pytorch/blob/3f1636eb/test/test_linalg.py#L55-L70)[](https://github.com/pytorch/pytorch/blob/3f1636eb/torch/testing/_internal/common_cuda.py#L28-L112)

[torch/testing/_internal/common_cuda.py28-112](https://github.com/pytorch/pytorch/blob/3f1636eb/torch/testing/_internal/common_cuda.py#L28-L112)

## ROCm Backend Support

PyTorch supports AMD ROCm platforms with both rocBLAS and hipBLASLt libraries, including architecture-specific optimizations and mixed precision support.

- `rocblas_gemm_strided_batched_ex()`: Used for advanced batched GEMM ([](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L650-L660)
- [aten/src/ATen/cuda/CUDABlas.cpp650-660](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L650-L660)).
- Architecture detection and selection logic for supported GCN architectures ([](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L19-L65)
- [aten/src/ATen/cuda/CUDABlas.cpp19-65](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L19-L65)).
- Mixed precision and TF32-like modes for MI300 and newer.
- Integration with Composable Kernel (CK) for BFloat16 ([](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/native/hip/ck_bgemm.h)
- [aten/src/ATen/native/hip/ck_bgemm.h](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/native/hip/ck_bgemm.h)).

Sources:[](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L19-L65)

[aten/src/ATen/cuda/CUDABlas.cpp19-65](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L19-L65) [](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L193-L227)[aten/src/ATen/cuda/CUDABlas.cpp193-227](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L193-L227) [](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L647-L660)[aten/src/ATen/cuda/CUDABlas.cpp647-660](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L647-L660)[](https://github.com/pytorch/pytorch/blob/3f1636eb/test/test_linalg.py#L55-L70)

[test/test_linalg.py55-70](https://github.com/pytorch/pytorch/blob/3f1636eb/test/test_linalg.py#L55-L70)

## Mixed Precision and Data Type Handling

The linear algebra backends support mixed precision computation, allowing lower-precision inputs (Half, BFloat16) to be accumulated in higher precision, and enabling special hardware modes such as TF32.

#### Diagram: Data Type and Precision Routing

![](assets/Linear_Algebra_Backends.assets/Data%20Type%20and%20Precision%20Routing.png)

- Precision control flags: `allowFP16AccumulationCuBLAS()`, `allowBF16ReductionCuBLAS()`, `allowTF32CuBLAS()`, `setFloat32MatmulPrecision()`.
- TF32 is available for FP32 inputs on Ampere+ GPUs.

Sources:[](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L387-L437)

[aten/src/ATen/cuda/CUDABlas.cpp387-437](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L387-L437) [](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/Context.cpp#L1000-L1050)[aten/src/ATen/Context.cpp1000-1050](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/Context.cpp#L1000-L1050)[](https://github.com/pytorch/pytorch/blob/3f1636eb/test/test_matmul_cuda.py#L82-L143)

[test/test_matmul_cuda.py82-143](https://github.com/pytorch/pytorch/blob/3f1636eb/test/test_matmul_cuda.py#L82-L143)

## Performance Optimization Features

### Workspace and Memory Layout Management

- `CublasLtWorkspace`: Manages temporary workspace for cuBLASLt ([](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L246-L253)
- [aten/src/ATen/cuda/CUDABlas.cpp246-253](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L246-L253)).
- `getCUDABlasLtWorkspaceSize()`: Workspace size is configurable via environment variables.
- Memory layout optimization: Automatic detection of optimal layouts, stride analysis, and transpose logic.

#### Diagram: Memory Layout and Transpose Handling

![](assets/Linear_Algebra_Backends.assets/Memory%20Layout%20and%20Transpose%20Handling.png)

Sources: [](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/native/cuda/Blas.cpp#L142-L216)[aten/src/ATen/native/cuda/Blas.cpp142-216](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/native/cuda/Blas.cpp#L142-L216)[](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L246-L253)

[aten/src/ATen/cuda/CUDABlas.cpp246-253](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L246-L253)

### Fused Activation Epilogues

Modern BLAS libraries support fused activation functions in GEMM operations:

- Supported activations: None, ReLU, GELU (see `GEMMAndBiasActivationEpilogue`).
- Eliminates the need for separate activation kernel launches.

Sources:[](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L235-L266)

[aten/src/ATen/cuda/CUDABlas.cpp235-266](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L235-L266) [](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/native/cuda/Blas.cpp#L136-L189)[aten/src/ATen/native/cuda/Blas.cpp136-189](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/native/cuda/Blas.cpp#L136-L189)[](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/native/cuda/Blas.cpp#L246-L258)

[aten/src/ATen/native/cuda/Blas.cpp246-258](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/native/cuda/Blas.cpp#L246-L258)

## Configuration and Environment Controls

The linear algebra backend system provides extensive configuration options through environment variables and runtime APIs.

### Environment Variable Controls

|Variable|Default|Description|
|---|---|---|
|`PYTORCH_TUNABLEOP_ENABLED`|0|Enable tunable operations|
|`PYTORCH_TUNABLEOP_TUNING`|1|Enable performance tuning|
|`PYTORCH_TUNABLEOP_FILENAME`|tunableop_results.csv|Tuning results file|
|`CUBLASLT_WORKSPACE_SIZE`|1024 (KB)|cuBLASLt workspace size|
|`DISABLE_ADDMM_CUDA_LT`|0|Disable cuBLASLt for addmm|
|`PYTORCH_TUNABLEOP_ROCBLAS_ENABLED`|1|Enable ROCm BLAS tuning|
|`PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED`|1|Enable hipBLASLt tuning|

### Runtime Configuration APIs

**Python Interface:**

```
import torch.cuda.tunable as tunable
tunable.enable(True)
tunable.set_max_tuning_duration(30)
tunable.set_filename("custom_results.csv")
```

**C++ Interface:**

```
auto tuning_ctx = at::cuda::tunable::getTuningContext();
tuning_ctx->EnableTunableOp(true);
tuning_ctx->SetMaxTuningDuration(30.0);
```

Sources:[](https://github.com/pytorch/pytorch/blob/3f1636eb/torch/cuda/tunable.py#L177-L298)

[torch/cuda/tunable.py177-298](https://github.com/pytorch/pytorch/blob/3f1636eb/torch/cuda/tunable.py#L177-L298) [](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/tunable/README.md#L135-L150) [aten/src/ATen/cuda/tunable/README.md135-150](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/tunable/README.md#L135-L150) [](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L188-L223) [aten/src/ATen/cuda/CUDABlas.cpp188-223](https://github.com/pytorch/pytorch/blob/3f1636eb/aten/src/ATen/cuda/CUDABlas.cpp#L188-L223)
