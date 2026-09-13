# NVSHMEM 最小示例

下面的程序演示一个环形 shift：每个 PE 把自己的 world PE 编号写入下一个 PE 的 `destination`。该 kernel 只执行单向标量 put，不在 kernel 内调用 wait、barrier 或 collective，因此可以用普通 CUDA kernel launch；kernel 结束后在 stream 上执行全局 barrier，再把本地结果复制到 host。

## 程序

```cpp
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t status_ = (call);                                      \
        if (status_ != cudaSuccess) {                                      \
            std::fprintf(stderr, "CUDA error: %s\n",                     \
                         cudaGetErrorString(status_));                     \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

__global__ void shift_right(int *destination) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int me = nvshmem_my_pe();
        int next = (me + 1) % nvshmem_n_pes();
        nvshmem_int_p(destination, me, next);
    }
}

int main() {
    nvshmem_init();

    int me = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int local_pe = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

    CUDA_CHECK(cudaSetDevice(local_pe));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int *destination = static_cast<int *>(nvshmem_malloc(sizeof(int)));
    if (destination == nullptr) {
        std::fprintf(stderr, "PE %d: symmetric allocation failed\n", me);
        nvshmem_global_exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaMemsetAsync(destination, 0xff, sizeof(int), stream));
    nvshmemx_barrier_all_on_stream(stream);

    shift_right<<<1, 1, 0, stream>>>(destination);
    CUDA_CHECK(cudaGetLastError());

    nvshmemx_barrier_all_on_stream(stream);

    int received = -1;
    CUDA_CHECK(cudaMemcpyAsync(&received, destination, sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int expected = (me - 1 + npes) % npes;
    std::printf("PE %d received %d, expected %d\n", me, received, expected);

    nvshmem_free(destination);
    CUDA_CHECK(cudaStreamDestroy(stream));
    nvshmem_finalize();
    return received == expected ? EXIT_SUCCESS : EXIT_FAILURE;
}
```

## 代码中的关键顺序

1. `nvshmem_init()` 建立 PE 世界。
2. `nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)` 得到节点内编号，用于选择 GPU；通信仍使用 world PE 编号。
3. 所有 PE 调用 `nvshmem_malloc(sizeof(int))`，创建对称对象。
4. 首个 on-stream barrier 确保所有 PE 完成本地初始化后再通信。
5. kernel 中的 PE `i` 对 PE `(i + 1) % npes` 执行标量 put。
6. 第二个 on-stream barrier 等待所有 PE 此前的远端更新完成，并让各 PE 在同一阶段继续。
7. `cudaMemcpyAsync` 读取当前 PE 本地的 `destination`，其期望值来自左邻居。
8. 所有 PE 按相同顺序释放对称对象并 finalize。

这个示例用 barrier 简化阶段同步。实际邻域算法若不要求所有 PE 会合，通常应改成 put-with-signal 加 wait/test，以缩小同步范围，见 [B02：完成、排序与可见性](../b02-completeness-ordering-and-visibility.md)。

## 编译

根据官方当前文档，NVSHMEM 程序需要同时链接 host 和 device 库，并启用 relocatable device code：

```bash
nvcc -rdc=true -ccbin g++ \
  -gencode="$NVCC_GENCODE" \
  -I "$NVSHMEM_HOME/include" \
  nvshmem_shift.cu -o nvshmem_shift \
  -L "$NVSHMEM_HOME/lib" \
  -lnvshmem_host -lnvshmem_device
```

`NVCC_GENCODE` 需要替换为目标 GPU 的 `-gencode` 参数内容。若 NVSHMEM 使用 UCX、IBGDA 或其他可选 transport 构建，还可能需要安装指南中对应的链接参数和运行时库路径。

## 运行

NVSHMEM 可通过兼容 PMI/PMIx 的进程管理器启动；MPI 不是 API 使用的必需依赖。具体命令取决于 NVSHMEM 的构建和集群环境，例如：

```bash
mpirun -n 4 ./nvshmem_shift
```

运行前至少检查：

- 每个节点可用 GPU 数量与本节点 PE 数量匹配；
- 所有节点的可执行文件和动态库路径一致；
- `LD_LIBRARY_PATH` 能找到 NVSHMEM、CUDA 以及所选 transport 的库；
- 跨节点 GPU 与 NIC 满足 P2P 或 GPUDirect RDMA 条件；
- launcher 所用 PMI/PMIx 与 NVSHMEM 构建配置兼容。

## 何时必须改用 collective launch

若把 kernel 改成下面这种在 device 内等待 signal 或调用 barrier/collective 的形式，就不能继续使用普通 `<<<...>>>`：

```cpp
__global__ void communicating_kernel(/* ... */) {
    // 发起通信
    // nvshmem_wait_until(...);  // device 侧阻塞同步
    // 继续计算
}
```

此时必须通过 `nvshmemx_collective_launch` 由所有 PE 集合启动，并确保网格不超过 cooperative launch 可并发执行的规模。原因是 CUDA 不保证普通网格中的所有 block 同时驻留；等待未调度 block 或另一 PE 的工作可能导致死锁。

## 参考资料

- [Using NVSHMEM: Example Program, Compiling and Running](https://docs.nvidia.com/nvshmem/api/latest/using.html)
- [NVSHMEM Examples](https://docs.nvidia.com/nvshmem/api/latest/examples.html)
- [Kernel Launch Routines](https://docs.nvidia.com/nvshmem/api/latest/api/launch.html)
- [NVSHMEM Installation Guide](https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/abstract.html)
