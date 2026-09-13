# shared_memory

我按当前实现把 `shared_memory.hpp` 的设计目的和关键调用链一起拆开讲，重点解释为什么它既能做 CUDA IPC，又能在 fabric 模式下做共享显存。

## 设计目标：它不是“普通的共享内存”，而是“跨 GPU / 跨进程可见的通信缓冲区”

这段实现位于 shared_memory.hpp。它的核心目的，是把一块 GPU 显存，变成所有参与通信的 rank 都能访问的共享资源。

这正好对应 buffer.hpp 里 `Buffer::sync()` 的逻辑：

- 每个 rank 都先分配自己的 local buffer
- 生成一个“共享句柄”
- 把 handle 广播给其他 rank
- 其他 rank 用这个 handle 打开 / 映射
- 最后所有 rank 都能访问到同一块逻辑上的共享内存

也就是说，`shared_memory.hpp` 是整个 EP 通信的底层“地址共享层”。

---

## 1. `MemHandleInner`：统一不同共享内存句柄的格式

代码开头：

```cpp
union MemHandleInner {
    cudaIpcMemHandle_t cuda_ipc_mem_handle;
    CUmemFabricHandle cu_mem_fabric_handle;
};
```

这里用 `union` 的关键原因是：

- CUDA IPC 的句柄类型是 `cudaIpcMemHandle_t`
- fabric 模式下的句柄类型是 `CUmemFabricHandle`
- 两者在语义上都代表“一个远端/共享内存的身份凭证”
- 但抽象上都可以看作“可传递的共享内存句柄”

所以它们被统一放进同一个结构体 `MemHandle`：

```cpp
struct MemHandle {
    MemHandleInner inner;
    size_t size;
};
```

这里 `size` 也保存下来，因为在打开共享内存时，远端需要知道这块区间大小，才能做 `AddressReserve` / `Map` / `Unmap`。

> 这就是设计上的统一：同一份数据结构既支持 Normal CUDA IPC，也支持 Fabric 共享内存。

---

## 2. `SharedMemoryAllocator`：提供统一的 alloc / export / import / close 接口

这个 class 的意义很明确：它把“共享内存管理”抽象成统一接口：

- `malloc`
- `free`
- `get_mem_handle`
- `open_mem_handle`
- `close_mem_handle`

这样上层代码不需要区分是：

- `cudaMalloc + cudaIpcGetMemHandle`
- 还是 `cuMemCreate + CUmem fabric handle`

它只需要调用统一 API 即可。

---

## 3. `malloc`: 两种模式的分配方式不同

### 3.1 普通模式：`cudaMalloc`

```cpp
CUDA_RUNTIME_CHECK(cudaMalloc(ptr, size));
```

这是最常见的方式。  
每个 rank 直接申请一块 GPU 显存，然后拿 `cudaIpcGetMemHandle` 导出。

这适合普通 CUDA IPC 场景。

---

### 3.2 Fabric 模式：`cuMemCreate` + `cuMemAddressReserve` + `cuMemMap`

```cpp
CUdevice device;
CUDA_DRIVER_CHECK(lazy_cuCtxGetDevice(&device));

CUmemAllocationProp prop = {};
prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
prop.location.id = device;

size_t alignment = 0;
CUDA_DRIVER_CHECK(lazy_cuMemGetAllocationGranularity(&alignment, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
size = ((size + alignment - 1) / alignment) * alignment;

CUmemGenericAllocationHandle handle;
CUDA_DRIVER_CHECK(lazy_cuMemCreate(&handle, size, &prop, 0));
CUDA_DRIVER_CHECK(lazy_cuMemAddressReserve(reinterpret_cast<CUdeviceptr*>(ptr), size, alignment, 0, 0));
CUDA_DRIVER_CHECK(lazy_cuMemMap(reinterpret_cast<CUdeviceptr>(*ptr), size, 0, handle, 0));
cu_mem_set_access_all(*ptr, size);
```

这里涉及的是 CUDA driver API 的更底层接口：

- `cuMemCreate`: 创建一块可共享的 GPU allocation
- `cuMemAddressReserve`: 在当前进程 VA 空间里保留虚拟地址区域
- `cuMemMap`: 把 allocation 映射进去
- `cu_mem_set_access_all`: 允许所有 device 访问这块 memory

这比 `cudaMalloc` 更底层，也更适合跨 GPU / 跨 node 的共享内存场景。

> 简单来说：fabric 模式把“共享显存”做成了更通用、更底层、更适合多设备协同的机制。

---

## 4. `cu_mem_set_access_all`: 让所有 GPU 都有访问权限

```cpp
static void cu_mem_set_access_all(void* ptr, size_t size) {
    int device_count;
    CUDA_RUNTIME_CHECK(cudaGetDeviceCount(&device_count));

    constexpr int kMaxDeviceCount = 8;
    EP_HOST_ASSERT(0 < device_count and device_count <= kMaxDeviceCount);

    CUmemAccessDesc access_desc[kMaxDeviceCount];
    for (int i = 0; i < device_count; ++ i) {
        access_desc[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc[i].location.id = i;
        access_desc[i].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }
    CUDA_DRIVER_CHECK(lazy_cuMemSetAccess(reinterpret_cast<CUdeviceptr>(ptr), size, access_desc, device_count));
}
```

这个函数非常关键，因为它对共享内存做了一步“授权”。

意思是：

- 这块显存不只属于当前 GPU
- 允许本机所有 GPU 都能读写它
- 否则远端 GPU 访问就会因为权限不足失败

这也是为什么它要在 alloc 完成后立即调用 `cuMemSetAccess`。

---

## 5. `get_mem_handle`: 导出共享句柄

```cpp
void get_mem_handle(MemHandle* mem_handle, void* ptr) const {
    size_t size = 0;
    CUDA_DRIVER_CHECK(lazy_cuMemGetAddressRange_v2(nullptr, &size, reinterpret_cast<CUdeviceptr>(ptr)));
    mem_handle->size = size;

    if (use_fabric) {
        CUmemGenericAllocationHandle handle;
        CUDA_DRIVER_CHECK(lazy_cuMemRetainAllocationHandle(&handle, ptr));
        CUDA_DRIVER_CHECK(lazy_cuMemExportToShareableHandle(&mem_handle->inner.cu_mem_fabric_handle, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
    } else {
        CUDA_RUNTIME_CHECK(cudaIpcGetMemHandle(&mem_handle->inner.cuda_ipc_mem_handle, ptr));
    }
}
```

### 普通 IPC 路径
```cpp
cudaIpcGetMemHandle(...)
```

会得到一个可跨进程传递的 IPC handle。

### Fabric 路径
```cpp
cuMemRetainAllocationHandle(...)
cuMemExportToShareableHandle(...)
```

这是更底层的“导出共享 allocation”机制。

它的意义是：

- 这块 GPU memory 不是只在当前进程可见
- 它被编码成一个“可以跨进程复现的远端引用”
- 这样其他 MPI rank / GPU 进程才能打开它

---

## 6. `open_mem_handle`: 在远端进程中重新映射共享内存

```cpp
void open_mem_handle(void** ptr, MemHandle* mem_handle) const {
    if (use_fabric) {
        size_t size = mem_handle->size;

        CUmemGenericAllocationHandle handle;
        CUDA_DRIVER_CHECK(lazy_cuMemImportFromShareableHandle(&handle, &mem_handle->inner.cu_mem_fabric_handle, CU_MEM_HANDLE_TYPE_FABRIC));

        CUDA_DRIVER_CHECK(lazy_cuMemAddressReserve(reinterpret_cast<CUdeviceptr*>(ptr), size, 0, 0, 0));
        CUDA_DRIVER_CHECK(lazy_cuMemMap(reinterpret_cast<CUdeviceptr>(*ptr), size, 0, handle, 0));
        cu_mem_set_access_all(*ptr, size);
    } else {
        CUDA_RUNTIME_CHECK(cudaIpcOpenMemHandle(ptr, mem_handle->inner.cuda_ipc_mem_handle, cudaIpcMemLazyEnablePeerAccess));
    }
}
```

这一步是“远端进程接收 handle 后，重新建立内存映射”。

### 普通模式
`cudaIpcOpenMemHandle` 直接把远端的 IPC handle 打开，并开启 peer access。

### Fabric 模式
`cuMemImportFromShareableHandle` 将 handle 恢复成 allocation handle，然后：

- `cuMemAddressReserve`
- `cuMemMap`
- `cu_mem_set_access_all`

重新把它映射到当前进程地址空间。

> 这就是共享内存在跨进程中的“打开连接”过程。

---

## 7. `close_mem_handle`: 释放远端映射

```cpp
void close_mem_handle(void* ptr) const {
    if (use_fabric) {
        cu_mem_free(ptr);
    } else {
        CUDA_RUNTIME_CHECK(cudaIpcCloseMemHandle(ptr));
    }
}
```

它和 `open_mem_handle` 是一对。

- 普通 IPC：`cudaIpcCloseMemHandle`
- fabric：`cu_mem_free`

底层语义是相同的：把当前进程对这段共享显存的映射关闭，并释放 VMA / handle。

---

## 8. `cu_mem_free`: fabric 释放时的底层逻辑

```cpp
static void cu_mem_free(void* ptr) {
    CUmemGenericAllocationHandle handle;
    CUDA_DRIVER_CHECK(lazy_cuMemRetainAllocationHandle(&handle, ptr));

    size_t size = 0;
    CUDA_DRIVER_CHECK(lazy_cuMemGetAddressRange_v2(nullptr, &size, reinterpret_cast<CUdeviceptr>(ptr)));

    CUDA_DRIVER_CHECK(lazy_cuMemUnmap(reinterpret_cast<CUdeviceptr>(ptr), size));
    CUDA_DRIVER_CHECK(lazy_cuMemAddressFree(reinterpret_cast<CUdeviceptr>(ptr), size));
    CUDA_DRIVER_CHECK(lazy_cuMemRelease(handle));
}
```

这个函数非常重要，因为 fabric 分配不是 `cudaFree` 那种简单释放，而是：

1. retain handle
2. query address range
3. unmap
4. address free
5. release allocation

这说明它的共享内存模型是“虚拟地址 + allocation handle + map/unmap”组合，而不是普通 `cudaMalloc` 的简单二元关系。

---

## 9. 为什么需要这个抽象层

这个设计的核心价值有 3 点：

### 9.1 统一不同底层共享内存实现
上层不需要知道究竟是：

- CUDA IPC
- Fabric
- or something else

它只需要调用 `SharedMemoryAllocator`。

### 9.2 让跨 rank 的 GPU 地址可统一访问
在 buffer.hpp 的 `sync()` 中，每个 rank 都会：

- 分配自己的 buffer
- export handle
- all_gather 其他 rank 的 handle
- open remote handle

这一步实际上就是建立“所有 rank 都能访问所有共享 buffer”的基础设施。

### 9.3 兼容多种硬件 / 多种 transport 路径
这个文件是为了把“共享显存”抽象成“跨进程可见的 memory block”，从而支持：

- NVLink local peer access
- RDMA / low-latency inter-node path
- fabric-aware allocation
- 未来更高层的通信封装

也就是说，`shared_memory.hpp` 不是为了“存数据”，而是为了“给不同 rank 提供统一的共享地址能力”。

---

## 10. 一句话总结

`shared_memory.hpp` 的设计本质是：

“把 GPU 显存包装成一个跨进程、跨 GPU 统一的共享内存抽象，支持普通 CUDA IPC 和 Fabric 两种底层实现，并为 EP 通信中的 buffer 共享提供统一入口。”

它是 buffer.hpp 里 `sync()` 能成立的关键基础设施。

---

如果你愿意，我下一步可以继续把它和 `Buffer::sync()` 的调用关系串起来，按这个顺序讲：

1. `Buffer` 构造
2. 分配本地 buffer
3. `get_mem_handle`
4. all-gather handle
5. `open_mem_handle`
6. 共享访问 ready
7. dispatch/combine 开始

这样会非常清楚。