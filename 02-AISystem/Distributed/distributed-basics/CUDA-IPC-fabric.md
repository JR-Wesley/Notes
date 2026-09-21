# CUDA MemHandleInner 联合体解析

```cpp
union MemHandleInner {
    cudaIpcMemHandle_t cuda_ipc_mem_handle;
    CUmemFabricHandle cu_mem_fabric_handle;
};
```

> 这是一个**联合体 (union)**，两块内存占用**同一块存储空间**，同一时刻只能有效使用其中一个成员。

- `cudaIpcMemHandle_t`：传统 CUDA IPC 句柄，用于**同节点多进程之间**共享 GPU 显存。
- `CUmemFabricHandle`：NVLink Fabric 句柄，用于**跨节点（多机）GPU Fabric 集群**显存共享，属于 CUDA 9+ / NVSwitch 生态。

> 注意：`cudaIpcMemHandle_t` 属于 Runtime API；`CUmemFabricHandle` 属于 Driver API。

---

## 1. cudaIpcMemHandle_t — 单机 IPC 显存句柄

### 是什么

`cudaIpcMemHandle_t` 是 CUDA Runtime API 提供的**单机进程间通信句柄**。

场景：**同一台机器，多个进程，多块 GPU，GPU 之间通过 PCIe/NVLink，但是所有进程在同一个主机操作系统下**。

### 工作流程

1. 进程 A 调用 `cudaMalloc()` 分配显存。
2. 调用 `cudaIpcGetMemHandle(&handle, d_ptr)`，把设备指针转换成 `cudaIpcMemHandle_t`。
3. 将这个句柄通过 socket / 管道 / 文件传给本机另一个进程 B。
4. 进程 B 拿到句柄，调用 `cudaIpcOpenMemHandle(&remote_dptr, handle, flags)`，得到可以直接访问的设备指针。
5. B 就可以在自己进程内，直接读写 A 分配的显存。
6. 使用完毕：`cudaIpcCloseMemHandle(remote_dptr)`。

### 关键限制（非常重要）

1. **仅限同一台主机，不能跨机器**。句柄里面包含 GPU 物理信息，无法通过网络传到另一台服务器。
2. 原始分配进程 A 不能释放显存，否则 B 的指针直接非法，访问会报错。
3. 句柄不携带内存，只是一个“凭证”，很小的二进制结构体，可以序列化传递。
4. 只支持设备显存，不支持系统内存。
5. 多 GPU 本机 NVLink 是可以走这套 API，但是仍然是单机。

### 典型使用场景

- 单机多进程推理服务，进程 0 分配大 KV Cache，其他 worker 进程直接 IPC 打开共享显存，避免重复分配显存。
- DeepSpeed / Megatron‑L 单机多进程训练会用到 cuda IPC。

### 结构体简要

```cpp
typedef struct cudaIpcMemHandle_st {
    char reserved[64];
} cudaIpcMemHandle_t;
```

固定 64 字节二进制 blob，内部对用户不透明。

---

## 2. CUmemFabricHandle — Fabric 跨节点句柄（Driver API）

### 是什么

`CUmemFabricHandle` 是 CUDA Driver API，面向 **NVLink Fabric（多机集群）** 的内存句柄。

> 硬件前提：服务器带 NVSwitch，开启 GPU Fabric Manager，多台机器组成 Fabric 域。
> 可以**跨服务器节点**，让一台机器上的进程打开另一台机器 GPU 上的显存。

> 注意：这套 API 和 `cudaIpcMemHandle_t` 不是一套，Runtime 没有封装，只能用 CU Driver API（cu* 系列函数）。

### 工作流程

1. 节点 0 进程：调用 `cuMemAlloc()` 分配显存。
2. 调用 `cuMemExportToFabricHandle(&fabric_handle, devPtr)`，得到 `CUmemFabricHandle`。
3. 把这个 fabric handle 通过网络（RPC/TCP）发送给**另一台物理机器节点 1**。
4. 节点 1 进程调用 `cuMemImportFromFabricHandle(&remote_devptr, fabric_handle)`，得到远端 GPU 的设备地址。
5. 节点 1 的 kernel 可以直接 load/store 访问远端节点 0 GPU 显存。
6. 用完 `cuMemRelease(remote_devptr)`。

### 关键限制

1. **硬件强依赖：NVSwitch + Fabric Manager 必须部署运行**。普通 PCIe 多机不能用。
2. 必须开启 GPU Fabric，GPU 属于同一个 Fabric 域。
3. 原始内存不能释放，否则远端访问直接崩溃。
4. 跨节点访问带宽受 NVLink fabric 约束，比本地显存慢。
5. 这是 Driver API，不能直接传给 cuda runtime 接口。
6. Fabric handle 结构体大小和 cudaIpcMemHandle_t 相同，所以代码中会用 union 共用内存，方便统一序列化存储。

### 典型场景

- DGX 集群多机 Fabric 共享显存，分布式训练，跨节点显存池，Global Memory Pool。
- 大模型分布式推理，跨机直接访问远端 KV cache，不用通过 RDMA 拷贝。

### 结构体

```cpp
typedef struct CUmemFabricHandle_st {
    char reserved[64];
} CUmemFabricHandle;
```

同样 64 字节不透明 blob。**大小完全一样，这就是为什么可以放在同一个 union 里**。

---

## 为什么要用 Union MemHandleInner

```cpp
union MemHandleInner {
    cudaIpcMemHandle_t cuda_ipc_mem_handle;
    CUmemFabricHandle cu_mem_fabric_handle;
};
```

1. 两者结构体都是 64 字节，内存布局大小完全一致，可以复用同一块内存缓冲区。
2. 上层代码只需要一个固定大小的 buffer，根据部署模式（单机 IPC / 多机 Fabric）选择使用哪一个成员。
3. 序列化的时候，直接把 union 的原始二进制发送出去；接收端根据标志位判断，解析成哪一种句柄。
4. 节省内存，统一存储，避免维护两套不同大小的缓冲区。

> ⚠️ union 坑点：**你必须自己额外维护一个 type 标记，记录当前联合体里面存的是哪一种句柄！**
联合体本身没有任何元信息，如果你存的是 fabric handle，却按 cudaIpcMemHandle_t 去解析，程序不会报错，但调用 cudaIpcOpenMemHandle 会直接返回非法参数。

示例逻辑伪代码：

```cpp
enum HandleType {
    HANDLE_CUDA_IPC,
    HANDLE_FABRIC
};

struct MyHandleWrapper {
    HandleType type;
    MemHandleInner inner;
};
```

传输时同时传输 `type + union二进制`，接收方依靠 type 决定调用哪一套 API。

---

## 两者横向对比表

| 项目 | cudaIpcMemHandle_t | CUmemFabricHandle |
|---|---|---|
| API 层级 | CUDA Runtime `cuda*` | CUDA Driver `cu*` |
| 通信范围 | **单机多进程，不能跨机器** | **跨节点多机 Fabric 集群** |
| 硬件依赖 | PCIe/NVLink 单机即可 | 需要 NVSwitch + Fabric Manager |
| 内存访问 | 本机 GPU 显存直接访问 | 跨机器远端 GPU 显存直接访问 |
| 结构体大小 | 64 字节 | 64 字节 |
| 获取句柄 | `cudaIpcGetMemHandle` | `cuMemExportToFabricHandle` |
| 打开句柄 | `cudaIpcOpenMemHandle` | `cuMemImportFromFabricHandle` |
| 关闭/释放 | `cudaIpcCloseMemHandle` | `cuMemRelease` |
| 序列化传输 | socket、管道、共享文件（本机） | RPC/TCP 网络跨机器传输 |

---

## 常见踩坑

1. 把 `CUmemFabricHandle` 传给 `cudaIpcOpenMemHandle`：不会编译报错（union 内存一样），运行时返回 cudaErrorInvalidHandle。
2. 分配进程释放显存，远端进程继续访问：kernel 报错、段错误、GPU hang。
3. Fabric 句柄在没有 NVSwitch 的普通多机环境调用，直接报错，硬件不支持。
4. 不要直接修改 reserved 数组，这是 CUDA 内部不透明 blob。

## 工程实践（Megatron‑L / DeepSpeed 源码里的真实模式）

很多分布式训练代码中就是用这个 union：

- 如果是单机多进程：填充 `cuda_ipc_mem_handle`，type 标记为 IPC；
- 如果运行在 DGX Fabric 集群：填充 `cu_mem_fabric_handle`，type 标记 Fabric；
- 序列化发送二进制 + type；接收端根据 type 选择调用 Runtime IPC 或者 Driver Fabric 接口。

如果你需要，我可以写一份极简可编译最小 demo，分别演示两种句柄的基础调用示例。

# deep_ep 懒加载 CUDA Driver API

> 背景：这是 DeepEP 库内部的**运行时动态加载 CUDA Driver API**代码。
CUDA 分为两套 API：

1. **CUDA Runtime API**：`cudaXXX`，链接 `libcudart.so`，编译期链接，日常最常用；
2. **CUDA Driver API**：`cuXXX`，`libcuda.so.1`，内核态驱动接口，用于 MMA、VMM 虚拟内存、共享句柄、kernel launch ex 等高级能力。

传统方式编译链接 `libcuda.so`，程序启动就强依赖驱动库；这套代码做**延迟动态加载（lazy load）Driver API**，编译阶段不直接链接 `libcuda.so.1`，运行时才打开库、查找符号。

---

## 整体概览

核心两个部件：

1. `get_driver_handle()`：单例获取动态库句柄，`dlopen` 打开 `libcuda.so.1`；
2. 宏 `DECL_LAZY_CUDA_DRIVER_FUNCTION(name)`：模板宏，给每个 `cu*` 驱动函数生成一层封装函数 `lazy_xxx`；
3. 批量展开宏，生成所有需要的 CUDA Driver API 的 lazy wrapper。

> 关键点：编译时**不链接 libcuda.so**，不会在 ELF 的依赖列表写入 libcuda.so。只有代码路径执行到对应 `lazy_cuXxx` 的时候，才会真正打开驱动库、查找函数符号。

---

## 1. `get_driver_handle()` 逐行解析

```cpp
static void* get_driver_handle() {
    static void* handle = nullptr;
    if (handle == nullptr) {
        handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
        EP_HOST_ASSERT(handle != nullptr and "Failed to load CUDA driver `libcuda.so.1`");
    }
    return handle;
}
```

### API 说明：`dlopen` Linux POSIX 动态链接 API

- `static void* handle = nullptr;`：**函数内静态局部变量**，C++11 之后线程安全初始化，整个进程生命周期只初始化一次，单例保存动态库句柄。
- `dlopen("libcuda.so.1", flags)`：打开 CUDA 驱动共享库
  - `libcuda.so.1`：CUDA Driver 的 SONAME，系统驱动提供，不是 CUDA toolkit，是显卡驱动安装出来的库；
  - `RTLD_LAZY`：**延迟符号解析**，只解析被 `dlsym` 显式查找的符号，库内部未使用的符号不做重定位，节省开销；
  - `RTLD_LOCAL`：该库符号**不对外暴露到全局符号表**。其他 `.so` 不能直接看到 libcuda 的符号，避免符号污染、版本冲突。
- `EP_HOST_ASSERT`：DeepEP 自定义断言，如果打开失败直接报错退出，说明机器没有安装 NVIDIA 驱动。

### 行为特征

- 第一次调用：执行 `dlopen` 加载驱动库；后续调用直接返回缓存的 handle；
- **只有第一次执行到任意 lazy_cuXXX 函数的时候，才会触发 dlopen**。如果程序路径完全没有走到这些 driver 接口，`libcuda.so.1` 根本不会被加载进进程。

> 这就是懒加载的第一个好处：环境没有 GPU 时，如果业务逻辑不走 GPU 路径，程序可以正常跑，不会启动直接因为缺失 `libcuda.so.1` 就 crash。如果编译期直接链接 libcuda，程序启动时动态链接器就会检查依赖，没有库直接无法启动。

---

## 2. 核心宏 `DECL_LAZY_CUDA_DRIVER_FUNCTION(name)`

```cpp
#define DECL_LAZY_CUDA_DRIVER_FUNCTION(name) \
template <typename... Args> \
static auto lazy_##name(Args&&... args) -> decltype(name(args...)) { \
    using FuncType = decltype(&name); \
    static FuncType func = nullptr; \
    if (func == nullptr) { \
        func = reinterpret_cast<FuncType>(dlsym(get_driver_handle(), #name)); \
        EP_HOST_ASSERT(func != nullptr and "Failed to load CUDA driver API"); \
    } \
    return func(std::forward<decltype(args)>(args)...); \
}
```

### 宏展开示例

写 `DECL_LAZY_CUDA_DRIVER_FUNCTION(cuGetErrorName)`，预处理器展开得到：

```cpp
template <typename... Args>
static auto lazy_cuGetErrorName(Args&&... args) -> decltype(cuGetErrorName(args...)) {
    using FuncType = decltype(&cuGetErrorName);
    static FuncType func = nullptr;
    if (func == nullptr) {
        func = reinterpret_cast<FuncType>(dlsym(get_driver_handle(), "cuGetErrorName"));
        EP_HOST_ASSERT(func != nullptr and "Failed to load CUDA driver API");
    }
    return func(std::forward<decltype(args)>(args)...);
}
```

逐段拆解语法和原理：

1. **`template<typename... Args>` 可变参数模板**
封装的目标 `cuXXX` 函数参数各不相同，使用可变模板可以适配任意参数列表，不需要手写每个函数的参数。

2. **返回值：`auto ... -> decltype(name(args...))`**
C++11 尾置返回类型。
`decltype(name(args...))`：编译器直接拿原始 CUDA 驱动函数 `name` 调用表达式，推导原始函数的返回类型。
👉 **不需要手动写 cu 函数返回值类型（CUresult 等），完全复用原始函数签名。**

3. `using FuncType = decltype(&name);`
`decltype(&cuGetErrorName)` 得到**原始函数指针类型**。
比如 `cuGetErrorName` 原型：`CUresult cuGetErrorName(CUresult error, const char** str);`，`FuncType` 就推导成这个函数指针类型。

> ⚠️ 编译期这里**并不需要 libcuda.so 的头文件中该函数的定义！**
> 只需要函数**声明**，`decltype(&name)` 只是编译期类型推导，不生成调用代码。链接阶段也不需要找到这个符号，因为我们运行时 dlsym 拿地址。
> 但是头文件必须要有 `cuXXX` 的函数声明，否则 `&name` 会编译报错。

4. `static FuncType func = nullptr;`
wrapper 函数内部静态局部变量，**进程全局单例缓存函数指针**。
第一次进入 wrapper 的时候才调用 `dlsym` 查找符号；后续调用直接用缓存好的函数指针，不再走 dlsym。
C++ 标准保证静态局部变量初始化线程安全。

5. `dlsym(get_driver_handle(), #name)`
- `#name` 宏字符串化：name 是 `cuGetErrorName`，#name 变成字符串字面量 `"cuGetErrorName"`，对应 libcuda.so 中导出的符号名；
- `dlsym(handle, symbol_name)`：从已经 dlopen 打开的库里面，按符号字符串查找函数的内存地址；
- `reinterpret_cast<FuncType>`：把 `void*` 的符号地址强转成我们推导出来的原始函数指针类型。

> 注意：dlsym 返回 void*，C++ 不能隐式转函数指针，必须强制 reinterpret_cast。

6. `return func(std::forward<decltype(args)>(args)...);`
`std::forward` 完美转发，保留入参的值类别（左值/右值引用），参数原样转发给真正的 CUDA driver 函数指针。保证和直接调用 cuXXX 行为完全一致。

### 使用方式

代码中不再直接调用 `cuGetErrorName(...)`，而是调用 `lazy_cuGetErrorName(...)`。

---

## 3. 批量宏实例化

```cpp
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuGetErrorName);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuGetErrorString);
// ...大量cu* Driver API
```

每一行调用宏，就生成一个 `lazy_cuXxx()` 模板包装函数。

列表里全部都是 CUDA Driver API，很多是高级接口：

- 模块加载：`cuModuleLoad / cuModuleGetFunction` PTX/JIT 编译加载；
- VMM 虚拟内存管理：`cuMemAddressReserve / cuMemMap / cuMemCreate / cuMemRelease`（CUDA Virtual Memory Management，做异构内存池）；
- 跨进程共享内存句柄：`cuMemExportToShareableHandle`；
- 高级 Kernel launch：`cuLaunchKernelEx`；

> DeepEP 做 EP（Expert Parallel），需要大量使用 CUDA VMM、跨设备共享内存，这些能力只有 Driver API 提供，Runtime API 没有封装。

---

# 关键技术要点 & 设计目的

## 为什么要做这套 Lazy 动态加载？

1. **编译期不需要链接 `libcuda.so`**
如果 CMake 中不 `target_link_libraries(libcuda)`，普通调用 `cuXXX` 会报链接错误（undefined reference）。这套方案编译阶段只是做类型推导，不引用符号，**链接阶段不依赖 libcuda**。
2. **真正延迟加载库**
程序启动不会加载 libcuda.so，只有代码走到 `lazy_cuXXX` 时，才 dlopen 驱动库。
场景：同一个二进制，既可以跑 CPU 路径，也可以跑 GPU EP 路径；没有 GPU 的机器 CPU 逻辑依然可以启动，不会启动就加载失败。
3. **符号隔离 RTLD_LOCAL**
RTLD_LOCAL，libcuda 符号不进入全局符号表，避免不同版本 cuda toolkit、驱动之间符号冲突。
4. **一次查找，永久缓存**
每个 API 只会执行一次 dlsym，后续直接调用函数指针，运行时开销几乎为 0。

## 依赖的前提条件

1. 编译时需要 CUDA driver 头文件 `cuda.h`，要有 `cuXXX` 函数声明。否则 `decltype(&cuXxx)` 编译无法推导类型。**但是不需要链接 libcuda 库**。
2. 运行环境必须存在 `libcuda.so.1`，这个文件来自 NVIDIA 显卡驱动，不是 CUDA Toolkit。没有 NVIDIA 驱动，dlopen 直接断言失败。
3. POSIX 系统（Linux），依赖 `dlopen/dlsym/dlclose`，Windows 平台这套代码不可用（Windows 是 LoadLibrary/GetProcAddress）。

## 潜在坑点

1. **ABI 符号匹配**：`dlsym` 查找的符号名字必须和 libcuda.so 导出符号完全一致。例如部分接口有后缀 `_v2`，代码中写 `cuMemGetAddressRange_v2`，字符串化后查找 `cuMemGetAddressRange_v2`，不能写错名字，否则断言失败。
2. **头文件版本与运行驱动版本**：编译时 `cuda.h` 声明的函数签名，必须和运行时驱动库里面实际的函数 ABI 兼容。如果驱动版本过低，dlsym 找不到新 API 符号，直接断言报错。
3. `RTLD_LAZY` 只是 dlopen 内部符号懒解析，这里我们依然显式 dlsym 所有需要的符号。
4. 静态局部变量线程安全：C++11 及以上标准，函数内 static 局部变量初始化是线程安全；老编译器可能存在竞态，DeepEP 编译环境一般都是 C++17。

## 和 dlopen/dlsym 手写封装对比

普通手写动态加载 CUDA driver：每个 API 手动定义函数指针类型，写 wrapper，重复模板样板代码。

这套 C++ 模板宏方案，**只需要一行宏调用，自动推导全部函数指针类型、返回值、参数转发，消除大量重复样板代码**。

---

# 调用时序完整流程

```
调用 lazy_cuLaunchKernelEx(xxx,yyy)
  └─ static func == nullptr，第一次调用
      └─ get_driver_handle()
          └─ static handle == nullptr，第一次访问driver
              └─ dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL) → 获取库句柄
      └─ dlsym(handle, "cuLaunchKernelEx") 获取函数地址
      └─ 保存到静态func缓存
  └─ func(xxx, yyy) 通过函数指针直接调用CUDA驱动接口

第二次调用 lazy_cuLaunchKernelEx：直接拿已经缓存好的func指针，跳过dlopen、dlsym
```

# 补充：DeepEP 为什么需要 cu* Driver API

Expert 并行需要：

1. CUDA VMM 管理异构设备内存；
2. 设备间内存共享句柄导出导入；
3. 精细的 kernel 属性设置、cuLaunchKernelEx 灵活启动 kernel；
这些功能**CUDA Runtime API(`cudaXXX`) 没有暴露，只能调用底层 Driver API**。
但是又不希望编译强制链接 libcuda.so，于是实现这套 lazy 加载层。

如果你需要，我可以再对比：CUDA Runtime、Driver API 关系，或者给出一个极简可独立编译的最小 demo。
