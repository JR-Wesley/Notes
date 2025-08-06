---
dateCreated: 2025-08-06
dateModified: 2025-08-06
---
# 内存层次

CUDA 的内存层次结构从靠近计算单元（线程）到远离，可分为**寄存器、共享内存、本地内存、全局内存、常量内存、纹理 / 表面内存**等，不同层次的内存具有不同的访问范围、生命周期、性能和操作方式。以下是各层次的**寻址方式、申请（allocate）、访问与存储**方法的详细说明：

### 一、寄存器（Registers）

- **特点**：
    线程私有（每个线程独立拥有），访问速度最快（纳秒级），容量最小（每个线程约 255 个 32 位寄存器，具体取决于 GPU 架构）。

#### 1. 申请（Allocate）

无需显式申请，由编译器**自动分配**。当线程定义局部变量（且未被编译器判定为需要放入本地内存时），变量会被分配到寄存器。

```cpp
__global__ void kernel() {
  int a = 0;         // 自动分配到寄存器
  float b = 3.14f;   // 自动分配到寄存器
}
```

#### 2. 访问与存储

- **访问**：通过变量名直接访问（编译器优化寻址，无显式地址操作）。
- **存储**：直接对变量赋值（数据存储在寄存器中，线程执行期间有效）。

#### 3. 寻址方式

编译器管理的**线程私有地址空间**，每个线程的寄存器独立，无法被其他线程访问。

### 二、共享内存（Shared Memory）

- **特点**：
    线程块（Block）私有（块内所有线程共享），访问速度接近寄存器（比全局内存快 10-100 倍），容量有限（每个块通常最大为 48KB-128KB，取决于 GPU 架构和配置）。支持用户显式控制，是线程间通信和数据复用的核心。

#### 1. 申请（Allocate）

在核函数中通过 `__shared__` 关键字显式声明，分为**静态分配**和**动态分配**：

- **静态分配**：编译时确定大小

    ```cpp
    __global__ void kernel() {
      __shared__ float s_data[256];  // 静态分配256个float（1KB）
    }
    ```

- **动态分配**：运行时指定大小（需用 `extern` 关键字，启动核函数时传入共享内存大小）

    ```cpp
    __global__ void kernel() {
      extern __shared__ float s_data[];  // 动态分配，大小在核函数启动时指定
    }
    
    // 主机端启动核函数：第三个参数指定共享内存大小（字节）
    kernel<<<gridDim, blockDim, 1024>>();  // 动态分配1024字节共享内存
    ```

#### 2. 访问与存储

- **访问**：通过数组索引访问（如 `s_data[threadIdx.x]`），块内所有线程可读写。
- **存储**：直接对数组元素赋值（如 `s_data[i] = 1.0f`）。
- **注意**：线程间访问共享内存需同步（用 `__syncthreads()`），避免数据竞争（如一个线程写入后，其他线程再读取）。

#### 3. 寻址方式

**块内局部地址空间**：每个线程块有独立的共享内存，地址范围为块内声明的数组索引（如 0~255 for `s_data[256]`），其他块无法访问。

### 三、本地内存（Local Memory）

- **特点**：
    线程私有，语义上是线程的 “本地存储”，但物理上位于全局内存（因此访问速度慢，与全局内存相当）。当寄存器不足（如变量占用过多寄存器）或变量无法被编译器优化（如动态长度数组、大型数组）时，编译器会自动将变量分配到本地内存。

#### 1. 申请（Allocate）

无需显式申请，由编译器**自动分配**（当寄存器不足时）。例如：

```cpp
__global__ void kernel() {
  float big_array[1024];  // 数组过大，寄存器存不下，自动分配到本地内存
}
```

#### 2. 访问与存储

- **访问**：通过变量名或数组索引访问（与寄存器变量语法一致）。
- **存储**：直接赋值（如 `big_array[i] = 2.0f`），但实际操作的是全局内存中的线程私有区域。

#### 3. 寻址方式

**线程私有地址空间**（映射到全局内存），地址由编译器生成，仅当前线程可访问。

### 四、全局内存（Global Memory）

- **特点**：
    设备级内存（所有线程、所有块可访问），容量最大（GB 级），访问速度较慢（数百纳秒），是主机与设备间数据传输的主要区域。生命周期与设备一致（需显式释放）。

#### 1. 申请与释放（Allocate/Free）

在**主机端**通过 CUDA Runtime API 显式申请和释放：

```cpp
// 申请全局内存（设备端）
float* d_data;
cudaMalloc(&d_data, size_in_bytes);  // 成功返回cudaSuccess，失败返回错误码

// 释放全局内存
cudaFree(d_data);
```

#### 2. 访问与存储

- **主机端与设备端的数据传输**：通过 `cudaMemcpy` 实现（方向：主机→设备、设备→主机、设备→设备）

    ```cpp
    float* h_data = (float*)malloc(size_in_bytes);  // 主机内存
    // 主机→设备
    cudaMemcpy(d_data, h_data, size_in_bytes, cudaMemcpyHostToDevice);
    ```

- **设备端（核函数）访问**：通过指针直接访问（读写均可）

    ```cpp
    __global__ void kernel(float* d_data) {
      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      d_data[idx] = 1.0f;  // 写入全局内存
      float val = d_data[idx];  // 读取全局内存
    }
    ```

#### 3. 寻址方式

**全局地址空间**：所有线程可见，地址是设备级的全局指针（如 `d_data` 指向的地址），需注意内存对齐（如 64 字节对齐可提升访问效率）和合并访问（连续线程访问连续地址，最大化带宽）。

### 五、常量内存（Constant Memory）

- **特点**：
    设备级只读内存，容量小（通常 64KB），配有专用缓存（常量缓存），适合存储所有线程都需要访问的常量数据（如参数、系数）。读取时若多个线程访问同一地址，效率极高（广播机制）。

#### 1. 申请与初始化

- 在**全局作用域**（核函数外）用 `__constant__` 声明常量变量（设备端可见）。
- 在**主机端**通过 `cudaMemcpyToSymbol` 复制数据到常量内存（不可在核函数中修改）。

```cpp
// 声明常量内存变量（设备端）
__constant__ float c_params[256];  // 64KB（256*4字节）

// 主机端初始化
float h_params[256] = {1.0f, 2.0f, …};  // 主机数据
cudaMemcpyToSymbol(c_params, h_params, 256 * sizeof(float));  // 主机→常量内存
```

#### 2. 访问与存储

- **访问**：核函数中通过变量名直接访问（只读）

    ```cpp
    __global__ void kernel() {
      float val = c_params[0];  // 读取常量内存（只读）
    }
    ```

- **存储**：仅能在主机端通过 `cudaMemcpyToSymbol` 写入，核函数中不可修改。

#### 3. 寻址方式

**全局只读地址空间**：所有线程可见，地址由 `__constant__` 变量名隐式指定，通过索引访问（如 `c_params[i]`）。

### 六、纹理内存（Texture Memory）与表面内存（Surface Memory）

- **特点**：
    设备级只读（纹理）或读写（表面）内存，配有专用缓存，针对**2D/3D 空间局部性**优化（如图像、网格数据），支持地址越界处理（如 clamping）和插值（纹理内存）。容量取决于全局内存（纹理 / 表面只是全局内存的 “视图”）。

#### 1. 申请与绑定

- 先申请**全局内存**（作为纹理 / 表面的底层存储）。
- 创建**纹理引用**（Texture Reference）或**纹理对象**（Texture Object，推荐），绑定到底层全局内存。

```cpp
// 1. 申请全局内存（底层存储）
float* d_tex_data;
cudaMalloc(&d_tex_data, width * height * sizeof(float));

// 2. 创建纹理对象（现代方式，支持动态配置）
cudaResourceDesc resDesc;
memset(&resDesc, 0, sizeof(resDesc));
resDesc.resType = cudaResourceTypeLinear;  // 1D线性内存
resDesc.res.linear.devPtr = d_tex_data;    // 绑定全局内存
resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
resDesc.res.linear.desc.x = 32;  // 32位float

cudaTextureDesc texDesc;
memset(&texDesc, 0, sizeof(texDesc));
texDesc.addressMode[0] = cudaAddressModeClamp;  // 越界处理：clamp

cudaTextureObject_t texObj;
cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);  // 创建纹理对象
```

#### 2. 访问与存储

- **纹理内存（只读）**：核函数中通过 `tex1D`/`tex2D` 等函数访问（根据维度），参数为纹理对象和坐标。

    ```cpp
    __global__ void kernel(cudaTextureObject_t texObj, float* d_out, int width) {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      if (x < width) {
        // 读取纹理内存（1D，坐标为float类型）
        d_out[x] = tex1D<float>(texObj, (float)x);
      }
    }
    ```

- **表面内存（读写）**：类似纹理，但支持写入，通过 `surf1Dwrite`/`surf2Dread` 等函数操作。

#### 3. 寻址方式

**基于坐标的空间寻址**：通过纹理坐标（如 1D 的 `x`、2D 的 `(x,y)`）访问，而非直接指针。支持整数或浮点坐标，缓存会自动优化空间局部性访问。

### 总结：各内存层次的核心操作对比

|内存层次|申请方式|访问方式|存储方式|寻址特点|
|---|---|---|---|---|
|寄存器|编译器自动分配|变量名直接访问|直接赋值|线程私有，编译器管理地址|
|共享内存|核函数中 `__shared__` 声明（静态 / 动态）|数组索引访问|直接赋值，需 `__syncthreads()` 同步|线程块私有，块内局部索引|
|本地内存|编译器自动分配（寄存器不足时）|变量名 / 数组索引访问|直接赋值|线程私有，映射到全局内存|
|全局内存|主机端 `cudaMalloc`|指针访问（核函数）+ `cudaMemcpy`（主机）|核函数中直接赋值，主机端 `cudaMemcpy`|设备级全局地址，指针寻址|
|常量内存|全局 `__constant__` 声明 + `cudaMemcpyToSymbol`|变量名 + 索引访问（只读）|主机端 `cudaMemcpyToSymbol`|全局只读地址，索引寻址|
|纹理内存|全局内存 + 绑定纹理对象|`tex1D`/`tex2D` 等函数（坐标）|主机端 `cudaMemcpy` 到底层全局内存|空间坐标寻址，缓存优化|

掌握各内存层次的操作方式是 CUDA 性能优化的核心：寄存器和共享内存用于高频访问数据，全局内存用于大规模数据存储，常量 / 纹理内存用于优化特定访问模式（如只读、空间局部性）。
