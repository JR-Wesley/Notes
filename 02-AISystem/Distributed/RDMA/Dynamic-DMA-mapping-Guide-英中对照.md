# Dynamic DMA mapping Guide — 英中对照笔记

Source: https://docs.kernel.org/6.2/core-api/dma-api-howto.html

依据用户提供的英文粘贴文本整理；恢复章节标题、图和压缩的代码换行。英文段落与中文译文相邻；少量解读专门标注。粘贴内容在 Platform Issues 后结束，未包含原文 Closing。此文是 Linux 内核驱动 DMA API 指南，不能直接替代 rdma-core 的 userspace verbs 文档。

**Authors / 作者：** David S. Miller <davem@redhat.com>；Richard Henderson <rth@cygnus.com>；Jakub Jelinek <jakub@redhat.com>。

> This is a guide to device driver writers on how to use the DMA API with example pseudo-code. For a concise description of the API, see DMA-API.txt.

本文面向设备驱动开发者，用伪代码示例说明如何使用 DMA API。简明的 API 说明见 `DMA-API.txt`。

## CPU and DMA addresses / CPU 与 DMA 地址

> There are several kinds of addresses involved in the DMA API, and it’s important to understand the differences.

DMA API 涉及几类不同的地址，理解它们之间的区别十分重要。

> The kernel normally uses virtual addresses. Any address returned by [kmalloc()](https://docs.kernel.org/6.2/core-api/mm-api.html#c.kmalloc), [vmalloc()](https://docs.kernel.org/6.2/core-api/mm-api.html#c.vmalloc), and similar interfaces is a virtual address and can be stored in a void *.

内核通常使用虚拟地址。`kmalloc()`、`vmalloc()` 等接口返回的都是虚拟地址，可以存放在 `void *` 中。

> The virtual memory system (TLB, page tables, etc.) translates virtual addresses to CPU physical addresses, which are stored as “phys_addr_t” or “resource_size_t”. The kernel manages device resources like registers as physical addresses. These are the addresses in /proc/iomem. The physical address is not directly useful to a driver; it must use [ioremap()](https://docs.kernel.org/6.2/driver-api/device-io.html#c.ioremap) to map the space and produce a virtual address.

虚拟内存系统（TLB、页表等）将虚拟地址转换为 CPU 物理地址；后者以 `phys_addr_t` 或 `resource_size_t` 表示。内核把寄存器等设备资源作为物理地址管理，这些地址会出现在 `/proc/iomem`。驱动不能直接使用物理地址访问该空间，而须通过 `ioremap()` 将其映射为虚拟地址。

> I/O devices use a third kind of address: a “bus address”. If a device has registers at an MMIO address, or if it performs DMA to read or write system memory, the addresses used by the device are bus addresses. In some systems, bus addresses are identical to CPU physical addresses, but in general they are not. IOMMUs and host bridges can produce arbitrary mappings between physical and bus addresses.

I/O 设备还使用第三类地址：总线地址。无论是设备寄存器的 MMIO 地址，还是设备通过 DMA 读写系统内存时使用的地址，从设备角度看都是总线地址。在某些系统上它等于 CPU 物理地址，但一般不能如此假定；IOMMU 和 host bridge 可以在物理地址与总线地址之间建立任意映射。

> From a device’s point of view, DMA uses the bus address space, but it may be restricted to a subset of that space. For example, even if a system supports 64-bit addresses for main memory and PCI BARs, it may use an IOMMU so devices only need to use 32-bit DMA addresses.

从设备角度看，DMA 使用总线地址空间，但设备可能只能访问其中一部分。例如，即使主内存和 PCI BAR 使用 64 位地址，系统仍可能通过 IOMMU 让设备只需使用 32 位 DMA 地址。

> Here’s a picture and some examples:

下图和示例说明这些关系：

```text
             CPU                  CPU                  Bus
           Virtual              Physical             Address
           Address              Address               Space
            Space                Space
           +-------+             +------+             +------+
           |       |             |MMIO  |   Offset    |      |
           |       |  Virtual    |Space |   applied   |      |
         C +-------+ --------> B +------+ ----------> +------+ A
           |       |  mapping    |      |   by host   |      |
 +-----+   |       |             |      |   bridge    |      |   +--------+
 |     |   |       |             +------+             |      |   |        |
 | CPU |   |       |             | RAM  |             |      |   | Device |
 |     |   |       |             |      |             |      |   |        |
 +-----+   +-------+             +------+             +------+   +--------+
           |       |  Virtual    |Buffer|   Mapping   |      |
         X +-------+ --------> Y +------+ <---------- +------+ Z
           |       |  mapping    | RAM  |   by IOMMU
           |       |             |      |
           |       |             |      |
           +-------+             +------+
```

> During the enumeration process, the kernel learns about I/O devices and their MMIO space and the host bridges that connect them to the system. For example, if a PCI device has a BAR, the kernel reads the bus address (A) from the BAR and converts it to a CPU physical address (B). The address B is stored in a struct resource and usually exposed via /proc/iomem. When a driver claims a device, it typically uses [ioremap()](https://docs.kernel.org/6.2/driver-api/device-io.html#c.ioremap) to map physical address B at a virtual address (C). It can then use, e.g., ioread32(C), to access the device registers at bus address A.

枚举设备时，内核获知 I/O 设备的 MMIO 空间及连接它们的 host bridge。例如，内核读取 PCI 设备 BAR 中的总线地址 A，并转换为 CPU 物理地址 B；B 存放在 `struct resource` 中，通常可在 `/proc/iomem` 看到。驱动接管设备后，通常调用 `ioremap()` 将 B 映射为虚拟地址 C，再通过 `ioread32(C)` 等操作访问总线地址 A 上的寄存器。

> If the device supports DMA, the driver sets up a buffer using [kmalloc()](https://docs.kernel.org/6.2/core-api/mm-api.html#c.kmalloc) or a similar interface, which returns a virtual address (X). The virtual memory system maps X to a physical address (Y) in system RAM. The driver can use virtual address X to access the buffer, but the device itself cannot because DMA doesn’t go through the CPU virtual memory system.

若设备支持 DMA，驱动通过 `kmalloc()` 等接口准备缓冲区，得到虚拟地址 X。虚拟内存系统把 X 映射到系统 RAM 中的物理地址 Y。驱动可以用 X 访问缓冲区；设备却不能直接使用 X，因为 DMA 不经过 CPU 的虚拟内存系统。

> In some simple systems, the device can do DMA directly to physical address Y. But in many others, there is IOMMU hardware that translates DMA addresses to physical addresses, e.g., it translates Z to Y. This is part of the reason for the DMA API: the driver can give a virtual address X to an interface like dma_map_single(), which sets up any required IOMMU mapping and returns the DMA address Z. The driver then tells the device to do DMA to Z, and the IOMMU maps it to the buffer at address Y in system RAM.

在简单系统中，设备可能直接向物理地址 Y 做 DMA；许多系统则由 IOMMU 把 DMA 地址 Z 转换为 Y。这正是 DMA API 存在的原因之一：驱动将虚拟地址 X 传给 `dma_map_single()` 等接口，由它建立必要的 IOMMU 映射并返回 DMA 地址 Z。驱动把 Z 告诉设备，设备向 Z 发起 DMA，IOMMU 再将其映射到 RAM 中的 Y。

**解读：** 这里的 X、Y、Z 分别属于 CPU 虚拟地址、CPU 物理地址和设备 DMA 地址空间；它们可指向同一缓冲区，却不能当作同一个数值。RNIC 的内存访问也需要沿设备侧地址路径理解。

> So that Linux can use the dynamic DMA mapping, it needs some help from the drivers, namely it has to take into account that DMA addresses should be mapped only for the time they are actually used and unmapped after the DMA transfer.

为了让 Linux 使用动态 DMA 映射，驱动应只在实际使用 DMA 地址期间保持映射，并在 DMA 传输结束后解除映射。

> The following API will work of course even on platforms where no such hardware exists.

即使平台没有这类硬件，下面介绍的 API 也能工作。

> Note that the DMA API works with any bus independent of the underlying microprocessor architecture. You should use the DMA API rather than the bus-specific DMA API, i.e., use the dma_map_*() interfaces rather than the pci_map_*() interfaces.

DMA API 与底层微处理器架构无关，适用于各种总线。应使用通用 DMA API，而非特定总线的旧接口；例如使用 `dma_map_*()` 而非 `pci_map_*()`。

> First of all, you should make sure:

首先，应确保驱动包含：

```c
#include <linux/dma-mapping.h>
```

> is in your driver, which provides the definition of dma_addr_t. This type can hold any valid DMA address for the platform and should be used everywhere you hold a DMA address returned from the DMA mapping functions.

该头文件定义 `dma_addr_t`。此类型能容纳平台上的任何有效 DMA 地址，凡保存 DMA 映射函数返回地址的位置都应使用它。

## What memory is DMA’able? / 哪些内存可用于 DMA？

> The first piece of information you must know is what kernel memory can be used with the DMA mapping facilities. There has been an unwritten set of rules regarding this, and this text is an attempt to finally write them down.

首先要知道哪些内核内存可用于 DMA 映射。过去这方面存在一组未成文的规则，本文尝试将其写明。

> If you acquired your memory via the page allocator (i.e. __get_free_page*()) or the generic memory allocators (i.e. [kmalloc()](https://docs.kernel.org/6.2/core-api/mm-api.html#c.kmalloc) or [kmem_cache_alloc()](https://docs.kernel.org/6.2/core-api/mm-api.html#c.kmem_cache_alloc)) then you may DMA to/from that memory using the addresses returned from those routines.

通过页分配器（如 `__get_free_page*()`）或通用内存分配器（如 `kmalloc()`、`kmem_cache_alloc()`）获得的内存，可以使用这些接口返回的地址进行 DMA 读写。

> This means specifically that you may _not_ use the memory/addresses returned from [vmalloc()](https://docs.kernel.org/6.2/core-api/mm-api.html#c.vmalloc) for DMA. It is possible to DMA to the _underlying_ memory mapped into a [vmalloc()](https://docs.kernel.org/6.2/core-api/mm-api.html#c.vmalloc) area, but this requires walking page tables to get the physical addresses, and then translating each of those pages back to a kernel address using something like __va(). [ EDIT: Update this when we integrate Gerd Knorr’s generic code which does this. ]

这意味着不能直接把 `vmalloc()` 返回的内存或地址用于 DMA。对映射在 `vmalloc()` 区域下的底层内存做 DMA 是可能的，但须遍历页表取得物理地址，再用 `__va()` 一类方法把各页转换回内核地址。方括号中的编辑备注是原文保留的待更新说明。

**解读：** 这一节限定的是普通内核 DMA 映射接口的输入，不能直接推出 userspace MR、ODP 或 GPU memory 的注册规则。

> This rule also means that you may use neither kernel image addresses (items in data/text/bss segments), nor module image addresses, nor stack addresses for DMA. These could all be mapped somewhere entirely different than the rest of physical memory. Even if those classes of memory could physically work with DMA, you’d need to ensure the I/O buffers were cacheline-aligned. Without that, you’d see cacheline sharing problems (data corruption) on CPUs with DMA-incoherent caches. (The CPU could write to one word, DMA would write to a different one in the same cache line, and one of them could be overwritten.)

同理，不能把内核映像的 data/text/bss 段地址、模块映像地址或栈地址用于 DMA。它们可能映射在与其他物理内存完全不同的位置。即使物理上可行，也要保证 I/O 缓冲区按缓存行对齐，否则在 DMA 不一致的 CPU 缓存上，CPU 和 DMA 对同一缓存行中不同字的写入可能相互覆盖，造成数据损坏。

> Also, this means that you cannot take the return of a [kmap()](https://docs.kernel.org/6.2/mm/highmem.html#c.kmap) call and DMA to/from that. This is similar to [vmalloc()](https://docs.kernel.org/6.2/core-api/mm-api.html#c.vmalloc).

也不能拿 `kmap()` 的返回值直接做 DMA；原因与 `vmalloc()` 类似。

> What about block I/O and networking buffers? The block I/O and networking subsystems make sure that the buffers they use are valid for you to DMA from/to.

块 I/O 与网络缓冲区怎么办？这两个子系统会确保其缓冲区对驱动的 DMA 访问有效。

## DMA addressing capabilities / DMA 寻址能力

> By default, the kernel assumes that your device can address 32-bits of DMA addressing. For a 64-bit capable device, this needs to be increased, and for a device with limitations, it needs to be decreased.

内核默认假设设备具有 32 位 DMA 寻址能力。对于支持 64 位地址的设备应调高该范围，对于受限设备则应调低。

> Special note about PCI: PCI-X specification requires PCI-X devices to support 64-bit addressing (DAC) for all transactions. And at least one platform (SGI SN2) requires 64-bit consistent allocations to operate correctly when the IO bus is in PCI-X mode.

关于 PCI 的特别说明：PCI-X 规范要求设备对所有事务支持 64 位寻址（DAC）。至少有一个平台（SGI SN2）在 I/O 总线处于 PCI-X 模式时，要求使用 64 位 consistent 分配才能正常工作。

> For correct operation, you must set the DMA mask to inform the kernel about your devices DMA addressing capabilities.

为保证正确运行，必须设置 DMA mask，告知内核设备的 DMA 寻址能力。

> This is performed via a call to dma_set_mask_and_coherent():

可调用 `dma_set_mask_and_coherent()` 设置：

```c
int dma_set_mask_and_coherent(struct device *dev, u64 mask);
```

> which will set the mask for both streaming and coherent APIs together. If you have some special requirements, then the following two separate calls can be used instead:

该函数同时设置 streaming 与 coherent API 的 mask。若有特殊要求，可分别调用下列两个函数。

> The setup for streaming mappings is performed via a call to dma_set_mask():

streaming 映射通过 `dma_set_mask()` 设置：

```c
int dma_set_mask(struct device *dev, u64 mask);
```

> The setup for consistent allocations is performed via a call to dma_set_coherent_mask():

consistent 分配通过 `dma_set_coherent_mask()` 设置：

```c
int dma_set_coherent_mask(struct device *dev, u64 mask);
```

> Here, dev is a pointer to the device struct of your device, and mask is a bit mask describing which bits of an address your device supports. Often the device struct of your device is embedded in the bus-specific device struct of your device. For example, &pdev->dev is a pointer to the device struct of a PCI device (pdev is a pointer to the PCI device struct of your device).

这里的 `dev` 指向设备的 `struct device`，`mask` 是说明设备支持哪些地址位的位掩码。该 `struct device` 通常嵌在特定总线的设备结构中；例如，对 PCI 设备来说，`&pdev->dev` 指向它的 `struct device`。

> These calls usually return zero to indicated your device can perform DMA properly on the machine given the address mask you provided, but they might return an error if the mask is too small to be supportable on the given system. If it returns non-zero, your device cannot perform DMA properly on this platform, and attempting to do so will result in undefined behavior. You must not use DMA on this device unless the dma_set_mask family of functions has returned success.

这些调用通常返回零，表示给定 mask 下设备可以在当前机器上正常 DMA；如果 mask 太小、平台无法支持，也可能返回错误。返回非零时，设备不能在该平台正常 DMA，强行使用会产生未定义行为。只有 `dma_set_mask` 系列函数成功后，才能在此设备上使用 DMA。

> This means that in the failure case, you have two options:

失败时有两种选择：

1. Use some non-DMA mode for data transfer, if possible.
   如果可行，改用非 DMA 数据传输模式。

2. Ignore this device and do not initialize it.
   忽略此设备，不初始化它。

> It is recommended that your driver print a kernel KERN_WARNING message when setting the DMA mask fails. In this manner, if a user of your driver reports that performance is bad or that the device is not even detected, you can ask them for the kernel messages to find out exactly why.

设置 DMA mask 失败时，建议驱动输出一条 `KERN_WARNING` 级别的内核消息。这样当用户报告性能差或设备未被识别时，可以根据内核日志查明原因。

> The standard 64-bit addressing device would do something like this:

标准的 64 位寻址设备可以这样处理：

```c
if (dma_set_mask_and_coherent(dev, DMA_BIT_MASK(64))) {
    dev_warn(dev, "mydev: No suitable DMA available\n");
    goto ignore_this_device;
}
```

> If the device only supports 32-bit addressing for descriptors in the coherent allocations, but supports full 64-bits for streaming mappings it would look like this:

若设备的 coherent descriptor 只能使用 32 位地址，但 streaming 映射支持完整的 64 位地址，可以采用下面的形式：

```c
if (dma_set_mask(dev, DMA_BIT_MASK(64))) {
    dev_warn(dev, "mydev: No suitable DMA available\n");
    goto ignore_this_device;
}
```

> The coherent mask will always be able to set the same or a smaller mask as the streaming mask. However for the rare case that a device driver only uses consistent allocations, one would have to check the return value from dma_set_coherent_mask().

coherent mask 总能设置为与 streaming mask 相同或更小。少数驱动只使用 consistent 分配时，则须检查 `dma_set_coherent_mask()` 的返回值。

> Finally, if your device can only drive the low 24-bits of address you might do something like:

如果设备只能驱动地址的低 24 位，可使用如下形式：

```c
if (dma_set_mask(dev, DMA_BIT_MASK(24))) {
    dev_warn(dev, "mydev: 24-bit DMA addressing not available\n");
    goto ignore_this_device;
}
```

> When dma_set_mask() or dma_set_mask_and_coherent() is successful, and returns zero, the kernel saves away this mask you have provided. The kernel will use this information later when you make DMA mappings.

`dma_set_mask()` 或 `dma_set_mask_and_coherent()` 成功返回零后，内核会保存所提供的 mask，并在后续创建 DMA 映射时使用。

> There is a case which we are aware of at this time, which is worth mentioning in this documentation. If your device supports multiple functions (for example a sound card provides playback and record functions) and the various different functions have _different_ DMA addressing limitations, you may wish to probe each mask and only provide the functionality which the machine can handle. It is important that the last call to dma_set_mask() be for the most specific mask.

还有一种情况值得说明：如果设备具备多个功能（如声卡的播放和录音），而各功能的 DMA 寻址限制不同，可以分别探测 mask，仅启用当前机器支持的功能。最后一次 `dma_set_mask()` 调用必须使用最具体的 mask。

> Here is pseudo-code showing how this might be done:

伪代码如下：

```c
#define PLAYBACK_ADDRESS_BITS DMA_BIT_MASK(32)
#define RECORD_ADDRESS_BITS DMA_BIT_MASK(24)
struct my_sound_card *card;
struct device *dev;
...
if (!dma_set_mask(dev, PLAYBACK_ADDRESS_BITS)) {
    card->playback_enabled = 1;
}
else {
    card->playback_enabled = 0;
    dev_warn(dev, "%s: Playback disabled due to DMA limitations\n", card->name);
}
if (!dma_set_mask(dev, RECORD_ADDRESS_BITS)) {
    card->record_enabled = 1;
}
else {
    card->record_enabled = 0;
    dev_warn(dev, "%s: Record disabled due to DMA limitations\n", card->name);
}
```

> A sound card was used as an example here because this genre of PCI devices seems to be littered with ISA chips given a PCI front end, and thus retaining the 16MB DMA addressing limitations of ISA.

这里使用声卡举例，是因为此类 PCI 设备中不少实际上把 ISA 芯片接在 PCI 前端，因而保留了 ISA 的 16 MB DMA 寻址限制。

## Types of DMA mappings / DMA 映射类型

> There are two types of DMA mappings:

DMA 映射有两种类型：

- Consistent DMA mappings which are usually mapped at driver initialization, unmapped at the end and for which the hardware should guarantee that the device and the CPU can access the data in parallel and will see updates made by each other without any explicit software flushing.
  consistent DMA 映射通常在驱动初始化时建立、结束时解除；硬件应保证设备与 CPU 可以并行访问数据，双方无需显式软件刷新就能看到对方的更新。

> Think of “consistent” as “synchronous” or “coherent”.

可将 consistent 理解为 synchronous 或 coherent。

> The current default is to return consistent memory in the low 32 bits of the DMA space. However, for future compatibility you should set the consistent mask even if this default is fine for your driver.

原文所述的默认行为是在 DMA 空间的低 32 位返回 consistent 内存。不过，即使这一默认行为已满足驱动需求，为兼容未来情况仍应显式设置 consistent mask。

> Good examples of what to use consistent mappings for are:

适合使用 consistent 映射的例子包括：

**整理注：** 粘贴文本在“例子包括”之后没有列出例项；此处按所提供文本保留空缺，不补写未提供的英文。

> The invariant these examples all require is that any CPU store to memory is immediately visible to the device, and vice versa. Consistent mappings guarantee this.

这些例子的共同要求是：CPU 对内存的写入立即对设备可见，反之亦然。consistent 映射提供这一保证。

**Important / 重要提示**

> Consistent DMA memory does not preclude the usage of proper memory barriers. The CPU may reorder stores to consistent memory just as it may normal memory. Example: if it is important for the device to see the first word of a descriptor updated before the second, you must do something like:

consistent DMA 内存并不免除正确使用内存屏障的必要。与普通内存一样，CPU 仍可能重排对 consistent 内存的写入。例如，若必须让设备先看到 descriptor 的第一个字，再看到第二个字，就需按下例设置屏障：

**解读：** coherent 处理的是 CPU 与设备之间的数据可见性，`wmb()` 处理发布多个字段时的顺序；两者回答不同问题。

```c
desc->word0 = address;
wmb();
desc->word1 = DESC_VALID;
```

> in order to get correct behavior on all platforms.

这样才能在所有平台上获得正确行为。

> Also, on some platforms your driver may need to flush CPU write buffers in much the same way as it needs to flush write buffers found in PCI bridges (such as by reading a register’s value after writing it).

某些平台上，驱动还可能需要刷新 CPU 写缓冲区，方式与刷新 PCI bridge 写缓冲区类似，例如写寄存器后再读其值。

- Streaming DMA mappings which are usually mapped for one DMA transfer, unmapped right after it (unless you use dma_sync_* below) and for which hardware can optimize for sequential accesses.
  streaming DMA 映射通常只为一次 DMA 传输建立，完成后立即解除，除非使用下文的 `dma_sync_*`；硬件可针对顺序访问优化它。

> Think of “streaming” as “asynchronous” or “outside the coherency domain”.

可将 streaming 理解为 asynchronous，或处于一致性域之外。

> Good examples of what to use streaming mappings for are:

适合使用 streaming 映射的例子包括：

**整理注：** 粘贴文本在“例子包括”之后没有列出例项；此处按所提供文本保留空缺。

> The interfaces for using this type of mapping were designed in such a way that an implementation can make whatever performance optimizations the hardware allows. To this end, when using such mappings you must be explicit about what you want to happen.

这类接口允许实现利用硬件提供的性能优化。因此，使用 streaming 映射时，调用方必须明确表达希望发生的操作。

> Neither type of DMA mapping has alignment restrictions that come from the underlying bus, although some devices may have such restrictions. Also, systems with caches that aren’t DMA-coherent will work better when the underlying buffers don’t share cache lines with other data.

两类 DMA 映射都没有来自底层总线的对齐限制，不过具体设备可能有限制。对于缓存不具备 DMA 一致性的系统，若底层缓冲区不与其他数据共享缓存行，效果会更好。

## Using Consistent DMA mappings / 使用 consistent DMA 映射

> To allocate and map large (PAGE_SIZE or so) consistent DMA regions, you should do:

要分配并映射较大的（约 `PAGE_SIZE`）consistent DMA 区域，可这样做：

```c
dma_addr_t dma_handle;
cpu_addr = dma_alloc_coherent(dev, size, &dma_handle, gfp);
```

> where device is a struct device *. This may be called in interrupt context with the GFP_ATOMIC flag.

其中设备参数是 `struct device *`；传入 `GFP_ATOMIC` 时，可在中断上下文调用。

> Size is the length of the region you want to allocate, in bytes.

`size` 是所需区域的字节数。

> This routine will allocate RAM for that region, so it acts similarly to __get_free_pages() (but takes size instead of a page order). If your driver needs regions sized smaller than a page, you may prefer using the dma_pool interface, described below.

此例程会为区域分配 RAM，作用类似 `__get_free_pages()`，但接收的是大小而非页阶数。若驱动需要小于一页的区域，可考虑下文的 `dma_pool` 接口。

> The consistent DMA mapping interfaces, will by default return a DMA address which is 32-bit addressable. Even if the device indicates (via the DMA mask) that it may address the upper 32-bits, consistent allocation will only return > 32-bit addresses for DMA if the consistent DMA mask has been explicitly changed via dma_set_coherent_mask(). This is true of the dma_pool interface as well.

默认情况下，consistent DMA 映射接口返回可由 32 位地址访问的 DMA 地址。即使设备的 DMA mask 表明它可访问高 32 位，只有经 `dma_set_coherent_mask()` 显式改变 coherent DMA mask，consistent 分配才会返回大于 32 位的 DMA 地址。`dma_pool` 同样如此。

> dma_alloc_coherent() returns two values: the virtual address which you can use to access it from the CPU and dma_handle which you pass to the card.

`dma_alloc_coherent()` 返回两个值：CPU 访问所用的虚拟地址，以及传给设备卡的 `dma_handle`。

> The CPU virtual address and the DMA address are both guaranteed to be aligned to the smallest PAGE_SIZE order which is greater than or equal to the requested size. This invariant exists (for example) to guarantee that if you allocate a chunk which is smaller than or equal to 64 kilobytes, the extent of the buffer you receive will not cross a 64K boundary.

CPU 虚拟地址与 DMA 地址都保证按不小于请求大小的最小 `PAGE_SIZE` 阶对齐。例如，申请不超过 64 KiB 的块时，所得缓冲区不会跨越 64 KiB 边界。

> To unmap and free such a DMA region, you call:

释放该 DMA 区域时调用：

```c
dma_free_coherent(dev, size, cpu_addr, dma_handle);
```

> where dev, size are the same as in the above call and cpu_addr and dma_handle are the values dma_alloc_coherent() returned to you. This function may not be called in interrupt context.

其中 `dev`、`size` 与分配时相同，`cpu_addr` 和 `dma_handle` 是 `dma_alloc_coherent()` 的返回值。该函数不能在中断上下文调用。

> If your driver needs lots of smaller memory regions, you can write custom code to subdivide pages returned by dma_alloc_coherent(), or you can use the dma_pool API to do that. A dma_pool is like a kmem_cache, but it uses dma_alloc_coherent(), not __get_free_pages(). Also, it understands common hardware constraints for alignment, like queue heads needing to be aligned on N byte boundaries.

若驱动需要许多较小内存区域，可以自行拆分 `dma_alloc_coherent()` 返回的页，也可以使用 `dma_pool` API。`dma_pool` 类似 `kmem_cache`，但底层使用 `dma_alloc_coherent()` 而不是 `__get_free_pages()`，还可处理队列头需按 N 字节边界对齐等常见硬件约束。

> Create a dma_pool like this:

创建 `dma_pool` 的形式如下：

```c
struct dma_pool *pool;
pool = dma_pool_create(name, dev, size, align, boundary);
```

> The “name” is for diagnostics (like a kmem_cache name); dev and size are as above. The device’s hardware alignment requirement for this type of data is “align” (which is expressed in bytes, and must be a power of two). If your device has no boundary crossing restrictions, pass 0 for boundary; passing 4096 says memory allocated from this pool must not cross 4KByte boundaries (but at that time it may be better to use dma_alloc_coherent() directly instead).

`name` 用于诊断，类似 `kmem_cache` 的名称；`dev` 和 `size` 含义如上。`align` 表示设备要求的字节对齐，必须是 2 的幂。若设备没有跨边界限制，`boundary` 传零；传 4096 则表示池中分配的内存不可跨越 4 KiB 边界，此时也可以考虑直接使用 `dma_alloc_coherent()`。

> Allocate memory from a DMA pool like this:

从 DMA pool 分配内存的形式如下：

```c
cpu_addr = dma_pool_alloc(pool, flags, &dma_handle);
```

> flags are GFP_KERNEL if blocking is permitted (not in_interrupt nor holding SMP locks), GFP_ATOMIC otherwise. Like dma_alloc_coherent(), this returns two values, cpu_addr and dma_handle.

允许阻塞时（既不在中断中，也不持有 SMP 锁）`flags` 用 `GFP_KERNEL`，否则用 `GFP_ATOMIC`。与 `dma_alloc_coherent()` 一样，它返回 `cpu_addr` 和 `dma_handle` 两个值。

> Free memory that was allocated from a dma_pool like this:

释放 DMA pool 分配的内存：

```c
dma_pool_free(pool, cpu_addr, dma_handle);
```

> where pool is what you passed to [dma_pool_alloc()](https://docs.kernel.org/6.2/core-api/mm-api.html#c.dma_pool_alloc), and cpu_addr and dma_handle are the values [dma_pool_alloc()](https://docs.kernel.org/6.2/core-api/mm-api.html#c.dma_pool_alloc) returned. This function may be called in interrupt context.

`pool` 是传给 `dma_pool_alloc()` 的池；`cpu_addr` 和 `dma_handle` 是其返回值。该函数可在中断上下文调用。

> Destroy a dma_pool by calling:

销毁 DMA pool 的调用如下：

```c
dma_pool_destroy(pool);
```

> Make sure you’ve called [dma_pool_free()](https://docs.kernel.org/6.2/core-api/mm-api.html#c.dma_pool_free) for all memory allocated from a pool before you destroy the pool. This function may not be called in interrupt context.

销毁池之前，要确保已对其中分配的全部内存调用 `dma_pool_free()`。`dma_pool_destroy()` 不能在中断上下文调用。

## DMA Direction / DMA 方向

> The interfaces described in subsequent portions of this document take a DMA direction argument, which is an integer and takes on one of the following values:

文档后续接口接收一个整数 DMA direction 参数，其值可为以下四种：

```text
DMA_BIDIRECTIONAL
DMA_TO_DEVICE
DMA_FROM_DEVICE
DMA_NONE
```

> You should provide the exact DMA direction if you know it.

知道准确 DMA 方向时就应明确指定。

> DMA_TO_DEVICE means “from main memory to the device” DMA_FROM_DEVICE means “from the device to main memory” It is the direction in which the data moves during the DMA transfer.

`DMA_TO_DEVICE` 表示“从主内存到设备”，`DMA_FROM_DEVICE` 表示“从设备到主内存”；方向描述 DMA 传输期间数据的移动方向。

> You are _strongly_ encouraged to specify this as precisely as you possibly can.

强烈建议尽可能准确地指定方向。

> If you absolutely cannot know the direction of the DMA transfer, specify DMA_BIDIRECTIONAL. It means that the DMA can go in either direction. The platform guarantees that you may legally specify this, and that it will work, but this may be at the cost of performance for example.

确实无法预先知道 DMA 传输方向时，指定 `DMA_BIDIRECTIONAL`，表示 DMA 可双向进行。平台保证该值合法且可工作，但可能付出性能代价。

> The value DMA_NONE is to be used for debugging. One can hold this in a data structure before you come to know the precise direction, and this will help catch cases where your direction tracking logic has failed to set things up properly.

`DMA_NONE` 用于调试。可先将其保存在数据结构中，待知道准确方向再更新，从而发现方向跟踪逻辑未正确设置的情况。

> Another advantage of specifying this value precisely (outside of potential platform-specific optimizations of such) is for debugging. Some platforms actually have a write permission boolean which DMA mappings can be marked with, much like page protections in the user program address space. Such platforms can and do report errors in the kernel logs when the DMA controller hardware detects violation of the permission setting.

准确指定方向还有利于调试。某些平台可为 DMA 映射标记写权限，类似用户地址空间中的页保护；DMA 控制器检测到违反该权限的访问时，会在内核日志中报告错误。

> Only streaming mappings specify a direction, consistent mappings implicitly have a direction attribute setting of DMA_BIDIRECTIONAL.

只有 streaming 映射显式指定方向；consistent 映射隐含 `DMA_BIDIRECTIONAL` 方向属性。

> The SCSI subsystem tells you the direction to use in the ‘sc_data_direction’ member of the SCSI command your driver is working on.

SCSI 子系统在驱动处理的 SCSI 命令的 `sc_data_direction` 成员中给出所用方向。

> For Networking drivers, it’s a rather simple affair. For transmit packets, map/unmap them with the DMA_TO_DEVICE direction specifier. For receive packets, just the opposite, map/unmap them with the DMA_FROM_DEVICE direction specifier.

网络驱动较简单：发送包用 `DMA_TO_DEVICE` 映射和解除映射，接收包则用 `DMA_FROM_DEVICE`。

## Using Streaming DMA mappings / 使用 streaming DMA 映射

> The streaming DMA mapping routines can be called from interrupt context. There are two versions of each map/unmap, one which will map/unmap a single memory region, and one which will map/unmap a scatterlist.

streaming DMA 映射例程可在中断上下文调用。每组 map/unmap 均有两个版本：处理单个内存区域，或处理 scatterlist。

> To map a single region, you do:

映射单一区域的做法如下：

```c
struct device *dev = &my_dev->dev;
dma_addr_t dma_handle;
void *addr = buffer->ptr;
size_t size = buffer->len;
dma_handle = dma_map_single(dev, addr, size, direction);
if (dma_mapping_error(dev, dma_handle)) {
    /*
     * reduce current DMA mapping usage,
     * delay and try again later or
     * reset driver.
     */
    goto map_error_handling;
}
```

> and to unmap it:

解除映射的调用如下：

```c
dma_unmap_single(dev, dma_handle, size, direction);
```

> You should call dma_mapping_error() as dma_map_single() could fail and return error. Doing so will ensure that the mapping code will work correctly on all DMA implementations without any dependency on the specifics of the underlying implementation. Using the returned address without checking for errors could result in failures ranging from panics to silent data corruption. The same applies to dma_map_page() as well.

`dma_map_single()` 可能失败并返回错误，因此应调用 `dma_mapping_error()` 检查。这样代码才能适用于各种 DMA 实现，不依赖底层细节。未经检查就使用返回地址，可能导致内核崩溃，也可能造成难以发现的数据损坏。`dma_map_page()` 同理。

> You should call dma_unmap_single() when the DMA activity is finished, e.g., from the interrupt which told you that the DMA transfer is done.

DMA 活动完成后应调用 `dma_unmap_single()`，例如在收到 DMA 完成中断时。

> Using CPU pointers like this for single mappings has a disadvantage: you cannot reference HIGHMEM memory in this way. Thus, there is a map/unmap interface pair akin to dma_{map,unmap}_single(). These interfaces deal with page/offset pairs instead of CPU pointers. Specifically:

这种以 CPU 指针做单区域映射的方式无法引用 HIGHMEM 内存。因此还有一组类似 `dma_{map,unmap}_single()` 的接口，它们使用页与页内偏移，而非 CPU 指针：

```c
struct device *dev = &my_dev->dev;
dma_addr_t dma_handle;
struct page *page = buffer->page;
unsigned long offset = buffer->offset;
size_t size = buffer->len;
dma_handle = dma_map_page(dev, page, offset, size, direction);
if (dma_mapping_error(dev, dma_handle)) {
    /*
     * reduce current DMA mapping usage,
     * delay and try again later or
     * reset driver.
     */
    goto map_error_handling;
}
...
dma_unmap_page(dev, dma_handle, size, direction);
```

> Here, “offset” means byte offset within the given page.

这里的 `offset` 指给定页面中的字节偏移。

> You should call dma_mapping_error() as dma_map_page() could fail and return error as outlined under the dma_map_single() discussion.

如 `dma_map_single()` 一节所述，`dma_map_page()` 也可能失败，应使用 `dma_mapping_error()` 检查。

> You should call dma_unmap_page() when the DMA activity is finished, e.g., from the interrupt which told you that the DMA transfer is done.

DMA 活动结束后，应调用 `dma_unmap_page()`，例如在报告传输完成的中断中。

> With scatterlists, you map a region gathered from several regions by:

对于 scatterlist，可如下映射由多个区域组成的内存：

```c
int i, count = dma_map_sg(dev, sglist, nents, direction);
struct scatterlist *sg;
for_each_sg(sglist, sg, count, i) {
    hw_address[i] = sg_dma_address(sg);
    hw_len[i] = sg_dma_len(sg);
}
```

> where nents is the number of entries in the sglist.

`nents` 是 `sglist` 中输入条目的数量。

> The implementation is free to merge several consecutive sglist entries into one (e.g. if DMA mapping is done with PAGE_SIZE granularity, any consecutive sglist entries can be merged into one provided the first one ends and the second one starts on a page boundary - in fact this is a huge advantage for cards which either cannot do scatter-gather or have very limited number of scatter-gather entries) and returns the actual number of sg entries it mapped them to. On failure 0 is returned.

实现可以将若干连续的 scatterlist 条目合并为一个，并返回实际映射得到的 SG 条目数。例如，若 DMA 映射以 `PAGE_SIZE` 为粒度，前一条目恰好在页边界结束、后一条目从该边界开始，就可合并。这对不支持 scatter/gather 或支持条目数很少的设备尤其有利。失败时返回零。

> Then you should loop count times (note: this can be less than nents times) and use sg_dma_address() and sg_dma_len() macros where you previously accessed sg->address and sg->length as shown above.

随后应循环 `count` 次（可能少于 `nents`），使用 `sg_dma_address()` 和 `sg_dma_len()` 取得设备实际使用的地址与长度，而非之前直接访问 `sg->address`、`sg->length`。

> To unmap a scatterlist, just call:

解除 scatterlist 映射时调用：

```c
dma_unmap_sg(dev, sglist, nents, direction);
```

> Again, make sure DMA activity has already finished.

同样，务必先确认 DMA 活动已经结束。

**Note / 注意**

> The ‘nents’ argument to the dma_unmap_sg call must be the _same_ one you passed into the dma_map_sg call, it should _NOT_ be the ‘count’ value _returned_ from the dma_map_sg call.

`dma_unmap_sg()` 的 `nents` 必须与传给 `dma_map_sg()` 的原始值相同，不能使用 `dma_map_sg()` 返回的 `count`。

**解读：** `count` 是映射后供设备遍历的段数；`nents` 是调用映射时传入的原始条目数。合并发生后两者可能不同，unmap/sync 仍用原始 `nents`。

> Every dma_map_{single,sg}() call should have its dma_unmap_{single,sg}() counterpart, because the DMA address space is a shared resource and you could render the machine unusable by consuming all DMA addresses.

每次 `dma_map_{single,sg}()` 都应有对应的 `dma_unmap_{single,sg}()`；DMA 地址空间是共享资源，不断消耗而不释放可能使机器无法正常工作。

> If you need to use the same streaming DMA region multiple times and touch the data in between the DMA transfers, the buffer needs to be synced properly in order for the CPU and device to see the most up-to-date and correct copy of the DMA buffer.

若要多次使用同一 streaming DMA 区域，并在两次设备传输之间由 CPU 访问数据，则必须正确同步缓冲区，使 CPU 和设备各自看到最新、正确的内容。

> So, firstly, just map it with dma_map_{single,sg}(), and after each DMA transfer call either:

首先用 `dma_map_{single,sg}()` 建立映射；每次 DMA 传输后，根据情况调用以下函数之一：

```c
dma_sync_single_for_cpu(dev, dma_handle, size, direction);
```

> or:

或者：

```c
dma_sync_sg_for_cpu(dev, sglist, nents, direction);
```

> as appropriate.

按具体情况选择。

> Then, if you wish to let the device get at the DMA area again, finish accessing the data with the CPU, and then before actually giving the buffer to the hardware call either:

随后，如果要再次把该 DMA 区域交给设备，CPU 先结束访问数据，再在真正交给硬件之前调用下列函数之一：

```c
dma_sync_single_for_device(dev, dma_handle, size, direction);
```

> or:

或者：

```c
dma_sync_sg_for_device(dev, sglist, nents, direction);
```

> as appropriate.

按具体情况选择。

**Note / 注意**

> The ‘nents’ argument to dma_sync_sg_for_cpu() and dma_sync_sg_for_device() must be the same passed to dma_map_sg(). It is _NOT_ the count returned by dma_map_sg().

`dma_sync_sg_for_cpu()` 与 `dma_sync_sg_for_device()` 的 `nents` 必须等于传给 `dma_map_sg()` 的值，而非 `dma_map_sg()` 返回的 `count`。

> After the last DMA transfer call one of the DMA unmap routines dma_unmap_{single,sg}(). If you don’t touch the data from the first dma_map_*() call till dma_unmap_*(), then you don’t have to call the dma_sync_*() routines at all.

最后一次 DMA 传输后，应调用相应的 `dma_unmap_{single,sg}()`。若从首次 `dma_map_*()` 到 `dma_unmap_*()` 之间 CPU 一直不碰数据，则无需调用 `dma_sync_*()`。

**解读：** map、sync、unmap 描述映射所有权与可见性的生命周期。它是理解 RDMA 数据路径的内核基础，不等同于应用层 `ibv_reg_mr()` 或 work completion 的完整语义。

> Here is pseudo code which shows a situation in which you would need to use the dma_sync_*() interfaces:

下列伪代码展示需要 `dma_sync_*()` 的情况：

```c
my_card_setup_receive_buffer(struct my_card *cp, char *buffer, int len) {
    dma_addr_t mapping;
    mapping = dma_map_single(cp->dev, buffer, len, DMA_FROM_DEVICE);
    if (dma_mapping_error(cp->dev, mapping)) {
        /*
         * reduce current DMA mapping usage,
         * delay and try again later or
         * reset driver.
         */
        goto map_error_handling;
    }
    cp->rx_buf = buffer;
    cp->rx_len = len;
    cp->rx_dma = mapping;
    give_rx_buf_to_card(cp);
}
...
my_card_interrupt_handler(int irq, void *devid, struct pt_regs *regs) {
    struct my_card *cp = devid;
    ...
    if (read_card_status(cp) == RX_BUF_TRANSFERRED) {
        struct my_card_header *hp;
        /*
         * Examine the header to see if we wish
         * to accept the data.  But synchronize
         * the DMA transfer with the CPU first
         * so that we see updated contents.
         */
        dma_sync_single_for_cpu(&cp->dev, cp->rx_dma, cp->rx_len, DMA_FROM_DEVICE);
        /*
         * Now it is safe to examine the buffer.
         */
        hp = (struct my_card_header *) cp->rx_buf;
        if (header_is_ok(hp)) {
            dma_unmap_single(&cp->dev, cp->rx_dma, cp->rx_len, DMA_FROM_DEVICE);
            pass_to_upper_layers(cp->rx_buf);
            make_and_setup_new_rx_buf(cp);
        }
        else {
            /*
             * CPU should not write to
             * DMA_FROM_DEVICE-mapped area,
             * so dma_sync_single_for_device() is
             * not needed here. It would be required
             * for DMA_BIDIRECTIONAL mapping if
             * the memory was modified.
             */
            give_rx_buf_to_card(cp);
        }
    }
}
```

## Handling Errors / 处理错误

> DMA address space is limited on some architectures and an allocation failure can be determined by:

某些架构的 DMA 地址空间有限，可通过以下方式判断分配失败：

- checking if dma_alloc_coherent() returns NULL or dma_map_sg returns 0
  检查 `dma_alloc_coherent()` 是否返回 `NULL`，或 `dma_map_sg()` 是否返回零。

- checking the dma_addr_t returned from dma_map_single() and dma_map_page() by using dma_mapping_error():
  对 `dma_map_single()` 与 `dma_map_page()` 返回的 `dma_addr_t` 调用 `dma_mapping_error()`。

```c
dma_addr_t dma_handle;
dma_handle = dma_map_single(dev, addr, size, direction);
if (dma_mapping_error(dev, dma_handle)) {
    /*
     * reduce current DMA mapping usage,
     * delay and try again later or
     * reset driver.
     */
    goto map_error_handling;
}
```

- unmap pages that are already mapped, when mapping error occurs in the middle of a multiple page mapping attempt. These example are applicable to dma_map_page() as well.
  在映射多个页面途中出错时，解除先前已建立的页面映射；下列例子也适用于 `dma_map_page()`。

> Example 1:

例 1：

```c
dma_addr_t dma_handle1;
dma_addr_t dma_handle2;
dma_handle1 = dma_map_single(dev, addr, size, direction);
if (dma_mapping_error(dev, dma_handle1)) {
    /*
     * reduce current DMA mapping usage,
     * delay and try again later or
     * reset driver.
     */
    goto map_error_handling1;
}
dma_handle2 = dma_map_single(dev, addr, size, direction);
if (dma_mapping_error(dev, dma_handle2)) {
    /*
     * reduce current DMA mapping usage,
     * delay and try again later or
     * reset driver.
     */
    goto map_error_handling2;
}
...
map_error_handling2: dma_unmap_single(dma_handle1);
map_error_handling1:
```

> Example 2:

例 2：

```c
/*
 * if buffers are allocated in a loop, unmap all mapped buffers when
 * mapping error is detected in the middle
 */
dma_addr_t dma_addr;
dma_addr_t array[DMA_BUFFERS];
int save_index = 0;
for (i = 0; i < DMA_BUFFERS; i++) {
    ...
    dma_addr = dma_map_single(dev, addr, size, direction);
    if (dma_mapping_error(dev, dma_addr)) {
        /*
         * reduce current DMA mapping usage,
         * delay and try again later or
         * reset driver.
         */
        goto map_error_handling;
    }
    array[i].dma_addr = dma_addr;
    save_index++;
}
...
map_error_handling:
for (i = 0; i < save_index; i++) {
    ...
    dma_unmap_single(array[i].dma_addr);
}
```

> Networking drivers must call dev_kfree_skb() to free the socket buffer and return NETDEV_TX_OK if the DMA mapping fails on the transmit hook (ndo_start_xmit). This means that the socket buffer is just dropped in the failure case.

网络驱动在发送入口 `ndo_start_xmit` 中若发生 DMA 映射失败，必须调用 `dev_kfree_skb()` 释放 socket buffer，并返回 `NETDEV_TX_OK`，也就是在失败时丢弃该 buffer。

> SCSI drivers must return SCSI_MLQUEUE_HOST_BUSY if the DMA mapping fails in the queuecommand hook. This means that the SCSI subsystem passes the command to the driver again later.

SCSI 驱动在 `queuecommand` 中映射失败时，必须返回 `SCSI_MLQUEUE_HOST_BUSY`，由 SCSI 子系统稍后再次提交命令。

## Optimizing Unmap State Space Consumption / 优化 unmap 状态空间占用

> On many platforms, dma_unmap_{single,page}() is simply a nop. Therefore, keeping track of the mapping address and length is a waste of space. Instead of filling your drivers up with ifdefs and the like to “work around” this (which would defeat the whole purpose of a portable API) the following facilities are provided.

在许多平台，`dma_unmap_{single,page}()` 实际上不做任何事，保存映射地址和长度会浪费空间。为避免驱动编写大量平台条件编译代码（这会破坏可移植 API 的意义），内核提供了下述工具。

> Actually, instead of describing the macros one by one, we’ll transform some example code.

这里不逐个介绍宏，而用前后对比改写一段示例代码。

1. Use DEFINE_DMA_UNMAP_{ADDR,LEN} in state saving structures. Example, before:
   在保存状态的结构中使用 `DEFINE_DMA_UNMAP_{ADDR,LEN}`。修改前：

```c
struct ring_state {
    struct sk_buff *skb;
    dma_addr_t mapping;
    __u32 len;
};
```

> after:

修改后：

```c
struct ring_state {
    struct sk_buff *skb;
    DEFINE_DMA_UNMAP_ADDR(mapping);
    DEFINE_DMA_UNMAP_LEN(len);
};
```

2. Use dma_unmap_{addr,len}_set() to set these values. Example, before:
   用 `dma_unmap_{addr,len}_set()` 保存这些值。修改前：

```c
ringp->mapping = FOO;
ringp->len = BAR;
```

> after:

修改后：

```c
dma_unmap_addr_set(ringp, mapping, FOO);
dma_unmap_len_set(ringp, len, BAR);
```

3. Use dma_unmap_{addr,len}() to access these values. Example, before:
   用 `dma_unmap_{addr,len}()` 读取这些值。修改前：

```c
dma_unmap_single(dev, ringp->mapping, ringp->len, DMA_FROM_DEVICE);
```

> after:

修改后：

```c
dma_unmap_single(dev, dma_unmap_addr(ringp, mapping), dma_unmap_len(ringp, len), DMA_FROM_DEVICE);
```

> It really should be self-explanatory. We treat the ADDR and LEN separately, because it is possible for an implementation to only need the address in order to perform the unmap operation.

其含义较直观。地址与长度分开处理，是因为某些实现执行 unmap 时只需要地址。

## Platform Issues / 平台问题

> If you are just writing drivers for Linux and do not maintain an architecture port for the kernel, you can safely skip down to “Closing”.

若只编写 Linux 驱动而不维护内核的架构移植，可直接跳到“Closing”。

1. Struct scatterlist requirements.
   `struct scatterlist` 的要求。

> You need to enable CONFIG_NEED_SG_DMA_LENGTH if the architecture supports IOMMUs (including software IOMMU).

如果架构支持 IOMMU（包括软件 IOMMU），需要启用 `CONFIG_NEED_SG_DMA_LENGTH`。

2. ARCH_DMA_MINALIGN
   `ARCH_DMA_MINALIGN`。

> Architectures must ensure that kmalloc’ed buffer is DMA-safe. Drivers and subsystems depend on it. If an architecture isn’t fully DMA-coherent (i.e. hardware doesn’t ensure that data in the CPU cache is identical to data in main memory), ARCH_DMA_MINALIGN must be set so that the memory allocator makes sure that kmalloc’ed buffer doesn’t share a cache line with the others. See arch/arm/include/asm/cache.h as an example.

架构必须保证 `kmalloc()` 分配的缓冲区可以安全用于 DMA，因为驱动和子系统依赖这一点。若架构并非完全 DMA coherent（即硬件不保证 CPU 缓存中的数据与主内存相同），必须设置 `ARCH_DMA_MINALIGN`，使分配器确保该缓冲区不与其他数据共享缓存行。原文以 `arch/arm/include/asm/cache.h` 为例。

> Note that ARCH_DMA_MINALIGN is about DMA memory alignment constraints. You don’t need to worry about the architecture data alignment constraints (e.g. the alignment constraints about 64-bit objects).

`ARCH_DMA_MINALIGN` 针对 DMA 内存的对齐约束，与架构对 64 位对象等数据类型的普通对齐要求无关。

**整理注：** 用户提供的文本到此结束。原文提到的 “Closing” 未包含在粘贴内容中，因此本笔记不补写该节。

## 附录：在 RDMA 系列学习规划中的位置

本篇对应[[02-AISystem/Distributed/RDMA/RDMA#八阶段学习规划|八阶段学习规划]]的第 1 阶段：先画清 CPU virtual address、physical address、DMA address 与 IOMMU 的关系，再进入 RDMA 总体模型和 Memory Registration。后续顺序是 verbs 资源模型 → verbs 执行模型 → Transport/Fabric 与 perftest → GPU RDMA → UCX/NCCL。每阶段的理解产物、实践产物及官方资料分工集中记录在 RDMA 主题入口；本篇正文仍按 Linux DMA 文档原有顺序供逐段对照。
