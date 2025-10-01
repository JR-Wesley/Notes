Ref:

1. LITE Kernel RDMA Support for Datacenter Applications（SOSP'17），https://cseweb.ucsd.edu/~yiying/LITE-sosp17.pdf

2. Scalable RDMA RPC on Reliable Connection with Efficient Resource Sharing (EuroSys'19), https://chenyoumin1993.github.io/papers/eurosys19-scalerpc.pdf

3. Design Guidelines for High Performance RDMA Systems (ATC'16), https://anujkalia.com/doc/atc16/rdma_bench_atc.pdf

4. https://zhuanlan.zhihu.com/p/567720023

1 RDMA 编程回顾

介绍具体 RDMA 编程前，先以类似于传统 socket 编程的 send/recv 为例，了解一下 RDMA 微观的数据传输过程，如下图所示，主要分为以下几步：

创建 PD (Protection Domain)、QP (Queue Pair)、CQ (Completion Queue)，建立收发两端 QP 之间的连接（类似于 socket 功能）；

接收端注册用户内存到网卡 RNIC，并发起接收的请求，该请求中就包括了注册到网卡的内存地址和长度；

发送端注册用户内存到网卡 RNIC，并发起发送的请求，该请求中就包括了注册到网卡的内存地址和长度；

发送端网卡执行发送请求，根据发送队列 SQ 中发送请求的内存地址和长度到用户态内存中直接读取数据发送给接收端；

当数据到达接收端时，接收端网卡执行接收请求，根据接收队列 RQ 中接收请求的内存地址和长度将接收的数据直接写到相应的位置；

接收端数据接收完成后产生一个完成通知 CQE 到完成队列 CQ 中，程序从完成队列中取出完成通知 CQE 代表整个传输过程的结束。

RDMA send 微观传输图.jpg

1.1 RDMA 参数
如下图所示，标注了整个过程涉及的主要参数：

QP 类型：RC、UC、UD（R: reliable, U: unreliable, C: connection, D: datagram），QP 的类型需要在建立连接时确定，就像在建立 socket 通信时，需要确定是 TCP 还是 UDP 的连接类型。其中，R、U 的区别在于是否是可靠传输，也就是是否返回 ack，C、D 的区别在于是否是面向连接的，C 指的是面向连接的，类似于 TCP，在数据传输前，会先建立好连接，也就是 QP 互相交换相应信息，而 D 指的是 datagram，QP 之间连接的信息不是事先确定好的，而是放在数据包的包头，由数据包决定所要发送的具体的接收端的 QP 是哪个；

Verb：send/recv、write、read，具体的传输数据的方式，send/recv 和 socket 类似，write 指的是将数据从本地直接写到远端内存，不需要远端 CPU 参与，read 指的是将数据从远端直接读到本地，同样不需要远端 CPU 参与；

Inline/non-inline：inline 在 C++ 里指的是内联，在程序编译时直接用函数代码替换函数调用，节省时间，但针对的是那些执行时间较短的函数。同样，在 RDMA 里面，inline 指的就是将一些小的数据包内联在发送请求中，这样在 2.1 中 RDMA 数据传输的第四步，就可以少一次取数据的过程；

Signal/unsignal：signal/unsignal 指的是是否产生 CQE，如果使用 unsignal，将不会产生 CQE，因此也就不需要 poll CQE，从而减少 CPU 使用，提升性能。但是，如果使用了 unsignal，必须保证隔一段时间发送一个 signal 的请求，因为如果没有 CQE 的产生以及 poll，那么这些 unsignal 的发送请求将一直占用着发送队列，当发送队列满时，将不能再 post 新的请求；

Poll 策略：poll 策略指的是 poll CQE 的方式，包括 busy polling、event-triggered polling，busy polling 以高 CPU 使用代价换取更快的 CQE poll 速度，event-triggered polling 则相应的 poll 速度慢，但 CPU 使用代价低；

在了解了 RDMA 各种参数后，接下来将介绍具体的 RDMA 编程中，这些参数是如何体现的（在具体参数选择出会用黄色背景标注出）。

RDMA 传输参数图.png

1.2 send/recv
这里 send/recv 指的就是上述 verb 的类型，因为不同 verb 的通信过程还是有一定区别的，因此分开介绍。那么 send/recv 在代码的哪里指定呢？我们首先按照 2.1 中 RDMA 数据传输的步骤一步一步来：

1.2.1 建立连接
为了简便建立连接的过程，我们利用 rdmacm 库提供的一系列接口来实现。值得注意的是 QP 的创建必须在 rdma_connect 以及 rdma_accept 之前，因为这两个函数主要封装了 QP 信息的交换，而如果不使用 rdmacm 库的话，自己也可以利用 socket 写一套交换 QP 信息的程序。

Server 端：

监听连接：

1

2

3

4

ec = rdma_create_event_channel();

rdma_create_id(ec, &listener, NULL, RDMA_PS_TCP);

rdma_bind_addr(listener, (struct sockaddr *)&addr);

rdma_listen(listener, 10);

创建 QP：

1

2

3

4

5

6

7

8

9

10

qp_attr->send_cq = s_ctx->cq;

qp_attr->recv_cq = s_ctx->cq;

qp_attr->qp_type = IBV_QPT_RC;

qp_attr->cap.max_send_wr = 10;

qp_attr->cap.max_recv_wr = 10;

qp_attr->cap.max_send_sge = 1;

qp_attr->cap.max_recv_sge = 1;

rdma_create_qp(id, s_ctx->pd, &qp_attr)；这里要注意的是 qp_attr，顾名思义，qp 的属性，那么 qp 的类型也就是在这个结构中指定，以 RC 连接类型为例：qp_attr->qp_type = IBV_QPT_RC;

完成连接：

1

rdma_accept(id, &cm_params);

Client 端：

解析地址、路由：

1

2

3

4

ec = rdma_create_event_channel();

rdma_create_id(ec, &conn, NULL, RDMA_PS_TCP);

rdma_resolve_addr(conn, NULL, addr->ai_addr, TIMEOUT_IN_MS);

rdma_resolve_route(id, TIMEOUT_IN_MS);

创建 QP：

参考 server 端；

发起连接：

1

rdma_connect(id, &cm_params);

1.2.2 注册内存

1

ibv_reg_mr( s_ctx->pd, conn->send_region, BUFFER_SIZE,IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE) );

其中，第二和第三个参数分别代表用户态内存地址和长度；最后一个参数指的是访问该内存的权限：IBV_ACCESS_LOCAL_WRITE 指的是允许本机的写权限、IBV_ACCESS_REMOTE_WRITE 指的是允许远端的写权限、本地的读权限是默认的、IBV_ACCESS_REMOTE_READ 指的是允许远端的读权限。

1.2.3 发起接收请求

前面步骤中介绍了接收请求主要包括了注册到网卡的内存地址和长度，那么体现在代码中就是 wr（接收请求）的 sg_list 成员，指定完内存后，ibv_post_recv 函数就完成了将请求发送给网卡接收队列的任务。其中第一个参数指的就是网卡中的具体 QP。

1

2

3

4

5

6

7

wr.sg_list = &sge;

sge.addr = (uintptr_t)conn->recv_region;

sge.length = BUFFER_SIZE;

sge.lkey = conn->recv_mr->lkey;

ibv_post_recv(conn->qp, &wr, &bad_wr);

1.2.4 发起发送请求
指定内存的部分和接收请求一样，发送请求比接收请求主要多了三个参数的设置，也就是前面介绍参数时说的 verb、Inline/non-inline 以及 Signal/unsignal。

1

2

3

4

5

6

7

8

9

wr.opcode = IBV_WR_SEND;

wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;

wr.sg_list = &sge;

sge.addr = (uintptr_t)conn->send_region;

sge.length = BUFFER_SIZE;

sge.lkey = conn->send_mr->lkey;

ibv_post_send(conn->qp, &wr, &bad_wr);

1.2.5 从完成队列中 poll 完成通知

event-triggered polling 指的就是需要下面前三行事件通知的代码，busy polling 就不需要前三行代码，直接循环地 poll cq，因此消耗更多的 CPU。

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

void * poll_cq(void *ctx)

{

  struct ibv_cq *cq;

  struct ibv_wc wc;

  while (1) {

    TEST_NZ(ibv_get_cq_event(s_ctx->comp_channel, &cq, &ctx));

    ibv_ack_cq_events(cq, 1);

    TEST_NZ(ibv_req_notify_cq(cq, 0));

 
    while (ibv_poll_cq(cq, 1, &wc))
      on_completion(&wc);

  }

  return NULL;

}

1.3 write/read
在具体介绍 write/read 编程前，先以 write 为例看一下 write 的微观传输图，比较它和 send/recv 的区别。如图 6 所示，首先，write 不需要在接收端发起接收请求，但是相应的需要将注册的内存地址和 key 值（代表写的权限）发送给发送端，然后发送端在发起发送请求时，就会包括接受端的这个内存地址和 key 值，直接将数据写到远端内存，而不需要接收端 CPU 参与。

那么接收端如何将自己注册的内存地址和 key 值发给发送端呢？很简单，既然我们上面已经掌握了 send/recv 的编程，那么就直接用 send/recv 发送，具体过程参考 2.3，这里主要介绍一下发送端在接收到这些信息后，如何将本地数据写到远端，和 send/recv 相同的参数不再赘述，主要看第二、三行，分别是远端地址和 key 值，另外 opcode 变成了 WRITE：

1

2

3

4

wr.opcode = IBV_WR_RDMA_WRITE;

wr.wr.rdma.remote_addr = (uintptr_t)M.remote_addr;

wr.wr.rdma.rkey = M.rkey;

ibv_post_send(conn->qp, &wr, &bad_wr);

2 RDMA 性能调优

背景和优化思路

RDMA 性能扩展背景和挑战

当旁路内核时，特权操作和数据结构被卸载到硬件 RNIC。由于 RNIC 上的 SRAM 有限，三个因素限制了 RDMA 性能的扩展：MRs 的数量，MRs 的总大小，qp 的总数量。

1、RNICs 存储所有注册的 MR 的 lkey, rkey 和虚拟内存地址 (VA)，随着 MR 的数量增加，RNIC 很快会面临内存压力。

使用大的 MR size 可以减少 MR 的数量，但是需要大量应用改写，同时产生内存浪费。

2、RNIC 缓存 MR 相关的页表项（PTE, Page Table Entry），将 RDMA 请求的虚拟内存地址转化成 DMA 地址。当 RNIC 追踪有 PTE miss 时，RNIC 需要从 host OS 中取 PTE。当注册的 MR 的总大小超过 RNIC 的处理能力，就会发生性能降级。

FaRM 使用 2GB huge page 来减小 MR size 的可扩展问题，但是使用大页会带来内存占用增加、碎片化、false memory sharing 和 NUMA 性能降低的问题。

3、RNIC 存储每个 QP 的元数据。当 QP 数量增加的时候，RDMA 的性能降低。这限制了采用可靠连接（RC）的集群规模。虽然可采用不可靠数据报（UD）来减少 QP 数量，但 UD 不可靠且不支持单边 RDMA 读写语义。

RDMA 原语中涉及很多 MMIO、DMA 操作和 CPU 参与的操作。

性能调优的主要思路

第一类：减少 MMIO/DMA 操作 或 CPU 参与

第二类：QP 数量、MR 相关的元数据、MR 大小

减少 CPU 使用

2.1 poll strategy

2.2 unsigned completion

减少 MMIO/DMA 操作

2.3 doorbell batching

2.4 inline

2.5 blue flame

减少 QP 数量

2.6 RDMA XRC

2.7 DCI/DCT

MR size & MR number

2.8 ODP

2.9 Streaming RDMA

2.11 rendezvous

使用模式

2.10 Stride Send/Recv

2.11 rendezvous（offload）

2.1 poll strategy
轮询 vs. 中断

poll 策略指的是 poll CQE 的方式，包括 busy polling、event-triggered polling，busy polling 以高 CPU 使用代价换取更快的 CQE poll 速度，event-triggered polling 则相应的 poll 速度慢，但 CPU 使用代价低。

以 send/recv 为例比较 poll 参数对延时性能的影响：

2.2 unsigned completion
概念：unsigned completion 允许应用程序发送请求时指定 WQE 完成之后是否要产生 CQE。如果发送请求以错误终止，即使已设置为 unsigned，仍会生成 CQE。

作用：通过选择性抑制发送请求的完成通知，减少 CPU 轮询开销，提升性能。

适用场景：

1、使用的内存区域等资源无需销毁或重用；

2、应用程序可以在无需确认完成前连续发送多条消息 或者 读取多个缓冲区。

如何使用 unsigned completion（无信号完成）？

无信号完成支持按队列对（Queue Pair）配置。创建队列对时，需设置其发送队列支持无信号完成（将属性 qp_init_attr.sq_sig_all 设为 0）。

对于每个提交的发送请求，若在 wr.send_flags 中设置 IBV_SEND_SIGNALED 标志，则处理结束时生成工作完成通知；若未设置，则无错误完成时不生成通知。

示例代码：https://www.rdmamojo.com/2014/06/30/working-unsignaled-completions/，参考该网站给出的 example。

注意：标志位是 IBV_SEND_SIGNALED 而不是 UNSIGNALED，在队列对创建时全局启用无信号支持，并通过每个发送请求的标志位动态控制通知生成。

2.3 doorbell batching
doorbell 概念：敲 doorbell 的目的是通知 NIC 工作队列中有新的 WQE，这个操作时通过 MMIO（memory-mapped I/O）修改 NIC 上的特定寄存器。

性能影响：MMIO write 会引入延时，由于旁路 CPU cache 且与硬件交互所以相对 expensive。减少 doorbell ring 操作数量可以减少 MMIO write 的个数，从而提升性能。

doorbell batching：将多个 wr（work request）链接成链表，批量 post 它们，而不是每个 wr 都敲一次，从而减少敲 doorbell 的次数。

使用方法：

doorbell batching 用法示例

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

// Create send WRs and SGEs

std::vector<struct ibv_sge> sge_list;

std::vector<struct ibv_send_wr> wr_list;

struct ibv_send_wr *bad_wr;

for (int i = 0; i < 10; ++i) {

    struct ibv_sge sge = create_sge((uint64_t)(local_buf + i * 100), 100, mr->lkey);

    sge_list.push_back(sge);

    struct ibv_send_wr wr = create_send_wr(&sge_list.back(), remote_addr + i * 100, rkey);

    wr_list.push_back(wr);

}

// Link WRs for doorbell batching

for (size_t i = 0; i < wr_list.size() - 1; ++i) {

    wr_list[i].next = &wr_list[i + 1];

}

// Post the batch of WRs

int ret = ibv_post_send(qp, &wr_list[0], &bad_wr);

CHECK_ERR(ret == 0);

2.4 inline/non-inline
inline: payload 和 WQE 通过 MMIO 一次性完成，减少了一次 DMA 取数据的操作。

使用场景：数据很小时可以用 inline，因为 WQE 一般只有 64 Byte 或 128 Byte。一般用于数据同步，和原子操作类似，但粒度更大。

以 send/recv 为例比较 inline/non-inline 参数对延时性能的影响：

2.5 blue flame
概念：将 WQE 和 doorbell 一起写，这样能省去一次 DMA 取 WQE 的时延。(非 SPEC，Mellanox 特有)

使用场景：时延要求敏感，且只有少部分 qp 时，使用 blue flame。

bar 空间资源有限，只能用于部分 qp。对于商用卡，比如 Mellanox 的卡，qp 可以达到 128K，没办法把所有 QP 的 WQE 都放到 bar 空间中。

如何使用：WQE 结构体里有个字段，告诉硬件是不是用的 blue flame。

NVShmem 用到了 blue flame，我们没用到。

SPEC 标准一般可以在 ibv_xx 接口看到，非 SPEC 在 ibv_xxx 接口看不到，但在用户态驱动里可以看到。

2.6 RDMA XRC
提出背景：RC 场景下，host 0 上的 m 个应用进程和 host 1 上的 n 个应用进程通信，需要 m * n 个 qp (即 full mesh 的 qp 个数）。

为了减少 qp 数量，提出 XRC（extended RC，扩展 RC）。

核心思想：当一个进程想与某个远程节点的 p 个进程通信时不需要跟各个进程建立 p 个连接，而只需要跟对端节点建立一个连接，连接上传输的报文携带了对端目的进程号（XRC SRQ），报文到达连接对端（XRC TGT QP）时根据进程号分发至各个进程对应的 XRC SRQ。这样远端进程只需要创建一个源端连接（XRC INI QP）就能跟对端所有进程通信了，所需总的 QP 数量除以 p。

2.7 DCI/DCT（Dynamic Connection initiator/target）
使用场景：主要用来解决 RC 场景下 QP 数量太多的问题。虽然可采用不可靠数据报（UD）来减少 QP 数量，但 UD 不可靠且不支持单边 RDMA 读写语义。

DCI/DCT 特点：可靠、支持写语义、QP 数量不会太多。

NVShmem 用到，NCCL 没用到。

使用说明：DCI/DCT 和 RC/UD 的概念并列。创建 qp 时会指定 qp_type，RC 是 0，UD 是 1。DCI/DCT 的 qp_type=0xffff。

RDMA_core 和开源社区已经支持。该特性需要硬件支持，目前，Mellanox 已经支持，国内没看过谁支持。

编程范式：如左图所示，在创建 QP 时，会指定 QP 属性中的 dc_type 到底是 DCI 还是 DCT。

发包的时候，通过 mlx5dv_wr_set_dc_addr (dv_qp, ah, rem_dest→dctn, DC_KEY); 将远端 dst 的地址和 key 信息发给远端，硬件识别后可以与相应的 dst 建链和通信。

其中，ah 表示 address handler。DC Connect 和 DC Disconnect 是硬件行为。

2.8 on-demand paging (ODP)
提出背景：正常情况下，会注册一个 MR，记录 VA 到 PA 的映射关系放到 MTT 表中，并把这块内存 pin 住。硬件在发起 DMA 操作时，需要这块内存不被换出。

如果 RDMA 需要的内存比较多，别人要用的内存都因为 pin 被占用了，那么整机性能会下降。

使用场景：用于优化内存占用，允许 MR 对应的内存被换入换出。

等到发起 DMA 时，如果访问的页不在，则触发 page fault，把内存页换入并更新页表。然后再 DMA 去取。

如何使用：

注册 MR 时指定 IBV_ACCESS_ON_DEMAND 标识则创建 ODP MR，其初始地址翻译表里 VA 对应的物理页并不存在，因此设备首次访问 MR VA 时会产生 IO page fault(IOPF)，HCA 驱动处理此 IOPF 并换入所需物理页更新 HCA 里的地址翻译表，则下次设备 DMA 时不再发生 IOPF。若操作系统决定 swap out VA 对应的物理页，也会由 HCA 驱动更新地址翻译表将 VA 对应 entry 置为 page none-present。

如果内存够用，一般不会开启 ODP，因为开启影响性能。

2.9 Streaming RDMA
提出背景: 传统的 QP receive WQE 中，每个接收到的消息都消耗一个 receive buffer。receive buffer 是预分配的，由于无法预测消息大小，一般按最大尺寸分配内存。

这样存在两个问题：

收到的消息比 receive buffer 大，无法接收只能被丢弃；如果收到的消息比 receive buffer 小很多， receive buffer 的内存利用率低。

频繁读取 receive WQE 会成为性能瓶颈。（WQE 64B, payload 64B）

概念：每个进来的消息只消费它自身的大小而不是整个 receive WQE，接收缓冲区剩下的部分可以接收下一个消息，因此一个 receive WQE 可以接收多个消息只要它能容纳。这就解决了接收缓冲区利用率低和频繁读取 receive WQE 的问题。

Streaming receive 时每个消息还是会上报一个 CQE，所以 CQE 中要指明消息放在 WQE receive buffer 的起始和终止位置。一个消息只能放在一个 WQE receive buffer 中而不能跨 WQE。

注意：

一般来说，一个 WQE 对应一个 CQE，send queue 和 cq 队列深度配置成一样；

使用 Streaming RDMA 时，CQE 的数量将大于 RQ 中 WQE 的数量，所以创建 CQ 时需要把 CQ 的队列深度配置成 Send Queue 的 4 倍或者 8 倍。

2.10 Stride Send/Recv
stride send (上)：把一段有规律的非连续数据发送到对端再存储在一段连续的内存空间中。一般用 base address, block size, stride size, repeat count 四个参数描述。

stride receive (下)：和 stride send 为对称操作，用于将本地一段连续的数据发送到对端，且在对端以有规律非连续的形式存储。在接收端以四个参数描述存放位置。

区别：stride send 和 SGL（scatter gather list, sglist）

SGL 一般只用于描述少于 16 块的非连续数据，且 SGL 链表中的内存区域 size 不一定相同；

stride send 所需发送的非连续内存块多至 K 级别，且步距一致。

注意：SCCL 两种模式都没用，share memory 用了 stride send (上) 没用 stride receive (下)。

stride send 应用场景举例：

一块数据写在 master die 上，一块数据写在 slave die 上，想等数据都生产好了之后发到远端，可以用 gather 这种模式。

SCCL 双 die 场景下没这么用，是用两个 sge 发到远端。

2.11 send/recv 的两种常用方式
eager：处理消息时直接将 payload 放在报文里发给接收方。

rendezvous：发送小的控制消息 rndv，告诉接收方 payload 的地址、长度和 key，接收方在接收缓冲区 ready 后反向读数据。

rendezvous 可以软件实现也可以硬件实现。硬件实现时，硬件自动执行红色方框内的拉取数据操作。

提出动机：

1、eager 模式下，如果消息太大，接收缓冲区放不下，则会失败。

2、双边 send/recv 模式需要 CPU 参与，如果消息很大，处理消息时 CPU 被占用导致 server 端并发上不去；如果将数据操作 offload 给 client（发送方/请求方）就能减少 server 端（接收方/响应方）的 CPU 占用。

如果厂家不支持，调用了这个接口，没有相应的回调函数，就会返回失败给用户。

Rendezvous 模式中，也需要显式地调用 post recv。后面红色的部分不需要接收方显式地调用。

Ref:

1. LITE Kernel RDMA Support for Datacenter Applications（SOSP'17），https://cseweb.ucsd.edu/~yiying/LITE-sosp17.pdf

2. Scalable RDMA RPC on Reliable Connection with Efficient Resource Sharing (EuroSys'19), https://chenyoumin1993.github.io/papers/eurosys19-scalerpc.pdf

3. Design Guidelines for High Performance RDMA Systems (ATC'16), https://anujkalia.com/doc/atc16/rdma_bench_atc.pdf

4. https://zhuanlan.zhihu.com/p/567720023

5. SCCL Framework and Knowledge Sharing
