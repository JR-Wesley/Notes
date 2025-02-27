---
dateCreated: 2024-05-18
dateModified: 2025-02-27
---
# 参考

[皓宇的筆記 (wordpress.com)](https://timsnote.wordpress.com/)

# RAM 生成

[数字后端 synopsys 生成SRAM ROM的方法_ra1shd-CSDN博客](https://blog.csdn.net/qq_31993233/article/details/104977870?spm=1001.2014.3001.5502)

# 各类型文件的作用

 1. .sdc

标准延时约束文件，里面包含对面积、输入输出 delay、maxfanout, maxtransition, 等约束，由 DC 产生

1. .ddc
包含时序约束之外，还包含基本的布局物理信息. 做一次综合，然后让后端布一个初步的物理信息，然后再做综合，吐出的. ddc 文件，里面的时序信息更准确，而且和后端的一致性更好。

2. .svf
DC 改变了 RTL 代码的结构，但是逻辑没有改变。比如很调整组合逻辑的位置，但是最终的功能是一样的。写出一个文件的后缀名为. svf，该文件用于后面的形式验证，即 formality.

3. .gds 2
集成电路版图（英语：integrated circuit layout），是真实集成电路物理情况的平面几何形状描述。集成电路版图是集成电路设计中最底层步骤物理设计的成果.
用来描述掩膜几何图形的事实标准，是二进制格式，人类不可读。内容包括层和几何图形，文本或标签，以及其他有关信息并可以由层次结构组成。GDSII 数据可用于重建所有或部分的版图信息。它可以用作制作光刻掩膜版
通过 synopsis 公司的 Milkyway 工具可将. gds 文件转为 ICC 可读的 CELL view 格式

4. .clf
Caltech intermediate format，叫 Caltech 中介格式，是另一种基本文本的掩膜描述语言。以前用来描述 power , timing , logic 信息

5. .lef
(library exchange format）, 叫库交换格式，它是描述库单元的物理属性，包括端口位置、层定义和通孔定义。它抽象了单元的底层几何细节，提供了足够的信息，以便允许布线器在不对内部单元约束来进行修订的基础上进行单元连接。
LEF 文件分为技术 LEF 和单元 LEF。其中单元 LEF 又分为标准单元 LEF 和 IP marco LEF
包含了工艺的技术信息，如布线的层数、最小的线宽、线与线之间的最小距离以及每个被选用 cell，BLOCK，PAD 的大小和 pin 的实际位置。cell，PAD 的这些信息由厂家提供的 LEF 文件给出，自己定制的 BLOCK 的 LEF 文件描述经 ABSTRACT 后生成，只要把这两个 LEF 文件整合起来就可以了。
通过 synopsis 公司的 Milkyway 工具可将. lef 文件转为 ICC 可读的 FRAM view 格式

6. .def
Design exchange format ,设计交换格式，描述的是实际的设计，对库单元以及他的位置和连接关系进行了列表，使用 def 来在不同的设计系统间传递设计，同时又可以保持设计的内容不变。def 还给出了器件的物理位置关系和时序限制信息等. 一般可由后端工具吐出，比如我做一个 floorplan, 为了下次直接使用这次的结果，我会保存一个. def 文件，下次直接读入。
DFT 阶段也可以吐出一个 scan. def，将来 ICC 可直接读入。

7. .sdf
标准延迟格式（英语：Standard Delay Format, SDF）是电气电子工程师学会关于集成电路设计中时序描述的标准表达格式。在整个设计流程中，标准延迟格式有着重要的应用，例如静态时序分析和后仿真。将 SDF 文件反标到设计中.

8. .DSPF
(detailed standard parasitic format), 属于 cadence 公司的详细标准寄生参数格式

9. .RSPF
（reduced standard parasitic format）属于 cadence 公司的精简标准寄生参数格式

10. .SBPF
(synposys binary parasitic format）属于 synopsis 新思科技二进制寄生格式

11. .SPEF
(standard parasitic exchange format ) 标准寄生交换格式，属于 IEEE 国际标准文件格式。
以上四种文件格式都是从网表中提取出来的寄生参数。用来时序分析

12. .itf
interconnect technology format file 含每层的厚度，面积等参数

13. .tluplus
(nxtgrd which consists of capacitance models), TLUPlus 是存储 RC 系数的二进制表格式。TLUPlus 模型通过包括宽度，空间，密度和温度对电阻系数的影响，可以实现精确的 RC 提取结果
Itf 文件转为 TLUplus
itf–>tlu+

grdgenxo -itf 2 TLUPlus -i *. ift -o *. tlu+ //tlu+

1. .nxtgrd
是半可读的, nxtgrd 不能直接转成 tlu+，但是你打开 nxtgrd 里面的内容看看，前面的部分其实就是 itf 的内容，你把那些内容 copy 下来，就可以用 itf 转 tlu+

itf–>nxtgrd:

grdgenxo *. itf [run long time]//得到 nxgrd 但很慢

逻辑综合：使用 WLM 或者 topology 模型

route 之前：使用 virtual route & tluplus 模型

route 之后：使用 real route & tluplus 模型

Sign off、STA：使用 real route & nxtgrd 模型 extract 提取的 .spef 文件

1. .alf
(advanced library format), 用于描述基本库单元的格式，包含电性能参数

2. .PDEF
（physical design exchange format）
Synopsis 公司在前端和后端之间传递信息的文件格式。描述单元层之间分组相关的互连信息。这种文件只有在使用 synopys 公司的 physical compile 才会用到。

3. .lib
描述 cell 时序的文件，标准单元的 rise timing, hold timing ,每个 pin 的上升下降时间。power 信息，都是查找表方法

有的工艺库. lib 还有别的区别，例如：

CCSM:

Synopsis 的复合电流源模型，与 NLDM 不同，它是电流源模型，表现为 lib 中有 IVtable。

ECSM:

Cadence 的有效电流源模型，与 CCSM 一样，都是电流源模型，不同的是 ECSM 是对 Liberty 的补充 (Liberty 为 SNPS 所有），在 lib 中以 V (t) 曲线来描述。

CCSM 与 ECSM 中的 input cap 值都有多个，这一点与 NLDM 不同，这是因为在 90 nm 以下，input pin cap 是同时由 input slew 和 output load 来决定的。NLDM 与 spice 之间的误差精度能达到正负 5% 以内，然而 CCSM 和 ECSM 却能够达到惊人的 2%～3%（正负）

signoff 用 CCS/ECSM，PR 可以用 NLDM，这样效率高, CCS/ECSM 都特别大，工具读入都费劲，

CCS/ECSM 还包含 noise 信息，是 NLDM 没有的

1. .db
.lib 的二进制格式，人类不可读

2. Milky Way
ICC 可读的物理格式，分为 FRAM view，CELL view。
CELL view，包含的物理信息更加详细，但是由于 ICC 不需要太详细的物理，否则运行时间过长，一般只用 FRAM view。

# Trouble Shooting

由于设置了`default_nettype，导致 vcs 编译时报错

[Verilog的编译指令_celldefine-CSDN博客](https://blog.csdn.net/qq397381823/article/details/110193964)

Warning-[SDFCOM_NNTC] Need timing check option +neg_tchk

../dc_first/netlist/cor.sdf, 9833

module: DRNQVHSV1, "instance: test.theCor.theCordic1_stages_8__cordicStgs_aout_reg_0_"

  SDF Error: Negative HOLD value replaced by 0.

  Add +neg_tchk to consider Negative delay value.

？？？？

1、DesignWare 是前端的代码里写好，然后综合时把 synthetic_library 的位置选对就 OK 了么？对于综合的人还需要做什么？2、synthetic_library 里的.sldb 是根据什么选的，也就是说 set synthetic_library [concat …….]，我都需要选哪些.sldb，根据什么？
3、designware 里的 synthetic_library 是 DesignWare 里自带的么？这个库只是一个模型么？具体工艺根据这个模型用自己的工艺去完成这个 cell 或 module 么？

designware 是 synopsys 公司优化的一些基本单元，是工具自带的 lib（license 收费），designware 实际就是一些 IP，是由基本的逻辑门搭建而成，在综合的时候，工具根据具体工艺替换对应的单元

designware 就是 DC 提供的一系列优化过的算法单元，比如加法 乘法
使用时候只需要设定 synthetic_library 位对应的库即可，从 compile 的时候会自动调用。
如果你有 ultra 权限，即 compile_ultra，那么在使用 dw 的时候不需要设置 synthetic_library，
如果没有 ultra 权限，那么需要设定。

# IC 库命名规则和选择方法

scc55nll_hd_pmk_hvt_ss_v1p08_125c_b[asic](http://bbs.eetop.cn/forum-69-1.html).db 为例

scc55nll 是指 smic 的 55nm low leakage 工艺。

SS FF TT 是 corner。v1p08 是电压 1.08V（1.2V 下浮 10%）。ECSM 和 CCSM 差不多，不同公司叫法不同，不只是有时序、功耗，还有 SI。

带 pmk 的库，是带低功耗控制的库，内有 ISO 等信息。可以在综合的开始就使用，也可以在插入 PMU 之后再用
ss、ff、tt 是指的综合和时序分析中最主要的 3 个 corner，分别对应的 worst corner、best corner、typical corner
V1p08 是指 voltage 1 point 08, 1.08 福

1.综合主要分 3 个 corner，如 2 楼所指，在综合时候一般都选 ss corner 的库，在 pt 做 sta 时，可以使用 ff corner 来分析 hold

2.pmk，低功耗部分，在没有插入低功耗的时候，前端综合可以不用带 pmk 的。当插入了 UPF 或者 CPF 之后就必须要选带 pmk 的库

3.HD 和 HVT 这两个参数是根据设计属性来选的，如果设计的时序比较紧张。则要选择高速库，比如 HS 和 LVT，注意 LVT 是低阈值电压，阈值低，开启的速度快，但是漏电流大，功耗高

4.电压如果设计或者厂家有要求则按照要求选择，如果没有建议选恶劣的环境，即低电压

5.温度不同对综合影响不大，一般选高温库进行综合，125c

6.综合选择 basic 库


以上就是我个人对综合选库的一些想法，新手入门，理解不深。有错误请指正

|   |
|---|
|tcb  （TSMC 标准单元库）<br>n65  （65nm）<br>gplus（G+ 的工艺）<br>bwp  （tapless）<br>12t  （12 track）<br>_200a  （版本，越大越新）|

[芯片PM该知道的IC术语（三）设计用库_ecsm ccs-CSDN博客](https://blog.csdn.net/zt5169/article/details/85121842)

[SMIC工艺库的命名规则_各工艺libname-CSDN博客](https://blog.csdn.net/yeshengjiushizhu/article/details/83752704)

对于 SMIC 的工艺，其 PDK 命名方式为：xPyM_(y-v-z-w)Ic_vSTMc_zTMc_wMTTc_ALPAu

举个例子：1P6M_5Ic_1TMc_ALPA1，所以这里的 x=1，y=6，z=1，w=0，v=0,u=1，因而 y-v-z-w=6-0-1-0=5，没有 STM 和 MTT。

则 1P6M_5Ic_1TMc_ALPA1 代表的是 1 层多晶硅，6 层金属，内部 5 层铜，顶层铜为 1 层，铝焊盘或者铝重布线的厚度为 14.5KÅ。

补充：14.5KÅ铝工艺用于一般性的芯片，28KÅ一般用于包含 RF 等要求高的芯片中，这两种选择，是出于对性能和成本的要求而作出的。

## Tsmc

请问 1p7m 后面跟着的 _5X1Z0U

1p8m 5x1z1u means:1 poly, 1 metal1, 5 metal layer of x thickness, 1 metal layer of z thickness, and 1 metal layer of u thickness.



In most cases, z means top metal layer, u means  ultra thick metal layer.

1、tcbn16ffcllbwp20p90cpdlvtssgnp0p5vm40c_hm_ccs.db

（1）tcb:tsmc 标准单元库；

（2）n16ffcll：16 nm 工艺节点；

（3）bwp：这种标准元库为 tapless 库；

（4）20：Gate Length；

（5）p90：Poly Pitch；

（6）cpd：用 common poly over od edge 工艺；

（7）lvt：Low V threshold；

（8）ssgnp0p5vm40c：pvt corner；

（9）hm：hold margin，分析 hold timing 的时候带了 margin；

（10）ccs：复合电流源模型；

（11）.db：文件的格式，有 db 格式，也有 lib 格式。

————————————————

# 工艺库的选择

 ls /srv/SMIC55/SCC55NLL_VHS_RVT_V2p1/liberty/1.2v/

scc55nll_vhs_rvt_ff_v1p32_0c_basic.db scc55nll_vhs_rvt_ff_v1p32_-40c_basic.db scc55nll_vhs_rvt_ss_v1p08_-40c_basic.db scc55nll_vhs_rvt_tt_v1p2_25c_basic.db

scc55nll_vhs_rvt_ff_v1p32_0c_basic.lib scc55nll_vhs_rvt_ff_v1p32_-40c_basic.lib scc55nll_vhs_rvt_ss_v1p08_-40c_basic.lib scc55nll_vhs_rvt_tt_v1p2_25c_basic.lib

scc55nll_vhs_rvt_ff_v1p32_0c_ccs.db scc55nll_vhs_rvt_ff_v1p32_-40c_ccs.db scc55nll_vhs_rvt_ss_v1p08_-40c_ccs.db scc55nll_vhs_rvt_tt_v1p2_25c_ccs.db

scc55nll_vhs_rvt_ff_v1p32_0c_ccs.lib scc55nll_vhs_rvt_ff_v1p32_-40c_ccs.lib scc55nll_vhs_rvt_ss_v1p08_-40c_ccs.lib scc55nll_vhs_rvt_tt_v1p2_25c_ccs.lib

scc55nll_vhs_rvt_ff_v1p32_0c_ecsm.db scc55nll_vhs_rvt_ff_v1p32_-40c_ecsm.db scc55nll_vhs_rvt_ss_v1p08_-40c_ecsm.db scc55nll_vhs_rvt_tt_v1p2_25c_ecsm.db

scc55nll_vhs_rvt_ff_v1p32_0c_ecsm.lib scc55nll_vhs_rvt_ff_v1p32_-40c_ecsm.lib scc55nll_vhs_rvt_ss_v1p08_-40c_ecsm.lib scc55nll_vhs_rvt_tt_v1p2_25c_ecsm.lib

scc55nll_vhs_rvt_ff_v1p32_125c_basic.db scc55nll_vhs_rvt_ss_v1p08_125c_basic.db scc55nll_vhs_rvt_tt_v1p2_125c_basic.db scc55nll_vhs_rvt_tt_v1p2_85c_basic.db

scc55nll_vhs_rvt_ff_v1p32_125c_basic.lib scc55nll_vhs_rvt_ss_v1p08_125c_basic.lib scc55nll_vhs_rvt_tt_v1p2_125c_basic.lib scc55nll_vhs_rvt_tt_v1p2_85c_basic.lib

scc55nll_vhs_rvt_ff_v1p32_125c_ccs.db scc55nll_vhs_rvt_ss_v1p08_125c_ccs.db scc55nll_vhs_rvt_tt_v1p2_125c_ccs.db scc55nll_vhs_rvt_tt_v1p2_85c_ccs.db

scc55nll_vhs_rvt_ff_v1p32_125c_ccs.lib scc55nll_vhs_rvt_ss_v1p08_125c_ccs.lib scc55nll_vhs_rvt_tt_v1p2_125c_ccs.lib scc55nll_vhs_rvt_tt_v1p2_85c_ccs.lib

scc55nll_vhs_rvt_ff_v1p32_125c_ecsm.db scc55nll_vhs_rvt_ss_v1p08_125c_ecsm.db scc55nll_vhs_rvt_tt_v1p2_125c_ecsm.db scc55nll_vhs_rvt_tt_v1p2_85c_ecsm.db

scc55nll_vhs_rvt_ff_v1p32_125c_ecsm.lib scc55nll_vhs_rvt_ss_v1p08_125c_ecsm.lib scc55nll_vhs_rvt_tt_v1p2_125c_ecsm.lib scc55nll_vhs_rvt_tt_v1p2_85c_ecsm.lib

  set target_library_ss "scc55nll_vhs_rvt_ss_v1p08_125c_ccs.db"

  set target_library_tt "scc55nll_vhs_rvt_tt_v1p2_25c_ccs.db"

  set target_library_ff "scc55nll_vhs_rvt_ff_v1p32_-40c_ccs.db"

# Dft

# Pad

# Lvs Drc

# Icc

## Lib name

[关与ICC-200809-SP5 "module is not defined" 原因以及解决办法 - 后端讨论区 - EETOP 创芯网论坛 (原名：电子顶级开发网) -](https://bbs.eetop.cn/thread-209153-1-1.html)

# Pad

Pad Limited 与 Core Limited

芯片设计，首先有一个芯片的面积和封装的期望，即便在你 RTL 还没有写完之前。这个期望怎么来？一般就是根据经验来确定。芯片面积和封装很大程度上决定了芯片的成本。但是芯片设计是个流程很长的活动，从前端 RTL coding，到功能验证，到综合，到版图规划，到电源规划，到时钟规划，到布局，到时钟树生成，到布线，等等。不可能等到芯片到了最后阶段布局布线完成，才确定面积是多少，引脚有多少。

为了确定是否能达到这个期望，需要尽快知道当前的设计能不能满足预期。所以，在 RTL coding 完之后，还没做任何功能验证的时候，马上会将第一版 design 用来版图规划，检查是否可以实现芯片规划预期，并及早纠正可能的问题。

什么叫纠正可能的问题？会出现什么问题呢？（1）芯片的 core 面积太小，而芯片需要的管脚太多，导致一个功能很小的芯片却用了非常大的硅片面积（die 的面积），增加了成本。（2）芯片的 core 面积太大，超出了预期的面积规划，成本增加不说，原封装可能塞不下这么大的面积，进一步增加了成本。

第（1）种因为芯片的 pad 太多，超出规划预期的，叫 Pad_Limited；第（2）种因为芯片的 Core 面积太大，超出规划预期的，叫做 Core_Limited。示意图如下，左边是 Pad_Limited，右边是 Core_Limited。简单看起来，Pad_Limited 的芯片，因为 pad 太多，所以 pad 排布十分密集，相对来讲，Core 就很小，**芯片面积由 Pad 决定**。Core_Limited 的芯片，Core 面积相对较大，而 Pad 则相对稀疏，**芯片面积由 Core 决定**。

往往，Pad 太多的时候，需要采用十分细长的 Pad，并且可能会将 Pad 分上下两层交错排列，两层的焊盘部分做适当的重叠，这样可以用同样的 die 面积内塞下更多的 Pad。当 Core 太大的时候，可以采用更加矮胖的 I/O 单元，并且需要在 IO 中间插入很多的填充单元，填充单元的基本作用是把相邻 I/O 单元的电源线连接在一起，也可以加入一些去耦电容和 ESD 保护单元来提高整个芯片的可靠性。

当然了，超出设计预期总不是好事。总的来讲，可以尝试的方法有这些：

★对于 Pad_Limited：

（1）仔细检查有没有不需要的 Pad，将其去除。

（2）仔细想想哪些 Pad 可以复用，将多个 Pad 的功能合并到一个 Pad 上。

（3）修改设计，比如说减小总线位宽，将其频率提高一倍，吞吐量不变，但是可以节省更多 Pad。

（4）如果都不行，那只能增大芯片面积和封装了，考虑到 Core 那么小，你可以给 Core 增加一些空白填充单元，然后把自己的名字写上去。

★对于 Core_Limited:

（1）去掉冗余的逻辑。

（2）去掉不重要的功能。

（3）采用更小的工艺尺寸。

（4）都不行？增大封装得了。
