---
dateCreated: 2023-07-28
dateModified: 2025-05-25
---
https://www.cnblogs.com/gujiangtaoFuture/articles/10170601.html

 PTPX UG 阅读

[(7条消息) PrimeTime PX(Power Analysis) userguide阅读笔记_primetime userguide_亓磊的博客-CSDN博客](https://blog.csdn.net/u011729865/article/details/54138703)

功耗分析，用 ptpx；

功耗优化，除了设计者从功能结构角度去修改设计外；也可以利用 DC power compiler 工具去优化功耗（优化策略：包括门控时钟之类的）。

# Introduction

PTPX，是基于 primetime 环境（简称 pt），对全芯片进行 power 静态和动态功耗分析的工具。包括门级的平均功耗和峰值功耗。

可以说 PTPX 就是 pt 工具的一个附加工具。个人补充理解：能与 pt 时序分析脚本运行环境放在一起，可以计算出平均功耗和 time-based 功耗。

## 用 PTPX 的好处？

作为前端设计人员来说，个人理解：

1. RTL/pregsim/postgsim 三个步骤环节，RTL 不支持 ptpx 功耗分析，netlist 才可以采用 ptpx 分析功耗。
2. 通过功耗的统计数据，可以在设计环节就可以发现功耗的问题，比如某些模块功耗太大；由于数据报告了所有层次结构下的功耗信息，所以很容易找到功耗关键的点。
3. 平均功耗分析，可以找到当前设计的关键功耗点。
4. time-based 功耗分析，可以找到瞬态功耗与仿真激励的关系。可以确认峰值功耗是否符合预期要求。
averaged power analysis
简单的说，就是平均功耗。

## Time-based Power Analysis

简单的说，基于时间的功耗分析。所以该分析下，会产生每一个时刻的功耗统计，包括平均功耗、峰值功耗等。

经过 ptpx 实践，个人理解:

1. 平均功耗，对应的是一个报告。
2. time-based 功耗分析，对应的是一个波形，显示每一时刻对应的瞬时功耗。而且报告会额外给出 peak-power（包括 peak-time 时刻）的信息。

ptpx 支持下述方式：

- vector-based/vector-free（基于 vcd/不用 vcd。）
- peak-power 和 average-power
- RTL/gate-level
- VCD/FSDB/SAIF
- multivoltage 和 powerDomain

## Power Modeling Overview

![[Pasted image 20230728173258.png]]

图中：

1. lib 表示单元库的.lib 文件，.lib 不仅仅有时序信息，还有 area、power 信息。
2. vcd 指用户通过仿真得到波形文件，里面记载着待分析模块的所有端口和内部线网的跳变信息。从而确定待分析模块的 switching 功耗。

功耗分类：

- Leakage Power
Leakage Power 由标准单元库的 cell .db 提供相关数据。
- Dynamic Power（包括 internal 和 switching power）
internal power 由标准单元库的 cell .db 提供相关数据，但与激励有关系。
switching power 由电压、网表电容、线开关计算得出（我觉得 net 不能仅仅说是线，而应该是管子源漏栅的连线。电容是管子和线的寄生电容）。与激励有关系。

PS: 静态漏电流功耗和动态功耗的计算结果，用于 peak power 和 average power 的分析。

.db 文件的 power model 包括：

- NLPM（nonlinear power model）
- CCS（composite current source）—推荐模型，相对 NLPM，更先进更精确。

### 静态漏电流功耗

静态漏电流功耗，是指 cell 不发生 switching 开关切换的时候，去计算得出。ptpx，完全使用单元库（.db 文件）的漏电流功耗查找表得出结果。

> 静态漏电流功耗 = 本征 intrisic 漏电流功耗 + 栅极 gate 漏电流功耗

#### 本征漏电流功耗

主要是源漏之间的电流引起。原因是深亚微米工艺下，阈值电压越来越低导致管子无法完全关闭，使得源漏之间发生导通。

本征漏电流功耗，主要跟电压、管子 state 有关。

#### Gate 漏电流功耗

源到栅、栅到漏之间的电流引起。

gate 漏电流功耗，主要跟电源电压、栅极氧化层厚度有关。跟温度关系很小。

### 动态功耗

动态功耗与电路的激励有关。

它又分为内部功耗和开关功耗两部分。

#### 内部功耗

内部的概念，是指 cell 以内。

包括充放电导致的功耗和短路功耗。

充放电的意思是，管子的寄生电容，导致充放电。

短路功耗的意思是，栅极在切换的时候，使得 P 管、N 管同时导通，从而引起电源到地的通路。

简单的库单元，动态功耗主要来自短路功耗。

复杂的库单元，动态功耗主要来自充放电。

#### Switching 开关功耗

负载电容的充放电过程，导致。具体原因：

1. P 管导通，负载电容充电；
2. 因为管子输入端口 switching，N 管导通，负载电容开始放电。
3. 正是由于上述充放电，导致产生了 switching 开关功耗。
与内部功耗的子类—充放电功耗，区别是范围不限于 cell 内部。

## 设置 Power Derating Factor

就是一个人为工程经验的比例因子。默认是 1.0。

注意：这个 power 是电源的意思，不是功耗。是电源因为环境不一样，导致的误差。纯属个人理解，未确认。

## 生成 Power Models

生成 power models 之前，要执行时序分析和功耗分析。

在此基础上，利用 extract_model -power 命令去产生包含时序和功耗数据的 power models。

# PTPX Power Analysis Flow

![[Pasted image 20230728173947.png]]

1. 使能 power 分析

```tcl
set power_enable_analysis true
```

2. 执行 vector 分析
因为动态功耗的分析，与仿真激励的 vcd 文件有很大关系。vcd 的 activity 活跃度越高，功耗越高。为了评估仿真激励 vcd 文件的 activity 活跃度，可以用下述命令：

```tcl
write_activity_waveforms
```

有参数的。具体使用办法查看 man

吃格式 vcd 或者 fsdb 的波形文件，并对该波形文件分析 activity 活跃度。这个是为了在 power 分析之前，查看仿真激励的活跃度的。这一步骤，对功耗分析来说，不是必要的。

3. 吃 design
支持 verilog、vhdl 网表；db、ddc、Milkyway 格式的网表也可以。
logic 库，必须是.db 格式。
寄生参数信息，应该在 Milkyway 格式的文件里。
为了计算平均功耗，需要吃 sdc 文件
PTPX，支持 CCS 和 NLPM 的功耗 model。如果 design 数据里，存在两种的话，可以设置 power_model_preference，以决定选择 NLPM 还是 CCS；默认是 nlpm。【个人补充理解：CCS 比 NLPM 模型，更先进; 与 hspice 仿真差别只有 2%。推荐用 CCS。】

4. 设置变量
set_operating_conditions；设置工艺及 PVT 信息。
power_limit_extrapolation_range；功耗相关的 lookup 查找表，有范围。默认功耗分析不限制这个范围。但是如果有很多 high-fanout 的 nets，比如 clocks 和 reset nets；那就得限制范围，从而得到较为准确的功耗数据。
power_use_ccsp_pin_capacitance；深亚微米，低电压的工艺下；建议设置为 true，默认是 false。是计算寄生电容参数的。与 Miller 效应有关（即包含栅到源，栅到漏的电容）。
timing 分析
执行命令 update_timing；

检查功耗分析的潜在 error

check_power

选择功耗分析模式

set_app_var power_analysis_mode averaged | time_based

1

默认是 averaged 功耗分析。

# 平均功耗的分析

平均功耗，是基于翻转率 toggle rate 来分析的。

翻转率的标注，可以是默认翻转率、用户定义 switching activity、SAIF 文件或者 VCD 文件。

功耗结果期望准确的话，首先要保证翻转率的标注要准确。这意味着需要后端布局布线、时钟树等已经完全稳定了。前期做功耗分析，可能只是一个评估作用吧。

工具支持基于仿真的 switching activity 文件类型，包括：

- VCD
- FSDB
- VPD
- SAIF
如果没有上述文件，那可以使用 user-define switching activity commands，来提供一个现实的 activity 去精确 power 结果。

# Time-based 功耗分析

与平均功耗类似，只需要设置参数：

```tcl
set_app_var power_analysis_mode time_based
```

# Multivoltage 功耗分析
# 时钟网络的功耗分析

multivoltage 功耗分析和时钟网络的功耗分析，对前端设计人员来说，没太大必要。对后端设计人员来说，应该很重要。

因为前端关心的是数字逻辑功能部分的功耗；时钟网络是后端布局布线才能确认的。

# 报告 Report 命令

```tcl
report_power
report_power_calculation
```

另外，它们有很多命令参数，可以实现各种定制报告。

# 功耗的图形界面

利用 PrimeTime 的图形界面，去查看功耗的数据报告（包含柱状图等），相对文本报告，显得更直观正式。

## Toggle-rate 和 Switching Activity 区别？

keyword	description

switching activity	开关活跃度，就是管子的开关，导致的 switching 功耗

toggle-rate	信号翻转率。

翻转率和 switching 有什么区别？

个人理解，翻转率对应的是信号变化；开关活跃度对应的是管子开关变化。信号变化，不一定会引起管子开关切换的变化。

# 基于 PTPX 的平均功耗分析

PrimeTime PX 支持两种功耗分析模式：averaged mode 和 time-based mode。在 EDA 工具的安装目录底下包含了这两种模式的 Lab 教程和相关设计文件。

本文将一步步地展示如何使用 PTPX 和这些 Lab 文件执行功耗分析

**Step1: 首先找到 PrimeTime 的安装目录，和相应的 Lab 文件**

```text
which primetime
/opt/Synopsys/PrimeTime2015/bin/primetime
pwd
/opt/Synopsys/PrimeTime2015/doc/pt/tutpx
ls
averaged PrimeTime_PX_Tutorials.pdf sim  src  syn time_based
```

可以发现 Lab 提供了所有设计数据，以及相应的仿真和综合脚本。用户有兴趣可以自行完成设计仿真和综合工作，本文仅展示 PTPX 功耗分析相关。

**Step2: 设置功耗分析模式**

```text
set power_enable_analysis TRUE
set power_analysis_mode averaged
```

**Step3: read&link 设计**

```text
set search_path "../src/hdl/gate ../src/lib/snps . "
set link_library " * core_typ.db"
read_verilog mac.vg
current_design mac
link
```

完成 netlist（[mac.vg](https://link.zhihu.com/?target=http%3A//mac.vg)）和工艺库（core_typ.db）之间的 link 工作。netlist 中描述了大量的 std cell 的例化，工艺库中建模了各个 std cell 的 internal power 和 leakage power

**Step4: 读 sdc，反标寄生参数**

```text
read_sdc ../src/hdl/gate/mac.sdc
set_disable_timing [get_lib_pins ssc_core_typ/*/G]
read_parasitics ../src/annotate/mac.spef.gz
```

sdc 指定了设计的驱动单元，用以计算输入的 transitiontime。寄生参数是影响动态功耗的因素之一，反标寄生参数文件能够提高功耗分析的准确性。

**Step5: check timing, update timing 和 report timing**

```text
check_timing
update_timing
report_timing
```

在之前的文章提到过，在改变设计约束时，需要 check timing，设计需求的准确描述很重要。时序违例，功耗分析也没有意义。

**Step6: 读入开关活动文件**

```text
read_vcd -strip_path tb/macinst ../sim/vcd.dump.gz
report_switching_activity -list_not_annotated
```

设计相关环境和输入描述地越多，功耗分析越准确。开关活动文件可以以 vcd 或者 saif 的格式。如果不指定开关活动文件，ptpx 就会采用默认的开关活动行为，降低功耗分析的准确性。

**Step7: 执行功耗分析**

```text
check_power
update_power
report_power -hierarchy
quit
```

下面是读入不同开关活动文件进行的功耗分析：

**读入 saif 文件**

```text
read_saif "../sim/mac.saif"-strip_path "tb/macinst"
```

**功耗报告**

```text
Int      Switch  Leak      Total
Power    Power   Power     Power    %
-----------------------------------------------
2.10e-03 1.55e-03 2.59e-07  3.65e-03  100.0
```

**读入 vcd 文件**

```text
read_vcd "../sim/vcd.dump.gz"-strip_path "tb/macinst"
```

**功耗分析**

```text
 Net Switching Power  = 1.549e-03   (41.00%)
 Cell Internal Power  = 2.229e-03   (58.99%)
 Cell Leakage Power   = 2.594e-07   ( 0.01%)
                         ---------
Total Power            = 3.778e-03  (100.00%)
```

**不读入开关活动文件**

**功耗分析**

```text
Int      Switch     Leak        Total
Power    Power      Power       Power    %
--------------------------------------------------------------------------------
1.42e-03  6.57e-04  2.59e-07   2.08e-03  100.0
```

我们可以发现，以 saif 文件功耗为基准，各个类型的**功耗差异**分别为：

```text
                      saif文件       vcd文件    不读入开关活动文件          
 internal power    2.10e-03       2.229e-03(6.1%)    1.42e-03(32.4%)
 switch power      1.55e-03       1.549e-03(0.06%)   6.57e-04(57.6%)
 dynamic power     3.65e-03       3.778e-03(3.5%)    2.08e-03(43.0%)
 leakage power     2.59e-07       2.594e-07(0.15%)    2.59e-07(0%)
```

所以，如果你不读入任何开关活动文件进行功耗分析，你可能需要接受非常大的动态功耗误差！

# IC 设计中的功耗分析流程

[(7条消息) IC设计中的功耗分析的流程_synopsys综合时怎么看电路的功耗_mikiah的博客-CSDN博客](https://blog.csdn.net/mikiah/article/details/8061532)

首先声明本文所讲的范围，在这篇文章中，是采用 synopsys 的设计流程，对数字电路进行功耗分析，生成功耗分析报告的流程。分析的对象是逻辑综合之后布局布线之前的功耗分析，以及布局布线之后的功耗分析。

  Synopsys 做功耗分析使用到的工具是：Primetime PX, PrimeRail。PTPX 可以在逻辑综合之后就进行功耗预估。PrimeTimePX 是集成在 PrimeTime 里面的工具，虽然他可以做功耗分析，但是毕竟不是 sign-off 工具。真正到最后的 sign-off,如果对功耗的要求很高的话，依然要用 PrimeRail 进行分析，所以，我们只要用到 PrimeTime PX 来做功耗分析就够了。

上图是布局布线后和逻辑综合后进行功耗分析的流程。

一. 逻辑综合后的功耗分析

  所用到的文件有：1. 逻辑综合后的 verilog 文件

                 2.静态时序分析时用到的约束文件

                  3.RTL 的仿真文件，我用的是 VCD，毕竟标准各个仿真器都支持~

                  4.有功耗信息的库文件.db，这个库文件可以 report 一个库里的 cell，看是否有。

    有了这些文件之后，就可以做功耗分析了。下面说一下功耗分析的流程：

1. 允许功耗分析功能 set power_enable_analysis
2. 设置分析模式     setpower_analysis_mode。他的模式有两种，一种是 average 模式，不用仿真文件，另一种是 time-based 模式，是根据时序仿真文件来确定 activityfactor。
3. 读入设计和库文件 
4. 指定 operating condition
5. 时序分析   update_timing
6. 获取 activity data

如果是 RTL 级别的网表文件，要用 -rtl 来告诉 pt 之前指定的 vcd file 是布局布线之前的。如果 VCD 是 zero_delay 的仿真，也就是说是纯纯的 functional simulation 的话，应该家用 -zero_delay 选项。如果都为指定，pt 默认是 gate-level。

1. 设置功耗分析选项 set_power_analysis_options：

-static_leakage_only option of the set_power_analysis_optionscommand is supported only in           theaveraged power analysis mode.

        -waveform_interval, -cycle_accurate_cycle_count,-cycle_accurate_clock,-waveform_format, -           waveform_output, -include, and -include_groupsoptions are  supported only in the time-            based poweranalysis mode.

1. 功耗分析   update_power
2. 生成功耗分析报告 report_power

要说明的是，PTPX 是 primetime 的一个增强功能，只用一个 PT 脚本就可以了，我把自己的 pt 脚本拿出来分享一下:
