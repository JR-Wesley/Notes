e# PTPX UG阅读
[(7条消息) PrimeTime PX(Power Analysis) userguide阅读笔记_primetime userguide_亓磊的博客-CSDN博客](https://blog.csdn.net/u011729865/article/details/54138703)
功耗分析，用ptpx；
功耗优化，除了设计者从功能结构角度去修改设计外；也可以利用DC power compiler工具去优化功耗（优化策略：包括门控时钟之类的）。

# introduction
PTPX，是基于primetime环境（简称pt），对全芯片进行power静态和动态功耗分析的工具。包括门级的平均功耗和峰值功耗。
可以说PTPX就是pt工具的一个附加工具。个人补充理解：能与pt时序分析脚本运行环境放在一起，可以计算出平均功耗和time-based功耗。

## 用PTPX的好处？
作为前端设计人员来说，个人理解：
1. RTL/pregsim/postgsim三个步骤环节，RTL不支持ptpx功耗分析，netlist才可以采用ptpx分析功耗。
2. 通过功耗的统计数据，可以在设计环节就可以发现功耗的问题，比如某些模块功耗太大；由于数据报告了所有层次结构下的功耗信息，所以很容易找到功耗关键的点。
3. 平均功耗分析，可以找到当前设计的关键功耗点。
4. time-based功耗分析，可以找到瞬态功耗与仿真激励的关系。可以确认峰值功耗是否符合预期要求。
averaged power analysis
简单的说，就是平均功耗。

## time-based power analysis
简单的说，基于时间的功耗分析。所以该分析下，会产生每一个时刻的功耗统计，包括平均功耗、峰值功耗等。
经过ptpx实践，个人理解:
1. 平均功耗，对应的是一个报告。
2. time-based功耗分析，对应的是一个波形，显示每一时刻对应的瞬时功耗。而且报告会额外给出peak-power（包括peak-time时刻）的信息。

ptpx支持下述方式：
- vector-based/vector-free（基于vcd/不用vcd。）
- peak-power和average-power
- RTL/gate-level
- VCD/FSDB/SAIF
- multivoltage和powerDomain

## power modeling overview
![[Pasted image 20230728173258.png]]

图中：
1. lib表示单元库的.lib文件，.lib不仅仅有时序信息，还有area、power信息。
2. vcd指用户通过仿真得到波形文件，里面记载着待分析模块的所有端口和内部线网的跳变信息。从而确定待分析模块的switching功耗。

功耗分类：
- Leakage Power
Leakage Power由标准单元库的cell .db提供相关数据。
- Dynamic Power（包括internal和switching power）
internal power由标准单元库的cell .db提供相关数据，但与激励有关系。
switching power由电压、网表电容、线开关计算得出（我觉得net不能仅仅说是线，而应该是管子源漏栅的连线。电容是管子和线的寄生电容）。与激励有关系。

PS: 静态漏电流功耗和动态功耗的计算结果，用于peak power和average power的分析。

.db文件的power model包括：
- NLPM（nonlinear power model）
- CCS（composite current source）—推荐模型，相对NLPM，更先进更精确。

### 静态漏电流功耗
静态漏电流功耗，是指cell不发生switching开关切换的时候，去计算得出。ptpx，完全使用单元库（.db文件）的漏电流功耗查找表得出结果。
> 静态漏电流功耗 = 本征intrisic漏电流功耗 + 栅极gate漏电流功耗

#### 本征漏电流功耗
主要是源漏之间的电流引起。原因是深亚微米工艺下，阈值电压越来越低导致管子无法完全关闭，使得源漏之间发生导通。
本征漏电流功耗，主要跟电压、管子state有关。

#### gate漏电流功耗
源到栅、栅到漏之间的电流引起。
gate漏电流功耗，主要跟电源电压、栅极氧化层厚度有关。跟温度关系很小。

### 动态功耗
动态功耗与电路的激励有关。
它又分为内部功耗和开关功耗两部分。

#### 内部功耗
内部的概念，是指cell以内。
包括充放电导致的功耗和短路功耗。
充放电的意思是，管子的寄生电容，导致充放电。
短路功耗的意思是，栅极在切换的时候，使得P管、N管同时导通，从而引起电源到地的通路。

简单的库单元，动态功耗主要来自短路功耗。
复杂的库单元，动态功耗主要来自充放电。

#### switching开关功耗
负载电容的充放电过程，导致。具体原因：
1. P管导通，负载电容充电；
2. 因为管子输入端口switching，N管导通，负载电容开始放电。
3. 正是由于上述充放电，导致产生了switching开关功耗。
与内部功耗的子类—充放电功耗，区别是范围不限于cell内部。

## 设置power derating factor
就是一个人为工程经验的比例因子。默认是1.0。
注意：这个power是电源的意思，不是功耗。是电源因为环境不一样，导致的误差。纯属个人理解，未确认。

## 生成power models
生成power models之前，要执行时序分析和功耗分析。
在此基础上，利用extract_model -power命令去产生包含时序和功耗数据的power models。

# PTPX power analysis flow
![[Pasted image 20230728173947.png]]
1. 使能power分析
```tcl
set power_enable_analysis true
```

2. 执行vector分析
因为动态功耗的分析，与仿真激励的vcd文件有很大关系。vcd的activity活跃度越高，功耗越高。为了评估仿真激励vcd文件的activity活跃度，可以用下述命令：
```tcl
write_activity_waveforms
```
有参数的。具体使用办法查看man
吃格式vcd或者fsdb的波形文件，并对该波形文件分析activity活跃度。这个是为了在power分析之前，查看仿真激励的活跃度的。这一步骤，对功耗分析来说，不是必要的。

3. 吃design
支持verilog、vhdl网表；db、ddc、Milkyway格式的网表也可以。
logic库，必须是.db格式。
寄生参数信息，应该在Milkyway格式的文件里。
为了计算平均功耗，需要吃sdc文件
PTPX，支持CCS和NLPM的功耗model。如果design数据里，存在两种的话，可以设置 power_model_preference，以决定选择NLPM还是CCS；默认是nlpm。【个人补充理解：CCS比NLPM模型，更先进;与hspice仿真差别只有2%。推荐用CCS。】

4. 设置变量
set_operating_conditions ；设置工艺及PVT信息。
power_limit_extrapolation_range ； 功耗相关的lookup查找表，有范围。默认功耗分析不限制这个范围。但是如果有很多high-fanout的nets，比如clocks和reset nets；那就得限制范围，从而得到较为准确的功耗数据。
power_use_ccsp_pin_capacitance ；深亚微米，低电压的工艺下；建议设置为true，默认是false。是计算寄生电容参数的。与Miller效应有关（即包含栅到源，栅到漏的电容）。
timing分析
执行命令update_timing ；

检查功耗分析的潜在error
check_power

选择功耗分析模式
set_app_var power_analysis_mode averaged | time_based
1
默认是averaged功耗分析。

# 平均功耗的分析
平均功耗，是基于翻转率toggle rate来分析的。
翻转率的标注，可以是默认翻转率、用户定义switching activity、SAIF文件或者VCD文件。

功耗结果期望准确的话，首先要保证翻转率的标注要准确。这意味着需要后端布局布线、时钟树等已经完全稳定了。前期做功耗分析，可能只是一个评估作用吧。

工具支持基于仿真的switching activity文件类型，包括：
- VCD
- FSDB
- VPD
- SAIF
如果没有上述文件，那可以使用user-define switching activity commands，来提供一个现实的activity去精确power结果。

# time-based功耗分析
与平均功耗类似，只需要设置参数：
```tcl
set_app_var power_analysis_mode time_based
```
# multivoltage功耗分析
# 时钟网络的功耗分析
multivoltage功耗分析和时钟网络的功耗分析，对前端设计人员来说，没太大必要。对后端设计人员来说，应该很重要。
因为前端关心的是数字逻辑功能部分的功耗；时钟网络是后端布局布线才能确认的。

# 报告report命令
```tcl
report_power
report_power_calculation
```
另外，它们有很多命令参数，可以实现各种定制报告。

# 功耗的图形界面
利用PrimeTime的图形界面，去查看功耗的数据报告（包含柱状图等），相对文本报告，显得更直观正式。

## toggle-rate和switching activity区别？
keyword	description
switching activity	开关活跃度，就是管子的开关，导致的switching功耗
toggle-rate	信号翻转率。
翻转率和switching有什么区别？
个人理解，翻转率对应的是信号变化；开关活跃度对应的是管子开关变化。信号变化，不一定会引起管子开关切换的变化。

# 基于PTPX的平均功耗分析
PrimeTime PX支持两种功耗分析模式： averaged mode和time-based mode。在EDA工具的安装目录底下包含了这两种模式的Lab教程和相关设计文件。

本文将一步步地展示如何使用PTPX和这些Lab文件执行功耗分析

**Step1: 首先找到PrimeTime的安装目录，和相应的Lab文件**

```text
which primetime
/opt/Synopsys/PrimeTime2015/bin/primetime
pwd
/opt/Synopsys/PrimeTime2015/doc/pt/tutpx
ls
averaged PrimeTime_PX_Tutorials.pdf sim  src  syn time_based
```

  

可以发现Lab提供了所有设计数据，以及相应的仿真和综合脚本。用户有兴趣可以自行完成设计仿真和综合工作，本文仅展示PTPX功耗分析相关。

**Step2: 设置功耗分析模式**

```text
set power_enable_analysis TRUE
set power_analysis_mode averaged
```

**Step3: read&link设计**

```text
set search_path "../src/hdl/gate ../src/lib/snps . "
set link_library " * core_typ.db"
read_verilog mac.vg
current_design mac
link
```

完成netlist（[mac.vg](https://link.zhihu.com/?target=http%3A//mac.vg)）和工艺库（core_typ.db）之间的link工作。netlist中描述了大量的std cell的例化，工艺库中建模了各个std cell的internal power和leakage power

**Step4: 读sdc，反标寄生参数**

```text
read_sdc ../src/hdl/gate/mac.sdc
set_disable_timing [get_lib_pins ssc_core_typ/*/G]
read_parasitics ../src/annotate/mac.spef.gz
```

sdc指定了设计的驱动单元，用以计算输入的transitiontime。寄生参数是影响动态功耗的因素之一，反标寄生参数文件能够提高功耗分析的准确性。

**Step5: check timing, update timing和report timing**

```text
check_timing
update_timing
report_timing
```

在之前的文章提到过，在改变设计约束时，需要check timing，设计需求的准确描述很重要。时序违例，功耗分析也没有意义。

**Step6: 读入开关活动文件**

```text
read_vcd -strip_path tb/macinst ../sim/vcd.dump.gz
report_switching_activity -list_not_annotated
```

设计相关环境和输入描述地越多，功耗分析越准确。开关活动文件可以以vcd或者saif的格式。如果不指定开关活动文件，ptpx就会采用默认的开关活动行为，降低功耗分析的准确性。

**Step7: 执行功耗分析**

```text
check_power
update_power
report_power -hierarchy
quit
```

  

下面是读入不同开关活动文件进行的功耗分析：

**读入saif文件**

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

**读入vcd文件**

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

  

我们可以发现，以saif文件功耗为基准，各个类型的**功耗差异**分别为：

```text
                      saif文件       vcd文件    不读入开关活动文件          
 internal power    2.10e-03       2.229e-03(6.1%)    1.42e-03(32.4%)
 switch power      1.55e-03       1.549e-03(0.06%)   6.57e-04(57.6%)
 dynamic power     3.65e-03       3.778e-03(3.5%)    2.08e-03(43.0%)
 leakage power     2.59e-07       2.594e-07(0.15%)    2.59e-07(0%)
```

  

所以，如果你不读入任何开关活动文件进行功耗分析，你可能需要接受非常大的动态功耗误差！

# IC设计中的功耗分析流程
[(7条消息) IC设计中的功耗分析的流程_synopsys综合时怎么看电路的功耗_mikiah的博客-CSDN博客](https://blog.csdn.net/mikiah/article/details/8061532)
首先声明本文所讲的范围，在这篇文章中，是采用synopsys的设计流程，对数字电路进行功耗分析，生成功耗分析报告的流程。分析的对象是逻辑综合之后布局布线之前的功耗分析，以及布局布线之后的功耗分析。

    Synopsys做功耗分析使用到的工具是：Primetime PX, PrimeRail。PTPX可以在逻辑综合之后就进行功耗预估。PrimeTimePX是集成在PrimeTime里面的工具，虽然他可以做功耗分析，但是毕竟不是sign-off工具。真正到最后的sign-off,如果对功耗的要求很高的话，依然要用PrimeRail进行分析，所以，我们只要用到PrimeTime PX来做功耗分析就够了。

  ![](http://t3.qpic.cn/mblogpic/165edbd88567c07b8ae4/2000)   

上图是布局布线后和逻辑综合后进行功耗分析的流程。

一. 逻辑综合后的功耗分析

  所用到的文件有：1. 逻辑综合后的verilog文件

                  2.静态时序分析时用到的约束文件

                  3.RTL的仿真文件，我用的是VCD，毕竟标准各个仿真器都支持~

                  4.有功耗信息的库文件.db，这个库文件可以report一个库里的cell，看是否有。

    有了这些文件之后，就可以做功耗分析了。下面说一下功耗分析的流程：

1. 允许功耗分析功能 set power_enable_analysis

2. 设置分析模式     setpower_analysis_mode。他的模式有两种，一种是average模式，不用仿真文件，另一种是time-based模式，是根据时序仿真文件来确定activityfactor。

3. 读入设计和库文件 

4. 指定operating condition

5. 时序分析   update_timing

6. 获取activity data

如果是RTL级别的网表文件，要用-rtl来告诉pt之前指定的vcd file是布局布线之前的。如果VCD是zero_delay的仿真，也就是说是纯纯的functional simulation的话，应该家用-zero_delay选项。如果都为指定，pt默认是gate-level。

7. 设置功耗分析选项 set_power_analysis_options ：

-static_leakage_only option of the set_power_analysis_optionscommand is supported only in           theaveraged power analysis mode.

        -waveform_interval, -cycle_accurate_cycle_count,-cycle_accurate_clock,-waveform_format, -           waveform_output, -include, and -include_groupsoptions are  supported only in the time-            based poweranalysis mode.

8. 功耗分析   update_power

9. 生成功耗分析报告 report_power

要说明的是，PTPX是primetime的一个增强功能，只用一个PT脚本就可以了，我把自己的pt脚本拿出来分享一下:

  

file: pt.tcl

  

###########################################

#   Set the power analysis mode

###########################################

set power_enable_analysis true;

set power_analysis_mode averaged;

###########################################

#     read and link the gatelevel netlist

###########################################

set search_path "../source db ./ ./result"

set link_library "typical.db"

set target_library "typical.db"

read_verilog jnd_90s.v

set top_name jnd

current_design JND

link

###########################################

#   Read SDC and set transition time orannotate parasitics

###########################################

read_sdc pt_con.tcl

  

###########################################

#   Check, update, or report timing

###########################################

check_timing

update_timing

report_timing

  

  

###########################################

#   read switching activity file

###########################################

read_vcd -rtl jnd_all.vcd -strip_path testbench

report_switching_activity -list_not_annotated

  

###########################################

#   check or update or report power

###########################################

check_power

update_power

report_power -hierarchy

  

二. 布局布线后的功耗分析

现在ptpx也支持多电压域的功耗分析，并提供了一个范例脚本，这里描述多时钟域时要用到UPF，叫unified powerformat。这里不介绍。

# Read libraries, design, enable power analysis

# and link design

set power_enable_analysis true

set link_library slow_pgpin.db

read_verilog power_pins.v

link

  

# Create back-up power nets

create_power_net_info vdd_backup -power

create_power_net_info vss_backup -gnd

# Create domain power nets

create_power_net_info t_vdd -power -switchable \

-nominal_voltages{1.2} -voltage_ranges{1.1 1.3}

create_power_net_info a_vdd -power

create_power_net_info b_vdd -power

# Create domain ground nets

create_power_net_info t_vss -gnd

create_power_net_info a_vss -gnd

create_power_net_info b_vss -gnd

# Create internal power nets

create_power_net_info int_vdd_1 -power \

-nominal_voltages{1.2} -voltage_ranges[1.1 1.3} \

-switchable

create_power_net_info int_vdd_2 -power \

-nominal_voltages{1.25} -voltage_ranges{1.1 1.3}

create_power_net_info int_vdd_3 -power \

-nominal_voltages{1.2} -voltage_ranges{1.1 1.3}

create_power_net_info int_vdd_4 -power

# Create power domains

create_power_domain t

create_power_domain a -object_list[get_cells PD0_inst]\

-power_down -power_down_ctrl[get_nets a] \

-power_down_ctrl_sense 0

create_power_domain b -object_list [get_cells PD1_inst]\

-power_down

# Connect rails to power domains

connect_power_domain t -primary_power_net t_vdd \

-primary_ground_net t_vss

connect_power_domain a -primary_power_net a_vdd \

-primary_ground_net a_vss \

-backup_power_net vdd_backup \

-backup_ground_net vss_backup

connect_power_domain b -primary_power_net b_vdd \

-primary_ground_net b_vss

  

# Set voltages of power nets

set_voltage 1.15 -object_list{t_vdd a_vdd b_vdd}

# Read SDC and other timing or power assertions

set_input_transition 0.0395 [all_inputs]

set_load 1.0 [all outputs]

# Perform timing analysis

update_timing

# Read switching activity

set_switching_activity...

set_switching_activity...

...

report_power

  

  

三. 关于报告 

一个标准的报告：

Power Group Power Power Power Power ( %) Attrs

---------------------------------------------------------------

io_pad 0.0000 0.0000 0.0000 0.0000 ( 0.00%)

memory 0.0000 0.0000 0.0000 0.0000 ( 0.00%)

black_box 0.0000 0.0000 0.0000 0.0000 ( 0.00%)

clock_network 0.0000 0.0000 0.0000 0.0000 ( 0.00%)

register 8.442e-05 1.114e-05 9.208e-09 9.557e-05 (29.97%)i

combinational 0.0000 0.0000 0.0000 0.0000 ( 0.00%)

sequential 0.0000 0.0000 0.0000 0.0000 ( 0.00%)

Attributes

----------

i - Including driven register power

Internal Switching Leakage Total Clock

Power Power Power Power ( %) Attrs

---------------------------------------------------------------

clk 1.813e-04 4.199e-05 4.129e-10 2.233e-04

---------------------------------------------------------------

Estimated Clock1.813e-04 4.199e-054.129e-102.233e-04(70.03%)

Net Switching Power = 5.313e-05 (16.66%)

Cell Internal Power = 2.657e-04 (83.33%)

Cell Leakage Power = 9.627e-09 ( 0.00%)

---------

Total Power = 3.188e-04 (100.00%)

  

关于门控时钟的报告：

report_clock_gate_savings

****************************************

Report : Clock Gate Savings

power_mode: Averaged

Design : mydesign

Version: D-2009.12

Date : Thu Oct 29 12:08:20 2009

****************************************

------------------------------------------------------------------

Clock: clk

+ Clock Toggle Rate: 0.392157

+ Number of Registers: 19262

+ Number of Clock Gates: 12

+ Average Clock Toggle Rate at Registers: 0.305872

+ Average Toggle Savings at Registers: 22.0%

------------------------------------------------------------------

Toggle Savings Number of % of

Distribution Registers Registers

------------------------------------------------------------------

100% 0 0.0%

80% - 100% 76 0.4%

60% - 80% 5660 29.4%

40% - 60% 0 0.0%

20% - 40% 8 0.0%

0% - 20% 0 0.0%

0% 13518 70.2%

------------------------------------------------------------------