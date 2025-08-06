---
dateCreated: 2025-04-04
dateModified: 2025-04-04
---
# 浮点单元设计

计算机中数值的表示方式分为两种：**定点数和浮点数**。定点数小数位置*固定*，硬件实现简单，但是数值表示范围小，运算精度较低。浮点数小数点位置*浮动*，可以表示的数值动态范围大，运算精度高，但是浮点运算单元 (Floating-Point Unit FPU) 的面积功耗大很多。

一个通用的 FPU 通常要包括的运算有：加法、减法、除法、开方等。经统计，不同运算的使用占比为，加减法运算占比 55%，乘法占比 37%，开方和除法分别为 1.3% 和 2%，其他比较、定点浮点转换运算占比不高。

## IEEE 754 浮点数据格式和运算标准

下面是浮点数通用格式，最高位为 1 位的符号位，接下来是加了偏置的指数位，最后是位数位。其中 $T_0$ 为尾数的隐含位，对于非规约类型的浮点数，$T_0=0$，对于规约类型的浮点数，$T_0=1$，偏置 bias 大小为 $2^{w-1}-1$。表达的二进制如下：

$$
d = (-1)^s \times T_0T_1...T_{p-1}\times 2^{E-bias}
$$

### 格式

IEEE 754-2008 定义了多种二进制浮点数。

<table>
<caption>浮点数格式</caption>
  <thead>
    <tr>
      <th>位宽</th>
      <th>符号</th>
      <th>指数</th>
      <th>尾数</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td></td>
      <td>S 1-bit</td>
      <td>E w-bit</td>
      <td>T p-bit</td>
    </tr>
    <tr>
      <td>binary16</td>
      <td>1</td>
      <td>5</td>
      <td>10</td>
    </tr>
    <tr>
      <td>binary32</td>
      <td>1</td>
      <td>8</td>
      <td>23</td>
    </tr>
    <tr>
    <tr>
      <td>binary64</td>
      <td>1</td>
      <td>11</td>
      <td>52</td>
    </tr>
    <tr>
      <td>binary128</td>
      <td>1</td>
      <td>15</td>
      <td>112</td>
    </tr>
    <tr>
      <td>binary{k}(k>=128)</td>
      <td>1</td>
      <td>round[4*log_2(k)]-13</td>
      <td>k-E+25</td>
    </tr>
  </tbody>
</table>

指数位和位数位的不同数值组合，可以表示不同类型的浮点数，在 IEEE754 标准下，浮点数分为几类，定义如下：

<table>
<caption>浮点数分类和含义</caption>
  <thead>
    <tr>
      <th>类型</th>
      <th>指数位</th>
      <th>尾数位</th>
      <th>隐含位T0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>零</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>规约数</td>
      <td>非 0非 max</td>
      <td>*</td>
      <td>1</td>
    </tr>
    <tr>
      <td>非规约数</td>
      <td>0</td>
      <td>非 0</td>
      <td>0</td>
    </tr>
    <tr>
    <tr>
      <td>无穷数 INF</td>
      <td>max</td>
      <td>0</td>
      <td>-</td>
    </tr>
    <tr>
      <td>非数 NaN</td>
      <td>max</td>
      <td>非 0</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

- **零**：指数位和位数位都为 0，此时符号位决定正负，有正零和负零之分。
- **规约数**：指数为不全为 1 也不全为 0，隐含位有效为 1，分正负。单精度 32 位，偏移的指数位 E 取值范围 $[1, 254]$，双精度 63 位，范围 $[1,2046]$。
- **非规约数**：指数全 0，尾数不全 0，隐含位有效，分正负。注意，以单精度为例，此时指数并非直接计算得来的 0-127=-126，而是 -126，保持和规约数统一。
- **无穷数**：指数位全 1，位数位全 0，分正负。
- **NaN**：指数位全 1，位数位不全位为 0，此时根据位数位首位是否为 1，NaN 还可以分为 SNaN 和 QNaN，前者参与运算时会发生异常。

### 舍入

IEEE 754 定义了几种**舍入模式**，需要强制实现的有四种。首先介绍舍入过程中的几个名词。舍入过程是丢弃舍弃位，根据舍弃位各位数值以及保留位最低位数值，确定是向上边界舍入还是向下边界舍入。
- Guard bit (G) 保留数值的最高位
- Round bit (R) 舍弃数据的最高位
- Sticky bit (STK) 舍弃数据次高位到最低位按位（逻辑）或的结果
<table>
  <caption>舍入示意图</caption>
<thead>
<tr>
	<th>保留位 </th>
	<th>舍弃位</th>
</tr>
</thead>
<tbody>
<tr>
	<td>DDDDG</td>
	<td>RXXX</td>
</tr>
</tbody>
</table>

舍入模式的计算如下，其中 $F 1/ F 2$ 分别代表可表示的浮点数下边界和上边界，$F 1<F 2$，sign 为符号位。

<table>
<caption>四种舍入模式</caption>
  <thead>
    <tr>
      <th>舍入模式</th>
      <th>描述</th>
      <th>计算公式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>roundTiesToEven</td>
      <td>向最近的可表示浮点数舍入，如果到上下边界距离相等，则向尾数为偶数的边界舍入（LSB=0）</td>
      <td>R&(G|STK)</td>
    </tr>
    <tr>
      <td>roundTowardPositive</td>
      <td>总是向 F 2 舍入</td>
      <td>~sign&(R|STK)</td>
    </tr>
    <tr>
      <td>roundTowardNegative</td>
      <td>总是向 F 1 舍入</td>
      <td>sign&(R|STK)</td>
    </tr>
    <tr>
    <tr>
      <td>roundTowardZero</td>
      <td>当结果为正数时向 F 1 舍入，否则向 F 2 舍入</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

### 异常

IEEE 754 标准规定，针对基本的五种算术运算，当异常发生时，必须给出相关异常指示信号，分为以下五种异常：

- **Invalid**: 无效。发生无效的情况是有操作数之一为 NaN，相同符号无穷大相减，不同符号无穷大相加，无穷大相除，0 乘无穷大，0 除以 0，被开方数是负数。运算结果值为 NaN。
- **DivisionByZero**。除法运算，除数为 0 时，结果为无穷大。
- **Overflow**。中间结果比浮点数可以表示的最大值还大，结果根据舍入模式置为无穷大或能表达的最大规约数。具体分类如下。
- **Underflow**。出现了极小 tininess 且发生了精度损失 loss of accuracy。运算结果绝对值小于最小规约数或舍入后绝对值仍小于最小规约数。
- **Inexact**。舍入时最低为后不全为 0，结果是一个近似值。


<table>
<caption>不同舍入overflow结果</caption>
  <thead>
    <tr>
      <th>roundTiesToEven</th>
      <th>roundTowardPositive</th>
      <th>roundTowardNegative</th>
      <th>roundTowardZero</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>sign is +</td>
      <td>正无穷</td>
      <td>最大规约数</td>
      <td>正无穷</td>
      <td>最大规约数</td>
    </tr>
    <tr>
      <td>sign is -</td>
      <td>负无穷</td>
      <td>最小规约数</td>
      <td>最小规约数</td>
      <td>负无穷</td>
    </tr>
  </tbody>
</table>

## 浮点加法运算原理和设计


## 浮点乘法运算原理和设计

## 浮点除法/开方运算原理和设计
