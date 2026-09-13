---
aliases: 有限域定义
dateCreated: 2024-10-25
dateModified: 2024-11-25
---

<a href=" https://zhuanlan.zhihu.com/p/655497897">一些教材的电子版收集整理</a>

<a href="https://blog.csdn.net/stu_yangpeng/category_11198639.html">一个完整的密码学的教程，包括了有限域</a>

<a href="https://zhuanlan.zhihu.com/p/262267121">知乎有限域计算简述</a>

<a href=" https://f.daixianiu.cn/csdn/5574743532826016.html">介绍了有限元上的各种运算</a>

# 密码学： 有限域
参考：<a href="https://blog.csdn.net/stu_yangpeng/category_11198639.html">一个完整的密码学的教程，包括了有限域</a>
## 代数基本概念
包括群、环、域。


# 有限域
## 域上多项式
基于有限域 $GF (p)$ ​的多项式：
$$
f(x) = a_nx^n + \dots + a_1x+0=\sum_{i=0}^na_ix^i
$$
### 域
第一种：系数运算是模 $p$ 运算的多项式运算，即系数在有限域 $GF (p)$ ​的集合中
在有限域 $GF (p)$ 中的多项式，$a_i \in \mathbb{Z}_p$。加减乘都是域中的模运算，即多项式运算后，系数取模。
- 域中除法

### 不可约多项式
若域 $F$ 中多项式 $f ( x )$ 不能表示域F上任两个多项式 $g_1 ( x ), g_2 (x)$ 的乘积（二者在域 $F$ 中，且次数都小于 $f ( x )$的次数），那么称多项式 $f ( x )$ 为既约多项式，也称不可约多项式、素多项式。

### primitive element
原始元素是某个不可约多项式的根，意味着这个元素是定义有限域的一个特定的多项式的解，并且这个多项式在有限域中不能被分解成更低次数的多项式的乘积。

在有限域 $\mathbb{F}_p$ （其中 $p$ 是一个素数）中，不可约多项式是指不能分解为两个非常数多项式乘积的多项式。如果 $\alpha$ 是 $\mathbb{F}_p$  上的不可约多项式 $G (x)=x^m + g_{m-1}x^{m-1}+\dots+g_1x+x_0$ 的根，那么 $\alpha$ 就是这个多项式的解，即 $f (\alpha)=0$。同时也有：
$$
\alpha^m=g_{m-1}\alpha^{m-1} + \dots + g_1\alpha +g_0
$$
$\alpha$ 可以生成 $GF(2^m)$ 的所有非零元素，所有非零元素都可以用 $\alpha$ 的幂次表示，即 $GF (2^m)=\{1,\alpha^1, \alpha^2,\dots,\alpha^{2^m-2}\}$。

### 举个例子

假设我们有一个素数 \( p = 2 \)，我们想要在 $\mathbb{F}_2$ 上找到一个不可约多项式。考虑多项式 $f (x) = x^2 + x + 1$。

1. **检查不可约性**：
   - 这个多项式在  $\mathbb{F}_2$ 中没有根，因为：
     - $f (0) = 0^2 + 0 + 1 = 1 \neq 0$
     - $f (1) = 1^2 + 1 + 1 = 1 + 1 + 1 = 1 \neq 0$
   - 由于 $f (x)$ 在  $\mathbb{F}_2$ 中没有根，它不能分解为一次多项式的乘积，因此它是不可约的。

2. **构造有限域**：
   - 多项式 $f (x) = x^2 + x + 1$ 定义了 $\mathbb{F}_2$ 的一个扩展域，记作 $\mathbb{F}_4$。这个域有4个元素：$\{0, 1, \alpha, \alpha + 1\}$，其中  $\alpha$ 是 $f (x)$ 的一个根。

3. **原始元素**：
   - 在  $\mathbb{F}_4$ 中，$\alpha$ 是一个原始元素，因为它生成了 $\mathbb{F}_4$ 的所有非零元素。具体来说，$\alpha$ 的幂如下：
     - $\alpha^0 = 1$
     - $\alpha^1 = \alpha$
     - $\alpha^2 = \alpha^2$（根据 $\alpha^2 + \alpha + 1 = 0$，我们得到 $\alpha^2 = \alpha + 1$ ）
     - $\alpha^3 = \alpha \cdot \alpha^2 = \alpha (\alpha + 1) = \alpha^2 + \alpha = (\alpha + 1) + \alpha = 1$
   - 因此，$\alpha$  的幂循环通过 $\{1, \alpha, \alpha + 1\}$，覆盖了  $\mathbb{F}_4$ 的所有非零元素。

通过这个例子，我们可以看到 $\alpha$  作为不可约多项式 $f (x) = x^2 + x + 1$ 的根，确实是有限域 $\mathbb{F}_4$ 的一个原始元素。




## 有限域 $GF (2^n)$

### 多项式的模运算
有限域 $GF(p)$，其上的集合 $\mathbb{Z}_p = \{0, 1, \dots, p-1\}$，定义在 $\mathbb{Z}_p$ 的多项式：
$$
f(x) = a_{n-1}x^{n-1} + \dots + a_1x+0=\sum_{i=0}^{n-1}a_ix^i, \forall i\in \mathbb{Z}_n,a_i\in \mathbb{Z}_p
$$
集合  $S$ 是定义在 $\mathbb{Z}_p$ 上的多项式集合, $S$ 有 $p^n$ 个元素：
$$
S=\{f(x)|f(x)=a_{n-1}x^{n-1} + \dots + a_1x+0=\sum_{i=0}^{n-1}a_ix^i, \forall i\in \mathbb{Z}_n,a_i\in \mathbb{Z}_p\}
$$
 那么下面就定义多项式的模运算：基于普通多项式的加法和乘法运算，并满足以下两条规则：
- 运算中，系数运算以 $p$ 为模数，即遵循有限域 $\mathbb{Z}_p$
- 若乘法运算的结果是次数大于 $n-1$ 的多项式，那么须将其除以某个次数为 $n$ 的既约多项式 $m (x)$ 并取余式。对于多项式 $f ( x )$，这个余数表示为 $r ( x ) = f ( x ) \mod m ( x )$

### B. LSB-First Multiplication Algorithm in Standard Basis
$A (x), B (x)$ 是 $GF (2^m)$ 的两个元素。
$A (x)+B (x)=\sum_{i=0}^{m-1}(a_i+b_i)x^i$ ，其中的加是模二加法。
$G (x)$ 是不可规约多项式，可以生成 $GF (2^m)$。$P (x )=A(x)B(x)\mod G(x)$
$$
\begin{align}
A(x) &= a_{m-1}x^{m-1} + \dots + a_1x+0=\sum_{i=0}^{m-1}a_ix^i\\
B(x) &= b_{m-1}x^{m-1} + \dots + b_1x+0=\sum_{i=0}^{m-1}b_ix^i\\
P(x) &= p_{m-1}x^{m-1} + \dots + p_1x+0=\sum_{i=0}^{m-1}p_ix^i\\
G(x) &= x^{m} + \dots + g_1x+0=x^m+\sum_{i=0}^{m-1}g_ix^i
\end{align}
$$
#### LSB
LSB 优先的算法从 $B (x)$ 的低位开始：
$$
\begin{align}
P(x) &= A(x)B(x)\mod G(x)\\
&= b_0 +\\
&+ b_1[A(x)x\mod G(x)]\\ 
&+ b_2[A(x)x^2\mod G(x)]+\dots\\ 
&+ b_{m-1}[A(x)x^{m-1}\mod G(x)]\\ 
\end{align}
$$
执行 $m$ 次，对每一步骤 $i, 1\le i \le m$，有：
$$
P(x)^{(i)}=A(x)^{(i-1)}b_{i-1}+P(x)^{(i-1)}
$$
其中，$P (x)^{(i)}=\sum_{k=0}^{i-1}A (x) b_kx^k\mod G(x),P(x)^{(0)}=0$，这里 $P (x)$ 就是每一步的累加，而系数：
$$
A(x)^{(i)}=[A(x)^{(i-1)}]x\mod G(x),A(x)^{(0)}=A(x)
$$
$A(x)$ 的计算是一个递归。对其推导，令 $A (x)'=A (x) x\mod G(x)$，带入 $A (x)'=\sum_{i=0}^{m-1}a_{i}'x^{i}$ 的系数和 primitive element有：
$$
a_k'= \left\{ \begin{array}{l}
a_{m-1}g_0,&k=0\\
a_{k-1}+a_{m-1}g_k, &1\le k \le m-1
\end{array} \right.
$$


#### MSB 优先
MSB 优先的算法从 $B (x)$ 的高位开始：
$$
\begin{align}
P(x) &= A(x)B(x)\mod G(x)\\
&= \{\dots [Ab_{m-1}\mod p(x) +Ab_{m-2}]\alpha \mod p(x)+\\
&\dots + Ab_1\}\alpha \mod G(x) + Ab_0
\end{align}
$$
这里，$P (x)$ 也分步骤递归：
$$
W^{(k)}=W^{(k-1)}\alpha \mod p(x) +Ab_{m-k}, 1\le k \le m
$$
$A(x)$ 系数的计算和上面一样有递归。




# 有限域定义

域（Field）的定义是有如下特性的**集合**：

- 定义了加法和乘法
- 集合内的元素经过加法和乘法计算，结果仍然在集合内
- 计算符合交换率、结合率、分配率
- 加法和乘法有单位元素（所有的集合内的值都有对应的负数，所有集合内非零值都有倒数）
如，实数集是域、整数域不是（除了 1，其他数的倒数不是整数）

**具有有限个元素（元素个数称为域的阶）的域就是有限域**（下文以 GF 表示，GF 是 Galois Field 的缩写，这个名字纪念发明者 Evariste Galois)。 

有限个元素，一个关键的操作就是*取模*。也就是在域的定义基础上，作如下修改：

- 定义**模 p 加法**和**模 p 乘法**（加或乘的结果超过 p 时，模 p 取余数。p 为素数）
- 集合内的元素经过加法和乘法计算，结果仍然在集合内
- 计算符合交换率、结合率、分配率
- 加法和乘法有单位元素（所有的集合内的值都有对应的负数，所有集合内非零值都有倒数）
如，对于 $GF (3)$，定义了模 3 加法和乘法，有三个元素 $\{0, 1, 2\}$。如果 p 不是素数，不能满足有限域的定义要求，因为有元素没有倒数。如 $GF (4)$，2 会么有倒数。但如果是 $GF(2^2)$，以 2 为 p 取模，则 4 个元素是有限域。

<table>
    <thead>
        <tr>
            <th>模加 +</th>
            <th>00</th>
            <th>01</th>
            <th>10</th>
            <th>11</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>00</td>
            <td>00</td>
            <td>01</td>
            <td>10</td>
            <td>11</td>
        </tr>
        <tr>
            <td>01</td>
            <td>01</td>
            <td>00</td>
            <td>11</td>
            <td>01</td>
        </tr>
        <tr>
            <td>10</td>
            <td>10</td>
            <td>11</td>
            <td>00</td>
            <td>01</td>
        </tr>
        <tr>
            <td>11</td>
            <td>11</td>
            <td>10</td>
            <td>01</td>
            <td>00</td>
        </tr>
    </tbody>
</table>

<table>
    <thead>
        <tr>
            <th>模乘 *</th>
            <th>00</th>
            <th>01</th>
            <th>10</th>
            <th>11</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>00</td>
            <td>00</td>
            <td>00</td>
            <td>00</td>
            <td>00</td>
        </tr>
        <tr>
            <td>01</td>
            <td>00</td>
            <td>01</td>
            <td>10</td>
            <td>11</td>
        </tr>
        <tr>
            <td>10</td>
            <td>00</td>
            <td>10</td>
            <td>11</td>
            <td>01</td>
        </tr>
        <tr>
            <td>11</td>
            <td>00</td>
            <td>11</td>
            <td>01</td>
            <td>10</td>
        </tr>
    </tbody>
</table>

# 多项式

规定 $GF(2^m)$ 中的二进制数可以表示为多项式的方式，系数只能为 $0, 1$：

$$
11000001 => x^7 + x^6 + 1
$$

## 不可约多项式

$m=8$ 中，有一个特殊的多项式 $m ( x ) = x 8 + x 4 + x 3 + x + 1$，称之为不可约多项式。简单来说，相当于自然数中的质数，也就是它的的因式只有 1 和它本身，所以称为不可约多项式。

## 乘法逆元

有限域中，定义：

$$
b(x)\times b^{-1}(x) \mod a(x) = 1
$$

满足这个条件的 $b^{-1}(x)$ 称为 $b (x)$ 模 $a (x)$ 的乘法逆元，上面条件等价于

$$
a(x)v(x)+b(x)w(x)=1
$$

此处的 $w (x)$ 就是要求的乘法逆元。

## 运算

$GF(2^m)$ 中的多项式加乘法是按位模 2。
