
# C++ 实现
在算法的具体实现过程中，需要使用到C++的NTL库和GMP库，所以首先要安装这两个库。

- GMP（GNU Multiple Precision Arithmetic Library）是一款用于**高精度计算**的C/C++库，可以**支持任意长度的整数运算、浮点数运算等操作**。
- NTL（Number Theory Library）是一款用于**高效实现数论算法**的C++库，可以**支持任意精度整数、多项式、矩阵等数据类型的操作，包括整数分解、离散对数、RSA公钥加密、椭圆曲线密码等常见数论算法**。NTL库的作者是Victor Shoup，目前最新版本为11.5.2，开源免费。

和GMP库相比，NTL库更专注于数论算法的实现，提供了一些高效的数论算法和数据结构，例如NTL提供了快速傅里叶变换（FFT）实现多项式乘法、大数NTT、CRT等。同时，NTL库还提供了一些常见的密码学算法的实现，例如RSA、椭圆曲线密码等。由于NTL库的高效性和专注性，NTL库在一些需要高效实现数论算法的应用中得到了广泛的运用，例如密码学、编码理论等领域。与GMP库相比，NTL库在支持高精度整数计算的同时，还提供了更多的数论算法和数据结构，因此可以看做是GMP库的一个补充和扩展。需要注意的是，由于NTL库的特殊性质，NTL库并不支持任意精度浮点数的计算，因此在需要进行浮点数运算时，还需要结合其他库（例如GMP库）来完成。


## gmp

## ntl
[A Tour of NTL: Introduction (libntl.org)](https://libntl.org/doc/tour-intro.html)
NTL is a high-performance, portable C++ library providing data structures and algorithms for arbitrary length integers; for vectors, matrices, and polynomials over the integers and over finite fields; and for arbitrary precision floating point arithmetic.
NTL provides high quality implementations of state-of-the-art algorithms for:
- arbitrary length integer arithmetic and arbitrary precision floating point arithmetic;
- polynomial arithmetic over the integers and finite fields including basic arithmetic, polynomial factorization, irreducibility testing, computation of minimal polynomials, traces, norms, and more;
- lattice basis reduction, including very robust and fast implementations of Schnorr-Euchner, block Korkin-Zolotarev reduction, and the new Schnorr-Horner pruning heuristic for block Korkin-Zolotarev;
- basic linear algebra over the integers, finite fields, and arbitrary precision floating point numbers.

模乘100，模幂4000，模逆4000


# 硬件规模

p, q 各扩展1位，n位宽取4100，以保证m<n。生成的随机数r\<n，和n取同位宽。
输入数据高位补零，明文扩展为2050，密文扩展为4100

加密：
 $c=Enc(m,n,g,r)=g^m r^n(mod\ n^2) = (mn+1)r^m\ mod\ n^2$

| 所有操作 | 输入位宽位宽，操作符）        | 输出位宽      |
| :--- | :----------------- | :-------- |
| 乘    | 2050 * 2050        | 4100      |
| 加    | 4100 + 1           | 4100（不进位） |
| 模幂1  | 2050^2050 mod 4100 | 4100      |
| 模乘1  | 4100*4100 mod 4100 | 4100      |
解密：
 $m=Dec(c,λ,μ)=L(c^λ\ mod\ n^2)∗ μ(mod\ n)= \frac{L(c^λ\ mod\ n^2)}{L(g^λ\ mod\ n^2)}(mod \ n)=(\frac{c^\lambda\ mod\ n^2 -1}{n}\cdot u)mod\ n$

| 所有操作 | 输入位宽（位宽，操作符）       | 输出位宽      |
| ---- | ------------------ | --------- |
| 模幂2  | 4100^2050 mod 4100 | 4100      |
| 减    | 4100-1             | 4100（不借位） |
| 除    | 4100 / 2050        | 2050      |
| 模乘2  | 2050*2050 mod 2050 | 2050      |
同态加：
 $c= (c_1 \cdot c_2) \ mod\ n^2$

| 所有操作 | 输入位宽（位宽，操作符）       | 输出位宽 |
| ---- | ------------------ | ---- |
| 模乘1  | 4100*4100 mod 4100 | 4100 |
同态乘：
$c= {c_1}^{m}\ mod\ n^2$

| 所有操作 | 输入位宽位宽，操作符）       | 输出位宽 |
| ---- | ----------------- | ---- |
| 模幂2  | 4100^2050mod 4100 | 4100 |

