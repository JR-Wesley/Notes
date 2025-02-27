# DDR



# ram
[Xilinx的分布式RAM和块RAM——单口、双口、简单双口、真双口的区别-腾讯云开发者社区-腾讯云 (tencent.com)](https://cloud.tencent.com/developer/article/1814213)
[深度详解简单双口RAM(Simple Dual Port RAM)和真双口RAM(True Dual Port RAM)的区别-CSDN博客](https://blog.csdn.net/weixin_46720928/article/details/136560553)


单口 RAM（Single RAM）、双口 RAM（Dual RAM）、简单双口 RAM（Simple-Dual RAM）、真双口 RAM（True-Dual RAM）

BRAM有两种RAM，简单双口RAM（simple dual port ram）和真双口RAM（true dual port ram）。
总结：

分布式 RAM，支持简单双口 RAM 和双口 RAM，不能配置成真双口 RAM

|      |     |      |             |
| ---- | --- | ---- | ----------- |
| BRAM | 单双  | 单口   | 1个口，不能同时读写  |
|      |     | 双口   | 两个口，可以同时读写  |
|      | 简真  | 简单双口 | 一个口只读，一个口只写 |
|      |     | 真双口  | 两个口都可读写     |
| DRAM | 单双  | 单口   | 一组读写共享地址线   |
|      |     | 双口   | 读写有各自独立的地址线 |
|      | 简真  | 简单双口 | 1个输出        |
|      |     | 真双口  | 2个输出        |



