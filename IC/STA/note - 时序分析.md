---
dateCreated: 2025-02-27
dateModified: 2025-02-27
---
## 跨时钟域

同步与异步

3 种方法跨时钟域处理方法如下:

1. 打两拍;
2. 异步 FIFO
3. 格雷码转换。对于格雷码,相邻的两个数间只有一个 bit 是不一样的 -

大家很清楚,处理跨时钟域的数据有单 bit 和多 bit 之分,而打两拍的方式常见

于处理单 bit 数据的跨时钟域问题。

打两拍，就是定义两级寄存器,对输入的数据进行延拍。两级寄存是一级寄存的平方,两级并不能完全消除亚稳态危害,但是提高了可靠性减少其发生概率。总的来讲,就是一级概率很大,三级改善不大

[FPGA跨时钟域的处理方法](https://blog.csdn.net/emperor_strange/article/details/82491085?utm_source=app)

 https://blog.csdn.net/emperor_strange/article/details/82491085?utm_source=app

《Clock Domain Crossing》翻译与理解（1）亚稳态 - 0431 大小回的文章 - 知乎

https://zhuanlan.zhihu.com/p/359325914

硬件架构的艺术.pdf 第三章

CDC-- 讲师卢子威.pptx

总结的文档：CDC 总结.md


### Glitch Free 时钟切换

[Glitch Free时钟切换技术](https://mp.weixin.qq.com/s/w3Wu7HkSr5v94kHrLvRIcw)

[Glitch Free时钟切换技术另一篇博客](https://blog.csdn.net/Reborn_Lee/article/details/90378355?tdsourcetag=s_pctim_aiomsg)
