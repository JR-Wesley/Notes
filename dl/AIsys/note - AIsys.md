---
dateCreated: 2025-02-11
dateModified: 2025-05-26
---

相关资源推荐：https://zhuanlan.zhihu.com/p/20076957712

AI 相关大学课程严忻恺：https://www.zhihu.com/people/yan-xin-kai-38/posts

综述：

- https://zhuanlan.zhihu.com/p/101544149
- https://blog.csdn.net/qq_42722197/article/details/119814538
- https://zhuanlan.zhihu.com/p/616138047
- <a href="https://zhuanlan.zhihu.com/p/33876622">一天搞懂</a>
- <a href=" https://zhuanlan.zhihu.com/p/20076957712?utm_psn=1872615359586111488">知乎总结</a>
- <a href=" https://chenzomi12.github.io/index.html">ZOMI AI 系统</a><img src=" https://chenzomi12.github.io/_images/03Architecture031.png" alt="系统框图">
- <a ref=" https://openmlsys.github.io/index.html">open ML sys </a>
- <a href="https://novel.ict.ac.cn/aics/">中科大智能计算系统课程和书</a>

https://eyeriss.mit.edu/tutorial-previous.html eyeriss tutorial

AL chip:

- <a href=" https://nycu-caslab.github.io/AAML2024/labs/lab_2.html">CSIC 30066 台湾课程</a>
- CSCS 10014: Computer Organization ( https://nycu-caslab.github.io/CO2024/index.html# )
- <a href=" https://hanlab.mit.edu/courses/2024-fall-65940">TinyML and Efficient Deep Learning Computing</a>by MIT hansong, 如何基于已有的硬件进行优化。
- <a href=" https://csg.csail.mit.edu/6.5930/index.html">6.5930/1 Hardware Architecture for Deep Learning</a>如何设计更好的面向深度学习的硬件
- <a href=" https://people.cs.nycu.edu.tw/~ttyeh/course/2024_Fall/IOC5009/outline.html">AAML2024 台湾课程</a>
- <a href="https://nycu-caslab.github.io/AAML2024/index.html"> AAML2024 实验网站</a>
- EE 290 笔记 https://www.zhihu.com/people/zfeng-xin-zw/posts
- ece 5545 https://www.bilibili.com/video/BV1kn4y1o7Eu/?spm_id_from=333.1387.favlist.content.click&vd_source=bc07d988d4ccb4ab77470cec6bb87b69 https://zhuanlan.zhihu.com/p/668411397

利用了谷歌的 CFU 平台<a href=" https://cfu-playground.readthedocs.io/en/latest/index.html">一个介绍的网站</a>

CS 217 MIT 课程整理：https://www.zhihu.com/people/yan-xin-kai-38/posts

# EE 290
# DNN

# Quantization

NVIDIA Ampere Architecture features the Third-generation Tensor Cores:

- Acceleration for all data types including FP 16, BF 16, TF 32, FP 64, INT 8, INT 4, and Binary.

## Floating-Point Arithmetic
## Fixed-Point Arithmetic
## Hardware Implication
## DNN Quantization


# 介绍

通俗来讲，机器学习是指从数据中学习出有用知识的技术。以学习模式分类，机器学习可以分为监督学习（Supervised Learning）、无监督学习（Unsupervised Learning）和强化学习（Reinforcement Learning）等。

- 监督学习是已知输入和输出的对应关系下的机器学习场景。比如给定输入图像和它对应的离散标签。
    
- 无监督学习是只有输入数据但不知道输出标签下的机器学习场景。比如给定一堆猫和狗的图像，自主学会猫和狗的分类，这种无监督分类也称为聚类（Clustering）。
    
- 强化学习则是给定一个学习环境和任务目标，算法自主地去不断改进自己以实现任务目标。比如 AlphaGo围棋就是用强化学习实现的，给定的环境是围棋的规则，而目标则是胜利得分。