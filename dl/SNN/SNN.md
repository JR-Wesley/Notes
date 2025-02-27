# 2024年顶会、顶刊SNN相关论文
https://blog.csdn.net/qq_43622216/article/details/135167498



# SNN与CNN
https://www.zhihu.com/question/297704400/answers/updated

**脉冲神经网络（SNN）**：
- 被认为是第三代神经网络，更接近于生物神经网络的工作方式。
- 神经元之间通过脉冲（或称为尖峰）进行通信，这些脉冲是离散的事件，发生时机和频率携带信息。
- 能量效率更高，因为神经元仅在发送脉冲时消耗能量。
- 计算模型通常是时间依赖的，需要处理时间上的动态行为。

它通过模拟真实的神经元活动状态和信息传递方式，以脉冲序列进行运算，具有较低能耗的优势，尤其适合对能耗敏感和需要高度仿生的场景，如分类、回归任务等。

**卷积神经网络（CNN）**：
- 属于第二代神经网络，特别适用于处理图像、视频和语音等网格状数据的前馈神经网络。
- 神经元以连续的方式处理信息，通过激活函数（如ReLU）进行非线性映射。其核心技术包括卷积层的空间权值共享和池化层的数据降维，能够有效提取局部特征并在图像识别、目标检测等领域取得卓越效果。
- 神经元在每次前向传播时都会进行计算，消耗固定的能量。
- 结构通常包括卷积层、池化层和全连接层，主要处理空间数据。
## SNN的发展和前景

**发展**：

- SNN的研究起步较晚，主要因为其复杂性以及缺乏高效的训练算法。
- 早期的SNN模型主要用于理解神经科学的基本原理。
- 近年来，随着深度学习的兴起和神经形态硬件的发展，研究者开始探索如何将深度学习技术应用于SNN，以及如何在硬件上高效地实现SNN。

**前景**：

- SNN在理论上具有较高的能效和对时间动态模式的处理能力，这使得它们在某些应用上具有独特优势，尤其是在需要低功耗或实时处理的场景。
- 神经形态硬件的发展，如IBM的TrueNorth和Intel的Loihi芯片，为SNN的实际应用提供了硬件基础。
- 尽管如此，SNN目前还没有在性能上超越传统的CNN和其他深度学习模型，特别是在复杂任务如图像和语音识别上。
- SNN的主要挑战之一是找到有效的学习规则（例如，类似于反向传播的算法），因为SNN的离散和非连续性质使得直接应用传统的学习算法变得困难。

综上所述，SNN作为一种更贴近生物神经网络的计算模型，具有一定的应用潜力和发展前景，特别是在能效要求极高的场景。然而，相比于成熟的CNN，SNN在算法发展、训练效率和应用广泛性方面还处于相对初级阶段。未来的研究需要进一步解决SNN的这些挑战，并探索其在特定领域的应用优势。


## SNN结合CNN
尽管SNN与CNN在应用场景和设计理念上有所差异，但研究者正尝试将两者优势融合，例如结合SNN的高效性和CNN的强大特征提取能力，以推动神经网络技术的发展。
脉冲神经网络的编码方式可以应用于CNN。将CNN转换成脉冲形式的CNN（也称为Spike-CNN）可以降低功耗，使其适合部署在低功耗的神经计算硬件上。这种转换涉及将CNN中的连续值操作替换为脉冲形式的操作，以适配SNN的处理方式。这通常用于神经形态计算领域，旨在模仿人脑的处理方式，以实现能效比的提升。

### **对神经编码的研究突破**

2023年，一篇名为《Neural encoding with unsupervised spiking convolutional neural networks (SCNN)-based framework to achieve neural encoding in a more biologically plausible manner》的论文在《Nature》上发表。这篇研究提出了一个基于脉冲神经网络（SNN）的框架，目的是以更符合生物学原理的方式实现神经编码。该框架利用无监督的SNN来提取图像刺激的视觉特征，并利用基于受体场的回归算法从SNN特征中预测fMRI响应。这一研究为现有的人工智能模型提高其认知能力指明了一种可能的生物类优化方向。

### **对能源效率和硬件实现的突破**

在2023年6月，《To Spike or Not to Spike? A Quantitative Comparison of SNN and CNN FPGA accelarator》
主要研究了SNN加速器是否真的能达到与CNN加速器相比的能源和资源效率。这是一个非常重要的研究问题，因为涉及到SNN和CNN在硬件实现上的比较和选择。这一研究为反向促进生命科学中生物神经网络的可塑性研究新发现提供了启发。


## 理论

讲解原理
https://www.bilibili.com/video/BV1sN41117kC/?spm_id_from=333.999.0.0&vd_source=bc07d988d4ccb4ab77470cec6bb87b69


理论神经科学
https://www.zhihu.com/people/helloguai/posts

## 项目
ODIN SNN
SNN的一些项目和论文（算法
https://blog.csdn.net/edward_zcl/article/details/103285222
https://blog.csdn.net/Kyrie6c/article/details/115289056

IC
https://cloud.tencent.com/developer/article/2016197
tiny_ODIN
https://blog.csdn.net/HFUT90S/article/details/136771691

SNN和其他加速器的数字IC实现
https://blog.csdn.net/hfut90s/category_12102566.html