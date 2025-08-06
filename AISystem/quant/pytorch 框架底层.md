---
tags:
  - note
dateCreated: 2024-11-13
dateModified: 2025-07-11
---

https://pytorch.ac.cn/

OpenMMLab 是一个国产的计算机视觉算法系统。

<a href=" https://pytorch.org/")>Pytorch</a> 是由 Facebook 开发的开源深度学习框架。Pytorch 提供了完整的工具链用于构建、训练和部署深度学习模型。

`torch` 基本模块包括：

- ` autograd` 自动求导
- `nn` 神经网络构建
	- `nn. Module` 通用模型接口
	- `nn. functional ` 函数库，提供了线性函数、卷积、池化、非线性激活等
- `optim` 优化器，支持常用算法如 SGD/Adam/RMSprop，以及常用的学习率策略（步长衰减、指数衰减、学习率循环）
- `utils. data ` 数据接口，包含统一的数据集模型 Dataset，以及支持多线程预读的数据加载器 DataLoader。
