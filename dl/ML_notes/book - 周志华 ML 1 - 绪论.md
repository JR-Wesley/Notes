---
banner: "[[../../300-以影像之/举伞司辰.jpg]]"
dateCreated: 2024-11-02
dateModified: 2024-11-17
---
# 1.1 定义

机器学习通过计算，利用经验来改善系统性能。“经验”对应“数据”，计算机从数据产生一个算法模型 model，即学习算法 learning algorithm。有了学习算法，计算机能根据提供的数据产生模型；面对新情况时作出判断。

另一本经典教材的作者 Mitchell 给出了一个形式化的定义，假设：

 - **P**：计算机程序在某任务类 T 上的性能。
 - **T**：计算机程序希望实现的任务类。
 - **E**：表示经验，即历史的数据集。
若该计算机程序通过利用经验 E 在任务 T 上获得了性能 P 的改善，则称该程序对 E 进行了学习。

# 1.2 基本术语

假设我们收集了一批西瓜的数据，例如：（色泽=青绿; 根蒂=蜷缩; 敲声=浊响)，(色泽=乌黑; 根蒂=稍蜷; 敲声=沉闷)，(色泽=浅自; 根蒂=硬挺; 敲声=清脆)……每对括号内是一个西瓜的记录，定义：

 - 所有记录的集合为：**数据集 dataset**。
 - 每一条记录为：一个**实例 instance 或样本 sample**。
 - 反映事件或对象的表现或性质的事项，如色泽或敲声，单个的特点为**特征 feature 或属性 attribute**。
 - 属性上的取值，称为**属性值 attribute value**。
 - 属性张成的空间称为**属性空间 attribute space、样本空间 sample space、输入空间**。
 - 对于一条记录，如果在三维空间表示，每个西瓜都可以用坐标轴中的一个点表示，一个点也是一个向量，例如（青绿，蜷缩，浊响），也可以把一个实例称为一个**特征向量 feature vector**。
 - 一个样本的特征数为：**维数 dimensionality**，该西瓜的例子维数为 3，当维数非常大时，也就是现在说的“维数灾难”。

> [!note]
一般，令 $D=\{\mathbf{x_1, x_2, , x_m}\}$ 表示包含 $m$ 个示例的数据集，每个示例由 $d$ 个属性描述，每个示例 $\mathbf{x_i}=(x_{i1}; x_{i2};\cdots;x_{id})$ 是 $d$ 维空间 $\mathcal{X}$ 中的一个向量。$\mathbf{x_i} \in \mathcal{X}$，其中 $x_{ij}$ 是 $\mathbf{x_i}$ 在第 $j$ 个属性上的取值。

从数据中学得模型的过程称为**学习 learning 或训练 training**。这个过程中使用的数据称为**训练数据 training data**，每一条记录称为一个**训练样本 training sample**，所有训练样本的集合为**训练集 trainning set**。学得模型对应了关于数据的某种潜在规律，称为**假设 hypothesis**；这种潜在规律自身，称为**真相或真实 ground-truth**，学习过程就是为了找出或逼近真相。也可将模型称为**学习器 learner**，可看作学习算法在给定数据和参数空间上的实例化。

希望学习一个模型，除了示例数据还要建立这样关于**预测 prediction** 的模型，要取得训练样本的结果信息。这里关于示例结果的信息，称为**标签 label**；拥有了标记信息的示例，称为**样例 example**。一般，用 $(\mathbf{x_i}, y_i)$ 表示第 $i$ 个样例，其中 $y_i \in \mathcal{Y}$ 是示例 $\mathbf{x_i}$ 的标记，$\mathcal{Y}$ 是所有标记的集合，亦称**标记空间 label space** 或输出空间。

若欲预测离散值，此类学习任务称为**分类 classification**；若欲预测连续值，称为**回归 regression**。对只涉及两个类的**二分类 binary classification**，通常称其中一个类为**正类 positive class**，另一个类为**反类 negative class**；涉及多个类别时，称为**多分类 multi-class classification**。

> [!note]
> 一般的，预测任务是希望通过对训练集 $\{(\mathbf{x_1},y_1), (\mathbf{x_2},y_2),\cdots, (\mathbf{x_m},y_m)\}$ 进行学习，建立一个从输入空间 $\mathcal{X}$ 到输出空间 $\mathcal{Y}$ 的映射 $f:\mathcal{X}->\mathcal{Y}$。对二分类，$\mathcal{Y}=\{-1, 1\} or \{0, 1\}$；对多分类，$|\mathcal{Y}|>2$；对回归，$\mathcal{Y}=\mathbb{R}$。

学得模型后，使用其进行预测的过程称为 testing，被预测的样本称为**测试样本 test sample**。

我们可以对西瓜做聚类 clustering，即将训练集中的西瓜分成若干组，每组称为一个 cluster。根据训练数据是否有标记信息。学习任务可以分为两类：

 - 训练数据有标记信息的学习任务为：**监督学习 supervised learning**。分类和回归都是监督学习的代表。
 - 训练数据没有标记信息的学习任务为：**无监督学习 unsupervised learning**，常见的有聚类和关联规则。
机器学习出来的模型适用于新样本的能力为：**泛化能力（generalization）**。具有强泛化能力的模型能很好的适用于整个样本空间。尽管训练集通常是样本空间的一个小采样，我们希望它能很好的反映整个样本空间的特性，否则在很难期望在训练集上学习的模型能在整个样本空间都工作得很好。通常加栓样本空间中全体样本服从一个未知的分布 distribution $\mathcal{D}$，我们获得的样本都是独立从这个样本上采样获得的，即**独立同分布 independent and identically distributed, i.i.d**。一般而言，训练样本越多，得到关于 $\mathcal(D)$ 的信息越多，越有可能通过学习能力获得强泛化能力的模型。

# 1.3 假设空间
**归纳 induction** 和**演绎 deduction** 是科学推理的两大基本手段。前者是从特殊到一般的**泛化 generaization** 过程，即从具体的事实归结出一般性规律；后者是从一般到特殊的**特化 specialization** 过程，即从基础原理推演出具体情况。如，数学公理中，基于一组公理和推理规则推导出与之相恰的定理是推导；从样例中学习是归纳。
广义的归纳学习大体相当于从样例中学习，狭义的归纳学习要求从训练数据中学得概念 concept，亦称概念学习或概念形成。可以把学习过程看成一个在所有假设 hypothesis 组成的空间进行搜索的过程，搜索目标是找到一个与训练集匹配 fit 的假设。

# 1.4 归纳偏好

 - ，[特殊]。
 - 所有测试样本的集合为：测试集（test set），[一般]。
	西瓜的例子中，我们是想计算机通过学习西瓜的特征数据，训练出一个决策模型，来判断一个新的西瓜是否是好瓜。可以得知我们预测的是：西瓜是好是坏，即好瓜与差瓜两种，是离散值。同样地，也有通过历年的人口数据，来预测未来的人口数量，人口数量则是连续值。定义：

 - 预测值为离散值的问题为：分类（classification）。
 - 预测值为连续值的问题为：回归（regression）。

	我们预测西瓜是否是好瓜的过程中，很明显对于训练集中的西瓜，我们事先已经知道了该瓜是否是好瓜，学习器通过学习这些好瓜或差瓜的特征，从而总结出规律，即训练集中的西瓜我们都做了标记，称为标记信息。但也有没有标记信息的情形，例如：我们想将一堆西瓜根据特征分成两个小堆，使得某一堆的西瓜尽可能相似，即都是好瓜或差瓜，对于这种问题，我们事先并不知道西瓜的好坏。

# 1.5 发展历程

20 世纪 50-70 年代，人工智能 artificial intelligence 处于推理期，人们认为只要赋予机器逻辑推理的能力，机器就具有智能。

20 世纪 70 年代，人工智能进入知识期，人们认识到仅有逻辑推理能力是远不够的。同时也面临知识工程瓶颈。

图灵在 20 实际 50 年代曾提到机器学习。50 年代后，基于神经网络的连接主义 connectionism 学习开始出现。6、70 年代，基于逻辑表示的符号主义 symbolism 学习技术发展。

# **2 模型的评估与选择**

**2.1 误差与过拟合**

我们将学习器对样本的实际预测结果与样本的真实值之间的差异成为：误差（error）。定义：

 - 在训练集上的误差称为训练误差（training error）或经验误差（empirical error）。
 - 在测试集上的误差称为测试误差（test error）。
 - 学习器在所有新样本上的误差称为泛化误差（generalization error）。

显然，我们希望得到的是在新样本上表现得很好的学习器，即泛化误差小的学习器。因此，我们应该让学习器尽可能地从训练集中学出普适性的“一般特征”，这样在遇到新样本时才能做出正确的判别。然而，当学习器把训练集学得“太好”的时候，即把一些训练样本的自身特点当做了普遍特征；同时也有学习能力不足的情况，即训练集的基本特征都没有学习出来。我们定义：

 - 学习能力过强，以至于把训练样本所包含的不太一般的特性都学到了，称为：过拟合（overfitting）。
 - 学习能太差，训练样本的一般性质尚未学好，称为：欠拟合（underfitting）。

可以得知：在过拟合问题中，训练误差十分小，但测试误差教大；在欠拟合问题中，训练误差和测试误差都比较大。目前，欠拟合问题比较容易克服，例如增加迭代次数等，但过拟合问题还没有十分好的解决方案，过拟合是机器学习面临的关键障碍。

![](https://i.loli.net/2018/10/17/5bc7181172996.png)

**2.2 评估方法**

在现实任务中，我们往往有多种算法可供选择，那么我们应该选择哪一个算法才是最适合的呢？如上所述，我们希望得到的是泛化误差小的学习器，理想的解决方案是对模型的泛化误差进行评估，然后选择泛化误差最小的那个学习器。但是，泛化误差指的是模型在所有新样本上的适用能力，我们无法直接获得泛化误差。

因此，通常我们采用一个“测试集”来测试学习器对新样本的判别能力，然后以“测试集”上的“测试误差”作为“泛化误差”的近似。显然：我们选取的测试集应尽可能与训练集互斥，下面用一个小故事来解释 why：

假设老师出了 10 道习题供同学们练习，考试时老师又用同样的这 10 道题作为试题，可能有的童鞋只会做这 10 道题却能得高分，很明显：这个考试成绩并不能有效地反映出真实水平。回到我们的问题上来，我们希望得到泛化性能好的模型，好比希望同学们课程学得好并获得了对所学知识 " 举一反三 " 的能力；训练样本相当于给同学们练习的习题，测试过程则相当于考试。显然，若测试样本被用作训练了，则得到的将是过于 " 乐观 " 的估计结果。

**2.3 训练集与测试集的划分方法**

如上所述：我们希望用一个“测试集”的“测试误差”来作为“泛化误差”的近似，因此我们需要对初始数据集进行有效划分，划分出互斥的“训练集”和“测试集”。下面介绍几种常用的划分方法：

**2.3.1 留出法**

将数据集 D 划分为两个互斥的集合，一个作为训练集 S，一个作为测试集 T，满足 D=S∪T 且 S∩T=∅，常见的划分为：大约 2/3-4/5 的样本用作训练，剩下的用作测试。需要注意的是：训练/测试集的划分要尽可能保持数据分布的一致性，以避免由于分布的差异引入额外的偏差，常见的做法是采取分层抽样。同时，由于划分的随机性，单次的留出法结果往往不够稳定，一般要采用若干次随机划分，重复实验取平均值的做法。

**2.3.2 交叉验证法**

将数据集 D 划分为 k 个大小相同的互斥子集，满足 D=D1∪D2∪…∪Dk，Di∩Dj=∅（i≠j），同样地尽可能保持数据分布的一致性，即采用分层抽样的方法获得这些子集。交叉验证法的思想是：每次用 k-1 个子集的并集作为训练集，余下的那个子集作为测试集，这样就有 K 种训练集/测试集划分的情况，从而可进行 k 次训练和测试，最终返回 k 次测试结果的均值。交叉验证法也称“k 折交叉验证”，k 最常用的取值是 10，下图给出了 10 折交叉验证的示意图。

![](https://i.loli.net/2018/10/17/5bc718115d224.png)

与留出法类似，将数据集 D 划分为 K 个子集的过程具有随机性，因此 K 折交叉验证通常也要重复 p 次，称为 p 次 k 折交叉验证，常见的是 10 次 10 折交叉验证，即进行了 100 次训练/测试。特殊地当划分的 k 个子集的每个子集中只有一个样本时，称为“留一法”，显然，留一法的评估结果比较准确，但对计算机的消耗也是巨大的。

**2.3.3 自助法**

我们希望评估的是用整个 D 训练出的模型。但在留出法和交叉验证法中，由于保留了一部分样本用于测试，因此实际评估的模型所使用的训练集比 D 小，这必然会引入一些因训练样本规模不同而导致的估计偏差。留一法受训练样本规模变化的影响较小，但计算复杂度又太高了。“自助法”正是解决了这样的问题。

自助法的基本思想是：给定包含 m 个样本的数据集 D，每次随机从 D 中挑选一个样本，将其拷贝放入 D'，然后再将该样本放回初始数据集 D 中，使得该样本在下次采样时仍有可能被采到。重复执行 m 次，就可以得到了包含 m 个样本的数据集 D'。可以得知在 m 次采样中，样本始终不被采到的概率取极限为：

![](https://i.loli.net/2018/10/17/5bc71811246dd.png)

这样，通过自助采样，初始样本集 D 中大约有 36.8% 的样本没有出现在 D' 中，于是可以将 D' 作为训练集，D-D' 作为测试集。自助法在数据集较小，难以有效划分训练集/测试集时很有用，但由于自助法产生的数据集（随机抽样）改变了初始数据集的分布，因此引入了估计偏差。在初始数据集足够时，留出法和交叉验证法更加常用。

**2.4 调参**

大多数学习算法都有些参数 (parameter) 需要设定，参数配置不同，学得模型的性能往往有显著差别，这就是通常所说的 " 参数调节 " 或简称 " 调参 " (parameter tuning)。

学习算法的很多参数是在实数范围内取值，因此，对每种参数取值都训练出模型来是不可行的。常用的做法是：对每个参数选定一个范围和步长λ，这样使得学习的过程变得可行。例如：假定算法有 3 个参数，每个参数仅考虑 5 个候选值，这样对每一组训练/测试集就有 5*5*5= 125 个模型需考察，由此可见：拿下一个参数（即经验值）对于算法人员来说是有多么的 happy。

最后需要注意的是：当选定好模型和调参完成后，我们需要使用初始的数据集 D 重新训练模型，即让最初划分出来用于评估的测试集也被模型学习，增强模型的学习效果。用上面考试的例子来比喻：就像高中时大家每次考试完，要将考卷的题目消化掉（大多数题目都还是之前没有见过的吧？），这样即使考差了也能开心的玩耍了~。
