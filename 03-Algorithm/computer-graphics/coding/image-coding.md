bpg, avif, webp, png
h.263/avc, h.265/hevc, h.266/vvc
vp9, av1, av2
avs1/2/3, svac


# raw图格式
[Research (columbia.edu)](http://www.stat.columbia.edu/~jakulin/jpeg-ls/mirror.htm)


## bayer格式

Bayer是相机内部的原始图片, 一般后缀名为.raw。可以利用一些软件查看, 比如picasa、irfanview、photoshop。
我们相机拍照下来存储在存储卡上的.jpeg或其它格式的图片, 都是从.raw格式转化过来的。 raw格式内部的存储方式有多种, 但不管如何, 都是前两行的排列不同。
输出格式有一定顺序，一般分为四种：
GRBG/ RGGB/ BGGR/ GBRG
每种格式种存在两个G分量。其中每个分量代表一个piexl，所以GR/BG代表4个piexl，就表示sensor上面的4个晶体管，每个晶体管只采集一个颜色分量，然后通过插值计算出每个piexl的其他分量，目的是降低功耗。

[【学习笔记】关于RAW图片的概念学习_raw10和raw16区别-CSDN博客](https://blog.csdn.net/moon9999/article/details/132012531)


## RAW存储格式
# 第1部分、什么是图像文件

简而言之，图像文件格式是用户存储和表示数字图片的方式。以某种特定的方式存储图像的数据，使它可以显示在电子屏幕上并且被打印。这个过程是通过“光栅化”实现的。理想情况下，图像由像素网格构成，每个像素都分配一个值（一个比特）。

用于存储和栅格图像的机制可以根据不同的标准而变化。这使得每一种图像格式都非常独特。大多数情况下，图像文件格式会使用压缩或未压缩（即丢失或无损压缩）技术。此外，一些图像格式也专用于矢量文件存储。一些常用的不同类型图像格式有JPEG，BMP，GIF，PNG，PSD和TIFF等。除了它们之外，还有许多其他类型的格式。

# 第2部分、JPG vs JPEG vs JPEG 2000

JPEG压缩包含一些最流行的图像类型。因为JPEG文件类型可以轻松压缩数据且被各种设备普遍接受。

**1.什么是JPEG？**

JPEG代表联合图像专家组。它是最流行的图像格式之一，具有6种主要格式：JPEG，JPG，JPE，JIF，JFI和JFIF。JPEG的压缩程度大多为10：1。这意味着JPEG格式可以在保持图像质量的同时压缩数据。

- 它消耗存储空间少，甚至可以将图像压缩到原始大小的5％。
- JPG文件扩展名被各种设备普遍接受。
- 它可以存储高清图像。此外，JPG格式也被打印机和其他硬件单元接受。
- 唯一的缺陷是JPG不支持图像层。

**2.什么是JPG？**

JPG文件扩展名是JPEG压缩组支持的格式之一。就像JPEG一样，JPG也采用有损压缩方法。这意味着，在压缩过程中照片的原始尺寸会被减小，但同时其中一些数据也会受到影响。

- JPG / Exif格式主要用于数码相机等摄影工具。
- JPG / JFIF格式用于在Web（WWW）上存储和传输图片。
- 它是JPEG组的一部分，符合世界所有主要标准中的图片格式要求。

**3.什么是JPEG 2000？**

JPEG 2000是一种高级压缩技术，是JPEG组的一部分。与JPG不同，它既支持有损压缩，也支持无损压缩。它提高了不同平台上图像的整体质量。

- 它是一种较新的格式，源自JPEG。
- 它同时具备有损技术和无损技术。
- 以图像编辑功能为主，主要用于拍摄单个电影帧。

**4. JPG和JPEG之间的区别**

理想情况下，JPG和JPEG之间没有太大区别。JPEG是一组不同的扩展名，JPG是其中的一部分。最初，Windows（或DOS）仅支持3个字母的扩展名。

因此，在Windows系统中，开发者删除了“E”，JPEG被截断为JPG。而Mac系统则继续使用JPEG。今天，这些格式可以互换使用，但JPG被世界各地的人们更广泛地接受。

**5. JPEG和JPEG 2000之间的区别**

JPEG 2000是一种高级图像格式，它为用户提供了更加动态的图片保存范围。用户可以通过应用无损压缩来保留原始照片的一些重要信息。

这些原始信息是JPEG所缺失的，因为它默认采用有损技术。然而，由于JPEG 2000有其局限性且误码率较高，JPEG反而成为被人们广泛接受的格式。

**5. JPG vs JPEG 2000**

JPG和JPEG 2000之间的区别和JPEG与JPEG 2000之间的差别基本类似。JPEG 2000提供了更好，更先进的压缩技术。文件大小可能更大，但会保留更多原始信息。而JPG是一种普遍推广的格式，可以将文件缩小到原始大小的5％。

# 第3部分、什么是GIF文件

网络或社交媒体的用户对GIF这一图片格式一定非常熟悉。接下来，我们就给大家介绍一下GIF在计算机术语的含义。

**1.什么是GIF？**

GIF指“图形交换格式”。今天，GIF大多用于表示动画和视频剪辑。这种格式早在1987年就被引入国内，但在过去几年才随着社交媒体的繁荣发展走入公众视野，获得广泛的普及。

GIF遵循LZW无损压缩技术，这意味着它保留了原始数据质量。但是，GIF仅支持8位像素，也就是只有256种

**2. GIF用于什么？**

今天，GIF常常用于表达个人感情，且因其强互动性而被用于娱乐甚至教育。虽然，GIF大多在WhatsApp，Messenger，Tumblr，Twitter等社交平台上被使用。但如果用户想要更多可供使用的GIF图像，还是需要前往Tenor，Giphy和其他专用GIF目录寻找。

**3. GIF和JPEG之间的区别**

GIF和JPEG之间最主要区别就是GIF是动态的，而JPEG是静态的。也就是说，GIF可以存储几秒钟的运动图像或进行迷你剪辑。

此外，GIF支持无损压缩，而JPEG采用有损技术。然而，GIF仅能使用256种颜色的光谱并且需要占据更大的存储空间。

# 第4部、什么是PNG文件

另一种常用的无损图像格式是PNG。接下来让我们介绍一下PNG格式的主要信息。

**1.什么是PNG文件？**

PNG代表便携式网络图形，是一种光栅图像格式。它于1997年推出，并于2004年获得ISO标准认可。PNG最初设计时只用于在互联网上传输图像（而不是用于打印），这就解释了它为什么只支持RGB频谱。但是，由于PNG的透明背景，它也常常被用于美术设计。

**2. PNG无损吗？**

答案是肯定的，PNG采用无损压缩技术，由Huffman和LZ77的组合代码构成，支持24位RGB颜色，并且遵循2阶段压缩过程。同时，PNG向用户提供可供修改的压缩参数。用户可以选择是否要保持原始质量或实施某种程度的有损压缩。

**3. JPEG与PNG**

JPEG和PNG之间的主要区别在于JPEG采用有损压缩技术，而PNG采用提供压缩参数的无损压缩技术。因此，PNG文件的大小通常大于JPEG。PNG具有透明背景，从而可以扩大其应用范围，而JPEG文件没有透明背景。

**4. PNG与JPG**

JPG和PNG之间的区别与JPEG和PNG之间的区别大致相同。由上可知，JPG是JPEG格式组中的一部分。JPG格式通常用于只存储照片，而PNG格式可以用于存储矢量，图形，图标，文本，绘图和编辑其他相关文件。因为PNG具有透明背景，可以轻松覆盖在另一张图片上，而JPEG文件无法做到这一点。

# 第5部分、什么是BMP文件

BMP文件格式已经存在了很长时间，但它的使用率正在逐渐降低。接下来给大家介绍BMP图像格式的相关信息。

**1.什么是BMP？**

BMP也称为位图图像文件，是一种用于存储位图的光栅图形格式。该格式最初由Microsoft开发，用于存储彩色和单色图像。

除了BMP之外，它也遵循DIB格式。这是一种采用无损算法的简单压缩技术，通过霍夫曼或RLE编码实现4位或8位编码技术。因此，BMP的图像尺寸明显大于PNG或JPEG等其他格式。

**2. BMP与PNG**

BMP和PNG的区别很大。BMP是一种无损但未压缩的格式，而PNG是一种无损压缩格式。PNG拥有透明背景，但并非所有BMP文件都支持透明度（Alpha通道）。此外，PNG的文件大小也远小于BMP。而且与BMP相比，PNG主要用于设计。

**3. BMP vs JPG**

BMP和JPEG之间的区别非常明显。BMP采用无损未压缩技术，而JPG正相反，采用有损压缩技术。JPG的文件大小比BMP小得多，但在压缩过程中图片质量也会受到影响。

**4. BMP vs JPEG**

BMP和JPEG都有不同的应用领域。BMP主要用于存储图像的原始数据，而JPEG则多用于文件传输。由于压缩技术的原因，JPEG文件的大小远小于BMP。

# 第6部分、什么是RAW图像

RAW文件最多出现在摄影师的相机内存卡中。那究竟什么是RAW格式呢？让我们向大家介绍它的具体信息。

**1. RAW在摄影中意味着什么？**

顾名思义，RAW文件是未经处理的照片。也就是说，它们是相机尚未处理的原始文件。因为未经处理，所以用户无法通过常规应用程序编辑RAW文件。

它是一种预转换格式，可用于Photoshop之类的高级编辑。但是，它们不适合被打印。与JPEG相似，RAW也是一组文件扩展名，其中一些常用格式为3FR，DNG，DATA，ARW，SR2等等。

**2. RAW与JPEG**

理想情况下，RAW图像为用户提供了更高级的编辑选项。例如，我们可以在RAW文件上轻松调整颜色，亮度，偏振等。这是JPEG文件不能做到的。

RAW和JPEG之间的另一个区别是文件大小和压缩技术。与JPEG的有损技术相比，RAW采用无损压缩（或高质量有损）技术。此外，RAW文件的大小明显大于JPEG（有时甚至大5倍）。

# 第7部分、什么是TIFF文件

TIFF代表标记图像文件格式，采用TIFF或TIF的扩展名。它是一种光栅图形格式，主要用于发布域。

除了图像发布，它还支持图片打印以及光学字符识别。该格式最初由Adobe在20世纪80年代开发，采用未压缩或CCIT（霍夫曼编码）技术。

**1. JPG vs TIFF**

JPG和TIFF的差别十分明显。TIFF是一种无损数据压缩格式，可保留图像的原始参数，而JPEG遵循有损技术。因此，TIFF文件大小也比JPEG大得多。

**2. TIFF与JPEG**

TIFF和JPEG之间的主要区别在于它们的应用。JPEG主要用于文件在万维网上的存储和传输，而TIFF在发布方面起着重要作用。

# 第8部分、什么是Photoshop

Photoshop是Adobe开发的最流行的图像编辑工具之一。它主要用于编辑光栅图形格式的图像。该工具早在1990年就已发布，到今天依然是最强大的图像编辑器之一。

它几乎兼容所有主要图像格式，并提供大量工具选项卡。例如，用户可以更改图像的属性、执行图层叠加、插入文本以及进行各种各样的图片编辑。

虽然Photoshop支持多种格式，但最常用的两种文件扩展名是PSD和CR2。

**1.什么是PSD？**

PSD和PSB是两种原生的Adobe Photoshop格式。Photoshop扩展的最大优势之一就是它完整保留了用户的编辑过程。也就是说，用户可以恢复所有已完成的文本、叠加、元数据和图像编辑。只需在Photoshop中打开PSD文件即可查看所有已保存的图层和原始图像。

**2.什么是CR2文件？**

CR2即Canon Raw Version 2，它用于存储由佳能相机拍摄的RAW文件。因为CR2是图片的原始格式，所以它可以保留图像所有原始参数和基础信息。该格式主要用于Photoshop等编辑工具。

# Compression for Bayer CFA Images: Review and Performance Comparison

## abstract
Bayer color filter array (CFA) images are captured by a single-chip image sensor covered with a Bayer CFA pattern.
Compression methods can be divided into:  the compression-first-based (CF-based) scheme and the demosaicing-first-based (DF-based) scheme.


## 4. Experimental Results


# JPEG_LS

[JPEGLS图像压缩算法的FPGA实现（一）压缩算法_jpeg-ls-CSDN博客](https://blog.csdn.net/alangaixiaoxiao/article/details/106664408)

JPEG-LS是在ISO/ITU的新标准中用于对静态连续色调图像进行无损或进无损压缩的一种算法。它不是JPEG的一个简单扩充或修正，是一种新的压缩方法。支持无损和近无损图像压缩，不使用离散余弦变换，也不使用算术编码，仅在近无损压缩模式下有限地使用量化。该算法具有实现复杂度低，保真度高等特点

JPEG-LS 图像压缩算法的结构框图如图所示。如果图像中有大块区域像素灰度比较平滑，则可以通过游程模式进行压缩。具体选择哪种编码模式，是根据上下文信息来确定。


该算法对图像压缩过程中，不需要对图像进行离散余弦变换（DCT）和算数编码等，只进行预测和Golomb-Rice 编码，算法复杂度较低。普通模式下编码步骤为：
a. 求当前像素的上下文参数；
b. 根据上下文模板中的相邻像素值进行预测，得到当前像素的预测值，以上述上下文参数修正预测值；
c. 利用预测值与原像素得到预测误差，对预测误差进行修正编码；
d. 更新上下文的相关参数；
e. 对预测残差进行 Golomb



# The LOCO-I Lossless Image Compression Algorit0hm: Principles and Standardization into JPEG-LS

LOCO-I is the core of JPEG-LS
Lossless data compression schemes often consist of two components: modeling and coding
- modeling - an inductive inference problem in which  the data is observed sample by sample in some pre-defined order: interfere the next sample from the previous data
- arithmetic codes seperate them by realizing any probability assignment P
The two milestones makes it possible to view the problem as one of probability assignment. LOCO-I projects the image modeling principles into a low complexity plane, from a modeling and coding perspective.


## 3 Detailed description of JPEG-LS
![[JPEG-LS.png]]
The prediction and modeling units in JPEG-LS are based on the causal template 
### 3.1 prediction
Adaptive model of the local edge direction is replaced by a fixed predictor
![[predictor.png]]




A low complexity lossless Bayer CFA image compression

Adaptive Pipeline Hardware Architecture Design
and Implementation for Image Lossless
Compression/Decompression Based on JPEG-LS

# VLSI Implementation of a Cost-Efficient Near-Lossless CFA Image Compressor for Wireless Capsule Endoscopy
## ABSTRACT
Propose: a novel near-lossless color filter array (CFA) image compression algorithm based on JPEG-LS
consists of a pixel restoration, a prediction, a run mode, and entropy coding modules
a context table & row memory consume HW -> a novel context-free and near-lossless image compression algorithm,  a novel prediction, run mode, and modified Golomb–Rice coding techniques


## I. INTRODUCTION
 For  wireless video capsule endoscopy limited the frequency 
JPEG, JPEG 2000 [3], Motion JPEG [4], MPEG-1/2/4 [5], MEPG-7 [6], AVC and HEVC [7] have high compression rate but high computational complexities
Low-complexity and hp image compression algorithm based on JPEG-LS are proposed

## II. NEAR-LOSSLESS IMAGE COMPRESSION ALGORITHM
JPEG-LS: five main components: a context, a prediction, a regular, a run mode, and an entropy coding models

Proposed: a pixel restoration, a prediction, a run mode, a modified Golomb-Rice coding, and a bitstream models



### A. PIXEL RESTORATION
CFA - one-third the number of the pixels but lose the dependence between two contiguous pixels
the image in CFA format was restored to a red, blue and green line buffers with the lengths of 1/4, 1/4, 1/2 width of the CFA image

### B. PREDICTION


# A low complexity lossless Bayer CFA image compression
## Abstract
This paper presents a lossless color transformation and compression algorithm for Bayer color filter array (CFA) images.
Conventional: compression is after demosaicking
Proposed: compression is performed first using a four-channel color space  to reduce the correlation among the Bayer CFA color components. After the color tranformation, components are independently encoded using DPCM. The compression includes an adaptive Golomb-Rice and unary coding

