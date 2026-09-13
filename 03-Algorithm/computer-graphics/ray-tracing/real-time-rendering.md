# Lesson 1: Bresenham’s Line Drawing Algorithm

a detailed walkthrough of implementing Bresenham's Line Drawing Algorithm in the context of a computer graphics lesson. It outlines several attempts at writing a function to draw a line between two points on a TGAImage, which is presumably a class used for handling image data in the tutorial.

Here's a summary of the key points from each attempt:

**First Attempt:**
- A simple implementation that interpolates the line's pixels using floating-point arithmetic and a loop with a constant step size.
- The code is inefficient and not optimized for performance.

**Second Attempt:**
- Attempts to optimize by removing the constant step size and using integer division to calculate the line's pixels.
- This version has issues with integer division causing holes in the line and missing lines when drawing in different directions.

**Third Attempt:**
- Addresses the issue of lines not being drawn correctly by swapping points to ensure a left-to-right drawing order and handling steep lines by transposing the coordinate system.
- This version is more robust but still has room for optimization.

**Fourth Attempt (Timings):**
- The author notes that compiler optimizations can sometimes outperform hand-optimized code.
- The focus is on profiling the code to identify bottlenecks, which in this case is the line drawing function itself and the TGAColor constructor calls.
- The optimization goal is to remove the expensive floating-point operations and divisions.

**Fourth Attempt Continued:**
- The algorithm is further optimized by removing the division from the loop and using an error term that only requires integer operations.
- The profiler confirms that this optimization significantly improves performance.

**Fifth and Final Attempt:**
- Eliminates floating-point operations entirely by replacing the error term with an integer equivalent.
- The final code is highly optimized and runs much faster than previous versions.

**Wireframe Rendering:**
- The lesson concludes with applying the optimized line drawing function to render a wireframe model from a wavefront obj file.
- The model's vertices are read, and lines are drawn between them to create a 3D model's wireframe representation on the 2D image.

# Lesson 2: Triangle rasterization and back face culling

In conclusion, the text provided delves into the intricacies of triangle rasterization and back face culling, two fundamental concepts in the field of computer graphics. The lesson begins with an introduction to the challenge of filling triangles, which is essential for rendering solid shapes in 2D and 3D graphics. The initial approach of using line drawing functions to outline the triangle is quickly recognized as inadequate for the task of filling the interior of the triangle.

The text then presents a more sophisticated method involving barycentric coordinates, which allows for the determination of whether a pixel lies within the bounds of a triangle. This method is not only elegant in its simplicity but also efficient in its execution, as it avoids the need for complex calculations and can be easily parallelized for modern, multi-threaded graphics processors.

As the lesson progresses, the implementation of flat shading is introduced, providing a basic level of detail to the rendered triangles by assigning colors based on vertex colors. This technique is a stepping stone towards more advanced shading models that take into account factors such as lighting and texture mapping.

The concept of back face culling is also explored, which is a crucial optimization technique that eliminates the rendering of triangles that face away from the viewer. This not only improves performance by reducing the number of triangles processed but also ensures that the rendered object appears correctly from any viewpoint.

Overall, the text provides a comprehensive overview of the foundational techniques used in rasterizing triangles, setting the stage for further exploration into more complex graphics rendering methods. The concepts and algorithms discussed are integral to the understanding of modern graphics pipelines and form the basis for the development of visually stunning and performant graphics applications.


# Ray Tracing Tech 
Ray tracing is a rendering technique that can produce incredibly realistic lighting effects. Essentially, an algorithm can trace the path of light and then simulate the way that the light interacts with the virtual objects it ultimately hits in the computer-generated world. 

Here is a video to show you.
[**12 Games That Look INSANE Due To Ray Tracing**](https://www.youtube.com/watch?v=lNSpiret-9g)

This is a collection organized by myself about the most important techs and some hard-core knowledge about ray tracing. These contents include the API, ray tracing algorithms (acceleration structure building, ray traversal& intersection& denoise& AI tech), benchmarks and so on.
This is my review of my knowledge system about ray tracing, and it can also help researchers engaged in research to quickly master the knowledge of ray tracing.

There is a collection of papers that interest me. The emphasis is focused on, but not limited to raytracing itself. Papers of significance are marked in **bold**. My comments are marked in *italic*.



## Intro about myself
My name is Jiale Yan. This is my [homepage](https://louivalley.github.io/). I'm currently a P.h.D candidate at [Artic Lab](http://www.artic.iir.titech.ac.jp/wp/en/), Tokyo Institute of Technology, Japan, advised by Prof. [Masato Motomura, IEEE Fellow](http://www.artic.iir.titech.ac.jp/wp/en/people/prof-motomura/). Before joining Tokyo Tech, I received my master's degree from the Institute of Microelectronics, Tsinghua University, advised by Prof. [Shaojun Wei](https://www.ime.tsinghua.edu.cn/info/1015/1151.htm) and Prof. [Shouyi Yin](https://www.ime.tsinghua.edu.cn/info/1015/1018.htm).


## Table of Contents
  - [Important Topics](#important-topics)
   - [Tutorial about RayTracing](#tutorial-and-survey) 
   - [Acceleration Structure in RT](#BVH)
   - [Traversal in RayTracing](#RTU)
   - [AI tech in RayTracing](#AIRT) 
   - [Denoise](#denoise)
   - [API about RayTracing](#API)
   - [Benchmarks](#benchmarks)
   - [Other Topics](#other-topics)
 - [Industry Contributions](#industry-contributions)

## Important Topics
### Tutorial about RayTracing
If you are a freshman and know nothing about the ray tracing, I strongly recommand you to read the following trilogy.
- **[Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html)** 
  - *Basic knowledge about ray tracing*
- **[Ray Tracing The Next Week](https://raytracing.github.io/books/RayTracingTheNextWeek.html)** 
  - *Advanced knowledge about ray tracing*
- **[Ray Tracing The Rest of Your Life](https://www.realtimerendering.com/raytracing/Ray%20Tracing_%20the%20Rest%20of%20Your%20Life.pdf)** 
  - *You can be a Ray tracing master*
  

### Acceleration Structure Building in Ray Tracing

Here we focus on the Bounding Volume Hierarchy (BVH), which is a popular ray tracing acceleration technique that uses a tree-based “acceleration structure” that contains multiple hierarchically-arranged bounding boxes (bounding volumes) that encompass or surround different amounts of scene geometry or primitives. 

 - **[Fast BVH Construction on GPUs](http://graphics.snu.ac.kr/class/graphics2011/references/2007_lauterbach.pdf)**(EUROGRAPHICS 2009)
   - *It presents two novel parallel algorithms for rapidly constructing bounding volume hierarchies on manycore GPUs. The first uses a linear ordering derived from spatial Morton codes to build hierarchies extremely quickly and with high parallel scalability. The second is a top-down approach that uses the surface area heuristic (SAH) to build hierarchies optimized for fast ray tracing.*

 - **[HLBVH: hierarchical LBVH construction for real-time ray tracing of dynamic geometry.](https://research.nvidia.com/sites/default/files/pubs/2010-06_HLBVH-Hierarchical-LBVH/HLBVH-final.pdf)** (Pantaleoni, Jacopo, and David Luebke. Proceedings of the Conference on High Performance Graphics. 2010.)
   - *It presents HLBVH and SAH-optimized HLBVH, two high performance BVH construction algorithms targeting real-time ray tracing of dynamic geometry.*

 - **[Heuristics for Ray Tracing Using Space Subdivision.](https://graphicsinterface.org/wp-content/uploads/gi1989-22.pdf)**
    - *This paper reports new construction algorithms which represent considerable improvement over conventional methods in terms of reducing the number of nodes, leaves, and objects visited by a ray . The algorithms employ the surface area heuristic and a heuristic for estimating the optimal splitting plane between the spatial median and the object median.*
 - **[PLOCTree: A Fast, High-Quality Hardware BVH Builder](https://dl.acm.org/doi/abs/10.1145/3233309)**
    - *This paper proposes PLOCTree, an accelerator for tree construction based on the Parallel Locally-Ordered Clustering (PLOC) algorithm*

### Traversal in Ray Tracing
It involves how light traverses objects in acceleration structures. It mainly involves two major strategies full stack and stackless.

- **Full Stack**
- **[Fast Ray Sorting and Breadth-First Packet Traversal for GPU Ray Tracing](http://charlesloop.com/GaranzhaLoop2010.pdf)** (Computer Graphics Forum, 2010)
  - *It presents a novel approach to ray tracing execution on commodity graphics hardware using CUDA.* 
  - *It includes sort rays, frustum creation, breadth-first Frustum Traversal three techs*

- **Stackless**
- **[Efficient Stack-less BVH Traversal for Ray Tracing](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.445.7529&rep=rep1&type=pdf)** (SCCG 2011 conference)
  - *It presents a traversal algorithm for BVH that does not need a stack and hence minimizes the memory needed for a ray*
- **[KD-Tree Acceleration Structures for a GPU Raytracer](https://graphics.stanford.edu/papers/gpu_kdtree/kdtree.pdf)**(Siggraph 2005)
  - *It presents kd-tree traversal algorithms kd-restart and kd-backtrack that run without a stack.*
- **[restart trail for stackless BVH traversal](https://research.nvidia.com/publication/restart-trail-stackless-bvh-traversal)** (HPG2010)
  - *In this paper, it introduces restart trail, a simple algorithmic method that makes restarts possible regardless of the type of hierarchy by storing one bit of data per level.*

### AI tech in RayTracing
- **[Learning light transport the reinforced way.](https://arxiv.org/abs/1701.07403)** (NVIDIA Research Team In ACM SIGGRAPH 2017 Talks(SIGGRAPH ’17))
  - *Combine reinforcement learning process with the ray tracing, brings up a new idea from the rendering equation.*
  - *Using neural network to do some inference to compute the ray direction instead of the importance sampling.*

### DeNoise
- **[Spatiotemporal Variance-Guided Filtering: Real-Time Reconstruction for Path-Traced Global Illumination.](https://research.nvidia.com/publication/2017-07_Spatiotemporal-Variance-Guided-Filtering%3A)** (High Performance Graphics 2017)
  - *It uses spatiotemporal variance-guided filtering to rebuild the statistic figures processed by ray tracing method*


- **[Reconstruction of Monte Carlo Image Sequences using a Recurrent Denoising Autoencoder.](https://research.nvidia.com/sites/default/files/publications/dnn_denoise_author.pdf)** (Chaitanya C R A , Kaplanyan A S , Schied C , et al. Acm Transactions on Graphics, 2017, 36(4):1-12.)
  - *A machine learning technique for reconstructing image sequences rendered using Monte Carlo methods.* 

### API about RayTracing
  - [DirectX Raytracing(DXR) Functional Spec](https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html)
  - [Ray Tracing In Vulkan - The Khronos Group](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/index.html)

## Industry Contributions
 - [NVIDIA](http://www.nvidia.com/)
   - RTX 2080/3080: PC platform for ray tracing. 
 - [Local Ray](https://www.adshir.com/)
   - Adshir's LocalRay™ is a Ray Tracing Graphics Enging scaling from High-End Gaming GPUs to mobile devices.

## BenchMarks
 - [Sponze]
 - [Cornell Boxes]

# 其他设计

[沉默的舞台剧-CSDN博客](https://blog.csdn.net/qq_35312463?type=blog)
对ray tracing 工程的总结提炼
[GPU架构与管线总结_rop fb 与crossbar-CSDN博客](https://blog.csdn.net/qq_35312463/article/details/108561115)
对GPU架构的总结
[孙小磊 - 知乎 (zhihu.com)](https://www.zhihu.com/people/sun-lei-22-19/posts?page=2)
图形学总结



# 图形渲染相关资料
学习路线和文本资料
[Gʀᴀᴘʜɪᴄs Cᴏᴅᴇx (nvidia.com)](https://graphicscodex.courses.nvidia.com/app.html?page=_rn_preface)
[Scratchapixel 4.0, Learn Computer Graphics Programming](https://www.scratchapixel.com/index.html)


[图形学渲染方向个人学习路线整理 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/445343440)
[计算机图形学（渲染方向）学习全攻略——学术科研篇_图形学blog-CSDN博客](https://blog.csdn.net/tiao_god/article/details/111146313)
[(26 封私信 / 80 条消息) 木头骨头石头 - 知乎 (zhihu.com)](https://www.zhihu.com/people/yan-mou-15/posts?page=1)

[CS 5643 Physically Based Animation](https://link.zhihu.com/?target=http%3A//www.cs.cornell.edu/courses/CS5643/2015sp/)
[[译]A trip through the Graphics Pipeline 2011 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/345153928)


GPU硬件
[深入GPU硬件架构及运行机制 - 0向往0 - 博客园 (cnblogs.com)](https://www.cnblogs.com/timlly/p/11471507.html)
[How GPU Computing Works | NVIDIA On-Demand](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31151/)

[Advanced Graphics 2021/2022 (uu.nl)](https://ics.uu.nl/docs/vakken/magr/2022-2023/index.html

对渲染管线：三种类型的解释
[实时渲染管线:（三）逻辑管线 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/440593877)


[(25 封私信 / 80 条消息) 实时光线追踪（real-time ray tracing）技术还有哪些未攻克的难题？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/310930978)


GAMES相关课程笔记，求职问题记录
[GAMES202 Lecture12：Real-Time Ray-Tracing - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/387619811)

[gfxcourses.stanford.edu/cs248/winter22](https://gfxcourses.stanford.edu/cs248/winter22)
[gfxcourses.stanford.edu/cs348b/spring22](https://gfxcourses.stanford.edu/cs348b/spring22)
[凌霄 - 知乎 (zhihu.com)](https://www.zhihu.com/people/xing-xiao-xiao-98/posts?page=2)

A trip through the Graphics Pipeline 2011

[一篇光线追踪的入门 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/41269520)
[闫令琪：Games101 现代计算机图形学-光线追踪(一)：whitted ray trace光线追踪 & 作业Assignment05解析_whittedrenderer::trace-CSDN博客](https://blog.csdn.net/weixin_39548859/article/details/107335754)
[《Real-Time Rendering 4th Edition》读书笔记--实时光线追踪 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/243061710)

介绍
[What Is Ray Tracing? (And What It Means for PC Gaming) | PCMag](https://www.pcmag.com/how-to/what-is-ray-tracing-and-what-it-means-for-pc-gaming)

非真实感渲染
[【Real-Time Rendering】非真实感渲染总结 | LycTechStack (lz328.github.io)](https://lz328.github.io/LycTechStack.github.io/2022/05/18/20220518-RTR-%E9%9D%9E%E7%9C%9F%E5%AE%9E%E6%84%9F%E6%B8%B2%E6%9F%93%E6%8A%80%E6%9C%AF%E6%80%BB%E7%BB%93/)




必读论文：
2003
Stylized Depiction in Computer Graphics - Non-Photorealistic, Painterly and Toon Rendering
[Non-Photorealistic, Painterly and 'Toon Rendering](https://link.zhihu.com/?target=https%3A//www.red3d.com/cwr/npr/)
这篇基本上收集了NPR领域的大部分重要文章。

2004
Spatio-Temporal Photon Density Estimation Using Bilateral Filtering
[http://resources.mpi-inf.mpg.de/anim/STdenest/cgi2004.pdf](https://link.zhihu.com/?target=http%3A//resources.mpi-inf.mpg.de/anim/STdenest/cgi2004.pdf)
利用Bilateral Filtering做Density Estimation本身就是挺有意思的一个想法。

Efficient GPU Screen-Space Ray Tracing
[http://jcgt.org/published/0003/04/04/paper.pdf](https://link.zhihu.com/?target=http%3A//jcgt.org/published/0003/04/04/paper.pdf)
这篇是[Screen-Space Reflection](https://www.zhihu.com/search?q=Screen-Space%20Reflection&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A533470575%7D)技术的始祖文章之一。

2005
Adaptive Numerical Cumulative Distribution Functions for Efficient Importance Sampling
[http://www.cs.columbia.edu/cg/pdfs/45_cdf.pdf](https://link.zhihu.com/?target=http%3A//www.cs.columbia.edu/cg/pdfs/45_cdf.pdf)
这篇利用了曲线压缩的方法来进行复杂函数的Importance Sampling，很有意思。

2006
Correlated Visibility Sampling for Direct Illumination
[https://vccimaging.org/Publications/Ghosh2006UCV/Ghosh2006UCV.pdf](https://link.zhihu.com/?target=https%3A//vccimaging.org/Publications/Ghosh2006UCV/Ghosh2006UCV.pdf)
这篇文章考虑了直接光的Visibility的Importance Sampling

2007
Soft Shadows by Ray Tracing Multilayer Transparent Shadow Maps
[http://www.tabellion.org/et/paper07/egsr07_mtsm.pdf](https://link.zhihu.com/?target=http%3A//www.tabellion.org/et/paper07/egsr07_mtsm.pdf)
有意思的地方是结合了Ray Tracing和Shadow Map

  

Global Illumination using Photon Ray Splatting
[http://dcgi.felk.cvut.cz/home/havran/ARTICLES/herzog07EG.pdf](https://link.zhihu.com/?target=http%3A//dcgi.felk.cvut.cz/home/havran/ARTICLES/herzog07EG.pdf)
这篇文章为Photon Density Estimation提供了一种新的思路。

2008
Raytracing Prefiltered Occlusion for Aggregate Geometry
[http://www.cs.utah.edu/~lacewell/prefilter-rt08/prefilter.pdf](https://link.zhihu.com/?target=http%3A//www.cs.utah.edu/~lacewell/prefilter-rt08/prefilter.pdf)
这篇其实对于大场景的处理很实用，通过Filtering来减少Visibility带来的噪点。

  

2009
Understanding the Efficiency of Ray Traversal on GPUs
[https://www.nvidia.com/docs/IO/76976/HPG2009-Trace-Efficiency.pdf](https://link.zhihu.com/?target=https%3A//www.nvidia.com/docs/IO/76976/HPG2009-Trace-Efficiency.pdf)
GPU[光线追踪](https://www.zhihu.com/search?q=%E5%85%89%E7%BA%BF%E8%BF%BD%E8%B8%AA&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A533470575%7D)的始祖文章，深入分析了GPU上光线追踪操作的效率问题。

  

Building with Bricks: CUDA-based GigaVoxel Rendering
[(PDF) Building with Bricks: CUDA-based GigaVoxel Rendering](https://link.zhihu.com/?target=https%3A//www.researchgate.net/publication/47863824_Building_with_Bricks_CUDA-based_GigaVoxel_Rendering)
这篇讲了如何突破显存限制，渲染超大体素，很有意思。

Accelerating Shadow Rays Using Volumetric Occluders and Modified kd-Tree Traversal
[https://www.highperformancegraphics.org/previous/www_2009/presentations/djeu-accelerating.pdf](https://link.zhihu.com/?target=https%3A//www.highperformancegraphics.org/previous/www_2009/presentations/djeu-accelerating.pdf)
这篇为简化复杂场景的[Shadow Ray](https://www.zhihu.com/search?q=Shadow%20Ray&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A533470575%7D)计算提供了一种思路。


Real-Time O(1) Bilateral Filtering
[http://vision.ai.illinois.edu/publications/yang_cvpr09.pdf](https://link.zhihu.com/?target=http%3A//vision.ai.illinois.edu/publications/yang_cvpr09.pdf)
O(1)复杂度的实时Bilateral Filtering，不需要解释了吧。

  
2010
Parallel Progressive Photon Mapping on GPUs
[https://www.ci.i.u-tokyo.ac.jp/~hachisuka/gpuppm_slides.pdf](https://link.zhihu.com/?target=https%3A//www.ci.i.u-tokyo.ac.jp/~hachisuka/gpuppm_slides.pdf)
GPU上实现并行的Progressive Photon Mapping，也是这个分支方向的始祖文章之一。

2011
  Intel OpenCL Implicit Vectorization Module
[http://llvm.org/devmtg/2011-11/Rotem_IntelOpenCLSDKVectorizer.pdf](https://link.zhihu.com/?target=http%3A//llvm.org/devmtg/2011-11/Rotem_IntelOpenCLSDKVectorizer.pdf)
和图形看似没有直接关系，但是详细讲了Intel如何从编译器和驱动层面进行Vectorization，对于理解并区别GPU和CPU的执行模型很有帮助。

  
Stratified Sampling for Stochastic Transparency
[https://research.nvidia.com/sites/default/files/pubs/2011-06_Stratified-Sampling-for/laine2011egsr_paper.pdf](https://link.zhihu.com/?target=https%3A//research.nvidia.com/sites/default/files/pubs/2011-06_Stratified-Sampling-for/laine2011egsr_paper.pdf)
Stratified Sampling在Stochastic Transparency的一个应用。

  

2012
OptiX Out-of-Core and CPU Rendering
[http://on-demand.gputechconf.com/gtc/2012/presentations/S0366-Optix-Out-of-Core-and-Cpu-Rendering.pdf](https://link.zhihu.com/?target=http%3A//on-demand.gputechconf.com/gtc/2012/presentations/S0366-Optix-Out-of-Core-and-Cpu-Rendering.pdf)

介绍了OptiX是如何支持Out-of-Core以及CPU渲染的。

  

Axis-Aligned Filtering for Interactively Sampled Soft Shadows
[http://graphics.berkeley.edu/papers/UdayMehta-AAF-2012-12/UdayMehta-AAF-2012-12.pdf](https://link.zhihu.com/?target=http%3A//graphics.berkeley.edu/papers/UdayMehta-AAF-2012-12/UdayMehta-AAF-2012-12.pdf)
如何通过Filtering去除软阴影的噪点。

  

Irradiance Volumes for Games
[https://developer.amd.com/wordpress/media/2012/10/Tatarchuk_Irradiance_Volumes.pdf](https://link.zhihu.com/?target=https%3A//developer.amd.com/wordpress/media/2012/10/Tatarchuk_Irradiance_Volumes.pdf)
这篇算是Irradiance Volume的始祖级文章，虽然技术有点过时，但是思路可以借鉴。

Mixed Resolution Rendering
[https://developer.amd.com/wordpress/media/2012/10/ShopfMixedResolutionRendering.pdf](https://link.zhihu.com/?target=https%3A//developer.amd.com/wordpress/media/2012/10/ShopfMixedResolutionRendering.pdf)
主要讲如何有效的进行Upsampling，里面的Tricks在今天的GPU[实时渲染](https://www.zhihu.com/search?q=%E5%AE%9E%E6%97%B6%E6%B8%B2%E6%9F%93&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A533470575%7D)领域还是很常用。


Metropolis Sampling
[http://cgg.mff.cuni.cz/~jaroslav/teaching/2012-npgr031/slides/npgr031-2012%20-%2007%20-%20metropolis.pdf](https://link.zhihu.com/?target=http%3A//cgg.mff.cuni.cz/~jaroslav/teaching/2012-npgr031/slides/npgr031-2012%2520-%252007%2520-%2520metropolis.pdf)
介绍了一堆Metropolis相关的技术实现和Tricks，对于需要具体实现的时候很有帮助。

Importance Sampling Techniques for Path Tracing in Participating Media
[https://www.arnoldrenderer.com/research/egsr2012_volume.pdf](https://link.zhihu.com/?target=https%3A//www.arnoldrenderer.com/research/egsr2012_volume.pdf)
SPI和Solid Angle合作开发的Arnold中的[体渲染技术](https://www.zhihu.com/search?q=%E4%BD%93%E6%B8%B2%E6%9F%93%E6%8A%80%E6%9C%AF&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A533470575%7D)。

  

  

2013
 Megakernels Considered Harmful: Wavefront Path Tracing on GPUs
[http://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15869-f11/www/readings/laine13_megakernels.pdf](https://link.zhihu.com/?target=http%3A//www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15869-f11/www/readings/laine13_megakernels.pdf)
文章比较啰嗦，但是详细解释了GPU的各种性能瓶颈，在处理[Path Tracing](https://www.zhihu.com/search?q=Path%20Tracing&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A533470575%7D)的复杂材质时候的体现。  

Path Space Regularization for Holistic and Robust Light Transport
[http://users-cs.au.dk/toshiya/starpm2013a/PSR_SIGAsia2013.pdf](https://link.zhihu.com/?target=http%3A//users-cs.au.dk/toshiya/starpm2013a/PSR_SIGAsia2013.pdf)
这篇只用了很简单的代码改动，就可以有效的渲染Caustics这类比较难的路径，虽然有偏，但是非常实用和易于集成。

  

2014
Conservative Morphological Anti-Aliasing (CMAA)
[Conservative Morphological Anti-Aliasing (CMAA) - March 2014 Update](https://link.zhihu.com/?target=https%3A//software.intel.com/en-us/articles/conservative-morphological-anti-aliasing-cmaa-update)
Intel搞出来的一个实时AA技术，号称比FXAA效果好。

  

High Quality Temporal Supersampling
[https://de45xmedrsdbp.cloudfront.net/Resources/files/TemporalAA_small-59732822.pdf](https://link.zhihu.com/?target=https%3A//de45xmedrsdbp.cloudfront.net/Resources/files/TemporalAA_small-59732822.pdf
详细讲解了UE4里面如何实现Temporal AA

  

Photon Differential Splatting for Rendering Caustics
[http://www.imm.dtu.dk/~jerf/papers/diffsplat_lowres.pdf](https://link.zhihu.com/?target=http%3A//www.imm.dtu.dk/~jerf/papers/diffsplat_lowres.pdf)
为[渲染Caustics](https://www.zhihu.com/search?q=%E6%B8%B2%E6%9F%93Caustics&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A533470575%7D)提供了另一种理论框架。

  

2015
Randomized Redundant DCT: Efficient Denoising by Using Random Subsampling of DCT Patches
[https://fukushima.web.nitech.ac.jp/paper/2015_siggraph_asia_tb_fujita.pdf](https://link.zhihu.com/?target=https%3A//fukushima.web.nitech.ac.jp/paper/2015_siggraph_asia_tb_fujita.pdf)
三星搞出来的一个照片降噪技术，亲测速度确实超快。


Portal-Masked Environment Map Sampling
[https://cs.dartmouth.edu/~wjarosz/publications/bitterli15portal.pdf](https://link.zhihu.com/?target=https%3A//cs.dartmouth.edu/~wjarosz/publications/bitterli15portal.pdf)
解决了Portal Light和Environment Map共同存在的时候的Importance Sampling问题。

  

Computer Graphics III – Approximate global illumination [computation](https://www.zhihu.com/search?q=computation&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A533470575%7D)
[http://cgg.mff.cuni.cz/~jaroslav/teaching/2015-npgr010/slides/12%20-%20npgr010-2015%20-%20IC-PBGI.pdf](https://link.zhihu.com/?target=http%3A//cgg.mff.cuni.cz/~jaroslav/teaching/2015-npgr010/slides/12%2520-%2520npgr010-2015%2520-%2520IC-PBGI.pdf)
这篇是一篇综述，讲了多种主要的有偏全局照明方法。

  

2016  
Temporal Reprojection Anti-Aliasing in INSIDE
[http://twvideo01.ubm-us.net/o1/vault/gdc2016/Presentations/Pedersen_LasseJonFuglsang_TemporalReprojectionAntiAliasing.pdf](https://link.zhihu.com/?target=http%3A//twvideo01.ubm-us.net/o1/vault/gdc2016/Presentations/Pedersen_LasseJonFuglsang_TemporalReprojectionAntiAliasing.pdf)
讲了手游INSIDE里面Temporal AA的实现，效果确实很不错。

  
Aggregate G-Buffer Anti-Aliasing in Unreal Engine 4
[http://advances.realtimerendering.com/s2016/AGAA_UE4_SG2016_6.pdf](https://link.zhihu.com/?target=http%3A//advances.realtimerendering.com/s2016/AGAA_UE4_SG2016_6.pdf)
NVIDIA给UE4搞出来的一种新的Temporal AA实现。

  

Advanced Graphics – GPU Ray Tracing
[http://www.cs.uu.nl/docs/vakken/magr/2015-2016/slides/lecture%2012%20-%20GPU%20ray%20tracing%20%282%29.pdf](https://link.zhihu.com/?target=http%3A//www.cs.uu.nl/docs/vakken/magr/2015-2016/slides/lecture%252012%2520-%2520GPU%2520ray%2520tracing%2520%25282%2529.pdf)
强烈推荐，讲了各种GPU光线追踪的重要效率问题。

  

Nonlinearly Weighted First-order Regression for Denoising Monte Carlo Renderings
[https://cs.dartmouth.edu/~wjarosz/publications/bitterli16nonlinearly.pdf](https://link.zhihu.com/?target=https%3A//cs.dartmouth.edu/~wjarosz/publications/bitterli16nonlinearly.pdf)
这篇基本上是基于经验的[渲染降噪技术](https://www.zhihu.com/search?q=%E6%B8%B2%E6%9F%93%E9%99%8D%E5%99%AA%E6%8A%80%E6%9C%AF&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A533470575%7D)的最好成果，最大的问题是速度太慢。

  

Photon Splatting Using a View-Sample Cluster Hierarchy
[http://fileadmin.cs.lth.se/graphics/research/papers/2016/splatting/MSKAD16.pdf](https://link.zhihu.com/?target=http%3A//fileadmin.cs.lth.se/graphics/research/papers/2016/splatting/MSKAD16.pdf)
基于Splatting和Cluster，提供了一种实时GI的新思路。

  

Real-Time Global Illumination Using Precomputed Illuminance Composition
with Chrominance Compression
[http://jcgt.org/published/0005/04/02/paper-lowres.pdf](https://link.zhihu.com/?target=http%3A//jcgt.org/published/0005/04/02/paper-lowres.pdf)
又是一种实时GI的实现方法。

  

Real-Time Rendering of Deformable Heterogeneous Translucent Objects using Multiresolution Splatting

[https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/a0e196dc4da7b47198d3c77f5a83b768fd7a.pdf](https://link.zhihu.com/?target=https%3A//www.microsoft.com/en-us/research/wp-content/uploads/2016/12/a0e196dc4da7b47198d3c77f5a83b768fd7a.pdf)

一种实时的半透材质的渲染方法。

  

2017
SSRTGI: Toughest Challenge in Real-Time 3D
[Toughest Challenge in Real-Time 3D](https://link.zhihu.com/?target=https%3A//80.lv/articles/ssrtgi-toughest-challenge-in-real-time-3d/)
Unigine引擎的SSRTGI算法的一个介绍（但是效果其实比较有限，没有解决Far-field）

  

Vectorized Production Path Tracing
[https://research.dreamworks.com/wp-content/uploads/2018/07/Vectorized_Production_Path_Tracing_DWA_2017.pdf](https://link.zhihu.com/?target=https%3A//research.dreamworks.com/wp-content/uploads/2018/07/Vectorized_Production_Path_Tracing_DWA_2017.pdf)
这篇基本上是集大成的产品化的一个东西，[梦工厂](https://www.zhihu.com/search?q=%E6%A2%A6%E5%B7%A5%E5%8E%82&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A533470575%7D)完全利用SIMD实现的Path Tracing渲染器，包含了对Ray Tracing和Shading的向量化。

  

Area-preserving parameterizations for spherical [ellipses](https://www.zhihu.com/search?q=ellipses&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A533470575%7D)
[https://www.arnoldrenderer.com/research/egsr2017_spherical_ellipse.pdf](https://link.zhihu.com/?target=https%3A//www.arnoldrenderer.com/research/egsr2017_spherical_ellipse.pdf)
这篇是Arnold里面的一个对于Disk Light进行更有效的Solid Angle Sampling的方法。
 

2018
Efficient Caustic Rendering with Lightweight Photon Mapping
[http://cgg.mff.cuni.cz/~jaroslav/papers/2018-lwpm/2018-grittmann-lwpm-paper.pdf](https://link.zhihu.com/?target=http%3A//cgg.mff.cuni.cz/~jaroslav/papers/2018-lwpm/2018-grittmann-lwpm-paper.pdf)
为Caustics渲染提供了新的思路。
 

Shiny Pixels and Beyond: Real-Time Raytracing at SEED
[https://media.contentapi.ea.com/content/dam/ea/seed/presentations/gdc2018-seed-shiny-pixels-and-beyond-real-time-](https://link.zhihu.com/?target=https%3A//media.contentapi.ea.com/content/dam/ea/seed/presentations/gdc2018-seed-shiny-pixels-and-beyond-real-time-)
raytracing-at-seed.pdf

讲了基于NVIDIA RTX的实时面积阴影、实时反射、GI等的渲染，很有启发性。

  

# AI提供的资料

找到了第 1 篇资料：[英伟达发布 DLSS 3.5：加入光线重建技术，提升光线追踪 ...](https://www.ithome.com/0/714/126.htm)

找到了第 2 篇资料：[RTX阵容更壮大！500款游戏已支持光线追踪、DLSS和AI ...](https://new.qq.com/rain/a/20231211A07HR300)

找到了第 3 篇资料：[NVIDIA DLSS 3.5：借助 AI 提升光线追踪；“心灵杀手 2 ...](https://www.nvidia.cn/geforce/news/nvidia-dlss-3-5-ray-reconstruction/)

找到了第 4 篇资料：[NVIDIA DLSS 深度学习超级采样 (DLSS) 技术 | NVIDIA](https://www.nvidia.cn/geforce/technologies/dlss/)

找到了第 5 篇资料：[【硬核科普】免费提升画质和帧数？详解DLSS2.0的工作 ...](https://www.bilibili.com/video/BV1PA41187g2/)

一共找到 5 篇资料，正在阅读...

找到了 5 篇资料作为参考：

DLSS（Deep Learning Super Sampling）是NVIDIA推出的一项利用人工智能技术来提升图形渲染性能和图像质量的技术。DLSS结合了AI技术和光线追踪，以实现在保持或提升图像质量的同时提高游戏的帧率。以下是DLSS如何将AI技术和光线追踪相结合的详细过程：

1. **AI训练**：NVIDIA使用超级计算机对DLSS AI模型进行训练。这些模型通过分析大量的高分辨率图像和它们的低分辨率对应图像来学习如何从低分辨率图像中重建高质量的细节5。
    
2. **Tensor Cores**：DLSS利用GeForce RTX GPU上的Tensor Cores来进行AI计算。Tensor Cores是专门设计用于高效处理深度学习矩阵计算的硬件核心，它们能够加速DLSS AI模型的运行6。
    
3. **光线追踪的集成**：在支持光线追踪的游戏中，DLSS可以与光线追踪技术一起工作。光线追踪负责生成逼真的光照和阴影效果，而DLSS则用于提升这些效果的渲染效率和性能2。
    
4. **图像重建**：DLSS通过AI算法对游戏中的每个像素进行分析和重建，生成额外的像素来填补图像的细节。这种方法可以在不牺牲图像质量的前提下，提高图像的分辨率和清晰度15。
    
5. **性能提升**：通过DLSS，游戏可以在较低的原生分辨率下运行，然后使用AI技术将图像上采样到更高的分辨率。这样不仅减少了GPU的计算负担，还能在保持或提升图像质量的同时提高帧率17。
    
6. **光线重建技术**：DLSS 3.5引入了光线重建技术，这是进阶版AI驱动的神经渲染器的一部分。光线重建技术通过NVIDIA超级计算机训练的AI网络，在采样光线之间生成更高质量的像素，进一步提升了光线追踪图像的质量5。
    
7. **实时优化**：DLSS能够实时分析游戏场景的需求，并根据不同的硬件配置和性能要求，智能调整图像渲染和游戏性能，提供更加流畅和稳定的游戏体验2。
    

通过上述方式，DLSS成功地将AI技术与光线追踪相结合，为玩家提供了更加逼真和流畅的游戏体验，同时确保了高性能和高保真的图形输出。随着技术的不断进步和优化，DLSS将继续在提升图形渲染性能方面发挥重要作用27。


找到了第 1 篇资料：[实时光线追踪技术：业界发展近况与未来挑战 - 知乎](https://sunocean.life/blog/backup/2021-01-11-shader-OpenGL-Lighting-Advanced-RayTracer-Advance.md/zhuanlan.zhihu.com/8582b378.html)

找到了第 2 篇资料：[业界广泛采用光线追踪和NVIDIA DLSS，NVIDIA RTX游戏 ...](https://blogs.nvidia.cn/2021/06/01/nvidia-rtx-titles-jump-to-130-on-widespread-industry-adoptionof-ray-tracing-nvidia-dlss/)

找到了第 3 篇资料：[科普 | 高帧率、好画质的「光追」是如何实现的？ - 少数派](https://sspai.com/post/83522#!)

找到了第 4 篇资料：[追光十二载——NVIDIA光线追踪技术杂谈 | 机核 GCORES](https://www.gcores.com/articles/102943)

一共找到 4 篇资料，正在阅读...

找到了 4 篇资料作为参考：

光线追踪技术已经被应用到多个行业中，并且随着时间的推移，其应用范围和深度都在不断扩展。以下是一些主要行业中光线追踪的应用情况及其引入时间：

1. **电影和视觉效果行业**： 光线追踪技术最早被广泛应用于电影和视觉效果行业中，用于生成逼真的CGI（计算机生成图像）和特效。早在20世纪70年代，光线追踪技术就被提出，并在80年代开始被用于电影制作中4。例如，1982年的迪士尼电影《创战纪》中就已经能看到光线追踪技术的应用3。
    
2. **视频游戏行业**： 随着计算机图形学的发展和硬件性能的提升，光线追踪技术开始向视频游戏行业渗透。2018年，NVIDIA推出了基于Turing架构的GeForce RTX系列GPU，这标志着实时光线追踪技术正式进入主流游戏市场4。随后，越来越多的游戏开始支持光线追踪，如《战地5》、《使命召唤：现代战争》等2。
    
3. **建筑可视化和设计行业**： 光线追踪也被用于建筑可视化和设计领域，帮助设计师和建筑师创建逼真的三维模型和渲染图。通过光线追踪技术，可以更真实地模拟光线在建筑空间中的行为，从而提供更加准确的视觉表现3。  
    
4. **工业设计和汽车行业**： 在工业设计和汽车行业中，光线追踪技术可以用来创建产品的逼真渲染图，帮助设计师在产品制造之前进行视觉评估和修改。这种技术的应用可以减少原型设计的成本和时间1。
    
5. **虚拟现实（VR）和增强现实（AR）**： 随着VR和AR技术的发展，光线追踪在这些领域中也展现出巨大的潜力。通过实时光线追踪，可以为用户创造更加真实和沉浸式的虚拟体验2。
    

光线追踪技术的引入和应用，不仅极大地提升了视觉效果的真实感和质量，也为各个行业带来了新的设计和表现方法。随着技术的不断进步和优化，预计光线追踪将继续在更多领域发挥重要作用，并为用户带来更加丰富和逼真的视觉体验。


找到了第 1 篇资料：[第十五章 全局光照](https://vcl.pku.edu.cn/course/vci/notes/15-global-illumination-notes.pdf)

找到了第 2 篇资料：[国家科技期刊开放平台](https://coaa.istic.ac.cn/openJournal/periodicalArticle/0120230300380099)

找到了第 3 篇资料：[实时光线追踪相关研究综述 - 百度学术](https://xueshu.baidu.com/usercenter/paper/show?paperid=1m4502u0gb3c0tx02h4b02k0vu574842)

找到了第 4 篇资料：[光线追踪技术综述 - 百度学术](https://xueshu.baidu.com/usercenter/paper/show?paperid=8daa765f5ccfa3a64e0cf0d33293ac94)

一共找到 4 篇资料，正在阅读...

找到了 4 篇资料作为参考：

光线追踪技术自20世纪70年代提出以来，已经经历了数十年的发展。以下是一些重要的光线追踪相关论文，按照时间顺序排列，并对每篇论文的主要内容进行简要介绍：

1. **1979年** - **L. Williams** - "A Note on the Rendering Equation" [[1](https://sunocean.life/blog/backup/2021-01-11-shader-OpenGL-Lighting-Advanced-RayTracer-Advance.md)]
    
    - 这篇论文提出了渲染方程的概念，为光线追踪的理论基础奠定了重要基石。论文中讨论了如何通过光线追踪来模拟真实世界中的光照效果。
2. **1980年代** - **Whitted** - "An Improved Illumination Model for Shaded Display" [[2](https://vcl.pku.edu.cn/course/vci/notes/15-global-illumination-notes.pdf)]
    
    - Whitted风格的光线追踪算法是最早的光线追踪算法之一，它引入了反射和透明度的概念，为后续的全局光照算法提供了基础。
3. **1990年** - **徐汝彪，刘慎权** - "光线追踪技术综述" [[13](https://xueshu.baidu.com/usercenter/paper/show?paperid=8daa765f5ccfa3a64e0cf0d33293ac94)]
    
    - 该论文对光线追踪技术的基本思想和典型算法进行了概述，强调了光线追踪在产生三维真实图像方面的潜力和重要性。
4. **2004年** - **K. Egido** - "Real-Time Ray Tracing on Commodity Hardware" [[未提供具体文献]]
    
    - 该论文探讨了实时光线追踪技术在普通硬件上的实现可能性，为后续的实时光线追踪技术发展提供了方向。
5. **2013年** - **N. Green** - "Real-Time Ray Tracing and Its Applications" [[未提供具体文献]]
    
    - 这篇论文讨论了实时光线追踪技术的应用前景，特别是在游戏和电影行业中的潜在用途。
6. **2018年** - **NVIDIA** - "Real-Time Ray Tracing on NVIDIA RTX" [[8](https://xueshu.baidu.com/usercenter/paper/show?paperid=1m4502u0gb3c0tx02h4b02k0vu574842)]
    
    - NVIDIA在其RTX系列产品中引入了实时光线追踪技术，这篇论文详细介绍了RTX架构如何实现实时光线追踪，标志着光线追踪技术的一个重要里程碑。
7. **2020年** - **闫润等** - "实时光线追踪相关研究综述" [[9](https://xueshu.baidu.com/usercenter/paper/show?paperid=1m4502u0gb3c0tx02h4b02k0vu574842)]
    
    - 这篇综述性文章总结了实时光线追踪的研究现状，包括算法优化、GPU加速、定制硬件等方面的进展，并对未来的发展方向进行了展望。

这些论文不仅展示了光线追踪技术的历史发展，也反映了该领域不断进步和创新的历程。随着技术的不断成熟和硬件性能的提升，光线追踪在图形渲染领域的应用将越来越广泛。


找到了第 1 篇资料：[英伟达正努力研究提高 GPU 光追性能：提升幅度最高可达 20%](https://www.ithome.com/0/600/692.htm)

找到了第 2 篇资料：[第十五章 全局光照](https://vcl.pku.edu.cn/course/vci/notes/15-global-illumination-notes.pdf)

找到了第 3 篇资料：[实时光线追踪相关研究综述](http://fcst.ceaj.org/CN/10.3778/j.issn.1673-9418.2207067)

找到了第 4 篇资料：[中国科学院云南天文台机构知识库(YNAO OpenIR): 光线追踪 ...](http://ir.ynao.ac.cn/handle/114a53/5093)

一共找到 4 篇资料，正在阅读...

找到了 4 篇资料作为参考：

光线追踪技术与GPU硬件的结合是近年来计算机图形学领域的一个重要研究方向。以下是一些关于光线追踪和GPU硬件的论文，它们涵盖了理论综述、技术创新、性能优化等多个方面：

1. **NVIDIA正努力研究提高GPU光追性能** 1
    
    - 该论文讨论了NVIDIA在提高GPU光追性能方面的最新研究成果。研究指出，通过一种名为“GPUSubwarpInterleaving”的技术，实时光线追踪的效率平均提高了6.8%，最佳案例结果提高了20%。这项技术通过解决GPU在进行大规模并行运算时可能出现的损耗来提升光追性能。然而，这项技术需要对GPU硬件的架构级别进行更改才能实现。
2. **实时光线追踪相关研究综述** 3
    
    - 这些论文提供了实时光线追踪技术的全面回顾和总结。论文首先介绍了光线追踪的基本概念、算法和加速数据结构，然后对比了三款支持光线追踪的商用图形处理器（GPU）之间的差异。文章还从多个角度综述了光线追踪的算法优化工作，包括光线束遍历、无栈遍历、光线重排序、多分支BVH、降噪技术，以及与神经网络结合的实时光线追踪方法。最后，论文对使用GPU优化加速和采用定制化设计的硬件加速进行了归纳分析，并对未来的发展方向进行了展望。
3. **第十五章 全局光照** 2
    
    - 这些论文深入探讨了全局光照的概念，并详细阐述了Whitted风格的光线追踪算法，这是最简单的光线追踪算法形式，对后续的研究有着重要的影响。论文还介绍了光线投射和光线追踪的基本原理，以及如何处理光线与物体的交点计算。此外，论文还讨论了加速结构，如层次包围盒（BVH）和KD树，以及它们在优化光线追踪性能方面的应用。
4. **光线追踪的求解新方法及其应用** 4
    
    - 这篇论文探讨了在Kerr时空中快速计算光子和静止质量非零粒子的测地线轨道的新方法。这对于天文研究具有重要意义，因为它不仅能够帮助理解黑洞周围吸积物质的运行轨迹，而且是物质辐射转移的路径。论文基于Carlson的椭圆积分方法，发展了一套计算Kerr时空中光子测地线的Fortran程序，并讨论了其测试和简单应用。

这些论文不仅展示了光线追踪技术与GPU硬件结合的最新进展，也为研究人员提供了丰富的理论基础和实践指导。随着技术的不断发展，我们可以期待在未来看到更多创新的研究成果。



开始获取资料...

找到了第 1 篇资料：[实时渲染管线:（一）基本概念与发展简史 - 知乎](https://zhuanlan.zhihu.com/p/440584180)

找到了第 2 篇资料：[第十五章 全局光照](https://vcl.pku.edu.cn/course/vci/notes/15-global-illumination-notes.pdf)

找到了第 3 篇资料：[GPU 渲染管线与着色器 大白话总结 ---- 一篇就够 - CSDN博客](https://blog.csdn.net/newchenxf/article/details/119803489)

找到了第 4 篇资料：[图形重塑：光线追踪、人工智能和可着色编程所带来的全新 ...](https://www.nvidia.cn/geforce/news/graphics-reinvented-new-technologies-in-rtx-graphics-cards/)

找到了第 5 篇资料：[计算机图形学入门：什么是光线追踪？_计算机图形学中的ray ...](https://blog.csdn.net/wubaohu1314/article/details/120381004)

找到了第 6 篇资料：[NVIDIA 发布全球首款实时光线追踪 GPU —— GeForce RTX](https://blogs.nvidia.cn/blog/gamescom-rtx-turing-real-time-ray-tracing/)

找到了第 7 篇资料：[计算机图形学项目二：GPU光线追踪 - CSDN博客](https://blog.csdn.net/GarsDong/article/details/118493509)

找到了第 8 篇资料：[计算机图形学 (豆瓣)](https://book.douban.com/subject/1392483/)

找到了第 9 篇资料：[计算机图形学_四川大学_中国大学MOOC(慕课)](https://www.icourse163.org/learn/-1460184167)

找到了第 10 篇资料：[Vulkan_Ray Tracing 05_光线追踪管线_hitattributeext-CSDN博客](https://blog.csdn.net/qq_35312463/article/details/120899739)

一共找到 10 篇资料，正在阅读...

找到了 10 篇资料作为参考：

在计算机图形学和实时渲染中，光线追踪是一种先进的渲染技术，它通过模拟光线在虚拟场景中的传播来生成图像，从而实现非常逼真的视觉效果。光线追踪在GPU的着色管线中实现的过程大致如下：

1. **光线投射（Ray Casting）**：这是光线追踪的基础，涉及从观察者（摄像机）的位置向屏幕的每个像素发射光线，并计算光线与场景中物体的交点12。
    
2. **着色器编程**：光线追踪通常需要可编程的着色器，尤其是当涉及到复杂的光照和材质效果时。开发者可以使用如Direct3D 12、Vulkan或OpenGL等现代图形API中的着色器语言（如GLSL或HLSL）来编写自定义的光线追踪着色器110。
    
3. **光线与物体的交点计算**：在光线追踪中，需要计算光线与场景中物体的交点。这通常涉及到数学上的几何交点检测算法，如Möller-Trumbore算法等2。
    
4. **光照和材质计算**：当光线与物体相交后，需要根据物体的材质属性和光源属性计算交点的颜色。这通常涉及到各种光照模型和材质着色模型的应用2。
    
5. **反射和折射**：光线追踪还可以模拟光线的反射和折射效果，这需要递归地进行光线投射，并计算反射或折射光线与场景中物体的交点2。
    
6. **加速结构**：为了提高效率，会使用各种加速结构来减少光线与物体交点计算的数量，如层次包围盒（Bounding Volume Hierarchies，BVH）等2。
    
7. **管线集成**：现代GPU架构，如NVIDIA的Turing架构，提供了专门的硬件支持来加速光线追踪的计算，包括专用的光线追踪核心（RT Core）16。
    
8. **API和框架支持**：Direct3D 12和Vulkan等图形API提供了对光线追踪的原生支持，通过扩展如DirectX Raytracing (DXR)和Vulkan的光线追踪扩展，开发者可以更容易地在GPU上实现光线追踪17。
    
9. **性能优化**：由于光线追踪计算密集，需要通过各种技术来优化性能，如使用GPU的并行计算能力、优化着色器代码、使用合适的数据结构和算法等15。
    

光线追踪技术在实时渲染中的应用正变得越来越广泛，尤其是在需要高真实感渲染的场合，如高端游戏、电影制作和虚拟现实等领域。随着GPU硬件和图形API的不断发展，光线追踪的实时渲染将变得更加高效和普及。


正在尝试为您在互联网搜索相关资料...

开始获取资料...

找到了第 1 篇资料：[GDC2024-Snowdrop中的光线追踪：主机的照明管线优化 - 知乎](https://zhuanlan.zhihu.com/p/693920769)

找到了第 2 篇资料：[图形API和GPU光线追踪分析 - 吴建明wujianming - 博客园](https://www.cnblogs.com/wujianming-110117/p/17863587.html)

找到了第 3 篇资料：[光线追踪性能指南（一） | 电子创新网 Imgtec 社区](https://imgtec.eetrend.com/blog/2024/100579001.html)

找到了第 4 篇资料：[Vulkan_Ray Tracing 05_光线追踪管线_hitattributeext-CSDN博客](https://blog.csdn.net/qq_35312463/article/details/120899739)

找到了第 5 篇资料：[NVIDIA Turing 架构深度介绍 - NVIDIA 技术博客](https://developer.nvidia.com/zh-cn/blog/nvidia-turing-architecture-in-depth/)

找到了第 6 篇资料：[使用 NVIDIA RTX 光线跟踪的最佳实践（更新）](https://developer.nvidia.com/zh-cn/blog/best-practices-for-using-nvidia-rtx-ray-tracing-updated/)

找到了第 7 篇资料：[NVIDIA RTX 光线追踪 | NVIDIA 开发者](https://developer.nvidia.cn/rtx/ray-tracing)

找到了第 8 篇资料：[优化光线跟踪的实用技巧 - NVIDIA 技术博客](https://developer.nvidia.com/zh-cn/blog/practical-tips-for-optimizing-ray-tracing/)

找到了第 9 篇资料：[光线追踪 - Vulkan入门精要](https://fuxiii.github.io/Essentials.of.Vulkan/Literature/Vulkan-GuideRayTracing.html)

找到了第 10 篇资料：[深入解读IMG CXT GPU：业界首个移动端光线追踪架构 ...](https://fpga.eetrend.com/content/2021/100556087.html)

一共找到 10 篇资料，正在阅读...

找到了 10 篇资料作为参考：

光线追踪管线是一种用于实现实时光线追踪渲染的架构设计，它与传统的光栅化渲染管线不同，主要利用硬件加速来模拟光线在虚拟场景中的传播和交互。光线追踪管线的架构设计通常包括以下几个关键组件：

1. **加速结构（Acceleration Structure）**：这是光线追踪管线的核心，用于存储场景中的所有几何物体信息，并在GPU内提供物体遍历、相交测试、光线构造等的加速算法2。
    
2. **光线生成（Ray Generation）**：这个阶段是光线追踪的起点，负责生成原始的光线，通常是从摄像机或观察者的位置发出的射线2。
    
3. **光线与物体相交（Intersection）**：当光线与场景中的物体相交时，会调用相交着色器来确定光线与物体的交点2。
    
4. **任意命中（Any Hit）**：这个阶段可以在光线与物体相交后调用，用于处理光线与物体相交的任意位置，通常用于实现透明度效果或修改光线路径2。
    
5. **最近命中（Closest Hit）**：当光线与物体相交并且是最近的命中时，会调用这个着色器来处理像素着色，如材质、纹理查找、光照计算等2。
    
6. **未命中（Miss）**：如果光线在场景中没有找到任何交点，会调用未命中着色器，通常用于处理光线没有击中任何物体的情况，如渲染背景或者天空盒2。
    
7. **着色器绑定表（Shader Binding Table, SBT）**：描述了shader与场景的哪个物体关联，并包含了shader中涉及的所有资源（纹理、buffer、常量等）2。
    
8. **光线追踪管线状态（Ray Tracing Pipeline State）**：允许指定多组光线追踪着色器和纹理资源，使得应用程序可以让光线交点使用它们所击中的材质的正确纹理运行正确的着色器代码2。
    
9. **DispatchRays**：是光线追踪管线中提交光线追踪工作到GPU的命令，它会调用光线生成着色器开始光线追踪过程2。
    

光线追踪管线的设计允许开发者利用GPU的并行处理能力来高效地计算光线与场景中物体的交互，从而实现逼真的渲染效果。随着技术的发展，光线追踪管线已经成为现代图形渲染中的一个重要方向，被广泛应用于游戏、电影制作、虚拟现实等领域。


光线追踪管线、计算管线和光栅化管线是现代图形渲染中的三种不同类型的管线，它们可以结合在一起，以利用各自的优势，实现高效且高质量的图形渲染。以下是它们如何结合在一起的概述：

1. **光栅化管线（Rasterization Pipeline）**：
    
    - 这是传统的渲染管线，用于处理大量几何细节和纹理映射，将3D模型转换为2D像素。
    - 它非常适合处理具有复杂几何形状和纹理的场景。
2. **光线追踪管线（Ray Tracing Pipeline）**：
    
    - 光线追踪管线用于模拟光线在虚拟场景中的传播和交互，提供逼真的反射、折射、阴影和全局光照效果。
    - 它通常用于渲染那些光栅化管线难以高效实现的视觉效果，如精确的反射和折射。
3. **计算管线（Compute Pipeline）**：
    
    - 计算管线用于执行通用计算任务，可以用于物理模拟、后处理效果、数据分析等。
    - 在图形渲染中，它可以用于如屏幕空间反射（SSR）、环境光遮蔽（AO）、体积渲染等计算密集型任务。

结合这些管线的一般流程如下：

- **场景设置**：首先，开发者会设置场景，包括几何体、材质、光源等信息，并将它们组织成适合不同管线处理的格式。
    
- **光栅化**：对于主要场景，使用光栅化管线进行渲染。这包括顶点处理、视锥体裁剪、屏幕映射等步骤，生成初步的像素数据。
    
- **光线追踪**：对于需要精确光照和反射效果的对象，如镜子、水面、玻璃等，使用光线追踪管线进行渲染。这通常在光栅化渲染之后进行，以确保光线追踪可以利用光栅化结果，如深度缓冲区信息。
    
- **计算管线**：在光栅化和光线追踪的基础上，使用计算管线来添加额外的效果，如后处理、屏幕空间反射、环境光遮蔽等。
    
- **合成（Composition）**：最后，将光栅化管线生成的像素数据、光线追踪管线计算的光照效果和计算管线处理的后效果进行合成，生成最终的图像。
    

通过这种方式，光栅化管线可以高效处理场景的主体部分，光线追踪管线可以增强关键视觉效果的真实性，而计算管线可以用于实现一些特定的计算密集型效果。这种多管线结合的架构使得渲染系统既能够保持高性能，又能够实现高度逼真的渲染效果。


正在尝试为您在互联网搜索相关资料...

开始获取资料...

找到了第 1 篇资料：[给水排水工程 埋地矩形管管道结构设计标准](https://pro5323b5d3-pic11.ysjianzhan.cn/upload/30TCECS145-2022GSPSGCMDJXGGDJGSJBZ.pdf)

找到了第 2 篇资料：[GDC2024-Snowdrop中的光线追踪：主机的照明管线优化 - 知乎](https://zhuanlan.zhihu.com/p/693920769)

找到了第 3 篇资料：[给水排水工程管道结构设计规范](http://www.commc.cn/Upload/2019-02-13/1014115370.pdf)

找到了第 4 篇资料：[NVIDIA Turing 架构 | NVIDIA](https://www.nvidia.cn/design-visualization/technologies/turing-architecture/)

找到了第 5 篇资料：[NVIDIA Turing 架构深度介绍 - NVIDIA 技术博客](https://developer.nvidia.com/zh-cn/blog/nvidia-turing-architecture-in-depth/)

找到了第 6 篇资料：[计算管线 | 学习 wgpu - GitHub Pages](https://jinleili.github.io/learn-wgpu-zh/intermediate/compute-pipeline/)

找到了第 7 篇资料：[WebGPU 计算管线、计算着色器（通用计算）入门案例：2D ...](https://www.cnblogs.com/onsummer/p/webgpu-all-of-the-cores-none-of-the-canvas.html)

找到了第 8 篇资料：[通用图形处理单元GPGPU计算管线（General Purpose ...](https://blog.csdn.net/Dontla/article/details/135571915)

找到了第 9 篇资料：[给水排水工程管道结构设计规范[附条文说明]GB50332-2002 ...](https://www.soujianzhu.cn/NormAndRules/NormContent.aspx?id=781)

一共找到 9 篇资料，正在阅读...

找到了 9 篇资料作为参考：

当然可以，以下是一些与光线追踪管线、计算管线和光栅化管线结合使用的相关资料：

1. **NVIDIA Turing 架构深度介绍** - NVIDIA 技术博客 这篇文章深入介绍了支持实时光线追踪的 NVIDIA Turing 架构，包括专用的光线追踪处理器 RT Core，以及如何加速光线和声音在 3D 环境中的传播计算。4
    
2. **图形API和GPU光线追踪分析** - 吴建明wujianming - 博客园 该博客文章分析了市面上几种流行的图形API对光线追踪的支持现状和技术，包括DirectX RayTracing（DXR）的介绍和光线追踪管线状态的详细说明。5
    
3. **优化光线跟踪的实用技巧** - NVIDIA 技术博客 NVIDIA 技术博客提供的实用技巧，帮助开发者实现光线追踪的高效率，从网格实例选择到数据处理，优化跟踪和着色。1
    
4. **光线追踪 - Vulkan入门精要** 这篇文章介绍了光追管线作为独立渲染管线的概念，以及它使用的专用着色器集。1
    
5. **深入解读IMG CXT GPU：业界首个移动端光线追踪架构** - FPGA 社区 文章深入解读了业界首个移动端光线追踪架构，提供了对移动光线追踪硬件的见解。1
    
6. **给水排水工程管道结构设计规范** 虽然这篇文章主要关注的是管道结构设计，但它提供了关于如何将不同的计算和分析方法结合起来以确保结构完整性的见解，这可以间接应用于理解管线如何在GPU中结合。39
    

这些资料涵盖了光线追踪管线的硬件支持、API集成、性能优化，以及管线在实际应用中的结合使用，可以为想要深入了解这一领域的读者提供理论和实践知识。



正在尝试为您在互联网搜索相关资料...

开始获取资料...

找到了第 1 篇资料：[计算机图形学十三：利用包围盒技术加速光线追踪（KD-Tree ...](https://zhuanlan.zhihu.com/p/144403802)

找到了第 2 篇资料：[硬件加速和软件加速的光线追踪有什么区别？ - 知乎](https://zhuanlan.zhihu.com/p/593523356)

找到了第 3 篇资料：[Apple 发布超先进个人电脑芯片 M3、M3 Pro 和 M3 Max](https://www.apple.com.cn/newsroom/2023/10/apple-unveils-m3-m3-pro-and-m3-max-the-most-advanced-chips-for-a-personal-computer/)

找到了第 4 篇资料：[NVIDIA Turing 架构 | NVIDIA](https://www.nvidia.cn/design-visualization/technologies/turing-architecture/)

找到了第 5 篇资料：[GPU迎来新突破：硬件实时光线追踪技术全球首次迈进移动 ...](https://new.qq.com/rain/a/20211108A06BQ700)

找到了第 6 篇资料：[通过 NVIDIA OptiX 8 实现灵活且强大的光线追踪功能](https://developer.nvidia.com/zh-cn/blog/flexible-and-powerful-ray-tracing-with-optix-8/)

找到了第 7 篇资料：[NVIDIA Turing 架构深度介绍 - NVIDIA 技术博客](https://developer.nvidia.com/zh-cn/blog/nvidia-turing-architecture-in-depth/)

找到了第 8 篇资料：[计算机图形学----光线追踪(Ray Tracing)----加 …](https://blog.csdn.net/VIPCCJ/article/details/119550359)

找到了第 9 篇资料：[通过 NVIDIA OptiX 8 实现灵活且强大的光线追踪功能](https://developer.nvidia.com/zh-cn/blog/flexible-and-powerful-ray-tracing-with-optix-8/)

一共找到 9 篇资料，正在阅读...

找到了 9 篇资料作为参考：

为了加速光线追踪，业界设计了多种算法和硬件架构，以下是一些主要的技术：

1. **专用硬件加速**：NVIDIA 图灵架构中的 RT Core 是专门为加速光线追踪而设计的硬件单元。它能够以极高的速度执行光线与场景中几何体的相交测试24。
    
2. **GPU 加速软件光线追踪**：NVIDIA OptiX 是一个利用 GPU 加速光线追踪的软件框架，它允许开发者利用 GPU 的并行处理能力来加速光线追踪算法29。
    
3. **包围盒技术**：包括轴对齐包围盒（AABB）和其他类型的包围盒，这些技术通过在场景中创建包围盒来减少光线需要相交的对象数量，从而加速光线追踪过程18。
    
4. **空间划分算法**：如KD-Tree和Oct-Tree，这些算法通过将空间划分为多个区域来加速光线与物体的相交测试18。
    
5. **边界体积层次结构（BVH）**：BVH 是一种以对象为中心的加速结构，它通过构建一个树状结构来组织场景中的几何体，从而提高光线追踪的效率81。
    
6. **动态缓存**：Apple M3 系列芯片中的新一代图形处理器引入了动态缓存技术，可以对内存的使用进行实时分配，提高图形处理器的平均利用率3。
    
7. **深度学习超级采样（DLSS）**：NVIDIA 的 Tensor Cores 通过 DLSS 技术，使用深度学习来加速图形密集型工作负载，包括光线追踪中的去噪和抗锯齿处理2。
    
8. **光线追踪API**：如Microsoft的DirectX光线追踪（DXR）API，它为开发者提供了一套完整的工具集，以便在支持的硬件上实现光线追踪27。
    
9. **移动硬件支持**：随着技术的发展，移动设备也开始支持硬件加速的光线追踪，例如报道中提到的RAC技术，它是一种低功耗专用硬件GPU组件，用于加速光线追踪计算5。
    

这些技术和架构的结合使用，使得实时光线追踪在游戏和专业图形应用中变得更加可行和高效。随着技术的不断进步，未来可能会有更多创新的算法和硬件架构被开发出来，以进一步提升光线追踪的性能。


光线追踪管线是实现逼真渲染效果的关键技术之一，它通过模拟光线在虚拟场景中的传播和交互来生成图像。光线追踪管线的组成部分和功能如下：

1. **光线生成（Ray Generation）**：
    
    - 这是光线追踪的起点，负责生成从摄像机或观察者位置发出的原始光线。
2. **加速结构（Acceleration Structure）**：
    
    - 用于存储和组织场景中的几何信息，以加速光线与物体的相交测试。常见的加速结构包括边界体积层次结构（BVH）和包围盒（如AABB）。
3. **交点计算（Intersection）**：
    
    - 计算光线与场景中物体的交点。如果光线与物体相交，将执行后续的着色计算。
4. **着色器（Shaders）**：
    
    - 包括多种类型的着色器，用于在光线与物体相交后计算光照、材质效果等：
        - **最近命中着色器（Closest Hit Shader）**：处理光线与物体相交点的光照和材质渲染。
        - **任意命中着色器（Any Hit Shader）**：用于实现后处理效果，如透明度或反射效果。
        - **未命中着色器（Miss Shader）**：当光线未与任何物体相交时被调用，通常用于渲染背景或天空盒。
5. **光线修改（Ray Modification）**：
    
    - 在光线与物体相交后，可能需要对光线进行修改，如反射、折射或散射等。
6. **递归光线追踪（Recursive Ray Tracing）**：
    
    - 对于反射、折射等效果，光线追踪管线会递归地生成新的光线并重新执行交点计算和着色。
7. **后处理（Post-Processing）**：
    
    - 在最终像素值确定之后，可能还需要一些后处理步骤，如色调映射、模糊、抗锯齿等，以改善最终图像的视觉效果。
8. **管线状态（Pipeline State）**：
    
    - 定义了管线的全局状态，包括使用的着色器、资源绑定等。
9. **着色器绑定表（Shader Binding Table, SBT）**：
    
    - 描述了shader与场景的哪个物体关联，并包含了shader中涉及的所有资源。
10. **光线追踪命令（如 DispatchRays）**：
    
    - 是光线追踪管线中提交光线追踪工作到GPU的命令，它会调用光线生成着色器开始光线追踪过程。

光线追踪管线的每个部分都承担着不同的职责，共同协作以实现高度逼真的渲染效果。随着硬件的发展和图形API的更新，光线追踪管线的效率和功能也在不断提升。
