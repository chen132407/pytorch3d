<!--
 * @Author: tiger
 * @Date: 2023-07-23 08:19:41
 * @FilePath: \AI_github_example\CV_git\pytorch3d-main\pytorch3d\docs\notes\renderer_CN.md
-->
---
隐藏标题：true
sidebar_label：概述
---

# 渲染概述

可微分渲染是计算机视觉中一个相对较新且令人兴奋的研究领域，它通过允许 2D 图像像素与场景的 3D 属性相关联来弥合 2D 和 3D 之间的差距。

例如，通过根据神经网络预测的 3D 形状渲染图像，可以使用参考图像计算 2D 损失。反转渲染步骤意味着我们可以将像素的 2D 损失与形状的 3D 属性（例如网格顶点的位置）相关联，从而无需任何显式 3D 监督即可学习 3D 形状。

我们广泛研究了现有的可微渲染代码库，发现：
- 渲染管道很复杂，有超过 7 个独立组件，需要互操作和可微分
- 流行的现有方法 [[1](#1)、[2](#2)] 基于相同的核心实现，它将许多关键组件捆绑到大型 CUDA 内核中，这需要大量的专业知识才能理解，并且扩展范围有限
- 现有方法要么不支持批处理，要么假设批次中的网格具有相同数量的顶点和面
- 现有项目仅提供 CUDA 实现，因此无法在没有 GPU 的情况下使用

为了尝试不同的方法，我们需要一个易于使用和扩展的模块化实现，并且支持[异构批处理](batching.md)。受到现有工作 [[1](#1)、[2](#2)] 的启发，我们创建了一个新的、模块化的、可微分的渲染器，在 PyTorch、C++ 和 CUDA 中具有**并行实现**，以及全面的文档和测试，旨在帮助该领域的进一步研究。

我们的实现将渲染的光栅化和着色步骤解耦。核心光栅化步骤（基于 [[2]](#2)）返回几个中间变量，并在 CUDA 中进行了优化实现。管道的其余部分纯粹在 PyTorch 中实现，并且旨在定制和扩展。通过这种方法，PyTorch3D 可微分渲染器可以作为库导入。

## <u>开始</u>

要了解更多实现并开始使用渲染器，请参阅[渲染器入门](renderer_getting_started.md)，其中还包含[架构概述](assets/architecture_renderer.jpg)和[坐标转换约定](assets/transforms_overview.jpg)。

## <u>技术报告</u>

有关渲染器设计、关键功能和基准的深入说明，请参阅 ArXiv 上的 PyTorch3D 技术报告：[使用 PyTorch3D 加速 3D 深度学习](https://arxiv.org/abs/2007.08501)，有关脉冲星后端，请参阅此处：[使用基于球体表示的神经渲染的快速可微分光线投射](https://arxiv.org/abs/2) 004.07484）。

---

**注意：CUDA 内存使用**

技术报告中的主要比较是与 SoftRasterizer [[2](#2)]。与输出 4 个张量的 PyTorch3D 光栅器前向 CUDA 内核相比，SoftRasterizer 前向 CUDA 内核仅输出 1 个“(N, H, W, 4)” FloatTensor：

  - `pix_to_face`，LongTensor `(N，H，W，K)`
  - `zbuf`，FloatTensor `(N，H，W，K)`
  - `dist`, FloatTensor `(N, H, W, K)`
  - `bary_coords`，FloatTensor `(N, H, W, K, 3)`

其中 **N** = 批量大小，**H/W** 是图像高度/宽度，**K** 是每像素的面数。PyTorch3D 向后传递返回“zbuf”、“dist”和“bary_coords”的梯度。

从光栅化返回中间变量具有相关的内存成本。我们可以计算前向和后向传递的内存使用量的理论下限，如下所示：

````
# 假设每个 float 为 4 个字节，long 为 8 个字节
# Assume 4 bytes per float, and 8 bytes for long

memory_forward_pass = ((N * H * W * K) * 2 + (N * H * W * K * 3)) * 4 + (N * H * W * K) * 8
memory_backward_pass = ((N * H * W * K) * 2 + (N * H * W * K * 3)) * 4

total_memory = memory_forward_pass + memory_backward_pass
             = (N * H * W * K) * (5 * 4 * 2 + 8)
             = (N * H * W * K) * 48
````

光栅化输出的每个面每个像素需要 48 个字节。为了保持在内存使用范围内，我们可以改变批量大小 (**N**)、图像大小 (**H/W**) 和每像素面数 (**K**)。例如，对于固定的批量大小，如果使用较大的图像大小，请尝试减少每个像素的面孔。

---

＃＃＃ 参考

<a id="1">[1]</a> Kato 等人，“神经 3D 网格渲染器”，CVPR 2018

<a id="2">[2]</a> Liu 等人，“软光栅化器：基于图像的 3D 推理的可微渲染器”，ICCV 2019

<a id="3">[3]</a> Loper 等人，“OpenDR：近似可微分渲染器”，ECCV 2014

<a id="4">[4]</a> De La Gorce 等人，“通过单目视频进行基于模型的 3D 手部姿势估计”，PAMI 2011

<a id="5">[5]</a> Li 等人，“通过边缘采样进行可微分蒙特卡洛射线追踪”，SIGGRAPH Asia 2018

<a id="6">[6]</a> Yifan 等人，“基于点的几何处理的可微分表面泼溅”，SIGGRAPH Asia 2019

<a id="7">[7]</a> Loubet 等人，“重新参数化不连续积分以实现可微渲染”，SIGGRAPH Asia 2019

<a id="8">[8]</a> Chen 等人，“使用基于插值的可微分渲染器学习预测 3D 对象”，NeurIPS 2019