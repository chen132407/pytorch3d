---
隐藏标题：true
sidebar_label：绘图可视化
---

＃ 概述

PyTorch3D 提供了模块化可微渲染器，但对于我们想要交互式绘图或不关心渲染过程的可微性的情况，我们提供了[在绘图中渲染网格和点云的函数](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/vis/plotly_vis.py)。这些绘图图形允许您旋转和缩放渲染的图像，并支持将批量数据绘制为单个绘图中的多个轨迹或划分为单独的子绘图。


＃ 例子

这些渲染函数接受绘图 x、y 和 z 轴参数作为“kwargs”，允许我们自定义绘图。这里有两个带有彩色轴的图，一个是[点云图](assets/plotly_pointclouds.png)，一个[子图中的批处理网格图](assets/plotly_meshes_batch.png)，以及一个[带有多个轨迹的批处理网格图](assets/plotly_meshes_trace.png)。有关代码示例，请参阅[渲染纹理网格](https://pytorch3d.org/tutorials/render_textured_meshes)和[渲染彩色点云](https://pytorch3d.org/tutorials/render_colored_points)教程。

# 将绘图保存到图像中

如果要保存这些绘图，则需要安装单独的库，例如 [Kaleido](https://plotly.com/python/static-image-export/)。

安装万花筒
````
$ pip 安装 Kaleido
````
将图形导出为 .png 图像。图像将保存在当前工作目录中。
````
无花果 = ...
Fig.write_image("image_name.png")
````