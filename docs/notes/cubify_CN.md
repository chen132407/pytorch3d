---
隐藏标题：true
sidebar_label：立方体
---

# 立体化

[cubify 运算符](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/cubify.py) 将形状为“BxDxHxW”的 3D 占用网格（其中“B”是批量大小）转换为实例化为[网格](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/structurals/meshes.py ) ‘B’元素的数据结构。操作员用 12 个面和 8 个顶点的长方体替换每个占用的体素（如果其占用概率大于用户定义的阈值）。共享顶点被合并，内部面被移除，从而形成**水密**的网格。

该运算符提供三种对齐模式{*topleft*、*corner*、*center*}，它们定义网格顶点相对于体素网格的跨度。下图描述了 2D 网格的对齐模式。

![输入](https://user-images.githubusercontent.com/4369065/81032959-af697380-8e46-11ea-91a8-fae89597f988.png)