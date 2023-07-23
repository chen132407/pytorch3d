---
隐藏标题：true
sidebar_label：批处理
---

# 批处理

在深度学习中，每个优化步骤都对多个输入示例进行操作，以实现稳健的训练。因此，高效的批处理至关重要。对于图像输入，批处理很简单；N 个图像被调整为相同的高度和宽度，并堆叠为形状为“N x 3 x H x W”的 4 维张量。对于网格来说，批处理不太简单。

<img src =“assets/batch_intro.png”alt =“batch_intro”align =“middle”/>

## 网格的批处理模式

假设您要构建一个包含两个网格的批处理，其中“mesh1 = (v1: V1 x 3, f1: F1 x 3)”包含“V1”顶点和“F1”面，“mesh2 = (v2: V2 x 3, f2: F2 x 3)”包含“V2 (!= V1)”顶点和“F2 (!= F1)”面。[Meshes][meshes] 数据结构提供了三种不同的方法来批处理*异构*网格。如果 `meshes = Meshes(verts = [v1, v2], faces = [f1, f2])` 是数据结构的实例化，那么

* 列表：以张量列表的形式返回批次中的示例。具体来说，“meshes.verts_list()”返回顶点“[v1, v2]”列表。类似地，“meshes.faces_list()”返回面列表“[f1, f2]”。
* 填充：填充表示通过填充额外的值来构造张量。具体来说，“meshes.verts_padded()”返回形状为“2 x max(V1, V2) x 3”的张量，并用“0”填充额外的顶点。类似地，“meshes.faces_padded()”返回形状为“2 x max(F1, F2) x 3”的张量，并用“-1”填充额外的面。
* Packed：打包表示将批次中的示例连接成张量。特别是，“meshes.verts_packed()”返回形状为“(V1 + V2) x 3”的张量。类似地，“meshes.faces_packed()”返回形状为“(F1 + F2) x 3”的面张量。在打包模式下，计算辅助变量，以实现打包模式和填充模式或列表模式之间的高效转换。

<img src =“assets/batch_modes.gif”alt =“batch_modes”height =“450”align =“middle”/>

## 批处理模式的用例

对不同网格批处理模式的需求是 PyTorch 运算符实现方式所固有的。为了充分利用优化的 PyTorch 操作，[Meshes][meshes] 数据结构允许在不同批处理模式之间进行高效转换。当目标是快速高效的培训周期时，这一点至关重要。[Mesh R-CNN][meshcnn] 就是一个例子。在这里，在同一个前向传递中，网络的不同部分采用不同的输入，这些输入是通过在不同批处理模式之间进行转换来计算的。特别是，[vert_align][vert_align]假设一个*填充*输入张量，而在[graph_conv][graphconv]之后立即假设一个*打包*输入张量。

<img src="assets/meshrcnn.png" alt="meshrcnn" width="700"align="middle" />


[网格]：https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/structs/meshes.py
[graphconv]：https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/graph_conv.py
[vert_align]：https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/vert_align.py
[meshcnn]：https://github.com/facebookresearch/meshrcnn