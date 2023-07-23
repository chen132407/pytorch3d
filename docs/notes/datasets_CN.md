---
隐藏标题：true
sidebar_label：数据加载器
---

# 常见 3D 数据集的数据加载器

### ShapetNetCore

ShapeNet 是 3D CAD 模型的数据集。ShapeNetCore 是 ShapeNet 数据集的子集，可以从 https://www.shapenet.org/ 下载。ShapeNetCore 有两个版本：v1（55 个类别）和 v2（57 个类别）。

PyTorch3D [ShapeNetCore 数据加载器](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/datasets/shapenet/shapenet_core.py) 继承自 `torch.utils.data.Dataset`。它采用 ShapeNetCore 数据集在本地存储的路径并加载数据集中的模型。ShapeNetCore 类加载并返回模型及其“categories”、“model_ids”、“vertices”和“faces”。“ShapeNetCore”数据加载器还具有自定义的“render”函数，可使用 PyTorch3D 的可微渲染器按指定的“model_ids (List[int])”、“categories (List[str])”或“indices (List[int])”渲染模型。

加载的数据集可以使用 PyTorch3D 的自定义 collat​​e_fn 传递到 `torch.utils.data.DataLoader`：`pytorch3d.dataset.utils` 模块中的 `collat​​e_batched_meshes`。模型的“顶点”和“面”用于构造表示批处理网格的网格对象。这种“网格”表示可以轻松地与 PyTorch3D 中的其他操作和渲染一起使用。

### R2N2

R2N2 数据集包含 13 个类别，它们是 ShapeNetCore v.1 数据集的子集。R2N2 数据集还包含每个对象和体素化模型的 24 个渲染图。R2N2 数据集可以按照[此处](http://3d-r2n2.stanford.edu/) 的说明下载。

PyTorch3D [R2N2 数据加载器](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/datasets/r2n2/r2n2.py) 使用 ShapeNet 数据集、R2N2 数据集和 R2N2 拆分文件的路径进行初始化。就像“ShapeNetCore”一样，它可以通过“pytorch3d.dataset.r2n2.utils”模块中的自定义 collat​​e_fn：“collat​​e_batched_R2N2”传递到“torch.utils.data.DataLoader”。它返回“ShapeNetCore”返回的所有数据，此外，它还返回 R2N2 渲染（每个模型 24 个视图）以及相机校准矩阵和每个模型的体素表示。与“ShapeNetCore”类似，它具有自定义的“render”函数，支持使用 PyTorch3D 可微渲染器渲染指定模型。此外，