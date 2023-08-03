---
隐藏标题：true
sidebar_label：文件IO
---

# 文件IO
有一个灵活的界面，用于加载和保存不同格式的点云和网格。

主要用法是通过“pytorch3d.io.IO”对象及其方法
`load_mesh`、`save_mesh`、`load_pointcloud` 和 `save_pointcloud`。

例如，要加载网格，您可能会这样做
````
from pytorch3d.io import IO

device=torch.device("cuda:0")
mesh = IO().load_mesh("mymesh.obj", device=device)
````

要保存点云，您可能会这样做
````
pcl = Pointclouds(...)
IO().save_pointcloud(pcl, "output_pointcloud.ply")
````

对于网格，这支持 OBJ、PLY 和 OFF 文件。

对于点云，这支持 PLY 文件。

此外，还提供了从以下位置加载网格的实验支持：
[glTF 2 资产](https://github.com/KhronosGroup/glTF/tree/master/specification/2.0)
存储在 GLB 容器文件或带有嵌入二进制数据的 glTF JSON 文件中。
必须显式启用此功能，如中所述
`pytorch3d/io/experimental_gltf_io.py`。