---
sidebar_label：从文件加载
隐藏标题：true
---

# 网格和 IO

Meshes 对象代表一批三角网格，并且是
PyTorch3D 的大部分功能。没有坚持每个网格
该批次具有相同数量的顶点或面。当可用时，它可以存储
与网格相关的其他数据，例如面法线、面区域
和纹理。

用于存储单个网格的两种常见文件格式是“.obj”和“.ply”文件，
PyTorch3D 有读取这些的函数。

## 对象

Obj 文件有一个标准方法来存储有关网格的额外信息。给定一个
obj 文件，可以用以下命令读取

````
  verts, faces, aux = load_obj(filename)
````

将“verts”设置为顶点的 (V,3) 张量，将“faces.verts_idx”设置为
面的每个角的顶点索引的 (F,3)- 张量。
非三角形的面将被分割成三角形。`aux` 是一个对象
其中可能包含法线、uv 坐标、材质颜色和纹理（如果它们）
存在，并且“faces”可能还包含这些法线的索引，
其 NamedTuple 结构中的纹理和材质。包含一个 Meshes 对象
可以使用以下命令仅从顶点和面创建单个网格
````
    <!-- 网格 = 网格(verts=[verts], faces=[faces.verts_idx]) -->
    meshes = Meshes(verts=[verts], faces=[faces.verts_idx])
````

如果`.obj`中有纹理信息，它可以用来初始化一个
`Textures` 类被传递到 `Meshes` 构造函数中。目前我们
支持为具有一个纹理贴图的网格加载纹理贴图
整个网格例如

````
verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)
tex_maps = aux.texture_images

# tex_maps是{材质名称：纹理图像}的字典。
# 拍摄第一张图像：
texture_image = list(tex_maps.values())[0]
texture_image = texture_image[None, ...]  # (1, H, W, 3)

# 创建一个纹理对象
tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image)

# 使用纹理初始化网格
meshes = Meshes(verts=[verts], faces=[faces.verts_idx], textures=tex)
````

`load_objs_as_meshes` 函数提供了这个过程。

## 层数

Ply 文件存储附加信息的方式很灵活。PyTorch3D
提供一个仅从层文件中读取顶点和面的函数。
电话
````
    <!-- 顶点，面= load_ply（文件名） -->
    verts, faces = load_ply(filename)
````
将“verts”设置为顶点的 (V,3)-张量，并将“faces”设置为 (F,3)-
面的每个角的顶点索引的张量。面对哪些
不是三角形就会被分割成三角形。包含一个 Meshes 对象
可以使用此数据创建单个网格
````
    <!-- 网格=网格（顶点=[顶点]，面=[面]） -->
    meshes = Meshes(verts=[verts], faces=[faces])
````