---
隐藏标题：true
sidebar_label：入门
---

# 渲染器入门

### 架构概述

渲染器被设计为模块化、可扩展的，并支持所有输入的批处理和渐变。下图描述了渲染管道的所有组件。

<img src="assets/architecture_renderer.jpg" width="1000">

##### 碎片

**光栅化器**在命名元组中返回 4 个输出张量。

- **`pix_to_face`**：形状为`(N, image_size, image_size, faces_per_pixel)`的长张量，指定与图像中每个像素重叠的面（在打包面中）的索引。
- **`zbuf`**：形状为“(N, image_size, image_size, faces_per_pixel)”的 FloatTensor，给出世界坐标中每个像素处最近的人脸的 z 坐标，按 z 升序排序。
- **`bary_coords`**：形状为“(N, image_size, image_size, faces_per_pixel, 3)”的 FloatTensor
  给出每个像素处最近面的 NDC 单位重心坐标，按 z 升序排序。
- **`pix_dists`**：形状为 `(N, image_size, image_size, faces_per_pixel)` 的 FloatTensor，给出最接近像素的每个点的 x/y 平面中的有符号欧几里得距离（以 NDC 单位）。


有关管道中每个组件的更多详细信息，请参阅渲染器 API 参考。

---

**笔记：**

可微分渲染器 API 是实验性的，可能会发生变化！。

---

### 坐标变换约定

渲染需要在几个不同的坐标系之间进行转换：世界空间、视图/相机空间、NDC 空间和屏幕空间。在每一步中，重要的是要了解相机的位置、+X、+Y、+Z 轴如何对齐以及可能的值范围。下图概述了 PyTorch3D 使用的约定。

<img src="assets/transforms_overview.jpg" width="1000">


例如，给定一个茶壶网格，世界坐标系、相机坐标系和图像如下图所示。请注意，世界坐标系和相机坐标系的 +z 方向指向页面。

<img src="assets/world_camera_image.jpg" width="1000">

---

**注意：PyTorch3D 与 OpenGL**

虽然我们尝试模拟 OpenGL 的多个方面，但坐标系约定存在差异。
- PyTorch3D 中的默认世界坐标系的 +Z 指向屏幕，而在 OpenGL 中，+Z 指向屏幕外。两人都是右利手。
- 与 OpenGL 中的**左手** NDC 坐标系相比，PyTorch3D 中的 NDC 坐标系是**右手**（投影矩阵切换惯用手）。

<imgalign=“center”src=“assets/opengl_coordframes.png”宽度=“300”>

---

### 光栅化非方形图像

要栅格化 H != W 的图像，您可以在“RasterizationSettings”中将“image_size”指定为 (H, W) 的元组。

长宽比需要特别考虑。有两个长宽比需要注意：
    - 每个像素的长宽比
    - 输出图像的纵横比
在相机中，例如“FoVPerspectiveCameras”，“aspect_ratio”参数可用于设置像素长宽比。在光栅化器中，我们假设正方形像素，但图像长宽比可变（即矩形图像）。

在大多数情况下，您需要将相机纵横比设置为 1.0（即方形像素），并且仅改变“RasterizationSettings”中的“image_size”（即以像素为单位的输出图像尺寸）。

---

### 脉冲星后端

从 v0.3 开始，[pulsar](https://arxiv.org/abs/2004.07484) 可以用作点渲染的后端。它注重效率，这有利有弊：它经过高度优化，所有渲染阶段都集成在 CUDA 内核中。这会带来显着更高的速度和更好的扩展行为。我们在 Facebook 现实实验室使用它来渲染和优化具有数百万个球体、分辨率高达 4K 的场景。您可以在下面找到运行时比较图（设置：“bin_size=None”、“points_per_pixel=5”、“image_size=1024”、“radius=1e-2”、“composite_params.radius=1e-4”；在 RTX 2070 GPU 上进行基准测试）。

<imgalign=“中心”src=“资产/pulsar_bm.png”宽度=“300”>

Pulsar 的处理步骤是紧密集成的 CUDA 内核，不能与自定义的“光栅器”和“合成器”组件一起使用。我们提供两种使用 Pulsar 的方式：（1）有统一的接口来无缝匹配 PyTorch3D 调用约定。例如，[点云教程](https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/render_colored_points.ipynb)对此进行了说明。(2) pulsar 后端有一个直接可用的接口，它公开了后端的全部功能（包括不透明度，这在 PyTorch3D 中尚不可用）。显示其使用以及匹配的 PyTorch3D 接口代码的示例可在[此文件夹](https://github.com/facebookresearch/pytorch3d/tree/master/docs/examples)中找到。

---

### 纹理选项

对于网格纹理，我们提供了几个选项（在“pytorch3d/renderer/mesh/texturing.py”中）：

1. **顶点纹理**：每个顶点的 D 维纹理（例如 RGB 颜色），可以在整个面上进行插值。这可以表示为“(N, V, D)”张量。但这是一个相当简单的表示，如果网格面很大，则无法对复杂的纹理进行建模。
2. **UV 纹理**：整个网格的顶点 UV 坐标和**一个** 纹理贴图。对于具有给定重心坐标的面上的点，可以通过对顶点 uv 坐标进行插值，然后从纹理贴图中采样来计算面颜色。这种表示需要两个张量（UV：“(N, V, 2)”，纹理贴图：“(N, H, W, 3)”），并且仅限于每个网格仅支持一个纹理贴图。
3. **面部纹理**：在更复杂的情况下，例如 ShapeNet 网格，每个网格有多个纹理贴图，并且某些面部具有纹理，而其他面部则没有。对于这些情况，更灵活的表示是纹理图集，其中每个面都表示为“(RxR)”纹理图，其中 R 是纹理分辨率。对于脸上的给定点，可以使用该点的重心坐标从每个面纹理映射中采样纹理值。这种表示需要一个形状为“(N, F, R, R, 3)”的张量。这种纹理方法的灵感来自于 SoftRasterizer 实现。有关更多详细信息，请参阅 [`make_material_atlas`](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/io/mtl_io.py#L123) 和 [`sample_textures`](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/renderer/mesh/textures)。py#L452) 函数。**注意：**：“TexturesAtlas”纹理采样仅相对于纹理图集可微，但相对于重心坐标不可微。


<img src="assets/texturing.jpg" width="1000">

---

### 一个简单的渲染器

PyTorch3D 中的渲染器由 **光栅器** 和 **着色器** 组成。只需几个简单的步骤即可创建渲染器：

````
# 进口
从 pytorch3d.renderer 导入（
    FoVPerspectiveCameras，look_at_view_transform，
    光栅化设置、混合参数、
    网格渲染器、网格光栅化器、HardPhongShader
）

# 初始化一个 OpenGL 透视相机。
R, T = Look_at_view_transform(2.7, 10, 20)
相机= FoVPerspectiveCameras（设备=设备，R=R，T=T）

# 定义光栅化和着色的设置。这里我们设置输出图像的大小
# 512x512。由于我们仅出于可视化目的渲染图像，因此我们将设置 faces_per_pixel=1
# 且模糊半径=0.0。有关这些参数的说明，请参阅 rasterize_meshes.py。
raster_settings = 光栅化设置（
    图像大小=512，
    模糊半径=0.0，
    faces_per_pixel=1，
）

# 通过组合光栅器和着色器来创建 Phong 渲染器。这里我们可以使用预定义的
# PhongShader，传入要初始化默认参数的设备
渲染器 = MeshRenderer(
    光栅化器=MeshRasterizer（相机=相机，raster_settings=raster_settings），
    着色器=HardPhongShader(设备=设备，相机=相机)
）
````

---

### 自定义着色器

着色器是 PyTorch3D 渲染 API 中最灵活的部分。我们在“shaders.py”中创建了一些着色器示例，但这不是详尽的集合。

着色器可以包含几个步骤：
- **纹理**（例如，顶点 RGB 颜色插值或顶点 UV 坐标插值，然后从纹理贴图采样（插值使用光栅化输出的重心坐标））
- **照明/阴影**（例如环境光、漫射光、镜面反射光、Phong、Gouraud、Flat）
- **混合**（例如，仅使用每个像素最近的面进行硬混合，或使用每个像素前 K 个面的加权和进行软混合）

 我们有基于我们当前拥有的纹理/着色/混合支持的这些功能的几种组合的示例。下表总结了这些内容。许多其他组合都是可能的，我们计划扩展可用于纹理、阴影和混合的选项。

|示例着色器 | 顶点纹理| 紫外线纹理 | 纹理图集| 平面阴影| Gouraud 着色| Phong 阴影 | 硬混 | 柔和混合|
| ------------- |:-------------: | :--------------:| :--------------:| :--------------:| :--------------:| :--------------:|:--------------:|:----------------:|
| HardPhongShader | ✔️ |✔️|✔️||| ✔️ | ✔️||
| SoftPhongShader | ✔️ |✔️|✔️||| ✔️ | | ✔️|
| HardGouraudShader | ✔️ |✔️|✔️|| ✔️ || ✔️||
| SoftGouraudShader | ✔️ |✔️|✔️|| ✔️ ||| ✔️|
| 硬平面着色器 | ✔️ |✔️|✔️| ✔️ ||| ✔️||
| 软轮廓着色器 |||||||| ✔️|