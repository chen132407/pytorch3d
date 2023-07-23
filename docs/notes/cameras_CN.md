---
隐藏标题：true
sidebar_label：相机
---

# 相机

## 相机坐标系

在处理 3D 数据时，用户需要了解 4 个坐标系
* **世界坐标系**
这就是物体/场景所生活的系统——世界。
* **相机视图坐标系**
该系统的原点位于像平面上，“Z”轴垂直于像平面。在 PyTorch3D 中，我们假设“+X”指向左，“+Y”指向上，“+Z”指向图像平面外。从世界到视图的转换发生在应用旋转（“R”）和平移（“T”）之后。
* **NDC坐标系**
这是标准化的坐标系，将对象/场景的渲染部分限制在体积中。也称为视图体积。对于方形图像，根据 PyTorch3D 约定，“(+1, +1, znear)”是体积的左上角近角，“(-1, -1, zfar)”是体积的右下角。对于非方形图像，“XY”中体积最小长度的边的范围是“[-1, 1]”，而较大边的范围是“[-s, s]”，其中“s”是长宽比，“s > 1”（较大边除以较小边）。
从视图到 NDC 的转换发生在应用相机投影矩阵（“P”）之后。
* **屏幕坐标系**
这是视图体积的另一种表示形式，其“XY”坐标是在像素空间而不是标准化空间中定义的。(0,0) 是左上角像素
(W,H) 是右下像素的右下角。

4个坐标系的示意图如下所示
![相机](https://user-images.githubusercontent.com/669761/145090051-67b506d7-6d73-4826-a677-5873b7cb92ba.png)

## 在 PyTorch3D 中定义相机

PyTorch3D 中的相机通过首先将对象/场景转换为视图（通过变换“R”和“T”），然后通过投影矩阵“P = K[R |”将 3D 对象/场景投影到标准化空间，将对象/场景从世界转换为视图。T]`，其中“K”是内在矩阵。“K”中的相机参数定义了标准化空间。如果用户在NDC空间中定义相机参数，则变换投影指向NDC。如果相机参数是在屏幕空间中定义的，则变换后的点位于屏幕空间中。

请注意，“CamerasBase”基类不对坐标系做出任何假设。所有上述变换都是纯粹由“R”、“T”和“K”定义的几何变换。这意味着用户可以在任何坐标系中定义相机并进行任何变换。“transform_points”方法会将“K”、“R”和“T”应用到输入点作为简单的矩阵变换。但是，如果用户希望将相机与 PyTorch3D 渲染器一起使用，则需要遵守 PyTorch3D 的坐标系假设（请参阅下文）。

我们提供了 PyTorch3D 中常见相机类型的实例化以及用户如何灵活定义下面的投影空间。

## 与 PyTorch3D 渲染器交互

网格和点云的 PyTorch3D 渲染器假设相机变换的点（意味着作为输入传递到光栅器的点）位于 PyTorch3D 的 NDC 空间中。因此，为了获得预期的渲染结果，用户需要确保他们的 3D 输入数据和相机遵守这些 PyTorch3D 坐标系假设。PyTorch3D 坐标系假设 `+X:left`、`+Y: up` 和 `+Z: from us to scene`（右手）。关于坐标系的混淆很常见，因此我们建议您花一些时间了解数据及其所在的坐标系，并在使用 PyTorch3D 渲染器之前进行相应的转换。

相机示例以及它们如何与 PyTorch3D 渲染器交互的示例可以在我们的教程中找到。

### 相机类型

所有相机都继承自“CamerasBase”，它是所有相机的基类。PyTorch3D 提供四种不同的相机类型。`CamerasBase` 定义了所有相机型号通用的方法：
* `get_camera_center` 返回世界坐标中相机的光学中心
* `get_world_to_view_transform` 返回从世界坐标到相机视图坐标 `(R, T)` 的 3D 变换
* `get_full_projection_transform` 将投影变换 (`K`) 与世界到视图变换 `(R, T)` 组合起来
* `transform_points` 接受世界坐标中的一组输入点并投影到范围从 [-1, -1, znear] 到 [+1, +1, zfar] 的 NDC 坐标。
* `get_ndc_camera_transform` 定义了到 PyTorch3D 的 NDC 空间的转换，并在与 PyTorch3D 渲染器交互时被调用。如果相机是在 NDC 空间中定义的，则返回恒等变换。如果相机是在屏幕空间中定义的，则返回从屏幕到 NDC 的转换。如果用户在屏幕空间中定义自己的相机，则需要考虑屏幕到NDC的转换。我们提供了“PerspectiveCameras”和“OrthographicCameras”的示例。
* `transform_points_ndc` 获取世界坐标中的一组点并将它们投影到 PyTorch3D 的 NDC 空间
* `transform_points_screen` 接受世界坐标中的一组输入点并将它们投影到范围从 [0, 0, znear] 到 [W, H, zfar] 的屏幕坐标

用户可以轻松定制自己的相机。对于每个新相机，用户应该实现“get_projection_transform”例程，该例程返回从相机视图坐标到 NDC 坐标的映射“P”。

#### FoVPerspectiveCameras、FoVorthographicCameras
这两个相机分别遵循透视相机和正交相机的 OpenGL 约定。用户提供近“znear”和远“zfar”字段，将视图体积限制在“Z”轴上。“XY”平面中的视图体积在“FoVPerspectiveCameras”的情况下由视场角 (“fov”) 定义，在“FoVOrthographicCameras”的情况下由“min_x、min_y、max_x、max_y”定义。
这些相机默认位于 NDC 空间中。

#### 透视相机、正交相机
这两款相机遵循相机的多视图几何约定。用户提供焦距（“fx”、“fy”）和主点（“px”、“py”）。例如，`camera = PerspectiveCameras(focal_length=((fx, fy),),principal_point=((px, py),))`

视图坐标中的 3D 点“(X, Y, Z)”到投影空间（NDC 或屏幕）中的点“(x, y, z)”的相机投影为

````
# 用于透视相机
x = fx * X / Z + px
y = fy * Y / Z + py
z = 1 / Z

# 用于正交相机
x = fx * X + px
y = fy * Y + py
z = Z
````

用户可以在NDC或屏幕空间中定义相机参数。屏幕空间相机参数很常见，在这种情况下，用户需要将“in_ndc”设置为“False”，并提供屏幕的“image_size=(height, width)”，即图像。

`get_ndc_camera_transform` 提供了 PyTorch3D 中从屏幕到 NDC 空间的转换。请注意，屏幕空间假设主点位于“+X left”、“+Y down”且原点位于图像左上角的空间中。为了转换为 NDC，我们需要考虑标准化空间的缩放以及“XY”方向的变化。

以下是分别在 NDC 和屏幕空间中等效“PerspectiveCameras”实例化的示例。

````蟒蛇
# NDC 太空相机
fcl_ndc = (1.2,)
prp_ndc = ((0.2, 0.5),)
cameras_ndc = PerspectiveCameras(focal_length=fcl_ndc,principal_point=prp_ndc)

# 屏幕空间相机
image_size = ((128, 256),) # (高, 宽)
fcl_screen = (76.8,) # fcl_ndc * min(image_size) / 2
prp_screen = ((115.2, 32), ) # w / 2 - px_ndc * min(image_size) / 2, h / 2 - py_ndc * min(image_size) / 2
cameras_screen = PerspectiveCameras（focal_length = fcl_screen，principal_point = prp_screen，in_ndc = False，image_size = image_size）
````

相机“focal_length”和“principal_point”的屏幕与 NDC 规格之间的关系由以下等式给出，其中“s = min(image_width, image_height)”。
屏幕和 NDC 之间 x 和 y 坐标的转换与 px 和 py 完全相同。

````
fx_ndc = fx_screen * 2.0 / 秒
fy_ndc = fy_screen * 2.0 / 秒

px_ndc = - (px_screen - image_width / 2.0) * 2.0 / s
py_ndc = - (py_screen - image_height / 2.0) * 2.0 / s
````