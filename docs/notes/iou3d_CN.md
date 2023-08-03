<!--
 * @Author: tiger
 * @Date: 2023-07-23 08:18:07
 * @FilePath: \AI_github_example\CV_git\pytorch3d-main\pytorch3d\docs\notes\iou3d_CN.md
-->
---
隐藏标题：true
侧边栏标签：IoU3D
---

# 定向 3D 框并集上的交集：一种新算法

作者：乔治亚·吉奥萨里

实施：Georgia Gkioxari 和 Nikhila Ravi

＃＃ 描述

框的并集交集 (IoU) 被广泛用作对象检测中的评估指标（[1][pascalvoc]、[2][coco]）。
在 2D 中，IoU 通常应用于轴对齐的框，即边缘平行于图像轴的框。
在 3D 中，盒子通常不是轴对齐的，并且可以以世界上任何方式定向。
我们引入了一种新算法，可以计算两个*定向 3D 框*的*精确* IoU。

我们的算法基于简单的观察，即两个定向 3D 框“box1”和“box2”的交集是一个凸多面体（2D 中的凸 n 边形），其中“n > 2”由连接的*平面单元*组成。
在 3D 中，这些平面单元是 3D 三角形面。
在 2D 中，它们是 2D 边。
每个平面单元严格属于“box1”或“box2”。
我们的算法通过迭代每个盒子的侧面来找到这些单位。

1. 对于“box1”中的每个 3D 三角形面“e”，我们检查“e”是否在“box2”*内部*。
2. 如果 `e` 不在*内部*，那么我们丢弃它。
3. 如果 `e` 是 *inside* 或 *partially inside*，则 `e` *inside* `box2` 的部分将添加到构成最终交叉形状的单元中。
4. 我们对“box2”重复。

下面，我们展示了针对 2D 定向框情况的算法的可视化。

<p对齐=“中心”>
<img src="assets/iou3d.gif" alt="绘图" width="400"/>
</p>

请注意，当盒子的单位“e”“部分位于”“盒子”内时，“e”就会分解为更小的单位。在 2D 中，“e”是一条边，并分成更小的边。在 3D 中，“e”是一个 3D 三角形面，并被与其相交的“box”平面剪裁成更多更小的面。
这是 2D 和 3D 算法之间唯一的根本区别。

## 与其他算法的比较

当前的 3D 框 IoU 算法依赖于粗略近似或进行框假设，例如它们限制 3D 框的方向。
[Objectron][objectron] 对先前作品的局限性进行了很好的讨论。
[Objectron][objectron] 引入了一种出色的算法，用于精确计算定向 3D 框的 IoU。
Objectron 的算法使用 [Sutherland-Hodgman 算法][clipalgo] 计算两个框的交点。
相交形状是使用 [Qhull 库][qhull] 由相交点的凸包形成的。

我们的算法比 Objectron 的算法有几个优点：

* 我们的算法还计算交点，类似于 Objectron，但此外还存储点所属的“平面单位”。这消除了对凸包计算的需要，该计算是“O(nlogn)”并且依赖于第三方库，该库经常因不伦不类的错误消息而崩溃。
* Objectron 的实现假设框是偏离轴对齐的旋转。我们的算法和实现没有做出这样的假设，并且适用于任何 3D 盒子。
* 我们的实现支持批处理，与 Objectron 不同，Objectron 假设“box1”和“box2”为单个元素输入。
* 我们的实现很容易并行化，事实上我们提供了一个自定义的 C++/CUDA 实现，它比 Objectron 快**450 倍**。

下面我们比较了 Objectron（C++ 语言）和我们的算法（C++ 和 CUDA 语言）的性能。我们对目标检测中的一个常见用例进行基准测试，其中“boxes1”保存 M 个预测，“boxes2”保存图像中的 N 个真实 3D 框并计算“MxN”IoU 矩阵。我们报告“M=N=16”的时间（以毫秒为单位）。

<p对齐=“中心”>
<img src="assets/iou3d_comp.png" alt="绘图" width="400"/>
</p>

## 用法和代码

````蟒蛇
<!-- 从 pytorch3d.ops 导入 box3d_overlap
# 假设输入：boxes1 (M, 8, 3) 和boxs2 (N, 8, 3)
交集_vol，iou_3d = box3d_overlap（框1，框2） -->
from pytorch3d.ops import box3d_overlap
# Assume inputs: boxes1 (M, 8, 3) and boxes2 (N, 8, 3)
intersection_vol, iou_3d = box3d_overlap(boxes1, boxes2)
````

有关更多详细信息，请阅读 [iou_box3d.py](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/iou_box3d.py)。

请注意，我们的实现目前是不可微分的。我们计划很快添加渐变支持。

我们还进行了广泛的[测试](https://github.com/facebookresearch/pytorch3d/blob/main/tests/test_iou_box3d.py)，将我们的实现与 Objectron 和 MeshLab 进行比较。


## 引用

如果您使用我们的 3D IoU 算法，请引用 PyTorch3D

````bibtex
@文章{ravi2020pytorch3d，
    作者 = {尼基拉·拉维、杰里米·雷森斯坦、大卫·诺沃特尼和泰勒·戈登
                  以及 Wan-Yen Lo、Justin Johnson 和 Georgia Gkioxari}，
    title = {使用 PyTorch3D 加速 3D 深度学习}，
    期刊 = {arXiv:2007.08501},
    年 = {2020}，
}
````

[pascalvoc]：http://host.robots.ox.ac.uk/pascal/VOC/
[可可]：https://cocodataset.org/
[objectron]：https://arxiv.org/abs/2012.09988
[qhull]：http://www.qhull.org/
[clipalgo]：https://en.wikipedia.org/wiki/Sutherland%E2%80%93Hodgman_algorithm