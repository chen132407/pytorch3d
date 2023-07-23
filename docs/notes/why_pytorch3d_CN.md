---
隐藏标题：true
sidebar_label：为什么选择 PyTorch3D
---


# 为什么选择 PyTorch3D


我们使用 PyTorch3D 的目标是帮助加速深度学习和 3D 交叉领域的研究。3D 数据比 2D 图像更复杂，在开展 [Mesh R-CNN](https://github.com/facebookresearch/meshrcnn) 和 [C3DPO](https://github.com/facebookresearch/c3dpo_nrsfm) 等项目时，我们遇到了一些挑战，包括 3D 数据表示、批处理和速度。我们开发了许多用于 3D 深度学习的有用运算符和抽象，并希望与社区分享，以推动该领域的新颖研究。

在 PyTorch3D 中，我们引入了高效的 3D 运算符、异构批处理功能和模块化可微分渲染 API，为该领域的研究人员提供了急需的工具包，以利用复杂的 3D 输入实施前沿研究。