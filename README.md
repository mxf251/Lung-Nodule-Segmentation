# 2D Lung Nodule Segmentation (2D 肺结节分割)

基于 PyTorch 实现的 2D U-Net 肺结节分割项目。使用 RIDER Lung CT 数据集进行训练，能够自动在肺部 CT 切片中定位并分割出结节病灶。

## 🛠️ 项目特点 (Features)
- **数据管道**: 使用 PyTorch `Dataset` 和 `DataLoader` 自定义数据加载，解决医疗图像单通道读取与 Mask 二值化问题。
- **数据增强**: 引入随机水平/垂直翻转（仅训练集），防止小样本过拟合。
- **防数据泄漏划分**: 按照 `Series`（病人维度）划分训练集和验证集，确保验证集不会出现训练时见过的相邻切片。
- **混合损失函数**: 采用 `BCEWithLogitsLoss + DiceLoss` 混合损失，有效解决正负样本极度不平衡（背景占99%以上）的问题。
- **自动化训练与评估**: 结合学习率余弦退火（CosineAnnealingLR），每轮自动保存最佳 Dice 权重及预测对比图。

## 📁 目录结构 (Structure)
```text
Lung-Nodule-Segmentation/
├── dataset.py       # 自定义数据集读取与数据增强
├── model.py         # 2D U-Net 模型搭建 (输入输出通道为1)
├── loss.py          # BCE + Dice 混合损失函数
├── train.py         # 训练主脚本 (包含训练、验证、自动保存最优模型)
├── predict.py       # 独立推理脚本 (加载.pth权重生成对比图)
└── README.md