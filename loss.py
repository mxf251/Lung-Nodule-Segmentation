import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss：直接优化“预测和真实重叠了多少”，专治结节小、背景多"""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth  # 平滑项，防止分母为 0

    def forward(self, logits, targets):
        # 1) 模型输出是 logits，先 sigmoid 压到 0~1 变成概率
        probs = torch.sigmoid(logits)

        # 2) 按“每张图”分别算 Dice，再取平均
        #    不这样写的话：一张图里结节再大也只占 1 票，大结节会把小结节的分数带偏
        dims = (1, 2, 3)  # 对 C、H、W 三个维度求和，保留 batch 维
        intersection = (probs * targets).sum(dim=dims)
        union = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # 3) Loss 越小越好，所以用 1 - dice
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    """BCE + Dice 混合：BCE 管每个像素判得对不对，Dice 管整体形状像不像"""

    def __init__(self, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice = DiceLoss(smooth)

    def forward(self, logits, targets):
        # 注意：这里用 binary_cross_entropy_with_logits
        # 它自带 sigmoid，所以模型输出不要再手动 sigmoid，否则会算错（双重 sigmoid 梯度会崩）
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        dice = self.dice(logits, targets)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice
