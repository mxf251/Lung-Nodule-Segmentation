import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth  # 平滑项，防止分母为0报错

    def forward(self, predictions, targets):
        # 1. 模型输出的是原始得分(logits)，必须先用 sigmoid 压到 0~1 之间变成概率
        predictions = torch.sigmoid(predictions)

        # 2. 把 [B, 1, H, W] 的立体结构“拍扁”成一维，方便算所有像素的重叠
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # 3. 算交集和并集
        intersection = (predictions * targets).sum()

        # 4. Dice 公式：(2 * 交集) / (预测总和 + 真实总和)
        dice = (2.0 * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)

        # 5. Loss 越小越好，所以用 1 减去 dice
        return 1 - dice