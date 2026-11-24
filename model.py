import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 基础积木：连续两次卷积
class DoubleConv(nn.Module):
    """两次 3x3 卷积 + BN + ReLU，用来提取特征"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), # 加速训练，防止梯度爆炸
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

# 2. 整个 U-Net 模型
class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=(32, 64, 128, 256)):
        super().__init__()
        self.downs = nn.ModuleList() # 存放左边向下提取特征的积木
        self.ups = nn.ModuleList()   # 存放右边向上恢复特征的积木
        self.pool = nn.MaxPool2d(2, 2) # 池化层，把 512x512 变成 256x256

        # --- 左边部分（编码器/下采样）：一步步把图变小，提取高级特征 ---
        ch = in_ch
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        # --- 底部（瓶颈层）：图最小的时候，特征最丰富 ---
        self.bottleneck = DoubleConv(ch, ch * 2)

        # --- 右边部分（解码器/上采样）：一步步把图放大回 512x512 ---
        for f in reversed(features):
            # 反卷积，把图放大 2 倍
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(f * 2, f))

        # --- 最后一步：把 32 个通道压缩成 1 个通道，输出最终结果 ---
        self.head = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, x):
        skips = [] # 用来保存左边每一步的图，一会要拼给右边

        # 1. 走下坡路（提取特征）
        for down in self.downs:
            x = down(x)
            skips.append(x) # 存下来！一会右边要用
            x = self.pool(x) # 缩小尺寸

        # 2. 走到底部
        x = self.bottleneck(x)

        # 3. 走上坡路（恢复尺寸）
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x) # 放大
            skip = skips[-(i // 2 + 1)] # 把刚才存下来的图拿出来

            # 如果因为某些原因尺寸差了一点点，强行拉伸对齐（防止报错）
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, skip.shape[-2:], mode="bilinear", align_corners=False)

            # 把左边提取的细节，和右边放大的图“拼”在一起（这就是 U-Net 的精髓）
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x) # 再卷一下

        # 4. 输出最终结果
        return self.head(x)