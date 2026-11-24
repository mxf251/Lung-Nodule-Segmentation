import torch
from model import UNet

# 1. 实例化模型
model = UNet(in_ch=1, out_ch=1)
print("模型初始化成功！")

# 2. 造一个假的输入（模拟 DataLoader 吐出来的一个 batch，batch_size=2）
fake_input = torch.randn(2, 1, 512, 512)
print("假输入形状:", fake_input.shape)

# 3. 把假输入喂给模型
output = model(fake_input)

# 4. 打印输出形状
print("模型输出形状:", output.shape)