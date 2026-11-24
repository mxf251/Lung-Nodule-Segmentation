import os
import torch
from torch.utils.data import DataLoader
from dataset import LungNoduleDataset
from model import UNet
from loss import DiceLoss

# ================= 1. 配置参数 =================
IMAGE_DIR = r"C:\Users\33987\Desktop\LUNA_Data\RIDER-Lung-CT-Processed\RIDER-Lung-CT-Processed\RIDER-1129164940_14310556\images"  # 确保路径正确
MASK_DIR = r"C:\Users\33987\Desktop\LUNA_Data\RIDER-Lung-CT-Processed\RIDER-Lung-CT-Processed\RIDER-1129164940_14310556\masks"
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-3  # 学习率

# ================= 2. 准备数据 =================
image_files = sorted(os.listdir(IMAGE_DIR))
mask_files = sorted(os.listdir(MASK_DIR))
common = sorted(set(image_files) & set(mask_files))
image_paths = [os.path.join(IMAGE_DIR, f) for f in common]
mask_paths = [os.path.join(MASK_DIR, f) for f in common]

dataset = LungNoduleDataset(image_paths, mask_paths)
# shuffle=True 防止一个 batch 全是没结节的图
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# ================= 3. 初始化模型、Loss、优化器 =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用设备: {device}")

model = UNet(in_ch=1, out_ch=1).to(device)
criterion = DiceLoss()  # 用我们刚写的 Dice Loss
# 优化器（这是帮模型更新参数的引擎）
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ================= 4. 开始训练！ =================
for epoch in range(EPOCHS):
    model.train()  # 告诉模型：现在是在训练模式
    epoch_loss = 0.0

    for batch_idx, batch in enumerate(loader):
        # 把数据丢到显卡上
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # --- 标准四步曲 ---
        optimizer.zero_grad()  # 1. 清空上一次的梯度
        outputs = model(images)  # 2. 模型前向预测
        loss = criterion(outputs, labels)  # 3. 算损失
        loss.backward()  # 4. 反向传播（算梯度）
        optimizer.step()  # 5. 更新参数

        epoch_loss += loss.item()

        # 每跑 10 个 batch 打印一下进度
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Step [{batch_idx + 1}/{len(loader)}], Loss: {loss.item():.4f}")

    # 每个 Epoch 结束，打印平均 loss
    avg_loss = epoch_loss / len(loader)
    print(f"=== Epoch {epoch + 1} 结束, 平均 Dice Loss: {avg_loss:.4f} ===")

# ================= 5. 保存模型权重 =================
# 建个 weights 文件夹保存
os.makedirs("weights", exist_ok=True)
torch.save(model.state_dict(), "weights/unet_epoch10.pth")
print("训练结束，模型已保存到 weights/unet_epoch10.pth")