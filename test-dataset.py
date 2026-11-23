import os
from torch.utils.data import DataLoader
from dataset import LungNoduleDataset

# ====== 改成你的实际路径 ======
IMAGE_DIR = r"C:\Users\33987\Desktop\LUNA_Data\RIDER-Lung-CT-Processed\RIDER-Lung-CT-Processed\RIDER-1129164940_14310556\images"
MASK_DIR = r"C:\Users\33987\Desktop\LUNA_Data\RIDER-Lung-CT-Processed\RIDER-Lung-CT-Processed\RIDER-1129164940_14310556\masks"

# 1) 列出两边文件，只保留同名文件（自动跳过缺失的）
image_files = sorted(os.listdir(IMAGE_DIR))
mask_files = sorted(os.listdir(MASK_DIR))
common = sorted(set(image_files) & set(mask_files))

image_paths = [os.path.join(IMAGE_DIR, f) for f in common]
mask_paths = [os.path.join(MASK_DIR, f) for f in common]

print(f"images: {len(image_files)}, masks: {len(mask_files)}, 同名可用: {len(common)}")

# 2) 建 Dataset
dataset = LungNoduleDataset(image_paths, mask_paths)
print(f"Dataset 长度: {len(dataset)}")

# 3) 单样本先测一下（最容易定位问题）
sample = dataset[0]
print("单个样本 image shape:", sample["image"].shape)
print("单个样本 label shape:", sample["label"].shape)
print("image dtype:", sample["image"].dtype)
print("label dtype:", sample["label"].dtype)
print("image min/max:", sample["image"].min().item(), sample["image"].max().item())
print("label 唯一值:", sample["label"].unique())

# 4) 再测 DataLoader 一个 batch
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
batch = next(iter(loader))
print("\n===== DataLoader 一个 batch =====")
print("image batch shape:", batch["image"].shape)
print("label batch shape:", batch["label"].shape)