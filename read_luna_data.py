import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # 🌟 全新的神级武器：Python 图像库

# 1. 设定路径（请替换为你电脑里真实的 RIDER Lung CT 路径）
image_dir = r"C:\Users\33987\Desktop\LUNA_Data\RIDER-Lung-CT-Processed\RIDER-Lung-CT-Processed\RIDER-1129164940_14310556\images"
mask_dir = r"C:\Users\33987\Desktop\LUNA_Data\RIDER-Lung-CT-Processed\RIDER-Lung-CT-Processed\RIDER-1129164940_14310556\masks"

# 2. 获取图片列表
image_names = sorted(os.listdir(image_dir))
first_image_name = image_names[0]

image_path = os.path.join(image_dir, first_image_name)
mask_path = os.path.join(mask_dir, first_image_name)

print(f"正在读取: {first_image_name}")

try:
    # 3. 使用 PIL 读取图像，并强制转换为单通道灰度图 ('L' 模式)
    # 转换为 numpy 数组后，尺寸格式就是 (高度, 宽度)
    img_array = np.array(Image.open(image_path).convert('L'))
    mask_array = np.array(Image.open(mask_path).convert('L'))

    print(f"✅ 原图尺寸: {img_array.shape}, 掩码尺寸: {mask_array.shape}")

    # 4. 画图对比（使用 matplotlib）
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("CT Image")
    plt.imshow(img_array, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Nodule Mask")
    plt.imshow(mask_array, cmap='gray')
    plt.axis('off')

    plt.show()

except Exception as e:
    print(f"❌ 报错了: {e}")
    print("请检查文件夹路径是否正确，以及里面是否有图片。")