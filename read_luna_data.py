import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image_dir = r"C:\Users\33987\Desktop\LUNA_Data\RIDER-Lung-CT-Processed\RIDER-Lung-CT-Processed\RIDER-1129164940_14310556\images"
mask_dir = r"C:\Users\33987\Desktop\LUNA_Data\RIDER-Lung-CT-Processed\RIDER-Lung-CT-Processed\RIDER-1129164940_14310556\masks"

image_names = sorted(os.listdir(image_dir))

# 🌟 写个循环，寻找第一张有结节的图（Mask里有白点）
found_nodule = False
for name in image_names:
    mask_path = os.path.join(mask_dir, name)
    mask_array = np.array(Image.open(mask_path).convert('L'))

    # 如果 Mask 里面的最大值大于 0，说明有白点（有结节）
    if mask_array.max() > 0:
        print(f"🎯 找到有结节的切片了：{name}")

        # 把原图和 Mask 读出来显示
        img_path = os.path.join(image_dir, name)
        img_array = np.array(Image.open(img_path).convert('L'))

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title(f"CT Image: {name}")
        plt.imshow(img_array, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Nodule Mask (White Spot)")
        plt.imshow(mask_array, cmap='gray')
        plt.axis('off')
        plt.show()

        found_nodule = True
        break  # 找到一张就停止

if not found_nodule:
    print("这个文件夹里没有找到有结节的图，可能在另一个子文件夹里。")