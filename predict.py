import os
import torch
import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import UNet


def predict_single_image(image_path, mask_path, weight_path, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型结构，并读取你训练好的权重
    model = UNet(in_ch=1, out_ch=1).to(device)
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"成功加载权重: {weight_path}")
    else:
        raise FileNotFoundError(f"找不到权重文件: {weight_path}")

    model.eval()  # 切换到测试模式（关掉 Dropout 等）

    # 2. 读取图像和真实 Mask (和我们 Dataset 里的处理一模一样)
    img_pil = Image.open(image_path).convert("L")
    img_np = np.array(img_pil, dtype=np.float32) / 255.0
    img_tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 512, 512]

    mask_pil = Image.open(mask_path).convert("L")
    mask_np = np.array(mask_pil, dtype=np.float32) / 255.0
    mask_np = (mask_np > 0.5).astype(np.float32)

    # 3. 模型预测
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output)[0, 0].cpu().numpy()
        pred_mask = (prob > 0.5).astype(np.float32)

    # 4. 画图保存
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Original CT")
    axes[1].imshow(mask_np, cmap="gray")
    axes[1].set_title("Ground Truth (True Mask)")
    axes[2].imshow(prob, cmap="jet")
    axes[2].set_title("Prediction Probability")
    axes[3].imshow(pred_mask, cmap="gray")
    axes[3].set_title("Final Predicted Mask")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"预测结果已保存至: {save_path}")


if __name__ == "__main__":
    # 你随便挑一张验证集里的图片和 mask 路径填进来
    # 路径参考：DATA_ROOT / series名 / "images" / 文件名
    TEST_IMG = r"C:\Users\33987\Desktop\LUNA_Data\RIDER-Lung-CT-Processed\RIDER-Lung-CT-Processed\RIDER-1129164940_14310556\images\slice_130.png"
    TEST_MSK = r"C:\Users\33987\Desktop\LUNA_Data\RIDER-Lung-CT-Processed\RIDER-Lung-CT-Processed\RIDER-1129164940_14310556\masks\slice_130.png"

    WEIGHTS = "weights/best_model.pth"
    SAVE_PATH = "test_prediction_result.png"

    predict_single_image(TEST_IMG, TEST_MSK, WEIGHTS, SAVE_PATH)