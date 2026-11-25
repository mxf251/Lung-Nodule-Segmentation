import os
import time
import argparse
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from dataset import LungNoduleDataset
from model import UNet
from loss import BCEDiceLoss

import matplotlib

matplotlib.use("Agg")  # 只存图不弹窗，PyCharm 里更稳，也不会因为弹窗卡住
import matplotlib.pyplot as plt

# ================= 1. 配置参数 =================
# 数据放在项目外面（图片不进 Git），所以这里必须写“绝对路径”
# 项目里写 r"data/images" 是找不到的：相对路径是相对于“运行目录”，而项目根目录下根本没有 data 文件夹
DATA_ROOT = r"C:\Users\33987\Desktop\LUNA_Data\RIDER-Lung-CT-Processed\RIDER-Lung-CT-Processed"

SAVE_DIR = "predictions"   # 预测对比图
WEIGHT_DIR = "weights"     # 权重


def collect_pairs(data_root, positive_only=True):
    """遍历 DATA_ROOT 下每个 series 的 images/masks，按文件名配对

    positive_only=True：丢掉 mask 全黑的切片（没结节，学不到东西，还会稀释 loss）
    """
    if not os.path.isdir(data_root):
        raise SystemExit(f"数据根目录不存在: {data_root}\n请检查 DATA_ROOT 这一行有没有写错")

    pairs = []
    for series in sorted(os.listdir(data_root)):
        img_dir = os.path.join(data_root, series, "images")
        msk_dir = os.path.join(data_root, series, "masks")
        if not (os.path.isdir(img_dir) and os.path.isdir(msk_dir)):
            continue  # 不是数据文件夹，跳过

        names = sorted(set(os.listdir(img_dir)) & set(os.listdir(msk_dir)))
        for name in names:
            img_path = os.path.join(img_dir, name)
            msk_path = os.path.join(msk_dir, name)
            if positive_only:
                mask = np.array(Image.open(msk_path).convert("L"))
                if (mask > 127).sum() == 0:
                    continue
            pairs.append((img_path, msk_path, series))
    return pairs


def split_by_series(pairs, val_ratio=0.15):
    """按 series（≈ 按病人）划分训练/验证

    为什么不能随机按切片分：同一个 CT 的相邻切片几乎一样，
    随机分会让验证集里出现“训练时见过的邻居”，Dice 虚高，属于数据泄漏
    """
    series_list = sorted({p[2] for p in pairs})
    n_val = max(1, int(round(len(series_list) * val_ratio)))
    step = max(1, len(series_list) // n_val)
    val_series = set(series_list[::step][:n_val])

    train = [p for p in pairs if p[2] not in val_series]
    val = [p for p in pairs if p[2] in val_series]
    return train, val


def dice_score(logits, targets, thr=0.5, smooth=1.0):
    """验证用：整批像素的 Dice，0~1，越高越好"""
    preds = (torch.sigmoid(logits) > thr).float()
    inter = (preds * targets).sum()
    return ((2 * inter + smooth) / (preds.sum() + targets.sum() + smooth)).item()


def save_prediction_image(model, batch, save_path, dice=None):
    """存一张 原图 | 真实Mask | 预测Mask 的对比图"""
    if batch is None:
        return
    model.eval()
    with torch.no_grad():
        images = batch["image"].to(next(model.parameters()).device)
        labels = batch["label"].to(next(model.parameters()).device)

        img = images[0, 0].cpu().numpy()
        label = labels[0, 0].cpu().numpy()
        prob = torch.sigmoid(model(images))[0, 0].cpu().numpy()
        pred = (prob > 0.5).astype(np.float32)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title("CT Image")
        axes[1].imshow(label, cmap="gray")
        axes[1].set_title("True Mask")
        axes[2].imshow(prob, cmap="jet")
        axes[2].set_title("Pred Probability")
        axes[3].imshow(pred, cmap="gray")
        axes[3].set_title(f"Pred Mask (dice={dice:.3f})" if dice is not None else "Pred Mask")
        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)  # Windows 下 0 最稳
    parser.add_argument("--smoke", action="store_true", help="冒烟测试：只跑 3 个 batch，确认能跑通")
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(WEIGHT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")
    if device.type == "cuda":
        print(f"显卡: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True

    # ================= 2. 准备数据 =================
    pairs = collect_pairs(DATA_ROOT, positive_only=True)
    if not pairs:
        raise SystemExit(f"{DATA_ROOT} 里没找到任何 images/masks 配对，检查一下路径层级")

    train_pairs, val_pairs = split_by_series(pairs)
    print(f"阳性切片共 {len(pairs)} 张 → 训练 {len(train_pairs)} 张 / 验证 {len(val_pairs)} 张")
    print(f"训练 series {len({p[2] for p in train_pairs})} 个，验证 series {len({p[2] for p in val_pairs})} 个")

    train_set = LungNoduleDataset([p[0] for p in train_pairs], [p[1] for p in train_pairs])
    val_set = LungNoduleDataset([p[0] for p in val_pairs], [p[1] for p in val_pairs])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # ================= 3. 模型 / Loss / 优化器 =================
    model = UNet(in_ch=1, out_ch=1).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_dice = 0.0

    # ================= 4. 训练 + 每轮验证 =================
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        step = 0

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (step + 1) % 20 == 0:
                mem = torch.cuda.memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0
                print(f"Epoch[{epoch + 1}/{args.epochs}] Step[{step + 1}/{len(train_loader)}] "
                      f"loss={loss.item():.4f} 显存={mem:.0f}MB")

            if args.smoke and step + 1 >= 3:
                print(">>> smoke 模式：只跑 3 个 batch，训练循环提前结束")
                break

        train_loss = running_loss / max(1, step + 1)

        # ---------- 验证 ----------
        model.eval()
        val_loss_sum, val_dice_sum, viz_batch, n_val = 0.0, 0.0, None, 0
        with torch.no_grad():
            for v_step, batch in enumerate(val_loader):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                outputs = model(images)
                val_loss_sum += criterion(outputs, labels).item() * images.size(0)
                val_dice_sum += dice_score(outputs, labels) * images.size(0)
                n_val += images.size(0)
                viz_batch = batch
                if args.smoke and v_step + 1 >= 2:
                    break

        # 除以“实际跑过的样本数”，而不是整个验证集大小（smoke 模式只跑几个 batch，不然会算小）
        n_val = max(1, n_val)
        val_loss = val_loss_sum / n_val
        val_dice = val_dice_sum / n_val
        scheduler.step()

        print(f"=== Epoch {epoch + 1}/{args.epochs} | 训练loss={train_loss:.4f} | "
              f"验证loss={val_loss:.4f} | 验证Dice={val_dice:.4f} | 用时={time.time() - t0:.0f}s ===")

        # 每轮都存一张验证集预测图，肉眼跟着看有没有变好
        save_prediction_image(model, viz_batch, os.path.join(SAVE_DIR, f"epoch_{epoch + 1:02d}.png"), val_dice)

        # 以验证 Dice 为“更好的模型”标准（loss 会骗人，Dice 更贴近你关心的重叠度）
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, "best_model.pth"))
            print(f"🌟 验证 Dice 新高 {best_dice:.4f}，已保存到 weights/best_model.pth")

    print(f"训练结束！最好的验证 Dice = {best_dice:.4f}")


if __name__ == "__main__":
    main()
