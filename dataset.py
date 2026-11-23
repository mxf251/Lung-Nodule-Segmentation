import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class LungNoduleDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        """
        image_paths: list，所有 image 的完整路径
        mask_paths:  list，所有 mask 的完整路径，顺序与 image_paths 一一对应
        """
        assert len(image_paths) == len(mask_paths), "images 和 masks 数量不一致"
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        # DataLoader 靠这个知道一轮有多少个样本
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1) 用 PIL 读图，统一转成灰度单通道
        #    convert("L") 是关键：不管原图是 RGB 还是 RGBA，读出来都是 1 通道
        image = Image.open(self.image_paths[idx]).convert("L")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        # 2) PIL -> numpy float32，并除以 255 归一化到 0~1
        image_np = np.array(image, dtype=np.float32) / 255.0
        mask_np = np.array(mask, dtype=np.float32) / 255.0

        # 3) Mask 二值化：大于 0.5 算结节(1)，否则背景(0)
        #    这一步保证 label 里只有 0.0 和 1.0，Dice Loss 才能正确算重叠
        mask_np = (mask_np > 0.5).astype(np.float32)

        # 4) numpy -> tensor，unsqueeze(0) 把 (H,W) 变成 (1,H,W)
        #    PyTorch 卷积要求输入是 (C,H,W)，所以必须加这个通道维
        image_tensor = torch.tensor(image_np).unsqueeze(0)  # (1, 512, 512)
        mask_tensor = torch.tensor(mask_np).unsqueeze(0)    # (1, 512, 512)

        return {"image": image_tensor, "label": mask_tensor}