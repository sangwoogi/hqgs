import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np

input_folder = "sr_seen"

def images_concat(image, kernel_size=5, sigma=1.0):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

features = []

for filename in sorted(os.listdir(input_folder)):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(input_folder, filename)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR, HWC
        if img is None:
            continue

        img = img.astype(np.float32) / 255.0

        filt = images_concat(img)
        out = img - filt  # HWC, 3 channels

        # HWC -> CHW
        out = torch.from_numpy(out).permute(2, 0, 1).contiguous()  # (3, H, W)
        features.append(out)

features_tensor = torch.stack(features, dim=0)  # (N, 3, H, W)
N, C, H, W = features_tensor.shape
assert C == 3, f"Expected 3 channels, got {C}"

features_tensor = features_tensor.view(N, C, H * W).float()            # (N, 3, HW)
features_tensor = F.adaptive_avg_pool1d(features_tensor, 1000)         # (N, 3, 1000)
features_tensor = features_tensor.permute(0, 2, 1).contiguous()        # (N, 1000, 3)

torch.save(features_tensor, "rgb_sr.pth")
