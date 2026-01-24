import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from adv_attack.util import *

sys.path.append(os.path.dirname(__file__))


class PatchTransformer(nn.Module):
    def __init__(self, device, cfg_patch):
        super().__init__()
        self.device = device
        self.cfg_patch = cfg_patch

    def median_pooler(self, patch):
        return patch

    def random_jitter(self, patch):
        return patch

    def cutout(self, patch):
        return patch

    def forward(self, adv_batch, bboxes_batch, patch_ori_size,
                rand_rotate_gate, rand_shift_gate):
        return adv_batch


# =========================================================
# Patch Random Applier（新增占比控制）
# =========================================================
class PatchRandomApplier(nn.Module):
    def __init__(self, device: torch.device, cfg_patch: object):
        super().__init__()
        self.cfg = cfg_patch
        self.device = device
        self.patch_transformer = PatchTransformer(device, cfg_patch).to(device)

        # >>>>>>> 新增：patch 占检测框比例 <<<<<<<<
        self.max_patch_ratio = getattr(cfg_patch, "MAX_PATCH_RATIO", 1.0)

    def apply_patch(self, img_batch: torch.Tensor,
                    patch_tensor: torch.Tensor,
                    results: dict) -> torch.Tensor:

        if img_batch.dim() == 3:
            img_batch = img_batch.unsqueeze(0)
        B, C, H, W = img_batch.shape

        if patch_tensor.dim() == 4:
            patch_tensor = patch_tensor.squeeze(0)
        if patch_tensor.dim() != 3:
            raise ValueError("patch_tensor must be C×H×W")

        boxes = results['boxes']
        N = len(boxes[0])
        gates = self.cfg.TRANSFORM

        adv_batch = patch_tensor.unsqueeze(0).unsqueeze(0)
        adv_batch = adv_batch.expand(B, N, -1, -1, -1)

        if gates.get('median_pool', False):
            adv_batch = self.patch_transformer.median_pooler(adv_batch)

        if gates.get('jitter', False):
            adv_batch = self.patch_transformer.random_jitter(adv_batch)

        adv_batch = torch.clamp(adv_batch, 0.0, 1.0)

        if gates.get('cutout', False):
            adv_batch = self.patch_transformer.cutout(adv_batch)

        adv_batch = self.patch_transformer(
            adv_batch,
            boxes,
            patch_tensor.size(-1),
            rand_rotate_gate=gates.get('rotate', False),
            rand_shift_gate=gates.get('shift', False),
        )

        return self.apply_transformed_patch(img_batch, adv_batch, boxes)

    def apply_transformed_patch(self,
                                img_batch: torch.Tensor,
                                adv_batch: torch.Tensor,
                                boxes: list) -> torch.Tensor:
        """
        新增：控制 adv_patch 占检测框比例
        """
        adv_img_batch = img_batch.clone()
        B = img_batch.size(0)

        for i in range(B):
            for j, box in enumerate(boxes[i]):
                x1, y1, x2, y2 = box.int()
                box_h = y2 - y1
                box_w = x2 - x1

                patch = adv_batch[i, j]  # [C, Hp, Wp]

                # >>>>>>> 新增：patch 尺寸占比控制 <<<<<<<<
                target_h = int(box_h * self.max_patch_ratio)
                target_w = int(box_w * self.max_patch_ratio)

                target_h = max(1, target_h)
                target_w = max(1, target_w)

                patch = F.interpolate(
                    patch.unsqueeze(0),
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

                # >>>>>>> 新增：居中贴在检测框内 <<<<<<<<
                y_start = y1 + (box_h - target_h) // 2
                x_start = x1 + (box_w - target_w) // 2
                y_end = y_start + target_h
                x_end = x_start + target_w

                adv_img_batch[i, :, y_start:y_end, x_start:x_end] = patch

        return adv_img_batch


# =========================================================
# main（保持你原来的写法）
# =========================================================
if __name__ == "__main__":

    img_path = r"exp\for_paper\exp_example\origin.jpg"
    patch_path = r"exp\for_paper\exp_example\sample.jpg"

    img = cv2.imread(img_path)
    img = cv2.resize(img, (512, 512))
    img_tensor = cv2_to_tensor(img)      # C H W

    patch_img = cv2.imread(patch_path)
    patch_img = cv2.resize(patch_img, (512, 512))
    patch_tensor = cv2_to_tensor(patch_img)

    boxes_all = [torch.tensor([[50, 300, 300, 500]])]
    scores_all = [torch.tensor([0.9])]
    labels_all = [torch.tensor([1])]
    scores_vector_all = [torch.tensor([[0.1, 0.9]])]

    results = {
        'boxes': boxes_all,
        'scores': scores_all,
        'labels': labels_all,
        'scores_vector': scores_vector_all
    }

    class Config:
        TRANSFORM = {
            'rotate': True,
            'shift': True,
            'median_pool': True,
            'jitter': True,
            'cutout': True
        }
        MAX_PATCH_RATIO = 0.4   # 👈 这里控制 patch 占检测框比例

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patch_applier = PatchRandomApplier(device, Config())

    adv_img_batch = patch_applier.apply_patch(img_tensor, patch_tensor, results)

    tensor2picture(adv_img_batch, 'output.png')
    print("调试完成，处理后的图像已保存。")
