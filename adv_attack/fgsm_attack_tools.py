import torch
from typing import Optional, List
import sys
import os
import cv2

sys.path.append(os.path.dirname(__file__))
from object_detection_class import ObjectDetection
from util import *

# ======================================================
# ROI mask builder
# ======================================================
def build_roi_mask(
    img_shape,
    boxes: torch.Tensor,
    device
):
    """
    img_shape: (B, C, H, W)
    boxes: Tensor [M, 4] (xyxy)
    return: mask [1, 1, H, W]
    """
    _, _, H, W = img_shape
    mask = torch.zeros((1, 1, H, W), device=device)

    for box in boxes:
        x1, y1, x2, y2 = box.int()
        x1 = torch.clamp(x1, 0, W - 1)
        x2 = torch.clamp(x2, 0, W - 1)
        y1 = torch.clamp(y1, 0, H - 1)
        y2 = torch.clamp(y2, 0, H - 1)
        mask[:, :, y1:y2, x1:x2] = 1.0

    return mask


# ======================================================
# FGSM for Object Detection (ROI-aware)
# ======================================================
def fgsm_od(
    x: torch.Tensor,
    od_model: ObjectDetection,
    model_type: str,
    boxes_list: Optional[List[torch.Tensor]] = None,
    targeted: bool = False,
    eps: float = 0.03,
    x_val_min: float = 0.0,
    x_val_max: float = 1.0,
    target_label: Optional[int] = None,
    loss_type: str = "confidence"
) -> torch.Tensor:

    x_adv = x.detach().clone().to(od_model.device)
    x_adv.requires_grad_(True)

    # ---- forward ----
    results, _ = od_model.detect_eval(
        images=x_adv,
        model_type=model_type,
        grad_status=True
    )

    loss_list = []
    B = x_adv.shape[0]

    for b in range(B):
        boxes = results["boxes"][b]
        scores = results["scores"][b]
        labels = results["labels"][b]
        scores_vector = results["scores_vector"][b]

        if len(boxes) == 0:
            continue

        # ----- confidence loss -----
        if loss_type in ["confidence", "combined"]:
            if targeted:
                if target_label is None:
                    raise ValueError("target_label required for targeted attack")
                mask = labels == target_label
                if mask.any():
                    conf_loss = -scores[mask].mean()
                else:
                    conf_loss = -scores_vector[:, target_label].mean()
            else:
                conf_loss = scores.mean()

        # ----- box loss -----
        box_loss = torch.zeros(1, device=od_model.device)
        if loss_type in ["box", "combined"]:
            cx = (boxes[:, 0] + boxes[:, 2]) / 2
            cy = (boxes[:, 1] + boxes[:, 3]) / 2
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            box_loss = (cx.abs() + cy.abs() + w.abs() + h.abs()).mean()

        # ----- total -----
        if loss_type == "confidence":
            loss_list.append(conf_loss)
        elif loss_type == "box":
            loss_list.append(box_loss)
        else:
            loss_list.append(conf_loss + 0.1 * box_loss)

    if len(loss_list) == 0:
        return x.detach()

    total_loss = torch.stack(loss_list).sum()

    if x_adv.grad is not None:
        x_adv.grad.zero_()
    total_loss.backward()

    grad_sign = x_adv.grad.sign()

    # ===== ROI constraint =====
    if boxes_list is not None:
        roi_mask = build_roi_mask(
            x_adv.shape,
            boxes_list[0],
            x_adv.device
        )
        grad_sign = grad_sign * roi_mask

    # ===== FGSM update =====
    if targeted:
        x_adv = x_adv + eps * grad_sign
    else:
        x_adv = x_adv - eps * grad_sign

    return torch.clamp(x_adv, x_val_min, x_val_max).detach()


# ======================================================
# i-FGSM (ROI-aware)
# ======================================================
def ifgsm_od(
    x: torch.Tensor,
    od_model: ObjectDetection,
    model_type: str,
    boxes_list: Optional[List[torch.Tensor]] = None,
    targeted: bool = False,
    eps: float = 0.03,
    alpha: float = 2 / 255,
    iteration: int = 10,
    x_val_min: float = 0.0,
    x_val_max: float = 1.0,
    target_label: Optional[int] = None,
    loss_type: str = "confidence"
) -> torch.Tensor:

    x_ori = x.detach().clone().to(od_model.device)
    x_adv = x_ori.clone()

    for _ in range(iteration):
        x_adv = fgsm_od(
            x=x_adv,
            od_model=od_model,
            model_type=model_type,
            boxes_list=boxes_list,
            targeted=targeted,
            eps=alpha,
            target_label=target_label,
            loss_type=loss_type
        )

        delta = torch.clamp(x_adv - x_ori, -eps, eps)
        x_adv = torch.clamp(x_ori + delta, x_val_min, x_val_max)

    return x_adv.detach()


# ======================================================
# Example Usage
# ======================================================
if __name__ == "__main__":

    od_model = ObjectDetection(
        nclass_yaml_path="models/coco_class.yaml",
        conf_threshold=0.25,
        iou_threshold=0.2,
        image_size=512
    )

    od_model.load_model(
        model_type="yolov11",
        model_path="models/yolo11m.pt"
    )

    img_path = r"exp/for_paper/exp_example/origin.jpg"

    img = cv2.imread(img_path)
    img = cv2.resize(img, (512, 512))
    img_tensor = cv2_to_tensor(img)
    test_img = img_tensor.unsqueeze(0).to(od_model.device)

    # ===== original detection =====
    res_ori, _ = od_model.detect(test_img, model_type="yolov11")
    boxes_list = [res_ori["boxes"][0]]

    # ===== ROI FGSM =====
    adv_fgsm = fgsm_od(
        x=test_img,
        od_model=od_model,
        model_type="yolov11",
        boxes_list=boxes_list,
        eps=0.08
    )

    # ===== ROI i-FGSM =====
    adv_ifgsm = ifgsm_od(
        x=test_img,
        od_model=od_model,
        model_type="yolov11",
        boxes_list=boxes_list,
        eps=0.08,
        iteration=10
    )

    # ===== save =====
    tensor2picture(test_img, "ori.png")
    tensor2picture(adv_fgsm, "adv_fgsm_roi.png")
    tensor2picture(adv_ifgsm, "adv_ifgsm_roi.png")

    print("Original boxes:", len(res_ori["boxes"][0]))
