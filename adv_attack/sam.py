import gc
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry

# 初始化 SAM 模型
def init_sam(model_type="vit_h", checkpoint_path="sam_vit_h_4b8939.pth"):
    """
    初始化 SAM 模型和预测器
    model_type: 模型类型 (vit_h, vit_l, vit_b)
    checkpoint_path: 预训练模型权重路径（需自行下载）
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    return sam_predictor

# 销毁模型
def destroy_sam(sam_predictor):
    """
    销毁 SAM 模型，释放占用的内存资源
    :param sam_predictor: 已初始化的 SamPredictor 实例
    """
    # 1. 释放模型占用的 GPU/CPU 内存（关键步骤）
    if hasattr(sam_predictor, 'model'):
        # 将模型移至 CPU（避免 GPU 显存残留）
        sam_predictor.model.cpu()
        # 清空模型参数（释放权重占用的内存）
        del sam_predictor.model
    
    # 2. 删除 SamPredictor 实例本身
    del sam_predictor
    
    # 3. 强制触发垃圾回收（立即释放未使用的内存）
    gc.collect()
    
    # 4. 若使用 GPU，清空 PyTorch 的 CUDA 缓存（彻底释放 GPU 显存）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # 清理 CUDA 进程间通信残留

# 运行 SAM 进行分割


# 处理 BCHW 格式的 tensor 输入（0-1 范围）
# def segment_tensor(predictor, tensor_img, input_point_list=None, input_label=None,mutil_mask=True):
#     """
#     tensor_img: 输入图像，格式为 BCHW（batch=1, channel=3, height, width），值范围 0-1
#     input_point: 引导点坐标 (x, y)，格式为 [[x1, y1], [x2, y2]]
#     input_label: 点标签（1=前景，0=背景）
#     mutil_mask: 是否返回多个掩码
#     返回：
#     - img_np: 输入图像，格式为 numpy 数组，值范围 0-255
#     - masks: SAM 输出的分割掩码，格式为 numpy 数组，形状为 (num_masks, H, W)
#     - masks_tensor: 转换为 tensor 的分割掩码，格式为 tensor，形状为 (num_masks, H, W)
#     - scores: 掩码置信度分数，格式为 numpy 数组，形状为 (num_masks,)
#     """

    
#     # 2. 转换为 SAM 所需的格式：
#     # - 去除 batch 维度（BCHW -> CHW）
#     # - 转换为 numpy 数组（HWC 格式，RGB 通道）
#     # - 缩放至 0-255 并转为 uint8 类型（SAM 要求）
#     if tensor_img.dim() > 3:  # 确保输入是单张图像
#         tensor_img = tensor_img.squeeze(0)

#     img_np = tensor_img.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
#     img_np = (img_np * 255).astype(np.uint8)  # 0-1 -> 0-255
    
#     # 3. 设置图像到 SAM 预测器
#     predictor.set_image(img_np)
    
#     # 4. 处理引导点（如果有）
#     if input_point is not None:
#         input_point = np.array(input_point)
#         input_label = np.array(input_label) if input_label is not None else np.array([1]*len(input_point))
#     else:
#         input_point = None
#         input_label = None
    
#     # 5. 预测分割掩码
#     masks, scores, logits = predictor.predict(
#         point_coords=input_point,
#         point_labels=input_label,
#         multimask_output=mutil_mask
#     )
    
#     # 6. 转换掩码为 tensor（可选，便于后续处理）
#     masks_tensor = torch.from_numpy(masks).float()  # 形状：(num_masks, H, W)
#     return img_np, masks, masks_tensor, scores

import numpy as np
import torch

def segment_tensor(predictor, tensor_img, input_points_batch=None, input_labels_batch=None, mutil_mask=True):
    """
    支持多batch处理的SAM分割函数
    
    Args:
        predictor: SAM预测器实例
        tensor_img: 输入图像，格式为 BCHW（batch, channel=3, height, width），值范围 0-1
        input_points_batch: 引导点坐标的batch列表，格式为 [[[x1, y1], [x2, y2]], [[x3, y3]]] 
                            每个元素对应一个batch的引导点，None表示该batch无引导点
        input_labels_batch: 点标签的batch列表（1=前景，0=背景），格式为 [[1,0], [1]]
                            每个元素对应一个batch的点标签
        mutil_mask: 是否返回多个掩码
    
    Returns:
        - imgs_np: 输入图像列表，每个元素为numpy数组 (H, W, 3)，值范围 0-255
        - masks_list: 分割掩码列表，每个元素为numpy数组 (num_masks, H, W)
        - masks_tensors: 转换为tensor的分割掩码列表，每个元素形状为 (num_masks, H, W)
        - scores_list: 掩码置信度分数列表，每个元素为numpy数组 (num_masks,)
    """
    # 初始化返回列表
    imgs_np_list = []
    masks_list = []
    masks_tensors_list = []
    scores_list = []
    
    # 获取batch大小
    batch_size = tensor_img.shape[0]
    _, C, H, W = tensor_img.shape
    if input_labels_batch is None:
        input_labels_batch = [None] * batch_size
    
    # 确保引导点和标签的长度匹配batch size
    assert len(input_points_batch) == batch_size, \
        f"input_points_batch长度({len(input_points_batch)})应等于batch size({batch_size})"
    assert len(input_labels_batch) == batch_size, \
        f"input_labels_batch长度({len(input_labels_batch)})应等于batch size({batch_size})"
    
    # 遍历每个batch处理
    for i in range(batch_size):
        # 获取当前batch的图像
        img_tensor = tensor_img[i]  # CHW格式
        
        # 转换为SAM所需格式：HWC, 0-255, uint8
        img_np_i = img_tensor.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
        img_np_i = (img_np_i * 255).astype(np.uint8)  # 0-1 -> 0-255
        
        # 设置当前图像到SAM预测器
        predictor.set_image(img_np_i)
        
        # 处理当前batch的引导点和标签
        input_point = input_points_batch[i]
        input_label = input_labels_batch[i]
        
        if len(input_point)>0 :
            input_point = np.array(input_point)
            # 如果标签未提供，默认全部为前景(1)
            if input_label is None:
                input_label = np.array([1] * len(input_point))
            else:
                input_label = np.array(input_label)

        
            # 预测分割掩码
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=mutil_mask
            )
        else:
            # 没有，则默认为全部为前景
            
            # 无引导点的情况：生成全前景掩码（形状与SAM输出对齐）
            num_masks = 3 if mutil_mask else 1  # 匹配SAM的multimask_output行为
            # 创建全1掩码 (num_masks, H, W)
            masks = np.ones((num_masks, H, W), dtype=np.bool_)  # 掩码用布尔型更高效
            # 生成默认置信度分数（模拟SAM输出）
            scores = np.array([1.0] * num_masks)  # 全1表示最高置信度
            # 生成空logits（保持与predict输出格式一致）
            logits = np.zeros((num_masks, H, W), dtype=np.float32)

        # 转换掩码为tensor,detach
        
        masks_tensor = torch.from_numpy(masks).float()  # (num_masks, H, W)
        masks_tensor = masks_tensor.detach()
        # 将结果添加到列表
        imgs_np_list.append(img_np_i)
        masks_list.append(masks)
        
        masks_tensors_list.append(masks_tensor)
        scores_list.append(scores)
    masks_tensors=torch.stack(masks_tensors_list)
    imgs_np=np.stack(imgs_np_list)
    return imgs_np, masks_list, masks_tensors, scores_list

def visualize_sam(tensor_img, masks, scores, save_path="sam_segment_result.png"):
    """
    可视化 SAM 分割结果并保存图片
    
    参数：
        tensor_img: 输入图像 tensor（BCHW 或 CHW，0-1 范围）
        masks: SAM 输出的分割掩码（numpy 数组，形状 (num_masks, H, W)）
        scores: 掩码置信度分数（numpy 数组，形状 (num_masks,)）
        save_path: 保存路径（默认保存为 sam_segment_result.png）
    """
    # 原始 tensor 转为 HWC 格式（0-1 范围，可直接用 Matplotlib 显示）
    if tensor_img.dim() > 3:  # 确保输入是单张图像（去除 batch 维度）
        tensor_img = tensor_img.squeeze(0)

    img_show = tensor_img.permute(1, 2, 0).cpu().numpy()
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, len(masks)+1, 1)
    plt.imshow(img_show)  # Matplotlib 直接支持 0-1 范围的 float 数组
    plt.title("Original Tensor")
    plt.axis("off")
    
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.subplot(1, len(masks)+1, i+2)
        plt.imshow(img_show)
        plt.imshow(mask, alpha=0.5, cmap="jet")  # 叠加掩码（半透明）
        plt.title(f"Mask {i+1} (Score: {score:.2f})")
        plt.axis("off")
    
    # 关键：保存图片（需在 plt.show() 之前调用，避免空白图片）
    plt.tight_layout()  # 自动调整布局，避免标签重叠
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 确保保存目录存在
    plt.savefig(save_path, dpi=150, bbox_inches="tight")  # dpi 控制分辨率

# 主函数示例
if __name__ == "__main__":
    # 初始化模型（需替换权重路径）
    predictor = init_sam(
        model_type="vit_b",
        checkpoint_path=r"D:\FILELin\postgraduate\little_paper\Adversariall_attack_project\ControlNet\models\sam_vit_b_01ec64.pth"  # 替换为你的权重路径
    )
    # 读取图片，
    image = cv2.imread(r"D:\FILELin\postgraduate\little_paper\Adversariall_attack_project\ControlNet\test_imgs\dog.png")
    image =cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    tensor_img = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

    # 生成示例 tensor（BCHW, 0-1 范围）
    
    # 引导点（图像中心）
    input_point = [[320, 240]]  # (x=320, y=240)，对应宽度 640、高度 480
    input_label = [1]  # 前景
    
    # 分割
    img_np, masks, masks_tensor, scores = segment_tensor(
        predictor, tensor_img, input_point, input_label
    )
    
    # 可视化
    visualize_sam(tensor_img, masks, scores)


