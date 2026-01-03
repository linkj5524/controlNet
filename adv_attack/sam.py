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


def segment_tensor(predictor, tensor_img, 
                   input_points_batch=None, 
                   input_labels_batch=None, 
                   input_boxes_batch=None,
                   mutil_mask=True):
    """
    兼容旧版SAM的多batch分割函数（适配box/coords旧参数）
    """
    imgs_np_list = []
    masks_list = []
    masks_tensors_list = []
    scores_list = []
    
    batch_size = tensor_img.shape[0]
    _, C, H, W = tensor_img.shape
    
    # 初始化默认值
    if input_points_batch is None:
        input_points_batch = [None] * batch_size
    if input_labels_batch is None:
        input_labels_batch = [None] * batch_size
    if input_boxes_batch is None:
        input_boxes_batch = [None] * batch_size
    
    # 校验batch长度
    assert len(input_points_batch) == batch_size
    assert len(input_labels_batch) == batch_size
    assert len(input_boxes_batch) == batch_size
    
    for i in range(batch_size):
        # 1. 处理图像格式
        img_tensor = tensor_img[i]
        img_np_i = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_np_i = (img_np_i * 255).astype(np.uint8)
        predictor.set_image(img_np_i)
        
        # 2. 提取当前batch的引导点/框
        input_point = input_points_batch[i]
        input_label = input_labels_batch[i]
        input_box = input_boxes_batch[i]
        
        # 3. 格式化输入（适配旧版SAM）
        coords = None       # 旧版点坐标参数
        labels = None       # 旧版点标签参数
        box = None          # 旧版框参数
        
        # 处理点
        if input_point is not None :
            coords = np.array(input_point)
            if input_label is None:
                labels = np.array([1] * len(coords))
            else:
                labels = np.array(input_label)
        
        # 处理框（旧版要求box为(1,4)格式）
        if input_box is not None :
            box = np.array(input_box.cpu().detach()).reshape(1, 4)
        
        # 4. SAM预测（核心：兼容旧版参数）
        if box is not None or coords is not None:
            predict_kwargs = {"multimask_output": mutil_mask}
            if coords is not None:
                predict_kwargs["coords"] = coords
                predict_kwargs["labels"] = labels
            if box is not None:
                predict_kwargs["box"] = box
            
            # 调用旧版predict（无box_coords/input_boxes）
            masks, scores, logits = predictor.predict(**predict_kwargs)
        else:
            # 无引导点/框：生成全前景掩码
            num_masks = 3 if mutil_mask else 1
            masks = np.ones((num_masks, H, W), dtype=np.bool_)
            scores = np.array([1.0] * num_masks)
            logits = np.zeros((num_masks, H, W), dtype=np.float32)
        
        # 5. 收集结果
        masks_tensor = torch.from_numpy(masks).float().detach()
        imgs_np_list.append(img_np_i)
        masks_list.append(masks)
        masks_tensors_list.append(masks_tensor)
        scores_list.append(scores)
    
    masks_tensors = torch.stack(masks_tensors_list)
    imgs_np = np.stack(imgs_np_list)
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


