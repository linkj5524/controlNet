import torch
import torchvision
import numpy as np
import cv2
import os
# ------------------------------
# 1. 适配 ObjectDetection 类的 IG/IDG 核心函数
# ------------------------------
# def IG_Detection(
#     input_img,  # 输入图像张量 (b, 3, H, W)，归一化到 0-1（符合 ObjectDetection 输入要求）
#     det_model,  # 你的 ObjectDetection 实例
#     steps=50,
#     batch_size=10,
#     alpha_star=1.0,
#     baseline=0.0,  # 基线图像（0=全黑，可传入张量）
#     target_obj_idx=0  # 要归因的目标索引（检测结果中第几个目标，默认第1个）
# ):
#     """
#     适配目标检测的集成梯度（IG/LIG）归因方法
#     核心：对检测结果中特定目标的类别置信度进行归因
#     """
#     device = det_model.device
#     if steps % batch_size != 0:
#         print(f"Error: steps ({steps}) must be divisible by batch_size ({batch_size})!")
#         return None

#     loops = int(steps / batch_size)

#     # 生成基线图像（适配输入尺寸和设备）
#     if torch.is_tensor(baseline):
#         baseline = baseline.to(device).float()
#         assert baseline.shape == input_img.shape, "Baseline shape must match input shape!"
#     else:
#         baseline = torch.full(input_img.shape, baseline, dtype=torch.float, device=device)

#     input_img = input_img.to(device)
#     baseline_diff = input_img - baseline  # 输入与基线的差值

#     # 生成插值系数（0→1）
#     alphas = torch.linspace(0, 1, steps, device=device).reshape(steps, 1, 1, 1)
#     alphas.requires_grad = True

#     # 存储梯度、目标置信度
#     gradients = torch.zeros((steps, input_img.shape[1], input_img.shape[2], input_img.shape[3]), device=device)
#     target_scores = torch.zeros(steps, device=device)  # 存储特定目标的类别置信度

#     # 批量计算插值图像的梯度和置信度
#     for i in range(loops):
#         start = i * batch_size
#         end = (i + 1) * batch_size

#         # 生成插值图像（baseline + alpha * (input - baseline)）
#         interp_imgs = baseline + alphas[start:end] * baseline_diff

#         # 计算当前批次的梯度和目标置信度
#         batch_grads, batch_scores = getGradientsDetection(interp_imgs, det_model, target_obj_idx)
#         gradients[start:end] = batch_grads
#         target_scores[start:end] = batch_scores

#     # 计算 IG/LIG 梯度积分
#     if alpha_star == 1.0:
#         # IG：积分所有步骤
#         integrated_grads = gradients.mean(dim=0)
#     else:
#         # LIG：积分到置信度达到 max_score * alpha_star 的步骤
#         max_score = torch.max(target_scores)
#         cutoff_score = max_score * alpha_star
#         cutoff_steps = torch.where(target_scores > cutoff_score)[0]

#         # 处理无有效截断点的情况
#         cutoff_step = cutoff_steps[0] if len(cutoff_steps) > 0 else 1
#         cutoff_step = max(cutoff_step, 1)  # 避免截断点为0

#         # 积分前 cutoff_step 个步骤
#         integrated_grads = gradients[:cutoff_step].mean(dim=0)

#     # 最终归因值 = 积分梯度 * (输入 - 基线)
#     attributions = integrated_grads * baseline_diff[0]  # (3, H, W)
#     return attributions.squeeze()  # (3, H, W) 或 (H, W)（若单通道）

def IG_Detection(
    input_img,  # 输入图像张量 (B, 3, H, W)，归一化到 0-1
    det_model,  # ObjectDetection 实例
    model_type=None,
    steps=50,
    batch_size=10,
    alpha_star=1.0,
    baseline=0.0,  # 基线图像（0=全黑，可传入(B,3,H,W)张量）
    target_obj=0  # 要归因的目标索引（每个batch内的第几个目标）
):
    """
    适配目标检测的集成梯度（IG/LIG）归因方法（支持多batch）
    核心：对检测结果中特定目标的类别置信度进行归因
    
    Args:
        input_img: (B, 3, H, W) 输入图像张量
        det_model: 目标检测模型实例
        steps: 插值步数
        batch_size: 批量计算的步长
        alpha_star: LIG截断系数（1.0=标准IG）
        baseline: 基线值/张量（标量或(B,3,H,W)张量）
        target_obj_idx: 每个batch内要归因的目标索引
    
    Returns:
        attributions: (B, 3, H, W) 多batch的归因结果
    """
    device = det_model.device
    B, C, H, W = input_img.shape  # 获取batch_size和图像维度
    
    # 输入校验
    if steps % batch_size != 0:
        print(f"Error: steps ({steps}) must be divisible by batch_size ({batch_size})!")
        return None
    loops = int(steps / batch_size)

    # 生成基线图像（适配多batch）
    if torch.is_tensor(baseline):
        baseline = baseline.to(device).float()
        assert baseline.shape == input_img.shape, f"Baseline shape {baseline.shape} must match input shape {input_img.shape}!"
    else:
        baseline = torch.full((B, C, H, W), baseline, dtype=torch.float, device=device)

    input_img = input_img.to(device)
    baseline_diff = input_img - baseline  # (B, 3, H, W) 每个batch的输入与基线差值

    # 生成插值系数（0→1）：(steps, 1, 1, 1, 1) 适配广播到 (steps, B, 3, H, W)
    alphas = torch.linspace(0, 1, steps, device=device).reshape(steps, 1, 1, 1, 1)
    alphas.requires_grad = True

    # 存储梯度：(steps, B, 3, H, W) → 新增batch维度
    gradients = torch.zeros((steps, B, C, H, W), device=device)
    # 存储目标置信度：(steps, B) → 每个step、每个batch的目标置信度
    target_scores = torch.zeros((steps, B), device=device)

    # 批量计算插值图像的梯度和置信度
    for i in range(loops):
        start = i * batch_size
        end = (i + 1) * batch_size
        current_alphas = alphas[start:end]  # (batch_size, 1, 1, 1, 1)

        # 生成插值图像：(batch_size, B, 3, H, W)
        interp_imgs = baseline + current_alphas * baseline_diff  # 广播适配

        # 展开维度用于模型推理：(batch_size*B, 3, H, W),先batch、后通道维度
        interp_imgs_flat = interp_imgs.reshape(-1, C, H, W)
        # 扩展目标索引

        target_obj_idx =  [item for item in target_obj for _ in range(batch_size)] 
        # 计算当前批次的梯度和目标置信度（需确保getGradientsDetection支持多batch）
        batch_grads_flat, batch_scores_flat = getGradientsDetection(
                                                interp_imgs=interp_imgs_flat,
                                                det_model= det_model,
                                                target_obj_idx= target_obj_idx,
                                                model_type= model_type
        )
        
        # 恢复维度：(batch_size, B, 3, H, W)
        batch_grads = batch_grads_flat.reshape(batch_size, B, C, H, W)
        # 恢复维度：(batch_size, B)
        batch_scores = batch_scores_flat.reshape(batch_size, B)

        gradients[start:end] = batch_grads
        target_scores[start:end] = batch_scores

    # 计算 IG/LIG 梯度积分（按batch维度分别计算）
    if alpha_star == 1.0:
        # IG：对steps维度求平均 → (B, 3, H, W)
        integrated_grads = gradients.mean(dim=0)
    else:
        # LIG：每个batch独立计算截断点
        integrated_grads = torch.zeros((B, C, H, W), device=device)
        max_scores = torch.max(target_scores, dim=0)[0]  # (B,) 每个batch的最大置信度
        cutoff_scores = max_scores * alpha_star  # (B,) 每个batch的截断阈值

        for b in range(B):
            # 找到当前batch的截断步长
            batch_scores = target_scores[:, b]
            cutoff_steps = torch.where(batch_scores > cutoff_scores[b])[0]
            
            # 处理无有效截断点的情况
            cutoff_step = cutoff_steps[0] if len(cutoff_steps) > 0 else 1
            cutoff_step = max(cutoff_step, 1)  # 避免截断点为0
            
            # 积分前cutoff_step个步骤 → (3, H, W)
            integrated_grads[b] = gradients[:cutoff_step, b].mean(dim=0)

    # 最终归因值：每个batch独立计算 (B, 3, H, W)
    attributions = integrated_grads * baseline_diff  # (B, 3, H, W)
    return attributions  # 返回多batch结果 (B, 3, H, W)

def IDG_Detection(
    input_img,
    det_model,
    steps=50,
    batch_size=10,
    baseline=0.0,
    target_obj_idx=0
):
    """
    适配目标检测的集成决策梯度（IDG）归因方法
    核心：基于置信度斜率的非均匀插值，提升归因精准度
    """
    device = det_model.device
    if batch_size == 0 or steps % batch_size != 0:
        print(f"Error: steps ({steps}) must be divisible by batch_size ({batch_size})!")
        return None

    loops = int(steps / batch_size)

    # 生成基线图像
    if torch.is_tensor(baseline):
        baseline = baseline.to(device).float()
        assert baseline.shape == input_img.shape, "Baseline shape must match input shape!"
    else:
        baseline = torch.full(input_img.shape, baseline, dtype=torch.float, device=device)

    input_img = input_img.to(device)
    baseline_diff = input_img - baseline

    # 第一步：计算初始均匀插值的置信度斜率
    slopes, step_size = getSlopesDetection(baseline, baseline_diff, det_model, steps, batch_size, target_obj_idx)
    # 第二步：基于斜率生成非均匀插值系数
    alphas, alpha_substep_size = getAlphaParameters(slopes, steps, step_size)

    alphas = alphas.to(device).reshape(steps, 1, 1, 1)
    alpha_substep_size = alpha_substep_size.to(device).reshape(steps, 1, 1, 1)
    alphas.requires_grad = True

    # 存储梯度、目标置信度、斜率
    gradients = torch.zeros((steps, input_img.shape[1], input_img.shape[2], input_img.shape[3]), device=device)
    target_scores = torch.zeros(steps, device=device)
    slopes = torch.zeros(steps, device=device)

    # 批量计算非均匀插值图像的梯度和置信度
    for i in range(loops):
        start = i * batch_size
        end = (i + 1) * batch_size

        interp_imgs = baseline + alphas[start:end] * baseline_diff
        batch_grads, batch_scores = getGradientsDetection(interp_imgs, det_model, target_obj_idx)
        gradients[start:end] = batch_grads
        target_scores[start:end] = batch_scores

    # 计算置信度斜率（相邻步骤的置信度变化率）
    slopes[0] = 0.0
    for i in range(steps - 1):
        alpha_diff = alphas[i+1] - alphas[i]
        score_diff = target_scores[i+1] - target_scores[i]
        slopes[i+1] = score_diff / (alpha_diff + 1e-8)  # 避免除零

    # 梯度加权：用斜率加权 + 非均匀插值步长修正
    gradients = gradients * slopes.reshape(steps, 1, 1, 1)  # 斜率加权
    gradients = gradients * alpha_substep_size  # 步长修正

    # 积分并计算最终归因值
    integrated_grads = gradients.mean(dim=0)
    attributions = integrated_grads * baseline_diff[0]
    return attributions.squeeze()


# ------------------------------
# 2. 适配检测模型的辅助函数
# ------------------------------
def getGradientsDetection(interp_imgs, det_model, target_obj_idx,model_type=None):
    """
    修复：直接为批量张量启用梯度，避免非叶子张量修改错误
    计算插值图像对特定目标置信度的梯度
    :param interp_imgs: 插值图像批次 (batch_size, 3, H, W)
    :param det_model: ObjectDetection 实例
    :param target_obj_idx: 目标索引（检测结果中第几个目标）
    :return: 梯度 (batch_size, 3, H, W) + 目标置信度 (batch_size,)
    """
    batch_size = interp_imgs.shape[0]
    gradients = []
    target_scores = []

    # 关键修复：为整个批量张量启用梯度（interp_imgs 是叶子张量）
    interp_imgs=interp_imgs.clone().detach().to(det_model.device)
    interp_imgs.requires_grad = True
    det_model.models[model_type].zero_grad()  # 清空模型梯度

    # 批量推理（一次处理整个批次，提升效率）
    for i in range(batch_size):
        img = interp_imgs[i:i+1]  # 取单个图像 (1, 3, H, W)，仍共享批量张量的梯度
        results,_ = det_model.detect(img,model_type=model_type, grad_status=True)

        # 提取特定目标的类别置信度（若检测不到目标，置信度设为0）
        if len(results['scores'][0]) <= 0:
            score = torch.tensor(0.0, device=img.device, requires_grad=True)
            target_scores.append(score)
            continue
            # score = torch.tensor(0.0, device=img.device, requires_grad=True)
        else:
            # 判断是否是目标label
            score=get_max_score_for_label(results,target_obj_idx[i])
            # score = results['scores'][0][target_obj_idx[i]]  # (1,) 张量

        # 存储置信度（后续统一反向传播）
        target_scores.append(score)

    # 批量反向传播：计算所有图像的梯度（求和后反向传播，等价于逐个传播）
    total_score = torch.stack(target_scores).sum()  # 批量置信度求和
    total_score.backward()  # 反向传播计算梯度
    # 提取每个图像的梯度
    ## 判断是否存在grad，不存在，则创建一个全0张量
    if interp_imgs.grad is None:
        interp_imgs.grad = torch.zeros_like(interp_imgs)
    
    
    for i in range(batch_size):
        
        grad = interp_imgs.grad[i:i+1].detach().squeeze()  # (3, H, W)
        gradients.append(grad)

    return torch.stack(gradients), torch.stack(target_scores)


# def getGradientsDetection(interp_imgs, det_model, target_obj_idx):
#     """
#     稳定版：计算插值图像对特定目标置信度的梯度（兼容检测不到目标的情况）
#     :param interp_imgs: 插值图像批次 (batch_size, 3, H, W)
#     :param det_model: ObjectDetection 实例（YOLO类）
#     :param target_obj_idx: 目标索引（检测结果中第几个目标）
#     :return: 
#         gradients: 梯度张量 (batch_size, 3, H, W)
#         target_scores: 目标置信度 (batch_size,)
#     """
#     # --------------------------
#     # 1. 核心准备：确保输入张量是叶子张量且开启梯度
#     # --------------------------
#     batch_size = interp_imgs.shape[0]
#     device = det_model.device
    
#     # 重新创建叶子张量（避免原张量是非叶子/梯度被冻结）
#     interp_imgs = interp_imgs.to(device, non_blocking=True)
#     interp_imgs.requires_grad_(True)  # 等价于 requires_grad = True（更安全）
#     interp_imgs.retain_grad()  # 强制保留叶子张量的梯度（关键！）
    
#     # 清空模型和张量的旧梯度（避免累积）
#     det_model.model.zero_grad(set_to_none=True)  # 比zero_()更高效
#     if interp_imgs.grad is not None:
#         interp_imgs.grad.zero_()

#     # --------------------------
#     # 2. 批量推理：禁用no_grad，确保梯度链路完整
#     # --------------------------
#     target_scores = []
#     det_model.model.train()  # 模型切到训练模式（启用梯度）
    
#     for i in range(batch_size):
#         # 避免切片：直接用索引访问（减少非叶子张量）
#         img = interp_imgs[i:i+1].contiguous()  # 保持维度 (1, 3, H, W)
        
#         # 关键：强制禁用no_grad（覆盖detect_return_dict内部的no_grad）
#         with torch.enable_grad():
#             results = det_model.detect_return_dict(img, grad_status=True)
        
#         # --------------------------
#         # 3. 安全提取目标得分（保证梯度链路不中断）
#         # --------------------------
#         if len(results['scores'][0]) <= target_obj_idx:
#             # 隐患修复：用与输入关联的0值张量（而非全新张量）
#             score = torch.zeros(1, device=device, requires_grad=True)
#             # 强制建立梯度关联（即使得分是0，也能回传梯度）
#             score = score + interp_imgs[i:i+1].sum() * 1e-10  # 极小值不影响结果，但保留链路
#         else:
#             # 提取得分并确保是标量（避免维度问题）
#             score = results['scores'][0][target_obj_idx].squeeze()  # 标量张量
        
#         target_scores.append(score)

#     # --------------------------
#     # 4. 批量反向传播 + 梯度校验
#     # --------------------------
#     # 求和后反向传播（等价于逐个传播，效率更高）
#     total_score = torch.stack(target_scores).sum()
    
#     # 反向传播（保留计算图，方便调试）
#     total_score.backward(retain_graph=True)
    
#     # 梯度非空校验（核心：避免TypeError）
#     if interp_imgs.grad is None:
#         raise RuntimeError(
#             f"梯度计算失败！原因：\n"
#             f"1. interp_imgs.requires_grad = {interp_imgs.requires_grad}\n"
#             f"2. total_score = {total_score.item()}\n"
#             f"3. 模型是否在train模式：{det_model.model.training}"
#         )

#     # --------------------------
#     # 5. 提取梯度（安全切片）
#     # --------------------------
#     gradients = []
#     for i in range(batch_size):
#         # 安全提取：先判断梯度是否存在，再切片
#         grad = interp_imgs.grad[i].detach().contiguous()  # (3, H, W)
#         gradients.append(grad)

#     # --------------------------
#     # 6. 结果整理
#     # --------------------------
#     gradients = torch.stack(gradients)  # (batch_size, 3, H, W)
#     target_scores = torch.stack(target_scores)  # (batch_size,)

#     return gradients, target_scores



def getSlopesDetection(baseline, baseline_diff, det_model, steps, batch_size, target_obj_idx):
    """
    计算均匀插值下的置信度斜率（用于 IDG 的非均匀插值生成）
    :return: 斜率数组 (steps,) + 均匀步长
    """
    device = det_model.device
    loops = int(steps / batch_size)

    # 生成均匀插值系数
    alphas = torch.linspace(0, 1, steps, device=device).reshape(steps, 1, 1, 1)
    target_scores = torch.zeros(steps, device=device)

    # 批量计算置信度
    for i in range(loops):
        start = i * batch_size
        end = (i + 1) * batch_size
        interp_imgs = baseline + alphas[start:end] * baseline_diff
        for idx, img in enumerate(interp_imgs):
            img = img.unsqueeze(0)
            results = det_model.detect_return_dict(img, grad_status=False)
            # 提取特定目标的置信度
            if len(results['scores'][0]) <= target_obj_idx:
                score = 0.0
            else:
                score = results['scores'][0][target_obj_idx].item()
            target_scores[start + idx] = score

    # 计算斜率（相邻步骤的置信度变化率）
    slopes = torch.zeros(steps, device=device)
    step_size = 1.0 / (steps - 1)  # 均匀步长
    for i in range(steps - 1):
        slopes[i+1] = (target_scores[i+1] - target_scores[i]) / step_size

    return slopes, step_size


def getAlphaParameters(slopes, steps, step_size):
    """
    基于斜率生成非均匀插值系数（复用原逻辑，无修改）
    """
    # 归一化斜率到 [0, 1]
    slopes_0_1_norm = (slopes - torch.min(slopes)) / (torch.max(slopes) - torch.min(slopes) + 1e-8)
    slopes_0_1_norm[0] = 0.0  # 第一个斜率设为0（基线点无变化）

    # 归一化斜率和为1
    slopes_sum_1_norm = slopes_0_1_norm / torch.sum(slopes_0_1_norm + 1e-8)

    # 分配每个插值点的采样步数
    sample_placements_float = slopes_sum_1_norm * steps
    sample_placements_int = sample_placements_float.type(torch.int)
    remaining_to_fill = steps - torch.sum(sample_placements_int)

    # 填充剩余步数
    non_zeros = torch.where(sample_placements_int != 0)[0]
    sample_placements_float[non_zeros] = -1.0
    remaining_hi_lo = torch.flip(torch.sort(sample_placements_float)[1], dims=[0])
    sample_placements_int[remaining_hi_lo[:remaining_to_fill]] = 1

    # 生成非均匀插值系数和步长
    alphas = torch.zeros(steps)
    alpha_substep_size = torch.zeros(steps)
    alpha_start_index = 0
    alpha_start_value = 0.0

    for num_samples in sample_placements_int:
        if num_samples == 0:
            continue
        # 线性分配当前区间的插值点
        alpha_range = torch.linspace(alpha_start_value, alpha_start_value + step_size, num_samples + 1)[:-1]
        alphas[alpha_start_index:alpha_start_index + num_samples] = alpha_range
        # 记录当前区间的步长
        alpha_substep_size[alpha_start_index:alpha_start_index + num_samples] = step_size / num_samples
        # 更新起始位置和值
        alpha_start_index += num_samples
        alpha_start_value += step_size

    return alphas, alpha_substep_size


# ------------------------------
# 3. 可视化工具函数（将归因结果转为热力图）
# ------------------------------
def visualize_attribution(input_img, attributions, save_path="attribution_result",file_name_pre='attribution1'):
    """
    可视化归因结果：原始图像 + 归因热力图 + 叠加图
    :param input_img: 输入图像张量 (1, 3, H, W)（归一化到0-1）
    :param attributions: 归因结果 (3, H, W) 或 (H, W)
    :param save_path: 保存路径
    """
    # 处理输入图像（张量→numpy）
    for i in range(input_img.shape[0]):  # 批量处理
        input_img_b= input_img[i].detach()
        attributions_b= attributions[i].detach()
        input_img_b = input_img_b.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        input_img_b = (input_img_b * 255).astype(np.uint8)  # 反归一化
        input_img_b = cv2.cvtColor(input_img_b, cv2.COLOR_RGB2BGR)  # RGB→BGR（适配OpenCV）

        # 处理归因结果（多通道→单通道热力图）
        if len(attributions_b.shape) == 3:
            attributions_b = torch.abs(attributions_b).sum(dim=0)  # 按通道取绝对值求和
        attributions_b = attributions_b.detach().cpu().numpy()

        # 归一化到 [0, 255]
        attributions_b = (attributions_b - attributions_b.min()) / (attributions_b.max() - attributions_b.min() + 1e-8)
        attributions_b = (attributions_b * 255).astype(np.uint8)

        # 生成彩色热力图
        heatmap = cv2.applyColorMap(attributions_b, cv2.COLORMAP_JET)
        # 叠加原始图像和热力图
        overlay = cv2.addWeighted(input_img_b, 0.6, heatmap, 0.4, 0)

        # 保存结果
        attribution_path = os.path.join(save_path[i], file_name_pre+'_attribution.jpg')
        attribution_path_cmp = os.path.join(save_path[i], file_name_pre+'_attribution_cmp.jpg')
        cv2.imwrite(attribution_path,heatmap )
        cv2.imwrite(attribution_path_cmp, np.hstack([input_img_b, heatmap, overlay]))
        print(f"Attribution visualization saved to {save_path[i]}")


# ------------------------------
# 4. 测试代码（完整流程：加载模型→预处理图像→归因→可视化）
# ------------------------------
if __name__ == "__main__":
    # 配置参数
    MODEL_TYPE = "yolov11"
    MODEL_PATH = r"models\yolo11n.pt"  # 替换为你的YOLOv11模型路径
    IMAGE_PATH = r"test_imgs\bird1.png"  # 替换为你的测试图像路径
    TARGET_OBJ_IDX = 0  # 归因第1个检测到的目标
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import os
    import sys
    # 此文件的根目录
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    #添加环境
    
    sys.path.append(ROOT_DIR)
    from object_detection_class import *
    # 1. 初始化目标检测模型
    det_model = ObjectDetection(
        model_type=MODEL_TYPE,
        model_path=MODEL_PATH,
        device=DEVICE,
        conf_threshold=0.25,
        iou_threshold=0.7
    )

    # 2. 预处理图像（适配 ObjectDetection 输入要求）
    def load_image_for_detection(image_path, device):
        """加载图像并转为 (1, 3, H, W) 张量（归一化到0-1）"""
        img = cv2.imread(image_path)
        img = cv2.resize(img, (640, 640))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR→RGB
        img_tensor = torch.from_numpy(img_rgb).float() / 255.0  # 归一化到0-1
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (H, W, 3)→(1, 3, H, W)
        return img_tensor.to(device)

    input_img = load_image_for_detection(IMAGE_PATH, DEVICE)

    # 3. 运行 IG 归因（也可替换为 IDG_Detection）
    print("Running IG attribution...")
    attributions = IG_Detection(
        input_img=input_img,
        det_model=det_model,
        steps=50,
        batch_size=10,
        alpha_star=1.0,
        baseline=0.0,
        target_obj_idx=TARGET_OBJ_IDX
    )

    # 4. 可视化结果
    if attributions is not None:
        visualize_attribution(input_img, attributions, save_path="ig_detection_result.png")
    else:
        print("Attribution failed!")








def get_max_score_for_label(results: dict, ref_label: int) -> torch.Tensor:
    """
    找到与参考label匹配的最大score（全程张量操作，保留梯度，不转换类型）
    
    Args:
        results: 检测结果字典，'labels'/'scores' 为张量列表（GPU/CPU均可）
        ref_label: 参考标签（目标整数）
    
    Returns:
        torch.Tensor: 匹配参考label的最大score张量（标量）；无匹配返回 0.0 标量张量（保留梯度图）
    """
    # 初始化最大score为0.0标量张量（与输入张量同设备、同dtype，保证梯度兼容）
    max_score = None



    # 遍历batch维度，全程张量操作
    for batch_idx in range(len(results['labels'])):
        label_tensor = results['labels'][batch_idx]  # 保留梯度的张量
        score_tensor = results['scores'][batch_idx]  # 保留梯度的张量

        # 1. 张量维度统一（兼容标量/一维张量）
        label_flat = label_tensor.flatten()  # 展平为一维，不影响梯度
        score_flat = score_tensor.flatten()  # 展平为一维，不影响梯度

        # 2. 筛选匹配ref_label的label（张量比较，保留梯度）
        # 生成匹配掩码：1表示匹配，0表示不匹配
        match_mask = (label_flat == ref_label).float()
        # 仅保留匹配项的score，不匹配项置0
        matched_score = score_flat * match_mask

        # 3. 取当前batch的最大匹配score（标量张量）
        curr_max = matched_score.max()

        # 4. 更新全局最大score（张量操作，保留梯度）
        if max_score is None:
            max_score = curr_max
        else:
            # 取两个张量的最大值（保留梯度）
            max_score = torch.max(max_score, curr_max)

    # 无匹配时返回0.0标量张量（开启梯度）
    if max_score is None:
        # 自动匹配输入张量的设备和dtype
        device = results['scores'][0].device
        dtype = results['scores'][0].dtype
        max_score = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

    return max_score