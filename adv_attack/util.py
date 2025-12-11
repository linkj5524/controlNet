import math
import os
import re
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import generalized_box_iou  # GIoU计算工具
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
# 纯PyTorch实现的匈牙利算法（无改动，确保不依赖外部库）
# def hungarian_matching(cost_matrix):
#     """
#     纯PyTorch实现的匈牙利算法，用于解决指派问题
#     Args:
#         cost_matrix: 成本矩阵，shape [M, K]
#     Returns:
#         row_ind: 行索引（预测框索引）
#         col_ind: 列索引（真实框索引）
#     """
#     device = cost_matrix.device
#     M, K = cost_matrix.shape
    
#     u = torch.zeros(M + 1, device=device)
#     v = torch.zeros(K + 1, device=device)
#     p = torch.zeros(K + 1, dtype=torch.long, device=device)
#     way = torch.zeros(K + 1, dtype=torch.long, device=device)
    
#     for i in range(1, M + 1):
#         p[0] = i
#         minv = torch.full((K + 1,), float('inf'), device=device)
#         used = torch.zeros(K + 1, dtype=torch.bool, device=device)
#         j0 = 0
#         while True:
#             used[j0] = True
#             i0 = p[j0]
#             delta = float('inf')
#             j1 = 0
#             for j in range(1, K + 1):
#                 if not used[j]:
#                     cur = cost_matrix[i0 - 1, j - 1] - u[i0] - v[j]
#                     if cur < minv[j]:
#                         minv[j] = cur
#                         way[j] = j0
#                     if minv[j] < delta:
#                         delta = minv[j]
#                         j1 = j
#             for j in range(K + 1):
#                 if used[j]:
#                     u[p[j]] += delta
#                     v[j] -= delta
#                 else:
#                     minv[j] -= delta
#             j0 = j1
#             if p[j0] == 0:
#                 break
#         while True:
#             j1 = way[j0]
#             p[j0] = p[j1]
#             j0 = j1
#             if j0 == 0:
#                 break
    
#     # 过滤无效匹配（p[j] == 0 表示无匹配）
#     valid_col_mask = p[1:K+1] != 0
#     col_ind = torch.arange(1, K + 1, device=device)[valid_col_mask] - 1  # 转为0-based
#     row_ind = p[1:K+1][valid_col_mask] - 1  # 转为0-based
    
#     return row_ind, col_ind



def hungarian_matching(cost_matrix):
    """
    纯PyTorch实现的匈牙利算法，用于解决指派问题（支持非方阵）
    Args:
        cost_matrix: 成本矩阵，shape [M, K]
    Returns:
        row_ind: 行索引（预测框索引）
        col_ind: 列索引（真实框索引）
    """
    device = cost_matrix.device
    M, K = cost_matrix.shape
    N = max(M, K)  # 方阵大小
    
    # 填充矩阵为方阵（添加虚拟行或列）
    if M != K:
        # 创建填充值（使用矩阵最大值的1.1倍，避免影响真实匹配）
        # 关键修复：将张量转换为数值（使用.item()）
        if M * K > 0:
            fill_value = (torch.max(cost_matrix) * 1.1).item()  # 转为Python数值
        else:
            fill_value = 0.0
        # 初始化方阵
        square_matrix = torch.full((N, N), fill_value, device=device)
        # 填充原始数据
        square_matrix[:M, :K] = cost_matrix
    else:
        square_matrix = cost_matrix.clone()
    
    # 算法核心变量
    u = torch.zeros(N + 1, device=device)  # 行标签
    v = torch.zeros(N + 1, device=device)  # 列标签
    p = torch.zeros(N + 1, dtype=torch.long, device=device)  # 记录列匹配的行
    way = torch.zeros(N + 1, dtype=torch.long, device=device)  # 记录交替路径
    
    for i in range(1, N + 1):
        p[0] = i
        minv = torch.full((N + 1,), float('inf'), device=device)  # 记录最小缩减成本
        used = torch.zeros(N + 1, dtype=torch.bool, device=device)  # 标记已使用的列
        j0 = 0
        
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float('inf')
            j1 = 0
            
            # 寻找下一个最佳列
            for j in range(1, N + 1):
                if not used[j]:
                    cur = square_matrix[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            
            # 更新标签和最小成本
            for j in range(N + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            
            j0 = j1
            if p[j0] == 0:
                break
        
        # 调整匹配
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    
    # 过滤有效匹配（排除虚拟行/列的匹配）
    valid_mask = (p[1:N+1] != 0) & (p[1:N+1] <= M) & (torch.arange(1, N+1, device=device) <= K)
    col_ind = torch.arange(1, N+1, device=device)[valid_mask] - 1  # 转为0-based
    row_ind = p[1:N+1][valid_mask] - 1  # 转为0-based
    
    return row_ind, col_ind




# 定义YOLOv11检测损失函数（适配输入结构：无logits）
class YOLOv11DetectionLoss(nn.Module):
    def __init__(self, num_classes=80, weight_class=1.0, 
                 weight_bbox_l1=1.0, weight_giou=1.0,** args):
        """
        初始化 YOLOv11 检测损失函数（适配无logits输入）
        Args:
            num_classes: 类别总数（默认 COCO 80类）
            conf_threshold: 置信度阈值（过滤低置信度预测框）
            iou_threshold: NMS/IoU匹配阈值
            weight_class: 分类损失权重
            weight_bbox_l1: 边界框L1损失权重
            weight_giou: GIoU损失权重
        """
        super().__init__()
        for key, value in args.items():
            setattr(self, key, value)  # 使用setattr动态设置属性
        self.num_classes = num_classes
        self.conf_thres = self.conf_threshold
        self.iou_thres = self.iou_threshold
        self.weight_class = weight_class
        self.weight_bbox_l1 = weight_bbox_l1
        self.weight_giou = weight_giou
        # 判断是否存在penalty_bbox，不存在则创建
        if not hasattr(self, 'penalty_bbox'):  # 判断是否存在属性
            self.penalty_bbox = 1
        if not hasattr(self, 'penalty_giou'):  # 判断是否存在属性
            self.penalty_giou =1
            
        if not hasattr(self, 'penalty_class'):  # 判断是否存在属性
            # 交叉熵损失的最大值，与类别有关
            
            self.penalty_class = 13.82
        if not hasattr(self, 'image_resolution'):  # 判断是否存在属性
            self.image_resolution = 512

        # 基础损失函数（分类损失改用交叉熵的简化形式，适配离散标签）
        self.class_criterion = nn.NLLLoss(reduction="none")  # 负对数似然损失（需输入log概率）
        self.bbox_l1_criterion = nn.L1Loss(reduction="none")  # 边界框L1损失（不自动降维）

    def _hungarian_matching(self, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels):
        """
        简化版匈牙利匹配：移除logits参数，仅用boxes/scores/labels
        Args:
            pred_boxes: 单样本预测框 [M, 4] (xyxy)
            pred_scores: 单样本预测置信度 [M]
            pred_labels: 单样本预测类别 [M] (0~num_classes-1)
            gt_boxes: 单样本真实框 [K, 4] (xyxy)
            gt_labels: 单样本真实类别 [K] (0~num_classes-1)
        Returns:
            pred_idx: 匹配成功的预测框索引 [T]
            gt_idx: 匹配成功的真实框索引 [T]
        """
        M, K = pred_boxes.shape[0], gt_boxes.shape[0]
        if M == 0 or K == 0:
            # 返回空张量（确保设备一致性）
            return torch.tensor([], dtype=torch.long, device=pred_boxes.device), \
                   torch.tensor([], dtype=torch.long, device=pred_boxes.device)

        # 1. 计算IoU矩阵（预测框×真实框）[M, K]
        iou_matrix = self._compute_iou(pred_boxes, gt_boxes)

        # 2. 计算成本矩阵（越低越优先匹配）
        # 成本 = -（置信度×IoU）（优先高置信+高IoU） + 类别不匹配惩罚（类别不同+100）
        class_match = (pred_labels.unsqueeze(1) == gt_labels.unsqueeze(0)).float()  # [M, K]
        confidence_term = pred_scores.unsqueeze(1)  # [M, 1] → 扩展为[M, K]
        cost_matrix = -(confidence_term * iou_matrix) + (1 - class_match) * 100  # 类别不匹配成本骤增

        # 3. 纯PyTorch匈牙利匹配
        pred_idx, gt_idx = hungarian_matching(cost_matrix)

        # 4. 过滤IoU < 阈值的无效匹配
        if len(pred_idx) > 0:  # 避免空索引导致的报错
            valid_mask = iou_matrix[pred_idx, gt_idx] >= self.iou_thres
            pred_idx = pred_idx[valid_mask]
            gt_idx = gt_idx[valid_mask]

        return pred_idx, gt_idx

    def _compute_iou(self, boxes1, boxes2):
        """计算两组框的IoU矩阵 [M, K]（xyxy格式，无改动）"""
        # 交集面积
        x1 = torch.max(boxes1[:, 0].unsqueeze(1), boxes2[:, 0].unsqueeze(0))
        y1 = torch.max(boxes1[:, 1].unsqueeze(1), boxes2[:, 1].unsqueeze(0))
        x2 = torch.min(boxes1[:, 2].unsqueeze(1), boxes2[:, 2].unsqueeze(0))
        y2 = torch.min(boxes1[:, 3].unsqueeze(1), boxes2[:, 3].unsqueeze(0))
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)  # [M, K]

        # 各自面积
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [M]
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [K]

        # IoU计算（加1e-6避免分母为0）
        iou = inter / (area1.unsqueeze(1) + area2.unsqueeze(0) - inter + 1e-6)
        return iou

    # def forward(self, pred_result, gt_result):
    #     """
    #     简化版前向传播：适配输入结构（pred_result仅含boxes/scores/labels）
    #     Args:
    #         pred_result: 模型预测结果（与你的输入结构完全一致）
    #             'boxes': list[tensor] → [B个元素，每个元素 shape [M_b, 4] (xyxy)]
    #             'scores': list[tensor] → [B个元素，每个元素 shape [M_b]]（置信度）
    #             'labels': list[tensor] → [B个元素，每个元素 shape [M_b]]（预测类别，0~num_classes-1）
    #         gt_result: 真实标签（格式与pred_result对齐，无scores）
    #             'boxes': list[tensor] → [B个元素，每个元素 shape [K_b, 4] (xyxy)]
    #             'labels': list[tensor] → [B个元素，每个元素 shape [K_b]]（真实类别，0~num_classes-1）
    #     Returns:
    #         total_loss: 总损失（标量）
    #         loss_dict: 各任务损失明细（dict of 标量）
    #     """
    #     # 确定设备（避免空输入报错）
    #     if len(pred_result['boxes']) == 0 or pred_result['boxes'][0].numel() == 0:
    #         device = torch.device('cpu')
    #     else:
    #         device = pred_result['boxes'][0].device
            
    #     # 初始化损失累计变量
    #     total_class_loss = torch.tensor(0.0, device=device)
    #     total_bbox_l1_loss = torch.tensor(0.0, device=device)
    #     total_giou_loss = torch.tensor(0.0, device=device)
    #     total_valid_boxes = 0  # 统计匹配成功的框对数量（用于平均损失）

    #     # 遍历每个batch样本计算损失
    #     batch_size = len(pred_result['boxes'])
    #     if batch_size == 0:
    #         # 没有检测到框
    #         print("没有检测到框")
    #         total_bbox_l1_loss = torch.tensor(self.penalty_bbox, device=device)
    #         total_giou_loss = torch.tensor(self.penalty_giou, device=device)
    #         total_class_loss=torch.tensor(self.penalty_class, device=device)
    #         total_loss=total_class_loss+total_bbox_l1_loss+total_giou_loss
    #         return total_loss, {'total_loss': total_loss, 'class_loss': total_class_loss,
    #                              'bbox_l1_loss': total_bbox_l1_loss, 'giou_loss': total_giou_loss}
            
    #     for b in range(batch_size):
    #         # 1. 提取单样本数据（确保设备一致）
    #         pred_boxes = pred_result['boxes'][b].to(device)  # [M_b, 4]
    #         pred_scores = pred_result['scores'][b].to(device)  # [M_b]
    #         pred_labels = pred_result['labels'][b].to(device)  # [M_b]
    #         gt_boxes = gt_result['boxes'][b].to(device)  # [K_b, 4]
    #         gt_labels = gt_result['labels'][b].to(device)  # [K_b]

    #         # 2. 过滤低置信度预测框（减少无效计算）
    #         pred_mask = pred_scores >= self.conf_thres
    #         pred_boxes = pred_boxes[pred_mask]
    #         pred_scores = pred_scores[pred_mask]
    #         pred_labels = pred_labels[pred_mask]
    #         M_b, K_b = pred_boxes.shape[0], gt_boxes.shape[0]

    #         # 3. 若无预测框或无真实框，跳过该样本
    #         if M_b == 0 or K_b == 0:
    #             continue

    #         # 4. 匈牙利算法匹配框对
    #         pred_idx, gt_idx = self._hungarian_matching(
    #             pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels
    #         )
    #         T = pred_idx.shape[0]  # 匹配成功的框对数量
    #         if T == 0:
    #             total_bbox_l1_loss += torch.tensor(self.penalty_bbox, device=device)
    #             total_giou_loss += torch.tensor(self.penalty_giou, device=device)
                
    #             total_class_loss+=torch.tensor(self.penalty_class, device=device)
    #             continue
    #         total_valid_boxes += T

    #         # 5. 提取匹配成功的框对和标签
    #         matched_pred_boxes = pred_boxes[pred_idx]  # [T, 4]
    #         matched_pred_scores = pred_scores[pred_idx]  # [T]（用于分类损失的置信度加权）
    #         matched_pred_labels = pred_labels[pred_idx]  # [T]
    #         matched_gt_boxes = gt_boxes[gt_idx]  # [T, 4]
    #         matched_gt_labels = gt_labels[gt_idx]  # [T]

    #         # 6. 计算分类损失（适配离散标签：用置信度作为"伪概率"的对数）
    #         # 逻辑：置信度→归一化到(0,1)→取log→作为NLLLoss的输入（模拟分类概率）
    #         pred_log_probs = torch.log(matched_pred_scores.clamp(min=1e-6, max=1.0))  # 避免log(0)
    #         # 构建"类别-概率"映射：仅匹配的类别对应log概率，其他类别为-∞（确保NLLLoss正确计算）
    #         class_log_probs = torch.full((T, self.num_classes), -float('inf'), device=device)
    #         class_log_probs[torch.arange(T), matched_pred_labels] = pred_log_probs
    #         # 计算分类损失（NLLLoss：输入[batch, class_num]，目标[batch]）
    #         class_loss = self.class_criterion(class_log_probs, matched_gt_labels).sum()
    #         total_class_loss += class_loss

    #         # 7. 计算边界框L1损失（坐标误差：x1,y1,x2,y2的绝对误差和）
    #         bbox_l1_loss = self.bbox_l1_criterion(matched_pred_boxes, matched_gt_boxes).sum(dim=1).sum()
    #         total_bbox_l1_loss += (bbox_l1_loss/(  4*self.image_resolution))

    #         # 8. 计算GIoU损失（重叠度误差：1 - GIoU，确保损失非负）
    #         giou = generalized_box_iou(matched_pred_boxes, matched_gt_boxes)  # [T]
    #         giou_loss = (1 - giou).sum()  # GIoU损失求和
    #         total_giou_loss += giou_loss

    #     # 9. 计算平均损失（避免无匹配框时除以0）
    #     # if total_valid_boxes == 0:
    #     #     class_loss_avg = torch.tensor(0.0, device=device)
    #     #     bbox_l1_loss_avg = torch.tensor(0.0, device=device)
    #     #     giou_loss_avg = torch.tensor(0.0, device=device)
    #     # else:
    #     if total_valid_boxes != 0:
            
    #         class_loss_avg = total_class_loss / total_valid_boxes
    #         bbox_l1_loss_avg = total_bbox_l1_loss / total_valid_boxes
    #         giou_loss_avg = total_giou_loss / total_valid_boxes

    #     # 10. 总损失 = 各任务损失 × 权重之和
    #     total_loss = (
    #         class_loss_avg * self.weight_class +
    #         bbox_l1_loss_avg * self.weight_bbox_l1 +
    #         giou_loss_avg * self.weight_giou
    #     )

    #     # 11. 输出损失明细（便于调试和监控）
    #     loss_dict = {
    #         'class_loss': class_loss_avg,
    #         'bbox_l1_loss': bbox_l1_loss_avg,
    #         'giou_loss': giou_loss_avg,
    #         'total_loss': total_loss
    #     }

    #     return total_loss, loss_dict

    def forward(self, pred_result, gt_result):
        """
        遍历每个真实框（gt）单独计算损失，确保每个gt都参与损失计算
        Args:
            pred_result: 模型预测结果
                'boxes': list[tensor] → [B个元素，每个元素 shape [M_b, 4] (xyxy)]
                'scores': list[tensor] → [B个元素，每个元素 shape [M_b]]
                'labels': list[tensor] → [B个元素，每个元素 shape [M_b]]
            gt_result: 真实标签
                'boxes': list[tensor] → [B个元素，每个元素 shape [K_b, 4] (xyxy)]
                'labels': list[tensor] → [B个元素，每个元素 shape [K_b]]
        Returns:
            total_loss: 总损失（标量）
            loss_dict: 各任务损失明细（dict of 标量）
        """   
        # 确定设备
        if len(pred_result['boxes']) == 0 or (len(pred_result['boxes']) > 0 and pred_result['boxes'][0].numel() == 0):
            device = torch.device('cpu')
        else:
            device = pred_result['boxes'][0].device
            
        # 初始化损失累计变量（按每个gt单独计算）
        total_class_loss = torch.tensor(0.0, device=device)
        total_bbox_l1_loss = torch.tensor(0.0, device=device)
        total_giou_loss = torch.tensor(0.0, device=device)
        total_gt_count = 0  # 统计总真实框数量（用于平均损失）

        batch_size = len(pred_result['boxes'])
        if batch_size == 0:
            # 空batch处理
            total_bbox_l1_loss = torch.tensor(self.penalty_bbox, device=device)
            total_giou_loss = torch.tensor(self.penalty_giou, device=device)
            total_class_loss = torch.tensor(self.penalty_class, device=device)
            total_loss = total_class_loss + total_bbox_l1_loss + total_giou_loss
            return total_loss, {
                'total_loss': total_loss,
                'class_loss': total_class_loss,
                'bbox_l1_loss': total_bbox_l1_loss,
                'giou_loss': total_giou_loss
            }
        
        for b in range(batch_size):
            # 提取单样本数据
            pred_boxes = pred_result['boxes'][b].to(device)  # [M_b, 4]
            pred_scores = pred_result['scores'][b].to(device)  # [M_b]
            pred_labels = pred_result['labels'][b].to(device)  # [M_b]
            pred_scores_vector=pred_result['scores_vector'][b].to(device)
            gt_boxes = gt_result['boxes'][b].to(device)  # [K_b, 4]
            gt_labels = gt_result['labels'][b].to(device)  # [K_b]
            gt_scores_vector=gt_result['scores_vector'][b].to(device)
            K_b = gt_boxes.shape[0]  # 该样本的真实框数量
            if K_b == 0:
                continue  # 无真实框则跳过
            total_gt_count += K_b  # 累计总真实框数量

            # 过滤低置信度预测框
            pred_mask = pred_scores >= self.conf_thres
            pred_boxes = pred_boxes[pred_mask]
            pred_scores = pred_scores[pred_mask]
            pred_labels = pred_labels[pred_mask]
            pred_scores_vector=pred_scores_vector[pred_mask]
            M_b = pred_boxes.shape[0]  # 过滤后的预测框数量

            # 遍历该样本的每个真实框（核心修改：循环所有gt）
            for gt_idx in range(K_b):
                # 提取当前真实框（单独处理）
                current_gt_box = gt_boxes[gt_idx:gt_idx+1]  # [1, 4]（保持维度）
                current_gt_label = gt_labels[gt_idx:gt_idx+1]  # [1]
                current_gt_scores_vector=gt_scores_vector[gt_idx:gt_idx+1]
                if M_b == 0:
                    # 无预测框：对当前gt施加惩罚损失
                    total_class_loss += torch.tensor(self.penalty_class, device=device)
                    total_bbox_l1_loss += torch.tensor(self.penalty_bbox, device=device)
                    total_giou_loss += torch.tensor(self.penalty_giou, device=device)
                    continue

                # 为当前真实框匹配最优预测框（简化匹配：计算与当前gt的成本）
                # 1. 计算当前gt与所有预测框的IoU [M_b, 1]
                iou_matrix = self._compute_iou(pred_boxes, current_gt_box)  # [M_b, 1]
                
                # 2. 计算成本矩阵 [M_b, 1]
                class_match = (pred_labels.unsqueeze(1) == current_gt_label).float()  # [M_b, 1]
                confidence_term = pred_scores.unsqueeze(1)  # [M_b, 1]
                cost_matrix = -(confidence_term * iou_matrix) + (1 - class_match) * 100  # 成本越低越好
                
                # 3. 选择成本最低的预测框作为匹配
                min_cost, best_pred_idx = torch.min(cost_matrix, dim=0)  # 找到最优预测框索引
                best_pred_idx = best_pred_idx.item()  # 转为标量索引

                # 4. 检查匹配有效性（IoU达标+类别匹配）
                valid = (iou_matrix[best_pred_idx] >= self.iou_thres) 
                if not valid:
                    # 无效匹配：施加惩罚
                    total_class_loss += torch.tensor(self.penalty_class, device=device)
                    total_bbox_l1_loss += torch.tensor(self.penalty_bbox, device=device)
                    total_giou_loss += torch.tensor(self.penalty_giou, device=device)
                    continue

                # 5. 提取匹配的预测框
                matched_pred_box = pred_boxes[best_pred_idx:best_pred_idx+1]  # [1, 4]
                matched_pred_score = pred_scores[best_pred_idx:best_pred_idx+1]  # [1]
                matched_pred_label = pred_labels[best_pred_idx:best_pred_idx+1]  # [1]
                matched_pred_scores_vector=pred_scores_vector[best_pred_idx:best_pred_idx+1]
                
                # 6. 计算分类损失（当前gt的分类损失）
                # pred_log_prob = torch.log(matched_pred_score.clamp(min=1e-6, max=1.0))  # [1]
                # class_log_probs = torch.full((1, self.num_classes),  -1e6, device=device)
                # class_log_probs[0, matched_pred_label] = pred_log_prob
                # class_loss = self.class_criterion(class_log_probs, current_gt_label).sum()
                # log计算
                matched_pred_scores_vector_log=torch.log(matched_pred_scores_vector.clamp(min=1e-6, max=1.0))
                class_loss = self.class_criterion(matched_pred_scores_vector_log, current_gt_label).sum()
                total_class_loss += class_loss

                # 7. 计算边界框L1损失
                bbox_l1_loss = self.bbox_l1_criterion(matched_pred_box, current_gt_box).sum(dim=1).sum()
                total_bbox_l1_loss += (bbox_l1_loss / (4 * self.image_resolution))  # 归一化

                # 8. 计算GIoU损失
                giou = generalized_box_iou(matched_pred_box, current_gt_box)  # [1]
                giou_loss = (1 - giou).sum()
                total_giou_loss += giou_loss

        # 计算平均损失（除以总真实框数量）

        if total_gt_count > 0:
            class_loss_avg = total_class_loss / total_gt_count
            bbox_l1_loss_avg = total_bbox_l1_loss / total_gt_count
            giou_loss_avg = total_giou_loss / total_gt_count

        # 总损失 = 加权和
        total_loss = (
            class_loss_avg * self.weight_class +
            bbox_l1_loss_avg * self.weight_bbox_l1 +
            giou_loss_avg * self.weight_giou
        )

        loss_dict = {
            'class_loss': class_loss_avg,
            'bbox_l1_loss': bbox_l1_loss_avg,
            'giou_loss': giou_loss_avg,
            'total_loss': total_loss
        }

        return total_loss, loss_dict
    


class TVLoss(nn.Module):
    """
    总变分损失（Total Variation Loss）
    用于衡量图像的平滑度，惩罚相邻像素间的剧烈变化
    """
    def __init__(self, tv_loss_weight=1.0):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        """
        计算TV损失
        Args:
            x: 输入张量，形状为 [batch_size, channels, height, width]
        Returns:
            tv_loss: 总变分损失值
        """
        # 计算水平方向的差异（相邻列像素差的L1范数）
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        
        # 水平方向：x[:, :, :, 1:] - x[:, :, :, :-1]
        count_h = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :-1]), 2).sum()
        
        # 垂直方向：x[:, :, 1:, :] - x[:, :, :-1, :]
        count_w = self.tensor_size(x[:, :, 1:, :])
        w_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :-1, :]), 2).sum()
        
        # 总变分损失 = 水平损失 + 垂直损失
        tv_loss = self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
        return tv_loss

    @staticmethod
    def tensor_size(t):
        """计算张量中元素的数量（除batch维度外）"""
        return t.size()[1] * t.size()[2] * t.size()[3]
    




def tensor2picture(tensor, save_path, data_range="auto", use_opencv=False):
    """
    将 Tensor 保存为图像文件
    
    参数:
        tensor: PyTorch Tensor 或 TensorFlow Tensor
            形状要求: 
                - PyTorch: (B, C, H, W) 或 (C, H, W)（B为批量，C为通道）
                - TensorFlow: (B, H, W, C) 或 (H, W, C)
        save_path: str
            图像保存路径（如 "output.png"）
        data_range: str 或 tuple, 可选
            输入张量的数据范围，默认为 "auto"（自动检测）：
            - "auto": 自动将张量归一化到 [0, 255]
            - (min_val, max_val): 手动指定范围，将其映射到 [0, 255]
        use_opencv: bool, 可选
            是否用 OpenCV 保存（默认用 PIL），OpenCV 会自动转换 RGB→BGR
    """
    # --------------------------
    # 1. 处理 Tensor 维度（移除批量维度）
    # --------------------------
    if "torch" in str(type(tensor)).lower():  # PyTorch Tensor
        # 移至 CPU 并转为 numpy
        tensor = tensor.cpu().detach() if tensor.requires_grad else tensor.cpu()
        img_np = tensor.numpy()
        
        # 移除批量维度（若有）
        if img_np.ndim == 4:  # (B, C, H, W) → (C, H, W)
            img_np = img_np.squeeze(0)
        
        # 调整通道顺序：(C, H, W) → (H, W, C)
        if img_np.shape[0] in [1, 3]:  # 单通道/三通道
            img_np = np.transpose(img_np, (1, 2, 0))
    
    elif "tensorflow" in str(type(tensor)).lower():  # TensorFlow Tensor
        # 转为 numpy（TF 默认在 CPU，无需手动转移）
        img_np = tensor.numpy()
        
        # 移除批量维度（若有）
        if img_np.ndim == 4:  # (B, H, W, C) → (H, W, C)
            img_np = img_np.squeeze(0)
    
    else:
        raise TypeError("不支持的 tensor 类型，请使用 PyTorch 或 TensorFlow 张量")
    
    # --------------------------
    # 2. 处理单通道图像（灰度图）
    # --------------------------
    if img_np.shape[-1] == 1:
        img_np = img_np.squeeze(-1)  # 移除通道维度，变为 (H, W)
    
    # --------------------------
    # 3. 数据范围映射到 [0, 255]
    # --------------------------
    if data_range == "auto":
        min_val = img_np.min()
        max_val = img_np.max()
    else:
        min_val, max_val = data_range
    
    # 防止除零（若所有值相同）
    if max_val == min_val:
        img_np = np.zeros_like(img_np, dtype=np.uint8)
    else:
        # 归一化到 [0, 1] 再映射到 [0, 255]
        img_np = ((img_np - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
    # --------------------------
    # 4. 保存图像
    # --------------------------
    if use_opencv:
        # OpenCV 保存 BGR 格式，若原是 RGB 需转换
        if img_np.ndim == 3 and img_np.shape[-1] == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img_np)
    else:
        # PIL 直接保存 RGB 格式
        img = Image.fromarray(img_np)
        img.save(save_path)


def cv2_to_tensor(img: np.ndarray, normalize: bool = True) -> torch.Tensor:
    """
    将OpenCV读取的图像（H×W×C，BGR格式，uint8）转换为C×H×W格式的Tensor
    
    参数:
        img: OpenCV读取的图像数组，形状为(H, W, C)，通道顺序为BGR，数据类型为uint8
        normalize: 是否将像素值归一化到[0.0, 1.0]（默认True）
    
    返回:
        tensor: 转换后的Tensor，形状为(C, H, W)，数据类型为float32
                若输入为彩色图，通道顺序转为RGB；若为灰度图，保持单通道
    """
    # 检查输入是否为合法的图像数组
    if not isinstance(img, np.ndarray) or img.ndim not in (2, 3):
        raise ValueError("输入必须是OpenCV读取的2D（灰度图）或3D（彩色图）数组")
    
    # 处理彩色图（3通道）：BGR转RGB
    if img.ndim == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # 灰度图（2通道）：保持不变，后续扩展为单通道
        img_rgb = img
    
    # 转换为float32类型（避免uint8计算溢出）
    img_float = img_rgb.astype(np.float32)
    
    # 归一化到[0.0, 1.0]（如果需要）
    if normalize:
        img_float /= 255.0
    
    # 维度重排：H×W×C → C×H×W
    # 灰度图会从(H, W)变为(1, H, W)
    tensor = torch.from_numpy(img_float).permute(2, 0, 1) if img.ndim == 3 else torch.from_numpy(img_float).unsqueeze(0)
    
    return tensor


def scale_tensor_to_resolution(tensor, new_height, new_width, mode='bilinear'):
    """
    将图像Tensor（C×H×W）缩放到指定分辨率（new_height × new_width）
    
    参数:
        tensor: 输入图像Tensor，形状为 (C, H, W)，数据类型为float32/float64
        new_height: 目标高度（整数）
        new_width: 目标宽度（整数）
        mode: 插值方法，可选 'bilinear'（默认，双线性）、'nearest'（最近邻）、'bicubic'（双三次）
    
    返回:
        scaled_tensor: 缩放后的Tensor，形状为 (C, new_height, new_width)
    """
    # 检查输入合法性
    if tensor.dim() != 3:
        raise ValueError(f"输入Tensor必须是3维 (C, H, W)，但得到 {tensor.dim()} 维")
    if not isinstance(new_height, int) or not isinstance(new_width, int):
        raise ValueError(f"目标分辨率（new_height, new_width）必须是整数，但得到 ({new_height}, {new_width})")
    if new_height <= 0 or new_width <= 0:
        raise ValueError(f"目标分辨率必须为正数， but得到 ({new_height}, {new_width})")
    
    # 增加batch维度（B=1），因为F.interpolate要求输入为4维 (B, C, H, W)
    tensor_with_batch = tensor.unsqueeze(0)  # 形状变为 (1, C, H, W)
    
    # 执行缩放：按目标分辨率（new_height, new_width）插值
    scaled = F.interpolate(
        input=tensor_with_batch,
        size=(new_height, new_width),  # 明确指定目标尺寸
        mode=mode,
        align_corners=False  # 默认为False，避免边缘扭曲
    )
    
    # 去除batch维度，返回 (C, new_height, new_width)
    return scaled.squeeze(0)




def binarize_image_tensor(img, threshold=0.5):
    """
    对输入的张量图像（0-1范围）进行二值化，生成0.0/1.0的浮点型掩码（与输入形状一致，含三通道）。
    逻辑：对每个像素取所有通道的最小值，小于阈值则为1.0，否则为0.0，最终扩展为与输入相同的通道数。
    
    参数：
        img: 输入张量，形状支持 (B, C, H, W)、(C, H, W) 或 (H, W)，值范围 [0,1]
        threshold: 阈值（默认0.5）
    
    返回：
        mask: 二值化掩码，与输入形状相同（含通道数），值为0.0或1.0（浮点型）
    """
    # 确保输入是PyTorch张量
    if not isinstance(img, torch.Tensor):
        raise TypeError("输入必须是PyTorch张量")
    
    # 确定通道维度和输入通道数
    if img.dim() == 4:  # (B, C, H, W)
        channel_dim = 1
        num_channels = img.size(channel_dim)  # 获取通道数 C
    elif img.dim() == 3:  # (C, H, W)
        channel_dim = 0
        num_channels = img.size(channel_dim)  # 获取通道数 C
    elif img.dim() == 2:  # (H, W) 单通道输入，默认输出3通道
        channel_dim = -1
        num_channels = 3  # 手动指定为3通道
    else:
        raise ValueError("输入张量维度必须为2、3或4")
    
    # 处理多通道：取每个像素的通道最小值
    if channel_dim != -1:
        img_min = torch.min(img, dim=channel_dim, keepdim=True)[0]  # 单通道 (B,1,H,W) 或 (1,H,W)
    else:  # 单通道输入，直接增加通道维度
        img_min = img.unsqueeze(0)  # (1, H, W)
    
    # 二值化：像素最小值 < 阈值 → 1.0，否则 → 0.0
    mask_single = (img_min < threshold).float()  # 单通道掩码
    
    # 将单通道掩码复制为与输入相同的通道数（通常为3通道）
    # 用repeat在通道维度复制，其他维度复制1次（保持不变）
    if img.dim() == 4:
        # 输入 (B,C,H,W) → 输出 (B,C,H,W)：在通道维度（dim=1）复制C次
        mask = mask_single.repeat(1, num_channels, 1, 1)
    elif img.dim() == 3:
        # 输入 (C,H,W) → 输出 (C,H,W)：在通道维度（dim=0）复制C次
        mask = mask_single.repeat(num_channels, 1, 1)
    else:  # 输入 (H,W) → 输出 (3,H,W)
        mask = mask_single.repeat(num_channels, 1, 1)  # 复制为3通道
    
    return mask



def tensor_to_pil(image_tensor):
    """Tensor转PIL Image（处理函数内部转换用）"""
    if isinstance(image_tensor, torch.Tensor):
        # 处理批量数据或单张图像
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]
        # 反归一化（0-1 → 0-255）
        img_np = image_tensor.cpu().detach().numpy()
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        # 通道顺序转换 [C, H, W] → [H, W, C]
        if img_np.shape[0] in [1, 3]:
            img_np = np.transpose(img_np, (1, 2, 0))
        # 处理灰度图
        if img_np.shape[-1] == 1:
            img_np = np.squeeze(img_np, axis=-1)
        return Image.fromarray(img_np)
    raise TypeError(f"不支持的Tensor类型：{type(image_tensor)}")


def generate_inpaint_prompt(original_caption, target_object="dog", background_desc=None):
    """
    根据图像描述生成消除目标的提示词
    
    Args:
        original_caption: 图像的原始文本描述
        target_object: 要消除的目标（如"dog"/"car"/"person"）
        background_desc: 自定义背景描述（可选）
    
    Returns:
        positive_prompt: 正面提示词（补全背景）
        negative_prompt: 负面提示词（禁止生成目标）
    """
    # 从原始描述中提取背景信息
    if not background_desc:
        # 移除目标关键词，保留背景描述
        background_desc = re.sub(rf"\b{target_object}\b.*?\b", "", original_caption, flags=re.IGNORECASE)
        background_desc = background_desc.replace("  ", " ").strip()
        
        # 兜底背景描述
        if not background_desc:
            background_desc = "natural outdoor scenery, grass, sky, realistic background"
    
    # 生成正面提示词（补全背景）
    positive_prompt = f"{background_desc}, seamless background, realistic details, high resolution, no {target_object}"
    
    # 生成负面提示词（禁止出现目标）
    negative_prompt = f"{target_object}, animal, person, object, blurry, low quality, artifacts, text, watermark"
    
    return positive_prompt, negative_prompt



class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, img_extensions=['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.PNG','JPEG']):
        """
        自定义数据集：遍历指定目录下所有图片（递归查找）
        :param root_dir: 图片根目录
        :param transform: 预处理变换
        :param img_extensions: 支持的图片格式
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_extensions = img_extensions
        # 递归获取所有图片路径
        self.img_paths = self._get_all_img_paths()

    def _get_all_img_paths(self):
        """递归遍历目录，获取所有图片路径"""
        img_paths = []
        # 递归遍历所有子目录
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                # 过滤图片格式
                if any(file.endswith(ext) for ext in self.img_extensions):
                    img_path = os.path.join(root, file)
                    img_paths.append(img_path)
        if not img_paths:
            raise ValueError(f"目录 {self.root_dir} 下未找到任何图片！")
        return img_paths

    def __len__(self):
        """返回图片总数"""
        return len(self.img_paths)

    def __getitem__(self, idx):
        """加载单张图片（返回：预处理后的张量 + 图片路径）"""
        img_path = self.img_paths[idx]
        try:
            # 读取图片并转为RGB（避免灰度图）
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"加载图片失败 {img_path}：{e}")
        
        # 应用预处理
        if self.transform:
            img = self.transform(img)
        
        # 返回：图片张量 + 图片路径（用于后续记录）
        return img, img_path
    
class ResizeMaxEdge:
    def __init__(self, max_edge_size):
        self.max_edge_size = max_edge_size

    def __call__(self, img):
        """
        将图片最大边缩放到max_edge_size，短边按比例缩放（保持宽高比）
        :param img: PIL Image对象
        :return: 缩放后的PIL Image
        """
        # 获取原图尺寸
        w, h = img.size
        # 计算缩放比例（最大边=max_edge_size）
        scale = self.max_edge_size / max(w, h)
        # 计算新尺寸（四舍五入为整数）
        new_w = int(math.ceil(w * scale))
        new_h = int(math.ceil(h * scale))
        # 缩放（保持宽高比）
        img_resized = torchvision.transforms.functional.resize(img, (new_h, new_w), antialias=True)  # resize参数是(H, W)
        return img_resized

# # --------------------------
# # 自定义变换：缩放后填充到固定尺寸（可选，替代中心裁剪）
# # --------------------------
# class PadToFixedSize:
#     def __init__(self, target_size, fill=0):
#         self.target_size = target_size  # (H, W)
#         self.fill = fill  # 填充值（默认黑色）

#     def __call__(self, img):
#         """缩放后填充到固定尺寸，中心对齐"""
#         w, h = img.size
#         target_h, target_w = self.target_size
#         # 计算填充量
#         pad_left = (target_w - w) // 2
#         pad_right = target_w - w - pad_left
#         pad_top = (target_h - h) // 2
#         pad_bottom = target_h - h - pad_top
#         # 填充
#         img_padded = torchvision.transforms.functional.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)
#         return img_padded
class PadToFixedSize:
    def __init__(self, target_size, fill=0):
        """
        修复：自动兼容单个整数/元组输入，避免解包错误
        :param target_size: 目标尺寸（int → (size, size)；tuple → (h, w)）
        :param fill: 填充值（默认黑色）
        """
        # 核心修复：统一转为元组
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)  # 单个整数→正方形
        elif isinstance(target_size, (tuple, list)) and len(target_size) == 2:
            self.target_size = (int(target_size[0]), int(target_size[1]))  # 元组/列表→(h,w)
        else:
            raise ValueError(
                f"target_size必须是整数或长度为2的元组/列表！当前输入：{target_size}"
            )
        self.fill = fill  # 填充值（默认黑色）

    def __call__(self, img):
        """缩放后填充到固定尺寸，中心对齐"""
        w, h = img.size  # PIL Image的size是(w, h)
        target_h, target_w = self.target_size  # 现在一定是元组，可安全解包
        
        # 计算填充量（保证中心对齐）
        pad_left = (target_w - w) // 2
        pad_right = target_w - w - pad_left
        pad_top = (target_h - h) // 2
        pad_bottom = target_h - h - pad_top
        
        # 填充（注意：pad的参数顺序是 (left, top, right, bottom)）
        img_padded = torchvision.transforms.functional.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)
        return img_padded