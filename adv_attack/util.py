import gc
import math
import os
import re
from collections.abc import Mapping, Iterable
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
from typing import Optional, Any,Tuple, Dict, List,Union
from contextlib import suppress
import pytorch_lightning as pl
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
        """计算两组框的IoU矩阵 [M, K]（xyxy格式，修复float16精度问题）"""
        # 保存原始数据类型和设备
        orig_dtype = boxes1.dtype
        orig_device = boxes1.device
        
        # 转为float32计算，避免半精度精度丢失
        boxes1 = boxes1.to(dtype=torch.float32)
        boxes2 = boxes2.to(dtype=torch.float32)

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

        # 转回原始数据类型和设备
        iou = iou.to(dtype=orig_dtype, device=orig_device)
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
def move_to_gpu(obj, device=None):
    """
    将嵌套结构（字典、列表、张量等）中的所有 PyTorch Tensor 移动到 GPU。
    
    参数:
        obj: 任意输入对象（字典、列表、张量、标量、嵌套结构等）
        device: 指定 GPU 设备（如 torch.device('cuda:0')），默认自动选择可用 GPU
    
    返回:
        与输入结构完全一致的对象，所有 Tensor 已移至 GPU（CPU 若无可使用 GPU）
    """
    # 自动选择 GPU 设备（优先用指定设备，无则用第一个可用 GPU，无 GPU 则用 CPU）
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, int):
        device = torch.device(f'cuda:{device}')
    
    # 递归终止条件 1：如果是 Tensor，直接移到 GPU
    if isinstance(obj, torch.Tensor):
        # 仅当 Tensor 不在目标设备时才移动（避免重复操作）
        if obj.device != device:
            return obj.to(device, non_blocking=True)  # non_blocking 加速 GPU 传输
        return obj
    
    # 递归终止条件 2：非可迭代对象（标量、字符串等），直接返回
    if not isinstance(obj, (Mapping, Iterable)) or isinstance(obj, (str, bytes)):
        return obj
    
    # 处理字典（包括普通 dict、OrderedDict 等 Mapping 类型）
    if isinstance(obj, Mapping):
        return type(obj)({
            k: move_to_gpu(v, device=device) 
            for k, v in obj.items()
        })
    
    # 处理列表/元组/集合等可迭代对象
    if isinstance(obj, tuple):
        # 区分命名元组（NamedTuple）和普通元组
        if hasattr(obj, '_fields'):  # 命名元组
            return type(obj)(*(move_to_gpu(x, device=device) for x in obj))
        else:
            return tuple(move_to_gpu(x, device=device) for x in obj)
    elif isinstance(obj, list):
        return [move_to_gpu(x, device=device) for x in obj]
    elif isinstance(obj, set):
        return {move_to_gpu(x, device=device) for x in obj}
    elif isinstance(obj, Iterable):
        # 处理其他可迭代对象（如生成器，转为列表）
        return [move_to_gpu(x, device=device) for x in obj]
    
    # 其他未匹配类型，直接返回
    return obj

def move_to_gpu_and_cast_dtype(obj, device=None, dtype=None):
    """
    将嵌套结构（字典、列表、张量等）中的所有 PyTorch Tensor 移动到指定设备，并转换目标数据类型（仅对浮点型张量生效）。
    
    参数:
        obj: 任意输入对象（字典、列表、张量、标量、嵌套结构等）
        device: 指定设备（如 torch.device('cuda:0')/int/None），默认自动选择可用 GPU
        dtype: 目标数据类型（如 torch.float16/torch.float32/None），None 则不转换类型（仅对浮点型张量生效）
    
    返回:
        与输入结构完全一致的对象，所有 Tensor 已移至指定设备；浮点型 Tensor 转换为目标类型，整型 Tensor 保留原类型
    """
    # 1. 设备处理：自动选择设备
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, int):
        device = torch.device(f'cuda:{device}')
    elif not isinstance(device, torch.device):
        raise TypeError(f"device 必须是 torch.device/int/None，当前类型：{type(device)}")

    # 2. 类型校验：确保 dtype 是合法的 PyTorch 数据类型或 None
    if dtype is not None and not isinstance(dtype, torch.dtype):
        raise TypeError(f"dtype 必须是 torch.dtype/None，当前类型：{type(dtype)}")

    def _is_float_tensor(tensor):
        """判断张量是否为浮点型"""
        return tensor.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16)

    def _process_tensor(tensor):
        """内部函数：处理单个张量的设备和类型转换（仅浮点型张量转换类型）"""
        # 先处理类型转换：仅对浮点型张量生效，整型张量跳过
        if dtype is not None and _is_float_tensor(tensor) and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype, non_blocking=True)
        
        # 再处理设备迁移：所有张量都执行设备迁移
        if tensor.device != device:
            tensor = tensor.to(device=device, non_blocking=True)
        
        return tensor

    # 3. 递归处理嵌套结构
    def _recursive_process(obj):
        # 终止条件1：Tensor 处理
        if isinstance(obj, torch.Tensor):
            return _process_tensor(obj)
        
        # 终止条件2：非可迭代对象（标量、字符串等），直接返回
        if not isinstance(obj, (Mapping, Iterable)) or isinstance(obj, (str, bytes)):
            return obj
        
        # 处理字典（包括普通 dict、OrderedDict 等 Mapping 类型）
        if isinstance(obj, Mapping):
            return type(obj)({k: _recursive_process(v) for k, v in obj.items()})
        
        # 处理元组（区分命名元组和普通元组）
        if isinstance(obj, tuple):
            if hasattr(obj, '_fields'):  # 命名元组
                return type(obj)(*(_recursive_process(x) for x in obj))
            else:
                return tuple(_recursive_process(x) for x in obj)
        
        # 处理列表
        elif isinstance(obj, list):
            return [_recursive_process(x) for x in obj]
        
        # 处理集合
        elif isinstance(obj, set):
            return {_recursive_process(x) for x in obj}
        
        # 处理其他可迭代对象（如生成器，转为列表）
        elif isinstance(obj, Iterable):
            return [_recursive_process(x) for x in obj]
        
        # 其他未匹配类型，直接返回
        return obj

    return _recursive_process(obj)

def release_torch_object_memory(
    obj_name: str, 
    namespace: Optional[dict] = None,
    verbose: bool = True
) -> None:
    """
    释放指定 PyTorch 对象（如模型、张量、LPIPS损失函数）占用的 CPU/GPU 内存
    
    参数:
        obj_name: 要释放的对象名（字符串格式，如 'perceptual_loss'）
        namespace: 对象所在的命名空间（默认使用局部变量空间 locals()，
                  若对象在全局空间则传 globals()）
        verbose: 是否打印内存释放前后的状态（便于调试）
    """
    # 默认使用局部变量空间，若未指定则取当前局部命名空间
    if namespace is None:
        namespace = locals()
    
    # 记录释放前的内存状态
    if verbose and torch.cuda.is_available():
        pre_allocated = torch.cuda.memory_allocated() / 1024**2
        pre_cached = torch.cuda.memory_reserved() / 1024**2
        print(f"【释放前】GPU已分配显存: {pre_allocated:.2f} MB | 缓存显存: {pre_cached:.2f} MB")

    # 核心释放逻辑
    if obj_name in namespace:
        try:
            # 1. 将对象移回CPU（避免GPU显存残留）
            obj = namespace[obj_name]
            if hasattr(obj, 'to'):
                obj = obj.cpu()
            # 2. 解除对象引用
            del namespace[obj_name]
            if verbose:
                print(f"✅ 成功解除 {obj_name} 的引用")
        except Exception as e:
            if verbose:
                print(f"⚠️ 释放 {obj_name} 时出现异常: {str(e)}")
    else:
        if verbose:
            print(f"ℹ️ 命名空间中未找到 {obj_name}，无需释放")

    # 3. 清空GPU缓存（释放未使用的显存）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    # 4. 强制Python垃圾回收（释放CPU内存）
    gc.collect()
    # 强制清理循环引用（可选，进一步确保释放）
    gc.collect()

    # 打印释放后的内存状态
    if verbose and torch.cuda.is_available():
        post_allocated = torch.cuda.memory_allocated() / 1024**2
        post_cached = torch.cuda.memory_reserved() / 1024**2
        print(f"【释放后】GPU已分配显存: {post_allocated:.2f} MB | 缓存显存: {post_cached:.2f} MB")
        print(f"🔍 显存释放量: 已分配 {pre_allocated - post_allocated:.2f} MB | 缓存 {pre_cached - post_cached:.2f} MB")



def normalize_to_01(tensor):
    """
    将张量缩放到 [0, 1] 范围（逐批次独立归一化，避免跨批次干扰）
    """

    # 按批次维度计算每个样本的最小值和最大值
    min_vals = tensor.amin(dim=[1,2,3], keepdim=True)
    max_vals = tensor.amax(dim=[1,2,3], keepdim=True)
    # 防止除零（若张量全为同一值，直接返回0）
    range_vals = max_vals - min_vals
    range_vals = torch.where(range_vals == 0, torch.ones_like(range_vals), range_vals)
    # 归一化到 [0, 1]
    normalized = (tensor - min_vals) / range_vals
    return normalized



def select_mask_by_criteria(
    masks_logic_mutil_all,
    masks_tensor_all,
    scores_all,
    exp_path,
    mask_select_statues: int = 1,
    mask_save_name: str = 'mask.jpg'
) :
    """
    根据指定规则（面积最大/置信度最高）从多个掩码中选择最优掩码，并保存掩码图像
    """
    mask_logic_list=[]
    mask_tensor_list=[]
    for i, masks_logic_mutil_b_in in enumerate(masks_logic_mutil_all):

        masks_tensor_b_in = masks_tensor_all[i]
        scores_b_in=scores_all[i]
        # 仅当掩码数量大于1时才需要选择，否则直接取第一个
        if masks_logic_mutil_b_in.shape[0] == 1:
            index = 0
            print(f"仅存在1个掩码，直接选择 index:{index}, score:{scores_b_in[index] if len(scores_b_in)>0 else 'N/A'}")
            masks_logic_b_out = masks_logic_mutil_b_in[index]
            masks_tensor_b_out = masks_tensor_b_in[index]
        else:
            if mask_select_statues == 1:
                # 规则1：选择面积最大的掩码
                mask_areas_b_in = masks_logic_mutil_b_in.sum(axis=(1, 2))  # 计算每个掩码的面积
                index = np.argmax(mask_areas_b_in)
                print(f"[面积优先] index:{index}, 面积:{mask_areas_b_in[index]}, score:{scores_b_in[index]}")
                masks_logic_b_out = masks_logic_mutil_b_in[index]
                masks_tensor_b_out = masks_tensor_b_in[index]
            else:
                # 规则2：选择置信度最高的掩码（鲁棒性处理）
                if len(scores_b_in) == 0:
                    raise ValueError("scores 数组为空，无法选择最大置信度的掩码！")
                if np.isnan(scores_b_in).all():
                    raise ValueError("scores 全为 NaN，无法选择最大置信度的掩码！")
                
                # 找到置信度最大的索引（自动跳过NaN）
                index = np.nanargmax(scores_b_in)
                print(f"[置信度优先] index:{index}, score:{scores_b_in[index]}")
                masks_logic_b_out = masks_logic_mutil_b_in[index]
                masks_tensor_b_out = masks_tensor_b_in[index]
        
        # 保存选中的掩码图像
        try:
            exp_path_b_out = exp_path[i]
            # 确保保存目录存在
            os.makedirs(exp_path_b_out, exist_ok=True)
            mask_path = os.path.join(exp_path_b_out, mask_save_name)
            tensor2picture(masks_tensor_b_out, mask_path)  # 假设tensor2picture是已定义的函数
            print(f"选中的掩码已保存至: {mask_path}")
        except Exception as e:
            print(f"警告：掩码图像保存失败 - {str(e)}")
        
        mask_logic_list.append(masks_logic_b_out)
        mask_tensor_list.append(masks_tensor_b_out)
    mask_tensor_tensor=torch.stack(mask_tensor_list)
    mask_logic_np=np.stack(mask_logic_list)
    return mask_logic_np, mask_tensor_tensor


def get_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """
    对B*H*W的mask，提取每个batch中面积最大的连通域，返回同尺寸的mask
    参数：
        mask: numpy数组，shape=(B, H, W)，支持布尔型/0-1数值型
    返回：
        largest_cc_mask: numpy数组，shape=(B, H, W)，仅保留每个batch的最大连通域（布尔型）
    """
    # 1. 输入校验与预处理
    assert mask.ndim == 3, f"mask必须是3维(B,H,W)，当前维度：{mask.ndim}"
    B, H, W = mask.shape
    
    # 统一转为布尔型（兼容0-1数值型mask）
    if mask.dtype != bool:
        mask = (mask > 0.5).astype(bool)
    
    largest_cc_mask = np.zeros_like(mask, dtype=bool)
    
    # 2. 遍历每个batch处理
    for b in range(B):
        single_mask = mask[b]  # (H, W)
        
        # 处理全False的情况（无连通域）
        if not np.any(single_mask):
            largest_cc_mask[b] = np.zeros((H, W), dtype=bool)
            continue
        
        # 处理全True的情况（整个mask就是最大连通域）
        if np.all(single_mask):
            largest_cc_mask[b] = np.ones((H, W), dtype=bool)
            continue
        
        # 3. 连通域分析（8邻域，更贴合视觉上的"连通"）
        # 转为uint8格式（cv2要求输入为0-255）
        mask_uint8 = single_mask.astype(np.uint8) * 255
        # 查找连通域：返回（连通域数量, 标签图, 统计信息, 中心坐标）
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_uint8, 
            connectivity=8,  # 8邻域（可选4邻域，根据需求调整）
            ltype=cv2.CV_32S
        )
        
        # 4. 排除背景（标签0），计算各连通域面积
        # stats格式：[x, y, width, height, area]
        cc_areas = stats[1:, 4]  # 跳过背景（标签0）的面积
        
        # 5. 找到面积最大的连通域标签
        max_area_idx = np.argmax(cc_areas) + 1  # +1是因为跳过了背景标签0
        
        # 6. 提取最大连通域
        largest_cc = (labels == max_area_idx).astype(bool)
        largest_cc_mask[b] = largest_cc
    
    return largest_cc_mask



def get_mask_min_rect_size(mask: np.ndarray) :
    """
    对B*H*W的mask，计算每个batch中覆盖所有有效区域的最小矩形的高和宽，返回列表形式
    参数：
        mask: numpy数组，shape=(B, H, W)，支持布尔型（True/False）或数值型（0/1）
    返回：
        rect_size_list: 二维列表，每个元素为 [h, w]，对应每个batch最小矩形的高度和宽度；
                        若batch无有效mask（全False/0），则返回 [0, 0]
    """
    # 1. 输入校验
    assert mask.ndim == 3, f"mask必须是3维(B,H,W)，当前维度：{mask.ndim}"
    B, H, W = mask.shape
    
    # 2. 统一转为布尔型（兼容数值型mask）
    if mask.dtype != bool:
        mask = (mask > 0.5).astype(bool)  # 数值型转布尔型，阈值0.5
    
    rect_size_list = []
    
    # 3. 遍历每个batch计算最小矩形
    for b in range(B):
        single_mask = mask[b]  # (H, W)
        
        # 获取所有有效像素的坐标 (y, x)
        y_coords, x_coords = np.where(single_mask)
        
        # 处理无有效区域的情况
        if len(y_coords) == 0 or len(x_coords) == 0:
            rect_size_list.append([0, 0])
            continue
        
        # 计算最小外接矩形的边界
        y_min = np.min(y_coords)
        y_max = np.max(y_coords)
        x_min = np.min(x_coords)
        x_max = np.max(x_coords)
        
        # 计算矩形的高和宽（注意：高对应y轴，宽对应x轴）
        rect_h = y_max - y_min + 1  # +1是因为坐标是闭区间（如y_min=0, y_max=1 → 高度2）
        rect_w = x_max - x_min + 1
        
        # 确保尺寸不超过原图（理论上不会，仅兜底）
        rect_h = min(rect_h, H)
        rect_w = min(rect_w, W)
        
        rect_size_list.append([rect_h, rect_w])
    
    return rect_size_list
def canny_with_mask_invert(background_imag, masks=None, canny_low=50, canny_high=240,blur_status=True):
    
    """
    对 tensor 图像计算 Canny 边缘，mask 以外区域置 0，同时添加 mask 自身的边界边缘；
    最终边缘处为 0、无边缘处为 1
    
    参数：
        background_imag: 输入图像 tensor（BCHW 或 CHW 格式，0-1 范围）
        masks: 分割掩码 array（shape: (B, num_masks, H, W) 或 (B, H, W)，元素为 True/False）
        canny_low: Canny 低阈值（默认 5）
        canny_high: Canny 高阈值（默认 150）
    
    返回：
        mask_tensor_invert: 反转后的边缘 tensor（BCHW，边缘=0，无边缘=1，含 mask 边界）
        mask_tensor: 原始边缘 tensor（BCHW，边缘=255/1，无边缘=0，含 mask 边界）
    """
    # 1. 处理输入图像 tensor → 适配批量维度
    if background_imag.dim() == 3:
        background_imag = background_imag.unsqueeze(0)  # CHW → BCHW
    batch_size,_,H,W = background_imag.shape
    if masks is None :
        masks=np.ones((batch_size,H,W))
    mask_invert_list = []
    mask_list = []

    # 2. 批量处理每个样本
    for i in range(batch_size):
        # ===== 步骤1：图像预处理（转 HWC + 0-255 整数）=====
        background_imag_b = background_imag[i].clamp(0, 1)
        img_np = background_imag_b.permute(1, 2, 0).cpu().numpy()  # CHW → HWC
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        H, W = img_np.shape[:2]

        # ===== 步骤2：mask 预处理（适配维度 + 转二值掩码）=====
        # 处理 mask 维度：若为 (num_masks, H, W) 则取第一个掩码；若为 (H,W) 直接用
        mask_i = masks[i] if len(masks.shape) == 4 else masks
        if mask_i.ndim == 3:
            mask_i = mask_i[0]  # 取第一个掩码（可根据需求调整索引）
        selected_mask = (mask_i.astype(np.uint8) * 255)  # (H, W)，True→255，False→0

        # ===== 步骤3：提取 mask 自身的边界边缘 =====
        # 用轮廓检测提取 mask 边界
        contours, _ = cv2.findContours(
            selected_mask, 
            cv2.RETR_EXTERNAL,  # 只提取最外层轮廓
            cv2.CHAIN_APPROX_SIMPLE  # 压缩轮廓点
        )
        # 创建空画布绘制 mask 边界
        mask_edge = np.zeros_like(selected_mask)
        cv2.drawContours(
            mask_edge, 
            contours, 
            -1,  # 绘制所有轮廓
            255,  # 轮廓颜色（白色）
            1  # 轮廓线宽度（可根据需求调整，如1/3）
        )  # mask_edge: 边界=255，其余=0

        # ===== 步骤4：图像 Canny 边缘检测 =====
        # 对图像进行模糊处理
        if blur_status:
            img_np = cv2.GaussianBlur(img_np, (5, 5), 0)

        gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        canny_edges = cv2.Canny(gray_img, canny_low, canny_high)  # 图像边缘=255，背景=0

        # ===== 步骤5：融合「图像边缘」和「mask 边界」=====
        # 按位或：只要有一个边缘（图像/mask）就保留
        fused_edges = cv2.bitwise_or(canny_edges, mask_edge)  # 融合后边缘=255，背景=0

        # ===== 步骤6：掩码过滤：仅保留 mask 内的融合边缘 =====
        fused_edges_masked = cv2.bitwise_and(fused_edges, selected_mask)  # mask 外→0，mask 内边缘→255

        # ===== 步骤7：像素值反转（边缘255→0，背景0→255）=====
        inverted_fused = cv2.bitwise_not(fused_edges_masked)  # 反转后：边缘=0，无边缘=255

        # ===== 步骤8：转换为 3 通道 + 0-1 范围 =====
        # 处理反转后的结果（最终返回的 invert 版本）
        result_inverted = cv2.cvtColor(inverted_fused.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        result_inverted = result_inverted / 255.0  # 0-1 float
        # 转为 tensor（CHW）
        result_inverted = torch.from_numpy(result_inverted).permute(2, 0, 1).float()

        # 处理原始边缘结果（未反转版本）
        result_origin = cv2.cvtColor(fused_edges_masked.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        result_origin = result_origin / 255.0  # 0-1 float
        # 转为 tensor（CHW）
        result_origin = torch.from_numpy(result_origin).permute(2, 0, 1).float()

        # ===== 步骤9：收集结果 =====
        mask_invert_list.append(result_inverted)
        mask_list.append(result_origin)

    # 拼接批量维度（BCHW）
    mask_tensor_invert = torch.stack(mask_invert_list)
    mask_tensor = torch.stack(mask_list)
    
    return mask_tensor_invert, mask_tensor





def crop_mask_region(
    img_tensor: torch.Tensor, 
    mask: np.ndarray
) -> Tuple[torch.Tensor, List[List[int]]]:
    """
    对B*H*W的mask找到每个batch的最大连通域，计算包含该连通域的最小矩形框，
    裁剪B*C*H*W图像张量中对应矩形框内的内容，同时返回矩形框坐标
    参数：
        img_tensor: 输入图像张量，shape=(B, C, H, W)，支持CPU/GPU，float32类型
        mask: numpy数组，shape=(B, H, W)，支持bool/uint8/float32等类型（非0为有效区域）
    返回：
        cropped_imgs: 裁剪后的图像张量，每个元素为对应batch的矩形框内图像（shape=(b,C, h, w)），全空mask返回空tensor
        rect_coord_list: 二维列表，每个元素为[x_min, y_min, x_max, y_max]，对应每个batch最大连通域的最小矩形框；全空mask返回[0,0,0,0]
    """
    # 1. 输入校验
    assert img_tensor.ndim == 4, f"图像张量必须是4维(B,C,H,W)，当前维度：{img_tensor.ndim}"
    assert mask.ndim == 3, f"mask必须是3维(B,H,W)，当前维度：{mask.ndim}"
    assert img_tensor.shape[0] == mask.shape[0], f"图像和mask的batch数不匹配：{img_tensor.shape[0]} vs {mask.shape[0]}"
    assert img_tensor.shape[2:] == mask.shape[1:], f"图像和mask的HW维度不匹配：{img_tensor.shape[2:]} vs {mask.shape[1:]}"
    
    B, C, H, W = img_tensor.shape
    device = img_tensor.device
    cropped_imgs = []  # 存储每个batch裁剪后的图像
    rect_coord_list = []  # 存储每个batch的矩形框坐标

    # 2. 遍历每个batch处理
    for b in range(B):
        single_mask = mask[b]  # (H, W) 当前batch的mask
        single_img = img_tensor[b]  # (C, H, W) 当前batch的图像

        # ---------------------- 步骤1：预处理mask为二值图 ----------------------
        # 转为bool型（非0为True，0为False）
        mask_bool = (single_mask > 0.5).astype(np.uint8) if single_mask.dtype != bool else single_mask.astype(np.uint8)
        
        # ---------------------- 步骤2：找最大连通域 ----------------------
        # 查找连通域（cv2.connectedComponentsWithStats支持uint8二值图）
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bool, connectivity=8)
        
        # 处理无有效连通域的情况（仅背景）
        if num_labels < 2:
            rect_coord_list.append([0, 0, 0, 0])
            cropped_imgs.append(torch.empty((C, 0, 0), device=device))  # 空tensor
            continue
        
        # ---------------------- 步骤3：筛选最大连通域（排除背景标签0） ----------------------
        # stats格式：[x, y, width, height, area]
        # 跳过背景（标签0），找面积最大的连通域
        max_area = 0
        max_label = 1
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area > max_area:
                max_area = area
                max_label = label
        
        # ---------------------- 步骤4：计算最大连通域的最小矩形框 ----------------------
        # 方法1：用stats直接获取外接矩形（更快）
        x_min = stats[max_label, cv2.CC_STAT_LEFT]
        y_min = stats[max_label, cv2.CC_STAT_TOP]
        width = stats[max_label, cv2.CC_STAT_WIDTH]
        height = stats[max_label, cv2.CC_STAT_HEIGHT]
        x_max = x_min + width
        y_max = y_min + height

        # 边界校验（确保在图像范围内）
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(W, x_max)
        y_max = min(H, y_max)

        # 保存矩形框坐标 [x_min, y_min, x_max, y_max]
        rect_coord_list.append([x_min, y_min, x_max, y_max])

        # ---------------------- 步骤5：裁剪图像中对应矩形框的内容 ----------------------
        # Tensor切片（保留梯度，支持GPU）
        cropped_img = single_img[:, y_min:y_max, x_min:x_max]  # (C, h, w)
        cropped_imgs.append(cropped_img)
        
    cropped_imgs=torch.stack(cropped_imgs)
    return cropped_imgs, rect_coord_list







def resize_images_keep_aspect(
    images: torch.Tensor,          # 输入图像张量，shape=(B, C, H1, W1)
    target_size: Tuple[int, int]   # 目标尺寸 (h, w) → (target_h, target_w)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对B*C*H1*W1格式的图像张量进行等比例缩放，空白区域填充0（从左上角开始填充），返回缩放后图像和每个batch的缩放比例
    参数：
        images: 输入图像张量，支持CPU/GPU，浮点型（float32/float64）/整数型（uint8/int）
        target_size: 目标尺寸，格式为 (target_h, target_w)
    返回：
        resized_images: 缩放后的图像张量，shape=(B, C, target_h, target_w)
        scales: 每个batch的缩放比例，shape=(B,)，为宽/高方向的最小缩放比
    """
    # 1. 输入校验
    assert images.ndim == 4, f"输入图像必须是4维(B,C,H,W)，当前维度：{images.ndim}"
    assert len(target_size) == 2, f"目标尺寸必须是(h,w)二元组，当前长度：{len(target_size)}"
    target_h, target_w = target_size
    assert target_h > 0 and target_w > 0, f"目标尺寸必须为正整数：{target_size}"

    B, C, H1, W1 = images.shape
    device = images.device
    dtype = images.dtype

    # 2. 初始化输出张量和缩放比例数组
    resized_images = torch.zeros((B, C, target_h, target_w), device=device, dtype=dtype)
    scales = torch.zeros(B, device=device, dtype=torch.float32)

    # 3. 逐batch计算缩放比例并缩放
    for b in range(B):
        # 单batch图像：(C, H1, W1)
        img = images[b]
        
        # 计算宽/高缩放比例
        scale_w = target_w / W1  # 宽度缩放比（目标宽/原图宽）
        scale_h = target_h / H1  # 高度缩放比（目标高/原图高）
        scale = min(scale_w, scale_h)  # 选最小比例，保证等比例且完全放入目标尺寸
        scales[b] = scale

        # 计算等比例缩放后的新尺寸（取整避免浮点误差）
        new_h = int(round(H1 * scale))
        new_w = int(round(W1 * scale))

        # 4. 等比例缩放（保留梯度，适配数值类型选择插值方式）
        img_4d = img.unsqueeze(0)  # 扩展为(1, C, H1, W1)适配interpolate
        resized_img = torch.nn.functional.interpolate(
            img_4d,
            size=(new_h, new_w),
            mode='bilinear' if dtype.is_floating_point else 'nearest',  # 浮点用双线性，整数用最近邻
            align_corners=False
        ).squeeze(0)  # 恢复为(C, new_h, new_w)

        # 5. 从左上角开始填充（核心修改：顶部/左侧无填充，底部/右侧补0）
        pad_top = 0  # 顶部无填充
        pad_left = 0  # 左侧无填充
        # 仅限制新尺寸不超过目标尺寸（避免越界）
        new_h_clamped = min(new_h, target_h)
        new_w_clamped = min(new_w, target_w)

        # 赋值到目标张量（左上角开始填充，超出目标尺寸部分裁剪）
        resized_images[b, :, pad_top:pad_top+new_h_clamped, pad_left:pad_left+new_w_clamped] = resized_img[:, :new_h_clamped, :new_w_clamped]

    return resized_images, scales


def resized_images(
    images: torch.Tensor,          # 输入图像张量，shape=(B, C, H, W)
    scale    # 缩放比例（单值/每个batch独立比例）
) -> torch.Tensor:
    """
    按指定比例对B*C*H*W图像张量进行等比例缩放，返回缩放后图像张量
    参数：
        images: 输入图像张量，支持CPU/GPU，浮点型（float32/float64）/整数型（uint8/int）
        scale: 缩放比例 - 单值float：所有batch使用同一比例；Tensor(B,)：每个batch独立比例
    返回：
        resized_images: 缩放后的图像张量，shape=(B, C, new_h, new_w)（new_h=H*scale, new_w=W*scale）
    """
    # 1. 输入校验
    assert images.ndim == 4, f"输入图像必须是4维(B,C,H,W)，当前维度：{images.ndim}"
    if isinstance(scale, torch.Tensor):
        assert scale.numel() in [1, images.shape[0]], \
            f"缩放比例张量长度需为1或batch数({images.shape[0]})，当前长度：{scale.numel()}"
        scale = scale.to(images.device, dtype=torch.float32)
    else:
        assert isinstance(scale, (int, float)) and scale > 0, \
            f"缩放比例必须为正数值，当前值：{scale}"
        scale = torch.tensor([scale]*images.shape[0], device=images.device, dtype=torch.float32)

    B, C, H, W = images.shape
    device = images.device
    dtype = images.dtype

    # 2. 逐batch计算新尺寸并缩放
    resized_list = []
    for b in range(B):
        current_scale = scale[b].item() if scale.numel() > 1 else scale[0].item()
        
        # 计算等比例缩放后的新尺寸（取整避免浮点误差）
        new_h = int(round(H * current_scale))
        new_w = int(round(W * current_scale))
        # 确保尺寸≥1（避免缩放比例过小导致尺寸为0）
        new_h = max(1, new_h)
        new_w = max(1, new_w)

        # 3. 缩放图像（保留梯度，适配数值类型选择插值方式）
        img = images[b].unsqueeze(0)  # (1, C, H, W)
        resized_img = torch.nn.functional.interpolate(
            img,
            size=(new_h, new_w),
            mode='bilinear' if dtype.is_floating_point else 'nearest',  # 浮点用双线性，整数用最近邻
            align_corners=False
        )
        resized_list.append(resized_img.squeeze(0))  # (C, new_h, new_w)

    # 4. 堆叠为批量张量
    resized_images = torch.stack(resized_list, dim=0)  # (B, C, new_h, new_w)
    return resized_images




def paste_images_to_background(
    images: torch.Tensor,               # 待粘贴的前景图像，shape=(B, C, H_img, W_img)
    rect_coord_list: List[List[int]],   # 每个batch的目标矩形框 [x_min, y_min, x_max, y_max]
    background: torch.Tensor            # 背景图像，shape=(B, C, H_bg, W_bg) 或 (C, H_bg, W_bg)
) -> torch.Tensor:
    """
    将批量前景图像粘贴到背景图像的指定矩形框位置，返回合成后的背景图像
    参数：
        images: 前景图像张量，B*C*H_img*W_img，支持CPU/GPU，浮点型（0-1）/整数型（0-255）
        rect_coord_list: 每个batch的目标矩形框坐标 [x_min, y_min, x_max, y_max]，长度需等于batch数
        background: 背景图像张量 - 批量模式(B*C*H_bg*W_bg) / 单背景模式(C*H_bg*W_bg)（自动广播到所有batch）
    返回：
        composite_bg: 合成后的背景图像，shape=(B, C, H_bg, W_bg)，前景图像被粘贴到指定矩形框位置
    """
    # 1. 输入校验
    assert images.ndim == 4, f"前景图像必须是4维(B,C,H,W)，当前维度：{images.ndim}"
    B, C, H_img, W_img = images.shape
    assert len(rect_coord_list) == B, f"矩形框列表长度({len(rect_coord_list)})需等于batch数({B})"
    
    # 处理背景图像维度（单背景→广播到所有batch）
    if background.ndim == 3:
        C_bg, H_bg, W_bg = background.shape
        assert C_bg == C, f"前景/背景通道数不匹配：前景{C}，背景{C_bg}"
        # 广播为批量背景：(C, H_bg, W_bg) → (B, C, H_bg, W_bg)
        background = background.unsqueeze(0).repeat(B, 1, 1, 1)
    elif background.ndim == 4:
        B_bg, C_bg, H_bg, W_bg = background.shape
        assert B_bg == B and C_bg == C, f"背景batch数({B_bg})/通道数({C_bg})需匹配前景({B}/{C})"
    else:
        raise ValueError(f"背景图像维度需为3维(C,H,W)或4维(B,C,H,W)，当前维度：{background.ndim}")
    
    # 设备/类型对齐
    device = images.device
    background = background.to(device, dtype=images.dtype)
    # 复制背景避免修改原张量
    composite_bg = background.clone()

    # 2. 逐batch粘贴图像到指定矩形框
    for b in range(B):
        # 获取当前batch的矩形框坐标
        x_min, y_min, x_max, y_max = rect_coord_list[b]
        # 计算目标矩形框尺寸
        target_h = y_max - y_min
        target_w = x_max - x_min

        # 跳过无效矩形框（尺寸≤0）
        if target_h <= 0 or target_w <= 0:
            continue

        # 校验矩形框是否在背景范围内
        assert 0 <= x_min < x_max <= W_bg, f"Batch{b}矩形框X范围[{x_min},{x_max}]超出背景宽度{W_bg}"
        assert 0 <= y_min < y_max <= H_bg, f"Batch{b}矩形框Y范围[{y_min},{y_max}]超出背景高度{H_bg}"

        # 3. 缩放前景图像到矩形框尺寸（等比例缩放+居中填充，保证不变形）
        img = images[b]  # (C, H_img, W_img)
        # 计算缩放比例（选最小比例，保证前景完全放入矩形框）
        scale_w = target_w / W_img
        scale_h = target_h / H_img
        scale = min(scale_w, scale_h)
        # 等比例缩放后的尺寸
        new_h = int(round(H_img * scale))
        new_w = int(round(W_img * scale))
        # 缩放前景图像
        img_4d = img.unsqueeze(0)  # (1, C, H_img, W_img)
        resized_img = torch.nn.functional.interpolate(
            img_4d,
            size=(new_h, new_w),
            mode='bilinear' if images.dtype.is_floating_point else 'nearest',
            align_corners=False
        ).squeeze(0)  # (C, new_h, new_w)

        # 4. 计算居中偏移（前景在矩形框内居中）
        offset_y = (target_h - new_h) // 2
        offset_x = (target_w - new_w) // 2
        # 计算在背景中的实际粘贴坐标
        paste_y1 = y_min + offset_y
        paste_y2 = paste_y1 + new_h
        paste_x1 = x_min + offset_x
        paste_x2 = paste_x1 + new_w

        # 5. 粘贴前景图像到背景指定位置
        composite_bg[b, :, paste_y1:paste_y2, paste_x1:paste_x2] = resized_img

    return composite_bg



def paste_images_to_background_no_scale(
    images: torch.Tensor,               # 待粘贴的前景图像，shape=(B, C, H_img, W_img)
    rect_coord_list: List[List[int]],   # 每个batch的目标矩形框 [x_min, y_min, x_max, y_max]
    background: torch.Tensor            # 背景图像，shape=(B, C, H_bg, W_bg) 或 (C, H_bg, W_bg)
) -> torch.Tensor:
    """
    【不缩放前景】将批量前景图像直接粘贴到背景图像的指定矩形框位置，超出框部分裁剪，不足仅贴有效区域
    参数：
        images: 前景图像张量，B*C*H_img*W_img，支持CPU/GPU，浮点型（0-1）/整数型（0-255）
        rect_coord_list: 每个batch的目标矩形框坐标 [x_min, y_min, x_max, y_max]，长度需等于batch数
        background: 背景图像张量 - 批量模式(B*C*H_bg*W_bg) / 单背景模式(C*H_bg*W_bg)（自动广播到所有batch）
    返回：
        composite_bg: 合成后的背景图像，shape=(B, C, H_bg, W_bg)
    """
    # 1. 输入校验
    assert images.ndim == 4, f"前景图像必须是4维(B,C,H,W)，当前维度：{images.ndim}"
    B, C, H_img, W_img = images.shape
    assert len(rect_coord_list) == B, f"矩形框列表长度({len(rect_coord_list)})需等于batch数({B})"
    
    # 处理背景图像维度（单背景→广播到所有batch）
    if background.ndim == 3:
        C_bg, H_bg, W_bg = background.shape
        assert C_bg == C, f"前景/背景通道数不匹配：前景{C}，背景{C_bg}"
        # 广播为批量背景：(C, H_bg, W_bg) → (B, C, H_bg, W_bg)
        background = background.unsqueeze(0).repeat(B, 1, 1, 1)
    elif background.ndim == 4:
        B_bg, C_bg, H_bg, W_bg = background.shape
        assert B_bg == B and C_bg == C, f"背景batch数({B_bg})/通道数({C_bg})需匹配前景({B}/{C})"
    else:
        raise ValueError(f"背景图像维度需为3维(C,H,W)或4维(B,C,H,W)，当前维度：{background.ndim}")
    
    # 设备/类型对齐，复制背景避免修改原张量
    device = images.device
    composite_bg = background.to(device, dtype=images.dtype).clone()

    # 2. 逐batch粘贴图像（不缩放，直接贴）
    for b in range(B):
        # 获取当前batch的矩形框坐标
        x_min, y_min, x_max, y_max = rect_coord_list[b]
        # 目标矩形框尺寸
        target_h = y_max - y_min
        target_w = x_max - x_min

        # 跳过无效矩形框（尺寸≤0）
        if target_h <= 0 or target_w <= 0:
            continue

        # 校验矩形框是否在背景范围内（仅警告，自动裁剪到背景边界）
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(W_bg, x_max)
        y_max = min(H_bg, y_max)

        # 3. 计算前景图像的粘贴区域（直接映射，超出部分裁剪）
        # 前景在矩形框内的有效粘贴范围
        paste_h = min(H_img, y_max - y_min)  # 前景高度 vs 框高度，取较小值
        paste_w = min(W_img, x_max - x_min)  # 前景宽度 vs 框宽度，取较小值

        # 4. 执行粘贴（仅粘贴有效区域）
        if paste_h > 0 and paste_w > 0:
            # 前景区域：取左上角paste_h*paste_w（超出部分裁剪）
            img_patch = images[b, :, :paste_h, :paste_w]
            # 背景区域：矩形框内对应位置
            composite_bg[b, :, y_min:y_min+paste_h, x_min:x_min+paste_w] = img_patch

    return composite_bg


