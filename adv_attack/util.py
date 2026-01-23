import copy
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
import yaml
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
# 默认输出的是对抗损失，输出的loss 需要最小化，
class YOLOv11DetectionLoss(nn.Module):
    def __init__(self, num_classes=80,
                 ** args):
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
        if not hasattr(self, 'adv_loss_type'):  # 判断是否存在属性
            self.adv_loss_type = 0
        else :
            self.adv_loss_type = self.adv_loss_type

        if not hasattr(self, 'adv_weight_class'):  # 判断是否存在属性
            self.weight_class = 1
        else :
            self.weight_class = self.adv_weight_class
   

        if not hasattr(self, 'adv_weight_bbox_l1'):  # 判断是否存在属性
            self.weight_bbox_l1 = 1
        else :
            self.weight_bbox_l1 = self.adv_weight_bbox_l1

        if not hasattr(self, 'adv_weight_giou'):  # 判断是否存在属性
            self.weight_giou = 1
        else :
            self.weight_giou = self.adv_weight_giou

        if not hasattr(self,'target_class'):
            self.target_class = 16
        else :
            self.y_adv=self.target_class

        # 判断是否存在penalty_bbox，不存在则创建
        if not hasattr(self, 'penalty_bbox'):  # 判断是否存在属性
            # 对抗损失，默认没有检测到就是最好，所以取-1，因为后面总损失都是相加
            self.penalty_bbox = -1
        if not hasattr(self, 'penalty_giou'):  # 判断是否存在属性
            # giou损失，默认没有检测到就是最好，所以取0，因为后面总损失都是相加
            # 这里直接计算的就是IOU，不包含负数
            self.penalty_giou =0
            
        if not hasattr(self, 'penalty_class'):  # 判断是否存在属性
            # 交叉熵损失的最大值，与类别有关
            if self.adv_loss_type == 0:  # 默认使用交叉熵
                self.penalty_class = -13.82
            elif self.adv_loss_type == 1:  # 默认预测的概率最小，最小是0
                self.penalty_class = 0
            elif self.adv_loss_type == 2:  # 默认使用L1
                self.penalty_class = 0    #  #   -13.8155 

            
        if not hasattr(self, 'image_size'):  # 判断是否存在属性
            self.image_resolution = 512
        else :
            self.image_resolution = self.image_size

        # 基础损失函数（分类损失改用交叉熵的简化形式，适配离散标签）
        self.class_criterion = nn.NLLLoss()  # 负对数似然损失（需输入log概率）
        self.bbox_l1_criterion = nn.L1Loss()  # 边界框L1损失（不自动降维）

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
    #     遍历每个真实框（gt）单独计算损失，确保每个gt都参与损失计算
    #     Args:
    #         pred_result: 模型预测结果
    #             'boxes': list[tensor] → [B个元素，每个元素 shape [M_b, 4] (xyxy)]
    #             'scores': list[tensor] → [B个元素，每个元素 shape [M_b]]
    #             'labels': list[tensor] → [B个元素，每个元素 shape [M_b]]
    #         gt_result: 真实标签
    #             'boxes': list[tensor] → [B个元素，每个元素 shape [K_b, 4] (xyxy)]
    #             'labels': list[tensor] → [B个元素，每个元素 shape [K_b]]
    #     Returns:
    #         total_loss: 总损失（标量）
    #         loss_dict: 各任务损失明细（dict of 标量）
    #     """   
    #     # 确定设备
    #     if len(pred_result['boxes']) == 0 or (len(pred_result['boxes']) > 0 and pred_result['boxes'][0].numel() == 0):
    #         device = torch.device('cpu')
    #     else:
    #         device = pred_result['boxes'][0].device
            
    #     # 初始化损失累计变量（按每个gt单独计算）
    #     total_class_loss = torch.tensor(0.0, device=device)
    #     total_bbox_l1_loss = torch.tensor(0.0, device=device)
    #     total_giou_loss = torch.tensor(0.0, device=device)
    #     total_gt_count = 0  # 统计总真实框数量（用于平均损失）

    #     batch_size = len(pred_result['boxes'])
    #     if batch_size == 0:
    #         # 空batch处理
    #         total_bbox_l1_loss = torch.tensor(self.penalty_bbox, device=device)
    #         total_giou_loss = torch.tensor(self.penalty_giou, device=device)
    #         total_class_loss = torch.tensor(self.penalty_class, device=device)
    #         total_loss = total_class_loss + total_bbox_l1_loss + total_giou_loss
    #         return total_loss, {
    #             'total_loss': total_loss,
    #             'class_loss': total_class_loss,
    #             'bbox_l1_loss': total_bbox_l1_loss,
    #             'giou_loss': total_giou_loss
    #         }
        
    #     for b in range(batch_size):
    #         # 提取单样本数据
    #         pred_boxes = pred_result['boxes'][b].to(device)  # [M_b, 4]
    #         pred_scores = pred_result['scores'][b].to(device)  # [M_b]
    #         pred_labels = pred_result['labels'][b].to(device)  # [M_b]
    #         pred_scores_vector=pred_result['scores_vector'][b].to(device)
    #         gt_boxes = gt_result['boxes'][b].to(device)  # [K_b, 4]
    #         gt_labels = gt_result['labels'][b].to(device)  # [K_b]
    #         gt_scores_vector=gt_result['scores_vector'][b].to(device)
    #         K_b = gt_boxes.shape[0]  # 该样本的真实框数量
    #         if K_b == 0:
    #             continue  # 无真实框则跳过
    #         total_gt_count += K_b  # 累计总真实框数量

    #         # 过滤低置信度预测框
    #         pred_mask = pred_scores >= self.conf_thres
    #         pred_boxes = pred_boxes[pred_mask]
    #         pred_scores = pred_scores[pred_mask]
    #         pred_labels = pred_labels[pred_mask]
    #         pred_scores_vector=pred_scores_vector[pred_mask]
    #         M_b = pred_boxes.shape[0]  # 过滤后的预测框数量

    #         # 遍历该样本的每个真实框（核心修改：循环所有gt）
    #         for gt_idx in range(K_b):
    #             # 提取当前真实框（单独处理）
    #             current_gt_box = gt_boxes[gt_idx:gt_idx+1]  # [1, 4]（保持维度）
    #             current_gt_label = gt_labels[gt_idx:gt_idx+1]  # [1]
    #             current_gt_scores_vector=gt_scores_vector[gt_idx:gt_idx+1]
    #             if M_b == 0:
    #                 # 无预测框：对当前gt施加惩罚损失
    #                 total_class_loss += torch.tensor(self.penalty_class, device=device)
    #                 # bbox 惩罚损失，偏差越大越好，所以减去偏差最大，减去1
    #                 total_bbox_l1_loss += torch.tensor(self.penalty_bbox, device=device)
    #                 # iou 正常是0-1 ，无效说明最好，所以取0
    #                 total_giou_loss += torch.tensor(self.penalty_giou, device=device)
    #                 continue

    #             # 为当前真实框匹配最优预测框（简化匹配：计算与当前gt的成本）
    #             # 1. 计算当前gt与所有预测框的IoU [M_b, 1]
    #             iou_matrix = self._compute_iou(pred_boxes, current_gt_box)  # [M_b, 1]
                
    #             # 2. 计算成本矩阵 [M_b, 1]
    #             class_match = (pred_labels.unsqueeze(1) == current_gt_label).float()  # [M_b, 1]
    #             confidence_term = pred_scores.unsqueeze(1)  # [M_b, 1]
    #             cost_matrix = -(confidence_term * iou_matrix) + (1 - class_match) * 100  # 成本越低越好
                
    #             # 3. 选择成本最低的预测框作为匹配
    #             min_cost, best_pred_idx = torch.min(cost_matrix, dim=0)  # 找到最优预测框索引
    #             best_pred_idx = best_pred_idx.item()  # 转为标量索引

    #             # 4. 检查匹配有效性（IoU达标+类别匹配）
    #             # valid = (iou_matrix[best_pred_idx] >= self.iou_thres) 
    #             valid = (iou_matrix[best_pred_idx] >= 0)
    #             if not valid:
    #                 # 无效匹配：施加惩罚

    #                 total_class_loss += torch.tensor(self.penalty_class, device=device)
    #                 # bbox 惩罚损失，偏差越大越好，所以减去偏差最大，减去1
    #                 total_bbox_l1_loss += torch.tensor(self.penalty_bbox, device=device)
    #                 # iou 正常是0-1 ，无效说明最好，所以取0
    #                 total_giou_loss += torch.tensor(self.penalty_giou, device=device)                    
                    
    #                 continue

    #             # 5. 提取匹配的预测框
    #             matched_pred_box = pred_boxes[best_pred_idx:best_pred_idx+1]  # [1, 4]
    #             matched_pred_score = pred_scores[best_pred_idx:best_pred_idx+1]  # [1]
    #             matched_pred_label = pred_labels[best_pred_idx:best_pred_idx+1]  # [1]
    #             matched_pred_scores_vector=pred_scores_vector[best_pred_idx:best_pred_idx+1]
                
    #             # 6. 计算分类损失（当前gt的分类损失）
    #             # pred_log_prob = torch.log(matched_pred_score.clamp(min=1e-6, max=1.0))  # [1]
    #             # class_log_probs = torch.full((1, self.num_classes),  -1e6, device=device)
    #             # class_log_probs[0, matched_pred_label] = pred_log_prob
    #             # class_loss = self.class_criterion(class_log_probs, current_gt_label).sum()
    #             # log计算
    #             matched_pred_scores_vector_log=torch.log(matched_pred_scores_vector.clamp(min=1e-6, max=1.0))
    #             # 默认返回的都是对抗损失
    #             if self.adv_loss_type==0:
    #                 # 0表示与gt的交叉熵最大
    #                 class_loss = self.class_criterion(matched_pred_scores_vector_log, current_gt_label)
    #                 class_loss=-class_loss
                
    #             elif self.adv_loss_type==1:
    #                 # 当前预测类别的概率最小

    #                 pred_label_scalar = matched_pred_label.item()  # 单个预测类别索引
    
    #                 pred_log_prob = matched_pred_scores_vector[0, pred_label_scalar]  # 当前类别的对数概率

    #                 class_loss = pred_log_prob
    #             elif self.adv_loss_type==2:
    #                 # 

    #                 log_probs = matched_pred_scores_vector_log # [1, num_classes]，log(p_i)

    #                 # 步骤2：提取原始类别y和目标类别y_adv的对数概率
    #                 # 默认预测的目标类型
    #                 y = matched_pred_label.item() if isinstance(matched_pred_label, torch.Tensor) else current_gt_label
    #                 # 当此模式，默认参考的是对抗样本的标签
    #                 y_adv = current_gt_label.item() if isinstance(current_gt_label, torch.Tensor) else target_adv_label

    #                 log_p_y = log_probs[0, y]  # log(p_y(x'))，原始类别的对数概率
    #                 log_p_yadv = log_probs[0, y_adv]  # log(p_yadv(x'))，目标类别的对数概率

    #                 # 步骤3：计算目标攻击损失：-log(p_yadv(x')) + log(p_y(x'))
    #                 class_loss = -log_p_yadv + log_p_y


    #             total_class_loss += class_loss
    #             # 7. 计算边界框L1损失
    #             bbox_l1_loss = self.bbox_l1_criterion(matched_pred_box, current_gt_box)
    #             # 要使得偏差变大，所以要取负
    #             total_bbox_l1_loss -= (bbox_l1_loss / (4 * self.image_resolution))  # 归一化

    #             # 8. 计算GIoU损失，对抗损失，所以不需要1-IOU
    #             giou = generalized_box_iou(matched_pred_box, current_gt_box)  # [1]
    #             giou_loss = (giou).sum()
    #             total_giou_loss += giou_loss

    #     # 计算平均损失（除以总真实框数量）

    #     if total_gt_count > 0:
    #         class_loss_avg = total_class_loss / total_gt_count
    #         bbox_l1_loss_avg = total_bbox_l1_loss / total_gt_count
    #         giou_loss_avg = total_giou_loss / total_gt_count

    #     # 总损失 = 加权和
    #     total_loss = (
    #         class_loss_avg * self.weight_class +
    #         bbox_l1_loss_avg * self.weight_bbox_l1 +
    #         giou_loss_avg * self.weight_giou
    #     )

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
        关键改造：每个GT匹配后排除对应预测框，后续GT仅在剩余预测框中匹配
        Args:
            pred_result: 模型预测结果
                'boxes': list[tensor] → [B个元素，每个元素 shape [M_b, 4] (xyxy)]
                'scores': list[tensor] → [B个元素，每个元素 shape [M_b]]
                'labels': list[tensor] → [B个元素，每个元素 shape [M_b]]
                'scores_vector': list[tensor] → [B个元素，每个元素 shape [M_b, num_classes]]
            gt_result: 真实标签
                'boxes': list[tensor] → [B个元素，每个元素 shape [K_b, 4] (xyxy)]
                'labels': list[tensor] → [B个元素，每个元素 shape [K_b]]
                'scores_vector': list[tensor] → [B个元素，每个元素 shape [K_b, num_classes]]
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
            pred_scores_vector = pred_result['scores_vector'][b].to(device)  # [M_b, num_classes]
            gt_boxes = gt_result['boxes'][b].to(device)  # [K_b, 4]
            gt_labels = gt_result['labels'][b].to(device)  # [K_b]
            gt_scores_vector = gt_result['scores_vector'][b].to(device)  # [K_b, num_classes]
            K_b = gt_boxes.shape[0]  # 该样本的真实框数量
            if K_b == 0:
                continue  # 无真实框则跳过
            total_gt_count += K_b  # 累计总真实框数量

            # 过滤低置信度预测框
            pred_mask = pred_scores >= self.conf_thres
            pred_boxes = pred_boxes[pred_mask]
            pred_scores = pred_scores[pred_mask]
            pred_labels = pred_labels[pred_mask]
            pred_scores_vector = pred_scores_vector[pred_mask]
            M_b = pred_boxes.shape[0]  # 过滤后的预测框数量

            # 核心改造1：维护未被匹配的预测框索引集合（初始为所有索引）
            unmatched_pred_indices = torch.arange(M_b, device=device)  # [0,1,...,M_b-1]

            # 遍历该样本的每个真实框
            for gt_idx in range(K_b):
                # 提取当前真实框（单独处理）
                current_gt_box = gt_boxes[gt_idx:gt_idx+1]  # [1, 4]（保持维度）
                current_gt_label = gt_labels[gt_idx:gt_idx+1]  # [1]
                current_gt_scores_vector = gt_scores_vector[gt_idx:gt_idx+1]  # [1, num_classes]

                # 核心改造2：检查剩余未匹配的预测框数量
                if len(unmatched_pred_indices) == 0:
                    # 无剩余预测框：对当前gt施加惩罚损失
                    total_class_loss += torch.tensor(self.penalty_class, device=device)
                    total_bbox_l1_loss += torch.tensor(self.penalty_bbox, device=device)
                    total_giou_loss += torch.tensor(self.penalty_giou, device=device)
                    continue

                # 提取未被匹配的预测框子集
                sub_pred_boxes = pred_boxes[unmatched_pred_indices]  # [M_remain, 4]
                sub_pred_scores = pred_scores[unmatched_pred_indices]  # [M_remain]
                sub_pred_labels = pred_labels[unmatched_pred_indices]  # [M_remain]
                sub_pred_scores_vector = pred_scores_vector[unmatched_pred_indices]  # [M_remain, num_classes]

                # 1. 计算当前gt与剩余预测框的IoU [M_remain, 1]
                iou_matrix = self._compute_iou(sub_pred_boxes, current_gt_box)  # [M_remain, 1]
                
                # 2. 计算成本矩阵 [M_remain, 1]
                class_match = (sub_pred_labels.unsqueeze(1) == current_gt_label).float()  # [M_remain, 1]
                confidence_term = sub_pred_scores.unsqueeze(1)  # [M_remain, 1]
                # cost_matrix = -(confidence_term * iou_matrix) + (1 - class_match) * 100  # 成本越低越好
                cost_matrix = -(confidence_term * iou_matrix) 
                # 3. 选择成本最低的预测框（在剩余子集内的索引）
                min_cost, sub_best_idx = torch.min(cost_matrix, dim=0)  # sub_best_idx: 子集内的索引
                sub_best_idx = sub_best_idx.item()  # 转为标量

                # 4. 映射回原始预测框的索引，并检查匹配有效性
                best_pred_idx = unmatched_pred_indices[sub_best_idx]  # 子集索引 → 原始索引
                valid = (iou_matrix[best_pred_idx] >= self.iou_thres) 
                # valid = (iou_matrix[sub_best_idx] >= 0)  # 检查IoU有效性
                if not valid:
                    # 无效匹配：施加惩罚
                    total_class_loss += torch.tensor(self.penalty_class, device=device)
                    total_bbox_l1_loss += torch.tensor(self.penalty_bbox, device=device)
                    total_giou_loss += torch.tensor(self.penalty_giou, device=device)                    
                    continue

                # 核心改造3：从未匹配集合中移除当前匹配的预测框索引
                unmatched_pred_indices = unmatched_pred_indices[unmatched_pred_indices != best_pred_idx]

                # 5. 提取匹配的预测框（原始索引）
                matched_pred_box = pred_boxes[best_pred_idx:best_pred_idx+1]  # [1, 4]
                matched_pred_score = pred_scores[best_pred_idx:best_pred_idx+1]  # [1]
                matched_pred_label = pred_labels[best_pred_idx:best_pred_idx+1]  # [1]
                matched_pred_scores_vector = pred_scores_vector[best_pred_idx:best_pred_idx+1]  # [1, num_classes]
                
                # 6. 计算分类损失（对抗损失逻辑）
                matched_pred_scores_vector_log = torch.log(matched_pred_scores_vector.clamp(min=1e-6, max=1.0))
                if self.adv_loss_type == 0:
                    # 与gt的交叉熵取负（最大化交叉熵）
                    class_loss = self.class_criterion(matched_pred_scores_vector_log, current_gt_label)
                    class_loss = -class_loss
                elif self.adv_loss_type == 1:
                    # 当前预测类别的概率最小（取该类别的log概率）
                    pred_label_scalar = matched_pred_label.item()
                    pred_log_prob = matched_pred_scores_vector[0, pred_label_scalar]
                    class_loss = pred_log_prob
                elif self.adv_loss_type == 2:
                    # 攻击损失：-log(p_yadv) + log(p_y)
                    log_probs = matched_pred_scores_vector_log  # [1, num_classes]
                    # y = matched_pred_label.item()
                    # y_adv = current_gt_label.item()
                    y = current_gt_label.item()
                    y_adv = self.y_adv
                    log_p_y = log_probs[0, y]
                    log_p_yadv = log_probs[0, y_adv]
                    class_loss = -log_p_yadv + log_p_y
                total_class_loss += class_loss

                # 7. 计算边界框L1损失（取负，最大化偏差）
                bbox_l1_loss = self.bbox_l1_criterion(matched_pred_box, current_gt_box)
                total_bbox_l1_loss -= (bbox_l1_loss / (4 * self.image_resolution))  # 归一化

                # 8. 计算GIoU损失（对抗损失，直接用GIoU而非1-GIoU）
                giou = generalized_box_iou(matched_pred_box, current_gt_box)  # [1]
                giou_loss = giou.sum()
                total_giou_loss += giou_loss

        # 计算平均损失（除以总真实框数量）
        if total_gt_count > 0:
            class_loss_avg = total_class_loss / total_gt_count
            bbox_l1_loss_avg = total_bbox_l1_loss / total_gt_count
            giou_loss_avg = total_giou_loss / total_gt_count
        else:
            # 无GT时的兜底（避免除0）
            class_loss_avg = torch.tensor(0.0, device=device)
            bbox_l1_loss_avg = torch.tensor(0.0, device=device)
            giou_loss_avg = torch.tensor(0.0, device=device)

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
    


class MaskedL1L2Loss(nn.Module):
    def __init__(self, loss_type: str = "l2", reduction: str = "mean"):
        """
        带掩码的 L1/L2 损失函数（支持 NumPy 掩码输入）
        :param loss_type: 损失类型，可选 "l1"（MAE）或 "l2"（MSE）
        :param reduction: 损失聚合方式，可选 "mean"（均值）、"sum"（求和）、"none"（逐元素）
        """
        super(MaskedL1L2Loss, self).__init__()
        if loss_type not in ["l1", "l2"]:
            raise ValueError(f"loss_type 必须是 'l1' 或 'l2'，当前为 {loss_type}")
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"reduction 必须是 'mean'/'sum'/'none'，当前为 {reduction}")
        
        self.loss_type = loss_type
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask) -> torch.Tensor:
        """
        前向传播：计算掩码区域内的 L1/L2 损失
        :param pred: 预测值张量，形状为 [B, C, H, W] 或 [B, H, W]（批量大小 + 特征维度）
        :param target: 真实值张量，形状需与 pred 一致
        :param mask: 掩码（NumPy 数组/张量），shape=(B, H, W)，支持布尔型/0-1数值型
        :return: 掩码区域内的损失值
        """
        # 1. 校验预测值与真实值形状匹配
        if pred.shape != target.shape:
            raise RuntimeError(f"预测值形状 {pred.shape} 与真实值形状 {target.shape} 不匹配")
        
        # 2. 处理掩码：NumPy 转张量 + 布尔型转数值型 + 扩展维度（适配通道）
        mask_tensor = self._process_mask(mask, pred)

        # 3. 计算逐元素的 L1 或 L2 误差
        if self.loss_type == "l1":
            element_wise_loss = F.l1_loss(pred, target, reduction="none")  # L1 误差（MAE）
        else:  # l2
            element_wise_loss = F.mse_loss(pred, target, reduction="none")  # L2 误差（MSE）

        # 4. 应用掩码：仅保留掩码为 1 的区域的误差
        masked_loss = element_wise_loss * mask_tensor

        # 5. 根据聚合方式计算最终损失
        if self.reduction == "none":
            return masked_loss  # 返回逐元素的掩码损失
        elif self.reduction == "sum":
            return masked_loss.sum()  # 掩码区域内的误差总和
        else:  # mean
            # 均值：有效误差之和 / 掩码中 1 的数量（避免除以 0）
            mask_sum = mask_tensor.sum()
            if mask_sum == 0:
                return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
            return masked_loss.sum() / mask_sum

    def _process_mask(self, mask, pred: torch.Tensor) -> torch.Tensor:
        """
        处理掩码：NumPy 转张量 + 布尔型转数值型 + 扩展通道维度
        :param mask: 输入掩码（NumPy 数组/张量）
        :param pred: 预测值张量（用于匹配设备、数据类型、形状）
        :return: 处理后的掩码张量，形状与 pred 一致
        """
        # Step 1: NumPy 数组转 PyTorch 张量
        if isinstance(mask, np.ndarray):
            # 布尔型 NumPy 数组自动转为 0/1 浮点型
            mask_tensor = torch.from_numpy(mask.astype(np.float32))
        elif isinstance(mask, torch.Tensor):
            mask_tensor = mask.float()  # 张量转为浮点型（布尔型张量会自动转 0/1）
        else:
            raise TypeError(f"掩码类型 {type(mask)} 不支持，仅支持 NumPy 数组或 PyTorch 张量")
        
        # Step 2: 匹配设备和数据类型
        mask_tensor = mask_tensor.to(device=pred.device, dtype=pred.dtype)
        
        # Step 3: 扩展通道维度（若 pred 有通道维度 [B, C, H, W]，mask 是 [B, H, W]）
        if len(pred.shape) == 4 and len(mask_tensor.shape) == 3:
            # 从 [B, H, W] 扩展为 [B, 1, H, W]，适配通道维度广播
            mask_tensor = mask_tensor.unsqueeze(1)
        
        # Step 4: 校验形状兼容性
        if not torch.broadcast_shapes(pred.shape, mask_tensor.shape) == pred.shape:
            raise RuntimeError(
                f"掩码形状 {mask_tensor.shape} 与预测值形状 {pred.shape} 不兼容！\n"
                f"掩码输入形状：{np.shape(mask)}（原始）→ {mask_tensor.shape}（处理后）"
            )
        
        return mask_tensor


def tensor2picture(tensor, save_path, norm_status=False, use_opencv=False):
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
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float32)
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

    min_val = img_np.min()
    max_val = img_np.max()

    if norm_status:
        # 防止除零（若所有值相同）
        if max_val == min_val:
            img_np = np.zeros_like(img_np, dtype=np.uint8)
        else:
            # 归一化到 [0, 1] 再映射到 [0, 255]
            img_np = ((img_np - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else :
        img_np=(img_np*255).astype(np.uint8)
        # 超过0-255的像素值，截取、
        img_np[img_np>255]=255
    # 
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






# class CustomFolderDataset(Dataset):
#     def __init__(self, root_dir, transform=None, 
#                  target_image_name_list=[],
#                  img_extensions=['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.PNG', '.JPEG','pt']):
#         """
#         自定义数据集：按文件夹分组加载图片（递归查找子文件夹）
#         :param root_dir: 根目录（子文件夹为不同的分组）
#         :param transform: 单张图片的预处理变换
#         :param img_extensions: 支持的图片格式
#         """
#         self.root_dir = root_dir
#         self.transform = transform
#         self.img_extensions = img_extensions
#         # 按文件夹分组存储数据：每个元素为 (文件夹名称, 该文件夹下所有图片路径列表)
#         self.folder_data = self._get_folder_img_paths()
#         self.target_image_name_list = target_image_name_list
#         if len(target_image_name_list)<=0:
#             # 报错
#             raise ValueError(f"请指定目标图片名称列表！")

#     def _get_folder_img_paths(self):
#         """递归遍历目录，按文件夹分组获取图片路径"""
#         folder_dict = {}  # key: 文件夹路径, value: 该文件夹下的图片路径列表
        
#         # 递归遍历所有子目录
#         for root, dirs, files in os.walk(self.root_dir):
#             # 过滤当前文件夹下的图片
#             img_paths = []
#             for file in files:
#                 if any(file.endswith(ext) for ext in self.img_extensions):
#                     img_paths.append(os.path.join(root, file))
#             # 仅保留包含图片的文件夹
#             if img_paths:
#                 # 用文件夹的相对路径作为名称（也可用basename）
#                 folder_name = os.path.relpath(root, self.root_dir)
#                 #
                
#                 folder_dict[folder_name] = img_paths
        
#         if not folder_dict:
#             raise ValueError(f"目录 {self.root_dir} 下未找到包含图片的文件夹！")
        
#         # 转换为列表，便于按索引访问：[(folder_name, img_paths), ...]
#         folder_data = [(name, paths) for name, paths in folder_dict.items()]
#         return folder_data

#     def __len__(self):
#         """返回文件夹的总数（而非图片总数）"""
#         return len(self.folder_data)

#     def __getitem__(self, idx):
#         """
#         按索引返回单个文件夹的所有图片数据
#         :param idx: 文件夹索引
#         :return: 
#             folder_name: 文件夹名称（str）
#             img_tensors: 该文件夹下所有图片的张量列表 (list[torch.Tensor])
#             img_names: 该文件夹下所有图片的名称列表 (list[str])
#         """
#         # 获取当前文件夹的名称和图片路径
#         folder_name, img_paths = self.folder_data[idx]
        
#         img_tensors = []  # 存储该文件夹下所有图片的张量
#         img_names = []    # 存储该文件夹下所有图片的名称（不含路径）
        
#         for img_path in img_paths:
#             image_name_1=os.path.basename(img_path)
#                 # 去除后缀
#             image_name=image_name_1.split('.')[0]
#             if image_name_1 in self.target_image_name_list:
#                 # 根据后缀，读取文件，如果是pt文件，则加载为张量

#                 try:
#                     if img_path.endswith('.pt'):
#                         img = torch.load(img_path)
#                     else:    
#                         # 读取图片并转为RGB
#                         img = Image.open(img_path).convert('RGB')
#                         # 应用预处理变换
#                         if self.transform:
#                             img = self.transform(img)
#                 except Exception as e:
#                     raise RuntimeError(f"加载图片失败 {img_path}：{e}")
                

                
#                 # 收集张量和图片名称
#                 img_tensors.append(img)

#                 img_names.append(os.path.basename(image_name_1))  # 仅保留图片文件名，去除后缀
#                 # img_names.append(os.path.basename(img_path))
#             else:
#                 continue
        
#         return folder_name, img_tensors, img_names

#     def get_folder_by_name(self, folder_name):
#         """按文件夹名称查找并返回该文件夹的图片数据（非索引方式）"""
#         for idx, (name, paths) in enumerate(self.folder_data):
#             if name == folder_name:
#                 return self.__getitem__(idx)
#         raise ValueError(f"未找到文件夹：{folder_name}")


class CustomFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None, 
                 target_image_name_list=[],
                 img_extensions=['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.PNG', '.JPEG','pt']):
        """
        自定义数据集：按文件夹分组加载图片（递归查找子文件夹）
        仅保留包含全部目标图片的文件夹
        :param root_dir: 根目录（子文件夹为不同的分组）
        :param transform: 单张图片的预处理变换
        :param target_image_name_list: 必须包含的目标图片名称列表（带后缀，如 ['1.jpg', '2.pt']）
        :param img_extensions: 支持的图片格式
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_extensions = img_extensions
        self.target_image_name_list = target_image_name_list
        
        # 验证目标列表非空
        if len(target_image_name_list) <= 0:
            raise ValueError(f"请指定目标图片名称列表！")
        
        # 先获取所有包含图片的文件夹，再过滤出包含全部目标文件的文件夹
        all_folder_data = self._get_all_folder_img_paths()
        self.folder_data = self._filter_complete_folders(all_folder_data)
        
        if not self.folder_data:
            raise ValueError(f"目录 {self.root_dir} 下未找到包含全部目标文件的文件夹！")

    def _get_all_folder_img_paths(self):
        """递归遍历目录，获取所有包含图片的文件夹路径（原始逻辑，无过滤）"""
        folder_dict = {}  # key: 文件夹路径, value: 该文件夹下的图片路径列表
        
        # 递归遍历所有子目录
        for root, dirs, files in os.walk(self.root_dir):
            # 过滤当前文件夹下的图片
            img_paths = []
            for file in files:
                if any(file.endswith(ext) for ext in self.img_extensions):
                    img_paths.append(os.path.join(root, file))
            # 仅保留包含图片的文件夹
            if img_paths:
                folder_name = os.path.relpath(root, self.root_dir)
                folder_dict[folder_name] = img_paths
        
        if not folder_dict:
            raise ValueError(f"目录 {self.root_dir} 下未找到包含图片的文件夹！")
        
        # 转换为列表：[(folder_name, img_paths), ...]
        folder_data = [(name, paths) for name, paths in folder_dict.items()]
        return folder_data

    def _filter_complete_folders(self, all_folder_data):
        """过滤出包含全部目标文件的文件夹"""
        complete_folders = []
        
        # 转换目标列表为集合，方便快速查找
        target_set = set(self.target_image_name_list)
        
        for folder_name, img_paths in all_folder_data:
            # 获取当前文件夹下的所有图片文件名（带后缀）
            folder_file_names = set([os.path.basename(path) for path in img_paths])
            
            # 检查是否包含所有目标文件
            if target_set.issubset(folder_file_names):
                complete_folders.append((folder_name, img_paths))
        
        return complete_folders

    def __len__(self):
        """返回符合条件的文件夹总数（而非图片总数）"""
        return len(self.folder_data)

    def __getitem__(self, idx):
        """
        按索引返回单个文件夹的所有图片数据
        :param idx: 文件夹索引
        :return: 
            folder_name: 文件夹名称（str）
            img_tensors: 该文件夹下所有目标图片的张量列表 (list[torch.Tensor])
            img_names: 该文件夹下所有目标图片的名称列表 (list[str])
        """
        # 获取当前文件夹的名称和图片路径
        folder_name, img_paths = self.folder_data[idx]
        
        img_tensors = []  # 存储该文件夹下所有图片的张量
        img_names = []    # 存储该文件夹下所有图片的名称（含后缀）
        
        # 按目标列表顺序加载，保证顺序一致性
        for target_name in self.target_image_name_list:
            # 找到对应文件的路径
            target_path = None
            for img_path in img_paths:
                if os.path.basename(img_path) == target_name:
                    target_path = img_path
                    break
            
            if not target_path:
                raise RuntimeError(f"理论上不会出现此错误：文件夹 {folder_name} 缺少目标文件 {target_name}")
            
            try:
                if target_path.endswith('.pt'):
                    img = torch.load(target_path)
                else:    
                    # 读取图片并转为RGB
                    img = Image.open(target_path).convert('RGB')
                    # 应用预处理变换
                    if self.transform:
                        img = self.transform(img)
            except Exception as e:
                raise RuntimeError(f"加载图片失败 {target_path}：{e}")
            
            # 收集张量和图片名称
            img_tensors.append(img)
            img_names.append(target_name)
        
        return folder_name, img_tensors, img_names

    def get_folder_by_name(self, folder_name):
        """按文件夹名称查找并返回该文件夹的图片数据（非索引方式）"""
        for idx, (name, paths) in enumerate(self.folder_data):
            if name == folder_name:
                return self.__getitem__(idx)
        raise ValueError(f"未找到文件夹：{folder_name}")





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
def canny_with_mask_invert(background_imag,
                                masks=None,
                                canny_low=50, 
                                canny_high=240,
                                blur_status=True,
                                blur_k_size=5,
                                with_mask_edge=True,
                                with_content_canny=True,
                                ):
    
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
        # 创建空画布绘制 mask 边界
        mask_edge = np.zeros_like(selected_mask)


        if blur_status:
            img_np = cv2.GaussianBlur(img_np, (blur_k_size, blur_k_size), 0)

        gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # ===== 步骤4：图像 Canny 边缘检测 =====
        if with_content_canny:
            # 对图像进行模糊处理

            canny_edges = cv2.Canny(gray_img, canny_low, canny_high)  # 图像边缘=255，背景=0
        else :
            canny_edges = np.zeros_like(selected_mask)

        if with_mask_edge:
            # 用轮廓检测提取 mask 边界
            
            contours, _ = cv2.findContours(
                gray_img, # selected_mask
                cv2.RETR_EXTERNAL,  # 只提取最外层轮廓
                cv2.CHAIN_APPROX_SIMPLE  # 压缩轮廓点
            )

            cv2.drawContours(
                mask_edge, 
                contours, 
                -1,  # 绘制所有轮廓
                255,  # 轮廓颜色（白色）
                1  # 轮廓线宽度（可根据需求调整，如1/3）
            )  # mask_edge: 边界=255，其余=0

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


def map_coco_to_yolo_labels(raw_labels_tensor: torch.Tensor) -> torch.Tensor:
    """
    将COCO官方ID的Tensor标签映射为YOLO ID的Tensor标签
    核心特性：
    1. 保持输入Tensor的设备（CPU/GPU）一致
    2. 不中断计算图的梯度传播
    3. 无效COCO ID（如12、99）映射为-1
    4. 自动处理索引越界问题
    
    :param raw_labels_tensor: 输入的COCO ID Tensor，支持任意形状，dtype需为long
    :return: 映射后的YOLO ID Tensor，设备/梯度属性与输入完全一致
    """
    # 1. 定义COCO -> YOLO 核心映射字典
    coco2yolo_class_mapping = {
        1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
        11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19,
        22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29,
        35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39,
        46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
        56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59,
        67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69,
        80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79
    }
    
    # 2. 校验输入类型（避免非long型tensor导致索引错误）
    if raw_labels_tensor.dtype != torch.long:
        raise TypeError(f"输入tensor的dtype必须是torch.long，当前为{raw_labels_tensor.dtype}")
    
    # 3. 获取输入设备，确保映射表与输入同设备
    device = raw_labels_tensor.device
    
    # 4. 构建Tensor版映射表（默认值-1，覆盖COCO ID 0-90）
    max_coco_id = 90
    mapping_tensor = torch.full((max_coco_id + 1,), -1, dtype=torch.long, device=device)
    for coco_id, yolo_id in coco2yolo_class_mapping.items():
        mapping_tensor[coco_id] = yolo_id
    
    # 5. 核心映射：限制索引范围+tensor索引（保留梯度/设备）
    clamped_labels = raw_labels_tensor.clamp(0, max_coco_id)  # 防止越界
    mapped_labels = mapping_tensor[clamped_labels]
    
    # 6. 继承输入的梯度属性（若输入有requires_grad=True，映射结果也保留）
    mapped_labels.requires_grad = raw_labels_tensor.requires_grad
    
    return mapped_labels




def calculate_box_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """
    计算两个检测框的IoU（Intersection over Union）
    修复：兼容BFloat16张量类型转换
    输入框格式：[x1, y1, x2, y2]（像素坐标或归一化坐标均可，需保持一致）
    """
    # 核心修复：处理BFloat16类型（先转float32再转numpy）
    def safe_tensor_to_numpy(tensor):
        # 若为BFloat16，先转换为float32
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)
        # 若为CUDA tensor，先转CPU
        if tensor.is_cuda:
            tensor = tensor.cpu()
        # detach()    
        tensor = tensor.detach()
        # 
        return tensor.numpy()
    
    # 安全转换为numpy（兼容BFloat16/CUDA/CPU）
    box1_np = safe_tensor_to_numpy(box1)
    box2_np = safe_tensor_to_numpy(box2)
    
    # 计算交集坐标
    x1 = max(box1_np[0], box2_np[0])
    y1 = max(box1_np[1], box2_np[1])
    x2 = min(box1_np[2], box2_np[2])
    y2 = min(box1_np[3], box2_np[3])
    
    # 计算交集面积
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0
    
    # 计算两个框的面积
    box1_area = (box1_np[2] - box1_np[0]) * (box1_np[3] - box1_np[1])
    box2_area = (box2_np[2] - box2_np[0]) * (box2_np[3] - box2_np[1])
    
    # 计算IoU
    iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
    return iou

# def count_matched_results(results_dict: dict, 
#                                    ref_result: dict, 
#                                    iou_threshold: float = 0.5,
#                                    conf_threshold: float = 0.5) -> tuple:
#     """
#     统计每个模型的batch整体匹配准确率（所有有效batch的匹配比例）
#     核心逻辑：
#         1. 遍历每个模型的每个result，逐batch判断是否匹配参考batch的第一个框；
#         2. 累计该模型的「总有效batch数」和「匹配成功batch数」；
#         3. 准确率 = 匹配成功batch数 / 总有效batch数（无数据时为0.0）。
    
#     单batch匹配条件（需同时满足）：
#         - 该batch有检测框；
#         - 存在至少一个框与参考框的IoU ≥ iou_threshold；
#         - 该框置信度 ≥ conf_threshold；
#         - 该框label与参考框完全一致。

#     参数：
#         results_dict: 模型检测结果字典，键=模型名，值=该模型的result列表（每个result含boxes/scores/labels）；
#         ref_result: 参考result字典（每个batch取第一个框作为匹配目标）；
#         iou_threshold: IoU匹配阈值（0-1），默认0.5；
#         conf_threshold: 置信度阈值（0-1），默认0.5。

#     返回：
#         (simple_accuracy, detailed_stats): 
#             - simple_accuracy: 简洁字典 {模型名: batch匹配准确率}；
#             - detailed_stats: 详细统计字典 {模型名: {total_batches: 总有效batch数, matched_batches: 匹配成功数, accuracy: 准确率}}。
#     """
#     # 基础校验：参考result的必填字段和有效batch数
#     required_keys = ['boxes', 'scores', 'labels']
#     for key in required_keys:
#         if key not in ref_result:
#             raise ValueError(f"参考result缺少关键字段：{key}")
    
#     ref_batch_num = len(ref_result['labels'])
#     if ref_batch_num == 0:
#         raise ValueError("参考result中无有效batch数据")

#     # 提取参考每个batch的第一个框特征（None表示参考batch为空，不统计）
#     ref_batch_features = []
#     for batch_idx in range(ref_batch_num):
#         if (len(ref_result['labels'][batch_idx]) == 0 or 
#             len(ref_result['boxes'][batch_idx]) == 0 or 
#             len(ref_result['scores'][batch_idx]) == 0):
#             ref_batch_features.append(None)
#             continue
#         ref_batch_features.append({
#             'box': ref_result['boxes'][batch_idx][0],
#             'label': ref_result['labels'][batch_idx][0]
#         })

#     # 初始化详细统计字典
#     detailed_stats = {}
#     for model_name in results_dict.keys():
#         detailed_stats[model_name] = {
#             "total_batches": 0,   # 该模型的总有效batch数
#             "matched_batches": 0, # 匹配成功的batch数
#             "accuracy": 0.0       # 最终准确率
#         }

#     # 遍历每个模型，逐result、逐batch统计
#     for model_name, results_list in results_dict.items():
#         total = 0
#         matched = 0

#         for res_idx, res_dict in enumerate(results_list):
#             # 跳过batch数量与参考不一致的result（无有效batch）
#             curr_batch_num = len(res_dict['labels'])
#             if curr_batch_num != ref_batch_num:
#                 print(f"警告：模型[{model_name}] Result[{res_idx}]的batch数({curr_batch_num})与参考({ref_batch_num})不一致，跳过")
#                 continue

#             # 遍历该result的每个batch，独立判断匹配状态
#             for batch_idx in range(ref_batch_num):
#                 ref_feat = ref_batch_features[batch_idx]
#                 if ref_feat is None:  # 参考batch为空，不统计
#                     continue

#                 # 累计总有效batch数
#                 total += 1

#                 # 获取当前batch的检测结果
#                 curr_labels = res_dict['labels'][batch_idx]
#                 curr_boxes = res_dict['boxes'][batch_idx]
#                 curr_scores = res_dict['scores'][batch_idx]

#                 # 情况1：当前batch无检测框 → 匹配失败
#                 if (len(curr_labels) == 0 or len(curr_boxes) == 0 or len(curr_scores) == 0):
#                     continue

#                 # 情况2：检查是否有符合条件的检测框
#                 batch_matched = False
#                 for box_idx in range(len(curr_labels)):
#                     curr_box = curr_boxes[box_idx]
#                     curr_label = curr_labels[box_idx]
#                     curr_score = curr_scores[box_idx]

#                     # 核心匹配条件
#                     label_match = torch.equal(curr_label, ref_feat['label']) if torch.is_tensor(curr_label) else (curr_label == ref_feat['label'])
#                     conf_match = curr_score >= conf_threshold
#                     iou_match = calculate_box_iou(curr_box, ref_feat['box']) >= iou_threshold

#                     if label_match and conf_match and iou_match:
#                         batch_matched = True
#                         break

#                 # 匹配成功则累计
#                 if batch_matched:
#                     matched += 1

#         # 更新该模型的统计结果
#         detailed_stats[model_name]["total_batches"] = total
#         detailed_stats[model_name]["matched_batches"] = matched
#         detailed_stats[model_name]["accuracy"] = round(matched / total if total > 0 else 0.0, 4)

#     # 生成简洁的准确率字典（仅返回模型名+准确率）
#     simple_accuracy = {name: stats["accuracy"] for name, stats in detailed_stats.items()}

#     return simple_accuracy, detailed_stats


def count_matched_results(model_results_dict: dict, 
                                   ref_result: dict, 
                                   iou_threshold: float = 0.5,
                                   conf_threshold: float = 0.5) -> dict:
    """
    统计不同模型的batch匹配准确率（每个模型对应单个result字典）
    匹配规则（单batch）：
    1. 对位匹配：result的第i个batch ↔ ref_result的第i个batch
    2. 单batch匹配条件（需同时满足）：
       - 该batch中存在至少一个检测框
       - 检测框与参考batch第一个框的IoU ≥ iou_threshold
       - 检测框置信度(scores) ≥ conf_threshold
       - 检测框label与参考框完全一致
    
    参数：
        model_results_dict: 模型检测结果字典，键=模型名称，值=该模型的单个检测结果字典（含boxes/scores/labels）
        ref_result: 参考result字典（每个batch取第一个检测框作为匹配目标）
        iou_threshold: IoU匹配阈值（0-1），默认0.5
        conf_threshold: 置信度匹配阈值（0-1），默认0.5
    
    返回：
        model_accuracy_dict: 各模型batch匹配准确率字典，结构：{模型名: 准确率（0-1）}
    """
    # 1. 基础校验
    required_keys = ['boxes', 'scores', 'labels']
    for key in required_keys:
        if key not in ref_result:
            raise ValueError(f"参考result缺少关键字段：{key}")
    
    ref_batch_num = len(ref_result['labels'])
    if ref_batch_num == 0:
        raise ValueError("参考result中无有效batch数据")
    
    # 2. 提取参考result的每个batch的第一个框特征
    ref_batch_features = []
    for batch_idx in range(ref_batch_num):
        # 跳过空的参考batch（后续该batch视为"无需匹配"）
        if (len(ref_result['labels'][batch_idx]) == 0 or 
            len(ref_result['boxes'][batch_idx]) == 0 or 
            len(ref_result['scores'][batch_idx]) == 0):
            ref_batch_features.append(None)
            continue
        
        ref_box = ref_result['boxes'][batch_idx][0]
        ref_label = ref_result['labels'][batch_idx][0]
        ref_batch_features.append({
            'box': ref_box,
            'label': ref_label
        })
    
    # 3. 初始化模型准确率字典
    model_accuracy_dict = {}
    
    # 4. 遍历每个模型（每个模型对应单个result）
    for model_name, res_dict in model_results_dict.items():
        # 校验当前模型result的batch数量与参考一致
        curr_batch_num = len(res_dict['labels'])
        if curr_batch_num != ref_batch_num:
            print(f"警告：模型[{model_name}]的batch数量({curr_batch_num})与参考({ref_batch_num})不一致，准确率记为0.0")
            model_accuracy_dict[model_name] = 0.0
            continue
        
        total_valid_batches = 0  # 该模型的总有效batch数
        matched_batches = 0      # 该模型匹配成功的batch数
        
        # 遍历该result的每个batch，逐一批配
        for batch_idx in range(ref_batch_num):
            ref_feat = ref_batch_features[batch_idx]
            
            # 情况1：参考batch为空 → 无需统计该batch
            if ref_feat is None:
                continue
            
            # 该batch计入总有效数
            total_valid_batches += 1
            
            # 情况2：当前batch为空 → 匹配失败
            curr_labels = res_dict['labels'][batch_idx]
            curr_boxes = res_dict['boxes'][batch_idx]
            curr_scores = res_dict['scores'][batch_idx]
            if (len(curr_labels) == 0 or 
                len(curr_boxes) == 0 or 
                len(curr_scores) == 0):
                continue  # 匹配失败，不计数
            
            # 情况3：检查当前batch是否有符合条件的检测框
            batch_matched = False
            for box_idx in range(len(curr_labels)):
                # 当前检测框特征
                curr_box = curr_boxes[box_idx]
                curr_label = curr_labels[box_idx]
                curr_score = curr_scores[box_idx]
                
                # 核心匹配条件
                label_match = torch.equal(curr_label, ref_feat['label']) if torch.is_tensor(curr_label) else (curr_label == ref_feat['label'])
                conf_match = curr_score >= conf_threshold
                iou_match = calculate_box_iou(curr_box, ref_feat['box']) >= iou_threshold
                
                if label_match and conf_match and iou_match:
                    batch_matched = True
                    break  # 找到一个匹配框即可
            
            # 该batch匹配成功 → 计数+1
            if batch_matched:
                matched_batches += 1
        
        # 计算该模型的准确率（避免除以0）
        if total_valid_batches == 0:
            model_accuracy = 0.0
        else:
            model_accuracy = round(matched_batches / total_valid_batches, 4)
        
        # 存入结果字典
        model_accuracy_dict[model_name] = model_accuracy
    
    return model_accuracy_dict




def resize_images(
    images: torch.Tensor,
    target_size
) -> Tuple[torch.Tensor, float, float]:
    if len(images.shape) != 4:
        raise ValueError(f"输入张量需为4维 (B,C,H,W)，当前shape: {images.shape}")

    
    # 2. 获取原尺寸和目标尺寸
    B, C, orig_h, orig_w = images.shape
     

    
    # 4. 逐Batch缩放（支持批量处理，保持维度一致）
    resized_images = []
    target_scale_list=[]
    for b in range(B):
        target_h, target_w = target_size[b]
        # 3. 计算缩放比例（非等比例，宽/高独立计算）
        scale_w = target_w / orig_w  # 宽度缩放比例
        scale_h = target_h / orig_h  # 高度缩放比例
        img_single = images[b:b+1]  # 取单Batch: (1, C, orig_h, orig_w)
        # 核心：非等比例拉伸到目标尺寸（antialias=True优化缩放质量）
        img_resized = torchvision.transforms.functional.resize(
            img_single, 
            size=(target_h, target_w),  # 直接拉伸适配目标尺寸
            antialias=True
        )
        resized_images.append(img_resized)
        target_scale_list.append((scale_w, scale_h))
    
    # 5. 合并Batch并返回结果
    resized_images = torch.cat(resized_images, dim=0)
    return resized_images,target_scale_list


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

def center_scale_image_tensor(img_tensor, scale=0.5, pad_value=0):
    """
    对张量格式的图像进行中心等比例缩放，空白区域填充指定值（默认0），输出尺寸与输入一致
    
    Args:
        img_tensor (torch.Tensor): 输入图像张量，格式为 [C, H, W] 或 [B, C, H, W]
        scale (float): 缩放比例（0 < scale ≤ 1 缩小，scale > 1 放大）
        pad_value (int/float): 填充值，默认0
    
    Returns:
        torch.Tensor: 缩放后图像张量，尺寸与输入完全一致
    """
    # 检查输入维度，统一处理为4维 [B, C, H, W]
    is_3d = len(img_tensor.shape) == 3
    if is_3d:
        img_tensor = img_tensor.unsqueeze(0)  # [C, H, W] → [1, C, H, W]
    
    B, C, H, W = img_tensor.shape
    
    # 1. 计算缩放后的尺寸（保持宽高比）
    new_h = int(H * scale)
    new_w = int(W * scale)
    # 确保缩放后尺寸至少为1（避免scale过小导致尺寸为0）
    new_h = max(1, new_h)
    new_w = max(1, new_w)
    
    # 2. 对图像进行缩放（使用bilinear插值，保持边缘平滑）
    scaled_img = F.interpolate(
        img_tensor, 
        size=(new_h, new_w), 
        mode='bilinear', 
        align_corners=False
    )
    
    # 3. 计算缩放后图像在输出张量中的中心位置
    start_h = (H - new_h) // 2
    start_w = (W - new_w) // 2
    end_h = start_h + new_h
    end_w = start_w + new_w
    
    # 4. 创建输出张量，初始填充pad_value
    output = torch.full_like(img_tensor, pad_value, dtype=img_tensor.dtype)
    
    # 5. 将缩放后的图像放入中心位置
    output[:, :, start_h:end_h, start_w:end_w] = scaled_img
    
    # 恢复原始维度（如果输入是3维）
    if is_3d:
        output = output.squeeze(0)
    
    return output

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



def load_yaml_config(config_path: str) -> Dict:
    """
    加载YAML配置文件
    
    参数:
        config_path: YAML配置文件路径
    
    返回:
        解析后的配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config




def modify_labels_and_scores(results_dict: dict, ref_label1: int, ref_label2: int = 0) -> dict:
    """
    根据目标标签与参考标签1的关系替换标签，并同步更新scores_vector：
    - 若目标标签≠ref_label1：将标签替换为ref_label1，对应概率向量的ref_label1位置设为1，其余为0
    - 若目标标签=ref_label1：将标签替换为ref_label2，对应概率向量的ref_label2位置设为1，其余为0
    
    Args:
        results_dict (dict): 输入字典，需包含'labels'和'scores_vector'键：
                            - 'labels': list of tensor，每个tensor为[目标数]的标签索引
                            - 'scores_vector': list of tensor，每个tensor为[目标数, 类别数]的概率向量
        ref_label1 (int): 第一参考标签（判断基准，必须是整数类别索引）
        ref_label2 (int, optional): 第二参考标签，默认为0
    
    Returns:
        dict: 修改后的新字典，其他键保持不变，'labels'和'scores_vector'被更新
    
    Raises:
        KeyError: 输入字典缺少'labels'或'scores_vector'键
        TypeError: 参考标签不是整数，或labels/scores_vector格式不符合要求
        IndexError: 参考标签超出scores_vector的类别索引范围
    """
    # 1. 检查输入字典的必要键
    required_keys = ['labels', 'scores_vector']
    for key in required_keys:
        if key not in results_dict:
            raise KeyError(f"输入字典必须包含'{key}'键")
    
    # 2. 检查参考标签类型
    if not isinstance(ref_label1, int):
        raise TypeError(f"第一参考标签必须是整数类别索引，当前类型：{type(ref_label1)}")
    if not isinstance(ref_label2, int):
        raise TypeError(f"第二参考标签必须是整数类别索引，当前类型：{type(ref_label2)}")
    
    # 3. 深拷贝原字典，避免修改原数据
    new_results = copy.deepcopy(results_dict)
    original_labels = new_results['labels']
    original_scores_vector = new_results['scores_vector']
    
    # 4. 检查labels和scores_vector的批次数量是否一致
    if len(original_labels) != len(original_scores_vector):
        raise ValueError(f"labels批次数量({len(original_labels)})与scores_vector批次数量({len(original_scores_vector)})不匹配")
    
    modified_labels = []
    modified_scores_vector = []
    
    # 5. 逐批次处理labels和scores_vector
    for batch_idx, (batch_labels, batch_scores) in enumerate(zip(original_labels, original_scores_vector)):
        # 检查当前批次的张量格式
        if not isinstance(batch_labels, torch.Tensor) or batch_labels.dim() != 1:
            raise TypeError(f"批次{batch_idx}的labels必须是1维张量，当前形状：{batch_labels.shape}")
        if not isinstance(batch_scores, torch.Tensor) or batch_scores.dim() != 2:
            raise TypeError(f"批次{batch_idx}的scores_vector必须是2维张量([目标数, 类别数])，当前形状：{batch_scores.shape}")
        
        num_targets, num_classes = batch_scores.shape
        # 检查两个参考标签是否在类别索引范围内
        for label, label_name in [(ref_label1, '第一'), (ref_label2, '第二')]:
            if label < 0 or label >= num_classes:
                raise IndexError(f"批次{batch_idx}的类别数为{num_classes}，{label_name}参考标签{label}超出索引范围[0, {num_classes-1}]")
        
        # ---- 处理labels：根据与ref_label1的关系替换 ----
        new_batch_labels = batch_labels.clone()
        # 掩码1：标签等于ref_label1的目标
        eq_ref1_mask = (new_batch_labels == ref_label1)
        # 掩码2：标签不等于ref_label1的目标
        ne_ref1_mask = ~eq_ref1_mask
        
        # 不等于ref_label1 → 替换为ref_label1
        new_batch_labels[ne_ref1_mask] = ref_label1
        # 等于ref_label1 → 替换为ref_label2
        new_batch_labels[eq_ref1_mask] = ref_label2
        modified_labels.append(new_batch_labels)
        
        # ---- 处理scores_vector：同步更新概率向量 ----
        new_batch_scores = torch.zeros_like(batch_scores)  # 先初始化全0
        # 不等于ref_label1的目标：ref_label1位置设为1
        new_batch_scores[ne_ref1_mask, ref_label1] = 1.0
        # 等于ref_label1的目标：ref_label2位置设为1
        new_batch_scores[eq_ref1_mask, ref_label2] = 1.0
        
        modified_scores_vector.append(new_batch_scores)
    
    # 6. 更新新字典的labels和scores_vector
    new_results['labels'] = modified_labels
    new_results['scores_vector'] = modified_scores_vector
    
    return new_results



def filter_max_box_per_batch(result: dict,class_names) -> dict:
    """
    筛选每个batch中面积最大的检测框，返回与输入格式一致的result_gt字典
    
    Args:
        result_gt: 原始gt字典，包含以下key（值均为列表，每个元素对应一个batch的Tensor）:
            - labels: 列表，每个元素是 [N,] Tensor（N为该batch的框数量，存储类别标签）
            - boxes: 列表，每个元素是 [N, 4] Tensor（框坐标，格式支持 xyxy/xywh）
            - scores: 列表，每个元素是 [N,] Tensor（置信度分数）
            - scores_vector: 列表，每个元素是 [N, D] Tensor（D为分数向量维度）
    
    Returns:
        filtered_gt: 筛选后的字典，结构与输入一致，每个batch仅保留面积最大的框
                    若某batch无框（Tensor为空），则保留空Tensor
                    
    注意：
        - boxes坐标格式支持 xyxy（左上x, 左上y, 右下x, 右下y）或 xywh（左上x, 左上y, 宽, 高）
        - 自动适配Tensor设备（CPU/GPU），保持与输入一致
    """
    # 初始化输出字典，与输入格式对齐
    filtered_gt = {
        'labels': [],
        'boxes': [],
        'scores': [],
        'scores_vector': []
    }
    filtered_class_names = []
    # 遍历每个batch的信息（按列表索引对齐）
    batch_num = len(result['labels'])
    for b_idx in range(batch_num):
        # 取出当前batch的所有数据（处理空值，避免索引报错）
        labels = result['labels'][b_idx] if b_idx < len(result['labels']) else torch.tensor([], dtype=torch.int64)
        boxes = result['boxes'][b_idx] if b_idx < len(result['boxes']) else torch.tensor([], dtype=torch.float32)
        scores = result['scores'][b_idx] if b_idx < len(result['scores']) else torch.tensor([], dtype=torch.float32)
        scores_vector = result['scores_vector'][b_idx] if b_idx < len(result['scores_vector']) else torch.tensor([], dtype=torch.float32)
        class_name_temp=class_names[b_idx] if b_idx < len(class_names) else ''

        # 处理空框场景：当前batch无检测框，直接添加空Tensor
        if boxes.numel() == 0:
            filtered_gt['labels'].append(labels)
            filtered_gt['boxes'].append(boxes)
            filtered_gt['scores'].append(scores)
            filtered_gt['scores_vector'].append(scores_vector)
            filtered_class_names.append(class_name_temp)
            continue
        
        # ========== 核心：计算每个框的面积，筛选最大面积的框 ==========
        # 统一转换为 xyxy 格式计算面积（兼容xyxy/xywh输入）
        if boxes.shape[1] == 4:
            # 区分 xyxy 和 xywh：xyxy的宽高为 (x2-x1, y2-y1)；xywh的宽高为 (w, h)
            if (boxes[:, 2] > boxes[:, 0]).all() and (boxes[:, 3] > boxes[:, 1]).all():
                # 判定为 xyxy 格式（x2>x1, y2>y1）
                w = boxes[:, 2] - boxes[:, 0]
                h = boxes[:, 3] - boxes[:, 1]
            else:
                # 判定为 xywh 格式
                w = boxes[:, 2]
                h = boxes[:, 3]
            area = w * h  # 计算每个框的面积 [N,]
        else:
            raise ValueError(f"boxes维度错误，需为 [N,4]，当前为 {boxes.shape}")
        
        # 找到最大面积的索引（若多个框面积相同，取第一个）
        max_area_idx = torch.argmax(area)
        
        # 筛选该索引对应的框、标签、分数
        filtered_labels = labels[max_area_idx:max_area_idx+1]  # 保留维度 [1,]
        filtered_boxes = boxes[max_area_idx:max_area_idx+1]    # 保留维度 [1,4]
        filtered_scores = scores[max_area_idx:max_area_idx+1]  # 保留维度 [1,]
        filtered_scores_vector = scores_vector[max_area_idx:max_area_idx+1]  # 保留维度 [1,D]
        filtered_class_names.append(class_name_temp[max_area_idx])
        # 将筛选结果添加到输出字典
        filtered_gt['labels'].append(filtered_labels)
        filtered_gt['boxes'].append(filtered_boxes)
        filtered_gt['scores'].append(filtered_scores)
        filtered_gt['scores_vector'].append(filtered_scores_vector)
    
    return filtered_gt,filtered_class_names





def yolo_boxes_to_corners(boxes):
    """
    将多batch YOLO检测框转换为中心点坐标列表
    
    参数：
        boxes: 检测框输入，支持两种格式：
                - 多batch列表：List[torch.Tensor]，每个元素形状为 (N, 4)（xyxy格式），对应一个batch的检测框
                - 单batch张量：torch.Tensor，形状为 (N, 4)（xyxy格式）
        img_shape: 图像尺寸 (height, width)，若需归一化坐标转绝对坐标则传入
    
    返回：
        corners_list: 嵌套列表，外层长度=batch数，内层每个元素为 [x_center, y_center]
    """
    # 统一输入格式为多batch列表
    if isinstance(boxes, torch.Tensor):
        boxes = [boxes]  # 单batch转为列表
    
    corners_list = []
    # 遍历每个batch
    for batch_idx, batch_boxes in enumerate(boxes):
        batch_corners = []
        if batch_boxes.numel() == 0:  # 当前batch无检测框
            corners_list.append(batch_corners)
            continue
        
        # 维度校验
        if batch_boxes.ndim != 2 or batch_boxes.shape[1] != 4:
            raise ValueError(f"Batch {batch_idx} 检测框形状错误，期望 (N, 4)，实际 {batch_boxes.shape}")
        
        # 转换为numpy（也可保留张量计算）
        batch_boxes_np = batch_boxes.detach().cpu().numpy()
        
        # 计算每个框的中心点
        for box in batch_boxes_np:
            x1, y1, x2, y2 = box
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            

            
            # 可选：转为整数
            x_center, y_center = map(int, [x_center, y_center])
            batch_corners.append([x_center, y_center])
        
        corners_list.append(batch_corners)
    
    return corners_list



                  
def extract_mask_content( input_tensor, mask, mask_value=1.0):
    """
    从输入 tensor 中抠出 mask 内的内容，mask 外区域设为指定值（默认为 1.0）
    
    参数：
        input_tensor: 输入图像 tensor（格式：BCHW 或 CHW，值范围 0-1）
        mask: 二维掩码 array 或 tensor（格式：(H, W)，元素为 True/False，True 表示保留区域）
        mask_value: mask 外区域的填充值（默认 1.0）
    
    返回：
        result_tensor: 处理后的 tensor，mask 内保留原图内容，mask 外为 mask_value
    """
    # 记录原始输入维度，用于最终格式恢复
    original_dim = input_tensor.dim()
    
    # 1. 统一输入 tensor 为 BCHW 格式（确保有 batch 维度）
    if original_dim == 3:  # CHW → BCHW
        input_tensor = input_tensor.unsqueeze(0)
    B, C, H, W = input_tensor.shape  # 此时 input_tensor 一定是 BCHW
    result_list=[]
    for i in range(B):  # 批量处理

        mask_b=mask[i]
        input_tensor_b=input_tensor[i]
        # 2. 处理二维 mask，转为 tensor 并扩展维度以匹配 BCHW
        if isinstance(mask_b, np.ndarray):
            mask_b = torch.from_numpy(mask_b).bool()  # numpy 转 bool tensor
        else:
            mask_b = mask_b.bool()  # 确保是 bool 类型
        
        # 扩展 mask 维度：(H, W) → (1, 1, H, W)，再通过广播匹配 (B, C, H, W)
        mask_b = mask_b.unsqueeze(0).unsqueeze(0)  # 增加 batch 和 channel 维度
        mask_b = mask_b.to(input_tensor_b.device)  # 确保与输入 tensor 同设备
        
        # 3. 生成填充值 tensor（与输入同形状）
        fill_tensor = torch.full_like(input_tensor_b, fill_value=mask_value)
        
        # 4. 核心操作：mask 内保留原图，mask 外填充
        result_tensor = torch.where(mask_b, input_tensor_b, fill_tensor)
        
        result_list.append(result_tensor)
    result_tensor_all=torch.cat(result_list)
    return result_tensor_all



def pad_to_square(img_tensor, pad_mode="constant", fill_value=0.0):
    """
    将输入的图像 tensor 填充为正方形（宽高相等，且为原始最大边长）
    
    参数：
        img_tensor: 输入图像 tensor，形状为 (C, H, W) 或 (B, C, H, W)
        pad_mode: 填充模式，同 F.pad 的 mode 参数（如 "constant", "edge", "reflect" 等）
        fill_value: 填充值（当 pad_mode 为 "constant" 时有效）
    
    返回：
        padded_tensor: 填充后的正方形 tensor，形状为 (C, max_dim, max_dim) 或 (B, C, max_dim, max_dim)
    """
    # 处理单张图像（C, H, W）或批量图像（B, C, H, W）
    if img_tensor.ndim == 3:
        C, H, W = img_tensor.shape
        batch_mode = False
    elif img_tensor.ndim == 4:
        B, C, H, W = img_tensor.shape
        batch_mode = True
    else:
        raise ValueError(f"输入 tensor 维度必须是 3 (C, H, W) 或 4 (B, C, H, W)，但得到 {img_tensor.ndim}")
    
    max_dim = max(H, W)
    
    # 计算填充量：(上, 下, 左, 右)
    # 上下填充总和 = max_dim - H；左右填充总和 = max_dim - W
    pad_top = (max_dim - H) // 2
    pad_bottom = max_dim - H - pad_top  # 确保上下填充总和正确（处理奇数情况）
    pad_left = (max_dim - W) // 2
    pad_right = max_dim - W - pad_left  # 确保左右填充总和正确
    
    # 构造填充参数（注意：pad 的顺序是 (左, 右, 上, 下) 对于最后两个维度）
    pad = (pad_left, pad_right, pad_top, pad_bottom)
    
    # 执行填充
    if batch_mode:
        # 批量图像：在 H 和 W 维度填充（即最后两个维度）
        padded_tensor = F.pad(img_tensor, pad, mode=pad_mode, value=fill_value)
    else:
        # 单张图像：同样在 H 和 W 维度填充
        padded_tensor = F.pad(img_tensor, pad, mode=pad_mode, value=fill_value)
    
    return padded_tensor




def batched_tensor_mask_overlay(background_tensor, image_tensor, mask_array):
    """
    批量处理四维张量的mask覆盖操作，同时保证梯度传导
    
    参数:
    background_tensor: 背景张量，形状为[B, C, H, W]
    image_tensor: 前景张量，形状为[B, C, H, W]，需与背景张量尺寸匹配
    mask_array: 布尔数组，形状为[B, H, W]或[H, W]，True表示需要覆盖的区域
    
    返回:
    result_tensor: 合成后的张量，形状为[B, C, H, W]，保留梯度信息
    """
    # 确保device一致
    on_device=image_tensor.device
    background_tensor=background_tensor.to(on_device)

    # 确保输入张量形状匹配
    assert background_tensor.shape == image_tensor.shape, "背景和前景张量形状必须相同"

    
    # 处理mask形状，确保与输入张量匹配
    if mask_array.ndim == 2:  # [H, W] - 对所有批次使用相同mask
        mask_array = np.expand_dims(mask_array, axis=0)  # [1, H, W]
    assert mask_array.shape == (background_tensor.shape[0], background_tensor.shape[2], background_tensor.shape[3]), \
        f"mask形状应为[B, H, W]，实际为{mask_array.shape}"
    
    # 将mask数组转换为与输入张量匹配的形状 [B, C, H, W]
    mask = mask_array.astype(np.float32)
    mask = np.expand_dims(mask, axis=1)  # [B, 1, H, W]
    mask = np.repeat(mask, background_tensor.shape[1], axis=1)  # [B, C, H, W]
    
    # 转换为张量并确保与输入在同一设备，同时保留梯度计算能力
    mask_tensor = torch.from_numpy(mask).to(on_device, dtype=image_tensor.dtype)
    
    # 执行覆盖操作: 背景*(1-mask) + 前景*mask
    # 所有操作均为PyTorch张量操作，会自动跟踪梯度
    result_tensor = background_tensor * (1 - mask_tensor) + image_tensor * mask_tensor
    
    return result_tensor



def match_detection_boxes(
    result1: Dict[str, List[torch.Tensor]],
    result2: Dict[str, List[torch.Tensor]],
    iou_threshold: float = 0.0
) -> Dict[str, Union[List, int, float, bool]]:
    """
    匹配两个检测结果的检测框，找到最匹配的框并统计类别是否相等
    核心逻辑：通过IOU（交并比）匹配框，取IOU最大且≥阈值的框为匹配结果

    Args:
        result1: 第一个检测结果，结构为：
            {
                "boxes": 列表[B]，每个元素是[N1=1,4]的tensor（检测框坐标[x1,y1,x2,y2]）
                "labels": 列表[B]，每个元素是[N1=1]的tensor（类别索引）
                "scores": 列表[B]，每个元素是[N1=1]的tensor（置信度）
                "scores_vector": 列表[B]，每个元素是[N1=1,C]的tensor（类别得分向量）
            }
        result2: 第二个检测结果，结构与result1一致，但N2可为任意值（≥0）
        iou_threshold: IOU阈值，只有IOU≥该值的框才视为有效匹配，默认0.0（无过滤）

    Returns:
        Dict: 匹配结果统计，包含：
            - batch_indices: List[int]，batch索引列表
            - has_match: List[bool]，每个batch是否找到有效匹配框
            - max_iou: List[float]，每个batch匹配框的最大IOU值
            - matched_box_idx: List[Optional[int]]，result2中匹配框的索引（无匹配则为None）
            - label1: List[int]，result1中每个batch的类别索引
            - label2: List[Optional[int]]，result2中匹配框的类别索引（无匹配则为None）
            - label_equal: List[bool]，类别是否相等（无匹配则为False）
            - score1: List[float]，result1中每个batch的置信度
            - score2: List[Optional[float]]，result2中匹配框的置信度（无匹配则为None）
            - total_matched_batches: int，有有效匹配的batch总数
            - correct_label_count: int，类别匹配正确的数量
            - label_match_rate: float，类别匹配率（正确数/有效匹配数）
    """
    # -------------- 内部辅助函数：计算IOU --------------
    def _calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """计算单个框与多个框的IOU，box1:[4], box2:[N,4]，返回[N]的IOU张量"""
        if len(box2.shape) == 1:
            box2 = box2.unsqueeze(0)
        # 计算交集坐标
        x1 = torch.max(box1[0], box2[:, 0])
        y1 = torch.max(box1[1], box2[:, 1])
        x2 = torch.min(box1[2], box2[:, 2])
        y2 = torch.min(box1[3], box2[:, 3])
        # 交集面积（无交集则为0）
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        # 并集面积
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        union = area1 + area2 - intersection
        # 避免除零
        union = torch.clamp(union, min=1e-6)
        return intersection / union

    # -------------- 输入校验 --------------
    B = len(result1["boxes"])
    for key in ["boxes", "labels", "scores", "scores_vector"]:
        assert key in result1 and key in result2, f"检测结果缺少关键键值：{key}"
        assert len(result2[key]) == B, f"两个检测结果的batch长度不一致（{key}）"
        if key == "boxes":
            # 校验result1的每个batch都是N=1
            for b in range(B):
                # 检测框数量不足，则跳过
                if result1["boxes"][b].shape[0] < 1: continue
                # assert result1["boxes"][b].shape[0] == 1, f"result1的batch {b} 检测框数量不是1"

    # -------------- 初始化结果容器 --------------
    match_results = {
        "batch_indices": [], "has_match": [], "max_iou": [], "matched_box_idx": [],
        "label1": [], "label2": [], "label_equal": [], "score1": [], "score2": []
    }

    # -------------- 逐Batch匹配 --------------
    for b in range(B):
        # 如果是空，则跳过
        if result1["boxes"][b].shape[0] < 1: continue
        # 获取result1当前batch的信息（N=1）
        box1 = result1["boxes"][b].squeeze(0)  # [4]
        label1 = result1["labels"][b].squeeze(0).item()
        score1 = result1["scores"][b].squeeze(0).item()

        # 获取result2当前batch的信息
        boxes2 = result2["boxes"][b]  # [N2,4]
        labels2 = result2["labels"][b]  # [N2]
        scores2 = result2["scores"][b]  # [N2]
        N2 = boxes2.shape[0] if boxes2.numel() > 0 else 0

        # 初始化当前batch的结果
        match_results["batch_indices"].append(b)
        match_results["label1"].append(label1)
        match_results["score1"].append(score1)

        # 情况1：result2当前batch无检测框
        if N2 == 0:
            match_results["has_match"].append(False)
            match_results["max_iou"].append(0.0)
            match_results["matched_box_idx"].append(None)
            match_results["label2"].append(None)
            match_results["label_equal"].append(False)
            match_results["score2"].append(None)
            continue

        # 情况2：计算IOU并找最大匹配
        ious = _calculate_iou(box1, boxes2)  # [N2]
        max_iou, matched_idx = torch.max(ious, dim=0)
        max_iou = max_iou.item()
        matched_idx = matched_idx.item()

        # 过滤IOU阈值：若最大IOU < 阈值，视为无匹配
        if max_iou < iou_threshold:
            match_results["has_match"].append(False)
            match_results["max_iou"].append(max_iou)
            match_results["matched_box_idx"].append(None)
            match_results["label2"].append(None)
            match_results["label_equal"].append(False)
            match_results["score2"].append(None)
            continue

        # 情况3：找到有效匹配框
        label2 = labels2[matched_idx].item()
        score2 = scores2[matched_idx].item()
        label_equal = (label1 == label2)

        # 保存结果
        match_results["has_match"].append(True)
        match_results["max_iou"].append(max_iou)
        match_results["matched_box_idx"].append(matched_idx)
        match_results["label2"].append(label2)
        match_results["label_equal"].append(label_equal)
        match_results["score2"].append(score2)

    # -------------- 统计整体指标 --------------
    total_matched = sum(match_results["has_match"])
    correct_label = sum(match_results["label_equal"])
    label_match_rate = correct_label / total_matched if total_matched > 0 else 0.0

    match_results["total_matched_batches"] = total_matched
    match_results["correct_label_count"] = correct_label
    match_results["label_match_rate"] = label_match_rate

    return match_results




def tensor_01_to_int8_and_back(x: torch.Tensor) -> torch.Tensor:
    """
    模拟0-1 tensor的int8量化+还原，保证计算图完全不中断（无真实int8类型转换）
    核心：用浮点类型模拟int8量化（数值上符合int8范围），避免整数类型截断梯度
    """
    # 保存原始类型和设备
    orig_dtype = x.dtype
    orig_device = x.device
    
    # 1. 固定缩放系数（0-1 → 0-255）
    scale = torch.tensor(255.0, device=orig_device, dtype=orig_dtype, requires_grad=False)
    
    # 2. 确保输入严格在0-1范围（可微操作）
    x_clamped = x.clamp(0.0, 1.0)
    
    # 3. 模拟int8量化（关键：全程保留浮点类型，不转真实int8）
    x_scaled = x_clamped * scale  # 0-1 → 0-255（浮点）
    x_scaled_rounded = torch.round(x_scaled)  # 四舍五入为整数（仍为浮点）
    x_int8_sim = x_scaled_rounded.clamp(0.0, 255.0)  # 限制0-255（模拟int8范围，浮点类型）
    
    # 4. 还原为0-1范围（浮点操作，梯度可回传）
    x_restored = x_int8_sim / scale  # 255 → 0-1
    x_restored = x_restored.clamp(0.0, 1.0)  # 确保0-1范围
    
    # 保证设备/类型与输入完全一致
    x_restored = x_restored.to(device=orig_device, dtype=orig_dtype)
    
    return x_restored




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




def get_true_ratio_per_channel(bool_arr: np.ndarray) -> np.ndarray:
    """
    计算shape为(n, H, W)的bool类型numpy数组中，每个通道（n维度）True的比例
    
    参数:
        bool_arr: 输入数组，必须是3维bool数组，shape=(n, H, W)
    
    返回:
        ratio_arr: 各通道True的比例数组，shape=(n,)，值范围[0,1]
    """
    # 1. 输入校验（避免维度/类型错误）
    if bool_arr.ndim != 3:
        raise ValueError(f"输入必须是3维数组（n*H*W），当前维度：{bool_arr.ndim}")
    if bool_arr.dtype != np.bool_:
        raise TypeError(f"输入必须是bool类型数组，当前类型：{bool_arr.dtype}")
    
    # 2. 拆分维度
    n, H, W = bool_arr.shape
    
    # 3. 按通道统计True数量（沿H、W维度求和，保留n维度）
    # True=1，False=0，sum后得到每个通道的True总数
    true_count_per_channel = np.sum(bool_arr, axis=(1, 2))
    
    # 4. 计算每个通道的总元素数（H*W）
    total_per_channel = H * W
    
    # 5. 计算每个通道的True比例
    ratio_per_channel = true_count_per_channel / total_per_channel
    
    return ratio_per_channel


def fit_canny_to_mask(canny_tensor: torch.Tensor, mask_np: np.ndarray) -> torch.Tensor:
    """
    将B*C*H*W的Canny边缘张量缩放适配到B*H*W的Mask区域，并仅保留Mask内的Canny边缘
    
    参数:
        canny_tensor: 输入Canny边缘张量，shape=(B, C, H_canny, W_canny)，dtype=torch.float32/uint8，值范围0~1或0~255
        mask_np: 输入Mask数组，shape=(B, H_mask, W_mask)，dtype=np.bool_/np.uint8（True/255表示有效区域）
    
    返回:
        masked_canny: 缩放后仅保留Mask区域的Canny张量，shape=(B, C, H_mask, W_mask)，值范围0~1
    """
    # ========== 1. 输入校验 ==========
    # 校验Batch维度一致
    if canny_tensor.shape[0] != mask_np.shape[0]:
        raise ValueError(f"Canny Batch数({canny_tensor.shape[0]})与Mask Batch数({mask_np.shape[0]})不匹配")
    # 校验Mask维度（必须是B*H*W）
    if mask_np.ndim != 3:
        raise ValueError(f"Mask必须是3维(B*H*W)，当前维度：{mask_np.ndim}")
    # 校验Canny维度（必须是B*C*H*W）
    if canny_tensor.ndim != 4:
        raise ValueError(f"Canny必须是4维(B*C*H*W)，当前维度：{canny_tensor.ndim}")
    
    # ========== 2. 预处理 ==========
    B, C, _, _ = canny_tensor.shape
    H_mask, W_mask = mask_np.shape[1], mask_np.shape[2]
    # Mask转张量（bool型）+ 扩展通道维度（B*H*W → B*1*H*W）
    mask_tensor = torch.from_numpy(mask_np).to(canny_tensor.device)
    if mask_tensor.dtype != torch.bool:
        mask_tensor = mask_tensor > 0  # 统一转为bool（255→True，0→False）
    mask_tensor = mask_tensor.unsqueeze(1)  # B*1*H*W
    
    # Canny值归一化到0~1（兼容0~255的输入）
    if canny_tensor.dtype == torch.uint8:
        canny_tensor = canny_tensor.float() / 255.0
    canny_tensor = torch.clamp(canny_tensor, 0.0, 1.0)
    
    # ========== 3. 逐Batch缩放Canny到Mask尺寸（保持比例） ==========
    masked_canny_list = []
    for b in range(B):
        # 取出单Batch数据
        canny_single = canny_tensor[b:b+1]  # 1*C*H_canny*W_canny
        mask_single = mask_tensor[b:b+1]    # 1*1*H_mask*W_mask
        
        # 3.1 计算缩放比例（保持Canny宽高比，适配Mask尺寸）
        H_canny, W_canny = canny_single.shape[2], canny_single.shape[3]
        # 计算宽/高缩放系数（取最小系数，避免超出Mask）
        scale_h = H_mask / H_canny
        scale_w = W_mask / W_canny
        scale = min(scale_h, scale_w)  # 等比例缩放，保证Canny完全放入Mask
        
        # 3.2 缩放Canny
        new_H = int(H_canny * scale)
        new_W = int(W_canny * scale)
        canny_scaled = torchvision.transforms.functional.resize(canny_single, size=(new_H, new_W), antialias=True)  # 1*C*new_H*new_W
        
        # 3.3 居中填充到Mask尺寸（空白区域补0）
        # 计算上下左右填充量
        pad_top = (H_mask - new_H) // 2
        pad_bottom = H_mask - new_H - pad_top
        pad_left = (W_mask - new_W) // 2
        pad_right = W_mask - new_W - pad_left
        # 填充（空白区域补0）
        canny_padded = torchvision.transforms.functional.pad(
            canny_scaled,
            padding=[pad_left, pad_top, pad_right, pad_bottom],
            fill=0.0,
            padding_mode="constant"
        )  # 1*C*H_mask*W_mask
        
        # 3.4 仅保留Mask区域的Canny（非Mask区域置0）
        canny_masked = canny_padded * mask_single  # 1*C*H_mask*W_mask
        masked_canny_list.append(canny_masked)
    
    # ========== 4. 合并结果 ==========
    masked_canny = torch.cat(masked_canny_list, dim=0)  # B*C*H_mask*W_mask
    
    return masked_canny


def fit_canny_to_xyxy_boxes(canny_tensor: torch.Tensor, xyxy_boxes: list,resize_scale) -> torch.Tensor:
    """
    将B*C*H*W的Canny边缘张量，逐Batch等比例缩放到对应xyxy方框内（以方框左上角为原点，不居中）
    
    参数:
        canny_tensor: 输入Canny张量，shape=(B, C, H_canny, W_canny)，dtype=torch.float32，值范围0~1
        xyxy_boxes: 长度为B的列表，每个元素是[min_x, min_y, max_x, max_y]（xyxy格式，基于Canny原图的像素坐标）
    
    返回:
        boxed_canny: 缩放后仅填充xyxy方框的Canny张量，shape=(B, C, H_canny, W_canny)，方框外为0
    """
    # ========== 1. 输入校验 ==========
    B, C, H_canny, W_canny = canny_tensor.shape
    # 校验列表长度与Batch数一致
    if len(xyxy_boxes) != B:
        raise ValueError(f"xyxy列表长度({len(xyxy_boxes)})与Canny Batch数({B})不匹配")
    # 校验每个xyxy的格式
    for b, box in enumerate(xyxy_boxes):
        if len(box) != 4:
            raise ValueError(f"Batch{b}的xyxy格式错误，需为[min_x, min_y, max_x, max_y]，当前长度：{len(box)}")
        min_x, min_y, max_x, max_y = box
        # 校验坐标有效性
        if min_x < 0 or min_y < 0 or max_x > W_canny or max_y > H_canny:
            raise ValueError(f"Batch{b}的xyxy坐标超出Canny范围(W={W_canny}, H={H_canny})：{box}")
        if min_x >= max_x or min_y >= max_y:
            raise ValueError(f"Batch{b}的xyxy坐标无效（min >= max）：{box}")
    scaled_boxes = [[int(coord * resize_scale[b].item()) for coord in box] for b, box in enumerate(xyxy_boxes)]
    # ========== 2. 逐Batch处理 ==========
    boxed_canny_list = []
    for b in range(B):
        # 取出单Batch数据
        canny_single = canny_tensor[b:b+1]  # 1*C*H_canny*W_canny
        min_x, min_y, max_x, max_y =scaled_boxes[b]
        
        # 2.1 计算方框的宽高
        box_w = max_x - min_x
        box_h = max_y - min_y
        
        # 2.2 计算等比例缩放系数（保证Canny完全放入方框，不拉伸）
        # Canny原图的宽高
        canny_h, canny_w = H_canny, W_canny
        # 宽/高方向的缩放系数
        scale_w = box_w / canny_w
        scale_h = box_h / canny_h
        scale = min(scale_w, scale_h)  # 取最小系数，避免超出方框
        scale*=0.8
        # 2.3 缩放Canny到方框适配尺寸
        new_canny_h = math.floor(canny_h * scale)
        new_canny_w = math.floor(canny_w * scale)
        canny_scaled = torchvision.transforms.functional.resize(canny_single, size=(new_canny_h, new_canny_w), antialias=True)

        canvas = torch.zeros_like(canny_single)  # 1*C*H_canny*W_canny
        

        fill_x1 = box_w//2-new_canny_w//2
        fill_x2 = fill_x1+new_canny_w
        fill_y1 = box_h//2-new_canny_h//2
        fill_y2 =  fill_y1+new_canny_h

        
        # 填充到画布（从方框左上角开始放置）
        canvas[:, :, fill_y1:fill_y2, fill_x1:fill_x2] = canny_scaled
        
        boxed_canny_list.append(canvas)
    
    # ========== 3. 合并结果 ==========
    boxed_canny = torch.cat(boxed_canny_list, dim=0)  # B*C*H_canny*W_canny
    
    return boxed_canny


def align_tensor_dtype(target_tensor: torch.Tensor, ref_tensor: torch.Tensor) -> torch.Tensor:
    """
    将目标张量的dtype和设备对齐到参考张量
    Args:
        target_tensor: 需要转换的张量
        ref_tensor: 参考张量（作为类型/设备模板）
    Returns:
        转换后的目标张量（与参考张量dtype+device完全一致）
    """
    # 1. 对齐数据类型
    if target_tensor.dtype != ref_tensor.dtype:
        target_tensor = target_tensor.to(dtype=ref_tensor.dtype)
    # 2. 对齐设备（CPU/GPU）
    if target_tensor.device != ref_tensor.device:
        target_tensor = target_tensor.to(device=ref_tensor.device)
    return target_tensor





def create_center_rect_mask(img_size, rect_size, batch_size=1, dtype=np.uint8):
    """
    生成中心矩形掩码矩阵：中心指定尺寸的矩形区域为1，其余区域为0，支持批次维度
    
    参数：
        img_size (int/tuple): 掩码矩阵尺寸，支持两种输入：
                             - int: 生成(img_size, img_size)的正方形掩码
                             - tuple: 生成(img_h, img_w)的矩形掩码（如(512, 640)）
        rect_size (int/tuple): 中心矩形尺寸，支持两种输入：
                              - int: 生成(rect_size, rect_size)的正方形矩形
                              - tuple: 生成(rect_h, rect_w)的矩形（如(100, 80)）
        batch_size (int): 批次维度大小（B），默认1，生成B×H×W的掩码张量
        dtype (np.dtype): 掩码矩阵的数据类型，默认np.uint8
    
    返回：
        np.ndarray: 掩码矩阵，形状为(batch_size, img_h, img_w)，矩形区域为1，其余为0
    
    异常：
        ValueError: 当输入尺寸为非正数、矩形尺寸超过掩码尺寸或batch_size≤0时抛出
    """
    # 新增：校验batch_size合法性
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"batch_size必须为正整数，当前输入：{batch_size}")
    
    # 1. 统一尺寸格式（兼容int/tuple输入）
    if isinstance(img_size, int):
        img_h, img_w = img_size, img_size
    elif isinstance(img_size, (tuple, list)) and len(img_size) == 2:
        img_h, img_w = img_size
    else:
        raise ValueError(f"img_size必须是int或长度为2的tuple/list，当前输入：{img_size}")
    
    if isinstance(rect_size, int):
        rect_h, rect_w = rect_size, rect_size
    elif isinstance(rect_size, (tuple, list)) and len(rect_size) == 2:
        rect_h, rect_w = rect_size
    else:
        raise ValueError(f"rect_size必须是int或长度为2的tuple/list，当前输入：{rect_size}")
    
    # 2. 参数合法性校验
    if img_h <= 0 or img_w <= 0:
        raise ValueError(f"掩码尺寸必须为正数，当前：({img_h}, {img_w})")
    if rect_h <= 0 or rect_w <= 0:
        raise ValueError(f"矩形尺寸必须为正数，当前：({rect_h}, {rect_w})")
    if rect_h > img_h or rect_w > img_w:
        raise ValueError(f"矩形尺寸({rect_h}, {rect_w})不能超过掩码尺寸({img_h}, {img_w})")
    
    # 3. 初始化单张全0掩码（H×W）
    single_mask = np.zeros((img_h, img_w), dtype=dtype)
    
    # 4. 计算中心矩形的边界（确保居中）
    center_h = img_h // 2
    center_w = img_w // 2
    
    h_start = center_h - rect_h // 2
    h_end = center_h + rect_h // 2
    w_start = center_w - rect_w // 2
    w_end = center_w + rect_w // 2
    
    # 处理偶数/奇数尺寸的边界对齐（确保矩形尺寸准确）
    if rect_h % 2 == 0:
        h_end = center_h + rect_h // 2
    else:
        h_end = center_h + rect_h // 2 + 1
    
    if rect_w % 2 == 0:
        w_end = center_w + rect_w // 2
    else:
        w_end = center_w + rect_w // 2 + 1
    
    # 5. 将中心矩形区域设为1（单张掩码）
    single_mask[h_start:h_end, w_start:w_end] = 1
    
    # 6. 扩展批次维度：将单张掩码复制batch_size次，形成B×H×W
    batch_mask = np.repeat(single_mask[np.newaxis, ...], batch_size, axis=0)
    
    return batch_mask



def calculate_resize_scale_back(resize_scale):
    """
    计算resize_scale的倒数（resize_scale_back = 1/resize_scale）
    兼容tensor/列表（含长度为b的元组）两种输入类型
    
    参数：
        resize_scale: 缩放因子，支持两种类型：
                      - torch.Tensor: 任意形状的张量（如[B,] / [B,2]）
                      - list: 元素为元组，每个元组长度为b（如[(s1, s2), (s3, s4)]）
    返回：
        resize_scale_back: 与输入类型/形状一致的倒数结果
    """
    # 情况1：输入为torch.Tensor
    if isinstance(resize_scale, torch.Tensor):
        # 防止除零错误
        if (resize_scale == 0).any():
            raise ValueError("resize_scale张量中包含0值，无法计算倒数！")
        resize_scale_back = 1.0 / resize_scale
        return resize_scale_back
    
    # 情况2：输入为list（元素是长度为b的元组）
    elif isinstance(resize_scale, list):
        resize_scale_back = []
        for tup in resize_scale:
            # 校验元组长度（确保是长度为b的元组）
            if not isinstance(tup, tuple):
                raise TypeError(f"列表元素必须是元组，当前元素类型：{type(tup)}")
            # 对元组中每个元素计算倒数
            tup_back = tuple(1.0 / s for s in tup if s != 0)
            # 校验元组长度是否一致（防止除零后长度变化）
            if len(tup_back) != len(tup):
                raise ValueError(f"元组{tup}中包含0值，无法计算倒数！")
            resize_scale_back.append(tup_back)
        return resize_scale_back
    
    # 情况3：不支持的类型
    else:
        raise TypeError(
            f"不支持的resize_scale类型：{type(resize_scale)}，仅支持torch.Tensor或包含元组的list！"
        )