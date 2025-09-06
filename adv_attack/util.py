import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou  # GIoU计算工具

# 纯PyTorch实现的匈牙利算法（无改动，确保不依赖外部库）
def hungarian_matching(cost_matrix):
    """
    纯PyTorch实现的匈牙利算法，用于解决指派问题
    Args:
        cost_matrix: 成本矩阵，shape [M, K]
    Returns:
        row_ind: 行索引（预测框索引）
        col_ind: 列索引（真实框索引）
    """
    device = cost_matrix.device
    M, K = cost_matrix.shape
    
    u = torch.zeros(M + 1, device=device)
    v = torch.zeros(K + 1, device=device)
    p = torch.zeros(K + 1, dtype=torch.long, device=device)
    way = torch.zeros(K + 1, dtype=torch.long, device=device)
    
    for i in range(1, M + 1):
        p[0] = i
        minv = torch.full((K + 1,), float('inf'), device=device)
        used = torch.zeros(K + 1, dtype=torch.bool, device=device)
        j0 = 0
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float('inf')
            j1 = 0
            for j in range(1, K + 1):
                if not used[j]:
                    cur = cost_matrix[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(K + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    
    # 过滤无效匹配（p[j] == 0 表示无匹配）
    valid_col_mask = p[1:K+1] != 0
    col_ind = torch.arange(1, K + 1, device=device)[valid_col_mask] - 1  # 转为0-based
    row_ind = p[1:K+1][valid_col_mask] - 1  # 转为0-based
    
    return row_ind, col_ind

# 定义YOLOv11检测损失函数（适配输入结构：无logits）
class YOLOv11DetectionLoss(nn.Module):
    def __init__(self, num_classes=80, conf_threshold=0.25, iou_threshold=0.45, 
                 weight_class=1.0, weight_bbox_l1=1.0, weight_giou=1.0):
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
        self.num_classes = num_classes
        self.conf_thres = conf_threshold
        self.iou_thres = iou_threshold
        self.weight_class = weight_class
        self.weight_bbox_l1 = weight_bbox_l1
        self.weight_giou = weight_giou

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

    def forward(self, pred_result, gt_result):
        """
        简化版前向传播：适配输入结构（pred_result仅含boxes/scores/labels）
        Args:
            pred_result: 模型预测结果（与你的输入结构完全一致）
                'boxes': list[tensor] → [B个元素，每个元素 shape [M_b, 4] (xyxy)]
                'scores': list[tensor] → [B个元素，每个元素 shape [M_b]]（置信度）
                'labels': list[tensor] → [B个元素，每个元素 shape [M_b]]（预测类别，0~num_classes-1）
            gt_result: 真实标签（格式与pred_result对齐，无scores）
                'boxes': list[tensor] → [B个元素，每个元素 shape [K_b, 4] (xyxy)]
                'labels': list[tensor] → [B个元素，每个元素 shape [K_b]]（真实类别，0~num_classes-1）
        Returns:
            total_loss: 总损失（标量）
            loss_dict: 各任务损失明细（dict of 标量）
        """
        # 确定设备（避免空输入报错）
        if len(pred_result['boxes']) == 0 or pred_result['boxes'][0].numel() == 0:
            device = torch.device('cpu')
        else:
            device = pred_result['boxes'][0].device
            
        # 初始化损失累计变量
        total_class_loss = torch.tensor(0.0, device=device)
        total_bbox_l1_loss = torch.tensor(0.0, device=device)
        total_giou_loss = torch.tensor(0.0, device=device)
        total_valid_boxes = 0  # 统计匹配成功的框对数量（用于平均损失）

        # 遍历每个batch样本计算损失
        batch_size = len(pred_result['boxes'])
        for b in range(batch_size):
            # 1. 提取单样本数据（确保设备一致）
            pred_boxes = pred_result['boxes'][b].to(device)  # [M_b, 4]
            pred_scores = pred_result['scores'][b].to(device)  # [M_b]
            pred_labels = pred_result['labels'][b].to(device)  # [M_b]
            gt_boxes = gt_result['boxes'][b].to(device)  # [K_b, 4]
            gt_labels = gt_result['labels'][b].to(device)  # [K_b]

            # 2. 过滤低置信度预测框（减少无效计算）
            pred_mask = pred_scores >= self.conf_thres
            pred_boxes = pred_boxes[pred_mask]
            pred_scores = pred_scores[pred_mask]
            pred_labels = pred_labels[pred_mask]
            M_b, K_b = pred_boxes.shape[0], gt_boxes.shape[0]

            # 3. 若无预测框或无真实框，跳过该样本
            if M_b == 0 or K_b == 0:
                continue

            # 4. 匈牙利算法匹配框对
            pred_idx, gt_idx = self._hungarian_matching(
                pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels
            )
            T = pred_idx.shape[0]  # 匹配成功的框对数量
            if T == 0:
                continue
            total_valid_boxes += T

            # 5. 提取匹配成功的框对和标签
            matched_pred_boxes = pred_boxes[pred_idx]  # [T, 4]
            matched_pred_scores = pred_scores[pred_idx]  # [T]（用于分类损失的置信度加权）
            matched_pred_labels = pred_labels[pred_idx]  # [T]
            matched_gt_boxes = gt_boxes[gt_idx]  # [T, 4]
            matched_gt_labels = gt_labels[gt_idx]  # [T]

            # 6. 计算分类损失（适配离散标签：用置信度作为"伪概率"的对数）
            # 逻辑：置信度→归一化到(0,1)→取log→作为NLLLoss的输入（模拟分类概率）
            pred_log_probs = torch.log(matched_pred_scores.clamp(min=1e-6, max=1.0))  # 避免log(0)
            # 构建"类别-概率"映射：仅匹配的类别对应log概率，其他类别为-∞（确保NLLLoss正确计算）
            class_log_probs = torch.full((T, self.num_classes), -float('inf'), device=device)
            class_log_probs[torch.arange(T), matched_pred_labels] = pred_log_probs
            # 计算分类损失（NLLLoss：输入[batch, class_num]，目标[batch]）
            class_loss = self.class_criterion(class_log_probs, matched_gt_labels).sum()
            total_class_loss += class_loss

            # 7. 计算边界框L1损失（坐标误差：x1,y1,x2,y2的绝对误差和）
            bbox_l1_loss = self.bbox_l1_criterion(matched_pred_boxes, matched_gt_boxes).sum(dim=1).sum()
            total_bbox_l1_loss += bbox_l1_loss

            # 8. 计算GIoU损失（重叠度误差：1 - GIoU，确保损失非负）
            giou = generalized_box_iou(matched_pred_boxes, matched_gt_boxes)  # [T]
            giou_loss = (1 - giou).sum()  # GIoU损失求和
            total_giou_loss += giou_loss

        # 9. 计算平均损失（避免无匹配框时除以0）
        if total_valid_boxes == 0:
            class_loss_avg = torch.tensor(0.0, device=device)
            bbox_l1_loss_avg = torch.tensor(0.0, device=device)
            giou_loss_avg = torch.tensor(0.0, device=device)
        else:
            class_loss_avg = total_class_loss / total_valid_boxes
            bbox_l1_loss_avg = total_bbox_l1_loss / total_valid_boxes
            giou_loss_avg = total_giou_loss / total_valid_boxes

        # 10. 总损失 = 各任务损失 × 权重之和
        total_loss = (
            class_loss_avg * self.weight_class +
            bbox_l1_loss_avg * self.weight_bbox_l1 +
            giou_loss_avg * self.weight_giou
        )

        # 11. 输出损失明细（便于调试和监控）
        loss_dict = {
            'class_loss': class_loss_avg,
            'bbox_l1_loss': bbox_l1_loss_avg,
            'giou_loss': giou_loss_avg,
            'total_loss': total_loss
        }

        return total_loss, loss_dict