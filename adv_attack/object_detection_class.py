
import os
import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
from torchvision.models.detection.ssd import ssd300_vgg16, SSD300_VGG16_Weights
from tqdm import tqdm
from ultralytics import YOLO
from util import *

'''
本类如果使用了yolo 来检测，需要注意通道的排列，需要注意输入的数据维度，更多详细的参数以及用法
可以参考官方文档： https://docs.ultralytics.com/zh/modes/predict.html


'''  


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

'''
class ObjectDetection:
表示一个物体检测类，可以加载不同类型的模型，并对输入的图像进行物体检测。


'''
'''

result 结构体说明：
{
    "boxes": 列表, 列表长度等于B里面存着[N, 4]的tensor
    "labels":   列表，长度等于B，里面是[N]的tensor。# 检测框类别索引
    "scores": 列表，长度等于B，里面是[N]的tensor。# 检测框置信度
    "scores_vector":列表，长度等于B， 里面存着[N, C]的tensor,  # 检测框类别得分向量
}

'''
            

class ObjectDetection:
    def __init__(self,
                 device: torch.device = None,
                 **args):

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")   
        self.args = args
        # 解析args参数
        for key, value in args.items():
            setattr(self, key, value)  # 使用setattr动态设置属性
        conf_threshold = args.get("conf_threshold",0.25)
        iou_threshold = args.get("iou_threshold",0.2)
        strides = args.get("strides",[8, 16, 32])
        self.strides = strides
        # 判断是否存在image_size
        if "image_size" not in args:
            self.image_size = 512





        
        # 推理参数
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        yaml_path=args.get("nclass_yaml_path",None)
        if yaml_path is  None:
            raise ValueError("请指定yaml文件路径")
        
        # yaml 加载
        yaml_config = load_yaml_config(yaml_path)
        self.nc = len(yaml_config['names'])  # 类别数量
        self.names = yaml_config['names']    # 类别名称映射
        self.models = {}

    def detect(self, img, file_path: Optional[str] = None,model_type = "yolov11",file_name='detect.jpg', grad_status: bool = False) :
        """
        支持多batch的YOLO检测主函数
        :param img: 输入图像，支持：
                    - np.ndarray: [B, H, W, C] (RGB, 0-255) 或 [H, W, C] (单张)
                    - torch.Tensor: [B, C, H, W] (0-1) 或 [C, H, W] (单张)
        :param file_path: 检测结果保存路径（多batch时自动生成 batch_0, batch_1 后缀）
        :param grad_status: 是否开启梯度计算
        :return:
            - results: 检测结果字典（boxes/scores/labels/scores_vector，每个key对应长度=B的列表）
            - class_names: 每个batch置信度最高的类别名称（长度=B，无检测则为None）
        """


        # --------------------------
        # 1. 多batch输入统一处理
        # --------------------------
        img_tensor = self._tensor_vailid(img)
        img_tensor = self._preprocess_images(img_tensor,model_type)
        # --------------------------
        # 2. 多batch模型推理
        # --------------------------
        if "yolo" in model_type :
            with torch.set_grad_enabled(grad_status):
                # 判断模型和输入的数据类型
                model_dtype = next(self.models[model_type].parameters()).dtype
                if img_tensor.dtype !=model_dtype :
                    img_tensor = img_tensor.to(model_dtype)

                infer_results = self.models[model_type](img_tensor)  
                out = infer_results[0]
                out = out.permute(0, 2, 1).contiguous()  # [B,N,84]
                width, height = img_tensor.shape[-1], img_tensor.shape[-2]

                # 解码多batch结果
                results = self._decode_yolo_output(out)
        else:
            raise ValueError(f"Unsupported model type for postprocess: {self.model_type}")

        # --------------------------
        # 3. 提取每个batch的主类别（置信度最高）
        # --------------------------
        # class_names = self._get_main_class_per_batch(results)
        class_names=self._get_class_per_batch(results)

        # --------------------------
        # 4. 多batch结果可视化保存
        # --------------------------
        if file_path is not None:
            self.visualize_detections(img_tensor, results, save_path=file_path,file_name=file_name)

        return results, class_names


    def _tensor_vailid(self, img) :
        """
        统一多batch输入预处理逻辑
        :param img: 原始输入（np.ndarray/torch.Tensor）
        :return: 标准化的张量 [B, C, H, W] (0-1, 设备与模型一致)
        """
        if isinstance(img, np.ndarray):
            # 处理numpy数组（RGB, 0-255）
            if img.ndim == 3:
                # 单张图像 -> 扩展batch维度 [H, W, C] -> [1, H, W, C]
                img = np.expand_dims(img, axis=0)
            elif img.ndim != 4:
                raise ValueError(f"Unsupported numpy shape: {img.shape} (expected [B, H, W, C] or [H, W, C])")
            
            # 转张量 + 归一化 + 调整维度 [B, H, W, C] -> [B, C, H, W]
            img_tensor = torch.from_numpy(img).to(self.device).float() / 255.0
            img_tensor = img_tensor.permute(0, 3, 1, 2)  # BHWC -> BCHW

        elif isinstance(img, torch.Tensor):
            # 处理张量（0-1）
            if img.ndim == 3:
                # 单张图像 -> 扩展batch维度 [C, H, W] -> [1, C, H, W]
                img_tensor = img.unsqueeze(0)
            elif img.ndim != 4:
                raise ValueError(f"Unsupported tensor shape: {img.shape} (expected [B, C, H, W] or [C, H, W])")
            else:
                img_tensor = img.to(self.device)
        else:
            raise ValueError(f"Unsupported input type: {type(img)} (expected np.ndarray or torch.Tensor)")

        return img_tensor
    def _get_class_per_batch(self, results: Dict[str, List[torch.Tensor]]) -> List[Optional[str]]:
        """
        提取每个batch置信度最高的类别名称
        :param results: 解码后的检测结果
        :return: 长度=B的类别名称列表（无检测则为None）
        """
        class_names = []
        batch_size = len(results['boxes'])

        for b in range(batch_size):
            labels = results['labels'][b]
            
            if len(labels) == 0:
                class_names.append(None)
                continue
            
            # 取置所有类别name
            names = [self.names[int(label)] for label in labels]

            class_names.append(names)

        return class_names
    def _get_main_class_per_batch(self, results: Dict[str, List[torch.Tensor]]) -> List[Optional[str]]:
        """
        提取每个batch置信度最高的类别名称
        :param results: 解码后的检测结果
        :return: 长度=B的类别名称列表（无检测则为None）
        """
        class_names = []
        batch_size = len(results['boxes'])

        for b in range(batch_size):
            scores = results['scores'][b]
            labels = results['labels'][b]
            
            if len(scores) == 0:
                class_names.append(None)
                continue
            
            # 取置信度最高的类别
            max_score_idx = torch.argmax(scores)
            main_class_id = labels[max_score_idx].detach().cpu().numpy()
            class_names.append(self.names[int(main_class_id)])

        return class_names

    def _postprocess_output(self, outputs: torch.Tensor, **preprocess_info: Any) -> List[Dict[str, Any]]:
        """
        后处理模型输出（兼容多batch）
        :param outputs: 模型输出 [B, num_dets, 6]
        :param preprocess_info: 预处理信息（包含图像尺寸）
        :return: 每个batch的检测结果列表
        """
        # 解析预处理信息
        for key, value in preprocess_info.items():
            setattr(self, key, value)
        
        batch_results = []
        batch_size = outputs.shape[0]

        for b in range(batch_size):
            predictions = outputs[b]  # 处理单个batch [num_dets, 6]
            
            # 应用置信度阈值
            conf_mask = predictions[:, 4] >= self.conf_threshold
            predictions = predictions[conf_mask]
            
            if predictions.numel() == 0:
                batch_results.append([])
                continue
            
            # 分离边界框、置信度和类别
            boxes = predictions[:, :4]  # (x1, y1, x2, y2)
            confidences = predictions[:, 4]
            classes = predictions[:, 5].long()
            
            # 边界框裁剪
            if hasattr(self, 'width') and hasattr(self, 'height'):
                orig_w, orig_h = self.width, self.height
                boxes[:, 0] = torch.clamp(boxes[:, 0], 0, orig_w)
                boxes[:, 1] = torch.clamp(boxes[:, 1], 0, orig_h)
                boxes[:, 2] = torch.clamp(boxes[:, 2], 0, orig_w)
                boxes[:, 3] = torch.clamp(boxes[:, 3], 0, orig_h)
            
            # NMS
            indices = torch.ops.torchvision.nms(boxes, confidences, self.iou_threshold)
            boxes = boxes[indices]
            confidences = confidences[indices]
            classes = classes[indices]
            
            # 整理单个batch结果
            single_batch_results = []
            for i in range(boxes.shape[0]):
                single_batch_results.append({
                    'box': (boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]),
                    'confidence': confidences[i],
                    'class_id': classes[i],
                    'class_name': self.names[int(classes[i].item())]
                })
            batch_results.append(single_batch_results)

        return batch_results

    def _decode_boxes(self, pred: torch.Tensor, anchors: List[List[float]], stride: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将原始偏移量转换为实际坐标（xywh），支持多batch
        :param pred: 模型输出 [B, num_anchors, H, W, num_params]
        :param anchors: 锚框列表
        :param stride: 下采样步长
        :return: boxes [B, total_anchors, 4], obj_conf [B, total_anchors, 1], cls_conf [B, total_anchors, num_classes]
        """
        batch_size, num_anchors, height, width, _ = pred.shape
        device = pred.device
        
        # 生成网格坐标
        grid_x = torch.arange(width, device=device).repeat(height, 1).unsqueeze(2)
        grid_y = torch.arange(height, device=device).repeat(width, 1).t().unsqueeze(2)
        grid = torch.cat((grid_x, grid_y), 2).repeat(1, 1, num_anchors).unsqueeze(0)
        
        # 解析原始输出
        dx = pred[..., 0]
        dy = pred[..., 1]
        dw = pred[..., 2]
        dh = pred[..., 3]
        obj_conf = pred[..., 4]
        cls_conf = pred[..., 5:]
        
        # 坐标转换
        x = (dx * 2 - 0.5 + grid[..., 0]) * stride
        y = (dy * 2 - 0.5 + grid[..., 1]) * stride
        w = (dw * 2) ** 2 * torch.tensor(anchors, device=device)[..., 0].unsqueeze(1).unsqueeze(1)
        h = (dh * 2) ** 2 * torch.tensor(anchors, device=device)[..., 1].unsqueeze(1).unsqueeze(1)
        
        # 重塑形状
        boxes = torch.stack([x, y, w, h], dim=-1).view(batch_size, -1, 4)
        obj_conf = obj_conf.view(batch_size, -1, 1)
        cls_conf = cls_conf.view(batch_size, -1, self.nc)
        
        return boxes, obj_conf, cls_conf

    def _decode_yolo_output(self, raw_outs: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        """
        解析YOLOv11模型输出（Anchor Free），完全支持多batch
        :param raw_outs: 模型输出 [B, N, 84]
        :return: results 字典（每个key对应长度=B的列表）
        """
        if raw_outs.ndim != 3:
            raise ValueError(f"Expected raw_outs shape [B, N, 84], got {raw_outs.shape}")
        
        batch_size = raw_outs.shape[0]
        preds = raw_outs  # [B, N, 84]

        boxes_all = []
        scores_all = []
        labels_all = []
        scores_vector_all = []

        # 遍历每个batch处理
        for b in range(batch_size):
            batch_pred = preds[b]  # [N, 84]
            boxes = batch_pred[..., :4]      # [N, 4] (cx, cy, w, h)
            cls_conf = batch_pred[..., 4:]   # [N, num_classes]

            # 每类最大置信度
            scores, labels = cls_conf.max(dim=1)  # [N], [N]
            scores_vector = cls_conf

            # 过滤低置信度
            mask = scores > self.conf_threshold
            valid_boxes = boxes[mask]
            valid_scores = scores[mask]
            valid_labels = labels[mask]
            valid_scores_vector = scores_vector[mask]

            if valid_boxes.numel() == 0:
                # 空检测结果占位
                boxes_all.append(torch.zeros((0, 4), device=preds.device, dtype=preds.dtype))
                scores_all.append(torch.zeros((0,), device=preds.device, dtype=preds.dtype))
                labels_all.append(torch.zeros((0,), device=preds.device, dtype=torch.long))
                scores_vector_all.append(torch.zeros((0, self.nc), device=preds.device, dtype=preds.dtype))
                continue

            # cxcywh -> xyxy
            cx, cy, w, h = valid_boxes[:, 0], valid_boxes[:, 1], valid_boxes[:, 2], valid_boxes[:, 3]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

            # NMS
            keep = torchvision.ops.nms(boxes_xyxy, valid_scores, self.iou_threshold)
            boxes_all.append(boxes_xyxy[keep])
            scores_all.append(valid_scores[keep])
            labels_all.append(valid_labels[keep])
            scores_vector_all.append(valid_scores_vector[keep])

        # 整理多batch结果
        results = {
            'boxes': boxes_all,          # list[Tensor], 每个元素 [M, 4]
            'scores': scores_all,        # list[Tensor], 每个元素 [M]
            'labels': labels_all,        # list[Tensor], 每个元素 [M]
            'scores_vector': scores_vector_all  # list[Tensor], 每个元素 [M, num_classes]
        }

        return results

    def visualize_detections(self, img_tensor_input: torch.Tensor, results: Dict[str, List[torch.Tensor]], save_path: str,file_name):
        """
        多batch检测结果可视化，每个batch保存独立文件
        :param img_tensor_input: 输入张量 [B, C, H, W]
        :param results: 检测结果字典
        :param save_path: 基础保存路径（如 "det.jpg" → 生成 "det_batch_0.jpg", "det_batch_1.jpg"）
        """
        batch_size = img_tensor_input.shape[0]
        # 分离梯度（避免影响计算图）
        img_tensor = img_tensor_input.detach()

        # 遍历每个batch可视化
        for b in range(batch_size):
            # 单个batch图像张量 [C, H, W]
            single_batch_tensor = img_tensor[b]
            # 张量转CV2格式 (C, H, W) → (H, W, C)，0-1 → 0-255
            img = np.clip(single_batch_tensor.mul(255).permute(1, 2, 0).cpu().numpy(), 0, 255)
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # 绘制当前batch的检测框
            try:
                boxes = results['boxes'][b]
                scores = results['scores'][b]
                labels = results['labels'][b]

                for idx in range(len(boxes)):
                    # 坐标转整数
                    x1, y1, x2, y2 = map(int, boxes[idx].detach().cpu().numpy())
                    confidence = scores[idx].detach().cpu().numpy()
                    class_id = labels[idx].detach().cpu().numpy()
                    if class_id == len(self.names):
                        class_name = "Other"
                    else:
                        class_name = self.names[int(class_id)]

                    # 绘制边界框
                    cv2.rectangle(img, (x1, y1), (x2, y2), [0, 255, 0], 2)
                    # 绘制标签
                    text = f"{class_name}: {confidence:.2f}"
                    cv2.putText(img, text, (x1, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 2)
            except IndexError:
                print(f"Batch {b}: No detections found")
                continue

            # 生成多batch保存路径
            if save_path is not None:
                # 拆分路径和后缀

                batch_save_path = os.path.join(save_path[b], file_name)
                
                # 保存图像
                try:
                    cv2.imwrite(batch_save_path, img)
                    print(f"Batch {b} detection saved to: {batch_save_path}")
                except Exception as e:
                    print(f"Failed to save batch {b} results: {e}")



    def load_model(self, model_type: str, model_path: str = None):
        """
        加载指定类型的检测模型
        Args:
            model_type: 模型类型，支持'yolov8'/'fasterrcnn'/'ssd300'
            model_path: 模型权重路径（YOLOv8需指定，其他模型用预训练权重）
        """
        model_type = model_type.lower()
        if model_type in self.models:
            print(f"模型{model_type}已加载，跳过重新加载")
            return

        # if model_type == "yolov8":
        #     if not model_path:
        #         raise ValueError("YOLOv8需要指定模型权重路径（如'yolov8n.pt'）")
        #     model = YOLO(model_path)
        #     model.model = model.model.to(self.device)
        #     self.models[model_type] = model
        #     print(f"成功加载YOLOv8模型: {model_path}")

        if "yolo" in model_type :

            

            
            model_yolo =  YOLO(model_path)  # load an official model
            self.models[model_type] = model_yolo.model.to(self.device)  # 加载到指定设备上
            # 模型参数
            self.nc = len(model_yolo.names)  # 类别数量
            self.names = model_yolo.names    # 类别名称映射
            self.ymal = self.models[model_type].yaml # 锚框配置


            # 锚框存储在模型的 Detect 模块中,获取锚框信息

            detect_module = next(m for m in model_yolo.model.modules() if hasattr(m, "anchors"))
            anchors = detect_module.anchors.cpu().numpy().tolist()
            self.anchors = anchors

        elif model_type == "fasterrcnn":
            # 加载预训练的Faster R-CNN（COCO数据集）
            model = fasterrcnn_resnet50_fpn(pretrained=True)
            model.to(self.device)
            model.eval()
            self.models[model_type] = model
            print("成功加载预训练Faster R-CNN模型")

        elif model_type == "ssd300":
            # 加载预训练的SSD-300（COCO数据集）
            weights = SSD300_VGG16_Weights.COCO_V1
            model = ssd300_vgg16(weights=weights)
            model.to(self.device)
            model.eval()
            self.models[model_type] = model
            print("成功加载预训练SSD-300模型")


        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    @torch.no_grad()
    def detect_eval(self, images: torch.Tensor, model_type: str,file_path: Optional[str] = None,file_name='detect.jpg',grad_status=True):
        """
        执行检测并返回统一格式的结果
        Args:
            images: 输入图像张量，shape [B, C, H, W]，已归一化到[0,1]或[0,255]
            model_type: 模型类型（需先通过load_model加载）
        Returns:
            BatchDetectionResult: 批量检测结果
        """
        model_type = model_type.lower()
        if model_type not in self.models:
            raise ValueError(f"模型{model_type}未加载，请先调用load_model")
        images=self._tensor_vailid(images)
        # 图像预处理（适配不同模型的输入要求）
        processed_images = self._preprocess_images(images, model_type)


        # 判断模型和输入的数据类型
        model_dtype = next(self.models[model_type].parameters()).dtype
        if images.dtype !=model_dtype :
            images = images.to(model_dtype)

        # # 模型前向推理
        with torch.set_grad_enabled(grad_status):
            if "yolo" in model_type:
                


                # 判断device
                device = next(self.models[model_type].parameters()).device
                images = images.to(device)
                infer_results = self.models[model_type](images)  
                out = infer_results[0]
                out = out.permute(0, 2, 1).contiguous()  # [B,N,84]
                width, height = images.shape[-1], images.shape[-2]

                # 解码多batch结果
                batch_results = self._decode_yolo_output(out)





            elif model_type in ["fasterrcnn", "ssd300"]:
                # 判断device
                device = next(self.models[model_type].parameters()).device
                processed_images = move_to_gpu_and_cast_dtype(images, device)
                raw_outputs = self._torchvision_inference(processed_images, model_type)
                batch_results = self._decode_output(raw_outputs, model_type)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
        
        # --------------------------
        # 3. 提取每个batch的主类别（置信度最高）
        # --------------------------
        class_names=self._get_class_per_batch(batch_results)  

        if file_path is not None:
            self.visualize_detections(images, batch_results, save_path=file_path,file_name=file_name)
        return batch_results,class_names

    def _preprocess_images(self, images: torch.Tensor, model_type: str) :
        """图像预处理，适配不同模型的输入要求"""
        # if images.ndim != 4:
        #     raise ValueError(f"输入图像必须是[B,C,H,W]的张量，当前shape: {images.shape}")

        # # YOLOv8接受[B,C,H,W]的张量，torchvision模型接受list[Tensor]
        # if model_type in ["fasterrcnn", "ssd300"]:
        #     # torchvision检测模型要求输入为0-255的float32，且为list[Tensor]
        #     processed = []
        #     for img in images:
        #         if img.max() <= 1.0:
        #             img = img * 255.0
        #         processed.append(img.to(torch.float32).to(self.device))
        #     return processed
        # elif  "yolo" in model_type:
        #     # YOLOv8自动处理归一化，直接传入张量即可
        #     return images.to(self.device)
        # else:
        #     return images.to(self.device)
        
        return images.to(self.device)

    def _yolov8_inference(self, images: torch.Tensor) -> list:
        """YOLOv8推理，返回原始输出"""
        results = self.models["yolov8"](images,
                                         conf=self.conf_threshold,
                                           iou=self.iou_threshold,
                                             verbose=False,
                                             imgsz=self.image_size )
        return results

    def _torchvision_inference(self, images: List[torch.Tensor], model_type: str) -> list:
        """TorchVision模型（Faster R-CNN/SSD-300）推理"""
        model = self.models[model_type]
        outputs = model(images)
        return outputs

    def _decode_output(self, raw_outputs: list, model_type: str) :
        """解析不同模型的原始输出为统一的DetectionResult"""
        



        if model_type == "yolov8":
            result = self._decode_yolov_infer_output(raw_outputs)
        elif model_type == "fasterrcnn" or model_type == "ssd300":
            result = self._decode_torchvision_output(raw_outputs, model_type)
        # elif model_type == "ssd300":
        #     result = self._decode_torchvision_output(raw_outputs, model_type)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")


        return result

    def _decode_yolov_infer_output(self, raw_outs) -> Dict[str, List[torch.Tensor]]:
        """
        解析YOLOv8的输出结果，返回与_decode_yolo_output一致的字典格式
        :param raw_out: YOLOv8的单批次输出对象
        :return: results 字典（每个key对应长度=1的列表，适配单batch）
        """
        # 提取检测框、置信度、标签
        length=len(raw_outs)
        boxes_list = []
        scores_list = []
        labels_list = []
        scores_vector_list = []
        for i in range(length):
            raw_out = raw_outs[i]
            if len(raw_out.boxes) > 0:
                boxes = torch.tensor(raw_out.boxes.xyxy.cpu()).to(self.device)
                scores = torch.tensor(raw_out.boxes.conf.cpu()).to(self.device)
                labels = torch.tensor(raw_out.boxes.cls.cpu(), dtype=torch.long).to(self.device)
                
                # 生成scores_vector：YOLOv8的boxes无类别得分向量，需从模型输出推导
                # 若有probs则用probs，否则生成全0向量（形状[M, num_classes]）
                if hasattr(raw_out, 'probs') and raw_out.probs is not None:
                    scores_vector = raw_out.probs.unsqueeze(0).repeat(boxes.shape[0], 1).to(self.device)
                else:
                    scores_vector = torch.zeros((boxes.shape[0], self.nc), device=self.device)
            else:
                # 空检测结果占位
                boxes = torch.zeros((0, 4), device=self.device)
                scores = torch.zeros((0,), device=self.device)
                labels = torch.zeros((0,), device=self.device, dtype=torch.long)
                scores_vector = torch.zeros((0, self.nc), device=self.device)


            boxes_list.append(boxes) 
            scores_list.append(scores) 
            labels_list.append(labels) 
            scores_vector_list.append(scores_vector)
            # 整理为与_decode_yolo_output一致的字典格式（长度为1的列表，代表单batch）
        results = {
            'boxes': boxes_list,          # list[Tensor], 长度=B，元素[M,4],M为每个批次里面检测框的数量
            'scores': scores_list,        # list[Tensor], 长度=B，元素[M]
            'labels': labels_list,        # list[Tensor], 长度=B，元素[M]
            'scores_vector': scores_vector_list  # list[Tensor], 长度=B，元素[M, num_classes]
        }
        return results

    def _decode_torchvision_output(self, raw_outs: dict, model_type: str) -> Dict[str, List[torch.Tensor]]:
        """
        解析Faster R-CNN/SSD-300的输出结果，返回与_decode_yolo_output一致的字典格式
        :param raw_out: TorchVision模型的单批次输出字典
        :param model_type: 模型类型（fasterrcnn/ssd300）
        :return: results 字典（每个key对应长度=1的列表，适配单batch）
        """
        boxes_list = []
        scores_list = []
        labels_list = []
        scores_vector_list = []
        length=len(raw_outs)
        for i in range(length):
            raw_out=raw_outs[i]
            # 过滤低置信度结果
            boxes = raw_out["boxes"]
            scores = raw_out["scores"]
            
            # label 映射
            map_labels=   map_coco_to_yolo_labels( raw_out["labels"]) 
            labels =  map_labels  

            mask = scores > self.conf_threshold
            boxes = boxes[mask].to(self.device)
            scores = scores[mask].to(self.device)
            labels = labels[mask].to(self.device)

            # 执行NMS
            if len(boxes) > 0:
                keep = torchvision.ops.nms(boxes, scores, self.iou_threshold)
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
            else:
                boxes = torch.zeros((0, 4), device=self.device)
                scores = torch.zeros((0,), device=self.device)
                labels = torch.zeros((0,), device=self.device, dtype=torch.long)

            # 生成scores_vector：TorchVision检测模型无原生类别得分向量，这里生成one-hot或全0
            # 方案1：生成全0向量（保持与原逻辑一致）
            scores_vector = torch.zeros((boxes.shape[0], self.nc), device=self.device)
            # 方案2：生成one-hot向量（可选，根据labels标记对应类别为1）
            # if boxes.shape[0] > 0:
            #     scores_vector[torch.arange(boxes.shape[0]), labels] = 1.0

            # 整理为与_decode_yolo_output一致的字典格式（长度为1的列表，代表单batch）

            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)
            scores_vector_list.append(scores_vector)
        results = {
            'boxes':boxes_list,          # list[Tensor], 长度=1，元素[M,4]
            'scores': scores_list,        # list[Tensor], 长度=1，元素[M]
            'labels': labels_list,        # list[Tensor], 长度=1，元素[M]
            'scores_vector': scores_vector_list  # list[Tensor], 长度=1，元素[M, num_classes]
        }
        return results

# class DetectionResult:
#     """
#     统一的检测结果结构体
#     Attributes:
#         boxes: 检测框坐标，格式为xyxy，shape [M, 4]，M为检测目标数
#         scores: 检测置信度，shape [M]
#         labels: 检测类别标签，shape [M]
#         scores_vector: 每个目标的类别得分向量（可选），shape [M, num_classes]
#         model_type: 检测模型类型（yolov8/fasterrcnn/ssd300）
#     """
#     boxes: torch.Tensor
#     scores: torch.Tensor
#     labels: torch.Tensor
#     scores_vector: Optional[torch.Tensor] = None
#     model_type: str = ""

#     def __post_init__(self):
#         """初始化后校验张量维度"""
#         if self.boxes.ndim != 2 or self.boxes.shape[1] != 4:
#             raise ValueError(f"boxes必须是[M,4]的张量，当前shape: {self.boxes.shape}")
#         if self.scores.ndim != 1 or self.scores.shape[0] != self.boxes.shape[0]:
#             raise ValueError(f"scores必须是[M]的张量且长度与boxes一致，当前shape: {self.scores.shape}")
#         if self.labels.ndim != 1 or self.labels.shape[0] != self.boxes.shape[0]:
#             raise ValueError(f"labels必须是[M]的张量且长度与boxes一致，当前shape: {self.labels.shape}")
#         if self.scores_vector is not None:
#             if self.scores_vector.ndim != 2 or self.scores_vector.shape[0] != self.boxes.shape[0]:
#                 raise ValueError(f"scores_vector必须是[M, num_classes]的张量，当前shape: {self.scores_vector.shape}")

# @dataclass
# class BatchDetectionResult:
#     """批量检测结果结构体"""
#     results: List[DetectionResult]  # 每个元素对应一个batch的检测结果
#     batch_size: int

#     def __getitem__(self, idx: int) -> DetectionResult:
#         return self.results[idx]

#     def __len__(self) -> int:
#         return self.batch_size
    




























# # --------------------------
# # 测试示例
# # --------------------------
# if __name__ == "__main__":
#     attack_config_path=r'models/attack_config.yaml'
#     adv_config=load_yaml_config(attack_config_path)
#     # 初始化检测器
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     detector = ObjectDetection(
#     )
#     detector.load_model( model_type="yolov11",
#         model_path=r"models/yolo11n.pt")
    


#     BATCH_SIZE = 2
#     IMG_SIZE = adv_config["experiment_params"]["image_size"]  
#     IMG_ROOT=adv_config["experiment_params"]["dataset_path"]
#     # --------------------------
#     # 2. 验证集预处理（无数据增强！）
#     # --------------------------
#     # 注意：验证集仅做resize、中心裁剪、归一化，禁止随机增强（保证评估公平）
#     val_transform = transforms.Compose([
#         ResizeMaxEdge(max_edge_size=IMG_SIZE),  # 最大边缩放到256
#         PadToFixedSize(target_size=IMG_SIZE),  # 填充到256×256（中心对齐）
#         transforms.ToTensor(),  # 转为张量（0-1）
#     ])



#     # --------------------------
#     # 3. 加载验证集
#     # --------------------------
#     # ImageFolder自动按文件夹名称分配类别标签（0-999）
#     img_dataset = CustomImageDataset(
#         root_dir=IMG_ROOT,
#         transform=val_transform
#     )


#     img_loader = DataLoader(
#         img_dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=True,  
#         num_workers=adv_config["experiment_params"]["num_workers"],
#         pin_memory=True
#     )



#     # --------------------------
#     # 4. 迭代（对抗样本生成）
#     # --------------------------


#     exp_root=adv_config["experiment_params"]["experiment_path"]
#     # 获取图片文件名,去除后缀

#     os.makedirs(exp_root,exist_ok=True) 
#     pbar = tqdm(enumerate(img_loader), total=len(img_loader), desc="Processing images", unit="batch")
#     for batch_idx, (images, images_path) in pbar:


#         # 1. 测试多batch numpy输入 [B=2, H=640, W=640, C=3]
       
#         results, class_names = detector.detect(
#             img=images,
#             file_path="./detection_result.jpg",  # 保存为 detection_result_batch_0.jpg / batch_1.jpg
#             grad_status=False
#         )

#         # 2. 解析多batch结果
#         for b in range(len(class_names)):
#             print(f"\n=== Batch {b} ===")
#             print(f"主类别: {class_names[b]}")
#             print(f"检测到目标数: {len(results['boxes'][b])}")
#             if len(results['boxes'][b]) > 0:
#                 print(f"类别列表: {[detector.names[int(l)] for l in results['labels'][b].cpu().numpy()]}")






# for test
if __name__ == "__main__":

    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from tqdm import tqdm
    import argparse  # 导入argparse库
    from adv_attack.util import *
    root_path=os.path.dirname(__file__)
    attack_config_path=r"models/attack_config.yaml"
    adv_config=load_yaml_config(attack_config_path)



    

    BATCH_SIZE = 2
    IMG_SIZE = adv_config["experiment_params"]["image_size"]  
    IMG_ROOT=r"exp/optim_latent"
    # --------------------------
    # 2. 验证集预处理（无数据增强！）
    # --------------------------
    # 注意：验证集仅做resize、中心裁剪、归一化，禁止随机增强（保证评估公平）
    val_transform = transforms.Compose([
        ResizeMaxEdge(max_edge_size=IMG_SIZE),  # 最大边缩放到256
        PadToFixedSize(target_size=IMG_SIZE),  # 填充到256×256（中心对齐）
        transforms.ToTensor(),  # 转为张量（0-1）
    ])



    # --------------------------
    # 3. 加载验证集
    # --------------------------
    # ImageFolder自动按文件夹名称分配类别标签（0-999）
    img_dataset = CustomFolderDataset(
        root_dir=IMG_ROOT,
        transform=val_transform
    )


    img_loader = DataLoader(
        img_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  
        num_workers=adv_config["experiment_params"]["num_workers"],
        pin_memory=True
    )



    # 1. 初始化多模型检测器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = ObjectDetection(device=device)
                                            

    # 2. 加载检测模型（可加载多个，按需选择）
    # 加载YOLOv8（需指定权重路径，如yolov8n.pt，可从ultralytics官网下载）
    detector.load_model(model_type="yolov8",model_path=r"models/yolov8n.pt")
    # 加载Faster R-CNN（自动使用COCO预训练权重）
    detector.load_model(model_type="fasterrcnn")
    # 加载SSD300（自动使用COCO预训练权重）
    detector.load_model(model_type="ssd300")

    # 3. 构造测试输入（模拟批量图像，shape [B, C, H, W]，值范围0-1）
    batch_size = 2




    exp_root=adv_config["experiment_params"]["experiment_path"]
    # 获取图片文件名,去除后缀

    os.makedirs(exp_root,exist_ok=True) 
    pbar = tqdm(enumerate(img_loader), total=len(img_loader), desc="Processing images", unit="batch")
    for batch_idx, (folder_name, img_tensors, img_names) in pbar:
    



        test_images=img_tensors[0]

    # 4. 执行检测（分别用不同模型检测）
    # --------------------------
    # 4.1 YOLOv8检测
    # --------------------------
        print("\n=== YOLOv8 检测结果 ===")
        yolo_results = detector.detect_eval(images=test_images, model_type="yolov8")
        # 解析批量结果
        for batch_idx in range(len(yolo_results["boxes"])):
            single_result = yolo_results
            print(f"\nBatch {batch_idx} 检测结果:")


        # --------------------------
        # 4.2 Faster R-CNN检测
        # --------------------------
        print("\n=== Faster R-CNN 检测结果 ===")
        frcnn_results = detector.detect_eval(images=test_images, model_type="fasterrcnn")
        for batch_idx in range(len(frcnn_results["boxes"])):
            single_result = frcnn_results
            print(f"\nBatch {batch_idx} 检测结果:")


        # --------------------------
        # 4.3 SSD300检测
        # --------------------------
        print("\n=== SSD300 检测结果 ===")
        ssd_results = detector.detect_eval(images=test_images, model_type="ssd300")
        for batch_idx in range(len(ssd_results["boxes"])):
            single_result = ssd_results
            print(f"\nBatch {batch_idx} 检测结果:")




