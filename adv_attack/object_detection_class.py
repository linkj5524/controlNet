
import os
import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np

'''
本类如果使用了yolo 来检测，需要注意通道的排列，需要注意输入的数据维度，更多详细的参数以及用法
可以参考官方文档： https://docs.ultralytics.com/zh/modes/predict.html


'''




'''
class ObjectDetection:
表示一个物体检测类，可以加载不同类型的模型，并对输入的图像进行物体检测。


'''

class ObjectDetection:
    def __init__(self,model_type,model_path,device:torch.device=torch.device("cpu"),**args):
        self.model_type = model_type
        self.model_path = model_path
        self.device = device    
        self.args = args
        # 解析args参数
        for key, value in args.items():
            setattr(self, key, value)  # 使用setattr动态设置属性
        conf_threshold = args.get("conf_threshold",0.25)
        iou_threshold = args.get("iou_threshold",0.7)
        strides = args.get("strides",[8, 16, 32])
        self.strides = strides

        if self.model_type == "yolov11":
            from ultralytics import YOLO
            model_yolo =  YOLO(model_path)  # load an official model
            self.model = model_yolo.model.to(self.device)  # 加载到指定设备上
            # 模型参数
            self.nc = len(model_yolo.names)  # 类别数量
            self.names = model_yolo.names    # 类别名称映射
            self.ymal = self.model.yaml # 锚框配置


            # 锚框存储在模型的 Detect 模块中,获取锚框信息

            detect_module = next(m for m in model_yolo.model.modules() if hasattr(m, "anchors"))
            anchors = detect_module.anchors.cpu().numpy().tolist()
            self.anchors = anchors





        elif self.model_type == "yolov5":
            from yolov5 import YOLOv5
            self.model = YOLOv5(self.model_path, device=self.device)
        else:
            raise ValueError("Unsupported model type")
                # 获取模型元数据

        
        # 推理参数
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect(self, img,file_path=None,grad_status=False):
        # 输入的图像默认是RGB
        # 确保输入符合要求
        if "yolo" in self.model_type:
            if isinstance(img, np.ndarray) :
                # 默认输入是RGB，8位，0-255
                # img = img[..., [2, 1, 0]]  # RGB转BGR（YOLO通常期望BGR输入）
            
                if img.ndim == 4:
                    img = img[0]  # 处理批量输入，取第一张图
                
                # 确保输入是PyTorch张量并在正确设备上

                img = torch.from_numpy(img).to(self.device).float() / 255.0  # 归一化到0-1
                # 调整维度：(H, W, C) -> (1, C, H, W)（YOLO需要的输入格式）
                img_tensor = img.permute(2, 0, 1).unsqueeze(0)
            elif isinstance(img,  torch.Tensor) :
                img_tensor=img
            else:
                raise ValueError("Unsupported input type")

        # 模型推理
        if "yolo" in self.model_type:
            if self.model_type == "yolov11":
                # 不同模型需要调用不同的postprocess函数
                # 对于Ultralytics YOLO，使用device参数确保在正确设备上运行
                with torch.set_grad_enabled(grad_status):  # 开启梯度信息
                    infer_results = self.model(img_tensor)  # 推理
                    # 后处理
                    # 计算tensor的尺寸
                    out = infer_results[0]  # 取第一个元素
                    out = out.permute(0, 2, 1).contiguous()  # [B,N,84]
                    width, height = img_tensor.shape[-1], img_tensor.shape[-2]


                    results = self._decode_yolo_output(out)
                    
            else:
                #抛出异常
                raise ValueError("Unsupported model type for postprocess")




        else:
            raise ValueError("Unsupported model type")
        
        # 保存结果
        if file_path is not None:
            # 判断results是否为空
            if len(results) > 0:
                self.visualize_detections(img_tensor, results, save_path=file_path)
            else :
                print("No detections found.")
                return None
        
        return results
    
    def detect_return_dict(self, img,file_path=None,grad_status=False):
        # 输入的图像默认是RGB
        # 确保输入符合要求
        if "yolo" in self.model_type:
            if isinstance(img, np.ndarray) :
                # 默认输入是RGB，8位，0-255
                # img = img[..., [2, 1, 0]]  # RGB转BGR（YOLO通常期望BGR输入）
            
                if img.ndim == 4:
                    img = img[0]  # 处理批量输入，取第一张图
                
                # 确保输入是PyTorch张量并在正确设备上

                img = torch.from_numpy(img).to(self.device).float() / 255.0  # 归一化到0-1
                # 调整维度：(H, W, C) -> (1, C, H, W)（YOLO需要的输入格式）
                img_tensor = img.permute(2, 0, 1).unsqueeze(0)
            elif isinstance(img,  torch.Tensor) :
                img_tensor=img
            else:
                raise ValueError("Unsupported input type")

        # 模型推理
        if "yolo" in self.model_type:
            if self.model_type == "yolov11":
                # 不同模型需要调用不同的postprocess函数
                # 对于Ultralytics YOLO，使用device参数确保在正确设备上运行
                with torch.set_grad_enabled(grad_status):  # 开启梯度信息
                    infer_results = self.model(img_tensor)  # 推理
                    # 后处理
                    # 计算tensor的尺寸
                    out = infer_results[0]  # 取第一个元素
                    out = out.permute(0, 2, 1).contiguous()  # [B,N,84]
                    width, height = img_tensor.shape[-1], img_tensor.shape[-2]


                    results = self._decode_yolo_output(out)
            else:
                #抛出异常
                raise ValueError("Unsupported model type for postprocess")




        else:
            raise ValueError("Unsupported model type")
        
        # 保存结果
        if file_path is not None:
            
            self.visualize_detections(img_tensor, results, save_path=file_path)
        
        return results



# 自定义后处理过程，防止计算图断开


    def _postprocess_output(self, outputs, **preprocess_info):
        """
        后处理模型输出
        :param outputs: 模型输出
        :param preprocess_info: 预处理信息,主要是图像的尺寸信息，这里使用输入的尺寸计算
        :return: 解析后的检测结果
        """
        # 解析预处理信息
        for key, value in preprocess_info.items():
            setattr(self, key, value)  # 使用setattr动态设置属性
        
        # 提取预测结果 (batch, num_dets, 6) -> (x1, y1, x2, y2, conf, cls)
        predictions = outputs[0]  # 取第一个批次
        
    # 应用置信度阈值（使用PyTorch操作）
        conf_mask = predictions[:, 4] >= self.conf_threshold
        predictions = predictions[conf_mask]
        
        if predictions.numel() == 0:
            return []
        
        # 分离边界框、置信度和类别（仍为张量）
        boxes = predictions[:, :4]  # (x1, y1, x2, y2) - 已经是原始图像坐标
        confidences = predictions[:, 4]
        classes = predictions[:, 5].long()  # 类别ID转换为长整数
        
        # 确保边界框在输入图像范围内（仅做边界限制，不缩放）
        assert self.width is not None and self.height is not None, "Input image size is not provided"
        orig_w, orig_h = self.width, self.height
        boxes[:, 0] = torch.clamp(boxes[:, 0], 0, orig_w)  # x1
        boxes[:, 1] = torch.clamp(boxes[:, 1], 0, orig_h)  # y1
        boxes[:, 2] = torch.clamp(boxes[:, 2], 0, orig_w)  # x2
        boxes[:, 3] = torch.clamp(boxes[:, 3], 0, orig_h)  # y2
        
        # 应用非极大值抑制（使用PyTorch的内置函数）
        indices = torch.ops.torchvision.nms(
            boxes, 
            confidences, 
            self.iou_threshold
        )
        
        # 应用NMS结果
        boxes = boxes[indices]
        confidences = confidences[indices]
        classes = classes[indices]
        
        # 整理结果（保持张量形式）
        results = []
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i]
            results.append({
                'box': (x1, y1, x2, y2),  # 原始图像坐标，张量形式
                'confidence': confidences[i],  # 张量形式的置信度
                'class_id': classes[i],  # 张量形式的类别ID
                'class_name': self.names[int(classes[i].item())]  # 类别名称
            })
        
        return results
    def _decode_boxes(self, pred, anchors, stride):
        """将原始偏移量转换为实际坐标（xywh）"""
        batch_size, num_anchors, height, width, _ = pred.shape
        device = pred.device
        
        # 生成网格坐标（每个网格的左上角坐标）
        grid_x = torch.arange(width, device=device).repeat(height, 1).unsqueeze(2)
        grid_y = torch.arange(height, device=device).repeat(width, 1).t().unsqueeze(2)
        grid = torch.cat((grid_x, grid_y), 2).repeat(1, 1, num_anchors).unsqueeze(0)
        
        # 解析原始输出（dx, dy, dw, dh, obj_conf, cls_conf*）
        dx = pred[..., 0]
        dy = pred[..., 1]
        dw = pred[..., 2]
        dh = pred[..., 3]
        obj_conf = pred[..., 4]
        cls_conf = pred[..., 5:]
        
        # 坐标转换：偏移量 -> 实际像素坐标（xywh）
        x = (dx * 2 - 0.5 + grid[..., 0]) * stride
        y = (dy * 2 - 0.5 + grid[..., 1]) * stride
        w = (dw * 2) ** 2 * anchors[..., 0].unsqueeze(1).unsqueeze(1)
        h = (dh * 2) ** 2 * anchors[..., 1].unsqueeze(1).unsqueeze(1)
        
        # 重塑形状：(batch, anchors, h, w, ...) -> (batch, total_anchors, ...)
        boxes = torch.stack([x, y, w, h], dim=-1).view(batch_size, -1, 4)
        obj_conf = obj_conf.view(batch_size, -1, 1)
        cls_conf = cls_conf.view(batch_size, -1, self.num_classes)
        
        return boxes, obj_conf, cls_conf


    # 解析YOLOv11模型输出，anchor free 版本
    def _decode_yolo_output(self, raw_outs):
        """
        解析 YOLOv11 模型输出，返回 tensor 而非列表
        输入: raw_outs -> list/tuple of layer outputs or a single tensor
            每层输出形状 [B, N_layer, 84] 或整体 [B, N, 84]
        输出: results -> dict(batch-wise tensor)
            'boxes': [B, M, 4] xyxy
            'scores': [B, M]
            'labels': [B, M]
        """
        batch_size = raw_outs[0].shape[0] if isinstance(raw_outs, (list, tuple)) else raw_outs.shape[0]

        # 拼接多层输出
        if isinstance(raw_outs, (list, tuple)):
            preds = torch.cat(raw_outs, dim=1)  # [B, N, 84]
        else:
            preds = raw_outs  # [B, N, 84]

        boxes_all = []
        scores_all = []
        labels_all = []

        boxes = preds[..., :4]      # [B, N, 4]
        cls_conf = preds[..., 4:]   # [B, N, num_classes]

        for b in range(batch_size):
            # 每类最大置信度
            scores, labels = cls_conf[b].max(dim=1)  # [N], [N]

            # 过滤低置信度
            mask = scores > self.conf_threshold
            valid_boxes = boxes[b][mask]
            valid_scores = scores[mask]
            valid_labels = labels[mask]

            if valid_boxes.numel() == 0:
                # 如果没有有效框，用空 tensor 占位
                boxes_all.append(torch.zeros((0, 4), device=preds.device, dtype=preds.dtype))
                scores_all.append(torch.zeros((0,), device=preds.device, dtype=preds.dtype))
                labels_all.append(torch.zeros((0,), device=preds.device, dtype=torch.long))
                continue

            # 直接计算 xyxy
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

        # 拼接 batch
        results = {
            'boxes': boxes_all,    # list of tensor per batch
            'scores': scores_all,
            'labels': labels_all
        }

        return results


    
    # visualize_detections函数，可视化检测结果
    def visualize_detections(self, img_tensor_input, results,save_path):
        """
        可视化检测结果  
        :param img: 原始图像
        :param results: 检测结果
        :param kwargs: 可视化参数，包括类别颜色、字体大小等
        :return: 可视化后的图像
        """

        img_tensor=img_tensor_input.detach()
        #将输入转化为cv2格式，从tensor转化为numyp格式
        if img_tensor.ndim == 4:
            img_tensor = img_tensor[0]  # 处理批量输入，取第一张图
        
        img =np.clip( img_tensor.mul(255).permute(1, 2, 0).numpy(), 0, 255)
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 绘制边界框
        count=0
        try:
            for batch in range(len(results['boxes'])):
                for index in range(len(results['boxes'][batch])):
                # 将tensor detach 断开
                    x1, y1, x2, y2 = results['boxes'][batch][index].detach().cpu().numpy()
                    confidence = results['scores'][batch][index].detach().cpu().numpy()
                    class_id = results['labels'][batch][index].detach().cpu().numpy()
                    class_name = self.names[int(class_id)]

                    
                    # 绘制边界框
                    # 坐标转换为整数
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), [0, 255, 0], 2)
                    
                    # 绘制标签
                    text = f"{class_name}: {confidence:.2f}"
                    cv2.putText(img, text, (x1, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 2)
        except:
            print("No detections found")
        # 保存结果
        if save_path is not None:
            try:
                cv2.imwrite(save_path, img)
            except:
                print(f"Failed to save detection results to {save_path}")
            
            