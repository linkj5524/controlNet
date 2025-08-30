
import torch
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
    def __init__(self,model_type,model_path,class_names,device:torch.device=torch.device("cpu")):
        self.model_type = model_type
        self.model_path = model_path
        self.class_names = class_names
        self.device = device    

        if self.model_type == "yolov11":
            from ultralytics import YOLO
            self.model =  YOLO(model_path)  # load an official model
        elif self.model_type == "yolov5":
            from yolov5 import YOLOv5
            self.model = YOLOv5(self.model_path, device=self.device)
        else:
            raise ValueError("Unsupported model type")



#     def detect(self,img,imgsize=320):
#         # 默认输入是RGB，8位，0-255
#         if  "yolo" in self.model_type :
#             img = img[..., [2, 1, 0]]
#         if img.ndim == 4:
            
#             img=img[0]
#         if  "yolo" in self.model_type :
#             results=self.model(np.ascontiguousarray(img),imgsz=imgsize)
#         else:
#             raise ValueError("Unsupported model type")
#         bbox_xyxy = []
#         confidences = []
#         class_ids = []  
#         for result in results:
#             bbox_xyxy.append(result.boxes.xyxy.numpy() if result.boxes.xyxy.device == torch.device("cpu") else result.boxes.xyxy.cpu().numpy())  # 转换为NumPy数组
#             confidences.append(result.boxes.conf.numpy() if result.boxes.conf.device == torch.device("cpu") else result.boxes.conf.cpu().numpy())   
#             class_ids.append(result.boxes.cls.numpy().astype(int) if result.boxes.cls.device == torch.device("cpu") else result.boxes.cls.cpu().numpy().astype(int))
#             result.save(filename=f"detection_result_{class_ids[0]}.jpg")
#         return bbox_xyxy, confidences, class_ids                                 



    def detect(self, img, imgsize=320,file_path=None):
        # 输入的图像默认是RGB
        if "yolo" in self.model_type:
            if isinstance(img, np.ndarray) :
                # 默认输入是RGB，8位，0-255
                # img = img[..., [2, 1, 0]]  # RGB转BGR（YOLO通常期望BGR输入）
            
                if img.ndim == 4:
                    img = img[0]  # 处理批量输入，取第一张图
                
                # 确保输入是PyTorch张量并在正确设备上

                img = torch.from_numpy(img).to(self.device).float() / 255.0  # 归一化到0-1
                # 调整维度：(H, W, C) -> (1, C, H, W)（YOLO需要的输入格式）
                img = img.permute(2, 0, 1).unsqueeze(0)
            elif isinstance(img,  torch.Tensor) :
                img=img
            else:
                raise ValueError("Unsupported input type")

        # 模型推理
        if "yolo" in self.model_type:
            # 对于Ultralytics YOLO，使用device参数确保在正确设备上运行
            results = self.model(img, imgsz=imgsize, device=self.device)
        else:
            raise ValueError("Unsupported model type")
        
        # 提取结果并保持为PyTorch张量（不转换为NumPy）
        bbox_xyxy = []
        confidences = []
        class_ids = []
        
        for result in results:
            # 直接获取PyTorch张量，不转换为NumPy
            bbox_xyxy.append(result.boxes.xyxy.to(self.device))  # 边界框
            confidences.append(result.boxes.conf.to(self.device))  # 置信度
            class_ids.append(result.boxes.cls.to(self.device).long())  # 类别ID（转为长整数）
            
            # 将结果绘制到原始图像上
            annotated_img = result.plot()
            
            # 转换回BGR格式用于OpenCV显示
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
            # 保存结果（可选）
            if file_path is not None:

                cv2.imwrite(file_path, annotated_img)
            else:
                raise ValueError("Unsupported input type")
        
        return bbox_xyxy, confidences, class_ids