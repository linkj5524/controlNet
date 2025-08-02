
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



    def detect(self,img,imgsize=320):
        # 默认输入是RGB，8位，0-255
        if  "yolo" in self.model_type :
            img = img[..., [2, 1, 0]]
        if img.ndim == 4:
            
            img=img[0]
        if  "yolo" in self.model_type :
            results=self.model(np.ascontiguousarray(img),imgsz=imgsize)
        else:
            raise ValueError("Unsupported model type")
        bbox_xyxy = []
        confidences = []
        class_ids = []
        for result in results:
            bbox_xyxy.append(result.boxes.xyxy.numpy() if result.boxes.xyxy.device == torch.device("cpu") else result.boxes.xyxy.cpu().numpy())  # 转换为NumPy数组
            confidences.append(result.boxes.conf.numpy() if result.boxes.conf.device == torch.device("cpu") else result.boxes.conf.cpu().numpy())   
            class_ids.append(result.boxes.cls.numpy().astype(int) if result.boxes.cls.device == torch.device("cpu") else result.boxes.cls.cpu().numpy().astype(int))
            result.save(filename=f"detection_result_{class_ids[0]}.jpg")
        return bbox_xyxy, confidences, class_ids                                 



