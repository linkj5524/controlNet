import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from share import *

import numpy as np
from PIL import Image
import cv2
import einops
import numpy as np
import torch
import random
import yaml
import os
from tqdm import tqdm
import sys
from pytorch_lightning import seed_everything
import time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import config
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from adv_attack import *
from adv_attack.attack_class import *
from adv_attack.util import *
import sys

from tqdm import tqdm
import argparse  # 导入argparse库



# 1. 创建参数解析器
parser = argparse.ArgumentParser(description="Adversarial Attack Main Program")  # 程序描述

# 2. 添加命令行参数

parser.add_argument('--attack_config_path', type=str, 
                    default="models/attack_config.yaml",
                      help="attack config path")


# 3. 解析命令行参数
args = parser.parse_args()

if __name__ == '__main__':


    root_path=os.path.dirname(__file__)
    attack_config_path=args.attack_config_path
    adv_config=load_yaml_config(attack_config_path)



    

    BATCH_SIZE = adv_config["experiment_params"]["batch_size"]
    IMG_SIZE = adv_config["experiment_params"]["image_size"]  
    # IMG_ROOT=adv_config["experiment_params"]["dataset_path"]
    IMG_ROOT=r"./exp/test"
    BATCH_SIZE=2
    # --------------------------
    # 2. 验证集预处理（无数据增强！）
    # --------------------------
    # 注意：验证集仅做resize、中心裁剪、归一化，禁止随机增强（保证评估公平）
    val_transform = transforms.Compose([
        ResizeMaxEdge(max_edge_size=IMG_SIZE), 
        PadToFixedSize(target_size=IMG_SIZE),  
        transforms.ToTensor(),  # 转为张量
    ])



    # --------------------------
    # 3. 加载验证集
    # --------------------------
    target_list=["origin.jpg","adv_example.pt"]
    # ImageFolder自动按文件夹名称分配类别标签（0-999）
    img_dataset = CustomFolderDataset(
        root_dir=IMG_ROOT,
        transform=val_transform,
        target_image_name_list=target_list
    )


    img_loader = DataLoader(
        img_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  
        num_workers=adv_config["experiment_params"]["num_workers"],
        pin_memory=True,
        # collate_fn=custom_collate_fn
    )



    # 1. 初始化多模型检测器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = ObjectDetection(device=device)
                                            

    # 2. 加载检测模型（可加载多个，按需选择）
    # 加载YOLOv8（需指定权重路径，如yolov8n.pt，可从ultralytics官网下载）
    detector.load_model(model_type="yolov8",model_path=r"./models/yolov8m.pt")
    # 加载Faster R-CNN（自动使用COCO预训练权重）
    detector.load_model(model_type="fasterrcnn")
    # 加载SSD300（自动使用COCO预训练权重）
    detector.load_model(model_type="ssd300")
    detector.load_model(model_type="yolov11",model_path=r"./models/yolo11m.pt")   
    # 3. 构造测试输入（模拟批量图像，shape [B, C, H, W]，值范围0-1）
    batch_size = 1




    exp_root=adv_config["experiment_params"]["experiment_path"]
    exp_root=r"exp/test_eval"
    # 获取图片文件名,去除后缀

    os.makedirs(exp_root,exist_ok=True) 
    pbar = tqdm(enumerate(img_loader), total=len(img_loader), desc="Processing images", unit="batch")
    for batch_idx, (folder_name, img_tensors, img_names) in pbar:
        
        file_roots=[os.path.join(exp_root,name) for name in folder_name]
        for file_root_i in file_roots:
            os.makedirs(file_root_i,exist_ok=True)

        origin_images=img_tensors[1]
        adv_images=img_tensors[0]

        adv_images=torch.load(r'exp\test\000000000785\adv_example.pt')
        # yolov11 初始化，作为参考
        result_gt,object_class=detector.detect(img=origin_images, model_type="yolov11", file_path=file_roots,file_name="origin_yolov11.jpg")

        result_gt,object_class=filter_max_box_per_batch(result_gt,object_class) 


        result_gt11,object_class111=detector.detect(img=adv_images, model_type="yolov11", file_path=file_roots,file_name="adv_yolov11.jpg")
        #评估
    







        print("\n=== YOLOv8 检测结果 ===")
        yolo_results = detector.detect_eval(images=adv_images, model_type="yolov8", file_path=file_roots,file_name="adv_yolov8.jpg")

        # match_yolo=match_detection_boxes(result_gt,yolo_results)

        # --------------------------
        # 4.2 Faster R-CNN检测
        # --------------------------
        print("\n=== Faster R-CNN 检测结果 ===")
        frcnn_results = detector.detect_eval(images=adv_images, model_type="fasterrcnn",file_path=file_roots,file_name='adv_fasterrcnn.jpg')



        # --------------------------
        # 4.3 SSD300检测
        # --------------------------
        print("\n=== SSD300 检测结果 ===")
        ssd_results = detector.detect_eval(images=adv_images, model_type="ssd300",file_path=file_roots,file_name='adv_ssd300.jpg')



        print("\n=== YOLOv8 检测结果 ===")
        yolo_results = detector.detect_eval(images=origin_images, model_type="yolov8", file_path=file_roots,file_name="origin_yolov8.jpg")

        match_yolo=match_detection_boxes(result_gt,yolo_results)

        # --------------------------
        # 4.2 Faster R-CNN检测
        # --------------------------
        print("\n=== Faster R-CNN 检测结果 ===")
        frcnn_results = detector.detect_eval(images=origin_images, model_type="fasterrcnn",file_path=file_roots,file_name='origin_fasterrcnn.jpg')



        # --------------------------
        # 4.3 SSD300检测
        # --------------------------
        print("\n=== SSD300 检测结果 ===")
        ssd_results = detector.detect_eval(images=origin_images, model_type="ssd300",file_path=file_roots,file_name='origin_ssd300.jpg')

