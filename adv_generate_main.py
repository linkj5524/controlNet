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





if __name__ == '__main__':
    # --------------------------
    # 1. 基础配置
    # --------------------------

    root_path=os.path.dirname(__file__)

    model_path=os.path.join(os.path.join(root_path,'models'),'yolo11n.pt')
    sam_path=os.path.join(os.path.join(root_path,'models'),'sam_vit_h_4b8939.pth')
    controlNet_model_path=os.path.join(os.path.join(root_path,'models'),'control_sd15_canny.pth')
    blip_model_path=os.path.join(os.path.join(root_path,'models'),'Salesforceblip_image_captioning_large')
    attack = ADV_ATTACK(device=torch.device("cuda"),model_path=controlNet_model_path,
                        model_path_object_detection=model_path,sam_model_type='vit_h',
                        sam_checkpoint_path=sam_path,
                        captioner_model_name=blip_model_path)
    
    
    


    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 2 
    IMG_SIZE = 128  
    IMG_ROOT = r"D:\FILELin\postgraduate\little_paper\coco\val2017\select_coco"  # 整理后的验证集根目录

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
    img_dataset = CustomImageDataset(
        root_dir=IMG_ROOT,
        transform=val_transform
    )


    img_loader = DataLoader(
        img_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  
        num_workers=4,
        pin_memory=True
    )



    # --------------------------
    # 4. 迭代（对抗样本生成）
    # --------------------------


 

    for batch_idx, (images, images_path) in enumerate(img_loader):
        exp_root=os.path.join(root_path,'exp/1213')
        # 获取图片文件名,去除后缀
        image_name = os.path.splitext(os.path.basename(images_path[0]))[0]

        os.makedirs(exp_root,exist_ok=True)
        start_time = time.time()
        attack.generate_adversarial_main_all_mask(images,exp_path=exp_root,images_path=images_path,mask_select_statues=1)
  
        # # attack.generate_adversarial_main(images,exp_path=exp_path,mask_select_statues=1)
        # try:
        #     # attack.generate_adversarial_main(images,exp_path=exp_path,mask_select_statues=1)
        #     attack.generate_adversarial_main_all_mask(images,exp_path=exp_path,mask_select_statues=0)
        # except:
        #     print(f"{image_name} error")
        
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"代码运行耗时：{elapsed_time:.2f} 秒")  # 保留2位小数，输出：2.00 秒
        attack.destroy_controlnet()