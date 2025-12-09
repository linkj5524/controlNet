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
    attack = ADV_ATTACK(device=torch.device("cuda"),model_path=controlNet_model_path,
                        model_path_object_detection=model_path,sam_model_type='vit_h',
                        sam_checkpoint_path=sam_path)
    
    
    


    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 1 
    IMG_SIZE = 256  
    VAL_ROOT = r"D:\FILELin\postgraduate\little_paper\imagenet2012\imagenet"  # 整理后的验证集根目录

    # --------------------------
    # 2. 验证集预处理（无数据增强！）
    # --------------------------
    # 注意：验证集仅做resize、中心裁剪、归一化，禁止随机增强（保证评估公平）
    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),  # 先resize到256×256（保留更多细节）
        transforms.CenterCrop(IMG_SIZE),  # 中心裁剪到224×224（模型输入尺寸）
        transforms.ToTensor(),  # 转为张量（0-1）
    ])



    # --------------------------
    # 3. 加载验证集
    # --------------------------
    # ImageFolder自动按文件夹名称分配类别标签（0-999）
    val_dataset = torchvision.datasets.ImageFolder(
        root=VAL_ROOT,
        transform=val_transform
    )

    # 构建数据加载器（验证集无需shuffle）
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # 关键：验证集shuffle=False，保证评估结果可复现
        num_workers=4,  # 线程数=CPU核心数/2（如8核CPU设4）
        pin_memory=True  # 加速GPU数据传输（减少卡顿）
    )

    # 查看验证集基本信息
    print(f"验证集样本总数：{len(val_dataset)}")  # 输出50000
    print(f"类别数：{len(val_dataset.classes)}")  # 输出1000
    print(f"前5个类别名称：{val_dataset.classes[:5]}")  # 输出类别WNID对应的名称

    # --------------------------
    # 4. 迭代验证集（对抗样本生成）
    # --------------------------




    for batch_idx, (images, labels) in enumerate(val_loader):
        exp_root=os.path.join(root_path,'exp/test')
        exp_path=os.path.join(exp_root,f"{batch_idx}")
        os.makedirs(exp_path,exist_ok=True)
        attack.generate_adversarial_main(images,exp_path=exp_path)