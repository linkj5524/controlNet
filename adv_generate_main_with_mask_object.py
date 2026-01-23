import shutil
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
    # --------------------------
    # 1. 基础配置 
    # -------------------------- 

    root_path=os.path.dirname(__file__)
    attack_config_path=args.attack_config_path
    adv_config=load_yaml_config(attack_config_path)


    attack = ADV_ATTACK(config_path=adv_config["model_paths"]["control_yaml_path"],
                        model_path=adv_config["model_paths"]["controlnet"],
                        device=torch.device("cuda"),
                        detect_model_type=adv_config["model_types"]["detect_model"],
                        model_path_object_detection=adv_config["model_paths"]["detect_model"],
                        sam_model_type=adv_config["model_types"]["sam_model"],
                        sam_checkpoint_path=adv_config["model_paths"]["sam_model"],
                        captioner_model_name=adv_config["model_paths"]["blip_model"],
                        inpaint_model_path=adv_config["model_paths"]["inpaint_model"],
                        vae_model_path=adv_config["model_paths"]["vae_model"],
                        kwargs=adv_config["attak_params"],
                        detect_params=adv_config["detect_params"],
                        )
    
    
    


    BATCH_SIZE = adv_config["experiment_params"]["batch_size"]
    IMG_SIZE = adv_config["experiment_params"]["image_size"]  
    IMG_ROOT=adv_config["experiment_params"]["dataset_path"]
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
        shuffle=False,  
        num_workers=adv_config["experiment_params"]["num_workers"],
        pin_memory=True
    )



    # --------------------------
    # 4. 迭代（对抗样本生成）
    # --------------------------
    ref_path=r"data/control1.jpg"
    ref_tenture=cv2.imread(ref_path,cv2.IMREAD_GRAYSCALE)
    ref_tenture=cv2.resize(ref_tenture, (IMG_SIZE, IMG_SIZE))
    ref_canny=cv2_to_tensor(ref_tenture)
    if ref_canny.dim()==3:  # 添加维度
        ref_canny = ref_canny.unsqueeze(0)
  
    exp_root=adv_config["experiment_params"]["experiment_path"]

    os.makedirs(exp_root,exist_ok=True) 
    # 将yaml文件复制到实验目录
    shutil.copy(attack_config_path, exp_root)

    accur_all_dict={}
    accur_num=0


    # numpy数组，shape=(B, H, W)，支持布尔型/0-1数值型
    crop_size=(350,350)
    mask_object=create_center_rect_mask(img_size=IMG_SIZE, 
                                        rect_size=crop_size,
                                        batch_size=BATCH_SIZE,
                                        dtype=np.bool_)
    

    pbar = tqdm(enumerate(img_loader), total=len(img_loader), desc="Processing images", unit="batch")
    for batch_idx, (images, images_path) in pbar:

        start_time = time.time()
        exp_paths_list=[]
        # 获取最后一击目录
        for images_path_idx in images_path:  
            temp=os.path.join(exp_root, os.path.splitext(os.path.basename(images_path_idx))[0])
            os.makedirs(temp, exist_ok=True)
            exp_paths_list.append(temp)
        

        accur_dict=attack.generate_adversarial_mainV5(background_imag=images,
                                            ref_texture=None,
                                            ref_canny=ref_canny,
                                            mask_adv=mask_object,
                                            exp_path=exp_paths_list,
                                            detect_params=adv_config["detect_params"],
                                            attribution_params=adv_config["attribution_params"],
                                            attack_params=adv_config["attak_params"]
                                        )
        if accur_dict is not None:
            print(f" current adv example accuracy : {accur_dict}")
            if accur_num==0:  # 获取第一个ASR
                accur_all_dict=accur_dict
            else:  # 累加ASR
                for key in accur_all_dict.keys():
                    accur_all_dict[key]+=accur_dict[key]
            accur_num+=1
        
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"代码运行耗时：{elapsed_time:.2f} 秒")  # 保留2位小数，输出：2.00 秒
        print(f" total adv example num : {accur_all_dict}")
        print(f" total adv num : {accur_num}")
        for key in accur_all_dict.keys():
            temp=accur_all_dict[key]/accur_num
            print(f"{key} : {temp}")
            
        attack.destroy_controlnet()


