from share import *
import config

import cv2
import einops
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


# advisarial attack class
from adv_attack.attack_class import  *






if __name__=='__main__':
    #判断gpu 是否存在，并给出版本
    if torch.cuda.is_available():
        print('cuda version:', torch.version.cuda)
        
    else:
        print('no cuda')
    # 模型参数里面包含 ControlNet 和ControlledUnetModel 的参数



    #参数
    # 定义提示词
    prompt = "a handsome boy with a long hair "
    a_prompt = ""
    n_prompt = ""

    # 设置参数
    num_samples = 1               # 生成图像数量
    image_resolution = 512        # 图像分辨率
    ddim_steps = 50               # 采样步数
    guess_mode = False            # 是否使用猜测模式
    strength = 1.0                # 控制生成与输入的相似度
    scale = 9.0                   # 引导系数
    seed = 42                     # 随机种子（用于结果可复现）
    eta = 0.0                     # DDIM采样器的eta参数

    img=cv2.imread('test_imgs\man.png')

    attack = ADV_ATTACK(device=torch.device("cuda"))
    attack.generate_adversarial_example(img)

    


    


 
