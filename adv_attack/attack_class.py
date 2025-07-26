
from share import *
import config

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

# 本地的包
## 添加本地包路径,即上一级的路径
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler





class ADV_ATTACK:
    def __init__(self, config_path:str='./models/cldm_v15.yaml',
                  model_path:str='./models/control_sd15_scribble.pth', 
                  device:torch.device=torch.device("cuda"),
                  **kwargs
                  ):
        """
        初始化对抗攻击类
        
        参数:
            config_path: 模型配置文件路径 (默认 "./models/cldm_v15.yaml")
            model_path: 预训练模型权重路径 (默认 "./models/control_sd15_scribble.pth")
            device: 运行设备 (默认 "cuda")
        """
        # 加载模型配置
        self.config_path = config_path
        self.model_path = model_path
        self.device = device
        
        # 初始化模型
        self.model = create_model(self.config_path).cpu()
        self.model.load_state_dict(load_state_dict(self.model_path, location='cuda'),strict=False)
        self.ddim_sampler = DDIMSampler(self.model)
        
        
        # 默认参数配置
        if kwargs:
            self.default_params = kwargs
        else:
            self.default_params = {
            "prompt": "a handsome boy with a long hair",
            "a_prompt": "",
            "n_prompt": "",
            "num_samples": 1,
            "image_resolution": 512,
            "ddim_steps": 50,
            "guess_mode": False,
            "strength": 1.0,
            "scale": 9.0,
            "seed": 42,
            "eta": 0.0,
            "save_memory": True
        }
    



    def set_params(self, **kwargs):
        """更新攻击参数"""
        for key, value in kwargs.items():
            if key in self.default_params:
                self.default_params[key] = value
            else:
                print(f"警告: 参数 {key} 不是有效参数，将被忽略")
    
    def generate_adversarial_example(self, control_image=None, params=None):
        """
        生成对抗样本
        
        参数:
            control_image: 控制图像 (用于ControlNet)
            params: 覆盖默认参数的字典
            
        返回:
            生成的对抗图像列表
        """
        # 使用默认参数或用户指定参数
        if params is None:
            params = self.default_params
        else:
            params = {**self.default_params, **params}
        self.default_params = params
        # 设置随机种子以确保结果可复现
        torch.manual_seed(params["seed"])
        np.random.seed(params["seed"])
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
       
       
       
        
        if params["save_memory"]:
            self.model.low_vram_shift(is_diffusing=False)




        # 预处理图像
        ## 确保图像通道正常,对图像的size有要求，不能随便的大小
        control_image=HWC3(control_image)
        control_image = cv2.resize(control_image, (self.default_params["image_resolution"],self.default_params["image_resolution"]))

        ## 处理控制图像，并返回边缘图和边缘的control
        self.edge_image,self.control=self.generate_edge_control_from_image(control_image)



        # c_concat 草图控制；c_crossattn 跨模态控制：正向和附加的文本提示;文本内容默认用clip编码
        cond = {
            "c_concat": [self.control],
            "c_crossattn": [
                self.model.get_learned_conditioning(
                    [params["prompt"] + ', ' + params["a_prompt"]] * params["num_samples"]
                )
            ]
        }
        un_cond = {
            "c_concat": None if params["guess_mode"] else [self.control],
            "c_crossattn": [
                self.model.get_learned_conditioning(
                    [params["n_prompt"]] * params["num_samples"]
                )
            ]
        }
 
 
        H, W, C = control_image.shape
        shape = (4, H // 8, W // 8)

        if params["save_memory"]:
            self.model.low_vram_shift(is_diffusing=True)


        self.model.control_scales = (
            [params["strength"] * (0.825 ** float(12 - i)) for i in range(13)]
            if params["guess_mode"]
            else [params["strength"]] * 13
        )  # Magic number. IDK why
        
        #对图像进行编码，转换为latent
        temp_tensor=self.to_tensor_from_image(control_image)
        latent=self.img_to_latent(temp_tensor)

        # 设置采样参数
        self.ddim_sampler.make_schedule(ddim_num_steps=params["ddim_steps"], ddim_eta=params["eta"], verbose=False)
        # 使用封装的ddim进行逆采样
        x_next, out=self.ddim_sampler.encode(x0=latent, c=cond, t_enc=params["ddim_steps"], use_original_steps=False, return_intermediates=True,
               unconditional_guidance_scale=1, unconditional_conditioning=un_cond, callback=None)
        
        
        # 对latent进行优化






        
        # 将所有的latent转换成图片
        for i in range(len(out["intermediates"])):
            image=self.latent_to_img(out["intermediates"][i])

            cv2.imwrite("{}/{}.png".format("./exp/result", i), image[0])
        # images = self.latent_to_img(latent)
        
        # img1=cv2.cvtColor(images[0], cv2.COLOR_RGB2BGR)
        
        # cv2.imwrite('result1.png',images[0])
        
        return 
    def to_tensor_from_image(self, image):


        # 转换到-1到1
        image = image.astype(np.float32) / 127.5 - 1.0

        tensor = torch.from_numpy(image).float()

        # 4. 调整维度顺序：从[H, W, C]转换为[C, H, W]（PyTorch要求的通道优先格式）
        tensor = tensor.permute(2, 0, 1)
        
        return tensor.unsqueeze(0)
    

    def generate_edge_control_from_image(self, image):
        '''
        作用：预处理图像，返回边缘图和边缘的control
        参数：
        image: 输入的图像
        返回：
        detected_map: 边缘图(size: [H, W])
        control: 边缘的control(size: [num_samples, 3, H, W])
        '''
        # resize,opencv 格式
        
        H, W, C = image.shape

        detected_map = np.zeros_like(image, dtype=np.uint8)
        detected_map[np.min(image, axis=2) < 127] = 255

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(self.default_params["num_samples"])], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        return detected_map,control
    
    # 将图片转化为latent
    def img_to_latent(self, img):
        '''
        img:[0-1],type:tensor
        return: latent, type:tensor
        '''
        #编码为潜变量（关闭梯度计算，提高效率）
        with torch.no_grad():
            posterior = self.model.first_stage_model.encode(img)  # 得到后验分布
            
            # # 4. 从分布中获取潜变量
            # if sample_posterior:
            #     z = posterior.sample()  # 随机采样（带随机性）
            # else:
            z = posterior.mode()    # 取均值（确定性结果，推荐用于推理）
        z=z.to(self.device)
        return z   
    def latent_to_img(self,latent):
        img = self.model.first_stage_model.decode(latent)
        return  (einops.rearrange(img, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

















# 使用示例
if __name__ == "__main__":
    # 初始化攻击类
    attacker = ADV_ATTACK(
        config_path='./models/cldm_v15.yaml',
        model_path='./models/control_sd15_scribble.pth'
    )
    
    # 可选：更新参数
    attacker.set_params(
        prompt="a beautiful princess with long hair",
        scale=10.0,
        seed=42
    )
    
    # 加载控制图像（如果需要）
    # control_image = Image.open("path/to/control_image.png").convert("RGB")
    
    # 生成对抗样本
    adversarial_images = attacker.generate_adversarial_example()
    
    # 保存结果
    for i, img in enumerate(adversarial_images):
        img.save(f"adversarial_example_{i}.png")    