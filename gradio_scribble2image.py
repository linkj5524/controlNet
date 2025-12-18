from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler












def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = np.zeros_like(img, dtype=np.uint8)
        detected_map[np.min(img, axis=2) < 127] = 255

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
        # 保存controltensor
        torch.save(control, 'control_good.pt')

        # c_concat 草图控制；c_crossattn 跨模态控制：正向和附加的文本提示;文本内容默认用clip编码
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
        # 将模型输出的图像进行解码
        x_samples = model.decode_first_stage(samples)
        #维度变换
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    #以列表形式，返回所有结果
    return [255 - detected_map] + results




if __name__=='__main__':
    #判断gpu 是否存在，并给出版本
    if torch.cuda.is_available():
        print('cuda version:', torch.version.cuda)
        
    else:
        print('no cuda')
    # 模型参数里面包含 ControlNet 和ControlledUnetModel 的参数
    model = create_model('./models/cldm_v15.yaml').cpu()
    temp=load_state_dict('./models/control_sd15_scribble.pth', location='cuda')
    #nn.model 自带的参数加载函数
    model.load_state_dict(temp,strict=False)
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)


    #参数
    # 定义提示词
    prompt = ""
    a_prompt = ""
    n_prompt = ""

    # 设置参数
    num_samples = 1               # 生成图像数量
    image_resolution = 512        # 图像分辨率
    ddim_steps = 10              # 采样步数
    guess_mode = False            # 是否使用猜测模式
    strength = 1.0                # 控制生成与输入的相似度
    scale = 11.0                   # 引导系数
    seed = 42                     # 随机种子（用于结果可复现）
    eta = 0.0                     # DDIM采样器的eta参数

    # img=cv2.imread('test_imgs\human_line.png')
    img=cv2.imread(r'exp\1213\000000363875\control.jpg')
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    # 调用函数
    resu = process(
        img, prompt, a_prompt, n_prompt,
        num_samples, image_resolution, ddim_steps,
        guess_mode, strength, scale, seed, eta
    )



    
    # resu=process(img, 'best quality, extremely detailed', 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', 1, 512, 20, False, 1.0, 9.0, -1, 0.0)
    img1=cv2.cvtColor(resu[0], cv2.COLOR_RGB2BGR)
    img2=cv2.cvtColor(resu[1], cv2.COLOR_RGB2BGR)
    cv2.imwrite('result1.png',img1)
    cv2.imwrite('result2.png',img2)
    cv2.imwrite('result_origin_chanell.png',resu[1])

 
