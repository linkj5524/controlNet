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

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [255 - detected_map] + results


# block = gr.Blocks().queue()
# with block:
    # with gr.Row():
        # gr.Markdown("## Control Stable Diffusion with Scribble Maps")
    # with gr.Row():
        # with gr.Column():
            # input_image = gr.Image(source='upload', type="numpy")
            # prompt = gr.Textbox(label="Prompt")
            # run_button = gr.Button(label="Run")
            # with gr.Accordion("Advanced options", open=False):
                # num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                # image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                # strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                # guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                # ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                # scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                # seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                # eta = gr.Number(label="eta (DDIM)", value=0.0)
                # a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                # n_prompt = gr.Textbox(label="Negative Prompt",
                                    #   value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        # with gr.Column():
            # result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
    # ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta]
    # run_button.click(fn=process, inputs=ips, outputs=[result_gallery])

# 
# block.launch(server_name='0.0.0.0')


if __name__=='__main__':
    #判断gpu 是否存在，并给出版本
    if torch.cuda.is_available():
        print('cuda version:', torch.version.cuda)
        
    else:
        print('no cuda')

    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict('./models/control_sd15_scribble.pth', location='cuda'),strict=False)
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)


    #参数
    # 定义提示词
    prompt = "一个美丽的花园，有鲜花和树木，高画质"
    a_prompt = "最佳质量，高分辨率，详细，逼真"
    n_prompt = "低质量，模糊，扭曲，噪点"

    # 设置参数
    num_samples = 2               # 生成图像数量
    image_resolution = 512        # 图像分辨率
    ddim_steps = 50               # 采样步数
    guess_mode = False            # 是否使用猜测模式
    strength = 0.8                # 控制生成与输入的相似度
    scale = 9.0                   # 引导系数
    seed = 42                     # 随机种子（用于结果可复现）
    eta = 0.0                     # DDIM采样器的eta参数

    img=cv2.imread('test_imgs\human_line.png')
    # 调用函数
    resu = process(
        img, prompt, a_prompt, n_prompt,
        num_samples, image_resolution, ddim_steps,
        guess_mode, strength, scale, seed, eta
    )



    
    # resu=process(img, 'best quality, extremely detailed', 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', 1, 512, 20, False, 1.0, 9.0, -1, 0.0)
    img1=cv2.cvtColor(resu[0], cv2.COLOR_BGR2RGB)
    img2=cv2.cvtColor(resu[1], cv2.COLOR_BGR2RGB)
    cv2.imwrite('result1.png',img1)
    cv2.imwrite('result2.png',img2)


