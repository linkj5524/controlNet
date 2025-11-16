import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler

# 本地模型路径（替换为你下载模型的文件夹路径）
local_model_path = "D:/FILELin/postgraduate/little_paper/Adversariall_attack_project/ControlNet/models/"  # 存放下载的模型文件的目录

# 加载本地模型
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# 后续预处理、生成代码不变...
def preprocess(image_path, mask_path, size=(512, 512)):
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    image = image.resize(size)
    mask = mask.resize(size)
    mask_np = np.array(mask)
    mask_np = (mask_np > 127).astype(np.uint8) * 255
    mask = Image.fromarray(mask_np)
    return image, mask

# 随机生成掩码的函数（新增，替代读取本地掩码）
def generate_random_mask(size=(512, 512), mask_ratio=0.2):
    """生成随机二值掩码，mask_ratio为掩码区域占比"""
    mask_np = np.random.randint(0, 2, size, dtype=np.uint8)
    # 调整掩码区域占比
    threshold = int(255 * (1 - mask_ratio))
    mask_np = (mask_np * 255).astype(np.uint8)
    return Image.fromarray(mask_np)

# 使用随机掩码
image = preprocess("input_image.jpg", "mask.png")[0]  # 仅读取原图
mask = generate_random_mask(size=(512, 512), mask_ratio=0.2)
mask.save("random_mask.png")  # 保存随机生成的掩码

# 生成修复结果
prompt = "a green grass field with flowers, sunny day, photorealistic"
negative_prompt = "blurry, low quality, disfigured, unnatural"

with torch.no_grad():
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=50,
        guidance_scale=7.5,
    ).images[0]

result.save("inpaint_result.png")