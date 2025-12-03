import torch
from diffusers import StableDiffusionXLInpaintPipeline
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
import numpy as np
from util import *





def create_fixed_mask(image_size, mask_coords, mask_color="white"):
    """创建固定坐标和大小的mask（返回PIL Image）"""
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    x1, y1, x2, y2 = mask_coords
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_size[0], x2)
    y2 = min(image_size[1], y2)
    fill_color = 255 if mask_color == "white" else 0
    draw.rectangle([x1, y1, x2, y2], fill=fill_color)
    return mask

# -------------------------- 核心函数：初始化、处理、销毁 --------------------------
def init_sdxl_inpaint(model_dir, device=None):
    """
    初始化SDXL Inpaint模型
    Args:
        model_dir: 本地模型目录路径
        device: 运行设备（"cuda"/"cpu"），自动检测
    Returns:
        pipe: 初始化后的SDXL Inpaint管道
    """
    # 自动选择设备
    _device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{_device}")
    
    # 校验模型目录
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"模型目录不存在：{model_dir}")
    if not os.path.exists(os.path.join(model_dir, "model_index.json")):
        raise FileNotFoundError(f"模型目录缺少关键文件：model_index.json")
    
    # 选择精度
    dtype = torch.float16 if (_device == "cuda" and torch.cuda.is_available()) else torch.float32
    
    # 加载模型
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        local_files_only=True,
        variant="fp16" if dtype == torch.float16 else None
    ).to(_device)
    
    # GPU优化
    if _device == "cuda":
        pipe.unet.to(memory_format=torch.channels_last)
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            print("警告：未安装xformers，跳过内存优化")
        pipe.enable_model_cpu_offload()
    
    print("SDXL Inpaint模型初始化成功")
    return pipe

def process_sdxl_inpaint(pipe, image_tensor, mask_tensor, prompt, negative_prompt="", 
                         num_steps=50, guidance_scale=7.5, strength=0.9):
    """
    处理SDXL图像修复（输入为Tensor）
    Args:
        pipe: 初始化后的模型管道
        image_tensor: 原图张量 [C, H, W] 或 [B, C, H, W]，值范围0-1
        mask_tensor: mask张量 [C, H, W] 或 [B, C, H, W]，值范围0-1（0为修复区域）
        prompt: 修复提示词
        negative_prompt: 负面提示词
        num_steps: 推理步数
        guidance_scale: 提示词强度
        strength: 修复强度（0-1）
    Returns:
        inpainted_tensor: 修复后的图像张量 [C, H, W]，值范围0-1
    """
    if pipe is None:
        raise RuntimeError("模型未初始化，请先调用init_sdxl_inpaint()")
    
    # Tensor转PIL（模型要求输入为PIL Image）
    image = tensor_to_pil(image_tensor)
    mask_image = tensor_to_pil(mask_tensor)
    
    # 确保mask为单通道
    if mask_image.mode != "L":
        mask_image = mask_image.convert("L")
    
    # 执行修复
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask_image,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            width=image.size[0],
            height=image.size[1],
            eta=0.0,
            strength=strength
        ).images[0]
    
    # 修复结果转Tensor（保持输出格式与输入一致）
    inpainted_tensor = torch.from_numpy(np.array(result)).permute(2, 0, 1).float() / 255.0
    return inpainted_tensor

def destroy_sdxl_inpaint(pipe):
    """
    销毁模型，释放资源
    Args:
        pipe: 已初始化的模型管道
    """
    if pipe is not None:
        # 释放模型组件
        del pipe.unet
        del pipe.text_encoder
        del pipe.text_encoder_2
        del pipe.vae
        del pipe
    
    # 清空GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("SDXL Inpaint模型已销毁，资源释放完成")

# -------------------------- 辅助函数：结果显示与保存 --------------------------
def display_and_save_results(original_tensor, mask_tensor, inpainted_tensor, mask_info="", 
                             save_path="sdxl_inpainted_result.png"):
    """
    显示原图、mask、修复结果（输入为Tensor）并保存
    """
    # Tensor转PIL用于显示
    original_img = tensor_to_pil(original_tensor)
    mask_img = tensor_to_pil(mask_tensor)
    inpainted_img = tensor_to_pil(inpainted_tensor)
    
    # 绘图显示
    plt.figure(figsize=(20, 6))
    titles = ["原图", f"Mask（{mask_info}）", "SDXL 修复结果"]
    imgs = [original_img, mask_img, inpainted_img]
    
    for i, (img, title) in enumerate(zip(imgs, titles)):
        plt.subplot(1, 3, i+1)
        plt.imshow(img, cmap="gray" if "Mask" in title else None)
        plt.title(title, fontsize=16)
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()
    
    # 保存结果
    inpainted_img.save(save_path, quality=95)
    print(f"修复结果已保存至：{os.path.abspath(save_path)}")
    mask_img.save("generated_mask.png")
    print(f"Mask已保存至：{os.path.abspath('generated_mask.png')}")

import re





# -------------------------- 主函数：测试流程 --------------------------
def main():
    # 配置参数
    LOCAL_MODEL_DIR = "./sdxl-inpaint-model"
    IMAGE_PATH = r"D:\FILELin\postgraduate\little_paper\Adversariall_attack_project\ControlNet\test_imgs\dog.png"
    TARGET_SIZE = (512, 512)
    MASK_COORDS = (20, 50, 500, 500)  # 修复区域坐标
    PROMPT = "a photo of a dog with clear fur, realistic details, natural lighting"  # 补充具体提示词
    NEGATIVE_PROMPT = "blurry, ugly, deformed, low quality, watermark, text, noise"
    NUM_STEPS = 50
    GUIDANCE_SCALE = 7.5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # 1. 加载图像并转Tensor
        original_img = Image.open(IMAGE_PATH).resize(TARGET_SIZE).convert("RGB")
        original_tensor = torch.from_numpy(np.array(original_img)).permute(2, 0, 1).float() / 255.0
        print(f"原图加载成功，张量形状：{original_tensor.shape}")
        
        # 2. 创建mask并转Tensor
        mask_img = create_fixed_mask(TARGET_SIZE, MASK_COORDS, mask_color="white")
        mask_tensor = torch.from_numpy(np.array(mask_img)).unsqueeze(0).float()  # [1, H, W]
        mask_info = f"Single region: {MASK_COORDS}"
        print(f"Mask创建成功，张量形状：{mask_tensor.shape}")
        
        # 3. 初始化模型
        pipe = init_sdxl_inpaint(LOCAL_MODEL_DIR, DEVICE)
        
        # 4. 执行修复（输入均为Tensor）
        inpainted_tensor = process_sdxl_inpaint(
            pipe=pipe,
            image_tensor=original_tensor,
            mask_tensor=mask_tensor,
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            num_steps=NUM_STEPS,
            guidance_scale=GUIDANCE_SCALE
        )
        print(f"修复完成，输出张量形状：{inpainted_tensor.shape}")
        
        # 5. 显示并保存结果
        display_and_save_results(original_tensor, mask_tensor, inpainted_tensor, mask_info)
        
    except Exception as e:
        print(f"执行失败：{str(e)}")
    finally:
        # 6. 销毁模型
        if 'pipe' in locals():
            destroy_sdxl_inpaint(pipe)

if __name__ == "__main__":
    main()