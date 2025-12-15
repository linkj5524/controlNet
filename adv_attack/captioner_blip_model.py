import os
import cv2
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np
from util import *

# 全局变量存储模型和处理器

def init_image_captioner(model_name="Salesforce/blip-image-captioning-large", device=None):
    """
    初始化图像描述模型
    
    Args:
        model_name: 预训练模型名称或本地路径
        device: 运行设备（"cuda"/"cpu"），自动检测GPU
    
    Returns:
        tuple: (model, processor, device) 初始化后的模型、处理器和设备
    """
    # 设置设备
    _device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {_device}")
    
    # 加载处理器和模型
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(_device)
    
    # 若有GPU且支持半精度，使用float16加速
    if _device == "cuda":
        model = model.half()
    
    print("图像描述模型初始化完成")
    return model, processor, _device

def image_captioner_process(model, processor, device, image_tensors, max_length=50, num_beams=4, temperature=1.0):
    """
    处理Tensor格式的图像，生成文本描述
    
    Args:
        model: 初始化后的BLIP模型
        processor: BLIP处理器
        device: 运行设备
        image_tensor: 图像张量
        max_length: 生成文本的最大长度
        num_beams: beam search的beam数量
        temperature: 控制生成多样性的温度参数
    
    Returns:
        caption: 生成的文本描述
    """
    if model is None or processor is None:
        raise RuntimeError("模型未正确初始化")
    
    # 将Tensor转换为PIL Image
    if isinstance(image_tensors, torch.Tensor):
        # 如果是批量数据，取第一个样本
        if len(image_tensors.shape) == 3:
            image_tensors=image_tensors.unsqueeze(0)
            
        
    else:
        raise TypeError(f"不支持的图像类型: {type(image_tensors)}")
    caption_list=[]
    for i in range(image_tensors.shape[0]): 
        image_tensor = image_tensors[i]  
        
        # 转换为numpy数组
        img_np = image_tensor.cpu().detach().numpy()
        
        # 如果值范围是0-1，转换为0-255
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        
        # 处理通道顺序 [C, H, W] -> [H, W, C]
        if img_np.shape[0] in [1, 3]:
            img_np = np.transpose(img_np, (1, 2, 0))
        
        # 处理灰度图像
        if img_np.shape[-1] == 1:
            img_np = np.squeeze(img_np, axis=-1)
        
        image = Image.fromarray(img_np)
        # 图像预处理
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        # 若使用半精度
        if device == "cuda":
            inputs = {k: v.half() for k, v in inputs.items()}
        
        # 生成文本描述
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.2
            )
        
        # 解码生成的文本
        caption = processor.decode(outputs[0], skip_special_tokens=True)   
        caption_list.append(caption) 

    return caption_list

def destroy_image_captioner(model):
    """
    销毁模型，释放资源
    
    Args:
        model: 需要销毁的模型对象
    """
    if model is not None:
        del model
    
    # 清空GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("图像描述模型已销毁，资源已释放")




if __name__ == "__main__":
    root_path=os.path.dirname(os.path.dirname(__file__))
    # 1. 初始化模型
    model_path = os.path.join(root_path, "models", "Salesforceblip-image-captioning-large")
    # 如果本地路径不存在，使用Hugging Face模型名
    if not os.path.exists(model_path):
        model_path = "Salesforce/blip-image-captioning-large"
    
    blip_model, blip_processor, blip_device = init_image_captioner(model_name=model_path)
    
    try:
        # 2. 创建测试Tensor（从本地图片加载）

        img_path = os.path.join(root_path, 'test_imgs', 'boy.png')
        
        # 使用OpenCV读取图像
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像文件: {img_path}")
        
        img = cv2.resize(img, (512, 512))
        
        # 3. 转换为Tensor并处理
        img_tensor = cv2_to_tensor(img, normalize=True)
        caption = image_captioner_process(blip_model, blip_processor, blip_device, img_tensor)
        print(f"\n图像描述: {caption}")
        
        # 可选：生成多个描述（可扩展process函数支持）
        # multiple_captions = process_multiple(blip_model, blip_processor, blip_device, img_tensor, num_captions=3)
        # for i, cap in enumerate(multiple_captions, 1):
        #     print(f"{i}. {cap}")

    except Exception as e:
        print(f"错误: {e}")
    
    finally:
        # 销毁模型
        destroy_image_captioner(blip_model)