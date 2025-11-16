import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional, Union, Tuple
import os


class LocalLlavaImageToText:
    """本地LLaVA模型封装类，支持从本地路径加载模型，实现图像转文本功能"""
    
    def __init__(
        self,
        model_dir: str,
        device: Optional[str] = None,
        load_in_4bit: bool = True,
        torch_dtype: torch.dtype = torch.float16,
        image_size: Tuple[int, int] = (336, 336)
    ):
        """
        初始化配置，指定本地模型路径
        
        Args:
            model_dir: 本地模型文件夹路径（需包含模型和处理器文件）
            device: 运行设备，None则自动选择（优先CUDA）
            load_in_4bit: 是否4-bit量化加载（节省显存）
            torch_dtype: 模型数据类型（建议float16）
            image_size: 图像输入尺寸（需与模型训练时一致）
        """
        # 验证本地模型路径有效性
        if not os.path.isdir(model_dir):
            raise ValueError(f"本地模型路径不存在: {model_dir}")
        
        self.model_dir = model_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_in_4bit = load_in_4bit
        self.torch_dtype = torch_dtype
        self.image_size = image_size
        self.model = None
        self.processor = None
        self._is_initialized = False
        
        # 初始化图像转换器
        self._init_image_transform()
        
    def _init_image_transform(self):
        """初始化图像到Tensor的转换管道"""
        self.image_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])
        
    def initialize(self):
        """从本地路径加载模型和处理器"""
        if self._is_initialized:
            print("模型已初始化，无需重复加载")
            return
            
        # 加载处理器（本地路径）
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_dir,
                local_files_only=True  # 强制使用本地文件，不联网下载
            )
        except Exception as e:
            raise RuntimeError(f"加载处理器失败，请检查本地路径是否包含processor文件: {str(e)}")
        
        # 加载模型（本地路径）
        try:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_dir,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                load_in_4bit=self.load_in_4bit if self.device == "cuda" else False,
                device_map="auto" if self.device == "cuda" else self.device,
                local_files_only=True  # 强制使用本地文件
            )
        except Exception as e:
            raise RuntimeError(f"加载模型失败，请检查本地路径是否包含模型文件: {str(e)}")
        
        self._is_initialized = True
        print(f"模型已从本地路径加载: {self.model_dir}")
        print(f"运行设备: {self.model.device}")
        
    def _preprocess_tensor(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """预处理输入的图像Tensor，确保格式符合模型要求"""
        # 检查通道数（必须为3通道RGB）
        if img_tensor.dim() != 3 or img_tensor.size(0) != 3:
            raise ValueError(f"图像Tensor需为3通道(RGB)，形状为[C, H, W]，实际形状: {img_tensor.shape}")
            
        # 调整尺寸（如果与模型要求不符）
        if img_tensor.shape[1:] != self.image_size:
            img_tensor = transforms.Resize(self.image_size)(img_tensor)
            
        return img_tensor
        
    def process(
        self,
        input_data: Union[torch.Tensor, str, Image.Image],
        prompt: str = "请描述这张图片的内容。",
        max_new_tokens: int = 300,
        do_sample: bool = False
    ) -> str:
        """
        执行图像转文本推理
        
        Args:
            input_data: 输入数据（支持Tensor、图像路径、PIL Image）
            prompt: 提示词（引导模型生成特定内容）
            max_new_tokens: 最大生成文本长度
            do_sample: 是否启用采样生成（False为确定性输出）
            
        Returns:
            生成的文本结果
        """
        if not self._is_initialized:
            raise RuntimeError("模型未初始化，请先调用initialize()方法加载本地模型")
            
        # 处理不同类型的输入
        if isinstance(input_data, str):
            # 输入为图像文件路径
            if not os.path.isfile(input_data):
                raise FileNotFoundError(f"图像文件不存在: {input_data}")
            img = Image.open(input_data).convert("RGB")
            img_tensor = self.image_transform(img)
            
        elif isinstance(input_data, Image.Image):
            # 输入为PIL Image对象
            img_tensor = self.image_transform(input_data)
            
        elif isinstance(input_data, torch.Tensor):
            # 输入为Tensor（需预处理）
            img_tensor = self._preprocess_tensor(input_data)
            
        else:
            raise TypeError(f"不支持的输入类型: {type(input_data)}，支持类型：Tensor、str(路径)、PIL.Image")
            
        # 准备模型输入
        inputs = self.processor(
            text=prompt,
            images=img_tensor,
            return_tensors="pt"
        ).to(self.model.device, self.torch_dtype)
        
        # 生成文本
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )
            
        # 解码并清理结果（移除提示词部分）
        result = self.processor.decode(outputs[0], skip_special_tokens=True).strip()
        # 移除提示词前缀（如果存在）
        if result.startswith(prompt):
            result = result[len(prompt):].strip()
            
        return result
        
    def destroy(self):
        """释放模型资源，清理显存"""
        if self._is_initialized:
            # 释放模型和处理器引用
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self._is_initialized = False
            print("模型资源已释放")
            
            # 清理CUDA缓存
            if self.device == "cuda":
                torch.cuda.empty_cache()
                print("CUDA缓存已清理")


# 使用示例
if __name__ == "__main__":
    # 本地模型路径（请替换为你的实际路径）
    LOCAL_MODEL_DIR = "/path/to/your/local/llava-model"  # 例如："./llava-1.5-7b-hf"
    
    # 初始化工具类
    llava = LocalLlavaImageToText(
        model_dir=LOCAL_MODEL_DIR,
        load_in_4bit=True,  # 4-bit量化（显存不足时启用）
        image_size=(336, 336)  # 需与模型训练时的输入尺寸一致
    )
    
    try:
        # 加载本地模型
        print("正在加载本地模型...")
        llava.initialize()
        
        # 1. 处理图像文件路径
        print("\n=== 处理图像文件 ===")
        image_path = "test_image.jpg"  # 替换为你的图像路径
        description = llava.process(
            input_data=image_path,
            prompt="请详细描述这张图片中的内容，包括物体、颜色和场景。"
        )
        print("生成结果:\n", description)
        
        # 2. 处理Tensor输入（示例）
        print("\n=== 处理Tensor输入 ===")
        img = Image.open(image_path).convert("RGB")
        img_tensor = transforms.ToTensor()(img)  # 模拟外部生成的Tensor
        subject = llava.process(
            input_data=img_tensor,
            prompt="这张图片的主体是什么？用一句话回答。",
            max_new_tokens=50
        )
        print("生成结果:\n", subject)
        
    except Exception as e:
        print(f"执行出错: {str(e)}")
    finally:
        # 释放资源
        llava.destroy()