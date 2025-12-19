import torch
from diffusers import AutoencoderKL
from PIL import Image
from torchvision import transforms
import numpy as np
import gc

class VAEInferencer:
    """
    Stable Diffusion VAE 推理工具类
    包含模型初始化、销毁、编码、解码、图像重建推理等功能
    """
    def __init__(self, model_name: str = "stabilityai/sd-vae-ft-mse", device: str = None, dtype: torch.dtype = torch.float32):
        """
        初始化 VAE 模型
        Args:
            model_name: Hugging Face 模型名称或本地权重路径
            device: 运行设备（"cuda"/"cpu"），默认自动检测
            dtype: 数据精度（torch.float32/torch.float16），显存不足时用 float16
        """
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.vae = None  # 初始化模型实例为 None
        self._init_model()  # 加载模型

    def _init_model(self):
        """内部方法：加载 VAE 模型和权重"""
        try:
            self.vae = AutoencoderKL.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                use_safetensors=True
            ).to(self.device)
            self.vae.eval()  # 切换为推理模式
            print(f"✅ VAE 模型 {self.model_name} 加载完成，设备：{self.device}，精度：{self.dtype}")
        except Exception as e:
            raise RuntimeError(f"加载 VAE 模型失败：{str(e)}")


    def encode_infer(self, img_tensor: torch.Tensor, sample_posterior: bool = False) -> torch.Tensor:
        """
        VAE 编码：图像张量 → Latent 张量
        Args:
            img_tensor: 预处理后的图像张量 [1, 3, H, W]
            use_mode: 是否取后验分布的均值（True）或采样（False），均值更稳定
        Returns:
            latent: Latent 张量 [1, 4, H//8, W//8]
        """
        if self.vae is None:
            raise RuntimeError("VAE 模型未初始化，请先调用 init_model 或实例化类")
        
        
        posterior = self.vae.encode(img_tensor).latent_dist
        if sample_posterior:
            latent = posterior.sample()  # 随机采样（带噪声）
        else:
            latent = posterior.mode()  # 取均值（稳定）
                
        return latent

    def decode_infer(self, latent: torch.Tensor) -> torch.Tensor:
        """
        VAE 解码：Latent 张量 → 重建图像张量
        Args:
            latent: Latent 张量 [1, 4, H//8, W//8]
        Returns:
            recon_img_tensor: 重建图像张量 [1, 3, H, W]，范围 [-1, 1]
        """
        if self.vae is None:
            raise RuntimeError("VAE 模型未初始化，请先调用 init_model 或实例化类")
        

        recon_img_tensor = self.vae.decode(latent, return_dict=False)[0]
        return recon_img_tensor

    def infer(self, images_tensor,sample_posterior=False) :
        """
        VAE 完整推理：图像 → 编码 → 解码 → 重建图像
        Args:
            sample_posterior: False 表示采用均值
        Returns:
            recon_img: 重建的 PIL 图像
        """

        # 编码
        latent = self.encode_infer(images_tensor,sample_posterior=sample_posterior)

        recon_img_tensor = self.decode_infer(latent)

        return recon_img_tensor

    def destroy(self):
        """销毁 VAE 模型，释放显存/内存"""
        if self.vae is not None:
            # 解除模型引用
            self.vae = None
            # 释放 GPU 显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            # 强制垃圾回收
            gc.collect()
            print("✅ VAE 模型已销毁，资源已释放")
        else:
            print("⚠️ VAE 模型未初始化，无需销毁")

    def __del__(self):
        """析构函数：实例销毁时自动调用 destroy"""
        self.destroy()

# ====================== 测试代码 ======================
if __name__ == "__main__":
    # 配置参数
    VAE_MODEL_NAME = r"models\sd_vae_ft_mse"
    IMAGE_PATH = "test_input.jpg"  # 替换为你的图像路径
    SAVE_RECON_PATH = "vae_reconstructed.png"
    TARGET_SIZE = (512, 512)  # 8 的倍数

    # 实例化 VAE 推理器（自动初始化模型）
    vae_inferencer = VAEInferencer(
        model_name=VAE_MODEL_NAME,
        dtype=torch.float16  # 显存不足时用 float16
    )

    a=0
