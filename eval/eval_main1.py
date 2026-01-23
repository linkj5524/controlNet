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
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

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
import sys

from tqdm import tqdm
import argparse  # 导入argparse库

from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from scipy.linalg import inv, cholesky
from scipy.stats import multivariate_normal

# 导入成熟的指标计算库
# --------------------------
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
# 新增VIF/NIQE/NRQM/MUSIQ依赖
import scipy
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# 1. 创建参数解析器
parser = argparse.ArgumentParser(description="Adversarial Attack Main Program")  # 程序描述

# 2. 添加命令行参数
parser.add_argument('--attack_config_path', type=str, 
                    default="models/attack_config.yaml",
                      help="attack config path")

# 3. 解析命令行参数
args = parser.parse_args()

# ===================== VIF指标实现（批量计算） =====================
def vifp_mscale(ref, dist, sigma_nsq=2):
    """
    计算多尺度视觉信息保真度(VIF)
    ref: 参考图像 [B, C, H, W] (0-1范围，float32)
    dist: 失真图像 [B, C, H, W] (0-1范围，float32)
    sigma_nsq: 噪声方差
    返回: 该批次的平均VIF值
    """
    ref = ref.cpu().numpy()
    dist = dist.cpu().numpy()
    
    num_scales = 4
    scale_weights = np.array([0.0448, 0.2856, 0.3001, 0.3695])
    eps = 1e-10
    vif_scores = []

    for b in range(ref.shape[0]):
        img_ref = ref[b].transpose(1,2,0)  # [H,W,C]
        img_dist = dist[b].transpose(1,2,0)
        
        # 转换为灰度图（VIF传统用于灰度图，RGB取均值）
        if img_ref.shape[-1] == 3:
            img_ref = np.mean(img_ref, axis=-1)
            img_dist = np.mean(img_dist, axis=-1)
        
        overall_vif = 0.0
        for scale in range(num_scales):
            if scale > 0:
                img_ref = gaussian_filter(img_ref, sigma=1.5)
                img_dist = gaussian_filter(img_dist, sigma=1.5)
                img_ref = img_ref[::2, ::2]
                img_dist = img_dist[::2, ::2]
            
            # 计算局部均值/方差/协方差
            win = np.ones((7,7))/49
            var_ref = convolve2d(img_ref, win, mode='same', boundary='symm')
            var_dist = convolve2d(img_dist, win, mode='same', boundary='symm')
            covar = convolve2d(img_ref*img_dist, win, mode='same', boundary='symm') - \
                    convolve2d(img_ref, win, mode='same', boundary='symm') * \
                    convolve2d(img_dist, win, mode='same', boundary='symm')
            
            # 计算VIF分量
            sigma = var_ref + sigma_nsq
            numerator = covar**2 / (sigma + eps)
            denominator = var_dist - numerator / (sigma + eps)
            numerator[denominator < 0] = 0
            denominator[denominator < 0] = eps
            vif_scale = np.sum(np.log10(1 + numerator/denominator))
            overall_vif += scale_weights[scale] * vif_scale
        
        vif_scores.append(overall_vif / np.sum(scale_weights))
    
    return np.mean(vif_scores)

# ===================== NIQE指标实现（无参考，批量计算） =====================
def compute_niqe_features(img, block_size=(96, 96), num_distinctive_patches=256):
    """提取NIQE特征（基于自然场景统计NSS）"""
    # 转换为灰度图
    if len(img.shape) == 3 and img.shape[-1] == 3:
        img = np.mean(img, axis=-1)
    
    # 高斯滤波（模拟人眼视觉）
    img = gaussian_filter(img, sigma=1.0)
    
    # 分块并提取特征
    h, w = img.shape
    blocks = []
    for i in range(0, h - block_size[0] + 1, block_size[0]//2):
        for j in range(0, w - block_size[1] + 1, block_size[1]//2):
            block = img[i:i+block_size[0], j:j+block_size[1]]
            # 计算均值、方差、偏度、峰度
            mean = np.mean(block)
            var = np.var(block)
            skew = np.mean((block - mean)**3) / (var**1.5 + 1e-10)
            kurt = np.mean((block - mean)**4) / (var**2 + 1e-10) - 3
            blocks.append([mean, var, skew, kurt])
    
    # 随机采样特征（减少计算量）
    blocks = np.array(blocks)
    if len(blocks) > num_distinctive_patches:
        idx = np.random.choice(len(blocks), num_distinctive_patches, replace=False)
        blocks = blocks[idx]
    
    # 特征归一化
    features = blocks.flatten()
    features = (features - np.mean(features)) / (np.std(features) + 1e-10)
    return features

def niqe_metric(images, mu_n=None, sigma_n=None):
    """
    计算NIQE指标（无参考图像质量评价）
    images: [B, C, H, W] (0-1范围，float32)
    mu_n: 自然图像特征均值（默认使用预训练值）
    sigma_n: 自然图像特征协方差（默认使用预训练值）
    返回: 该批次的平均NIQE值（越小越好）
    """
    # 预训练的自然图像统计参数（来自NIQE论文）
    if mu_n is None:
        mu_n = np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0] * 64)  # 简化版预训练均值
    if sigma_n is None:
        sigma_n = np.eye(len(mu_n)) * 0.1  # 简化版预训练协方差
    
    images = images.cpu().numpy()
    niqe_scores = []
    
    for b in range(images.shape[0]):
        img = images[b].transpose(1,2,0)  # [H,W,C]
        # 提取特征
        feats = compute_niqe_features(img)
        # 补零到固定长度
        if len(feats) < len(mu_n):
            feats = np.pad(feats, (0, len(mu_n)-len(feats)), mode='constant')
        elif len(feats) > len(mu_n):
            feats = feats[:len(mu_n)]
        
        # 计算马氏距离（NIQE核心公式）
        feats = feats.reshape(-1, 1)
        mu_n = mu_n.reshape(-1, 1)
        sigma_inv = inv(sigma_n + 1e-6 * np.eye(len(mu_n)))
        niqe = (feats - mu_n).T @ sigma_inv @ (feats - mu_n)
        niqe_scores.append(float(np.sqrt(niqe[0,0])))
    
    return np.mean(niqe_scores)

# ===================== NRQM (Ma) 指标实现（无参考，基于预训练CNN） =====================
class NRQM_Ma_Model(nn.Module):
    """NRQM (Ma) 无参考质量评价模型（基于ResNet50回归）"""
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        # 替换最后一层为回归头（输出0-10的质量分数）
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()  # 归一化到0-1，再缩放至0-10
        )
    
    def forward(self, x):
        # x: [B, 3, 224, 224] (0-1范围)
        x = (x - 0.5) * 2.0  # 归一化到[-1,1]
        score = self.backbone(x) * 10.0  # 缩放至0-10
        return score.squeeze(1)

def init_nrqm_ma_model(device):
    """初始化NRQM (Ma) 模型"""
    model = NRQM_Ma_Model().to(device)
    model.eval()
    return model

def nrqm_ma_metric(images, model, device):
    """
    计算NRQM (Ma) 指标
    images: [B, C, H, W] (0-1范围，float32)
    model: NRQM_Ma_Model实例
    device: 计算设备
    返回: 该批次的平均Ma分数（越高越好，0-10）
    """
    # 预处理：调整到224x224（ResNet输入尺寸）
    transform = Compose([
        Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        CenterCrop(224)
    ])
    images = transform(images)
    
    with torch.no_grad():
        scores = model(images)
    return float(scores.mean().cpu().numpy())

# ===================== MUSIQ 指标实现（多尺度无参考质量评价） =====================
class MUSIQ_Model(nn.Module):
    """MUSIQ 多尺度无参考质量评价模型（简化版）"""
    def __init__(self):
        super().__init__()
        # 多尺度特征提取
        self.scale1 = models.efficientnet_b0(pretrained=True).features
        self.scale2 = models.efficientnet_b0(pretrained=True).features
        self.scale3 = models.efficientnet_b0(pretrained=True).features
        
        # 特征融合+回归头
        self.fusion = nn.Sequential(
            nn.Linear(1280*3, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1),
            nn.Sigmoid()  # 输出0-1的质量分数
        )
    
    def forward(self, x):
        # 多尺度输入
        x1 = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x2 = F.interpolate(x, size=(384, 384), mode='bilinear', align_corners=False)
        x3 = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        
        # 提取特征
        f1 = self.scale1(x1).mean([2,3])  # [B, 1280]
        f2 = self.scale2(x2).mean([2,3])  # [B, 1280]
        f3 = self.scale3(x3).mean([2,3])  # [B, 1280]
        
        # 融合特征并回归
        f = torch.cat([f1, f2, f3], dim=1)  # [B, 3840]
        score = self.fusion(f)
        return score.squeeze(1)

def init_musiq_model(device):
    """初始化MUSIQ模型"""
    model = MUSIQ_Model().to(device)
    model.eval()
    return model

def musiq_metric(images, model, device):
    """
    计算MUSIQ指标
    images: [B, C, H, W] (0-1范围，float32)
    model: MUSIQ_Model实例
    device: 计算设备
    返回: 该批次的平均MUSIQ分数（越高越好，0-1）
    """
    # 预处理：归一化到[-1,1]
    images = (images - 0.5) * 2.0
    
    with torch.no_grad():
        scores = model(images)
    return float(scores.mean().cpu().numpy())

# ===================== 批量指标计算函数（核心优化，适配旧版torchmetrics） =====================
def init_batch_metrics(device):
    """初始化批量SSIM/PSNR/MS-SSIM指标计算器（适配torchmetrics<1.0）"""
    # SSIM：默认参数（win_size=11，符合论文常用设置）
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    # PSNR：数据范围0-1
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    # MS-SSIM：移除power_factors，仅保留兼容参数（旧版默认4尺度）
    ms_ssim_metric = MultiScaleStructuralSimilarityIndexMeasure(
        data_range=1.0,
        kernel_size=11  # 替代原win_size，旧版唯一兼容的核心参数
    ).to(device)
    return ssim_metric, psnr_metric, ms_ssim_metric

def init_inception_model(device):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_model = InceptionV3([block_idx]).to(device)
    inception_model.eval()
    return inception_model

def preprocess_for_fid(images, device):
    """将图像预处理为InceptionV3要求的格式"""
    # 1. 调整大小到299x299（FID标准）
    transform = Compose([
        Resize(299, interpolation=transforms.InterpolationMode.BILINEAR),
        CenterCrop(299),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. 确保图像在0-1范围，调整维度
    images = torch.clamp(images, 0.0, 1.0)
    processed = transform(images)
    return processed.to(device)

def calculate_fid_batch(origin_batch, adv_batch, inception_model, device):
    """使用pytorch-fid库批量计算FID"""
    # 预处理图像
    origin_processed = preprocess_for_fid(origin_batch, device)
    adv_processed = preprocess_for_fid(adv_batch, device)
    
    # 提取特征
    def get_features(batch):
        with torch.no_grad():
            features = inception_model(batch)[0]
        # 调整形状 [B, 2048, 1, 1] -> [B, 2048]
        features = features.squeeze(-1).squeeze(-1)
        return features.cpu().numpy()
    
    origin_feat = get_features(origin_processed)
    adv_feat = get_features(adv_processed)
    
    # 使用pytorch-fid的官方计算函数
    fid_score_val = fid_score.calculate_frechet_distance(
        np.mean(origin_feat, axis=0),
        np.cov(origin_feat, rowvar=False),
        np.mean(adv_feat, axis=0),
        np.cov(adv_feat, rowvar=False)
    )
    return fid_score_val

if __name__ == '__main__':
    BATCH_SIZE = 64
    IMG_SIZE = 512
    # IMG_ROOT = r"exp/260112_optim_RGB_direct_ablation" 
    # IMG_ROOT = r"exp/260112_optimlatent_target_loss"
    # IMG_ROOT=r"exp/260112_optimlatent_target_loss_ssd300"
    IMG_ROOT=r'exp/260112_optim_RGB_withVAE_ablation'
    base_file_name=os.path.basename(IMG_ROOT)
    exp_root = os.path.join( r"exp/test_eval" , base_file_name)
    os.makedirs(exp_root, exist_ok=True) 

    # --------------------------
    # 2. 验证集预处理（无数据增强！）
    # --------------------------
    val_transform = transforms.Compose([
        ResizeMaxEdge(max_edge_size=IMG_SIZE), 
        PadToFixedSize(target_size=IMG_SIZE),  
        transforms.ToTensor(),  # 转为张量
    ])

    # --------------------------
    # 3. 加载验证集
    # --------------------------
    target_list = ["origin.jpg", "adv_example.pt"]
    img_dataset = CustomFolderDataset(
        root_dir=IMG_ROOT,
        transform=val_transform,
        target_image_name_list=target_list
    )

    img_loader = DataLoader(
        img_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  
        num_workers=2,
        pin_memory=True,
    )

    # 初始化设备和指标
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ssim_metric, psnr_metric, ms_ssim_metric = init_batch_metrics(device)  # 新增MS-SSIM
    inception_model = init_inception_model(device)  # 初始化FID用的Inception模型
    
    # 初始化无参考指标模型
    nrqm_ma_model = init_nrqm_ma_model(device)  # NRQM (Ma)
    musiq_model = init_musiq_model(device)      # MUSIQ

    # 初始化累计指标
    total_fid = 0.0
    total_vif = 0.0                # VIF累计值
    total_niqe_origin = 0.0        # 原始图像NIQE累计
    total_niqe_adv = 0.0           # 对抗图像NIQE累计
    total_nrqm_ma_origin = 0.0     # 原始图像NRQM (Ma)累计
    total_nrqm_ma_adv = 0.0        # 对抗图像NRQM (Ma)累计
    total_musiq_origin = 0.0       # 原始图像MUSIQ累计
    total_musiq_adv = 0.0          # 对抗图像MUSIQ累计
    
    num_batches = 0
    num_valid_vif_samples = 0      # 有效VIF样本数
    num_valid_niqe_samples = 0     # 有效NIQE样本数
    num_valid_nrqm_samples = 0     # 有效NRQM样本数
    num_valid_musiq_samples = 0    # 有效MUSIQ样本数

    # 处理每个批次
    pbar = tqdm(enumerate(img_loader), total=len(img_loader), desc="Processing images", unit="batch")
    for batch_idx, (folder_name, img_tensors, img_names) in pbar:
        adv_images  = img_tensors[1]
        origin_images = img_tensors[0]
        
        # 确保张量设备和类型一致
        adv_images = align_tensor_dtype(adv_images, origin_images)
        adv_images = move_to_gpu_and_cast_dtype(adv_images, device)
        origin_images = move_to_gpu_and_cast_dtype(origin_images, device)
        
        batch_size_cur = origin_images.shape[0]
        num_batches += 1

        # ===================== 批量计算SSIM/PSNR/MS-SSIM =====================
        ssim_batch = ssim_metric(adv_images, origin_images)
        psnr_batch = psnr_metric(adv_images, origin_images)
        ms_ssim_batch = ms_ssim_metric(adv_images, origin_images)  # 计算MS-SSIM

        # ===================== 批量计算FID =====================
        try:
            fid_batch = calculate_fid_batch(origin_images, adv_batch=adv_images, 
                                          inception_model=inception_model, device=device)
            total_fid += fid_batch
        except Exception as e:
            fid_batch = 0.0
            print(f"\nBatch {batch_idx} FID Error: {e}")

        # ===================== 批量计算VIF =====================
        try:
            # 确保图像值域在0-1之间
            origin_vif = torch.clamp(origin_images, 0.0, 1.0)
            adv_vif = torch.clamp(adv_images, 0.0, 1.0)
            vif_batch = vifp_mscale(origin_vif, adv_vif)
            total_vif += vif_batch * batch_size_cur
            num_valid_vif_samples += batch_size_cur
        except Exception as e:
            vif_batch = 0.0
            print(f"\nBatch {batch_idx} VIF Error: {e}")

        # ===================== 批量计算NIQE（无参考） =====================
        try:
            origin_niqe = niqe_metric(origin_images)
            adv_niqe = niqe_metric(adv_images)
            total_niqe_origin += origin_niqe * batch_size_cur
            total_niqe_adv += adv_niqe * batch_size_cur
            num_valid_niqe_samples += batch_size_cur
        except Exception as e:
            origin_niqe = 0.0
            adv_niqe = 0.0
            print(f"\nBatch {batch_idx} NIQE Error: {e}")

        # ===================== 批量计算NRQM (Ma)（无参考） =====================
        try:
            origin_nrqm = nrqm_ma_metric(origin_images, nrqm_ma_model, device)
            adv_nrqm = nrqm_ma_metric(adv_images, nrqm_ma_model, device)
            total_nrqm_ma_origin += origin_nrqm * batch_size_cur
            total_nrqm_ma_adv += adv_nrqm * batch_size_cur
            num_valid_nrqm_samples += batch_size_cur
        except Exception as e:
            origin_nrqm = 0.0
            adv_nrqm = 0.0
            print(f"\nBatch {batch_idx} NRQM (Ma) Error: {e}")

        # ===================== 批量计算MUSIQ（无参考） =====================
        try:
            origin_musiq = musiq_metric(origin_images, musiq_model, device)
            adv_musiq = musiq_metric(adv_images, musiq_model, device)
            total_musiq_origin += origin_musiq * batch_size_cur
            total_musiq_adv += adv_musiq * batch_size_cur
            num_valid_musiq_samples += batch_size_cur
        except Exception as e:
            origin_musiq = 0.0
            adv_musiq = 0.0
            print(f"\nBatch {batch_idx} MUSIQ Error: {e}")

        # 更新进度条
        pbar.set_postfix({
            'Batch SSIM': f"{ssim_batch.item():.4f}",
            'Batch PSNR': f"{psnr_batch.item():.2f}dB",
            'Batch MS-SSIM': f"{ms_ssim_batch.item():.4f}",
            'Batch VIF': f"{vif_batch:.4f}",
            'Batch FID': f"{fid_batch:.2f}",
            'Origin NIQE': f"{origin_niqe:.2f}",
            'Adv NIQE': f"{adv_niqe:.2f}",
            'Origin NRQM': f"{origin_nrqm:.2f}",
            'Adv NRQM': f"{adv_nrqm:.2f}",
            'Origin MUSIQ': f"{origin_musiq:.4f}",
            'Adv MUSIQ': f"{adv_musiq:.4f}"
        })

    # ===================== 计算最终平均指标 =====================
    # 全参考指标
    avg_ssim = ssim_metric.compute().item()
    avg_psnr = psnr_metric.compute().item()
    avg_ms_ssim = ms_ssim_metric.compute().item()
    avg_fid = total_fid / num_batches if num_batches > 0 else 0.0
    avg_vif = total_vif / num_valid_vif_samples if num_valid_vif_samples > 0 else 0.0
    
    # 无参考指标
    avg_niqe_origin = total_niqe_origin / num_valid_niqe_samples if num_valid_niqe_samples > 0 else 0.0
    avg_niqe_adv = total_niqe_adv / num_valid_niqe_samples if num_valid_niqe_samples > 0 else 0.0
    avg_nrqm_ma_origin = total_nrqm_ma_origin / num_valid_nrqm_samples if num_valid_nrqm_samples > 0 else 0.0
    avg_nrqm_ma_adv = total_nrqm_ma_adv / num_valid_nrqm_samples if num_valid_nrqm_samples > 0 else 0.0
    avg_musiq_origin = total_musiq_origin / num_valid_musiq_samples if num_valid_musiq_samples > 0 else 0.0
    avg_musiq_adv = total_musiq_adv / num_valid_musiq_samples if num_valid_musiq_samples > 0 else 0.0

    # 重置指标计算器
    ssim_metric.reset()
    psnr_metric.reset()
    ms_ssim_metric.reset()

    # ===================== 输出和保存结果 =====================
    print("\n==================== 批量指标计算结果 ====================")
    print("---------- 全参考指标（Full-Reference） ----------")
    print(f"总批次数量: {num_batches}")
    print(f"平均SSIM: {avg_ssim:.4f} (越接近1越好)")
    print(f"平均PSNR: {avg_psnr:.2f} dB (越高越好)")
    print(f"平均MS-SSIM: {avg_ms_ssim:.4f} (越接近1越好)")
    print(f"平均VIF: {avg_vif:.4f} (越高越好)")
    print(f"平均FID: {avg_fid:.2f} (越低越好)")
    
    print("\n---------- 无参考指标（No-Reference） ----------")
    print("NIQE (越小越好):")
    print(f"  原始图像平均NIQE: {avg_niqe_origin:.2f}")
    print(f"  对抗图像平均NIQE: {avg_niqe_adv:.2f}")
    print("NRQM (Ma) (越高越好，0-10):")
    print(f"  原始图像平均NRQM: {avg_nrqm_ma_origin:.2f}")
    print(f"  对抗图像平均NRQM: {avg_nrqm_ma_adv:.2f}")
    print("MUSIQ (越高越好，0-1):")
    print(f"  原始图像平均MUSIQ: {avg_musiq_origin:.4f}")
    print(f"  对抗图像平均MUSIQ: {avg_musiq_adv:.4f}")

    # 保存结果
    metrics_result = {
        "total_batches": int(num_batches),
        "experiment_dataset": str(IMG_ROOT),
        # 全参考指标
        "full_reference_metrics": {
            "average_ssim": float(avg_ssim),
            "average_psnr": float(avg_psnr),
            "average_ms_ssim": float(avg_ms_ssim),
            "average_vif": float(avg_vif),
            "average_fid": float(avg_fid)
        },
        # 无参考指标
        "no_reference_metrics": {
            "niqe": {
                "origin_average": float(avg_niqe_origin),
                "adv_average": float(avg_niqe_adv),
                "note": "越小越好，衡量自然图像质量"
            },
            "nrqm_ma": {
                "origin_average": float(avg_nrqm_ma_origin),
                "adv_average": float(avg_nrqm_ma_adv),
                "note": "越高越好，0-10，感知质量分数"
            },
            "musiq": {
                "origin_average": float(avg_musiq_origin),
                "adv_average": float(avg_musiq_adv),
                "note": "越高越好，0-1，多尺度质量分数"
            }
        },
        # 指标说明
        "metric_notes": {
            "SSIM": "单尺度结构相似性，范围[0,1]，越接近1越好",
            "MS-SSIM": "多尺度结构相似性，范围[0,1]，越接近1越好",
            "PSNR": "峰值信噪比，单位dB，越高越好",
            "VIF": "视觉信息保真度，越高越好",
            "FID": "Fréchet Inception距离，越低越好",
            "NIQE": "无参考自然图像质量，越小越好",
            "NRQM (Ma)": "无参考感知质量，0-10，越高越好",
            "MUSIQ": "多尺度无参考质量，0-1，越高越好"
        }
    }
    with open(os.path.join(exp_root, "batch_metrics.yaml"), 'w', encoding='utf-8') as f:
        yaml.dump(metrics_result, f, indent=4, sort_keys=False)
    print(f"\n结果已保存至: {os.path.join(exp_root, 'batch_metrics.yaml')}")

