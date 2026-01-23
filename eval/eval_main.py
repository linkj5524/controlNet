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

# 导入成熟的指标计算库
# --------------------------
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# 1. 创建参数解析器
parser = argparse.ArgumentParser(description="Adversarial Attack Main Program")  # 程序描述

# 2. 添加命令行参数

parser.add_argument('--attack_config_path', type=str, 
                    default="models/attack_config.yaml",
                      help="attack config path")


# 3. 解析命令行参数
args = parser.parse_args()


# ===================== 批量指标计算函数（核心优化） =====================
# ===================== 批量指标计算函数（使用成熟库） =====================
def init_batch_metrics(device):
    """初始化批量SSIM/PSNR指标计算器（支持GPU）"""
    # SSIM：默认参数（win_size=11，符合论文常用设置）
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    # PSNR：数据范围0-1
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    return ssim_metric, psnr_metric

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



# if __name__ == '__main__':


#     root_path=os.path.dirname(__file__)
#     attack_config_path=args.attack_config_path
#     adv_config=load_yaml_config(attack_config_path)



    

#     BATCH_SIZE = adv_config["experiment_params"]["batch_size"]
#     IMG_SIZE = adv_config["experiment_params"]["image_size"]  
#     # IMG_ROOT=adv_config["experiment_params"]["dataset_path"]
#     IMG_ROOT=r"./exp/test"
#     BATCH_SIZE=2
#     # --------------------------
#     # 2. 验证集预处理（无数据增强！）
#     # --------------------------
#     # 注意：验证集仅做resize、中心裁剪、归一化，禁止随机增强（保证评估公平）
#     val_transform = transforms.Compose([
#         ResizeMaxEdge(max_edge_size=IMG_SIZE), 
#         PadToFixedSize(target_size=IMG_SIZE),  
#         transforms.ToTensor(),  # 转为张量
#     ])



#     # --------------------------
#     # 3. 加载验证集
#     # --------------------------
#     target_list=["origin.jpg","adv_example.pt"]
#     # ImageFolder自动按文件夹名称分配类别标签（0-999）
#     img_dataset = CustomFolderDataset(
#         root_dir=IMG_ROOT,
#         transform=val_transform,
#         target_image_name_list=target_list
#     )


#     img_loader = DataLoader(
#         img_dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=False,  
#         num_workers=adv_config["experiment_params"]["num_workers"],
#         pin_memory=True,
#         # collate_fn=custom_collate_fn
#     )



#     # 1. 初始化多模型检测器
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     detector = ObjectDetection(device=device)
                                            

#     # 2. 加载检测模型（可加载多个，按需选择）
#     # 加载YOLOv8（需指定权重路径，如yolov8n.pt，可从ultralytics官网下载）
#     detector.load_model(model_type="yolov8",model_path=r"./models/yolov8m.pt")
#     # 加载Faster R-CNN（自动使用COCO预训练权重）
#     detector.load_model(model_type="fasterrcnn")
#     # 加载SSD300（自动使用COCO预训练权重）
#     detector.load_model(model_type="ssd300")
#     detector.load_model(model_type="yolov11",model_path=r"./models/yolo11m.pt")   
#     # 3. 构造测试输入（模拟批量图像，shape [B, C, H, W]，值范围0-1）
#     batch_size = 1




#     exp_root=adv_config["experiment_params"]["experiment_path"]
#     exp_root=r"exp/test_eval"
#     # 获取图片文件名,去除后缀

#     os.makedirs(exp_root,exist_ok=True) 
#     pbar = tqdm(enumerate(img_loader), total=len(img_loader), desc="Processing images", unit="batch")
#     for batch_idx, (folder_name, img_tensors, img_names) in pbar:
        
#         file_roots=[os.path.join(exp_root,name) for name in folder_name]
#         for file_root_i in file_roots:
#             os.makedirs(file_root_i,exist_ok=True)

#         origin_images=img_tensors[1]
#         adv_images=img_tensors[0]

#         adv_images=torch.load(r'exp\test\000000000785\adv_example.pt')
#         # yolov11 初始化，作为参考
#         result_gt,object_class=detector.detect(img=origin_images, model_type="yolov11", file_path=file_roots,file_name="origin_yolov11.jpg")

#         result_gt,object_class=filter_max_box_per_batch(result_gt,object_class) 


#         result_gt11,object_class111=detector.detect(img=adv_images, model_type="yolov11", file_path=file_roots,file_name="adv_yolov11.jpg")
#         #评估
    







#         print("\n=== YOLOv8 检测结果 ===")
#         yolo_results = detector.detect_eval(images=adv_images, model_type="yolov8", file_path=file_roots,file_name="adv_yolov8.jpg")

#         # match_yolo=match_detection_boxes(result_gt,yolo_results)

#         # --------------------------
#         # 4.2 Faster R-CNN检测
#         # --------------------------
#         print("\n=== Faster R-CNN 检测结果 ===")
#         frcnn_results = detector.detect_eval(images=adv_images, model_type="fasterrcnn",file_path=file_roots,file_name='adv_fasterrcnn.jpg')



#         # --------------------------
#         # 4.3 SSD300检测
#         # --------------------------
#         print("\n=== SSD300 检测结果 ===")
#         ssd_results = detector.detect_eval(images=adv_images, model_type="ssd300",file_path=file_roots,file_name='adv_ssd300.jpg')



#         print("\n=== YOLOv8 检测结果 ===")
#         yolo_results = detector.detect_eval(images=origin_images, model_type="yolov8", file_path=file_roots,file_name="origin_yolov8.jpg")

#         match_yolo=match_detection_boxes(result_gt,yolo_results)

#         # --------------------------
#         # 4.2 Faster R-CNN检测
#         # --------------------------
#         print("\n=== Faster R-CNN 检测结果 ===")
#         frcnn_results = detector.detect_eval(images=origin_images, model_type="fasterrcnn",file_path=file_roots,file_name='origin_fasterrcnn.jpg')



#         # --------------------------
#         # 4.3 SSD300检测
#         # --------------------------
#         print("\n=== SSD300 检测结果 ===")
#         ssd_results = detector.detect_eval(images=origin_images, model_type="ssd300",file_path=file_roots,file_name='origin_ssd300.jpg')



if __name__ == '__main__':


    


    BATCH_SIZE = 64
    IMG_SIZE =512
    IMG_ROOT = r"exp/260112_optim_RGB_direct_ablation" 
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
    ssim_metric, psnr_metric = init_batch_metrics(device)
    inception_model = init_inception_model(device)  # 初始化FID用的Inception模型

    total_fid = 0.0
    num_batches = 0


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

        # ===================== 批量计算SSIM/PSNR（torchmetrics库） =====================
        ssim_batch = ssim_metric(adv_images, origin_images)
        psnr_batch = psnr_metric(adv_images, origin_images)

        # ===================== 批量计算FID（pytorch-fid库） =====================
        try:
            fid_batch = calculate_fid_batch(origin_images, adv_batch=adv_images, 
                                          inception_model=inception_model, device=device)
            total_fid += fid_batch
        except Exception as e:
            fid_batch = 0.0
            print(f"\nBatch {batch_idx} FID Error: {e}")

        # 更新进度条
        pbar.set_postfix({
            'Batch SSIM': f"{ssim_batch.item():.4f}",
            'Batch PSNR': f"{psnr_batch.item():.2f}dB",
            'Batch FID': f"{fid_batch:.2f}"
        })

    # ===================== 计算最终平均指标 =====================
    avg_ssim = ssim_metric.compute().item()  # torchmetrics自动累计所有批次的平均
    avg_psnr = psnr_metric.compute().item()
    avg_fid = total_fid / num_batches if num_batches > 0 else 0.0

    # 重置指标计算器
    ssim_metric.reset()
    psnr_metric.reset()

    # ===================== 输出和保存结果 =====================
    print("\n==================== 批量指标计算结果 ====================")
    print(f"总批次数量: {num_batches}")
    print(f"平均SSIM: {avg_ssim:.4f} (越接近1越好)")
    print(f"平均PSNR: {avg_psnr:.2f} dB (越高越好)")
    print(f"平均FID: {avg_fid:.2f} (越低越好)")

    # 保存结果

    metrics_result = {
        "total_batches": int(num_batches),  # 确保是原生int
        "average_ssim": float(avg_ssim),    # 转换为原生float
        "average_psnr": float(avg_psnr),    # 转换为原生float
        "average_fid": float(avg_fid),      # 转换为原生float
        "experiment_dataset": str(IMG_ROOT)
    }
    with open(os.path.join(exp_root, "batch_metrics.yaml"), 'w', encoding='utf-8') as f:
        yaml.dump(metrics_result, f, indent=4)
    print(f"\n结果已保存至: {os.path.join(exp_root, 'batch_metrics.yaml')}")