
import time

from omegaconf import OmegaConf

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
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.amp import autocast, GradScaler  
# 本地的包
## 添加本地包路径,即上一级的路径
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from ldm.util import instantiate_from_config as instantiate_from_config_vae
# from util import *

# 
from adv_attack.util import *

##
sys.path.append(os.path.dirname(__file__))
from object_detection_class import *
from sam import *
from captioner_blip_model import *
from sd_inpaint import *
from IDG_util.attribution_methods.saliencyMethods import * 
from vae import *
from ADVLogo_attack_tools import *
from fgsm_attack_tools import *

class ADV_ATTACK:
    def __init__(self, config_path:str='./models/cldm_v15.yaml',
                  model_path:str='./models/control_sd15_scribble.pth', 
                  device:torch.device=torch.device("cuda"),
                  detect_model_type:str='yolov11',
                  model_path_object_detection:str=None,
                  sam_model_type:str="vit_h",
                  sam_checkpoint_path:str="sam_vit_h_4b8939.pth",
                  captioner_model_name:str=r"./models/Salesforceblip_image_captioning_large",
                  inpaint_model_path:str=r"./sdxl-inpaint-model",
                  vae_model_path:str=r"models\sd_vae_ft_mse",
                  kwargs:dict=None,
                  detect_params:dict=None
                  ):
        """
        初始化对抗攻击类
        
        参数:
            config_path: 模型配置文件路径 (默认 "./models/cldm_v15.yaml")
            device: 运行设备 (默认 "cuda")
            model_type: 目标检测模型类型 (默认 "yolov5")
            class_names: 目标检测模型类别 (默认 ['person'])
            device: 运行设备 (默认 "cuda")
        """
        

        # 默认参数配置
        if kwargs:
            self.default_params = kwargs
        else:
            self.default_params = {
            "prompt": "covered with jungle camouflage pattern, high detail, realistic texture, 8k, ultra sharp",
            "a_prompt": "",
            "n_prompt": "blurry, low resolution, ugly, deformed, noisy texture, pixelated, unrealistic, bad detail, distorted pattern",
            "num_samples": 1,
            "ddim_steps": 30,
            "guess_mode": False,
            "strength": 1.0,
            "scale": 9,
            "scale_optim":9, # 优化过程的控制
            "seed": 42,
            "eta": 0.0,
            "save_memory": True,
            "optim_epochs":30, # 默认 20
            "latent_fit_optim_epochs":5,
            "attribution_loss_weight" :0,
            "TV_loss_weight":0,
            "lr":5e-2, # V2 5e-3 ；v3 5e-3
            "conext_loss_weight":100, # 100
            "perceptual_loss_weight":0
        }
            # scale  encode 8,优化为2，目前测试效果比较好
            # 后续改成一样，效果需要试验
    
        # 加载模型配置
        self.config_path = config_path
        self.model_path = model_path
        self.device = device
        self.class_names_ymal=detect_params['nclass_yaml_path']

        self.detect_model_type = detect_model_type
        self.model_path_object_detection=model_path_object_detection
        self.sam_model_type = sam_model_type
        self.sam_checkpoint_path = sam_checkpoint_path
        self.captioner_model_name=captioner_model_name
        self.inpaint_model_path=inpaint_model_path
        self.vae_model_path=vae_model_path
        self.detect_params=detect_params




    def set_params(self, **kwargs):
        """更新攻击参数"""
        for key, value in kwargs.items():
            if key in self.default_params:
                self.default_params[key] = value
            else:
                print(f"警告: 参数 {key} 不是有效参数，将被忽略")



# 初始化controlnet模型
    def init_controlnet(self):
        """初始化ControlNet模型"""
                # 初始化模型
        self.model = create_model(self.config_path).cpu()
        self.model.load_state_dict(load_state_dict(self.model_path, location='cuda'),strict=False)
        self.ddim_sampler = DDIMSampler(self.model)


    # 模型destroy
    def destroy_controlnet(self):
        """销毁模型"""
        # 判断模型是否已经初始化
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'ddim_sampler'):
            del self.ddim_sampler

        torch.cuda.empty_cache()



    # def init_object_detection(self,device=None):
    #     """初始化目标检测模型"""
    #     if device is None:
    #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     # 加载检测模型
    #     self.object_detection=ObjectDetection(yaml_path=self.class_names_ymal,
    #                                           device=device,
    #                                           **self.detect_params)
        
        
    #     self.object_detection.load_model( model_type=self.detect_model_type,
    #                                 model_path=self.model_path_object_detection
    #                                 )

    def init_object_detection(self,device=None,**kwargs):
        """初始化目标检测模型"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 加载检测模型


        self.object_detection=ObjectDetection(device=device,
                                              **self.detect_params)
        
        model_type=kwargs['attack_model']["model_type"]
        modelt_path=kwargs['attack_model']["model_path"]
        self.object_detection.load_model( model_type=model_type,
                                    model_path=modelt_path
                                    )
        
    def init_object_detection_return(self,device=None,detect_params=None):
        """初始化目标检测模型"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 加载检测模型


        object_detection=ObjectDetection(device=device,
                                            **detect_params)
        
        model_type=detect_params['attack_model']["model_type"]
        modelt_path=detect_params['attack_model']["model_path"]
        object_detection.load_model( model_type=model_type,
                                    model_path=modelt_path
                                    )
        return object_detection

    def destroy_object_detection(self,object_detection):
        """销毁目标检测模型"""
        if object_detection is not None:
            del object_detection
        # 清空内存
        torch.cuda.empty_cache()


    def detect_val(self,input_image,
                   input_path,
                   input_file_name,
                   detect_params=None 
                   ):
        """初始化目标检测模型"""

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 加载检测模型
        object_detection_val=ObjectDetection(device=device,
                                              ** detect_params)
        detect_result_list={}
        for val_model_key in detect_params['val_model1'].keys():
            model_type_val=detect_params['val_model1'][val_model_key]["model_type"]
            model_path_val=detect_params['val_model1'][val_model_key]["model_path"]
            if len(model_path_val)<=0 :
                object_detection_val.load_model( model_type=model_type_val)
            else:                        
                object_detection_val.load_model( model_type=model_type_val,
                                        model_path=model_path_val
                                        )
            # 检测
            image_name=input_file_name+model_type_val+'.jpg'
            temp_result,_=object_detection_val.detect_eval(images=input_image,
                                                model_type=model_type_val,
                                                file_path=input_path,
                                                file_name=image_name)
            # 删除模型
            if    model_type_val in object_detection_val.models:                     
                    del object_detection_val.models[model_type_val]
            detect_result_list[model_type_val]=temp_result
            
            # 清空内存
            torch.cuda.empty_cache()
        return detect_result_list
            
        







    def generate_adversarial_main(self,background_imag=None, exp_path=r'./exp',images_path=['name'],mask_select_statues=0,params=None):
        """
        生成对抗样本
        
        参数:
            control_image: 控制图像 (用于ControlNet)
            params: 覆盖默认参数的字典
            
        返回:
            生成的对抗图像列表
        """
        # 图像预处理，
        background_imag=self.pad_to_square(background_imag)
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
        


        # 初始化模型
        self.init_controlnet()
        if params["save_memory"]:
            self.model.low_vram_shift(is_diffusing=True)
        background_imag=background_imag.unsqueeze(0)
        background_imag_n1_1=background_imag*2-1
        latent_input = self.imgTensor_to_latent(background_imag_n1_1)

        # 
        image_test=self.latent_to_imgTensor01(latent_input)
        tensor2picture((image_test+1)/2,"image_test.png")





        # background——image 的文本描述提取
        blip_model, blip_processor, blip_device=init_image_captioner(self.captioner_model_name)
        background_imag_caption = image_captioner_process(blip_model, blip_processor, blip_device, background_imag)

        print(f"\n背景图像描述: {background_imag_caption}")
        destroy_image_captioner(blip_model) 






       # detect model 初始化
        self.init_object_detection()


        all_exp_root=[]
        for image_name in images_path:
            image_name = os.path.splitext(os.path.basename(image_name))[0]

            exp_root_dir=os.path.join(exp_path,f"{image_name}")
            os.makedirs(exp_root_dir,exist_ok=True)
            all_exp_root.append(exp_root_dir)

        result_gt,object_class =self.object_detection.detect(background_imag,file_path=all_exp_root,file_name='detect_object_ref.jpg',grad_status=True)
            
        








        input_point_list=self.yolo_boxes_to_corners(result_gt['boxes'])
        
        # 基于输入的background_imag，利用sam，得到mask，返回mask。
        # 模型初始化
        sam_predicter=init_sam(model_type=self.sam_model_type, checkpoint_path=self.sam_checkpoint_path)
        # 处理,注意mask——logic 的维度，是否是多个通道
        sam_img_np, sam_masks_logic_mutil_list, sam_masks_tensor_all, sam_scores_all_list=segment_tensor(sam_predicter,
                                                                                                          background_imag, 
                                                                                                          input_labels_batch=object_class,
                                                                                                          input_points_batch=input_point_list,
                                                                                                          mutil_mask=True)
        
        
        
        

        # visualize_sam(background_imag, masks_logic_mutil, scores)
        destroy_sam(sam_predicter)



        # control 处理

        # mask 选择
        mask_logic_np_select, mask_tensor_select=self.select_mask_by_criteria(
            masks_logic_mutil_all=sam_masks_logic_mutil_list,
            masks_tensor_all=sam_masks_tensor_all,
            scores_all=sam_scores_all_list,
            exp_path=all_exp_root,
            mask_select_statues=mask_select_statues
        )


        # 创建mask_logic_temp,里面全是True,numpy
        # mask_logic_np_for_optim 用于生成control，mask_logic_np_select用于最后生成的图像crop 到背景图像
        mask_logic_np_for_optim = mask_logic_np_select
           
        canny_for_visual,control_image=self.canny_with_mask_invert(background_imag,mask_logic_np_for_optim)
        # 保存图片
        for i,exp_root_dir in enumerate(all_exp_root):
            tensor2picture(canny_for_visual[i],os.path.join(exp_root_dir, 'control.jpg'))

       



        # 初始化模型
        self.init_controlnet()
        if params["save_memory"]:
            self.model.low_vram_shift(is_diffusing=False)




        # 添加b
        # 判断control 的维度
        if control_image.dim()==3:
            control_image=control_image.unsqueeze(0)

        


        
        # 获取batch
        B,C,H, W= background_imag.shape
        shape = (4, H // 8, W // 8)
        
        # c_concat 草图控制；c_crossattn 跨模态控制：正向和附加的文本提示;文本内容默认用clip编码
        cond = {
            "c_concat": [control_image],
            "c_crossattn": [
                self.model.get_learned_conditioning(
                    background_imag_caption  
                )
            ]
        }
        un_cond = {
            "c_concat": None if params["guess_mode"] else [control_image],
            "c_crossattn": [
                self.model.get_learned_conditioning(
                    [params["n_prompt"]] * B
                )
            ]
        }
 










        if params["save_memory"]:
            self.model.low_vram_shift(is_diffusing=True)


        self.model.control_scales = (
            [params["strength"] * (0.825 ** float(12 - i)) for i in range(13)]
            if params["guess_mode"]
            else [params["strength"]] * 13
        )  # Magic number. IDK why
        
        object_image=self.extract_mask_content(background_imag,mask_logic_np_select)

        # tensor2picture(object_image,"object_image.png") 
        # 输入是-1~1
        # 将0,1 转换成-1,1
        object_image_n1_1=object_image*2-1
        latent_input = self.imgTensor_to_latent(object_image_n1_1)

        # 
        image_test=self.latent_to_imgTensor01(latent_input)
        tensor2picture((image_test+1)/2,"image_test.png")
        # 参考归因
        attributions_ref = IG_Detection(
            input_img=object_image,
            det_model=self.object_detection,
            steps=50,
            batch_size=10,
            alpha_star=1.0,
            baseline=0.0,
            target_obj_idx=0
        )

        # 4. 可视化结果
        if attributions_ref is not None:
            # attribution_ref_path=os.path.join(exp_path, 'attribution_ref.png')
            visualize_attribution(object_image, attributions_ref, save_path=all_exp_root,file_name_pre='attribution_ref')
        else:
            print("Attribution failed!")




        # 设置采样参数
        self.ddim_sampler.make_schedule(ddim_num_steps=params["ddim_steps"], ddim_eta=params["eta"], verbose=False)

        

        # 使用封装的ddim进行逆采样
        ## latent_start 表示逆采样的结果（噪声最大的latent），out 表示所有的中间结果
        with torch.no_grad():
            latent_start,out=self.ddim_sampler.encode_return_all(x0=latent_input, c=cond, t_enc=params["ddim_steps"], use_original_steps=False, return_intermediates=True,
            unconditional_guidance_scale=params["scale"], unconditional_conditioning=un_cond, callback=None)

        # 获取目标检测模型的输出，也可以直接传入这些已知的信息,用于后续计算，确保只用某个目标
        result_gt_temp,class_name=self.object_detection.detect(object_image,file_path=all_exp_root,file_name='object_detect.jpg',grad_status=False)



        # 保存中间结果
        # 对初始latent进行优化
        # 需要 1. 优化目标 2. 优化器 3. 优化参数 4. 后处理函数
        # 优化目标：目标检测模型的输出与原始的检测框，类别等的差值
        # 优化器：Adam
        # 优化参数：latent
        # 后处理函数，根据检测模型的输出，得到结果，并进行优化











        latent_start=latent_start.detach().clone()
        latent_start.requires_grad = True
        optimizer = torch.optim.Adam([latent_start], lr=params["lr"])
        cross_entro_loss = YOLOv11DetectionLoss(** self.default_params)
        attr_loss_l2 = nn.MSELoss()
        TV_Loss=TVLoss()
        #开始步骤
        t_start=self.ddim_sampler.ddim_timesteps[-1]
        pbar = tqdm(range(params["optim_epochs"]), desc="Optimizing Adversarial Sample", unit="epoch")
        for epoch in pbar:
            # 循环，优化
            end_latent=self.ddim_sampler.decode(  latent_start, cond, t_start, unconditional_guidance_scale=params["scale_optim"], unconditional_conditioning=un_cond,
                use_original_steps=False, callback=None)



            # 转换成图片
            image=self.latent_to_imgTensor01(end_latent)
            image_object_on_background=batched_tensor_mask_overlay(background_imag,image,mask_logic_np_select)
           
            result_temp,_=self.object_detection.detect(image,file_path=all_exp_root,file_name='result_generate.jpg',grad_status=True)
            # 目标检测模型的输出
            
            result,_  =self.object_detection.detect(image_object_on_background,file_path=all_exp_root,file_name='result_temp.jpg',grad_status=True)

            if image is None:
                print("对抗样本为空")
                continue
            attributions = IG_Detection(
                input_img=image,
                det_model=self.object_detection, 
                steps=50,
                batch_size=10,
                alpha_star=1.0,
                baseline=0.0,
                target_obj_idx=0
            )

            # # 4. 可视化结果
            # if attributions is not None:
            #     visualize_attribution(background_imag, attributions, save_path=r"exp\attribution")
            # else:
            #     print("Attribution failed!")

            attr_loss=attr_loss_l2(attributions,attributions_ref)
            print(f"attr_loss:{attr_loss}")
            loss ,loss_dict= cross_entro_loss(result, result_gt)
            print(f"total_loss:{loss}")
            print(f"class_loss:{loss_dict['class_loss']}")
            print(f"label_gt:{result_gt['labels']},label_pred:{result['labels']}")
            print(f"score_gt:{result_gt['scores']},score_pred:{result['scores']}")
            tv_loss=TV_Loss(image)
            print(f"tv_loss:{tv_loss}")
            optimizer.zero_grad()
            

            

            ( params['TV_loss_weight'] *tv_loss+params["attribution_loss_weight"]*attr_loss-loss_dict['class_loss']).backward()

                          
               
            optimizer.step()
            # 手动清理变量，帮助回收内存
            del loss,loss_dict,tv_loss,attr_loss
            torch.cuda.empty_cache()


        # image_tensor=self.latent_to_imgTensor01(end_latent)

        image_adv=self.tensor_01_to_numpy_255(image_object_on_background)
        # 保存对抗样本
        for i in range(len(all_exp_root)):

            adv_path=os.path.join(all_exp_root[i], 'adv_example.jpg')
            cv2.imwrite(adv_path,image_adv)




        
        
        return 

# 现用controlnet模型，生成大致的纹理，再优化内容
    def generate_adversarial_main_two_stage(self,background_imag=None, exp_path=r'./exp',images_path=None,mask_select_statues=0,params=None):
        """
        生成对抗样本
        
        参数:
            control_image: 控制图像 (用于ControlNet)
            params: 覆盖默认参数的字典
            
        """
        """
            ====================================================
            ============ 图像预处理，初始目标的获取 ==============
            ====================================================
        """
        # 图像预处理，
        background_imag=self.pad_to_square(background_imag)
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
        
        B,C,H, W= background_imag.shape
        shape = (4, H // 8, W // 8)








       # detect model 初始化
        self.init_object_detection()


        all_exp_root=[]
        for image_name in images_path:
            image_name = os.path.splitext(os.path.basename(image_name))[0]

            exp_root_dir=os.path.join(exp_path,f"{image_name}")
            os.makedirs(exp_root_dir,exist_ok=True)
            all_exp_root.append(exp_root_dir)

        result_gt,object_class =self.object_detection.detect(background_imag,file_path=all_exp_root,file_name='detect_object_ref.jpg',grad_status=True)

        for i,exp_root_dir in enumerate(all_exp_root):
            tensor2picture(background_imag[i],os.path.join(exp_root_dir, 'origin.jpg'))
        # 过滤筛选出最大的物体
        result_gt,object_class=self.filter_max_box_per_batch(result_gt,object_class)      
        








        input_point_list=self.yolo_boxes_to_corners(result_gt['boxes'])
        input_boxes_list=result_gt['boxes']
        # 如果没有，直接跳过
        if len(input_boxes_list[0])==0:
            return
        """
            ====================================================
            =========== 图像掩码的获取,canny的获取 ===============
            ====================================================
        """
        # 基于输入的background_imag，利用sam，得到mask，返回mask。
        # 模型初始化
        sam_predicter=init_sam(model_type=self.sam_model_type, checkpoint_path=self.sam_checkpoint_path)
        # 处理,注意mask——logic 的维度，是否是多个通道
        
        #sam_masks_logic_mutil_list 列表里面，为numpy，N*H*W
        sam_img_np, sam_masks_logic_mutil_list, sam_masks_tensor_all, sam_scores_all_list=segment_tensor(predictor=sam_predicter, 
                                                                                                         tensor_img=background_imag,
                                                                                                         input_labels_batch=object_class,
                                                                                                        input_boxes_batch=input_boxes_list
                                                                                                           ,mutil_mask=False)
        
        
        
        

        # visualize_sam(background_imag, masks_logic_mutil, scores)
        destroy_sam(sam_predicter)



        # control 处理

        # # mask 选择
        # mask_logic_np_select, mask_tensor_select=select_mask_by_criteria(
        #     masks_logic_mutil_all=sam_masks_logic_mutil_list,
        #     masks_tensor_all=sam_masks_tensor_all,
        #     scores_all=sam_scores_all_list,
        #     exp_path=all_exp_root,
        #     mask_select_statues=mask_select_statues
        # )
        mask_logic_np_select=np.concatenate(sam_masks_logic_mutil_list, axis=0)
        mask_logic_np_select=get_largest_connected_component(mask_logic_np_select)

        for i in range(len(all_exp_root)):
            tensor2picture(sam_masks_tensor_all[i],os.path.join(all_exp_root[i], 'mask.jpg'))


       # 提取物体
        object_image=self.extract_mask_content(background_imag,mask_logic_np_select)
        for i,exp_root_dir in enumerate(all_exp_root):
            tensor2picture(object_image[i],os.path.join(exp_root_dir, 'object_origin.jpg')) 


        canny_for_visual,control_image=canny_with_mask_invert(object_image,mask_logic_np_select)
        # 保存图片
        for i,exp_root_dir in enumerate(all_exp_root):
            tensor2picture(canny_for_visual[i],os.path.join(exp_root_dir, 'control.jpg'))

        control_image,rect_list=crop_mask_region(control_image,mask_logic_np_select)
        control_image,control_image_scale=resize_images_keep_aspect(control_image,(H,W))
        for i,exp_root_dir in enumerate(all_exp_root):
            tensor2picture(control_image[i],os.path.join(exp_root_dir, 'control_resized.jpg')) 


        # # background_image 的文本描述提取,object_image的文本描述提取
        # blip_model, blip_processor, blip_device=init_image_captioner(self.captioner_model_name)
        # # background_imag_caption = image_captioner_process(blip_model, blip_processor, blip_device, background_imag)
        # object_imag_caption = image_captioner_process(blip_model, blip_processor, blip_device, object_image)
        # # print(f"\n背景图像描述: {background_imag_caption}")
        # print(f"物体图像描述: {object_imag_caption}")
        # destroy_image_captioner(blip_model) 
        
        """
            ====================================================
            =========== controlnet 的初始化,采样 ===============
            ====================================================
        """

        # 初始化模型
        self.init_controlnet()
        if params["save_memory"]:
            self.model.low_vram_shift(is_diffusing=False)


        if control_image.dim()==3:
            control_image=control_image.unsqueeze(0)

        
        # 获取batch

        # 缩放control image


        # control_text=[s1+" . "+s2+" . "+s1+params["prompt"] for s1,s2 in   zip(object_class,object_imag_caption)]
        control_text=[s1+" . "+" . "+s1+params["prompt"] for s1 in   object_class] # 目前较正常
        control_text=[" . "+s1+params["prompt"]+'.'+params["prompt"]+'.'+params["prompt"] for s1 in   object_class] # 目前较正常
        # control_text=[params["prompt"] ]*B


        # c_concat 草图控制；c_crossattn 跨模态控制：正向和附加的文本提示;文本内容默认用clip编码
        cond = {
            "c_concat": [control_image],
            "c_crossattn": [
                self.model.get_learned_conditioning(
                    control_text  
                )
            ]
        }
        un_cond = {
            "c_concat": None if params["guess_mode"] else [control_image],
            "c_crossattn": [
                self.model.get_learned_conditioning(
                    [params["n_prompt"]] * B
                )
            ]
        }
 

        if params["save_memory"]:
            self.model.low_vram_shift(is_diffusing=True)



        samples, intermediates = self.ddim_sampler.sample(params["ddim_steps"], B,
                                                     shape, cond, verbose=False, eta=params["eta"],
                                                     unconditional_guidance_scale=params["scale"],
                                                     unconditional_conditioning=un_cond)
        
        controlnet_adv_sample = self.model.decode_first_stage(samples)

        self.destroy_controlnet() 

        controlnet_adv_sample=(controlnet_adv_sample+1)/2 # 采样原始范围为-1到1，这里转为0-1
        # for i,exp_root_dir in enumerate(all_exp_root):
        #     tensor2picture(controlnet_adv_sample[i],os.path.join(exp_root_dir, 'ref_sample0.jpg'))
        # 缩放回去
        controlnet_adv_sample=resized_images(controlnet_adv_sample,1./control_image_scale)
        controlnet_adv_sample=paste_images_to_background_no_scale(controlnet_adv_sample,rect_list,background_imag)
        for i,exp_root_dir in enumerate(all_exp_root):
            tensor2picture(controlnet_adv_sample[i],os.path.join(exp_root_dir, 'ref_sample.jpg'))

        """
            ====================================================
            =========== controlnet采样后的图像优化 ===============
            ====================================================
        """


        # 参考归因获取
        attributions_gt = IG_Detection(
            input_img=background_imag,
            det_model=self.object_detection,
            steps=50,
            batch_size=10,
            alpha_star=1.0,
            baseline=0.0,
            target_obj_idx=0
        )

        # 4. 可视化结果
        if attributions_gt is not None:
            visualize_attribution(background_imag, attributions_gt, save_path=all_exp_root,file_name_pre='attribution_gt')
        else:
            print("Attribution failed!")


        #  优化图像 获取，基于mask
        adv_init_tensor=batched_tensor_mask_overlay(background_imag,controlnet_adv_sample,mask_logic_np_select)
        for i,exp_root_dir in enumerate(all_exp_root):
            tensor2picture(adv_init_tensor[i],os.path.join(exp_root_dir, 'adv_init.jpg'))
        adv_init_tensor=adv_init_tensor.detach().clone()

        # 移动到GPU
        optim_device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        adv_init_tensor = move_to_gpu(adv_init_tensor,optim_device)
        result_gt=move_to_gpu(result_gt,optim_device)
        attributions_gt = move_to_gpu(attributions_gt,optim_device)
        background_imag=move_to_gpu(background_imag,optim_device)

        adv_init_tensor_gt=adv_init_tensor.clone()

        # 检测，做参考
        result_epoch,_=self.object_detection.detect(adv_init_tensor_gt,file_path=all_exp_root,file_name='adv_init_detect.jpg',grad_status=False)


        adv_init_tensor.requires_grad = True
        # 优化初始化
        optimizer = torch.optim.Adam([adv_init_tensor], lr=params["lr"])
        cross_entro_loss = YOLOv11DetectionLoss(** self.default_params,** self.detect_params).to(optim_device)
        attr_loss_l2 = nn.MSELoss()
        TV_Loss=TVLoss()    
        conext_loss_l2 = nn.MSELoss()
        perceptual_loss = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg",  # 可选：'alex', 'vgg', 'squeeze'
            normalize=True   # 自动归一化输入（匹配ImageNet规范）
        ).to(optim_device)

        pbar = tqdm(range(params["optim_epochs"]), desc="Optimizing Adversarial Sample", unit="epoch")
        for epoch in pbar:

            # 利用mask，只优化mask部分
            adv_tensor_optim=batched_tensor_mask_overlay(background_imag,adv_init_tensor,mask_logic_np_select)
           
            result_epoch,_=self.object_detection.detect(adv_tensor_optim,file_path=all_exp_root,file_name='result_generate.jpg',grad_status=True)


            attributions_epoch = IG_Detection(
                input_img=adv_tensor_optim,
                det_model=self.object_detection, 
                steps=50,
                batch_size=10,
                alpha_star=1.0,
                baseline=0.0,
                target_obj_idx=0
            )

            # # 4. 可视化结果
            # if attributions_epoch is not None:
            #     visualize_attribution(adv_tensor_optim, attributions_epoch, save_path=all_exp_root,file_name_pre='attribution_gt')
            # else:
            #     print("Attribution failed!")
            # 这里损失的使用需要注意顺序，不能改变顺序
            attr_loss=attr_loss_l2(attributions_epoch,attributions_gt)
            
            loss ,loss_dict= cross_entro_loss(result_epoch, result_gt)
            # tv_loss=TV_Loss(adv_tensor_optim)
            conext_loss=conext_loss_l2(adv_tensor_optim,adv_init_tensor_gt)
            pr_loss=perceptual_loss(normalize_to_01(adv_tensor_optim),background_imag)
            print(f"attr_loss:{attr_loss}")
            print(f"total_loss:{loss}")
            print(f"class_loss:{loss_dict['class_loss']}")
            print(f"label_gt:{result_gt['labels']},label_pred:{result_epoch['labels']}")
            print(f"score_gt:{result_gt['scores']},score_pred:{result_epoch['scores']}")
            # print(f"tv_loss:{tv_loss}")
            print(f"conext_loss:{conext_loss}")
            print(f"perceptual_loss:{pr_loss}")
            optimizer.zero_grad()
            

            

            (pr_loss* params["perceptual_loss_weight"]+ conext_loss*params['conext_loss_weight']+params["attribution_loss_weight"]*attr_loss-loss_dict['class_loss']).backward()

                          
               
            optimizer.step()
            # 手动清理变量，帮助回收内存
            del loss,loss_dict,attr_loss,pr_loss,conext_loss
            torch.cuda.empty_cache()


        
        # 保存对抗样本
        for i,exp_root_dir in enumerate(all_exp_root):

            adv_path=os.path.join(exp_root_dir, 'adv_example.jpg')
            tensor2picture(adv_init_tensor[i],adv_path)
        release_torch_object_memory("perceptual_loss",namespace=locals())

        
        
        return 



    def generate_adversarial_main_two_stage_V4(self,background_imag=None, exp_path=r'./exp',images_path=None,mask_select_statues=0,params=None):
        """
        生成对抗样本
        
        参数:
            control_image: 控制图像 (用于ControlNet)
            params: 覆盖默认参数的字典
            
        """
        """
            ====================================================
            ============ 图像预处理，初始目标的获取 ==============
            ====================================================
        """
        # 图像预处理，
        background_imag=pad_to_square(background_imag)
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
        
        B,C,H, W= background_imag.shape
        shape = (4, H // 8, W // 8)








       # detect model 初始化
        self.init_object_detection()


        all_exp_root=[]
        for image_name in images_path:
            image_name = os.path.splitext(os.path.basename(image_name))[0]

            exp_root_dir=os.path.join(exp_path,f"{image_name}")
            os.makedirs(exp_root_dir,exist_ok=True)
            all_exp_root.append(exp_root_dir)

        result_gt,object_class =self.object_detection.detect(background_imag,
                                                             file_path=all_exp_root,
                                                             file_name='detect_object_ref.jpg',
                                                             grad_status=True,
                                                             model_type=self.detect_model_type)
        for i,exp_root_dir in enumerate(all_exp_root):
            tensor2picture(background_imag[i],os.path.join(exp_root_dir, 'origin.jpg')) 
        # 过滤筛选出最大的物体
        result_gt,object_class=filter_max_box_per_batch(result_gt,object_class)      
        








        input_point_list=yolo_boxes_to_corners(result_gt['boxes'])
        input_boxes_list=result_gt['boxes']
        # 如果没有，直接跳过
        if len(input_boxes_list[0])==0:
            return
        """
            ====================================================
            =========== 图像掩码的获取,canny的获取 ===============
            ====================================================
        """
        # 基于输入的background_imag，利用sam，得到mask，返回mask。
        # 模型初始化
        sam_predicter=init_sam(model_type=self.sam_model_type, checkpoint_path=self.sam_checkpoint_path)
        # 处理,注意mask——logic 的维度，是否是多个通道
        
        #sam_masks_logic_mutil_list 列表里面，为numpy，N*H*W
        sam_img_np, sam_masks_logic_mutil_list, sam_masks_tensor_all, sam_scores_all_list=segment_tensor(predictor=sam_predicter, 
                                                                                                         tensor_img=background_imag,
                                                                                                         input_labels_batch=object_class,
                                                                                                        input_boxes_batch=input_boxes_list
                                                                                                           ,mutil_mask=False)
        
        
        
        

        # visualize_sam(background_imag, masks_logic_mutil, scores)
        destroy_sam(sam_predicter)



        # control 处理

        # # mask 选择
        # mask_logic_np_select, mask_tensor_select=select_mask_by_criteria(
        #     masks_logic_mutil_all=sam_masks_logic_mutil_list,
        #     masks_tensor_all=sam_masks_tensor_all,
        #     scores_all=sam_scores_all_list,
        #     exp_path=all_exp_root,
        #     mask_select_statues=mask_select_statues
        # )
        mask_logic_np_select=np.concatenate(sam_masks_logic_mutil_list, axis=0)
        mask_logic_np_select=get_largest_connected_component(mask_logic_np_select)

        for i in range(len(all_exp_root)):
            tensor2picture(sam_masks_tensor_all[i],os.path.join(all_exp_root[i], 'mask.jpg'))


       # 提取物体
        object_image=extract_mask_content(background_imag,mask_logic_np_select)
        object_image,rect_list=crop_mask_region(object_image,mask_logic_np_select)
        object_image,object_image_scale=resize_images_keep_aspect(object_image,(H,W))
        for i,exp_root_dir in enumerate(all_exp_root):
            tensor2picture(object_image[i],os.path.join(exp_root_dir, 'object_origin.jpg')) 


        canny_for_visual,control_image=canny_with_mask_invert(object_image,blur_status=False)
        # 保存图片
        for i,exp_root_dir in enumerate(all_exp_root):
            tensor2picture(canny_for_visual[i],os.path.join(exp_root_dir, 'control.jpg'))



        # # background_image 的文本描述提取,object_image的文本描述提取
        # blip_model, blip_processor, blip_device=init_image_captioner(self.captioner_model_name)
        # # background_imag_caption = image_captioner_process(blip_model, blip_processor, blip_device, background_imag)
        # object_imag_caption = image_captioner_process(blip_model, blip_processor, blip_device, object_image)
        # # print(f"\n背景图像描述: {background_imag_caption}")
        # print(f"物体图像描述: {object_imag_caption}")
        # destroy_image_captioner(blip_model) 
        
        """
            ====================================================
            =========== controlnet 的初始化,采样 ===============
            ====================================================
        """

        # 初始化模型
        self.init_controlnet()
        if params["save_memory"]:
            self.model.low_vram_shift(is_diffusing=False)


        if control_image.dim()==3:
            control_image=control_image.unsqueeze(0)

        
        # 获取batch

        # 缩放control image


        # control_text=[s1+" . "+s2+" . "+s1+params["prompt"] for s1,s2 in   zip(object_class,object_imag_caption)]
        # control_text=[s1+" . "+" . "+s1+params["prompt"] for s1 in   object_class] # 目前较正常
        # control_text=[" . "+s1+params["prompt"]+'.'+params["prompt"]+'.'+params["prompt"] for s1 in   object_class] # 目前较正常
        control_text=[params["prompt"] ]*B


        # c_concat 草图控制；c_crossattn 跨模态控制：正向和附加的文本提示;文本内容默认用clip编码
        cond = {
            "c_concat": [control_image],
            "c_crossattn": [
                self.model.get_learned_conditioning(
                    control_text  
                )
            ]
        }
        un_cond = {
            "c_concat": None if params["guess_mode"] else [control_image],
            "c_crossattn": [
                self.model.get_learned_conditioning(
                    [params["n_prompt"]] * B
                )
            ]
        }
 

        if params["save_memory"]:
            self.model.low_vram_shift(is_diffusing=True)



        samples, intermediates = self.ddim_sampler.sample(params["ddim_steps"], B,
                                                     shape, cond, verbose=False, eta=params["eta"],
                                                     unconditional_guidance_scale=params["scale"],
                                                     unconditional_conditioning=un_cond)
        start_time=time.time()
        controlnet_adv_sample = self.model.decode_first_stage(samples)
        end_time=time.time()
        print(f"解码耗时：{end_time-start_time:.2f} 秒")
        self.destroy_controlnet() 

        controlnet_adv_sample=(controlnet_adv_sample+1)/2 # 采样原始范围为-1到1，这里转为0-1
        for i,exp_root_dir in enumerate(all_exp_root):
            tensor2picture(controlnet_adv_sample[i],os.path.join(exp_root_dir, 'ref_sample_origin.jpg'))
        # 缩放回去
        controlnet_adv_sample=resized_images(controlnet_adv_sample,1./object_image_scale)
        controlnet_adv_sample=paste_images_to_background_no_scale(controlnet_adv_sample,rect_list,background_imag)
        for i,exp_root_dir in enumerate(all_exp_root):
            tensor2picture(controlnet_adv_sample[i],os.path.join(exp_root_dir, 'ref_sample.jpg'))

        """
            ====================================================
            =========== controlnet采样后的图像优化 ===============
            ====================================================
        """

        if params["attribution_loss_weight"]>0:
            # 参考归因获取
            attributions_gt = IG_Detection(
                input_img=background_imag,
                det_model=self.object_detection,
                steps=50,
                batch_size=10,
                alpha_star=1.0,
                baseline=0.0,
                target_obj_idx=0
            )

            # 4. 可视化结果
            if attributions_gt is not None:
                visualize_attribution(background_imag, attributions_gt, save_path=all_exp_root,file_name_pre='attribution_gt')
            else:
                print("Attribution failed!")


        #  优化图像 获取，基于mask
        adv_init_tensor=batched_tensor_mask_overlay(background_imag,controlnet_adv_sample,mask_logic_np_select)
        # 归一化
        adv_init_tensor=normalize_to_01(adv_init_tensor)
        # adv_init_tensor.data = torch.clamp(adv_init_tensor.data, 0.0, 1.0)

        for i,exp_root_dir in enumerate(all_exp_root):
            tensor2picture(adv_init_tensor[i],os.path.join(exp_root_dir, 'adv_init.jpg'))
        adv_init_tensor=adv_init_tensor.detach().clone()

        # 检测，做参考
        result_epoch,_=self.object_detection.detect(adv_init_tensor,file_path=all_exp_root,file_name='adv_init_detect.jpg',grad_status=False,model_type=self.detect_model_type)
        # 移动到GPU
        optim_device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        optim_data_type=torch.float32
        adv_init_tensor = move_to_gpu_and_cast_dtype(adv_init_tensor,optim_device,optim_data_type)
        adv_init_tensor_gt=adv_init_tensor.clone()

        # # 根据类型，修改result_gt,
        # if params["adv_loss_type"]==2:
        #     target_label=params["target_class"]
        #     other_label=params["target_class_ref"]
        #     result_gt=modify_labels_and_scores(result_gt,target_label,other_label)


        result_gt=move_to_gpu_and_cast_dtype(result_gt,optim_device,optim_data_type)
        
        if params["attribution_loss_weight"]>0:
            attributions_gt = move_to_gpu_and_cast_dtype(attributions_gt,optim_device,optim_data_type)


        background_imag=move_to_gpu_and_cast_dtype(background_imag,optim_device,optim_data_type)
        result_epoch=move_to_gpu_and_cast_dtype(result_epoch,optim_device,optim_data_type)

        # vae初始化
        self.vae_optim=VAEInferencer(model_name=self.vae_model_path,
                                      dtype=optim_data_type)
        if params["optim_object_type"]==0:
            
            # VAE的范围默认是-1到1
            adv_init_latent=self.vae_optim.encode_infer(adv_init_tensor*2-1)
            adv_init_latent=move_to_gpu_and_cast_dtype(adv_init_latent,optim_device,optim_data_type)
            adv_init_latent=adv_init_latent.clone()
            adv_init_latent = adv_init_latent.detach()
            adv_init_latent.requires_grad = True

            # 优化初始化
            optimizer = torch.optim.Adam([adv_init_latent], lr=params["lr"])

        # 后两种，在RGB上优化
        else :
            adv_init_tensor.requires_grad = True
            # 优化初始化
            optimizer = torch.optim.Adam([adv_init_tensor], lr=params["lr"])



        # 1. StepLR：每隔step_size个epoch，学习率乘以gamma
        scheduler = StepLR(
            optimizer, 
            step_size=params["lr_step"],  # 每10个epoch调整一次
            gamma=params["lr_decay"]     # 学习率衰减系数
        )
        cross_entro_loss = YOLOv11DetectionLoss(** self.default_params,** self.detect_params).to(optim_device)
        if params["attribution_loss_weight"]>0:
            attr_loss_l2 = nn.MSELoss().to(optim_device)

        if params["TV_loss_weight"]>0:
            TV_Loss=TVLoss().to(optim_device) 
        if  params["conext_loss_weight"  ]>0:

            # conext_loss_l2 = nn.MSELoss().to(optim_device)
            conext_loss_l2 =MaskedL1L2Loss().to(optim_device)
        if params["perceptual_loss_weight"]>0:
            perceptual_loss = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg",  # 可选：'alex', 'vgg', 'squeeze'
                normalize=True   # 自动归一化输入（匹配ImageNet规范）
            ).to(optim_device)
        # ========== 关键：初始化AMP梯度缩放器 ==========
        scaler = GradScaler(optim_device)  

        pbar = tqdm(range(params["optim_epochs"]), desc="Optimizing Adversarial Sample", unit="epoch")
        for epoch in pbar:

            optimizer.zero_grad()
            with autocast(device_type='cuda', dtype=torch.float16):
                    
                if params["optim_object_type"]==0:
                    # 优化latent
                    adv_tensor_generate01=self.vae_optim.decode_infer(adv_init_latent)
                    # 转化回来，将默认-1到1 的范围转化为0到1
                    adv_tensor_generate=(adv_tensor_generate01+1)/2
                elif params["optim_object_type"]==1:
                    adv_tensor_generate=adv_init_tensor

                elif params["optim_object_type"]==2:
                    #直接优化RGB
                    adv_init_tensor1=adv_init_tensor*2-1
                    adv_tensor_generate01=self.vae_optim.infer(adv_init_tensor1,sample_posterior=False)
                    # 转化回来，将默认-1到1 的范围转化为0到1
                    adv_tensor_generate=(adv_tensor_generate01+1)/2

                # 利用mask，只优化mask部分
                adv_tensor_optim=batched_tensor_mask_overlay(background_imag,
                                                                adv_tensor_generate,
                                                                mask_logic_np_select)
                adv_tensor_optim = adv_tensor_optim.clamp(0.0, 1.0)
                # adv_tensor_optim=tensor_01_to_int8_and_back(adv_tensor_optim)
                # 注意限制范围 
                result_epoch,_=self.object_detection.detect(adv_tensor_optim,
                                                            file_path=all_exp_root,
                                                            file_name='result_generate.jpg',
                                                            grad_status=True,
                                                            model_type=self.detect_model_type
                                                            )

                if  params["attribution_loss_weight"]>0:
                    attributions_epoch = IG_Detection(
                        input_img=adv_tensor_optim,
                        det_model=self.object_detection, 
                        steps=50,
                        batch_size=10,
                        alpha_star=1.0,
                        baseline=0.0,
                        target_obj_idx=0
                    )

                result_epoch_f=move_to_gpu_and_cast_dtype(result_epoch,optim_device,optim_data_type)
                if params["attribution_loss_weight"]>0:
                    attributions_epoch_f = move_to_gpu_and_cast_dtype(attributions_epoch,
                                                                      optim_device,
                                                                      optim_data_type)
                
                # # 4. 可视化结果
                # if attributions_epoch is not None:
                #     visualize_attribution(adv_tensor_optim, attributions_epoch, save_path=all_exp_root,file_name_pre='attribution_gt')
                # else:
                #     print("Attribution failed!")
                if params["attribution_loss_weight"]>0:
                    # 这里损失的使用需要注意顺序，不能改变顺序
                    attr_loss=attr_loss_l2(attributions_epoch_f,attributions_gt)
                else :
                    attr_loss=torch.tensor(0)
                
                if params["TV_loss_weight"]>0:
                    tv_loss=TV_Loss(adv_tensor_optim)
                else :
                    tv_loss=torch.tensor(0)
                if  params["conext_loss_weight"  ]>0:    
                    # conext_loss=conext_loss_l2(adv_tensor_optim,adv_init_tensor_gt)
                    conext_loss=conext_loss_l2(adv_tensor_optim,adv_init_tensor_gt,mask_logic_np_select)   
                else :
                    conext_loss=torch.tensor(0)
                if params["perceptual_loss_weight"]>0:
                    pr_loss=perceptual_loss(normalize_to_01(adv_tensor_optim),background_imag)
                else :
                    pr_loss=torch.tensor(0)
                loss ,loss_dict= cross_entro_loss(result_epoch_f, result_gt)
                # 根据是否存在损失，选择对应的权重
                total_loss=params["attribution_loss_weight"]*attr_loss+ \
                    params["TV_loss_weight"]*tv_loss+ \
                    params["perceptual_loss_weight"]*pr_loss+ \
                    params["conext_loss_weight"]*conext_loss +\
                    loss_dict['class_loss']
                   
            # 梯度裁剪：防止float16梯度爆炸
            torch.nn.utils.clip_grad_norm_([adv_init_tensor], max_norm=1.0)
            
            # ========== 反向传播：新版scaler用法不变 ==========
            scaler.scale(total_loss).backward()
            
            # ========== 参数更新：新版scaler用法不变 ==========
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            # 限制对抗样本数值范围：防止float16溢出
            adv_init_tensor.data = torch.clamp(adv_init_tensor.data, 0.0, 1.0)

            # 打印损失
            print(f"Epoch {epoch}, total_loss:{total_loss.item():.4f}")
            print(f"class_loss:{loss_dict['class_loss'].item():.4f}, conext_loss:{conext_loss.item():.4f}")
            if params["attribution_loss_weight"]>0:
                print(f"attr_loss:{attr_loss.item():.4f}")

            if params["TV_loss_weight"]>0:
                print(f"tv_loss:{tv_loss.item():.4f}")

            if params["perceptual_loss_weight"]>0:
                print(f"pr_loss:{pr_loss.item():.4f}")

            if params["conext_loss_weight"]>0:
                print(f"conext_loss:{conext_loss.item():.4f}")

            # ========== 清理内存（仅删除张量变量） ==========
            tensor_vars = [attr_loss, tv_loss, conext_loss, pr_loss, loss, total_loss]
            for var in tensor_vars:
                del var
            del loss_dict
            # 仅在迭代最后一次调用empty_cache，减少开销
            if epoch == params["optim_epochs"] - 1:
                torch.cuda.empty_cache()


        self.detect_val(input_image=adv_tensor_optim,
                        input_path=all_exp_root,
                        input_file_name='adv_example')
    
        # 保存对抗样本
        for i,exp_root_dir in enumerate(all_exp_root):

            adv_path=os.path.join(exp_root_dir, 'adv_example.jpg')
            adv_tendor_path=os.path.join(exp_root_dir, 'adv_example.pt')
            adv_tensor_idx=adv_tensor_optim[i]
            adv_tensor_idx=adv_tensor_idx.cpu().detach()
            torch.save(adv_tensor_idx,adv_tendor_path)
            tensor2picture(adv_tensor_optim[i],adv_path)
        release_torch_object_memory("perceptual_loss",namespace=locals())

        
        
        return 



    # 初始纹理生成
    def init_tex_generate(self,canny_object=None, 
                            canny_ref=None,
                            weight_object=0.5,
                            threshold=0.5,
                            ref_class=None,
                            negtive_class=None,
                            cam_target_class=None,
                            amp_status=False,

                            params=None):
        """

        
        参数:
            canny_object: object canny image
            canny_ref: ref canny image
            weight_object: object canny weight
            threshold: canny threshold
            images_path: image path,完整的图像路径列表
            params: 参数
            ref_class: 参考的目标物体信息,优先使用ref_class
            cam_target_class: 伪装的目标信息
            amp_status: 解码部分是否使用bf16
        return:
            controlnet_adv_sample: 根据Canny边缘生成初始纹理

        """
        if params is None:
            print("参数未定义")
            raise Exception 
        
            return 
        
        """
            ====================================================
            =========== controlnet 的初始化,采样 ===============
            ====================================================
        """

        B,C,H, W= canny_object.shape
        shape = (4, H // 8, W // 8)

        

        # canny 合并
        if canny_ref is not None:
            control_image=canny_object*weight_object+canny_ref*(1-weight_object)
            # 重新变为0或者1
            control_image = torch.where(control_image > threshold, torch.tensor(1.0), torch.tensor(0.0))
        else :
            control_image=canny_object

        # 初始化模型
        self.init_controlnet()
        # 条件编码部分 放在GPU
        if params["save_memory"]:
            self.model.low_vram_shift(is_diffusing=False)


        if control_image.dim()==3:
            control_image=control_image.unsqueeze(0)

        
        # 获取batch

        # 缩放control image





        """
            ====================================================
            =========== controlnet采样 ===============
            ====================================================
        """
        # control_text=[s1+" . "+s2+" . "+s1+params["prompt"] for s1,s2 in   zip(object_class,object_imag_caption)]
        if ref_class is not None:
            control_text=[' '.join([s1]*1)+" . "+" . "+s1+params["prompt"] for s1 in   ref_class] # 目前较正常
        elif cam_target_class is not None:
            control_text=[' '.join([s1]*1)+" . "+" . "+s1+params["prompt"] for s1 in   cam_target_class]
        else :
            control_text=[params["prompt"] ]*B

        if negtive_class is not None:
            negtive_control_text=[' '.join([s1]*5)+" . "+" . "+s1+params["n_prompt"] for s1 in   negtive_class]
        else :
            negtive_control_text=[params["n_prompt"]] * B
        # c_concat 草图控制；c_crossattn 跨模态控制：正向和附加的文本提示;文本内容默认用clip编码
        cond = {
            "c_concat": [control_image],
            "c_crossattn": [
                self.model.get_learned_conditioning(
                    control_text  
                )
            ]
        }
        un_cond = {
            "c_concat": None if params["guess_mode"] else [control_image],
            "c_crossattn": [
                self.model.get_learned_conditioning(
                     negtive_control_text   # [params["n_prompt"]] * B
                )
            ]
        }
 
        self.model.control_scales = (
            [params["strength"] * (0.825 ** float(12 - i)) for i in range(13)]
            if params["guess_mode"]
            else [params["strength"]] * 13
        ) 
        # 切换扩散部分放在GPU
        if params["save_memory"]:
            self.model.low_vram_shift(is_diffusing=True)
        st_time=time.time()
        with torch.no_grad():
            samples, intermediates = self.ddim_sampler.sample(params["ddim_steps"], B,
                                                    shape, cond, verbose=False, eta=params["eta"],
                                                    unconditional_guidance_scale=params["scale"],
                                                    unconditional_conditioning=un_cond)            
        
        
            controlnet_adv_sample = self.model.decode_first_stage(samples)
        end = time.time()
        print(f"sample time:{end-st_time:.2f}")        
        self.destroy_controlnet() 


        controlnet_adv_sample=(controlnet_adv_sample+1)/2 # 采样原始范围为-1到1，这里转为0-1
        # 限制范围
        controlnet_adv_sample=torch.clamp(controlnet_adv_sample,0,1)

        # 返回0-1的tensor
        return controlnet_adv_sample,control_image


    def mask_generate(self,
                      background_imag,
                      result_gt=None,
                      object_class=None,
                      mask_type=0,
                      connected_component=1,
                      mutil_mask=False,
                      ):
        '''
        参数：
            background_imag: 背景图像
            images_path: 输出图像路径，如果为None，则不保存
            mask_type: 0-单目标的mask,1-多个目标的mask
            connected_component: 0-不进行连通性处理,1-进行连通性处理,针对单个目标的情况
        '''
        if len(result_gt['boxes'])==0:
            return None
        if mask_type==0:
            # 过滤筛选出最大的物体
            result_gt,object_class=filter_max_box_per_batch(result_gt,object_class)      
    

        input_boxes_list=result_gt['boxes']
        # 如果没有，直接跳过
        if len(input_boxes_list[0])==0:
            
            return None
        """
            ====================================================
            =========== 利用sam 模型，得到图像掩码 ===============
            ====================================================
        """
        # 基于输入的background_imag，利用sam，得到mask，返回mask。
        # 模型初始化
        sam_predicter=init_sam(model_type=self.sam_model_type, 
                               checkpoint_path=self.sam_checkpoint_path)
        # 处理,注意mask——logic 的维度，是否是多个通道
        
        #sam_masks_logic_mutil_list 列表里面，为numpy，N*H*W
        sam_img_np, \
        sam_masks_logic_mutil_list, \
        sam_masks_tensor_all,\
        sam_scores_all_list=segment_tensor(predictor=sam_predicter, 
                                                    tensor_img=background_imag,
                                                    input_labels_batch=object_class,
                                                    input_boxes_batch=input_boxes_list
                                                    ,mutil_mask=mutil_mask)
        
        
        # visualize_sam(background_imag, masks_logic_mutil, scores)
        destroy_sam(sam_predicter)


        # control 处理

        # # mask 选择
        # mask_logic_np_select, mask_tensor_select=select_mask_by_criteria(
        #     masks_logic_mutil_all=sam_masks_logic_mutil_list,
        #     masks_tensor_all=sam_masks_tensor_all,
        #     scores_all=sam_scores_all_list,
        #     exp_path=all_exp_root,
        #     mask_select_statues=mask_select_statues
        # )
        mask_logic_np_select=np.concatenate(sam_masks_logic_mutil_list, axis=0)
        if connected_component==1:
            mask_logic_np_select=get_largest_connected_component(mask_logic_np_select)


        return mask_logic_np_select,result_gt,object_class


    def canny_get_mask(self,
                      background_imag,
                      mask=None,
                      canny_type=0,
                      with_mask_edge=False,
                      with_content_canny=True,
                      blur_status=True,
                      kern_size=5,
                      canny_low=50,
                      canny_high=150
                      ):
        '''
        参数：
            background_imag: 背景图像
            images_path: 输出图像路径，如果为None，则不保存
            canny_type: 0-按照比例缩放,1-不缩放
        '''
            
        # 提取物体
        B,C,H,W=background_imag.shape
        # 默认使用0填充
        object_image=extract_mask_content(background_imag,mask,mask_value=0)
        object_image,rect_coord_list=crop_mask_region(object_image,mask)
        if canny_type==0 or canny_type==2: 
            object_image,resize_scale=resize_images_keep_aspect(object_image,(H,W))
        elif canny_type==1:
            # 不等比例缩放
            rect_list=[(H,W) for i in range(B)]
            object_image,resize_scale=resize_images(object_image,rect_list)
        else:
            resize_scale=1

        canny_for_visual,control_image=canny_with_mask_invert(background_imag=object_image,
                                                              with_mask_edge=with_mask_edge,
                                                              with_content_canny=with_content_canny,
                                                              blur_k_size=kern_size,
                                                              canny_high=canny_high,
                                                              canny_low=canny_low,
                                                              blur_status=blur_status)

        return control_image,rect_coord_list,resize_scale

    def init_tex_postprocess(self,
            init_texture,
            background_imag,
            mask_np,
            rect_list=None,
            resize_scale=None,
            statues=0
        ):
        """
        初始化纹理
        参数：
            init_texture: 初始纹理
            background_imag: 背景图像
            rect_list: 矩形框列表
            all_exp_root: 所有实验根目录
            statues: 0-表示需要缩放,1
        """
        if statues==0 or statues==2: 
            
            init_texture=resized_images(init_texture,resize_scale)
            init_texture=paste_images_to_background_no_scale(init_texture,rect_list,background_imag)

        elif statues==1:
            shale_list=[ (temp[3]-temp[1]  ,temp[2]-temp[0])  for temp in    rect_list]
            init_texture,_=resize_images(init_texture,shale_list)
            init_texture=paste_images_to_background_no_scale(init_texture,rect_list,background_imag)
        init_texture=batched_tensor_mask_overlay(background_imag,
                                                    init_texture,
                                                    mask_np)

        return init_texture

    def optim_prepare(self,
                      background_imag,
                      adv_init_tensor=None,
                      result_gt=None,
                      device=None,
                      data_type=None,
                      detect_params=None,
                      attribution_params=None,
                      attack_params=None,
                      ):

        params=attack_params
        if device is None:

        # 移动到GPU
            optim_device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            optim_device =device

        if data_type is None:
            optim_data_type=torch.float32
        else :
            optim_data_type=data_type



        # adv_init_tensor=normalize_to_01(adv_init_tensor)

        adv_init_tensor=adv_init_tensor.detach().clone()

        object_detect= self.init_object_detection_return(device=optim_device,
                                                         detect_params=detect_params)





        if params["attribution_loss_weight"]>0 and params["mask_type"]==0:
            # 参考归因获取
            attri_target_label=result_gt["labels"]
            # 转化为列表，里面是int
            attri_target_label=[ label_idx.detach().cpu().item()    for label_idx in attri_target_label  ]
            # attri_target_label=list(attri_target_label)
            attributions_gt = IG_Detection(
                input_img=background_imag,
                det_model=object_detect,
                model_type=detect_params['attack_model']['model_type'],
                steps=attribution_params['steps'],
                batch_size=attribution_params['batch_size'],
                alpha_star=attribution_params['alpha_star'],
                baseline=attribution_params['baseline'],
                target_obj=attri_target_label
            )
        else :
            attributions_gt=None

        adv_init_tensor = move_to_gpu_and_cast_dtype(adv_init_tensor,optim_device,optim_data_type)
        adv_init_tensor_gt=adv_init_tensor.clone()

        # # 根据类型，修改result_gt,
        # if params["adv_loss_type"]==2:  
        #     target_label=params["target_class"]
        #     other_label=params["target_class_ref"]
        #     result_gt=modify_labels_and_scores(result_gt,target_label,other_label)


        result_gt=move_to_gpu_and_cast_dtype(result_gt,optim_device,optim_data_type)
        
        if params["attribution_loss_weight"]>0:
            attributions_gt = move_to_gpu_and_cast_dtype(attributions_gt,optim_device,optim_data_type)


        background_imag=move_to_gpu_and_cast_dtype(background_imag,optim_device,optim_data_type)


        # vae初始化
        if params["optim_object_type"]==1:
            vae_optim=None
        else :
            vae_optim=VAEInferencer(model_name=self.vae_model_path,
                                        dtype=optim_data_type)
            

        if params["optim_object_type"]==0:
            
            # VAE的范围默认是-1到1
            adv_init_latent=vae_optim.encode_infer(adv_init_tensor*2-1)
            adv_init_latent=move_to_gpu_and_cast_dtype(adv_init_latent,optim_device,optim_data_type)
            adv_init_latent=adv_init_latent.clone()
            adv_init_latent = adv_init_latent.detach()
            adv_init_latent.requires_grad = True

            # 优化初始化
            optimizer = torch.optim.Adam([adv_init_latent], lr=params["lr"])

        # 后两种，在RGB上优化
        else :
            adv_init_tensor.requires_grad = True
            # 优化初始化
            optimizer = torch.optim.Adam([adv_init_tensor], lr=params["lr"])



        # 1. StepLR：每隔step_size个epoch，学习率乘以gamma
        scheduler = StepLR(
            optimizer, 
            step_size=params["lr_step"],  # 每10个epoch调整一次
            gamma=params["lr_decay"]     # 学习率衰减系数
        )

        cross_entro_loss = YOLOv11DetectionLoss(** detect_params,
                                                ** params).to(optim_device)
        if params["attribution_loss_weight"]>0:
            attr_loss_l2 = nn.MSELoss().to(optim_device)
        else :
            attr_loss_l2 = None
        if params["TV_loss_weight"]>0:
            TV_Loss=TVLoss().to(optim_device) 
        else :
            TV_Loss=None

        if  params["conext_loss_weight"  ]>0:

            # conext_loss_l2 = nn.MSELoss().to(optim_device)
            conext_loss_l2 =MaskedL1L2Loss().to(optim_device)
        else :
            conext_loss_l2=None

        if params["perceptual_loss_weight"]>0:
            perceptual_loss = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg",  # 可选：'alex', 'vgg', 'squeeze'
                normalize=True   # 自动归一化输入（匹配ImageNet规范）
            ).to(optim_device)
        else :
            perceptual_loss=None
        if  params["optim_object_type"]==0:

            return adv_init_latent,\
                    optimizer,\
                    scheduler,\
                    cross_entro_loss,\
                    attr_loss_l2,\
                    TV_Loss,\
                    conext_loss_l2,\
                    perceptual_loss,\
                    result_gt,\
                    adv_init_tensor_gt,\
                    background_imag,\
                    attributions_gt ,\
                    vae_optim,\
                    object_detect

        else :
            return adv_init_tensor,\
                    optimizer,\
                    scheduler,\
                    cross_entro_loss,\
                    attr_loss_l2,\
                    TV_Loss,\
                    conext_loss_l2,\
                    perceptual_loss,\
                    result_gt,\
                    adv_init_tensor_gt,\
                    background_imag,\
                    attributions_gt ,\
                    vae_optim,\
                    object_detect


    def optim_loop(self,
                    adv_init_tensor,
                    optimizer,
                    scheduler,
                    cross_entro_loss,
                    attr_loss_l2,
                    TV_Loss,
                    conext_loss_l2,
                    perceptual_loss,
                    result_gt,
                    adv_init_tensor_gt,
                    background_imag,
                    attributions_gt ,
                    vae_optim,
                    object_detect,
                    mask,
                    all_exp_root=None,
                    detect_params=None,
                    attribution_params=None,
                    attack_params=None,
                    ):

        params=attack_params
        detect_model_type=detect_params["attack_model"]["model_type"]
        # adv_init_tensor=adv_init_tensor.clone()
        pbar = tqdm(range(params["optim_epochs"]), desc="Optimizing Adversarial Sample", unit="epoch")
        for epoch in pbar:

            optimizer.zero_grad()

                    
            if params["optim_object_type"]==0:
                # 优化latent
                adv_tensor_generate01=vae_optim.decode_infer(adv_init_tensor)
                # 转化回来，将默认-1到1 的范围转化为0到1
                adv_tensor_generate=(adv_tensor_generate01+1)/2
            elif params["optim_object_type"]==1:
                adv_tensor_generate=adv_init_tensor

            elif params["optim_object_type"]==2:
                #直接优化RGB
                adv_init_tensor1=adv_init_tensor*2-1
                adv_tensor_generate01=vae_optim.infer(adv_init_tensor1,sample_posterior=False)
                # 转化回来，将默认-1到1 的范围转化为0到1
                adv_tensor_generate=(adv_tensor_generate01+1)/2

            # 利用mask，只优化mask部分
            adv_tensor_optim=batched_tensor_mask_overlay(background_imag,
                                                            adv_tensor_generate,
                                                            mask)
            adv_tensor_optim = adv_tensor_optim.clamp(0.0, 1.0)
            # adv_tensor_optim=tensor_01_to_int8_and_back(adv_tensor_optim)
            # 注意限制范围 
            result_epoch,_=object_detect.detect(adv_tensor_optim,
                                                        file_path=all_exp_root,
                                                        file_name='result_generate.jpg',
                                                        grad_status=True,
                                                        model_type=detect_model_type
                                                        )





            if params["attribution_loss_weight"]>0 and params["mask_type"]==0:
                # 参考归因获取
                attri_target_label=result_gt["labels"]
                # 转化为列表，里面是int
                attri_target_label=[ label_idx.detach().cpu().item()    for label_idx in attri_target_label  ]
                # attri_target_label=list(attri_target_label)
                attributions_epoch = IG_Detection(
                    input_img=adv_tensor_optim,
                    det_model=object_detect,
                    model_type=detect_params['attack_model']['model_type'],
                    steps=attribution_params['steps'],
                    batch_size=attribution_params['batch_size'],
                    alpha_star=attribution_params['alpha_star'],
                    baseline=attribution_params['baseline'],
                    target_obj=attri_target_label
                )
            else :
                attributions_epoch=None






            

            # result_epoch_f=move_to_gpu_and_cast_dtype(result_epoch,optim_device,optim_data_type)
            result_epoch_f = result_epoch
            if params["attribution_loss_weight"]>0:
                # attributions_epoch_f = move_to_gpu_and_cast_dtype(attributions_epoch,
                #                                                     optim_device,
                #                                                     optim_data_type)
                attributions_epoch_f=attributions_epoch
            # # 4. 可视化结果
            # if attributions_epoch is not None:
            #     visualize_attribution(adv_tensor_optim, attributions_epoch, save_path=all_exp_root,file_name_pre='attribution_gt')
            # else:
            #     print("Attribution failed!")
            if params["attribution_loss_weight"]>0:
                # 这里损失的使用需要注意顺序，不能改变顺序
                attr_loss=attr_loss_l2(attributions_epoch_f,attributions_gt)
            else :
                attr_loss=torch.tensor(0)
            
            if params["TV_loss_weight"]>0:
                tv_loss=TV_Loss(adv_tensor_optim)
            else :
                tv_loss=torch.tensor(0)
            if  params["conext_loss_weight"  ]>0:    
                # conext_loss=conext_loss_l2(adv_tensor_optim,adv_init_tensor_gt)
                conext_loss=conext_loss_l2(adv_tensor_optim,adv_init_tensor_gt,mask)   
            else :
                conext_loss=torch.tensor(0)
            if params["perceptual_loss_weight"]>0:
                pr_loss=perceptual_loss(normalize_to_01(adv_tensor_optim),background_imag)
            else :
                pr_loss=torch.tensor(0)
            loss ,loss_dict= cross_entro_loss(result_epoch_f, result_gt)
            # 根据是否存在损失，选择对应的权重
            total_loss=params["attribution_loss_weight"]*attr_loss+ \
                params["TV_loss_weight"]*tv_loss+ \
                params["perceptual_loss_weight"]*pr_loss+ \
                params["conext_loss_weight"]*conext_loss +\
                loss_dict['class_loss']
                   

            
            # ========== 反向传播：新版scaler用法不变 ==========
            total_loss.backward()
            
            # ========== 参数更新：新版scaler用法不变 ==========
            optimizer.step()

            scheduler.step()
            
            adv_init_tensor.data = torch.clamp(adv_init_tensor.data, 0.0, 1.0)

            # 打印损失
            print(f"Epoch {epoch}, total_loss:{total_loss.item():.4f}")
            print(f"class_loss:{loss_dict['class_loss'].item():.4f}, conext_loss:{conext_loss.item():.4f}")
            if params["attribution_loss_weight"]>0:
                print(f"attr_loss:{attr_loss.item():.4f}")

            if params["TV_loss_weight"]>0:
                print(f"tv_loss:{tv_loss.item():.4f}")

            if params["perceptual_loss_weight"]>0:
                print(f"pr_loss:{pr_loss.item():.4f}")

            if params["conext_loss_weight"]>0:
                print(f"conext_loss:{conext_loss.item():.4f}")

            # ========== 清理内存（仅删除张量变量） ==========
            tensor_vars = [attr_loss, tv_loss, conext_loss, pr_loss, loss, total_loss]
            for var in tensor_vars:
                del var
            del loss_dict
            # 仅在迭代最后一次调用empty_cache，减少开销
            if epoch == params["optim_epochs"] - 1:
                torch.cuda.empty_cache()


        self.detect_val(input_image=adv_tensor_optim,
                        input_path=all_exp_root,
                        input_file_name='adv_example',
                        detect_params=detect_params)
                        
        if adv_tensor_optim is not None:
            # 保存对抗样本
            for i,exp_root_dir in enumerate(all_exp_root):

                adv_path=os.path.join(exp_root_dir, 'adv_example.jpg')
                adv_tendor_path=os.path.join(exp_root_dir, 'adv_example.pt')
                adv_tensor_idx=adv_tensor_optim[i]
                adv_tensor_idx=adv_tensor_idx.cpu().detach()
                torch.save(adv_tensor_idx,adv_tendor_path)
                tensor2picture(adv_tensor_optim[i],adv_path)

        release_torch_object_memory("perceptual_loss",namespace=locals())

        return 





    def optim_main(
        self,
        background_imag, 
        ref_tenture=None,
        adv_init_tensor=None,
        result_gt=None,
        mask=None,
        all_exp_root=[r"./exp/example"],
        device=None,
        data_type=None,
        detect_params=None,
        attribution_params=None,
        attack_params=None,
    ):
        # ========== 1. 新增AMP/BF16参数解析 ==========
        use_amp = attack_params.get("use_amp", False)  # 是否启用AMP
        use_bf16 = attack_params.get("use_bf16", False)  # 是否启用BF16（优先级高于FP32）
        assert not (use_amp and use_bf16), "AMP和BF16不能同时启用"

        # ========== 2. 设备和精度初始化 ==========
        if device is None:
            optim_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            optim_device = device

        # 精度适配：BF16 / FP32
        if use_bf16 and torch.cuda.is_bf16_supported():
            optim_data_type = torch.bfloat16
        elif data_type is not None:
            optim_data_type = data_type
        else:
            optim_data_type = torch.float32

        # ========== 3. AMP GradScaler初始化（仅AMP模式） ==========
        scaler = GradScaler() if use_amp else None

        # ========== 原有逻辑：数据预处理 ==========
        if ref_tenture is not None and ref_tenture.dim() == 3:
            ref_tenture = ref_tenture.unsqueeze(0)

        adv_init_tensor = adv_init_tensor.detach().clone()






        object_detect = self.init_object_detection_return(
            device=optim_device, detect_params=detect_params
        )
        mask_pre=get_true_ratio_per_channel(mask)
        # 归因损失预处理
        attributions_gt = None
        if attack_params["attribution_loss_weight"] > 0 and attack_params["mask_type"] == 0:
            attri_target_label = [label_idx.detach().cpu().item() for label_idx in result_gt["labels"]]
            attributions_gt = IG_Detection(
                input_img=background_imag,
                det_model=object_detect,
                model_type=detect_params['attack_model']['model_type'],
                steps=attribution_params['steps'],
                batch_size=attribution_params['batch_size'],
                alpha_star=attribution_params['alpha_star'],
                baseline=attribution_params['baseline'],
                target_obj=attri_target_label
            )

        # 数据移至设备并转换精度
        adv_init_tensor = move_to_gpu_and_cast_dtype(adv_init_tensor, optim_device, optim_data_type)
        adv_init_tensor_gt = adv_init_tensor.clone()

        # # 修改GT标签（对抗损失类型2）
        # if attack_params["adv_loss_type"] == 2:
        #     target_label = attack_params["target_class"]
        #     other_label = attack_params["target_class_ref"]
        #     result_gt = modify_labels_and_scores(result_gt, target_label, other_label)

        result_gt = move_to_gpu_and_cast_dtype(result_gt, optim_device, optim_data_type)
        if attack_params["attribution_loss_weight"] > 0 and attributions_gt is not None:
            attributions_gt = move_to_gpu_and_cast_dtype(attributions_gt, optim_device, optim_data_type)
        background_imag = move_to_gpu_and_cast_dtype(background_imag, optim_device, optim_data_type)
        if ref_tenture is not None:
            ref_tenture = move_to_gpu_and_cast_dtype(ref_tenture, optim_device, optim_data_type)

        # VAE初始化
        vae_optim = None
        if attack_params["optim_object_type"] != 1:
            vae_optim = VAEInferencer(model_name=self.vae_model_path, dtype=optim_data_type)

        # 优化器初始化
        if attack_params["optim_object_type"] == 0:
            # VAE latent优化
            adv_init_latent = vae_optim.encode_infer(adv_init_tensor * 2 - 1)
            adv_init_latent = move_to_gpu_and_cast_dtype(adv_init_latent, optim_device, optim_data_type)
            adv_init_latent = adv_init_latent.clone().detach()
            adv_init_latent.requires_grad = True
            optimizer = torch.optim.Adam([adv_init_latent], lr=attack_params["lr"])
        else:
            # RGB优化
            adv_init_tensor.requires_grad = True
            optimizer = torch.optim.Adam([adv_init_tensor], lr=attack_params["lr"])

        # 学习率调度器
        scheduler = StepLR(
            optimizer,
            step_size=attack_params["lr_step"],
            gamma=attack_params["lr_decay"]
        )

        # 损失函数初始化（适配精度）
        cross_entro_loss = YOLOv11DetectionLoss(**detect_params, **attack_params).to(optim_device, dtype=optim_data_type)
        attr_loss_l2 = nn.MSELoss().to(optim_device, dtype=optim_data_type) if attack_params["attribution_loss_weight"] > 0 else None
        TV_Loss = TVLoss().to(optim_device, dtype=optim_data_type) if attack_params["TV_loss_weight"] > 0 else None
        conext_loss_l2 = MaskedL1L2Loss().to(optim_device, dtype=optim_data_type) if attack_params["conext_loss_weight"] > 0 else None
        perceptual_loss = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", normalize=True
        ).to(optim_device, dtype=optim_data_type) if attack_params["perceptual_loss_weight"] > 0 else None

        detect_model_type = detect_params["attack_model"]["model_type"]
        pbar =tqdm(range(attack_params["optim_epochs"]), desc="Optimizing Adversarial Sample", unit="epoch")

        pr_scale=torch.tensor(mask_pre[0])
        pr_scale=1/pr_scale # 单位像素感知损失差距，所以需要除以比例
        # ========== 4. 核心优化循环（适配AMP/BF16） ==========
        for epoch in pbar:
            optimizer.zero_grad()

            # ========== 前向传播（AMP上下文） ==========
            with autocast(device_type="cuda",
                            enabled=use_amp,
                            dtype=torch.bfloat16 if use_bf16 else torch.float16):
                # 生成对抗样本
                if attack_params["optim_object_type"] == 0:
                    adv_tensor_generate01 = vae_optim.decode_infer(adv_init_latent)
                    adv_tensor_generate = (adv_tensor_generate01 + 1) / 2
                elif attack_params["optim_object_type"] == 1:
                    adv_tensor_generate = adv_init_tensor
                elif attack_params["optim_object_type"] == 2:
                    adv_init_tensor1 = adv_init_tensor * 2 - 1
                    adv_tensor_generate01 = vae_optim.infer(adv_init_tensor1, sample_posterior=False)
                    adv_tensor_generate = (adv_tensor_generate01 + 1) / 2

                # 应用mask并裁剪范围
                adv_tensor_optim = batched_tensor_mask_overlay(background_imag, adv_tensor_generate, mask)
                adv_tensor_optim = adv_tensor_optim.clamp(0.0, 1.0)

                # 检测模型前向
                result_epoch, _ = object_detect.detect_eval(
                    adv_tensor_optim,
                    file_path=all_exp_root,
                    file_name='result_generate.jpg',
                    grad_status=True,
                    model_type=detect_model_type
                )

                # 归因损失计算
                attributions_epoch = None
                if attack_params["attribution_loss_weight"] > 0 and attack_params["mask_type"] == 0:
                    attri_target_label = [label_idx.detach().cpu().item() for label_idx in result_gt["labels"]]
                    attributions_epoch = IG_Detection(
                        input_img=adv_tensor_optim,
                        det_model=object_detect,
                        model_type=detect_params['attack_model']['model_type'],
                        steps=attribution_params['steps'],
                        batch_size=attribution_params['batch_size'],
                        alpha_star=attribution_params['alpha_star'],
                        baseline=attribution_params['baseline'],
                        target_obj=attri_target_label
                    )
                    if attributions_epoch is not None:
                        attributions_epoch = move_to_gpu_and_cast_dtype(attributions_epoch, optim_device, optim_data_type)

                # 各损失计算
                attr_loss = torch.tensor(0.0, device=optim_device, dtype=optim_data_type)
                if attack_params["attribution_loss_weight"] > 0 and attributions_epoch is not None and attributions_gt is not None:
                    attr_loss = attr_loss_l2(attributions_epoch, attributions_gt)

                tv_loss = torch.tensor(0.0, device=optim_device, dtype=optim_data_type)
                if attack_params["TV_loss_weight"] > 0:
                    tv_loss = TV_Loss(adv_tensor_optim)

                conext_loss = torch.tensor(0.0, device=optim_device, dtype=optim_data_type)
                if attack_params["conext_loss_weight"] > 0:
                    conext_loss = conext_loss_l2(adv_tensor_optim, adv_init_tensor_gt, mask)

                pr_loss = torch.tensor(0.0, device=optim_device, dtype=optim_data_type)
                if attack_params["perceptual_loss_weight"] > 0 and ref_tenture is not None:
                    pr_loss = perceptual_loss(normalize_to_01(adv_tensor_optim), ref_tenture)

                # 检测损失
                loss, loss_dict = cross_entro_loss(result_epoch, result_gt)

                # 总损失
                total_loss = (
                    attack_params["attribution_loss_weight"] * attr_loss
                    + attack_params["TV_loss_weight"] * tv_loss
                    + attack_params["perceptual_loss_weight"] * pr_loss*pr_scale
                    + attack_params["conext_loss_weight"] * conext_loss
                    + attack_params["class_loss_weight"]*loss_dict['class_loss']
                )

            # ========== 反向传播 + 优化（AMP适配） ==========
            if use_amp:
                # AMP模式：缩放梯度避免下溢
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 普通模式/BF16模式
                total_loss.backward()
                optimizer.step()

            # 学习率调度
            scheduler.step()

            # 裁剪参数范围
            if attack_params["optim_object_type"] != 0:
                adv_init_tensor.data = torch.clamp(adv_init_tensor.data, 0.0, 1.0)

            # # ========== 日志打印 ==========
            # pbar.set_postfix({
            #     "total_loss": f"{total_loss.item():.4f}",
            #     "class_loss": f"{loss_dict['class_loss'].item():.4f}",
            #     "conext_loss": f"{conext_loss.item():.4f}",
            #     "lr": f"{scheduler.get_last_lr()[0]:.6f}"
            # })
            # pbar.set_postfix({"class_loss": f"{loss_dict['class_loss'].item():.4f}"}, update=True)
            # if attack_params["attribution_loss_weight"] > 0:
            #     pbar.set_postfix({"attr_loss": f"{attr_loss.item():.4f}"}, update=True)
            # if attack_params["TV_loss_weight"] > 0:
            #     pbar.set_postfix({"tv_loss": f"{tv_loss.item():.4f}"}, update=True)
            # if attack_params["perceptual_loss_weight"] > 0:
            #     pbar.set_postfix({"pr_loss": f"{pr_loss.item():.4f}"}, update=True)
            # if attack_params['conext_loss_weight'] > 0:
            #     pbar.set_postfix({"conext_loss": f"{conext_loss.item():.4f}"}, update=True)



            # ========== 日志打印 ==========
            # 1. 初始化基础后缀字典（必显项）
            postfix_dict = {
                "total_loss": f"{total_loss.item():.4f}",
                "class_loss": f"{loss_dict['class_loss'].item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.6f}"
            }

            # 2. 按需添加可选损失项（有则加，无则不加）
            if attack_params["attribution_loss_weight"] > 0:
                postfix_dict["attr_loss"] = f"{attr_loss.item():.4f}"
            if attack_params["TV_loss_weight"] > 0:
                postfix_dict["tv_loss"] = f"{tv_loss.item():.4f}"
            if attack_params["perceptual_loss_weight"] > 0:
                postfix_dict["pr_loss"] = f"{pr_loss.item():.4f}"
            if attack_params['conext_loss_weight'] > 0:
                postfix_dict["conext_loss"] = f"{conext_loss.item():.4f}"

            # 3. 一次性更新进度条（核心：避免覆盖）
            pbar.set_postfix(postfix_dict)



            # ========== 内存清理 ==========
            del attr_loss, tv_loss, conext_loss, pr_loss, loss, total_loss
            del loss_dict
            if epoch == attack_params["optim_epochs"] - 1:
                torch.cuda.empty_cache()

        # ========== 5. 验证和保存结果 ==========
        adv_result_dict=self.detect_val(
            input_image=adv_tensor_optim,
            input_path=all_exp_root,
            input_file_name='adv_example',
            detect_params=detect_params
        )

        adv_count_dict=count_matched_results(adv_result_dict,result_gt)


        temp_tensor=torch.ones_like(background_imag)
        ref_texture_mask_adv=batched_tensor_mask_overlay(temp_tensor,
                                        adv_tensor_optim,
                                        mask)


            
        if adv_tensor_optim is not None:
            for i, exp_root_dir in enumerate(all_exp_root):
                adv_path = os.path.join(exp_root_dir, 'adv_example.jpg')
                adv_tendor_path = os.path.join(exp_root_dir, 'adv_example.pt')
                adv_tendor_mask_path=os.path.join(exp_root_dir, 'adv_texture.jpg')
                adv_tensor_idx = adv_tensor_optim[i].cpu().detach()
                torch.save(adv_tensor_idx, adv_tendor_path)
                tensor2picture(adv_tensor_optim[i], adv_path)

                tensor2picture(ref_texture_mask_adv[i], adv_tendor_mask_path)

        release_torch_object_memory("perceptual_loss", namespace=locals())
        return adv_tensor_optim,adv_count_dict



    def generate_adversarial_mainV5(self,
                                    background_imag=None,
                                    ref_texture=None, 
                                    ref_canny=None,
                                    mask_adv=None,
                                    exp_path=[r'./exp/exp_example'],
                                    detect_params=None,
                                    attribution_params=None,
                                    attack_params=None):
        """
        生成对抗样本
        
        参数:
            control_image: 控制图像 (用于ControlNet)
            params: 覆盖默认参数的字典
            
        """
        """
            ====================================================
            ============ 图像预处理，初始目标的获取 ==============
            ====================================================
        """
        # 图像预处理，
        background_imag=pad_to_square(background_imag)

        params=attack_params
        # 设置随机种子以确保结果可复现
        torch.manual_seed(params["seed"])
        np.random.seed(params["seed"])
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        B,C,H, W= background_imag.shape
        shape = (4, H // 8, W // 8)


       # detect model 初始化
        detect_model=self.init_object_detection_return(device,
                                                       detect_params)


        detect_model_type=detect_params['attack_model']['model_type']

        result_gt,object_class =detect_model.detect_eval(background_imag,
                                                             file_path=exp_path,
                                                             file_name='detect_object_ref.jpg',
                                                             grad_status=True,
                                                             model_type=detect_model_type)
        self.destroy_object_detection(detect_model)
        for i,exp_root_dir in enumerate(exp_path):
            tensor2picture(background_imag[i],os.path.join(exp_root_dir, 'origin.jpg')) 

        # 多框的分割存在问题，目前只能支持单框，多框需要循环;图片保存未添加
        mask_path=[    os.path.join(exp_path[i], 'mask.jpg') for i in range(len(exp_path))]
        # result_gt 是否覆盖
        if len(result_gt['boxes'][0])<1:
            return 
        # 返回的检测框可能是挑选过的
        mask_logic_np_select,result_gt,object_class=self.mask_generate(
                         background_imag=background_imag,
                        result_gt=result_gt,
                        object_class=object_class,
                        mask_type=params["mask_type"],
                        connected_component=params["connected_component"],
                        )


        # 调试
        ref_result_dict=self.detect_val(
            input_image=background_imag,
            input_path=exp_path,
            input_file_name='origin_example',
            detect_params=detect_params
        )
        ref_count_dict=count_matched_results(ref_result_dict,result_gt)

        for model_key in ref_count_dict:
            if ref_count_dict[model_key] <1:
                # 不完全匹配
                return None


        if mask_logic_np_select is None:
            return None
        
         
       # 提取物体
        control_image,rect_boxes,resize_scale=self.canny_get_mask(
                    background_imag=background_imag,
                    mask=mask_logic_np_select,
                    canny_type=params["canny_type"],
                    kern_size=params["kern_size"],
                    canny_high=params["canny_high"],
                    canny_low=params["canny_low"],
                    blur_status=params["canny_blur"],
                    with_mask_edge=params["canny_with_mask_edge"],
                    with_content_canny=params["canny_with_content_canny"],
                    )



        
        """
            ====================================================
            =========== controlnet 的初始化,采样 ===============
            ====================================================
        """
        # 参考canny的处理
        cam_target=[params["cam_target_class"]]*B
        if ref_canny is not None:
            if params["canny_type"]==0:
                ref_canny=fit_canny_to_xyxy_boxes(ref_canny,rect_boxes,resize_scale)
            if params["canny_type"]==2:
                ref_canny=center_scale_image_tensor(ref_canny,params["canny_scale"])
            else :
                ref_canny=ref_canny

            control_path=[os.path.join(exp_path[i], 'control_ref.jpg') for i in range(len(exp_path))]
            for i,exp_root_dir in enumerate(exp_path):
                tensor2picture(ref_canny[i],control_path[i])
        else :
            ref_canny=None


        controlnet_adv_sample,control_final=self.init_tex_generate(canny_object=control_image, 
                                canny_ref=ref_canny,
                                weight_object=0.5,
                                threshold=0.1,
                                cam_target_class=cam_target,
                                ref_class=None, # object_class
                                negtive_class=object_class,
                                params=params)
        control_path=[os.path.join(exp_path[i], 'control.jpg') for i in range(len(exp_path))]
        for i,exp_root_dir in enumerate(exp_path):
            tensor2picture(control_final[i],control_path[i])
        control_path=[os.path.join(exp_path[i], 'sample.jpg') for i in range(len(exp_path))]
        for i,exp_root_dir in enumerate(exp_path):
            tensor2picture(controlnet_adv_sample[i],control_path[i])

        # scale back
        resize_scale_back=calculate_resize_scale_back(resize_scale)
        init_texture=self.init_tex_postprocess(init_texture=controlnet_adv_sample,
                                                background_imag=background_imag,
                                                mask_np=mask_logic_np_select,
                                                rect_list=rect_boxes,
                                                resize_scale=resize_scale_back,
                                                statues=params["canny_type"])


        control_path=[os.path.join(exp_path[i], 'init.jpg') for i in range(len(exp_path))]
        for i,exp_root_dir in enumerate(exp_path):
            tensor2picture(init_texture[i],control_path[i])
        
        # # 抠出纹理
        # temp=torch.zeros_like(background_imag)
        # object_texture=batched_tensor_mask_overlay(temp,
        #                                         init_texture,
        #                                         mask_logic_np_select)
        # #
        # object_texture_path=[os.path.join(exp_path[i], 'texture.jpg') for i in range(len(exp_path))]
        # for i,exp_root_dir in enumerate(exp_path):
        #     tensor2picture(object_texture[i],object_texture_path[i])
        """
            ====================================================
            =========== controlnet采样后的图像优化 ===============
            ====================================================
        """


        # 处理
        if ref_texture is not None:
            if ref_texture.dim()==3:
                ref_texture=ref_texture.unsqueeze(0)
            
            ref_texture=batched_tensor_mask_overlay(background_imag,
                                                        ref_texture,
                                                        mask_logic_np_select)


        # 
        if attack_params["ref_status"]==0:     
            ref_texture_optim=init_texture
            init_texture_optim=background_imag
        elif attack_params["ref_status"]==1:
            ref_texture_optim=ref_texture
            init_texture_optim=init_texture
                 

        #
        if mask_adv is  None:
            mask_adv=mask_logic_np_select

            #
        temp_tensor=torch.ones_like(background_imag)
        ref_texture_mask_adv=batched_tensor_mask_overlay(temp_tensor,
                                        init_texture,
                                        mask_adv)

        path_temp=[os.path.join(exp_path[i], 'mask_adv.jpg') for i in range(len(exp_path))]
        for i,exp_root_dir in enumerate(exp_path):
            tensor2picture(ref_texture_mask_adv[i],path_temp[i])

        _,accur=self.optim_main(background_imag=background_imag,
                        ref_tenture=ref_texture_optim,
                        adv_init_tensor=init_texture_optim,
                        result_gt=result_gt,
                        mask=mask_adv,
                        all_exp_root=exp_path,
                        detect_params=detect_params,
                        attribution_params=attribution_params,
                        attack_params=attack_params,
                        )
        return accur

        # 优化初始化，包含优化器，调度器，损失函数，优化目标等
        # adv_init_tensor,\
        # optimizer,\
        # scheduler,\
        # cross_entro_loss,\
        # attr_loss_l2,\
        # TV_Loss,\
        # conext_loss_l2,\
        # perceptual_loss,\
        # result_gt,\
        # adv_init_tensor_gt,\
        # background_imag,\
        # attributions_gt ,\
        # vae_optim,\
        # object_detect=self.optim_prepare(
        #                 background_imag=background_imag,
        #                 adv_init_tensor=init_texture,
        #                 result_gt=result_gt,
        #                 detect_params=detect_params,
        #                 attribution_params=attribution_params,
        #                 attack_params=params,
        #                 )
       


        # self.optim_loop(
        #             adv_init_tensor=adv_init_tensor,
        #             optimizer=optimizer,
        #             scheduler=scheduler,
        #             cross_entro_loss=cross_entro_loss,
        #             attr_loss_l2=attr_loss_l2,
        #             TV_Loss=TV_Loss,
        #             conext_loss_l2=conext_loss_l2,
        #             perceptual_loss=perceptual_loss,
        #             result_gt=result_gt,
        #             adv_init_tensor_gt=adv_init_tensor_gt,
        #             background_imag=background_imag,
        #             attributions_gt=attributions_gt ,
        #             vae_optim=vae_optim,
        #             object_detect=object_detect,
        #             mask=mask_logic_np_select,
        #             all_exp_root=exp_path,
        #             detect_params=detect_params,
        #             attribution_params=attribution_params,
        #             attack_params=params,
        # )


        
        return 




    def generate_adversarial_advlogo_mainV5(self,
                                    background_imag=None,
                                    adv_patch_tensor=None, 
                                    exp_path=[r'./exp/exp_example'],
                                    detect_params=None,
                                    attribution_params=None,
                                    attack_params=None,
                                    config_yaml_path='./config.yaml'):
        """
        生成对抗样本
        
        参数:
            control_image: 控制图像 (用于ControlNet)
            params: 覆盖默认参数的字典
            
        """
        """
            ====================================================
            ============ 图像预处理，初始目标的获取 ==============
            ====================================================
        """
        # 图像预处理，
        background_imag=pad_to_square(background_imag)

        params=attack_params
        # 设置随机种子以确保结果可复现
        torch.manual_seed(params["seed"])
        np.random.seed(params["seed"])
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        B,C,H, W= background_imag.shape
        shape = (4, H // 8, W // 8)


       # detect model 初始化
        detect_model=self.init_object_detection_return(device,
                                                       detect_params)


        detect_model_type=detect_params['attack_model']['model_type']

        result_gt,object_class =detect_model.detect_eval(background_imag,
                                                             file_path=exp_path,
                                                             file_name='detect_object_ref.jpg',
                                                             grad_status=True,
                                                             model_type=detect_model_type)
        self.destroy_object_detection(detect_model)
        for i,exp_root_dir in enumerate(exp_path):
            tensor2picture(background_imag[i],os.path.join(exp_root_dir, 'origin.jpg')) 


        if len(result_gt['boxes'][0])<1:
            return 
        # 返回的检测框可能是挑选过的
        mask_logic_np_select,result_gt,object_class=self.mask_generate(
                         background_imag=background_imag,
                        result_gt=result_gt,
                        object_class=object_class,
                        mask_type=params["mask_type"],
                        connected_component=params["connected_component"],
                        )


        # 
        ref_result_dict=self.detect_val(
            input_image=background_imag,
            input_path=exp_path,
            input_file_name='origin_example',
            detect_params=detect_params
        )
        ref_count_dict=count_matched_results(ref_result_dict,result_gt)

        for model_key in ref_count_dict:
            if ref_count_dict[model_key] <1:
                # 不完全匹配
                return None


        if mask_logic_np_select is None:
            return None

        with open(config_yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f)

        class Config:
            TRANSFORM = yaml_config.get('TRANSFORM', {
                'rotate': True,
                'shift': True,
                'median_pool': True,
                'jitter': True,
                'cutout': True
            })
            MAX_PATCH_RATIO = yaml_config.get('MAX_PATCH_RATIO', 0.4)

        patch_applier = PatchRandomApplier(device, Config())

        adv_img_batch = patch_applier.apply_patch(background_imag, adv_patch_tensor, result_gt)
        

        adv_result_dict=self.detect_val(
            input_image=adv_img_batch,
            input_path=exp_path,
            input_file_name='adv_example',
            detect_params=detect_params
        )

        adv_count_dict=count_matched_results(adv_result_dict,result_gt)

        for i,exp_root_dir in enumerate(exp_path):
            tensor2picture(adv_img_batch[i],os.path.join(exp_root_dir, 'adv_example.jpg')) 

        return adv_count_dict


    def generate_adversarial_fgsm_mainV5(
            self,
            background_imag=None,                  # [B,C,H,W]
            exp_path=[r'./exp/exp_example'],
            detect_params=None,
            config_yaml_parmars=None):

        """
        FGSM / i-FGSM adversarial example generation (ROI-aware)
        """

        # ====================================================
        # 1. preprocessing
        # ====================================================
        # 图像预处理，
        background_imag=pad_to_square(background_imag)

        params=config_yaml_parmars
        # 设置随机种子以确保结果可复现
        torch.manual_seed(params["seed"])
        np.random.seed(params["seed"])
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        B,C,H, W= background_imag.shape
        shape = (4, H // 8, W // 8)


       # detect model 初始化
        detect_model=self.init_object_detection_return(device,
                                                       detect_params)


        detect_model_type=detect_params['attack_model']['model_type']

        result_gt,object_class =detect_model.detect_eval(background_imag,
                                                             file_path=exp_path,
                                                             file_name='detect_object_ref.jpg',
                                                             grad_status=True,
                                                             model_type=detect_model_type)
        
        for i,exp_root_dir in enumerate(exp_path):
            tensor2picture(background_imag[i],os.path.join(exp_root_dir, 'origin.jpg')) 


        if len(result_gt['boxes'][0])<1:
            self.destroy_object_detection(detect_model)
            return 
        # 返回的检测框可能是挑选过的
        mask_logic_np_select,result_gt,object_class=self.mask_generate(
                         background_imag=background_imag,
                        result_gt=result_gt,
                        object_class=object_class,
                        mask_type=params["mask_type"],
                        connected_component=params["connected_component"],
                        )


        # 
        ref_result_dict=self.detect_val(
            input_image=background_imag,
            input_path=exp_path,
            input_file_name='origin_example',
            detect_params=detect_params
        )
        ref_count_dict=count_matched_results(ref_result_dict,result_gt)

        for model_key in ref_count_dict:
            if ref_count_dict[model_key] <1:
                # 不完全匹配
                return None


        if mask_logic_np_select is None:
            return None


        # ROI boxes list (for FGSM / i-FGSM)
        boxes_list = result_gt["boxes"]

        # ====================================================
        # 6. choose FGSM / i-FGSM
        # ====================================================
        attack_method = params["method"].lower()

        if attack_method == "fgsm":
            adv_img_batch = fgsm_od(
                x=background_imag,
                od_model=detect_model,
                model_type=detect_model_type,
                boxes_list=boxes_list,
                eps=params["eps"],
                targeted=params["targeted"],
                target_label=params["target_label"],
                loss_type=params["loss_type"]
            )

        elif attack_method == "ifgsm":
            adv_img_batch = ifgsm_od(
                x=background_imag,
                od_model=detect_model,
                model_type=detect_model_type,
                boxes_list=boxes_list,
                eps=params["eps"],
                alpha=params["alpha"],
                iteration=params["iteration"],
                targeted=params["targeted"],
                target_label=params["target_label"],
                loss_type=params["loss_type"]
            )

        else:
            raise ValueError(f"Unsupported attack method: {attack_method}")

        # ====================================================
        # 7. detect adversarial image
        # ====================================================
        adv_result_dict = self.detect_val(
            input_image=adv_img_batch,
            input_path=exp_path,
            input_file_name='adv_example',
            detect_params=detect_params
        )

        adv_count_dict = count_matched_results(
            adv_result_dict,
            result_gt
        )

        # ====================================================
        # 8. save adversarial images
        # ====================================================
        for i, exp_root_dir in enumerate(exp_path):
            tensor2picture(
                adv_img_batch[i],
                os.path.join(exp_root_dir, 'adv_example.jpg')
            )

        # ====================================================
        # 9. clean & return
        # ====================================================
        self.destroy_object_detection(detect_model)

        return adv_count_dict




    def to_imgTensor_from_numpy_int8(self, image):
        """
        作用：将numpy数组转换为PyTorch张量。
        参数：
        image: 输入的numpy数组，形状为[C, H, W]。
        返回：
        tensor: 转换后的PyTorch张量，形状为[C, H, W]。
        """

        # 转换到-1到1
        image = image.astype(np.float32) / 127.5 - 1.0

        tensor = torch.from_numpy(image).float()

        
        return tensor.unsqueeze(0)
    

    def generate_edge_control_from_image(self, image,file_path=None):
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
        # 使用opencv 进行边缘检测
        canny_map = cv2.Canny(image, 100, 200)
        # 黑白交换（位运算反转）
        detected_single_map = cv2.bitwise_not(canny_map)
        detected_map = np.stack([detected_single_map]*3, axis=-1)
        if file_path is not None:
            cv2.imwrite(file_path,detected_map)
        # detected_map = np.zeros_like(image, dtype=np.uint8)
        # detected_map[np.min(image, axis=2) < 127] = 255

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(self.default_params["num_samples"])], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        return detected_map,control
    
    # 将图片转化为latent
    def imgTensor_to_latent(self, img,scale=0.18215):
        '''
        img:[-1-1],type:tensor
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
        z=z*scale
        z=z.to(self.device)
        return z   
    # def latent_to_imgTensor01(self,latent):
    #     img = self.model.first_stage_model.decode(latent)
    #     return  (einops.rearrange(img, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

    def latent_to_imgTensor01(self,latent,scale=0.18215):
        latent = latent / scale
        img = self.model.first_stage_model.decode(latent)
        # sd 对应的区间为-1到1，需要转换到0到1
        img = ((img + 1)*0.5 ).to(dtype=torch.float32)
    
        # # 确保与YOLO模型在同一设备
        # img = img.to(self.yolo_model.device)  # 假设self.yolo_model是加载的YOLO模型
        
        return img
        # return  (einops.rearrange(img, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)


    def tensor_01_to_numpy_255(self,tensor):
        """
        将模型输出的0-1范围图像张量转换为0-255范围numpy数组，并调整通道顺序
        
        Args:
            tensor: 模型输出的图像张量，格式为 [B, C, H, W] 或 [C, H, W]（单张图像）
                    数值范围必须是 [0, 1]，通道数通常为1（灰度）或3（RGB/BGR）
            is_rgb: 若为True，默认输入通道为RGB（无需额外转换）；
                    若为False，会将RGB转为BGR（适配opencv的默认通道顺序）
        
        Returns:
            numpy_array: 转换后的numpy数组，格式为 [B, H, W, C] 或 [H, W, C]（单张图像）
                        数值范围 [0, 255]，数据类型 uint8
        """
        # -------------------------- 1. 处理单张图像（无批量维度） --------------------------
        if tensor.dim() == 3:  # 输入为 [C, H, W]（单张图像），添加批量维度变为 [1, C, H, W]
            tensor = tensor.unsqueeze(0)
        

        # -------------------------- 2. 设备迁移 + 张量转numpy --------------------------
        # 推理阶段用 .detach() 切断梯度，训练阶段若需保留梯度可移除（但通常图像转换用于推理）
        if tensor.is_cuda:
            tensor = tensor.cpu()  # 移到CPU（numpy不支持CUDA数据）
        np_array = tensor.detach().numpy()  # 张量 → numpy数组，格式 [B, C, H, W]

        # -------------------------- 3. 0-1 → 0-255 缩放 + 数据类型转换 --------------------------
        # 乘以255后用np.clip确保数值在0-255（避免浮点误差导致的超界，如1.0001→255.025）
        np_array = np.clip(np_array * 255.0, a_min=0, a_max=255)

        np_array = np_array.astype(np.uint8)

        np_array = np.transpose(np_array, axes=(0, 2, 3, 1))  # 调整维度顺序

 

        if np_array.shape[0] == 1:
            np_array = np_array.squeeze(0)  # 从 [1, H, W, C] 变为 [H, W, C]

        return np_array


















# #
# if __name__=='__main__':
#     # 添加本地包路径,即上一级的路径
#     os.path.join(os.path.dirname(__file__), "..")





#     from annotator.util import resize_image, HWC3
#     from cldm.model import create_model, load_state_dict
#     from cldm.ddim_hacked import DDIMSampler
#     import config


#     #判断gpu 是否存在，并给出版本
#     if torch.cuda.is_available():
#         print('cuda version:', torch.version.cuda)
        
#     else:
#         print('no cuda')
#     # 模型参数里面包含 ControlNet 和ControlledUnetModel 的参数

#     #RGB 参数
#     attack_config_path=r"models/attack_config.yaml"


#     adv_config=load_yaml_config(attack_config_path)


#     attack = ADV_ATTACK(config_path=adv_config["model_paths"]["control_yaml_path"],
#                         model_path=adv_config["model_paths"]["controlnet"],
#                         device=torch.device("cuda"),
#                         detect_model_type=adv_config["model_types"]["detect_model"],
#                         model_path_object_detection=adv_config["model_paths"]["detect_model"],
#                         sam_model_type=adv_config["model_types"]["sam_model"],
#                         sam_checkpoint_path=adv_config["model_paths"]["sam_model"],
#                         captioner_model_name=adv_config["model_paths"]["blip_model"],
#                         inpaint_model_path=adv_config["model_paths"]["inpaint_model"],
#                         vae_model_path=adv_config["model_paths"]["vae_model"],
#                         kwargs=adv_config["attak_params"],
#                         detect_params=adv_config["detect_params"],
#                         )


#     # 读取图像，转化为tensor

#     # img_path=r"../data\select_coco\000000005477.jpg"
#     # img_path=r"../data\select_coco\000000000724.jpg"
#     img_path=r'data/select_coco/000000079408.jpg'   
#     # img_path=r'data/select_coco/000000001296.jpg'
#     img = cv2.imread(img_path)
#     img=cv2.resize(img, (512, 512))
#     # 转化为tensor
#     # 转换通道

    
#     img = cv2_to_tensor(img)


#     ref_path=r"data/control1.jpg"
#     ref_tenture=cv2.imread(ref_path,cv2.IMREAD_GRAYSCALE)
#     ref_tenture=cv2.resize(ref_tenture, (512, 512))
#     ref_canny=cv2_to_tensor(ref_tenture)
#     if ref_canny.dim()==3:  # 添加维度
#         ref_canny = ref_canny.unsqueeze(0)

#     if img.dim()==3:  # 添加维度
#         img = img.unsqueeze(0)

#     exp_root_test=adv_config["experiment_params"]["experiment_path"]
#     os.makedirs(exp_root_test,exist_ok=True)
#     st_time=time.time()
#     attack.generate_adversarial_mainV5(background_imag=img,
#                                        ref_texture=None,
#                                        ref_canny=ref_canny,
#                                        exp_path=[exp_root_test],
#                                         detect_params=adv_config["detect_params"],
#                                         attribution_params=adv_config["attribution_params"],
#                                         attack_params=adv_config["attak_params"]
#                                        )
#     end_time=time.time()
#     print(f"time: {end_time - st_time:.2f}s")


#     # attack.generate_adversarial_main(img)



#     # attack.init_vae()
#     # # 测试
#     # if img.dim()==3:  # 添加维度
#     #     img = img.unsqueeze(0)
#     # # VAE测试
#     # img=move_to_gpu(img)
#     # attack.vae.to(device=torch.device("cuda"))
#     # img.requires_grad=True
#     # attack.vae.eval()
#     # start_time=time.time()
#     # posterior_vae=attack.vae.encode(img*2-1)
#     # latent=posterior_vae.mode()
#     # img_g=attack.vae.decode(latent)
#     # end_time=time.time()
#     # print(f"VAE time: {end_time - start_time:.2f}s")
#     # # 转化为0-1
#     # img_g=(img_g+1)/2
#     # tensor2picture(img_g[0],'test.jpg')
#     # temp,_=attack.vae(img*2-1,sample_posterior=False)
#     # temp=(temp+1)/2
#     # tensor2picture(img_g[0],'test.jpg')
#     # loss = torch.nn.functional.mse_loss(img_g, img)  # 对比重建图和原图
#     # loss.backward()  # 反向传播，计算梯度

#     # # 4. 验证梯度是否回传
#     # print("===== 梯度验证结果 =====")
#     # # 检查 img 的梯度是否存在且非全零
#     # if img.grad is not None:
#     #     grad_norm = torch.norm(img.grad).item()  # 计算梯度范数（标量）
#     #     print(f"img.grad 存在，梯度范数：{grad_norm:.6f}")
#     #     if grad_norm > 1e-8:  # 梯度非全零（浮点误差容忍）
#     #         print("✅ 梯度成功回传到 img！")
#     #     else:
#     #         print("❌ img.grad 为全零，梯度未有效回传！")
#     # else:
#     #     print("❌ img.grad 不存在，梯度被截断！")
#     # attack.generate_adversarial_example_optim_control_v2(img)
#     # attack.generate_adversarial_example(img,control_img)
#     # attack.generate_adversarial_example_optim_control_v2(img,control_img)
#     # attack.generate_adversarial_example_optim_control(img,control_img)






#
if __name__=='__main__':
    # 添加本地包路径,即上一级的路径
    os.path.join(os.path.dirname(__file__), "..")





    from annotator.util import resize_image, HWC3
    from cldm.model import create_model, load_state_dict
    from cldm.ddim_hacked import DDIMSampler
    import config


    #判断gpu 是否存在，并给出版本
    if torch.cuda.is_available():
        print('cuda version:', torch.version.cuda)
        
    else:
        print('no cuda')
    # 模型参数里面包含 ControlNet 和ControlledUnetModel 的参数

    #RGB 参数
    # attack_config_path=r"models/attack_config.yaml"
    attack_config_path=r"models/fgsm_config.yaml"

    adv_config=load_yaml_config(attack_config_path)


    attack = ADV_ATTACK(config_path=adv_config["model_paths"]["control_yaml_path"],
                        model_path=adv_config["model_paths"]["controlnet"],
                        device=torch.device("cuda"),
                        detect_model_type=adv_config["model_types"]["detect_model"],
                        model_path_object_detection=adv_config["model_paths"]["detect_model"],
                        sam_model_type=adv_config["model_types"]["sam_model"],
                        sam_checkpoint_path=adv_config["model_paths"]["sam_model"],
                        captioner_model_name=adv_config["model_paths"]["blip_model"],
                        inpaint_model_path=adv_config["model_paths"]["inpaint_model"],
                        vae_model_path=adv_config["model_paths"]["vae_model"],
                        kwargs=adv_config["attak_params"],
                        detect_params=adv_config["detect_params"],
                        )


    # 读取图像，转化为tensor

    # img_path=r"../data\select_coco\000000005477.jpg"
    # img_path=r"../data\select_coco\000000000724.jpg"
    # img_path=r'exp\for_paper\exp_example\origin.jpg'   
    img_path=r'data/select_coco/000000001296.jpg'
    img = cv2.imread(img_path)
    img=cv2.resize(img, (512, 512))
    # 转化为tensor
    # 转换通道

    
    img = cv2_to_tensor(img)


    ref_path=r"data\adv_patch\v2-dog.png"
    adv_patch=cv2.imread(ref_path)
    adv_patch=cv2.resize(adv_patch, (256, 256))
    adv_patch_tensor=cv2_to_tensor(adv_patch)
    if adv_patch_tensor.dim()==3:  # 添加维度
        adv_patch_tensor = adv_patch_tensor.unsqueeze(0)

    if img.dim()==3:  # 添加维度
        img = img.unsqueeze(0)

    exp_root_test=adv_config["experiment_params"]["experiment_path"]
    os.makedirs(exp_root_test,exist_ok=True)
    st_time=time.time()
    # attack.generate_adversarial_mainV5(background_imag=img,
    #                                    ref_texture=None,
    #                                    ref_canny=ref_canny,
    #                                    exp_path=[exp_root_test],
    #                                     detect_params=adv_config["detect_params"],
    #                                     attribution_params=adv_config["attribution_params"],
    #                                     attack_params=adv_config["attak_params"]
    #                                    )


    attack.generate_adversarial_fgsm_mainV5(
        background_imag=img,
        exp_path=[exp_root_test],
        detect_params=adv_config["detect_params"],
        config_yaml_parmars=adv_config["attak_params"],
    )
    end_time=time.time()
    print(f"time: {end_time - st_time:.2f}s")


    # attack.generate_adversarial_main(img)



