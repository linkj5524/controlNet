
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

class ADV_ATTACK:
    def __init__(self, config_path:str='./models/cldm_v15.yaml',
                  model_path:str='./models/control_sd15_scribble.pth', 
                  device:torch.device=torch.device("cuda"),
                  detect_model_type:str='yolov11',
                  class_names:list=[],
                  model_path_object_detection:str=None,
                  sam_model_type:str="vit_h",
                  sam_checkpoint_path:str="sam_vit_h_4b8939.pth",
                  captioner_model_name:str=r"./models/Salesforceblip_image_captioning_large",
                  inpaint_model_path:str=r"./sdxl-inpaint-model",
                  vae_model_path:str=r"models\sd_vae_ft_mse",
                  kwargs:dict=None
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
            "conf_threshold":0.25,
            "iou_threshold":0.1,
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
        self.class_names = class_names
        self.detect_model_type = detect_model_type
        self.model_path_object_detection=model_path_object_detection
        self.sam_model_type = sam_model_type
        self.sam_checkpoint_path = sam_checkpoint_path
        self.captioner_model_name=captioner_model_name
        self.inpaint_model_path=inpaint_model_path
        self.vae_model_path=vae_model_path
        




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



    def init_object_detection(self,device=None):
        """初始化目标检测模型"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 加载检测模型
        self.object_detection=ObjectDetection(self.detect_model_type,
                                              model_path=self.model_path_object_detection,
                                              class_names=self.class_names,
                                              device=device,
                                              **self.default_params)




    def destroy_object_detection(self):
        """销毁目标检测模型"""
        if hasattr(self, 'object_detection'):
            del self.object_detection
        # 清空内存
        torch.cuda.empty_cache()


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
            image_object_on_background=self.batched_tensor_mask_overlay(background_imag,image,mask_logic_np_select)
           
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
        adv_init_tensor=self.batched_tensor_mask_overlay(background_imag,controlnet_adv_sample,mask_logic_np_select)
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
        cross_entro_loss = YOLOv11DetectionLoss(** self.default_params)
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
            adv_tensor_optim=self.batched_tensor_mask_overlay(background_imag,adv_init_tensor,mask_logic_np_select)
           
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


#  缩放目标图像，再计算mask
    def generate_adversarial_main_two_stage_V2(self,background_imag=None, exp_path=r'./exp',images_path=None,mask_select_statues=0,params=None):
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
                visualize_attribution(background_imag,
                                    attributions_gt, 
                                    save_path=all_exp_root,
                                    file_name_pre='attribution_gt')
            else:
                print("Attribution failed!")


        #  优化图像 获取，基于mask
        adv_init_tensor=self.batched_tensor_mask_overlay(background_imag,
                                                         controlnet_adv_sample,
                                                         mask_logic_np_select)
        for i,exp_root_dir in enumerate(all_exp_root):
            tensor2picture(adv_init_tensor[i],os.path.join(exp_root_dir,
                                                            'adv_init.jpg'))
        adv_init_tensor=adv_init_tensor.detach().clone()

        # 检测，做参考
        result_epoch,_=self.object_detection.detect(adv_init_tensor,
                                                    file_path=all_exp_root,
                                                    file_name='adv_init_detect.jpg',grad_status=False)
        # 移动到GPU
        optim_device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        optim_data_type=torch.float32
        adv_init_tensor = move_to_gpu_and_cast_dtype(adv_init_tensor,optim_device,optim_data_type)
        adv_init_tensor_gt=adv_init_tensor.clone()
        result_gt=move_to_gpu_and_cast_dtype(result_gt,optim_device,optim_data_type)
        if params["attribution_loss_weight"]>0:
            attributions_gt = move_to_gpu_and_cast_dtype(attributions_gt,optim_device,optim_data_type)
        background_imag=move_to_gpu_and_cast_dtype(background_imag,optim_device,optim_data_type)
        result_epoch=move_to_gpu_and_cast_dtype(result_epoch,optim_device,optim_data_type)


        

        adv_init_tensor.requires_grad = True
        # 优化初始化

        optimizer = torch.optim.Adam([adv_init_tensor], lr=params["lr"])
        cross_entro_loss = YOLOv11DetectionLoss(** self.default_params).to(optim_device)
        if params["attribution_loss_weight"]>0:
            attr_loss_l2 = nn.MSELoss().to(optim_device)

        if params["TV_loss_weight"]>0:
            TV_Loss=TVLoss().to(optim_device) 
        if  params["conext_loss_weight"  ]>0:

            conext_loss_l2 = nn.MSELoss().to(optim_device)

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
                # 利用mask，只优化mask部分
                adv_tensor_optim=self.batched_tensor_mask_overlay(background_imag,
                                                                adv_init_tensor,
                                                                mask_logic_np_select)
            
                result_epoch,_=self.object_detection.detect(adv_tensor_optim,
                                                            file_path=all_exp_root,
                                                            file_name='result_generate.jpg',grad_status=True)


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
                # 这里损失的使用需要注意顺序，不能改变顺序
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
                    conext_loss=conext_loss_l2(adv_tensor_optim,adv_init_tensor_gt)
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
                    params["conext_loss_weight"]*conext_loss -\
                    loss_dict['class_loss']

            # 梯度裁剪：防止float16梯度爆炸
            torch.nn.utils.clip_grad_norm_([adv_init_tensor], max_norm=1.0)

            # ========== 反向传播：新版scaler用法不变 ==========
            scaler.scale(total_loss).backward()
            
            # ========== 参数更新：新版scaler用法不变 ==========
            scaler.step(optimizer)
            scaler.update()

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


        adv_tensor_final=self.batched_tensor_mask_overlay(background_imag,
                                                                adv_init_tensor,
                                                                mask_logic_np_select)        
        # 保存对抗样本
        for i,exp_root_dir in enumerate(all_exp_root):

            adv_path=os.path.join(exp_root_dir, 'adv_example.jpg')

            tensor2picture(adv_tensor_final[i],adv_path)
        release_torch_object_memory("perceptual_loss",namespace=locals())
        
        
        return 

#  缩放目标图像，再计算mask，利用vae 抑制伪纹理，vae加入到优化过程
    def generate_adversarial_main_two_stage_V3(self,background_imag=None, exp_path=r'./exp',images_path=None,mask_select_statues=0,params=None):
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
        adv_init_tensor=self.batched_tensor_mask_overlay(background_imag,controlnet_adv_sample,mask_logic_np_select)
        for i,exp_root_dir in enumerate(all_exp_root):
            tensor2picture(adv_init_tensor[i],os.path.join(exp_root_dir, 'adv_init.jpg'))
        adv_init_tensor=adv_init_tensor.detach().clone()

        # 检测，做参考
        result_epoch,_=self.object_detection.detect(adv_init_tensor,file_path=all_exp_root,file_name='adv_init_detect.jpg',grad_status=False)
        # 移动到GPU
        optim_device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        optim_data_type=torch.float32
        adv_init_tensor = move_to_gpu_and_cast_dtype(adv_init_tensor,optim_device,optim_data_type)
        adv_init_tensor_gt=adv_init_tensor.clone()
        result_gt=move_to_gpu_and_cast_dtype(result_gt,optim_device,optim_data_type)

        if params["attribution_loss_weight"]>0:
            attributions_gt = move_to_gpu_and_cast_dtype(attributions_gt,optim_device,optim_data_type)


        background_imag=move_to_gpu_and_cast_dtype(background_imag,optim_device,optim_data_type)
        result_epoch=move_to_gpu_and_cast_dtype(result_epoch,optim_device,optim_data_type)





        # vae初始化
        self.vae_optim=VAEInferencer(model_name=self.vae_model_path,
                                      dtype=optim_data_type)


        adv_init_tensor.requires_grad = True
        # 优化初始化
        optimizer = torch.optim.Adam([adv_init_tensor], lr=params["lr"])
        cross_entro_loss = YOLOv11DetectionLoss(** self.default_params).to(optim_device)
        if params["attribution_loss_weight"]>0:
            attr_loss_l2 = nn.MSELoss().to(optim_device)

        if params["TV_loss_weight"]>0:
            TV_Loss=TVLoss().to(optim_device) 
        if  params["conext_loss_weight"  ]>0:

            conext_loss_l2 = nn.MSELoss().to(optim_device)

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
                    
                # vae 的默认范围是是-1到1，这里将0到1 转化为-1到1
                adv_init_tensor1=adv_init_tensor*2-1
                adv_tensor_generate01=self.vae_optim.infer(adv_init_tensor1,sample_posterior=False)
                # 转化回来，将默认-1到1 的范围转化为0到1
                adv_tensor_generate=(adv_tensor_generate01+1)/2
                # 利用mask，只优化mask部分
                adv_tensor_optim=self.batched_tensor_mask_overlay(background_imag,
                                                                adv_tensor_generate,
                                                                mask_logic_np_select)
            
                result_epoch,_=self.object_detection.detect(adv_tensor_optim,
                                                            file_path=all_exp_root,
                                                            file_name='result_generate.jpg',
                                                            grad_status=True)

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
                    conext_loss=conext_loss_l2(adv_tensor_optim,adv_init_tensor_gt)
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
                    params["conext_loss_weight"]*conext_loss -\
                    loss_dict['class_loss']
                   
            # 梯度裁剪：防止float16梯度爆炸
            torch.nn.utils.clip_grad_norm_([adv_init_tensor], max_norm=1.0)
            
            # ========== 反向传播：新版scaler用法不变 ==========
            scaler.scale(total_loss).backward()
            
            # ========== 参数更新：新版scaler用法不变 ==========
            scaler.step(optimizer)
            scaler.update()

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



        adv_tensor_final=self.batched_tensor_mask_overlay(background_imag,
                                                                adv_init_tensor,
                                                                mask_logic_np_select)        
        # 保存对抗样本
        for i,exp_root_dir in enumerate(all_exp_root):

            adv_path=os.path.join(exp_root_dir, 'adv_example.jpg')

            tensor2picture(adv_tensor_final[i],adv_path)
        release_torch_object_memory("perceptual_loss",namespace=locals())

        
        
        return 
#  缩放目标图像，再计算mask，利用vae 抑制伪纹理，优化VAE得到的潜变量
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
        adv_init_tensor=self.batched_tensor_mask_overlay(background_imag,controlnet_adv_sample,mask_logic_np_select)
        for i,exp_root_dir in enumerate(all_exp_root):
            tensor2picture(adv_init_tensor[i],os.path.join(exp_root_dir, 'adv_init.jpg'))
        adv_init_tensor=adv_init_tensor.detach().clone()

        # 检测，做参考
        result_epoch,_=self.object_detection.detect(adv_init_tensor,file_path=all_exp_root,file_name='adv_init_detect.jpg',grad_status=False)
        # 移动到GPU
        optim_device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        optim_data_type=torch.float32
        adv_init_tensor = move_to_gpu_and_cast_dtype(adv_init_tensor,optim_device,optim_data_type)
        adv_init_tensor_gt=adv_init_tensor.clone()
        result_gt=move_to_gpu_and_cast_dtype(result_gt,optim_device,optim_data_type)

        if params["attribution_loss_weight"]>0:
            attributions_gt = move_to_gpu_and_cast_dtype(attributions_gt,optim_device,optim_data_type)


        background_imag=move_to_gpu_and_cast_dtype(background_imag,optim_device,optim_data_type)
        result_epoch=move_to_gpu_and_cast_dtype(result_epoch,optim_device,optim_data_type)

        # vae初始化
        self.vae_optim=VAEInferencer(model_name=self.vae_model_path,
                                      dtype=optim_data_type)
        # VAE的范围默认是-1到1
        adv_init_latent=self.vae_optim.encode_infer(adv_init_tensor*2-1)
        adv_init_latent=move_to_gpu_and_cast_dtype(adv_init_latent,optim_device,optim_data_type)
        adv_init_latent=adv_init_latent.clone()
        adv_init_latent = adv_init_latent.detach()
        adv_init_latent.requires_grad = True







        # 优化初始化
        optimizer = torch.optim.Adam([adv_init_latent], lr=params["lr"])
        cross_entro_loss = YOLOv11DetectionLoss(** self.default_params).to(optim_device)
        if params["attribution_loss_weight"]>0:
            attr_loss_l2 = nn.MSELoss().to(optim_device)

        if params["TV_loss_weight"]>0:
            TV_Loss=TVLoss().to(optim_device) 
        if  params["conext_loss_weight"  ]>0:

            conext_loss_l2 = nn.MSELoss().to(optim_device)

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
                    

                adv_tensor_generate01=self.vae_optim.decode_infer(adv_init_latent)
                # 转化回来，将默认-1到1 的范围转化为0到1
                adv_tensor_generate=(adv_tensor_generate01+1)/2
                # 利用mask，只优化mask部分
                adv_tensor_optim=self.batched_tensor_mask_overlay(background_imag,
                                                                adv_tensor_generate,
                                                                mask_logic_np_select)
            
                result_epoch,_=self.object_detection.detect(adv_tensor_optim,
                                                            file_path=all_exp_root,
                                                            file_name='result_generate.jpg',
                                                            grad_status=True)

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
                    conext_loss=conext_loss_l2(adv_tensor_optim,adv_init_tensor_gt)
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
                    params["conext_loss_weight"]*conext_loss -\
                    loss_dict['class_loss']
                   
            # 梯度裁剪：防止float16梯度爆炸
            torch.nn.utils.clip_grad_norm_([adv_init_tensor], max_norm=1.0)
            
            # ========== 反向传播：新版scaler用法不变 ==========
            scaler.scale(total_loss).backward()
            
            # ========== 参数更新：新版scaler用法不变 ==========
            scaler.step(optimizer)
            scaler.update()

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



        adv_tensor_final=self.batched_tensor_mask_overlay(background_imag,
                                                                adv_init_tensor,
                                                                mask_logic_np_select)        
        # 保存对抗样本
        for i,exp_root_dir in enumerate(all_exp_root):

            adv_path=os.path.join(exp_root_dir, 'adv_example.jpg')

            tensor2picture(adv_tensor_final[i],adv_path)
        release_torch_object_memory("perceptual_loss",namespace=locals())

        
        
        return 

#  缩放目标图像，再计算mask
    def generate_adversarial_main_two_stage_V2_withref_canny(self,background_imag=None,ref_img=None,  exp_path=r'./exp',images_path=None,mask_select_statues=0,params=None):
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
        adv_init_tensor=self.batched_tensor_mask_overlay(background_imag,controlnet_adv_sample,mask_logic_np_select)
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
        cross_entro_loss = YOLOv11DetectionLoss(** self.default_params)
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
            adv_tensor_optim=self.batched_tensor_mask_overlay(background_imag,adv_init_tensor,mask_logic_np_select)
           
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

# canny里面 加入参考引导
    def generate_adversarial_main_two_stage_withref_canny(self,background_imag=None,ref_img=None, exp_path=r'./exp',images_path=None,mask_select_statues=0,params=None):
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
        adv_init_tensor=self.batched_tensor_mask_overlay(background_imag,controlnet_adv_sample,mask_logic_np_select)
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
        cross_entro_loss = YOLOv11DetectionLoss(** self.default_params)
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
            adv_tensor_optim=self.batched_tensor_mask_overlay(background_imag,adv_init_tensor,mask_logic_np_select)
           
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


    def pad_to_square(self,img_tensor, pad_mode="constant", fill_value=0.0):
        """
        将输入的图像 tensor 填充为正方形（宽高相等，且为原始最大边长）
        
        参数：
            img_tensor: 输入图像 tensor，形状为 (C, H, W) 或 (B, C, H, W)
            pad_mode: 填充模式，同 F.pad 的 mode 参数（如 "constant", "edge", "reflect" 等）
            fill_value: 填充值（当 pad_mode 为 "constant" 时有效）
        
        返回：
            padded_tensor: 填充后的正方形 tensor，形状为 (C, max_dim, max_dim) 或 (B, C, max_dim, max_dim)
        """
        # 处理单张图像（C, H, W）或批量图像（B, C, H, W）
        if img_tensor.ndim == 3:
            C, H, W = img_tensor.shape
            batch_mode = False
        elif img_tensor.ndim == 4:
            B, C, H, W = img_tensor.shape
            batch_mode = True
        else:
            raise ValueError(f"输入 tensor 维度必须是 3 (C, H, W) 或 4 (B, C, H, W)，但得到 {img_tensor.ndim}")
        
        max_dim = max(H, W)
        
        # 计算填充量：(上, 下, 左, 右)
        # 上下填充总和 = max_dim - H；左右填充总和 = max_dim - W
        pad_top = (max_dim - H) // 2
        pad_bottom = max_dim - H - pad_top  # 确保上下填充总和正确（处理奇数情况）
        pad_left = (max_dim - W) // 2
        pad_right = max_dim - W - pad_left  # 确保左右填充总和正确
        
        # 构造填充参数（注意：pad 的顺序是 (左, 右, 上, 下) 对于最后两个维度）
        pad = (pad_left, pad_right, pad_top, pad_bottom)
        
        # 执行填充
        if batch_mode:
            # 批量图像：在 H 和 W 维度填充（即最后两个维度）
            padded_tensor = F.pad(img_tensor, pad, mode=pad_mode, value=fill_value)
        else:
            # 单张图像：同样在 H 和 W 维度填充
            padded_tensor = F.pad(img_tensor, pad, mode=pad_mode, value=fill_value)
        
        return padded_tensor


    def yolo_boxes_to_corners(self, boxes):
        """
        将多batch YOLO检测框转换为中心点坐标列表
        
        参数：
            boxes: 检测框输入，支持两种格式：
                   - 多batch列表：List[torch.Tensor]，每个元素形状为 (N, 4)（xyxy格式），对应一个batch的检测框
                   - 单batch张量：torch.Tensor，形状为 (N, 4)（xyxy格式）
            img_shape: 图像尺寸 (height, width)，若需归一化坐标转绝对坐标则传入
        
        返回：
            corners_list: 嵌套列表，外层长度=batch数，内层每个元素为 [x_center, y_center]
        """
        # 统一输入格式为多batch列表
        if isinstance(boxes, torch.Tensor):
            boxes = [boxes]  # 单batch转为列表
        
        corners_list = []
        # 遍历每个batch
        for batch_idx, batch_boxes in enumerate(boxes):
            batch_corners = []
            if batch_boxes.numel() == 0:  # 当前batch无检测框
                corners_list.append(batch_corners)
                continue
            
            # 维度校验
            if batch_boxes.ndim != 2 or batch_boxes.shape[1] != 4:
                raise ValueError(f"Batch {batch_idx} 检测框形状错误，期望 (N, 4)，实际 {batch_boxes.shape}")
            
            # 转换为numpy（也可保留张量计算）
            batch_boxes_np = batch_boxes.detach().cpu().numpy()
            
            # 计算每个框的中心点
            for box in batch_boxes_np:
                x1, y1, x2, y2 = box
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                

                
                # 可选：转为整数
                x_center, y_center = map(int, [x_center, y_center])
                batch_corners.append([x_center, y_center])
            
            corners_list.append(batch_corners)
        
        return corners_list


    def filter_max_box_per_batch(self,result: dict,class_names) -> dict:
        """
        筛选每个batch中面积最大的检测框，返回与输入格式一致的result_gt字典
        
        Args:
            result_gt: 原始gt字典，包含以下key（值均为列表，每个元素对应一个batch的Tensor）:
                - labels: 列表，每个元素是 [N,] Tensor（N为该batch的框数量，存储类别标签）
                - boxes: 列表，每个元素是 [N, 4] Tensor（框坐标，格式支持 xyxy/xywh）
                - scores: 列表，每个元素是 [N,] Tensor（置信度分数）
                - scores_vector: 列表，每个元素是 [N, D] Tensor（D为分数向量维度）
        
        Returns:
            filtered_gt: 筛选后的字典，结构与输入一致，每个batch仅保留面积最大的框
                        若某batch无框（Tensor为空），则保留空Tensor
                        
        注意：
            - boxes坐标格式支持 xyxy（左上x, 左上y, 右下x, 右下y）或 xywh（左上x, 左上y, 宽, 高）
            - 自动适配Tensor设备（CPU/GPU），保持与输入一致
        """
        # 初始化输出字典，与输入格式对齐
        filtered_gt = {
            'labels': [],
            'boxes': [],
            'scores': [],
            'scores_vector': []
        }
        filtered_class_names = []
        # 遍历每个batch的信息（按列表索引对齐）
        batch_num = len(result['labels'])
        for b_idx in range(batch_num):
            # 取出当前batch的所有数据（处理空值，避免索引报错）
            labels = result['labels'][b_idx] if b_idx < len(result['labels']) else torch.tensor([], dtype=torch.int64)
            boxes = result['boxes'][b_idx] if b_idx < len(result['boxes']) else torch.tensor([], dtype=torch.float32)
            scores = result['scores'][b_idx] if b_idx < len(result['scores']) else torch.tensor([], dtype=torch.float32)
            scores_vector = result['scores_vector'][b_idx] if b_idx < len(result['scores_vector']) else torch.tensor([], dtype=torch.float32)
            class_name_temp=class_names[b_idx] if b_idx < len(class_names) else ''

            # 处理空框场景：当前batch无检测框，直接添加空Tensor
            if boxes.numel() == 0:
                filtered_gt['labels'].append(labels)
                filtered_gt['boxes'].append(boxes)
                filtered_gt['scores'].append(scores)
                filtered_gt['scores_vector'].append(scores_vector)
                filtered_class_names.append(class_name_temp)
                continue
            
            # ========== 核心：计算每个框的面积，筛选最大面积的框 ==========
            # 统一转换为 xyxy 格式计算面积（兼容xyxy/xywh输入）
            if boxes.shape[1] == 4:
                # 区分 xyxy 和 xywh：xyxy的宽高为 (x2-x1, y2-y1)；xywh的宽高为 (w, h)
                if (boxes[:, 2] > boxes[:, 0]).all() and (boxes[:, 3] > boxes[:, 1]).all():
                    # 判定为 xyxy 格式（x2>x1, y2>y1）
                    w = boxes[:, 2] - boxes[:, 0]
                    h = boxes[:, 3] - boxes[:, 1]
                else:
                    # 判定为 xywh 格式
                    w = boxes[:, 2]
                    h = boxes[:, 3]
                area = w * h  # 计算每个框的面积 [N,]
            else:
                raise ValueError(f"boxes维度错误，需为 [N,4]，当前为 {boxes.shape}")
            
            # 找到最大面积的索引（若多个框面积相同，取第一个）
            max_area_idx = torch.argmax(area)
            
            # 筛选该索引对应的框、标签、分数
            filtered_labels = labels[max_area_idx:max_area_idx+1]  # 保留维度 [1,]
            filtered_boxes = boxes[max_area_idx:max_area_idx+1]    # 保留维度 [1,4]
            filtered_scores = scores[max_area_idx:max_area_idx+1]  # 保留维度 [1,]
            filtered_scores_vector = scores_vector[max_area_idx:max_area_idx+1]  # 保留维度 [1,D]
            filtered_class_names.append(class_name_temp[max_area_idx])
            # 将筛选结果添加到输出字典
            filtered_gt['labels'].append(filtered_labels)
            filtered_gt['boxes'].append(filtered_boxes)
            filtered_gt['scores'].append(filtered_scores)
            filtered_gt['scores_vector'].append(filtered_scores_vector)
        
        return filtered_gt,filtered_class_names





                  
    def extract_mask_content(self, input_tensor, mask, mask_value=1.0):
        """
        从输入 tensor 中抠出 mask 内的内容，mask 外区域设为指定值（默认为 1.0）
        
        参数：
            input_tensor: 输入图像 tensor（格式：BCHW 或 CHW，值范围 0-1）
            mask: 二维掩码 array 或 tensor（格式：(H, W)，元素为 True/False，True 表示保留区域）
            mask_value: mask 外区域的填充值（默认 1.0）
        
        返回：
            result_tensor: 处理后的 tensor，mask 内保留原图内容，mask 外为 mask_value
        """
        # 记录原始输入维度，用于最终格式恢复
        original_dim = input_tensor.dim()
        
        # 1. 统一输入 tensor 为 BCHW 格式（确保有 batch 维度）
        if original_dim == 3:  # CHW → BCHW
            input_tensor = input_tensor.unsqueeze(0)
        B, C, H, W = input_tensor.shape  # 此时 input_tensor 一定是 BCHW
        result_list=[]
        for i in range(B):  # 批量处理

            mask_b=mask[i]
            input_tensor_b=input_tensor[i]
            # 2. 处理二维 mask，转为 tensor 并扩展维度以匹配 BCHW
            if isinstance(mask_b, np.ndarray):
                mask_b = torch.from_numpy(mask_b).bool()  # numpy 转 bool tensor
            else:
                mask_b = mask_b.bool()  # 确保是 bool 类型
            
            # 扩展 mask 维度：(H, W) → (1, 1, H, W)，再通过广播匹配 (B, C, H, W)
            mask_b = mask_b.unsqueeze(0).unsqueeze(0)  # 增加 batch 和 channel 维度
            mask_b = mask_b.to(input_tensor_b.device)  # 确保与输入 tensor 同设备
            
            # 3. 生成填充值 tensor（与输入同形状）
            fill_tensor = torch.full_like(input_tensor_b, fill_value=mask_value)
            
            # 4. 核心操作：mask 内保留原图，mask 外填充
            result_tensor = torch.where(mask_b, input_tensor_b, fill_tensor)
            
            result_list.append(result_tensor)
        result_tensor_all=torch.cat(result_list)
        return result_tensor_all





    def batched_tensor_mask_overlay(self,background_tensor, image_tensor, mask_array):
        """
        批量处理四维张量的mask覆盖操作，同时保证梯度传导
        
        参数:
        background_tensor: 背景张量，形状为[B, C, H, W]
        image_tensor: 前景张量，形状为[B, C, H, W]，需与背景张量尺寸匹配
        mask_array: 布尔数组，形状为[B, H, W]或[H, W]，True表示需要覆盖的区域
        
        返回:
        result_tensor: 合成后的张量，形状为[B, C, H, W]，保留梯度信息
        """
        # 确保device一致
        on_device=image_tensor.device
        background_tensor=background_tensor.to(on_device)

        # 确保输入张量形状匹配
        assert background_tensor.shape == image_tensor.shape, "背景和前景张量形状必须相同"

        
        # 处理mask形状，确保与输入张量匹配
        if mask_array.ndim == 2:  # [H, W] - 对所有批次使用相同mask
            mask_array = np.expand_dims(mask_array, axis=0)  # [1, H, W]
        assert mask_array.shape == (background_tensor.shape[0], background_tensor.shape[2], background_tensor.shape[3]), \
            f"mask形状应为[B, H, W]，实际为{mask_array.shape}"
        
        # 将mask数组转换为与输入张量匹配的形状 [B, C, H, W]
        mask = mask_array.astype(np.float32)
        mask = np.expand_dims(mask, axis=1)  # [B, 1, H, W]
        mask = np.repeat(mask, background_tensor.shape[1], axis=1)  # [B, C, H, W]
        
        # 转换为张量并确保与输入在同一设备，同时保留梯度计算能力
        mask_tensor = torch.from_numpy(mask).to(on_device, dtype=image_tensor.dtype)
        
        # 执行覆盖操作: 背景*(1-mask) + 前景*mask
        # 所有操作均为PyTorch张量操作，会自动跟踪梯度
        result_tensor = background_tensor * (1 - mask_tensor) + image_tensor * mask_tensor
        
        return result_tensor










# 使用示例
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



    #参数

    root_path=os.path.join(os.path.dirname(__file__), "..")
    img_path=os.path.join(os.path.join(root_path,'test_imgs'),'man.png')
    img=cv2.imread(img_path)
    img=cv2.resize(img,(256,256))



    model_path=os.path.join(os.path.join(root_path,'models'),'yolo11n.pt')
    sam_path=os.path.join(os.path.join(root_path,'models'),'sam_vit_h_4b8939.pth')
    controlNet_model_path=os.path.join(os.path.join(root_path,'models'),'control_sd15_canny.pth')
    # sam_path=r"D:\FILELin\postgraduate\little_paper\Adversariall_attack_project\ControlNet\models\sam_vit_b_01ec64.pth"
    # controlNet_model_path=r"D:\FILELin\postgraduate\little_paper\Adversariall_attack_project\ControlNet\models\control_sd15_canny.pth"
    attack = ADV_ATTACK(device=torch.device("cuda"),model_path=controlNet_model_path,model_path_object_detection=model_path,sam_model_type='vit_h',sam_checkpoint_path=sam_path)
    img=cv2_to_tensor(img,normalize=True)
    attack.generate_adversarial_main(img)
    attack.init_vae()
    # 测试
    if img.dim()==3:  # 添加维度
        img = img.unsqueeze(0)
    # VAE测试
    img=move_to_gpu(img)
    attack.vae.to(device=torch.device("cuda"))
    img.requires_grad=True
    attack.vae.eval()
    start_time=time.time()
    posterior_vae=attack.vae.encode(img*2-1)
    latent=posterior_vae.mode()
    img_g=attack.vae.decode(latent)
    end_time=time.time()
    print(f"VAE time: {end_time - start_time:.2f}s")
    # 转化为0-1
    img_g=(img_g+1)/2
    tensor2picture(img_g[0],'test.jpg')
    temp,_=attack.vae(img*2-1,sample_posterior=False)
    temp=(temp+1)/2
    tensor2picture(img_g[0],'test.jpg')
    loss = torch.nn.functional.mse_loss(img_g, img)  # 对比重建图和原图
    loss.backward()  # 反向传播，计算梯度

    # 4. 验证梯度是否回传
    print("===== 梯度验证结果 =====")
    # 检查 img 的梯度是否存在且非全零
    if img.grad is not None:
        grad_norm = torch.norm(img.grad).item()  # 计算梯度范数（标量）
        print(f"img.grad 存在，梯度范数：{grad_norm:.6f}")
        if grad_norm > 1e-8:  # 梯度非全零（浮点误差容忍）
            print("✅ 梯度成功回传到 img！")
        else:
            print("❌ img.grad 为全零，梯度未有效回传！")
    else:
        print("❌ img.grad 不存在，梯度被截断！")
    # attack.generate_adversarial_example_optim_control_v2(img)
    # attack.generate_adversarial_example(img,control_img)
    # attack.generate_adversarial_example_optim_control_v2(img,control_img)
    # attack.generate_adversarial_example_optim_control(img,control_img)