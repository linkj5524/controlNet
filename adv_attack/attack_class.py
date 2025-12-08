
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

# 本地的包
## 添加本地包路径,即上一级的路径
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
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

class ADV_ATTACK:
    def __init__(self, config_path:str='./models/cldm_v15.yaml',
                  model_path:str='./models/control_sd15_scribble.pth', 
                  device:torch.device=torch.device("cuda"),
                  detect_model_type:str='yolov11',
                  class_names:list=[],
                  model_path_object_detection:str=None,
                  sam_model_type:str="vit_h",
                  sam_checkpoint_path:str="sam_vit_h_4b8939.pth",
                  captioner_model_name:str="models\\Salesforceblip-image-captioning-large",
                  inpaint_model_path:str="./sdxl-inpaint-model",
                  **kwargs
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
            "prompt": "a handsome boy with a long hair",
            "a_prompt": "",
            "n_prompt": "",
            "num_samples": 1,
            "image_resolution":512,
            "ddim_steps": 10,
            "guess_mode": False,
            "strength": 1.0,
            "scale": 9,
            "scale_optim":2, # 优化过程的控制
            "seed": 42,
            "eta": 0.0,
            "save_memory": True,
            "optim_epochs":30,
            "latent_fit_optim_epochs":10,
            "conf_threshold":0.25,
            "iou_threshold":0.1,
            "attribution_loss_weight" :100,
            "TV_loss_weight":10,
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



    def init_object_detection(self):
        """初始化目标检测模型"""

        # 加载检测模型
        self.object_detection=ObjectDetection(self.detect_model_type,
                                              model_path=self.model_path_object_detection,
                                              class_names=self.class_names,
                                              device=torch.device("cpu"),
                                              **self.default_params)

    def destroy_object_detection(self):
        """销毁目标检测模型"""
        if hasattr(self, 'object_detection'):
            del self.object_detection
        # 清空内存
        torch.cuda.empty_cache()


    def generate_adversarial_example(self, input_image=None, control_image=None,params=None):
        """
        生成对抗样本
        
        参数:
            control_image: 控制图像 (用于ControlNet)
            params: 覆盖默认参数的字典
            
        返回:
            生成的对抗图像列表
        """
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
        
       
       
       
        
        if params["save_memory"]:
            self.model.low_vram_shift(is_diffusing=False)




        # 预处理图像
        ## 确保图像通道正常,对图像的size有要求，不能随便的大小
        # 确保是三个通道的numpy，int8类型
        input_image=HWC3(input_image) ## input_image RGB,int 8,主要判断输入图像的通道数，输出RGB，int8
        input_image = cv2.resize(input_image, (self.default_params["image_resolution"],self.default_params["image_resolution"])) # input_image RGB

        ## 处理控制图像，并返回边缘图和边缘的control
        if control_image is not None:
            control_image=HWC3(control_image) ## control_image RGB
            control_image = cv2.resize(control_image, (self.default_params["image_resolution"],self.default_params["image_resolution"])) # control_image RGB
            self.edge_image,self.control=self.generate_edge_control_from_image(control_image)
        else:
            ## 处理控制图像，并返回边缘图和边缘的control
            self.edge_image,self.control=self.generate_edge_control_from_image(input_image,'./exp/canny_edge.jpg')
        


        # c_concat 草图控制；c_crossattn 跨模态控制：正向和附加的文本提示;  文本内容默认用clip编码
        cond = {
            "c_concat": [self.control],
            "c_crossattn": [
                self.model.get_learned_conditioning(
                    [params["prompt"] + ', ' + params["a_prompt"]] * params["num_samples"]
                )
            ]
        }
        un_cond = {
            "c_concat": None if params["guess_mode"] else [self.control],
            "c_crossattn": [
                self.model.get_learned_conditioning(
                    [params["n_prompt"]] * params["num_samples"]
                )
            ]
        }
 
 
        H, W, C = input_image.shape
        shape = (4, H // 8, W // 8)

        if params["save_memory"]:
            self.model.low_vram_shift(is_diffusing=True)


        self.model.control_scales = (
            [params["strength"] * (0.825 ** float(12 - i)) for i in range(13)]
            if params["guess_mode"]
            else [params["strength"]] * 13
        )  # Magic number. IDK why
        
        #对图像进行编码，转换为latent
        temp_tensor=self.to_imgTensor_from_numpy_int8(input_image) # 输出-1,1 ，目前正常
        latent_input=self.imgTensor_to_latent(temp_tensor)

        # 设置采样参数
        self.ddim_sampler.make_schedule(ddim_num_steps=params["ddim_steps"], ddim_eta=params["eta"], verbose=False)
        # 使用封装的ddim进行逆采样
        ## latent_start 表示逆采样的结果（噪声最大的latent），out 表示所有的中间结果
        with torch.no_grad():
            latent_start,out=self.ddim_sampler.encode_return_all(x0=latent_input, c=cond, t_enc=params["ddim_steps"], use_original_steps=False, return_intermediates=True,
            unconditional_guidance_scale=params["scale"], unconditional_conditioning=un_cond, callback=None)

        # 获取目标检测模型的输出，也可以直接传入这些已知的信息，输入bchw，0-1
        result_gt =self.object_detection.detect(input_image,file_path='result1.jpg',grad_status=True)
        





        cond_ref = {
            "c_concat": [self.control],
            "c_crossattn": [
                self.model.get_learned_conditioning(
                    ["military camouflage pattern,army green and brown color sheme"] * params["num_samples"]
                )
            ]
        }
        # 保存中间结果
        # 对初始latent进行优化
        # 需要 1. 优化目标 2. 优化器 3. 优化参数 4. 后处理函数
        # 优化目标：目标检测模型的输出与原始的检测框，类别等的差值
        # 优化器：Adam
        # 优化参数：latent
        # 后处理函数，根据检测模型的输出，得到结果，并进行优化
        


        # require grad
        #断开，减少内存消耗
        latent_start=latent_start.detach().clone()
        latent_start.requires_grad = True
        optimizer = torch.optim.Adam([latent_start], lr=1e-1)
        cross_entro_loss = YOLOv11DetectionLoss(** self.default_params)
  
        #开始步骤
        t_start=self.ddim_sampler.ddim_timesteps[-1]
        for epoch in range(params["optim_epochs"]):
            # 循环，优化
            end_latent=self.ddim_sampler.decode(  latent_start, cond, t_start, unconditional_guidance_scale=params["scale"], unconditional_conditioning=un_cond,
                use_original_steps=False, callback=None)



            # 转换成图片
            image=self.latent_to_imgTensor01(end_latent,1)
 
 
            # 目标检测模型的输出
            result  =self.object_detection.detect(image,file_path='restore.jpg',grad_status=True)
            print(f"result:{result}")
            loss ,loss_dict= cross_entro_loss(result, result_gt)
            print(f"total_loss:{loss}")
            optimizer.zero_grad()
            
            loss_dict['giou_loss']
            try:
                # loss_dict['bbox_l1_loss'].backward()
                # loss_dict['giou_loss']
                (-loss_dict['class_loss']).backward()
                # (-loss).backward()
            except:
                print("loss_dict['class_loss'] is None")
                break          

            optimizer.step()
            # 手动清理变量，帮助回收内存
            del loss,loss_dict
            torch.cuda.empty_cache()


        # 得到最终的对抗样本    
        # end_latent=self.ddim_sampler.decode(  latent_start, cond_ref, t_start, unconditional_guidance_scale=20.0, unconditional_conditioning=un_cond,
        #         use_original_steps=False, callback=None)
        image_tensor=self.latent_to_imgTensor01(end_latent)

        image1=self.tensor_01_to_numpy_255(image_tensor)
        # 保存对抗样本
        cv2.imwrite('result_adv_attack.png',image1)





        
        # 采样过程实现
        # t_start=self.ddim_sampler.ddim_timesteps[-1]
        # # 得到最终的latent
        # end_latent=self.ddim_sampler.decode(  latent_start, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
        #        use_original_steps=False, callback=None)




        # # # 调试
        # # temp=out['intermediates'][0]
        # image_temp=self.latent_to_imgTensor01(end_latent)
        
        # bbox_xyxy, confidences, class_ids_gt  =self.object_detection.detect(image_temp,imgsize=self.default_params["image_resolution"],file_path='result2.jpg')
        # class_ids_gt_tensor = torch.tensor(class_ids_gt, dtype=torch.long, device=latent_input.device) # self.device
        # confidences_gt_tensor = torch.tensor(confidences, dtype=torch.float, device=latent_input.device)

        # # require grad
        # latent_start.requires_grad = True
        # optimizer = torch.optim.Adam([latent_start], lr=1e-3)
        # cross_entro_loss = torch.nn.CrossEntropyLoss()
        # print("输入模型前的x_next.requires_grad：", latent_start.requires_grad)  # 必须为 True
        # print("输入模型前的x_next.grad_fn：", latent_start.grad_fn)   
        # for epoch in range(params["optim_epochs"]):
            





        #     # image=self.latent_to_imgTensor01(x_next)
        #     print("输入模型前的image.requires_grad：", image.requires_grad)  # 必须为 True
        #     print("输入模型前的image.grad_fn：", image.grad_fn)   
        #     bbox_xyxy, confidences, class_ids  =self.object_detection.detect(image,imgsize=self.default_params["image_resolution"])
        #     print("输入模型前的confidences.confidences", confidences.requires_grad)  # 必须为 True
        #     print("输入模型前的confidences.grad_fn：", confidences.grad_fn)   

        #     # 计算检测的损失
        #     class_ids_tensor = torch.tensor(class_ids, dtype=torch.long, device=x_next.device)
        #     confidences_tensor = torch.tensor(confidences, dtype=torch.float, device=x_next.device)
        #     loss = cross_entro_loss(confidences_tensor, confidences_gt_tensor)
        #     optimizer.zero_grad()
        #     (-loss).backward()
        #     optimizer.step()

        # # 得到最终的对抗样本    
        # image=self.latent_to_imgTensor01(x_next)
        
        # # 保存对抗样本
        # cv2.imwrite('result.png',image[0])

        # 保存中间结果
        # 对初始latent进行优化
        # 需要 1. 优化目标 2. 优化器 3. 优化参数 4. 后处理函数
        # 优化目标：目标检测模型的输出与原始的检测框，类别等的差值
        # 优化器：Adam
        # 优化参数：latent
        # 后处理函数，根据检测模型的输出，得到结果，并进行优化
        






        # x_next, out=self.ddim_sampler.encode_return_all(x0=latent, c=cond, t_enc=params["ddim_steps"], use_original_steps=False, return_intermediates=True,
        #        unconditional_guidance_scale=9, unconditional_conditioning=un_cond, callback=None)
        # 初始图片的推理结果
        # bbox_xyxy, confidences, class_ids  =self.object_detection.detect(input_image,imgsize=self.default_params["image_resolution"])






        
        # #实验
        # image=self.latent_to_imgTensor01(out["intermediates"][0])

        
        # bbox_xyxy, confidences, class_ids  =self.object_detection.detect(image)
        
        # # 将所有的latent转换成图片
        # path="./exp/result"
        # ## 判断路径是否存在，不存在则创建
        # if not os.path.exists(path):
        #     os.mkdir(path)
        
        # for i in range(len(out["intermediates"])):
        #     image=self.latent_to_imgTensor01(out["intermediates"][i])

        #     cv2.imwrite("{}/{}.png".format("./exp/result", i), image[0])
        # images = self.latent_to_imgTensor01(latent)
        
        # img1=cv2.cvtColor(images[0], cv2.COLOR_RGB2BGR)
        
        # cv2.imwrite('result1.png',images[0])
        
        return 
    # def generate_adversarial_preprocess(self,background_imag):
    #     # 预处理图像,做分割，检测，返回mask（tensor）

    #     mask=self.generate_mask(background_imag)

    def generate_adversarial_main(self,background_imag=None, exp_path=r'./exp',params=None):
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
        


        # background——image 的文本描述提取
        blip_model, blip_processor, blip_device=init_image_captioner(self.captioner_model_name)
        background_imag_caption = image_captioner_process(blip_model, blip_processor, blip_device, background_imag)
        print(f"\n背景图像描述: {background_imag_caption}")
        destroy_image_captioner(blip_model)






       # detect model 初始化
        self.init_object_detection()
        if background_imag.ndim == 3:
            background_imag_temp=background_imag.unsqueeze(0)
        else:
            background_imag_temp=background_imag
        ref_detect_path=os.path.join(exp_path, 'background_detect.jpg')
        result_gt,object_class =self.object_detection.detect(background_imag_temp,file_path=ref_detect_path,grad_status=True)
        








        input_point_list=self.yolo_boxes_to_corners(result_gt['boxes'])
        input_point=[input_point_list[0]]
        # 基于输入的background_imag，利用sam，得到mask，返回mask。
        # 模型初始化
        sam_predicter=init_sam(model_type=self.sam_model_type, checkpoint_path=self.sam_checkpoint_path)
        # 处理,注意mask——logic 的维度，是否是多个通道
        img_np, masks_logic_mutil, masks_tensor_all, scores=segment_tensor(sam_predicter, background_imag, input_point=input_point, input_label=[1],mutil_mask=True)
        # detach
        masks_tensor_all=masks_tensor_all.detach()
        
        

        # visualize_sam(background_imag, masks_logic_mutil, scores)
        destroy_sam(sam_predicter)



        # control 处理

        # 填充，计算canny，返回tensor
        if masks_logic_mutil.shape[0]>1:
            # 计算掩码区域最大的索引
            mask_areas = masks_logic_mutil.sum(axis=(1, 2))  # 对 H 和 W 维度求和，得到每个通道的面积

            # 找到面积最大的通道索引
            index = np.argmax(mask_areas)
            print(f"index:{index},score:{scores[index]}")
            masks_logic=masks_logic_mutil[index]
            masks_tensor=masks_tensor_all[index]
            mask_path=os.path.join(exp_path, 'mask.jpg')
            tensor2picture(masks_tensor,mask_path)


            
        control_image=self.canny_with_mask_invert(background_imag,masks_logic)
        control_path=os.path.join(exp_path, 'control.jpg')
        tensor2picture(control_image,control_path) 
       



        # # 背景补全
        # ## 提示词修改
        # inpaint_caption=generate_inpaint_prompt(background_imag_caption, target_object=object_class)

        # ## inpaint model 初始化
        # inpaint_pipe=init_sdxl_inpaint(self.inpaint_model_path)
        # # 这里的模型mask1表示不许掩盖的，0表示背景，需要注意
        # inpainted_tensor = process_sdxl_inpaint(
        #     pipe=inpaint_pipe,
        #     image_tensor=background_imag,
        #     mask_tensor=masks_tensor,
        #     prompt=inpaint_caption[0],
        #     num_steps=30,
        #     guidance_scale=9
        # )
        # display_and_save_results(background_imag, masks_tensor, inpainted_tensor)
        # destroy_sdxl_inpaint(inpaint_pipe)




 






        # 初始化模型
        self.init_controlnet()
        if params["save_memory"]:
            self.model.low_vram_shift(is_diffusing=False)




        # # 预处理图像
        # ## 确保图像通道正常,对图像的size有要求，不能随便的大小
         
        # input_image=scale_tensor_to_resolution(input_image,self.default_params["image_resolution"],self.default_params["image_resolution"])
        
        # # control_image 处理，背景图像，mask，得到control，计算canny边缘。
        
        # control_image=scale_tensor_to_resolution(control_image,self.default_params["image_resolution"],self.default_params["image_resolution"])
        
        # self.control=binarize_image_tensor(control_image)     
        # 添加b
        # 判断control 的维度
        if control_image.dim()==3:
            control_image=control_image.unsqueeze(0)

        


        

        # c_concat 草图控制；c_crossattn 跨模态控制：正向和附加的文本提示;文本内容默认用clip编码
        cond = {
            "c_concat": [control_image],
            "c_crossattn": [
                self.model.get_learned_conditioning(
                    [background_imag_caption + ', ' + params["a_prompt"]] * params["num_samples"]
                )
            ]
        }
        un_cond = {
            "c_concat": None if params["guess_mode"] else [control_image],
            "c_crossattn": [
                self.model.get_learned_conditioning(
                    [params["n_prompt"]] * params["num_samples"]
                )
            ]
        }
 
        # 判断维度  
        if background_imag.dim()==3:
            background_imag=background_imag.unsqueeze(0)
        B,C,H, W= background_imag.shape
        shape = (4, H // 8, W // 8)








        if params["save_memory"]:
            self.model.low_vram_shift(is_diffusing=True)


        self.model.control_scales = (
            [params["strength"] * (0.825 ** float(12 - i)) for i in range(13)]
            if params["guess_mode"]
            else [params["strength"]] * 13
        )  # Magic number. IDK why
        
        object_image=self.extract_mask_content(background_imag,masks_logic)

        # tensor2picture(object_image,"object_image.png") 
        # 输入是-1~1
        # 将0,1 转换成-1,1
        object_image_n1_1=object_image*2-1
        latent_input = self.imgTensor_to_latent(object_image_n1_1)

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
            visualize_attribution(object_image, attributions_ref, save_path=exp_path,file_name_pre='attribution_ref')
        else:
            print("Attribution failed!")

        # # test
        # x_samples = self.model.decode_first_stage(latent_input)
        # # 维度变换
        # x_samples1 = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        # img2=cv2.cvtColor(x_samples1[0], cv2.COLOR_RGB2BGR)
        # cv2.imwrite('result_sample.png',img2)
        # # 转换成图片
        # image=self.latent_to_imgTensor01(latent_input)

        # image1=self.tensor_01_to_numpy_255(image)
        # # RGB to BGR
        # image1=cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
        # # 保存对抗样本
        # cv2.imwrite('result_adv_attack.png',image1)





        # 设置采样参数
        self.ddim_sampler.make_schedule(ddim_num_steps=params["ddim_steps"], ddim_eta=params["eta"], verbose=False)

        
        # samples, intermediates = self.ddim_sampler.sample(params['ddim_steps'], params['num_samples'],
        #                                                     shape, cond, verbose=False, eta=params['eta'],
        #                                                     unconditional_guidance_scale=params['scale'],
        #                                                     unconditional_conditioning=un_cond)


        # x_samples = self.model.decode_first_stage(samples)
        # # 维度变换
        # x_samples1 = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        # img2=cv2.cvtColor(x_samples1[0], cv2.COLOR_RGB2BGR)
        # cv2.imwrite('result_sample.png',img2)


        # 使用封装的ddim进行逆采样
        ## latent_start 表示逆采样的结果（噪声最大的latent），out 表示所有的中间结果
        with torch.no_grad():
            latent_start,out=self.ddim_sampler.encode_return_all(x0=latent_input, c=cond, t_enc=params["ddim_steps"], use_original_steps=False, return_intermediates=True,
            unconditional_guidance_scale=params["scale"], unconditional_conditioning=un_cond, callback=None)

        # 获取目标检测模型的输出，也可以直接传入这些已知的信息,用于后续计算，确保只用某个目标
        object_path = os.path.join(exp_path, 'object_detext.jpg')
        result_gt_temp,class_name =self.object_detection.detect(object_image,file_path=object_path,grad_status=True)
        





        cond_ref = {
            "c_concat": [control_image],
            "c_crossattn": [
                self.model.get_learned_conditioning(
                    ["military camouflage pattern,army green and brown color sheme"] * params["num_samples"]
                )
            ]
        }
        # 保存中间结果
        # 对初始latent进行优化
        # 需要 1. 优化目标 2. 优化器 3. 优化参数 4. 后处理函数
        # 优化目标：目标检测模型的输出与原始的检测框，类别等的差值
        # 优化器：Adam
        # 优化参数：latent
        # 后处理函数，根据检测模型的输出，得到结果，并进行优化











        latent_start=latent_start.detach().clone()
        latent_start.requires_grad = True
        optimizer = torch.optim.Adam([latent_start], lr=5e-2)
        cross_entro_loss = YOLOv11DetectionLoss(** self.default_params)
        attr_loss_l2 = nn.MSELoss()
        TV_Loss=TVLoss()
        #开始步骤
        t_start=self.ddim_sampler.ddim_timesteps[-1]
        for epoch in range(params["optim_epochs"]):
            # 循环，优化
            end_latent=self.ddim_sampler.decode(  latent_start, cond, t_start, unconditional_guidance_scale=params["scale_optim"], unconditional_conditioning=un_cond,
                use_original_steps=False, callback=None)



            # 转换成图片
            image=self.latent_to_imgTensor01(end_latent)
            image_object_on_background=self.batched_tensor_mask_overlay(background_imag,image,masks_logic)
            # result_temp,_=self.object_detection.detect(image,file_path='restore.jpg',grad_status=True)
            # 目标检测模型的输出
            temp_path=os.path.join(exp_path, 'result_temp.jpg')
            result,_  =self.object_detection.detect(image_object_on_background,file_path=temp_path,grad_status=True)

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
            tv_loss=TV_Loss(image)
            print(f"tv_loss:{tv_loss}")
            optimizer.zero_grad()
            

            try:

                ( params['TV_loss_weight'] *tv_loss+params["attribution_loss_weight"]*attr_loss-loss_dict['class_loss']).backward()
            except:
                print("loss_dict['class_loss'] is None")
                          

            optimizer.step()
            # 手动清理变量，帮助回收内存
            del loss,loss_dict
            torch.cuda.empty_cache()


        # 得到最终的对抗样本    
        # end_latent=self.ddim_sampler.decode(  latent_start, cond_ref, t_start, unconditional_guidance_scale=20.0, unconditional_conditioning=un_cond,
        #         use_original_steps=False, callback=None)
        image_tensor=self.latent_to_imgTensor01(end_latent)

        image1=self.tensor_01_to_numpy_255(image_tensor)
        # 保存对抗样本
        adv_path=os.path.join(exp_path, 'adv_path.jpg')
        cv2.imwrite(adv_path,image1)




        
        
        return 



    # 中间latent 结果，
    def generate_adversarial_example_optim_control(self, background_imag=None, params=None):
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
        
       # detect model 初始化
        self.init_object_detection()
        if background_imag.ndim == 3:
            background_imag_temp=background_imag.unsqueeze(0)
        else:
            background_imag_temp=background_imag
        result_gt =self.object_detection.detect(background_imag_temp,file_path='background_detect.jpg',grad_status=True)
        
        input_point_list=self.yolo_boxes_to_corners(result_gt['boxes'])
        input_point=[input_point_list[0]]
       # 基于输入的background_imag，利用sam，得到mask，返回mask。
        # 模型初始化
        sam_predicter=init_sam(model_type=self.sam_model_type, checkpoint_path=self.sam_checkpoint_path)
        # 处理,注意mask——logic 的维度，是否是多个通道
        img_np, masks_logic_mutil, masks_tensor, scores=segment_tensor(sam_predicter, background_imag, input_point=input_point, input_label=[1],mutil_mask=True)
        # detach
        masks_tensor=masks_tensor.detach()
        
        

        visualize_sam(background_imag, masks_logic_mutil, scores)
        destroy_sam(sam_predicter)



        # control 处理

        # 填充，计算canny，返回tensor
        if masks_logic_mutil.shape[0]>1:
            # 计算掩码区域最大的索引
            mask_areas = masks_logic_mutil.sum(axis=(1, 2))  # 对 H 和 W 维度求和，得到每个通道的面积

            # 找到面积最大的通道索引
            index = np.argmax(mask_areas)
            print(f"index:{index},score:{scores[index]}")
            masks_logic=masks_logic_mutil[index]
            tensor2picture(masks_tensor[index],"masks_tensor.png")


            
        control_image=self.canny_with_mask_invert(background_imag,masks_logic)
        tensor2picture(control_image,"control_image.png") 
       
        # 初始化模型
        self.init_controlnet()
        if params["save_memory"]:
            self.model.low_vram_shift(is_diffusing=False)




        # # 预处理图像
        # ## 确保图像通道正常,对图像的size有要求，不能随便的大小
         
        # input_image=scale_tensor_to_resolution(input_image,self.default_params["image_resolution"],self.default_params["image_resolution"])
        
        # # control_image 处理，背景图像，mask，得到control，计算canny边缘。
        
        # control_image=scale_tensor_to_resolution(control_image,self.default_params["image_resolution"],self.default_params["image_resolution"])
        
        # self.control=binarize_image_tensor(control_image)
        # 添加b
        # 判断control 的维度
        if control_image.dim()==3:
            control_image=control_image.unsqueeze(0)

        




        # c_concat 草图控制；c_crossattn 跨模态控制：正向和附加的文本提示;文本内容默认用clip编码
        cond = {
            "c_concat": [control_image],
            "c_crossattn": [
                self.model.get_learned_conditioning(
                    [params["prompt"] + ', ' + params["a_prompt"]] * params["num_samples"]
                )
            ]
        }
        un_cond = {
            "c_concat": None if params["guess_mode"] else [control_image],
            "c_crossattn": [
                self.model.get_learned_conditioning(
                    [params["n_prompt"]] * params["num_samples"]
                )
            ]
        }
 
        # 判断维度  
        if background_imag.dim()==3:
            background_imag=background_imag.unsqueeze(0)
        B,C,H, W= background_imag.shape
        shape = (4, H // 8, W // 8)

        if params["save_memory"]:
            self.model.low_vram_shift(is_diffusing=True)


        self.model.control_scales = (
            [params["strength"] * (0.825 ** float(12 - i)) for i in range(13)]
            if params["guess_mode"]
            else [params["strength"]] * 13
        )  # Magic number. IDK why
        
        object_image=self.extract_mask_content(background_imag,masks_logic)
        tensor2picture(object_image,"object_image.png") 
        # 输入是-1~1
        # 将0,1 转换成-1,1
        object_image_n1_1=object_image*2-1
        latent_input = self.imgTensor_to_latent(object_image_n1_1)







        # 设置采样参数
        self.ddim_sampler.make_schedule(ddim_num_steps=params["ddim_steps"], ddim_eta=params["eta"], verbose=False)




        # 使用封装的ddim进行逆采样
        ## latent_start 表示逆采样的结果（噪声最大的latent），out 表示所有的中间结果
        with torch.no_grad():
            latent_start,out=self.ddim_sampler.encode_return_all(x0=latent_input, c=cond, t_enc=params["ddim_steps"], use_original_steps=False, return_intermediates=True,
            unconditional_guidance_scale=params["scale"], unconditional_conditioning=un_cond, callback=None)

        # 获取目标检测模型的输出，也可以直接传入这些已知的信息,用于后续计算，确保只用某个目标
        result_gt =self.object_detection.detect(object_image,file_path='result1.jpg',grad_status=True)
        





        cond_ref = {
            "c_concat": [control_image],
            "c_crossattn": [
                self.model.get_learned_conditioning(
                    ["military camouflage pattern,army green and brown color sheme"] * params["num_samples"]
                )
            ]
        }













        # require grad
        #断开，减少内存消耗
        latent_start=latent_start.detach().clone()
        latent_start.requires_grad = True
        cond_optim=cond 


        

        original_crossattn = cond["c_crossattn"][0]  # 假设是
        # 2. 克隆并开启梯度（关键步骤）
        learnable_crossattn = original_crossattn.detach().clone()  # 剥离
        learnable_crossattn.requires_grad_(True)  # 显式
        # 3. 构建可优化的条件字典
        cond_optim = {
            "c_concat": cond["c_concat"],  # 保持草图控制不变
            "c_crossattn": [learnable_crossattn]  # 使用可求导的文本编码
        }
        optimizer = torch.optim.Adam(cond_optim["c_crossattn"], lr=5e-2) 
        cross_entro_loss = YOLOv11DetectionLoss(** self.default_params)
        
        #开始步骤
        t_start=self.ddim_sampler.ddim_timesteps[-1]
        for epoch in range(params["optim_epochs"]):
            # 循环，优化
            end_latent=self.ddim_sampler.decode(  latent_start, cond_optim, t_start, unconditional_guidance_scale=8, unconditional_conditioning=un_cond,
                use_original_steps=False, callback=None)



            # 转换成图片
            image=self.latent_to_imgTensor01(end_latent)
 
 
            # 目标检测模型的输出
            result  =self.object_detection.detect(image,file_path='restore.jpg',grad_status=True)
            print(result)
            loss ,loss_dict= cross_entro_loss(result, result_gt)
            print(f"total_loss:{loss}")
            optimizer.zero_grad()
            try:
                (-loss_dict['class_loss']).backward()
            except:
                print("loss_dict['class_loss'] is None")
                break
            
            optimizer.step()
            # 手动清理变量，帮助回收内存


        # 得到最终的对抗样本    
        # end_latent=self.ddim_sampler.decode(  latent_start, cond_ref, t_start, unconditional_guidance_scale=20.0, unconditional_conditioning=un_cond,
        #         use_original_steps=False, callback=None)
        image_tensor=self.latent_to_imgTensor01(end_latent)

        image1=self.tensor_01_to_numpy_255(image_tensor)
        # RGB转换为BGR
        image1=cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
        # 保存对抗样本
        cv2.imwrite('result_adv_attack.png',image1)





        
        
        return 
    
    def generate_adversarial_example_optim_control_v2(self, background_imag=None, params=None):
        """
        生成对抗样本
        
        参数:
            control_image: 控制图像 (用于ControlNet)
            params: 覆盖默认参数的字典
            
        返回:
            生成的对抗图像列表
        """
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
        
       # detect model 初始化
        self.init_object_detection()
        if background_imag.ndim == 3:
            background_imag_temp=background_imag.unsqueeze(0)
        else:
            background_imag_temp=background_imag
        result_gt =self.object_detection.detect(background_imag_temp,file_path='background_detect.jpg',grad_status=True)
        
        input_point_list=self.yolo_boxes_to_corners(result_gt['boxes'])
        input_point=[input_point_list[0]]
       # 基于输入的background_imag，利用sam，得到mask，返回mask。
        # 模型初始化
        sam_predicter=init_sam(model_type=self.sam_model_type, checkpoint_path=self.sam_checkpoint_path)
        # 处理,注意mask——logic 的维度，是否是多个通道
        img_np, masks_logic_mutil, masks_tensor, scores=segment_tensor(sam_predicter, background_imag, input_point=input_point, input_label=[1],mutil_mask=True)
        # detach
        masks_tensor=masks_tensor.detach()
        
        

        visualize_sam(background_imag, masks_logic_mutil, scores)
        destroy_sam(sam_predicter)



        # control 处理

        # 填充，计算canny，返回tensor
        if masks_logic_mutil.shape[0]>1:
            # 计算掩码区域最大的索引
            mask_areas = masks_logic_mutil.sum(axis=(1, 2))  # 对 H 和 W 维度求和，得到每个通道的面积

            # 找到面积最大的通道索引
            index = np.argmax(mask_areas)
            print(f"index:{index},score:{scores[index]}")
            masks_logic=masks_logic_mutil[index]
            tensor2picture(masks_tensor[index],"masks_tensor.png")


            
        control_image=self.canny_with_mask_invert(background_imag,masks_logic)
        tensor2picture(control_image,"control_image.png") 
       
        # 初始化模型
        self.init_controlnet()
        if params["save_memory"]:
            self.model.low_vram_shift(is_diffusing=False)




        # # 预处理图像
        # ## 确保图像通道正常,对图像的size有要求，不能随便的大小
         
        # input_image=scale_tensor_to_resolution(input_image,self.default_params["image_resolution"],self.default_params["image_resolution"])
        
        # # control_image 处理，背景图像，mask，得到control，计算canny边缘。
        
        # control_image=scale_tensor_to_resolution(control_image,self.default_params["image_resolution"],self.default_params["image_resolution"])
        
        # self.control=binarize_image_tensor(control_image)
        # 添加b
        # 判断control 的维度
        if control_image.dim()==3:
            control_image=control_image.unsqueeze(0)

        




        # c_concat 草图控制；c_crossattn 跨模态控制：正向和附加的文本提示;文本内容默认用clip编码
        cond = {
            "c_concat": [control_image],
            "c_crossattn": [
                self.model.get_learned_conditioning(
                    [params["prompt"] + ', ' + params["a_prompt"]] * params["num_samples"]
                )
            ]
        }
        un_cond = {
            "c_concat": None if params["guess_mode"] else [control_image],
            "c_crossattn": [
                self.model.get_learned_conditioning(
                    [params["n_prompt"]] * params["num_samples"]
                )
            ]
        }
 
        # 判断维度  
        if background_imag.dim()==3:
            background_imag=background_imag.unsqueeze(0)
        B,C,H, W= background_imag.shape
        shape = (4, H // 8, W // 8)

        if params["save_memory"]:
            self.model.low_vram_shift(is_diffusing=True)


        self.model.control_scales = (
            [params["strength"] * (0.825 ** float(12 - i)) for i in range(13)]
            if params["guess_mode"]
            else [params["strength"]] * 13
        )  # Magic number. IDK why
        
        object_image=self.extract_mask_content(background_imag,masks_logic)
        tensor2picture(object_image,"object_image.png") 
        # 输入是-1~1
        # 将0,1 转换成-1,1
        object_image_n1_1=object_image*2-1
        latent_input = self.imgTensor_to_latent(object_image_n1_1)







        # 设置采样参数
        self.ddim_sampler.make_schedule(ddim_num_steps=params["ddim_steps"], ddim_eta=params["eta"], verbose=False)




        # 使用封装的ddim进行逆采样
        ## latent_start 表示逆采样的结果（噪声最大的latent），out 表示所有的中间结果
        with torch.no_grad():
            latent_start,out=self.ddim_sampler.encode_return_all(x0=latent_input, c=cond, t_enc=params["ddim_steps"], use_original_steps=False, return_intermediates=True,
            unconditional_guidance_scale=params["scale"], unconditional_conditioning=un_cond, callback=None)

        # 获取目标检测模型的输出，也可以直接传入这些已知的信息,用于后续计算，确保只用某个目标
        result_gt =self.object_detection.detect(object_image,file_path='result1.jpg',grad_status=True)
        





        cond_ref = {
            "c_concat": [control_image],
            "c_crossattn": [
                self.model.get_learned_conditioning(
                    ["military camouflage pattern,army green and brown color sheme"] * params["num_samples"]
                )
            ]
        }













        


        







        # require grad
        #断开，减少内存消耗
        
        original_crossattn = cond["c_crossattn"][0]  # 假设是
        # 2. 克隆并开启梯度（关键步骤）
        learnable_crossattn = original_crossattn.detach().clone()  # 剥离
        learnable_crossattn.requires_grad_(True)  # 显式
        # 3. 构建可优化的条件字典
        cond_optim = {
            "c_concat": cond["c_concat"],  # 保持草图控制不变
            "c_crossattn": [learnable_crossattn]  # 使用可求导的文本编码
        }
        optimizer = torch.optim.Adam(cond_optim["c_crossattn"], lr=5e-2) 
        cross_entro_loss = YOLOv11DetectionLoss(** self.default_params)
        shape = (4, H // 8, W // 8)

        for epoch in range(params["optim_epochs"]):
            # 循环，优化



            samples, intermediates = self.ddim_sampler.sample_without_grad(params['ddim_steps'], params['num_samples'],
                                                     shape, cond_optim, verbose=False, eta=params['eta'],
                                                     unconditional_guidance_scale=params['scale'],
                                                     unconditional_conditioning=un_cond)
            x_samples = self.model.decode_first_stage(samples)
        # 维度变换
            x_samples1 = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            img2=cv2.cvtColor(x_samples1[0], cv2.COLOR_RGB2BGR)
            cv2.imwrite('result_sample.png',img2)
            # 转换成图片
            image=self.latent_to_imgTensor01(samples)
 
 
            # 目标检测模型的输出
            result  =self.object_detection.detect(image,file_path='restore.jpg',grad_status=True)
            print(result)
            loss ,loss_dict= cross_entro_loss(result, result_gt)
            print(f"total_loss:{loss}")
            optimizer.zero_grad()
            (-loss_dict['class_loss']).backward()
            # try:
            #     (-loss_dict['class_loss']).backward()
            # except:
            #     print("loss_dict['class_loss'] is None")
            #     break
            
            optimizer.step()
            # 手动清理变量，帮助回收内存


        # 得到最终的对抗样本    
        # end_latent=self.ddim_sampler.decode(  latent_start, cond_ref, t_start, unconditional_guidance_scale=20.0, unconditional_conditioning=un_cond,
        #         use_original_steps=False, callback=None)


        image1=self.tensor_01_to_numpy_255(image)
        # RGB to BGR
        image1=cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
        # 保存对抗样本
        cv2.imwrite('result_adv_attack.png',image1)





        
        
        return 
    

    def generate_adversarial_test(self, input_image=None,control_image=None, params=None):
        """
        生成对抗样本
        
        参数:
            control_image: 控制图像 (用于ControlNet)
            params: 覆盖默认参数的字典
            
        返回:
            生成的对抗图像列表
        """
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
        
       
       
       
        
        if params["save_memory"]:
            self.model.low_vram_shift(is_diffusing=False)




        # 预处理图像
        ## 确保图像通道正常,对图像的size有要求，不能随便的大小
        # 确保是三个通道的numpy，int8类型  
        input_image=HWC3(input_image) ## input_image RGB
        input_image = cv2.resize(input_image, (self.default_params["image_resolution"],self.default_params["image_resolution"])) # input_image RGB

        if control_image is not None:
            control_image=HWC3(control_image) ## control_image RGB
            control_image = cv2.resize(control_image, (self.default_params["image_resolution"],self.default_params["image_resolution"])) # control_image RGB
            self.edge_image,self.control=self.generate_edge_control_from_image(control_image)
        else:
            ## 处理控制图像，并返回边缘图和边缘的control
            self.edge_image,self.control=self.generate_edge_control_from_image(input_image,'./exp/canny_edge.jpg')



        # c_concat 草图控制；c_crossattn 跨模态控制：正向和附加的文本提示;文本内容默认用clip编码
        cond = {
            "c_concat": [self.control],
            "c_crossattn": [
                self.model.get_learned_conditioning(
                    [params["prompt"] + ', ' + params["a_prompt"]] * params["num_samples"]
                )
            ]
        }
        un_cond = {
            "c_concat": None if params["guess_mode"] else [self.control],
            "c_crossattn": [
                self.model.get_learned_conditioning(
                    [params["n_prompt"]] * params["num_samples"]
                )
            ]
        }
 
 
        H, W, C = input_image.shape
        shape = (4, H // 8, W // 8)

        if params["save_memory"]:
            self.model.low_vram_shift(is_diffusing=True)


        self.model.control_scales = (
            [params["strength"] * (0.825 ** float(12 - i)) for i in range(13)]
            if params["guess_mode"]
            else [params["strength"]] * 13
        )  # Magic number. IDK why
        
        #对图像进行编码，转换为latent
        temp_tensor=self.to_imgTensor_from_numpy_int8(input_image)
        latent_input=self.imgTensor_to_latent(temp_tensor)
        ref_image_tensor=temp_tensor.clone()
        # 设置采样参数
        self.ddim_sampler.make_schedule(ddim_num_steps=params["ddim_steps"], ddim_eta=params["eta"], verbose=False)
        # 使用封装的ddim进行逆采样
        ## latent_start 表示逆采样的结果（噪声最大的latent），out 表示所有的中间结果
        with torch.no_grad():
            latent_start,out=self.ddim_sampler.encode_return_all(x0=latent_input, c=cond, t_enc=params["ddim_steps"], use_original_steps=False, return_intermediates=True,
            unconditional_guidance_scale=params["scale"], unconditional_conditioning=un_cond, callback=None)

        # 获取目标检测模型的输出，也可以直接传入这些已知的信息
        result_gt =self.object_detection.detect(input_image,file_path='result1.jpg',grad_status=True)
        





        cond_ref = {
            "c_concat": [self.control],
            "c_crossattn": [
                self.model.get_learned_conditioning(
                    ["military camouflage pattern,army green and brown color sheme"] * params["num_samples"]
                )
            ]
        }












        # require grad
        #断开，减少内存消耗
        latent_start=latent_start.detach().clone()
        latent_start.requires_grad = True
        cond_optim=cond 


        

        original_crossattn = cond["c_crossattn"][0]  # 假设是
        # 2. 克隆并开启梯度（关键步骤）
        learnable_crossattn = original_crossattn.detach().clone()  # 剥离
        learnable_crossattn.requires_grad_(True)  # 显式
        # 3. 构建可优化的条件字典
        cond_optim = {
            "c_concat": cond["c_concat"],  # 保持草图控制不变
            "c_crossattn": [learnable_crossattn]  # 使用可求导的文本编码
        }
        optimizer = torch.optim.Adam(cond_optim["c_crossattn"], lr=5e-2) 
        cross_entro_loss = YOLOv11DetectionLoss(** self.default_params)
        
        #开始步骤
        t_start=self.ddim_sampler.ddim_timesteps[-1]
        for epoch in range(params["optim_epochs"]):
            # 循环，优化
            end_latent=self.ddim_sampler.decode(  latent_start, cond_optim, t_start, unconditional_guidance_scale=8, unconditional_conditioning=un_cond,
                use_original_steps=False, callback=None)



            # 转换成图片
            image=self.latent_to_imgTensor01(end_latent)
 
 
            # 目标检测模型的输出
            result  =self.object_detection.detect(image,file_path='restore.jpg',grad_status=True)
            print(result)
            loss ,loss_dict= cross_entro_loss(result, result_gt)
            print(f"total_loss:{loss}")
            optimizer.zero_grad()
            try:
                (-loss_dict['class_loss']).backward()
            except:
                print("loss_dict['class_loss'] is None")
                break
            
            optimizer.step()
            # 手动清理变量，帮助回收内存


        # 得到最终的对抗样本    
        # end_latent=self.ddim_sampler.decode(  latent_start, cond_ref, t_start, unconditional_guidance_scale=20.0, unconditional_conditioning=un_cond,
        #         use_original_steps=False, callback=None)
        image_tensor=self.latent_to_imgTensor01(end_latent)

        image1=self.tensor_01_to_numpy_255(image_tensor)
        # 保存对抗样本
        cv2.imwrite('result_adv_attack.png',image1)





        
        
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


    def yolo_boxes_to_corners(self,boxes):
        """
        将 YOLO 检测框转换为 [[x1,y1], [x2,y2]] 格式的列表
        
        参数：
            boxes: YOLO 输出的检测框，形状为 (N, 4)，每个框为 [x_center, y_center, width, height]（归一化坐标）
        
        返回：
            corners_list: 列表，每个元素为 [[x1, y1], [x2, y2]]（绝对坐标）
        """
        corners_list = []
        # 遍历每个检测框
        for box in boxes:
            x1,y1,x2,y2 = box[0]  # 提取 YOLO 格式的框
            
            x_center,y_center=(x1+x2)/2,(y1+y2)/2
            
            # 转为整数（可选，根据需求保留小数或取整）
            x_center,y_center= map(int, [x_center,y_center])
            
            # 添加到结果列表
            corners_list.append([ x_center,y_center])
        
        return corners_list


    def canny_with_mask_invert(self,background_imag, masks, canny_low=0, canny_high=100):
        """
        对 tensor 图像计算 Canny 边缘，mask 以外区域置 0，最终边缘处为 0、无边缘处为 1
        
        参数：
            background_imag: 输入图像 tensor（BCHW 或 CHW 格式，0-1 范围）
            masks: 分割掩码 array（shape: (num_masks, H, W)，元素为 True/False）
            mask_idx: 选择使用哪个掩码（默认第 1 个，索引 0）
            canny_low: Canny 低阈值（默认 100）
            canny_high: Canny 高阈值（默认 200）
        
        返回：
            final_result: 最终结果（numpy 数组，HWC 格式，边缘=0，无边缘=1）
            final_tensor: 结果转为 tensor（BCHW 格式，0-1 范围，与输入维度匹配）
        """
        # 1. 处理输入图像 tensor → HWC 格式 numpy 数组（0-255 整数）
        if background_imag.dim() > 3:
            background_imag = background_imag.squeeze(0)  # BCHW → CHW
        img_np = background_imag.permute(1, 2, 0).cpu().numpy()  # CHW → HWC
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        
        # 2. 处理掩码（True→255，False→0）
        selected_mask =(masks.astype(np.uint8) * 255)# (H, W) 二值掩码
        assert img_np.shape[:2]==masks.shape ,"图像与 mask 尺寸不匹配"
        
        # 3. Canny 边缘检测（单通道）
        gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        canny_edges = cv2.Canny(gray_img, canny_low, canny_high)  # 边缘=255，背景=0
        
        #保存cannny
        # cv2.imwrite("canny.png",canny_edges)







        # 4. 掩码过滤：仅保留 mask 内的边缘
        canny_masked = cv2.bitwise_and(canny_edges, selected_mask)  # mask 外→0，mask 内边缘→255、背景→0
        # 保存canny_masked
        # cv2.imwrite("canny_masked.png",canny_masked)
        # 5. 关键：像素值反转（边缘255→0，背景0→255）
        inverted_canny = cv2.bitwise_not(canny_masked)  # 反转后：边缘=0，无边缘=255
        # cv2.imwrite("inverted_canny.png",inverted_canny)
   
        
        # 7. 扩展为 3 通道（与输入图像格式对齐）
        final_result = cv2.cvtColor((inverted_canny).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        # cv2.imwrite("final_result.png",final_result)
        final_result = final_result / 255.0  # 转回 0-1 范围的 float 数组
        
        # 8. 转为 tensor 格式（BCHW）
        final_tensor = torch.from_numpy(final_result).permute(2, 0, 1) 
        final_tensor = final_tensor.float()
        
        return final_tensor

                  
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
        
        # 2. 处理二维 mask，转为 tensor 并扩展维度以匹配 BCHW
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).bool()  # numpy 转 bool tensor
        else:
            mask = mask.bool()  # 确保是 bool 类型
        
        # 扩展 mask 维度：(H, W) → (1, 1, H, W)，再通过广播匹配 (B, C, H, W)
        mask = mask.unsqueeze(0).unsqueeze(0)  # 增加 batch 和 channel 维度
        mask = mask.to(input_tensor.device)  # 确保与输入 tensor 同设备
        
        # 3. 生成填充值 tensor（与输入同形状）
        fill_tensor = torch.full_like(input_tensor, fill_value=mask_value)
        
        # 4. 核心操作：mask 内保留原图，mask 外填充
        result_tensor = torch.where(mask, input_tensor, fill_tensor)
        
        # 5. 恢复原始维度（若输入是 CHW，去除 batch 维度）
        if original_dim == 3:
            result_tensor = result_tensor.squeeze(0)
        
        return result_tensor





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
        assert background_tensor.dtype == image_tensor.dtype, "背景和前景张量数据类型必须相同"
        
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



# def tensor2picture(tensor, save_path, data_range="auto", use_opencv=False):
#     """
#     将 Tensor 保存为图像文件
    
#     参数:
#         tensor: PyTorch Tensor 或 TensorFlow Tensor
#             形状要求: 
#                 - PyTorch: (B, C, H, W) 或 (C, H, W)（B为批量，C为通道）
#                 - TensorFlow: (B, H, W, C) 或 (H, W, C)
#         save_path: str
#             图像保存路径（如 "output.png"）
#         data_range: str 或 tuple, 可选
#             输入张量的数据范围，默认为 "auto"（自动检测）：
#             - "auto": 自动将张量归一化到 [0, 255]
#             - (min_val, max_val): 手动指定范围，将其映射到 [0, 255]
#         use_opencv: bool, 可选
#             是否用 OpenCV 保存（默认用 PIL），OpenCV 会自动转换 RGB→BGR
#     """
#     # --------------------------
#     # 1. 处理 Tensor 维度（移除批量维度）
#     # --------------------------
#     if "torch" in str(type(tensor)).lower():  # PyTorch Tensor
#         # 移至 CPU 并转为 numpy
#         tensor = tensor.cpu().detach() if tensor.requires_grad else tensor.cpu()
#         img_np = tensor.numpy()
        
#         # 移除批量维度（若有）
#         if img_np.ndim == 4:  # (B, C, H, W) → (C, H, W)
#             img_np = img_np.squeeze(0)
        
#         # 调整通道顺序：(C, H, W) → (H, W, C)
#         if img_np.shape[0] in [1, 3]:  # 单通道/三通道
#             img_np = np.transpose(img_np, (1, 2, 0))
    
#     elif "tensorflow" in str(type(tensor)).lower():  # TensorFlow Tensor
#         # 转为 numpy（TF 默认在 CPU，无需手动转移）
#         img_np = tensor.numpy()
        
#         # 移除批量维度（若有）
#         if img_np.ndim == 4:  # (B, H, W, C) → (H, W, C)
#             img_np = img_np.squeeze(0)
    
#     else:
#         raise TypeError("不支持的 tensor 类型，请使用 PyTorch 或 TensorFlow 张量")
    
#     # --------------------------
#     # 2. 处理单通道图像（灰度图）
#     # --------------------------
#     if img_np.shape[-1] == 1:
#         img_np = img_np.squeeze(-1)  # 移除通道维度，变为 (H, W)
    
#     # --------------------------
#     # 3. 数据范围映射到 [0, 255]
#     # --------------------------
#     if data_range == "auto":
#         min_val = img_np.min()
#         max_val = img_np.max()
#     else:
#         min_val, max_val = data_range
    
#     # 防止除零（若所有值相同）
#     if max_val == min_val:
#         img_np = np.zeros_like(img_np, dtype=np.uint8)
#     else:
#         # 归一化到 [0, 1] 再映射到 [0, 255]
#         img_np = ((img_np - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
#     # --------------------------
#     # 4. 保存图像
#     # --------------------------
#     if use_opencv:
#         # OpenCV 保存 BGR 格式，若原是 RGB 需转换
#         if img_np.ndim == 3 and img_np.shape[-1] == 3:
#             img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
#         cv2.imwrite(save_path, img_np)
#     else:
#         # PIL 直接保存 RGB 格式
#         img = Image.fromarray(img_np)
#         img.save(save_path)


# def cv2_to_tensor(img: np.ndarray, normalize: bool = True) -> torch.Tensor:
#     """
#     将OpenCV读取的图像（H×W×C，BGR格式，uint8）转换为C×H×W格式的Tensor
    
#     参数:
#         img: OpenCV读取的图像数组，形状为(H, W, C)，通道顺序为BGR，数据类型为uint8
#         normalize: 是否将像素值归一化到[0.0, 1.0]（默认True）
    
#     返回:
#         tensor: 转换后的Tensor，形状为(C, H, W)，数据类型为float32
#                 若输入为彩色图，通道顺序转为RGB；若为灰度图，保持单通道
#     """
#     # 检查输入是否为合法的图像数组
#     if not isinstance(img, np.ndarray) or img.ndim not in (2, 3):
#         raise ValueError("输入必须是OpenCV读取的2D（灰度图）或3D（彩色图）数组")
    
#     # 处理彩色图（3通道）：BGR转RGB
#     if img.ndim == 3:
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     else:
#         # 灰度图（2通道）：保持不变，后续扩展为单通道
#         img_rgb = img
    
#     # 转换为float32类型（避免uint8计算溢出）
#     img_float = img_rgb.astype(np.float32)
    
#     # 归一化到[0.0, 1.0]（如果需要）
#     if normalize:
#         img_float /= 255.0
    
#     # 维度重排：H×W×C → C×H×W
#     # 灰度图会从(H, W)变为(1, H, W)
#     tensor = torch.from_numpy(img_float).permute(2, 0, 1) if img.ndim == 3 else torch.from_numpy(img_float).unsqueeze(0)
    
#     return tensor


# def scale_tensor_to_resolution(tensor, new_height, new_width, mode='bilinear'):
#     """
#     将图像Tensor（C×H×W）缩放到指定分辨率（new_height × new_width）
    
#     参数:
#         tensor: 输入图像Tensor，形状为 (C, H, W)，数据类型为float32/float64
#         new_height: 目标高度（整数）
#         new_width: 目标宽度（整数）
#         mode: 插值方法，可选 'bilinear'（默认，双线性）、'nearest'（最近邻）、'bicubic'（双三次）
    
#     返回:
#         scaled_tensor: 缩放后的Tensor，形状为 (C, new_height, new_width)
#     """
#     # 检查输入合法性
#     if tensor.dim() != 3:
#         raise ValueError(f"输入Tensor必须是3维 (C, H, W)，但得到 {tensor.dim()} 维")
#     if not isinstance(new_height, int) or not isinstance(new_width, int):
#         raise ValueError(f"目标分辨率（new_height, new_width）必须是整数，但得到 ({new_height}, {new_width})")
#     if new_height <= 0 or new_width <= 0:
#         raise ValueError(f"目标分辨率必须为正数， but得到 ({new_height}, {new_width})")
    
#     # 增加batch维度（B=1），因为F.interpolate要求输入为4维 (B, C, H, W)
#     tensor_with_batch = tensor.unsqueeze(0)  # 形状变为 (1, C, H, W)
    
#     # 执行缩放：按目标分辨率（new_height, new_width）插值
#     scaled = F.interpolate(
#         input=tensor_with_batch,
#         size=(new_height, new_width),  # 明确指定目标尺寸
#         mode=mode,
#         align_corners=False  # 默认为False，避免边缘扭曲
#     )
    
#     # 去除batch维度，返回 (C, new_height, new_width)
#     return scaled.squeeze(0)




# def binarize_image_tensor(img, threshold=0.5):
#     """
#     对输入的张量图像（0-1范围）进行二值化，生成0.0/1.0的浮点型掩码（与输入形状一致，含三通道）。
#     逻辑：对每个像素取所有通道的最小值，小于阈值则为1.0，否则为0.0，最终扩展为与输入相同的通道数。
    
#     参数：
#         img: 输入张量，形状支持 (B, C, H, W)、(C, H, W) 或 (H, W)，值范围 [0,1]
#         threshold: 阈值（默认0.5）
    
#     返回：
#         mask: 二值化掩码，与输入形状相同（含通道数），值为0.0或1.0（浮点型）
#     """
#     # 确保输入是PyTorch张量
#     if not isinstance(img, torch.Tensor):
#         raise TypeError("输入必须是PyTorch张量")
    
#     # 确定通道维度和输入通道数
#     if img.dim() == 4:  # (B, C, H, W)
#         channel_dim = 1
#         num_channels = img.size(channel_dim)  # 获取通道数 C
#     elif img.dim() == 3:  # (C, H, W)
#         channel_dim = 0
#         num_channels = img.size(channel_dim)  # 获取通道数 C
#     elif img.dim() == 2:  # (H, W) 单通道输入，默认输出3通道
#         channel_dim = -1
#         num_channels = 3  # 手动指定为3通道
#     else:
#         raise ValueError("输入张量维度必须为2、3或4")
    
#     # 处理多通道：取每个像素的通道最小值
#     if channel_dim != -1:
#         img_min = torch.min(img, dim=channel_dim, keepdim=True)[0]  # 单通道 (B,1,H,W) 或 (1,H,W)
#     else:  # 单通道输入，直接增加通道维度
#         img_min = img.unsqueeze(0)  # (1, H, W)
    
#     # 二值化：像素最小值 < 阈值 → 1.0，否则 → 0.0
#     mask_single = (img_min < threshold).float()  # 单通道掩码
    
#     # 将单通道掩码复制为与输入相同的通道数（通常为3通道）
#     # 用repeat在通道维度复制，其他维度复制1次（保持不变）
#     if img.dim() == 4:
#         # 输入 (B,C,H,W) → 输出 (B,C,H,W)：在通道维度（dim=1）复制C次
#         mask = mask_single.repeat(1, num_channels, 1, 1)
#     elif img.dim() == 3:
#         # 输入 (C,H,W) → 输出 (C,H,W)：在通道维度（dim=0）复制C次
#         mask = mask_single.repeat(num_channels, 1, 1)
#     else:  # 输入 (H,W) → 输出 (3,H,W)
#         mask = mask_single.repeat(num_channels, 1, 1)  # 复制为3通道
    
#     return mask




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
    
    attack.generate_adversarial_main(img,exp_path=r"./exp/test")
    # attack.generate_adversarial_example_optim_control_v2(img)
    # attack.generate_adversarial_example(img,control_img)
    # attack.generate_adversarial_example_optim_control_v2(img,control_img)
    # attack.generate_adversarial_example_optim_control(img,control_img)