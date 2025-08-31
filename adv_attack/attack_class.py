
from share import *


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
from object_detection_class import ObjectDetection




class ADV_ATTACK:
    def __init__(self, config_path:str='./models/cldm_v15.yaml',
                  model_path:str='./models/control_sd15_scribble.pth', 
                  device:torch.device=torch.device("cuda"),
                  model_type:str='yolov11',
                  class_names:list=[],
                  model_path_object_detection:str=None,
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
        # 加载模型配置
        self.config_path = config_path
        self.model_path = model_path
        self.device = device
        
        # 初始化模型
        self.model = create_model(self.config_path).cpu()
        self.model.load_state_dict(load_state_dict(self.model_path, location='cuda'),strict=False)
        self.ddim_sampler = DDIMSampler(self.model)
        
  
        # 加载检测模型
        self.object_detection=ObjectDetection(model_type,
                                              model_path=model_path_object_detection,
                                              class_names=class_names,
                                              device=torch.device("cpu"))
        
        # 默认参数配置
        if kwargs:
            self.default_params = kwargs
        else:
            self.default_params = {
            "prompt": "a handsome boy with a long hair",
            "a_prompt": "",
            "n_prompt": "",
            "num_samples": 1,
            "image_resolution": 256,
            "ddim_steps": 5,
            "guess_mode": False,
            "strength": 1.0,
            "scale": 9.0,
            "seed": 42,
            "eta": 0.0,
            "save_memory": True,
            "optim_epochs":10
        }
    



    def set_params(self, **kwargs):
        """更新攻击参数"""
        for key, value in kwargs.items():
            if key in self.default_params:
                self.default_params[key] = value
            else:
                print(f"警告: 参数 {key} 不是有效参数，将被忽略")
    
    def generate_adversarial_example(self, input_image=None, params=None):
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

        ## 处理控制图像，并返回边缘图和边缘的control
        self.edge_image,self.control=self.generate_edge_control_from_image(input_image)



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
        temp_tensor=self.to_imgTensor_from_image(input_image)
        latent_input=self.imgTensor_to_latent(temp_tensor)

        # 设置采样参数
        self.ddim_sampler.make_schedule(ddim_num_steps=params["ddim_steps"], ddim_eta=params["eta"], verbose=False)
        # 使用封装的ddim进行逆采样
        ## latent_start 表示逆采样的结果（噪声最大的latent），out 表示所有的中间结果
        with torch.no_grad():
            latent_start,out=self.ddim_sampler.encode_return_all(x0=latent_input, c=cond, t_enc=params["ddim_steps"], use_original_steps=False, return_intermediates=True,
            unconditional_guidance_scale=params["scale"], unconditional_conditioning=un_cond, callback=None)

        # 获取目标检测模型的输出，也可以直接传入这些已知的信息
        result_gt =self.object_detection.detect(input_image,file_path='result1.jpg',grad_status=True)
        
        #         'box': (x1, y1, x2, y2),  # 原始图像坐标，张量形式
        # 'confidence': confidences[i],  # 张量形式的置信度 
        # 'class_id': classes[i],  # 张量形式的类别ID
        # 'class_name': self.names[int(classes[i].item())]  # 类别名称
        


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
        optimizer = torch.optim.Adam([latent_start], lr=1e-3)
        cross_entro_loss = torch.nn.CrossEntropyLoss()
  
        #开始步骤
        t_start=self.ddim_sampler.ddim_timesteps[-1]
        for epoch in range(params["optim_epochs"]):
            # 循环，优化
            end_latent=self.ddim_sampler.decode(  latent_start, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
                use_original_steps=False, callback=None)



            # 转换成图片
            image=self.latent_to_img(end_latent)

            print("image.requires_grad：", image.requires_grad)  
            print("image.grad_fn：", image.grad_fn)   
            # 目标检测模型的输出
            result  =self.object_detection.detect(image,file_path='restore.jpg',grad_status=True)
            result
            # loss = cross_entro_loss(confidences, confidences_gt)
            # optimizer.zero_grad()
            # (-loss).backward()
            # optimizer.step()





        
        # 采样过程实现
        # t_start=self.ddim_sampler.ddim_timesteps[-1]
        # # 得到最终的latent
        # end_latent=self.ddim_sampler.decode(  latent_start, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
        #        use_original_steps=False, callback=None)




        # # # 调试
        # # temp=out['intermediates'][0]
        # image_temp=self.latent_to_img(end_latent)
        
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
            





        #     # image=self.latent_to_img(x_next)
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
        # image=self.latent_to_img(x_next)
        
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
        # image=self.latent_to_img(out["intermediates"][0])

        
        # bbox_xyxy, confidences, class_ids  =self.object_detection.detect(image)
        
        # # 将所有的latent转换成图片
        # path="./exp/result"
        # ## 判断路径是否存在，不存在则创建
        # if not os.path.exists(path):
        #     os.mkdir(path)
        
        # for i in range(len(out["intermediates"])):
        #     image=self.latent_to_img(out["intermediates"][i])

        #     cv2.imwrite("{}/{}.png".format("./exp/result", i), image[0])
        # images = self.latent_to_img(latent)
        
        # img1=cv2.cvtColor(images[0], cv2.COLOR_RGB2BGR)
        
        # cv2.imwrite('result1.png',images[0])
        
        return 
    def to_imgTensor_from_image(self, image):


        # 转换到-1到1
        image = image.astype(np.float32) / 127.5 - 1.0

        tensor = torch.from_numpy(image).float()

        # 4. 调整维度顺序：从[H, W, C]转换为[C, H, W]（PyTorch要求的通道优先格式）
        tensor = tensor.permute(2, 0, 1)
        
        return tensor.unsqueeze(0)
    

    def generate_edge_control_from_image(self, image):
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
        # detected_map = np.zeros_like(image, dtype=np.uint8)
        # detected_map[np.min(image, axis=2) < 127] = 255

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(self.default_params["num_samples"])], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        return detected_map,control
    
    # 将图片转化为latent
    def imgTensor_to_latent(self, img):
        '''
        img:[0-1],type:tensor
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
        z=z.to(self.device)
        return z   
    # def latent_to_img(self,latent):
    #     img = self.model.first_stage_model.decode(latent)
    #     return  (einops.rearrange(img, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

    def latent_to_img(self,latent):
        img = self.model.first_stage_model.decode(latent)
        # sd 对应的区间为-1到1，需要转换到0到1
        img = ((img + 1)*0.5 ).to(dtype=torch.float32)
    
        # # 确保与YOLO模型在同一设备
        # img = img.to(self.yolo_model.device)  # 假设self.yolo_model是加载的YOLO模型
        
        return img
        # return  (einops.rearrange(img, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)















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
    # 定义提示词
    prompt = "a handsome boy with a long hair "
    a_prompt = ""
    n_prompt = ""

    # 设置参数
    num_samples = 1               # 生成图像数量
    image_resolution = 512        # 图像分辨率
    ddim_steps = 50               # 采样步数
    guess_mode = False            # 是否使用猜测模式
    strength = 1.0                # 控制生成与输入的相似度
    scale = 9.0                   # 引导系数
    seed = 42                     # 随机种子（用于结果可复现）
    eta = 0.0                     # DDIM采样器的eta参数
    path=os.path.join(os.path.dirname(__file__), "..")
    img_path=os.path.join(os.path.join(path,'test_imgs'),'old.png')
    img=cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    model_path=os.path.join(os.path.join(path,'models'),'yolo11n.pt')
    attack = ADV_ATTACK(device=torch.device("cuda"),model_path_object_detection=model_path)
    #默认输入是RBG格式
    attack.generate_adversarial_example(img)