# 一、使用说明
## 0. git失败

> * 在windows下，经常会出现git上传仓库，或者git clone 失败的情况。可以按照如下教程进行解决： https://cloud.tencent.com/developer/article/2527142
> * 该教程 大致步骤为：
>     * 1、在windows系统设置代理 
>     * 2、使用git 设置代理。（注意：此代理不会影响代理软件等的正常使用）
## 1. 环境配置
### 1.1 安装依赖


* pip install -r requirements.txt
    *  下载与cuda toolkit版本匹配的pytorch，根据自己的cuda toolkit版本下载对应的pytorch安装包，并安装。建议使用国内镜像源：
        pip3 install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cu124
        
        比如上述命令，后缀cu124  表示版本（如这里表示12.4版本），根据cuda toolkit版本选择，需对应好。
* 安装必要的包，按照environment.yaml 的版本按照，如果显示加载权重的时候，出现key 对不上等问题，则是版本未完全按照要求安装。如果为完全按照版本安装，则也可以按如下修改：

    * 1.1.1 No module named 'pytorch_lightning.utilities.distributed'
        问题原因：pytorch_lightning版本过高，部分函数已经更换过，故会报错
        解决方案：将老版的接口函数，按照新的接口函数修改
        如rom pytorch_lightning.utilities.distributed import rank_zero_only ——> from pytorch_lightning.utilities.rank_zero import rank_zero_only
    * 1.1.2 No module named 'pytorch_lightning.metrics.functional'
### 1.2 环境问题
ldm\modules\diffusionmodules\model.py 下面有个xformers 不能正确安装（windows系统，只有cpu版本的torch才能安装），
### 1.3 下载预训练模型

* 下载预训练模型（所需要的sd模型），并放到./models/ 文件夹下
* 模型初始化会加载标准的sd模型，以及vae等，上述模型在初始化的时候会直接从huggingface官网下载，国内可能会出现报错。以下是两种解决方法：
    * 1、直接从官网或者其他途径下载模型，并放到huggingface的缓存文件夹下，也可以直接设置缓存文件夹的路径，或者是加载模型时，改成路径。
    * 2、利用国内的镜像，本项目目前使用本方法，更为方便。
        * 2.1  安装依赖：pip install -U huggingface_hub
        * 2.2  设置环境变量：
            * Windows：$env:HF_ENDPOINT = "https://hf-mirror.com"


    
                <span style="color:red">
                在windows 系统下，需要在vscode的终端 运行上述命令，不然不起作用。
                此情况下，如果有代理，需关闭代理
            
            
            
                </span>
        
            * linux： export HF_ENDPOINT=https://hf-mirror.com


## 2. 训练模型
```
```







# 二、工程文件说明
## 类等的关系说明
* ControlLDM 类，继承自LatentDiffusion类
* ControlLDM 类 包含ControlNet 模型，以及其他辅助功能
* ControlNet 类，继承自nn.Module
* DDIMSampler 类，封装 ControlLDM类（输入ControlLDM 类），可以调用类函数，实现采样，编码，解码等功能
* LatentDiffusion类，继承自DDPM 类，并给出了一些辅助函数。
* DDPM类 继承自pl.LightningModule

## cldm\cldm.py
controlnet的模型结构定义，包括解码latent ；clip 的文本编码器；变分自编码器等
ControlLDM 类，继承自LatentDiffusion，整个controlnet 模型调用的主要类，包含预测噪声，编码能能力;里面包含ControlNet模型，该类主要功能是预测噪声，还包含clip等文本编码器




### ControlLDM
* get_input : 过父类方法获取图像的 latent 表示 x 和条件输入 c（如文本提示）。将条件输入和控制信号分别封装到 c_crossattn 和 c_concat 中，供模型后续使用。
* apply_model： 利用模型生成噪声
* get_unconditional_conditioning： 获得无条件控制信号的编码向量
* log_images： 可视化噪声、中间的latents等内容（调用下面的函数，可视化扩散过程）
* sample_log 方法：执行扩散采样过程
* configure_optimizers 方法：配置模型优化器
* low_vram_shift 方法：低显存优化策略


### ControlNet
* 就是一个controlnet 模型的定义，包含模型定义，forward 方法。

## ldm\util.py
* log_txt_as_img ：将文本转换为图像张量
* ismap 函数：判断张量是否为特征图
* isimage 函数：判断张量是否为图像


## ldm\models\diffusion\ddim.py
实现了ddim算法，即diffusion 扩散过程的核心算法。
对 所需要的模型 提供了一个DDIM 方法的类。
* 定义了DDIMSampler 类，实现了DDIM 算法，包括预处理，采样，latent 编码，解码等。与下面函数一致。
### 使用方法
输入所需要的模型， 返回一个类，可以实现 预处理，DDIM 采样，encoder，decoder等功能。
在此工程中，主要是对ControlLDM 类进行封装，输入此类，实现ControlNet模型的DDIM 采样。

## cldm\ddim_hacked.py

实现了ddim算法，即diffusion 扩散过程的核心算法。
对 所需要的模型 提供了一个DDIM 方法的类。

* 定义了DDIMSampler 类，实现了DDIM 算法，包括预处理，采样，latent 编码，解码等。与上面函数一致。原始代码默认使用此类，而非上面函数的同名的类

### DDIMSampler 类
* register_buffer：作用是将一个属性（通常是张量）注册到类的实例中，并确保该属性存储在 GPU 上
* make_schedule 方法：其核心功能是为 去噪扩散隐式模型（DDIM） 配置采样所需的时间步和参数，比如$\alpha$，$\beta$等参数
>> 调用了make_ddim_sampling_parameters, make_ddim_timesteps （ldm.modules.diffusionmodules.util ），生成各种参数。该方法，实际上就是调用了两个函数，生成参数，并注册到类的实例中。
* sample： 采样的程序入口，对$x_0$，以及输入参数进行预处理，再调用 方法，实现采样过程。它是实现扩散模型采样的核心接口。该方法支持多种采样配置，包括条件生成、去噪控制、引导采样等功能。
* ddim_sampling：  代码实现了 DDIM（Denoising Diffusion Implicit Models）采样算法的核心逻辑。
* p_sample_ddim：  DDIM 采样器中单步去噪的核心方法，负责从当前时间步的样本 x_t 计算上一时间步的样本 x_{t-1}

* encode： 其功能是将原始数据（如输入图像 x0）通过正向扩散过程逐步编码为指定时间步的噪声样本，同时支持条件引导和中间结果跟踪。
* stochastic_encode：随机编码（Stochastic Encoding） 功能，其作用是将原始数据（如图片）快速转换为特定时间步的噪声表示。与之前的 encode 方法不同，随机编码不依赖模型预测的噪声，而是直接基于预定义的噪声调度参数，通过添加随机噪声实现编码。
>> 该方法与之前的方法区别在于 噪声的生成，本方法是随机的，之前的方法是基于噪声预测器生成的
* decode： 解码器，将初始噪声样本 $x_T$ 逐步解码$x_0$

## ldm\models\diffusion\ddpm.py
### DDPM 类


### LatentDiffusion 类


