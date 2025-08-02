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
## 模型初始化流程
模型直接导入ControlLDM类，该类的初始化方法继承自 LatentDiffusion类，LatentDiffusion类的初始化方法又继承自DDPM类，DDPM类的初始化方法又继承自pl.LightningModule类。 DDPM类初始化方法中，定义了模型，模型为同文件的DiffusionWrapper 类；latentDiffusion类定义了其他内容。模型文本编码为向量等 都在ldm\models\diffusion\ddpm.py文件的类初始化，较为复杂。

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


### 模型关系

* ControlLDM 初始化
    * 继承父类LatentDiffusion 的初始化方法，进行初始化，包括参数的设置，实例化扩散模型（U-Net）等，具体内容参考DDPM 类初始化方法、LatentDiffusion 类初始化方法。

    >> 此部分内容较多，较复杂。里面的U-Net （cldm\cldm.py 的ControlledUnetModel类）与后续的导入ControlNet（cldm\cldm.py 的ControlNet 类） 模型的关系是什么，还需要搞清楚。

    * 导入ControlNet 模型，
    * 控制 scale 等参数的设置


* ControlledUnetModel类 与ControlNet 类模型关系说明：

ControlledUnetModel：可接收控制信号的 U-Net 变体；ControlNet：条件到控制信号的转换器；

ControlNet 类： 
* 包括 输入条件处理（input_hint_block）：专门处理用户输入的条件（如边缘图、深度图），通过卷积层将条件转换为与 U-Net 特征兼容的格式。
* 包括 与 U-Net 对齐的特征提取块：包含输入块（input_blocks）、中间块（middle_block），结构与 ControlledUnetModel 的 U-Net 对应（下采样、注意力层、残差块），确保输出的控制信号与 U-Net 各层特征维度匹配。
* 包括 零卷积（zero_convs）：每个输入块和中间块后接 “零初始化卷积”，用于生成控制信号（初始权重为 0，确保训练初期不干扰原始 U-Net）

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
该文件比较重要，主函数的 模型初始化的过程中，继承了很多本文件类的初始化的操作。会有很多的  参数的配置

### LatentDiffusion 类

1、初始化步骤：


* 设置 条件生成策略 （是否 强制使用空条件、条件信息在扩散时间步中生效的步数，默认为1）、数据预处理（是否 使用标准差归一化）
* 动态配置条件信息融合方式，默认使用 交叉注意力机制融合，（直接特征拼接、禁用条件融合； 图像与文本信息的融合方式）
* 从字典中加载参数，包括模型的参数路径（ckpt_path ），EMA参数（是否重置 EMA 权重（布尔值）；是否重置 EMA 更新计数器（布尔值） ） 等
* 调用 父类（DDPM）的初始化函数
    * 参数化模式配置： 比如模型预测的是噪声还是$x_0$,还是预测速度场$v$
    * 条件模型初始化: 将条件信息（如文本、图像）转换为模型可理解的嵌入表示。比如clip，此处设置为None，后续再具体初始化条件模型
    * 日志记录频率，first_stage_key；图片的size，通道数；是否使用位置编码（默认为True）
    * model 的初始化，调用了DiffusionWrapper （同一文件夹内的类）模型
        * 顺序交叉注意力配置，是否顺序处理交叉注意力，如果是，则计算慢，但是省内存；反之 效果相反。
        * 实例化扩散模型（U-Net）： sd模型的初始化，即U-Net结构的噪声预测器（或者是速度场预测器等，默认预测噪声，其他参数的预测，实际上也是基于此，只不过计算公式不一样）
        
        注意：实例化模型，就是controlNet模型，包含sd模型，以及控制部分。模型的定义在 ControlledUnetModel类（cldm\cldm.py 文件中定义）

        * 记录条件的混合参数，交叉注意力融合 还是直接连接的融合
    * 模型的训练稳定性（EMA 配置）、损失优化（权重参数）、硬件适配（make_it_fit）和权重管理（检查点加载与重置）

*  XXX 参数的设置等
* VAE 的初始化
* 文本编码器的初始化


2. 类成员函数说明
*  register_schedule： 扩散模型中注册噪声调度参数的核心逻辑，主要用于计算和缓存扩散过程中所需的各种参数，
*  ema_scope: 用于临时将模型参数切换到EMA 的权重，使用完以后再恢复原始模型参数。下面是使用方法：
* init_from_ckpt： init_from_ckpt 的核心作用是：加载预训练检查点文件中的权重，处理可能的键过滤和形状不匹配问题，最终将适配后的权重加载到当前模型中。
* q_mean_variance 方法的作用是：根据原始数据 $x_0$（x_start）和扩散步数 t，计算t时刻带噪声样本 的均值方差，用于加噪过程的参数计算，方便后续根据此参数重采样，生成$x_t$

注意：此函数用于训练过程

* predict_start_from_noise： 从噪声，预测 $x_0$ 
* predict_start_from_z_and_v： 从用于从第 t步的噪声图像$ x_t$和预测的辅助变量 v(从速度场)中反推初始图像$ x_0$
​* predict_eps_from_z_and_v :根据噪声图像 $x_t$、时间步 t和模型预测的辅助变量 v，计算噪声 ϵ的估计值。

注意： 上述三个预测都是用于反向过程，即去噪的过程。

* q_posterior :计算后验分布所需要的参数，包括均值和方差

* p_mean_variance：整合 q_posterior与predict_start_from_noise等函数，生成后验分布参数，返回均值，方差
* p_sample ：采样，整个过程，跟据$x_t$ 生成$x_{t-1}$ 
* p_sample_loop: c创建一个循环，从$x_T$ 迭代到$x_0$，最忌生成噪声
* sample 封装了上一个函数，输入batchsize等，随机生成噪声，迭代多个batchsize的图像
* q_sample： 前向传播，根据噪声，加噪，如果没有指定噪声，随机加噪
* get_v：用于计算扩散模型中的 velocity 参数 v
* p_losses： 前向过程，计算整个过程的loss
* forward： 封装了前过程，计算loss
* get_input： 将数据转化成符合要求的输入

注意：
注意： 反向过程的步骤
* 预测噪声或者v： 
* 根据噪声或者v 反向推出$x_0$
* 根据$x_0$和t，计算后验分布参数，均值，方差
* 根据均值，方差，采样$x_{t-1}$。此步骤即为重参数化的步骤
* 重复上述步骤


反向加噪的过程，根据原始公式，$x_t$ 与$x_0$的关系，再利用$x_t$ 与$x_{t-1}$的关系，得到￥$x_t$与$x_{t-1}$的关系。
前向去噪的过程，根据原始公式，$x_t$ 与$x_0$的关系，再利用$x_t$ 与$x_{t+1}$的关系，得到￥$x_t$与$x_{t+1}$的关系。


 

### DDPM 类
1、初始化步骤：

参考LatentDiffusion 类 调用父类的初始化函数

2、类成员函数说明
*  register_schedule： 扩散模型中注册噪声调度参数的核心逻辑，主要用于计算和缓存扩散过程中所需的各种参数，
*  ema_scope: 用于临时将模型参数切换到EMA 的权重，使用完以后再恢复原始模型参数。下面是使用方法：
```
with self.ema_scope(context="Validation"):
    # 用EMA权重生成验证样本，计算FID等指标
    val_samples = self.sample(batch_size=8)
    val_fid = compute_fid(val_samples, real_data)
```
* init_from_ckpt： 初始化模型
* q_mean_variance 方法的作用是：根据原始数据 $x_0$（x_start）和扩散步数 t，计算t时刻带噪声样本 的均值方差，用于加噪过程的参数计算，方便后续根据此参数重采样，生成$x_t$
 
 % 均值、方差及对数方差公式
$$
\begin{aligned}
% 均值公式
\mu_q(x_t \mid x_0) &= \sqrt{\bar{\alpha}_t} \cdot x_0 \\
% 方差公式
\sigma_q^2(x_t \mid x_0) &= 1 - \bar{\alpha}_t \\
% 对数方差公式
\log \sigma_q^2(x_t \mid x_0) &= \log\left(1 - \bar{\alpha}_t\right)
\end{aligned}
$$ 

% 符号说明
其中：
- $ x_0 $：原始无噪声数据（如输入图像）；
- $ t $：扩散过程的时间步$ 0 \leq t \leq T $，$ T $ 为总步数；
- $ \alpha_s = 1 - \beta_s $：第 $ s $ 步的“保持系数”（$ \beta_s $ 为第 $ s $ 步的噪声强度）；
- $ \bar{\alpha}_t = \prod_{s=1}^t \alpha_s $：前 $ t $ 步的累积保持系数，反映 $ x_t $ 中原始数据的残留比例。

% 完整分布定义
$$
q(x_t \mid x_0) \sim \mathcal{N}\left( \mu_q(x_t \mid x_0) = \sqrt{\bar{\alpha}_t} \cdot x_0 \,,\ \sigma_q^2(x_t \mid x_0) = 1 - \bar{\alpha}_t \right)
$$

* predict_start_from_noise :从带噪声的样本 $x_t$和预测的噪声 noise中还原出原始数据 $x_0$.用于计算下一步$x_{t-1}$的latent
* predict_start_from_z_and_v: 通过预测速度场 v来还原原始数据$ x_0$
​
* predict_eps_from_z_and_v: 通过预测速度场 v来还原噪声样本$ x_t$
* q_posterior: 扩散模型中的后验分布计算，即给定原始数据 $x_0$和 t时刻的带噪声样本 x_t，计算后验分布$ q(x_{-1} \mid x_t,x_0)$,返回分布的均值方差，用于重采样
* p_mean_variance ：扩散模型的反向过程（去噪）的核心逻辑，通过模型预测和后验分布计算，从带噪声的样本 $x_t$逐步恢复出原始数据 $x_0$
​* p_sample:扩散模型反向过程中的单步采样，即从 t时刻的噪声样本 $x_t$生成 t−1时刻的去噪样本$ x_{t−1}$
​* p_sample_loop :初始化随机噪声 $x_T$ ∼$N(0,I)$。从最大时间步 T到 0，迭代调用 p_sample 进行单步去噪。可选地记录中间步骤的样本，用于可视化或分析。
* sample:扩散模型的高层采样接口，用于快速生成一批样本（如图像）。它封装了完整的反向采样流程，让用户只需指定批量大小即可获得生成结果，无需关心底层实现细节。
* q_sample:扩散模型的前向加噪过程，即从原始数据 $x_0$逐步添加噪声，生成 t时刻的带噪声样本 $x_t$
​* get_v:扩散模型中的速度场（velocity field）计算，即根据带噪声样本$x_t$、噪声 ϵ和时间步 t，计算对应的速度场 v。
* get_loss:扩散模型训练中的损失函数计算，支持两种常见的损失类型：L1 损失（绝对误差）和 L2 损失（均方误差）。
* p_losses: 扩散模型的损失计算，根据输入添加噪声，再计算损失。
* forward：扩散模型的前向传播逻辑，调用p_losses计算损失。
* get_input: 数据预处理功能，将输入批次中的特定键值数据转换为模型所需的格式。包括维度变换等
* shared_step： 扩散模型的共享训练 / 验证步骤，将数据预处理和损失计算封装为一个统一接口
* training_step ：是扩散模型训练的核心流程，通过 UCG 数据增强、损失计算和全面的日志监控，确保模型高效稳定地学习。
* validation_step ：通过同时评估原始模型和 EMA 模型，提供了更全面的模型性能指标。这种双模型评估策略在扩散模型训练中至关重要，有助于选择最佳模型和分析训练稳定性。
* on_train_batch_end：训练批次结束后的 EMA 更新
* _get_rows_from_list：图像网格生成工具
* log_images：图像生成与日志记录
* configure_optimizers：优化器配置













# 注意事项

**如果运行程序，发现没有报错，但是停止了，多半是显存不够。**

**如果运行程序，发现没有报错，但是停止了，多半是cpu 内存不够。** 



 





