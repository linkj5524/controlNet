# 使用说明
## 1. 环境配置
### 1.1 安装依赖
```
    pip install -r requirements.txt
    安装必要的包，按照environment.yaml 的版本按照，如果显示加载权重的时候，出现key 对不上等问题，则是版本未完全按照要求安装。如果为完全按照版本安装，则也可以按如下修改：
        1.1.1 No module named 'pytorch_lightning.utilities.distributed'
            问题原因：pytorch_lightning版本过高，部分函数已经更换过，故会报错
            解决方案：将老版的接口函数，按照新的接口函数修改
            如rom pytorch_lightning.utilities.distributed import rank_zero_only ——> from pytorch_lightning.utilities.rank_zero import rank_zero_only
        1.1.2 No module named 'pytorch_lightning.metrics.functional'
```
### 1.2 下载预训练模型
```
    下载预训练模型（所需要的sd模型），并放到./models/ 文件夹下
    模型初始化会加载标准的sd模型，以及vae等，上述模型在初始化的时候会直接从huggingface官网下载，国内可能会出现报错。以下是两种解决方法：
    1、直接从官网或者其他途径下载模型，并放到huggingface的缓存文件夹下，也可以直接设置缓存文件夹的路径，或者是加载模型时，改成路径。
    2、利用国内的镜像，本项目目前使用本方法，更为方便。
        2.1  安装依赖：pip install -U huggingface_hub
        2.2  设置环境变量：
             Windows：$env:HF_ENDPOINT = "https://hf-mirror.com"


        

            
            * linux： export HF_ENDPOINT=https://hf-mirror.com


```
## 2. 训练模型
```
```