from torch.nn import Module
import torch
def get_module_device(module: Module) -> torch.device:
    """
    获取PyTorch模块所在的设备（torch.device对象）
    
    参数:
        module: 要检查的PyTorch模块（如nn.Sequential、nn.ModuleList等）
    
    返回:
        torch.device: 模块所在的设备（CPU或GPU）
    
    异常:
        ValueError: 模块没有任何参数，无法确定设备
    """
    # 递归检查模块及其子模块的参数
    def _find_device_recursive(m: Module) -> torch.device:
        # 检查当前模块的参数
        for param in m.parameters():
            return param.device
        
        # 检查当前模块的缓冲区（如BN层的running_mean）
        for buf in m.buffers():
            return buf.device
        
        # 递归检查子模块
        for child in m.children():
            device = _find_device_recursive(child)
            if device is not None:
                return device
        
        # 未找到任何参数或缓冲区
        return None
    
    # 执行递归查找
    device = _find_device_recursive(module)
    
    if device is None:
        raise ValueError("模块及其子模块没有任何参数或缓冲区，无法确定设备。请先初始化模块参数。")
    
    return device


def concat_dicts(c, unconditional_conditioning):
    # 确保两个字典的键相同
    assert c.keys() == unconditional_conditioning.keys(), "两个字典的键必须一致"
    
    merged = {}
    for key in c:
        # 对每个键对应的张量沿第0维拼接（可根据需求修改dim）
        merged[key] = c[key]+unconditional_conditioning[key]
    return merged