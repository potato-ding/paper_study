import numpy as np

def uniform_binning(action_value, min_val=-1.0, max_val=1.0, bins=256):
    """模拟 RT-1 的动作分词过程"""
    # 1. 归一化到 [0, 1]
    norm_val = (action_value - min_val) / (max_val - min_val)
    # 2. 映射到 [0, 255]
    token = int(norm_val * (bins - 1))
    return np.clip(token, 0, bins - 1)

def de_binning(token, min_val=-1.0, max_val=1.0, bins=256):
    """模拟机器人解码动作"""
    norm_val = token / (bins - 1)
    action_val = norm_val * (max_val - min_val) + min_val
    return action_val

# 测试精度损失
original_action = 0.53782  # 假设这是真实的精确动作
token = uniform_binning(original_action)
decoded_action = de_binning(token)

print(f"原始动作: {original_action}")
print(f"RT-1 Token: {token}")
print(f"解码后动作: {decoded_action}")
print(f"精度损失: {abs(original_action - decoded_action):.6f}")