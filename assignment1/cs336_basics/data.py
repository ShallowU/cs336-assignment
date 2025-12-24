import torch
import numpy as np

def My_get_batch(x, batch_size, context_length, device="cpu"):
    corpus_len = len(x)
    # 一次性生成所有随机索引
    idx = np.random.randint(0, corpus_len - context_length, batch_size)
    
    # 使用高级索引直接提取（避免循环）
    inputs = np.array([x[i:i+context_length] for i in idx])
    targets = np.array([x[i+1:i+context_length+1] for i in idx])
    
    return (
        torch.from_numpy(inputs).to(device),
        torch.from_numpy(targets).to(device)
    )