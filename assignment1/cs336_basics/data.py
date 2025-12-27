import torch
import numpy as np

class BatchIterator:
    """顺序遍历数据集的批次迭代器"""
    def __init__(self, data, batch_size, context_length, device="cpu"):
        self.data = data
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        self.corpus_len = len(data)
        
        # 计算可用的起始位置数量
        self.max_start_idx = self.corpus_len - context_length
        # 初始化索引
        self.current_idx = 0
        
    def get_batch(self):
        """获取下一个批次"""
        batch_inputs = []
        batch_targets = []
        
        for _ in range(self.batch_size):
            # 如果到达末尾,重新开始并打乱
            if self.current_idx >= self.max_start_idx:
                self.current_idx = 0
            
            # 提取序列
            start = self.current_idx
            batch_inputs.append(self.data[start:start + self.context_length])
            batch_targets.append(self.data[start + 1:start + self.context_length + 1])
            
            # 移动索引
            self.current_idx += 1
        
        inputs = np.array(batch_inputs)
        targets = np.array(batch_targets)
        
        return (
            torch.from_numpy(inputs).to(self.device),
            torch.from_numpy(targets).to(self.device)
        )
    
    def reset(self):
        """重置迭代器"""
        self.current_idx = 0


def My_get_batch(x, batch_size, context_length, device="cpu"):
    """保留原函数接口以兼容测试"""
    corpus_len = len(x)
    idx = np.random.randint(0, corpus_len - context_length, batch_size)
    
    inputs = np.array([x[i:i+context_length] for i in idx])
    targets = np.array([x[i+1:i+context_length+1] for i in idx])
    
    return (
        torch.from_numpy(inputs).to(device),
        torch.from_numpy(targets).to(device)
    )