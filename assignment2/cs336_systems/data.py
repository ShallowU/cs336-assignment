import torch
import numpy as np

class BatchIterator:
    def __init__(self, data,type:str, batch_size, context_length, device="cpu"):
        """
        基于流式读取的 BatchIterator
        逻辑参考：nanoGPT / GPT-2 standard data loader
        """
        self.data = data # 预期是 mmap 的 numpy array
        self.type=type
        self.B = batch_size
        self.T = context_length
        self.device = device
        self.current_position = 0
        self.total_len = len(data)
        
        print(f"BatchIterator initialized: {self.total_len:,} tokens.")

    def get_batch(self):
        B, T = self.B, self.T
        
        # 1. 边界检查：如果剩余数据不够一个 Batch，则重置 (Epoch 结束)
        # 我们需要 B*T + 1 个 token (因为要有 target)
        if self.current_position + B * T + 1 > self.total_len:
            self.current_position = 0
            if self.type=='train':
                print("Train Epoch finished, resetting to start of data.")
            
        # 2. 提取 Buffer
        # 从长流中直接切出一大块：大小为 B*T + 1
        buf = self.data[self.current_position : self.current_position + B * T + 1]
        
        # 3. 转换为 Tensor 并移至 GPU
        # 注意：先转 int64 确保兼容 Embedding 层
        buf = torch.from_numpy(buf.astype(np.int64)).to(self.device)
        
        # 4. 构建 Input (x) 和 Target (y)
        # buf[:-1] 是 0 到 BT-1
        # buf[1:]  是 1 到 BT
        # .view(B, T) 将一维流折叠成二维 Batch
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        
        # 5. 推进指针
        # 下一次读取从当前这一块的末尾开始，绝不重叠
        self.current_position += B * T
        
        return x, y

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