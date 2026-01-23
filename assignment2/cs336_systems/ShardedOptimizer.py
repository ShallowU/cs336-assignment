import torch
import torch.distributed as dist
from typing import (
    Any,
    Dict,
    List,
)
class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer, **kwargs):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        super().__init__(params, kwargs)
        self.optimizer = optimizer(self.local_param_group['params'], **kwargs) # 先准备好local_param_group，再实例化本地的optimizer
        
    @torch.no_grad()
    def step(self,closure=None, **kwargs):# 注意参数传递，这个closure
        self.optimizer.step(closure, **kwargs)
        for src_rank in range(self.world_size): # 一定注意broadcase的方法，源是谁，接收是谁
            for param_group in self.param_groups:
                for i, param in enumerate(param_group['params']):
                    if i % self.world_size == src_rank:
                        dist.broadcast(param.data,src=src_rank)
    
    def add_param_group(self, param_group: Dict[str, Any]):
        # 传入的 param_group 是一个字典，其中 'params' 键对应的是一个参数列表。我们需要遍历这个列表。
        # self.param_groups: List[Dict[str, Any]] = []
        # 需要考虑到传入的是多个param_group的情况
        super().add_param_group(param_group) # 先保存一份完整的模型参数
        local_param_group: Dict[str, Any] = {'params':[]} # 只需要存param，不需要存其他的，因为实例化的时候有**kwargs
        for i, param in enumerate(param_group['params']):
            if i % self.world_size == self.rank:
                local_param_group['params'].append(param)
        self.local_param_group = local_param_group