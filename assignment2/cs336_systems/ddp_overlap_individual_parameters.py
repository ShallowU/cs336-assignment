import torch
import torch.nn as nn
import torch.multiprocessing as mp
import os
import torch.distributed as dist
import time 
from layer import TransformerLM
from loss import cross_entropy
from optimizer import My_AdamW
from ddp_overlap_bucketed import DDPBucketed
class Toymodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.Linear(3,16)
        self.norm = nn.LayerNorm(16)
        self.ln2 = nn.Linear(16,1)
    def forward(self, x):
        x = self.ln1(x)
        x = self.norm(x)
        x = self.ln2(x)
        return x
def loss_func(x):
    return torch.sum(x**2)
def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29100"
    dist.init_process_group(backend=backend,rank=rank,world_size=world_size)
class My_DDP(nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.handlers = []
        # 使用 broadcast 而不是 all_reduce，将 rank 0 的参数广播到所有 ranks
        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)
        for parameter in self.module.parameters():
            if parameter.requires_grad == True:
                parameter.register_post_accumulate_grad_hook(self.sync_model_grad_async)
    def sync_model_grad_async(self, parameter):
        handler = dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, async_op=True)
        self.handlers.append((handler, parameter))
    def finish_gradient_synchronization(self):
        for handler, param in self.handlers:  # 修改解包方式
            handler.wait()
            if param.grad is not None:
                param.grad.data = param.grad.data / dist.get_world_size()
        self.handlers.clear()
    def forward(self, x):
        return self.module(x)
def distributed(rank, world_size, backend, warmup, bucket_size):
    setup(rank, world_size, backend)
    if warmup == True and rank == 0:
        print("warmup......")
    if warmup == False and rank == 0:
        print("training......")
    if backend == 'nccl':
        device = f"cuda:{rank}"
    else:
        device = 'cpu'
    # model = Toymodel().to(device=device)
    # 1. prepare model
    # model = My_transformer_lm(vocab_size=10000, context_length=256, d_model=1024, num_layers=24, num_heads=16, d_ff=4096, rope_theta=10000)
    model = TransformerLM(
        vocab_size=10000, 
        context_length=256, 
        d_model=768,      # 1024 -> 768
        num_layers=12,    # 24 -> 12
        num_heads=12,     # 16 -> 12
        d_ff=3072,        # 4096 -> 3072
        rope_theta=10000
    )
    model.to(device)
    model = DDPBucketed(model, bucket_size_mb=bucket_size)  # 推荐值25MB

    # if sharded_opt == True:
    #     optimizer = ShardedOptimizer(model.parameters(), My_AdamW, lr=1e-5)
    # else:
    optimizer = My_AdamW(model.parameters(), lr=0.00001)
    # batched_data_x = torch.randint(0,10000,(1,256)).to(device)
    batched_data_x = torch.randint(0,10000,(1,32)).to(device)
    batched_data_y = torch.zeros_like(batched_data_x)
    if rank == 0 and warmup is not True:
        full_time = []
    for _ in (range(100)):
        if rank == 0 and warmup is not True:
            start = time.time()
        output = model(batched_data_x)
        loss = cross_entropy(output, batched_data_y)
        optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            model.finish_gradient_synchronization()
        optimizer.step()
        torch.cuda.synchronize()
        if rank == 0 and warmup is not True:
            end = time.time()
            full_time.append(end - start)
            print(f"loss:{loss.item()}")
    if rank == 0 and warmup is not True:
        print(f"average full time: {sum(full_time) / len(full_time)}")
    # 确保所有进程同步后再清理
    dist.barrier()
    # 清理分布式进程组
    dist.destroy_process_group()

def ddp_train(backend, process, warmup, bucket_size):
    world_size = process
    mp.spawn(fn=distributed, args=(world_size,backend,warmup,bucket_size),nprocs=world_size,join=True)
if __name__ == "__main__":
    type = 'ddp'
    if type == 'single':
        model = Toymodel().to('mps')
        optimizer = torch.optim.SGD(model.parameters(),lr = 0.00001)
        batched_x = torch.rand((8, 1024, 3),dtype=torch.float32,device='mps',requires_grad=True)
        for _ in range(100):
            output = model(batched_x)
            loss = loss_func(output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"loss:{loss.item()}")
    elif type == 'ddp':
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            backend = 'nccl'  # 使用NCCL后端进行GPU训练
            process = 2       # 使用2个GPU
        else:
            backend = 'gloo'  # 如果GPU不够，回退到CPU训练
            process = 2
            # 测试不同的 bucket size
        for bucket_size in [10, 25, 35]:
            print(f"\n{'='*50}")
            print(f"Testing with bucket_size={bucket_size}MB")
            print(f"{'='*50}\n")
            
            ddp_train(backend, process, warmup=True, bucket_size=bucket_size)
            ddp_train(backend, process, warmup=False, bucket_size=bucket_size)
    