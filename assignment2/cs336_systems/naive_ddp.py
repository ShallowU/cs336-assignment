
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import os
import torch.distributed as dist
import time 
from layer import TransformerLM
from loss import cross_entropy
from optimizer import My_AdamW

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
def sync_model_params(model: nn.Module):
    for param in model.parameters():
        dist.all_reduce(param, op=dist.ReduceOp.AVG, async_op=False)
def sync_model_grad(model: nn.Module,flatten):
    grads = []
    if flatten == True:
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad)
        flattened_grads = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flattened_grads, op=dist.ReduceOp.AVG, async_op=False)
        unflattened_grad = torch._utils._unflatten_dense_tensors(flattened_grads, grads)
        for i, param in enumerate(model.parameters()):
            if param.grad is not None:
                param.grad = unflattened_grad[i]
    else:
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=False)   
                
def distributed(rank, world_size, backend, warmup, flatten):
    setup(rank, world_size, backend)
    # 每个进程用不同的种子
    torch.manual_seed(42)
    if warmup == True and rank == 0:
        print("warmup......")
    if warmup == False and rank == 0:
        print("training......")
    if backend == 'nccl':
        device = f"cuda:{rank}"
    else:
        device = 'cpu'
    model = Toymodel().to(device=device)
    # 1. prepare model
    # model = TransformerLM(vocab_size=10000, context_length=256, d_model=1024, num_layers=24, num_heads=16, d_ff=4096, rope_theta=10000)
    model.to(device)
    with torch.no_grad():
        sync_model_params(model)
    optimizer = torch.optim.SGD(model.parameters(),lr = 0.00001)
    # 再设置不同的种子生成不同的数据
    torch.manual_seed(42 + rank)
    batched_x = torch.rand((8, 1024, 3),dtype=torch.float32,requires_grad=True,device=device)
    # optimizer = My_AdamW(model.parameters(), lr=0.00001)
    # batched_data_x = torch.randint(0,10000,(1,256)).to(device)
    # batched_data_y = torch.zeros_like(batched_data_x)
    if rank == 0 and warmup is not True:
        full_time = []
        sync_time = []
        all_losses = []  # 添加这行
    for _ in (range(100)):
        optimizer.zero_grad()  # 移到这里
        if rank == 0 and warmup is not True:
            start = time.time()
        output = model(batched_x)
        # loss = cross_entropy(output, batched_x)
        loss= loss_func(output)
        loss.backward()
        # torch.cuda.synchronize()
        if rank == 0 and warmup is not True:
            sync_start = time.time()
        with torch.no_grad():
            sync_model_grad(model,flatten)
            # torch.cuda.synchronize()
        if rank == 0 and warmup is not True:
            sync_end = time.time()
        optimizer.step()
        # torch.cuda.synchronize()
        if rank == 0 and warmup is not True:
            end = time.time()
            full_time.append(end - start)
            sync_time.append(sync_end - sync_start)
            all_losses.append(loss.item())  # 添加这行
    
    print(f"Rank {rank} final loss: {loss.item()}")
    if rank == 0 and warmup is not True:
        print(f"use flatten: {flatten}")
        print(f"average full time: {sum(full_time) / len(full_time)}")
        print(f"average sync time: {sum(sync_time) / len(sync_time)}")
        print(f"average loss over 100 iters: {sum(all_losses) / len(all_losses)}")  # 添加这行
    # 确保所有进程同步后再退出
    dist.barrier()
def ddp_train(backend, process, warmup, flatten=True):
    world_size = process
    mp.spawn(fn=distributed, args=(world_size,backend,warmup,flatten),nprocs=world_size,join=True)
if __name__ == "__main__":
    type = 'single'  # single or ddp
    if type == 'single':
        torch.manual_seed(42)
        model = Toymodel().to('cpu')
        optimizer = torch.optim.SGD(model.parameters(),lr = 0.00001)
        # 模拟2个进程的数据
        torch.manual_seed(42)
        batch1 = torch.rand((8, 1024, 3),dtype=torch.float32,device='cpu',requires_grad=True)
        torch.manual_seed(43)
        batch2 = torch.rand((8, 1024, 3),dtype=torch.float32,device='cpu',requires_grad=True)
        all_losses = []
        for _ in range(100):
            optimizer.zero_grad()
            # 用batch1计算梯度
            output1 = model(batch1)
            loss1 = loss_func(output1)
            loss1.backward()
            
            # 用batch2计算梯度并累加
            output2 = model(batch2)
            loss2 = loss_func(output2)
            loss2.backward()
            
            # 平均梯度（模拟DDP的all_reduce）
            with torch.no_grad():
                for param in model.parameters():
                    param.grad /= 2
            
            optimizer.step()
            all_losses.append((loss1.item() + loss2.item()) / 2)  # 记录平均损失
        
        print(f"Single final loss (averaged):{(loss1.item() + loss2.item())/2}")
        print(f"Single average loss over 100 iters: {sum(all_losses) / len(all_losses)}")  # 添加这行
    elif type == 'ddp':
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            backend = 'nccl'  # 使用NCCL后端进行GPU训练
            process = 2       # 使用2个GPU
        else:
            backend = 'gloo'  # 如果GPU不够，回退到CPU训练
            process = 2
        # ddp_train(backend, process, warmup=True)
        # ddp_train(backend, process, warmup=False, flatten=True)
        ddp_train(backend, process, warmup=False, flatten=False)
    
