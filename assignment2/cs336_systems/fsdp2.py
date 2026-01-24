import torch
import torch.nn as nn
import torch.multiprocessing as mp
import os
import torch.distributed as dist
import time

# [FSDP2 关键点 1]: 引入 FSDP2 的核心 API
# fully_shard: 用于将模型参数切分的函数
# FSDPModule: 包装后的模型类型，用于类型检查
from torch.distributed.fsdp import fully_shard, FSDPModule

# 引入混合精度策略 (可选，但推荐)
from torch.distributed.fsdp import MixedPrecisionPolicy

try:
    from .layer import TransformerLM
    from .loss import cross_entropy
    # 建议使用 PyTorch 原生优化器以确保对 DTensor 的最佳支持
    # 如果 My_AdamW 是标准 AdamW 实现也可以，这里为了演示稳定性换回原生
    from torch.optim import AdamW 
except ImportError:
    from layer import TransformerLM
    from loss import cross_entropy
    from torch.optim import AdamW

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29100"
    # FSDP2 通常使用 nccl 后端
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    # 设置当前设备，确保 FSDP 正确分配显存
    torch.cuda.set_device(rank)

def distributed_fsdp(rank, world_size, backend, warmup):
    setup(rank, world_size, backend)
    
    device = torch.device(f"cuda:{rank}")

    if warmup and rank == 0:
        print("warmup......")
    if not warmup and rank == 0:
        print(f"training...... (World Size: {world_size})")

    # 1. 实例化模型
    # 注意：通常建议在 'meta' 设备上初始化以节省 CPU 内存，
    # 但为了简单迁移，我们先在 CPU 上初始化，FSDP 会负责移动到 GPU
    model = TransformerLM(
        vocab_size=10000, 
        context_length=256, 
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=3072,
        rope_theta=10000
    )

    # [FSDP2 关键点 2]: 定义混合精度策略 (可选)
    # param_dtype=bfloat16: 计算时参数转为 bfloat16 (加速，省显存)
    # reduce_dtype=float32: 梯度同步时用 float32 (保证精度)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32
    )

    # 第一步：遍历 ModuleList，单独包装每一个 TransformerBlock
    # 你的类中叫 self.blocks，所以我们直接遍历它
    for block in model.blocks:
        fully_shard(block, mp_policy=mp_policy)

    # 第二步：包装最外层的模型
    # 这会把剩下的部分（tok_emb, ln_final, lm_head）归为一组进行管理
    fully_shard(model, mp_policy=mp_policy)
    # 3. 检查包装结果
    # model.blocks[0] 应该是 FSDPModule
    # model 本身也应该是 FSDPModule
    if rank == 0:
        print(f"Block 0 wrapped: {isinstance(model.blocks[0], FSDPModule)}")
        print(f"Root model wrapped: {isinstance(model, FSDPModule)}")
        print("Model structure:")
        print(model)
        # 输出示例:
        # FSDPModule(
        #   (_fsdp_wrapped_module): TransformerLM(
        #     (layers): ModuleList(
        #       (0): FSDPModule( ... )  <-- 每一层都被包住了
        #       (1): FSDPModule( ... )
        # ...
    # 验证是否包装成功
    # assert isinstance(model, FSDPModule)
    
    # 将模型显式移动到 GPU (fully_shard 有时会自动处理，但显式调用更安全)
    model = model.to(device)

    # [FSDP2 关键点 5]: 初始化优化器
    # 必须在 fully_shard 之后！
    # 此时 model.parameters() 已经是 DTensor (Distributed Tensor)，
    # 优化器会自动处理分片的参数状态。
    optimizer = AdamW(model.parameters(), lr=0.00001)

    batched_data_x = torch.randint(0, 10000, (1, 32)).to(device)
    batched_data_y = torch.zeros_like(batched_data_x)

    if rank == 0 and not warmup:
        full_time = []

    # 训练循环
    for i in range(100):
        if rank == 0 and not warmup:
            start = time.time()
        
        # [FSDP2 流程]: Forward
        # FSDP 会自动处理：
        # 1. All-Gather 当前层需要的参数 (从切片还原完整层)
        # 2. 计算
        # 3. 释放完整参数 (Free)，只保留切片，节省显存
        output = model(batched_data_x)
        
        loss = cross_entropy(output, batched_data_y)
        
        optimizer.zero_grad()
        
        # [FSDP2 流程]: Backward
        # FSDP 会自动处理：
        # 1. Reduce-Scatter 梯度 (同步并切分梯度)
        # 2. 不需要你手动调用 synchronization hook
        loss.backward()
        
        # [FSDP2 流程]: Optimizer Step
        # 更新本地的那一小部分切片参数
        optimizer.step()
        
        # 只有在测速时需要 synchronize，平时训练不需要
        torch.cuda.synchronize()

        if rank == 0 and not warmup:
            end = time.time()
            full_time.append(end - start)
            # 打印 Loss (注意：FSDP 的 loss 是本地 Tensor，通常不需要 all-reduce loss 本身，除非为了打印日志)
            if i % 10 == 0:
                print(f"Step {i}, loss: {loss.item()}")

    if rank == 0 and not warmup:
        avg_time = sum(full_time) / len(full_time)
        print(f"FSDP2 Average full time: {avg_time:.4f} s")

    dist.barrier()
    dist.destroy_process_group()

def fsdp_main(process, warmup):
    world_size = process
    # FSDP 强依赖 NCCL
    backend = 'nccl' 
    mp.spawn(fn=distributed_fsdp, args=(world_size, backend, warmup), nprocs=world_size, join=True)

if __name__ == "__main__":
    # 确保有 GPU
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        print(f"\n{'='*50}")
        print(f"Starting FSDP2 Training")
        print(f"{'='*50}\n")
        
        process = 2
        # Warmup run
        fsdp_main(process, warmup=True)
        # Actual run
        fsdp_main(process, warmup=False)
    else:
        print("FSDP2 requires at least 2 GPUs with NCCL backend.")