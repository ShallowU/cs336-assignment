import torch
import argparse
import timeit
import numpy as np
from tqdm import tqdm
from config import config
from layer import TransformerLM
from loss import cross_entropy
from optimizer import My_AdamW, My_lr_cosine_schedule, My_gradient_clipping
from util import My_save_checkpoint,My_load_checkpoint
from data import My_get_batch

def benchmark(d_model, d_ff, num_layers, num_heads, size):
    device=config['device']
    model = TransformerLM(
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=config['rope_theta']
    ).to(device)
    trainable_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    optimizer = My_AdamW(model.parameters(), lr=config['max_lr'])
    batched_data_x = torch.randint(0,10000,(1,config['context_length'])).to(device)
    batched_data_y=batched_data_x+1

    def forward_pass_only():
        optimizer.zero_grad()
        output=model(batched_data_x)
        loss=cross_entropy(output,batched_data_y)
        torch.cuda.synchronize()
        return loss
    
    def backward_pass_no_step():
        optimizer.zero_grad()
        output=model(batched_data_x)
        loss=cross_entropy(output,batched_data_y)
        loss.backward()
        torch.cuda.synchronize()
    
    def backward_pass_full():
        optimizer.zero_grad()
        output=model(batched_data_x)
        loss=cross_entropy(output,batched_data_y)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
    
    # 使用timeit测试前向传播时间
    print("=== 性能测试 ===")
    print("测试配置:")
    print(f"- 设备: {config['device']}")
    print(f"- d_model: {d_model}")
    print(f"- d_ff: {d_ff}")
    print(f"- num_layers: {num_layers}")
    print(f"- num_heads: {num_heads}")
    print(f"- model size: {size}")
    print(f"- trainable_parameters: {trainable_parameters/1e9:.6f} B")
    print()

    forward_pass_time = []
    for i in range(15):
        if i < 5:
            forward_pass_only()
            print(f"预热轮次 {i+1}/5 完成")
        else:
            forward_pass_time.append(timeit.timeit(forward_pass_only, number=1))
            print(f"测试轮次 {i-4}/10 完成")
    forward_pass_time = np.array(forward_pass_time)

    backward_pass_no_step_time = []
    for i in range(15):
        if i < 5:
            backward_pass_no_step()
            print(f"预热轮次 {i+1}/5 完成")
        else:
            backward_pass_no_step_time.append(timeit.timeit(backward_pass_no_step, number=1))
            print(f"测试轮次 {i-4}/10 完成")
    backward_pass_no_step_time = np.array(backward_pass_no_step_time)

    backward_pass_full_time = []
    for i in range(15):
        if i < 5:
            backward_pass_full()
            print(f"预热轮次 {i+1}/5 完成")
        else:
            backward_pass_full_time.append(timeit.timeit(backward_pass_full, number=1))
            print(f"测试轮次 {i-4}/10 完成")
    backward_pass_full_time = np.array(backward_pass_full_time)

    print(f"前向传播平均时间: {forward_pass_time.mean():.6f} 秒, 标准差: {forward_pass_time.std():.6f} 秒")
    print(f"前向+反向传播时间: {backward_pass_no_step_time.mean():.6f} 秒， 标准差: {backward_pass_no_step_time.std():.6f} 秒")
    print(f"纯反向传播时间(估算): {(backward_pass_no_step_time.mean() - forward_pass_time.mean()):.6f} 秒")
    print(f"完整一步更新时间(包括优化器更新): {backward_pass_full_time.mean():.6f} 秒， 标准差: {backward_pass_full_time.std():.6f} 秒")

if __name__ == "__main__":
    model_size_dict = {
        "d_model":[768, 1024, 1280, 1600, 2560],
        "d_ff":[3072, 4096, 5120, 6400, 10240],
        "num_layers":[12, 24, 36, 48, 32],
        "num_heads":[12, 16, 20, 25, 32],
        "size":['small','medium','large','xl','2.7B']
    }
    print("-------------------------------")
    # print("测试普通attention耗时")
    for i in range(5):
        # type = "normal"
        d_model = model_size_dict['d_model'][i]
        d_ff = model_size_dict['d_ff'][i]
        num_layers = model_size_dict['num_layers'][i]
        num_heads = model_size_dict['num_heads'][i]
        size = model_size_dict['size'][i]
        try:
            benchmark(d_model, d_ff, num_layers, num_heads, size)
        except Exception as e:
            print(e)











