import torch
import torch.nn as nn
import time
import itertools
from contextlib import contextmanager

# Context manager for timing
@contextmanager
def timer(name: str):
    torch.cuda.synchronize()  # 确保之前的操作完成
    start = time.time()
    yield
    torch.cuda.synchronize()  # 确保当前操作完成
    print(f"{name} took {(time.time() - start)*1000:.2f} ms")

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, Q, K, V):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k**0.5
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V)

# Warm-up function
def warmup(steps=10, model=None, batch_size=8):
    for _ in range(steps):
        with torch.no_grad():
            Q = torch.randn(batch_size, 1024, 64, device='cuda')
            K = torch.randn(batch_size, 1024, 64, device='cuda')
            V = torch.randn(batch_size, 1024, 64, device='cuda')
            _ = model(Q, K, V)
        torch.cuda.synchronize()

# Benchmark function
def benchmark_attention(d_model, seq_len, model, batch_size=8):
    print(f"\nBenchmarking batch_size={batch_size}, d_model={d_model}, seq_len={seq_len}...")

    # Allocate inputs
    Q = torch.randn(batch_size, seq_len, d_model, device='cuda', requires_grad=True)
    K = torch.randn(batch_size, seq_len, d_model, device='cuda', requires_grad=True)
    V = torch.randn(batch_size, seq_len, d_model, device='cuda', requires_grad=True)

    # Warm-up
    for _ in range(10):
        out = model(Q, K, V)
        out.sum().backward()
        # 清零梯度
        Q.grad = None
        K.grad = None
        V.grad = None
    torch.cuda.synchronize()

    # === Forward timing ===
    torch.cuda.reset_peak_memory_stats()
    with timer("100 forward passes"):
        for _ in range(100):
            out = model(Q, K, V)

    # === Memory measurement ===
    # 执行一次前向传播后测量内存
    torch.cuda.reset_peak_memory_stats()
    out = model(Q, K, V)
    torch.cuda.synchronize()
    
    mem_reserved = torch.cuda.memory_reserved() / 1024**2
    mem_allocated = torch.cuda.memory_allocated() / 1024**2
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    
    print(f"Memory reserved: {mem_reserved:.2f} MB")
    print(f"Memory allocated: {mem_allocated:.2f} MB")
    print(f"Peak memory: {peak_mem:.2f} MB")

    # === Backward timing ===
    with timer("100 backward passes"):
        for _ in range(100):
            Q.grad = None
            K.grad = None
            V.grad = None
            out = model(Q, K, V)
            out.sum().backward()
        torch.cuda.synchronize()

# Main script
if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")

    torch.cuda.empty_cache()
    device = torch.device('cuda')
    print(f"Using device: {torch.cuda.get_device_name(0)}")

    # Model
    model = Attention().to(device)
    # model = torch.compile(model) 

    # Warm up GPU
    print("Warming up...")
    warmup(model=model, batch_size=8)

    # Parameters
    batch_size = 8
    d_models = [16, 32, 64, 128]
    seq_lengths = [256, 1024, 4096, 8192, 16384]

    # Cartesian product
    for d_model, seq_len in itertools.product(d_models, seq_lengths):
        try:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            benchmark_attention(d_model, seq_len, model=model, batch_size=batch_size)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM error for d_model={d_model}, seq_len={seq_len}")
                torch.cuda.empty_cache()
            else:
                print(f"Error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    print("\nBenchmarking complete.")