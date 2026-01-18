import torch
import triton
from flash_forward import Flash_attention_triton, annotated_scaled_dot_product_attention
import pandas as pd
from typing import Tuple, List
import gc

def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """创建 causal mask"""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    return ~mask  # 返回下三角为 True 的 mask

def benchmark_pytorch_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> float:
    """Benchmark PyTorch forward pass (inference mode, no grad)"""
    def fn():
        with torch.no_grad():
            return annotated_scaled_dot_product_attention(Q, K, V, mask)
            # 使用 PyTorch 内置的 SDPA，更公平的对比
            # return torch.nn.functional.scaled_dot_product_attention(
            #     Q, K, V, is_causal=is_causal
            # )
    
    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    return ms

def benchmark_pytorch_forward_backward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> float:
    """Benchmark PyTorch forward + backward pass"""
    Q_grad = Q.clone().requires_grad_(True)
    K_grad = K.clone().requires_grad_(True)
    V_grad = V.clone().requires_grad_(True)
    grad_output = torch.randn(Q.shape, device=Q.device, dtype=Q.dtype)
    def fn():
        O = annotated_scaled_dot_product_attention(Q_grad, K_grad, V_grad, mask)
        # O = torch.nn.functional.scaled_dot_product_attention(
        #     Q_grad, K_grad, V_grad, is_causal=True
        # )
        O.backward(grad_output)
        
        if Q_grad.grad is not None:
            Q_grad.grad.zero_()
        if K_grad.grad is not None:
            K_grad.grad.zero_()
        if V_grad.grad is not None:
            V_grad.grad.zero_()
    
    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    return ms

def benchmark_pytorch_backward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> float:
    """Benchmark PyTorch backward pass (estimated by subtraction)"""
    fwd_bwd_time = benchmark_pytorch_forward_backward(Q, K, V, mask)
    fwd_time = benchmark_pytorch_forward(Q, K, V, mask)
    return fwd_bwd_time - fwd_time

def benchmark_triton_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> float:
    """Benchmark Triton forward pass"""
    def fn():
        with torch.no_grad():
            return Flash_attention_triton.apply(Q, K, V, True)
    
    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    return ms

def benchmark_triton_forward_backward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> float:
    """Benchmark Triton forward + backward pass"""
    Q_grad = Q.clone().requires_grad_(True)
    K_grad = K.clone().requires_grad_(True)
    V_grad = V.clone().requires_grad_(True)
    grad_output = torch.randn(Q.shape, device=Q.device, dtype=Q.dtype)
    def fn():
        O = Flash_attention_triton.apply(Q_grad, K_grad, V_grad, True)
        O.backward(grad_output)
        
        if Q_grad.grad is not None:
            Q_grad.grad.zero_()
        if K_grad.grad is not None:
            K_grad.grad.zero_()
        if V_grad.grad is not None:
            V_grad.grad.zero_()
    
    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    return ms

def benchmark_triton_backward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> float:
    """Benchmark Triton backward pass (estimated by subtraction)"""
    fwd_bwd_time = benchmark_triton_forward_backward(Q, K, V)
    fwd_time = benchmark_triton_forward(Q, K, V)
    return fwd_bwd_time - fwd_time

def run_benchmark(seq_len: int, d: int, dtype: torch.dtype, device: str = 'cuda') -> dict:
    """运行单组 benchmark"""
    print(f"Running benchmark: seq_len={seq_len}, d={d}, dtype={dtype}")
    
    batch_size = 1
    
    # 创建输入数据
    Q = torch.randn(batch_size, seq_len, d, device=device, dtype=dtype)
    K = torch.randn(batch_size, seq_len, d, device=device, dtype=dtype)
    V = torch.randn(batch_size, seq_len, d, device=device, dtype=dtype)
    
    # 创建 causal mask (PyTorch 需要)
    mask = create_causal_mask(seq_len, device)
    
    # 确保 CUDA 同步
    torch.cuda.synchronize()
    
    try:
        # Benchmark PyTorch
        pt_fwd = benchmark_pytorch_forward(Q, K, V, mask)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        
        pt_bwd = benchmark_pytorch_backward(Q, K, V, mask)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        
        pt_fwd_bwd = benchmark_pytorch_forward_backward(Q, K, V, mask)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        
        # Benchmark Triton
        triton_fwd = benchmark_triton_forward(Q, K, V)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        
        triton_bwd = benchmark_triton_backward(Q, K, V)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        
        triton_fwd_bwd = benchmark_triton_forward_backward(Q, K, V)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        
        return {
            'seq_len': seq_len,
            'd': d,
            'dtype': str(dtype).split('.')[-1],
            'pytorch_fwd_ms': pt_fwd,
            'pytorch_bwd_ms': pt_bwd,
            'pytorch_fwd_bwd_ms': pt_fwd_bwd,
            'triton_fwd_ms': triton_fwd,
            'triton_bwd_ms': triton_bwd,
            'triton_fwd_bwd_ms': triton_fwd_bwd,
            'speedup_fwd': pt_fwd / triton_fwd,
            'speedup_bwd': pt_bwd / triton_bwd,
            'speedup_fwd_bwd': pt_fwd_bwd / triton_fwd_bwd,
        }
    except RuntimeError as e:
        print(f"  Error: {e}")
        return {
            'seq_len': seq_len,
            'd': d,
            'dtype': str(dtype).split('.')[-1],
            'pytorch_fwd_ms': float('nan'),
            'pytorch_bwd_ms': float('nan'),
            'pytorch_fwd_bwd_ms': float('nan'),
            'triton_fwd_ms': float('nan'),
            'triton_bwd_ms': float('nan'),
            'triton_fwd_bwd_ms': float('nan'),
            'speedup_fwd': float('nan'),
            'speedup_bwd': float('nan'),
            'speedup_fwd_bwd': float('nan'),
        }
def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 让 cuDNN 行为更确定（可能牺牲一点性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(42)
    # 检查 CUDA 是否可用
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run on a GPU.")
        return
    
    device = 'cuda'
    print(f"Running on: {torch.cuda.get_device_name(0)}")
    
    # 配置参数
    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    dims = [16, 32, 64, 128]
    dtypes = [torch.bfloat16, torch.float32]
    
    results = []
    
    # 运行所有组合
    for dtype in dtypes:
        for seq_len in seq_lengths:
            for d in dims:
                # 跳过可能 OOM 的配置
                # if seq_len >= 32768 and d >= 128:
                #     print(f"Skipping seq_len={seq_len}, d={d} (likely OOM)")
                #     continue
                
                result = run_benchmark(seq_len, d, dtype, device)
                results.append(result)
                
                # 打印当前结果
                print(f"  PyTorch: fwd={result['pytorch_fwd_ms']:.2f}ms, "
                      f"bwd={result['pytorch_bwd_ms']:.2f}ms, "
                      f"fwd+bwd={result['pytorch_fwd_bwd_ms']:.2f}ms")
                print(f"  Triton:  fwd={result['triton_fwd_ms']:.2f}ms, "
                      f"bwd={result['triton_bwd_ms']:.2f}ms, "
                      f"fwd+bwd={result['triton_fwd_bwd_ms']:.2f}ms")
                print(f"  Speedup: fwd={result['speedup_fwd']:.2f}x, "
                      f"bwd={result['speedup_bwd']:.2f}x, "
                      f"fwd+bwd={result['speedup_fwd_bwd']:.2f}x\n")
    
    # 创建 DataFrame 并保存
    df = pd.DataFrame(results)
    
    # 保存为 CSV
    df.to_csv('flash_attention_benchmark.csv', index=False)
    print("Results saved to flash_attention_benchmark.csv")
    
    # 打印汇总表格
    print("\n" + "="*100)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*100)
    
    # 格式化输出
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.2f}'.format)
    
    print(df.to_string(index=False))
    
    # 打印平均加速比
    print("\n" + "="*100)
    print("AVERAGE SPEEDUPS")
    print("="*100)
    print(f"Forward:          {df['speedup_fwd'].mean():.2f}x")
    print(f"Backward:         {df['speedup_bwd'].mean():.2f}x")
    print(f"Forward+Backward: {df['speedup_fwd_bwd'].mean():.2f}x")

if __name__ == "__main__":
    main()