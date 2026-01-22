import torch
import time
import itertools
import torch.multiprocessing as mp # 多进程，并行
import torch.distributed as dist
import os

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend=backend,rank=rank,world_size=world_size)
def distributed(rank, world_size, backend, data_size, result_queue):
    setup(rank, world_size, backend)
    if backend == 'gloo':
        device = 'cpu'
    elif backend == 'nccl':
        device = f"cuda:{rank}"
    data = torch.randint(0, 10, (int((data_size * 1e6) // 4),), device=device)
    start = time.time()
    dist.all_reduce(data, async_op=False)
    end = time.time()
    result_queue.put(end - start)


def benchmarking(backend, process, data_size):
    world_size = process
    result_queue = mp.Queue()
    mp.spawn(fn=distributed, args=(world_size,backend,data_size,result_queue), nprocs=world_size, join=True)
    times = []
    for _ in range(world_size):
        times.append(result_queue.get())
    avg_time = sum(times) / len(times)
    print(f"Backend: {backend}, Processes: {process}, Data size: {data_size}MB")
    print(f"Times: {[f'{t*1000:.2f}ms' for t in times]}")
    print(f"Average time: {avg_time*1000:.2f}ms\n")
if __name__ == "__main__":
    backends = ["gloo"]
    processes = [2, 4, 6]
    datasizes = [1, 10, 100, 1000] # MB
    for backend, process, data_size in itertools.product(backends, processes, datasizes):
        try:
            benchmarking(backend, process, data_size)
        except Exception as e:
            print(e)