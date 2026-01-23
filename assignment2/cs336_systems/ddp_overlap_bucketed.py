import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Tuple, Optional
import threading

# ============ 反向传播阶段 ============
# t0: loss.backward() 开始

# t1: Layer2 (param_5, param_4) 梯度计算完成
#     → _grad_hook 触发
#     → Bucket 0 就绪
#     → flatten [param_5.grad, param_4.grad]
#     → 启动异步 all_reduce (Bucket 0)  ← 通信开始

# t2: Layer1 (param_3, param_2) 梯度计算完成
#     → Bucket 1 就绪
#     → 启动异步 all_reduce (Bucket 1)  ← 通信开始
#     （Bucket 0 通信仍在进行中）

# t3: Layer0 (param_1, param_0) 梯度计算完成
#     → Bucket 2 就绪
#     → 启动异步 all_reduce (Bucket 2)
#     （Bucket 0, 1, 2 可能同时在通信）

# # ============ 同步等待阶段 ============
# t4: finish_gradient_synchronization() 被调用
#     → 等待 Bucket 0 通信完成 → 平均 → 解扁平化 → 写回
#     → 等待 Bucket 1 通信完成 → 平均 → 解扁平化 → 写回
#     → 等待 Bucket 2 通信完成 → 平均 → 解扁平化 → 写回

# ============ 优化器更新阶段 ============
# t5: optimizer.step()
class DDPBucketed(nn.Module):
    """
    Distributed Data Parallel with bucketed gradient communication.
    
    This class implements gradient bucketing to reduce the number of communication
    calls while still overlapping communication with computation.
    """
    
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb
        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)  # Convert MB to bytes
        
        # Storage for buckets and communication handles
        self.buckets: List[List[torch.nn.Parameter]] = [] # List of buckets, each bucket is a list of parameters
        self.bucket_tensors: List[torch.Tensor] = [] # Flattened gradient tensors for each bucket
        self.communication_handles: List[dist.Work] = [] # List of (handle, bucket_idx, original_grads)
        self.bucket_ready: List[bool] = [] # Flags to indicate if a bucket's communication has started
        self.param_to_bucket: dict = {} # Map from parameter to its bucket index
        
        # Initialize parameter synchronization and bucket setup
        self._initialize_parameters()
        self._create_buckets()
        self._register_hooks()
        
    def _initialize_parameters(self):
        """Broadcast parameters from rank 0 to all other ranks to ensure consistency."""
        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0) # Broadcast from rank 0
    
    def _create_buckets(self):
        """
        Create buckets of parameters based on the bucket size limit.
        Parameters are allocated in reverse order since gradients become ready
        in approximately that order during backward pass.
        """
        # Get parameters in reverse order
        params = list(self.module.parameters())[::-1]
        
        current_bucket = []
        current_bucket_size = 0
        
        for param in params:
            if param.requires_grad:
                param_size = param.numel() * param.element_size()
                
                # If adding this parameter would exceed bucket size and current bucket is not empty,
                # finalize current bucket and start a new one
                if current_bucket_size + param_size > self.bucket_size_bytes and current_bucket:
                    self._finalize_bucket(current_bucket)
                    current_bucket = []
                    current_bucket_size = 0
                
                current_bucket.append(param)
                current_bucket_size += param_size
                # Map parameter to the bucket it will be in (current bucket index)
                self.param_to_bucket[param] = len(self.buckets)
        
        # Add the last bucket if it's not empty
        if current_bucket:
            self._finalize_bucket(current_bucket)
    
    def _finalize_bucket(self, bucket: List[torch.nn.Parameter]):
        """Finalize a bucket by adding it to the bucket list and initializing tracking variables."""
        self.buckets.append(bucket)
        self.bucket_ready.append(False)
        
        # Create a flattened tensor for this bucket
        # We'll create this when gradients are ready to save memory
        self.bucket_tensors.append(None)
    
    def _register_hooks(self):
        """Register post-accumulate gradient hooks for all parameters."""
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._grad_hook)
    
    def _grad_hook(self, param: torch.nn.Parameter):
        """
        Hook called when a parameter's gradient is ready.
        Checks if the bucket is ready and initiates communication if so.
        """
        bucket_idx = self.param_to_bucket[param]
        bucket = self.buckets[bucket_idx] # 包含该参数的 bucket
        
        # Check if all gradients in this bucket are ready
        if self._is_bucket_ready(bucket):
            self._start_bucket_communication(bucket_idx)
    
    def _is_bucket_ready(self, bucket: List[torch.nn.Parameter]) -> bool:
        """Check if all parameters in a bucket have their gradients ready."""
        for param in bucket:
            if param.grad is None:
                return False
        return True
    
    def _start_bucket_communication(self, bucket_idx: int):
        """
        Start asynchronous all-reduce communication for a bucket.
        This flattens the gradients in the bucket and initiates all-reduce.
        """
        if self.bucket_ready[bucket_idx]:
            return  # Already started communication for this bucket
        
        bucket = self.buckets[bucket_idx]
        
        # Flatten gradients in this bucket
        grads = [param.grad for param in bucket if param.grad is not None]
        if not grads:
            return
        
        # Use PyTorch's internal flattening utilities
        flattened_grad = torch._utils._flatten_dense_tensors(grads)
        self.bucket_tensors[bucket_idx] = flattened_grad
        
        # Start asynchronous all-reduce
        handle = dist.all_reduce(flattened_grad, op=dist.ReduceOp.SUM, async_op=True)
        self.communication_handles.append((handle, bucket_idx, grads))
        self.bucket_ready[bucket_idx] = True
    
    def finish_gradient_synchronization(self):
        """
        Wait for all asynchronous communication to complete and update parameter gradients.
        This should be called after the backward pass but before the optimizer step.
        """
        # Wait for all communication handles to complete
        for handle, bucket_idx, original_grads in self.communication_handles:
            handle.wait()
            
            # Average the gradients
            flattened_grad = self.bucket_tensors[bucket_idx]
            flattened_grad.div_(dist.get_world_size())
            
            # Unflatten and assign back to parameters
            unflattened_grads = torch._utils._unflatten_dense_tensors(flattened_grad, original_grads)
            bucket = self.buckets[bucket_idx]
            
            grad_idx = 0
            for param in bucket:
                if param.grad is not None:
                    param.grad.copy_(unflattened_grads[grad_idx])
                    grad_idx += 1
        
        # Clear communication handles and reset bucket status
        self.communication_handles.clear()
        self.bucket_ready = [False] * len(self.buckets)
        self.bucket_tensors = [None] * len(self.buckets)
    
    def forward(self, *inputs, **kwargs):
        """Forward pass through the wrapped module."""
        return self.module(*inputs, **kwargs)
    
    def parameters(self, recurse: bool = True):
        """Return parameters of the wrapped module."""
        return self.module.parameters(recurse)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """Return named parameters of the wrapped module."""
        return self.module.named_parameters(prefix, recurse)
    
    def state_dict(self, *args, **kwargs):
        """Return state dict of the wrapped module."""
        return self.module.state_dict(*args, **kwargs)
    
    def load_state_dict(self, *args, **kwargs):
        """Load state dict into the wrapped module."""
        return self.module.load_state_dict(*args, **kwargs)