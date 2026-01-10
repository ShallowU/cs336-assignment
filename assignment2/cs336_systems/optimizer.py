import torch
from collections.abc import Callable, Iterable
from typing import Optional
import math
import numpy as np

class SGD(torch.optim.Optimizer):
    """
    随机梯度下降优化器，带学习率衰减
    公式：θ_{t+1} = θ_t - (lr / sqrt(t+1)) * ∇L(θ_t)
    # 例子：不同层使用不同学习率
    optimizer = SGD([
        {"params": model.layer1.parameters(), "lr": 1e-3},
        {"params": model.layer2.parameters(), "lr": 1e-4},
    ])
    """
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        # 调用父类构造函数
        # 父类会将 params 组织成 param_groups
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        """
        执行一步参数更新
        
        Args:
            closure: 可选的重新计算损失的函数
                     用于某些高级优化算法（LBFGS）
                     是一个可选的函数参数，用于在优化器更新参数之前重新计算损失值。虽然在基本的 SGD 中很少用到，但在某些高级优化算法中是必需的。
        
        Returns:
            loss: 如果提供了 closure，返回损失值
        """
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                # 每个参数 p 是 torch.nn.Parameter 对象
                # # p.data: 参数的实际值（权重张量）
                # print(p.data.shape)  # 例如：torch.Size([512, 256])
                # # p.grad: 参数的梯度（反向传播后填充）
                # print(p.grad.shape)  # 与 p.data 相同
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss

class My_AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), weight_decay=1e-2, eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr":lr, "beta": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            weight_decay = group['weight_decay']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                m = state.get("m", torch.zeros_like(p.grad))
                v = state.get("v", torch.zeros_like(p.grad))
                m = beta[0] * m + (1 - beta[0]) * p.grad.data
                v = beta[1] * v + (1 - beta[1]) * p.grad.data ** 2
                t = state.get("t",1)
                lr_t = lr * ((1 - beta[1] ** t) ** 0.5) / (1 - beta[0] ** t)
                p.data -= lr_t * m / (v ** 0.5 + eps)
                p.data -= lr * weight_decay * p.data
                state["m"] = m
                state["v"] = v
                # 注意这里要自增1
                state["t"] = t + 1

def My_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        lr = (it / warmup_iters) * max_learning_rate
    elif it >= warmup_iters and it <= cosine_cycle_iters:
        lr = min_learning_rate + 0.5 * (1 + math.cos((it - warmup_iters) * math.pi / (cosine_cycle_iters - warmup_iters))) * (max_learning_rate - min_learning_rate)
    elif it > cosine_cycle_iters:
        lr = min_learning_rate
    return lr

def My_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    # 注意l2 norm的计算方法是平方和开根号，并且是在梯度上做计算，不是参数本身
    total_norm = 0
    for param in parameters:
        if param.grad is not None:
            total_norm += ((param.grad.data) ** 2).sum().item()
    total_norm = total_norm ** 0.5
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for param in parameters:
            if param.grad is not None:
                param.grad.data = param.grad.data * scale
    return total_norm * scale if total_norm > max_l2_norm else total_norm


