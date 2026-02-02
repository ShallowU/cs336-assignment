"""
GRPO 训练工具函数模块

包含以下核心功能：
1. 数据预处理：tokenize_prompt_and_output
2. 概率计算：get_response_log_probs
3. 优势函数计算：compute_group_normalized_rewards
4. 策略梯度损失计算：compute_policy_gradient_loss
5. 学习率调度：使用 PyTorch 原生实现
6. 梯度裁剪：使用 PyTorch 原生实现
"""

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from typing import Dict, List, Tuple, Optional, Callable
import math


# ==================== 数据预处理 ====================

def tokenize_prompt_and_output(
    prompt_strs: List[str], 
    output_strs: List[str], 
    tokenizer
) -> Dict[str, torch.Tensor]:
    """
    将 prompt 和 output 字符串转换为模型输入格式
    
    核心思想：
    1. 将 prompt 和 output 拼接并 tokenize
    2. 创建 response_mask 标记哪些 token 是模型生成的（需要计算损失）
    3. 进行 padding 使 batch 内长度一致
    
    Args:
        prompt_strs: prompt 字符串列表，长度为 batch_size
        output_strs: 模型生成的回答字符串列表，长度为 batch_size
        tokenizer: HuggingFace tokenizer
    
    Returns:
        包含以下键的字典：
        - input_ids: (batch_size, max_len - 1) 输入 token IDs
        - labels: (batch_size, max_len - 1) 目标 token IDs（向右移动一位）
        - response_mask: (batch_size, max_len - 1) 响应部分的掩码
    
    示例：
        prompt: "What is 2+2?"  -> tokens: [101, 102, 103, 104, 105]
        output: "The answer is 4" -> tokens: [201, 202, 203, 204]
        
        拼接后: [101, 102, 103, 104, 105, 201, 202, 203, 204]
        
        input_ids: [101, 102, 103, 104, 105, 201, 202, 203]  # 去掉最后一个
        labels:    [102, 103, 104, 105, 201, 202, 203, 204]  # 去掉第一个
        
        response_mask: [0, 0, 0, 0, 1, 1, 1, 1]  # 只在 output 部分为 1
        注意：response_mask 的第一个 1 对应 labels 中 output 的第一个 token
    """
    batch_size = len(prompt_strs)
    
    # 存储每个样本的信息
    all_input_ids = []
    all_prompt_lens = []
    all_total_lens = []
    
    # 第一遍：tokenize 并计算长度
    for i in range(batch_size):
        # 分别 tokenize prompt 和 output
        # add_special_tokens=False 避免添加额外的 [CLS]、[SEP] 等
        prompt_ids = tokenizer.encode(prompt_strs[i], add_special_tokens=False)
        output_ids = tokenizer.encode(output_strs[i], add_special_tokens=False)
        
        # 拼接成完整序列
        full_ids = prompt_ids + output_ids
        
        all_input_ids.append(full_ids)
        all_prompt_lens.append(len(prompt_ids))
        all_total_lens.append(len(full_ids))
    
    # 计算 batch 内的最大长度
    max_len = max(all_total_lens)
    
    # 第二遍：padding 并创建 mask
    input_ids_batch = []
    labels_batch = []
    response_mask_batch = []
    
    for i in range(batch_size):
        full_ids = all_input_ids[i]
        prompt_len = all_prompt_lens[i]
        total_len = all_total_lens[i]
        
        # 计算需要 padding 的长度
        padding_len = max_len - total_len
        
        # Padding 到最大长度
        if padding_len > 0:
            padded_ids = full_ids + [tokenizer.pad_token_id] * padding_len
        else:
            padded_ids = full_ids
        
        # 创建 input_ids 和 labels
        # input_ids: 去掉最后一个 token（作为输入）
        # labels: 去掉第一个 token（作为预测目标）
        input_ids = padded_ids[:-1]  # [0, max_len-2]
        labels = padded_ids[1:]      # [1, max_len-1]
        
        # 创建 response_mask
        # 只在 output 部分为 1，prompt 和 padding 部分为 0
        # 
        # 详细解释：
        # 假设 prompt_len = 5, total_len = 9, max_len = 10
        # 
        # padded_ids: [p0, p1, p2, p3, p4, o0, o1, o2, o3, PAD]
        #              0   1   2   3   4   5   6   7   8   9
        #
        # input_ids:  [p0, p1, p2, p3, p4, o0, o1, o2, o3]
        #              0   1   2   3   4   5   6   7   8
        #
        # labels:     [p1, p2, p3, p4, o0, o1, o2, o3, PAD]
        #              0   1   2   3   4   5   6   7   8
        #
        # 我们想要计算损失的是 output 部分的 token
        # 在 labels 中，output 从 index=4 开始（即 prompt_len - 1）
        # 到 index=7 结束（即 total_len - 2）
        # 
        # response_mask: [0, 0, 0, 0, 1, 1, 1, 1, 0]
        #                 0  1  2  3  4  5  6  7  8
        
        mask = [0.0] * (max_len - 1)
        
        # output 部分的起始位置（在 labels 中）
        output_start = prompt_len - 1
        # output 部分的结束位置（不包含 padding）
        output_end = total_len - 1
        
        for j in range(output_start, output_end):
            mask[j] = 1.0
        
        input_ids_batch.append(input_ids)
        labels_batch.append(labels)
        response_mask_batch.append(mask)
    
    return {
        "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
        "labels": torch.tensor(labels_batch, dtype=torch.long),
        "response_mask": torch.tensor(response_mask_batch, dtype=torch.bool)
    }


# ==================== 概率和熵计算 ====================

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    计算 token 级别的熵
    
    熵是衡量模型预测不确定性的指标：
    - 高熵：模型对预测不确定，概率分布较平坦
    - 低熵：模型对预测很确定，概率集中在少数 token
    
    在 RL 训练中，监控熵可以帮助判断：
    - 熵过低：模型可能过拟合，缺乏探索
    - 熵过高：模型可能学习不够
    
    公式：H(p) = -Σ p(x) * log(p(x))
    
    Args:
        logits: (batch_size, seq_len, vocab_size) 模型输出的 logits
    
    Returns:
        (batch_size, seq_len) 每个位置的熵
    """
    with torch.no_grad():
        # 使用 log_softmax 比 softmax + log 更数值稳定
        # log_softmax 内部使用 logsumexp 技巧，避免数值上溢/下溢
        log_prob = F.log_softmax(logits, dim=-1)
        prob = torch.exp(log_prob)
        
        # 熵 = -Σ p * log(p)
        entropy = -(prob * log_prob).sum(dim=-1)
    
    return entropy


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False
) -> Dict[str, torch.Tensor]:
    """
    计算模型对 response token 的对数概率
    
    这是策略梯度算法的核心计算：
    - log π(a|s)：在给定 prompt 下生成特定 response 的对数概率
    
    Args:
        model: 语言模型
        input_ids: (batch_size, seq_len) 输入 token IDs
        labels: (batch_size, seq_len) 目标 token IDs
        return_token_entropy: 是否同时返回 token 熵
    
    Returns:
        包含 'log_probs' 的字典，可选 'token_entropy'
    """
    # 前向传播获取 logits
    logits = model(input_ids).logits  # (batch_size, seq_len, vocab_size)
    
    # 计算所有 token 的对数概率
    log_probs = F.log_softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)
    
    # 提取目标 token 的对数概率
    # gather 操作：从 log_probs 中按 labels 索引提取值
    # labels.unsqueeze(-1): (batch_size, seq_len, 1)
    # gather 结果: (batch_size, seq_len, 1)
    # squeeze(-1): (batch_size, seq_len)
    token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    
    result = {"log_probs": token_log_probs}
    
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    
    return result


# ==================== 优势函数计算 ====================

def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: List[str],
    repeated_ground_truths: List[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    计算组内归一化的优势函数
    
    GRPO 的核心思想：
    1. 对每个问题生成 group_size 个回答
    2. 计算每个回答的奖励
    3. 计算组内的相对优势（相对于组内平均奖励）
    
    优势函数的作用：
    - 正优势：鼓励这个行为（比平均更好）
    - 负优势：抑制这个行为（比平均更差）
    - 零优势：维持当前概率
    
    使用组内归一化的好处：
    - 降低方差：每个组独立归一化
    - 相对评估：不同难度的问题可比
    
    Args:
        reward_fn: 奖励函数，接受 (response, ground_truth)，返回包含 'reward' 的字典
        rollout_responses: 所有生成的回答，长度为 n_prompts * group_size
        repeated_ground_truths: 重复的标准答案，与 rollout_responses 对应
        group_size: 每个问题的回答数量
        advantage_eps: 防止除零的小常数
        normalize_by_std: 是否按标准差归一化
    
    Returns:
        - normalized_rewards: 归一化后的优势函数
        - unnormalized_rewards: 原始奖励
        - info: 额外信息（如 format_rewards）
    
    示例：
        假设有 2 个问题，每个问题 4 个回答：
        
        问题 1 的奖励: [1.0, 0.0, 1.0, 0.0]
        问题 1 的均值: 0.5
        问题 1 的优势: [0.5, -0.5, 0.5, -0.5]
        
        问题 2 的奖励: [0.0, 0.0, 0.0, 1.0]
        问题 2 的均值: 0.25
        问题 2 的优势: [-0.25, -0.25, -0.25, 0.75]
    """
    n_prompts = len(rollout_responses) // group_size
    
    all_rewards = []
    all_format_rewards = []
    all_advantages = []
    
    # 对每个问题单独计算优势
    for i in range(n_prompts):
        group_rewards = []
        
        # 收集组内所有回答的奖励
        for j in range(group_size):
            idx = i * group_size + j
            result = reward_fn(
                rollout_responses[idx], 
                repeated_ground_truths[idx]
            )
            group_rewards.append(result['reward'])
            all_format_rewards.append(result['format_reward'])
        
        all_rewards.extend(group_rewards)
        
        # 计算组内优势
        group_rewards_tensor = torch.tensor(group_rewards, dtype=torch.float32)
        group_mean = group_rewards_tensor.mean()
        advantages = group_rewards_tensor - group_mean
        
        # 可选：按标准差归一化
        if normalize_by_std:
            group_std = advantages.std()
            advantages = advantages / (group_std + advantage_eps)
        
        all_advantages.extend(advantages.tolist())
    
    return (
        torch.tensor(all_advantages, dtype=torch.float32),
        torch.tensor(all_rewards, dtype=torch.float32),
        {"format_rewards": torch.tensor(all_format_rewards, dtype=torch.float32)}
    )


# ==================== 策略梯度损失计算 ====================

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor
) -> torch.Tensor:
    """
    计算基本的策略梯度损失（REINFORCE）
    
    公式：L = -E[R * log π(a|s)]
    
    直觉解释：
    - 如果奖励 R > 0，我们想增加 log π，所以 loss = -R * log π
    - 如果奖励 R < 0，我们想减少 log π，loss 仍然正确处理
    
    Args:
        raw_rewards_or_advantages: 奖励或优势值
        policy_log_probs: 策略的对数概率
    
    Returns:
        策略梯度损失（未取均值）
    """
    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float
) -> Tuple[torch.Tensor, Dict]:
    """
    计算 GRPO Clip 损失（类似 PPO）
    
    PPO 的核心思想是限制策略更新的幅度，防止更新过大导致训练不稳定。
    
    公式：
        ratio = exp(log π_new - log π_old) = π_new / π_old
        clipped_ratio = clip(ratio, 1 - ε, 1 + ε)
        loss = -min(ratio * A, clipped_ratio * A)
    
    解释：
    1. ratio 表示新旧策略的概率比
    2. 如果 ratio 偏离 1 太多（策略变化太大），就裁剪它
    3. 取 min 确保我们不会过度优化
    
    Args:
        advantages: 优势值
        policy_log_probs: 当前策略的对数概率
        old_log_probs: 旧策略的对数概率
        cliprange: 裁剪范围 ε
    
    Returns:
        裁剪后的策略梯度损失和额外信息
    """
    # 计算概率比 ratio = π_new / π_old
    # 使用 exp(log π_new - log π_old) 避免直接计算概率（数值更稳定）
    ratio = torch.exp(policy_log_probs - old_log_probs)
    
    # 裁剪 ratio
    clipped_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
    
    # 计算两种目标
    objective = ratio * advantages
    clipped_objective = clipped_ratio * advantages
    
    # 取 min（对应 max 损失，因为我们要最小化 loss）
    # 这里的直觉是：
    # - 当 advantage > 0（好行为），我们想增加概率，但不要增加太多
    # - 当 advantage < 0（坏行为），我们想减少概率，但不要减少太多
    loss = -torch.min(objective, clipped_objective)
    
    # 额外信息：用于监控
    info = {
        "ratio_mean": ratio.mean().item(),
        "ratio_max": ratio.max().item(),
        "ratio_min": ratio.min().item(),
        "clipped_fraction": ((ratio < 1 - cliprange) | (ratio > 1 + cliprange)).float().mean().item()
    }
    
    return loss, info


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: Optional[torch.Tensor],
    cliprange: float
) -> Tuple[torch.Tensor, Dict]:
    """
    根据指定的类型计算策略梯度损失
    
    三种损失类型对比：
    
    1. no_baseline:
       - 使用原始奖励作为权重
       - 方差大，但无偏
       - 适合：简单任务，奖励分布窄
    
    2. reinforce_with_baseline:
       - 使用优势函数（减去组内均值）
       - 方差较小，仍然无偏
       - 适合：大多数场景
    
    3. grpo_clip:
       - 加入 PPO 风格的概率比裁剪
       - 更稳定，防止策略崩溃
       - 适合：off-policy 训练，需要多次使用同一批数据
    
    Args:
        policy_log_probs: 当前策略的对数概率
        loss_type: 损失类型
        raw_rewards: 原始奖励
        advantages: 优势值
        old_log_probs: 旧策略的对数概率（grpo_clip 需要）
        cliprange: 裁剪范围
    
    Returns:
        损失值和额外信息
    """
    if loss_type == "no_baseline":
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    
    elif loss_type == "reinforce_with_baseline":
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    
    elif loss_type == "grpo_clip":
        return compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange
        )
    
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


# ==================== 辅助函数 ====================

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[int] = None
) -> torch.Tensor:
    """
    计算带掩码的均值
    
    只对 mask=1 的位置计算均值，忽略 mask=0 的位置
    
    Args:
        tensor: 输入张量
        mask: 掩码张量（1 表示有效，0 表示忽略）
        dim: 求均值的维度
    
    Returns:
        带掩码的均值
    """
    tensor = tensor * mask.float()
    
    if dim is not None:
        return tensor.sum(dim=dim) / mask.sum(dim=dim).clamp(min=1)
    else:
        return tensor.sum() / mask.sum().clamp(min=1)


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float
) -> Tuple[torch.Tensor, Dict]:
    """
    执行一个 micro-batch 的训练步骤
    
    这是训练循环的核心函数，完成以下工作：
    1. 计算策略梯度损失
    2. 应用 response mask（只在生成部分计算损失）
    3. 缩放损失（用于梯度累积）
    4. 反向传播
    
    Args:
        policy_log_probs: 当前策略的对数概率
        response_mask: 响应部分的掩码
        gradient_accumulation_steps: 梯度累积步数
        loss_type: 损失类型
        raw_rewards: 原始奖励
        advantages: 优势值
        old_log_probs: 旧策略的对数概率
        cliprange: 裁剪范围
    
    Returns:
        损失值和额外信息
    """
    # 计算每个 token 的损失
    token_loss, info = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange
    )
    
    # 应用掩码，只计算 response 部分的损失
    # 对每个序列求均值，然后对 batch 求均值
    loss = masked_mean(token_loss, response_mask, dim=-1).mean()
    
    # 缩放损失用于梯度累积
    # 因为 loss.backward() 会累积梯度
    # 所以要除以 gradient_accumulation_steps 使最终梯度正确
    loss = loss / gradient_accumulation_steps
    
    # 反向传播
    loss.backward()
    
    return loss * gradient_accumulation_steps, info  # 返回未缩放的 loss 用于日志


# ==================== 学习率调度（使用 PyTorch 原生） ====================

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0
) -> LambdaLR:
    """
    创建带预热的余弦学习率调度器
    
    学习率变化过程：
    1. 预热阶段：从 0 线性增加到 max_lr
    2. 余弦衰减阶段：从 max_lr 余弦衰减到 min_lr
    
    使用 PyTorch 原生的 LambdaLR，比自定义函数更高效
    
    Args:
        optimizer: 优化器
        num_warmup_steps: 预热步数
        num_training_steps: 总训练步数
        min_lr_ratio: 最终学习率与初始学习率的比值
    
    Returns:
        LambdaLR 调度器
    """
    def lr_lambda(current_step: int) -> float:
        # 预热阶段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # 余弦衰减阶段
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            min_lr_ratio,
            min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        )
    
    return LambdaLR(optimizer, lr_lambda)


# ==================== 梯度裁剪（使用 PyTorch 原生） ====================

def clip_grad_norm(
    model: torch.nn.Module,
    max_norm: float
) -> float:
    """
    裁剪模型梯度的 L2 范数
    
    直接使用 PyTorch 的 clip_grad_norm_，更高效且经过优化
    
    Args:
        model: 模型
        max_norm: 最大梯度范数
    
    Returns:
        裁剪前的梯度范数
    """
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(), 
        max_norm, 
        error_if_nonfinite=False  # 允许 inf/nan（会被裁剪处理）
    ).item()


# ==================== SFT 训练步骤（保留兼容性） ====================

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0
) -> Tuple[torch.Tensor, Dict]:
    """
    监督微调（SFT）的训练步骤
    
    SFT 损失就是标准的交叉熵损失（负对数似然）
    
    Args:
        policy_log_probs: 对数概率
        response_mask: 响应掩码
        gradient_accumulation_steps: 梯度累积步数
        normalize_constant: 归一化常数
    
    Returns:
        损失值和额外信息
    """
    # SFT loss = -mean(log_probs)
    loss = masked_mean(-policy_log_probs, response_mask, dim=-1).mean()
    loss = loss / gradient_accumulation_steps
    loss.backward()
    
    return loss * gradient_accumulation_steps, {}


# ==================== 向后兼容的函数 ====================

def My_gradient_clipping(parameters, max_l2_norm: float) -> float:
    """向后兼容的梯度裁剪函数，内部使用 PyTorch 原生实现"""
    return torch.nn.utils.clip_grad_norm_(
        parameters, 
        max_l2_norm, 
        error_if_nonfinite=False
    ).item()


def My_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """向后兼容的学习率计算函数"""
    if it < warmup_iters:
        return (it / max(1, warmup_iters)) * max_learning_rate
    elif it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / max(1, cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: Optional[int]
) -> torch.Tensor:
    """向后兼容的掩码归一化函数"""
    masked_tensor = tensor * mask.float()
    if dim is not None:
        return masked_tensor.sum(dim=dim) / normalize_constant
    else:
        return masked_tensor.sum() / normalize_constant