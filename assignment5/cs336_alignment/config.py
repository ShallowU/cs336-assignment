"""
GRPO (Group Relative Policy Optimization) 训练配置文件

该文件包含所有训练超参数的定义和详细说明。
"""

from typing import List
from dataclasses import dataclass, field
import os


@dataclass
class GRPOConfig:
    """
    GRPO 训练配置类
    
    针对 GSM8K 数据集优化：
    - 训练集：7400+ 条
    - 测试集：1319 条
    - 模型：Qwen2.5-1.5B-Instruct
    - 硬件：A100 80G
    """
    
    # ==================== 实验设置 ====================
    experiment_name: str = "grpo_gsm8k"
    
    # 要运行的 loss_type 列表
    # 建议顺序：先跑最稳定的，再对比其他
    loss_types_to_run: List[str] = field(default_factory=lambda: [
        "grpo_clip",                # 最稳定，适合 off-policy
        "reinforce_with_baseline",  # 经典方法
        "no_baseline",              # 对比基准
    ])
    
    # ==================== 模型路径 ====================
    model_path: str = "./models/Qwen2.5-1.5B-Instruct"
    
    # ==================== 训练循环参数 ====================
    # GRPO 步数计算：
    # - 每步处理 32 个问题（256 / 8）
    # - 100 步 = 3200 个问题
    # - 200 步 = 6400 个问题（覆盖大部分训练集）
    # - 建议：100-150 步，避免过拟合
    n_grpo_steps: int = 100
    
    # ==================== 优化器参数 ====================
    # 学习率：RL 微调建议使用较小值
    # - 太大：策略崩溃，奖励下降
    # - 太小：收敛慢
    # 建议范围：5e-7 ~ 2e-5
    learning_rate: float = 5e-6
    
    # Adam 优化器参数
    # beta2=0.999 对于 RL 更稳定（相比 0.95）
    betas: tuple = (0.9, 0.999)
    
    # 权重衰减：轻微正则化防止过拟合
    weight_decay: float = 0.01
    
    # ==================== 梯度参数 ====================
    max_grad_norm: float = 1.0
    
    # ==================== 学习率调度 ====================
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.1
    
    # ==================== Rollout 采样参数 ====================
    # rollout_batch_size = n_prompts × group_size
    # 
    # 权衡：
    # - 太小：梯度方差大
    # - 太大：显存不足，采样慢
    # 
    # 推荐：128-256
    rollout_batch_size: int = 128
    
    # group_size：每个问题生成的回答数
    # - 太小（2-4）：优势估计不准确
    # - 太大（16+）：计算成本高
    # 推荐：4-8
    group_size: int = 4
    
    # 采样参数
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4
    sampling_max_tokens: int = 1024
    
    # ==================== 训练参数 ====================
    # epochs_per_rollout_batch：
    # - 1: on-policy，每个样本只用一次
    # - 2-4: off-policy，充分利用样本
    # 
    # 注意：off-policy 时必须使用 grpo_clip 以保证稳定性
    epochs_per_rollout_batch: int = 4
    
    # train_batch_size 通常等于 rollout_batch_size
    train_batch_size: int = 128
    
    # gradient_accumulation_steps：
    # micro_batch_size = train_batch_size / gradient_accumulation_steps
    # 
    # A100 80G + 1.5B 模型 + bfloat16：
    # micro_batch_size = 8 应该没问题
    # 
    # 优化器更新次数/epoch = n_microbatches / gradient_accumulation_steps
    #                      = 128 / 8 / 16 = 1（这样不太好）
    # 
    # 改成：gradient_accumulation_steps = 8
    # 优化器更新次数/epoch = 128 / 8 / 8 = 2
    gradient_accumulation_steps: int = 4
    
    # ==================== GPU 资源 ====================
    # A100 80G 可以适当提高
    # 注意：训练时模型占用约 3-6GB，vLLM 可以用更多
    gpu_memory_utilization: float = 0.5
    
    # ==================== 优势函数参数 ====================
    advantage_eps: float = 1e-6
    
    # 是否按标准差归一化
    # True: 更稳定，但可能削弱信号
    # False: 信号更强，但可能不稳定
    use_std_normalization: bool = True
    
    # ==================== GRPO Clip 参数 ====================
    # cliprange：PPO 风格的裁剪范围
    # - 0.1-0.2: 保守，更稳定
    # - 0.2-0.3: 更激进，学习更快
    cliprange: float = 0.2
    
    # ==================== 日志和保存 ====================
    # 评估频率
    # 100 步训练，每 10 步评估一次 = 10 次评估
    eval_every_n_steps: int = 10
    
    save_root_path: str = "/content/drive/MyDrive/cs336/assignment5"
    save_checkpoints: bool = True
    
    # ==================== Prompt 模板 ====================
    prompt_template: str = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""

    def get_save_path(self, loss_type: str) -> str:
        """获取特定 loss_type 的保存路径"""
        return os.path.join(self.save_root_path, f"grpo_{loss_type}_logs")
    
    def __post_init__(self):
        """验证配置的有效性"""
        assert self.rollout_batch_size % self.group_size == 0, \
            f"rollout_batch_size ({self.rollout_batch_size}) must be divisible by group_size ({self.group_size})"
        assert self.train_batch_size % self.gradient_accumulation_steps == 0, \
            f"train_batch_size ({self.train_batch_size}) must be divisible by gradient_accumulation_steps ({self.gradient_accumulation_steps})"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert 0 < self.cliprange < 1, "cliprange must be between 0 and 1"
        
        # 打印关键计算
        n_prompts_per_rollout = self.rollout_batch_size // self.group_size
        micro_batch_size = self.train_batch_size // self.gradient_accumulation_steps
        n_microbatches = self.rollout_batch_size // micro_batch_size
        optimizer_steps_per_epoch = n_microbatches // self.gradient_accumulation_steps
        total_optimizer_steps = self.n_grpo_steps * self.epochs_per_rollout_batch * optimizer_steps_per_epoch
        
        print(f"\n配置验证:")
        print(f"  每次 rollout 问题数: {n_prompts_per_rollout}")
        print(f"  micro_batch_size: {micro_batch_size}")
        print(f"  n_microbatches: {n_microbatches}")
        print(f"  每 epoch 优化器更新次数: {optimizer_steps_per_epoch}")
        print(f"  总优化器更新次数: {total_optimizer_steps}")
        print(f"  总训练问题数: {self.n_grpo_steps * n_prompts_per_rollout}")
        print()


# 向后兼容
_default_config = GRPOConfig()