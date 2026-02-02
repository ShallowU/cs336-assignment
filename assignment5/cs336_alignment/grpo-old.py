"""
GRPO (Group Relative Policy Optimization) 训练脚本

本脚本实现了完整的 GRPO 训练流程，支持三种不同的策略梯度损失函数：
1. no_baseline: 使用原始奖励的 REINFORCE
2. reinforce_with_baseline: 使用组内归一化优势的 REINFORCE
3. grpo_clip: 带 PPO 风格裁剪的 GRPO

训练流程：
1. 加载模型和数据
2. 对每个问题采样多个回答（rollout）
3. 计算奖励和优势函数
4. 使用策略梯度更新模型
5. 重复以上步骤

主要优化：
1. 使用 vLLM 加速推理
2. 使用 Flash Attention 2 加速训练
3. 梯度累积减少显存使用
4. Off-policy 训练充分利用样本
"""

import torch
import torch.nn as nn
import json
import logging
import os
import gc
import random
from typing import List, Dict, Optional, Tuple
from dataclasses import asdict

from transformers import AutoModelForCausalLM, AutoTokenizer
from tensorboardX import SummaryWriter
from vllm import LLM, SamplingParams
from tqdm import tqdm

# 导入本地模块
try:
    from drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
    import utils
    from math_baseline import evaluate
    from config import GRPOConfig
except ImportError:
    from .drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
    from . import utils
    from .math_baseline import evaluate
    from .config import GRPOConfig


# ==================== 配置日志 ====================
logging.getLogger("vllm").setLevel(logging.WARNING)
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ==================== 核心函数 ====================

def load_policy_into_vllm_instance(policy: nn.Module, llm: LLM) -> None:
    """
    将训练中的模型权重加载到 vLLM 实例中
    
    vLLM 维护了自己的模型副本用于高效推理。
    每次 rollout 前需要同步最新的模型权重。
    
    这个函数来自 HuggingFace TRL 库。
    
    Args:
        policy: 训练中的 PyTorch 模型
        llm: vLLM 实例
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def get_responses(
    vllm_model: LLM,
    prompts: List[str],
    sampling_params: SamplingParams
) -> List[List]:
    """
    使用 vLLM 批量生成回答
    
    vLLM 使用 PagedAttention 和连续批处理，
    比 HuggingFace 的 generate 快 10-100 倍。
    
    Args:
        vllm_model: vLLM 模型实例
        prompts: prompt 列表
        sampling_params: 采样参数（温度、最大长度等）
    
    Returns:
        每个 prompt 的多个回答（由 sampling_params.n 决定）
    """
    outputs = vllm_model.generate(prompts, sampling_params)
    return [output.outputs for output in outputs]


def load_gsm8k_dataset(data_path: str, prompt_template: str) -> Tuple[List[str], List[str]]:
    """
    加载 GSM8K 数据集
    
    GSM8K 是一个小学数学问题数据集，包含约 7500 个训练样本。
    每个样本包含问题和详细解答，答案以 "####" 分隔。
    
    Args:
        data_path: 数据文件路径（JSONL 格式）
        prompt_template: prompt 模板
    
    Returns:
        - prompts: 格式化后的 prompt 列表
        - answers: 标准答案列表
    """
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    prompts = []
    answers = []
    
    for item in data:
        # 格式化 prompt
        prompt = prompt_template.format(question=item['question'])
        prompts.append(prompt)
        
        # 提取最终答案（在 "####" 之后）
        answer_text = item['answer']
        final_answer = answer_text[answer_text.find("####") + 5:].strip()
        answers.append(final_answer)
    
    return prompts, answers


def run_single_experiment(
    config: GRPOConfig,
    loss_type: str,
    model: nn.Module,
    tokenizer,
    llm: LLM,
    train_prompts: List[str],
    train_answers: List[str],
    reward_fn,
    device: str = 'cuda'
) -> Dict:
    """运行单个损失类型的完整训练实验"""
    
    print(f"\n{'='*60}")
    print(f"开始训练: loss_type = {loss_type}")
    print(f"{'='*60}\n")
    
    # ==================== 初始化 ====================
    save_path = config.get_save_path(loss_type)
    os.makedirs(save_path, exist_ok=True)
    
    writer = SummaryWriter(save_path)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay
    )
    
    sampling_params = SamplingParams(
        temperature=config.sampling_temperature,
        min_tokens=config.sampling_min_tokens,
        max_tokens=config.sampling_max_tokens,
        n=config.group_size,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )
    
    # ==================== 关键计算 ====================
    n_prompts_per_rollout = config.rollout_batch_size // config.group_size
    micro_batch_size = config.train_batch_size // config.gradient_accumulation_steps
    n_microbatches = config.rollout_batch_size // micro_batch_size
    optimizer_steps_per_epoch = n_microbatches // config.gradient_accumulation_steps
    
    # 修正：总优化器步数应该考虑每个 epoch 的实际更新次数
    total_optimizer_steps = config.n_grpo_steps * config.epochs_per_rollout_batch * optimizer_steps_per_epoch
    warmup_steps = int(total_optimizer_steps * config.warmup_ratio)
    
    print(f"训练参数:")
    print(f"  - 每次 rollout 问题数: {n_prompts_per_rollout}")
    print(f"  - micro_batch_size: {micro_batch_size}")
    print(f"  - n_microbatches: {n_microbatches}")
    print(f"  - 每 epoch 优化器更新次数: {optimizer_steps_per_epoch}")
    print(f"  - 总优化器更新步数: {total_optimizer_steps}")
    print(f"  - warmup 步数: {warmup_steps}")
    
    scheduler = utils.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
        min_lr_ratio=config.min_lr_ratio
    )
    
    # 统计信息
    global_step = 0
    optimizer_step = 0  # 单独跟踪优化器更新步数
    best_val_accuracy = 0.0
    training_stats = {
        'train_rewards': [],
        'val_rewards': [],
        'losses': []
    }
    
    # ==================== 训练循环 ====================
    for grpo_step in range(config.n_grpo_steps):
        print(f"\n--- GRPO Step {grpo_step + 1}/{config.n_grpo_steps} ---")
        
        # 1. 同步模型权重到 vLLM
        load_policy_into_vllm_instance(model, llm)
        
        # 2. 定期验证（在 rollout 之前，确保使用最新权重）
        if grpo_step % config.eval_every_n_steps == 0:
            print("运行验证...")
            model.eval()
            with torch.no_grad():
                val_accuracy, val_format_reward = evaluate(
                    model_path=None,
                    llm=llm,
                    rl=True,
                    reward_fn=reward_fn,
                    prompt=config.prompt_template
                )
            model.train()
            
            writer.add_scalar("val/accuracy", val_accuracy, grpo_step)
            writer.add_scalar("val/format_reward", val_format_reward, grpo_step)
            training_stats['val_rewards'].append(val_accuracy)
            
            print(f"验证准确率: {val_accuracy:.4f}, 格式奖励: {val_format_reward:.4f}")
            
            if val_accuracy > best_val_accuracy and config.save_checkpoints:
                best_val_accuracy = val_accuracy
                best_model_path = os.path.join(save_path, "best_model")
                os.makedirs(best_model_path, exist_ok=True)
                model.save_pretrained(best_model_path)
                tokenizer.save_pretrained(best_model_path)
                print(f"保存最佳模型，准确率: {val_accuracy:.4f}")
        
        # 3. 采样训练数据
        indices = random.sample(range(len(train_prompts)), n_prompts_per_rollout)
        batch_prompts = [train_prompts[i] for i in indices]
        batch_answers = [train_answers[i] for i in indices]
        
        # 4. Rollout: 生成回答
        print("生成回答中...")
        response_outputs = get_responses(llm, batch_prompts, sampling_params)
        
        all_responses = []
        repeated_prompts = []
        repeated_answers = []
        
        for i, outputs in enumerate(response_outputs):
            for output in outputs:
                all_responses.append(output.text)
                repeated_prompts.append(batch_prompts[i])
                repeated_answers.append(batch_answers[i])
        
        # 5. 计算奖励和优势
        advantages, raw_rewards, reward_info = utils.compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=all_responses,
            repeated_ground_truths=repeated_answers,
            group_size=config.group_size,
            advantage_eps=config.advantage_eps,
            normalize_by_std=config.use_std_normalization
        )
        
        mean_reward = raw_rewards.mean().item()
        mean_format_reward = reward_info['format_rewards'].mean().item()
        training_stats['train_rewards'].append(mean_reward)
        
        print(f"平均奖励: {mean_reward:.4f}, 格式奖励: {mean_format_reward:.4f}")
        writer.add_scalar("train/reward", mean_reward, grpo_step)
        writer.add_scalar("train/format_reward", mean_format_reward, grpo_step)
        
        # 6. 准备训练数据
        train_batch = utils.tokenize_prompt_and_output(
            repeated_prompts, 
            all_responses, 
            tokenizer
        )
        
        # 7. 计算旧策略的对数概率
        old_log_probs_list = []
        
        model.eval()
        with torch.no_grad():
            for mb_idx in range(n_microbatches):
                start_idx = mb_idx * micro_batch_size
                end_idx = start_idx + micro_batch_size
                
                input_ids = train_batch['input_ids'][start_idx:end_idx].to(device)
                labels = train_batch['labels'][start_idx:end_idx].to(device)
                
                old_log_probs = utils.get_response_log_probs(model, input_ids, labels)['log_probs']
                old_log_probs_list.append(old_log_probs.detach())  # 添加 detach()
        model.train()
        
        # 8. 多 epoch 训练
        for epoch in range(config.epochs_per_rollout_batch):
            print(f"  Epoch {epoch + 1}/{config.epochs_per_rollout_batch}")
            
            optimizer.zero_grad()
            epoch_loss = 0.0
            accumulated_loss = 0.0
            
            for mb_idx in tqdm(range(n_microbatches), desc="训练", leave=False):
                start_idx = mb_idx * micro_batch_size
                end_idx = start_idx + micro_batch_size
                
                input_ids = train_batch['input_ids'][start_idx:end_idx].to(device)
                labels = train_batch['labels'][start_idx:end_idx].to(device)
                response_mask = train_batch['response_mask'][start_idx:end_idx].to(device)
                
                mb_raw_rewards = raw_rewards[start_idx:end_idx].to(device).unsqueeze(1)
                mb_advantages = advantages[start_idx:end_idx].to(device).unsqueeze(1)
                mb_old_log_probs = old_log_probs_list[mb_idx]
                
                result = utils.get_response_log_probs(
                    model, input_ids, labels, return_token_entropy=True
                )
                
                loss, loss_info = utils.grpo_microbatch_train_step(
                    policy_log_probs=result['log_probs'],
                    response_mask=response_mask,
                    gradient_accumulation_steps=config.gradient_accumulation_steps,
                    loss_type=loss_type,
                    raw_rewards=mb_raw_rewards,
                    advantages=mb_advantages,
                    old_log_probs=mb_old_log_probs,
                    cliprange=config.cliprange
                )
                
                epoch_loss += loss.item()
                accumulated_loss += loss.item()
                
                # 梯度累积完成后更新
                if (mb_idx + 1) % config.gradient_accumulation_steps == 0:
                    grad_norm = utils.clip_grad_norm(model, config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    optimizer_step += 1
                    current_lr = scheduler.get_last_lr()[0]
                    
                    writer.add_scalar("train/loss_per_update", accumulated_loss, optimizer_step)
                    writer.add_scalar("train/grad_norm", grad_norm, optimizer_step)
                    writer.add_scalar("train/lr", current_lr, optimizer_step)
                    
                    accumulated_loss = 0.0
                
                # 记录每个 microbatch 的指标
                writer.add_scalar("train/entropy", result['token_entropy'].mean().item(), global_step)
                
                if loss_type == "grpo_clip" and loss_info:
                    for key, value in loss_info.items():
                        writer.add_scalar(f"train/{key}", value, global_step)
                
                global_step += 1
            
            avg_epoch_loss = epoch_loss / n_microbatches
            training_stats['losses'].append(avg_epoch_loss)
            print(f"    Epoch 损失: {avg_epoch_loss:.4f}")
        
        torch.cuda.empty_cache()
    
    # ==================== 最终评估和保存 ====================
    load_policy_into_vllm_instance(model, llm)
    model.eval()
    with torch.no_grad():
        final_accuracy, final_format_reward = evaluate(
            model_path=None,
            llm=llm,
            rl=True,
            reward_fn=reward_fn,
            prompt=config.prompt_template
        )
    print(f"最终验证准确率: {final_accuracy:.4f}")
    
    if final_accuracy > best_val_accuracy and config.save_checkpoints:
        best_val_accuracy = final_accuracy
        best_model_path = os.path.join(save_path, "best_model")
        os.makedirs(best_model_path, exist_ok=True)
        model.save_pretrained(best_model_path)
        tokenizer.save_pretrained(best_model_path)
    
    if config.save_checkpoints:
        final_model_path = os.path.join(save_path, "final_model")
        os.makedirs(final_model_path, exist_ok=True)
        model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        print(f"最终模型已保存到: {final_model_path}")
    
    writer.close()
    
    print(f"\n{'='*60}")
    print(f"训练完成: loss_type = {loss_type}")
    print(f"最佳验证准确率: {best_val_accuracy:.4f}")
    print(f"{'='*60}\n")
    
    return {
        'loss_type': loss_type,
        'best_val_accuracy': best_val_accuracy,
        'final_train_reward': training_stats['train_rewards'][-1] if training_stats['train_rewards'] else 0,
        'training_stats': training_stats
    }


def main():
    """
    主函数：运行所有配置的实验
    """
    # ==================== 配置 ====================
    config = GRPOConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("GRPO 训练实验")
    print("="*60)
    print(f"\n配置:")
    print(f"  - 模型: {config.model_path}")
    print(f"  - 设备: {device}")
    print(f"  - 损失类型: {config.loss_types_to_run}")
    print(f"  - GRPO 步数: {config.n_grpo_steps}")
    print(f"  - Epochs per rollout: {config.epochs_per_rollout_batch}")
    print(f"  - 学习率: {config.learning_rate}")
    print(f"  - Rollout batch size: {config.rollout_batch_size}")
    print(f"  - Group size: {config.group_size}")
    print()
    
    # ==================== 加载数据 ====================
    print("加载数据集...")
    train_prompts, train_answers = load_gsm8k_dataset(
        "data/gsm8k/train.jsonl",
        config.prompt_template
    )
    print(f"训练样本数: {len(train_prompts)}")
    
    # ==================== 选择奖励函数 ====================
    reward_fn = r1_zero_reward_fn  # 使用完整格式的奖励函数
    
    # ==================== 加载分词器 ====================
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ==================== 运行实验 ====================
    all_results = []
    
    for loss_type in config.loss_types_to_run:
        print(f"\n{'#'*60}")
        print(f"# 实验: {loss_type}")
        print(f"{'#'*60}")
        
        # 每个实验重新加载模型（从原始检查点开始）
        print("加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        model = model.to(device)
        model.train()
        
        # 创建 vLLM 实例
        print("初始化 vLLM...")
        llm = LLM(
            model=config.model_path,
            dtype="bfloat16",
            gpu_memory_utilization=config.gpu_memory_utilization,
            device=device,
            enforce_eager=True  # 避免 CUDA graph 与训练冲突
        )
        
        # 运行实验
        result = run_single_experiment(
            config=config,
            loss_type=loss_type,
            model=model,
            tokenizer=tokenizer,
            llm=llm,
            train_prompts=train_prompts,
            train_answers=train_answers,
            reward_fn=reward_fn,
            device=device
        )
        
        all_results.append(result)
        
        # 清理显存
        del model
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    
    # ==================== 打印实验总结 ====================
    print("\n" + "="*60)
    print("实验总结")
    print("="*60)
    
    for result in all_results:
        print(f"\n{result['loss_type']}:")
        print(f"  最佳验证准确率: {result['best_val_accuracy']:.4f}")
        print(f"  最终训练奖励: {result['final_train_reward']:.4f}")
    
    # 找出最佳实验
    best_result = max(all_results, key=lambda x: x['best_val_accuracy'])
    print(f"\n最佳配置: {best_result['loss_type']}")
    print(f"最佳准确率: {best_result['best_val_accuracy']:.4f}")
    
    return all_results


if __name__ == "__main__":
    main()