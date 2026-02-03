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
1. 使用 vLLM 加速推理（rollout）
2. 训练结束后再进行评估（避免显存冲突）
3. 使用 Flash Attention 2 加速训练
4. 梯度累积减少显存使用
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
    # 1. 强制同步，确保训练计算图已执行完毕
    torch.cuda.synchronize()
    
    # 2. 【关键】移动到 CPU。
    #    这既避免了 GPU 显存翻倍（OOM），又彻底隔绝了 CUDA 指针冲突（Illegal Access）。
    #    虽然有数据传输开销，但对于几十步才做一次的 Rollout 来说，稳定性远比这点速度重要。
    state_dict = {k: v.cpu() for k, v in policy.state_dict().items()}
    
    # 3. 加载到 vLLM (vLLM 会自动处理从 CPU 到 GPU 的搬运)
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())
    
    # 4. 清理内存
    del state_dict
    
    # 5. 再次同步，确保 vLLM 加载完成前不进行后续操作
    torch.cuda.synchronize()



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
    """
    运行单个损失类型的完整训练实验
    
    注意：训练期间不进行评估，避免 vLLM 显存冲突
    """
    
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
    optimizer_step = 0
    training_stats = {
        'train_rewards': [],
        'losses': []
    }
    
    # ==================== 训练循环 ====================
    for grpo_step in range(config.n_grpo_steps):
        print(f"\n--- GRPO Step {grpo_step + 1}/{config.n_grpo_steps} ---")
#        [新增] 循环开始前强制同步
        torch.cuda.synchronize()

        # 1. 同步模型权重到 vLLM
        model.eval()
        load_policy_into_vllm_instance(model, llm)
        
        # [新增] 权重加载后再次同步（双重保险）
        torch.cuda.synchronize()
        # 2. 采样训练数据
        indices = random.sample(range(len(train_prompts)), n_prompts_per_rollout)
        batch_prompts = [train_prompts[i] for i in indices]
        batch_answers = [train_answers[i] for i in indices]
        
        # 3. Rollout: 使用 vLLM 生成回答
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
        
        # 4. 计算奖励和优势
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
        
        # 5. 准备训练数据
        train_batch = utils.tokenize_prompt_and_output(
            repeated_prompts, 
            all_responses, 
            tokenizer
        )
        
        # 6. 计算旧策略的对数概率
        old_log_probs_list = []
        
        with torch.no_grad():
            for mb_idx in range(n_microbatches):
                start_idx = mb_idx * micro_batch_size
                end_idx = start_idx + micro_batch_size
                
                input_ids = train_batch['input_ids'][start_idx:end_idx].to(device)
                labels = train_batch['labels'][start_idx:end_idx].to(device)
                
                old_log_probs = utils.get_response_log_probs(model, input_ids, labels)['log_probs']
                old_log_probs_list.append(old_log_probs.detach())
        
        # 切换到训练模式
        model.train()
        
        # 7. 多 epoch 训练
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
        
        # 清理显存
        # 清理显存 - 更彻底的清理
        del train_batch
        del old_log_probs_list
        
        # 强制同步和清理
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
    
    # ==================== 保存模型 ====================
    if config.save_checkpoints:
        final_model_path = os.path.join(save_path, "final_model")
        os.makedirs(final_model_path, exist_ok=True)
        model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        print(f"最终模型已保存到: {final_model_path}")
    
    writer.close()
    
    print(f"\n{'='*60}")
    print(f"训练完成: loss_type = {loss_type}")
    print(f"{'='*60}\n")
    
    return {
        'loss_type': loss_type,
        'final_train_reward': training_stats['train_rewards'][-1] if training_stats['train_rewards'] else 0,
        'training_stats': training_stats,
        'model_path': os.path.join(save_path, "final_model") if config.save_checkpoints else None
    }


def evaluate_with_vllm(model_path: str, reward_fn, prompt_template: str) -> Tuple[float, float]:
    """
    使用 vLLM 评估模型（训练结束后单独调用）
    
    在新的 vLLM 实例中加载保存的模型进行评估，
    避免与训练过程中的显存冲突。
    """
    print(f"\n使用 vLLM 评估模型: {model_path}")
    
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        enforce_eager=True
    )
    
    accuracy, format_reward = evaluate(
        model_path=None,
        llm=llm,
        rl=True,
        reward_fn=reward_fn,
        prompt=prompt_template
    )
    
    # 清理 vLLM
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    
    return accuracy, format_reward


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
    reward_fn = r1_zero_reward_fn
    
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
        
        # 每个实验重新加载模型
        print("加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        model = model.to(device)
        model.train()
        
        # 创建 vLLM 实例（仅用于 rollout）
        print("初始化 vLLM...")
        llm = LLM(
            model=config.model_path,
            dtype="bfloat16",
            gpu_memory_utilization=config.gpu_memory_utilization,
            device=device,
            enforce_eager=True
        )
        
        # 运行训练（不进行中间评估）
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
        
        # 清理训练资源（释放显存）
        del model
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        
        # 训练结束后，使用新的 vLLM 实例评估
        if result['model_path']:
            accuracy, format_reward = evaluate_with_vllm(
                result['model_path'],
                reward_fn,
                config.prompt_template
            )
            result['final_accuracy'] = accuracy
            result['final_format_reward'] = format_reward
            print(f"最终准确率: {accuracy:.4f}, 格式奖励: {format_reward:.4f}")
        
        all_results.append(result)
    
    # ==================== 打印实验总结 ====================
    print("\n" + "="*60)
    print("实验总结")
    print("="*60)
    
    for result in all_results:
        print(f"\n{result['loss_type']}:")
        print(f"  最终训练奖励: {result['final_train_reward']:.4f}")
        if 'final_accuracy' in result:
            print(f"  最终测试准确率: {result['final_accuracy']:.4f}")
            print(f"  最终格式奖励: {result['final_format_reward']:.4f}")
    
    # 找出最佳实验
    results_with_acc = [r for r in all_results if 'final_accuracy' in r]
    if results_with_acc:
        best_result = max(results_with_acc, key=lambda x: x['final_accuracy'])
        print(f"\n最佳配置: {best_result['loss_type']}")
        print(f"最佳准确率: {best_result['final_accuracy']:.4f}")
    
    return all_results


if __name__ == "__main__":
    main()