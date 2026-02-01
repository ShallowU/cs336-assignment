from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import logging
import os
import gc
import math
import random
from tqdm import tqdm
from tensorboardX import SummaryWriter
from vllm import LLM, SamplingParams

# 设置日志级别，减少 VLLM 的输出
logging.getLogger("vllm").setLevel(logging.WARNING)
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

def load_policy_into_vllm_instance(policy, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

try:
    from drgrpo_grader import r1_zero_reward_fn
    import utils
    from math_baseline import evaluate_vllm, evaluate
except:
    from .drgrpo_grader import r1_zero_reward_fn
    from . import utils
    from .math_baseline import evaluate_vllm, evaluate

# ==================== A100 40G 配置 ====================
device = 'cuda'

# 从上次保存的模型继续训练
checkpoint_path = "/content/drive/MyDrive/cs336/assignment5/sft_logs_1.5B/final_model"

# 使用 bfloat16 加载模型
model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path,  # 从 checkpoint 加载
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
model = model.to(device)

# 启用梯度检查点
model.gradient_checkpointing_enable()

# VLLM 配置 - 用于评估
llm = LLM(
    model=checkpoint_path,  # 也从 checkpoint 加载
    gpu_memory_utilization=0.3,
    max_model_len=2048,
    dtype="bfloat16",
    enforce_eager=True,
)

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

# 优化器配置 - 继续训练使用更小的学习率
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-6,  # 从 5e-6 降低到 3e-6，继续训练应更保守
    betas=(0.9, 0.999),
    weight_decay=0.01,
)

# ==================== 数据准备 ====================
reward_fn = r1_zero_reward_fn
r1_zero_prompt = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""

gsm8k = []
with open("data/gsm8k/train.jsonl") as f:
    lines = f.readlines()
    for line in lines:
        gsm8k.append(json.loads(line))

prompts = []
answer = []
for item in gsm8k:
    prompts.append(r1_zero_prompt.format(question=item['question']))
    answer.append(" " + item['answer'].replace("#### ", " </think> <answer> ") + " </answer>")

# ==================== 继续训练配置 ====================
start_epoch = 3  # 从第 4 个 epoch 开始 (已完成 3 个)
total_epoch = 6  # 训练到第 6 个 epoch
additional_epochs = total_epoch - start_epoch  # 额外训练 3 个 epoch

batch_size = 16
micro_batch_size = 4
gradient_accumulation_steps = batch_size // micro_batch_size  # = 4

# 动态计算步数
steps_per_epoch = len(prompts) // micro_batch_size  # 7473 // 4 = 1868
total_steps = steps_per_epoch * additional_epochs  # 1868 × 3 = 5604

# 继续训练不需要 warmup，或者很短的 warmup
warmup_steps = 50  # 很短的 warmup

from torch.optim.lr_scheduler import LambdaLR

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    # Cosine decay
    progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = LambdaLR(optimizer, lr_lambda)

local_step = 0  # 重置为 0，因为是新的训练阶段
log_directory = '/content/drive/MyDrive/cs336/assignment5/sft_logs_1.5B_continued'
os.makedirs(log_directory, exist_ok=True)
writer = SummaryWriter(log_directory)

# 记录最佳模型
best_accuracy = 0.3707  # 上次训练的最终准确率
best_model_path = None

# ==================== 训练循环 ====================
print(f"=" * 60)
print(f"继续训练 Qwen2.5-1.5B-Instruct")
print(f"从 checkpoint 加载: {checkpoint_path}")
print(f"上次最佳准确率: {best_accuracy:.4f}")
print(f"=" * 60)
print(f"总样本数: {len(prompts)}, 每epoch步数: {steps_per_epoch}")
print(f"Batch size: {batch_size}, Micro batch: {micro_batch_size}, 梯度累积: {gradient_accumulation_steps}")
print(f"继续训练 {additional_epochs} 个 epochs (Epoch {start_epoch+1} to {total_epoch})")
print(f"总训练步数: {total_steps}, Warmup步数: {warmup_steps}")
print(f"学习率: 2e-6 (降低后)")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"=" * 60)

model.train()

for i in range(additional_epochs):
    current_epoch = start_epoch + i + 1  # 实际的 epoch 编号 (4, 5, 6)
    
    # 每个 epoch 打乱数据顺序
    indices = list(range(len(prompts)))
    random.shuffle(indices)
    shuffled_prompts = [prompts[idx] for idx in indices]
    shuffled_answers = [answer[idx] for idx in indices]
    
    pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {current_epoch}/{total_epoch}")
    epoch_loss = 0.0
    
    for j in pbar:
        prompt_strs = shuffled_prompts[j * micro_batch_size:(j + 1) * micro_batch_size]
        answer_strs = shuffled_answers[j * micro_batch_size:(j + 1) * micro_batch_size]
        
        train_batch = utils.tokenize_prompt_and_output(prompt_strs, answer_strs, tokenizer)
        
        # 前向传播
        result_dict = utils.get_response_log_probs(
            model,
            train_batch['input_ids'].to(device),
            train_batch['labels'].to(device)
        )
        log_probs = result_dict['log_probs']
        
        # 计算损失
        loss, log_info = utils.sft_microbatch_train_step(
            log_probs,
            train_batch['response_mask'].to(device),
            gradient_accumulation_steps
        )
        
        epoch_loss += loss.item()
        
        # 记录损失 (使用全局 step 编号)
        global_step = start_epoch * steps_per_epoch + local_step
        writer.add_scalar('train/loss', loss.item(), global_step)
        writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
        
        # 梯度累积后更新
        if (local_step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'LR': f'{scheduler.get_last_lr()[0]:.2e}',
            'Mem': f'{torch.cuda.memory_allocated()/1e9:.1f}G'
        })
        
        local_step += 1
    
    # ==================== 每个 Epoch 结束后评估 ====================
    avg_epoch_loss = epoch_loss / steps_per_epoch
    print(f"\n[Epoch {current_epoch}] Average Loss: {avg_epoch_loss:.4f}")
    writer.add_scalar('train/epoch_loss', avg_epoch_loss, current_epoch)
    
    model.eval()
    
    save_directory = f'{log_directory}/epoch_{current_epoch}'
    os.makedirs(save_directory, exist_ok=True)
    
    # 保存模型检查点
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    
    # 清理显存后再评估
    gc.collect()
    torch.cuda.empty_cache()
    
    # 加载权重到 VLLM 进行评估
    try:
        load_policy_into_vllm_instance(model, llm)
        accuracy, type1_num, type2_num, type3_num = evaluate(save_directory, llm)
        
        print(f"[Epoch {current_epoch}] Accuracy: {accuracy:.4f}, "
              f"Type1: {type1_num}, Type2: {type2_num}, Type3: {type3_num}")
        
        writer.add_scalar('val/accuracy', accuracy, current_epoch)
        writer.add_scalar('val/type1', type1_num, current_epoch)
        writer.add_scalar('val/type2', type2_num, current_epoch)
        writer.add_scalar('val/type3', type3_num, current_epoch)
        
        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = f'{log_directory}/best_model'
            os.makedirs(best_model_path, exist_ok=True)
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"  ✓ 新最佳模型! Accuracy: {best_accuracy:.4f}, 已保存到 {best_model_path}")
        else:
            print(f"  ✗ 未超过最佳准确率 ({best_accuracy:.4f})")
            
    except Exception as e:
        print(f"评估失败: {e}")
    
    # 清理显存
    gc.collect()
    torch.cuda.empty_cache()
    
    model.train()


# 总结
print(f"\n" + "=" * 60)
print(f"训练总结:")
print(f"  - 最佳准确率: {best_accuracy:.4f}")
if best_model_path:
    print(f"  - 最佳模型路径: {best_model_path}")
print(f"=" * 60)

writer.close()