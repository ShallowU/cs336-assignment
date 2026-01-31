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
model_path = "./models/Qwen2.5-1.5B-Instruct"

# 使用 bfloat16 加载模型，A100 原生支持，更省显存且更稳定
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
model = model.to(device)

# 启用梯度检查点，显著减少显存占用
model.gradient_checkpointing_enable()

# VLLM 配置 - 用于评估（降低显存占用）
llm = LLM(
    model=model_path,
    gpu_memory_utilization=0.3,  # 从 0.4 降到 0.3，给训练留更多空间
    max_model_len=2048,  # 启用序列长度限制
    dtype="bfloat16",
    enforce_eager=True,  # 禁用 CUDA Graph，减少显存碎片
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

# 优化器配置
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-6,
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

# ==================== 训练配置 ====================
epoch = 5
batch_size = 16
micro_batch_size = 4
gradient_accumulation_steps = batch_size // micro_batch_size  # = 4

# 动态计算步数
steps_per_epoch = len(prompts) // micro_batch_size  # 1319 // 4 = 329
total_steps = steps_per_epoch * epoch  # 329 × 5 = 1645

warmup_steps = min(100, total_steps // 20)  # 约 82 步

from torch.optim.lr_scheduler import LambdaLR

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = LambdaLR(optimizer, lr_lambda)

local_step = 0
log_directory = '/content/drive/MyDrive/cs336/assignment5/sft_logs_1.5B'
os.makedirs(log_directory, exist_ok=True)
writer = SummaryWriter(log_directory)

# ==================== 训练循环 ====================
print(f"=" * 60)
print(f"开始训练 Qwen2.5-1.5B-Instruct")
print(f"总样本数: {len(prompts)}, 每epoch步数: {steps_per_epoch}")
print(f"Batch size: {batch_size}, Micro batch: {micro_batch_size}, 梯度累积: {gradient_accumulation_steps}")
print(f"总训练步数: {total_steps}, Warmup步数: {warmup_steps}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"=" * 60)

model.train()

for i in range(epoch):
    # 每个 epoch 打乱数据顺序
    indices = list(range(len(prompts)))
    random.shuffle(indices)
    shuffled_prompts = [prompts[idx] for idx in indices]
    shuffled_answers = [answer[idx] for idx in indices]
    
    pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {i+1}/{epoch}")
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
        
        # 记录损失
        writer.add_scalar('train/loss', loss.item(), local_step)
        writer.add_scalar('train/lr', scheduler.get_last_lr()[0], local_step)
        
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
    print(f"\n[Epoch {i+1}] Average Loss: {avg_epoch_loss:.4f}")
    writer.add_scalar('train/epoch_loss', avg_epoch_loss, i + 1)
    
    model.eval()
    
    save_directory = f'{log_directory}/epoch_{i+1}'
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
        
        print(f"[Epoch {i+1}] Accuracy: {accuracy:.4f}, "
              f"Type1: {type1_num}, Type2: {type2_num}, Type3: {type3_num}")
        
        writer.add_scalar('val/accuracy', accuracy, i + 1)
        writer.add_scalar('val/type1', type1_num, i + 1)
        writer.add_scalar('val/type2', type2_num, i + 1)
        writer.add_scalar('val/type3', type3_num, i + 1)
    except Exception as e:
        print(f"评估失败: {e}")
    
    # 清理显存
    gc.collect()
    torch.cuda.empty_cache()
    
    model.train()

# ==================== 保存最终模型 ====================
final_save_path = f'{log_directory}/final_model'
os.makedirs(final_save_path, exist_ok=True)
model.save_pretrained(final_save_path)
tokenizer.save_pretrained(final_save_path)
print(f"\n训练完成! 模型已保存到 {final_save_path}")

# 最终模型评估
try:
    load_policy_into_vllm_instance(model, llm)
    accuracy, type1_num, type2_num, type3_num = evaluate(final_save_path, llm)
    print(f"[Final Model] Accuracy: {accuracy:.4f}, "
          f"Type1: {type1_num}, Type2: {type2_num}, Type3: {type3_num}")
except Exception as e:
    print(f"最终评估失败: {e}")

writer.close()