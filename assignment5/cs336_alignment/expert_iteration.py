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
from typing import List

# ┌─────────────────────────────────────────────────────────────┐
# │                Expert Iteration 训练流程                     │
# ├─────────────────────────────────────────────────────────────┤
# │                                                               │
# │  1. 初始化阶段                                               │
# │     ├─ 加载预训练模型 (Qwen2.5-1.5B)                        │
# │     ├─ 配置优化器 (AdamW)                                    │
# │     └─ 加载 VLLM 推理引擎                                    │
# │                                                               │
# │  2. Expert Iteration 循环 (5次)                              │
# │     ├─ 生成阶段:                                             │
# │     │   ├─ 随机采样 1024 个问题                              │
# │     │   ├─ 每个问题生成 4 个候选答案                         │
# │     │   └─ 筛选正确答案加入训练集                            │
# │     │                                                          │
# │     └─ 训练阶段:                                              │
# │         ├─ 使用筛选的数据训练 3 个 epoch                     │
# │         ├─ 梯度累积优化                                       │
# │         └─ 定期评估并保存模型                                 │
# │                                                               │
# │  3. 保存最终模型                                              │
# └─────────────────────────────────────────────────────────────┘

# 设置日志级别，减少 VLLM 的输出
logging.getLogger("vllm").setLevel(logging.WARNING)
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

try:
    from drgrpo_grader import r1_zero_reward_fn
    import utils
    from math_baseline import evaluate
except:
    from .drgrpo_grader import r1_zero_reward_fn
    from . import utils
    from .math_baseline import evaluate


def load_policy_into_vllm_instance(policy, llm: LLM):
    """
    将训练模型的权重同步到 VLLM 推理引擎
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def get_response(vllm_model: LLM, prompts: List[str], sampling_params) -> List:
    """
    使用 VLLM 批量生成回答
    """
    outputs = vllm_model.generate(prompts, sampling_params)
    res = [output.outputs for output in outputs]
    return res


# ==================== 配置 ====================
device = 'cuda'
model_path = "./models/Qwen2.5-1.5B-Instruct"
log_directory = '/content/drive/MyDrive/cs336/assignment5/EI_logs'
os.makedirs(log_directory, exist_ok=True)

# ==================== 模型加载 ====================
print("=" * 60)
print("加载模型...")

# 使用 bfloat16 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
model = model.to(device)
model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(model_path)

# VLLM 配置 - 用于生成和评估
llm = LLM(
    model=model_path,
    gpu_memory_utilization=0.3,
    max_model_len=2048,
    dtype="bfloat16",
    enforce_eager=True,
)

# 采样参数：每个问题生成4个候选答案
sampling_params = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    max_tokens=1024,
    min_tokens=4,
    stop=["\n"],
    n=4  # 每个问题生成4个候选
)

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

# 加载 GSM8K 训练数据
gsm8k = []
with open("data/gsm8k/train.jsonl") as f:
    for line in f:
        gsm8k.append(json.loads(line))

# 准备待筛选的数据
prompts_all = []
answers_all = []
for item in gsm8k:
    prompts_all.append(r1_zero_prompt.format(question=item['question']))
    # 提取 "#### " 后面的数字作为正确答案
    answer_text = item['answer']
    answers_all.append(answer_text[answer_text.find("####") + 5:].strip())

# ==================== 训练参数 ====================
n_ei_steps = 5          # Expert Iteration 迭代次数
n_samples_per_iter = 1024  # 每次迭代采样的问题数
epoch_per_iter = 3      # 每次迭代训练的 epoch 数
batch_size = 16         # 逻辑批次大小
micro_batch_size = 4    # 物理批次大小
gradient_accumulation_steps = batch_size // micro_batch_size  # = 4

# ==================== 打印配置信息 ====================
print("=" * 60)
print("Expert Iteration 训练配置")
print("=" * 60)
print(f"模型路径: {model_path}")
print(f"总样本数: {len(prompts_all)}")
print(f"EI 迭代次数: {n_ei_steps}")
print(f"每次迭代采样: {n_samples_per_iter} 个问题")
print(f"每问题候选数: 4")
print(f"每次迭代训练: {epoch_per_iter} epochs")
print(f"Batch size: {batch_size}, Micro batch: {micro_batch_size}")
print(f"梯度累积步数: {gradient_accumulation_steps}")
print(f"学习率: 5e-6")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("=" * 60)

# ==================== TensorBoard ====================
writer = SummaryWriter(log_directory)

# ==================== 训练状态 ====================
prompts_filtered = []   # 筛选后的 prompts（累积）
answers_filtered = []   # 筛选后的 answers（累积）
global_step = 0         # 全局训练步数
best_accuracy = 0.0     # 最佳准确率
best_model_path = None  # 最佳模型路径

# ==================== Expert Iteration 主循环 ====================
for ei_iter in range(n_ei_steps):
    print(f"\n{'='*60}")
    print(f"Expert Iteration Step {ei_iter + 1}/{n_ei_steps}")
    print(f"{'='*60}")
    
    # ========== 阶段1: 生成并筛选数据 ==========
    print(f"\n[生成阶段] 采样 {n_samples_per_iter} 个问题，每个生成 4 个候选...")
    
    # 同步最新模型权重到 VLLM
    model.eval()
    load_policy_into_vllm_instance(model, llm)
    
    # 随机采样问题
    indices = random.sample(range(len(prompts_all)), k=n_samples_per_iter)
    sampled_prompts = [prompts_all[i] for i in indices]
    sampled_answers = [answers_all[i] for i in indices]
    
    # 生成候选答案
    gc.collect()
    torch.cuda.empty_cache()
    
    outputs = get_response(llm, sampled_prompts, sampling_params)
    
    # 筛选正确答案
    new_correct = 0
    for j in range(len(outputs)):
        for k in range(len(outputs[j])):
            generated_text = outputs[j][k].text
            result = reward_fn(generated_text, sampled_answers[j])
            
            # 只保留格式和答案都正确的
            if result['format_reward'] == 1.0 and result['answer_reward'] == 1.0:
                prompts_filtered.append(sampled_prompts[j])
                answers_filtered.append(generated_text)
                new_correct += 1
    
    print(f"  本轮新增正确答案: {new_correct}")
    print(f"  累积正确答案总数: {len(prompts_filtered)}")
    
    # 如果没有正确答案，跳过训练
    if len(prompts_filtered) == 0:
        print("  警告: 没有正确答案可用于训练，跳过本轮")
        continue
    
    # ========== 阶段2: 训练模型 ==========
    print(f"\n[训练阶段] 使用 {len(prompts_filtered)} 个样本训练 {epoch_per_iter} epochs...")
    
    model.train()
    
    steps_per_epoch = len(prompts_filtered) // micro_batch_size
    if steps_per_epoch == 0:
        steps_per_epoch = 1
    
    for epoch in range(epoch_per_iter):
        # 每个 epoch 打乱数据
        shuffle_indices = list(range(len(prompts_filtered)))
        random.shuffle(shuffle_indices)
        shuffled_prompts = [prompts_filtered[idx] for idx in shuffle_indices]
        shuffled_answers = [answers_filtered[idx] for idx in shuffle_indices]
        
        epoch_loss = 0.0
        pbar = tqdm(range(steps_per_epoch), 
                    desc=f"EI-{ei_iter+1} Epoch {epoch+1}/{epoch_per_iter}")
        
        for step in pbar:
            # 获取 micro batch
            start_idx = step * micro_batch_size
            end_idx = min((step + 1) * micro_batch_size, len(shuffled_prompts))
            
            prompt_strs = shuffled_prompts[start_idx:end_idx]
            answer_strs = shuffled_answers[start_idx:end_idx]
            
            # 跳过空批次
            if len(prompt_strs) == 0:
                continue
            
            # Tokenize
            train_batch = utils.tokenize_prompt_and_output(
                prompt_strs, answer_strs, tokenizer
            )
            
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
            
            # 记录到 TensorBoard
            writer.add_scalar('train/loss', loss.item(), global_step)
            
            # 梯度累积后更新
            if (global_step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Step': global_step,
                'Mem': f'{torch.cuda.memory_allocated()/1e9:.1f}G'
            })
            
            # 定期评估
            should_eval = (
                (global_step <= 500 and global_step % 100 == 0) or
                (500 < global_step <= 2000 and global_step % 500 == 0) or
                (global_step > 2000 and global_step % 1000 == 0)
            )
            
            if should_eval and global_step > 0:
                print(f"\n[评估] Step {global_step}...")
                model.eval()
                
                gc.collect()
                torch.cuda.empty_cache()
                
                try:
                    load_policy_into_vllm_instance(model, llm)
                    
                    eval_dir = f'{log_directory}/step_{global_step}'
                    os.makedirs(eval_dir, exist_ok=True)
                    
                    accuracy, type1, type2, type3 = evaluate(eval_dir, llm)
                    
                    print(f"  Accuracy: {accuracy:.4f}, "
                          f"Type1: {type1}, Type2: {type2}, Type3: {type3}")
                    
                    writer.add_scalar('val/accuracy', accuracy, global_step)
                    writer.add_scalar('val/type1', type1, global_step)
                    writer.add_scalar('val/type2', type2, global_step)
                    writer.add_scalar('val/type3', type3, global_step)
                    
                    # 保存最佳模型
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model_path = f'{log_directory}/best_model'
                        os.makedirs(best_model_path, exist_ok=True)
                        model.save_pretrained(best_model_path)
                        tokenizer.save_pretrained(best_model_path)
                        print(f"  ✓ 新最佳模型! Accuracy: {best_accuracy:.4f}")
                    
                except Exception as e:
                    print(f"  评估失败: {e}")
                
                gc.collect()
                torch.cuda.empty_cache()
                model.train()
            
            global_step += 1
        
        # Epoch 结束
        avg_loss = epoch_loss / max(steps_per_epoch, 1)
        print(f"  [Epoch {epoch+1}] Average Loss: {avg_loss:.4f}")
    
    # ========== EI 迭代结束后评估 ==========
    print(f"\n[EI Step {ei_iter+1} 完成] 进行评估...")
    model.eval()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    try:
        load_policy_into_vllm_instance(model, llm)
        
        eval_dir = f'{log_directory}/ei_step_{ei_iter+1}'
        os.makedirs(eval_dir, exist_ok=True)
        
        # 保存模型
        model.save_pretrained(eval_dir)
        tokenizer.save_pretrained(eval_dir)
        
        accuracy, type1, type2, type3 = evaluate(eval_dir, llm)
        
        print(f"  EI Step {ei_iter+1} Accuracy: {accuracy:.4f}")
        print(f"  Type1: {type1}, Type2: {type2}, Type3: {type3}")
        
        writer.add_scalar('ei/accuracy', accuracy, ei_iter + 1)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = f'{log_directory}/best_model'
            os.makedirs(best_model_path, exist_ok=True)
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"  ✓ 新最佳模型! Accuracy: {best_accuracy:.4f}")
        
    except Exception as e:
        print(f"  评估失败: {e}")
    
    gc.collect()
    torch.cuda.empty_cache()

# ==================== 训练完成 ====================
print(f"\n{'='*60}")
print("Expert Iteration 训练完成!")
print(f"{'='*60}")
print(f"总训练步数: {global_step}")
print(f"最终累积正确样本数: {len(prompts_filtered)}")
print(f"最佳准确率: {best_accuracy:.4f}")
if best_model_path:
    print(f"最佳模型路径: {best_model_path}")
print(f"{'='*60}")
writer.close()