"""
GSM8K 数学问题评估脚本

用于评估语言模型在 GSM8K 测试集上的准确率。
支持使用 vLLM 进行高效批量推理。
"""

import json
import logging
import os
import gc
import torch
from typing import List, Tuple, Optional, Callable

from vllm import LLM, SamplingParams

try:
    from drgrpo_grader import r1_zero_reward_fn
except ImportError:
    from .drgrpo_grader import r1_zero_reward_fn

# 配置日志
logging.getLogger("vllm").setLevel(logging.WARNING)
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"


def evaluate_vllm(
    vllm_model: LLM,
    prompts: List[str],
    sampling_params: SamplingParams
) -> List[str]:
    """
    使用 vLLM 批量生成回答
    
    Args:
        vllm_model: vLLM 模型实例
        prompts: prompt 列表
        sampling_params: 采样参数
    
    Returns:
        生成的回答列表
    """
    outputs = vllm_model.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]


def evaluate(
    model_path: Optional[str] = None,
    llm: Optional[LLM] = None,
    rl: bool = False,
    reward_fn: Optional[Callable] = None,
    prompt: Optional[str] = None,
    data_path: str = "data/gsm8k/test.jsonl",
    output_path: str = "/content/drive/MyDrive/cs336/assignment5/grpo-eval/test.json"
) -> Tuple:
    """
    在 GSM8K 测试集上评估模型
    
    Args:
        model_path: 模型路径（如果 llm 为 None 则使用此路径加载模型）
        llm: vLLM 实例（可选，如果提供则直接使用）
        rl: 是否是 RL 训练模式（影响返回值）
        reward_fn: 奖励函数
        prompt: prompt 模板
        data_path: 测试数据路径
        output_path: 输出日志路径
    
    Returns:
        如果 rl=True: (accuracy, format_reward)
        如果 rl=False: (accuracy, type1_num, type2_num, type3_num)
    """
    # 采样参数
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )
    
    # 创建 vLLM 实例（如果未提供）
    if llm is None:
        llm = LLM(model=model_path, gpu_memory_utilization=0.4)
    
    # 默认奖励函数
    if reward_fn is None:
        reward_fn = r1_zero_reward_fn
    
    # 默认 prompt 模板
    if prompt is None:
        prompt = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""
    
    # 加载测试数据
    test_data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    # 准备 prompts 和答案
    prompts = []
    answers = []
    for item in test_data:
        prompts.append(prompt.format(question=item['question']))
        answer_text = item['answer']
        final_answer = answer_text[answer_text.find("####") + 5:].strip()
        answers.append(final_answer)
    
    print(f"测试样本数: {len(prompts)}")
    
    # 生成回答
    outputs = evaluate_vllm(llm, prompts, sampling_params)
    
    # 评估结果
    total_reward = 0.0
    total_format_reward = 0.0
    type1_num = 0  # 格式正确 + 答案正确
    type2_num = 0  # 格式正确 + 答案错误
    type3_num = 0  # 格式错误
    
    for i, (output, answer) in enumerate(zip(outputs, answers)):
        # 添加 <think> 前缀（因为 prompt 以 <think> 结尾）
        test_data[i]['outputs'] = "<think>" + output
        
        # 计算奖励
        result = reward_fn(output, answer)
        test_data[i]['result'] = result
        
        # 分类结果
        if result['format_reward'] == 1.0 and result['answer_reward'] == 1.0:
            type1_num += 1
            test_data[i]['type'] = 1
        elif result['format_reward'] == 1.0 and result['answer_reward'] == 0.0:
            type2_num += 1
            test_data[i]['type'] = 2
        else:
            type3_num += 1
            test_data[i]['type'] = 3
        
        total_reward += result['reward']
        if rl:
            total_format_reward += result['format_reward']
    
    accuracy = total_reward / len(outputs)
    format_reward = total_format_reward / len(outputs) if rl else 0.0
    
    # 保存详细日志
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"警告: 无法保存日志到 {output_path}: {e}")
    
    if rl:
        return accuracy, format_reward
    else:
        return accuracy, type1_num, type2_num, type3_num


if __name__ == "__main__":
    model_path = "./models/Qwen2.5-1.5B-Instruct"
    accuracy, type1_num, type2_num, type3_num = evaluate(model_path)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Type 1 (格式正确+答案正确): {type1_num}")
    print(f"Type 2 (格式正确+答案错误): {type2_num}")
    print(f"Type 3 (格式错误): {type3_num}")