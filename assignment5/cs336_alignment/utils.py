from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import math

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer): 
    """
    Tokenize the prompt and output strings, and construct a mask that is 1 for the response tokens and 0 for other tokens (prompt or padding).

    Args:
    prompt_strs: list[str] List of prompt strings.
    output_strs: list[str] List of output strings.
    tokenizer: PreTrainedTokenizer Tokenizer to use for tokenization.

    Returns:
    dict[str, torch.Tensor]. Let prompt_and_output_lens be a list containing the lengths of the tokenized prompt and output strings. Then the returned dictionary should have the following keys:
    input_ids: torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1): the tokenized prompt and output strings, with the final token sliced off.
    labels: torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1): shifted input ids, i.e., the input ids without the first token. 
    response_mask: torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1): a mask on the response tokens in the labels."""
    prompt_and_output_lens = [] # 存储每个样本的prompt和output的总长度
    prompt_len = [] # 存储每个样本的prompt长度
    response_mask = [] # 存储每个样本的response mask
    input_ids = [] # 存储每个样本的input_ids
    labels = [] # 存储每个样本的labels
    for i in range(len(prompt_strs)):
        input_id = tokenizer.encode(prompt_strs[i], add_special_tokens=False)
        output_id = tokenizer.encode(output_strs[i], add_special_tokens=False)
        input_id_full = input_id + output_id # 拼接prompt和output的token ids
        local_len = len(input_id) + len(output_id) # 计算当前样本的prompt和output的总长度
        prompt_len.append(len(input_id)) # 记录当前样本的prompt长度
        prompt_and_output_lens.append(local_len) # 记录当前样本的prompt和output的总长度
        mask = [0.0] * (local_len - 1) # 初始化mask，长度为local_len - 1，因为最后一个token会被切掉
        response_mask.append(mask)
        input_ids.append(input_id_full)
        labels.append(input_id_full)
    max_len = max(prompt_and_output_lens) # 计算当前batch中最长的prompt和output的总长度
    # 对每个样本进行padding和截断
    # 问题出现在，需要在加了padding之后再将full的token_ids进行截断第一个和最后一个，而不是先截断再加padding!
    for i in range(len(prompt_strs)):
        if prompt_and_output_lens[i] < max_len:
            padding_num = max_len - prompt_and_output_lens[i] # 计算需要padding的长度
            input_ids[i] = input_ids[i] + [tokenizer.pad_token_id] * padding_num
            labels[i] = labels[i] + [tokenizer.pad_token_id] * padding_num
            response_mask[i] = response_mask[i] + [0.0] * padding_num

            input_ids[i] = input_ids[i][:-1] # 截断最后一个token
            labels[i] = labels[i][1:] # 截断第一个token
            # 修复mask计算逻辑，确保正确标记响应部分
            response_mask[i][prompt_len[i]-1:prompt_and_output_lens[i]-1] = [1.0] * (prompt_and_output_lens[i]-prompt_len[i])
        else:
            input_ids[i] = input_ids[i][:-1]
            labels[i] = labels[i][1:]
            # 修复mask计算逻辑
            response_mask[i][prompt_len[i]-1:] = [1.0] * (prompt_and_output_lens[i]-prompt_len[i])
    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)
    response_mask = torch.tensor(response_mask)
    return {
        "input_ids": input_ids.to(torch.long),
        "labels": labels.to(torch.long),
        "response_mask": response_mask.to(torch.bool)
    }

def compute_entropy(logits):
    """
    logits: (batch_size, seq_len, vocab_size)
    """
    # 这个log_softmax内部使用了logsumexp的技术，也就是减去最大值再计算softmax的技巧，防止数值上溢
    with torch.no_grad():
        log_prob = torch.nn.functional.log_softmax(logits,dim=-1)
        prob = torch.exp(log_prob)
    return -(torch.sum(prob * log_prob,dim=-1))

def get_response_log_probs(
        model,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        return_token_entropy: bool = False,
    ) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)# batch,seq,vocab
    log_probs = log_probs.gather(dim=-1,index=labels.unsqueeze(-1)).squeeze(-1) # 需要索引维度与数据维度匹配，最后再去掉最后的多余维度


    if return_token_entropy:
        return {
            "log_probs": log_probs,# batch_size, seq_len
            "token_entropy": compute_entropy(logits)
        }
    else:
        return {
            "log_probs": log_probs
        }

def masked_normalize(
        tensor,
        mask,
        normalize_constant,
        dim
    ):
    """
    tensor: (batch_size, seq_len)
    mask: (batch_size, seq_len)
    tensor: torch.Tensor The tensor to sum and normalize.
    mask: torch.Tensor Same shape as tensor; positions with 1 are included in the sum.
    normalize_constant: float the constant to divide by for normalization.
    dim: int | None the dimension to sum along before normalization. If None, sum over all
    dimensions.
    dim=1: 沿seq_len维度求和（每个样本分别求和）
    dim=None: 全局求和，是标量
    """
    # 创建一个新的张量而不是原地修改
    masked_tensor = tensor * mask.float()
    if dim is not None:
        res = torch.sum(masked_tensor,dim=dim) / normalize_constant
    else:
        res = torch.sum(masked_tensor) / normalize_constant
    return res # (batch_size,)

def sft_microbatch_train_step(
        policy_log_probs,
        response_mask,
        gradient_accumulation_steps,
        normalize_constant = 1.0
    ):
    """Compute the supervised fine-tuning loss with microbatching.
    输入:
    policy_log_probs    [-0.5, -0.9, -0.8, -1.2, -2.0]  样本1
                        [-1.5, -0.9, -0.5, -3.0, -4.0]  样本2

    response_mask       [0, 1, 1, 1, 0]  样本1
                        [0, 0, 1, 1, 0]  样本2

    ↓ 步骤1: 计算response长度
    response_lengths    [3, 2]

    ↓ 步骤2: 掩码过滤
    masked_log_probs    [0.0, -0.9, -0.8, -1.2, 0.0]
                        [0.0,  0.0, -0.5, -3.0, 0.0]

    ↓ 步骤3: 求和并归一化
    sequence_losses     [-2.9/3, -3.5/2] = [-0.967, -1.75]

    ↓ 步骤4: 取负值、平均、除以累积步数
    loss = -((-0.967 + -1.75) / 2) / 4 = 0.3396

    ↓ 步骤5: 反向传播
    loss.backward()
    """
    # 计算每个序列的响应长度
    response_lengths = response_mask.sum(dim=-1)  # (batch_size,)
    
    # 对每个序列按其响应长度归一化
    masked_log_probs = policy_log_probs * response_mask.float()  # (batch_size, seq_len)
    sequence_losses = masked_log_probs.sum(dim=-1) / response_lengths.clamp(min=1)  # (batch_size,)，将所有 <1 的值替换为1
    
    # 计算平均损失
    loss = -sequence_losses.mean() / gradient_accumulation_steps
    loss.backward()
    return loss, {} 