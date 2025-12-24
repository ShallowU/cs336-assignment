import torch

def choose_next_token(logits, tempereture, top_p, top_k):
    next_token_logits = logits.squeeze()[-1]
    
    # 如果温度为0，直接返回最大概率的token（贪心解码）
    if tempereture == 0:
        return torch.argmax(next_token_logits).item()
    
    # 应用温度缩放
    next_token_logits = next_token_logits / tempereture
    
    # Top-k 采样：只保留概率最高的k个token
    if top_k > 0:
        # 获取top-k的值和索引
        top_k_values, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
        # 创建一个mask，只保留top-k的logits
        logits_mask = torch.full_like(next_token_logits, float('-inf'))
        logits_mask[top_k_indices] = top_k_values
        next_token_logits = logits_mask
    
    # 计算概率分布
    probabilities = torch.softmax(next_token_logits, dim=0)
    
    # Top-p 采样（Nucleus Sampling）
    if top_p < 1.0:
        # 按概率从大到小排序
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
        # 计算累积概率
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)
        # 找到累积概率超过top_p的位置
        cutoff_index = torch.where(cumulative_probs > top_p)[0]
        if len(cutoff_index) > 0:
            # 保留累积概率在top_p范围内的token
            cutoff_index = cutoff_index[0].item()
            # 创建新的概率分布
            nucleus_probs = torch.zeros_like(probabilities)
            nucleus_probs[sorted_indices[:cutoff_index + 1]] = sorted_probs[:cutoff_index + 1]
            # 重新归一化
            nucleus_probs = nucleus_probs / nucleus_probs.sum()
            probabilities = nucleus_probs
    
    # 从概率分布中采样
    next_token = torch.multinomial(probabilities, num_samples=1).item()
    
    return next_token

def generate(
    model, 
    tokenizer, 
    prompt,
    max_tokens=256,
    temperature=1.0,
    top_p=0.9,
    top_k=0,
    device='cuda',
    eos_token_id=256
):
    """
    从 prompt 生成文本补全
    
    Args:
        model: TransformerLM 模型
        tokenizer: 分词器（需要有 encode/decode 方法）
        prompt: 输入文本
        max_tokens: 最大生成 token 数
        temperature: 温度参数（0=贪心）
        top_p: nucleus sampling 阈值
        top_k: top-k sampling 阈值
        device: 设备
        eos_token_id: 结束 token ID（通常是 <|endoftext|>）
    """
    model.eval()
    
    # 编码 prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # 如果未指定 eos，尝试从 tokenizer 获取
    if eos_token_id is None:
        eos_token_id = tokenizer.vocab_to_id[b"<|endoftext|>"]
    
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # 截断到模型的 context_length
            context = input_ids[:, -model.context_length:]
            
            # 前向传播
            logits = model(context)  # (batch=1, seq, vocab)
            
            # 采样下一个 token
            next_token = choose_next_token(logits, temperature, top_p, top_k)
            
            # 检查是否遇到结束符
            if next_token.item() == eos_token_id:
                break
            
            generated_tokens.append(next_token.item())
            
            # 拼接到输入序列
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    
    # 解码生成的文本
    completion = tokenizer.decode(generated_tokens)
    
    return prompt + completion