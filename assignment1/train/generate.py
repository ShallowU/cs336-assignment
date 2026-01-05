import torch
from cs336_basics.layer import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.util import My_load_checkpoint

def choose_next_token(logits, temperature, top_p, top_k):
    """
    temperature: 温度参数，控制采样的随机性
    top_p: 累积概率阈值
    top_k: top-k 采样的 k 值
    先top-k，再top-p
    使用torch.multinomial进行采样
    """
    next_token_logits = logits.squeeze()[-1]
    # next_token_logits.shape = (10000,)
    # 这是第 256 个位置对应的 10000 个词的 logits

    # 如果温度为0，直接返回最大概率的token（贪心解码）索引
    if temperature == 0:
        return torch.argmax(next_token_logits).item()
    
    # 应用温度缩放，如果 temperature > 1，分布更平坦，更创造；如果 < 1，分布更陡峭，更保守
    next_token_logits = next_token_logits / temperature
    
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
            cutoff_index = cutoff_index[0].item() # 第一个超过top_p的索引
            # 创建新的概率分布，只保留累积概率在top_p范围内的token
            nucleus_probs = torch.zeros_like(probabilities)
            # 将累积概率在top_p范围内的token的概率赋值给新的概率分布
            nucleus_probs[sorted_indices[:cutoff_index + 1]] = sorted_probs[:cutoff_index + 1]
            # 重新归一化
            nucleus_probs = nucleus_probs / nucleus_probs.sum()
            probabilities = nucleus_probs
    
    # 从概率分布中采样
    next_token = torch.multinomial(probabilities, num_samples=1).item()
    
    return next_token

def generate_stream(
    model, 
    tokenizer, 
    prompt,
    max_tokens=256,
    temperature=1.0,
    top_p=0.9,
    top_k=0,
    device='cuda',
    eos_token_id=None
):
    """
    流式生成文本,逐 token yield
    
    Yields:
        str: 每次生成的单个 token 解码后的文本
    """
    model.eval()
    
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    # input_ids shape: (1, seq_len)
    
    if eos_token_id is None:
        eos_token_id = tokenizer.vocab_to_id.get(b"<|endoftext|>", None)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            context_length = getattr(model, 'context_length', 256)
            # 截断到模型的 context_length
            context = input_ids[:, -context_length:]
            
            logits = model(context)
            next_token = choose_next_token(logits, temperature, top_p, top_k)
            
            if eos_token_id is not None and next_token == eos_token_id:
                break
            
            # 实时解码并 yield 单个 token
            token_text = tokenizer.decode([next_token])
            yield token_text
            
            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=1)

def generate(
    model, 
    tokenizer, 
    prompt,
    max_tokens=256,
    temperature=1.0,
    top_p=0.9,
    top_k=0,
    device='cuda',
    eos_token_id=None
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
        eos_token_id = tokenizer.vocab_to_id.get(b"<|endoftext|>", None)
    
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # 截断到模型的 context_length
            context_length = getattr(model, 'context_length', 256)
            context = input_ids[:, -context_length:]
            
            # 前向传播
            logits = model(context)  # (batch=1, seq, vocab)
            
            # 采样下一个 token (返回 int)
            next_token = choose_next_token(logits, temperature, top_p, top_k)
            
            # 检查是否遇到结束符
            if eos_token_id is not None and next_token == eos_token_id:
                break
            
            generated_tokens.append(next_token)
            
            # 拼接到输入序列
            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
    
    # 解码生成的文本
    completion = tokenizer.decode(generated_tokens)
    
    return prompt + completion

if __name__ == "__main__":
    config = {
        'vocab_size': 10000,
        'context_length': 256,
        'd_model': 512,
        'num_layers': 4,
        'num_heads': 16,
        'd_ff': 1344,
        'rope_theta': 10000
    }
    model = TransformerLM(**config)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.to(device)
    model_path = "model/checkpoint_final.pt"
    My_load_checkpoint(model_path, model)
    num_params = sum(p.numel() for p in model.parameters())
    
    # 模型详细配置说明
    print("============================================================")
    print("Model Configuration:")
    print("model_path:", model_path)
    print(f"Total Parameters: {num_params / 1e6:.2f}M")
    print(f"Vocab Size: {config['vocab_size']}")
    print(f"Context Length: {config['context_length']}")
    print(f"Model Dimension: {config['d_model']}")
    print(f"Number of Layers: {config['num_layers']}")
    print(f"Number of Heads: {config['num_heads']}")
    print(f"Feedforward Dimension: {config['d_ff']}")
    print("device:", device)

    tokenizer_instance = Tokenizer.from_files(
        'train/tinystories_bpe_vocab.txt', 
        'train/tinystories_bpe_merges.txt', 
        special_tokens=["<|endoftext|>"]
    )
    
    # Tokenizer 详细配置说明
    print("============================================================")
    print("Tokenizer Configuration:")
    print(f"Vocab Size: {len(tokenizer_instance.vocab)}")
    print(f"Special Tokens: {tokenizer_instance.special_tokens}")
    print("============================================================")
    
    prompt = "What's your favorite movie"
    # generated_text = generate(
    #     model=model,
    #     tokenizer=tokenizer_instance,
    #     prompt=prompt,
    #     max_tokens=256,
    #     temperature=1.0,
    #     top_p=0.9,
    #     top_k=0,
    #     device=device,
    #     eos_token_id=tokenizer_instance.vocab_to_id[b"<|endoftext|>"]
    # )
    # print("============================================================")
    # print("Generated Text:")
    # print(generated_text)

    generated_stream = generate_stream(
        model=model,
        tokenizer=tokenizer_instance,
        prompt=prompt,
        max_tokens=256,
        temperature=1.0,
        top_p=0.9,
        top_k=0,
        device=device,
        eos_token_id=tokenizer_instance.vocab_to_id[b"<|endoftext|>"]
    )
    print("============================================================")
    print("Generated Streamed Text:")
    print(prompt, end='', flush=True)
    for token in generated_stream:
        print(token, end='', flush=True)   
    print()

    generated_stream_high = generate_stream(
        model=model,
        tokenizer=tokenizer_instance,
        prompt=prompt,
        max_tokens=256,
        temperature=2.0,
        top_p=0.9,
        top_k=0,
        device=device,
        eos_token_id=tokenizer_instance.vocab_to_id[b"<|endoftext|>"]
    )
    print("============================================================")
    print("Generated Streamed Text (High Temperature):")
    print(prompt, end='', flush=True)
    for token in generated_stream_high:
        print(token, end='', flush=True)   
    print()

    generated_stream_low = generate_stream(
        model=model,
        tokenizer=tokenizer_instance,
        prompt=prompt,
        max_tokens=256,
        temperature=0.5,
        top_p=0.9,
        top_k=0,
        device=device,
        eos_token_id=tokenizer_instance.vocab_to_id[b"<|endoftext|>"]
    )
    print("============================================================")
    print("Generated Streamed Text (Low Temperature):")
    print(prompt, end='', flush=True)
    for token in generated_stream_low:
        print(token, end='', flush=True)   
    print()