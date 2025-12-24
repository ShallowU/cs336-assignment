import torch

def cross_entropy(logits, targets):
    """
    计算交叉熵损失：ℓ = -log(softmax(logits)[target_class])
    
    Args:
        inputs: 模型输出的 logits，形状 (..., vocab_size)
        targets: 目标类别索引，形状 (...)
    
    Returns:
        标量损失（所有样本的平均）
    """
    # 展平批次维度
    vocab_size=logits.size(-1)
    logits=logits.view(-1,vocab_size)
    targets=targets.view(-1)
    # 数值稳定性：减去每行最大值
    logits=logits-torch.max(logits,dim=-1,keepdim=True)[0]
    # log-sum-exp 技巧
    logexpsum=torch.log(torch.sum(torch.exp(logits),dim=-1))
    # 提取目标类别的 logit，输出形状与 index 相同
    target_logits=logits.gather(dim=-1,index=targets.unsqueeze(1))

    loss=logexpsum-target_logits
    return loss.mean()