import torch
import argparse
import os
import time
from pathlib import Path
import numpy as np
import json
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from cs336_basics.layer import TransformerLM
from cs336_basics.loss import cross_entropy
from cs336_basics.optimizer import My_AdamW, My_lr_cosine_schedule, My_gradient_clipping
from cs336_basics.util import My_save_checkpoint
from cs336_basics.data import BatchIterator
torch.set_float32_matmul_precision('high')
def parse_args():
    parser = argparse.ArgumentParser(description='Train Transformer LM')
    # Model
    parser.add_argument('--vocab-size', type=int, default=10000)
    parser.add_argument('--context-length', type=int, default=256)
    parser.add_argument('--d-model', type=int, default=512)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--num-heads', type=int, default=16)
    parser.add_argument('--d-ff', type=int, default=1344)
    parser.add_argument('--rope-theta', type=float, default=10000)
    
    # Training
    parser.add_argument('--max-lr', type=float, default=4e-3)
    parser.add_argument('--min-lr', type=float, default=1e-4)
    parser.add_argument('--total-iterations', type=int, default=2500)
    parser.add_argument('--warmup-iters', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    
    # Data
    parser.add_argument('--train-data', type=str, required=True)
    parser.add_argument('--val-data', type=str, required=True)
    
    # Logging
    parser.add_argument('--output-dir', type=str, default='exp2')
    parser.add_argument('--save-every', type=int, default=1250)
    parser.add_argument('--val-every', type=int, default=25)
    parser.add_argument('--device', type=str, default='cuda')
    
    return parser.parse_args()

def main():
    args = parse_args()
    if torch.cuda.is_available() and args.device == 'cuda':
        device = 'cuda'
    elif torch.mps.is_available() and args.device == 'mps':
        device = 'mps'
    else:
        device = 'cpu'
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # 1. Model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta
    ).to(device)
    model = torch.compile(model)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {trainable_params / 1e6:.2f}M trainable parameters")
    
    # 2. Data (memory-mapped)
    train_data = np.load(args.train_data, mmap_mode='r')
    val_data = np.load(args.val_data, mmap_mode='r')
    print(f"Train data size: {len(train_data):,} tokens")
    print(f"Val data size: {len(val_data):,} tokens")
    # 创建批次迭代器
    train_iterator = BatchIterator(train_data,"train", args.batch_size, args.context_length, device)
    val_iterator = BatchIterator(val_data, "val", args.batch_size, args.context_length, device)
    # 3. Optimizer
    optimizer = My_AdamW(model.parameters(), lr=args.max_lr)
    
    # 4. Training loop
    writer = SummaryWriter(output_dir / 'logs')
    pbar = tqdm(total=args.total_iterations, desc='Training')
    
    # Timing variables
    start_time = time.time()

    for step in range(args.total_iterations):
        step_start_time = time.time()
        # Learning rate schedule
        lr = My_lr_cosine_schedule(
            step, args.max_lr, args.min_lr, 
            args.warmup_iters, args.total_iterations
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Training step
        model.train()
        x, y = train_iterator.get_batch()
        step_tokens = args.batch_size * args.context_length
        logits = model(x)
        loss = cross_entropy(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        grad_norm = My_gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()
        
        # Calculate timing metrics
        step_end_time = time.time()
        step_time = step_end_time - step_start_time
        tokens_per_sec = step_tokens / step_time if step_time > 0 else 0
        # Logging
        writer.add_scalar('train/loss', loss.item(), step)
        writer.add_scalar('train/lr', lr, step)
        writer.add_scalar('train/grad_norm', grad_norm, step)
        writer.add_scalar('train/tokens_per_sec', tokens_per_sec, step)
        pbar.update(1)
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'tok/s': f'{tokens_per_sec:.2f}'
        })
        
        # Checkpoint
        if step % args.save_every == 0 and step > 0:
            checkpoint_path = output_dir / f'checkpoint_{step}.pt'
            My_save_checkpoint(model, optimizer, step, str(checkpoint_path))
        
        # Validation
        if step % args.val_every == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(20): # 减少循环次数，增加 batch size，验证更准更快
                    x_val, y_val = val_iterator.get_batch()
                    logits_val = model(x_val)
                    val_loss = cross_entropy(logits_val, y_val)
                    val_losses.append(val_loss.item())

            avg_val_loss = np.mean(val_losses)
            print(f"\nStep {step}: val_loss = {avg_val_loss:.4f}")
            writer.add_scalar('val/loss', avg_val_loss, step)
    
    # Final checkpoint
    final_path = output_dir / f'checkpoint_final.pt'
    My_save_checkpoint(model, optimizer, args.total_iterations, str(final_path))
    
    pbar.close()
    writer.close()
    END_time = time.time()
    total_time = END_time - start_time
    print(f"Training completed in {total_time/60:.2f} minutes.")

if __name__ == '__main__':
    main()