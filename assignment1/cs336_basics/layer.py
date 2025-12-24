import math
import torch
from  torch import nn, Tensor
from torch.nn import init
from einops import rearrange,einsum
from jaxtyping import Float, Int, Bool

class Linear(nn.Module):
    def __init__(self,in_features:int,out_features:int,device=None,dtype=None):
        super().__init__()
        self.in_features=in_features
        self.out_features=out_features
        factory_kwargs={'device':device,'dtype':dtype}

        # Weight shape is (out_features, in_features)
        self.weight=nn.Parameter(torch.empty((out_features,in_features),**factory_kwargs))
        # Initialize weights using truncated normal
        std=(2/(in_features+out_features))**0.5
        init.trunc_normal_(self.weight,mean=0.0,std=std,a=-3*std,b=3*std)

    def forward(self,x):
        return einsum(x,self.weight,"... d_in,d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, 
                 num_embeddings, # vocabulary size
                 embedding_dim, # embedding dimension
                 device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        factory_kwargs = {'device': device, 'dtype': dtype}

        # Weight shape is (num_embeddings, embedding_dim)
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))

        # Initialize weights using truncated normal
        std = 1
        init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]  # Select rows corresponding to token_ids

class RMSNorm(nn.Module):
    def __init__(self,d_model: int, eps: float = 1e-5, device=None, dtype=None ):
        super().__init__()
        self.d_model=d_model
        self.eps=eps
        factory_kwargs={'device':device,'dtype':dtype}
        self.weight=nn.Parameter(torch.ones(d_model,**factory_kwargs))

    def forward(self, x: Tensor) -> Tensor:
        in_dtype=x.dtype
        x=x.to(torch.float32)

        # official implementation:
        # rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) 
        # normalized_x = x * rms
        RMS=(x.pow(2).mean(dim=-1,keepdim=True)+self.eps).sqrt()
        normalized_x=x/RMS

        result =normalized_x*self.weight # W will automatically broadcast to ..., d_model
        return result.to(in_dtype)
    
def sigmoid(x:Tensor):
    return 1/(1+torch.exp(-x))

def silu(x:Tensor):
    return x*torch.sigmoid(x)

def glu(a:Tensor,b:Tensor):
    return a*b

def swiglu_fn(a:Tensor,b:Tensor):
    return glu(silu(a),b)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.linear1 = Linear(d_model, d_ff, **factory_kwargs)  # W1
        self.linear2 = Linear(d_ff, d_model, **factory_kwargs)  # W2
        self.linear3 = Linear(d_model, d_ff, **factory_kwargs)  # W3

    def forward(self,x:Float[Tensor,"... d_model"]) -> Float[Tensor, "... d_model"]:
        w1x=self.linear1(x)
        w3x=self.linear3(x)
        h=swiglu_fn(w1x,w3x)
        return self.linear2(h)
    
class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) layer.

    Args
    ----
    theta : float
        Base used to generate inverse frequencies (e.g. 10_000).
    d_k : int
        Dimension of the key / query vectors (must be even).
    max_seq_len : int
        Maximum sequence length expected at inference / training time.
    device : torch.device | None
        Where to place the pre-computed sine / cosine tables.
    """
    def __init__(self,
                 theta: float,
                 d_k: int,
                 max_seq_len: int,
                 device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even for RoPE.")
        self.d_k = d_k
        # ---- pre-compute inverse frequencies ----
        # freq[k] = 1 / theta ** (2k / d_k)          (k = 0,1,…,d_k/2-1)
        freq = 1.0 / (theta ** (torch.arange(0,d_k,2, device=device).float() / d_k))

        # shape: (max_seq_len, d_k // 2)
        positions = torch.arange(max_seq_len, device=device).float()
        freqs = torch.outer(positions, freq)

        # cache cos/sin; no gradients needed → persistent=False
        self.register_buffer('cos_cached', torch.cos(freqs),persistent=False) # persistent=False does not save to state_dict
        self.register_buffer('sin_cached', torch.sin(freqs), persistent=False)
    
    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"]
        ) -> Float[Tensor, "... seq_len d_k"]:
        """
        Apply RoPE to `x`.  Works with any batch shape prefix.
        """
        # Check if the last dimension matches d_k
        if x.size(-1) != self.d_k:
            raise ValueError(f"Last dim of x ({x.size(-1)}) ≠ d_k ({self.d_k}).")
        
        # Gather the cached tables for the required positions
        cos_pos = self.cos_cached[token_positions]
        sin_pos = self.sin_cached[token_positions]

        # Split even / odd channels
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        # Apply the 2-D rotation to each pair
        out_even = x_even * cos_pos - x_odd * sin_pos
        out_odd = x_even * sin_pos + x_odd * cos_pos

        # Re-interleave
        out = torch.empty_like(x)
        out[..., ::2] = out_even
        out[..., 1::2] = out_odd
        return out

class RoPE(nn.Module): # 工业实现,测试不通过是因为两两配对不同的方式导致
    """
    Rotary Position Embedding (RoPE) layer.

    Args
    ----
    theta : float
        Base used to generate inverse frequencies (e.g. 10_000).
    d_k : int
        Dimension of the key / query vectors (must be even).
    max_seq_len : int
        Maximum sequence length expected at inference / training time.
    device : torch.device | None
        Where to place the pre-computed sine / cosine tables.
    """
    def __init__(self,
                 theta: float,
                 d_k: int,
                 max_seq_len: int,
                 device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even for RoPE.")

        # 1. 计算频率 (freqs)
        # 我们不希望所有维度都转得一样快。
        # 第 0 维转得最快，最后几维转得最慢 (长程衰减)。
        # 公式是：theta ^ (-2i / d)
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2,device=device).float() / d_k))
        
        # 2. 生成位置索引 (t)
        # 就是 [0, 1, 2, ..., end-1]
        t = torch.arange(max_seq_len, device=device).float()  # type: ignore

        # 3. 计算外积 (Outer Product) -> 生成角度表
        # 这里用外积是因为我们要算：(位置 0 * 频率 0), (位置 0 * 频率 1)... (位置 1 * 频率 0)...
        # torch.outer(t, freqs) 的形状是 [end, dim/2]
        freqs = torch.outer(t, freqs)  # 结果就是所有位置的所有旋转角度 (angle)

        # 4. 把角度变成 cos 和 sin
        # 工业界有个 trick：为了方便后续计算，我们把 cos 和 sin 复制两份拼接起来
        # 形状从 [end, dim/2] 变成 [end, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        # emb = torch.repeat_interleave(freqs, 2, dim=-1)  # (max_seq_len, d_k)
        self.register_buffer('cos_cached', emb.cos(),persistent=False) # persistent=False does not save to state_dict
        self.register_buffer('sin_cached', emb.sin(), persistent=False)
    
    @staticmethod
    def rotate_half(x):
        """
        这个辅助函数的作用是实现公式里的 (-y, x) 部分。
        假设输入 x 是 [x1, x2, ..., xd/2, y1, y2, ..., yd/2]
        """
        # 1. 把向量 x 最后一维切成两半
        x1 = x[..., : x.shape[-1] // 2] # 取前半部分 (相当于 x)
        x2 = x[..., x.shape[-1] // 2 :] # 取后半部分 (相当于 y)
        
        # 2. 拼接成 [-y, x]
        # 为什么是负号？因为旋转公式里有 (x*cos - y*sin)
        # 我们稍后会用 x * cos + (-y) * sin 来计算
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"]
        ) -> Float[Tensor, "... seq_len d_k"]:
        """
        Apply RoPE to `x`. Works with any batch shape prefix.
        
        Industrial implementation:
        - Duplicate cos/sin along full d_k dimension (already done in __init__)
        - Use rotate_half helper to avoid explicit even/odd splitting
        - Single fused computation: x * cos + rotate_half(x) * sin
        """
        
        # 1. 获取对应位置的 cos 和 sin 表
        # cos_cached 和 sin_cached 的形状是 [max_seq_len, d_k]
        # 通过 token_positions 索引后，形状变成 [..., seq_len, d_k]
        cos = self.cos_cached[token_positions]  # (..., seq_len, d_k)
        sin = self.sin_cached[token_positions]  # (..., seq_len, d_k)
        
        # 2. 应用旋转公式（一行搞定！）
        # 公式: x_rotated = x * cos + rotate_half(x) * sin
        # 这等价于:
        #   - 前半部分: x_even * cos - x_odd * sin
        #   - 后半部分: x_even * sin + x_odd * cos
        return x * cos + self.rotate_half(x) * sin
    
def softmax_stable(x:Tensor,dim: int =-1)->Tensor:
    x_max=x.max(dim=dim,keepdim=True).values
    x_exp=torch.exp(x-x_max)
    return x_exp/x_exp.sum(dim=dim,keepdim=True)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.scale = 1.0 / math.sqrt(d_k)

    def forward(
        self,
        query: Float[Tensor, "... seq_len_q d_k"],
        key: Float[Tensor, "... seq_len_k d_k"],
        value: Float[Tensor, "... seq_len_k d_v"],
        mask: Bool[Tensor, "seq_len_q seq_len_k"] = None
    ) -> Float[Tensor, "... seq_len_q d_v"]:
        attn_scores=einsum(query,key,"... q d,... k d -> ... q k")*self.scale
        if mask is not None:
            # masked_fill(condition, value) 会把 condition=True 的位置填充为 value
            attn_scores=attn_scores.masked_fill(~mask,float("-inf"))
        
        attn_probs=softmax_stable(attn_scores,dim=-1)
        output=einsum(attn_probs,value,"... q k,... k d_v->... q d_v")
        return output

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        use_rope: bool = True,
        device=None,
        dtype=None
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k  # match d_k for simplicity
        self.use_rope = use_rope

        factory_kwargs={'device':device,'dtype':dtype}
        self.q_proj,self.k_proj,self.v_proj,self.o_proj=[Linear(d_model,d_model,**factory_kwargs) for _ in range(4)]
        self.attn=ScaledDotProductAttention(self.d_k)

        # Create a causal mask for the attention mechanism
        # Shape: (1, 1, max_seq_len, max_seq_len)
        mask=torch.tril(torch.ones(max_seq_len,max_seq_len,dtype=bool,device=device))
        self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(0), persistent=False)
        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                theta=rope_theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)
    def forward(
        self,
        x: Float[Tensor, "batch seq_len d_model"],
        token_positions: Int[Tensor, "batch seq_len"]| None = None,
    ) -> Float[Tensor, "batch seq_len d_model"]:
        B, S, _ = x.shape
        q,k,v=[rearrange(proj(x),"b s (h d) -> b h s d ",h=self.num_heads)
            for proj in  [self.q_proj, self.k_proj, self.v_proj]]
        if self.use_rope:
            q,k=self.rope(q,token_positions),self.rope(k,token_positions)
        out=self.attn(q,k,v,mask=self.causal_mask[...,:S,:S])
        out=rearrange(out,"b h s d -> b s (h d)")
        return self.o_proj(out)
    
class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block with two sub-layers:

       x ──► RMSNorm ──► MHA ──► + ──►
         │                     ▲
         └─────────────────────┘     (sublayer-1)

       y ──► RMSNorm ──► FF  ──► + ──► out
         │                     ▲
         └─────────────────────┘     (sublayer-2)
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        rope_theta: float = 10_000.0,
        use_rope: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        kwargs = {"device": device, "dtype": dtype}

        # ── sub-layer 1: (RMSNorm → causal MHA) ──────────────────────────────
        self.norm1 = RMSNorm(d_model, **kwargs)
        self.attn  = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            use_rope=use_rope,
            **kwargs,
        )

        # ── sub-layer 2: (RMSNorm → feed-forward) ────────────────────────────
        self.norm2 = RMSNorm(d_model, **kwargs)
        self.ff    = SwiGLU(d_model=d_model, d_ff=d_ff, **kwargs)

    # -----------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,               # (batch, seq_len, d_model)
        token_positions: torch.Tensor | None = None,  # (batch, seq_len)
    ) -> torch.Tensor:
        b, s, _ = x.shape

        # ---- sub-layer-1: RMSNorm → MHA → residual -------------------------
        attn_out = self.attn(self.norm1(x), token_positions=token_positions)
        x = x + attn_out                       # residual connection

        # ---- sub-layer-2: RMSNorm → FF → residual --------------------------
        ff_out   = self.ff(self.norm2(x))
        x        = x + ff_out                  # residual connection
        return x    

def _copy_param(target: torch.Tensor, source: torch.Tensor) -> None:
    """
    Copy `source` into `target` in-place, transposing `source` if that
    is what makes the shapes line up.
    """
    if source.shape == target.shape:
        target.data.copy_(source)
    elif source.T.shape == target.shape:
        target.data.copy_(source.T)
    else:
        raise ValueError(f"Shape mismatch: cannot load parameter of shape {source.shape} "
                         f"into tensor of shape {target.shape}")
    
class TransformerLM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float,
                 device=None,
                 dtype=None):
        super().__init__()
        kw = dict(device=device, dtype=dtype)

        # token embedding  (no separate pos-emb: RoPE lives inside blocks)
        self.tok_emb = Embedding(vocab_size, d_model, **kw)

        # L Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                rope_theta=rope_theta,
                use_rope=True,
                **kw,
            )
            for _ in range(num_layers)
        ])

        # final norm
        self.ln_final = RMSNorm(d_model, **kw)
        self.lm_head = Linear(d_model, vocab_size, **kw)

        self.context_length = context_length

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        b, s = token_ids.shape
        if s > self.context_length:
            raise ValueError(f"seq_len {s} exceeds context_length {self.context_length}")

        # token embeddings
        x = self.tok_emb(token_ids)                         # (b, s, d)

        # token positions for RoPE
        pos = torch.arange(s, device=token_ids.device).unsqueeze(0).expand(b, s)

        # transformer stack
        for blk in self.blocks:
            x = blk(x, token_positions=pos)                # (b, s, d)

        # final norm → tied linear projection (logits)
        x = self.ln_final(x)                                 # (b, s, d)

        logits = self.lm_head(x)  # (b, s, vocab_size)
        return logits
