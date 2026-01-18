import torch
import numpy as np
import timeit
import triton
import triton.language as tl

# 三种注意力实现：标准pytorch实现，pytorch分块实现，triton实现
# 标准pytorch实现
def annotated_scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / d_k**0.5
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)
# Kernel 1: 计算 dQ
# 并行维度: (num_Q_blocks, batch_size)
# 每个线程块处理: 一个 Q 块，遍历所有 K 块

# Kernel 2: 计算 dK, dV  
# 并行维度: (num_K_blocks, batch_size)
# 每个线程块处理: 一个 K 块，遍历所有 Q 块
#         K_0    K_1    K_2    K_3
# Q_0 → [dQ_0] ← 累加来自所有 K 块的贡献
# Q_1
# Q_2
# Q_3

# 计算 dQ_0 时：
# - Q_0, dO_0, D_0, L_0 固定（加载一次）
# - 遍历 K_0, K_1, K_2, K_3（循环加载）
@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr,
    L_ptr, dO_ptr, dQ_ptr, D_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
    # 输入变量
    # Q_ptr, K_ptr, V_ptr     # 前向传播的输入（已保存）
    # L_ptr                    # logsumexp值，shape: (batch, seq_len_Q)
    # dO_ptr                   # 输出梯度，shape: (batch, seq_len_Q, D)
    # D_ptr                    # D_i = sum(O * dO, dim=-1)，shape: (batch, seq_len_Q)

    # # 输出变量
    # dQ_ptr                   # Q的梯度，shape: (batch, seq_len_Q, D)

    # # 中间变量
    # S                        # 注意力分数: Q @ K.T * scale
    # P                        # 注意力权重: exp(S - L)
    # dP                       # P的梯度: dO @ V.T
    # dS                       # S的梯度: P * (dP - D_i)
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape = (N_QUERIES, D),
        strides = (stride_qq, stride_qd),
        offsets = (query_tile_index*Q_TILE_SIZE, 0),
        block_shape = (Q_TILE_SIZE, D),
        order = (1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_qb,
        shape = (N_QUERIES, D),
        strides = (stride_qq, stride_qd),
        offsets = (query_tile_index*Q_TILE_SIZE, 0),
        block_shape = (Q_TILE_SIZE, D),
        order = (1, 0),
    )
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_qb,
        shape = (N_QUERIES, D),
        strides = (stride_qq, stride_qd),
        offsets = (query_tile_index*Q_TILE_SIZE, 0),
        block_shape = (Q_TILE_SIZE, D),
        order = (1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape = (N_KEYS, D),
        strides = (stride_kk, stride_kd),
        offsets = (0, 0), # 从头开始遍历
        block_shape = (K_TILE_SIZE, D),
        order = (1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape = (N_KEYS, D),
        strides = (stride_vk, stride_vd),
        offsets = (0, 0),
        block_shape = (K_TILE_SIZE, D),
        order = (1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape = (N_QUERIES, ),
        strides = (stride_lq, ),
        offsets = (query_tile_index*Q_TILE_SIZE,),
        block_shape = (Q_TILE_SIZE,),
        order = (0,),
    )    
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape = (N_QUERIES, ),
        strides = (stride_dq, ),
        offsets = (query_tile_index*Q_TILE_SIZE,),
        block_shape = (Q_TILE_SIZE,),
        order = (0,),
    )
    Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    dO = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
    D_i = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")
    l = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
    # 转换为 float32
    Q = Q.to(tl.float32)
    dO = dO.to(tl.float32)
    D_i = D_i.to(tl.float32)
    l = l.to(tl.float32)

    dS = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE),dtype=tl.float32) # 注意形状
    dP = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE),dtype=tl.float32)
    S = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE),dtype=tl.float32)
    P = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE),dtype=tl.float32)
    dQ = tl.zeros((Q_TILE_SIZE, D),dtype=tl.float32)
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        j_start = j * K_TILE_SIZE
        K = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        # 转换为 float32
        K = K.to(tl.float32)
        V = V.to(tl.float32)
        S = tl.dot(Q, tl.trans(K)) * scale
        if is_causal:
            q_idx = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE
            k_idx = tl.arange(0, K_TILE_SIZE) + j_start
            causal_mask = q_idx[:,None] >= k_idx[None, :]
            S = tl.where(causal_mask, S, S - 1e6)
        
        # 从数学公式看：
        # dQ = dS @ K * scale          # dS: (seq_len_Q, seq_len_K), K: (seq_len_K, D)
        # dK = dS.T @ Q * scale        # dS.T: (seq_len_K, seq_len_Q), Q: (seq_len_Q, D)  
        # dV = P.T @ dO                # P.T: (seq_len_K, seq_len_Q), dO: (seq_len_Q, D)
        P = tl.exp(S - l[:,None])
        dP = tl.dot(dO, tl.trans(V))
        dS = P * (dP - D_i[:,None])
        dQ += tl.dot(dS, K) * scale
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    tl.store(dQ_block_ptr, dQ.to(dQ_block_ptr.type.element_ty), boundary_check=(0, 1))

#         K_0    K_1    K_2    K_3
# Q_0
# Q_1                ↓
# Q_2              [dK_2] ← 累加来自所有 Q 块的贡献
# Q_3                ↓

# 计算 dK_2 时：
# - K_2, V_2 固定（加载一次）
# - 遍历 Q_0, Q_1, Q_2, Q_3（循环加载）
# - 同时遍历 dO_0, dO_1, dO_2, dO_3（循环加载）
# - 同时遍历 D_0, D_1, D_2, D_3（循环加载）
# - 同时遍历 L_0, L_1, L_2, L_3（循环加载）
@triton.jit
def flash_bwd_dk_dv_kernel(
    # dK 的第 j 行 依赖于 K 的第 j 行 和 所有 Q 的行
    # 每个 Q 块对应不同的 D_i, L_i, dO_i，这些值随 Q 块变化
    Q_ptr, K_ptr, V_ptr,
    L_ptr, dO_ptr, dK_ptr, dV_ptr, D_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape = (N_QUERIES, D),
        strides = (stride_qq, stride_qd),
        offsets = (0, 0), # Q从头开始遍历
        block_shape = (Q_TILE_SIZE, D),
        order = (1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_qb,
        shape = (N_QUERIES, D),
        strides = (stride_qq, stride_qd),
        offsets = (0, 0), # dO从头开始遍历
        block_shape = (Q_TILE_SIZE, D),
        order = (1, 0),
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_kb,
        shape = (N_KEYS, D),
        strides = (stride_kk, stride_kd),
        offsets = (key_tile_index*K_TILE_SIZE, 0),
        block_shape = (K_TILE_SIZE, D),
        order = (1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_vb,
        shape = (N_KEYS, D),
        strides = (stride_vk, stride_vd),
        offsets = (key_tile_index*K_TILE_SIZE, 0),
        block_shape = (K_TILE_SIZE, D),
        order = (1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape = (N_KEYS, D),
        strides = (stride_kk, stride_kd),
        offsets = (key_tile_index*K_TILE_SIZE, 0),
        block_shape = (K_TILE_SIZE, D),
        order = (1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape = (N_KEYS, D),
        strides = (stride_vk, stride_vd),
        offsets = (key_tile_index*K_TILE_SIZE, 0),
        block_shape = (K_TILE_SIZE, D),
        order = (1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape = (N_QUERIES, ),
        strides = (stride_lq, ),
        offsets = (0,),
        block_shape = (Q_TILE_SIZE,),
        order = (0,),
    )    
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape = (N_QUERIES, ),
        strides = (stride_dq, ),
        offsets = (0,),
        block_shape = (Q_TILE_SIZE,),
        order = (0,),
    )
    K = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)
    V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)
    # 转换为 float32
    K = K.to(tl.float32)
    V = V.to(tl.float32)
    dS = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE),dtype=tl.float32)
    dP = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE),dtype=tl.float32)
    S = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE),dtype=tl.float32)
    P = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE),dtype=tl.float32)
    dK = tl.zeros((K_TILE_SIZE, D),dtype=tl.float32)
    dV = tl.zeros((K_TILE_SIZE, D),dtype=tl.float32)
    for j in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)): # 注意循环条件，以Q作为内循环
        j_start = j * Q_TILE_SIZE
        Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, D),
        dO = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, D),
        D_i = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")# (Q_TILE_SIZE,),
        l = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        # 转换为 float32
        Q = Q.to(tl.float32)
        dO = dO.to(tl.float32)
        D_i = D_i.to(tl.float32)
        l = l.to(tl.float32)
        
        S = tl.dot(Q, tl.trans(K)) * scale # (Q_TILE_SIZE, K_TILE_SIZE)
        if is_causal:
            q_idx = tl.arange(0, Q_TILE_SIZE) + j_start
            k_idx = tl.arange(0, K_TILE_SIZE) + K_TILE_SIZE * key_tile_index
            causal_mask = q_idx[:,None] >= k_idx[None, :]
            S = tl.where(causal_mask, S, S - 1e6)
        P = tl.exp(S - l[:,None])
        dV += tl.dot(tl.trans(P),dO) # (K_TILE_SIZE, D)
        dP = tl.dot(dO, tl.trans(V)) # (Q_TILE_SIZE, K_TILE_SIZE)
        dS = P * (dP - D_i[:,None]) # (Q_TILE_SIZE, K_TILE_SIZE)
        dK += tl.dot(tl.trans(dS),Q) * scale
        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
    tl.store(dK_block_ptr, dK.to(dK_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(dV_block_ptr, dV.to(dV_block_ptr.type.element_ty), boundary_check=(0, 1))


# Triton kernel for flash attention forward pass
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, # 输入指针
    O_ptr, L_ptr,# 输出指针，L式logsumexp需要保存用于反向传播
    stride_qb, stride_qq, stride_qd, # Q的stride，包括跳一个batch，跳一个query，跳一个dimension所需要的步长
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq, # L的stride，二维tensor
    N_QUERIES, N_KEYS, # 一般等于seq_len
    scale, # 1/sqrt(d)
    D: tl.constexpr, # d维度大小
    Q_TILE_SIZE: tl.constexpr, # 每次处理的query块大小
    K_TILE_SIZE: tl.constexpr, # 每次处理的key value块大小
    is_causal: tl.constexpr # 是否是causal attention，需要掩码
):
    query_tile_index = tl.program_id(0) # 第几个query块，第0维度并行性更好，用来划分query块
    batch_index = tl.program_id(1) # 第几个batch，第1维度并行性较差，用来划分batch
    #     tl.make_block_ptr(
    #     base,         # 基地址（指针）
    #     shape,        # 整个张量的形状
    #     strides,      # 步长（stride）
    #     offsets,      # 块的起始位置
    #     block_shape,  # 块的大小
    #     order         # 内存布局顺序
    # )
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb, # batch维度偏移，这个batch的起始位置
        shape = (N_QUERIES, D), # Q的整体shape，即(seq_len, d),也是（N_QUERIES, D)，用于边界检查
        strides = (stride_qq, stride_qd),
        offsets = (query_tile_index*Q_TILE_SIZE, 0), # 每个query块的起始位置
        block_shape = (Q_TILE_SIZE, D), # 每个块的shape
        order = (1, 0), # 内存布局顺序，遍历顺序，这里是d维度优先
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape = (N_KEYS, D),
        strides = (stride_kk, stride_kd),
        offsets = (0, 0), # K, V的offsets都应该从0开始，随j遍历是内循环
        block_shape = (K_TILE_SIZE, D),
        order = (1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape = (N_KEYS, D),
        strides = (stride_vk, stride_vd),
        offsets = (0, 0),
        block_shape = (K_TILE_SIZE, D),
        order = (1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape = (N_QUERIES, D),
        strides = (stride_oq, stride_od),
        offsets = (query_tile_index*Q_TILE_SIZE, 0),
        block_shape = (Q_TILE_SIZE, D),
        order = (1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape = (N_QUERIES, ), # L是一维tensor，shape是(seq_len,)
        strides = (stride_lq, ),
        offsets = (query_tile_index*Q_TILE_SIZE,),
        block_shape = (Q_TILE_SIZE,),
        order = (0,), # 一维tensor只有一个维度
    )
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32) # 需要初始化，注o数据类型
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,),  -float("inf"), dtype=tl.float32) # tl.full的使用，-float("inf")

    #  加载Q块，有边界检查
    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    Q_i = tl.cast(Q_i, tl.float32) # 数据类型转换

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # j_start表示当前块在key维度的起始位置
        j_start = j * K_TILE_SIZE  
        # 创建 key index 向量并判断哪些是有效（< N_KEYS）
        # 这两行代码的作用是创建有效性掩码（validity mask），用于处理序列长度不能被块大小整除的边界情况
        # 具体来说：
        # j=3, j_start = 3 * 32 = 96
        # k_idx = tl.arange(0, 32) + 96
        # k_idx = [96, 97, 98, 99, 100, 101, ..., 127]
        # shape: (32,)

        # N_KEYS = 100
        # valid_k = k_idx < 100
        # valid_k = [True, True, True, True, False, False, ..., False]
        #           (96-99 为 True，100-127 为 False)
        # shape: (32,)
        k_idx = tl.arange(0, K_TILE_SIZE) + j_start  # shape (K_TILE_SIZE,)
        valid_k = k_idx < N_KEYS                      # boolean mask shape (K_TILE_SIZE,)
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        K_j = tl.cast(K_j, tl.float32) # 数据类型
        V_j = tl.cast(V_j, tl.float32)
        S_i = tl.dot(Q_i, tl.trans(K_j)) * scale
        # mask operation 
        if is_causal:
            # 计算块中元素在S中的的行数
            q_idx = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE  # shape (Q_TILE_SIZE,)
            # 计算块中元素在S中的的列数 
            k_idx_tile = tl.arange(0, K_TILE_SIZE) + j_start  # shape (K_TILE_SIZE,)
            # 行数大于列数就被mask，合理利用广播机制
            causal_mask = q_idx[:, None] >= k_idx_tile[None, :]  # shape (Q_TILE_SIZE, K_TILE_SIZE)
            #        Key 位置 →
            #         k4  k5  k6  k7
            # Query q4 [T,  F,  F,  F]  ← query 4 只能看到 key 4 (自己)
            # 位置  q5 [T,  T,  F,  F]  ← query 5 可以看到 key 4-5
            # ↓     q6 [T,  T,  T,  F]  ← query 6 可以看到 key 4-6
            #       q7 [T,  T,  T,  T]  ← query 7 可以看到 key 4-7

            # 解释：
            # q4 >= k4 → T,  q4 >= k5 → F,  q4 >= k6 → F,  q4 >= k7 → F
            # q5 >= k4 → T,  q5 >= k5 → T,  q5 >= k6 → F,  q5 >= k7 → F
            # q6 >= k4 → T,  q6 >= k5 → T,  q6 >= k6 → T,  q6 >= k7 → F
            # q7 >= k4 → T,  q7 >= k5 → T,  q7 >= k6 → T,  q7 >= k7 → T
            # Apply causal mask: add -1e6 to masked out elements
            S_i = tl.where(causal_mask, S_i, S_i - 1e6)
            # tl.where(condition, if_true, if_false)
            # 如果 causal_mask[i, j] == True:  保持 S_i[i, j]
            # 如果 causal_mask[i, j] == False: S_i[i, j] = S_i[i, j] - 1e6
        
        # 广播应用：整列掩码
        # valid_k[None, :] -> [1, K_TILE_SIZE]
        #        k96 k97 k98 k99 k100 k101 ... k127
        # q0  [[ T,  T,  T,  T,  F,   F,  ... F],   # 每列统一
        # q1   [ T,  T,  T,  T,  F,   F,  ... F],
        # q2   [ T,  T,  T,  T,  F,   F,  ... F],
        # q3   [ T,  T,  T,  T,  F,   F,  ... F]]
        S_i = tl.where(valid_k[None,:], S_i, -float("inf"))
        m_i_old = m_i
        m_i = tl.maximum(m_i_old, tl.max(S_i, axis=-1))
        P_i = tl.exp(S_i - m_i[:, None]) # triton中不能用...,None
        l_i = tl.exp(m_i_old - m_i) * l_i + tl.sum(P_i, axis=-1)
        # O_i = tl.exp(m_i_old - m_i)[:, None] * O_i + tl.dot(P_i, V_j)
        # 优化版本（使用 acc）
        O_i = tl.exp(m_i_old - m_i)[:, None] * O_i
        O_i = tl.dot(P_i, V_j, acc=O_i)  # 累积到 O_i
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0)) # 相当于移动offset
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    O_i = O_i / l_i[:, None]
    L_i = m_i + tl.log(l_i) # 真实的logsumexp值
    # 将 O_i 转换回原始数据类型（与 Q 相同）
    tl.store(O_block_ptr, O_i.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, L_i.to(L_block_ptr.type.element_ty), boundary_check=(0,))

# Pytorch autograd Function for flash attention implemented in pytorch
class Flash_attention_pytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        ctx.is_causal = is_causal
        N_q, d = Q.shape[-2:]
        ctx.scale = 1 / d ** 0.5
        N_k = K.shape[-2]
        B_q = 32
        B_k = 32  # Reduced to match Triton implementation

        # 初始化输出和中间变量
        O = torch.zeros_like(Q,device=Q.device)
        l = torch.zeros(Q.shape[:-1],device=Q.device)
        m = torch.zeros(Q.shape[:-1],device=Q.device) - torch.inf
        # logsumexp trick需要的变量,后续用于反向传播
        L = torch.zeros(Q.shape[:-1],device=Q.device)

        # 分快处理查询和键值对即i_max,j_max
        i_max = int(np.ceil(N_q / B_q))
        j_max = int(np.ceil(N_k / B_k))
        for i in range(i_max):
            # 分块处理查询，在seq_len维度上切片
            Q_i = Q[...,i*B_q:(i+1)*B_q,:]
            O_i = O[...,i*B_q:(i+1)*B_q,:]
            # l_i, m_i, L_i分别是最后一个维度的exp和，最大值，logsumexp
            l_i = l[...,i*B_q:(i+1)*B_q]
            m_i = m[...,i*B_q:(i+1)*B_q]
            L_i = L[...,i*B_q:(i+1)*B_q]

            for j in range(j_max):
                # 分块处理键值对，在seq_len维度上切片，不过一块是B_k大小
                K_j = K[...,j*B_k:(j+1)*B_k,:]
                V_j = V[...,j*B_k:(j+1)*B_k,:]
                # 计算相似度得分矩阵
                S_i = (Q_i @ K_j.transpose(-1,-2)) / d ** 0.5
                # 如果是因果注意力，需要对S_i进行mask，此处暂时不考虑mask
                # 存储旧的m_i用于更新
                m_i_old = m_i.clone()
                m_i = torch.max(m_i_old,torch.max(S_i,dim=-1)[0])
                # 增加m_i最后一个维度以便广播
                P_i = torch.exp(S_i - m_i[...,None])
                l_i = torch.exp(m_i_old - m_i) * l_i + torch.sum(P_i,dim=-1,keepdim=False)
                # 增加m_i最后一个维度以便广播
                O_i = torch.exp(m_i_old - m_i)[...,None] * O_i + P_i @ V_j
                # 更新m和l到对应位置
                m[...,i*B_q:(i+1)*B_q] = m_i
                l[...,i*B_q:(i+1)*B_q] = l_i
            
            # 最终归一化全局的O_i
            O_i = O_i / l_i[...,None]
            O[...,i*B_q:(i+1)*B_q,:] = O_i
            # 计算L_i用于反向传播
            L_i = m_i + torch.log(l_i)
            L[...,i*B_q:(i+1)*B_q] = L_i
        ctx.save_for_backward(L, Q, K, V, O)
        return O
    @staticmethod
    def backward(ctx, grad_O):
        L, Q, K, V, O = ctx.saved_tensors
        scale = ctx.scale
        is_causal = ctx.is_causal
        batch_size,N_QUERIES = Q.shape[:-1]
        N_KEYS = K.shape[-2]
        # D_i should be the sum of element-wise multiplication along the last dimension
        D_i = torch.sum(O * grad_O, dim=-1)  # Shape: (batch_size, seq_len)
        S = (Q @ K.transpose(-1, -2)) * scale # 需要对S进行causal mask，后面的P_ij, dS_ij自动就mask了
        if is_causal:
            causal_mask = torch.arange(N_QUERIES)[...,None] >= torch.arange(N_KEYS)[...,None,:]
            causal_mask = causal_mask.to(Q.device)
            S = torch.where(causal_mask, S, torch.full_like(S, -1e6))
        P_ij = torch.exp(S - L[...,None])
        dV = P_ij.transpose(-1, -2) @ grad_O
        dP = grad_O @ V.transpose(-1, -2)
        dS_ij = P_ij * (dP - D_i[...,None])
        dQ = (dS_ij @ K) * scale
        dK = (dS_ij.transpose(-1, -2) @ Q) * scale
        return dQ, dK, dV, None # 输出的梯度个数必须与输入参数数量一致，is_causal的梯度应该是None

# Triton autograd Function for flash attention implemented in Triton
class Flash_attention_triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Tensors must be on CUDA"
        # ensure contiguous to make strides consistent
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
        original_shape = Q.shape
        ctx.original_shape = original_shape
        N_QUERIES, D = Q.shape[-2:]
        N_KEYS = K.shape[-2]
        
        def next_power_of_2(x):
            return 1 << (x - 1).bit_length()
        
        # 填充是为了让D是2的幂次方，方便triton处理，获取最佳性能
        D_padded = next_power_of_2(D)
        ctx.D_original = D
        ctx.D_padded = D_padded
        
        # Pad tensors if necessary
        if D_padded != D:
            pad_size = D_padded - D
            Q = torch.nn.functional.pad(Q, (0, pad_size), mode='constant', value=0)
            K = torch.nn.functional.pad(K, (0, pad_size), mode='constant', value=0)
            V = torch.nn.functional.pad(V, (0, pad_size), mode='constant', value=0)
        
        Q = Q.view(-1, N_QUERIES, D_padded) # 这一步必须要做，不然会有illegal memory access的问题
        K = K.view(-1, N_KEYS, D_padded)
        V = V.view(-1, N_KEYS, D_padded)
        batch_size = Q.shape[0]
        scale = 1 / (D ** 0.5)  # Use original D for scale
        O = torch.empty((batch_size, N_QUERIES, D_padded), device=Q.device, dtype=Q.dtype)
        L = torch.empty((batch_size, N_QUERIES), device=Q.device, dtype=Q.dtype)
        # tile_size = 32 if D > 64 else 64
        Q_TILE_SIZE = 16  # Reduced from 256 to 64， 不能太大了，shared memory空间有限
        K_TILE_SIZE = 16 # Reduced from 256 to 64
        Q_TILE_SIZE = min(Q_TILE_SIZE, N_QUERIES)  #防止tile size大于seq_len然后越界
        K_TILE_SIZE = min(K_TILE_SIZE, N_KEYS)
        grid = (triton.cdiv(N_QUERIES, Q_TILE_SIZE), batch_size)
        ctx.is_causal = is_causal
        ctx.scale = scale
        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES, N_KEYS,
            scale,
            D_padded,
            Q_TILE_SIZE = Q_TILE_SIZE,
            K_TILE_SIZE = K_TILE_SIZE,
            is_causal = is_causal
        )
        ctx.save_for_backward(L, Q, K, V, O)
        
        # Remove padding from output before returning
        if D_padded != D:
            O = O[..., :D]
        
        return O.view(original_shape)

    @staticmethod
    def backward(ctx, grad_O):
        L, Q, K, V, O = ctx.saved_tensors
        scale = ctx.scale
        is_causal = ctx.is_causal
        original_shape = ctx.original_shape
        D_original = ctx.D_original
        D_padded = ctx.D_padded
        
        batch_size, N_QUERIES, _ = Q.shape  # Q is already padded
        N_KEYS = K.shape[-2]
        
        # Pad grad_O if necessary
        grad_O = grad_O.view(-1, N_QUERIES, D_original)
        if D_padded != D_original:
            pad_size = D_padded - D_original
            grad_O = torch.nn.functional.pad(grad_O, (0, pad_size), mode='constant', value=0)
        
        D_i = torch.sum(O * grad_O, dim=-1)  # Shape: (batch_size, seq_len)
        # triton backward
        Q_TILE_SIZE = 16  # Reduced from 256 to 64， 不能太大了，shared memory空间有限
        K_TILE_SIZE = 16 # Reduced from 256 to 64
        dQ = torch.empty((batch_size, N_QUERIES, D_padded), device=Q.device, dtype=Q.dtype)
        dK = torch.empty((batch_size, N_KEYS, D_padded), device=Q.device, dtype=Q.dtype)
        dV = torch.empty((batch_size, N_KEYS, D_padded), device=Q.device, dtype=Q.dtype)
        # compute dQ
        grid = (triton.cdiv(N_QUERIES, Q_TILE_SIZE), batch_size)
        flash_bwd_dq_kernel[grid](
            Q, K, V,
            L, grad_O, dQ, D_i, 
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            L.stride(0), L.stride(1),
            D_i.stride(0), D_i.stride(1),
            N_QUERIES, N_KEYS,
            scale,
            D_padded,
            Q_TILE_SIZE = Q_TILE_SIZE,
            K_TILE_SIZE = K_TILE_SIZE,
            is_causal = is_causal
        )
        # compute dK, dV
        grid = (triton.cdiv(N_KEYS, K_TILE_SIZE), batch_size)
        flash_bwd_dk_dv_kernel[grid](
            Q, K, V,
            L, grad_O, dK, dV, D_i,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            L.stride(0), L.stride(1),
            D_i.stride(0), D_i.stride(1),
            N_QUERIES, N_KEYS,
            scale,
            D_padded,
            Q_TILE_SIZE = Q_TILE_SIZE,
            K_TILE_SIZE = K_TILE_SIZE,
            is_causal = is_causal
        )
        
        # Remove padding from gradients before returning
        if D_padded != D_original:
            dQ = dQ[..., :D_original]
            dK = dK[..., :D_original]
            dV = dV[..., :D_original]
        
        return dQ.view(original_shape), dK.view(original_shape), dV.view(original_shape), None
        #################
        # S = (Q @ K.transpose(-1, -2)) * scale # 需要对S进行causal mask，后面的P_ij, dS_ij自动就mask了

def apply_flash_atn_pt(Q, K, V, is_causal):
    return Flash_attention_pytorch.apply(Q, K, V, is_causal)

