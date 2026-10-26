# from sympy import Q
import torch
import triton
import triton.language as tl

@triton.jit
def flash_attention_fwd_kernel(
    # -------- 全局内存指针 --------
    Q_ptr, K_ptr, V_ptr,           # 输入 Q, K, V
    O_ptr, L_ptr,                  # 输出 O, logsumexp L

    # -------- stride 信息（用于 block ptr）--------
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_on, stride_od,
    stride_lb, stride_ln,

    # -------- 序列长度 --------
    Nq, Nk,

    # -------- scale --------
    scale,

    # -------- 编译期常量 --------
    D: tl.constexpr,
    Q_BLOCK: tl.constexpr,
    K_BLOCK: tl.constexpr,
    is_causal: tl.constexpr,
):
    """
    每个 Triton program 负责：
        - 一个 batch
        - 一个 Query block（Q_BLOCK 行）
    """

    # ------------------------------------------------
    # Triton 并行索引
    # ------------------------------------------------
    q_block_id = tl.program_id(0)     # 第几个 Q block
    batch_id   = tl.program_id(1)     # 第几个 batch

    # =================================================
    # 构造 block pointer（这是 Triton 的关键）
    # =================================================
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_id * stride_qb,
        shape=(Nq, D),
        strides=(stride_qn, stride_qd),
        offsets=(q_block_id * Q_BLOCK, 0),
        block_shape=(Q_BLOCK, D),
        order=(1, 0),
    )

    #  K / V 从第 0 行开始，后面在 loop 中 advance
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_id * stride_kb,
        shape=(Nk, D),
        strides=(stride_kn, stride_kd),
        offsets=(0, 0),
        block_shape=(K_BLOCK, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_id * stride_vb,
        shape=(Nk, D),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(K_BLOCK, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_id * stride_ob,
        shape=(Nq, D),
        strides=(stride_on, stride_od),
        offsets=(q_block_id * Q_BLOCK, 0),
        block_shape=(Q_BLOCK, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_id * stride_lb,
        shape=(Nq, 1),
        strides=(stride_ln, 1),
        offsets=(q_block_id * Q_BLOCK, 0),
        block_shape=(Q_BLOCK, 1),
        order=(1, 0),
    )

    # =================================================
    # 加载 Query block
    # =================================================
    Q_i = tl.load(Q_block_ptr)  # (Q_BLOCK, D)

    # =================================================
    # FlashAttention 核心状态（每个 Q block 独立）
    # =================================================
    O_acc = tl.zeros((Q_BLOCK, D), dtype=tl.float32)       # 未归一化输出累积
    L_acc = tl.zeros((Q_BLOCK, 1), dtype=tl.float32)      # softmax 分母
    M_acc = tl.full((Q_BLOCK, 1), -float("inf"), tl.float32)  # running max

    # =================================================
    # 内层循环：遍历所有 K/V block
    # =================================================
    for k_block_id in range(tl.cdiv(Nk, K_BLOCK)):
        K_j = tl.load(K_block_ptr)   # (K_BLOCK, D)
        V_j = tl.load(V_block_ptr)   # (K_BLOCK, D)

        # ---------------------------------------------
        # S_ij = Q_i @ K_j^T / sqrt(d)
        # ---------------------------------------------
        S_ij = tl.dot(Q_i, K_j.T, input_precision="ieee") * scale  # (Q_BLOCK, K_BLOCK)

        # ---------------------------------------------
        # Causal mask（编译期 if）
        # ---------------------------------------------
        if is_causal:
            q_idx = q_block_id * Q_BLOCK + tl.arange(0, Q_BLOCK)[:, None]
            k_idx = k_block_id * K_BLOCK + tl.arange(0, K_BLOCK)[None, :]
            causal_mask = q_idx >= k_idx
            S_ij = tl.where(causal_mask, S_ij, -1e6)

        # ---------------------------------------------
        # 数值稳定 softmax（FlashAttention 核心）
        # ---------------------------------------------
        M_block = tl.max(S_ij, axis=1, keep_dims=True)
        M_new = tl.maximum(M_acc, M_block)

        P_ij = tl.exp(S_ij - M_block)

        L_new = (
            tl.exp(M_acc - M_new) * L_acc +
            tl.exp(M_block - M_new) * tl.sum(P_ij, axis=1, keep_dims=True)
        )

        # 类型对齐（Triton 细节）
        P_cast = P_ij.to(V_block_ptr.type.element_ty)

        O_new = (
            tl.exp(M_acc - M_new) * O_acc +
            tl.exp(M_block - M_new) * tl.dot(P_cast, V_j, input_precision="ieee")
        )

        # ---------------------------------------------
        # 更新 running 状态
        # ---------------------------------------------
        M_acc = M_new
        L_acc = L_new
        O_acc = O_new

        # ---------------------------------------------
        # 移动 K/V block 指针
        # ---------------------------------------------
        K_block_ptr = K_block_ptr.advance((K_BLOCK, 0))
        V_block_ptr = V_block_ptr.advance((K_BLOCK, 0))

    # =================================================
    # softmax 归一化
    # =================================================
    O_i = O_acc / L_acc
    L_i = M_acc + tl.log(L_acc)

    tl.store(O_block_ptr, O_i)
    tl.store(L_block_ptr, L_i)

@triton.jit
def flash_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, dO_ptr, D_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_dob, stride_doq, stride_dod,
    stride_db, stride_dq_d,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # program ids
    k_tile_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    # -------------------------
    # block pointers (Q-side)
    # -------------------------
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_id * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_id * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_id * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_id * stride_lb,
        shape=(N_QUERIES, 1),
        strides=(stride_lq, 1),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, 1),
        order=(1, 0),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_id * stride_db,
        shape=(N_QUERIES, 1),
        strides=(stride_dq_d, 1),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, 1),
        order=(1, 0),
    )

    # -------------------------
    # block pointers (K/V-side)
    # -------------------------
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_id * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(k_tile_id * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_id * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(k_tile_id * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_id * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(k_tile_id * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_id * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(k_tile_id * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # -------------------------
    # load K/V tile
    # -------------------------
    K_j = tl.load(K_block_ptr)  # (K_TILE_SIZE, D)
    V_j = tl.load(V_block_ptr)

    dK_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dV_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

    # -------------------------
    # loop over Q tiles
    # -------------------------
    for q_tile_id in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        Q_i = tl.load(Q_block_ptr)
        O_i = tl.load(O_block_ptr)
        dO_i = tl.load(dO_block_ptr)
        L_i = tl.load(L_block_ptr)
        D_i = tl.load(D_block_ptr)

        # S = QK^T
        S_ij = tl.dot(Q_i, K_j.T, input_precision="ieee") * scale

        if is_causal:
            q_idx = q_tile_id * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)[:, None]
            k_idx = k_tile_id * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)[None, :]
            S_ij = tl.where(q_idx >= k_idx, S_ij, -1e6)

        # P = softmax
        P_ij = tl.exp(S_ij - L_i)

        # dV
        dV_j += tl.dot(P_ij.to(V_j.dtype).T, dO_i)

        # dP
        dP_ij = tl.dot(dO_i, V_j.T)

        # dS
        dS_ij = P_ij * (dP_ij - D_i)
        dS_ij = dS_ij * scale

        # dQ (atomic add)
        dQ_i = tl.dot(dS_ij, K_j)

        q_idx = q_tile_id * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)[:, None]
        d_idx = tl.arange(0, D)[None, :]
        mask = (q_idx < N_QUERIES) & (d_idx < D)
        dq_ptrs = dQ_ptr + batch_id * stride_qb + q_idx * stride_qq + d_idx * stride_qd
        tl.atomic_add(dq_ptrs, dQ_i.to(dQ_ptr.dtype.element_ty), mask=mask)

        # dK
        dK_j += tl.dot(dS_ij.T, Q_i)

        # advance Q-side blocks
        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        O_block_ptr = O_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE, 0))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE, 0))

    # -------------------------
    # store dK / dV
    # -------------------------
    tl.store(dK_block_ptr, dK_j.to(dK_ptr.dtype.element_ty))
    tl.store(dV_block_ptr, dV_j.to(dV_ptr.dtype.element_ty))



class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Q, K, V: (B, N, D)
        """

        B, Nq, D = Q.shape
        Nk = K.shape[1]

        Q_BLOCK = 16
        K_BLOCK = 16

        scale = D ** -0.5

        O = torch.empty_like(Q)
        L = torch.empty(B, Nq, device=Q.device)

        grid = (Nq // Q_BLOCK, B)

        flash_attention_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            Nq, Nk,
            scale,
            D=D,
            Q_BLOCK=Q_BLOCK,
            K_BLOCK=K_BLOCK,
            is_causal=is_causal,
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O
    @staticmethod
    def backward(ctx, do):
        q, k, v, o, l = ctx.saved_tensors
        is_causal = ctx.is_causal
    
        B, Nq, D = q.shape
        Nk = k.shape[1]
    
        Bq = 16
        Bk = 16
        scale = D ** -0.5
    
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
    
        # D_i = sum_j dO_ij * O_ij
        D_sum = torch.sum(o * do, dim=-1, keepdim=True)
    
        grid = (triton.cdiv(Nk, Bk), B)
    
        flash_bwd_kernel[grid](
            q, k, v, o, l, do, D_sum,
            dq, dk, dv,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            l.stride(0), l.stride(1),
            do.stride(0), do.stride(1), do.stride(2),
            D_sum.stride(0), D_sum.stride(1),
            Nq, Nk,
            scale,
            D=D,
            Q_TILE_SIZE=Bq,
            K_TILE_SIZE=Bk,
            is_causal=is_causal,
        )
    
        return dq, dk, dv, None
    
