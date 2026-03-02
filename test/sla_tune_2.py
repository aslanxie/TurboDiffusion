import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl
import statistics  # for median

# ────────────────────────────────────────────────────────────────
#   compress_kernel and mean_pool  (unchanged)
# ────────────────────────────────────────────────────────────────

@triton.jit
def compress_kernel(
    X, XM,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    idx_l = tl.program_id(0)
    idx_bh = tl.program_id(1)

    offs_l = idx_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, D)

    x_offset = idx_bh * L * D
    xm_offset = idx_bh * ((L + BLOCK_L - 1) // BLOCK_L) * D
    x = tl.load(X + x_offset + offs_l[:, None] * D + offs_d[None, :], mask=offs_l[:, None] < L)

    nx = min(BLOCK_L, L - idx_l * BLOCK_L)
    x_mean = tl.sum(x, axis=0, dtype=tl.float32) / nx
    tl.store(XM + xm_offset + idx_l * D + offs_d, x_mean.to(XM.dtype.element_ty))


def mean_pool(x, BLK):
    assert x.is_contiguous()
    B, H, L, D = x.shape
    L_BLOCKS = (L + BLK - 1) // BLK
    x_mean = torch.empty((B, H, L_BLOCKS, D), device=x.device, dtype=x.dtype)
    grid = (L_BLOCKS, B * H)
    compress_kernel[grid](x, x_mean, L, D, BLK)
    return x_mean


def get_block_map(q, k, topk_ratio, BLKQ, BLKK):
    arg_k = k - torch.mean(k, dim=-2, keepdim=True)
    pooled_qblocks = mean_pool(q, BLKQ)
    pooled_kblocks = mean_pool(arg_k, BLKK)
    pooled_score = pooled_qblocks @ pooled_kblocks.transpose(-1, -2)

    K = pooled_score.shape[-1]
    topk = min(K, int(topk_ratio * K))
    lut = torch.topk(pooled_score, topk, dim=-1, sorted=False).indices     # shape: (B,H,M_blocks,topk)

    # ─── NEW: sort each row of LUT ─────────────────────────────────────
    sort_idx = torch.argsort(lut, dim=-1)                                   # (B,H,M_blocks,topk)
    lut = torch.gather(lut, dim=-1, index=sort_idx)                         # now sorted by block id

    sparse_map = torch.zeros_like(pooled_score, dtype=torch.int8)
    sparse_map.scatter_(-1, lut, 1)   # note: sparse_map doesn't need sorting
    # ────────────────────────────────────────────────────────────────────

    return sparse_map, lut, topk


# ────────────────────────────────────────────────────────────────
#   Kernel  (same as before, but now we pass BLOCK_M / BLOCK_N explicitly)
# ────────────────────────────────────────────────────────────────

@triton.jit
def _attn_fwd(
    Q, K, V,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    LUT, OS,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    qkv_offset = idx_bh * L * D
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    LUT_ptr = LUT + lut_offset

    # Q descriptor
    q_base = Q + qkv_offset
    q_desc = tl.make_tensor_descriptor(
        base=q_base, shape=(L, D), strides=(D, 1), block_shape=(BLOCK_M, D),
    )
    coord_m = (idx_m * BLOCK_M).to(tl.int32)
    q = q_desc.load([coord_m, 0])  # safe because of padding

    # K and V descriptors
    k_base = K + qkv_offset
    v_base = V + qkv_offset

    k_desc = tl.make_tensor_descriptor(
        base=k_base, shape=(L, D), strides=(D, 1), block_shape=(BLOCK_N, D),
    )
    v_desc = tl.make_tensor_descriptor(
        base=v_base, shape=(L, D), strides=(D, 1), block_shape=(BLOCK_N, D),
    )

    # Output descriptor
    o_base = OS + qkv_offset
    o_desc = tl.make_tensor_descriptor(
        base=o_base, shape=(L, D), strides=(D, 1), block_shape=(BLOCK_M, D),
    )

    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_s = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    scale_factor = qk_scale * 1.4426950408889634

    for block_idx in tl.range(topk):
        idx_n = tl.load(LUT_ptr + block_idx).to(tl.int64)
        coord = (idx_n * BLOCK_N).to(tl.int32)

        k_tile = k_desc.load([coord, 0])
        v_tile = v_desc.load([coord, 0])

        qk = tl.dot(q, tl.trans(k_tile)) * scale_factor

        local_m = tl.max(qk, axis=1)
        new_m = tl.maximum(m_i, local_m)

        alpha = tl.math.exp2(m_i - new_m)
        qk_scaled = qk - new_m[:, None]
        p = tl.math.exp2(qk_scaled)

        l_ij = tl.sum(p, axis=1)
        o_s = o_s * alpha[:, None]
        o_s += tl.dot(p.to(v_tile.dtype), v_tile)

        l_i = l_i * alpha + l_ij
        m_i = new_m

    o_s = o_s / l_i[:, None]
    o_desc.store([coord_m, 0], o_s.to(OS.dtype.element_ty))


# ────────────────────────────────────────────────────────────────
#   Model
# ────────────────────────────────────────────────────────────────

class SparseLinearAttention(nn.Module):
    def __init__(self, head_dim, topk=0.1, feature_map='softmax', use_bf16=True, tie_feature_map_qk=True):
        super().__init__()
        self.dtype = torch.bfloat16 if use_bf16 else torch.float16
        self.topk = topk
        self.proj_l = nn.Linear(head_dim, head_dim, dtype=torch.bfloat16)

        if feature_map == 'elu':
            def elu_feature_map(x): return F.elu(x) + 1
            self.feature_map_q = self.feature_map_k = elu_feature_map
        elif feature_map == 'relu':
            self.feature_map_q = self.feature_map_k = nn.ReLU()
        elif feature_map == 'softmax':
            def softmax_feature_map(x): return F.softmax(x, dim=-1)
            self.feature_map_q = self.feature_map_k = softmax_feature_map
        else:
            raise ValueError(f"Unknown feature_map: {feature_map}")

        if tie_feature_map_qk:
            self.feature_map_k = self.feature_map_q

        nn.init.normal_(self.proj_l.weight)
        nn.init.normal_(self.proj_l.bias)

    def forward(self, q, k, v, BLKQ, BLKK, num_warps, num_stages, return_sparsity=False):
        torch.xpu.synchronize()
        t0 = time.time()

        orig_dtype = q.dtype
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        B, H, seq_len, D = q.shape

        # Pad to multiple of BLKQ and BLKK
        seq_len_padded_q = triton.cdiv(seq_len, BLKQ) * BLKQ
        seq_len_padded_kv = triton.cdiv(seq_len, BLKK) * BLKK
        pad_q = seq_len_padded_q - seq_len
        pad_kv = seq_len_padded_kv - seq_len

        if pad_q > 0:
            q = F.pad(q, (0, 0, 0, pad_q), value=0.0)
        if pad_kv > 0:
            k = F.pad(k, (0, 0, 0, pad_kv), value=0.0)
            v = F.pad(v, (0, 0, 0, pad_kv), value=0.0)

        L_padded = max(seq_len_padded_q, seq_len_padded_kv)

        sparse_map, lut, real_topk = get_block_map(q, k, self.topk, BLKQ, BLKK)

        q = q.to(self.dtype)
        k = k.to(self.dtype)
        v = v.to(self.dtype)

        torch.xpu.synchronize()
        t1 = time.time()

        M_BLOCKS = triton.cdiv(L_padded, BLKQ)

        o_s = torch.empty((B, H, L_padded, D), device=v.device, dtype=v.dtype)

        grid = (M_BLOCKS, B * H)
        #print(f"grid {grid}, q.shape {q.shape}")

        _attn_fwd[grid](
            q, k, v,
            qk_scale = D ** -0.5,
            topk     = real_topk,
            LUT      = lut,
            OS       = o_s,
            L        = L_padded,
            M_BLOCKS = M_BLOCKS,
            D        = D,
            BLOCK_M  = BLKQ,
            BLOCK_N  = BLKK,
            num_warps   = num_warps,
            num_stages  = num_stages,
        )

        torch.xpu.synchronize()
        t2 = time.time()

        # Slice back for linear part
        q_fm = self.feature_map_q(q[:, :, :seq_len, :]).contiguous().to(self.dtype)
        k_fm = self.feature_map_k(k[:, :, :seq_len, :]).contiguous().to(self.dtype)

        kvsum = k_fm.transpose(-1, -2) @ v[:, :, :seq_len, :]
        ksum = torch.sum(k_fm, dim=-2, keepdim=True)
        o_l = (q_fm @ kvsum) / (1e-6 + (q_fm * ksum).sum(dim=-1, keepdim=True))
        o_l = self.proj_l(o_l)

        o_s = o_s[:, :, :seq_len, :]
        o = (o_s + o_l).to(orig_dtype).transpose(1, 2)

        torch.xpu.synchronize()
        t3 = time.time()

        print(f"  [BLK {BLKQ}/{BLKK}  warps {num_warps}  stages {num_stages}] "
              f"total {t3-t0:.4f}s | prep {t1-t0:.4f}s | triton {t2-t1:.4f}s | linear {t3-t2:.4f}s")

        if return_sparsity:
            return o, real_topk / sparse_map.shape[-1]
        return o


# ────────────────────────────────────────────────────────────────
#   Manual autotuning over configurations
# ────────────────────────────────────────────────────────────────

def autotune_best_config(q, k, v, topk_ratio=0.1, reps=5, warmup_reps=3):
   
    candidates = [
        #( 64,  64,  4, 3),
        #( 64,  64,  8, 4),
        #( 64,  64,  8, 5),
        #( 64,  64,  16, 3),
        #( 64,  64,  16, 5),
        #( 64,  64,  16, 6),
        
        (128,  64,  8, 3),
        (128,  64, 16, 4),
        (128,  64, 16, 5),
        (128,  64, 16, 6),
        (128,  64, 32, 4),
        (128,  64, 32, 5),
        (128,  64, 32, 6),
        
        #(128, 128,  8, 3),
        #(128, 128, 16, 4),
        #(128, 128, 32, 4),
        #(128, 128, 32, 5),
        
        #(256, 64,  8, 3),
        (256, 64, 16, 4),
        (256, 64, 32, 4),
        (256, 64, 32, 5), 
        #(256, 64, 64, 5), # Total number of work-items in a work-group cannot exceed 512 for this kernel
        (256, 64, 32, 6),
        (256, 64, 32, 8),
        
        #(256, 128,  8, 3),
        #(256, 128, 16, 4),
        #(256, 128, 32, 5),       
        
        (512,  64, 16, 4),
        (512,  64, 16, 5),
        (512,  64, 16, 6),
        (512,  64, 32, 5),
        (512,  64, 32, 6),
        #(512,  64, 64, 6), # Total number of work-items in a work-group cannot exceed 512 for this kernel
        
        #(512, 128, 32, 5),
    ]

    best_time = float('inf')
    best_config = None
    timings = {}

    attn = SparseLinearAttention(
        head_dim=q.shape[-1],
        topk=topk_ratio,
        feature_map='relu',
    ).xpu()

    print("Starting manual autotuning...\n")

    for cfg in candidates:
        BLKQ, BLKK, warps, stages = cfg
        times = []

        # Warmup
        for _ in range(warmup_reps):
            _ = attn(q, k, v, BLKQ=BLKQ, BLKK=BLKK, num_warps=warps, num_stages=stages)

        # Measure
        for _ in range(reps):
            torch.xpu.synchronize()
            t_start = time.time()
            _ = attn(q, k, v, BLKQ=BLKQ, BLKK=BLKK, num_warps=warps, num_stages=stages)
            torch.xpu.synchronize()
            times.append(time.time() - t_start)

        median_time = statistics.median(times)
        timings[cfg] = median_time

        print(f"  Config {cfg} → median {median_time:.4f} s")

        if median_time < best_time:
            best_time = median_time
            best_config = cfg

    print(f"\nBest config: {best_config}  ({best_time:.4f} s median)")
    print("All timings:", {str(k): f"{v:.4f}" for k,v in timings.items()})
    return best_config, attn


# ────────────────────────────────────────────────────────────────
#   Usage / benchmark
# ────────────────────────────────────────────────────────────────

B, H, L, D = 1, 32760, 12, 128

q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device='xpu')
k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device='xpu')
v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device='xpu')

# Run autotuning once
best_cfg, model = autotune_best_config(q, k, v, topk_ratio=0.1, reps=5, warmup_reps=3)

BLKQ, BLKK, num_warps, num_stages = best_cfg

# Now run final benchmark with the winner
print("\nFinal benchmark with best config:\n")

lat = []
torch.xpu.synchronize()

for i in range(7):
    t0 = time.time()
    o = model(q, k, v, BLKQ=BLKQ, BLKK=BLKK, num_warps=num_warps, num_stages=num_stages)
    torch.xpu.synchronize()
    dt = time.time() - t0
    lat.append(dt)
    print(f"run {i+1:2d}   dt = {dt:.4f} s    shape = {o.shape}")

print("\nlatencies:", [f"{x:.4f}" for x in lat])
print(f"median (after warmup): {statistics.median(lat[2:]):.4f} s")
