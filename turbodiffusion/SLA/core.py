""" 
Copyright (c) 2025 by SLA team.

Licensed under the Apache License, Version 2.0 (the "License");

Citation (please cite if you use this code):

@article{zhang2025sla,
  title={SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention}, 
  author={Jintao Zhang and Haoxu Wang and Kai Jiang and Shuo Yang and Kaiwen Zheng and Haocheng Xi and Ziteng Wang and Hongzhou Zhu and Min Zhao and Ion Stoica and Joseph E. Gonzalez and Jun Zhu and Jianfei Chen},
  journal={arXiv preprint arXiv:2509.24006},
  year={2025}
}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import triton
import triton.language as tl

SAGESLA_ENABLED = True
try:
    import spas_sage_attn._qattn as qattn
    import spas_sage_attn._fused as fused
    from spas_sage_attn.utils import get_vanilla_qk_quant, block_map_lut_triton
except ImportError:
    SAGESLA_ENABLED = False

SAGE2PP_ENABLED = True
try:
    from spas_sage_attn._qattn import qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold
except ImportError:
    SAGE2PP_ENABLED = False

from .kernel import _attention, _attn_fwd, _attn_fwd_sage_tma
from .utils import get_block_map, get_cuda_arch, _device_sync, get_vanilla_qk_quant


class SparseLinearAttention(nn.Module):
    def __init__(self, head_dim, topk, feature_map='softmax', BLKQ=64, BLKK=64, use_bf16=True, tie_feature_map_qk=True):
        R'''
        Args:
            head_dim: dimension of each head.
            topk: ratio of keys selected for sparse attention, shared across all queries.
            feature_map: feature map for linear attention, one of ['hedgehog', 'elu', 'relu', 'softmax'].
            BLKQ: block size for query.
            BLKK: block size for key.
            use_bf16: whether to use bfloat16 (default) or float16 for computation. The conversion to bf16/fp16 is done inside the module.
            tie_feature_map_qk: whether to use the same feature map for query and key.
        '''
        super().__init__()
        self.dtype = torch.bfloat16 if use_bf16 else torch.float16
        self.topk = topk
        self.BLKQ = BLKQ
        self.BLKK = BLKK
        self.proj_l = nn.Linear(head_dim, head_dim, dtype=torch.bfloat16)

        if feature_map == 'elu':
            def elu_feature_map(x):
                return F.elu(x) + 1
            self.feature_map_q = elu_feature_map
            self.feature_map_k = elu_feature_map
        elif feature_map == 'relu':
            self.feature_map_q = nn.ReLU()
            self.feature_map_k = nn.ReLU()
        elif feature_map == 'softmax':
            def softmax_feature_map(x):
                return F.softmax(x, dim=-1)
            self.feature_map_q = softmax_feature_map
            self.feature_map_k = softmax_feature_map
        else:
            raise NotImplementedError(f'Not supported feature map {feature_map}.')

        if tie_feature_map_qk:
            self.feature_map_k = self.feature_map_q

        self.init_weights_()

    def init_weights_(self):
        with torch.no_grad():
            nn.init.zeros_(self.proj_l.weight)
            nn.init.zeros_(self.proj_l.bias)

    def forward(self, q, k, v, return_sparsity=False):
        R'''
        Args:
            q: queries of shape (B, H, L, D).
            k: keys of shape (B, H, L, D).
            v: values of shape (B, H, L, D).
            return_sparsity: whether to return the actual sparsity.
        '''
        dtype = q.dtype
        
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        
        B, H, seq_len, D = q.shape

        # Pad to multiple of BLKQ and BLKK
        seq_len_padded_q = triton.cdiv(seq_len, self.BLKQ) * self.BLKQ
        seq_len_padded_kv = triton.cdiv(seq_len, self.BLKK) * self.BLKK
        pad_q = seq_len_padded_q - seq_len
        pad_kv = seq_len_padded_kv - seq_len

        if pad_q > 0:
            q = F.pad(q, (0, 0, 0, pad_q), value=0.0)
        if pad_kv > 0:
            k = F.pad(k, (0, 0, 0, pad_kv), value=0.0)
            v = F.pad(v, (0, 0, 0, pad_kv), value=0.0)

        L_padded = max(seq_len_padded_q, seq_len_padded_kv)
        
        sparse_map, lut, real_topk = get_block_map(q, k, topk_ratio=self.topk, BLKQ=self.BLKQ, BLKK=self.BLKK)

        q = q.to(self.dtype)
        k = k.to(self.dtype)
        v = v.to(self.dtype)
        M_BLOCKS = triton.cdiv(L_padded, self.BLKQ)

        o_s = torch.empty((B, H, L_padded, D), device=v.device, dtype=v.dtype)

        grid = (M_BLOCKS, B * H)
        _attn_fwd[grid](
            q, k, v,
            qk_scale = D ** -0.5,
            topk     = real_topk,
            LUT      = lut,
            OS       = o_s,
            L        = L_padded,
            M_BLOCKS = M_BLOCKS,
            D        = D,
            BLOCK_M  = self.BLKQ,
            BLOCK_N  = self.BLKK,
            num_warps   = 32,
            num_stages  = 5,
        )
        
        # Slice back for linear part
        q_fm = self.feature_map_q(q[:, :, :seq_len, :]).contiguous().to(self.dtype)
        k_fm = self.feature_map_k(k[:, :, :seq_len, :]).contiguous().to(self.dtype)

        kvsum = k_fm.transpose(-1, -2) @ v[:, :, :seq_len, :]
        ksum = torch.sum(k_fm, dim=-2, keepdim=True)
        o_l = (q_fm @ kvsum) / (1e-6 + (q_fm * ksum).sum(dim=-1, keepdim=True))
        o_l = self.proj_l(o_l)

        o_s = o_s[:, :, :seq_len, :]
        o = (o_s + o_l).to(dtype).transpose(1, 2)

        if return_sparsity:
            return o, real_topk / sparse_map.shape[-1]
        else:
            return o


@triton.jit
def transpose_pad_kernel(
    x_in, x_out,
    B: tl.constexpr, L: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    L_PAD: tl.constexpr, BLOCK_L: tl.constexpr,
):
    idx_l = tl.program_id(0)
    idx_bh = tl.program_id(1)
    b = idx_bh // H
    h = idx_bh % H

    offs_l = idx_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, D)
    mask = offs_l < L

    in_offset = b * (L * H * D) + offs_l[:, None] * (H * D) + h * D + offs_d[None, :]
    x = tl.load(x_in + in_offset, mask=mask[:, None], other=0.0)

    out_offset = b * (H * L_PAD * D) + h * (L_PAD * D) + offs_l[:, None] * D + offs_d[None, :]
    out_mask = offs_l < L_PAD
    tl.store(x_out + out_offset, x, mask=out_mask[:, None])

def fused_transpose_pad(x, l_pad, block_l=64):
    B, L, H, D = x.shape
    out = torch.empty((B, H, l_pad, D), device=x.device, dtype=x.dtype)
    n_blocks_l = triton.cdiv(l_pad, block_l)
    grid = (n_blocks_l, B * H)
    transpose_pad_kernel[grid](x, out, B, L, H, D, l_pad, block_l)
    return out


@triton.jit
def transpose_pad_kernel(
    x_in, x_out,
    B: tl.constexpr, L: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    L_PAD: tl.constexpr, BLOCK_L: tl.constexpr,
):
    idx_l = tl.program_id(0)
    idx_bh = tl.program_id(1)
    b = idx_bh // H
    h = idx_bh % H

    offs_l = idx_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, D)
    mask = offs_l < L

    in_offset = b * (L * H * D) + offs_l[:, None] * (H * D) + h * D + offs_d[None, :]
    x = tl.load(x_in + in_offset, mask=mask[:, None], other=0.0)

    out_offset = b * (H * L_PAD * D) + h * (L_PAD * D) + offs_l[:, None] * D + offs_d[None, :]
    out_mask = offs_l < L_PAD
    tl.store(x_out + out_offset, x, mask=out_mask[:, None])


def fused_transpose_pad(x, l_pad, block_l=64):
    B, L, H, D = x.shape
    out = torch.empty((B, H, l_pad, D), device=x.device, dtype=x.dtype)
    n_blocks_l = triton.cdiv(l_pad, block_l)
    grid = (n_blocks_l, B * H)
    transpose_pad_kernel[grid](x, out, B, L, H, D, l_pad, block_l)
    return out


class SageSparseLinearAttention(nn.Module):
    def __init__(self, head_dim, topk=0.1, feature_map='softmax', use_bf16=True, tie_feature_map_qk=True):
        super().__init__()
        self.dtype = torch.bfloat16 if use_bf16 else torch.float16
        self.topk = topk
        self.proj_l = nn.Linear(head_dim, head_dim, dtype=torch.bfloat16)

        if feature_map == 'elu':
            def elu_feature_map(x):
                return F.elu(x) + 1
            self.feature_map_q = self.feature_map_k = elu_feature_map
        elif feature_map == 'relu':
            self.feature_map_q = self.feature_map_k = nn.ReLU()
        elif feature_map == 'softmax':
            def softmax_feature_map(x):
                return F.softmax(x, dim=-1)
            self.feature_map_q = self.feature_map_k = softmax_feature_map
        else:
            raise ValueError(f"Unknown feature_map: {feature_map}")

        if tie_feature_map_qk:
            self.feature_map_k = self.feature_map_q

        nn.init.normal_(self.proj_l.weight)
        nn.init.normal_(self.proj_l.bias)

    def forward(self, q, k, v, BLKQ=64, BLKK=32, num_warps=8, num_stages=4, return_sparsity=False):
        _device_sync(q.device)

        B, seq_len, H, D = q.shape

        # Calculate lcm to perfectly align view pooling
        lcm_blk = (BLKQ * BLKK) // math.gcd(BLKQ, BLKK)
        L_padded = triton.cdiv(seq_len, lcm_blk) * lcm_blk

        # Only transpose+pad Q and K; V is read directly via strided TMA descriptor (zero-copy).
        q_tp = fused_transpose_pad(q, L_padded)
        k_tp = fused_transpose_pad(k, L_padded)

        # Block map: subtract global mean BEFORE view-based pooling to avoid diff regression.
        km = k_tp.mean(dim=-2, keepdim=True)
        pooled_q = q_tp.view(B, H, L_padded // BLKQ, BLKQ, D).mean(dim=3)
        pooled_k = (k_tp - km).view(B, H, L_padded // BLKK, BLKK, D).mean(dim=3)
        pooled_score = pooled_q @ pooled_k.transpose(-1, -2)

        K_blocks = pooled_score.shape[-1]
        real_topk = min(K_blocks, int(self.topk * K_blocks))
        topk_idx = torch.topk(pooled_score, real_topk, dim=-1, sorted=False).indices
        lut, _ = torch.sort(topk_idx, dim=-1)
        lut = lut.to(torch.int32).contiguous()

        _device_sync(q.device)

        # ---- Triton sparse attention --------------------------------- #
        M_BLOCKS = triton.cdiv(L_padded, BLKQ)
        N_BLOCKS = triton.cdiv(L_padded, BLKK)
        o_s = torch.empty((B, H, L_padded, D), device=v.device, dtype=torch.bfloat16)

        q_int8, q_scale, k_int8, k_scale = get_vanilla_qk_quant(q_tp, k_tp, km, BLKQ=BLKQ, BLKK=BLKK)

        grid = (M_BLOCKS, B * H)
        _attn_fwd_sage_tma[grid](
            q_int8, q_scale, k_int8, k_scale,
            v,                       # original [B, L, H, D] — zero-copy via strided TMA
            seq_len,                  # L_orig for V descriptor
            H,                       # num heads for V descriptor
            lut,
            o_s,
            L=L_padded,
            M_BLOCKS=M_BLOCKS,
            N_BLOCKS=N_BLOCKS,
            D=D,
            TOPK=real_topk,
            BLOCK_M=BLKQ,
            BLOCK_N=BLKK,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        _device_sync(q.device)

        # ---- Linear attention ---------------------------------------- #
        q_slice = q_tp[:, :, :seq_len, :]
        k_slice = k_tp[:, :, :seq_len, :]
        v_slice = v.transpose(1, 2)                       # lazy view, no copy

        q_fm = self.feature_map_q(q_slice)
        k_fm = self.feature_map_k(k_slice)

        kvsum = k_fm.transpose(-1, -2) @ v_slice
        ksum = torch.sum(k_fm, dim=-2, keepdim=True)

        denom = torch.matmul(q_fm, ksum.transpose(-1, -2)) + 1e-6
        o_l = torch.matmul(q_fm, kvsum)
        o_l.div_(denom)
        o_l = self.proj_l(o_l)

        # In-place merge: avoids allocating a temporary for the sum.
        o_l += o_s[:, :, :seq_len, :]
        o = o_l.transpose(1, 2)

        _device_sync(q.device)

        if return_sparsity:
            return o, real_topk / K_blocks
        return o
