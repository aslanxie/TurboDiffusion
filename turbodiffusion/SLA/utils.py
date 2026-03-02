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
import triton
import triton.language as tl


def _device_sync(device: torch.device):
    if device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize()


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


def get_block_map(q, k, topk_ratio, BLKQ=64, BLKK=64):
    arg_k = k - torch.mean(k, dim=-2, keepdim=True) # smooth-k technique in SageAttention
    pooled_qblocks = mean_pool(q, BLKQ)
    pooled_kblocks = mean_pool(arg_k, BLKK)
    pooled_score = pooled_qblocks @ pooled_kblocks.transpose(-1, -2)

    K = pooled_score.shape[-1]
    topk = min(K, int(topk_ratio * K))
    lut = torch.topk(pooled_score, topk, dim=-1, sorted=False).indices

    sparse_map = torch.zeros_like(pooled_score, dtype=torch.int8)
    sparse_map.scatter_(-1, lut, 1)
    return sparse_map, lut, topk


def get_cuda_arch(device_index):
    major, minor = torch.cuda.get_device_capability(device_index)
    return f"sm{major}{minor}"
@triton.jit
def triton_block_map_to_lut_kernel(map_ptr, lut_ptr, valid_block_num_ptr, num_block_k: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    q = tl.program_id(2)

    map_ptr = map_ptr + (b * tl.num_programs(1) * tl.num_programs(2) + h * tl.num_programs(2) + q) * num_block_k
    lut_ptr = lut_ptr + (b * tl.num_programs(1) * tl.num_programs(2) + h * tl.num_programs(2) + q) * num_block_k
    valid_ptr = valid_block_num_ptr + b * tl.num_programs(1) * tl.num_programs(2) + h * tl.num_programs(2) + q

    valid_block_num = 0
    prev_block = 0
    for i in range(num_block_k):
        cur_block = tl.load(map_ptr + i)
        if cur_block:
            tl.store(lut_ptr + valid_block_num, i - prev_block)
            prev_block = i
            valid_block_num += 1

    tl.store(valid_ptr, valid_block_num)


def block_map_lut_triton(block_map):
    assert block_map.dim() == 4
    assert block_map.is_contiguous()
    B, H, Q, K = block_map.shape
    lut = torch.zeros((B, H, Q, K), dtype=torch.int32, device=block_map.device)
    valid = torch.zeros((B, H, Q), dtype=torch.int32, device=block_map.device)
    grid = (B, H, Q)
    triton_block_map_to_lut_kernel[grid](block_map, lut, valid, K)
    return lut, valid


@triton.jit
def qk_quantize(
    x_ptr,
    xm_ptr,
    x_quant_ptr,
    scale_ptr,
    N: tl.constexpr,
    D: tl.constexpr,
    BS: tl.constexpr,
    sm_scale: tl.constexpr,
    fuse_mean: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    nb = tl.program_id(2)

    H = tl.num_programs(1)
    block_offset = b * H * N * D + h * N * D + nb * BS * D
    offs_n = tl.arange(0, BS)
    offs_d = tl.arange(0, D)

    x_ptrs = x_ptr + block_offset + offs_n[:, None] * D + offs_d[None, :]
    xmask = (nb * BS + offs_n[:, None]) < N
    x = tl.load(x_ptrs, mask=xmask, other=0.0)

    if fuse_mean:
        xm_ptrs = xm_ptr + b * H * D + h * D + offs_d
        x_mean = tl.load(xm_ptrs)
        x = x - x_mean[None, :]
        x = tl.where(xmask, x, 0.0)

    x_fp32 = x.to(tl.float32) * sm_scale
    scale = tl.max(tl.abs(x_fp32)) / 127.0
    scale += 1e-7

    x_int8 = x_fp32 / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)

    x_quant_ptrs = x_quant_ptr + block_offset + offs_n[:, None] * D + offs_d[None, :]
    tl.store(x_quant_ptrs, x_int8, mask=xmask)

    NB = (N + BS - 1) // BS
    scale_ptrs = scale_ptr + b * H * NB + h * NB + nb
    tl.store(scale_ptrs, scale)


def get_quant(x, x_mean, block_size, sm_scale=1.0):
    x = x.contiguous()
    B, H, N, D = x.shape
    nblock = (N + block_size - 1) // block_size
    x_quant = torch.empty_like(x, dtype=torch.int8)
    x_scale = torch.empty((B, H, nblock), device=x.device, dtype=torch.float32)
    grid = (B, H, nblock)
    qk_quantize[grid](
        x,
        x_mean,
        x_quant,
        x_scale,
        N=N,
        D=D,
        BS=block_size,
        sm_scale=sm_scale,
        fuse_mean=(x_mean is not None),
    )
    return x_quant, x_scale


def get_vanilla_qk_quant(q, k, km=None, BLKQ=128, BLKK=64):
    head_dim = q.shape[-1]
    q_int8, q_scale = get_quant(q, None, BLKQ, sm_scale=(head_dim ** -0.5) * 1.44269504)
    k_int8, k_scale = get_quant(k, km, BLKK, sm_scale=1.0)
    return q_int8, q_scale, k_int8, k_scale