from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _int8_quant_kernel_tma(
    input_ptr,
    output_ptr,
    scale_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    """TMA-accelerated int8 block-wise quantization.
    Uses tensor descriptors for HW-accelerated 2D block loads/stores.
    1D grid: each program handles one (BLOCK_SIZE x BLOCK_SIZE) tile.
    """
    blk_id = tl.program_id(0)
    blk_m = blk_id // tl.cdiv(N, BLOCK_SIZE)
    blk_n = blk_id % tl.cdiv(N, BLOCK_SIZE)

    coord_m = (blk_m * BLOCK_SIZE).to(tl.int32)
    coord_n = (blk_n * BLOCK_SIZE).to(tl.int32)

    desc = tl.make_tensor_descriptor(
        base=input_ptr, shape=(M, N), strides=(N, 1),
        block_shape=(BLOCK_SIZE, BLOCK_SIZE),
    )
    x = desc.load([coord_m, coord_n]).to(tl.float32)

    abs_x = tl.abs(x)
    amax = tl.max(abs_x)
    amax = tl.maximum(amax, 1e-8)

    scale = amax / 127.0
    scale_inv = 127.0 / amax

    x_scaled = x * scale_inv
    x_rounded = tl.where(x_scaled >= 0, x_scaled + 0.5, x_scaled - 0.5)
    x_quant = x_rounded.to(tl.int8)

    out_desc = tl.make_tensor_descriptor(
        base=output_ptr, shape=(M, N), strides=(N, 1),
        block_shape=(BLOCK_SIZE, BLOCK_SIZE),
    )
    out_desc.store([coord_m, coord_n], x_quant)

    n_blocks = tl.cdiv(N, BLOCK_SIZE)
    scale_idx = blk_m * n_blocks + blk_n
    tl.store(scale_ptr + scale_idx, scale)


@triton.jit
def _int8_gemm_kernel(
    # Pointers
    a_ptr, a_scale_ptr,
    b_ptr, b_scale_ptr,
    c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    # Scale dimensions
    n_scale_blocks_k,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for int8 GEMM with per-block scaling.
    C = A @ B^T where A and B are quantized int8 with block-wise scales.
    
    A: [M, K] with scales [m_blocks, k_blocks]
    B: [N, K] with scales [n_blocks, k_blocks]
    C: [M, N]
    """
    # Block coordinates
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets for output tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Iterate over K dimension in blocks
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Load A block [BLOCK_M, BLOCK_K] - keep as int8
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=a_mask,
            other=0
        )
        
        # Load B block [BLOCK_N, BLOCK_K] - keep as int8
        b_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        b = tl.load(
            b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk,
            mask=b_mask,
            other=0
        )
        
        # Load scales for this K block
        k_block_idx = k_start // QUANT_BLOCK_SIZE
        
        # A scales: [m_blocks, k_blocks] -> need scales for rows in offs_m
        m_block_idx = offs_m // QUANT_BLOCK_SIZE
        a_scale_idx = m_block_idx * n_scale_blocks_k + k_block_idx
        a_scale = tl.load(a_scale_ptr + a_scale_idx, mask=offs_m < M, other=1.0)
        
        # B scales: [n_blocks, k_blocks] -> need scales for rows in offs_n  
        n_block_idx = offs_n // QUANT_BLOCK_SIZE
        b_scale_idx = n_block_idx * n_scale_blocks_k + k_block_idx
        b_scale = tl.load(b_scale_ptr + b_scale_idx, mask=offs_n < N, other=1.0)
        
        # Compute partial matmul with int8 inputs (accumulates in int32 automatically)
        # tl.dot expects int8 inputs, not int32
        partial = tl.dot(a, tl.trans(b), allow_tf32=False)
        
        # Apply scales: scale is per (M-block, K-block) x (N-block, K-block)
        scaled_partial = partial.to(tl.float32) * a_scale[:, None] * b_scale[None, :]
        acc += scaled_partial
    
    # Store result
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


@triton.jit
def _int8_gemm_kernel_tma(
    a_ptr, a_scale_ptr,
    b_ptr, b_scale_ptr,
    c_ptr,
    M, N, K,
    n_scale_blocks_k,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_K_ITERS: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    """TMA-optimized int8 GEMM v2: C = A @ B^T with scalar per-block scaling.

    Key optimizations over v1:
      - QUANT_BLOCK == BLOCK_K == BLOCK_M == BLOCK_N (all 64)
        → each tile maps to exactly ONE scale value (scalar, not vector)
        → eliminates per-element outer product a_scale[:, None] * b_scale[None, :]
      - tl.range(NUM_K_ITERS, num_stages=N) for SW pipelining
        → TMA prefetches next iteration while current one computes

    Best config (32760×1536×1536): GROUP=32 warps=4 stages=2 pipeline=2 → 3.8 ms
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    a_desc = tl.make_tensor_descriptor(
        base=a_ptr, shape=[M, K], strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    b_desc = tl.make_tensor_descriptor(
        base=b_ptr, shape=[N, K], strides=[K, 1],
        block_shape=[BLOCK_N, BLOCK_K],
    )
    c_desc = tl.make_tensor_descriptor(
        base=c_ptr, shape=[M, N], strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    coord_m = (pid_m * BLOCK_M).to(tl.int32)
    coord_n = (pid_n * BLOCK_N).to(tl.int32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Scale layout: [num_blocks_row, num_blocks_col] stored row-major.
    # With QUANT_BLOCK_SIZE == BLOCK_K, scale[pid_m, k_iter] and scale[pid_n, k_iter]
    # are simple scalar lookups — no division, no masking, no vector outer product.
    a_scale_base = a_scale_ptr + pid_m * n_scale_blocks_k
    b_scale_base = b_scale_ptr + pid_n * n_scale_blocks_k

    for k_iter in tl.range(NUM_K_ITERS, num_stages=PIPELINE_STAGES):
        coord_k = (k_iter * BLOCK_K).to(tl.int32)
        a = a_desc.load([coord_m, coord_k])
        b = b_desc.load([coord_n, coord_k])

        a_s = tl.load(a_scale_base + k_iter)
        b_s = tl.load(b_scale_base + k_iter)

        partial = tl.dot(a, tl.trans(b))
        acc += partial.to(tl.float32) * (a_s * b_s)

    c_desc.store([coord_m, coord_n], acc.to(c_ptr.dtype.element_ty))


def int8_quant(x: torch.Tensor, BLOCK_SIZE: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a floating-point tensor to int8 using TMA-accelerated Triton kernel.

    Args:
        x (torch.Tensor): Input tensor of type float16/bfloat16.
        BLOCK_SIZE (int): Block size for quantization (default: 64).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - x_q: Quantized int8 tensor.
            - x_scale: Per-block scale tensor used for quantization.
    """
    assert x.is_contiguous(), "Input must be contiguous"

    M, N = x.shape
    m_blocks = triton.cdiv(M, BLOCK_SIZE)
    n_blocks = triton.cdiv(N, BLOCK_SIZE)

    x_q = torch.empty_like(x, dtype=torch.int8)
    x_scale = torch.empty((m_blocks, n_blocks), dtype=torch.float32, device=x.device)

    grid = (m_blocks * n_blocks,)
    _int8_quant_kernel_tma[grid](
        x, x_q, x_scale, M, N, BLOCK_SIZE=BLOCK_SIZE,
    )

    return x_q, x_scale


def int8_linear(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """
    Perform an int8 GEMM (matrix multiplication) using quantized weights and a
    quantized version of the input. The underlying compute is performed by a
    Triton kernel.

    Args:
        x (torch.Tensor): Input activation of shape (M, K) in float16/bfloat16.
        w_q (torch.Tensor): Quantized int8 weight tensor of shape (N, K).
        w_s (torch.Tensor): Scale tensor associated with w_q.
        **kwargs: Additional options (reserved for future use).

    Returns:
        torch.Tensor: Output tensor of shape (M, N) in float16/bfloat16.
    """
    assert w_q.dtype == torch.int8, "Weight tensor must be int8."
    #print(x.shape, w_q.shape, w_s.shape, x.dtype, w_q.dtype, w_s.dtype)
    shape = x.shape
    x = x.reshape(-1, shape[-1])
    M = x.shape[0]
    N = w_q.shape[0]
    K = w_q.shape[1]
    
    # Quantize input
    x_q, x_s = int8_quant(x)
    
    # Allocate output
    y = torch.zeros(M, N, dtype=x.dtype, device=x.device)
    
    # Launch GEMM kernel
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 128  # Must match quantization block size
    QUANT_BLOCK_SIZE = 128
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    n_scale_blocks_k = triton.cdiv(K, QUANT_BLOCK_SIZE)
    
    _int8_gemm_kernel[grid](
        x_q, x_s.flatten(),
        w_q, w_s.flatten(),
        y,
        M, N, K,
        x_q.stride(0), x_q.stride(1),
        w_q.stride(0), w_q.stride(1),
        y.stride(0), y.stride(1),
        n_scale_blocks_k,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
    )
    
    return y.reshape(*shape[:-1], N)


def int8_linear_tma(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    BLOCK_SIZE: int = 64,
    GROUP_SIZE_M: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
    pipeline_stages: int = 2,
    **kwargs,
) -> torch.Tensor:
    """TMA-optimized int8 GEMM v2 with scalar scales + SW pipelining.

    Key optimizations:
      - QUANT_BLOCK == BLOCK_K == BLOCK_M == BLOCK_N (all 64)
        → scalar scale per tile, no per-element outer product
      - tl.range SW pipelining: TMA prefetches next tile while current computes
      - grf_mode=256: large GRF to avoid register spilling on Intel XPU

    Best config (32760×1536×1536): GROUP=32 warps=4 stages=2 pipeline=2 → 3.8 ms

    Args:
        x (torch.Tensor): Input activation (M, K) or (B, L, K) in float16/bfloat16.
        w_q (torch.Tensor): Quantized int8 weight tensor of shape (N, K).
        w_s (torch.Tensor): Scale tensor associated with w_q (quantized with BLOCK_SIZE=64).
        BLOCK_SIZE: Unified tile + quant block size (default: 64).
        GROUP_SIZE_M: Swizzle group size for L2 cache locality.
        num_warps: Number of warps per tile.
        num_stages: Number of Triton pipeline stages.
        pipeline_stages: Number of SW pipeline stages for K-loop.

    Returns:
        torch.Tensor: Output tensor of shape (M, N) or (B, L, N).
    """
    assert w_q.dtype == torch.int8, "Weight tensor must be int8."
    shape = x.shape
    x = x.reshape(-1, shape[-1]).contiguous()
    M = x.shape[0]
    N = w_q.shape[0]
    K = w_q.shape[1]
    BLOCK_M = BLOCK_N = BLOCK_K = BLOCK_SIZE

    x_q, x_s = int8_quant(x, BLOCK_SIZE=BLOCK_SIZE)

    y = torch.empty(M, N, dtype=x.dtype, device=x.device)
    n_scale_blocks_k = triton.cdiv(K, BLOCK_SIZE)
    NUM_K_ITERS = triton.cdiv(K, BLOCK_K)

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _int8_gemm_kernel_tma[grid](
        x_q, x_s.flatten(),
        w_q, w_s.flatten(),
        y,
        M, N, K,
        n_scale_blocks_k,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_K_ITERS=NUM_K_ITERS,
        PIPELINE_STAGES=pipeline_stages,
        num_warps=num_warps,
        num_stages=num_stages,
        #grf_mode=256,
    )
    return y.reshape(*shape[:-1], N)


def flatten_if_batched(*tensors):
    """
    Flattens all input tensors from (B, N, D_i) to (B * N, D_i) if they are batched (3D).

    Args:
        *tensors: Any number of input tensors, each must have shape (B, N, D_i) or (N, D_i)

    Returns:
        flat_tensors: List of flattened tensors
        batched: Boolean flag indicating whether inputs were batched
        batch_size: Batch size if batched, else None
    """
    if not tensors:
        raise ValueError("At least one tensor must be provided.")

    first = tensors[0]
    assert len(first.shape) in [
        2,
        3,
    ], "Input tensors must be batched (3D) or not batched (2D)"

    if len(first.shape) == 3:  # batched
        batched = True
        batch_size = first.shape[0]
        assert all(t.shape[0] == batch_size for t in tensors), "All input tensors must have the same batch size"
        assert all(
            t.shape[1] == first.shape[1] for t in tensors
        ), "All input tensors must have the same sequence length"
        flat_tensors = [t.reshape(-1, t.shape[-1]) for t in tensors]
    else:
        batched = False
        batch_size = None
        flat_tensors = list(tensors)

    return flat_tensors, batched, batch_size


@triton.jit
def _rms_norm_fwd_fused(
    X,
    Y,
    W,
    Rstd,
    x_stride,
    y_stride,
    N: tl.constexpr,  # number of columns in X,
    N2: tl.constexpr,
    eps,  # epsilon to avoid division by zero
    BLOCK_M: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, N2)
    mask = cols < N

    x_ptr = X + rows[:, None] * x_stride + cols[None, :]
    y_ptr = Y + rows[:, None] * y_stride + cols[None, :]

    x = tl.load(x_ptr, mask=mask[None, :], other=0.0).to(tl.float32)

    # Compute variance
    _var = x * x
    var = tl.sum(_var, axis=1) / N
    rstd = 1 / tl.sqrt(var + eps)

    # Write mean / rstd
    tl.store(Rstd + rows, rstd)
    rstd = tl.reshape(rstd, (BLOCK_M, 1))

    # Normalize and apply linear transformation
    w = tl.load(W + cols)
    x_hat = x * rstd
    y = x_hat * w

    # Write output
    y = y.to(Y.type.element_ty)
    tl.store(y_ptr, y, mask=mask[None, :])


def rmsnorm(x, w, eps):
    """
    Forward pass of the RMSNorm.

    Args:
        x (torch.Tensor): Input tensor, High precision.
        w (torch.Tensor): RMSNorm weight tensor.
        eps (float): RMSNorm epsilon value.

    Returns:
        y (torch.Tensor): Output tensor, High precision.
        (w, rstd, num_warps) (tuple): RMSNorm weight tensor, rstd tensor, and number of warps.
    """
    assert x.is_contiguous(), "Input must be contiguous"
    # Change batched 3D input to 2D
    [x], batched, BS = flatten_if_batched(x)

    # allocate output
    M, N = x.shape
    y = torch.empty_like(x, dtype=x.dtype)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

    # heuristics for number of warps
    num_warps = 8
    
    # Avoid illegal memory access
    N2 = triton.next_power_of_2(N)
    
    if N <= 512:
        BLOCK_M = 32
    else:
        BLOCK_M = 1

    # Call the triton kernel
    _rms_norm_fwd_fused[(triton.cdiv(M, BLOCK_M),)](  #
        x,
        y,
        w,
        rstd,  #
        x.stride(0),
        y.stride(0),
        N,
        N2,
        eps,
        num_warps=num_warps,
        BLOCK_M=BLOCK_M,
    )

    # Recover 2D to 3D
    if batched:
        y = y.reshape(BS, -1, y.shape[-1])

    return y

@triton.jit
def _layer_norm_param_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    x_stride,  # how much to increase the pointer when moving by 1 row
    y_stride,  # how much to increase the pointer when moving by 1 row
    N: tl.constexpr,  # number of columns in X,
    N2: tl.constexpr,  # number of columns in X,
    eps,  # epsilon to avoid division by zero
    BLOCK_M: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, N2)
    mask = cols < N

    x_ptr = X + rows[:, None] * x_stride + cols[None, :]
    y_ptr = Y + rows[:, None] * y_stride + cols[None, :]

    x = tl.load(x_ptr, mask=mask[None, :], other=0.0).to(tl.float32)

    # Compute mean and Variance
    mean = tl.sum(x, axis=1, keep_dims=True) / N
    # Compute variance
    _var = (x - mean) * (x - mean)
    var = tl.sum(_var, axis=1, keep_dims=True) / N
    rstd = 1 / tl.sqrt(var + eps)

    # Write mean / rstd
    _mean = tl.reshape(mean, (BLOCK_M))
    _rstd = tl.reshape(rstd, (BLOCK_M))
    tl.store(Mean + rows, _mean)
    tl.store(Rstd + rows, _rstd)

    # Normalize and apply linear transformation
    x_hat = (x - mean) * rstd

    w = tl.load(W + cols)
    b = tl.load(B + cols)
    
    x_hat = x_hat * w + b

    # Write output
    x_hat = x_hat.to(Y.type.element_ty)
    tl.store(y_ptr, x_hat, mask=mask[None, :])


def layernorm_param(x, w, b, eps):
    # Change batched 3D input to 2D
    [x], batched, BS = flatten_if_batched(x)

    # allocate output
    M, N = x.shape
    y = torch.empty_like(x)
    mean = torch.empty((M,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    # heuristics for number of warps
    num_warps = 8

    N2 = triton.next_power_of_2(N)
    
    if N <= 512:
        BLOCK_M = 32
    else:
        BLOCK_M = 1

    # enqueue kernel
    _layer_norm_param_fwd_fused[(triton.cdiv(M, BLOCK_M),)](  #
        x,
        y,
        w,
        b,
        mean,
        rstd,  #
        x.stride(0),
        y.stride(0),
        N,
        N2,
        eps,
        num_warps=num_warps,
        BLOCK_M=BLOCK_M,
    )

    # Recover 2D to 3D
    if batched:
        y = y.reshape(BS, -1, y.shape[-1])

    return y


########################################################
# Elementwise_affine=False
########################################################


@triton.jit
def _layer_norm_noparam_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    x_stride,  # how much to increase the pointer when moving by 1 row
    y_stride,  # how much to increase the pointer when moving by 1 row
    N: tl.constexpr,  # number of columns in X,
    N2: tl.constexpr,  # number of columns in X,
    eps,  # epsilon to avoid division by zero
    BLOCK_M: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, N2)
    mask = cols < N

    x_ptr = X + rows[:, None] * x_stride + cols[None, :]
    y_ptr = Y + rows[:, None] * y_stride + cols[None, :]

    x = tl.load(x_ptr, mask=mask[None, :], other=0.0).to(tl.float32)

    # Compute mean and Variance
    mean = tl.sum(x, axis=1, keep_dims=True) / N
    # Compute variance
    _var = (x - mean) * (x - mean)
    var = tl.sum(_var, axis=1, keep_dims=True) / N
    rstd = 1 / tl.sqrt(var + eps)

    # Write mean / rstd
    _mean = tl.reshape(mean, (BLOCK_M))
    _rstd = tl.reshape(rstd, (BLOCK_M))
    tl.store(Mean + rows, _mean)
    tl.store(Rstd + rows, _rstd)

    # Normalize and apply linear transformation
    x_hat = (x - mean) * rstd

    # Write output
    x_hat = x_hat.to(Y.type.element_ty)
    tl.store(y_ptr, x_hat, mask=mask[None, :])


def layernorm_noparam(x, eps):
    assert x.is_contiguous(), "Input must be contiguous"

    # Change batched 3D input to 2D
    [x], batched, BS = flatten_if_batched(x)

    # allocate output
    M, N = x.shape
    y = torch.empty_like(x)
    mean = torch.empty((M,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    # heuristics for number of warps
    num_warps = 8

    N2 = triton.next_power_of_2(N)
    
    if N <= 512:
        BLOCK_M = 32
    else:
        BLOCK_M = 1

    # enqueue kernel
    _layer_norm_noparam_fwd_fused[(triton.cdiv(M, BLOCK_M),)](  #
        x,
        y,
        mean,
        rstd,  #
        x.stride(0),
        y.stride(0),
        N,
        N2,
        eps,
        num_warps=num_warps,
        BLOCK_M=BLOCK_M,
    )

    # Recover 2D to 3D
    if batched:
        y = y.reshape(BS, -1, y.shape[-1])

    return y

def layernorm(x, w, b, eps, elementwise_affine=True):
    if elementwise_affine:
        assert w is not None and b is not None
        return layernorm_param(x, w, b, eps)
    else:
        assert w is None and b is None
        return layernorm_noparam(x, eps)


########################################################
# Fused Modulated LayerNorm: layernorm(x) * (1 + scale) + shift
########################################################

@triton.jit
def _modulated_layernorm_fwd(
    X, SCALE, SHIFT, Y,
    x_stride, y_stride, scale_stride,
    M,   # total rows (B * L)
    L,   # sequence length per batch
    N: tl.constexpr,
    N2: tl.constexpr,
    eps,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    cols = tl.arange(0, N2)
    col_mask = cols < N

    batch_idx = rows // L

    x_ptr = X + rows[:, None] * x_stride + cols[None, :]
    y_ptr = Y + rows[:, None] * y_stride + cols[None, :]

    x = tl.load(x_ptr, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)

    # LayerNorm (no affine params)
    mean = tl.sum(x, axis=1, keep_dims=True) / N
    var = tl.sum((x - mean) * (x - mean), axis=1, keep_dims=True) / N
    rstd = 1 / tl.sqrt(var + eps)
    x_hat = (x - mean) * rstd

    # Load modulation: scale=[B, C], shift=[B, C]
    s_ptr = batch_idx[:, None] * scale_stride + cols[None, :]
    scale = tl.load(SCALE + s_ptr, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)
    shift = tl.load(SHIFT + s_ptr, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)

    y = x_hat * (1.0 + scale) + shift
    y = y.to(Y.type.element_ty)
    tl.store(y_ptr, y, mask=row_mask[:, None] & col_mask[None, :])


def modulated_layernorm(x, scale, shift, eps):
    """Fused LayerNorm + modulation: layernorm(x) * (1 + scale) + shift

    Combines LayerNorm, scale, and shift into a single kernel launch, eliminating
    3 separate elementwise kernel launches and intermediate memory allocations.

    Args:
        x: [B, L, C] input tensor
        scale: [B, 1, C] scale modulation
        shift: [B, 1, C] shift modulation
        eps: layernorm epsilon

    Returns:
        [B, L, C] output tensor
    """
    B, L, C = x.shape
    x_flat = x.reshape(-1, C).contiguous()
    M = x_flat.shape[0]

    scale_sq = scale.squeeze(1).contiguous()  # [B, C]
    shift_sq = shift.squeeze(1).contiguous()  # [B, C]

    y_flat = torch.empty_like(x_flat)

    N2 = triton.next_power_of_2(C)
    BLOCK_M = 32 if C <= 512 else 1

    _modulated_layernorm_fwd[(triton.cdiv(M, BLOCK_M),)](
        x_flat, scale_sq, shift_sq, y_flat,
        x_flat.stride(0), y_flat.stride(0), scale_sq.stride(0),
        M, L, C, N2, eps,
        BLOCK_M=BLOCK_M,
        num_warps=8,
    )

    return y_flat.reshape(B, L, C)


########################################################
# Fused Gated Residual Add: x + y * gate
########################################################

@triton.jit
def _gated_residual_add_fwd(
    X, Y, GATE, OUT,
    xy_stride, gate_stride,
    M, L,
    N: tl.constexpr,
    N2: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    cols = tl.arange(0, N2)
    col_mask = cols < N

    batch_idx = rows // L

    x_ptr = X + rows[:, None] * xy_stride + cols[None, :]
    y_ptr = Y + rows[:, None] * xy_stride + cols[None, :]
    out_ptr = OUT + rows[:, None] * xy_stride + cols[None, :]

    x = tl.load(x_ptr, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
    y = tl.load(y_ptr, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
    g = tl.load(GATE + batch_idx[:, None] * gate_stride + cols[None, :],
                mask=row_mask[:, None] & col_mask[None, :], other=0.0)

    result = x + y * g
    tl.store(out_ptr, result, mask=row_mask[:, None] & col_mask[None, :])


def gated_residual_add(x, y, gate):
    """Fused: output = x + y * gate

    Combines multiply and add into a single kernel launch.

    Args:
        x: [B, L, C] residual tensor
        y: [B, L, C] branch output tensor
        gate: [B, 1, C] gating factor

    Returns:
        [B, L, C] result tensor
    """
    B, L, C = x.shape

    x_flat = x.reshape(-1, C).contiguous()
    y_flat = y.reshape(-1, C).contiguous()
    gate_sq = gate.squeeze(1).contiguous()  # [B, C]
    out_flat = torch.empty_like(x_flat)
    M = x_flat.shape[0]

    N2 = triton.next_power_of_2(C)
    BLOCK_M = 32 if C <= 512 else 1

    _gated_residual_add_fwd[(triton.cdiv(M, BLOCK_M),)](
        x_flat, y_flat, gate_sq, out_flat,
        x_flat.stride(0), gate_sq.stride(0),
        M, L, C, N2,
        BLOCK_M=BLOCK_M,
        num_warps=8,
    )

    return out_flat.reshape(B, L, C)


def cdiv(a: int, b: int):
    return (a + b - 1) // b


def _dequantize_int8_blockwise(
    x_q: torch.Tensor, scales: torch.Tensor, block_size: int
) -> torch.Tensor:
    """Dequantize a block-wise int8 tensor back to bfloat16.

    Args:
        x_q: Quantized int8 tensor of shape (M, N).
        scales: Per-block scale tensor of shape (m_blocks, n_blocks).
        block_size: Block size used during quantization.

    Returns:
        Dequantized bfloat16 tensor of shape (M, N).
    """
    M, N = x_q.shape
    m_blocks = M // block_size
    n_blocks = N // block_size
    # (M, N) → (m_blocks, block_size, n_blocks, block_size)
    x_reshaped = x_q.reshape(m_blocks, block_size, n_blocks, block_size).float()
    # scales: (m_blocks, n_blocks) → (m_blocks, 1, n_blocks, 1)
    result = x_reshaped * scales[:, None, :, None]
    return result.reshape(M, N).to(torch.bfloat16)


class Int8Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        row_blocks = cdiv(out_features, b=64)
        col_blocks = cdiv(in_features, b=64)
        
        self.register_buffer("int8_weight", torch.empty((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scale", torch.empty((row_blocks, col_blocks), dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=dtype))
        else:
            self.bias = None
        

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        """Handle loading checkpoints quantized with a different block size.

        If the checkpoint scale shape doesn't match (e.g. old BLOCK_SIZE=128
        vs current BLOCK_SIZE=64), dequantize with the old block size and
        re-quantize with the current one.
        """
        scale_key = prefix + "scale"
        weight_key = prefix + "int8_weight"

        if scale_key in state_dict and weight_key in state_dict:
            old_scale = state_dict[scale_key]
            if old_scale.shape != self.scale.shape:
                old_int8_weight = state_dict[weight_key]
                # Infer old block size from scale shape
                old_block_size = self.in_features // old_scale.shape[1]

                # Dequantize → re-quantize on GPU
                device = old_int8_weight.device
                float_weight = _dequantize_int8_blockwise(
                    old_int8_weight.to(device), old_scale.to(device), old_block_size
                )
                new_block_size = self.in_features // self.scale.shape[1]
                new_int8_weight, new_scale = int8_quant(
                    float_weight.xpu(), BLOCK_SIZE=new_block_size
                )
                state_dict[scale_key] = new_scale.to(device)
                state_dict[weight_key] = new_int8_weight.to(device)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def forward(self, x):
        out = int8_linear_tma(x, self.int8_weight, self.scale)
        if self.bias is not None:
            out = out + self.bias
        return out

    @classmethod
    def from_linear(cls, original_linear: nn.Linear, quantize: bool = True):
    
        int8_layer = cls(
            original_linear.in_features,
            original_linear.out_features,
            bias=original_linear.bias is not None,
            dtype=original_linear.weight.dtype
        )
        if quantize:
            w_data = original_linear.weight.data.cuda()
            int8_w, scale = int8_quant(w_data)

            int8_layer.int8_weight.copy_(int8_w)
            int8_layer.scale.copy_(scale)
            if original_linear.bias is not None:
                int8_layer.bias.data.copy_(original_linear.bias.data.cuda())
            
        return int8_layer
    
class FastRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.register_buffer("weight", torch.ones(dim))

    def forward(self, x):
        return rmsnorm(x, self.weight, self.eps)

    @classmethod
    def from_rmsnorm(cls, original_rmsnorm):
        rmsnorm_layer = cls(
            dim=original_rmsnorm.dim,
            eps=original_rmsnorm.eps
        )
        if original_rmsnorm.weight.device != torch.device('meta'):
            rmsnorm_layer.weight.data.copy_(original_rmsnorm.weight.float().data)
        return rmsnorm_layer
    
class FastLayerNorm(nn.Module):

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        elementwise_affine: bool = False,
        bias: bool = True
    ) :
        super().__init__()
        self.dim = dim  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.register_buffer("weight", torch.empty(self.dim))
            if bias:
                self.register_buffer("bias", torch.empty(self.dim))
            else:
                self.bias = None
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        return layernorm(x, self.weight, self.bias, self.eps, self.elementwise_affine)
    
    @classmethod
    def from_layernorm(cls, original_layernorm):
        layernorm_layer = cls(
            dim=original_layernorm.normalized_shape[0],
            eps=original_layernorm.eps,
            elementwise_affine=False if original_layernorm.weight is None else True,
            bias=original_layernorm.bias is not None
        )
        if original_layernorm.weight is not None and original_layernorm.weight.device != torch.device('meta'):
            layernorm_layer.weight.data.copy_(original_layernorm.weight.data)
        if original_layernorm.bias is not None and original_layernorm.bias.device != torch.device('meta'):
            layernorm_layer.bias.data.copy_(original_layernorm.bias.data)
        return layernorm_layer