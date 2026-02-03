import time
import torch
import torch.nn as nn
import numpy as np
from FastNorm import FastLayerNorm, FastRMSNorm

# ────────────────────────────────────────────────
# Your implementations
# ────────────────────────────────────────────────
class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


# ────────────────────────────────────────────────
# Benchmark settings
# ────────────────────────────────────────────────
DEVICE = "xpu" if torch.xpu.is_available() else "cpu"   # fallback to cpu if xpu not detected
DTYPE = torch.float
B, L, C = 1, 32760, 1536

NUM_WARMUP = 30
NUM_MEASURE = 100

print(f"Using device: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")
if DEVICE == "xpu":
    print(f"XPU device name: {torch.xpu.get_device_name(0)}")

torch.manual_seed(0)
x = torch.randn(B, L, C, device=DEVICE, dtype=DTYPE)

# Create modules
wan_rms = WanRMSNorm(C).to(device=DEVICE, dtype=DTYPE)
wan_ln  = WanLayerNorm(C, elementwise_affine=False).to(device=DEVICE, dtype=DTYPE)


fast_rms = FastRMSNorm.from_rmsnorm(wan_rms).to(device=DEVICE)
fast_ln  = FastLayerNorm.from_layernorm(wan_ln).to(device=DEVICE)

modules = {
    "wan_rms_norm": wan_rms,
    "wan_layer_norm": wan_ln,
    "fast_rms_norm": fast_rms,
    "fast_layer_norm": fast_ln,
}

# ────────────────────────────────────────────────
# Benchmark function
# ────────────────────────────────────────────────
def measure_latency(fn, num_warmup=NUM_WARMUP, num_runs=NUM_MEASURE):
    # Warm-up
    for _ in range(num_warmup):
        _ = fn()
    if DEVICE == "xpu":
        torch.xpu.synchronize()
    elif DEVICE.startswith("cuda"):
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(num_runs):
        t0 = time.perf_counter_ns()
        _ = fn()
        if DEVICE == "xpu":
            torch.xpu.synchronize()
        elif DEVICE.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        times_ms.append((t1 - t0) / 1_000_000)

    times = np.array(times_ms)
    return {
        "median_ms": np.median(times),
        "mean_ms": times.mean(),
        "std_ms": times.std(),
        "p90_ms": np.percentile(times, 90),
        "min_ms": times.min(),
        "max_ms": times.max()
    }


# ────────────────────────────────────────────────
# Run benchmark
# ────────────────────────────────────────────────
print("\n" + "="*70)
print(f"Benchmark | shape = {tuple(x.shape)} | dtype = {DTYPE} | device = {DEVICE}")
print(f"Warmup runs: {NUM_WARMUP} | Measured runs: {NUM_MEASURE}")
print("="*70)

results = {}

for name, mod in modules.items():
    def forward():
        return mod(x)

    stats = measure_latency(forward)
    results[name] = stats

    print(f"{name:18}  median {stats['median_ms']:6.3f} ms   "
          f"(mean {stats['mean_ms']:6.3f} ± {stats['std_ms']:5.3f})   "
          f"p90 {stats['p90_ms']:6.3f} ms")

print("="*70)