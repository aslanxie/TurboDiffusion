# Test Code

```
# Fast*Norm benchmark
PYTHONPATH=../turbodiffusion/inference/ python bench_fast_norm.py 

# Extract SLA code and reproduce Triton kernel _attn_fwd running time in inference. 
python sla_baseline_test.py 

# Optimized _attn_fwd code 
python sla_tune_1.py 

# Extend tune parameters range
python sla_tune_2.py
```