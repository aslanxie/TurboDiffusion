# Optimize TurboDiffusion Inference on Intel GPU


4 steps sampleing running time is reduced from 83.57s to 21.70s on B60.


## Setup

1. Prepare model weights
```
# Reference orignal link to prepare model weight: https://github.com/thu-ml/TurboDiffusion/
checkpoints/
├── models_t5_umt5-xxl-enc-bf16.pth
├── TurboWan2.1-T2V-1.3B-480P.pth
├── TurboWan2.1-T2V-1.3B-480P-quant.pth
└── Wan2.1_VAE.pth

# Download below files from link: https://huggingface.co/google/umt5-xxl
umt5-xxl/
├── config.json
├── generation_config.json
├── special_tokens_map.json
├── spiece.model
├── tokenizer_config.json
└── tokenizer.json
```

2. Running below command line. Notes: we needn't build extra kernel and skip setup step.
```
PYTHONPATH=./turbodiffusion/	 python turbodiffusion/inference/wan2.1_t2v_infer.py \
    --model Wan2.1-1.3B \
    --dit_path checkpoints/TurboWan2.1-T2V-1.3B-480P.pth \
    --resolution 480p \
    --prompt "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about." \
    --num_samples 1 \
    --num_steps 4 \
    --attention_type sla  \
    --sla_topk 0.1
```

## Optimize Methods

### Initialization
While enabling TurboDiffusion on Intel GPU, lots of walkaround is applied. For example:
1. Intel GPUs don't support autocast to float32, like ```amp.autocast("cuda", dtype=torch.float32)```.  So, we're forcing the conversion manually.
2. Use SLA to replace SageSLA which need extra effort to migrate CUDA kernel.
3. Use apply_rotary_emb based on torch without Intel GPU flash_attn package.

After enabled on B60, 4 steps sampling need take 83.57s. 

### Optimize

After replace original LayerNorm and RMSNorm of Wan models with FastLayerNorm and FastRMSNorm implemented by Triton kernel, no obvious gain. So, let's enable  PyTorch profiler to identify performance bottlenecks.

#### Profile and Optimize

#### Round 1 Profile
```
#
# sort_by="self_xpu_time_total"
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg      Self XPU    Self XPU %     XPU total  XPU time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                             _attention         0.03%      21.533ms         0.10%      65.844ms     548.703us       49.789s        56.87%       49.789s     414.910ms           120  
                                              _attn_fwd         0.00%       0.000us         0.00%       0.000us       0.000us       49.789s        56.87%       49.789s     414.910ms           120  
                                            gemm_kernel         0.00%       0.000us         0.00%       0.000us       0.000us       28.908s        33.02%       28.908s      14.840ms          1948  
                        onednn_addmm(32760, 1536, 1536)         0.18%     119.001ms         0.54%     366.234ms     508.659us        9.338s        10.67%        9.338s      12.969ms           720  
                        onednn_addmm(32760, 8960, 1536)         0.05%      36.593ms         0.12%      78.455ms     653.793us        9.152s        10.45%        9.152s      76.270ms           120  
                        onednn_addmm(32760, 1536, 8960)         0.02%      11.188ms         0.08%      53.044ms     442.034us        8.980s        10.26%        8.980s      74.833ms           120  
                                            aten::copy_        92.45%       62.290s       185.95%      125.282s      43.897ms        2.440s         2.79%        2.440s     854.937us          2854  
                                              aten::add         0.20%     136.638ms         0.94%     635.801ms     436.676us        1.508s         1.72%        1.508s       1.036ms          1456  
                                              aten::mul         0.06%      41.947ms         0.64%     428.219ms     378.286us        1.287s         1.47%        1.287s       1.137ms          1132  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 67.373s
Self XPU time total: 87.553s

# Comments:
# _attention/_attn_fwd is SLA Triton kernel w/o optimization on Intel GPU, no surprise on it.
# gemm_kernel may be triggered by low level library. XPU running time 28.908s looks like too long to kernel in production library.
# "Self CPU time total: 67.373s", most workload is expected offload to GPU, so CPU time is too high comparing with "Self XPU time total: 87.553s".

#
# sort_by="self_cpu_time_total"
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg      Self XPU    Self XPU %     XPU total  XPU time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                            aten::copy_        92.45%       62.290s       185.95%      125.282s      43.897ms        2.440s         2.79%        2.440s     854.937us          2854  
                                  urEnqueueKernelLaunch         5.42%        3.653s         5.42%        3.653s     323.086us     599.527ms         0.68%     599.527ms      53.027us         11306  
                                              aten::add         0.20%     136.638ms         0.94%     635.801ms     436.676us        1.508s         1.72%        1.508s       1.036ms          1456  
                        onednn_addmm(32760, 1536, 1536)         0.18%     119.001ms         0.54%     366.234ms     508.659us        9.338s        10.67%        9.338s      12.969ms           720  
                                              aten::pow         0.17%     111.623ms         0.34%     226.159ms       7.067ms      79.577us         0.00%     180.294us       5.634us            32  
                                     urEnqueueUSMMemcpy         0.16%     109.036ms         0.16%     109.036ms     434.408us       0.000us         0.00%       0.000us       0.000us           251  
                                       urUSMDeviceAlloc         0.16%     108.298ms         0.16%     108.298ms       4.923ms       0.000us         0.00%       0.000us       0.000us            22  
                                             aten::mean         0.15%     103.454ms         0.27%     184.286ms       1.536ms      77.725ms         0.09%      77.725ms     647.712us           120 

# Comments:
# aten::copy_  take the most of CPU time. It need call stack to investigate who trigger the copy and whether it is required.

#
# with_stack=True, sort_by="self_cpu_time_total"
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ---------------------------------------------------------------------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg      Self XPU    Self XPU %     XPU total  XPU time avg    # of Calls  Source Location                                                              
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ---------------------------------------------------------------------------  
                                            aten::copy_        92.34%       62.047s       184.69%      124.099s       10.342s      19.998us         0.00%      39.996us       3.333us            12  <built-in method to of Tensor object at 0x78a574c705f0>                      
                                                                                                                                                                                                     rcm/networks/wan2pt1.py(168): sinusoidal_embedding_1d                        
                                                                                                                                                                                                     rcm/networks/wan2pt1.py(645): forward                                        
                                                                                                                                                                                                     torch/nn/modules/module.py(1777): _call_impl                                 
                                                                                                                                                                                                     nn.Module: WanModel_0                                                        
                                                                                                                                                                                                                                                                                  
                                  urEnqueueKernelLaunch         0.24%     164.312ms         0.24%     164.312ms     342.317us      52.981ms         0.06%      52.981ms     110.376us           480  triton/backends/intel/driver.py(718): __call__                               
                                                                                                                                                                                                     triton/runtime/jit.py(531): run                                              
                                                                                                                                                                                                     triton/runtime/jit.py(374): <lambda>                                         
                                                                                                                                                                                                     FastNorm.py(90): rmsnorm                                                     
                                                                                                                                                                                                     FastNorm.py(348): forward
# Comments:
# Locate the point: "rcm/networks/wan2pt1.py(168): sinusoidal_embedding_1d"
# Let's review the code.
```

#### Fix High CPU Time

Move to sinusoidal_embedding_1d code. We could find:
1. It uses ```position.type(torch.float64)```, and Intel GPU not support float64 in recent generations. So, these ops should fall back to CPU.
2. ```dim``` is fixed in the model, so ```torch.pow(10000, -torch.arange(half).to(position).div(half))``` should be a fixed array which could be initialized while loading model.

```
# https://github.com/thu-ml/TurboDiffusion/blob/main/turbodiffusion/rcm/networks/wan2pt1.py#L144C1-L153C13
def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


# https://github.com/thu-ml/TurboDiffusion/blob/main/turbodiffusion/rcm/networks/wan2pt1.py#L672
with amp.autocast("cuda", dtype=torch.float32):
    e_B_D = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t_B).float())
    e0_B_6_D = self.time_projection(e_B_D).unflatten(1, (6, self.dim))
    assert e_B_D.dtype == torch.float32 and e0_B_6_D.dtype == torch.float32
```

In general, inference could run on 16-bit w/o obvious precision lose, 64-bit floating point may not be required. Let's skip type convert on B60, and move the fixed array computation to create_model function. Below is code updates snapshot.
```
#
#  Prepare the fixed array in create_model(https://github.com/aslanxie/TurboDiffusion/blob/main/turbodiffusion/inference/modify_model.py)
freq_dim = 256
assert freq_dim % 2 == 0
half = freq_dim // 2
exponents = -torch.arange(half) / half
div_term_cpu = torch.pow(torch.tensor(10000.0), exponents)
div_term_cpu = div_term_cpu.bfloat16()
net.register_buffer( "div_term", div_term_cpu)

# And, function sinusoidal_embedding_1d
def sinusoidal_embedding_1d(div_term, position):
    sinusoid = torch.outer(position, div_term)
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x

# 
emb = sinusoidal_embedding_1d(self.div_term, t_B)
e_B_D = self.time_embedding(emb)
```

From new round profile result, we could find ```aten::copy_ ``` issue is fixed and more workload is moved to GPU.
```
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg      Self XPU    Self XPU %     XPU total  XPU time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                       urUSMDeviceAlloc        23.80%     807.846ms        23.80%     807.846ms      36.720ms       0.000us         0.00%       0.000us       0.000us            22  
                                  urEnqueueKernelLaunch        17.04%     578.421ms        17.04%     578.421ms      51.333us     252.431ms         0.31%     252.431ms      22.402us         11268  
                                              aten::mul         5.16%     175.202ms         7.85%     266.445ms     235.376us        1.068s         1.33%        1.068s     943.485us          1132  
                        onednn_addmm(32760, 1536, 1536)         4.79%     162.508ms         6.00%     203.764ms     283.006us        9.356s        11.61%        9.356s      12.995ms           720  
                                              aten::pow         4.55%     154.314ms         9.19%     311.844ms      12.993ms     110.412us         0.00%     251.131us      10.464us            24  
                                     urEnqueueUSMMemcpy         4.40%     149.237ms         4.40%     149.237ms     604.200us       0.000us         0.00%       0.000us       0.000us           247  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 3.395s
Self XPU time total: 80.597s
```

#### Refine Model Description 
TurboDiffusion bases on rCM, training and inference share the same model description code which is mixing float32(```with amp.autocast("cuda", dtype=torch.float32)```) and bfloat16. It may be required for training and keeping the precise of gradient. For inference, 16-bit float should be OK.
So, let's force to bfloat16 on B60: ```with amp.autocast("xpu", dtype=torch.bfloat16)```. It could offload matrix operation to matrix engine.
We could observe ```gemm_kernel``` running time is reduced from 28.9s to 3.8s, x7.6 improvement from new profile result.
```
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg      Self XPU    Self XPU %     XPU total  XPU time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                             _attention         1.29%      32.611ms         1.74%      43.843ms     365.356us       44.285s        78.65%       44.285s     369.040ms           120  
                                              _attn_fwd         0.00%       0.000us         0.00%       0.000us       0.000us       44.285s        78.65%       44.285s     369.040ms           120  
                                            gemm_kernel         0.00%       0.000us         0.00%       0.000us       0.000us        3.807s         6.76%        3.807s       2.229ms          1708  
                                            aten::copy_         6.94%     174.922ms        13.22%     333.264ms      86.026us        3.228s         5.73%        3.228s     833.314us          3874  
                        onednn_addmm(32760, 1536, 1536)         4.29%     108.112ms         5.63%     141.968ms     197.177us        1.239s         2.20%        1.239s       1.721ms           720  
                        onednn_addmm(32760, 8960, 1536)         1.13%      28.479ms         1.33%      33.643ms     280.356us        1.239s         2.20%        1.239s      10.321ms           120  
at::native::xpu::UnrolledElementwiseKernel<at::nativ...         0.00%       0.000us         0.00%       0.000us       0.000us        1.138s         2.02%        1.138s     945.091us          1204  
                        onednn_addmm(32760, 1536, 8960)         0.94%      23.778ms         1.16%      29.362ms     244.686us        1.134s         2.01%        1.134s       9.451ms           120  
                                              aten::add         2.25%      56.638ms         4.45%     112.326ms      77.147us        1.054s         1.87%        1.054s     723.953us          1456  
                                              aten::mul         6.89%     173.831ms        10.46%     263.816ms     233.053us        1.009s         1.79%        1.009s     891.340us          1132  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 2.522s
Self XPU time total: 56.309s
```

#### Use Tensor Descriptors to load tl.dot arguments and save results

Intel XPU Backend for Triton provides several suggestion to improve Triton kernel source code performance. The No. 1 is "Use Tensor Descriptors". And Triton kernel _attn_fwd load data through pointer. So:
1. Migrate the code from pointers to Tensor Descriptors
2. Padding buffer and eliminate ```tl.where``` in kernel code
3. Remove unused operation for inference in kernel

And _attn_fwd running time is reduced from 49.789s to 8.129s.
```
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg      Self XPU    Self XPU %     XPU total  XPU time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                              _attn_fwd         0.00%       0.000us         0.00%       0.000us       0.000us        8.129s        39.31%        8.129s      67.738ms           120  
                                            gemm_kernel         0.00%       0.000us         0.00%       0.000us       0.000us        3.786s        18.31%        3.786s       2.217ms          1708  
                                            aten::copy_         7.17%     215.382ms        13.03%     391.737ms      87.558us        3.639s        17.60%        3.639s     813.302us          4474  
                        onednn_addmm(32760, 8960, 1536)         1.10%      33.064ms         2.57%      77.280ms     643.996us        1.238s         5.99%        1.238s      10.320ms           120  
                        onednn_addmm(32760, 1536, 1536)         3.91%     117.645ms         5.01%     150.450ms     208.958us        1.218s         5.89%        1.218s       1.691ms           720  
at::native::xpu::UnrolledElementwiseKernel<at::nativ...         0.00%       0.000us         0.00%       0.000us       0.000us        1.153s         5.58%        1.153s     958.006us          1204  
                        onednn_addmm(32760, 1536, 8960)         0.89%      26.719ms         1.06%      31.794ms     264.954us        1.134s         5.48%        1.134s       9.450ms           120  
                                              aten::add         2.08%      62.512ms         4.68%     140.773ms      96.685us        1.064s         5.15%        1.064s     731.031us          1456  
                                              aten::mul         6.62%     199.068ms        22.18%     666.613ms     588.881us        1.021s         4.94%        1.021s     902.233us          1132  
at::native::xpu::UnrolledElementwiseKernel<at::nativ...         0.00%       0.000us         0.00%       0.000us       0.000us        1.003s         4.85%        1.003s     833.141us          1204  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 3.006s
Self XPU time total: 20.680s
```







