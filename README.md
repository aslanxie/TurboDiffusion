# Optimized TurboDiffusion Inference on Intel GPU

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




