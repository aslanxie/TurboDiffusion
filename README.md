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



