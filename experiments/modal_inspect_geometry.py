"""
Investigate the exact attention scores and geometry of Needle and Query in Qwen3.5.
Checks:
1. What are the dot products q_raw.T @ k_raw (pre-RoPE) vs q_rope.T @ k_rope (post-RoPE)?
2. Which heads in the 6 full attention layers attend to the needle?
3. How does RoPE affect the attention distribution across context length?
"""

import modal

app = modal.App("ols-inspect-attention-geometry")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "huggingface_hub", "triton", "numpy", "matplotlib")
    .add_local_python_source("src", "experiments")
)


@app.function(image=image, gpu="A10G", timeout=300)
def inspect_attention_geometry():
    import math
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    from experiments.run_qwen_full_validation import HAYSTACK_BASE, NEEDLE, RETRIEVAL_QUERY, FULL_ATTN_LAYERS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Qwen/Qwen3.5-0.8B-Base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)

    # Construct sequence with T=2048, depth=0.10
    target_len = 2048
    depth = 0.10
    needle_tokens = tokenizer.encode(NEEDLE, add_special_tokens=False)
    query_tokens = tokenizer.encode(RETRIEVAL_QUERY, add_special_tokens=False)
    haystack_tokens = tokenizer.encode(HAYSTACK_BASE, add_special_tokens=False)

    num_haystack_needed = target_len - len(needle_tokens) - len(query_tokens)
    repeats = int(math.ceil(num_haystack_needed / len(haystack_tokens)))
    full_haystack = (haystack_tokens * repeats)[:num_haystack_needed]

    insert_pos = int(len(full_haystack) * depth)
    final_tokens = full_haystack[:insert_pos] + needle_tokens + full_haystack[insert_pos:] + query_tokens
    input_ids = torch.tensor([final_tokens], device=device)
    T = input_ids.shape[1]

    needle_start = insert_pos
    needle_end = insert_pos + len(needle_tokens)
    query_pos = T - 1

    print(f"Sequence length T={T}, Needle at [{needle_start}:{needle_end}], Query at {query_pos}")
    print(f"Needle text: {tokenizer.decode(final_tokens[needle_start:needle_end])}")
    print(f"Query token text: '{tokenizer.decode([final_tokens[query_pos]])}'")

    # Inspect each of the 6 full-attention layers
    results = {}
    with torch.no_grad():
        # Run forward pass to extract raw and rotated keys/queries
        for layer_idx in FULL_ATTN_LAYERS:
            layer = model.model.layers[layer_idx]
            orig_attn = layer.self_attn
            head_dim = orig_attn.head_dim
            num_heads = orig_attn.q_proj.weight.shape[0] // (head_dim * 2)  # 8
            num_kv_heads = orig_attn.k_proj.weight.shape[0] // head_dim     # 2
            num_groups = num_heads // num_kv_heads                           # 4

            # Hook to capture hidden_states
            extracted = {}
            def hook_fn(module, args, kwargs, output):
                hs = kwargs.get("hidden_states", args[0] if len(args) > 0 else None)
                pos_emb = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
                extracted["hs"] = hs
                extracted["pos_emb"] = pos_emb

            h = orig_attn.register_forward_hook(hook_fn, with_kwargs=True)
            _ = model(input_ids)
            h.remove()

            hs = extracted["hs"]
            pos_emb = extracted["pos_emb"]
            input_shape = hs.shape[:-1]
            hidden_shape = (*input_shape, -1, head_dim)

            # Raw projections
            q_proj, _ = torch.chunk(orig_attn.q_proj(hs).view(*input_shape, -1, head_dim * 2), 2, dim=-1)
            raw_q = orig_attn.q_norm(q_proj.view(hidden_shape)).transpose(1, 2)  # (1, 8, T, 256)
            raw_k = orig_attn.k_norm(orig_attn.k_proj(hs).view(hidden_shape)).transpose(1, 2)  # (1, 2, T, 256)
            cos, sin = pos_emb
            rope_q, rope_k = apply_rotary_pos_emb(raw_q, raw_k, cos, sin)

            # Compare dot products at query position T-1 against needle tokens vs distractor tokens
            raw_k_exp = raw_k.repeat_interleave(num_groups, dim=1)
            rope_k_exp = rope_k.repeat_interleave(num_groups, dim=1)

            scale = 1.0 / math.sqrt(head_dim)
            raw_scores = torch.matmul(raw_q[:, :, query_pos:query_pos+1, :], raw_k_exp.transpose(-1, -2)) * scale
            rope_scores = torch.matmul(rope_q[:, :, query_pos:query_pos+1, :], rope_k_exp.transpose(-1, -2)) * scale

            # Softmax attention weights
            rope_attn = F.softmax(rope_scores, dim=-1)[0, :, 0, :]  # (8, T)
            raw_attn = F.softmax(raw_scores, dim=-1)[0, :, 0, :]    # (8, T)

            needle_rope_mass = rope_attn[:, needle_start:needle_end].sum(dim=-1).tolist()
            needle_raw_mass = raw_attn[:, needle_start:needle_end].sum(dim=-1).tolist()

            max_rope_pos = rope_attn.argmax(dim=-1).tolist()
            max_raw_pos = raw_attn.argmax(dim=-1).tolist()

            print(f"\n--- Layer {layer_idx} ---")
            for head in range(num_heads):
                print(f"  Head {head}: RoPE Needle Attn Mass = {needle_rope_mass[head]*100:.2f}% (Peak at {max_rope_pos[head]}) | Raw (No RoPE) Needle Attn Mass = {needle_raw_mass[head]*100:.2f}% (Peak at {max_raw_pos[head]})")

            results[f"layer_{layer_idx}"] = {
                "rope_needle_mass": needle_rope_mass,
                "raw_needle_mass": needle_raw_mass,
            }

    return results
