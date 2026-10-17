"""
Diagnostic: Inspect salience weights omega and retrieval logits directly.
"""

import modal

app = modal.App("ols-inspect-salience")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "huggingface_hub", "triton", "numpy", "matplotlib")
    .add_local_python_source("src", "experiments")
)


@app.function(image=image, gpu="A10G", timeout=300)
def inspect_salience():
    import math
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.cognitive_dual_control_attention import apply_cdca_adapter, reset_cdca_states
    from experiments.run_qwen_full_validation import HAYSTACK_BASE, NEEDLE, RETRIEVAL_QUERY

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Qwen/Qwen3.5-0.8B-Base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)

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

    # Hook Layer 15 self_attn
    layer15 = model.model.layers[15].self_attn
    head_dim = layer15.head_dim
    hidden_shape = (*input_ids.shape, -1, head_dim)
    extracted = {}

    def hook_fn(module, args, kwargs, output):
        hs = kwargs.get("hidden_states", args[0] if len(args) > 0 else None)
        extracted["hs"] = hs

    h = layer15.register_forward_hook(hook_fn, with_kwargs=True)
    with torch.no_grad():
        _ = model(input_ids)
    h.remove()

    hs = extracted["hs"]
    q_proj, _ = torch.chunk(layer15.q_proj(hs).view(*input_ids.shape, -1, head_dim * 2), 2, dim=-1)
    raw_q = layer15.q_norm(q_proj.view(hidden_shape)).transpose(1, 2)
    raw_k = layer15.k_norm(layer15.k_proj(hs).view(hidden_shape)).transpose(1, 2)
    v = layer15.v_proj(hs).view(hidden_shape).transpose(1, 2)

    # Compute novelty on raw_k
    k_unit = F.normalize(raw_k.float(), p=2, dim=-1)
    cos_sim = torch.sum(k_unit[:, :, 1:] * k_unit[:, :, :-1], dim=-1, keepdim=True)
    cos_sim = torch.cat([torch.ones_like(cos_sim[:, :, :1]), cos_sim], dim=2)
    novelty = 1.0 - cos_sim

    mean_nov = novelty.mean(dim=2, keepdim=True)
    std_nov = novelty.std(dim=2, keepdim=True) + 1e-6
    z_nov = (novelty - mean_nov) / std_nov

    print(f"Total T={T}")
    print(f"Needle span: [{needle_start}:{needle_end}] ({len(needle_tokens)} tokens)")
    print(f"Haystack mean novelty: {mean_nov.mean().item():.4f}, std: {std_nov.mean().item():.4f}")

    needle_nov = novelty[:, :, needle_start:needle_end, :].mean().item()
    needle_z = z_nov[:, :, needle_start:needle_end, :].mean().item()
    print(f"Needle mean novelty: {needle_nov:.4f}, mean z-score: {needle_z:.4f}")

    # Top 20 tokens with highest novelty across the entire sequence
    topk_vals, topk_inds = torch.topk(novelty[0, 0, :, 0], k=20)
    print("\nTop 20 Most Novel Tokens across 2048 sequence:")
    for rank, (val, idx) in enumerate(zip(topk_vals.tolist(), topk_inds.tolist())):
        tok_id = final_tokens[idx]
        tok_str = tokenizer.decode([tok_id])
        is_needle = (needle_start <= idx < needle_end)
        print(f"  #{rank+1:2d} | Pos {idx:4d} | Novelty: {val:.4f} | Tok: '{tok_str}' | In Needle: {is_needle}")

    return {"needle_nov": needle_nov, "needle_z": needle_z}
