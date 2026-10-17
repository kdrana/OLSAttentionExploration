"""
Inspect norm_z distribution in 4096 sequence.
"""

import modal

app = modal.App("ols-inspect-normz")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "huggingface_hub", "triton", "numpy")
    .add_local_python_source("src", "experiments")
)


@app.function(image=image, gpu="A10G", timeout=300)
def inspect_normz():
    import math
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from experiments.run_qwen_full_validation import HAYSTACK_BASE, NEEDLE, RETRIEVAL_QUERY

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Qwen/Qwen3.5-0.8B-Base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)

    target_len = 4096
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
    print(f"T={T}, needle at [{needle_start}:{needle_end}]")

    # Hook Layer 15
    layer15 = model.model.layers[15].self_attn
    head_dim = layer15.head_dim
    hidden_shape = (*input_ids.shape, -1, head_dim)
    extracted = {}

    def hook_fn(module, args, kwargs, output):
        extracted["hs"] = kwargs.get("hidden_states", args[0] if len(args) > 0 else None)

    h = layer15.register_forward_hook(hook_fn, with_kwargs=True)
    with torch.no_grad():
        _ = model(input_ids)
    h.remove()

    hs = extracted["hs"]
    raw_k = layer15.k_norm(layer15.k_proj(hs).view(hidden_shape)).transpose(1, 2)

    hist_len = T - 1024
    norms = torch.norm(raw_k[:, :, :hist_len].float(), dim=-1).mean(dim=1)[0]
    norm_z = (norms - norms.mean()) / (norms.std() + 1e-6)

    print(f"Needle norm_z max: {norm_z[needle_start:needle_end].max().item():.4f}, mean: {norm_z[needle_start:needle_end].mean().item():.4f}")
    print(f"Haystack norm_z max: {norm_z[:needle_start].max().item():.4f}, mean: {norm_z[:needle_start].mean().item():.4f}")

    # Find the rank of the needle tokens among all tokens
    needle_max_idx = needle_start + norm_z[needle_start:needle_end].argmax().item()
    needle_max_val = norm_z[needle_max_idx].item()
    rank = (norm_z > needle_max_val).sum().item() + 1
    print(f"Needle peak token '{tokenizer.decode([final_tokens[needle_max_idx]])}' is Rank #{rank} out of {hist_len} tokens!")

    return {}
