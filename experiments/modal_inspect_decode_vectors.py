"""
Inspect y_salient vs y_local during NIAH decode.
"""

import modal

app = modal.App("ols-inspect-decode-vectors")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "huggingface_hub", "triton", "numpy")
    .add_local_python_source("src", "experiments")
)


@app.function(image=image, gpu="A10G", timeout=300)
def inspect_decode():
    import math
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.cognitive_dual_control_attention import apply_cdca_adapter, reset_cdca_states
    from experiments.run_qwen_full_validation import HAYSTACK_BASE, NEEDLE, RETRIEVAL_QUERY, FULL_ATTN_LAYERS

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

    # First, let's see what the original UNMODIFIED model predicts at token T!
    with torch.no_grad():
        orig_logits = model(input_ids).logits[0, -1, :]
        top5_toks = torch.topk(orig_logits, 5).indices.tolist()
        print("Original Softmax Model Top-5 Predictions at pos T:")
        for rank, tid in enumerate(top5_toks):
            print(f"  #{rank+1}: '{tokenizer.decode([tid])}' (id={tid}, logit={orig_logits[tid].item():.2f})")

    # Now hook CDCA
    apply_cdca_adapter(model, window_size=1024, rank=16, use_derotation=True, device=device)
    reset_cdca_states(model)

    # Run forward pass (prefill)
    with torch.no_grad():
        cdca_logits = model(input_ids).logits[0, -1, :]
        top5_cdca = torch.topk(cdca_logits, 5).indices.tolist()
        print("\nCDCA Model Top-5 Predictions at pos T (after prefill):")
        for rank, tid in enumerate(top5_cdca):
            print(f"  #{rank+1}: '{tokenizer.decode([tid])}' (id={tid}, logit={cdca_logits[tid].item():.2f})")

    # Let's inspect Layer 15 cdca_attn module
    cdca_15 = model.model.layers[15].cdca_attn
    print("\nLayer 15 CDCA State:")
    print(f"  P_state norm: {torch.norm(cdca_15.P_state).item():.4f}, shape: {cdca_15.P_state.shape}")
    print(f"  R_state norm: {torch.norm(cdca_15.R_state).item():.4f}, shape: {cdca_15.R_state.shape}")
    print(f"  local_k_cache shape: {cdca_15.local_k_cache.shape}")

    return {}
