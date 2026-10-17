"""
Step-by-step token generation comparison between Softmax and CDCA.
"""

import modal

app = modal.App("ols-step-generation-comparison")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "huggingface_hub", "triton", "numpy")
    .add_local_python_source("src", "experiments")
)


@app.function(image=image, gpu="A10G", timeout=300)
def compare_step_generation():
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

    print("Generating with Original Softmax Model:")
    with torch.no_grad():
        out_orig = model.generate(input_ids, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen_orig = tokenizer.decode(out_orig[0, input_ids.shape[1]:], skip_special_tokens=True)
    print(f"  Softmax Generated: '{gen_orig}'")
    for i in range(input_ids.shape[1], out_orig.shape[1]):
        tid = out_orig[0, i].item()
        print(f"    Step {i - input_ids.shape[1] + 1}: token '{tokenizer.decode([tid])}' (id={tid})")

    # Now hook CDCA
    apply_cdca_adapter(model, window_size=1024, rank=16, use_derotation=True, device=device)
    reset_cdca_states(model)

    print("\nGenerating with CDCA Model:")
    with torch.no_grad():
        out_cdca = model.generate(input_ids, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen_cdca = tokenizer.decode(out_cdca[0, input_ids.shape[1]:], skip_special_tokens=True)
    print(f"  CDCA Generated: '{gen_cdca}'")
    for i in range(input_ids.shape[1], out_cdca.shape[1]):
        tid = out_cdca[0, i].item()
        print(f"    Step {i - input_ids.shape[1] + 1}: token '{tokenizer.decode([tid])}' (id={tid})")

    return {}
