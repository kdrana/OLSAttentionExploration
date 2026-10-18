"""
Diagnostic script for SmolLM2-1.7B UUID retrieval failure on Modal.
Traces layer-by-layer salience, needle token preservation, and decode output.
"""

import modal

app = modal.App("ols-attention-diagnose-smollm")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "huggingface_hub", "triton", "numpy", "matplotlib", "requests")
    .add_local_python_source("src", "experiments")
)

@app.function(image=image, gpu="A10G", timeout=300)
def diagnose_uuid_trial(depth: float = 0.51):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.cognitive_dual_control_attention import apply_halo_adapter_generic, reset_cdca_states
    from experiments.run_comprehensive_review_suite import build_synthetic_context, run_generation

    hf_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    print(f"Loading {hf_id} on CUDA...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16).cuda()
    model.eval()

    target_len = 4096
    window_size = 512
    max_salient = 128

    code = "V2R-892-CZ8"
    n_str = f"\n[The target alpha-numeric registration identifier assigned to the primary node is strictly recorded in the vault as {code} for verification.]\n"
    q_str = f"\nQuestion: What is the target alpha-numeric registration identifier assigned to the primary node? Answer with only the identifier:"

    toks, needle_insert_pos = build_synthetic_context(tokenizer, target_len, n_str, q_str, depth=depth, is_instruct=True)
    needle_tok_ids = tokenizer.encode(code, add_special_tokens=False)
    needle_tok_len = len(needle_tok_ids)
    needle_range = set(range(needle_insert_pos, needle_insert_pos + len(tokenizer.encode(n_str, add_special_tokens=False))))
    exact_code_range = set(range(needle_insert_pos + 18, needle_insert_pos + 18 + needle_tok_len))

    print(f"\n--- Diagnostic Info ---")
    print(f"Target code: '{code}' (Tokens: {tokenizer.convert_ids_to_tokens(needle_tok_ids)})")
    print(f"Context length: {len(toks)} tokens")
    print(f"Needle insert position: {needle_insert_pos} (depth {depth})")
    print(f"Total layers: {len(model.model.layers)}")

    # 1. Test Full Softmax baseline
    apply_halo_adapter_generic(model, window_size=target_len + 100, max_salient=0, mode="original_softmax", device="cuda")
    inp = torch.tensor([toks], device="cuda")
    gen_sm = run_generation(model, tokenizer, "cuda", inp, max_new_tokens=16)
    print(f"\n[1] FULL SOFTMAX output: '{gen_sm}' | Matched: {code in gen_sm}")

    # 2. Test HALO-Shared with trace recording
    apply_halo_adapter_generic(model, window_size=window_size, max_salient=max_salient, mode="halo", per_head=False, record_trace=True, device="cuda")
    reset_cdca_states(model)
    with torch.no_grad():
        out_ids = model.generate(inp, max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen_halo = tokenizer.decode(out_ids[0][inp.shape[1]:], skip_special_tokens=True)
    print(f"\n[2] HALO-SHARED output: '{gen_halo}' | Matched: {code in gen_halo}")

    # Check which layers preserved the needle tokens
    layer_retention = {}
    hist_len = len(toks) - window_size
    for idx, layer in enumerate(model.model.layers):
        if hasattr(layer, "cdca_attn"):
            mod = layer.cdca_attn
            if mod.last_salience_trace is not None:
                selected = set(mod.last_salience_trace["selected_indices"])
                needle_in_sel = len(needle_range.intersection(selected))
                exact_in_sel = len(exact_code_range.intersection(selected))
                layer_retention[idx] = {
                    "needle_tokens_kept": needle_in_sel,
                    "exact_code_tokens_kept": exact_in_sel,
                    "needle_total": len(needle_range),
                    "code_total": len(exact_code_range),
                }
                print(f"Layer {idx:02d}: kept {exact_in_sel}/{len(exact_code_range)} code tokens, {needle_in_sel}/{len(needle_range)} sentence tokens")

    return {
        "softmax_gen": gen_sm,
        "softmax_match": code in gen_sm,
        "halo_gen": gen_halo,
        "halo_match": code in gen_halo,
        "needle_insert_pos": needle_insert_pos,
        "hist_len": hist_len,
    }

@app.local_entrypoint()
def main():
    print("Launching diagnostic run on Modal...")
    res = diagnose_uuid_trial.remote(depth=0.51)
    print("Result:", res)
