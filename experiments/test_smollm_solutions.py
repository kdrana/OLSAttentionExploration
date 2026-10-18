"""
Modal experiment to test solutions for SmolLM2-1.7B UUID retrieval:
1. Baseline HALO (Shared, 22 layers, M=128) -> got 'V2R-892-CZ' (missed '8')
2. HALO with M=256
3. HALO with Layer-Selective Hooking (e.g. 6 or 8 layers, exactly like Qwen!)
4. HALO with M=256 on 8 layers
5. HALO with skipping early & terminal layers (Layers 6-20)
"""

import modal

app = modal.App("ols-attention-test-solutions")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "huggingface_hub", "triton", "numpy", "matplotlib", "requests")
    .add_local_python_source("src", "experiments")
)

@app.function(image=image, gpu="A10G", timeout=300)
def test_smollm_solutions():
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

    # Test 5 difficult depths where HALO-shared previously failed (0.11, 0.14, 0.43, 0.51, 0.66)
    test_cases = [
        (0.11, "A3K-456-WP2"),
        (0.14, "Q7B-789-RN5"),
        (0.43, "M4L-321-TK7"),
        (0.51, "V2R-892-CZ8"),
        (0.66, "Z8J-654-DY4"),
    ]

    configs = {
        "1_full_softmax": {"window_size": target_len + 100, "max_salient": 0, "layers": None, "mode": "original_softmax"},
        "2_halo_base_m128_all22": {"window_size": 512, "max_salient": 128, "layers": None, "mode": "halo"},
        "3_halo_m256_all22": {"window_size": 512, "max_salient": 256, "layers": None, "mode": "halo"},
        "4_halo_m128_8layers": {"window_size": 512, "max_salient": 128, "layers": [2, 5, 8, 11, 14, 17, 20, 23], "mode": "halo"},
        "5_halo_m256_8layers": {"window_size": 512, "max_salient": 256, "layers": [2, 5, 8, 11, 14, 17, 20, 23], "mode": "halo"},
        "6_halo_m128_skip_early_terminal": {"window_size": 512, "max_salient": 128, "layers": list(range(6, 21)), "mode": "halo"},
    }

    results = {cfg_name: [] for cfg_name in configs}

    for cfg_name, cfg in configs.items():
        print(f"\n=======================================================")
        print(f"Testing Config: {cfg_name}")
        print(f"=======================================================")
        apply_halo_adapter_generic(
            model,
            window_size=cfg["window_size"],
            max_salient=cfg["max_salient"],
            mode=cfg["mode"],
            layer_indices=cfg["layers"],
            device="cuda",
        )

        for depth, code in test_cases:
            n_str = f"\n[The target alpha-numeric registration identifier assigned to the primary node is strictly recorded in the vault as {code} for verification.]\n"
            q_str = f"\nQuestion: What is the target alpha-numeric registration identifier assigned to the primary node? Answer with only the identifier:"
            toks, _ = build_synthetic_context(tokenizer, target_len, n_str, q_str, depth=depth, is_instruct=True)
            inp = torch.tensor([toks], device="cuda")
            gen = run_generation(model, tokenizer, "cuda", inp, max_new_tokens=16)
            matched = (code in gen)
            results[cfg_name].append({
                "depth": depth,
                "code": code,
                "gen": gen,
                "matched": matched,
            })
            print(f"  [Depth {depth:.2f}] Code: '{code}' | Gen: '{gen}' | Matched: {matched}")

    summary = {}
    for cfg_name, res_list in results.items():
        acc = sum(1 for r in res_list if r["matched"]) / len(res_list) * 100.0
        summary[cfg_name] = acc
        print(f"\nSummary {cfg_name}: {acc:.1f}% ({sum(1 for r in res_list if r['matched'])}/{len(res_list)})")

    return {"summary": summary, "details": results}

@app.local_entrypoint()
def main():
    print("Launching test of SmolLM2 UUID solutions on Modal...")
    res = test_smollm_solutions.remote()
    print("\nFINAL COMPARISON ON DIFFICULT DEPTHS:")
    for k, v in res["summary"].items():
        print(f"  {k:<35}: {v:.1f}%")
