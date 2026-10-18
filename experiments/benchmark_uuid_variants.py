"""
Evaluate SmolLM2-1.7B UUID retrieval across all n=30 trials with:
1. HALO (M=128, all 22 layers) - Baseline (56.7%)
2. HALO (M=256, all 22 layers)
3. HALO (M=128, 8 layers)
4. HALO (M=256, 8 layers)
"""

import modal

app = modal.App("ols-attention-smollm-uuid-scaling")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "huggingface_hub", "triton", "numpy", "matplotlib", "requests")
    .add_local_python_source("src", "experiments")
)

@app.function(image=image, gpu="A10G", timeout=300)
def benchmark_uuid_variants():
    import random
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.cognitive_dual_control_attention import apply_halo_adapter_generic
    from experiments.run_comprehensive_review_suite import build_synthetic_context, run_generation, compute_stats

    hf_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    print(f"Loading {hf_id} on CUDA...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16).cuda()
    model.eval()

    target_len = 4096
    window_size = 512

    # Deterministic n=30 UUID trials identical to EXP-28 and EXP-30
    rng_uuid = random.Random(1337)
    uuid_depths = [round(float(x), 4) for x in np.linspace(0.08, 0.92, 30)]
    uuid_trials = []
    for d in uuid_depths:
        code = f"{rng_uuid.choice(['X9F','A3K','Q7B','M4L','V2R','Z8J'])}-{rng_uuid.randint(100, 999)}-{rng_uuid.choice(['CZ8','WP2','RN5','TK7','DY4'])}"
        n_str = f"\n[The target alpha-numeric registration identifier assigned to the primary node is strictly recorded in the vault as {code} for verification.]\n"
        q_str = f"\nQuestion: What is the target alpha-numeric registration identifier assigned to the primary node? Answer with only the identifier:"
        uuid_trials.append((d, n_str, q_str, code))

    configs = {
        "halo_m128_all22": {"window_size": 512, "max_salient": 128, "layers": None},
        "halo_m256_all22": {"window_size": 512, "max_salient": 256, "layers": None},
        "halo_m128_8layers": {"window_size": 512, "max_salient": 128, "layers": [2, 5, 8, 11, 14, 17, 20, 23]},
        "halo_m256_8layers": {"window_size": 512, "max_salient": 256, "layers": [2, 5, 8, 11, 14, 17, 20, 23]},
    }

    results = {}

    for cfg_name, cfg in configs.items():
        print(f"\n=======================================================")
        print(f"Benchmarking Variant: {cfg_name.upper()} on UUID (n=30)...")
        print(f"=======================================================")
        apply_halo_adapter_generic(
            model,
            window_size=cfg["window_size"],
            max_salient=cfg["max_salient"],
            mode="halo",
            layer_indices=cfg["layers"],
            device="cuda",
        )

        scores = []
        for idx, (d, n_str, q_str, code) in enumerate(uuid_trials):
            toks, _ = build_synthetic_context(tokenizer, target_len, n_str, q_str, depth=d, is_instruct=True)
            inp = torch.tensor([toks], device="cuda")
            gen = run_generation(model, tokenizer, "cuda", inp, max_new_tokens=16)
            matched = (code in gen)
            scores.append(100.0 if matched else 0.0)
            if idx % 5 == 0 or idx == len(uuid_trials) - 1:
                print(f"  [{idx+1:02d}/30] Depth {d:.2f} | Code: '{code}' | Gen: '{gen}' | Matched: {matched}")

        stats = compute_stats(scores)
        results[cfg_name] = {"scores": scores, "stats": stats}
        print(f"  ==> {cfg_name}: Mean = {stats['mean']}% (±{stats['ci95']}%, n=30)")

    return results

@app.local_entrypoint()
def main():
    print("Launching UUID scaling benchmark on Modal...")
    res = benchmark_uuid_variants.remote()
    print("\n" + "=" * 65)
    print("FULL N=30 UUID RETRIEVAL RESULTS:")
    print("=" * 65)
    for k, v in res.items():
        print(f"  {k:<22}: {v['stats']['mean']}% (±{v['stats']['ci95']}%)")
    print("=" * 65)
