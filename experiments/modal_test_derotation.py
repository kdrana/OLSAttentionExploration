"""
EXP-21 Test Runner: Calibrate De-Rotated Pre-Cache Translation Layer & Test Depth 10% NIAH.
"""

import modal

app = modal.App("ols-test-derotation-calibrated")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "huggingface_hub", "triton", "numpy", "matplotlib", "requests")
    .add_local_python_source("src", "experiments")
)


@app.function(image=image, gpu="A10G", timeout=300)
def test_derotation_calibrated_worker():
    import time
    import math
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    from src.cognitive_dual_control_attention import apply_cdca_adapter, reset_cdca_states
    from experiments.run_qwen_full_validation import run_niah_test, FULL_ATTN_LAYERS
    from experiments.run_direct_operator_estimation import DOMAIN_PROMPTS, DirectOperatorFitter

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Worker] Testing on {device} ({torch.cuda.get_device_name(0)})")

    model_id = "Qwen/Qwen3.5-0.8B-Base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)

    # 1. Fit De-Rotated Direct Operator on all 6 full attention layers
    print("\n[Phase 1] Calibrating De-Rotated Operator Projections (Pre-RoPE)...")
    t0 = time.perf_counter()
    all_adapter_state = {}
    rank = 16
    steps_per_layer = 150

    for layer_idx in FULL_ATTN_LAYERS:
        layer = model.model.layers[layer_idx]
        orig_attn = layer.self_attn
        head_dim = orig_attn.head_dim
        num_heads = orig_attn.q_proj.weight.shape[0] // (head_dim * 2)  # 8
        num_kv_heads = orig_attn.k_proj.weight.shape[0] // head_dim     # 2
        num_groups = num_heads // num_kv_heads                           # 4

        extracted = {"raw_q": [], "raw_k": [], "v": [], "y_true": []}
        orig_forward = orig_attn.forward

        def intercepted_forward(*args, **kwargs):
            hidden_states = kwargs.get("hidden_states", args[0] if len(args) > 0 else None)
            pos_emb = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, head_dim)

            query_states, gate = torch.chunk(
                orig_attn.q_proj(hidden_states).view(*input_shape, -1, head_dim * 2), 2, dim=-1
            )
            raw_q = orig_attn.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
            raw_k = orig_attn.k_norm(orig_attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = orig_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = pos_emb
            rope_q, rope_k = apply_rotary_pos_emb(raw_q, raw_k, cos, sin)

            key_expanded = rope_k.repeat_interleave(num_groups, dim=1)
            val_expanded = value_states.repeat_interleave(num_groups, dim=1)

            scale = 1.0 / math.sqrt(head_dim)
            scores = torch.matmul(rope_q, key_expanded.transpose(-1, -2)) * scale
            T = rope_q.shape[2]
            causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=hidden_states.device), diagonal=1)
            scores = scores + causal_mask.view(1, 1, T, T)
            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(rope_q.dtype)
            y_true = torch.matmul(attn_weights, val_expanded)

            extracted["raw_q"].append(raw_q.detach().cpu())
            extracted["raw_k"].append(raw_k.detach().cpu())
            extracted["v"].append(value_states.detach().cpu())
            extracted["y_true"].append(y_true.detach().cpu())

            return orig_forward(*args, **kwargs)

        orig_attn.forward = intercepted_forward
        model.eval()

        with torch.no_grad():
            for domain_name, texts in DOMAIN_PROMPTS.items():
                for text in texts:
                    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
                    _ = model(inp.input_ids)

        orig_attn.forward = orig_forward

        # Fit operator on raw (un-rotated) keys and queries
        operator = DirectOperatorFitter(head_dim=256, num_heads=8, num_kv_heads=2, rank=rank).to(device)
        optimizer = torch.optim.AdamW(operator.parameters(), lr=2e-3, weight_decay=1e-4)

        n_seqs = len(extracted["raw_q"])
        indices = list(range(n_seqs))

        for step in range(steps_per_layer):
            optimizer.zero_grad()
            b_sample = np.random.choice(indices, size=2, replace=False)
            for idx in b_sample:
                qs = extracted["raw_q"][idx].to(device)
                ks = extracted["raw_k"][idx].to(device)
                vs = extracted["v"][idx].to(device)
                yt = extracted["y_true"][idx].to(device)

                y_hat = operator(qs, ks, vs, gamma=0.99, lambda_reg=0.05)
                mse = F.mse_loss(y_hat, yt)
                cos_sim = F.cosine_similarity(y_hat.float(), yt.float(), dim=-1).mean()
                loss = (mse + 0.1 * (1.0 - cos_sim)) / len(b_sample)
                loss.backward()

            torch.nn.utils.clip_grad_norm_(operator.parameters(), max_norm=1.0)
            optimizer.step()

        all_adapter_state[f"layer_{layer_idx}.proj_k"] = operator.proj_k.data.cpu()
        all_adapter_state[f"layer_{layer_idx}.proj_q"] = operator.proj_q.data.cpu()

        # Compute validation R^2
        with torch.no_grad():
            r2_scores = []
            for vi in range(n_seqs - 4, n_seqs):
                qs = extracted["raw_q"][vi].to(device)
                ks = extracted["raw_k"][vi].to(device)
                vs = extracted["v"][vi].to(device)
                yt = extracted["y_true"][vi].to(device)
                y_hat = operator(qs, ks, vs)
                var = torch.var(yt)
                r2 = 1.0 - (F.mse_loss(y_hat, yt) / (var + 1e-8))
                r2_scores.append(r2.item())
        print(f"  Layer {layer_idx:2d} | De-Rotated Operator R^2: {np.mean(r2_scores):.4f}")

    print(f"Calibration Complete in {time.perf_counter() - t0:.2f}s")

    # 2. Hook CDCA Adapter with calibrated state dict
    print("\n[Phase 2] Hooking CDCA Adapter with calibrated De-Rotated Projections...")
    apply_cdca_adapter(model, window_size=1024, rank=16, use_derotation=True, adapter_source=all_adapter_state, device=device)

    # 3. Test NIAH T=2048, depth=0.10 (Where EXP-20 failed!)
    print("\n[Phase 3] Testing NIAH T=2048, Depth=10%...")
    reset_cdca_states(model)
    res2k = run_niah_test(model, tokenizer, device, target_len=2048, depth=0.10, is_linearized=False)
    print(f"  T=2048 D=10% | Success: {res2k['success']} | Output: '{res2k['gen_text']}'")

    # 4. Test NIAH T=4096, depth=0.10 (Where EXP-20 failed!)
    print("\n[Phase 4] Testing NIAH T=4096, Depth=10%...")
    reset_cdca_states(model)
    res4k = run_niah_test(model, tokenizer, device, target_len=4096, depth=0.10, is_linearized=False)
    print(f"  T=4096 D=10% | Success: {res4k['success']} | Output: '{res4k['gen_text']}'")

    return {"res2k": res2k, "res4k": res4k}
