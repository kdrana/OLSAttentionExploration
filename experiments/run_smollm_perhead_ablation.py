"""
EXP-30: SmolLM2-1.7B Per-Head Landmark Allocation (HALO-PerHead) Ablation Suite.
================================================================================
Empirical evaluation of Asynchronous Per-Head Landmark Allocation on SmolLM2-1.7B
(32-head Multi-Head Attention, 22 hooked layers, T=4096).

Compares:
1. original_softmax: Full Attention Reference (no eviction)
2. snapkv: Head-Shared Baseline (Li et al., 2024; K=2 box, no derotation, no surprise)
3. halo_shared: Head-Shared HALO (Exponential kernel K=16, PRISM, surprise; per_head=False)
4. halo_perhead: HALO-PerHead (Asynchronous Per-Head Allocation; per_head=True)

Across 3 canonical recall tasks:
- uuid: Multi-token alphanumeric identifier recall (n=30)
- sniah: Single Needle-in-a-Haystack passkey recall (n=30)
- retroactive: Displaced context override recall (n=30)
Total: n=90 composite trials per method.
"""

import os
import gc
import math
import random
import json
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.cognitive_dual_control_attention import apply_halo_adapter_generic, reset_cdca_states
from experiments.run_comprehensive_review_suite import (
    compute_stats,
    load_model_and_tokenizer,
    build_synthetic_context,
    run_generation,
)


def get_task_trials(task_name: str):
    """Generates deterministic evaluation trials identical to EXP-28 protocol."""
    if task_name == "sniah":
        rng = random.Random(42)
        depths = [round(float(x), 4) for x in np.linspace(0.08, 0.92, 30)]
        trials = []
        for d in depths:
            code = f"{rng.randint(10000, 99999)}"
            n_str = f"\n[CRITICAL NOTE: The secret operational access code for Project Antigravity is {code}.]\n"
            q_str = f"\nQuestion: What is the secret operational access code for Project Antigravity? Answer with only the 5-digit code:"
            trials.append((d, n_str, q_str, code, 10))
        return trials

    elif task_name == "uuid":
        rng = random.Random(1337)
        depths = [round(float(x), 4) for x in np.linspace(0.08, 0.92, 30)]
        trials = []
        for d in depths:
            code = f"{rng.choice(['X9F','A3K','Q7B','M4L','V2R','Z8J'])}-{rng.randint(100, 999)}-{rng.choice(['CZ8','WP2','RN5','TK7','DY4'])}"
            n_str = f"\n[The target alpha-numeric registration identifier assigned to the primary node is strictly recorded in the vault as {code} for verification.]\n"
            q_str = f"\nQuestion: What is the target alpha-numeric registration identifier assigned to the primary node? Answer with only the identifier:"
            trials.append((d, n_str, q_str, code, 16))
        return trials

    elif task_name == "retroactive":
        rng = random.Random(2026)
        depths = [round(float(x), 4) for x in np.linspace(0.08, 0.92, 30)]
        trials = []
        for d in depths:
            code = f"{rng.randint(10000, 99999)}"
            n_str = f"\n[The secret emergency override authorization code is {code}. Make sure you remember this confirmation note.]\n"
            q_str = f"\nQuestion: What is the secret emergency override authorization code? Answer with only the code:"
            trials.append((d, n_str, q_str, code, 10))
        return trials

    else:
        raise ValueError(f"Unknown task_name: {task_name}")


def run_smollm_task_eval(task_name: str, device: str = "cuda") -> Dict[str, Any]:
    """Runs a single task across the 4 methods on SmolLM2-1.7B."""
    print(f"\n=======================================================")
    print(f"RUNNING SMOLLM2-1.7B EVALUATION: TASK = {task_name.upper()} (n=30)")
    print(f"=======================================================")

    hf_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    model, tokenizer = load_model_and_tokenizer(hf_id, device=device)

    target_len = 4096
    window_size = 512
    max_salient = 128
    is_instruct = True

    trials = get_task_trials(task_name)
    methods = ["original_softmax", "snapkv", "halo_shared", "halo_perhead"]
    results = {}

    for method in methods:
        print(f"\n  --> Benchmarking Method: {method.upper()} on {task_name}...")
        if method == "original_softmax":
            apply_halo_adapter_generic(
                model, window_size=target_len + 100, max_salient=0, mode="original_softmax",
                layer_indices=None, device=device
            )
        elif method == "snapkv":
            apply_halo_adapter_generic(
                model, window_size=window_size, max_salient=max_salient, mode="snapkv",
                use_derotation=False, use_surprise=False, diffusion_radius=2, kernel_type="uniform_box",
                layer_indices=None, device=device, per_head=False
            )
        elif method == "halo_shared":
            apply_halo_adapter_generic(
                model, window_size=window_size, max_salient=max_salient, mode="halo",
                use_derotation=True, use_surprise=True, diffusion_radius=16, kernel_type="exponential",
                layer_indices=None, device=device, per_head=False
            )
        elif method == "halo_perhead":
            apply_halo_adapter_generic(
                model, window_size=window_size, max_salient=max_salient, mode="halo",
                use_derotation=True, use_surprise=True, diffusion_radius=16, kernel_type="exponential",
                layer_indices=None, device=device, per_head=True
            )

        scores = []
        for idx, (d, n_str, q_str, code, max_new_toks) in enumerate(trials):
            toks, _ = build_synthetic_context(tokenizer, target_len, n_str, q_str, d, is_instruct=is_instruct)
            inp = torch.tensor([toks], device=device)
            gen = run_generation(model, tokenizer, device, inp, max_new_tokens=max_new_toks)
            matched = (code in gen)
            scores.append(100.0 if matched else 0.0)
            if idx % 5 == 0 or idx == len(trials) - 1:
                print(f"      [Trial {idx+1:02d}/30] Depth {d:.2f} | GroundTruth='{code}' | Matched={matched}")

        stats = compute_stats(scores)
        results[method] = {
            "scores": scores,
            "stats": stats,
        }
        print(f"    ==> {method.upper()} on {task_name}: Mean = {stats['mean']}% (±{stats['ci95']}%, n={stats['n']})")

    del model
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    return results


def compile_experiment_results(all_task_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compiles task results into final master metrics dictionary with composite scores."""
    methods = ["original_softmax", "snapkv", "halo_shared", "halo_perhead"]
    compiled = {
        "model": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "architecture": "32-Head Multi-Head Attention (MHA)",
        "context_length": 4096,
        "window_size": 512,
        "max_salient": 128,
        "cache_compression_ratio": "5.33x",
        "tasks": all_task_results,
        "summary": {},
    }

    for method in methods:
        sniah_scores = all_task_results["sniah"][method]["scores"]
        uuid_scores = all_task_results["uuid"][method]["scores"]
        retro_scores = all_task_results["retroactive"][method]["scores"]
        composite_scores = sniah_scores + uuid_scores + retro_scores

        compiled["summary"][method] = {
            "composite": compute_stats(composite_scores),
            "sniah": compute_stats(sniah_scores),
            "uuid": compute_stats(uuid_scores),
            "retroactive": compute_stats(retro_scores),
        }

    return compiled
