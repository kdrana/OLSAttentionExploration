"""
EXP-21: De-Rotated Pre-Cache Translation Layer & Bidirectional Salience Diffusion (CDCA-V2).
Comprehensive benchmark on Qwen/Qwen3.5-0.8B-Base.

Evaluates:
1. Academic Benchmarks:
   - MMLU / ARC-Challenge (Multiple-Choice Science & Factual Reasoning)
   - GSM8K (Multi-Step Arithmetic Reasoning)
   - HumanEval (Python Algorithmic Coding Syntax & Unit Tests)
2. Long-Context Needle-In-A-Haystack (NIAH) Grid:
   - Context lengths: 1024, 2048, 4096, 8192, 16384, 32768
   - Depths: 10%, 50%, 90% (18 trials)
3. Retroactive Alert Salience Stress-Test:
   - T = 32,768 (Depth 10%) with retroactive framing:
     "[The access code is 74921. Make sure you remember this.]"
"""

import math
import time
import os
import gc
import json
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.cognitive_dual_control_attention import (
    apply_cdca_adapter,
    reset_cdca_states,
    set_salient_landmark_indices,
)
from experiments.run_qwen_full_validation import (
    evaluate_mmlu,
    evaluate_gsm8k,
    evaluate_humaneval,
    HAYSTACK_BASE,
    NEEDLE,
    RETRIEVAL_QUERY,
)


def run_cdca_v2_niah_trial(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    device: str,
    target_len: int,
    depth: float,
    custom_needle: str | None = None,
    custom_query: str | None = None,
) -> dict:
    needle_str = custom_needle if custom_needle else NEEDLE
    query_str = custom_query if custom_query else RETRIEVAL_QUERY

    needle_tokens = tokenizer.encode(needle_str, add_special_tokens=False)
    query_tokens = tokenizer.encode(query_str, add_special_tokens=False)
    haystack_tokens = tokenizer.encode(HAYSTACK_BASE, add_special_tokens=False)

    num_haystack_needed = target_len - len(needle_tokens) - len(query_tokens)
    if num_haystack_needed <= 0:
        num_haystack_needed = 100
    repeats = int(math.ceil(num_haystack_needed / len(haystack_tokens)))
    full_haystack = (haystack_tokens * repeats)[:num_haystack_needed]

    insert_pos = int(len(full_haystack) * depth)
    final_tokens = full_haystack[:insert_pos] + needle_tokens + full_haystack[insert_pos:] + query_tokens
    input_ids = torch.tensor([final_tokens], device=device)
    actual_len = input_ids.shape[1]

    reset_cdca_states(model)
    set_salient_landmark_indices(model, tokenizer, input_ids, device=device, max_salient=1024)

    gc.collect()
    if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
        torch.cuda.synchronize()
    gen_time_s = time.perf_counter() - t0

    gen_text = tokenizer.decode(out_ids[0][actual_len:], skip_special_tokens=True)
    peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)) if (device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda")) else 0.0
    success = ("74921" in gen_text)

    # State memory: Local window (min(T, 1024) * 12 KB) + Landmark cache (min(max(0, T - 1024), 1024) * 12 KB)
    local_tokens = min(actual_len, 1024)
    salient_tokens = min(max(0, actual_len - 1024), 1024)
    state_kb = float((local_tokens + salient_tokens) * 12.0)

    return {
        "context_length": actual_len,
        "target_length": target_len,
        "depth": depth,
        "success": success,
        "gen_time_s": gen_time_s,
        "peak_vram_mb": peak_vram_mb,
        "attn_state_kb": state_kb,
        "gen_text": gen_text,
    }


def run_exp21_benchmark(device: str = "cuda") -> dict:
    print("==================================================================")
    print(" EXP-21: De-Rotated Pre-Cache & Bidirectional Salience Diffusion")
    print(f" Target Device: {device}")
    print("==================================================================")

    # 1. Load Pre-Trained Model & Tokenizer
    model_id = "Qwen/Qwen3.5-0.8B-Base"
    print(f"\n[1/4] Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if device == "cuda" else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)

    # 2. Hook CDCA-V2 Adapter into all 6 layers
    print("\n[2/4] Hooking CDCA-V2 Adapter into all 6 Full-Attention layers...")
    apply_cdca_adapter(model, window_size=1024, max_salient=1024, device=device)

    # 3. Academic Benchmarks Evaluation
    print("\n[3/4] Evaluating Academic Benchmarks under CDCA-V2...")
    print("  Evaluating MMLU / ARC-Challenge...")
    reset_cdca_states(model)
    cdca_mmlu = evaluate_mmlu(model, tokenizer, device, is_linearized=False)
    print(f"    CDCA-V2 MMLU/ARC Accuracy: {cdca_mmlu:.1f}%")

    print("  Evaluating GSM8K Math Reasoning...")
    reset_cdca_states(model)
    cdca_gsm8k = evaluate_gsm8k(model, tokenizer, device, is_linearized=False)
    print(f"    CDCA-V2 GSM8K Accuracy: {cdca_gsm8k:.1f}%")

    print("  Evaluating HumanEval Coding...")
    reset_cdca_states(model)
    cdca_humaneval = evaluate_humaneval(model, tokenizer, device, is_linearized=False)
    print(f"    CDCA-V2 HumanEval: Syntax Valid: {cdca_humaneval['syntax_valid_pct']:.1f}% | Tests Passed: {cdca_humaneval['tests_passed_pct']:.1f}%")

    # 4. Long-Context Needle-In-A-Haystack Grid (18 trials)
    print("\n[4/4] Evaluating Full NIAH Grid (T=1,024 to 32,768 tokens, Depths 10%, 50%, 90%)...")
    context_lengths = [1024, 2048, 4096, 8192, 16384, 32768]
    depths = [0.10, 0.50, 0.90]
    niah_trials = []

    for T_len in context_lengths:
        for d in depths:
            res = run_cdca_v2_niah_trial(model, tokenizer, device, target_len=T_len, depth=d)
            niah_trials.append(res)
            print(f"  T={T_len:5d} | D={int(d*100):2d}% | Success: {res['success']} | VRAM: {res['peak_vram_mb']:7.1f}MB | State: {res['attn_state_kb']:7.1f}KB | Time: {res['gen_time_s']:.2f}s | Out: \"{res['gen_text'][:35]}\"")

    total_success = sum(1 for r in niah_trials if r["success"])
    success_rate = (total_success / len(niah_trials)) * 100.0
    print(f"\n>>> Total NIAH Success Rate: {total_success} / {len(niah_trials)} ({success_rate:.1f}%) <<<")

    # 5. Retroactive Alert Stress-Test at T=32,768
    print("\n[5/4] Stress-Testing Retroactive Confirmation Alert at T=32,768...")
    retro_needle = "\n[The access code is 74921. Make sure you remember this.]\n"
    retro_query = "\nQuestion: What is the access code? Answer: The access code is"
    res_retro = run_cdca_v2_niah_trial(
        model, tokenizer, device, target_len=32768, depth=0.10,
        custom_needle=retro_needle, custom_query=retro_query
    )
    print(f"  Retroactive Note T=32,768 D=10% | Success: {res_retro['success']} | Time: {res_retro['gen_time_s']:.2f}s | Out: \"{res_retro['gen_text'][:45]}\"")

    # Aggregate Benchmark Dictionary
    results = {
        "experiment_id": "EXP-21-TRANSLATION-LAYER",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_id,
        "device": device,
        "architecture": {
            "name": "Cognitive Dual-Control Memory Attention (CDCA-V2)",
            "stm_window_size": 1024,
            "max_salient_landmarks": 1024,
            "diffusion_kernel_radius": 16,
            "pre_cache_translation": "De-Rotated & Whitened",
            "full_attn_layers_modified": [3, 7, 11, 15, 19, 23],
        },
        "academic_benchmarks": {
            "mmlu_arc_pct": {
                "reported_baseline": 42.3,
                "original_softmax": 100.0,
                "pure_flash_ols": 20.0,
                "cdca_v1": 100.0,
                "cdca_v2": cdca_mmlu,
            },
            "gsm8k_math_pct": {
                "reported_baseline": 49.6,
                "original_softmax": 0.0,
                "pure_flash_ols": 0.0,
                "cdca_v1": 0.0,
                "cdca_v2": cdca_gsm8k,
            },
            "humaneval_coding": {
                "syntax_valid_pct": {
                    "original_softmax": 0.0,
                    "pure_flash_ols": 100.0,
                    "cdca_v1": 0.0,
                    "cdca_v2": cdca_humaneval["syntax_valid_pct"],
                },
                "tests_passed_pct": {
                    "reported_baseline": 35.4,
                    "original_softmax": 0.0,
                    "pure_flash_ols": 40.0,
                    "cdca_v1": 0.0,
                    "cdca_v2": cdca_humaneval["tests_passed_pct"],
                },
            },
        },
        "niah_stress_test": {
            "total_trials": len(niah_trials),
            "total_success": total_success,
            "success_rate_pct": success_rate,
            "trials": niah_trials,
            "retroactive_alert_trial": res_retro,
        },
    }

    return results
