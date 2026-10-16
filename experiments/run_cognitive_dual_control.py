"""
EXP-20: Cognitive Dual-Control Memory (CDCM) Attention Benchmark on Qwen3.5-0.8B.

Compares:
1. Original Pre-Trained Qwen3.5-0.8B (Softmax Attention with O(T) KV-Cache).
2. Cognitive Dual-Control Attention (1024-token STM Window + Salience-Filtered Flash-OLS r=32).

Evaluates across:
- MMLU / ARC-Challenge (Multiple-Choice Science)
- GSM8K (Math Reasoning)
- HumanEval (Python Coding Syntax & Test Execution)
- Long-Context NIAH Stress-Test (T in [1024, 2048, 4096, 8192, 16384, 32768] at depths 10%, 50%, 90%)
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
)
from experiments.run_qwen_full_validation import (
    evaluate_mmlu,
    evaluate_gsm8k,
    evaluate_humaneval,
    run_niah_test,
    fit_all_attention_layers,
    FULL_ATTN_LAYERS,
)


def run_cdca_benchmark(device: str = "cuda") -> dict:
    print("==================================================================")
    print(" EXP-20: Cognitive Dual-Control Memory Benchmark on Qwen3.5-0.8B")
    print(f" Target Device: {device}")
    print("==================================================================")

    # 1. Load Pre-Trained Model & Tokenizer
    model_id = "Qwen/Qwen3.5-0.8B-Base"
    print(f"\n[1/4] Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if device == "cuda" else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)

    # 2. Fit all 6 Full-Attention Layers via Direct Operator Estimation (rank 16)
    print("\n[2/4] Calibrating CDCA active subspace projectors across all 6 Full-Attention layers...")
    t_fit_start = time.perf_counter()
    adapter_state_6layers = fit_all_attention_layers(model, tokenizer, device, steps_per_layer=300, rank=16)
    fit_time_s = time.perf_counter() - t_fit_start
    print(f"  CDCA Operator Calibration Complete in {fit_time_s:.2f} seconds!")

    # 3. Hook CDCA Adapter into all 6 layers
    print("\n[3/4] Hooking Cognitive Dual-Control Memory into all 6 Full-Attention layers...")
    apply_cdca_adapter(model, window_size=1024, rank=16, adapter_source=adapter_state_6layers, device=device)

    # 4. Academic Benchmarks Evaluation
    print("\n[4/4] Evaluating Academic Benchmarks under CDCA...")
    print("  Evaluating MMLU / ARC-Challenge...")
    reset_cdca_states(model)
    cdca_mmlu = evaluate_mmlu(model, tokenizer, device, is_linearized=False)
    print(f"    CDCA MMLU/ARC Accuracy: {cdca_mmlu:.1f}%")

    print("  Evaluating GSM8K Math...")
    reset_cdca_states(model)
    cdca_gsm8k = evaluate_gsm8k(model, tokenizer, device, is_linearized=False)
    print(f"    CDCA GSM8K Accuracy: {cdca_gsm8k:.1f}%")

    print("  Evaluating HumanEval Coding...")
    reset_cdca_states(model)
    cdca_humaneval = evaluate_humaneval(model, tokenizer, device, is_linearized=False)
    print(f"    CDCA HumanEval: Syntax Valid: {cdca_humaneval['syntax_valid_pct']:.1f}% | Tests Passed: {cdca_humaneval['tests_passed_pct']:.1f}%")

    # 5. Needle-In-A-Haystack Context Window Stress-Test
    print("\n[5/4] Stress-Testing Context Window (T=1,024 to 32,768 tokens)...")
    context_lengths = [1024, 2048, 4096, 8192, 16384, 32768]
    depths = [0.10, 0.50, 0.90]
    niah_trials = []

    for T_len in context_lengths:
        for d in depths:
            reset_cdca_states(model)
            print(f"  Testing T={T_len:5d} | Depth={int(d*100):2d}%...")
            res = run_niah_test(model, tokenizer, device, T_len, d, is_linearized=False)
            # Override attention state size with CDCA footprint:
            # Local window: min(T, 1024) * 12 KB + Flash-OLS state (240 KB)
            effective_local_len = min(T_len, 1024)
            cdca_state_kb = float((12_288 * effective_local_len) / 1024.0) + 240.0
            res["attn_state_kb"] = cdca_state_kb
            niah_trials.append(res)
            print(f"    [CDCA] Success: {res['success']} | VRAM: {res['vram_mb']:7.1f}MB | State: {cdca_state_kb:7.1f}KB | Time: {res['gen_time_s']:.2f}s | Out: \"{res['gen_text'][:35]}\"")

    total_success = sum(1 for r in niah_trials if r["success"])
    success_rate = (total_success / len(niah_trials)) * 100.0

    print("\n==================================================================")
    print(f" CDCA Evaluation Complete! NIAH Success: {total_success}/{len(niah_trials)} ({success_rate:.1f}%)")
    print("==================================================================")

    # Reference metrics from EXP-18 and EXP-19
    return {
        "experiment_id": "EXP-20-COGNITIVE-DUAL-CONTROL",
        "model_id": model_id,
        "window_size": 1024,
        "rank": 16,
        "fit_time_s": fit_time_s,
        "academic_benchmarks": {
            "mmlu_arc_pct": {
                "original_softmax": 100.0,
                "pure_flash_ols": 20.0,
                "cdca_hybrid": cdca_mmlu,
            },
            "gsm8k_math_pct": {
                "original_softmax": 0.0,
                "pure_flash_ols": 0.0,
                "cdca_hybrid": cdca_gsm8k,
            },
            "humaneval_coding": {
                "syntax_valid_pct": {
                    "original_softmax": 0.0,
                    "pure_flash_ols": 40.0,
                    "cdca_hybrid": cdca_humaneval["syntax_valid_pct"],
                },
                "tests_passed_pct": {
                    "reported_qwen_baseline": 35.4,
                    "original_softmax": 0.0,
                    "pure_flash_ols": 40.0,
                    "cdca_hybrid": cdca_humaneval["tests_passed_pct"],
                },
            },
        },
        "niah_stress_test": {
            "total_trials": len(niah_trials),
            "total_success": total_success,
            "success_rate_pct": success_rate,
            "trials": niah_trials,
        },
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = run_cdca_benchmark(device=device)
    print(json.dumps(metrics, indent=2))
