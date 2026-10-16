"""
Benchmark: Needle-In-A-Haystack (NIAH) Retrieval for Original Qwen3.5-0.8B (Softmax).

Tests the un-modified pre-trained Qwen3.5-0.8B-Base with full KV-cache across:
- Sequence lengths: T in [1024, 2048, 4096, 8192, 16384, 32768]
- Depths: 10%, 50%, 90%
Measures retrieval accuracy, peak VRAM, and generation latency.
"""

import math
import time
import os
import json
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from experiments.run_qwen_full_validation import (
    run_niah_test,
    HAYSTACK_BASE,
    NEEDLE,
    RETRIEVAL_QUERY,
)


def run_original_qwen_niah_benchmark(device: str = "cuda") -> dict:
    print("==================================================================")
    print(" Running NIAH Benchmark on Original Pre-Trained Qwen3.5-0.8B")
    print(f" Target Device: {device}")
    print("==================================================================")

    model_id = "Qwen/Qwen3.5-0.8B-Base"
    print(f"[1/2] Loading {model_id} (Native Softmax, un-modified)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if device == "cuda" else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)

    context_lengths = [1024, 2048, 4096, 8192, 16384, 32768]
    depths = [0.10, 0.50, 0.90]

    results = []
    print(f"\n[2/2] Stress-testing context window from T=1024 to T=32768...")

    for T_len in context_lengths:
        for d in depths:
            print(f"  Testing T={T_len:5d} | Depth={int(d*100):2d}%...")
            res = run_niah_test(model, tokenizer, device, T_len, d, is_linearized=False)
            results.append(res)
            print(f"    [Softmax] Success: {res['success']} | VRAM: {res['vram_mb']:7.1f}MB | KV Cache: {res['attn_state_kb']:8.1f}KB | Time: {res['gen_time_s']:.2f}s | Out: \"{res['gen_text'][:35]}\"")

    total_success = sum(1 for r in results if r["success"])
    success_rate = (total_success / len(results)) * 100.0
    print(f"\n==================================================================")
    print(f" Original Qwen3.5-0.8B NIAH Success Rate: {total_success}/{len(results)} ({success_rate:.1f}%)")
    print(f"==================================================================")

    return {
        "model_id": model_id,
        "total_trials": len(results),
        "total_success": total_success,
        "success_rate_pct": success_rate,
        "trials": results,
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_original_qwen_niah_benchmark(device=device)
